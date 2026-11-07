"""CARE Chinese adaptation: original attention/layers, stable logits and balanced loss."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer

from care_data import ROOT, assets

source = ROOT / "author_source/code/model/care.py"
spec = importlib.util.spec_from_file_location("care_author", source)
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)

class LogitNER(author.ner_unit):
    def forward(self, hidden, shared, mask):
        length = hidden.shape[0]
        start = hidden.unsqueeze(1).expand(-1, length, -1, -1)
        end = hidden.unsqueeze(0).expand(length, -1, -1, -1)
        features = torch.cat((start, end, shared), dim=-1)
        return self.hid2tag(self.elu(self.dropout(self.ln(self.hid2hid(features)))))

class LogitRE(author.re_unit):
    def forward(self, hidden, shared, mask):
        length = hidden.shape[0]
        head = hidden.unsqueeze(1).expand(-1, length, -1, -1)
        tail = hidden.unsqueeze(0).expand(length, -1, -1, -1)
        features = torch.cat((head, tail, shared), dim=-1)
        return self.hid2rel(self.elu(self.dropout(self.ln(self.hid2hid(features)))))

def valid_cells(token_mask, logits):
    mask = token_mask.squeeze(0)
    pairs = mask[:, None] & mask[None, :]
    return (pairs.triu().unsqueeze(-1).unsqueeze(-1).expand_as(logits[0]),
            pairs.unsqueeze(-1).unsqueeze(-1).expand_as(logits[1]))

def balanced_loss(logits, targets, token_mask):
    losses = []
    for scores, gold, valid in zip(logits, targets, valid_cells(token_mask, logits)):
        if (gold.bool() & ~valid).any():
            raise RuntimeError('Gold positive lies outside valid text cells')
        positive = valid & gold.bool()
        negative = valid & ~gold.bool()
        components = []
        if positive.any():
            components.append(nn.functional.softplus(-scores[positive]).mean())
        if negative.any():
            components.append(nn.functional.softplus(scores[negative]).mean())
        losses.append(torch.stack(components).mean())
    return sum(losses)

def probabilities(logits, token_mask):
    return tuple(torch.sigmoid(x) * valid for x, valid in zip(logits, valid_cells(token_mask, logits)))

class ChineseCARE(author.CARE):
    def __init__(self, model_path, config):
        nn.Module.__init__(self)
        self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.use_checkpointing = config.get("gradient_checkpointing", False)
        if self.use_checkpointing:
            self.bert.gradient_checkpointing_enable()
        hidden = self.bert.config.hidden_size
        args = SimpleNamespace(**config, hidden_size=hidden)
        types, relations, _ = assets()
        self.ner = LogitNER(args, dict(enumerate(types)), hidden, args.share_hidden_size)
        self.re = LogitRE(args, dict(enumerate(relations)), hidden, args.share_hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.dist_emb = nn.Embedding(20, args.dist_emb_size)
        self.conv_attention = author.ConvAttention(hidden, 1, args.dist_emb_size, args.share_hidden_size,
                                                  args.co_attention_layers, 1, args.dropout)

    def forward(self, inputs, token_mask):
        author.device = inputs["input_ids"].device
        hidden = self.bert(**inputs).last_hidden_state
        if self.training:
            hidden = self.dropout(hidden)
        size = hidden.shape[1]
        positions = torch.arange(size, device=hidden.device)
        delta = positions[:, None] - positions[None, :]
        distance = delta.abs()
        # Original bins: 1,2,4,...,256; negative distances offset by nine; zero=19.
        bucket = torch.where(distance > 0, torch.log2(distance.clamp_min(1).float()).floor().long()+1, 19).clamp(max=19)
        bucket = torch.where(delta < 0, bucket.clamp(max=9) + 9, bucket.clamp(max=9))
        bucket = torch.where(delta == 0, 19, bucket)
        dist = self.dist_emb(bucket.unsqueeze(0)).permute(0, 3, 1, 2)
        attention = inputs["attention_mask"].bool()
        pair_mask = (attention[:, None, :, None] & attention[:, None, None, :])
        h_ner, h_re, h_shared = self.module_forward(self.conv_attention, hidden, hidden, dist, pair_mask)
        mask = token_mask.T
        h_shared = h_shared.permute(1, 2, 0, 3)
        return self.module_forward(self.ner, h_ner.permute(1, 0, 2), h_shared, mask), self.module_forward(self.re, h_re.permute(1, 0, 2), h_shared, mask)

    def module_forward(self, module, *inputs):
        if self.training and self.use_checkpointing:
            return checkpoint(module, *inputs, use_reentrant=False, preserve_rng_state=True)
        return module(*inputs)

def encode(tokenizer, text, max_length, device):
    # Per-character WordPiece follows previous Chinese baselines and keeps offsets.
    packed = tokenizer(list(text), is_split_into_words=True, return_tensors="pt", truncation=False)
    words = packed.word_ids(0)
    offsets = [(i, i+1) if i is not None else (0, 0) for i in words]
    if len(words) > max_length:
        raise RuntimeError(f"{len(words)} tokens exceeds {max_length}; no silent truncation")
    inputs = {k: v.to(device) for k, v in packed.items()}
    token_mask = torch.tensor([[a != b for a, b in offsets]], device=device)
    return inputs, token_mask, offsets

def labels(row, offsets, device):
    types, relations, legal = assets()
    size = len(offsets)
    ner = torch.zeros(size, size, 1, len(types), device=device)
    re = torch.zeros(size, size, 1, len(relations), device=device)
    starts = {a: i for i, (a, b) in reversed(list(enumerate(offsets))) if a != b}
    ends = {b: i for i, (a, b) in enumerate(offsets) if a != b}
    spans = []
    for entity in row["span_entities"]:
        a, b = entity["start"], entity["end"]
        if a not in starts or b not in ends:
            raise RuntimeError(f"Gold boundary cannot be token-aligned: {row['record_id']} {entity}")
        i, j = starts[a], ends[b]
        ner[i, j, 0, types.index(entity["entity_type"])] = 1
        spans.append((entity, i))
    for h, r, t in row["relations"]:
        count = 0
        for head, hi in spans:
            for tail, ti in spans:
                if head["surface_form"] == h and tail["surface_form"] == t and (head["entity_type"], r, tail["entity_type"]) in legal:
                    re[hi, ti, 0, relations.index(r)] = 1
                    count += 1
        if not count:
            raise RuntimeError("Gold relation cannot be aligned")
    return ner, re

def tokenizer_for(model_path):
    return AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
