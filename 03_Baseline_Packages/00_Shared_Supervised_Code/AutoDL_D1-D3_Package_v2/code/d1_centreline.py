from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertModel


class CharVocab:
    def __init__(self, texts: list[str], entity_types: list[str]):
        special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + [f"[S-{x}]" for x in entity_types] + [f"[/S-{x}]" for x in entity_types] + [f"[O-{x}]" for x in entity_types] + [f"[/O-{x}]" for x in entity_types]
        self.itos = special + sorted(set("".join(texts)) - set(special))
        self.stoi = {x: i for i, x in enumerate(self.itos)}
    def encode(self, text: str, max_len: int) -> list[int]:
        return [self.stoi["[CLS]"]] + [self.stoi.get(x, 1) for x in text[:max_len - 2]] + [self.stoi["[SEP]"]]


def locate(text: str, surface: str) -> tuple[int, int] | None:
    start = text.find(surface)
    return None if start < 0 else (start + 1, start + len(surface))  # +1 for CLS; end is inclusive grid index


class GridNER(nn.Module):
    """CENTRELINE核心复现：BERT-BiLSTM-CLN-多尺度空洞卷积-双CTP。"""
    def __init__(self, vocab_size: int, type_count: int, hidden: int = 64, pretrained: str | None = None):
        super().__init__()
        config = BertConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=2, num_attention_heads=4, intermediate_size=hidden * 4, max_position_embeddings=256)
        self.bert = AutoModel.from_pretrained(pretrained) if pretrained else BertModel(config)
        self.lstm = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.cond = nn.Linear(hidden, hidden * 2)
        self.reduce = nn.Linear(hidden + 10, hidden)
        self.dist = nn.Embedding(129, 8); self.region = nn.Embedding(2, 2)
        self.convs = nn.ModuleList([nn.Conv2d(hidden, hidden // 2, 3, padding=d, dilation=d) for d in (1, 2, 3)])
        grid_dim = hidden // 2 * 3
        self.grid_heads = nn.ModuleList([nn.Linear(grid_dim, type_count + 1) for _ in range(2)])
        self.left = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.right = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.biaffine = nn.ParameterList([nn.Parameter(torch.empty(type_count + 1, hidden, hidden)) for _ in range(2)])
        for x in self.biaffine: nn.init.xavier_uniform_(x)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.lstm(self.bert(ids).last_hidden_state)[0]
        gamma, beta = self.cond(h).chunk(2, -1)
        base = nn.functional.layer_norm(h[:, None, :, :], (h.size(-1),))
        grid = base * (1 + gamma[:, :, None, :]) + beta[:, :, None, :]
        n = h.size(1); pos = torch.arange(n, device=h.device)
        delta = (pos[:, None] - pos[None, :]).clamp(-64, 64) + 64
        region = (pos[:, None] <= pos[None, :]).long()
        extra = torch.cat([self.dist(delta), self.region(region)], -1).unsqueeze(0).expand(h.size(0), -1, -1, -1)
        grid = torch.nn.functional.gelu(self.reduce(torch.cat([grid, extra], -1)))
        conv = torch.cat([torch.nn.functional.gelu(x(grid.permute(0, 3, 1, 2))) for x in self.convs], 1).permute(0, 2, 3, 1)
        outputs = []
        for k in range(2):
            l, r = self.left[k](h), self.right[k](h)
            bia = torch.einsum("bih,chd,bjd->bijc", l, self.biaffine[k], r)
            outputs.append(bia + self.grid_heads[k](conv))
        return outputs[0], outputs[1]


class RelationClassifier(nn.Module):
    def __init__(self, vocab_size: int, relation_count: int, hidden: int = 64, pretrained: str | None = None):
        super().__init__()
        config = BertConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=2, num_attention_heads=4, intermediate_size=hidden * 4, max_position_embeddings=256)
        self.bert = AutoModel.from_pretrained(pretrained) if pretrained else BertModel(config)
        self.classifier = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, relation_count + 1))
    def forward(self, ids: torch.Tensor, subj: torch.Tensor, obj: torch.Tensor):
        h = self.bert(ids,attention_mask=ids.ne(0)).last_hidden_state
        batch = torch.arange(ids.size(0), device=ids.device)
        return self.classifier(torch.cat([h[batch, subj], h[batch, obj]], -1))


def pad(sequences: list[list[int]]) -> torch.Tensor:
    width = max(map(len, sequences)); out = torch.zeros(len(sequences), width, dtype=torch.long)
    for i, x in enumerate(sequences): out[i, :len(x)] = torch.tensor(x)
    return out


def marked_pair(vocab: CharVocab, text: str, head: tuple[str, str], tail: tuple[str, str], max_len: int = 160):
    hs, ht = head; os, ot = tail
    h0, o0 = text.find(hs), text.find(os)
    if h0 < 0 or o0 < 0: return None
    h1, o1 = h0 + len(hs) - 1, o0 + len(os) - 1
    tokens = ["[CLS]"]; subj = obj = None
    for i, char in enumerate(text):
        if i == h0: subj = len(tokens); tokens.append(f"[S-{ht}]")
        if i == o0: obj = len(tokens); tokens.append(f"[O-{ot}]")
        tokens.append(char)
        if i == h1: tokens.append(f"[/S-{ht}]")
        if i == o1: tokens.append(f"[/O-{ot}]")
    tokens.append("[SEP]")
    if subj is None or obj is None or subj >= max_len or obj >= max_len: return None
    tokens = tokens[:max_len]
    if tokens[-1] != "[SEP]": tokens[-1] = "[SEP]"
    return [vocab.stoi.get(x, 1) for x in tokens], subj, obj


def smoke_run(train: list[dict], valid: list[dict], schema: dict, out: Path, seed: int = 20260823) -> dict:
    random.seed(seed); torch.manual_seed(seed); out.mkdir(parents=True, exist_ok=True)
    entity_types = schema["entity_types"]; type_id = {x: i + 1 for i, x in enumerate(entity_types)}
    relation_labels = sorted({x["relation"] for x in schema["relation_schemas"]}); rel_id = {x: i + 1 for i, x in enumerate(relation_labels)}
    vocab = CharVocab([x["text"] for x in train + valid], entity_types)
    ner = GridNER(len(vocab.itos), len(entity_types)); optimizer = torch.optim.AdamW(ner.parameters(), lr=2e-3)
    ner.train(); losses = []
    for row in train:
        ids = pad([vocab.encode(row["text"], 96)]); a, b = ner(ids); n = ids.size(1)
        target = torch.zeros((1, n, n), dtype=torch.long)
        for surface, typ in row["entities"]:
            span = locate(row["text"][:94], surface)
            if span and span[1] < n: target[0, span[0], span[1]] = type_id[typ]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool)).unsqueeze(0)
        loss = nn.functional.cross_entropy(a[mask], target[mask]) + nn.functional.cross_entropy(b[mask], target[mask])
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    # RE按论文采用Gold实体训练，并显式学习无关系epsilon类。
    re_model = RelationClassifier(len(vocab.itos), len(relation_labels)); re_optimizer = torch.optim.AdamW(re_model.parameters(), lr=2e-3)
    relation_losses = []
    re_model.train()
    for row in train:
        gold_rel = {(h, t): r for h, r, t in row["relations"]}
        pairs = [(h, t) for h in row["entities"] for t in row["entities"] if h != t]
        for head, tail in pairs[:80]:
            item = marked_pair(vocab, row["text"], head, tail)
            if not item: continue
            seq, subj, obj = item; ids = pad([seq])
            label = torch.tensor([rel_id.get(gold_rel.get((head[0], tail[0]), ""), 0)])
            logits = re_model(ids, torch.tensor([subj]), torch.tensor([obj]))
            loss = nn.functional.cross_entropy(logits, label)
            re_optimizer.zero_grad(); loss.backward(); re_optimizer.step(); relation_losses.append(float(loss.detach()))
    # Smoke prediction deliberately decodes trained models; noGold复制进预测。
    predictions = []
    ner.eval()
    with torch.no_grad():
        for row in valid:
            ids = pad([vocab.encode(row["text"], 96)]); a, b = ner(ids); labels_a = a.argmax(-1)[0]; labels_b = b.argmax(-1)[0]
            entities = set()
            n = ids.size(1)
            for i in range(1, n - 1):
                for j in range(i, n - 1):
                    for label in {int(labels_a[i, j]), int(labels_b[i, j])} - {0}:
                        surface = row["text"][i - 1:j]
                        if surface: entities.add((surface, entity_types[label - 1]))
            relations = set(); entity_list = sorted(entities)
            for head in entity_list:
                for tail in entity_list:
                    if head == tail: continue
                    item = marked_pair(vocab, row["text"], head, tail)
                    if not item: continue
                    seq, subj, obj = item
                    logits = re_model(pad([seq]), torch.tensor([subj]), torch.tensor([obj]))
                    label = int(logits.argmax(-1))
                    if label: relations.add((head[0], relation_labels[label - 1], tail[0]))
            predictions.append({"record_id": row["record_id"], "entities": entity_list, "relations": sorted(relations)})
    torch.save({"ner_model": ner.state_dict(), "relation_model": re_model.state_dict(), "vocab": vocab.itos, "schema": schema}, out / "D1_小样本检查点.pt")
    (out / "D1_小样本Predictions.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n", encoding="utf-8")
    return {"train_records": len(train), "validation_records": len(valid), "epochs": 1, "encoder": "随机初始化微型BERT（仅流程验证）", "mean_ner_loss": sum(losses) / len(losses), "mean_relation_loss": sum(relation_losses) / len(relation_losses), "relation_training_instances": len(relation_losses), "prediction_records": len(predictions), "relations_note": "RE使用Gold实体训练、使用D1预测实体推理，端到端链路已执行。", "predictions": predictions}
