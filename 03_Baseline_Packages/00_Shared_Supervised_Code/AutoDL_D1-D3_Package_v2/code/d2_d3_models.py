from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertModel


def tiny_bert(vocab_size: int, hidden: int = 64) -> BertModel:
    return BertModel(BertConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=hidden * 4, max_position_embeddings=512))


class CompGCNLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.node = nn.Linear(dim, dim); self.message = nn.Linear(dim, dim); self.rel = nn.Linear(dim, dim)
    def forward(self, nodes, relation_embeddings, edge_scores):
        # edge_scores: [B,N,N,R]; relation-aware weighted messages
        weights = torch.sigmoid(edge_scores)
        messages = torch.einsum("bijr,rd->bijd", weights, relation_embeddings)
        messages = messages + nodes[:, None, :, :]
        aggregate = messages.mean(2)
        return torch.nn.functional.elu(self.node(nodes) + self.message(aggregate)), self.rel(relation_embeddings)


class TCMERE(nn.Module):
    """论文优先的TCMERE核心：字符/词典融合、BiLSTM、span、3层CompGCN、BCE多标签。"""
    def __init__(self, vocab_size: int, entity_count: int, relation_count: int, hidden: int = 64, max_span: int = 15, pretrained: str | None = None):
        super().__init__(); self.max_span = max_span; self.hidden = hidden
        self.encoder = AutoModel.from_pretrained(pretrained) if pretrained else tiny_bert(vocab_size, hidden)
        self.dict_role = nn.Linear(5, 40, bias=False)  # START/MIDDLE/END/SINGLE/DICTIONARY多热通道
        self.lstm = nn.LSTM(hidden + 40, hidden // 2, bidirectional=True, batch_first=True)
        self.width = nn.Embedding(max_span + 1, 8)
        self.attention = nn.Linear(hidden, 1)
        self.span_ffn = nn.Sequential(nn.Linear(hidden * 3 + 8, 500), nn.ELU(), nn.Dropout(.3), nn.Linear(500, hidden), nn.ELU())
        self.entity = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, entity_count))
        self.relation = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ELU(), nn.Linear(hidden, relation_count))
        self.rel_embeddings = nn.Parameter(torch.randn(relation_count, hidden) * .02)
        self.gcn = nn.ModuleList([CompGCNLayer(hidden) for _ in range(3)])

    def encode(self, ids: torch.Tensor, dictionary_roles: torch.Tensor):
        h = self.encoder(ids).last_hidden_state
        if dictionary_roles.dim()==2:
            dictionary_roles=torch.nn.functional.one_hot(dictionary_roles.clamp(0,5),6)[...,1:].float()
        return self.lstm(torch.cat([h, self.dict_role(dictionary_roles.float())], -1))[0]

    def score_entities(self, ids, dictionary_roles, spans):
        return self.entity(self.span_vectors(self.encode(ids,dictionary_roles),spans))

    def span_vectors(self, h: torch.Tensor, spans: torch.Tensor):
        # spans [B,N,2], inclusive
        vectors = []
        for b in range(h.size(0)):
            row = []
            for start, end in spans[b].tolist():
                segment = h[b, start:end + 1]; att = torch.softmax(self.attention(segment).squeeze(-1), 0)
                inside = (segment * att[:, None]).sum(0); width = self.width.weight[min(end - start + 1, self.max_span)]
                row.append(torch.cat([h[b, start], h[b, end], inside, width]))
            vectors.append(torch.stack(row))
        return self.span_ffn(torch.stack(vectors))

    def forward(self, ids, dictionary_roles, spans):
        nodes = self.span_vectors(self.encode(ids, dictionary_roles), spans)
        entity_logits = self.entity(nodes)
        left = nodes[:, :, None, :].expand(-1, -1, nodes.size(1), -1); right = nodes[:, None, :, :].expand_as(left)
        relation_logits = self.relation(torch.cat([left, right, left * right], -1))
        rel = self.rel_embeddings
        for layer in self.gcn: nodes, rel = layer(nodes, rel, relation_logits)
        left = nodes[:, :, None, :].expand(-1, -1, nodes.size(1), -1); right = nodes[:, None, :, :].expand_as(left)
        return self.entity(nodes), self.relation(torch.cat([left, right, left * right], -1))


class GlobalPointer(nn.Module):
    def __init__(self, hidden: int, heads: int, head_size: int = 32):
        super().__init__(); self.heads = heads; self.head_size = head_size; self.proj = nn.Linear(hidden, heads * head_size * 2)
    def forward(self, h):
        b, n, _ = h.shape; x = self.proj(h).view(b, n, self.heads, self.head_size * 2)
        q, k = x.chunk(2, -1)
        pos = torch.arange(n, device=h.device, dtype=h.dtype); freq = torch.pow(10000, -torch.arange(0, self.head_size, 2, device=h.device, dtype=h.dtype) / self.head_size)
        angle = torch.einsum("n,d->nd", pos, freq); sin, cos = angle.sin()[None, :, None, :], angle.cos()[None, :, None, :]
        def rope(z):
            even, odd = z[..., ::2], z[..., 1::2]
            return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)
        q, k = rope(q), rope(k)
        return torch.einsum("bnhd,bmhd->bhnm", q, k) / self.head_size ** .5


class BSTRGPRel(nn.Module):
    """BSTR-GPRel核心：BERT+词性融合+Transformer+三级GlobalPointer。"""
    def __init__(self, vocab_size: int, entity_count: int, relation_count: int, hidden: int = 64, pretrained: str | None = None, pos_count: int = 64):
        super().__init__(); self.encoder = AutoModel.from_pretrained(pretrained) if pretrained else tiny_bert(vocab_size, hidden); self.pos = nn.Embedding(pos_count, 16)
        self.fuse = nn.Linear(hidden + 16, hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout=.1, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.entity_pointer = GlobalPointer(hidden, entity_count)
        self.head_pointer = GlobalPointer(hidden, relation_count)
        self.tail_pointer = GlobalPointer(hidden, relation_count)
        self.loss_log_vars = nn.Parameter(torch.zeros(3))
    def forward(self, ids, pos_ids):
        h = self.encoder(ids).last_hidden_state
        h = self.transformer(self.fuse(torch.cat([h, self.pos(pos_ids)], -1)))
        return self.entity_pointer(h), self.head_pointer(h), self.tail_pointer(h)


def focal_multilabel(logits, labels, gamma: float = 2.0):
    base = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits); pt = p * labels + (1 - p) * (1 - labels)
    return ((1 - pt).pow(gamma) * base).mean()

def dynamic_task_loss(model, losses):
    return sum(torch.exp(-model.loss_log_vars[i])*loss+model.loss_log_vars[i] for i,loss in enumerate(losses))
