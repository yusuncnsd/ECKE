"""Standard-library data checks, source-aligned decoding and frozen set metrics."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "05_Frozen_Data_Indices/data"
def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text("utf-8-sig").splitlines() if line.strip()]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

def assets():
    schema = json.loads((DATA / "xiaoke_schema_v5.json").read_text("utf-8"))
    types = schema["entity_types"]
    legal = {(s["head_type"], s["relation"], s["tail_type"]) for s in schema["relation_schemas"]}
    return types, sorted({s[1] for s in legal}), legal

def verify_assets():
    manifest = json.loads((ROOT / "asset_manifest.json").read_text("utf-8"))
    for relative, expected in manifest["files"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen asset changed: {relative}")
    types, relations, legal = assets()
    if len(types) != 7 or len(legal) != 7 or len(relations) != 6:
        raise RuntimeError("Expected seven entity types, seven Schema patterns and six relation labels")
    dev = read_jsonl(DATA / "development_100_span_supervision.jsonl")
    test = read_jsonl(DATA / "independent_test_300_input_only.jsonl")
    did = {r["record_id"] for r in dev}
    tid = {r["record_id"] for r in test}
    if len(dev) != 100 or len(did) != 100 or len(test) != 300 or len(tid) != 300 or did & tid:
        raise RuntimeError("100/300 size, uniqueness or separation check failed")
    for row in test:
        if set(row) != {"ann_id", "record_id", "text"}:
            raise RuntimeError("Test must contain input only")
    for row in dev:
        check_entities(row["text"], row["span_entities"], types)
        if len(gold_edges(row)) != len(row["relations"]):
            raise RuntimeError(f"Relation endpoint missing: {row['record_id']}")
        for head, relation, tail in gold_edges(row):
            htypes = {e["entity_type"] for e in row["span_entities"] if e["surface_form"] == head}
            ttypes = {e["entity_type"] for e in row["span_entities"] if e["surface_form"] == tail}
            if not any((h, relation, t) in legal for h in htypes for t in ttypes):
                raise RuntimeError("Illegal development relation")
    seen = []
    index = {r["record_id"]: r for r in dev}
    for fold in range(5):
        train = read_jsonl(DATA / f"cv_fold{fold}_train.jsonl")
        valid = read_jsonl(DATA / f"cv_fold{fold}_valid.jsonl")
        train_ids = {r["record_id"] for r in train}
        valid_ids = {r["record_id"] for r in valid}
        if len(train) != 80 or len(valid) != 20 or train_ids & valid_ids or train_ids | valid_ids != did:
            raise RuntimeError("Frozen CV split check failed")
        for row in train + valid:
            if row != index[row["record_id"]]:
                raise RuntimeError("Fold content differs from development data")
        seen.extend(valid_ids)
    if len(seen) != len(set(seen)) or set(seen) != did:
        raise RuntimeError("Five validation folds must partition the 100 records")
    return {"development": 100, "test_input_only": 300, "cv_folds": 5,
            "span_instances": sum(len(r["span_entities"]) for r in dev),
            "max_development_characters": max(len(r["text"]) for r in dev),
            "max_test_characters": max(len(r["text"]) for r in test),
            "schema_patterns": len(legal), "relation_labels": len(relations)}

def gold_entities(row):
    return {(e["surface_form"], e["entity_type"]) for e in row["span_entities"]}

def gold_edges(row):
    return {tuple(r) for r in row["relations"]}

def check_entities(text, entities, types):
    for item in entities:
        if not (item["entity_type"] in types and 0 <= item["start"] < item["end"] <= len(text)
                and text[item["start"]:item["end"]] == item["surface_form"]):
            raise RuntimeError(f"Invalid entity evidence: {item}")

def decode(text, offsets, entity_hits, relation_hits, types, relations, legal):
    """Expand only predicted spans sharing each predicted relation start token."""
    entities = set()
    by_start = {}
    for start, end, label in entity_hits:
        if not 0 <= start <= end < len(offsets):
            continue
        a, b = offsets[start][0], offsets[end][1]
        if offsets[start][0] == offsets[start][1] or offsets[end][0] == offsets[end][1]:
            continue
        if a >= b:
            continue
        item = (text[a:b], types[label], a, b)
        entities.add(item)
        by_start.setdefault(start, set()).add(item)
    triples = set()
    for head, tail, label in relation_hits:
        for h in by_start.get(head, ()):
            for t in by_start.get(tail, ()):
                if h != t and (h[1], relations[label], t[1]) in legal:
                    triples.add((h[0], relations[label], t[0]))
    return {
        "entities": [{"surface_form": s, "entity_type": typ, "start": a, "end": b} for s, typ, a, b in sorted(entities)],
        "folded_entities": sorted({(s, typ) for s, typ, _, _ in entities}),
        "relations": sorted(triples),
    }

def prf(counts):
    tp, fp, fn = counts
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}

def cv_scores(predictions, gold):
    index = {r["record_id"]: r for r in gold}
    if len(predictions) != len(index) or {r["record_id"] for r in predictions} != set(index):
        raise RuntimeError("CV predictions are incomplete or duplicated")
    counts = [[0, 0, 0], [0, 0, 0]]
    for pred in predictions:
        row = index[pred["record_id"]]
        for total, p, g in zip(counts, [set(map(tuple, pred["folded_entities"])), set(map(tuple, pred["relations"]))], [gold_entities(row), gold_edges(row)]):
            for i, n in enumerate([len(p & g), len(p - g), len(g - p)]):
                total[i] += n
    return {"entity": prf(counts[0]), "relation": prf(counts[1])}

def audit_predictions(predictions, inputs):
    types, relations, legal = assets()
    index = {r["record_id"]: r for r in inputs}
    if len(predictions) != len(index) or {r["record_id"] for r in predictions} != set(index):
        raise RuntimeError("Prediction size/ID completeness failed")
    for pred in predictions:
        row = index[pred["record_id"]]
        check_entities(row["text"], pred["entities"], types)
        folded = {(e["surface_form"], e["entity_type"]) for e in pred["entities"]}
        if set(map(tuple, pred["folded_entities"])) != folded:
            raise RuntimeError("Entity folding mismatch")
        for h, r, t in pred["relations"]:
            if not any((ht, r, tt) in legal for hs, ht in folded for ts, tt in folded if hs == h and ts == t):
                raise RuntimeError("Relation endpoint or Schema check failed")
    return {"complete": True, "records": len(predictions)}

def self_check():
    types, relations, legal = assets()
    text = "消渴用人参，人参。"
    offsets = [(0, 0)] + [(i, i+1) for i in range(len(text))] + [(0, 0)]
    hits = [(1, 2, types.index("疾病")), (4, 5, types.index("中药")), (7, 8, types.index("中药"))]
    out = decode(text, offsets, hits, [(4, 1, relations.index("治疗")), (7, 1, relations.index("治疗")), (1, 4, relations.index("治疗"))], types, relations, legal)
    assert len(out["entities"]) == 3 and len(out["folded_entities"]) == 2
    assert out["relations"] == [("人参", "治疗", "消渴")]
    no_entities = decode(text, offsets, [], [(4, 1, relations.index("治疗"))], types, relations, legal)
    assert not no_entities["relations"]
    score = cv_scores([dict(record_id="a", **out)], [{"record_id": "a", "span_entities": out["entities"], "relations": out["relations"]}])
    assert score["entity"]["f1"] == score["relation"]["f1"] == 1
    print("Synthetic decoding, repeated mention, direction and strict-set checks passed")

if __name__ == "__main__":
    self_check()
    print(json.dumps(verify_assets(), ensure_ascii=False, indent=2))
