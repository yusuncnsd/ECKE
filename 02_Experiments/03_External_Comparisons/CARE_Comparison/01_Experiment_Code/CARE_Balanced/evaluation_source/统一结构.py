import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

ENTITY_TYPES = ("中药", "方药", "疾病", "症状", "病机", "病位", "治法")
RELATION_SCHEMAS = (
    ("中药", "治疗", "疾病"), ("方药", "治疗", "疾病"),
    ("治法", "主治", "疾病"), ("病机", "导致", "疾病"),
    ("病机", "涉及", "病位"), ("症状", "提示", "病机"),
    ("疾病", "表现", "症状"),
)
LEGAL = set(RELATION_SCHEMAS)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def extract_json_object(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        raise ValueError("模型响应不是字符串或对象")
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidates += re.findall(r"(\{.*\})", text, flags=re.S)
    for value in candidates:
        try:
            obj = json.loads(value)
            if isinstance(obj, dict): return obj
        except Exception:
            continue
    raise ValueError("未找到可解析JSON对象")


def normalize_prediction(record_id, text, entities, relations, raw=None, method=None):
    clean_entities = set()
    for e in entities or []:
        if isinstance(e, dict):
            surface = str(e.get("text", e.get("surface_form", e.get("name", "")))).strip()
            etype = str(e.get("type", e.get("entity_type", ""))).strip()
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            surface, etype = str(e[0]).strip(), str(e[1]).strip()
        else:
            continue
        if surface and etype in ENTITY_TYPES and surface in text:
            clean_entities.add((surface, etype))
    surfaces = {}
    for surface, etype in clean_entities:
        surfaces.setdefault(surface, set()).add(etype)
    clean_relations = set()
    for r in relations or []:
        if isinstance(r, dict):
            head = str(r.get("head", r.get("subject", ""))).strip()
            rel = str(r.get("relation", r.get("type", ""))).strip()
            tail = str(r.get("tail", r.get("object", ""))).strip()
        elif isinstance(r, (list, tuple)) and len(r) >= 3:
            head, rel, tail = map(lambda x: str(x).strip(), r[:3])
        else:
            continue
        if head not in surfaces or tail not in surfaces:
            continue
        if any((ht, rel, tt) in LEGAL for ht in surfaces[head] for tt in surfaces[tail]):
            clean_relations.add((head, rel, tail))
    return {
        "record_id": record_id,
        "text": text,
        "entities": [list(x) for x in sorted(clean_entities)],
        "relations": [list(x) for x in sorted(clean_relations)],
        "method": method,
        "raw": raw,
    }

