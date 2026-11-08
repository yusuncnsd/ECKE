from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from shared_structures import ENTITY_TYPES, RELATION_SCHEMAS, atomic_json, read_jsonl, sha256

MAIN = HERE.parents[2]
GOLD = MAIN / "03_External_Comparisons/02_Input_Data/Independent_Test_300/Independent_Test_300_Expert_Gold.txt"
OUT = MAIN / "03_External_Comparisons/05_Evaluation_and_Verification/复现运行"
BOOTSTRAP_N = 5000
SEED = 20260822
RELATION_TYPES = ("治疗", "主治", "导致", "涉及", "提示", "表现")


def section(block, header, headers):
    alternatives = "|".join(re.escape(x) for x in headers)
    m = re.search(rf"(?ms)^{re.escape(header)}[ \t]*\r?\n(.*?)(?=^(?:{alternatives})[ \t]*\r?$|^[ \t]*={{20,}}[ \t]*\r?$|\Z)", block)
    if not m: raise RuntimeError(f"Gold缺少栏目：{header}")
    return m.group(1).strip()


def lines(raw, width):
    if not raw or raw == "无": return set()
    result = set()
    for line in raw.splitlines():
        if not line.strip(): continue
        parts = tuple(x.strip() for x in re.split(r"[|｜]", line))
        if len(parts) != width or not all(parts): raise RuntimeError(f"Gold字段错误：{line}")
        result.add(parts)
    return result


def parse_gold():
    text = GOLD.read_bytes().decode("utf-8-sig")
    starts = list(re.finditer(r"(?m)^ann_id:\s*test_\d+\s*$", text))
    headers = ("【原文】", "【盲标实体】", "【盲标关系】")
    result = []
    for i, m in enumerate(starts):
        block = text[m.start():starts[i+1].start() if i+1 < len(starts) else len(text)]
        ann = re.search(r"(?m)^ann_id:\s*(\S+)\s*$", block).group(1)
        pid = re.search(r"(?m)^paragraph_id:\s*(\S+)\s*$", block).group(1)
        result.append({"ann_id": ann, "record_id": pid, "text": section(block, "【原文】", headers),
                       "entities": lines(section(block, "【盲标实体】", headers), 2),
                       "relations": lines(section(block, "【盲标关系】", headers), 3)})
    return result


def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else (1.0 if fn == 0 else 0.0)
    r = tp/(tp+fn) if tp+fn else (1.0 if fp == 0 else 0.0)
    f = 2*p*r/(p+r) if p+r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def count(a, b): return len(a & b), len(a - b), len(b - a)


def ci(samples):
    rng = random.Random(SEED); vals = []
    for _ in range(BOOTSTRAP_N):
        chosen = [samples[rng.randrange(len(samples))] for _ in samples]
        vals.append(prf(*(sum(x[i] for x in chosen) for i in range(3)))["f1"])
    vals.sort(); return [vals[int(.025*(BOOTSTRAP_N-1))], vals[int(.975*(BOOTSTRAP_N-1))]]


def main(method, prediction_path):
    prediction_path = Path(prediction_path)
    pred = read_jsonl(prediction_path); gold = parse_gold()
    if len(pred) != 300 or len(gold) != 300: raise RuntimeError(f"数量错误：预测{len(pred)}，Gold{len(gold)}")
    pm = {x["record_id"]: x for x in pred}
    if set(pm) != {x["record_id"] for x in gold}: raise RuntimeError("预测与Gold ID集合不一致")
    totals_e=[0,0,0]; totals_r=[0,0,0]; per_e=[]; per_r=[]
    by_e=defaultdict(lambda:[0,0,0]); by_r=defaultdict(lambda:[0,0,0]); errors=[]; exact_e=exact_r=exact_joint=0
    for g in gold:
        p=pm[g["record_id"]]; pe={tuple(x) for x in p.get("entities",[])}; pr={tuple(x) for x in p.get("relations",[])}
        ce=count(pe,g["entities"]); cr=count(pr,g["relations"]); per_e.append(ce); per_r.append(cr)
        for i in range(3): totals_e[i]+=ce[i]; totals_r[i]+=cr[i]
        for typ in ENTITY_TYPES:
            c=count({x for x in pe if x[1]==typ},{x for x in g["entities"] if x[1]==typ})
            for i in range(3): by_e[typ][i]+=c[i]
        for typ in RELATION_TYPES:
            c=count({x for x in pr if x[1]==typ},{x for x in g["relations"] if x[1]==typ})
            for i in range(3): by_r[typ][i]+=c[i]
        ee,rr=pe==g["entities"],pr==g["relations"]; exact_e+=ee; exact_r+=rr; exact_joint+=ee and rr
        if not (ee and rr):
            errors.append({"ann_id":g["ann_id"],"record_id":g["record_id"],"text":g["text"],
                           "entity_fp":sorted(map(list,pe-g["entities"])),"entity_fn":sorted(map(list,g["entities"]-pe)),
                           "relation_fp":sorted(map(list,pr-g["relations"])),"relation_fn":sorted(map(list,g["relations"]-pr))})
    em,rm=prf(*totals_e),prf(*totals_r)
    result={"method":method,"records":300,"matching":"strict_set_based_order_invariant",
            "files":{"prediction":str(prediction_path),"prediction_sha256":sha256(prediction_path),"gold":str(GOLD),"gold_sha256":sha256(GOLD)},
            "entity":{"micro":em,"ci95_f1":ci(per_e),"passage_exact":exact_e/300,"by_type":{x:prf(*by_e[x]) for x in ENTITY_TYPES}},
            "relation":{"micro":rm,"ci95_f1":ci(per_r),"passage_exact":exact_r/300,"by_type":{x:prf(*by_r[x]) for x in RELATION_TYPES}},
            "joint_passage_exact":exact_joint/300,"mismatched_records":len(errors),"errors":errors}
    OUT.mkdir(parents=True,exist_ok=True); stem=f"{method}_Final_Performance_300"; atomic_json(OUT/f"{stem}.json",result)
    def fmt(x): return f"P={x['precision']:.4f} R={x['recall']:.4f} F1={x['f1']:.4f} (TP={x['tp']}, FP={x['fp']}, FN={x['fn']})"
    lines=[f"{method}：300条独立测试集正式性能评价","="*72,f"预测SHA-256：{sha256(prediction_path)}",f"Gold SHA-256：{sha256(GOLD)}",
           "严格集合评价；实体/关系列表顺序不参与匹配；关系头尾方向严格区分。",
           f"实体：{fmt(em)}  95%CI=[{result['entity']['ci95_f1'][0]:.4f},{result['entity']['ci95_f1'][1]:.4f}]",
           f"关系：{fmt(rm)}  95%CI=[{result['relation']['ci95_f1'][0]:.4f},{result['relation']['ci95_f1'][1]:.4f}]",
           f"实体条文完全正确率={exact_e/300:.4f}",f"关系条文完全正确率={exact_r/300:.4f}",f"联合完全正确率={exact_joint/300:.4f}","",
           "实体分类型："]
    lines += [f"- {x}: {fmt(result['entity']['by_type'][x])}" for x in ENTITY_TYPES]
    lines += ["","关系分类型："]+[f"- {x}: {fmt(result['relation']['by_type'][x])}" for x in RELATION_TYPES]
    (OUT/f"{stem}.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(OUT/f"{stem}.txt"); print("实体",fmt(em)); print("关系",fmt(rm))


if __name__ == "__main__":
    if len(sys.argv)!=3: raise SystemExit("用法：python 03_Unified_Evaluation.py E1|E3|E4 Predictions.jsonl")
    main(sys.argv[1],sys.argv[2])
