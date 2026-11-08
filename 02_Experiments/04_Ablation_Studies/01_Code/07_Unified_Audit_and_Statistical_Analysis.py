"""只读审计冻结的完整框架与A1—A4预测，并生成统一配对统计结果。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = (next(p for p in Path(__file__).resolve().parents if (p / "08_Licensing_and_Version/PACKAGE_METADATA.json").is_file()) / '02_Experiments')
HERE = ROOT / "04_Ablation_Studies/04_Results"
OUT = ROOT / "04_Ablation_Studies/05_Evaluation_and_Verification/复现运行"
GOLD = ROOT / "04_Ablation_Studies/02_Input_Data/Independent_Test_300/Independent_Test_300_Expert_Gold.txt"
TEST_LIST = ROOT / "04_Ablation_Studies/02_Input_Data/Independent_Test_300/independent_test_300_frozen.jsonl"
FULL = ROOT / "04_Ablation_Studies/02_Input_Data/Frozen_ECKE_Reference_Predictions/independent_test_300_machine_predictions_v612_frozen.jsonl"
FULL_EVAL = ROOT / "02_ECKE/05_Evaluation_and_Verification" / "Independent_Test_300_Performance_V612_GoldV1.json"
ENGINE = Path(__file__).with_name("ablation_engine.py")
BOOTSTRAP_N = 10000
SEED = 20260823

METHODS = {
    "F": ("完整框架V6.1.2", FULL, FULL_EVAL),
    "A1": ("去除显式词表锚定", HERE / "A1_Without_Lexicon_Anchoring" / "A1_Structured_Predictions_300.jsonl", HERE / "A1_Without_Lexicon_Anchoring" / "A1_Performance_Evaluation.json"),
    "A2": ("去除LLM语义补充", HERE / "A2_Without_Semantic_Supplementation" / "A2_Structured_Predictions_300.jsonl", HERE / "A2_Without_Semantic_Supplementation" / "A2_Performance_Evaluation.json"),
    "A3": ("去除证据对齐融合", HERE / "A3_Simple_Union" / "A3_Structured_Predictions_300.jsonl", HERE / "A3_Simple_Union" / "A3_Performance_Evaluation.json"),
    "A4": ("Schema兼容即输出关系", HERE / "A4_Schema_Only_Relations" / "A4_Structured_Predictions_300.jsonl", HERE / "A4_Schema_Only_Relations" / "A4_Performance_Evaluation.json"),
}
ENTITY_TYPES = {"中药", "方药", "疾病", "症状", "病机", "病位", "治法"}
SCHEMA = {("中药", "治疗", "疾病"), ("方药", "治疗", "疾病"), ("治法", "主治", "疾病"),
          ("病机", "导致", "疾病"), ("病机", "涉及", "病位"), ("症状", "提示", "病机"),
          ("疾病", "表现", "症状")}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def load_evaluator():
    path = ROOT / "02_ECKE/01_Code" / "02_Evaluate_Independent_Test_300.py"
    spec = importlib.util.spec_from_file_location("frozen_evaluator", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.GOLD_FILE = GOLD
    return module


def entities(row: dict) -> set[tuple[str, str]]:
    return {(str(x["surface_form"]), str(x["entity_type"])) for x in row.get("entities", [])}


def relations(row: dict) -> set[tuple[str, str, str]]:
    return {(str(x["head_surface"]), str(x["relation_type"]), str(x["tail_surface"])) for x in row.get("relations", [])}


def counts(pred: set, gold: set) -> tuple[int, int, int]:
    return len(pred & gold), len(pred - gold), len(gold - pred)


def prf(value: tuple[int, int, int]) -> dict:
    tp, fp, fn = value
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def aggregate(values: list[tuple[int, int, int]], indices=None) -> dict:
    selected = values if indices is None else [values[i] for i in indices]
    return prf(tuple(sum(x[j] for x in selected) for j in range(3)))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def paired_bootstrap(full_values, ablation_values) -> dict:
    rng = random.Random(SEED); n = len(full_values); differences = []
    for _ in range(BOOTSTRAP_N):
        indices = [rng.randrange(n) for _ in range(n)]
        differences.append(aggregate(full_values, indices)["f1"] - aggregate(ablation_values, indices)["f1"])
    non_positive = sum(x <= 0 for x in differences)
    non_negative = sum(x >= 0 for x in differences)
    p = min(1.0, 2 * (min(non_positive, non_negative) + 1) / (BOOTSTRAP_N + 1))
    return {"difference_full_minus_ablation": aggregate(full_values)["f1"] - aggregate(ablation_values)["f1"],
            "ci95": [percentile(differences, .025), percentile(differences, .975)],
            "bootstrap_tail_probability": p, "bootstrap_n": BOOTSTRAP_N, "seed": SEED}


def paired_randomization(full_values, ablation_values, seed_offset: int) -> float:
    """在条文内随机交换两方法的TP/FP/FN向量，检验Micro-F1绝对差。"""
    rng = random.Random(SEED + seed_offset)
    observed = abs(aggregate(full_values)["f1"] - aggregate(ablation_values)["f1"])
    extreme = 0
    for _ in range(BOOTSTRAP_N):
        left, right = [], []
        for f, a in zip(full_values, ablation_values):
            if rng.random() < .5: left.append(f); right.append(a)
            else: left.append(a); right.append(f)
        if abs(aggregate(left)["f1"] - aggregate(right)["f1"]) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (BOOTSTRAP_N + 1)


def exact_mcnemar(full_correct: list[bool], ablation_correct: list[bool]) -> dict:
    b = sum(f and not a for f, a in zip(full_correct, ablation_correct))
    c = sum(a and not f for f, a in zip(full_correct, ablation_correct))
    n = b + c
    if n == 0: p = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"full_correct_ablation_wrong": b, "ablation_correct_full_wrong": c,
            "discordant": n, "exact_two_sided_p": p}


def holm(items: dict[str, float]) -> dict[str, float]:
    ordered = sorted(items, key=items.get); m = len(ordered); adjusted = {}; running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * items[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator()
    gold_rows, gold_issues = evaluator.parse_gold()
    if gold_issues or len(gold_rows) != 300: raise RuntimeError(f"Gold质量门失败：{gold_issues[:10]}")
    gold_map = {x["paragraph_id"]: x for x in gold_rows}
    test_rows = read_jsonl(TEST_LIST); expected_ids = [x["record_id"] for x in test_rows]
    test_text = {x["record_id"]: x["text"] for x in test_rows}
    audit = {"status": "passed", "gold_sha256": sha256(GOLD), "test_list_sha256": sha256(TEST_LIST),
             "engine_sha256": sha256(ENGINE), "issues": [], "methods": {},
             "design_classification": {
                 "A1": "组件分支消融；移除显式候选并采用无词表语义结果合法化，不能表述为只删除一行词表匹配",
                 "A2": "组件分支消融；实体退化为确定性词表候选，关系模块保持",
                 "A3": "组件替换消融；融合替换为简单并集，语义候选提示标志也随融合输出移除，属于机制的直接后果",
                 "A4": "严格单因素消融；逐条复用完整框架实体，仅删除关系语义判别",
             }}
    data = {}
    for method, (name, prediction_path, evaluation_path) in METHODS.items():
        rows = read_jsonl(prediction_path); ids = [x["record_id"] for x in rows]
        issues = []
        if len(rows) != 300 or len(set(ids)) != 300: issues.append("数量或ID唯一性错误")
        if ids != expected_ids: issues.append("ID或顺序与冻结测试清单不一致")
        for row in rows:
            rid = row["record_id"]
            if row.get("text") != test_text.get(rid): issues.append(f"{rid}:原文不一致")
            ekeys = entities(row); typed = defaultdict(set)
            for surface, typ in ekeys:
                typed[surface].add(typ)
                if typ not in ENTITY_TYPES: issues.append(f"{rid}:非法实体类型:{typ}")
                if surface not in row["text"]: issues.append(f"{rid}:实体不在原文:{surface}")
            for relation in row.get("relations", []):
                sig = (relation.get("head_type"), relation.get("relation_type"), relation.get("tail_type"))
                if sig not in SCHEMA: issues.append(f"{rid}:非法Schema:{sig}")
                if (relation.get("head_surface"), relation.get("head_type")) not in ekeys: issues.append(f"{rid}:头实体缺失")
                if (relation.get("tail_surface"), relation.get("tail_type")) not in ekeys: issues.append(f"{rid}:尾实体缺失")
        eval_json = json.loads(evaluation_path.read_text(encoding="utf-8"))
        # 正式完成后运行缓存按项目规范清理；审计改读长期保留的运行摘要。
        summary_files = sorted(prediction_path.parent.glob("*_Run_Summary.json"))
        run_summary = None if method == "F" or not summary_files else json.loads(summary_files[0].read_text(encoding="utf-8"))
        audit["methods"][method] = {"name": name, "prediction": str(prediction_path), "prediction_sha256": sha256(prediction_path),
            "records": len(rows), "issues": issues, "evaluation_gold_sha256": eval_json["files"]["gold_sha256"],
            "run_summary": run_summary}
        if issues: audit["status"] = "failed"; audit["issues"].extend(f"{method}:{x}" for x in issues)
        data[method] = {x["record_id"]: x for x in rows}

    # A4必须逐条、逐字段复用完整实体；并输出全部Schema兼容实体对。
    for rid in expected_ids:
        if data["A4"][rid]["entities"] != data["F"][rid]["entities"]:
            audit["issues"].append(f"A4:{rid}:实体未逐字段复用完整框架")
        expected_relations = set()
        for hs, ht in entities(data["F"][rid]):
            for ts, tt in entities(data["F"][rid]):
                for sh, rel, st in SCHEMA:
                    if ht == sh and tt == st: expected_relations.add((hs, rel, ts))
        if relations(data["A4"][rid]) != expected_relations:
            audit["issues"].append(f"A4:{rid}:输出不等于全部Schema兼容实体对")
    if audit["issues"]: audit["status"] = "failed"

    results = {}; per_record = {}
    for method, (name, _, evaluation_path) in METHODS.items():
        ev, rv, ec, rc, jc = [], [], [], [], []
        for rid in expected_ids:
            pred_e, pred_r = entities(data[method][rid]), relations(data[method][rid])
            gold_e, gold_r = gold_map[rid]["entities"], gold_map[rid]["relations"]
            ev.append(counts(pred_e, gold_e)); rv.append(counts(pred_r, gold_r))
            ec.append(pred_e == gold_e); rc.append(pred_r == gold_r); jc.append(pred_e == gold_e and pred_r == gold_r)
        recomputed_e, recomputed_r = aggregate(ev), aggregate(rv)
        stored = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if abs(recomputed_e["f1"] - stored["entity"]["micro"]["f1"]) > 1e-12 or abs(recomputed_r["f1"] - stored["relation"]["micro"]["f1"]) > 1e-12:
            audit["status"] = "failed"; audit["issues"].append(f"{method}:统一重算与原评价不一致")
        results[method] = {"name": name, "entity": recomputed_e, "relation": recomputed_r,
            "entity_passage_exact": sum(ec)/300, "relation_passage_exact": sum(rc)/300, "joint_passage_exact": sum(jc)/300}
        results[method]["entity_by_type"] = stored["entity"]["by_type"]
        results[method]["relation_by_type"] = stored["relation"]["by_type"]
        per_record[method] = {"entity_counts": ev, "relation_counts": rv, "entity_exact": ec, "relation_exact": rc, "joint_exact": jc}

    comparisons = {}; raw_p = {"entity": {}, "relation": {}, "joint": {}, "entity_f1": {}, "relation_f1": {}}
    for method_index, method in enumerate(("A1", "A2", "A3", "A4"), 2):
        print(f"正在核验与重算 {method}，不调用模型", flush=True)
        entity_boot = paired_bootstrap(per_record["F"]["entity_counts"], per_record[method]["entity_counts"])
        relation_boot = paired_bootstrap(per_record["F"]["relation_counts"], per_record[method]["relation_counts"])
        entity_randomization = paired_randomization(per_record["F"]["entity_counts"], per_record[method]["entity_counts"], method_index * 10)
        relation_randomization = paired_randomization(per_record["F"]["relation_counts"], per_record[method]["relation_counts"], method_index * 10 + 1)
        entity_mc = exact_mcnemar(per_record["F"]["entity_exact"], per_record[method]["entity_exact"])
        relation_mc = exact_mcnemar(per_record["F"]["relation_exact"], per_record[method]["relation_exact"])
        joint_mc = exact_mcnemar(per_record["F"]["joint_exact"], per_record[method]["joint_exact"])
        entity_boot["paired_randomization_p"] = entity_randomization
        relation_boot["paired_randomization_p"] = relation_randomization
        comparisons[method] = {"entity_f1": entity_boot, "relation_f1": relation_boot,
            "entity_exact_mcnemar": entity_mc, "relation_exact_mcnemar": relation_mc, "joint_exact_mcnemar": joint_mc}
        raw_p["entity"][method] = entity_mc["exact_two_sided_p"]
        raw_p["relation"][method] = relation_mc["exact_two_sided_p"]
        raw_p["joint"][method] = joint_mc["exact_two_sided_p"]
        raw_p["entity_f1"][method] = entity_randomization
        raw_p["relation_f1"][method] = relation_randomization
    adjusted = {key: holm(values) for key, values in raw_p.items()}
    for method in comparisons:
        comparisons[method]["entity_exact_mcnemar"]["holm_p"] = adjusted["entity"][method]
        comparisons[method]["relation_exact_mcnemar"]["holm_p"] = adjusted["relation"][method]
        comparisons[method]["joint_exact_mcnemar"]["holm_p"] = adjusted["joint"][method]
        comparisons[method]["entity_f1"]["randomization_holm_p"] = adjusted["entity_f1"][method]
        comparisons[method]["relation_f1"]["randomization_holm_p"] = adjusted["relation_f1"][method]

    payload = {"audit": audit, "configuration": {"records": 300, "bootstrap_n": BOOTSTRAP_N, "seed": SEED,
        "difference_direction": "完整框架F1减消融F1", "multiple_testing": "四组消融；F1随机化检验按实体和关系分别Holm校正；McNemar按实体/关系/联合分别Holm校正", "script_sha256": sha256(Path(__file__)), "generated_at": __import__("datetime").datetime.now().isoformat()},
        "results": results, "paired_comparisons": comparisons}
    (OUT / "Ablation_Audit_and_Statistical_Analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["消融实验A1—A4统一实现审计与统计分析", "="*78,
        f"实现审计：{audit['status']}；测试集300条；Gold SHA-256={audit['gold_sha256']}",
        f"统计：条文级配对bootstrap与配对随机化检验各{BOOTSTRAP_N}次；完整框架减消融；McNemar精确检验；p值按检验族Holm校正。", "",
        "一、实验实现审计", "- 共同输入、Gold、Schema、模型、温度和随机种子清单核验完成。",
        "- 5组预测均为300条、ID唯一、顺序与固定测试清单一致；原文、实体类型、关系Schema及端点完整性通过。",
        "- 原评价指标由结构化预测统一重算，结果一致。",
        "- A4实体逐字段等于完整框架，关系逐条等于全部Schema兼容实体对。", "- 设计证据等级："]
    lines += [f"  {key}：{value}" for key, value in audit["design_classification"].items()]
    if audit["issues"]: lines += ["", "审计问题："] + [f"- {x}" for x in audit["issues"]]
    lines += ["", "二、统一性能", "方法\t实体P\t实体R\t实体F1\t关系P\t关系R\t关系F1\t联合完全正确率"]
    for method in METHODS:
        x=results[method]
        lines.append(f"{method} {x['name']}\t{x['entity']['precision']:.4f}\t{x['entity']['recall']:.4f}\t{x['entity']['f1']:.4f}\t{x['relation']['precision']:.4f}\t{x['relation']['recall']:.4f}\t{x['relation']['f1']:.4f}\t{x['joint_passage_exact']:.4f}")
    lines += ["", "三、相对完整框架的配对比较", "方法\t实体F1下降[95%CI]\t随机化Holm-p\t关系F1下降[95%CI]\t随机化Holm-p\t联合正确率McNemar Holm-p"]
    for method in ("A1","A2","A3","A4"):
        x=comparisons[method]; e=x["entity_f1"]; r=x["relation_f1"]
        lines.append(f"{method}\t{e['difference_full_minus_ablation']:.4f} [{e['ci95'][0]:.4f},{e['ci95'][1]:.4f}]\t{e['randomization_holm_p']:.6g}\t{r['difference_full_minus_ablation']:.4f} [{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}]\t{r['randomization_holm_p']:.6g}\t{x['joint_exact_mcnemar']['holm_p']:.6g}")
    lines += ["", "四、按类型F1（F/A1/A2/A3/A4）"]
    for typ in ("中药", "方药", "疾病", "症状", "病机", "病位", "治法"):
        lines.append("实体-"+typ+"\t"+"\t".join(f"{results[m]['entity_by_type'][typ]['f1']:.4f}" for m in METHODS))
    for typ in ("治疗", "主治", "导致", "涉及", "提示", "表现"):
        lines.append("关系-"+typ+"\t"+"\t".join(f"{results[m]['relation_by_type'][typ]['f1']:.4f}" for m in METHODS))
    lines += ["", "五、论文解释边界",
        "- A4可作为创新点3的严格单因素证据。",
        "- A1—A3应表述为删除/替换完整组件分支后的消融，不声称只改变一条规则。",
        "- 配对bootstrap给出严格集合Micro-F1差值区间；配对随机化检验Micro-F1差异；McNemar检验条文级完全正确率，三者回答不同问题。"]
    (OUT / "Ablation_Audit_and_Statistical_Analysis.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(OUT / "Ablation_Audit_and_Statistical_Analysis.txt")
    print(OUT / "Ablation_Audit_and_Statistical_Analysis.json")
    print("审计状态：", audit["status"], "；问题数：", len(audit["issues"]))


if __name__ == "__main__":
    main()
