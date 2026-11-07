"""V6.1.2消融实验公共引擎。

四个编号入口只传入实验编号和手工API密钥。本文件导入冻结主脚本并仅替换
对应消融机制；每个实验使用独立事务缓存，完成预测后自动调用同一正式评价程序。
"""
from __future__ import annotations

import importlib.util
import http.client
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = (next(p for p in Path(__file__).resolve().parents if (p / "08_Licensing_and_Version/PACKAGE_METADATA.json").is_file()) / '02_Experiments')
BASE_SCRIPT = ROOT / "02_ECKE/01_Code" / "01_运行V612全库知识抽取.py"
EVALUATOR = ROOT / "02_ECKE/01_Code" / "02_评价300条独立测试集.py"
TEST_LIST = ROOT / "04_Ablation_Studies/02_Input_Data/Independent_Test_300/independent_test_300_frozen.jsonl"
GOLD_FILE = ROOT / "04_Ablation_Studies/02_Input_Data/Independent_Test_300/消渴知识抽取_300条独立测试集专家标注-盲标定.txt"
FROZEN_TEST_PREDICTION = ROOT / "04_Ablation_Studies/02_Input_Data/Frozen_ECKE_Reference_Predictions/independent_test_300_machine_predictions_v612_frozen.jsonl"
HERE = Path(__file__).resolve().parent

NAMES = {
    "A1": "去除显式词表锚定",
    "A2": "去除LLM语义补充",
    "A3": "去除证据对齐融合_简单并集",
    "A4": "去除关系候选语义判别_Schema即输出",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_base(base, ablation_id: str, api_key: str) -> Path:
    out = ROOT / "04_Ablation_Studies/04_Results/复现运行" / f"{ablation_id}_{NAMES[ablation_id]}"
    out.mkdir(parents=True, exist_ok=True)
    base.API密钥 = api_key
    base.INPUT_PATH = TEST_LIST
    base.CACHE_DIR = out / "运行缓存"
    base.ENTITY_CACHE = base.CACHE_DIR / "entity_results.jsonl"
    base.RELATION_CACHE = base.CACHE_DIR / "relation_results.jsonl"
    base.FAILURE_PATH = base.CACHE_DIR / "failures.jsonl"
    base.MANIFEST_PATH = base.CACHE_DIR / "manifest.json"
    base.FINAL_PATH = out / f"{ablation_id}_300条结构化预测.jsonl"
    base.TEST_PREDICTION_PATH = base.FINAL_PATH
    base.TEST_EXTRACTION_MANIFEST = out / f"{ablation_id}_运行清单.json"
    base.DYNAMIC_QA_TXT_PATH = out / f"{ablation_id}_动态核验.txt"
    base.DYNAMIC_QA_MANIFEST = out / f"{ablation_id}_动态核验清单.json"
    base.FULL_COMPACT_TXT_PATH = out / f"{ablation_id}_300条预测.txt"
    base.SEMANTIC_DISAGREEMENT_PATH = out / f"{ablation_id}_语义分歧审计.json"
    base.RUN_SUMMARY_PATH = out / f"{ablation_id}_运行摘要.json"
    base.PROMPT_VERSION = f"{base.PROMPT_VERSION}_ablation_{ablation_id}"
    base.POSTPROCESS_VERSION = f"{base.POSTPROCESS_VERSION}_ablation_{ablation_id}"
    # 正式API函数未捕获服务端主动断开连接。这里只补充传输层重试，不改变提示或语义结果。
    original_api_call = base.api_call
    def robust_api_call(messages, seed_offset=0):
        last_error = None
        for attempt in range(1, base.MAX_RETRIES + 1):
            try:
                return original_api_call(messages, seed_offset)
            except (http.client.RemoteDisconnected, ConnectionResetError, ConnectionAbortedError) as exc:
                last_error = exc
                if attempt == base.MAX_RETRIES:
                    break
                time.sleep(min(30, 2 ** (attempt - 1)))
        raise RuntimeError(f"连接被服务端中断，附加重试{base.MAX_RETRIES}次后仍失败：{last_error}")
    base.api_call = robust_api_call

    # 统一进度字段：失败=当前批次实际失败数；待处理=当前批次尚未返回数。
    def progress_line(stage, done, total, success, failed, started):
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 else 0
        remaining = max(0, total - done)
        eta = remaining / rate if rate > 0 else 0
        ratio = done / total if total else 1.0
        return (f"[{stage}] {done}/{total} ({ratio:.1%}) 本批成功={success} 本批失败={failed} "
                f"待处理={remaining} 已用={elapsed:.1f}s 预计剩余={eta:.1f}s")
    base.progress_line = progress_line
    # 300条固定测试运行不生成全库动态质检；中断续跑仍由正式缓存负责。
    base.refresh_dynamic_quality_entities = lambda *args, **kwargs: None
    base.refresh_dynamic_quality_relations = lambda *args, **kwargs: None
    return out


def exact_entities_from_items(base, row: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """仅执行所有方法共同需要的输出合法化，不加入V6.1.2专题融合判断。"""
    values: dict[tuple[int, int, str], dict[str, Any]] = {}
    for item in items:
        surface = str(item.get("surface_form", "")).strip()
        entity_type = str(item.get("entity_type", "")).strip()
        if not surface or entity_type not in base.ENTITY_TYPES or surface not in row["text"]:
            continue
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or row["text"][start:end] != surface:
            start = row["text"].find(surface)
            end = start + len(surface)
        if start < 0:
            continue
        concept = str(item.get("normalized_concept", "")).strip() or surface
        values[(start, end, entity_type)] = {
            "surface_form": surface, "start": start, "end": end,
            "normalized_concept": concept, "entity_type": entity_type,
            "evidence_text": surface, "recognition_source": item.get("recognition_source", "ablation"),
            "alignment_status": "exact", "relevance_reason": item.get("reason", "消融配置输出"),
        }
    result = sorted(values.values(), key=lambda x: (x["start"], x["end"], x["entity_type"]))
    for index, entity in enumerate(result, 1):
        entity["entity_id"] = f"{row['record_id']}_E{index:03d}"
    return result


def entity_from_explicit(base, row: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A2：词表显式候选，采用最长非重叠同类mention作为最低限度结构处理。"""
    ordered = sorted(candidates, key=lambda x: (x["start"], -(x["end"] - x["start"]), x["entity_type"]))
    selected: list[dict[str, Any]] = []
    for candidate in ordered:
        overlaps = [x for x in selected if x["entity_type"] == candidate["entity_type"]
                    and x["start"] < candidate["end"] and candidate["start"] < x["end"]]
        if overlaps:
            continue
        selected.append({**candidate, "recognition_source": "explicit_lexicon_only"})
    return exact_entities_from_items(base, row, selected)


def raw_response_items(base, row: dict[str, Any], candidates: list[dict[str, Any]], response: dict[str, Any], include_all: bool) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not include_all:
        items.extend({**x, "recognition_source": "explicit_candidate"} for x in candidates)
    for claim in response.get("claims", []):
        if not isinstance(claim, dict):
            continue
        for side in ("head", "tail"):
            items.append({
                "surface_form": claim.get(f"{side}_surface", ""),
                "normalized_concept": claim.get(f"{side}_normalized", ""),
                "entity_type": claim.get(f"{side}_type", ""),
                "recognition_source": "llm_raw_claim",
            })
    for entity in response.get("core_entities", []):
        if isinstance(entity, dict):
            items.append({**entity, "recognition_source": "llm_raw_core"})
    if include_all:
        for entity in response.get("entities", []):
            if isinstance(entity, dict):
                items.append({**entity, "recognition_source": "llm_unbounded"})
    return items


def simple_union_validator(base):
    def validator(row, candidates, response, concept_index):
        items = raw_response_items(base, row, candidates, response, include_all=False)
        entities = exact_entities_from_items(base, row, items)
        response["_validated_hypotheses"] = []
        response["_validated_exclusions"] = response.get("excluded_claims", [])
        return entities, ["A3采用显式候选与LLM原始候选简单并集"]
    return validator


def semantic_only_validator(base):
    """A1：不依赖词表候选，以LLM原文mention完成证据定位和Schema合法化。"""
    def validator(row, candidates, response, concept_index):
        items = raw_response_items(base, row, [], response, include_all=False)
        entities = exact_entities_from_items(base, row, items)
        types: dict[str, set[str]] = {}
        for entity in entities:
            types.setdefault(entity["surface_form"], set()).add(entity["entity_type"])
        hypotheses = []
        for claim in response.get("claims", []):
            if not isinstance(claim, dict): continue
            relation = base.CLAIM_RELATIONS.get(str(claim.get("claim_type", "")).strip())
            head, tail = str(claim.get("head_surface", "")).strip(), str(claim.get("tail_surface", "")).strip()
            head_type, tail_type = str(claim.get("head_type", "")).strip(), str(claim.get("tail_type", "")).strip()
            if (head_type, relation, tail_type) not in base.RELATION_SIGNATURES: continue
            if head not in row["text"] or tail not in row["text"]: continue
            if head_type not in types.get(head, set()) or tail_type not in types.get(tail, set()): continue
            hypotheses.append({"head_surface": head, "head_type": head_type, "relation_type": relation,
                               "tail_surface": tail, "tail_type": tail_type,
                               "evidence_unit_ids": claim.get("evidence_unit_ids", []),
                               "reason": str(claim.get("scope_reason", "")).strip(),
                               "claim_type": claim.get("claim_type", "")})
        response["_validated_hypotheses"] = hypotheses
        response["_validated_exclusions"] = response.get("excluded_claims", [])
        return entities, ["A1不使用词表显式候选；保留原文定位与Schema合法化"]
    return validator






def schema_all_relation_processor(base):
    def processor(row, entity_result, schemas):
        candidates = base.relation_candidates(entity_result["entities"], schemas, entity_result.get("semantic_claims", []))
        relations = []
        for candidate in candidates:
            head = next(x for x in entity_result["entities"] if x["entity_id"] == candidate["head_entity_id"])
            tail = next(x for x in entity_result["entities"] if x["entity_id"] == candidate["tail_entity_id"])
            start, end = min(head["start"], tail["start"]), max(head["end"], tail["end"])
            relations.append({**candidate, "schema_compatible": True, "relation_decision": True,
                              "evidence_text": row["text"][start:end], "evidence_start": start, "evidence_end": end,
                              "decision_reason": "A4类型Schema兼容即输出"})
        result = {"record_id": row["record_id"], "status": "success", "relations": relations,
                  "relation_candidate_count": len(candidates), "usage": {}}
        base.append_jsonl(base.RELATION_CACHE, result)
        return result
    return processor






def export_prediction_txt(base, rows, path: Path):
    separator = "=" * 80
    blocks = []
    for index, row in enumerate(rows, 1):
        entities = "\n".join(f"{x['surface_form']} | {x['entity_type']}" for x in row["entities"]) or "无"
        relations = "\n".join(f"{x['head_surface']} | {x['relation_type']} | {x['tail_surface']}" for x in row["relations"]) or "无"
        blocks.append(f"ann_id: test_{index:03d}\nparagraph_id: {row['record_id']}\n\n【原文】\n{row['text']}\n\n【预标实体】\n{entities}\n\n【预标关系】\n{relations}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(separator + "\n" + ("\n\n" + separator + "\n").join(blocks) + "\n", encoding="utf-8")
    os.replace(temp, path)


def evaluate(prediction_txt: Path, out: Path, ablation_id: str):
    evaluator = load_module(EVALUATOR, f"evaluator_{ablation_id}")
    evaluator.PRED_FILE = prediction_txt
    evaluator.GOLD_FILE = GOLD_FILE
    evaluator.OUT_JSON = out / f"{ablation_id}_性能评价.json"
    evaluator.OUT_TXT = out / f"{ablation_id}_性能评价.txt"
    evaluator.main()


def run(ablation_id: str, api_key: str):
    if ablation_id not in NAMES: raise ValueError(f"未知消融编号：{ablation_id}")
    if not api_key.strip(): raise RuntimeError("API密钥为空：请在当前编号入口脚本顶部填写")
    base = load_module(BASE_SCRIPT, f"v612_base_{ablation_id}")
    out = configure_base(base, ablation_id, api_key)
    base.random.seed(base.RANDOM_SEED)
    base.GOLD_EXAMPLES = base.parse_gold()
    rows = base.read_jsonl(TEST_LIST)
    if len(rows) != 300: raise RuntimeError(f"固定测试清单不是300条：{len(rows)}")
    lexicon_payload = json.loads(base.LEXICON_PATH.read_text(encoding="utf-8"))
    schema = json.loads(base.SCHEMA_PATH.read_text(encoding="utf-8"))
    items = lexicon_payload["items"]
    surface_index, concept_index = base.build_lexicon_index(items)
    catalog = base.compact_concept_catalog(items)

    if ablation_id == "A1":
        base.explicit_candidates = lambda text, index: []
        base.validate_and_fuse_entities = semantic_only_validator(base)
    elif ablation_id == "A3":
        base.validate_and_fuse_entities = simple_union_validator(base)
    elif ablation_id == "A4":
        base.process_relation_record = schema_all_relation_processor(base)

    base.validate_manifest()
    base.preflight()
    if ablation_id == "A4":
        # A4只改变关系判别：严格复用完整框架已冻结的300条实体，排除API波动造成的实体差异。
        frozen = {x["record_id"]: x for x in base.read_jsonl(FROZEN_TEST_PREDICTION)}
        missing = [x["record_id"] for x in rows if x["record_id"] not in frozen]
        if missing: raise RuntimeError(f"A4冻结实体来源缺少测试ID：{missing[:5]}")
        entity_map = {row["record_id"]: {"record_id": row["record_id"], "status": "success",
            "entities": frozen[row["record_id"]]["entities"], "semantic_claims": [], "excluded_claims": [],
            "explicit_candidate_count": frozen[row["record_id"]].get("preannotation_meta", {}).get("explicit_candidate_count", 0),
            "usage": {}} for row in rows}
        base.write_jsonl_atomic(base.ENTITY_CACHE, list(entity_map.values()))
        relation_map = base.run_relation_stage(rows, entity_map, schema["relation_schemas"])
    elif ablation_id == "A2":
        entity_map = {}
        a2_started = time.time()
        for index, row in enumerate(rows, 1):
            entities = entity_from_explicit(base, row, base.explicit_candidates(row["text"], surface_index))
            entity_map[row["record_id"]] = {"record_id": row["record_id"], "status": "success", "entities": entities,
                "semantic_claims": [], "excluded_claims": [], "explicit_candidate_count": len(entities), "usage": {}}
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(base.progress_line("A2词表实体", index, len(rows), index, 0, a2_started), flush=True)
        # A2实体确定性生成后一次性安全落盘，关系阶段仍可中断续跑。
        base.write_jsonl_atomic(base.ENTITY_CACHE, list(entity_map.values()))
        relation_map = base.run_relation_stage(rows, entity_map, schema["relation_schemas"])
    else:
        entity_map = base.run_entity_stage(rows, surface_index, concept_index, catalog)
        relation_map = base.run_relation_stage(rows, entity_map, schema["relation_schemas"])

    final_rows = []
    for row in rows:
        rid = row["record_id"]
        final_rows.append({**row, "entities": entity_map[rid]["entities"], "relations": relation_map[rid]["relations"],
                           "ablation_id": ablation_id, "ablation_name": NAMES[ablation_id]})
    base.write_jsonl_atomic(base.FINAL_PATH, final_rows)
    prediction_txt = out / f"{ablation_id}_300条预标预测.txt"
    export_prediction_txt(base, final_rows, prediction_txt)
    summary = {"ablation_id": ablation_id, "name": NAMES[ablation_id], "records": 300,
               "entities": sum(len(x["entities"]) for x in final_rows), "relations": sum(len(x["relations"]) for x in final_rows),
               "model": base.MODEL, "temperature": base.TEMPERATURE, "seed": base.RANDOM_SEED,
               "prediction": str(base.FINAL_PATH), "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    base.RUN_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluate(prediction_txt, out, ablation_id)
    print(f"[{ablation_id}] 抽取与性能评价均已完成：{out}", flush=True)
