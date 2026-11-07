from __future__ import annotations

import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API密钥 = ""
运行模式 = "full300"  # pilot10 或 full300

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMMON_DIR = ROOT / "04_Unified_Evaluation"
sys.path.insert(0, str(COMMON_DIR))
from 统一结构 import append_jsonl, atomic_json, extract_json_object, normalize_prediction, read_jsonl, sha256

DATA = ROOT.parent / "02_Input_Data/Model_Inference_Inputs"
DEV_PATH = DATA / "development_100_gold.jsonl"
TEST_PATH = DATA / "independent_test_300_input_only.jsonl"
SCHEMA_PATH = DATA / "xiaoke_schema_v5.json"
CONFIG_PATH = ROOT.parent / "03_Configuration_and_Instructions/E1-E2正式实验冻结配置.json"
PRED_DIR = ROOT.parent / "04_Results/E1/复现运行"
CACHE_PATH = PRED_DIR / "E1_OneKE-Qwen_raw_cache.jsonl"
FINAL_PATH = PRED_DIR / "E1_OneKE-Qwen_300条正式预测.jsonl"
MANIFEST_PATH = PRED_DIR / "E1_OneKE-Qwen_run_manifest.json"
FAIL_PATH = PRED_DIR / "E1_OneKE-Qwen_failures.json"

MODEL = "qwen-plus-2025-07-28"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
WORKERS = 5
MAX_RETRIES = 6
SEED = 20260813
ENTITY_TYPES = ["中药", "方药", "疾病", "症状", "病机", "病位", "治法"]
RELATION_SCHEMA = [
    ["中药", "治疗", "疾病"], ["方药", "治疗", "疾病"], ["治法", "主治", "疾病"],
    ["病机", "导致", "疾病"], ["病机", "涉及", "病位"], ["症状", "提示", "病机"],
    ["疾病", "表现", "症状"],
]
LOCK = threading.Lock()


def api_call(prompt: str, temperature: float, salt: int) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": SEED + salt,
        "response_format": {"type": "json_object"},
    }
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers={"Authorization": f"Bearer {API密钥}", "Content-Type": "application/json"}, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last = exc
            if attempt + 1 < MAX_RETRIES:
                time.sleep(min(30, 2 ** attempt) + random.Random(SEED + salt + attempt).random())
    raise RuntimeError(f"API在{MAX_RETRIES}次重试后仍失败：{last}")


class FrozenCaseRepository:
    """OneKE检索思想的本地只读适配：语义与字符串两路得分等权融合。"""
    def __init__(self, rows):
        self.rows = rows
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([x["text"] for x in rows])

    @staticmethod
    def _norm(values):
        lo, hi = min(values), max(values)
        return [(x - lo) / (hi - lo) if hi > lo else x for x in values]

    def query(self, text, top_k=2):
        semantic = cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0].tolist()
        string = [SequenceMatcher(None, text, x["text"]).ratio() for x in self.rows]
        a, b = self._norm(semantic), self._norm(string)
        scores = [0.5 * x + 0.5 * y for x, y in zip(a, b)]
        indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        return [self.rows[i] for i in indices], [scores[i] for i in indices]


def output_schema():
    return json.dumps({
        "entities": [{"text": "原文逐字表面形式", "type": "七类实体之一"}],
        "relations": [{"head": "头实体表面形式", "relation": "关系名称", "tail": "尾实体表面形式"}],
    }, ensure_ascii=False)


def case_text(case):
    answer = {"entities": [{"text": x[0], "type": x[1]} for x in case["entities"]],
              "relations": [{"head": x[0], "relation": x[1], "tail": x[2]} for x in case["relations"]]}
    return f"**Text**: {case['text']}\n**Correct Answer**: {json.dumps(answer, ensure_ascii=False)}"


def extraction_prompt(text, cases):
    examples = "\n\n".join(f"Example {i+1}:\n{case_text(x)}" for i, x in enumerate(cases))
    return f"""**Instruction**: You are an agent skilled in information extraction. Extract only xiaoke-related entities and relations from the current Chinese medical classical text.

Entity types: {json.dumps(ENTITY_TYPES, ensure_ascii=False)}
Legal relation schemas (head type, relation, tail type): {json.dumps(RELATION_SCHEMA, ensure_ascii=False)}
Do not extract formula ingredients merely because a formula treats xiaoke. Every entity surface must occur verbatim in the current text. Return empty lists when no item exists.

Here are the Top-2 cases retrieved from the frozen 100-record development case repository:
{examples}

**Text**: {text}
**Output Schema**: {output_schema()}

Return one JSON object only. Do not add explanation."""


def reflection_prompt(text, cases, original):
    refs = "\n\n".join(case_text(x) for x in cases)
    return f"""**Instruction**: You are the OneKE reflection agent. Review the original extraction against the current text, the fixed Schema, and the retrieved reference cases. Correct omissions, unsupported items, type errors, illegal relation schemas, and relation endpoints absent from the entity set. If it is already correct, return it unchanged.

**Reflection Reference**:
{refs}

**Text**: {text}
**Entity Types**: {json.dumps(ENTITY_TYPES, ensure_ascii=False)}
**Relation Schemas**: {json.dumps(RELATION_SCHEMA, ensure_ascii=False)}
**Output Schema**: {output_schema()}
**Original Result**: {json.dumps(original, ensure_ascii=False)}

Return one JSON object only."""


def canonical(obj):
    entities, relations = [], []
    if isinstance(obj, dict):
        for e in obj.get("entities", []) or []:
            if isinstance(e, dict):
                entities.append((str(e.get("text", e.get("surface_form", e.get("name", "")))).strip(),
                                 str(e.get("type", e.get("entity_type", ""))).strip()))
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                entities.append((str(e[0]).strip(), str(e[1]).strip()))
        for r in obj.get("relations", []) or []:
            if isinstance(r, dict):
                relations.append((str(r.get("head", r.get("subject", ""))).strip(),
                                  str(r.get("relation", r.get("type", ""))).strip(),
                                  str(r.get("tail", r.get("object", ""))).strip()))
            elif isinstance(r, (list, tuple)) and len(r) >= 3:
                relations.append(tuple(str(x).strip() for x in r[:3]))
    # OneKE一致性比较按结构集合进行，避免仅因输出行顺序不同触发反思。
    return json.dumps({"entities": sorted(set(entities)), "relations": sorted(set(relations))}, ensure_ascii=False, sort_keys=True)


def process(row, repo, ordinal):
    cases, scores = repo.query(row["text"], 2)
    prompt = extraction_prompt(row["text"], cases)
    raw = []; parsed = []
    for i, temperature in enumerate((0.0, 0.5, 1.0)):
        response = api_call(prompt, temperature, ordinal * 10 + i)
        raw.append(response); parsed.append(extract_json_object(response))
    counts = Counter(canonical(x) for x in parsed)
    majority = next((key for key, count in counts.items() if count >= 2), None)
    reflected = False
    if majority is not None:
        chosen = parsed[[canonical(x) for x in parsed].index(majority)]
    else:
        chosen = max(parsed, key=lambda x: len(canonical(x)))
        response = api_call(reflection_prompt(row["text"], cases, chosen), 0.0, ordinal * 10 + 3)
        raw.append(response); chosen = extract_json_object(response); reflected = True
    normalized = normalize_prediction(row["record_id"], row["text"], chosen.get("entities", []), chosen.get("relations", []), method="E1_OneKE-Qwen")
    normalized.update({"ann_id": row["ann_id"], "case_ids": [x["ann_id"] for x in cases], "case_scores": scores,
                       "self_consistency": counts.most_common(), "reflected": reflected, "raw_responses": raw})
    return normalized


def preflight():
    if not API密钥.strip(): raise RuntimeError("请先在脚本顶部填写API密钥")
    if 运行模式 not in ("pilot10", "full300"): raise RuntimeError("运行模式只能是pilot10或full300")
    dev, test = read_jsonl(DEV_PATH), read_jsonl(TEST_PATH)
    if len(dev) != 100 or len(test) != 300: raise RuntimeError(f"数据数量错误：开发{len(dev)}，测试{len(test)}")
    if any("entities" in x or "relations" in x for x in test): raise RuntimeError("测试输入疑似包含Gold")
    response = api_call('仅返回JSON：{"status":"ok"}', 0.0, 0)
    if extract_json_object(response).get("status") != "ok": raise RuntimeError("API预检响应异常")
    return dev, test


def main():
    dev, all_test = preflight()
    rows = all_test[:10] if 运行模式 == "pilot10" else all_test
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    binding = {"version": "E1_OneKE-Qwen_v1", "model": MODEL, "mode_shared_cache": True,
               "development_sha256": sha256(DEV_PATH), "test_sha256": sha256(TEST_PATH),
               "schema_sha256": sha256(SCHEMA_PATH), "config_sha256": sha256(CONFIG_PATH)}
    if MANIFEST_PATH.exists() and json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) != binding:
        raise RuntimeError("E1缓存与当前实验指纹不一致，禁止混用")
    atomic_json(MANIFEST_PATH, binding)
    existing = {x["record_id"]: x for x in read_jsonl(CACHE_PATH)} if CACHE_PATH.exists() else {}
    pending = [(i, x) for i, x in enumerate(rows, 1) if x["record_id"] not in existing]
    print(f"[E1] 已恢复{len(rows)-len(pending)}条，待处理{len(pending)}条。")
    failures = [] ; start = time.time(); completed = 0
    repo = FrozenCaseRepository(dev)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process, row, repo, i): row for i, row in pending}
        for future in as_completed(futures):
            row = futures[future]; completed += 1
            try:
                result = future.result(); append_jsonl(CACHE_PATH, result); existing[result["record_id"]] = result
            except Exception as exc:
                failures.append({"record_id": row["record_id"], "error": repr(exc)})
            elapsed = time.time() - start; rate = elapsed / completed; remain = rate * (len(pending) - completed)
            print(f"[E1] {completed}/{len(pending)} ({100*completed/max(1,len(pending)):.1f}%) 成功={completed-len(failures)} 失败={len(failures)} 待处理={len(pending)-completed} 已用={elapsed:.1f}s 预计剩余={remain:.1f}s", flush=True)
    atomic_json(FAIL_PATH, failures)
    missing = [x["record_id"] for x in rows if x["record_id"] not in existing]
    if missing: raise RuntimeError(f"E1未全部完成：缺少{len(missing)}条；重新运行只处理失败项")
    output = []
    for row in rows:
        item = dict(existing[row["record_id"]]); item.pop("raw_responses", None); output.append(item)
    target = FINAL_PATH if 运行模式 == "full300" else PRED_DIR / "E1_OneKE-Qwen_固定10条验证.jsonl"
    target.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in output) + "\n", encoding="utf-8")
    print(f"E1完成：{target}")
    if 运行模式 == "full300":
        import subprocess
        subprocess.run([sys.executable, str(COMMON_DIR / "03_统一评价.py"), "E1", str(FINAL_PATH)], check=True)


if __name__ == "__main__": main()
