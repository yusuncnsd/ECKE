from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
API密钥 = ''
ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / '主实验'
CORPUS_PATH = MAIN / '02_ECKE/02_输入数据/专题条文库' / '正式消渴主题条文库.jsonl'
TEST_PATH = MAIN / '02_ECKE/02_输入数据/独立测试集300' / 'independent_test_300_frozen.jsonl'
LEXICON_PATH = MAIN / '02_ECKE/03_配置与运行说明' / 'xiaoke_lexicon_normalized_v1.json'
SCHEMA_PATH = MAIN / '02_ECKE/03_配置与运行说明' / 'xiaoke_schema_v5.json'
STRATEGY_PATH = MAIN / '02_ECKE/03_配置与运行说明' / '消渴知识抽取最终冻结策略_v612.txt'
GOLD_PATH = MAIN / '02_ECKE/02_输入数据/开发集100' / '消渴知识抽取_100条专家标注_V3.txt'
INPUT_PATH = CORPUS_PATH
REPRODUCTION_DIR = MAIN / '02_ECKE/04_结果数据/复现运行'
CACHE_DIR = REPRODUCTION_DIR / '运行缓存'
FINAL_PATH = REPRODUCTION_DIR / 'xiaoke_full_corpus_predictions_v612_reproduction.jsonl'
TEST_PREDICTION_PATH = REPRODUCTION_DIR / 'independent_test_300_machine_predictions_v612_reproduction.jsonl'
TEST_EXTRACTION_MANIFEST = REPRODUCTION_DIR / 'independent_test_300_extraction_manifest_v612_reproduction.json'
DYNAMIC_QA_TXT_PATH = REPRODUCTION_DIR / '全库动态质量核验_V612_复现累计抽样.txt'
DYNAMIC_QA_MANIFEST = REPRODUCTION_DIR / '全库动态质量核验_V612_复现抽样清单.json'
FULL_COMPACT_TXT_PATH = REPRODUCTION_DIR / '消渴知识抽取_全库复现结果_V612.txt'
SEMANTIC_DISAGREEMENT_PATH = REPRODUCTION_DIR / 'V612复现实体关系语义分歧审计.json'
RUN_SUMMARY_PATH = REPRODUCTION_DIR / 'V612全库复现运行摘要.json'
API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
MODEL = 'qwen-plus-2025-07-28'
TEMPERATURE = 0.0
RANDOM_SEED = 20260813
MAX_RETRIES = 6
MAX_WORKERS = 5
TIMEOUT_SECONDS = 180
PROMPT_VERSION = 'xiaoke_final_v6_overlap_disambiguation_20260817'
POSTPROCESS_VERSION = 'xiaoke_final_rules_v612_quality_gate_separation_20260817'
ENTITY_TYPES = {'中药', '方药', '疾病', '症状', '病机', '病位', '治法'}
RELATION_SIGNATURES = {('中药', '治疗', '疾病'), ('方药', '治疗', '疾病'), ('治法', '主治', '疾病'), ('病机', '导致', '疾病'), ('病机', '涉及', '病位'), ('症状', '提示', '病机'), ('疾病', '表现', '症状')}
CLAIM_RELATIONS = {'treatment': '治疗', 'disease_manifestation': '表现', 'pathogenesis_cause': '导致', 'pathogenesis_location': '涉及', 'symptom_indicates_pathogenesis': '提示', 'treatment_method': '主治'}
ALLOWED_TREATMENT_ROLES = {'entry_title', 'prescription_title', 'named_formula', 'single_herb'}
TOPIC_STANDALONE_SYMPTOMS = {'内消', '胃消', '渴疾', '渴症', '大渴', '中焦渴', '下焦渴', '谷消'}
DIABETES_ALIAS = '糖尿病'
GOLD_EXAMPLES: dict[str, dict[str, Any]] = {}
ENTITY_CACHE = CACHE_DIR / 'entity_results.jsonl'
RELATION_CACHE = CACHE_DIR / 'relation_results.jsonl'
FAILURE_PATH = CACHE_DIR / 'failures.jsonl'
MANIFEST_PATH = CACHE_DIR / 'manifest.json'
WRITE_LOCK = threading.Lock()
API_SEMAPHORE = threading.BoundedSemaphore(MAX_WORKERS)

class DataInspectionError(RuntimeError):
    pass

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def pipeline_semantic_sha256() -> str:
    source = Path(__file__).resolve().read_text(encoding='utf-8')
    source = re.sub('(?m)^API密钥\\s*=.*$', 'API密钥 = "<MANUAL_INPUT>"', source)
    return hashlib.sha256(source.encode('utf-8')).hexdigest()

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}第{line_no}行JSON损坏') from exc
    return rows

def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in rows)), encoding='utf-8')
    os.replace(temp, path)

def dynamic_run_fingerprint() -> dict[str, Any]:
    return {'input_sha256': sha256_file(INPUT_PATH), 'lexicon_sha256': sha256_file(LEXICON_PATH), 'schema_sha256': sha256_file(SCHEMA_PATH), 'strategy_sha256': sha256_file(STRATEGY_PATH), 'model': MODEL, 'prompt_version': PROMPT_VERSION, 'postprocess_version': POSTPROCESS_VERSION, 'random_seed': RANDOM_SEED}

def load_dynamic_quality_state() -> dict[str, Any]:
    fingerprint = dynamic_run_fingerprint()
    state = {'completed_entity_batches': 0, 'selected_ids': [], 'entity_batched_ids': [], 'run_fingerprint': fingerprint}
    if DYNAMIC_QA_MANIFEST.exists():
        state = json.loads(DYNAMIC_QA_MANIFEST.read_text(encoding='utf-8'))
        if state.get('run_fingerprint') != fingerprint:
            raise RuntimeError('V6.1.2动态质检清单与当前实验指纹不一致，禁止混用')
    return state

def export_dynamic_quality_txt(rows: list[dict[str, Any]], entity_map: dict[str, dict[str, Any]], relation_map: dict[str, dict[str, Any]], selected_ids: list[str]) -> None:
    rowmap = {x['record_id']: x for x in rows}
    separator = '=' * 80
    blocks = ['【说明】本文件由V6.1.2程序动态更新。每完成100条实体固定抽10条，并立即完成这10条的关系识别。运行期间请勿直接校改本文件。']
    for idx, rid in enumerate(selected_ids, 1):
        row, entity_result = (rowmap[rid], entity_map[rid])
        entity_lines = '\n'.join((f"{e['surface_form']} | {e['entity_type']}" for e in entity_result.get('entities', []))) or '无'
        relation_result = relation_map.get(rid)
        if relation_result is None:
            relation_lines = '关系识别失败，待续跑'
        else:
            relation_lines = '\n'.join((f"{r['head_surface']} | {r['relation_type']} | {r['tail_surface']}" for r in relation_result.get('relations', []))) or '无'
        blocks.append(f"ann_id: quality_{idx:03d}\nparagraph_id: {rid}\n\n【原文】\n{row['text']}\n\n【实体】\n{entity_lines}\n\n【关系】\n{relation_lines}")
    DYNAMIC_QA_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = DYNAMIC_QA_TXT_PATH.with_suffix(DYNAMIC_QA_TXT_PATH.suffix + '.tmp')
    temp.write_text(('\n\n' + separator + '\n').join(blocks) + '\n', encoding='utf-8')
    os.replace(temp, DYNAMIC_QA_TXT_PATH)

def export_full_compact_txt(rows: list[dict[str, Any]]) -> None:
    separator = '=' * 80
    blocks = []
    for idx, row in enumerate(rows, 1):
        entity_lines = '\n'.join((f"{e['surface_form']} | {e['entity_type']}" for e in row.get('entities', []))) or '无'
        relation_lines = '\n'.join((f"{r['head_surface']} | {r['relation_type']} | {r['tail_surface']}" for r in row.get('relations', []))) or '无'
        blocks.append(f"ann_id: full_{idx:04d}\nparagraph_id: {row['record_id']}\n\n【原文】\n{row['text']}\n\n【实体】\n{entity_lines}\n\n【关系】\n{relation_lines}")
    FULL_COMPACT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = FULL_COMPACT_TXT_PATH.with_suffix(FULL_COMPACT_TXT_PATH.suffix + '.tmp')
    temp.write_text(separator + '\n' + ('\n\n' + separator + '\n').join(blocks) + '\n', encoding='utf-8')
    os.replace(temp, FULL_COMPACT_TXT_PATH)

def refresh_dynamic_quality_entities(rows: list[dict[str, Any]], entity_map: dict[str, dict[str, Any]]) -> None:
    state = load_dynamic_quality_state()
    batched = set(state.get('entity_batched_ids', []))
    completed = [x for x in rows if x['record_id'] in entity_map]
    unbatched = [x for x in completed if x['record_id'] not in batched]
    changed = False
    while len(unbatched) >= 100:
        batch_no = state['completed_entity_batches'] + 1
        batch, unbatched = (unbatched[:100], unbatched[100:])
        rng = random.Random(RANDOM_SEED + 700000 + batch_no)
        state['selected_ids'].extend((x['record_id'] for x in rng.sample(batch, 10)))
        state['entity_batched_ids'].extend((x['record_id'] for x in batch))
        state['completed_entity_batches'] = batch_no
        changed = True
    if not changed and (not state['selected_ids']):
        return
    state.update({'completed_entity_records': len(completed), 'sample_records': len(state['selected_ids']), 'sampling_rate': '实体阶段每完整100条固定抽10条并立即识别关系', 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    DYNAMIC_QA_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temp = DYNAMIC_QA_MANIFEST.with_suffix(DYNAMIC_QA_MANIFEST.suffix + '.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, DYNAMIC_QA_MANIFEST)
    run_dynamic_sample_relations(rows, entity_map, state['selected_ids'])
    relation_map = successful_map(RELATION_CACHE)
    export_dynamic_quality_txt(rows, entity_map, relation_map, state['selected_ids'])
    state.update({'completed_relation_records': len(relation_map), 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    temp = DYNAMIC_QA_MANIFEST.with_suffix(DYNAMIC_QA_MANIFEST.suffix + '.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, DYNAMIC_QA_MANIFEST)
    if changed:
        print(f"[动态质检-实体] 已完成{len(completed)}条，累计固定抽样{len(state['selected_ids'])}条。", flush=True)

def refresh_dynamic_quality_relations(rows: list[dict[str, Any]], entity_map: dict[str, dict[str, Any]], relation_map: dict[str, dict[str, Any]]) -> None:
    state = load_dynamic_quality_state()
    if not state.get('selected_ids'):
        return
    export_dynamic_quality_txt(rows, entity_map, relation_map, state['selected_ids'])
    state.update({'completed_entity_records': len(entity_map), 'completed_relation_records': len(relation_map), 'sample_records': len(state['selected_ids']), 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    temp = DYNAMIC_QA_MANIFEST.with_suffix(DYNAMIC_QA_MANIFEST.suffix + '.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, DYNAMIC_QA_MANIFEST)
    completed_selected = sum((rid in relation_map for rid in state['selected_ids']))
    print(f"[动态质检-关系] 固定样本关系已补全{completed_selected}/{len(state['selected_ids'])}条。", flush=True)

def extract_and_freeze_test300(full_rows: list[dict[str, Any]]) -> None:
    frozen = read_jsonl(TEST_PATH)
    test_ids = [x['record_id'] for x in frozen]
    if len(test_ids) != 300 or len(set(test_ids)) != 300:
        raise ValueError('独立测试集固定ID异常')
    full_map = {x['record_id']: x for x in full_rows}
    missing = [x for x in test_ids if x not in full_map]
    if missing:
        raise ValueError(f'全库预测缺少测试ID：{missing[:5]}')
    extracted = [full_map[x] for x in test_ids]
    write_jsonl_atomic(TEST_PREDICTION_PATH, extracted)
    manifest = {'source_full_prediction': str(FINAL_PATH), 'source_full_sha256': sha256_file(FINAL_PATH), 'frozen_id_file': str(TEST_PATH), 'frozen_id_sha256': sha256_file(TEST_PATH), 'test_prediction': str(TEST_PREDICTION_PATH), 'test_prediction_sha256': sha256_file(TEST_PREDICTION_PATH), 'records': 300, 'model': MODEL, 'prompt_version': PROMPT_VERSION, 'postprocess_version': POSTPROCESS_VERSION, 'note': '复现运行仅提取结构化机器预测；既有专家盲标Gold保持冻结且绝不覆盖。', 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    TEST_EXTRACTION_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + '\n'
    with WRITE_LOCK:
        with path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

def successful_map(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        if row.get('status') == 'success' and row.get('record_id'):
            result[row['record_id']] = row
    return result

def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub('^```(?:json)?\\s*', '', cleaned, flags=re.I)
    cleaned = re.sub('\\s*```$', '', cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = (cleaned.find('{'), cleaned.rfind('}'))
        if start < 0 or end <= start:
            raise ValueError('模型输出中未找到JSON对象')
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError('模型输出顶层必须为JSON对象')
    return value

def api_call(messages: list[dict[str, str]], seed_offset: int=0) -> tuple[dict[str, Any], str, dict[str, Any]]:
    payload = {'model': MODEL, 'messages': messages, 'temperature': TEMPERATURE, 'seed': RANDOM_SEED + seed_offset, 'response_format': {'type': 'json_object'}}
    encoded = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        request = urllib.request.Request(API_URL, data=encoded, headers={'Authorization': f'Bearer {API密钥.strip()}', 'Content-Type': 'application/json'}, method='POST')
        try:
            with API_SEMAPHORE:
                with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                    raw_response = response.read().decode('utf-8')
            envelope = json.loads(raw_response)
            content = envelope['choices'][0]['message']['content']
            return (extract_json(content), content, envelope.get('usage', {}))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode('utf-8', 'replace')
            try:
                error_code = str(json.loads(error_body).get('error', {}).get('code', ''))
            except json.JSONDecodeError:
                error_code = ''
            if exc.code == 400 and error_code == 'data_inspection_failed':
                raise DataInspectionError('模型服务拒绝完整输入：data_inspection_failed') from exc
            last_error = RuntimeError(f'HTTP {exc.code}: {error_code or error_body[:300]}')
            if attempt == MAX_RETRIES:
                break
            wait_seconds = min(30, 2 ** (attempt - 1) + random.random())
            time.sleep(wait_seconds)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_seconds = min(30, 2 ** (attempt - 1) + random.random())
            time.sleep(wait_seconds)
    raise RuntimeError(f'API调用在{MAX_RETRIES}次尝试后失败：{last_error}')

def build_lexicon_index(lexicon_items: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, str]]], dict[tuple[str, str], dict[str, Any]]]:
    surface_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    concept_index = {}
    for item in lexicon_items:
        concept_index[item['entity_type'], item['concept']] = item
        for surface in [item['concept'], *item.get('aliases', [])]:
            if surface:
                entry = {'surface': surface, 'normalized_concept': item['concept'], 'entity_type': item['entity_type']}
                if entry not in surface_index[surface]:
                    surface_index[surface].append(entry)
    return (dict(surface_index), concept_index)

def explicit_candidates(text: str, surface_index: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    candidates = []
    for surface in sorted(surface_index, key=lambda x: (-len(x), x)):
        start = 0
        while True:
            pos = text.find(surface, start)
            if pos < 0:
                break
            for entry in surface_index[surface]:
                candidates.append({'candidate_id': '', 'surface_form': surface, 'start': pos, 'end': pos + len(surface), 'normalized_concept': entry['normalized_concept'], 'entity_type': entry['entity_type'], 'evidence_text': surface, 'recognition_source': 'explicit_lexicon'})
            start = pos + 1
    unique = {(x['start'], x['end'], x['normalized_concept'], x['entity_type']): x for x in candidates}
    result = sorted(unique.values(), key=lambda x: (x['start'], -(x['end'] - x['start']), x['entity_type'], x['normalized_concept']))
    for idx, item in enumerate(result, 1):
        item['candidate_id'] = f'C{idx:03d}'
    return result

def compact_concept_catalog(items: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in items:
        grouped[item['entity_type']].append(item['concept'])
    return '\n'.join((f"{entity_type}：{'、'.join(concepts)}" for entity_type, concepts in grouped.items()))

def parse_gold() -> dict[str, dict[str, Any]]:
    text = GOLD_PATH.read_text(encoding='utf-8')
    sep = '=' * 80
    result = {}

    def section(block, a, b=None):
        p = re.escape(a) + '\\s*\\n(.*?)(?=\\n' + re.escape(b) + ')' if b else re.escape(a) + '\\s*\\n(.*)$'
        m = re.search(p, block, re.S)
        return m.group(1).strip() if m else ''

    def items(value, n):
        out = []
        for line in value.splitlines():
            if not line.strip() or line.strip() == '无':
                continue
            parts = [x.strip() for x in re.split('[|｜]', line)]
            if len(parts) != n or not all(parts):
                raise ValueError(f'Gold格式错误：{line}')
            out.append(parts)
        return out
    for part in text.split(sep):
        if not part.strip():
            continue
        block = part.strip()
        aid = re.search('(?m)^ann_id:\\s*(\\S+)', block)
        if not aid:
            continue
        result[aid.group(1)] = {'ann_id': aid.group(1), 'text': section(block, '【原文】', '【译文】'), 'entities': items(section(block, '【校对实体】', '【预标关系】'), 2), 'relations': items(section(block, '【校对关系】'), 3)}
    if len(result) != 100:
        raise ValueError(f'Gold条数错误：{len(result)}')
    return result

def select_examples(text: str) -> list[dict[str, Any]]:
    ids = []
    if re.search('止消渴|除消渴|主消渴|治消渴|解消渴', text):
        ids += ['ann_057', 'ann_060', 'ann_072', 'ann_088']
    if re.search('丸|散|汤|饮|膏|方', text):
        ids += ['ann_075', 'ann_078', 'ann_083', 'ann_095']
    if re.search('名曰|者|故曰', text):
        ids += ['ann_077', 'ann_091', 'ann_093', 'ann_094']
    if re.search('皆由|一原于|发为|转而为', text):
        ids += ['ann_080', 'ann_093', 'ann_099']
    if re.search('兼治|另治|宜|戒|忌', text):
        ids += ['ann_072', 'ann_085', 'ann_088']
    ids += ['ann_071', 'ann_076']
    unique = []
    for aid in ids:
        if aid in GOLD_EXAMPLES and aid not in unique:
            unique.append(aid)
    return [GOLD_EXAMPLES[x] for x in unique[:6]]

def entity_messages(row: dict[str, Any], candidates: list[dict[str, Any]], catalog: str) -> list[dict[str, str]]:
    system = '你负责中医古籍消渴专题的受控语义声明恢复，不得直接自由生成实体或三元组。\n先划分标题、主治、疾病定义、病机、组成、服法、并列/兼治、并发风险和注文，再恢复真实主语及局部作用域。\nclaims只允许6类：treatment、disease_manifestation、pathogenesis_cause、pathogenesis_location、symptom_indicates_pathogenesis、treatment_method。\nexcluded_claims用于记录并排除：formula_composition、formula_naming、parallel_indication、administration、complication_or_prognosis、commentary_or_physiology、lexical_gloss。\n治疗主语role只允许entry_title、prescription_title、named_formula、single_herb；component_herb、administration_vehicle、action不能进入treatment。\n疾病使用核心最小mention：泄利消渴、男子消渴、消渴之人只写消渴。方名剥离“名、一名”；治消渴方可为prescription_title，后置玉壶丸为named_formula。\n无“名曰、某病者、其证、故为”等定义/证候统摄标记时，相邻内容默认并列主治。组成药不继承整方治疗；并发风险不改写为疾病表现；解释其他功效的括注不回连消渴。\n本草条目若以单味药、本草品或药用食物名称起首，后续同一条目明确“主/治/止/除消渴”，条首名称是中药实体，head_type必须写“中药”，head_role写“entry_title”。标题与主治被句号分开时，evidence_unit_ids必须同时包含标题单元和主治单元。不得把entry_title或single_herb写入head_type。\n若正文明确给出承接条目标题的完整具体药物名称（如“白雄鸡肉……消渴”），以治疗谓词实际支配的连续原文主语为head_surface，同时把条目标题单元纳入证据。单独的“根、叶、汁、肉”等只是回指药用部位，不是可独立识别的中药名，此时必须回指条目标题作为head_surface；不得把标题误记为formula_naming。\n普通surface必须逐字来自原文。领域词表是主题锚定与规范化参照，不是封闭白名单。输出严格JSON。'
    user = f"""【原文】\n{row['text']}\n【证据单元】\n{json.dumps(evidence_units(row['text']), ensure_ascii=False)}\n【词表显式候选】\n{json.dumps(candidates, ensure_ascii=False)}\n【同场景专家正反例】\n{json.dumps(select_examples(row['text']), ensure_ascii=False)}\n【允许的唯一映射】\ntreatment=中药/方药—治疗—疾病；disease_manifestation=疾病—表现—症状；pathogenesis_cause=病机—导致—疾病；pathogenesis_location=病机—涉及—病位；symptom_indicates_pathogenesis=症状—提示—病机；treatment_method=治法—主治—疾病。\n任何其他因果方向、类型组合和方剂组成都不得进入claims；只能记入excluded_claims。evidence_unit_ids只能复制上方已给出的ID。\n疾病词表候选若字符区间互相重叠，必须判断哪些是真正独立病名。不得在“热中消渴、伤中消渴”中跨界识别“中消”，不得在“消渴病”中跨界识别“渴病”；真正的“中消者、为中消、上消中消下消”仍须保留。explicit_disease_mentions列出原文中所有经语义确认的独立疾病surface；无法消歧时不要猜测。\n输出：{{"topic_relevant":true,"scene_types":["prescription_treatment"],"main_disease":"消渴","explicit_disease_mentions":["消渴"],\n"claims":[{{"claim_type":"treatment","head_surface":"治消渴方","head_normalized":"治消渴方","head_type":"方药","head_role":"prescription_title","tail_surface":"消渴","tail_normalized":"消渴","tail_type":"疾病","evidence_unit_ids":["U001"],"scope_reason":"直接治疗谓词"}}],\n"excluded_claims":[{{"claim_type":"formula_composition","surface_forms":["人参"],"evidence_unit_ids":["U002"],"reason":"组成药不继承治疗"}}],\n"core_entities":[{{"surface_form":"消渴","normalized_concept":"消渴","entity_type":"疾病","evidence_unit_ids":["U001"],"reason":"专题核心但当前无合法关系"}}],\n"scope_audit":"排除依据","translation":"忠实简洁的现代汉语译文"}}。\nformula_naming放excluded_claims；若后置正式方名与当前处方共同治疗消渴，另为该方名输出treatment claim。无合法claim时只允许core_entities保留消渴疾病。"""
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

def locate_surface(text: str, surface: str, evidence: str='') -> tuple[int, int]:
    if not surface:
        return (-1, -1)
    if evidence and evidence in text and (surface in evidence):
        evidence_start = text.find(evidence)
        return (evidence_start + evidence.find(surface), evidence_start + evidence.find(surface) + len(surface))
    start = text.find(surface)
    return (start, start + len(surface)) if start >= 0 else (-1, -1)

def repair_surface_alignment(row: dict[str, Any], response: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    targets = []
    for idx, claim in enumerate(response.get('claims', []) if isinstance(response.get('claims', []), list) else []):
        for side in ('head', 'tail'):
            surface = str(claim.get(f'{side}_surface', '')).strip()
            if surface and surface not in row['text']:
                targets.append({'target_id': f'claim_{idx}_{side}', 'surface_form': surface, 'normalized_concept': str(claim.get(f'{side}_normalized', '')), 'entity_type': str(claim.get(f'{side}_type', '')), 'evidence_unit_ids': claim.get('evidence_unit_ids', [])})
    for idx, entity in enumerate(response.get('core_entities', []) if isinstance(response.get('core_entities', []), list) else []):
        surface = str(entity.get('surface_form', '')).strip()
        if surface and surface not in row['text']:
            targets.append({'target_id': f'core_{idx}', 'surface_form': surface, 'normalized_concept': str(entity.get('normalized_concept', '')), 'entity_type': str(entity.get('entity_type', '')), 'evidence_unit_ids': entity.get('evidence_unit_ids', [])})
    if not targets:
        return (response, {})
    messages = [{'role': 'system', 'content': '你只做原文证据表面形式对齐，不重新识别实体。每个target的规范概念、实体类型和语义判断已冻结，不得修改。\n你必须从当前原文中复制与该规范概念对应的连续原文字符串；不得输出原文不存在的规范名、改写文字或拼接跨段文字。\n例如原文为“消中”、规范概念为“消渴”，surface_form必须返回“消中”。若原文只有“舌瘅渴数饮、多渴引饮、饮一溲二者”等临床表现而无明确疾病名，疾病target必须返回null，不得把整个表现当作疾病surface。\n无法找到可对齐表达时surface_form返回null。只输出严格JSON。'}, {'role': 'user', 'content': f"""【原文】\n{row['text']}\n【证据单元】\n{json.dumps(evidence_units(row['text']), ensure_ascii=False)}\n【待对齐项】\n{json.dumps(targets, ensure_ascii=False)}\n输出：{{"repairs":[{{"target_id":"claim_0_tail","surface_form":"消中"}}]}}"""}]
    value, _, usage = api_call(messages, seed_offset=500000 + int(row['record_id'].split('_')[-1]))
    repairs = value.get('repairs', [])
    if not isinstance(repairs, list):
        raise ValueError('表面形式对齐修复返回格式错误')
    requested = {x['target_id'] for x in targets}
    fixed = {}
    for item in repairs:
        if not isinstance(item, dict):
            continue
        target_id = str(item.get('target_id', ''))
        surface = item.get('surface_form')
        if target_id not in requested or not isinstance(surface, str) or (not surface.strip()):
            continue
        surface = surface.strip()
        if surface not in row['text']:
            continue
        fixed[target_id] = surface
    repaired = json.loads(json.dumps(response, ensure_ascii=False))
    for target_id, surface in fixed.items():
        parts = target_id.split('_')
        if parts[0] == 'claim':
            repaired['claims'][int(parts[1])][f'{parts[2]}_surface'] = surface
        else:
            repaired['core_entities'][int(parts[1])]['surface_form'] = surface
    repaired['_surface_alignment_repairs'] = fixed
    return (repaired, {k: v for k, v in usage.items() if isinstance(v, int)})

def validate_and_fuse_entities(row: dict[str, Any], candidates: list[dict[str, Any]], response: dict[str, Any], concept_index: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings = []
    entities = []
    hypotheses = []
    units = {x['evidence_unit_id']: x for x in evidence_units(row['text'])}
    claims = response.get('claims', [])
    core = response.get('core_entities', [])
    excluded = response.get('excluded_claims', [])
    if not isinstance(claims, list) or not isinstance(core, list) or (not isinstance(excluded, list)):
        raise ValueError('claims、core_entities和excluded_claims必须为数组')
    explicit_disease_mentions = response.get('explicit_disease_mentions', [])
    if not isinstance(explicit_disease_mentions, list) or any((not isinstance(x, str) for x in explicit_disease_mentions)):
        raise ValueError('explicit_disease_mentions必须为原文疾病surface字符串数组')
    explicit_disease_mentions = {x.strip() for x in explicit_disease_mentions if x.strip()}
    invalid_auxiliary_mentions = {x for x in explicit_disease_mentions if x not in row['text']}
    if invalid_auxiliary_mentions:
        warnings.append('删除原文不存在的辅助病名规范项：' + '、'.join(sorted(invalid_auxiliary_mentions)))
        explicit_disease_mentions -= invalid_auxiliary_mentions
    disease_candidates = [c for c in candidates if c['entity_type'] == '疾病' and c['surface_form'] not in TOPIC_STANDALONE_SYMPTOMS]
    overlapping_disease_keys = set()
    for i, left in enumerate(disease_candidates):
        for right in disease_candidates[i + 1:]:
            if left['start'] < right['end'] and right['start'] < left['end'] and ((left['start'], left['end']) != (right['start'], right['end'])):
                overlapping_disease_keys.update({(left['start'], left['end'], left['surface_form']), (right['start'], right['end'], right['surface_form'])})
    if '三消渴' in row['text']:
        explicit_disease_mentions.add('三消')
        for claim in claims:
            evidence_text = ''.join((units[uid]['text'] for uid in claim.get('evidence_unit_ids', []) if uid in units))
            if '三消渴' not in evidence_text:
                continue
            for side in ('head', 'tail'):
                if str(claim.get(f'{side}_type', '')).strip() == '疾病' and str(claim.get(f'{side}_surface', '')).strip() == '消渴':
                    claim[f'{side}_surface'] = '三消'
                    claim[f'{side}_normalized'] = '三消'
                    warnings.append('按冻结口径将‘三消渴’中的重叠消渴端点解析为三消')
    if overlapping_disease_keys and 'explicit_disease_mentions' not in response:
        raise ValueError('存在重叠疾病候选，模型必须返回explicit_disease_mentions完成语义消歧')
    if response.get('topic_relevant') is not True:
        if claims or core:
            raise ValueError('非专题条目不得输出命题或实体')
        return ([], warnings)

    def canonical_surface(surface, concept, etype):
        surface = str(surface).strip()
        concept = str(concept).strip() or surface
        etype = str(etype).strip()
        if etype == '疾病' and '消渴' in surface and ((etype, surface) not in concept_index):
            surface = '消渴'
            concept = '消渴'
        elif etype == '疾病' and concept and (concept in surface) and (concept in row['text']):
            surface = concept
        if etype == '方药' and re.match('^(?:一名|名|号)', surface):
            stripped = re.sub('^(?:一名|名|号)', '', surface).strip()
            if stripped and stripped in row['text']:
                surface = stripped
        return surface

    def is_explicit_disease_surface(surface):
        surface = str(surface).strip()
        if surface == DIABETES_ALIAS:
            return True
        if surface in TOPIC_STANDALONE_SYMPTOMS:
            return False
        matching = [c for c in disease_candidates if c['surface_form'] == surface]
        if not matching:
            return False
        if any(((c['start'], c['end'], c['surface_form']) in overlapping_disease_keys for c in matching)):
            return surface in explicit_disease_mentions
        return True

    def unit_ids_for_span(start, end):
        return [uid for uid, u in units.items() if u['start'] <= start < end <= u['end']]

    def first_special_symptom(value):
        value = str(value).strip()
        hits = [x for x in TOPIC_STANDALONE_SYMPTOMS if x in value and x in row['text']]
        return max(hits, key=len) if hits else ''

    def add(surface, concept, etype, unit_ids, reason, source, preferred_span=None):
        surface = str(surface).strip()
        concept = str(concept).strip() or surface
        etype = str(etype).strip()
        if etype not in ENTITY_TYPES:
            raise ValueError(f'非法实体类型：{etype}')
        selected = [units[x] for x in unit_ids if x in units]
        if not selected or len(selected) != len(unit_ids):
            anchors = []
            pos = row['text'].find(surface)
            if pos >= 0:
                anchors = [(pos, pos + len(surface))]
            else:
                anchors = [(c['start'], c['end']) for c in candidates if c['entity_type'] == etype]
            selected = [u for u in units.values() if any((u['start'] <= a < b <= u['end'] for a, b in anchors))]
            if not selected:
                raise ValueError('实体证据单元非法且无法根据原文锚点恢复')
            warnings.append(f'根据原文锚点恢复证据单元：{surface or concept}')
        if surface not in row['text']:
            ev_start = min((x['start'] for x in selected))
            ev_end = max((x['end'] for x in selected))
            repairs = [c for c in candidates if c['entity_type'] == etype and ev_start <= c['start'] < c['end'] <= ev_end]
            if len(repairs) == 1:
                surface = repairs[0]['surface_form']
                if concept == '消渴' and etype == '疾病':
                    concept = repairs[0]['normalized_concept']
                warnings.append(f'依证据单元恢复原文表面形式：{surface}')
        surface = canonical_surface(surface, concept, etype)
        if etype == '疾病' and surface == '消渴':
            concept = '消渴'
        if preferred_span and len(preferred_span) == 2 and (row['text'][preferred_span[0]:preferred_span[1]] == surface):
            start, end = preferred_span
        else:
            start, end = locate_surface(row['text'], surface)
        alignment = 'exact'
        if start < 0:
            ev = row['text'][min((x['start'] for x in selected)):max((x['end'] for x in selected))]
            compact = re.sub('（[^）]*）|\\([^)]*\\)|〔[^〕]*〕|\\s+', '', ev)
            if concept not in compact:
                raise ValueError(f'实体无法定位：{surface}/{concept}')
            start = min((x['start'] for x in selected))
            end = max((x['end'] for x in selected))
            surface = concept
            alignment = 'ocr_interrupted'
        entities.append({'surface_form': surface, 'start': start, 'end': end, 'normalized_concept': concept, 'entity_type': etype, 'evidence_text': row['text'][min((x['start'] for x in selected)):max((x['end'] for x in selected))], 'recognition_source': source, 'alignment_status': alignment, 'relevance_reason': reason})
    allowed_excluded = {'formula_composition', 'formula_naming', 'parallel_indication', 'administration', 'complication_or_prognosis', 'commentary_or_physiology', 'lexical_gloss'}
    for item in excluded:
        if not isinstance(item, dict) or str(item.get('claim_type', '')) not in allowed_excluded:
            warnings.append(f'忽略非标准排除审计项：{item}')
    generic_formula_heads = []
    for p in claims:
        hs = str(p.get('head_surface', '')).strip()
        if str(p.get('claim_type', '')) == 'treatment' and str(p.get('head_type', '')) == '方药' and re.fullmatch('治[^。；\\n]{1,12}方', hs):
            generic_formula_heads.append(hs)
    first_generic = min(generic_formula_heads, key=lambda x: row['text'].find(x)) if generic_formula_heads else ''

    def claim_evidence(p):
        ids = p.get('evidence_unit_ids', [])
        chosen = [units[x] for x in ids if x in units]
        return row['text'][min((x['start'] for x in chosen)):max((x['end'] for x in chosen))] if chosen else ''

    def topic_disease(surface, concept):
        surface = str(surface).strip()
        concept = str(concept).strip()
        if '消渴' in surface or '消渴' in concept:
            return True
        if ('疾病', concept) in concept_index:
            return True
        return any((c['entity_type'] == '疾病' and c['surface_form'] == surface for c in candidates))
    anchor_nodes = set()
    for p in claims:
        ctype = str(p.get('claim_type', '')).strip()
        hs = str(p.get('head_surface', '')).strip()
        ts = str(p.get('tail_surface', '')).strip()
        hc = str(p.get('head_normalized', '')).strip()
        tc = str(p.get('tail_normalized', '')).strip()
        if ctype in {'treatment', 'treatment_method', 'pathogenesis_cause'} and topic_disease(ts, tc):
            anchor_nodes.update({(hs, str(p.get('head_type', ''))), (ts, str(p.get('tail_type', '')))})
        elif ctype == 'disease_manifestation' and topic_disease(hs, hc):
            ev = claim_evidence(p)
            same_unit = any((hs in u['text'] and ts in u['text'] for u in units.values()))
            definition = bool(re.search('(?:其证|症见|见证|表现|者|名曰|故曰)', ev))
            enumerated = bool(re.search('(?:以致|致生|发为)[^。；\\n]{0,40}' + re.escape(hs) + '、[^。；\\n]*' + re.escape(ts), ev))
            if hs in ev and ts in ev and (same_unit or definition) and (not enumerated):
                anchor_nodes.update({(hs, '疾病'), (ts, '症状')})
    for p in claims:
        claim_type = str(p.get('claim_type', '')).strip()
        if claim_type not in CLAIM_RELATIONS:
            warnings.append(f'舍弃Schema外语义命题：{claim_type}')
            continue
        raw_head_surface = str(p.get('head_surface', '')).strip()
        raw_tail_surface = str(p.get('tail_surface', '')).strip()
        raw_head_type = str(p.get('head_type', '')).strip()
        raw_tail_type = str(p.get('tail_type', '')).strip()
        head_role = str(p.get('head_role', '')).strip()
        generic_parts = {'根', '叶', '汁', '肉', '实', '花'}
        title_surface = re.split('[。〈〔]', row['text'], maxsplit=1)[0].strip()
        title_base = re.sub('（[^）]*）|\\([^)]*\\)', '', title_surface).strip()
        bare_part_anaphor = raw_head_surface in generic_parts and head_role == 'anaphor'
        synthesized_part_name = raw_head_surface not in row['text'] and any((raw_head_surface.endswith(part) and raw_head_surface[:-len(part)] == title_base for part in generic_parts))
        if claim_type == 'treatment' and (bare_part_anaphor or synthesized_part_name):
            if title_surface and title_surface in row['text'] and (len(title_surface) <= 20):
                raw_head_surface = title_surface
                raw_head_type = '中药'
                p['head_surface'] = title_surface
                p['head_normalized'] = title_surface
                p['head_type'] = '中药'
                p['head_role'] = 'entry_title'
                warnings.append(f'裸药用部位回指本草条目标题：{title_surface}')
        if claim_type == 'treatment' and (raw_head_type == 'single_herb' or (raw_head_type == '方药' and str(p.get('head_role', '')).strip() == 'single_herb')):
            old_head_type = raw_head_type
            raw_head_type = '中药'
            p['head_type'] = '中药'
            warnings.append(f'纠正单味药角色与实体类型冲突：{raw_head_surface}|{old_head_type}→中药')
        special_head = first_special_symptom(raw_head_surface)
        special_tail = first_special_symptom(raw_tail_surface)
        if special_head or special_tail:
            if claim_type == 'treatment' and raw_head_type == '方药' and special_tail:
                ids = p.get('evidence_unit_ids', [])
                reason = str(p.get('scope_reason', '')).strip()
                add(raw_head_surface, p.get('head_normalized', ''), '方药', ids, reason or '原文方药直接指向指定消渴专题症状', 'topic_symptom_subject')
            warnings.append(f'指定专题症状不作为疾病关系端点：{special_head or special_tail}')
            continue
        sig = (raw_head_type, CLAIM_RELATIONS[claim_type], raw_tail_type)
        if sig not in RELATION_SIGNATURES:
            warnings.append(f'舍弃类型签名不合法的命题：{claim_type}/{sig}')
            continue
        head_surface = str(p.get('head_surface', '')).strip()
        tail_surface = str(p.get('tail_surface', '')).strip()
        tail_concept = str(p.get('tail_normalized', '')).strip()
        if sig[0] == '疾病' and (not is_explicit_disease_surface(canonical_surface(head_surface, p.get('head_normalized', ''), '疾病'))):
            warnings.append(f'舍弃无明确病名mention的疾病端点：{head_surface}')
            continue
        if sig[2] == '疾病' and (not is_explicit_disease_surface(canonical_surface(tail_surface, tail_concept, '疾病'))):
            warnings.append(f'舍弃无明确病名mention的疾病端点：{tail_surface}')
            continue
        if head_surface not in row['text'] or tail_surface not in row['text']:
            warnings.append(f'舍弃仍无法逐字对齐的命题：{head_surface}-{tail_surface}')
            continue
        if claim_type in {'treatment', 'treatment_method', 'pathogenesis_cause'} and (not topic_disease(tail_surface, tail_concept)):
            warnings.append(f'舍弃非消渴专题尾实体：{tail_surface}')
            continue
        if claim_type in {'treatment', 'treatment_method', 'pathogenesis_cause'}:
            ev = claim_evidence(p)
            if head_surface not in ev or tail_surface not in ev:
                named_inheritance = claim_type == 'treatment' and str(p.get('head_role', '')) == 'named_formula' and (head_surface in row['text']) and (tail_surface in row['text'])
                first_unit = next(iter(units.values()), {})
                entry_title_inheritance = claim_type == 'treatment' and str(p.get('head_role', '')) == 'entry_title' and (head_surface in str(first_unit.get('text', ''))) and (head_surface in row['text']) and (tail_surface in row['text']) and (raw_head_type == '中药')
                if named_inheritance or entry_title_inheritance:
                    repaired = [u['evidence_unit_id'] for u in units.values() if head_surface in u['text'] or tail_surface in u['text']]
                    p['evidence_unit_ids'] = repaired
                    if entry_title_inheritance:
                        warnings.append(f'本草条目标题受控继承主治并联合证据：{head_surface}-{tail_surface}')
                    else:
                        warnings.append(f'后置正式方名继承当前处方主治：{head_surface}-{tail_surface}')
                else:
                    warnings.append(f'舍弃证据不同时覆盖两端的直接命题：{head_surface}-{tail_surface}')
                    continue
        if claim_type == 'disease_manifestation':
            ev = claim_evidence(p)
            same_unit = any((head_surface in u['text'] and tail_surface in u['text'] for u in units.values()))
            definition = bool(re.search('(?:其证|症见|见证|表现|者|名曰|故曰)', ev))
            enumerated_outcomes = bool(re.search('(?:以致|致生|发为)[^。；\\n]{0,40}' + re.escape(head_surface) + '、[^。；\\n]*' + re.escape(tail_surface), ev))
            if not topic_disease(head_surface, p.get('head_normalized', '')) or head_surface not in ev or tail_surface not in ev or (not same_unit and (not definition)) or enumerated_outcomes:
                warnings.append(f'舍弃未被消渴疾病局部统摄的伪表现：{tail_surface}')
                continue
        if claim_type == 'pathogenesis_location' and (head_surface, '病机') not in anchor_nodes:
            warnings.append(f'舍弃未连通消渴核心命题的病位声明：{head_surface}-{tail_surface}')
            continue
        if claim_type == 'symptom_indicates_pathogenesis' and (not ((head_surface, '症状') in anchor_nodes or (tail_surface, '病机') in anchor_nodes)):
            warnings.append(f'舍弃未连通消渴核心命题的辨机声明：{head_surface}-{tail_surface}')
            continue
        if claim_type == 'treatment':
            role = str(p.get('head_role', '')).strip()
            if role not in ALLOWED_TREATMENT_ROLES:
                warnings.append(f"舍弃非治疗主体：{head_surface}({role or '未标角色'})")
                continue
            if sig[0] == '方药':
                named = bool(re.fullmatch('[^，。；\\n]{1,12}(?:方|丸|汤|散|饮|膏|丹|煎|剂)', head_surface))
                lexicon_named = any((c['entity_type'] == '方药' and c['surface_form'] == head_surface for c in candidates))
                if not (named or lexicon_named):
                    ids = p.get('evidence_unit_ids', [])
                    reason = str(p.get('scope_reason', '')).strip()
                    add(p.get('tail_surface', ''), p.get('tail_normalized', ''), '疾病', ids, reason or '消渴病名明示但治疗主语无法实例化', 'isolated_core')
                    warnings.append(f'舍弃无法实例化的省略治疗主语：{head_surface}')
                    continue
        if sig[0] == '方药' and re.fullmatch('治[^。；\\n]{1,12}方', head_surface) and first_generic and (head_surface != first_generic):
            warnings.append(f'舍弃同一处方重复适应证标题：{head_surface}')
            continue
        if sig[1] == '治疗' and sig[0] in {'方药', '中药'}:
            delivery = re.search(re.escape(head_surface) + '\\s*(?:送服|调下|下)', row['text'])
            direct = re.search(re.escape(head_surface) + '[^。；\\n]{0,8}(?:主治|治|主|止|除|解)', row['text'])
            if delivery and (not direct):
                warnings.append(f'舍弃服法药引伪治疗主体：{head_surface}')
                continue
            if sig[0] == '中药':
                component_context = re.search('(?:用|取)[^。；\\n]{0,80}' + re.escape(head_surface) + '[^。；\\n]{0,40}(?:等分|各|为末|炼蜜|丸)', row['text'])
                if component_context and (not direct):
                    warnings.append(f'舍弃方剂组成药继承治疗：{head_surface}')
                    continue
        if sig == ('疾病', '表现', '症状'):
            disease_core = str(p.get('head_normalized', '')).strip() or head_surface
            dpos = row['text'].find(disease_core)
            spos = row['text'].find(tail_surface)
            definition = bool(re.search('(?:名曰|故曰|谓之|表现为|其证|者)[^。；\\n]{0,40}' + re.escape(disease_core), row['text']))
            if spos >= 0 and dpos >= 0 and (spos < dpos) and (not definition):
                warnings.append(f'舍弃疾病前并列适应证伪表现：{tail_surface}')
                continue
            if re.search('(?:预防|须防|思虑|专虑|常虑)[^。；\\n]{0,20}(?:痈|疽)', row['text']) and re.search('痈|疽', tail_surface):
                warnings.append(f'舍弃并发风险伪表现：{tail_surface}')
                continue
        ids = p.get('evidence_unit_ids', [])
        reason = str(p.get('scope_reason', '')).strip()
        add(p.get('head_surface', ''), p.get('head_normalized', ''), sig[0], ids, reason, 'semantic_claim_endpoint')
        add(p.get('tail_surface', ''), p.get('tail_normalized', ''), sig[2], ids, reason, 'semantic_claim_endpoint')
        hypotheses.append({'head_surface': canonical_surface(p.get('head_surface', ''), p.get('head_normalized', ''), sig[0]), 'head_type': sig[0], 'relation_type': sig[1], 'tail_surface': canonical_surface(p.get('tail_surface', ''), p.get('tail_normalized', ''), sig[2]), 'tail_type': sig[2], 'evidence_unit_ids': ids, 'reason': reason, 'claim_type': claim_type, 'head_role': str(p.get('head_role', '')).strip()})
    for e in core:
        surface = str(e.get('surface_form', '')).strip()
        etype = str(e.get('entity_type', '')).strip()
        special = first_special_symptom(surface)
        if special:
            pos = row['text'].find(special)
            add(special, special, '症状', unit_ids_for_span(pos, pos + len(special)), str(e.get('reason', '')).strip(), 'expert_fixed_topic_symptom')
            continue
        if etype == '疾病' and (not is_explicit_disease_surface(canonical_surface(surface, e.get('normalized_concept', ''), etype))):
            warnings.append(f'舍弃无明确病名mention的孤立疾病：{surface}')
            continue
        if surface not in row['text']:
            warnings.append(f'舍弃仍无法逐字对齐的孤立实体：{surface}')
            continue
        if etype in {'症状', '中药', '病位'}:
            warnings.append(f'舍弃非指定类型的无命题孤立实体：{surface}|{etype}')
            continue
        add(surface, e.get('normalized_concept', ''), etype, e.get('evidence_unit_ids', []), str(e.get('reason', '')).strip(), 'isolated_core')
    for symptom in sorted(TOPIC_STANDALONE_SYMPTOMS, key=lambda x: (-len(x), x)):
        pos = row['text'].find(symptom)
        if pos >= 0:
            add(symptom, symptom, '症状', unit_ids_for_span(pos, pos + len(symptom)), '专家固定的消渴专题症状', 'expert_fixed_topic_symptom')
    diabetes_pos = row['text'].find(DIABETES_ALIAS)
    if diabetes_pos >= 0:
        add(DIABETES_ALIAS, '消渴', '疾病', unit_ids_for_span(diabetes_pos, diabetes_pos + len(DIABETES_ALIAS)), '专家确认的消渴变名', 'expert_fixed_alias')
    seen_disease_surfaces = set()
    for c in candidates:
        if c['entity_type'] != '疾病' or c['surface_form'] in TOPIC_STANDALONE_SYMPTOMS:
            continue
        surface = c['surface_form']
        end = c['end']
        if surface in seen_disease_surfaces:
            continue
        if end < len(row['text']) and row['text'][end] in '丸汤散饮方膏丹论篇门法':
            continue
        if surface == '消渴' and c['start'] > 0 and (row['text'][c['start'] - 1:c['end']] == '三消渴'):
            warnings.append(f"按冻结口径舍弃‘三消渴’中的重叠消渴@{c['start']}:{c['end']}")
            continue
        if (c['start'], c['end'], surface) in overlapping_disease_keys and surface not in explicit_disease_mentions:
            warnings.append(f"舍弃未获语义确认的重叠疾病候选：{surface}@{c['start']}:{c['end']}")
            continue
        ids = unit_ids_for_span(c['start'], c['end'])
        if ids:
            add(surface, c['normalized_concept'], '疾病', ids, '同条明示病名分别保留', 'explicit_disease_mention', preferred_span=(c['start'], c['end']))
            seen_disease_surfaces.add(surface)
    entities = [e for e in entities if not (e['entity_type'] == '疾病' and e['surface_form'] == '消渴' and (e['start'] > 0) and (row['text'][e['start'] - 1:e['end']] == '三消渴'))]
    fused: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for item in entities:
        key = (item['start'], item['end'], item['normalized_concept'], item['entity_type'])
        if key in fused:
            fused[key]['recognition_source'] = 'explicit_implicit_fused'
        else:
            fused[key] = item
    result = sorted(fused.values(), key=lambda x: (x['start'], x['end'], x['entity_type'], x['normalized_concept']))
    disease_entities = [x for x in result if x['entity_type'] == '疾病']
    for i, left in enumerate(disease_entities):
        for right in disease_entities[i + 1:]:
            if left['start'] < right['end'] and right['start'] < left['end'] and ((left['start'], left['end']) != (right['start'], right['end'])):
                raise ValueError(f"疾病实体发生未消解字符重叠：{left['surface_form']} / {right['surface_form']}")
    for idx, item in enumerate(result, 1):
        item['entity_id'] = f"{row['record_id']}_E{idx:03d}"
        item.pop('candidate_id', None)
    response['_validated_hypotheses'] = hypotheses
    response['_validated_exclusions'] = excluded
    return (result, warnings)

def relation_candidates(entities: list[dict[str, Any]], schemas: list[dict[str, str]], hypotheses: list[dict[str, Any]] | None=None) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        by_type[entity['entity_type']].append(entity)
    result = []
    for schema in schemas:
        for head in by_type[schema['head_type']]:
            for tail in by_type[schema['tail_type']]:
                item = {'candidate_relation_id': f'R{len(result) + 1:03d}', 'head_entity_id': head['entity_id'], 'head_surface': head['surface_form'], 'head_type': head['entity_type'], 'relation_type': schema['relation'], 'tail_entity_id': tail['entity_id'], 'tail_surface': tail['surface_form'], 'tail_type': tail['entity_type']}
                item['stage1_hypothesis'] = any((x['head_surface'] == head['surface_form'] and x['head_type'] == head['entity_type'] and (x['relation_type'] == schema['relation']) and (x['tail_surface'] == tail['surface_form']) and (x['tail_type'] == tail['entity_type']) for x in hypotheses or []))
                result.append(item)
    return result

def evidence_units(text: str) -> list[dict[str, Any]]:
    units = []
    for match in re.finditer('[^。！？；\\n]+[。！？；]?|\\n', text):
        value = match.group(0)
        if not value.strip():
            continue
        units.append({'evidence_unit_id': f'U{len(units) + 1:03d}', 'start': match.start(), 'end': match.end(), 'text': value})
    if not units and text:
        units.append({'evidence_unit_id': 'U001', 'start': 0, 'end': len(text), 'text': text})
    return units

def relation_messages(row: dict[str, Any], entities: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = '你是中医古籍消渴专题关系标注专家。实体集合已经冻结，你只能判断给定候选关系在当前条文中是否实际成立。\n必须遵守：\n1. Schema类型兼容只表示关系可能存在，不表示当前文本已经表达。\n2. 只接受当前条文有直接语义证据支持的关系，不使用文本外医学常识补关系。\n3. 不得新增、删除、改名或改变任何实体类型。\n4. 方剂组成不在现行关系Schema中；不保留因处方组成而出现的中药，不判断任何方剂组成关系。\n5. 每个候选都必须给出成立/不成立判断；成立时只返回支持关系的证据单元ID，不得重新抄写证据文字。\n6. 必须先区分疾病证候描述和并列主治；只有确认被疾病统摄的症状才能建立表现关系。\n7. 明确的并列病因可以连接共同上位疾病；不得将共同病机无依据地只连接某一下位分型。\n8. 同义病名不机械复制关系；辨经、兼治、另治等分句不得回连前文疾病。\n9. 输出严格JSON，不要附加说明。'
    user = f"""【原文】\n{row['text']}\n\n【冻结实体】\n{json.dumps(entities, ensure_ascii=False)}\n\n【Schema兼容候选关系】\n{json.dumps(candidates, ensure_ascii=False)}\n\n【原文证据单元】\n{json.dumps(evidence_units(row['text']), ensure_ascii=False)}\n\n输出：\n{{"relation_decisions":[{{"candidate_relation_id":"R001","accept":true,"evidence_unit_ids":["U001"],"reason":"简短语义判断"}}]}}\n所有候选ID必须且只能出现一次。"""
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

def minimal_relation_messages(row: dict[str, Any], entities: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    unit_list = evidence_units(row['text'])
    compact_entities = [{key: entity[key] for key in ('entity_id', 'surface_form', 'normalized_concept', 'entity_type', 'start', 'end')} for entity in entities]
    snippets = []
    for unit in unit_list:
        anchors = [entity for entity in entities if unit['start'] <= entity['start'] < entity['end'] <= unit['end']]
        if not anchors:
            continue
        local_start = max(unit['start'], min((x['start'] for x in anchors)) - 12)
        local_end = min(unit['end'], max((x['end'] for x in anchors)) + 12)
        snippets.append({'evidence_unit_id': unit['evidence_unit_id'], 'start': local_start, 'end': local_end, 'text': row['text'][local_start:local_end]})
    system = '你是中医古籍消渴专题关系标注专家。完整原文因模型服务内容审核无法提交，当前仅提供覆盖冻结实体的原文最小窗口。只能判断给定候选关系是否由这些逐字原文片段直接支持。不得使用文本外知识，不得改变实体或Schema。每个候选ID必须且只能判断一次；成立时只返回所给的原始证据单元ID，且所选证据单元组合必须同时覆盖头实体和尾实体。输出严格JSON。'
    user = f'【冻结实体】\n{json.dumps(compact_entities, ensure_ascii=False)}\n\n【Schema兼容候选关系】\n{json.dumps(candidates, ensure_ascii=False)}\n\n【覆盖关系端点的最小原文窗口】\n{json.dumps(snippets, ensure_ascii=False)}\n\n输出：\n{{"relation_decisions":[{{"candidate_relation_id":"R001","accept":true,"evidence_unit_ids":["U001"],"reason":"简短语义判断"}}]}}\n所有候选ID必须且只能出现一次。'
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

def validate_relations(row: dict[str, Any], entities: list[dict[str, Any]], candidates: list[dict[str, Any]], response: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = response.get('relation_decisions', [])
    if not isinstance(decisions, list):
        raise ValueError('relation_decisions必须为数组')
    decision_map = {str(x.get('candidate_relation_id', '')): x for x in decisions if isinstance(x, dict)}
    expected = {x['candidate_relation_id'] for x in candidates}
    if set(decision_map) != expected:
        raise ValueError(f'关系候选决策不完整；missing={sorted(expected - set(decision_map))}, extra={sorted(set(decision_map) - expected)}')
    relations = []
    entity_map = {entity['entity_id']: entity for entity in entities}
    unit_map = {unit['evidence_unit_id']: unit for unit in evidence_units(row['text'])}
    for candidate in candidates:
        decision = decision_map[candidate['candidate_relation_id']]
        if decision.get('accept') is not True:
            continue
        selected_unit_ids = decision.get('evidence_unit_ids', [])
        evidence = ''
        evidence_source = ''
        if isinstance(selected_unit_ids, list) and selected_unit_ids:
            invalid_units = [unit_id for unit_id in selected_unit_ids if unit_id not in unit_map]
            if invalid_units:
                raise ValueError(f"成立关系{candidate['candidate_relation_id']}返回非法证据单元ID：{invalid_units}")
            selected_units = [unit_map[unit_id] for unit_id in selected_unit_ids]
            evidence_start = min((unit['start'] for unit in selected_units))
            evidence_end = max((unit['end'] for unit in selected_units))
            evidence = row['text'][evidence_start:evidence_end]
            evidence_source = 'model_selected_unit_python_extracted'
        else:
            evidence = str(decision.get('evidence_text', '')).strip()
            evidence_source = 'llm_exact_quote'
        if not evidence or evidence not in row['text']:
            head = entity_map[candidate['head_entity_id']]
            tail = entity_map[candidate['tail_entity_id']]
            span_start = min(head['start'], tail['start'])
            span_end = max(head['end'], tail['end'])
            evidence = row['text'][span_start:span_end]
            if not evidence or head['surface_form'] not in evidence or tail['surface_form'] not in evidence:
                raise ValueError(f"成立关系{candidate['candidate_relation_id']}无法由冻结实体位置回取原文证据")
            evidence_source = 'deterministic_entity_span'
        relation = dict(candidate)
        relation['relation_id'] = f"{row['record_id']}_{candidate['candidate_relation_id']}"
        relation['evidence_text'] = evidence
        relation['evidence_start'] = row['text'].find(evidence)
        relation['evidence_end'] = relation['evidence_start'] + len(evidence)
        relation['evidence_source'] = evidence_source
        relation['decision_reason'] = str(decision.get('reason', '')).strip()
        relation.pop('candidate_relation_id', None)
        relations.append(relation)
    return relations

def progress_line(stage: str, done: int, total: int, success: int, failed: int, started: float) -> str:
    elapsed = time.time() - started
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    return f'[{stage}] {done}/{total} ({done / total:.1%}) 成功={success} 失败={failed} 已用={elapsed:.1f}s 预计剩余={eta:.1f}s'

def run_entity_stage(rows: list[dict[str, Any]], surface_index: dict[str, list[dict[str, str]]], concept_index: dict[tuple[str, str], dict[str, Any]], catalog: str) -> dict[str, dict[str, Any]]:
    existing = successful_map(ENTITY_CACHE)
    pending = [row for row in rows if row['record_id'] not in existing]
    print(f'[实体阶段] 已恢复{len(existing)}条，待处理{len(pending)}条。', flush=True)
    if len(existing) >= 100:
        refresh_dynamic_quality_entities(rows, existing)

    def task(row: dict[str, Any]) -> dict[str, Any]:
        candidates = explicit_candidates(row['text'], surface_index)
        try:
            messages = entity_messages(row, candidates, catalog)
            total_usage: Counter[str] = Counter()
            validation_error = None
            surface_repair_attempted = False
            for repair_attempt in range(5):
                response, _, usage = api_call(messages, seed_offset=int(row['record_id'].split('_')[-1]) + repair_attempt)
                total_usage.update({key: value for key, value in usage.items() if isinstance(value, int)})
                try:
                    entities, warnings = validate_and_fuse_entities(row, candidates, response, concept_index)
                    validation_error = None
                    break
                except ValueError as exc:
                    validation_error = exc
                    if not surface_repair_attempted and ('无法定位' in str(exc) or '证据单元非法' in str(exc)):
                        surface_repair_attempted = True
                        repaired, repair_usage = repair_surface_alignment(row, response)
                        total_usage.update(repair_usage)
                        try:
                            entities, warnings = validate_and_fuse_entities(row, candidates, repaired, concept_index)
                            response = repaired
                            validation_error = None
                            break
                        except ValueError as repaired_exc:
                            validation_error = repaired_exc
                            response = repaired
                            raise repaired_exc
                    if repair_attempt == 4:
                        raise validation_error
                    messages = messages + [{'role': 'assistant', 'content': json.dumps(response, ensure_ascii=False)}, {'role': 'user', 'content': f'上一次JSON未通过V6.1冻结规则校验：{validation_error}。请重新区分允许的6类claims与必须排除的7类excluded_claims；治疗主语必须标注合法head_role；疾病只取核心最小mention；explicit_disease_mentions必须逐项消解重叠疾病候选，禁止从热中消渴跨界生成中消；‘三消渴’只取三消；除8类专家指定专题症状外，症状、中药、病位不得孤立。surface_form必须可回溯原文。请返回完整JSON。'}]
            if validation_error is not None:
                raise validation_error
            result = {'record_id': row['record_id'], 'status': 'success', 'entities': entities, 'semantic_claims': response.get('_validated_hypotheses', []), 'excluded_claims': response.get('_validated_exclusions', []), 'surface_alignment_repairs': response.get('_surface_alignment_repairs', {}), 'scene_types': response.get('scene_types', []), 'main_disease': str(response.get('main_disease', '')).strip(), 'scope_audit': str(response.get('scope_audit', '')).strip(), 'translation': str(response.get('translation', '')).strip(), 'explicit_candidate_count': len(candidates), 'warnings': warnings, 'usage': usage}
            result['usage'] = dict(total_usage)
            append_jsonl(ENTITY_CACHE, result)
            return result
        except Exception as exc:
            failure = {'record_id': row['record_id'], 'stage': 'entity', 'status': 'failed', 'error': str(exc), 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            append_jsonl(FAILURE_PATH, failure)
            return failure
    started, success, failed = (time.time(), 0, 0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(task, row): row['record_id'] for row in pending}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            if result.get('status') == 'success':
                success += 1
            else:
                failed += 1
            print(progress_line('实体阶段', done, len(pending), success, failed, started), flush=True)
            completed_total = len(existing) + success
            if completed_total > 0 and completed_total % 100 == 0:
                refresh_dynamic_quality_entities(rows, successful_map(ENTITY_CACHE))
    final = successful_map(ENTITY_CACHE)
    refresh_dynamic_quality_entities(rows, final)
    if len(final) != len(rows):
        raise RuntimeError(f'实体阶段未全部完成：成功{len(final)}/{len(rows)}。重新运行将仅处理失败项。')
    return final

def process_relation_record(row: dict[str, Any], entity_result: dict[str, Any], schemas: list[dict[str, str]]) -> dict[str, Any]:
    entities = entity_result['entities']
    candidates = relation_candidates(entities, schemas, entity_result.get('semantic_claims', []))
    if not candidates:
        result = {'record_id': row['record_id'], 'status': 'success', 'relations': [], 'relation_candidate_count': 0, 'usage': {}}
        append_jsonl(RELATION_CACHE, result)
        return result
    try:
        messages = relation_messages(row, entities, candidates)
        total_usage: Counter[str] = Counter()
        validation_error = None
        minimal_evidence_retry = False
        for repair_attempt in range(3):
            try:
                response, _, usage = api_call(messages, seed_offset=100000 + int(row['record_id'].split('_')[-1]) + repair_attempt)
            except DataInspectionError:
                if minimal_evidence_retry:
                    raise
                minimal_evidence_retry = True
                messages = minimal_relation_messages(row, entities, candidates)
                response, _, usage = api_call(messages, seed_offset=200000 + int(row['record_id'].split('_')[-1]))
            total_usage.update({key: value for key, value in usage.items() if isinstance(value, int)})
            try:
                relations = validate_relations(row, entities, candidates, response)
                validation_error = None
                break
            except ValueError as exc:
                validation_error = exc
                if repair_attempt == 2:
                    raise
                messages = messages + [{'role': 'assistant', 'content': json.dumps(response, ensure_ascii=False)}, {'role': 'user', 'content': f'上一次JSON未通过结构校验：{exc}。请保持语义判断原则不变，返回修正后的完整JSON；每个候选关系ID必须且只能出现一次，成立关系只返回有效的证据单元ID。'}]
        if validation_error is not None:
            raise validation_error
        result = {'record_id': row['record_id'], 'status': 'success', 'relations': relations, 'relation_candidate_count': len(candidates), 'usage': dict(total_usage), 'minimal_evidence_retry': minimal_evidence_retry}
        append_jsonl(RELATION_CACHE, result)
        return result
    except Exception as exc:
        failure = {'record_id': row['record_id'], 'stage': 'relation', 'status': 'failed', 'error': str(exc), 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        append_jsonl(FAILURE_PATH, failure)
        return failure

def run_dynamic_sample_relations(rows: list[dict[str, Any]], entity_map: dict[str, dict[str, Any]], selected_ids: list[str]) -> None:
    rowmap = {x['record_id']: x for x in rows}
    schemas = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))['relation_schemas']
    existing = successful_map(RELATION_CACHE)
    pending = [rid for rid in selected_ids if rid in entity_map and rid not in existing]
    if not pending:
        return
    started = time.time()
    success = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_relation_record, rowmap[rid], entity_map[rid], schemas): rid for rid in pending}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            if result.get('status') == 'success':
                success += 1
            else:
                failed += 1
            print(progress_line('动态样本关系', done, len(pending), success, failed, started), flush=True)

def run_relation_stage(rows: list[dict[str, Any]], entity_map: dict[str, dict[str, Any]], schemas: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    existing = successful_map(RELATION_CACHE)
    pending = [row for row in rows if row['record_id'] not in existing]
    print(f'[关系阶段] 已恢复{len(existing)}条，待处理{len(pending)}条。', flush=True)
    refresh_dynamic_quality_relations(rows, entity_map, existing)

    def task(row: dict[str, Any]) -> dict[str, Any]:
        return process_relation_record(row, entity_map[row['record_id']], schemas)
    started, success, failed = (time.time(), 0, 0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(task, row): row['record_id'] for row in pending}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            if result.get('status') == 'success':
                success += 1
            else:
                failed += 1
            print(progress_line('关系阶段', done, len(pending), success, failed, started), flush=True)
            completed_total = len(existing) + success
            if completed_total > 0 and completed_total % 100 == 0:
                refresh_dynamic_quality_relations(rows, entity_map, successful_map(RELATION_CACHE))
    final = successful_map(RELATION_CACHE)
    refresh_dynamic_quality_relations(rows, entity_map, final)
    if len(final) != len(rows):
        raise RuntimeError(f'关系阶段未全部完成：成功{len(final)}/{len(rows)}。重新运行将仅处理失败项。')
    return final

def current_manifest_payload() -> dict[str, Any]:
    return {'pipeline_semantic_sha256': pipeline_semantic_sha256(), 'input_sha256': sha256_file(INPUT_PATH), 'lexicon_sha256': sha256_file(LEXICON_PATH), 'schema_sha256': sha256_file(SCHEMA_PATH), 'strategy_sha256': sha256_file(STRATEGY_PATH), 'model': MODEL, 'temperature': TEMPERATURE, 'random_seed': RANDOM_SEED, 'prompt_version': PROMPT_VERSION, 'postprocess_version': POSTPROCESS_VERSION}

def validate_manifest() -> None:
    current = current_manifest_payload()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        old = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        if old != current:
            raise RuntimeError('V6.1.2缓存清单与当前输入/词表/Schema/策略/脚本或模型配置不一致。禁止误续跑；请保留旧缓存并使用新的空目录。')
    else:
        MANIFEST_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')

def preflight() -> None:
    if not API密钥.strip():
        raise RuntimeError('API密钥为空：请在本脚本顶部手工填写API密钥后重新运行。')
    messages = [{'role': 'user', 'content': '仅返回JSON：{"ok":true}'}]
    value, _, _ = api_call(messages, seed_offset=999999)
    if value.get('ok') is not True:
        raise RuntimeError('API预检响应不符合预期')
    print(f'API预检通过：{MODEL}', flush=True)

def final_quality_gate(final_rows: list[dict[str, Any]], entity_map: dict[str, dict[str, Any]]) -> None:
    problems = []
    disagreements = []
    for row in final_rows:
        entities = row['entities']
        relations = row['relations']
        ids = {x['entity_id'] for x in entities}
        for e in entities:
            if e.get('alignment_status') == 'exact' and row['text'][e['start']:e['end']] != e['surface_form']:
                problems.append(f"{row['record_id']}:实体证据错位:{e['surface_form']}")
            if e['entity_type'] in {'症状', '中药', '病位'} and e['surface_form'] not in TOPIC_STANDALONE_SYMPTOMS and (not any((e['entity_id'] in {r['head_entity_id'], r['tail_entity_id']} for r in relations))):
                disagreements.append(f"{row['record_id']}:关系语义否决后无最终关系实体:{e['surface_form']}|{e['entity_type']}")
        for r in relations:
            if r['head_entity_id'] not in ids or r['tail_entity_id'] not in ids:
                problems.append(f"{row['record_id']}:关系端点缺失")
            if (r['head_type'], r['relation_type'], r['tail_type']) not in RELATION_SIGNATURES:
                problems.append(f"{row['record_id']}:Schema非法")
            if not r.get('evidence_text') or r['evidence_text'] not in row['text']:
                problems.append(f"{row['record_id']}:关系证据非法")
        actual = {(r['head_surface'], r['head_type'], r['relation_type'], r['tail_surface'], r['tail_type']) for r in relations}
        for h in entity_map[row['record_id']].get('semantic_claims', []):
            sig = (h['head_surface'], h['head_type'], h['relation_type'], h['tail_surface'], h['tail_type'])
            if sig not in actual:
                disagreements.append(f"{row['record_id']}:实体阶段候选命题被关系阶段否决:{sig}")
    SEMANTIC_DISAGREEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEMANTIC_DISAGREEMENT_PATH.write_text(json.dumps({'description': '两阶段语义分歧审计，不属于缓存或结构错误，不改变系统预测结果。', 'count': len(disagreements), 'items': disagreements}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[语义分歧审计] 共{len(disagreements)}项，已单独保存；不作为结构质量门失败。', flush=True)
    if problems:
        FAILURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        (FAILURE_PATH.parent / 'quality_gate_failures.json').write_text(json.dumps(problems, ensure_ascii=False, indent=2), encoding='utf-8')
        raise RuntimeError(f'最终结构质量门未通过，共{len(problems)}项；结果未导出')

def main() -> None:
    global GOLD_EXAMPLES
    random.seed(RANDOM_SEED)
    GOLD_EXAMPLES = parse_gold()
    rows = read_jsonl(INPUT_PATH)
    expected = 3065
    if len(rows) != expected:
        raise ValueError(f'输入数量错误：{len(rows)} != {expected}')
    lexicon_payload = json.loads(LEXICON_PATH.read_text(encoding='utf-8'))
    schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
    lexicon_items = lexicon_payload['items']
    surface_index, concept_index = build_lexicon_index(lexicon_items)
    catalog = compact_concept_catalog(lexicon_items)
    validate_manifest()
    preflight()
    entity_map = run_entity_stage(rows, surface_index, concept_index, catalog)
    relation_map = run_relation_stage(rows, entity_map, schema['relation_schemas'])
    final_rows = []
    for row in sorted(rows, key=lambda x: (x.get('sample_order', 10 ** 9), x['record_id'])):
        rid = row['record_id']
        final_rows.append({**row, 'entities': entity_map[rid]['entities'], 'relations': relation_map[rid]['relations'], 'preannotation_meta': {'model': MODEL, 'temperature': TEMPERATURE, 'random_seed': RANDOM_SEED, 'prompt_version': PROMPT_VERSION, 'postprocess_version': POSTPROCESS_VERSION, 'explicit_candidate_count': entity_map[rid]['explicit_candidate_count'], 'relation_candidate_count': relation_map[rid]['relation_candidate_count']}})
    final_quality_gate(final_rows, entity_map)
    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = FINAL_PATH.with_suffix('.tmp')
    temp_path.write_text(''.join((json.dumps(x, ensure_ascii=False) + '\n' for x in final_rows)), encoding='utf-8')
    os.replace(temp_path, FINAL_PATH)
    export_full_compact_txt(final_rows)
    extract_and_freeze_test300(final_rows)
    summary = {'passages': len(final_rows), 'entities': sum((len(x['entities']) for x in final_rows)), 'relations': sum((len(x['relations']) for x in final_rows)), 'entity_types': dict(Counter((e['entity_type'] for x in final_rows for e in x['entities']))), 'relation_types': dict(Counter((r['relation_type'] for x in final_rows for r in x['relations']))), 'model': MODEL, 'prompt_version': PROMPT_VERSION, 'postprocess_version': POSTPROCESS_VERSION, 'run_mode': 'full_reproduction', 'input_path': str(INPUT_PATH), 'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    RUN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f'全库复现完成：{FINAL_PATH}', flush=True)
    print(f'300条复现机器预测：{TEST_PREDICTION_PATH}', flush=True)
if __name__ == '__main__':
    main()
