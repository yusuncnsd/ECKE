from __future__ import annotations
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / '主实验'
PRED_FILE = ROOT / '02_ECKE/04_结果数据/测试集预测/消渴知识抽取_300条独立测试集专家预标注_V612.txt'
GOLD_FILE = ROOT / '02_ECKE/02_输入数据/独立测试集300/消渴知识抽取_300条独立测试集专家标注-盲标定.txt'
OUT_JSON = ROOT / '02_ECKE/05_评价与核验/复现运行' / '300条独立测试集正式性能评估_V612_GoldV1.json'
OUT_TXT = ROOT / '02_ECKE/05_评价与核验/复现运行' / '300条独立测试集正式性能评估_V612_GoldV1.txt'
ENTITY_TYPES = ('中药', '方药', '疾病', '症状', '病机', '病位', '治法')
RELATION_TYPES = ('治疗', '主治', '导致', '涉及', '提示', '表现')
SCHEMA = {('中药', '治疗', '疾病'), ('方药', '治疗', '疾病'), ('治法', '主治', '疾病'), ('病机', '导致', '疾病'), ('病机', '涉及', '病位'), ('症状', '提示', '病机'), ('疾病', '表现', '症状')}
BOOTSTRAP_N = 5000
SEED = 20260822
SEP = '=' * 80

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def split_blocks(text: str) -> list[str]:
    starts = list(re.finditer('(?m)^ann_id:\\s*test_\\d+\\s*$', text))
    return [text[m.start():starts[i + 1].start() if i + 1 < len(starts) else len(text)] for i, m in enumerate(starts)]

def get_section(block: str, header: str, headers: tuple[str, ...]) -> str:
    alternatives = '|'.join((re.escape(x) for x in headers))
    match = re.search(f'(?ms)^{re.escape(header)}[ \\t]*\\r?\\n(.*?)(?=^(?:{alternatives})[ \\t]*\\r?$|^[ \\t]*={{20,}}[ \\t]*\\r?$|\\Z)', block)
    if not match:
        raise RuntimeError(f'缺少栏目{header}')
    return match.group(1).strip()

def parse_lines(raw: str, width: int, ann_id: str, field: str, issues: list[dict]) -> list[tuple[str, ...]]:
    if not raw or raw == '无':
        return []
    values: list[tuple[str, ...]] = []
    expected = '[^|｜]+ \\| [^|｜]+' if width == 2 else '[^|｜]+ \\| [^|｜]+ \\| [^|｜]+'
    for n, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        if not re.fullmatch(expected, line):
            issues.append({'ann_id': ann_id, 'kind': 'pipe_spacing', 'field': field, 'line_no': n, 'line': line})
        parts = tuple((x.strip() for x in re.split('[|｜]', line)))
        if len(parts) != width or not all(parts):
            issues.append({'ann_id': ann_id, 'kind': 'field_count', 'field': field, 'line_no': n, 'line': line})
            continue
        values.append(parts)
    return values

def parse_prediction() -> list[dict]:
    headers = ('【原文】', '【译文】', '【预标实体】', '【校对实体】', '【预标关系】', '【校对关系】')
    text = PRED_FILE.read_bytes().decode('utf-8-sig')
    result = []
    for block in split_blocks(text):
        ann = re.search('(?m)^ann_id:\\s*(\\S+)\\s*$', block)
        pid = re.search('(?m)^paragraph_id:\\s*(\\S+)\\s*$', block)
        result.append({'ann_id': ann.group(1), 'paragraph_id': pid.group(1), 'text': get_section(block, '【原文】', headers), 'entities_raw': get_section(block, '【预标实体】', headers), 'relations_raw': get_section(block, '【预标关系】', headers)})
    return result

def parse_gold() -> tuple[list[dict], list[dict]]:
    headers = ('【原文】', '【盲标实体】', '【盲标关系】')
    text = GOLD_FILE.read_bytes().decode('utf-8-sig')
    result, issues = ([], [])
    for block in split_blocks(text):
        ann = re.search('(?m)^ann_id:\\s*(\\S+)\\s*$', block)
        pid = re.search('(?m)^paragraph_id:\\s*(\\S+)\\s*$', block)
        ann_id = ann.group(1)
        original = get_section(block, '【原文】', headers)
        entities = parse_lines(get_section(block, '【盲标实体】', headers), 2, ann_id, '盲标实体', issues)
        relations = parse_lines(get_section(block, '【盲标关系】', headers), 3, ann_id, '盲标关系', issues)
        surfaces: dict[str, set[str]] = defaultdict(set)
        for surface, typ in entities:
            surfaces[surface].add(typ)
            if typ not in ENTITY_TYPES:
                issues.append({'ann_id': ann_id, 'kind': 'illegal_entity_type', 'line': f'{surface} | {typ}'})
            if surface not in original:
                issues.append({'ann_id': ann_id, 'kind': 'entity_not_in_text', 'line': f'{surface} | {typ}'})
        if len(entities) != len(set(entities)):
            issues.append({'ann_id': ann_id, 'kind': 'duplicate_entity', 'line': ''})
        if len(relations) != len(set(relations)):
            issues.append({'ann_id': ann_id, 'kind': 'duplicate_relation', 'line': ''})
        for head, rel, tail in relations:
            if head not in surfaces:
                issues.append({'ann_id': ann_id, 'kind': 'missing_head', 'line': f'{head} | {rel} | {tail}'})
            if tail not in surfaces:
                issues.append({'ann_id': ann_id, 'kind': 'missing_tail', 'line': f'{head} | {rel} | {tail}'})
            if head in surfaces and tail in surfaces and (not any(((ht, rel, tt) in SCHEMA for ht in surfaces[head] for tt in surfaces[tail]))):
                issues.append({'ann_id': ann_id, 'kind': 'illegal_schema', 'line': f'{head} | {rel} | {tail}'})
        result.append({'ann_id': ann_id, 'paragraph_id': pid.group(1), 'text': original, 'entities': set(entities), 'relations': set(relations)})
    return (result, issues)

def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 1.0 if fn == 0 else 0.0
    r = tp / (tp + fn) if tp + fn else 1.0 if fp == 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': f1}

def count(pred: set, gold: set) -> tuple[int, int, int]:
    return (len(pred & gold), len(pred - gold), len(gold - pred))

def verify_order_invariance(pred_records: list[dict], gold_records: list[dict]) -> dict:
    checked = 0
    for pred, gold in zip(pred_records, gold_records):
        issues: list[dict] = []
        pe = set(parse_lines(pred['entities_raw'], 2, pred['ann_id'], '预标实体', issues))
        pr = set(parse_lines(pred['relations_raw'], 3, pred['ann_id'], '预标关系', issues))
        reversed_e_raw = '\n'.join(reversed(pred['entities_raw'].splitlines()))
        reversed_r_raw = '\n'.join(reversed(pred['relations_raw'].splitlines()))
        pe_reversed = set(parse_lines(reversed_e_raw, 2, pred['ann_id'], '预标实体-反序', issues))
        pr_reversed = set(parse_lines(reversed_r_raw, 3, pred['ann_id'], '预标关系-反序', issues))
        if issues or pe != pe_reversed or pr != pr_reversed:
            raise RuntimeError(f"顺序无关性验证失败：{pred['ann_id']}")
        if gold['entities'] != set(reversed(sorted(gold['entities']))) or gold['relations'] != set(reversed(sorted(gold['relations']))):
            raise RuntimeError(f"Gold顺序无关性验证失败：{gold['ann_id']}")
        checked += 1
    return {'passed': True, 'records_checked': checked, 'entity_key': '(surface_form, entity_type)', 'relation_key': '(head_surface, relation_type, tail_surface)', 'note': '记录内行顺序不参与匹配；关系三元组内部头尾方向仍严格区分。'}

def bootstrap_ci(per_record: list[tuple[int, int, int]]) -> dict[str, list[float]]:
    rng = random.Random(SEED)
    n = len(per_record)
    values = {'precision': [], 'recall': [], 'f1': []}
    for _ in range(BOOTSTRAP_N):
        sample = [per_record[rng.randrange(n)] for _ in range(n)]
        metric = prf(sum((x[0] for x in sample)), sum((x[1] for x in sample)), sum((x[2] for x in sample)))
        for key in values:
            values[key].append(metric[key])
    result = {}
    for key, vals in values.items():
        vals.sort()
        result[key] = [vals[int(0.025 * (BOOTSTRAP_N - 1))], vals[int(0.975 * (BOOTSTRAP_N - 1))]]
    return result

def fmt(m: dict) -> str:
    return f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})"

def main() -> None:
    pred_records = parse_prediction()
    gold_records, gold_issues = parse_gold()
    if len(pred_records) != 300 or len(gold_records) != 300:
        raise RuntimeError(f'数量错误：预测{len(pred_records)}，Gold{len(gold_records)}')
    if [x['ann_id'] for x in pred_records] != [x['ann_id'] for x in gold_records]:
        raise RuntimeError('预测与Gold的ID或顺序不一致')
    if gold_issues:
        raise RuntimeError(f'Gold质量门未通过，共{len(gold_issues)}项：{gold_issues[:10]}')
    order_audit = verify_order_invariance(pred_records, gold_records)
    totals_e, totals_r = ([0, 0, 0], [0, 0, 0])
    per_e, per_r = ([], [])
    by_e = defaultdict(lambda: [0, 0, 0])
    by_r = defaultdict(lambda: [0, 0, 0])
    exact_e = exact_r = exact_joint = 0
    errors = []
    fp_e_types, fn_e_types, fp_r_types, fn_r_types = (Counter(), Counter(), Counter(), Counter())
    rows = []
    for pred, gold in zip(pred_records, gold_records):
        parse_issues: list[dict] = []
        pe = set(parse_lines(pred['entities_raw'], 2, pred['ann_id'], '预标实体', parse_issues))
        pr = set(parse_lines(pred['relations_raw'], 3, pred['ann_id'], '预标关系', parse_issues))
        if parse_issues:
            raise RuntimeError(f'机器预测格式异常：{parse_issues[:10]}')
        ge, gr = (gold['entities'], gold['relations'])
        ce, cr = (count(pe, ge), count(pr, gr))
        per_e.append(ce)
        per_r.append(cr)
        for i in range(3):
            totals_e[i] += ce[i]
            totals_r[i] += cr[i]
        for typ in ENTITY_TYPES:
            c = count({x for x in pe if x[1] == typ}, {x for x in ge if x[1] == typ})
            for i in range(3):
                by_e[typ][i] += c[i]
        for typ in RELATION_TYPES:
            c = count({x for x in pr if x[1] == typ}, {x for x in gr if x[1] == typ})
            for i in range(3):
                by_r[typ][i] += c[i]
        ee, rr = (pe == ge, pr == gr)
        exact_e += ee
        exact_r += rr
        exact_joint += ee and rr
        efp, efn, rfp, rfn = (pe - ge, ge - pe, pr - gr, gr - pr)
        fp_e_types.update((x[1] for x in efp))
        fn_e_types.update((x[1] for x in efn))
        fp_r_types.update((x[1] for x in rfp))
        fn_r_types.update((x[1] for x in rfn))
        if efp or efn or rfp or rfn:
            errors.append({'ann_id': pred['ann_id'], 'paragraph_id': pred['paragraph_id'], 'text': gold['text'], 'entity_fp': sorted(map(list, efp)), 'entity_fn': sorted(map(list, efn)), 'relation_fp': sorted(map(list, rfp)), 'relation_fn': sorted(map(list, rfn)), 'total_differences': len(efp) + len(efn) + len(rfp) + len(rfn)})
        rows.append({'ann_id': pred['ann_id'], 'length': len(gold['text']), 'gold_entities': len(ge), 'gold_relations': len(gr), 'entity_counts': ce, 'relation_counts': cr})
    entity_micro, relation_micro = (prf(*totals_e), prf(*totals_r))
    entity_types = {x: prf(*by_e[x]) for x in ENTITY_TYPES}
    relation_types = {x: prf(*by_r[x]) for x in RELATION_TYPES}
    entity_macro = sum((x['f1'] for x in entity_types.values())) / len(entity_types)
    present_rel = [x for x in relation_types.values() if x['tp'] + x['fp'] + x['fn'] > 0]
    relation_macro = sum((x['f1'] for x in present_rel)) / len(present_rel)

    def strata(key: str) -> list[dict]:
        ordered = sorted((x[key] for x in rows))
        q1, q2 = (ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3])
        if key == 'gold_entities':
            groups = [('低（0—2）', lambda v: v <= 2), ('中（3—5）', lambda v: 3 <= v <= 5), ('高（≥6）', lambda v: v >= 6)]
            rule = '固定区间：0—2、3—5、≥6'
        elif key == 'gold_relations':
            groups = [('低（0—1）', lambda v: v <= 1), ('中（2—4）', lambda v: 2 <= v <= 4), ('高（≥5）', lambda v: v >= 5)]
            rule = '固定区间：0—1、2—4、≥5'
        else:
            groups = [('低', lambda v: v <= q1), ('中', lambda v: q1 < v <= q2), ('高', lambda v: v > q2)]
            rule = f'三分位：q1={q1}, q2={q2}'
        output = []
        for name, predicate in groups:
            sample = [x for x in rows if predicate(x[key])]
            ec = [sum((x['entity_counts'][i] for x in sample)) for i in range(3)]
            rc = [sum((x['relation_counts'][i] for x in sample)) for i in range(3)]
            output.append({'group': name, 'n': len(sample), 'range_rule': rule, 'entity': prf(*ec), 'relation': prf(*rc)})
        return output
    errors.sort(key=lambda x: (-x['total_differences'], x['ann_id']))
    result = {'status': 'V612预测在300条独立测试集GoldV1上的正式评价；Gold质量门通过。后续如Gold勘误须版本化并统一重评。', 'files': {'prediction': str(PRED_FILE), 'prediction_sha256': sha256(PRED_FILE), 'gold': str(GOLD_FILE), 'gold_sha256': sha256(GOLD_FILE)}, 'configuration': {'records': 300, 'matching': 'strict_set_based_order_invariant', 'bootstrap_n': BOOTSTRAP_N, 'seed': SEED}, 'order_invariance_audit': order_audit, 'entity': {'micro': entity_micro, 'macro_f1': entity_macro, 'ci95': bootstrap_ci(per_e), 'passage_exact': exact_e / 300, 'by_type': entity_types}, 'relation': {'micro': relation_micro, 'macro_f1': relation_macro, 'ci95': bootstrap_ci(per_r), 'passage_exact': exact_r / 300, 'by_type': relation_types}, 'joint_passage_exact': exact_joint / 300, 'mismatched_records': len(errors), 'error_counts': {'entity_fp_by_type': dict(fp_e_types), 'entity_fn_by_type': dict(fn_e_types), 'relation_fp_by_type': dict(fp_r_types), 'relation_fn_by_type': dict(fn_r_types)}, 'stratified': {'text_length': strata('length'), 'gold_entity_density': strata('gold_entities'), 'gold_relation_complexity': strata('gold_relations')}, 'top_error_records': errors[:20], 'all_error_records': errors}
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['300条独立测试集盲标Gold正式性能评估 V1', '=' * 72, 'Gold质量门：通过（300条、ID一致、实体类型合法、关系Schema合法、端点完整、原文证据可定位）。', '机器预测：V612冻结预标；未重新调用模型，未依据Gold调整方法。', '实体严格匹配=表面形式+类型；关系严格匹配=头实体+关系+尾实体。', '顺序无关性质量门：通过。实体行和关系列表顺序不参与评价，三元组内部头尾方向仍严格区分。', f'Bootstrap：按条文配对重采样{BOOTSTRAP_N}次，种子{SEED}。', '', '一、整体结果', f'实体Micro：{fmt(entity_micro)}', f"实体F1 95% CI：[{result['entity']['ci95']['f1'][0]:.4f}, {result['entity']['ci95']['f1'][1]:.4f}]", f'实体Macro-F1：{entity_macro:.4f}', f'实体条文级完全正确率：{exact_e}/300 = {exact_e / 300:.4f}', '', f'关系Micro：{fmt(relation_micro)}', f"关系F1 95% CI：[{result['relation']['ci95']['f1'][0]:.4f}, {result['relation']['ci95']['f1'][1]:.4f}]", f'关系Macro-F1：{relation_macro:.4f}', f'关系条文级完全正确率：{exact_r}/300 = {exact_r / 300:.4f}', f'实体与关系联合完全正确率：{exact_joint}/300 = {exact_joint / 300:.4f}', f'存在任一差异的条文：{len(errors)}/300', '', '二、实体分类型']
    lines += [f'- {typ}：{fmt(entity_types[typ])}' for typ in ENTITY_TYPES]
    lines += ['', '三、关系分类型'] + [f'- {typ}：{fmt(relation_types[typ])}' for typ in RELATION_TYPES]
    lines += ['', '四、错误数量分布', '实体FP：' + '；'.join((f'{k}={v}' for k, v in fp_e_types.most_common())), '实体FN：' + '；'.join((f'{k}={v}' for k, v in fn_e_types.most_common())), '关系FP：' + '；'.join((f'{k}={v}' for k, v in fp_r_types.most_common())), '关系FN：' + '；'.join((f'{k}={v}' for k, v in fn_r_types.most_common())), '']
    for title, key in [('文本长度', 'text_length'), ('Gold实体密度', 'gold_entity_density'), ('Gold关系复杂度', 'gold_relation_complexity')]:
        lines.append(f'五、{title}分层' if title == '文本长度' else f'{title}分层')
        for item in result['stratified'][key]:
            lines.append(f"- {item['group']}（n={item['n']}，{item['range_rule']}）：实体F1={item['entity']['f1']:.4f}，关系F1={item['relation']['f1']:.4f}")
        lines.append('')
    lines += ['六、差异最集中的条文']
    for item in errors[:20]:
        lines.append(f"- {item['ann_id']}：总差异={item['total_differences']}，实体FP={len(item['entity_fp'])}，实体FN={len(item['entity_fn'])}，关系FP={len(item['relation_fp'])}，关系FN={len(item['relation_fn'])}")
    lines += ['', '说明：本结果是完整框架的主方法性能，不等于已经完成外部基线、消融或显著性比较。']
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(OUT_TXT)
    print(OUT_JSON)
    print('实体', fmt(entity_micro))
    print('关系', fmt(relation_micro))
if __name__ == '__main__':
    main()
