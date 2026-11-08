"""Evaluate frozen CARE second-version predictions with the existing shared metrics."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
ROOT = CODE / 'CARE_Balanced'
HOME = CODE.parent
MAIN = next(p for p in HOME.parents if p.name == '主实验')
OUT = HOME / '03_Result_Reports'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE))
from care_data import read_jsonl, atomic_json, verify_assets
from CARE_resume_decode import indexed_audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'evaluation_source' / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    verify_assets()
    ev = load('frozen_evaluator', '03_Unified_Evaluation.py')
    pair = load('frozen_paired', '01_外部方法配对统计比较.py')
    old = MAIN / '03_External_Comparisons'
    gold_path = old / '02_Input_Data/Independent_Test_300/Independent_Test_300_Expert_Gold.txt'
    frozen = json.loads((old / '05_Evaluation_and_Verification/External_Baseline_Frozen_Manifest_20260826.json').read_text('utf-8'))
    if sha(gold_path) != frozen['gold_sha256']:
        raise RuntimeError('Frozen test Gold fingerprint differs')
    ev.GOLD = pair.GOLD = gold_path
    gold = ev.parse_gold()
    gm = {g['record_id']: g for g in gold}
    inputs = read_jsonl(ROOT / '05_Frozen_Data_Indices/independent_test_300_input_only.jsonl')
    if len(gold) != 300 or len(gm) != 300 or set(gm) != {p['record_id'] for p in inputs}:
        raise RuntimeError('Gold/input ID completeness differs')
    results_dir = ROOT / 'results_calibrated_v1'
    selection = json.loads((results_dir / 'threshold_selection.json').read_text('utf-8'))
    binding_path = results_dir / 'binding.json'
    if not selection['development_gate_passed'] or selection['binding_sha256'] != sha(binding_path):
        raise RuntimeError('Development threshold selection is not frozen')
    audit = json.loads((results_dir / 'formal_audit.json').read_text('utf-8'))
    reference_path = old / '02_Input_Data/Frozen_ECKE_Reference_Predictions/independent_test_300_machine_predictions_v612_frozen.jsonl'
    reference = pair.read_pred(reference_path)
    if set(reference) != set(gm):
        raise RuntimeError('Main-method reference IDs differ')
    scores, comparisons, table = [], {}, []
    OUT.mkdir(exist_ok=True)
    for seed in (20260823, 20260824, 20260825):
        path = results_dir / f'seed_{seed}/predictions.jsonl'
        if sha(path) != audit[str(seed)]['sha256'] or audit[str(seed)]['thresholds'] != selection['selected']['thresholds']:
            raise RuntimeError('Prediction fingerprint or threshold differs')
        pred = read_jsonl(path)
        indexed_audit(pred, inputs)
        counts = [[], []]
        by_type = [{t: [0, 0, 0] for t in ev.ENTITY_TYPES}, {t: [0, 0, 0] for t in ev.RELATION_TYPES}]
        exact, errors, baseline = [0, 0, 0], [], {}
        passage_counts = []
        for p in pred:
            g = gm[p['record_id']]
            pe = {tuple(x) for x in p['folded_entities']}
            pr = {tuple(x) for x in p['relations']}
            baseline[p['record_id']] = {'entities': pe, 'relations': pr}
            for i, (ps, gs) in enumerate(((pe, g['entities']), (pr, g['relations']))):
                counts[i].append(ev.count(ps, gs))
                for typ in by_type[i]:
                    c = ev.count({x for x in ps if x[1] == typ}, {x for x in gs if x[1] == typ})
                    for j in range(3):
                        by_type[i][typ][j] += c[j]
            ee, rr = pe == g['entities'], pr == g['relations']
            exact[0] += ee
            exact[1] += rr
            exact[2] += ee and rr
            passage_counts.append({'record_id': p['record_id'], 'entity': counts[0][-1], 'relation': counts[1][-1]})
            if not (ee and rr):
                # Keep counts, omissions and a bounded FP sample; full predictions remain available.
                errors.append({'record_id': p['record_id'], 'entity_fp_count': len(pe - g['entities']),
                               'relation_fp_count': len(pr - g['relations']),
                               'entity_fn': sorted(g['entities'] - pe), 'relation_fn': sorted(g['relations'] - pr),
                               'entity_fp_first10': sorted(pe - g['entities'])[:10],
                               'relation_fp_first10': sorted(pr - g['relations'])[:10]})
        result = {'version': 'CARE_second_calibrated_20261005', 'seed': seed, 'records': 300,
                  'matching': 'strict_set_based_order_invariant', 'thresholds': selection['selected']['thresholds'],
                  'prediction_sha256': sha(path), 'gold_sha256': sha(gold_path),
                  'binding_sha256': sha(binding_path), 'threshold_selection_sha256': sha(results_dir / 'threshold_selection.json'),
                  'evidence_and_schema_audit': 'passed; structural validity does not imply semantic correctness',
                  'joint_passage_exact': exact[2] / 300, 'mismatched_records': len(errors),
                  'passage_counts': passage_counts, 'error_summary': errors}
        for i, task in enumerate(('entity', 'relation')):
            types = {t: ev.prf(*c) for t, c in by_type[i].items()}
            result[task] = {'micro': ev.prf(*(sum(c[j] for c in counts[i]) for j in range(3))),
                            'ci95_f1': ev.ci(counts[i]), 'by_type': types,
                            'macro_f1': statistics.mean(x['f1'] for x in types.values()),
                            'passage_exact': exact[i] / 300}
            table.append({'seed': seed, 'task': task, **result[task]['micro'],
                          'f1_ci95_low': result[task]['ci95_f1'][0], 'f1_ci95_high': result[task]['ci95_f1'][1]})
        atomic_json(OUT / f'CARE第二版_seed_{seed}_正式评价.json', result)
        scores.append(result)
        print('Evaluated', seed, result['entity']['micro']['f1'], result['relation']['micro']['f1'], flush=True)
        comparisons[str(seed)] = {}
        for task in ('entities', 'relations'):
            print('Paired statistics', seed, task, flush=True)
            comparisons[str(seed)][task] = pair.paired(task, gold, reference, baseline, pair.SEED + seed)
    for task in ('entities', 'relations'):
        pair.holm([r[task] for r in comparisons.values()], 'paired_randomization_p', 'holm_p_CARE_three_runs')
    atomic_json(OUT / 'CARE_V2_Paired_Comparison_with_ECKE.json', {'design': 'Each task Holm correction across three CARE seeds only; not a global all-baseline correction',
                'bootstrap_n': pair.N_BOOT, 'permutation_n': pair.N_PERM, 'reference_sha256': sha(reference_path), 'runs': comparisons})
    summary = {t: {m: {'mean': statistics.mean(s[t]['micro'][m] for s in scores),
                       'sample_std': statistics.stdev(s[t]['micro'][m] for s in scores)}
                   for m in ('precision', 'recall', 'f1')} for t in ('entity', 'relation')}
    atomic_json(OUT / 'CARE_V2_Three_Seed_Summary.json', {'records_per_seed': 300, 'seeds': [s['seed'] for s in scores],
                'unit': 'fraction_0_to_1', 'summary': summary, 'thresholds': selection['selected']['thresholds'],
                'bootstrap_n': ev.BOOTSTRAP_N, 'bootstrap_seed': ev.SEED,
                'macro_definition': 'Seven entity types; six relation labels; frozen evaluator empty-class convention'})
    with (OUT / 'CARE_V2_Metrics_by_Seed.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ['# CARE第二版正式实验结果', '', '300条固定独立测试集；严格集合匹配；三个种子的均值±样本标准差。',
             '', '以下指标单位为百分比。', '', '|任务|Precision|Recall|Micro-F1|', '|---|---:|---:|---:|']
    for task, label in (('entity', '实体'), ('relation', '关系')):
        cells = [f"{100*summary[task][m]['mean']:.2f}±{100*summary[task][m]['sample_std']:.2f}" for m in ('precision', 'recall', 'f1')]
        lines.append('|' + '|'.join([label, *cells]) + '|')
    lines += ['', '|种子|实体F1|关系F1|', '|---|---:|---:|']
    lines += [f"|{s['seed']}|{100*s['entity']['micro']['f1']:.2f}|{100*s['relation']['micro']['f1']:.2f}|" for s in scores]
    lines += ['', '实体与关系阈值均为0.9999；阈值只用100条开发数据的固定五折模型选择，未改变模型训练。',
              '逐种子置信区间、类别结果、错误摘要和配对统计见同目录JSON。',
              '五折输出参与阈值选择，其选择后得分不能作为独立验证性能。',
              '第二版是在第一版正式测试已经评价后开展的开发集校准，应在论文中披露这一顺序；未用测试Gold选择阈值。',
              '配对统计仅对CARE三个种子分别作任务内Holm校正，不能声称已完成所有外部方法的全局多重比较。',
              '关系F1仍低，不因本次改进而推断CARE在其他任务中的性能。']
    (OUT / 'CARE_V2_Final_Results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Second-version evaluation and exports complete:', OUT, flush=True)


if __name__ == '__main__':
    main()
