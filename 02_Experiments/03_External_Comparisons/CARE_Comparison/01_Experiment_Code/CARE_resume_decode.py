"""Resume fixed100 CARE with equivalent indexed decoding; original training files unchanged."""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import sys
import time

EXPECTED_RUN_SHA256 = '948a67eaf907342929b51070e385d1050df0534935e5e3bf5604cb3df77d15d5'


def indexed_decode(text, offsets, entity_hits, relation_hits, types, relations, legal):
    # Every frozen Schema connects different entity types. Thus a legal pair
    # cannot be an identical entity; positions can be folded within each start.
    if any(head_type == tail_type for head_type, _, tail_type in legal):
        raise RuntimeError('This indexed decoder requires the frozen different-type Schema')
    entities = set()
    groups = defaultdict(lambda: defaultdict(set))
    for start, end, label in entity_hits:
        if not 0 <= start <= end < len(offsets):
            continue
        a, b = offsets[start][0], offsets[end][1]
        if offsets[start][0] == offsets[start][1] or offsets[end][0] == offsets[end][1] or a >= b:
            continue
        surface, typ = text[a:b], types[label]
        entities.add((surface, typ, a, b))
        groups[start][typ].add(surface)
    compatibility = defaultdict(list)
    for head_type, relation, tail_type in sorted(legal):
        compatibility[relation].append((head_type, tail_type))
    # Union equivalent surface candidates before expanding them. This preserves
    # the original strict-set output while avoiding repeated positional products.
    unions = defaultdict(set)
    for head, tail, label in relation_hits:
        if head not in groups or tail not in groups:
            continue
        relation = relations[label]
        for head_type, tail_type in compatibility[relation]:
            if head_type in groups[head] and tail_type in groups[tail]:
                unions[(head, head_type, relation)].update(groups[tail][tail_type])
    triples = set()
    for (start, head_type, relation), tails in unions.items():
        for head in groups[start][head_type]:
            triples.update((head, relation, tail) for tail in tails)
    return {
        'entities': [{'surface_form': s, 'entity_type': typ, 'start': a, 'end': b}
                     for s, typ, a, b in sorted(entities)],
        'folded_entities': sorted({(s, typ) for s, typ, _, _ in entities}),
        'relations': sorted(triples)}


def indexed_audit(predictions, inputs):
    from care_data import assets, check_entities
    types, _, legal = assets()
    index = {row['record_id']: row for row in inputs}
    if len(predictions) != len(index) or {p['record_id'] for p in predictions} != set(index):
        raise RuntimeError('Prediction size/ID completeness failed')
    for pred in predictions:
        row = index[pred['record_id']]
        check_entities(row['text'], pred['entities'], types)
        folded = {(e['surface_form'], e['entity_type']) for e in pred['entities']}
        if set(map(tuple, pred['folded_entities'])) != folded:
            raise RuntimeError('Entity folding mismatch')
        surface_types = defaultdict(set)
        for surface, typ in folded:
            surface_types[surface].add(typ)
        for head, relation, tail in pred['relations']:
            if not any((ht, relation, tt) in legal
                       for ht in surface_types.get(head, ()) for tt in surface_types.get(tail, ())):
                raise RuntimeError('Relation endpoint or Schema check failed')
    return {'complete': True, 'records': len(predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent / 'CARE_Balanced')
    args, extra = parser.parse_known_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    import CARE_run as runner
    if runner.fingerprint(root / 'CARE_run.py') != EXPECTED_RUN_SHA256:
        raise RuntimeError('Original runner differs; this compatibility patch cannot be applied')
    runner.verify_assets()
    before = inspect.getsource(runner.evaluate)
    block = '''            starts = {}
            for start, end, label in hits[0]:
                starts[start] = starts.get(start, 0) + 1
            work = sum(starts.get(head, 0) * starts.get(tail, 0) for head, tail, label in hits[1])
            if work > 1000000:
                if not allow_skip:
                    raise RuntimeError('Excessive prediction expansion; not truncated: ' + row['record_id'])
                skipped.append(row['record_id'])
                continue
'''
    if before.count(block) != 1:
        raise RuntimeError('Expected decoder guard was not found exactly once')
    after = before.replace(block,
        '            print(f"Decoding {row[\'record_id\']}: entity_cells={len(hits[0])} relation_cells={len(hits[1])}", flush=True)\n')
    after = after.replace('            predictions.append(prediction)',
        '            predictions.append(prediction)\n'
        '            print(f"Decoded {row[\'record_id\']}: entities={len(prediction[\'entities\'])} relations={len(prediction[\'relations\'])}", flush=True)')
    runner.decode = indexed_decode
    runner.audit_predictions = indexed_audit
    exec(compile(after, str(Path(__file__)), 'exec'), runner.__dict__)
    output = root / 'results_fixed100'
    output.mkdir(exist_ok=True)
    from care_data import append_jsonl
    patch_hash = runner.fingerprint(Path(__file__))
    source_file = output / ('decoder_patch_' + patch_hash + '.txt')
    if not source_file.exists():
        source_file.write_text(Path(__file__).read_text('utf-8'), encoding='utf-8')
    append_jsonl(output / 'decoder_patch_history.jsonl', {
        'time_unix': time.time(), 'patch_sha256': patch_hash, 'source_file': source_file.name,
        'original_runner_sha256': EXPECTED_RUN_SHA256, 'threshold_changed': False,
        'training_changed': False, 'predictions_truncated': False,
        'change': 'Equivalent indexed Schema joins and endpoint audits; remove raw Cartesian-count guard'})
    print('Equivalent decoder patch applied. Existing bindings/checkpoints retained; no retraining of completed runs.', flush=True)
    sys.argv = [str(root / 'CARE_run.py'), 'full', *extra]
    runner.main()


if __name__ == '__main__':
    main()
