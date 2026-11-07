"""Development-only threshold calibration; reuse fixed100 checkpoints, never train."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time

GRID = (0.5, 0.8, 0.95, 0.99, 0.999, 0.9999)
VERSION = 'CARE_dev_threshold_calibration_v1_20261005'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def row_digest(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def choose(results):
    base = next(r for r in results if r['thresholds'] == [0.5, 0.5])
    # Require both tasks to improve; otherwise keep the existing frozen run.
    better = [r for r in results if all(r[t]['f1'] > base[t]['f1'] + 1e-12
                                      for t in ('entity', 'relation'))]
    best = max(better, key=lambda r: ((r['entity']['f1'] + r['relation']['f1']) / 2,
                                     r['relation']['precision'], *r['thresholds'])) if better else base
    return base, best, bool(better)


def self_test():
    def score(a, b, x, y):
        return {'thresholds': [a, b], 'entity': {'f1': x},
                'relation': {'f1': y, 'precision': y}}
    base = score(.5, .5, .2, .1)
    bad = score(.8, .8, .8, .09)
    good = score(.95, .99, .3, .2)
    assert choose([base, bad]) == (base, base, False)
    assert choose([base, bad, good]) == (base, good, True)
    import CARE_resume_decode as decoder
    text = '甲乙'; offsets = [(0, 1), (1, 2)]
    p = decoder.indexed_decode(text, offsets, [[0, 0, 0], [1, 1, 1]],
                               [[0, 1, 0]], ['中药', '疾病'], ['治疗'],
                               {('中药', '治疗', '疾病')})
    assert p['relations'] == [('甲', '治疗', '乙')]
    print('PASS: development gate, threshold tie policy, indexed Schema decoding', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent / 'CARE_Balanced')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from care_data import verify_assets, assets, read_jsonl, atomic_json, prf, gold_entities, gold_edges
    from CARE_resume_decode import indexed_decode, indexed_audit
    from CARE_run import progress
    verify_assets()
    config = json.loads((root / 'config.json').read_text('utf-8'))
    old = root / 'results_fixed100'
    model_path = (args.model or root.parent / 'AutoDL_CARE/models/bert-base-chinese').resolve()
    dev = read_jsonl(root / '05_Frozen_Data_Indices/development_100_span_supervision.jsonl')
    test = read_jsonl(root / '05_Frozen_Data_Indices/independent_test_300_input_only.jsonl')
    folds = [(old / f'cv/epochs_100/fold_{i}',
              read_jsonl(root / f'05_Frozen_Data_Indices/cv_fold{i}_train.jsonl'),
              read_jsonl(root / f'05_Frozen_Data_Indices/cv_fold{i}_valid.jsonl'), config['cv_seed']) for i in range(5)]
    formal = [(old / f'seed_{seed}', dev, test, seed) for seed in config['seeds']]
    missing = [str(folder / 'checkpoint.pt') for folder, _, _, _ in folds + formal
               if not (folder / 'checkpoint.pt').is_file()]
    if missing:
        raise RuntimeError('Existing checkpoints required; no automatic retraining:\n' + '\n'.join(missing))
    pretrained = json.loads((model_path / 'model_manifest.json').read_text('utf-8'))
    if pretrained['model_revision'] != config['model_revision']:
        raise RuntimeError('Pretrained revision changed')
    print('STAGE verify pretrained files and eight existing checkpoints', flush=True)
    for rel, expected in pretrained['files'].items():
        if digest(model_path / rel) != expected:
            raise RuntimeError('Pretrained file changed: ' + rel)
    sources = [root / name for name in ('config.json', 'asset_manifest.json',
               'CARE_run.py', 'care_model.py', 'care_data.py')]
    sources += [Path(__file__).resolve(), Path(__file__).resolve().parent / 'CARE_resume_decode.py']
    binding = {'version': VERSION, 'sources': {p.name: digest(p) for p in sources},
               'model_manifest_sha256': digest(model_path / 'model_manifest.json'),
               'grid': list(GRID), 'checkpoint_sha256': {}, 'test_gold_access': False}
    for folder, train, valid, seed in folds + formal:
        b = json.loads((folder / 'binding.json').read_text('utf-8'))
        if (b['train_sha256'] != row_digest(train) or b['valid_sha256'] != row_digest(valid)
                or b['seed'] != seed or b['epochs'] != 100
                or b['config_sha256'] != digest(root / 'config.json')
                or b['asset_manifest_sha256'] != digest(root / 'asset_manifest.json')
                or b['model_manifest_sha256'] != binding['model_manifest_sha256']):
            raise RuntimeError('Frozen checkpoint/input binding differs: ' + str(folder))
        print('Verifying checkpoint:', folder.relative_to(old), flush=True)
        binding['checkpoint_sha256'][str(folder.relative_to(old))] = digest(folder / 'checkpoint.pt')
    out = root / 'results_calibrated_v1'
    out.mkdir(exist_ok=True)
    # Linux advisory lock releases automatically even after a forced interruption.
    import fcntl
    with (out / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Calibration is already running')
        bp = out / 'binding.json'
        if bp.exists() and json.loads(bp.read_text('utf-8')) != binding:
            raise RuntimeError('Existing calibration fingerprint differs; do not overwrite it')
        atomic_json(bp, binding)
        import torch
        import numpy as np
        from care_model import ChineseCARE, tokenizer_for, encode, probabilities
        if not torch.cuda.is_available():
            raise RuntimeError('Use the existing CUDA server environment')
        device = torch.device('cuda')
        tokenizer = tokenizer_for(model_path)
        types, relations, legal = assets()

        def load_model(folder):
            saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
            if saved['binding'] != json.loads((folder / 'binding.json').read_text('utf-8')) or saved['epoch'] != 100:
                raise RuntimeError('Checkpoint is incomplete or binding differs')
            model = ChineseCARE(model_path, config)
            model.load_state_dict(saved['model'], strict=True)
            del saved
            model.to(device).eval()
            return model

        def sparse(model, row):
            with torch.inference_mode():
                inputs, mask, offsets = encode(tokenizer, row['text'], config['max_sequence_length'], device)
                probs = probabilities(model(inputs, mask), mask)
                result = {'offsets': np.asarray(offsets, dtype=np.int32)}
                for i, p in enumerate(probs):
                    p = p.squeeze(2)
                    if not torch.isfinite(p).all().item():
                        raise RuntimeError('Nonfinite probability: ' + row['record_id'])
                    coords = (p >= GRID[0]).nonzero()
                    values = p[coords[:, 0], coords[:, 1], coords[:, 2]]
                    result[f'coords{i}'] = coords.cpu().numpy().astype(np.int32)
                    result[f'values{i}'] = values.cpu().numpy()
                return result

        def decode(row, values, thresholds):
            hits = [values[f'coords{i}'][values[f'values{i}'] >= threshold].tolist()
                    for i, threshold in enumerate(thresholds)]
            return dict(record_id=row['record_id'], ann_id=row.get('ann_id'),
                        **indexed_decode(row['text'], values['offsets'].tolist(), *hits, types, relations, legal))

        selection_path = out / 'threshold_selection.json'
        if not selection_path.exists():
            print('STAGE development-only calibration: 100 held-out records, 36 fixed threshold pairs', flush=True)
            totals = {(a, b): [[0, 0, 0], [0, 0, 0]] for a in GRID for b in GRID}
            cache = out / 'development_probability_cache'
            cache.mkdir(exist_ok=True)
            started = time.monotonic()
            done = 0
            for fold, (folder, _, rows, _) in enumerate(folds):
                model = None
                for row in rows:
                    cp = cache / (row['record_id'] + '.npz')
                    if cp.exists():
                        with np.load(cp, allow_pickle=False) as data:
                            values = {k: data[k] for k in data.files}
                    else:
                        if model is None:
                            model = load_model(folder)
                        values = sparse(model, row)
                        fd, tmp = tempfile.mkstemp(dir=cache, suffix='.npz')
                        os.close(fd)
                        try:
                            np.savez_compressed(tmp, **values)
                            os.replace(tmp, cp)
                        finally:
                            if os.path.exists(tmp):
                                os.unlink(tmp)
                    for pair, counts in totals.items():
                        pred = decode(row, values, pair)
                        # Same strict folded-set metric as the frozen evaluator.
                        for i, (p, g) in enumerate(((set(map(tuple, pred['folded_entities'])), gold_entities(row)),
                                                   (set(map(tuple, pred['relations'])), gold_edges(row)))):
                            c = (len(p & g), len(p - g), len(g - p))
                            for j in range(3):
                                counts[i][j] += c[j]
                    done += 1
                    progress(f'development fold={fold+1}/5', done, 100, started)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            results = [{'thresholds': list(pair), 'entity': prf(c[0]), 'relation': prf(c[1])}
                       for pair, c in totals.items()]
            base, best, improved = choose(results)
            expected = [[0, 0, 0], [0, 0, 0]]
            for folder, _, _, _ in folds:
                scores = json.loads((folder / 'complete.json').read_text('utf-8'))['scores']
                for i, task in enumerate(('entity', 'relation')):
                    for j, key in enumerate(('tp', 'fp', 'fn')):
                        expected[i][j] += scores[task][key]
            if totals[(0.5, 0.5)] != expected:
                raise RuntimeError('Recomputed 0.5 development counts differ from frozen CV; stop before test')
            selection = {'version': VERSION, 'binding_sha256': digest(bp), 'baseline': base,
                         'selected': best, 'development_gate_passed': improved, 'candidates': results,
                         'selection': 'Max mean task F1 among pairs improving both pooled out-of-fold task F1s',
                         'limitation': 'OOF scores used for tuning, not an independent estimate of tuned performance',
                         'test_gold_access': False}
            atomic_json(selection_path, selection)
        selection = json.loads(selection_path.read_text('utf-8'))
        if selection['binding_sha256'] != digest(bp):
            raise RuntimeError('Selection fingerprint differs')
        print('DEVELOPMENT RESULT:', json.dumps({k: selection[k] for k in ('baseline', 'selected', 'development_gate_passed')},
                                              ensure_ascii=False), flush=True)
        if not selection['development_gate_passed']:
            print('No pair improves both development tasks. Stopped without new test predictions or retraining.', flush=True)
        else:
            thresholds = selection['selected']['thresholds']
            audit = {}
            for folder, _, rows, seed in formal:
                print(f'STAGE FORMAL inference-only seed={seed} thresholds={thresholds}', flush=True)
                dst = out / f'seed_{seed}'
                dst.mkdir(exist_ok=True)
                rowcache = dst / 'record_cache'
                rowcache.mkdir(exist_ok=True)
                prediction_path = dst / 'predictions.jsonl'
                complete_path = dst / 'complete.json'
                if complete_path.exists():
                    complete = json.loads(complete_path.read_text('utf-8'))
                    if (complete['sha256'] != digest(prediction_path) or complete['thresholds'] != thresholds
                            or complete['seed'] != seed or complete['records'] != len(rows)):
                        raise RuntimeError('Completed prediction fingerprint differs')
                    audit[str(seed)] = complete
                    continue
                started = time.monotonic()
                model = None
                temporary = dst / 'predictions.tmp'
                with temporary.open('w', encoding='utf-8') as stream:
                    for i, row in enumerate(rows, 1):
                        cp = rowcache / (row['record_id'] + '.json')
                        if cp.exists():
                            pred = json.loads(cp.read_text('utf-8'))
                        else:
                            if model is None:
                                model = load_model(folder)
                            pred = decode(row, sparse(model, row), thresholds)
                            indexed_audit([pred], [row])
                            atomic_json(cp, pred)
                        indexed_audit([pred], [row])
                        stream.write(json.dumps(pred, ensure_ascii=False) + '\n')
                        progress(f'formal seed={seed}', i, len(rows), started)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, prediction_path)
                complete = {'complete': True, 'records': len(rows), 'sha256': digest(prediction_path),
                            'thresholds': thresholds, 'seed': seed, 'test_gold_access': False}
                atomic_json(complete_path, complete)
                audit[str(seed)] = complete
                for p in rowcache.glob('*.json'):
                    p.unlink()
                rowcache.rmdir()
                del model
                gc.collect()
                torch.cuda.empty_cache()
            atomic_json(out / 'formal_audit.json', audit)
            print('All three calibrated formal predictions complete. Original results unchanged.', flush=True)
        package = root.parent / 'CARE_Calibrated_results.tar.gz'
        temporary = package.with_suffix('.tmp')
        print('STAGE package final outputs (no checkpoints or probability cache)', flush=True)
        with tarfile.open(temporary, 'w:gz') as archive:
            for path in out.rglob('*'):
                if path.is_file() and path.suffix in ('.json', '.jsonl') and 'record_cache' not in path.parts:
                    archive.add(path, arcname='results_calibrated_v1/' + str(path.relative_to(out)))
            for path in sources:
                archive.add(path, arcname='code/' + path.name)
        os.replace(temporary, package)
        atomic_json(package.with_suffix('.sha256.json'), {'file': package.name, 'sha256': digest(package)})
        print('DOWNLOAD:', package, flush=True)


if __name__ == '__main__':
    main()
