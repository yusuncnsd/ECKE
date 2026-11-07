"""CARE Chinese low-resource adaptation: balanced loss; frozen development-only selection."""
import argparse
import gc
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from care_data import ROOT, DATA, assets, read_jsonl, atomic_json, append_jsonl, verify_assets, decode, cv_scores, audit_predictions


def fingerprint(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def progress(stage, done, total, started):
    elapsed = time.monotonic() - started
    eta = elapsed / done * (total - done) if done else 0
    print(f'[{stage}] total={total} done={done} success={done} failed=0 pending={total-done} '
          f'{100*done/max(1,total):.1f}% elapsed={elapsed:.1f}s eta={eta:.1f}s', flush=True)


def select_lengths(rows, count):
    ordered = sorted(rows, key=lambda row: (len(row['text']), row['record_id']))
    if len(rows) < count or count < 2:
        raise RuntimeError('Not enough development records')
    return [ordered[round(i * (len(ordered)-1)/(count-1))] for i in range(count)]


def ready_for_full(result):
    return (result.get('complete') is True and not result.get('skipped_ids')
            and all(sum(result.get(group, {}).get(task, {}).get(key, 0) for key in ('tp', 'fp')) > 0
                    for group in ('training_scores', 'scores') for task in ('entity', 'relation')))


def accept_quick_evidence(output, current_binding, config):
    candidates = [output / 'quick_gate.json', ROOT / 'results/quick_gate.json']
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise RuntimeError('Completed quick check not found; run quick first')
    gate = json.loads(source.read_text('utf-8'))
    expected_legacy = dict(current_binding, **config['legacy_quick_binding'])
    if gate['binding'] not in (current_binding, expected_legacy):
        raise RuntimeError('Quick fingerprint does not match the known old/new version')
    if not ready_for_full(gate['result']):
        raise RuntimeError('Quick run is incomplete or has empty training/development output')
    training = select_lengths(read_jsonl(DATA / 'cv_fold0_train.jsonl'), 6)
    validation = select_lengths(read_jsonl(DATA / 'cv_fold0_valid.jsonl'), 4)
    selected = json.loads((source.parent / 'quick_selection.json').read_text('utf-8'))
    if selected['train'] != [r['record_id'] for r in training] or selected['validation'] != [r['record_id'] for r in validation]:
        raise RuntimeError('Quick subset differs from the fixed input-length selection')
    predictions = read_jsonl(source.parent / 'quick/predictions.jsonl')
    audit_predictions(predictions, validation)
    if cv_scores(predictions, validation) != gate['result']['scores']:
        raise RuntimeError('Stored quick score differs from actual frozen predictions')
    atomic_json(output / 'quick_acceptance.json', {
        'accepted': True, 'previous_passed_flag': gate['passed'], 'source': str(source),
        'source_sha256': fingerprint(source), 'prior_binding': gate['binding'], 'current_binding': current_binding,
        'criterion': config['quick_criterion'], 'reason': 'User explicitly accepts low precision; no training/threshold/evaluation change'})
    print('Completed quick check accepted; no quick retraining. Low precision remains recorded.', flush=True)


def initialize(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, payload, torch):
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def self_test():
    import torch
    from types import SimpleNamespace
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from care_model import ChineseCARE, author, encode, labels, balanced_loss, probabilities
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    author.device = device
    config = json.loads((ROOT / 'config.json').read_text('utf-8'))
    with tempfile.TemporaryDirectory(prefix='care_check_', dir=ROOT) as temporary:
        path = Path(temporary)
        (path / 'vocab.txt').write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '消', '渴', '用', '人', '参', '。']), encoding='utf-8')
        tok = BertTokenizerFast(vocab_file=str(path / 'vocab.txt'))
        tok.save_pretrained(path)
        BertModel(BertConfig(vocab_size=11, hidden_size=24, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=48)).save_pretrained(path)
        tiny = dict(config, share_hidden_size=8, dist_emb_size=4, dropout=0.0)
        model = ChineseCARE(path, tiny).to(device)
        row = {'record_id': 'synthetic', 'text': '消渴用人参。', 'span_entities': [
            {'surface_form': '消渴', 'entity_type': '疾病', 'start': 0, 'end': 2},
            {'surface_form': '人参', 'entity_type': '中药', 'start': 3, 'end': 5}], 'relations': [['人参', '治疗', '消渴']]}
        inputs, mask, offsets = encode(tok, row['text'], 512, device)
        gold = labels(row, offsets, device)
        assert gold[0].sum().item() == 2 and gold[1].sum().item() == 1
        model.train()
        logits = model(inputs, mask)
        loss = balanced_loss(logits, gold, mask)
        loss.backward()
        for parameter in (model.bert.embeddings.word_embeddings.weight, model.ner.hid2tag.weight, model.re.hid2rel.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
        # The classifier layers are identical to the author; only sigmoid/masking
        # move to inference. Check exact probability parity using shared weights.
        hidden = torch.randn(8, 1, 24, device=device)
        shared = torch.randn(8, 8, 1, 8, device=device)
        args = SimpleNamespace(dropout=0.0)
        types, relations, _ = assets()
        reference_ner = author.ner_unit(args, dict(enumerate(types)), 24, 8).to(device)
        reference_re = author.re_unit(args, dict(enumerate(relations)), 24, 8).to(device)
        reference_ner.load_state_dict(model.ner.state_dict())
        reference_re.load_state_dict(model.re.state_dict())
        model.eval()
        with torch.no_grad():
            raw = (model.ner(hidden, shared, mask.T), model.re(hidden, shared, mask.T))
            actual = probabilities(raw, mask)
            expected = (reference_ner(hidden, shared, mask.T), reference_re(hidden, shared, mask.T))
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(actual, expected))
        import copy
        reference_model = copy.deepcopy(model)
        reference_model.ner.__class__ = author.ner_unit
        reference_model.re.__class__ = author.re_unit
        reference_model.tokenizer = tok
        all_tokens = torch.ones_like(mask)
        positions = torch.arange(len(offsets), device=device)
        delta = positions[:, None] - positions[None, :]
        bins = torch.zeros(4000, dtype=torch.long, device=device)
        bins[1] = 1
        for boundary, value in ((2, 2), (4, 3), (8, 4), (16, 5), (32, 6), (64, 7), (128, 8), (256, 9)):
            bins[boundary:] = value
        distances = bins[delta.abs()]
        distances[delta < 0] += 9
        distances[delta == 0] = 19
        with torch.no_grad():
            actual = probabilities(model(inputs, all_tokens), all_tokens)
            expected = author.CARE.forward(reference_model, list(row['text']), all_tokens.T, distances.unsqueeze(0))
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(actual, expected))
        # Extreme logits must yield finite stable loss and gradients.
        extreme = tuple(torch.full_like(y, 1000.0, requires_grad=True) for y in gold)
        stable = balanced_loss(extreme, gold, mask)
        stable.backward()
        assert torch.isfinite(stable) and all(torch.isfinite(x.grad).all() for x in extreme)
    print('PASS: label alignment, stable loss, encoder/head gradients, author classifier and full-forward parity', flush=True)


def evaluate(model, rows, tokenizer, device, config, output=None, allow_skip=False):
    import torch
    from care_model import encode, probabilities
    types, relations, legal = assets()
    model.eval()
    predictions = []
    skipped = []
    started = time.monotonic()
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            inputs, mask, offsets = encode(tokenizer, row['text'], 512, device)
            probs = probabilities(model(inputs, mask), mask)
            if not all(torch.isfinite(p).all().item() for p in probs):
                raise RuntimeError('Nonfinite prediction: ' + row['record_id'])
            hits = [(p.squeeze(2) >= config['threshold']).nonzero().cpu().tolist() for p in probs]
            starts = {}
            for start, end, label in hits[0]:
                starts[start] = starts.get(start, 0) + 1
            work = sum(starts.get(head, 0) * starts.get(tail, 0) for head, tail, label in hits[1])
            if work > 1000000:
                if not allow_skip:
                    raise RuntimeError('Excessive prediction expansion; not truncated: ' + row['record_id'])
                skipped.append(row['record_id'])
                continue
            prediction = dict(record_id=row['record_id'], ann_id=row.get('ann_id'),
                              **decode(row['text'], offsets, hits[0], hits[1], types, relations, legal))
            predictions.append(prediction)
            if index % 20 == 0 or index == len(rows):
                progress('prediction', index, len(rows), started)
    result = {'records': len(rows), 'skipped_ids': skipped,
              'scores': None if skipped or not all('span_entities' in row for row in rows) else cv_scores(predictions, rows)}
    if not skipped:
        audit_predictions(predictions, rows)
    if output is not None:
        if skipped:
            raise RuntimeError('Incomplete prediction must not be committed')
        # Completed predictions are committed atomically; interrupted inference is
        # rerun from the saved training checkpoint without another training run.
        temporary = output.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in predictions), encoding='utf-8')
        os.replace(temporary, output)
    return result


def train(train_rows, valid_rows, seed, epochs, folder, model_path, config, global_binding, quick=False):
    import torch
    from care_model import ChineseCARE, tokenizer_for, encode, labels, balanced_loss
    folder.mkdir(parents=True, exist_ok=True)
    binding = dict(global_binding, seed=seed, epochs=epochs,
                   train_sha256=hashlib.sha256(json.dumps(train_rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                   valid_sha256=hashlib.sha256(json.dumps(valid_rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
    binding_file = folder / 'binding.json'
    if binding_file.exists() and json.loads(binding_file.read_text('utf-8')) != binding:
        raise RuntimeError('Fingerprint differs; choose a new result directory')
    atomic_json(binding_file, binding)
    if (folder / 'complete.json').exists():
        audit_predictions(read_jsonl(folder / 'predictions.jsonl'), valid_rows)
        return json.loads((folder / 'complete.json').read_text('utf-8'))
    initialize(seed)
    device = torch.device('cuda')
    tokenizer = tokenizer_for(model_path)
    model = ChineseCARE(model_path, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    checkpoint = folder / 'checkpoint.pt'
    start_epoch = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['binding'] != binding:
            raise RuntimeError('Checkpoint fingerprint differs')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_epoch = saved['epoch']
        del saved
    started = time.monotonic()
    try:
        for epoch in range(start_epoch, epochs):
            order = list(train_rows)
            random.Random(seed + epoch).shuffle(order)
            for cursor in range(0, len(order), config['effective_batch_size']):
                model.train()
                batch = order[cursor:cursor + config['effective_batch_size']]
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.0
                for row in batch:
                    inputs, mask, offsets = encode(tokenizer, row['text'], 512, device)
                    logits = model(inputs, mask)
                    loss = balanced_loss(logits, labels(row, offsets, device), mask) / len(batch)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError('Nonfinite loss: ' + row['record_id'])
                    loss.backward()
                    loss_value += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'], error_if_nonfinite=True)
                optimizer.step()
                append_jsonl(folder / 'training_log.jsonl', {'epoch': epoch+1, 'done_in_epoch': cursor+len(batch),
                    'loss': loss_value, 'gradient_norm_before_clip': float(norm)})
                progress(f'train seed={seed} epoch={epoch+1}/{epochs} loss={loss_value:.4f}',
                         (epoch-start_epoch)*len(order)+cursor+len(batch), (epochs-start_epoch)*len(order), started)
            # Restart repeats at most five epochs; input/config/RNG-bound checkpoints.
            if (epoch+1) % 5 == 0 or epoch+1 == epochs:
                save_checkpoint(checkpoint, {'binding': binding, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all()}, torch)
            if quick and ((epoch+1) % 10 == 0 or epoch+1 == epochs):
                score = evaluate(model, valid_rows, tokenizer, device, config, allow_skip=True)
                append_jsonl(folder / 'development_metrics.jsonl', dict(epoch=epoch+1, **score))
                print('Held-out development scores:', json.dumps(score, ensure_ascii=False), flush=True)
        training_scores = None
        if quick or not all('span_entities' in row for row in valid_rows):
            training_scores = evaluate(model, train_rows, tokenizer, device, config)['scores']
            atomic_json(folder / 'training_fit.json', training_scores)
            if not quick and any(training_scores[task]['tp'] + training_scores[task]['fp'] == 0 for task in ('entity', 'relation')):
                raise RuntimeError('Full model has empty training predictions; stopped before test prediction')
        result = evaluate(model, valid_rows, tokenizer, device, config, folder / 'predictions.jsonl')
        if training_scores is not None:
            result['training_scores'] = training_scores
        result.update(complete=True, seed=seed, epochs=epochs, generated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(folder / 'complete.json', result)
        return result
    except Exception as error:
        atomic_json(folder / 'failure.json', {'exception': type(error).__name__, 'message': str(error), 'complete': False})
        raise
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['check', 'quick', 'full', 'package'])
    parser.add_argument('--model', type=Path, default=ROOT.parent / 'AutoDL_CARE/models/bert-base-chinese')
    parser.add_argument('--output', type=Path, default=ROOT / 'results_fixed100')
    args = parser.parse_args()
    if args.phase == 'check':
        verify_assets()
        self_test()
        return
    if args.phase == 'package':
        import tarfile
        if not (args.output / 'formal_audit.json').exists():
            raise RuntimeError('Formal run is not complete')
        target = ROOT.parent / 'CARE_Balanced_results.tar.gz'
        with tarfile.open(target, 'w:gz') as archive:
            for path in args.output.rglob('*'):
                if path.is_file() and path.name != 'checkpoint.pt':
                    archive.add(path, arcname='results_fixed100/' + str(path.relative_to(args.output)))
            archive.add(ROOT / 'config.json', arcname='config.json')
            archive.add(ROOT / 'asset_manifest.json', arcname='asset_manifest.json')
            for path in ROOT.glob('*.py'):
                archive.add(path, arcname='code/' + path.name)
        print('Results package (checkpoints remain on server):', target)
        return
    import torch
    import transformers
    config = json.loads((ROOT / 'config.json').read_text('utf-8'))
    verify_assets()
    if args.phase == 'quick':
        self_test()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; reuse the existing AutoDL environment')
    if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise RuntimeError('GPU needs at least 20GiB')
    model_path = args.model.resolve()
    manifest_path = model_path / 'model_manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    if manifest['model_revision'] != config['model_revision']:
        raise RuntimeError('Pretrained revision differs')
    for relative, expected in manifest['files'].items():
        if fingerprint(model_path / relative) != expected:
            raise RuntimeError('Pretrained asset differs: ' + relative)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / 'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        binding = {'version': config['version'], 'config_sha256': fingerprint(ROOT / 'config.json'),
                   'asset_manifest_sha256': fingerprint(ROOT / 'asset_manifest.json'),
                   'model_manifest_sha256': fingerprint(manifest_path),
                   'torch': torch.__version__, 'transformers': transformers.__version__,
                   'gpu': torch.cuda.get_device_name(0), 'test_gold_access': False}
        if args.phase == 'quick':
            train_rows = select_lengths(read_jsonl(DATA / 'cv_fold0_train.jsonl'), 6)
            valid_rows = select_lengths(read_jsonl(DATA / 'cv_fold0_valid.jsonl'), 4)
            if {r['record_id'] for r in train_rows} & {r['record_id'] for r in valid_rows}:
                raise RuntimeError('Quick train/validation overlap')
            atomic_json(args.output / 'quick_selection.json', {'train': [r['record_id'] for r in train_rows],
                'validation': [r['record_id'] for r in valid_rows], 'selection': 'evenly spaced input lengths, ties by ID'})
            result = train(train_rows, valid_rows, config['cv_seed'], config['quick_epochs'],
                           args.output / 'quick', model_path, config, binding, quick=True)
            passed = ready_for_full(result)
            atomic_json(args.output / 'quick_gate.json', {'passed': passed, 'binding': binding,
                'criterion': config['quick_criterion'],
                'result': result})
            print('Quick check result:', json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            print('Ready for full run:', passed, flush=True)
            return
        accept_quick_evidence(args.output, binding, config)
        candidates = []
        for epochs in config['epoch_candidates']:
            scores = []
            for fold in range(5):
                print(f'STAGE CV fold={fold+1}/5 fixed_epochs={epochs}', flush=True)
                result = train(read_jsonl(DATA / f'cv_fold{fold}_train.jsonl'),
                               read_jsonl(DATA / f'cv_fold{fold}_valid.jsonl'), config['cv_seed'], epochs,
                               args.output / f'cv/epochs_{epochs}/fold_{fold}', model_path, config, binding)
                scores.append((result['scores']['entity']['f1'] + result['scores']['relation']['f1'])/2)
            candidates.append({'epochs': epochs, 'fold_joint_f1': scores, 'mean_joint_f1': sum(scores)/5})
        choice = max(candidates, key=lambda row: (row['mean_joint_f1'], -row['epochs']))
        atomic_json(args.output / 'frozen_cv_config.json', {'binding': binding, 'candidates': candidates,
                    'epochs': choice['epochs'], 'selection': 'User-fixed100 epochs; fivefold validation is diagnostic, not epoch tuning'})
        if choice['mean_joint_f1'] == 0:
            print('WARNING: fivefold mean F1 is zero; retained as a development result. User accepts low performance.', flush=True)
        dev = read_jsonl(DATA / 'development_100_span_supervision.jsonl')
        inputs = read_jsonl(DATA / 'independent_test_300_input_only.jsonl')
        audit = {}
        for seed in config['seeds']:
            print(f'STAGE FORMAL seed={seed} train=100 predict=300 epochs={choice["epochs"]}', flush=True)
            train(dev, inputs, seed, choice['epochs'], args.output / f'seed_{seed}', model_path, config, binding)
            path = args.output / f'seed_{seed}/predictions.jsonl'
            audit[str(seed)] = dict(audit_predictions(read_jsonl(path), inputs), sha256=fingerprint(path))
        atomic_json(args.output / 'formal_audit.json', audit)
        print('All three formal predictions complete. No test Gold was accessed.', flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
