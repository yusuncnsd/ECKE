"""仅在用户运行时下载官方权重；锁定实际commit，无main回退。"""
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = 'THU-KEG/ADELIE-DPO'

def main(endpoint):
    from huggingface_hub import HfApi, snapshot_download
    target = ROOT / 'model/ADELIE-DPO'
    binding_path = ROOT / 'model/model_binding.json'
    api = HfApi(endpoint=endpoint)
    if binding_path.exists():
        previous = json.loads(binding_path.read_text(encoding='utf-8'))
        if previous['repository'] != REPO:
            raise RuntimeError('模型仓库与已有绑定不一致')
        revision = previous['resolved_revision']
    else:
        revision = api.model_info(REPO).sha
        if not revision or len(revision) != 40:
            raise RuntimeError('未解析到完整commit，禁止回退main')
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        binding_path.write_text(json.dumps({'repository': REPO, 'resolved_revision': revision,
                                          'status': 'downloading'}, indent=2), encoding='utf-8')
    info = api.model_info(REPO, revision=revision, files_metadata=True)
    names = [x.rfilename for x in info.siblings]
    if not any(n.endswith('.bin') or n.endswith('.safetensors') for n in names):
        raise RuntimeError('官方仓库未提供完整权重')
    snapshot_download(REPO, revision=revision, local_dir=target, endpoint=endpoint,
                      allow_patterns=['*.json', '*.model', '*.bin', '*.safetensors', '*.txt', 'README.md'],
                      max_workers=3)
    # 某些公开快照只有bin分片、未上传index。仅构建加载索引，不更改权重。
    shards = sorted(target.glob('pytorch_model-*.bin'))
    index_path = target / 'pytorch_model.bin.index.json'
    index_generated = False
    if shards and not index_path.exists():
        import torch
        weight_map, total = {}, 0
        for shard in shards:
            print('校验权重分片并构建索引：', shard.name, flush=True)
            state = torch.load(shard, map_location='cpu', weights_only=True)
            for key, value in state.items():
                if key in weight_map:
                    raise RuntimeError(f'分片键重复：{key}')
                weight_map[key] = shard.name
                total += value.numel() * value.element_size()
            del state
        index_path.write_text(json.dumps({'metadata': {'total_size': total}, 'weight_map': weight_map}, indent=2), encoding='utf-8')
        index_generated = True
    files = {}
    for p in sorted(target.iterdir()):
        if p.is_file():
            print('指纹校验：', p.name, flush=True)
            digest = hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            files[p.name] = {'size': p.stat().st_size, 'sha256': digest.hexdigest()}
    for asset in info.siblings:
        if asset.rfilename in files and getattr(asset, 'lfs', None):
            expected = asset.lfs.sha256
            if expected and files[asset.rfilename]['sha256'] != expected:
                raise RuntimeError(f'官方LFS指纹不匹配：{asset.rfilename}')
    binding_path.write_text(json.dumps({'repository': REPO, 'resolved_revision': revision,
                                      'status': 'complete', 'files': files,
                                      'index_generated': index_generated}, indent=2), encoding='utf-8')
    print('官方权重准备完成，commit=', revision)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='https://huggingface.co')
    main(parser.parse_args().endpoint)
