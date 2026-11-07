from pathlib import Path
import json
from huggingface_hub import HfApi, snapshot_download

ROOT=Path(__file__).resolve().parent; TARGET=ROOT/'model/YAYI-UIE'; REPO='wenge-research/yayi-uie'
def main():
    TARGET.mkdir(parents=True,exist_ok=True)
    api=HfApi(endpoint='https://hf-mirror.com')
    try: revision=api.model_info(REPO).sha
    except Exception: revision='main'
    metadata=['*.json','*.py','*.model','*.txt','README.md']
    path=snapshot_download(REPO,revision=revision,local_dir=TARGET,endpoint='https://hf-mirror.com',allow_patterns=metadata,max_workers=4)
    indexes=list(TARGET.glob('*index.json'))
    if not indexes: raise RuntimeError('未找到YAYI-UIE权重索引，停止以避免下载错误资产')
    index=json.loads(indexes[0].read_text(encoding='utf-8')); shards=sorted(set(index['weight_map'].values()))
    path=snapshot_download(REPO,revision=revision,local_dir=TARGET,endpoint='https://hf-mirror.com',allow_patterns=metadata+shards,max_workers=3)
    (ROOT/'model/model_binding.json').write_text(json.dumps({'repository':REPO,'resolved_revision':revision,'local_path':str(path)},indent=2),encoding='utf-8')
    print('E3模型下载完成：',path,'revision=',revision,'权重分片=',shards)
if __name__=='__main__': main()
