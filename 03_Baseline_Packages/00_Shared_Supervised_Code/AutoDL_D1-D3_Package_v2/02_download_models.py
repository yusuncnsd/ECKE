import hashlib,json
from pathlib import Path
from transformers import AutoModel,AutoTokenizer
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parent; MODEL_ROOT=ROOT/"models"
MODELS={"bert-base-chinese":"google-bert/bert-base-chinese","chinese-roberta-wwm-ext":"hfl/chinese-roberta-wwm-ext"}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result={}
    for alias,repo in MODELS.items():
        target=MODEL_ROOT/alias; target.mkdir(parents=True,exist_ok=True)
        try: snapshot_download(repo_id=repo,local_dir=target)
        except Exception:
            print(f"官方端点下载失败，改用镜像下载同一仓库：{repo}")
            snapshot_download(repo_id=repo,local_dir=target,endpoint="https://hf-mirror.com")
        tok=AutoTokenizer.from_pretrained(target); model=AutoModel.from_pretrained(target)
        tok.save_pretrained(target); model.save_pretrained(target,safe_serialization=True)
        result[alias]={"repository":repo,"files":{str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}}
        print(f"已冻结：{repo} -> {target}")
    (MODEL_ROOT/"pretrained_model_manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__": main()
