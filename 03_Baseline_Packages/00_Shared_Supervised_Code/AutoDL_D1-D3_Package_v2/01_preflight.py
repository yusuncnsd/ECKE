import hashlib,json,sys
from pathlib import Path
import torch
from packaging.version import Version

ROOT=Path(__file__).resolve().parent
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA不可用，禁止正式训练")
    if Version(torch.__version__.split('+')[0])<Version("2.1.0"): raise RuntimeError(f"PyTorch版本过低：{torch.__version__}，要求>=2.1")
    props=torch.cuda.get_device_properties(0); memory=props.total_memory/1024**3
    if memory<20: raise RuntimeError(f"显存仅{memory:.1f}GB；本实验包要求至少20GB")
    manifest=json.loads((ROOT/"asset_manifest.json").read_text(encoding="utf-8"))
    paths={"development":ROOT/"05_Frozen_Data_Indices/development_100_gold.jsonl","test_input":ROOT/"05_Frozen_Data_Indices/independent_test_300_input_only.jsonl","schema":ROOT/"05_Frozen_Data_Indices/xiaoke_schema_v5.json"}
    for key,path in paths.items():
        if digest(path)!=manifest["hashes"][key]: raise RuntimeError(f"资产指纹不一致：{key}")
    for group,folder in (("cv_files",ROOT/"05_Frozen_Data_Indices/data"),("code_files",ROOT/"code")):
        for name,expected in manifest["hashes"][group].items():
            if digest(folder/name)!=expected: raise RuntimeError(f"资产指纹不一致：{group}/{name}")
    if manifest["test_gold_included"]: raise RuntimeError("云端包不应包含测试Gold")
    print(f"预检通过：{props.name}，显存{memory:.1f}GB；100条开发数据、300条无答案测试输入。")
if __name__=="__main__": main()
