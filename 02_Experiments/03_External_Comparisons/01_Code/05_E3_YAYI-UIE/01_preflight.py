from __future__ import annotations
import hashlib, json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
EXPECTED_TEST='FDF183FA85D337EF34DE8CDAFC7C67BCF1DBE78450815E56B5E9602EBCBAEF49'
EXPECTED_SCHEMA='32A37F9F1D210F1059A561460D32701394AAB098D333E52B63B7BEBC90E06DE4'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA不可用')
    mem=torch.cuda.get_device_properties(0).total_memory/1024**3
    if mem<20: raise RuntimeError(f'显存{mem:.1f}GB，小于20GB')
    test=ROOT/'05_Frozen_Data_Indices/independent_test_300_input_only.jsonl'; schema=ROOT/'05_Frozen_Data_Indices/xiaoke_schema_v5.json'
    rows=[json.loads(x) for x in test.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(rows)!=300 or len({x['record_id'] for x in rows})!=300: raise RuntimeError('测试输入不是300个唯一ID')
    if any(set(x)-{'ann_id','record_id','text'} for x in rows): raise RuntimeError('测试输入包含非必要字段')
    if sha(test)!=EXPECTED_TEST or sha(schema)!=EXPECTED_SCHEMA: raise RuntimeError('测试输入或Schema指纹不一致')
    s=json.loads(schema.read_text(encoding='utf-8'))
    if len(s['entity_types'])!=7 or len(s['relation_schemas'])!=7: raise RuntimeError('不是冻结的7实体/7关系Schema')
    print(f'E3预检通过：{torch.cuda.get_device_name(0)}，显存{mem:.1f}GB；300条无答案输入，7实体/7关系。')
if __name__=='__main__': main()
