"""Rebuild deterministic passage units and candidate gates only; never call an API.

Historical LLM screening decisions are not regenerated or invented.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, time
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'reproduced/screening');args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
 src=ROOT/'02_Experiments/01_Corpus_Preparation/01_Code/03_运行消渴主题条文筛选.py'
 spec=importlib.util.spec_from_file_location('historical_screening',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 def blocked(*a,**kw):raise RuntimeError('API calls prohibited during offline screening verification')
 m.调用模型=blocked;m.API连通性预检=blocked
 m.证据单元缓存=out/'reconstructed_units.jsonl';m.证据单元清单=out/'reconstructed_manifest.json';m.候选缓存=out/'candidates.jsonl';m.候选清单=out/'candidate_manifest.json'
 started=time.monotonic();manifest=m.输入指纹();units=m.重建全部条文(manifest);candidates=m.生成候选(units,m.读取词表xlsx(m.词表路径),manifest)
 report={'generated_at':datetime.now(timezone.utc).isoformat(),'books':len(manifest['files']),'reconstructed_units':len(units),'candidate_passages':len(candidates),'manuscript_candidate_count':5924,'candidate_count_matches':len(candidates)==5924,'historical_LLM_responses_available':False,'preliminary_3230_independently_verified':False,'final_3065_from_frozen_corpus':True,'network_calls':0,'elapsed_seconds':time.monotonic()-started,'code_sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
 (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False),flush=True)
 return 0 if report['candidate_count_matches'] else 1
if __name__=='__main__':raise SystemExit(main())
