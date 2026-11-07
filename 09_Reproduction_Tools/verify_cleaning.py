"""Compare deterministic cleaning of all 488 raw books with the frozen clean books."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, time
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'reproduced/cleaning_verification.json');a=ap.parse_args()
 base=ROOT/'02_Experiments/01_Corpus_Preparation';src=base/'01_Code/02_运行488部古籍深度清洗.py';spec=importlib.util.spec_from_file_location('cleaning',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 raw=base/'02_Input_Data/488_Original_Books';clean=base/'04_Results/488_Cleaned_Books';files=sorted(raw.glob('*.txt'));issues=[];rows=[];start=time.monotonic()
 for i,p in enumerate(files,1):
  got,_=m.清洗(m.读取(p));expected=m.读取(clean/p.name);ok=got==expected
  rows.append({'file':p.name,'text_matches':ok,'recleaned_sha256':hashlib.sha256(got.encode()).hexdigest(),'frozen_text_sha256':hashlib.sha256(expected.encode()).hexdigest()})
  if not ok:issues.append(p.name)
  if i%25==0 or i==len(files):print(f'Cleaning verification: {i}/{len(files)}; mismatches={len(issues)}; elapsed={time.monotonic()-start:.1f}s',flush=True)
 report={'generated_at':datetime.now(timezone.utc).isoformat(),'books':len(files),'all_match':len(files)==488 and not issues,'mismatches':issues,'files':rows,'network_calls':0,'comparison':'decoded UTF-8 text after original reader; transport CRLF and BOM are not scientific differences'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return 0 if report['all_match'] else 1
if __name__=='__main__':raise SystemExit(main())
