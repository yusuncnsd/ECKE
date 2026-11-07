from __future__ import annotations

import hashlib, importlib.util, json, os, re, time
from pathlib import Path

API密钥 = ""  # 仅运行Gold实体Oracle时由用户手动填写；离线误差传播分解不需要密钥。运行完成后应恢复为空。
HERE=Path(__file__).resolve().parent; MAIN=HERE.parents[1]
GOLD=MAIN/"02_ECKE/02_Input_Data/Independent_Test_300/消渴知识抽取_300条独立测试集专家标注-盲标定.txt"
MAIN_PRED=MAIN/"02_ECKE/04_Results/Test_Predictions/independent_test_300_machine_predictions_v612_frozen.jsonl"
SCHEMA=MAIN/"02_ECKE/03_Configuration_and_Instructions/xiaoke_schema_v5.json"
V612=MAIN/"02_ECKE/01_Code/01_运行V612全库知识抽取.py"
OUT=MAIN/"02_ECKE/05_Evaluation_and_Verification/复现运行/Gold_Entity_Oracle_and_Error_Propagation"
CACHE=OUT/"oracle_relation_cache.jsonl"; FAIL=OUT/"oracle_failures.jsonl"

def sec(b,h,hs):
 a="|".join(re.escape(x) for x in hs);m=re.search(rf"(?ms)^{re.escape(h)}[ \t]*\r?\n(.*?)(?=^(?:{a})[ \t]*\r?$|^[ \t]*={{20,}}[ \t]*\r?$|\Z)",b);return m.group(1).strip()
def vals(s,n):
 if not s or s=="无":return set()
 out=set()
 for line in s.splitlines():
  if not line.strip():continue
  p=tuple(x.strip() for x in re.split(r"[|｜]",line));
  if len(p)!=n:raise RuntimeError(f"Gold格式错误：{line}")
  out.add(p)
 return out
def gold():
 t=GOLD.read_bytes().decode("utf-8-sig");ss=list(re.finditer(r"(?m)^ann_id:\s*test_\d+\s*$",t));hs=("【原文】","【盲标实体】","【盲标关系】");out=[]
 for i,m in enumerate(ss):
  b=t[m.start():ss[i+1].start() if i+1<len(ss) else len(t)];out.append({"ann_id":re.search(r"(?m)^ann_id:\s*(\S+)",b).group(1),"record_id":re.search(r"(?m)^paragraph_id:\s*(\S+)",b).group(1),"text":sec(b,hs[0],hs),"entities":vals(sec(b,hs[1],hs),2),"relations":vals(sec(b,hs[2],hs),3)})
 return out
def prediction():
 out={}
 for r in [json.loads(x) for x in MAIN_PRED.read_text(encoding="utf-8").splitlines() if x.strip()]:
  out[r["record_id"]]={"entities":{(e["surface_form"],e["entity_type"]) for e in r["entities"]},"relations":{(x["head_surface"],x["relation_type"],x["tail_surface"]) for x in r["relations"]}}
 return out
def prf(p,g):
 tp=len(p&g);fp=len(p-g);fn=len(g-p);prec=tp/(tp+fp) if tp+fp else (1 if fn==0 else 0);rec=tp/(tp+fn) if tp+fn else (1 if fp==0 else 0);f=2*prec*rec/(prec+rec) if prec+rec else 0;return {"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f}
def offline(rows,pred):
 d={"gold_relation_fn_entity_endpoint_missing":0,"gold_relation_fn_endpoints_present_but_relation_missed":0,"pred_relation_fp_non_gold_endpoint":0,"pred_relation_fp_gold_endpoints_but_semantically_wrong":0,"details":[]}
 for g in rows:
  p=pred[g["record_id"]];fn=g["relations"]-p["relations"];fp=p["relations"]-g["relations"];detail={"ann_id":g["ann_id"],"record_id":g["record_id"],"fn_entity_stage":[],"fn_relation_stage":[],"fp_entity_stage":[],"fp_relation_stage":[]}
  for x in fn:
   key="fn_entity_stage" if not any(e[0]==x[0] for e in p["entities"]) or not any(e[0]==x[2] for e in p["entities"]) else "fn_relation_stage";detail[key].append(list(x));d["gold_relation_fn_entity_endpoint_missing" if key=="fn_entity_stage" else "gold_relation_fn_endpoints_present_but_relation_missed"]+=1
  gs={e[0] for e in g["entities"]}
  for x in fp:
   key="fp_entity_stage" if x[0] not in gs or x[2] not in gs else "fp_relation_stage";detail[key].append(list(x));d["pred_relation_fp_non_gold_endpoint" if key=="fp_entity_stage" else "pred_relation_fp_gold_endpoints_but_semantically_wrong"]+=1
  if any(detail.values()):d["details"].append(detail)
 return d
def load_module():
 spec=importlib.util.spec_from_file_location("v612_oracle",V612);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.API密钥=API密钥;return m
def gold_entities(g):
 out=[]
 for i,(surface,typ) in enumerate(sorted(g["entities"]),1):
  start=g["text"].find(surface)
  if start<0:raise RuntimeError(f"{g['ann_id']} Gold实体无法逐字定位：{surface}|{typ}")
  out.append({"entity_id":f"{g['record_id']}_G{i:03d}","surface_form":surface,"normalized_concept":surface,"entity_type":typ,"start":start,"end":start+len(surface),"evidence_text":surface,"recognition_source":"expert_gold"})
 return out
def cache_map():
 out={}
 if CACHE.exists():
  for line in CACHE.read_text(encoding="utf-8").splitlines():
   if line.strip():
    x=json.loads(line)
    if x.get("status")=="success":out[x["record_id"]]=x
 return out
def append(path,x):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("a",encoding="utf-8",newline="\n") as f:f.write(json.dumps(x,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
def run_oracle(rows):
 if not API密钥.strip():raise RuntimeError("离线误差传播分解已完成；如需正式Gold实体Oracle，请先在脚本顶部填写API密钥后重新运行。")
 m=load_module();schemas=json.loads(SCHEMA.read_text(encoding="utf-8"))["relation_schemas"];done=cache_map();pending=[g for g in rows if g["record_id"] not in done];started=time.time()
 for i,g in enumerate(pending,1):
  try:
   entities=gold_entities(g);c=m.relation_candidates(entities,schemas,[])
   if c:
    response,_,usage=m.api_call(m.relation_messages(g,entities,c),seed_offset=300000+int(g["ann_id"].split("_")[-1]));relations=m.validate_relations(g,entities,c,response)
   else:relations=[];usage={}
   x={"record_id":g["record_id"],"ann_id":g["ann_id"],"status":"success","entities":[[e["surface_form"],e["entity_type"]] for e in entities],"relations":[[r["head_surface"],r["relation_type"],r["tail_surface"]] for r in relations],"usage":usage};append(CACHE,x);done[g["record_id"]]=x
  except Exception as e:append(FAIL,{"record_id":g["record_id"],"error":str(e),"time":time.strftime("%F %T")})
  elapsed=time.time()-started;print(f"[Gold实体Oracle] {i}/{len(pending)} 成功={len(done)} 待处理={len(rows)-len(done)} 已用={elapsed:.1f}s",flush=True)
 if len(done)!=len(rows):raise RuntimeError(f"Oracle未全部完成：{len(done)}/{len(rows)}；再次运行只处理未完成项")
 return done
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=gold();pred=prediction();off=offline(rows,pred);(OUT/"端到端关系误差传播分解.json").write_text(json.dumps(off,ensure_ascii=False,indent=2),encoding="utf-8")
 lines=["端到端关系误差传播分解","="*80,f"Gold关系漏检：因预测实体端点缺失={off['gold_relation_fn_entity_endpoint_missing']}；端点齐全但关系漏判={off['gold_relation_fn_endpoints_present_but_relation_missed']}",f"预测关系误报：含非Gold端点={off['pred_relation_fp_non_gold_endpoint']}；端点均为Gold但语义误判={off['pred_relation_fp_gold_endpoints_but_semantically_wrong']}","说明：这是严格离线归因，不冒充Gold实体Oracle。"]
 (OUT/"端到端关系误差传播分解.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print(OUT/"端到端关系误差传播分解.txt")
 if not API密钥.strip():
  print("Gold实体Oracle尚需API密钥；脚本已完成全部离线结果并安全退出。")
  return
 done=run_oracle(rows);tp=fp=fn=0;exact=0
 for g in rows:
  p={tuple(x) for x in done[g["record_id"]]["relations"]};x=prf(p,g["relations"]);tp+=x["tp"];fp+=x["fp"];fn+=x["fn"];exact+=p==g["relations"]
 total=prf(set(range(tp)),set(range(tp+fn))) if fp==0 else {"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp),"recall":tp/(tp+fn),"f1":2*(tp/(tp+fp))*(tp/(tp+fn))/((tp/(tp+fp))+(tp/(tp+fn))) if tp else 0}
 result={"condition":"Gold实体输入；V6.1.2正式关系模块不变","relation":total,"passage_exact":exact/len(rows),"records":len(rows)};(OUT/"Gold实体Oracle正式结果.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");(OUT/"Gold实体Oracle正式结果.txt").write_text(f"Gold实体Oracle正式结果\n关系 P={total['precision']:.4f} R={total['recall']:.4f} F1={total['f1']:.4f} (TP={tp}, FP={fp}, FN={fn})\n条文完全正确率={exact/len(rows):.4f}\n",encoding="utf-8")
if __name__=="__main__":main()
