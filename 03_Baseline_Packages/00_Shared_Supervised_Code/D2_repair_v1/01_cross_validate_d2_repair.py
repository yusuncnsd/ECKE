import json, shutil, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=ROOT/"05_Frozen_Data_Indices/data"
MODELS=ROOT/"models"
RESULTS=ROOT/"results"/"D2_repair_v1"
TRAIN=HERE/"formal_train_d2_repair.py"
EPOCHS=(12,20)
THRESHOLDS=(0.05,0.10,0.20,0.30,0.40,0.50)

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def f1(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
    return 2*p*r/(p+r) if p+r else 0.0

def decode(raw,text,entity_threshold,relation_threshold,types,legal):
    chosen=[]; endpoints={}
    for surface,typ,a,b,span_i,score in raw["entity_candidates"]:
        if score>=entity_threshold:
            chosen.append((surface,typ)); endpoints[(span_i,typ)]=surface
    entities=set(chosen); relations=set()
    for hi,rel,ti,score in raw["relation_candidates"]:
        if score<relation_threshold: continue
        for ht in types:
            hs=endpoints.get((hi,ht))
            if hs is None: continue
            for tt in types:
                ts=endpoints.get((ti,tt))
                if ts is not None and (ht,rel,tt) in legal: relations.add((hs,rel,ts))
    return entities,relations

def score(raw_rows,gold_rows,et,rt,types,legal):
    raw_map={x["record_id"]:x for x in raw_rows}; e=[0,0,0]; r=[0,0,0]
    for g in gold_rows:
        pe,pr=decode(raw_map[g["record_id"]],g["text"],et,rt,types,legal)
        ge={(x["surface_form"],x["entity_type"]) for x in g["span_entities"]}; gr={tuple(x) for x in g["relations"]}
        for target,pred,gold in ((e,pe,ge),(r,pr,gr)):
            target[0]+=len(pred&gold); target[1]+=len(pred-gold); target[2]+=len(gold-pred)
    ef,rf=f1(*e),f1(*r)
    return {"entity_f1":ef,"relation_f1":rf,"joint":(ef+rf)/2}

def main():
    schema=json.loads((DATA/"xiaoke_schema_v5.json").read_text(encoding="utf-8")); types=schema["entity_types"]
    legal={(x["head_type"],x["relation"],x["tail_type"]) for x in schema["relation_schemas"]}
    RESULTS.mkdir(parents=True,exist_ok=True); progress_path=RESULTS/"cv_progress.json"
    all_results=json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else []
    for epochs in EPOCHS:
        for fold in range(5):
            if len([x for x in all_results if x["epochs"]==epochs and x["fold"]==fold])==len(THRESHOLDS)**2: continue
            run_dir=RESULTS/"D2"/f"seed_20260823_cv_e{epochs}_f{fold}"
            score_path=run_dir/"d2_score_cache.jsonl"
            if not score_path.exists():
                cmd=[sys.executable,str(TRAIN),"D2","--development",f"cv_fold{fold}_train.jsonl","--test",f"cv_fold{fold}_valid.jsonl","--seed","20260823","--epochs",str(epochs),"--result-suffix",f"_cv_e{epochs}_f{fold}","--result-root",str(RESULTS),"--model-root",str(MODELS),"--data-root",str(DATA),"--d2-export-scores"]
                subprocess.run(cmd,check=True)
            raw=read_jsonl(score_path); gold=read_jsonl(DATA/f"cv_fold{fold}_valid.jsonl")
            if len(raw)!=len(gold): raise RuntimeError(f"D2修复版fold数量错误：e{epochs} f{fold} raw={len(raw)} gold={len(gold)}")
            for et in THRESHOLDS:
                for rt in THRESHOLDS:
                    all_results.append({"epochs":epochs,"fold":fold,"entity_threshold":et,"relation_threshold":rt,**score(raw,gold,et,rt,types,legal)})
            shutil.rmtree(run_dir)
            progress_path.write_text(json.dumps(all_results,ensure_ascii=False,indent=2),encoding="utf-8")
    configs=[]
    for epochs in EPOCHS:
        for et in THRESHOLDS:
            for rt in THRESHOLDS:
                rows=[x for x in all_results if x["epochs"]==epochs and x["entity_threshold"]==et and x["relation_threshold"]==rt]
                configs.append({"epochs":epochs,"entity_threshold":et,"relation_threshold":rt,"mean_entity_f1":sum(x["entity_f1"] for x in rows)/5,"mean_relation_f1":sum(x["relation_f1"] for x in rows)/5,"mean_joint":sum(x["joint"] for x in rows)/5})
    best=max(configs,key=lambda x:(x["mean_joint"],x["mean_relation_f1"],x["mean_entity_f1"],x["entity_threshold"],x["relation_threshold"]))
    if best["mean_entity_f1"]<=0 or best["mean_relation_f1"]<=0: raise RuntimeError("D2修复版阶段门未通过：五折实体或关系F1仍为0，禁止正式运行")
    payload={"method":"D2_TCMERE_faithful_reimplementation_repair_v1","selection_data":"100条开发集五折；未使用300条测试Gold","epoch_candidates":list(EPOCHS),"threshold_candidates":list(THRESHOLDS),"optimizer_groups":{"encoder":"AdamW@5e-5","extraction":"Adam@8e-4","CompGCN":"Adam@1e-4"},"all_config_means":configs,"selected":best,"stage_gate":"passed"}
    (RESULTS/"frozen_cv_config.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(best,ensure_ascii=False,indent=2)); print(RESULTS/"frozen_cv_config.json")

if __name__=="__main__": main()
