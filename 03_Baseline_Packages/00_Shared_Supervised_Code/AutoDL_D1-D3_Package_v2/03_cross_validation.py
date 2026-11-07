import json,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent; CODE=ROOT/"code"; DATA=ROOT/"05_Frozen_Data_Indices/data"; RESULTS=ROOT/"results"
CANDIDATES={"D1":[8,12],"D2":[12,20],"D3":[20,30]}
def read(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def f1(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; return 2*p*r/(p+r) if p+r else 0
def score(pred,gold):
    pm={x["record_id"]:x for x in pred}; et=ef=en=rt=rf=rn=0
    for g in gold:
        ge={(x["surface_form"],x["entity_type"]) for x in g["span_entities"]}; gr={tuple(x) for x in g["relations"]}; p=pm[g["record_id"]]; pe={tuple(x) for x in p["folded_entities"]}; pr={tuple(x) for x in p["relations"]}
        et+=len(pe&ge); ef+=len(pe-ge); en+=len(ge-pe); rt+=len(pr&gr); rf+=len(pr-gr); rn+=len(gr-pr)
    return {"entity_f1":f1(et,ef,en),"relation_f1":f1(rt,rf,rn),"joint":(f1(et,ef,en)+f1(rt,rf,rn))/2}
def main(method):
    progress_path=RESULTS/method/"cv_progress.json"; progress_path.parent.mkdir(parents=True,exist_ok=True)
    results=json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else []
    for epochs in CANDIDATES[method]:
        for fold in range(5):
            if any(x["epochs"]==epochs and x["fold"]==fold for x in results): continue
            suffix=f"_cv_e{epochs}_f{fold}"; cmd=[sys.executable,str(CODE/"formal_train.py"),method,"--development",f"cv_fold{fold}_train.jsonl","--test",f"cv_fold{fold}_valid.jsonl","--seed","20260823","--epochs",str(epochs),"--result-suffix",suffix]; subprocess.run(cmd,check=True)
            run_dir=RESULTS/method/f"seed_20260823{suffix}"; pred=read(run_dir/"predictions_300.jsonl"); gold=read(DATA/f"cv_fold{fold}_valid.jsonl"); results.append({"epochs":epochs,"fold":fold,**score(pred,gold)}); progress_path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
            # 五折模型和逐条预测均可由固定数据/配置重建；指标落盘后立即删除，避免占满50GB数据盘。
            shutil.rmtree(run_dir)
    means={e:sum(x["joint"] for x in results if x["epochs"]==e)/5 for e in CANDIDATES[method]}; best=max(means,key=means.get); payload={"method":method,"candidates":CANDIDATES[method],"fold_results":results,"mean_joint":means,"selected_epochs":best,"entity_threshold":0.5 if method=="D2" else None,"relation_threshold":0.5 if method=="D2" else None,"globalpointer_logit_threshold":0.0 if method=="D3" else None}
    path=RESULTS/method/"frozen_cv_config.json"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); print(path)
if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in CANDIDATES: raise SystemExit("用法：python 03_五折开发.py D1|D2|D3")
    main(sys.argv[1])
