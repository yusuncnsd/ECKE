import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"05_Frozen_Data_Indices/data"; RESULTS=ROOT/"results"; TEST=[json.loads(x) for x in (DATA/"independent_test_300_input_only.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]; EXPECT=[x["record_id"] for x in TEST]; TEXT={x["record_id"]:x["text"] for x in TEST}; SCHEMA=json.loads((DATA/"xiaoke_schema_v5.json").read_text(encoding="utf-8")); TYPES=set(SCHEMA["entity_types"]); LEGAL={(x["head_type"],x["relation"],x["tail_type"]) for x in SCHEMA["relation_schemas"]}
def read(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def main():
    report={"status":"passed","methods":{}}
    for method in ("D1","D2","D3"):
        rows=[]
        for seed in (20260823,20260824,20260825):
            p=RESULTS/method/f"seed_{seed}/predictions_300.jsonl"
            if not p.exists(): raise RuntimeError(f"缺少：{p}")
            x=read(p); ids=[r["record_id"] for r in x]
            if ids!=EXPECT: raise RuntimeError(f"{method}/{seed}数量、ID或顺序错误")
            for r in x:
                surfaces={}
                for e in r["entities"]:
                    if e["entity_type"] not in TYPES or TEXT[r["record_id"]][e["start"]:e["end"]]!=e["surface_form"]: raise RuntimeError(f"实体类型或位置错误：{method}/{seed}/{r['record_id']}/{e}")
                    surfaces.setdefault(e["surface_form"],set()).add(e["entity_type"])
                if {tuple(z) for z in r["folded_entities"]}!={(e["surface_form"],e["entity_type"]) for e in r["entities"]}: raise RuntimeError("实体折叠错误")
                for h,rel,t in r["relations"]:
                    if h not in surfaces or t not in surfaces or not any((ht,rel,tt) in LEGAL for ht in surfaces[h] for tt in surfaces[t]): raise RuntimeError(f"关系端点或Schema错误：{method}/{seed}/{r['record_id']}/{h}|{rel}|{t}")
            rows.append({"seed":seed,"records":len(x),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
        report["methods"][method]=rows
    (RESULTS/"D1-D3_cloud_output_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print("D1-D3输出审计通过")
if __name__=="__main__": main()
