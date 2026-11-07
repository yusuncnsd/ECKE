import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; DATA=ROOT/"05_Frozen_Data_Indices/data"; RESULTS=ROOT/"results"/"D2_repair_v1"
schema=json.loads((DATA/"xiaoke_schema_v5.json").read_text(encoding="utf-8")); types=set(schema["entity_types"]); legal={(x["head_type"],x["relation"],x["tail_type"]) for x in schema["relation_schemas"]}
test=[json.loads(x) for x in (DATA/"independent_test_300_input_only.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]; expected=[x["record_id"] for x in test]; report=[]
for seed in (20260823,20260824,20260825):
    path=RESULTS/"D2"/f"seed_{seed}"/"predictions_300.jsonl"; rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if [x["record_id"] for x in rows]!=expected: raise RuntimeError(f"D2 seed={seed} ID或顺序错误")
    nonempty=0
    for row in rows:
        entities={tuple(x) for x in row["folded_entities"]}; nonempty+=bool(entities)
        if any(t not in types for _,t in entities): raise RuntimeError(f"D2 seed={seed} 非法实体类型")
        typed={s:{t for x,t in entities if x==s} for s,_ in entities}
        for h,r,t in row["relations"]:
            if h not in typed or t not in typed or not any((ht,r,tt) in legal for ht in typed[h] for tt in typed[t]): raise RuntimeError(f"D2 seed={seed} 非法关系端点或Schema")
    if nonempty==0: raise RuntimeError(f"D2 seed={seed} 仍全部为空，阶段门未通过")
    report.append({"seed":seed,"records":len(rows),"nonempty_records":nonempty,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
(RESULTS/"D2_repair_output_audit.json").write_text(json.dumps({"status":"passed","runs":report},ensure_ascii=False,indent=2),encoding="utf-8")
print("D2修复版输出审计通过"); print(RESULTS/"D2_repair_output_audit.json")
