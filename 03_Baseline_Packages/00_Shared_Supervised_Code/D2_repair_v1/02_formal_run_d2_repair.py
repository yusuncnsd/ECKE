import json, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; DATA=ROOT/"05_Frozen_Data_Indices/data"; MODELS=ROOT/"models"; RESULTS=ROOT/"results"/"D2_repair_v1"
cfg=json.loads((RESULTS/"frozen_cv_config.json").read_text(encoding="utf-8")); selected=cfg["selected"]
if cfg.get("stage_gate")!="passed": raise RuntimeError("D2修复版开发阶段门未通过，禁止正式运行")
cmd=[sys.executable,str(HERE/"formal_train_d2_repair.py"),"D2","--epochs",str(selected["epochs"]),"--entity-threshold",str(selected["entity_threshold"]),"--relation-threshold",str(selected["relation_threshold"]),"--result-root",str(RESULTS),"--model-root",str(MODELS),"--data-root",str(DATA)]
subprocess.run(cmd,check=True)
