import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent; CODE=ROOT/"code"; RESULTS=ROOT/"results"
def main(method):
    cfg=json.loads((RESULTS/method/"frozen_cv_config.json").read_text(encoding="utf-8")); epochs=cfg["selected_epochs"]
    subprocess.run([sys.executable,str(CODE/"formal_train.py"),method,"--epochs",str(epochs)],check=True)
if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ("D1","D2","D3"): raise SystemExit("用法：python 04_正式运行.py D1|D2|D3")
    main(sys.argv[1])
