import subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
for script in ("01_cross_validate_d2_repair.py","02_formal_run_d2_repair.py","03_audit_d2_repair.py"):
    print(f"\n[D2修复流程] 开始：{script}",flush=True)
    subprocess.run([sys.executable,str(HERE/script)],check=True)
print("\nD2修复版五折开发、正式运行和输出审计全部完成。",flush=True)
