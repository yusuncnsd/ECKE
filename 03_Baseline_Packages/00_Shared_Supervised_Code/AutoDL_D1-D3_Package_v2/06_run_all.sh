#!/bin/bash
set -euo pipefail
python 01_preflight.py
python 02_download_models.py
for method in D1 D2 D3; do
  python code/formal_train.py "$method" --quick
  python 03_cross_validation.py "$method"
  python 04_formal_run.py "$method"
done
python 05_output_audit.py
