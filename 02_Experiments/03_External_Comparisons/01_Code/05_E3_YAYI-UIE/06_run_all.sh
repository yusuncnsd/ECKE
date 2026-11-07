#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python 01_preflight.py
if [ ! -s model/YAYI-UIE/config.json ]; then python 02_download_model.py; fi
python 02a_patch_tokenizer.py
rm -rf -- /root/.cache/huggingface/modules/transformers_modules/YAYI-UIE
python 03_run_e3.py pilot10
python 04_output_audit.py pilot10
echo '固定10条格式验证完成。人工确认结果后，再单独执行：python 03_run_e3.py full300'
echo '300条完成后执行：python 04_output_audit.py && bash 05_package_results.sh'
