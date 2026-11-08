#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
zip -j /root/autodl-tmp/E3_YAYI_UIE_Formal_Results.zip 01_preflight.py 03_run_e3.py 04_output_audit.py asset_manifest.json results/E3_YAYI-UIE_300_predictions.jsonl results/E3_cloud_output_audit.json results/E3_full300_run_manifest.json results/E3_full300_failures.json results/E3_full300_raw_cache.jsonl
sha256sum /root/autodl-tmp/E3_YAYI_UIE_Formal_Results.zip > /root/autodl-tmp/E3_YAYI_UIE_Formal_Results_SHA256.txt
ls -lh /root/autodl-tmp/E3_YAYI_UIE_Formal_Results*
