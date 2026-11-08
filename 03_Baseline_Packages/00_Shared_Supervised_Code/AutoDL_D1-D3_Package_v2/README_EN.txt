AutoDL single-GPU run sequence

Recommended: RTX 4090 24GB. Choose RTX 3090 24GB if significantly cheaper; A10 24GB is usually slower.

1. Upload the entire D1–D3 package, not separate test Gold.
2. Enter this directory and run bash 00_install_environment.sh.
3. Run python 01_preflight.py.
4. Run python 02_download_models.py.
5. Run bash 06_run_all.sh.
6. The entry performs preflight and model freezing, then GPU smoke tests, five-fold selection and three-seed training/prediction for D1, D2 and D3, followed by integrity checks.
7. Restart the same command after interruption; checkpoints, fingerprints and per-record caches prevent repeating completed stages.
8. Results may be downloaded method by method; finally download models/pretrained_model_manifest.json, all results and asset_manifest.json.
9. Stop the instance after verifying local backups. Closing the browser does not stop billing.

Test Gold is not in the cloud package. The cloud generates predictions; formal evaluation is local. The paper's final TCMERE result uses D2_repair_v1 rather than the older D2 route.
