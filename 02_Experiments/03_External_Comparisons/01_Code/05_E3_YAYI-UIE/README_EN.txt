YAYI-UIE strict zero-shot package (2026-08-26)

Paper: YAYI-UIE: A Chat-Enhanced Instruction Tuning Framework for Universal Information Extraction.
Official model: wenge-research/yayi-uie.
Condition: strict zero-shot transfer from a general Chinese information-extraction LLM to the ancient-text wasting-thirst Schema.

The package provides only 300 unlabeled test inputs and frozen Schema, not test Gold, preannotations or development Gold examples. The official model-card training templates and original Chinese entity/relation task markers are retained. Entity output is a seven-type dictionary; relations follow the frozen list. No target-domain parameter training is performed.

Postprocessing is structural only: JSON/safe Python literal parsing, verbatim source checks, and Schema checks. Test answers are not used to repair predictions. A 13B model is loaded with NF4 4-bit quantization on RTX 4090 24GB for memory compatibility; this does not retrain parameters.

Run:
bash 00_install_environment.sh
bash 06_run_all.sh

The first stage validates ten records, including cross-type-copy rejection. Inspect results/E3_YAYI-UIE_pilot10.jsonl, then run:
python 03_run_e3.py full300
python 04_output_audit.py
bash 05_package_results.sh

The same mode resumes by record_id. Fingerprint checks reject changed inputs, Schema or key parameters. Download /root/autodl-tmp/E3_YAYI_UIE_Formal_Results.zip and its SHA256 file. Formal Gold is evaluated locally, outside prediction.
