# New inference and training runs

The offline evaluation in `09_Reproduction_Tools/reproduce.py` requires only Python 3.10+ and does not use the network. The commands in this document are separate, optional new model runs. They were not executed during packaging.

## Keep frozen and new runs separate

Use `09_Reproduction_Tools/prepare_run.py STAGE --workdir NEW_DIRECTORY` from the package root. It refuses an existing directory and creates a separate input/program workspace. Never use a frozen result directory as a continuation cache for a new condition. Fill an empty credential placeholder only in the prepared workspace, never in public Git files.

Historical hash bindings remain records of the original runs. Copies with personal root paths made relative have a new code hash, recorded in `07_Provenance/source_to_package.json`. A new run requires new bindings; old frozen asset-manifest hashes must not be silently claimed as those of modified code.

## ECKE

```text
python 09_Reproduction_Tools/prepare_run.py ecke --workdir ../ECKE-new-run
```

Fill the empty API credential placeholder in the prepared `02_Experiments/02_ECKE/01_Code/01_Run_V612_Full_Corpus_Extraction.py`, then run that file. It uses the supplied complete passage corpus, full lexicon, development annotations and Schema. The model is `qwen-plus-2025-07-28`, temperature 0, seed base 20260813, max retries 6 and workers 5. The original pipeline writes new results under its `复现运行` directory.

The script uses the Python standard library. A valid personal DashScope-compatible credential and access to the named model are necessary. Temperature 0 and a seed do not guarantee byte-identical future API outputs.

## Passage screening

```text
python 09_Reproduction_Tools/prepare_run.py screening --workdir ../ECKE-screening-new-run
```

The prepared stage-01 directory contains the 488 raw and cleaned books, original cleaning/reconstruction/screening programs, actual XLSX lexicon and risk-review workbook. The screening entry is `01_Code/03_Screen_Xiaoke_Passages.py`. Fill your own credential before explicitly running it. Deterministic reconstruction and candidate retrieval alone can be checked with `09_Reproduction_Tools/verify_screening.py`, without any credential.

## Ablations

```text
python 09_Reproduction_Tools/prepare_run.py ablation --workdir ../ECKE-ablation-new-run
```

The four entries in `02_Experiments/04_Ablation_Studies/01_Code/` are `02_Run_A1_Without_Lexicon_Anchoring.py`, `03_Run_A2_Without_Semantic_Supplementation.py`, `04_Run_A3_Simple_Union.py`, and `05_Run_A4_Schema_Only_Relations.py`. They share `ablation_engine.py` and the stage-02 main program. A1–A3 may require API calls; do not execute them merely to re-evaluate the supplied predictions.

## OneKE-Qwen and LLMRE

Prepare `oneke` or `llmre`. OneKE entry: `02_Experiments/03_External_Comparisons/01_Code/02_E1_OneKE-Qwen/01_Run_OneKE_Qwen.py`. It additionally imports scikit-learn for character TF-IDF retrieval. Install `scikit-learn==1.7.2`, as used by the retained supervised package environment; the original OneKE-specific environment freeze was not retained separately.

LLMRE entry: `02_Experiments/03_External_Comparisons/LLMRE_Comparison/01_Run_LLMRE_Adaptation.py`. Its `Inputs/Prompts.json`, development-text-only input, test input and Schema are included. It uses the standard library and the same Qwen API model. The preparation command excludes the old SQLite cache so a new run cannot silently reuse historical responses.

## CENTRELINE, TCMERE and BSTR-GPRel

```text
python 09_Reproduction_Tools/prepare_run.py supervised --workdir ../ECKE-supervised-new-run
```

Within the prepared `03_Baseline_Packages/00_Shared_Supervised_Code/AutoDL_D1-D3_Package_v2/`, follow the original `README_EN.txt`, install script, model downloader and preflight. The original environment specifies transformers 4.55.4, scikit-learn 1.7.2, jieba 0.42.1 and CUDA PyTorch on a suitable GPU. Model downloads are separate and governed by provider terms.

CENTRELINE and BSTR-GPRel use the original D1/D3 cross-validation and formal training programs. The reported TCMERE predictions come from **D2_repair_v1**, not the older D2 path. The preparation command places this repair folder inside the base package as expected by its original README; run `python D2_repair_v1/00_run_all_d2_repair.py` there. It performs development-fold selection followed by three formal seeds. Do not select settings using test Gold.

## CARE

The `supervised` workspace also contains `02_Experiments/03_External_Comparisons/CARE_Comparison/01_Experiment_Code/`. Original dependencies recorded for CARE: torch 2.8.0+cu128 and transformers 4.55.4 on RTX 4090D. The BERT revision is recorded in its config. Final model checkpoints are not bundled; new inference therefore requires training.

After obtaining the specified local BERT model, run from that code directory:

```text
python CARE_Balanced/CARE_run.py quick --model /absolute/path/to/bert-base-chinese
python CARE_resume_decode.py --model /absolute/path/to/bert-base-chinese
python CARE_calibrate.py --model /absolute/path/to/bert-base-chinese
```

The calibrated run is the reported version. Keep the frozen five-fold development partition and final seeds. `results_calibrated_v1/threshold_selection.json` records the original chosen thresholds and selection inputs; the original process document provides the full run order.

## YAYI-UIE

Prepare `yayi`. In `02_Experiments/03_External_Comparisons/01_Code/05_E3_YAYI-UIE/`, follow the retained README and scripts: environment installation, model download, tokenizer patch, preflight, then `python 03_run_e3.py full300`. The original environment uses transformers 4.45.2, huggingface-hub 0.26.5, accelerate 1.1.1 and bitsandbytes 0.46.1. A compatible GPU, model weights and provider access may be required. The result cache is not required for offline evaluation.

## ADELIE

Prepare `adelie`. In `03_Baseline_Packages/08_ADELIE/Experiment_Workspace/`, use the supplied installation and download scripts to restore the model directory and model binding, then run the original `02_run_adelie_optimized.py` entry according to its command-line interface. Keep `config_optimized.json` unchanged to regenerate this condition. It specifies official ADELIE-DPO weights, greedy decoding, repetition penalty 1.1 and max_new_tokens 768. The retained original process document describes the actual run.

## Environments

Use separate virtual environments for the different baseline families; their transformers versions differ. Model weights, vendor API service implementation and deleted historical checkpoints are not reconstructed by dependency installation. New-run logs, version information and predictions should be archived separately before comparing them with the frozen submission outputs.
