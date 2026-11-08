# CARE version 2: final experimental description

## Design and fixed conditions

The retained condition is a Chinese low-resource adaptation of CARE with decoding thresholds calibrated on 100 development records. It may be labeled CARE (reimplementation), but adaptations must be disclosed rather than claiming all original training settings.

- Author code commit: 93ea51306a94b617deb412e9bb8372a94a59dacf; source paper/license records are in the source-material directory.
- Encoder: google-bert/bert-base-chinese, revision 8f23c25b06e129b6c986331a13d8d025a92cf0ea.
- Author co-attention and classifier structure retained; logits retained for stable training loss.
- Seven entity types, seven relation patterns, six labels; authoritative configuration: data/xiaoke_schema_v5.json.
- Development annotations supervise this external baseline and threshold selection; ECKE itself does not fine-tune an LLM.
- Fixed five folds, 80 training/20 validation records per fold, with disjoint validation IDs covering all 100.
- 100 epochs, learning rate 0.0001, Adam, dropout 0.1, gradient clipping 0.25, effective batch 8/microbatch 1, gradient checkpointing.
- Positive/negative valid-grid softplus losses are separately averaged, then equally weighted. A head without positives uses its negative mean alone. Entity grids are upper triangular; relation grids include all valid text pairs.
- CV seed 20260823; final seeds 20260823/20260824/20260825 each train on all 100 and predict the fixed 300.
- Maximum 512 tokens; no silent long-input truncation. Chinese WordPiece positions are aligned to source characters.
- Recorded environment: torch 2.8.0+cu128, transformers 4.55.4, RTX4090D. No LLM API is called.

## Version-2 threshold selection

The pre-fixed grid is 0.5, 0.8, 0.95, 0.99, 0.999, 0.9999: 36 entity/relation combinations. Reuse the five final fold models to obtain out-of-fold predictions on all 100 development records. Only development Gold is scored. Candidates must improve both aggregated Micro-F1 values over 0.5/0.5; maximize their arithmetic mean, breaking ties by relation precision and thresholds.

Chosen entity/relation thresholds: 0.9999/0.9999. Selection-set entity F1=0.3298611111 and relation F1=0.0641025641. These are parameter-selection scores, not independent validation performance. All 36 conditions and fingerprints are in results_calibrated_v1/threshold_selection.json.

## Evaluation and actual chronology

Gold has 869 entity set items and 508 relations. Match surface/type pairs and directed head/relation/tail triples; collapse duplicates, ignore list order. Report Micro-P/R/F1 as the arithmetic mean and sample SD over three runs; 900 repeated predictions are not 900 independent test passages.

Per-seed CIs use 5000 passage bootstrap samples, seed 20260822. Paired analysis uses 10000 bootstrap and 10000 permutation samples, with within-task Holm correction over the three CARE seeds only. Relation Macro-F1 uses six labels, not seven endpoint-type patterns.

Development calibration was performed after version 1 had already been evaluated on the test set. The calibration program did not read test Gold or select thresholds from test answers, but the test had previously been evaluated; do not claim otherwise.

## Final run sequence

1. Freeze 100 development labels, disjoint 300 test inputs, Schema and five-fold split.
2. Train the author co-attention/classifier adaptation with Chinese BERT and approved positive/negative balanced loss for 100 epochs in each fold and each final seed.
3. Use equivalent index decoding in CARE_resume_decode.py, preserving legal Schema links and strict set output without prediction truncation.
4. Compare the 36 threshold pairs only on 100 out-of-fold development records; select 0.9999/0.9999.
5. Apply frozen thresholds to each of the three trained final models: 300 successful outputs per seed. Cloud calibration reads no test Gold.
6. Download results and verify source, fingerprints, IDs, evidence spans and Schema; evaluate frozen Gold: entity F1=38.66±3.42%, relation F1=6.13±2.22%.

results_calibrated_v1 contains original predictions, completion/binding records, audit and threshold grid. results_fixed100 contains eight training bindings/logs used to generate version 2; 0.5 fold counts in complete.json are calibration-check inputs, not a different formal result. Original calibration logs/archive hashes were retained in the run-log archive; Result_Reports contains local evaluations and summaries.

Model weights were not retained, as requested. Obsolete predictions/evaluation entries/instructions and nonfinal logs were removed. Prediction regeneration requires retraining, but the retained predictions can be re-evaluated directly. Cleanup of asset_manifest removed only old evaluator/README entries; training code, inputs, configurations, final predictions and cloud bindings were not rewritten. Original manifest fingerprints remain in submission_cleanup/final-condition records and the archived cloud snapshot.

## Usage

Read CARE_V2_Final_Results.md. JSON metrics use 0–1; narrative tables use percentages. CSV contains per-seed P/R/F1, counts and CIs.

For local re-evaluation from 01_Experiment_Code:
```text
python CARE_evaluate_second.py
```
This locates common frozen Gold/ECKE predictions, verifies fingerprints and rebuilds reports. For full retraining, preserve the original code/data structure and obtain the exact BERT revision and CUDA environment. The eight checkpoints must be recreated under CARE_Balanced/results_fixed100. Use a new server copy without old run metadata, preserving frozen archives:
```text
python -u CARE_Balanced/CARE_run.py quick --model /root/autodl-tmp/AutoDL_CARE/models/bert-base-chinese
python -u CARE_resume_decode.py --model /root/autodl-tmp/AutoDL_CARE/models/bert-base-chinese
python -u CARE_calibrate.py --model /root/autodl-tmp/AutoDL_CARE/models/bert-base-chinese
```
The first two recreate shared fixed-100-epoch models; second-stage 0.5 outputs are intermediate, not the reported final condition. The third recreates calibration and final predictions. GPU/dependency changes may alter numerical outputs: record a new run and never overwrite frozen results or retrain inside the retained archive.
