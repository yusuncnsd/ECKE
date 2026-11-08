# ADELIE retained run: original experimental archive

This is the sole retained dataset for the selected adapted condition, not an unadapted first validation. Execution and organization were completed on 2026-10-06. Predictions, original responses, evaluation numbers, run times, prompts, and decoding settings were not changed during archiving.

All 300 records were processed: 221 structurally successful outputs and 79 failures. Entity Micro-F1 is 19.96% and relation Micro-F1 is 9.85%. Marked empty predictions for failures remain in the 300-record evaluation denominator.

## Retained materials

- Experiment_Workspace/data: frozen 300-record input and the seven-entity/seven-pattern Schema.
- Experiment_Workspace/results_optimized: final predictions, successful resume cache, original responses for successes and failures, terminal failure list, run fingerprints, structural audit, and context-capacity records.
- config_optimized.json and asset_manifest.json: parameters and input fingerprints.
- Python and installation programs: actual generation, download, offline-check, and evaluation entry points.
- Evaluation_Dependencies and Evaluation_Gold_Snapshot: frozen evaluator and reference labels.
- evaluation_optimized: full metrics, error details, 95% intervals, and coverage.

Official ADELIE-DPO weights are used with direct joint entity/relation output, greedy decoding, repetition_penalty=1.1, and a 768-token budget. No Gold examples or lexicon are added. Model weights are not bundled; the run manifest records revisions and weight hashes. Adaptation previously used fixed-test outputs; the resulting test-independence limitation must be disclosed.

## Archive cleanup

Duplicate result/upload ZIPs and checksum copies, obsolete transfer/run instructions, redundant code listings, packaging helpers, and full300_failures copies identical to terminal_failures were removed. Raw response records and raw_cache overlap in part but are both retained for audit and continuation. Cleanup records retain hashes of deleted files, not their contents.

## Re-evaluation and rerunning

Run `python Evaluation_Dependencies/evaluate_archived_run.py`. It uses the same supplied Gold and evaluation rules and writes a separate re-evaluation directory without overwriting original reports. Generation preserves the original workspace layout. Restore official weights to Experiment_Workspace/model/ADELIE-DPO and model/model_binding.json before running check/full300. Original historical project paths in the old evaluator are retained; use the archive-specific entry for portable re-evaluation.

The checksum manifest identifies original files. The external evaluation TXT is a convenient summary copy; pre-cleanup nested delivery ZIPs are unnecessary.
