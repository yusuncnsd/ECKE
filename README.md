# ECKE: Evidence-Constrained Knowledge Extraction

Core code and supporting data for disease-focused knowledge extraction from ancient Chinese medical texts.

Repository: https://github.com/yusuncnsd/ECKE

## Contents

| Directory or file | Contents |
| --- | --- |
| `01_Main Experiment Code` | Formal extraction and evaluation code, plus an offline verification script for the shared test labels. |
| `02_Schema and Runtime Configuration` | Seven entity categories, seven relation patterns and recorded model/evaluation settings. |
| `03_Domain Vocabulary` | An illustrative 200-concept subset of the actual 680-concept experimental lexicon, in JSON and TXT formats. |
| `04_Test Labels and Predictions` | Expert reference labels and ECKE predictions for all 300 test records, with per-record evaluation counts. |
| `05_Experimental Results` | Overall/category-specific ECKE performance, confidence intervals, comparison and ablation summaries, full-corpus statistics and gold-entity Oracle results. |
| `TCM-Books.zip` | 150 TXT ancient Chinese medical texts from the source materials used in this study, provided as a subset of the original texts. |

## Scope of the shared materials

This repository provides a subset of the original ancient medical texts and selected supporting materials. It does not contain the complete source corpus, the complete 680-concept experimental lexicon, the full development annotations, or the complete record-level predictions for all comparison and ablation methods. The 200-concept lexicon sample was selected for public illustration and was not used to produce the reported experimental results.

The shared test-label files contain record identifiers, expert reference labels and system predictions. Source passages, translations and evidence sentences are not included in those files. The TXT books in `TCM-Books.zip` are separate source materials; their inclusion does not replace the full passage inputs and annotations required to reproduce all experiments. Original-text alignment and evidence positions cannot be fully verified from the reduced test labels alone.

## Code and runtime requirements

Python 3.10 or later is required. The provided scripts use the Python standard library.

- `01_运行V612全库知识抽取.py` is the formal extraction entry point, including entity recognition/fusion and contextual relation assessment prompts. It requires the original corpus, development annotations, full lexicon and other inputs in the original `主实验/02_ECKE/...` layout beneath the repository root. These inputs are not all included in this repository. The API credential placeholder is empty; a user's own credential is required for extraction.
- `02_评价300条独立测试集.py` is the original strict-matching evaluator. It expects the original annotation TXT format in that same input layout, rather than the reduced JSONL export shared here.
- `03_verify_exported_counts.py` verifies the shared JSONL labels offline. It requires no API credential, makes no network requests and does not write output files.

The shared material is not a ready-to-run reproduction package for the complete extraction experiment. The source programs' scientific settings and processing logic are preserved; exported copies omit comments/docstrings and personal absolute root paths.

## Verify the shared primary results

Run the following command from the repository root:

```text
python "01_Main Experiment Code/03_verify_exported_counts.py"
```

All 300 test records are included. Within each record, strict set matching uses entity surface/type pairs and directed head-surface/relation/tail-surface triples. Across-record occurrences remain separate.

| Task | TP | FP | FN | Gold count | Micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Entity recognition | 789 | 64 | 80 | 869 | 0.9164 |
| Relation extraction | 422 | 103 | 86 | 508 | 0.8170 |

## File formats and interpretation

- Entity arrays in `01_test300_gold_and_ECKE_predictions.jsonl` contain `[surface form, entity type]`.
- Relation arrays contain `[head surface, relation label, tail surface]`.
- Gold labels and ECKE predictions are kept in separate fields. System outputs are not human annotations.
- `source_rows` in the lexicon subset refers to the original 680-row lexicon workbook.
- The Schema has seven relation patterns but six distinct relation labels because the Chinese medicinal and Formula patterns share the Treats label.
- The Schema version identifier `v5` denotes the valid machine-readable configuration.
- CENTRELINE, TCMERE, BSTR-GPRel and CARE precision, recall and F1 values are separately averaged over three seeds. Mean F1 need not equal the harmonic mean of mean precision and mean recall.
- Confidence intervals are taken from the completed official evaluation; they were not newly estimated for publication.
- The gold-entity Oracle results are diagnostic and are not the primary end-to-end results.

## Integrity

The following SHA256 checksums identify the files distributed in this repository. No API credentials are included. Sharing these files does not introduce a new licence for the original source materials.

- `01_Main Experiment Code/01_运行V612全库知识抽取.py`: `1a17006b0613f76d03da7e03309e4c97c631f5b5f688f9d8a13870706940b3e5`.
- `01_Main Experiment Code/02_评价300条独立测试集.py`: `86bd3f4d537dc8d062e15d3cb2df736e63311ff8b85485d075535a51e00d6b96`.
- `01_Main Experiment Code/03_verify_exported_counts.py`: `3b9725d485ecc0fb64ece7479c2d27a3ae5f9c000823cf5a0054ffac47fe5379`.
- `02_Schema and Runtime Configuration/05_entity_relation_schema.json`: `32a37f9f1d210f1059a561460d32701394aab098d333e52b63b7bebc90e06de4`.
- `02_Schema and Runtime Configuration/model_and_evaluation_config.json`: `303b265c5e7be1814f917579e45cff4791ac88188a84d7c78ca821a0a390f494`.
- `03_Domain Vocabulary/domain_lexicon_sample_200.txt`: `be9ae59200427289b92ef488a8dd515794b0263b53a238f731e6bc70931a1fe0`.
- `03_Domain Vocabulary/xiaoke_lexicon_sample_200.json`: `50ab7bc5e144afbc88b6889584d513d4acb1b24c3737556ad547cdb36bf5592b`.
- `04_Test Labels and Predictions/01_test300_gold_and_ECKE_predictions.jsonl`: `15f92bb1ca7fdd4818a831b487bda9a99da5f74d878aba0b97a612ff21b6586a`.
- `04_Test Labels and Predictions/02_test300_record_level_counts.csv`: `def27decebc7ca3e11245957c1f9c284b564dd9517f74d70c6f01dd97be1e809`.
- `05_Experimental Results/03_ECKE_performance_summary.json`: `e046d512534b7a81d716a1cb114151f4145e961f6fbfceb1aa0adc90f58b68a7`.
- `05_Experimental Results/04_comparison_and_ablation_summary.csv`: `53279c86ddc445fb637125f3f4ceb9efb0a514869626e5a3f037a42efa9465d8`.
- `05_Experimental Results/full_corpus_run_summary.json`: `61de7363e6367358bb665d8b92ed0d8e2f0572100bc3e3cb1312466a6f106c4c`.
- `05_Experimental Results/gold_entity_oracle_results.json`: `77017b5083afb3127203d187aa222ac62fe24ca15f7e300f8940f2154e9ba23f`.
- `TCM-Books.zip`: `3ec99843e07a0cd29f9e24f31cece3ce72c379bf03a4c964718931f53bdec5f0`.
