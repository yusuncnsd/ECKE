# ECKE core code and supporting data

Prepared: 2026-10-07. This package contains a small selection of finalized code, domain resources and supporting experimental data. It excludes ancient medical books, source passages, translations, evidence sentences, credentials and execution caches. It is not the complete raw corpus or a complete reproduction package.

## Folder contents

### 01_主实验代码

- `01_运行V612全库知识抽取.py`: publication copy of the existing formal ECKE extraction entry point, including entity recognition/fusion and relation assessment prompts. Comments and docstrings were removed, the API-key assignment was emptied, and the personal absolute root was replaced with a relative path. Scientific settings and processing logic remain unchanged. Running this program requires the original corpus, development annotations and supporting files, which are intentionally not distributed here. It also requires the user's own API credentials and may incur API charges.
- `02_评价300条独立测试集.py`: publication copy of the existing official strict-matching evaluation program. Comments/docstrings and the personal absolute root were removed. This original evaluator expects the original annotation TXT layout, not the reduced exported JSONL.
- `03_verify_exported_counts.py`: small offline verification script added for this package; it computes strict-set TP/FP/FN and Micro-F1 directly from the distributed JSONL and checks them against the archived summary. It does not call an API or write results. It is not an original experiment entry point.

The two original program copies expect the original `主实验/02_ECKE/...` data layout beneath the package root. This preserves their relative file organization without exposing the author's computer path. Do not run extraction against this incomplete package. Original experiment files and caches were not changed.

### 02_Schema与运行配置

The actual entity/relation Schema and the recorded model/evaluation configuration. The v5 identifier refers to the valid Schema configuration, not an obsolete manuscript. Seven relation patterns use six distinct labels.

### 03_领域词表

A 200-concept public illustrative subset from the actual 680-concept experimental lexicon, supplied as JSON and readable TXT. The original 680-concept lexicon remains unchanged in the research archive and was used for the reported experiments. This 200-concept subset was not used for those results and must not replace the full lexicon in experiment reproduction. Source-row references point to the original workbook. Sampling covers all seven entity types and preserves nine principal disease-topic anchors; remaining items were selected by deterministic hash ranking independently of test labels and performance. Type counts: Chinese medicinal 45, Formula 58, Disease 9, Symptom 41, Pathogenesis 16, Disease location 8, Therapeutic method 23.

### 04_测试标签与预测

All 300 frozen test record IDs, expert reference labels and ECKE predictions, with per-record TP/FP/FN. Entity arrays contain [surface form, entity type]; relation arrays contain [head surface, relation label, tail surface]. Short extracted expressions remain as research labels. Source passages and evidence sentences are excluded. No records were selected by correctness. Predictions remain explicitly identified as ECKE outputs; they are not presented as human annotations.

### 05_实验结果

Archived overall/category-specific ECKE performance and confidence intervals; external comparison/ablation summary supporting manuscript Tables 3 and 4; full-corpus counts; gold-entity Oracle diagnostic results. Confidence intervals were copied from the completed evaluation, not newly estimated. CENTRELINE, TCMERE, BSTR-GPRel and CARE P/R/F1 values are separately averaged across three seeds; mean F1 need not equal the harmonic mean of mean P and mean R. Baseline/ablation record-level predictions are not included.

## Verified primary results

Strict matching uses sets within each record and preserves relation direction. Across-record occurrences remain separate.

- Entities: TP=789, FP=64, FN=80; 869 gold entities; Micro-F1=0.9164.
- Relations: TP=422, FP=103, FN=86; 508 gold relations; Micro-F1=0.8170.

Offline verification: `python 01_主实验代码/03_verify_exported_counts.py`.

## Scope and provenance

本包为少量正式代码、词表及支持数据，不含古籍正文，也不能独立复现全部实验。主结果严格匹配计数可以核验，但原文证据/字符位置、完整抽取及对比/消融的逐条结果不能由本包独立复核。原始资料均未修改。本包未上传，也未新设数据许可证。

Comments and docstrings in all exported Python files have been removed. Prompt text, model identity and genuine machine-prediction provenance are retained as research information. Removal of editorial notes does not establish or alter code authorship. The following hashes record source/export integrity for the two formal code copies.


- `01_运行V612全库知识抽取.py`: source SHA256 `d2257830e05d3794b114bd64855f9e5679fe818d7ba43efb502c701cd7e434b0`; export SHA256 `1a17006b0613f76d03da7e03309e4c97c631f5b5f688f9d8a13870706940b3e5`.

- `02_评价300条独立测试集.py`: source SHA256 `ccf9a8a5c36be2325765ed247f03e8dcec832c692d65f79359a35d7247719d38`; export SHA256 `86bd3f4d537dc8d062e15d3cb2df736e63311ff8b85485d075535a51e00d6b96`.

## Data integrity

- `01_主实验代码/01_运行V612全库知识抽取.py`: SHA256 `1a17006b0613f76d03da7e03309e4c97c631f5b5f688f9d8a13870706940b3e5`.
- `01_主实验代码/02_评价300条独立测试集.py`: SHA256 `86bd3f4d537dc8d062e15d3cb2df736e63311ff8b85485d075535a51e00d6b96`.
- `01_主实验代码/03_verify_exported_counts.py`: SHA256 `144e343bd84a3fe2e91ecd9c30845809cb4a83afa0dcbf31d1b86919829752cc`.
- `02_Schema与运行配置/05_entity_relation_schema.json`: SHA256 `32a37f9f1d210f1059a561460d32701394aab098d333e52b63b7bebc90e06de4`.
- `02_Schema与运行配置/model_and_evaluation_config.json`: SHA256 `303b265c5e7be1814f917579e45cff4791ac88188a84d7c78ca821a0a390f494`.
- `03_领域词表/domain_lexicon_sample_200.txt`: SHA256 `be9ae59200427289b92ef488a8dd515794b0263b53a238f731e6bc70931a1fe0`.
- `03_领域词表/xiaoke_lexicon_sample_200.json`: SHA256 `50ab7bc5e144afbc88b6889584d513d4acb1b24c3737556ad547cdb36bf5592b`.
- `04_测试标签与预测/01_test300_gold_and_ECKE_predictions.jsonl`: SHA256 `15f92bb1ca7fdd4818a831b487bda9a99da5f74d878aba0b97a612ff21b6586a`.
- `04_测试标签与预测/02_test300_record_level_counts.csv`: SHA256 `def27decebc7ca3e11245957c1f9c284b564dd9517f74d70c6f01dd97be1e809`.
- `05_实验结果/03_ECKE_performance_summary.json`: SHA256 `e046d512534b7a81d716a1cb114151f4145e961f6fbfceb1aa0adc90f58b68a7`.
- `05_实验结果/04_comparison_and_ablation_summary.csv`: SHA256 `53279c86ddc445fb637125f3f4ceb9efb0a514869626e5a3f037a42efa9465d8`.
- `05_实验结果/full_corpus_run_summary.json`: SHA256 `61de7363e6367358bb665d8b92ed0d8e2f0572100bc3e3cb1312466a6f106c4c`.
- `05_实验结果/gold_entity_oracle_results.json`: SHA256 `77017b5083afb3127203d187aa222ac62fe24ca15f7e300f8940f2154e9ba23f`.
