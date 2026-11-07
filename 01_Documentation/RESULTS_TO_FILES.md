# Submitted results to supplied files

All paths below are relative to this package. The manuscript version is identified by `PACKAGE_METADATA.json` and `01_Documentation/submitted_claims.json`.

| Result | Frozen file or directory | Verification |
|---|---|---|
| 488 original / cleaned books | `02_Experiments/01_Corpus_Preparation/02_Input_Data/488_Original_Books/` and `04_Results/488_Cleaned_Books/` | `09_Reproduction_Tools/verify_cleaning.py` |
| 5924 screening candidates | `05_Frozen_Data_Indices/screening/candidates.jsonl` | `09_Reproduction_Tools/verify_screening.py` rebuilds original deterministic code |
| Manual screening review | Risk-review workbook and original screening code under stage 01 | Original review materials |
| 3065 final passages | `02_Experiments/02_ECKE/02_Input_Data/Topic_Passage_Corpus/正式消渴主题条文库.jsonl` | unique ID count and prediction/input consistency |
| 680-concept actual lexicon | `02_Experiments/02_ECKE/03_Configuration_and_Instructions/xiaoke_lexicon_normalized_v1.json` and corresponding XLSX | complete concept count; historical original/JSON comparison note retained |
| ECKE Table 3 / Figure 2 | `02_Experiments/02_ECKE/04_Results/Test_Predictions/` | TXT score recomputed; structured JSONL score cross-checked; bootstrap and categories recomputed |
| CENTRELINE [11] | `02_Experiments/03_External_Comparisons/04_Results/D1-D3/D1_seed_*.jsonl` | three runs evaluated separately, then arithmetic mean |
| TCMERE [12] | Same directory, `D2_repair_seed_*.jsonl` | final repaired implementation; three runs, not obsolete D2 outputs |
| BSTR-GPRel [13] | Same directory, `D3_seed_*.jsonl` | three runs, arithmetic mean |
| CARE [28] | `02_Experiments/03_External_Comparisons/CARE_Comparison/01_Experiment_Code/CARE_Balanced/results_calibrated_v1/seed_*/predictions.jsonl` | three runs using folded entities; development calibration retained |
| OneKE-Qwen [24] | `02_Experiments/03_External_Comparisons/04_Results/E1/E1_OneKE-Qwen_300条正式预测.jsonl` | 300 records; development example IDs retained |
| YAYI-UIE [29] | `02_Experiments/03_External_Comparisons/04_Results/E3/E3_YAYI-UIE_300_predictions.jsonl` | 300 records |
| LLMRE [30] | `02_Experiments/03_External_Comparisons/LLMRE_Comparison/Results/Formal_300/预测300.txt` | final completion hash and 300 predictions; response-cache export in `03_Baseline_Packages/Original_Run_Records/LLMRE/` |
| ADELIE [31] | `03_Baseline_Packages/08_ADELIE/Experiment_Workspace/results_optimized/ADELIE_full300_predictions.jsonl` | all 300 records including 79 failures; raw response files retained |
| Table 4 A1–A4 | `02_Experiments/04_Ablation_Studies/04_Results/A*/A*_300条结构化预测.jsonl` | exact scores; A2 category results recomputed |
| Oracle | `02_Experiments/02_ECKE/05_Evaluation_and_Verification/Gold_Entity_Oracle_and_Error_Propagation/oracle_relation_cache.jsonl` | final cached outputs; not rerun with API |
| Entity-to-relation propagation | Structured ECKE test predictions + Gold | recompute 82 missing-endpoint false negatives and four with endpoints present |
| Full-corpus counts | `02_Experiments/02_ECKE/04_Results/Full_Corpus_Extraction/xiaoke_full_corpus_predictions_v612.jsonl` | 8176 entity mentions, 4867 relations, 2378 distinct entity expressions, 3220 surface-only distinct relations |
| Figure 3 Top100 / 104 nodes | `04_Figure_Data/Top100_关系及来源.csv` | every frequency and source ID checked against full predictions |

The verification command reproduces the scores reported in the submitted version; it does not add new experimental conditions or statistical claims.
