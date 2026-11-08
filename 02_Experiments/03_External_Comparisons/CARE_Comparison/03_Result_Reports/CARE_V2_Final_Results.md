# CARE version 2: formal results

Fixed 300-passage independent test set; strict set matching; mean ± sample SD over three seeds. Values below are percentages.

| Task | Precision | Recall | Micro-F1 |
|---|---:|---:|---:|
| Entity | 27.88±3.46 | 63.48±1.68 | 38.66±3.42 |
| Relation | 3.36±1.34 | 38.85±2.38 | 6.13±2.22 |

| Seed | Entity F1 | Relation F1 |
|---|---:|---:|
|20260823|35.16|4.55|
|20260824|41.99|8.66|
|20260825|38.82|5.18|

Both thresholds are 0.9999, chosen on the fixed five-fold models using only 100 development labels; model training was unchanged. Per-seed CIs, category scores, error summaries and paired statistics are in adjacent JSON files. Fold outputs participate in selection, so selection-set scores are not independent validation performance.

Version-2 calibration followed an earlier version-1 test evaluation; disclose that chronology. Test Gold did not select thresholds. Paired statistics use within-task Holm correction across three CARE seeds, not global correction over all external methods. The low relation F1 here does not establish CARE performance on other tasks.
