TCMERE D2 repair package v1

Changes are limited to D2:
1. Separate optimizers following the paper: encoder AdamW 5e-5; extraction Adam 8e-4; CompGCN Adam 1e-4.
2. Jointly select epochs and entity/relation thresholds only within five-fold development on 100 records.
3. A zero five-fold mean entity or relation F1 blocks the formal 300-record run.
4. Test Gold is not included and participates in no selection.
5. Write results/D2_repair_v1 without replacing D1, D3 or the older D2 condition.

From AutoDL_D1-D3_Package_v2, with this repair folder restored there:
python D2_repair_v1/00_run_all_d2_repair.py

The entry gates five-fold development, formal prediction and output audit; any failed stage stops subsequent processing.
