import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
records = [json.loads(line) for line in (root / "04_Test Labels and Predictions/01_test300_gold_and_ECKE_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
summary = json.loads((root / "05_Experimental Results/03_ECKE_performance_summary.json").read_text(encoding="utf-8"))
assert len(records) == len({row["record_id"] for row in records}) == 300
for task in ("entity", "relation"):
    suffix = "entities" if task == "entity" else "relations"
    totals = [0, 0, 0]
    for row in records:
        gold = {tuple(value) for value in row["gold_" + suffix]}
        predicted = {tuple(value) for value in row["ecke_" + suffix]}
        counts = (len(gold & predicted), len(predicted - gold), len(gold - predicted))
        totals = [left + right for left, right in zip(totals, counts)]
    tp, fp, fn = totals
    expected = summary[task]["micro"]
    assert totals == [expected[key] for key in ("tp", "fp", "fn")]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    assert abs(f1 - expected["f1"]) < 1e-12
    print(f"{task}: TP={tp}, FP={fp}, FN={fn}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
print("All 300 records and primary counts verified.")
