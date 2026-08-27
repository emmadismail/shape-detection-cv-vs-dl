import os
import csv
from opencv_pipeline import detect_and_classify

TEST_DIR = "test_photos/click_photos"
GT_CSV = "test_photos/ground_truth.csv"
OUT_CSV = "outputs/baseline_results.csv"

ground_truth = {}
with open(GT_CSV, newline="") as f:
    for row in csv.DictReader(f):
        ground_truth[row["filename"]] = {
            "true_count": int(row["true_count"]),
            "condition": row["condition"]
        }

rows = []
for filename, gt in ground_truth.items():
    path = os.path.join(TEST_DIR, filename)
    try:
        detections = detect_and_classify(path, debug=False)
        detected_count = len(detections)
    except Exception as e:
        print(f"Error on {filename}: {e}")
        continue

    true_count = gt["true_count"]

    efficiency = 1 - abs(detected_count - true_count) / true_count if true_count > 0 else 0
    efficiency = max(0, efficiency) * 100

    rows.append({
        "filename": filename,
        "condition": gt["condition"],
        "true_count": true_count,
        "detected_count": detected_count,
        "efficiency_pct": round(efficiency, 2)
    })
    print(f"{filename:10s} [{gt['condition']:9s}] true={true_count:3d} detected={detected_count:3d} eff={efficiency:.1f}%")

os.makedirs("outputs", exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "condition", "true_count", "detected_count", "efficiency_pct"])
    writer.writeheader()
    writer.writerows(rows)

overall_avg = sum(r["efficiency_pct"] for r in rows) / len(rows)
print(f"\nOverall average efficiency: {overall_avg:.2f}%")


conditions = sorted(set(r["condition"] for r in rows))
print("\nPer-condition breakdown:")
for cond in conditions:
    cond_rows = [r for r in rows if r["condition"] == cond]
    avg = sum(r["efficiency_pct"] for r in cond_rows) / len(cond_rows)
    print(f"  {cond:10s}: {avg:.2f}% (n={len(cond_rows)})")