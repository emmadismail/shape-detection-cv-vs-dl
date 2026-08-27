import csv
from opencv_pipeline import detect_and_classify as track_a_detect
from resnet_pipeline import detect_and_classify_resnet as track_b_detect

GT_CSV = "test_photos/ground_truth_with_shapes.csv"
TEST_DIR = "test_photos/click_photos"
OUT_CSV = "outputs/comparison_results.csv"

TRACK_A_EXPECTED = {
    "triangle": "triangle",
    "rectangle": "rectangle",
    "circle": "other",          # Track A has no circle concept - matches the paper's own scheme
    "other": "other",           # Track A's catch-all bucket can correctly absorb ambiguous shapes
}

TRACK_B_EXPECTED = {
    "triangle": "Triangle",
    "rectangle": "Square",
    "circle": "Circle",
    "other": "__IMPOSSIBLE__", 
}


def multiset_overlap(expected_list, predicted_list):
    """Counts how many items match between two label multisets (order-independent)."""
    remaining = list(predicted_list)
    matches = 0
    for label in expected_list:
        if label in remaining:
            remaining.remove(label)
            matches += 1
    return matches


rows = []
with open(GT_CSV, newline="") as f:
    for row in csv.DictReader(f):
        filename = row["filename"]
        condition = row["condition"]
        true_count = int(row["true_count"])
        true_shapes = [s.strip() for s in row["true_shapes"].split(",")]

        path = f"{TEST_DIR}/{filename}"

        # --- Track A ---
        a_detections = track_a_detect(path, debug=False)
        a_predicted = [d["shape"] for d in a_detections]
        a_expected = [TRACK_A_EXPECTED[s] for s in true_shapes]
        a_matches = multiset_overlap(a_expected, a_predicted)
        a_class_acc = a_matches / true_count * 100

        # --- Track B ---
        b_detections = track_b_detect(path)
        b_predicted = [d["shape"] for d in b_detections]
        b_expected = [TRACK_B_EXPECTED[s] for s in true_shapes]
        b_matches = multiset_overlap(b_expected, b_predicted)
        b_class_acc = b_matches / true_count * 100

        rows.append({
            "filename": filename,
            "condition": condition,
            "true_count": true_count,
            "detected_count": len(a_predicted), 
            "track_a_classification_acc": round(a_class_acc, 1),
            "track_b_classification_acc": round(b_class_acc, 1),
        })

        print(f"{filename:8s} [{condition:7s}] true={true_count:2d}  "
              f"TrackA={a_class_acc:5.1f}%  TrackB={b_class_acc:5.1f}%")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

avg_a = sum(r["track_a_classification_acc"] for r in rows) / len(rows)
avg_b = sum(r["track_b_classification_acc"] for r in rows) / len(rows)
print(f"\nOverall - Track A: {avg_a:.1f}%   Track B: {avg_b:.1f}%")

print("\nPer-condition breakdown:")
for cond in sorted(set(r["condition"] for r in rows)):
    cond_rows = [r for r in rows if r["condition"] == cond]
    a = sum(r["track_a_classification_acc"] for r in cond_rows) / len(cond_rows)
    b = sum(r["track_b_classification_acc"] for r in cond_rows) / len(cond_rows)
    print(f"  {cond:8s} (n={len(cond_rows)}): Track A={a:.1f}%  Track B={b:.1f}%")