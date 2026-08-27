import csv

GT_CSV = "test_photos/ground_truth.csv"
OUT_CSV = "test_photos/ground_truth_with_shapes.csv"

rows = []
with open(GT_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"\n{row['filename']} (true_count={row['true_count']}, condition={row['condition']})")
        shapes = input("  List true shapes, comma-separated (e.g. cube,cube,pyramid,sphere): ")
        row["true_shapes"] = shapes.strip()
        rows.append(row)

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "true_count", "condition", "true_shapes"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved to {OUT_CSV}")