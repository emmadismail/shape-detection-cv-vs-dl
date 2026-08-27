import os
import csv
import cv2

TEST_PHOTOS_DIR = "test_photos/click_photos"
OUTPUT_CSV = "test_photos/ground_truth.csv"

# Sort numerically (1, 2, 3 ... 40) instead of alphabetically (1, 10, 11 ...)
files = sorted(
    [f for f in os.listdir(TEST_PHOTOS_DIR) if f.lower().endswith(".jpg")],
    key=lambda name: int(os.path.splitext(name)[0])
)

results = []

for filename in files:
    img_path = os.path.join(TEST_PHOTOS_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not read {filename}, skipping")
        continue

    display = cv2.resize(img, (960, 720))
    cv2.imshow(f"{filename} - press any key here, then answer in terminal", display)
    cv2.waitKey(1)

    print(f"\nPhoto: {filename}")
    count = input("  How many objects are in this photo? ")
    condition = input("  Condition tag (clean/close/shadow/angle/occluded/other): ")

    results.append({"filename": filename, "true_count": count, "condition": condition})
    cv2.destroyAllWindows()

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "true_count", "condition"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved ground truth to {OUTPUT_CSV}")