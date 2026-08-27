import cv2
import os

SAMPLE_FILES = ["1.jpg", "19.jpg", "34.jpg"] 
SRC_DIR = "test_photos/click_photos"
OUT_DIR = "outputs/crop_preview"
os.makedirs(OUT_DIR, exist_ok=True)

TOP_CROP_PCT = 0.16
LEFT_CROP_PCT = 0.17
RIGHT_CROP_PCT = 0.05     
BOTTOM_CROP_PCT = 0.0     

for filename in SAMPLE_FILES:
    img = cv2.imread(os.path.join(SRC_DIR, filename))
    h, w = img.shape[:2]

    top = int(TOP_CROP_PCT * h)
    left = int(LEFT_CROP_PCT * w)
    right = int((1 - RIGHT_CROP_PCT) * w)
    bottom = int((1 - BOTTOM_CROP_PCT) * h)

    cropped = img[top:bottom, left:right]
    cv2.imwrite(os.path.join(OUT_DIR, f"preview_{filename}"), cropped)

print(f"Saved previews to {OUT_DIR}")