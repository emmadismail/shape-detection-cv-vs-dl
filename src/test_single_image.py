import os
from opencv_pipeline import detect_and_classify

os.makedirs("outputs/debug", exist_ok=True)

TEST_IMAGE = "test_photos/click_photos/34.jpg"  

detections = detect_and_classify(TEST_IMAGE, debug=True)

print(f"Detected {len(detections)} objects:")
for d in detections:
    print(f"  {d['shape']}, area={d['area']:.0f}, bbox={d['bbox']}")