from opencv_pipeline import detect_and_classify

detections = detect_and_classify(
    "test_photos/click_photos/1.jpg",
    debug=True,
    debug_prefix="outputs/debug1"
)
print(f"{len(detections)} detected")
for d in detections:
    print(d)