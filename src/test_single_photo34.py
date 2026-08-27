from opencv_pipeline import detect_and_classify

detections = detect_and_classify(
    "test_photos/click_photos/34.jpg",
    debug=True,
    debug_prefix="outputs/debug34"
)
print(f"{len(detections)} detected")
for d in detections:
    print(d)