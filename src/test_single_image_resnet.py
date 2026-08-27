from resnet_pipeline import detect_and_classify_resnet

TEST_IMAGE = "test_photos/click_photos/34.jpg"

results = detect_and_classify_resnet(TEST_IMAGE, debug=True)
print(f"Detected {len(results)} objects:")
for r in results:
    print(f"  {r['shape']} (confidence {r['confidence']:.2f}), bbox={r['bbox']}")