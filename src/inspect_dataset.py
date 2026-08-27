import os

DATASET_DIR = os.path.join("dataset", "geometric shapes dataset")

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if os.path.isdir(class_path):
        num_images = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        print(f"{class_name}: {num_images} images")