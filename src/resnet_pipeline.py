import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn

from opencv_pipeline import detect_and_classify as detect_only, find_table_mask

MODEL_PATH = "models/resnet_shape_classifier.pth"
device = torch.device("cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def classify_crop(bgr_crop):
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return class_names[pred.item()], conf.item()


def detect_and_classify_resnet(image_path, debug=False, debug_prefix="outputs/debug_resnet"):
    """Same detection as Track A, but classification comes from the ResNet."""
    img = cv2.imread(image_path)
    scale = 900 / img.shape[1]
    img = cv2.resize(img, (900, int(img.shape[0] * scale)))
    top_crop = int(0.16 * img.shape[0])
    left_crop = int(0.17 * img.shape[1])
    img = img[top_crop:, left_crop:]

    detections = detect_only(image_path, debug=False)

    results = []
    output = img.copy()
    for d in detections:
        x, y, w, h = d["bbox"]
        crop = img[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        shape, confidence = classify_crop(crop)
        results.append({"shape": shape, "confidence": confidence, "bbox": d["bbox"]})

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{shape} ({confidence:.0%})"
        cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)

    if debug:
        cv2.imwrite(f"{debug_prefix}_output.jpg", output)

    return results