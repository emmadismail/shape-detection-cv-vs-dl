import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, models

DATASET_DIR = os.path.join("dataset", "geometric shapes dataset")
MODEL_PATH = os.path.join("models", "resnet_shape_classifier.pth")

TRAIN_PER_CLASS = 1500
VAL_PER_CLASS = 300
BATCH_SIZE = 32

random.seed(42)
torch.manual_seed(42)
device = torch.device("cpu")

# Load the saved model
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Rebuild the exact same validation split used during training
# (same seed, same logic as train_resnet.py, so this reconstructs the
# identical held-out images - not a new random sample)
val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_base = datasets.ImageFolder(root=DATASET_DIR, transform=val_transform)

indices_by_class = {i: [] for i in range(len(val_base.classes))}
for idx, (_, label) in enumerate(val_base.samples):
    indices_by_class[label].append(idx)

val_indices = []
for label, indices in indices_by_class.items():
    random.shuffle(indices)
    val_indices.extend(indices[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS])

val_dataset = Subset(val_base, val_indices)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Evaluating on {len(val_dataset)} held-out validation images...")

correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"Validation accuracy: {accuracy:.2f}% ({correct}/{total} correct)")