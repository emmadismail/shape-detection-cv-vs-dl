import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, models

DATASET_DIR = os.path.join("dataset", "geometric shapes dataset")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = 224
TRAIN_PER_CLASS = 1500
VAL_PER_CLASS = 300
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 0.001
MODEL_PATH = os.path.join(MODEL_DIR, "resnet_shape_classifier.pth")

random.seed(42)
torch.manual_seed(42)
device = torch.device("cpu")

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomAffine(degrees=15, scale=(0.9, 1.1), shear=5),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_base = datasets.ImageFolder(root=DATASET_DIR, transform=train_transform)
val_base = datasets.ImageFolder(root=DATASET_DIR, transform=val_transform)

class_names = train_base.classes
print("Classes found:", class_names)

indices_by_class = {i: [] for i in range(len(class_names))}
for idx, (_, label) in enumerate(train_base.samples):
    indices_by_class[label].append(idx)

train_indices, val_indices = [], []
for label, indices in indices_by_class.items():
    random.shuffle(indices)
    train_indices.extend(indices[:TRAIN_PER_CLASS])
    val_indices.extend(indices[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS])

train_dataset = Subset(train_base, train_indices)
val_dataset = Subset(val_base, val_indices)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

def run_epoch(loader, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = run_epoch(train_loader, True)
    val_loss, val_acc = run_epoch(val_loader, False)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
          f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": IMAGE_SIZE
        }, MODEL_PATH)
        print(f"  -> saved new best model ({best_val_acc:.4f})")

print("Training complete. Best val accuracy:", best_val_acc)