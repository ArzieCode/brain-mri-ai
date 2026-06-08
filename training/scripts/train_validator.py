"""
Train Image Validator (MobileNetV3-Small)
==========================================
Trains the lightweight classifier that validates whether an uploaded
image is a valid brain MRI before passing it to the tumor classifier.

Dataset structure expected:
  datasets/image_validator/
    train/
      brain_mri/      ← real brain MRI scans
      xray/           ← chest X-rays (wrong modality)
      blood_cell/     ← blood cell microscopy
      natural_image/  ← natural photographs
      non_medical/    ← other non-medical images
    val/
      ...

Tip: Use public datasets:
- Brain MRI: Kaggle Brain Tumor MRI Dataset
- X-Ray: NIH Chest X-Ray Dataset
- Blood cells: Blood Cell Images (Kaggle)
- Natural images: Imagenette (fast.ai)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json

torch.manual_seed(42)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

CONFIG = {
    "dataset_root": "datasets/image_validator",
    "models_dir": "models",
    "output_name": "mobilenet_v3_validator.pth",
    "image_size": 224,
    "batch_size": 64,
    "num_workers": 0,
    "epochs": 30,
    "lr": 5e-4,
    "early_stopping_patience": 6,
    "classes": ["brain_mri", "xray", "blood_cell", "natural_image", "non_medical"],
    "num_classes": 5,
}

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_validator(num_classes):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0

    for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return correct / total, np.array(all_preds), np.array(all_labels)


def main():
    dataset_root = Path(CONFIG["dataset_root"])
    if not dataset_root.exists():
        print(f"❌ Validator dataset not found at {dataset_root}")
        print("   See script docstring for dataset setup instructions")
        sys.exit(1)

    train_ds = datasets.ImageFolder(dataset_root / "train", TRAIN_TRANSFORMS)
    val_ds = datasets.ImageFolder(dataset_root / "val", VAL_TRANSFORMS)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = build_validator(CONFIG["num_classes"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    models_dir = Path(CONFIG["models_dir"])
    models_dir.mkdir(exist_ok=True)
    best_path = models_dir / CONFIG["output_name"]

    best_acc = 0.0
    no_improve = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc, _, _ = evaluate(model, val_loader)
        print(f"Epoch {epoch:3d} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "val_accuracy": best_acc}, best_path)
            print(f"  💾 Best validator saved! Acc: {best_acc*100:.2f}%")
        else:
            no_improve += 1
            if no_improve >= CONFIG["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final report
    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_preds, final_labels = evaluate(model, val_loader)
    print("\n" + classification_report(final_labels, final_preds, target_names=CONFIG["classes"]))
    print(f"✅ Validator saved: {best_path}")


if __name__ == "__main__":
    main()
