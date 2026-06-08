"""
Train Tumor Classifier
======================
Complete training pipeline for EfficientNet-B0 brain tumor classifier.

Dataset structure expected:
  datasets/brain_tumor/
    train/
      glioma/         ← tumor images
      meningioma/
      pituitary/
      normal/
    val/
      glioma/
      ...

Recommended dataset: Brain Tumor MRI Dataset (Kaggle)
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Features:
- Transfer learning from ImageNet
- Medically safe augmentations
- Class-balanced sampling to reduce bias
- Early stopping
- Learning rate scheduling
- Mixed precision (MPS/CUDA)
- Comprehensive metrics: accuracy, precision, recall, F1, AUC
- Confusion matrix
- Best model checkpoint saving
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Reproducibility ────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Device Selection ───────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("🍎 Using Apple Silicon MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"🟢 Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("⚠️  Using CPU")
    return torch.device("cpu")

DEVICE = get_device()

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "dataset_root": "datasets/brain_tumor",
    "models_dir": "models",
    "output_name": "efficientnet_b0_tumor.pth",
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 4 if sys.platform != "darwin" else 0,  # 0 on macOS
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3,
    "early_stopping_patience": 8,
    "classes": ["glioma", "meningioma", "pituitary", "normal"],
    "num_classes": 4,
}

# ─────────────────────────────────────────────────────────────
# Medically Safe Data Augmentation
# ─────────────────────────────────────────────────────────────
# IMPORTANT: MRI augmentations must not introduce artificial patterns.
# Only use transforms that could plausibly occur due to patient/scanner variation.

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((CONFIG["image_size"] + 32, CONFIG["image_size"] + 32)),
    transforms.RandomCrop(CONFIG["image_size"]),
    transforms.RandomHorizontalFlip(p=0.5),       # Anatomically acceptable for axial slices
    transforms.RandomRotation(degrees=10),          # Small rotation: scanner positioning variation
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,               # Scanner intensity variation
        saturation=0.0, hue=0.0,                    # NO hue/saturation (grayscale MRI)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
def build_model(num_classes: int, dropout_rate: float) -> nn.Module:
    """Build EfficientNet-B0 with custom classification head."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze early backbone layers (optional - unfreeze for full fine-tuning)
    # Only fine-tune the last few blocks + classifier head
    for name, param in model.named_parameters():
        if "features.0" in name or "features.1" in name or "features.2" in name:
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, num_classes),
    )

    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────
# Bias Reduction: Balanced Class Sampling
# ─────────────────────────────────────────────────────────────
def create_balanced_sampler(dataset) -> WeightedRandomSampler:
    """
    Create a sampler that oversamples underrepresented classes.
    This is critical for medical datasets which often have class imbalance
    (e.g., fewer normal cases than tumor cases).

    Bias reduction strategy: inverse frequency weighting.
    Each sample's weight = 1 / class_count
    """
    targets = [label for _, label in dataset.imgs]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]

    print(f"📊 Class distribution: {dict(zip(dataset.classes, class_counts))}")
    print(f"📊 Class weights: {dict(zip(dataset.classes, class_weights.round(4)))}")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type != "mps"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute comprehensive metrics
    metrics = {
        "loss": total_loss / total,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    # ROC-AUC (one-vs-rest for multi-class)
    try:
        metrics["auc"] = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="weighted"
        )
    except ValueError:
        metrics["auc"] = 0.0

    return metrics, all_preds, all_labels


# ─────────────────────────────────────────────────────────────
# Confusion Matrix Plot
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix — Brain Tumor Classifier", fontsize=13)
    plt.ylabel("True Label"), plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Main Training Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    # ── Data Loading ─────────────────────────────────────────
    dataset_root = Path(CONFIG["dataset_root"])
    if not dataset_root.exists():
        print(f"❌ Dataset not found at {dataset_root}")
        print("   Download from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(dataset_root / "train", transform=TRAIN_TRANSFORMS)
    val_dataset = datasets.ImageFolder(dataset_root / "val", transform=VAL_TRANSFORMS)

    print(f"📦 Train samples: {len(train_dataset)}")
    print(f"📦 Val samples:   {len(val_dataset)}")
    print(f"📦 Classes: {train_dataset.classes}")

    # Balanced sampler for bias reduction
    sampler = create_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    # ── Model, Loss, Optimizer ───────────────────────────────
    model = build_model(CONFIG["num_classes"], CONFIG["dropout_rate"])

    # Class-weighted loss to further reduce bias
    class_counts = np.bincount([l for _, l in train_dataset.imgs])
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # GradScaler for mixed precision (CUDA only; MPS has native support)
    scaler = torch.GradScaler(enabled=DEVICE.type == "cuda")

    # ── Training Loop ────────────────────────────────────────
    best_val_f1 = 0.0
    no_improve_count = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    models_dir = Path(CONFIG["models_dir"])
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / CONFIG["output_name"]

    print(f"\n{'='*60}")
    print(f"Training EfficientNet-B0 Tumor Classifier")
    print(f"Device: {DEVICE} | Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']}")
    print(f"{'='*60}\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_metrics, val_preds, val_labels = validate(model, val_loader, criterion)

        scheduler.step()
        epoch_time = time.time() - t0

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
            f"Time: {epoch_time:.1f}s | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics.get('auc', 0):.4f}"
        )

        # ── Save Best Model ──────────────────────────────────
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_accuracy": val_metrics["accuracy"],
                "config": CONFIG,
            }, best_model_path)
            print(f"  💾 Best model saved! F1: {best_val_f1:.4f}")
        else:
            no_improve_count += 1

        # ── Early Stopping ───────────────────────────────────
        if no_improve_count >= CONFIG["early_stopping_patience"]:
            print(f"\n⏹️  Early stopping at epoch {epoch} (no improvement for {no_improve_count} epochs)")
            break

    # ── Final Evaluation ────────────────────────────────────
    print(f"\n{'='*60}")
    print("Final Evaluation on Validation Set (Best Model)")
    print(f"{'='*60}")

    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics, final_preds, final_labels = validate(model, val_loader, criterion)

    print(classification_report(final_labels, final_preds, target_names=CONFIG["classes"]))
    print(f"Best Val F1:  {best_val_f1:.4f}")
    print(f"Final Val Accuracy: {final_metrics['accuracy']*100:.2f}%")
    print(f"Final AUC: {final_metrics.get('auc', 0):.4f}")

    # Confusion matrix
    plot_confusion_matrix(
        final_labels, final_preds, CONFIG["classes"],
        models_dir / "confusion_matrix.png"
    )

    # Save training history
    with open(models_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete! Model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
