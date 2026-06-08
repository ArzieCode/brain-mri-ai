"""
Image Validator - MobileNetV3-Small
=====================================
Lightweight classifier that determines whether an uploaded image
is a valid brain MRI before passing it to the tumor classifier.

This is a CRITICAL safety gate — it prevents the tumor classifier
from making predictions on non-medical or irrelevant images.

Classes:
- brain_mri      → valid, proceed to tumor classification
- xray           → invalid, wrong modality
- blood_cell     → invalid, wrong organ
- natural_image  → invalid, not medical
- non_medical    → invalid, not medical
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ImageValidator(nn.Module):
    """
    MobileNetV3-Small based image type classifier.

    Chosen for its speed (critical for the validation gate):
    ~2ms per image on Apple M1, ~5ms on CPU.
    """

    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.2):
        super().__init__()

        # ── Backbone ─────────────────────────────────────────
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Reuse all layers except the final classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # MobileNetV3-Small: 576 → 1024 → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_image_validator(num_classes: int = 5, pretrained: bool = True) -> ImageValidator:
    """Factory function to build and return the image validator."""
    return ImageValidator(num_classes=num_classes, dropout_rate=0.2)
