"""
Tumor Classifier - EfficientNet-B0
=====================================
Transfer learning classifier for brain tumor detection.

Architecture:
- EfficientNet-B0 backbone (pretrained on ImageNet)
- Custom classification head with dropout for uncertainty
- Monte Carlo Dropout compatible (dropout active during inference for uncertainty)

Classes: glioma, meningioma, pituitary, normal
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class TumorClassifier(nn.Module):
    """
    EfficientNet-B0 based brain tumor classifier.

    The custom head uses dropout which remains active during uncertainty
    estimation (Monte Carlo Dropout) via train() mode for inference passes.
    """

    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3):
        super().__init__()

        # ── Backbone ─────────────────────────────────────────
        # Load EfficientNet-B0 with ImageNet pretrained weights
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Extract feature layers (everything except the final classifier)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # EfficientNet-B0 outputs 1280-dim feature vectors
        in_features = backbone.classifier[1].in_features

        # ── Custom Classification Head ────────────────────────
        # Two-layer MLP with BatchNorm and Dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),          # MC-Dropout layer
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),          # MC-Dropout layer
            nn.Linear(256, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings for OOD detection."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def enable_mc_dropout(self):
        """
        Enable Monte Carlo Dropout for uncertainty estimation.
        Sets only dropout layers to train mode while keeping
        BatchNorm in eval mode (to preserve running stats).
        """
        self.eval()  # Start from eval state
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Re-enable dropout only

    def disable_mc_dropout(self):
        """Return to standard eval mode."""
        self.eval()


def build_tumor_classifier(num_classes: int = 4, pretrained: bool = True) -> TumorClassifier:
    """Factory function to build and return the tumor classifier."""
    model = TumorClassifier(num_classes=num_classes, dropout_rate=0.3)
    return model
