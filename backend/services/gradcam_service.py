"""
GradCAM Service
===============
Generates gradient-weighted class activation maps for explainable AI.

GradCAM highlights the regions of the MRI that most influenced the
model's prediction. This gives radiologists and users a visual
explanation of WHERE the AI is "looking."

Implementation:
- Hooks into the last convolutional layer of EfficientNet-B0
- Computes gradients of the target class score w.r.t. feature maps
- Produces a heatmap overlaid on the original image

Reference: Selvaraju et al., 2017 (https://arxiv.org/abs/1610.02391)
"""

import io
import uuid
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from loguru import logger

from backend.core.config import settings
from backend.api.schemas.schemas import GradCAMResult


# ─────────────────────────────────────────────────────────────
# Brain region heuristics based on activation location
# ─────────────────────────────────────────────────────────────
REGION_HINTS = {
    "top-left":     "Left Frontal Lobe",
    "top-center":   "Frontal Lobe",
    "top-right":    "Right Frontal Lobe",
    "center-left":  "Left Temporal/Parietal",
    "center":       "Central / Basal Ganglia",
    "center-right": "Right Temporal/Parietal",
    "bottom-left":  "Left Occipital / Cerebellum",
    "bottom-center":"Brainstem / Cerebellum",
    "bottom-right": "Right Occipital / Cerebellum",
}

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GradCAMService:
    """
    Generates GradCAM explanations for EfficientNet-B0 predictions.

    Uses the last feature block (features[-1]) as the target layer
    because it has the highest semantic content while maintaining
    sufficient spatial resolution for localization.
    """

    def __init__(self, classifier_model, device: torch.device):
        self.classifier = classifier_model
        self.device = device
        self.outputs_dir = Path(settings.OUTPUTS_DIR)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Gradient and activation storage
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None
        self._hooks: list = []

    def _register_hooks(self):
        """
        Register forward and backward hooks on the last conv block.
        EfficientNet-B0's last block is features[-1].
        """
        target_layer = self.classifier.features[-1]

        def save_activations(module, input, output):
            self._activations = output.detach()

        def save_gradients(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks = [
            target_layer.register_forward_hook(save_activations),
            target_layer.register_full_backward_hook(save_gradients),
        ]

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _compute_heatmap(self, target_class_idx: int, tensor: torch.Tensor) -> np.ndarray:
        """
        Core GradCAM computation.

        Steps:
        1. Forward pass → get class score
        2. Backward pass → get gradients at target layer
        3. Global average pool gradients → importance weights
        4. Weighted sum of activation maps → heatmap
        5. ReLU (keep only positive contributions)
        6. Normalize to [0, 1]
        """
        self.classifier.eval()
        self._register_hooks()

        try:
            # Forward pass (needs grad)
            tensor.requires_grad_(True)
            logits = self.classifier(tensor)

            # Backward pass for target class
            self.classifier.zero_grad()
            score = logits[0, target_class_idx]
            score.backward()

            # Compute importance weights: global average pool of gradients
            # Shape: (C,) — one weight per feature map channel
            weights = self._gradients.mean(dim=(2, 3))[0]  # (C,)

            # Weighted sum of activation maps
            # activations shape: (1, C, H, W) → (C, H, W)
            activation_maps = self._activations[0]  # (C, H, W)

            # Weighted combination: sum over channels
            heatmap = torch.zeros(activation_maps.shape[1:], device=self.device)
            for i, w in enumerate(weights):
                heatmap += w * activation_maps[i]

            # Apply ReLU — only positive contributions matter
            heatmap = F.relu(heatmap)

            # Normalize to [0, 1]
            heatmap = heatmap.cpu().numpy()
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        finally:
            self._remove_hooks()

        return heatmap

    def _heatmap_to_colormap(self, heatmap: np.ndarray, original_size: tuple) -> np.ndarray:
        """
        Resize heatmap to original image size and apply JET colormap.
        Returns BGR image (OpenCV format).
        """
        heatmap_resized = cv2.resize(heatmap, (original_size[0], original_size[1]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return colored  # BGR

    def _create_overlay(
        self, original_bgr: np.ndarray, colored_heatmap: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """
        Blend original image with GradCAM heatmap.
        alpha controls heatmap opacity.
        """
        return cv2.addWeighted(original_bgr, 1 - alpha, colored_heatmap, alpha, 0)

    def _estimate_attention_region(self, heatmap: np.ndarray) -> str:
        """
        Estimate brain region based on center of mass of the heatmap.
        Divides the image into a 3×3 grid and returns the region name.
        """
        # Find center of mass of the heatmap
        h, w = heatmap.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        total = heatmap.sum()

        if total < 1e-6:
            return "Indeterminate"

        cx = float((x_coords * heatmap).sum() / total)
        cy = float((y_coords * heatmap).sum() / total)

        # Map to 3×3 grid
        col = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
        row = "top" if cy < h / 3 else ("center" if cy < 2 * h / 3 else "bottom")

        if row == "center" and col == "center":
            key = "center"
        else:
            key = f"{row}-{col}"

        return REGION_HINTS.get(key, "Unspecified Region")

    def _save_image(self, img_array: np.ndarray, filename: str) -> str:
        """Save image to outputs directory and return relative URL."""
        path = self.outputs_dir / filename
        cv2.imwrite(str(path), img_array)
        return f"/outputs/{filename}"

    async def generate(
        self, image_bytes: bytes, image_id: str, target_class_idx: int, class_name: str
    ) -> GradCAMResult:
        """
        Generate GradCAM visualization for a given prediction.

        Returns paths to:
        - Original image (saved copy)
        - Heatmap-only image
        - Overlay (original + heatmap blended)
        """
        # Load original image
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        orig_w, orig_h = pil_img.size

        # Prepare tensor for model
        tensor = INFERENCE_TRANSFORM(pil_img).unsqueeze(0).to(self.device)

        # Compute GradCAM heatmap
        heatmap = self._compute_heatmap(target_class_idx, tensor)

        # Generate visualizations
        colored_heatmap = self._heatmap_to_colormap(heatmap, (orig_w, orig_h))
        overlay = self._create_overlay(original_bgr, colored_heatmap)

        # Save all three images
        original_url = self._save_image(original_bgr, f"{image_id}_original.jpg")
        heatmap_url = self._save_image(colored_heatmap, f"{image_id}_heatmap.jpg")
        overlay_url = self._save_image(overlay, f"{image_id}_overlay.jpg")

        # Estimate attention region
        attention_region = self._estimate_attention_region(heatmap)

        explanation = (
            f"The AI model focused primarily on the {attention_region} area when "
            f"classifying this scan as '{class_name}'. The heatmap shows regions "
            f"with the highest influence on the prediction — warmer colors (red/yellow) "
            f"indicate areas of strongest AI attention."
        )

        return GradCAMResult(
            original_image_url=original_url,
            heatmap_url=heatmap_url,
            overlay_url=overlay_url,
            attention_region=attention_region,
            explanation=explanation,
        )
