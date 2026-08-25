"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet50.

Hooks into ResNet50's layer4 to produce heatmaps showing which leaf regions
influenced the model's prediction.

Usage:
    cam = GradCAM(model)
    heatmap = cam.generate(image_tensor, class_idx)
    overlay  = cam.overlay(original_pil_image, heatmap)
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for ResNet50 (hooks onto layer4).

    Args:
        model: A ResNet50 model in eval mode.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        # Hook onto layer4 (last convolutional block of ResNet50)
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap.

        Args:
            image_tensor: Preprocessed image tensor of shape (1, 3, 224, 224).
            class_idx:    Target class index. If None, uses the predicted class.

        Returns:
            heatmap: Float32 numpy array of shape (224, 224), values in [0, 1].
        """
        self.model.zero_grad()

        output = self.model(image_tensor)          # (1, num_classes)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backprop through the target class score
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Resize to 224×224 and normalise to [0, 1]
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.astype(np.float32)

    def overlay(
        self,
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """
        Blends the Grad-CAM heatmap over the original image.

        Args:
            pil_image: Original PIL image (any size — will be resized to 224×224).
            heatmap:   Float32 array in [0, 1] from generate().
            alpha:     Heatmap opacity (0 = invisible, 1 = full).
            colormap:  OpenCV colourmap (default: COLORMAP_JET).

        Returns:
            Blended PIL image (RGB, 224×224).
        """
        # Resize original to 224×224 for alignment
        img = pil_image.convert("RGB").resize((224, 224))
        img_np = np.array(img)

        # Colourmap the heatmap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(heatmap_uint8, colormap)
        coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

        # Blend
        blended = (alpha * coloured + (1 - alpha) * img_np).astype(np.uint8)
        return Image.fromarray(blended)
