"""
model.py — Model loading and inference utilities for Plant Disease Detection.

Usage:
    model, class_names = load_model("models/plant_disease_resnet50.pth")
    result = predict(model, class_names, pil_image)
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image

from src.transforms import get_inference_transforms

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38


# ── Model builder ────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Builds the ResNet50 architecture with a custom classification head.

    Layer strategy:
        - Layers 1–3 : frozen (pretrained ImageNet features)
        - Layer 4    : trainable (domain-specific fine-tuning)
        - FC head    : replaced with Linear(2048 → num_classes)

    Args:
        num_classes: Number of output classes (default 38 for PlantVillage).

    Returns:
        ResNet50 model (not yet loaded with weights — call load_model instead).
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classification head
    in_features = model.fc.in_features          # 2048
    model.fc = nn.Linear(in_features, num_classes)

    return model.to(DEVICE)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str | Path,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """
    Loads a trained ResNet50 model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint saved during training.
        num_classes:     Number of output classes.

    Returns:
        Model in eval mode on the appropriate device (CUDA / CPU).

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_class_names(json_path: str | Path) -> list[str]:
    """
    Loads the class names list from a JSON file.

    Args:
        json_path: Path to class_names.json (list of strings).

    Returns:
        List of class name strings, e.g. ["Apple___Apple_scab", ...].
    """
    with open(json_path, "r") as f:
        return json.load(f)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    model: nn.Module,
    class_names: list[str],
    image: Image.Image,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Runs inference on a PIL image and returns structured prediction results.

    Args:
        model:       Loaded ResNet50 model in eval mode.
        class_names: Ordered list of class names (must match training order).
        image:       PIL Image (RGB) — any size, will be preprocessed internally.
        top_k:       Number of top predictions to return.

    Returns:
        Dictionary with keys:
            - "class_name"   (str)  : Human-readable predicted class
            - "class_raw"    (str)  : Raw folder name, e.g. "Tomato___Early_blight"
            - "confidence"   (float): Probability of top prediction (0–1)
            - "is_healthy"   (bool) : True if "healthy" in class name
            - "top_k"        (list) : List of dicts {class_name, confidence} for top K
            - "image_tensor" (Tensor): Preprocessed tensor (for Grad-CAM)
    """
    transform = get_inference_transforms()
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)           # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)  # (1, num_classes)

    top_probs, top_indices = probs.topk(min(top_k, len(class_names)), dim=1)
    top_probs   = top_probs[0].cpu().tolist()
    top_indices = top_indices[0].cpu().tolist()

    # Best prediction
    best_idx  = top_indices[0]
    best_raw  = class_names[best_idx]
    best_prob = top_probs[0]

    # Human-readable name: "Tomato___Early_blight" → "Tomato — Early Blight"
    def humanise(raw: str) -> str:
        parts = raw.split("___")
        if len(parts) == 2:
            plant, disease = parts
            disease = disease.replace("_", " ").title()
            return f"{plant} — {disease}"
        return raw.replace("_", " ").title()

    top_k_results = [
        {"class_name": humanise(class_names[i]), "class_raw": class_names[i], "confidence": p}
        for i, p in zip(top_indices, top_probs)
    ]

    return {
        "class_name":   humanise(best_raw),
        "class_raw":    best_raw,
        "confidence":   best_prob,
        "is_healthy":   "healthy" in best_raw.lower(),
        "top_k":        top_k_results,
        "image_tensor": image_tensor,          # kept on device for Grad-CAM
    }
