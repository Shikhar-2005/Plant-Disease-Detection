"""
transforms.py — Image preprocessing pipelines for training and inference.

Usage:
    from src.transforms import get_train_transforms, get_inference_transforms
"""

from torchvision import transforms

# ImageNet statistics (must match ResNet50 pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def get_train_transforms() -> transforms.Compose:
    """
    Returns augmentation pipeline used during training.
    Applies random flips, crops, colour jitter and rotation to prevent overfitting.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Returns deterministic preprocessing pipeline used for validation, test, and inference.
    No augmentation — only resize, centre-crop, and normalise.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
