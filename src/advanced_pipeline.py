"""Advanced, research-only utilities for the BreaKHis classifier.

This module is intentionally framework-light around the existing notebook. It
adds patient-group inspection, clinically relevant metrics, Grad-CAM,
robustness checks, calibration, and optional ONNX export.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn


PATIENT_PATTERN = re.compile(r"^SOB_[BM]_[A-Z]-(\d+-[^-]+)-\d+-\d+$")


def patient_id(image_path: Path) -> str:
    """Extract the patient identifier encoded in a BreaKHis filename."""
    match = PATIENT_PATTERN.match(image_path.stem)
    if match is None:
        raise ValueError(f"Unexpected BreaKHis filename: {image_path.name}")
    return match.group(1)


def check_patient_disjointness(
    train_paths: Iterable[Path], test_paths: Iterable[Path]
) -> set[str]:
    """Return any patient groups appearing in both train and test."""
    train_patients = {patient_id(path) for path in train_paths}
    test_patients = {patient_id(path) for path in test_paths}
    return train_patients & test_patients


def binary_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate sensitivity, specificity, F1, ROC-AUC, and PR-AUC."""
    y_true = np.asarray(list(labels), dtype=np.int64)
    y_prob = np.asarray(list(probabilities), dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def expected_calibration_error(
    labels: Iterable[int],
    probabilities: Iterable[float],
    bins: int = 10,
) -> float:
    """Estimate expected calibration error using equal-width confidence bins."""
    y_true = np.asarray(list(labels), dtype=np.int64)
    y_prob = np.asarray(list(probabilities), dtype=np.float64)
    predictions = (y_prob >= 0.5).astype(np.int64)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    correctness = (predictions == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence >= lower) & (confidence < upper)
        if np.any(in_bin):
            error += np.mean(in_bin) * abs(
                np.mean(confidence[in_bin]) - np.mean(correctness[in_bin])
            )
    return float(error)


class ResNetGradCAM:
    """Grad-CAM for a torchvision ResNet-style classifier."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(
        self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_inputs: tuple[torch.Tensor | None, ...],
        grad_outputs: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_outputs[0].detach()

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        index = int(logits.argmax(dim=1).item()) if class_index is None else class_index
        logits[:, index].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = torch.nn.functional.interpolate(
            heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        return (heatmap - heatmap.amin(dim=(2, 3), keepdim=True)) / (
            heatmap.amax(dim=(2, 3), keepdim=True) + 1e-8
        )

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def apply_robustness_variant(image: Image.Image, variant: str) -> Image.Image:
    """Apply a deterministic image degradation for robustness evaluation."""
    if variant == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    if variant == "contrast":
        return ImageEnhance.Contrast(image).enhance(0.5)
    if variant == "brightness":
        return ImageEnhance.Brightness(image).enhance(0.7)
    if variant == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=35)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        return compressed.copy()
    raise ValueError(f"Unsupported robustness variant: {variant}")


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    image_size: int = 224,
) -> None:
    """Export a trained classifier with a fixed RGB image input shape."""
    model.eval()
    example = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        example,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
