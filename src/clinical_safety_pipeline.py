"""Auditable clinical-safety orchestration for the research imaging prototype.

Each pipeline component has one responsibility and exchanges typed dictionaries
only. This is not an autonomous diagnostic system and does not call an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageStat
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


@dataclass(frozen=True)
class ImagingReport:
    accepted: bool
    width: int
    height: int
    channels: int
    mean_brightness: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class VerificationReport:
    accepted: bool
    metrics: Mapping[str, float | int]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SafetyReport:
    allowed: bool
    flags: tuple[str, ...]


def _binary_metrics(
    labels: Sequence[int], probabilities: Sequence[float]
) -> dict[str, float | int]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
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
        "false_negatives": int(fn),
    }


def _expected_calibration_error(
    labels: Sequence[int], probabilities: Sequence[float], bins: int = 10
) -> float:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
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


def imaging_agent(
    image_path: Path,
    minimum_size: tuple[int, int] = (64, 64),
) -> ImagingReport:
    """Validate readable RGB input and basic quality indicators."""
    warnings: list[str] = []
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            channels = len(rgb.getbands())
            brightness = float(ImageStat.Stat(rgb).mean[0] / 255.0)
            if width < minimum_size[0] or height < minimum_size[1]:
                warnings.append("image_resolution_below_minimum")
            if brightness < 0.05 or brightness > 0.95:
                warnings.append("extreme_brightness")
    except (OSError, ValueError) as error:
        return ImagingReport(False, 0, 0, 0, 0.0, (f"image_read_error:{error}",))
    return ImagingReport(not warnings, width, height, channels, brightness, tuple(warnings))


def inference_agent(
    predictor: Callable[[Path], Mapping[str, object]],
    image_path: Path,
) -> dict[str, object]:
    """Run only the supplied classifier/segmenter and preserve its output."""
    result = dict(predictor(image_path))
    required = {"label", "confidence", "benign_probability", "malignant_probability"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Inference output missing fields: {sorted(missing)}")
    return result


def verification_agent(
    labels: Sequence[int],
    probabilities: Sequence[float],
    repeated_probabilities: Sequence[Sequence[float]] | None = None,
) -> VerificationReport:
    """Check metrics, calibration, and optional repeated-inference consistency."""
    metrics = _binary_metrics(labels, probabilities)
    metrics["ece"] = _expected_calibration_error(labels, probabilities)
    warnings: list[str] = []
    if metrics["ece"] > 0.15:
        warnings.append("poor_calibration")
    if repeated_probabilities:
        repeated = np.asarray(repeated_probabilities, dtype=float)
        if repeated.ndim != 2 or repeated.shape[1] != len(probabilities):
            raise ValueError("Repeated probabilities must have shape (runs, samples)")
        spread = float(np.mean(np.std(repeated, axis=0)))
        metrics["mean_prediction_spread"] = spread
        if spread > 0.10:
            warnings.append("inconsistent_repeated_predictions")
    return VerificationReport(not warnings, metrics, tuple(warnings))


def safety_agent(
    imaging: ImagingReport,
    verification: VerificationReport | None,
    inference: Mapping[str, object],
) -> SafetyReport:
    """Block unsupported claims and uncertain outputs."""
    flags: list[str] = []
    if not imaging.accepted:
        flags.append("image_quality_not_accepted")
    confidence = inference.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < 0.70:
        flags.append("low_confidence_human_review_required")
    if verification is not None and not verification.accepted:
        flags.extend(verification.warnings)
    if inference.get("diagnosis") is not None:
        flags.append("unsupported_diagnostic_claim")
    return SafetyReport(not flags, tuple(flags))


def explanation_agent(
    inference: Mapping[str, object],
    metadata: Mapping[str, object],
    safety: SafetyReport,
) -> dict[str, object]:
    """Produce a structured summary from verified facts only."""
    return {
        "statement": "This is a research-model summary, not a diagnosis.",
        "model_output_label": inference["label"],
        "confidence": inference["confidence"],
        "uncertainty": 1.0 - float(inference["confidence"]),
        "lesion_location": metadata.get("lesion_location"),
        "salient_regions": inference.get("salient_regions", []),
        "human_review_required": not safety.allowed,
        "safety_flags": list(safety.flags),
        "evidence": {
            "model_output_fields": sorted(inference.keys()),
            "metadata_fields": sorted(metadata.keys()),
            "safety_decision": safety.allowed,
        },
    }


def run_pipeline(
    image_path: Path,
    predictor: Callable[[Path], Mapping[str, object]],
    metadata: Mapping[str, object],
    verification: VerificationReport | None = None,
) -> dict[str, object]:
    """Run the five separated responsibilities in a fixed, auditable order."""
    imaging = imaging_agent(image_path)
    if not imaging.accepted:
        inference = {
            "label": "unavailable",
            "confidence": 0.0,
            "benign_probability": 0.0,
            "malignant_probability": 0.0,
        }
    else:
        inference = inference_agent(predictor, image_path)
    safety = safety_agent(imaging, verification, inference)
    explanation = explanation_agent(inference, metadata, safety)
    return {
        "imaging": imaging.__dict__,
        "inference": inference,
        "verification": verification.__dict__ if verification else None,
        "safety": safety.__dict__,
        "explanation": explanation,
    }
