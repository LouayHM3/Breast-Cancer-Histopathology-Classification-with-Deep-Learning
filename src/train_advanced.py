"""Reproducible advanced training entry point for BreaKHis 400X.

This is a research prototype. It creates patient-grouped train/validation
splits, compares a small CNN with ResNet-18, reports clinical metrics,
generates a Grad-CAM image, evaluates simple corruptions, and exports ONNX.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from advanced_pipeline import (
    ResNetGradCAM,
    apply_robustness_variant,
    binary_metrics,
    check_patient_disjointness,
    expected_calibration_error,
    export_onnx,
    patient_id,
)


@dataclass(frozen=True)
class Record:
    path: Path
    label: int
    patient: str


class ImageRecords(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: list[Record], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
        return self.transform(image), record.label


class CorruptedImageRecords(ImageRecords):
    def __init__(
        self,
        records: list[Record],
        transform: transforms.Compose,
        variant: str,
    ) -> None:
        super().__init__(records, transform)
        self.variant = variant

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.path) as image:
            corrupted = apply_robustness_variant(image.convert("RGB"), self.variant)
        return self.transform(corrupted), record.label


class SmallCNN(nn.Module):
    def __init__(self, classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image).flatten(1))


def collect_records(root: Path) -> list[Record]:
    dataset = datasets.ImageFolder(root)
    return [
        Record(
            path=Path(path),
            label=label,
            patient=patient_id(Path(path)),
        )
        for path, label in dataset.samples
    ]


def make_model(name: str) -> nn.Module:
    if name == "baseline":
        return SmallCNN()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    raise ValueError(f"Unknown model: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += int((logits.argmax(1) == labels).sum())
        count += labels.size(0)
    return total_loss / count, correct / count


def predict(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            probabilities.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.numpy())
    return np.asarray(labels), np.asarray(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", choices=["baseline", "resnet18"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_root = args.dataset_root / "train"
    test_root = args.dataset_root / "test"
    train_records = collect_records(train_root)
    test_records = collect_records(test_root)
    overlap = check_patient_disjointness(
        [record.path for record in train_records],
        [record.path for record in test_records],
    )
    if overlap:
        raise RuntimeError(f"Patient leakage detected between train and test: {sorted(overlap)}")

    groups = np.asarray([record.patient for record in train_records])
    indices = np.arange(len(train_records))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(splitter.split(indices, groups=groups))
    train_subset = [train_records[index] for index in train_indices]
    val_subset = [train_records[index] for index in val_indices]
    assert not {r.patient for r in train_subset} & {r.patient for r in val_subset}

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_loader = DataLoader(ImageRecords(train_subset, train_transform), 32, shuffle=True)
    val_loader = DataLoader(ImageRecords(val_subset, eval_transform), 32)
    test_loader = DataLoader(ImageRecords(test_records, eval_transform), 32)
    model = make_model(args.model).to(device)
    counts = np.bincount([r.label for r in train_subset], minlength=2)
    weights = torch.tensor(len(train_subset) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None, device)
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
        )
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_state_dict(best_state)
    labels, probabilities = predict(model, test_loader, device)
    metrics = binary_metrics(labels, probabilities)
    metrics["ece"] = expected_calibration_error(labels, probabilities)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.model}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

    if args.model == "resnet18":
        cam = ResNetGradCAM(model, model.layer4[-1])
        first_image, _ = next(iter(test_loader))
        heatmap = cam(first_image[:1].to(device)).squeeze().cpu().numpy()
        cam.close()
        plt.imsave(args.output_dir / "gradcam.png", heatmap, cmap="magma")

    model_path = args.output_dir / f"{args.model}.pt"
    torch.save({"model_state_dict": model.state_dict(), "classes": ["benign", "malignant"]}, model_path)
    robustness = {}
    for variant in ["blur", "contrast", "brightness", "jpeg"]:
        corrupted_loader = DataLoader(
            CorruptedImageRecords(test_records, eval_transform, variant),
            32,
        )
        corrupted_labels, corrupted_probabilities = predict(
            model, corrupted_loader, device
        )
        robustness[variant] = binary_metrics(
            corrupted_labels, corrupted_probabilities
        )
    (args.output_dir / f"{args.model}_robustness.json").write_text(
        json.dumps(robustness, indent=2), encoding="utf-8"
    )
    print(json.dumps(robustness, indent=2))
    export_model = copy.deepcopy(model).cpu()
    export_onnx(export_model, args.output_dir / f"{args.model}.onnx")
    del export_model
    start = time.perf_counter()
    cpu_model = copy.deepcopy(model).cpu()
    _ = cpu_model(torch.randn(1, 3, 224, 224))
    del cpu_model
    print(f"CPU inference latency: {(time.perf_counter() - start) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
