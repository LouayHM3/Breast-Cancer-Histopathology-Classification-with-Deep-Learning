"""Generate a constrained, evidence-cited summary of model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def require_number(value: Any, field: str, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return number


def make_explanation(
    model_output: dict[str, Any],
    metadata: dict[str, Any],
    documentation: dict[str, Any],
) -> dict[str, Any]:
    required = ("label", "confidence", "benign_probability", "malignant_probability")
    missing = [field for field in required if field not in model_output]
    if missing:
        raise ValueError(f"Model output is missing required fields: {missing}")
    label = model_output["label"]
    if label not in {"benign", "malignant"}:
        raise ValueError("Model output label must be benign or malignant")
    benign = require_number(model_output["benign_probability"], "benign_probability")
    malignant = require_number(model_output["malignant_probability"], "malignant_probability")
    confidence = require_number(model_output["confidence"], "confidence")
    if abs((benign + malignant) - 1.0) > 0.02:
        raise ValueError("Class probabilities must sum to approximately one")
    expected_confidence = max(benign, malignant)
    if abs(confidence - expected_confidence) > 0.02:
        raise ValueError("Confidence does not match the highest class probability")
    allowed_claims = documentation.get("allowed_claims")
    limitations = documentation.get("limitations")
    if not isinstance(allowed_claims, dict) or not isinstance(limitations, list):
        raise ValueError("Documentation must define allowed_claims and limitations")

    location = metadata.get("lesion_location")
    if location is not None and not isinstance(location, dict):
        raise ValueError("lesion_location metadata must be an object")
    salient_regions = model_output.get("salient_regions")
    if salient_regions is not None and not isinstance(salient_regions, list):
        raise ValueError("salient_regions must be a list of artifact references")

    return {
        "schema_version": "1.0",
        "case_id": metadata.get("case_id", "unidentified-case"),
        "model_output_label": label,
        "statement": "This is a summary of verified model output, not a diagnosis.",
        "confidence": confidence,
        "uncertainty": 1.0 - confidence,
        "lesion_location": {
            "value": location,
            "available": location is not None,
            "interpretation": allowed_claims["location"],
        },
        "salient_regions": {
            "artifacts": salient_regions or [],
            "available": bool(salient_regions),
            "interpretation": allowed_claims["salient_regions"],
        },
        "limitations": limitations,
        "evidence": [
            {"source": "model_output", "fields": list(model_output.keys())},
            {"source": "case_metadata", "fields": list(metadata.keys())},
            {
                "source": "documentation",
                "document_id": documentation.get("document_id"),
                "version": documentation.get("version"),
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--documentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    explanation = make_explanation(
        load_object(args.model_output),
        load_object(args.metadata),
        load_object(args.documentation),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(explanation, indent=2), encoding="utf-8")
    print(f"Wrote structured explanation to {args.output}")


if __name__ == "__main__":
    main()
