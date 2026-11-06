import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from src.clinical_safety_pipeline import (
    imaging_agent,
    run_pipeline,
    safety_agent,
)


class ClinicalSafetyPipelineTests(unittest.TestCase):
    def test_low_confidence_requires_review(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "case.png"
            Image.new("RGB", (128, 128), (120, 80, 90)).save(image_path)
            result = run_pipeline(
                image_path,
                lambda _: {
                    "label": "malignant",
                    "confidence": 0.55,
                    "benign_probability": 0.45,
                    "malignant_probability": 0.55,
                },
                {},
            )
        self.assertFalse(result["safety"]["allowed"])
        self.assertIn("low_confidence_human_review_required", result["safety"]["flags"])

    def test_diagnostic_claim_is_blocked(self):
        report = imaging_agent(Path(__file__).parent / "missing.png")
        safety = safety_agent(
            report,
            None,
            {
                "label": "malignant",
                "confidence": 0.9,
                "diagnosis": "cancer",
            },
        )
        self.assertFalse(safety.allowed)
        self.assertIn("unsupported_diagnostic_claim", safety.flags)

    def test_unreadable_image_is_rejected(self):
        report = imaging_agent(Path("does-not-exist.png"))
        self.assertFalse(report.accepted)
        self.assertTrue(any(flag.startswith("image_read_error:") for flag in report.warnings))


if __name__ == "__main__":
    unittest.main()
