# Separated agent responsibilities

`clinical_safety_pipeline.py` provides a deterministic orchestration layer:

1. **Imaging validation** checks readability, RGB conversion, resolution, and basic brightness warnings.
2. **Inference** runs only a supplied model callable and validates required output fields.
3. **Verification** computes clinical metrics, calibration error, and optional repeated-prediction spread.
4. **Structured explanation** formats verified outputs, metadata, salient regions, and review status.
5. **Safety gating** blocks unsupported diagnostic claims and flags low confidence or failed checks.

These are deterministic software components, not autonomous medical agents. The pipeline does not
call an LLM, read external sources, or diagnose a patient.
