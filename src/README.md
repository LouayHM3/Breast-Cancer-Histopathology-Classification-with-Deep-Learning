# Separated agent responsibilities

`multi_agent_pipeline.py` provides a deterministic orchestration layer:

1. **Imaging agent** checks readability, RGB conversion, resolution, and basic brightness warnings.
2. **Inference agent** runs only a supplied model callable and validates required output fields.
3. **Verification agent** computes clinical metrics, calibration error, and optional repeated-prediction spread.
4. **Explanation agent** formats verified outputs, metadata, salient regions, and review status.
5. **Safety agent** blocks unsupported diagnostic claims and flags low confidence or failed checks.

These are software roles, not autonomous medical agents. The pipeline does not
call an LLM, read external sources, or diagnose a patient.
