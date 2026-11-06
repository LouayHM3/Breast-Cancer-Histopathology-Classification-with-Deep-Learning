# Constrained model-output assistant

This component turns verified inference artifacts into a structured summary.
It is deliberately deterministic: it does not call an LLM, access the
internet, inspect the source image, or infer a diagnosis.

The assistant accepts only:

- model output JSON;
- case metadata JSON;
- an allowlisted medical-imaging documentation JSON file.

It reports the model's label and probabilities, confidence, uncertainty,
salient-region artifact references, limitations, and evidence citations.

## Example

```powershell
python assistant\explain_outputs.py `
  --model-output examples\model_output.json `
  --metadata examples\case_metadata.json `
  --documentation assistant\medical_imaging_documentation.json `
  --output artifacts\structured_explanation.json
```

The output is a machine-readable JSON object. The `label` field is explicitly
named `model_output_label`; it is never described as a diagnosis.
