<div align="center">

# 🔬 Breast Cancer Histopathology Classification

### Deep learning · Explainable AI · ONNX · C++ deployment · Safety-first reporting

<p>
  <img src="https://img.shields.io/badge/status-research%20prototype-6f42c1?style=for-the-badge" alt="Research prototype">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-transfer%20learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/C%2B%2B17-ONNX%20Runtime-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++">
</p>

<p>
  <a href="https://www.kaggle.com/datasets/forderation/breakhis-400x">Dataset</a> ·
  <a href="cpp/README.md">C++ deployment</a> ·
  <a href="assistant/README.md">Constrained assistant</a> ·
  <a href="segmentation/README.md">Segmentation prototype</a>
</p>

<p><strong>Benign vs malignant histopathology classification with an auditable path from notebook to native inference API.</strong></p>

</div>

> ⚠️ **Medical safety notice:** This is an educational/research prototype, not a medical device and not a diagnostic system. It must not be used to make clinical decisions.

---

## ✨ Project at a glance

| Capability | Implemented |
|---|:---:|
| BreaKHis 400X exploration | ✅ |
| ResNet-18 transfer learning | ✅ |
| Patient-grouped validation pipeline | ✅ |
| Sensitivity, specificity, F1, ROC-AUC, PR-AUC | ✅ |
| Calibration and uncertainty signals | ✅ |
| Grad-CAM explanations | ✅ |
| Blur/contrast/brightness/JPEG robustness | ✅ |
| ONNX export | ✅ |
| Native C++ HTTP inference service | ✅ |
| Streamlit upload demo | ✅ |
| Constrained evidence-cited assistant | ✅ |
| Expert pixel masks / U-Net | ⏳ Requires reviewed annotations |

## 🖼️ Visual overview

### System architecture

```mermaid
flowchart LR
    A[Histopathology image] --> B[Imaging agent]
    B --> C[Preprocessing]
    C --> D{Inference}
    D -->|ResNet-18 / CNN| E[Class probabilities]
    D -->|Grad-CAM| F[Salient regions]
    E --> G[Verification agent]
    F --> G
    G --> H[Safety agent]
    H --> I[Structured explanation]
    E --> J[ONNX export]
    J --> K[C++ OpenCV + ONNX Runtime]
    K --> L[HTTP API /v1/classify]
```

### Separated agent responsibilities

```mermaid
flowchart TB
    subgraph INPUT["Verified inputs only"]
      IMG[Image]
      META[Case metadata]
      DOC[Allowlisted documentation]
    end
    IMG --> IA["🖼️ Imaging agent<br/>quality & preprocessing checks"]
    IA --> INF["🧠 Inference agent<br/>classification / segmentation output"]
    INF --> VER["📊 Verification agent<br/>metrics, calibration, consistency"]
    VER --> SAFE["🛡️ Safety agent<br/>unsupported claims & review flags"]
    META --> EXP["📝 Explanation agent<br/>structured report + evidence"]
    DOC --> EXP
    SAFE --> EXP
    EXP --> OUT["Auditable JSON summary<br/>never a diagnosis"]
```

### Reproducible ML workflow

```mermaid
flowchart LR
    A[Download BreaKHis] --> B[Explore classes & dimensions]
    B --> C[Patient-level split]
    C --> D[Train baseline CNN]
    C --> E[Train ResNet-18]
    D --> F[Compare metrics]
    E --> F
    F --> G[Grad-CAM + robustness]
    G --> H[ONNX + C++ benchmark]
    H --> I[Streamlit/API demo]
```

## 📈 Initial reference results

The first notebook run achieved the following on the supplied test split:

| Metric | Result |
|---|---:|
| Accuracy | **80.37%** |
| ROC-AUC | **0.9125** |
| Benign F1 | **0.7434** |
| Malignant F1 | **0.8410** |

<details>
<summary>Confusion matrix interpretation</summary>

- 155 benign images correctly classified
- 21 benign images classified as malignant
- 283 malignant images correctly classified
- 86 malignant images classified as benign

The false-negative count is why accuracy alone is insufficient and why this
prototype must not be interpreted as a clinical diagnostic tool.

</details>

## 🧰 Repository map

```text
oncology/
├── 01_breakhis_exploration.ipynb     # dataset exploration
├── 02_breakhis_training.ipynb        # notebook baseline
├── src/
│   ├── advanced_pipeline.py          # metrics, Grad-CAM, calibration, ONNX
│   ├── train_advanced.py             # grouped training + robustness artifacts
│   └── multi_agent_pipeline.py       # imaging → inference → safety → report
├── assistant/
│   ├── explain_outputs.py             # constrained evidence summarizer
│   └── medical_imaging_documentation.json
├── segmentation/
│   ├── validate_masks.py              # image/mask pair validation
│   └── generate_pseudo_masks.py       # demo-only heuristic masks
├── cpp/
│   ├── inference.cpp                  # OpenCV preprocessing + ONNX inference
│   ├── server.cpp                     # /health and /v1/classify
│   └── benchmark.cpp                  # latency and tensor-memory benchmark
├── app.py                             # Streamlit inference demo
├── requirements.txt
└── .gitignore
```

## 🚀 Quick start

### 1. Install Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Download the dataset

```python
import kagglehub
path = kagglehub.dataset_download("forderation/breakhis-400x")
print(path)
```

Then use:

```python
DATASET_ROOT = Path(path) / "BreaKHis 400X"
```

### 3. Run the advanced pipeline

```powershell
python src\train_advanced.py `
  --dataset-root "C:\path\to\BreaKHis 400X" `
  --model baseline

python src\train_advanced.py `
  --dataset-root "C:\path\to\BreaKHis 400X" `
  --model resnet18
```

Generated artifacts include metrics JSON, robustness JSON, Grad-CAM, PyTorch
weights, and ONNX weights under `artifacts/`.

### 4. Launch the demo

```powershell
streamlit run app.py
```

The demo displays a model output, confidence signal, low-confidence review
flag, and Grad-CAM visualization. It does not diagnose.

## ⚙️ Native C++ deployment

The C++ service mirrors the Python preprocessing contract:

```mermaid
sequenceDiagram
    participant Client
    participant API as C++ HTTP API
    participant CV as OpenCV
    participant ORT as ONNX Runtime
    Client->>API: POST /v1/classify (image)
    API->>CV: decode, RGB, resize, normalize
    CV->>ORT: NCHW float tensor
    ORT-->>API: class logits
    API-->>Client: label, probabilities, timings
```

See [`cpp/README.md`](cpp/README.md) for CMake configuration, API examples,
and the latency benchmark. The service exposes:

```text
GET  /health
POST /v1/classify  multipart field: image
```

## 🤖 Constrained explanation layer

The assistant under [`assistant/`](assistant/) is intentionally deterministic.
It reads only:

1. verified model output;
2. case metadata;
3. predefined allowlisted documentation.

It produces structured fields for model label, confidence, uncertainty,
location when verified metadata exists, salient-region artifacts, limitations,
and evidence sources. It never accesses the original image, the internet, or
an unrestricted language model.

## 🧩 Segmentation status

BreaKHis 400X provides image-level labels but no reviewed pixel-level masks.
Therefore:

- `segmentation/validate_masks.py` validates genuine manually reviewed masks;
- `generate_pseudo_masks.py` creates **demo-only heuristic artifacts**;
- pseudo-masks must not be used as medical ground truth;
- U-Net/UNet++ training requires a separate reviewed mask dataset.

## 🔐 Safety and limitations

- Research prototype; not a medical device.
- Dataset shift and external validity are not established.
- Confidence is not a calibrated clinical probability.
- Grad-CAM is an explanation artifact, not a lesion segmentation.
- Low-confidence or inconsistent cases require qualified human review.
- Patient-level grouping should be preserved for every evaluation.
- Any future segmentation masks require expert annotation and quality review.

## 📚 Citation and data

The project uses the public
[BreaKHis 400X dataset](https://www.kaggle.com/datasets/forderation/breakhis-400x).
Please follow the dataset’s stated terms and cite the original dataset
publication when presenting results.

<div align="center">

### Built as a research prototype for explainable, deployment-aware medical imaging

</div>
