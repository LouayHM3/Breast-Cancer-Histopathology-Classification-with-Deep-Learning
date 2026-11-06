# Breast Cancer Histopathology Classification with Deep Learning

This project explores the BreaKHis 400X histopathology image dataset and trains a binary classifier for benign versus malignant breast-tumor images.

> **Important:** This is an educational/research project, not a medical diagnostic system. The model must not be used to make clinical decisions.

## Project contents

- [`01_breakhis_exploration.ipynb`](./01_breakhis_exploration.ipynb): dataset structure, class distribution, image dimensions, and sample images.
- [`02_breakhis_training.ipynb`](./02_breakhis_training.ipynb): preprocessing, augmentation, ResNet-18 transfer learning, validation, test metrics, ROC-AUC, confusion matrix, and model export.
- [`src/advanced_pipeline.py`](./src/advanced_pipeline.py): patient-group checks, sensitivity/specificity, PR-AUC, calibration, Grad-CAM, robustness variants, and ONNX export helpers.
- [`src/train_advanced.py`](./src/train_advanced.py): patient-grouped training, baseline-CNN/ResNet comparison, metrics, Grad-CAM, robustness artifacts, and ONNX export.
- [`app.py`](./app.py): small Streamlit inference demonstration with a confidence flag and Grad-CAM.
- [`cpp/README.md`](./cpp/README.md): native ONNX Runtime service and latency benchmark.
- [`segmentation/README.md`](./segmentation/README.md): manual mask-annotation prototype and validation rules.
- [`assistant/README.md`](./assistant/README.md): constrained, evidence-cited model-output summaries.
- [`src/multi_agent_pipeline.py`](./src/multi_agent_pipeline.py): separated imaging, inference, verification, explanation, and safety agents.
- [`requirements.txt`](./requirements.txt): Python dependencies.

## Dataset

The notebooks use the public [BreaKHis 400X dataset](https://www.kaggle.com/datasets/forderation/breakhis-400x).

The dataset is not included in this repository. Download it with KaggleHub:

```python
!pip install -q kagglehub

import kagglehub

path = kagglehub.dataset_download("forderation/breakhis-400x")
print(path)
```

In `02_breakhis_training.ipynb`, set `DATASET_ROOT` to:

```python
DATASET_ROOT = Path(path) / "BreaKHis 400X"
```

For a local Windows download, use the path printed by KaggleHub, for example:

```python
DATASET_ROOT = (
    Path(r"C:\Users\<your-user>\.cache\kagglehub\datasets\forderation\breakhis-400x\versions\1")
    / "BreaKHis 400X"
)
```

## Running the project

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open the notebooks in VS Code or Google Colab, select the Python kernel, download the dataset, update `DATASET_ROOT`, and run the cells in order. A GPU is recommended for training.

The training notebook saves the trained weights as `breakhis_resnet18.pt`. Model weights are ignored by Git because they are generated artifacts.

For the advanced pipeline:

```powershell
python src\train_advanced.py --dataset-root "C:\path\to\BreaKHis 400X" --model baseline
python src\train_advanced.py --dataset-root "C:\path\to\BreaKHis 400X" --model resnet18
streamlit run app.py
```

## Reported evaluation

The initial run achieved approximately:

- Test accuracy: 80.37%
- ROC-AUC: 0.9125
- Benign F1-score: 0.7434
- Malignant F1-score: 0.8410

The confusion matrix included 86 malignant images classified as benign. This is a key limitation and shows why accuracy alone is not sufficient for a medical-imaging model.

## Reproducibility and limitations

- The notebook fixes random seeds where supported.
- Augmentation is applied only to the training subset.
- Class-weighted loss addresses the benign/malignant imbalance.
- The supplied train/test organization is used; patient-level leakage should be investigated before making stronger claims.
- Results can vary with hardware, library versions, random state, and pretrained-weight availability.
- The advanced pipeline keeps patients disjoint between training and validation and refuses a train/test patient overlap.
- Confidence is a review signal only; it is not a validated clinical probability.

## Upgrade roadmap

The advanced utilities are prepared for the next experiment:

- Verify patient-level disjointness before training and report any overlap.
- Compare the current ResNet-18 model with a small CNN baseline and an EfficientNet/ViT configuration.
- Report sensitivity, specificity, F1, ROC-AUC, PR-AUC, calibration error, and the confusion matrix.
- Generate Grad-CAM heatmaps and evaluate blur, contrast, brightness, and JPEG-compression robustness.
- Export the selected model to ONNX and measure inference latency.
- Add an upload-based inference application with a confidence threshold that flags uncertain cases for human review.

BreaKHis 400X as downloaded here contains class labels but no lesion masks, so supervised U-Net/UNet++ segmentation cannot be implemented without a separate pixel-annotated dataset. Likewise, the current result is not externally validated and remains a research prototype, not a diagnostic medical device.

## Manual segmentation prototype

The repository includes a safe annotation workflow under [`segmentation/`](./segmentation/).
It validates real manually drawn masks but does not generate synthetic medical
ground truth. U-Net training can begin after a reviewed image/mask sample has
been added.

For a visualization-only experiment, `segmentation/generate_pseudo_masks.py`
can create heuristic pseudo-masks. These are not annotations and must not be
used to report segmentation quality or train a clinical model.
