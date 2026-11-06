# Breast Cancer Histopathology Classification with Deep Learning

This project explores the BreaKHis 400X histopathology image dataset and trains a binary classifier for benign versus malignant breast-tumor images.

> **Important:** This is an educational/research project, not a medical diagnostic system. The model must not be used to make clinical decisions.

## Project contents

- [`01_breakhis_exploration.ipynb`](./01_breakhis_exploration.ipynb): dataset structure, class distribution, image dimensions, and sample images.
- [`02_breakhis_training.ipynb`](./02_breakhis_training.ipynb): preprocessing, augmentation, ResNet-18 transfer learning, validation, test metrics, ROC-AUC, confusion matrix, and model export.
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
