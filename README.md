# 🌿 Plant Disease Detection

> **Deep learning system for automated crop disease identification using ResNet50 + Grad-CAM**

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage%2087K-green)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📸 Demo

Upload a photo of a crop leaf → get instant disease diagnosis + visual explanation of *where* the model is looking.

| Input | Prediction | Grad-CAM |
|---|---|---|
| Leaf photo | Disease name + confidence | Heatmap overlay showing infected region |

---

## 🎯 What It Does

- **38-class classification** across 14 crop species (Apple, Tomato, Corn, Grape, Potato, etc.)
- **Grad-CAM heatmaps** — visualises which part of the leaf triggered the prediction
- **Top-5 predictions** with confidence scores
- **Disease info cards** — treatment advice and description for each disease
- **Streamlit web UI** — dark teal theme, drag-and-drop image upload

---

## 🗂️ Dataset

**New Plant Diseases Dataset (PlantVillage)**
- Source: [Kaggle — vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- 87,856 images | 38 classes | 14 crop species
- Split: 80% train / 10% valid / 10% test

| Split | Images |
|---|---|
| Train | ~70,428 |
| Valid | ~8,714 |
| Test | ~8,714 |

---

## 🧠 Model Architecture

```
ResNet50 (ImageNet pretrained)
├── Layer 1, 2, 3  →  Frozen (ImageNet features reused)
├── Layer 4        →  Fine-tuned (lr = 1e-4)
└── FC Head        →  Linear(2048 → 38)  (lr = 1e-3)
```

**Training config:**
- Optimizer: Adam (differential learning rates)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.1)
- Batch size: 64 | Max epochs: 20 | Early stopping: 5
- Loss: CrossEntropyLoss

**Augmentation (train only):**
RandomResizedCrop · RandomHorizontalFlip · RandomVerticalFlip · ColorJitter · RandomRotation

---

## 📊 Results

> *(Filled after training completes)*

| Metric | Value |
|---|---|
| Test Accuracy | — |
| Val Accuracy (best) | — |
| Training Epochs | — |

---

## 🚀 Running Locally

### 1. Clone
```bash
git clone https://github.com/Shikhar-2005/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. Create environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Get the trained model
Download `plant_disease_resnet50.pth` and `class_names.json` from the [Releases](https://github.com/Shikhar-2005/Plant-Disease-Detection/releases) page and place them in `models/`.

### 4. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
Plant-Disease-Detection/
├── app.py                          # Streamlit web app
├── requirements.txt                # Python dependencies
├── setup_dataset.py                # Dataset download + split script
├── disease_info.json               # Disease descriptions + treatments
│
├── src/
│   ├── model.py                    # ResNet50 build + inference
│   ├── gradcam.py                  # Grad-CAM implementation
│   ├── transforms.py               # Train / inference transforms
│   └── __init__.py
│
├── notebooks/
│   ├── eda_and_training.ipynb      # Full EDA + training (local)
│   └── colab_training.ipynb        # Self-contained Google Colab version
│
├── figures/                        # EDA + evaluation plots
│   ├── class_distribution.png
│   ├── sample_grid.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── gradcam_examples.png
│
├── models/
│   ├── class_names.json            # 38 class labels (tracked)
│   └── plant_disease_resnet50.pth  # Trained weights (not in git, ~100 MB)
│
└── data/                           # Dataset (not in git, ~3.5 GB)
    └── plantvillage/
        ├── train/   (38 classes)
        ├── valid/   (38 classes)
        └── test/    (38 classes)
```

---

## ⚠️ Known Limitations

- **Lab vs field gap**: PlantVillage images are taken in controlled conditions. Real field images show ~62% accuracy vs ~99% on test set. Model works best on clean, well-lit leaf photos.
- **38 fixed classes**: Only covers supported crops. Unknown diseases will be misclassified as the closest match.
- **Model size**: `.pth` file is ~100 MB — hosted separately, not in the repo.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Model | PyTorch · ResNet50 · torchvision |
| Explainability | Grad-CAM (custom, no external lib) |
| UI | Streamlit |
| Data | PlantVillage (Kaggle) |
| Training | Google Colab T4 GPU |
| Language | Python 3.13 |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
