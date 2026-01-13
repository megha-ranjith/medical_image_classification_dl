# Confidence-Aware Medical Image Classification using Deep Ensembles

> **Project | Machine Learning / Deep Learning**  
> *CPU-efficient, confidence-aware ensemble framework for medical image classification*

---

## Overview

This project presents a confidence-aware medical image classification framework that combines deep learning–based feature extraction with classical machine learning ensembles. Chest X-ray images are processed using a pretrained CNN to extract discriminative features, which are reduced using PCA and classified through an ensemble of Logistic Regression, Support Vector Machine, and Gradient Boosting models. An entropy-based rejection mechanism is integrated to abstain from low-confidence predictions, improving reliability in clinical decision support. The system is evaluated using stratified 5-fold cross-validation and ROC analysis, and is designed to operate efficiently on CPU-only environments.

---

## Key Contributions

- ✅ Pretrained CNN-based feature extraction (EfficientNet-B0)
- ✅ Dimensionality reduction using PCA
- ✅ Hybrid ensemble of classical ML classifiers:
  - Logistic Regression
  - Support Vector Machine (RBF kernel)
  - Gradient Boosting
- ✅ Confidence-aware **rejection mechanism** using entropy
- ✅ **5-fold stratified cross-validation**
- ✅ Mean ROC curve analysis
- ✅ Fully **CPU-only**, no specialized hardware required
- ✅ Reproducible and modular codebase

---

## Project Structure
```
medical_image_classification_dl/
│
├── data/
│ ├── raw/ # Original datasets (ignored in Git)
│ └── processed/ # PCA-reduced features
│
├── models/
│ ├── cnn_feature_extractor.py
│ ├── classical_models.py
│ ├── ensemble.py
│ └── rejection.py
│
├── preprocessing/
│ ├── image_preprocessing.py
│ └── pca_reduction.py
│
├── evaluation/
│ ├── metrics.py
│ └── plots.py
│
├── results/
│ ├── tables/ # Cross-validation results
│ └── plots/ # ROC curves, entropy histograms
│
├── main.py # Main execution script
├── prepare_data.py # Dataset download & organization
├── config.py # Configuration parameters
├── requirements.txt
└── README.md
```
---

## Dataset

Publicly available **Chest X-ray Tuberculosis datasets**:

- **Montgomery County CXR Dataset**
- **Shenzhen Hospital CXR Dataset**

The datasets are automatically downloaded and organized using `kagglehub`.  
Images are labeled using strict filename-based ground truth (`_0.png` → Normal, `_1.png` → TB).

> ⚠️ Raw datasets are excluded from this repository due to size constraints.

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/megha-ranjith/medical_image_classification_dl.git
cd medical_image_classification_dl
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## ▶️ How to Run
### Step 1: Prepare Dataset
```bash
python prepare_data.py
```
This will:
- Download datasets
- Organize images into NORMAL and TB classes

### Step 2: Run Main Pipeline
```bash
python main.py
```
This performs:
- Feature extraction
- PCA
- 5-fold cross-validation
- Rejection-based evaluation
- ROC curve plotting

##  Outputs
After execution, the following artifacts are generated:

## Results Tables
```bash
results/tables/final_results_5fold.csv
```
Includes:
- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Coverage

## 📊 Plots
```bash

results/plots/
 ├── mean_roc_curve.png
 ├── entropy_histogram_fold_1.png
 ├── ...
 └── entropy_histogram_fold_5.png
```

## Evaluation Methodology

- Stratified 5-fold cross-validation
- Metrics computed only on accepted predictions
- Coverage reported to quantify abstention behavior
- ROC curves averaged across folds
- This ensures robust, leakage-free evaluation.


## Author
Megha Ranjith
MTech Computer Science and Engineering
Mar Athanasius College of Engineering, Kothamangalam

## License
This project is intended for academic and research use only.