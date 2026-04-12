<div align="center">

<img src="assets/brain.gif" width="380" alt="Brain MRI scan"/>

# Brain Tumor MRI Classification

**Deep learning model to classify brain MRI scans into 4 clinical categories**

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/aakcodebreaker/brain-tumor-classification)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/-Keras-D00000?style=flat&logo=Keras)](https://keras.io)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98%25-brightgreen)](#results)

</div>

---

## Overview

Brain tumors can be life-threatening, and early, accurate classification is critical for treatment planning. This project builds a deep learning classifier that identifies tumor type directly from MRI scans — automating what would otherwise require expert radiologist interpretation.

The model classifies scans into four categories:

| Class | Description |
|-------|-------------|
| **Glioma** | Tumors arising from glial cells; often aggressive |
| **Meningioma** | Originates in the meninges; typically slower growing |
| **Pituitary** | Affects the pituitary gland at the brain's base |
| **No Tumor** | Healthy brain scan |

---

## Dataset

**7,200 brain MRI images** sourced from Kaggle, recombined and re-split for cleaner stratification:

```
Training set   → 5,037 images (stratified)
Validation set → 1,083 images (stratified)
Test set       → 1,080 images (held-out)
```

All images resized to **299×299** to match Xception's expected input.

<img src="assets/images.png" width="600" alt="Sample MRI images from each class"/>

---

## Approach

### Architecture: Xception + Custom Head

The backbone is **Xception** (pretrained on ImageNet), chosen for its depthwise separable convolutions — highly parameter-efficient and strong on texture-rich inputs like MRI scans.

A custom classification head is attached on top:

```
Xception (frozen) → GAP + GMP → Concatenate(4096)
→ BN → Dense(256, ReLU) → BN → Dropout(0.36)
→ Dense(128, ReLU) → BN → Dropout(0.24)
→ Dense(32, ReLU) → Dropout(0.1)
→ Dense(4, Softmax)
```

Total params: **21.9M** | Trainable (Phase 1): **1.09M**

### Training Strategy

**Phase 1 — Feature Extraction** (10 epochs, base frozen)
- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- Loss: Categorical Crossentropy with label smoothing (0.1)
- Callbacks: EarlyStopping + ModelCheckpoint

**Phase 2 — Fine-tuning** (up to 20 epochs, last 40 layers unfrozen)
- Optimizer: AdamW (`lr=5e-5`, `weight_decay=1e-4`)
- Label smoothing reduced to 0.05
- `ReduceLROnPlateau` with factor 0.3

<img src="assets/learning_curves.png" alt="Learninig Curves"/>

### Custom Metric: TumorRecall

A domain-aware metric tracking **average recall across glioma and meningioma** — the two classes where false negatives carry the highest clinical cost. EarlyStopping in Phase 2 monitors this metric rather than generic accuracy.

---

## Results

**Test Accuracy: 98%** across 1,080 held-out images

<img src="assets/conf_matrix.png" width="500" alt="Confusion Matrix"/>

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.98 | 0.96 | 0.97 |
| Meningioma | 0.97 | 0.97 | 0.97 |
| No Tumor | 0.99 | 0.99 | 0.99 |
| Pituitary | 0.97 | 1.00 | 0.98 |
| **Macro Avg** | **0.98** | **0.98** | **0.98** |

---

## Stack

`TensorFlow 2.21` · `Keras 3.14` · `Xception` · `OpenCV` · `scikit-learn` · `Matplotlib` · `Seaborn`

---

## Run It

```bash
# Open on Kaggle (GPU recommended)
# Dataset: Brain Tumor MRI Dataset (Kaggle)
```

The notebook is fully self-contained — data loading, preprocessing, model build, training, and evaluation are all in sequence.

---

<div align="center">

Made with curiosity · [Kaggle Profile](https://www.kaggle.com/aakcodebreaker) · [GitHub](https://github.com/Ayaan-Ali-Khan)

</div>
