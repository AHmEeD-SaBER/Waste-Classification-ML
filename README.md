# 🗑️ Waste Classification using Machine Learning

A machine learning project for classifying waste materials into 6 categories: **Glass, Paper, Cardboard, Plastic, Metal, and Trash** using traditional feature extraction methods and CNN-based features.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Feature Extraction Methods](#feature-extraction-methods)
- [What We Tried](#what-we-tried)
- [Final Approach](#final-approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Files](#model-files)
- [Team](#team)

---

## 🎯 Overview

This project implements a waste classification system using:
- **SVM (Support Vector Machine)** and **KNN (K-Nearest Neighbors)** classifiers
- **Handcrafted features**: Color Histograms (RGB + HSV) and Multi-scale LBP (Local Binary Patterns)
- **CNN features**: MobileNetV2 pre-trained on ImageNet for deep feature extraction

The system supports:
- ✅ Real-time camera classification
- ✅ Single image classification
- ✅ Batch classification from folders
- ✅ Unknown class rejection (low confidence threshold)

---

## 📊 Dataset

- **Total Images**: ~1,865 images (after 30% augmentation)
- **Original Images**: ~1,435 images
- **Image Size**: 224 × 224 pixels
- **Classes**: 6 categories

| Class     | Description                 |
| --------- | --------------------------- |
| Glass     | Glass bottles, jars         |
| Paper     | Paper sheets, documents     |
| Cardboard | Cardboard boxes             |
| Plastic   | Plastic bottles, containers |
| Metal     | Metal cans, aluminum        |
| Trash     | General waste               |

### Data Augmentation
We applied **uniform 30% augmentation** per class using:
- Random rotation (±15°)
- Horizontal flip
- Brightness adjustment
- Slight zoom

---

## 🔬 Feature Extraction Methods

### 1. Color Histogram (192 features)
Extracts color distribution from both **RGB** and **HSV** color spaces:
- **RGB Histogram**: 3 channels × 32 bins = 96 features
- **HSV Histogram**: 3 channels × 32 bins = 96 features

```python
# RGB histogram captures color distribution
# HSV histogram captures hue, saturation, value (better for material distinction)
```

### 2. Multi-Scale LBP (80 features)
**Local Binary Patterns** capture texture information at multiple scales:

| Scale  | num_points | radius | Bins | Captures                         |
| ------ | ---------- | ------ | ---- | -------------------------------- |
| Fine   | 8          | 1      | 10   | Micro-texture (glass smoothness) |
| Small  | 16         | 2      | 18   | Small patterns                   |
| Medium | 24         | 3      | 26   | Medium texture                   |
| Large  | 24         | 5      | 26   | Coarse texture (paper/cardboard) |

**Total handcrafted features**: 192 (color) + 80 (LBP) = **272 features**

### 3. CNN Features (1280 features)
Using **MobileNetV2** pre-trained on ImageNet:
- Removes classification head (`include_top=False`)
- Global average pooling
- Outputs 1280-dimensional feature vector

---

## 🧪 What We Tried

### ❌ Approaches That Decreased Accuracy

| Approach                                | Result         | Why It Failed                                                                                            |
| --------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------- |
| **HOG Features**                        | -3% accuracy   | Captures edges/shape, but waste items have similar shapes. Material distinction matters more than shape. |
| **CLAHE Preprocessing**                 | No improvement | Histogram equalization didn't help for this dataset                                                      |
| **Removing Edge Features**              | -2% accuracy   | Some edge information was still useful                                                                   |
| **Single-scale LBP**                    | -2% accuracy   | Multi-scale captures more texture variety                                                                |
| **Wide GridSearch (C: 1-100)**          | Suboptimal     | 10× jumps missed optimal values                                                                          |
| **Brightness/Reflectivity Features**    | Mixed results  | Added complexity without significant improvement                                                         |
| **Color Moments (mean, std, skewness)** | No improvement | Histogram already captures distribution                                                                  |

### ✅ Approaches That Improved Accuracy

| Approach                       | Result   | Why It Worked                                       |
| ------------------------------ | -------- | --------------------------------------------------- |
| **RGB + HSV Color Histograms** | +2%      | Captures both color and brightness/saturation       |
| **Multi-scale LBP**            | +3%      | Different scales capture different texture patterns |
| **Fine GridSearch (C: 1-20)**  | +1%      | Found optimal C=10, gamma='scale'                   |
| **CNN (MobileNetV2)**          | +6%      | Deep features capture semantic information          |
| **StandardScaler**             | Critical | SVM/KNN require normalized features                 |
| **Varying LBP num_points**     | +1%      | (8,1), (16,2), (24,3), (24,5) captures more variety |

---

## ✨ Final Approach

### Handcrafted Pipeline
```
Image → Resize(224×224) → Color Histogram(RGB+HSV) + Multi-scale LBP → StandardScaler → SVM/KNN
```
- **Total features**: 272
- **Accuracy**: ~82%

### CNN Pipeline
```
Image → Resize(224×224) → MobileNetV2 → 1280 features → StandardScaler → SVM/KNN
```
- **Total features**: 1280
- **Accuracy**: ~88%

### Hyperparameter Tuning (GridSearchCV)
```python
param_grid = {
    'C': [1, 5, 10, 15, 20],
    'gamma': ['scale', 'auto', 0.001, 0.005, 0.01],
    'kernel': ['rbf']
}
# Best: C=10, gamma='scale', kernel='rbf'
# 3-fold cross-validation with 75 total fits
```

---

## 📁 Project Structure

```
Waste-Classification-ML/
├── dataset/                    # Raw image dataset
│   ├── glass/
│   ├── paper/
│   ├── cardboard/
│   ├── plastic/
│   ├── metal/
│   └── trash/
├── data/
│   └── processed/              # Cached preprocessed data (.npz)
├── models/                     # Saved trained models
│   ├── svm_classifier_cnn.pkl
│   ├── svm_classifier_handcrafted.pkl
│   ├── knn_classifier_cnn.pkl
│   ├── knn_classifier_handcrafted.pkl
│   ├── feature_scaler_cnn.pkl
│   ├── feature_scaler_handcrafted.pkl
│   └── class_names.pkl
├── notebooks/
│   └── exploratory_analysis.ipynb  # Main training notebook
├── src/
│   ├── data_preprocessing.py   # Data loading & augmentation
│   ├── feature_extraction.py   # Feature extraction functions
│   └── real_time_classifier.py # Real-time classification app
├── test.py                     # Evaluation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AHmEeD-SaBER/Waste-Classification-ML.git
cd Waste-Classification-ML
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements
```
numpy>=1.24.0
opencv-contrib-python>=4.8.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
joblib>=1.3.0
tensorflow>=2.15.0
albumentations>=1.3.0
pandas>=2.0.0
```

---

## 🚀 Usage

### 1. Training (Jupyter Notebook)
Open and run all cells in:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

This will:
1. Load and augment the dataset
2. Extract features (handcrafted + CNN)
3. Train SVM and KNN classifiers
4. Save models to `models/` folder

### 2. Real-Time Classification
```bash
python src/real_time_classifier.py
```

Interactive menu:
```
============================================================
       WASTE CLASSIFICATION SYSTEM
============================================================

📷 Select Mode:
  1. Real-time Camera
  2. Upload Image

🔍 Select Feature Extraction:
  1. CNN (MobileNetV2) - More accurate, slower
  2. Handcrafted (Color+LBP) - Faster, less accurate

🤖 Select Classifier:
  1. SVM
  2. KNN
```

**Controls in Camera Mode:**
- Press `Q` to quit
- Press `S` to save current frame

### 3. Test on Hidden Dataset
```bash
python test.py
```

Interactive menu shows available models with their accuracies:
```
============================================================
       WASTE CLASSIFICATION - TEST MODE
============================================================

📊 Available Models & Accuracy:
----------------------------------------
  ✓ SVM + Handcrafted:  ~82% accuracy
  ✓ SVM + CNN:          ~88% accuracy
----------------------------------------

🔍 Select Feature Extraction Method:
  1. Handcrafted (Color + LBP) - Faster
  2. CNN (MobileNetV2) - More accurate
```

### 4. Programmatic Usage (for Evaluation)
```python
from test import predict

# Returns list of class names
predictions = predict("path/to/images/folder", "models/svm_classifier_handcrafted.pkl")
print(predictions)  # ['glass', 'paper', 'plastic', ...]
```

The `predict` function:
1. Loads the model and scaler from the given path
2. Loads all images from the folder
3. Extracts features (auto-detects CNN vs handcrafted based on model name)
4. Returns list of predicted class names

---

## 📈 Results

### Model Accuracy Comparison

| Model             | Features           | Accuracy |
| ----------------- | ------------------ | -------- |
| SVM + CNN         | MobileNetV2 (1280) | **~93%** |
| KNN + CNN         | MobileNetV2 (1280) | ~88%     |
| SVM + Handcrafted | Color + LBP (272)  | **~82%** |
| KNN + Handcrafted | Color + LBP (272)  | ~78%     |

### Per-Class Performance (SVM + Handcrafted)

| Class     | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| Glass     | 0.78      | 0.75   | 0.76     |
| Paper     | 0.85      | 0.88   | 0.86     |
| Cardboard | 0.89      | 0.91   | 0.90     |
| Plastic   | 0.76      | 0.74   | 0.75     |
| Metal     | 0.82      | 0.80   | 0.81     |
| Trash     | 0.79      | 0.78   | 0.78     |

### Confusion Analysis
- **Glass vs Plastic**: Most confused pair (similar transparency/reflectivity)
- **Cardboard vs Paper**: Second most confused (similar texture)
- **Metal**: Well distinguished due to unique reflective properties

---

## 💾 Model Files

| File                             | Description                    | Size   |
| -------------------------------- | ------------------------------ | ------ |
| `svm_classifier_cnn.pkl`         | SVM trained on CNN features    | ~17 MB |
| `svm_classifier_handcrafted.pkl` | SVM trained on Color+LBP       | ~3 MB  |
| `knn_classifier_cnn.pkl`         | KNN trained on CNN features    | ~15 MB |
| `knn_classifier_handcrafted.pkl` | KNN trained on Color+LBP       | ~2 MB  |
| `feature_scaler_cnn.pkl`         | StandardScaler for CNN         | ~31 KB |
| `feature_scaler_handcrafted.pkl` | StandardScaler for handcrafted | ~8 KB  |
| `class_names.pkl`                | Class ID to name mapping       | ~1 KB  |

---

## 🔧 Technical Details

### Why SVM Works Well
- **RBF Kernel**: Handles non-linear boundaries between classes
- **C=10**: Good regularization (not too strict, not too loose)
- **gamma='scale'**: Auto-scales based on feature variance
- **probability=True**: Enables confidence scores for unknown rejection

### Unknown Class Rejection
```python
confidence_threshold = 0.5
if max_probability < confidence_threshold:
    prediction = "UNKNOWN"
```
This prevents misclassification of images that don't belong to any known class.

### Feature Normalization
- **StandardScaler**: Zero mean, unit variance
- Critical for SVM (distance-based) and KNN (distance-based)
- Fitted on training data only, applied to test data

### LBP Parameters Explained
- **radius**: Distance from center pixel to sample neighbors (larger = coarser texture)
- **num_points**: Number of pixels sampled around the circle (more = finer angular resolution)
- **uniform patterns**: Reduces features from 2^n to n+2 bins

---

## 📝 Key Learnings

1. **Material > Shape**: For waste classification, color and texture (material properties) matter more than shape. HOG (edge-based) features didn't help.

2. **Multi-scale is key**: Single-scale LBP misses texture patterns at other scales. Using multiple radii (1, 2, 3, 5) significantly improved accuracy.

3. **Don't over-augment**: Class balancing through heavy augmentation can cause overfitting. Uniform augmentation is safer.

4. **Fine-grained hyperparameter search**: Wide ranges like [1, 10, 100] for C miss optimal values. Finer grids like [1, 5, 10, 15, 20] work better.

5. **CNN features are powerful**: Even without fine-tuning, pre-trained CNN features (MobileNetV2) significantly outperform handcrafted features.

---

## 🔮 Future Improvements

1. **Fine-tune CNN**: Train last few layers of MobileNetV2 on waste dataset
2. **Ensemble Methods**: Combine SVM + KNN predictions
3. **Data Augmentation**: More sophisticated augmentation (CutMix, MixUp)
4. **Larger Dataset**: Collect more diverse waste images
5. **Edge Deployment**: Convert to TensorFlow Lite for mobile

---

## 👥 Team

- **Ahmed Saber** - [GitHub](https://github.com/AHmEeD-SaBER)

---

## 🙏 Acknowledgments
- [MobileNetV2](https://arxiv.org/abs/1801.04381) - Efficient CNN architecture
- [scikit-learn](https://scikit-learn.org/) - ML library
- [scikit-image](https://scikit-image.org/) - LBP implementation
- [OpenCV](https://opencv.org/) - Image processing
