# Automated Material Stream Identification System

An ML-based waste classification system using traditional computer vision and machine learning techniques.

## Project Overview

This project implements an automated waste sorting system that classifies materials into 7 categories:
- Glass (ID: 0)
- Paper (ID: 1)
- Cardboard (ID: 2)
- Plastic (ID: 3)
- Metal (ID: 4)
- Trash (ID: 5)
- Unknown (ID: 6)

## Project Structure

```
Project/
├── dataset/                    # Training and validation data
├── src/                        # Source code
│   ├── data_preprocessing.py   # Data augmentation and preprocessing
│   ├── feature_extraction.py   # Image to feature vector conversion
│   ├── train_svm.py           # SVM classifier training
│   ├── train_knn.py           # k-NN classifier training
│   ├── evaluate.py            # Model evaluation and comparison
│   ├── realtime_classification.py  # Live camera classification
│   └── utils.py               # Helper functions
├── models/                     # Saved trained models
├── notebooks/                  # Jupyter notebooks for exploration
├── results/                    # Evaluation results and plots
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing and Augmentation
```bash
python src/data_preprocessing.py
```

### 2. Train Classifiers
```bash
python src/train_svm.py
python src/train_knn.py
```

### 3. Evaluate Models
```bash
python src/evaluate.py
```

### 4. Run Real-time Classification
```bash
python src/realtime_classification.py
```

## Technical Approach

### Feature Extraction Methods
- Histogram of Oriented Gradients (HOG)
- Color Histograms
- Local Binary Patterns (LBP)
- SIFT/ORB descriptors

### Classifiers
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)

### Data Augmentation
- Rotation
- Flipping
- Scaling
- Color jittering
- Brightness adjustment

## Performance Target

- Minimum validation accuracy: 0.85 across six primary classes
- Robust handling of "Unknown" class

## Deliverables

- [x] Source code repository
- [ ] Trained model weights
- [ ] Technical report (PDF)
- [ ] Real-time classification demo

## Authors

ML Project - December 2025
