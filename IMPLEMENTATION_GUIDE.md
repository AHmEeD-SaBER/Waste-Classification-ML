# Implementation Guide

## Project Structure Overview

```
Project/
├── dataset/                    # Your training data (already exists)
├── src/                        # Source code modules
├── models/                     # Saved trained models
├── notebooks/                  # Jupyter notebooks for exploration
├── results/                    # Evaluation results
├── config.json                 # Configuration file
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Implementation Steps

### Phase 1: Data Preparation (Week 1)
**File: `src/data_preprocessing.py`**

1. **Load Dataset**
   - Read images from dataset folders (classes 0-5)
   - Store images with their corresponding labels
   - Check image sizes and formats

2. **Data Augmentation** (Minimum 30% increase)
   - Rotation: ±15 degrees
   - Horizontal/Vertical flipping
   - Scaling: 90%-110%
   - Brightness adjustment: ±20%
   - Gaussian noise
   - Color jittering
   - **Important**: Apply randomly, not all at once

3. **Train/Validation Split**
   - Use 80/20 split
   - Maintain class balance
   - Set random_state for reproducibility

### Phase 2: Feature Extraction (Week 1-2)
**File: `src/feature_extraction.py`**

1. **Implement HOG Features**
   - Convert image to grayscale
   - Use scikit-image's `hog()` function
   - Parameters: 9 orientations, 8x8 pixels_per_cell
   - Output: ~3,000-5,000 features

2. **Implement Color Histograms**
   - Extract RGB or HSV histograms
   - Use 32 bins per channel
   - Concatenate all channels
   - Output: 96 features (32 bins × 3 channels)

3. **Implement LBP Features**
   - Convert to grayscale
   - Apply Local Binary Patterns
   - Create histogram of patterns
   - Output: 256 features

4. **Combined Features**
   - Concatenate HOG + Color + LBP
   - Total: ~3,500-5,500 features
   - Apply StandardScaler normalization

### Phase 3: Model Training (Week 2)
**Files: `src/train_svm.py` and `src/train_knn.py`**

#### SVM Implementation:
1. **Hyperparameter Tuning**
   - Test kernels: 'linear', 'rbf', 'poly'
   - Test C values: [0.1, 1, 10, 100]
   - Test gamma: ['scale', 'auto', 0.001, 0.01]
   - Use GridSearchCV with 5-fold CV

2. **Unknown Class Rejection**
   - Use `decision_function()` to get confidence scores
   - If max confidence < threshold (e.g., 0.7), classify as "Unknown"
   - Threshold should be tuned on validation set

#### k-NN Implementation:
1. **Hyperparameter Tuning**
   - Test k values: [3, 5, 7, 9, 11]
   - Test weights: ['uniform', 'distance']
   - Test metrics: ['euclidean', 'manhattan']
   - Use GridSearchCV

2. **Unknown Class Rejection**
   - Use `predict_proba()` to get class probabilities
   - If max probability < threshold (e.g., 0.6), classify as "Unknown"
   - Also consider distance to nearest neighbors

### Phase 4: Evaluation (Week 3)
**File: `src/evaluate.py`**

1. **Performance Metrics**
   - Accuracy score (target: 0.85+)
   - Precision, Recall, F1-score per class
   - Confusion matrix
   - ROC curves (optional)

2. **Comparison Analysis**
   - Compare SVM vs k-NN
   - Compare different feature methods
   - Analyze inference time
   - Memory usage

3. **Visualization**
   - Generate confusion matrices
   - Plot accuracy comparison charts
   - Create per-class performance bars

### Phase 5: Real-time System (Week 3-4)
**File: `src/realtime_classification.py`**

1. **Camera Integration**
   - Use OpenCV's `VideoCapture(0)`
   - Capture frames at ~30 FPS
   - Resize frames to match training size

2. **Real-time Processing**
   - Extract features from each frame
   - Classify using best model
   - Display result with confidence
   - Add colored bounding boxes

3. **User Interface**
   - Show class name and confidence
   - Display FPS counter
   - Add controls (pause, save, quit)

## Configuration Management

Use `config.json` to store all hyperparameters:
- Dataset paths
- Augmentation settings
- Feature extraction parameters
- Model hyperparameters
- Confidence thresholds

## Tips for Success

### To Achieve 0.85+ Accuracy:
1. **Good Features**: Combine multiple feature types
2. **Proper Augmentation**: Don't overdo it, maintain quality
3. **Feature Normalization**: Always use StandardScaler
4. **Hyperparameter Tuning**: Use GridSearchCV thoroughly
5. **Class Balance**: Ensure equal representation
6. **Unknown Class**: Set appropriate confidence thresholds

### Common Pitfalls to Avoid:
- ❌ Not normalizing features
- ❌ Over-augmentation (distorted images)
- ❌ Using raw pixels instead of features
- ❌ Not handling image size variations
- ❌ Forgetting the "Unknown" class
- ❌ Not saving models properly

### Recommended Order:
1. Start with exploratory_analysis.ipynb
2. Implement and test feature extraction
3. Train both classifiers with default params
4. Tune hyperparameters
5. Implement rejection mechanism
6. Evaluate and compare
7. Integrate into real-time system

## Testing Your Implementation

After each phase, test:
```bash
# Phase 1
python src/data_preprocessing.py

# Phase 2
python src/feature_extraction.py

# Phase 3
python src/train_svm.py
python src/train_knn.py

# Phase 4
python src/evaluate.py

# Phase 5
python src/realtime_classification.py
```

## Technical Report Sections

Your report should include:
1. **Introduction**: Problem statement and objectives
2. **Data Augmentation**: Techniques used and justification
3. **Feature Extraction**: Methods compared and selection reasoning
4. **Classifiers**: SVM and k-NN configurations
5. **Unknown Class**: Rejection mechanism implementation
6. **Results**: Performance metrics and comparisons
7. **Real-time System**: Architecture and performance
8. **Conclusion**: Best approach and future improvements

## Timeline Suggestion

- **Week 1**: Data prep, augmentation, feature extraction
- **Week 2**: Model training and hyperparameter tuning
- **Week 3**: Evaluation and real-time system
- **Week 4**: Report writing and final testing

Good luck with your implementation! 🚀
