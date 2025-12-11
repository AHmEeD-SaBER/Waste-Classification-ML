"""
k-NN Classifier Training Module

This module trains a k-Nearest Neighbors classifier on extracted features.

Key Tasks:
- Load extracted features
- Configure k-NN hyperparameters (k, distance metric, weighting)
- Implement "Unknown" class rejection mechanism
- Train and save the model
- Evaluate on validation set

k-NN Configuration Considerations:
- k value: Number of neighbors
- Distance metric: Euclidean, Manhattan, Minkowski
- Weights: Uniform vs distance-based
- Algorithm: Ball tree, KD tree, brute force
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json


# TODO: Load features
def load_training_data(feature_path):
    """
    Load pre-extracted features and labels
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    pass


# TODO: Configure k-NN
def create_knn_classifier(n_neighbors=5, weights='uniform', metric='euclidean'):
    """
    Create and configure k-NN classifier
    
    Args:
        n_neighbors: Number of neighbors to use
        weights: Weight function ('uniform', 'distance')
        metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
        
    Returns:
        classifier: Configured k-NN classifier
    """
    pass


# TODO: Hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    """
    Perform grid search for optimal hyperparameters
    
    Test different values of k, weights, and metrics
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        best_params: Dictionary of best parameters
        best_classifier: Trained classifier with best parameters
    """
    pass


# TODO: Implement rejection mechanism for "Unknown" class
def predict_with_rejection(classifier, X, confidence_threshold=0.6):
    """
    Predict with rejection mechanism for uncertain samples
    
    If nearest neighbors are too far or not agreeing, classify as "Unknown" (class 6)
    
    Args:
        classifier: Trained k-NN classifier
        X: Feature vectors to classify
        confidence_threshold: Minimum confidence for acceptance
        
    Returns:
        predictions: Class predictions (including "Unknown")
    """
    pass


# TODO: Train and evaluate
def train_knn(X_train, y_train, X_val, y_val):
    """
    Train k-NN classifier and evaluate performance
    
    Returns:
        classifier: Trained k-NN model
        metrics: Dictionary of performance metrics
    """
    pass


# TODO: Save model
def save_model(classifier, filepath="../models/knn_classifier.pkl"):
    """Save trained k-NN model"""
    pass


if __name__ == "__main__":
    print("Training k-NN Classifier...")
    
    # TODO: Load features
    # X_train, X_val, y_train, y_val = load_training_data("../data/features.npz")
    
    # TODO: Tune hyperparameters
    # best_params, classifier = tune_hyperparameters(X_train, y_train)
    
    # TODO: Train with best parameters
    # classifier, metrics = train_knn(X_train, y_train, X_val, y_val)
    
    # TODO: Save model
    # save_model(classifier)
    
    print("k-NN training complete!")
