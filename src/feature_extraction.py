"""
Feature Extraction Module

This module converts raw images into fixed-size feature vectors.
This is the critical step that transforms pixel data into numerical features
that can be used by ML classifiers.

Key Tasks:
- Implement multiple feature extraction methods
- Convert 2D/3D images to 1D feature vectors
- Ensure fixed-length output for all images
- Compare different feature extraction approaches

Suggested Feature Extraction Methods:
1. Histogram of Oriented Gradients (HOG)
2. Color Histograms (RGB/HSV)
3. Local Binary Patterns (LBP)
4. Statistical features (mean, std, etc.)
5. SIFT/ORB bag-of-words features
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler


# TODO: Implement HOG feature extraction
def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract Histogram of Oriented Gradients features
    
    Args:
        image: Input image (BGR or grayscale)
        orientations: Number of orientation bins
        pixels_per_cell: Size of a cell
        cells_per_block: Number of cells in each block
        
    Returns:
        feature_vector: 1D numpy array of HOG features
    """
    pass


# TODO: Implement color histogram features
def extract_color_histogram(image, bins=32):
    """
    Extract color histogram features from image
    
    Args:
        image: Input image (BGR)
        bins: Number of bins per channel
        
    Returns:
        feature_vector: 1D numpy array of histogram features
    """
    pass


# TODO: Implement LBP features
def extract_lbp_features(image, num_points=24, radius=8):
    """
    Extract Local Binary Pattern features
    
    Args:
        image: Input image (grayscale)
        num_points: Number of circularly symmetric neighbor points
        radius: Radius of circle
        
    Returns:
        feature_vector: 1D numpy array of LBP histogram
    """
    pass


# TODO: Combine multiple features
def extract_combined_features(image):
    """
    Extract and combine multiple feature types
    
    Args:
        image: Input image
        
    Returns:
        feature_vector: Combined 1D feature vector
    """
    pass


# TODO: Extract features from dataset
def extract_features_from_dataset(images, method='combined'):
    """
    Extract features from all images in dataset
    
    Args:
        images: List of images
        method: Feature extraction method ('hog', 'color', 'lbp', 'combined')
        
    Returns:
        features: 2D numpy array (n_samples, n_features)
    """
    pass


# TODO: Save/Load features
def save_features(features, labels, filepath):
    """Save extracted features and labels"""
    pass


def load_features(filepath):
    """Load extracted features and labels"""
    pass


if __name__ == "__main__":
    # Test feature extraction
    print("Testing feature extraction methods...")
    
    # TODO: Load sample images and test each method
    # TODO: Compare feature vector sizes
    # TODO: Save features for training
    
    print("Feature extraction module ready!")
