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


def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract Histogram of Oriented Gradients features
    
    HOG captures edge and gradient structure, useful for object shape detection.
    
    Args:
        image: Input image (BGR or grayscale)
        orientations: Number of orientation bins (default: 9)
        pixels_per_cell: Size of a cell (default: 8x8)
        cells_per_block: Number of cells in each block (default: 2x2)
        
    Returns:
        feature_vector: 1D numpy array of HOG features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Extract HOG features
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    
    return features


def extract_color_histogram(image, bins=32):
    """
    Extract color histogram features from image
    
    Captures color distribution in both RGB and HSV color spaces.
    This helps distinguish materials by color patterns.
    
    Args:
        image: Input image (BGR)
        bins: Number of bins per channel (default: 32)
        
    Returns:
        feature_vector: 1D numpy array of histogram features
    """
    # Convert BGR to RGB and HSV
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hist_features = []
    
    # RGB histogram (3 channels × bins)
    for i in range(3):
        hist = cv2.calcHist([rgb_image], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        hist_features.extend(hist)
    
    # HSV histogram (3 channels × bins)
    for i in range(3):
        hist = cv2.calcHist([hsv_image], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        hist_features.extend(hist)
    
    return np.array(hist_features)


def extract_lbp_features(image, num_points=24, radius=8):
    """
    Extract Local Binary Pattern features
    
    LBP captures texture patterns, useful for distinguishing surface textures
    like paper vs plastic vs metal.
    
    Args:
        image: Input image (grayscale)
        num_points: Number of circularly symmetric neighbor points (default: 24)
        radius: Radius of circle (default: 8)
        
    Returns:
        feature_vector: 1D numpy array of LBP histogram
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Compute LBP
    lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
    
    # Compute histogram of LBP
    n_bins = num_points + 2  # uniform patterns + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    hist = hist.astype(float)
    hist = hist / (hist.sum() + 1e-7)
    
    return hist


def extract_combined_features(image):
    """
    Extract and combine multiple feature types
    
    Combines HOG (shape/edges), Color Histograms (color distribution),
    and LBP (texture) for comprehensive image representation.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        feature_vector: Combined 1D feature vector
    """
    # Extract individual features
    hog_features = extract_hog_features(image)
    color_features = extract_color_histogram(image, bins=32)
    lbp_features = extract_lbp_features(image, num_points=24, radius=8)
    
    # Concatenate all features
    combined = np.concatenate([hog_features, color_features, lbp_features])
    
    return combined


def extract_features_from_dataset(images, method='combined'):
    """
    Extract features from all images in dataset
    
    Args:
        images: Array of images (n_samples, height, width, channels)
        method: Feature extraction method ('hog', 'color', 'lbp', 'combined')
        
    Returns:
        features: 2D numpy array (n_samples, n_features)
    """
    from tqdm import tqdm
    
    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES: {method.upper()} METHOD")
    print(f"{'='*60}")
    print(f"Total images: {len(images)}")
    
    # Select feature extraction method
    if method == 'hog':
        extractor = extract_hog_features
    elif method == 'color':
        extractor = extract_color_histogram
    elif method == 'lbp':
        extractor = extract_lbp_features
    elif method == 'combined':
        extractor = extract_combined_features
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hog', 'color', 'lbp', or 'combined'")
    
    # Extract features from each image
    features_list = []
    for img in tqdm(images, desc="Extracting features"):
        features = extractor(img)
        features_list.append(features)
    
    # Convert to numpy array
    features = np.array(features_list)
    
    print(f"\nFeature extraction complete!")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Features per image: {features.shape[1]}")
    print(f"{'='*60}\n")
    
    return features
