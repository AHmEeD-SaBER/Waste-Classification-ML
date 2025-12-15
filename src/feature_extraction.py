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
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from skimage.measure import moments_hu
from sklearn.preprocessing import StandardScaler


def extract_hog_features(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    """
    Extract Histogram of Oriented Gradients features
    
    Captures object edges, reflections, and structure.
    Uses CLAHE preprocessing to improve edge detection.
    
    Args:
        image: Input image (BGR or grayscale)
        orientations: Number of orientation bins (default: 9)
        pixels_per_cell: Size of a cell (default: 16x16)
        cells_per_block: Number of cells in each block (default: 2x2)
        
    Returns:
        feature_vector: 1D numpy array of HOG features (~576 features for 224x224 image)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # to improve edge detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
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
    Extract ENHANCED color histogram features from HSV color space
    
    Enhancements:
    - Applies contrast enhancement (similar to CLAHE for HOG)
    - Adds color moments (mean, std, skewness) for better discrimination
    - Uses HSV for better color representation
    
    Args:
        image: Input image (BGR)
        bins: Number of bins per channel (default: 32)
        
    Returns:
        feature_vector: 1D numpy array (96 histogram + 9 moments = 105 features)
    """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Apply histogram equalization to V channel for better contrast
    hsv_image[:,:,2] = cv2.equalizeHist(hsv_image[:,:,2])
    
    hist_features = []
    
    # HSV histogram (3 channels × 32 bins = 96 features)
    for i in range(3):
        hist = cv2.calcHist([hsv_image], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        hist_features.extend(hist)
    
    # Add color moments for each channel (mean, std, skewness)
    # These capture global color properties
    for i in range(3):
        channel = hsv_image[:,:,i].flatten().astype(float)
        mean = channel.mean() / 255.0
        std = channel.std() / 255.0
        skewness = ((channel - channel.mean()) ** 3).mean() / (channel.std() ** 3 + 1e-7)
        hist_features.extend([mean, std, skewness])
    
    return np.array(hist_features)


def extract_lbp_features(image, num_points=24, radius=8):
    """
    Extract ENHANCED Local Binary Pattern features
    
    Enhancements:
    - Applies CLAHE preprocessing (like HOG) for better texture detection
    - Uses multi-scale LBP (multiple radii) for richer texture representation
    - Rotation invariant for robustness
    
    Args:
        image: Input image (grayscale)
        num_points: Number of circularly symmetric neighbor points (default: 24)
        radius: Radius of circle (default: 8)
        
    Returns:
        feature_vector: 1D numpy array of multi-scale LBP histogram
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE preprocessing (same as HOG) for better texture detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    all_hist = []
    
    # Multi-scale LBP: use different radii to capture textures at different scales
    # Helps distinguish fine textures (paper) from coarse textures (metal scratches)
    radii = [1, 3, 8]  # Small, medium, large scale
    points_list = [8, 16, 24]  # Corresponding points
    
    for r, p in zip(radii, points_list):
        # Compute LBP with rotation invariance
        lbp = local_binary_pattern(gray, p, r, method='uniform')
        
        # Compute histogram of LBP
        n_bins = p + 2  # uniform patterns + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize histogram
        hist = hist.astype(float)
        hist = hist / (hist.sum() + 1e-7)
        
        all_hist.extend(hist)
    
    return np.array(all_hist)


def extract_gabor_features(image, frequencies=[0.1, 0.2, 0.3], num_orientations=4):
    """
    Extract Gabor filter features for texture analysis
    
    Gabor filters help distinguish between:
    - Glass: smooth, reflective textures
    - Plastic: semi-smooth with slight texture
    - Metal: reflective with distinct grain patterns
    
    Args:
        image: BGR image (224, 224, 3)
        frequencies: List of frequencies to analyze
        num_orientations: Number of orientations (0 to π)
    
    Returns:
        1D feature vector with mean and std of Gabor responses
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features = []
    for frequency in frequencies:
        for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
            # Apply Gabor filter
            real, imag = gabor(gray, frequency=frequency, theta=theta)
            
            # Extract statistical features from response
            features.append(np.mean(real))
            features.append(np.std(real))
            features.append(np.mean(np.abs(real)))
    
    return np.array(features)


def extract_glcm_features(image, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extract Gray-Level Co-occurrence Matrix (GLCM) texture features
    
    GLCM captures spatial relationships in texture:
    - Glass: homogeneous, low contrast
    - Plastic: moderate texture variation  
    - Metal: higher contrast, more structured patterns
    
    Args:
        image: BGR image (224, 224, 3)
        distances: Pixel pair distance offsets
        angles: Pixel pair angles
    
    Returns:
        1D feature vector with GLCM texture properties
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Quantize to 32 levels for faster computation
    gray = (gray // 8).astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, 
                        levels=32, symmetric=True, normed=True)
    
    # Extract texture properties
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    for prop in properties:
        values = graycoprops(glcm, prop)
        features.append(np.mean(values))
        features.append(np.std(values))
    
    return np.array(features)


def extract_shape_features(image):
    """
    Extract Hu moments for shape-invariant features
    
    Hu moments are invariant to translation, scale, and rotation.
    Helps distinguish container shapes across materials.
    
    Args:
        image: BGR image (224, 224, 3)
    
    Returns:
        7 Hu moments
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate Hu moments
    hu = cv2.HuMoments(cv2.moments(binary)).flatten()
    
    # Log transform for better scale
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    return hu


def extract_edge_features(image):
    """
    Extract edge density and sharpness features
    
    Edge characteristics differ by material:
    - Glass: sharp, well-defined edges
    - Plastic: softer, less defined edges
    - Metal: sharp edges with distinct boundaries
    
    Args:
        image: BGR image (224, 224, 3)
    
    Returns:
        Edge statistics: density, mean gradient, std gradient
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Edge density (percentage of strong edges)
    edge_threshold = np.percentile(gradient_magnitude, 90)
    edge_density = np.mean(gradient_magnitude > edge_threshold)
    
    # Gradient statistics
    features = [
        edge_density,
        np.mean(gradient_magnitude),
        np.std(gradient_magnitude),
        np.max(gradient_magnitude),
        np.percentile(gradient_magnitude, 75),
        np.percentile(gradient_magnitude, 95)
    ]
    
    return np.array(features)


def extract_combined_features(image):
    """
    Extract and combine OPTIMIZED feature types with balanced weighting
    
    Removed expensive Gabor and GLCM (slow, minimal accuracy gain)
    Kept fast, effective features:
    - HOG: CLAHE + edge detection (~6084 features)
    - LBP: CLAHE + multi-scale texture (54 features)
    - Color: Histogram equalization + moments (105 features)
    - Edge: Density and sharpness (6 features) - FAST
    - Shape: Hu moments (7 features) - FAST
    
    Total: ~6256 features (all normalized separately for equal influence)
    
    Args:
        image: Input image (BGR)
        
    Returns:
        feature_vector: Combined and balanced 1D feature vector
    """
    # Extract optimized features (removed slow Gabor and GLCM)
    hog_features = extract_hog_features(image)
    color_features = extract_color_histogram(image, bins=32)
    lbp_features = extract_lbp_features(image)
    shape_features = extract_shape_features(image)
    edge_features = extract_edge_features(image)
    
    # Normalize each feature type separately for equal voting power
    hog_norm = (hog_features - hog_features.mean()) / (hog_features.std() + 1e-7)
    color_norm = (color_features - color_features.mean()) / (color_features.std() + 1e-7)
    lbp_norm = (lbp_features - lbp_features.mean()) / (lbp_features.std() + 1e-7)
    shape_norm = (shape_features - shape_features.mean()) / (shape_features.std() + 1e-7)
    edge_norm = (edge_features - edge_features.mean()) / (edge_features.std() + 1e-7)
    
    # Concatenate: Optimized feature set
    combined = np.concatenate([
        hog_norm, 
        lbp_norm, 
        color_norm,
        shape_norm,
        edge_norm
    ])
    
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
