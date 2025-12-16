import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from skimage.measure import moments_hu
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet


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
    - GLASS/METAL/PLASTIC SPECIFIC: Brightness and reflectivity features
      * Glass: transparent/bright, high V channel, low saturation
      * Metal: metallic/shiny, high variance in V, specular highlights
      * Plastic: colored/matte, moderate saturation, uniform brightness
    
    Args:
        image: Input image (BGR)
        bins: Number of bins per channel (default: 32)
        
    Returns:
        feature_vector: 1D numpy array (~118 features)
          - 96 histogram bins (HSV)
          - 9 color moments (HSV mean/std/skew)
          - 13 brightness/reflectivity features
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
    
    # === BRIGHTNESS & REFLECTIVITY FEATURES (13 features) ===
    # Critical for glass/metal/plastic distinction
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    h_channel = hsv_image[:,:,0].astype(float)
    s_channel = hsv_image[:,:,1].astype(float)
    v_channel = hsv_image[:,:,2].astype(float)
    
    # 1-2. V channel statistics (brightness)
    # Glass: high mean V (bright/transparent)
    # Metal: high variance in V (shiny reflections)
    # Plastic: moderate, uniform V
    v_mean = v_channel.mean() / 255.0
    v_variance = v_channel.var() / (255.0 ** 2)
    
    # 3-4. Saturation statistics
    # Glass: low saturation (clear/transparent)
    # Plastic: moderate-high saturation (colored)
    # Metal: low saturation (gray/metallic)
    s_mean = s_channel.mean() / 255.0
    s_variance = s_channel.var() / (255.0 ** 2)
    
    # 5. Saturation-to-brightness ratio
    # Glass: low S, high V → low ratio
    # Plastic: high S, moderate V → high ratio
    # Metal: low S, variable V → low ratio
    s_v_ratio = (s_mean + 1e-7) / (v_mean + 1e-7)
    
    # 6-7. Specular highlight detection (very bright pixels)
    # Metal/glass have more specular highlights than plastic
    bright_threshold = np.percentile(v_channel, 90)
    specular_pixels = (v_channel > bright_threshold).sum() / v_channel.size
    specular_intensity = v_channel[v_channel > bright_threshold].mean() / 255.0 if specular_pixels > 0 else 0
    
    # 8-9. Dark region statistics (shadows/depth)
    dark_threshold = np.percentile(v_channel, 10)
    dark_pixels = (v_channel < dark_threshold).sum() / v_channel.size
    dark_intensity = v_channel[v_channel < dark_threshold].mean() / 255.0 if dark_pixels > 0 else 0
    
    # 10. Brightness uniformity (coefficient of variation)
    # Plastic: more uniform
    # Metal/glass: more variable (reflections)
    brightness_uniformity = (v_channel.std() + 1e-7) / (v_channel.mean() + 1e-7)
    
    # 11. High-frequency brightness changes (reflectivity indicator)
    # Metal/glass: more edges in brightness
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    brightness_gradient = np.mean(np.sqrt(sobelx**2 + sobely**2)) / 255.0
    
    # 12-13. Color purity (distance from grayscale)
    # Glass/metal: closer to grayscale (low saturation)
    # Plastic: more colorful (high saturation)
    color_purity_mean = s_channel.mean() / 255.0
    color_purity_max = s_channel.max() / 255.0
    
    # Combine all brightness/reflectivity features
    brightness_features = [
        v_mean, v_variance,
        s_mean, s_variance,
        s_v_ratio,
        specular_pixels, specular_intensity,
        dark_pixels, dark_intensity,
        brightness_uniformity,
        brightness_gradient,
        color_purity_mean, color_purity_max
    ]
    
    hist_features.extend(brightness_features)
    
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
    
    # ENHANCED Multi-scale LBP: 4 scales with more points for finer texture discrimination
    # Glass vs Plastic vs Metal requires capturing textures at multiple granularities:
    # - Fine scale (r=1): Surface smoothness (glass=very smooth, plastic=semi-smooth, metal=scratches)
    # - Small scale (r=2): Micro-texture patterns
    # - Medium scale (r=5): Grain patterns (metal grain vs plastic molding)
    # - Large scale (r=8): Macro structure
    radii = [1, 2, 5, 8]  # 4 scales instead of 3
    points_list = [8, 16, 24, 24]  # More points for finer detail
    
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
    Extract and combine ENHANCED COLOR + TEXTURE features
    
    REMOVED HOG - it was dominating (6084/6256 = 97%) and not discriminative
    for materials with similar shapes (bottles, containers, sheets).
    
    ENHANCED FOR GLASS/METAL/PLASTIC DISCRIMINATION:
    - Color: HSV + brightness/reflectivity (~118 features)
      * Glass: bright, transparent, low saturation, specular highlights
      * Metal: shiny, variable brightness, metallic, specular highlights
      * Plastic: colored, uniform brightness, moderate saturation
    - LBP: CLAHE + 4-scale multi-scale texture (~68 features)
      * Glass: very smooth surface
      * Plastic: semi-smooth with molding patterns
      * Metal: grain patterns and scratches
    - Edge: Density and sharpness (6 features) - edge quality differences
    - Shape: Hu moments (7 features) - shape invariants
    
    Total: ~199 features (all normalized separately)
    
    Benefits:
    - Enhanced color discrimination with brightness/reflectivity
    - Finer texture detail with 4-scale LBP
    - Better glass/metal/plastic distinction
    
    Args:
        image: Input image (BGR)
        
    Returns:
        feature_vector: Combined and balanced 1D feature vector
    """
    # Extract ENHANCED COLOR and TEXTURE features
    color_features = extract_color_histogram(image, bins=32)  # ~118 features (was 105)
    lbp_features = extract_lbp_features(image)  # ~68 features (was 54)
    shape_features = extract_shape_features(image)  # 7 features
    edge_features = extract_edge_features(image)  # 6 features
    
    # Normalize each feature type separately for equal voting power
    color_norm = (color_features - color_features.mean()) / (color_features.std() + 1e-7)
    lbp_norm = (lbp_features - lbp_features.mean()) / (lbp_features.std() + 1e-7)
    shape_norm = (shape_features - shape_features.mean()) / (shape_features.std() + 1e-7)
    edge_norm = (edge_features - edge_features.mean()) / (edge_features.std() + 1e-7)
    
    # Concatenate: Material-focused features (no shape bias)
    combined = np.concatenate([
        color_norm,  # ~118 features - PRIMARY discriminator with brightness/reflectivity
        lbp_norm,    # ~68 features - 4-scale texture patterns
        edge_norm,   # 6 features - edge quality
        shape_norm   # 7 features - shape invariants
    ])
    
    return combined


def extract_cnn_features(image, model_name='mobilenetv2'):
    """
    Extract features using pre-trained CNN (Transfer Learning)
    
    Uses a pre-trained CNN model (trained on ImageNet) as a feature extractor.
    The CNN automatically learns discriminative features from images.
    
    Advantages over handcrafted features:
    - Learns hierarchical features (edges → textures → objects)
    - Pre-trained on millions of images
    - Often achieves 5-10% higher accuracy
    - Captures complex patterns humans can't easily design
    
    Models:
    - MobileNetV2: Fast, lightweight (1280 features), recommended
    - ResNet50: Accurate, heavier (2048 features)
    - EfficientNetB0: Best balance (1280 features)
    
    Args:
        image: BGR image (224, 224, 3)
        model_name: CNN model to use ('mobilenetv2', 'resnet50', 'efficientnet')
        
    Returns:
        feature_vector: 1D numpy array (1280 or 2048 features depending on model)
    """
    # Convert BGR to RGB (Keras expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Expand dimensions for batch (model expects (batch_size, height, width, channels))
    image_batch = np.expand_dims(image_rgb, axis=0)
    
    # Select model and preprocessing
    if model_name.lower() == 'mobilenetv2':
        if not hasattr(extract_cnn_features, 'mobilenet_model'):
            print("Loading MobileNetV2 model (first time only)...")
            extract_cnn_features.mobilenet_model = MobileNetV2(
                weights='imagenet',
                include_top=False,  # Remove classification layer
                pooling='avg'  # Global average pooling
            )
        model = extract_cnn_features.mobilenet_model
        image_preprocessed = preprocess_mobilenet(image_batch)
        
    elif model_name.lower() == 'resnet50':
        if not hasattr(extract_cnn_features, 'resnet_model'):
            print("Loading ResNet50 model (first time only)...")
            extract_cnn_features.resnet_model = ResNet50(
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )
        model = extract_cnn_features.resnet_model
        image_preprocessed = preprocess_resnet(image_batch)
        
    elif model_name.lower() == 'efficientnet':
        if not hasattr(extract_cnn_features, 'efficientnet_model'):
            print("Loading EfficientNetB0 model (first time only)...")
            extract_cnn_features.efficientnet_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )
        model = extract_cnn_features.efficientnet_model
        image_preprocessed = preprocess_efficientnet(image_batch)
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'mobilenetv2', 'resnet50', or 'efficientnet'")
    
    # Extract features
    features = model.predict(image_preprocessed, verbose=0)
    
    # Flatten to 1D array
    return features.flatten()


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
    elif method in ['cnn', 'mobilenetv2', 'resnet50', 'efficientnet']:
        # CNN-based feature extraction
        model_name = method if method != 'cnn' else 'mobilenetv2'
        extractor = lambda img: extract_cnn_features(img, model_name=model_name)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hog', 'color', 'lbp', 'combined', 'cnn', 'mobilenetv2', 'resnet50', or 'efficientnet'")
    
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
