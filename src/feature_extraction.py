import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet


def extract_color_histogram(image, bins=32):
    hist_features = []
    
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7) 
        hist_features.extend(hist)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    for i in range(3):
        hist = cv2.calcHist([hsv_image], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  
        hist_features.extend(hist)
    
    return np.array(hist_features)


def extract_lbp_features(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    all_hist = []
    # Multi-scale LBP with varying (num_points, radius)
    configs = [
        (8, 1),   # Fine texture, coarse angular - 10 bins
        (16, 2),  # Medium scale - 18 bins  
        (24, 3),  # Medium-large scale - 26 bins
        (24, 5),  # Large/coarse texture - 26 bins
    ]
    
    for p, r in configs:
        lbp = local_binary_pattern(gray, p, r, method='uniform')
        n_bins = p + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist = hist / (hist.sum() + 1e-7)
        all_hist.extend(hist)
    
    return np.array(all_hist)  # 80 features (10 + 18 + 26 + 26)


def extract_combined_features(image):
    color_features = extract_color_histogram(image, bins=32)  # 192 features
    lbp_features = extract_lbp_features(image)  # 26 features
    
    combined = np.concatenate([color_features, lbp_features])    
    return combined


def extract_cnn_features(image, model_name='mobilenetv2'):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Expand dimensions for batch (model expects (batch_size, height, width, channels))
    image_batch = np.expand_dims(image_rgb, axis=0)
    
    # Select model and preprocessing
    if model_name.lower() == 'mobilenetv2':
        if not hasattr(extract_cnn_features, 'mobilenet_model'):
            print("Loading MobileNetV2 model")
            extract_cnn_features.mobilenet_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3)
            )
        model = extract_cnn_features.mobilenet_model
        image_preprocessed = preprocess_mobilenet(image_batch)
        
    elif model_name.lower() == 'resnet50':
        if not hasattr(extract_cnn_features, 'resnet_model'):
            print("Loading ResNet50 model")
            extract_cnn_features.resnet_model = ResNet50(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3)
            )
        model = extract_cnn_features.resnet_model
        image_preprocessed = preprocess_resnet(image_batch)
        
    elif model_name.lower() == 'efficientnet':
        if not hasattr(extract_cnn_features, 'efficientnet_model'):
            print("Loading EfficientNetB0 model")
            extract_cnn_features.efficientnet_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3)
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
    from tqdm import tqdm
    
    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES: {method.upper()} METHOD")
    print(f"{'='*60}")
    print(f"Total images: {len(images)}")
    
    # Select feature extraction method
    if method == 'color':
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
        raise ValueError(f"Unknown method: {method}. Use 'color', 'lbp', 'combined', 'cnn', 'mobilenetv2', 'resnet50', or 'efficientnet'")
    
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
