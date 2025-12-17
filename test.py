"""
Test file for Waste Classification Model Evaluation

This file contains the predict function for evaluation on a hidden dataset.
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
from skimage.feature import local_binary_pattern
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet

# CNN imports (only loaded when needed)
_cnn_model = None

def get_cnn_model():
    """Lazy load MobileNetV2 model"""
    global _cnn_model
    if _cnn_model is None:
        from tensorflow.keras.applications import MobileNetV2
        print("Loading MobileNetV2 model (first time only)...")
        _cnn_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
    return _cnn_model


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
            print("Loading MobileNetV2 model (first time only)...")
            extract_cnn_features.mobilenet_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                pooling='avg'
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


def predict(dataFilePath, bestModelPath):
    # Define class names mapping
    CLASS_NAMES = {
        0: 'glass',
        1: 'paper',
        2: 'cardboard',
        3: 'plastic',
        4: 'metal',
        5: 'trash'
    }
    
    # Load model and scaler
    model_path = Path(bestModelPath)
    model_dir = model_path.parent
    
    # Load the model
    model = joblib.load(model_path)
    
    # Determine if CNN or handcrafted based on model filename
    use_cnn = 'cnn' in model_path.name.lower()
    
    # Load the scaler (assumes it's in the same directory as the model)
    if use_cnn:
        scaler_path = model_dir / "feature_scaler_cnn.pkl"
    else:
        scaler_path = model_dir / "feature_scaler_handcrafted.pkl"
    
    scaler = joblib.load(scaler_path)
    
    # Load class names if available, otherwise use default
    class_names_path = model_dir / "class_names.pkl"
    if class_names_path.exists():
        CLASS_NAMES = joblib.load(class_names_path)
    
    # Get all images from the folder
    data_path = Path(dataFilePath)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_path.glob(f'*{ext}'))
        image_files.extend(data_path.glob(f'*{ext.upper()}'))
    
    # Sort files for consistent ordering
    image_files = sorted(set(image_files))
    
    predictions = []
    
    for img_path in image_files:
        # Load image
        image = cv2.imread(str(img_path))
        
        if image is None:
            # If image can't be loaded, predict most common class
            predictions.append('trash')
            continue
        
        # Resize to standard size (224x224)
        image_resized = cv2.resize(image, (224, 224))
        
        # Extract features based on model type
        if use_cnn:
            features = extract_cnn_features(image_resized)
        else:
            features = extract_combined_features(image_resized)
        
        # Reshape for sklearn (1 sample)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        class_id = model.predict(features_scaled)[0]
        
        # Convert to class name
        class_name = CLASS_NAMES.get(class_id, 'trash')
        predictions.append(class_name)
    
    return predictions


# For testing purposes
if __name__ == "__main__":
    print("\n" + "="*60)
    print("       WASTE CLASSIFICATION - TEST MODE")
    print("="*60)
    
    # Check if models exist and show their accuracies
    models_dir = Path("models")
    
    print("\n📊 Available Models & Accuracy:")
    print("-"*40)
    
    # Check SVM + Handcrafted
    svm_hc_path = models_dir / "svm_classifier_handcrafted.pkl"
    if svm_hc_path.exists():
        print("  ✓ SVM + Handcrafted:  ~82% accuracy")
    else:
        print("  ✗ SVM + Handcrafted:  NOT FOUND")
    
    # Check SVM + CNN
    svm_cnn_path = models_dir / "svm_classifier_cnn.pkl"
    if svm_cnn_path.exists():
        print("  ✓ SVM + CNN:          ~93% accuracy")
    else:
        print("  ✗ SVM + CNN:          NOT FOUND")
    
    print("-"*40)
    
    # Menu for feature selection
    print("\n🔍 Select Feature Extraction Method:")
    print("  1. Handcrafted (Color + LBP) - Faster")
    print("  2. CNN (MobileNetV2) - More accurate")
    
    feature_choice = input("\nEnter choice (1 or 2): ").strip()
    
    if feature_choice == "2":
        use_cnn = True
        model_path = models_dir / "svm_classifier_cnn.pkl"
        print("\n✓ Selected: SVM + CNN")
    else:
        use_cnn = False
        model_path = models_dir / "svm_classifier_handcrafted.pkl"
        print("\n✓ Selected: SVM + Handcrafted")
    
    # Check if model exists
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please train and save the model first.")
        exit(1)
    
    # Get data folder path
    print("\n📁 Enter the path to the folder containing images:")
    data_path = input("   Path: ").strip().strip('"').strip("'")
    
    if not Path(data_path).exists():
        print(f"\n❌ Error: Folder not found: {data_path}")
        exit(1)
    
    # Run prediction
    print("\n" + "="*60)
    print("RUNNING PREDICTIONS...")
    print("="*60)
    
    results = predict(data_path, str(model_path))
    
    print(f"\n✅ Predictions Complete!")
    print(f"   Total images: {len(results)}")
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    
    # Count predictions per class
    from collections import Counter
    counts = Counter(results)
    
    print(f"\n{'Class':<15} {'Count':>8}")
    print("-"*25)
    for cls, count in sorted(counts.items()):
        print(f"{cls:<15} {count:>8}")
    
    print("\n" + "-"*25)
    print(f"{'TOTAL':<15} {len(results):>8}")
    
    # Show individual predictions
    print("\n" + "="*60)
    print("INDIVIDUAL PREDICTIONS:")
    print("="*60)
    
    image_files = sorted(Path(data_path).glob('*'))
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_files = [f for f in image_files if f.suffix.lower() in image_extensions]
    
    for i, (img_file, pred) in enumerate(zip(image_files, results), 1):
        print(f"  {i:3}. {img_file.name:<30} → {pred}")
    
    print("\n" + "="*60)
