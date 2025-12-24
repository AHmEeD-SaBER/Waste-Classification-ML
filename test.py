"""
Test file for Waste Classification Model Evaluation

This file contains the predict function for evaluation on a hidden dataset.
Saves results to Excel file.
"""

import cv2
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
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

def predict(dataFilePath, bestModelPath, confidence_threshold=0.5):
    """
    Predict waste classification for images in a folder and save to Excel.
    
    Args:
        dataFilePath: Path to folder containing images
        bestModelPath: Path to the trained model (.pkl file)
        confidence_threshold: Minimum confidence to accept prediction (default: 0.5)
                            If max probability < threshold, classify as 'unknown'
    
    Returns:
        List of predicted class names
    """
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
    confidences = []
    valid_images = []
    valid_indices = []
    
    # First pass: load and collect all valid images
    for idx, img_path in enumerate(image_files):
        # Load image
        image = cv2.imread(str(img_path))
        
        if image is None:
            # Mark this index as invalid
            continue
        
        # Resize to standard size (224x224)
        image_resized = cv2.resize(image, (224, 224))
        valid_images.append(image_resized)
        valid_indices.append(idx)
    
    # Extract features for all valid images at once
    if valid_images:
        if use_cnn:
            all_features = extract_features_from_dataset(valid_images, method='cnn')
        else:
            all_features = extract_features_from_dataset(valid_images, method='combined')
        
        # Scale all features
        all_features_scaled = scaler.transform(all_features)
        
        # Get predictions for all images
        all_proba = model.predict_proba(all_features_scaled)
    
    # Build predictions list maintaining original order
    feature_idx = 0
    for idx in range(len(image_files)):
        if idx in valid_indices:
            proba = all_proba[feature_idx]
            max_proba = np.max(proba)
            class_id = np.argmax(proba)
            
            # Unknown rejection: if confidence is below threshold, classify as 'unknown'
            if max_proba < confidence_threshold:
                predictions.append('unknown')
                confidences.append(max_proba)
            else:
                class_name = CLASS_NAMES.get(class_id, 'unknown')
                predictions.append(class_name)
                confidences.append(max_proba)
            feature_idx += 1
        else:
            # Image couldn't be loaded, classify as unknown
            predictions.append('unknown')
            confidences.append(0.0)
    
    # Create DataFrame with results - only image name and predicted label
    results_df = pd.DataFrame({
        'Image_Name': [img.name for img in image_files],
        'Predicted_Label': predictions
    })
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_excel_path = results_dir / f"waste_classification_results_{timestamp}.xlsx"
    
    # Save to Excel - try openpyxl first, fallback to xlsxwriter
    try:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Get the worksheet
            worksheet = writer.sheets['Predictions']
            
            # Adjust column widths
            worksheet.column_dimensions['A'].width = 35  # Image_Name
            worksheet.column_dimensions['B'].width = 20  # Predicted_Label
    except ImportError:
        # Fallback to xlsxwriter if openpyxl not available
        print("\nNote: openpyxl not found, using xlsxwriter as fallback...")
        with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Predictions']
            worksheet.set_column(0, 0, 35)  # Image_Name
            worksheet.set_column(1, 1, 20)  # Predicted_Label
    
    print(f"\n✅ Results saved to: {output_excel_path}")
    
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
    print("RESULTS SUMMARY:")
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
    print("\n" + "="*60)