import cv2
import numpy as np
import joblib
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent))
from feature_extraction import extract_combined_features, extract_cnn_features


class WasteClassifier:
    """Waste classification inference engine"""
    
    def __init__(self, model_dir='models', use_cnn=True):
        self.model_dir = Path(model_dir)
        self.use_cnn = use_cnn
        
        print("="*60)
        print("LOADING WASTE CLASSIFIER")
        print("="*60)
        
        # Load model components
        self.model = joblib.load(self.model_dir / ("svm_classifier_cnn.pkl" if use_cnn else "svm_classifier_handcrafted.pkl"))
        self.scaler = joblib.load(self.model_dir / ("feature_scaler_cnn.pkl" if use_cnn else "feature_scaler_handcrafted.pkl"))
        self.class_names = joblib.load(self.model_dir / "class_names.pkl")
        
        print(f"✓ Model loaded from: {self.model_dir.absolute()}")
        print(f"✓ Feature method: {'CNN (MobileNetV2)' if use_cnn else 'Handcrafted (Color+LBP+Edge+Shape)'}")
        print(f"✓ Classes: {list(self.class_names.values())}")
        print("="*60)
    
    def classify_image(self, image, confidence_threshold=0.5):
        # Resize to standard size
        image_resized = cv2.resize(image, (224, 224))
        
        # Extract features
        if self.use_cnn:
            features = extract_cnn_features(image_resized)
        else:
            features = extract_combined_features(image_resized)
        
        features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        
        proba = self.model.predict_proba(features_scaled)[0]
        class_id = self.model.predict(features_scaled)[0]
        confidence = proba.max()
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            return "UNKNOWN", confidence, True
        else:
            return self.class_names[class_id], confidence, False
    
    def classify_from_path(self, image_path, confidence_threshold=0.5, display=True):
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Classify
        prediction, confidence, is_unknown = self.classify_image(
            image, confidence_threshold
        )
        
        # Display results
        if display:
            self._display_result(image, prediction, confidence, is_unknown)
        
        return prediction, confidence, is_unknown
    
    def classify_directory(self, directory_path, confidence_threshold=0.5):
        directory = Path(directory_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        results = []
        for img_path in directory.iterdir():
            if img_path.suffix.lower() in image_extensions:
                try:
                    pred, conf, unknown = self.classify_from_path(
                        img_path, confidence_threshold, display=False
                    )
                    results.append((img_path.name, pred, conf, unknown))
                    print(f"✓ {img_path.name}: {pred} ({conf:.2%})")
                except Exception as e:
                    print(f"✗ {img_path.name}: Error - {e}")
        
        return results
    
    def run_camera(self, confidence_threshold=0.5, camera_id=0):
        print("\n" + "="*60)
        print("STARTING CAMERA MODE")
        print("="*60)
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Classify every 10 frames (reduce computation)
            if frame_count % 10 == 0:
                try:
                    prediction, confidence, is_unknown = self.classify_image(
                        frame, confidence_threshold
                    )
                    
                    # Draw prediction on frame
                    self._draw_prediction(frame, prediction, confidence, is_unknown)
                    
                except Exception as e:
                    cv2.putText(frame, f"Error: {str(e)}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Waste Classifier - Press Q to quit', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f"results/classified_frame_{frame_count}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"✓ Saved frame to: {save_path}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Camera closed")
    
    def _draw_prediction(self, frame, prediction, confidence, is_unknown):
        """Draw prediction text on frame"""
        # Prepare text
        status = "LOW CONFIDENCE" if is_unknown else "CONFIDENT"
        color = (0, 0, 255) if is_unknown else (0, 255, 0)  # Red if unknown, green otherwise
        
        text1 = f"Class: {prediction}"
        text2 = f"Confidence: {confidence:.2%}"
        text3 = f"Status: {status}"
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, text3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _display_result(self, image, prediction, confidence, is_unknown):
        """Display image with classification result"""
        # Resize for display if too large
        height, width = image.shape[:2]
        if height > 800 or width > 800:
            scale = 800 / max(height, width)
            image = cv2.resize(image, (int(width*scale), int(height*scale)))
        
        # Draw prediction
        self._draw_prediction(image, prediction, confidence, is_unknown)
        
        # Display
        cv2.imshow('Classification Result - Press any key to close', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    print("\n" + "="*60)
    print("       WASTE CLASSIFICATION SYSTEM")
    print("="*60)
    
    # 1. Select mode
    print("\n📷 Select Mode:")
    print("  1. Real-time Camera")
    print("  2. Upload Image")
    mode = input("\nEnter choice (1 or 2): ").strip()
    
    # 2. Select feature type
    print("\n🔍 Select Feature Extraction:")
    print("  1. CNN (MobileNetV2) - More accurate, slower")
    print("  2. Handcrafted (Color+LBP) - Faster, less accurate")
    feature = input("\nEnter choice (1 or 2): ").strip()
    use_cnn = feature != "2"
    
    # 3. Select classifier
    print("\n🤖 Select Classifier:")
    print("  1. SVM")
    print("  2. KNN")
    clf_choice = input("\nEnter choice (1 or 2): ").strip()
    
    # Determine model file
    if clf_choice == "2":
        model_file = "knn_classifier_cnn.pkl" if use_cnn else "knn_classifier_handcrafted.pkl"
    else:
        model_file = "svm_classifier_cnn.pkl" if use_cnn else "svm_classifier_handcrafted.pkl"
    
    print("\n" + "="*60)
    print("LOADING MODEL...")
    print("="*60)
    
    # Initialize classifier
    try:
        classifier = WasteClassifier(model_dir='models', use_cnn=use_cnn)
        # Override model if KNN selected
        if clf_choice == "2":
            classifier.model = joblib.load(Path('models') / model_file)
            print(f"✓ Switched to KNN classifier")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run selected mode
    try:
        if mode == "1":
            # Camera mode
            classifier.run_camera(confidence_threshold=0.5, camera_id=0)
        else:
            # Image mode
            image_path = input("\n📁 Enter image path: ").strip()
            image_path = image_path.strip('"').strip("'")  # Remove quotes if any
            
            if not Path(image_path).exists():
                print(f"❌ File not found: {image_path}")
                return
            
            prediction, confidence, is_unknown = classifier.classify_from_path(
                image_path, confidence_threshold=0.5
            )
            
            print(f"\n{'='*60}")
            print(f"CLASSIFICATION RESULT")
            print(f"{'='*60}")
            print(f"Image: {image_path}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Status: {'⚠️ LOW CONFIDENCE' if is_unknown else '✓ CONFIDENT'}")
            print(f"{'='*60}")
    
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
