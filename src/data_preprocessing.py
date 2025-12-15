"""
Data Preprocessing and Augmentation Module

This module handles:
1. Loading images from the dataset
2. Applying data augmentation (minimum 30% increase)
3. Splitting data into training and validation sets
4. Saving augmented dataset

Key Tasks:
- Implement data loading from dataset folders
- Apply augmentation techniques: rotation, flipping, scaling, color jitter
- Ensure at least 30% increase in training samples
- Maintain class balance during augmentation
- Save processed data for feature extraction
"""

import os
import numpy as np
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
from tqdm import tqdm


# Class names mapping
CLASS_NAMES = {
    0: 'glass',
    1: 'paper',
    2: 'cardboard',
    3: 'plastic',
    4: 'metal',
    5: 'trash',
    6: 'unknown'  # Out-of-distribution or blurred inputs
}


def create_augmentation_pipeline():
    transform = A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),              # 50% chance to flip horizontally
        A.VerticalFlip(p=0.2),                # 20% chance to flip vertically
        A.Rotate(limit=15, p=0.7),            # Rotate ±15 degrees, 70% chance
        A.Affine(                             # Affine transform (replaces ShiftScaleRotate)
            scale=(0.9, 1.1),                 # Scale 90%-110%
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # Shift ±10%
            rotate=0,                         # No extra rotation (handled above)
            p=0.5
        ),
        
        # Color/Intensity transformations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,             # Brightness ±20%
            contrast_limit=0.2,               # Contrast ±20%
            p=0.5
        ),
        
        # Noise and blur
        A.GaussNoise(
            var_limit=(10, 50),               # Add slight noise (variance range)
            mean=0,
            p=0.3
        ),
        A.GaussianBlur(
            blur_limit=(3, 5),                # Slight blur
            p=0.2
        ),
    ])
    
    return transform


def load_dataset(dataset_path):
    images = []
    labels = []
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path.absolute()}")
    
    # Iterate through each class folder
    for class_id, class_name in CLASS_NAMES.items():
        class_folder = dataset_path / class_name
        
        if not class_folder.exists():
            print(f"Warning: Class folder not found: {class_folder}")
            continue
        
        # Get all image files
        image_files = list(class_folder.glob("*.jpg")) + \
                     list(class_folder.glob("*.jpeg")) + \
                     list(class_folder.glob("*.png"))
        
        print(f"Loading {len(image_files)} images from '{class_name}' (ID: {class_id})")
        
        # Load each image
        for img_path in tqdm(image_files, desc=f"Loading {class_name}"):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                
                if img is None:
                    print(f"Warning: Could not load: {img_path}")
                    continue
                
                # Resize to standard size (224x224)
                img = cv2.resize(img, (224, 224))
                
                images.append(img)
                labels.append(class_id)
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Print statistics
    print(f"\nDataset loaded successfully!")
    print(f"Total images: {len(images)}")
    print(f"Image shape: {images[0].shape if len(images) > 0 else 'N/A'}")
    print(f"Data Columns Names: {images.dtype.names if images.dtype.names else 'N/A'}")
    print("\nClass distribution:")
    for class_id in range(len(CLASS_NAMES)):
        count = np.sum(labels == class_id)
        print(f"  {class_id} ({CLASS_NAMES[class_id]}): {count} images")
    
    return images, labels, CLASS_NAMES


def generate_unknown_class(images, labels, unknown_ratio=0.15):
    print(f"\n{'='*60}")
    print("GENERATING UNKNOWN CLASS (Out-of-Distribution)")
    print(f"{'='*60}")
    
    # Calculate number of unknown samples to generate
    num_originals = len(images)
    num_unknown = int(num_originals * unknown_ratio)
    
    print(f"\nOriginal dataset size: {num_originals}")
    print(f"Unknown samples to generate: {num_unknown} ({unknown_ratio*100}% of dataset)")
    
    # Create extreme transformation pipeline for unknown class
    unknown_transform = A.Compose([
        # Extreme blur to make unrecognizable
        A.OneOf([
            A.GaussianBlur(blur_limit=(15, 25), p=1.0),  # Heavy blur
            A.MotionBlur(blur_limit=(15, 25), p=1.0),    # Motion blur
            A.MedianBlur(blur_limit=15, p=1.0),          # Median blur
        ], p=0.8),
        
        # Heavy noise
        A.OneOf([
            A.GaussNoise(var_limit=(100.0, 200.0), p=1.0),  # Extreme noise
            A.ISONoise(color_shift=(0.3, 0.8), intensity=(0.7, 1.0), p=1.0),
        ], p=0.7),
        
        # Severe distortions
        A.OneOf([
            A.ElasticTransform(alpha=200, sigma=20, p=1.0),
            A.GridDistortion(num_steps=10, distort_limit=0.5, p=1.0),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.3, p=1.0),
        ], p=0.6),
        
        # Extreme color/brightness changes
        A.RandomBrightnessContrast(
            brightness_limit=(-0.5, 0.5),
            contrast_limit=(-0.5, 0.5),
            p=0.8
        ),
        
        # Random crops and rotations
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, p=0.5),
    ])
    
    unknown_images = []
    unknown_labels = []
    
    print("Generating unknown samples...")
    for _ in tqdm(range(num_unknown), desc="Creating unknown class"):
        # Randomly select an image from any class
        idx = np.random.randint(0, len(images))
        original_img = images[idx]
        
        # Apply extreme transformations
        transformed = unknown_transform(image=original_img)
        unknown_img = transformed['image']
        
        unknown_images.append(unknown_img)
        unknown_labels.append(6)  # Class ID for unknown
    
    # Combine original + unknown
    images_with_unknown = np.concatenate([images, np.array(unknown_images)])
    labels_with_unknown = np.concatenate([labels, np.array(unknown_labels)])
    
    print(f"\n{'='*60}")
    print("UNKNOWN CLASS GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total dataset size: {len(images_with_unknown)}")
    print(f"Unknown samples added: {num_unknown}")
    print(f"\nFinal class distribution:")
    for class_id in range(7):  # 0-6 including unknown
        count = np.sum(labels_with_unknown == class_id)
        class_name = CLASS_NAMES[class_id]
        print(f"  {class_id} ({class_name}): {count} images")
    
    return images_with_unknown, labels_with_unknown


def augment_dataset(images, labels, augmentation_factor=0.3):
    print(f"\n{'='*60}")
    print("STARTING DATA AUGMENTATION")
    print(f"{'='*60}")
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Calculate how many augmented samples to create
    num_originals = len(images)
    num_to_augment = int(num_originals * augmentation_factor)
    
    print(f"\nOriginal dataset size: {num_originals}")
    print(f"Augmentation factor: {augmentation_factor} ({augmentation_factor*100}%)")
    print(f"Augmented samples to generate: {num_to_augment}")
    print(f"Target total size: {num_originals + num_to_augment}")
    
    # Start with original images and labels
    augmented_images = list(images)
    augmented_labels = list(labels)
    
    # Get unique classes for proportional augmentation
    unique_classes = np.unique(labels)
    
    print(f"\nAugmenting each class proportionally...")
    
    # Augment each class separately to maintain balance
    for class_id in unique_classes:
        # Get indices and images for this class
        class_indices = np.where(labels == class_id)[0]
        class_images = images[class_indices]
        
        # Calculate how many augmented samples for this class
        class_aug_count = int(len(class_images) * augmentation_factor)
        
        if class_aug_count == 0:
            continue
        
        class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
        print(f"  {class_name}: generating {class_aug_count} augmented images...")
        
        # Generate augmented images for this class
        for _ in tqdm(range(class_aug_count), desc=f"  Augmenting {class_name}", leave=False):
            # Randomly select an image from this class
            idx = np.random.randint(0, len(class_images))
            original_img = class_images[idx]
            
            # Apply augmentation
            augmented = transform(image=original_img)
            aug_img = augmented['image']
            
            # Add to augmented dataset
            augmented_images.append(aug_img)
            augmented_labels.append(class_id)
    
    # Convert back to numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    # Print results
    print(f"\n{'='*60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Final dataset size: {len(augmented_images)}")
    print(f"Increase: {len(augmented_images) - num_originals} images")
    print(f"Actual increase: {((len(augmented_images) - num_originals) / num_originals * 100):.1f}%")
    
    print(f"\nAugmented class distribution:")
    for class_id in unique_classes:
        count = np.sum(augmented_labels == class_id)
        original_count = np.sum(labels == class_id)
        increase = count - original_count
        print(f"  {CLASS_NAMES[class_id]}: {original_count} → {count} (+{increase})")
    
    return augmented_images, augmented_labels


def split_data(images, labels, test_size=0.2, random_state=42):
    pass


# Main execution
if __name__ == "__main__":
    # Define paths - use Path to handle relative paths correctly
    # Get the directory where this script is located
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent  # Go up one level to project root
    
    DATASET_PATH = PROJECT_ROOT / "dataset"  # Absolute path to dataset
    OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
    
    print("="*60)
    print("TESTING load_dataset() FUNCTION")
    print("="*60)
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset path: {DATASET_PATH}")
    print("="*60)
    
    try:
        # Test loading dataset
        images, labels, class_names = load_dataset(DATASET_PATH)
        
        # Generate unknown class (out-of-distribution)
        images, labels = generate_unknown_class(images, labels, unknown_ratio=0.15)
        
        # Test augmentation (with smaller factor for testing)
        augmented_images, augmented_labels = augment_dataset(images, labels, augmentation_factor=0.3)
        
        # Display results
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        
        print(f"\nOriginal dataset:")
        print(f"   - Images shape: {images.shape}")
        print(f"   - Labels shape: {labels.shape}")
        
        print(f"\nAugmented dataset:")
        print(f"   - Images shape: {augmented_images.shape}")
        print(f"   - Labels shape: {augmented_labels.shape}")
        print(f"   - Total increase: {len(augmented_images) - len(images)} images")
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
