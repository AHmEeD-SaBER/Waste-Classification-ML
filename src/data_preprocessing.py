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
    6: 'unknown'
}


def create_augmentation_pipeline():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),              # 50% chance to flip horizontally
        A.VerticalFlip(p=0.2),                # 20% chance to flip vertically
        A.Rotate(limit=15, p=0.7),            # Rotate ±15 degrees, 70% chance
        A.Affine(                             # Affine transform (replaces ShiftScaleRotate)
            scale=(0.9, 1.1),                 # Scale 90%-110%
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # Shift ±10%
            rotate=0,                         # No extra rotation (handled above)
            p=0.5
        ),
        
        # Color/Intensity transformations (PRESERVED - critical for color features)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,             # Brightness ±20%
            contrast_limit=0.2,               # Contrast ±20%
            p=0.5
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


def augment_dataset(images, labels, augmentation_factor=0.3, balance_classes=False):
    print(f"\n{'='*60}")
    print("STARTING DATA AUGMENTATION")
    print(f"{'='*60}")
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Calculate class sizes
    class_sizes = {class_id: np.sum(labels == class_id) for class_id in unique_classes}
    num_originals = len(images)
    
    print(f"\nOriginal dataset size: {num_originals}")
    print(f"Augmentation budget: {augmentation_factor*100}% ({int(num_originals * augmentation_factor)} new samples)")
    print(f"\nOriginal class distribution:")
    for class_id in unique_classes:
        count = class_sizes[class_id]
        class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
        pct = (count / num_originals * 100)
        print(f"  {class_name}: {count} images ({pct:.1f}%)")
    
    # Start with original images and labels
    augmented_images = list(images)
    augmented_labels = list(labels)
    
    if balance_classes:
        print(f"\n🎯 SMART BALANCING: Distributing {int(num_originals * augmentation_factor)} new samples to balance classes")
        
        # Calculate total budget
        total_budget = int(num_originals * augmentation_factor)
        
        # Calculate class weights (inverse of class size - smaller classes get more)
        class_weights = {}
        total_weight = 0
        for class_id in unique_classes:
            # Inverse proportion: smaller classes have higher weight
            weight = 1.0 / class_sizes[class_id]
            class_weights[class_id] = weight
            total_weight += weight
        
        # Normalize weights and calculate augmentation for each class
        class_aug_counts = {}
        for class_id in unique_classes:
            # Proportional to inverse class size
            normalized_weight = class_weights[class_id] / total_weight
            aug_count = int(total_budget * normalized_weight)
            class_aug_counts[class_id] = aug_count
        
        print("\n📊 Smart distribution (more samples to smaller classes):")
    else:
        print(f"\nProportional augmentation: {augmentation_factor*100}% increase per class")
        class_aug_counts = {}
        for class_id in unique_classes:
            class_aug_counts[class_id] = int(class_sizes[class_id] * augmentation_factor)
    
    print(f"\nAugmenting classes...")
    
    # Augment each class
    for class_id in unique_classes:
        # Get indices and images for this class
        class_indices = np.where(labels == class_id)[0]
        class_images = images[class_indices]
        original_count = len(class_images)
        
        class_aug_count = class_aug_counts[class_id]
        
        if class_aug_count <= 0:
            class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
            print(f"  {class_name}: no augmentation")
            continue
        
        class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
        target_count = original_count + class_aug_count
        increase_pct = (class_aug_count / original_count * 100)
        final_pct = (target_count / (num_originals + sum(class_aug_counts.values())) * 100)
        print(f"  {class_name}: {original_count} → {target_count} (+{class_aug_count}, +{increase_pct:.0f}%) = {final_pct:.1f}% of total")
        
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


def save_preprocessed_data(images, labels, output_path, filename='preprocessed_data.npz'):
    """
    Save preprocessed images and labels to disk for fast loading later
    
    This avoids having to reload from folders and reprocess images every time.
    Uses numpy's compressed format for efficient storage.
    
    Args:
        images: Numpy array of images (N, H, W, C)
        labels: Numpy array of labels (N,)
        output_path: Directory to save the file
        filename: Name of the output file (default: preprocessed_data.npz)
    
    Returns:
        Path to saved file
    """
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    
    print(f"\n{'='*60}")
    print("SAVING PREPROCESSED DATA")
    print(f"{'='*60}")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Saving to: {filepath}")
    
    # Save using numpy's compressed format
    np.savez_compressed(
        filepath,
        images=images,
        labels=labels
    )
    
    # Get file size
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"✓ Saved successfully!")
    print(f"✓ File size: {file_size_mb:.2f} MB")
    print(f"{'='*60}")
    
    return str(filepath)


def load_preprocessed_data(filepath):
    """
    Load preprocessed images and labels from disk
    
    Much faster than loading from folders - loads directly into RAM.
    
    Args:
        filepath: Path to the .npz file
    
    Returns:
        images: Numpy array of images (N, H, W, C)
        labels: Numpy array of labels (N,)
    """
    from pathlib import Path
    import time
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Preprocessed data not found at: {filepath}")
    
    print(f"\n{'='*60}")
    print("LOADING PREPROCESSED DATA FROM CACHE")
    print(f"{'='*60}")
    print(f"Loading from: {filepath}")
    
    start_time = time.time()
    
    # Load the data
    data = np.load(filepath)
    images = data['images']
    labels = data['labels']
    
    load_time = time.time() - start_time
    
    print(f"✓ Loaded in {load_time:.2f} seconds")
    print(f"✓ Images shape: {images.shape}")
    print(f"✓ Labels shape: {labels.shape}")
    print(f"✓ Unique classes: {np.unique(labels)}")
    print(f"{'='*60}")
    
    return images, labels


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
