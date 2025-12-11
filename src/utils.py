"""
Utility Functions Module

This module contains helper functions used across the project.

Functions:
- Data loading and saving
- Image preprocessing
- Visualization helpers
- Configuration management
- Logging utilities
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config, config_path):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def resize_image(image, target_size=(224, 224)):
    """Resize image to target size"""
    return cv2.resize(image, target_size)


def normalize_image(image):
    """Normalize image to [0, 1] range"""
    return image.astype(np.float32) / 255.0


def display_images(images, labels, predictions=None, n_cols=5):
    """
    Display grid of images with labels and predictions
    
    Args:
        images: List of images
        labels: True labels
        predictions: Predicted labels (optional)
        n_cols: Number of columns in grid
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        if idx >= len(axes):
            break
        
        axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        title = f"True: {label}"
        if predictions is not None:
            title += f"\nPred: {predictions[idx]}"
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def log_metrics(metrics, log_file):
    """Append metrics to log file"""
    with open(log_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


def get_class_distribution(labels):
    """Calculate and return class distribution"""
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


if __name__ == "__main__":
    print("Utility functions module loaded successfully!")
