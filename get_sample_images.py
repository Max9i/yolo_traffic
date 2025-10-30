"""
Get sample images from the dataset for testing
"""

import os
import shutil
import random
from pathlib import Path

# Paths
val_images = Path("yolov8/datasets/traffic/images/val")
val_labels = Path("yolov8/datasets/traffic/labels/val")
output_dir = Path("sample_test_images")

# Create output directory
output_dir.mkdir(exist_ok=True)

# Class names
class_names = {
    0: "5kilometer",
    1: "15kilometer",
    2: "30kilometer",
    3: "40kilometer",
    4: "50kilometer",
    5: "60kilometer",
    6: "70kilometer",
    7: "90kilometer",
}

def find_images_for_class(class_id, num_samples=3):
    """Find sample images containing specific class"""
    images_with_class = []

    for label_file in val_labels.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0 and int(parts[0]) == class_id:
                    img_name = label_file.stem + ".png"
                    img_path = val_images / img_name
                    if img_path.exists():
                        images_with_class.append(img_path)
                    break

    return random.sample(images_with_class, min(num_samples, len(images_with_class)))

# Get samples for each speed limit class
print("Collecting sample images...")
for class_id, class_name in class_names.items():
    samples = find_images_for_class(class_id, num_samples=3)

    if samples:
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)

        for i, img_path in enumerate(samples):
            dest = class_dir / f"{class_name}_sample_{i+1}.png"
            shutil.copy(img_path, dest)
            print(f"  Copied: {dest}")
    else:
        print(f"  No images found for {class_name}")

print(f"\nSample images saved to: {output_dir}")
print("\nYou can now use these images to test the demo!")
