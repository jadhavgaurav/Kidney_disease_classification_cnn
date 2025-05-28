import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Source dataset path (all classes)
source_dir = 'data/extract/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'

# Target base directory
target_base = 'data/kidney_split'
train_dir = os.path.join(target_base, 'train')
val_dir = os.path.join(target_base, 'val')

# Create target directories
for split_dir in [train_dir, val_dir]:
    os.makedirs(split_dir, exist_ok=True)

# 80-20 train-val split
split_ratio = 0.8

# Loop through each class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # List all image files
    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create class subfolders in train and val
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Copy training images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.copy2(src, dst)

    # Copy validation images
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_dir, class_name, img)
        shutil.copy2(src, dst)

print("✅ Dataset split into train/val folders successfully.")
