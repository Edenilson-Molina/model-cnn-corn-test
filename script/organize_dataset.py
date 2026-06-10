"""
Script to split the corn dataset from data/test into train, val, and test sets.
The split ratio is:
- 70% Train
- 15% Validation
- 15% Test

The files are moved from data/test into data/train and data/val, leaving the remaining 15% in data/test.
"""

import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

def organize_dataset():
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory not found at {TEST_DIR}")
        return

    # Get the classes (subdirectories in data/test)
    classes = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    print(f"Found classes: {classes}")
    
    # We want to make sure train and val directories exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    summary_counts = {}

    for cls in classes:
        cls_test_dir = os.path.join(TEST_DIR, cls)
        cls_train_dir = os.path.join(TRAIN_DIR, cls)
        cls_val_dir = os.path.join(VAL_DIR, cls)

        # Create target directories
        os.makedirs(cls_train_dir, exist_ok=True)
        os.makedirs(cls_val_dir, exist_ok=True)

        # Get all files in the class folder
        all_files = [f for f in os.listdir(cls_test_dir) if os.path.isfile(os.path.join(cls_test_dir, f))]
        
        # Shuffle the files
        random.shuffle(all_files)

        total_files = len(all_files)
        if total_files == 0:
            print(f"Warning: No files found for class {cls}")
            continue

        # Calculate splits
        train_count = int(total_files * 0.70)
        val_count = int(total_files * 0.15)
        test_count = total_files - train_count - val_count

        train_files = all_files[:train_count]
        val_files = all_files[train_count:train_count + val_count]
        test_files = all_files[train_count + val_count:]

        print(f"\nClass '{cls}':")
        print(f"  Total files: {total_files}")
        print(f"  Moving to train: {len(train_files)} ({len(train_files)/total_files*100:.1f}%)")
        print(f"  Moving to val:   {len(val_files)} ({len(val_files)/total_files*100:.1f}%)")
        print(f"  Remaining in test: {len(test_files)} ({len(test_files)/total_files*100:.1f}%)")

        # Move to Train
        for f in train_files:
            src = os.path.join(cls_test_dir, f)
            dst = os.path.join(cls_train_dir, f)
            shutil.move(src, dst)

        # Move to Val
        for f in val_files:
            src = os.path.join(cls_test_dir, f)
            dst = os.path.join(cls_val_dir, f)
            shutil.move(src, dst)

        summary_counts[cls] = {
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files),
            'total': total_files
        }

    print("\n" + "="*40)
    print("Dataset Organization Completed Successfully!")
    print("="*40)
    for cls, counts in summary_counts.items():
        print(f"{cls.upper():<25} | Train: {counts['train']:<5} | Val: {counts['val']:<5} | Test: {counts['test']:<5} | Total: {counts['total']}")
    print("="*40)

if __name__ == "__main__":
    organize_dataset()
