"""
Script to limit the corn dataset to exactly 600 images per class
and split them into:
- Train: 70% (420 images)
- Val: 15% (90 images)
- Test: 15% (90 images)

Extra images are moved to data/unused/<class_name> for safekeeping.
"""

import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
UNUSED_DIR = os.path.join(DATA_DIR, "unused")

CLASSES = ['gray_leaf_spot', 'healthy', 'magnesium_deficiency']
LIMIT_PER_CLASS = 600

def balance_and_split():
    print("Starting dataset balancing and splitting...")
    print(f"Target: {LIMIT_PER_CLASS} images per class (Train: 420, Val: 90, Test: 90)\n")

    summary_counts = {}

    for cls in CLASSES:
        # 1. Gather all files currently in train, val, and test for this class
        all_file_entries = []
        for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                for f in os.listdir(cls_dir):
                    path = os.path.join(cls_dir, f)
                    if os.path.isfile(path):
                        all_file_entries.append((f, path))

        total_available = len(all_file_entries)
        print(f"Class '{cls}': Found {total_available} total images across train/val/test directories.")

        if total_available < LIMIT_PER_CLASS:
            print(f"Error: Class '{cls}' only has {total_available} images, which is less than the requested {LIMIT_PER_CLASS}.")
            return

        # 2. Shuffle the files
        random.shuffle(all_file_entries)

        # 3. Select 600 files for active dataset, rest go to unused
        selected_entries = all_file_entries[:LIMIT_PER_CLASS]
        unused_entries = all_file_entries[LIMIT_PER_CLASS:]

        # 4. Create splits for the selected 600
        train_count = 420 # 70%
        val_count = 90    # 15%
        test_count = 90   # 15%

        train_entries = selected_entries[:train_count]
        val_entries = selected_entries[train_count:train_count + val_count]
        test_entries = selected_entries[train_count + val_count:]

        # 5. Prepare destination folders
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(UNUSED_DIR, cls), exist_ok=True)

        # Helper to safely move a file
        def move_file(filename, current_path, target_dir):
            target_path = os.path.join(target_dir, filename)
            # If already at destination, do nothing
            if os.path.abspath(current_path) == os.path.abspath(target_path):
                return
            
            # If destination already has a file with the same name, remove it first to avoid conflicts
            if os.path.exists(target_path):
                os.remove(target_path)
                
            shutil.move(current_path, target_path)

        # Move to Train
        for filename, path in train_entries:
            move_file(filename, path, os.path.join(TRAIN_DIR, cls))

        # Move to Val
        for filename, path in val_entries:
            move_file(filename, path, os.path.join(VAL_DIR, cls))

        # Move to Test
        for filename, path in test_entries:
            move_file(filename, path, os.path.join(TEST_DIR, cls))

        # Move to Unused
        for filename, path in unused_entries:
            move_file(filename, path, os.path.join(UNUSED_DIR, cls))

        summary_counts[cls] = {
            'train': len(train_entries),
            'val': len(val_entries),
            'test': len(test_entries),
            'unused': len(unused_entries),
            'total_active': len(selected_entries)
        }

    print("\n" + "="*50)
    print("Dataset Balancing and Organization Completed Successfully!")
    print("="*50)
    for cls, counts in summary_counts.items():
        print(f"{cls.upper():<25} | Train: {counts['train']:<4} | Val: {counts['val']:<4} | Test: {counts['test']:<4} | Unused: {counts['unused']:<5} | Active: {counts['total_active']}")
    print("="*50)

if __name__ == "__main__":
    balance_and_split()
