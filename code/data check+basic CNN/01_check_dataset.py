from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from project_config import (
    CLASSES,
    CLASS_FOLDER_NAMES,
    DATASET_DIR,
    IMAGE_EXTENSIONS,
    TEST_DIR,
    TRAIN_DIR,
)


"""
Step 1: Dataset checking.

This script verifies the dataset folder structure, counts images in each
emotion class, checks corrupted images, and displays sample images.
"""


def find_class_dir(split_dir, class_name):
    for folder_name in CLASS_FOLDER_NAMES[class_name]:
        class_dir = split_dir / folder_name
        if class_dir.exists():
            return class_dir
    return split_dir / class_name


def get_image_files(class_dir):
    if not class_dir.exists():
        return []

    return [
        f for f in class_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():
    print("Checking dataset path...")
    print("Dataset path:", DATASET_DIR.resolve())

    if not DATASET_DIR.exists():
        print("ERROR: Dataset folder does not exist.")
        return

    print("\nClass counts:")
    total_images = 0

    for split_name, split_dir in [("Train", TRAIN_DIR), ("Test", TEST_DIR)]:
        print(f"\n{split_name} set:")

        for class_name in CLASSES:
            class_dir = find_class_dir(split_dir, class_name)
            image_files = get_image_files(class_dir)

            if not class_dir.exists():
                print(f"{class_name}: folder not found")
                continue

            print(f"{class_name}: {len(image_files)} files")
            total_images += len(image_files)

    print("\nTotal images:", total_images)

    print("\nChecking corrupted images...")
    bad_files = []

    for split_dir in [TRAIN_DIR, TEST_DIR]:
        for class_name in CLASSES:
            class_dir = find_class_dir(split_dir, class_name)

            for file_path in get_image_files(class_dir):
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

                if img is None:
                    bad_files.append(file_path)

    print("Bad images:", len(bad_files))

    if bad_files:
        print("Examples of bad files:")
        for file_path in bad_files[:10]:
            print(file_path)

    print("\nShowing sample images...")

    fig, axes = plt.subplots(len(CLASSES), 5, figsize=(10, 14))

    for row, class_name in enumerate(CLASSES):
        class_dir = find_class_dir(TRAIN_DIR, class_name)
        image_files = get_image_files(class_dir)

        for col in range(5):
            ax = axes[row, col]
            ax.axis("off")

            if col < len(image_files):
                img = cv2.imread(str(image_files[col]), cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    ax.imshow(img, cmap="gray")
                    ax.set_title(class_name, fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
