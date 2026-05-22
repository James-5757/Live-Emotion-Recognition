import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from emotion_common import (
    EmotionCNN,
    apply_clahe,
    get_device,
)

from project_config import (
    CLASSES,
    CLASS_FOLDER_NAMES,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    MODEL_DIR,
    RESULT_DIR,
    TEST_DIR,
    TRAIN_DIR,
)


"""
Emotion recognition training pipeline:

1. Dataset checking is handled separately in 01_check_dataset.py.
2. Image path collection and label encoding.
3. Stratified train / validation split using archive/train.
4. Independent test-set evaluation using archive/test.
5. Optional CLAHE preprocessing experiment.
6. Reduced-strength training data augmentation.
7. CNN-based emotion classifier construction.
8. Softened class-weighted loss for imbalanced data.
9. Model training with validation monitoring.
10. Learning rate scheduling.
11. Best model checkpoint saving.
12. Test-set evaluation.
13. Performance visualization and report generation.
14. CLAHE vs non-CLAHE comparison summary.
"""


BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
RANDOM_STATE = 42

MIN_CLASS_WEIGHT = 0.5
MAX_CLASS_WEIGHT = 3.0

RUN_CLAHE_COMPARISON = True

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

device = get_device()
print("Using device:", device)


def find_class_dir(split_dir, class_name):
    for folder_name in CLASS_FOLDER_NAMES[class_name]:
        class_dir = split_dir / folder_name
        if class_dir.exists():
            return class_dir
    return split_dir / class_name


def collect_image_paths(split_dir):
    image_paths = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        class_dir = find_class_dir(split_dir, class_name)

        if not class_dir.exists():
            print("Folder not found:", class_dir)
            continue

        files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"{class_name}: {len(files)} files")

        for file_path in files:
            image_paths.append(str(file_path))
            labels.append(label)

    return np.array(image_paths), np.array(labels)


class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False, use_clahe=True):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.image_paths)

    def random_augment(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        if random.random() < 0.4:
            angle = random.uniform(-10, 10)
            h, w = img.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(
                img,
                matrix,
                (w, h),
                borderMode=cv2.BORDER_REFLECT,
            )

        if random.random() < 0.3:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-5, 5)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if random.random() < 0.2:
            h, w = img.shape
            tx = random.randint(-2, 2)
            ty = random.randint(-2, 2)
            matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(
                img,
                matrix,
                (w, h),
                borderMode=cv2.BORDER_REFLECT,
            )

        if random.random() < 0.1:
            noise = np.random.normal(0, 3, img.shape).astype(np.float32)
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = int(self.labels[index])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if self.use_clahe:
            img = apply_clahe(img)

        if self.augment:
            img = self.random_augment(img)

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        img_tensor = torch.tensor(img, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


def calculate_class_weights(labels):
    class_counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    total = len(labels)

    safe_class_counts = np.maximum(class_counts, 1.0)

    raw_weights = total / (len(CLASSES) * safe_class_counts)
    softened_weights = np.sqrt(raw_weights)
    clipped_weights = np.clip(
        softened_weights,
        MIN_CLASS_WEIGHT,
        MAX_CLASS_WEIGHT,
    )

    weights = torch.tensor(clipped_weights, dtype=torch.float32)

    print("\nClass counts:")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: {int(class_counts[i])}")

    print("\nRaw class weights:")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: {raw_weights[i]:.4f}")

    print("\nSoftened and clipped class weights:")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: {weights[i].item():.4f}")

    return weights


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


def plot_training_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    experiment_name,
):
    plt.figure()
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy ({experiment_name})")
    plt.legend()
    plt.savefig(RESULT_DIR / f"accuracy_curve_{experiment_name}.png")
    plt.show()

    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss ({experiment_name})")
    plt.legend()
    plt.savefig(RESULT_DIR / f"loss_curve_{experiment_name}.png")
    plt.show()


def plot_confusion_matrix(cm, experiment_name):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix ({experiment_name})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(len(CLASSES)), CLASSES, rotation=45)
    plt.yticks(np.arange(len(CLASSES)), CLASSES)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"confusion_matrix_{experiment_name}.png")
    plt.show()


def create_dataloaders(
    train_paths,
    val_paths,
    test_paths,
    train_labels,
    val_labels,
    test_labels,
    use_clahe,
):
    train_dataset = EmotionDataset(
        train_paths,
        train_labels,
        augment=True,
        use_clahe=use_clahe,
    )

    val_dataset = EmotionDataset(
        val_paths,
        val_labels,
        augment=False,
        use_clahe=use_clahe,
    )

    test_dataset = EmotionDataset(
        test_paths,
        test_labels,
        augment=False,
        use_clahe=use_clahe,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def run_experiment(
    experiment_name,
    use_clahe,
    train_paths,
    val_paths,
    test_paths,
    train_labels,
    val_labels,
    test_labels,
):
    print("\n" + "=" * 70)
    print("Experiment:", experiment_name)
    print("Use CLAHE:", use_clahe)
    print("=" * 70)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths,
        val_paths,
        test_paths,
        train_labels,
        val_labels,
        test_labels,
        use_clahe,
    )

    class_weights = calculate_class_weights(train_labels).to(device)

    model = EmotionCNN(num_classes=len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")

    best_model_path = MODEL_DIR / f"best_emotion_cnn_pytorch_{experiment_name}.pth"
    final_model_path = MODEL_DIR / f"final_emotion_cnn_pytorch_{experiment_name}.pth"

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("\nStart training...")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "img_size": IMG_SIZE,
                    "use_clahe": use_clahe,
                    "experiment_name": experiment_name,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                },
                best_model_path,
            )

            print("Best model saved.")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "img_size": IMG_SIZE,
            "use_clahe": use_clahe,
            "experiment_name": experiment_name,
            "final_val_acc": val_accs[-1],
            "final_val_loss": val_losses[-1],
        },
        final_model_path,
    )

    print("\nFinal model saved to:", final_model_path)
    print("Best model saved to:", best_model_path)

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        experiment_name,
    )

    print("\nTesting best model...")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, true_labels, pred_labels = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\nTest Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=CLASSES,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    with open(RESULT_DIR / f"classification_report_{experiment_name}.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=np.arange(len(CLASSES)),
    )

    plot_confusion_matrix(cm, experiment_name)

    result = {
        "experiment_name": experiment_name,
        "use_clahe": use_clahe,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }

    return result


def write_experiment_summary(results):
    summary_path = RESULT_DIR / "experiment_comparison_summary.txt"

    lines = []
    lines.append("Experiment Comparison Summary")
    lines.append("=" * 40)
    lines.append("")

    for result in results:
        lines.append(f"Experiment: {result['experiment_name']}")
        lines.append(f"Use CLAHE: {result['use_clahe']}")
        lines.append(f"Best Validation Accuracy: {result['best_val_acc']:.4f}")
        lines.append(f"Best Validation Loss: {result['best_val_loss']:.4f}")
        lines.append(f"Test Accuracy: {result['test_acc']:.4f}")
        lines.append(f"Test Loss: {result['test_loss']:.4f}")
        lines.append(f"Best Model Path: {result['best_model_path']}")
        lines.append("")

    best_result = max(results, key=lambda x: x["best_val_acc"])

    lines.append("Selected Best Experiment")
    lines.append("-" * 40)
    lines.append(f"Experiment: {best_result['experiment_name']}")
    lines.append(f"Use CLAHE: {best_result['use_clahe']}")
    lines.append(f"Best Validation Accuracy: {best_result['best_val_acc']:.4f}")
    lines.append(f"Test Accuracy: {best_result['test_acc']:.4f}")
    lines.append(f"Best Model Path: {best_result['best_model_path']}")
    lines.append("")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print("\nExperiment summary saved to:", summary_path)

    return best_result


def save_default_best_model(best_result):
    selected_checkpoint = torch.load(
        best_result["best_model_path"],
        map_location=device,
    )

    default_best_path = MODEL_DIR / "best_emotion_cnn_pytorch.pth"

    torch.save(
        selected_checkpoint,
        default_best_path,
    )

    print("\nDefault best model updated:")
    print(default_best_path)
    print("Selected experiment:", best_result["experiment_name"])


def main():
    print("\nCollecting training images from:", TRAIN_DIR)
    train_all_paths, train_all_labels = collect_image_paths(TRAIN_DIR)

    print("\nCollecting test images from:", TEST_DIR)
    test_paths, test_labels = collect_image_paths(TEST_DIR)

    print("\nTotal training/validation images:", len(train_all_paths))
    print("Total test images:", len(test_paths))

    if len(train_all_paths) == 0 or len(test_paths) == 0:
        print("No images found.")
        return

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_all_paths,
        train_all_labels,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=train_all_labels,
    )

    print("\nDataset split:")
    print("Train:", len(train_paths))
    print("Validation:", len(val_paths))
    print("Test:", len(test_paths))

    if RUN_CLAHE_COMPARISON:
        experiments = [
            {
                "experiment_name": "with_clahe",
                "use_clahe": True,
            },
            {
                "experiment_name": "without_clahe",
                "use_clahe": False,
            },
        ]
    else:
        experiments = [
            {
                "experiment_name": "with_clahe",
                "use_clahe": True,
            }
        ]

    results = []

    for experiment in experiments:
        result = run_experiment(
            experiment_name=experiment["experiment_name"],
            use_clahe=experiment["use_clahe"],
            train_paths=train_paths,
            val_paths=val_paths,
            test_paths=test_paths,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
        )

        results.append(result)

    best_result = write_experiment_summary(results)
    save_default_best_model(best_result)

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
