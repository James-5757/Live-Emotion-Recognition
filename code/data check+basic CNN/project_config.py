from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "archive"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_FOLDER_NAMES = {
    "Angry": ["Angry", "angry"],
    "Disgust": ["Disgust", "disgusted", "Disgusted"],
    "Fear": ["Fear", "fearful", "Fearful"],
    "Happy": ["Happy", "happy"],
    "Neutral": ["Neutral", "neutral"],
    "Sad": ["Sad", "sad"],
    "Surprise": ["Surprise", "Surprised", "surprise", "surprised"],
}
IMG_SIZE = 48
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".pgm"]
