# Emotion Recognition Project

GitHub repository:

```text
https://github.com/James-5757/Live-Emotion-Recognition
```

This project contains three emotion-recognition implementations, ordered as follows:

1. **Basic CNN model** (`data check+ basic CNN/`) - run locally
2. **Pretrained ResNet18 model** (`resnetdipmy.ipynb`) - run on Kaggle
3. **Custom ResNet-style model** (`resnet.ipynb`) - run on Kaggle

The models are trained and evaluated on the RAF-DB emotion dataset with 7 classes: surprise, fear, disgust, happy, sad, angry, and neutral.

---

## Dataset

This project uses two emotion recognition datasets.

For submission and review, both datasets are also included directly in this GitHub repository:

```text
DATASET/
FER/
```

1. **RAF-DB / RAF-DB style dataset**  
   This dataset is mainly used for the ResNet-based models, including the pretrained ResNet18 model and the custom ResNet-style model. These notebooks were run on Kaggle with GPU enabled.
   Link: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset

   Kaggle input path used in this project:

   ```text
   /kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET
   ```

2. **Emotion Detection Dataset / FER-style dataset**  
   This dataset is mainly used for the Basic CNN model and local testing. The Basic CNN code is designed to run locally after the dataset path is correctly configured.
   Link: https://www.kaggle.com/datasets/jayeshrohansingh/emotion-detection-dataset/data

Please make sure the dataset paths are updated correctly before running each notebook or script.


---

## 1. Run the Basic CNN locally

Folder:

```text
data check+ basic CNN/
```

Recommended project structure:

```text
project/
├── archive/
│   ├── train/
│   └── test/
├── data check+ basic CNN/
│   ├── 01_check_dataset.py
│   ├── 02_train_model_pytorch.py
│   ├── 03_video_emotion_recognition_pytorch.py
│   ├── emotion_common.py
│   └── project_config.py
```

Install dependencies:

```bash
pip install torch torchvision opencv-python scikit-learn matplotlib numpy
```

Check the dataset:

```bash
cd "data check+ basic CNN"
python 01_check_dataset.py
```

Train the Basic CNN:

```bash
python 02_train_model_pytorch.py
```

Run webcam emotion recognition:

```bash
python 03_video_emotion_recognition_pytorch.py --source 0 --mirror
```

Run on a video file:

```bash
python 03_video_emotion_recognition_pytorch.py --source path/to/video.mp4 --save-output results/output.mp4
```

Main outputs:

```text
models/best_emotion_cnn_pytorch.pth
models/final_emotion_cnn_pytorch.pth
results/
```

---

## 2. Run the Pretrained ResNet18 notebook on Kaggle

This part is the pretrained ResNet18 implementation. It uses transfer learning with an ImageNet-pretrained ResNet18 backbone and fine-tunes the final classifier for 7-class facial emotion recognition on the RAF-DB dataset.

Open:

```text
Pretrained ResNet18.ipynb
```

Dataset input:

```text
/kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET
```


Steps:

1. Upload `Pretrained ResNet18.ipynb` to Kaggle.
2. Add the RAF-DB dataset as notebook input.
3. Make sure the dataset path points to:

```python
RAF_BASIC_DIR = "/kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET"
```

4. Enable GPU in Kaggle notebook settings.
5. Run all cells.

The notebook loads the RAF-DB training and test folders, applies preprocessing and data augmentation, trains the pretrained ResNet18 model, and evaluates the best checkpoint on the independent test set.

Main outputs:

```text
best_resnet18_emotion.pth
final_resnet18_emotion.pth
```

### Note on Running Environment

This notebook was tested in Kaggle and runs normally there with GPU enabled. The model training, validation, and test evaluation complete successfully in the Kaggle environment.

Some warnings may appear when running the same notebook locally in VS Code or Jupyter, especially messages related to PyTorch DataLoader multiprocessing or tqdm widget rendering. These are environment-specific issues and do not affect the correctness of the model or the Kaggle results. For local runs, setting `num_workers=0` is recommended.

---

## 3. Run the Custom ResNet-style notebook on Kaggle

Open:

```text
custom_resnet.ipynb
```

Steps:

1. Upload the notebook to Kaggle.
2. Add the RAF-DB dataset as notebook input.
3. Enable GPU.
4. Check the dataset path in the notebook:

```python
RAF_BASIC_DIR = "/kaggle/input/datasets/shuvoalok/raf-db-dataset/DATASET"
```

If your Kaggle dataset path is different, update this line.

5. Run all cells.

Main outputs:

```text
best_rafdb_resnet_model.keras
final_rafdb_resnet_model.keras
```
We will use best_rafdb_resnet_model.keras model as our final deployment to do the test of live emotion judgement.

Notice: this model is trained by our framework not downloaded from the internet!

---

---

## 4. Demo

A real-time emotion recognition demo is also provided in the `demo/` folder.

The demo uses a trained ResNet-style model to recognise facial expressions from a webcam. It loads the saved model file:

```text
best_rafdb_resnet_model.keras
```

```text
demo/
├── best_rafdb_resnet_model.keras
├── renet.ipynb
└── models/
    └── opencv_face_detector.prototxt
```

---

## Notes

- The Basic CNN code is designed to run locally.
- The pretrained ResNet18 and custom ResNet-style notebooks are designed to run on Kaggle with GPU enabled.
- We recommend using the pretrained ResNet18 model for the final deployment.
