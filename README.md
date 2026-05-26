# Deepfake Classification (CNN + KNN)

This repo contains two reference pipelines for multi-class deepfake image classification:

- **CNN (PyTorch)**: [cnn_deepfake-classification.py](cnn_deepfake-classification.py) (also in the notebook [cnn_deepfake-classification.ipynb](cnn_deepfake-classification.ipynb))
- **KNN (scikit-learn)**: [knn_deepfake-classification.py](knn_deepfake-classification.py)

The dataset used by both scripts is stored under [deepfake-classification/](deepfake-classification/).

## Dataset layout

Expected structure:

- [deepfake-classification/train.csv](deepfake-classification/train.csv) (columns: `image_id`, `label`)
- [deepfake-classification/validation.csv](deepfake-classification/validation.csv) (columns: `image_id`, `label`)
- [deepfake-classification/test.csv](deepfake-classification/test.csv) (column: `image_id`)
- [deepfake-classification/train/](deepfake-classification/train/) (`<image_id>.png`)
- [deepfake-classification/validation/](deepfake-classification/validation/) (`<image_id>.png`)
- [deepfake-classification/test/](deepfake-classification/test/) (`<image_id>.png`)

Label set (from the CSVs): **5 classes**: `0..4`.

## Quickstart

### 1) Python environment

Activate the provided environment (so dependencies like `cv2` are available):

- `source myenvironment/bin/activate`

Run the scripts:

- KNN:
  - `python knn_deepfake-classification.py`
- CNN (WARNING: slow):
  - `python cnn_deepfake-classification.py`


If you prefer creating a clean venv instead:

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -U pip`
- `pip install numpy pandas matplotlib seaborn pillow scikit-learn opencv-python torch torchvision`

### 2) Outputs

Both scripts produce a submission-style CSV with predicted labels:

- CNN: [deepfake-classification/cnn_prediction.csv](deepfake-classification/cnn_prediction.csv)
- KNN: [deepfake-classification/knn_prediction.csv](deepfake-classification/knn_prediction.csv)

## CNN pipeline (PyTorch)

Implemented in [cnn_deepfake-classification.py](cnn_deepfake-classification.py) and mirrored in the notebook [cnn_deepfake-classification.ipynb](cnn_deepfake-classification.ipynb).

### Pre-processing

For each image:

1. Read with OpenCV (`cv2.imread`) in **BGR**.
2. Convert to **HSV** (`cv2.cvtColor(..., cv2.COLOR_BGR2HSV)`).
3. Resize to **100×100** (`cv2.resize`).
4. Normalize pixel values to **[0, 1]** using min-max normalization (`cv2.normalize(..., 0, 1, cv2.NORM_MINMAX)`).
5. Convert to a PyTorch tensor and reorder to **CHW** (`permute(2, 0, 1)`).

### Data augmentation

The training set is doubled by creating one augmented copy per training image.

Augmentations (via PIL):

- Horizontal flip
- Vertical flip
- Rotation by 40 degrees
- Brightness × 1.185
- Color × 0.8
- Sharpness × 0.968

Augmented images are then normalized and converted to tensors exactly like the originals.

### Model architecture

`NeuralNet` is a custom CNN with:

- 4 convolutional blocks with **3×3 kernels**
- Batch normalization after each convolution
- ReLU activations
- MaxPool2d(2,2) after each block (downsampling)
- Global average pooling (`AdaptiveAvgPool2d(1)`) → flatten
- Dropout(0.5)
- Final linear layer for class logits

### Training

- Batch size: 32
- Loss: `CrossEntropyLoss`
- Optimizer: `SGD(lr=0.001, momentum=0.82, weight_decay=1e-4, nesterov=True)`
- LR schedule: `CyclicLR(base_lr=0.001, max_lr=0.01, step_size_up=2000)`
- Epochs: 186
- Device: CUDA if available, else CPU

During training the script logs per-epoch loss and accuracy and plots them at the end.

### Evaluation (validation)

Validation runs with `model.eval()` and `torch.no_grad()` and reports:

- **Accuracy**
- **Confusion matrix** (+ seaborn heatmap)

Last saved notebook output (from [cnn_deepfake-classification.ipynb](cnn_deepfake-classification.ipynb)):

- Validation Accuracy: **90.32%**
- Confusion Matrix:

```text
[[228   4   6   2  10]
 [  0 228   2   2  18]
 [  3   2 232   0  13]
 [  0   1   1 248   0]
 [ 16  31  10   0 193]]
```

### Inference (test)

The test set uses the same preprocessing (BGR → HSV → resize → normalize → tensor). Predictions are computed by `argmax` over the class logits and written to [deepfake-classification/cnn_prediction.csv](deepfake-classification/cnn_prediction.csv).

## KNN pipeline (HSV histogram + KNeighborsClassifier)

Implemented in [knn_deepfake-classification.py](knn_deepfake-classification.py).

### Feature extraction (pre-processing)

Instead of using raw pixels, KNN uses a compact color descriptor:

1. Read image with OpenCV.
2. Convert to HSV.
3. Resize to 100×100.
4. Compute a **3D HSV histogram** with 8 bins per channel: `8×8×8 = 512` features.
5. Normalize and flatten the histogram to a 512-D vector.

### Data augmentation

The same augmentation function is applied to each training image to produce one additional sample.

In KNN, the augmentation is applied in RGB and then the augmented image is converted to HSV before computing the histogram.

### Training

- Classifier: `KNeighborsClassifier`
- Hyperparameters used:
  - `n_neighbors=5`
  - `metric='manhattan'`
  - `weights='distance'`

The script reports training accuracy (on the training set itself) and then evaluates on the validation set.

### Evaluation (validation)

Reproduced by running:

- `MPLBACKEND=Agg ./myenvironment/bin/python knn_deepfake-classification.py`

Output:

- Training Accuracy: **100.00%**
- Validation Accuracy: **72.88%**
- Confusion Matrix:

```text
[[177  13  14   4  42]
 [  4 179   8  12  47]
 [  6  22 183  19  20]
 [  1  10   2 229   8]
 [ 20  70   9   8 143]]
```

### Inference (test)

The script extracts histogram features for the test set and writes predictions to [deepfake-classification/knn_prediction.csv](deepfake-classification/knn_prediction.csv).

## Notes / limitations

- **Hard-coded paths**: both scripts currently use absolute paths (e.g. `/home/davide/Documents/...`). If you clone this repo elsewhere, update `folder_dir`, `img_dir`, `val_dir`, `test_dir` at the top of each script.
- **Class count**: the dataset labels are `0..4` (5 classes). If you modify the CNN architecture, ensure the final layer output matches the number of classes.
- **Reproducibility**: no explicit random seeds are set, and training uses shuffling; expect small run-to-run differences.

## Reference report

The original project write-up is in [Documentation.pdf](Documentation.pdf) and [Documentation.docx](Documentation.docx).