# Zebra Crossing Detection from Aerial Images

Deep learning project for detecting zebra crossings in aerial imagery using custom CNNs, pretrained ResNet architectures, and systematic experimentation with class imbalance handling, augmentation strategies, and transfer learning.

---

## Project Overview

The goal of this project is to classify aerial images into:

| Class | Description            |
| ----- | ---------------------- |
| `y`   | Zebra crossing present |
| `n`   | No zebra crossing      |

The project focuses on:

* binary image classification
* aerial/satellite imagery
* class imbalance handling
* transfer learning
* seed stability
* probability calibration
* false negative reduction
* model interpretability

---

## Dataset

The dataset consists of aerial image patches labelled as:

- `y`: zebra crossing present
- `n`: no zebra crossing

The dataset is highly imbalanced, with significantly fewer positive samples (`y`) than negative samples (`n`).

To address this, the project investigates:

- weighted sampling
- class-specific augmentation
- threshold tuning
- hard positive mining

---

## Repository Structure

```text
zebra/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── documentation/
│
├── keras_training/
│   ├── keras_cnn.py
│   ├── keras_training_pipeline.py
│   └── ...
│
├── trained_models/
│   ├── selected_models/
│   └── exp_*/
│
├── torch_training/
│   ├── model_registry.py
│   ├── data_augmentation.py
│   ├── torch_cnn.py
│   ├── training_pipeline.py
│   ├── predict_single.py
│   └── ...
│
├── helpers.py
│
└── README.md
```

---

# Features

## Models

Implemented and tested:

### Custom CNNs

* shallow CNN baselines
* CrossEntropy variants
* BCEWithLogits variants

#### Keras / TensorFlow Models

Additional experiments were conducted using TensorFlow/Keras to investigate 
whether the strong seed dependency observed in PyTorch was framework-specific.

Tested architectures included:

- Sequential CNN classifiers
- Conv2D + MaxPooling architectures
- Dense classification heads
- Binary sigmoid output models

Main findings:

- seed instability remained present
- performance fluctuations were similar to PyTorch
- the main issue was therefore likely dataset-related rather than framework-related

The Keras experiments helped confirm that:

- class imbalance
- oversampling dynamics
- difficult positive samples
- stochastic optimization

were the dominant challenges of the project.

### Pretrained Models

* ResNet18
* ResNet50

using:

```python
torchvision.models
```

#### ResNet Training Strategy

Pretrained ResNet models are trained in two stages:

1. Feature Extraction
   - backbone frozen
   - only classification head is trained

2. Fine Tuning
   - last residual block (`layer4`) is unfrozen
   - model is trained with a smaller learning rate

This approach preserves useful ImageNet features while adapting the model to aerial imagery.


#### Fine-Tuning Pipeline

Supports:

* frozen backbone training
* progressive unfreezing
* head-only training
* transfer learning workflows

Example:

```python
model.freeze_backbone()
model.unfreeze_last_block()
model.unfreeze_all()
```

---

## Data Augmentation

Implemented augmentations:

* horizontal flips
* vertical flips
* color jitter
* perspective distortion
* rotation
* normalization

Class-specific augmentation is supported.

---

## Class Imbalance Handling

Implemented techniques:

* `WeightedRandomSampler`
* configurable oversampling strength
* hard positive mining
* threshold tuning

---

## Experiment Tracking

Each experiment automatically stores:

* best model weights
* final model weights
* metrics
* confusion matrices
* false negative visualizations
* hard positive samples

Example:

```text
trained_models/exp_20260520_123456/
```

---

## Reproducibility

Experiments are executed with fixed random seeds.

Multiple seeds are evaluated because performance can vary substantially between runs, especially for smaller CNN architectures.

The project therefore emphasizes:

- multi-seed evaluation
- stability analysis
- reproducible experiment tracking

---

## Evaluation

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Precision-Recall Curves

Special focus is placed on reducing False Negatives, since missing a zebra crossing is considered more critical than producing a False Positive.

---

# Results

## Best Performing Model

The best results were achieved using:

* pretrained ResNet50
* transfer learning
* weighted sampling
* threshold tuning

Typical performance:

| Metric    | Value      |
| --------- | ---------- |
| Precision | ~0.95–0.97 |
| Recall    | ~0.84      |
| F1 Score  | ~0.89–0.90 |

Example confusion matrix:

```text
[[5739    9]
 [  39  213]]
```

---

# Main Findings

- Transfer learning consistently outperformed custom CNN architectures.
- ResNet50 achieved the best overall performance.
- False negatives were more difficult to eliminate than false positives.
- Aggressive augmentation did not necessarily improve recall.
- Model performance remained sensitive to random seed selection.

---

# Key Learnings

## Transfer Learning Matters

Pretrained ResNet models significantly improved:

* stability
* calibration
* recall
* robustness

compared to custom CNNs.

---

## Seed Dependency Is Real

Different random seeds produced surprisingly different results, even with identical configurations.

This project therefore evaluates:

* multiple seeds
* mean/std performance
* stability-aware optimization

instead of relying on single runs.

---

## Error Analysis Is Extremely Valuable

False negative visualization revealed that difficult samples often contained:

* shadows
* occlusions
* unusual perspectives
* partial zebra crossings

This analysis guided most later improvements.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Luisantopy/Zebra.git
cd Zebra
```

Create environment:

```bash
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

---

# Training

Different pipelines are used for different model types:

- torch CNN: `torch_training/training_pipeline.py`
- torch ResNet: `torch_training/training_pipeline_resnet.py`
- keras CNN: `keras_training/keras_training_pipeline.py`
- keras EfficientNet: `keras_training/keras_training_pipeline_effnet.py`

Example:

```bash
uv run python -m torch_training.training_pipeline
```

## Output

Each training run creates a new experiment directory in `trained_models/`.

The directory contains:

- `best_model.pth` – model with the best validation score
- `final_model.pth` – model after the final training epoch
- `config.txt` – training configuration and hyperparameters
- `metrics.txt` – training, validation and test metrics
- precision–recall curves
- probability histograms
- false negative analysis plots

Selected reference models can be found in:

```text
trained_models/selected_models/
```

## Notes

- Validation data is used for hyperparameter tuning and threshold selection.
- Test data is used only for the final evaluation.

---

# Single Image Prediction

Use a trained model to classify a single image.

Example:

```bash
uv run python -m torch_training.predict_single data/test/y/2499000_1118275.png \
  --weights trained_models/selected_models/torch_resnet50/best_model.pth \
  --model resnet50_cross_entropy \
  --threshold 0.30
```

### Arguments

| Argument | Description |
|-----------|-------------|
| `image_path` | Path to the image that should be classified. |
| `--weights` | Path to the trained model weights (`.pth`). |
| `--model` | Model architecture used during training (e.g. `cross_entropy`, `resnet18_cross_entropy`, `resnet50_cross_entropy`). |
| `--threshold` | Probability threshold for predicting the positive class (`y`). |
| `--classes` | Class names in the correct order (default: `n y`). |

### Notes

- The model architecture must match the saved weights.
- Images are automatically converted to RGB and normalized using ImageNet statistics.
- For CrossEntropy models, class probabilities are computed using Softmax.
- For binary BCE models, probabilities are computed using Sigmoid.

---

# Evaluation

Evaluate a saved PyTorch model on all images in a test directory.

`torch_training/predict_all.py` performs inference only:
- no training
- no optimizer
- no backpropagation

It loads a trained model, predicts all images in a test folder,
computes a confusion matrix and saves the confusion matrix plot
in the same folder as the model weights.

Example:

```bash
uv run python -m torch_training.predict_all \
  --data data/test \
  --weights trained_models/selected_models/torch_resnet50/best_model.pth \
  --model resnet50_cross_entropy \
  --threshold 0.19
```

---

# Technologies Used

* PyTorch
* Torchvision
* TensorFlow / Keras
* NumPy
* Matplotlib
* scikit-learn
* Optuna

---

# Future Work

Potential next steps:

* Vision Transformers (ViT)
* Focal Loss for further false negative reduction
* Hard Negative Mining
* Test-Time Augmentation
* Ensemble Learning

---

# Author

Created by Luisa Plasczymonka as part of a deep learning experimentation 
project on aerial image classification and robust zebra crossing detection.
For in-depth project discussion see `documentation/projektbericht.md`
