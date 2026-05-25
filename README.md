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

## Repository Structure

```text
torch_training/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── trained_models/
│   └── exp_*/
│
├── torch_training/
│   ├── training_pipeline.py
│   ├── predict_single.py
│   ├── data_augmentation.py
│   ├── model_registry.py
│   ├── torch_cnn.py
│   ├── torch_cnn_simple.py
│   └── helpers.py
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

---

## Fine-Tuning Pipeline

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

Example:

```bash
uv run python -m torch_training.training_pipeline
```

---

# Single Image Prediction

Example:

```bash
uv run python -m torch_training.predict_single path/to/image.png \
  --weights trained_models/best_model.pth \
  --model resnet50_cross_entropy \
  --threshold 0.33
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

* Vision Transformers
* segmentation models
* focal loss
* hard negative mining
* test-time augmentation
* ensemble methods
* uncertainty estimation

---

# Author

Created by Luisa Plasczymonka as part of a deep learning experimentation 
project on aerial image classification and robust zebra crossing detection.
