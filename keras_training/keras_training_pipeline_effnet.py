import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from .keras_cnn import EfficientNetBinary
from helpers import plot_precision_recall_curve, plot_probability_histogram

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================
# Config
# =========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 10

CONFIG = {
    "lr": 0.00115,
    "momentum": 0.913,
    "class_weight_factor": 0.2,
    "rotation": 0.07,
    "contrast": 0.12,
    "zoom": 0.056,
}

EPOCHS = 15
MODEL_PATH = TRAINED_MODELS_DIR / "zebra_model_effnet.keras"


# =========================================================
# Dataset
# =========================================================

def build_train_dataset(data_dir, seed, aug_config):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )

    normalize = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255.0)
    ])

    augmenter_y = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(aug_config["rotation"], seed=seed),
        tf.keras.layers.RandomContrast(aug_config["contrast"], seed=seed),
        tf.keras.layers.RandomZoom(aug_config["zoom"], seed=seed),
    ])

    augmenter_n = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed)
    ])

    def augment(images, labels):
        labels_exp = tf.reshape(labels, (-1, 1, 1, 1))

        aug_y = augmenter_y(images, training=True)
        aug_n = augmenter_n(images, training=True)

        images = tf.where(
            tf.equal(labels_exp, 1),
            aug_y,
            aug_n,
        )

        images = normalize(images)

        return images, labels

    return ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def build_eval_dataset(data_dir):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    normalize = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255.0)
    ])

    ds = ds.map(lambda x, y: (normalize(x), y))

    return ds.prefetch(tf.data.AUTOTUNE)


def collect_labels(dataset):
    return np.concatenate([y.numpy() for _, y in dataset])


def collect_probs(model, dataset):
    return model.predict(dataset, verbose=0).ravel()


# =========================================================
# Evaluation
# =========================================================

def evaluate_with_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)

    return {
        "acc": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def find_best_threshold(labels, probs, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1.0

    for threshold in thresholds:
        metrics = evaluate_with_threshold(labels, probs, threshold)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics

    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


def get_class_counts(dataset):
    labels = collect_labels(dataset)
    negatives = np.sum(labels == 0)
    positives = np.sum(labels == 1)
    return negatives, positives


# =========================================================
# Validation
# =========================================================

class ValidationF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
        self.val_labels = collect_labels(val_ds)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_probs = collect_probs(self.model, self.val_ds)
        _, metrics = find_best_threshold(self.val_labels, val_probs)

        logs["val_f1"] = metrics["f1"]

        print(
            f" - val_f1: {metrics['f1']:.4f}"
            f" - val_precision: {metrics['precision']:.4f}"
            f" - val_recall: {metrics['recall']:.4f}"
            f" - best_threshold: {metrics['threshold']:.3f}"
        )

# =========================================================
# Main
# =========================================================

def main():
    set_seed(SEED)

    train_ds = build_train_dataset(
        DATA_DIR / "train",
        seed=SEED,
        aug_config=CONFIG,
    )

    val_ds = build_eval_dataset(DATA_DIR / "val")
    test_ds = build_eval_dataset(DATA_DIR / "test")

    negatives, positives = get_class_counts(train_ds)
    base_pos_weight = negatives / positives

    class_weight = {
        0: 1.0,
        1: float(base_pos_weight * CONFIG["class_weight_factor"]),
    }

    print("\nClass counts:")
    print("negatives:", negatives)
    print("positives:", positives)
    print("class_weight:", class_weight)

    model = EfficientNetBinary(dropout_rate=0.3, trainable_backbone=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        class_weight=class_weight,
        callbacks=[
            ValidationF1Callback(val_ds),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_f1",
                mode="max",
                patience=2,
                restore_best_weights=True,
            ),
        ],
    )

    model.backbone.trainable = True

    for layer in model.backbone.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weight,
        callbacks=[
            ValidationF1Callback(val_ds),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_f1",
                mode="max",
                patience=3,
                restore_best_weights=True,
            ),
        ],
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    val_labels = collect_labels(val_ds)
    val_probs = collect_probs(model, val_ds)

    best_threshold, val_metrics = find_best_threshold(
        labels=val_labels,
        probs=val_probs,
    )

    print("\n==========================")
    print("VALIDATION")
    print("==========================")
    print("best threshold:", best_threshold)
    print(val_metrics)

    plot_probability_histogram(
        labels=val_labels,
        probs=val_probs,
        output_path=TRAINED_MODELS_DIR / "validation_probability_histogram_effnet.png",
    )

    plot_precision_recall_curve(
        labels=val_labels,
        probs=val_probs,
        output_path=TRAINED_MODELS_DIR / "validation_precision_recall_curve_effnet.png",
    )

    # -------------------------------------------------
    # Test
    # -------------------------------------------------

    test_labels = collect_labels(test_ds)
    test_probs = collect_probs(model, test_ds)

    test_metrics = evaluate_with_threshold(
        labels=test_labels,
        probs=test_probs,
        threshold=best_threshold,
    )

    test_preds = (test_probs >= best_threshold).astype(int)
    cm = confusion_matrix(test_labels, test_preds)

    print("\n==========================")
    print("TEST")
    print("==========================")
    print("threshold from validation:", best_threshold)
    print(test_metrics)

    print("\nConfusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()