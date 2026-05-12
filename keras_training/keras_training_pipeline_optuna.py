import os
import random
import numpy as np
import tensorflow as tf
import optuna

from .keras_cnn import CNNKeras

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================
# Dataset
# =========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_train_dataset(
    data_dir,
    seed,
    aug_config,
):
    """
    Conditional augmentation:
    stärkere Augmentation für Klasse y
    """

    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )

    # -------------------------------------------------
    # Normalisierung
    # -------------------------------------------------

    normalize = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255.0)
    ])

    # -------------------------------------------------
    # Augmentation
    # -------------------------------------------------

    augmenter_y = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(aug_config["rotation"], seed=seed),
        tf.keras.layers.RandomContrast(aug_config["contrast"], seed=seed),
        tf.keras.layers.RandomZoom(aug_config["zoom"], seed=seed),
    ])

    augmenter_n = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            "horizontal",
            seed=seed,
        )
    ])

    # -------------------------------------------------
    # Conditional augmentation
    # -------------------------------------------------

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


def build_eval_dataset(
    data_dir,
):
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

    ds = ds.map(
        lambda x, y: (normalize(x), y)
    )

    return ds.prefetch(tf.data.AUTOTUNE)


def collect_labels(dataset):
    return np.concatenate([y.numpy() for _, y in dataset])


def collect_probs(model, dataset):
    logits = model.predict(dataset, verbose=0)
    probs = tf.nn.softmax(logits, axis=1)[:, 1].numpy()
    return probs

# =========================================================
# Evaluation
# =========================================================

def evaluate_with_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)

    return {
        "acc": accuracy_score(labels, preds),
        "recall": recall_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


# ========================================================
# Class Weighting 
# ========================================================

def get_class_counts(dataset):
    labels = collect_labels(dataset)
    negatives = np.sum(labels == 0)
    positives = np.sum(labels == 1)
    return negatives, positives


# ========================================================
# Threshold Optimization
# ========================================================

def find_best_threshold(labels, probs, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 41)

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


# ========================================================
# Optuna Pruning
# ========================================================

class OptunaPruningCallback(tf.keras.callbacks.Callback):
    """
    Keras Callback, der am Ende jeder Epoche die Validierungs-F1 berechnet und an Optuna meldet.
        - Berechnet die Validierungs-F1 über alle Validierungsbeispiele (nicht nur Batch-Mittelwert)
        - Findet den besten Threshold für die aktuelle Epoche
        - Meldet die F1 an Optuna
        - Prunt ggf. den Trial
    """

    def __init__(self, trial, val_ds):
        super().__init__()
        self.trial = trial
        self.val_ds = val_ds
        self.val_labels = collect_labels(val_ds)

    def on_epoch_end(self, epoch, logs=None):
        val_probs = collect_probs(self.model, self.val_ds)

        _, metrics = find_best_threshold(
            labels=self.val_labels,
            probs=val_probs,
        )

        val_f1 = metrics["f1"]

        logs = logs or {}
        logs["val_f1"] = val_f1

        print(
            f" - val_f1: {val_f1:.4f}"
            f" - val_precision: {metrics['precision']:.4f}"
            f" - val_recall: {metrics['recall']:.4f}"
            f" - best_threshold: {metrics['threshold']:.3f}"
        )

        if self.trial is not None:
            self.trial.report(val_f1, step=epoch)

            if self.trial.should_prune():
                raise optuna.TrialPruned()

# =========================================================
# Experiment
# =========================================================

def run_experiment(
    seed,
    config,
    trial=None,
):

    set_seed(seed)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    train_ds = build_train_dataset(
        "data/train",
        seed,
        aug_config=config,
    )

    val_ds = build_eval_dataset("data/val")
    test_ds = build_eval_dataset("data/test")

    negatives, positives = get_class_counts(train_ds)
    base_pos_weight = negatives / positives

    class_weight = {
        0: 1.0,
        1: float(base_pos_weight * config["class_weight_factor"]),
    }


    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = CNNKeras(
        num_classes=2,
        dropout_rate=0.0,
    )

    model.build((None, 224, 224, 3))

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=config["lr"],
        momentum=config["momentum"],
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],

    )

    # -------------------------------------------------
    # Early stopping & pruning
    # -------------------------------------------------

    callbacks = [
        OptunaPruningCallback(
            trial=trial,
            val_ds=val_ds,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=3,
            restore_best_weights=True,
        )
    ]

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # -------------------------------------------------
    # Validation predictions
    # -------------------------------------------------

    val_labels = collect_labels(val_ds)
    val_probs = collect_probs(model, val_ds)

    best_threshold, metrics = find_best_threshold(
        labels=val_labels,
        probs=val_probs,
    )

    return metrics["f1"], metrics


# =========================================================
# Optuna Objective
# =========================================================

def objective(trial):

    config = {
        # Optimizer
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "momentum": trial.suggest_float("momentum", 0.80, 0.95),

        # Threshold direkt optimieren
        # "threshold": trial.suggest_float("threshold", 0.10, 0.90),

        # Class weighting
        "class_weight_factor": trial.suggest_float("class_weight_factor", 0.1, 2.0),

        # y-Augmentation
        "rotation": trial.suggest_float("rotation", 0.00, 0.25),
        "contrast": trial.suggest_float("contrast", 0.00, 0.50),
        "zoom": trial.suggest_float("zoom", 0.00, 0.30),
    }


    seeds = [10]

    scores = []

    for seed in seeds:

        f1, metrics = run_experiment(
            seed=seed,
            config=config,
            trial=trial,
        )

        scores.append(f1)
        print("val metrics:", metrics)

    return float(np.mean(scores))


# =========================================================
# Main
# =========================================================

def main():

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=3,
        )
    )

    study.optimize(
        objective,
        n_trials=15,
    )

    print("\n==========================")
    print("BEST TRIAL")
    print("==========================")

    print("Value:")
    print(study.best_trial.value)

    print("\nParams:")
    print(study.best_trial.params)


if __name__ == "__main__":
    main()