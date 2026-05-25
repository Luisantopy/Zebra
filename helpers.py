import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchvision.utils import draw_bounding_boxes, draw_keypoints, draw_segmentation_masks
from torch.utils.data import WeightedRandomSampler
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.functional import to_pil_image

from collections import Counter
from pathlib import Path
from datetime import datetime
import random
import numpy as np
from sklearn.metrics import precision_recall_curve

# --- image plot helper from https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py
def plot(imgs, row_title=None, bbox_width=3, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            points = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target

                    # Conversion necessary because draw_bounding_boxes() only
                    # work with this specific format.
                    if tv_tensors.is_rotated_bounding_format(boxes.format):
                        boxes = v2.ConvertBoundingBoxFormat("xyxyxyxy")(boxes)
                elif isinstance(target, tv_tensors.KeyPoints):
                    points = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=bbox_width)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)
            if points is not None:
                img = draw_keypoints(img, points, colors="red", radius=10)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def plot_probability_histogram(labels, probs, output_path):
    labels = np.asarray(labels).ravel()
    probs = np.asarray(probs).ravel()

    plt.figure(figsize=(8, 5))

    plt.hist(
        probs[labels == 0],
        bins=50,
        alpha=0.5,
        label="negative",
    )

    plt.hist(
        probs[labels == 1],
        bins=50,
        alpha=0.5,
        label="positive",
    )

    plt.xlabel("Predicted probability for class 1")
    plt.ylabel("Count")
    plt.title("Validation probability distributions")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.show()


def plot_precision_recall_curve(labels, probs, output_path):
    labels = np.asarray(labels).ravel()
    probs = np.asarray(probs).ravel()

    precision, recall, _ = precision_recall_curve(labels, probs)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.show()


def plot_false_negatives(
    model,
    dataset,
    loader,
    device,
    threshold,
    output_dir,
    save_path,
    max_images=25,
    save_hard_positives=False,
):
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    false_negatives = []

    sample_idx = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            

            # multiclass cross_entropy
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            for i in range(len(images)):
                label = labels[i].item()
                pred = preds[i].item()

                # False Negative
                if label == 1 and pred == 0:
                    
                    false_negatives.append(
                        {
                            "image": images[i].cpu(),
                            "prob": probs[i].item(),
                            "idx": sample_idx,
                            "path": dataset.samples[sample_idx][0],
                        }
                    )

                sample_idx += 1

    # -------------------------------------------------
    # Hard positives speichern
    # -------------------------------------------------
    if save_hard_positives:
        hard_positive_path = output_dir / "hard_positives.txt"

        with open(hard_positive_path, "w") as f:
            for item in false_negatives:

                # nur wirklich harte FN speichern
                if item["prob"] < 0.05:
                    f.write(item["path"] + "\n")

        print(f"Saved hard positives to: {hard_positive_path}")
    
    print(f"Found {len(false_negatives)} false negatives")

    # -------------------------------------------------
    # Plotten
    # -------------------------------------------------

    num_images = min(max_images, len(false_negatives))

    cols = 5
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

    if rows == 1:
        axes = [axes]

    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flatten():
        ax.axis("off")

    # -------------------------------------------------
    # Bilder plotten
    # -------------------------------------------------

    for idx in range(num_images):
        item = false_negatives[idx]

        image = item["image"]
        prob = item["prob"]

        # Denormalisieren
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        image = image * std + mean
        image = torch.clamp(image, 0, 1)

        image = to_pil_image(image)

        ax = axes[idx // cols, idx % cols]

        ax.imshow(image)
        ax.set_title(f"p(y)={prob:.3f}")
        ax.axis("off")

    # -------------------------------------------------
    # Speichern + Anzeigen
    # -------------------------------------------------

    plt.tight_layout()

    plt.savefig(save_path, dpi=200)

    plt.show()

    plt.close()

    print(f"Saved to: {save_path}")
    


# === Helper für Training Pipeline ===
# --- Training + Validation ---
def run_epoch(model, loader, device, optimizer=None):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # für Konfusionsmatrix
    true_positive = 0
    false_negative = 0
    false_positive = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = model.compute_loss(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            preds = model.predict(outputs)

            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            positive_label = getattr(model, "positive_label", 1)

            true_positive += ((preds == positive_label) & (labels == positive_label)).sum().item()
            false_negative += ((preds != positive_label) & (labels == positive_label)).sum().item()
            false_positive += ((preds == positive_label) & (labels != positive_label)).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return epoch_loss, epoch_acc, recall, precision, f1


# --- Experiment Umgebung erstellen ---
def setup_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("trained_models") / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = exp_dir / "best_model.pth"
    final_model_path = exp_dir / "final_model.pth"
    metrics_path = exp_dir / "metrics.txt"

    return exp_dir, best_model_path, final_model_path, metrics_path


# --- Weighted Random Sampling ---
def build_weighted_sampler(
    dataset,
    seed,
    alpha=0.25,
    hard_positive_paths=None,
    hard_positive_weight=3.0,
):    
    """
    Erstellt einen WeightedRandomSampler für unausgeglichene Datensätze.

    Die Samplinggewichte basieren auf der Klassenhäufigkeit:
    seltene Klassen erhalten höhere Gewichte, häufige Klassen
    niedrigere Gewichte.

    Zusätzlich können bestimmte schwierige Positivbeispiele
    ("Hard Positives") gezielt stärker gewichtet werden,
    um deren Samplingwahrscheinlichkeit während des Trainings
    zu erhöhen.

    Parameter
    ----------
    dataset :
        Datensatz mit:
        - dataset.targets
        - dataset.samples

    seed : int
        Zufallsseed für reproduzierbares Sampling.

    alpha : float, optional
        Stärke der Klassengewichtung.

        Gewicht = 1 / (class_count ** alpha)

        alpha = 0:
            keine Gewichtung

        alpha = 1:
            vollständige inverse Klassenhäufigkeit

        Typischer Bereich:
            0.2 - 0.6

    hard_positive_paths : set[str] | None, optional
        Menge von Bildpfaden schwieriger Positivbeispiele,
        die stärker gewichtet werden sollen.

    hard_positive_weight : float, optional
        Multiplikativer Zusatzfaktor für Hard Positives.

        Beispiel:
            weight *= hard_positive_weight

    Returns
    -------
    sampler : WeightedRandomSampler
        Sampler für den DataLoader.

    class_counts : Counter
        Anzahl der Samples pro Klasse.
    """
    if hard_positive_paths is None:
        hard_positive_paths = set()

    class_counts = Counter(dataset.targets)

    sample_weights = []

    for idx, label in enumerate(dataset.targets):

        # normaler Klassengewichtungs-Teil
        weight = 1.0 / (class_counts[label] ** alpha)

        # Pfad des Samples
        path, _ = dataset.samples[idx]

        # schwierige Positives stärker gewichten
        if path in hard_positive_paths:
            weight *= hard_positive_weight

        sample_weights.append(weight)

    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed)
    )

    return sampler, class_counts


# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.0, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.stop = False

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return True

        if self.mode == "max":
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False
        

# --- Threshold helpers ---
@torch.no_grad()
def evaluate_thresholds(model, loader, device, thresholds=None, positive_label=1):
    if thresholds is None:
        thresholds = [round(x, 2) for x in torch.linspace(0.3, 0.6, steps=10).tolist()]

    model.eval()
    results = []

    for threshold in thresholds:
        correct = 0
        total = 0

        true_positive = 0
        false_negative = 0
        false_positive = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Wahrscheinlichkeit für Klasse 1 = "y"
            probs = torch.softmax(outputs, dim=1)[:, positive_label]

            preds = (probs >= threshold).long()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            true_positive += ((preds == positive_label) & (labels == positive_label)).sum().item()
            false_negative += ((preds != positive_label) & (labels == positive_label)).sum().item()
            false_positive += ((preds == positive_label) & (labels != positive_label)).sum().item()

        acc = correct / total if total > 0 else 0.0

        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results.append({
            "threshold": threshold,
            "acc": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        })

    return results


@torch.no_grad()
def evaluate_with_threshold(model, loader, device, threshold=0.5, positive_label=1):
    model.eval()

    correct = 0
    total = 0

    true_positive = 0
    false_negative = 0
    false_positive = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, positive_label]
        preds = (probs >= threshold).long()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        true_positive += ((preds == positive_label) & (labels == positive_label)).sum().item()
        false_negative += ((preds != positive_label) & (labels == positive_label)).sum().item()
        false_positive += ((preds == positive_label) & (labels != positive_label)).sum().item()

    acc = correct / total if total > 0 else 0.0

    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return acc, recall, precision, f1


def select_best_threshold(
    results,
    min_recall=0.2,
    #min_precision=0.8,
    metric="precision"
):
    """
    Wählt besten Threshold unter Nebenbedingung:
    recall >= min_recall

    metric:
        - "precision"
        - "f1"
    """

    # nur gültige Thresholds
    valid = [r for r in results if r["recall"] >= min_recall]

    if len(valid) == 0:
        print("⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1")
        return max(results, key=lambda x: x["f1"])

    if metric == "precision":
        best = max(valid, key=lambda x: x["precision"])
    elif metric == "f1":
        best = max(valid, key=lambda x: x["f1"])
    elif metric == "recall":
        best = max(valid, key=lambda x: x["recall"])
    else:
        raise ValueError("metric muss 'precision' oder 'f1' sein")

    return best


# --- Seeds ---
def set_seed(seed: int = 42):
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- Evaluation helpers ---
@torch.no_grad()
def collect_labels_and_probs(model, loader, device, positive_label=1):
    model.eval()

    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # CrossEntropy-Modell: outputs = [batch_size, 2]
        probs = torch.softmax(outputs, dim=1)[:, positive_label]

        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    labels = np.concatenate(all_labels).ravel()
    probs = np.concatenate(all_probs).ravel()

    return labels, probs