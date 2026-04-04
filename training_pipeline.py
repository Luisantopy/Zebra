import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
from datetime import datetime

from data_augmentation import get_train_dataset, get_eval_dataset, get_loader
from model_registry import build_model, get_model_type


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
def build_weighted_sampler(dataset, alpha=0.25):
    class_counts = Counter(dataset.targets)

    sample_weights = [
        1.0 / (class_counts[label] ** alpha)
        for label in dataset.targets
    ]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler, class_counts


def main():
    # --- Device definieren
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # --- Experiment Setup ---
    exp_dir, best_model_path, final_model_path, metrics_path = setup_experiment()
    print(f"📁 Experiment directory: {exp_dir}")

    # --- Daten laden ---
    train_dataset = get_train_dataset("data/train")
    val_dataset   = get_eval_dataset("data/val")
    test_dataset  = get_eval_dataset("data/test")
    print("Classes:", train_dataset.classes)
    print("Class mapping:", train_dataset.class_to_idx)

    # --- Weighted sampler für Trainingsdaten ---
    alpha = 0.30
    train_sampler, class_counts = build_weighted_sampler(train_dataset, alpha=alpha)
    print("Train class counts:", class_counts)

    # --- DataLoader ---
    train_loader = get_loader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = get_loader(val_dataset, batch_size=32)
    test_loader = get_loader(test_dataset, batch_size=32)

    # --- Model ---
    model_name = "binary_bce_simple"  # hier Modellname austauschen für anderes Modell aus registry, zB "cross_entropy" oder "binary_bce"
    model_type = get_model_type(model_name)

    if model_type == "binary":
        negatives = class_counts[0]
        positives = class_counts[1]
        pos_weight = negatives / positives

        model = build_model(model_name, num_classes=len(train_dataset.classes), pos_weight=pos_weight)

    elif model_type == "multiclass":
        total = sum(class_counts.values())
        class_weights = [total / class_counts[i] for i in range(len(train_dataset.classes))]

        model = build_model(model_name, num_classes=len(train_dataset.classes), class_weights=class_weights)

    model = model.to(device)

    # Lernrate anpassen
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # aktuell nicht aktiv
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    num_epochs = 5

    # --- Config speichern ---
    with open(exp_dir / "config.txt", "w") as f:
        f.write(f"model_name={model_name}\n")
        f.write(f"epochs={num_epochs}\n")
        f.write(f"lr={lr}\n")
        f.write("batch_size=32\n")
        f.write(f"classes={train_dataset.classes}\n")
        f.write(f"class_counts={dict(class_counts)}\n")
        f.write(f"sampler_alpha={alpha}\n")

    # --- Metrics speichern
    with open(metrics_path, "w") as f:
        f.write(
            "epoch,train_loss,train_acc,train_recall,train_precision,train_f1,"
            "val_loss,val_acc,val_recall,val_precision,val_f1,best_model\n"
        )
        
    # --- Training ---    
    best_val_f1 = -1 
    for epoch in range(num_epochs):
        train_loss, train_acc, train_recall, train_precision, train_f1 = run_epoch(
            model, train_loader, device, optimizer=optimizer
        )
        val_loss, val_acc, val_recall, val_precision, val_f1 = run_epoch(
            model, val_loader, device
        )

        #scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print("✅ Best model saved")

        with open(metrics_path, "a") as f:
            f.write(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Train Recall: {train_recall:.4f} | Train Precision: {train_precision:.4f} | Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val Recall: {val_recall:.4f} | Val Precision: {val_precision:.4f} | Val F1: {val_f1:.4f}"
            )

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Train Recall: {train_recall:.4f} | Train Precision: {train_precision:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Recall: {val_recall:.4f} | Val Precision: {val_precision:.4f} | Val F1: {val_f1:.4f}"
        )

    torch.save(model.state_dict(), final_model_path)
    print(f"📦 Final model saved: {final_model_path}")

    # --- Evaluierung des besten Modells ---
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("✅ Best model loaded for test evaluation")

    test_loss, test_acc, test_recall, test_precision, test_f1 = run_epoch(
        model, test_loader, device
    )

    print(
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.4f} | "
        f"Test Recall: {test_recall:.4f} | "
        f"Test Precision: {test_precision:.4f} | "
        f"Test F1: {test_f1:.4f}"
    )
    with open(metrics_path, "a") as f:
        f.write(
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Test Recall: {test_recall:.4f} | "
            f"Test Precision: {test_precision:.4f} | "
            f"Test F1: {test_f1:.4f}"
        )
        


if __name__ == "__main__":
    main()