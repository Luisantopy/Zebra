import torch
import optuna
import numpy as np
from pathlib import Path

from .data_augmentation import get_train_dataset, get_eval_dataset, get_loader
from .model_registry import build_model, get_model_type
from helpers import (
    EarlyStopping,
    collect_labels_and_probs, 
    run_epoch, setup_experiment, build_weighted_sampler, 
    evaluate_with_threshold, evaluate_thresholds, select_best_threshold,
    set_seed, plot_precision_recall_curve, plot_probability_histogram, plot_false_negatives,
)
from sklearn.metrics import (
    confusion_matrix,
)

def run_experiment(seed, config, create_plots=True):
    set_seed(seed=seed)

    # --- Optimierungsparameter ---
    lr = config["lr"]
    alpha = config["alpha"]
    min_recall = config["min_recall"]
    y_aug_params = config.get("y_aug_params", None)

    # --- Device definieren ---
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

    # --- Hard positives laden ---
    hard_positive_paths = set()

    if config.get("use_hard_positives", False):

        trained_models_dir = Path("trained_models")

        # alle vorherigen hard_positive Dateien suchen
        hard_positive_files = sorted(
            trained_models_dir.glob("exp_*/hard_positives.txt")
        )

        # aktuelle Experimentdatei ausschließen
        hard_positive_files = [
            p for p in hard_positive_files
            if exp_dir.name not in str(p)
        ]

        if len(hard_positive_files) > 0:

            for file in hard_positive_files:
                with open(file, "r") as f:
                    paths = set(
                        line.strip()
                        for line in f.readlines()
                    )
                    hard_positive_paths.update(paths)

            print(f"Loaded {len(hard_positive_paths)} unique hard positives")
            print(f"Using {len(hard_positive_files)} hard positive files")

    # --- Daten ---
    train_dataset = get_train_dataset("data/train", y_aug_params=y_aug_params)
    val_dataset   = get_eval_dataset("data/val")
    test_dataset  = get_eval_dataset("data/test")

    # --- Weighted sampler für Trainingsdaten ---
    train_sampler, class_counts = build_weighted_sampler(
        train_dataset,
        seed=seed,
        alpha=alpha,
        hard_positive_paths=hard_positive_paths,
        hard_positive_weight=15.0,  # Hard Positive Mining: Schwierige Fälle öfter angucken
    )

    # --- Loader ---
    train_loader = get_loader(train_dataset, batch_size=32, sampler=train_sampler, seed=seed)
    val_loader = get_loader(val_dataset, batch_size=32, seed=seed)
    test_loader = get_loader(test_dataset, batch_size=32, seed=seed)

    # --- Modell ---
    model_name = "resnet50_cross_entropy"  # hier Modellname austauschen für anderes Modell aus registry.py, zB "cross_entropy" oder "cross_entropy_simple"
    model_type = get_model_type(model_name)

    if model_type == "binary":
        negatives = class_counts[0]
        positives = class_counts[1]
        pos_weight = negatives / positives
        model = build_model(model_name, num_classes=len(train_dataset.classes), pos_weight=pos_weight)

    elif model_type == "multiclass":
        model = build_model(model_name, num_classes=len(train_dataset.classes), class_weights=None)

    model = model.to(device)

    # -------------------------------------------------
    # PHASE 1
    # -------------------------------------------------

    # --- Nur Head trainieren ---
    model.freeze_backbone()
    
    # --- 1. Optimizer ---
    optimizer = torch.optim.Adam(
        model.get_trainable_parameters(),
        lr=lr,
    )
    
    phase1_epochs = 5

    # --- Config speichern ---
    with open(exp_dir / "config.txt", "w") as f:
        f.write(f"model_name={model_name}\n")
        f.write(f"phase1_epochs={phase1_epochs}\n")
        f.write(f"lr={lr}\n")
        f.write("batch_size=32\n")
        f.write(f"classes={train_dataset.classes}\n")
        f.write(f"class_counts={dict(class_counts)}\n")
        f.write(f"sampler_alpha={alpha}\n")
        f.write(f"seed={seed}\n")
        f.write(f"y_aug_params={y_aug_params}\n")
        
    # --- Training ---    
    early_stopping = EarlyStopping(patience=6, min_delta=0.001, mode="max")

    for epoch in range(phase1_epochs):
        train_loss, train_acc, train_recall, train_precision, train_f1 = run_epoch(
            model, train_loader, device, optimizer=optimizer
        )
        val_loss, val_acc, val_recall, val_precision, val_f1 = run_epoch(
            model, val_loader, device
        )

        improved = early_stopping(val_f1)
        if improved:
            torch.save(model.state_dict(), best_model_path)
            print("✅ Best model saved")

        with open(metrics_path, "a") as f:
            f.write(
                f"Epoch {epoch+1}/{phase1_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Train Recall: {train_recall:.4f} | Train Precision: {train_precision:.4f} | Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val Recall: {val_recall:.4f} | Val Precision: {val_precision:.4f} | Val F1: {val_f1:.4f} \n"
            )

        if early_stopping.stop:
            print(f"⏹ Early stopping after epoch {epoch+1}")
            break

    # -------------------------------------------------
    # PHASE 2
    # -------------------------------------------------
    # --- Letzte Layer freigeben ---
    model.unfreeze_last_block()

    # --- Neuer Optimizer (kleinere lr) ---    
    optimizer = torch.optim.Adam(
        model.get_trainable_parameters(),
        lr=lr * 0.01,
    )

    phase2_epochs = 10

    # --- Config speichern ---
    with open(exp_dir / "config.txt", "a") as f:
        f.write(f"model_name={model_name}\n")
        f.write(f"phase2_epochs={phase2_epochs}\n")
        f.write(f"lr={lr}\n")
        f.write("batch_size=32\n")
        f.write(f"classes={train_dataset.classes}\n")
        f.write(f"class_counts={dict(class_counts)}\n")
        f.write(f"sampler_alpha={alpha}\n")
        f.write(f"seed={seed}\n")
        f.write(f"y_aug_params={y_aug_params}\n")
        
    # --- Training ---    
    early_stopping = EarlyStopping(patience=8, min_delta=0.001, mode="max")

    for epoch in range(phase2_epochs):
        train_loss, train_acc, train_recall, train_precision, train_f1 = run_epoch(
            model, train_loader, device, optimizer=optimizer
        )
        val_loss, val_acc, val_recall, val_precision, val_f1 = run_epoch(
            model, val_loader, device
        )

        improved = early_stopping(val_f1)
        if improved:
            torch.save(model.state_dict(), best_model_path)
            print("✅ Best model saved")

        with open(metrics_path, "a") as f:
            f.write(
                f"Epoch {epoch+1}/{phase2_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Train Recall: {train_recall:.4f} | Train Precision: {train_precision:.4f} | Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val Recall: {val_recall:.4f} | Val Precision: {val_precision:.4f} | Val F1: {val_f1:.4f} \n"
            )

        if early_stopping.stop:
            print(f"⏹ Early stopping after epoch {epoch+1}")
            break

    torch.save(model.state_dict(), final_model_path)
    print(f"📦 Final model saved: {final_model_path}")

    # --- Bestes Modell laden ---
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("✅ Best model loaded for test evaluation")

    # --- Threshold tuning ---
    thresholds = [round(x, 2) for x in torch.linspace(0.1, 0.5, steps=10).tolist()]
    threshold_results = evaluate_thresholds(model, val_loader, device, thresholds=thresholds)

    best = select_best_threshold(
        threshold_results,
        min_recall=0.7,
        metric="f1",
    )

    # --- Evaluierung ---
    test_acc, test_recall, test_precision, test_f1 = evaluate_with_threshold(
        model,
        test_loader,
        device,
        threshold=best["threshold"],
    )

    val_labels, val_probs = collect_labels_and_probs(
        model,
        val_loader,
        device,
    )

    if create_plots:
        plot_probability_histogram(
            labels=val_labels,
            probs=val_probs,
            output_path=exp_dir / "validation_probability_histogram.png",
        )

        plot_precision_recall_curve(
            labels=val_labels,
            probs=val_probs,
            output_path=exp_dir / "validation_precision_recall_curve.png",
        )

    test_labels, test_probs = collect_labels_and_probs(
        model,
        test_loader,
        device,
    )

    test_preds = (test_probs >= best["threshold"]).astype(int)
    cm = confusion_matrix(test_labels, test_preds)

    print("\nConfusion matrix:")
    print(cm)

    # NEU: False Negatives aus Validation, nicht aus Test, 
    # damit keine Data Leakage entsteht, wenn die hard positives
    # in weiteren runs als Train Daten verwendet werden
    plot_false_negatives(
        model=model,
        dataset=val_dataset,
        loader=val_loader,
        device=device,
        threshold=best["threshold"],
        output_dir=exp_dir,
        max_images=25,
        save_path=exp_dir / "false_negatives.png",
        save_hard_positives=True,
    )
    plot_false_negatives(
        model=model,
        dataset=test_dataset,
        loader=test_loader,
        device=device,
        threshold=best["threshold"],
        output_dir=exp_dir,
        max_images=25,
        save_path=exp_dir / "test_false_negatives.png",
        save_hard_positives=False,
    )

    with open(metrics_path, "a") as f:
        f.write("\nThreshold tuning on validation set:\n")

        for r in threshold_results:
            f.write(
                f"Threshold: {r['threshold']:.2f} | "
                f"Acc: {r['acc']:.4f} | "
                f"Recall: {r['recall']:.4f} | "
                f"Precision: {r['precision']:.4f} | "
                f"F1: {r['f1']:.4f}\n"
            )
        f.write(
            f"\nBest threshold (min_recall constraint): {best['threshold']:.2f} | "
            f"Recall: {best['recall']:.4f} | "
            f"Precision: {best['precision']:.4f} | "
            f"F1: {best['f1']:.4f}\n"
        )
        f.write(
            f"Test with tuned threshold={best['threshold']:.2f} | "
            f"Acc: {test_acc:.4f} | "
            f"Recall: {test_recall:.4f} | "
            f"Precision: {test_precision:.4f} | "
            f"F1: {test_f1:.4f}\n"
        )

    return {
        "seed": seed,
        "lr": lr,
        "threshold": best["threshold"],
        "val_f1": best["f1"],
        "test_acc": test_acc,
        "test_recall": test_recall,
        "test_precision": test_precision,
        "test_f1": test_f1,
    }


def main():

    results = []
    seeds = [10, 20, 30, 40, 50]
    lr = 1e-3
    for seed in seeds:
        print(f"\n--- Seed {seed} --- | lr={lr} ---")

        # Config bauen
        config = {
            "optimizer": "adam",
            "lr": lr,
            "alpha": 0.30,
            "min_recall": 0.7,
            "y_aug_params": {
                "hflip_p": 0.5,
                "vflip_p": 0.2,
                "brightness": 0.3,
                "contrast": 0.5,
                "saturation": 0.5,
                "perspective": 0.2, 
                "rotation_deg": 20,
            },
            "use_hard_positives": True,
        }

        res = run_experiment(seed=seed, config=config, create_plots=True)
        results.append(res)

        print(
            f"Seed {seed} → "
            f"Test F1: {res['test_f1']:.4f} | "
            f"Recall: {res['test_recall']:.4f} | "
            f"Precision: {res['test_precision']:.4f}"
        )


if __name__ == "__main__":
    main()