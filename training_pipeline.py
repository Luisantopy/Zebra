import torch
import optuna
import numpy as np

from data_augmentation import get_train_dataset, get_eval_dataset, get_loader
from model_registry import build_model, get_model_type
from helpers import (
    EarlyStopping, 
    run_epoch, setup_experiment, build_weighted_sampler, 
    evaluate_with_threshold, evaluate_thresholds, select_best_threshold,
    set_seed
)


def run_experiment(seed, config):
    set_seed(seed=seed)
    lr = config["lr"]
    optimizer_name = config["optimizer"]
    alpha = config["alpha"]
    min_recall = config["min_recall"]
    momentum = config.get("momentum", None)

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

    # --- Weighted sampler für Trainingsdaten ---
    train_sampler, class_counts = build_weighted_sampler(train_dataset, alpha=alpha)

    # --- DataLoader ---
    train_loader = get_loader(train_dataset, batch_size=32, sampler=train_sampler, seed=seed)
    val_loader = get_loader(val_dataset, batch_size=32, seed=seed)
    test_loader = get_loader(test_dataset, batch_size=32, seed=seed)

    # --- Model ---
    model_name = "cross_entropy"  # hier Modellname austauschen für anderes Modell aus registry, zB "cross_entropy" oder "cross_entropy_simple"
    model_type = get_model_type(model_name)

    if model_type == "binary":
        negatives = class_counts[0]
        positives = class_counts[1]
        pos_weight = negatives / positives

        model = build_model(model_name, num_classes=len(train_dataset.classes), pos_weight=pos_weight)

    elif model_type == "multiclass":
        model = build_model(model_name, num_classes=len(train_dataset.classes), class_weights=None)

    model = model.to(device)

    # --- Optimizer ---
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
    
    num_epochs = 15

    # --- Scheduluer ---
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=0.5,
    #     patience=2
    # )

    # --- Config speichern ---
    with open(exp_dir / "config.txt", "w") as f:
        f.write(f"model_name={model_name}\n")
        f.write(f"epochs={num_epochs}\n")
        f.write(f"lr={lr}\n")
        f.write("batch_size=32\n")
        f.write(f"classes={train_dataset.classes}\n")
        f.write(f"class_counts={dict(class_counts)}\n")
        f.write(f"sampler_alpha={alpha}\n")
        f.write(f"seed={seed}\n")
        
    # --- Training ---    
    early_stopping = EarlyStopping(patience=2, min_delta=0.001, mode="max")

    for epoch in range(num_epochs):
        train_loss, train_acc, train_recall, train_precision, train_f1 = run_epoch(
            model, train_loader, device, optimizer=optimizer
        )
        val_loss, val_acc, val_recall, val_precision, val_f1 = run_epoch(
            model, val_loader, device
        )

        #scheduler.step(val_loss)

        improved = early_stopping(val_f1)
        if improved:
            torch.save(model.state_dict(), best_model_path)
            print("✅ Best model saved")

        with open(metrics_path, "a") as f:
            f.write(
                f"Epoch {epoch+1}/{num_epochs} | "
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

    # --- Threshold Tuning ---
    threshold_results = evaluate_thresholds(model, val_loader, device)
    best = select_best_threshold(
        threshold_results,
        min_recall=min_recall,      
        metric="f1"   
    )
    test_acc, test_recall, test_precision, test_f1 = evaluate_with_threshold(
        model,
        test_loader,
        device,
        threshold=best["threshold"]
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


# def run_multiple_seeds(lr, seeds=[10, 20, 30, 40, 50]):
#     results = []

#     for seed in seeds:
#         print(f"\n--- Running seed {seed} (lr={lr}) ---")
#         res = run_experiment(seed, lr)
#         results.append(res)
#         print(f"Seed {seed} → Test F1: {res['test_f1']:.4f}")

#     return results


def objective(trial):
    optimizer_name = "sgd" # trial.suggest_categorical("optimizer", ["adam", "sgd"])
    lr = trial.suggest_float("lr", 0.03, 0.08, log=True)
    alpha = trial.suggest_float("alpha", 0.55, 0.70)
    min_recall = trial.suggest_float("min_recall", 0.33, 0.45)

    config = {
        "optimizer": optimizer_name,
        "lr": lr,
        "alpha": alpha,
        "min_recall": min_recall,
    }

    if optimizer_name == "sgd":
        config["momentum"] = trial.suggest_float("momentum", 0.82, 0.88)

    result = run_experiment(seed=42, config=config)

    return result["val_f1"]


def evaluate_best_trial(study, seeds=[10, 20, 30, 40, 50]):
    best_params = study.best_trial.params

    print("\n🚀 Evaluating best trial with multiple seeds")
    print("Best params:", best_params)

    results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Config bauen
        config = {
            "optimizer": "sgd",
            "lr": best_params["lr"],
            "alpha": best_params["alpha"],
            "min_recall": best_params["min_recall"],
            "momentum": best_params["momentum"],
        }

        res = run_experiment(seed=seed, config=config)
        results.append(res)

        print(
            f"Seed {seed} → "
            f"Test F1: {res['test_f1']:.4f} | "
            f"Recall: {res['test_recall']:.4f} | "
            f"Precision: {res['test_precision']:.4f}"
        )

    # --- Aggregation ---
    avg_f1 = np.mean([r["test_f1"] for r in results])
    std_f1 = np.std([r["test_f1"] for r in results])

    avg_recall = np.mean([r["test_recall"] for r in results])
    avg_precision = np.mean([r["test_precision"] for r in results])

    print("\n📊 Summary over seeds:")
    print(f"Avg Test F1: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"Avg Recall:  {avg_recall:.4f}")
    print(f"Avg Precision: {avg_precision:.4f}")

    return results

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    print("Best trial:")
    print(study.best_trial.params)
    print(study.best_trial.value)

    evaluate_best_trial(study=study)


if __name__ == "__main__":
    main()