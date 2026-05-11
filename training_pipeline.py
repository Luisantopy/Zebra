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

    # --- Optimierungsparameter ---
    lr = config["lr"]
    alpha = config["alpha"]
    optimizer_name = "sgd"
    min_recall = config["min_recall"]
    momentum = config["momentum"]
    #y_aug_params = config["y_aug_params"]

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

    # --- Daten ---
    train_dataset = get_train_dataset("data/train")
    val_dataset   = get_eval_dataset("data/val")
    test_dataset  = get_eval_dataset("data/test")

    # --- Weighted sampler für Trainingsdaten ---
    train_sampler, class_counts = build_weighted_sampler(train_dataset, alpha=alpha)

    # --- Loader ---
    train_loader = get_loader(train_dataset, batch_size=32, sampler=train_sampler, seed=seed)
    val_loader = get_loader(val_dataset, batch_size=32, seed=seed)
    test_loader = get_loader(test_dataset, batch_size=32, seed=seed)

    # --- Modell ---
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
            momentum=momentum, 
            #weight_decay=5e-4 #L2 Regularisierung
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
    
    num_epochs = 15

    # --- Scheduluer ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

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
       # f.write(f"y_aug_params={y_aug_params}\n")
        
    # --- Training ---    
    early_stopping = EarlyStopping(patience=3, min_delta=0.001, mode="max")

    for epoch in range(num_epochs):
        train_loss, train_acc, train_recall, train_precision, train_f1 = run_epoch(
            model, train_loader, device, optimizer=optimizer
        )
        val_loss, val_acc, val_recall, val_precision, val_f1 = run_epoch(
            model, val_loader, device
        )

        scheduler.step(val_f1)

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


def objective(trial):
    optimizer_name = "sgd" # trial.suggest_categorical("optimizer", ["adam", "sgd"])
    lr = trial.suggest_float("lr", 0.03, 0.06, log=True)
    alpha = trial.suggest_float("alpha", 0.50, 0.65)
    min_recall = trial.suggest_float("min_recall", 0.33, 0.45)

    config = {
        "optimizer": optimizer_name,
        "lr": lr,
        "alpha": alpha,
        "min_recall": min_recall,
        "momentum": trial.suggest_float("momentum", 0.82, 0.88)
    }

    seeds = [10, 20, 30]
    vals_f1 =[]

    for seed in seeds:
        result = run_experiment(seed=seed, config=config)
        vals_f1.append(result["val_f1"])

    # return float(np.mean(vals_f1)) 
    return float(np.mean(vals_f1) - np.std(vals_f1)) # Instabilität bestrafen


def objective(trial):
    # --- nur y-Augmentation tunen ---
    y_aug_params = {
        "hflip_p": trial.suggest_float("y_hflip_p", 0.3, 0.6),
        "vflip_p": trial.suggest_float("y_vflip_p", 0.0, 0.3),
        "brightness": trial.suggest_float("y_brightness", 0.1, 0.5),
        "contrast": trial.suggest_float("y_contrast", 0.2, 0.7),
        "saturation": trial.suggest_float("y_saturation", 0.2, 0.7),
        "perspective_p": trial.suggest_float("y_perspective_p", 0.0, 0.3),
        "rotation_deg": trial.suggest_int("y_rotation_deg", 5, 25),
    }

    config = {
        "y_aug_params": y_aug_params
    }

    seeds = [10, 20, 30]
    vals_f1 = []

    for seed in seeds:
        result = run_experiment(seed=seed, config=config)
        vals_f1.append(result["val_f1"])

    # Mittelwert minus Std, um Instabilität zu bestrafen
    return float(np.mean(vals_f1) - np.std(vals_f1))


def evaluate_best_trial(study, seeds=[10, 20, 30, 40, 50]):
    best_params = study.best_trial.params

    print("\n🚀 Evaluating best trial with multiple seeds")
    print("Best params:", best_params)

    y_aug_params = {
        "hflip_p": best_params["y_hflip_p"],
        "vflip_p": best_params["y_vflip_p"],
        "brightness": best_params["y_brightness"],
        "contrast": best_params["y_contrast"],
        "saturation": best_params["y_saturation"],
        "perspective_p": best_params["y_perspective_p"],
        "rotation_deg": best_params["y_rotation_deg"],
    }
    config = {"y_aug_params": y_aug_params}
    results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

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
    print(f"Min Test F1: {np.min([r['test_f1'] for r in results]):.4f}")
    print(f"Max Test F1: {np.max([r['test_f1'] for r in results]):.4f}")

    return results

def main():
    #study = optuna.create_study(direction="maximize")
    #study.optimize(objective, n_trials=15)

    #print("Best trial:")
    #print(study.best_trial.params)
    #print(study.best_trial.value)

    #evaluate_best_trial(study=study, seeds=[40, 50, 60, 70, 80])
    results = []
    seeds=[10, 20, 30, 40, 50]
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Config bauen
        config = {
            "optimizer": "sgd",
            "lr": 0.061,
            "alpha": 0.60,
            "min_recall": 0.39,
            "momentum": 0.87,
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
    print(f"Min Test F1: {np.min([r['test_f1'] for r in results]):.4f}")
    print(f"Max Test F1: {np.max([r['test_f1'] for r in results]):.4f}")

if __name__ == "__main__":
    main()