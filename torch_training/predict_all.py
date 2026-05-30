import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .model_registry import build_model, get_model_type
from helpers import save_confusion_matrix


# --- Device bestimmen ---
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# --- Evaluation Transformation ---
def get_eval_transform():
    """
    Gleiche Vorverarbeitung wie im Training
    (ohne Data Augmentation).

    Schritte:
    - PIL -> Tensor
    - float32 + Skalierung [0,1]
    - ImageNet Normalisierung
    """
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
# --- Modell laden --- 
def load_model(weights_path: str, device: torch.device, model_name: str, num_classes: int):
    """
    Baut Modell aus Registry,
    lädt Gewichte
    und setzt eval()-Modus.
    """
    model = build_model(model_name=model_name, num_classes=num_classes).to(device)

    # gespeicherte Gewichte laden
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # wichtig für Inference
    model.eval()
    return model


# --- Predictions für einen Batch ---
def predict_batch(outputs, model_name, threshold=0.5):
    """
    Wandelt rohe Modelloutputs in Klassenpredictions um.

    Unterstützt:
    - BCE Modelle
    - CrossEntropy Modelle
    - pretrained ResNet18 / ResNet50 CrossEntropy-Modelle
    """
    model_type = get_model_type(model_name)

    # --- BCE Modell ---
    if model_type == "binary":
        # BCE: ein Logit -> Sigmoid -> p(y)
        # sigmoid -> Wahrscheinlichkeit positive Klasse
        probs_pos = torch.sigmoid(outputs).view(-1)
        preds = (probs_pos >= threshold).long()
    
    # --- CrossEntropy Modelle --
    else:
        # CrossEntropy / ResNet: zwei Logits -> Softmax -> p(y)
        probs = torch.softmax(outputs, dim=1)
        probs_pos = probs[:, 1]
        preds = (probs_pos >= threshold).long()

    return preds, probs_pos


# --- Evaluation über gesamten Testdatensatz ---
def evaluate_dataset(
    model,
    loader,
    device,
    model_name,
    threshold,
):
    """
    Führt Inference auf allen Bildern im Loader aus.
    """
    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            preds, probs_pos = predict_batch(
                outputs=outputs,
                model_name=model_name,
                threshold=threshold,
            )

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs_pos.cpu().tolist())

    return all_labels, all_preds, all_probs


def main():

    # --- CLI Argumente ---
    parser = argparse.ArgumentParser(description="Modell auf gesamten Testdatensatz evaluieren")

    # Testdatensatz
    parser.add_argument(
        "--data",
        type=str,
        default="data/test",
        help="Pfad zum Testdatensatz im ImageFolder-Format.",
    )

    # Modellgewichte
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Pfad zu den gespeicherten Modellgewichten (.pth).",
    )

    # Modelltyp
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "cross_entropy",
            "binary_bce",
            "resnet18_cross_entropy",
            "resnet50_cross_entropy",
        ],
        help="Modellarchitektur passend zu den gespeicherten Gewichten.",
    )

    # Klassenreihenfolge
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["n", "y"],
        help="Klassenreihenfolge (z. B. --classes n y)",
    )

    # Threshold für positive Klasse
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Threshold für Klasse y",
    )

    # Batch Size für Evaluation
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch Size für Evaluation.",
    )

    # Dateiname für gespeicherte Confusion Matrix
    parser.add_argument(
        "--output-name",
        type=str,
        default="test_confusion_matrix.png",
        help="Dateiname für gespeicherte Confusion Matrix.",
    )

    args = parser.parse_args()

    # --- Pfade vorbereiten --- 
    data_path = Path(args.data)
    weights_path = Path(args.weights)
    class_names = args.classes

    # --- Validierung ----
    if not data_path.exists():
        raise FileNotFoundError(f"Testdatensatz nicht gefunden: {data_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Gewichtedatei nicht gefunden: {weights_path}")
    
    if args.model == "binary_bce" and len(class_names) != 2:
        raise ValueError("binary_bce erwartet genau 2 Klassen.")

    # --- Device bestimmen ---
    device = get_device()

    # ImageFolder erwartet z. B.:
    # data/test/n/*.png
    # data/test/y/*.png

    # --- Testdatensatz laden ---
    dataset = datasets.ImageFolder(
        root=data_path,
        transform=get_eval_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Loaded {len(dataset)} test images")
    print(f"Dataset classes: {dataset.classes}")
    print(f"Using class names for plot: {class_names}")

    # --- Modell laden --- 
    model = load_model(
        weights_path=str(weights_path),
        device=device,
        model_name=args.model,
        num_classes=len(class_names),
    )

    # --- Evaluation ---
    y_true, y_pred, y_prob = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        model_name=args.model,
        threshold=args.threshold,
    )

    # Modellordner = Ordner der Gewichte
    model_dir = weights_path.parent
    save_path = model_dir / args.output_name

    cm = save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=save_path,
        title=f"Test Confusion Matrix ({args.model}, threshold={args.threshold})",
    )
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    print("\\nConfusion matrix:")
    print(cm)

    print("\\nTest metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1:        {f1:.4f}")

    print(f"\\nSaved confusion matrix to: {save_path}")


if __name__ == "__main__":
    main()