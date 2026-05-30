import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from .model_registry import build_model, get_model_type


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


# --- Einzelbild vorhersagen ---
def predict_image(image_path, model, device, class_names, model_name, threshold=0.5):
    """
    Führt Vorhersage für einzelnes Bild aus.

    Unterstützt:
    - BCE Modelle
    - CrossEntropy Modelle
    - pretrained ResNet18 / ResNet50 CrossEntropy-Modelle

    Wichtig:
    ResNet18 und ResNet50 liefern wie das CrossEntropy-CNN
    zwei Logits zurück. Deshalb kann dieselbe Softmax- und Threshold-
    Logik verwendet werden.
    """
    model_type = get_model_type(model_name)

    # Bild laden
    image = Image.open(image_path).convert("RGB")

    # gleiche Eval-Transformation wie im Training
    transform = get_eval_transform()

    # Batch-Dimension ergänzen:
    # [C,H,W] -> [1,C,H,W]
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        # logits berechnen
        outputs = model(x)

        # --- BCE Modell ---
        if model_type == "binary":

            # sigmoid -> Wahrscheinlichkeit positive Klasse
            prob_pos = torch.sigmoid(outputs).item()
            probabilities = [1.0 - prob_pos, prob_pos]

            # threshold-basierte Entscheidung
            pred_idx = int(prob_pos >= threshold)
        
        # --- CrossEntropy Modelle --
        else:

            # softmax Wahrscheinlichkeiten
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().tolist()

            # Wahrscheinlichkeit Klasse y
            prob_pos = probabilities[1]

            # threshold-basierte Entscheidung
            pred_idx = int(prob_pos >= threshold)

    return {
        "pred_idx": pred_idx,
        "pred_class": class_names[pred_idx],
        "probabilities": probabilities,
        "threshold": threshold,
    }


def main():

    # --- CLI Argumente ---
    parser = argparse.ArgumentParser(description="Vorhersage für ein einzelnes Bild")

    # Pflichtargument:
    # Bildpfad
    parser.add_argument("image_path", type=str, help="Pfad zum Bild")

    # Modellgewichte
    parser.add_argument(
        "--weights",
        type=str,
        default="trained_models/simple_cnn_best.pth",
        help="Pfad zu den gespeicherten Modellgewichten",
    )

    # Modelltyp
    parser.add_argument(
        "--model",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "binary_bce", "resnet18_cross_entropy", "resnet50_cross_entropy"],
    )

    # Optionale Argumente:
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
    args = parser.parse_args()

    # --- Pfade vorbereiten --- 
    image_path = Path(args.image_path)
    weights_path = Path(args.weights)
    class_names = args.classes

    # --- Validierung ----
    if not image_path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Gewichtedatei nicht gefunden: {weights_path}")
    
    if args.model == "binary_bce" and len(class_names) != 2:
        raise ValueError("binary_bce erwartet genau 2 Klassen.")

    # --- Device bestimmen ---
    device = get_device()

    # --- Modell laden --- 
    model = load_model(
        weights_path=str(weights_path),
        device=device,
        model_name=args.model,
        num_classes=len(class_names),
    )

    # --- Prediction ---
    result = predict_image(
        image_path=str(image_path),
        model=model,
        device=device,
        class_names=class_names,
        model_name=args.model,
        threshold=args.threshold,
    )

    # --- Ausgabe ---
    print(f"\nBild: {image_path}")
    print(f"Vorhersage: {result['pred_class']} (Klasse {result['pred_idx']})")
    print("\nWahrscheinlichkeiten:")
    for class_name, prob in zip(class_names, result["probabilities"]):
        print(f"  {class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()