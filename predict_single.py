import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from model_registry import build_model, get_model_type


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_eval_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
def load_model(weights_path: str, device: torch.device, model_name: str, num_classes: int):
    model = build_model(model_name=model_name, num_classes=num_classes).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def predict_image(image_path: str, model, device: torch.device, class_names: list[str], model_name: str):
    model_type = get_model_type(model_name)

    image = Image.open(image_path).convert("RGB")
    transform = get_eval_transform()

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        preds = model.predict(outputs)
        pred_idx = preds.item()

        if model_type == "binary":
            prob_pos = torch.sigmoid(outputs).item()
            probabilities = [1.0 - prob_pos, prob_pos]
        else:
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().tolist()

    return {
        "pred_idx": pred_idx,
        "pred_class": class_names[pred_idx],
        "probabilities": probabilities,
    }


def main():
    parser = argparse.ArgumentParser(description="Vorhersage für ein einzelnes Bild")
    parser.add_argument("image_path", type=str, help="Pfad zum Bild")
    parser.add_argument(
        "--weights",
        type=str,
        default="trained_models/simple_cnn_best.pth",
        help="Pfad zu den gespeicherten Modellgewichten",
    )
    parser.add_argument(
    "--model",
    type=str,
    default="cross_entropy",
    choices=["cross_entropy", "binary_bce"],
)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["n", "y"],
        help="Klassenreihenfolge (z. B. --classes n y)",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    weights_path = Path(args.weights)
    class_names = args.classes

    if not image_path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Gewichtedatei nicht gefunden: {weights_path}")
    
    if args.model == "binary_bce" and len(class_names) != 2:
        raise ValueError("binary_bce erwartet genau 2 Klassen.")

    device = get_device()

    model = load_model(
        weights_path=str(weights_path),
        device=device,
        model_name=args.model,
        num_classes=len(class_names),
    )

    result = predict_image(
        image_path=str(image_path),
        model=model,
        device=device,
        class_names=class_names,
        model_name=args.model,
    )

    print(f"\nBild: {image_path}")
    print(f"Vorhersage: {result['pred_class']} (Klasse {result['pred_idx']})")
    print("\nWahrscheinlichkeiten:")
    for class_name, prob in zip(class_names, result["probabilities"]):
        print(f"  {class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()