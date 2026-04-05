from torch_cnn_simple import SimpleCNNCrossEntropy, SimpleCNNBinary
from torch_cnn import CNNCrossEntropy


MODEL_REGISTRY = {
    "cross_entropy_simple": {
        "builder": lambda num_classes, **kwargs: SimpleCNNCrossEntropy(num_classes=num_classes, **kwargs),
        "type": "multiclass",
    },
    "binary_bce_simple": {
        "builder": lambda num_classes, **kwargs: SimpleCNNBinary(**kwargs),
        "type": "binary",
    },
    "cross_entropy": {
        "builder": lambda num_classes, **kwargs: CNNCrossEntropy(num_classes=num_classes, **kwargs),
        "type": "multiclass",
    },
}


def build_model(model_name: str, num_classes: int, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

    return MODEL_REGISTRY[model_name]["builder"](num_classes=num_classes, **kwargs)


def get_model_type(model_name: str):
    return MODEL_REGISTRY[model_name]["type"]