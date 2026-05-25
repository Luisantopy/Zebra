from .torch_cnn_simple import SimpleCNNCrossEntropy, SimpleCNNBinary
from .torch_cnn import CNNCrossEntropy
from .torch_resnet import ResNet18CrossEntropy, ResNet50CrossEntropy


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
    "resnet18_cross_entropy": {
        "builder": lambda num_classes, **kwargs: ResNet18CrossEntropy(num_classes=num_classes, **kwargs),
        "type": "multiclass",
    },
    "resnet50_cross_entropy": {
        "builder": lambda num_classes, **kwargs: ResNet50CrossEntropy(num_classes=num_classes, **kwargs),
        "type": "multiclass",
    },
}


def build_model(model_name: str, num_classes: int, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

    return MODEL_REGISTRY[model_name]["builder"](num_classes=num_classes, **kwargs)


def get_model_type(model_name: str):
    return MODEL_REGISTRY[model_name]["type"]