import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights


class BaseClassifier(nn.Module):
    positive_label = 1

    def __init__(self):
        super().__init__()

    def compute_loss(self, outputs, labels):
        raise NotImplementedError

    def predict(self, outputs):
        raise NotImplementedError


class ResNet18CrossEntropy(BaseClassifier):
    """
    Pretrained ResNet18 Modell
    - nutzt vortrainiertes ResNet18 als Feature Extractor 
    - ersetzt nur den letzten Klassifikationslayer für spezifisches 
    Klassifikationsproblem

    Phase 1: trainiert nur Klassifikationskopf
    - stabil
    - schnell
    - kein sofortiges Overfitting

    Phase 2: trainiert letzte ResNet-Features + Head
    - Domänenanpassung
    - bessere Spezialisierung auf Zebra-Crossings
    """

    def __init__(self, num_classes=2, class_weights=None):
        super().__init__()

        # --- Backbone laden ---
        self.backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        # --- letzten Layer ersetzen ---
        in_features = self.backbone.fc.in_features

        # vorher: 512 → 1000 Klassen 
        # nachher: 512 → 2 Klassen
        self.backbone.fc = nn.Linear(
            in_features,
            num_classes,
        )

        # --- class weights initialisieren ---
        if class_weights is not None:
            class_weights = torch.as_tensor(
                class_weights,
                dtype=torch.float32,
            )

        self.class_weights = class_weights

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------

    def forward(self, x):
        """
        ResNet macht intern:

        - Convolutions
        - Residual Blocks
        - Pooling
        - Final FC

        return shape: [batch_size, 2]
        """
        return self.backbone(x)

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    def compute_loss(self, outputs, labels):
        """
        CrossEntropyLoss
        - keine Softmax im Modell nötig
        - CrossEntropy rechnet intern mit Logits
        """
        weight = self.class_weights

        if weight is not None:
            weight = weight.to(outputs.device)

        return F.cross_entropy(
            outputs,
            labels.long(),
            weight=weight,
        )

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    def predict(self, outputs):
        """
        Nimmt grössere der 2 Klassenwahrscheinlichkeiten
        """
        return outputs.argmax(dim=1)

    # -------------------------------------------------
    # Fine-tuning helpers
    # -------------------------------------------------

    def freeze_backbone(self):
        """
        Friert kompletten Backbone ein,
        trainiert nur den FC-Head.
        """

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_last_block(self):
        """
        Gibt letzten ResNet-Block frei.
        """

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """
        Gibt komplettes Modell frei.
        """

        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """
        Liefert nur trainierbare Parameter
        für den Optimizer.
        """

        return filter(
            lambda p: p.requires_grad,
            self.parameters(),
        )
    

class ResNet50CrossEntropy(BaseClassifier):
    """
    Pretrained ResNet50 Modell
    für binäre Klassifikation.
    """

    def __init__(self, num_classes=2, class_weights=None):
        super().__init__()

        # -------------------------------------------------
        # Backbone
        # -------------------------------------------------

        self.backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        # -------------------------------------------------
        # Letzten Layer ersetzen
        # -------------------------------------------------

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            num_classes,
        )

        # -------------------------------------------------
        # Class Weights
        # -------------------------------------------------

        if class_weights is not None:
            class_weights = torch.as_tensor(
                class_weights,
                dtype=torch.float32,
            )

        self.class_weights = class_weights

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------

    def forward(self, x):
        return self.backbone(x)

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    def compute_loss(self, outputs, labels):

        weight = self.class_weights

        if weight is not None:
            weight = weight.to(outputs.device)

        return F.cross_entropy(
            outputs,
            labels.long(),
            weight=weight,
        )

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    def predict(self, outputs):
        return outputs.argmax(dim=1)

    # -------------------------------------------------
    # Fine-tuning helpers
    # -------------------------------------------------

    def freeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_last_block(self):

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_all(self):

        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):

        return filter(
            lambda p: p.requires_grad,
            self.parameters(),
        )