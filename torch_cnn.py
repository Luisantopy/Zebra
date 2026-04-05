import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseClassifier(nn.Module):
    positive_label = 1

    def __init__(self):
        super().__init__()

    def compute_loss(self, outputs, labels):
        raise NotImplementedError

    def predict(self, outputs):
        raise NotImplementedError


class _CNNBackboneMixin:
    """Architektur mit 3 Blocks:
    + Conv Layer
    + Aktivierungsfunktion
    + Pooling
    und Verdichtung. 
    """
    def _init_backbone(self):
        self.features = nn.Sequential(
            # - Input Layer -
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
           # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # - Hidden Layer -
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
         #   nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # - Hidden Layer -
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
          #  nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # - Verdichtung -
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward_features(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class CNNCrossEntropy(BaseClassifier, _CNNBackboneMixin):
    """
    Binary Classification mit CrossEntropyLoss.
    KEINE Aktivierungsfunktion im Output Layer, CrossEntropy übernimmt Softmax Logik
    Output: [batch_size, num_classes]
    """

    def __init__(self, num_classes=2, class_weights=None):
        super().__init__()
        self._init_backbone()

        # Output Layer
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.class_weights = class_weights

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    # Loss Function definieren
    def compute_loss(self, outputs, labels):
        weight = self.class_weights
        if weight is not None:
            weight = weight.to(outputs.device)
        return F.cross_entropy(outputs, labels.long(), weight=weight)

    def predict(self, outputs):
        return outputs.argmax(dim=1)


# class CNNBinary(BaseClassifier, _CNNBackboneMixin):
#     """
#     Binary Classification mit BCEWithLogitsLoss.
#     Output: [batch_size, 1]
#     """

#     def __init__(self, pos_weight=None):
#         super().__init__()
#         self._init_backbone()

#         self.head = nn.Linear(128, 1)

#         if pos_weight is not None:
#             pos_weight = torch.as_tensor([pos_weight], dtype=torch.float32)
#         self.pos_weight = pos_weight

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x

#     def compute_loss(self, outputs, labels):
#         labels = labels.float().unsqueeze(1)
#         pos_weight = self.pos_weight
#         if pos_weight is not None:
#             pos_weight = pos_weight.to(outputs.device)
#         return F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=pos_weight)

#     def predict(self, outputs):
#         probs = torch.sigmoid(outputs)
#         return (probs >= 0.5).long().squeeze(1)