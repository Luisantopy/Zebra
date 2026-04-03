import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class SimpleCNNKeras(keras.Model):
    """ - 3 Conv-Blöcke
        - jeweils ReLU + MaxPooling
        - Global Average Pooling-artige Verdichtung
        - eine Dense-Schicht für die Klassen."""
    
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = keras.Sequential([
            layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
        ])

        self.classifier = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes)
        ])

    def call(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x