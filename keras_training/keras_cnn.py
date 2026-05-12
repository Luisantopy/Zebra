import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# CNN Architektur analog zur PyTorch-Version.
class CNNKeras(keras.Model):
    """
    CNN Architektur analog zur PyTorch-Version.

    - 3 Conv-Blöcke
    - ReLU Aktivierungen
    - MaxPooling
    - GlobalAveragePooling
    - kleiner Dense Head
    - Output ohne Softmax (Logits)
    """

    def __init__(
        self,
        num_classes=2,
        dropout_rate=0.2,
    ):
        super().__init__()

        # --- Block 1 ---
        self.conv1 = layers.Conv2D(
            32,
            kernel_size=3,
            padding="same",
            activation="relu",
        )
        self.pool1 = layers.MaxPooling2D(pool_size=2)

        # --- Block 2 ---
        self.conv2 = layers.Conv2D(
            64,
            kernel_size=3,
            padding="same",
            activation="relu",
        )
        self.pool2 = layers.MaxPooling2D(pool_size=2)

        # --- Block 3 ---
        self.conv3 = layers.Conv2D(
            128,
            kernel_size=3,
            padding="same",
            activation="relu",
        )
        self.pool3 = layers.MaxPooling2D(pool_size=2)

        # --- Pooling ---
        self.global_pool = layers.GlobalAveragePooling2D()

        # --- Head ---
        self.fc1 = layers.Dense(
            64,
            activation="relu"
        )

        self.dropout = layers.Dropout(dropout_rate)

        # --- Output ---
        self.output_layer = layers.Dense(num_classes)

    def build(self, input_shape):
        self.conv1.build(input_shape)
        x_shape = self.conv1.compute_output_shape(input_shape)
        x_shape = self.pool1.compute_output_shape(x_shape)

        self.conv2.build(x_shape)
        x_shape = self.conv2.compute_output_shape(x_shape)
        x_shape = self.pool2.compute_output_shape(x_shape)

        self.conv3.build(x_shape)
        x_shape = self.conv3.compute_output_shape(x_shape)
        x_shape = self.pool3.compute_output_shape(x_shape)

        self.global_pool.build(x_shape)
        x_shape = self.global_pool.compute_output_shape(x_shape)

        self.fc1.build(x_shape)
        x_shape = self.fc1.compute_output_shape(x_shape)

        self.dropout.build(x_shape)

        self.output_layer.build(x_shape)

        super().build(input_shape)
        
    def call(self, x, training=False):

        # Block 1
        x = self.conv1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.pool3(x)

        # Pooling
        x = self.global_pool(x)

        # Head
        x = self.fc1(x)

        x = self.dropout(
            x,
            training=training
        )

        # Output (Logits)
        x = self.output_layer(x)

        return x


# Pretrained EfficientNetB0 für Vergleich
class EfficientNetBinary(keras.Model):
    """
    EfficientNetB0 mit ImageNet Pretraining
    für binäre Klassifikation.

    Architektur:
    - EfficientNetB0 Backbone
    - GlobalAveragePooling
    - kleiner Dense Head
    - Sigmoid Output

    Output:
    - Wahrscheinlichkeit für Klasse 1
    """

    def __init__(
        self,
        dropout_rate=0.2,
        trainable_backbone=False,
    ):
        super().__init__()

        # =====================================================
        # Backbone
        # =====================================================

        self.backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )

        self.backbone.trainable = trainable_backbone

        # =====================================================
        # Head
        # =====================================================

        self.global_pool = layers.GlobalAveragePooling2D()

        self.fc1 = layers.Dense(
            128,
            activation="relu",
        )

        self.dropout = layers.Dropout(dropout_rate)

        # Binary output
        self.output_layer = layers.Dense(
            1,
            activation="sigmoid",
        )

    def build(self, input_shape):

        self.backbone.build(input_shape)

        x_shape = self.backbone.compute_output_shape(input_shape)

        self.global_pool.build(x_shape)
        x_shape = self.global_pool.compute_output_shape(x_shape)

        self.fc1.build(x_shape)
        x_shape = self.fc1.compute_output_shape(x_shape)

        self.dropout.build(x_shape)

        self.output_layer.build(x_shape)

        super().build(input_shape)

    def call(self, x, training=False):

        # =====================================================
        # Backbone
        # =====================================================

        x = self.backbone(
            x,
            training=training,
        )

        # =====================================================
        # Head
        # =====================================================

        x = self.global_pool(x)

        x = self.fc1(x)

        x = self.dropout(
            x,
            training=training,
        )

        # =====================================================
        # Output probability
        # =====================================================

        x = self.output_layer(x)

        return x