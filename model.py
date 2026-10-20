"""CNN + bidirectional LSTM for log-Mel input."""
import keras
from keras import layers

import config


def build_model(
    n_mels: int = config.N_MELS,
    n_frames: int = config.N_FRAMES,
    n_classes: int = len(config.TARGET_EMOTIONS),
    dropout: float = config.DROPOUT,
    conv_dropout: float = config.CONV_DROPOUT,
) -> keras.Model:
    inputs = keras.Input(shape=(n_mels, n_frames, 1), name="log_mel")
    x = inputs
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(conv_dropout)(x)

    # Compress channels so the sequence model stays small.
    x = layers.Conv2D(32, 1, activation="relu")(x)
    x = layers.Permute((2, 1, 3))(x)  # (time, mel, channels)
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="emotion_probs")(x)
    return keras.Model(inputs, outputs, name="cnn_bilstm_emotion")