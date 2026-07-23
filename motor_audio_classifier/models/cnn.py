"""2D CNN that classifies a single Mel-spectrogram segment as OK / NG."""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
    Layer,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Recall

from config import (
    N_MELS,
    MEL_HOP,
    LEARNING_RATE,
    L2_REG,
    CONV_DROPOUT,
    DENSE_DROPOUT,
)


class SpecAugment(Layer):
    """Lightweight frequency + time masking (SpecAugment) for regularization.

    Applied only during training. Masks a random band of mel bins and a random
    stretch of time frames with zeros, which forces the network to rely on the
    whole spectrogram rather than memorizing a single region -- critical for a
    small dataset where a plain CNN otherwise overfits to noise.
    """

    def __init__(self, freq_mask=8, time_mask=12, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def call(self, x, training=None):
        if not training:
            return x
        f = tf.shape(x)[1]
        t = tf.shape(x)[2]

        # Frequency mask.
        f0 = tf.random.uniform([], 0, tf.maximum(f - self.freq_mask, 1),
                               dtype=tf.int32)
        fm = tf.minimum(self.freq_mask, f - f0)
        x = tf.concat(
            [x[:, :f0, :, :],
             tf.zeros_like(x[:, f0:f0 + fm, :, :]),
             x[:, f0 + fm:, :, :]],
            axis=1,
        )

        # Time mask.
        t0 = tf.random.uniform([], 0, tf.maximum(t - self.time_mask, 1),
                               dtype=tf.int32)
        tm = tf.minimum(self.time_mask, t - t0)
        x = tf.concat(
            [x[:, :, :t0, :],
             tf.zeros_like(x[:, :, t0:t0 + tm, :]),
             x[:, :, t0 + tm:, :]],
            axis=2,
        )
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"freq_mask": self.freq_mask, "time_mask": self.time_mask})
        return cfg


def build_model(input_shape=(N_MELS, MEL_HOP, 1)):
    """Build and compile the binary classification CNN.

    Regularized (SpecAugment + L2 + dropout + batch norm) to fight the severe
    overfitting observed on this small dataset, and the ``recall_ng`` metric
    tracks how often a faulty (NG) motor is correctly caught -- the quantity
    we must never let collapse.
    """
    reg = l2(L2_REG)
    model = Sequential(
        [
            Input(input_shape),
            SpecAugment(freq_mask=6, time_mask=10),
            Conv2D(32, (3, 3), activation="relu", kernel_regularizer=reg),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(CONV_DROPOUT),
            Conv2D(64, (3, 3), activation="relu", kernel_regularizer=reg),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(CONV_DROPOUT),
            Conv2D(128, (3, 3), activation="relu", kernel_regularizer=reg),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(CONV_DROPOUT),
            Flatten(),
            Dense(128, activation="relu", kernel_regularizer=reg),
            Dropout(DENSE_DROPOUT),
            Dense(1, activation="sigmoid"),
        ],
        name="motor_cnn",
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Recall(name="recall_ng"),  # recall of the NG (positive) class
        ],
    )
    return model
