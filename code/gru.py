import tensorflow as tf
import numpy as np

gru_model = tf.keras.Sequential(
    [
        tf.keras.layers.GRU(200),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(300),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)