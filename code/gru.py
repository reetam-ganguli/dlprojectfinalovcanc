import tensorflow as tf
import numpy as np

gru_conv_model = tf.keras.Sequential(
    [
        tf.keras.layers.GRU(150),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Reshape((30, 5)),
        tf.keras.layers.Conv1D(32, 5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

gru_model = tf.keras.Sequential(
    [
        tf.keras.layers.GRU(150),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

conv_gru_model = tf.keras.Sequential(
    [
        tf.keras.layers.Reshape((103, 2)),
        tf.keras.layers.Conv1D(9, 5, input_shape=(103, 2)),
        tf.keras.layers.GRU(150),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

conv_model = tf.keras.Sequential(
    [
        tf.keras.layers.Reshape((103, 2)),
        tf.keras.layers.Conv1D(9, 5, input_shape=(103, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)
