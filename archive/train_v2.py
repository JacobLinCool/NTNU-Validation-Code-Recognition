"""
This version tries cropping input images.
"""


import datetime
import numpy as np
import tensorflow as tf
from PIL import Image
from module import binarize


dataset_num = 200
class_num = 41
class_map = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13,
    'o': 14,
    'p': 15,
    'q': 16,
    'r': 17,
    's': 18,
    't': 19,
    'u': 20,
    'v': 21,
    'w': 22,
    'x': 23,
    'y': 24,
    'z': 25,
    '0': 26,
    '1': 27,
    '2': 28,
    '3': 29,
    '4': 30,
    '5': 31,
    '6': 32,
    '7': 33,
    '8': 34,
    '9': 35,
    '+': 36,
    '-': 37,
    '*': 38,
    '/': 39,
    '=': 40,
}
crop = [ 14, 33, 52, 71, 90 ]


def read_dataset():

    # X = [ np.asarray(Image.open(f"dataset/image_{i:05}.jpg")) for i in range(dataset_num) ]
    X = [ np.asarray(Image.open(f"dataset/image_{i:05}.jpg").convert('L')) for i in range(dataset_num) ]

    X = [ image[:, crop[i]:crop[i+1]] for i in range(4) for image in X ]
    Y = [ ]
    for i in range(dataset_num):
        with open(f"dataset/label_{i:05}.txt") as txt_file:
            labels = txt_file.readline()
            labels = [ class_map[word] for word in labels ]
            for label in labels: Y.append(label)
    X = np.array(X)
    X = np.expand_dims(X, axis=3)
    Y = np.array(Y)
    return X, Y


class Detector(tf.keras.layers.Layer):
    def __init__(self):
        super(Detector, self).__init__()
        self.detects = [
            tf.keras.models.Sequential([
                tf.keras.layers.Dense(2048, activation="relu"),
                tf.keras.layers.Dense(1028, activation="relu"),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(class_num, activation="softmax") 
            ]) for _ in range(4)
        ]

    def call(self, inputs):
        outputs = [ tf.expand_dims(detect(inputs), axis=1) for detect in self.detects ]
        outputs = tf.concat(outputs, 1)
        return outputs


if __name__ == "__main__":

    X, Y = read_dataset()

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(30, 19, 1)),

        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation=tf.nn.silu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same", activation=tf.nn.silu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding="same", activation=tf.nn.silu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(2048, activation="relu"),
        tf.keras.layers.Dense(1028, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),

        tf.keras.layers.Dense(class_num, activation="softmax") 
        # Detector()
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=10000,
        decay_rate=0.8)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary()

    log_dir = "train/" + datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)
    model.fit(X, Y, validation_split=0.2, batch_size=1, epochs=200, callbacks=[tensorboard_callback])