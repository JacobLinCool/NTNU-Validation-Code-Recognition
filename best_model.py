weights_file_path = "path/to/val_loss.h5"


# ================================================== #


import tensorflow as tf
from module import Mish


class_num = 41
word_to_id = {
    'b': 0,
    '7': 1,
    'e': 2,
    '*': 3,
    'g': 4,
    '0': 5,
    'i': 6,
    'c': 7,
    'k': 8,
    '9': 9,
    '+': 10,
    'z': 11,
    'l': 12,
    'r': 13,
    'w': 14,
    '=': 15,
    '1': 16,
    'n': 17,
    'o': 18,
    '3': 19,
    't': 20,
    'x': 21,
    'p': 22,
    '5': 23,
    '8': 24,
    'v': 25,
    'h': 26,
    '-': 27,
    's': 28,
    'd': 29,
    'm': 30,
    '4': 31,
    'j': 32,
    'u': 33,
    'q': 34,
    'f': 35,
    'a': 36,
    '/': 37,
    'y': 38,
    '6': 39,
    '2': 40,
}
id_to_word = {
    0 : 'b',
    1 : '7',
    2 : 'e',
    3 : '*',
    4 : 'g',
    5 : '0',
    6 : 'i',
    7 : 'c',
    8 : 'k',
    9 : '9',
    10: '+',
    11: 'z',
    12: 'l',
    13: 'r',
    14: 'w',
    15: '=',
    16: '1',
    17: 'n',
    18: 'o',
    19: '3',
    20: 't',
    21: 'x',
    22: 'p',
    23: '5',
    24: '8',
    25: 'v',
    26: 'h',
    27: '-',
    28: 's',
    29: 'd',
    30: 'm',
    31: '4',
    32: 'j',
    33: 'u',
    34: 'q',
    35: 'f',
    36: 'a',
    37: '/',
    38: 'y',
    39: '6',
    40: '2',
}


class Detector(tf.keras.layers.Layer):
    def __init__(self):
        super(Detector, self).__init__()
        self.denses = [ tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=Mish()),
            tf.keras.layers.Dense(32, activation=Mish()),
            tf.keras.layers.Dense(16, activation=Mish()),
            tf.keras.layers.Dense( 8, activation=Mish()),
        ]) for _ in range(4) ]
        self.detect = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, x):
        y = tf.concat([
            tf.expand_dims(self.detect(self.denses[i](x)), axis=1) for i in range(4)
        ], axis=1)
        return y


def loadMyBestModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(  32, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(  64, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D( 128, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D( 256, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D( 512, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(1024, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.93),
        Detector(),
    ])
    model.build(input_shape=(None, 60, 216, 1))
    model.load_weights(weights_file_path)
    return model