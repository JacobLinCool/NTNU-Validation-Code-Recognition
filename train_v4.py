"""
This version used the wrong layer.
I should have use CSPBottleneck but not Bottleneck.
Also, this version tries to different the last 4 detecting processes.
"""


import datetime
import numpy as np
import tensorflow as tf
from PIL import Image
from module import MyConv, Bottleneck


dataset_num = 200
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
english = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' ]
numbers = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '=' ]


def read_dataset():
    X = [ np.asarray(Image.open(f"dataset/image_{i:05}.jpg").convert('L')) for i in range(dataset_num) ]
    Y = [ ]
    for i in range(dataset_num):
        with open(f"dataset/label_{i:05}.txt") as txt_file:
            labels     = txt_file.readline()
            eng_or_num = 0 if labels[0] in english else 1
            labels     = [ class_map[word] for word in labels ]
            # Y.append([eng_or_num, *labels])
            Y.append(labels)
    X = np.array(X)
    X = np.expand_dims(X, axis=3)
    Y = np.array(Y)
    return X, Y


class Detector(tf.keras.layers.Layer):
    def __init__(self):
        super(Detector, self).__init__()

        self.lstm = tf.keras.layers.LSTM(256)
        self.denses = [ tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
        ]) for _ in range(4) ]

        self.eng_or_num = tf.keras.layers.Dense( 2, activation="softmax")
        self.eng        = tf.keras.layers.Dense(26, activation="softmax")
        self.num        = tf.keras.layers.Dense(10, activation="softmax")
        self.sym        = tf.keras.layers.Dense( 5, activation="softmax")

    def call(self, x):
        x = self.lstm(x)
        zero_engs = tf.zeros(shape=(tf.shape(x)[0], 26), dtype=tf.float32)
        zero_nums = tf.zeros(shape=(tf.shape(x)[0], 10), dtype=tf.float32)
        zero_syms = tf.zeros(shape=(tf.shape(x)[0],  5), dtype=tf.float32)
        eng_or_num = self.eng_or_num(x)
        eng_or_num = tf.math.argmax(eng_or_num, axis=1, output_type=tf.int32)

        xs = [ self.denses[i](x) for i in range(4) ]

        y = tf.cond(
            tf.equal(eng_or_num, 0),
            lambda:
                tf.concat([
                    tf.expand_dims(tf.concat([self.eng(xs[i]), zero_nums, zero_syms], axis=1), axis=1) for i in range(4)
                ], axis=1),
            lambda:
                tf.concat([
                    tf.expand_dims(tf.concat([zero_engs, self.num(xs[0]), zero_syms], axis=1), axis=1),
                    tf.expand_dims(tf.concat([zero_engs, zero_nums, self.sym(xs[1])], axis=1), axis=1),
                    tf.expand_dims(tf.concat([zero_engs, self.num(xs[2]), zero_syms], axis=1), axis=1),
                    tf.expand_dims(tf.concat([zero_engs, zero_nums, self.sym(xs[3])], axis=1), axis=1),
                ], axis=1)
        )

        # if tf.equal(eng_or_num, 0):
        #     y = tf.concat([
        #         tf.expand_dims(tf.concat([self.eng[i](x), zero_nums, zero_syms], axis=1), axis=1) for i in range(4)
        #     ], axis=1)
        # else:
        #     y = tf.concat([
        #         tf.expand_dims(tf.concat([zero_engs, self.num[0](x), zero_syms], axis=1), axis=1),
        #         tf.expand_dims(tf.concat([zero_engs, zero_nums, self.sym[0](x)], axis=1), axis=1),
        #         tf.expand_dims(tf.concat([zero_engs, self.num[1](x), zero_syms], axis=1), axis=1),
        #         tf.expand_dims(tf.concat([zero_engs, zero_nums, self.sym[1](x)], axis=1), axis=1),
        #     ], axis=1)

        return y


if __name__ == "__main__":

    bs = 1
    lr = 1e-6
    dr = 0.2
    log_dir = "train/" + datetime.datetime.now().strftime(f"%d-%H.%M_bs={bs}_lr={lr}_dr={dr}")

    X, Y = read_dataset()
    print(X.shape)
    print(Y.shape)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(30, 108, 1)),

        MyConv(32, kernel_size=3, strides=1),
        MyConv(64, kernel_size=3, strides=2),
        *[ Bottleneck(64) for _ in range(1)],
        MyConv(128, kernel_size=3, strides=2),
        *[ Bottleneck(128) for _ in range(3)],
        MyConv(256, kernel_size=3, strides=2),
        *[ Bottleneck(256) for _ in range(15)],
        MyConv(512, kernel_size=3, strides=2),
        *[ Bottleneck(512) for _ in range(15)],
        MyConv(1024, kernel_size=3, strides=2),
        *[ Bottleneck(1024) for _ in range(7)],

        tf.keras.layers.Reshape((4, 1024)),
        tf.keras.layers.Dropout(0.7),

        Detector()
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=dr)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=5)
    model.fit(X, Y, validation_split=0.2, batch_size=bs, epochs=200, callbacks=[tensorboard_callback])