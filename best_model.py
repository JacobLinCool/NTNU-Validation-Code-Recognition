import tensorflow as tf
from module import MyConv, MyCSPBottleneck


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
            tf.keras.layers.Dense(  64, activation="relu"),
            tf.keras.layers.Dense(  32, activation="relu"),
            tf.keras.layers.Dense(  16, activation="relu"),
        ]) for _ in range(4) ]
        self.detect = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, x):
        y = tf.concat([
            tf.expand_dims(self.detect(self.denses[i](x)), axis=1) for i in range(4)
        ], axis=1)
        return y


class MyModel(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(MyModel, self).__init__()
        self.cv    = MyConv(32, kernel_size=3, strides=1)
        self.cv_p1 = MyConv(64, kernel_size=3, strides=2)    # (30, 108,   64)
        self.bn_p1 = MyCSPBottleneck(64, 1)
        self.cv_p2 = MyConv(128, kernel_size=3, strides=2)   # (15,  54,  128)
        self.bn_p2 = MyCSPBottleneck(128, 3)
        self.cv_p3 = MyConv(256, kernel_size=3, strides=2)   # ( 8,  27,  256)
        self.bn_p3 = MyCSPBottleneck(256, 15)
        self.cv_p4 = MyConv(512, kernel_size=3, strides=2)   # ( 4,  14,  512)
        self.bn_p4 = MyCSPBottleneck(512, 15)
        self.cv_p5 = MyConv(1024, kernel_size=3, strides=2)  # ( 2,   7, 1024)
        self.bn_p5 = MyCSPBottleneck(1024, 7)
        self.cv_p6 = MyConv(2048, kernel_size=3, strides=2)  # ( 1,   4, 2048)
        self.bn_p6 = MyCSPBottleneck(2048, 7)
        self.flatten  = tf.keras.layers.Flatten()
        self.dropout  = tf.keras.layers.Dropout(dropout_rate)
        self.detector = Detector()

    def call(self, x):
        x = self.bn_p1(self.cv_p1(self.cv(x)))
        x = self.bn_p2(self.cv_p2(x))
        x = self.bn_p3(self.cv_p3(x))
        x = self.bn_p4(self.cv_p4(x))
        x = self.bn_p5(self.cv_p5(x))
        x = self.bn_p6(self.cv_p6(x))
        y = self.flatten(self.dropout(x))
        y = self.detector(y)
        return y


def loadMyBestModel():
    model = MyModel(dropout_rate=0.955)
    model.build(input_shape=(None, 60, 216, 1))
    model.load_weights("train/resize_2/25-22.35_lr=0.0015_dr=0.05_do=0.955_bs=16_img=600/val_acc.h5")