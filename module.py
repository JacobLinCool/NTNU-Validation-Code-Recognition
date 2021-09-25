import numpy as np
import tensorflow as tf
from PIL import Image


def binarize(input, threshold):
    output = np.array(input)
    for i, row in enumerate(input):
        for j, _ in enumerate(row):
            if input[i][j] < threshold: output[i][j] = 0
    return output


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


class Mish(tf.keras.layers.Layer):
    def forward(self, x):
        return x * tf.nn.softplus(x).tanh()


class MyConv(tf.keras.layers.Layer):
    def __init__(self, filter, kernel_size, strides):
        super(MyConv, self).__init__()
        self.cv  = tf.keras.layers.Conv2D(filter, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False)
        self.bn  = tf.keras.layers.BatchNormalization()
        self.act = tf.nn.silu

    def call(self, inputs):
        return self.act(self.bn(self.cv(inputs)))


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, filter, shortcut=True):
        super(Bottleneck, self).__init__(filter)
        self.cv1 = MyConv(filter, kernel_size=1, strides=1)
        self.cv2 = MyConv(filter, kernel_size=3, strides=1)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class MyBottleneck(tf.keras.layers.Layer):
    def __init__(self, filter, shortcut=True):
        super().__init__()
        self.cv  = MyConv(filter, kernel_size=3, strides=1)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv(x) if self.add else self.cv(x)


class CSPBottleneck(tf.keras.layers.Layer):
    def __init__(self, filter, n, shortcut=True):
        super().__init__()
        self.cv1 = MyConv(filter, kernel_size=1, strides=1)
        self.b   = [ Bottleneck(filter, shortcut) for _ in range(n) ]
        self.cv2 = tf.keras.layers.Conv2D(filter, kernel_size=1, strides=1, use_bias=False)
        self.cv3 = tf.keras.layers.Conv2D(filter, kernel_size=1, strides=1, use_bias=False)
        self.bn  = tf.keras.layers.BatchNormalization()
        self.act = tf.nn.leaky_relu
        self.cv4 = MyConv(filter, kernel_size=1, strides=1)

    def forward(self, x):
        y1 = self.cv1(x)
        for b in self.b: y1 = b(y1)
        y1 = self.cv2(y1)
        y2 = self.cv3(x)
        return self.cv4(self.act(self.bn(tf.concat([y1, y2], axis=1)), alpha=0.1))


class MyCSPBottleneck(tf.keras.layers.Layer):
    def __init__(self, filter, n=1, shortcut=True):
        super().__init__()
        self.cv1 = MyConv(filter, kernel_size=1, strides=1)
        self.b   = [ MyBottleneck(filter, shortcut) for _ in range(n) ]
        self.cv3 = tf.keras.layers.Conv2D(filter, kernel_size=1, strides=1, use_bias=False)
        self.bn  = tf.keras.layers.BatchNormalization()
        self.act = tf.nn.leaky_relu
        self.cv4 = MyConv(filter, kernel_size=1, strides=1)

    def forward(self, x):
        y1 = self.cv1(x)
        for b in self.b: y1 = b(y1)
        y2 = self.cv3(x)
        return self.cv4(self.act(self.bn(tf.concat([y1, y2], axis=1)), alpha=0.1))


class CSPBottleneck2(tf.keras.layers.Layer):
    def __init__(self, filter, n, shortcut=False):
        super(CSPBottleneck2, self).__init__()
        self.cv1 = MyConv(filter, kernel_size=1, strides=1)
        self.m   = [ Bottleneck(filter, shortcut) for _ in range(n) ]
        self.cv2 = tf.keras.layers.Conv2D(filter, kernel_size=1, strides=1, use_bias=False)
        self.bn  = tf.keras.layers.BatchNormalization()
        self.act = Mish()
        self.cv3 = MyConv(filter, kernel_size=1, strides=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(y1)
        for m in self.m: y1 = m(y1)
        return self.cv3(self.act(self.bn(tf.concat([y1, y2], axis=1))))


class CSPSPP(tf.keras.layers.Layer):
    def __init__(self, filter, k=(5, 9, 13)):
        super(CSPSPP, self).__init__()
        self.cv1 = MyConv(filter, kernel_size=1, strides=1)
        self.cv2 = tf.keras.layers.Conv2D(filter, kernel_size=1, strides=1, use_bias=False)
        self.cv3 = MyConv(filter, kernel_size=3, strides=1)
        self.cv4 = MyConv(filter, kernel_size=1, strides=1)
        self.m  = [ tf.keras.layers.MaxPool2D(pool_size=x, strides=1, padding="same") for x in k ]
        self.cv5 = MyConv(filter, kernel_size=1, strides=1)
        self.cv6 = MyConv(filter, kernel_size=3, strides=1)
        self.bn  = tf.keras.layers.BatchNormalization() 
        self.act = Mish()
        self.cv7 = MyConv(filter, kernel_size=1, strides=1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(tf.concat([x1] + [m(x1) for m in self.m], axis=1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(tf.concat([y1, y2], axis=1))))