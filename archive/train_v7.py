"""
This version tries with FPN just like YOLOv4.
And because of FPN, the size of input images are better 2^x.
Therefore, I did not resize but pad the original images from (30, 108) to (32, 128).
"""


import datetime
import numpy as np
import tensorflow as tf
from PIL import Image
from module import add_margin, MyConv, CSPBottleneck, CSPSPP, CSPBottleneck2


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
"""
class_map = {
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
"""


def read_dataset(img_no):
    X = [ np.asarray(add_margin(Image.open(f"dataset/image_{i:05}.jpg").convert('L'), 1, 10, 1, 10, 256)) for i in range(img_no) ]
    Y = [ ]
    for i in range(img_no):
        with open(f"dataset/label_{i:05}.txt") as txt_file:
            labels     = txt_file.readline()
            labels     = [ class_map[word] for word in labels ]
            Y.append(labels)
    X = np.array(X, dtype=np.float)
    X = np.expand_dims(X, axis=3)
    Y = np.array(Y)
    return X, Y


class Detector(tf.keras.layers.Layer):
    def __init__(self):
        super(Detector, self).__init__()
        self.denses = [ tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
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
        """ Backbone """
        self.cv_bb    = MyConv(32, kernel_size=3, strides=1)
        self.cv_bb_p1 = MyConv(64, kernel_size=3, strides=2)  # (15, 54, 64)
        self.bn_bb_p1 = CSPBottleneck(64, 1)
        self.cv_bb_p2 = MyConv(128, kernel_size=3, strides=2)  # (8, 27, 128)
        self.bn_bb_p2 = CSPBottleneck(128, 3)
        self.cv_bb_p3 = MyConv(256, kernel_size=3, strides=2)  # (4, 14, 256)
        self.bn_bb_p3 = CSPBottleneck(256, 15)
        self.cv_bb_p4 = MyConv(512, kernel_size=3, strides=2)  # (2, 7, 512)
        self.bn_bb_p4 = CSPBottleneck(512, 15)
        self.cv_bb_p5 = MyConv(1024, kernel_size=3, strides=2)  # (1, 4, 1024)
        self.bn_bb_p5 = CSPBottleneck(1024, 7)

        """ Upsample """
        self.cspspp = CSPSPP(512)  # (1, 4, 512)

        self.cv_us_p4_1 = MyConv(256, kernel_size=1, strides=1)
        self.up_us_p4   = tf.keras.layers.UpSampling2D()         # (2, 8, 256)
        self.cv_us_p4_2 = MyConv(256, kernel_size=1, strides=1)
        self.bn2_us_p4  = CSPBottleneck2(256, 3)

        self.cv_us_p3_1 = MyConv(128, kernel_size=1, strides=1)
        self.up_us_p3   = tf.keras.layers.UpSampling2D()         # (4, 16, 128)
        self.cv_us_p3_2 = MyConv(128, kernel_size=1, strides=1)
        self.bn2_us_p3  = CSPBottleneck2(128, 3)

        # self.cv_us_p2_1 = MyConv(64, kernel_size=1, strides=1)
        # self.up_us_p2   = tf.keras.layers.UpSampling2D()         # (8, 32, 64)
        # self.cv_us_p2_2 = MyConv(64, kernel_size=1, strides=1)
        # self.bn2_us_p2  = CSPBottleneck2(64, 3)

        # self.cv_us_p1_1 = MyConv(32, kernel_size=1, strides=1)
        # self.up_us_p1   = tf.keras.layers.UpSampling2D()         # (16, 64, 32)
        # self.cv_us_p1_2 = MyConv(32, kernel_size=1, strides=1)
        # self.bn2_us_p1  = CSPBottleneck2(32, 3)

        """ Downsample """
        # self.cv_ds_p2_1 = MyConv(64, kernel_size=1, strides=2)  # (8, 32, 64)
        # self.bn2_ds_p2  = CSPBottleneck2(64, 3)

        # self.cv_ds_p3_1 = MyConv(128, kernel_size=1, strides=2)  # (4, 16, 128)
        # self.bn2_ds_p3  = CSPBottleneck2(128, 3)

        self.cv_ds_p4_1 = MyConv(256, kernel_size=1, strides=2)  # (2, 8, 256)
        self.bn2_ds_p4  = CSPBottleneck2(256, 3)

        self.cv_ds_p5_1 = MyConv(512, kernel_size=1, strides=2)  # (1, 4, 512)
        self.bn2_ds_p5  = CSPBottleneck2(512, 3)
        self.cv_ds_p5_2 = MyConv(1024, kernel_size=1, strides=1)  # (1, 4, 1024)

        """ Detector """
        self.flatten  = tf.keras.layers.Flatten()
        self.dropout  = tf.keras.layers.Dropout(dropout_rate)
        self.detector = Detector()


    def call(self, x):

        """ Backbone """
        bb_p1 = self.bn_bb_p1(self.cv_bb_p1(self.cv_bb(x)))
        bb_p2 = self.bn_bb_p2(self.cv_bb_p2(bb_p1))
        bb_p3 = self.bn_bb_p3(self.cv_bb_p3(bb_p2))
        bb_p4 = self.bn_bb_p4(self.cv_bb_p4(bb_p3))
        bb_p5 = self.bn_bb_p5(self.cv_bb_p5(bb_p4))

        """ Upsample """
        up_p5 = self.cspspp(bb_p5)
        up_p4 = self.bn2_us_p4(tf.concat([
            self.up_us_p4(self.cv_us_p4_1(up_p5)), self.cv_us_p4_2(bb_p4)
        ], axis=-1))
        up_p3 = self.bn2_us_p3(tf.concat([
            self.up_us_p3(self.cv_us_p3_1(up_p4)), self.cv_us_p3_2(bb_p3)
        ], axis=-1))
        # up_p2 = self.bn2_us_p2(tf.concat([
        #     self.up_us_p2(self.cv_us_p2_1(up_p3)), self.cv_us_p2_2(bb_p2)
        # ], axis=-1))
        # up_p1 = self.bn2_us_p1(tf.concat([
        #     self.up_us_p1(self.cv_us_p1_1(up_p2)), self.cv_us_p1_2(bb_p1)
        # ], axis=-1))

        """ Downsample """
        # ds_p2 = self.bn2_ds_p2(tf.concat([
        #     self.cv_ds_p2_1(up_p1), up_p2
        # ], axis=-1))
        # ds_p3 = self.bn2_ds_p3(tf.concat([
        #     self.cv_ds_p3_1(ds_p2), up_p3
        # ], axis=-1))
        ds_p4 = self.bn2_ds_p4(tf.concat([
            self.cv_ds_p4_1(up_p3), up_p4
            # self.cv_ds_p4_1(ds_p3), up_p4
        ], axis=-1))
        ds_p5 = self.bn2_ds_p5(tf.concat([
            self.cv_ds_p5_1(ds_p4), up_p5
        ], axis=-1))
        ds_p5 = self.cv_ds_p5_2(ds_p5)
        
        """ Detector """
        y  = self.flatten(self.dropout(ds_p5))
        y  = self.detector(y)
        return y


if __name__ == "__main__":

    img_no       = 600
    dropout_rate = 0.975
    bs = 16
    lr = 2e-3
    dr = 0.06
    log_dir = "train/" + datetime.datetime.now().strftime(f"%d-%H.%M_bs={bs}_lr={lr}_dr={dr}_img={img_no}_do={dropout_rate}")

    X, Y = read_dataset(img_no)
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    model = MyModel(dropout_rate)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=dr)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.build(input_shape=(None, 32, 128, 1))
    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10)
    val_loss_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        log_dir + "/val_loss.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    val_acc_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        log_dir + "/val_acc.h5",
        monitor="val_sparse_categorical_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    model.fit(X, Y, validation_split=0.2, batch_size=bs, epochs=1000, callbacks=[
        tensorboard_callback, val_loss_ckpt_callback, val_acc_ckpt_callback
    ])