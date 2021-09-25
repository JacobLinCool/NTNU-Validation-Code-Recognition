"""
This version tries with another ImageNet, Xception.
And because of the limitation of Xception,
validate code images were resized from (30, 108) to (90, 324).
"""


import datetime
import numpy as np
import tensorflow as tf
from PIL import Image


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
  except RuntimeError as e:
    print(e)


class_num = 41
"""
word_to_id = {
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
resize_height, resize_width = 90, 324


def read_dataset(img_no):
    X = [ np.asarray(Image.open(f"dataset/image_{i:05}.jpg").convert('L').resize((resize_width, resize_height))) for i in range(img_no) ]
    Y = [ ]
    for i in range(img_no):
        with open(f"dataset/label_{i:05}.txt") as txt_file:
            labels     = txt_file.readline()
            labels     = [ word_to_id[word] for word in labels ]
            Y.append(labels)
    X = np.array(X, dtype=np.float)
    X = np.expand_dims(X, axis=3)
    Y = np.array(Y)
    return X, Y


class Detector(tf.keras.layers.Layer):
    def __init__(self):
        super(Detector, self).__init__()
        self.denses = [ tf.keras.Sequential([
            # tf.keras.layers.Dense( 512, activation="relu"),
            tf.keras.layers.Dense( 256, activation="relu"),
            tf.keras.layers.Dense( 128, activation="relu"),
            tf.keras.layers.Dense(  64, activation="relu"),
        ]) for _ in range(4) ]
        self.detect = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, x):
        y = tf.concat([
            tf.expand_dims(self.detect(self.denses[i](x)), axis=1) for i in range(4)
        ], axis=1)
        # y = tf.concat([
        #     tf.expand_dims(self.detect(x), axis=1) for _ in range(4)
        # ], axis=1)
        return y


class MyModel(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(MyModel, self).__init__()
        self.img_net  = tf.keras.applications.Xception(
            include_top=True,
            weights=None,
            input_shape=(resize_height, resize_width, 1),
            classifier_activation=None,
        )
        self.dropout  = tf.keras.layers.Dropout(dropout_rate)
        self.detector = Detector()

    def call(self, x):
        return self.detector(self.dropout(self.img_net(x)))


if __name__ == "__main__":

    img_no       = 600
    dropout_rate = 0.87
    bs           = 16
    lr           = 1e-3
    dr           = 0.05
    log_dir = "train/resize/" + datetime.datetime.now().strftime(f"%d-%H.%M") + \
             f"_bs={bs}_lr={lr}_dr={dr}_img={img_no}_do={dropout_rate}"
    
    epochs   = 800
    patience = 40

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=10
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_sparse_categorical_accuracy", mode="max",
        verbose=1, patience=patience,
    )
    val_loss_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = log_dir + "/val_loss.h5", verbose=1,
        monitor="val_loss",
        save_best_only=True, save_weights_only=True,
    )
    val_acc_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = log_dir + "/val_acc.h5", verbose=1,
        monitor="val_sparse_categorical_accuracy",
        save_best_only=True, save_weights_only=True,
    )

    X, Y = read_dataset(img_no)

    model = MyModel(dropout_rate)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=dr)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.build(input_shape=(None, resize_height, resize_width, 1))
    model.summary()
    model.fit(X, Y, validation_split=0.2, batch_size=bs, epochs=epochs, callbacks=[
        tensorboard_callback, early_stopping, # val_loss_ckpt_callback, val_acc_ckpt_callback
    ])