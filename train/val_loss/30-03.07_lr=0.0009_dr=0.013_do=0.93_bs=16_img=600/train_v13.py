"""
Test
"""


import os
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image
from shutil import copy
from module import Mish


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
resize_height, resize_width = 60, 216


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_train_begin(self, logs=None):
        copy(__file__, self.filepath)
        self.best_loss = np.Inf
        self.best_acc  = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if np.less(current_loss, self.best_loss): self.best_loss = current_loss
        current_acc  = logs.get("val_sparse_categorical_accuracy")
        if np.greater(current_acc, self.best_acc): self.best_acc = current_acc

    def on_train_end(self, logs=None):
        loss = int(self.best_loss*10000) / 10000
        acc  = int(self.best_acc*10000) / 100
        try:
            os.rename(self.filepath, f"{self.filepath}_acc={acc}%_loss={loss}")
        except:
            with open(f"{self.filepath}/acc={acc}%_loss={loss}.txt", "w") as _: pass


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
    print(X.shape)
    Y = np.array(Y)
    return X, Y


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


def train(epochs, patience, do, bs, lr, dr, img_no, log_dir):
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=dr)

    copy_file_callback = MyCallback(filepath=log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=10
    )
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_sparse_categorical_accuracy", mode="max",
    #     verbose=1, patience=patience,
    # )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min",
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

    model = tf.keras.Sequential([
        # batch, 60, 216,    1
        tf.keras.layers.Conv2D(  32, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        # batch, 30, 108,   32
        tf.keras.layers.Conv2D(  64, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        # batch, 15,  54,   64
        tf.keras.layers.Conv2D( 128, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        # batch,  8,  27,  128
        tf.keras.layers.Conv2D( 256, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        # batch,  4,  14,  256
        tf.keras.layers.Conv2D( 512, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        # batch,  2,   7,  512
        tf.keras.layers.Conv2D(1024, 3, strides=1, padding="same", activation=tf.nn.silu),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        # batch,  1,   4, 1024
        tf.keras.layers.Flatten(),
        # batch,  4096
        tf.keras.layers.Dropout(rate=do),
        Detector(),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.build(input_shape=(None, resize_height, resize_width, 1))
    model.summary()
    model.fit(X, Y, validation_split=0.2, batch_size=bs, epochs=epochs, callbacks=[
        copy_file_callback, tensorboard_callback, early_stopping,
        val_loss_ckpt_callback, val_acc_ckpt_callback
    ])


if __name__ == "__main__":

    learning_rate_list = [  9e-4 ]
    decay_rate_list    = [ 0.012, 0.013, 0.014 ]
    dropout_rate_list  = [  0.93 ]
    bactch_size_list   = [    16 ]
    
    img_no   = 600
    epochs   = 600
    patience = 60

    list_len = max(len(dropout_rate_list), len(bactch_size_list), len(learning_rate_list), len(decay_rate_list))

    for i in range(list_len):

        print(f"\nTrain round: {i+1}/{list_len}\n")

        do = dropout_rate_list[i]  if len(dropout_rate_list)  == list_len else dropout_rate_list[0]
        bs = bactch_size_list[i]   if len(bactch_size_list)   == list_len else bactch_size_list[0]
        lr = learning_rate_list[i] if len(learning_rate_list) == list_len else learning_rate_list[0]
        dr = decay_rate_list[i]    if len(decay_rate_list)    == list_len else decay_rate_list[0]

        log_dir = "train/test_val_loss/" + datetime.datetime.now().strftime(f"%d-%H.%M") + \
                 f"_lr={lr}_dr={dr}_do={do}_bs={bs}_img={img_no}"
        train(epochs, patience, do, bs, lr, dr, img_no, log_dir)