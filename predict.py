import numpy as np
import tensorflow as tf
from PIL import Image

from best_model import MyModel


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

Xs = np.expand_dims(np.array([
    np.asarray(Image.open(f"test/image_{i:05}.jpg").convert('L')) for i in range(30)
], dtype=np.float), axis=3)

model = MyModel(dropout_rate=0.97)
model.build(input_shape=(None, 30, 108, 1))
model.load_weights("train/23-00.55_bs=8_lr=0.001_dr=0.075_img=600_do=0.955/val_acc.h5")

Ys = model.predict(Xs)
Ys = np.argmax(Ys, axis=2)
Ys = np.array([ [ id_to_word[y] for y in Y ] for Y in Ys ])
for i in range(30):
    print(f"image_{i:05}:", Ys[i])