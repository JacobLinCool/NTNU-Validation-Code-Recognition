""" Libraries """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
from best_model import id_to_word, loadMyBestModel
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='')
args = parser.parse_args()


""" Functions """
def main():
    model = loadMyBestModel()
    
    Xs = np.expand_dims(np.array([
        np.asarray(Image.open(args.image).convert('L').resize((216, 60)), dtype=float)
    ]), axis=3)

    Ys = model.predict(Xs)

    Ys = np.argmax(Ys, axis=2)
    Ys = np.array([ [ id_to_word[y] for y in Y ] for Y in Ys ])

    print(''.join(Ys[0]))


""" Execution """
if __name__ == "__main__":
    main()
