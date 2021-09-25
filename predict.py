""" Libraries """
import numpy as np
import tensorflow as tf
from PIL import Image
from best_model import id_to_word, loadMyBestModel


""" Functions """
def main():
    
    model = loadMyBestModel()
    
    Xs = np.expand_dims(np.array([
        np.asarray(Image.open(f"test/image_{i:05}.jpg").convert('L')) for i in range(30)
    ], dtype=np.float), axis=3)

    Ys = model.predict(Xs)
    Ys_confidences = np.max(Ys, axis=2)

    Ys = np.argmax(Ys, axis=2)
    Ys = np.array([ [ id_to_word[y] for y in Y ] for Y in Ys ])

    np.set_printoptions(formatter={'all':lambda x: f"{x:<5.2}"})
    for i in range(30): print(f"image_{i:05}:", Ys_confidences[i], Ys[i])
    

""" Execution """
if __name__ == "__main__":
    main()