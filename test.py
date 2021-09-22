import os
import numpy as np
from PIL import Image
from train import binarize


if not os.path.exists("test"): os.makedirs("test")
for i in range(10):

    # Image.fromarray(binarize(np.asarray(Image.open(f"dataset/image_{i:05}.jpg").convert('L')))).save(f"test/image_{i:05}.jpg")

    image_array = np.asarray(Image.open(f"dataset/image_{i:05}.jpg"))
    Image.fromarray(image_array[:, 14: 33]).save(f"test/image_{i:05}_1.jpg")
    Image.fromarray(image_array[:, 33: 52]).save(f"test/image_{i:05}_2.jpg")
    Image.fromarray(image_array[:, 52: 71]).save(f"test/image_{i:05}_3.jpg")
    Image.fromarray(image_array[:, 71: 90]).save(f"test/image_{i:05}_4.jpg")