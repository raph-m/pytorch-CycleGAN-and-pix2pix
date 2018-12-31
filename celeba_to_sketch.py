from my_align_script import get_file_paths
import os
from PIL import Image
import cv2

import numpy as np

np.random.seed(0)

paths, names = get_file_paths(os.path.join("my_data", "celeba", "trainA"), also_filenames=True)

if not os.path.exists(os.path.join("my_data", "celeba", "train")):
    os.makedirs(os.path.join("my_data", "celeba", "train"))

if not os.path.exists(os.path.join("my_data", "celeba", "trainB")):
    os.makedirs(os.path.join("my_data", "celeba", "trainB"))

import time

start = time.time()

for i in range(len(paths)):

    if i % 1000 == 0:
        print(i / 1000)
        if i == 1000:
            end = time.time()
            print("time for 1000 it.: ", end - start)
        """Applies pencil sketch effect to an RGB image
            :param img_rgb: RGB image to be processed
            :returns: Processed RGB image
        """

        img_a = Image.open(paths[i])

        img_rgb = cv2.imread(paths[i])
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0)
        img_blend = cv2.divide(img_gray, img_blur, scale=256)

        img_b = np.array(img_blend)

        img_b = Image.fromarray(img_b).convert("RGB")

        aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
        aligned_image.paste(img_a, (0, 0))
        aligned_image.paste(img_b, (img_a.size[0], 0))

        aligned_path = os.path.join("my_data", "celeba", "train", names[i])
        aligned_image.save(aligned_path)

        unaligned_path = os.path.join("my_data", "celeba", "trainB", names[i])
        img_b.save(unaligned_path)

    if i > 30:
        break

