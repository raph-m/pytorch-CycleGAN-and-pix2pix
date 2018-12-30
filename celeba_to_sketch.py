from my_align_script import get_file_paths
import os
from PIL import Image

import numpy as np
from skimage import filters

np.random.seed(0)

paths, names = get_file_paths(os.path.join("my_data", "celeba", "trainA"), also_filenames=True)

if not os.path.exists(os.path.join("my_data", "celeba", "train")):
    os.makedirs(os.path.join("my_data", "celeba", "train"))

if not os.path.exists(os.path.join("my_data", "celeba", "trainB")):
    os.makedirs(os.path.join("my_data", "celeba", "trainB"))

for i in range(len(paths)):

    if i % 1000 == 0:
        print(i)

    img_a = Image.open(paths[i])
    img_b = np.array(img_a.convert("L"))

    sigma = np.random.rand() * 0.05 + 0.8
    img_b = filters.gaussian(img_b, sigma=sigma)

    img_b = filters.sobel(img_b)
    img_b = 1 - img_b

    sigma = np.random.rand() * 0.1 + 0.1
    percentile = np.random.rand() * 10 + 10

    img_b = filters.gaussian(img_b, sigma=sigma)

    img_b = img_b * (img_b > np.percentile(img_b, percentile))
    img_b = (255 * img_b).astype(int)

    img_b = Image.fromarray(img_b).convert("RGB")

    aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
    aligned_image.paste(img_a, (0, 0))
    aligned_image.paste(img_b, (img_a.size[0], 0))

    aligned_path = os.path.join("my_data", "celeba", "train", names[i])
    aligned_image.save(aligned_path)

    unaligned_path = os.path.join("my_data", "celeba", "trainB", names[i])
    img_b.save(unaligned_path)

