import os

from PIL import Image


def get_file_paths(folder, also_filenames=False):
    image_file_paths = []
    my_filenames = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        my_filenames = filenames.copy()
        print(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_file_paths.append(file_path)

        break  # prevent descending into subfolders
    if also_filenames:
        return image_file_paths, my_filenames
    return image_file_paths


def align_images(a_file_paths, b_file_paths, target_path, mode="RGB"):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for i in range(len(a_file_paths)):

        img_a = Image.open(a_file_paths[i])
        img_b = Image.open(b_file_paths[i])

        if mode == "grayscale":
            img_a = Image.open(a_file_paths[i]).convert('L')

        if mode == "RGB":
            img_b = Image.open(b_file_paths[i]).convert("RGB")

        assert(img_a.size == img_b.size)

        aligned_image = Image.new("RGB" if mode == "RGB" else "L", (img_a.size[0] * 2, img_a.size[1]))
        aligned_image.paste(img_a, (0, 0))
        aligned_image.paste(img_b, (img_a.size[0], 0))
        aligned_image.save(os.path.join(target_path, '{:04d}.jpg'.format(i)))


if __name__ == '__main__':

    dataset_folder = os.path.join("cuhk")
    print(dataset_folder)

    for phase in ["train", "test"]:

        a_path = os.path.join(dataset_folder, phase + 'A')
        b_path = os.path.join(dataset_folder, phase + 'B')

        a_file_paths = get_file_paths(a_path)
        b_file_paths = get_file_paths(b_path)

        assert(len(a_file_paths) == len(b_file_paths))
        test_path = os.path.join(dataset_folder, phase)
        align_images(a_file_paths, b_file_paths, test_path)
