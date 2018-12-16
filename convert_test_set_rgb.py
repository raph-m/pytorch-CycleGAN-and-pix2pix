import os

from PIL import Image


def get_file_paths(folder):
    image_file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_file_paths.append(file_path)

        break  # prevent descending into subfolders
    return image_file_paths


if __name__ == '__main__':

    dataset_folder = os.path.join("datasets", "cuhk")
    print(dataset_folder)

    for phase in ["trainB", "testB", "valB"]:

        b_path = os.path.join(dataset_folder, phase)

        print(b_path)

        b_file_paths = get_file_paths(b_path)

        for (i, filepath) in enumerate(b_file_paths):
            img_b = Image.open(filepath).convert("RGB")
            img_b.save(filepath)























































