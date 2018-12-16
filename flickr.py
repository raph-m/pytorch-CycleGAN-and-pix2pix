import flickr_api
per_page = 500
n_errors = 0


def download_flickr(query, directory, n_pages=1, per_page=500):
    n_errors = 0
    for page in range(1, n_pages + 1):
        print("page = ", page)
        results = flickr_api.Photo.search(
            text=query, sort="relevance", per_page=per_page, page=page
        )

        for i, result in enumerate(results):
            if i % 50 == 0:
                print("i = ", i)
            try:
                result.save(os.path.join(directory, str((page - 1) * per_page + i)))
            except:
                n_errors += 1
                print("error, n_errors: ", n_errors)

        print("")
        print("")


import os
from convert_test_set_rgb import get_file_paths
from PIL import Image
def convert_directory(directory, target_format="L"):
    paths = get_file_paths(directory)

    for (i, filepath) in enumerate(paths):
        print(filepath)
        img_b = Image.open(filepath).convert(target_format)
        img_b.save(filepath)

    return

directory_b = os.path.join("datasets", "flickr", "trainB")
convert_directory(directory_b, target_format="RGB")

if __name__ == "__main__":
    query = "sketch portrait charcoal drawing"
    directory_a = os.path.join("datasets", "flickr", "trainA")
    download_flickr(query, directory_a, n_pages=10, per_page=500)

    query = "portrait"
    directory_b = os.path.join("datasets", "flickr", "trainB")
    download_flickr(query, directory_b, n_pages=10, per_page=500)

    convert_directory(directory_a, target_format="L")


