import os

for setup in ["train", "test"]:
    print("*" * 20)
    print(setup)

    for filename in os.listdir(os.path.join("datasets", "cuhk", setup + "B")):

        filename_without_ext = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]

        new_file_name = filename_without_ext[:-4]

        # dont ask why but there is capital letters in the trainB set
        if new_file_name[0] == 'M':
            new_file_name = 'm' + new_file_name[2:]

        if new_file_name[0] == 'F':
            new_file_name = 'f' + new_file_name[2:]

        new_file_name_with_ext = new_file_name + extension
        os.rename(
            os.path.join("datasets", "cuhk", setup + "B", filename),
            os.path.join("datasets", "cuhk", setup + "B", new_file_name_with_ext)
        )

for setup in ["train", "test"]:
    for filename in os.listdir(os.path.join("datasets", "cuhk", setup + "B")):
        assert os.path.exists(os.path.join("datasets", "cuhk", setup + "A", filename)),\
            os.path.join("datasets", "cuhk", setup + "A", filename)
