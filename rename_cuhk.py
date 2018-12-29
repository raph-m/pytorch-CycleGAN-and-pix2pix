import os

# take dataset as you find it on internet. rename the folders you want to trainA, trainB, testA... remove the "photos"
# folder.
# run this code
# then I rearanged the train/test repartition
# then run "my_align_script"

for setup in ["train", "test"]:
    print("*" * 20)
    print(setup)

    for filename in os.listdir(os.path.join("cuhk", setup + "B")):

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
            os.path.join("cuhk", setup + "B", filename),
            os.path.join("cuhk", setup + "B", new_file_name_with_ext)
        )

for setup in ["train", "test"]:
    for filename in os.listdir(os.path.join("cuhk", setup + "B")):
        assert os.path.exists(os.path.join("cuhk", setup + "A", filename)),\
            os.path.join("cuhk", setup + "A", filename)
