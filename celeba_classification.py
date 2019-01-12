import numpy as np
import cv2
import os

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import SGD

from scipy.stats import entropy
from my_align_script import get_file_paths

# set variables
main_folder = 'my_data/celeba/'
images_folder = main_folder + 'trainA/'

EXAMPLE_PIC = images_folder + '000506.jpg'

TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
NUM_EPOCHS = 20

# Import InceptionV3 Model
inc_model = InceptionV3(
    # weights="my_data/celeba/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

print("number of layers:", len(inc_model.layers))

# Adding custom Layers
x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)


# creating the final model 
model_ = Model(inputs=inc_model.input, outputs=predictions)

# Lock initial layers to do not be trained
for layer in model_.layers[:52]:
    layer.trainable = False

# compile the model
model_.compile(
    optimizer=SGD(lr=0.0001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_.load_weights(os.path.join(main_folder, 'weights.best.inc.male.hdf5'))


def gender_prediction(filename):
    """
    predict the gender
    
    input:
        filename: str of the file name
        
    return:
        array of the prob of the targets.
    
    """

    im = cv2.imread(filename)
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (178, 218)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis=0)

    # prediction
    result = model_.predict(im)

    return result


experiments = ["pix2pix_5_epochs", "pix2pix_10_epochs", "cycle"]

results_dir = {
    "pix2pix_5_epochs": "my_data/celeba/testA/",
    "pix2pix_10_epochs": "my_data/celeba/testA/",
    "cycle": "my_data/celeba/testA/",
}

for experiment in experiments:
    print("experiment: ", experiment)

    paths, fnames = get_file_paths(results_dir[experiment], also_filenames=True)

    preds = np.zeros((len(paths), 2))

    for i in range(len(paths)):
        pred = gender_prediction(paths[i])
        preds[i, :] = pred[0]

    fnames = np.array(fnames)

    # Now compute the mean kl-div
    splits = 10
    split_scores = []
    n = len(paths)

    for k in range(splits):
        part = preds[k * (n // splits): (k + 1) * (n // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    print("avg split score: ", np.mean(split_scores))
    print("std split score: ", np.std(split_scores))

    max_prob = np.max(preds, axis=1)
    sorted_indexes = np.argsort(max_prob)
    worst_inception_score = fnames[sorted_indexes[:5]]
    best_inception_score = fnames[sorted_indexes[-5:]]

    print("best images are: ", best_inception_score)
    print("worst images are: ", worst_inception_score)
    print("")
    print("")
