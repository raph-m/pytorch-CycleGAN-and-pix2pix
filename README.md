# Final Project for course "object Recognition" at MVA-2018 ENS Paris-Saclay
Hind Dadoun and Raphael Montaud. Please find the project report in `report.pdf`.

## Code usage
- python requirements in the `requirements.txt` file
- launch a Virtual Machine on Google Compute Engine with the preinstallation package "pytorch fast AI" and follow instructions in `setup.txt` for a quick setup
- we built the dataset with the scripts `celeba_to_sketch.py`, `convert_test_set_rgb.py`, `flickr.py` and `my_align_script.py` 
- `download_google_drive.py` enables to download quickly the database from our google drive link (ready to use)
- `train.py` and `test.py` are wrapper functions to train models and to test the generators
- please find the training scripts in the `train_....py` files, at the end of a training, we generally process the "benchmark" folder with images from the 3 different datasets
- the `utils.py` file contains some useful methods like "copy_networks" that enables to import networks from a model to another
- in `produce_image_for_inception_score.py` we use three generators to produce 10k generated images
- in `celeba_classification.py` we make predictions on the generated images with a gender classifer and compute the inception score, as well as the best and worst generated images for each model 

## Acknowledgments
Most of the training framework is adapted from junyanz/pytorch-CycleGAN-and-pix2pix. Please note that we also used this kaggle kernel: https://www.kaggle.com/bmarcos/image-recognition-gender-detection-inceptionv3/data for the gender predictions on the celebA dataset. Also, our implementation of the inception score was inspired by: https://github.com/sbarratt/inception-score-pytorch.
