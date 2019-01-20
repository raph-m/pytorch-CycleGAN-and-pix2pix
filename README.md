# Final Project for course "object Recognition" at MVA-2018 ENS Paris-Saclay
Hind Dadoun and Raphael Montaud 
Please find the project report in `report.pdf`.

## Code usage
- python requirements in the `requirements.txt` file
- launch a Virtual Machine on Google Compute Engine with the preinstallation package "pytorch fast AI" and follow instructions in `setup.txt` for a quick setup
- `download_google_drive.py` enables to download quickly the database from our google drive link
- please find the training scripts in the `train_....py` files

## Acknowledgments
Most of the training framework is adapted from junyanz/pytorch-CycleGAN-and-pix2pix. Please note that we also used this kaggle kernel: https://www.kaggle.com/bmarcos/image-recognition-gender-detection-inceptionv3/data to the gender predictions on the celebA dataset. Also, our implementation of the inception score war inspired by: https://github.com/sbarratt/inception-score-pytorch.
