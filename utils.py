import sys

from train import train
from test import test
from env import env_name
from options.train_options import TrainOptions
from options.test_options import TestOptions
from shutil import copyfile
import os


if env_name == "raph":
    cycle_batch_size = "1"
    pix2pix_batch_size = "8"

elif env_name == "hind":
    cycle_batch_size = "1"
    pix2pix_batch_size = "8"

elif env_name == "compute_engine":
    cycle_batch_size = "64"
    pix2pix_batch_size = "16"

cuhk = {"n_epochs": 400, "save_epoch_freq": "100"}
flickr = {"n_epochs": 1000, "save_epoch_freq": "100"}
celeba = {"n_epochs": 20, "save_epoch_freq": "1"}

netg = "unet_256"

base_params = {
    "input_nc": "3",
    "output_nc": "3",
    "netG": netg
}

cuhk_pix2pix_params = {
    "dataset_mode": "unaligned",
    "dataroot": "my_data/cuhk",
    "model": "pix2pix",
    "batch_size": pix2pix_batch_size,
    "name": "cuhk_pix2pix"
}

cuhk_cycle_params = {
    "dataroot": "my_data/cuhk",
    "model": "pix2pix",
    "batch_size": cycle_batch_size,
    "name": "cuhk_cycle"
}

cuhk_pix2pix_train_params = {
    "save_epoch_freq": cuhk["save_epoch_freq"],
    "niter_decay": str(int(cuhk["n_epochs"] / 2)),
    "niter": str(int(cuhk["n_epochs"] / 2)),
    "lambda_L1": "10.0"
}

cuhk_cycle_train_params = {
    "save_epoch_freq": cuhk["save_epoch_freq"],
    "niter_decay": str(int(cuhk["n_epochs"] / 2)),
    "niter": str(int(cuhk["n_epochs"] / 2)),
    "lambda_A": "10.0",
    "lambda_B": "10.0",
    "lambda_identity": " 0"
}

cuhk_pix2pix_params.update(base_params)
cuhk_cycle_params.update(base_params)

celeba_cycle_params = {
    "dataroot": "my_data/celeba",
    "model": "cycle_gan",
    "batch_size": cycle_batch_size,
    "name": "celeba_cycle"
}

celeba_pix2pix_params = {
    "dataroot": "my_data/celeba",
    "model": "cycle_gan",
    "batch_size": pix2pix_batch_size,
    "name": "celeba_pix2pix"
}

celeba_pix2pix_train_params = {
    "niter_decay": str(int(celeba["n_epochs"] / 2)),
    "niter": str(int(celeba["n_epochs"] / 2)),
    "save_epoch_freq": "1",
    "lambda_L1": "10.0"
}

celeba_cycle_train_params = {
    "niter_decay": str(int(celeba["n_epochs"] / 2)),
    "niter": str(int(celeba["n_epochs"] / 2)),
    "save_epoch_freq": "1",
    "lambda_A": "10.0",
    "lambda_B": "10.0",
    "lambda_identity": " 0"
}

celeba_pix2pix_params.update(base_params)
celeba_cycle_params.update(base_params)

flickr_params = {
    "dataroot": "my_data/flickr",
    "name": "flickr",
    "model": "cycle_gan",
    "batch_size": cycle_batch_size
}

flickr_train_params = {
    "save_epoch_freq": "1",
    "niter_decay": str(int(flickr["n_epochs"] / 2)),
    "niter": str(int(flickr["n_epochs"] / 2)),
}

flickr_params.update(base_params)


def set_argv(params, first_arg):
    ans = [first_arg]
    for key, value in params.items():
        if not isinstance(value, bool):
            ans.append("--" + key)
            ans.append(value)
        else:
            if value:
                ans.append("--" + key)
    return ans


def my_train(params, first_arg):

    sys.argv = set_argv(params, first_arg)
    opt = TrainOptions().parse()
    train(opt)


def my_test(params, first_arg, benchmark=False, results_dir="benchmark_results"):
    current_params = params.copy()
    current_params["num_test"] = "100000"

    if benchmark:
        current_params["dataroot"] = "my_data/benchmark"
        current_params["results_dir"] = results_dir
        current_params["dataset_mode"] = "unaligned"

    sys.argv = set_argv(current_params, first_arg)

    opt = TestOptions().parse()
    test(opt)


def create_env_file():
    content = """env_name = "compute_engine" """
    f = open("env.py", "w")

    f.write(content)
    f.close()


def copy_generator(origin="AtoB", model_to_import="celeba"):
    path1 = os.path.join("checkpoints", model_to_import + "_pix2pix_" + origin, "latest_net_G.pth")
    if origin == "AtoB":
        target = "latest_net_G_A.pth"
    else:
        target = "latest_net_G_B.pth"

    path2 = os.path.join("checkpoints", "flickr", target)
    copyfile(path1, path2)


if __name__ == "__main__":
    create_env_file()
