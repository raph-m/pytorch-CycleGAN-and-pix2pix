import sys

from train import train
from test import test
from env import env_name
from options.train_options import TrainOptions
from options.test_options import TestOptions

if env_name == "raph":
    cuhk = {"batch_size": "8", "n_epochs": 50}
    flickr = {"batch_size": "1", "n_epochs": 2}
    local_params = {"cuhk": cuhk, "flickr": flickr}

elif env_name == "hind":
    cuhk = {"batch_size": "8", "n_epochs": 5}
    flickr = {"batch_size": "1", "n_epochs": 2}
    local_params = {"cuhk": cuhk, "flickr": flickr}

elif env_name == "compute_engine":
    cuhk = {"batch_size": "8", "n_epochs": 100}
    flickr = {"batch_size": "32", "n_epochs": 10}
    local_params = {"cuhk": cuhk, "flickr": flickr}

netg = "resnet_9blocks"

base_params = {
    "input_nc": "3",
    "output_nc": "3",
    "no_dropout": True,
    "norm": "batch",
    "loadSize": "128",
    "fineSize": "128"
}

cuhk_params = {
    "dataset_mode": "aligned",
    "dataroot": "my_data/cuhk",
    "model": "pix2pix",
    "netG": netg,
    "batch_size": local_params["cuhk"]["batch_size"]
}

cuhk_train_params = {
    "save_epoch_freq": "5",
    "niter_decay": str(int(local_params["cuhk"]["n_epochs"] / 2)),
    "niter": str(int(local_params["cuhk"]["n_epochs"] / 2)),
    "no_lsgan": True,
    "continue_train": False
}

cuhk_params.update(base_params)
cuhk_params_a_to_b = {"direction": "AtoB", "name": "cuhk_pix2pix_AtoB"}
cuhk_params_a_to_b.update(cuhk_params)
cuhk_params_b_to_a = {"direction": "BtoA", "name": "cuhk_pix2pix_BtoA"}
cuhk_params_b_to_a.update(cuhk_params)

flickr_params = {
    "dataroot": "my_data/flickr",
    "name": "flickr",
    "model": "cycle_gan",
    "netG": netg,
    "batch_size": local_params["flickr"]["batch_size"],
}

flickr_train_params = {
    "save_epoch_freq": "1",
    "niter_decay": str(int(local_params["flickr"]["n_epochs"] / 2)),
    "niter": str(int(local_params["flickr"]["n_epochs"] / 2)),
    "no_lsgan": True,
    "continue_train": False
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


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) > 1:
        batch_size = str(sys.argv[1])
    else:
        batch_size = "2"

    create_env_file()
