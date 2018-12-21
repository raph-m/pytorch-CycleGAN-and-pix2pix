from train import train
from test import test
from env import local_params
from options.train_options import TrainOptions
from options.test_options import TestOptions

import sys

import pprint

pp = pprint.PrettyPrinter()


def set_argv(params, first_arg):
    ans = [first_arg]
    for key, value in params.items():
        ans.append("--" + key)
        if not isinstance(value, bool):
            ans.append(value)
    return ans


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
    "dataroot": "gdrive/datasets/cuhk",
    "name": "cuhk_pix2pix_AtoB",
    "model": "pix2pix",
    "netG": netg,
    "batch_size": local_params["cuhk"]["batch_size"]
}

cuhk_train_params = {
    "save_epoch_freq": "5",
    "niter_decay": "5",
    "niter": "5",
    "no_lsgan": True
}

cuhk_params.update(base_params)
cuhk_params_a_to_b = {"direction": "AtoB"}
cuhk_params_a_to_b.update(cuhk_params)
cuhk_params_b_to_a = {"direction": "BtoA"}
cuhk_params_b_to_a.update(cuhk_params)

flickr_params = {
    "dataroot": "gdrive/datasets/flickr",
    "name": "flickr",
    "model": "cycle_gan",
    "netG": netg,
    "batch_size": local_params["cuhk"]["batch_size"],
}

flickr_train_params = {
    "save_epoch_freq": "1",
    "niter_decay": "5",
    "niter": "5",
    "no_lsgan": True
}

flickr_params.update(base_params)


def my_train(params, first_arg):

    sys.argv = set_argv(params, first_arg)
    opt = TrainOptions().parse()
    train(opt)


def test_cuhk_a_to_b(params, first_arg, benchmark=False):
    current_params = params.copy()

    if benchmark:
        current_params["dataroot"] = "gdrive/datasets/benchmark"
        current_params["results_dir"] = "benchmark_results"
        current_params["dataset_mode"] = "unaligned"

    sys.argv = set_argv(current_params, first_arg)

    opt = TestOptions().parse()
    test(opt)


if __name__ == "__main__":

    first_arg = sys.argv[0]

    params = cuhk_params_a_to_b.copy()
    params.update(cuhk_train_params)
    my_train(params, first_arg)

    params = cuhk_params_b_to_a.copy()
    params.update(cuhk_train_params)
    my_train(params, first_arg)

    test_cuhk_a_to_b(cuhk_params_a_to_b, first_arg, benchmark=False)
    test_cuhk_a_to_b(cuhk_params_b_to_a, first_arg, benchmark=True)
































