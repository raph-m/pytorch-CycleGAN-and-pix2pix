import sys
from utils import celeba_pix2pix_params, celeba_cycle_params
from utils import set_argv
from options.test_options import TestOptions
from test import test

if __name__ == "__main__":
    first_arg = sys.argv[0]

    """
    current_params = celeba_pix2pix_params.copy()
    current_params["epoch"] = "5"
    current_params["num_test"] = "10000"
    current_params["dataroot"] = "my_data/celeba"
    current_params["results_dir"] = "inception_results_epoch5"
    current_params["dataset_mode"] = "unaligned"
    sys.argv = set_argv(current_params, first_arg)
    opt = TestOptions().parse()
    test(opt)
    """

    current_params = celeba_pix2pix_params.copy()
    current_params["epoch"] = "10"
    current_params["num_test"] = "10000"
    current_params["dataroot"] = "my_data/celeba"
    current_params["results_dir"] = "inception_results"
    current_params["dataset_mode"] = "unaligned"
    sys.argv = set_argv(current_params, first_arg)
    opt = TestOptions().parse()
    test(opt)

    current_params = celeba_cycle_params.copy()
    current_params["epoch"] = "5"  # TODO: check this
    current_params["num_test"] = "10000"
    current_params["dataroot"] = "my_data/celeba"
    current_params["results_dir"] = "inception_results_epoch5"
    current_params["dataset_mode"] = "unaligned"
    sys.argv = set_argv(current_params, first_arg)
    opt = TestOptions().parse()
    test(opt)


