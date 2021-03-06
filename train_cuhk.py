import sys
from utils import my_train, cuhk_cycle_params, cuhk_cycle_train_params, cuhk_pix2pix_params, cuhk_pix2pix_train_params, my_test
from utils import my_train, flickr_train_params, flickr_params, my_test, copy_generator

if __name__ == "__main__":
    first_arg = sys.argv[0]

    params = cuhk_pix2pix_params.copy()
    params.update(cuhk_pix2pix_train_params)
    my_train(params, first_arg)

    cuhk_pix2pix_params["epoch"] = "1000"
    my_test(cuhk_pix2pix_params, first_arg, benchmark=True, results_dir="benchmark_results")

    params = cuhk_cycle_params.copy()
    params.update(cuhk_cycle_train_params)
    my_train(params, first_arg)

    cuhk_cycle_params["epoch"] = "1000"
    my_test(cuhk_cycle_params, first_arg, benchmark=True, results_dir="benchmark_results")




