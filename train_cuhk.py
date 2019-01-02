import sys
from utils import my_train, cuhk_cycle_params, cuhk_cycle_train_params, cuhk_pix2pix_params, cuhk_pix2pix_train_params, my_test

if __name__ == "__main__":
    first_arg = sys.argv[0]

    params = cuhk_cycle_params.copy()
    params.update(cuhk_cycle_train_params)
    my_train(params, first_arg)

    my_test(cuhk_cycle_params, first_arg, benchmark=True, results_dir="benchmark_results")

    params = cuhk_pix2pix_params.copy()
    params.update(cuhk_pix2pix_train_params)
    my_train(params, first_arg)

    my_test(cuhk_cycle_params, first_arg, benchmark=True, results_dir="benchmark_results")




