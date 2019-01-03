import sys
from utils import my_train, celeba_cycle_params, celeba_cycle_train_params, celeba_pix2pix_params,\
    celeba_pix2pix_train_params, my_test

if __name__ == "__main__":
    first_arg = sys.argv[0]

    # train cycle
    params = celeba_cycle_params.copy()
    params.update(celeba_cycle_train_params)

    my_train(params, first_arg)

    my_test(celeba_cycle_params, first_arg, benchmark=True, results_dir="benchmark_results")



