import sys
from utils import my_train, celeba_cycle_params, celeba_cycle_train_params, celeba_pix2pix_params,\
    celeba_pix2pix_train_params, my_test

if __name__ == "__main__":
    first_arg = sys.argv[0]

    # train pix2pix
    params = celeba_pix2pix_params.copy()
    params.update(celeba_pix2pix_train_params)

    my_train(params, first_arg)

    celeba_pix2pix_params["epoch"] = "5"
    #my_test(celeba_pix2pix_params, first_arg, benchmark=True, results_dir="train_results_5")

    celeba_pix2pix_params["epoch"] = "10"
    #my_test(celeba_pix2pix_params, first_arg, benchmark=True, results_dir="train_results_10")




