import sys
from utils import my_train, celeba_params_a_to_b, celeba_train_params, \
    celeba_params_b_to_a, my_test

if __name__ == "__main__":
    first_arg = sys.argv[0]

    # train AtoB
    params = celeba_params_b_to_a.copy()
    params.update(celeba_train_params)

    params["direction"] = "AtoB"
    params["save_latest_freq"] = str(4992 * 5)

    my_train(params, first_arg)

    # train BtoA
    params = celeba_params_b_to_a.copy()
    params.update(celeba_train_params)

    params["direction"] = "BtoA"
    params["continue_train"] = True
    params["load_iter"] = "1307904"

    my_train(params, first_arg)

    # my_test(celeba_params_b_to_a, first_arg, benchmark=True, results_dir="benchmark_results")



