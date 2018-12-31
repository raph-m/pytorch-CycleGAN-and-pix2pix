import sys
from utils import my_train, celeba_params_a_to_b, celeba_train_params, \
    celeba_params_b_to_a, my_test

if __name__ == "__main__":
    first_arg = sys.argv[0]

    params = celeba_params_a_to_b.copy()
    params.update(celeba_train_params)
    my_train(params, first_arg)

    my_test(celeba_params_a_to_b, first_arg, benchmark=True, results_dir="benchmark_results")



