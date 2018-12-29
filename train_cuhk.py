import sys
from utils import my_train, cuhk_params_a_to_b, cuhk_train_params, \
    cuhk_params_b_to_a, my_test

if __name__ == "__main__":
    first_arg = sys.argv[0]

    params = cuhk_params_a_to_b.copy()
    params.update(cuhk_train_params)
    my_train(params, first_arg)

    params = cuhk_params_b_to_a.copy()
    params.update(cuhk_train_params)
    my_train(params, first_arg)

    my_test(cuhk_params_b_to_a, first_arg, benchmark=True, results_dir="benchmark_results")
    my_test(cuhk_params_a_to_b, first_arg, benchmark=True, results_dir="benchmark_results")



