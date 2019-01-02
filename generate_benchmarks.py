import sys
import os
from utils import my_train, celeba_params_a_to_b, celeba_train_params, \
    celeba_params_b_to_a, my_test

if __name__ == "__main__":
    first_arg = sys.argv[0]

    params = celeba_params_b_to_a.copy()
    params.update(celeba_train_params)

    # 4992

    for i in range(1, 11):
        n_iter = str(4992 * 40)
        results_dir = os.path.join("celeba_benchmark", n_iter)
        params["load_iter"] = n_iter
        my_test(celeba_params_b_to_a, first_arg, benchmark=True, results_dir=results_dir)

