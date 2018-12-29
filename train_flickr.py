import sys
from utils import my_train, flickr_train_params, flickr_params, my_test

if __name__ == "__main__":
    first_arg = sys.argv[0]

    for i in range(5):

        params = flickr_params.copy()
        params.update(flickr_train_params)
        my_train(params, first_arg)

        my_test(flickr_params, first_arg, benchmark=True, results_dir="benchmark_results" + str(i))

        flickr_params["continue_train"] = True


