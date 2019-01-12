import sys
from utils import my_train, flickr_train_params, flickr_params, my_test, copy_networks

if __name__ == "__main__":

    do_import = True

    first_arg = sys.argv[0]

    if do_import:
        copy_networks(model_to_import="celeba_cycle", iter="2")
        flickr_train_params["continue_train"] = True
        flickr_params["name"] = "flickr_import"

    params = flickr_params.copy()
    params.update(flickr_train_params)
    my_train(params, first_arg)

    my_test(flickr_params, first_arg, benchmark=True, results_dir="benchmark_results")


