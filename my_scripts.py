

import sys

import pprint

pp = pprint.PrettyPrinter()

if __name__ == "__main__":

    first_arg = sys.argv[0]

    params = flickr_params.copy()
    params.update(flickr_train_params)
    my_train(params, first_arg)

    params = cuhk_params_a_to_b.copy()
    params.update(cuhk_train_params)
    my_train(params, first_arg)

    params = cuhk_params_b_to_a.copy()
    params.update(cuhk_train_params)
    my_train(params, first_arg)

    test_cuhk_a_to_b(cuhk_params_b_to_a, first_arg, benchmark=True)
    test_cuhk_a_to_b(cuhk_params_a_to_b, first_arg, benchmark=True)


































