import sys
from utils import my_train, flickr_train_params, flickr_params, my_test, copy_generator

if __name__ == "__main__":
    first_arg = sys.argv[0]

    import_celeba = True

    for i in range(2):
        results_dir = "benchmark_results"

        if import_celeba:
            results_dir += "_import_celeba"
            copy_generator(origin="AtoB", model_to_import="celeba")
            copy_generator(origin="BtoA", model_to_import="celeba")
            flickr_train_params["continue_train"] = True

        params = flickr_params.copy()
        params.update(flickr_train_params)

        params["display_freq"] = "100"
        params["dataroot"] = "my_data/celeba"

        if i == 0:
            params["lambda_A"] = "1.0"
            params["lambda_B"] = "1.0"
            params["lambda_identity"] = "0"

        else:
            params["lambda_A"] = "10.0"
            params["lambda_B"] = "10.0"
            params["lambda_identity"] = " 0.5"

        if (not import_celeba) and i == 0:
            params["continue_train"] = False
        else:
            params["continue_train"] = True

        my_train(params, first_arg)

        my_test(flickr_params, first_arg, benchmark=True, results_dir=results_dir + str(i))


