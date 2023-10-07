import utils

def test_hparam_combinations():
    Cs=[0.25, 0.5, 1, 2, 4, 8, 16]
    gammas = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.0016]
    h_param_combs = utils.gen_hparams(Cs, gammas)
    assert (1,0.001) in h_param_combs
    assert (0.5,0.002) in h_param_combs

def test_data-splitting():
    X,y = datasets.load_digits()
    X=X[:100,:,:]

def test_model_saving():
    pass

def create_dummy_data():
    pass

def tune_params(X_train, y_train, X_dev, y_dev, h_params_combinations):
    best_accuracy=-1
    best_model_path=''
    for h_params in h_params_combinations:
        pass

