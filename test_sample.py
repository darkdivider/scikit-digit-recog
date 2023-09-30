import utils

def test_hparam_combinations():
    Cs=[0.25, 0.5, 1, 2, 4, 8, 16]
    gammas = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.0016]
    h_param_combs = utils.gen_hparams(Cs, gammas)
    assert (1,0.001) in h_param_combs
    assert (0.5,0.002) in h_param_combs
    