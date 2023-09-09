"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Harikrishnan <nair dot 2 at iitj dot ac dot in>
# Courtest: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: MIT

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets

# utility functions
import utils

# download dataset
digits = datasets.load_digits()

# combination sof test sizes and dev sizes
test_sizes =[0.1, 0.2, 0.3];
dev_sizes = [0.1,0.2,0.3];
test_dev_combs=sum([[(test_size, dev_size)for test_size in test_sizes]for dev_size in dev_sizes],[])

# combination of hyperparamters
Cs=[0.25, 0.5, 1, 2, 4, 8, 16]
gammas = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.0016]
h_param_combs = utils.gen_hparams(Cs, gammas)

# hyperparameter tuning
for test_size, dev_size in test_dev_combs:
    print(f'test_size: {test_size:0.3f}, dev_size={dev_size:0.3f}, train_size:{1-dev_size-test_size:0.3f}')
    X_train, X_dev, X_test, y_train, y_dev, y_test = utils.train_dev_test_split(digits.images, digits.target, test_size, dev_size)
    model, best_h_params = utils.tune_params(X_train, y_train, X_dev, y_dev, h_param_combs)
    print(f'test_acc: {utils.check_acc(model, X_test, y_test):0.3f}')
    print('----------------------------------------------------')