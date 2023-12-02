"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Harikrishnan <nair dot 2 at iitj dot ac dot in>
# Courtesy: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: MIT

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import joblib

# utility functions
import utils
import pdb

# download dataset
digits = datasets.load_digits()
print(f'Number of total samples in dataset: {digits.target.__len__()}');
print(f'size of images: {digits.images[0].shape}')

# normalization of images
[normalize(image, copy=False) for image in digits.images]

# combination sof test sizes and dev sizes
test_sizes =[0.2];
dev_sizes = [0.2];
test_dev_combs=sum([[(test_size, dev_size)for test_size in test_sizes]for dev_size in dev_sizes],[])

h_param_combs={}
# combination of hyperparamters for svm
Cs=[1, 2, 4, 8]
gammas = [0.00025, 0.0005, 0.001, 0.002]
h_param_combs['svm'] = utils.gen_hparams([Cs, gammas])

# combination of hyperparamters for decision tree
criteria=['gini', 'entropy','log_loss']
max_depths = [2,4,6,8,16,32,64,128]
h_param_combs['tree'] = utils.gen_hparams([criteria, max_depths])

# combination of hyperparamters for logistic-regression
solvers=['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
h_param_combs['lr'] = utils.gen_hparams([solvers])

# hyperparameter tuning
for test_size, dev_size in test_dev_combs:
    print(f'test_size: {test_size:0.3f}, dev_size={dev_size:0.3f}, train_size:{1-dev_size-test_size:0.3f}') 
    print()
    clf_lr = utils.tune_params(digits.images, digits.target, test_size, dev_size, h_param_combs['lr'], 
                            'lr', ['solver'],True)
    clf_prod = joblib.load('models/svc.joblib')
    clf_new = joblib.load('models/tree.joblib')
    clf_new = joblib.load('models/lr.joblib')



