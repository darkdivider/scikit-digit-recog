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
from sklearn.metrics import confusion_matrix

# utility functions
import utils
import pickle

# download dataset
digits = datasets.load_digits()
print(f'Number of total samples in dataset: {digits.target.__len__()}');
print(f'size of images: {digits.images[0].shape}')

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

# hyperparameter tuning
test_acc=0
for test_size, dev_size in test_dev_combs:
    print(f'test_size: {test_size:0.3f}, dev_size={dev_size:0.3f}, train_size:{1-dev_size-test_size:0.3f}')
    X_train, X_dev, X_test, y_train, y_dev, y_test = utils.train_dev_test_split(digits.images, digits.target, test_size, dev_size)
    model, best_h_params = utils.tune_params(X_train, y_train, X_dev, y_dev, h_param_combs)
    new_test_acc = utils.check_acc(model, X_test, y_test)
    if test_acc<new_test_acc:
        test_acc = new_test_acc
        with open('models/svc.pkl','wb') as f:
            pickle.dump(model,f)
    print(f'test_acc: {utils.check_acc(model, X_test, y_test):0.3f}')
    print('----------------------------------------------------')
