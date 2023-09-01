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

# get trained model
# the predict_and_eval is incorporated in the split_train.. in utils.
model = utils.split_train_dev_test(digits.images, digits.target, )

