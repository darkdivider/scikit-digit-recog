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
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

# utility functions
import utils

digits = datasets.load_digits()

# display a set of samples
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# hyperparameters
train_parameters = {'svc':{'gamma':0.001},
                    'split':{'test_size':0.66,
                             'shuffle':True}}

# data split for testing and training
X_train, X_test, y_train, y_test = train_test_split(utils.flatten_X(digits),
                                                    digits.target, 
                                                    **train_parameters['split'])

# train the classifier of train dataset
clf = utils.train((X_train, y_train), train_parameters['svc'])

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

# display prediction samples
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# classification metrics
print(f"Classification report for classifier (test) {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")

# display confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix(test)")
print(f"Confusion matrix(test):\n{disp.confusion_matrix}")
plt.show()

# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix(test):\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
