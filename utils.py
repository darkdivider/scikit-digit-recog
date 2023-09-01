from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# flatten function for sklearn digits
def flatten_X(X):
    return X.reshape(len(X),-1)


# training on data based on training parameters
def train(data, model_parameters, model='svc'):
    # data is a tuple containing X:inputs, and y:targets
    X, y = data
    # This model is now created only for svc
    if model=='svc':
        # classifier is initiated as an svm with model parameters
        clf=svm.SVC(**model_parameters)
    else:
        print(f'Model name "{model}" recieved. Not found.')
        return
    # fit the data and return the classifier
    clf.fit(X, y)
    return clf

def split_train_dev_test(X, y, test_size = 0.25, dev_size = 0.25):
    # display a set of samples
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, X, y):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

    # hyperparameters
    train_parameters = {'svc':{'gamma':0.001},
                        'split':{'test_size':0.66,
                                'shuffle':True}}

    # data split for testing and training
    X_train, X_test, y_train, y_test = train_test_split(flatten_X(X),
                                                        y, 
                                                        **train_parameters['split'])

    # train the classifier of train dataset
    clf = train((X_train, y_train), train_parameters['svc'])

    predict_and_eval(clf, X_test, y_test)

    return clf

def predict_and_eval(model, X_test, y_test):
    # Predict the value of the digit on the test subset
    predicted = model.predict(X_test)

    # display prediction samples
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    # classification metrics
    print(f"Classification report for classifier (test) {model}:\n"
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