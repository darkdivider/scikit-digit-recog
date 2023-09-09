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

def split_train_dev_test(X, y, test_size = 0.25, dev_size = 0.25,):
    # display a set of samples
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, X, y):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

    # hyperparameters
    train_parameters = {'svc':{'gamma':0.001},
                            'split_train_test':{'test_size':test_size + dev_size,
                                    'shuffle':True},
                            'split_dev_test':{'test_size':test_size/(dev_size+test_size),
                                            "shuffle": True}}

    # data split for testing and training
    X_train, X_test, y_train, y_test = train_test_split(flatten_X(X),
                                                        y, 
                                                        **train_parameters['split_train_test'])
    
    X_dev, X_test, y_dev, y_test = train_test_split(flatten_X(X),
                                                        y, 
                                                        **train_parameters['split_dev_test'])

    # train the classifier of train dataset
    clf = train((X_train, y_train), train_parameters['svc'])

    # predict_and_eval(clf, X_test, y_test)
    acc_train = check_acc(clf, X_train, y_train)
    acc_dev = check_acc(clf, X_dev, y_dev)
    acc_test = check_acc(clf, X_test, y_test)
    print(f'train_acc: {acc_train:0.3f}',end=' ')
    print(f'dev_acc: {acc_dev:0.3f}',end= '')
    print(f'test_acc: {acc_test:0.3f}')
    return clf

def check_acc(model, X_test, y_test):
    predicted = model.predict(X_test)
    return (predicted==y_test).sum()/len(y_test)

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

def gen_hparams(Cs, gammas):
    h_params = sum([[(c,gamma) for c in Cs] for gamma in gammas],[])
    return h_params

def train_dev_test_split(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = train_test_split(flatten_X(X),y, test_size=test_size+dev_size, shuffle=False)
    X_test, X_dev,y_test, y_dev = train_test_split(X_test, y_test, test_size=test_size/(test_size+dev_size), shuffle=False)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def tune_params(X_train, y_train, X_dev, y_dev, all_param_combs):
    max_dev_acc=0
    best_clf = None
    best_h_params = None
    for c, gamma in all_param_combs:
        model_params = {'C': c, 'gamma': gamma}
        clf = train((X_train, y_train), model_params)
        dev_acc = check_acc(clf, X_dev, y_dev)
        if dev_acc>max_dev_acc:
            max_dev_acc = dev_acc
            best_clf = clf
            best_h_params=(c, gamma)
        # print(f'c:{c}, gamma:{gamma}, dev_acc:{dev_acc:0.3f}, best_acc = {max_dev_acc:0.3f}')
    train_acc = check_acc(clf, X_train, y_train)
    print(f'best_h_params : c: {c}, gamma: {gamma},\ntrain_acc: {train_acc:0.3f}, dev_acc:{max_dev_acc:0.3f}', end = ', ')
    return best_clf, best_h_params


