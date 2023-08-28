from sklearn import svm

# flatten function for sklearn digits
def flatten_X(digits):
    return digits.images.reshape(len(digits.images),-1)

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