from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import os

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
    elif model=='tree':
        # classifier is initiated as an tree with model parameters
        clf = dtc(**model_parameters)
        # pdb.set_trace()
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

def get_next_comb(comb_index, lens):
    for i in range(len(lens)):
        if comb_index[i]<lens[i]-1:
            comb_index[i]+=1
            while i:
                i-=1
                comb_index[i]=0
            return comb_index
    return None

def get_combinations(params):
    lens = [len(param) for param in params]
    comb_index=[0 for _ in lens]
    comb_indexes = [comb_index]
    
    while True:
        comb_index=get_next_comb(list(comb_index), lens)
        if comb_index!=None:
            comb_indexes.append(comb_index)
        else:
            break
    return [[params[i][j] for i,j in enumerate(comb_indexes[k])] for k in range(len(comb_indexes))]

def gen_hparams(params_list):
    return get_combinations(params_list)

def train_dev_test_split(X, y, test_size, dev_size, shuffle):
    X_train, X_test, y_train, y_test = train_test_split(flatten_X(X),y, test_size=test_size+dev_size, shuffle=shuffle)
    X_test, X_dev,y_test, y_dev = train_test_split(X_test, y_test, test_size=test_size/(test_size+dev_size), shuffle=shuffle)
    # pdb.set_trace()
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def tune_params(X, y, test_size, dev_size, all_param_combs, model_type = 'svc', param_key = ['C', 'gamma'], shuffle=False):
    if not os.path.exists('models'):
        os.makedirs('models')
    filename = 'models/accs_'+model_type+'.pkl'
    accs = {'train_acc':[],'dev_acc':[],'test_acc':[]}
    for _ in range(10):
        X_train, X_dev, X_test, y_train, y_dev, y_test = train_dev_test_split(X, y, test_size, dev_size, shuffle)
        dev_acc=0
        max_dev_acc=0
        best_clf = None
        best_h_params = None
        for param_comb in all_param_combs:
            model_params = {key:value for key,value in zip(param_key, param_comb)}
            clf = train((X_train, y_train), model_params, model_type)
            try:
                dev_acc = check_acc(clf, X_dev, y_dev)
            except:
                print('error')
                # pdb.set_trace()
            if dev_acc>max_dev_acc:
                max_dev_acc = dev_acc
                best_clf = clf
                best_h_params=model_params
            # print(f'params:{param_comb} ,dev_acc:{dev_acc:0.3f}, best_acc = {max_dev_acc:0.3f}     best_params = {best_h_params}')
        train_acc = check_acc(clf, X_train, y_train)
        # print(f'{model_type}->params:{best_h_params}, train_acc: {train_acc:0.3f}, dev_acc:{max_dev_acc:0.3f}', end = ', ')
        test_acc = check_acc(best_clf, X_test, y_test)
        # print(f'test_acc: {test_acc:0.3f}')
        accs['train_acc'].append(train_acc)
        accs['dev_acc'].append(dev_acc)
        accs['test_acc'].append(test_acc)
    print(model_type,end='\t')
    print(f"train\t :{mean(accs['train_acc']):0.3f} +/- {std(accs['train_acc']):0.3f}",end='\t')
    print(f"dev\t :{mean(accs['dev_acc']):0.3f} +/- {std(accs['dev_acc']):0.3f}",end='\t')
    print(f"test\t :{mean(accs['test_acc']):0.3f} +/- {std(accs['test_acc']):0.3f}")
    joblib.dump(accs, filename)
    joblib.dump(best_clf, 'models/'+model_type+'.pkl')
    return best_clf

def mean(L):
    return sum(L)/len(L)

def std(L):
    if len(L)<=1:
        return 0
    # import pdb;pdb.set_trace()
    m=mean(L)
    return (sum([(i-m)*(i-m) for i in L])/(len(L)-1))**0.5
