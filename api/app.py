from flask import Flask, request
import json
import joblib
import numpy as np
import pdb
from sklearn.preprocessing import normalize

app = Flask(__name__)

def load_model(model):
    if model=='lr':
        return joblib.load('models/M20AIE239_lr_lbfgs.joblib')
    elif model=='svc':
        return joblib.load('models/svc.joblib')
    elif model=='tree':
        return joblib.load('models/tree.joblib')
 
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict/<algo>", methods = ['GET'])
def predict(algo):
    X = normalize(np.array(request.get_json()['X']),copy=False)
    clf=load_model(algo)
    y_pred = clf.predict(X.flatten().reshape(1,-1))
    return 'Predicted value('+str(clf)+'): '+json.dumps(y_pred.tolist()[0])+'\n'