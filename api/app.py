from flask import Flask, request
import json
import pickle
import numpy as np
import pdb

app = Flask(__name__)
model = pickle.load(open('svc.pkl', 'rb'))

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods = ['POST'])
def predict():
    data = request.get_json()
    X = np.array(data['X'])
    y_pred = model.predict(X.flatten().reshape(1,-1))
    return 'Predicted value: '+json.dumps(y_pred.tolist()[0])+'\n'
