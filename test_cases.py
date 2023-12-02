import utils
from sklearn.linear_model import LogisticRegression
import joblib

filename = 'models/M20AIE239_lr_lbfgs.joblib'

def test_clf_isLR():
    clf=joblib.load(filename)
    assert isinstance(clf, LogisticRegression)
    assert clf.get_params()['solver'] in filename
    
def test_solver_in_filename():
    clf=joblib.load(filename)
    assert clf.get_params()['solver'] in filename