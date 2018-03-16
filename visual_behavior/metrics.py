"""
the metrics contained in this module conform to scikit-learn styled `metric(y_true,y_pred,**args,**kwargs)`

"""
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from .utilities import dprime as __dprime


def hit_rate(y_true, y_pred):
    return y_pred[y_true].mean()


def false_alarm_rate(y_true, y_pred):
    return y_pred[~y_true].mean()


def d_prime(y_true, y_pred, eps=None):
    if len(y_true) > 0:
        if eps is None:
            # a reasonable minimum value is 1 / sum of trials
            eps = 1.0 / len(y_true)

        limits = (eps, 1 - eps)

        HR = hit_rate(y_true, y_pred)
        FAR = false_alarm_rate(y_true, y_pred)

        return __dprime(HR, FAR, limits)
    else:
        return np.nan


def discrim_p(y_true, y_pred):

    C = confusion_matrix(y_true, y_pred)

    try:
        return stats.chi2_contingency(C)[1]
    except ValueError:
        return 1.0
