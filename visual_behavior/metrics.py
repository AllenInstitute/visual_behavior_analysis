"""
the metrics contained in this module conform to scikit-learn styled `metric(y_true,y_pred,**args,**kwargs)`

"""
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from .utilities import dprime as __dprime
from .utilities import trial_number_limit
from scipy.stats import norm


def N_go_trials(y_true, y_pred):
    return len(y_pred[y_true])


def N_catch_trials(y_true, y_pred):
    return len(y_pred[~y_true])


def hit_rate(y_true, y_pred, apply_trial_number_limit=False):
    go_responses = y_pred[y_true]
    response_probability = go_responses.mean()
    if apply_trial_number_limit:
        trial_count = N_go_trials(y_true, y_pred)
        return trial_number_limit(response_probability, trial_count)
    else:
        return response_probability


def false_alarm_rate(y_true, y_pred, apply_trial_number_limit=False):
    catch_responses = y_pred[~y_true]
    response_probability = catch_responses.mean()
    if apply_trial_number_limit:
        trial_count = N_catch_trials(y_true, y_pred)
        return trial_number_limit(response_probability, trial_count)
    else:
        return response_probability


def d_prime(y_true, y_pred, apply_trial_number_limit=False, eps=None):
    if len(y_true) > 0:
        if eps is None:
            # a reasonable minimum value is 1 / sum of trials
            eps = 1.0 / len(y_true)

        limits = (eps, 1 - eps)

        HR = hit_rate(y_true, y_pred, apply_trial_number_limit=apply_trial_number_limit)
        FAR = false_alarm_rate(y_true, y_pred, apply_trial_number_limit=apply_trial_number_limit)

        return __dprime(HR, FAR, limits)
    else:
        return np.nan


def discrim_p(y_true, y_pred):

    C = confusion_matrix(y_true, y_pred)

    try:
        return stats.chi2_contingency(C)[1]
    except ValueError:
        return 1.0


def criterion(hit_rate, false_alarm_rate):
    '''see chapter 2 of Macmillan Creelman Detection Theory'''
    Z = norm.ppf
    c = -0.5 * (Z(hit_rate) + Z(false_alarm_rate))

    return c
