"""
the metrics contained in this module conform to scikit-learn styled `metric(y_true,y_pred,**args,**kwargs)`

"""
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy import stats
from .utilities import dprime as __dprime
from .utilities import trial_number_limit
from scipy.stats import norm


def N_go_trials(y_true, y_pred):
    return len(y_pred[y_true])


def N_catch_trials(y_true, y_pred):
    return len(y_pred[~y_true])


def hit_rate(y_true, y_pred, apply_trial_number_limit=False, clip_vals=[0, 1]):
    go_responses = y_pred[y_true]
    response_probability = go_responses.mean()
    if apply_trial_number_limit:
        trial_count = N_go_trials(y_true, y_pred)
        return np.clip(trial_number_limit(response_probability, trial_count), clip_vals[0], clip_vals[1])
    else:
        return np.clip(response_probability, clip_vals[0], clip_vals[1])


def false_alarm_rate(y_true, y_pred, apply_trial_number_limit=False, clip_vals=[0, 1]):
    catch_responses = y_pred[~y_true]
    response_probability = catch_responses.mean()
    if apply_trial_number_limit:
        trial_count = N_catch_trials(y_true, y_pred)
        return np.clip(trial_number_limit(response_probability, trial_count), clip_vals[0], clip_vals[1])
    else:
        return np.clip(response_probability, clip_vals[0], clip_vals[1])


def d_prime(y_true, y_pred, apply_trial_number_limit=False, clip_vals=[0, 1]):
    if len(y_true) > 0:

        HR = hit_rate(y_true, y_pred, apply_trial_number_limit=apply_trial_number_limit, clip_vals=clip_vals)
        FAR = false_alarm_rate(y_true, y_pred, apply_trial_number_limit=apply_trial_number_limit, clip_vals=clip_vals)

        return __dprime(HR, FAR, limits=clip_vals)
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
