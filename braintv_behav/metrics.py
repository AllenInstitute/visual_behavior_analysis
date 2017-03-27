## sklearn-style discrimination metrics

import numpy as np
from sklearn import metrics
from scipy import stats

from braintv_behav import masks
from braintv_behav.utilities import get_response_rates

def d_prime(y_true,y_pred,eps=None):

    C = metrics.confusion_matrix(y_true,y_pred).astype(float) 
    
    try:
        tn, fp, fn, tp = C.ravel()

        HR = tp / (tp + fn)
        FAR = fp / (tn + fp)

        if eps is None:
            # a reasonable minimum value is 1 / sum of trials
            eps = 1.0/C.sum()

        HR = max(HR,eps)
        HR = min(HR,1.0-eps)

        FAR = max(FAR,eps)
        FAR = min(FAR,1.0-eps)

        return stats.norm.ppf(HR) - stats.norm.ppf(FAR)
    except ValueError:
        return np.nan

def discrim_p(y_true,y_pred):

    C = metrics.confusion_matrix(y_true,y_pred)
    
    try:
        return stats.chi2_contingency(C)[1]
    except ValueError:
        return 1.0
    
def discrim(trials,change,detect,trial_types=('go','catch'),metric=None,metric_kws=None):
    
    if metric is None:
        metric = d_prime
    
    if metric_kws is None:
        metric_kws = dict()
        
    mask = masks.trial_types(trials,trial_types)
        
    y_true = trials[mask]['change']
    y_pred = trials[mask]['detect']
        
    return metric(y_true,y_pred,**metric_kws)

def response_bias(trials,detect,trial_types=('go','catch')):
    mask = masks.trial_types(trials,trial_types)
    
    return trials[mask][detect].mean()

def num_trials(trials):
    return len(trials)


def num_contingent_trials(trials):
    return trials['trial_type'].isin(['go','catch']).sum()

def reaction_times(trials,percentile=50,trial_types=('go',)):
    """
    reaction times to GO trials
    
    """
    mask = masks.trial_types(trials,trial_types)
    quantile = trials[mask]['reaction_time'].dropna().quantile(percentile/100.0)
    
    return quantile

def total_water(trials,trial_types=()):
    
    mask = masks.trial_types(trials,trial_types)
    
    return trials[mask][(trials['reward_times'].map(len)>0)]['reward_volume'].sum()

def earned_water(trials):
    
    return total_water(trials,('go',))

def peak_dprime(trials):
    mask = (trials['trial_type']!='aborted')
    _,_,dp = get_response_rates(trials[mask],sliding_window=100)
    try:
        return np.nanmax(dp[50:])
    except ValueError:
        return np.nan

def fraction_time_aborted(trials):

    trial_fractions = trials.trialsby('trial_type')['trial_length'].sum() / trials['trial_length'].sum()
    try:
        return trial_fractions['aborted']
    except KeyError:
        return 0.0