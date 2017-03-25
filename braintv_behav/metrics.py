## sklearn-style discrimination metrics

import numpy as np
from sklearn import metrics
from scipy import stats

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
    
# change detection metrics

def _trial_types_mask(trial_types,behav_df):
    
    if trial_types is not None and len(trial_types)>0:
        return behav_df['trial_type'].isin(trial_types)
    else:
        return np.ones((len(behav_df),),dtype=bool)
    
def discrim(behav_df,change,detect,trial_types=('go','catch'),metric=None,metric_kws=None):
    
    if metric is None:
        metric = d_prime
    
    if metric_kws is None:
        metric_kws = dict()
        
    mask = _trial_types_mask(trial_types,behav_df)
        
    y_true = behav_df[mask]['change']
    y_pred = behav_df[mask]['detect']
        
    return metric(y_true,y_pred,**metric_kws)

def response_bias(behav_df,detect,trial_types=('go','catch')):
    mask = _trial_types_mask(trial_types,behav_df)
    
    return behav_df[mask][detect].mean()

def num_trials(behav_df):
    return len(behav_df)


def num_contingent_trials(behav_df):
    return behav_df['trial_type'].isin(['go','catch']).sum()

def reaction_times(behav_df,percentile=50,trial_types=('go',)):
    """
    reaction times to GO trials
    
    """
    mask = _trial_types_mask(trial_types,behav_df)
    quantile = behav_df[mask]['reaction_time'].dropna().quantile(percentile/100.0)
    
    return quantile

def total_water(behav_df,trial_types=()):
    
    mask = _trial_types_mask(trial_types,behav_df)
    
    return behav_df[mask][(behav_df['reward_times'].map(len)>0)]['reward_volume'].sum()

def earned_water(behav_df):
    
    return total_water(behav_df,('go',))

def peak_dprime(group):
    mask = (group['trial_type']!='aborted')
    _,_,dp = get_response_rates(group[mask],sliding_window=100)
    try:
        return np.nanmax(dp[50:])
    except ValueError:
        return np.nan

def fraction_time_aborted(group):

    trial_fractions = group.groupby('trial_type')['trial_length'].sum() / group['trial_length'].sum()
    try:
        return trial_fractions['aborted']
    except KeyError:
        return 0.0