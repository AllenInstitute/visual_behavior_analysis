## sklearn-style discrimination metrics

import numpy as np
from scipy import stats

from visual_behavior import masks
from visual_behavior.utilities import get_response_rates

def discrim(session_trials,change,detect,trial_types=('go','catch'),metric=None,metric_kws=None):

    if metric is None:
        metric = d_prime

    if metric_kws is None:
        metric_kws = dict()

    mask = masks.trial_types(session_trials,trial_types)

    y_true = session_trials[mask][change]
    y_pred = session_trials[mask][detect]

    return metric(y_true,y_pred,**metric_kws)

def response_bias(session_trials,detect,trial_types=('go','catch')):
    mask = masks.trial_types(session_trials,trial_types)

    return session_trials[mask][detect].mean()

def num_trials(session_trials):
    return len(session_trials)

def num_usable_trials(session_trials):

    usable_trials = session_trials[masks.reward_rate(session_trials)]

    return num_contingent_trials(usable_trials)


def num_contingent_trials(session_trials):
    return session_trials['trial_type'].isin(['go','catch']).sum()

def lick_latency(session_trials,percentile=50,trial_types=('go',)):
    """
    median (or some other %ile) time to first lick to GO trials

    """
    mask = masks.trial_types(session_trials,trial_types)
    quantile = session_trials[mask]['response_latency'].dropna().quantile(percentile/100.0)

    return quantile

def lick_rate(session_trials,percentile=50,trial_types=('go',)):
    mask = masks.trial_types(session_trials,trial_types)
    quantile = session_trials[mask]['lick_rate'].dropna().quantile(percentile/100.0)
    return quantile

def lick_quantity(session_trials,trial_types=('go',)):
    mask = masks.trial_types(session_trials,trial_types)
    quantile = session_trials[mask]['number_of_licks'].mean()
    return quantile

def total_water(session_trials,trial_types=()):

    mask = masks.trial_types(session_trials,trial_types)

    return session_trials[mask][(session_trials['reward_times'].map(len)>0)]['reward_volume'].sum()

def earned_water(session_trials):

    return total_water(session_trials,('go',))

def peak_dprime(session_trials):
    mask = (session_trials['trial_type']!='aborted')
    _,_,dp = get_response_rates(session_trials[mask],sliding_window=100)
    try:
        return np.nanmax(dp[50:])
    except ValueError:
        return np.nan

def fraction_time_aborted(session_trials):

    trial_fractions = session_trials.groupby('trial_type')['trial_length'].sum() / session_trials['trial_length'].sum()
    try:
        return trial_fractions['aborted']
    except KeyError:
        return 0.0
