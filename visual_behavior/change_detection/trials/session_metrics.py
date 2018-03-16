# sklearn-style discrimination metrics

import numpy as np
import pandas as pd

from ..trials import masks
from ...utilities import get_response_rates, flatten_list
from ...metrics import d_prime


def discrim(
        session_trials,
        change,
        detect,
        trial_types=('go', 'catch'),
        metric=None,
        metric_kws=None
):
    if metric is None:
        metric = d_prime

    if metric_kws is None:
        metric_kws = dict()

    mask = masks.trial_types(session_trials, trial_types)

    y_true = session_trials[mask][change]
    y_pred = session_trials[mask][detect]

    return metric(y_true, y_pred, **metric_kws)


def response_bias(session_trials, detect, trial_types=('go', 'catch')):
    mask = masks.trial_types(session_trials, trial_types)

    return session_trials[mask][detect].mean()


def num_trials(session_trials):
    return len(session_trials)


def num_usable_trials(session_trials):
    usable_trials = session_trials[masks.reward_rate(session_trials)]

    return num_contingent_trials(usable_trials)


def num_contingent_trials(session_trials):
    return session_trials['trial_type'].isin(['go', 'catch']).sum()


def lick_latency(session_trials, percentile=50, trial_types=('go', )):
    """
    median (or some other %ile) time to first lick to GO trials

    """
    mask = masks.trial_types(session_trials, trial_types)
    quantile = session_trials[mask]['response_latency']\
        .dropna() \
        .quantile(percentile / 100.0)

    return quantile


def reward_lick_count(session_trials):
    quantile = session_trials['reward_lick_count'].mean()
    return quantile


def reward_lick_latency(session_trials):
    quantile = session_trials['reward_lick_latency'].mean()
    return quantile


def total_water(session_trials, trial_types=()):
    mask = masks.trial_types(session_trials, trial_types)

    return session_trials[mask][(session_trials['reward_times'].map(len) > 0)]['reward_volume'].sum()


def earned_water(session_trials):
    return total_water(session_trials, ('go', ))


def peak_dprime(session_trials):
    mask = (session_trials['trial_type'] != 'aborted')
    _, _, dp = get_response_rates(session_trials[mask], sliding_window=100)
    try:
        return np.nanmax(dp[50:])
    except ValueError:
        return np.nan


def peak_hit_rate(session_trials):
    mask = (session_trials['trial_type'] != 'aborted')
    hr, _, _ = get_response_rates(session_trials[mask], sliding_window=100)
    try:
        return np.nanmax(hr[50:])
    except ValueError:
        return np.nan


def peak_false_alarm_rate(session_trials):
    mask = (session_trials['trial_type'] != 'aborted')
    _, far, _ = get_response_rates(session_trials[mask], sliding_window=100)
    try:
        return np.nanmax(far[50:])
    except ValueError:
        return np.nan


def fraction_time_aborted(session_trials):
    trial_fractions = session_trials.groupby('trial_type')['trial_length'].sum() / session_trials['trial_length'].sum()
    try:
        return trial_fractions['aborted']
    except KeyError:
        return 0.0


def total_number_of_licks(session_trials):
    '''
    total number of licks in the session
    if too low (<~50), could signal lick detection trouble
    '''
    return len(flatten_list(session_trials.lick_frames.values))


def session_id(session_trials):
    return session_trials.iloc[0].session_id


def isnull(a):
    try:
        return pd.isnull(a).any()
    except AttributeError:
        return pd.isnull(a)


def blank_duration(session_trials):
    '''blank screen duration between each stimulus flash'''
    if not isnull(session_trials.iloc[0].blank_duration_range):
        if len(session_trials.iloc[0].blank_duration_range) == 1:
            return session_trials.iloc[0].blank_duration_range
        elif len(session_trials.iloc[0].blank_duration_range) == 2:
            return session_trials.iloc[0].blank_duration_range[0]
    else:
        return np.nan


def session_duration(session_trials):
    return session_trials['trial_length'].sum()


def day_of_week(session_trials):
    return session_trials['dayofweek'].iloc[0]


def change_time_distribution(session_trials):
    return session_trials['stimulus_distribution'].iloc[0]


def trial_duration(session_trials):
    return session_trials['trial_duration'].iloc[0]


def user_id(session_trials):
    return session_trials.iloc[0].user_id


def filename(session_trials):
    return session_trials.iloc[0].filename


def stimulus(session_trials):
    return session_trials['stimulus'].iloc[0]


def reward_rate(session_trials, epoch_length):

    mask = (
        masks.trial_types(session_trials, ('go',))
        & (session_trials['reward_times'].map(len) > 0)
    )

    return mask.sum() / epoch_length
