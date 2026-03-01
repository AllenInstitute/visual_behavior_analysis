# sklearn-style discrimination metrics

import numpy as np
import pandas as pd

from ..trials import masks
from ...utilities import get_response_rates, flatten_list
from ...metrics import d_prime
from ...translator.core.annotate import assign_trial_description


def discrim(
        session_trials,
        change,
        detect,
        trial_types=('go', 'catch'),
        metric=None,
        metric_kws=None
):
    """Compute a discrimination metric between ground-truth change and detected response.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame (e.g. from ``VisualBehaviorOphysDataset.trials``).
    change : str
        Column name indicating whether a change occurred (boolean or 0/1).
    detect : str
        Column name indicating whether the animal responded (boolean or 0/1).
    trial_types : tuple of str, optional
        Trial types to include.  Defaults to ``('go', 'catch')``.
    metric : callable, optional
        Discrimination function with signature ``metric(y_true, y_pred)``.
        Defaults to :func:`visual_behavior.metrics.d_prime`.
    metric_kws : dict, optional
        Extra keyword arguments forwarded to *metric*.

    Returns
    -------
    float
        Scalar discrimination value (e.g. d').
    """
    if metric is None:
        metric = d_prime

    if metric_kws is None:
        metric_kws = dict()

    mask = masks.trial_types(session_trials, trial_types)

    y_true = session_trials[mask][change]
    y_pred = session_trials[mask][detect]

    return metric(y_true, y_pred, **metric_kws)


def response_bias(session_trials, detect, trial_types=('go', 'catch')):
    """Compute the overall response rate across specified trial types.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.
    detect : str
        Column name indicating whether the animal responded (boolean or 0/1).
    trial_types : tuple of str, optional
        Trial types to include.  Defaults to ``('go', 'catch')``.

    Returns
    -------
    float
        Mean response rate (0–1) across the selected trials.
    """
    mask = masks.trial_types(session_trials, trial_types)

    return session_trials[mask][detect].mean()


def flashwise_lick_probability(session_trials, flash_blank_duration=0.75):
    '''
    calculates probability of a lick on each flash
    Excludes:
        * flashes following autorewards
        * flashes following first lick in a trial (to avoid issues due to subsequent licks in a lick bout)
    inputs:
        session trials: the trial dataframe
        flash_blank_duration (default = 0.75): the sum of the flash display duration and the intervening gray screen (this is the same as the inter-flash-interval)
    IMPORTANT: This algorithm infers the number of flashes per unit time; it has no way of accounting for omitted flashes or exceptionally long frame intervals
    '''
    trials = session_trials.copy()  # operate on a copy because we will add some columns to the trials. If this were a slice, we run into trouble

    def get_first_lick_time(lick_times):
        if len(lick_times) > 0:
            return lick_times[0]
        else:
            return None

    def get_pre_lick_flashes(row, flash_blank_duration=flash_blank_duration):
        first_lick = row['first_lick']
        trial_length = row['trial_length']
        if pd.notnull(first_lick):
            return np.floor(first_lick / flash_blank_duration)
        else:
            return np.floor(trial_length / flash_blank_duration)

    def get_flashes_with_licks(row):
        trial_type = row['trial_type']
        first_lick = row['first_lick']
        if trial_type == 'autorewarded' or trial_type == 'go':
            # don't count licks following change or free reward presentation
            return 0
        elif pd.isnull(first_lick):
            return 0
        else:
            return 1

    # get first lick time
    trials['first_lick'] = trials['lick_times'].map(get_first_lick_time)
    # reference first lick to trial start
    trials['first_lick'] = trials['first_lick'] - trials['starttime']
    # count pre-lick flashes
    trials['flashes_without_licks'] = trials[['first_lick', 'trial_length']].apply(get_pre_lick_flashes, axis=1)
    # count first post-lick flash (0 if autorewarded or if there was not lick, 1 otherwise)
    trials['flashes_with_licks'] = trials[['first_lick', 'trial_type']].apply(get_flashes_with_licks, axis=1)

    return trials['flashes_with_licks'].sum() / (trials['flashes_with_licks'].sum() + trials['flashes_without_licks'].sum())


def num_trials(session_trials):
    """Return the total number of trials in the session.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.

    Returns
    -------
    int
        Total row count.
    """
    return len(session_trials)


def num_usable_trials(session_trials):
    """Count contingent (go/catch) trials within the high-reward-rate epoch.

    Filters to trials that pass the reward-rate mask (indicating the mouse is
    engaged), then counts only go and catch trial types.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.

    Returns
    -------
    int
        Number of usable go/catch trials.
    """
    usable_trials = session_trials[masks.reward_rate(session_trials)]

    return num_contingent_trials(usable_trials)


def num_contingent_trials(session_trials):
    """Count go and catch (contingent) trials.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.

    Returns
    -------
    int
        Number of trials whose ``trial_type`` is ``'go'`` or ``'catch'``.
    """
    return session_trials['trial_type'].isin(['go', 'catch']).sum()


def lick_latency(session_trials, percentile=50, trial_types=('go', )):
    """
    median (or some other %ile) time to first lick to GO trials

    """
    mask = masks.trial_types(session_trials, trial_types)
    quantile = session_trials[mask]['response_latency']\
        .replace([np.inf, -np.inf], np.nan) \
        .dropna() \
        .quantile(percentile / 100.0)

    return quantile


def reward_lick_count(session_trials):
    '''
    returns the mean number of licks on rewarded go trials
    '''
    go_trials_with_rewards = session_trials.query('trial_type == "go" and number_of_rewards > 0')
    if len(go_trials_with_rewards) > 0:
        return go_trials_with_rewards['reward_lick_count'].mean()
    else:
        return 0


def reward_lick_latency(session_trials):
    """Return the mean latency from reward delivery to first lick.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame containing a ``reward_lick_latency`` column.

    Returns
    -------
    float
        Mean reward lick latency in seconds.
    """
    quantile = session_trials['reward_lick_latency'].mean()
    return quantile


def total_water(session_trials, trial_types=()):
    """Return the total volume of water delivered across specified trial types.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.
    trial_types : tuple of str, optional
        Restrict to these trial types.  An empty tuple includes all trials.

    Returns
    -------
    float
        Sum of ``reward_volume`` (µL) across selected trials.
    """
    mask = masks.trial_types(session_trials, trial_types)

    return session_trials[mask]['reward_volume'].sum()


def earned_water(session_trials):
    """Return total water earned on go trials (excludes auto-rewards).

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.

    Returns
    -------
    float
        Sum of ``reward_volume`` (µL) on go trials only.
    """
    return total_water(session_trials, ('go', ))


def peak_dprime(session_trials, first_valid_trial=50, sliding_window=100, apply_trial_number_limit=False):
    """Return the peak d' over the session using a rolling window.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame (aborted trials are excluded internally).
    first_valid_trial : int, optional
        Trial index before which d' values are ignored.  Defaults to 50.
    sliding_window : int, optional
        Number of trials in the rolling window.  Defaults to 100.
    apply_trial_number_limit : bool, optional
        Passed through to :func:`get_response_rates`.  Defaults to ``False``.

    Returns
    -------
    float
        Peak d' value after *first_valid_trial*, or ``np.nan`` if all values
        are NaN or the session is too short.
    """
    mask = (session_trials['trial_type'] != 'aborted')
    _, _, dp = get_response_rates(
        session_trials[mask],
        sliding_window=sliding_window,
        apply_trial_number_limit=apply_trial_number_limit
    )
    if np.all(np.isnan(dp)):
        return np.nan
    try:
        return np.nanmax(dp[first_valid_trial:])
    except (IndexError, ValueError):
        return np.nan


def peak_hit_rate(session_trials):
    """Return the peak hit rate over a 100-trial rolling window (after trial 50).

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.

    Returns
    -------
    float
        Peak hit rate (0–1), or ``np.nan`` if the session is too short.
    """
    mask = (session_trials['trial_type'] != 'aborted')
    hr, _, _ = get_response_rates(session_trials[mask], sliding_window=100)
    if all(np.isnan(hr)):
        return np.nan
    try:
        return np.nanmax(hr[50:])
    except (IndexError, ValueError):
        return np.nan


def peak_false_alarm_rate(session_trials):
    """Return the peak false-alarm rate over a 100-trial rolling window (after trial 50).

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.

    Returns
    -------
    float
        Peak false-alarm rate (0–1), or ``np.nan`` if the session is too short.
    """
    mask = (session_trials['trial_type'] != 'aborted')
    _, far, _ = get_response_rates(session_trials[mask], sliding_window=100)
    if all(np.isnan(far)):
        return np.nan
    try:
        return np.nanmax(far[50:])
    except ValueError:
        return np.nan


def fraction_time_by_trial_type(session_trials, trial_type='aborted'):
    """Return the fraction of total session time spent in a given trial type.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.
    trial_type : str, optional
        Full trial type label (e.g. ``'aborted'``, ``'hit'``).  Defaults to
        ``'aborted'``.

    Returns
    -------
    float
        Fraction (0–1) of total ``trial_length`` attributable to *trial_type*,
        or 0.0 if the trial type is absent.
    """
    session_trials['full_trial_type'] = session_trials.apply(assign_trial_description, axis=1)
    trial_fractions = session_trials.groupby('full_trial_type')['trial_length'].sum() / session_trials['trial_length'].sum()
    try:
        return trial_fractions[trial_type]
    except KeyError:
        return 0.0


def trial_count_by_trial_type(session_trials, trial_type='hit'):
    """Return the number of trials of a given full trial type.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.
    trial_type : str, optional
        Full trial type label.  Defaults to ``'hit'``.

    Returns
    -------
    int or float
        Trial count, or 0.0 if the trial type is absent.
    """
    session_trials['full_trial_type'] = session_trials.apply(assign_trial_description, axis=1)
    trial_count = session_trials.groupby('full_trial_type')['trial_length'].count()
    try:
        return trial_count[trial_type]
    except KeyError:
        return 0.0


def total_number_of_licks(session_trials):
    '''
    total number of licks in the session
    if too low (<~50), could signal lick detection trouble
    '''
    return len(flatten_list(session_trials.lick_frames.values))


def session_id(session_trials):
    '''
    gets session id.
    deals with variable syntax
    '''
    if 'session_id' in session_trials.columns:
        return session_trials.iloc[0].session_id
    elif 'behavior_session_uuid' in session_trials.columns:
        return session_trials.iloc[0].behavior_session_uuid
    else:
        return None


def isnull(a):
    """Check whether *a* contains any null values, handling both scalars and arrays.

    Parameters
    ----------
    a : scalar or array-like
        Value to test.

    Returns
    -------
    bool
        ``True`` if *a* is null or contains any null element.
    """
    try:
        return pd.isnull(a).any()
    except AttributeError:
        return pd.isnull(a)


def blank_duration(session_trials):
    '''blank screen duration between each stimulus flash'''

    blank_duration_range = session_trials.iloc[0].blank_duration_range

    if not isnull(blank_duration_range):
        if len(blank_duration_range) == 1:
            return blank_duration_range
        elif len(blank_duration_range) == 2:

            return blank_duration_range[0]
    else:
        return np.nan


def training_stage(session_trials):
    """Return the training stage string for the session.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame with a ``stage`` column.

    Returns
    -------
    str
        Training stage label from the first trial row.
    """
    return session_trials['stage'].iloc[0]


def session_duration(session_trials):
    """Return the total session duration in seconds.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.

    Returns
    -------
    float
        Sum of ``trial_length`` across all trials.
    """
    return session_trials['trial_length'].sum()


def day_of_week(session_trials):
    """Return the day-of-week integer for the session.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame with a ``dayofweek`` column.

    Returns
    -------
    int
        Day of week (0 = Monday … 6 = Sunday).
    """
    return session_trials['dayofweek'].iloc[0]


def change_time_distribution(session_trials):
    """Return the change-time distribution type for the session.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame with a ``stimulus_distribution`` column.

    Returns
    -------
    str
        Distribution label (e.g. ``'exponential'`` or ``'geometric'``).
    """
    return session_trials['stimulus_distribution'].iloc[0]


def trial_duration(session_trials):
    """Return the nominal trial duration for the session.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame with a ``trial_duration`` column.

    Returns
    -------
    float
        Trial duration in seconds.
    """
    return session_trials['trial_duration'].iloc[0]


def user_id(session_trials):
    """Return the user/experimenter ID for the session.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame with a ``user_id`` column.

    Returns
    -------
    str
        User identifier from the first trial row.
    """
    return session_trials.iloc[0].user_id


def rig_id(session_trials):
    """Return the rig identifier for the session.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.  May or may not contain a ``rig_id`` column.

    Returns
    -------
    str
        Rig ID, or ``'unknown'`` if the column is absent.
    """
    try:
        rig_id = session_trials['rig_id'].iloc[0]
    except KeyError:
        rig_id = 'unknown'
    return rig_id


def filename(session_trials):
    """Return the source data filename for the session.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame with a ``filename`` column.

    Returns
    -------
    str
        Path to the original data file.
    """
    return session_trials.iloc[0].filename


def stimulus(session_trials):
    """Return the stimulus class used in the session.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame with a ``stimulus`` column.

    Returns
    -------
    str
        Stimulus label (e.g. ``'natural_images'``).
    """
    return session_trials['stimulus'].iloc[0]


def reward_rate(session_trials, epoch_length):
    """Compute the reward rate (rewards per second) over a given epoch.

    Only counts rewarded go trials; auto-rewarded trials are excluded.

    Parameters
    ----------
    session_trials : pd.DataFrame
        Trial-level DataFrame.
    epoch_length : float
        Duration of the epoch in seconds used as the denominator.

    Returns
    -------
    float
        Rewards per second.
    """

    mask = (
        masks.trial_types(session_trials, ('go',))
        & (session_trials['reward_times'].map(len) > 0)
    )

    return mask.sum() / epoch_length
