import six
import pandas as pd
import numpy as np
from . import session_metrics
from .. import metrics


def create_summarizer(**kwargs):
    """ creates a function that can be applied to a dataframe

    importantly, this function is specially formed to be used in the "apply"
    method after a groupby.

    """

    def apply_func(group):
        result = {
            metric: metric_func(group)
            for metric, metric_func in six.iteritems(kwargs)
        }
        return pd.Series(result, name='metrics')

    return apply_func


def calc_minimum_delta_ori(session_trials):
    '''
    specialized function when a range of delta oris have been displayed
    '''
    delta_oris = session_trials.delta_ori.abs().unique()
    if True not in pd.isnull(delta_oris):
        return np.min(delta_oris[delta_oris > 0])
    else:
        return np.nan


def number_of_delta_oris(session_trials):
    return len(session_trials.delta_ori.abs().unique())


def number_of_mask_contrasts(session_trials):
    '''specific to masking sessions'''
    return len(session_trials.mask_contrast.abs().unique())


def number_of_stim_mask_ISIs(session_trials):
    '''specific to masking sessions'''
    return len(session_trials.ISI.unique())


def session_level_summary(trials, **kwargs):
    """ computes session-level summary table
    """

    summarizer = create_summarizer(
        session_id=session_metrics.session_id,
        session_duration=session_metrics.session_duration,
        d_prime_peak=session_metrics.peak_dprime,
        d_prime=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.d_prime),
        discrim_p=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.discrim_p),
        response_bias=lambda grp: session_metrics.response_bias(grp, 'detect'),
        earned_water=session_metrics.earned_water,
        total_water=session_metrics.total_water,
        num_contingent_trials=session_metrics.num_contingent_trials,
        lick_latency_median=session_metrics.lick_latency,
        fraction_time_aborted=session_metrics.fraction_time_aborted,
        hit_rate=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.hit_rate),
        false_alarm_rate=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.false_alarm_rate),
        hit_rate_peak=session_metrics.peak_hit_rate,
        false_alarm_rate_peak=session_metrics.peak_false_alarm_rate,
        number_of_licks=session_metrics.total_number_of_licks,
        blank_duration=session_metrics.blank_duration,
        day_of_week=session_metrics.day_of_week,
        change_time_distribution=session_metrics.change_time_distribution,
        trial_duration=session_metrics.trial_duration,
        user_id=session_metrics.user_id,
        filename=session_metrics.filename,
        stimulus=session_metrics.stimulus,
        **kwargs
    )

    session_summary = (
        trials
        .groupby(['mouse_id', 'startdatetime'])
        .apply(summarizer)
        .reset_index()
    )

    return session_summary


def epoch_level_summary(trials, epoch_length=5.0, **kwargs):
    from visual_behavior.data import annotate_epochs
    trials = annotate_epochs(trials, epoch_length)

    summarizer = create_summarizer(
        num_contingent_trials=session_metrics.num_contingent_trials,
        d_prime=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.d_prime),
        response_bias=lambda grp: session_metrics.response_bias(grp, 'detect'),
        earned_water=session_metrics.earned_water,
        lick_latency_median=session_metrics.lick_latency,
        fraction_time_aborted=session_metrics.fraction_time_aborted,
        hit_rate=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.hit_rate),
        false_alarm_rate=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.false_alarm_rate),
        reward_lick_count=session_metrics.reward_lick_count,
        reward_lick_latency=session_metrics.reward_lick_latency,
        **kwargs
    )

    epoch_summary = (
        trials
        .groupby(['mouse_id', 'session_id', 'epoch'])
        .apply(summarizer)
        .reset_index()
    )

    epoch_summary['mean(HR,FA)'] = (
        epoch_summary['hit_rate']
        + epoch_summary['false_alarm_rate']
    ) / 2

    return epoch_summary
