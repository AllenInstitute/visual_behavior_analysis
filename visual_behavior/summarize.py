import six
import pandas as pd
import numpy as np
from visual_behavior.metrics import session
from visual_behavior.metrics import classification


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
        session_id=session.session_id,
        session_duration=session.session_duration,
        d_prime_peak=session.peak_dprime,
        d_prime=lambda grp: session.discrim(grp, 'change', 'detect', metric=classification.d_prime),
        discrim_p=lambda grp: session.discrim(grp, 'change', 'detect', metric=classification.discrim_p),
        response_bias=lambda grp: session.response_bias(grp, 'detect'),
        earned_water=session.earned_water,
        total_water=session.total_water,
        num_contingent_trials=session.num_contingent_trials,
        lick_latency_median=session.lick_latency,
        fraction_time_aborted=session.fraction_time_aborted,
        hit_rate=lambda grp: session.discrim(grp, 'change', 'detect', metric=classification.hit_rate),
        false_alarm_rate=lambda grp: session.discrim(grp, 'change', 'detect', metric=classification.false_alarm_rate),
        hit_rate_peak=session.peak_hit_rate,
        false_alarm_rate_peak=session.peak_false_alarm_rate,
        number_of_licks=session.total_number_of_licks,
        blank_duration=session.blank_duration,
        day_of_week=session.day_of_week,
        change_time_distribution=session.change_time_distribution,
        trial_duration=session.trial_duration,
        user_id=session.user_id,
        filename=session.filename,
        stimulus=session.stimulus,
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
        num_contingent_trials=session.num_contingent_trials,
        d_prime=lambda grp: session.discrim(grp, 'change', 'detect', metric=classification.d_prime),
        response_bias=lambda grp: session.response_bias(grp, 'detect'),
        earned_water=session.earned_water,
        lick_latency_median=session.lick_latency,
        fraction_time_aborted=session.fraction_time_aborted,
        hit_rate=lambda grp: session.discrim(grp, 'change', 'detect', metric=classification.hit_rate),
        false_alarm_rate=lambda grp: session.discrim(grp, 'change', 'detect', metric=classification.false_alarm_rate),
        hit_lick_rate=session.hit_lick_rate,
        hit_lick_quantity=session.hit_lick_quantity,
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
