import six
import pandas as pd
import numpy as np
from . import session_metrics
from ... import metrics
from ...translator.core.annotate import annotate_epochs, annotate_change_detect


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


DEFAULT_SUMMARY_METRICS = dict(
    # behavior_session_uuid=session_metrics.session_id,
    session_duration=session_metrics.session_duration,
    d_prime_peak=session_metrics.peak_dprime,
    discrim_p=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.discrim_p),
    response_bias=lambda grp: session_metrics.response_bias(grp, 'detect'),
    flashwise_lick_probability=session_metrics.flashwise_lick_probability,
    earned_water=session_metrics.earned_water,
    total_water=session_metrics.total_water,
    num_contingent_trials=session_metrics.num_contingent_trials,
    lick_latency_median=session_metrics.lick_latency,
    fraction_time_aborted=lambda grp: session_metrics.fraction_time_by_trial_type(grp, 'aborted'),
    fraction_time_hit=lambda grp: session_metrics.fraction_time_by_trial_type(grp, 'hit'),
    fraction_time_miss=lambda grp: session_metrics.fraction_time_by_trial_type(grp, 'miss'),
    fraction_time_correct_reject=lambda grp: session_metrics.fraction_time_by_trial_type(grp, 'correct_reject'),
    fraction_time_false_alarm=lambda grp: session_metrics.fraction_time_by_trial_type(grp, 'false_alarm'),
    fraction_time_auto_rewarded=lambda grp: session_metrics.fraction_time_by_trial_type(grp, 'auto_rewarded'),
    number_of_hit_trials=lambda grp: session_metrics.trial_count_by_trial_type(grp, 'hit'),
    number_of_miss_trials=lambda grp: session_metrics.trial_count_by_trial_type(grp, 'miss'),
    number_of_false_alarm_trials=lambda grp: session_metrics.trial_count_by_trial_type(grp, 'false_alarm'),
    number_of_correct_reject_trials=lambda grp: session_metrics.trial_count_by_trial_type(grp, 'correct_reject'),
    hit_rate_peak=session_metrics.peak_hit_rate,
    false_alarm_rate_peak=session_metrics.peak_false_alarm_rate,
    number_of_licks=session_metrics.total_number_of_licks,
    blank_duration=session_metrics.blank_duration,
    day_of_week=session_metrics.day_of_week,
    change_time_distribution=session_metrics.change_time_distribution,
    # trial_duration=session_metrics.trial_duration,
    user_id=session_metrics.user_id,
    rig_id=session_metrics.rig_id,
    # filename=session_metrics.filename,
    stimulus=session_metrics.stimulus,
    stage=session_metrics.training_stage,
)


def session_level_summary(trials, groupby=('mouse_id', 'behavior_session_uuid', 'startdatetime'), apply_trial_number_limit=False, **kwargs):
    """ computes session-level summary table
    """

    summary_metrics = DEFAULT_SUMMARY_METRICS.copy()

    trial_number_dependent_metrics = dict(
        hit_rate=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.hit_rate, metric_kws={'apply_trial_number_limit': apply_trial_number_limit}),
        false_alarm_rate=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.false_alarm_rate, metric_kws={'apply_trial_number_limit': apply_trial_number_limit}),
        d_prime=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.d_prime, metric_kws={'apply_trial_number_limit': apply_trial_number_limit}),
    )

    summary_metrics.update(trial_number_dependent_metrics)

    summary_metrics.update(kwargs)
    summarizer = create_summarizer(**summary_metrics)

    trials = annotate_change_detect(trials)

    # print('calling label_auto_rewards')
    # trials = label_auto_rewards(trials)
    # print(trials.trial_type.unique())

    session_summary = (
        trials
        .groupby(list(groupby))
        .apply(summarizer)
        .reset_index()
    )

    return session_summary


def epoch_level_summary(trials, epoch_length=10.0, apply_trial_number_limit=False, clip_vals=[0, 1], **kwargs):
    '''
    clip_vals ensures that hit and false alarm rates cannot exceed set limits (defaults to [0,1], which will have no effect)
    '''
    trials = annotate_change_detect(trials)
    trials = annotate_epochs(trials, epoch_length)

    summarizer = create_summarizer(
        num_contingent_trials=session_metrics.num_contingent_trials,
        response_bias=lambda grp: session_metrics.response_bias(grp, 'detect'),
        discrim_p=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.discrim_p),
        flashwise_lick_probability=session_metrics.flashwise_lick_probability,
        earned_water=session_metrics.earned_water,
        lick_latency_median=session_metrics.lick_latency,
        fraction_time_aborted=lambda grp: session_metrics.fraction_time_by_trial_type(grp, 'aborted'),
        number_of_go_trials=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.N_go_trials),
        number_of_catch_trials=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.N_catch_trials),
        hit_rate=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.hit_rate, metric_kws={'apply_trial_number_limit': apply_trial_number_limit, 'clip_vals': clip_vals}),
        false_alarm_rate=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.false_alarm_rate, metric_kws={'apply_trial_number_limit': apply_trial_number_limit, 'clip_vals': clip_vals}),
        d_prime=lambda grp: session_metrics.discrim(grp, 'change', 'detect', metric=metrics.d_prime, metric_kws={'apply_trial_number_limit': apply_trial_number_limit, 'clip_vals': clip_vals}),
        reward_lick_count=session_metrics.reward_lick_count,
        reward_lick_latency=session_metrics.reward_lick_latency,
        **kwargs
    )

    epoch_summary = (
        trials
        .groupby(['mouse_id', 'behavior_session_uuid', 'startdatetime', 'epoch'])
        .apply(summarizer)
        .reset_index()
    )

    epoch_summary['mean(HR,FA)'] = (
        epoch_summary['hit_rate']
        + epoch_summary['false_alarm_rate']
    ) / 2

    epoch_summary['criterion'] = metrics.criterion(epoch_summary['hit_rate'], epoch_summary['false_alarm_rate'])

    return epoch_summary
