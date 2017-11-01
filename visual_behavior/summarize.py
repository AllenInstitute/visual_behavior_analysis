import pandas as pd
from visual_behavior import metrics
from visual_behavior.metrics import classification

def create_summarizer(**kwargs):
    """ creates a function that can be applied to a dataframe

    importantly, this function is specially formed to be used in the "apply"
    method after a groupby.

    """

    def apply_func(group):
        result = {metric:metric_func(group) for metric,metric_func in six.iteritems(kwargs)}
        return pd.Series(result,name='metrics')

    return apply_func


# def calc_minimum_delta_ori(dft):
#     # doug's code
#     return minimum_delta_ori
#
# def get_task(dft):
#     return dft.iloc[0].task
#
# session_level_summary(trials,
#                       minimum_delta_ori=calc_minimum_delta_ori,
#                       task=get_task,
#                       )

def session_level_summary(trials,**kwargs):

    summarizer = create_summarizer(
        d_prime_peak = metrics.peak_dprime,
        d_prime = lambda grp: metrics.discrim(grp,'change','detect',metric=classification.d_prime),
        discrim_p = lambda grp: metrics.discrim(grp,'change','detect',metric=classification.discrim_p),
        response_bias = lambda grp: metrics.response_bias(grp,'detect'),
        earned_water = metrics.earned_water,
        num_contingent_trials = metrics.num_contingent_trials,
        lick_latency_median = session.reaction_times,
        fraction_time_aborted = metrics.fraction_time_aborted,
        hit_rate = lambda grp: metrics.discrim(grp,'change','detect',metric=classification.hit_rate),
        false_alarm_rate = lambda grp: metrics.discrim(grp,'change','detect',metric=classification.false_alarm_rate),
        **kwargs
    )

    session_summary = (
        trials
        .groupby(['mouse','startdatetime'])
        .apply(summarizer)
        .reset_index()
        )

    return session_summary

def epoch_level_summary(trials,epoch_length=5.0):

    summarizer = create_summarizer(
        d_prime = lambda grp: metrics.discrim(grp,'change','detect',metric=classification.d_prime),
        response_bias = lambda grp: metrics.response_bias(grp,'detect'),
        earned_water = metrics.earned_water,
        lick_latency_median = session.reaction_times,
        fraction_time_aborted = metrics.fraction_time_aborted,
        hit_rate = lambda grp: metrics.discrim(grp,'change','detect',metric=classification.hit_rate),
        false_alarm_rate = lambda grp: metrics.discrim(grp,'change','detect',metric=classification.false_alarm_rate),
    )

    epoch_summary = (
        trials
        .groupby(['mouse','startdatetime','epoch'])
        .apply(compute_metrics)
        .reset_index()
        )


    epoch_summary['engagement'] = (
        epoch_summary['hit_rate']
        + epoch_summary['false_alarm_rate']
        ) / 2

    return epoch_summary
