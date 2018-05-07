import warnings

from . import extended_trials as et
from ..translator.core import create_extended_dataframe

warnings.warn('Some validators are turned off; remove this warning when done')
extended_trial_validators = (
    # et.validate_schema,
    # et.validate_autoreward_volume <- needs autoreward_volume
    # et.validate_number_of_warmup_trials <- needs expected number or warmup trials
    et.validate_aborted_change_time,
    et.validate_reward_delivery_on_warmup_trials,
    # et.validate_autorewards_after_N_consecutive_misses, <- needs n_consecutive_misses
    et.validate_change_on_all_go_trials,
    et.validate_no_change_on_all_catch_trials,
    et.validate_intial_and_final_in_non_aborted,
    # et.validate_min_change_time <- need pre_change_time
    # et.validate_max_change_time <- need pre_change_time, stimulus_window
    et.validate_reward_follows_first_lick_in_window,
    et.validate_never_more_than_one_reward,
    # et.validate_lick_before_scheduled_on_aborted_trials,
    et.validate_lick_after_scheduled_on_go_catch_trials,
    et.validate_initial_matches_final,
    et.validate_first_lick_after_change_on_nonaborted,
    # et.validate_trial_ends_without_licks <- need minimum_no_lick_time
    # et.validate_session_within_expected_duration <- need expected_duration_seconds
    # et.validate_session_ends_at_max_cumulative_volume <- need volume_limit
    # et.validate_duration_and_volume_limit <- need volume_limit and expected_duration
    et.validate_stage_present,
    et.validate_task_id_present,
    # et.validate_number_aborted_trial_repeats <- need failure_repeats
    # et.validate_params_change_after_aborted_trial_repeats <- need failure_repeats
    # et.validate_ignore_false_alarms <- need ignore_false_alarms
    # et.validate_stimulus_distribution_key <- need expected_distribution_name
    # et.validate_monotonically_decreasing_number_of_change_times <- need expected_distribution_name
    et.validate_trial_times_never_overlap,
    et.validate_no_abort_on_lick_before_response_window,
    et.validate_licks_on_go_trials_earn_reward,
    et.validate_licks_on_catch_trials_do_not_earn_reward,
    # et.validate_even_sampling # <- needs even_sampling_enabled
    # et.validate_flash_blank_durations <- needs expected_flash_duration, expected_blank_duration
    et.validate_new_params_on_nonaborted_trials,
)


def compute_qc_metrics(core_data, extended_trial_validators=()):
    """generate a set of validation metrics

    Parameters
    ----------
    core_data: dict
        visual behavior core data object

    Returns
    -------
    dict:
        validation metrics
    """

    trials = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks=core_data['licks'],
        time=core_data['time'],
    )

    results = {
        func.__name__: func(trials)
        for func
        in extended_trial_validators
    }
    return results


def check_session_passes(qc_metrics):
    """
    determines whether a set of qc metrics "passes".

    returns True/False

    """

    return all(qc_metrics.values())


def generate_qc_report(core_data):

    results = compute_qc_metrics(
        core_data,
        extended_trial_validators,
    )

    results['passes'] = check_session_passes(results)

    return results
