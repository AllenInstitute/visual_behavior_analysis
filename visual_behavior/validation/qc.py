from . import extended_trials as et
from ..translator.core import create_extended_dataframe


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

    extended_trial_validators = (
        # et.validate_schema,
        et.validate_reward_delivery_on_warmup_trials,
        et.validate_autorewards_after_N_consecutive_misses,
        et.validate_change_on_all_go_trials,
        et.validate_no_change_on_all_catch_trials,
        et.validate_intial_and_final_in_non_aborted,
        et.validate_reward_follows_first_lick_in_window,
        et.validate_never_more_than_one_reward,
        # et.validate_lick_before_scheduled_on_aborted_trials,
        et.validate_lick_after_scheduled_on_go_catch_trials,
        et.validate_initial_matches_final,
        et.validate_first_lick_after_change_on_nonaborted,
        et.validate_stage_present,
        et.validate_task_id_present,
        et.validate_trial_times_never_overlap,
        et.validate_no_abort_on_lick_before_response_window,
    )

    results = compute_qc_metrics(
        core_data,
        extended_trial_validators,
    )

    results['passes'] = check_session_passes(results)

    return results
