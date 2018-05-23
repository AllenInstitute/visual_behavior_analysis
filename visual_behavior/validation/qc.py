import warnings

from . import extended_trials as et
from . import core as cd
from . import foraging2 as f2
from ..translator.core import create_extended_dataframe

functions_yet_to_implement = [

]
warnings.warn('The following are turned off; remove this warning when done {}'.format(
    '\n' +
    ' * validate_schema' + '\n' +
    ' * validate_monotonically_decreasing_number_of_change_times' + '\n' +
    ' * validate_frame_intervals_exists'
))


def define_validation_functions(core_data):
    '''
    creates a dictionary containing all validation functions as keys, input arguments as values
    '''

    trials = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks=core_data['licks'],
        time=core_data['time'],
    )

    AUTO_REWARD_VOLUME = core_data['metadata']['auto_reward_vol']
    PRE_CHANGE_TIME = core_data['metadata']['delta_minimum']
    DISTRIBUTION = core_data['metadata']['stimulus_distribution']
    SESSION_DURATION = core_data['metadata']['max_session_duration']
    MIN_NO_LICK_TIME = core_data['metadata']['min_no_lick_time']
    FREE_REWARD_TRIALS = core_data['metadata']['free_reward_trials']
    ABORT_ON_EARLY_RESPONSE = core_data['metadata']['abort_on_early_response']
    EXPECTED_CHANGE_DISTRIBUTION_MEAN = core_data['metadata']['delta_mean']
    EVEN_SAMPLING = core_data['metadata']['even_sampling_enabled']
    FAILURE_REPEATS = core_data['metadata']['failure_repeats']

    INITIAL_BLANK = core_data['metadata']['initial_blank_duration']

    PERIODIC_FLASH = et.fix_periodic_flash(core_data['metadata']['params']['periodic_flash'])
    WARM_UP_TRIALS = core_data['metadata']['params']['warm_up_trials']
    STIMULUS_WINDOW = core_data['metadata']['params']['stimulus_window']
    VOLUME_LIMIT = core_data['metadata']['params']['volume_limit']
    if 'catch_frequency' in core_data['metadata']['params'].keys():
        CATCH_FREQUENCY = core_data['metadata']['params']['catch_frequency']
    else:
        CATCH_FREQUENCY = 0

    validation_functions = {
        # et.validate_schema
        et.validate_autoreward_volume: (trials, AUTO_REWARD_VOLUME,),
        et.validate_number_of_warmup_trials: (trials, WARM_UP_TRIALS,),
        et.validate_reward_delivery_on_warmup_trials: (trials,),
        et.validate_autorewards_after_N_consecutive_misses: (trials, FREE_REWARD_TRIALS,),
        et.validate_change_on_all_go_trials: (trials,),
        et.validate_no_change_on_all_catch_trials: (trials,),
        et.validate_intial_and_final_in_non_aborted: (trials,),
        et.validate_min_change_time: (trials, PRE_CHANGE_TIME,),
        et.validate_max_change_time: (trials, PRE_CHANGE_TIME, STIMULUS_WINDOW,),
        et.validate_reward_follows_first_lick_in_window: (trials,),
        et.validate_never_more_than_one_reward: (trials,),
        et.validate_lick_before_scheduled_on_aborted_trials: (trials,),
        et.validate_lick_after_scheduled_on_go_catch_trials: (trials, ABORT_ON_EARLY_RESPONSE,),
        et.validate_initial_matches_final: (trials,),
        et.validate_first_lick_after_change_on_nonaborted: (trials, ABORT_ON_EARLY_RESPONSE),
        et.validate_trial_ends_without_licks: (trials, MIN_NO_LICK_TIME,),
        et.validate_session_within_expected_duration: (trials, SESSION_DURATION * 60,),
        et.validate_session_ends_at_max_cumulative_volume: (trials, VOLUME_LIMIT,),
        et.validate_duration_and_volume_limit: (trials, SESSION_DURATION * 60, VOLUME_LIMIT,),
        et.validate_catch_frequency: (trials, CATCH_FREQUENCY,),
        et.validate_stage_present: (trials,),
        et.validate_task_id_present: (trials,),
        et.validate_number_aborted_trial_repeats: (trials, FAILURE_REPEATS,),
        et.validate_params_change_after_aborted_trial_repeats: (trials, FAILURE_REPEATS,),
        et.validate_ignore_false_alarms: (trials, ABORT_ON_EARLY_RESPONSE,),
        et.validate_trial_times_never_overlap: (trials,),
        et.validate_stimulus_distribution_key: (trials, DISTRIBUTION,),
        et.validate_change_time_mean: (trials, EXPECTED_CHANGE_DISTRIBUTION_MEAN,),
        # et.validate_monotonically_decreasing_number_of_change_times:(trials,DISTRIBUTION,),
        et.validate_no_abort_on_lick_before_response_window: (trials,),
        et.validate_licks_on_go_trials_earn_reward: (trials,),
        et.validate_licks_on_catch_trials_do_not_earn_reward: (trials,),
        et.validate_even_sampling: (trials, EVEN_SAMPLING,),
        et.validate_two_stimuli_per_go_trial: (trials, core_data['visual_stimuli'],),
        et.validate_one_stimulus_per_catch_trial: (trials, core_data['visual_stimuli'],),
        et.validate_one_stimulus_per_aborted_trial: (trials, core_data['visual_stimuli'],),
        et.validate_change_frame_at_flash_onset: (trials, core_data['visual_stimuli'], PERIODIC_FLASH,),
        et.validate_initial_blank: (trials, core_data['visual_stimuli'], INITIAL_BLANK, PERIODIC_FLASH),
        et.validate_new_params_on_nonaborted_trials: (trials,),
        et.validate_flash_blank_durations: (core_data['visual_stimuli'], PERIODIC_FLASH,),  # this one doesn't take trials
        cd.validate_running_data: (core_data,),  # this one doesn't take trials
        cd.validate_licks: (core_data,),  # this one doesn't take trials
        cd.validate_minimal_dropped_frames:(core_data,), # this one doesn't take trials
        # f2.validate_frame_intervals_exists:(data), # this one doesn't take trials
    }

    return validation_functions


def compute_qc_metrics(core_data):
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
    validation_functions = define_validation_functions(core_data)

    results = {
        func.__name__: func(*validation_functions[func])
        for func
        in validation_functions
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
    )

    results['passes'] = check_session_passes(results)

    return results
