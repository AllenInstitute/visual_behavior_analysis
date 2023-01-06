import pandas as pd
import datetime
import json
import numpy as np
import logging

from visual_behavior.translator.core import annotate

logger = logging.getLogger(__name__)


def create_extended_dataframe(
        trials,
        metadata,
        licks,
        time,
        visual_stimuli=None,
        *args,
        **kwargs
):
    """ creates a trials dataframe from a detection-of-change session

    Parameters
    ----------
    filename : str
        file path of a pickled detection-of-change session

    Returns
    -------
    trials : pandas DataFrame
    """
    # data = pd.read_pickle(filename)

    trials = trials[~pd.isnull(trials['reward_times'])].reset_index(drop=True)

    # add some columns to the dataframe
    # column : key
    keydict = {
        'mouse_id': 'mouseid',
        'response_window': 'response_window',
        'task': 'task',
        'stage': 'stage',
        'session_duration': 'stoptime',
        'user_id': 'userid',
        'LDT_mode': 'lick_detect_training_mode',
        'blank_screen_timeout': 'blankscreen_on_timeout',
        'stim_duration': 'stim_duration',
        'blank_duration_range': 'blank_duration_range',
        'prechange_minimum': 'delta_minimum',
        'stimulus_distribution': 'stimulus_distribution',
        'stimulus': 'stimulus',
        'distribution_mean': 'delta_mean',
        'computer_name': 'computer_name',
        'behavior_session_uuid': 'behavior_session_uuid',
    }

    annotate.annotate_parameters(trials, metadata, keydict=keydict, inplace=True)
    annotate.annotate_startdatetime(trials, metadata, inplace=True)
    annotate.explode_startdatetime(trials, inplace=True)
    annotate.annotate_n_rewards(trials, inplace=True)
    annotate.annotate_rig_id(trials, metadata, inplace=True)
    annotate.fix_autorearded(trials, inplace=True)
    annotate.annotate_cumulative_reward(trials, metadata, inplace=True)
    # annotate.annotate_filename(trials, filename, inplace=True)

    for col in ('auto_rewarded', 'change_time', ):
        if col not in trials.columns:
            trials[col] = None

    # add some columns that require calculation
    annotate.make_trials_contiguous(trials, time, visual_stimuli, inplace=True)

    trials['trial_type'] = annotate.categorize_trials(trials)
    trials['lick_frames'] = annotate.get_lick_frames(trials, licks)

  

    annotate.update_times(trials, time, inplace=True)

    # bouts MJD (must be after update_times, where lick_times are corrected)
    trials["lick_bout_times"] = annotate.get_lick_bout_times(trials, licks)

    annotate.annotate_lick_vigor(trials, licks, inplace=True)
    annotate.calculate_latency(trials, inplace=True)
    annotate.calculate_reward_rate(trials, inplace=True)

    trials['response'] = annotate.check_responses(trials)
    trials['color'] = trials.apply(annotate.assign_color, axis=1)
    trials['response_type'] = annotate.get_response_type(trials)

    try:
        annotate.remove_repeated_licks(trials, inplace=True)
    except Exception as e:
        print('FAILED TO REMOVE REPEATED LICKS')
        print(e)

    return trials


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
