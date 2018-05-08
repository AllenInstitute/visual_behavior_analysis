import pandas as pd
import datetime
import json
import numpy as np

from . import annotate


def create_extended_dataframe(trials, metadata, licks, time):
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
        'trial_duration': 'trial_duration',
        'computer_name': 'computer_name',
    }

    annotate.annotate_parameters(trials, metadata, keydict=keydict, inplace=True)
    annotate.annotate_startdatetime(trials, metadata, inplace=True)
    annotate.annotate_n_rewards(trials, inplace=True)
    annotate.annotate_rig_id(trials, metadata, inplace=True)
    annotate.fix_autorearded(trials, inplace=True)
    annotate.annotate_cumulative_reward(trials, metadata, inplace=True)
    # annotate.annotate_filename(trials, filename, inplace=True)

    for col in ('auto_rewarded', 'change_time', ):
        if col not in trials.columns:
            trials[col] = None

    # add some columns that require calculation

    trials['trial_type'] = annotate.categorize_trials(trials)
    trials['endframe'] = annotate.get_end_frame(trials, metadata)
    trials['lick_frames'] = annotate.get_lick_frames(trials, licks)

    annotate.update_times(trials, time, inplace=True)

    annotate.annotate_lick_vigor(trials, licks, inplace=True)
    annotate.calculate_latency(trials, inplace=True)
    annotate.calculate_reward_rate(trials, inplace=True)

    trials['response'] = annotate.check_responses(trials)
    trials['trial_length'] = annotate.calculate_trial_length(trials)
    trials['endtime'] = annotate.get_end_time(trials, time)
    trials['color'] = annotate.assign_color(trials)
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


def df_to_json(input_df):
    return json.dumps(input_df.to_dict('records'), cls=JSONEncoder)
