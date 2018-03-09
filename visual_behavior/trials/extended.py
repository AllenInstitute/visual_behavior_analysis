import pandas as pd

from visual_behavior.devices import get_rig_id
from visual_behavior.io import load_trials, load_licks, load_time
from visual_behavior.trials import annotate

def create_extended_dataframe(trials, data, time=None):
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

    if time is None:
        time = load_time(data)

    df = load_trials(data, time=time)
    df = df[~pd.isnull(df['reward_times'])].reset_index(drop=True)

    # add some columns to the dataframe
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

    annotate.annotate_parameters(df, data, keydict=keydict, inplace=True)
    annotate.annotate_startdatetime(df, data, inplace=True)
    annotate.explode_startdatetime(df, inplace=True)
    annotate.annotate_n_rewards(df, inplace=True)
    annotate.annotate_rig_id(df, data, inplace=True)
    annotate.fix_autorearded(df, inplace=True)
    annotate.annotate_cumulative_reward(df, data, inplace=True)
    annotate.annotate_filename(df, filename, inplace=True)

    for col in ('auto_rewarded', 'change_time', ):
        if col not in df.columns:
            df[col] = None

    # add some columns that require calculation

    df['trial_type'] = annotate.categorize_trials(df)
    df['endframe'] = annotate.get_end_frame(df, data)
    df['lick_frames'] = annotate.get_lick_frames(df, data)

    annotate.update_times(df, data, time=time, inplace=True)

    annotate.annotate_lick_vigor(df, data, inplace=True)
    annotate.calculate_latency(df,inplace=True)
    annotate.calculate_reward_rate(df,inplace=True)

    df['response'] = annotate.check_responses(df)
    df['trial_length'] = annotate.calculate_trial_length(df)
    df['endtime'] = annotate.get_end_time(df, data, time=time)
    df['color'] = annotate.assign_color(df)
    df['response_type'] = annotate.get_response_type(df)

    try:
        remove_repeated_licks(df,inplace=True)
    except Exception as e:
        print('FAILED TO REMOVE REPEATED LICKS')
        print(e)

    return df
