import pandas as pd

from .extract import load_trials
from .transform import annotate_parameters, explode_startdatetime, \
    annotate_n_rewards, annotate_rig_id, annotate_startdatetime, \
    annotate_cumulative_reward, annotate_filename, fix_autorearded, \
    annotate_lick_vigor, update_times
from ...utilities import get_lick_frames, remove_repeated_licks, \
    calculate_latency, calculate_reward_rate, categorize_trials, get_end_frame, \
    check_responses, calculate_trial_length, get_end_time, assign_color, \
    get_response_type


def data_to_monolith(data, time=None):
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

    if time is None:
        df = load_trials(data)
    else:
        df = load_trials(data, time=time)

    df = df[~pd.isnull(df['reward_times'])].reset_index(drop=True)

    annotate_parameters(df, data, keydict=keydict, inplace=True)
    annotate_startdatetime(df, data, inplace=True)
    explode_startdatetime(df, inplace=True)
    annotate_n_rewards(df, inplace=True)
    annotate_rig_id(df, data, inplace=True)
    fix_autorearded(df, inplace=True)
    annotate_cumulative_reward(df, data, inplace=True)
    # annotate_filename(df, filename, inplace=True)

    for col in ('auto_rewarded', 'change_time', ):
        if col not in df.columns:
            df[col] = None

    # add some columns that require calculation

    df['trial_type'] = categorize_trials(df)
    df['endframe'] = get_end_frame(df, data)
    df['lick_frames'] = get_lick_frames(df, data)
    # df['last_lick'] = get_last_licktimes(df,data)

    update_times(df, data, time=time, inplace=True)

    annotate_lick_vigor(df, data, inplace=True)
    calculate_latency(df)
    calculate_reward_rate(df)

    df['response'] = check_responses(df)
    df['trial_length'] = calculate_trial_length(df)
    df['endtime'] = get_end_time(df, data, time=time)
    df['color'] = assign_color(df)
    df['response_type'] = get_response_type(df)

    try:
        remove_repeated_licks(df)
    except Exception as e:
        print('FAILED TO REMOVE REPEATED LICKS')
        print(e)

    return df
