import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal

import visual_behavior.database as db
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.foraging import data_to_change_detection_core as data_to_change_detection_core_legacy


def load_running_df(bsid=None, pkl_path=None, camstim_type='foraging2'):
    '''
    loads running data from pkl file using VBA
    input is either the behavior session ID (bsid) or the pkl path (not both!)
    '''
    if bsid:
        pkl_path = db.get_pkl_path(int(bsid))

    data = pd.read_pickle(pkl_path)
    if camstim_type == 'foraging2':
        core_data = data_to_change_detection_core(data)
    else:
        core_data = data_to_change_detection_core_legacy(data)
    return core_data['running']


def identify_wraps(row, lower_threshold=1.5, upper_threshold=3.5):
    '''
    identify "wraps" in the voltage signal as any point where the crosses from 5V to 0V or vice-versa
    '''
    if row['v_sig'] < lower_threshold and row['v_sig_last'] > upper_threshold:
        return 1  # positive wrap
    elif row['v_sig'] > upper_threshold and row['v_sig_last'] < lower_threshold:
        return -1  # negative wrap
    else:
        return 0


def calculate_wrap_corrected_diff(row, max_diff=1, nan_transitions=False, v_max='v_in'):
    '''
    calculate the change in voltage at each timestep, accounting for the wraps.
    '''
    if v_max == 'v_in':
        v_max = row['v_in']

    if row['wrap_ID'] == 1:
        # unrwap the current value, subtract the last valueif nan_transitions:
        if nan_transitions:
            diff = np.nan
        else:
            diff = (row['v_sig'] + v_max) - row['v_sig_last']
    elif row['wrap_ID'] == -1:
        # unwrap the last value, subtract it from the current value
        if nan_transitions:
            diff = np.nan
        else:
            diff = row['v_sig'] - (row['v_sig_last'] + v_max)
    else:
        diff = row['v_sig'] - row['v_sig_last']

    if np.abs(diff) > max_diff:
        return np.nan
    else:
        return diff


def remove_outliers(df_in, column_to_filter, boolean_col, t_span, time_column='timestamps'):
    '''
    removes potential outliers using the following algorithm
        * operates only on 'column_to_filter'
        * For every value where 'boolean_col' is True:
            * identifies all values in time range of +/- t_span, excluding any other rows where 'boolean_col' is True
            * If value is greater than any other values in the range, sets value to max of other values in range
            * If values is less than any other values in range, sets value to min of other values in range

    Thus, possible outliers are identified in advance aand are not allowed to exceed range identifed by other values
    that have not been identified as outliers

    Note: this function is slow due to the for-loop.
    '''
    df = df_in.copy()
    df['outlier_removed'] = df[column_to_filter]
    df_to_filter = df[df[boolean_col]]
    for idx, row in df_to_filter.iterrows():
        t_now = row[time_column]  # NOQA F841
        local_vals = df.query('{0} >= @t_now - @t_span and {0} <= @t_now + @t_span and {1} == False'.format(time_column, boolean_col))[column_to_filter]
        df.at[idx, 'outlier_removed'] = np.clip(df.at[idx, column_to_filter], np.nanmin(local_vals), np.nanmax(local_vals))

    return df['outlier_removed']


def add_columns_and_unwrap(df, v_max='v_sig_max'):
    '''
    add columns to the running dataframe representing:
        v_sig_last: shifted voltage signal
        wrap_ID: 1 for postive wraps, -1 for negative wraps, 0 otherwise
        v_sig_diff: voltage derivative, after accounting for wraps
        v_sig_unwrapped: the cumulative voltage signal (no longer bounded by 0 to 5V)
    inputs:
        running_dataframe (with columns 'timestamps', 'v_in', 'v_sig')
        v_max - the value to use as the max voltage before the encoder 'wraps' back to 0V
                       'v_in' uses the measured input voltage
                       'v_sig_max' (default) uses the maximum observed voltage in the 'v_sig' column

    '''
    if v_max == 'v_sig_max':
        threshold = 5.1  # just in case some outlier got into the data, voltage should never exceed ~5V
        v_max = df[df['v_sig'] < threshold]['v_sig'].max()

    df['v_sig_last'] = df['v_sig'].shift()
    df['wrap_ID'] = df.apply(identify_wraps, axis=1)
    df['v_sig_diff'] = df.apply(calculate_wrap_corrected_diff, axis=1, nan_transitions=False, v_max=v_max)
    df['v_sig_unwrapped'] = np.cumsum(df['v_sig_diff']) + df['v_sig'].iloc[0]

    return df


def calculate_derivative(df, column_to_differentiate, time_column='timestamps'):
    '''a simple derivative function'''
    return df[column_to_differentiate].diff() / df[time_column].diff()


def calculate_speed(df, voltage_column, time_column='timestamps', v_max='v_sig_max'):
    '''a function to calculate speed from the voltage signal'''

    # which signal to use for v_max (converts delta_voltage to delta_rotation (radians))
    if v_max == 'v_sig_max':
        # use the max of the signal
        threshold = 5.1  # just in case some outlier got into the data, voltage should never exceed ~5V
        v_max = df[df['v_sig'] < threshold]['v_sig'].max()
    elif v_max == 'v_in':
        # use the time-varying v_in signal
        v_max = df['v_in']
    delta_theta = df[voltage_column].diff() / v_max * 2 * np.pi  # delta theta at each step in radians

    wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter, converted to cm
    running_radius = 0.5 * (2.0 * wheel_diameter / 3.0)  # assume the animal runs at 2/3 the distance from the wheel center

    df_temp = pd.DataFrame({time_column: df[time_column], 'theta_cumulative': np.cumsum(delta_theta)})

    speed = calculate_derivative(df_temp, column_to_differentiate='theta_cumulative', time_column=time_column) * running_radius  # linear speed in cm/s

    return speed


def add_speed(df_in, column_label, voltage_column='v_sig_unwrapped', v_max='v_sig_max', remove_outliers_at_wraps=True, zscore_thresold=5, time_column='timestamps'):
    '''
    add columns for speed (cm/s), acceleration (cm/s^2) and jerk (cm/s^3) to the dataframe
    inputs:
        df_in (dataframe): encoder dataframe with (at minimum) columns for v_in and v_sig
        column_label (string): what to call the output columns (default = 'raw', eg. speed_raw, acceleration_raw, jerk_raw)
        voltage_column (string): the column on which to calculate derivative to calculate speed
        v_max (string): how maxiumum encoder voltage should be defined. See add_columns_and_unwrap for detailed explanation. (default = 'v_sig_max')
        remove_outliers_at_wrap (boolean): If True, ensures that velocity at voltage wrap does not exceed range of velocities in nearby +/- 0.25 second window (default = True)
        zscore_threshold (float): any remaining velocity values that are more than `zscore_threshold` SDs from the mean will be replaced with NaN (default = 5)
    returns:
        a new dataframe with additional columns
    '''
    df_in = add_columns_and_unwrap(df_in, v_max=v_max)
    speed_label = 'speed_{}'.format(column_label)
    df_in[speed_label] = calculate_speed(df_in, voltage_column=voltage_column, v_max=v_max, time_column=time_column)

    if remove_outliers_at_wraps:
        df_in['wrap_bool'] = df_in['wrap_ID'] != 0
        df_in[speed_label + '_pre_wrap_correction'] = df_in[speed_label]
        df_in[speed_label] = remove_outliers(
            df_in,
            speed_label,
            'wrap_bool',
            t_span=0.25
        )

    # replace any values that exceed the z-score threshold with NaN
    df_in['zscored_speed_{}'.format(column_label)] = stats.zscore(
        df_in[speed_label].fillna(df_in[speed_label].mean())
    )
    df_in.loc[
        df_in[df_in['zscored_speed_{}'.format(column_label)].abs() >= zscore_thresold].index.values,
        speed_label
    ] = np.nan

    return df_in


def apply_lowpass_filter(df_in, column_to_filter, cutoff_frequency=10, N=3, time_column='timestamps'):
    '''
    applies a butterworth lowpass filter to `column_to_filter'
    inputs:
        df_in (dataframe): dataframe to filter
        column_to_filter (str): name of column to filter
        cutoff_frequency (float): cutoff frequency for lowpass filter, in Hz (default = 10). Cannot exceed Nyquist frequency.
        N (int): order of filter (default = 3)
        time_column (str): name of time column (default = 'timestamps')

    '''
    fs = 1 / df_in[time_column].diff().median()
    b, a = signal.butter(N, Wn=cutoff_frequency, fs=fs, btype='lowpass')

    filtered_data = signal.filtfilt(b, a, df_in[column_to_filter].fillna(0))

    return filtered_data


def process_encoder_data(running_data_df, time_column='timestamps', v_max='v_sig_max', filter_cutoff_frequency=10, remove_outliers_at_wraps=True, zscore_thresold=5):
    '''
    encoder data is processed following two major steps:
    * Remove artifacts in the voltage derivative (speed) at threshold crossings
    * Apply a simple Butterworth lowpass filter with a 10 Hz cutoff frequency to the resulting signal to remove oscillatory measurement errors

    inputs:
        running_data_df (dataframe): dataframe to process. Must include columns for time, v_in, v_sig
        time_column (str): name of time column (default = 'timestamps')
        v_max (str): value to use for v_max (either 'v_in' or 'v_sig_max', default = 'v_sig_max')
        filter_cutoff_frequency (float): cutoff frequency for lowpass filter, in Hz (default = 10). Cannot exceed Nyquist frequency.
    returns:
        a new dataframe with following columns:
            - {time_column}: unchanged from input df
            - v_sig and v_in: unchanged from input df
            - v_sig_unwrapped: an unwrapped version of 'v_sig', after correcting for wrap artifacts
            - speed (overwrites existing speed column if it exists): contains fully processed low-pass-filtered speed (cm/s)
            - acceleration: first derivative of `speed` column (in units of cm/s^2)
            - jerk: second derivative of speed column (in units of cm/s^3)
            - wrap_ID: 1 if the encoder wrapped from ~5V to ~0V on this timestep, -1 if it wrapped from ~0V to ~5V, 0 otherwise
            - speed_raw_pre_wrap_correction: raw encoder speed in cm/s, before correcting for wrap artifacts
            - speed_raw: raw encoder speed in cm/s, after correcting for wrap artifacts

    '''
    running_data_df = add_speed(
        running_data_df,
        column_label='raw',
        v_max=v_max,
        remove_outliers_at_wraps=remove_outliers_at_wraps,
        zscore_thresold=zscore_thresold
    )
    filtered_speed = apply_lowpass_filter(
        running_data_df,
        column_to_filter='speed_raw',
        cutoff_frequency=filter_cutoff_frequency,
        time_column=time_column,
    )
    running_data_df['speed'] = filtered_speed
    running_data_df['acceleration'] = calculate_derivative(running_data_df, 'speed', time_column=time_column)
    running_data_df['jerk'] = calculate_derivative(running_data_df, 'acceleration', time_column=time_column)

    # remove some columns before returning:
    cols_to_drop = ['dx', 'v_sig_last', 'v_sig_diff', 'wrap_bool', 'zscored_speed_raw']
    for col in cols_to_drop:
        running_data_df = running_data_df.drop(columns=[col]) if col in running_data_df.columns else running_data_df

    return running_data_df
