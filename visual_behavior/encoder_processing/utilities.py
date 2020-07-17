import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from visual_behavior.encoder_processing.spline_regression import spline_regression
import visual_behavior.database as db
from visual_behavior.translator.foraging2 import data_to_change_detection_core


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


def calculate_wrap_corrected_diff(row, max_diff=1, nan_transitions=False):
    '''
    calculate the change in voltage at each timestep, accounting for the wraps.
    '''
    if row['wrap_ID'] == 1:
        # unrwap the current value, subtract the last valueif nan_transitions:
        if nan_transitions:
            diff = np.nan
        else:
            diff = (row['v_sig'] + row['v_in']) - row['v_sig_last']
    elif row['wrap_ID'] == -1:
        # unwrap the last value, subtract it from the current value
        if nan_transitions:
            diff = np.nan
        else:
            diff = row['v_sig'] - (row['v_sig_last'] + row['v_in'])
    else:
        diff = row['v_sig'] - row['v_sig_last']

    if np.abs(diff) > max_diff:
        return np.nan
    else:
        return diff


def add_columns_and_unwrap(df):
    '''
    add columns to the running dataframe representing:
        v_sig_last: shifted voltage signal
        wrap_ID: 1 for postive wraps, -1 for negative wraps, 0 otherwise
        v_sig_diff: voltage derivative, after accounting for wraps
        v_sig_unwrapped: the cumulative voltage signal (no longer bounded by 0 to 5V)
    '''
    df['v_sig_last'] = df['v_sig'].shift()
    df['wrap_ID'] = df.apply(identify_wraps, axis=1)
    df['v_sig_diff'] = df.apply(calculate_wrap_corrected_diff, axis=1, nan_transitions=False)
    df['v_sig_unwrapped'] = np.cumsum(df['v_sig_diff']) + df['v_sig'].iloc[0]

    return df


def calculate_derivative(df, column_to_differentiate, time_column='time'):
    '''a simple derivative function'''
    return np.gradient(df[column_to_differentiate], df[time_column])


def calculate_speed(df, voltage_column, time_column='time'):
    '''a function to calculate speed from the voltage signal'''
    delta_theta = df[voltage_column].diff() / df['v_in'] * 2 * np.pi  # delta theta at each step in radians

    wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter
    running_radius = 0.5 * (2.0 * wheel_diameter / 3.0)  # assume the animal runs at 2/3 the distance from the wheel center

    df_temp = pd.DataFrame({'time': df[time_column], 'theta_cumulative': np.cumsum(delta_theta)})

    speed = calculate_derivative(df_temp, column_to_differentiate='theta_cumulative', time_column=time_column) * running_radius  # linear speed in cm/s

    return speed


def make_visualization(df_to_plot, n_knot_factors):

    fig, ax = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    ax[0].plot(
        df_to_plot['time'],
        df_to_plot['v_sig_unwrapped'],
        marker='.',
        linewidth=3
    )

    legend = ['raw (unwrapped)']

    for n_knot_factor in n_knot_factors:
        ax[0].plot(
            df_to_plot['time'],
            df_to_plot['v_spline_smoothed_F={}'.format(n_knot_factor)],
        )
        legend.append('v_spline_smoothed_F={}'.format(n_knot_factor))

    ax[0].set_ylabel('voltage')
    ax[0].legend(legend, loc='upper right')

    ax[0].set_title('voltage (position)')

    ax[1].set_title('voltage derivative (speed)')

    ax[1].plot(
        df_to_plot['time'],
        df_to_plot['speed_raw']
    )

    legend = ['raw (unwrapped)']
    for n_knot_factor in n_knot_factors:
        ax[1].plot(
            df_to_plot['time'],
            df_to_plot['speed_spline_smoothed_F={}'.format(n_knot_factor)],
        )
        legend.append('speed_spline_smoothed_F={}'.format(n_knot_factor))
    ax[1].legend(legend, loc='upper right')

    ax[2].plot(
        df_to_plot['time'],
        df_to_plot['acceleration_raw']
    )

    legend = ['raw (unwrapped)']
    for n_knot_factor in n_knot_factors:
        ax[2].plot(
            df_to_plot['time'],
            df_to_plot['acceleration_spline_smoothed_F={}'.format(n_knot_factor)],
        )
        legend.append('acceleration_spline_smoothed_F={}'.format(n_knot_factor))
    ax[2].legend(legend, loc='upper right')
    ax[2].set_title('voltage second derivative (acceleration)')

    ax[3].plot(
        df_to_plot['time'],
        df_to_plot['jerk_raw']
    )

    legend = ['raw (unwrapped)']
    for n_knot_factor in n_knot_factors:
        ax[3].plot(
            df_to_plot['time'],
            df_to_plot['jerk_spline_smoothed_F={}'.format(n_knot_factor)],
        )
        legend.append('jerk_spline_smoothed_F={}'.format(n_knot_factor))
    ax[3].legend(legend, loc='upper right')
    ax[3].set_title('voltage third derivative (jerk)')
    ax[3].set_xlabel('time (s)')
    ax[3].set_xlim(df_to_plot['time'].min(), df_to_plot['time'].max())

    fig.tight_layout()

    return fig, ax


def mean_squared_jerk(df_in, voltage_col):
    df = df_in.copy()
    df['speed'] = calculate_speed(df, voltage_column=voltage_col)
    df['acceleration'] = calculate_derivative(df, 'speed')
    df['jerk'] = calculate_derivative(df, 'acceleration')
    return np.mean(df['jerk']**2)


def jerk_std(df_in, voltage_col):
    df = df_in.copy()
    df['speed'] = calculate_speed(df, voltage_column=voltage_col)
    df['acceleration'] = calculate_derivative(df, 'speed')
    df['jerk'] = calculate_derivative(df, 'acceleration')
    return np.std(df['jerk'])


def total_jerk(df_in, voltage_col='v_spline_smoothed'):
    df = df_in.copy()
    df['speed'] = calculate_speed(df, voltage_column=voltage_col)
    df['acceleration'] = calculate_derivative(df, 'speed')
    df['jerk'] = calculate_derivative(df, 'acceleration')
    return np.sum(df['jerk'].abs())


def jerk_integral(df_in, voltage_col='v_spline_smoothed'):
    df = df_in.copy()
    df['speed'] = calculate_speed(df, voltage_column=voltage_col)
    df['acceleration'] = calculate_derivative(df, 'speed')
    df['jerk'] = calculate_derivative(df, 'acceleration')
    return np.sum(df['jerk'].abs())


def mean_squared_error(df_in, col_1, col_2):
    return np.mean((df_in[col_1] - df_in[col_2])**2)


def optimization_function(row, k1, k2):
    return k1 * row['jerk_std'] + k2 * row['mean_squared_error']


def calculate_cost_function(df_in):
    optimization_results = []
    df_temp = df_in.copy()
    for F in range(1, 100):
        n_knots = len(df_temp) / F
        df_temp['v_spline_smoothed'] = spline_regression(df_temp, col_to_smooth='v_sig_unwrapped', n_knots=n_knots)
        df_temp['speed_spline_smoothed'] = calculate_speed(df_temp, voltage_column='v_spline_smoothed')

        optimization_results.append({
            'n_knot_factor': F,
            'n_knots': n_knots,
            'mean_squared_jerk': mean_squared_jerk(df_temp, voltage_col='v_spline_smoothed'),
            'total_jerk': total_jerk(df_temp, voltage_col='v_spline_smoothed'),
            'jerk_std': jerk_std(df_temp, voltage_col='v_spline_smoothed'),
            'mean_squared_error': mean_squared_error(df_temp, 'speed_spline_smoothed', 'speed_raw')
        })
    optimization_results = pd.DataFrame(optimization_results)
    optimization_results['j'] = optimization_results.apply(optimization_function, axis=1, k1=0.01, k2=1)

    return optimization_results


def plot_optimization_results(optimization_results):
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 5))
    ax[0].plot(optimization_results['n_knot_factor'], optimization_results['jerk_std'])
    ax[1].plot(optimization_results['n_knot_factor'], optimization_results['mean_squared_error'])
    ax[2].plot(optimization_results['n_knot_factor'], optimization_results['j'])
    ax[2].set_ylim(-1, 200)

    optimal_index = optimization_results['j'].argmin()
    for i in range(3):
        ax[i].axvline(optimization_results.iloc[optimal_index]['n_knot_factor'])

    ax[0].set_title('standard deviation of jerk')
    ax[1].set_title('mean squared error of velocity estimate (relative to raw)')
    ax[2].set_title('optimization function (k1*total_jerk + k2*MSE)')
    fig.tight_layout()

    return fig, ax


def calculate_optimal_knot_factor(optimization_results):
    optimal_index = optimization_results['j'].argmin()
    return optimization_results.iloc[optimal_index]['n_knot_factor']


def apply_spline_regression(df_in, n_knot_factor):
    df_in = add_columns_and_unwrap(df_in.copy())
    df_in['speed_raw'] = calculate_speed(df_in, voltage_column='v_sig_unwrapped')
    df_in['acceleration_raw'] = calculate_derivative(df_in, 'speed_raw')
    df_in['jerk_raw'] = calculate_derivative(df_in, 'acceleration_raw')

    n_knots = len(df_in) / n_knot_factor
    df_in['v_spline_smoothed_F={}'.format(n_knot_factor)] = spline_regression(df_in, col_to_smooth='v_sig_unwrapped', n_knots=n_knots)
    df_in['speed_spline_smoothed_F={}'.format(n_knot_factor)] = calculate_speed(df_in, voltage_column='v_spline_smoothed_F={}'.format(n_knot_factor))
    df_in['acceleration_spline_smoothed_F={}'.format(n_knot_factor)] = calculate_derivative(df_in, 'speed_spline_smoothed_F={}'.format(n_knot_factor))
    df_in['jerk_spline_smoothed_F={}'.format(n_knot_factor)] = calculate_derivative(df_in, 'acceleration_spline_smoothed_F={}'.format(n_knot_factor))

    return df_in
