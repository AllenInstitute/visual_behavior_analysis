from __future__ import print_function
import warnings
import numpy as np
import pandas as pd
from .extended_trials import get_first_lick_relative_to_scheduled_change
from visual_behavior.change_detection.running.metrics import count_wraps
from scipy.ndimage import median_filter as medfilt


def parse_log(log_record):
    """ Parses a log record.

    This function looks for entries in core_data['log']. This function counts
    the number of errors for each level (e.g. 'INFO', 'ERROR' or 'CRITICAL',
    etc), and returns a dictionary like `{'ERROR': 2, 'CRITICAL': 1, 'DEBUG':
    42}` with the number of log messages at each log level.

    Parameters
    ----------
    log_record: str
        single log record with the syntax `"<levelname>::<name>::<message>"`

    Returns
    -------
    dict
        {'levelname': <levelname>, 'name': <name>, 'message': <message>}

    See Also
    --------
    count_read_errors : counts the number of log messages at each log level
    validate_no_read_errors : returns False if there are any read errors

    """
    levelname, name, message = log_record.split('::')
    return dict(
        levelname=levelname,
        name=name,
        message=message,
    )


def count_read_errors(core_data):
    """ Counts the number of errors raised at different levels when reading the data file.

    This function looks for entries in core_data['log']. This function counts
    the number of errors for each level (e.g. 'INFO', 'ERROR' or 'CRITICAL',
    etc), and returns a dictionary like `{'ERROR': 2, 'CRITICAL': 1, 'DEBUG':
    42}` with the number of log messages at each log level.

    Parameters
    ----------
    core_data: dictionary
        core visual_behavior data structure

    Returns
    -------
    dict
        keys are log levels and values are number of messages at that log level

    See Also
    --------
    validate_no_read_errors : returns False if there are any read errors

    """
    log = [parse_log(log_record) for log_record in core_data['log']]
    for rec in log:
        if rec['levelname'] in ['WARNING', 'ERROR', 'CRITICAL']:
            warnings.warn(
                '{} error while reading file: "{}"'.format(rec['levelname'], rec['message'])
            )
            print(rec)
    log = pd.DataFrame(log, columns=['levelname', 'name', 'message'])
    return log.groupby('levelname').size().to_dict()


def validate_no_read_errors(core_data):
    """ Validates that there were no errors raised when reading the data file.

    This function looks for entries in core_data['log']. If any errors are found
    at the 'ERROR' or 'CRITICAL' level, this check fails and this function
    returns `False`

    Parameters
    ----------
    core_data: dictionary
        core visual_behavior data structure

    Returns
    -------
    bool
        `True` if there were no read errors, else `False`


    See Also
    --------
    count_read_errors : counts the number of log messages at each log level

    """
    error_counts = count_read_errors(core_data)

    n_errors = error_counts.get('ERROR', 0) + error_counts.get('CRITICAL', 0)

    return (n_errors == 0)


def validate_running_data(core_data):
    '''
    for each sampling frame, the value of the encoder should be known
    takes core_data as input
    '''
    running_data = core_data['running']
    # make sure length matches length of time vector
    length_correct = len(running_data) == len(core_data['time'])
    # check if all speed values are the same
    all_same = all(running_data['speed'].values == running_data['speed'].values[0])

    return length_correct and not all_same


def validate_encoder_voltage(core_data, range_threshold=3, wrap_threshold=2):
    '''
    check for potentially anomolous encoder voltage traces

    Two failure modes we're checking here:
    1) voltage range is less than the range_threshold.
        A single rotation of the encoder will give a voltage range of ~5V, so a small range indicates
        that the encoder makes less than 1 full rotation.

    2) The ratio of forward wraps to backward wraps is less than the wrap_threshold:
        When the encoder spinning clockwise (rotation direction for forward motion by the mouse),
        the encoder transitions (wraps) from 5V to 0V once per rotation. If the encoder spinning CCW,
        the transition will be from 0V to 5V. So assuming that the mouse is walking forward, there should be
        far more forward wraps (5V-to-0V) than backward wraps (0V-to-5V). If the ratio is close to 1, this likely indicates
        an encoder that is oscillating back and forth across the zero-point due to noise

    Note that both failure modes can result from a stationary mouse (or a test session with no mouse).
    '''
    running = core_data['running']

    # filter out voltage artifacts
    filtered_vsig = medfilt(running['v_sig'], size=5)

    v_sig_range = filtered_vsig.max() - filtered_vsig.min()

    def get_wrap_ratio(forward_wraps, backward_wraps):
        if backward_wraps == 0:
            return np.inf
        else:
            return forward_wraps / backward_wraps

    wrap_ratio = get_wrap_ratio(
        count_wraps(running, 'forward'),
        count_wraps(running, 'backward')
    )

    if v_sig_range < range_threshold or wrap_ratio < wrap_threshold:
        return False
    else:
        return True


def validate_licks(core_data, lick_spout_present):
    """
    Validates that licks exist
    Ignores any session with 'lick_spout_present' = False (returns True)

    Parameters
    ----------
    core_data: dictionary
        legacy style output data structure

    Returns
    -------
    bool
        `True` if licks exist, else `False`

    """
    if lick_spout_present == False:
        return True
    else:
        return len(core_data['licks']) > 0


def validate_minimal_dropped_frames(core_data, allowable_fraction_dropped=0.01):
    '''
    ensures that no more than `allowable_fraction_dropped` frames are greater than 2/60. seconds long
    '''
    intervals = np.diff(core_data['time'])
    fraction_dropped = 1.0 * len(intervals[intervals >= (2 / 60.)]) / len(intervals)
    return fraction_dropped <= allowable_fraction_dropped


def validate_lick_before_scheduled_on_aborted_trials(core_data):
    '''
    if licks occur before a scheduled change time/flash, the trial ends
    Therefore, every aborted trial should have a lick before the scheduled change time
    '''
    flash_blank_buffer = 0.75  # allow licks to come this long after scheduled change time without failing validation
    aborted_trials = core_data['trials'][pd.isnull(core_data['trials']['change_time'])]
    # can only run this test if there are aborted trials
    if len(aborted_trials) > 0:
        first_lick = aborted_trials.apply(
            get_first_lick_relative_to_scheduled_change,
            axis=1,
        )
        # don't use nanmax. If there's a nan, we expect this to fail
        return np.max(first_lick.values) < flash_blank_buffer
    # if no aborted trials, return True
    else:
        return True


def validate_omitted_flashes_are_omitted(core_data):
    '''
    If a stimulus shows up in the omitted_stimuli log, there should never be another stimulus
    in the visual_stimuli log within three frames in either direction

    This catches an explicit bug that showed up in the first implementation of the omitted flash
    logic in the foraging2 stimulus code
    '''
    # core_data['omitted_stimuli'] will be an empty dataframe if no omitted flashes?
    try:
        omitted_stimuli = core_data['omitted_stimuli']
    except KeyError:  # is there a better way?
        warnings.warn(
            'no ommited_stimuli key on core_data object, '
            'this is likely really really bad'
        )

    if omitted_stimuli.empty:
        warnings.warn(
            'empty omitted_stimuli dataframe, '
            'skipping \'validate_omitted_flashes\''
        )
        return True

    for frame in omitted_stimuli['frame']:
        for offset in range(-3, 4, 1):
            if frame + offset in core_data['visual_stimuli']['frame'].tolist():
                return False
    else:
        return True


def get_licks_in_response_window(row, response_window=[0, 0], response_window_threshold=1 / 60 / 2):
    '''
    get all licks in the response window
    adds response windows_threshold to start of window and subtracts same from end of window:
        threshold is 1/2 frame width by defaul
        this gives a 1/2 frame buffer on timing
    '''
    if len(row['lick_times']) > 0 and not pd.isnull(row['change_time']):
        lick_times = np.array(row['lick_times'])
        in_window_indices = np.where(
            (lick_times - row['change_time'] >= response_window[0] + response_window_threshold)
            & (lick_times - row['change_time'] <= response_window[1] - response_window_threshold)
        )
        licks_in_window = np.array(row['lick_times'])[in_window_indices]
    else:
        licks_in_window = np.array([])
    return licks_in_window


def get_first_lick_reward_latency(row):
    if len(row['licks_in_response_window']) > 0:
        if len(row['reward_times']) == 0:
            return np.inf
        else:
            return row['reward_times'][0] - row['licks_in_response_window'][0]


def validate_reward_follows_first_lick_in_window(core_data, reward_latency_threshold=0.001):
    '''
    on non-autorewarded go trials, the first lick in the response window should have
    a reward that follows it by a short latency (reward_latency_threshold, default = 1 ms)
    '''
    tr = core_data['trials']

    # add some columns for convenience
    tr['licks_in_response_window'] = tr.apply(get_licks_in_response_window, axis=1, response_window=core_data['metadata']['response_window'])
    tr['number_of_licks_in_response_window'] = tr['licks_in_response_window'].map(lambda x: len(x))
    tr['lick_reward_latency'] = tr.apply(get_first_lick_reward_latency, axis=1)

    # find errant trials
    errant_trials = tr[
        (tr['number_of_licks_in_response_window'] > 0)  # trials with licks
        & (tr['rewarded'] == True)  # go trials (rewarded means reward was available, not necessarily delivered)
        & (tr['auto_rewarded'] == False)  # not autorewarded trials
        & (tr['lick_reward_latency'] > reward_latency_threshold)  # trials that exceed threshold
    ]

    # fail if there were any errant trials
    if len(errant_trials) > 0:
        for idx, trial in errant_trials.iterrows():
            # print out the reason for the failure
            print('trial {} had a latency of {:.5f} seconds between the first lick and the reward, the maximum allowable latency is {}'.format(
                idx,
                trial['lick_reward_latency'],
                reward_latency_threshold
            ))
        return False
    else:
        return True
