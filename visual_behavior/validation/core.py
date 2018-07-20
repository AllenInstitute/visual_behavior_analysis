import numpy as np
import pandas as pd
from .extended_trials import get_first_lick_relative_to_scheduled_change


def parse_log(log_record):
    levelname, name, message = log_record.split('::')
    return dict(
        levelname=levelname,
        name=name,
        message=message,
    )


def count_read_errors(core_data):
    log = [parse_log(log_record) for log_record in core_data['log']]
    log = pd.DataFrame(log, columns=['levelname', 'name', 'message'])
    return log.groupby('levelname').size().to_dict()


def validate_no_read_errors(core_data):
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


def validate_licks(core_data):
    '''
    validate that licks exist
    '''
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
