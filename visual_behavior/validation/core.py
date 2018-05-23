import numpy as np

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

def validate_minimal_dropped_frames(core_data,allowable_fraction_dropped=0.01):
    '''
    ensures that no more than `allowable_fraction_dropped` frames are greater than 2/60. seconds long
    '''
    intervals = np.diff(core_data['time'])
    fraction_dropped = len(intervals[intervals >= (2/60.)]) / len(intervals)
    return fraction_dropped <= allowable_fraction_dropped
