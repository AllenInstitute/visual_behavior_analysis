from __future__ import print_function
import six
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from .utilities import calc_deriv, rad_to_dist
from functools import wraps


def data_or_pkl(func):
    """ Decorator that allows a function to accept a pickled experiment object
    or a path to the object.

    >>> @data_or_pkl
    >>> def print_keys(data):
    >>>     print data.keys()

    """
    @wraps(func)
    def pkl_wrapper(first_arg, *args, **kwargs):
        if isinstance(first_arg, six.string_types):
            return func(pd.read_pickle(first_arg), *args, **kwargs)
        else:
            return func(first_arg, *args, **kwargs)

    return pkl_wrapper

@data_or_pkl
def load_metadata(data):

    fields = (
        'startdatetime',
        'rig_id',
        'computer_name',
        'startdatetime',
        'rewardvol',
        'params',
        'mouseid',
        'response_window',
        'task',
        'stage',
        'stoptime',
        'userid',
        'lick_detect_training_mode',
        'blankscreen_on_timeout',
        'stim_duration',
        'blank_duration_range',
        'delta_minimum',
        'stimulus_distribution',
        'stimulus',
        'delta_mean',
        'trial_duration',
    )
    metadata = {}

    for field in fields:
        try:
            metadata[field] = data[field]
        except KeyError:
            metadata[field] = None

    metadata['n_stimulus_frames'] = len(data['vsyncintervals'])

    return metadata


@data_or_pkl
def load_trials(data, time=None):
    """ Returns the trials generated in an experiment.

    Parameters
    ----------
    data : dict, unpickled experiment (or path to pickled object)

    Returns
    -------
    trials : pandas DataFrame

    """
    trials = pd.DataFrame(data['triallog'])

    return trials


@data_or_pkl
def load_time(data):
    """ Returns the times of each stimulus frame in an experiment.

    Parameters
    ----------
    data : dict, unpickled experiment (or path to pickled object)

    Returns
    -------
    time : numpy array

    """
    vsync = np.hstack((0, data['vsyncintervals']))
    return (vsync.cumsum()) / 1000.0


@data_or_pkl
def load_licks(data, time=None):
    """ Returns each lick in an experiment.

    Parameters
    ----------
    data : dict, unpickled experiment (or path to pickled object)
    time : np.array, optional
        array of times for each stimulus frame

    Returns
    -------
    licks : numpy array

    """
    licks = pd.DataFrame(data['responselog'], columns=['frame', 'time'])

    if time is None:
        time = load_time(data)

    licks['time'] = time[licks['frame']]

    licks = licks[licks['frame'].diff() != 1]

    return licks


@data_or_pkl
def load_rewards(data, time=None):
    """ Returns each reward in an experiment.

    Parameters
    ----------
    data : dict, unpickled experiment (or path to pickled object))
    time : np.array, optional
        array of times for each stimulus frame

    Returns
    -------
    rewards : numpy array

    """
    try:
        reward_frames = data['rewards'][:, 1].astype(int)
    except IndexError:
        reward_frames = np.array([], dtype=int)
    if time is None:
        time = load_time(data)
    rewards = pd.DataFrame(dict(
        frame=reward_frames,
        time=time[reward_frames],
    ))
    return rewards


@data_or_pkl
def load_running_speed(data, smooth=False, time=None):
    if time is None:
        print('`time` not passed. using vsync from pkl file')
        time = load_time(data)

    dx = np.array(data['dx'])
    dx = medfilt(dx, kernel_size=5)  # remove big, single frame spikes in encoder values
    dx = np.cumsum(dx)  # wheel rotations

    time = time[:len(dx)]

    speed = calc_deriv(dx, time)
    speed = rad_to_dist(speed)

    if smooth:
        # running_speed_cm_per_sec = pd.rolling_mean(running_speed_cm_per_sec, window=6)
        raise NotImplementedError

    accel = calc_deriv(speed, time)
    jerk = calc_deriv(accel, time)

    running_speed = pd.DataFrame({
        'time': time,
        'speed (cm/s)': speed,
        'acceleration (cm/s^2)': accel,
        'jerk (cm/s^3)': jerk,
    })
    return running_speed
