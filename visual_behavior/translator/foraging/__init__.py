import warnings
import pandas as pd
import numpy as np
from scipy.signal import medfilt
from ...utilities import calc_deriv, rad_to_dist

warnings.warn(
    "support for the loading stimulus_code outputs will be deprecated in a future version",
    PendingDeprecationWarning,
)


def data_to_change_detection_core(data, time=None):
    """Core data structure to be used across all analysis code?

    Parameters
    ----------
    data: Mapping
        legacy style output data structure

    Returns
    -------
    pandas.DataFrame
        core data structure for the change detection task

    Notes
    -----
    - currently doesn't require or check that the `task` field in the
    experiment data is "DoC" (Detection of Change)
    """
    if time is None:
        time = load_time(data)

    return {
        "time": time,
        "metadata": load_metadata(data),
        "licks": load_licks(data, time=time),
        "trials": load_trials(data, time=time),
        "running": load_running_speed(data, time=time),
        "rewards": load_rewards(data, time=time),
        "visual_stimuli": None,  # not yet implemented
    }


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


def load_trials(data, time=None):
    """ Returns the trials generated in an experiment.

    Parameters
    ----------
    data : dict, unpickled experiment (or path to pickled object)

    Returns
    -------
    trials : pandas DataFrame

    """

    columns = (
        'auto_rewarded',
        'change_contrast',
        'change_frame',
        'change_image_category',
        'change_image_name',
        'change_ori',
        'change_time',
        'cumulative_reward_number',
        'cumulative_volume',
        'delta_ori',
        'index',
        'initial_contrast',
        'initial_image_category',
        'initial_image_name',
        'initial_ori',
        'lick_times',
        'optogenetics',
        'publish_time',
        'response_latency',
        'response_time',
        'response_type',
        'reward_frames',
        'reward_times',
        'reward_volume',
        'rewarded',
        'scheduled_change_time',
        'startframe',
        'starttime',
    )

    forced_types = {
        'initial_ori': float,
        'change_ori': float,
        'delta_ori': float,
        'initial_contrast': float,
        'change_contrast': float,
    }

    trials = (
        pd.DataFrame(data['triallog'], columns=columns)
        .astype(dtype=forced_types)
    )

    forced_string = (
        'initial_image_category',
        'initial_image_name',
        'change_image_category',
        'change_image_name',
    )

    def stringify(x):
        try:
            if np.isfinite(x) == True:
                return str(x)
            else:
                return ''
        except TypeError:
            if x is None:
                return ''

        return str(x)

    for col in forced_string:
        trials[col] = trials[col].map(stringify)

    trials['change_frame'] = trials['change_frame'].map(lambda x: int(x) if np.isfinite(x) else None)
    trials["change_image_category"] = trials["change_image_category"].apply(lambda x: x if x else np.nan)  # use np.nan instead of NoneType
    trials["change_image_name"] = trials["change_image_name"].apply(lambda x: x if x else np.nan)  # use np.nan instead of NoneType
    trials["initial_image_category"] = trials["initial_image_category"].apply(lambda x: x if x else np.nan)  # use np.nan instead of NoneType
    trials["initial_image_name"] = trials["initial_image_name"].apply(lambda x: x if x else np.nan)  # use np.nan instead of NoneType

    # make scheduled_change_time relative
    trials["scheduled_change_time"] = trials["scheduled_change_time"] - trials['starttime']

    return trials


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

    # accel = calc_deriv(speed, time)
    # jerk = calc_deriv(accel, time)

    running_speed = pd.DataFrame({
        'time': time,
        'speed': speed,
        # 'acceleration (cm/s^2)': accel,
        # 'jerk (cm/s^3)': jerk,
    })
    return running_speed
