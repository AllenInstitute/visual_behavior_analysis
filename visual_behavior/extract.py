import logging
import numpy as np
import pandas as pd

from scipy.signal import medfilt
from .analyze import calc_deriv, rad_to_dist


logger = logging.getLogger(__name__)

"""extracts a series (pandas Series) or feature (object) doesn't affect
supplied experiment data object

series are guaranteed to be syncronized across the same timebase
(or whatever quantity)
"""

"""Most of these are from Derric's translator object in camstim

New idea...no more time or frames sometimes, all output in frames, the
downstream can handle time pairing

or maybe just use time...this was more complex than i realized? frame to time
relationship might not be the same for
"""


def behavior_items_or_top_level(exp_data):
    """Some kind of weird thing related to derrics output...TODO ask him to stop
    """
    try:
        return exp_data["items"]["behavior"]
    except KeyError:
        logger.warn("got an unexepcted shape", exc_info=True)
        return exp_data


def get_dx(exp_data):
    """Get wheel rotation for each frame

    Parameters
    ----------
    exp_data : Mapping
        experiment data

    Returns
    -------
    numpy.ndarray
        log of whether or not a stimulus was presented, 1 dimensional with index
        representing frame number and value position of the frame

    Notes
    -----
    - asssumes data is in the Foraging2 format
    - no idea what the units are
    """
    return behavior_items_or_top_level(exp_data)["encoders"][0]["dx"]


def get_licks(exp_data, time=None):
    """Returns each lick in an experiment.

    Parameters
    ----------
    exp_data : Mapping
        experiment data
    time : np.array, optional
        array of times for each stimulus frame

    Returns
    -------
    pandas.DataFrame
        dataframe with `frame` and `time` associated with each lick

    Notes
    -----
    - TODO remove the time thing, standardize where time evaluation occurs
    """
    lick_frames = [
        trial["licks"][0][1]
        for (idx, trial) in get_trials(exp_data).iterrows()
        if trial["licks"]
    ]

    time = time or get_time(exp_data)  # that coverage syntax tho...

    lick_times = [time[frame] for frame in lick_frames]

    return pd.DataFrame(data={"frame": lick_frames, "time": lick_times, })


def get_params(exp_data):
    """Get params

    Parameters
    ---------
    exp_data : Mapping
        exactly what you think it is...

    Returns
    -------
    Mapping:
        params

    Notes
    -----
    - asssumes data is in the Foraging2 format
    """
    return exp_data.get("params", {})


def get_rewards(exp_data):
    """Get rewards

    Parameters
    ----------
    exp_data: Mapping
        experiment data

    Returns
    -------
    pandas.DataFrame
        dataframe with `frame` and `time` associated with each reward

    Notes
    -----
    - asssumes data is in the Foraging2 format
    - no idea what the units are
    """
    return pd.DataFrame(
        data=np.array([
            trial["rewards"][0]
            for (idx, trial) in get_trials(exp_data).iterrows()
            if trial["rewards"]
        ], dtype=np.float),
        columns=["time", "frame", ]
    )


def get_running_speed(exp_data, smooth=False, time=None):
    """Get running speed

    Parameters
    ----------
    exp_data: Mapping
        experiment data
    smooth: boolean, default=False
        not implemented
    time: numpy.array, default=None
        array of times for each stimulus frame

    Returns
    -------
    pandas.DataFrame
        dataframe with 'time', 'speed (cm/s)', 'acceleration (cm/s^2)',
        'jerk (cm/s^3)'

    Notes
    -----
    - asssumes data is in the Foraging2 format
    """
    if time is None:
        time = get_time(exp_data)
        logger.info("`time` not passed. using intervalms as `time`")

    dx = get_dx(exp_data)
    dx = medfilt(dx, kernel_size=5)  # remove big, single frame spikes in encoder values
    dx = np.cumsum(dx)  # wheel rotations

    if len(time) != len(dx):
        raise ValueError("dx and time must be the same length")

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


def get_stimulus_log(exp_data):
    """Get whether or not a stimulus was presented on a frame by frame basis

    Parameters
    ----------
    exp_data: Mapping
        experiment data

    Returns
    -------
    numpy.ndarray
        log of whether or not a stimulus was presented, 1 dimensional with index
        representing frame number and value as a boolean whether or not a
        stimulus was shown

    Notes
    -----
    - asssumes data is in the Foraging2 format
    """
    return np.array(
        behavior_items_or_top_level(exp_data)["stimuli"]["natual_scenes"]["draw_log"],
        dtype=np.bool
    )  # will likely break in the future, explicit is better here tho...


def get_time(exp_data):
    """Get cumulative time for each vsync frame

    Parameters
    ----------
    exp_data: Mapping
        experiment data

    Returns
    -------
    numpy.ndarray
        cumulative time for each frame in the experiment, 1 dimensional with
        index representing frame number and value as a float representing time
        since the start of the experiment

    Notes
    -----
    - asssumes data is in the Foraging2 format
    """
    return np.hstack((0, get_vsyncs(exp_data))).cumsum()  # cumulative time


def get_trials(exp_data):
    """Get trials

    Parameters
    ----------
    exp_data: Mapping
        experiment data

    Returns
    -------
    pandas dataframe
        each row represents a trial in an experiment session

    Notes
    -----
    - asssumes data is in the Foraging2 format
    """
    return pd.DataFrame(
        behavior_items_or_top_level(exp_data).get("trial_log", [])
    )


def get_vsyncs(exp_data):
    """Get vsync times

    Parameters
    ----------
    exp_data: Mapping
        experiment data

    Returns
    -------
    numpy.ndarray
        cumulative time for each vsync, 1 dimensional with index representing
        vsync number and value as a float representing time since the start of
        the experiment

    Notes
    -----
    - asssumes data is in the Foraging2 format
    """
    return behavior_items_or_top_level(exp_data).get(
        "intervalsms",
        [16.0] * behavior_items_or_top_level(exp_data)["update_count"]
    )  # less stable but that coverage tho...
