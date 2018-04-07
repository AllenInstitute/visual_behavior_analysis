import logging
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from six import iteritems

from scipy.signal import medfilt
from ...analyze import calc_deriv, rad_to_dist


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


def annotate_licks(trial):
    """Annotate lick information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log

    Returns
    -------
    dict
        lick_times: list of float
            times (s) in which a lick occured
        lick_frames: list of int
            frame indices in which a lick occurred

    Notes
    -----
    - time is seconds since start of experiment
    """
    return {
        "lick_times": [lick[0] for lick in trial["licks"]],
        # "lick_frames": [lick[1] for lick in trial["licks"]],
    }


def annotate_optogenetics(trial):
    """Annotate optogenetics information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log

    Returns
    -------
    dict
        optogenetics: bool or None
            whether or not optogenetics was used for the trial, or None if optogenetics
            isnt enabled

    Notes
    -----
    - this idea of existance checking wasn't my choice...
    """
    return {
        "optogenetics": trial["trial_params"].get("optogenetics"),
    }


def annotate_responses(trial):
    """Annotate response-related information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log

    Returns
    -------
    dict
        response_frame: integer
            frame index of the response
        response_time: float
            time (s) of the response
        response_type: str
            type of response
        response_latency: float, np.nan, np.inf
            difference between stimulus change and response, np.nan if a change doesn't
            occur, np.inf if a change occurs but a response doesn't occur

    Notes
    -----
    - time is seconds since start of experiment
    - 03/13/18: justin did not know what to use for `response_type` key so it will be None
    """
    try:
        change_frame, change_time = trial["stimulus_changes"][0][2:4]  # assume one stimulus change per trial, idx 3, 4 will have the frame, time
    except IndexError:
        return {
            # "response_frame": None,
            "response_time": None,
            "response_type": None,
            "response_latency": np.nan,
        }

    
    for (idx, (name, direction, time, frame), ) in enumerate(trial["events"]):
        if name == "hit":
            return {
                # "response_frame": frame,
                "response_time": time,
                "response_latency": time - change_time,
                "response_type": None,
            }
    else:
        return {
            # "response_frame": None,
            "response_time": None,
            "response_type": None,
            "response_latency": np.inf,
        }


def annotate_rewards(trial):
    """Annotate rewards-related information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log

    Returns
    -------
    dict
        auto_rewarded_trial: bool
            whether or not the trial is autorewarded
        cumulative_volume: float
            cumulative water volume dispensed relative to start of experiment
        cumulative_reward_number: int
            cumulative reward number dispensed relative to start of experiment
        reward_times: list of float
            times (s) in which rewards occurred
        reward_frames: list of int
            frame indices in which rewards occurred
        rewarded: bool
            whether or not it's a rewarded trial

    Notes
    -----
    - time is seconds since start of experiment
    """
    return {
        "auto_rewarded_trial": trial["trial_params"]["auto_reward"] if trial['trial_params']['catch']==False else None,
        "cumulative_volume": trial["cumulative_volume"],
        "cumulative_reward_number": trial["cumulative_rewards"],
        "reward_volume": trial.get("volume_dispensed"),  # this doesn't exist in our current iteration of foraging2 outputs but should exist very soon
        "reward_times": [reward[1] for reward in trial["rewards"]],
        "reward_frames": [reward[2] for reward in trial["rewards"]],
        "rewarded": trial["trial_params"]["catch"]==False,  # justink said: assume go trials are catch != True, assume go trials are the only type of rewarded trials
    }


def annotate_schedule_time(trial, pre_change_time):
    """Annotate time/scheduling-related information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log
    pre_change_time: float
        minimum time before a stimulus change occurs in the trial

    Returns
    -------
    dict
        start_time: float
            time (s) of the start of the trial
        start_frame: int
            frame index of the start of the trial
        scheduled_change_time: float
            time (s) scheduled for the change to occur
        end_time: float
            time (s) of the end of the trial
        end_frame: int
            frame index of the end of the trial

    Notes
    -----
    - time is seconds since start of experiment
    - as of 03/13/18 scheduled_change_time isn't retrievable and will be None
    """
    warnings.warn("`scheduled_change_time` isn't retrievable and will be None")
    try:
        start_time, start_frame = trial["events"][0][2:4]
        # end_time, end_frame = trial["events"][-1][2:4]
    except IndexError:
        return {
            "start_time": None,
            "start_frame": None,
            "scheduled_change_time": None,
            "end_time": None,
            "end_frame": None,
        }

    return {
        "start_time": start_time,
        "start_frame": start_frame,
        "scheduled_change_time": None,
        # "end_time": end_time,
        # "end_frame": end_frame,
    }


def annotate_stimuli(trial, stimuli):
    """Annotate stimuli-related information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log
    stimuli: Mapping
        foraging2 shape stimuli mapping

    Returns
    -------
    dict
        initial_image_name: str
            name of the initial image
        initial_image_category: str
            category of the initial image
        change_image_name: str
            name of the change image
        change_image_category: str
            category of the change image
        change_frame: int
            frame index of "change event"
        change_time: float
            time (s) of the "change event"
        change_orientation: float
            the orientation of the change image
        change_contrast: float
            the contrast of the change image
        initial_orientation: float
            the orientation of the initial image
        initial_contrast: float
            the contrast of the initial image
        delta_orientation: float
            difference in orientation between the initial image and the change image
        stimulus_on_frames: list of int
            frame indices of whether or not the stimulus was drawn

    Notes
    -----
    - time is seconds since start of experiment
    - the initial image is the image shown before the "change event" occurs
    - the change image is the image shown after the "change event" occurs
    - assumes only one stimulus change can occur in a single trial and that
    each change will only be intra-classification (ie: "natural_images",
    "gratings", etc.)
    - currently only supports the following stimuli types and none of their
    subclasses: DoCGratingStimulus, DoCImageStimulus
    - if you are mixing more than one classification of stimulus, it will
    resolve the first group_name, stimulus_name pair it encounters...so maybe
    name things uniquely everywhere...
    """
    try:
        stimulus_change = trial["stimulus_changes"][0]
    except IndexError:
        return {
            "initial_image_category": None,
            "initial_image_name": None,
            "change_image_name": None,
            "change_image_category": None,
            "change_frame": None,
            "change_time": None,
            "change_orientation": None,
            "change_contrast": None,
            "initial_orientation": None,
            "initial_contrast": None,
            "delta_orientation": None,
            "stimulus_on_frames": None,
        }

    (from_group, from_name, ), (to_group, to_name), frame, time = stimulus_change
    _, stim_dict = _resolve_stimulus_dict(stimuli, from_group)
    implied_type = stim_dict["obj_type"]

    # initial is like ori before, contrast before...etc
    # type is image or grating, changes is a dictionary of changes make sure each change type is lowerccase string...

    if implied_type in ("DoCGratingStimulus", ):
        first_frame, last_frame = _get_trial_frame_bounds(trial)
        initial_changes, change_changes = _get_stimulus_attr_changes(
            stim_dict, frame, first_frame, last_frame
        )

        initial_orientation = initial_changes.get("ori")
        change_orientation = change_changes.get("ori")

        if initial_orientation and change_orientation:
            delta_orientation = initial_orientation - change_orientation
        else:
            delta_orientation = np.nan

        return {
            "initial_image_category": None,
            "initial_image_name": None,
            "change_image_name": None,
            "change_image_category": None,
            "change_frame": frame,
            "change_time": time,
            "change_orientation": change_orientation,
            "change_contrast": change_changes.get("constrast"),
            "initial_orientation": initial_orientation,
            "initial_contrast": initial_changes.get("contrast"),
            "delta_orientation": delta_orientation,
            "stimulus_on_frames": np.array(stim_dict["draw_log"], dtype=np.bool),
        }
    elif implied_type in ("DoCImageStimulus", ):
        return {
            "initial_image_category": from_group,
            "initial_image_name": from_name,
            "change_image_name": to_name,
            "change_image_category": to_group,
            "change_frame": frame,
            "change_time": time,
            "change_orientation": None,
            "change_contrast": None,
            "initial_contrast": None,
            "delta_orientation": None,
            "stimulus_on_frames": np.array(stim_dict["draw_log"], dtype=np.bool),
        }
    else:
        raise ValueError("invalid implied type: {}".format(implied_type))


def _get_stimulus_attr_changes(stim_dict, change_frame, first_frame, last_frame):
    """
    Notes
    -----
    - assumes only two stimuli are ever shown
    - converts attr_names to lowercase
    """
    initial_attr = {}
    change_attr = {}

    for attr_name, set_value, set_frame, set_time in stim_dict["set_log"]:
        if first_frame <= set_frame < change_frame:
            initial_attr[attr_name.lower()] = set_value
        elif change_frame <= set_frame < last_frame:
            change_attr[attr_name.lower()] = set_value
        else:
            pass

    return initial_attr, change_attr


def _get_trial_frame_bounds(trial):
    """Get start, stop frame for a trial

    Parameters
    ----------
    trial: Mapping
        foraging2 style trial_log trial

    Notes
    -----
    - assumes the first frame is the frame of the first event and the last frame
    is the frame of the last event
    """
    return trial["events"][0][-1], trial["events"][-1][-1]


def _resolve_stimulus_dict(stimuli, group_name):
    """
    Notes
    -----
    - will return the first stimulus dict with group_name present in its list of
    stim_groups
    """
    for classification_name, stim_dict in iteritems(stimuli):
        if group_name in stim_dict["stim_groups"]:
            return classification_name, stim_dict
    else:
        raise ValueError("unable to resolve stimulus_dict from group_name...")


def annotate_trials(trial):
    """Annotate trial information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log

    Returns
    -------
    dict
        trial_type: str
            "go" or "catch"
        trial_duration: float or None
            duration of trial in seconds or None if fails
    """
    try:
        trial_duration = trial["events"][-1][2] - trial["events"][0][2]
    except IndexError:
        trial_duration = None

    return {
        "trial_type": "catch" if trial["trial_params"]["catch"] else "go",
        "trial_duration": trial_duration,
    }


# def annotate_visual_stimuli(trial, stim_table, times):
#     draw_log = stim_table["draw_log"]
#     trial_start_frame = trial["events"][0][3]
#     trial_stop_frame = trial["events"][-1][3]
#
#     stim_change = trial["stimulus_changes"]
#
#     initial_stimulus_category, initial_stimulus_name = stim_change[0]
#     change_stimulus_category, change_stimulus_name = stim_change[1]
#     stim_change_frame = stim_change[2]
#
#     display_array = []
#     previous_draw = None
#
#     draw_log = draw_log.copy()
#
#
#     change_arr = np.where(np.diff(draw_log, axis=0))
#     np.insert(change_arr, 0, 0)
#     np.append(change_arr, len(change_arr) - 1)
#
#
#     for idx, draw_frame in \
#             enumerate(draw_log[trial_start_frame:trial_stop_frame + 1]):
#         if draw_frame != previous_draw:
#             if idx < stimulus_change_frame:
#                 display_array.append(
#                     (idx, initial_stimulus_category, initial_stimulus_name)
#                 )
#             else:
#                 display_array.append(
#                     (idx, change_stimulus_category, change_stimulus_name)
#                 )
#
#
# def _get_stimuli_intervals(stim_table):
#     draw_log = stim_table["draw_log"]
#
#     change_arr = np.diff(np.where(draw_log), axis=0)
#
#     start_stim = np.where(change_arr == 1)
#     end_stim = np.where(change_arr == -1)


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
    """Get params dict supplied to a foraging2 experiment

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
    params = deepcopy(exp_data["items"]["behavior"].get("params", {}))
    params.update(exp_data["items"]["behavior"].get("cl_params", {}))
    return params


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
            (trial["rewards"][0][1], trial["rewards"][0][2], )
            for (idx, trial) in get_trials(exp_data).iterrows()
            if trial["rewards"]
        ], dtype=np.float),
        columns=["time", "frame", ]
    )


def get_response_window(data):
    """Get response window set for all trials

    Parameters
    ----------
    data: Mapping
        foraging2 experiment output data

    Returns
    -------
    tuple:
        float
            minimum time in seconds after a change in which a response should occur
        float
            maximum time in seconds after a change in which a response should occur
    """
    return data["items"]["behavior"] \
        .get("params", {}) \
        .get("response_window")


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
    - the first value of speed is assumed to be nan
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
    return {
        k: np.array(stim_dict["draw_log"], dtype=np.bool)
        for k, stim_dict in iteritems(exp_data["items"]["behavior"]["stimuli"])
    }


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
    return np.hstack((0, get_vsyncs(exp_data))).cumsum() / 1000.0  # cumulative time


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
        [None] * behavior_items_or_top_level(exp_data)["update_count"]
    )  # less stable but that coverage tho...


def get_datetime_info(exp_data):
    """Get all datetime information from exp data

    Parameters
    ----------
    exp_data: Mapping
        experiment data

    Returns
    -------
    dict
        startdatetime: datetime.datetime
            start datetime for the experiment
        date: datetime.date
            lower precision
        year: int
        month: int
        day: int
        hour: int
        dayofweek: int
            based in Monday
    Notes
    -----
    - "explodes" datetime because scientist(s) want it
    """
    startdatetime = exp_data.get("start_time")

    if startdatetime is not None:
        return {
            "startdatetime": startdatetime,
            "year": startdatetime.year,
            "month": startdatetime.month,
            "day": startdatetime.day,
            "hour": startdatetime.hour,
            "date": startdatetime.date(),
            "dayofweek": startdatetime.weekday(),
        }
    else:
        return {
            "startdatetime": startdatetime,
            "year": None,
            "month": None,
            "day": None,
            "hour": None,
            "date": None,
            "dayofweek": None,
        }


def get_mouse_id(exp_data):
    """Gets the id of the mouse

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    str or None
        id of the mouse or None if not found
    """
    return behavior_items_or_top_level(exp_data) \
        .get("config", {}) \
        .get("behavior", {}) \
        .get("mouse_id")


def get_pre_change_time(data):
    """Gets the prechange time associated with an experiment

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    float
        time (s) of the prechange time
    """
    return behavior_items_or_top_level(data) \
        .get("config", {}) \
        .get("DoC", {}) \
        .get("pre_change_time", 0)


def get_blank_duration_range(exp_data):
    """Gets the range of the blank duration

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    tuple
        float
            time (s) of start of the blank duration
        float
            time (s) of end of the blank duration

    Notes
    -----
    - the blank duration occurs at the very start of a trial and starts/ends before the
    prechange window
    """
    return behavior_items_or_top_level(exp_data) \
        .get("config", {}) \
        .get("DoC", {}) \
        .get("blank_duration_range")


def get_user_id(exp_data):
    """Gets the id of the user associated with the experiment

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    str or None
        the id of the user or None if not found
    """
    return behavior_items_or_top_level(exp_data) \
        .get("cl_params", {}) \
        .get("user_id")


def get_session_id(exp_data):
    """Gets the id associated with the experiment session

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    int or None
        id of experiment session or None if not found
    """
    return behavior_items_or_top_level(exp_data).get("session_id")


def get_stimuli(data):
    """Get stimuli mapping

    Parameters
    ----------
    data: Mapping
        foraging2 experiment output data

    Returns
    -------
    Mapping
        stimuli mapping

    Notes
    -----
    - assumes that the only type of stimuli will be "natural_scenes" stimuli
    """
    return data["items"]["behavior"]["stimuli"]


def get_filename(exp_data):
    """Gets the filename associated with the experiment data

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    str or None
        filename or None if not found
    """
    return behavior_items_or_top_level(exp_data).get("behavior_path")


def get_device_name(exp_data):
    """Get the name of the device on which the experiment was run

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    str or None
        name of the device or None if not found

    Notes
    -----
    - the device is assumed to be the device running the agent process that runs the
    experiment
    - 03/13/18: device_name is not yet accessible and will be None
    """
    logger.warn("get_device_name is not working and will return None...")
    return behavior_items_or_top_level(exp_data) \
        .get("platform_info", {}) \
        .get("device_name")


def get_image_path(data):
    """Get path to pickle that stores images to be drawn

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    dict
        stimulus_name, image_path

    Notes
    -----
    - image_path is None if image_path couldnt be found
    """
    return {
        s_name: s_dict.get("image_path")
        for s_name, s_dict in iteritems(data["items"]["behavior"]["stimuli"])
    }


def get_session_duration(exp_data):
    """Gets the actual duration of the experiment session

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    float
        duration of the experiment session in time (s)

    Notes
    -----
    - time is seconds since start of experiment
    """
    time = get_time(exp_data)
    return time[-1] - time[0]


def get_stimulus_duration(exp_data):
    """Gets the configured stimulus duration set for each trial

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    float or None
        configured stimulus duration in time (s)

    Notes
    -----
    - time is seconds since start of experiment
    """
    return behavior_items_or_top_level(exp_data) \
        .get("config", {}) \
        .get("DoC", {}) \
        .get("stimulus_window")


def get_stimulus_times(exp_data):
    """I don't think I'm using this...
    """
    events = behavior_items_or_top_level(exp_data) \
        .get("trial_log", [])

    stimulus_events = []

    for (from_group, to_group, img_name, frame, time, ) in events:
        stimulus_events.append({
            "index": time,
            "stimulus": img_name,
            "frame": frame,
        })


def get_trial_log(data):
    """Get a trial log mapping from foraging2 shaped data

    Parameters
    ----------
    data: Mapping
        foraging2 experiment output data

    Returns
    -------
    Mapping
        foraging2 shaped trial log
    """
    return behavior_items_or_top_level(data)["trial_log"]


def get_stimulus_on_frames(exp_data):
    """pd.Series frame, boolean whether shown
    """
    return pd.Series([
        was_drawn == 1
        for was_drawn in get_unified_draw_log(exp_data)
    ])


def get_task_id(data):
    """Get task id

    Parameters
    ----------
    data : Mapping
        foraging2 output data

    Returns
    -------
    str or None
        task id or None if not found
    """
    return behavior_items_or_top_level(data) \
        .get("config", {}) \
        .get("behavior", {}) \
        .get("task_id")


def get_scheduled_trial_duration(data):
    """Get scheduled trial duration

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float or None
        trial duration in seconds or None if not found
    """
    return behavior_items_or_top_level(data) \
        .get("params", {}) \
        .get("trial_duration")


def get_unified_draw_log(data):
    unified_draw_log = []
    stimuli = data["items"]["behavior"]["stimuli"]
    stimuli_names = tuple(stimuli.keys())

    try:
        unified_draw_log = np.array(stimuli[stimuli_names[0]]["draw_log"])  # more efficient adds for large draw logs

        for stim_name in stimuli_names[1:]:
            unified_draw_log += np.array(stimuli[stim_name]["draw_log"])
    except IndexError:
        pass

    return unified_draw_log
