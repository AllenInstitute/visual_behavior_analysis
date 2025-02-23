import logging
import uuid
import numpy as np
import pandas as pd
from copy import deepcopy
from six import iteritems
import re

from ...analyze import compute_running_speed  # , calc_deriv
from ...uuid_utils import make_deterministic_session_uuid
from ...utilities import local_time
from ... import devices

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
        "optogenetics": trial["trial_params"].get("optogenetics", False),
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
        response_time: float
            time (s) of the response
        response_type: str
            type of response
        response_latency: float, np.nan, np.inf
            difference between stimulus change and response, np.nan if a change doesn't
            occur, np.inf if a change occurs but a response doesn't occur
        rewarded: boolean, None


    Notes
    -----
    - time is seconds since start of experiment
    - 03/13/18: justin did not know what to use for `response_type` key so it will be None
    """

    for event in trial['events']:
        if event[0] == 'stimulus_changed':
            change_event = event
            break
        elif event[0] == 'sham_change':
            change_event = event
            break
        else:
            change_event = None

    for event in trial['events']:
        if event[0] == 'hit':
            response_event = event
            break
        elif event[0] == 'false_alarm':
            response_event = event
            break
        else:
            response_event = None

    if change_event is None:
        # aborted
        return {
            # "response_frame": frame,
            "response_type": [],
            "response_time": [],
            "response_latency": None,
            "change_frame": None,
            "change_time": None,
        }

    elif change_event[0] == 'stimulus_changed':
        if response_event is None:
            # miss
            return {
                # "response_frame": frame,
                "response_type": [],
                "response_time": [],
                "change_time": change_event[2],
                "change_frame": change_event[3],
                "response_latency": np.inf,
            }
        elif response_event[0] == 'hit':
            # hit
            return {
                # "response_frame": frame,
                "response_type": [],
                "response_time": [],
                "change_time": change_event[2],
                "change_frame": change_event[3],
                "response_latency": response_event[2] - change_event[2],
            }

        else:
            raise Exception('something went wrong')
    elif change_event[0] == 'sham_change':
        if response_event is None:
            # correct reject
            return {
                # "response_frame": frame,
                "response_type": [],
                "response_time": [],
                "change_time": change_event[2],
                "change_frame": change_event[3],
                "response_latency": np.inf,
            }
        elif response_event[0] == 'false_alarm':
            # false alarm
            return {
                # "response_frame": frame,
                "response_type": [],
                "response_time": [],
                "change_time": change_event[2],
                "change_frame": change_event[3],
                "response_latency": response_event[2] - change_event[2],
            }
        else:
            raise Exception('something went wrong')
    else:
        raise Exception('something went wrong')


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
        "auto_rewarded_trial": trial["trial_params"]["auto_reward"] if trial['trial_params']['catch'] == False else None,
        "cumulative_volume": trial["cumulative_volume"],
        "cumulative_reward_number": trial["cumulative_rewards"],
        "reward_volume": sum([r[0] for r in trial.get("rewards", [])]),
        "reward_times": [reward[1] for reward in trial["rewards"]],
        "reward_frames": [reward[2] for reward in trial["rewards"]],
        "rewarded": trial["trial_params"]["catch"] == False,  # justink said: assume go trials are catch != True, assume go trials are the only type of rewarded trials
    }


def annotate_schedule_time(trial, pre_change_time, initial_blank_duration):
    """Annotate time/scheduling-related information

    Parameters
    ----------
    trial: Mapping
        foraging2 shape trial from trial_log
    pre_change_time: float
        minimum time before a stimulus change occurs in the trial
    initial_blank_duration: float
        time at start before the prechange window

    Returns
    -------
    dict
        start_time: float
            time (s) of the start of the trial
        start_frame: int
            frame index of the start of the trial
        scheduled_change_time: float
            time (s) scheduled for the change to occur

    Notes
    -----
    - time is seconds since start of experiment
    """
    try:
        start_time, start_frame = trial["events"][0][2:4]
        end_time, end_frame = trial["events"][-1][2:4]
        trial_length = end_time - start_time
    except IndexError:
        return {
            "start_time": None,
            "start_frame": None,
            "trial_length": None,
            "scheduled_change_time": None,
            "end_time": None,
            "end_frame": None,
        }

    return {
        "start_time": start_time,
        "start_frame": start_frame,
        "trial_length": trial_length,
        "scheduled_change_time": (
            pre_change_time +
            initial_blank_duration +
            trial["trial_params"]["change_time"]
        ),  # adding start time will put this in time relative to start of exp, change_time is relative to after prechange + initial_blank_duration, using () because of annoying flake8 bug + noqa directive not working for multiline
        "end_time": end_time,
        "end_frame": end_frame,
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
        (from_group, from_name, ), (to_group, to_name), _, change_frame = stimulus_change
        _, stim_dict = _resolve_stimulus_dict(stimuli, from_group)
    except IndexError:
        trial_start_frame = trial["events"][0][3]
        category, from_group, from_name = _resolve_initial_image(
            stimuli,
            trial_start_frame
        )

        to_group, to_name = from_group, from_name
        change_frame = np.nan

        if category:
            stim_dict = stimuli[category]
        else:
            stim_dict = {}

    implied_type = stim_dict["obj_type"]

    # initial is like ori before, contrast before...etc
    # type is image or grating, changes is a dictionary of changes make sure each change type is lowerccase string...

    if implied_type in ("DoCGratingStimulus", ):
        first_frame, last_frame = _get_trial_frame_bounds(trial)

        initial_changes, change_changes = _get_stimulus_attr_changes(
            stim_dict, change_frame, first_frame, last_frame
        )

        initial_orientation = initial_changes.get("ori")
        initial_contrast = initial_changes.get("contrast")
        change_orientation = change_changes.get("ori", initial_orientation)
        change_contrast = change_changes.get("constrast", initial_contrast)

        if initial_orientation is not None and change_orientation is not None:
            delta_orientation = change_orientation - initial_orientation
        else:
            delta_orientation = np.nan

        return {
            "initial_image_category": '',
            "initial_image_name": '',
            "change_image_name": '',
            "change_image_category": '',
            "change_orientation": change_orientation,
            "change_contrast": change_contrast,
            "initial_orientation": initial_orientation,
            "initial_contrast": initial_contrast,
            "delta_orientation": delta_orientation,
        }
    elif implied_type in ("DoCImageStimulus", ):
        return {
            "initial_image_category": from_group,
            "initial_image_name": from_name,
            "change_image_name": to_name,
            "change_image_category": to_group,
            "change_orientation": None,
            "change_contrast": None,
            "initial_orientation": None,
            "initial_contrast": None,
            "delta_orientation": None,
        }
    else:
        raise ValueError("invalid implied type: {}".format(implied_type))


def _resolve_initial_image(stimuli, start_frame):
    """Attempts to resolve the initial image for a given start_frame for a trial

    Parameters
    ----------
    stimuli: Mapping
        foraging2 shape stimuli mapping
    start_frame: int
        start frame of the trial

    Returns
    -------
    initial_image_category_name: str
        stimulus category of initial image
    initial_image_group: str
        group name of the initial image
    initial_image_name: str
        name of the initial image
    """
    max_frame = float("-inf")
    initial_image_group = ''
    initial_image_name = ''
    initial_image_category_name = ''

    for stim_category_name, stim_dict in iteritems(stimuli):
        for set_event in stim_dict["set_log"]:
            set_frame = set_event[3]
            if set_frame <= start_frame and set_frame >= max_frame:
                initial_image_group = initial_image_name = set_event[1]  # hack assumes initial_image_group == initial_image_name, only initial_image_name is present for natual_scenes
                # initial_image_group, initial_image_name = change_event[0]
                initial_image_category_name = stim_category_name
                max_frame = set_frame

    return initial_image_category_name, initial_image_group, initial_image_name


def _get_stimulus_attr_changes(stim_dict, change_frame, first_frame, last_frame):
    """
    Notes
    -----
    - assumes only two stimuli are ever shown
    - converts attr_names to lowercase
    - gets the net attr changes from the start of a trial to the end of a trial
    """
    initial_attr = {}
    change_attr = {}

    for attr_name, set_value, set_time, set_frame in stim_dict["set_log"]:
        if set_frame <= first_frame:
            initial_attr[attr_name.lower()] = set_value
        elif set_frame <= last_frame:
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


def get_vsig(exp_data):
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

    vsig = behavior_items_or_top_level(exp_data)["encoders"][0]["vsig"]
    if len(vsig) == 0:
        dx = get_dx(exp_data)
        vsig = np.empty_like(dx)
        vsig[:] = np.nan
    return vsig


def get_vin(exp_data):
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
    vin = behavior_items_or_top_level(exp_data)["encoders"][0]["vin"]
    if len(vin) == 0:
        dx = get_dx(exp_data)
        vin = np.empty_like(dx)
        vin[:] = np.nan
    return vin


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
    lick_frames = exp_data['items']['behavior']['lick_sensors'][0]['lick_events']

    time = time or get_time(exp_data)  # that coverage syntax tho...

    # there's an occasional bug where the number of logged frames is one greater
    # than the number of vsync intervals. If the animal licked on this last frame
    # it will cause an error here. This fixes the problem.
    # see: https://github.com/AllenInstitute/visual_behavior_analysis/issues/572
    #    & https://github.com/AllenInstitute/visual_behavior_analysis/issues/379
    if len(lick_frames) > 0 and lick_frames[-1] == len(time):
        lick_frames = lick_frames[:-1]
        logger.error('removed last lick - it fell outside of intervalsms range')

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

    if "response_window" in params:
        params["response_window"] = list(params["response_window"])  # tuple to list

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
        .get("config", {}) \
        .get("DoC", {}) \
        .get("response_window")


def get_even_sampling(data):
    """Get status of even_sampling

    Parameters
    ----------
    data: Mapping
        foraging2 experiment output data

    Returns
    -------
    bool:
        True if even_sampling is enabled
    """

    stimuli = data['items']['behavior']['stimuli']
    for stimuli_group_name, stim in iteritems(stimuli):
        if stim['obj_type'].lower() == 'docimagestimulus' and stim['sampling'] in ['even', 'file']:
            return True
    return False


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

    dx_raw = get_dx(exp_data)
    v_sig = get_vsig(exp_data)
    v_in = get_vin(exp_data)

    if len(time) < len(dx_raw):
        logger.error('intervalsms record appears to be missing entries')
        dx_raw = dx_raw[:len(time)]
        v_sig = v_sig[:len(time)]
        v_in = v_in[:len(time)]

    speed = compute_running_speed(dx_raw, time, v_sig, v_in)

    # accel = calc_deriv(speed, time)
    # jerk = calc_deriv(accel, time)

    # remove first values which is always high
    speed = speed[1:]
    time = time[1:]

    running_speed = pd.DataFrame({
        'time': time,
        'frame': range(len(time)),
        'speed': speed,
        'dx': dx_raw,
        'v_sig': v_sig,
        'v_in': v_in,
        # 'acceleration (cm/s^2)': accel,
        # 'jerk (cm/s^3)': jerk,
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
    behavior_items = behavior_items_or_top_level(exp_data)
    return behavior_items.get("config", {}).get("behavior", {}).get("mouse_id") or \
        behavior_items.get("cl_params", {}).get("mouse_id")


def get_platform_info(exp_data):
    """Gets platform info for the experiment

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    dict

    """
    return exp_data['platform_info']


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
    behavior_items = behavior_items_or_top_level(exp_data)
    return behavior_items.get("config", {}).get("behavior", {}).get("user_id", '') or \
        behavior_items.get("cl_params", {}).get("user_id", '')


def get_session_id(exp_data, create_if_missing=False):
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

    if 'session_uuid' in exp_data:
        return uuid.UUID(exp_data["session_uuid"])
    elif create_if_missing:

        start_time_datetime_local = local_time(exp_data["start_time"], timezone='America/Los_Angeles')
        mouse_id = get_mouse_id(exp_data)

        logger.warning('`session_uuid` not found. generating a deterministic UUID')
        behavior_session_uuid = make_deterministic_session_uuid(
            mouse_id,
            start_time_datetime_local,
        )
        return behavior_session_uuid
    else:
        logger.warning('`session_uuid` not found. returning None')
        return None


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
    """
    return exp_data['platform_info']['computer_name']


def convert_rig_id(mpe_notation):
    '''
    converts MPE rig notation to historical Visual Behavior notation

    ### MPE nomenclature rules:
        * If it's a behavior box, it's 'BEH.' + {cluster} + '-Box' + {rig_number}
        * If it's a physiology rig, it's {cluster} + '.' + {rig_number} + '-STIM' (or '-Stim') where cluster = 'CAM2P', 'MESO' or 'NP'

    ### Visual Behavior Group nomenclature rules:
        * name is {cluster} + {rig_number}
    '''
    rig_map = {
        'CAM2P.3-STIM': '2P3',
        'CAM2P.4-STIM': '2P3',
        'CAM2P.5-Stim': '2P5',
        'MESO.1-Stim': 'MS1',
        'MESO.2-Stim': 'MS2',
        'NP.1-Stim': 'NP1',
        'NP.2-Stim': 'NP2',
        'NP.3-Stim': 'NP3',
        'NP.4-Stim': 'NP4',
    }

    # this is the pattern, for behavior boxes: 'BEH' + {cluster_name} + '-Box' + {rig_number}
    pattern = re.compile(r'BEH\.(?P<cluster_name>[A-Z])-Box(?P<rig_number>\d+)')
    regex_output = pattern.match(mpe_notation)

    if regex_output:
        # if the notation mateches the pattern, regex output will be a dictionary with keys 'cluster_name', 'rig_number'
        rig_id = regex_output['cluster_name'] + regex_output['rig_number']
    else:
        # otherwise, the regex output will be None, in which case we check the rig_map dictionary above
        rig_id = rig_map.get(mpe_notation, mpe_notation)
        # note that the syntax above will return the input key if the key doesn't exist in the dictionary
        # so, if all else fails, this function will just return the input
    return rig_id


def get_rig_id(exp_data):
    """ Get the ID of the rig on which the experiment was run

    Rig ID is now included in the behavior pickle files (as of XXXX).
    For loading old data we still need to use the mapping in
    devices to get the rig ID from the computer name.

    Parameters
    ----------
    exp_data: Mapping
        foraging2 experiment output data

    Returns
    -------
    rig_id: str
        id of the rig or 'unknown' if not found
    """
    try:
        rig_id = exp_data['comp_id']
    except KeyError:
        # If there is no 'rig_id' field, we need to get the ID by using devices.
        # This should only happen with old data from before 'rig_id' was included.
        experiment_date = exp_data['start_time']
        computer_name = get_device_name(exp_data)
        if experiment_date < devices.VALID_BEFORE_DATE:
            # This experiment comes from a time when the mappings in the devices.py
            # file are valid.
            return devices.get_rig_id(computer_name)
        else:
            logger.warning(("rig_id unknown, no valid mapping exists for computer "
                            "{} on {}").format(computer_name, experiment_date))
            return 'unknown'
    return convert_rig_id(rig_id)


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
    return data["items"]["behavior"].get("config", {}).get("behavior", {}).get("task_id") or \
        data["items"]["behavior"]["params"].get("task_id")


def get_scheduled_trial_length(data):
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
    doc_config = behavior_items_or_top_level(data).get("DoC", {})
    return doc_config["stimulus_window"] + doc_config["pre_change_time"] \
        + doc_config["initial_blank"]  # we probably want to hard-fail if one of these isnt present


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


def get_stimulus_distribution(data):
    """Get stimulus distribution type

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    str or None
        distribution type or None if not found
    """
    return data["items"]["behavior"]["config"]["DoC"]["change_time_dist"]


def get_delta_mean(data):
    """Get delta mean

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float or None
        delta mean or None if not found
    """
    return data["items"]["behavior"]['config']["DoC"]["change_time_scale"]


def get_initial_blank_duration(data):
    """Get initial blank duration for each trial
    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float or None
        blank duration in seconds or None if not found
    """
    return data["items"]["behavior"]["config"]["DoC"]["initial_blank"]


def get_stage(data):
    """Get stage name

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    str
        stage name
    """
    return data["items"]["behavior"]["params"].get("stage")


def get_reward_volume(data):
    """Get reward volume per reward for experiment

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float or None
        reward volume per reward or None if not found
    """
    return data["items"]["behavior"]["config"]["reward"]["reward_volume"]


def get_auto_reward_volume(data):
    """Get reward volume per auto reward for experiment

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float or None
        auto reward volume per reward or None if not found
    """
    return data["items"]["behavior"]["config"]["DoC"]["auto_reward_volume"]


def get_warm_up_trials(data):
    """Get number of warm up trials for a foraging2 behavior session

    Parameters
    ----------
    data : Mapping

    Returns
    -------
    int or None
        number of warm up trials or None if not found
    """
    return data["items"]["behavior"]["config"]["DoC"]["warm_up_trials"]


def get_stimulus_window(data):
    """Get the stimulus window set for a foraging2 behavior session

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float or None
        stimulus window
    """
    return data["items"]["behavior"]["config"]["DoC"]["stimulus_window"]


def get_volume_limit(data):
    """Get the volume limit set for a foraging2 behavior session

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float or None
        volume limit or None if not found
    """
    return data["items"]["behavior"]["config"]["behavior"]["volume_limit"]


def get_failure_repeats(data):
    """Get the number of failure repeats set for a foraging2 behavior session

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    int or None
        number of failure repeats or None if not found
    """
    return data["items"]["behavior"]["config"]["DoC"]["failure_repeats"]


def get_catch_frequency(data):
    """Get the catch frequency set for a foraging2 behavior session

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float or None
        catch frequency or None if not found
    """
    return data["items"]["behavior"]["config"]["DoC"]["catch_freq"]


def get_auto_reward_delay(data):
    """Get the auto reward delay set for a foraging2 behavior session

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float
    """
    return data["items"]["behavior"]["config"]["DoC"].get("auto_reward_delay", 0.0)


def get_free_reward_trials(data):
    """Get free reward trials set for a foraging2 behavior session

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    int or none
        number of free reward trials or None if not found
    """
    return data["items"]["behavior"]["config"]["DoC"]["free_reward_trials"]


def get_min_no_lick_time(data):
    """Get minimum no lick time set for a foraging2 behavior session

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float or none
        minimum no lick time or None if not found
    """
    return data["items"]["behavior"]["config"]["DoC"]["min_no_lick_time"]


def get_max_session_duration(data):
    """Get max session duration (min) set for a foraging2 behavior session

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    float or none
        max session duration (min) or None if not found
    """
    return data["items"]["behavior"]["config"]["DoC"]["max_task_duration_min"]


def get_abort_on_early_response(data):
    """Get abort on early response boolean set for a foraging2 behavior session

    Parameters
    ----------
    data: Mapping
        foraging2 output data

    Returns
    -------
    boolean or none
        abort on early response boolean or None if not found
    """
    return data["items"]["behavior"]["config"]["DoC"]["abort_on_early_response"]


def get_periodic_flash(data):
    periodic_flash = data['items']['behavior']['config']['DoC']['periodic_flash']
    if periodic_flash is None:
        pass
    else:
        try:
            periodic_flash = tuple([float(val) for val in periodic_flash])
        except ValueError:
            logger.error('`periodic_flash` is `{} ({})`, which is not supported. Setting this to `None`'.format(periodic_flash, type(periodic_flash)))
            periodic_flash = None

    return periodic_flash
