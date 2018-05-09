import pandas as pd
from dateutil import tz

from .extract import get_trial_log, get_stimuli, get_pre_change_time, \
    annotate_licks, annotate_rewards, annotate_optogenetics, annotate_responses, \
    annotate_schedule_time, annotate_stimuli, get_user_id, get_mouse_id, \
    get_blank_duration_range, get_device_name, get_session_duration, \
    get_stimulus_duration, get_task_id, get_response_window, get_licks, \
    get_running_speed, get_params, get_time, get_trials, \
    get_stimulus_distribution, get_delta_mean, get_initial_blank_duration, \
    get_stage, get_reward_volume, get_auto_reward_volume


from .extract_stimuli import get_visual_stimuli


def data_to_change_detection_core(data):
    """Core data structure to be used across all analysis code?

    Parameters
    ----------
    data: Mapping
        foraging2 style output data structure

    Returns
    -------
    pandas.DataFrame
        core data structure for the change detection task

    Notes
    -----
    - currently doesn't require or check that the `task` field in the
    experiment data is "DoC" (Detection of Change)
    """
    return {
        "metadata": data_to_metadata(data),
        "time": data_to_time(data),
        "licks": data_to_licks(data),
        "trials": data_to_trials(data),
        "running": data_to_running(data),
        "rewards": data_to_rewards(data),
        "visual_stimuli": data_to_visual_stimuli(data),
    }


def expand_dict(out_dict, from_dict, index):
    """there is obviously a better way...

    Notes
    -----
    - TODO replace this by making every `annotate_...` function return a pandas Series
    """
    for k, v in from_dict.items():
        if not out_dict.get(k):
            out_dict[k] = {}

        out_dict.get(k)[index] = v


def data_to_licks(data):
    """Get a dataframe of licks

    Parameters
    ----------
    data:
        foraging2 style output data structure

    Returns
    -------
    pandas.DataFrame
        licks dataframe
    """
    return get_licks(data)


def data_to_metadata(data):
    """Get metadata associated with an experiment

    Parameters
    ---------
    data: Mapping
        foraging2 style output data structure

    Returns
    -------
    Mapping
        experiment metadata

    Notes
    -----
    - as of 03/15/2018 the following were not obtainable:
        - device_name
        - stage
        - lick_detect_training_mode
        - blank_screen_on_timeout
        - stimulus_distribution
        - delta_mean
        - trial_duration
    """
    start_time_datetime = data["start_time"]

    if not start_time_datetime.tzinfo:
        start_time_datetime = start_time_datetime.replace(tzinfo=tz.gettz("US/Pacific"))

    device_name = get_device_name(data)
    params = get_params(data)  # this joins both params and commandline params

    stim_tables = data["items"]["behavior"]["stimuli"]
    n_stimulus_frames = 0

    for stim_type, stim_table in stim_tables.items():
        n_stimulus_frames += sum(stim_table.get("draw_log", []))

    params["response_window"] = list(params["response_window"])  # tuple to list
    params["periodic_flash"] = list(params["periodic_flash"])  # tuple to list

    # ugly python3 compat dict key iterating...
    try:
        stimulus_category = next(iter(stim_tables))
    except StopIteration:
        stimulus_category = None

    return {
        "startdatetime": start_time_datetime.astimezone(tz.gettz("UTC")).isoformat(),
        "rig_id": None,  # not obtainable because device_name is not obtainable
        "computer_name": device_name,
        "reward_vol": get_reward_volume(data),
        "rewardvol": get_reward_volume(data),  # for compatibility with legacy code
        "auto_reward_vol": get_auto_reward_volume(data),
        "params": params,
        "mouseid": get_mouse_id(data),
        "response_window": list(get_response_window(data)),  # tuple to list
        "task": get_task_id(data),
        "stage": get_stage(data),  # not implemented currently
        "stoptime": get_session_duration(data),
        "userid": get_user_id(data),
        "lick_detect_training_mode": False,  # currently no choice
        "blankscreen_on_timeout": False,  # always false for now 041318
        "stim_duration": get_stimulus_duration(data) * 1000,  # seconds to milliseconds
        "blank_duration_range": [
            get_blank_duration_range(data)[0],
            get_blank_duration_range(data)[1],
        ],  # seconds to miliseconds
        "delta_minimum": get_pre_change_time(data),
        "stimulus_distribution": get_stimulus_distribution(data),  # not obtainable
        "delta_mean": get_delta_mean(data),  # not obtainable
        "trial_duration": None,  # not obtainable
        "n_stimulus_frames": n_stimulus_frames,
        "stimulus": stimulus_category,  # needs to be a string so we will just grab the first stimulus category we find even though there can be many
    }


def data_to_rewards(data):
    """Generate a rewards dataframe

    Parameters
    ----------
    data: Mapping
        foraging2 style output data structure

    Returns
    -------
    pandas.DataFrame
        rewards data structure
    """
    rewards_dict = {
        "frame": {},
        "time": {},
        "volume": {},
        "lickspout": {},
    }

    for idx, trial in get_trials(data).iterrows():
        rewards = trial["rewards"]  # as i write this there can only ever be one reward per trial

        if rewards:
            rewards_dict["volume"][idx] = rewards[0][0]
            rewards_dict["time"][idx] = rewards[0][1]
            rewards_dict["frame"][idx] = rewards[0][2]
            rewards_dict["lickspout"][idx] = None  # not yet implemented in the foraging2 output

    return pd.DataFrame(data=rewards_dict)


def data_to_running(data):
    """Get experiments running data

    Parameters
    ----------
    data: Mapping
        foraging2 style output data structure

    Returns
    -------
    pandas.DataFrame
        running data structure

    Notes
    -----
    - the index of each time is the frame number
    """
    speed_df = get_running_speed(data)[["speed (cm/s)", "time"]]  # yeah...it's dumb i kno...

    n_frames = len(speed_df)

    frames_df = pd.DataFrame(data={"frame": range(-1, n_frames - 1)})

    return speed_df.join(frames_df, how="outer") \
        .rename(index=str, columns={"speed (cm/s)": "speed", })


def data_to_time(data):
    """Get experiments times

    Parameters
    ----------
    data: Mapping
        foraging2 style output data structure

    Returns
    -------
    np.array
        array of times for the experiement
    """
    return get_time(data)


def data_to_trials(data):
    """Generate a trial structure that very closely mirrors the original trials
    structure output by foraging legacy code

    Parameters
    ----------
    data: Mapping
        foraging2 style output data structur

    Returns
    -------
    pandas.DataFrame
        trials data structure
    """
    stimuli = get_stimuli(data)
    trial_log = get_trial_log(data)
    pre_change_time = get_pre_change_time(data)
    initial_blank_duration = get_initial_blank_duration(data) or 0  # woohoo!
    experiment_params = {
        "publish_time": None,  # not obtainable
    }

    trials = {}

    for trial in trial_log:
        index = trial["index"]  # trial index

        expand_dict(trials, annotate_licks(trial), index)
        expand_dict(trials, annotate_rewards(trial), index)
        expand_dict(trials, annotate_optogenetics(trial), index)
        expand_dict(trials, annotate_responses(trial), index)
        expand_dict(
            trials,
            annotate_schedule_time(
                trial,
                pre_change_time,
                initial_blank_duration
            ),
            index
        )
        expand_dict(trials, annotate_stimuli(trial, stimuli), index)
        # expand_dict(trials, annotate_trials(trial), index)
        expand_dict(trials, experiment_params, index)

    trials = pd.DataFrame(data=trials)

    return trials.rename(
        columns={
            "start_time": "starttime",
            "start_frame": "startframe",
            "delta_orientation": "delta_ori",
            "auto_rewarded_trial": "auto_rewarded",
            "change_orientation": "change_ori",
            "initial_orientation": "initial_ori",
        }
    ).reset_index()


def data_to_visual_stimuli(data, time=None):

    stimuli = data['items']['behavior']['stimuli']

    if time is None:
        time = get_time(data)

    visual_stimuli = get_visual_stimuli(stimuli, time)

    return visual_stimuli
