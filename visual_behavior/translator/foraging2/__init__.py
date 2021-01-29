import pandas as pd
from six import PY3
import pickle

from ...utilities import local_time, ListHandler, DoubleColonFormatter

from .extract import get_trial_log, get_stimuli, get_pre_change_time, \
    annotate_licks, annotate_rewards, annotate_optogenetics, annotate_responses, \
    annotate_schedule_time, annotate_stimuli, get_user_id, get_mouse_id, \
    get_blank_duration_range, get_device_name, get_session_duration, \
    get_stimulus_duration, get_task_id, get_response_window, get_licks, \
    get_running_speed, get_params, get_time, get_trials, \
    get_stimulus_distribution, get_delta_mean, get_initial_blank_duration, \
    get_stage, get_reward_volume, get_auto_reward_volume, get_warm_up_trials, \
    get_stimulus_window, get_volume_limit, get_failure_repeats, \
    get_catch_frequency, get_free_reward_trials, get_min_no_lick_time, \
    get_max_session_duration, get_abort_on_early_response, get_session_id, \
    get_even_sampling, get_auto_reward_delay, get_periodic_flash, get_platform_info, \
    get_rig_id

from .extract_stimuli import get_visual_stimuli, check_for_omitted_flashes
from .extract_images import get_image_metadata
from .extract_movies import get_movie_image_epochs, get_movie_metadata
from ..foraging.extract_images import get_image_data

import logging

logger = logging.getLogger(__name__)


if PY3:
    import zipfile36 as zipfile

    def load_pickle(pstream):

        return pickle.load(pstream, encoding="bytes")
else:
    import zipfile2 as zipfile

    FileNotFoundError = IOError

    def load_pickle(pstream):

        return pickle.load(pstream)


def data_to_change_detection_core(data, time=None):
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

    if time is None:
        time = data_to_time(data)

    log_messages = []
    handler = ListHandler(log_messages)
    handler.setFormatter(
        DoubleColonFormatter
    )
    handler.setLevel(logging.INFO)

    logger.addHandler(
        handler
    )

    core_data = {
        "metadata": data_to_metadata(data),
        "time": time,
        "licks": data_to_licks(data, time=time),
        "trials": data_to_trials(data, time=time),
        "running": data_to_running(data, time=time),
        "rewards": data_to_rewards(data, time=time),
        "visual_stimuli": data_to_visual_stimuli(data, time=time),
        "omitted_stimuli": data_to_omitted_stimuli(data, time=time),
        "image_set": data_to_images(data),
    }

    core_data['log'] = log_messages

    return core_data


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


def data_to_licks(data, time=None):
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
    licks = get_licks(data)

    if time is not None:
        licks['time'] = time[licks['frame']]

    return licks


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
    start_time_datetime_local = local_time(data["start_time"], timezone='America/Los_Angeles')

    mouse_id = get_mouse_id(data)

    behavior_session_uuid = get_session_id(data, create_if_missing=True)

    params = get_params(data)  # this joins both params and commandline params

    stim_tables = data["items"]["behavior"]["stimuli"]
    n_stimulus_frames = 0

    for stim_type, stim_table in stim_tables.items():
        n_stimulus_frames += sum(stim_table.get("draw_log", []))

    # ugly python3 compat dict key iterating...
    try:
        stimulus_category = next(iter(stim_tables))
    except StopIteration:
        stimulus_category = None

    metadata = {
        "startdatetime": start_time_datetime_local,
        "rig_id": get_rig_id(data),
        "computer_name": get_device_name(data),
        "reward_vol": get_reward_volume(data),
        "rewardvol": get_reward_volume(data),  # for compatibility with legacy code
        "auto_reward_vol": get_auto_reward_volume(data),
        "params": params,
        "mouseid": mouse_id,
        "response_window": list(get_response_window(data)),  # tuple to list
        "task": get_task_id(data),
        "stage": get_stage(data),  # not implemented currently
        "stoptime": get_session_duration(data),
        "userid": get_user_id(data),
        "lick_detect_training_mode": "single",  # currently no choice
        "blankscreen_on_timeout": False,  # always false for now 041318
        "stim_duration": get_stimulus_duration(data) * 1000,  # seconds to milliseconds
        "blank_duration_range": [
            get_blank_duration_range(data)[0],
            get_blank_duration_range(data)[1],
        ],
        "delta_minimum": get_pre_change_time(data),
        "stimulus_distribution": get_stimulus_distribution(data),  # not obtainable
        "delta_mean": get_delta_mean(data),  # not obtainable
        "trial_duration": None,  # not obtainable
        "n_stimulus_frames": n_stimulus_frames,
        "stimulus": stimulus_category,  # needs to be a string so we will just grab the first stimulus category we find even though there can be many
        "warm_up_trials": get_warm_up_trials(data),
        "stimulus_window": get_stimulus_window(data),
        "volume_limit": get_volume_limit(data),
        "failure_repeats": get_failure_repeats(data),
        "catch_frequency": get_catch_frequency(data),
        "auto_reward_delay": get_auto_reward_delay(data),
        "free_reward_trials": get_free_reward_trials(data),
        "min_no_lick_time": get_min_no_lick_time(data),
        "max_session_duration": get_max_session_duration(data),
        "abort_on_early_response": get_abort_on_early_response(data),
        "initial_blank_duration": get_initial_blank_duration(data),
        "even_sampling_enabled": get_even_sampling(data),
        "behavior_session_uuid": behavior_session_uuid,
        "periodic_flash": get_periodic_flash(data),
        "platform_info": get_platform_info(data),
    }

    return metadata


def data_to_rewards(data, time=None):
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

    rewards = pd.DataFrame(data=rewards_dict)

    if time is not None:
        rewards['time'] = time[rewards['frame']]

    return rewards


def data_to_running(data, time=None):
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
    speed_df = get_running_speed(data)

    if time is not None:
        speed_df['time'] = time[speed_df['frame']]

    return speed_df


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


def data_to_trials(data, time=None):
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

    trials = {}

    for trial in trial_log:
        index = trial["index"]  # trial index

        if 'trial_params' not in trial:
            logger.error("No 'trial_params' in trial {}. Skipping. See https://github.com/AllenInstitute/visual_behavior_analysis/issues/289".format(index))
            continue

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

    trials = pd.DataFrame(data=trials)

    trials = trials.rename(
        columns={
            "start_time": "starttime",
            "start_frame": "startframe",
            "end_time": "endtime",
            "end_frame": "endframe",
            "delta_orientation": "delta_ori",
            "auto_rewarded_trial": "auto_rewarded",
            "change_orientation": "change_ori",
            "initial_orientation": "initial_ori",
        }
    ).reset_index()

    return trials


def data_to_visual_stimuli(data, time=None):
    MAYBE_A_MOVIE = [
        'countdown',
        'fingerprint',
    ]

    if time is None:
        time = get_time(data)

    static_image_epochs = get_visual_stimuli(
        data['items']['behavior']['stimuli'],
        time,
    )

    for name, stim_item in data['items']['behavior'].get('items', {}).items():
        if name.lower() in MAYBE_A_MOVIE:
            static_image_epochs.extend(
                get_movie_image_epochs(name, stim_item, time, )
            )

    return pd.DataFrame(data=static_image_epochs)


def data_to_omitted_stimuli(data, time=None):
    if time is None:
        time = get_time(data)

    if 'omitted_flash_frame_log' in data['items']['behavior'].keys():
        omitted_flash_frame_log = data['items']['behavior']['omitted_flash_frame_log']
    else:
        omitted_flash_frame_log = None

    return check_for_omitted_flashes(data_to_visual_stimuli(data, time=time), time, omitted_flash_frame_log, get_periodic_flash(data))


def data_to_images(data):
    MAYBE_A_MOVIE = [
        'countdown',
        'fingerprint',
    ]
    if 'images' in data["items"]["behavior"]["stimuli"]:

        # Sometimes the source is a zipped pickle:
        metadata = get_image_metadata(data)
        try:
            image_set = load_pickle(open(metadata['image_set'], 'rb'))
            images, images_meta = get_image_data(image_set)
            image_table = dict(
                metadata=metadata,
                images=images,
                image_attributes=images_meta,
            )
        except (AttributeError, UnicodeDecodeError, pickle.UnpicklingError):
            zfile = zipfile.ZipFile(metadata['image_set'])
            finfo = zfile.infolist()[0]
            ifile = zfile.open(finfo)
            image_set = load_pickle(ifile)
            images, images_meta = get_image_data(image_set)
            image_table = dict(
                metadata=metadata,
                images=images,
                image_attributes=images_meta,
            )
        except FileNotFoundError:
            logger.critical('Image file not found: {0}'.format(metadata['image_set']))
            image_table = dict(
                metadata={},
                images=[],
                image_attributes=[],
            )
    else:
        image_table = dict(
            metadata={},
            images=[],
            image_attributes=[],
        )

    # TODO: make this better, all we need to know is if there's at least one key...
    static_stimuli_names = [
        k
        for k in data['items']['behavior'].get('items', {}).keys()
        if k in MAYBE_A_MOVIE
    ]
    if len(static_stimuli_names) > 0:  # has static stimuli
        for name, meta in get_movie_metadata(data).items():
            image_table['metadata']['movie:%s' % name] = meta  # prefix each name with 'movie:' maybe this is good?

    return image_table
