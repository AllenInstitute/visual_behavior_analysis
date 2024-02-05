import warnings
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter as medfilt
from .extract import get_end_time
from .extract_images import get_image_data, get_image_metadata
from ...utilities import calc_deriv, deg_to_dist, local_time, ListHandler, DoubleColonFormatter
from ...uuid_utils import make_deterministic_session_uuid

import logging

logger = logging.getLogger(__name__)

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

    log_messages = []
    handler = ListHandler(log_messages)
    handler.setFormatter(
        DoubleColonFormatter
    )
    handler.setLevel(logging.INFO)
    logger.addHandler(
        handler
    )

    if time is None:
        time = load_time(data)

    images = load_images(data)

    core_data = {
        "time": time,
        "metadata": load_metadata(data),
        "licks": load_licks(data, time=time),
        "trials": load_trials(data, time=time),
        "running": load_running_speed(data, time=time),
        "rewards": load_rewards(data, time=time),
        "visual_stimuli": load_visual_stimuli(data, time=time),
        "omitted_stimuli": load_omitted_stimuli(data, time=time),
        "image_set": images,
    }

    core_data['log'] = log_messages
    return core_data


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

    metadata['startdatetime'] = local_time(
        metadata["startdatetime"],
        timezone='America/Los_Angeles',
    )

    logger.warning('`session_uuid` not found. generating a deterministic UUID')
    metadata['behavior_session_uuid'] = make_deterministic_session_uuid(
        metadata['mouseid'],
        metadata['startdatetime'],
    )

    metadata['auto_reward_vol'] = 0.05  # hard coded
    metadata['max_session_duration'] = 60.0  # hard coded
    metadata['min_no_lick_time'] = data['minimum_no_lick_time']
    metadata['abort_on_early_response'] = data['ignore_false_alarms'] == False
    metadata['even_sampling_enabled'] = data['image_category_sampling_mode'] == 'even_sampling'
    metadata['failure_repeats'] = data['max_number_trial_repeats']
    metadata['catch_frequency'] = data['catch_frequency']
    metadata['volume_limit'] = data['volumelimit']
    metadata['initial_blank_duration'] = data['initial_blank']
    metadata['warm_up_trials'] = data['warmup_trials']
    metadata['stimulus_window'] = data['trial_duration'] - data['delta_minimum']
    metadata['auto_reward_delay'] = 0.0  # hard coded
    metadata['platform_info'] = {}

    blank_dur = np.mean(data['blank_duration_range'])
    metadata['periodic_flash'] = (data['stim_duration'], blank_dur) if blank_dur > 0 else None

    block_length = data['lick_detect_training_block_length']

    try:
        metadata['free_reward_trials'] = block_length[0]
    except TypeError:
        metadata['free_reward_trials'] = block_length

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

    if time is None:
        time = load_time(data)

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
        'endframe'
    )

    forced_types = {
        'initial_ori': float,
        'change_ori': float,
        'delta_ori': float,
        'initial_contrast': float,
        'change_contrast': float,
        'optogenetics': bool,
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

    trials['change_frame'] = trials['change_frame'].map(lambda x: int(x) if np.isfinite(x) else None) + 1

    trials["change_image_category"] = trials["change_image_category"].apply(lambda x: x if x else '')  # use empty string instead of NoneType
    trials["change_image_name"] = trials["change_image_name"].apply(lambda x: x if x else '')  # use empty string instead of NoneType
    trials["initial_image_category"] = trials["initial_image_category"].apply(lambda x: x if x else '')  # use empty string instead of NoneType
    trials["initial_image_name"] = trials["initial_image_name"].apply(lambda x: x if x else '')  # use empty string instead of NoneType

    # make scheduled_change_time relative
    trials["scheduled_change_time"] = trials["scheduled_change_time"] - trials['starttime']

    # add endframe column as startframe of last frame. Last endframe is last frame of session
    trials['endframe'] = trials['startframe'].shift(periods=-1)
    trials.at[trials.index[-1], 'endframe'] = len(time) - 1

    trials['endtime'] = get_end_time(trials, time)
    trials['trial_length'] = trials['endtime'] - trials['starttime']

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

    dx_raw = np.array(data['dx'])
    dx = medfilt(dx_raw, size=5)  # remove big, single frame spikes in encoder values
    dx = np.cumsum(dx)  # wheel rotations

    time = time[:len(dx)]

    speed = calc_deriv(dx, time)  # speed is in deg/s
    speed = deg_to_dist(speed)  # converts speed to cm/s

    # remove first values which is always high
    speed = speed[1:]
    time = time[1:]
    
    if smooth:
        # running_speed_cm_per_sec = pd.rolling_mean(running_speed_cm_per_sec, window=6)
        raise NotImplementedError

    # accel = calc_deriv(speed, time)
    # jerk = calc_deriv(accel, time)

    running_speed = pd.DataFrame({
        'time': time,
        'frame': range(len(time)),
        'speed': speed,
        'dx': dx_raw,
        'v_sig': data['vsig'],
        'v_in': data['vin'],
        # 'acceleration (cm/s^2)': accel,
        # 'jerk (cm/s^3)': jerk,
    })
    return running_speed


class EndOfStateFinder(object):

    def __init__(self):
        self.end_frame = 0

    def check(self, state):
        if state is True:
            self.end_frame += 1
        else:
            self.end_frame = 0
        return self.end_frame


def find_ends(stimdf):
    # first, grab a copy of the reversed dataframe
    stimdf = stimdf.sort_values('frame', ascending=False)

    stimdf['frames_to_end'] = stimdf['state'].map(EndOfStateFinder().check)

    stimdf = stimdf.sort_values('frame', ascending=True)
    return stimdf


def load_visual_stimuli(data, time=None):
    """ Returns the stimulus flashes in an experiment.

    NOTE: Currently only works for images & gratings.

    Parameters
    ----------
    data : dict, unpickled experiment (or path to pickled object))
    time : np.array, optional
        array of times for each stimulus frame

    Returns
    -------
    flashes : pandas DataFrame

    See Also
    --------
    load_trials : loads trials

    """
    if time is None:
        print('`time` not passed. using vsync from pkl file')
        time = load_time(data)

    stimdf = pd.DataFrame(data['stimuluslog'])

    stimdf = find_ends(stimdf)
    stimdf['end_frame'] = stimdf['frames_to_end'] + stimdf['frame']

    stimdf['frame'] += 1
    stimdf['end_frame'] += 1

    def find_time(fr):
        try:
            return time[fr]
        except IndexError:
            return np.nan

    stimdf['end_time'] = stimdf['end_frame'].map(find_time)
    stimdf['duration'] = stimdf['end_time'] - stimdf['time']

    onset_mask = (
        stimdf['state']
        .astype(int)
        .diff()
        .fillna(1.0)
    ) > 0

    if pd.isnull(stimdf['image_name']).any() == False:  # 'image_category' in stimdf.columns:
        cols = ['frame', 'end_frame', 'time', 'duration', 'image_category', 'image_name']
        stimuli = stimdf[onset_mask][cols]
        stimuli['orientation'] = None
        stimuli['contrast'] = None

    elif 'ori' in stimdf.columns:
        cols = ['frame', 'end_frame', 'time', 'duration', 'ori', 'contrast']
        stimuli = stimdf[onset_mask][cols]
        stimuli.rename(columns={'ori': 'orientation'}, inplace=True)
        stimuli['image_category'] = None
        stimuli['image_name'] = None

    return stimuli


def load_omitted_stimuli(data, time=None):

    if time is None:
        print('`time` not passed. using vsync from pkl file')
        time = load_time(data)

    if 'omitted_flash_frame_log' in data.keys():
        omitted_flash_list = []
        for omitted_flash_frame in data['omitted_flash_frame_log']:

            omitted_flash_list.append({
                'frame': omitted_flash_frame,
                'time': time[omitted_flash_frame],
            })
        return pd.DataFrame(omitted_flash_list)
    else:
        return pd.DataFrame(columns=['frame', 'time'])


def load_images(data):

    if 'image_dict' in data.keys():
        image_dict = data['image_dict']

        images, images_meta = get_image_data(image_dict)

        images = dict(
            metadata=get_image_metadata(data),
            images=images,
            image_attributes=images_meta,
        )
    else:
        images = dict(
            metadata={},
            images=[],
            image_attributes=[],
        )

    return images
