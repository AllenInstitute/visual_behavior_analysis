import os
import warnings
import pandas as pd
import numpy as np
from six import iteritems
from dateutil import parser

from ...utilities import inplace


def _infer_last_trial_frame(trials, visual_stimuli):
    """attempts to infer the last frame index of the trials

    Parameters
    ----------
    trials : `pd.Dataframe`
        core trials dataframe
    visual_stimuli : `pd.Dataframe`
        core visual stimuli dataframe

    Returns
    -------
    int or None
        last frame index of the last trial

    Notes
    -----
    - this is meant to address issue #482 with `make_trials_contiguous` which
    assumed that the last experiment frame would be a trial frame but breaks for
    experiments where the last frame is a movie frame or grayscreen
    """
    last_trial_end_frame = (trials.iloc[-1]).endframe
    filtered_vs = visual_stimuli[visual_stimuli.frame >= last_trial_end_frame]

    if len(filtered_vs.index) > 0:
        return (filtered_vs.iloc[0]).frame


@inplace
def make_trials_contiguous(trials, time, visual_stimuli=None):
    if visual_stimuli is not None:
        trial_end_frame = _infer_last_trial_frame(trials, visual_stimuli, )
    else:
        trial_end_frame = None

    if trial_end_frame is None:
        trial_end_frame = len(time) - 1

    trial_end_frame = min(trial_end_frame, len(time) - 1)  # ensure that trial end frame doesn't exceed length of time vector

    trial_end_time = time[trial_end_frame]

    trials['endframe'] = trials['startframe'] \
        .shift(-1) \
        .fillna(trial_end_frame) \
        .astype(int)
    trials['endtime'] = trials['starttime'] \
        .shift(-1) \
        .fillna(trial_end_time)
    trials['trial_length'] = trials['endtime'] - trials['starttime']


@inplace
def annotate_parameters(trials, metadata, keydict=None):
    """ annotates a dataframe with session parameters

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials (or flashes)
    data : unpickled session
    keydict : dict
        {'column': 'parameter'}
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    io.load_flashes
    """
    if keydict is None:
        return None
    else:
        for key, value in iteritems(keydict):
            try:
                trials[key] = [metadata[value]] * len(trials)
            except KeyError:
                trials[key] = None


@inplace
def explode_startdatetime(df):
    """ explodes the 'startdatetime' column into date/year/month/day/hour/dayofweek

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials (or flashes)
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    df['date'] = df['startdatetime'].dt.date
    df['year'] = df['startdatetime'].dt.year
    df['month'] = df['startdatetime'].dt.month
    df['day'] = df['startdatetime'].dt.day
    df['hour'] = df['startdatetime'].dt.hour
    df['dayofweek'] = df['startdatetime'].dt.weekday


@inplace
def annotate_n_rewards(df):
    """ computes the number of rewards from the 'reward_times' column

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    try:
        df['number_of_rewards'] = df['reward_times'].map(len)
    except KeyError:
        df['number_of_rewards'] = None


@inplace
def annotate_rig_id(trials, metadata):
    """ adds a column with rig id

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    try:
        trials['rig_id'] = metadata['rig_id']
    except KeyError:
        from visual_behavior.devices import get_rig_id
        trials['rig_id'] = get_rig_id(trials['computer_name'][0])


@inplace
def annotate_startdatetime(trials, metadata):
    """ adds a column with the session's `startdatetime`

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    data : pickled session
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    trials['startdatetime'] = parser.parse(metadata['startdatetime'])


@inplace
def assign_session_id(trials):
    """ adds a column with a unique ID for the session defined as
            a combination of the mouse ID and startdatetime

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    trials['session_id'] = trials['mouse_id'] + '_' + trials['startdatetime'].map(lambda x: x.isoformat())


@inplace
def annotate_cumulative_reward(trials, metadata):
    """ adds a column with the session's cumulative volume

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    data : pickled session
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    try:
        trials['cumulative_volume'] = trials['reward_volume'].cumsum()
    except Exception:
        trials['reward_volume'] = metadata['rewardvol'] * trials['number_of_rewards']
        trials['cumulative_volume'] = trials['reward_volume'].cumsum()


@inplace
def annotate_filename(df, filename):
    """ adds `filename` and `filepath` columns to dataframe

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    filename : full filename & path of session's pickle file
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    df['filepath'] = os.path.split(filename)[0]
    df['filename'] = os.path.split(filename)[-1]


@inplace
def fix_autorearded(df):
    """ renames `auto_rearded` columns to `auto_rewarded`

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    df.rename(columns={'auto_rearded': 'auto_rewarded'}, inplace=True)


@inplace
def annotate_change_detect(trials):
    """ adds `change` and `detect` columns to dataframe

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """

    trials['change'] = trials['trial_type'] == 'go'
    trials['detect'] = trials['response'] == 1.0


@inplace
def fix_change_time(trials):
    """ forces `None` values in the `change_time` column to numpy NaN

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    trials['change_time'] = trials['change_time'].map(lambda x: np.nan if x is None else x)


@inplace
def explode_response_window(trials):
    """ explodes the `response_window` column in lower & upper columns

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    trials['response_window_lower'] = trials['response_window'].map(lambda x: x[0])
    trials['response_window_upper'] = trials['response_window'].map(lambda x: x[1])


@inplace
def annotate_epochs(trials, epoch_length=5.0, adjust_first_trial=True):
    """ annotates the dataframe with an additional column which designates
    the "epoch" from session start

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    epoch_length : float
        length of epochs in seconds
    adjust_first_trial : boolean
        subtracts first trial time from every trial to account for delay in first trial time
        (for example, if session is preceded by 5 minutes of gray time, there would be no trials in the first 5 minute epoch.
        this adjustment would shift the first trial from 5 minutes to 0 minutes)
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """

    assert trials.index.is_unique, "input dataframe must have unique indices"

    for uuid in trials.behavior_session_uuid.unique():
        df_slice = trials.query('behavior_session_uuid == @uuid')
        indices = df_slice.index

        if adjust_first_trial:
            trials.loc[indices, 'start_time_epoch_adjusted'] = df_slice['starttime'] - df_slice['starttime'].min()
        else:
            trials[indices, 'start_time_epoch_adjusted'] = df_slice['starttime']

        epoch = (
            trials.loc[indices, 'start_time_epoch_adjusted']
            .map(lambda x: x / (60.0 * epoch_length))
            .map(np.floor)
            .map(lambda x: x * epoch_length)
            # .map(lambda x: "{:0.1f} min".format(x))
        )
        trials.loc[indices, 'epoch'] = epoch


@inplace
def annotate_lick_vigor(trials, licks, window=5):
    """ annotates the dataframe with two columns that indicate the number of
    licks and lick rate in 1/s

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    window : window relative to reward time for licks to include
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """

    def find_licks(reward_times):
        try:
            reward_time = reward_times[0]
        except IndexError:
            return []

        reward_lick_mask = (
            (licks['time'] > reward_time)
            & (licks['time'] < (reward_time + window))
        )

        tr_licks = licks[reward_lick_mask].copy()
        tr_licks['time'] -= reward_time
        return tr_licks['time'].values

    def number_of_licks(licks):
        return len(licks)

    trials['reward_licks'] = trials['reward_times'].map(find_licks)
    trials['reward_lick_count'] = trials['reward_licks'].map(lambda lks: len(lks) if lks is not None else None)
    # trials['reward_lick_rate'] = trials['reward_lick_number'].map(lambda n: n / window)

    def min_licks(lks):
        if lks is None:
            return None
        elif len(lks) == 0:
            return None
        else:
            return np.min(lks)

    trials['reward_lick_latency'] = trials['reward_licks'].map(min_licks)


@inplace
def annotate_trials(trials):
    """ performs multiple annotatations:

    - annotate_change_detect
    - fix_change_time
    - explode_response_window

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    # build arrays for change detection
    annotate_change_detect(trials, inplace=True)

    # assign a session ID to each row
    assign_session_id(trials, inplace=True)

    # calculate reaction times
    fix_change_time(trials, inplace=True)

    # unwrap the response window
    explode_response_window(trials, inplace=True)


@inplace
def update_times(trials, time):

    time_frame_map = {
        'change_time': 'change_frame',
        'starttime': 'startframe',
        'endtime': 'endframe',
        'lick_times': 'lick_frames',
        'reward_times': 'reward_frames',
    }

    def update(fr):
        try:
            if (fr is None) or (type(fr) == float and np.isnan(fr) == True):  # this should catch np.nans
                return None
            else:  # this should be for floats
                return time[int(fr)]
        except (TypeError, ValueError):  # this should catch lists
            return time[[int(f) for f in fr]]

    for time_col, frame_col in iteritems(time_frame_map):
        try:
            trials[time_col] = trials[frame_col].map(update)
        except KeyError:
            print('oops! {} does not exist'.format(frame_col))
            pass

    def make_array(val):
        try:
            len(val)
        except TypeError:
            val = [val, ]
        return val

    must_be_arrays = ('lick_times', 'reward_times')
    for col in must_be_arrays:
        trials[col] = trials[col].map(make_array)


def categorize_one_trial(tr):
    if pd.isnull(tr['change_time']):
        if (len(tr['lick_times']) > 0):
            trial_type = 'aborted'
        else:
            trial_type = 'other'
    else:
        if (tr['auto_rewarded'] is True):
            return 'autorewarded'

        elif (tr['rewarded'] is True):
            return 'go'

        elif (tr['rewarded'] == 0):
            return 'catch'

        else:
            return 'other'

    return trial_type


def categorize_trials(trials):
    '''trial types:
         'aborted' = lick before stimulus
         'autorewarded' = reward autodelivered at time of stimulus
         'go' = stimulus delivered, rewarded if lick emitted within 1 second
         'catch' = no stimulus delivered
         'other' = uncategorized (shouldn't be any of these, but this allows me to find trials that don't meet this conditions)

         returns a series with trial types
         '''
    return trials.apply(categorize_one_trial, axis=1)


def get_lick_frames(trials, licks):
    """
    returns a list of arrays of lick frames, with one entry per trial
    """
    # Note: [DRO - 1/12/18] it used to be the case that the lick sensor was polled on every frame in the stimulus code, giving
    #      a 1:1 correspondence between the frame number and the index in the 'lickData' array. However, that assumption was
    #      violated when we began displaying a frame at the beginning of the session without a corresponding call the 'checkLickSensor'
    #      method. Using the 'responselog' instead will provide a more accurate measure of actual like frames and times.
    # lick_frames = data['lickData'][0]

    lick_frames = licks['frame']

    local_licks = []
    for idx, row in trials.iterrows():
        local_licks.append(
            lick_frames[np.logical_and(lick_frames > int(row['startframe']), lick_frames <= int(row['endframe']))]
        )

    return local_licks


# HACK MJD
def get_lick_bout_times(trials, licks):
    
    # TODO ASSERT has annotation bouts
    #bout_times = licks.query("bout_start == True").time.to_numpy()
    bout_times = licks.query("bout_start == True").time.reset_index(drop=True)

    trial_bouts = []
    for idx, row in trials.iterrows():
        bouts = bout_times[np.logical_and(bout_times > row['starttime'],
                   bout_times <= row['endtime'])].dropna()

        trial_bouts.append(list(bouts.values))

    return trial_bouts

@inplace
def calculate_latency(trials):
    # For each trial, calculate the difference between each stimulus change and the next lick (response lick)
    for idx in trials.index:
        if pd.isnull(trials['change_time'][idx]) is False and len(trials['lick_times'][idx]) > 0:
            licks = np.array(trials['lick_times'][idx])

            post_stimulus_licks = licks - trials['change_time'][idx]

            post_window_licks = post_stimulus_licks[post_stimulus_licks > trials.loc[idx]['response_window'][0]]

            if len(post_window_licks) > 0:
                trials.loc[idx, 'response_latency'] = post_window_licks[0]


@inplace
def calculate_reward_rate(
        df,
        trial_window=25,
        remove_aborted=False
):
    # written by Dan Denman (stolen from http://stash.corp.alleninstitute.org/users/danield/repos/djd/browse/calculate_reward_rate.py)
    # add a column called reward_rate to the input dataframe
    # the reward_rate column contains a rolling average of rewards/min
    # remove_aborted flag needs work, don't use it for now
    reward_rate = np.zeros(np.shape(df.change_time))
    c = 0
    for startdatetime in df.startdatetime.unique():  # go through the dataframe by each behavior session
        # get a dataframe for just this session
        if remove_aborted is True:
            warnings.warn("Don't use remove_aborted yet. Code needs work")
            df_temp = df[(df.startdatetime == startdatetime) & (df.trial_type != 'aborted')].reset_index()
        else:
            df_temp = df[df.startdatetime == startdatetime]
        trial_number = 0
        for trial in range(len(df_temp)):
            if trial_number < 10:  # if in first 10 trials of experiment
                reward_rate_on_this_lap = np.inf  # make the reward rate infinite, so that you include the first trials automatically.
            else:
                # ensure that we don't run off the ends of our dataframe # get the correct response rate around the trial
                min_index = np.max((0, trial - trial_window))
                max_index = np.min((trial + trial_window, len(df_temp)))
                df_roll = df_temp.iloc[min_index:max_index]
                correct = len(df_roll[df_roll.response_type == 'HIT'])
                time_elapsed = df_roll.starttime.iloc[-1] - df_roll.starttime.iloc[0]  # get the time elapsed over the trials
                reward_rate_on_this_lap = correct / time_elapsed  # calculate the reward rate

            reward_rate[c] = reward_rate_on_this_lap  # store the rolling average
            c += 1
            trial_number += 1  # increment some dumb counters
    df['reward_rate'] = reward_rate * 60.  # convert to rewards/min


def check_responses(trials, reward_window=None):
    '''trial types:
         'aborted' = lick before stimulus
         'autorewarded' = reward autodelivered at time of stimulus
         'go' = stimulus delivered, rewarded if lick emitted within 1 second
         'catch' = no stimulus delivered
         'other' = uncategorized (shouldn't be any of these, but this allows me to find trials that don't meet this conditions)

         adds a column called 'response' to the input dataframe
         '''

    if reward_window is not None:
        rw_low = reward_window[0]
        rw_high = reward_window[1]

    did_respond = np.zeros(len(trials))
    for ii, idx in enumerate(trials.index):

        if reward_window is None:
            rw_low = trials.loc[idx]['response_window'][0]
            rw_high = trials.loc[idx]['response_window'][1]
        if pd.isnull(trials.loc[idx]['change_time']) == False and \
                pd.isnull(trials.loc[idx]['response_latency']) == False and \
                trials.loc[idx]['response_latency'] >= rw_low and \
                trials.loc[idx]['response_latency'] <= rw_high:
            did_respond[ii] = True

    return did_respond


def trial_translator(trial_type, response_type, auto_rewarded=False):
    if trial_type == 'aborted':
        return 'aborted'
    elif auto_rewarded == True:
        return 'auto_rewarded'
    elif trial_type == 'autorewarded':
        return 'auto_rewarded'
    elif trial_type == 'go':
        if response_type in ['HIT', True, 1]:
            return 'hit'
        else:
            return 'miss'
    elif trial_type == 'catch':
        if response_type in ['FA', True, 1]:
            return 'false_alarm'
        else:
            return 'correct_reject'


def is_hit(trial):
    '''returns:
        True if trial is a hit
        False if trial is a miss
        NaN otherwise
    '''
    trial_description = assign_trial_description(trial)
    if trial_description == 'hit':
        return True
    elif trial_description == 'miss':
        return False
    else:
        return np.nan


def is_catch(trial):
    '''returns:
        True if trial is a false alarm
        False if trial is a correct rejection
        NaN otherwise
    '''
    trial_description = assign_trial_description(trial)
    if trial_description == 'false_alarm':
        return True
    elif trial_description == 'correct_reject':
        return False
    else:
        return np.nan


def colormap(trial_type, palette='trial_types'):
    if palette.lower() == 'trial_types':
        colors = {
            'aborted': 'lightgray',
            'auto_rewarded': 'darkblue',
            'hit': '#55a868',
            'miss': '#ccb974',
            'false_alarm': '#c44e52',
            'correct_reject': '#4c72b0',
        }
    else:
        # colors = {
        #     'aborted': 'red',
        #     'auto_rewarded': 'blue',
        #     'hit': 'darkgreen',
        #     'miss': 'lightgreen',
        #     'false_alarm': 'darkorange',
        #     'correct_reject': 'yellow',
        # }
        colors = {
            'aborted': 'red',
            'auto_rewarded': 'blue',
            'hit': 'green',
            'miss': 'white',
            'false_alarm': 'orange',
            'correct_reject': 'yellow',
        }
    return colors.get(trial_type, 'white')


def assign_trial_description(trial, palette='trial_types'):
    return trial_translator(trial['trial_type'], trial['response'], trial['auto_rewarded'])


def assign_color(trial, palette='trial_types'):

    trial_type = trial_translator(trial['trial_type'], trial['response'])
    return colormap(trial_type, palette)


def get_response_type(trials):

    response_type = []
    for idx in trials.index:
        if trials.loc[idx].trial_type.lower() == 'aborted':
            response_type.append('EARLY_RESPONSE')
        elif (trials.loc[idx].rewarded == True) & (trials.loc[idx].response == 1):
            response_type.append('HIT')
        elif (trials.loc[idx].rewarded == True) & (trials.loc[idx].response != 1):
            response_type.append('MISS')
        elif (trials.loc[idx].rewarded == False) & (trials.loc[idx].response == 1):
            response_type.append('FA')
        elif (trials.loc[idx].rewarded == False) & (trials.loc[idx].response != 1):
            response_type.append('CR')
        else:
            response_type.append('other')

    return response_type


@inplace
def remove_repeated_licks(trials):
    """
    the stimulus code records one lick for each frame in which the tongue was in contact with the spout
    if a single lick spans multiple frames, it'll be recorded as multiple licks
    this method will throw out the lick times and lick frames after the initial contact
    """
    lt = []
    lf = []
    for idx, row in trials.iterrows():

        # get licks for this frame
        lick_frames_on_this_trial = row.lick_frames
        lick_times_on_this_trial = row.lick_times
        if len(lick_frames_on_this_trial) > 0:
            # use the number of frames between each lick to determine which to keep
            if len(lick_frames_on_this_trial) > 1:
                lick_intervals = np.hstack((np.inf, np.diff(lick_frames_on_this_trial)))
            else:
                lick_intervals = np.array([np.inf])

            # only keep licks that are preceded by at least one frame without a lick
            lf.append(list(np.array(lick_frames_on_this_trial)[lick_intervals > 1]))
            lt.append(list(np.array(lick_times_on_this_trial)[lick_intervals[:len(lick_times_on_this_trial)] > 1]))
        else:
            lt.append([])
            lf.append([])

    # replace the appropriate rows of the dataframe
    trials['lick_times'] = lt
    trials['lick_frames'] = lf
