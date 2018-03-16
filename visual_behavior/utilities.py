from __future__ import print_function
import numpy as np
import pandas as pd
<<<<<<< HEAD
from fnmatch import fnmatch
from email.mime.text import MIMEText
from scipy.interpolate import interp1d

from .devices import get_rig_id
from .translator.foraging.extract import load_licks as get_licks, \
    load_time as get_time


"""DEPRECIATED DO NOT WASTE TIME HERE...
"""


# -> io.py
def create_doc_dataframe(data, time=None, legacy=False):
    """ creates a trials dataframe from a detection-of-change session

    Parameters
    ----------
    data: Mapping
        experiment data

    Returns
    -------
    trials : pandas DataFrame
    """
    if legacy:
        if time is None:
            time = get_time_legacy(data)
        get_licks = get_licks_legacy
        df = get_trials_legacy(data, time=time)
    else:
        if time is None:
            time = get_time(data)
        df = get_trials(data)

    df = df[~pd.isnull(df['reward_times'])].reset_index(drop=True)

    # add some columns to the dataframe
    keydict = {
        'mouse_id': 'mouseid',
        'response_window': 'response_window',
        'task': 'task',
        'stage': 'stage',
        'session_duration': 'stoptime',
        'user_id': 'userid',
        'LDT_mode': 'lick_detect_training_mode',
        'blank_screen_timeout': 'blankscreen_on_timeout',
        'stim_duration': 'stim_duration',
        'blank_duration_range': 'blank_duration_range',
        'prechange_minimum': 'delta_minimum',
        'stimulus_distribution': 'stimulus_distribution',
        'stimulus': 'stimulus',
        'distribution_mean': 'delta_mean',
        'trial_duration': 'trial_duration',
        'computer_name': 'computer_name',
    }

    annotate_parameters(df, data, keydict=keydict, inplace=True)
    annotate_startdatetime(df, data, inplace=True)
    explode_startdatetime(df, inplace=True)
    annotate_n_rewards(df, inplace=True)
    annotate_rig_id(df, data, inplace=True)
    fix_autorearded(df, inplace=True)
    annotate_cumulative_reward(df, data, inplace=True)
    annotate_filename(df, filename, inplace=True)

    for col in ('auto_rewarded', 'change_time', ):
        if col not in df.columns:
            df[col] = None

    # add some columns that require calculation

    df['trial_type'] = categorize_trials(df)
    df['endframe'] = get_end_frame(df, data)
    df['lick_frames'] = get_lick_frames(df, data)
    # df['last_lick'] = get_last_licktimes(df,data)

    update_times(df, data, time=time, inplace=True)

    annotate_lick_vigor(df, data, inplace=True)
    calculate_latency(df)
    calculate_reward_rate(df)

    df['response'] = check_responses(df)
    df['trial_length'] = calculate_trial_length(df)
    df['endtime'] = get_end_time(df, data, time=time)
    df['color'] = assign_color(df)
    df['response_type'] = get_response_type(df)

    try:
        remove_repeated_licks(df)
    except Exception as e:
        print('FAILED TO REMOVE REPEATED LICKS')
        print(e)

    return df


def load_behavior_data(mice, progressbar=True, save_dataframe=True, load_existing_dataframe=True):
    """ Loads DoC behavior dataframe for all mice in a list

    Parameters
    ----------
    mice : list of strings
        list of mice to extract data for

    progressbar : boolean (optional argument, default True)
        optionally displays progress bar during loading process
        (can be slow, so it's good to track progress. Maybe this is a good time to go get a coffee?)

    save_dataframe : boolean (optional argument, default True)
        will save a cached dataframe for each mouse as it loads. This will make the next load faster

    Returns
    -------
    trials : pandas DataFrame for all mice

    """

    if type(mice) == 'str':
        mice = [mice]
    df = pd.DataFrame()
    basepath = '//allen/programs/braintv/workgroups/neuralcoding/Behavior/Data'
    unloaded = []
    if progressbar is True:
        pb = Progress_Bar_Text(len(mice))
    for mouse in mice:
        dft = load_from_folder(
            os.path.join(basepath, mouse, 'output'),
            load_existing_dataframe=load_existing_dataframe,
            save_dataframe=save_dataframe
        )
        cohort = get_cohort_info(mouse)
        dft['cohort'] = cohort
        unloaded.append(dft)
        if progressbar is True:
            pb.update(message="{}, C{}".format(mouse, cohort))
    df = pd.concat([df, ] + unloaded, ignore_index=True)

    return df


def get_cohort_info(mouse_id):
    """ uses cohort spreadsheet to get cohort identity for a mouse
        (does mouse_info do this?)

    Parameters
    ----------
    mouse_id : string
        mouse of interest

    Returns
    -------
    cohort : integer, or None if not found

    """
    cohorts_path = "//allen/programs/braintv/workgroups/neuralcoding/Behavior/Data/VisualBehaviorDevelopment_CohortIDs.xlsx"
    cohorts = pd.read_excel(cohorts_path)
    try:
        cohort = int(cohorts[cohorts.mouse == mouse_id].cohort.values[0])
    except Exception:
        cohort = None

    return cohort


# -> DEPRECATE
def get_mouse_info(mouse_id):
    '''
    Gets data from the info.txt file in each mouse's folder on aibsdata

    .. note:: Deprecated
              `utilities.get_mouse_info` will be removed in a future release,
              it is replaced by `cohorts.mouse_info`

    '''
    import warnings
    warnings.warn(
        "this function is deprecated. please use the `mouse_info` package",
        DeprecationWarning
    )
    from visual_behavior.cohorts import mouse_info
    return mouse_info(mouse_id)


# -> analyze.py
def get_lick_frames(df_in, data, time=None):
    """
    returns a list of arrays of lick frames, with one entry per trial
    """
    # Note: [DRO - 1/12/18] it used to be the case that the lick sensor was polled on every frame in the stimulus code, giving
    #      a 1:1 correspondence between the frame number and the index in the 'lickData' array. However, that assumption was
    #      violated when we began displaying a frame at the beginning of the session without a corresponding call the 'checkLickSensor'
    #      method. Using the 'responselog' instead will provide a more accurate measure of actual like frames and times.
    # lick_frames = data['lickData'][0]

    lick_frames = get_licks(data, time=time)['frame']

    local_licks = []
    for idx, row in df_in.iterrows():
        local_licks.append(
            lick_frames[np.logical_and(lick_frames >= int(row['startframe']), lick_frames <= int(row['endframe']))]
        )

    return local_licks


# -> analyze.py
def get_last_licktimes(df_in, data):
    '''
    get time of most recent lick before every change
    '''
    time_arr = np.hstack((0, np.cumsum(data['vsyncintervals']) / 1000.))

    # licks = time_arr[data['lickData'][0]]
    responsedf = pd.DataFrame(data['responselog'])
    licks = responsedf.time.values

    # get times of all changes in this dataset
    change_frames = df_in.change_frame.values

    change_times = np.zeros(len(change_frames)) * np.nan
    last_lick = np.zeros(len(change_frames)) * np.nan
    for ii, frame in enumerate(change_frames):
        if np.isnan(frame) is False:
            change_times[ii] = time_arr[frame.astype(int)]
            # when was the last lick?
            a = licks - change_times[ii]
            try:
                last_lick[ii] = np.max(a[a < 0])
            except Exception:
                pass

    return last_lick


# -> analyze.py
def remove_repeated_licks(df_in):
    """
    the stimulus code records one lick for each frame in which the tongue was in contact with the spout
    if a single lick spans multiple frames, it'll be recorded as multiple licks
    this method will throw out the lick times and lick frames after the initial contact
    """
    lt = []
    lf = []
    for idx, row in df_in.iterrows():

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
    df_in['lick_times'] = lt
    df_in['lick_frames'] = lf


# -> io.py
def load_from_folder(
        foldername,
        load_existing_dataframe=True,
        save_dataframe=True,
        filename_contains='*'
):
    '''
    Loads all PKL files in a given directory
    if load_existing_dataframe is True, will attempt to load previously saved dataframe in same folder to save time
    if save_dataframe is True, will save dataframe in folder to save loading time on next run
    '''

    if load_existing_dataframe is True:
        try:
            df = pd.read_pickle(os.path.join(foldername, 'summary_df.pkl'))
            previously_loaded_filenames = df.filename.unique()
        except Exception:
            df = pd.DataFrame()
            previously_loaded_filenames = []
    else:
        df = pd.DataFrame()
        previously_loaded_filenames = []

    unloaded = []
    for ii, filename in enumerate(os.listdir(foldername)):
        if filename not in previously_loaded_filenames and ".pkl" in filename and 'summary' not in filename and fnmatch(filename, '*' + filename_contains + '*'):
            try:
                dft = create_doc_dataframe(os.path.join(foldername, filename))
                unloaded.append(dft)
            except Exception as e:
                print("error loading file {}: {}".format(filename, e))
                continue
    df = pd.concat([df, ] + unloaded, ignore_index=True)
    # count unique days of training for each session
    try:
        df['training_day'] = get_training_day(df)
    except Exception as e:
        warnings.warn("Couldn't calculate training day")

    if save_dataframe is True:
        df.to_pickle(os.path.join(foldername, 'summary_df.pkl'))

    return df


# -> devices.py
def return_reward_volumes(cluster_id):
    '''
    Prints a report to the console window telling the user how much water the animals
    in cluster_id have:
        a) received so far, if the session is still running
        b) received in the last session, if no session is currently running
    '''
    from zro import Proxy
    for rig in range(1, 7):
        computer_name = get_rig_id('{}{}'.format(cluster_id, rig), input_type='rig_id')
        print('Rig {}{}'.format(cluster_id, rig))
        try:
            agent = Proxy('{}:5000'.format(computer_name), timeout=2)
            # print agent.status
            if agent.status['running'] is True:
                print("mouse {} is currently running in this rig".format(agent.status['mouse_id']))
                session = Proxy('{}:12000'.format(computer_name), timeout=2)
                N, V = len(np.array(session.rewards)[:, 0]), np.sum(np.array(session.rewards)[:, 3])
            elif agent.status['running'] is False:
                print("getting data for last session from mouse: {}".format(agent.status['mouse_id']))
                N, V = get_reward_volume_last_session(agent.status['mouse_id'])
            print("number of rewards = {}".format(N))
            print("total volume = {} mL".format(V))
        except Exception as e:
            print("failed to get data")
            print(e)
        print("")


def get_reward_volume_last_session(mouse_id):
    '''
    returns number of rewards and total volume for last session for a given mouse
    '''
    fn = get_datafile(mouse_id)
    data = pd.read_pickle(fn)
    return len(data['rewards'][:, 0]), np.sum(data['rewards'][:, 3])


def get_datafile(
        mouse_id,
        year=None,
        month=None,
        day=None,
        return_longest=True,
        location=None
):
    '''
    returns path to filenames for a given mouse and date
    year should be four digits

    if return_longest is True (default):
        returns only the longest file for the given day (to avoid short sessions that were aborted early)
    if return_longest is False:
        returns a list of all filenames for the given day
    '''
    if location is None and sys.platform == 'darwin':
        location = os.path.join('/Volumes/neuralcoding/behavior/data', mouse_id, 'output')
    elif location is None and sys.platform != 'darwin':
        location = os.path.join('//allen/programs/braintv/workgroups/neuralcoding/behavior/data', mouse_id, 'output')

    if year is None or day is None or month is None:
        # if any of the date arguments are none, return the newest
        fnames = glob.glob('{}/*.pkl'.format(location))
        fnames_lim = [
            fn for fn in fnames
            if not os.path.basename(fn).endswith('df.pkl')
        ]
        newest = max(fnames_lim, key=os.path.getctime)
        return newest
    else:
        matches = []
        filesizes = []
        for filename in os.listdir(location):
            try:
                timestamp = pd.to_datetime(
                    filename.split('-')[0],
                    format='%y%m%d%H%M%S'
                )
                if timestamp.year == int(year) \
                        and timestamp.month == int(month) \
                        and timestamp.day == int(day):
                    # check to see if mouseID is in filename:
                    if 'mouse=' + mouse_id in filename or 'mouse=' not in filename:
                        matches.append(filename)
                        filesizes.append(os.path.getsize(location))

            except Exception:
                pass

        if len(matches) > 1 and return_longest is False:
            return [os.path.join(location, fn) for fn in matches]
        elif len(matches) > 1 and return_longest is True:
            longest_file_index = filesizes.index(max(filesizes))
            return os.path.join(location, matches[longest_file_index])
        elif len(matches) == 1:
            return os.path.join(location, matches[0])
        else:
            return None


# -> plotting.py
def save_figure(
        fig,
        fname,
        formats=['.png'],
        transparent=False,
        dpi=300,
        **kwargs
):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])
    else:
        fig.set_size_inches(11, 8.5)
    for f in formats:
        fig.savefig(
            fname + f,
            transparent=transparent,
            orientation='landscape',
            dpi=dpi
        )


# -> analyze
def calculate_reward_rate(
        df,
        window=1.0,
        trial_window=25,
        remove_aborted=False
):
    # written by Dan Denman (stolen from http://stash.corp.alleninstitute.org/users/danield/repos/djd/browse/calculate_reward_rate.py)
    # add a column called reward_rate to the input dataframe
    # the reward_rate column contains a rolling average of rewards/min
    # window sets the window in which a response is considered correct, so a window of 1.0 means licks before 1.0 second are considered correct
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

                correct = len(df_roll[df_roll.response_latency < window])  # get a rolling number of correct trials
                time_elapsed = df_roll.starttime.iloc[-1] - df_roll.starttime.iloc[0]  # get the time elapsed over the trials
                reward_rate_on_this_lap = correct / time_elapsed  # calculate the reward rate

            reward_rate[c] = reward_rate_on_this_lap  # store the rolling average
            c += 1
            trial_number += 1  # increment some dumb counters
    df['reward_rate'] = reward_rate * 60.  # convert to rewards/min


def flatten_array(in_array):
    out_array = np.array([])
    for entry in in_array:
        if len(entry) > 0:
            out_array = np.hstack((out_array, entry))
    return out_array
=======
from scipy.stats import norm
>>>>>>> upstream/foraging-2


def flatten_list(in_list):
    out_list = []
    for i in range(len(in_list)):
        # check to see if each entry is a list or array
        if isinstance(in_list[i], list) or isinstance(in_list[i], np.ndarray):
            # if so, iterate over each value and append to out_list
            for entry in in_list[i]:
                out_list.append(entry)
        else:
            # otherwise, append the value itself
            out_list.append(in_list[i])

    return out_list


<<<<<<< HEAD
# -> analyze
def calculate_latency(df_in):
    # For each trial, calculate the difference between each stimulus change and the next lick (response lick)
    for idx in df_in.index:
        if pd.isnull(df_in['change_time'][idx]) is False and len(df_in['lick_times'][idx]) > 0:
            licks = np.array(df_in['lick_times'][idx])

            post_stimulus_licks = licks - df_in['change_time'][idx]

            post_window_licks = post_stimulus_licks[post_stimulus_licks > df_in.loc[idx]['response_window'][0]]

            if len(post_window_licks) > 0:
                df_in.loc[idx, 'response_latency'] = post_window_licks[0]

    return df_in


# -> analyze
def calculate_trial_length(df_in):
    trial_length = np.zeros(len(df_in))
    for ii, idx in enumerate(df_in.index):
        try:
            tl = df_in.loc[idx + 1].starttime - df_in.loc[idx].starttime
            if tl < 0 or tl > 1000:
                tl = np.nan
            trial_length[ii] = tl
        except Exception:
            pass

    return trial_length


def get_end_time(df_in, data, time=None):
    '''creates a vector of end times for each trial, which is just the start time for the next trial'''
    if time is None:
        time = get_time(data)

    end_times = df_in['endframe'].map(lambda fr: time[int(fr)])
    return end_times


# -> analyze
def get_end_frame(df_in, data):

    last_frame = len(data['vsyncintervals'])

    end_frames = np.zeros_like(df_in.index) * np.nan

    for ii, index in enumerate(df_in.index[:-1]):
        end_frames[ii] = int(df_in.loc[index + 1].startframe - 1)
    if last_frame is not None:
        end_frames[-1] = int(last_frame)

    return end_frames.astype(np.int32)


# -> analyze
def categorize_one_trial(row):
    if pd.isnull(row['change_time']):
        if (len(row['lick_times']) > 0):
            trial_type = 'aborted'
        else:
            trial_type = 'other'
    else:
        if (row['rewarded'] is True):
            return 'go'

        elif (row['rewarded'] == 0):
            return 'catch'

        elif (row['auto_rewarded'] is True):
            return 'autorewarded'

        else:
            return 'other'

    return trial_type


def categorize_trials(df_in):
    '''trial types:
         'aborted' = lick before stimulus
         'autorewarded' = reward autodelivered at time of stimulus
         'go' = stimulus delivered, rewarded if lick emitted within 1 second
         'catch' = no stimulus delivered
         'other' = uncategorized (shouldn't be any of these, but this allows me to find trials that don't meet this conditions)

         adds a column called 'trial_type' to the input dataframe
         '''
    return df_in.apply(categorize_one_trial, axis=1)


# -> analyze
def get_training_day(df_in):
    '''adds a column to the dataframe with the number of unique training days up to that point
    '''

    training_day_lookup = {}
    for key, group in df_in.groupby(['mouse_id', ]):
        dates = np.sort(group['date'].unique())
        training_day_lookup[key] = {date: training_day for training_day, date in enumerate(dates)}
    return df_in.apply(lambda row: training_day_lookup[row['mouse_id']][row['date']], axis=1)


# -> analyze
def get_response_type(df_in):

    response_type = []
    for idx in df_in.index:
        if (df_in.loc[idx].rewarded == True) & (df_in.loc[idx].response == 1):
            response_type.append('HIT')
        elif (df_in.loc[idx].rewarded == True) & (df_in.loc[idx].response != 1):
            response_type.append('MISS')
        elif (df_in.loc[idx].rewarded == False) & (df_in.loc[idx].response == 1):
            response_type.append('FA')
        elif (df_in.loc[idx].rewarded == False) & (df_in.loc[idx].response != 1):
            response_type.append('CR')
        else:
            response_type.append('other')

    return response_type


# -> analyze
def assign_color(df_in, palette='default'):
    color = [None] * len(df_in)
    for idx in df_in.index:

        if df_in.loc[idx]['trial_type'] == 'aborted':
            if palette.lower() == 'marina':
                color[idx] = 'lightgray'
            else:
                color[idx] = 'red'

        elif df_in.loc[idx]['auto_rewarded'] is True:
            if palette.lower() == 'marina':
                color[idx] = 'darkblue'
            else:
                color[idx] = 'blue'

        elif df_in.loc[idx]['trial_type'] == 'go':
            if df_in.loc[idx]['response'] == 1:
                if palette.lower() == 'marina':
                    color[idx] = '#55a868'
                else:
                    color[idx] = 'darkgreen'

            elif df_in.loc[idx]['response'] != 1:
                if palette.lower() == 'marina':
                    color[idx] = '#ccb974'
                else:
                    color[idx] = 'lightgreen'

        elif df_in.loc[idx]['trial_type'] == 'catch':
            if df_in.loc[idx]['response'] == 1:
                if palette.lower() == 'marina':
                    color[idx] = '#c44e52'
                else:
                    color[idx] = 'darkorange'

            elif df_in.loc[idx]['response'] != 1:
                if palette.lower() == 'marina':
                    color[idx] = '#4c72b0'
                else:
                    color[idx] = 'yellow'

    return color


# -> analyze
def check_responses(df_in, reward_window=None):
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

    did_respond = np.zeros(len(df_in))
    for ii, idx in enumerate(df_in.index):

        if reward_window is None:
            rw_low = df_in.loc[idx]['response_window'][0]
            rw_high = df_in.loc[idx]['response_window'][1]
        if pd.isnull(df_in.loc[idx]['change_time']) == False and \
                pd.isnull(df_in.loc[idx]['response_latency']) == False and \
                df_in.loc[idx]['response_latency'] >= rw_low and \
                df_in.loc[idx]['response_latency'] <= rw_high:
            did_respond[ii] = True

    return did_respond


def resample(t, y, new_t):
    f = interp1d(t, y, bounds_error=False)
    return f(new_t)


# -> analyze
def get_reward_window(df_in):
    try:
        reward_window = df_in.iloc[0].response_window
    except Exception:
        reward_window = [0.15, 1]
    return reward_window


# -> analyze
def get_licktimes(df, reference="change"):
    if reference == 'start':
        licktimes = []
        for idx in df.index:
            licktimes.append(df.loc[idx].lick_times - df.loc[idx].starttime)
    elif reference == 'change':
        licktimes = []
        for idx in df.index:
            licktimes.append(df.loc[idx].lick_times - df.loc[idx].change_time)

    return licktimes


def dprime(hit_rate, fa_rate, limits=(0.01, 0.99)):
    from scipy.stats import norm
    Z = norm.ppf

    # Limit values in order to avoid d' infinity
    hit_rate = np.clip(hit_rate, limits[0], limits[1])
    fa_rate = np.clip(fa_rate, limits[0], limits[1])

    return Z(hit_rate) - Z(fa_rate)


# -> analyze
=======
>>>>>>> upstream/foraging-2
def get_response_rates(df_in2, sliding_window=100, reward_window=None):

    df_in = df_in2.copy()
    try:
        df_in.reset_index(inplace=True)
    except ValueError:
        del df_in['level_0']
        df_in.reset_index(inplace=True)

    go_responses = pd.Series([np.nan] * len(df_in))
    go_responses[
        df_in[
            (df_in.trial_type == 'go')
            & (df_in.response == 1)
            & (df_in.auto_rewarded != True)
        ].index
    ] = 1
    go_responses[
        df_in[
            (df_in.trial_type == 'go')
            & ((df_in.response == 0) | np.isnan(df_in.response))
            & (df_in.auto_rewarded != True)
        ].index
    ] = 0

    hit_rate = go_responses.rolling(
        window=sliding_window,
        min_periods=0,
    ).mean()

    catch_responses = pd.Series([np.nan] * len(df_in))
    catch_responses[
        df_in[
            (df_in.trial_type == 'catch')
            & (df_in.response == 1)
        ].index
    ] = 1
    catch_responses[
        df_in[
            (df_in.trial_type == 'catch')
            & ((df_in.response == 0) | np.isnan(df_in.response))
        ].index
    ] = 0

    catch_rate = catch_responses.rolling(
        window=sliding_window,
        min_periods=0,
    ).mean()

    d_prime = dprime(hit_rate, catch_rate)

    return hit_rate.values, catch_rate.values, d_prime


class RisingEdge():
    """
    This object implements a "rising edge" detector on a boolean array.

    It takes advantage of how pandas applies functions in order.

    For example, if the "criteria" column in the `df` dataframe consists of booleans indicating
    whether the row meets a criterion, we can detect the first run of three rows above criterion
    with the following

        first_run_of_three = (
            df['criteria']
            .rolling(center=False,window=3)
            .apply(func=RisingEdge().check)
            )

    ```

    """
    def __init__(self):
        self.firstall = False

    def check(self, arr):
        if arr.all():
            self.firstall = True
        return self.firstall


# -> metrics
def dprime(hit_rate, fa_rate, limits=(0.01, 0.99)):
    """ calculates the d-prime for a given hit rate and false alarm rate

    https://en.wikipedia.org/wiki/Sensitivity_index

    Parameters
    ----------
    hit_rate : float
        rate of hits in the True class
    fa_rate : float
        rate of false alarms in the False class
    limits : tuple, optional
        limits on extreme values, which distort. default: (0.01,0.99)

    Returns
    -------
    d_prime

    """
    assert limits[0] > 0.0, 'limits[0] must be greater than 0.0'
    assert limits[1] < 1.0, 'limits[1] must be less than 1.0'
    Z = norm.ppf

    # Limit values in order to avoid d' infinity
    hit_rate = np.clip(hit_rate, limits[0], limits[1])
    fa_rate = np.clip(fa_rate, limits[0], limits[1])

    return Z(hit_rate) - Z(fa_rate)


def calc_deriv(x, time):
    dx = np.diff(x)
    dt = np.diff(time)
    dxdt_rt = np.hstack((np.nan, dx / dt))
    dxdt_lt = np.hstack((dx / dt, np.nan))

    dxdt = np.vstack((dxdt_rt, dxdt_lt))

    dxdt = np.nanmean(dxdt, axis=0)

    return dxdt


def rad_to_dist(speed_rad_per_s):
    wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter
    running_radius = 0.5 * (
        2.0 * wheel_diameter / 3.0)  # assume the animal runs at 2/3 the distance from the wheel center
    running_speed_cm_per_sec = np.pi * speed_rad_per_s * running_radius / 180.
    return running_speed_cm_per_sec
