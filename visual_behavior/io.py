from __future__ import print_function
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from .analyze import calc_deriv, rad_to_dist


from functools import wraps
def data_or_pkl(func):
    """ Decorator that allows a function to accept a pickled experiment object
    or a path to the object.

    >>> @data_or_pkl
    >>> def print_keys(data):
    >>>     print data.keys()

    """

    @wraps(func)
    def pkl_wrapper(first_arg,*args,**kwargs):

        if isinstance(first_arg, basestring):
            return func(pd.read_pickle(first_arg),*args,**kwargs)
        else:
            return func(first_arg,*args,**kwargs)


    return pkl_wrapper

@data_or_pkl
def load_params(data):
    """ Returns the parameters passed in to an experiment.

    Parameters
    ----------
    data : dict, unpickled experiment (or path to pickled object)

    Returns
    -------
    params : dict

    """
    return data['params']

@data_or_pkl
def load_trials(data):
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
    vsync = np.hstack((0,data['vsyncintervals']))
    return (vsync.cumsum()) / 1000.0

@data_or_pkl
def load_licks(data,time=None):
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
    responsedf=pd.DataFrame(data['responselog'])
    lick_frames = responsedf.frame.values
    if time is None:
        time = load_time(data)

    licks = pd.DataFrame(dict(
            frame = lick_frames,
            time = time[lick_frames],
    ))

    licks[licks['frame'].diff()!=1]

    return licks

@data_or_pkl
def load_rewards(data,time=None):
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
        reward_frames = data['rewards'][:,1].astype(int)
    except IndexError:
        reward_frames = np.array([],dtype=int)
    if time is None:
        time = load_time(data)
    rewards = pd.DataFrame(dict(
            frame = reward_frames,
            time = time[reward_frames],
    ))
    return rewards


@data_or_pkl
def load_flashes(data,time=None):
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

    # for some reason, we sometimes have stim frames with no corresponding time in the vsyncintervals.
    # we should probably fix the problem, but for now, let's just delete them.
    stimdf = stimdf[stimdf['frame']<len(time)]
    # assert stimdf['frame'].max() < len(time)

    # first we find the flashes
    try:
        assert pd.isnull(stimdf['image_category']).any()==False
        flashes = stimdf[stimdf['state'].astype(int).diff()>0].reset_index()[['image_category','image_name','frame']]
#         flashes['change'] = (flashes['image_category'].diff()!=0)
        flashes['prior_image_category'] = flashes['image_category'].shift()
        flashes['image_category_change'] = flashes['image_category'].ne(flashes['prior_image_category']).astype(int)

        flashes['prior_image_name'] = flashes['image_name'].shift()
        flashes['image_name_change'] = flashes['image_name'].ne(flashes['prior_image_name']).astype(int)

        flashes['change'] = flashes['image_category_change']
    except (AssertionError,KeyError) as e:
        # print "error in {}: {}".format(pkl,e)
        flashes = stimdf[stimdf['state'].astype(int).diff()>0].reset_index()[['ori','frame']]
        flashes['prior_ori'] = flashes['ori'].shift()
        flashes['ori_change'] = flashes['ori'].ne(flashes['prior_ori']).astype(int)

        flashes['change'] = flashes['ori_change']

    if len(flashes)==0:
        return flashes

    flashes['time'] = time[flashes['frame']]

    # then we find the licks
    licks = load_licks(data)
    licks['flash_index'] = np.searchsorted(
        flashes['frame'].values,
        licks['frame'].values,
        side='right',
        ) - 1
    licks = licks[licks['flash_index'].diff()>0] # get first lick from each flash

    # then we merge in the licks
    flashes = flashes.merge(
        licks,
        left_index=True,
        right_on='flash_index',
        suffixes=('','_lick'),
        how='left'
    ).set_index('flash_index')


    flashes['lick'] = ~pd.isnull(flashes['time_lick'])
    class Counter():
        def __init__(self):
            self.count = np.nan
        def count_it(self,val):
            count = self.count
            if val>0:
                self.count = 1
            else:
                self.count += 1
            return count

    flashes['last_lick'] = flashes['lick'].map(Counter().count_it)

    #then we find the rewards
    rewards = load_rewards(data)
    rewards['flash_index'] = np.searchsorted(
        flashes['frame'].values,
        rewards['frame'].values,
        side='right',
        ) - 1

    # then we merge in the rewards
    flashes = flashes.merge(
        rewards,
        left_index=True,
        right_on='flash_index',
        suffixes=('','_reward'),
        how='left',
    ).set_index('flash_index')

    flashes['reward'] = ~pd.isnull(flashes['time_reward'])

    # finally, we assign the trials
    try:
        trial_bounds = [dict(index=tr['index'],startframe=tr['startframe']) for tr in data['triallog'] if 'startframe' in tr]
    except KeyError:
        trial_bounds = [dict(index=tr_index,startframe=tr['startframe']) for tr_index,tr in enumerate(data['triallog']) if 'startframe' in tr]

    trial_bounds = pd.DataFrame(trial_bounds)
    flashes['trial'] = np.searchsorted(
        trial_bounds['startframe'].values,
        flashes['frame'].values,
        side='right',
        ) - 1

    flashes['flashed'] = data['blank_duration_range'][1]>0

    flashes['mouse_id'] = data['mouseid']
    flashes['datetime'] = data['startdatetime']
    flashes['task'] = data['task']
    flashes['stage'] = data['stage']

    session_params = [
        'initial_blank',
        'delta_minimum',
        'delta_mean',
        'stimulus_distribution',
        'trial_duration',
        'stimulus',
        ]

    for param in session_params:
        try:
            flashes[param] = data[param]
        except KeyError:
            pass

    return flashes


@data_or_pkl
def load_running_speed(data,smooth=False,time=None):

    if time is None:
        print('`time` not passed. using vsync from pkl file')
        time = load_time(data)

    dx = np.array(data['dx'])
    dx = medfilt(dx, kernel_size=5)  # remove big, single frame spikes in encoder values
    dx = np.cumsum(dx)  # wheel rotations

    time = time[:len(dx)]

    speed = calc_deriv(dx,time)
    speed = rad_to_dist(speed)

    if smooth:
        #running_speed_cm_per_sec = pd.rolling_mean(running_speed_cm_per_sec, window=6)
        raise NotImplementedError

    accel = calc_deriv(speed,time)
    jerk = calc_deriv(accel,time)


    running_speed = pd.DataFrame(
        {
            'time': time,
            'speed (cm/s)':speed,
            'acceleration (cm/s^2)': accel,
            'jerk (cm/s^3)': jerk,
            },
        )
    return running_speed
