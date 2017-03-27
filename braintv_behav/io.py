import pandas as pd

from functools import wraps
def data_or_pkl(func):

    def pkl_wrapper(first_arg,*args,**kwargs):

        if isinstance(obj, basestring):
            return pd.read_pickle(pkl)
        else:
            return first_arg


    return pkl_wrapper
    
@data_or_pkl
def load_flashes(data):
    """
    Loads a dataframe of flashes presented during a DoC session.
    
    Accepts a data object from a DoC session or a path to a pkl file for the session.

    Returns: pandas dataframe
    """

    stimdf = pd.DataFrame(data['stimuluslog'])
    
    # first we find the flashes
    try:
        assert pd.isnull(stimdf['image_category']).any()==False
        flashes = stimdf[stimdf['state'].astype(int).diff()>0].reset_index()[['image_category','image_name','frame']]
        flashes['change'] = (flashes['image_category'].diff()!=0)
    except (AssertionError,KeyError) as e:
        print e
        flashes = stimdf[stimdf['state'].astype(int).diff()>0].reset_index()[['ori','frame']]
        flashes['change'] = (flashes['ori'].diff()!=0)
    flashes['time'] = data['vsyncintervals'].cumsum()[flashes['frame']]
    

    # then we find the licks
    lick_frames = data['lickData'][0]-1

    licks = pd.DataFrame(dict(
            frame = lick_frames,
            time = data['vsyncintervals'].cumsum()[lick_frames],
            flash = np.searchsorted(flashes['frame'].values,lick_frames) - 1,
    ))
    licks = licks[licks['frame'].diff()!=1] # filter out redundant licks
    licks = licks[licks['flash'].diff()>0] # get first lick from each flash
    
    # then we merge in the licks
    flashes = flashes.merge(
        licks,
        left_index=True,
        right_on='flash',
        suffixes=('','_lick'),
        how='left'
    ).set_index('flash')
    
    
    flashes['lick'] = ~pd.isnull(flashes['time_lick'])
    class Counter():
        def __init__(self):
            self.count = np.nan
        def check(self,val):
            count = self.count
            if val>0:
                self.count = 1
            else:
                self.count += 1
            return count

    flashes['last_lick'] = flashes['lick'].map(Counter().check)
    
    #then we find the rewards
    reward_frames = data['rewards'][:,1].astype(int) - 1
    
    rewards = pd.DataFrame(dict(
            frame = reward_frames,
            time = data['vsyncintervals'].cumsum()[reward_frames],
            flash = np.searchsorted(flashes['frame'].values,reward_frames) - 1,
    ))
    
#     assert False
    
    # then we merge in the rewards
    flashes = flashes.merge(
        rewards,
        left_index=True,
        right_on='flash',
        suffixes=('','_reward'),
        how='left',
    ).set_index('flash')
    
    # finally, we assign the trials
    trial_bounds = pd.DataFrame([dict(index=tr['index'],startframe=tr['startframe']) for tr in data['triallog']])
    flashes['trial'] = np.searchsorted(trial_bounds['startframe'].values,flashes['frame'].values) - 1
    
    flashes['mouse_id'] = data['mouseid']
    flashes['datetime'] = data['startdatetime']
    flashes['task'] = data['task']
    flashes['stage'] = data['stage']
    flashes['flashed'] = data['blank_duration_range'][1]>0
    flashes['stimulus'] = data['stimulus']
    
    return flashes