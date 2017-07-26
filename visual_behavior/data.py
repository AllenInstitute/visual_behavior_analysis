import os
import pandas as pd
import numpy as np
# from visual_behavior.utilities import load_from_folder


from functools import wraps

def inplace(func):
    """ decorator which allows functions that modify a dataframe inplace
    to use a copy instead
    """

    @wraps(func)
    def df_wrapper(df,*args,**kwargs):

        try:
            inplace = kwargs.pop('inplace')
        except KeyError:
            inplace = False

        if inplace==False:
            df = df.copy()

        func(df,*args,**kwargs)

        if inplace==False:
            return df
        else:
            return None

    return df_wrapper


@inplace
def annotate_parameters(trials,data,keydict=None):
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
        return
    else:
        for key,value in keydict.iteritems():
            try:
                trials[key] = [data[value]]*len(trials)
            except KeyError as e:
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
    df['date'] = df['startdatetime'].dt.date.astype(str)
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
def annotate_rig_id(df,data):
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
        df['rig_id'] = data['rig_id']
    except KeyError:
        from visual_behavior.devices import get_rig_id
        df['rig_id'] = get_rig_id(df['computer_name'][0])

@inplace
def annotate_startdatetime(df,data):
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
    df['startdatetime'] = pd.to_datetime(data['startdatetime'])

@inplace
def annotate_cumulative_reward(trials,data):
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
    except:
        trials['reward_volume'] = data['rewardvol']*trials['number_of_rewards']
        trials['cumulative_volume'] = trials['reward_volume'].cumsum()

@inplace
def annotate_filename(df,filename):
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

    trials['change'] = trials['trial_type']=='go'
    trials['detect'] = trials['response']==1.0



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

    ## build arrays for change detection
    annotate_change_detect(trials,inplace=True)

    ## calculate reaction times
    fix_change_time(trials,inplace=True)

    ## unwrap the response window
    explode_response_window(trials,inplace=True)
