import os
import pandas as pd
import numpy as np
from six import iteritems
from functools import wraps
from .processing import get_licks, get_time
from .legacy_data import inplace


@inplace
def annotate_parameters(trials, data, keydict=None):
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
        for key, value in iteritems(keydict):
            try:
                trials[key] = [data[value]] * len(trials)
            except KeyError:
                trials[key] = None


# @inplace
# def explode_startdatetime(df):
#     """ explodes the 'startdatetime' column into date/year/month/day/hour/dayofweek
#
#     Parameters
#     ----------
#     trials : pandas DataFrame
#         dataframe of trials (or flashes)
#     inplace : bool, optional
#         modify `trials` in place. if False, returns a copy. default: True
#
#     See Also
#     --------
#     io.load_trials
#     """
#     df['date'] = df['startdatetime'].dt.date.astype(str)
#     df['year'] = df['startdatetime'].dt.year
#     df['month'] = df['startdatetime'].dt.month
#     df['day'] = df['startdatetime'].dt.day
#     df['hour'] = df['startdatetime'].dt.hour
#     df['dayofweek'] = df['startdatetime'].dt.weekday


@inplace
def annotate_n_rewards(trials):
    """computes the number of rewards from the 'reward_times' column

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
        trials['number_of_rewards'] = trials['rewards'].map(len)
    except KeyError:
        trials['number_of_rewards'] = None


@inplace
def annotate_reward_volume(trials, data):
    """adds a column with each trial's total volume

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    data : exp data
        dictionary probably

    See Also
    --------
    io.load_trials
    """
    volume_per_reward = data["config"]["reward"]["reward_volume"]

    trials["reward_volume"] = trials["rewards"].map(
        lambda r: len(r) * volume_per_reward
    )


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
    trials["change"] = [
        params["catch"] is not True
        for params in trials["trial_params"]
    ]
    trials["detect"] = trials["success"] is True
