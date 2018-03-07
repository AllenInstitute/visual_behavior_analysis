import os
import pandas as pd
import numpy as np
from six import iteritems
from functools import wraps
from .extract import get_licks, get_time  # maybe revise this relationship...this could get out of hand quickly...
# from .legacy_data import inplace


"""transform functions take a trials dataframe (and potentially other things)
and modifies the supplied dataframe, so watch out about mutability
"""


def annotate_parameters(trials, data, keydict=None):
    """ annotates a dataframe with session parameters

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials (or flashes)
    data : unpickled session
    keydict : dict
        {'column': 'parameter'}
    """
    if keydict is None:
        return
    else:
        for key, value in iteritems(keydict):
            try:
                trials[key] = [data[value]] * len(trials)
            except KeyError:
                trials[key] = None


def annotate_n_rewards(trials):
    """computes the number of rewards from the 'reward_times' column

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    """
    try:
        trials['number_of_rewards'] = trials['rewards'].map(len)
    except KeyError:
        trials['number_of_rewards'] = None


def annotate_reward_volume(trials, data):
    """adds a column with each trial's total volume

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    data : exp data
        dictionary probably
    """
    volume_per_reward = data["config"]["reward"]["reward_volume"]

    trials["reward_volume"] = trials["rewards"].map(
        lambda r: len(r) * volume_per_reward
    )


def annotate_change_detect(trials):
    """ adds `change` and `detect` columns to dataframe

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    """
    trials["change"] = [
        params["catch"] is not True
        for params in trials["trial_params"]
    ]
    trials["detect"] = trials["success"] is True
