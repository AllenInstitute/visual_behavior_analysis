from __future__ import print_function
from dateutil import tz
from functools import wraps
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm


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


def get_response_rates(df_in, sliding_window=100):
    """
    calculates the rolling hit rate, false alarm rate, and dprime value
    Note that the pandas rolling metric deals with NaN values by propogating the previous non-NaN value

    Parameters
    ----------
    sliding_window : int
        Number of trials over which to calculate metrics

    Returns
    -------
    tuple containing hit rate, false alarm rate, d_prime

    """

    from visual_behavior.translator.core.annotate import is_catch, is_hit

    go_responses = df_in.apply(is_hit, axis=1)

    hit_rate = go_responses.rolling(
        window=sliding_window,
        min_periods=0,
    ).mean()

    catch_responses = go_responses = df_in.apply(is_catch, axis=1)

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

    try:
        last_hit_nan = np.where(np.isnan(hit_rate))[0].max()
    except ValueError:
        last_hit_nan = 0

    try:
        last_fa_nan = np.where(np.isnan(fa_rate))[0].max()
    except ValueError:
        last_fa_nan = 0

    last_nan = np.max((last_hit_nan, last_fa_nan))

    # fill nans with 0.5 to avoid warning about nans
    d_prime = Z(pd.Series(hit_rate).fillna(0.5)) - Z(pd.Series(fa_rate).fillna(0.5))

    # fill all values up to the last nan with nan
    d_prime[:last_nan + 1] = np.nan

    if len(d_prime) == 1:
        # if the result is a 1-length vector, return as a scalar
        return d_prime[0]
    else:
        return d_prime


def calc_deriv(x, time):
    dx = np.diff(x)
    dt = np.diff(time)
    dxdt_rt = np.hstack((np.nan, dx / dt))
    dxdt_lt = np.hstack((dx / dt, np.nan))

    dxdt = np.vstack((dxdt_rt, dxdt_lt))

    dxdt = np.nanmean(dxdt, axis=0)

    return dxdt


def deg_to_dist(speed_deg_per_s):
    '''
    takes speed in degrees per second
    converts to radians
    multiplies by radius (in cm) to get linear speed in cm/s
    '''
    wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter
    running_radius = 0.5 * (
        2.0 * wheel_diameter / 3.0)  # assume the animal runs at 2/3 the distance from the wheel center
    running_speed_cm_per_sec = np.pi * speed_deg_per_s * running_radius / 180.
    return running_speed_cm_per_sec


def local_time(iso_timestamp, timezone=None):
    datetime = pd.to_datetime(iso_timestamp)
    if not datetime.tzinfo:
        datetime = datetime.replace(tzinfo=tz.gettz('America/Los_Angeles'))
    return datetime.isoformat()


class ListHandler(logging.Handler):
    """docstring for ListHandler."""

    def __init__(self, log_list):
        super(ListHandler, self).__init__()
        self.log_list = log_list

    def emit(self, record):
        entry = self.format(record)
        self.log_list.append(entry)


DoubleColonFormatter = logging.Formatter(
    "%(levelname)s::%(name)s::%(message)s",
)


def inplace(func):
    """ decorator which allows functions that modify a dataframe inplace
    to use a copy instead
    """

    @wraps(func)
    def df_wrapper(df, *args, **kwargs):

        try:
            inplace = kwargs.pop('inplace')
        except KeyError:
            inplace = False

        if inplace is False:
            df = df.copy()

        func(df, *args, **kwargs)

        if inplace is False:
            return df
        else:
            return None

    return df_wrapper
