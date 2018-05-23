from __future__ import print_function
from dateutil import tz
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


def local_time(iso_timestamp, timezone=None):
    datetime = pd.to_datetime(iso_timestamp)
    if not datetime.tzinfo:
        datetime = datetime.replace(tzinfo=tz.gettz('America/Los_Angeles'))
    return datetime.isoformat()
