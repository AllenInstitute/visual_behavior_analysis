import numpy as np
from scipy.ndimage import median_filter as medfilt


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


def compute_running_speed(dx_raw, time, v_sig, v_in, smooth=False):
    """Calculate running speed

    Parameters
    ----------
    dx_raw: numpy.ndarray
        dx values for each stimulus frame
    time: numpy.ndarray
        timestamps for each stimulus frame
    v_sig: numpy.ndarray
        v_sig for each stimulus frame: currently unused
    v_in: numpy.ndarray
        v_in for each stimulus frame: currently unused
    smooth: boolean, default=False
        flag to smooth output: not implemented

    Returns
    -------
    numpy.ndarray
        Running speed (cm/s)
    """
    dx = medfilt(dx_raw, size=5)  # remove big, single frame spikes in encoder values
    dx = np.cumsum(dx)  # wheel rotations
    speed = calc_deriv(dx, time)  # speed in degrees/s
    speed = deg_to_dist(speed)

    if smooth:
        raise NotImplementedError

    return speed, time
