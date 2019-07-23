import numpy as np
from scipy.signal import medfilt


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


def compute_running_speed(dx_raw, time, v_sig, v_in):
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
        flag to smooth output using medfilt

    Returns
    -------
    numpy.ndarray
        Running speed (cm/s)
    """

    dx = np.cumsum(dx)  # wheel rotations
    speed = calc_deriv(dx, time)  # speed in degrees/s
    speed = deg_to_dist(speed)

    return speed


def compute_running_speed_sdk(dx_raw, 
                                time, 
                                v_sig, 
                                v_in,
                                wheel_radius=3.25*2.54, #6.5 inch wheel diameter, 2.54 cm/inch
                                subject_position = 2.0/3.0
                                ):
    """Calculate running speed using translation of "extract_running_speeds" from allen sdk.
    Originally written by Nils G, translated by Matt V
    see: camstim.behavior.BehaviorEncoder for derivation of dx from v_sig and v_in

    Parameters
    ----------
    dx_raw: numpy.ndarray
        dx values for each stimulus frame
    time: numpy.ndarray
        timestamps for each stimulus frame
    smooth: boolean, default=False
        flag to smooth output using medfilt
    wheel_radius: 
        radius in cm
    subject position:
        position of mouse from center of wheel: fraction of radius from center 

    Returns
    -------
    numpy.ndarray
        Running speed (cm/s)
    """

    # due to an acquisition bug (the buffer of raw orientations may be updated
    # more slowly than it is read, leading to a 0 value for the change in
    # orientation over an interval) there may be exact zeros in the velocity.
    nonzero_indices = np.flatnonzero(dx_raw)
    dx_raw = dx_raw[nonzero_indices]
    frame_times = time[nonzero_indices]
    v_sig = np.array(v_sig)[nonzero_indices]
    v_in = np.array(v_in)[nonzero_indices]

    # the first interval does not have a known start time, so we can't compute
    # an average velocity from dx
    dx_rad = np.array(dx_raw[1:]) * (np.pi / 180.0)

    start_times = frame_times[:-1]
    end_times = frame_times[1:]

    durations = end_times - start_times
    angular_velocity = dx_rad / durations

    radius = wheel_radius * subject_position
    linear_velocity = np.multiply(angular_velocity, radius)

    return linear_velocity, dx_raw[1:], frame_times[1:], list(v_sig[1:]), list(v_in[1:])