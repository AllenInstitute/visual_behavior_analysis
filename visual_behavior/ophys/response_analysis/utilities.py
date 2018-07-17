"""
Created on Saturday July 14 2018

@author: marinag
"""

import numpy as np
from scipy import stats


def get_nearest_frame(timepoint, timestamps):
    return int(np.nanargmin(abs(timestamps - timepoint)))

def get_trace_around_timepoint(timepoint, trace, timestamps, window, frame_rate):
    frame_for_timepoint = get_nearest_frame(timepoint, timestamps)
    lower_frame = frame_for_timepoint + (window[0] * frame_rate)
    upper_frame = frame_for_timepoint + (window[1] * frame_rate)
    trace = trace[int(lower_frame):int(upper_frame)]
    timepoints = timestamps[int(lower_frame):int(upper_frame)]
    return trace, timepoints

def get_mean_in_window(trace, window, frame_rate):
    return np.nanmean(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])

def get_sd_in_window(trace, window, frame_rate):
    return np.std(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])

def get_sd_over_baseline(trace, response_window, baseline_window, frame_rate):
    # baseline_mean = self.get_mean_in_window(trace, self.baseline_window, self.metadata['ophys_frame_rate'])
    baseline_std = get_sd_in_window(trace, baseline_window, frame_rate)
    response_mean = get_mean_in_window(trace, response_window, frame_rate)
    return response_mean / (baseline_std)

def get_p_val(trace, response_window, frame_rate):
    # method borrowed from Brain Observatory analysis in allensdk
    response_window_duration = response_window[1] - response_window[0]
    baseline_end = int(response_window[0] * frame_rate)
    baseline_start = int((response_window[0] - response_window_duration) * frame_rate)
    stim_start = int(response_window[0] * frame_rate)
    stim_end = int((response_window[0] + response_window_duration) * frame_rate)
    (_, p) = stats.f_oneway(trace[baseline_start:baseline_end], trace[stim_start:stim_end])
    return p

def ptest(x, num_conditions):
    ptest = len(np.where(x < (0.05 / num_conditions))[0])
    return ptest