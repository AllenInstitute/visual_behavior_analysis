import numpy as np


def get_end_frame(trials, metadata):

    last_frame = metadata['n_stimulus_frames']

    end_frames = np.zeros_like(trials.index) * np.nan

    for ii, index in enumerate(trials.index[:-1]):
        end_frames[ii] = int(trials.loc[index + 1].startframe - 1)
    if last_frame is not None:
        end_frames[-1] = int(last_frame)

    return end_frames.astype(np.int32)


def get_end_time(trials, time):
    '''creates a vector of end times for each trial, which is just the start time for the next trial'''

    end_times = trials['endframe'].map(lambda fr: time[int(fr)])
    return end_times


def calculate_trial_length(trials):
    trial_length = np.zeros(len(trials))
    for ii, idx in enumerate(trials.index):
        try:
            tl = trials.loc[idx + 1].starttime - trials.loc[idx].starttime
            if tl < 0 or tl > 1000:
                tl = np.nan
            trial_length[ii] = tl
        except Exception:
            pass

    return trial_length
