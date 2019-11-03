## @nickponvert
## For calculating trial and flash responses
import sys
import os
import numpy as np
import math
import pandas as pd
from scipy import stats
import itertools
import xarray as xr

OPHYS_FRAME_RATE = 31.


def get_default_trial_response_params():
    '''
        Get default parameters for computing trial_response_xr
        including the window around each change_time to take a snippet of the dF/F trace for each cell,
        the duration of time after each change to take the mean_response,
        and the duration of time before each change to take the baseline_response.
        Args:
            None
        Returns
            (dict) dict of response window params for computing trial_response_xr
        '''
    trial_response_params = {
        "window_around_timepoint_seconds": [-4, 8],
        "response_window_duration_seconds": 0.5,
        "baseline_window_duration_seconds": 0.25
    }
    return trial_response_params


def get_default_stimulus_response_params():
    '''
        Get default parameters for computing stimulus_response_xr
        including the window around each stimulus presentation start_time to take a snippet of the dF/F trace for each cell,
        the duration of time after each start_time to take the mean_response,
        and the duration of time before each start_time to take the baseline_response.
        Args:
            None
        Returns
            (dict) dict of response window params for computing stimulus_response_xr
        '''
    stimulus_response_params = {
        "window_around_timepoint_seconds": [-0.5, 0.75],
        "response_window_duration_seconds": 0.5,
        "baseline_window_duration_seconds": 0.25
    }
    return stimulus_response_params


def get_default_omission_response_params():
    '''
        Get default parameters for computing omission_response_xr
        including the window around each omission start_time to take a snippet of the dF/F trace for each cell,
        the duration of time after each start_time to take the mean_response,
        and the duration of time before each start_time to take the baseline_response.
        Args:
            None
        Returns
            (dict) dict of response window params for computing omission_response_xr
        '''
    stimulus_response_params = {
        "window_around_timepoint_seconds": [-3, 3],
        "response_window_duration_seconds": 0.5,
        "baseline_window_duration_seconds": 0.25
    }
    return omission_response_params


def index_of_nearest_value(sample_times, event_times):
    '''
    The index of the nearest sample time for each event time.
    Args:
        sample_times (np.ndarray of float): sorted 1-d vector of sample timestamps
        event_times (np.ndarray of float): 1-d vector of event timestamps
    Returns
        (np.ndarray of int) nearest sample time index for each event time
    '''
    insertion_ind = np.searchsorted(sample_times, event_times)
    # is the value closer to data at insertion_ind or insertion_ind-1?
    ind_diff = sample_times[insertion_ind] - event_times
    ind_minus_one_diff = np.abs(sample_times[np.clip(insertion_ind - 1, 0, np.inf).astype(int)] - event_times)
    return insertion_ind - (ind_diff > ind_minus_one_diff).astype(int)


def eventlocked_traces(dff_traces_arr, event_indices, start_ind_offset, end_ind_offset):
    '''
    Extract trace for each cell, for each event-relative window.
    Args:
        dff_traces (np.ndarray): shape (nSamples, nCells) with dff traces for each cell
        event_indices (np.ndarray): 1-d array of shape (nEvents) with closest sample ind for each event
        start_ind_offset (int): Where to start the window relative to each event ind
        end_ind_offset (int): Where to end the window relative to each event ind
    Returns:
        sliced_dataout (np.ndarray): shape (nSamples, nEvents, nCells)
    '''
    all_inds = event_indices + np.arange(start_ind_offset, end_ind_offset)[:, None]
    sliced_dataout = dff_traces_arr.T[all_inds]
    return sliced_dataout


def slice_inds_and_offsets(ophys_times, event_times, window_around_timepoint_seconds, frame_rate=None):
    '''
    Get nearest indices to event times, plus ind offsets for slicing out a window around the event from the trace.
    Args:
        ophys_times (np.array): timestamps of ophys frames
        event_times (np.array): timestamps of events around which to slice windows
        window_around_timepoint_seconds (list): [start_offset, end_offset] for window
        frame_rate (float): we shouldn't need this. leave none to infer from the ophys timestamps
    '''
    if frame_rate is None:
        frame_rate = 1 / np.diff(ophys_times).mean()
    event_indices = index_of_nearest_value(ophys_times, event_times)
    trace_len = (window_around_timepoint_seconds[1] - window_around_timepoint_seconds[0]) * frame_rate
    start_ind_offset = int(window_around_timepoint_seconds[0] * frame_rate)
    end_ind_offset = int(start_ind_offset + trace_len)
    trace_timebase = np.arange(start_ind_offset, end_ind_offset) / frame_rate
    return event_indices, start_ind_offset, end_ind_offset, trace_timebase


def trial_response_xr(session, response_analysis_params=None):
    if response_analysis_params is None:
        response_analysis_params = get_default_trial_response_params()

    dff_traces_arr = np.stack(session.dff_traces['dff'].values)
    change_trials = session.trials[~pd.isnull(session.trials['change_time'])]
    event_times = change_trials['change_time'].values

    event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(
        ophys_times=session.ophys_timestamps,
        event_times=event_times,
        window_around_timepoint_seconds=response_analysis_params['window_around_timepoint_seconds']
    )
    sliced_dataout = eventlocked_traces(dff_traces_arr, event_indices, start_ind_offset, end_ind_offset)

    eventlocked_traces_xr = xr.DataArray(
        data=sliced_dataout,
        dims=("eventlocked_timestamps", "trials_id", "cell_specimen_id"),
        coords={
            "eventlocked_timestamps": trace_timebase,
            "trials_id": change_trials.index.values,
            "cell_specimen_id": session.cell_specimen_table.index.values
        }
    )

    response_range = [0, response_analysis_params['response_window_duration_seconds']]
    baseline_range = [-1 * response_analysis_params['baseline_window_duration_seconds']]

    mean_response = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*response_range)}
    ].mean(['eventlocked_timestamps'])

    mean_baseline = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*baseline_range)}
    ].mean(['eventlocked_timestamps'])

    dff_traces_arr = np.stack(session.dff_traces['dff'].values)
    p_values = get_p_value_from_shuffled_spontaneous(mean_response,
                                                     session.stimulus_presentations,
                                                     session.ophys_timestamps,
                                                     dff_traces_arr,
                                                     response_analysis_params['response_window_duration_seconds'])
    result = xr.Dataset({
        'eventlocked_traces': eventlocked_traces_xr,
        'mean_response': mean_response,
        'mean_baseline': mean_baseline,
        'p_value': p_values
    })

    return result


def get_nan_trace_cell_specimen_ids(dff_traces):
    nan_indices = np.unique(np.where(np.isnan(np.vstack(dff_traces.dff.values)))[0])
    nan_cell_ids = dff_traces.index[nan_indices]
    return nan_cell_ids


def stimulus_response_xr(session, response_analysis_params=None):
    if response_analysis_params is None:
        response_analysis_params = get_default_stimulus_response_params()

    # # get rid of cells that are invalid and have NaNs in their traces
    # if not session.filter_invalid_rois:
    #     nan_cell_ids = get_nan_trace_cell_specimen_ids(session.dff_traces)
    #     dff_traces = session.dff_traces.drop(labels=nan_cell_ids)
    #     cell_specimen_table = session.cell_specimen_table.drop(labels=nan_cell_ids)
    # else:
    dff_traces = session.dff_traces
    # cell_specimen_table = session.cell_specimen_table

    dff_traces_arr = np.stack(dff_traces['dff'].values)
    event_times = session.stimulus_presentations['start_time'].values
    event_indices = index_of_nearest_value(session.ophys_timestamps, event_times)

    event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(
        ophys_times=session.ophys_timestamps,
        event_times=event_times,
        window_around_timepoint_seconds=response_analysis_params['window_around_timepoint_seconds']
    )
    sliced_dataout = eventlocked_traces(dff_traces_arr, event_indices, start_ind_offset, end_ind_offset)

    eventlocked_traces_xr = xr.DataArray(
        data=sliced_dataout,
        dims=("eventlocked_timestamps", "stimulus_presentations_id", "cell_specimen_id"),
        coords={
            "eventlocked_timestamps": trace_timebase,
            "stimulus_presentations_id": session.stimulus_presentations.index.values,
            "cell_specimen_id": session.cell_specimen_ids
        }
    )

    response_range = [0, response_analysis_params['response_window_duration_seconds']]
    baseline_range = [-1 * response_analysis_params['baseline_window_duration_seconds'], 0]

    mean_response = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*response_range)}
    ].mean(['eventlocked_timestamps'])

    mean_baseline = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*baseline_range)}
    ].mean(['eventlocked_timestamps'])

    p_values = get_p_value_from_shuffled_spontaneous(mean_response,
                                                     session.stimulus_presentations,
                                                     session.ophys_timestamps,
                                                     dff_traces_arr,
                                                     response_analysis_params['response_window_duration_seconds'])
    result = xr.Dataset({
        'eventlocked_traces': eventlocked_traces_xr,
        'mean_response': mean_response,
        'mean_baseline': mean_baseline,
        'p_value': p_values
    })

    result = result.merge(session.stimulus_presentations[['image_index', 'image_name']])
    mean_response_per_image = result[['mean_response', 'image_index']].groupby('image_index').mean(
        dim='stimulus_presentations_id')
    image_indices = mean_response_per_image.coords['image_index']
    pref_image_index = mean_response_per_image.drop(8, dim='image_index')['mean_response'].argmax(
        dim='image_index')  # drop omitted
    result['pref_image_index'] = pref_image_index
    result['pref_image_bool'] = result['image_index'] == result['pref_image_index']

    return result


def omission_response_xr(session, response_analysis_params=None):
    if response_analysis_params is None:
        response_analysis_params = get_default_omission_response_params()

    # get rid of cells that are invalid and have NaNs in their traces
    if not session.filter_invalid_rois:
        nan_cell_ids = get_nan_trace_cell_specimen_ids(session.dff_traces)
        dff_traces = session.dff_traces.drop(labels=nan_cell_ids)
        cell_specimen_table = session.cell_specimen_table.drop(labels=nan_cell_ids)
    else:
        dff_traces = session.dff_traces
        cell_specimen_table = session.cell_specimen_table

    dff_traces_arr = np.stack(dff_traces['dff'].values)
    # get omissions only
    stimuli = session.stimulus_presentations
    omission_presentations = stimuli[stimuli.image_name == 'omitted']
    event_times = omission_presentations['start_time'].values
    event_indices = index_of_nearest_value(session.ophys_timestamps, event_times)

    event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(
        ophys_times=session.ophys_timestamps,
        event_times=event_times,
        window_around_timepoint_seconds=response_analysis_params['window_around_timepoint_seconds']
    )
    sliced_dataout = eventlocked_traces(dff_traces_arr, event_indices, start_ind_offset, end_ind_offset)

    eventlocked_traces_xr = xr.DataArray(
        data=sliced_dataout,
        dims=("eventlocked_timestamps", "stimulus_presentations_id", "cell_specimen_id"),
        coords={
            "eventlocked_timestamps": trace_timebase,
            "stimulus_presentations_id": omission_presentations.index.values,
            "cell_specimen_id": cell_specimen_table.index.values
        }
    )

    response_range = [0, response_analysis_params['response_window_duration_seconds']]
    baseline_range = [-1 * response_analysis_params['baseline_window_duration_seconds']]

    mean_response = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*response_range)}
    ].mean(['eventlocked_timestamps'])

    mean_baseline = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*baseline_range)}
    ].mean(['eventlocked_timestamps'])

    p_values = get_p_value_from_shuffled_spontaneous(mean_response,
                                                     omission_presentations,
                                                     session.ophys_timestamps,
                                                     dff_traces_arr,
                                                     response_analysis_params['response_window_duration_seconds'])
    result = xr.Dataset({
        'eventlocked_traces': eventlocked_traces_xr,
        'mean_response': mean_response,
        'mean_baseline': mean_baseline,
        'p_value': p_values
    })

    return result


def trial_response_df(trial_response_xr):
    '''
    Smash things into df format if you want.
    '''
    traces = trial_response_xr['eventlocked_traces']
    mean_response = trial_response_xr['mean_response']
    mean_baseline = trial_response_xr['mean_baseline']
    p_vals = trial_response_xr['p_value']
    stacked_traces = traces.stack(multi_index=('trials_id', 'cell_specimen_id')).transpose()
    stacked_response = mean_response.stack(multi_index=('trials_id', 'cell_specimen_id')).transpose()
    stacked_baseline = mean_baseline.stack(multi_index=('trials_id', 'cell_specimen_id')).transpose()
    stacked_pval = p_vals.stack(multi_index=('trials_id', 'cell_specimen_id')).transpose()

    num_repeats = len(stacked_traces)
    trace_timestamps = np.repeat(
        stacked_traces.coords['eventlocked_timestamps'].data[np.newaxis, :],
        repeats=num_repeats, axis=0)

    df = pd.DataFrame({
        'trials_id': stacked_traces.coords['trials_id'],
        'cell_specimen_id': stacked_traces.coords['cell_specimen_id'],
        'dff_trace': list(stacked_traces.data),
        'dff_trace_timestamps': list(trace_timestamps),
        'mean_response': stacked_response.data,
        'baseline_response': stacked_baseline.data,
        'p_value': stacked_pval
    })
    return df


def stimulus_response_df(stimulus_response_xr):
    '''
    Smash things into df format if you want.
    '''
    traces = stimulus_response_xr['eventlocked_traces']
    mean_response = stimulus_response_xr['mean_response']
    mean_baseline = stimulus_response_xr['mean_baseline']
    p_vals = stimulus_response_xr['p_value']
    preferred_bool = stimulus_response_xr['pref_image_bool']
    stacked_traces = traces.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_response = mean_response.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_baseline = mean_baseline.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_pval = p_vals.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_preferred_bool = preferred_bool.stack(
        multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()

    num_repeats = len(stacked_traces)
    trace_timestamps = np.repeat(
        stacked_traces.coords['eventlocked_timestamps'].data[np.newaxis, :],
        repeats=num_repeats, axis=0)

    df = pd.DataFrame({
        'stimulus_presentations_id': stacked_traces.coords['stimulus_presentations_id'],
        'cell_specimen_id': stacked_traces.coords['cell_specimen_id'],
        'dff_trace': list(stacked_traces.data),
        'dff_trace_timestamps': list(trace_timestamps),
        'mean_response': stacked_response.data,
        'baseline_response': stacked_baseline.data,
        'p_value': stacked_pval,
        'pref_stim': stacked_preferred_bool
    })
    return df


def omission_response_df(omission_response_xr):
    '''
    Smash things into df format if you want.
    '''
    traces = omission_response_xr['eventlocked_traces']
    mean_response = omission_response_xr['mean_response']
    mean_baseline = omission_response_xr['mean_baseline']
    p_vals = omission_response_xr['p_value']

    stacked_traces = traces.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_response = mean_response.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_baseline = mean_baseline.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()
    stacked_pval = p_vals.stack(multi_index=('stimulus_presentations_id', 'cell_specimen_id')).transpose()

    num_repeats = len(stacked_traces)
    trace_timestamps = np.repeat(
        stacked_traces.coords['eventlocked_timestamps'].data[np.newaxis, :],
        repeats=num_repeats, axis=0)

    df = pd.DataFrame({
        'stimulus_presentations_id': stacked_traces.coords['stimulus_presentations_id'],
        'cell_specimen_id': stacked_traces.coords['cell_specimen_id'],
        'dff_trace': list(stacked_traces.data),
        'dff_trace_timestamps': list(trace_timestamps),
        'mean_response': stacked_response.data,
        'baseline_response': stacked_baseline.data,
        'p_value': stacked_pval,
    })
    return df


def to_df(response_xr):
    eventlocked_traces = response_xr['eventlocked_traces']
    eventlocked_traces_stacked = eventlocked_traces.stack(
        multi_index=('stimulus_presentations_id', 'cell_specimen_id')
    ).transpose()
    eventlocked_timestamps = response_xr['eventlocked_timestamps']

    df_pre = response_xr.drop(['eventlocked_traces', 'eventlocked_timestamps']).to_dataframe().reset_index()
    df_pre.insert(loc=1, column='dff_trace', value=traces_list)
    df_pre.insert(loc=2, column='dff_trace_timestamps', value=trace_timestamps_list)


def get_spontaneous_frames(stimulus_presentations_df, ophys_timestamps):
    '''
        Returns a list of the frames that occur during the before and after spontaneous windows. This is copied from VBA. Does not use the full spontaneous period because that is what VBA did. It only uses 4 minutes of the before and after spontaneous period.

    Args:
        stimulus_presentations_df (pandas.DataFrame): table of stimulus presentations, including start_time and stop_time
        ophys_timestamps (np.array): Timestamps of each ophys frame
    Returns:
        spontaneous_inds (np.array): indices of ophys frames during the spontaneous period
    '''
    # dont use full 5 mins to avoid fingerprint and countdown
    # spont_duration_frames = 4 * 60 * 60  # 4 mins * * 60s/min * 60Hz
    spont_duration = 4 * 60  # 4mins * 60sec

    # for spontaneous at beginning of session
    behavior_start_time = stimulus_presentations_df.iloc[0].start_time
    spontaneous_start_time_pre = behavior_start_time - spont_duration
    spontaneous_end_time_pre = behavior_start_time
    spontaneous_start_frame_pre = index_of_nearest_value(ophys_timestamps, spontaneous_start_time_pre)
    spontaneous_end_frame_pre = index_of_nearest_value(ophys_timestamps, spontaneous_end_time_pre)
    spontaneous_frames_pre = np.arange(spontaneous_start_frame_pre, spontaneous_end_frame_pre, 1)

    # for spontaneous epoch at end of session
    behavior_end_time = stimulus_presentations_df.iloc[-1].start_time
    spontaneous_start_time_post = behavior_end_time + 0.75
    spontaneous_end_time_post = spontaneous_start_time_post + spont_duration
    spontaneous_start_frame_post = index_of_nearest_value(ophys_timestamps, spontaneous_start_time_post)
    spontaneous_end_frame_post = index_of_nearest_value(ophys_timestamps, spontaneous_end_time_post)
    spontaneous_frames_post = np.arange(spontaneous_start_frame_post, spontaneous_end_frame_post, 1)

    spontaneous_frames = np.concatenate([spontaneous_frames_pre, spontaneous_frames_post])
    return spontaneous_frames


def get_p_value_from_shuffled_spontaneous(mean_responses,
                                          stimulus_presentations_df,
                                          ophys_timestamps,
                                          dff_traces_arr,
                                          response_window_duration,
                                          ophys_frame_rate=None,
                                          number_of_shuffles=10000):
    '''
    Args:
        mean_responses (xarray.DataArray): Mean response values, shape (nConditions, nCells)
        stimulus_presentations_df (pandas.DataFrame): Table of stimulus presentations, including start_time and stop_time
        ophys_timestamps (np.array): Timestamps of each ophys frame
        dff_traces_arr (np.array): Dff values, shape (nSamples, nCells)
        response_window_duration (int): Number of frames averaged to produce mean response values
        number_of_shuffles (int): Number of shuffles of spontaneous activity used to produce the p-value
    Returns:
        p_values (xarray.DataArray): p-value for each response mean, shape (nConditions, nCells)
    '''

    spontaneous_frames = get_spontaneous_frames(stimulus_presentations_df, ophys_timestamps)
    shuffled_spont_inds = np.random.choice(spontaneous_frames, number_of_shuffles)

    if ophys_frame_rate is None:
        ophys_frame_rate = 1 / np.diff(ophys_timestamps).mean()

    trace_len = np.round(response_window_duration * ophys_frame_rate).astype(int)
    start_ind_offset = 0
    end_ind_offset = trace_len
    spont_traces = eventlocked_traces(dff_traces_arr, shuffled_spont_inds, start_ind_offset, end_ind_offset)
    spont_mean = spont_traces.mean(axis=0)  # Returns (nShuffles, nCells)

    # Goal is to figure out how each response compares to the shuffled distribution, which is just
    # a searchsorted call if we first sort the shuffled.
    spont_mean_sorted = np.sort(spont_mean, axis=0)
    response_insertion_ind = np.empty(mean_responses.data.shape)
    for ind_cell in range(mean_responses.data.shape[1]):
        response_insertion_ind[:, ind_cell] = np.searchsorted(spont_mean_sorted[:, ind_cell],
                                                              mean_responses.data[:, ind_cell])

    proportion_spont_larger_than_sample = 1 - (response_insertion_ind / number_of_shuffles)
    result = xr.DataArray(data=proportion_spont_larger_than_sample,
                          coords=mean_responses.coords)
    return result


def get_flash_response_df():
    fdf = rp.stimulus_response_df(
        rp.stimulus_response_xr(session, response_analysis_params=rp.get_default_flash_response_params()))


if __name__ == "__main__":
    import time
    from allensdk.brain_observatory.behavior import behavior_project_cache as bpc

    manifest_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/sfn_2019/manifest_20190916.json'
    cache = bpc.InternalCacheFromLims(manifest=manifest_path)
    session = cache.get_session(806203732)
    trial_response_xr = trial_response_xr(session)
    #  result = trial_response_df(trial_response_xr)
