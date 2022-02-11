# @nickponvert
# For calculating trial and flash responses
import numpy as np
import pandas as pd
import xarray as xr

from visual_behavior.ophys.response_analysis import utilities as ut


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
        "window_around_timepoint_seconds": [-2, 2],
        "response_window_duration_seconds": 0.25,
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
        "response_window_duration_seconds": 0.25,
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
    omission_response_params = {
        "window_around_timepoint_seconds": [-1, 2],
        "response_window_duration_seconds": 0.75,
        "baseline_window_duration_seconds": 0.25
    }
    return omission_response_params


def get_default_lick_response_params():
    '''
        Get default parameters for computing lick triggered average dFF or events
        including the window around each lick time to take a snippet of the dF/F trace for each cell,
        the duration of time after each lick time to take the mean_response,
        and the duration of time before each lick time to take the baseline_response.
        Args:
            None
        Returns
            (dict) dict of response window params for computing lick_response_df
        '''
    lick_response_params = {
        "window_around_timepoint_seconds": [-5, 5],
        "response_window_duration_seconds": 0.75,
        "baseline_window_duration_seconds": 0.25
    }
    return lick_response_params


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


def get_nan_trace_cell_specimen_ids(dff_traces):
    nan_indices = np.unique(np.where(np.isnan(np.vstack(dff_traces.dff.values)))[0])
    nan_cell_ids = dff_traces.index[nan_indices]
    return nan_cell_ids


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


def get_p_value_from_shuffled_omissions(mean_responses,
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
        number_of_shuffles (int): Number of shuffles of omission activity used to produce the p-value
    Returns:
        p_values (xarray.DataArray): p-value for each response mean, shape (nConditions, nCells)
    '''

    import visual_behavior.ophys.response_analysis.utilities as ut

    stim_table = stimulus_presentations_df.copy()
    omitted_flashes = stim_table[stim_table.omitted == True]
    omitted_flashes.at[:, 'start_frame'] = [ut.get_nearest_frame(start_time, ophys_timestamps) for start_time in
                                            omitted_flashes.start_time.values]
    omitted_start_frames = omitted_flashes.start_frame.values
    # exclude last omission for cases where it occured at the end of the recording
    omitted_start_frames = omitted_start_frames[:-1]
    # shuffle omitted flash frames
    shuffled_omitted_start_frames = np.random.choice(omitted_start_frames, number_of_shuffles)

    if ophys_frame_rate is None:
        ophys_frame_rate = 1 / np.diff(ophys_timestamps).mean()

    trace_len = np.round(response_window_duration * ophys_frame_rate).astype(int)
    start_ind_offset = 0
    end_ind_offset = trace_len
    omission_traces = eventlocked_traces(dff_traces_arr, shuffled_omitted_start_frames, start_ind_offset,
                                         end_ind_offset)
    omission_mean = omission_traces.mean(axis=0)  # Returns (nShuffles, nCells)

    # Goal is to figure out how each response compares to the shuffled distribution, which is just
    # a searchsorted call if we first sort the shuffled.
    omission_mean_sorted = np.sort(omission_mean, axis=0)
    response_insertion_ind = np.empty(mean_responses.data.shape)
    for ind_cell in range(mean_responses.data.shape[1]):
        response_insertion_ind[:, ind_cell] = np.searchsorted(omission_mean_sorted[:, ind_cell],
                                                              mean_responses.data[:, ind_cell])

    proportion_omission_larger_than_sample = 1 - (response_insertion_ind / number_of_shuffles)
    result = xr.DataArray(data=proportion_omission_larger_than_sample, coords=mean_responses.coords)
    return result


def get_p_value_from_shuffled_flashes(mean_responses,
                                      stimulus_presentations_df,
                                      ophys_timestamps,
                                      dff_traces_arr,
                                      response_window_duration,
                                      ophys_frame_rate=None,
                                      number_of_shuffles=10000):
    '''
    Args:
        mean_responses (xarray.DataArray): Mean response values for omissions, shape (nConditions, nCells)
        stimulus_presentations_df (pandas.DataFrame): Table of stimulus presentations, including start_time and stop_time
        ophys_timestamps (np.array): Timestamps of each ophys frame
        dff_traces_arr (np.array): Dff values, shape (nSamples, nCells)
        response_window_duration (int): Number of frames averaged to produce mean response values
        number_of_shuffles (int): Number of shuffles of flash responses used to produce the p-value
    Returns:
        p_values (xarray.DataArray): p-value for each response mean, shape (nConditions, nCells)
    '''

    import visual_behavior.ophys.response_analysis.utilities as ut

    stim_table = stimulus_presentations_df.copy()
    stimulus_flashes = stim_table[stim_table.omitted == False]
    stimulus_flashes.at[:, 'start_frame'] = [ut.get_nearest_frame(start_time, ophys_timestamps) for start_time in
                                             stimulus_flashes.start_time.values]
    stimulus_flash_start_frames = stimulus_flashes.start_frame.values
    stimulus_flash_start_frames = stimulus_flash_start_frames[:-1]  # exclude last one
    # shuffle flash frames
    shuffled_stimulus_flash_start_frames = np.random.choice(stimulus_flash_start_frames, number_of_shuffles)

    if ophys_frame_rate is None:
        ophys_frame_rate = 1 / np.diff(ophys_timestamps).mean()

    trace_len = np.round(response_window_duration * ophys_frame_rate).astype(int)
    start_ind_offset = 0
    end_ind_offset = trace_len
    flash_traces = eventlocked_traces(dff_traces_arr, shuffled_stimulus_flash_start_frames, start_ind_offset,
                                      end_ind_offset)
    flash_mean = flash_traces.mean(axis=0)  # Returns (nShuffles, nCells)

    # Goal is to figure out how each response compares to the shuffled distribution, which is just
    # a searchsorted call if we first sort the shuffled.
    flash_mean_sorted = np.sort(flash_mean, axis=0)
    response_insertion_ind = np.empty(mean_responses.data.shape)
    for ind_cell in range(mean_responses.data.shape[1]):
        response_insertion_ind[:, ind_cell] = np.searchsorted(flash_mean_sorted[:, ind_cell],
                                                              mean_responses.data[:, ind_cell])

    proportion_flash_larger_than_sample = 1 - (response_insertion_ind / number_of_shuffles)
    result = xr.DataArray(data=proportion_flash_larger_than_sample, coords=mean_responses.coords)
    return result


def get_response_xr(session, traces, timestamps, event_times, event_ids, trace_ids, response_analysis_params,
                    frame_rate=None):
    event_indices, start_ind_offset, end_ind_offset, trace_timebase = slice_inds_and_offsets(
        ophys_times=timestamps,
        event_times=event_times,
        window_around_timepoint_seconds=response_analysis_params['window_around_timepoint_seconds'],
        frame_rate=frame_rate
    )
    sliced_dataout = eventlocked_traces(traces, event_indices, start_ind_offset, end_ind_offset)

    eventlocked_traces_xr = xr.DataArray(
        data=sliced_dataout,
        dims=("eventlocked_timestamps", "trial_id", "trace_id"),
        coords={
            "eventlocked_timestamps": trace_timebase,
            "trial_id": event_ids,
            "trace_id": trace_ids
        }
    )

    response_range = [0, response_analysis_params['response_window_duration_seconds']]
    baseline_range = [-response_analysis_params['baseline_window_duration_seconds'], 0]

    mean_response = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*response_range)}
    ].mean(['eventlocked_timestamps'])

    mean_baseline = eventlocked_traces_xr.loc[
        {'eventlocked_timestamps': slice(*baseline_range)}
    ].mean(['eventlocked_timestamps'])

    if True not in session.stimulus_presentations.omitted.unique():
        nan_values = np.zeros((len(mean_response), len(trace_ids)))
        nan_values[:] = np.nan
        p_values_omission = xr.DataArray(data=nan_values,
                                         coords=mean_response.coords)
    else:
        p_values_omission = get_p_value_from_shuffled_omissions(mean_response,
                                                                session.stimulus_presentations,
                                                                timestamps,
                                                                traces,
                                                                response_analysis_params[
                                                                    'response_window_duration_seconds'],
                                                                frame_rate)
    p_values_stimulus = get_p_value_from_shuffled_flashes(mean_response,
                                                          session.stimulus_presentations,
                                                          timestamps,
                                                          traces,
                                                          response_analysis_params['response_window_duration_seconds'],
                                                          frame_rate)

    p_value_gray_screen = get_p_value_from_shuffled_spontaneous(mean_response,
                                                                session.stimulus_presentations,
                                                                timestamps,
                                                                traces,
                                                                response_analysis_params[
                                                                    'response_window_duration_seconds'],
                                                                frame_rate)
    result = xr.Dataset({
        'eventlocked_traces': eventlocked_traces_xr,
        'mean_response': mean_response,
        'mean_baseline': mean_baseline,
        'p_value_omission': p_values_omission,
        'p_value_stimulus': p_values_stimulus,
        'p_value_gray_screen': p_value_gray_screen,
    })

    return result


def response_df(response_xr):
    '''
    Smash things into df format if you want.
    '''
    traces = response_xr['eventlocked_traces']
    mean_response = response_xr['mean_response']
    mean_baseline = response_xr['mean_baseline']
    p_vals_omission = response_xr['p_value_omission']
    p_vals_stimulus = response_xr['p_value_stimulus']
    p_vals_gray_screen = response_xr['p_value_gray_screen']
    stacked_traces = traces.stack(multi_index=('trial_id', 'trace_id')).transpose()
    stacked_response = mean_response.stack(multi_index=('trial_id', 'trace_id')).transpose()
    stacked_baseline = mean_baseline.stack(multi_index=('trial_id', 'trace_id')).transpose()
    stacked_pval_omission = p_vals_omission.stack(multi_index=('trial_id', 'trace_id')).transpose()
    stacked_pval_stimulus = p_vals_stimulus.stack(multi_index=('trial_id', 'trace_id')).transpose()
    stacked_pval_gray_screen = p_vals_gray_screen.stack(multi_index=('trial_id', 'trace_id')).transpose()

    num_repeats = len(stacked_traces)
    trace_timestamps = np.repeat(
        stacked_traces.coords['eventlocked_timestamps'].data[np.newaxis, :],
        repeats=num_repeats, axis=0)

    df = pd.DataFrame({
        'trial_id': stacked_traces.coords['trial_id'],
        'trace_id': stacked_traces.coords['trace_id'],
        'trace': list(stacked_traces.data),
        'trace_timestamps': list(trace_timestamps),
        'mean_response': stacked_response.data,
        'baseline_response': stacked_baseline.data,
        'p_value_gray_screen': stacked_pval_gray_screen,
        'p_value_omission': stacked_pval_omission,
        'p_value_stimulus': stacked_pval_stimulus,
    })
    return df


def filter_events_array(trace_arr, scale=2, t_scale=20):
    from scipy import stats
    filt = stats.halfnorm(loc=0, scale=scale).pdf(np.arange(t_scale))
    filt = filt / np.sum(filt)  # normalize filter
    filtered_arr = np.empty(trace_arr.shape)
    for ind_cell in range(trace_arr.shape[0]):
        this_trace = trace_arr[ind_cell, :]
        this_trace_filtered = np.convolve(this_trace, filt)[:len(this_trace)]
        filtered_arr[ind_cell, :] = this_trace_filtered
    return filtered_arr


def get_trials_response_xr(dataset, use_events=False, filter_events=False, frame_rate=None, time_window=None):
    if use_events:
        if filter_events:
            traces = np.stack(dataset.events['filtered_events'].values)
        else:
            traces = np.stack(dataset.events['events'].values)
    else:
        traces = np.stack(dataset.dff_traces['dff'].values)
    # trace_ids = dataset.dff_traces.cell_roi_id.values # use cell_roi_ids when no cell_specimen_ids
    trace_ids = dataset.dff_traces.index.values

    timestamps = dataset.ophys_timestamps
    change_trials = dataset.trials[~pd.isnull(dataset.trials['change_time'])]  # [:-1]  # last trial can get cut off
    event_times = change_trials['change_time'].values
    event_ids = change_trials.index.values
    if time_window is None:
        response_analysis_params = get_default_trial_response_params()
        # if use_events:
        #     response_analysis_params['response_window_duration_seconds'] = 0.25
    else:
        response_analysis_params = get_default_trial_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    return response_xr


def get_trials_response_df(dataset, use_events=False, filter_events=False, frame_rate=None, df_format='wide', time_window=None):
    response_xr = get_trials_response_xr(dataset, use_events, filter_events, frame_rate, time_window)

    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()

    # include response window used to create df
    params = get_default_trial_response_params()
    df['response_window'] = params['response_window_duration_seconds']

    df = df.rename(columns={'trial_id': 'trials_id', 'trace_id': 'cell_specimen_id'})
    return df


def get_stimulus_response_xr(dataset, use_events=False, filter_events=True, frame_rate=None, time_window=None):
    if use_events:
        if filter_events:
            traces = np.stack(dataset.events['filtered_events'].values)
        else:
            traces = np.stack(dataset.events['events'].values)
    else:
        traces = np.stack(dataset.dff_traces['dff'].values)
    # trace_ids = dataset.dff_traces.cell_roi_id.values # use cell_roi_ids when no cell_specimen_ids
    trace_ids = dataset.dff_traces.index.values

    timestamps = dataset.ophys_timestamps
    event_times = dataset.stimulus_presentations['start_time'].values
    event_ids = dataset.stimulus_presentations.index.values
    if time_window is None:
        response_analysis_params = get_default_stimulus_response_params()
        # if use_events:
        #     response_analysis_params['response_window_duration_seconds'] = 0.25
    else:
        response_analysis_params = get_default_stimulus_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    return response_xr


def get_stimulus_response_df(dataset, use_events=False, filter_events=False, frame_rate=None, df_format='wide', time_window=None):
    response_xr = get_stimulus_response_xr(dataset, use_events, filter_events, frame_rate, time_window=time_window)

    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()

    # include response window used to create df
    params = get_default_stimulus_response_params()
    df['response_window'] = params['response_window_duration_seconds']

    df = df.rename(columns={'trial_id': 'stimulus_presentations_id', 'trace_id': 'cell_specimen_id'})
    return df


def get_omission_response_xr(dataset, use_events=False, filter_events=False, frame_rate=None, time_window=None):
    if use_events:
        if filter_events:
            traces = np.stack(dataset.events['filtered_events'].values)
        else:
            traces = np.stack(dataset.events['events'].values)
    else:
        traces = np.stack(dataset.dff_traces['dff'].values)
    # trace_ids = dataset.dff_traces.cell_roi_id.values # use cell_roi_ids when no cell_specimen_ids
    trace_ids = dataset.dff_traces.index.values

    timestamps = dataset.ophys_timestamps
    stimuli = dataset.stimulus_presentations
    omission_presentations = stimuli[stimuli.image_name == 'omitted']
    event_times = omission_presentations['start_time'].values
    event_ids = omission_presentations.index.values
    if time_window is None:
        response_analysis_params = get_default_omission_response_params()
        # if use_events:
        #     response_analysis_params['response_window_duration_seconds'] = 0.25
    else:
        response_analysis_params = get_default_omission_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    return response_xr


def get_omission_response_df(dataset, use_events=False, filter_events=False, frame_rate=None, df_format='wide', time_window=None):
    response_xr = get_omission_response_xr(dataset, use_events, filter_events, frame_rate, time_window)

    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()

    # include response window used to create df
    params = get_default_omission_response_params()
    df['response_window'] = params['response_window_duration_seconds']

    df = df.rename(
        columns={'trial_id': 'stimulus_presentations_id', 'trace_id': 'cell_specimen_id'})
    return df


def get_trials_run_speed_df(dataset, frame_rate=None, df_format='wide', time_window=None):
    traces = np.vstack((dataset.running_speed.speed.values, dataset.running_speed.speed.values))
    trace_ids = np.asarray([0, 1])
    timestamps = dataset.stimulus_timestamps
    change_trials = dataset.trials[~pd.isnull(dataset.trials['change_time'])][:-1]  # last trial can get cut off
    event_times = change_trials['change_time'].values
    event_ids = change_trials.index.values
    if time_window is None:
        response_analysis_params = get_default_trial_response_params()
    else:
        response_analysis_params = get_default_trial_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()
    df = df.rename(columns={'trial_id': 'trials_id', 'trace_id': 'tmp'})
    df = df[df['tmp'] == 0].drop(columns=['tmp']).reset_index()
    return df


def get_stimulus_run_speed_df(dataset, frame_rate=None, df_format='wide', time_window=None):
    traces = np.vstack((dataset.running_speed.speed.values, dataset.running_speed.speed.values))
    trace_ids = [0, 1]
    timestamps = dataset.stimulus_timestamps
    event_times = dataset.stimulus_presentations['start_time'].values[:-1]  # last one can get truncated
    event_ids = dataset.stimulus_presentations.index.values[:-1]
    if time_window is None:
        response_analysis_params = get_default_stimulus_response_params()
    else:
        response_analysis_params = get_default_stimulus_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()

    df = df.rename(columns={'trial_id': 'stimulus_presentations_id', 'trace_id': 'tmp'})
    df = df[df['tmp'] == 0].drop(columns=['tmp']).reset_index()

    window = response_analysis_params['window_around_timepoint_seconds']
    response_window = [np.abs(window[0]),
                       np.abs(window[0]) + response_analysis_params['response_window_duration_seconds']]
    if frame_rate is None:
        frame_rate = 1 / np.diff(timestamps).mean()
    if df_format == 'wide':
        df['p_value_baseline'] = [ut.get_p_val(trace, response_window, frame_rate) for trace in df.trace.values]
    return df


def get_omission_run_speed_df(dataset, frame_rate=None, df_format='wide', time_window=None):
    traces = np.vstack((dataset.running_speed.speed.values, dataset.running_speed.speed.values))
    trace_ids = [0, 1]
    timestamps = dataset.stimulus_timestamps
    stimuli = dataset.stimulus_presentations
    omission_presentations = stimuli[stimuli.image_name == 'omitted']
    event_times = omission_presentations['start_time'].values[:-1]  # last omission can get truncated
    # event_indices = index_of_nearest_value(timestamps, event_times)
    event_ids = omission_presentations.index.values[:-1]
    if time_window is None:
        response_analysis_params = get_default_omission_response_params()
    else:
        response_analysis_params = get_default_omission_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()

    df = df.rename(columns={'trial_id': 'stimulus_presentations_id', 'trace_id': 'tmp'})
    df = df[df['tmp'] == 0].drop(columns=['tmp']).reset_index()
    return df


def get_trials_pupil_area_df(dataset, frame_rate=None, df_format='wide', time_window=None):
    pupil_area = dataset.eye_tracking.pupil_area.values
    traces = np.vstack((pupil_area, pupil_area))
    trace_ids = [0, 1]
    timestamps = dataset.eye_tracking.timestamps.values
    change_trials = dataset.trials[~pd.isnull(dataset.trials['change_time'])][:-1]  # last trial can get cut off
    event_times = change_trials['change_time'].values
    event_ids = change_trials.index.values
    if time_window is None:
        response_analysis_params = get_default_trial_response_params()
    else:
        response_analysis_params = get_default_trial_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()

    df = df.rename(columns={'trial_id': 'trials_id', 'trace_id': 'tmp'})
    df = df[df['tmp'] == 0].drop(columns=['tmp']).reset_index()

    window = response_analysis_params['window_around_timepoint_seconds']
    response_window = [np.abs(window[0]),
                       np.abs(window[0]) + response_analysis_params['response_window_duration_seconds']]
    if frame_rate is None:
        frame_rate = 1 / np.diff(timestamps).mean()
    df['p_value_baseline'] = [ut.get_p_val(trace, response_window, frame_rate) for trace in df.trace.values]
    # remove trials with nan values
    ind = np.where(np.asarray([np.isnan(trace).any() for trace in df.trace.values]) == True)[0]
    df = df.drop(index=ind)
    df = df.drop(columns='index')
    return df


def get_stimulus_pupil_area_df(dataset, frame_rate=None, df_format='wide', time_window=None):
    pupil_area = dataset.eye_tracking.pupil_area.values
    traces = np.vstack((pupil_area, pupil_area))
    trace_ids = [0, 1]
    timestamps = dataset.eye_tracking.timestamps.values
    event_times = dataset.stimulus_presentations['start_time'].values[:-1]  # last one can get truncated
    event_ids = dataset.stimulus_presentations.index.values[:-1]
    if time_window is None:
        response_analysis_params = get_default_stimulus_response_params()
    else:
        response_analysis_params = get_default_stimulus_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()

    df = df.rename(columns={'trial_id': 'stimulus_presentations_id', 'trace_id': 'tmp'})
    df = df[df['tmp'] == 0].drop(columns=['tmp']).reset_index()

    window = response_analysis_params['window_around_timepoint_seconds']
    response_window = [np.abs(window[0]),
                       np.abs(window[0]) + response_analysis_params['response_window_duration_seconds']]
    if frame_rate is None:
        frame_rate = 1 / np.diff(timestamps).mean()
    df['p_value_baseline'] = [ut.get_p_val(trace, response_window, frame_rate) for trace in df.trace.values]
    # remove trials with nan values
    ind = np.where(np.asarray([np.isnan(trace).any() for trace in df.trace.values]) == True)[0]
    df = df.drop(index=ind)
    df = df.drop(columns='index')
    return df


def get_omission_pupil_area_df(dataset, frame_rate=30, df_format='wide', time_window=None):
    pupil_area = dataset.eye_tracking.pupil_area.values
    traces = np.vstack((pupil_area, pupil_area))
    trace_ids = [0, 1]
    timestamps = dataset.eye_tracking.timestamps.values
    stimuli = dataset.stimulus_presentations
    omission_presentations = stimuli[stimuli.image_name == 'omitted']
    event_times = omission_presentations['start_time'].values[:-1]  # last omission can get truncated
    # event_indices = index_of_nearest_value(timestamps, event_times)
    event_ids = omission_presentations.index.values[:-1]
    if time_window is None:
        response_analysis_params = get_default_omission_response_params()
    else:
        response_analysis_params = get_default_omission_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()

    df = df.rename(columns={'trial_id': 'stimulus_presentations_id', 'trace_id': 'tmp'})
    df = df[df['tmp'] == 0].drop(columns=['tmp']).reset_index()
    # remove trials with nan values
    ind = np.where(np.asarray([np.isnan(trace).any() for trace in df.trace.values]) == True)[0]
    df = df.drop(index=ind)
    df = df.drop(columns='index')
    return df


def get_lick_binary(dataset):
    licks = dataset.licks.timestamps.values.copy()
    times = dataset.stimulus_timestamps.copy()
    lick_binary = np.asarray([1 if time in licks else 0 for time in times])
    return lick_binary


def get_trials_licks_df(dataset, frame_rate=None, df_format='wide', time_window=None):
    licks = get_lick_binary(dataset)
    traces = np.vstack((licks, licks))
    trace_ids = [0, 1]
    timestamps = dataset.stimulus_timestamps
    change_trials = dataset.trials[~pd.isnull(dataset.trials['change_time'])][:-1]  # last trial can get cut off
    event_times = change_trials['change_time'].values
    event_ids = change_trials.index.values
    if time_window is None:
        response_analysis_params = get_default_trial_response_params()
    else:
        response_analysis_params = get_default_trial_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()
    df = df.rename(columns={'trial_id': 'trials_id', 'trace_id': 'tmp'})
    df = df[df['tmp'] == 0].drop(columns=['tmp']).reset_index()
    return df


def get_omission_licks_df(dataset, frame_rate=None, df_format='wide', time_window=None):
    licks = get_lick_binary(dataset)
    traces = np.vstack((licks, licks))
    trace_ids = [0, 1]
    timestamps = dataset.stimulus_timestamps
    stimuli = dataset.stimulus_presentations
    omission_presentations = stimuli[stimuli.image_name == 'omitted']
    event_times = omission_presentations['start_time'].values[:-1]  # last omission can get truncated
    # event_indices = index_of_nearest_value(timestamps, event_times)
    event_ids = omission_presentations.index.values[:-1]
    if time_window is None:
        response_analysis_params = get_default_omission_response_params()
    else:
        response_analysis_params = get_default_omission_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()

    df = df.rename(columns={'trial_id': 'stimulus_presentations_id', 'trace_id': 'tmp'})
    df = df[df['tmp'] == 0].drop(columns=['tmp']).reset_index()
    return df


def get_lick_triggered_response_xr(dataset, use_events=False, filter_events=False, frame_rate=None, time_window=None):
    import visual_behavior.data_access.processing as processing
    if use_events:
        if filter_events:
            traces = np.stack(dataset.events['filtered_events'].values)
        else:
            traces = np.stack(dataset.events['events'].values)
    else:
        traces = np.stack(dataset.dff_traces['dff'].values)
    trace_ids = dataset.dff_traces.index.values
    timestamps = dataset.ophys_timestamps
    licks = dataset.licks.copy()
    licks = processing.add_bouts_to_licks(licks)
    # only use the first lick in a bout to trigger averaging
    licks = licks[licks.lick_in_bout == False]
    event_times = licks.timestamps.values
    event_ids = licks.index.values
    if time_window is None:
        response_analysis_params = get_default_lick_response_params()
    else:
        response_analysis_params = get_default_lick_response_params()
        response_analysis_params['window_around_timepoint_seconds'] = time_window

    response_xr = get_response_xr(dataset, traces, timestamps, event_times, event_ids, trace_ids,
                                  response_analysis_params, frame_rate)
    return response_xr


def get_lick_triggered_response_df(dataset, use_events=False, filter_events=False, frame_rate=None, df_format='wide'):
    response_xr = get_lick_triggered_response_xr(dataset, use_events, filter_events, frame_rate)

    if df_format == 'wide':
        df = response_df(response_xr)
    elif df_format == 'tidy' or df_format == 'long':
        df = response_xr.to_dataframe().reset_index()
    df = df.rename(
        columns={'trial_id': 'lick_id', 'trace_id': 'cell_specimen_id'})
    return df


if __name__ == "__main__":
    from allensdk.brain_observatory.behavior import behavior_project_cache as bpc

    manifest_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/sfn_2019/manifest_20190916.json'
    cache = bpc.InternalCacheFromLims(manifest=manifest_path)
    session = cache.get_session(806203732)
    trial_response_xr = get_trials_response_xr(session)
    #  result = trial_response_df(trial_response_xr)
