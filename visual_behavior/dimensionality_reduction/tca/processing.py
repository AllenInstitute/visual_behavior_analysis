from visual_behavior.ophys.response_analysis import response_processing as rp
import xarray as xr
import numpy as np
# import os
# import tensortools
import visual_behavior.data_access.loading as loading
import pandas as pd


def zscore_scaler(group):
    return (group - np.mean(group)) / np.std(group)


def min_max_scaler(group):
    X = (group - np.min(group)) / (np.max(group) - np.min(group))
    X = np.nan_to_num(X, nan=0)
    return X


def select_traces(xr, time=[-0.5, 0.75]):
    xr_sel = xr['eventlocked_traces'].sel({'eventlocked_timestamps': slice(time[0], time[1])})
    return xr_sel


def transpose_tensor(xr_sel):
    X = xr_sel.transpose('trace_id', 'eventlocked_timestamps', 'trial_id').data
    return X


def create_container_tensor(oeids, use_events=True, filter_events=True, time_window=None):
    '''
    Creates xr tensor from datasets in one container.
    Tensor is concatinated across sessions
    Tensor = num_cells x response profile x sessions
    :param ids:
    :return: xr
    '''
    same_ids = get_matched_specimen_id(oeids)

    for i, oeid in enumerate(oeids):
        dataset = loading.get_ophys_dataset(oeid)
        response_xr = rp.get_stimulus_response_xr(dataset, use_events=use_events, filter_events=filter_events, time_window=time_window)
        response_xr.coords['eventlocked_timestamps'] = response_xr.coords['eventlocked_timestamps'].round(3)

        index = response_xr['trace_id'].isin(same_ids).values
        same_ids = response_xr.isel({'trace_id': index})['trace_id'].values
        if i == 0:
            response_xrs = response_xr.sel(trace_id=same_ids)
        else:
            response_xr = response_xr.sel(trace_id=same_ids)
            response_xr['trial_id'] = response_xr['trial_id'] + response_xrs['trial_id'].max() + 1
            response_xrs = xr.concat([response_xrs, response_xr], dim='trial_id')
    return response_xrs


def create_session_tensor(oeids, use_events=True, filter_events=True, time_window=[-.5, .75]):
    '''
    Creates tensor from datasets in one session.
    Concatinate across cortical layers and areas.
    Tensor = num_cells x response profile x trials

    :param oeids:
    :return: response_xrs
    '''

    for i, oeid in enumerate(oeids):
        dataset = loading.get_ophys_dataset(oeid)
        response_xr = rp.get_stimulus_response_xr(dataset, use_events=use_events, filter_events=filter_events, time_window=time_window )
        response_xr.coords['eventlocked_timestamps'] = response_xr.coords['eventlocked_timestamps'].round(3)

        if i == 0:
            response_xrs = response_xr
        else:
            response_xrs = xr.concat((response_xrs, response_xr), dim='trace_id')
    return response_xrs


def get_cells_df(oeids):
    cells_df = pd.DataFrame(columns=['cell_specimen_id', 'targeted_structure', 'imaging_depth'])
    for i, oeid in enumerate(oeids):
        dataset = loading.get_ophys_dataset(oeid)
        cell_specimen_ids = dataset.events.index.values
        targeted_structure = dataset.metadata['targeted_structure']
        imaging_depth = dataset.metadata['imaging_depth']
        for i in range(0, len(cell_specimen_ids)):
            cells_df = cells_df.append({'cell_specimen_id': cell_specimen_ids[i],
                                        'targeted_structure': targeted_structure,
                                        'imaging_depth': imaging_depth}, ignore_index=True)
    return cells_df


def get_matched_specimen_id(oeids):
    for i, oeid in enumerate(oeids):
        dataset = loading.get_ophys_dataset(oeid)
        df = dataset.dff_traces.reset_index()
        if i == 0:
            same_ids = dataset.events.index.values
        else:
            index = df.cell_specimen_id.isin(same_ids).values
            same_ids = df.cell_specimen_id[index].values
    return same_ids


def get_stimulus_df(oeids):
    for i, oeid in enumerate(oeids):
        dataset = loading.get_ophys_dataset(oeid)
        df = dataset.extended_stimulus_presentations.copy()
        df['session_number'] = np.repeat(int(dataset.metadata['session_type'][6]), len(df))
        if i == 0:
            stim_df = df.copy()
        else:
            stim_df = stim_df.append(df, ignore_index=True)
    # stim_df['engagement_binary'] = stim_df['engagement_state'].apply(lambda x: True if x == 'engaged' else False)
    return stim_df
