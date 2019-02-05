import os
import h5py
import pandas as pd
import numpy as np

import matplotlib.image as mpimg  # NOQA: E402

def save_data_h5(data, filename, loc):
    f = h5py.File(filename, 'w')
    f.create_dataset(loc, data=data)
    f.close()

def read_data_h5(filename, loc):
    with h5py.File(filename, 'r') as f: 
        data = f[loc].value
    return data

def save_df_h5(df, filename, key):
    df.to_hdf(filename, key=key, format='fixed')

def read_df_h5(filename, key):
    return pd.read_hdf(filename, key=key, format='fixed')

class VisualBehaviorFileSystemAPI:

    def __init__(self, cache_dir):

        self.cache_dir = cache_dir

    def save(self, obj):

        self.save_max_projection(obj)
        self.save_roi_metrics(obj)
        self.save_dff_traces(obj)
        self.save_roi_masks(obj)
        self.save_stimulus_timestamps(obj)


    @property
    def max_proj_cache_file_info(self):
        return os.path.join(self.cache_dir, 'max_projection.h5'), '/data'

    def get_max_projection(self, obj):
        return read_data_h5(*self.max_proj_cache_file_info)

    def save_max_projection(self, obj):
        save_data_h5(obj.max_projection, *self.max_proj_cache_file_info)


    @property
    def roi_metrics_cache_file_info(self):
        return os.path.join(self.cache_dir, 'roi_metrics.h5'), 'df'

    def save_roi_metrics(self, obj):
        save_df_h5(obj.roi_metrics, *self.roi_metrics_cache_file_info)

    def get_roi_metrics(self, obj):
        return read_df_h5(*self.roi_metrics_cache_file_info)


    @property
    def dff_cache_file_info(self):
        return os.path.join(self.cache_dir, 'dff.h5'), 'data'

    def save_dff_traces(self, obj):
        save_df_h5(obj.dff_traces, *self.dff_cache_file_info)

    def get_dff_traces(self, obj):
        return read_df_h5(*self.dff_cache_file_info)

    def get_cell_specimen_ids(self, obj):
        df = read_df_h5(*self.dff_cache_file_info)
        return df['cell_specimen_id'].values


    @property
    def roi_mask_file_info(self):
        return os.path.join(self.cache_dir, 'roi_masks.h5'), 'data'

    def save_roi_masks(self, obj):
        save_df_h5(obj.roi_masks, *self.roi_mask_file_info)

    def get_roi_masks(self, obj):
        return read_df_h5(*self.roi_mask_file_info)


    @property
    def running_speed_file_info(self):
        return os.path.join(self.cache_dir, 'running_speed.h5'), 'data'

    def save_running_speed(self, obj):
        save_df_h5(obj.running_speed, *self.running_speed_file_info)

    def get_running_speed(self, obj):
        return read_df_h5(*self.running_speed_file_info)


    @property
    def stimulus_timestamps_file_info(self):
        return os.path.join(self.cache_dir, 'stimuls_timestamps.h5'), 'data'

    def save_stimulus_timestamps(self, obj):
        save_df_h5(obj.running_speed, *self.stimulus_timestamps_file_info)

    def get_stimulus_timestamps(self, obj):
        return read_df_h5(*self.stimulus_timestamps_file_info)


    # def save_timestamp_data(self, obj):
    #     timestamps = obj.timestamp_data
    # # def save_timestamps(timestamps, dff_traces, core_data, roi_metrics, lims_data):
    #     # remove spurious frames at end of ophys session - known issue with Scientifica data
    #     if dff_traces.shape[1] < timestamps['ophys_frames']['timestamps'].shape[0]:
    #         difference = timestamps['ophys_frames']['timestamps'].shape[0] - dff_traces.shape[1]
    #         # logger.info('length of ophys timestamps >  length of traces by', str(difference), 'frames , truncating ophys timestamps')
    #         timestamps['ophys_frames']['timestamps'] = timestamps['ophys_frames']['timestamps'][:dff_traces.shape[1]]
    #     # account for dropped ophys frames - a rare but unfortunate issue
    #     if dff_traces.shape[1] > timestamps['ophys_frames']['timestamps'].shape[0]:
    #         difference = timestamps['ophys_frames']['timestamps'].shape[0] - dff_traces.shape[1]
    #         # logger.info('length of ophys timestamps <  length of traces by', str(difference), 'frames , truncating traces')
    #         dff_traces = dff_traces[:, :timestamps['ophys_frames']['timestamps'].shape[0]]
    #         save_dff_traces(dff_traces, roi_metrics, lims_data)
    #     # make sure length of timestamps equals length of running traces
    #     running_speed = core_data['running'].speed.values
    #     if len(running_speed) < timestamps['stimulus_frames']['timestamps'].shape[0]:
    #         timestamps['stimulus_frames']['timestamps'] = timestamps['stimulus_frames']['timestamps'][:len(running_speed)]
    #     save_dataframe_as_h5(timestamps, 'timestamps', get_analysis_dir(lims_data))
    
