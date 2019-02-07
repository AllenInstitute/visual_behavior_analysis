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
        self.save_running_speed(obj)
        self.save_stimulus_timestamps(obj)
        self.save_stimulus_table(obj)
        self.save_stimulus_template(obj)
        self.save_licks(obj)
        self.save_rewards(obj)
        self.save_stimulus_metadata(obj)
        self.save_task_parameters(obj)
        self.save_extended_dataframe(obj)
        self.save_corrected_fluorescence_traces(obj)
        self.save_average_image(obj)
        self.save_motion_correction(obj)


    @property
    def max_projection_cache_file_info(self):
        return os.path.join(self.cache_dir, 'max_projection.h5'), '/data'

    def get_max_projection(self, *args, **kwargs):
        return read_data_h5(*self.max_projection_cache_file_info)

    def save_max_projection(self, obj):
        save_data_h5(obj.max_projection, *self.max_projection_cache_file_info)


    @property
    def roi_metrics_cache_file_info(self):
        return os.path.join(self.cache_dir, 'roi_metrics.h5'), 'df'

    def get_roi_metrics(self, *args, **kwargs):
        return read_df_h5(*self.roi_metrics_cache_file_info)

    def save_roi_metrics(self, obj):
        save_df_h5(obj.roi_metrics, *self.roi_metrics_cache_file_info)


    @property
    def dff_cache_file_info(self):
        return os.path.join(self.cache_dir, 'dff.h5'), 'data'

    def get_dff_traces(self, *args, **kwargs):
        return read_df_h5(*self.dff_cache_file_info)

    def get_cell_roi_ids(self, *args, **kwargs):
        df = read_df_h5(*self.dff_cache_file_info)
        return df['cell_roi_id'].values

    def get_ophys_timestamps(self, *args, **kwargs):
        df = read_df_h5(*self.dff_cache_file_info)
        return df['timestamps'].values[0]

    def save_dff_traces(self, obj):
        save_df_h5(obj.dff_traces, *self.dff_cache_file_info)


    @property
    def roi_mask_file_info(self):
        return os.path.join(self.cache_dir, 'roi_masks.h5'), 'data'

    def get_roi_masks(self, *args, **kwargs):
        return read_df_h5(*self.roi_mask_file_info)

    def save_roi_masks(self, obj):
        save_df_h5(obj.roi_masks, *self.roi_mask_file_info)


    @property
    def running_speed_file_info(self):
        return os.path.join(self.cache_dir, 'running_speed.h5'), 'data'

    def get_running_speed(self, *args, **kwargs):
        return read_df_h5(*self.running_speed_file_info)

    def save_running_speed(self, obj):
        save_df_h5(obj.running_speed, *self.running_speed_file_info)


    @property
    def stimulus_timestamps_file_info(self):
        return os.path.join(self.cache_dir, 'stimulus_timestamps.h5'), 'data'

    def get_stimulus_timestamps(self, *args, **kwargs):
        return read_data_h5(*self.stimulus_timestamps_file_info)

    def save_stimulus_timestamps(self, obj):
        save_data_h5(obj.stimulus_timestamps, *self.stimulus_timestamps_file_info)


    @property
    def stimulus_template_file_info(self):
        return os.path.join(self.cache_dir, 'stimulus_template.h5'), 'data'

    def get_stimulus_template(self, *args, **kwargs):
        return read_data_h5(*self.stimulus_template_file_info)

    def save_stimulus_template(self, obj):
        save_data_h5(obj.stimulus_template, *self.stimulus_template_file_info)


    @property
    def stimulus_table_file_info(self):
        return os.path.join(self.cache_dir, 'stimulus_table.h5'), 'data'

    def get_stimulus_table(self, *args, **kwargs):
        return read_df_h5(*self.stimulus_table_file_info)

    def save_stimulus_table(self, obj):
        save_df_h5(obj.stimulus_table, *self.stimulus_table_file_info)


    @property
    def stimulus_metadata_file_info(self):
        return os.path.join(self.cache_dir, 'stimulus_metadata.h5'), 'data'

    def get_stimulus_metadata(self, *args, **kwargs):
        return read_df_h5(*self.stimulus_metadata_file_info)

    def save_stimulus_metadata(self, obj):
        save_df_h5(obj.stimulus_metadata, *self.stimulus_metadata_file_info)


    @property
    def licks_file_info(self):
        return os.path.join(self.cache_dir, 'licks.h5'), 'data'

    def get_licks(self, *args, **kwargs):
        return read_df_h5(*self.licks_file_info)

    def save_licks(self, obj):
        save_df_h5(obj.licks, *self.licks_file_info)


    @property
    def rewards_file_info(self):
        return os.path.join(self.cache_dir, 'rewards.h5'), 'data'

    def get_rewards(self, *args, **kwargs):
        return read_df_h5(*self.rewards_file_info)

    def save_rewards(self, obj):
        save_df_h5(obj.rewards, *self.rewards_file_info)

    @property
    def task_parameters_file_info(self):
        return os.path.join(self.cache_dir, 'task_parameters.h5'), 'data'

    def get_task_parameters(self, *args, **kwargs):
        return read_df_h5(*self.task_parameters_file_info)

    def save_task_parameters(self, obj):
        save_df_h5(obj.task_parameters, *self.task_parameters_file_info)

    @property
    def extended_dataframe_file_info(self):
        return os.path.join(self.cache_dir, 'extended_dataframe.h5'), 'data'

    def get_extended_dataframe(self, *args, **kwargs):
        return read_df_h5(*self.extended_dataframe_file_info)

    def save_extended_dataframe(self, obj):
        save_df_h5(obj.extended_dataframe, *self.extended_dataframe_file_info)


    @property
    def corrected_fluorescence_traces_file_info(self):
        return os.path.join(self.cache_dir, 'corrected_fluorescence_traces.h5'), 'data'

    def get_corrected_fluorescence_traces(self, *args, **kwargs):
        return read_df_h5(*self.corrected_fluorescence_traces_file_info)

    def save_corrected_fluorescence_traces(self, obj):
        save_df_h5(obj.corrected_fluorescence_traces, *self.corrected_fluorescence_traces_file_info)


    @property
    def average_image_cache_file_info(self):
        return os.path.join(self.cache_dir, 'average_image.h5'), '/data'

    def get_average_image(self, *args, **kwargs):
        return read_data_h5(*self.average_image_cache_file_info)

    def save_average_image(self, obj):
        save_data_h5(obj.average_image, *self.average_image_cache_file_info)


    @property
    def motion_correction_file_info(self):
        return os.path.join(self.cache_dir, 'motion_correction.h5'), 'data'

    def get_motion_correction(self, *args, **kwargs):
        return read_df_h5(*self.motion_correction_file_info)

    def save_motion_correction(self, obj):
        save_df_h5(obj.motion_correction, *self.motion_correction_file_info)