"""
Created on Sunday July 15 2018

@author: marinag
"""

from visual_behavior.ophys.response_analysis import utilities as ut
import visual_behavior.ophys.response_analysis.response_processing as rp
import visual_behavior.ophys.dataset.stimulus_processing as sp
import visual_behavior.data_access.loading as loading

import os
import numpy as np
import pandas as pd
import itertools



class LazyLoadable(object):
    def __init__(self, name, calculate):
        ''' Wrapper for attributes intended to be computed or loaded once, then held in memory by a containing object.

        Parameters
        ----------
        name : str
            The name of the hidden attribute in which this attribute's data will be stored.
        calculate : fn
            a function (presumably expensive) used to calculate or load this attribute's data

        '''

        self.name = name
        self.calculate = calculate

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, self.name):
            setattr(obj, self.name, self.calculate(obj))
        return getattr(obj, self.name)

class ResponseAnalysis(object):
    """ Contains methods for organizing responses by trial or by  visual stimulus flashes in a DataFrame.

    For trial responses, a segment of the dF/F trace for each cell is extracted for each trial in the trials records in a +/-4 seconds window (the 'trial_window') around the change time.
    The mean_response for each cell is taken in a 500ms window after the change time (the 'response_window').
    The trial_response_df also contains behavioral metadata from the trial records such as lick times, running speed, reward rate, and initial and change stimulus names.

    For stimulus flashes, the mean response is taken in a 500ms window after each stimulus presentation (the 'response_window') in the stimulus_table.
    The flash_response_df contains the mean response for every cell, for every stimulus flash.

    Parameters
    ----------
    dataset: VisualBehaviorOphysDataset instance
    overwrite_analysis_files: Boolean, if True will create and overwrite response analysis files.
    This can be used if new functionality is added to the ResponseAnalysis class to modify existing structures or make new ones.
    If False, will load existing analysis files from dataset.analysis_dir, or generate and save them if none exist.
    """

    def __init__(self, dataset, overwrite_analysis_files=False, use_events=False):
        self.dataset = dataset
        self.use_events = use_events
        self.overwrite_analysis_files = overwrite_analysis_files
        self.cache_dir = self.dataset.cache_dir
        self.experiment_id = self.dataset.experiment_id
        self.dataset.analysis_dir = self.get_analysis_dir()
        self.trial_window = rp.get_default_trial_response_params()['window_around_timepoint_seconds']
        self.flash_window = rp.get_default_stimulus_response_params()['window_around_timepoint_seconds']
        self.omitted_flash_window = rp.get_default_omission_response_params()['window_around_timepoint_seconds']
        self.response_window_duration = 0.25  # window, in seconds, over which to take the mean for a given trial or flash
        self.omission_response_window_duration = 0.75 # compute omission mean response and baseline response over 750ms
        self.stimulus_duration = 0.25  # self.dataset.task_parameters['stimulus_duration'].values[0]
        self.blank_duration = self.dataset.task_parameters['blank_duration_sec'][0]
        self.ophys_frame_rate = self.dataset.metadata['ophys_frame_rate']
        self.stimulus_frame_rate = self.dataset.metadata['stimulus_frame_rate']


    # def get_trial_response_df(self):
    #     self._trial_response_df = rp.get_trials_response_df(self.dataset, self.use_events)
    #     return self._trial_response_df
    #
    # trial_response_df = LazyLoadable('_trial_response_df', get_trial_response_df)
    #
    # def get_flash_response_df(self):
    #     self._flash_response_df = rp.get_stimulus_response_df(self.dataset, self.use_events)
    #     return self._flash_response_df
    #
    # flash_response_df = LazyLoadable('_flash_response_df', get_flash_response_df)
    #
    # def get_omitted_flash_response_df(self):
    #     self._omitted_flash_response_df = rp.get_omission_response_df(self.dataset, self.use_events)
    #     return self._omitted_flash_response_df
    #
    # omitted_flash_response_df = LazyLoadable('_omitted_flash_response_df', get_omitted_flash_response_df)

    def get_analysis_folder(self):
        candidates = [file for file in os.listdir(self.dataset.cache_dir) if str(self.dataset.experiment_id) in file]
        if len(candidates) == 1:
            self._analysis_folder = candidates[0]
        elif len(candidates) < 1:
            raise OSError(
                'unable to locate analysis folder for experiment {} in {}'.format(self.dataset.experiment_id, self.dataset.cache_dir))
        elif len(candidates) > 1:
            raise OSError('{} contains multiple possible analysis folders: {}'.format(self.dataset.cache_dir, candidates))

        return self._analysis_folder

    analysis_folder = LazyLoadable('_analysis_folder', get_analysis_folder)

    def get_analysis_dir(self):
        self._analysis_dir = os.path.join(self.dataset.cache_dir, self.analysis_folder)
        return self._analysis_dir

    analysis_dir = LazyLoadable('_analysis_dir', get_analysis_dir)

    def get_response_df_path(self, df_name):
        if self.use_events:
            if 'response' in df_name:
                path = os.path.join(self.dataset.analysis_dir, df_name+'_events.h5')
            else:
                path = os.path.join(self.dataset.analysis_dir, df_name+'.h5')
        else:
            path = os.path.join(self.dataset.analysis_dir, df_name+'.h5')
        return path

    def save_response_df(self, df, df_name):
        print('saving', df_name)
        df.to_hdf(self.get_response_df_path(df_name), key='df')

    def get_df_for_df_name(self, df_name):
        if df_name == 'trials_response_df':
            df = rp.get_trials_response_df(self.dataset, self.use_events)
        elif df_name == 'stimulus_response_df':
            df = rp.get_stimulus_response_df(self.dataset, self.use_events)
        elif df_name == 'omission_response_df':
            df = rp.get_omission_response_df(self.dataset, self.use_events)
        elif df_name == 'trials_run_speed_df':
            df = rp.get_trials_run_speed_df(self.dataset)
        elif df_name == 'stimulus_run_speed_df':
            df = rp.get_stimulus_run_speed_df(self.dataset)
        elif df_name == 'stimulus_pupil_area_df':
            df = rp.get_stimulus_pupil_area_df(self.dataset)
        elif df_name == 'omission_run_speed_df':
            df = rp.get_omission_run_speed_df(self.dataset)
        elif df_name == 'omission_pupil_area_df':
            df = rp.get_omission_pupil_area_df(self.dataset)
        elif df_name == 'omission_licks_df':
            df = rp.get_omission_licks_df(self.dataset)
        return df

    def get_response_df(self, df_name='trials_response_df'):
        if self.overwrite_analysis_files:
            # delete old file or else it will keep growing in size
            import h5py
            file_path = self.get_response_df_path(df_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            df = self.get_df_for_df_name(df_name)
            self.save_response_df(df, df_name)
        else:
            if os.path.exists(self.get_response_df_path(df_name)):
                print('loading', df_name)
                df = pd.read_hdf(self.get_response_df_path(df_name), key='df')
            else:
                df = self.get_df_for_df_name(df_name)
                self.save_response_df(df, df_name)
        if 'trials' in df_name:
            trials = self.dataset.trials
            trials = trials.rename(
                columns={'response': 'behavioral_response', 'response_type': 'behavioral_response_type',
                         'response_time': 'behavioral_response_time',
                         'response_latency': 'behavioral_response_latency'})
            df = df.merge(trials, right_on='trials_id', left_on='trials_id')
        elif ('stimulus' in df_name) or ('omission' in df_name):
            stimulus_presentations = self.dataset.extended_stimulus_presentations.copy()
            running_speed = self.dataset.running_speed_df
            response_params = rp.get_default_omission_response_params()
            stimulus_presentations = sp.add_window_running_speed(running_speed, stimulus_presentations, response_params)
            df = df.merge(stimulus_presentations, right_on = 'stimulus_presentations_id', left_on = 'stimulus_presentations_id')
        if ('response' in df_name):
            df['cell'] = [loading.get_cell_index_for_cell_specimen_id(self.dataset, cell_specimen_id) for
                         cell_specimen_id in df.cell_specimen_id.values]
        return df


    def get_trials_response_df(self):
        df_name = 'trials_response_df'
        df = self.get_response_df(df_name)
        self._trials_response_df = df
        return self._trials_response_df

    trials_response_df = LazyLoadable('_trials_response_df', get_trials_response_df)

    def get_stimulus_response_df(self):
        df_name = 'stimulus_response_df'
        df = self.get_response_df(df_name)
        self._stimulus_response_df = df
        return self._stimulus_response_df

    stimulus_response_df = LazyLoadable('_stimulus_response_df', get_stimulus_response_df)

    def get_omission_response_df(self):
        df_name = 'omission_response_df'
        df = self.get_response_df(df_name)
        self._omission_response_df = df
        return self._omission_response_df

    omission_response_df = LazyLoadable('_omission_response_df', get_omission_response_df)

    def get_trials_run_speed_df(self):
        df_name = 'trials_run_speed_df'
        df = self.get_response_df(df_name)
        self._trials_run_speed_df = df
        return self._trials_run_speed_df

    trials_run_speed_df = LazyLoadable('_trials_run_speed_df', get_trials_run_speed_df)

    def get_stimulus_run_speed_df(self):
        df_name = 'stimulus_run_speed_df'
        df = self.get_response_df(df_name)
        self._stimulus_run_speed_df = df
        return self._stimulus_run_speed_df

    stimulus_run_speed_df = LazyLoadable('_stimulus_run_speed_df', get_stimulus_run_speed_df)

    def get_omission_run_speed_df(self):
        df_name = 'omission_run_speed_df'
        df = self.get_response_df(df_name)
        self._omission_run_speed_df = df
        return self._omission_run_speed_df

    omission_run_speed_df = LazyLoadable('_omission_run_speed_df', get_omission_run_speed_df)

    def get_omission_pupil_area_df(self):
        df_name = 'omission_pupil_area_df'
        df = self.get_response_df(df_name)
        self._omission_pupil_area_df = df
        return self._omission_pupil_area_df

    omission_pupil_area_df = LazyLoadable('_omission_pupil_area_df', get_omission_pupil_area_df)

    def get_stimulus_run_speed_df(self):
        df_name = 'stimulus_run_speed_df'
        df = self.get_response_df(df_name)
        self._stimulus_run_speed_df = df
        return self._stimulus_run_speed_df

    stimulus_run_speed_df = LazyLoadable('_stimulus_run_speed_df', get_stimulus_run_speed_df)

    def get_stimulus_pupil_area_df(self):
        df_name = 'stimulus_pupil_area_df'
        df = self.get_response_df(df_name)
        self._stimulus_pupil_area_df = df
        return self._stimulus_pupil_area_df

    stimulus_pupil_area_df = LazyLoadable('_stimulus_pupil_area_df', get_stimulus_pupil_area_df)






