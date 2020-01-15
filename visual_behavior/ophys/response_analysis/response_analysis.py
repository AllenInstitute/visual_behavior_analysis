"""
Created on Sunday July 15 2018

@author: marinag
"""

from visual_behavior.ophys.response_analysis import utilities as ut
import visual_behavior.ophys.response_analysis.response_processing as rp

import os
import numpy as np
import pandas as pd
import itertools
import scipy as sp



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
        self.trial_window = rp.get_default_trial_response_params()['window_around_timepoint_seconds']
        self.flash_window = rp.get_default_stimulus_response_params()['window_around_timepoint_seconds']
        self.omitted_flash_window = rp.get_default_omission_response_params()['window_around_timepoint_seconds']
        self.response_window_duration = 0.25  # window, in seconds, over which to take the mean for a given trial or flash
        # self.response_window = [np.abs(self.trial_window[0]), np.abs(self.trial_window[
        #                         0]) + self.response_window_duration]  # time, in seconds, around change time to take the mean response
        # self.baseline_window = np.asarray(
        #     self.response_window) - self.response_window_duration  # time, in seconds, relative to change time to take baseline mean response
        self.omission_response_window_duration = 0.75 # compute omission mean response and baseline response over 750ms
        self.stimulus_duration = 0.25  # self.dataset.task_parameters['stimulus_duration'].values[0]
        self.blank_duration = self.dataset.task_parameters['blank_duration'].values[0]
        self.ophys_frame_rate = self.dataset.metadata['ophys_frame_rate'].values[0]
        self.stimulus_frame_rate = self.dataset.metadata['stimulus_frame_rate'].values[0]


    # def get_trial_response_df_path(self):
    #     if self.use_events:
    #         path = os.path.join(self.dataset.analysis_dir, 'trial_response_df_events.h5')
    #     else:
    #         path = os.path.join(self.dataset.analysis_dir, 'trial_response_df.h5')
    #     return path
    #
    # def save_trial_response_df(self, trial_response_df):
    #     print('saving trial response dataframe')
    #     trial_response_df.to_hdf(self.get_trial_response_df_path(), key='df')
    #
    # def get_trial_response_df(self):
    #     if self.overwrite_analysis_files:
    #         print('overwriting analysis files')
    #         import h5py
    #         file_path = self.get_trial_response_df_path()
    #         if os.path.exists(file_path):
    #             os.remove(file_path)
    #         # trial_response_df = rp.trial_response_df(rp.trial_response_xr(self.dataset))
    #         trial_response_df = rp.get_trial_response_df(self.dataset)
    #         self.save_trial_response_df(trial_response_df)
    #     else:
    #         if os.path.exists(self.get_trial_response_df_path()):
    #             print('loading trial response dataframe')
    #             trial_response_df = pd.read_hdf(self.get_trial_response_df_path(), key='df')
    #         else:
    #             # trial_response_df = rp.trial_response_df(rp.trial_response_xr(self.dataset))
    #             trial_response_df = rp.get_trial_response_df(self.dataset)
    #             self.save_trial_response_df(trial_response_df)
    #     trials = self.dataset.trials
    #     trials = trials.rename(columns={'response': 'behavioral_response', 'response_type': 'behavioral_response_type',
    #                            'response_time': 'behavioral_response_time', 'response_latency': 'behavioral_response_latency'})
    #     trial_response_df = trial_response_df.merge(trials, right_on='trials_id', left_on='trials_id')
    #     trial_response_df.rename(columns={'dff_trace': 'trace', 'dff_trace_timestamps': 'trace_timestamps'},
    #                                   inplace=True)
    #     trial_response_df['cell'] = [self.dataset.get_cell_index_for_cell_specimen_id(cell_specimen_id)
    #                                               for cell_specimen_id in
    #                                               trial_response_df.cell_specimen_id.values]
    #     self._trial_response_df = ut.annotate_trial_response_df_with_pref_stim(trial_response_df)
    #     return self._trial_response_df
    #
    # trial_response_df = LazyLoadable('_trial_response_df', get_trial_response_df)
    #
    #
    # def get_flash_response_df_path(self):
    #     if self.use_events:
    #         path = os.path.join(self.dataset.analysis_dir, 'flash_response_df_events.h5')
    #     else:
    #         path = os.path.join(self.dataset.analysis_dir, 'flash_response_df.h5')
    #     return path
    #
    #
    # def save_flash_response_df(self, flash_response_df):
    #     print('saving flash response dataframe')
    #     flash_response_df.to_hdf(self.get_flash_response_df_path(), key='df')
    #
    #
    # def get_flash_response_df(self):
    #     if self.overwrite_analysis_files:
    #         # delete old file or else it will keep growing in size
    #         import h5py
    #         file_path = self.get_flash_response_df_path()
    #         if os.path.exists(file_path):
    #             os.remove(file_path)
    #         # flash_response_df = rp.stimulus_response_df(rp.stimulus_response_xr(self.dataset))
    #         flash_response_df = rp.get_stimulus_response_df(self.dataset)
    #         self.save_flash_response_df(flash_response_df)
    #     else:
    #         if os.path.exists(self.get_flash_response_df_path()):
    #             print('loading flash response dataframe')
    #             flash_response_df = pd.read_hdf(self.get_flash_response_df_path(), key='df')
    #         else:
    #             # flash_response_df = rp.stimulus_response_df(rp.stimulus_response_xr(self.dataset))
    #             flash_response_df = rp.get_stimulus_response_df(self.dataset)
    #             self.save_flash_response_df(flash_response_df)
    #     flash_response_df = flash_response_df.merge(self.dataset.extended_stimulus_presentations.reset_index(),
    #                                  right_on='stimulus_presentations_id',
    #                                  left_on='stimulus_presentations_id')
    #     flash_response_df['repeat'] = flash_response_df.index_within_block.values
    #     flash_response_df.rename(columns={'dff_trace': 'trace', 'dff_trace_timestamps': 'trace_timestamps'}, inplace=True)
    #     flash_response_df['cell'] = [self.dataset.get_cell_index_for_cell_specimen_id(cell_specimen_id) for cell_specimen_id in flash_response_df.cell_specimen_id.values]
    #     self._flash_response_df = flash_response_df
    #     return self._flash_response_df
    #
    # flash_response_df = LazyLoadable('_flash_response_df', get_flash_response_df)
    #
    # def get_omitted_flash_response_df_path(self):
    #     if self.use_events:
    #         path = os.path.join(self.dataset.analysis_dir, 'omitted_flash_response_df_events.h5')
    #     else:
    #         path = os.path.join(self.dataset.analysis_dir, 'omitted_flash_response_df.h5')
    #     return path
    #
    # def save_omitted_flash_response_df(self, omitted_flash_response_df):
    #     print('saving omitted flash response dataframe')
    #     omitted_flash_response_df.to_hdf(self.get_omitted_flash_response_df_path(), key='df')
    #
    # def get_omitted_flash_response_df(self):
    #     if self.overwrite_analysis_files:
    #         # delete old file or else it will keep growing in size
    #         import h5py
    #         file_path = self.get_omitted_flash_response_df_path()
    #         if os.path.exists(file_path):
    #             os.remove(file_path)
    #         # omitted_flash_response_df = rp.omission_response_df(rp.omission_response_xr(self.dataset))
    #         omitted_flash_response_df = rp.get_omission_flash_response_df(self.dataset)
    #         self.save_omitted_flash_response_df(omitted_flash_response_df)
    #     else:
    #         if os.path.exists(self.get_omitted_flash_response_df_path()):
    #             print('loading omitted flash response dataframe')
    #             omitted_flash_response_df = pd.read_hdf(self.get_omitted_flash_response_df_path(), key='df')
    #
    #         else:
    #             # omitted_flash_response_df = rp.omission_response_df(rp.omission_response_xr(self.dataset))
    #             omitted_flash_response_df = rp.get_omission_flash_response_df(self.dataset)
    #             self.save_omitted_flash_response_df(omitted_flash_response_df)
    #     omitted_flash_response_df = omitted_flash_response_df.merge(
    #         self.dataset.extended_stimulus_presentations[self.dataset.extended_stimulus_presentations.omitted == True],
    #         right_on='stimulus_presentations_id', left_on='stimulus_presentations_id')
    #     omitted_flash_response_df.rename(columns={'dff_trace': 'trace', 'dff_trace_timestamps': 'trace_timestamps'},
    #                                   inplace=True)
    #     omitted_flash_response_df['cell'] = [self.dataset.get_cell_index_for_cell_specimen_id(cell_specimen_id)
    #                                               for cell_specimen_id in omitted_flash_response_df.cell_specimen_id.values]
    #     self._omitted_flash_response_df = omitted_flash_response_df
    #     return self._omitted_flash_response_df
    #
    # omitted_flash_response_df = LazyLoadable('_omitted_flash_response_df', get_omitted_flash_response_df)

    def get_trial_response_df(self):
        self._trial_response_df = rp.get_trials_response_df(self.dataset, self.use_events)
        return self._trial_response_df

    trial_response_df = LazyLoadable('_trial_response_df', get_trial_response_df)

    def get_flash_response_df(self):
        self._flash_response_df = rp.get_stimulus_response_df(self.dataset, self.use_events)
        return self._flash_response_df

    flash_response_df = LazyLoadable('_flash_response_df', get_flash_response_df)

    def get_omitted_flash_response_df(self):
        self._omitted_flash_response_df = rp.get_omission_response_df(self.dataset, self.use_events)
        return self._omitted_flash_response_df

    omitted_flash_response_df = LazyLoadable('_omitted_flash_response_df', get_omitted_flash_response_df)

    def get_response_df_path(self, df_name):
        if self.use_events:
            path = os.path.join(self.dataset.analysis_dir, df_name + '_events.h5')
        else:
            path = os.path.join(self.dataset.analysis_dir, df_name+'.h5')
        return path

    def save_response_df(self, df, df_name):
        print('saving', df_name)
        df.to_hdf(self.get_response_df_path(df_name), key='df')

    def get_df_for_df_name(self, df_name):
        if df_name == 'trials_response_df':
            df = rp.get_trial_response_df(self.dataset, self.use_events)
        elif df_name == 'stimulus_response_df':
            df = rp.get_stimulus_response_df(self.dataset, self.use_events)
        elif df_name == 'omission_response_df':
            df = rp.get_omission_response_df(self.dataset, self.use_events)
        elif df_name == 'trial_run_speed_df':
            df = rp.get_trial_run_speed_df(self.dataset)
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
            df = df.merge(self.dataset.extended_stimulus_presentations,
                          right_on='stimulus_presentations_id', left_on='stimulus_presentations_id')
        if ('response' in df_name):
            df['cell'] = [self.dataset.get_cell_index_for_cell_specimen_id(cell_specimen_id) for
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

    def get_trial_run_speed_df(self):
        df_name = 'trial_run_speed_df'
        df = self.get_response_df(df_name)
        self._trial_run_speed_df = df
        return self._trial_run_speed_df

    trial_run_speed_df = LazyLoadable('_trial_run_speed_df', get_trial_run_speed_df)

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













    # def compute_pairwise_correlations(self):
    #     fdf = self.flash_response_df.copy()
    #     if 'omitted' in fdf.keys():
    #         fdf = fdf[fdf.omitted == False].copy()
    #     # compute correlations independently for each repeat of a given image after a change
    #     # repeat = 1 is the first flash after change, repeat = 5 is the 5th flash after a change
    #     fdf['engaged'] = [True if reward_rate > 2 else False for reward_rate in fdf.reward_rate.values]
    #     s_corr_data = []
    #     n_corr_data = []
    #     for repeat in [1, 5, 10, 15]:
    #         for engaged in fdf.engaged.unique():
    #             # print('repeat', repeat, 'engaged', engaged)
    #             tmp_fdf = fdf[(fdf.repeat == repeat) & (fdf.engaged == engaged)]
    #             # create a df with the trial averaged response for each image across flash repeat number
    #             mfdf = ut.get_mean_df(tmp_fdf, conditions=['cell_specimen_id', 'image_name', 'repeat', 'engaged'],
    #                                   flashes=True)
    #
    #             # signal correlations
    #             # create table with the trial averaged response of each cell to each image
    #             # print('getting signal correlations')
    #             signal_responses = mfdf.pivot_table(
    #                 index=['cell_specimen_id', 'repeat', 'engaged'],
    #                 columns='image_name',
    #                 values='mean_response')
    #             # loop through pairs to compute signal correlations
    #             for cell1_data, cell2_data in itertools.combinations(signal_responses.iterrows(), 2):
    #                 #     print cell1_data, cell2_data
    #                 (cell_specimen_id_1, repeat_number_1, engaged_1), image_responses_1 = cell1_data
    #                 (cell_specimen_id_2, repeat_number_2, engaged_2), image_responses_2 = cell2_data
    #                 try:
    #                     # correlation between 2 cells tuning curves
    #                     scorr = sp.stats.pearsonr(image_responses_1[~np.isnan(image_responses_1)],
    #                                               image_responses_2[~np.isnan(image_responses_2)])[0]
    #                 except ValueError:
    #                     scorr = np.nan
    #                 s_corr_data.append(
    #                     dict(repeat=repeat_number_1, engaged=engaged_1, cell1=cell_specimen_id_1,
    #                          cell2=cell_specimen_id_2, signal_correlation=scorr))
    #             # noise correlations
    #             # get trial average response for each cell to subtract from each trial's response
    #             # print('getting noise correlations')
    #             mfdf['trial_average'] = mfdf.mean_response.values
    #             mfdf2 = mfdf[['cell_specimen_id', 'image_name', 'repeat', 'engaged', 'trial_average']]
    #             # add trial average column to flash response df
    #             ndf = mfdf2.merge(tmp_fdf, on=['cell_specimen_id', 'image_name', 'repeat', 'engaged'])
    #             # subtract the trial average from the mean response on each individual flash
    #             ndf['noise_response'] = ndf.mean_response - ndf.trial_average
    #             # create table with trial average subtracted response for each flash
    #             noise_responses = ndf.pivot_table(
    #                 index=['cell_specimen_id', 'repeat', 'engaged'],
    #                 columns='flash_number',
    #                 values='noise_response')
    #             # compute noise correlations
    #             for cell1_data, cell2_data in itertools.combinations(noise_responses.iterrows(), 2):
    #                 (cell_specimen_id_1, repeat_number_1, engaged_1), trial_responses_1 = cell1_data
    #                 (cell_specimen_id_2, repeat_number_2, engaged_2), trial_responses_2 = cell2_data
    #                 try:
    #                     ncorr = sp.stats.pearsonr(trial_responses_1[~np.isnan(trial_responses_1)],
    #                                               trial_responses_2[~np.isnan(trial_responses_2)])[0]
    #                 except ValueError:
    #                     ncorr = np.nan
    #                 n_corr_data.append(
    #                     dict(repeat=repeat_number_1, engaged=engaged_1, cell1=cell_specimen_id_1,
    #                          cell2=cell_specimen_id_2, noise_correlation=ncorr))
    #
    #     # create dataframes from signal and noise correlation dictionaries and merge
    #     cdf1 = pd.DataFrame(s_corr_data)
    #     cdf2 = pd.DataFrame(n_corr_data)
    #     pairwise_correlations_df = cdf1.merge(cdf2, on=['cell1', 'cell2', 'repeat', 'engaged'])
    #     pairwise_correlations_df['experiment_id'] = int(self.dataset.experiment_id)
    #     return pairwise_correlations_df
    #
    # def get_pairwise_correlations_path(self):
    #     if self.use_events:
    #         path = os.path.join(self.dataset.analysis_dir, 'pairwise_correlations_events.h5')
    #     else:
    #         path = os.path.join(self.dataset.analysis_dir, 'pairwise_correlations.h5')
    #     return path
    #
    # def save_pairwise_correlations_df(self, pairwise_correlations_df):
    #     print('saving pairwise correlations dataframe')
    #     pairwise_correlations_df.to_hdf(self.get_pairwise_correlations_path(), key='df')
    #
    # def get_pairwise_correlations_df(self):
    #     # if os.path.exists(self.get_pairwise_correlations_path()):
    #     #     print('loading pairwise correlations dataframe')
    #     #     self.pairwise_correlations_df = pd.read_hdf(self.get_pairwise_correlations_path(), key='df')
    #     # else:
    #     print('generating pairwise correlations dataframe')
    #     self.pairwise_correlations_df = self.compute_pairwise_correlations()
    #     self.save_pairwise_correlations_df(self.pairwise_correlations_df)
    #     return self.pairwise_correlations_df
