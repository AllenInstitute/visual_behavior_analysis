"""
Created on Sunday July 15 2018

@author: marinag
"""

from visual_behavior.ophys.response_analysis import utilities as ut

import os
import numpy as np
import pandas as pd
import itertools
import scipy as sp

import logging

logger = logging.getLogger(__name__)


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
        self.trial_window = [-4, 4]  # time, in seconds, around change time to extract portion of cell trace
        self.flash_window = [-0.5,
                             0.75]  # time, in seconds, around stimulus flash onset time to extract portion of cell trace
        self.response_window_duration = 0.5  # window, in seconds, over which to take the mean for a given trial or flash
        self.response_window = [np.abs(self.trial_window[0]), np.abs(self.trial_window[
                                0]) + self.response_window_duration]  # time, in seconds, around change time to take the mean response
        self.baseline_window = np.asarray(
            self.response_window) - self.response_window_duration  # time, in seconds, relative to change time to take baseline mean response
        self.stimulus_duration = 0.25  # self.dataset.task_parameters['stimulus_duration'].values[0]
        self.blank_duration = self.dataset.task_parameters['blank_duration'].values[0]
        self.ophys_frame_rate = self.dataset.metadata['ophys_frame_rate'].values[0]
        self.stimulus_frame_rate = self.dataset.metadata['stimulus_frame_rate'].values[0]

        self.get_trial_response_df()
        self.get_flash_response_df()

    def get_trial_response_df_path(self):
        if self.use_events:
            path = os.path.join(self.dataset.analysis_dir, 'trial_response_df_events.h5')
        else:
            path = os.path.join(self.dataset.analysis_dir, 'trial_response_df.h5')
        return path

    def generate_trial_response_df(self):
        logger.info('generating trial response dataframe')
        running_speed = self.dataset.running_speed.running_speed.values
        df_list = []
        for cell_index in self.dataset.cell_indices:
            if self.use_events:
                cell_trace = self.dataset.events[cell_index, :].copy()
            else:
                cell_trace = self.dataset.dff_traces[cell_index, :].copy()
            for trial in self.dataset.trials.trial.values[:-1]:  # ignore last trial to avoid truncated traces
                cell_specimen_id = self.dataset.get_cell_specimen_id_for_cell_index(cell_index)
                change_time = self.dataset.trials[self.dataset.trials.trial == trial].change_time.values[0]
                # get dF/F trace & metrics
                trace, timestamps = ut.get_trace_around_timepoint(change_time, cell_trace,
                                                                  self.dataset.timestamps_ophys,
                                                                  self.trial_window, self.ophys_frame_rate)
                mean_response = ut.get_mean_in_window(trace, self.response_window, self.ophys_frame_rate,
                                                      self.use_events)
                baseline_response = ut.get_mean_in_window(trace, self.baseline_window, self.ophys_frame_rate,
                                                          self.use_events)
                p_value = ut.get_p_val(trace, self.response_window, self.ophys_frame_rate)
                sd_over_baseline = ut.get_sd_over_baseline(trace, self.response_window, self.baseline_window,
                                                           self.ophys_frame_rate)
                n_events = ut.get_n_nonzero_in_window(trace, self.response_window, self.ophys_frame_rate)
                # this is redundant because its the same for every cell. do we want to keep this?
                running_speed_trace, running_speed_timestamps = ut.get_trace_around_timepoint(change_time,
                                                                                              running_speed,
                                                                                              self.dataset.timestamps_stimulus,
                                                                                              self.trial_window,
                                                                                              self.stimulus_frame_rate)
                mean_running_speed = ut.get_mean_in_window(running_speed_trace, self.response_window,
                                                           self.stimulus_frame_rate)
                df_list.append(
                    [trial, cell_index, cell_specimen_id, trace, timestamps, mean_response, baseline_response, n_events,
                     p_value, sd_over_baseline, running_speed_trace, running_speed_timestamps,
                     mean_running_speed, self.dataset.experiment_id])

        columns = ['trial', 'cell', 'cell_specimen_id', 'trace', 'timestamps', 'mean_response', 'baseline_response',
                   'n_events', 'p_value', 'sd_over_baseline', 'running_speed_trace', 'running_speed_timestamps',
                   'mean_running_speed', 'experiment_id']
        trial_response_df = pd.DataFrame(df_list, columns=columns)
        trial_metadata = self.dataset.trials
        trial_metadata = trial_metadata.rename(columns={'response': 'behavioral_response'})
        trial_metadata = trial_metadata.rename(columns={'response_type': 'behavioral_response_type'})
        trial_metadata = trial_metadata.rename(columns={'response_time': 'behavioral_response_time'})
        trial_metadata = trial_metadata.rename(columns={'response_latency': 'behavioral_response_latency'})
        trial_response_df = trial_response_df.merge(trial_metadata, on='trial')
        trial_response_df = ut.annotate_trial_response_df_with_pref_stim(trial_response_df)
        return trial_response_df

    def save_trial_response_df(self, trial_response_df):
        print('saving trial response dataframe')
        trial_response_df.to_hdf(self.get_trial_response_df_path(), key='df', format='fixed')

    def get_trial_response_df(self):
        if self.overwrite_analysis_files:
            print('overwriting analysis files')
            self.trial_response_df = self.generate_trial_response_df()
            self.save_trial_response_df(self.trial_response_df)
        else:
            if os.path.exists(self.get_trial_response_df_path()):
                print('loading trial response dataframe')
                self.trial_response_df = pd.read_hdf(self.get_trial_response_df_path(), key='df', format='fixed')
            else:
                self.trial_response_df = self.generate_trial_response_df()
                self.save_trial_response_df(self.trial_response_df)
        return self.trial_response_df

    def get_flash_response_df_path(self):
        if self.use_events:
            path = os.path.join(self.dataset.analysis_dir, 'flash_response_df_events.h5')
        else:
            path = os.path.join(self.dataset.analysis_dir, 'flash_response_df.h5')
        return path

    def generate_flash_response_df(self):
        stimulus_table = ut.annotate_flashes_with_reward_rate(self.dataset)
        row = []
        for cell in self.dataset.cell_indices:
            cell = int(cell)
            cell_specimen_id = int(self.dataset.get_cell_specimen_id_for_cell_index(cell))
            if self.use_events:
                cell_trace = self.dataset.events[cell, :].copy()
            else:
                cell_trace = self.dataset.dff_traces[cell, :].copy()
            for flash in stimulus_table.flash_number:
                flash = int(flash)
                flash_data = stimulus_table[stimulus_table.flash_number == flash]
                flash_time = flash_data.start_time.values[0]
                image_name = flash_data.image_name.values[0]
                # flash_window = [-self.response_window_duration, self.response_window_duration]
                flash_window = self.flash_window
                trace, timestamps = ut.get_trace_around_timepoint(flash_time, cell_trace,
                                                                  self.dataset.timestamps_ophys,
                                                                  flash_window, self.ophys_frame_rate)
                # response_window = [self.response_window_duration, self.response_window_duration * 2]
                response_window = [np.abs(flash_window[0]), np.abs(flash_window[
                    0]) + self.response_window_duration]  # time, in seconds, around flash time to take the mean response
                baseline_window = [np.abs(flash_window[0]) - self.response_window_duration, (np.abs(flash_window[0]))]
                # baseline_window = [np.abs(flash_window[0]), np.abs(flash_window[0]) - self.response_window_duration]
                p_value = ut.get_p_val(trace, response_window, self.ophys_frame_rate)
                sd_over_baseline = ut.get_sd_over_baseline(cell_trace, flash_window,
                                                           baseline_window, self.ophys_frame_rate)
                mean_response = ut.get_mean_in_window(trace, response_window, self.ophys_frame_rate, self.use_events)
                baseline_response = ut.get_mean_in_window(trace, baseline_window,
                                                          self.ophys_frame_rate, self.use_events)
                n_events = ut.get_n_nonzero_in_window(trace, response_window, self.ophys_frame_rate)
                reward_rate = flash_data.reward_rate.values[0]

                row.append([cell, cell_specimen_id, flash, flash_time, image_name, trace, timestamps, mean_response,
                            baseline_response, n_events, p_value, sd_over_baseline, reward_rate,
                            self.dataset.experiment_id])

        flash_response_df = pd.DataFrame(data=row,
                                         columns=['cell', 'cell_specimen_id', 'flash_number', 'start_time',
                                                  'image_name', 'trace', 'timestamps', 'mean_response',
                                                  'baseline_response', 'n_events', 'p_value', 'sd_over_baseline',
                                                  'reward_rate', 'experiment_id'])
        flash_response_df = ut.annotate_flash_response_df_with_pref_stim(flash_response_df)
        flash_response_df = ut.add_repeat_number_to_flash_response_df(flash_response_df, stimulus_table)
        flash_response_df = ut.add_image_block_to_flash_response_df(flash_response_df, stimulus_table)

        flash_response_df['change_time'] = flash_response_df.start_time.values
        flash_response_df = pd.merge(flash_response_df, self.dataset.all_trials[['change_time', 'trial_type']],
                                     on='change_time', how='outer')

        # tmp = self.trial_response_df[self.trial_response_df.cell == 0]
        # flash_response_df = pd.merge(flash_response_df, tmp[['change_time',
        #                             'lick_times', 'reward_times']], on='change_time', how='outer')
        flash_response_df = flash_response_df[
            (flash_response_df.flash_number > 10) & (flash_response_df.trial_type != 'autorewarded')]
        flash_response_df['engaged'] = [True if rw > 2 else False for rw in flash_response_df.reward_rate.values]

        return flash_response_df

    def save_flash_response_df(self, flash_response_df):
        print('saving flash response dataframe')
        flash_response_df.to_hdf(self.get_flash_response_df_path(), key='df', format='fixed')

    def get_flash_response_df(self):
        if self.overwrite_analysis_files:
            self.flash_response_df = self.generate_flash_response_df()
            self.save_flash_response_df(self.flash_response_df)
        else:
            if os.path.exists(self.get_flash_response_df_path()):
                print('loading flash response dataframe')
                self.flash_response_df = pd.read_hdf(self.get_flash_response_df_path(), key='df', format='fixed')
            else:
                self.flash_response_df = self.generate_flash_response_df()
                self.save_flash_response_df(self.flash_response_df)
        return self.flash_response_df

    def compute_pairwise_correlations(self):
        fdf = self.flash_response_df.copy()
        # compute correlations independently for each repeat of a given image after a change
        # repeat = 1 is the first flash after change, repeat = 5 is the 5th flash after a change
        s_corr_data = []
        n_corr_data = []
        for repeat in [1, 5, 10, 15]:
            tmp_fdf = fdf[fdf.repeat == repeat]
            # create a df with the trial averaged response for each image across flash repeat number
            mfdf = ut.get_mean_df(tmp_fdf, conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)

            # signal correlations
            # create table with the trial averaged response of each cell to each image
            signal_responses = mfdf.pivot_table(
                index=['cell_specimen_id', 'repeat'],
                columns='image_name',
                values='mean_response')
            # loop through pairs to compute signal correlations
            for cell1_data, cell2_data in itertools.combinations(signal_responses.iterrows(), 2):
                (cell_specimen_id_1, repeat_number_1), image_responses_1 = cell1_data
                (cell_specimen_id_2, repeat_number_2), image_responses_2 = cell2_data
                try:
                    # correlation between 2 cells tuning curves
                    scorr = sp.stats.pearsonr(image_responses_1[~np.isnan(image_responses_1)],
                                              image_responses_2[~np.isnan(image_responses_2)])[0]
                except ValueError:
                    scorr = np.nan
                s_corr_data.append(
                    dict(repeat=repeat_number_1, cell1=cell_specimen_id_1,
                         cell2=cell_specimen_id_2, signal_correlation=scorr))

            # noise correlations
            # get trial average response for each cell to subtract from each trial's response
            mfdf['trial_average'] = mfdf.mean_response.values
            mfdf2 = mfdf[['cell_specimen_id', 'image_name', 'repeat', 'trial_average']]
            # add trial average column to flash response df
            ndf = mfdf2.merge(tmp_fdf, on=['cell_specimen_id', 'image_name', 'repeat'])
            # subtract the trial average from the mean response on each individual flash
            ndf['noise_response'] = ndf.mean_response - ndf.trial_average
            # create table with trial average subtracted response for each flash
            noise_responses = ndf.pivot_table(
                index=['cell_specimen_id', 'repeat'],
                columns='flash_number',
                values='noise_response')
            # compute noise correlations
            for cell1_data, cell2_data in itertools.combinations(noise_responses.iterrows(), 2):
                (cell_specimen_id_1, repeat_number_1), trial_responses_1 = cell1_data
                (cell_specimen_id_2, repeat_number_2), trial_responses_2 = cell2_data
                try:
                    ncorr = sp.stats.pearsonr(trial_responses_1[~np.isnan(trial_responses_1)],
                                              trial_responses_2[~np.isnan(trial_responses_2)])[0]
                except ValueError:
                    ncorr = np.nan
                n_corr_data.append(
                    dict(repeat=repeat_number_1, cell1=cell_specimen_id_1,
                         cell2=cell_specimen_id_2, noise_correlation=ncorr))

        # create dataframes from signal and noise correlation dictionaries and merge
        cdf1 = pd.DataFrame(s_corr_data)
        cdf2 = pd.DataFrame(n_corr_data)
        pairwise_correlations_df = cdf1.merge(cdf2, on=['cell1', 'cell2', 'repeat'])
        pairwise_correlations_df['experiment_id'] = int(self.dataset.experiment_id)

        return pairwise_correlations_df

    def get_pairwise_correlations_path(self):
        if self.use_events:
            path = os.path.join(self.dataset.analysis_dir, 'pairwise_correlations_events.h5')
        else:
            path = os.path.join(self.dataset.analysis_dir, 'pairwise_correlations.h5')
        return path

    def save_pairwise_correlations_df(self, pairwise_correlations_df):
        print('saving pairwise correlations dataframe')
        pairwise_correlations_df.to_hdf(self.get_pairwise_correlations_path(), key='df', format='fixed')

    def get_pairwise_correlations_df(self):
        if os.path.exists(self.get_pairwise_correlations_path()):
            print('loading pairwise correlations dataframe')
            self.pairwise_correlations_df = pd.read_hdf(self.get_pairwise_correlations_path(), key='df', format='fixed')
        else:
            print('generating pairwise correlations dataframe')
            self.pairwise_correlations_df = self.compute_pairwise_correlations()
            self.save_pairwise_correlations_df(self.pairwise_correlations_df)
        return self.pairwise_correlations_df
