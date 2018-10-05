"""
Created on Sunday July 15 2018

@author: marinag
"""

from visual_behavior.ophys.response_analysis import utilities as ut

import os
import numpy as np
import pandas as pd

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

    def __init__(self, dataset, overwrite_analysis_files=False):
        self.dataset = dataset
        self.overwrite_analysis_files = overwrite_analysis_files
        self.trial_window = [-4, 4]  # time, in seconds, around change time to extract portion of cell trace
        self.response_window_duration = 0.5  # window, in seconds, over which to take the mean for a given trial or flash
        self.response_window = [np.abs(self.trial_window[0]), np.abs(self.trial_window[0]) + self.response_window_duration]  # time, in seconds, around change time to take the mean response
        self.baseline_window = np.asarray(
            self.response_window) - self.response_window_duration  # time, in seconds, relative to change time to take baseline mean response
        self.stimulus_duration = self.dataset.task_parameters['stimulus_duration'].values[0]
        self.blank_duration = self.dataset.task_parameters['blank_duration'].values[0]
        self.ophys_frame_rate = self.dataset.metadata['ophys_frame_rate'].values[0]
        self.stimulus_frame_rate = self.dataset.metadata['stimulus_frame_rate'].values[0]

        self.get_trial_response_df()
        self.get_flash_response_df()

    def get_trial_response_df_path(self):
        path = os.path.join(self.dataset.analysis_dir, 'trial_response_df.h5')
        return path

    def generate_trial_response_df(self):
        logger.info('generating trial response dataframe')
        running_speed = self.dataset.running_speed.running_speed.values
        df_list = []
        for cell_index in self.dataset.cell_indices:
            for trial in self.dataset.trials.trial.values[:-1]:  # ignore last trial to avoid truncated traces
                cell_specimen_id = self.dataset.get_cell_specimen_id_for_cell_index(cell_index)
                change_time = self.dataset.trials[self.dataset.trials.trial == trial].change_time.values[0]
                # get dF/F trace & metrics
                cell_trace = self.dataset.dff_traces[cell_index, :]
                trace, timestamps = ut.get_trace_around_timepoint(change_time, cell_trace,
                                                                  self.dataset.timestamps_ophys,
                                                                  self.trial_window, self.ophys_frame_rate)
                mean_response = ut.get_mean_in_window(trace, self.response_window, self.ophys_frame_rate)
                baseline_response = ut.get_mean_in_window(trace, self.baseline_window, self.ophys_frame_rate)
                p_value = ut.get_p_val(trace, self.response_window, self.ophys_frame_rate)
                sd_over_baseline = ut.get_sd_over_baseline(trace, self.response_window, self.baseline_window,
                                                           self.ophys_frame_rate)
                # get events and metrics
                if self.dataset.events is not None:
                    event_trace = self.dataset.events[cell_index, :]
                    events, timestamps = ut.get_trace_around_timepoint(change_time, event_trace,
                                                                       self.dataset.timestamps_ophys, self.trial_window,
                                                                       self.ophys_frame_rate)
                    mean_response_events = ut.get_mean_in_window(events, self.response_window, self.ophys_frame_rate)
                    baseline_response_events = ut.get_mean_in_window(events, self.baseline_window,
                                                                     self.ophys_frame_rate)
                    p_value_events = ut.get_p_val(events, self.response_window, self.ophys_frame_rate)
                    sd_over_baseline_events = ut.get_sd_over_baseline(events, self.response_window,
                                                                      self.baseline_window, self.ophys_frame_rate)
                else:
                    events = None
                    mean_response_events = None
                    baseline_response_events = None
                    p_value_events = None
                    sd_over_baseline_events = None

                # this is redundant because its the same for every cell. do we want to keep this?
                running_speed_trace, running_speed_timestamps = ut.get_trace_around_timepoint(change_time,
                                                                                              running_speed,
                                                                                              self.dataset.timestamps_stimulus,
                                                                                              self.trial_window,
                                                                                              self.stimulus_frame_rate)
                mean_running_speed = ut.get_mean_in_window(running_speed_trace, self.response_window,
                                                           self.stimulus_frame_rate)

                df_list.append(
                    [trial, cell_index, cell_specimen_id, trace, timestamps, mean_response, baseline_response,
                     p_value, sd_over_baseline,
                     events, mean_response_events, baseline_response_events, p_value_events, sd_over_baseline_events,
                     running_speed_trace, running_speed_timestamps,
                     mean_running_speed, self.dataset.experiment_id])

        columns = ['trial', 'cell', 'cell_specimen_id', 'trace', 'timestamps', 'mean_response', 'baseline_response',
                   'p_value', 'sd_over_baseline',
                   'events', 'mean_response_events', 'baseline_response_events', 'p_value_events',
                   'sd_over_baseline_events',
                   'running_speed_trace', 'running_speed_timestamps',
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
        logger.info('saving trial response dataframe')
        trial_response_df.to_hdf(self.get_trial_response_df_path(), key='df', format='fixed')

    def get_trial_response_df(self):
        if self.overwrite_analysis_files:
            logger.info('overwriting analysis files')
            self.trial_response_df = self.generate_trial_response_df()
            self.save_trial_response_df(self.trial_response_df)
        else:
            if os.path.exists(self.get_trial_response_df_path()):
                logger.info('loading trial response dataframe')
                self.trial_response_df = pd.read_hdf(self.get_trial_response_df_path(), key='df', format='fixed')
            else:
                self.trial_response_df = self.generate_trial_response_df()
                self.save_trial_response_df(self.trial_response_df)
        return self.trial_response_df

    def get_flash_response_df_path(self):
        path = os.path.join(self.dataset.analysis_dir, 'flash_response_df.h5')
        return path

    def generate_flash_response_df(self):
        stimulus_table = ut.annotate_flashes_with_reward_rate(self.dataset)
        row = []
        for cell in range(self.dataset.dff_traces.shape[0]):
            cell_specimen_id = self.dataset.get_cell_specimen_id_for_cell_index(cell)
            for flash in stimulus_table.flash_number:
                flash_data = stimulus_table[stimulus_table.flash_number == flash]
                flash_time = flash_data.start_time.values[0]
                image_name = flash_data.image_name.values[0]
                trace_window = [-self.response_window_duration, self.response_window_duration]
                trace, timestamps = ut.get_trace_around_timepoint(flash_time, self.dataset.dff_traces[cell],
                                                                  self.dataset.timestamps_ophys,
                                                                  trace_window, self.ophys_frame_rate)
                response_window = [self.response_window_duration, self.response_window_duration * 2]
                p_value = ut.get_p_val(trace, response_window, self.ophys_frame_rate)
                mean_response = ut.get_mean_in_window(trace, response_window, self.ophys_frame_rate)
                reward_rate = flash_data.reward_rate.values[0]

                row.append([cell, cell_specimen_id, flash, flash_time, image_name, mean_response, p_value, reward_rate,
                            self.dataset.experiment_id])
        flash_response_df = pd.DataFrame(data=row,
                                         columns=['cell', 'cell_specimen_id', 'flash_number', 'start_time',
                                                  'image_name', 'mean_response',
                                                  'p_value', 'reward_rate', 'experiment_id'])
        flash_response_df = ut.annotate_flash_response_df_with_pref_stim(flash_response_df)
        flash_response_df = ut.add_repeat_number_to_flash_response_df(flash_response_df, stimulus_table)
        flash_response_df = ut.add_image_block_to_flash_response_df(flash_response_df, stimulus_table)
        return flash_response_df

    def save_flash_response_df(self, flash_response_df):
        logger.info('saving flash response dataframe')
        flash_response_df.to_hdf(self.get_flash_response_df_path(), key='df', format='fixed')

    def get_flash_response_df(self):
        if self.overwrite_analysis_files:
            self.flash_response_df = self.generate_flash_response_df()
            self.save_flash_response_df(self.flash_response_df)
        else:
            if os.path.exists(self.get_flash_response_df_path()):
                logger.info('loading flash response dataframe')
                self.flash_response_df = pd.read_hdf(self.get_flash_response_df_path(), key='df', format='fixed')
            else:
                self.flash_response_df = self.generate_flash_response_df()
                self.save_flash_response_df(self.flash_response_df)
        return self.flash_response_df
