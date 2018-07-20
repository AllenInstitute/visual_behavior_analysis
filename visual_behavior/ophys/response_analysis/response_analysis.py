import os
import numpy as np
import pandas as pd

from visual_behavior.ophys.response_analysis import utilities as ut

class ResponseAnalysis(object):
    """ Contains methods for organizing responses by trial or by visual stimulus flash in a DataFrame,
    with a portion of the cell trace and running around the change time for every change trial, for every cell.
    Response DataFrame also contains behavioral metadata such as lick times, running, reward rate, and stimulus.

    Parameters
    ----------
    dataset: VisualBehaviorOphysDataset instance
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.trial_window = [-4, 4]  # time, in seconds, around change time to extract portion of cell trace
        self.response_window_duration = 0.5  # window, in seconds, over which to take the mean for a given trial or flash
        self.response_window = [np.abs(self.trial_window[0]), np.abs(self.trial_window[
                                                                         0]) + self.response_window_duration]  # time, in seconds, around change time to take the mean response
        self.baseline_window = np.asarray(
            self.response_window) - self.response_window_duration  # time, in seconds, relative to change time to take baseline mean response
        self.stimulus_duration = self.dataset.task_parameters['stimulus_duration'].values[0]
        self.blank_duration = self.dataset.task_parameters['blank_duration'].values[0]
        self.ophys_frame_rate = self.dataset.metadata['ophys_frame_rate'].values[0]
        self.stimulus_frame_rate = self.dataset.metadata['stimulus_frame_rate'].values[0]

        self.get_trial_response_df()


    def get_trial_response_df_path(self):
        path = os.path.join(self.dataset.analysis_dir, 'trial_response_df.h5')
        return path

    def get_trial_response_df(self):
        if os.path.exists(self.get_trial_response_df_path()):
            print 'loading trial response dataframe'
            self.trial_response_df = pd.read_hdf(self.get_trial_response_df_path(), key='df', format='fixed')
        else:
            self.trial_response_df = self.generate_trial_response_df()
            self.save_response_df(self.trial_response_df)
        return self.trial_response_df

    def generate_trial_response_df(self):
        print 'generating trial response dataframe'
        # frames_in_trial_window = np.int(self.trial_window[1] - self.trial_window[0]) * self.ophys_frame_rate
        # frames_in_run_window = np.int(self.trial_window[1] - self.trial_window[0]) * self.stimulus_frame_rate
        running_speed = self.dataset.running.speed.values
        df_list = []
        for cell_index in self.dataset.cell_indices:
            for trial in self.dataset.trials.trial.values:
                cell_specimen_id = self.dataset.get_cell_specimen_id_for_cell_index(cell_index)
                cell_trace = self.dataset.dff_traces[cell_index, :]
                change_time = self.dataset.trials[self.dataset.trials.trial == trial].change_time.values[0]

                trace, timestamps = ut.get_trace_around_timepoint(change_time, cell_trace, self.dataset.timestamps_ophys,
                                                               self.trial_window, self.ophys_frame_rate)
                mean_response = ut.get_mean_in_window(trace, self.response_window, self.ophys_frame_rate)
                baseline_response = ut.get_mean_in_window(trace, self.baseline_window, self.ophys_frame_rate)
                p_value = ut.get_p_val(trace, self.response_window, self.ophys_frame_rate)
                sd_over_baseline = ut.get_sd_over_baseline(trace, self.response_window, self.baseline_window,
                                                        self.ophys_frame_rate)

                # this is redundant because its the same for every cell. do we want to keep this?
                running_speed_trace, running_speed_timestamps = ut.get_trace_around_timepoint(change_time,
                                                                                           running_speed,
                                                                                           self.dataset.timestamps_stimulus,
                                                                                           self.trial_window,
                                                                                           self.stimulus_frame_rate)
                mean_running_speed = ut.get_mean_in_window(running_speed_trace, self.response_window, self.stimulus_frame_rate)

                df_list.append([trial, cell_index, cell_specimen_id, trace, timestamps, mean_response, baseline_response,
                                p_value, sd_over_baseline, running_speed_trace, running_speed_timestamps,
                                mean_running_speed])

        columns = ['trial', 'cell', 'cell_specimen_id', 'trace', 'timestamps', 'mean_response', 'baseline_response',
                   'p_value', 'sd_over_baseline', 'running_speed_trace', 'running_speed_timestamps', 'mean_running_speed']
        trial_response_df = pd.DataFrame(df_list, columns=columns)
        #     trial_response_df = df.merge(self.dataset.trials, on='trial')
        return trial_response_df

    def save_response_df(self, trial_response_df):
        print 'saving trial response dataframe'
        trial_response_df.to_hdf(self.get_trial_response_df_path(), key='df', format='fixed')
