"""
Created on Sunday July 15 2018

@author: marinag
"""


import os
import pandas as pd
import visual_behavior.ophys.response_analysis.response_processing as rp


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


def get_analysis_cache_dir():
    return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'


class ResponseAnalysis(object):
    """

    Contains methods for organizing cell responses by trials, stimulus presentations, or omissions in a DataFrame.

    For each trial, stimulus presentation, or omission, a segment of the dF/F trace (or events if available and use_events=True) is extracted for each cell aligned to the time of the relevant event (ex: at the change time for each trial).
    The duration of each trace segment is a window around the event time, with default values for trials = [-5,5], stimulus presentations = [-0.5,0.75], and omissions = [-5,5].
    The mean_response for each cell is taken in a 500ms window after the change time or stimulus onset time, or in a 750ms window after omission time.
    Response dataframes also include extensive metadata for each event condition, such as image name, lick times, reward rate, etc.

    To load a response dataframe, use the method get_response_df() by providing an accepted response df type.
    Available response df types include: ['trials_response_df', 'stimulus_response_df', 'omission_response_df',
                                                'trials_run_speed_df', 'stimulus_run_speed_df', 'omission_run_speed_df',
                                                'trials_pupil_area_df', 'stimulus_pupil_area_df', 'omission_pupil_area_df']

    Parameters
    ----------
    dataset {object} -- VisualBehaviorOphysDataset or BehaviorOphysDataset class instance

    analysis_cache_dir {str} --  directory to save and load analysis files from. If None, defaults to:
        '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'

    load_from_cache {Boolean} -- if True, will load pre-existing response dataframes from analysis_cache_dir. If response dataframes do not exist, they will be generated and saved.
                    If False, will generate new response dataframes.

    use_events {Boolean} -- if True, will use extracted events instead of dF/F traces. Events files must be located in a folder called 'events' within the analysis_cache_dir.
                    If False, uses dF/F traces by default.

    use_extended_stimulus_presentations {Boolean} -- if True, will include extended stimulus presentations table columns in the response dataframe. This takes longer to load.
                        Note that some columns of extended_stimulus_presentations are also merged with the trials_response_df. These metrics are computed on each stimulus presentation, but
                        only the values on the change-stimulus are included in the trials_response_df.
                    If False, will use the simple version of stimulus presentations.

    overwrite_analysis_files {Boolean} -- if True will create and overwrite response analysis files.
        *** Please do not overwrite files in the default cache without consulting other scientists using this cache!!! Only overwrite things in your own cache ***
        This can be used if new functionality is added to the ResponseAnalysis class to modify existing structures or make new ones.
        If False, will load existing analysis files from analysis_cache_dir, or generate and save them if none exist.

    """

    def __init__(self, dataset, analysis_cache_dir=None, load_from_cache=False, use_events=False, filter_events=False,
                 use_extended_stimulus_presentations=False, overwrite_analysis_files=False, dataframe_format='wide'):
        self.dataset = dataset
        # promote ophys timestamps up to the top level
        self.ophys_timestamps = self.dataset.ophys_timestamps.copy()  # THROWS WARNING
        # promote stimulus_presentations to the top level
        self.stimulus_presentations = self.dataset.stimulus_presentations.copy()
        # promote dff_traces to the top level
        self.dff_traces = self.dataset.dff_traces.copy()
        # promote cell_specimen_table to the top level
        self.cell_specimen_table = self.dataset.cell_specimen_table.copy()
        # promote metadata to the top level
        self.metadata = self.dataset.metadata.copy()
        self.use_events = use_events
        self.filter_events = filter_events
        if analysis_cache_dir is None:
            self.analysis_cache_dir = get_analysis_cache_dir()
        else:
            self.analysis_cache_dir = analysis_cache_dir
        self.ophys_experiment_id = self.dataset.ophys_experiment_id
        self.load_from_cache = load_from_cache
        self.overwrite_analysis_files = overwrite_analysis_files
        self.dataframe_format = dataframe_format
        self.use_extended_stimulus_presentations = use_extended_stimulus_presentations
        self.trials_window = rp.get_default_trial_response_params()['window_around_timepoint_seconds']
        self.stimulus_presentations_window = rp.get_default_stimulus_response_params()[
            'window_around_timepoint_seconds']
        self.omissions_window = rp.get_default_omission_response_params()['window_around_timepoint_seconds']
        self.stimulus_duration = 0.25  # self.dataset.task_parameters['stimulus_duration'].values[0]
        self.ophys_frame_rate = self.metadata['ophys_frame_rate']
        self.stimulus_frame_rate = self.metadata['stimulus_frame_rate']
        task_parameters = self.dataset.task_parameters.copy()
        if 'blank_duration_sec' in task_parameters.keys():
            self.blank_duration = task_parameters['blank_duration_sec'][0]
        else:
            self.blank_duration = self.dataset.task_parameters['blank_duration'][0]

    def get_analysis_folder(self):
        candidates = [file for file in os.listdir(self.analysis_cache_dir) if str(self.ophys_experiment_id) in file]
        if len(candidates) == 1:
            self._analysis_folder = candidates[0]
        elif len(candidates) == 0:
            print('unable to locate analysis folder for experiment {} in {}'.format(self.ophys_experiment_id,
                                                                                    self.analysis_cache_dir))
            print('creating new analysis folder')
            m = self.dataset.metadata
            date = m['experiment_datetime']
            date = str(date)[:10]
            date = date[2:4] + date[5:7] + date[8:10]
            self._analysis_folder = str(m['ophys_experiment_id']) + '_' + str(m['donor_id']) + '_' + date + '_' + m[
                'targeted_structure'] + '_' + str(m['imaging_depth']) + '_' + m['driver_line'][0] + '_' + m[
                                        'rig_name'] + '_' + m['session_type']
            os.mkdir(os.path.join(self.analysis_cache_dir, self._analysis_folder))
        elif len(candidates) > 1:
            raise OSError(
                '{} contains multiple possible analysis folders: {}'.format(self.analysis_cache_dir, candidates))
        return self._analysis_folder

    analysis_folder = LazyLoadable('_analysis_folder', get_analysis_folder)

    def get_analysis_dir(self):
        self._analysis_dir = os.path.join(self.analysis_cache_dir, self.analysis_folder)
        return self._analysis_dir

    analysis_dir = LazyLoadable('_analysis_dir', get_analysis_dir)

    def get_response_df_path(self, df_name):
        if self.use_events:
            if 'response' in df_name:
                path = os.path.join(self.dataset.analysis_dir, df_name + '_events.h5')
            else:
                path = os.path.join(self.dataset.analysis_dir, df_name + '.h5')
        else:
            # path = os.path.join(self.dataset.analysis_dir, df_name + '.h5')
            path = os.path.join(self.dataset.analysis_dir, df_name + '_post_decrosstalk.h5')
        return path

    def save_response_df(self, df, df_name):
        print('saving', df_name)
        df.to_hdf(self.get_response_df_path(df_name), key='df')

    def get_df_for_df_name(self, df_name, df_format):
        if df_name == 'trials_response_df':
            df = rp.get_trials_response_df(self.dataset, self.use_events, self.filter_events, df_format=df_format)
        elif df_name == 'stimulus_response_df':
            df = rp.get_stimulus_response_df(self.dataset, self.use_events, self.filter_events, df_format=df_format)
        elif df_name == 'omission_response_df':
            df = rp.get_omission_response_df(self.dataset, self.use_events, self.filter_events, df_format=df_format)
        elif df_name == 'trials_run_speed_df':
            df = rp.get_trials_run_speed_df(self.dataset, df_format=df_format)
        elif df_name == 'stimulus_run_speed_df':
            df = rp.get_stimulus_run_speed_df(self.dataset, df_format=df_format)
        elif df_name == 'omission_run_speed_df':
            df = rp.get_omission_run_speed_df(self.dataset, df_format=df_format)
        elif df_name == 'trials_pupil_area_df':
            df = rp.get_trials_pupil_area_df(self.dataset, df_format=df_format)
        elif df_name == 'stimulus_pupil_area_df':
            df = rp.get_stimulus_pupil_area_df(self.dataset, df_format=df_format)
        elif df_name == 'omission_pupil_area_df':
            df = rp.get_omission_pupil_area_df(self.dataset, df_format=df_format)
        elif df_name == 'stimulus_licks_df':
            df = rp.get_stimulus_licks_df(self.dataset, df_format=df_format)
        elif df_name == 'trials_licks_df':
            df = rp.get_trials_licks_df(self.dataset, df_format=df_format)
        elif df_name == 'omission_licks_df':
            df = rp.get_omission_licks_df(self.dataset, df_format=df_format)
        elif df_name == 'lick_triggered_response_df':
            df = rp.get_lick_triggered_response_df(self.dataset, df_format=df_format)
        return df

    def get_response_df_types(self):
        return ['trials_response_df', 'stimulus_response_df', 'omission_response_df',
                'trials_run_speed_df', 'stimulus_run_speed_df', 'omission_run_speed_df',
                'trials_pupil_area_df', 'stimulus_pupil_area_df', 'omission_pupil_area_df',
                'trials_licks_df', 'stimulus_licks_df', 'omission_licks_df',
                'lick_triggered_response_df']

    def get_response_df(self, df_name='trials_response_df', df_format=None):
        if self.load_from_cache:  # get saved response df
            if os.path.exists(self.get_response_df_path(df_name)):
                print('loading', df_name)
                df = pd.read_hdf(self.get_response_df_path(df_name), key='df')
            else:
                print(df_name, 'not cached for this experiment')
        elif self.overwrite_analysis_files:  # delete any old files, generate new df and save
            file_path = self.get_response_df_path(df_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            df = self.get_df_for_df_name(df_name, df_format if df_format is not None else self.dataframe_format)
            self.save_response_df(df, df_name)
        else:  # default behavior - create the df
            df = self.get_df_for_df_name(df_name, df_format if df_format is not None else self.dataframe_format)

        if 'trials' in df_name:
            trials = self.dataset.trials.copy()
            # trials = trials.rename(
            #     columns={'response': 'behavioral_response', 'response_type': 'behavioral_response_type',
            #              'response_time': 'behavioral_response_time',
            #              'response_latency': 'behavioral_response_latency'})
            df = df.merge(trials, right_on='trials_id', left_on='trials_id')
            if self.use_extended_stimulus_presentations:
                # merge in the extended stimulus presentations df on the change_time/start_time columns
                stimulus_presentations = self.dataset.extended_stimulus_presentations.copy()
                stimulus_presentations['trials_id'] = None
                stimulus_presentations.loc[stimulus_presentations[stimulus_presentations.is_change].index, 'trials_id'] = trials[trials.stimulus_change].index.values
                columns_to_keep = [
                    'trials_id',
                    'epoch',
                    # 'bias',
                    # 'omissions1',
                    # 'task0',
                    # 'timing1D',
                    # 'engagement_state',
                    'pre_change',
                    'pre_omitted',
                    'flash_after_change',
                    'flash_after_omitted',
                    'licked',
                    'lick_on_next_flash',
                    'lick_on_previous_flash',
                    'mean_running_speed',
                    'mean_pupil_area'
                ]
                try:
                    df = df.merge(
                        stimulus_presentations[columns_to_keep],
                        left_on='trials_id',
                        right_on='trials_id',
                        how='left',
                        suffixes=('', '_duplicate'))
                except KeyError:  # if it cant merge them in, make empty columns
                    for column in columns_to_keep:
                        df[column] = None
        elif ('stimulus' in df_name) or ('omission' in df_name):
            if self.use_extended_stimulus_presentations:
                stimulus_presentations = self.dataset.extended_stimulus_presentations.copy()
            else:
                stimulus_presentations = self.dataset.stimulus_presentations.copy()
            df = df.merge(stimulus_presentations, right_on='stimulus_presentations_id',
                          left_on='stimulus_presentations_id')
        return df

    def get_trials_response_df(self, df_format=None):
        df_name = 'trials_response_df'
        df = self.get_response_df(df_name, df_format)
        self._trials_response_df = df
        return self._trials_response_df

    trials_response_df = LazyLoadable('_trials_response_df', get_trials_response_df)

    def get_stimulus_response_df(self, df_format=None):
        df_name = 'stimulus_response_df'
        df = self.get_response_df(df_name, df_format)
        self._stimulus_response_df = df
        return self._stimulus_response_df

    stimulus_response_df = LazyLoadable('_stimulus_response_df', get_stimulus_response_df)

    def get_omission_response_df(self, df_format=None):
        df_name = 'omission_response_df'
        df = self.get_response_df(df_name, df_format)
        self._omission_response_df = df
        return self._omission_response_df

    omission_response_df = LazyLoadable('_omission_response_df', get_omission_response_df)

    def get_trials_run_speed_df(self, df_format=None):
        df_name = 'trials_run_speed_df'
        df = self.get_response_df(df_name, df_format)
        self._trials_run_speed_df = df
        return self._trials_run_speed_df

    trials_run_speed_df = LazyLoadable('_trials_run_speed_df', get_trials_run_speed_df)

    def get_stimulus_run_speed_df(self, df_format=None):
        df_name = 'stimulus_run_speed_df'
        df = self.get_response_df(df_name, df_format)
        self._stimulus_run_speed_df = df
        return self._stimulus_run_speed_df

    stimulus_run_speed_df = LazyLoadable('_stimulus_run_speed_df', get_stimulus_run_speed_df)

    def get_omission_run_speed_df(self, df_format=None):
        df_name = 'omission_run_speed_df'
        df = self.get_response_df(df_name, df_format)
        self._omission_run_speed_df = df
        return self._omission_run_speed_df

    omission_run_speed_df = LazyLoadable('_omission_run_speed_df', get_omission_run_speed_df)

    def get_omission_pupil_area_df(self, df_format=None):
        df_name = 'omission_pupil_area_df'
        df = self.get_response_df(df_name, df_format)
        self._omission_pupil_area_df = df
        return self._omission_pupil_area_df

    omission_pupil_area_df = LazyLoadable('_omission_pupil_area_df', get_omission_pupil_area_df)

    def get_stimulus_run_speed_df(self, df_format=None):
        df_name = 'stimulus_run_speed_df'
        df = self.get_response_df(df_name, df_format)
        self._stimulus_run_speed_df = df
        return self._stimulus_run_speed_df

    stimulus_run_speed_df = LazyLoadable('_stimulus_run_speed_df', get_stimulus_run_speed_df)

    def get_stimulus_pupil_area_df(self, df_format=None):
        df_name = 'stimulus_pupil_area_df'
        df = self.get_response_df(df_name, df_format)
        self._stimulus_pupil_area_df = df
        return self._stimulus_pupil_area_df

    stimulus_pupil_area_df = LazyLoadable('_stimulus_pupil_area_df', get_stimulus_pupil_area_df)
