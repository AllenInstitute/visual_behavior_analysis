"""
Created on Sunday July 15 2018

@author: marinag
"""
import os
import h5py
import platform
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


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


class VisualBehaviorOphysDataset(object):
    # TODO getter methods no longer need to set attributes directly

    def __init__(self, experiment_id, cache_dir=None, **kwargs):
        """Initialize visual behavior ophys experiment dataset.
            Loads experiment data from cache_dir, including dF/F traces, roi masks, stimulus metadata, running speed, licks, rewards, and metadata.

            All available experiment data is read upon initialization of the object.
            Data can be accessed directly as attributes of the class (ex: dff_traces = dataset.dff_traces) or by using functions (ex: dff_traces = dataset.get_dff_traces()

        Parameters
        ----------
        experiment_id : ophys experiment ID
        cache_dir : directory where data files are located
        append_omitted_to_stim_metadata (bool): adds an additional omitted stimulus row to the end of stimulus_metadata
        """
        self.experiment_id = experiment_id
        self.cache_dir = cache_dir
        self.cache_dir = self.get_cache_dir()
        self.roi_metrics = self.get_roi_metrics()
        self.append_omitted_to_stim_metadata = True
        # if self.roi_metrics.cell_specimen_id.values[0] is None:
        #     self.cell_matching = False
        # else:
        #     self.cell_matching = True

    def get_cache_dir(self):
        if self.cache_dir is None:
            if platform.system() == 'Linux':
                cache_dir = r'/allen/programs/mindscope/workgroups/learning/ophys/learning_project_cache'
            else:
                cache_dir = r'\\allen\programs\mindscope\workgroups\learning\ophys\learning_project_cache'

            logger.info('using default cache_dir: {}'.format(cache_dir))
        else:
            cache_dir = self.cache_dir
        self.cache_dir = cache_dir
        return self.cache_dir

    def get_analysis_folder(self):
        candidates = [file for file in os.listdir(self.cache_dir) if str(self.experiment_id) in file]
        if len(candidates) == 1:
            self._analysis_folder = candidates[0]
        elif len(candidates) < 1:
            raise OSError(
                'unable to locate analysis folder for experiment {} in {}'.format(self.experiment_id, self.cache_dir))
        elif len(candidates) > 1:
            raise OSError('{} contains multiple possible analysis folders: {}'.format(self.cache_dir, candidates))

        return self._analysis_folder

    analysis_folder = LazyLoadable('_analysis_folder', get_analysis_folder)

    def get_analysis_dir(self):
        self._analysis_dir = os.path.join(self.cache_dir, self.analysis_folder)
        return self._analysis_dir

    analysis_dir = LazyLoadable('_analysis_dir', get_analysis_dir)

    def get_metadata(self):
        self._metadata = pd.read_hdf(os.path.join(self.analysis_dir, 'metadata.h5'), key='df')
        return self._metadata

    metadata = LazyLoadable('_metadata', get_metadata)

    def get_timestamps(self):
        self._timestamps = pd.read_hdf(os.path.join(self.analysis_dir, 'timestamps.h5'), key='df')
        return self._timestamps

    timestamps = LazyLoadable('_timestamps', get_timestamps)

    def get_stimulus_timestamps(self):
        self._stimulus_timestamps = self.timestamps['stimulus_frames']['timestamps']
        return self._stimulus_timestamps

    stimulus_timestamps = LazyLoadable('_stimulus_timestamps', get_stimulus_timestamps)

    def get_ophys_timestamps(self):
        self._ophys_timestamps = self.timestamps['ophys_frames']['timestamps']
        return self._ophys_timestamps

    ophys_timestamps = LazyLoadable('_ophys_timestamps', get_ophys_timestamps)

    def get_eye_tracking_timestamps(self):
        self._eye_tracking_timestamps = self.timestamps['behavior_monitoring']['timestamps']  # line labels swapped
        return self._eye_tracking_timestamps

    eye_tracking_timestamps = LazyLoadable('_eye_tracking_timestamps', get_eye_tracking_timestamps)

    def get_stimulus_table(self):
        self._stimulus_table = pd.read_hdf(
            os.path.join(self.analysis_dir, 'stimulus_table.h5'),
            key='df'
        )
        self._stimulus_table = self._stimulus_table.reset_index()
        # self._stimulus_table = self._stimulus_table.drop(
        #     columns=['orientation', 'image_category', 'start_frame', 'end_frame', 'duration', 'index']
        # )
        self._stimulus_table = self._stimulus_table.drop(
            columns=['start_frame', 'end_frame', 'index'])
        if 'level_0' in list(self._stimulus_table.keys()):
            self._stimulus_table = self._stimulus_table.drop(columns=['level_0'])
        return self._stimulus_table

    stimulus_table = LazyLoadable('_stimulus_table', get_stimulus_table)

    def get_stimulus_template(self):
        with h5py.File(os.path.join(self.analysis_dir, 'stimulus_template.h5'), 'r') as stimulus_template_file:
            self._stimulus_template = np.asarray(stimulus_template_file['data'])
        return self._stimulus_template

    stimulus_template = LazyLoadable('_stimulus_template', get_stimulus_template)

    def get_stimulus_metadata(self, append_omitted_to_stim_metadata=True):
        stimulus_metadata = pd.read_hdf(
            os.path.join(self.analysis_dir, 'stimulus_metadata.h5'), key='df')
        stimulus_metadata = stimulus_metadata.drop(columns='image_category')
        stimulus_metadata['image_name'] = [image_name for image_name in stimulus_metadata.image_name.values]
        # Add an entry for omitted stimuli
        if append_omitted_to_stim_metadata:
            omitted_df = pd.DataFrame({'image_name': ['omitted'], 'image_index': [stimulus_metadata['image_index'].max() + 1]})
            stimulus_metadata = stimulus_metadata.append(omitted_df, ignore_index=True, sort=False)
        # stimulus_metadata.set_index(['image_index'], inplace=True, drop=True)
        self._stimulus_metadata = stimulus_metadata
        return self._stimulus_metadata

    stimulus_metadata = LazyLoadable('_stimulus_metadata', get_stimulus_metadata)

    def get_running_speed(self):
        self._running_speed = pd.read_hdf(os.path.join(self.analysis_dir, 'running_speed.h5'), key='df')
        return self._running_speed

    running_speed = LazyLoadable('_running_speed', get_running_speed)

    def get_licks(self):
        self._licks = pd.read_hdf(os.path.join(self.analysis_dir, 'licks.h5'), key='df')
        return self._licks

    licks = LazyLoadable('_licks', get_licks)

    def get_rewards(self):
        self._rewards = pd.read_hdf(os.path.join(self.analysis_dir, 'rewards.h5'), key='df')
        return self._rewards

    rewards = LazyLoadable('_rewards', get_rewards)

    def get_task_parameters(self):
        self._task_parameters = pd.read_hdf(
            os.path.join(self.analysis_dir, 'task_parameters.h5'),
            key='df'
        )
        return self._task_parameters

    task_parameters = LazyLoadable('_task_parameters', get_task_parameters)

    def get_all_trials(self):
        self._all_trials = pd.read_hdf(os.path.join(self.analysis_dir, 'trials.h5'), key='df')
        return self._all_trials

    all_trials = LazyLoadable('_all_trials', get_all_trials)

    def get_trials(self):
        all_trials = self.all_trials.copy()
        trials = all_trials[(all_trials.auto_rewarded != True) & (all_trials.trial_type != 'aborted')].reset_index()
        trials = trials.rename(columns={'level_0': 'original_trial_index'})
        trials.insert(loc=0, column='trial', value=trials.index.values)
        trials = trials.rename(
            columns={'starttime': 'start_time', 'endtime': 'end_time', 'startdatetime': 'start_date_time',
                     'level_0': 'original_trial_index', 'color': 'trial_type_color'})
        trials = trials[
            ['trial', 'change_time', 'initial_image_name', 'change_image_name', 'trial_type', 'trial_type_color',
             'response', 'response_type', 'response_window', 'lick_times', 'response_latency', 'rewarded',
             'reward_times',
             'reward_volume', 'reward_rate', 'start_time', 'end_time', 'trial_length', 'mouse_id',
             'start_date_time']]
        trials['trials_id'] = trials.trial.values
        trials.set_index('trials_id', inplace=True)
        import visual_behavior.ophys.dataset.stimulus_processing as sp
        trials = sp.add_run_speed_to_trials(self.running_speed, trials)
        self._trials = trials
        return self._trials

    trials = LazyLoadable('_trials', get_trials)

    def get_dff_traces_array(self):
        with h5py.File(os.path.join(self.analysis_dir, 'dff_traces.h5'), 'r') as dff_traces_file:
            dff_traces = []
            for key in dff_traces_file.keys():
                dff_traces.append(np.asarray(dff_traces_file[key]))
        self._dff_traces_array = np.asarray(dff_traces)
        return self._dff_traces_array

    dff_traces_array = LazyLoadable('_dff_traces_array', get_dff_traces_array)

    def get_corrected_fluorescence_traces(self):
        with h5py.File(os.path.join(self.analysis_dir, 'corrected_fluorescence_traces.h5'),
                       'r') as corrected_fluorescence_traces_file:
            corrected_fluorescence_traces = []
            for key in corrected_fluorescence_traces_file.keys():
                corrected_fluorescence_traces.append(np.asarray(corrected_fluorescence_traces_file[key]))
        self._corrected_fluorescence_traces = np.asarray(corrected_fluorescence_traces)
        return self._corrected_fluorescence_traces

    corrected_fluorescence_traces = LazyLoadable('_corrected_fluorescence_traces', get_corrected_fluorescence_traces)

    def get_events_array(self):
        events_folder = os.path.join(self.cache_dir, 'events')
        if os.path.exists(events_folder):
            events_file = [file for file in os.listdir(events_folder) if
                           str(self.experiment_id) + '_events.npz' in file]
            if len(events_file) > 0:
                print('getting L0 events')
                f = np.load(os.path.join(events_folder, events_file[0]))
                events = np.asarray(f['ev'])
                f.close()
                # account for excess ophys frame times at end of recording due to Scientifica software error
                if events.shape[1] > self.ophys_timestamps.shape[0]:
                    # difference = self.ophys_timestamps.shape[0] - events.shape[1]
                    events = events[:, :self.ophys_timestamps.shape[0]]
            else:
                print('no events for this experiment')
                events = None
        else:
            print('no events for this experiment')
            events = None
        self._events_array = events
        return self._events_array

    events_array = LazyLoadable('_events_array', get_events_array)

    def get_roi_metrics(self):
        self._roi_metrics = pd.read_hdf(os.path.join(self.analysis_dir, 'roi_metrics.h5'), key='df')
        return self._roi_metrics

    roi_metrics = LazyLoadable('_roi_metrics', get_roi_metrics)

    def get_roi_mask_dict(self):
        f = h5py.File(os.path.join(self.analysis_dir, 'roi_masks.h5'), 'r')
        roi_mask_dict = {}
        for key in list(f.keys()):
            roi_mask_dict[key] = np.asarray(f[key])
        f.close()
        self._roi_mask_dict = roi_mask_dict
        return self._roi_mask_dict

    roi_mask_dict = LazyLoadable('_roi_mask_dict', get_roi_mask_dict)

    def get_roi_mask_array(self):
        w, h = self.roi_mask_dict[list(self.roi_mask_dict.keys())[0]].shape
        roi_mask_array = np.empty((len(list(self.roi_mask_dict.keys())), w, h))
        for cell_specimen_id in self.cell_specimen_ids:
            cell_index = self.get_cell_index_for_cell_specimen_id(int(cell_specimen_id))
            roi_mask_array[cell_index] = self.roi_mask_dict[str(cell_specimen_id)]
        self._roi_mask_array = roi_mask_array
        return self._roi_mask_array

    roi_mask_array = LazyLoadable('_roi_mask_array', get_roi_mask_array)

    def get_max_projection(self):
        with h5py.File(os.path.join(self.analysis_dir, 'max_projection.h5'), 'r') as max_projection_file:
            self._max_projection = np.asarray(max_projection_file['data'])
        return self._max_projection

    max_projection = LazyLoadable('_max_projection', get_max_projection)

    def get_average_image(self):
        with h5py.File(os.path.join(self.analysis_dir, 'average_image.h5'), 'r') as average_image_file:
            self._average_image = np.asarray(average_image_file['data'])
        return self._average_image

    average_image = LazyLoadable('_average_image', get_average_image)

    def get_motion_correction(self):
        self._motion_correction = pd.read_hdf(
            os.path.join(self.analysis_dir, 'motion_correction.h5'),
            key='df'
        )
        return self._motion_correction

    motion_correction = LazyLoadable('_motion_correction', get_motion_correction)

    def get_cell_specimen_ids(self):
        roi_metrics = self.roi_metrics
        self._cell_specimen_ids = np.asarray([roi_metrics[roi_metrics.roi_id == roi_id].id.values[0]
                                              for roi_id in np.sort(self.roi_metrics.roi_id.values)])
        return self._cell_specimen_ids

    cell_specimen_ids = LazyLoadable('_cell_specimen_ids', get_cell_specimen_ids)

    def get_cell_indices(self):
        self._cell_indices = np.sort(self.roi_metrics.cell_index.values)
        return self._cell_indices

    cell_indices = LazyLoadable('_cell_indices', get_cell_indices)

    def get_valid_cell_indices(self):
        roi_metrics = self.roi_metrics.copy()
        valid_cell_indices = roi_metrics[roi_metrics.valid_roi == True].cell_index.values

        self._valid_cell_indices = np.sort(valid_cell_indices)
        return self._valid_cell_indices

    valid_cell_indices = LazyLoadable('_valid_cell_indices', get_valid_cell_indices)

    def get_cell_specimen_id_for_cell_index(self, cell_index):
        roi_metrics = self.roi_metrics
        cell_specimen_id = roi_metrics[roi_metrics.cell_index == cell_index].id.values[0]
        return cell_specimen_id

    def get_cell_index_for_cell_specimen_id(self, cell_specimen_id):
        roi_metrics = self.roi_metrics
        cell_index = roi_metrics[roi_metrics.id == cell_specimen_id].cell_index.values[0]
        return cell_index

    def get_dff_traces(self):
        self._dff_traces = pd.DataFrame({'dff': [x for x in self.dff_traces_array]},
                                        index=pd.Index(self.cell_specimen_ids, name='cell_specimen_id'))
        return self._dff_traces

    dff_traces = LazyLoadable('_dff_traces', get_dff_traces)

    def get_events(self):
        self._events = pd.DataFrame({'events': [x for x in self.events_array]},
                                    index=pd.Index(self.cell_specimen_ids, name='cell_specimen_id'))
        return self._events

    events = LazyLoadable('_events', get_events)

    def get_pupil_area(self):
        session_id = self.metadata.session_id.values[0]
        data_dir = os.path.join(self.cache_dir, 'pupil_data')
        filename = [file for file in os.listdir(data_dir) if str(session_id) in file]
        if len(filename) > 0:
            df = pd.read_csv(os.path.join(data_dir, filename[0]))
            area = df.blink_corrected_area.values
            # compare with timestamps
            timestamps = self.timestamps.behavior_monitoring.values[0]
            diff = len(area) - len(timestamps)
            if diff <= 5:
                df = pd.DataFrame()
                df['pupil_area'] = area[:len(timestamps)]
                df['time'] = timestamps[:len(area)]
                self._pupil_area = df
            else:
                self._pupil_area = None
        else:
            print('no pupil data for', self.experiment_id)
            self._pupil_area = None
        return self._pupil_area

    pupil_area = LazyLoadable('_pupil_area', get_pupil_area)

    def get_stimulus_presentations(self):
        stimulus_presentations_df = self.stimulus_table.copy()
        stimulus_presentations_df = stimulus_presentations_df.rename(
            columns={'flash_number': 'stimulus_presentations_id'})
        stimulus_presentations_df.set_index('stimulus_presentations_id', inplace=True)
        stimulus_metadata_df = self.get_stimulus_metadata(self.append_omitted_to_stim_metadata)
        idx_name = stimulus_presentations_df.index.name
        stimulus_presentations_df = stimulus_presentations_df.reset_index().merge(stimulus_metadata_df,
                                                                                  on=['image_name']).set_index(idx_name)
        stimulus_presentations_df.sort_index(inplace=True)
        return stimulus_presentations_df

    stimulus_presentations = LazyLoadable('_stimulus_presentations', get_stimulus_presentations)

    def get_extended_stimulus_presentations(self):
        '''
        Calculates additional information for each stimulus presentation
        '''
        import visual_behavior.ophys.dataset.stimulus_processing as sp
        stimulus_presentations_pre = self.get_stimulus_presentations()
        change_times = self.trials['change_time'].values
        change_times = change_times[~np.isnan(change_times)]
        extended_stimulus_presentations = sp.get_extended_stimulus_presentations(
            stimulus_presentations_df=stimulus_presentations_pre,
            licks=self.licks,
            rewards=self.rewards,
            change_times=change_times,
            running_speed_df=self.running_speed,
            pupil_area=self.pupil_area
        )
        return extended_stimulus_presentations

    extended_stimulus_presentations = LazyLoadable('_extended_stimulus_presentations',
                                                   get_extended_stimulus_presentations)

    @classmethod
    def construct_and_load(cls, experiment_id, cache_dir=None, **kwargs):
        ''' Instantiate a VisualBehaviorOphysDataset and load its data

        Parameters
        ----------
        experiment_id : int
            identifier for this experiment
        cache_dir : str
            filesystem path to directory containing this experiment's

        '''

        obj = cls(experiment_id, cache_dir=cache_dir, **kwargs)

        obj.get_analysis_dir()
        obj.get_metadata()
        obj.get_timestamps()
        obj.get_ophys_timestamps()
        obj.get_stimulus_timestamps()
        obj.get_eye_tracking_timestamps()
        obj.get_stimulus_table()
        obj.get_stimulus_template()
        obj.get_stimulus_metadata()
        obj.get_running_speed()
        obj.get_licks()
        obj.get_rewards()
        obj.get_task_parameters()
        obj.get_trials()
        obj.get_dff_traces_array()
        obj.get_corrected_fluorescence_traces()
        obj.get_events_array()
        obj.get_roi_metrics()
        obj.get_roi_mask_dict()
        obj.get_roi_mask_array()
        obj.get_cell_specimen_ids()
        obj.get_cell_indices()
        obj.get_valid_cell_indices()
        obj.get_max_projection()
        obj.get_average_image()
        obj.get_motion_correction()
        obj.get_dff_traces()
        obj.get_events()
        obj.get_pupil_area()
        obj.get_stimulus_presentations()
        # obj.get_extended_stimulus_presentations()

        return obj
