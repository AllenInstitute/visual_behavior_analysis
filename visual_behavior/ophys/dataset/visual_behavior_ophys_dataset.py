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
        if self.roi_metrics.cell_specimen_id.values[0] is None:
            self.cell_matching = False
        else:
            self.cell_matching = True

    def get_cache_dir(self):
        """Resolve the cache directory path, falling back to platform defaults.

        If ``cache_dir`` was not provided at construction, uses a hard-coded
        Allen Institute network path (Linux or Windows).

        .. note::
            The default paths are only accessible from within the Allen
            Institute network.  External users must pass ``cache_dir``
            explicitly.

        Returns
        -------
        str
            Absolute path to the cache directory.
        """
        if self.cache_dir is None:
            if platform.system() == 'Linux':
                cache_dir = r'/allen/aibs/informatics/swdb2018/visual_behavior'
            else:
                cache_dir = r'\\allen\aibs\informatics\swdb2018\visual_behavior'

            logger.info('using default cache_dir: {}'.format(cache_dir))
        else:
            cache_dir = self.cache_dir
        self.cache_dir = cache_dir
        return self.cache_dir

    def get_analysis_folder(self):
        """Find the per-experiment analysis folder within ``cache_dir``.

        Searches ``cache_dir`` for a directory whose name contains the
        experiment ID string.

        Returns
        -------
        str
            Name of the analysis folder (not the full path).

        Raises
        ------
        OSError
            If zero or more than one candidate folder is found.
        """
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
        """Return the full path to the per-experiment analysis directory.

        Returns
        -------
        str
            Absolute path: ``<cache_dir>/<analysis_folder>``.
        """
        self._analysis_dir = os.path.join(self.cache_dir, self.analysis_folder)
        return self._analysis_dir

    analysis_dir = LazyLoadable('_analysis_dir', get_analysis_dir)

    def get_metadata(self):
        """Load experiment metadata from ``metadata.h5``.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with columns such as ``ophys_experiment_id``,
            ``session_type``, ``targeted_structure``, ``imaging_depth``,
            ``driver_line``, ``ophys_frame_rate``, and ``stimulus_frame_rate``.
        """
        self._metadata = pd.read_hdf(os.path.join(self.analysis_dir, 'metadata.h5'), key='df')
        return self._metadata

    metadata = LazyLoadable('_metadata', get_metadata)

    def get_timestamps(self):
        """Load all hardware timestamps from ``timestamps.h5``.

        Returns
        -------
        pd.DataFrame
            DataFrame with multi-level columns grouping timestamps by stream:
            ``ophys_frames``, ``stimulus_frames``, and
            ``behavior_monitoring`` (eye-tracking).
        """
        self._timestamps = pd.read_hdf(os.path.join(self.analysis_dir, 'timestamps.h5'), key='df')
        return self._timestamps

    timestamps = LazyLoadable('_timestamps', get_timestamps)

    def get_stimulus_timestamps(self):
        """Extract the stimulus-frame timestamps array from ``timestamps``.

        Returns
        -------
        np.ndarray
            1-D array of timestamps (seconds) for each stimulus frame.
        """
        self._stimulus_timestamps = self.timestamps['stimulus_frames']['timestamps']
        return self._stimulus_timestamps

    stimulus_timestamps = LazyLoadable('_stimulus_timestamps', get_stimulus_timestamps)

    def get_ophys_timestamps(self):
        """Extract the 2-photon imaging-frame timestamps array from ``timestamps``.

        Returns
        -------
        np.ndarray
            1-D array of timestamps (seconds) for each ophys frame.
        """
        self._ophys_timestamps = self.timestamps['ophys_frames']['timestamps']
        return self._ophys_timestamps

    ophys_timestamps = LazyLoadable('_ophys_timestamps', get_ophys_timestamps)

    def get_eye_tracking_timestamps(self):
        """Extract eye-tracking camera timestamps from ``timestamps``.

        .. note::
            In the raw HDF5 file the stream is stored under
            ``'behavior_monitoring'`` due to a label-swap in the acquisition
            software; the returned values correspond to the eye-tracking camera.

        Returns
        -------
        np.ndarray
            1-D array of timestamps (seconds) for each eye-tracking frame.
        """
        self._eye_tracking_timestamps = self.timestamps['behavior_monitoring']['timestamps']  # line labels swapped
        return self._eye_tracking_timestamps

    eye_tracking_timestamps = LazyLoadable('_eye_tracking_timestamps', get_eye_tracking_timestamps)

    def get_stimulus_table(self):
        """Load the raw stimulus presentation table from ``stimulus_table.h5``.

        Drops ``start_frame``, ``end_frame``, and ``index`` columns and resets
        the index.  Use ``stimulus_presentations`` for the merged, enriched
        version.

        Returns
        -------
        pd.DataFrame
            One row per stimulus presentation with columns such as
            ``image_name``, ``start_time``, ``stop_time``, and
            ``flash_number``.
        """
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
        """Load the image stimulus template array from ``stimulus_template.h5``.

        Returns
        -------
        np.ndarray
            3-D array of shape ``(n_images, height, width)`` containing the
            grayscale pixel values for each image in the stimulus set.
        """
        with h5py.File(os.path.join(self.analysis_dir, 'stimulus_template.h5'), 'r') as stimulus_template_file:
            self._stimulus_template = np.asarray(stimulus_template_file['data'])
        return self._stimulus_template

    stimulus_template = LazyLoadable('_stimulus_template', get_stimulus_template)

    def get_stimulus_metadata(self, append_omitted_to_stim_metadata=True):
        """Load the stimulus image metadata lookup table.

        Parameters
        ----------
        append_omitted_to_stim_metadata : bool, optional
            If ``True`` (default), appends an extra row representing omitted
            stimuli with ``image_name='omitted'`` and an incremented
            ``image_index``.

        Returns
        -------
        pd.DataFrame
            One row per unique image in the stimulus set with columns
            ``image_name`` and ``image_index``.
        """
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
        """Load running-speed data from ``running_speed.h5``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``speed`` (cm/s) and ``time`` (seconds),
            sampled at the stimulus frame rate.
        """
        self._running_speed = pd.read_hdf(os.path.join(self.analysis_dir, 'running_speed.h5'), key='df')
        return self._running_speed

    running_speed = LazyLoadable('_running_speed', get_running_speed)

    def get_licks(self):
        """Load lick event times from ``licks.h5``.

        Returns
        -------
        pd.DataFrame
            DataFrame with a ``time`` column containing the timestamp (seconds)
            of each lick event.
        """
        self._licks = pd.read_hdf(os.path.join(self.analysis_dir, 'licks.h5'), key='df')
        return self._licks

    licks = LazyLoadable('_licks', get_licks)

    def get_rewards(self):
        """Load water reward delivery times from ``rewards.h5``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``time`` (seconds) and ``volume`` (µL)
            for each delivered water reward.
        """
        self._rewards = pd.read_hdf(os.path.join(self.analysis_dir, 'rewards.h5'), key='df')
        return self._rewards

    rewards = LazyLoadable('_rewards', get_rewards)

    def get_task_parameters(self):
        """Load task configuration parameters from ``task_parameters.h5``.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame containing task settings such as
            ``stimulus_duration_sec``, ``blank_duration_sec``,
            ``response_window``, and ``reward_volume``.
        """
        self._task_parameters = pd.read_hdf(
            os.path.join(self.analysis_dir, 'task_parameters.h5'),
            key='df'
        )
        return self._task_parameters

    task_parameters = LazyLoadable('_task_parameters', get_task_parameters)

    def get_all_trials(self):
        """Load the raw trials table (including aborted and auto-rewarded trials).

        Use ``trials`` for the filtered version that excludes aborted and
        auto-rewarded trials.

        Returns
        -------
        pd.DataFrame
            One row per trial with columns such as ``starttime``,
            ``change_time``, ``trial_type``, ``response``, and
            ``auto_rewarded``.
        """
        self._all_trials = pd.read_hdf(os.path.join(self.analysis_dir, 'trials.h5'), key='df')
        return self._all_trials

    all_trials = LazyLoadable('_all_trials', get_all_trials)

    def get_trials(self):
        """Return behavioural trials, excluding aborted and auto-rewarded trials.

        Filters ``all_trials`` to keep only genuine go/catch trials, renames
        columns to snake_case, restricts to a curated set of columns, and
        appends mean running speed per trial.

        Returns
        -------
        pd.DataFrame
            One row per trial (indexed by ``trials_id``) with columns
            including ``change_time``, ``initial_image_name``,
            ``change_image_name``, ``trial_type``, ``response``,
            ``response_type``, ``lick_times``, ``reward_times``,
            ``reward_rate``, and ``mean_running_speed``.
        """
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
        """Load dF/F traces as a 2-D NumPy array from ``dff_traces.h5``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_cells, n_timepoints)`` where rows are
            ordered by cell index (matching ``cell_indices``).
        """
        with h5py.File(os.path.join(self.analysis_dir, 'dff_traces.h5'), 'r') as dff_traces_file:
            dff_traces = []
            for key in dff_traces_file.keys():
                dff_traces.append(np.asarray(dff_traces_file[key]))
        self._dff_traces_array = np.asarray(dff_traces)
        return self._dff_traces_array

    dff_traces_array = LazyLoadable('_dff_traces_array', get_dff_traces_array)

    def get_corrected_fluorescence_traces(self):
        """Load neuropil-corrected fluorescence traces from HDF5.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_cells, n_timepoints)`` containing neuropil-
            corrected (but not dF/F-normalised) fluorescence values.
        """
        with h5py.File(os.path.join(self.analysis_dir, 'corrected_fluorescence_traces.h5'),
                       'r') as corrected_fluorescence_traces_file:
            corrected_fluorescence_traces = []
            for key in corrected_fluorescence_traces_file.keys():
                corrected_fluorescence_traces.append(np.asarray(corrected_fluorescence_traces_file[key]))
        self._corrected_fluorescence_traces = np.asarray(corrected_fluorescence_traces)
        return self._corrected_fluorescence_traces

    corrected_fluorescence_traces = LazyLoadable('_corrected_fluorescence_traces', get_corrected_fluorescence_traces)

    def get_events_array(self):
        """Load L0-event arrays from an ``events/`` subdirectory of ``cache_dir``.

        Events are sparse binary-like arrays produced by the L0 event-detection
        algorithm applied to dF/F traces.  If no events file is found,
        returns ``None``.

        Returns
        -------
        np.ndarray or None
            Array of shape ``(n_cells, n_timepoints)`` if events exist,
            otherwise ``None``.
        """
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
        """Load ROI quality metrics from ``roi_metrics.h5``.

        Returns
        -------
        pd.DataFrame
            One row per ROI with columns including ``roi_id``,
            ``cell_specimen_id``, ``cell_index``, ``valid_roi``,
            ``width``, ``height``, and ``mask_image_plane``.
        """
        self._roi_metrics = pd.read_hdf(os.path.join(self.analysis_dir, 'roi_metrics.h5'), key='df')
        return self._roi_metrics

    roi_metrics = LazyLoadable('_roi_metrics', get_roi_metrics)

    def get_roi_mask_dict(self):
        """Load ROI pixel masks as a dictionary keyed by cell specimen ID.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from ``str(cell_specimen_id)`` to a 2-D boolean array of
            shape ``(height, width)`` indicating which pixels belong to the ROI.
        """
        f = h5py.File(os.path.join(self.analysis_dir, 'roi_masks.h5'), 'r')
        roi_mask_dict = {}
        for key in list(f.keys()):
            roi_mask_dict[key] = np.asarray(f[key])
        f.close()
        self._roi_mask_dict = roi_mask_dict
        return self._roi_mask_dict

    roi_mask_dict = LazyLoadable('_roi_mask_dict', get_roi_mask_dict)

    def get_roi_mask_array(self):
        """Return all ROI pixel masks stacked into a 3-D NumPy array.

        Rows are ordered by ``cell_specimen_ids``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_cells, height, width)`` containing each cell's
            pixel mask.
        """
        w, h = self.roi_mask_dict[list(self.roi_mask_dict.keys())[0]].shape
        roi_mask_array = np.empty((len(list(self.roi_mask_dict.keys())), w, h))
        for cell_specimen_id in self.cell_specimen_ids:
            cell_index = self.get_cell_index_for_cell_specimen_id(int(cell_specimen_id))
            roi_mask_array[cell_index] = self.roi_mask_dict[str(cell_specimen_id)]
        self._roi_mask_array = roi_mask_array
        return self._roi_mask_array

    roi_mask_array = LazyLoadable('_roi_mask_array', get_roi_mask_array)

    def get_max_projection(self):
        """Load the maximum-intensity projection image from ``max_projection.h5``.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(height, width)`` representing the max-intensity
            projection across all acquired imaging frames.
        """
        with h5py.File(os.path.join(self.analysis_dir, 'max_projection.h5'), 'r') as max_projection_file:
            self._max_projection = np.asarray(max_projection_file['data'])
        return self._max_projection

    max_projection = LazyLoadable('_max_projection', get_max_projection)

    def get_average_image(self):
        """Load the mean fluorescence image from ``average_image.h5``.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(height, width)`` representing the pixel-wise
            mean fluorescence across all acquired imaging frames.
        """
        with h5py.File(os.path.join(self.analysis_dir, 'average_image.h5'), 'r') as average_image_file:
            self._average_image = np.asarray(average_image_file['data'])
        return self._average_image

    average_image = LazyLoadable('_average_image', get_average_image)

    def get_motion_correction(self):
        """Load the per-frame rigid-body motion correction offsets.

        Returns
        -------
        pd.DataFrame
            One row per ophys frame with columns ``x`` and ``y`` (pixel
            shifts applied by the motion correction algorithm).
        """
        self._motion_correction = pd.read_hdf(
            os.path.join(self.analysis_dir, 'motion_correction.h5'),
            key='df'
        )
        return self._motion_correction

    motion_correction = LazyLoadable('_motion_correction', get_motion_correction)

    def get_cell_specimen_ids(self):
        """Return cell specimen IDs sorted by ROI ID.

        Returns
        -------
        np.ndarray
            1-D integer array of global cell specimen IDs, one per ROI,
            ordered by ascending ``roi_id``.
        """
        roi_metrics = self.roi_metrics
        self._cell_specimen_ids = np.asarray([roi_metrics[roi_metrics.roi_id == roi_id].id.values[0]
                                              for roi_id in np.sort(self.roi_metrics.roi_id.values)])
        return self._cell_specimen_ids

    cell_specimen_ids = LazyLoadable('_cell_specimen_ids', get_cell_specimen_ids)

    def get_cell_indices(self):
        """Return sorted integer cell indices for all ROIs.

        Returns
        -------
        np.ndarray
            1-D integer array of zero-based cell indices, sorted in ascending
            order.  Used to index into ``dff_traces_array`` and
            ``events_array``.
        """
        self._cell_indices = np.sort(self.roi_metrics.cell_index.values)
        return self._cell_indices

    cell_indices = LazyLoadable('_cell_indices', get_cell_indices)

    def get_cell_specimen_id_for_cell_index(self, cell_index):
        """Look up the global cell specimen ID for a local cell index.

        Parameters
        ----------
        cell_index : int
            Zero-based index into ``dff_traces_array``/``events_array``.

        Returns
        -------
        int
            Corresponding global cell specimen ID from the Allen Institute
            cell registry.
        """
        roi_metrics = self.roi_metrics
        cell_specimen_id = roi_metrics[roi_metrics.cell_index == cell_index].id.values[0]
        return cell_specimen_id

    def get_cell_index_for_cell_specimen_id(self, cell_specimen_id):
        """Look up the local cell index for a global cell specimen ID.

        Parameters
        ----------
        cell_specimen_id : int
            Global cell specimen ID from the Allen Institute cell registry.

        Returns
        -------
        int
            Zero-based index into ``dff_traces_array``/``events_array``.
        """
        roi_metrics = self.roi_metrics
        cell_index = roi_metrics[roi_metrics.id == cell_specimen_id].cell_index.values[0]
        return cell_index

    def get_dff_traces(self):
        """Return dF/F traces as a DataFrame indexed by cell specimen ID.

        Wraps ``dff_traces_array`` in a DataFrame so traces can be looked up
        by global cell identifier rather than integer index.

        Returns
        -------
        pd.DataFrame
            DataFrame with index ``cell_specimen_id`` and a single column
            ``dff``, where each entry is a 1-D NumPy array of length
            ``n_timepoints``.
        """
        self._dff_traces = pd.DataFrame({'dff': [x for x in self.dff_traces_array]},
                                        index=pd.Index(self.cell_specimen_ids, name='cell_specimen_id'))
        return self._dff_traces

    dff_traces = LazyLoadable('_dff_traces', get_dff_traces)

    def get_events(self):
        """Return L0 event arrays as a DataFrame indexed by cell specimen ID.

        Wraps ``events_array`` in a DataFrame so events can be looked up by
        global cell identifier.  Returns ``None`` if no events file was found.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with index ``cell_specimen_id`` and a single column
            ``events``, where each entry is a 1-D NumPy array of length
            ``n_timepoints``, or ``None`` if events are unavailable.
        """
        self._events = pd.DataFrame({'events': [x for x in self.events_array]},
                                    index=pd.Index(self.cell_specimen_ids, name='cell_specimen_id'))
        return self._events

    events = LazyLoadable('_events', get_events)

    def get_pupil_area(self):
        """Load blink-corrected pupil area traces from the ``pupil_data/`` directory.

        Looks for a CSV file containing the session ID within
        ``<cache_dir>/pupil_data/``.  Aligns the pupil area vector to
        ``behavior_monitoring`` timestamps, truncating if within 5 frames.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns ``pupil_area`` (arbitrary units) and
            ``time`` (seconds), or ``None`` if no pupil data file is found or
            if the length mismatch is too large.
        """
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
        """Return the stimulus presentations table merged with image metadata.

        Renames ``flash_number`` → ``stimulus_presentations_id``, sets it as
        the index, then merges in ``stimulus_metadata`` (image name → index
        mapping, plus omitted-stimulus row).

        Returns
        -------
        pd.DataFrame
            One row per stimulus presentation (indexed by
            ``stimulus_presentations_id``) with columns including
            ``image_name``, ``image_index``, ``start_time``, ``stop_time``,
            and ``is_change``.
        """
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
        obj.get_max_projection()
        obj.get_average_image()
        obj.get_motion_correction()
        obj.get_dff_traces()
        obj.get_events()
        obj.get_pupil_area()
        obj.get_stimulus_presentations()
        # obj.get_extended_stimulus_presentations()

        return obj
