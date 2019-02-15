import os
import h5py
import platform
import numpy as np
import pandas as pd
import tempfile
import inspect

from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin
from visual_behavior.ophys.io.lims_api import VisualBehaviorLimsAPI
from visual_behavior.ophys.io.filesystem_api import VisualBehaviorFileSystemAPI
from pandas.util.testing import assert_frame_equal

class VisualBehaviorOphysSession(LazyPropertyMixin):

    def __init__(self, ophys_experiment_id, api=None, use_acq_trigger=False):

        self.ophys_experiment_id = ophys_experiment_id
        self.api = VisualBehaviorLimsAPI() if api is None else api
        self.use_acq_trigger = use_acq_trigger

        self.max_projection = LazyProperty(self.api.get_max_projection, ophys_experiment_id=self.ophys_experiment_id)
        self.stimulus_timestamps = LazyProperty(self.api.get_stimulus_timestamps, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.ophys_timestamps = LazyProperty(self.api.get_ophys_timestamps, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.metadata = LazyProperty(self.api.get_metadata, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.dff_traces = LazyProperty(self.api.get_dff_traces, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.roi_metrics = LazyProperty(self.api.get_roi_metrics, ophys_experiment_id=self.ophys_experiment_id)
        self.roi_masks = LazyProperty(self.api.get_roi_masks, ophys_experiment_id=self.ophys_experiment_id)
        self.cell_roi_ids = LazyProperty(self.api.get_cell_roi_ids, ophys_experiment_id=self.ophys_experiment_id)
        self.running_speed = LazyProperty(self.api.get_running_speed, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.stimulus_table = LazyProperty(self.api.get_stimulus_table, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.stimulus_template = LazyProperty(self.api.get_stimulus_template, ophys_experiment_id=self.ophys_experiment_id)
        self.stimulus_metadata = LazyProperty(self.api.get_stimulus_metadata, ophys_experiment_id=self.ophys_experiment_id)
        self.licks = LazyProperty(self.api.get_licks, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.rewards = LazyProperty(self.api.get_rewards, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.task_parameters = LazyProperty(self.api.get_task_parameters, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.extended_dataframe = LazyProperty(self.api.get_extended_dataframe, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        self.corrected_fluorescence_traces = LazyProperty(self.api.get_corrected_fluorescence_traces, ophys_experiment_id=self.ophys_experiment_id, use_acq_trigger=self.use_acq_trigger)
        # self.events = LazyProperty(self.api.get_events, ophys_experiment_id=self.ophys_experiment_id)
        self.average_image = LazyProperty(self.api.get_average_image, ophys_experiment_id=self.ophys_experiment_id)
        self.motion_correction = LazyProperty(self.api.get_motion_correction, ophys_experiment_id=self.ophys_experiment_id)

    def get_trials(self, columns=None, auto_rewarded=False, aborted=False):

        trials = self.extended_dataframe
        if 'trial' not in trials.columns:
            trials.insert(loc=0, column='trial', value=trials.index.values)

        if auto_rewarded == False:
            trials = trials[(trials.auto_rewarded != True)].reset_index()
            trials = trials.rename(columns={'level_0': 'original_trial_index'})
        if aborted == False:
            trials = trials[(trials.trial_type != 'aborted')].reset_index()
            trials = trials.rename(columns={'level_0': 'original_trial_index'})
        trials.rename(
            columns={'starttime': 'start_time', 'endtime': 'end_time', 'startdatetime': 'start_date_time',
                     'level_0': 'original_trial_index', 'color': 'trial_type_color'}, inplace=True)

        if columns is None:
            columns = ['trial', 'change_time', 'initial_image_name', 'change_image_name', 'trial_type', 'trial_type_color',
             'response', 'response_type', 'response_window', 'lick_times', 'response_latency', 'rewarded',
             'reward_times', 'reward_volume', 'reward_rate', 'start_time', 'end_time', 'trial_length', 'mouse_id', 'start_date_time']

        trials = trials[columns]

        return trials

    @property
    def lazy_properties(self):
        fields = []
        for member_name, member_object in inspect.getmembers(self.__class__):
            if inspect.isdatadescriptor(member_object) and member_name != '__weakref__':
                fields.append(member_name)
        return sorted(fields)

    def __eq__(self, other):

        field_set = set()
        for key, val in self.__dict__.items():
            if isinstance(val, LazyProperty):
                field_set.add(key) 
        for key, val in other.__dict__.items():
            if isinstance(val, LazyProperty):
                field_set.add(key) 

        try:
            for field in field_set: 
                x1, x2 = getattr(self, field), getattr(other, field)
                if isinstance(x1, pd.DataFrame):
                    assert_frame_equal(x1, x2)
                elif isinstance(x1, np.ndarray):
                    np.testing.assert_array_almost_equal(x1, x2)
                elif isinstance(x1, (dict, list)):
                    assert x1 == x2
                else:
                    raise Exception('Comparator not implemented')

        except NotImplementedError as e:
            self_implements_get_field = hasattr(self.api, getattr(type(self), field).getter_name)
            other_implements_get_field = hasattr(other.api, getattr(type(other), field).getter_name)
            assert self_implements_get_field == other_implements_get_field == False

        except (AssertionError, AttributeError) as e:
            return False

        return True


# def test_visbeh_ophys_data_set_events():
    
#     ophys_experiment_id = 702134928
#     api = VisualBehaviorLimsAPI_hackEvents(event_cache_dir='/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis/events')
#     data_set = VisualBehaviorOphysSession(ophys_experiment_id, api=api)

#     # Not round-tripped
#     data_set.events

import datetime
def test_visbeh_ophys_data_set(ophys_experiment_id, api):

    ophys_experiment_id = 789359614

    data_set = VisualBehaviorOphysSession(ophys_experiment_id, api=api)

    # TODO: need to improve testing here:
    # for _, row in data_set.roi_metrics.iterrows():
    #     print np.array(row.to_dict()['mask']).sum()
    # print
    # for _, row in data_set.roi_masks.iterrows():
    #     print np.array(row.to_dict()['mask']).sum()


    print data_set.api.get_foraging_id(ophys_experiment_id)
    # print data_set.metadata

    for key, val in data_set.task_parameters.iloc[0].to_dict().items():
        print key, val
    # data_set.extended_dataframe
    # data_set.corrected_fluorescence_traces
    # data_set.motion_correction

    # All sorts of assert relationships:
    # assert data_set.stimulus_template.shape == (8, 918, 1174)
    # assert len(data_set.licks) == 5941 and list(data_set.licks.columns) == ['frame', 'time']
    # assert len(data_set.rewards) == 138 and list(data_set.rewards.columns) == ['frame', 'time']
    # assert sorted(data_set.stimulus_metadata['image_category'].unique()) == sorted(data_set.stimulus_table['image_category'].unique())
    # assert sorted(data_set.stimulus_metadata['image_name'].unique()) == sorted(data_set.stimulus_table['image_name'].unique())
    # np.testing.assert_array_almost_equal(data_set.running_speed['time'], data_set.stimulus_timestamps)
    # assert len(data_set.cell_roi_ids) == len(data_set.dff_traces)
    # assert data_set.ophys_timestamps.shape == data_set.dff_traces.timestamps.values[0].shape
    # assert data_set.average_image.shape == data_set.max_projection.shape
    # assert data_set.metadata == {'stimulus_frame_rate': 60.0, 
    #                              'full_genotype': 'Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt', 
    #                              'ophys_experiment_id': 702134928, 
    #                              'session_type': None, 
    #                              'driver_line': 'Vip-IRES-Cre', 
    #                              'experiment_date': datetime.datetime(2018, 5, 24, 21, 27, 25), 
    #                              'ophys_frame_rate': 31.0, 
    #                              'imaging_depth': 175, 
    #                              'LabTracks_ID': '363887', 
    #                              'experiment_container_id': None, 
    #                              'targeted_structure': 'VISal', 
    #                              'reporter_line': 'Ai148(TIT2L-GC6f-ICL-tTA2)'}

if __name__ == '__main__':
    test_visbeh_ophys_data_set(702134928, VisualBehaviorLimsAPI())

# def test_get_trials():

#     ophys_experiment_id = 702134928 
#     data_set = VisualBehaviorOphysSession(ophys_experiment_id)
#     print data_set.get_trials()


# def test_plot_traces_heatmap():

#     from visual_behavior.visualization.ophys.experiment_summary_figures import plot_traces_heatmap
    
#     oeid = 702134928
#     data_set = VisualBehaviorOphysSession(oeid)

#     plot_traces_heatmap(data_set)



    # test_visbeh_ophys_data_set_events()
    # test_get_trials()
    # test_plot_traces_heatmap()
    






    # def get_timestamps(self):
    #     self._timestamps = pd.read_hdf(os.path.join(self.analysis_dir, 'timestamps.h5'), key='df', format='fixed')
    #     return self._timestamps

    # def get_metadata(self):
    #     self._metadata = pd.read_hdf(os.path.join(self.analysis_dir, 'metadata.h5'), key='df', format='fixed')
    #     return self._metadata
