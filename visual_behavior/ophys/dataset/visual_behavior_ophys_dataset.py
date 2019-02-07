import os
import h5py
import platform
import numpy as np
import pandas as pd
import tempfile
from allensdk.experimental.lazy_property import LazyProperty as LazyPropertyBase
from visual_behavior.ophys.io.lims_api import VisualBehaviorLimsAPI
from visual_behavior.ophys.io.filesystem_api import VisualBehaviorFileSystemAPI
from pandas.util.testing import assert_frame_equal
import inspect


class LazyProperty(LazyPropertyBase):
    
    def calculate(self, obj):
        return getattr(obj.api, self.getter_name)(ophys_experiment_id=obj.ophys_experiment_id, use_acq_trigger=obj.use_acq_trigger)

class VisualBehaviorOphysSession(object):

    max_projection = LazyProperty(api_method='get_max_projection')
    stimulus_timestamps = LazyProperty(api_method='get_stimulus_timestamps')
    ophys_timestamps = LazyProperty(api_method='get_ophys_timestamps')
    dff_traces = LazyProperty(api_method='get_dff_traces')
    roi_metrics = LazyProperty(api_method='get_roi_metrics')
    roi_masks = LazyProperty(api_method='get_roi_masks')
    cell_roi_ids = LazyProperty(api_method='get_cell_roi_ids')
    running_speed = LazyProperty(api_method='get_running_speed')
    stimulus_table = LazyProperty(api_method='get_stimulus_table')
    stimulus_template = LazyProperty(api_method='get_stimulus_template')
    stimulus_metadata = LazyProperty(api_method='get_stimulus_metadata')
    licks = LazyProperty(api_method='get_licks')
    rewards = LazyProperty(api_method='get_rewards')
    task_parameters = LazyProperty(api_method='get_task_parameters')
    extended_dataframe = LazyProperty(api_method='get_extended_dataframe')
    corrected_fluorescence_traces = LazyProperty(api_method='get_corrected_fluorescence_traces')
    events = LazyProperty(api_method='get_events')
    average_image = LazyProperty(api_method='get_average_image')
    motion_correction = LazyProperty(api_method='get_motion_correction')


    def __init__(self, ophys_experiment_id, api=None, use_acq_trigger=False):

        self.ophys_experiment_id = ophys_experiment_id
        self.api = VisualBehaviorLimsAPI() if api is None else api
        self.use_acq_trigger = use_acq_trigger

    def get_trials(self, columns=None, auto_rewarded=False, aborted=False):

        trials = self.extended_dataframe
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

        try:
            for field in set(self.lazy_properties).union(other.lazy_properties): 
                x1, x2 = getattr(self, field), getattr(other, field)
                if isinstance(x1, pd.DataFrame):
                    assert_frame_equal(x1, x2)
                elif isinstance(x1, np.ndarray):
                    np.testing.assert_array_almost_equal(x1, x2)
                elif isinstance(x1, (dict, list)):
                    assert x1 == x2
                else:
                    raise Exception('Comparator not implmeneted')

        except NotImplementedError as e:
            self_implements_get_field = hasattr(self.api, getattr(type(self), field).getter_name)
            other_implements_get_field = hasattr(other.api, getattr(type(other), field).getter_name)
            assert self_implements_get_field == other_implements_get_field == False

        except (AssertionError, AttributeError) as e:
            return False

        return True

def test_equal():

    oeid = 702134928
    d1 = VisualBehaviorOphysSession(oeid)
    d2 = VisualBehaviorOphysSession(oeid)

    assert d1 == d2




def test_visbeh_ophys_data_set_events():
    
    ophys_experiment_id = 702134928
    api = VisualBehaviorLimsAPI_hackEvents(event_cache_dir='/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis/events')
    data_set = VisualBehaviorOphysSession(ophys_experiment_id, api=api)

    # Not round-tripped
    data_set.events


def test_visbeh_ophys_data_set(ophys_experiment_id, api):

    data_set = VisualBehaviorOphysSession(ophys_experiment_id, api=api)

    # # Round tripped DataFrames:
    # data_set.roi_metrics
    # data_set.dff_traces
    # data_set.roi_masks
    # data_set.running_speed
    # data_set.stimulus_metadata
    # data_set.stimulus_table
    # data_set.licks
    # data_set.rewards
    # data_set.task_parameters
    # data_set.extended_dataframe
    # data_set.corrected_fluorescence_traces
    # data_set.motion_correction

    # # Round tripped ndarrays:
    # data_set.max_projection
    # data_set.cell_roi_ids
    # data_set.stimulus_timestamps
    # data_set.ophys_timestamps
    # data_set.stimulus_template
    # data_set.average_image

    # Not roud trip tested:

    # assert data_set.max_projection.shape == (512, 449)

def test_get_trials():

    ophys_experiment_id = 702134928 
    data_set = VisualBehaviorOphysSession(ophys_experiment_id)
    print data_set.get_trials()


def test_plot_traces_heatmap():

    from visual_behavior.visualization.ophys.experiment_summary_figures import plot_traces_heatmap
    
    oeid = 702134928
    data_set = VisualBehaviorOphysSession(oeid)

    plot_traces_heatmap(data_set)


if __name__ == '__main__':

    # test_visbeh_ophys_data_set(702134928, VisualBehaviorLimsAPI())
    # test_visbeh_ophys_data_set_events()
    # test_get_trials()
    # test_plot_traces_heatmap()
    test_equal()






    # def get_timestamps(self):
    #     self._timestamps = pd.read_hdf(os.path.join(self.analysis_dir, 'timestamps.h5'), key='df', format='fixed')
    #     return self._timestamps

    # def get_metadata(self):
    #     self._metadata = pd.read_hdf(os.path.join(self.analysis_dir, 'metadata.h5'), key='df', format='fixed')
    #     return self._metadata
