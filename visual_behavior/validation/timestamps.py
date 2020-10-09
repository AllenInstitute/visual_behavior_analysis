import numpy as np
import pandas as pd

from visual_behavior.utilities import Movie
from visual_behavior.data_access import loading
from visual_behavior.data_access import utilities
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession


def get_sync_timestamps(ophys_experiment_id):
    lims_data = utilities.get_lims_data(ophys_experiment_id)
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    timestamps = utilities.get_timestamps(lims_data, dataset.analysis_dir)
    return timestamps


def stim_timestamps_from_sync(ophys_experiment_id):
    sync_timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_stim_timestamps = sync_timestamps["stimulus_frames"].timestamps
    return sync_stim_timestamps


def physio_timestamps_from_sync(ophys_experiment_id):
    sync_timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_physio_timestamps = sync_timestamps["ophys_frames"].timestamps
    return sync_physio_timestamps


def eyetracking_timestamps_from_sync(ophys_experiment_id):
    sync_timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_eyetracking_timestamps = sync_timestamps['eye_tracking'].timestamps
    return sync_eyetracking_timestamps


def behaviormon_timestamps_from_sync(ophys_experiment_id):
    sync_timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_behavior_timestamps = sync_timestamps['behavior_minotring'].timestamps
    return sync_behavior_timestamps


def photodiode_timestamps_from_sync(ophys_experiment_id):
    sync_timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_photodiode_timestamps = sync_timestamps['stim_photodiode'].timestamps
    return sync_photodiode_timestamps


def lick_timestamps_from_sync(ophys_experiment_id):
    sync_timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_lick_timestamps = sync_timestamps['lick_times'].timestamps
    return sync_lick_timestamps


def sync_timestamp_df(ophys_experiment_id):
    timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_timestamp_df = pd.DataFrame({'ophys_frames': len(timestamps['ophys_frames'].timestamps),
                                      'stimulus_frames': len(timestamps['stimulus_frames'].timestamps),
                                      'eye_tracking': len(timestamps['eye_tracking'].timestamps),
                                      'behavior_mon': len(timestamps['behavior_monitoring'].timestamps),
                                      'lick_times': len(timestamps['lick_times'].timestamps),
                                      'stim_photodiode': len(timestamps['stim_photodiode'].timestamps),
                                      'ophys_trigger': len(timestamps['ophys_trigger'].timestamps)},
                                     index=[0])
    return sync_timestamp_df


def stim_timestamps_from_pkl(ophys_session_id):
    pkl_data = loading.load_session_pkl_coredata(ophys_session_id)
    pkl_stim_timestamps = pkl_data["time"]
    return pkl_stim_timestamps


def lick_timestamps_from_pkl(ophys_session_id):
    pkl_data = loading.load_session_pkl_coredata(ophys_session_id)
    pkl_lick_timestamps = pkl_data["licks"]
    return pkl_lick_timestamps


def running_timestamps_from_pkl(ophys_session_id):
    pkl_data = loading.load_session_pkl_coredata(ophys_session_id)
    pkl_running_timestamps = pkl_data["time"]
    return pkl_running_timestamps


def pkl_timestamps_df(ophys_session_id):
    pkl_data = loading.load_session_pkl_coredata(ophys_session_id)
    pkl_timestamp_df = pd.DataFrame({"visual_stim_timestamps": len(pkl_data["time"]),
                                     "running_timestamps": len(pkl_data["time"]),
                                     "licks": len(pkl_data["licks"])}, index=[0])
    return pkl_timestamp_df


def framecount_from_avi(avi_filepath):
    avi_video = Movie(avi_filepath)
    avi_framecount = avi_video.frame_count
    return avi_framecount


def behavior_mon_framecount_from_avi(ophys_session_id):
    avi_filepath = utilities.get_wkf_behavior_avi_filepath
    avi_framecount = framecount_from_avi(avi_filepath)
    return avi_framecount


def eye_tracking_framecount_from_avi(ophys_session_id):
    avi_filepath = utilities.get_wkf_eye_tracking_avi_filepath
    avi_framecount = framecount_from_avi(avi_filepath)
    return avi_framecount


def physio_frames_from_motion_corrected_movie(ophys_experiment_id):
    motion_corrected_movie = loading.load_motion_corrected_movie(ophys_experiment_id)
    physio_frames = len(motion_corrected_movie)
    return physio_frames


def behavior_mon_timestamps_from_VBAdataset(ophys_experiment_id):
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    timestamps = dataset.behavior_movie_timestamps
    return timestamps


def eye_tracking_timestamps_from_VBAdataset(ophys_experiment_id):
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    timestamps = np.asarray(dataset.eye_tracking["time"])
    return timestamps


def physio_timestamps_from_SDK(ophys_experiment_id):
    SDK_dataset = BehaviorOphysSession.from_lims(ophys_experiment_id)
    timestamps = SDK_dataset.ophys_timestamps
    return timestamps


def stim_timestamps_from_SDK(ophys_experiment_id):
    SDK_dataset = BehaviorOphysSession.from_lims(ophys_experiment_id)
    timestamps = SDK_dataset.stimulus_timestamps
    return timestamps


def dff_trace_length_from_SDK(ophys_experiment_id):
    SDK_dataset = BehaviorOphysSession.from_lims(ophys_experiment_id)
    num_timestamps = len(SDK_dataset.dff_traces.dff.values[0])
    return num_timestamps


def lick_timestamps_from_SDK(ophys_experiment_id):
    SDK_dataset = BehaviorOphysSession.from_lims(ophys_experiment_id)
    timestamps = SDK_dataset.licks
    return timestamps


def stimulus_timestamps_df(ophys_session_id):
    ophys_experiment_id = loading.get_ophys_experiment_id_for_ophys_session_id(ophys_session_id)    
    sync_timestamps = len(stim_timestamps_from_sync(ophys_experiment_id))
    pkl_timestamps = len(stim_timestamps_from_pkl(ophys_session_id))
    sdk_timestamps = len(stim_timestamps_from_SDK)
    df = pd.DataFrame({"sync": sync_timestamps,
                        "pkl": pkl_timestamps, 
                        "SDK": sdk_timestamps},index=[0])
    return df


def physio_timestamps_df(ophys_experiment_id):
    sync_timestamps = len(physio_timestamps_from_sync(ophys_experiment_id))
    sdk_timestamps = len(physio_timestamps_from_SDK(ophys_experiment_id))
    mcm_timestamps = physio_frames_from_motion_corrected_movie(ophys_experiment_id)
    df = pd.DataFrame({"sync": sync_timestamps,
                        "SDK": sdk_timestamps,
                        "mcm": mcm_timestamps}, index=[0])
    return df
