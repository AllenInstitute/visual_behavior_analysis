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


def eye_tracking_timestamps_from_sync(ophys_experiment_id):
    sync_timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_eyetracking_timestamps = sync_timestamps['eye_tracking'].timestamps
    return sync_eyetracking_timestamps


def behaviormon_timestamps_from_sync(ophys_experiment_id):
    sync_timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_behavior_timestamps = sync_timestamps['behavior_monitoring'].timestamps
    return sync_behavior_timestamps


def photodiode_timestamps_from_sync(ophys_experiment_id):
    sync_timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_photodiode_timestamps = sync_timestamps['stim_photodiode'].timestamps
    return sync_photodiode_timestamps


def lick_timestamps_from_sync(ophys_experiment_id):
    sync_timestamps = get_sync_timestamps(ophys_experiment_id)
    sync_lick_timestamps = sync_timestamps['lick_times'].timestamps
    return sync_lick_timestamps


def sync_timestamps_df(ophys_experiment_id):
    timestamps = get_sync_timestamps(ophys_experiment_id)
    df = pd.DataFrame({'ophys_experiment_id': ophys_experiment_id,
                       'ophys_session_id': loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id),
                       'source': 'sync',
                       'physio': len(timestamps['ophys_frames'].timestamps),
                       'stimulus': len(timestamps['stimulus_frames'].timestamps),
                       'running': len(timestamps['stimulus_frames'].timestamps),
                       'eye_tracking': len(timestamps['eye_tracking'].timestamps),
                       'behavior_mon': len(timestamps['behavior_monitoring'].timestamps),
                       'licks': len(timestamps['lick_times'].timestamps),
                       'stim_photodiode': len(timestamps['stim_photodiode'].timestamps),
                       'ophys_trigger': len(timestamps['ophys_trigger'].timestamps)}, index=[0])
    return df


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


def pkl_timestamps_df(ophys_experiment_id):
    ophys_session_id = loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    pkl_data = loading.load_session_pkl_coredata(ophys_session_id)
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "ophys_session_id": ophys_session_id,
                       "source": "pkl",
                       "stimulus": len(pkl_data["time"]),
                       "running": len(pkl_data["time"]),
                       "licks": len(pkl_data["licks"])}, index=[0])
    return df


def framecount_from_avi(avi_filepath):
    avi_video = Movie(avi_filepath)
    avi_framecount = avi_video.frame_count
    return avi_framecount


def behavior_mon_framecount_from_avi(ophys_session_id):
    avi_filepath = utilities.get_wkf_behavior_avi_filepath(ophys_session_id)
    avi_framecount = framecount_from_avi(avi_filepath)
    return avi_framecount


def eye_tracking_framecount_from_avi(ophys_session_id):
    avi_filepath = utilities.get_wkf_eye_tracking_avi_filepath(ophys_session_id)
    avi_framecount = framecount_from_avi(avi_filepath)
    return avi_framecount


def avi_timestamps_df(ophys_experiment_id):
    ophys_session_id = loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "ophys_session_id": ophys_session_id,
                       "source": "avi",
                       "eye_tracking": eye_tracking_framecount_from_avi(ophys_session_id),
                       "behavior_mon": behavior_mon_framecount_from_avi(ophys_session_id)}, index=[0])
    return df


def mcm_timestamps_df(ophys_experiment_id):
    ophys_session_id = loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "ophys_session_id": ophys_session_id,
                       "source": 'mcm',
                       "physio": physio_frames_from_motion_corrected_movie(ophys_experiment_id)}, index=[0])
    return df


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


def VBA_timestamps_df(ophys_experiment_id):
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    try:
        eye_tracking = len(np.asarray(dataset.eye_tracking["time"]))
    except RuntimeError:
        eye_tracking = np.nan
    try:
        behavior_mon = len(dataset.behavior_movie_timestamps)
    except RuntimeError:
        behavior_mon = np.nan
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "ophys_session_id": loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id),
                       "source": "VBA",
                       "eye_tracking": eye_tracking,
                       "behavior_mon": behavior_mon}, index=[0])
    return df


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


def running_timestamps_from_SDK(ophys_experiment_id):
    SDK_dataset = BehaviorOphysSession.from_lims(ophys_experiment_id)
    timestamps = SDK_dataset.running_speed[0]
    return timestamps


def eye_tracking_timestamps_from_SDK(ophys_experiment_id):
    SDK_dataset = BehaviorOphysSession.from_lims(ophys_experiment_id)
    timestamps = np.asarray(SDK_dataset.eye_tracking["time"])
    return timestamps


def SDK_timestamps_df(ophys_experiment_id):
    SDK_dataset = BehaviorOphysSession.from_lims(ophys_experiment_id)
    physio = len(SDK_dataset.ophys_timestamps)
    physio_dff = len(SDK_dataset.dff_traces.dff.values[0])
    stimulus = len(SDK_dataset.stimulus_timestamps)
    running = len(SDK_dataset.running_speed[0])
    try:
        licks = len(SDK_dataset.licks)
    except ValueError:
        licks = np.nan
    try:
        eye_tracking = len(SDK_dataset.eye_tracking["time"])
    except RuntimeError:
        eye_tracking = np.nan
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "ophys_session_id": loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id),
                       "source": "SDK",
                       "physio": physio,
                       "physio_dff": physio_dff,
                       "stimulus": stimulus,
                       "running": running,
                       "eye_tracking": eye_tracking,
                       "licks": licks}, index=[0])
    return df


def running_timestamps_df(ophys_experiment_id):
    ophys_session_id = loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "ophys_session_id": loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id),
                       "datastream": "running",
                       "sync": len(stim_timestamps_from_sync(ophys_experiment_id)),
                       "pkl": len(running_timestamps_from_pkl(ophys_session_id)),
                       "SDK": len(running_timestamps_from_SDK(ophys_experiment_id))}, index=[0])
    df = datastream_mismatch_row(df, timestamp_columns_dict["running"])
    return df


def eye_tracking_timestamps_df(ophys_experiment_id):
    ophys_session_id = loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "ophys_session_id": ophys_session_id,
                       "datastream": "eye_tracking",
                       "sync": len(eye_tracking_timestamps_from_sync(ophys_experiment_id)),
                       "avi": eye_tracking_framecount_from_avi(ophys_session_id),
                       "SDK": len(eye_tracking_timestamps_from_SDK(ophys_experiment_id)),
                       "VBA": len(eye_tracking_timestamps_from_VBAdataset(ophys_experiment_id))}, index=[0])
    df = datastream_mismatch_row(df, timestamp_columns_dict["eye_tracking"])
    return df


def behavior_monitoring_timestamps_df(ophys_experiment_id):
    ophys_session_id = loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "ophys_session_id": ophys_session_id,
                       "datastream": "behavior_monitoring",
                       "sync": len(behaviormon_timestamps_from_sync(ophys_experiment_id)),
                       "avi": behavior_mon_framecount_from_avi(ophys_session_id),
                       "VBA": len(behavior_mon_timestamps_from_VBAdataset(ophys_experiment_id))}, index=[0])
    df = datastream_mismatch_row(df, timestamp_columns_dict["behavior_monitoring"])
    return df


def stimulus_timestamps_df(ophys_experiment_id):
    ophys_session_id = loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "ophys_session_id": ophys_session_id,
                       "datastream": "stimulus",
                       "sync": len(stim_timestamps_from_sync(ophys_experiment_id)),
                       "pkl": len(stim_timestamps_from_pkl(ophys_session_id)),
                       "SDK": len(stim_timestamps_from_SDK(ophys_experiment_id))}, index=[0])
    df = datastream_mismatch_row(df, timestamp_columns_dict["stimulus"])
    return df


def physio_timestamps_df(ophys_experiment_id):
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "ophys_session_id": loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id),
                       "datastream": "physio",
                       "sync": len(physio_timestamps_from_sync(ophys_experiment_id)),
                       "SDK": len(physio_timestamps_from_SDK(ophys_experiment_id)),
                       "SDK_dff": dff_trace_length_from_SDK(ophys_experiment_id),
                       "mcm": physio_frames_from_motion_corrected_movie(ophys_experiment_id)}, index=[0])
    df = datastream_mismatch_row(df, timestamp_columns_dict["physio"])
    return df


def lick_timestamps_df(ophys_experiment_id, stage_name=False):
    ophys_session_id = loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    if stage_name == True:
        df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                           "ophys_session_id": ophys_session_id,
                           "datastream": "lick",
                           "stage_name": loading.get_lims_experiment_info(ophys_experiment_id)["stage_name_lims"][0],
                           "sync": len(lick_timestamps_from_sync(ophys_experiment_id)),
                           "pkl": len(lick_timestamps_from_pkl(ophys_session_id)),
                           "SDK": len(lick_timestamps_from_SDK(ophys_experiment_id))}, index=[0])
    else:
        df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                           "ophys_session_id": ophys_session_id,
                           "datastream": 'lick',
                           "sync": len(lick_timestamps_from_sync(ophys_experiment_id)),
                           "pkl": len(lick_timestamps_from_pkl(ophys_session_id)),
                           "SDK": len(lick_timestamps_from_SDK(ophys_experiment_id))}, index=[0])
    df = datastream_mismatch_row(df, timestamp_columns_dict["running"])
    return df


def experiment_timestamps_df(ophys_experiment_id):
    df = melt_source_timestamp_df(SDK_timestamps_df(ophys_experiment_id))
    df = df.append(melt_source_timestamp_df(sync_timestamps_df(ophys_experiment_id)))
    df = df.append(melt_source_timestamp_df(VBA_timestamps_df(ophys_experiment_id)))
    df = df.append(melt_source_timestamp_df(pkl_timestamps_df(ophys_experiment_id)))
    df = df.append(melt_source_timestamp_df(avi_timestamps_df(ophys_experiment_id)))
    df = df.append(melt_source_timestamp_df(mcm_timestamps_df(ophys_experiment_id)))
    df = df.sort_values("datastream").reset_index(drop=True)
    df.loc[df["datastream"] == "physio_dff", ["source"]] = "SDK_dff"
    df.loc[df["datastream"] == "physio_dff", ["datastream"]] = "physio"
    df = diff_from_rawdata_column(df)
    df = datastream_mismatch_column(df)
    return df


timestamp_columns_dict = {"physio": ["sync", "SDK", "SDK_dff", "mcm"],
                          "stimulus": ["sync", "pkl", "SDK"],
                          "running": ["sync", "pkl", "SDK"],
                          "eye_tracking": ["sync", "avi", "SDK", "VBA"],
                          "behavior_monitoring": ["sync", "avi", "VBA"],
                          "lick": ["sync", "pkl", "SDK"]}


def datastream_mismatch_row(dataframe, timestamp_columns_list):
    timestamp_col_df = dataframe[timestamp_columns_list]
    unique_timestamp_numbers = timestamp_col_df.nunique(axis=1)[0]
    if unique_timestamp_numbers == 1:
        dataframe["mismatch_present"] = False
    else:
        dataframe["mismatch_present"] = True
    return dataframe


def datastream_mismatch_column(dataframe):
    dataframe["mismatch_present"] = dataframe["diff_from_rawdata"].apply(lambda x: False if x ==0.0 else (np.nan if  pd.isnull(x) else True))
    return dataframe


def melt_source_timestamp_df(dataframe):
    melted_df = pd.melt(dataframe,
                        id_vars=["ophys_experiment_id", "ophys_session_id", 'source'],
                        var_name="datastream",
                        value_name="num_timestamps")
    return melted_df


def melt_datastream_timestamp_df(dataframe):
    melted_df = pd.melt(dataframe,
                        id_vars=["ophys_experiment_id", "ophys_session_id", "mismatch_present"],
                        var_name="source",
                        value_name="num_timestamps")
    return melted_df


def diff_from_rawdata_column(dataframe):
    rawdata_ts = get_datastream_rawdata_ts(dataframe)
    dataframe = pd.merge(dataframe, rawdata_ts, how="left", on="datastream")
    dataframe["diff_from_rawdata"] = dataframe["rawdata_timestamps"] - dataframe["num_timestamps"]
    dataframe = dataframe.drop(columns=["rawdata_timestamps"])
    return dataframe


def unify_source_rawdata_column_values(dataframe):
    dataframe.loc[dataframe["source"].isin(["avi", "pkl", "mcm"]), ["source"]] = "raw_data"
    return dataframe


def get_datastream_rawdata_ts(dataframe):
    rawdata_timestamps = dataframe.loc[dataframe["source"].isin(["avi", "pkl", "mcm"]), ["datastream", "num_timestamps"]]
    rawdata_timestamps = rawdata_timestamps.rename(columns={"num_timestamps": "rawdata_timestamps"})
    return rawdata_timestamps


def pivot_experiment_timestamp_df(dataframe):
    dataframe["datastream_source"] = dataframe["datastream"] + "_" + dataframe["source"]
    dataframe = dataframe.drop(columns=["ophys_session_id", "source", "datastream",
                                        "num_timestamps", "mismatch_present"])
    pivot = dataframe.pivot(index="ophys_experiment_id", columns="datastream_source", values="diff_from_rawdata").reset_index().rename_axis(None, axis=1)
    return pivot


def dataframe_for_mismatch_present_heatmap_plot(dataframe):
    dataframe["datastream_source"] = dataframe["datastream"] + "_" + dataframe["source"]
    dataframe = dataframe.drop(columns=["ophys_session_id", "source", "datastream",
                                        "num_timestamps", "diff_from_rawdata"])
    dataframe["mismatch_present"] = dataframe["mismatch_present"].apply(lambda x: 0 if x == False else 1)
    dataframe = dataframe.pivot(index="ophys_experiment_id", columns="datastream_source")
    return dataframe


def identify_sync_line_mislabel(pivoted_dataframe):
    pivoted_dataframe["sync_line_mislabel"] = (pivoted_dataframe["behavior_mon_sync"] != 0) & (pivoted_dataframe["eye_tracking_sync"] != 0) & (pivoted_dataframe["behavior_mon_sync"] == -(pivoted_dataframe["eye_tracking_sync"]))
    return pivoted_dataframe
