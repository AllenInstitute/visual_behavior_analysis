#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:58:22 2019
@author: farzaneh.najafi
"""


#%% 
def get_lims_data(lims_id):
    ld = LimsDatabase(lims_id)
    lims_data = ld.get_qc_param()
    lims_data.insert(loc=2, column='experiment_id', value=lims_data.lims_id.values[0])
    lims_data.insert(loc=2, column='session_type',
                     value='behavior_' + lims_data.experiment_name.values[0].split('_')[-1])
    lims_data.insert(loc=2, column='ophys_session_dir', value=lims_data.datafolder.values[0][:-28])
    return lims_data

def get_ophys_session_dir(lims_data):
    ophys_session_dir = lims_data.ophys_session_dir.values[0]
    return ophys_session_dir


def get_stage(lims_id):
    #a = [str(all_sess.stage.values[i]).find('B') for i in range(len(all_sess))]
    #aa = all_sess[[a[i]==15 for i in range(len(a))]].experiment_id
    #lims_id = aa.iloc[25]
    #lims_id = 875587466
    
    lims_data = get_lims_data(lims_id)
    ophys_session_dir = get_ophys_session_dir(lims_data)
    
    # first try lims folder
    pkl_file = [file for file in os.listdir(ophys_session_dir) if '.pkl' in file]
    stimulus_pkl_path = os.path.join(ophys_session_dir, pkl_file[0])
    
    pkl = pd.read_pickle(stimulus_pkl_path)    
    stage = pkl['items']['behavior']['params']['stage']
    
    return stage

#pkl['items']['behavior']['stimuli']['images']
#image_set =np.unique([pkl['items']['behavior']['stimuli']['images']['set_log'][i][1] for i in range(l)])
#print image_set


#%% Adapted from "convert" code.

"""            
import os
import h5py
import json
import shutil
import platform
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from collections import OrderedDict

import matplotlib.image as mpimg  # NOQA: E402

import logging

logger = logging.getLogger(__name__)

#
# from ...translator import foraging2, foraging
# from ...translator.core import create_extended_dataframe
# from ..sync.process_sync import get_sync_data
# from ..plotting.summary_figures import save_figure, plot_roi_validation
# from .lims_database import LimsDatabase

# relative import doesnt work on cluster
from visual_behavior.ophys.io.lims_database import LimsDatabase  # NOQA: E402
from visual_behavior.translator import foraging2, foraging  # NOQA: E402
from visual_behavior.translator.core import create_extended_dataframe  # NOQA: E402
from visual_behavior.ophys.sync.sync_dataset import Dataset as SyncDataset  # NOQA: E402
from visual_behavior.ophys.sync.process_sync import filter_digital, calculate_delay  # NOQA: E402
from visual_behavior.visualization.ophys.summary_figures import plot_roi_validation  # NOQA: E402
from visual_behavior.visualization.utils import save_figure  # NOQA: E402


def get_cache_dir(cache_dir=None):
    if not cache_dir:
        if platform.system() == 'Linux':
            cache_dir = r'/allen/aibs/informatics/swdb2018/visual_behavior'
        else:
            cache_dir = r'\\allen\aibs\informatics\swdb2018\visual_behavior'
        return cache_dir
    else:
        return cache_dir


def get_lims_data(lims_id):
    ld = LimsDatabase(lims_id)
    lims_data = ld.get_qc_param()
    lims_data.insert(loc=2, column='experiment_id', value=lims_data.lims_id.values[0])
    lims_data.insert(loc=2, column='session_type',
                     value='behavior_' + lims_data.experiment_name.values[0].split('_')[-1])
    lims_data.insert(loc=2, column='ophys_session_dir', value=lims_data.datafolder.values[0][:-28])
    return lims_data


def get_analysis_folder_name(lims_data):
    date = str(lims_data.experiment_date.values[0])[:10].split('-')
    specimen_driver_lines = lims_data.specimen_driver_line.values[0].split(';')
    if len(specimen_driver_lines) > 1:
        for i in range(len(specimen_driver_lines)):
            if 'S' in specimen_driver_lines[i]:
                specimen_driver_line = specimen_driver_lines[i].split('-')[0]
    else:
        specimen_driver_line = specimen_driver_lines[0]
    if lims_data.depth.values[0] is None:
        depth = 0
    else:
        depth = lims_data.depth.values[0]
    date = str(lims_data.experiment_date.values[0])[:10].split('-')
    specimen_driver_line = lims_data.specimen_driver_line.values[0].split(';')
    if len(specimen_driver_line) > 1:
        specimen_driver_line = specimen_driver_line[0].split('-')[0]
    else:
        specimen_driver_line = specimen_driver_line[0]

    if lims_data.rig.values[0][0] == 'M':
        analysis_folder_name = str(lims_data.lims_id.values[0]) + '_' + \
                               str(lims_data.external_specimen_id.values[0]) + '_' + date[0][2:] + date[1] + date[
                                   2] + '_' + \
                               lims_data.structure.values[0] + '_' + str(depth) + '_' + \
                               specimen_driver_line + '_' + lims_data.rig.values[0] + \
                               '_' + lims_data.session_type.values[0]  # NOQA: E127
    else:
        analysis_folder_name = str(lims_data.lims_id.values[0]) + '_' + \
                               str(lims_data.external_specimen_id.values[0]) + '_' + date[0][2:] + date[1] + date[
                                   2] + '_' + \
                               lims_data.structure.values[0] + '_' + str(depth) + '_' + \
                               specimen_driver_line + '_' + lims_data.rig.values[0][3:5] + \
                               lims_data.rig.values[0][6] + '_' + lims_data.session_type.values[0]  # NOQA: E127

    return analysis_folder_name


def get_analysis_dir(lims_data, cache_dir=None, cache_on_lims_data=True):
    cache_dir = get_cache_dir(cache_dir=cache_dir)

    if 'analysis_dir' in lims_data.columns:
        return lims_data['analysis_dir'].values[0]

    analysis_dir = os.path.join(cache_dir, get_analysis_folder_name(lims_data))
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    if cache_on_lims_data:
        lims_data.insert(loc=2, column='analysis_dir', value=analysis_dir)
    return analysis_dir



def get_ophys_session_dir(lims_data):
    ophys_session_dir = lims_data.ophys_session_dir.values[0]
    return ophys_session_dir



def get_sync_path(lims_data):
    ophys_session_dir = get_ophys_session_dir(lims_data)
    analysis_dir = get_analysis_dir(lims_data)

    # First attempt
    sync_file = [file for file in os.listdir(ophys_session_dir) if 'sync' in file]
    if len(sync_file) > 0:
        sync_file = sync_file[0]
    else:
        json_path = [file for file in os.listdir(ophys_session_dir) if '_platform.json' in file][0]
        with open(os.path.join(ophys_session_dir, json_path)) as pointer_json:
            json_data = json.load(pointer_json)
            sync_file = json_data['sync_file']
    sync_path = os.path.join(ophys_session_dir, sync_file)

    if sync_file not in os.listdir(analysis_dir):
        logger.info('moving %s to analysis dir', sync_file)  # flake8: noqa: E999
        shutil.copy2(sync_path, os.path.join(analysis_dir, sync_file))
    return sync_path



def get_roi_group(lims_data):
    experiment_id = int(lims_data.experiment_id.values[0])
    ophys_session_dir = get_ophys_session_dir(lims_data)
    import json
    json_file = [file for file in os.listdir(ophys_session_dir) if ('SPLITTING' in file) and ('input.json' in file)]
    json_path = os.path.join(ophys_session_dir, json_file[0])
    with open(json_path, 'r') as w:
        jin = json.load(w)
    # figure out which roi_group the current experiment belongs to
    # plane_data = pd.DataFrame()
    for i, roi_group in enumerate(range(len(jin['plane_groups']))):
        group = jin['plane_groups'][roi_group]['ophys_experiments']
        for j, plane in enumerate(range(len(group))):
            expt_id = int(group[plane]['experiment_id'])
            if expt_id == experiment_id:
                expt_roi_group = i
    return expt_roi_group


def get_sync_data(lims_data, use_acq_trigger):
    logger.info('getting sync data')
    sync_path = get_sync_path(lims_data)
    sync_dataset = SyncDataset(sync_path)
    # Handle mesoscope missing labels
    try:
        sync_dataset.get_rising_edges('2p_vsync')
    except ValueError:
        sync_dataset.line_labels = ['2p_vsync', '', 'stim_vsync', '', 'photodiode', 'acq_trigger', '', '', 'behavior_monitoring', 'eye_tracking', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'lick_sensor']
        sync_dataset.meta_data['line_labels'] = sync_dataset.line_labels

    meta_data = sync_dataset.meta_data
    sample_freq = meta_data['ni_daq']['counter_output_freq']
    # 2P vsyncs
    vs2p_r = sync_dataset.get_rising_edges('2p_vsync')
    vs2p_f = sync_dataset.get_falling_edges(
        '2p_vsync', )  # new sync may be able to do units = 'sec', so conversion can be skipped
    vs2p_rsec = vs2p_r / sample_freq
    vs2p_fsec = vs2p_f / sample_freq
    if use_acq_trigger:  # if 2P6, filter out solenoid artifacts
        vs2p_r_filtered, vs2p_f_filtered = filter_digital(vs2p_rsec, vs2p_fsec, threshold=0.01)
        frames_2p = vs2p_r_filtered
    else:  # dont need to filter out artifacts in pipeline data
        frames_2p = vs2p_rsec
    # use rising edge for Scientifica, falling edge for Nikon http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
    # Convert to seconds - skip if using units in get_falling_edges, otherwise convert before doing filter digital
    # vs2p_rsec = vs2p_r / sample_freq
    # frames_2p = vs2p_rsec
    # stimulus vsyncs
    # vs_r = d.get_rising_edges('stim_vsync')
    vs_f = sync_dataset.get_falling_edges('stim_vsync')
    # convert to seconds
    # vs_r_sec = vs_r / sample_freq
    vs_f_sec = vs_f / sample_freq
    # vsyncs = vs_f_sec
    # add display lag
    monitor_delay = calculate_delay(sync_dataset, vs_f_sec, sample_freq)
    vsyncs = vs_f_sec + monitor_delay  # this should be added, right!?
    # line labels are different on 2P6 and production rigs - need options for both
    if 'lick_times' in meta_data['line_labels']:
        lick_times = sync_dataset.get_rising_edges('lick_1') / sample_freq
    elif 'lick_sensor' in meta_data['line_labels']:
        lick_times = sync_dataset.get_rising_edges('lick_sensor') / sample_freq
    else:
        lick_times = None
    if '2p_trigger' in meta_data['line_labels']:
        trigger = sync_dataset.get_rising_edges('2p_trigger') / sample_freq
    elif 'acq_trigger' in meta_data['line_labels']:
        trigger = sync_dataset.get_rising_edges('acq_trigger') / sample_freq
    if 'stim_photodiode' in meta_data['line_labels']:
        stim_photodiode = sync_dataset.get_rising_edges('stim_photodiode') / sample_freq
    elif 'photodiode' in meta_data['line_labels']:
        stim_photodiode = sync_dataset.get_rising_edges('photodiode') / sample_freq
    if 'cam1_exposure' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('cam1_exposure') / sample_freq
    elif 'cam1' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('cam1') / sample_freq
    elif 'eye_tracking' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('eye_tracking') / sample_freq
    if 'cam2_exposure' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('cam2_exposure') / sample_freq
    elif 'cam2' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('cam2') / sample_freq
    elif 'behavior_monitoring' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('behavior_monitoring') / sample_freq
    # some experiments have 2P frames prior to stimulus start - restrict to timestamps after trigger for 2P6 only
    if use_acq_trigger:
        frames_2p = frames_2p[frames_2p > trigger[0]]
    print(len(frames_2p))
    if lims_data.rig.values[0][0] == 'M':  # if Mesoscope
        print('resampling mesoscope 2P frame times')
        roi_group = get_roi_group(lims_data)  # get roi_group order
        frames_2p = frames_2p[roi_group::4]  # resample sync times
    print(len(frames_2p))
    logger.info('stimulus frames detected in sync: {}'.format(len(vsyncs)))
    logger.info('ophys frames detected in sync: {}'.format(len(frames_2p)))
    # put sync data in format to be compatible with downstream analysis
    times_2p = {'timestamps': frames_2p}
    times_vsync = {'timestamps': vsyncs}
    times_lick = {'timestamps': lick_times}
    times_trigger = {'timestamps': trigger}
    times_eye_tracking = {'timestamps': eye_tracking}
    times_behavior_monitoring = {'timestamps': behavior_monitoring}
    times_stim_photodiode = {'timestamps': stim_photodiode}
    sync_data = {'ophys_frames': times_2p,
                 'stimulus_frames': times_vsync,
                 'lick_times': times_lick,
                 'eye_tracking': times_eye_tracking,
                 'behavior_monitoring': times_behavior_monitoring,
                 'stim_photodiode': times_stim_photodiode,
                 'ophys_trigger': times_trigger,
                 }
    return sync_data


def get_timestamps(lims_data, analysis_dir):
    if '2P6' in analysis_dir:
        use_acq_trigger = True
    else:
        use_acq_trigger = False
    sync_data = get_sync_data(lims_data, use_acq_trigger)
    timestamps = pd.DataFrame(sync_data)
    return timestamps

def get_timestamps_stimulus(timestamps):
    timestamps_stimulus = timestamps['stimulus_frames']['timestamps']
    return timestamps_stimulus


def get_stimulus_pkl_path(lims_data):
    ophys_session_dir = get_ophys_session_dir(lims_data)
    # first try lims folder
    pkl_file = [file for file in os.listdir(ophys_session_dir) if '.pkl' in file]
    if len(pkl_file) > 0:
        stimulus_pkl_path = os.path.join(ophys_session_dir, pkl_file[0])
    else:  # then try behavior directory
        expt_date = get_experiment_date(lims_data)
        mouse_id = get_mouse_id(lims_data)
        pkl_dir = os.path.join(r'/allen/programs/braintv/workgroups/neuralcoding/Behavior/Data',
                               'M' + str(mouse_id), 'output')
        if os.name == 'nt':
            pkl_dir = pkl_dir.replace('/', '\\')
            pkl_dir = '\\' + pkl_dir
        pkl_file = [file for file in os.listdir(pkl_dir) if file.startswith(expt_date)][0]
        stimulus_pkl_path = os.path.join(pkl_dir, pkl_file)
    return stimulus_pkl_path



def get_pkl(lims_data):
    stimulus_pkl_path = get_stimulus_pkl_path(lims_data)
    pkl_file = os.path.basename(stimulus_pkl_path)
    analysis_dir = get_analysis_dir(lims_data)
    if pkl_file not in os.listdir(analysis_dir):
        logger.info('moving %s to analysis dir', pkl_file)
        shutil.copy2(stimulus_pkl_path, os.path.join(analysis_dir, pkl_file))
    logger.info('getting stimulus data from pkl')
    pkl = pd.read_pickle(stimulus_pkl_path)
    # from visual_behavior.translator.foraging2 import data_to_change_detection_core
    # core_data = data_to_change_detection_core(pkl)
    # logger.info('visual frames in pkl file:', len(core_data['time']))
    return pkl


def get_core_data(pkl, timestamps_stimulus):
    try:
        core_data = foraging.data_to_change_detection_core(pkl, time=timestamps_stimulus)
    except KeyError:
        core_data = foraging2.data_to_change_detection_core(pkl, time=timestamps_stimulus)
    return core_data


#%%
    
lims_id = list_mesoscope_exp[i]


lims_data = get_lims_data(lims_id)
analysis_dir = get_analysis_dir(lims_data, cache_dir=None, cache_on_lims_data=True)
timestamps = get_timestamps(lims_data, analysis_dir)
timestamps_stimulus = get_timestamps_stimulus(timestamps)
pkl = get_pkl(lims_data)
core_data = get_core_data(pkl, timestamps_stimulus)

metadata['stage'] = core_data['metadata']['stage']

"""

