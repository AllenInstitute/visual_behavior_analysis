from __future__ import print_function

"""
Created on Saturday July 14 2018

@author: marinag
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
from psycopg2 import connect, extras  # NOQA: E402


def get_psql_dict_cursor(dbname='lims2', user='limsreader', host='limsdb2', password='limsro', port=5432):
    """Set up a connection to a psql db server with a dict cursor"""
    con = connect(dbname=dbname,
                  user=user,
                  host=host,
                  password=password,
                  port=port)
    con.set_session(readonly=True, autocommit=True)
    return con.cursor(cursor_factory=extras.RealDictCursor)


def save_data_as_h5(data, name, analysis_dir):
    f = h5py.File(os.path.join(analysis_dir, name + '.h5'), 'w')
    f.create_dataset('data', data=data)
    f.close()


def save_dataframe_as_h5(df, name, analysis_dir):
    df.to_hdf(os.path.join(analysis_dir, name + '.h5'), key='df', format='fixed')


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


def get_lims_id(lims_data):
    lims_id = lims_data.lims_id.values[0]
    return lims_id


def get_analysis_folder_name(lims_data):
    date = str(lims_data.experiment_date.values[0])[:10].split('-')
    specimen_driver_lines = lims_data.specimen_driver_line.values[0].split(';')
    if len(specimen_driver_lines) > 1:
        for i in range(len(specimen_driver_lines)):
            if 'S' in specimen_driver_lines[i]:
                specimen_driver_line = specimen_driver_lines[i]  # .split('-')[0]
            else:
                specimen_driver_line = specimen_driver_lines[0]
    else:
        specimen_driver_line = specimen_driver_lines[0]
    if specimen_driver_line == '':
        raise Exception('specimen_driver_line is empty! check why! This will result in "multiple analysis folders".')
    if lims_data.depth.values[0] is None:
        depth = 0
    else:
        depth = lims_data.depth.values[0]
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


def get_mouse_id(lims_data):
    mouse_id = int(lims_data.external_specimen_id.values[0])
    return mouse_id


def get_experiment_date(lims_data):
    experiment_date = str(lims_data.experiment_date.values[0])[:10].split('-')
    return experiment_date


def get_analysis_dir(lims_data, cache_dir=None, cache_on_lims_data=True):
    cache_dir = get_cache_dir(cache_dir=cache_dir)
    if 'analysis_dir' in lims_data.columns:
        return lims_data['analysis_dir'].values[0]
    analysis_dir = os.path.join(cache_dir, get_analysis_folder_name(lims_data))
    print(analysis_dir)
    if not os.path.exists(analysis_dir):
        print('Creating a new analysis folder')
        os.mkdir(analysis_dir)
    else:
        print('Analysis folder already exists!')
    # Check if more than one analysis folder exists
    allFiles = os.listdir(cache_dir)
    inds = [allFiles[i].find(str(np.squeeze(lims_data.lims_id.values))) for i in range(len(allFiles))]
    existingFolders = np.argwhere(np.array(inds) != -1)
    # Get the modification times of the existing analysis folders
    modifTimes = [os.path.getmtime(os.path.join(cache_dir, allFiles[np.squeeze(existingFolders[i])])) for i in range(len(existingFolders))]
    # Find all the old analysis folders
    indsOld = np.argsort(modifTimes)[:-1]
    oldFiles = [os.path.join(cache_dir, allFiles[np.squeeze(existingFolders[indsOld[i]])]) for i in range(len(indsOld))]
    # Remove old analysis folders
    for i in range(len(oldFiles)):
        print('Removing old analysis folder : %s' % oldFiles[i])
        shutil.rmtree(oldFiles[i])
    if cache_on_lims_data:
        lims_data.insert(loc=2, column='analysis_dir', value=analysis_dir)
    return analysis_dir


def get_ophys_session_dir(lims_data):
    ophys_session_dir = lims_data.ophys_session_dir.values[0]
    return ophys_session_dir


def get_ophys_experiment_dir(lims_data):
    lims_id = get_lims_id(lims_data)
    ophys_session_dir = get_ophys_session_dir(lims_data)
    ophys_experiment_dir = os.path.join(ophys_session_dir, 'ophys_experiment_' + str(lims_id))
    return ophys_experiment_dir


def get_demix_dir(lims_data):
    ophys_experiment_dir = get_ophys_experiment_dir(lims_data)
    demix_dir = os.path.join(ophys_experiment_dir, 'demix')
    return demix_dir


def get_processed_dir(lims_data):
    ophys_experiment_dir = get_ophys_experiment_dir(lims_data)
    processed_dir = os.path.join(ophys_experiment_dir, 'processed')
    return processed_dir


def get_segmentation_dir(lims_data):
    processed_dir = get_processed_dir(lims_data)
    segmentation_folder = [file for file in os.listdir(processed_dir) if 'segmentation' in file]
    segmentation_folder.sort()
    segmentation_dir = os.path.join(processed_dir, segmentation_folder[-1])
    return segmentation_dir


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


def get_sync_path(lims_data):
    #    import shutil
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
        #        print(sync_path, os.path.join(analysis_dir, sync_file))
        try:
            shutil.copy2(sync_path, os.path.join(analysis_dir, sync_file))
        except Exception as e:
            print(e)
            print('shutil.copy2 gave an error perhaps related to copying stat data... passing!')
            pass
    return sync_path


def get_sync_data(lims_data, use_acq_trigger):
    logger.info('getting sync data')
    sync_path = get_sync_path(lims_data)
    sync_dataset = SyncDataset(sync_path)
    # Handle mesoscope missing labels
    try:
        sync_dataset.get_rising_edges('2p_vsync')
    except ValueError:
        sync_dataset.line_labels = ['2p_vsync', '', 'stim_vsync', '', 'photodiode', 'acq_trigger', '', '',
                                    'behavior_monitoring', 'eye_tracking', '', '', '', '', '', '', '', '', '', '', '',
                                    '', '', '', '', '', '', '', '', '', '', 'lick_sensor']
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
    # print(len(frames_2p))
    if lims_data.rig.values[0][0] == 'M':  # if Mesoscope
        print('resampling mesoscope 2P frame times')
        roi_group = get_roi_group(lims_data)  # get roi_group order
        frames_2p = frames_2p[roi_group::4]  # resample sync times
    # print(len(frames_2p))
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


def get_timestamps_ophys(timestamps):
    timestamps_ophys = timestamps['ophys_frames']['timestamps']
    return timestamps_ophys


def get_metadata(lims_data, timestamps, core_data):
    timestamps_stimulus = get_timestamps_stimulus(timestamps)
    timestamps_ophys = get_timestamps_ophys(timestamps)
    metadata = OrderedDict()
    metadata['ophys_experiment_id'] = lims_data['experiment_id'].values[0]
    if lims_data.parent_session_id.values[0]:
        metadata['experiment_container_id'] = int(lims_data.parent_session_id.values[0])
    else:
        metadata['experiment_container_id'] = None
    metadata['targeted_structure'] = lims_data.structure.values[0]
    if lims_data.depth.values[0] is None:
        metadata['imaging_depth'] = None
    else:
        metadata['imaging_depth'] = int(lims_data.depth.values[0])

    specimen_driver_lines = lims_data.specimen_driver_line.values[0].split(';')
    if len(specimen_driver_lines) > 1:
        if 'Camk2a' in specimen_driver_lines[0]:
            specimen_driver_line = specimen_driver_lines[1]
        else:
            specimen_driver_line = specimen_driver_lines[0]
    else:
        specimen_driver_line = specimen_driver_lines[0]
    metadata['specimen_driver_line'] = specimen_driver_line
    metadata['cre_line'] = specimen_driver_line
    if len(lims_data['specimen_driver_line'].values[0].split(';')) > 1:
        metadata['reporter_line'] = lims_data['specimen_driver_line'].values[0].split(';')[0] + ';' + \
                                    lims_data['specimen_reporter_line'].values[0].split('(')[0]  # NOQA: E126
    else:
        metadata['reporter_line'] = lims_data['specimen_reporter_line'].values[0].split('(')[0]
    metadata['full_genotype'] = metadata['cre_line'] + ';' + metadata['reporter_line']
    metadata['session_type'] = 'behavior_session_' + lims_data.session_type.values[0].split('_')[-1]
    metadata['stage'] = core_data['metadata']['stage']
    metadata['donor_id'] = int(lims_data.external_specimen_id.values[0])
    metadata['experiment_date'] = str(lims_data.experiment_date.values[0])[:10]
    metadata['donor_id'] = int(lims_data.external_specimen_id.values[0])
    metadata['specimen_id'] = int(lims_data.specimen_id.values[0])
    # metadata['session_name'] = lims_data.session_name.values[0]
    metadata['session_id'] = int(lims_data.session_id.values[0])
    metadata['project_id'] = lims_data.project_id.values[0]
    metadata['rig'] = lims_data.rig.values[0]
    metadata['ophys_frame_rate'] = np.round(1 / np.mean(np.diff(timestamps_ophys)), 0)
    metadata['stimulus_frame_rate'] = np.round(1 / np.mean(np.diff(timestamps_stimulus)), 0)
    # metadata['eye_tracking_frame_rate'] = np.round(1 / np.mean(np.diff(self.timestamps_eye_tracking)),1)
    metadata = pd.DataFrame(metadata, index=[metadata['ophys_experiment_id']])
    return metadata


def save_metadata(metadata, lims_data):
    save_dataframe_as_h5(metadata, 'metadata', get_analysis_dir(lims_data))


def get_stimulus_pkl_path(lims_data):
    ophys_session_dir = get_ophys_session_dir(lims_data)
    # first try lims folder
    pkl_file = [file for file in os.listdir(ophys_session_dir) if '.pkl' in file]
    # remove tiny pkl files (less than 8KB)
    all_files = [os.path.join(ophys_session_dir, pkl_file[i]) for i in range(len(pkl_file))]
    all_files_valid = np.array(all_files)[[os.path.getsize(i) > 8000 for i in all_files]]
    pkl_file = [os.path.basename(all_files_valid[i]) for i in range(len(all_files_valid))]
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
    #    print(stimulus_pkl_path)
    return stimulus_pkl_path


def get_pkl(lims_data):
    #    import shutil
    stimulus_pkl_path = get_stimulus_pkl_path(lims_data)
    pkl_file = os.path.basename(stimulus_pkl_path)
    analysis_dir = get_analysis_dir(lims_data)
    if pkl_file not in os.listdir(analysis_dir):
        logger.info('moving %s to analysis dir', pkl_file)
        #        print(stimulus_pkl_path, os.path.join(analysis_dir, pkl_file))
        try:
            shutil.copy2(stimulus_pkl_path, os.path.join(analysis_dir, pkl_file))
        except Exception as e: #ValueError:
            print(e)
            print('shutil.copy2 gave an error perhaps related to copying stat data... passing!')
            pass

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


def get_task_parameters(core_data):
    task_parameters = {}
    task_parameters['blank_duration'] = core_data['metadata']['blank_duration_range'][0]
    task_parameters['stimulus_duration'] = core_data['metadata']['stim_duration']
    if 'omitted_flash_fraction' in core_data['metadata']['params'].keys():
        task_parameters['omitted_flash_fraction'] = core_data['metadata']['params']['omitted_flash_fraction']
    elif 'flash_omit_probability' in core_data['metadata']['params'].keys():
        task_parameters['omitted_flash_fraction'] = core_data['metadata']['params']['flash_omit_probability']
    else:
        task_parameters['omitted_flash_fraction'] = None
    task_parameters['response_window'] = [core_data['metadata']['response_window']]
    task_parameters['reward_volume'] = core_data['metadata']['rewardvol']
    task_parameters['stage'] = core_data['metadata']['stage']
    task_parameters['stimulus'] = core_data['metadata']['stimulus']
    task_parameters['stimulus_distribution'] = core_data['metadata']['stimulus_distribution']
    task_parameters['task'] = core_data['metadata']['task']
    task_parameters['n_stimulus_frames'] = core_data['metadata']['n_stimulus_frames']
    task_parameters = pd.DataFrame(task_parameters, columns=task_parameters.keys(), index=['params'])
    return task_parameters


def save_core_data_components(core_data, lims_data, timestamps_stimulus):
    rewards = core_data['rewards']
    save_dataframe_as_h5(rewards, 'rewards', get_analysis_dir(lims_data))

    running = core_data['running']
    running_speed = running.rename(columns={'speed': 'running_speed'})
    # filter to get rid of encoder spikes
    running_speed['running_speed'] = medfilt(running_speed.running_speed.values, kernel_size=5)
    save_dataframe_as_h5(running_speed, 'running_speed', get_analysis_dir(lims_data))

    licks = core_data['licks']
    save_dataframe_as_h5(licks, 'licks', get_analysis_dir(lims_data))

    stimulus_table = core_data['visual_stimuli'][:-10]  # ignore last 10 flashes
    if 'omitted_stimuli' in core_data.keys():
        if len(core_data['omitted_stimuli']) > 0:  # sometimes there is a key but empty values
            omitted_flash = core_data['omitted_stimuli'].copy()
            omitted_flash = omitted_flash[['frame']]
            omitted_flash['omitted'] = True
            flashes = stimulus_table.merge(omitted_flash, how='outer', on='frame')
            flashes['omitted'] = [True if omitted is True else False for omitted in flashes.omitted.values]
            flashes = flashes.sort_values(by='frame').reset_index().drop(columns=['index']).fillna(method='ffill')
            flashes = flashes[['frame', 'end_frame', 'time', 'image_category', 'image_name', 'omitted']]
            flashes = flashes.reset_index()
            flashes.image_name = ['omitted' if flashes.iloc[row].omitted == True else flashes.iloc[row].image_name for
                                  row
                                  in range(len(flashes))]
            # infer end time for omitted flashes as 16 frames after start frame (250ms*60Hz stim frame rate)
            flashes['end_frame'] = [flashes.loc[idx, 'end_frame'] if flashes.loc[idx, 'omitted'] == False else
                                    flashes.loc[idx, 'frame'] + 16 for idx in flashes.index.values]
            stimulus_table = flashes.copy()
        else:
            stimulus_table['omitted'] = False
    else:
        stimulus_table['omitted'] = False
    # rename columns to harmonize with visual coding and rebase timestamps to sync time
    stimulus_table.insert(loc=0, column='flash_number', value=np.arange(0, len(stimulus_table)))
    stimulus_table = stimulus_table.rename(columns={'frame': 'start_frame', 'time': 'start_time'})
    start_time = [timestamps_stimulus[start_frame] for start_frame in stimulus_table.start_frame.values]
    stimulus_table.start_time = start_time
    end_time = [timestamps_stimulus[int(end_frame)] for end_frame in stimulus_table.end_frame.values]
    stimulus_table.insert(loc=4, column='end_time', value=end_time)
    if 'level_0' in stimulus_table.keys():
        stimulus_table.drop(columns=['level_0'])
    save_dataframe_as_h5(stimulus_table, 'stimulus_table', get_analysis_dir(lims_data))

    task_parameters = get_task_parameters(core_data)
    save_dataframe_as_h5(task_parameters, 'task_parameters', get_analysis_dir(lims_data))


def get_trials(core_data):
    trials = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks=core_data['licks'],
        time=core_data['time'])
    return trials


def save_trials(trials, lims_data):
    save_dataframe_as_h5(trials, 'trials', get_analysis_dir(lims_data))


def get_visual_stimulus_data(pkl):
    try:
        images = foraging2.data_to_images(pkl)
    except (ValueError, KeyError):
        images = foraging.load_images(pkl)
    stimulus_template = images['images']
    stimulus_metadata = pd.DataFrame(images['image_attributes'])
    return stimulus_template, stimulus_metadata


def save_visual_stimulus_data(stimulus_template, stimulus_metadata, lims_data):
    save_dataframe_as_h5(stimulus_metadata, 'stimulus_metadata', get_analysis_dir(lims_data))
    save_data_as_h5(stimulus_template, 'stimulus_template', get_analysis_dir(lims_data))


def parse_mask_string(mask_string):
    # convert ruby json array ouput to python 2D array
    # needed for segmentation output prior to 10/10/17 due to change in how masks were saved
    mask = []
    row_length = -1
    for i in range(1, len(mask_string) - 1):
        c = mask_string[i]
        if c == '{':
            row = []
        elif c == '}':
            mask.append(row)
            if row_length < 1:
                row_length = len(row)
        elif c == 'f':
            row.append(False)
        elif c == 't':
            row.append(True)
    return np.asarray(mask)


def get_extract_json_file(experiment_id):
    QUERY = '''
    SELECT storage_directory || filename 
    FROM well_known_files 
    WHERE well_known_file_type_id = (SELECT id FROM well_known_file_types WHERE name = 'OphysExtractedTracesInputJson') 
    AND attachable_id = '{0}';
    '''  # NOQA: W291
    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(QUERY.format(experiment_id))
    return (lims_cursor.fetchall())  # return(lims_cursor.fetchone())


def get_input_extract_traces_json(lims_data):
    f = get_extract_json_file(lims_data['lims_id'].values[0])
    json_path = f[0]['?column?']
    json_file_name = json_path.split('/')[-1]
    processed_dir = get_processed_dir(lims_data)
    # print(json_path)
    if json_file_name in os.listdir(processed_dir):
        json_path = os.path.join(processed_dir, json_file_name)
    else:
        experiment_dir = get_ophys_experiment_dir(lims_data)
        json_path = os.path.join(experiment_dir, json_file_name)
    with open(json_path, 'rb') as w:
        jin = json.load(w)
    return jin


def get_roi_locations(lims_data):
    jin = get_input_extract_traces_json(lims_data)
    rois = jin["rois"]
    # get data out of json and into dataframe
    roi_locations_list = []
    for i in range(len(rois)):
        roi = rois[i]
        if roi['mask'][0] == '{':
            mask = parse_mask_string(roi['mask'])
        else:
            mask = roi["mask"]
        roi_locations_list.append([roi["id"], roi["x"], roi["y"], roi["width"], roi["height"], roi["valid"], mask])
    roi_locations = pd.DataFrame(data=roi_locations_list, columns=['id', 'x', 'y', 'width', 'height', 'valid', 'mask'])
    return roi_locations


def add_roi_ids_to_roi_metrics(roi_metrics, roi_locations):
    # add roi ids to objectlist
    ids = []
    for row in roi_metrics.index:
        minx = roi_metrics.iloc[row][' minx']
        miny = roi_metrics.iloc[row][' miny']
        id = roi_locations[(roi_locations.x == minx) & (roi_locations.y == miny)].id.values[0]
        ids.append(id)
    roi_metrics['roi_id'] = ids
    return roi_metrics


def get_roi_metrics(lims_data):
    # objectlist.txt contains metrics associated with segmentation masks
    segmentation_dir = get_segmentation_dir(lims_data)
    roi_metrics = pd.read_csv(os.path.join(segmentation_dir, 'objectlist.txt'))
    # get roi_locations and add unfiltered cell index
    roi_locations = get_roi_locations(lims_data)
    roi_names = np.sort(roi_locations.id.values)
    roi_locations['unfiltered_cell_index'] = [np.where(roi_names == id)[0][0] for id in roi_locations.id.values]
    # add cell ids to roi_metrics from roi_locations
    roi_metrics = add_roi_ids_to_roi_metrics(roi_metrics, roi_locations)
    # merge roi_metrics and roi_locations
    roi_metrics['id'] = roi_metrics.roi_id.values
    roi_metrics = pd.merge(roi_metrics, roi_locations, on='id')
    unfiltered_roi_metrics = roi_metrics
    # remove invalid roi_metrics
    roi_metrics = roi_metrics[roi_metrics.valid == True]
    # hack to get rid of cases with 2 rois at the same location
    for roi_id in roi_metrics.roi_id.values:
        roi_data = roi_metrics[roi_metrics.roi_id == roi_id]
        if len(roi_data) > 1:
            ind = roi_data.index
            roi_metrics = roi_metrics.drop(index=ind.values)
    # add filtered cell index
    cell_index = [np.where(np.sort(roi_metrics.roi_id.values) == id)[0][0] for id in
                  roi_metrics.roi_id.values]
    roi_metrics['cell_index'] = cell_index
    return roi_metrics, unfiltered_roi_metrics


def save_roi_metrics(roi_metrics, lims_data):
    save_dataframe_as_h5(roi_metrics, 'roi_metrics', get_analysis_dir(lims_data))


def save_unfiltered_roi_metrics(unfiltered_roi_metrics, lims_data):
    save_dataframe_as_h5(unfiltered_roi_metrics, 'unfiltered_roi_metrics', get_analysis_dir(lims_data))


def get_roi_ids(roi_metrics):
    roi_ids = np.unique(np.sort(roi_metrics.roi_id.values))
    return roi_ids


# def add_cell_specimen_ids_to_roi_metrics(lims_data, roi_metrics, cache_dir):
#     roi_ids = get_roi_ids(roi_metrics)
#     if cache_dir is not None:
#         matching_file = [file for file in os.listdir(cache_dir) if 'cell_matching_lookup_table' in file]
#         lookup = pd.read_csv(os.path.join(cache_dir, matching_file[0]))
#         df = lookup[lookup.ophys_experiment_id == int(lims_data.experiment_id.values[0])]
#         if (len(df) > 0):
#             if np.isnan(df.cell_specimen_id.values[0])==False:
#                 try:
#                     # if lookup file exists, and lookup table has entries for this experiment,
#                     # add cell_specimen_id values to roi_metrics and overwrite 'id' column with cell_specimen_ids
#                     print('this session has cell matching results, adding cell_specimen_ids and overwriting roi_metrics.id')
#                     # cell_specimen_ids = [int(df[df.cell_roi_id == roi_id].cell_specimen_id.values[0]) for roi_id in roi_ids]
#                     # cell_specimen_ids = np.asarray(cell_specimen_ids)
#                     roi_metrics['cell_specimen_id'] = [int(df[df.cell_roi_id == roi_id].cell_specimen_id.values[0]) for roi_id in
#                                                        roi_metrics.id.values]
#                     roi_metrics['id'] = roi_metrics['cell_specimen_id'].values
#                 except:
#                     print('something bad happened when trying to assign cell_specimen_ids using lookup table, setting to None')
#                     roi_metrics['cell_specimen_id'] = None
#             else:
#                 # no cell_specimen_ids in table, set cell_specimen_id to None
#                 print('no cell_specimen_ids for this experiment, using roi_ids')
#                 roi_metrics['cell_specimen_id'] = None
#         else:
#             # if lookup table doesnt exist, set cell_specimen_id to None
#             print('this experiment not in lookup table, using roi_ids')
#             roi_metrics['cell_specimen_id'] = None
#     else:
#         print('no cache_dir provided, cant look up cell_specimen_ids, using roi_ids')
#         roi_metrics['cell_specimen_id'] = None
#     return roi_metrics


def sql_lims_query(query):
    import psycopg2
    '''
    Performs SQL query to lims using any properly formated SQL query.

    :param query (str): properly formatted SQL query

    '''
    # connect to LIMS using read only
    conn = psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2", password="limsro", port=5432)
    cur = conn.cursor()

    # execute query
    cur.execute(query)

    # fetch results
    return (cur.fetchall())


def get_cell_specimen_ids_from_lims(mouse_id):
    '''
    Performs SQL query to lims to get matching specimen ids when they exist (for cell matching across imaging sessions)

    :param mouse_id (int): external_specimen_id, typically mouse_id

    '''
    # define sql lims query
    big_query = '''
    SELECT sp.external_specimen_name, vbec.id AS container_id, csd.id AS cell_specimen_id, cr.valid_roi, cr.id AS cell_roi_id
    FROM ophys_experiments_visual_behavior_experiment_containers oevbec
    JOIN visual_behavior_experiment_containers vbec ON vbec.id=oevbec.visual_behavior_experiment_container_id
    JOIN visual_behavior_container_runs runs ON runs.visual_behavior_experiment_container_id=vbec.id AND runs.current = 't'
    JOIN ophys_experiments oe ON oe.id=oevbec.ophys_experiment_id
    JOIN ophys_sessions os ON os.id=oe.ophys_session_id JOIN specimens sp ON sp.id=os.specimen_id
    LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
    LEFT JOIN cell_rois cr ON cr.ophys_cell_segmentation_run_id=ocsr.id
    LEFT JOIN specimens csd ON csd.id=cr.cell_specimen_id
    WHERE 
    cr.valid_roi = 't' AND 
    sp.external_specimen_name IN('{mouse_id}')
    ORDER BY 1,2,3,4,5;'''.format(mouse_id=mouse_id)  # NOQA: W291
    # query = big_query
    # Result of the query are
    # external_specimen_name,container_id,cell_specimen_id,valid_roi,cell_roi_id

    results = sql_lims_query(big_query)
    if len(results) == 0:
        print("No results found")
    return results


def add_cell_specimen_ids_to_roi_metrics(lims_data, roi_metrics, cache_dir):
    # roi_ids = get_roi_ids(roi_metrics)
    mouse_id = int(lims_data['external_specimen_id'])
    # Sql query lims to see if there are matching cell ids
    ids = get_cell_specimen_ids_from_lims(mouse_id)
    if len(ids) > 0:
        try:
            df = pd.DataFrame(ids, columns=['mouse', 'container_id', 'cell_specimen_id', 'valid_roi', 'cell_roi_id'])
            # match the ROI ids to the corresponding specimen ids
            roi_metrics['cell_specimen_id'] = [int(df[df.cell_roi_id == roi_id].cell_specimen_id.values) for roi_id in
                                               roi_metrics.id.values]
            # replace the id with the cell specimen ID
            roi_metrics['id'] = roi_metrics['cell_specimen_id'].values
        except Exception as e:
            print('something bad happened when trying to get cell specimen ids from lims, setting to None')
            print(e)
            roi_metrics['cell_specimen_id'] = None
    else:
        # if lims query returns nothing, set cell_specimen_id to None
        print('no cell specimen ids from lims, using roi_ids')
        roi_metrics['cell_specimen_id'] = None
    return roi_metrics


def get_cell_specimen_ids(roi_metrics):
    roi_ids = get_roi_ids(roi_metrics)
    if roi_metrics.cell_specimen_id.values[0] is None:
        cell_specimen_ids = roi_ids
    else:
        # make sure cell_specimen_ids are retrieved in the same order as roi_ids, as these correspond to order of trace and roi maskindices
        cell_specimen_ids = [int(roi_metrics[roi_metrics.roi_id == roi_id].cell_specimen_id.values[0]) for roi_id in
                             roi_ids]
        cell_specimen_ids = np.asarray(cell_specimen_ids)
    return cell_specimen_ids


def get_cell_indices(roi_metrics):
    cell_indices = np.unique(np.sort(roi_metrics.cell_index.values))
    return cell_indices


def get_cell_specimen_id_for_cell_index(cell_index, cell_specimen_ids):
    cell_specimen_id = cell_specimen_ids[cell_index]
    return cell_specimen_id


def get_cell_index_for_cell_specimen_id(cell_specimen_id, cell_specimen_ids):
    cell_index = np.where(cell_specimen_ids == cell_specimen_id)[0][0]
    return cell_index


def get_fov_dims(experiment_id):
    from allensdk.internal.api.ophys_lims_api import OphysLimsApi
    api = OphysLimsApi(experiment_id)
    #    print(api.get_metadata())
    image_metadata = api.get_metadata()
    return image_metadata


def get_roi_masks(roi_metrics, lims_data):
    # make roi_dict with ids as keys and roi_mask_array
    jin = get_input_extract_traces_json(lims_data)

    try:
        h = jin["image"]["height"]
        w = jin["image"]["width"]
    except Exception as e:
        print(e)
        image_metadata = get_fov_dims(lims_data['lims_id'].iloc[0])
        h = image_metadata['field_of_view_height']
        w = image_metadata['field_of_view_width']

    cell_specimen_ids = get_cell_specimen_ids(roi_metrics)
    roi_masks = {}
    for i, id in enumerate(cell_specimen_ids):
        m = roi_metrics[roi_metrics.id == id].iloc[0]
        mask = np.asarray(m['mask'])
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        binary_mask[int(m.y):int(m.y) + int(m.height), int(m.x):int(m.x) + int(m.width)] = mask
        roi_masks[int(id)] = binary_mask
    return roi_masks


def save_roi_masks(roi_masks, lims_data):
    f = h5py.File(os.path.join(get_analysis_dir(lims_data), 'roi_masks.h5'), 'w')
    for id, roi_mask in roi_masks.items():
        f.create_dataset(str(id), data=roi_mask)
    f.close()


def get_corrected_fluorescence_traces(roi_metrics, lims_data):
    file_path = os.path.join(get_ophys_experiment_dir(lims_data), 'demix',
                             str(get_lims_id(lims_data)) + '_demixed_traces.h5')
    g = h5py.File(file_path)
    corrected_fluorescence_traces = np.asarray(g['data'])
    valid_roi_indices = np.sort(roi_metrics.unfiltered_cell_index.values)
    corrected_fluorescence_traces = corrected_fluorescence_traces[valid_roi_indices]
    return corrected_fluorescence_traces


def save_corrected_fluorescence_traces(corrected_fluorescence_traces, roi_metrics, lims_data):
    traces_path = os.path.join(get_analysis_dir(lims_data), 'corrected_fluorescence_traces.h5')
    f = h5py.File(traces_path, 'w')
    for i, index in enumerate(get_cell_specimen_ids(roi_metrics)):
        f.create_dataset(str(index), data=corrected_fluorescence_traces[i])
    f.close()


def get_dff_traces(roi_metrics, lims_data):
    dff_path = os.path.join(get_ophys_experiment_dir(lims_data), str(get_lims_id(lims_data)) + '_dff.h5')
    g = h5py.File(dff_path)
    dff_traces = np.asarray(g['data'])
    #    print(dff_traces.shape)
    #    print(roi_metrics.unfiltered_cell_index)
    valid_roi_indices = np.sort(roi_metrics.unfiltered_cell_index.values)
    dff_traces = dff_traces[valid_roi_indices]
    print(dff_traces.shape)
    # find cells with NaN traces
    bad_cell_indices = []
    final_dff_traces = []
    for i, dff in enumerate(dff_traces):
        if np.isnan(dff).any():
            logger.info('NaN trace detected, removing cell_index: %d', i)
            bad_cell_indices.append(i)
        elif np.amax(dff) > 20:
            logger.info('outlier trace detected, removing cell_index %d', i)
            bad_cell_indices.append(i)
        else:
            final_dff_traces.append(dff)
    dff_traces = np.asarray(final_dff_traces)
    print(dff_traces.shape)
    roi_metrics = roi_metrics[roi_metrics.cell_index.isin(bad_cell_indices) == False]
    # reset cell index after removing bad cells
    cell_index = [np.where(np.sort(roi_metrics.roi_id.values) == id)[0][0] for id in
                  roi_metrics.roi_id.values]
    roi_metrics['cell_index'] = cell_index
    logger.info('length of traces: %d', dff_traces.shape[1])
    logger.info('number of segmented cells: %d', dff_traces.shape[0])
    return dff_traces, roi_metrics


def save_dff_traces(dff_traces, roi_metrics, lims_data):
    traces_path = os.path.join(get_analysis_dir(lims_data), 'dff_traces.h5')
    f = h5py.File(traces_path, 'w')
    for i, index in enumerate(get_cell_specimen_ids(roi_metrics)):
        f.create_dataset(str(index), data=dff_traces[i])
    f.close()


def save_timestamps(timestamps, dff_traces, core_data, roi_metrics, lims_data):
    # remove spurious frames at end of ophys session - known issue with Scientifica data
    if dff_traces.shape[1] < timestamps['ophys_frames']['timestamps'].shape[0]:
        difference = timestamps['ophys_frames']['timestamps'].shape[0] - dff_traces.shape[1]
        logger.info('length of ophys timestamps >  length of traces by %s frames, truncating ophys timestamps',
                    str(difference))

        timestamps['ophys_frames']['timestamps'] = timestamps['ophys_frames']['timestamps'][:dff_traces.shape[1]]
    # account for dropped ophys frames - a rare but unfortunate issue
    if dff_traces.shape[1] > timestamps['ophys_frames']['timestamps'].shape[0]:
        difference = timestamps['ophys_frames']['timestamps'].shape[0] - dff_traces.shape[1]
        logger.info('length of ophys timestamps <  length of traces by %s frames, truncating traces', str(difference))
        dff_traces = dff_traces[:, :timestamps['ophys_frames']['timestamps'].shape[0]]
        save_dff_traces(dff_traces, roi_metrics, lims_data)
    # make sure length of timestamps equals length of running traces
    running_speed = core_data['running'].speed.values
    if len(running_speed) < timestamps['stimulus_frames']['timestamps'].shape[0]:
        timestamps['stimulus_frames']['timestamps'] = timestamps['stimulus_frames']['timestamps'][:len(running_speed)]
    save_dataframe_as_h5(timestamps, 'timestamps', get_analysis_dir(lims_data))


def get_motion_correction(lims_data):
    csv_file = [file for file in os.listdir(get_processed_dir(lims_data)) if file.endswith('.csv')]
    csv_file = os.path.join(get_processed_dir(lims_data), csv_file[0])
    csv = pd.read_csv(csv_file, header=None)
    motion_correction = pd.DataFrame()
    motion_correction['x_corr'] = csv[1].values
    motion_correction['y_corr'] = csv[2].values
    return motion_correction


def save_motion_correction(motion_correction, lims_data):
    analysis_dir = get_analysis_dir(lims_data)
    save_dataframe_as_h5(motion_correction, 'motion_correction', analysis_dir)


def get_max_projection(lims_data):
    # max_projection = mpimg.imread(os.path.join(get_processed_dir(lims_data), 'max_downsample_4Hz_0.png'))
    max_projection = mpimg.imread(os.path.join(get_segmentation_dir(lims_data), 'maxInt_a13a.png'))
    return max_projection


def save_max_projections(lims_data):
    analysis_dir = get_analysis_dir(lims_data)
    # regular one
    if os.path.exists(os.path.join(get_processed_dir(lims_data), 'max_downsample_4Hz_0.png')):
        print('loading max')
        max_projection = mpimg.imread(os.path.join(get_processed_dir(lims_data), 'max_downsample_4Hz_0.png'))
        save_data_as_h5(max_projection, 'max_projection', analysis_dir)
        mpimg.imsave(os.path.join(get_analysis_dir(lims_data), 'max_intensity_projection.png'), arr=max_projection,
                     cmap='gray')
    else:
        print(
            'max_downsample_4Hz_0.png no longer exists, using normalized max projection from segmentation output instead')
        # contrast enhanced one
        max_projection = mpimg.imread(os.path.join(get_segmentation_dir(lims_data), 'maxInt_a13a.png'))
        save_data_as_h5(max_projection, 'max_projection', analysis_dir)
        mpimg.imsave(os.path.join(get_analysis_dir(lims_data), 'max_intensity_projection.png'),
                     arr=max_projection, cmap='gray')
    # contrast enhanced one
    max_projection = mpimg.imread(os.path.join(get_segmentation_dir(lims_data), 'maxInt_a13a.png'))
    save_data_as_h5(max_projection, 'normalized_max_projection', analysis_dir)
    mpimg.imsave(os.path.join(get_analysis_dir(lims_data), 'normalized_max_intensity_projection.png'),
                 arr=max_projection,
                 cmap='gray')


def get_average_image(lims_data):
    average_image = mpimg.imread(os.path.join(get_segmentation_dir(lims_data), 'avgInt_a1X.png'))
    return average_image


def save_average_image(average_image, lims_data):
    analysis_dir = get_analysis_dir(lims_data)
    save_data_as_h5(average_image, 'average_image', analysis_dir)
    mpimg.imsave(os.path.join(get_analysis_dir(lims_data), 'average_image.png'), arr=average_image,
                 cmap='gray')


def run_roi_validation(lims_data, cache_dir):
    processed_dir = get_processed_dir(lims_data)
    file_path = os.path.join(processed_dir, 'roi_traces.h5')

    with h5py.File(file_path) as g:
        roi_traces = np.asarray(g['data'])
        roi_names = np.asarray(g['roi_names'])

    experiment_dir = get_ophys_experiment_dir(lims_data)
    lims_id = get_lims_id(lims_data)

    dff_path = os.path.join(experiment_dir, str(lims_id) + '_dff.h5')

    with h5py.File(dff_path) as f:
        dff_traces_original = np.asarray(f['data'])

    roi_df = get_roi_locations(lims_data)
    roi_metrics, unfiltered_roi_metrics = get_roi_metrics(lims_data)
    roi_metrics = add_cell_specimen_ids_to_roi_metrics(lims_data, roi_metrics, cache_dir)
    roi_masks = get_roi_masks(roi_metrics, lims_data)
    dff_traces, roi_metrics = get_dff_traces(roi_metrics, lims_data)
    # cell_specimen_ids = get_cell_specimen_ids(roi_metrics)
    roi_ids = get_roi_ids(roi_metrics)
    max_projection = get_max_projection(lims_data)

    # cell_indices = {id: get_cell_index_for_cell_specimen_id(id, cell_specimen_ids) for id in cell_specimen_ids}
    cell_indices = {id: np.where(roi_ids == id)[0][0] for id in roi_ids}

    return roi_names, roi_df, roi_traces, dff_traces_original, roi_ids, cell_indices, roi_masks, roi_metrics, max_projection, dff_traces


def get_roi_validation(lims_data, cache_dir=None, save_plots=False):
    roi_names, roi_df, roi_traces, dff_traces_original, cell_specimen_ids, cell_indices, roi_masks, roi_metrics, max_projection, dff_traces = run_roi_validation(
        lims_data, cache_dir)

    roi_validation = plot_roi_validation(
        roi_names,
        roi_df,
        roi_traces,
        dff_traces_original,
        cell_specimen_ids,
        cell_indices,
        roi_masks,
        roi_metrics,
        max_projection,
        dff_traces,
    )

    return roi_validation


def save_roi_validation(roi_validation, lims_data):
    analysis_dir = get_analysis_dir(lims_data)

    for roi in roi_validation:
        fig = roi['fig']
        index = roi['index']
        id = roi['id']
        cell_index = roi['cell_index']

        save_figure(fig, (20, 10), analysis_dir, 'roi_validation',
                    str(index) + '_' + str(id) + '_' + str(cell_index))


def convert_level_1_to_level_2(lims_id, cache_dir=None, plot_roi_validation=True):
    logger.info('converting %d', lims_id)
    print('converting', lims_id)

    lims_data = get_lims_data(lims_id)

    # print(lims_data.iloc[0])

    analysis_dir = get_analysis_dir(lims_data, cache_on_lims_data=True, cache_dir=cache_dir)

    timestamps = get_timestamps(lims_data, analysis_dir)

    pkl = get_pkl(lims_data)
    timestamps_stimulus = get_timestamps_stimulus(timestamps)
    core_data = get_core_data(pkl, timestamps_stimulus)
    save_core_data_components(core_data, lims_data, timestamps_stimulus)

    metadata = get_metadata(lims_data, timestamps, core_data)
    save_metadata(metadata, lims_data)

    trials = get_trials(core_data)
    save_trials(trials, lims_data)

    stimulus_template, stimulus_metadata = get_visual_stimulus_data(pkl)
    save_visual_stimulus_data(stimulus_template, stimulus_metadata, lims_data)

    roi_metrics, unfiltered_roi_metrics = get_roi_metrics(lims_data)
    roi_metrics = add_cell_specimen_ids_to_roi_metrics(lims_data, roi_metrics, cache_dir)

    dff_traces, roi_metrics = get_dff_traces(roi_metrics, lims_data)
    save_dff_traces(dff_traces, roi_metrics, lims_data)

    roi_masks = get_roi_masks(roi_metrics, lims_data)
    save_roi_masks(roi_masks, lims_data)

    save_roi_metrics(roi_metrics, lims_data)
    save_unfiltered_roi_metrics(unfiltered_roi_metrics, lims_data)

    corrected_fluorescence_traces = get_corrected_fluorescence_traces(roi_metrics, lims_data)
    save_corrected_fluorescence_traces(corrected_fluorescence_traces, roi_metrics, lims_data)

    save_timestamps(timestamps, dff_traces, core_data, roi_metrics, lims_data)

    motion_correction = get_motion_correction(lims_data)
    save_motion_correction(motion_correction, lims_data)

    max_projection = get_max_projection(lims_data)
    save_max_projections(lims_data)

    average_image = get_average_image(lims_data)
    save_average_image(average_image, lims_data)

    if plot_roi_validation:
        roi_validation = get_roi_validation(lims_data, cache_dir)
        save_roi_validation(roi_validation, lims_data)

    logger.info('done converting')
    print('done converting')

    core_data.update(
        dict(
            lims_data=lims_data,
            timestamps=timestamps,
            metadata=metadata,
            roi_metrics=roi_metrics,
            roi_masks=roi_masks,
            dff_traces=dff_traces,
            motion_correction=motion_correction,
            max_projection=max_projection,
            average_image=average_image,
        ))  # flake8: noqa: F841
    return core_data




#%% For the cluster, call the function

cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'
plot_roi_validation = False
import sys
lims_id = sys.argv[1]
print(lims_id)
ophys_data = convert_level_1_to_level_2_pbs(lims_id, cache_dir, plot_roi_validation=False)
