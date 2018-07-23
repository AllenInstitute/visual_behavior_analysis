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
    from visual_behavior.ophys.io.lims_database import LimsDatabase
    ld = LimsDatabase(lims_id)
    lims_data = ld.get_qc_param()
    lims_data.insert(loc=2, column='experiment_id', value=lims_data.lims_id.values[0])
    lims_data.insert(loc=2, column='session_name', value=lims_data.experiment_name.values[0].split('_')[-1])
    lims_data.insert(loc=2, column='ophys_session_dir', value=lims_data.datafolder.values[0][:-28])
    return lims_data


def get_lims_id(lims_data):
    lims_id = lims_data.lims_id.values[0]
    return lims_id


def get_analysis_folder_name(lims_data):
    date = str(lims_data.experiment_date.values[0])[:10].split('-')
    analysis_folder_name = str(lims_data.external_specimen_id.values[0]) + '_' + date[0][2:] + date[1] + date[2] + '_' + \
                           str(lims_data.lims_id.values[0]) + '_' + \
                           lims_data.structure.values[0] + '_' + str(lims_data.depth.values[0]) + '_' + \
                           lims_data.specimen_driver_line.values[0].split('-')[0] + '_' + lims_data.rig.values[0][3:5] + \
                           lims_data.rig.values[0][6] + '_' + lims_data.session_name.values[0]
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
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
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
    segmentation_dir = os.path.join(processed_dir, segmentation_folder[0])
    return segmentation_dir


def get_sync_path(lims_data):
    ophys_session_dir = get_ophys_session_dir(lims_data)
    sync_file = [file for file in os.listdir(ophys_session_dir) if 'sync' in file][0]
    sync_path = os.path.join(ophys_session_dir, sync_file)
    analysis_dir = get_analysis_dir(lims_data)
    if sync_file not in os.listdir(analysis_dir):
        print('moving ', sync_file, ' to analysis dir')  # flake8: noqa: E999
        shutil.copy2(sync_path, os.path.join(analysis_dir, sync_file))
    return sync_path


def get_timestamps(lims_data):
    from visual_behavior.ophys.sync.process_sync import get_sync_data
    sync_data = get_sync_data(lims_data)
    timestamps = pd.DataFrame(sync_data)
    return timestamps


def save_timestamps(timestamps, lims_data):
    save_dataframe_as_h5(timestamps, 'timestamps', get_analysis_dir(lims_data))


def get_timestamps_stimulus(timestamps):
    timestamps_stimulus = timestamps['stimulus_frames']['timestamps']
    return timestamps_stimulus


def get_timestamps_ophys(timestamps):
    timestamps_ophys = timestamps['ophys_frames']['timestamps']
    return timestamps_ophys


def get_metadata(lims_data, timestamps):
    timestamps_stimulus = get_timestamps_stimulus(timestamps)
    timestamps_ophys = get_timestamps_ophys(timestamps)
    from collections import OrderedDict
    metadata = OrderedDict()
    metadata['experiment_id'] = lims_data['experiment_id'].values[0]
    metadata['experiment_date'] = str(lims_data.experiment_date.values[0])[:10]
    metadata['mouse_id'] = int(lims_data.external_specimen_id.values[0])
    metadata['area'] = lims_data.structure.values[0]
    metadata['depth'] = int(lims_data.depth.values[0])
    metadata['driver_line'] = lims_data['specimen_driver_line'].values[0]
    metadata['reporter_line'] = lims_data['specimen_reporter_line'].values[0]
    metadata['image_set'] = lims_data.session_name.values[0][-1]
    metadata['session_name'] = lims_data.session_name.values[0]
    metadata['session_id'] = int(lims_data.session_id.values[0])
    metadata['parent_session_id'] = int(lims_data.parent_session_id.values[0])
    metadata['specimen_id'] = int(lims_data.specimen_id.values[0])
    metadata['project_id'] = lims_data.project_id.values[0]
    metadata['rig'] = lims_data.rig.values[0]
    metadata['ophys_frame_rate'] = np.round(1 / np.mean(np.diff(timestamps_ophys)), 1)
    metadata['stimulus_frame_rate'] = np.round(1 / np.mean(np.diff(timestamps_stimulus)), 1)
    # metadata['eye_tracking_frame_rate'] = np.round(1 / np.mean(np.diff(self.timestamps_eye_tracking)),1)
    metadata = pd.DataFrame(metadata, index=[metadata['experiment_id']])
    return metadata


def save_metadata(metadata, lims_data):
    save_dataframe_as_h5(metadata, 'metadata', get_analysis_dir(lims_data))


def get_stimulus_pkl_path(lims_data):
    ophys_session_dir = get_ophys_session_dir(lims_data)
    # first try lims folder
    pkl_file = [file for file in os.listdir(ophys_session_dir) if 'stim.pkl' in file]
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
        print('moving ', pkl_file, ' to analysis dir')
        shutil.copy2(stimulus_pkl_path, os.path.join(analysis_dir, pkl_file))
    print('getting stimulus data from pkl')
    pkl = pd.read_pickle(stimulus_pkl_path)
    # from visual_behavior.translator.foraging2 import data_to_change_detection_core
    # core_data = data_to_change_detection_core(pkl)
    # print('visual frames in pkl file:', len(core_data['time']))
    return pkl


def get_core_data(pkl, timestamps_stimulus):
    from visual_behavior.translator.foraging import data_to_change_detection_core
    core_data = data_to_change_detection_core(pkl, time=timestamps_stimulus)
    return core_data


def get_task_parameters(core_data):
    task_parameters = {}
    task_parameters['blank_duration'] = core_data['metadata']['blank_duration_range'][0]
    task_parameters['stimulus_duration'] = core_data['metadata']['stim_duration']
    task_parameters['omitted_flash_fraction'] = core_data['metadata']['params']['omitted_flash_fraction']
    task_parameters['response_window'] = [core_data['metadata']['response_window']]
    task_parameters['reward_volume'] = core_data['metadata']['rewardvol']
    task_parameters['stage'] = core_data['metadata']['stage']
    task_parameters['stimulus'] = core_data['metadata']['stimulus']
    task_parameters['stimulus_distribution'] = core_data['metadata']['stimulus_distribution']
    task_parameters['task'] = core_data['metadata']['task']
    task_parameters['n_stimulus_frames'] = core_data['metadata']['n_stimulus_frames']
    task_parameters = pd.DataFrame(task_parameters, columns=task_parameters.keys(), index=['params'])
    return task_parameters


def save_core_data_components(core_data, lims_data):
    rewards = core_data['rewards']
    save_dataframe_as_h5(rewards, 'rewards', get_analysis_dir(lims_data))

    running = core_data['running']
    save_dataframe_as_h5(running, 'running', get_analysis_dir(lims_data))

    licks = core_data['licks']
    save_dataframe_as_h5(licks, 'licks', get_analysis_dir(lims_data))

    visual_stimuli = core_data['visual_stimuli']
    save_dataframe_as_h5(visual_stimuli, 'visual_stimuli', get_analysis_dir(lims_data))

    task_parameters = get_task_parameters(core_data)
    save_dataframe_as_h5(task_parameters, 'task_parameters', get_analysis_dir(lims_data))


def get_trials(core_data):
    from visual_behavior.translator.core import create_extended_dataframe
    trials = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks=core_data['licks'],
        time=core_data['time'])
    return trials


def save_trials(trials, lims_data):
    save_dataframe_as_h5(trials, 'trials', get_analysis_dir(lims_data))


def get_running_speed(pkl, timestamps):
    # from visual_behavior.translator.foraging2 import get_running_speed
    # speed = get_running_speed(pkl, smooth=False, time=None)
    # running_speed = speed['speed (cm/s)'].values
    import visual_behavior.io as vbio
    timestamps_stimulus = get_timestamps_stimulus(timestamps)
    speed = vbio.load_running_speed(pkl, time=timestamps_stimulus)
    running_speed = speed['speed (cm/s)'].values
    print('length of running speed trace: ', str(len(running_speed)))
    return running_speed


def save_running_speed(running_speed, lims_data):
    save_data_as_h5(running_speed, 'running_speed', get_analysis_dir(lims_data))


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


def get_input_extract_traces_json(lims_data):
    processed_dir = get_processed_dir(lims_data)
    json_file = [file for file in os.listdir(processed_dir) if 'input_extract_traces.json' in file]
    json_path = os.path.join(processed_dir, json_file[0])
    with open(json_path, 'r') as w:
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


def add_cell_specimen_ids_to_roi_metrics(roi_metrics, roi_locations):
    # add roi ids to objectlist
    ids = []
    for row in roi_metrics.index:
        minx = roi_metrics.iloc[row][' minx']
        miny = roi_metrics.iloc[row][' miny']
        id = roi_locations[(roi_locations.x == minx) & (roi_locations.y == miny)].id.values[0]
        ids.append(id)
    roi_metrics['cell_specimen_id'] = ids
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
    roi_metrics = add_cell_specimen_ids_to_roi_metrics(roi_metrics, roi_locations)
    # merge roi_metrics and roi_locations
    roi_metrics['id'] = roi_metrics.cell_specimen_id.values
    roi_metrics = pd.merge(roi_metrics, roi_locations, on='id')
    # remove invalid roi_metrics
    roi_metrics = roi_metrics[roi_metrics.valid == True]
    # add filtered cell index
    cell_index = [np.where(np.sort(roi_metrics.cell_specimen_id.values) == id)[0][0] for id in
                  roi_metrics.cell_specimen_id.values]
    roi_metrics['cell_index'] = cell_index
    return roi_metrics


def save_roi_metrics(roi_metrics, lims_data):
    save_dataframe_as_h5(roi_metrics, 'roi_metrics', get_analysis_dir(lims_data))


def get_cell_specimen_ids(roi_metrics):
    cell_specimen_ids = np.sort(roi_metrics.cell_specimen_id.values)
    return cell_specimen_ids


def get_cell_indices(roi_metrics):
    cell_indices = np.sort(roi_metrics.cell_index.values)
    return cell_indices


def get_cell_specimen_id_for_cell_index(cell_index, cell_specimen_ids):
    cell_specimen_id = cell_specimen_ids[cell_index]
    return cell_specimen_id


def get_cell_index_for_cell_specimen_id(cell_specimen_id, cell_specimen_ids):
    cell_index = np.where(cell_specimen_ids == cell_specimen_id)[0][0]
    return cell_index


def get_roi_masks(roi_metrics, lims_data):
    # make roi_dict with ids as keys and roi_mask_array
    jin = get_input_extract_traces_json(lims_data)
    h = jin["image"]["height"]
    w = jin["image"]["width"]
    cell_specimen_ids = get_cell_specimen_ids(roi_metrics)
    roi_masks = {}
    for i, id in enumerate(cell_specimen_ids):
        m = roi_metrics[roi_metrics.id == id]
        mask = np.asarray(m['mask'].values[0])
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        binary_mask[int(m.y):int(m.y) + int(m.height), int(m.x):int(m.x) + int(m.width)] = mask
        roi_masks[id] = binary_mask
    return roi_masks


def save_roi_masks(roi_masks, lims_data):
    f = h5py.File(os.path.join(get_analysis_dir(lims_data), 'roi_masks.h5'), 'w')
    for id, roi_mask in roi_masks.items():
        f.create_dataset(str(id), data=roi_mask)
    f.close()


def get_dff_traces(roi_metrics, lims_data):
    dff_path = os.path.join(get_ophys_experiment_dir(lims_data), str(get_lims_id(lims_data)) + '_dff.h5')
    g = h5py.File(dff_path)
    dff_traces = np.asarray(g['data'])
    # # filter out NaN traces - how did they get in here anyway? very rare
    # for i in range(dff_traces.shape[0]):
    #     if np.isnan(dff_traces[i][0]):
    #         index = self.roi_df[self.roi_df.unfiltered_cell_index == i].index[0]
    #         self.roi_df.at[index, 'valid'] = False
    # only include valid roi traces
    valid_roi_indices = np.sort(roi_metrics.unfiltered_cell_index.values)
    dff_traces = dff_traces[valid_roi_indices]
    print('length of traces:', dff_traces.shape[1])
    print('number of segmented cells:', dff_traces.shape[0])
    return dff_traces


def save_dff_traces(dff_traces, roi_metrics, lims_data):
    f = h5py.File(os.path.join(get_analysis_dir(lims_data), 'dff_traces.h5'), 'w')
    for i, id in enumerate(get_cell_specimen_ids(roi_metrics)):
        f.create_dataset(str(id), data=dff_traces[i])
    f.close()


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
    import matplotlib.image as mpimg
    # max_projection = mpimg.imread(os.path.join(get_processed_dir(lims_data), 'max_downsample_4Hz_0.png'))
    max_projection = mpimg.imread(os.path.join(get_segmentation_dir(lims_data), 'maxInt_a13a.png'))
    return max_projection


def save_max_projection(max_projection, lims_data):
    analysis_dir = get_analysis_dir(lims_data)
    save_data_as_h5(max_projection, 'max_projection', analysis_dir)
    import matplotlib.image as mpimg
    mpimg.imsave(os.path.join(get_analysis_dir(lims_data), 'max_intensity_projection.png'), arr=max_projection,
                 cmap='gray')


def convert_level_1_to_level_2(lims_id, cache_dir=None):
    lims_data = get_lims_data(lims_id)

    get_analysis_dir(lims_data, cache_on_lims_data=True, cache_dir=cache_dir)

    timestamps = get_timestamps(lims_data)
    save_timestamps(timestamps, lims_data)

    metadata = get_metadata(lims_data, timestamps)
    save_metadata(metadata, lims_data)

    pkl = get_pkl(lims_data)
    timestamps_stimulus = get_timestamps_stimulus(timestamps)
    core_data = get_core_data(pkl, timestamps_stimulus)
    save_core_data_components(core_data, lims_data)

    trials = get_trials(core_data)
    save_trials(trials, lims_data)

    roi_metrics = get_roi_metrics(lims_data)
    save_roi_metrics(roi_metrics, lims_data)

    roi_masks = get_roi_masks(roi_metrics, lims_data)
    save_roi_masks(roi_masks, lims_data)

    dff_traces = get_dff_traces(roi_metrics, lims_data)
    save_dff_traces(dff_traces, roi_metrics, lims_data)

    motion_correction = get_motion_correction(lims_data)
    save_motion_correction(motion_correction, lims_data)

    max_projection = get_max_projection(lims_data)
    save_max_projection(max_projection, lims_data)

    import visual_behavior.ophys.plotting.summary_figures as sf
    sf.plot_roi_validation(lims_data)


if __name__ == '__main__':
    lims_id = 702134928
    convert_level_1_to_level_2(lims_id, cache_dir='/allen/aibs/technology/nicholasc/tmp2')
