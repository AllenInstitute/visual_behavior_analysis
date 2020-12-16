import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import visual_behavior.visualization.utils as utils
import visual_behavior.visualization.qc.plotting_utils as pu
import visual_behavior.visualization.qc.single_cell_plots as scp

from visual_behavior.data_access import loading as data_loading
from visual_behavior.data_access import processing as data_processing

import visual_behavior.database as db
from visual_behavior.utilities import EyeTrackingData


# OPHYS
bitdepth_16 = 65536


def plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    max_projection = data_loading.get_sdk_max_projection(ophys_experiment_id)
    ax.imshow(max_projection, cmap='gray', vmax=np.percentile(max_projection, 99))
    ax.axis('off')
    return ax


def plot_average_image_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    average_image = data_loading.get_sdk_ave_projection(ophys_experiment_id)
    ax.imshow(average_image, cmap='gray', vmax=np.amax(average_image))
    ax.axis('off')
    return ax


def plot_motion_correction_average_image_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    average_image = data_processing.experiment_average_FOV_from_motion_corrected_movie(ophys_experiment_id)
    ax.imshow(average_image, cmap='gray', vmin=0, vmax=8000)
    ax.axis('off')
    return ax


def plot_motion_correction_max_image_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    max_image = data_processing.experiment_max_FOV_from_motion_corrected_movie(ophys_experiment_id)
    ax.imshow(max_image, cmap='gray', vmin=0, vmax=8000)
    ax.axis('off')
    return ax


def plot_segmentation_mask_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    # segmentation_mask = data_loading.get_sdk_segmentation_mask_image(ophys_experiment_id)
    segmentation_mask = data_loading.get_valid_segmentation_mask(ophys_experiment_id)
    ax.imshow(segmentation_mask, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    return ax


def plot_segmentation_mask_overlay_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax = plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax)
    # segmentation_mask = data_loading.get_sdk_segmentation_mask_image(ophys_experiment_id)
    segmentation_mask = data_loading.get_valid_segmentation_mask(ophys_experiment_id)
    mask = np.zeros(segmentation_mask.shape)
    mask[:] = np.nan
    mask[segmentation_mask == 1] = 1
    ax.imshow(mask, cmap='hsv', vmax=1, alpha=0.5)
    ax.axis('off')
    return ax


def plot_traces_heatmap_for_experiment(ophys_experiment_id, ax=None):
    dff_traces = data_loading.get_sdk_dff_traces_array(ophys_experiment_id)
    if ax is None:
        figsize = (14, 5)
        fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(dff_traces, cmap='magma', vmin=0, vmax=0.5)
    ax.set_ylabel('cells')
    ax.set_xlabel('2P frames')
    return ax


def plot_csid_snr_for_experiment(ophys_experiment_id, ax=None):
    experiment_df = data_processing.ophys_experiment_info_df(ophys_experiment_id)
    exp_snr = data_processing.experiment_cell_specimen_id_snr_table(ophys_experiment_id)
    exp_snr["stage_name_lims"] = experiment_df["stage_name_lims"][0]
    exp_stage_color_dict = pu.experiment_id_stage_color_dict_for_experiment(ophys_experiment_id)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax = sns.violinplot(x="stage_name_lims", y="robust_snr", data=exp_snr.loc[exp_snr["snr_zscore"] < 3],
                        color=exp_stage_color_dict[ophys_experiment_id])
    ax.set_ylabel("robust snr")
    ax.set_xlabel("stage name")
    return ax


def plot_average_intensity_timeseries_for_experiment(ophys_experiment_id, ax=None, color='gray'):
    """plots the average intensity of a subset of the motion corrected movie
        subset: inner portion of every 500th frame
        the color of the plot is based onthe stage name of the experiment

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Keyword Arguments:
        ax {[type]} -- [description] (default: {None})
        color {str} -- [description] (default: {'gray'})

    Returns:
        plot -- x: frame number, y: fluroescence value
    """
    experiment_df = data_processing.ophys_experiment_info_df(ophys_experiment_id)
    exp_stage_color_dict = pu.map_stage_name_colors_to_ophys_experiment_ids(experiment_df)
    average_intensity, frame_numbers = data_processing.get_experiment_average_intensity_timeseries(ophys_experiment_id)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(frame_numbers, average_intensity,
            color=exp_stage_color_dict[ophys_experiment_id],
            label=experiment_df["stage_name_lims"][0])
    ax.set_ylabel('fluorescence value')
    ax.set_xlabel('frame #')
    return ax


def plot_motion_correction_xy_shift_for_experiment(ophys_experiment_id, ax=None):
    df = data_loading.load_rigid_motion_transform_csv(ophys_experiment_id)
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(df.framenumber.values, df.x.values, color='red', label='x_shift')
    ax.plot(df.framenumber.values, df.y.values, color='blue', label='y_shift')
    ax.set_xlim(df.framenumber.values[0], df.framenumber.values[-1])
    ax.legend(fontsize='small', loc='upper right')
    ax.set_xlabel('2P frames')
    ax.set_ylabel('pixels')
    return ax

# BEHAVIOR


def make_eye_matrix_plot(ophys_experiment_id, ax):
    ax = np.array(ax)
    try:
        ophys_session_id = db.convert_id({'ophys_experiment_id': ophys_experiment_id}, 'ophys_session_id')
        ed = EyeTrackingData(ophys_session_id)

        frames = np.linspace(0, len(ed.ellipse_fits['pupil']) - 1, len(ax.flatten())).astype(int)
        for ii, frame in enumerate(frames):
            axis = ax.flatten()[ii]
            axis.imshow(ed.get_annotated_frame(frame))
            axis.axis('off')
            axis.text(5, 5, 'frame {}'.format(frame), ha='left', va='top', color='yellow', fontsize=8)

        ax[0][0].set_title('ophys_experiment_id = {}, {} evenly spaced sample eye tracking frames'.format(ophys_experiment_id, len(frames)), ha='left')

    except Exception as e:
        for ii in range(len(ax.flatten())):
            axis = ax.flatten()[ii]
            axis.axis('off')

            error_text = 'could not generate pupil plot for ophys_experiment_id {}\n{}'.format(ophys_experiment_id, e)
            ax[0][0].set_title(error_text, ha='left')
    return ax


def make_pupil_area_plot(ophys_experiment_id, ax, label_x=True):
    '''plot pupil area vs time'''
    try:
        ophys_session_id = db.convert_id({'ophys_experiment_id': ophys_experiment_id}, 'ophys_session_id')
        ed = EyeTrackingData(ophys_session_id)

        time = ed.ellipse_fits['pupil']['time'] / 60.
        area = ed.ellipse_fits['pupil']['blink_corrected_area']
        ax.plot(time, area)
        if label_x:
            ax.set_xlabel('time (minutes)')
        ax.set_ylabel('pupil diameter\n(pixels$^2$)')
        ax.set_xlim(min(time), max(time))

        ax.set_title('ophys_experiment_id = {}, pupil area vs. time'.format(ophys_experiment_id), ha='center')

    except Exception as e:
        ax.axis('off')

        error_text = 'could not generate pupil area plot for ophys_experiment_id {}\n{}'.format(ophys_experiment_id, e)
        ax.set_title(error_text, ha='left')
    return ax


def make_pupil_position_plot(ophys_experiment_id, ax, label_x=True):
    '''plot pupil position vs time'''
    try:
        ophys_session_id = db.convert_id({'ophys_experiment_id': ophys_experiment_id}, 'ophys_session_id')
        ed = EyeTrackingData(ophys_session_id)

        time = ed.ellipse_fits['pupil']['time'] / 60.
        x = ed.ellipse_fits['pupil']['blink_corrected_center_x']
        y = ed.ellipse_fits['pupil']['blink_corrected_center_y']

        ax.plot(time, x, color='darkorange')
        ax.plot(time, y, color='olive')

        if label_x:
            ax.set_xlabel('time (minutes)')
        ax.set_ylabel('pupil position (pixel)')
        ax.legend(['x position', 'y position'])
        ax.set_xlim(min(time), max(time))

        ax.set_title('ophys_experiment_id = {}, pupil center position vs. time'.format(ophys_experiment_id), ha='center')

    except Exception as e:
        ax.axis('off')

        error_text = 'could not generate pupil position plot for ophys_experiment_id {}\n{}'.format(ophys_experiment_id, e)
        ax.set_title(error_text, ha='left')
    return ax


def plot_event_detection_for_experiment(ophys_experiment_id, save_figure=True):
    dataset = data_loading.get_ophys_dataset(ophys_experiment_id)
    metadata_string = dataset.metadata_string
    colors = sns.color_palette()
    ophys_timestamps = dataset.ophys_timestamps.copy()
    dff_traces = dataset.dff_traces.copy()
    events = dataset.events.copy()

    for cell_specimen_id in dataset.cell_specimen_ids:
        n_rows = 5
        figsize = (15, 10)
        fig, ax = plt.subplots(n_rows, 1, figsize=figsize)
        ax = ax.ravel()
        x = 5
        for i in range(n_rows):
            ax[i].plot(ophys_timestamps, dff_traces.loc[cell_specimen_id].dff, color=colors[0], label='dff_trace')
            ax[i].plot(ophys_timestamps, events.loc[cell_specimen_id].events, color=colors[3], label='events')
            ax[i].set_xlim(60 * 10, (60 * 10) + x)
            x = x * 5
        ax[0].set_title('oeid: ' + str(ophys_experiment_id) + ', csid: ' + str(cell_specimen_id))
        ax[i].legend(loc='upper left')
        ax[i].set_xlabel('time (seconds)')
        fig.tight_layout()

        if save_figure:
            utils.save_figure(fig, figsize, data_loading.get_single_cell_plots_dir(), 'event_detection',
                              str(cell_specimen_id) + '_' + metadata_string + '_events_validation')


def plot_dff_trace_and_behavior_for_experiment(ophys_experiment_id, save_figure=True):
    dataset = data_loading.get_ophys_dataset(ophys_experiment_id)

    for cell_specimen_id in dataset.cell_specimen_ids:
        scp.plot_single_cell_activity_and_behavior(dataset, cell_specimen_id, save_figure=save_figure)


def plot_population_activity_and_behavior_for_experiment(ophys_experiment_id, save_figure=True):
    dataset = data_loading.get_ophys_dataset(ophys_experiment_id)
    traces = dataset.dff_traces.copy()
    trace_timestamps = dataset.ophys_timestamps

    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))

    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values

    pupil_area = dataset.eye_tracking.pupil_area.values
    pupil_timestamps = dataset.eye_tracking.time.values

    face_motion = dataset.behavior_movie_pc_activations[:, 0]
    face_timestamps = dataset.timestamps['eye_tracking'].timestamps

    figsize = (20, 10)
    fig, ax = plt.subplots(5, 1, figsize=figsize, sharex=True)
    colors = sns.color_palette()

    trace = np.nanmean(np.vstack(traces.dff.values), axis=0)
    ax[0].plot(trace_timestamps, trace, label='mean_trace', color=colors[0])
    ax[0].set_ylabel('dF/F')
    ax[1].plot(lick_timestamps, licks, '|', label='licks', color=colors[3])
    ax[1].set_ylabel('licks')
    ax[1].set_yticklabels([])
    ax[2].plot(running_timestamps, running_speed, label='running_speed', color=colors[4])
    ax[2].set_ylabel('run speed\n(cm/s)')
    ax[3].plot(pupil_timestamps, pupil_area, label='pupil_area', color=colors[9])
    ax[3].set_ylabel('pupil area\n pixels**2')
    ax[3].set_ylim(-50, 20000)
    ax[4].plot(face_timestamps, face_motion, label='face_motion_PC0', color=colors[2])
    ax[4].set_ylabel('face motion\n PC0 activation')

    for x in range(5):
        ax[x].tick_params(which='both', bottom=False, top=False, right=False, left=True,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True)
    ax[4].tick_params(which='both', bottom=False, top=False, right=False, left=True,
                      labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    #     ax[x].legend(loc='upper left', fontsize='x-small')
    plt.subplots_adjust(wspace=0, hspace=0.1)
    ax[0].set_title(dataset.metadata_string)
    if save_figure:
        utils.save_figure(fig, figsize, utils.get_experiment_plots_dir(), 'population_activity_and_behavior',
                          dataset.metadata_string + '_population_activity_and_behavior')
        plt.close()


def get_suite2p_rois(fname):
    import json
    with open(fname, "r") as f:
        j = json.load(f)
    cell_table = pd.DataFrame(j)
    return cell_table

def get_matching_output(fname):
    import json
    with open(fname, "r") as f:
        j = json.load(f)
    return j

def place_masks_in_full_image(cell_table, max_projection):
    cell_table['image_mask'] = None
    for index in cell_table.index:
        mask = cell_table.loc[index, 'mask_matrix']
        dims = max_projection.shape
        image = np.zeros(dims)
        mask = np.asarray(mask)
        image[0:mask.shape[0], 0:mask.shape[1]] = mask
        cell_table.at[index, 'image_mask'] = image
    cell_table = processing.shift_image_masks(cell_table)
    return cell_table

def plot_classifier_validation_for_experiment(ophys_experiment_id, save_figure=True):
    """This is a quick and dirty function to get plots needed for a rapid decision to be made"""

    import visual_behavior.visualization.ophys.summary_figures as sf
    from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    expt = ophys_experiment_id
    # get new classifier output
    data = pd.read_csv(r"//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/classifier_validation/inference_with_production_ids_annotated.csv",
                       dtype={'production-id': 'Int64'})
    # get suite2P segmentation output
    output_dir = r'//allen/aibs/informatics/danielk/dev_LIMS/new_labeling'
    folder = [folder for folder in os.listdir(output_dir) if str(expt) in folder]
    segmentation_output_file = os.path.join(output_dir, folder[0], 'binarize_output.json')
    cell_table = get_suite2p_rois(segmentation_output_file)
    cell_table['experiment_id'] = expt
    # move suite2P masks to the proper place
    dataset = loading.get_ophys_dataset(expt, include_invalid_rois=True)
    cell_table = place_masks_in_full_image(cell_table, dataset.max_projection.data)
    # merge with classifier results
    cell_table = cell_table.merge(data, on=['experiment_id', 'id'])
    cell_table['roi_id'] = cell_table['id']
    # make a mask dictionary for suite2P outputs
    cell_table_roi_masks = {}
    for roi_id in cell_table.roi_id.values:
        cell_table_roi_masks[str(roi_id)] = cell_table[cell_table.roi_id == roi_id].roi_mask.values[0]
    # limit to classifier results for this experiment
    expt_data = data[data.experiment_id == expt].copy()
    # get production segmentation & classification from SDK
    dataset = loading.get_ophys_dataset(expt, include_invalid_rois=True)
    ct = dataset.cell_specimen_table.copy()
    roi_masks = dataset.roi_masks.copy()
    max_projection = dataset.max_projection.data
    dff_traces = dataset.dff_traces.copy()
    ophys_timestamps = dataset.ophys_timestamps
    metadata_string = dataset.metadata_string
    # get average response df
    analysis = ResponseAnalysis(dataset)
    sdf = analysis.get_response_df(df_name='stimulus_response_df')

    # plots for all ROIs in experiment
    for roi_id in expt_data.roi_id.unique():
        present_in = expt_data[expt_data.roi_id == roi_id].roi_present_in.values[0]
        if present_in == 'both':
            cell_roi_id = expt_data[expt_data.roi_id == roi_id].cell_roi_id.values[0]
            cell_specimen_id = ct[ct.cell_roi_id == cell_roi_id].index.values[0]
        if present_in == 'prod':
            cell_roi_id = expt_data[expt_data.roi_id == roi_id].cell_roi_id.values[0]
            cell_specimen_id = ct[ct.cell_roi_id == cell_roi_id].index.values[0]
        if present_in == 'dev':
            cell_roi_id = roi_id
            cell_specimen_id = roi_id
        folder = present_in

        figsize = (20, 10)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 4)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[1, :3])
        ax5 = fig.add_subplot(gs[1, 3])

        if (present_in == 'prod') or (present_in == 'both'):
            masks_array = ct.loc[cell_specimen_id]['roi_mask'].copy()
            masks_to_plot = np.empty(masks_array.shape)
            masks_to_plot[:] = np.nan
            masks_to_plot[masks_array == True] = 1
            #     masks_to_plot = processing.gen_transparent_multi_roi_mask(masks_array)
            ax0.imshow(max_projection, cmap='gray')
            ax0.imshow(masks_to_plot, cmap='hsv', vmin=0, vmax=1, alpha=0.5)
            ax0.set_title('production roi mask')
            ax0.axis(False)

            valid = ct.loc[cell_specimen_id].valid_roi
            ax1 = sf.plot_cell_zoom(roi_masks, max_projection, cell_specimen_id, spacex=40, spacey=60, show_mask=True,
                                    ax=ax1)
            ax1.set_title('production roi mask, valid: ' + str(valid))
        else:
            folder = 'dev_only'
            masks_array = ct[ct.valid_roi == True]['roi_mask'].values
            masks_to_plot = processing.gen_transparent_multi_roi_mask(masks_array)
            ax0.imshow(max_projection, cmap='gray')
            ax0.imshow(masks_to_plot, cmap='hsv', vmin=0, vmax=1, alpha=0.5)
            ax0.set_title('valid production roi masks')

            masks_array = ct[ct.valid_roi == False]['roi_mask'].values
            masks_to_plot = processing.gen_transparent_multi_roi_mask(masks_array)
            ax1.imshow(max_projection, cmap='gray')
            ax1.imshow(masks_to_plot, cmap='hsv', vmin=0, vmax=1, alpha=0.5)
            ax1.set_title('invalid production roi masks')

        if (present_in == 'dev') or (present_in == 'both'):
            valid_CNN = expt_data[expt_data.roi_id == roi_id].valid_CNN.values[0]
            masks_array = cell_table[cell_table.roi_id == roi_id]['roi_mask'].values[0]
            masks_array[masks_array == 0] = np.nan
            ax2.imshow(max_projection, cmap='gray')
            ax2.imshow(masks_array, cmap='hsv', vmin=0, vmax=1, alpha=0.5)
            ax2.set_title('suite2p roi mask')

            valid_CNN = expt_data[expt_data.roi_id == roi_id].valid_CNN.values[0]
            ax3 = sf.plot_cell_zoom(cell_table_roi_masks, max_projection, roi_id, spacex=40, spacey=60, show_mask=True,
                                    ax=ax3)
            ax3.set_title('suite2p roi mask, valid: ' + str(valid_CNN))
            if present_in == 'both':
                if (valid_CNN == True) and (valid == True):
                    folder = 'dev_valid_prod_valid'
                elif (valid_CNN == True) and (valid == False):
                    folder = 'dev_valid_prod_invalid'
                elif (valid_CNN == False) and (valid == True):
                    folder = 'dev_invalid_prod_valid'
                elif (valid_CNN == False) and (valid == False):
                    folder = 'dev_invalid_prod_invalid'
            elif present_in == 'dev':
                if valid_CNN == True:
                    folder = 'dev_valid_not_in_prod'
                elif valid_CNN == False:
                    folder = 'dev_invalid_not_in_prod'
        else:
            folder = 'prod_only'
            masks_array = cell_table[cell_table.valid_roi == True]['roi_mask'].values
            masks_to_plot = processing.gen_transparent_multi_roi_mask(masks_array)
            masks_to_plot = np.sum(masks_array, 0)
            #         ax2.imshow(max_projection, cmap='gray')
            ax2.imshow(masks_to_plot, cmap='hsv', vmin=0, vmax=1, alpha=0.5)
            ax2.set_title('valid suite2p roi masks')

            masks_array = cell_table[cell_table.valid_roi == False]['roi_mask'].values
            #         masks_to_plot = processing.gen_transparent_multi_roi_mask(masks_array)
            masks_to_plot = np.sum(masks_array, 0)
            ax3.imshow(max_projection, cmap='gray')
            ax3.imshow(masks_to_plot, cmap='hsv', vmin=0, vmax=1, alpha=0.5)
            ax3.set_title('invalid suite2p roi masks')

        if (present_in == 'prod') or (present_in == 'both'):
            ax5 = sf.plot_mean_trace(sdf[sdf.cell_specimen_id == cell_specimen_id].trace.values,
                                     frame_rate=analysis.ophys_frame_rate,
                                     xlims=[-0.5, 0.75], interval_sec=0.5, ax=ax5)
            ax5 = sf.plot_flashes_on_trace(ax5, analysis, window=[-0.5, 0.75])
            ax5.set_title('mean image response')

            ax4.plot(ophys_timestamps, dff_traces.loc[cell_specimen_id].dff)
            ax4.set_xlim(ophys_timestamps[0], ophys_timestamps[-1])
            ax4.set_xlabel('time (seconds)')
            ax4.set_ylabel('dF/F')
        ax4.set_title(metadata_string)

        s = 'present_in: ' + present_in + ', roi_id: ' + str(roi_id) + ', cell_roi_id: ' + str(
            cell_roi_id) + ', cell_specimen_id: ' + str(cell_specimen_id)
        plt.suptitle(s, x=0.5, y=1.02)

        fig.tight_layout()
        save_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/classifier_validation/CNN_rois'
        utils.save_figure(fig, figsize, save_dir, folder, str(cell_specimen_id) + '_' + metadata_string)
