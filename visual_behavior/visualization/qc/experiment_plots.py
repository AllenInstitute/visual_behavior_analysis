import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import visual_behavior.visualization.utils as utils

import visual_behavior.visualization.qc.plotting_utils as pu

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


def plot_event_detection_for_experiment(ophys_experiment_id):
    dataset = data_loading.get_ophys_dataset(ophys_experiment_id)
    metadata_string = dataset.metadata_string
    colors = sns.color_palette()

    for cell_specimen_id in dataset.cell_specimen_ids:
        n_rows = 5
        figsize = (15, 10)
        fig, ax = plt.subplots(n_rows, 1, figsize=figsize)
        ax = ax.ravel()
        x = 5
        for i in range(n_rows):
            ax[i].plot(dataset.ophys_timestamps, dataset.traces.loc[cell_specimen_id].dff, color=colors[0], label='dff_trace')
            ax[i].plot(dataset.ophys_timestamps, dataset.events.loc[cell_specimen_id].events, color=colors[3], label='events')
            ax[i].set_xlim(60 * 10, (60 * 10) + x)
            x = x * 5
        ax[0].set_title('oeid: ' + str(ophys_experiment_id) + ', csid: ' + str(cell_specimen_id))
        ax[i].legend(loc='upper left')
        ax[i].set_xlabel('time (seconds)')
        fig.tight_layout()

        utils.save_figure(fig, figsize, data_loading.get_container_plots_dir(), 'event_detection',
                       'container_' + metadata_string +'_'+ str(cell_specimen_id))
