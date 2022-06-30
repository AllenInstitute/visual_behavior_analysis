import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import visual_behavior.visualization.utils as utils
import visual_behavior.visualization.qc.plotting_utils as pu
import visual_behavior.visualization.qc.single_cell_plots as scp
import visual_behavior.visualization.ophys.summary_figures as sf


from visual_behavior.data_access import loading as loading
from visual_behavior.data_access import processing as processing

import visual_behavior.database as db
from visual_behavior.utilities import EyeTrackingData


# OPHYS
bitdepth_16 = 65536


def get_metadata_string(dataset):
    """
    Create a string of metadata information to be used in filenames and figure titles.
    Includes information such as experiment_id, cre_line, acquisition_date, rig_id, etc
    :param dataset: BehaviorOphysExperiment object
    :return:
    """
    # dataset = loading.get_ophys_dataset(ophys_experiment_id)
    m = dataset.metadata.copy()
    metadata_string = str(m['mouse_id']) + '_' + str(m['ophys_experiment_id']) + '_' + m['cre_line'].split('-')[0] + '_' + m['targeted_structure'] + '_' + str(m['imaging_depth']) + '_' + m['session_type']
    return metadata_string


def get_file_name_for_experiment(ophys_experiment_id):
    """
    gets standardized filename for saving figures
    format "experiment_id_"+str(ophys_experiment_id) is necessary for files to be able to be viewed in Dougs QC viewer
    using get_metadata_string(ophys_experiment_id) gives a more interpretable filename with cre line, area, etc
    :param ophys_experiment_id:
    :return:
    """
    filename = 'experiment_id'+str(ophys_experiment_id)
    return filename


def plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    max_projection = dataset.max_projection.data
    ax.imshow(max_projection, cmap='gray', vmax=np.percentile(max_projection, 99))
    ax.set_title(str(ophys_experiment_id))
    ax.axis('off')
    if save_figure:
        print('saving max intensity projection for ', ophys_experiment_id)
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'max_intensity_projection', get_metadata_string(dataset))
    return ax


def plot_average_image_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    average_projection = dataset.average_projection.data
    ax.imshow(average_projection, cmap='gray', vmax=np.amax(average_projection))
    ax.axis('off')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'average_intensity_image', get_metadata_string(dataset))
    return ax


def plot_motion_correction_average_image_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    average_image = processing.experiment_average_FOV_from_motion_corrected_movie(ophys_experiment_id)
    ax.imshow(average_image, cmap='gray', vmin=0, vmax=8000)
    ax.axis('off')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'average_image_movie', get_metadata_string(dataset))
    return ax


def plot_motion_correction_max_image_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    max_image = processing.experiment_max_FOV_from_motion_corrected_movie(ophys_experiment_id)
    ax.imshow(max_image, cmap='gray', vmin=0, vmax=8000)
    ax.set_title(str(ophys_experiment_id))
    ax.axis('off')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'max_image_movie', get_metadata_string(dataset))
    return ax


def plot_segmentation_mask_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False

    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    segmentation_mask = dataset.segmentation_mask_image  # i am not sure if this is correct, check relevant SDK issue to see what they did
    ax.imshow(segmentation_mask, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'segmentation_mask_image', get_metadata_string(dataset))
    return ax


def plot_valid_segmentation_mask_overlay_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    ax = plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax)
    try:
        dataset = loading.get_ophys_dataset(ophys_experiment_id, include_invalid_rois=False)
        segmentation_mask = dataset.segmentation_mask_image  # i am not sure if this is correct, check relevant SDK issue to see what they did
        mask = np.zeros(segmentation_mask[0].shape)
        mask[:] = np.nan
        mask[segmentation_mask[0] == 1] = 1
        ax.imshow(mask, cmap='hsv', vmax=1, alpha=0.5)
    except BaseException:
        pass
    ax.axis('off')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'segmentation_mask_overlay', get_metadata_string(dataset))
    return ax


def plot_all_segmentation_mask_overlay_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    ax = plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax)
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    segmentation_mask = dataset.segmentation_mask_image  # i am not sure if this is correct, check relevant SDK issue to see what they did
    mask = np.zeros(segmentation_mask[0].shape)
    mask[:] = np.nan
    mask[segmentation_mask[0] == 1] = 1
    ax.imshow(mask, cmap='hsv', vmax=1, alpha=0.5)
    ax.axis('off')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'segmentation_mask_overlay_all', get_metadata_string(dataset))
    return ax


def plot_valid_segmentation_mask_outlines_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    ax = plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax)
    dataset = loading.get_ophys_dataset(ophys_experiment_id, include_invalid_rois=False)
    segmentation_mask = dataset.segmentation_mask_image  # i am not sure if this is correct, check relevant SDK issue to see what they did
    mask = np.zeros(segmentation_mask[0].shape)
    mask[segmentation_mask[0] == 1] = 1
    ax.contour(mask, levels=0, colors=['red'], linewidths=[0.6])
    ax.set_title(str(ophys_experiment_id))
    ax.axis('off')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'segmentation_mask_valid', get_metadata_string(dataset))
    return ax


def plot_valid_segmentation_mask_outlines_per_cell_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    ax = plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax)
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    cell_specimen_table = dataset.cell_specimen_table.copy()
    if len(cell_specimen_table) > 0:
        for cell_roi_id in cell_specimen_table.cell_roi_id.values:
            mask = cell_specimen_table[cell_specimen_table.cell_roi_id == cell_roi_id].roi_mask.values[0]
            ax.contour(mask, levels=0, colors=['red'], linewidths=[0.6])
    ax.set_title(str(ophys_experiment_id)+'\nn valid ROIs = ' + str(len(cell_specimen_table.cell_roi_id.values)))
    ax.axis('off')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'segmentation_mask_valid_outlines', get_metadata_string(dataset))
    return ax


def plot_valid_and_invalid_segmentation_mask_overlay_per_cell_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        figsize = (5,5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    ax = plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax)
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    cell_specimen_table = dataset.cell_specimen_table.copy()
    exclusion_labels = loading.get_lims_cell_exclusion_labels(ophys_experiment_id)
    try:
        for cell_roi_id in cell_specimen_table[cell_specimen_table.valid_roi == True].cell_roi_id.values:
            mask = cell_specimen_table[cell_specimen_table.cell_roi_id == cell_roi_id].roi_mask.values[0]
            ax.contour(mask, levels=0, colors=['red'], linewidths=[1])
    except BaseException:
        pass
    try:
        for cell_roi_id in cell_specimen_table[cell_specimen_table.valid_roi == False].cell_roi_id.values:
            mask = cell_specimen_table[cell_specimen_table.cell_roi_id == cell_roi_id].roi_mask.values[0]
            excl_labels = exclusion_labels[exclusion_labels.cr_id == cell_roi_id].excl_label.values
            decrosstalk_in_labels = ['decrosstalk' in excl_label for excl_label in excl_labels]
            if (True in decrosstalk_in_labels) and (len(excl_labels) == 1):
                ax.contour(mask, levels=0, colors=['green'], linewidths=[2])
            elif (True in decrosstalk_in_labels) and (len(excl_labels) > 1):
                ax.contour(mask, levels=0, colors=['cyan'], linewidths=[1])
            else:
                ax.contour(mask, levels=0, colors=['blue'], linewidths=[1])
    except BaseException:
        pass
    ax.axis('off')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'segmentation_mask_valid_invalid_overlay', get_metadata_string(dataset))
    return ax


def plot_traces_heatmap_for_experiment(ophys_experiment_id, ax=None):
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    dff_traces = dataset.dff_traces.dff.values
    dff_traces = np.vstack(dff_traces)
    if ax is None:
        figsize = (14, 5)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    # ax.pcolormesh(dff_traces, cmap='magma', vmin=0, vmax=0.5)
    ax = sns.heatmap(dff_traces, cmap='magma', vmin=0, vmax=0.5, cbar_kws={'label': 'dF/F'}, ax=ax)
    ax.set_ylim(-0.5, dff_traces.shape[0] + 0.5)
    ax.set_ylabel('cells')
    ax.set_xlabel('2P frames')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'dff_traces_heatmap', get_metadata_string(dataset))
    return ax


def plot_cell_roi_masks_and_dff_traces_for_experiment(ophys_experiment_id, save_figure=True):
    """
    For each cell in the experiment, plot the ROI mask and dff traces, and save to single cell plots directory
    :param ophys_experiment_id:
    :param save_figure:
    :return:
    """
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    cell_roi_ids = dataset.cell_specimen_table.cell_roi_id.values
    for cell_roi_id in cell_roi_ids:
        scp.plot_cell_roi_mask_and_dff_trace(dataset, cell_roi_id, save_figure=save_figure)


def plot_csid_snr_for_experiment(ophys_experiment_id, ax=None):
    experiment_df = processing.ophys_experiment_info_df(ophys_experiment_id)
    exp_snr = processing.experiment_cell_specimen_id_snr_table(ophys_experiment_id)
    exp_snr["stage_name_lims"] = experiment_df["stage_name_lims"][0]
    exp_stage_color_dict = pu.experiment_id_stage_color_dict_for_experiment(ophys_experiment_id)
    if ax is None:
        figsize = (6,4)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    ax = sns.violinplot(x="stage_name_lims", y="robust_snr", data=exp_snr.loc[exp_snr["snr_zscore"] < 3],
                        color=exp_stage_color_dict[ophys_experiment_id])
    ax.set_ylabel("robust snr")
    ax.set_xlabel("stage name")
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'cell_trace_snr', get_metadata_string(dataset))
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
    experiment_df = processing.ophys_experiment_info_df(ophys_experiment_id)
    exp_stage_color_dict = pu.map_stage_name_colors_to_ophys_experiment_ids(experiment_df)
    average_intensity, frame_numbers = processing.get_experiment_average_intensity_timeseries(ophys_experiment_id)
    if ax is None:
        figsize = (6,4)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False

    ax.plot(frame_numbers, average_intensity,
            color=exp_stage_color_dict[ophys_experiment_id],
            label=experiment_df["stage_name_lims"][0])
    ax.set_ylabel('fluorescence value')
    ax.set_xlabel('frame #')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'average_intensity_timeseries', get_metadata_string(dataset))
    return ax


def plot_motion_correction_xy_shift_for_experiment(ophys_experiment_id, ax=None):
    # df = loading.load_rigid_motion_transform_csv(ophys_experiment_id)
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    df = dataset.motion_correction.copy()
    timestamps = dataset.ophys_timestamps
    if ax is None:
        figsize = (20,4)
        fig, ax = plt.subplots(figsize=figsize)
        save_figure = True
    else:
        save_figure = False
    ax.plot(timestamps, df.x.values, color=sns.color_palette()[3], label='x_shift')
    ax.plot(timestamps, df.y.values, color=sns.color_palette()[2], label='y_shift')
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.legend(fontsize='x-small', loc='upper right')
    ax.set_xlabel('time (sec)')
    ax.set_ylabel('pixels')
    # get metrics from saved file and add to plot title
    save_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/motion_correction'
    # motion_df = pd.read_csv(os.path.join(save_dir, 'motion_correction_values_passing_experiments.csv'))
    motion_df = pd.read_hdf(os.path.join(save_dir, 'motion_correction_values_all_experiments.h5'), key='df')
    motion_df = motion_df.set_index('ophys_experiment_id')
    cols_to_plot = ['x_mean', 'x_min', 'x_max', 'x_range', 'x_std',
                    'y_mean', 'y_min', 'y_max', 'y_range', 'y_std']
    row_data = motion_df.loc[ophys_experiment_id]
    title = str(ophys_experiment_id) + ' - '
    for col in cols_to_plot:  # plot all metric values
        title = title + col + ': ' + str(np.round(row_data[col], 2)) + ', '
    if len(row_data.values_over_threshold) > 0:
        title = title + '\n outlier for: '
        for col in row_data.values_over_threshold:
            title = title + col + ', '
    ax.set_title(title)
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'motion_correction_xy_shift', get_metadata_string(dataset))
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


def make_pupil_area_plot(ophys_experiment_id, ax=None, label_x=True):
    '''plot pupil area vs time'''
    try:
        # ophys_session_id = db.convert_id({'ophys_experiment_id': ophys_experiment_id}, 'ophys_session_id')
        # ed = EyeTrackingData(ophys_session_id)
        dataset = loading.get_ophys_dataset(ophys_experiment_id)
        ed = dataset.eye_tracking.copy()
        time = ed['time'].values  # might need to be updated to timestamps in the future'
        area = ed['pupil_area'].values  # this should be blink corrected - no giant spikes
        if ax is None:
            figsize = (20,4)
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(time, area)
            save_figure = True
        else:
            save_figure = False
        if label_x:
            ax.set_xlabel('time (minutes)')
        ax.set_ylabel('pupil diameter\n(pixels$^2$)')
        ax.set_xlim(min(time), max(time))

        ax.set_title('ophys_experiment_id = {}, pupil area vs. time'.format(ophys_experiment_id), ha='center')

    except Exception as e:
        ax.axis('off')

        error_text = 'could not generate pupil area plot for ophys_experiment_id {}\n{}'.format(ophys_experiment_id, e)
        ax.set_title(error_text, ha='left')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'pupil_area_plot', get_metadata_string(dataset))
    return ax


def make_pupil_area_plot_sdk(ophys_experiment_id, ax=None, label_x=True):
    '''plot pupil area vs time'''
    try:
        dataset = loading.get_ophys_dataset(ophys_experiment_id)
        et = dataset.eye_tracking.copy()
        # filtered = et[et.likely_blink == False]
        time = et['time'].values / 60.  # might need to be updated to timestamps in the future'
        area = et['pupil_area_raw'].values  # this will have blink artifacts in it
        if ax is None:
            figsize = (20,4)
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(time, area)
            save_figure = True
        else:
            save_figure = False
        if label_x:
            ax.set_xlabel('time (seconds)')
        ax.set_ylabel('pupil diameter\n(pixels$^2$)')
        ax.set_xlim(min(time), max(time))

        ax.set_title('ophys_experiment_id = {}, pupil area vs. time'.format(ophys_experiment_id), ha='center')

    except Exception as e:
        ax.axis('off')

        error_text = 'could not generate pupil area plot for ophys_experiment_id {}\n{}'.format(ophys_experiment_id, e)
        ax.set_title(error_text, ha='left')
    return ax


def make_pupil_position_plot(ophys_experiment_id, ax=None, label_x=True):
    '''plot pupil position vs time'''
    try:
        dataset = loading.get_ophys_dataset(ophys_experiment_id)
        ed = dataset.eye_tracking.copy()

        time = ed['time'].values / 60.  # might need to be updated to timestamps in the future'
        x = ed['pupil_center_x'].values  # i actually have no idea what these are called
        y = ed['pupil_center_y'].values  # need to check eye_tracking table in SDK and replace with proper names

        if ax is None:
            figsize = (20, 4)
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(time, x, color='darkorange')
            ax.plot(time, y, color='olive')
            save_figure = True
        else:
            save_figure = False

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
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'pupil_position', get_metadata_string(dataset))
    return ax


def plot_cell_snr_distribution_for_experiment(ophys_experiment_id, ax=None):
    import visual_behavior.data_access.processing as processing
    if ax is None:
        fig, ax = plt.subplots()
        save_figure = True
    else:
        save_figure = False
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    dff_traces = processing.compute_robust_snr_on_dataframe(dataset.dff_traces.copy())
    ax.hist(dff_traces.robust_snr.values)
    ax.set_xlabel('robust_snr')
    ax.set_ylabel('n_cells')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'cell_trace_robust_snr', get_metadata_string(dataset))
    return ax


def plot_behavior_timeseries_for_experiment(ophys_experiment_id, xlim_seconds=None, plot_stimuli=False,
                                            plot_face_motion_energy=False, save_figure=False, ax=None):
    """
    Plots the population average dFF trace for an experiment, along with licking behavior, rewards, running speed, pupil area, and face motion.
    Useful to visualize whether the overal activity tracks the behavior variables
    """
    import visual_behavior.visualization.ophys.summary_figures as sf

    dataset = loading.get_ophys_dataset(ophys_experiment_id, load_from_lims=True)

    if xlim_seconds is None:
        xlim_seconds = [dataset.stimulus_timestamps[0], dataset.stimulus_timestamps[-1]]

    lick_timestamps = dataset.licks["timestamps"].values
    licks = np.ones(len(lick_timestamps))

    reward_timestamps = dataset.rewards.index.values  # the index is "timestamps"
    rewards = np.zeros(len(reward_timestamps))

    running_speed = dataset.running_speed["speed"].values
    running_timestamps = dataset.running_speed["timestamps"].values

    pupil_area = dataset.eye_tracking["pupil_area"].values
    pupil_timestamps = dataset.eye_tracking["timestamps"].values

    if ax is None:
        save_figure = True
        if plot_face_motion_energy:
            figsize = (20, 8)
            fig, ax = plt.subplots(4, 1, figsize=figsize, sharex=True)
        else:
            figsize = (20, 6)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    else:
        save_figure = False
    colors = sns.color_palette()
    ax[2].plot(lick_timestamps, licks, '|', label='licks', color='gray')
    ax[2].plot(reward_timestamps, rewards, 'o', label='rewards', color=colors[9])
    ax[2].set_ylim(-2, 3)
    ax[2].legend(fontsize='x-small')
    ax[2].set_yticklabels([])
    ax[0].plot(running_timestamps, running_speed, label='running_speed', color=colors[4])
    ax[0].set_ylabel('run speed\n(cm/s)')
    ax[1].plot(pupil_timestamps, pupil_area, label='pupil_area', color=colors[1])
    ax[1].set_ylabel('pupil area\n(pixels**2)')
    ax[1].set_ylim(-100, 30000)
    ax[1].set_xlim(pupil_timestamps[0], pupil_timestamps[-1])
    if plot_face_motion_energy:
        try:
            face_motion = dataset.behavior_movie_pc_activations[:, 0]
            face_timestamps = dataset.timestamps['behavior_monitoring'].timestamps
            ax[3].plot(face_timestamps, face_motion, label='face_motion_PC0', color=colors[3])
            ax[3].set_ylabel('face motion\n PC0 activation')
            i = 3
        except Exception as e:
            print(ophys_experiment_id)
            print(e)
    else:
        i = 2

    for x in range(i + 1):
        ax[x].set_xlim(xlim_seconds)
        if plot_stimuli:
            ax[x] = sf.add_stim_color_span(dataset, ax[x], xlim=xlim_seconds)
        ax[x].tick_params(which='both', bottom=False, top=False, right=False, left=True,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True)
    ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=True,
                      labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    ax[i].set_xlabel('time (sec)')
    #     ax[x].legend(loc='upper left', fontsize='x-small')
    plt.subplots_adjust(wspace=0, hspace=0.1)
    #     ax[0].set_title(dataset.metadata_string)
    if save_figure:
        utils.save_figure(fig, figsize, utils.get_experiment_plots_dir(), 'population_activity_and_behavior',
                          get_metadata_string(dataset) + '_population_activity_and_behavior')
        plt.close()
    return ax


def plot_high_low_snr_trace_examples(experiment_id, xlim_seconds=None, plot_stimuli=False, ax=None):
    dataset = loading.get_ophys_dataset(experiment_id, include_invalid_rois=False)
    if xlim_seconds is None:
        xlim_seconds = [dataset.ophys_timestamps[0], dataset.ophys_timestamps[-1]]
    #     inds = [0, len(dataset.ophys_timestamps) - 1]
    # else:
    #     inds = [int(xlim_seconds[0] * dataset.metadata['ophys_frame_rate']),
    #             int(xlim_seconds[-1] * dataset.metadata['ophys_frame_rate'])]

    dff_traces = dataset.dff_traces.copy()
    dff_traces = processing.compute_robust_snr_on_dataframe(dff_traces)
    events = dataset.events.copy()

    lowest_snr = np.argsort(dff_traces[dff_traces.robust_snr.isnull() == False].robust_snr.values)[:2]
    lowest_cells = dff_traces[dff_traces.robust_snr.isnull() == False].cell_roi_id.values[lowest_snr]
    highest_snr = np.argsort(dff_traces[dff_traces.robust_snr.isnull() == False].robust_snr.values)[-4:]
    highest_cells = dff_traces[dff_traces.robust_snr.isnull() == False].cell_roi_id.values[highest_snr]

    cell_roi_ids = np.hstack((lowest_cells, highest_cells))

    colors = sns.color_palette()
    if ax is None:
        figsize = (15, 10)
        fig, ax = plt.subplots(len(cell_roi_ids), 1, figsize=figsize, sharex=True)
        ax = ax.ravel()
    for i, cell_roi_id in enumerate(cell_roi_ids):
        dff_trace = dff_traces[dff_traces.cell_roi_id == cell_roi_id].dff.values[0]
        ax[i].plot(dataset.ophys_timestamps, dff_trace, color=colors[0], label='dff_trace')
        ax[i].plot(dataset.ophys_timestamps, events[events.cell_roi_id == cell_roi_id].events.values[0],
                   color=colors[3], label='events')
        ax[i].set_xlim(xlim_seconds)

        frame_range = [int(time * dataset.metadata['ophys_frame_rate']) for time in xlim_seconds]
        ymin = np.min(dff_trace[frame_range[0]:frame_range[1]]) - (
            np.min(dff_trace[frame_range[0]:frame_range[1]]) * .05)
        ymax = np.max(dff_trace[frame_range[0]:frame_range[1]]) * 1.2
        ax[i].set_ylim(ymin, ymax)
        ax[i].set_ylabel('dF/F')
        #         ax[i].set_title(str(cell_roi_id))
        if plot_stimuli:
            ax[i] = sf.add_stim_color_span(dataset, ax[i], xlim=xlim_seconds)
    ax[i].legend(loc='upper left', fontsize='x-small')
    ax[i].set_xlabel('time (seconds)')
    return ax


def plot_motion_correction_and_population_average(experiment_id, ax=None):
    if ax is None:
        figsize = (20, 10)
        fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
        save_figure = True
    else:
        save_figure = False

    dataset = loading.get_ophys_dataset(experiment_id)
    timestamps = dataset.ophys_timestamps
    corrected_traces = dataset.corrected_fluorescence_traces.corrected_fluorescence.values
    corrected_traces = np.vstack(corrected_traces)
    population_average_np = np.nanmean(corrected_traces, axis=0)
    ax[0].plot(timestamps, population_average_np)
    ax[0].set_ylabel('fluorescence')
    ax[0].set_xlabel('time (sec)')
    ax[0].set_xlim(timestamps[0], timestamps[-1])

    dff_traces = dataset.dff_traces.dff.values
    dff_traces = np.vstack(dff_traces)
    population_average_dff = np.nanmean(dff_traces, axis=0)
    ax[1].plot(timestamps, population_average_dff)
    ax[1].set_ylabel('dF/F')
    ax[1].set_xlabel('time (sec)')
    ax[1].set_xlim(timestamps[0], timestamps[-1])

    ax[2] = plot_motion_correction_xy_shift_for_experiment(experiment_id, ax=ax[2])

    #     running_speed = dataset.running_speed.speed.values
    #     running_timestamps = dataset.running_speed.timestamps.values
    #     ax[1].plot(running_timestamps, running_speed, label='running_speed', color=sns.color_palette()[4])
    #     ax[1].set_ylabel('run speed\n(cm/s)')
    #     ax[1].set_xlim(running_timestamps[0], running_timestamps[-1])
    #     ax[1].set_xlabel('time (sec)')
    if save_figure:
        utils.save_figure(fig, figsize, loading.get_experiment_plots_dir(), 'pop_avg_and_motion_corr', get_metadata_string(dataset))

    return ax


def plot_remaining_decrosstalk_masks_for_experiment(experiment_id, ax=None):
    dataset = loading.get_ophys_dataset(experiment_id, include_invalid_rois=True)
    remaining_crosstalk_dict = loading.get_remaining_crosstalk_amount_dict(experiment_id)
    decrosstalk_rois = [int(cell_roi_id) for cell_roi_id in list(remaining_crosstalk_dict.keys())]
    roi_masks = {k: dataset.roi_masks[k] for k in decrosstalk_rois}
    # cmap_range = [np.nanmin(np.asarray(list(remaining_crosstalk_dict.values()))), np.nanmax(np.asarray(list(remaining_crosstalk_dict.values())))]
    cmap_range = [0, 50]
    # if cmap_range[0] < -50:
    #     cmap_range[0] = 50
    # if cmap_range[1] > 100:
    #     cmap_range[1] = 100
    ax = plot_metrics_mask(roi_masks, remaining_crosstalk_dict, 'remaining_crosstalk', dataset.max_projection.data, title=None, outlines=False,
                           cmap='viridis', cmap_range=cmap_range, ax=ax, colorbar=True)
    # cell_specimen_table = dataset.cell_specimen_table.copy()
    # for cell_roi_id in cell_specimen_table[cell_specimen_table.valid_roi == True].cell_roi_id.values:
    #     mask = cell_specimen_table[cell_specimen_table.cell_roi_id == cell_roi_id].roi_mask.values[0]
    #     ax.contour(mask, levels=0, colors=['red'], linewidths=[0.6])
    return ax


def plot_event_detection_for_experiment(ophys_experiment_id, save_figure=True):
    """
    Generates plots of dFF traces and events for each cell in an experiment, for different segments of time.
    Useful to validate whether detected events line up with dFF transients.
    :param ophys_experiment_id:
    :param save_figure:
    :return:
    """
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    metadata_string = get_metadata_string(dataset)
    colors = sns.color_palette()
    ophys_timestamps = dataset.ophys_timestamps.copy()
    dff_traces = dataset.dff_traces.copy()
    events = dataset.events.copy()

    for cell_specimen_id in dataset.cell_specimen_ids:
        n_rows = 5
        figsize = (15, 10)
        fig, ax = plt.subplots(n_rows, 1, figsize=figsize)
        ax = ax.ravel()
        x = 0
        for i in range(n_rows):
            ax[i].plot(ophys_timestamps, dff_traces.loc[cell_specimen_id].dff, color=colors[0], label='dff_trace')
            ax[i].plot(ophys_timestamps, events.loc[cell_specimen_id].events, color=colors[3], label='events')
            ax[i].set_xlim((60 * 10) + (x * 60), (60 * 10) + 90 + (x * 60))
            x = x + 5
        ax[0].set_title('oeid: ' + str(ophys_experiment_id) + ', csid: ' + str(cell_specimen_id))
        ax[i].legend(loc='upper left')
        ax[i].set_xlabel('time (seconds)')
        fig.tight_layout()

        if save_figure:
            utils.save_figure(fig, figsize, loading.get_single_cell_plots_dir(), 'event_detection',
                              str(cell_specimen_id) + '_' + metadata_string + '_events_validation')


def plot_dff_trace_and_behavior_for_experiment(ophys_experiment_id, save_figure=True):
    """
    Plots the full dFF trace for each cell, along with licking behavior, rewards, running speed, pupil area, and face motion.
    Useful to visualize whether the dFF trace tracks the behavior variables
    """
    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    for cell_specimen_id in dataset.dff_traces.index.values:
        scp.plot_single_cell_activity_and_behavior(dataset, cell_specimen_id, save_figure=save_figure)


def plot_population_activity_and_behavior_for_experiment(ophys_experiment_id, save_figure=True):
    """
    Plots the population average dFF trace for an experiment, along with licking behavior, rewards, running speed, pupil area, and face motion.
    Useful to visualize whether the overal activity tracks the behavior variables
    """
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    traces = dataset.dff_traces.copy()
    trace_timestamps = dataset.ophys_timestamps

    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))

    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values

    pupil_area = dataset.eye_tracking.pupil_area.values
    pupil_timestamps = dataset.eye_tracking.timestamps.values

    try:
        face_motion = dataset.behavior_movie_pc_activations[:, 0]
        face_timestamps = dataset.timestamps['eye_tracking'].timestamps
    except BaseException:
        pass

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
    try:
        ax[4].plot(face_timestamps, face_motion, label='face_motion_PC0', color=colors[2])
        ax[4].set_ylabel('face motion\n PC0 activation')
    except BaseException:
        pass

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
                          get_metadata_string(dataset) + '_population_activity_and_behavior')
        plt.close()


def plot_population_average_for_experiment(experiment_id, response_df, mean_df, df_name, trace_type='mean_trace', color=None,
                                           label=None, ax=None):
    """
    """
    import visual_behavior.visualization.qc.container_plots as cp

    if 'trials' in df_name:
        omitted = False
        trial_type = 'go'
        xlabel = 'time relative to change (sec)'
    elif 'omission' in df_name:
        omitted = True
        trial_type = None
        xlabel = 'time relative to omission (sec)'
    elif 'stimulus' in df_name:
        omitted = False
        trial_type = None
        xlabel = 'time (sec)'

    if color is None:
        color = sns.color_palette()[0]
    if label is None:
        label = ''

    if ax is None:
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize)

    timestamps = response_df.trace_timestamps.mean()

    traces = mean_df[trace_type].values
    mean_trace = mean_df[trace_type].mean()
    ax.plot(timestamps, mean_trace, color=color, label=label)
    sem = (traces.std()) / np.sqrt(float(len(traces)))
    ax.fill_between(timestamps, mean_trace + sem, mean_trace - sem, alpha=0.5, color=color)
    ax = cp.plot_flashes_on_trace(ax, timestamps, trial_type=trial_type, omitted=omitted, alpha=0.2, facecolor='gray')
    ax.set_xlabel(xlabel)
    if omitted:
        ax.axvline(x=0, ymin=0, ymax=1, linestyle='--', color='gray')
    ax.set_ylabel('dF/F')
    ax.set_xlim(timestamps[0], timestamps[-1])
    return ax


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
    """
    Creates a plot showing ROI masks matched between production and development versions of segmentation classifier.

    This is a quick and dirty function to get plots needed for a rapid decision to be made
    """
    import visual_behavior.visualization.ophys.summary_figures as sf
    from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    classification_threshold = 0.4

    expt = ophys_experiment_id
    # get new classifier output
    data = pd.read_csv(r"//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/classifier_validation/inference_annotated_threshold_" + str(
        classification_threshold) + ".csv", dtype={'production_id': 'Int64', 'cell_roi_id': 'Int64'})
    # get suite2P segmentation output
    output_dir = r'//allen/aibs/informatics/danielk/dev_LIMS/new_labeling'
    folder = [folder for folder in os.listdir(output_dir) if str(expt) in folder]
    segmentation_output_file = os.path.join(output_dir, folder[0], 'binarize_output.json')
    cell_table = get_suite2p_rois(segmentation_output_file)
    cell_table['experiment_id'] = expt
    # move suite2P masks to the proper place
    dataset = loading.get_ophys_dataset(expt, include_invalid_rois=True)
    # cell_table = place_masks_in_full_image(cell_table, dataset.max_projection.data)
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
    # dataset = loading.get_ophys_dataset(expt, include_invalid_rois=True)
    ct = dataset.cell_specimen_table.copy()
    roi_masks = dataset.roi_masks.copy()
    max_projection = dataset.max_projection.data
    dff_traces = dataset.dff_traces.copy()
    ophys_timestamps = dataset.ophys_timestamps
    metadata_string = get_metadata_string(dataset)
    # get average response df
    analysis = ResponseAnalysis(dataset)
    sdf = analysis.get_response_df(df_name='stimulus_response_df')

    # plots for all ROIs in experiment
    expt_data = data[data.experiment_id == expt].copy()

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
            ax0.imshow(max_projection, cmap='gray')
            ax0.imshow(masks_to_plot, cmap='hsv', vmin=0, vmax=1, alpha=0.5)
            ax0.set_title('production roi mask')
            ax0.axis(False)

            valid = ct.loc[cell_specimen_id].valid_roi
            ax1 = sf.plot_cell_zoom(roi_masks, max_projection, cell_specimen_id, spacex=40, spacey=60, show_mask=True,
                                    ax=ax1)
            ax1.set_title('production roi mask, valid: ' + str(valid))
            if present_in == 'prod':
                if valid == True:
                    folder = 'prod_valid_not_in_dev'
                elif valid == False:
                    folder = 'prod_invalid_not_in_dev'
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
            y_score = np.round(expt_data[expt_data.roi_id == roi_id].y_score.values[0], 3)
            masks_array = cell_table[cell_table.roi_id == roi_id]['roi_mask'].values[0]
            masks_array[masks_array == 0] = np.nan
            ax2.imshow(max_projection, cmap='gray')
            ax2.imshow(masks_array, cmap='hsv', vmin=0, vmax=1, alpha=0.5)
            ax2.set_title('suite2p roi mask\nprediction_score = ' + str(y_score))
            ax2.axis('off')

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
            masks_array = cell_table[cell_table.valid_roi == True]['roi_mask'].values
            masks_to_plot = processing.gen_transparent_multi_roi_mask(masks_array)
            masks_to_plot = np.sum(masks_array, 0)
            ax2.imshow(masks_to_plot, cmap='hsv', vmin=0, vmax=1, alpha=0.5)
            ax2.set_title('valid suite2p roi masks')

            masks_array = cell_table[cell_table.valid_roi == False]['roi_mask'].values
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
        # save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\qc_plots\classifier_validation\CNN_rois'
        save_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/classifier_validation/last_ditch_effort_annotation/classification_threshold_' + str(
            classification_threshold)
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(cell_roi_id) + '_' + str(roi_id))


def plot_metrics_mask(roi_mask_dict, metrics_dict, metric_name, max_projection=None, title=None, outlines=False,
                      cmap='RdBu', cmap_range=[0, 1], ax=None, colorbar=False):
    """
        roi_mask_dict: dictionary with keys as cell_specimen_id or cell_roi_id and values as the ROI masks,
                        placed within the full 512x512 image
        metrics_dict: dictionary with keys as cell_specimen_id or cell_roi_id and corresponding metric value for each ROI
        metric_name: name of metric provided to be used for colorbar label and filename of saved figure
        max_projection: maximum intensity projection. If None, only ROI masks will be shown, without max projection overlay.
        cmap_range: min and max value of metric to scale image by
        cmap: colormap to use
        ax: if axis is provided, image will be plotted on that axis. If None, a figure and axis will be created.
        colorbar: Boolean to indicate whether colorbar is displayed
        """
    if ax is None:
        figsize = (6, 6)
        fig, ax = plt.subplots(figsize=figsize)
    if cmap_range is None:
        cmap_range = [np.nanmin(np.asarray(list(metrics_dict.values()))), np.nanmax(np.asarray(list(metrics_dict.values())))]
    if max_projection is not None:
        ax.imshow(max_projection, cmap='gray', vmin=0, vmax=np.percentile(max_projection, 99))
    for i, roi_id in enumerate(list(roi_mask_dict.keys())):
        tmp = roi_mask_dict[roi_id]
        mask = np.empty(tmp.shape, dtype=np.float)
        mask[:] = np.nan
        mask[tmp == 1] = metrics_dict[roi_id]
        cax = ax.imshow(mask, cmap=cmap, alpha=0.5, vmin=cmap_range[0], vmax=cmap_range[1])
        ax.set_title(title)
        ax.grid(False)
        ax.axis('off')
    if colorbar:
        cbar = plt.colorbar(cax, ax=ax, use_gridspec=True, fraction=0.046, pad=0.04)
        cbar.set_label(metric_name)
    return ax


def plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=True, ax=None):
    dataset = loading.get_ophys_dataset(ophys_experiment_id, include_invalid_rois=include_invalid_rois)
    cell_table = dataset.cell_specimen_table.copy()
    metrics_df = loading.get_metrics_df(ophys_experiment_id)

    roi_mask_dict, metrics_dict = loading.get_roi_mask_and_metrics_dict(cell_table, metrics_df, metric)

    if metric == 'area':
        cmap_range = [20, 400]
    elif metric == 'mean_intensity':
        cmap_range = [0, 200]
    elif metric == 'ellipseness':
        cmap_range = [0.1, 0.9]
    elif metric == 'compactness':
        cmap_range = [8, 22]
    elif metric == 'filtered_masks':
        cmap_range = [0, 1]
    else:
        cmap_range = [np.min(list(metrics_dict.values())), np.max(list(metrics_dict.values()))]

    if ax is None:
        fig, ax = plt.subplots()

    ax = plot_metrics_mask(roi_mask_dict, metrics_dict, metric, max_projection=dataset.max_projection.data,
                           title=metric, cmap_range=cmap_range, cmap='viridis', ax=ax, colorbar=True)
    return ax


def plot_filtered_masks_for_experiment(ophys_experiment_id, include_invalid_rois=True, ax=None):
    dataset = loading.get_ophys_dataset(ophys_experiment_id, include_invalid_rois=include_invalid_rois)
    max_projection = dataset.max_projection.data
    cell_table = dataset.cell_specimen_table.copy()
    metrics_df = loading.get_metrics_df(ophys_experiment_id)

    filtered_metrics = metrics_df[
        (metrics_df.area > 40) & (metrics_df.ellipseness > 0.2) & (metrics_df.compactness < 18)]
    filtered_metrics['filtered_masks'] = 1
    cell_table = cell_table[cell_table.cell_roi_id.isin(filtered_metrics.cell_roi_id.unique())]
    metric = 'filtered_masks'
    roi_mask_dict, metrics_dict = loading.get_roi_mask_and_metrics_dict(cell_table, filtered_metrics, metric)

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(max_projection, cmap='gray', vmin=0, vmax=np.percentile(max_projection, 99))

    for cell_roi_id in list(roi_mask_dict.keys()):
        mask = roi_mask_dict[cell_roi_id]
        mask[np.isnan(mask) == True] = 0
        ax.contour(mask, levels=0, colors=['red'], linewidths=[0.6])
    ax.axis('off')

    # cmap_range = [0, 1]
    # ax = plot_metrics_mask(roi_mask_dict, metrics_dict, metric, max_projection=dataset.max_projection.data,
    #                           title=metric, cmap_range=cmap_range, cmap='hsv', ax=ax, colorbar=False)
    ax.set_title('area > 40\nellipseness > 0.2\ncompactness < 18')
    return ax
