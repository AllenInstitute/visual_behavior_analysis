import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import visual_behavior.plotting as vbp
import visual_behavior.database as db
from visual_behavior.database import lims_query
import imageio

from pathlib import PureWindowsPath
import platform

from visual_behavior.data_access import loading as loading
from visual_behavior.data_access import utilities as utilities
from visual_behavior.data_access import processing as processing
from visual_behavior.visualization import utils as ut
from visual_behavior.visualization.qc import session_plots as sp
from visual_behavior.visualization.qc import plotting_utils as pu
from visual_behavior.visualization.qc import experiment_plots as ep
from visual_behavior.visualization.qc import single_cell_plots as scp


def ax_to_array(ax):
    '''
    convert any axis elements that aren't already an array into an array
    '''
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    return ax

# Container sequence


def plot_container_session_sequence(ophys_container_id, save_figure=True):
    experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
    expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values('date_of_acquisition')
    specimen_id = expts.specimen_id.unique()[0]
    experiment_ids = expts.index.values
    session_type_color_map = ut.get_session_type_color_map()

    n_expts = len(expts)
    img = np.empty((n_expts, 1, 3))
    fail_x = []
    fail_tags = []
    for expt_ind, expt_id in enumerate(experiment_ids):
        this_expt = expts.loc[expt_id]
        img[expt_ind, 0, :] = session_type_color_map[this_expt['session_type']]
        if this_expt['experiment_workflow_state'] == 'failed':
            fail_x.append(expt_ind)
            fail_tags.append(this_expt['failure_tags'])

    # create plot with expt colors image
    figsize = (20, n_expts)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img.astype(int))
    ax.axis('off')

    # plot text for acquisition date and session type
    for i, expt_id in enumerate(experiment_ids):
        this_expt = expts.loc[expt_id]
        ax.text(x=0.75, y=i, s=str(this_expt['date_of_acquisition']).split(' ')[0],
                ha='left', va='center', fontsize=20)
        ax.text(x=3, y=i, s=this_expt['session_type'], ha='left', va='center', fontsize=20)
        ax.text(x=20, y=i, s=' ')

    # add X for fails and list fail tags
    for ind_fail, x in enumerate(fail_x):
        ax.text(x=0, y=x, s='X', ha='center', va='center', fontsize=60)
        fail_string = 'Failure: ' + str(fail_tags[ind_fail])
        ax.text(x=8.5, y=x, s=fail_string, ha='left', va='center', fontsize=20)

    plt.suptitle('specimen_id: {}'.format(specimen_id) + ', ophys_container_id: {}'.format(ophys_container_id),
                 fontsize=25, ha='left', x=0.06, y=.97)
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(right=0.1)
    fig.subplots_adjust(top=0.9)
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'ophys_session_sequence',
                       get_file_name_for_container(ophys_container_id))


# OPHYS

def plot_sdk_max_projection_images_for_container(ophys_container_id, save_figure=True):
    """for every passed experiment in an experiment container, gets plots
        the max intensity projections from the sdk (normalized) next to one another
        in order of acquisition date

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    # exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    # ophys_experiment_ids = list(exp_order_and_stage["ophys_experiment_id"])
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    ax = ax_to_array(ax)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        # exp_stage_name = exp_order_and_stage.loc[exp_order_and_stage["ophys_experiment_id"]== ophys_experiment_id, "stage_name_lims"].reset_index(drop=True)[0]
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'max_intensity_projection',
                       get_file_name_for_container(ophys_container_id))


def plot_movie_max_projection_images_for_container(ophys_container_id, save_figure=True):
    """for every passed experiment in an experiment container, gets plots
        the max intensity of the motion corrected movie (unnormalized) next to one another
        in order of acquisition date

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    # exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    # ophys_experiment_ids = list(exp_order_and_stage["ophys_experiment_id"])
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    ax = ax_to_array(ax)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_motion_correction_max_image_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        # exp_stage_name = exp_order_and_stage.loc[exp_order_and_stage["ophys_experiment_id"]== ophys_experiment_id, "stage_name_lims"].reset_index(drop=True)[0]
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'max_intensity_projection_movies',
                       get_file_name_for_container(ophys_container_id))


def plot_sdk_average_images_for_container(ophys_container_id, save_figure=True):
    """for every passed experiment in an experiment container, gets plots
        the average intensity projections from the sdk (normalized) next to one another
        in order of acquisition date

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    # exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    # ophys_experiment_ids = list(exp_order_and_stage["ophys_experiment_id"])
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    ax = ax_to_array(ax)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_average_image_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        # exp_stage_name = exp_order_and_stage.loc[exp_order_and_stage["ophys_experiment_id"]== ophys_experiment_id, "stage_name_lims"].reset_index(drop=True)[0]
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'average_images',
                       get_file_name_for_container(ophys_container_id))


def plot_movie_average_images_for_container(ophys_container_id, save_figure=True):
    """for every passed experiment in an experiment container, gets plots
        the average intensity of the motion corrected movie (unnormalized) next to one another
        in order of acquisition date

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    # exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    # ophys_experiment_ids = list(exp_order_and_stage["ophys_experiment_id"])
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    ax = ax_to_array(ax)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_motion_correction_average_image_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        # exp_stage_name = exp_order_and_stage.loc[exp_order_and_stage["ophys_experiment_id"]== ophys_experiment_id, "stage_name_lims"].reset_index(drop=True)[0]
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'average_images_movies',
                       get_file_name_for_container(ophys_container_id))


def plot_eye_tracking_sample_frames(ophys_container_id, save_figure=True):
    table = loading.get_filtered_ophys_experiment_table()
    table = table.reset_index()
    ophys_experiment_ids = table.query('ophys_container_id == {}'.format(ophys_container_id)).sort_values(by='date_of_acquisition')['ophys_experiment_id']

    figsize = (16, 5 * len(ophys_experiment_ids))
    fig = plt.figure(figsize=figsize)
    axes = []
    nplots = len(ophys_experiment_ids)
    buffer = 0.05
    for ii, ophys_experiment_id in enumerate(ophys_experiment_ids):
        print('on ophys_experiment_id {}, #{} of {}'.format(ophys_experiment_id, ii + 1, nplots))
        axes.append(vbp.placeAxesOnGrid(fig, dim=(3, 10), xspan=(0, 1), yspan=(ii / nplots + buffer, (ii + 1) / nplots)))
        axes[-1] = ep.make_eye_matrix_plot(ophys_experiment_id, axes[-1])

    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'eyetracking_sample_frames',
                       get_file_name_for_container(ophys_container_id))

    return fig, axes


def plot_segmentation_masks_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    ax = ax_to_array(ax)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_valid_segmentation_mask_outlines_per_cell_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'segmentation_masks',
                       get_file_name_for_container(ophys_container_id))


def plot_segmentation_mask_overlays_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 18)
    n = len(ophys_experiment_ids)
    fig, ax = plt.subplots(4, n, figsize=figsize)
    ax = ax.ravel()
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):

        ax[i] = ep.plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

        try:
            ax[i + n] = ep.plot_valid_segmentation_mask_outlines_per_cell_for_experiment(ophys_experiment_id, ax=ax[i + n])
        except Exception as e:
            print('cant plot valid masks for', ophys_experiment_id)
            print('error: {}'.format(e))
        # try:
        #     ax[i + (n * 2)] = ep.plot_valid_segmentation_mask_overlay_for_experiment(ophys_experiment_id, ax=ax[i + (n * 2)])
        #     ax[i + (n * 2)].set_title('valid ROI masks')
        # except:
        #     print('cant plot valid masks for', ophys_experiment_id)

        ax[i + (n * 2)] = ep.plot_valid_and_invalid_segmentation_mask_overlay_per_cell_for_experiment(ophys_experiment_id, ax=ax[i + (n * 2)])
        ax[i + (n * 2)].set_title('red=valid, blue=invalid, \ngreen=crosstalk, cyan=both')

        try:
            ax[i + (n * 3)] = ep.plot_remaining_decrosstalk_masks_for_experiment(ophys_experiment_id, ax=ax[i + (n * 3)])
            ax[i + (n * 3)].set_title('remaining crosstalk')
        except Exception as e:
            print('cant plot remaining decrosstalk for', ophys_experiment_id)
            print('error: {}'.format(e))

    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'segmentation_mask_overlays', get_file_name_for_container(ophys_container_id))


def plot_roi_filtering_metrics_for_all_rois_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 22)
    n = len(ophys_experiment_ids)
    fig, ax = plt.subplots(5, n, figsize=figsize)
    ax = ax.ravel()
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):

        ax[i] = ep.plot_valid_and_invalid_segmentation_mask_overlay_per_cell_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type + '\nred = valid, blue = invalid')

        metric = 'area'
        ax[i + n] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=True, ax=ax[i + n])

        metric = 'mean_intensity'
        ax[i + (n * 2)] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=True, ax=ax[i + (n * 2)])

        metric = 'ellipseness'
        ax[i + (n * 3)] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=True, ax=ax[i + (n * 3)])

        metric = 'compactness'
        ax[i + (n * 4)] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=True, ax=ax[i + (n * 4)])

    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'roi_filtering_metrics_all_rois',
                       get_file_name_for_container(ophys_container_id))


def plot_roi_filtering_metrics_for_valid_rois_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 22)
    n = len(ophys_experiment_ids)
    fig, ax = plt.subplots(5, n, figsize=figsize)
    ax = ax.ravel()
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):

        ax[i] = ep.plot_valid_segmentation_mask_overlay_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

        metric = 'area'
        ax[i + n] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=False, ax=ax[i + n])

        metric = 'mean_intensity'
        ax[i + (n * 2)] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=False, ax=ax[i + (n * 2)])

        metric = 'ellipseness'
        ax[i + (n * 3)] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=False, ax=ax[i + (n * 3)])

        metric = 'compactness'
        ax[i + (n * 4)] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=False, ax=ax[i + (n * 4)])

    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'roi_filtering_metrics_valid_rois',
                       get_file_name_for_container(ophys_container_id))


def plot_filtered_roi_masks_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 20)
    n = len(ophys_experiment_ids)
    fig, ax = plt.subplots(5, n, figsize=figsize)
    ax = ax.ravel()
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_valid_and_invalid_segmentation_mask_overlay_per_cell_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type + '\nred = valid, blue = invalid')

        ax[i + n] = ep.plot_filtered_masks_for_experiment(ophys_experiment_id, include_invalid_rois=True, ax=ax[i + n])

        metric = 'area'
        ax[i + (n * 2)] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=True,
                                                              ax=ax[i + (n * 2)])
        metric = 'ellipseness'
        ax[i + (n * 3)] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=True,
                                                              ax=ax[i + (n * 3)])
        metric = 'compactness'
        ax[i + (n * 4)] = ep.plot_metrics_mask_for_experiment(ophys_experiment_id, metric, include_invalid_rois=True,
                                                              ax=ax[i + (n * 4)])
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'filtered_roi_masks',
                       get_file_name_for_container(ophys_container_id))


def plot_dff_traces_heatmaps_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 20)
    fig, ax = plt.subplots(len(ophys_experiment_ids), 1, figsize=figsize)
    ax = ax_to_array(ax)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_traces_heatmap_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + ' - ' + session_type)

    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'dff_traces_heatmaps',
                       get_file_name_for_container(ophys_container_id))


def plot_average_intensity_timeseries_for_container(ophys_container_id, save_figure=True):
    """a seaborn timeseries where all passed experiments in a container are plotted.
        x= is frame number, y = average intensity of the fov and eachc line is an experiment
        the timeseries lines are colored by the experiment stage name in lims and the
        stage name legend is displayed in order of experiment acquisition date

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    container_df = (processing.passed_experiment_info_for_container(ophys_container_id)).sort_values('date_of_acquisition').reset_index(drop=True)
    exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    figsize = (9, 5)
    fig, ax = plt.subplots(figsize=figsize)
    for i, ophys_experiment_id in enumerate(container_df["ophys_experiment_id"].unique()):
        ax = ep.plot_average_intensity_timeseries_for_experiment(ophys_experiment_id, ax=ax)
    ax.legend(exp_order_and_stage["stage_name_lims"], fontsize='xx-small', title='stage name', title_fontsize='xx-small',
              bbox_to_anchor=(1.01, 1), loc=2)
    ax.set_title('full field average fluorescence intensity over time')
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'average_intensity_timeseries',
                       get_file_name_for_container(ophys_container_id))


def plot_pmt_for_container(ophys_container_id, save_figure=True):
    """seaborn scatter plot where x= session stage name
        y= pmt setting for that session

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    pmt_settings = processing.container_pmt_settings(ophys_container_id)
    exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    df = pd.merge(pmt_settings, exp_order_and_stage, how="left", on="ophys_experiment_id")

    stage_color_dict = pu.gen_ophys_stage_name_colors_dict()
    figsize = (6, 9)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(x="stage_name_lims", y="pmt_gain", data=df,
                         hue="stage_name_lims", palette=stage_color_dict,
                         legend=False)
    ax.set_ylim(df["pmt_gain"].min() - 1, df["pmt_gain"].max() + 1)
    plt.xticks(rotation=90)
    plt.ylabel('pmt gain')
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'pmt_settings',
                       get_file_name_for_container(ophys_container_id))


def plot_average_intensity_for_container(ophys_container_id, save_figure=True):
    """seaborn scatter plot where x= session stage name
        y= average intensity of the FOV for the entire session

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    FOV_intensities = processing.container_intensity_mean_and_std(ophys_container_id)
    exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    df = pd.merge(FOV_intensities, exp_order_and_stage, how="left", on="ophys_experiment_id")

    stage_color_dict = pu.gen_ophys_stage_name_colors_dict()
    figsize = (6, 9)
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        x="stage_name_lims",
        y="intensity_mean",
        data=df,
        hue="stage_name_lims",
        palette=stage_color_dict,
        legend=False
    )
    plt.xticks(rotation=90)
    plt.ylabel('FOV average intensity')
    plt.title("FOV mean intensity for container")
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'FOV_average_intensity',
                       get_file_name_for_container(ophys_container_id))


def plot_average_intensity_by_pmt_for_container(ophys_container_id, save_figure=True):
    """a seaborn scatter plot where x = pmt gain setting
        y= mean intensity for the FOV
        the points are the passed experiments in the container,
        colored by stage name from lims.
        The legend is in order of experiment acquisition date

    Arguments:
        ophys_container_id {int} -- 9 digit unique container id

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    pmt_settings = processing.container_pmt_settings(ophys_container_id)
    FOV_intensities = processing.container_intensity_mean_and_std(ophys_container_id)
    exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    df = pd.merge(pmt_settings, FOV_intensities, how="left", on=["ophys_experiment_id", "ophys_container_id"])
    df = pd.merge(df, exp_order_and_stage, how="left", on="ophys_experiment_id")

    stage_color_dict = pu.gen_ophys_stage_name_colors_dict()
    figsize = (9, 5.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(x="pmt_gain", y="intensity_mean", data=df,
                         hue="stage_name_lims", palette=stage_color_dict)
    ax.set_xlim(df["pmt_gain"].min() - 1, df["pmt_gain"].max() + 1)
    ax.legend(exp_order_and_stage["stage_name_lims"], fontsize='xx-small', title='stage name', title_fontsize='xx-small',
              bbox_to_anchor=(1.01, 1), loc=2)
    plt.xlabel('pmt gain')
    plt.ylabel('mean intensity')
    plt.title("FOV mean intensity for experiments by pmt gain")
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'average_intensity_by_pmt',
                       get_file_name_for_container(ophys_container_id))


def plot_snr_by_pmt_gain_and_intensity_for_container(ophys_container_id, save_figure=True):
    df = processing.container_FOV_information(ophys_container_id)
    figsize = (7, 5.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax = plt.scatter(df["pmt_gain"], df["median_rsnr_all_csids"],
                     c=df["intensity_mean"], s=75,
                     cmap="cool", edgecolors='k')
    plt.xlim(df["pmt_gain"].min() - 1, df["pmt_gain"].max() + 1)
    cbar = plt.colorbar(ax)
    cbar.set_label('fov mean intensity', rotation=270, labelpad=25)
    plt.xlabel('pmt gain')
    plt.ylabel('median snr across cells')
    plt.suptitle("median robust snr across cells by pmt gain")
    plt.title("container: " + str(ophys_container_id))
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'snr_by_pmt_and_intensity',
                       get_file_name_for_container(ophys_container_id))


def plot_snr_by_pmt_for_container(ophys_container_id, save_figure=True):
    """a seaborn scatter plot where x = pmt gain setting
        y= median robust snr for all the cell specimen ids in an experiment
        the points are the passed experiments in the container,
        colored by stage name from lims.
        The legend is in order of experiment acquisition date

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    pmt_settings = processing.container_pmt_settings(ophys_container_id)
    snr_summary = processing.container_snr_summary_table(ophys_container_id)
    exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    df = pd.merge(pmt_settings, snr_summary, how="left", on=["ophys_experiment_id", "ophys_container_id"])
    df = pd.merge(df, exp_order_and_stage, how="left", on="ophys_experiment_id")
    stage_color_dict = pu.gen_ophys_stage_name_colors_dict()

    figsize = (9, 5.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(x="pmt_gain", y="median_rsnr_all_csids", data=df,
                         hue="stage_name_lims", palette=stage_color_dict)
    ax.set_xlim(df["pmt_gain"].min() - 1, df["pmt_gain"].max() + 1)
    ax.legend(exp_order_and_stage["stage_name_lims"], fontsize='xx-small', title='stage name', title_fontsize='xx-small',
              bbox_to_anchor=(1.01, 1), loc=2)
    plt.xlabel('pmt gain')
    plt.ylabel('median snr across cells')
    plt.title("robust snr for experiments by pmt gain")
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'snr_by_pmt',
                       get_file_name_for_container(ophys_container_id))


def plot_cell_snr_for_container(ophys_container_id, save_figure=True):
    """a seaborn violin plot where x = experiment stage name ordered
        by experiment acquisition date
        y= robust snr for all the cell specimen ids in an experiment
        with the outliers removed
        the violins are colored by stage name from lims

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    container_snr_table = processing.container_csid_snr_table(ophys_container_id)
    container_snr_table = pd.merge(container_snr_table, exp_order_and_stage, how="left", on="ophys_experiment_id")
    stage_color_dict = pu.gen_ophys_stage_name_colors_dict()
    figsize = (6, 8)
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(x="stage_name_lims", y="robust_snr",
                   data=container_snr_table.loc[container_snr_table["snr_zscore"] < 3],
                   palette=stage_color_dict,
                   order=exp_order_and_stage["stage_name_lims"])
    plt.xticks(rotation=90)
    plt.xlabel("stage name")
    plt.ylabel('robust SNR')
    plt.title("distribution of cell signal to noise ratio", pad=5 )
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'cell_snr_by_experiment',
                       get_file_name_for_container(ophys_container_id))


def plot_number_segmented_rois_for_container(ophys_container_id, save_figure=True):
    """uses seaborn barplot
        x axis is all passed experiments for a container, listed as their stage name from lims
        and listed in their acquisition date order.
        y axis is number of rois
        the bars are:
        total_rois : number of segmented objects
        valid_count: nsumner of valid rois (same thing as cell_specimen_ids)
        invalid_count: number of invalid rois (deemed not a cell, or failed for another reason
                        like being on the motion border etc.)

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    seg_count_df = processing.segmentation_validity_count_for_container(ophys_container_id)
    exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
    plotting_df = pd.merge(seg_count_df, exp_order_and_stage, how="left", on="ophys_experiment_id")
    validity_palette = ['steelblue', 'lightgreen', 'tomato']

    figsize = (6.5, 9)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(x="stage_name_lims", y="roi_count",
                     data=plotting_df,
                     hue="roi_category",
                     palette=validity_palette,
                     order=exp_order_and_stage["stage_name_lims"])

    ax.legend(fontsize='xx-small',
              title_fontsize='xx-small',
              title=None,
              bbox_to_anchor=(1.01, 1),
              loc=2)
    plt.xlabel("stage name")
    plt.ylabel('roi count')
    plt.title('number segmented rois by experiment for container', pad=5)
    plt.xticks(rotation=90)
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'segmented_rois_by_experiment',
                       get_file_name_for_container(ophys_container_id))


def plot_number_matched_cells_for_container(ophys_container_id, save_figure=True):
    df = processing.container_cell_matching_count_heatmap_df(ophys_container_id)

    matrix = df.values
    labels = df.index

    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(matrix, square=True, cmap='magma', ax=ax, cbar_kws={'shrink': 0.7, 'label': '# matched cells'},
                     annot=True, fmt='.3g')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title('number of matched cells across sessions')
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'number_matched_cells',
                       get_file_name_for_container(ophys_container_id))


def plot_fraction_matched_cells_for_container(ophys_container_id, save_figure=True):
    df = processing.container_cell_matching_percent_heatmap_df(ophys_container_id)

    matrix = df.values
    labels = df.index

    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(matrix, square=True, cmap='magma', ax=ax, cbar_kws={'shrink': 0.7, 'label': '% matched cells'},
                     annot=True, vmin=0, vmax=1, fmt='.3g')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title('fraction matched cells across sessions')
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'fraction_matched_cells',
                       get_file_name_for_container(ophys_container_id))


def plot_cell_matching_registration_overlay_grid(ophys_container_id, save_figure=True):
    """
    Creates a plot of average intensity images for pairs of experiments that are registered during cell matching,
    using the output of the cell matching algorithm directly to obtain images to plot. Plots registered images as
    a red-green overlay, in a grid for all registration pairs. The structural similarity index metric (SSIM) for the pair
    of registered images is shown in the plot title.
    :param ophys_container_id:
    :param save_figure:
    :return:
    """
    import tifffile
    import visual_behavior.data_access.utilities as utilities

    experiments_table = loading.get_filtered_ophys_experiment_table()
    container_expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id]
    ophys_experiment_ids = container_expts.index.values

    dataset_dict = {}
    for ophys_experiment_id in container_expts.index.values:
        dataset = loading.get_ophys_dataset(ophys_experiment_id)
        dataset_dict[ophys_experiment_id] = dataset

    # cell_matching_output_dir = utilities.get_cell_matching_output_dir_for_container(ophys_container_id, experiments_table)
    cell_matching_output_dir = utilities.get_cell_matching_output_dir_for_container(ophys_experiment_id)
    registration_images = [file for file in os.listdir(cell_matching_output_dir) if 'register' in file]

    figsize = (25, 25)
    fig, ax = plt.subplots(len(ophys_experiment_ids), len(ophys_experiment_ids), figsize=figsize)
    ax = ax.ravel()
    i = 0
    for k, exp1 in enumerate(ophys_experiment_ids):
        for j, exp2 in enumerate(ophys_experiment_ids):
            if exp1 == exp2:
                ax[i].axis('off')
                i += 1
            else:
                reg_file = [file for file in registration_images if ((str(exp1) in file) and (str(exp2) in file))]
                if len(reg_file) > 0:
                    registration_image = reg_file[0]
                    # registered_expt = int(registration_image.split('_')[1].split('.')[0])
                    target_expt = int(registration_image.split('_')[3].split('.')[0])
                    dataset = dataset_dict[int(target_expt)]
                    avg_image = dataset.average_projection.data
                    avg_im_target = avg_image / float(np.amax(avg_image))

                    reg = tifffile.imread(os.path.join(cell_matching_output_dir, registration_image))
                    reg = reg / float(np.amax(reg))
                    reg = reg.astype('float32')

                    image = np.empty((reg.shape[0], reg.shape[1], 3))
                    image[:, :, 0] = avg_im_target
                    image[:, :, 1] = reg
                    image[:, :, 2] = np.nan

                    ssim = utilities.get_ssim(avg_im_target, reg)

                    ax[i].imshow(image)
                    ax[i].set_title('exp1: ' + str(exp1) + '\nexp2: ' + str(exp2) + '\nssim:' + str(np.round(ssim, 3)),
                                    fontsize=16)
                    ax[i].axis('off')
                    i += 1
        fig.tight_layout()
        fig.suptitle(get_metadata_string(ophys_container_id), x=0.5, y=1.02, horizontalalignment='center', fontsize=14)
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'cell_matching_registration_overlay_grid',
                       get_file_name_for_container(ophys_container_id))


def plot_cell_matching_registration_output(ophys_container_id, save_figure=True):
    """
        Creates a plot of average intensity images for pairs of experiments that are registered during cell matching,
        using the output of the cell matching algorithm directly to obtain images to plot. Target registered images are
        shown column-wise for each experiment pair with a red-green overlay of the post-registration images in the bottom row.
        Multiple figures are created and saved as .pngs when the number of experiment pairs is large.
        :param ophys_container_id:
        :param save_figure:
        :return:
        """
    import tifffile
    import visual_behavior.data_access.utilities as utilities

    experiments_table = loading.get_filtered_ophys_experiment_table()
    container_expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id]
    ophys_experiment_ids = container_expts.index.values

    dataset_dict = {}
    for ophys_experiment_id in container_expts.index.values:
        dataset = loading.get_ophys_dataset(ophys_experiment_id)
        dataset_dict[ophys_experiment_id] = dataset

    # cell_matching_output_dir = utilities.get_cell_matching_output_dir_for_container(ophys_container_id, experiments_table)
    cell_matching_output_dir = utilities.get_cell_matching_output_dir_for_container(ophys_experiment_id)
    registration_images = [file for file in os.listdir(cell_matching_output_dir) if 'register' in file]

    n_images = len(registration_images)
    n_per_plot = 8
    intervals = np.arange(0, n_images + n_per_plot, n_per_plot)
    for x, interval in enumerate(intervals):
        figsize = (20, 15)
        fig, ax = plt.subplots(4, n_per_plot, figsize=figsize)
        ax = ax.ravel()
        i = 0
        if x < len(intervals) - 1:
            for y, registration_image in enumerate(registration_images[intervals[x]:intervals[x + 1]]):
                registered_expt = registration_image.split('_')[1]
                if int(registered_expt) in ophys_experiment_ids:
                    target_expt = registration_image.split('_')[3].split('.')[0]
                    if int(target_expt) in ophys_experiment_ids:
                        reg = tifffile.imread(os.path.join(cell_matching_output_dir, registration_image))
                        reg = reg / float(np.amax(reg))
                        reg = reg.astype('float32')
                        data = dataset_dict[int(target_expt)]
                        avg_image = data.average_projection.data
                        avg_im_target = avg_image / float(np.amax(avg_image))
                        data = dataset_dict[int(registered_expt)]
                        avg_image = data.average_projection.data
                        avg_im_candidate = avg_image / float(np.amax(avg_image))
                        image = np.empty((reg.shape[0], reg.shape[1], 3))
                        image[:, :, 0] = avg_im_target
                        image[:, :, 1] = reg
                        image[:, :, 2] = np.nan
                        ax[i].imshow(avg_im_candidate, cmap='gray')
                        ax[i].set_title('candidate\n' + registered_expt)
                        ax[i + n_per_plot].imshow(avg_im_target, cmap='gray')
                        ax[i + n_per_plot].set_title('target\n' + target_expt)
                        ax[i + 2 * n_per_plot].imshow(reg, cmap='gray')
                        ax[i + 2 * n_per_plot].set_title('registered\n' + registered_expt)
                        ssim = utilities.get_ssim(avg_im_target, reg)
                        ax[i + 3 * n_per_plot].imshow(image)
                        ax[i + 3 * n_per_plot].set_title('ssim:' + str(np.round(ssim, 3)))
                        for xx in range(len(ax)):
                            ax[xx].axis('off')
                        i += 1

            if save_figure:
                ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'cell_matching_registration_output',
                               get_file_name_for_container(ophys_container_id) + '_' + str(x))


def plot_motion_correction_xy_shift_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 20)
    fig, ax = plt.subplots(len(ophys_experiment_ids), 1, figsize=figsize)
    ax = ax_to_array(ax)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_motion_correction_xy_shift_for_experiment(ophys_experiment_id, ax=ax[i])

        session_type = loading.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'motion_correction_xy_shift',
                       get_file_name_for_container(ophys_container_id))


def plot_flashes_on_trace(ax, timestamps, trial_type=None, omitted=False, alpha=0.15, facecolor='gray'):
    """
    plot stimulus flash durations on the given axis according to the provided timestamps
    """
    stim_duration = 0.2502
    blank_duration = 0.5004
    change_time = 0
    start_time = timestamps[0]
    end_time = timestamps[-1]
    interval = (blank_duration + stim_duration)
    # after time 0
    if omitted:
        array = np.arange((change_time + interval), end_time, interval)
    else:
        array = np.arange(change_time, end_time, interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    if trial_type == 'go':
        alpha = alpha * 3
    else:
        alpha
    # before time 0
    array = np.arange(change_time, start_time - interval, -interval)
    array = array[1:]
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    return ax


def get_metadata_string(ophys_container_id):
    """
    Create a string of metadata information to be used in filenames and figure titles.
    Includes information such as experiment_id, cre_line, acquisition_date, rig_id, etc
    :param ophys_container_id:
    :return:
    """
    ophys_experiment_ids = loading.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)
    dataset = loading.get_ophys_dataset(ophys_experiment_ids[0])
    m = dataset.metadata.copy()
    metadata_string = str(m['mouse_id']) + '_' + str(m['experiment_container_id']) + '_' + m['cre_line'].split('-')[0] + '_' + m['targeted_structure'] + '_' + str(m['imaging_depth']) + '_' + m['session_type']
    return metadata_string


def get_file_name_for_container(ophys_container_id):
    """
    gets standardized filename for saving figures
    format "container_id_"+str(ophys_container_id) is necessary for files to be able to be viewed in Dougs QC viewer
    using get_metadata_string(ophys_container_id) gives a more interpretable filename with cre line, area, etc
    :param ophys_container_id:
    :return:
    """
    # filename =  "container_id_"+str(ophys_container_id)
    filename = get_metadata_string(ophys_container_id)
    return filename


def plot_population_average_across_sessions(container_df, ophys_container_id, df_name, trials=False, omitted=False, save_figure=True):
    """
    Plots population average response across all sessions within a container
    :param container_df: response dataframe for all sessions in a container, can be stimulus_response_df, omission_response_df, etc
    :param ophys_container_id: ID of container being analyzed
    :param df_name: name of response dataframe, ex: 'stimulus_response_df', 'omission_response_df', etc
    :param trials: Boolean, whether or not a trial_response_df is being used, determines how image & change times are shaded in plot
    :param omitted: Boolean, whether or not an omission_response_df is being used, determines how image & change times are shaded in plot
    :param save_figure: Boolean, whether or not to save figure to default QC plots directory
    :return:
    """
    dataset = loading.get_ophys_dataset(container_df.ophys_experiment_id.unique()[0])
    # title = dataset.metadata_string
    title = get_metadata_string(ophys_container_id)
    # frame_rate = dataset.metadata['ophys_frame_rate']
    if omitted:
        figsize = (12, 5)
        m = title.split('_')  # dataset.analysis_folder.split('_')
        title = str(ophys_container_id) + '_' + m[1] + '_' + m[2] + '_' + m[3] + '_' + m[4] + '_' + m[5] + '_' + m[6]
    elif trials:
        figsize = (12, 5)
        container_df = container_df[container_df.go == True]
        m = title.split('_')  # dataset.analysis_folder.split('_')
        title = str(ophys_container_id) + '_' + m[1] + '_' + m[2] + '_' + m[3] + '_' + m[4] + '_' + m[5] + '_' + m[6]
    else:
        figsize = (6, 4)
        container_df = container_df[container_df.image_name != 'omitted']
        title = str(ophys_container_id)
    fig, ax = plt.subplots(figsize=figsize)
    colors = ut.get_colors_for_session_numbers()

    for i, ophys_experiment_id in enumerate(container_df.ophys_experiment_id.unique()):
        df = container_df[container_df.ophys_experiment_id == ophys_experiment_id].copy()
        session_number = df.session_number.unique()[0]
        traces = df.trace.values
        mean_trace = df.trace.mean()
        timestamps = df.trace_timestamps.mean()
        ax.plot(timestamps, mean_trace, color=colors[int(session_number - 1)], label=session_number)
        sem = (traces.std()) / np.sqrt(float(len(traces)))
        ax.fill_between(timestamps, mean_trace + sem, mean_trace - sem, alpha=0.5, color=colors[int(session_number - 1)])
    if omitted:
        ax = plot_flashes_on_trace(ax, timestamps, trial_type=None, omitted=True, alpha=0.2,
                                   facecolor='gray')
        ax.axvline(x=0, ymin=0, ymax=1, linestyle='--', color='gray')
        ax.set_xlabel('time relative to omission (sec)')
    elif trials:
        ax = plot_flashes_on_trace(ax, timestamps, trial_type='go', omitted=False, alpha=0.2,
                                   facecolor='gray')
        ax.set_xlabel('time relative to change (sec)')
    else:
        ax = plot_flashes_on_trace(ax, timestamps, trial_type=None, omitted=False, alpha=0.2,
                                   facecolor='gray')
        ax.set_xlabel('time (sec)')
    ax.set_ylabel('dF/F')
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1, 1), title='session_number', fontsize='small', title_fontsize='x-small')
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(),
                       'population_average_by_session_' + df_name.split('_')[0],
                       get_file_name_for_container(ophys_container_id))


def plot_omission_population_average_across_sessions(ophys_container_id, save_figure=True):
    """
    Plot population average response to omissions, across all cells in a session, for each session type in the container
    :param ophys_container_id:
    :param save_figure:
    :return:
    """
    df_name = 'omission_response_df'
    container_df = loading.get_container_response_df(ophys_container_id, df_name, use_events=False)
    plot_population_average_across_sessions(container_df, ophys_container_id, df_name, trials=False, omitted=True,
                                            save_figure=save_figure)


def plot_trials_population_average_across_sessions(ophys_container_id, save_figure=True):
    """
    Plot population average response to change trials, across all cells in a session, for each session type in the container
    :param ophys_container_id:
    :param save_figure:
    :return:
    """
    df_name = 'trials_response_df'
    container_df = loading.get_container_response_df(ophys_container_id, df_name, use_events=False)
    plot_population_average_across_sessions(container_df, ophys_container_id, df_name, trials=True, omitted=False,
                                            save_figure=save_figure)


def plot_stimulus_population_average_across_sessions(ophys_container_id, save_figure=True):
    """
    Plot population average response across all stimuli, for all cells in a session, for each session type in the container
    :param ophys_container_id:
    :param save_figure:
    :return:
    """
    df_name = 'stimulus_response_df'
    container_df = loading.get_container_response_df(ophys_container_id, df_name, use_events=False)
    plot_population_average_across_sessions(container_df, ophys_container_id, df_name, trials=False, omitted=False,
                                            save_figure=save_figure)


# def plot_average_intensity_by_pmt_for_experiments(ophys_container_id, save_figure=True):
#     """seaborn scatter plot where x = pmt gain, y= fov intensity mean
#         and each point is a passed experiment in the container.
#         The points are colored by their stage name and the legend is stage names
#         displayed in order of acquisition date.

#     Arguments:
#         ophys_container_id {[type]} -- [description]

#     Keyword Arguments:
#         save_figure {bool} -- [description] (default: {True})
#     """
#     container_pmt_settings = processing.container_pmt_settings(ophys_container_id)
#     container_intensity_info = processing.container_intensity_mean_and_std(ophys_container_id)
#     df = pd.merge(container_pmt_settings, container_intensity_info, how="left", on=["ophys_experiment_id", "ophys_container_id"])
#     exp_order_and_stage = processing.experiment_order_and_stage_for_container(ophys_container_id)
#     df = pd.merge(df, exp_order_and_stage, how="left", on="ophys_experiment_id")
#     stage_color_dict = pu.gen_ophys_stage_name_colors_dict()

#     figsize = (9, 5.5)
#     fig, ax = plt.subplots(figsize=figsize)
#     ax = sns.scatterplot(x="pmt_gain", y="intensity_mean", data=df,
#                          hue="stage_name_lims", palette=stage_color_dict)
#     ax.legend(exp_order_and_stage["stage_name_lims"], fontsize='xx-small', title='stage name', title_fontsize='xx-small',
#               bbox_to_anchor=(1.01, 1), loc=2)
#     plt.xlabel('pmt gain')
#     plt.ylabel('FOV intensity mean')
#     plt.title("fov intensity mean by pmt gain")
#     fig.tight_layout()
#     if save_figure:
#         ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'fov_ave_intensity_by_pmt',
#                        get_file_name_for_container(ophys_container_id))


# BEHAVIOR

def plot_running_speed_for_container(ophys_container_id, save_figure=True):
    ophys_session_ids = loading.get_ophys_session_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 15)
    fig, ax = plt.subplots(len(ophys_session_ids), 1, figsize=figsize)
    ax = ax_to_array(ax)
    for i, ophys_session_id in enumerate(ophys_session_ids):
        ax[i] = sp.plot_running_speed(ophys_session_id, ax=ax[i])
        session_type = loading.get_session_type_for_ophys_session_id(ophys_session_id)
        ax[i].set_title(str(ophys_session_id) + '\n' + session_type)
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'running_speed',
                       get_file_name_for_container(ophys_container_id))


def plot_lick_rasters_for_container(ophys_container_id, save_figure=True):
    ophys_session_ids = loading.get_ophys_session_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 7)
    fig, ax = plt.subplots(1, len(ophys_session_ids), figsize=figsize)
    ax = ax_to_array(ax)
    for i, ophys_session_id in enumerate(ophys_session_ids):
        ax[i] = sp.plot_lick_raster(ophys_session_id, ax=ax[i])
        ax[i].invert_yaxis()
        session_type = loading.get_session_type_for_ophys_session_id(ophys_session_id)
        ax[i].set_title(str(ophys_session_id) + '\n' + session_type)
    # plt.gca().invert_yaxis()

    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'lick_rasters',
                       get_file_name_for_container(ophys_container_id))


def plot_pupil_area_sdk(ophys_container_id, save_figure=True):
    table = loading.get_filtered_ophys_experiment_table()
    table = table.reset_index()
    ophys_experiment_ids = table.query('ophys_container_id == {}'.format(ophys_container_id)).sort_values(by='date_of_acquisition')['ophys_experiment_id']

    figsize = (16, 4 * len(ophys_experiment_ids))
    fig = plt.figure(figsize=figsize)
    axes = []
    nplots = len(ophys_experiment_ids)
    buffer = 0.075
    for ii, ophys_experiment_id in enumerate(ophys_experiment_ids):
        print('on ophys_experiment_id {}, #{} of {}'.format(ophys_experiment_id, ii + 1, nplots))
        axes.append(vbp.placeAxesOnGrid(fig, xspan=(0, 1), yspan=(ii / nplots + buffer, (ii + 1) / nplots)))
        if ii + 1 == len(ophys_experiment_ids):
            label_x = True
        else:
            label_x = False
        axes[-1] = ep.make_pupil_area_plot_sdk(ophys_experiment_id, axes[-1], label_x=label_x)

    if save_figure:
        print('saving')
        if save_figure:
            ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'pupil_area_vs_time_sdk', get_file_name_for_container(ophys_container_id))

    return fig, axes


def plot_pupil_area(ophys_container_id, save_figure=True):
    table = loading.get_filtered_ophys_experiment_table()
    table = table.reset_index()
    ophys_experiment_ids = table.query('ophys_container_id == {}'.format(ophys_container_id)).sort_values(by='date_of_acquisition')['ophys_experiment_id']

    figsize = (16, 4 * len(ophys_experiment_ids))
    fig = plt.figure(figsize=figsize)
    axes = []
    nplots = len(ophys_experiment_ids)
    buffer = 0.075
    for ii, ophys_experiment_id in enumerate(ophys_experiment_ids):
        print('on ophys_experiment_id {}, #{} of {}'.format(ophys_experiment_id, ii + 1, nplots))
        axes.append(vbp.placeAxesOnGrid(fig, xspan=(0, 1), yspan=(ii / nplots + buffer, (ii + 1) / nplots)))
        if ii + 1 == len(ophys_experiment_ids):
            label_x = True
        else:
            label_x = False
        axes[-1] = ep.make_pupil_area_plot(ophys_experiment_id, axes[-1], label_x=label_x)

    if save_figure:
        print('saving')
        if save_figure:
            ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'pupil_area_vs_time', get_file_name_for_container(ophys_container_id))

    return fig, axes


def plot_pupil_position(ophys_container_id, save_figure=True):
    table = loading.get_filtered_ophys_experiment_table()
    table = table.reset_index()
    ophys_experiment_ids = table.query('ophys_container_id == {}'.format(ophys_container_id)).sort_values(by='date_of_acquisition')['ophys_experiment_id']

    figsize = (16, 4 * len(ophys_experiment_ids))
    fig = plt.figure(figsize=figsize)
    axes = []
    nplots = len(ophys_experiment_ids)
    buffer = 0.075
    for ii, ophys_experiment_id in enumerate(ophys_experiment_ids):
        print('on ophys_experiment_id {}, #{} of {}'.format(ophys_experiment_id, ii + 1, nplots))
        axes.append(vbp.placeAxesOnGrid(fig, xspan=(0, 1), yspan=(ii / nplots + buffer, (ii + 1) / nplots)))
        if ii + 1 == len(ophys_experiment_ids):
            label_x = True
        else:
            label_x = False
        axes[-1] = ep.make_pupil_position_plot(ophys_experiment_id, axes[-1], label_x=label_x)

    if save_figure:
        if save_figure:
            ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'pupil_position_vs_time', get_file_name_for_container(ophys_container_id))
    return fig, axes


def get_session_stats(behavior_session_uuid):
    vb = db.Database('visual_behavior_data')
    stats = vb.behavior_data['summary'].find_one({'behavior_session_uuid': behavior_session_uuid})
    vb.close()
    return stats


def get_value(behavior_session_uuid, value):
    try:
        session_stats = get_session_stats(behavior_session_uuid)
        return session_stats[value]
    except KeyError:
        return None


def oeid_to_uuid(oeid):
    return db.convert_id({'ophys_experiment_id': oeid}, 'behavior_session_uuid')


def plot_behavior_summary(ophys_container_id, save_figure=True):
    '''
    plots a handful of behavior session metrics for every experiment in the container
    '''
    table = loading.get_filtered_ophys_experiment_table()
    table = table.reset_index()
    ophys_experiment_ids = table.query('ophys_container_id == {}'.format(ophys_container_id)).sort_values(by='date_of_acquisition')['ophys_experiment_id']

    vals_to_plot = {
        'd_prime_peak': [],
        'hit_rate_peak': [],
        'hit_rate': [],
        'false_alarm_rate_peak': [],
        'false_alarm_rate': [],
        'fraction_time_aborted': [],
        'earned_water': [],
        'total_water': [],
        'number_of_licks': [],
        'num_contingent_trials': [],
    }

    figsize = (16, 5)
    fig, ax = plt.subplots(1, 7, figsize=figsize, sharey=True)
    y_labels = []

    for ii, oeid in enumerate(ophys_experiment_ids):
        uuid = oeid_to_uuid(oeid)
        session_type = table.query('ophys_container_id == {} and ophys_experiment_id == {}'.format(
            ophys_container_id, oeid))['session_type'].iloc[0]
        y_labels.append('expt_id = {}\n{}'.format(oeid, session_type))
        for key in vals_to_plot.keys():
            try:
                vals_to_plot[key].append(get_value(uuid, key))
            except TypeError:
                vals_to_plot[key].append(np.nan)

    ax[0].set_ylim(-0.5, len(ophys_experiment_ids) - 0.5)
    ax[0].invert_yaxis()
    ax[0].plot(vals_to_plot['d_prime_peak'], range(len(ophys_experiment_ids)), color='black', marker='o')
    ax[0].set_yticks(range(len(ophys_experiment_ids)))
    ax[0].set_yticklabels(y_labels, fontsize=11)
    ax[0].set_xlabel("d' peak", fontsize=14)
    ax[0].set_title("d' peak")

    ax[1].plot(vals_to_plot['hit_rate_peak'], range(len(ophys_experiment_ids)), color='darkgreen', marker='o')
    ax[1].plot(vals_to_plot['hit_rate'], range(len(ophys_experiment_ids)), color='lightgreen', marker='o')
    ax[1].set_title("Hit rate\nPeak (dark)\nAvg (light)")
    ax[1].set_xlabel('resp. prob.', fontsize=14)
    ax[1].set_xlim(-0.05, 1.05)

    ax[2].plot(vals_to_plot['false_alarm_rate_peak'], range(len(ophys_experiment_ids)), color='darkorange', marker='o')
    ax[2].plot(vals_to_plot['false_alarm_rate'], range(len(ophys_experiment_ids)), color='yellow', marker='o')
    ax[2].set_title("FA rate\nPeak (dark)\nAvg (light)")
    ax[2].set_xlabel('resp. prob.', fontsize=14)
    ax[2].set_xlim(-0.05, 1.05)

    ax[3].plot(vals_to_plot['fraction_time_aborted'], range(len(ophys_experiment_ids)), color='darkred', marker='o')
    ax[3].set_title("Fraction\ntime\naborted")
    ax[3].set_xlabel('fraction time', fontsize=14)
    ax[3].set_xlim(-0.05, 1.05)

    ax[4].plot(vals_to_plot['earned_water'], range(len(ophys_experiment_ids)), color='darkblue', marker='o')
    ax[4].plot(vals_to_plot['total_water'], range(len(ophys_experiment_ids)), color='dodgerblue', marker='o')
    ax[4].set_title("Earned $H_2$$O$ (dark)\nTotal $H_2$$O$ (light)")
    ax[4].set_xlabel('$H_2$$O$ (mL)', fontsize=14)

    ax[5].plot(vals_to_plot['number_of_licks'], range(len(ophys_experiment_ids)), color='teal', marker='o')
    ax[5].set_title("Number\nof Licks")
    ax[5].set_xlabel('count', fontsize=14)

    ax[6].plot(vals_to_plot['num_contingent_trials'], range(len(ophys_experiment_ids)), color='Maroon', marker='o')
    ax[6].set_title("Number\nof Trials")
    ax[6].set_xlabel('count', fontsize=14)
    ax[6].set_xlim(0, 450)

    colors = ['DarkSlateGray', 'lightgray']
    for row in range(len(ophys_experiment_ids)):
        color = colors[row % 2]
        for col in range(len(ax)):
            ax[col].axhspan(row - 0.5, row + 0.5, color=color, alpha=0.5)

    if save_figure:
        if save_figure:
            ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'behavior_metric_summary',
                           get_file_name_for_container(ophys_container_id))


def plot_event_detection_for_container(ophys_container_id, save_figure=True):
    """
    Generates plots of dFF traces and events for each cell in each experiment of a container.
    Useful to validate whether detected events line up with dFF transients.
    :param ophys_container_id:
    :param save_figure:
    :return:
    """
    experiments_table = loading.get_filtered_ophys_experiment_table()
    ophys_experiments = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(by='date_of_acquisition')
    ophys_experiment_ids = ophys_experiments.index.values

    for ophys_experiment_id in ophys_experiment_ids:
        ep.plot_event_detection_for_experiment(ophys_experiment_id, save_figure=save_figure)


def plot_single_cell_response_plots_for_container(ophys_container_id, save_figure=True):
    """
    Generates plots characterizing single cell activity in response to stimulus, omissions, and changes.
    Compares across all sessions in a container for each cell, including the ROI mask across days.
    Useful to validate cell matching as well as examine changes in activity profiles over days.
    """
    cell_specimen_ids = loading.get_unique_cell_specimen_ids_for_container(ophys_container_id)

    for cell_specimen_id in cell_specimen_ids:
        scp.plot_across_session_responses(ophys_container_id, cell_specimen_id, use_events=False, save_figure=save_figure)


def plot_dff_trace_and_behavior_for_container(ophys_container_id, save_figure=True):
    """
    Plots the full dFF trace for each cell, along with licking behavior, rewards, running speed, pupil area, and face motion.
    Useful to visualize whether the dFF trace tracks the behavior variables
    """
    experiments_table = loading.get_filtered_ophys_experiment_table()
    ophys_experiments = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(by='date_of_acquisition')
    ophys_experiment_ids = ophys_experiments.index.values

    for ophys_experiment_id in ophys_experiment_ids:
        ep.plot_population_activity_and_behavior_for_experiment(ophys_experiment_id, save_figure=save_figure)
        ep.plot_dff_trace_and_behavior_for_experiment(ophys_experiment_id, save_figure=save_figure)


def plot_classifier_validation_for_container(ophys_container_id, save_figure=True):
    """
    Creates a plot showing ROI masks matched between production and development versions of segmentation classifier.
    This is a one off validation figure and will be removed in future versions of the code
    :param ophys_container_id:
    :param save_figure:
    :return:
    """
    experiments_table = loading.get_filtered_ophys_experiment_table()
    ophys_experiments = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(by='date_of_acquisition')
    ophys_experiment_ids = ophys_experiments.index.values

    for ophys_experiment_id in ophys_experiment_ids:
        ep.plot_classifier_validation_for_experiment(ophys_experiment_id, save_figure=save_figure)


def motion_correction_artifacts(experiment_id: int) -> pd.DataFrame:
    result = lims_query(
        f"""
    SELECT wkft.name, wkf.storage_directory || wkf.filename as path
    FROM well_known_files AS wkf
    JOIN well_known_file_types AS wkft on wkft.id=wkf.well_known_file_type_id
    JOIN ophys_experiments AS oe on wkf.attachable_id=oe.id
    WHERE wkft.name IN (
        'OphysMotionPreview',
        'OphysMotionXyOffsetData',
        'MotionCorrectedImageStack',
        'OphysRegistrationSummaryImage')
    AND oe.id={experiment_id}""")
    for i in range(result.shape[0]):
        result.loc[i, 'path'] = system_friendly_filename(
            result.loc[i, 'path'])
    return result


def system_friendly_filename(fname: str) -> str:
    if platform.system() == "Windows":
        return "\\" + str(PureWindowsPath(fname))
    else:
        return fname


def plot_OphysRegistrationSummaryImage(ophys_container_id, save_figure=True):
    experiments_table = loading.get_filtered_ophys_experiment_table()
    ophys_experiments = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(by='date_of_acquisition')
    oeids = ophys_experiments.index.values

    figsize = (15, 10 * len(oeids))
    fig, ax = plt.subplots(len(oeids), 1, figsize=figsize)
    for ii, oeid in enumerate(np.sort(oeids)):
        image_path = motion_correction_artifacts(oeid).set_index('name').loc['OphysRegistrationSummaryImage']['path']
        image = imageio.imread(image_path)
        ax.flatten()[ii].imshow(image)
        ax.flatten()[ii].axis('off')
        ax.flatten()[ii].set_title('ophys_experiment_id = {}'.format(oeid))
    try:
        # remove the last axis unless there is an index error
        ax.flatten()[ii + 1].axis('off')
    except IndexError:
        pass
    fig.tight_layout()
    if save_figure:
        if save_figure:
            ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'OphysRegistrationSummaryImage',
                           get_file_name_for_container(ophys_container_id))


def plot_nway_match_fraction(ophys_container_id, save_figure=True):
    experiments_table = loading.get_filtered_ophys_experiment_table()
    ophys_experiments = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(by='date_of_acquisition')
    ophys_experiment_id = ophys_experiments.index.values[0]
    cell_matching_output_dir = utilities.get_cell_matching_output_dir_for_container(ophys_experiment_id)
    file_to_plot = [file for file in os.listdir(cell_matching_output_dir) if 'nway_match_fraction' in file]

    figsize = (20, 20)
    fig, ax = plt.subplots(figsize=figsize)
    image_path = os.path.join(cell_matching_output_dir, file_to_plot[0])
    image = imageio.imread(image_path)
    ax.imshow(image)
    ax.axis('off')

    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'nway_match_fraction', get_file_name_for_container(ophys_container_id))


def plot_nway_warp_overlay(ophys_container_id, save_figure=True):
    experiments_table = loading.get_filtered_ophys_experiment_table()
    ophys_experiments = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(by='date_of_acquisition')
    ophys_experiment_id = ophys_experiments.index.values[0]
    cell_matching_output_dir = utilities.get_cell_matching_output_dir_for_container(ophys_experiment_id)
    file_to_plot = [file for file in os.listdir(cell_matching_output_dir) if 'nway_warp_overlay_plot' in file]

    figsize = (20, 20)
    fig, ax = plt.subplots(figsize=figsize)
    image_path = os.path.join(cell_matching_output_dir, file_to_plot[0])
    image = imageio.imread(image_path)
    ax.imshow(image)
    ax.axis('off')

    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'nway_warp_overlay', get_file_name_for_container(ophys_container_id))


def plot_nway_warp_summary(ophys_container_id, save_figure=True):
    experiments_table = loading.get_filtered_ophys_experiment_table()
    ophys_experiments = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(by='date_of_acquisition')
    ophys_experiment_id = ophys_experiments.index.values[0]
    cell_matching_output_dir = utilities.get_cell_matching_output_dir_for_container(ophys_experiment_id)
    file_to_plot = [file for file in os.listdir(cell_matching_output_dir) if 'nway_warp_summary_plot' in file]

    figsize = (25, 15)
    fig, ax = plt.subplots(figsize=figsize)
    image_path = os.path.join(cell_matching_output_dir, file_to_plot[0])
    image = imageio.imread(image_path)
    ax.imshow(image)
    ax.axis('off')

    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, loading.get_container_plots_dir(), 'nway_warp_summary', get_file_name_for_container(ophys_container_id))


def plot_experiment_summary_figure_for_container(ophys_container_id, save_figure=True):
    import visual_behavior.visualization.qc.experiment_summary as es
    experiments_table = loading.get_filtered_ophys_experiment_table()
    ophys_experiments = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(
        by='date_of_acquisition')
    for ophys_experiment_id in ophys_experiments.index.values:
        # try:
        es.plot_experiment_summary_figure(ophys_experiment_id, save_figure=save_figure)
        # except:
        #     print('could not plot experiment summary for', ophys_experiment_id)


def generate_snr_metrics_df_for_container(ophys_container_id):
    import visual_behavior.visualization.qc.compute_snr_metrics as snr_metrics
    experiments_table = loading.get_filtered_ophys_experiment_table()
    ophys_experiments = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(
        by='date_of_acquisition')
    metrics_df, problems_list = snr_metrics.get_snr_metrics_df_for_experiments(ophys_experiments.index.values)
    save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/snr_metrics'
    # save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\qc_plots\snr_metrics'
    metrics_df.to_csv(os.path.join(save_dir, str(ophys_container_id) + '_snr_metrics.csv'))
    problems_list = pd.DataFrame(problems_list)
    problems_list.to_csv(os.path.join(save_dir, str(ophys_container_id) + '_problem_expts.csv'))
