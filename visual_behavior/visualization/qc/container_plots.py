import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import visual_behavior.plotting as vbp
import visual_behavior.database as db

from visual_behavior.visualization import utils as ut
from visual_behavior.visualization.qc import data_loading as dl
from visual_behavior.visualization.qc import session_plots as sp
from visual_behavior.visualization.qc import plotting_utils as pu
from visual_behavior.visualization.qc import data_processing as dp
from visual_behavior.visualization.qc import experiment_plots as ep


# Container sequence

def plot_container_session_sequence(ophys_container_id, save_figure=True):
    experiments_table = dl.get_filtered_ophys_experiment_table(include_failed_data=True)
    expts = experiments_table[experiments_table.container_id == ophys_container_id].copy()
    specimen_id = expts.specimen_id.unique()[0]
    experiment_ids = expts.ophys_experiment_id.unique()
    session_type_color_map = ut.get_session_type_color_map()

    n_expts = len(expts)
    img = np.empty((n_expts, 1, 3))
    fail_x = []
    fail_tags = []
    for expt_ind, expt_id in enumerate(experiment_ids):
        this_expt = expts[expts.ophys_experiment_id == expt_id]
        img[expt_ind, 0, :] = session_type_color_map[this_expt['session_type'].values[0]]
        if this_expt['experiment_workflow_state'].values[0] == 'failed':
            fail_x.append(expt_ind)
            fail_tags.append(this_expt['failure_tags'].values[0])

    # create plot with expt colors image
    figsize = (20, n_expts)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img.astype(int))
    ax.axis('off')

    # plot text for acquisition date and session type
    for i, expt_id in enumerate(experiment_ids):
        this_expt = expts[expts.ophys_experiment_id == expt_id]
        ax.text(x=0.75, y=i, s=str(this_expt['date_of_acquisition'].values[0]).split(' ')[0],
                ha='left', va='center', fontsize=20)
        ax.text(x=3, y=i, s=this_expt['session_type'].values[0], ha='left', va='center', fontsize=20)
        ax.text(x=20, y=i, s=' ')

    # add X for fails and list fail tags
    for ind_fail, x in enumerate(fail_x):
        ax.text(x=0, y=x, s='X', ha='center', va='center', fontsize=60)
        fail_string = 'Failure: ' + str(fail_tags[ind_fail])
        ax.text(x=8.5, y=x, s=fail_string, ha='left', va='center', fontsize=20)

    plt.suptitle('specimen_id: {}'.format(specimen_id) + ', container_id: {}'.format(ophys_container_id),
                 fontsize=25, ha='left', x=0.06, y=.97)
    fig.subplots_adjust(left=0.05)
    fig.subplots_adjust(right=0.1)
    fig.subplots_adjust(top=0.9)
    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'ophys_session_sequence',
                       'container_' + str(ophys_container_id))


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
    # exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
    # ophys_experiment_ids = list(exp_order_and_stage["ophys_experiment_id"])
    ophys_experiment_ids = dl.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = dl.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        # exp_stage_name = exp_order_and_stage.loc[exp_order_and_stage["ophys_experiment_id"]== ophys_experiment_id, "stage_name_lims"].reset_index(drop=True)[0]
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'max_intensity_projection',
                       'container_' + str(ophys_container_id))


def plot_movie_max_projection_images_for_container(ophys_container_id, save_figure=True):
    """for every passed experiment in an experiment container, gets plots
        the max intensity of the motion corrected movie (unnormalized) next to one another
        in order of acquisition date

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    # exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
    # ophys_experiment_ids = list(exp_order_and_stage["ophys_experiment_id"])
    ophys_experiment_ids = dl.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_motion_correction_max_image_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = dl.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        # exp_stage_name = exp_order_and_stage.loc[exp_order_and_stage["ophys_experiment_id"]== ophys_experiment_id, "stage_name_lims"].reset_index(drop=True)[0]
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'max_intensity_projection_movies',
                       'container_' + str(ophys_container_id))


def plot_sdk_average_images_for_container(ophys_container_id, save_figure=True):
    """for every passed experiment in an experiment container, gets plots
        the average intensity projections from the sdk (normalized) next to one another
        in order of acquisition date

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    # exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
    # ophys_experiment_ids = list(exp_order_and_stage["ophys_experiment_id"])
    ophys_experiment_ids = dl.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_average_image_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = dl.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        # exp_stage_name = exp_order_and_stage.loc[exp_order_and_stage["ophys_experiment_id"]== ophys_experiment_id, "stage_name_lims"].reset_index(drop=True)[0]
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'average_images',
                       'container_' + str(ophys_container_id))


def plot_movie_average_images_for_container(ophys_container_id, save_figure=True):
    """for every passed experiment in an experiment container, gets plots
        the average intensity of the motion corrected movie (unnormalized) next to one another
        in order of acquisition date

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    # exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
    # ophys_experiment_ids = list(exp_order_and_stage["ophys_experiment_id"])
    ophys_experiment_ids = dl.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_motion_correction_average_image_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = dl.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        # exp_stage_name = exp_order_and_stage.loc[exp_order_and_stage["ophys_experiment_id"]== ophys_experiment_id, "stage_name_lims"].reset_index(drop=True)[0]
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'average_images_movies',
                       'container_' + str(ophys_container_id))


def plot_eye_tracking_sample_frames(ophys_container_id, save_figure=True):
    table = dl.get_filtered_ophys_experiment_table()
    ophys_experiment_ids = table.query('container_id == {}'.format(ophys_container_id)).sort_values(by='date_of_acquisition')['ophys_experiment_id']

    fig = plt.figure(figsize=(16, 5 * len(ophys_experiment_ids)))
    axes = []
    nplots = len(ophys_experiment_ids)
    buffer = 0.05
    for ii, ophys_experiment_id in enumerate(ophys_experiment_ids):
        print('on ophys_experiment_id {}, #{} of {}'.format(ophys_experiment_id, ii + 1, nplots))
        axes.append(vbp.placeAxesOnGrid(fig, dim=(3, 10), xspan=(0, 1), yspan=(ii / nplots + buffer, (ii + 1) / nplots)))
        axes[-1] = ep.make_eye_matrix_plot(ophys_experiment_id, axes[-1])

    savepath = os.path.join(dl.get_container_plots_dir(), 'eyetracking_sample_frames', 'container_{}.png'.format(ophys_container_id))
    fig.savefig(savepath, dpi=300, pad_inches=0.0, bbox_inches='tight')

    return fig, axes


def plot_segmentation_masks_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = dl.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_segmentation_mask_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = dl.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'segmentation_masks',
                       'container_' + str(ophys_container_id))


def plot_segmentation_mask_overlays_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = dl.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, len(ophys_experiment_ids), figsize=figsize)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_segmentation_mask_overlay_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = dl.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'segmentation_mask_overlays',
                       'container_' + str(ophys_container_id))


def plot_dff_traces_heatmaps_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = dl.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 20)
    fig, ax = plt.subplots(len(ophys_experiment_ids), 1, figsize=figsize)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_traces_heatmap_for_experiment(ophys_experiment_id, ax=ax[i])
        session_type = dl.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + ' - ' + session_type)

    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'dff_traces_heatmaps',
                       'container_' + str(ophys_container_id))


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
    container_df = (dp.passed_experiment_info_for_container(ophys_container_id)).sort_values('date_of_acquisition').reset_index(drop=True)
    exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
    figsize = (9, 5)
    fig, ax = plt.subplots(figsize=figsize)
    for i, ophys_experiment_id in enumerate(container_df["ophys_experiment_id"].unique()):
        ax = ep.plot_average_intensity_timeseries_for_experiment(ophys_experiment_id, ax=ax)
    ax.legend(exp_order_and_stage["stage_name_lims"], fontsize='xx-small', title='stage name', title_fontsize='xx-small',
              bbox_to_anchor=(1.01, 1), loc=2)
    ax.set_title('full field average fluorescence intensity over time')
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'average_intensity_timeseries',
                       'container_' + str(ophys_container_id))


def plot_pmt_for_container(ophys_container_id, save_figure=True):
    """seaborn scatter plot where x= session stage name
        y= pmt setting for that session

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    pmt_settings = dp.container_pmt_settings(ophys_container_id)
    exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
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
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'pmt_settings',
                       'container_' + str(ophys_container_id))


def plot_average_intensity_for_container(ophys_container_id, save_figure=True):
    """seaborn scatter plot where x= session stage name
        y= average intensity of the FOV for the entire session

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        save_figure {bool} -- [description] (default: {True})
    """
    FOV_intensities = dp.container_intensity_mean_and_std(ophys_container_id)
    exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
    df = pd.merge(FOV_intensities, exp_order_and_stage, how="left", on="ophys_experiment_id")

    stage_color_dict = pu.gen_ophys_stage_name_colors_dict()
    figsize = (6, 9)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(x="stage_name_lims", y="intensity_mean", data=df,
                         hue="stage_name_lims", palette=stage_color_dict,
                         legend=False)
    plt.xticks(rotation=90)
    plt.ylabel('FOV average intensity')
    plt.title("FOV mean intensity for container")
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'FOV_average_intensity',
                       'container_' + str(ophys_container_id))


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
    pmt_settings = dp.container_pmt_settings(ophys_container_id)
    FOV_intensities = dp.container_intensity_mean_and_std(ophys_container_id)
    exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
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
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'average_intensity_by_pmt',
                       'container_' + str(ophys_container_id))


def plot_snr_by_pmt_gain_and_intensity_for_container(ophys_container_id, save_figure=True):
    df = dp.container_FOV_information(ophys_container_id)
    figsize = (7, 5.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax = plt.scatter(df["pmt_gain"], df["median_rsnr_all_csids"],
                     c=df["intensity_mean"], s=75,
                     cmap="cool", edgecolors='k')
    ax.set_xlim(df["pmt_gain"].min() - 1, df["pmt_gain"].max() + 1)
    cbar = plt.colorbar(ax)
    cbar.set_label('fov mean intensity', rotation=270, labelpad=25)
    plt.xlabel('pmt gain')
    plt.ylabel('median snr across cells')
    plt.suptitle("median robust snr across cells by pmt gain")
    plt.title("container: " + str(ophys_container_id))
    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'snr_by_pmt_and_intensity',
                       'container_' + str(ophys_container_id))


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
    pmt_settings = dp.container_pmt_settings(ophys_container_id)
    snr_summary = dp.container_snr_summary_table(ophys_container_id)
    exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
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
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'snr_by_pmt',
                       'container_' + str(ophys_container_id))


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
    exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
    container_snr_table = dp.container_csid_snr_table(ophys_container_id)
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
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'cell_snr_by_experiment',
                       'container_' + str(ophys_container_id))


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
    seg_count_df = dp.segmentation_validity_count_for_container(ophys_container_id)
    exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
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
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'segmented_rois_by_experiment',
                       'container_' + str(ophys_container_id))


def plot_number_matched_cells_for_container(ophys_container_id, save_figure=True):
    df = dp.container_cell_matching_count_heatmap_df(ophys_container_id)

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
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'number_matched_cells',
                       'container_' + str(ophys_container_id))


def plot_fraction_matched_cells_for_container(ophys_container_id, save_figure=True):
    df = dp.container_cell_matching_percent_heatmap_df(ophys_container_id)

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
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'fraction_matched_cells',
                       'container_' + str(ophys_container_id))


def plot_motion_correction_xy_shift_for_container(ophys_container_id, save_figure=True):
    ophys_experiment_ids = dl.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 20)
    fig, ax = plt.subplots(len(ophys_experiment_ids), 1, figsize=figsize)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_motion_correction_xy_shift_for_experiment(ophys_experiment_id, ax=ax[i])

        session_type = dl.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'motion_correction_xy_shift',
                       'container_' + str(ophys_container_id))


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
#     container_pmt_settings = dp.container_pmt_settings(ophys_container_id)
#     container_intensity_info = dp.container_intensity_mean_and_std(ophys_container_id)
#     df = pd.merge(container_pmt_settings, container_intensity_info, how="left", on=["ophys_experiment_id", "ophys_container_id"])
#     exp_order_and_stage = dp.experiment_order_and_stage_for_container(ophys_container_id)
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
#         ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'fov_ave_intensity_by_pmt',
#                        'container_' + str(ophys_container_id))


# BEHAVIOR

def plot_running_speed_for_container(ophys_container_id, save_figure=True):
    ophys_session_ids = dl.get_ophys_session_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 15)
    fig, ax = plt.subplots(len(ophys_session_ids), 1, figsize=figsize)
    for i, ophys_session_id in enumerate(ophys_session_ids):
        try:
            ax[i] = sp.plot_running_speed(ophys_session_id, ax=ax[i])
        except:
            pass
        session_type = dl.get_session_type_for_ophys_session_id(ophys_session_id)
        ax[i].set_title(str(ophys_session_id) + '\n' + session_type)
    fig.tight_layout()
    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'running_speed',
                       'container_' + str(ophys_container_id))


def plot_lick_rasters_for_container(ophys_container_id, save_figure=True):
    ophys_session_ids = dl.get_ophys_session_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 7)
    fig, ax = plt.subplots(1, len(ophys_session_ids), figsize=figsize)
    for i, ophys_session_id in enumerate(ophys_session_ids):
        ax[i] = sp.plot_lick_raster(ophys_session_id, ax=ax[i])
        ax[i].invert_yaxis()
        session_type = dl.get_session_type_for_ophys_session_id(ophys_session_id)
        ax[i].set_title(str(ophys_session_id) + '\n' + session_type)
    # plt.gca().invert_yaxis()

    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'lick_rasters',
                       'container_' + str(ophys_container_id))


def plot_pupil_area(ophys_container_id, save_figure=True):
    table = dl.get_filtered_ophys_experiment_table()
    ophys_experiment_ids = table.query('container_id == {}'.format(ophys_container_id)).sort_values(by='date_of_acquisition')['ophys_experiment_id']

    fig = plt.figure(figsize=(16, 4 * len(ophys_experiment_ids)))
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
        save_folder = os.path.join(dl.get_container_plots_dir(), 'pupil_area_vs_time')
        if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
        savepath = os.path.join(save_folder, 'container_{}.png'.format(ophys_container_id))
        fig.savefig(savepath, dpi=300, pad_inches=0.0, bbox_inches='tight')

    return fig, axes


def plot_pupil_position(ophys_container_id, save_figure=True):
    table = dl.get_filtered_ophys_experiment_table()
    ophys_experiment_ids = table.query('container_id == {}'.format(ophys_container_id)).sort_values(by='date_of_acquisition')['ophys_experiment_id']

    fig = plt.figure(figsize=(16, 4 * len(ophys_experiment_ids)))
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
        print('saving')
        save_folder = os.path.join(dl.get_container_plots_dir(), 'pupil_position_vs_time')
        if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
        savepath = os.path.join(save_folder, 'container_{}.png'.format(ophys_container_id))
        fig.savefig(savepath, dpi=300, pad_inches=0.0, bbox_inches='tight')

    return fig, axes

def get_session_stats(behavior_session_uuid):
    vb = db.Database('visual_behavior_data')
    stats = vb.behavior_data['summary'].find_one({'behavior_session_uuid':behavior_session_uuid})
    vb.close()
    return stats

def get_value(behavior_session_uuid, value):
    session_stats = get_session_stats(behavior_session_uuid)
    return session_stats[value]

def oeid_to_uuid(oeid):
    return db.convert_id({'ophys_experiment_id':oeid},'behavior_session_uuid')

def plot_behavior_summary(ophys_container_id, save_figure=True):
    '''
    plots a handful of behavior session metrics for every experiment in the container
    '''
    table = dl.get_filtered_ophys_experiment_table()
    ophys_experiment_ids = table.query('container_id == {}'.format(ophys_container_id)).sort_values(by='date_of_acquisition')['ophys_experiment_id']

    vals_to_plot = {
        'd_prime_peak':[],
        'hit_rate_peak':[],
        'hit_rate':[],
        'false_alarm_rate_peak':[],
        'false_alarm_rate':[],
        'fraction_time_aborted':[],
        'earned_water':[],
        'total_water':[],
        'number_of_licks':[],
        'num_contingent_trials':[],
    }

    fig,ax=plt.subplots(1,7,figsize=(16,5),sharey=True)
    y_labels = []

    for ii,oeid in enumerate(ophys_experiment_ids):
        uuid = oeid_to_uuid(oeid)
        y_labels.append('expt_id = {}\n{}'.format(oeid,get_value(uuid, 'stage')))
        for key in vals_to_plot.keys():
            vals_to_plot[key].append(get_value(uuid, key))


    ax[0].set_ylim(-0.5,len(ophys_experiment_ids)-0.5)
    ax[0].invert_yaxis()
    ax[0].plot(vals_to_plot['d_prime_peak'],range(len(ophys_experiment_ids)),color='black',marker='o')
    ax[0].set_yticks(range(len(ophys_experiment_ids)))
    ax[0].set_yticklabels(y_labels);
    ax[0].set_xlabel("d' peak")
    ax[0].set_title("d' peak")

    ax[1].plot(vals_to_plot['hit_rate_peak'],range(len(ophys_experiment_ids)),color='darkgreen',marker='o')
    ax[1].plot(vals_to_plot['hit_rate'],range(len(ophys_experiment_ids)),color='lightgreen',marker='o')
    ax[1].set_title("Hit rate\nPeak (dark)\nAvg (light)")
    ax[1].set_xlabel('resp. prob.')
    ax[1].set_xlim(0,1.05)

    ax[2].plot(vals_to_plot['false_alarm_rate_peak'],range(len(ophys_experiment_ids)),color='darkorange',marker='o')
    ax[2].plot(vals_to_plot['false_alarm_rate'],range(len(ophys_experiment_ids)),color='yellow',marker='o')
    ax[2].set_title("FA rate\nPeak (dark)\nAvg (light)")
    ax[2].set_xlabel('resp. prob.')
    ax[2].set_xlim(0,1.05)

    ax[3].plot(vals_to_plot['fraction_time_aborted'],range(len(ophys_experiment_ids)),color='darkred',marker='o')
    ax[3].set_title("Fraction\ntime\naborted")
    ax[3].set_xlabel('fraction time')
    ax[3].set_xlim(0,1.05)

    ax[4].plot(vals_to_plot['earned_water'],range(len(ophys_experiment_ids)),color='darkblue',marker='o')
    ax[4].plot(vals_to_plot['total_water'],range(len(ophys_experiment_ids)),color='dodgerblue',marker='o')
    ax[4].set_title("Earned $H_2$$O$ (dark)\nTotal $H_2$$O$ (light)")
    ax[4].set_xlabel('$H_2$$O$ (mL)')

    ax[5].plot(vals_to_plot['number_of_licks'],range(len(ophys_experiment_ids)),color='teal',marker='o')
    ax[5].set_title("Number\nof Licks")
    ax[5].set_xlabel('count')

    ax[6].plot(vals_to_plot['num_contingent_trials'],range(len(ophys_experiment_ids)),color='Maroon',marker='o')
    ax[6].set_title("Number\nof Trials")
    ax[6].set_xlabel('count')
    ax[6].set_xlim(0,450)

    colors = ['DarkSlateGray','lightgray']
    for row in range(len(ophys_experiment_ids)):
        color = colors[row%2]
        for col in range(len(ax)):
            ax[col].axhspan(row-0.5,row+0.5,color=color,alpha=0.5)

    if save_figure:
        print('saving')
        save_folder = os.path.join(dl.get_container_plots_dir(), 'behavior_metric_summary')
        if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
        savepath = os.path.join(save_folder, 'container_{}.png'.format(ophys_container_id))
        fig.savefig(savepath, dpi=300, pad_inches=0.0, bbox_inches='tight')