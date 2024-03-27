#!/usr/bin/env python

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

import visual_behavior.data_access.utilities as utilities
import visual_behavior.data_access.loading as loading
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.visualization.utils as utils
import visual_behavior.visualization.ophys.summary_figures as sf


def plot_metrics_mask(roi_mask_dict, max_projection=None, metric_values=None, title=None,
                      cmap='RdBu', cmap_range=[0, 1], ax=None, colorbar=False):
    if ax is None:
        figsize = (6, 6)
        fig, ax = plt.subplots(figsize=figsize)
    if max_projection is not None:
        ax.imshow(max_projection, cmap='gray', vmin=0, vmax=np.percentile(max_projection, 99))
    if metric_values is None:
        metric_values = np.ones(len(roi_mask_dict))
    for i, roi_id in enumerate(list(roi_mask_dict.keys())):
        tmp = roi_mask_dict[roi_id]
        mask = np.empty(tmp.shape, dtype=np.float)
        mask[:] = np.nan
        mask[tmp == 1] = metric_values[i]
        cax = ax.imshow(mask, cmap=cmap, alpha=0.5, vmin=cmap_range[0], vmax=cmap_range[1])
        ax.set_title(title)
        ax.grid(False)
        ax.axis('off')
    if colorbar:
        plt.colorbar(cax, ax=ax)
    return ax


def get_metrics_df(experiment_id):
    metrics_df = loading.load_current_objectlisttxt_file(experiment_id)
    # ROI locations from lims, including cell_roi_id
    roi_loc = loading.roi_locations_from_cell_rois_table(experiment_id)
    # limit to current segmentation run, otherwise gives old ROIs
    run_id = loading.get_current_segmentation_run_id(experiment_id)
    roi_loc = roi_loc[roi_loc.ophys_cell_segmentation_run_id == run_id]
    # link ROI metrics with cell_roi_id from ROI locations dict using ROI location
    metrics_df = metrics_df.merge(roi_loc, on=['bbox_min_x', 'bbox_min_y'])
    return metrics_df


def plot_metrics_distribution(metrics_df, title, folder):
    metrics = ['area', 'ellipseness', 'compactness', 'mean_intensity', 'max_intensity', 'intensity_ratio',
               'soma_minus_np_mean', 'soma_minus_np_std', 'sig_active_frames_2_5', 'sig_active_frames_4']
    figsize = (20, 8)
    fig, ax = plt.subplots(2, 5, figsize=figsize)
    ax = ax.ravel()

    for i, metric in enumerate(metrics):
        ax[i] = sns.distplot(metrics_df[metrics_df.valid_roi == True][metric].values, bins=30, ax=ax[i],
                             color='blue')
        ax[i] = sns.distplot(metrics_df[metrics_df.valid_roi == False][metric].values, bins=30, ax=ax[i],
                             color='red')
        ax[i].set_xlabel(metric)
        ax[i].set_ylabel('density')
        ax[i].legend(['valid', 'invalid'], fontsize='x-small', loc='upper right')

    fig.tight_layout()
    fig.suptitle(title, x=0.5, y=1.01, fontsize=16)
    save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/roi_filtering_validation'
    utils.save_figure(fig, figsize, save_dir, folder, title + '_metric_distributions')


def plot_metric_range_dataset(dataset, cell_specimen_table, max_projection, metrics_df, metric, thresholds,
                              title, less_than=False):
    ct = cell_specimen_table.copy()

    figsize = (20, 10)
    fig, ax = plt.subplots(2, 4, figsize=figsize)
    ax = ax.ravel()

    all_roi_mask_dict = {}
    for cell_roi_id in ct.cell_roi_id.values:
        all_roi_mask_dict[cell_roi_id] = ct[ct.cell_roi_id == cell_roi_id].roi_mask.values[0]
    ax[0] = plot_metrics_mask(all_roi_mask_dict, max_projection, metric_values=None, title='all ROIs',
                              cmap='hsv', cmap_range=[0, 1], ax=ax[0], colorbar=False)

    valid_roi_mask_dict = {}
    for cell_roi_id in ct[ct.valid_roi == True].cell_roi_id.values:
        valid_roi_mask_dict[cell_roi_id] = ct[ct.cell_roi_id == cell_roi_id].roi_mask.values[0]
    ax[1] = plot_metrics_mask(valid_roi_mask_dict, max_projection, metric_values=None, title='valid ROIs',
                              cmap='hsv', cmap_range=[0, 1], ax=ax[1], colorbar=False)

    for i, threshold in enumerate(thresholds):

        i = i + 2

        filtered_roi_mask_dict = {}
        if less_than:
            filtered_roi_ids = metrics_df[metrics_df[metric] < threshold].cell_roi_id.values
        else:
            filtered_roi_ids = metrics_df[metrics_df[metric] > threshold].cell_roi_id.values
        for cell_roi_id in filtered_roi_ids:
            filtered_roi_mask_dict[cell_roi_id] = ct[ct.cell_roi_id == cell_roi_id].roi_mask.values[0]
        if less_than:
            ax[i] = plot_metrics_mask(filtered_roi_mask_dict, max_projection, metric_values=None,
                                      title=metric + ' < ' + str(threshold),
                                      cmap='hsv', cmap_range=[0, 1], ax=ax[i], colorbar=False)
        else:
            ax[i] = plot_metrics_mask(filtered_roi_mask_dict, max_projection, metric_values=None,
                                      title=metric + ' > ' + str(threshold),
                                      cmap='hsv', cmap_range=[0, 1], ax=ax[i], colorbar=False)
        ax[i].axis('off')

    fig.tight_layout()
    plt.suptitle(title, x=0.5, y=1.01, fontsize=18)
    save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/roi_filtering_validation'
    utils.save_figure(fig, figsize, save_dir, metric, title + '_' + metric)


def get_roi_masks_dict(cell_table):
    roi_masks = {}
    for cell_specimen_id in cell_table.index:
        roi_masks[cell_specimen_id] = cell_table.loc[cell_specimen_id].roi_mask
    return roi_masks


def plot_roi_metrics_for_cell(dataset, metrics_df, cell_specimen_id, title):
    # make roi masks dict
    cell_table = dataset.cell_specimen_table.copy()
    roi_masks = get_roi_masks_dict(cell_table)

    figsize = (15, 8)
    fig, ax = plt.subplots(figsize=figsize, nrows=2, ncols=4)

    # get flattened segmentation mask and binarize
    boolean_mask = cell_table.loc[cell_specimen_id].roi_mask
    binary_mask = np.zeros(boolean_mask.shape)
    binary_mask[:] = np.nan
    binary_mask[boolean_mask == True] = 1

    ax[0, 0].imshow(dataset.max_projection.data, cmap='gray')
    ax[0, 0].imshow(binary_mask, cmap='hsv', vmin=0, vmax=1, alpha=0.5)

    ax[0, 1] = sf.plot_cell_zoom(roi_masks, dataset.max_projection.data, cell_specimen_id,
                                 spacex=40, spacey=40, show_mask=True, ax=ax[0, 1])

    ax[0, 2] = sf.plot_cell_zoom(roi_masks, dataset.max_projection.data, cell_specimen_id,
                                 spacex=40, spacey=40, show_mask=False, ax=ax[0, 2])

    metrics = ['valid_roi', 'area', 'ellipseness', 'compactness', 'mean_intensity', 'max_intensity', 'intensity_ratio',
               'soma_minus_np_mean', 'soma_minus_np_std', 'sig_active_frames_2_5', 'sig_active_frames_4']
    cell_metrics = metrics_df[metrics_df.cell_specimen_id == cell_specimen_id]
    string = ''
    for metric in metrics[:7]:
        string = string + metric + ': ' + str(cell_metrics[metric].values[0]) + '\n'
    ax[0, 3].text(x=0, y=0, s=string)

    for i in range(1, 4):
        ax[0, i].axis('off')
    gs = ax[1, 0].get_gridspec()
    for ax in ax[1, :]:
        ax.remove()
    trace_ax = fig.add_subplot(gs[1, :])
    trace_ax = sf.plot_trace(dataset.ophys_timestamps, dataset.dff_traces.loc[cell_specimen_id].dff, ax=trace_ax,
                             title='cell_specimen_id: ' + str(cell_specimen_id), ylabel='dF/F')

    fig.tight_layout()
    title = title + '_' + str(cell_specimen_id)
    plt.suptitle(title, x=0.5, y=1.01, fontsize=18)
    save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/roi_filtering_validation'
    utils.save_figure(fig, figsize, save_dir, 'single_cell_metrics', title)


if __name__ == '__main__':
    import sys

    experiment_id = sys.argv[1]

    dataset = loading.get_ophys_dataset(experiment_id, include_invalid_rois=True)
    max_projection = dataset.max_projection.data
    ct = dataset.cell_specimen_table.copy()
    title = dataset.metadata_string
    metrics_df = get_metrics_df(experiment_id)
    # metrics_df = metrics_df[metrics_df.valid_roi==True] #only filter valid ROIs

    for cell_specimen_id in dataset.cell_specimen_ids:
        plot_roi_metrics_for_cell(dataset, metrics_df, cell_specimen_id, title)

    metric = 'area'
    thresholds = [50, 75, 100, 150, 200, 250]
    plot_metric_range_dataset(dataset, ct, max_projection, metrics_df, metric, thresholds, title, less_than=False)
    plot_metrics_distribution(metrics_df, title, metric)

    metric = 'compactness'
    thresholds = [6, 8, 10, 12, 14, 16]
    plot_metric_range_dataset(dataset, ct, max_projection, metrics_df, metric, thresholds, title, less_than=False)
    plot_metrics_distribution(metrics_df, title, metric)

    metric = 'compactness'
    thresholds = [20, 18, 16, 14, 12, 10]
    plot_metric_range_dataset(dataset, ct, max_projection, metrics_df, metric, thresholds, title, less_than=True)
    plot_metrics_distribution(metrics_df, title+'_less_than', metric)

    metric = 'ellipseness'
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    plot_metric_range_dataset(dataset, ct, max_projection, metrics_df, metric, thresholds, title, less_than=False)
    plot_metrics_distribution(metrics_df, title, metric)

    metric = 'intensity_ratio'
    thresholds = [5, 4, 3, 2, 1, 0.5]
    plot_metric_range_dataset(dataset, ct, max_projection, metrics_df, metric, thresholds, title, less_than=True)
    plot_metrics_distribution(metrics_df, title, metric)

    metric = 'mean_intensity'
    thresholds = [20, 40, 60, 80, 100, 120]
    plot_metric_range_dataset(dataset, ct, max_projection, metrics_df, metric, thresholds, title, less_than=False)
    plot_metrics_distribution(metrics_df, title, metric)

    metric = 'max_intensity'
    thresholds = [10, 20, 30, 40, 50, 60]
    plot_metric_range_dataset(dataset, ct, max_projection, metrics_df, metric, thresholds, title, less_than=False)
    plot_metrics_distribution(metrics_df, title, metric)

