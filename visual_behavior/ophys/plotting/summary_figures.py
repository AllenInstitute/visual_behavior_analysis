"""
Created on Sunday July 15 2018

@author: marinag
"""

import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from ..io import convert_level_1_to_level_2 as io

# formatting
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_palette('deep')


def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png']):
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    filename = os.path.join(fig_dir, fig_title)
    for f in formats:
        fig.savefig(filename + f, transparent=True, orientation='landscape')


def plot_cell_zoom(roi_masks, max_projection, cell_id, spacex=10, spacey=10, show_mask=False, ax=None):
    m = roi_masks[cell_id]
    (y, x) = np.where(m == 1)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    mask = np.empty(m.shape)
    mask[:] = np.nan
    mask[y, x] = 1
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(max_projection, cmap='gray', vmin=0, vmax=np.amax(max_projection))
    if show_mask:
        ax.imshow(mask, cmap='jet', alpha=0.3, vmin=0, vmax=1)
    ax.set_xlim(xmin - spacex, xmax + spacex)
    ax.set_ylim(ymin - spacey, ymax + spacey)
    ax.set_title('cell ' + str(cell_id))
    ax.grid(False)
    ax.axis('off')
    return ax


def plot_roi_validation(lims_data):
    file_path = os.path.join(io.get_processed_dir(lims_data), 'roi_traces.h5')
    g = h5py.File(file_path)
    roi_traces = np.asarray(g['data'])
    roi_names = np.asarray(g['roi_names'])
    g.close()

    dff_path = os.path.join(io.get_ophys_experiment_dir(lims_data), str(io.get_lims_id(lims_data)) + '_dff.h5')
    f = h5py.File(dff_path)
    dff_traces_original = np.asarray(f['data'])
    f.close()

    roi_df = io.get_roi_locations(lims_data)
    roi_metrics = io.get_roi_metrics(lims_data)
    roi_masks = io.get_roi_masks(roi_metrics, lims_data)
    dff_traces = io.get_dff_traces(roi_metrics, lims_data)
    cell_specimen_ids = io.get_cell_specimen_ids(roi_metrics)
    max_projection = io.get_max_projection(lims_data)
    analysis_dir = io.get_analysis_dir(lims_data)

    for index, id in enumerate(roi_names):
        fig, ax = plt.subplots(3, 2, figsize=(20, 10))
        ax = ax.ravel()

        id = int(id)
        x = roi_df[roi_df.id == id]['x'].values[0]
        y = roi_df[roi_df.id == id]['y'].values[0]
        valid = roi_df[roi_df.id == id]['valid'].values[0]
        ax[0].imshow(roi_df[roi_df.id == id]['mask'].values[0])
        ax[0].set_title(str(id) + ', ' + str(valid) + ', x: ' + str(x) + ', y: ' + str(y))
        ax[0].grid(False)

        ax[1].plot(roi_traces[index])
        ax[1].set_title('index: ' + str(index) + ', id: ' + str(id))
        ax[1].set_ylabel('fluorescence counts')

        ax[3].plot(dff_traces_original[index])
        ax[3].set_title('index: ' + str(index) + ', id: ' + str(id))
        ax[3].set_ylabel('dF/F')

        if id in cell_specimen_ids:
            cell_index = io.get_cell_index_for_cell_specimen_id(cell_specimen_ids, id)
            ax[2] = plot_cell_zoom(roi_masks, max_projection, id, spacex=10, spacey=10, show_mask=True, ax=ax[2])
            ax[2].grid(False)

            ax[4].imshow(max_projection, cmap='gray')
            mask = np.empty(roi_masks[id].shape)
            mask[:] = np.nan
            (y, x) = np.where(roi_masks[id] == 1)
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            ax[4].imshow(mask, cmap='RdBu', alpha=0.5)
            ax[4].set_xlim(xmin - 20, xmax + 20)
            ax[4].set_ylim(ymin - 20, ymax + 20)
            ax[4].grid(False)

            ax[5].plot(dff_traces[cell_index])
            ax[5].set_title('roi index: ' + str(cell_index) + ', id: ' + str(id))
            ax[5].set_ylabel('dF/F')
            ax[5].set_xlabel('frames')
        else:
            cell_index = ''

        fig.tight_layout()
        save_figure(fig, (20, 10), analysis_dir, 'roi_validation',
                    str(index) + '_' + str(id) + '_' + str(cell_index))
        plt.close()
