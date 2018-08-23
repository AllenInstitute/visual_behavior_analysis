"""
Created on Wednesday August 22 2018

@author: marinag
"""
import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import visual_behavior.ophys.response_analysis.utilities as ut
import visual_behavior.ophys.plotting.summary_figures as sf

# formatting
sns.set_style('white')
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

def plot_lick_raster(trials, ax=None, save_dir=None):
    if ax is None:
        figsize = (5, 10)
        fig, ax = plt.subplots(figsize=figsize)
    for trial in trials.trial.values:
        trial_data = trials.iloc[trial]
        # get times relative to change time
        trial_start = trial_data.start_time - trial_data.change_time
        lick_times = [(t - trial_data.change_time) for t in trial_data.lick_times]
        reward_time = [(t - trial_data.change_time) for t in trial_data.reward_times]
        # plot trials as colored rows
        ax.axhspan(trial, trial + 1, -200, 200, color=trial_data.trial_type_color, alpha=.5)
        # plot reward times
        if len(reward_time) > 0:
            ax.plot(reward_time[0], trial + 0.5, '.', color='b', label='reward', markersize=6)
        ax.vlines(trial_start, trial, trial + 1, color='black', linewidth=1)
        # plot lick times
        ax.vlines(lick_times, trial, trial + 1, color='k', linewidth=1)
        # annotate change time
        ax.vlines(0, trial, trial + 1, color=[.5, .5, .5], linewidth=1)
    # gray bar for response window
    ax.axvspan(trial_data.response_window[0], trial_data.response_window[1], facecolor='gray', alpha=.4,
        edgecolor='none')
    ax.grid(False)
    ax.set_ylim(0, len(trials))
    ax.set_xlim([-1, 4])
    ax.set_ylabel('trials')
    ax.set_xlabel('time (sec)')
    ax.set_title('lick raster')
    plt.gca().invert_yaxis()

    if save_dir:
        save_figure(fig, figsize, save_dir, 'behavior', 'lick_raster')

def plot_traces_heatmap(dff_traces, ax=None, save_dir=None):
    if ax is None:
        figsize = (20, 8)
        fig, ax = plt.subplots(figsize=figsize)
    cax = ax.pcolormesh(dff_traces, cmap='magma', vmin=0, vmax=np.percentile(dff_traces[np.isnan(dff_traces)==False], 99))
    ax.set_ylim(0, dff_traces.shape[0])
    ax.set_xlim(0,dff_traces.shape[1])
    ax.set_ylabel('cells')
    ax.set_xlabel('2P frames')
    cb = plt.colorbar(cax, pad = 0.015);
    cb.set_label('dF/F', labelpad=3)
    if save_dir:
        save_figure(fig, figsize, save_dir, 'experiment_summary', 'traces_heatmap')
    return ax

def plot_mean_image_response_heatmap(mean_df, title=None, ax=None, save_dir=None):
    df  = mean_df.copy()
    images = np.sort(df.change_image_name.unique())
    cell_list = []
    for image in images:
        tmp = df[(df.change_image_name == image) & (df.pref_stim == True)]
        order = np.argsort(tmp.mean_response.values)[::-1]
        cell_ids = list(tmp.cell.values[order])
        cell_list = cell_list + cell_ids

    response_matrix = np.empty((len(cell_list), len(images)))
    for i, cell in enumerate(cell_list):
        responses = []
        for image in images:
            response = df[(df.cell == cell) & (df.change_image_name == image)].mean_response.values[0]
            responses.append(response)
        response_matrix[i, :] = np.asarray(responses)

    if ax is None:
        figsize=(5, 8)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(response_matrix, cmap='magma', linewidths=0, linecolor='white', square=False,
                     vmin=0, vmax=0.3, robust=True,
                     cbar_kws={"drawedges": False, "shrink": 1, "label": "mean dF/F"}, ax=ax)

    if title is None:
        title = 'mean response by image'
    ax.set_title(title, va='bottom', ha='center')
    ax.set_xticklabels(images, rotation=90)
    ax.set_ylabel('cells')
    interval = 10
    ax.set_yticks(np.arange(0, response_matrix.shape[0], interval));
    ax.set_yticklabels(np.arange(0, response_matrix.shape[0], interval));
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'experiment_summary', 'mean_image_response_heatmap')

def plot_mean_trace_heatmap(mean_df, condition='trial_type', condition_values=['go','catch'], ax=None, save_dir=None):
    data = mean_df[mean_df.pref_stim==True].copy()

    cell_list = []
    for cell in data.cell.unique():
        # only include cells that have a mean response for both conditions
        if len(data[(data.cell == cell) & (data[condition].isin(condition_values))]) == len(condition_values):
            cell_list.append(cell)
    data = data[(data.cell.isin(cell_list))]

    vmax = 0.5
    if ax is None:
        figsize = (3*len(condition_values), 6)
        fig, ax = plt.subplots(1, len(condition_values), figsize=figsize, sharey=True)
        ax = ax.ravel()

    for i, condition_value in enumerate(condition_values):
        im_df = data[(data[condition] == condition_value)]
        if i == 0:
            order = np.argsort(im_df.mean_response.values)
        mean_traces = im_df.mean_trace.values[order][::-1]
        response_array = np.empty((len(im_df), mean_traces[0].shape[0]))
        for x, trace in enumerate(mean_traces):
            response_array[x, :] = trace

        sns.heatmap(data=response_array, vmin=0, vmax=vmax, ax=ax[i], cmap='viridis', cbar=False)
        xticks, xticklabels = sf.get_xticks_xticklabels(trace, 31., interval_sec=1)
        ax[i].set_xticks(xticks);
        ax[i].set_xticklabels([int(x) for x in xticklabels]);
        ax[i].set_yticks([0, response_array.shape[0]])
        ax[i].set_yticklabels([0, response_array.shape[0]])
        ax[i].set_xlabel('time after change (s)', fontsize=16)
        ax[i].set_title(condition_value)
        ax[0].set_ylabel('cells')

        if save_dir:
            fig.tight_layout()
            save_figure(fig, figsize, save_dir, 'experiment_summary', 'mean_trace_heatmap_'+condition)



