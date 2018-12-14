"""
Created on Wednesday August 22 2018

@author: marinag
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import visual_behavior.ophys.response_analysis.utilities as ut
import visual_behavior.visualization.ophys.summary_figures as sf
from visual_behavior.visualization.utils import save_figure
from visual_behavior import utilities as vbut
import seaborn as sns



def placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 1], wspace=None, hspace=None, sharex=False, sharey=False):
    '''
    Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec

    Takes as arguments:
        fig: figure handle - required
        dim: number of rows and columns in the subaxes - defaults to 1x1
        xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
        yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
        wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively

    returns:
        subaxes handles
    '''
    import matplotlib.gridspec as gridspec

    outer_grid = gridspec.GridSpec(100, 100)
    inner_grid = gridspec.GridSpecFromSubplotSpec(dim[0], dim[1],
                    subplot_spec=outer_grid[int(100 * yspan[0]):int(100 * yspan[1]), # flake8: noqa: E999
                    int(100 * xspan[0]):int(100 * xspan[1])], wspace=wspace, hspace=hspace) # flake8: noqa: E999

    # NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0] * [dim[1] * [
        fig]]  # filling the list with figure objects prevents an error when it they are later replaced by axis handles
    inner_ax = np.array(inner_ax)
    idx = 0
    for row in range(dim[0]):
        for col in range(dim[1]):
            if row > 0 and sharex == True:
                share_x_with = inner_ax[0][col]
            else:
                share_x_with = None

            if col > 0 and sharey == True:
                share_y_with = inner_ax[row][0]
            else:
                share_y_with = None

            inner_ax[row][col] = plt.Subplot(fig, inner_grid[idx], sharex=share_x_with, sharey=share_y_with)
            fig.add_subplot(inner_ax[row, col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist()  # remove redundant dimension
    return inner_ax


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


def plot_traces_heatmap(traces, ax=None, save_dir=None, use_events=False):
    if use_events:
        # vmax = 0.03
        vmax = np.percentile(traces, 99)
        label = 'event magnitude'
        suffix = '_events'
    else:
        vmax = np.percentile(traces, 99)
        label = 'dF/F'
        suffix = ''
    if ax is None:
        figsize = (20, 8)
        fig, ax = plt.subplots(figsize=figsize)
    cax = ax.pcolormesh(traces, cmap='magma', vmin=0, vmax=vmax)
    ax.set_ylim(0, traces.shape[0])
    ax.set_xlim(0, traces.shape[1])
    ax.set_ylabel('cells')
    ax.set_xlabel('2P frames')
    cb = plt.colorbar(cax, pad=0.015)
    cb.set_label(label, labelpad=3)
    if save_dir:
        save_figure(fig, figsize, save_dir, 'experiment_summary', 'traces_heatmap'+suffix)
    return ax


def plot_mean_image_response_heatmap(mean_df, title=None, ax=None, save_dir=None, use_events=False):
    df = mean_df.copy()
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
        figsize = (5, 8)
        fig, ax = plt.subplots(figsize=figsize)
    if use_events:
        vmax = 0.03
        label = 'mean event magnitude'
        suffix = '_events'
    else:
        vmax = 0.3
        label = 'mean dF/F'
        suffix = ''
    ax = sns.heatmap(response_matrix, cmap='magma', linewidths=0, linecolor='white', square=False,
                     vmin=0, vmax=vmax, robust=True,
                     cbar_kws={"drawedges": False, "shrink": 1, "label": label}, ax=ax)

    if title is None:
        title = 'mean response by image'
    ax.set_title(title, va='bottom', ha='center')
    ax.set_xticklabels(images, rotation=90)
    ax.set_ylabel('cells')
    interval = 10
    ax.set_yticks(np.arange(0, response_matrix.shape[0], interval))
    ax.set_yticklabels(np.arange(0, response_matrix.shape[0], interval))
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'experiment_summary', 'mean_image_response_heatmap'+suffix)


def plot_mean_trace_heatmap(mean_df, condition='trial_type', condition_values=['go', 'catch'], ax=None, save_dir=None, use_events=False):
    data = mean_df[mean_df.pref_stim == True].copy()
    if use_events:
        vmax = 0.03
        suffix = '_events'
    else:
        vmax = 0.5
        suffix = ''
    if ax is None:
        figsize = (3 * len(condition_values), 6)
        fig, ax = plt.subplots(1, len(condition_values), figsize=figsize, sharey=True)
        ax = ax.ravel()

    for i, condition_value in enumerate(condition_values):
        im_df = data[(data[condition] == condition_value)]
        if len(im_df) != 0:
            if i == 0:
                order = np.argsort(im_df.mean_response.values)[::-1]
                cells = im_df.cell.unique()[order]
            len_trace = len(im_df.mean_trace.values[0])
            response_array = np.empty((len(cells), len_trace))
            for x, cell in enumerate(cells):
                tmp = im_df[im_df.cell == cell]
                if len(tmp) >= 1:
                    trace = tmp.mean_trace.values[0]
                else:
                    trace = np.empty((len_trace))
                    trace[:] = np.nan
                response_array[x, :] = trace

            sns.heatmap(data=response_array, vmin=0, vmax=vmax, ax=ax[i], cmap='magma', cbar=False)
            xticks, xticklabels = sf.get_xticks_xticklabels(trace, 31., interval_sec=1)
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels([int(x) for x in xticklabels])
            ax[i].set_yticks(np.arange(0, response_array.shape[0], 10))
            ax[i].set_yticklabels(np.arange(0, response_array.shape[0], 10))
            ax[i].set_xlabel('time after change (s)', fontsize=16)
            ax[i].set_title(condition_value)
            ax[0].set_ylabel('cells')

    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'experiment_summary', 'mean_trace_heatmap_' + condition + suffix)


def get_upper_limit_and_intervals(dff_traces, timestamps_ophys):
    upper = np.round(dff_traces.shape[1], -3) + 1000
    interval = 5 * 60
    frame_interval = np.arange(0, len(dff_traces), interval * 31)
    time_interval = np.uint64(np.round(np.arange(timestamps_ophys[0], timestamps_ophys[-1], interval), 1))
    return upper, time_interval, frame_interval


def plot_run_speed(running_speed, timestamps_stimulus, ax=None, label=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(timestamps_stimulus, running_speed, color='gray')
    if label:
        ax.set_ylabel('run speed (cm/s)')
        ax.set_xlabel('time(s)')
    return ax


def plot_d_prime(trials, d_prime, ax=None):
    colors = sns.color_palette()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(trials.change_time.values, d_prime, color=colors[4], linewidth=4, label='d_prime')
    ax.set_ylabel('d prime')
    return ax


def plot_hit_false_alarm_rates(trials, ax=None):
    trials['auto_rewarded'] = False
    hr, cr, d_prime = vbut.get_response_rates(trials, sliding_window=100, reward_window=None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(trials.change_time.values, hr, color='#55a868', linewidth=4, label='hit_rate')
    ax.plot(trials.change_time.values, cr, color='#c44e52', linewidth=4, label='fa_rate')
    ax.set_ylabel('response rate')
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc='upper right')
    return ax


def plot_reward_rate(trials, ax=None):
    colors = sns.color_palette()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(trials.change_time.values, trials.reward_rate.values, color=colors[0], linewidth=4, label='reward_rate')
    ax.set_ylabel('reward rate')
    return ax


def format_table_data(dataset):
    table_data = dataset.metadata.copy()
    table_data = table_data[['donor_id', 'targeted_structure', 'imaging_depth', 'cre_line',
                             'experiment_date', 'session_type', 'ophys_experiment_id']]
    table_data = table_data.transpose()
    return table_data


def plot_experiment_summary_figure(analysis, save_dir=None, use_events=False):
    if use_events:
        traces = analysis.dataset.events.copy()
        suffix = '_events'
    else:
        traces = analysis.dataset.dff_traces.copy()
        suffix = ''

    interval_seconds = 600
    ophys_frame_rate = 31

    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.8, 0.95), yspan=(0, .3))
    table_data = format_table_data(analysis.dataset)
    xtable = ax.table(cellText=table_data.values, cellLoc='left', rowLoc='left', loc='center', fontsize=12)
    xtable.scale(1.5, 3)
    ax.axis('off')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .22), yspan=(0, .27))
    ax.imshow(analysis.dataset.max_projection, cmap='gray', vmin=0, vmax=np.amax(analysis.dataset.max_projection) / 2.)
    ax.set_title(analysis.dataset.experiment_id)
    ax.axis('off')

    upper_limit, time_interval, frame_interval = get_upper_limit_and_intervals(traces,
                                                                               analysis.dataset.timestamps_ophys)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.22, 0.9), yspan=(0, .3))
    ax = plot_traces_heatmap(traces, ax=ax, use_events=use_events)
    ax.set_xticks(np.arange(0, upper_limit, interval_seconds * ophys_frame_rate))
    ax.set_xticklabels(np.arange(0, upper_limit / ophys_frame_rate, interval_seconds))
    ax.set_xlabel('time (seconds)')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.22, 0.8), yspan=(.26, .41))
    ax = plot_run_speed(analysis.dataset.running_speed.running_speed, analysis.dataset.timestamps_stimulus, ax=ax,
                        label=True)
    ax.set_xlim(time_interval[0], np.uint64(upper_limit / ophys_frame_rate))
    ax.set_xticks(np.arange(interval_seconds, upper_limit / ophys_frame_rate, interval_seconds))
    ax.set_xlabel('time (seconds)')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.22, 0.8), yspan=(.37, .52))
    ax = plot_hit_false_alarm_rates(analysis.dataset.trials, ax=ax)
    ax.set_xlim(time_interval[0], np.uint64(upper_limit / ophys_frame_rate))
    ax.set_xticks(np.arange(interval_seconds, upper_limit / ophys_frame_rate, interval_seconds))
    ax.legend(loc='upper right', ncol=2, borderaxespad=0.)
    ax.set_xlabel('time (seconds)')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .22), yspan=(.25, .8))
    ax = plot_lick_raster(analysis.dataset.trials, ax=ax, save_dir=None)

    ax = placeAxesOnGrid(fig, dim=(1, 4), xspan=(.2, .8), yspan=(.5, .8), wspace=0.35)
    mdf = ut.get_mean_df(analysis.trial_response_df,
                         conditions=['cell', 'change_image_name', 'behavioral_response_type'])
    ax = plot_mean_trace_heatmap(mdf, condition='behavioral_response_type',
                                 condition_values=['HIT', 'MISS', 'CR', 'FA'], ax=ax, save_dir=None, use_events=use_events)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.78, 0.97), yspan=(.3, .8))
    mdf = ut.get_mean_df(analysis.trial_response_df, conditions=['cell', 'change_image_name'])
    ax = plot_mean_image_response_heatmap(mdf, title=None, ax=ax, save_dir=None, use_events=use_events)

    fig.tight_layout()

    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'experiment_summary', analysis.dataset.analysis_folder+suffix)


def plot_mean_first_flash_response_by_image_block(analysis, save_dir=None, ax=None):
    fdf = analysis.flash_response_df.copy()
    fdf.image_block = [int(image_block) for image_block in fdf.image_block.values]
    data = fdf[(fdf.repeat == 1) & (fdf.pref_stim == True)]
    mean_response = data.groupby(['cell']).apply(ut.get_mean_sem)
    mean_response = mean_response.unstack()

    cell_order = np.argsort(mean_response.mean_response.values)
    if ax is None:
        figsize = (15, 5)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.pointplot(data=data, x="image_block", y="mean_response", kind="point", hue='cell', hue_order=cell_order,
                       palette='Blues', ax=ax)
    # ax.legend(bbox_to_anchor=(1,1))
    ax.legend_.remove()
    min = mean_response.mean_response.min()
    max = mean_response.mean_response.max()
    norm = plt.Normalize(min, max)
    #     norm = plt.Normalize(0,5)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    ax.figure.colorbar(mappable=sm, ax=ax, label='mean response across blocks')
    ax.set_title('mean response to first flash of pref stim across image blocks')
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'first_flash_by_image_block', analysis.dataset.analysis_folder)
    return ax


def plot_mean_response_across_image_block_sets(data, analysis_folder, save_dir=None, ax=None):
    order = np.argsort(data[data.image_block == 1].early_late_block_ratio.values)
    cell_order = data[data.image_block == 1].cell.values[order]
    if ax is None:
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.pointplot(data=data, x="block_set", y="mean_response", kind="point", palette='RdBu', ax=ax,
                       hue='cell', hue_order=cell_order)
    # ax.legend(bbox_to_anchor=(1,1))
    ax.legend_.remove()
    min = np.amin(data.early_late_block_ratio.unique())
    max = np.amax(data.early_late_block_ratio.unique())
    norm = plt.Normalize(min, max)
    #     norm = plt.Normalize(0,5)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])
    ax.figure.colorbar(mappable=sm, ax=ax, label='first/last ratio')
    ax.set_title('mean response across image blocks\ncolored by ratio of first to last block')
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'first_flash_by_image_block_set', analysis_folder)
    return ax
