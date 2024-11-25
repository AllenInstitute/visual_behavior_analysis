"""
Created on Wednesday February 23 2022

@author: marinag
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.visualization.ophys.platform_paper_figures as ppf

import mindscope_utilities.general_utilities as ms_utils

import visual_behavior_glm.GLM_params as glm_params

# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
# sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
sns.set_palette('deep')


def plot_reliable_example_cells(multi_session_mean_df, cells_to_plot, cell_type, event_type='changes',
                                linewidth=2, save_dir=None, folder=None, suffix=''):
    '''
    Plot mean response for each experience level for a subset of cells that are reliably responsive

    multi_session_mean_df: dataframe where each row is 1 cell in 1 session
                            and columns contain `mean_trace`, `sem_trace`, and `trace_timestamps`
    cells_to_plot: list of cell_specimen_ids of cells that are reliably responsive
    cell_type: cell type of cells in cells_to_plot list, one of 'Excitatory', 'Sst Inhibitory', or 'Vip Inhibitory'
    event_type: 'changes', 'omissions', or 'all

    '''

    if event_type == 'changes':
        change = True
        omitted = False
        window = [-1, 0.75]
        window = [-0.25, 0.75]
        col_size = 0.75
        label = 'Image change'
        label_color = sns.color_palette()[0]
    elif event_type == 'omissions':
        change = False
        omitted = True
        window = [-1, 1.5]
        col_size = 1.25
        label = 'Image omission'
        label_color = sns.color_palette()[9]
    else:
        change = False
        omitted = False
        window = [-0.5, 0.75]
        col_size = 0.75

    interval_sec = 1

    sdf = multi_session_mean_df.copy()
    experience_levels = utils.get_experience_levels()
    experience_level_colors = utils.get_experience_level_colors()

    # set this here so there is something to compare to for the first cell
    last_container_id = sdf[(sdf.cell_specimen_id == cells_to_plot[0])].ophys_container_id.values[0]

    n_cols = 4
    figsize = (n_cols*col_size, (len(cells_to_plot) + 2) / 1.5)
    fig, ax = plt.subplots(len(cells_to_plot) + 2, n_cols, figsize=figsize, sharey='row')
    ax = ax.ravel()

    c = 0
    # plot population average
    for e, experience_level in enumerate(experience_levels):
        i = (c * n_cols) + e
        color = experience_level_colors[e]
        exp_data = sdf[(sdf.cell_type == cell_type) & (sdf.experience_level == experience_level)]
        traces = exp_data.mean_trace.values
        timestamps = exp_data.trace_timestamps.values[0]
        ax[i] = utils.plot_mean_trace(traces, timestamps,
                                      ylabel='', legend_label=None, color=color, linewidth=linewidth,
                                      interval_sec=interval_sec, xlim_seconds=window, plot_sem=True, ax=ax[i])
        ax[i] = utils.plot_flashes_on_trace(ax[i], timestamps, change=change, omitted=omitted, alpha=0.15,
                                            linewidth=0.75)
        ax[i].set_xticklabels([])
        ax[i].set_xlabel('')
        ymin, ymax = ax[i].get_ylim()
        ax[i].set_yticks([0, np.round(ymax * .3, 3)])
        ax[i].set_yticklabels(['', np.round(ymax * .3, 3)], va='top')

        if e == 0:
            # plot a line for 1/3 of the axis, corresponding to the yticklabel for y axis, which is set at 1/3 of the max y value
            # ax[i].axvline(x=window[0] - 0.1, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)
            if event_type == 'omissions':
                dist = 0.18
            else:
                dist = 0.1
            ax[i].axvline(x=window[0] - dist, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)

        sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
        ax[i].tick_params(bottom=False, left=False, right=False, top=False, labelsize=7, pad=-1)

    # annotate first axis
    i = 0
    ax[i].set_ylabel('grand avg.', rotation=0, fontsize=10, ha='right', y=0.4)

    # annotate time axis and change/omission for excitatory only
    if cell_type == 'Excitatory':
        # plot time axis bar
        # ax[i].set_xticks([-0.75, -0.25])
        ax[i].set_xticks([0, 0.5])
        ax[i].set_xticklabels(['', '0.5 s'], va='top')
        ax[i].annotate('', xy=(0, -0.08), xycoords=ax[i].get_xaxis_transform(), xytext=(0.5, -0.08), fontsize=8,
                       arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0), clip_on=False)
        # label image change or image omission
        ax[i].annotate(label, xy=(-0.17, 1.4), xycoords=ax[i].get_xaxis_transform(), ha="right", va="top",
                        color=label_color, fontsize=10, clip_on=False)
        ax[i].annotate('', xy=(0.01, 1.35), xycoords=ax[i].get_xaxis_transform(), xytext=(0.01, 0.95), fontsize=8,
                            arrowprops=dict(arrowstyle="<-", color=label_color, lw=1), clip_on=False)

    for i in np.arange(3, 8):
        sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
        ax[i].tick_params(bottom=False, left=False, right=False, top=False)
        ax[i].axis('off')

    # plot example cells
    for c, cell_specimen_id in enumerate(cells_to_plot):
        ophys_container_id = sdf[(sdf.cell_specimen_id == cell_specimen_id)].ophys_container_id.values[0]
        for e, experience_level in enumerate(experience_levels):
            i = ((c + 2) * n_cols) + e
            color = experience_level_colors[e]
            cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.experience_level == experience_level)]
            if len(cell_data) > 0:  # only plot if there is data for this exp level
                ax[i] = utils.plot_mean_trace_from_mean_df(cell_data, ylabel='', xlabel='', xlims=window,
                                                           color=color, interval_sec=interval_sec, linewidth=linewidth,
                                                           ax=ax[i])
                ax[i] = utils.plot_flashes_on_trace(ax[i], cell_data.trace_timestamps.values[0],
                                                    change=change, omitted=omitted, alpha=0.15, linewidth=0.75)
                ax[i].set_xticklabels([])
            else:
                ax[i].set_xticklabels([])
            ax[i].set_xlabel('')
            ymin, ymax = ax[i].get_ylim()
            ax[i].set_yticks([0, np.round(ymax * .3, 2)])
            ax[i].set_yticklabels(['', np.round(ymax * .3, 2)], va='top')
            if e == 0: # plot response magnitude bar
                if event_type == 'omissions':
                    dist = 0.18
                else:
                    dist = 0.1
                ax[i].axvline(x=window[0] - dist, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)
                ax[i].set_ylabel('cell ' + str(c + 1), rotation=0, fontsize=10, ha='right', y=0.4)
            sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
            ax[i].tick_params(bottom=False, left=False, right=False, top=False, labelsize=7, pad=-1, )
        # ax[(i*n_cols)+3].set_title('csid:'+str(cell_specimen_id)+'\nocid:'+str(ophys_container_id), fontsize=6)
        # 4th column with cell metadata
        i = ((c + 2) * n_cols) + 3
        if ophys_container_id != last_container_id:
            ylims = ax[i].get_ylim()
            ax[i].axhline(y=ylims[1], xmin=0, xmax=1)
            ax[i].set_xlim(0, 1)
            # ax[(i*n_cols)+3].set_ylim(0,2)
        ax[i].text(s='csid:' + str(cell_specimen_id) + '\nocid:' + str(ophys_container_id), x=0, y=0, fontsize=8)
        sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
        ax[i].tick_params(bottom=False, left=False, right=False, top=False)
        ax[i].set_xticklabels([])
        last_container_id = ophys_container_id
    ax[1].set_title(cell_type, fontsize=14)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, cell_type.split(' ')[0] + '_' + event_type + suffix)


def plot_reliable_example_cells_container(multi_session_mean_df, cells_to_plot, cell_type, event_type='changes',
                                linewidth=2, label_csids=True, save_dir=None, folder=None, suffix='', ax=None):
    '''
    Plot mean response for each experience level for a subset of cells that are reliably responsive

    multi_session_mean_df: dataframe where each row is 1 cell in 1 session
                            and columns contain `mean_trace`, `sem_trace`, and `trace_timestamps`
    cells_to_plot: list of cell_specimen_ids of cells that are reliably responsive
    cell_type: cell type of cells in cells_to_plot list, one of 'Excitatory', 'Sst Inhibitory', or 'Vip Inhibitory'
    event_type: 'changes', 'omissions', or 'all
    label_csids: Boolean, whether or not to include cell_specimen_id label in 4th column of plot

    '''

    if event_type == 'changes':
        change = True
        omitted = False
        window = [-1, 0.75]
        label = 'Image change'
        label_color = sns.color_palette()[0]
    elif event_type == 'omissions':
        change = False
        omitted = True
        window = [-1, 1.5]
        label = 'Image omission'
        label_color = sns.color_palette()[9]
    else:
        change = False
        omitted = False
        window = [-0.5, 0.75]

    interval_sec = 1

    sdf = multi_session_mean_df.copy()
    experience_levels = utils.get_experience_levels()
    experience_level_colors = utils.get_experience_level_colors()

    # set this here so there is something to compare to for the first cell
    last_container_id = sdf[(sdf.cell_specimen_id == cells_to_plot[0])].ophys_container_id.values[0]

    n_cols = 4
    if ax is None:
        figsize = (n_cols, (len(cells_to_plot) + 2) / 1.5)
        fig, ax = plt.subplots(len(cells_to_plot) + 2, n_cols, figsize=figsize, sharey='row')

    i = 0
    # plot population average
    for e, experience_level in enumerate(experience_levels):
        color = experience_level_colors[e]
        exp_data = sdf[(sdf.cell_type == cell_type) & (sdf.experience_level == experience_level)]
        traces = exp_data.mean_trace.values
        timestamps = exp_data.trace_timestamps.values[0]
        ax[i][e] = utils.plot_mean_trace(traces, timestamps,
                                         ylabel='', legend_label=None, color=color, linewidth=linewidth,
                                         interval_sec=interval_sec, xlim_seconds=window, plot_sem=True, ax=ax[i][e])
        ax[i][e] = utils.plot_flashes_on_trace(ax[i][e], timestamps, change=change, omitted=omitted, alpha=0.15,
                                               linewidth=0.75)
        ax[i][e].set_xticklabels([])
        ax[i][e].set_xlabel('')
        ymin, ymax = ax[i][e].get_ylim()
        # ax[i][e].set_yticks([0, np.round(ymax * .3, 3)])
        # ax[i][e].set_yticklabels(['', np.round(ymax * .3, 3)], va='top')
        if e == 0:
            ax[i][e].annotate(np.round(ymax * .3, 2), xy=(window[0] - 0.2, 0.3),
                          xycoords=ax[i][e].get_xaxis_transform(), ha="right", va="top", fontsize=6, clip_on=False,
                          annotation_clip=False)
        ax[i][e].set_yticks([0, np.round(ymax * .3, 2)])
        ax[i][e].set_yticklabels('', va='top')

        if e == 0:
            # plot a line for 1/3 of the axis, corresponding to the yticklabel for y axis, which is set at 1/3 of the max y value
            ax[i][e].axvline(x=window[0] - 0.1, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)

        sns.despine(ax=ax[i][e], top=True, right=True, left=True, bottom=True)
        ax[i][e].tick_params(bottom=False, left=False, right=False, top=False, labelsize=7, pad=-1)

    # annotate first axis
    i = 0
    e = 0
    ax[i][e].set_ylabel('pop. avg.', rotation=0, fontsize=10, ha='right', y=0.4)
    # annotate time axis and change/omission for excitatory only
    if cell_type == 'Excitatory':
        # plot time axis bar
        ax[i][e].set_xticks([-0.75, -0.25])
        ax[i][e].set_xticklabels(['', '0.5 s'], va='top')
        ax[i][e].annotate('', xy=(-0.75, -0.08), xycoords=ax[i][e].get_xaxis_transform(), xytext=(-0.25, -0.08),
                          fontsize=8,
                          arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0), clip_on=False)
        # label image change or image omission
        ax[i][e].annotate(label, xy=(-0.17, 1.4), xycoords=ax[i][e].get_xaxis_transform(), ha="right", va="top",
                          color=label_color, fontsize=10, clip_on=False)
        ax[i][e].annotate('', xy=(0.01, 1.35), xycoords=ax[i][e].get_xaxis_transform(), xytext=(0.01, 0.95), fontsize=8,
                          arrowprops=dict(arrowstyle="<-", color=label_color, lw=1), clip_on=False)

    # despine blank axes
    # top right
    sns.despine(ax=ax[0][3], top=True, right=True, left=True, bottom=True)
    ax[0][3].tick_params(bottom=False, left=False, right=False, top=False)
    ax[0][3].axis('off')
    # second row
    i = 1
    for e in np.arange(0, 4):
        sns.despine(ax=ax[i][e], top=True, right=True, left=True, bottom=True)
        ax[i][e].tick_params(bottom=False, left=False, right=False, top=False)
        ax[i][e].axis('off')

    # plot example cells
    for i, cell_specimen_id in enumerate(cells_to_plot):
        i = i + 2
        ophys_container_id = sdf[(sdf.cell_specimen_id == cell_specimen_id)].ophys_container_id.values[0]
        for e, experience_level in enumerate(experience_levels):
            # i = ((c + 2) * n_cols) + e
            color = experience_level_colors[e]
            cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.experience_level == experience_level)]
            if len(cell_data) > 0:  # only plot if there is data for this exp level
                ax[i][e] = utils.plot_mean_trace_from_mean_df(cell_data, ylabel='', xlabel='', xlims=window,
                                                              color=color, interval_sec=interval_sec,
                                                              linewidth=linewidth,
                                                              ax=ax[i][e])
                ax[i][e] = utils.plot_flashes_on_trace(ax[i][e], cell_data.trace_timestamps.values[0],
                                                       change=change, omitted=omitted, alpha=0.15, linewidth=0.75)
                ax[i][e].set_xticklabels([])
            else:
                ax[i][e].set_xticklabels([])
            ax[i][e].set_xlabel('')

            ymin, ymax = ax[i][e].get_ylim()
            if e == 0:
                ax[i][e].axvline(x=window[0] - 0.1, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)
                ax[i][e].set_ylabel('cell ' + str(i - 1), rotation=0, fontsize=10, ha='right', y=0.4)
                ax[i][e].annotate(np.round(ymax * .3, 2), xy=(window[0]-0.2, 0.3),
                                  xycoords=ax[i][e].get_xaxis_transform(), ha="right", va="top", fontsize=6, clip_on=False, annotation_clip=False)
            ax[i][e].set_yticks([0, np.round(ymax * .3, 2)])
            ax[i][e].set_yticklabels('', va='top')
            # ax[i][e].set_yticklabels(['', np.round(ymax * .3, 2)], va='top')
            sns.despine(ax=ax[i][e], top=True, right=True, left=True, bottom=True)
            ax[i][e].tick_params(bottom=False, left=False, right=False, top=False, labelsize=7, pad=-1, )
        # ax[(i*n_cols)+3].set_title('csid:'+str(cell_specimen_id)+'\nocid:'+str(ophys_container_id), fontsize=6)


        # 4th column with cell metadata
        e = 3
        if ophys_container_id != last_container_id:
            ylims = ax[i][e].get_ylim()
            ax[i][e].axhline(y=ylims[1], xmin=0, xmax=1)
            ax[i][e].set_xlim(0, 1)
            # ax[(i*n_cols)+3].set_ylim(0,2)
        if label_csids:
            ax[i][e].text(s='csid:' + str(cell_specimen_id) + '\nocid:' + str(ophys_container_id), x=0, y=0, fontsize=8)
        sns.despine(ax=ax[i][e], top=True, right=True, left=True, bottom=True)
        ax[i][e].tick_params(bottom=False, left=False, right=False, top=False)
        ax[i][e].set_xticklabels([])
        # ax[i][e].set_yticklabels([])
        last_container_id = ophys_container_id

    # for c, cell_specimen_id in enumerate(cells_to_plot):
    #     c = c + 2
    #     if e == 0:
    #         ymin, ymax = ax[c][e].get_ylim()
    #         ax[c][e].set_yticks([0, np.round(ymax * .3, 2)])
    #         ax[c][e].axvline(x=window[0] - 0.1, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)
    #         ax[c][e].set_ylabel('cell ' + str(i - 1), rotation=0, fontsize=10, ha='right', y=0.4)
    #         ax[c][e].annotate(np.round(ymax * .3, 2), xy=(window[0] - 0.2, 0.3),
    #                           xycoords=ax[c][e].get_xaxis_transform(), ha="right", va="top", fontsize=6, clip_on=False,
    #                           annotation_clip=False)

    ax[0][1].set_title(cell_type, fontsize=14)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, cell_type.split(' ')[0] + '_' + event_type + suffix)
    return ax


def plot_behavior_and_physio_timeseries_stacked(dataset, start_time, duration_seconds=20,
                                                label_changes=True, label_omissions=True,
                                                save_dir=None, ax=None):
    """
    Plots licking behavior, rewards, running speed, pupil area, and dff traces for a defined window of time.
    Each timeseries gets its own row. If label_changes=True, all flashes are gray, changes are blue.
    If label_changes=False, unique colors are given to each image.
    If label_omissions=True, a dotted line will be plotted at the time of omissions.
    Selects the top 6 cell traces with highest SNR to plot
    """

    if label_changes:
        suffix = '_changes'
    else:
        suffix = '_colors'

    xlim_seconds = [start_time - (duration_seconds / 4.), start_time + duration_seconds * 2]

    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))
    licks[:] = -2

    reward_timestamps = dataset.rewards.timestamps.values
    rewards = np.zeros(len(reward_timestamps))
    rewards[:] = -4

    # get run speed trace and timestamps
    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values
    # limit running trace to window so yaxes scale properly
    start_ind = np.where(running_timestamps < xlim_seconds[0])[0][-1]
    stop_ind = np.where(running_timestamps > xlim_seconds[1])[0][0]
    running_speed = running_speed[start_ind:stop_ind]
    running_timestamps = running_timestamps[start_ind:stop_ind]

    # get pupil width trace and timestamps
    eye_tracking = dataset.eye_tracking.copy()
    pupil_diameter = eye_tracking.pupil_width.values
    pupil_diameter[eye_tracking.likely_blink == True] = np.nan
    pupil_timestamps = eye_tracking.timestamps.values
    # smooth pupil diameter
    from scipy.signal import medfilt
    pupil_diameter = medfilt(pupil_diameter, kernel_size=5)
    # limit pupil trace to window so yaxes scale properly
    start_ind = np.where(pupil_timestamps < xlim_seconds[0])[0][-1]
    stop_ind = np.where(pupil_timestamps > xlim_seconds[1])[0][0]
    pupil_diameter = pupil_diameter[start_ind:stop_ind]
    pupil_timestamps = pupil_timestamps[start_ind:stop_ind]

    # get cell traces and events
    ophys_timestamps = dataset.ophys_timestamps.copy()
    dff_traces = dataset.dff_traces.copy()
    events = dataset.events.copy()
    events = events.loc[dff_traces.index.values]

    if ax is None:
        figsize = (15, 8)
        fig, ax = plt.subplots(10, 1, figsize=figsize, sharex=True,
                               gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 1.5, 1.5, 1, 1, ]})
        ax = ax.ravel()

    colors = sns.color_palette()

    ax[8].plot(lick_timestamps, licks, '|', label='licks', color='gray', markersize=10)
    ax[8].set_yticklabels([])
    ax[8].set_ylabel('licks', rotation=0, horizontalalignment='right', verticalalignment='center')

    ax[9].plot(reward_timestamps, rewards, 'o', label='rewards', color='gray', markersize=10)
    ax[9].set_yticklabels([])
    ax[9].set_ylabel('rewards', rotation=0, horizontalalignment='right', verticalalignment='center')

    ax[6].plot(running_timestamps, running_speed, label='running_speed', color='gray')
    ax[6].set_ylabel('running\nspeed\n(cm/s)', rotation=0, horizontalalignment='right', verticalalignment='center')
    ax[6].set_ylim(ymin=-8)

    ax[7].plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color='gray')
    ax[7].set_ylabel('pupil\ndiameter\n(pixels)', rotation=0, horizontalalignment='right', verticalalignment='center')

    #     for experiment_id = 807753334
    #     indices = [277, 84, 183, 236, 73, 142]
    #     cell_specimen_ids = dff_traces.iloc[indices].index.values
    cell_specimen_ids = oof.sort_trace_csids_by_max_in_window(dff_traces, ophys_timestamps, xlim_seconds)
    for cell_index, cell_specimen_id in enumerate(cell_specimen_ids[:6]):
        dff_trace = dff_traces.loc[cell_specimen_id]['dff']
        events_trace = events.loc[cell_specimen_id]['events']
        events_trace[events_trace == 0]
        # limit cell trace to window so yaxes scale properly
        start_ind = np.where(ophys_timestamps < xlim_seconds[0])[0][-1]
        stop_ind = np.where(ophys_timestamps > xlim_seconds[1])[0][0]
        dff_trace = dff_trace[start_ind:stop_ind]
        events_trace = events_trace[start_ind:stop_ind]
        timestamps = ophys_timestamps[start_ind:stop_ind]
        ax[cell_index].plot(timestamps, dff_trace, label=str(cell_specimen_id), color='gray')
        for timepoint in np.where(events_trace != 0)[0]:
            ax[cell_index].axvline(x=timestamps[timepoint], ymin=0, ymax=events_trace[timepoint], color=colors[6])
            ax[cell_index].set_yticks((0, 2))

    for i in range(10):
        ax[i] = ppf.add_stim_color_span(dataset, ax[i], xlim=xlim_seconds, label_changes=label_changes,
                                    label_omissions=label_omissions)
        ax[i].set_xlim(xlim_seconds)
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=True,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True)

    # label bottom row of plot
    ax[i].set_xlabel('time in session (seconds)')
    ax[i].tick_params(which='both', bottom=True, top=False, right=False, left=True,
                      labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    # add title to top row
    metadata_string = utils.get_metadata_string(dataset.metadata)
    ax[0].set_title(metadata_string)

    plt.subplots_adjust(hspace=0)
    if save_dir:
        print('saving')
        folder = 'behavior_physio_timeseries_stacked'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)) + '_' + duration_seconds + suffix,
                          formats=['.png'])
    return ax


def plot_single_cell_example_timeseries_stacked(dataset, start_time, duration_seconds=20, cell_specimen_ids=None,
                                                label_changes=True, label_omissions=True,
                                                save_dir=None, ax=None):
    """
    Plots licking behavior, rewards, running speed, pupil area, and dff traces for a defined window of time.
    Each timeseries gets its own row. If label_changes=True, all flashes are gray, changes are blue.
    If label_changes=False, unique colors are given to each image.
    If label_omissions=True, a dotted line will be plotted at the time of omissions.
    Selects the top 20 cell traces with highest SNR to plot
    """

    if label_changes:
        suffix = '_changes'
    else:
        suffix = '_colors'

    xlim_seconds = [start_time - (duration_seconds / 3.), start_time + duration_seconds * 2]

    # get cell traces and events
    ophys_timestamps = dataset.ophys_timestamps.copy()
    dff_traces = dataset.dff_traces.copy()
    events = dataset.events.copy()
    events = events.loc[dff_traces.index.values]

    if cell_specimen_ids is None:
        cell_specimen_ids = ppf.sort_trace_csids_by_max_in_window(dff_traces, ophys_timestamps, xlim_seconds)[:20]

    if ax is None:
        figsize = (8, 6)
        fig, ax = plt.subplots(len(cell_specimen_ids), 1, figsize=figsize, sharex=True, )
        ax = ax.ravel()

    colors = sns.color_palette()

    for i, cell_specimen_id in enumerate(cell_specimen_ids):
        dff_trace = dff_traces.loc[cell_specimen_id]['dff']
        events_trace = events.loc[cell_specimen_id]['events']
        events_trace[events_trace == 0]
        # limit cell trace to window so yaxes scale properly
        start_ind = np.where(ophys_timestamps < xlim_seconds[0])[0][-1]
        stop_ind = np.where(ophys_timestamps > xlim_seconds[1])[0][0]
        dff_trace = dff_trace[start_ind:stop_ind]
        events_trace = events_trace[start_ind:stop_ind]
        timestamps = ophys_timestamps[start_ind:stop_ind]
        ax[i].plot(timestamps, dff_trace, label=str(cell_specimen_id), color='gray')
        for timepoint in np.where(events_trace != 0)[0]:
            # ax[i].axvline(x=timestamps[timepoint], ymin=0, ymax=events_trace[timepoint], color=colors[6])
            ax[i].annotate('', xy=(timestamps[timepoint], 0), xycoords='data',
                           xytext=(timestamps[timepoint], events_trace[timepoint]), fontsize=8,
                           arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0))
            # ax[i].set_yticks((0, 2))
        ax[i].set_xlim(xlim_seconds)

        # label cell id on right side so it can be cropped out later
        xmin, xmax = ax[i].get_xlim()
        ax[i].text(s='csid:' + str(cell_specimen_id), x=xmax + 0.2, y=0, fontsize=8)

        # add stimuli, despine and such
        ax[i] = ppf.add_stim_color_span(dataset, ax[i], xlim=xlim_seconds, label_changes=label_changes,
                                        label_omissions=label_omissions)

        # plot line on y axis for scale bar
        ymin, ymax = ax[i].get_ylim()
        ax[i].set_yticks([0, np.round(ymax * .3, 2)])
        ax[i].set_yticklabels(['', np.round(ymax * .3, 2)], va='top', ha='right', fontsize=8)
        # ax[i].axvline(x=xlim_seconds[0] - 0.1, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)
        ax[i].annotate('', xy=(xmin, 0), xycoords='data', xytext=(xmin, np.round(ymax * 0.3, 2)), fontsize=8,
                       arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0), clip_on=False)

        sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
        if i < len(cell_specimen_ids) - 1:
            ax[i].set_xticklabels([])
            ax[i].set_xlabel('')
            ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                              labelbottom=False, labeltop=False, labelright=False, labelleft=True, labelsize=7, pad=-1)

        # plot cell ID on left with index
        ax[i].set_ylabel('cell ' + str(i + 1), rotation=0, fontsize=10, ha='right', y=0.4)

    ax[i].set_xticks(np.arange(int(xlim_seconds[0]), int(xlim_seconds[1]), 10))
    ax[i].set_xticklabels(np.arange(int(xlim_seconds[0]), int(xlim_seconds[1]), 10))
    # label bottom row of plot
    ax[i].set_xlabel('Time in session (seconds)')
    ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                      labelbottom=True, labeltop=False, labelright=False, labelleft=True, labelsize=7, pad=-1)

    # add title to top row
    metadata_string = utils.get_metadata_string(dataset.metadata)
    ax[0].set_title(metadata_string)

    plt.subplots_adjust(hspace=0)
    if save_dir:
        print('saving')
        folder = 'example_cell_timeseries_stacked'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)) + '_' + duration_seconds + suffix,
                          formats=['.png'])
    return ax

def plot_single_cell_example_timeseries_and_behavior(dataset, start_time, duration_seconds=20, cell_specimen_ids=None,
                                                use_filtered_events=False, sort_within_expt=True, label_csids=True, fontsize=8,
                                                dff_metrics=None, short_title=False, skip_behavior=False,
                                                save_dir=None, ax=None, suffix=''):
    """
    Plots licking behavior, rewards, running speed, pupil area, and dff traces for a defined window of time.
    Each timeseries gets its own row.
    If no cell_specimen_ids are provided, selects the top 20 cell traces with highest SNR to plot
    if use_filtered_events is True, will plot filtered events trace, if False, will plot dFF
    """

    # xlim_seconds = [start_time - (duration_seconds / 3.), start_time + duration_seconds * 2]
    xlim_seconds = [start_time, start_time + duration_seconds]

    # get cell traces and events
    ophys_timestamps = dataset.ophys_timestamps.copy()
    dff_traces = dataset.dff_traces.copy()
    events = dataset.events.copy()
    events = events.loc[dff_traces.index.values]

    if cell_specimen_ids is None:
        # if no cell ids are provided, get the top 20 highest SNR cells
        cell_specimen_ids = ppf.sort_trace_csids_by_max_in_window(dff_traces, ophys_timestamps, xlim_seconds)[:20]
    else:
        if sort_within_expt:
            # sort provided matched cells by SNR
            # limit dff traces to matched cells
            matched_dff_traces = dff_traces.loc[cell_specimen_ids]
            # get the top 20 highest SNR cells
            cell_specimen_ids = ppf.sort_trace_csids_by_max_in_window(matched_dff_traces, ophys_timestamps, xlim_seconds)
            # print('sorting matched cell IDs by SNR')
            # if cell specimen ids are provided but there are more than 20 of them
        else:
            print('using provided cell_specimen_ids in the order they were provided')
        if len(cell_specimen_ids) > 20:
            # get the top 20 highest SNR cells
            cell_specimen_ids = cell_specimen_ids[:40]


    # get behavior timeseries
    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))
    # licks[:] = -2

    reward_timestamps = dataset.rewards.timestamps.values
    rewards = np.zeros(len(reward_timestamps))
    # rewards[:] = -2

    # get run speed trace and timestamps
    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values
    # limit running trace to window so yaxes scale properly
    start_ind = np.where(running_timestamps < xlim_seconds[0])[0][-1]
    stop_ind = np.where(running_timestamps > xlim_seconds[1])[0][0]
    running_speed = running_speed[start_ind:stop_ind]
    running_timestamps = running_timestamps[start_ind:stop_ind]

    # get pupil width trace and timestamps
    eye_tracking = dataset.eye_tracking.copy()
    pupil_diameter = eye_tracking.pupil_width.values
    pupil_diameter[eye_tracking.likely_blink == True] = np.nan
    pupil_timestamps = eye_tracking.timestamps.values
    # smooth pupil diameter
    from scipy.signal import medfilt
    pupil_diameter = medfilt(pupil_diameter, kernel_size=5)
    # limit pupil trace to window so yaxes scale properly
    start_ind = np.where(pupil_timestamps < xlim_seconds[0])[0][-1]
    stop_ind = np.where(pupil_timestamps > xlim_seconds[1])[0][0]
    pupil_diameter = pupil_diameter[start_ind:stop_ind]
    pupil_timestamps = pupil_timestamps[start_ind:stop_ind]

    # make the plot
    if ax is None:
        figsize = (8, 6)
        if skip_behavior:
            n_rows = len(cell_specimen_ids)
        else:
            n_rows = len(cell_specimen_ids)+3
        fig, ax = plt.subplots(n_rows, 1, figsize=figsize, sharex=True, )
        ax = ax.ravel()

    colors = sns.color_palette()

    for i, cell_specimen_id in enumerate(cell_specimen_ids):
        if use_filtered_events:
            trace = events.loc[cell_specimen_id]['filtered_events']
        else:
            trace = dff_traces.loc[cell_specimen_id]['dff']
        events_trace = events.loc[cell_specimen_id]['events']
        events_trace[events_trace == 0]
        # limit cell trace to window so yaxes scale properly
        start_ind = np.where(ophys_timestamps < xlim_seconds[0])[0][-1]
        stop_ind = np.where(ophys_timestamps > xlim_seconds[1])[0][0]
        trace = trace[start_ind:stop_ind]
        events_trace = events_trace[start_ind:stop_ind]
        timestamps = ophys_timestamps[start_ind:stop_ind]
        ax[i].plot(timestamps, trace, label=str(cell_specimen_id), color='gray')
        for timepoint in np.where(events_trace != 0)[0]:
            # ax[i].axvline(x=timestamps[timepoint], ymin=0, ymax=events_trace[timepoint], color=colors[6])
            ax[i].annotate('', xy=(timestamps[timepoint], 0), xycoords='data',
                           xytext=(timestamps[timepoint], events_trace[timepoint]), fontsize=fontsize,
                           arrowprops=dict(arrowstyle='-', color='k', #sns.color_palette()[0],
                                           lw=1, shrinkA=0, shrinkB=0))
            # ax[i].set_yticks((0, 2))
        ax[i].set_xlim(xlim_seconds)

        # label cell id on right side so it can be cropped out later
        xmin, xmax = ax[i].get_xlim()
        if label_csids:
            ax[i].text(s='csid:' + str(cell_specimen_id), x=xmax + 0.2, y=0, fontsize=6)

        # get ylims of data in this window
        ymin, ymax = ax[i].get_ylim()
        if dff_metrics is not None: # if metrics are provided, set the ymax value to the relevant trace metric
            metric = '99_pct'
            metric_value = dff_metrics.loc[cell_specimen_id, metric]
            ax[i].set_ylim(ymin, metric_value*1.5)
        # get it again to determine where the scale bar will go
        ymin, ymax = ax[i].get_ylim()
        ax[i].set_yticks([0, np.round(ymax * .3, 2)])
        ax[i].set_yticklabels(['', np.round(ymax * .3, 2)], va='top', ha='right', fontsize=fontsize-1)
        # ax[i].axvline(x=xlim_seconds[0] - 0.1, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)
        # add scale bar
        scalex = (duration_seconds*0.01)
        ax[i].annotate('', xy=(xmin-scalex, 0), xycoords='data', xytext=(xmin-scalex, np.round(ymax * 0.3, 2)), fontsize=fontsize-2,
                       arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0), annotation_clip=False)

        sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
        # if i < len(cell_specimen_ids) - 1:
        ax[i].set_xticklabels([])
        ax[i].set_xlabel('')
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True)

        # plot cell ID on left with index
        ax[i].set_ylabel('cell ' + str(i + 1), rotation=0, fontsize=fontsize, ha='center', y=0.4)

        # add stimuli, despine and such
        if i == 0:
            annotate_changes = True
        else:
            annotate_changes = False
        ax[i] = ppf.add_stim_color_span(dataset, ax[i], xlim=xlim_seconds, annotate_changes=annotate_changes,
                                        label_changes=True, label_omissions=True)

    if not skip_behavior:
        # plot behavior timeseries
        colors = sns.color_palette()

        ax[i+2].plot(running_timestamps, running_speed, label='running_speed', color='gray')
        ax[i+2].set_ylabel('running\nspeed\n(cm/s)', rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
        ax[i+2].set_ylim(ymin=-8)
        # ax[i+2].tick_params(which='both', bottom=False, top=False, right=False, left=False,
        #                   labelbottom=False, labeltop=False, labelright=False, labelleft=True, size=fontsize)

        ax[i+3].plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color='gray')
        ax[i+3].set_ylabel('pupil\ndiameter\n(pixels)', rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
        # ax[i+3].tick_params(which='both', bottom=False, top=False, right=False, left=False,
        #                   labelbottom=False, labeltop=False, labelright=False, labelleft=True, size=fontsize)

        ax[i+1].plot(lick_timestamps, licks, '|', label='licks', color='gray', markersize=10)
        ax[i+1].plot(reward_timestamps, rewards, 'o', label='rewards', color='cyan', markersize=5)
        ax[i+1].set_ylabel('licks &\nrewards', rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
        ax[i+1].set_yticklabels([])
        ax[i+1].set_ylim(-1,3)

        # add stimuli, despine and such
        for x in range(i+1, i+4):
            ax[x] = ppf.add_stim_color_span(dataset, ax[x], xlim=xlim_seconds, label_changes=True, label_omissions=True)
            sns.despine(ax=ax[x], top=True, right=True, left=True, bottom=True)
            ax[x].set_xticklabels([])
            ax[x].set_xlabel('')
            ax[x].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                              labelbottom=False, labeltop=False, labelright=False, labelleft=True, labelsize=fontsize)
        i = i+3

    # Label bottom row with times
    ax[i].set_xticks(np.arange(int(xlim_seconds[0]), int(xlim_seconds[1])+1, 5))
    ax[i].set_xticklabels(np.arange(int(xlim_seconds[0]), int(xlim_seconds[1])+1, 5), fontsize=fontsize)
    # label bottom row of plot
    ax[i].set_xlabel('Time in session (seconds)', fontsize=fontsize+2)
    ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                      labelbottom=True, labeltop=False, labelright=False, labelleft=True, labelsize=fontsize)

    # add title to top row
    # ophys_container_id = dataset.metadata['ophys_container_id']
    # metadata_string = str(ophys_container_id)+'_'+utils.get_metadata_string(dataset.metadata) +'_'+ str(int(start_time)) +'_'+ str(duration_seconds)
    m = dataset.metadata.copy()
    metadata_string = str(m['ophys_container_id']) + '_' + str(m['mouse_id']) + '_' + m['cre_line'].split('-')[
        0] + '_' + str(m['targeted_structure']) + '_' + str(m['imaging_depth']) + '_' + m['session_type']  +'_'+ str(int(start_time)) +'_'+ str(duration_seconds)

    if short_title:
        metadata_short = str(m['ophys_container_id']) + '_' + m['session_type']  +'_'+ str(int(start_time)) +'_'+ str(duration_seconds)
        # ax[0].set_title(metadata_short, fontsize=6, va='top', )
        ax[0].set_title('')
    else:
        plt.suptitle(metadata_string, x=0.5, y=0.97, fontsize=14)
    plt.subplots_adjust(hspace=0)
    if save_dir:
        print('saving')
        folder = 'example_cell_timeseries_and_behavior_stacked'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + suffix)
    return ax


def add_metrics_to_dff_traces(dff_traces):
    import visual_behavior.data_access.processing as processing

    dff_traces = processing.compute_robust_snr_on_dataframe(dff_traces)
    dff_traces['99_pct'] = [np.round(np.percentile(trace, 99), 2) for trace in dff_traces.dff.values]
    dff_traces['95_pct'] = [np.round(np.percentile(trace, 95), 2) for trace in dff_traces.dff.values]
    dff_traces['max'] = [np.round(np.max(trace), 2) for trace in dff_traces.dff.values]
    dff_traces = dff_traces.reset_index()

    return dff_traces


def get_high_snr_matched_cells_for_container(ophys_container_id, dataset_dict,
                                             matched_cells_table, platform_experiments, xlim_seconds=[500, 2500]):
    '''
    Loop through each expt in the container, get the highest SNR dFF traces,
    filter out any that arent matched across sessions, then return the list of matched high SNR cells

    Parameters
    ----------
    ophys_container_id
    dataset_dict
    matched_cells_table
    platform_experiments
    xlim_seconds

    Returns
    -------

    '''
    matched_cells = matched_cells_table[matched_cells_table.ophys_container_id==ophys_container_id].cell_specimen_id.unique()

    matched_high_snr_cells = []
    ophys_experiment_ids = platform_experiments[platform_experiments.ophys_container_id==ophys_container_id].sort_values(by='experience_level').index.values
    all_dff_traces = pd.DataFrame()
    for i,ophys_experiment_id in enumerate(ophys_experiment_ids):
        dataset = dataset_dict[ophys_container_id][ophys_experiment_id]
        ophys_timestamps = dataset.ophys_timestamps.copy()
        dff_traces = dataset.dff_traces.copy()
        dff_traces['ophys_experiment_id'] = ophys_experiment_id
        tmp = add_metrics_to_dff_traces(dff_traces)
        all_dff_traces = pd.concat([all_dff_traces, tmp.drop(columns='dff')])
        cell_specimen_ids = ppf.sort_trace_csids_by_max_in_window(dff_traces, ophys_timestamps, xlim_seconds)
        matched_cell_ids = [cell_id for cell_id in cell_specimen_ids if cell_id in matched_cells ]
        if len(matched_cell_ids)>10:
            matched_cell_ids = matched_cell_ids[:10]
        matched_high_snr_cells = matched_high_snr_cells + matched_cell_ids
    matched_high_snr_cells = np.unique(matched_high_snr_cells)

    # now get the max value of each metric for each cell across sessions
    all_dff_traces = all_dff_traces.reset_index()
    # all_dff_traces = all_dff_traces[all_dff_traces.cell_specimen_id.isin(matched_high_snr_cells)]
    dff_metrics = all_dff_traces.groupby('cell_specimen_id').max()
    # dff_metrics = dff_metrics.drop(columns='dff')

    return matched_high_snr_cells, dff_metrics



###### FOV image plots  #########

def plot_matched_max_projections_for_container(ophys_container_id, platform_experiments,
                                               dataset_dict, save_dir=None, ax=None):
    '''
    Loop through platform ophys experiments belonging to this container
    and plot the max projection image for each experiment with experience level in title
    platform_experiments is the ophys_experiment_table limited to the experiments for the platform paper (one F, one N and one N+ per FOV)
    dataset_dict is a dictionary with keys ophys_container_id, ophys_experiment_id
    containing the dataset object for each experiment within a container
    '''
    experience_level_colors = utils.get_experience_level_colors()
    ophys_experiment_ids = platform_experiments[platform_experiments.ophys_container_id==ophys_container_id].sort_values(by='experience_level').index.values
    if ax is None:
        figsize = (10, 3)
        fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i,ophys_experiment_id in enumerate(ophys_experiment_ids):
        dataset = dataset_dict[ophys_container_id][ophys_experiment_id]
        experience_level = platform_experiments.loc[ophys_experiment_id].experience_level
        ax[i].imshow(dataset.max_projection, cmap='gray', vmin=0, vmax=np.percentile(dataset.max_projection, 99))
        ax[i].axis('off')
        session_type = platform_experiments.loc[ophys_experiment_id].session_type
        ax[i].set_title(str(ophys_experiment_id), color=experience_level_colors[i], fontsize=12)
        # ax[i].set_title(str(ophys_experiment_id)+'\n'+session_type+'\n'+experience_level,
        #                 color=experience_level_colors[i], fontsize=12)
        ax[i].set_aspect('equal')

    if save_dir:
        metadata = utils.get_metadata_string(dataset.metadata)
        fig.suptitle(str(ophys_container_id) + '_' + metadata, x=0.5, y=1.1)
        filename = str(ophys_container_id) +'_'+ metadata
        folder = 'matched_roi_images'
        utils.save_figure(fig, figsize, save_dir, folder, filename)
    return ax


def plot_max_intensity_projection(dataset, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    max_projection = dataset.max_projection.data
    ax.imshow(max_projection, cmap='gray', vmax=np.percentile(max_projection, 99))
    ax.axis('off')
    return ax

def plot_roi_mask_outlines(dataset, cell_specimen_ids=None, include_max_projection=True,
                           roi_color='red', label_rois=True, label_color='yellow', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if include_max_projection:
        ax = plot_max_intensity_projection(dataset, ax=ax)
    cell_specimen_table = dataset.cell_specimen_table.copy()
    if cell_specimen_ids is None:
        cell_specimen_ids = cell_specimen_table.index.values
    if len(cell_specimen_table) > 0:
        for i, cell_specimen_id in enumerate(cell_specimen_ids):
            mask_data = cell_specimen_table.loc[cell_specimen_id]
            ax.contour(mask_data.roi_mask, levels=0, colors=roi_color, linewidths=[0.5])
            if label_rois:
                ax.text(s=str(i+1), x=mask_data.x, y=mask_data.y,
                        ha='right', va='bottom', fontsize=12, color=label_color)
    sns.despine(ax=ax, top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    ax.tick_params(which='both', bottom=False, top=False, right=False, left=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

    #ax.axis('off')
    return ax


def plot_max_and_roi_outlines_for_container(ophys_container_id, platform_experiments, dataset_dict,
                                               cell_specimen_ids=None, label_rois=True, save_dir=None, ax=None):
    '''
    Loop through platform ophys experiments belonging to this container
    and plot the max projection image for each experiment with experience level in title
    platform_experiments is the ophys_experiment_table limited to the experiments for the platform paper (one F, one N and one N+ per FOV)
    dataset_dict is a dictionary with keys ophys_container_id, ophys_experiment_id
    containing the dataset object for each experiment within a container
    cell_specimen_ids is a list of cells within the container to plot roi masks for
    '''
    experience_level_colors = utils.get_experience_level_colors()
    ophys_experiment_ids = platform_experiments[platform_experiments.ophys_container_id==ophys_container_id].sort_values(by='experience_level').index.values
    if ax is None:
        figsize = (10, 3)
        fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i,ophys_experiment_id in enumerate(ophys_experiment_ids):
        dataset = dataset_dict[ophys_container_id][ophys_experiment_id]
        if cell_specimen_ids is None:
            cells_to_plot = dataset.dff_traces.index.values
            suffix = '_all'
        else:
            cells_to_plot = cell_specimen_ids
            suffix = '_matched'
        experience_level = platform_experiments.loc[ophys_experiment_id].experience_level
        ax[i] = plot_roi_mask_outlines(dataset, cell_specimen_ids=cells_to_plot, label_rois=label_rois,
                                       include_max_projection=False, roi_color='red', ax=ax[i])
        ax[i].set_title(experience_level, color=experience_level_colors[i], fontsize=14)
    if save_dir:
        metadata = utils.get_metadata_string(dataset.metadata)
        fig.suptitle(str(ophys_container_id) + '_' + metadata, x=0.5, y=1.1)
        filename = str(ophys_container_id) +'_'+ metadata + '_outlines' + suffix
        folder = 'matched_roi_images'
        utils.save_figure(fig, figsize, save_dir, folder, filename)
    return ax



def plot_matched_roi_outlines_for_container(ophys_container_id, platform_experiments, matched_cells_table,
                                            dataset_dict, cells_to_label=None, save_dir=None, ax=None):
    '''
    Loop through platform ophys experiments belonging to this container
    and plot the ROI masks for each experiment with experience level in title,
    with matched ROIs in red and unmatched in yellow
    Only label the cell IDs for cells_to_label
    platform_experiments is the ophys_experiment_table limited to the experiments for the platform paper (one F, one N and one N+ per FOV)
    matched_cells_table is the ophys_cells_table limited to the ROIs matched across sessions (for the sessions in platform_experiments)
    dataset_dict is a dictionary with keys ophys_container_id, ophys_experiment_id
    containing the dataset object for each experiment within a container
    cell_specimen_ids is a list of cells within the container to plot roi masks for
    '''
    matched_cells = matched_cells_table[matched_cells_table.ophys_container_id==ophys_container_id].cell_specimen_id.unique()
    experience_level_colors = utils.get_experience_level_colors()
    ophys_experiment_ids = platform_experiments[platform_experiments.ophys_container_id==ophys_container_id].sort_values(by='experience_level').index.values
    if ax is None:
        figsize = (15, 3)
        fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i,ophys_experiment_id in enumerate(ophys_experiment_ids):
        dataset = dataset_dict[ophys_container_id][ophys_experiment_id]
        cell_specimen_table = dataset.cell_specimen_table.copy()
        all_cells = cell_specimen_table.index.values
        unmatched_cells = all_cells[all_cells!=matched_cells][0]
        experience_level = platform_experiments.loc[ophys_experiment_id].experience_level
        ax[i] = plot_roi_mask_outlines(dataset, matched_cells, include_max_projection=False, roi_color='k', label_rois=False, label_color='w', ax=ax[i])
        ax[i] = plot_roi_mask_outlines(dataset, unmatched_cells, include_max_projection=False, roi_color='gray', label_rois=False, label_color='w', ax=ax[i])
        if cells_to_label is not None:
            ax[i] = plot_roi_mask_outlines(dataset, cells_to_label, include_max_projection=False, roi_color='red', label_rois=True, label_color='k', ax=ax[i])
        ax[i].set_title(experience_level, color=experience_level_colors[i], fontsize=14)
    if save_dir:
        metadata = utils.get_metadata_string(dataset.metadata)
        fig.suptitle(str(ophys_container_id) + '_' + metadata, x=0.5, y=1.1)
        filename = str(ophys_container_id) +'_'+ metadata + '_outlines'
        folder = 'matched_roi_mask_outlines'
        utils.save_figure(fig, figsize, save_dir, folder, filename)
    return ax

####### multi panel plot for matched FOVs ########


def plot_matched_rois_and_traces_for_container(ophys_container_id, dataset_dict,
                                        matched_cells_table, platform_experiments, fontsize=12,
                                        start_times=np.arange(1500, 1600, 40), duration_seconds=20,
                                        matched_cells=None, save_dir=None, suffix=''):
    '''
    dataset_dict: dictionary containing ophys_container_ids as top level keys,
        with ophys_experiment_ids as second level keys, and dataset obejcts for those experiments as values
    ophys_container_id: key within dataset_dict to use for plots
    matched_cells_table: ophys_cells_table instance filtered to limit to matched cells across sessions
    platform_experiments: ophys_epxeriment_table instance limited to experiments in VB platform paper
    start_times: array of time values, in seconds, to use for start of each timeseries plot axis
    duration_seconds: duration of timeseries to plot
    matched_reliable_cells: list of cell_specimen_ids from the provided ophys_container_id,
        sorted in order of reliability. reliability can be computed based on change or omission multi session df
    change_mdf: multi session mean response df for changes
    omission_mf: multi session mean response df for omissions
    cell_type: ['Excitatory', 'Vip', 'Sst']
    '''

    matched_cells_snr, dff_metrics = get_high_snr_matched_cells_for_container(ophys_container_id, dataset_dict,
                                                                            matched_cells_table,
                                                                            platform_experiments,
                                                                            xlim_seconds=[500, 2500])
    if matched_cells is None:  # if matched cell ids  not provided, get high SNR cells
        matched_cells = matched_cells_snr
        suffix = '_high_snr_cells'

    if len(matched_cells) > 20:
        matched_cells = matched_cells[:20]

    for s, start_time in enumerate(start_times):
        print(np.where(start_times == start_time)[0][0], 'out of', len(start_times))

        figsize = [20, 6]
        fig = plt.figure(figsize=figsize, facecolor='white')

        # plot max projections
        ax = utils.placeAxesOnGrid(fig, dim=(1, 3), xspan=(0, 0.3), yspan=(0, 0.4), wspace=0.2)
        ax = plot_matched_max_projections_for_container(ophys_container_id, platform_experiments,
                                                            dataset_dict, ax=ax)

        # plot max projections with ROI masks
        ax = utils.placeAxesOnGrid(fig, dim=(1, 3), xspan=(0.4, 0.7), yspan=(0, 0.4), wspace=0.2)
        ax = plot_matched_roi_outlines_for_container(ophys_container_id, platform_experiments, matched_cells_table,
                                                dataset_dict, cells_to_label=matched_cells, save_dir=None, ax=ax)


        expt_ids = list(dataset_dict[ophys_container_id].keys())
        for i, ophys_experiment_id in enumerate(expt_ids):
            dataset = dataset_dict[ophys_container_id][ophys_experiment_id]

            # plot timeseries
            x = i * 0.25
            ax = utils.placeAxesOnGrid(fig, dim=(len(matched_cells), 1),
                                           xspan=(x, x + 0.22), yspan=(0.5, 1), sharex=True, wspace=0.5)
            if i == 2:
                label_csids = True
            else:
                label_csids = False
            ax = plot_single_cell_example_timeseries_and_behavior(dataset, start_time=start_time,
                                                                      duration_seconds=duration_seconds,
                                                                      cell_specimen_ids=matched_cells, save_dir=False,
                                                                      sort_within_expt=False, dff_metrics=dff_metrics,
                                                                      label_csids=label_csids, short_title=True,
                                                                      fontsize=fontsize,
                                                                      skip_behavior=True, ax=ax, suffix='')

        if save_dir:
            print('saving')
            folder = 'example_rois_and_traces_for_container'
            m = dataset.metadata.copy()
            metadata_string = utils.get_container_metadata_string(m) + '_' + str(int(start_time)) + '_' + str(duration_seconds)
            plt.suptitle(metadata_string, x=0.4, y=0.98, fontsize=16)
            utils.save_figure(fig, figsize, save_dir, folder, metadata_string + suffix)


def plot_matched_traces_across_sessions(ophys_container_id, dataset_dict,
                                        matched_cells_table, platform_experiments,
                                        change_mdf, omission_mdf, cell_type, skip_behavior=False, fontsize=12,
                                        start_times=np.arange(1500, 1600, 40), duration_seconds=20,
                                        matched_cells=None, save_dir=None, suffix=''):
    '''
    dataset_dict: dictionary containing ophys_container_ids as top level keys,
        with ophys_experiment_ids as second level keys, and dataset obejcts for those experiments as values
    ophys_container_id: key within dataset_dict to use for plots
    matched_cells_table: ophys_cells_table instance filtered to limit to matched cells across sessions
    platform_experiments: ophys_epxeriment_table instance limited to experiments in VB platform paper
    start_times: array of time values, in seconds, to use for start of each timeseries plot axis
    duration_seconds: duration of timeseries to plot
    matched_reliable_cells: list of cell_specimen_ids from the provided ophys_container_id,
        sorted in order of reliability. reliability can be computed based on change or omission multi session df
    change_mdf: multi session mean response df for changes
    omission_mf: multi session mean response df for omissions
    cell_type: ['Excitatory', 'Vip', 'Sst']
    '''

    matched_cells_snr, dff_metrics = get_high_snr_matched_cells_for_container(ophys_container_id, dataset_dict,
                                                                            matched_cells_table,
                                                                            platform_experiments,
                                                                            xlim_seconds=[500, 2500])
    if matched_cells is None:  # if matched cell ids  not provided, get high SNR cells
        matched_cells = matched_cells_snr
        suffix = '_high_snr_cells'

    if len(matched_cells) > 20:
        matched_cells = matched_cells[:20]

    for s, start_time in enumerate(start_times):
        print(np.where(start_times == start_time)[0][0], 'out of', len(start_times))

        figsize = [20, 10]
        fig = plt.figure(figsize=figsize, facecolor='white')

        # plot max projections
        ax = utils.placeAxesOnGrid(fig, dim=(1, 3), xspan=(0, 0.3), yspan=(0, 0.2), wspace=0.2)
        ax = plot_matched_max_projections_for_container(ophys_container_id, platform_experiments,
                                                            dataset_dict, ax=ax)

        # plot max projections with ROI masks
        ax = utils.placeAxesOnGrid(fig, dim=(1, 3), xspan=(0, 0.3), yspan=(0.25, 0.45), wspace=0.2)
        ax = plot_max_and_roi_outlines_for_container(ophys_container_id, platform_experiments, dataset_dict,
                                                         cell_specimen_ids=matched_cells, ax=ax)
        # plot change responses for matched cells
        ax = utils.placeAxesOnGrid(fig, dim=(len(matched_cells) + 2, 4), xspan=(0.38, 0.53), yspan=(0, 0.45),
                                   sharey=True)
        ax = plot_reliable_example_cells_container(change_mdf, matched_cells, cell_type,
                                                       event_type='changes', label_csids=False, linewidth=1, ax=ax)

        # plot max projections with ROI masks
        ax = utils.placeAxesOnGrid(fig, dim=(len(matched_cells) + 2, 4), xspan=(0.55, 0.7), yspan=(0, 0.45),
                                   sharey=True)
        ax = plot_reliable_example_cells_container(omission_mdf, matched_cells, cell_type,
                                                       event_type='omissions', label_csids=False, linewidth=1, ax=ax)

        expt_ids = list(dataset_dict[ophys_container_id].keys())
        for i, ophys_experiment_id in enumerate(expt_ids):
            dataset = dataset_dict[ophys_container_id][ophys_experiment_id]

            # plot timeseries
            x = i * 0.24
            if skip_behavior:
                ax = utils.placeAxesOnGrid(fig, dim=(len(matched_cells), 1),
                                       xspan=(x, x + 0.18), yspan=(0.5, 0.8), sharex=True, wspace=0.5)
            else:
                ax = utils.placeAxesOnGrid(fig, dim=(len(matched_cells) + 3, 1),
                                           xspan=(x, x + 0.18), yspan=(0.5, 1), sharex=True, wspace=0.5)
            if i == 2:
                label_csids = True
            else:
                label_csids = False
            ax = plot_single_cell_example_timeseries_and_behavior(dataset, start_time=start_time,
                                                                      duration_seconds=duration_seconds,
                                                                      cell_specimen_ids=matched_cells, save_dir=False,
                                                                      sort_within_expt=False, dff_metrics=dff_metrics,
                                                                      label_csids=label_csids, short_title=True,
                                                                      fontsize=fontsize,
                                                                      skip_behavior=skip_behavior, ax=ax, suffix='')

        if save_dir:
            print('saving')
            folder = 'example_cell_timeseries_and_behavior_joint'
            m = dataset.metadata.copy()
            metadata_string = str(m['ophys_container_id']) + '_' + str(m['mouse_id']) + '_' + m['cre_line'].split('-')[
                0] + '_' + str(m['targeted_structure']) + '_' + str(m['imaging_depth']) + '_' + str(
                int(start_time)) + '_' + str(duration_seconds)
            plt.suptitle(metadata_string, x=0.5, y=0.98)
            utils.save_figure(fig, figsize, save_dir, folder, metadata_string + suffix)


####### GLM example figures #######

def load_GLM_outputs(glm_version, experiments_table, cells_table, glm_output_dir=None):
    """
    loads results_pivoted and weights_df from files in base_dir, or generates them from mongo and save to base_dir
    results_pivoted and weights_df will be limited to the ophys_experiment_ids and cell_specimen_ids present in experiments_table and cells_table
    because this function simply loads the results, any pre-processing applied to results_pivoted (such as across session normalization or signed weights) will be used here

    :param glm_version: example = '24_events_all_L2_optimize_by_session'
    :param glm_output_dir: directory containing GLM output files to load and save processed data files to for this iteration of the analysis
                            if None, GLM results will be obtained from mongo and will not be saved out
    :param experiments_table: SDK ophys_experiment table limited to experiments intended for analysis
    :param cells_table: SDK ophys_cell_table limited to cell_specimen_ids intended for analysis
    :return:
        results_pivoted: table of dropout scores for all cell_specimen_ids in cells_table
        weights_df: table with model weights for all cell_specimen_ids in cells_table
    """
    # get GLM kernels and params for this version of the model
    run_params = glm_params.load_run_json(glm_version)
    kernels = run_params['kernels']
    # if glm_output_dir is not None:
    # load GLM results for all cells and sessions from file if it exists otherwise load from mongo
    glm_results_path = os.path.join(glm_output_dir, glm_version + '_results_pivoted.h5')
    if os.path.exists(glm_results_path):
        results_pivoted = pd.read_hdf(glm_results_path, key='df')
    else:
        print('no results_pivoted at', glm_results_path)
        print('please generate before running single cell plots')

    weights_path = os.path.join(glm_output_dir, glm_version + '_weights_df.h5')
    if os.path.exists(weights_path):  # if it exists, load it
        weights_df = pd.read_hdf(weights_path, key='df')
    else:
        print('no weights at', weights_path)
        print('please generate before running single cell plots')

    return results_pivoted, weights_df, kernels


def get_time_window_for_kernel(kernels, feature):
    time_window = (kernels[feature]['offset'], kernels[feature]['offset'] + kernels[feature]['length'])
    return time_window


def get_t_array_for_kernel(kernels, feature, frame_rate):
    time_window = get_time_window_for_kernel(kernels, feature)
    t_array = ms_utils.get_time_array(t_start=time_window[0], t_end=time_window[1], sampling_rate=frame_rate,
                                      include_endpoint=False)
    if 'image' in feature:
        t_array = t_array[:-1]  # not sure why we have to do this
    return t_array


def plot_cell_rois_and_GLM_weights(cell_specimen_id, cells_table, experiments_table, dropout_features, results_pivoted, weights_df,
                                   weights_features, kernels, save_dir=None, folder=None, data_type='dff'):
    """
    This function limits inputs just to the provided cell_specimen_id to hand off to the function plot_matched_roi_and_traces_example_GLM
    That function will plot the following panels:
        cell ROI masks matched across sessions for a given cell_specimen_id,
        change and omission triggered average respones across sessions,
        image locked running and pupil if included 'running' and 'pupil' in included weights_features
        dropout scores across dropout_features and sessions as a heatmap,
        kernels weights across sessions for the kernels in weights_features
    :param cell_specimen_id: cell_specimen_id for cell to plot
    :param cells_table: must only include a max of one one experiment per experience level for a given container. ok if less than 1.
    :param experiments_table: must only include one experiment per experience level for a given container
    :param results_pivoted: must only include one experiment per experience level for a given container
                            must be limited to features for plotting, plus cell_specimen_id + experiment_id
    :param weights_df: must only include one experiment per experience level for a given container
    :param weights_features: columns in weights_df to use for plotting
    :param save_dir: top level directory where files exist for this run of analysis
                        code will create a folder within save_dir called 'matched_cell_examples'
    :param subfolder: name of subfolder to create within os.path.join(save_dir, 'matched_cell_examples') to save plots, ex: 'cluster_0' or 'without_exp_var_full_model' or 'with_running_and_pupil'
    :param data_type: can be 'dff', 'events', or 'filtered_events' - to be used for cell response plots
    :return:
    """

    # make sure weights and dropouts are limited to matched experiments / cells
    cells_table = loading.get_cell_table(platform_paper_only=True, limit_to_closest_active=True,
                                         limit_to_matched_cells=True, add_extra_columns=True)
    experiments_table = loading.get_platform_paper_experiment_table(add_extra_columns=True, limit_to_closest_active=True)
    matched_cells = cells_table.cell_specimen_id.unique()
    matched_experiments = cells_table.ophys_experiment_id.unique()
    weights_df = weights_df[weights_df.ophys_experiment_id.isin(matched_experiments)]
    weights_df = weights_df[weights_df.cell_specimen_id.isin(matched_cells)]
    results_pivoted = results_pivoted.reset_index()  # reset just in case
    results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(matched_cells)]

    # get cell info
    cell_metadata = cells_table[cells_table.cell_specimen_id == cell_specimen_id]

    # get metadata for this cell
    cell_metadata = cells_table[cells_table.cell_specimen_id == cell_specimen_id]
    # get weights for example cell
    cell_weights = weights_df[weights_df.cell_specimen_id == cell_specimen_id]
    # if exp var full model is in features (must be first feature), scale it by 10x so its on similar scale as dropouts
    if 'variance_explained_full' in results_pivoted.keys():
        results_pivoted['variance_explained_full'] = results_pivoted['variance_explained_full'] * 10
    # get dropouts just for one cell
    cell_dropouts = results_pivoted[results_pivoted.cell_specimen_id == cell_specimen_id]

    plot_matched_roi_and_traces_example_GLM(cell_metadata, cell_dropouts, cell_weights, weights_features, kernels,
                                            dropout_features, experiments_table, data_type, save_dir, folder)


def plot_matched_roi_and_traces_example_GLM(cell_metadata, cell_dropouts, cell_weights, weights_features, kernels,
                                            dropout_features, experiments_table, data_type, save_dir=None, folder=None):
    """
    This function will plot the following panels:
        cell ROI masks matched across sessions for a given cell_specimen_id,
        change and omission triggered average respones across sessions,
        image locked running and pupil if included 'running' and 'pupil' in included weights_features,
        dropout scores across features and sessions as a heatmap,
        kernels weights across sessions for the kernels in weights_features.
    Plots the ROI masks and cell traces for a cell matched across sessions, along with dropout scores and weights for images, hits, misses and omissions
    Cell_metadata is a subset of the ophys_cells_table limited to the cell_specimen_id of interest
    cell_dropouts is a subset of the results_pivoted version of GLM output limited to cell_specimen_id of interest
    cell_weights is a subset of the weights matrix from GLM limited to cell_specimen_id of interest
    all input dataframes must be limited to last familiar and second novel active (i.e. max of one session per type)
    if one session type is missing, the max projection but no ROI will be plotted and the traces and weights will be missing for that experience level
    """

    if len(cell_metadata.cell_specimen_id.unique()) > 1:
        print('There is more than one cell_specimen_id in the provided cell_metadata table')
        print('Please limit input to a single cell_specimen_id')

    # set up plotting for each experience level
    experience_levels = ['Familiar', 'Novel 1', 'Novel >1']
    colors = utils.get_experience_level_colors()
    n_exp_levels = len(experience_levels)
    # get relevant info for this cell
    cell_metadata = cell_metadata.sort_values(by='experience_level')
    cell_specimen_id = cell_metadata.cell_specimen_id.unique()[0]
    ophys_container_id = cell_metadata.ophys_container_id.unique()[0]
    # need to get all experiments for this container, not just for this cell
    ophys_experiment_ids = experiments_table[experiments_table.ophys_container_id == ophys_container_id].index.values
    n_expts = len(ophys_experiment_ids)
    if n_expts > 3:
        print('There are more than 3 experiments for this cell. There should be a max of 1 experiment per experience level')
        print('Please limit input to only one experiment per experience level')

    # set up labels for different trace types
    if data_type == 'dff':
        ylabel = 'dF/F'
    else:
        ylabel = 'response'

    # number of columns is one for each experience level,
    # plus additional columns for stimulus and omission traces, and running and pupil averages (TBD)
    extra_cols = 2
    if 'running' in weights_features:
        extra_cols += 1
    if 'running' in weights_features:
        extra_cols += 1
    n_cols = n_exp_levels + extra_cols
    print(extra_cols, 'extra cols')

    figsize = (3.5 * n_cols, 6)
    fig, ax = plt.subplots(2, n_cols, figsize=figsize)
    ax = ax.ravel()

    print('cell_specimen_id:', cell_specimen_id)
    # loop through experience levels for this cell
    for e, experience_level in enumerate(experience_levels):
        print('experience_level:', experience_level)

        # get ophys_experiment_id for this experience level
        # experiments_table must only include one experiment per experience level for a given container
        ophys_experiment_id = experiments_table[(experiments_table.ophys_container_id == ophys_container_id) &
                                                (experiments_table.experience_level == experience_level)].index.values[0]
        print('ophys_experiment_id:', ophys_experiment_id)
        ind = experience_levels.index(experience_level)
        color = colors[ind]

        # load dataset for this experiment
        dataset = loading.get_ophys_dataset(ophys_experiment_id, get_extended_stimulus_presentations=False)

        try:  # attempt to generate plots for this cell in this this experience level. if cell does not have this exp level, skip
            # plot ROI mask for this experiment
            ct = dataset.cell_specimen_table.copy()
            cell_roi_id = ct.loc[cell_specimen_id].cell_roi_id  # typically will fail here if the cell_specimen_id isnt in the session
            roi_masks = dataset.roi_masks.copy()  # save this to get approx ROI position if subsequent session is missing the ROI (fails if the first session is the one missing the ROI)
            ax[e] = sf.plot_cell_zoom(dataset.roi_masks, dataset.max_projection, cell_roi_id,
                                      spacex=50, spacey=50, show_mask=True, ax=ax[e])
            ax[e].set_title(experience_level, color=color)

            # get change responses and plot on second to next axis after ROIs (there are n_expts # of ROIs)
            window = [-1, 1.5]  # window around event
            sdf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True, output_sampling_rate=30,
                                                   data_type=data_type, event_type='changes', load_from_file=True)
            cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.is_change == True)]

            ax[n_expts] = utils.plot_mean_trace(cell_data.trace.values, cell_data.trace_timestamps.values[0],
                                                ylabel=ylabel, legend_label=None, color=color, interval_sec=1,
                                                xlim_seconds=window, plot_sem=True, ax=ax[n_expts])
            ax[n_expts] = utils.plot_flashes_on_trace(ax[n_expts], cell_data.trace_timestamps.values[0],
                                                      change=True, omitted=False)
            ax[n_expts].set_title('changes')

            # get omission responses and plot on last axis
            sdf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True, output_sampling_rate=30,
                                                   data_type=data_type, event_type='omissions', load_from_file=True)
            cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.omitted == True)]

            ax[n_expts + 1] = utils.plot_mean_trace(cell_data.trace.values, cell_data.trace_timestamps.values[0],
                                                    ylabel=ylabel, legend_label=None, color=color, interval_sec=1,
                                                    xlim_seconds=window, plot_sem=True, ax=ax[n_expts + 1])
            ax[n_expts + 1] = utils.plot_flashes_on_trace(ax[n_expts + 1], cell_data.trace_timestamps.values[0],
                                                          change=False, omitted=True)
            ax[n_expts + 1].set_title('omissions')

            if 'running' in weights_features:
                pass
            if 'pupil' in weights_features:
                pass

        except BaseException:  # plot area of max projection where ROI would have been if it was in this session
            # plot the max projection image with the xy location of the previous ROI
            # this will fail if the familiar session is the one without the cell matched
            print('no cell ROI for', experience_level)
            ax[e] = sf.plot_cell_zoom(roi_masks, dataset.max_projection, cell_roi_id,
                                      spacex=50, spacey=50, show_mask=False, ax=ax[e])
            ax[e].set_title(experience_level)

        # try: # try plotting GLM outputs for this experience level
        if 'running' in weights_features:
            pass
        if 'pupil' in weights_features:
            pass

        # GLM plots start after n_expts for each ROI mask, plus n_extra_cols more axes for omission and change responses (and running and pupil if added)
        # plus one more axes for dropout heatmaps
        i = n_expts + extra_cols + 1

        # weights
        exp_weights = cell_weights[cell_weights.experience_level == experience_level]

        # image kernels
        image_weights = []
        for f, feature in enumerate(weights_features[:8]):  # first 8 are images
            image_weights.append(exp_weights[feature + '_weights'].values[0])
        mean_image_weights = np.mean(image_weights, axis=0)

        # GLM output is all resampled to 30Hz now
        frame_rate = 31
        t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
        ax[i].plot(t_array, mean_image_weights, color=color)
        ax[i].set_ylabel('weight')
        ax[i].set_title('images')
        ax[i].set_xlabel('time (s)')
        ax_to_share = i

        i += 1
        # all other kernels
        for f, feature in enumerate(weights_features[8:]):
            kernel_weights = exp_weights[feature + '_weights'].values[0]
            if feature == 'omissions':
                n_frames_to_clip = int(kernels['omissions']['length'] * frame_rate) + 1
                kernel_weights = kernel_weights[:n_frames_to_clip]
            t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
            ax[i + f].plot(t_array, kernel_weights, color=color)
            ax[i + f].set_ylabel('')
            ax[i + f].set_title(feature)
            ax[i + f].set_xlabel('time (s)')
            ax[i + f].get_shared_y_axes().join(ax[i + f], ax[ax_to_share])

        # except:
        #     print('could not plot GLM kernels for', experience_level)

    # try:
    # plot dropout score heatmaps
    i = n_expts + extra_cols  # change to extra_cols = 4 if running and pupil are added
    # cell_dropouts['cre_line'] = cre_line
    cell_dropouts = cell_dropouts.groupby(['experience_level']).mean()
    if 'ophys_experiment_id' in cell_dropouts.keys():
        cell_dropouts = cell_dropouts.drop(columns='ophys_experiment_id')
    if 'cell_specimen_id' in cell_dropouts.keys():
        cell_dropouts = cell_dropouts.drop(columns='cell_specimen_id')
    cell_dropouts = cell_dropouts[dropout_features]  # order dropouts properly
    dropouts = cell_dropouts.T
    if len(np.where(dropouts < 0)[0]) > 0:
        vmin = -1
        cmap = 'RdBu'
    else:
        vmin = 0
        cmap = 'Blues'
    ax[i] = sns.heatmap(dropouts, cmap=cmap, vmin=vmin, vmax=1, ax=ax[i], cbar=False)
    # ax[i].set_title('coding scores')
    ax[i].set_yticklabels(dropouts.index.values, rotation=0, fontsize=14)
    ax[i].set_xticklabels(dropouts.columns.values, rotation=90, fontsize=14)
    ax[i].set_ylim(0, dropouts.shape[0])
    ax[i].set_xlabel('')

    metadata_string = utils.get_container_metadata_string(dataset.metadata)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6, wspace=0.7)
    fig.suptitle(str(cell_specimen_id) + '_' + metadata_string, x=0.53, y=1.02,
                 horizontalalignment='center', fontsize=16)

    if save_dir:
        print('saving plot for', cell_specimen_id)
        utils.save_figure(fig, figsize, save_dir, folder, str(cell_specimen_id) + '_' + metadata_string + '_' + data_type)
        print('saved')



