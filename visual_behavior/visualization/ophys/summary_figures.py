"""
Created on Sunday July 15 2018

@author: marinag
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from visual_behavior.visualization.utils import save_figure, placeAxesOnGrid
from visual_behavior.ophys.response_analysis import utilities as ut
import visual_behavior.data_access.loading as loading
import visual_behavior.ophys.response_analysis.response_analysis as ra
import pandas as pd

# from visual_behavior.visualization.ophys import population_summary_figures as psf


# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white',
              {'axes.spines.right': False, 'axes.spines.top': False, 'xtick.bottom': True, 'ytick.left': True, })
sns.set_palette('deep')


def plot_max_projection_image(dataset, save_dir=None, folder='max_projection'):
    figsize = (5, 5)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(dataset.max_projection, cmap='gray', vmax=dataset.max_projection.max() / 2.)
    ax.axis('off')
    if save_dir:
        save_figure(fig, figsize, save_dir, folder, str(dataset.experiment_id))


def plot_cell_zoom(roi_masks, max_projection, cell_roi_id, spacex=10, spacey=10, show_mask=False, alpha=0.3, full_image=False, ax=None):
    """
    Plot roi mask image, with or without max projection, either in full image or in a zoomed in portion of image

    :param roi_masks: dataframe with columns 'cell_roi_id', and 'roi_mask'; typically cell_specimen_table attribute of SDK dataset
    :param max_projection: max intensity projection image, typically from 'max_projection' attribute of SDK dataset
    :param cell_roi_id: cell ROI ID of cell to plot
    :param spacex: how much space in x you want to show around the provided OI
    :param spacey: how much space in y you want to show around the provided OI
    :param show_mask: Boolean, whether or not to plot the ROI mask over the max projection
    :param full_image: if True, dont use spacex and y to zoom in, just show the full max image
    :param ax:
    :return:
    """
    m = roi_masks[roi_masks.cell_roi_id == cell_roi_id].roi_mask.values[0]
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
    ax.imshow(max_projection, cmap='gray', vmin=0, vmax=np.amax(max_projection) / 2.)
    if show_mask:
        ax.imshow(mask, cmap='jet', alpha=alpha, vmin=0, vmax=1)
    if not full_image:
        ax.set_xlim(xmin - spacex, xmax + spacex)
        ax.set_ylim(ymax + spacey, ymin - spacey)
    ax.set_title('cell_roi_id ' + str(cell_roi_id))
    ax.grid(False)
    ax.axis('off')
    return ax


def plot_roi_validation(roi_names,
                        roi_df,
                        roi_traces,
                        dff_traces_original,
                        roi_ids,
                        cell_indices,
                        roi_masks,
                        roi_metrics,
                        max_projection,
                        dff_traces_array,
                        ):
    roi_validation = []

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

        if id in roi_ids:
            cell_index = roi_metrics[roi_metrics.roi_id == id].cell_index.values[0]
            cell_specimen_id = roi_metrics[roi_metrics.roi_id == id].id.values[0]
            # cell_index = cell_indices[id]
            ax[2] = plot_cell_zoom(roi_masks, max_projection, cell_specimen_id, spacex=10, spacey=10, show_mask=True,
                                   ax=ax[2])
            ax[2].grid(False)

            ax[4].imshow(max_projection, cmap='gray')
            mask = np.empty(roi_masks[cell_specimen_id].shape)
            mask[:] = np.nan
            (y, x) = np.where(roi_masks[cell_specimen_id] == 1)
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            ax[4].imshow(mask, cmap='RdBu', alpha=0.5)
            ax[4].set_xlim(xmin - 10, xmax + 10)
            ax[4].set_ylim(ymin - 10, ymax + 10)
            ax[4].grid(False)

            ax[5].plot(dff_traces_array[cell_index])
            ax[5].set_title('roi index: ' + str(cell_index) + ', id: ' + str(id))
            ax[5].set_ylabel('dF/F')
            ax[5].set_xlabel('frames')
        else:
            cell_index = ''

        fig.tight_layout()

        roi_validation.append(dict(
            fig=fig,
            index=index,
            id=id,
            cell_index=cell_index,
        ))

    return roi_validation


def get_xticks_xticklabels(trace, frame_rate, interval_sec=1, window=None):
    """
    Function that accepts a timeseries, evaluates the number of points in the trace, and converts from acquisition frames to timestamps

    :param trace: a single trace where length = the number of timepoints
    :param frame_rate: ophys frame rate if plotting a calcium trace, stimulus frame rate if plotting running speed
    :param interval_sec: interval in seconds in between labels

    :return: xticks, xticklabels = xticks in frames corresponding to timepoints in the trace, xticklabels in seconds
    """
    interval_frames = interval_sec * frame_rate
    n_frames = len(trace)
    n_sec = n_frames / frame_rate
    xticks = np.arange(0, n_frames + 5, interval_frames)
    xticklabels = np.arange(0, n_sec + 0.1, interval_sec)
    if not window:
        xticklabels = xticklabels - n_sec / 2
    else:
        xticklabels = xticklabels + window[0]
    return xticks, xticklabels


def plot_mean_trace(traces, frame_rate, ylabel='dF/F', legend_label=None, color='k', interval_sec=1, xlims=[-4, 4],
                    plot_sem=True, ax=None):
    """
    Function that accepts an array of single trial traces and plots the mean and SEM of the trace, with xticklabels in seconds

    :param traces: array of individual trial traces to average and plot. traces must be of equal length
    :param frame_rate: ophys frame rate if plotting a calcium trace, stimulus frame rate if plotting running speed
    :param y_label: 'dF/F' for calcium trace, 'running speed (cm/s)' for running speed trace
    :param legend_label: string describing trace for legend (ex: 'go', 'catch', image name or other condition identifier)
    :param color: color to plot the trace
    :param interval_sec: interval in seconds for x_axis labels
    :param xlims: range in seconds to plot. Must be <= the length of the traces
    :param ax: if None, create figure and axes to plot. If axis handle is provided, plot is created on that axis

    :return: axis handle
    """
    # xlims = [xlims[0] + np.abs(xlims[1]), xlims[1] + xlims[1]]
    xlim = [0, xlims[1] + np.abs(xlims[0])]
    if ax is None:
        fig, ax = plt.subplots()
    if len(traces) > 0:
        trace = np.mean(traces, axis=0)
        times = np.arange(0, len(trace), 1)
        sem = (traces.std()) / np.sqrt(float(len(traces)))
        ax.plot(trace, label=legend_label, linewidth=2, color=color)
        if plot_sem:
            ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=color)

        xticks, xticklabels = get_xticks_xticklabels(trace, frame_rate, interval_sec, window=xlims)
        ax.set_xticks(xticks)
        if interval_sec < 1:
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_xticklabels([int(x) for x in xticklabels])
        ax.set_xlim(xlim[0] * int(frame_rate), xlim[1] * int(frame_rate))
        ax.set_xlabel('time (sec)')
        ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def plot_flashes_on_trace(ax, analysis, window=[-4, 8], trial_type=None, omitted=False, alpha=0.15, facecolor='gray'):
    """
    Function to create transparent gray bars spanning the duration of visual stimulus presentations to overlay on existing figure

    :param ax: axis on which to plot stimulus presentation times
    :param analysis: ResponseAnalysis class instance
    :param trial_type: 'go' or 'catch'. If 'go', different alpha levels are used for stimulus presentations before and after change time
    :param omitted: boolean, use True if plotting response to omitted flashes
    :param alpha: value between 0-1 to set transparency level of gray bars demarcating stimulus times

    :return: axis handle
    """

    frame_rate = analysis.ophys_frame_rate
    stim_duration = analysis.stimulus_duration
    blank_duration = analysis.blank_duration
    if window:
        window = window
    # elif flashes and not omitted:
    #     window = analysis.flash_window
    elif omitted:
        window = analysis.omitted_flash_window
    else:
        window = analysis.trial_window
    change_frame = int(np.abs(window[0]) * frame_rate)
    end_frame = int((np.abs(window[0]) + window[1]) * frame_rate)
    interval = int((blank_duration + stim_duration) * frame_rate)
    if omitted:
        array = np.arange((change_frame + interval), end_frame, interval)
        array = array[1:]
    else:
        array = np.arange(change_frame, end_frame, interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + int(stim_duration * frame_rate)
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    if trial_type == 'go':
        alpha = alpha * 3
    else:
        alpha
    array = np.arange(change_frame - (blank_duration * frame_rate), 0, -interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] - int(stim_duration * frame_rate)
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    return ax


def plot_single_trial_trace(trace, timestamps=None, dataset=None, frame_rate=31, ylabel='dF/F', legend_label=None,
                            color='k', interval_sec=1, xlims=[-4, 4], ax=None):
    """
    Function to plot a single timeseries trace with xticklabels in secconds

    :return: axis handle
    :param trace: single trial timeseries trace to plot
    :param timestamps: optional, if provided will plot xaxis as time in session, if not provided will plot as time relative to change
    :param dataset: optional, if provided will plot accurate stimulus durations and plot in color, if not provided will infer flash times and plot in gray
    :param frame_rate: ophys frame rate if plotting a calcium trace, stimulus frame rate if plotting running speed
    :param y_label: 'dF/F' for calcium trace, 'running speed (cm/s)' for running speed trace
    :param legend_label: string describing trace for legend (ex: 'go', 'catch', image name or other condition identifier)
    :param color: color to plot the trace
    :param interval_sec: interval in seconds for x_axis labels
    :param xlims: range in seconds to plot. Must be <= the length of the traces
    :param ax: if None, create figure and axes to plot. If axis handle is provided, plot is created on that axis

    :return: axis handle
    """
    xlim = [xlims[0] + np.abs(xlims[1]), xlims[1] + xlims[1]]
    if ax is None:
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize)
    if (timestamps is not None) and (dataset is not None):
        ax.plot(timestamps, trace, label=legend_label, linewidth=3, color=color)
        xlim = [timestamps[0], timestamps[-1]]
        ax = add_stim_color_span(dataset, ax, xlim=xlim)
        ax.set_xlim(xlim)
        ax.set_xlabel('time in session (sec)')
    else:
        ax.plot(trace, label=legend_label, linewidth=3, color=color)
        xticks, xticklabels = get_xticks_xticklabels(trace, frame_rate, interval_sec, window=xlims)
        ax.set_xticks([int(x) for x in xticks])
        ax.set_xticklabels([int(x) for x in xticklabels])
        ax.set_xlim(xlim[0] * int(frame_rate), xlim[1] * int(frame_rate))
        ax.set_xlabel('time after change (sec)')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def plot_image_response_for_trial_types(analysis, cell_index, legend=True, save=False, ax=None):
    """
    Function to plot trial avereraged response of a cell for all images separately for 'go' and 'catch' trials. Creates figure and axes to plot.

    :param analysis: ResponseAnalysis class instance
    :param cell: cell index for cell to plot
    :param save: boolean, if True, saves figure to a folder called 'image_responses' in the analysis_dir attribute of the analysis object

    :return: None
    """
    if analysis.use_events:
        suffix = '_events'
    else:
        suffix = ''
    df = analysis.trials_response_df.copy()
    images = np.sort(df.change_image_name.unique())
    trials = analysis.dataset.trials.copy()
    if ax is None:
        figsize = (20, 5)
        fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
        title = str(cell_index) + '_' + str(
            df[df.cell == cell_index].cell_specimen_id.values[0]) + '_' + analysis.dataset.analysis_folder
        plt.suptitle(title, x=0.47, y=1., horizontalalignment='center')
    for i, trial_type in enumerate(['go', 'catch']):
        for c, change_image_name in enumerate(images):
            color = get_color_for_image_name(analysis.dataset, change_image_name)
            selected_trials = trials[
                (trials.change_image_name == change_image_name) & (trials.trial_type == trial_type)].trial.values
            traces = df[(df.cell == cell_index) & (df.trial.isin(selected_trials))].trace.values
            ax[i] = plot_mean_trace(traces, analysis.ophys_frame_rate, legend_label=None, color=color,
                                    interval_sec=1,
                                    xlims=analysis.trial_window, ax=ax[i])
        ax[i] = plot_flashes_on_trace(ax[i], analysis, trial_type=trial_type, omitted=False, alpha=0.3)
        ax[i].set_title(trial_type)
    ax[i].set_ylabel('')
    if legend:
        ax[i].legend(images, loc=9, bbox_to_anchor=(1.2, 1))
    if save:
        fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(right=0.85)
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'image_responses' + suffix,
                    analysis.dataset.analysis_folder + '_' + str(cell_index))

        summary_figure_dir = os.path.join(analysis.dataset.cache_dir, 'summary_figures')
        if not os.path.exists(summary_figure_dir):
            os.makedirs(summary_figure_dir)
        save_figure(fig, figsize, summary_figure_dir,
                    'image_responses' + suffix,
                    analysis.dataset.analysis_folder + '_' + str(cell_index))
        plt.close()
    return ax


def plot_image_change_response(analysis, cell_index, cell_order, legend=False, save=False, show=False, ax=None):
    """
    Function to plot trial avereraged response of a cell for all images 'go' trials. Creates figure and axes to plot.

    :param analysis: ResponseAnalysis class instance
    :param cell: cell index for cell to plot
    :param save: boolean, if True, saves figure to a folder called 'image_responses' in the analysis_dir attribute of the analysis object

    :return: None
    """
    if analysis.use_events:
        suffix = '_events'
        ylabel = 'event_magnitude'
    else:
        suffix = ''
        ylabel = 'mean dF/F'
    df = analysis.trials_response_df.copy()
    df = df[df.trial_type == 'go']
    images = np.sort(df.change_image_name.unique())
    images = images[images != 'omitted']
    trials = analysis.dataset.trials.copy()
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell_index)
    if ax is None:
        figsize = (5, 3)
        fig, ax = plt.subplots(figsize=figsize)
    for c, change_image_name in enumerate(images):
        color = get_color_for_image_name(analysis.dataset.stimulus_presentations, change_image_name)
        selected_trials = trials[(trials.change_image_name == change_image_name)].trial.values
        traces = df[(df.cell == cell_index) & (df.trial.isin(selected_trials))].trace.values
        ax = plot_mean_trace(traces, analysis.ophys_frame_rate, legend_label=None, color=color,
                             interval_sec=2, xlims=analysis.trial_window, ax=ax)
    ax = plot_flashes_on_trace(ax, analysis, trial_type='go', alpha=0.3)
    ax.set_title('cell_index: ' + str(cell_index) + ', cell_specimen_id: ' + str(cell_specimen_id))
    ax.set_title('cell ' + str(cell_order))
    ax.set_xlabel('time after change (sec)')
    ax.set_ylabel('dF/F')
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(images, loc=9, bbox_to_anchor=(1.18, 1), fontsize='x-small')
    if save:
        #         fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(right=0.78)
        plt.gcf().subplots_adjust(left=0.2)
        plt.gcf().subplots_adjust(bottom=0.25)
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'change_responses' + suffix,
                    analysis.dataset.analysis_folder + '_' + str(cell_index))
        save_figure(fig, figsize, os.path.join(analysis.dataset.cache_dir, 'summary_figures'),
                    'change_responses' + suffix,
                    analysis.dataset.analysis_folder + '_' + str(cell_specimen_id) + '_' + str(cell_order) + '_' + str(
                        cell_index))
    if not show:
        plt.close()
    return ax


def plot_event_detection(dff_traces_array, events, analysis_dir):
    figsize = (20, 15)
    xlims_list = [[0, dff_traces_array[0].shape[0]], [10000, 12000], [60000, 62000]]
    for cell in range(len(dff_traces_array)):
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        ax = ax.ravel()
        for i, xlims in enumerate(xlims_list):
            ax[i].plot(dff_traces_array[cell], label='dF/F from L0')
            ax[i].plot(events[cell], color='r', label='events')
            ax[i].set_title('roi ' + str(cell))
            ax[i].set_xlabel('2P frames')
            ax[i].set_ylabel('dF/F')
            ax[i].set_xlim(xlims)
        plt.legend()
        fig.tight_layout()
        save_figure(fig, figsize, analysis_dir, 'event_detection', str(cell))
        plt.close()


def get_colors_for_response_types(response_types):
    c = sns.color_palette()
    colors_dict = {'HIT': c[1], 'MISS': c[4], 'CR': c[0], 'FA': c[2]}
    # note: the following colors were used in another version of this function:
    # colors_dict = {'HIT': c[2], 'MISS': c[8], 'CR': c[0], 'FA': c[3]}
    colors = []
    for val in response_types:
        colors.append(colors_dict[val])
    return colors


def plot_trial_trace_heatmap(trial_response_df, cell, cmap='viridis', vmax=0.5, colorbar=False, ax=None, save_dir=None,
                             window=[-4, 4]):
    response_types = ['HIT', 'MISS', 'FA', 'CR']
    df = trial_response_df.copy()
    rows = 1
    cols = len(df.change_image_name.unique())
    colors = get_colors_for_response_types(response_types)
    if ax is None:
        figsize = (15, 5)
        fig, ax = plt.subplots(rows, cols, figsize=figsize, sharex=True)
        ax = ax.ravel()
    resp_types = []
    for i, change_image_name in enumerate(np.sort(df.change_image_name.unique())):
        im_df = df[(df.cell == cell) & (df.change_image_name == change_image_name)]
        n_frames = im_df.trace.values[0].shape[0]
        n_trials = im_df.trace.shape[0]
        response_matrix = np.empty((n_trials, n_frames))
        response_type_list = []
        segments = []
        idx = 0
        segments.append(idx)
        for y, response_type in enumerate(response_types):
            sub_df = im_df[(im_df.behavioral_response_type == response_type)]
            traces = sub_df.trace.values
            for pos, trial in enumerate(range(traces.shape[0])[::-1]):
                response_matrix[idx, :] = traces[int(trial)]
                response_type_list.append(response_type)
                idx += 1
            segments.append(idx)
            if vmax:
                cax = ax[i].pcolormesh(response_matrix, cmap=cmap, vmax=vmax, vmin=0)
            else:
                cax = ax[i].pcolormesh(response_matrix, cmap=cmap)
            ax[i].set_ylim(0, response_matrix.shape[0])
            ax[i].set_xlim(0, response_matrix.shape[1])
            ax[i].set_yticks(segments)
            ax[i].set_yticklabels('')
            ax[i].set_xlabel('time (s)')
            xticks, xticklabels = get_xticks_xticklabels(np.arange(0, response_matrix.shape[1], 1), 31., interval_sec=2,
                                                         window=window)
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels([int(x) for x in xticklabels])
            #             ax[i].vlines(x=np.mean(xticks), ymin=0, ymax=response_matrix.shape[0], color='w', linewidth=1)
            ax[i].set_title(change_image_name)
        for s in range(len(segments) - 1):
            ax[i].vlines(x=-10, ymin=segments[s], ymax=segments[s + 1], color=colors[s], linewidth=30)
        ax[0].set_ylabel('trials')
        resp_types.append(response_type_list)
    plt.tight_layout()
    if colorbar:
        plt.colorbar(cax, ax=ax[i], use_gridspec=True)
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'trial_trace_heatmap', 'roi_' + str(cell))
    return ax


def plot_mean_response_by_repeat(analysis, cell, save_dir=None, ax=None):
    flash_response_df = analysis.stimulus_response_df.copy()
    flash_response_df = flash_response_df[flash_response_df.omitted == False].copy()
    n_repeats = 15
    palette = sns.color_palette("RdBu", n_colors=n_repeats)
    norm = plt.Normalize(0, n_repeats)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])

    df = flash_response_df[flash_response_df.cell == cell]
    df = df[df['repeat'] < n_repeats]
    figsize = (10, 5)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.stripplot(data=df, x='image_name', y='mean_response', jitter=.2, size=3, ax=ax, hue='repeat',
                       palette=palette)
    ax.set_xticklabels(df.image_name.unique(), rotation=90)
    ax.legend_.remove()
    cbar = ax.figure.colorbar(mappable=sm, ax=ax, use_gridspec=True)
    cbar.set_label('repeat')
    ax.set_title(str(cell) + '_' + analysis.dataset.analysis_folder, fontsize=14)
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'mean_response_by_repeat',
                    analysis.dataset.analysis_folder + '_' + str(cell))
        plt.close()
    return ax


def plot_mean_response_by_image_block(analysis, cell, save_dir=None, ax=None):
    flash_response_df = analysis.stimulus_response_df.copy()
    flash_response_df = flash_response_df[flash_response_df.omitted == False].copy()
    n_blocks = len(flash_response_df.image_block.unique())
    palette = sns.color_palette("RdBu", n_colors=n_blocks)
    norm = plt.Normalize(0, n_blocks)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])

    df = flash_response_df[flash_response_df.cell == cell]
    figsize = (10, 5)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.stripplot(data=df, x='image_name', y='mean_response', jitter=.2, size=3, ax=ax, hue='image_block',
                       palette=palette)
    ax.set_xticklabels(df.image_name.unique(), rotation=90)
    ax.legend_.remove()
    cbar = ax.figure.colorbar(mappable=sm, ax=ax, use_gridspec=True)
    cbar.set_label('image_block')
    ax.set_title(str(cell) + '_' + analysis.dataset.analysis_folder, fontsize=14)
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'mean_response_by_image_block',
                    analysis.dataset.analysis_folder + '_' + str(cell))
        plt.close()
    return ax


def plot_trace(timestamps, trace, ax=None, xlabel='time (seconds)', ylabel='fluorescence', title='roi',
               color=sns.color_palette()[0], width=1.75):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(timestamps, trace, color=color, linewidth=width)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim([timestamps[0], timestamps[-1]])
    return ax


def get_ylabel_and_suffix(use_events):
    if use_events:
        ylabel = 'event_magnitude'
        suffix = '_events'
    else:
        ylabel = 'dF/F'
        suffix = ''
    return ylabel, suffix


# def get_color_for_image_name(dataset, image_name):
#     images = np.sort(dataset.stimulus_presentations.image_name.unique())
#     images = images[images != 'omitted']
#     colors = sns.color_palette("hls", len(images))
#     image_index = np.where(images == image_name)[0][0]
#     color = colors[image_index]
#     return color

def get_color_for_image_name(stim_table, image_name):
    images = np.sort(stim_table.image_name.unique())
    if 'omitted' in images:
        images = images[images != 'omitted']
    colors = sns.color_palette("hls", len(images))
    image_index = np.where(images == image_name)[0][0]
    color = colors[image_index]
    return color


def addSpan(ax, amin, amax, color='k', alpha=0.3, axtype='x', zorder=1):
    if axtype == 'x':
        ax.axvspan(amin, amax, facecolor=color, edgecolor='none', alpha=alpha, linewidth=0, zorder=zorder)
    if axtype == 'y':
        ax.axhspan(amin, amax, facecolor=color, edgecolor='none', alpha=alpha, linewidth=0, zorder=zorder)


def add_stim_color_span(dataset, ax, xlim=None, color=None):
    # xlim should be in seconds
    if xlim is None:
        stim_table = dataset.stimulus_presentations.copy()
    else:
        stim_table = dataset.stimulus_presentations.copy()
        stim_table = stim_table[(stim_table.start_time >= xlim[0]) & (stim_table.stop_time <= xlim[1])]
    if 'omitted' in stim_table.keys():
        stim_table = stim_table[stim_table.omitted == False].copy()
    for idx in stim_table.index:
        start_time = stim_table.loc[idx]['start_time']
        stop_time = stim_table.loc[idx]['stop_time']
        image_name = stim_table.loc[idx]['image_name']
        if color is None:
            image_color = get_color_for_image_name(stim_table, image_name)
        else:
            image_color = color
        # color = ut.get_color_for_image_name(image_names, image_name)
        addSpan(ax, start_time, stop_time, color=image_color)
    return ax


def plot_behavior_events(dataset, ax, behavior_only=False, linewidth=2):
    lick_times = dataset.licks.time.values
    reward_times = dataset.rewards.time.values
    if behavior_only:
        lick_y = 0
        reward_y = -0.2
        ax.set_ylim([-0.5, 0.5])
    else:
        ymin, ymax = ax.get_ylim()
        lick_y = ymin + (ymax * 0.05)
        reward_y = ymin + (ymax * 0.1)
        ax.set_ylim([ymin * 1.3, ymax])
    lick_y_array = np.empty(len(lick_times))
    lick_y_array[:] = lick_y
    reward_y_array = np.empty(len(reward_times))
    reward_y_array[:] = reward_y
    ax.plot(lick_times, lick_y_array, '|', color='g', markeredgewidth=linewidth, label='licks')
    reward_color = 'purple'
    ax.plot(reward_times, reward_y_array, '^', markerfacecolor=reward_color, markeredgecolor=reward_color,
            markeredgewidth=0.3,
            label='rewards')
    return ax


def plot_behavior_for_session(dataset, cache_dir):
    flashes = dataset.stimulus_presentations.copy()
    flashes = ut.annotate_flashes_with_reward_rate(dataset)
    figsize = (15, 4)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dataset.stimulus_timestamps, dataset.running_speed.running_speed.values)
    ax.set_ylabel('running speed(cm/s)')
    ax2 = ax.twinx()
    ax2.plot(flashes.start_time.values, flashes.reward_rate.values, color='red')
    ax2.set_ylabel('reward rate')
    ax = plot_behavior_events(dataset, ax)
    sns.despine(ax=ax2, right=False)
    ax.set_xlabel('time in session (seconds)')
    ax.set_title(dataset.analysis_folder)
    fig.tight_layout()
    save_figure(fig, figsize, dataset.analysis_dir, 'behavior', 'behavior')
    save_figure(fig, figsize, os.path.join(cache_dir, 'summary_figures'), 'behavior', dataset.analysis_folder)


def plot_behavior_annotated(dataset, xmin=1800, duration=20, plot_reward_rate=False, cache_dir=None, show=True):
    xlim = (xmin, xmin + duration)
    all_trials = dataset.all_trials.copy()
    trial_starts = all_trials[all_trials.trial_type != 'aborted'].starttime.values
    ts = trial_starts[(trial_starts > xlim[0])]
    ts = ts[(ts < xlim[1])]
    abort_starts = all_trials[all_trials.trial_type == 'aborted'].starttime.values
    ab = abort_starts[(abort_starts > xlim[0])]
    ab = ab[(ab < xlim[1])]
    flashes = dataset.stimulus_presentations.copy()
    flashes = ut.annotate_flashes_with_reward_rate(dataset)
    figsize = (15, 4)
    fig, ax = plt.subplots(figsize=figsize)
    ax = add_stim_color_span(dataset, ax=ax, xlim=xlim)
    ax.plot(dataset.stimulus_timestamps, dataset.running_speed.running_speed.values, label='run speed', linewidth=2)
    ax.set_ylabel('running speed(cm/s)')
    if plot_reward_rate:
        ax2 = ax.twinx()
        ax2.plot(flashes.start_time.values, flashes.reward_rate.values, color=sns.color_palette()[4],
                 label='reward rate')
        ax2.set_ylabel('reward rate')
        sns.despine(ax=ax2, right=False)
    ax = plot_behavior_events(dataset, ax, linewidth=4)
    ymin, ymax = ax.get_ylim()
    for a in ab:
        ax.vlines(x=a, ymin=ymin, ymax=ymax, color='red', linewidth=2)
    if len(ab) > 0:
        a = ab[0]
        ax.vlines(x=a, ymin=ymin, ymax=ymax, color='red', linewidth=2, label='aborted trial start')
    for t in ts:
        ax.vlines(x=t, ymin=ymin, ymax=ymax, color='green', linewidth=2)
    if len(ts) > 0:
        t = ts[0]
        ax.vlines(x=t, ymin=ymin, ymax=ymax, color='green', linewidth=2, label='go/catch trial start')
    ax.set_xlabel('time in session (seconds)')
    ax.set_title(dataset.analysis_folder)
    ax.set_xlim(xlim)
    ax.legend(loc='upper left', fontsize='x-small')
    #     ax.legend(bbox_to_anchor=(0.05,1.1), ncol=5, loc='upper left')
    fig.tight_layout()
    fig.subplots_adjust(top=.8)
    save_figure(fig, figsize, dataset.analysis_dir, 'behavior', 'behavior_' + str(xlim[0]))
    if cache_dir:
        save_figure(fig, figsize, os.path.join(cache_dir, 'summary_figures'), 'behavior',
                    dataset.analysis_folder + '_' + str(xlim[0]))
    if not show:
        plt.close()


def restrict_axes(xmin, xmax, interval, ax):
    xticks = np.arange(xmin, xmax, interval)
    ax.set_xticks(xticks)
    ax.set_xlim([xmin, xmax])
    return ax


def plot_licks_and_rewards(dataset, ax, behavior_only=False):
    lick_times = dataset.licks.timestamps.values
    reward_times = dataset.rewards.timestamps.values
    if behavior_only:
        lick_y = 0
        reward_y = -0.2
        ax.set_ylim([-0.5, 0.5])
    else:
        ymin, ymax = ax.get_ylim()
        lick_y = ymin + (ymax * 0.05)
        reward_y = ymin - (ymax * 0.1)
    lick_y_array = np.empty(len(lick_times))
    lick_y_array[:] = lick_y
    reward_y_array = np.empty(len(reward_times))
    reward_y_array[:] = reward_y
    ax.plot(lick_times, lick_y_array, '|', color='g', markeredgewidth=1, label='licks')
    reward_color = sns.color_palette()[0]
    reward_color = 'purple'
    ax.plot(reward_times, reward_y_array, '^', markerfacecolor=reward_color, markeredgecolor=reward_color,
            markeredgewidth=0.3,
            label='rewards')
    return ax


def plot_behavior_for_epoch(dataset, start_time, duration, exclude_running=False, legend=False,
                            save_figures=False, save_dir=None, ax=None):
    xlim = (start_time, start_time + duration + 1)
    running_speed = dataset.running_speed.speed.values
    running_times = dataset.running_speed.timestamps.values
    if ax is None:
        figsize = (15, 2)
        fig, ax = plt.subplots(figsize=figsize)
    if exclude_running:
        folder = 'behavior_epoch_licks'
        ax.set_ylim(0, 1)
        ax = plot_licks_and_rewards(dataset, ax)
        ax.set_ylim(-0.4, 0.5)
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.tick_params(axis='x', left='off')
        ax.tick_params(axis='y', left='off')
        sns.despine(ax=ax, top=False, right=False)
    else:
        folder = 'behavior_epoch_running_licks'
        ax.plot(running_times, running_speed, color=sns.color_palette()[0], label='run_speed')
        ax.set_ylabel('run speed\n(cm/s)')
        ax = plot_licks_and_rewards(dataset, ax, behavior_only=False)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin * 1.1, ymax)
    if legend:
        ax.legend(bbox_to_anchor=(1, .46), ncol=3)
    ax = add_stim_color_span(dataset, ax, xlim=xlim)
    ax.set_xlim(xlim)
    ax.set_xlabel('time (seconds)')
    ax.set_xticks(np.arange(xlim[0], xlim[1], 10))
    xticklabels = np.arange(xlim[0], xlim[1], 10) - xlim[0]
    ax.set_xticklabels([int(x) for x in xticklabels])
    ax.set_title(dataset.analysis_folder + ' - ' + str(start_time), fontsize=14)
    plt.gcf().subplots_adjust(bottom=0.3)
    if save_figures:
        if save_dir:
            save_figure(fig, figsize, save_dir, folder, dataset.analysis_folder + '_' + str(start_time))
        save_figure(fig, figsize, dataset.analysis_dir, folder, str(start_time))
        plt.close()
    return ax


def plot_behavior_block(dataset, initial_time, duration=60, save_figures=False, save_dir=None):
    rows = 4
    figsize = (15, 6)
    fig, ax = plt.subplots(rows, 1, figsize=figsize)
    for i, start_time in enumerate(np.arange(initial_time, initial_time + (duration * rows), duration)):
        ax[i] = plot_behavior_for_epoch(dataset, start_time, duration, exclude_running=True, ax=ax[i],
                                        save_figures=False, save_dir=None)
        ax[i].set_title('')
        ax[i].tick_params(axis='x', bottom=False)
        ax[i].set_xlabel('')
        ax[i].set_xticklabels([])
        ax[i].set_ylabel('epoch ' + str(i + 1), fontsize=14)
    ax[i].tick_params(axis='x', bottom=True)
    # ax[0].set_title(dataset.analysis_folder+' - '+str(initial_time), fontsize=14)
    ax[0].set_title('example task flow and behavior performance', fontsize=18)
    xlim = (start_time, start_time + duration + 1)
    ax[i].set_xticks(np.arange(xlim[0], xlim[1], 10))
    xticklabels = np.arange(xlim[0], xlim[1], 10) - xlim[0]
    ax[i].set_xticklabels([int(x) for x in xticklabels])
    ax[i].set_xlabel('time (seconds)')
    legend = ax[i].legend(bbox_to_anchor=(0.125, -0.2), fontsize='x-small')
    plt.setp(legend.get_title(), fontsize='x-small')
    if save_figures:
        folder = 'behavior_blocks'
        if save_dir:
            save_figure(fig, figsize, save_dir, folder, dataset.analysis_folder + '_' + str(start_time))
        save_figure(fig, figsize, dataset.analysis_dir, folder, str(start_time))


def plot_lick_raster(dataset, ax=None, save_figures=False, save_dir=None):
    trials = dataset.trials
    # image_set = dataset.metadata.session_type.values[0][-1]
    # mouse_id = str(dataset.metadata.donor_id.values[0])
    if ax is None:
        figsize = (4, 5)
        fig, ax = plt.subplots(figsize=figsize)
    for trial in trials.trial.values:
        trial_data = trials.iloc[trial]
        trial_start = trial_data.start_time - trial_data.change_time
        lick_times = [(t - trial_data.change_time) for t in trial_data.lick_times]
        reward_time = [(t - trial_data.change_time) for t in trial_data.reward_times]
        color = trial_data.trial_type_color
        color = 'white'
        if trial_data.trial_type == 'go':
            color = 'green'  # sns.color_palette()[0]
            linewidth = 1
        else:
            color = 'red'  # sns.color_palette()[1]
            linewidth = 3
        ax.axhspan(trial, trial + 1, -200, 200, color='white', alpha=.1)
        if len(reward_time) > 0:
            ax.plot(reward_time[0], trial + 0.5, '^', color='purple', label='reward', markersize=3)
        ax.vlines(trial_start, trial, trial + 1, color='k', linewidth=1, alpha=0.8)
        ax.vlines(lick_times, trial, trial + 1, color=color, linewidth=linewidth)
        ax.vlines(0, trial, trial + 1, color=[.5, .5, .5], linewidth=1)
    ax.axvspan(trial_data.response_window[0], trial_data.response_window[1], facecolor='gray', alpha=.4,
               edgecolor='none')
    ax.grid(False)
    ax.set_ylim(0, len(trials))
    ax.set_xlim([-1, 5])
    ax.set_ylabel('trial #')
    ax.set_xlabel('time after change (sec)')
    # ax.set_title('M'+mouse_id+' image set '+image_set, fontsize=14)
    ax.set_title('change aligned lick raster', fontsize=16)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.3)
    if save_figures:
        save_figure(fig, figsize, dataset.analysis_dir, 'behavior', 'lick_raster')
        if save_dir:
            save_figure(fig, figsize, save_dir, 'lick_rasters', dataset.analysis_folder)
    return ax


def plot_behavior_events_trace(dataset, cell_list, xmin=360, length=3, ax=None, save=False, use_events=False):
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    xmax = xmin + 60 * length
    interval = 20
    for cell_index in cell_list:
        cell_specimen_id = dataset.get_cell_specimen_id_for_cell_index(cell_index)
        if ax is None:
            figsize = (15, 4)
            fig, ax = plt.subplots(figsize=figsize)
        if use_events:
            ax = plot_trace(dataset.ophys_timestamps, dataset.events[cell_index, :], ax,
                            title='cell_specimen_id: ' + str(cell_specimen_id), ylabel=ylabel)
        else:
            ax = plot_trace(dataset.ophys_timestamps, dataset.dff_traces_array[cell_index, :], ax,
                            title='cell_specimen_id: ' + str(cell_specimen_id), ylabel=ylabel)
        ax = add_stim_color_span(dataset, ax, xlim=[xmin, xmax])
        ax = plot_behavior_events(dataset, ax)
        ax = restrict_axes(xmin, xmax, interval, ax)
        if save:
            fig.tight_layout()
            save_figure(fig, figsize, dataset.analysis_dir, 'behavior_events_traces',
                        'behavior_events_trace_' + str(cell_specimen_id) + suffix)
            plt.close()
            ax = None
    return ax


# only plots events or traces
def plot_average_flash_response_example_cells(analysis, active_cell_indices, include_changes=False,
                                              save_figures=False, save_dir=None, folder=None, ax=None):
    dataset = analysis.dataset
    fdf = analysis.get_response_df(df_name='stimulus_response_df')
    last_flash = fdf.stimulus_presentations_id.unique()[-1]  # sometimes last flash is truncated
    fdf = fdf[fdf.stimulus_presentations_id != last_flash]

    conditions = ['cell_specimen_id', 'image_name']
    mdf = ut.get_mean_df(fdf, analysis, conditions=conditions, flashes=True)

    if include_changes:
        rdf = ut.get_mean_df(fdf[fdf.change == True], analysis, conditions=conditions, flashes=True)

    cell_specimen_ids = [dataset.get_cell_specimen_id_for_cell_index(cell_index) for cell_index in active_cell_indices]
    image_names = np.sort(mdf.image_name.unique())

    if ax is None:
        if len(active_cell_indices) < 5:
            figsize = (12, 4)
        elif len(active_cell_indices) < 10:
            figsize = (12, 7.8)
        else:
            figsize = (12, 10)
        fig, ax = plt.subplots(len(cell_specimen_ids), len(image_names), figsize=figsize, sharex=True)
        ax = ax.ravel()

    i = 0
    for c, cell_specimen_id in enumerate(cell_specimen_ids):
        cell_data = mdf[(mdf.cell_specimen_id == cell_specimen_id)]
        if include_changes:
            cell_data_go = rdf[(rdf.cell_specimen_id == cell_specimen_id)]
            combined_traces = np.hstack([cell_data.mean_trace.values, cell_data_go.mean_trace.values])
            maxs = [np.amax(trace) for trace in combined_traces]
            ymax = np.amax(maxs) * 1.2
        else:
            ymax = np.amax(np.hstack(cell_data.mean_trace.values)) * 1.2

        for m, image_name in enumerate(image_names):
            color = ut.get_color_for_image_name(image_names, image_name)

            cdf = cell_data[(cell_data.image_name == image_name)]
            ax[i] = plot_mean_trace_from_mean_df(cdf, analysis.ophys_frame_rate, color=[.4, .4, .4],
                                                 interval_sec=0.5,
                                                 xlims=analysis.flash_window, ax=ax[i], plot_sem=False)
            if include_changes:
                if image_name != 'omitted':  # changes cant be omitted
                    rcdf = cell_data_go[(cell_data_go.image_name == image_name)]
                    ax[i] = plot_mean_trace_from_mean_df(rcdf, analysis.ophys_frame_rate, color=[.6, .6, .6],
                                                         interval_sec=0.5,
                                                         xlims=analysis.flash_window, ax=ax[i], plot_sem=False)

            ax[i] = plot_flashes_on_trace(ax[i], analysis, window=analysis.flash_window, facecolor=color, alpha=0.3)
            ax[i].axis('off')
            ax[i].set_ylim(-0.05, ymax)
            if m == 0:
                ax[i].set_ylabel('x')
            if c == 0:
                ax[i].set_title(image_name)
            if c == len(cell_specimen_ids):
                ax[i].set_xlabel('time (s)')
            i += 1
    for x in range(len(cell_specimen_ids)):
        ax[x * len(image_names)].vlines(x=0, ymin=0, ymax=.05, linewidth=3)

    if save_figures:
        if save_dir:
            save_figure(fig, figsize, save_dir, folder, str(dataset.experiment_id) + '_example_cell_responses')


def plot_example_traces_and_behavior(dataset, cell_indices, xmin_seconds, length_mins, save_figures=False, dff_max=3,
                                     include_running=False, cell_label=False, use_events=False, save_dir=None,
                                     include_pupil_area=True, folder='example_traces'):
    if use_events:
        suffix = '_events'
    else:
        suffix = ''
    if include_running and include_pupil_area:
        n = 3
    elif include_running or include_pupil_area:
        n = 2
    else:
        n = 1
    interval_seconds = 20
    xmax_seconds = xmin_seconds + (length_mins * 60) + 1
    xlim = [xmin_seconds, xmax_seconds]

    figsize = (16, 10)
    fig, ax = plt.subplots(len(cell_indices) + n, 1, figsize=figsize, sharex=True)
    ax = ax.ravel()
    if dff_max > 3:
        scale = 2.
    else:
        scale = 1.

    dff_traces_array = np.vstack(dataset.dff_traces.dff.values)
    ymins = []
    ymaxs = []
    for i, cell_index in enumerate(cell_indices):
        ax[i] = plot_trace(dataset.ophys_timestamps, dff_traces_array[cell_index, :], ax=ax[i],
                           title='', ylabel=str(cell_index), color=[.6, .6, .6])
        ax[i] = add_stim_color_span(dataset, ax=ax[i], xlim=xlim)
        ax[i] = restrict_axes(xmin_seconds, xmax_seconds, interval_seconds, ax=ax[i])
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                          labeltop=False, labelright=False, labelleft=False, labelbottom=False)
        ax[i].vlines(x=xmin_seconds, ymin=0, ymax=dff_max / 2., linewidth=4)
        ax[i].set_ylim(ymin=-1, ymax=dff_max)
        if use_events:
            ax2 = ax[i].twinx()
            events_array = None  # note: the following line is failing the lint test because events_array is undefined
            #       the function will fail with the undefined variable
            #       setting it to None will avoid the error, but the function will still fail (DRO, 9/1/2020)
            ax2 = plot_trace(dataset.ophys_timestamps, events_array[cell_index, :], ax=ax2,
                             title='', ylabel=None, color=sns.color_palette()[0], width=1.5)  # color=[.4,.4,.4])
            sns.despine(ax=ax2, left=True, bottom=True)
            ax2.tick_params(which='both', bottom=False, top=False, right=False, left=False,
                            labeltop=False, labelright=False, labelleft=False, labelbottom=False)
            ax2 = restrict_axes(xmin_seconds, xmax_seconds, interval_seconds, ax=ax2)
            ax2.set_ylim(ymin=0, ymax=dff_max / scale)

        ymin, ymax = ax[i].get_ylim()
        ymins.append(ymin)
        ymaxs.append(ymax)
        if cell_label:
            ax[i].set_ylabel('cell ' + str(i), fontsize=12)
        else:
            ax[i].set_ylabel('')
        ax[i].set_xlabel('')

        sns.despine(ax=ax[i], left=True, bottom=True)
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                          labeltop=False, labelright=False, labelleft=False, labelbottom=False)

    i += 1
    ax[i] = plot_behavior_for_epoch(dataset, xmin_seconds, length_mins * 60, exclude_running=True, ax=ax[i])
    ax[i].set_xlim(xlim)
    ax[i].set_title('')
    #     ax[i].legend(bbox_to_anchor=(-1,-1), fontsize=14)
    legend = ax[i].legend(bbox_to_anchor=(0.15, -0.2), fontsize='small')
    plt.setp(legend.get_title(), fontsize='small')
    sns.despine(ax=ax[i], left=True, bottom=True)
    ax[i].tick_params(which='both', bottom=True, top=False, right=False, left=False,
                      labeltop=False, labelright=False, labelleft=False, labelbottom=True)

    if include_running:
        i += 1
        ax[i] = plot_behavior_for_epoch(dataset, xmin_seconds, length_mins * 60, exclude_running=False, ax=ax[i])
        ax[i] = restrict_axes(xmin_seconds, xmax_seconds, interval_seconds, ax=ax[i])
        ax[i].set_ylabel('run speed\n(cm/s)', fontsize=12)
        ax[i].set_title('')
        ax[i].vlines(x=xmin_seconds, ymin=0, ymax=30, linewidth=4)
        sns.despine(ax=ax[i], left=True, bottom=True)
        ax[i].tick_params(which='both', bottom=True, top=False, right=False, left=False,
                          labelbottom=True, labeltop=False, labelright=False, labelleft=False)

    if include_pupil_area:
        if dataset.eye_tracking is not None:
            i += 1
            ax[i].plot(dataset.eye_tracking.time.values, dataset.eye_tracking.pupil_area.values, color=sns.color_palette()[0], label='pupil_area')
            ax[i].set_ylabel('pupil area\n(pixels**2')
            ymin, ymax = ax[i].get_ylim()
            ax[i].set_ylim(np.percentile(ymin, 5) * 1.1, np.percentile(ymax, 95))
            ax[i] = add_stim_color_span(dataset, ax[i], xlim=xlim)
            ax[i] = restrict_axes(xmin_seconds, xmax_seconds, interval_seconds, ax=ax[i])
            ax[i].set_xlabel('time (seconds)')
            ax[i].set_title('')
            sns.despine(ax=ax[i], left=True, bottom=True)
            ax[i].tick_params(which='both', bottom=True, top=False, right=False, left=False,
                              labelbottom=True, labeltop=False, labelright=False, labelleft=False)

    xticks = np.arange(xmin_seconds, xmax_seconds, interval_seconds)
    xticklabels = np.arange(0, xmax_seconds - xmin_seconds, interval_seconds)
    ax[i].set_xticks(xticks)
    ax[i].set_xticklabels([int(xticklabel) for xticklabel in xticklabels])
    ax[i].set_xlabel('time (seconds)')
    ax[0].set_title(dataset.analysis_folder)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(bottom=0.2)

    if save_figures:
        if save_dir:
            save_figure(fig, figsize, save_dir, folder, str(dataset.experiment_id) + '_' + str(xlim[0]) + suffix)


def plot_reliability_trials(analysis, tdf, mean_tdf, cell, xlims=[-4, 8], save_figures=False, save_dir=None,
                            folder=None):
    cell_specimen_id = tdf[tdf.cell == cell].cell_specimen_id.values[0]
    figsize = (4, 3)
    fig, ax = plt.subplots(figsize=figsize)
    traces = tdf[(tdf.cell == cell) & (tdf.pref_stim == True) & (tdf.trial_type == 'go')].trace.values
    ax = plot_mean_trace_with_variability(traces, frame_rate=31., ylabel='dF/F', label=None, color=[.3, .3, .3],
                                          interval_sec=2,
                                          xlims=xlims, ax=ax)
    ax = plot_flashes_on_trace(ax, analysis, trial_type='go', window=xlims)
    reliability = \
        mean_tdf[(mean_tdf.cell == cell) & (mean_tdf.pref_stim == True) & (mean_tdf.trial_type == 'go')].reliability.values[
            0]
    ax.set_title('reliability: ' + str(np.round(reliability, 2)))
    fig.tight_layout()

    if save_figures:
        save_figure(fig, figsize, save_dir, folder,
                    analysis.dataset.analysis_folder + '_reliability_trials_' + str(cell_specimen_id))
        plt.close()


def plot_reliability_flashes(analysis, fdf, mean_fdf, cell, xlims=[-0.5, 0.75], save_figures=False, save_dir=None,
                             folder=None):
    cell_specimen_id = fdf[fdf.cell == cell].cell_specimen_id.values[0]
    figsize = (4, 3)
    fig, ax = plt.subplots(figsize=figsize)
    traces = fdf[(fdf.cell == cell) & (fdf.pref_stim == True) & (fdf.repeat == 1)].trace.values
    ax = plot_mean_trace_with_variability(traces, frame_rate=31., ylabel='dF/F', label=None, color=[.3, .3, .3],
                                          interval_sec=0.5,
                                          xlims=xlims, ax=ax)
    ax = plot_flashes_on_trace(ax, analysis, flashes=True, window=xlims)
    reliability = \
        mean_fdf[(mean_fdf.cell == cell) & (mean_fdf.pref_stim == True) & (mean_fdf.repeat == 1)].reliability.values[0]
    ax.set_title('reliability: ' + str(np.round(reliability, 2)))

    fig.tight_layout()
    if save_figures:
        save_figure(fig, figsize, save_dir, folder,
                    analysis.dataset.analysis_folder + '_reliability_flashes_' + str(cell_specimen_id))
        plt.close()


def plot_transition_type_heatmap(analysis, cell_list, cmap='jet', vmax=None, save=False, ax=None, colorbar=True):
    response_types = ['HIT', 'MISS', 'FA', 'CR']
    df = analysis.trials_response_df.copy()
    images = np.sort(df.change_image_name.unique())
    rows = 1
    cols = int(len(images) / float(rows))
    figsize = (15, 4 * rows)
    colors = get_colors_for_response_types(response_types)
    for cell in cell_list:
        cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
        if ax is None:
            fig, ax = plt.subplots(rows, cols, figsize=figsize, sharex=True)
            ax = ax.ravel()
        resp_types = []
        for i, image_name in enumerate(images):
            im_df = df[(df.cell == cell) & (df.change_image_name == image_name) & (df.trial_type != 'autorewarded')]
            n_frames = im_df.trace.values[0].shape[0]
            n_trials = im_df.trace.shape[0]
            response_matrix = np.empty((n_trials, n_frames))
            response_type_list = []
            segments = []
            idx = 0
            segments.append(idx)
            for y, response_type in enumerate(response_types):
                subset = im_df[(im_df.behavioral_response_type == response_type)]
                responses = subset.trace.values
                for pos, trial in enumerate(range(responses.shape[0])[::-1]):
                    response_matrix[idx, :] = responses[int(trial)]
                    response_type_list.append(response_type)
                    idx += 1
                segments.append(idx)
                if vmax:
                    cax = ax[i].pcolormesh(response_matrix, cmap=cmap, vmax=vmax, vmin=0)
                else:
                    cax = ax[i].pcolormesh(response_matrix, cmap=cmap)
                ax[i].set_ylim(0, response_matrix.shape[0])
                ax[i].set_xlim(0, response_matrix.shape[1])
                ax[i].set_yticks(segments)
                ax[i].set_yticklabels('')
                ax[i].set_xlabel('time (s)')
                xticks, xticklabels = get_xticks_xticklabels(im_df.trace.values[0], analysis.ophys_frame_rate,
                                                             interval_sec=2, window=analysis.trial_window)
                xticklabels = [int(label) for label in xticklabels]
                ax[i].set_xticks(xticks)
                ax[i].set_xticklabels(xticklabels)
                ax[i].set_title(image_name)
            for s in range(len(segments) - 1):
                ax[i].vlines(x=-10, ymin=segments[s], ymax=segments[s + 1], color=colors[s], linewidth=25)
            ax[0].set_ylabel('trials')
            resp_types.append(response_type_list)
            if colorbar:
                plt.colorbar(cax, ax=ax[i], use_gridspec=True)
        plt.tight_layout()
        if save:
            save_figure(fig, figsize, ra.analysis_dir, 'transition_type_heatmap', str(cell_specimen_id))
            plt.close()
            ax = None
    return ax


def plot_mean_cell_response_heatmap(analysis, cell, values='mean_response', index='initial_image_name',
                                    columns='change_image_name',
                                    save=False, ax=None, use_events=False):
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    df = analysis.trials_response_df.copy()
    resp_df = df[df.cell == cell]
    resp_df = resp_df[[index, columns, values]]
    if ax is None:
        figsize = (6, 6)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('roi ' + str(cell) + ' - ' + values, fontsize=16, va='bottom', ha='center')
    response_matrix = pd.pivot_table(resp_df,
                                     values='mean_response',
                                     index=[index],
                                     columns=[columns])

    sns.heatmap(response_matrix, cmap='magma', linewidths=0, linecolor='white', square=True, annot=False,
                annot_kws={"fontsize": 10}, vmin=0, vmax=np.amax(np.amax(response_matrix)),
                robust=True, cbar_kws={"drawedges": False, "shrink": 0.7, "label": ylabel}, ax=ax)

    ax.set_xlabel(columns, fontsize=16)
    ax.set_ylabel(index, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if save:
        fig.tight_layout()
        save_figure(fig, figsize, analysis.analysis_dir, 'cell_response_heatmaps', str(cell_specimen_id))
        plt.close()
    return ax


def plot_ranked_image_tuning_curve_trial_types(analysis, cell, ax=None, save=False, use_events=False):
    from scipy.stats import sem as compute_sem
    from visual_behavior.ophys.response_analysis import utilities as ut
    c = sns.color_palette()
    colors = [c[3], c[0], c[2]]
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    tdf = ut.get_mean_df(analysis.trials_response_df, analysis,
                         conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    if ax is None:
        figsize = (6, 4)
        fig, ax = plt.subplots(figsize=figsize)
    ls_list = []
    for t, trial_type in enumerate(['go', 'catch']):
        tmp = tdf[(tdf.cell_specimen_id == cell_specimen_id) & (tdf.trial_type == trial_type)]
        responses = tmp.mean_response.values
        ls = ut.compute_lifetime_sparseness(responses)
        ls_list.append(ls)
        order = np.argsort(responses)[::-1]
        images = tmp.change_image_name.unique()[order]
        for i, image in enumerate(images):
            means = tmp[tmp.change_image_name == image].mean_response.values[0]
            sem = compute_sem(means)
            ax.errorbar(i, np.mean(means), yerr=sem, color=colors[t])
            ax.plot(i, np.mean(means), 'o', color=colors[t])
        ax.plot(i, np.mean(means), 'o', color=colors[t], label=trial_type)
    # ax.set_ylim(ymin=0)
    ax.set_ylabel('mean ' + ylabel)
    ax.set_xticks(np.arange(0, len(responses), 1))
    ax.set_xticklabels(images, rotation=90)
    ax.legend()
    ax.set_title('lifetime sparseness go: ' + str(np.round(ls_list[0], 3)) + '\nlifetime sparseness catch: ' + str(
        np.round(ls_list[1], 3)))
    if save:
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'lifetime_sparseness' + suffix,
                    'trial_types_tc_' + str(cell_specimen_id))
    return ax


def plot_ranked_image_tuning_curve_all_flashes(analysis, cell, ax=None, save=None, save_dir=None, use_events=False):
    from scipy.stats import sem as compute_sem
    from visual_behavior.ophys.response_analysis import utilities as ut
    c = sns.color_palette()
    colors = [c[3], c[0], c[2]]
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    fdf = analysis.stimulus_response_df.copy()
    fdf = fdf[fdf.omitted == False].copy()
    fmdf = ut.get_mean_df(fdf, analysis, conditions=['cell_specimen_id', 'image_name'], flashes=True)
    if ax is None:
        figsize = (6, 4)
        fig, ax = plt.subplots(figsize=figsize)
    tmp = fdf[(fdf.cell_specimen_id == cell_specimen_id)]
    responses = fmdf[(fmdf.cell_specimen_id == cell_specimen_id)].mean_response.values
    ls = ut.compute_lifetime_sparseness(responses)
    order = np.argsort(responses)[::-1]
    images = fmdf[(fmdf.cell_specimen_id == cell_specimen_id)].image_name.values
    images = images[order]
    for i, image in enumerate(images):
        means = tmp[tmp.image_name == image].mean_response.values
        sem = compute_sem(means)
        ax.errorbar(i, np.mean(means), yerr=sem, color=colors[1])
        ax.plot(i, np.mean(means), 'o', color=colors[1])
        ax.plot(i, np.mean(means), 'o', color=colors[1])
    # ax.set_ylim(ymin=0)
    ax.set_ylabel('mean dF/F')
    ax.set_xticks(np.arange(0, len(responses), 1))
    ax.set_xticklabels(images, rotation=90)
    ax.set_title('lifetime sparseness all flashes: ' + str(np.round(ls, 3)))
    ax.legend()
    if save:
        save_figure(fig, figsize, save_dir, 'lifetime_sparseness_flashes', 'roi_' + str(cell))
    return ax


def plot_ranked_image_tuning_curve_flashes(analysis, cell, repeats=[1, 5, 10], ax=None, save=None, use_events=False):
    from scipy.stats import sem as compute_sem
    from visual_behavior.ophys.response_analysis import utilities as ut
    c = sns.color_palette()
    colors = [c[3], c[0], c[2]]
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    fdf = analysis.stimulus_response_df.copy()
    fdf = fdf[fdf.repeat.isin(repeats)]
    fdf = fdf[fdf.omitted == False]
    fmdf = ut.get_mean_df(fdf, analysis, conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)
    if ax is None:
        figsize = (6, 4)
        fig, ax = plt.subplots(figsize=figsize)
    ls_list = []
    for r, repeat in enumerate(fmdf.repeat.unique()):
        tmp = fdf[(fdf.cell_specimen_id == cell_specimen_id) & (fdf.repeat == repeat)]
        responses = fmdf[(fmdf.cell_specimen_id == cell_specimen_id) & (fmdf.repeat == repeat)].mean_response.values
        ls = ut.compute_lifetime_sparseness(responses)
        ls_list.append(ls)
        if r == 0:
            order = np.argsort(responses)[::-1]
            images = fmdf[(fmdf.cell_specimen_id == cell_specimen_id) & (fmdf.repeat == repeat)].image_name.values
            images = images[order]
        for i, image in enumerate(images):
            means = tmp[tmp.image_name == image].mean_response.values
            sem = compute_sem(means)
            ax.errorbar(i, np.mean(means), yerr=sem, color=colors[r])
            ax.plot(i, np.mean(means), 'o', color=colors[r])
            ax.plot(i, np.mean(means), 'o', color=colors[r])
    # ax.set_ylim(ymin=0)
    ax.set_ylabel('mean ' + ylabel)
    ax.set_xticks(np.arange(0, len(responses), 1))
    ax.set_xticklabels(images, rotation=90)
    ax.set_title('lifetime sparseness repeat ' + str(repeats[0]) + ': ' + str(np.round(ls_list[0], 3)) +
                 '\nlifetime sparseness repeat ' + str(repeats[1]) + ': ' + str(np.round(ls_list[1], 3)) +
                 '\nlifetime sparseness repeat ' + str(repeats[2]) + ': ' + str(np.round(ls_list[2], 3)))
    ax.legend()
    if save:
        fig.tight_layout()
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'lifetime_sparseness_flashes' + suffix,
                    str(cell_specimen_id))
    return ax


def plot_mean_trace_from_mean_df(cell_data, frame_rate=31., ylabel='dF/F', legend_label=None, color='k', interval_sec=1,
                                 xlims=[-4, 4],
                                 ax=None, plot_sem=True, width=3):
    xlim = [0, xlims[1] + np.abs(xlims[0])]
    if ax is None:
        fig, ax = plt.subplots()
    trace = cell_data.mean_trace.values[0]
    times = np.arange(0, len(trace), 1)
    sem = cell_data.sem_trace.values[0]
    ax.plot(trace, label=legend_label, linewidth=width, color=color)
    if plot_sem:
        ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=color)
    xticks, xticklabels = get_xticks_xticklabels(trace, frame_rate, interval_sec, window=xlims)
    ax.set_xticks(xticks)
    if interval_sec >= 1:
        ax.set_xticklabels([int(x) for x in xticklabels])
    else:
        ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim[0] * int(frame_rate), xlim[1] * int(frame_rate))
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def plot_mean_trace_with_variability(traces, frame_rate, ylabel='dF/F', label=None, color='k', interval_sec=1,
                                     xlims=[-4, 4], ax=None, flashes=True):
    #     xlim = [xlims[0] + np.abs(xlims[0]), xlims[1] + np.abs(xlims[0])]
    if ax is None:
        fig, ax = plt.subplots()
    if len(traces) > 0:
        mean_trace = np.mean(traces, axis=0)
        # times = np.arange(0, len(mean_trace), 1)
        # sem = (traces.std()) / np.sqrt(float(len(traces)))
        for trace in traces:
            ax.plot(trace, linewidth=1, color='gray')
        ax.plot(mean_trace, label=label, linewidth=3, color=color, zorder=100)
        xticks, xticklabels = get_xticks_xticklabels(mean_trace, frame_rate, interval_sec, window=xlims)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(0, (np.abs(xlims[0]) + xlims[1]) * int(frame_rate))
        ax.set_xlabel('time (sec)')
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
    return ax


def plot_mean_response_pref_stim_metrics(analysis, cell, ax=None, save=None, use_events=False, xlims=[-2, 2]):
    import visual_behavior.ophys.response_analysis.utilities as ut
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    tdf = analysis.trials_response_df
    tdf = tdf[tdf.cell_specimen_id == cell_specimen_id]
    fdf = analysis.stimulus_response_df
    fdf = fdf[fdf.cell_specimen_id == cell_specimen_id]
    mdf = ut.get_mean_df(analysis.trials_response_df, analysis,
                         conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    mdf = mdf[mdf.cell_specimen_id == cell_specimen_id]
    trial_window = analysis.trial_window
    xlims = [-2, 2]
    if ax is None:
        figsize = (12, 6)
        fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
        ax = ax.ravel()
    pref_image = tdf[tdf.pref_stim == True].change_image_name.values[0]
    images = ut.get_image_names(mdf)
    color = ut.get_color_for_image_name(images, pref_image)
    for i, trial_type in enumerate(['go', 'catch']):
        tmp = tdf[(tdf.trial_type == trial_type) & (tdf.change_image_name == pref_image)]
        mean_df = mdf[(mdf.trial_type == trial_type) & (mdf.change_image_name == pref_image)]
        ax[i] = plot_mean_trace_with_variability(tmp.trace.values, analysis.ophys_frame_rate, label=None, color=color,
                                                 interval_sec=1, xlims=trial_window, ax=ax[i])
        ax[i] = plot_flashes_on_trace(ax[i], analysis, trial_type=trial_type, omitted=False, alpha=.05 * 8)
        ax[i].set_xlim((np.abs(trial_window[0]) + xlims[0]) * analysis.ophys_frame_rate,
                       (np.abs(trial_window[0]) + xlims[1]) * analysis.ophys_frame_rate)
        mean = np.round(mean_df.mean_response.values[0], 3)
        p_val = np.round(mean_df.p_value.values[0], 4)
        # sd = np.round(mean_df.sd_over_baseline.values[0], 2)
        # time_to_peak = np.round(mean_df.time_to_peak.values[0], 3)
        # fano_factor = np.round(mean_df.fano_factor.values[0], 3)
        # fraction_active_trials = np.round(mean_df.fraction_active_trials.values[0], 3)
        fraction_sig_trials = np.round(mean_df.fraction_significant_trials.values[0], 3)
        ax[i].set_title(trial_type + ' - mean: ' + str(mean) + '\np_val: ' + str(p_val) +  # ', sd: ' + str(sd) +
                        # '\ntime_to_peak: ' + str(time_to_peak) +
                        '\nfraction_sig_trials: ' + str(fraction_sig_trials))  # +
        # '\nfano_factor: ' + str(fano_factor));
    ax[1].set_ylabel('')
    if save:
        fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.7)
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'mean_response_pref_stim_metrics',
                    'metrics_' + str(cell_specimen_id))
        plt.close()
    return ax


def format_table_data(dataset):
    table_data = dataset.metadata.copy()
    table_data = table_data[['specimen_id', 'donor_id', 'targeted_structure', 'imaging_depth',
                             'experiment_date', 'cre_line', 'reporter_line', 'session_type']]
    table_data['experiment_date'] = str(table_data['experiment_date'].values[0])[:10]
    table_data = table_data.transpose()
    return table_data


def plot_images(dataset, orientation='row', rows=1, color_box=True, save_dir=None, ax=None):
    meta = dataset.stimulus_metadata
    meta = meta[meta.image_name != 'omitted']
    n_images = len(meta)
    if orientation == 'row':
        figsize = (20, 5)
        cols = n_images
        rows = 1
    if rows == 2:
        cols = cols / 2
        figsize = (10, 4)
    elif orientation == 'column':
        figsize = (5, 20)
        cols = 1
        rows = n_images
    if ax is None:
        fig, ax = plt.subplots(rows, cols, figsize=figsize)

    stimuli = dataset.stimulus_metadata
    image_names = np.sort(dataset.stimulus_presentations.image_name.unique())
    image_names = image_names[image_names != 'omitted']
    colors = sns.color_palette("hls", len(image_names))
    for i, image_name in enumerate(image_names):
        image_index = stimuli[stimuli.image_name == image_name].image_index.values[0]
        image = dataset.stimulus_template[image_index]
        ax[i].imshow(image, cmap='gray', vmin=0, vmax=np.amax(image))
        ax[i].grid('off')
        ax[i].axis('off')
        ax[i].set_title(image_name, color='k')
        if color_box:
            linewidth = 6
            ax[i].axhline(y=-20, xmin=0.04, xmax=0.95, linewidth=linewidth, color=colors[i])
            ax[i].axhline(y=image.shape[0] - 20, xmin=0.04, xmax=0.95, linewidth=linewidth, color=colors[i])
            ax[i].axvline(x=-30, ymin=0.05, ymax=0.95, linewidth=linewidth, color=colors[i])
            ax[i].axvline(x=image.shape[1], ymin=0, ymax=0.95, linewidth=linewidth, color=colors[i])
            # ax[i].set_title(str(stim_code), color=colors[i])
    if save_dir:
        title = 'images_' + dataset.metadata.session_type[0]
        if color_box:
            title = title + '_c'
        save_figure(fig, figsize, save_dir, 'images', title)
    return ax


def plot_omitted_flash_response_all_stim(analysis, cell_specimen_id, ax=None, save_dir=None, window=[-3, 3],
                                         legend=False):
    if window is None:
        window = analysis.omitted_flash_window
    fdf = analysis.stimulus_response_df
    image_names = np.sort(fdf.image_name.unique())[:-1]
    odf = analysis.omission_response_df.copy()
    # image_names = np.sort(odf.image_name.unique())
    # colors = get_colors_for_stim_codes(np.arange(0,len(image_names),1))
    if ax is None:
        figsize = (7, 5)
        fig, ax = plt.subplots(figsize=figsize)
    for image_name in image_names:
        color = ut.get_color_for_image_name(image_names, image_name)
        traces = odf[(odf.cell_specimen_id == cell_specimen_id) & (odf.image_category == image_name)].trace.values
        ax = plot_mean_trace(np.asarray(traces[:-1]), frame_rate=analysis.ophys_frame_rate,
                             legend_label=image_name, color=color, interval_sec=1, xlims=window, ax=ax)
    ax = plot_flashes_on_trace(ax, analysis, omitted=True, alpha=0.15)
    ax.set_xlabel('time (sec)')
    ax.set_title('omitted flash response')
    if legend:
        ax.legend(loc=9, bbox_to_anchor=(1.3, 1.3))
    if save_dir:
        fig.tight_layout()
        save_figure(fig, (6, 5), save_dir, 'omitted_flash_response', str(cell_specimen_id))
        plt.close()
    return ax


def plot_reward_triggered_average(dataset, cell, window=[-2, 3], variability=True, show_reliability=True,
                                  ax=None, save_figures=False, save_dir=None):
    reward_times = dataset.rewards.time.values
    trace = dataset.dff_traces_array[cell]
    cell_specimen_id = dataset.get_cell_specimen_id_for_cell_index(cell)
    responses = ut.get_responses_around_event_times(trace, dataset.ophys_timestamps, reward_times,
                                                    frame_rate=31., window=window)

    if ax is None:
        figsize = (5, 4)
        fig, ax = plt.subplots(figsize=figsize)
    if variability:
        suffix = '_variability'
        ax = plot_mean_trace_with_variability(responses, frame_rate=31., ylabel='dF/F', label=None,
                                              color=[.3, .3, .3],
                                              interval_sec=1, xlims=window, ax=ax)
    else:
        suffix = '_sem'
        ax = plot_mean_trace(responses, frame_rate=31., ylabel='dF/F', color=[.3, .3, .3], interval_sec=1,
                             xlims=window, ax=ax)
    ax.set_xlabel('time since reward (sec)')
    if show_reliability:
        reliability = ut.compute_reliability_for_traces(responses)
        ax.set_title('reliability: ' + str(np.round(reliability, 2)))
    fig.tight_layout()

    if save_figures:
        if save_dir:
            save_figure(fig, figsize, save_dir, 'reward_triggered_average',
                        dataset.analysis_folder + '_' + str(cell_specimen_id) + suffix)
        save_figure(fig, figsize, dataset.analysis_dir, 'reward_triggered_average', str(cell_specimen_id) + suffix)
        plt.close()
    return ax


def plot_lick_triggered_average(dataset, cell, window=[-2, 3], variability=True, show_reliability=False,
                                ax=None, save_figures=False, save_dir=None):
    lick_times = ut.get_unrewarded_first_lick_times(dataset)
    trace = dataset.dff_traces_array[cell]
    cell_specimen_id = dataset.get_cell_specimen_id_for_cell_index(cell)
    responses = ut.get_responses_around_event_times(trace, dataset.ophys_timestamps, lick_times,
                                                    frame_rate=31., window=window)
    responses = responses[:-1]  # last one can be trucated if at end of session

    if ax is None:
        figsize = (5, 4)
        fig, ax = plt.subplots(figsize=figsize)
    if variability:
        suffix = '_variability'
        ax = plot_mean_trace_with_variability(responses, frame_rate=31., ylabel='dF/F', label=None,
                                              color=[.3, .3, .3],
                                              interval_sec=1, xlims=window, ax=ax)
    else:
        suffix = '_sem'
        ax = plot_mean_trace(responses, frame_rate=31., ylabel='dF/F', color=[.3, .3, .3], interval_sec=1,
                             xlims=window, ax=ax)
    ax.set_xlabel('time since first lick (sec)')
    if show_reliability:
        reliability = ut.compute_reliability_for_traces(responses)
        ax.set_title('reliability: ' + str(np.round(reliability, 2)))
    fig.tight_layout()

    if save_figures:
        if save_dir:
            save_figure(fig, figsize, save_dir, 'lick_triggered_average',
                        dataset.analysis_folder + '_' + str(cell_specimen_id) + suffix)
        save_figure(fig, figsize, dataset.analysis_dir, 'lick_triggered_average', str(cell_specimen_id) + suffix)
        plt.close()
    return ax


def plot_lick_triggered_running_average(dataset, cell, window=[-2, 3], variability=True, show_reliability=False,
                                        ax=None, save_figures=False, save_dir=None):
    # window = [-2,3]
    # variability = False
    # show_reliability = False
    # ax = None
    lick_times = ut.get_unrewarded_first_lick_times(dataset)
    trace = dataset.running_speed.running_speed.values
    #     cell_specimen_id = dataset.get_cell_specimen_id_for_cell_index(cell)
    responses = ut.get_responses_around_event_times(trace, dataset.stimulus_timestamps, lick_times,
                                                    frame_rate=60, window=window)
    responses = responses[1:]  # first one is zero for some reason
    responses = responses[:-1]  # last one is truncated

    if ax is None:
        figsize = (5, 4)
        fig, ax = plt.subplots(figsize=figsize)
    if variability:
        suffix = '_variability'
        ax = plot_mean_trace_with_variability(responses, frame_rate=60., ylabel='run speed (cm/s)', label=None,
                                              color=[.3, .3, .3],
                                              interval_sec=1, xlims=window, ax=ax)
    else:
        suffix = '_sem'
        ax = plot_mean_trace(responses, frame_rate=60., ylabel='run speed (cm/s)', color=[.3, .3, .3], interval_sec=1,
                             xlims=window, ax=ax)
    ax.set_xlabel('time since first lick (sec)')
    if show_reliability:
        reliability = ut.compute_reliability_for_traces(responses)
        ax.set_title('reliability: ' + str(np.round(reliability, 2)))
    fig.tight_layout()

    if save_figures:
        if save_dir:
            save_figure(fig, figsize, save_dir, 'lick_triggered_average',
                        dataset.analysis_folder + '_running_speed' + suffix)
        save_figure(fig, figsize, dataset.analysis_dir, 'lick_triggered_average', 'running_speed' + suffix)
        plt.close()
    return ax


def diff(x):
    return x[-1] - x[0]


def plot_running_and_behavior(dataset, start_time, duration, save_figures=False, save_dir=None):
    xlim = (start_time, start_time + duration + 1)
    running_speed = dataset.running_speed.running_speed.values
    running_times = dataset.running_speed.time.values
    # running_diff = dataset.running_speed.running_speed.rolling(window=15).apply(diff)
    figsize = (15, 2)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(running_times, running_speed, color=sns.color_palette()[0])
    #     ax.plot(running_times, running_diff, color=sns.color_palette()[3])
    ax = plot_behavior_events(dataset, ax)
    ax = add_stim_color_span(dataset, ax, xlim=xlim)
    ax.set_xlim(xlim)
    ax.set_ylabel('run speed\n(cm/s)')
    ax.set_xlabel('time (seconds)')
    ax.set_xticks(np.arange(xlim[0], xlim[1], 10))
    xticklabels = np.arange(xlim[0], xlim[1], 10) - xlim[0]
    ax.set_xticklabels([int(x) for x in xticklabels])
    # fig.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.3)
    if save_figures:
        if save_dir:
            save_figure(fig, figsize, save_dir, 'running_and_behavior', dataset.analysis_folder + '_' + str(start_time))
        save_figure(fig, figsize, dataset.analysis_dir, 'running_and_behavior', str(start_time))
        plt.close()


def plot_omission_over_time(dataset, omitted_flash_response_df, cell_specimen_id, save=True):
    odf = omitted_flash_response_df.copy()
    cdf = odf[(odf.cell_specimen_id == cell_specimen_id)]

    traces = cdf.trace.values
    response_matrix = np.empty((len(traces), len(traces[0])))
    for i, trace in enumerate(traces):
        response_matrix[i, :] = trace

    figsize = (5, 6)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(response_matrix, vmin=0, vmax=np.percentile(response_matrix, 95), cmap='magma', ax=ax,
                     cbar_kws={'label': 'dF/F'})
    xticks, xticklabels = get_xticks_xticklabels(response_matrix[0, :], 31, interval_sec=1, window=[-3, 3])
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(x) for x in xticklabels])
    ax.set_yticks(np.arange(0, len(response_matrix), 30))
    ax.set_ylabel('omission number')
    ax.set_xlabel('time after omission (sec)')
    ax.set_title('cell ' + str(cell_specimen_id))
    fig.tight_layout()
    if save:
        save_figure(fig, figsize, dataset.analysis_dir, 'omission_over_time', str(cell_specimen_id))
        plt.close()


def plot_cell_summary_figure(analysis, cell_index, save=False, show=False, cache_dir=None):
    use_events = analysis.use_events
    dataset = analysis.dataset
    rdf = analysis.trials_response_df
    # fdf = analysis.stimulus_response_df
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell_index)

    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.2, .7), yspan=(0, .2))
    ax = plot_behavior_events_trace(dataset, [cell_index], xmin=600, length=2, ax=ax, save=False, use_events=use_events)
    ax.set_title(dataset.analysis_folder)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .20), yspan=(0, .22))
    ax = plot_cell_zoom(dataset.roi_masks, dataset.max_projection, cell_specimen_id, spacex=25, spacey=25,
                        show_mask=True, ax=ax)
    ax.set_title('cell ' + str(cell_index) + ' - ' + str(cell_specimen_id))

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .7), yspan=(.16, .35))
    if use_events:
        ax = plot_trace(dataset.ophys_timestamps, dataset.events[cell_index, :], ax,
                        title='cell_specimen_id: ' + str(cell_specimen_id), ylabel=ylabel)
    else:
        ax = plot_trace(dataset.ophys_timestamps, dataset.dff_traces_array[cell_index, :], ax,
                        title='cell_specimen_id: ' + str(cell_specimen_id), ylabel=ylabel)
    ax = plot_behavior_events(dataset, ax)
    ax.set_title('')

    ax = placeAxesOnGrid(fig, dim=(1, len(rdf.change_image_name.unique())), xspan=(.0, .7), yspan=(.35, .55),
                         wspace=0.35)
    if use_events:
        vmax = 0.03
    else:
        vmax = np.percentile(dataset.dff_traces_array[cell_index, :], 99.9)
    ax = plot_transition_type_heatmap(analysis, [cell_index], vmax=vmax, ax=ax, cmap='magma', colorbar=False)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .25), yspan=(.55, .75), wspace=0.25, sharex=True, sharey=True)
    # ax = plot_image_response_for_trial_types(analysis, cell_index, legend=False, save=False, ax=ax)
    ax = plot_image_change_response(analysis, cell_index, cell_index, legend=False, save=False, ax=ax)

    if 'omitted' in analysis.stimulus_response_df.image_name.unique():
        # try:
        ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.25, .5), yspan=(.55, .75))
        ax = plot_omitted_flash_response_all_stim(analysis, cell_specimen_id, ax=ax)
        # ax.legend(bbox_to_anchor=(1.4, 2))
        # except:
        #     print('cant plot omitted flashes')

    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.39, .59))
    # ax = plot_running_not_running(rdf, sdf, cell, trial_type='go', ax=ax)
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.58, 0.78))
    # ax = plot_engaged_disengaged(rdf, sdf, cell, code='change_image', trial_type='go', ax=ax)
    fig.tight_layout()

    ax = placeAxesOnGrid(fig, dim=(8, 1), xspan=(.68, .86), yspan=(.2, .99), wspace=0.25, hspace=0.25)
    try:
        ax = plot_images(dataset, orientation='column', color_box=True, save=False, ax=ax)
    except:  # NOQA E722
        pass

    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.0, 0.2), yspan=(.79, 1))
    # ax = plot_mean_cell_response_heatmap(analysis, cell, values='mean_response', index='initial_image_name',
    #                                     columns='change_image_name', save=False, ax=ax, use_events=use_events)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.5, 0.7), yspan=(0.55, 0.75))
    ax = plot_ranked_image_tuning_curve_trial_types(analysis, cell_index, ax=ax, save=False, use_events=use_events)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.5, 0.7), yspan=(0.78, 0.99))
    ax = plot_ranked_image_tuning_curve_flashes(analysis, cell_index, ax=ax, save=False, use_events=use_events)

    ax = placeAxesOnGrid(fig, dim=(1, 2), xspan=(0.0, 0.5), yspan=(.78, .99), wspace=0.25, sharex=True, sharey=True)
    ax = plot_mean_response_pref_stim_metrics(analysis, cell_index, ax=ax, save=False, use_events=use_events)

    # # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.83, 1), yspan=(.78, 1))
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, .99), yspan=(0.05, .16))
    # table_data = format_table_data(dataset)
    # xtable = ax.table(cellText=table_data.values, cellLoc='left', rowLoc='left', loc='center', fontsize=12)
    # xtable.scale(1, 3)
    # ax.axis('off');

    fig.canvas.draw()
    fig.tight_layout()
    # plt.gcf().subplots_adjust(bottom=0.05)
    if save:
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'cell_summary_plots',
                    str(cell_index) + '_' + str(cell_specimen_id) + suffix)
        if cache_dir:
            save_figure(fig, figsize, cache_dir, 'cell_summary_plots',
                        dataset.analysis_folder + '_' + str(cell_specimen_id) + '_' + str(cell_index) + suffix)
    if not show:
        plt.close()


def colormap():
    colormap = {
        'engagement_state': {
            'disengaged': 'firebrick',
            'engaged': 'olivedrab',
        },
        'lick_on_next_flash': {
            0: 'blue',
            1: 'orange',
        },
        'lick_on_previous_flash': {
            0: 'indigo',
            1: 'turquoise',
        },
    }
    return colormap


def get_title(ophys_experiment_id, cell_specimen_id):
    '''
    generate a standardized figure title containing identifying information
    '''
    experiments_table = loading.get_filtered_ophys_experiment_table().reset_index()

    row = experiments_table.query('ophys_experiment_id == @ophys_experiment_id').iloc[0].to_dict()
    title = '{}_spec_id={}_exp_id={}_{}_{}_{}_depth={}_cell_id={}'.format(
        row['cre_line'],
        row['specimen_id'],
        row['ophys_experiment_id'],
        row['equipment_name'],
        row['session_type'],
        row['targeted_structure'],
        row['imaging_depth'],
        cell_specimen_id,
    )
    return title


def make_time_summary_plot(analysis, cell_specimen_id, split_by, axes):
    '''
    plots raw F and deltaF/F with engagement state denoted by background color
    inputs:
        analysis object
        cell_specimen_id (int))
        axes (list or array): an array of axes, expected length = 3
    returns:
        None
    '''

    sdf = analysis.stimulus_response_df

    sdf_subset = sdf.drop_duplicates('start_time')[['start_time', split_by]].copy().reset_index().dropna()
    values_in_split = np.sort(sdf_subset[split_by].unique())

    sdf_subset['next_start_time'] = sdf_subset['start_time'].shift(-1)
    sdf_subset['state_change'] = sdf_subset[split_by] != sdf_subset[split_by].shift()

    state_changes = sdf_subset.query('state_change == True').copy()
    state_changes['next_state_change'] = state_changes['start_time'].shift(-1)
    state_changes.loc[state_changes.index.max(), 'next_state_change'] = sdf['start_time'].max()

    cmap = colormap()

    for ii, ax in enumerate(axes):
        ax.axvspan(0, state_changes.iloc[0]['start_time'], color='gray', alpha=0.5)
        for idx, row in state_changes.iterrows():
            ax.axvspan(row['start_time'] / 60., row['next_state_change'] / 60., color=cmap[split_by][row[split_by]])
        ax.axvspan(
            row['next_state_change'] / 60.,
            row['next_state_change'] / 60. + 5,
            color='gray',
            alpha=0.5
        )
        ax.axvspan(
            row['next_state_change'] / 60. + 5,
            analysis.dataset.ophys_timestamps.max() / 60.,
            color='yellow',
            alpha=0.5
        )

        ax.set_xlim(0, analysis.dataset.ophys_timestamps.max() / 60.)

        if ii == 0:
            ax.plot(
                analysis.dataset.ophys_timestamps / 60,
                analysis.dataset.corrected_fluorescence_traces.loc[cell_specimen_id]['corrected_fluorescence'],
                linewidth=2,
                color='black'
            )
            ax.set_ylabel('Corrected\nFluor.', rotation=0, ha='right', va='center')
            ax.set_xticks([])
            ax.set_title('engagement state vs. session time\n(gray = gray screen, {} = {}:{}, {} = {}:{}, yellow = fingerprint movie)'.format(
                cmap[split_by][values_in_split[0]],
                split_by,
                values_in_split[0],
                cmap[split_by][values_in_split[1]],
                split_by,
                values_in_split[1],
            ))
        if ii == 1:
            ax.plot(
                analysis.dataset.ophys_timestamps / 60,
                analysis.dataset.dff_traces.loc[cell_specimen_id]['dff'],
                linewidth=2,
                color='black'
            )
            ax.set_ylabel('$\Delta$F/F', rotation=0, ha='right', va='center')  # NOQA W605
            ax.set_xticks([])
        if ii == 2:
            ax.plot(
                analysis.dataset.running_speed['timestamps'] / 60,
                analysis.dataset.running_speed['speed'],
                linewidth=2,
                color='black'
            )
            ax.set_ylabel('running\nspeed\n(cm/s)', rotation=0, ha='right', va='center')
            ax.set_xlabel('session time (minutes)')


def make_cell_plot(dataset, cell_specimen_id, ax):
    roi_masks = loading.get_sdk_roi_masks(dataset.cell_specimen_table)
    ax[0].imshow(dataset.max_projection, cmap='gray')
    ax[0].set_title('max projection')

    ax[1].imshow(roi_masks[cell_specimen_id], cmap='gray')
    ax[1].set_title('ROI mask for\ncell_specimen_id {}'.format(cell_specimen_id))

    ax[2].imshow(dataset.max_projection, cmap='gray')
    v = roi_masks[cell_specimen_id].copy()
    v[v == 0] = np.nan
    ax[2].imshow(v, cmap='spring_r', alpha=0.5)
    ax[2].set_title('overlay (ROI mask in yellow)')

    for axis in ax:
        axis.axis('off')


def seaborn_lineplot(df, ax, split_by, legend='brief', xlabel='time (s)', ylabel='$\Delta$F/F', n_boot=1000):  # NOQA W605
    cmap = colormap()

    df = df[['eventlocked_timestamps', 'eventlocked_traces', split_by]].dropna()
    values_in_split = np.sort(df[split_by].unique())

    sns.lineplot(
        x='eventlocked_timestamps',
        y='eventlocked_traces',
        data=df,
        hue=split_by,
        hue_order=np.sort(values_in_split),
        palette=[cmap[split_by][state] for state in values_in_split],
        ax=ax,
        legend=legend,
        n_boot=n_boot,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0, ha='right')


def make_cell_response_summary_plot(analysis, cell_specimen_id, split_by, save=False, show=True, errorbar_bootstrap_iterations=1000):
    figure_savedir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/summary_plots/single_cell_plots/response_plots/split_by_{}'.format(split_by)
    if os.path.exists(figure_savedir) == False:
        os.mkdir(figure_savedir)
    oeid = analysis.dataset.ophys_experiment_id\

    params_dict = {
        'stimulus response': {
            'ophys_df': analysis.stimulus_response_df,
            'running_df': analysis.stimulus_run_speed_df,
            'xlims': (-0.5, 0.75),
            'omit': None,
            'pre_color': 'blue',
            'post_color': 'blue'
        },
        'omission response': {
            'ophys_df': analysis.omission_response_df,
            'running_df': analysis.omission_run_speed_df,
            'xlims': (-3, 3),
            'omit': 0,
            'pre_color': 'blue',
            'post_color': 'blue'
        },
        'change response': {
            'ophys_df': analysis.trials_response_df,
            'running_df': analysis.trials_run_speed_df,
            'pupil_df': None,
            'xlims': (-3, 3),
            'omit': None,
            'pre_color': 'gray',
            'post_color': 'green'
        },
    }
    ylabels = {
        'ophys': '$\Delta$F/F',  # NOQA W605
        'running': 'Running Speed\n(cm/s)'
    }

    fig = plt.figure(figsize=(20, 18))
    ax = {
        'state_summary': placeAxesOnGrid(fig, dim=[3, 1], xspan=[0, 1], yspan=[0, 0.2], sharex=False, hspace=0),
        'cell_images': placeAxesOnGrid(fig, dim=[1, 3], xspan=[0, 1], yspan=[0.3, 0.65], sharey=True, wspace=0),
        'ophys_response_plots': placeAxesOnGrid(fig, dim=[1, 3], xspan=[0, 1], yspan=[0.7, 0.84], sharey=True),
        'running_response_plots': placeAxesOnGrid(fig, dim=[1, 3], xspan=[0, 1], yspan=[0.86, 1], sharey=True),
    }

    make_time_summary_plot(analysis, cell_specimen_id, split_by, ax['state_summary'])
    make_cell_plot(analysis.dataset, cell_specimen_id, ax['cell_images'])

    for col, response_type in enumerate(params_dict.keys()):
        for row, datastream in enumerate(['ophys', 'running']):
            if response_type == 'stimulus response' and datastream == 'ophys':
                legend = 'brief'
            else:
                legend = False

            if datastream == 'ophys':
                data = params_dict[response_type]['{}_df'.format(datastream)].query('cell_specimen_id == @cell_specimen_id')
                ax['{}_response_plots'.format(datastream)][col].set_title(response_type)
                ax['{}_response_plots'.format(datastream)][col].set_xticklabels([])
            else:
                data = params_dict[response_type]['{}_df'.format(datastream)]
                ax['{}_response_plots'.format(datastream)][col].set_title('')

            seaborn_lineplot(
                data,
                ax['{}_response_plots'.format(datastream)][col],
                split_by=split_by,
                legend=legend,
                n_boot=errorbar_bootstrap_iterations,
                xlabel='time (s)' if row == 1 else '',
                ylabel=ylabels[datastream] if col == 0 else '',
            )

            ax['{}_response_plots'.format(datastream)][col].set_xlim(params_dict[response_type]['xlims'])
            designate_flashes(
                ax['{}_response_plots'.format(datastream)][col],
                omit=params_dict[response_type]['omit'],
                pre_color=params_dict[response_type]['pre_color'],
                post_color=params_dict[response_type]['post_color']
            )

    plt.subplots_adjust(top=0.9)
    title = get_title(oeid, cell_specimen_id)
    fig.suptitle(title)
    if save:
        fig.savefig(os.path.join(figure_savedir, title + '.png'), dpi=200)
    if not show:
        plt.close()


def designate_flashes(ax, omit=None, pre_color='blue', post_color='blue'):
    '''
    Function to add vertical spans to designate stimulus flashes

    :param ax: axis on which to plot stimulus presentation times
    :param omit: time to omit (default = None)
    :param pre_color: color of vspans before time = 0 (default = 'blue')
    :param post_color: color of vspans after time = 0 (default = 'blue')

    :return: None
    '''
    lims = ax.get_xlim()
    for flash_start in np.arange(0, lims[1], 0.75):
        if flash_start != omit:
            ax.axvspan(flash_start, flash_start + 0.25,
                       color=post_color, alpha=0.25, zorder=-np.inf)
    for flash_start in np.arange(-0.75, lims[0] - 0.001, -0.75):
        if flash_start != omit:
            ax.axvspan(flash_start, flash_start + 0.25,
                       color=pre_color, alpha=0.25, zorder=-np.inf)


def designate_flashes_plotly(fig, omit=None, pre_color='blue', post_color='blue', alpha=0.25, plotnumbers=[1], lims=[-10, 10]):
    '''add vertical spans to designate stimulus flashes'''

    post_flashes = np.arange(0, lims[1], 0.75)
    post_flash_colors = np.array([post_color] * len(post_flashes))
    pre_flashes = np.arange(-0.75, lims[0] - 0.001, -0.75)
    pre_flash_colors = np.array([pre_color] * len(pre_flashes))

    flash_times = np.hstack((pre_flashes, post_flashes))
    flash_colors = np.hstack((pre_flash_colors, post_flash_colors))

    shape_list = list(fig.layout.shapes)

    for plotnumber in plotnumbers:
        for flash_start, flash_color in zip(flash_times, flash_colors):
            if flash_start != omit:
                shape_list.append(
                    go.layout.Shape(
                        type="rect",
                        x0=flash_start,
                        x1=flash_start + 0.25,
                        y0=-100,
                        y1=100,
                        fillcolor=flash_color,
                        opacity=alpha,
                        layer="below",
                        line_width=0,
                        xref='x{}'.format(plotnumber),
                        yref='y{}'.format(plotnumber),
                    ),
                )

    fig.update_layout(shapes=shape_list)
