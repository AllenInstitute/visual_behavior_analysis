"""
Created on Sunday July 15 2018

@author: marinag
"""
import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from visual_behavior.visualization.utils import save_figure
from visual_behavior.ophys.response_analysis import utilities as ut
# from visual_behavior.visualization.ophys import population_summary_figures as psf


# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white',
              {'axes.spines.right': False, 'axes.spines.top': False, 'xtick.bottom': True, 'ytick.left': True, })
sns.set_palette('deep')


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
                                                  subplot_spec=outer_grid[int(100 * yspan[0]):int(100 * yspan[1]),
                                                               # flake8: noqa: E999
                                                               int(100 * xspan[0]):int(100 * xspan[1])], wspace=wspace,
                                                  hspace=hspace)  # flake8: noqa: E999

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


def plot_cell_zoom(roi_mask_dict, max_projection, cell_specimen_id, spacex=10, spacey=10, show_mask=False, ax=None):
    if type(roi_mask_dict.keys()[0]) == int:
        m = roi_mask_dict[int(cell_specimen_id)]
    else:
        m = roi_mask_dict[str(cell_specimen_id)]
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
    ax.set_title('cell ' + str(cell_specimen_id))
    ax.grid(False)
    ax.axis('off')
    return ax


def plot_roi_validation(roi_names,
                        roi_df,
                        roi_traces,
                        dff_traces_original,
                        cell_specimen_ids,
                        cell_indices,
                        roi_masks,
                        max_projection,
                        dff_traces,
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

        if id in cell_specimen_ids:
            cell_index = cell_indices[id]
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
            ax[4].set_xlim(xmin - 10, xmax + 10)
            ax[4].set_ylim(ymin - 10, ymax + 10)
            ax[4].grid(False)

            ax[5].plot(dff_traces[cell_index])
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
    interval_frames = int(interval_sec * frame_rate)
    n_frames = len(trace)
    n_sec = n_frames / frame_rate
    xticks = np.arange(0, n_frames + 1, interval_frames)
    xticklabels = np.arange(0, n_sec + 0.1, interval_sec)
    if not window:
        xticklabels = xticklabels - n_sec / 2
    else:
        xticklabels = xticklabels + window[0]
    return xticks, xticklabels


def plot_mean_trace(traces, frame_rate, ylabel='dF/F', legend_label=None, color='k', interval_sec=1, xlims=[-4, 4],
                    ax=None):
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
        ax.plot(trace, label=legend_label, linewidth=3, color=color)
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


def plot_flashes_on_trace(ax, analysis, window=[-4,8], trial_type=None, omitted=False, flashes=False, alpha=0.15, facecolor='gray'):
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
    elif flashes and not omitted:
        window = analysis.flash_window
    elif omitted:
        window = analysis.omitted_flash_window
    else:
        window = analysis.trial_window
    change_frame = np.abs(window[0]) * frame_rate
    end_frame = (window[1] + np.abs(window[0])) * frame_rate
    interval = blank_duration + stim_duration
    if omitted:
        array = np.arange((change_frame + interval), end_frame, interval * frame_rate)
        array = array[1:]
    else:
        array = np.arange(change_frame, end_frame, interval * frame_rate)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + (stim_duration * frame_rate)
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    if trial_type == 'go':
        alpha = alpha * 3
    else:
        alpha
    array = np.arange(change_frame - ((blank_duration) * frame_rate), 0, -interval * frame_rate)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] - (stim_duration * frame_rate)
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
    df = analysis.trial_response_df.copy()
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
    df = analysis.trial_response_df.copy()
    df = df[df.trial_type == 'go']
    images = np.sort(df.change_image_name.unique())
    images = images[images != 'omitted']
    trials = analysis.dataset.trials.copy()
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell_index)
    if ax is None:
        figsize = (5,3)
        fig, ax = plt.subplots(figsize=figsize)
    for c, change_image_name in enumerate(images):
        color = get_color_for_image_name(analysis.dataset, change_image_name)
        selected_trials = trials[(trials.change_image_name == change_image_name)].trial.values
        traces = df[(df.cell == cell_index) & (df.trial.isin(selected_trials))].trace.values
        ax = plot_mean_trace(traces, analysis.ophys_frame_rate, legend_label=None, color=color,
                             interval_sec=2, xlims=analysis.trial_window, ax=ax)
    ax = plot_flashes_on_trace(ax, analysis, trial_type='go', alpha=0.3)
    ax.set_title('cell_index: '+str(cell_index)+', cell_specimen_id: ' + str(cell_specimen_id))
    ax.set_title('cell '+str(cell_order))
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
                    analysis.dataset.analysis_folder+'_'+str(cell_specimen_id)+'_'+str(cell_order)+'_'+str(cell_index))
    if not show:
        plt.close()
    return ax


def plot_event_detection(dff_traces, events, analysis_dir):
    figsize = (20, 15)
    xlims_list = [[0, dff_traces[0].shape[0]], [2000, 10000], [60000, 62000]]
    for cell in range(len(dff_traces)):
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        ax = ax.ravel()
        for i, xlims in enumerate(xlims_list):
            ax[i].plot(dff_traces[cell], label='dF/F from L0')
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
        plt.colorbar(cax, ax=ax[i])
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'trial_trace_heatmap', 'roi_' + str(cell))
    return ax


def plot_mean_response_by_repeat(analysis, cell, save_dir=None, ax=None):
    flash_response_df = analysis.flash_response_df.copy()
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
    cbar = ax.figure.colorbar(mappable=sm, ax=ax)
    cbar.set_label('repeat')
    ax.set_title(str(cell) + '_' + analysis.dataset.analysis_folder, fontsize=14)
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'mean_response_by_repeat',
                    analysis.dataset.analysis_folder + '_' + str(cell))
        plt.close()
    return ax


def plot_mean_response_by_image_block(analysis, cell, save_dir=None, ax=None):
    flash_response_df = analysis.flash_response_df.copy()
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
    cbar = ax.figure.colorbar(mappable=sm, ax=ax)
    cbar.set_label('image_block')
    ax.set_title(str(cell) + '_' + analysis.dataset.analysis_folder, fontsize=14)
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'mean_response_by_image_block',
                    analysis.dataset.analysis_folder + '_' + str(cell))
        plt.close()
    return ax


def plot_trace(timestamps, trace, ax=None, xlabel='time (seconds)', ylabel='fluorescence', title='roi',
               color=sns.color_palette()[0]):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(timestamps, trace, color=color, linewidth=2)
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


def get_color_for_image_name(dataset, image_name):
    images = np.sort(dataset.stimulus_table.image_name.unique())
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


def add_stim_color_span(dataset, ax, xlim=None):
    # xlim should be in seconds
    if xlim is None:
        stim_table = dataset.stimulus_table.copy()
    else:
        stim_table = dataset.stimulus_table.copy()
        stim_table = stim_table[(stim_table.start_time >= xlim[0]) & (stim_table.end_time <= xlim[1])]
    if 'omitted' in stim_table.keys():
        stim_table = stim_table[stim_table.omitted == False].copy()
    for idx in stim_table.index:
        start_time = stim_table.loc[idx]['start_time']
        end_time = stim_table.loc[idx]['end_time']
        image_name = stim_table.loc[idx]['image_name']
        color = get_color_for_image_name(dataset, image_name)
        addSpan(ax, start_time, end_time, color=color)
    return ax


def plot_behavior_events(dataset, ax, behavior_only=False, linewidth=2):
    lick_times = dataset.licks.time.values
    reward_times = dataset.rewards.time.values
    if behavior_only:
        lick_y = 0
        reward_y = 0.25
        ax.set_ylim([-0.5, 1])
    else:
        ymin, ymax = ax.get_ylim()
        lick_y = ymin + (ymax * 0.05)
        reward_y = ymin + (ymax * 0.1)
        ax.set_ylim([ymin*1.3, ymax])
    lick_y_array = np.empty(len(lick_times))
    lick_y_array[:] = lick_y
    reward_y_array = np.empty(len(reward_times))
    reward_y_array[:] = reward_y
    ax.plot(lick_times, lick_y_array, '|', color='g', markeredgewidth=linewidth, label='licks')
    ax.plot(reward_times, reward_y_array, 'o', markerfacecolor='purple', markeredgecolor='purple', markeredgewidth=linewidth/10.,
            label='rewards')
    return ax


def plot_behavior(dataset, cache_dir):
    flashes = dataset.stimulus_table.copy()
    flashes = ut.annotate_flashes_with_reward_rate(dataset)
    figsize = (15,4)
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
    save_figure(fig, figsize, os.path.join(cache_dir, 'multi_session_summary_figures'), 'behavior', dataset.analysis_folder)


def plot_behavior_annotated(dataset, xmin=1800, duration=20, plot_reward_rate=False, cache_dir=None, show=True):
    xlim = (xmin, xmin + duration)
    all_trials = dataset.all_trials.copy()
    trial_starts = all_trials[all_trials.trial_type!='aborted'].starttime.values
    ts = trial_starts[(trial_starts>xlim[0])]
    ts = ts[(ts<xlim[1])]
    abort_starts = all_trials[all_trials.trial_type=='aborted'].starttime.values
    ab = abort_starts[(abort_starts>xlim[0])]
    ab = ab[(ab<xlim[1])]
    flashes = dataset.stimulus_table.copy()
    flashes = ut.annotate_flashes_with_reward_rate(dataset)
    figsize = (15,4)
    fig, ax = plt.subplots(figsize=figsize)
    ax = add_stim_color_span(dataset, ax=ax, xlim=xlim)
    ax.plot(dataset.stimulus_timestamps, dataset.running_speed.running_speed.values, label='run speed', linewidth=2)
    ax.set_ylabel('running speed(cm/s)')
    if plot_reward_rate:
        ax2 = ax.twinx()
        ax2.plot(flashes.start_time.values, flashes.reward_rate.values, color=sns.color_palette()[4], label='reward rate')
        ax2.set_ylabel('reward rate')
        sns.despine(ax=ax2, right=False)
    ax = plot_behavior_events(dataset, ax, linewidth=4)
    ymin, ymax = ax.get_ylim()
    for a in ab:
        ax.vlines(x=a, ymin=ymin, ymax=ymax, color='red', linewidth=2)
    if len(ab)>0:
        a = ab[0]
        ax.vlines(x=a, ymin=ymin, ymax=ymax, color='red', linewidth=2, label='aborted trial start')
    for t in ts:
        ax.vlines(x=t, ymin=ymin, ymax=ymax, color='green', linewidth=2)
    if len(ts)>0:
        t = ts[0]
        ax.vlines(x=t, ymin=ymin, ymax=ymax, color='green', linewidth=2, label='go/catch trial start')
    ax.set_xlabel('time in session (seconds)')
    ax.set_title(dataset.analysis_folder)
    ax.set_xlim(xlim)
    ax.legend(loc='upper left', fontsize='x-small')
#     ax.legend(bbox_to_anchor=(0.05,1.1), ncol=5, loc='upper left')
    fig.tight_layout()
    fig.subplots_adjust(top=.8)
    save_figure(fig, figsize, dataset.analysis_dir, 'behavior', 'behavior_'+str(xlim[0]))
    if cache_dir:
        save_figure(fig, figsize, os.path.join(cache_dir, 'multi_session_summary_figures'), 'behavior', dataset.analysis_folder+'_'+str(xlim[0]))
    if not show:
        plt.close()

def restrict_axes(xmin, xmax, interval, ax):
    xticks = np.arange(xmin, xmax, interval)
    ax.set_xticks(xticks)
    ax.set_xlim([xmin, xmax])
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
            ax = plot_trace(dataset.ophys_timestamps, dataset.dff_traces[cell_index, :], ax,
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


def plot_example_traces_and_behavior(dataset, cell_indices, xmin_seconds, length_mins, save_figures=False,
                                     include_running=False, cell_label=False, use_events=False, save_dir=None):
    if use_events:
        traces = dataset.events
        cell_label = True
        suffix = '_events'
    else:
        traces = dataset.dff_traces
        suffix = ''
    if include_running:
        n = 2
    else:
        n = 1
    interval_seconds = 20
    xmax_seconds = xmin_seconds + (length_mins * 60) + 1
    xlim = [xmin_seconds, xmax_seconds]

    if len(cell_indices) < 10:
        figsize = (12, 8.3)
    else:
        figsize = (12, 10)
    fig, ax = plt.subplots(len(cell_indices) + n, 1, figsize=figsize, sharex=True)
    ax = ax.ravel()

    ymins = []
    ymaxs = []
    for i, cell_index in enumerate(cell_indices):
        ax[i] = plot_trace(dataset.ophys_timestamps, traces[cell_index, :], ax=ax[i],
                           title='', ylabel=str(cell_index), color=[.5, .5, .5])
        ax[i] = add_stim_color_span(dataset, ax=ax[i], xlim=xlim)
        ax[i] = restrict_axes(xmin_seconds, xmax_seconds, interval_seconds, ax=ax[i])
        ax[i].set_xticks(np.arange(xmin_seconds, xmax_seconds, interval_seconds))
        xticklabels = np.arange(xmin_seconds, xmax_seconds, interval_seconds) - xmin_seconds
        ax[i].set_xticklabels([int(x) for x in xticklabels])
        ax[i].set_xlabel('')
        ymin, ymax = ax[i].get_ylim()
        ymins.append(ymin)
        ymaxs.append(ymax)
        ax[i].set_ylabel('')
        ax[i].set_yticklabels('')
        sns.despine(ax=ax[i], left=True, bottom=False)
        #         ymin, ymax = ax[i].get_ylim()
        if 'Vip' in dataset.metadata.cre_line.values[0]:
            ax[i].vlines(x=xmin_seconds - 0.05, ymin=0, ymax=2, linewidth=4)
            ax[i].set_ylim(ymin=-0.5, ymax=5)
        elif 'Slc' in dataset.metadata.cre_line.values[0]:
            ax[i].vlines(x=xmin_seconds - 0.05, ymin=0, ymax=1, linewidth=4)
            ax[i].set_ylim(ymin=-0.5, ymax=3)

            #     for i, cell_index in enumerate(cell_indices):
            #         ax[i].set_ylim([np.amin(ymins), np.amax(ymaxs)])

    i += 1
    ax[i] = plot_behavior_events(dataset, ax=ax[i], behavior_only=True)
    ax[i] = add_stim_color_span(dataset, ax=ax[i], xlim=xlim)
    ax[i].set_xlim(xlim)
    ax[i].set_ylabel('')
    ax[i].axes.get_yaxis().set_visible(False)
    # ax[i].legend(loc='upper left', fontsize=14)
    sns.despine(ax=ax[i], left=True)
    l = ax[i].legend(fontsize='small', loc='upper left')
    plt.setp(l.get_title(), fontsize='small')

    if include_running:
        i += 1
        ax[i].plot(dataset.stimulus_timestamps, dataset.running_speed.running_speed.values)
        ax[i] = add_stim_color_span(dataset, ax=ax[i], xlim=xlim)
        ax[i] = restrict_axes(xmin_seconds, xmax_seconds, interval_seconds, ax=ax[i])
        #         ax[i].set_ylabel('run speed\n(cm/s)')
        #         ax[i].axes.get_yaxis().set_visible(False)
        ax[i].set_ylabel('')
        ax[i].set_yticklabels('')
        sns.despine(ax=ax[i], left=True)
        ax[i].vlines(x=-0.5, ymin=0, ymax=30, linewidth=4)

    ax[i].set_xlabel('time (seconds)')
    ax[0].set_title(dataset.analysis_folder)
    # fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_figures:
        save_figure(fig, figsize, dataset.analysis_dir, 'example_traces',
                       str(dataset.experiment_id) + '_' + str(xlim[0]) + suffix)
        save_figure(fig, figsize, dataset.cache_dir, 'example_traces',
                       str(dataset.experiment_id) + '_' + str(xlim[0]) + suffix)
        if save_dir:
            save_figure(fig, figsize, save_dir, 'example_traces',
                           str(dataset.experiment_id) + '_' + str(xlim[0]) + suffix)
        plt.close()


def plot_average_flash_response_example_cells(analysis, active_cell_indices, save_figures=False, save_dir=None, folder=None, ax=None):
    dataset = analysis.dataset
    fdf = analysis.flash_response_df.copy()
    last_flash = fdf.flash_number.unique()[-1] #sometimes last flash is truncated
    fdf = fdf[fdf.flash_number!=last_flash]

    conditions=['cell_specimen_id', 'image_name']
    mdf = ut.get_mean_df(fdf, analysis, conditions=conditions, flashes=True)

    cell_specimen_ids = [dataset.get_cell_specimen_id_for_cell_index(cell_index) for cell_index in active_cell_indices]

    image_names = np.sort(mdf.image_name.unique())

    if ax is None:
        if len(active_cell_indices)<10:
            figsize = (12, 8.3)
        else:
            figsize = (12, 10)
        fig, ax = plt.subplots(len(cell_specimen_ids), len(image_names), figsize=figsize, sharex=True)
        ax = ax.ravel()

    i = 0
    for c, cell_specimen_id in enumerate(cell_specimen_ids):
        cell_data = mdf[(mdf.cell_specimen_id==cell_specimen_id)]
        maxs = [np.amax(trace) for trace in cell_data.mean_trace.values]
        ymax = np.amax(maxs)*1.2
        for m, image_name in enumerate(image_names):
            cdf = cell_data[(cell_data.image_name==image_name)]
            color = ut.get_color_for_image_name(image_names, image_name)
            ax[i] = plot_mean_trace_from_mean_df(cdf, color=[.5,.5,.5], interval_sec=0.5,
                                                     xlims=analysis.flash_window, ax=ax[i])
            ax[i] = plot_flashes_on_trace(ax[i], analysis, flashes=True, facecolor=color, alpha=0.3)
            if 'Vip' in dataset.metadata.cre_line.values[0]:
                ax[i].vlines(x=0, ymin=0, ymax=.25, linewidth=4)
            elif 'Slc' in dataset.metadata.cre_line.values[0]:
                ax[i].vlines(x=0, ymin=0, ymax=.25, linewidth=4)
    #         sns.despine(ax=ax[i])
            ax[i].axis('off')
            ax[i].set_ylim(-0.05, ymax)
            if m == 0:
                ax[i].set_ylabel('x')
            if c == 0:
                ax[i].set_title(image_name)
            if c == len(cell_specimen_ids):
                ax[i].set_xlabel('time (s)')
            i+=1

    # fig.tight_layout()
    if save_figures:
        if save_dir:
            save_figure(fig ,figsize, save_dir, folder, dataset.analysis_folder)
        save_figure(fig ,figsize, analysis.dataset.analysis_dir, 'example_traces_all_flashes', dataset.analysis_folder)
        plt.close()


def plot_reliability_trials(analysis, tdf, mean_tdf, cell, xlims=[-4,8], save_figures=False, save_dir=None, folder=None):
    cell_specimen_id = tdf[tdf.cell==cell].cell_specimen_id.values[0]
    figsize=(4,3)
    fig, ax = plt.subplots(figsize=figsize)
    traces = tdf[(tdf.cell==cell)&(tdf.pref_stim==True) &(tdf.trial_type=='go')].trace.values
    ax = plot_mean_trace_with_variability(traces, frame_rate=31., ylabel='dF/F', label=None, color=[.3,.3,.3], interval_sec=2,
                                         xlims=xlims, ax=ax)
    ax = plot_flashes_on_trace(ax, analysis, trial_type='go', window=xlims)
    reliability = mean_tdf[(mean_tdf.cell==cell)&(mean_tdf.pref_stim==True)&(mean_tdf.trial_type=='go')].reliability.values[0]
    ax.set_title('reliability: '+str(np.round(reliability,2)))
    fig.tight_layout()

    if save_figures:
        save_figure(fig ,figsize, save_dir, folder, analysis.dataset.analysis_folder+'_reliability_trials_'+str(cell_specimen_id))
        plt.close()


def plot_reliability_flashes(analysis, fdf, mean_fdf, cell, xlims=[-0.5, 0.75], save_figures=False, save_dir=None, folder=None):
    cell_specimen_id = fdf[fdf.cell==cell].cell_specimen_id.values[0]
    figsize=(4,3)
    fig, ax = plt.subplots(figsize=figsize)
    traces = fdf[(fdf.cell==cell)&(fdf.pref_stim==True) &(fdf.repeat==1)].trace.values
    ax = plot_mean_trace_with_variability(traces, frame_rate=31., ylabel='dF/F', label=None, color=[.3,.3,.3], interval_sec=0.5,
                                         xlims=xlims, ax=ax)
    ax = plot_flashes_on_trace(ax, analysis, flashes=True, window=xlims)
    reliability = mean_fdf[(mean_fdf.cell==cell)&(mean_fdf.pref_stim==True)&(mean_fdf.repeat==1)].reliability.values[0]
    ax.set_title('reliability: '+str(np.round(reliability,2)))

    fig.tight_layout()
    if save_figures:
        save_figure(fig ,figsize, save_dir, folder, analysis.dataset.analysis_folder+'_reliability_flashes_'+str(cell_specimen_id))
        plt.close()


def get_colors_for_response_types(values):
    c = sns.color_palette()
    colors_dict = {'HIT': c[2], 'MISS': c[8], 'CR': c[0], 'FA': c[3]}
    colors = []
    for val in values:
        colors.append(colors_dict[val])
    return colors


def plot_transition_type_heatmap(analysis, cell_list, cmap='jet', vmax=None, save=False, ax=None, colorbar=True):
    response_types = ['HIT', 'MISS', 'FA', 'CR']
    df = analysis.trial_response_df.copy()
    images = np.sort(df.change_image_name.unique())
    rows = 1
    cols = int(len(images) / float(rows))
    figsize = (15, 4 * rows)
    colors = get_colors_for_response_types(response_types)
    for cell in cell_list:
        cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
        if ax is None:
            fig, ax = plt.subplots(rows, cols, figsize=figsize, sharex=True);
            ax = ax.ravel();
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
                    cax = ax[i].pcolormesh(response_matrix, cmap=cmap, vmax=vmax, vmin=0);
                else:
                    cax = ax[i].pcolormesh(response_matrix, cmap=cmap);
                ax[i].set_ylim(0, response_matrix.shape[0]);
                ax[i].set_xlim(0, response_matrix.shape[1]);
                ax[i].set_yticks(segments);
                ax[i].set_yticklabels('')
                ax[i].set_xlabel('time (s)');
                xticks, xticklabels = get_xticks_xticklabels(im_df.trace.values[0], analysis.ophys_frame_rate,
                                                             interval_sec=2, window=analysis.trial_window)
                xticklabels = [int(label) for label in xticklabels]
                ax[i].set_xticks(xticks);
                ax[i].set_xticklabels(xticklabels);
                ax[i].set_title(image_name);
            for s in range(len(segments) - 1):
                ax[i].vlines(x=-10, ymin=segments[s], ymax=segments[s + 1], color=colors[s], linewidth=25)
            ax[0].set_ylabel('trials')
            resp_types.append(response_type_list)
            if colorbar:
                plt.colorbar(cax, ax=ax[i]);
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
    df = analysis.trial_response_df.copy()
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
    tdf = ut.get_mean_df(analysis.trial_response_df, analysis,
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
    ax.set_xticklabels(images, rotation=90);
    ax.legend()
    ax.set_title('lifetime sparseness go: ' + str(np.round(ls_list[0], 3)) + '\nlifetime sparseness catch: ' + str(
        np.round(ls_list[1], 3)));
    if save:
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'lifetime_sparseness' + suffix,
                    'trial_types_tc_' + str(cell_specimen_id))
    return ax


def plot_ranked_image_tuning_curve_all_flashes(analysis, cell, ax=None, save=None, use_events=False):
    from scipy.stats import sem as compute_sem
    from visual_behavior.ophys.response_analysis import utilities as ut
    c = sns.color_palette()
    colors = [c[3], c[0], c[2]]
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    fdf = analysis.flash_response_df.copy()
    fdf = fdf[fdf.omitted == False].copy()
    fmdf = ut.get_mean_df(fdf, analysis, conditions=['cell_specimen_id', 'image_name'], flashes=True)
    if ax is None:
        figsize = (6, 4)
        fig, ax = plt.subplots(figsize=figsize)
    tmp = fdf[(fdf.cell_specimen_id == cell_specimen_id)]
    responses = fmdf[(fmdf.cell_specimen_id == cell_specimen_id)].mean_response.values
    ls = compute_lifetime_sparseness(responses)
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
    ax.set_xticklabels(images, rotation=90);
    ax.set_title('lifetime sparseness all flashes: ' + str(np.round(ls, 3)));
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
    fdf = analysis.flash_response_df.copy()
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
    ax.set_xticklabels(images, rotation=90);
    ax.set_title('lifetime sparseness repeat ' + str(repeats[0]) + ': ' + str(np.round(ls_list[0], 3)) +
                 '\nlifetime sparseness repeat ' + str(repeats[1]) + ': ' + str(np.round(ls_list[1], 3)) +
                 '\nlifetime sparseness repeat ' + str(repeats[2]) + ': ' + str(np.round(ls_list[2], 3)))
    ax.legend()
    if save:
        fig.tight_layout()
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'lifetime_sparseness_flashes' + suffix,
                    str(cell_specimen_id))
    return ax


def plot_mean_trace_from_mean_df(mean_df, ophys_frame_rate, label=None, color='k', interval_sec=1, xlims=(2, 6),
                                 ax=None, use_events=False):
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    if ax is None:
        fig, ax = plt.subplots()
    mean_trace = mean_df.mean_trace.values[0]
    sem = mean_df.sem_trace.values[0]
    times = np.arange(0, len(mean_trace), 1)
    ax.plot(mean_trace, label=label, linewidth=3, color=color)
    ax.fill_between(times, mean_trace + sem, mean_trace - sem, alpha=0.5, color=color)
    xticks, xticklabels = get_xticks_xticklabels(mean_trace, ophys_frame_rate, interval_sec=interval_sec, window=xlims)
    ax.set_xticks(xticks);
    ax.set_xticklabels(xticklabels);
    ax.set_xlim([xlims[0] * ophys_frame_rate, (np.abs(xlims[0])+xlims[1]) * ophys_frame_rate])
    ax.set_xlabel('time (sec)')
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
        times = np.arange(0, len(mean_trace), 1)
        sem = (traces.std()) / np.sqrt(float(len(traces)))
        for trace in traces:
            ax.plot(trace, linewidth=1, color='gray')
        ax.plot(mean_trace, label=label, linewidth=3, color=color, zorder=100)
        xticks, xticklabels = get_xticks_xticklabels(mean_trace, frame_rate, interval_sec, window=xlims)
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(x) for x in xticklabels])
        ax.set_xlim(0,(np.abs(xlims[0])+xlims[1]) * int(frame_rate))
        ax.set_xlabel('time (sec)')
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
    return ax


def plot_mean_response_pref_stim_metrics(analysis, cell, ax=None, save=None, use_events=False):
    import visual_behavior.ophys.response_analysis.utilities as ut
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    tdf = analysis.trial_response_df
    tdf = tdf[tdf.cell_specimen_id == cell_specimen_id]
    fdf = analysis.flash_response_df
    fdf = fdf[fdf.cell_specimen_id == cell_specimen_id]
    mdf = ut.get_mean_df(analysis.trial_response_df, analysis,
                         conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    mdf = mdf[mdf.cell_specimen_id == cell_specimen_id]
    trial_window = analysis.trial_window
    xlims = [-2, 2]
    if ax is None:
        figsize = (12, 6)
        fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
        ax = ax.ravel()
    # pref_image = fdf[fdf.pref_stim == True].image_name.values[0]
    pref_image = tdf[tdf.pref_stim == True].change_image_name.values[0]
    images = ut.get_image_names(mdf)
    color = ut.get_color_for_image_name(images, pref_image)
    for i, trial_type in enumerate(['go', 'catch']):
        tmp = tdf[(tdf.trial_type == trial_type) & (tdf.change_image_name == pref_image)]
        mean_df = mdf[(mdf.trial_type == trial_type) & (mdf.change_image_name == pref_image)]
        ax[i] = plot_mean_trace_with_variability(tmp.trace.values, analysis.ophys_frame_rate, label=None, color=color,
                                                 interval_sec=1, xlims=trial_window, ax=ax[i])
        ax[i] = plot_flashes_on_trace(ax[i], analysis, trial_type=trial_type, omitted=False, alpha=.05 * 8)
        ax[i].set_xlim((np.abs(trial_window[0])+xlims[0])*analysis.ophys_frame_rate,
                       (np.abs(trial_window[0])+xlims[1])*analysis.ophys_frame_rate)
        mean = np.round(mean_df.mean_response.values[0], 3)
        p_val = np.round(mean_df.p_value.values[0], 4)
        sd = np.round(mean_df.sd_over_baseline.values[0], 2)
        time_to_peak = np.round(mean_df.time_to_peak.values[0], 3)
        fano_factor = np.round(mean_df.fano_factor.values[0], 3)
        fraction_active_trials = np.round(mean_df.fraction_active_trials.values[0], 3)
        ax[i].set_title(trial_type + ' - mean: ' + str(mean) + '\np_val: ' + str(p_val) + ', sd: ' + str(sd) +
                        # '\ntime_to_peak: ' + str(time_to_peak) +
                        '\nfraction_active_trials: ' + str(fraction_active_trials) +
                        '\nfano_factor: ' + str(fano_factor));
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


def get_color_for_image_name(dataset, image_name):
    images = np.sort(dataset.stimulus_table.image_name.unique())
    if 'omitted' in images:
        images = images[images != 'omitted']
    colors = sns.color_palette("hls", len(images))
    image_index = np.where(images == image_name)[0][0]
    color = colors[image_index]
    return color


def plot_images(dataset, orientation='row', rows=1, color_box=True, save=False, ax=None):
    orientation = 'row'
    if orientation == 'row':
        figsize = (20, 5)
        cols = len(dataset.stimulus_metadata)
        rows = 1
    if rows == 2:
        cols = cols / 2
        figsize = (10, 4)
    elif orientation == 'column':
        figsize = (5, 20)
        cols = 1
        rows = len(dataset.stim_codes.stim_code.unique())
    if ax is None:
        fig, ax = plt.subplots(rows, cols, figsize=figsize)
        ax = ax.ravel();

    stimuli = dataset.stimulus_metadata
    image_names = np.sort(dataset.stimulus_table.image_name.unique())
    image_names = image_names[image_names != 'omitted']
    colors = sns.color_palette("hls", len(image_names))
    for i, image_name in enumerate(image_names):
        image_index = stimuli[stimuli.image_name == image_name].image_index.values[0]
        image = dataset.stimulus_template[image_index]
        ax[i].imshow(image, cmap='gray', vmin=0, vmax=np.amax(image));
        ax[i].grid('off')
        ax[i].axis('off')
        ax[i].set_title(image_name, color='k');
        if color_box:
            linewidth = 6
            ax[i].axhline(y=-20, xmin=0.04, xmax=0.95, linewidth=linewidth, color=colors[i]);
            ax[i].axhline(y=image.shape[0] - 20, xmin=0.04, xmax=0.95, linewidth=linewidth, color=colors[i]);
            ax[i].axvline(x=-30, ymin=0.05, ymax=0.95, linewidth=linewidth, color=colors[i]);
            ax[i].axvline(x=image.shape[1], ymin=0, ymax=0.95, linewidth=linewidth, color=colors[i]);
            # ax[i].set_title(str(stim_code), color=colors[i])
    if save:
        title = 'images_' + str(rows)
        if color_box:
            title = title + '_c'
        save_figure(fig, figsize, dataset.analysis_dir, 'images', title, formats=['.png'])
    return ax


def plot_omitted_flash_response_all_stim(analysis, cell_specimen_id, ax=None, save_dir=None, window=[-3,3], legend=False):
    if window is None:
        window = analysis.omitted_flash_window
    fdf = analysis.flash_response_df
    image_names = np.sort(fdf.image_name.unique())[:-1]
    odf = analysis.omitted_flash_response_df.copy()
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
    trace = dataset.dff_traces[cell]
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
    trace = dataset.dff_traces[cell]
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

def plot_lick_triggered_running_average(dataset, cell, window=[-2,3], variability=True, show_reliability=False,
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
    responses = responses[1:] #first one is zero for some reason
    responses = responses[:-1] #last one is truncated

    if ax is None:
        figsize=(5,4)
        fig, ax = plt.subplots(figsize=figsize)
    if variability:
        suffix = '_variability'
        ax = plot_mean_trace_with_variability(responses, frame_rate=60., ylabel='run speed (cm/s)', label=None, color=[.3,.3,.3],
                                                 interval_sec=1, xlims=window, ax=ax)
    else:
        suffix = '_sem'
        ax = plot_mean_trace(responses, frame_rate=60., ylabel='run speed (cm/s)', color=[.3,.3,.3], interval_sec=1,
                                         xlims=window, ax=ax)
    ax.set_xlabel('time since first lick (sec)')
    if show_reliability:
        reliability = ut.compute_reliability_for_traces(responses)
        ax.set_title('reliability: '+str(np.round(reliability,2)))
    fig.tight_layout()

    if save_figures:
        if save_dir:
            save_figure(fig, figsize, save_dir, 'lick_triggered_average', dataset.analysis_folder+'_running_speed'+suffix)
        save_figure(fig, figsize, dataset.analysis_dir, 'lick_triggered_average', 'running_speed'+suffix)
        plt.close()
    return ax


def diff(x):
    return x[-1] - x[0]

def plot_running_and_behavior(dataset, start_time, duration, save_figures=False, save_dir=None):
    xlim = (start_time,start_time+duration+1)
    running_speed = dataset.running_speed.running_speed.values
    running_times = dataset.running_speed.time.values
    running_diff = dataset.running_speed.running_speed.rolling(window=15).apply(diff)
    figsize = (15,2)
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
            save_figure(fig, figsize, save_dir, 'running_and_behavior', dataset.analysis_folder+'_'+str(start_time))
        save_figure(fig, figsize, dataset.analysis_dir, 'running_and_behavior', str(start_time))
        plt.close()

def plot_omission_over_time(dataset, omitted_flash_response_df, cell_specimen_id, save=True):
    odf = omitted_flash_response_df.copy()
    cdf = odf[(odf.cell_specimen_id==cell_specimen_id)]

    traces = cdf.trace.values
    response_matrix = np.empty((len(traces),len(traces[0])))
    for i,trace in enumerate(traces):
        response_matrix[i,:] = trace

    figsize = (5,6)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(response_matrix, vmin=0, vmax=np.percentile(response_matrix,95), cmap='magma', ax=ax, cbar_kws={'label': 'dF/F'})
    xticks, xticklabels = get_xticks_xticklabels(response_matrix[0,:], 31, interval_sec=1, window=[-3,3])
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(x) for x in xticklabels])
    ax.set_yticks(np.arange(0,len(response_matrix), 30))
    ax.set_ylabel('omission number')
    ax.set_xlabel('time after omission (sec)')
    ax.set_title('cell '+str(cell_specimen_id))
    fig.tight_layout()
    if save:
        save_figure(fig, figsize, dataset.analysis_dir, 'omission_over_time', str(cell_specimen_id))
        plt.close()

def plot_cell_summary_figure(analysis, cell_index, save=False, show=False, cache_dir=None):
    use_events = analysis.use_events
    dataset = analysis.dataset
    rdf = analysis.trial_response_df
    fdf = analysis.flash_response_df
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell_index)

    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.2, .7), yspan=(0, .2))
    ax = plot_behavior_events_trace(dataset, [cell_index], xmin=600, length=2, ax=ax, save=False, use_events=use_events)
    ax.set_title(dataset.analysis_folder)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .20), yspan=(0, .22))
    ax = plot_cell_zoom(dataset.roi_mask_dict, dataset.max_projection, cell_specimen_id, spacex=25, spacey=25,
                        show_mask=True, ax=ax)
    ax.set_title('cell ' + str(cell_index) + ' - ' + str(cell_specimen_id))

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .7), yspan=(.16, .35))
    if use_events:
        ax = plot_trace(dataset.ophys_timestamps, dataset.events[cell_index, :], ax,
                        title='cell_specimen_id: ' + str(cell_specimen_id), ylabel=ylabel)
    else:
        ax = plot_trace(dataset.ophys_timestamps, dataset.dff_traces[cell_index, :], ax,
                        title='cell_specimen_id: ' + str(cell_specimen_id), ylabel=ylabel)
    ax = plot_behavior_events(dataset, ax)
    ax.set_title('')

    ax = placeAxesOnGrid(fig, dim=(1, len(rdf.change_image_name.unique())), xspan=(.0, .7), yspan=(.33, .55),
                         wspace=0.35)
    if use_events:
        vmax = 0.03
    else:
        vmax = np.percentile(dataset.dff_traces[cell_index, :], 99.9)
    ax = plot_transition_type_heatmap(analysis, [cell_index], vmax=vmax, ax=ax, cmap='magma', colorbar=False)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .25), yspan=(.53, .75), wspace=0.25, sharex=True, sharey=True)
    # ax = plot_image_response_for_trial_types(analysis, cell_index, legend=False, save=False, ax=ax)
    ax = plot_image_change_response(analysis, cell_index, cell_index, legend=False, save=False, ax=ax)

    if 'omitted' in analysis.flash_response_df.image_name.unique():
        # try:
        ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.25, .5), yspan=(.53, .75))
        ax = plot_omitted_flash_response_all_stim(analysis, cell_specimen_id, ax=ax)
        # ax.legend(bbox_to_anchor=(1.4, 2))
        # except:
        #     print('cant plot omitted flashes')

    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.0, .2))
    # ax = plot_mean_trace_behavioral_response_types_pref_image(rdf, sdf, cell, behavioral_response_types=['HIT', 'MISS'],
    #                                                              ax=ax)
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.2, .4))
    # ax = plot_mean_trace_behavioral_response_types_pref_image(rdf, sdf, cell, behavioral_response_types=['FA', 'CR'],
    #                                                              ax=ax)
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.39, .59))
    # ax = plot_running_not_running(rdf, sdf, cell, trial_type='go', ax=ax)
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.58, 0.78))
    # ax = plot_engaged_disengaged(rdf, sdf, cell, code='change_image', trial_type='go', ax=ax)
    fig.tight_layout()

    ax = placeAxesOnGrid(fig, dim=(8, 1), xspan=(.68, .86), yspan=(.2, .99), wspace=0.25, hspace=0.25)
    try:
        ax = plot_images(dataset, orientation='column', color_box=True, save=False, ax=ax);
    except:
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

    fig.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.05)
    if save:
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'cell_summary_plots',
                    str(cell_index) + '_' + str(cell_specimen_id) + suffix)
        if cache_dir:
            save_figure(fig, figsize, cache_dir, 'cell_summary_plots',
                        dataset.analysis_folder + '_' + str(cell_specimen_id) + '_' + str(cell_index) + suffix)
    if not show:
        plt.close()
