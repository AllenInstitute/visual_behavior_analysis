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


#
# def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png']):
#     fig_dir = os.path.join(save_dir, folder)
#     if not os.path.exists(fig_dir):
#         os.mkdir(fig_dir)
#     mpl.rcParams['pdf.fonttype'] = 42
#     fig.set_size_inches(figsize)
#     filename = os.path.join(fig_dir, fig_title)
#     for f in formats:
#         fig.savefig(filename + f, transparent=True, orientation='landscape')


def plot_lick_raster(trials, response_window=[0.15, 0.75],ax=None, save_dir=None):
    from visual_behavior.translator.core.annotate import colormap
    import visual_behavior.data_access.reformat as reformat
    trials = reformat.add_trial_type_to_trials_table(trials)
    if ax is None:
        figsize = (5, 10)
        fig, ax = plt.subplots(figsize=figsize)
    for trial, trials_id in enumerate(trials.trials_id.values):
        trial_data = trials.loc[trials_id]
        trial_type = trial_data.trial_type
        # get times relative to change time
        trial_start = trial_data.start_time - trial_data.change_time
        lick_times = [(t - trial_data.change_time) for t in trial_data.lick_times]
        reward_time = trial_data.reward_time - trial_data.change_time
        # plot trials as colored rows
        ax.axhspan(trial, trial + 1, -200, 200, color=colormap(trial_type, 'trial_types'), alpha=.5)
        # plot reward times
        if np.isnan(reward_time)==False:
            ax.plot(reward_time, trial + 0.5, '.', color='b', label='reward', markersize=6)
        ax.vlines(trial_start, trial, trial + 1, color='black', linewidth=1)
        # plot lick times
        ax.vlines(lick_times, trial, trial + 1, color='k', linewidth=1)
        # annotate change time
        ax.vlines(0, trial, trial + 1, color=[.5, .5, .5], linewidth=1)
    # gray bar for response window
    ax.axvspan(response_window[0], response_window[1], facecolor='gray', alpha=.4, edgecolor='none')
    ax.grid(False)
    ax.set_ylim(0, len(trials))
    ax.set_xlim([-1, 4])
    ax.set_ylabel('trials')
    ax.set_xlabel('time (sec)')
    ax.set_title('lick raster')
    # plt.gca().invert_yaxis()
    if save_dir:
        save_figure(fig, figsize, save_dir, 'behavior', 'lick_raster')
    return ax


def reorder_traces(original_traces, analysis):
    tdf = analysis.trials_response_df
    df = ut.get_mean_df(tdf, analysis, conditions=['cell', 'change_image_name'])

    images = np.sort(df.change_image_name.unique())

    cell_list = []
    for image in images:
        tmp = df[(df.change_image_name == image) & (df.pref_stim == True)]
        order = np.argsort(tmp.mean_response.values)[::-1]
        cell_ids = list(tmp.cell.values[order])
        cell_list = cell_list + cell_ids

    reordered_traces = []
    for cell_index in cell_list:
        reordered_traces.append(original_traces[cell_index, :])
    return np.asarray(reordered_traces)


def plot_sorted_traces_heatmap(dataset, analysis, ax=None, save=False, use_events=False):
    import visual_behavior.ophys.response_analysis.response_processing as rp
    if use_events:
        traces = dataset.events_array.copy()
        traces = rp.filter_events_array(traces, scale=2)
        traces = reorder_traces(traces, analysis)
        vmax = 0.01
        # vmax = np.percentile(traces, 99)
        label = 'event magnitude'
        suffix = '_events'
    else:
        traces = dataset.dff_traces_array
        traces = reorder_traces(traces, analysis)
        vmax = np.percentile(traces, 99)
        label = 'dF/F'
        suffix = ''
    if ax is None:
        figsize = (14, 5)
        fig, ax = plt.subplots(figsize=figsize)

    cax = ax.pcolormesh(traces, cmap='magma', vmin=0, vmax=vmax)
    ax.set_ylabel('cells')

    interval_seconds = 5 * 60
    ophys_frame_rate = int(dataset.metadata.ophys_frame_rate.values[0])
    upper_limit, time_interval, frame_interval = get_upper_limit_and_intervals(traces, dataset.ophys_timestamps,
                                                                               ophys_frame_rate)
    ax.set_xticks(np.arange(0, upper_limit, interval_seconds * ophys_frame_rate))
    ax.set_xticklabels(np.arange(0, upper_limit / ophys_frame_rate, interval_seconds))
    ax.set_xlabel('time (seconds)')

    cb = plt.colorbar(cax, pad=0.015)
    cb.set_label(label, labelpad=3)
    if save:
        save_figure(fig, figsize, dataset.analysis_dir, 'experiment_summary',
                    str(dataset.experiment_id) + 'sorted_traces_heatmap' + suffix)
    return ax


def plot_traces_heatmap(dataset, ax=None, save=False, use_events=False):
    if use_events:
        traces = dataset.events_array.copy()
        vmax = 0.03
        # vmax = np.percentile(traces, 99)
        label = 'event magnitude'
        suffix = '_events'
    else:
        traces = dataset.dff_traces_array
        vmax = np.percentile(traces, 99)
        label = 'dF/F'
        suffix = ''
    if ax is None:
        figsize = (14, 5)
        fig, ax = plt.subplots(figsize=figsize)
    cax = ax.pcolormesh(traces, cmap='magma', vmin=0, vmax=vmax)
    ax.set_ylabel('cells')

    interval_seconds = 5 * 60
    ophys_frame_rate = int(dataset.metadata['ophys_frame_rate'])
    upper_limit, time_interval, frame_interval = get_upper_limit_and_intervals(traces, dataset.ophys_timestamps,
                                                                               ophys_frame_rate)
    ax.set_xticks(np.arange(0, upper_limit, interval_seconds * ophys_frame_rate))
    ax.set_xticklabels(np.arange(0, upper_limit / ophys_frame_rate, interval_seconds))
    ax.set_xlabel('time (seconds)')

    cb = plt.colorbar(cax, pad=0.015)
    cb.set_label(label, labelpad=3)
    if save:
        save_figure(fig, figsize, dataset.analysis_dir, 'experiment_summary',
                    str(dataset.experiment_id) + 'traces_heatmap' + suffix)
    return ax


def plot_mean_image_response_heatmap(mean_df, title=None, ax=None, save_dir=None, use_events=False):
    df = mean_df.copy()
    images = np.sort(df.change_image_name.unique())
    if 'cell_specimen_id' in df.keys():
        cell_name = 'cell_specimen_id'
    else:
        cell_name = 'cell'
    cell_list = []
    for image in images:
        tmp = df[(df.change_image_name == image) & (df.pref_stim == True)]
        order = np.argsort(tmp.mean_response.values)[::-1]
        cell_ids = list(tmp[cell_name].values[order])
        cell_list = cell_list + cell_ids

    response_matrix = np.empty((len(cell_list), len(images)))
    for i, cell in enumerate(cell_list):
        responses = []
        for image in images:
            response = df[(df[cell_name] == cell) & (df.change_image_name == image)].mean_response.values[0]
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
        save_figure(fig, figsize, save_dir, 'experiment_summary', 'mean_image_response_heatmap' + suffix)


def plot_mean_trace_heatmap(mean_df, condition='trial_type', condition_values=['go', 'catch'], ax=None, save_dir=None,
                            use_events=False, window=[-4, 4]):
    data = mean_df[mean_df.pref_stim == True].copy()
    if use_events:
        vmax = 0.05
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
                cells = im_df.cell_specimen_id.unique()[order]
            len_trace = len(im_df.mean_trace.values[0])
            response_array = np.empty((len(cells), len_trace))
            for x, cell in enumerate(cells):
                tmp = im_df[im_df.cell_specimen_id == cell]
                if len(tmp) >= 1:
                    trace = tmp.mean_trace.values[0]
                else:
                    trace = np.empty((len_trace))
                    trace[:] = np.nan
                response_array[x, :] = trace

            sns.heatmap(data=response_array, vmin=0, vmax=vmax, ax=ax[i], cmap='magma', cbar=False)
            xticks, xticklabels = sf.get_xticks_xticklabels(trace, 31., interval_sec=2, window=window)
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


def get_upper_limit_and_intervals(traces, ophys_timestamps, ophys_frame_rate):
    upper = np.round(traces.shape[1], -3) + 1000
    interval = 5 * 60  # use 5 min interval
    frame_interval = np.arange(0, traces.shape[1], interval * ophys_frame_rate)
    time_interval = np.uint64(np.round(np.arange(ophys_timestamps[0], ophys_timestamps[-1], interval), 1))
    return upper, time_interval, frame_interval


def plot_run_speed(running_speed, stimulus_timestamps, ax=None, label=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(stimulus_timestamps, running_speed, color='gray')
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
    hr, cr, d_prime = vbut.get_response_rates(trials, sliding_window=100)

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


def plot_metrics_mask(dataset, metrics, cell_list, metric_name, max_image=True, cmap='RdBu', ax=None, save=False,
                      colorbar=False):
    # roi_dict = dataset.roi_dict.copy()
    roi_mask_array = dataset.roi_mask_array.copy()
    if cmap == 'hls':
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(sns.color_palette('hls', 8))
        vmin = 0
        vmax = 8
    else:
        vmin = np.amin(metrics)
        vmax = np.amax(metrics)
    if ax is None:
        figsize = (10, 10)
        fig, ax = plt.subplots(figsize=figsize)
    if max_image is True:
        ax.imshow(dataset.max_projection, cmap='gray', vmin=0, vmax=np.amax(dataset.max_projection))
    for roi in cell_list:
        tmp = roi_mask_array[roi, :, :].copy()
        mask = np.empty(tmp.shape, dtype=np.float)
        mask[:] = np.nan
        mask[tmp == 1] = metrics[roi]
        cax = ax.imshow(mask, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
        ax.set_title(metric_name)
        ax.grid(False)
        ax.axis('off')
    if colorbar:
        plt.colorbar(cax, ax=ax)
    if save:
        plt.tight_layout()
        sf.save_figure(fig, figsize, dataset.analysis_dir, fig_title=metric_name, folder='experiment_summary')
    return ax


def plot_mean_first_flash_response_by_image_block(analysis, save_dir=None, ax=None):
    fdf = analysis.stimulus_response_df
    fdf.image_block = [int(image_block) for image_block in fdf.image_block.values]
    data = fdf[(fdf.repeat == 1) & (fdf.pref_stim == True)]
    mean_response = data.groupby(['cell_specimen_id']).apply(ut.get_mean_sem)
    mean_response = mean_response.unstack()

    cell_order = np.argsort(mean_response.mean_response.values)
    if ax is None:
        figsize = (15, 5)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.pointplot(data=data, x="image_block", y="mean_response", kind="point", hue='cell_specimen_id',
                       hue_order=cell_order,
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
                       hue='cell_specimen_id', hue_order=cell_order)
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


def plot_roi_masks(dataset, save=False):
    figsize = (20, 10)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.ravel()

    ax[0].imshow(dataset.max_projection, cmap='gray', vmin=0, vmax=np.amax(dataset.max_projection))
    ax[0].axis('off')
    ax[0].set_title('max intensity projection')

    metrics = np.empty(len(dataset.cell_indices))
    metrics[:] = -1
    cell_list = dataset.cell_indices
    plot_metrics_mask(dataset, metrics, cell_list, 'roi masks', max_image=True, cmap='hls', ax=ax[1], save=False,
                      colorbar=False)

    plt.suptitle(dataset.analysis_folder, fontsize=16, x=0.5, y=1., horizontalalignment='center')
    if save:
        save_figure(fig, figsize, dataset.analysis_dir, 'experiment_summary', dataset.analysis_folder + '_roi_masks')
        save_figure(fig, figsize, dataset.cache_dir, 'roi_masks', dataset.analysis_folder + '_roi_masks')


def plot_average_flash_response_example_cells(analysis, save_figures=False, save_dir=None, folder=None, ax=None):
    import visual_behavior.ophys.response_analysis.utilities as ut
    fdf = analysis.stimulus_response_df
    last_flash = fdf.flash_number.unique()[-1]  # sometimes last flash is truncated
    fdf = fdf[fdf.flash_number != last_flash]

    conditions = ['cell_specimen_id', 'image_name']
    mdf = ut.get_mean_df(fdf, analysis, conditions=conditions, flashes=True)

    active_cell_indices = ut.get_active_cell_indices(analysis.dataset.dff_traces_array)
    random_order = np.arange(0, len(active_cell_indices), 1)
    np.random.shuffle(random_order)
    active_cell_indices = active_cell_indices[random_order]
    cell_specimen_ids = [analysis.dataset.get_cell_specimen_id_for_cell_index(cell_index) for cell_index in
                         active_cell_indices]

    image_names = np.sort(mdf.image_name.unique())

    if ax is None:
        figsize = (12, 10)
        fig, ax = plt.subplots(len(cell_specimen_ids), len(image_names), figsize=figsize, sharex=True)
        ax = ax.ravel()

    i = 0
    for c, cell_specimen_id in enumerate(cell_specimen_ids):
        cell_data = mdf[(mdf.cell_specimen_id == cell_specimen_id)]
        maxs = [np.amax(trace) for trace in cell_data.mean_trace.values]
        ymax = np.amax(maxs) * 1.2
        for m, image_name in enumerate(image_names):
            cdf = cell_data[(cell_data.image_name == image_name)]
            color = ut.get_color_for_image_name(image_names, image_name)
            #             ax[i] = psf.plot_mean_trace_from_mean_df(cdf, 31., color=sns.color_palette()[0], interval_sec=0.5,
            #                                                      xlims=analysis.flash_window, ax=ax[i])
            ax[i] = sf.plot_mean_trace_from_mean_df(cdf, analysis.ophys_frame_rate,
                                                    color=sns.color_palette()[0], interval_sec=0.5,
                                                    xlims=analysis.flash_window, ax=ax[i])
            ax[i] = sf.plot_flashes_on_trace(ax[i], analysis, flashes=True, facecolor=color, alpha=0.3)
            #             ax[i] = psf.plot_flashes_on_trace(ax[i], flashes=True, facecolor=color, window=analysis.flash_window, alpha=0.3)
            ax[i].vlines(x=-0.05, ymin=0, ymax=0.1, linewidth=3)
            #         sns.despine(ax=ax[i])
            ax[i].axis('off')
            ax[i].set_ylim(-0.05, ymax)
            if m == 0:
                ax[i].set_ylabel('x')
            if c == 0:
                ax[i].set_title(image_name)
            if c == len(cell_specimen_ids):
                ax[i].set_xlabel('time (s)')
            i += 1

    # fig.tight_layout()
    if save_figures:
        if save_dir:
            sf.save_figure(fig, figsize, save_dir, folder, analysis.dataset.analysis_folder)
        sf.save_figure(fig, figsize, analysis.dataset.analysis_dir, 'example_traces_all_flashes',
                       analysis.dataset.analysis_folder)


def plot_experiment_summary_figure(analysis, save_dir=None):
    use_events = analysis.use_events
    if use_events:
        traces = analysis.dataset.events_array.copy()
        suffix = '_events'
    else:
        traces = analysis.dataset.dff_traces_array.copy()
        suffix = ''

    interval_seconds = 600
    ophys_frame_rate = int(analysis.ophys_frame_rate)

    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.8, 0.95), yspan=(0, .3))
    table_data = format_table_data(analysis.dataset)
    xtable = ax.table(cellText=table_data.values, cellLoc='left', rowLoc='left', loc='center', fontsize=12)
    xtable.scale(1.5, 3)
    ax.axis('off')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .22), yspan=(0, .27))
    # metrics = dataset.cell_indices
    metrics = np.empty(len(analysis.dataset.cell_indices))
    metrics[:] = -1
    cell_list = analysis.dataset.cell_indices
    plot_metrics_mask(analysis.dataset, metrics, cell_list, 'cell masks', max_image=True, cmap='hls', ax=ax, save=False,
                      colorbar=False)
    # ax.imshow(analysis.dataset.max_projection, cmap='gray', vmin=0, vmax=np.amax(analysis.dataset.max_projection))
    ax.set_title(analysis.dataset.experiment_id)
    ax.axis('off')

    upper_limit, time_interval, frame_interval = get_upper_limit_and_intervals(traces,
                                                                               analysis.dataset.ophys_timestamps,
                                                                               analysis.ophys_frame_rate)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.22, 0.9), yspan=(0, .3))
    # ax = plot_traces_heatmap(analysis.dataset, ax=ax, use_events=use_events)
    ax = plot_sorted_traces_heatmap(analysis.dataset, analysis, ax=ax, use_events=use_events)
    ax.set_xticks(np.arange(0, upper_limit, interval_seconds * ophys_frame_rate))
    ax.set_xticklabels(np.arange(0, upper_limit / ophys_frame_rate, interval_seconds))
    ax.set_xlabel('time (seconds)')
    ax.set_title(analysis.dataset.analysis_folder)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.22, 0.8), yspan=(.26, .41))
    ax = plot_run_speed(analysis.dataset.running_speed.running_speed, analysis.dataset.stimulus_timestamps, ax=ax,
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
    try:
        mdf = ut.get_mean_df(analysis.trials_response_df, analysis,
                             conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'])
        ax = plot_mean_trace_heatmap(mdf, condition='behavioral_response_type',
                                     condition_values=['HIT', 'MISS', 'CR', 'FA'], ax=ax, save_dir=None,
                                     use_events=use_events, window=analysis.trial_window)
    except BaseException:
        pass

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.78, 0.97), yspan=(.3, .8))
    mdf = ut.get_mean_df(analysis.trials_response_df, analysis, conditions=['cell_specimen_id', 'change_image_name'])
    ax = plot_mean_image_response_heatmap(mdf, title=None, ax=ax, save_dir=None, use_events=use_events)

    # fig.canvas.draw()
    fig.tight_layout()

    if save_dir:
        save_figure(fig, figsize, save_dir, 'experiment_summary_figures',
                    str(analysis.dataset.experiment_id) + '_experiment_summary' + suffix)
