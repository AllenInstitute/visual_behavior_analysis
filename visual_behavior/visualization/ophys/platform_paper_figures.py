"""
Created on Thursday September 23 2021

@author: marinag
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.reformat as reformat
import visual_behavior.data_access.utilities as utilities
import visual_behavior.ophys.response_analysis.utilities as ut
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.ophys.response_analysis.response_processing as rp
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
sns.set_palette('deep')


def plot_population_averages_for_conditions(multi_session_df, data_type, event_type, axes_column, hue_column, hue_order=None,
                                            project_code=None, timestamps=None, palette=None, title=None, suptitle=None,
                                            horizontal=True, xlim_seconds=None, save_dir=None, folder=None, suffix='', ax=None):
    """
    Plots population average response across cells in a multi session dataframe for various conditions defined by
    axes_column and hue_column, which are columns in the multi_session_df.
    There will be one axis for each value of axes_column in the multi_session_df (ex: axes_column='cre_line').
    Within each axis, data will be further split and colorized based on the values of hue_column (ex: hue_column='session_type')

    multi_session_df is created by vba.ophys.io.create_multi_session_df.get_multi_session_df(), which aggregates the output of
    vba.ophys.response_analysis.utilities.get_mean_df(), which takes the average across trials of a stimulus_response_df for a set of conditions.
    stimulus_response_df comes from mindscope_utilities.visual_behavior_ophys.data_formatting.get_stimulus_response_df()
    or vba.data_access.loading.get_stimulus_response_df()


    :param multi_session_df: dataframe containing trial averaged responses for a set of conditions, aggregated over multiple cells and sessions
    :param data_type: can be ['dff', 'events', 'filtered_events', 'running_speed', 'pupil_width', 'lick_rate]
                        must be the same value as was used to create the stimulus_response_df that was used to create multi_session_df
    :param event_type: can be either ['changes', 'omissions', 'all'], whichever was used to create the stimulus_response_df
                        that was used to create multi_session_df
    :param axes_column: column in multi_session_df to split data by to plot on each axis
    :param hue_column: column in multi_session_df to colorize data by within each axis
    :param hue_order: order of hue values. If None, will sort hue values for each axis.
    :param project_code: project_code string used for filename when saving plot
    :param timestamps: timestamps to use. If None, will use timestamps available in multi_session_df (if any)
    :param palette: color palette for hue labels. If None, uses experience_level_colors
    :param metadata_as_title: if True, creates a title composed of mouse_id, container_id, cre_line, imaging_depth, and targeted_structure
                            If False, use axes_column value.
    :param suptitle: suptitle for entire figure; if None is provided, title will be auto generated
    :param horizontal: Boolean, Whether to plot axes in horizontal dimension, if False, plot vertical
    :param xlim_seconds: time window around the event of interest to limit plot xaxis to. value must be less than the time_window used to create stimulus_response_df.
                        If None, infers xlims from timestamps
    :param save_dir: top level directory to save figure to
    :param folder: folder within save_dir to save to
    :param suffix: string to append to filename
    :param ax: if axis is provided, plot on that axis, otherwise generate a new figure and axes
    :return: ax
    """
    if palette is None:
        palette = utils.get_experience_level_colors()

    sdf = multi_session_df.copy()
    if 'trace_timestamps' in sdf.keys():
        timestamps = sdf.trace_timestamps.values[0]
    elif timestamps is not None:
        timestamps = timestamps
    else:
        print('provide timestamps or provide a multi_session_df with a trace_timestamps column')

    if project_code is not None:
        # remove traces with incorrect length - why does this happen?
        sdf = sdf.reset_index(drop=True)
        indices = [index for index in sdf.index if len(sdf.iloc[index].mean_trace) == len(sdf.mean_trace.values[100])]
        sdf = sdf.loc[indices]

    if xlim_seconds is None:
        xlim_seconds = [timestamps[0], timestamps[-1]]
    if 'dff' in data_type:
        ylabel = 'dF/F'
    elif 'events' in data_type:
        ylabel = 'population response'
    elif 'pupil' in data_type:
        ylabel = data_type+'\n normalized'
    elif 'running' in data_type:
        ylabel = 'running speed (cm/s)'
    else:
        ylabel = 'response'
    if event_type == 'omissions':
        omitted = True
        change = False
        xlabel = 'time after omission (s)'
    elif event_type == 'changes':
        omitted = False
        change = True
        xlabel = 'time after change (s)'
    else:
        omitted = False
        change = False
        xlabel = 'time (s)'

    if axes_column == 'experience_level':
        axes_conditions = ['Familiar', 'Novel 1', 'Novel >1']
    else:
        axes_conditions = np.sort(sdf[axes_column].unique())[::-1]
    # if there is only one axis condition, set n conditions for plotting to 2 so it can still iterate
    if len(axes_conditions) == 1:
        n_axes_conditions = 2
    else:
        n_axes_conditions = len(axes_conditions)
    if ax is None:
        format_fig = True
        if horizontal:
            figsize = (3 * n_axes_conditions, 3)
            fig, ax = plt.subplots(1, n_axes_conditions, figsize=figsize, sharey=False)
        else:
            figsize = (5, 3.5 * n_axes_conditions)
            fig, ax = plt.subplots(n_axes_conditions, 1, figsize=figsize, sharex=True)
    else:
        format_fig = False
    for i, axis in enumerate(axes_conditions):
        # set hue order here in case each axis has different values for hue_column
        if not hue_order:
            if hue_column == 'experience_level':
                hue_conditions = ['Familiar', 'Novel 1', 'Novel >1']
            else:
                hue_conditions = np.sort(sdf[(sdf[axes_column] == axis)][hue_column].unique())
        else:
            hue_conditions = hue_order
        # now plot for each unique hue value
        for c, hue in enumerate(hue_conditions):
            # try:
            cdf = sdf[(sdf[axes_column] == axis) & (sdf[hue_column] == hue)]
            traces = cdf.mean_trace.values
            #             traces = [trace for trace in traces if np.amax(trace) < 4]
            ax[i] = utils.plot_mean_trace(np.asarray(traces), timestamps, ylabel=ylabel,
                                          legend_label=hue, color=palette[c], interval_sec=1,
                                          xlim_seconds=xlim_seconds, ax=ax[i])
            ax[i] = utils.plot_flashes_on_trace(ax[i], timestamps, change=change, omitted=omitted)
            if omitted:
                omission_color = sns.color_palette()[9]
                ax[i].axvline(x=0, ymin=0, ymax=1, linestyle='--', color=omission_color)
            if metadata_as_title:
                metadata_string = utils.get_container_metadata_string(utils.get_metadata_for_row_of_multi_session_df(cdf))
                ax[i].set_title(metadata_string)
            else:
                ax[i].set_title(axis)
            ax[i].set_xlim(xlim_seconds)
            ax[i].set_xlabel(xlabel, fontsize=16)
            if horizontal:
                ax[i].set_ylabel('')
            else:
                ax[i].set_ylabel(ylabel)
                ax[i].set_xlabel('')
            # except:
            #     print('no data for', axis, hue)
    if format_fig:
        if horizontal:
            ax[0].set_ylabel(ylabel)
        else:
            ax[i].set_xlabel(xlabel)
    # ax[0].legend(loc='upper left', fontsize='xx-small')

    if project_code:
        if suptitle is None:
            suptitle = 'population average - ' + data_type + ' response - ' + project_code[14:]
    else:
        if suptitle is None:
            suptitle = 'population average response - ' + data_type + '_' + event_type
    if format_fig:
        plt.suptitle(suptitle, x=0.52, y=1.04, fontsize=18)
        fig.tight_layout()
        if horizontal:
            fig.subplots_adjust(wspace=0.3)

    if save_dir:
        fig_title = 'population_average_' + axes_column + '_' + hue_column + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)

    return ax


def get_timestamps_for_response_df_type(cache, experiment_id, df_name):
    """
    get timestamps from response_df
    """

    dataset = cache.get_behavior_ophys_experiment(experiment_id)
    analysis = ResponseAnalysis(dataset)
    response_df = analysis.get_response_df(df_name=df_name)
    timestamps = response_df.trace_timestamps.values[0]
    print(len(timestamps))

    return timestamps


def get_fraction_responsive_cells(multi_session_df, conditions=['cell_type', 'experience_level'], responsiveness_threshold=0.1):
    """
    Computes the fraction of cells for each condition with fraction_significant_p_value_gray_screen > responsiveness_threshold
    :param multi_session_df: dataframe of trial averaged responses for each cell for some set of conditions
    :param conditions: conditions defined by columns in df over which to group to quantify fraction responsive cells
    :param responsiveness_threshold: threshold on fraction_significant_p_value_gray_screen to determine whether a cell is responsive or not
    :return:
    """
    df = multi_session_df.copy()
    total_cells = df.groupby(conditions).count()[['cell_specimen_id']].rename(columns={'cell_specimen_id':'total_cells'})
    responsive = df[df.fraction_significant_p_value_gray_screen>responsiveness_threshold].copy()
    responsive_cells = responsive.groupby(conditions).count()[['cell_specimen_id']].rename(columns={'cell_specimen_id':'responsive_cells'})
    fraction = total_cells.merge(responsive_cells, on=conditions, how='left')  # need to use 'left' to prevent dropping of NaN values
    # set sessions with no responsive cells (NaN) to zero
    fraction.loc[fraction[fraction.responsive_cells.isnull()].index.values, 'responsive_cells'] = 0
    fraction['fraction_responsive'] = fraction.responsive_cells/fraction.total_cells
    return fraction


def plot_fraction_responsive_cells(multi_session_df, responsiveness_threshold=0.1, horizontal=True, ylim=(0,1),
                                   save_dir=None, folder=None, suffix='', ax=None):
    """
    Plots the fraction of responsive cells across cre lines
    :param multi_session_df: dataframe of trial averaged responses for each cell for some set of conditions
    :param df_name: name of the type of response_df used to make multi_session_df, such as 'omission_response_df' or 'stimulus_response_df'
    :param responsiveness_threshold: threshold on fraction_significant_p_value_gray_screen to determine whether a cell is responsive or not
    :param save_dir: directory to save figures to. if None, will not save.
    :param folder: folder within save_dir to save figures to
    :param suffix: string starting with '_' to append to end of filename of saved plot
    :return:
    """
    df = multi_session_df.copy()

    experience_levels = np.sort(df.experience_level.unique())
    cell_types = np.sort(df.cell_type.unique())[::-1]

    fraction_responsive = get_fraction_responsive_cells(df, conditions=['cell_type', 'experience_level', 'ophys_container_id', 'ophys_experiment_id'],
                                                        responsiveness_threshold=responsiveness_threshold)
    fraction_responsive = fraction_responsive.reset_index()

    palette = utils.get_experience_level_colors()
    if ax is None:
        format_fig = True
        if horizontal:
            figsize = (9, 3.5)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False)
        else:
            figsize = (3, 8)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    else:
        format_fig = False
    for i, cell_type in enumerate(cell_types):
        data = fraction_responsive[fraction_responsive.cell_type==cell_type]
        for ophys_container_id in data.ophys_container_id.unique():
            ax[i] = sns.pointplot(data=data[data.ophys_container_id==ophys_container_id], x='experience_level', y='fraction_responsive',
                     color='gray', join=True, markers='.', scale=0.25, errwidth=0.25, ax=ax[i], zorder=500)
        plt.setp(ax[i].collections, alpha=.3) #for the markers
        plt.setp(ax[i].lines, alpha=.3)
        ax[i] = sns.pointplot(data=data, x='experience_level', y='fraction_responsive', hue='experience_level',
                     hue_order=experience_levels, palette=palette, dodge=0, join=False, ax=ax[i])
        ax[i].set_xticklabels(experience_levels, rotation=45)
        ax[i].set_ylabel('fraction\nresponsive')
    #     ax[i].legend(fontsize='xx-small', title='')
        ax[i].get_legend().remove()
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('')
        if ylim is not None:
            ax[i].set_ylim(0,1)

    if format_fig:
        fig.tight_layout()
        fig_title ='fraction_responsive_cells' + suffix
        plt.suptitle(fig_title, x=0.52, y=1.02, fontsize=16)

    if save_dir and format_fig:
        fig_title = 'fraction_responsive_cells' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)

    return ax


def plot_average_metric_value_for_experience_levels_across_containers(df, metric, ylim=None, horizontal=True,
                                                                      save_dir=None, folder=None, suffix='', ax=None):
    """
    Plots the average metric value across experience levels for each cre line in color,
    with individual containers shown as connected gray lines

    :param df: dataframe with columns ['cell_type', 'experience_level', 'ophys_container_id', 'ophys_experiment_id']
                and a column with some metric value to compute the mean of, such as 'mean_response' or 'reliability'
                if 'cell_specimen_id' is included in the dataframe, will average across cells per experiment / container for the plot
    :param ylim: ylimits, in units of metric value provided, to constrain the plot.
    :param save_dir: directory to save figures to. if None, will not save.
    :param folder: sub folder of save_dir to save figures to
    :param suffix: string starting with '_' to append to end of filename of saved plot
    :return:
    """

    experience_levels = np.sort(df.experience_level.unique())
    cell_types = np.sort(df.cell_type.unique())[::-1]

    # get mean value per container
    mean_df = df.groupby(['cell_type', 'experience_level', 'ophys_container_id', 'ophys_experiment_id']).mean()[
        [metric]].reset_index()

    palette = utils.get_experience_level_colors()
    if ax is None:
        format_fig = True
        if horizontal:
            figsize = (10, 4)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False)
        else:
            figsize = (3.5, 10.5)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    else:
        format_fig = False
    for i, cell_type in enumerate(cell_types):
        data = mean_df[mean_df.cell_type == cell_type]
        # plot each container as gray lines
        for ophys_container_id in data.ophys_container_id.unique():
            ax[i] = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x='experience_level',
                                  y=metric,
                                  color='gray', join=True, markers='.', scale=0.25, errwidth=0.25, ax=ax[i], zorder=500)
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
        plt.setp(ax[i].lines, alpha=.3)
        # plot the population average in color
        ax[i] = sns.pointplot(data=data, x='experience_level', y=metric, hue='experience_level',
                              hue_order=experience_levels, palette=palette, dodge=0, join=False, ax=ax[i])
        ax[i].set_xticklabels(experience_levels, rotation=45)
        #     ax[i].legend(fontsize='xx-small', title='')
        ax[i].get_legend().remove()
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('')
        if ylim is not None:
            ax[i].set_ylim(ylim)
    if format_fig:
        fig.tight_layout()
        fig_title = metric + '_across_containers' + suffix
        plt.suptitle(fig_title, x=0.52, y=1.02, fontsize=16)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)
    return ax


def plot_n_segmented_cells(multi_session_df, df_name, horizontal=True, save_dir=None, folder='cell_matching', suffix='', ax=None):
    """
    Plots the fraction of responsive cells across cre lines
    :param multi_session_df: dataframe of trial averaged responses for each cell for some set of conditions
    :param df_name: name of the type of response_df used to make multi_session_df, such as 'omission_response_df' or 'stimulus_response_df'
    :param responsiveness_threshold: threshold on fraction_significant_p_value_gray_screen to determine whether a cell is responsive or not
    :param save_dir: directory to save figures to. if None, will not save.
    :param suffix: string starting with '_' to append to end of filename of saved plot
    :return:
    """
    df = multi_session_df.copy()

    experience_levels = np.sort(df.experience_level.unique())
    cell_types = np.sort(df.cell_type.unique())[::-1]

    fraction_responsive = get_fraction_responsive_cells(df, conditions=['cell_type', 'experience_level', 'ophys_container_id', 'ophys_experiment_id'])
    fraction_responsive = fraction_responsive.reset_index()

    palette = utils.get_experience_level_colors()
    if ax is None:
        format_fig = True
        if horizontal:
            figsize = (10, 4)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False)
        else:
            figsize = (3.5, 10.5)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    else:
        format_fig = False

    for i, cell_type in enumerate(cell_types):
        data = fraction_responsive[fraction_responsive.cell_type==cell_type]
        for ophys_container_id in data.ophys_container_id.unique():
            ax[i] = sns.pointplot(data=data[data.ophys_container_id==ophys_container_id], x='experience_level', y='total_cells',
                     color='gray', join=True, markers='.', scale=0.25, errwidth=0.25, ax=ax[i], zorder=500)
        plt.setp(ax[i].collections, alpha=.3) #for the markers
        plt.setp(ax[i].lines, alpha=.3)
        ax[i] = sns.pointplot(data=data, x='experience_level', y='total_cells', hue='experience_level',
                     hue_order=experience_levels, palette=palette, dodge=0, join=False, ax=ax[i])
        ax[i].set_xticklabels(experience_levels, rotation=45)
    #     ax[i].legend(fontsize='xx-small', title='')
        ax[i].get_legend().remove()
        ax[i].set_title(cell_type)
        ax[i].set_ylim(ymin=0)
        ax[i].set_xlabel('')
#         ax[i].set_ylim(0,1)
    if format_fig:
        fig.tight_layout()
    if save_dir:
        fig_title = df_name.split('-')[0] + '_n_total_cells' + suffix
        utils.save_figure(fig, figsize, save_dir, 'n_segmented_cells', fig_title)


def plot_mean_response_by_epoch(df, metric='mean_response', horizontal=True, ymin=0, ylabel='mean response', estimator=np.mean,
                                save_dir=None, folder='epochs', suffix='', ax=None):
    """
    Plots the mean metric value across 10 minute epochs within a session
    :param df: dataframe of cell activity with one row per cell_specimen_id / ophys_experiment_id
                must include columns 'cell_type', 'experience_level', 'epoch', and a column for the metric provided (ex: 'mean_response')
    :param metric: metric value to average over epochs; must be a column of df
    :param save_dir: top level directory to save figure to
    :param folder: folder within save_dir to save figure to; will create folder if it doesnt exist
    :param suffix: string to append at end of saved filename
    :return:
    """

    # get rid of short 7th epoch (just a few mins at end of session)
    df = df[df.epoch != 6]

    # add experience epoch column in case it doesnt already exist
    if 'experience_epoch' not in df.keys():
        def merge_experience_epoch(row):
            return row.experience_level + ' epoch ' + str(int(row.epoch) + 1)
        df['experience_epoch'] = df[['experience_level', 'epoch']].apply(axis=1, func=merge_experience_epoch)

    xticks = [experience_epoch.split(' ')[-1] for experience_epoch in np.sort(df.experience_epoch.unique())]

    cell_types = np.sort(df.cell_type.unique())[::-1]
    experience_epoch = np.sort(df.experience_epoch.unique())
    experience_levels = np.sort(df.experience_level.unique())


    palette = utils.get_experience_level_colors()
    if ax is None:
        format_fig = True
        if horizontal:
            figsize = (13, 3.5)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=True)
        else:
            figsize = (4.5, 10.5)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    else:
        format_fig = False

    for i, cell_type in enumerate(cell_types):
        data = df[df.cell_type==cell_type]
        ax[i] = sns.pointplot(data=data, x='experience_epoch', y=metric, hue='experience_level', hue_order=experience_levels,
                           order=experience_epoch, palette=palette, ax=ax[i], estimator=estimator)
        if ymin is not None:
            ax[i].set_ylim(ymin=ymin)
        ax[i].set_title(cell_type)
        ax[i].set_ylabel(ylabel)
        ax[i].set_xlabel('')
        ax[i].get_legend().remove()
        ax[i].set_xticklabels(xticks, fontsize=14)
        ax[i].vlines(x=5.5, ymin=0, ymax=1, color='gray', linestyle='--')
        ax[i].vlines(x=11.5, ymin=0, ymax=1, color='gray', linestyle='--')
    ax[i].set_xlabel('10 min epoch within session', fontsize=14)
    if format_fig:
        plt.suptitle(metric+' over time', x=0.52, y=1.03, fontsize=18)
        fig.tight_layout()
    if save_dir:
        fig_title = metric + '_epochs' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)
    return ax


def plot_cell_response_heatmap(data, timestamps, xlabel='time after change (s)', vmax=0.05,
                               microscope='Multiscope', cbar=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.heatmap(data, cmap='binary', linewidths=0, linecolor='white', square=False,
                     vmin=0, vmax=vmax, robust=True, cbar=cbar,
                     cbar_kws={"drawedges": False, "shrink": 1, "label": 'response'}, ax=ax)

    zero_index = np.where(timestamps==0)[0][0]
    ax.vlines(x=zero_index, ymin=0, ymax=len(data), color='gray', linestyle='--')

    # if microscope == 'Multiscope':
    #     ax.set_xticks(np.arange(0, 10 * 11, 11))
    #     ax.set_xticklabels(np.arange(-5, 5, 1))
    # ax.set_xlim(3 * 11, 7 * 11)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('cells')
    ax.set_ylim(0, len(data))
    ax.set_yticks(np.arange(0, len(data), 100))
    ax.set_yticklabels(np.arange(0, len(data), 100))

    return ax


def plot_response_heatmaps_for_conditions(multi_session_df, timestamps, data_type, event_type,
                                          row_condition, col_condition, cols_to_sort_by=None, suptitle=None,
                                          microscope=None, vmax=0.05, xlim_seconds=None, match_cells=False, cbar=True,
                                          save_dir=None, folder=None, suffix='', ax=None):
    sdf = multi_session_df.copy()

    if 'omission' in event_type:
        xlabel = 'time after omission (s)'
        trace_type = 'omitted'
    elif 'change' in event_type:
        xlabel = 'time after change (s)'
        trace_type = 'change'
    else:
        xlabel = 'time (s)'
        trace_type = 'unknown'

    if xlim_seconds is None:
        xlim_seconds = (timestamps[0], timestamps[-1])

    row_conditions = np.sort(sdf[row_condition].unique())
    col_conditions = np.sort(sdf[col_condition].unique())

    if ax is None:
        figsize = (3 * len(col_conditions), 3 * len(row_conditions))
        fig, ax = plt.subplots(len(row_conditions), len(col_conditions), figsize=figsize, sharex=True)
        ax = ax.ravel()

    i = 0
    for r, row in enumerate(row_conditions):
        row_sdf = sdf[(sdf[row_condition] == row)]
        for c, col in enumerate(col_conditions):

            if row == 'Excitatory':
                vmax = 0.01
            elif row == 'Vip Inhibitory':
                vmax = 0.02
            elif row == 'Sst Inhibitory':
                vmax = 0.03
            else:
                vmax = 0.02

            tmp = row_sdf[(row_sdf[col_condition] == col)]
            tmp = tmp.reset_index()
            if cols_to_sort_by:
                tmp = tmp.sort_values(by=cols_to_sort_by, ascending=True)
            else:
                if match_cells:
                    if c == 0:
                        tmp = tmp.sort_values(by='mean_response', ascending=True)
                        order = tmp.index.values
                    else:
                        tmp = tmp.loc[order]
                else:
                    tmp = tmp.sort_values(by='mean_response', ascending=True)
            data = pd.DataFrame(np.vstack(tmp.mean_trace.values), columns=timestamps)
            n_cells = len(data)

            ax[i] = plot_cell_response_heatmap(data, timestamps, vmax=vmax, xlabel=xlabel, cbar=cbar,
                                               microscope=microscope, ax=ax[i])
            ax[i].set_title(row + '\n' + col)
            # label y with total number of cells
            ax[i].set_yticks([0, n_cells])
            ax[i].set_yticklabels([0, n_cells], fontsize=12)
            # set xticks to every 1 second, assuming 30Hz traces
            ax[i].set_xticks(np.arange(0, len(timestamps), 30)) # assuming 30Hz traces
            ax[i].set_xticklabels([int(t) for t in timestamps[::30]])
            # set xlims according to input
            start_index = np.where(timestamps == xlim_seconds[0])[0][0]
            end_index = np.where(timestamps == xlim_seconds[1])[0][0]
            xlims = [start_index, end_index]
            ax[i].set_xlim(xlims)
            ax[i].set_ylabel('')

            if r == len(row_conditions)-1:
                ax[i].set_xlabel(xlabel)
            else:
                ax[i].set_xlabel('')
            i += 1

    for i in np.arange(0, (len(col_conditions)*len(row_conditions)), len(col_conditions)):
        ax[i].set_ylabel('cells')

    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=1.04, fontsize=18)
    fig.tight_layout()

    if save_dir:
        fig_title = event_type + '_response_heatmap_' + data_type + '_' + col_condition + '_' + row_condition + '_' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)

    return ax


def addSpan(ax, amin, amax, color='k', alpha=0.3, axtype='x', zorder=1):
    """
    adds a vertical span to an axis
    """
    if axtype == 'x':
        ax.axvspan(amin, amax, facecolor=color, edgecolor='none', alpha=alpha, linewidth=0, zorder=zorder)
    if axtype == 'y':
        ax.axhspan(amin, amax, facecolor=color, edgecolor='none', alpha=alpha, linewidth=0, zorder=zorder)


def add_stim_color_span(dataset, ax, xlim=None, color=None, label_changes=False, label_omissions=False):
    """
    adds a vertical span for all stimulus presentations contained within xlim
    xlim is a time in seconds during a behavior session
    if label_changes is True, changes will be blue and all other flashes will be gray
    if label_changes is False, each flash will be colored according to image identity
    if label_omissions is True, a dotted line will be shown at the time of omission
    if a color is provided, all stimulus presentations will be that color
    """
    # set default alpha. If label_changes=True, alphas will be reset below.
    alpha = 0.3
    # get stim table
    stim_table = dataset.stimulus_presentations.copy()
    # remove omissions because they dont get labeled
    #     stim_table = stim_table[stim_table.omitted == False].copy()
    # get all images & assign colors (image colors wont be used if a color is provided or if label_changes is True)
    images = np.sort(stim_table[stim_table.omitted == False].image_name.unique())
    image_colors = sns.color_palette("hls", len(images))
    # limit to time window if provided
    if xlim != None:
        stim_table = stim_table[(stim_table.start_time >= xlim[0]) & (stim_table.stop_time <= xlim[1])]
    # loop through stimulus presentations and add a span with appropriate color
    for idx in stim_table.index:
        start_time = stim_table.loc[idx]['start_time']
        stop_time = stim_table.loc[idx]['stop_time']
        image_name = stim_table.loc[idx]['image_name']
        image_index = stim_table.loc[idx]['image_index']
        if image_name == 'omitted':
            if label_omissions:
                ax.axvline(x=start_time, ymin=0, ymax=1, linestyle='--', color=sns.color_palette()[9])
        else:
            if label_changes:
                if stim_table.loc[idx]['is_change']:  # if its a change, make it blue with higher alpha
                    image_color = sns.color_palette()[0]
                    alpha = 0.5
                else:  # if its a non-change make it gray with low alpha
                    image_color = 'gray'
                    alpha = 0.25
            else:
                if color == None:
                    image_color = image_colors[image_index]
                else:
                    image_color = color
            addSpan(ax, start_time, stop_time, color=image_color, alpha=alpha)
    return ax



def plot_behavior_timeseries(dataset, start_time, duration_seconds=20, xlim_seconds=None, save_dir=None, ax=None):
    """
    Plots licking behavior, rewards, running speed, and pupil area for a defined window of time
    """
    if xlim_seconds is None:
        xlim_seconds = [start_time - (duration_seconds / 4.), start_time + duration_seconds * 2]
    else:
        if start_time != xlim_seconds[0]:
            start_time = xlim_seconds[0]

    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))
    licks[:] = -2

    reward_timestamps = dataset.rewards.timestamps.values
    rewards = np.zeros(len(reward_timestamps))
    rewards[:] = -4

    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values

    eye_tracking = dataset.eye_tracking.copy()
    pupil_diameter = eye_tracking.pupil_width.values
    pupil_diameter[eye_tracking.likely_blink == True] = np.nan
    pupil_timestamps = eye_tracking.timestamps.values

    if ax is None:
        figsize = (15, 2.5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = sns.color_palette()

    ln0 = ax.plot(lick_timestamps, licks, '|', label='licks', color=colors[3], markersize=10)
    ln1 = ax.plot(reward_timestamps, rewards, 'o', label='rewards', color=colors[9], markersize=10)

    ln2 = ax.plot(running_timestamps, running_speed, label='running_speed', color=colors[2], zorder=100)
    ax.set_ylabel('running speed\n(cm/s)')
    ax.set_ylim(ymin=-8)

    ax2 = ax.twinx()
    ln3 = ax2.plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color=colors[4], zorder=0)

    ax2.set_ylabel('pupil diameter \n(pixels)')
    #     ax2.set_ylim(0, 200)

    axes_to_label = ln0 + ln1 + ln2 + ln3  # +ln4
    labels = [label.get_label() for label in axes_to_label]
    ax.legend(axes_to_label, labels, bbox_to_anchor=(1, 1), fontsize='small')

    ax = add_stim_color_span(dataset, ax, xlim=xlim_seconds)

    ax.set_xlim(xlim_seconds)
    ax.set_xlabel('time in session (seconds)')
    metadata_string = utils.get_metadata_string(dataset.metadata)
    ax.set_title(metadata_string)

    # ax.tick_params(which='both', bottom=True, top=False, right=False, left=True,
    #                 labelbottom=True, labeltop=False, labelright=True, labelleft=True)
    # ax2.tick_params(which='both', bottom=True, top=False, right=True, left=False,
    #                 labelbottom=True, labeltop=False, labelright=True, labelleft=True)
    if save_dir:
        folder = 'behavior_timeseries'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)),
                          formats=['.png'])
    return ax


def plot_behavior_timeseries_stacked(dataset, start_time, duration_seconds=20,
                                     label_changes=True, label_omissions=True,
                                     save_dir=None, ax=None):
    """
    Plots licking behavior, rewards, running speed, and pupil area for a defined window of time.
    Each timeseries gets its own row. If label_changes=True, all flashes are gray, changes are blue.
    If label_changes=False, unique colors are given to each image.
    If label_omissions=True, a dotted line will be plotted at the time of omissions.
    """

    if label_changes:
        suffix = '_changes'
    else:
        suffix = '_colors'

    xlim_seconds = [start_time-(duration_seconds/4.), start_time+duration_seconds*2]

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
    start_ind = np.where(running_timestamps<xlim_seconds[0])[0][-1]
    stop_ind = np.where(running_timestamps>xlim_seconds[1])[0][0]
    running_speed = running_speed[start_ind:stop_ind]
    running_timestamps = running_timestamps[start_ind:stop_ind]

    # get pupil width trace and timestamps
    eye_tracking = dataset.eye_tracking.copy()
    pupil_diameter = eye_tracking.pupil_width.values
    pupil_diameter[eye_tracking.likely_blink==True] = np.nan
    pupil_timestamps = eye_tracking.timestamps.values
    # smooth pupil diameter
    from scipy.signal import medfilt
    pupil_diameter = medfilt(pupil_diameter, kernel_size=5)
    # limit pupil trace to window so yaxes scale properly
    start_ind = np.where(pupil_timestamps<xlim_seconds[0])[0][-1]
    stop_ind = np.where(pupil_timestamps>xlim_seconds[1])[0][0]
    pupil_diameter = pupil_diameter[start_ind:stop_ind]
    pupil_timestamps = pupil_timestamps[start_ind:stop_ind]

    if ax is None:
        figsize = (15, 5)
        fig, ax = plt.subplots(4, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [1, 1, 3, 3]})
        ax = ax.ravel()

    colors = sns.color_palette()

    ax[0].plot(lick_timestamps, licks, '|', label='licks', color=colors[3], markersize=10)
    ax[0].set_yticklabels([])
    ax[0].set_ylabel('licks', rotation=0, horizontalalignment='right', verticalalignment='center')
    ax[0].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                      labelbottom=False, labeltop=False, labelright=False, labelleft=False)

    ax[1].plot(reward_timestamps, rewards, 'o', label='rewards', color=colors[8], markersize=10)
    ax[1].set_yticklabels([])
    ax[1].set_ylabel('rewards', rotation=0, horizontalalignment='right', verticalalignment='center')
    ax[1].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                      labelbottom=False, labeltop=False, labelright=False, labelleft=False)

    ax[2].plot(running_timestamps, running_speed, label='running_speed', color=colors[2], zorder=100)
    ax[2].set_ylabel('running\nspeed\n(cm/s)', rotation=0, horizontalalignment='right', verticalalignment='center')
    ax[2].set_ylim(ymin=-8)

    ax[3].plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color=colors[4], zorder=0)
    ax[3].set_ylabel('pupil\ndiameter\n(pixels)', rotation=0, horizontalalignment='right', verticalalignment='center')

    for i in range(4):
        ax[i] = add_stim_color_span(dataset, ax[i], xlim=xlim_seconds, label_changes=label_changes, label_omissions=label_omissions)
        ax[i].set_xlim(xlim_seconds)
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=True,
                    labelbottom=False, labeltop=False, labelright=False, labelleft=True)
        sns.despine(ax=ax[i], bottom=True)
    sns.despine(ax=ax[i], bottom=False)

    # label bottom row of plot
    ax[i].set_xlabel('time in session (seconds)')
    ax[i].tick_params(which='both', bottom=True, top=False, right=False, left=True,
                    labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    # add title to top row
    metadata_string = utils.get_metadata_string(dataset.metadata)
    ax[0].set_title(metadata_string)

    plt.subplots_adjust(hspace=0)
    if save_dir:
        folder = 'behavior_timeseries_stacked'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time))+'_'+suffix,
                          formats=['.png', '.pdf'])
    return ax


def plot_matched_roi_and_trace(ophys_container_id, cell_specimen_id, limit_to_last_familiar_second_novel=True,
                               use_events=False, filter_events=False, save_figure=True):
    """
    Generates plots characterizing single cell activity in response to stimulus, omissions, and changes.
    Compares across all sessions in a container for each cell, including the ROI mask across days.
    Useful to validate cell matching as well as examine changes in activity profiles over days.
    """
    experiments_table = loading.get_platform_paper_experiment_table()
    if limit_to_last_familiar_second_novel: # this ensures only one session per experience level
        experiments_table = utilities.limit_to_last_familiar_second_novel_active(experiments_table)
        experiments_table = utilities.limit_to_containers_with_all_experience_levels(experiments_table)

    container_expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id]
    container_expts = container_expts.sort_values(by=['experience_level'])
    expts = np.sort(container_expts.index.values)

    if use_events:
        if filter_events:
            suffix = 'filtered_events'
        else:
            suffix = 'events'
        ylabel = 'response'
    else:
        suffix = 'dff'
        ylabel = 'dF/F'

    n = len(expts)
    if limit_to_last_familiar_second_novel:
        figsize = (9, 6)
        folder = 'matched_cells_exp_levels'
    else:
        figsize = (20, 6)
        folder = 'matched_cells_all_sessions'
    fig, ax = plt.subplots(2, n, figsize=figsize, sharey='row')
    ax = ax.ravel()
    print('ophys_container_id:', ophys_container_id)
    for i, ophys_experiment_id in enumerate(expts):
        print('ophys_experiment_id:', ophys_experiment_id)
        try:
            dataset = loading.get_ophys_dataset(ophys_experiment_id, get_extended_stimulus_presentations=False)
            if cell_specimen_id in dataset.dff_traces.index:

                ct = dataset.cell_specimen_table.copy()
                cell_roi_id = ct.loc[cell_specimen_id].cell_roi_id
                roi_masks = dataset.roi_masks.copy()  # save this to get approx ROI position if subsequent session is missing the ROI (fails if the first session is the one missing the ROI)
                ax[i] = sf.plot_cell_zoom(dataset.roi_masks, dataset.max_projection, cell_roi_id,
                                          spacex=50, spacey=50, show_mask=True, ax=ax[i])
                ax[i].set_title(container_expts.loc[ophys_experiment_id].experience_level)

                analysis = ResponseAnalysis(dataset, use_events=use_events, filter_events=filter_events,
                                            use_extended_stimulus_presentations=False)
                sdf = analysis.get_response_df(df_name='stimulus_response_df')
                cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.is_change == True)]

                window = rp.get_default_stimulus_response_params()["window_around_timepoint_seconds"]
                ax[i + n] = utils.plot_mean_trace(cell_data.trace.values, cell_data.trace_timestamps.values[0],
                                                  ylabel=ylabel, legend_label=None, color='gray', interval_sec=0.5,
                                                  xlim_seconds=window, plot_sem=True, ax=ax[i + n])

                ax[i + n] = utils.plot_flashes_on_trace(ax[i + n], cell_data.trace_timestamps.values[0], change=True, omitted=False,
                                                        alpha=0.15, facecolor='gray')
                ax[i + n].set_title('')
                if i != 0:
                    ax[i + n].set_ylabel('')
            else:
                # plot the max projection image with the xy location of the previous ROI
                # this will fail if the familiar session is the one without the cell matched
                ax[i] = sf.plot_cell_zoom(roi_masks, dataset.max_projection, cell_roi_id,
                                          spacex=50, spacey=50, show_mask=False, ax=ax[i])
                ax[i].set_title(container_expts.loc[ophys_experiment_id].experience_level)

            metadata_string = utils.get_metadata_string(dataset.metadata)

            fig.tight_layout()
            fig.suptitle(str(cell_specimen_id) + '_' + metadata_string, x=0.53, y=1.02,
                         horizontalalignment='center', fontsize=16)
        except Exception as e:
            print('problem for cell_specimen_id:', cell_specimen_id, ', ophys_experiment_id:', ophys_experiment_id)
            print(e)
    if save_figure:
        save_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/cell_matching'
        utils.save_figure(fig, figsize, save_dir, folder, str(cell_specimen_id) + '_' + metadata_string + '_' + suffix)
        plt.close()



def plot_matched_roi_and_traces_example(cell_metadata, include_omissions=True,
                                        use_events=False, filter_events=False, save_dir=None, folder=None):
    """
    Plots the ROI masks and cell traces for a cell matched across sessions
    Cell_metadata is a subset of the ophys_cells_table limited to the cell_specimen_id of interest
    Masks and traces will be plotted for all ophys_experiment_ids in the cell_metadata table
    To limit to a single session of each type, set last_familiar_second_novel to True
    ROI mask for each ophys_experiment_id in cell_metadata is plotted on its own axis
    Average cell traces across all experiments are plotted on a single axis with each trace colored by its experience_level
    if include_omissions is True, there will be one axis for the change response and one axis for the omission response across sessions
    if include_omissions is False, only change responses will be plotted
    Only plots data for ophys_experiment_ids where the cell_specimen_id is present, does not plot max projections without an ROI mask for expts in a container where the cell was not detected
    To generate plots showing max projections from experiments in a container where a cell was not detected, use plot_matched_roi_and_trace
    """

    if len(cell_metadata.cell_specimen_id.unique()) > 1:
        print('There is more than one cell_specimen_id in the provided cell_metadata table')
        print('Please limit input to a single cell_specimen_id')

    # get relevant info for this cell
    cell_metadata = cell_metadata.sort_values(by='experience_level')
    cell_specimen_id = cell_metadata.cell_specimen_id.unique()[0]
    ophys_container_id = cell_metadata.ophys_container_id.unique()[0]
    ophys_experiment_ids = cell_metadata.ophys_experiment_id.unique()
    n_expts = len(ophys_experiment_ids)

    # set up labels for different trace types
    if use_events:
        if filter_events:
            suffix = 'filtered_events'
        else:
            suffix = 'events'
        ylabel = 'response'
    else:
        suffix = 'dff'
        ylabel = 'dF/F'

    # number of columns is one for each experiments ROI mask, plus additional columns for stimulus and omission traces
    if include_omissions:
        n_cols = n_expts + 2
    else:
        n_cols = n_expts + 1

    experience_levels = ['Familiar', 'Novel 1', 'Novel >1']
    colors = utils.get_experience_level_colors()

    figsize = (3 * n_cols, 3)
    fig, ax = plt.subplots(1, n_cols, figsize=figsize)

    print('cell_specimen_id:', cell_specimen_id)
    print('ophys_container_id:', ophys_container_id)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        print('ophys_experiment_id:', ophys_experiment_id)
        experience_level = \
        cell_metadata[cell_metadata.ophys_experiment_id == ophys_experiment_id].experience_level.values[0]
        ind = experience_levels.index(experience_level)
        color = colors[ind]
        try:
            dataset = loading.get_ophys_dataset(ophys_experiment_id, get_extended_stimulus_presentations=False)
            if cell_specimen_id in dataset.dff_traces.index:

                ct = dataset.cell_specimen_table.copy()
                cell_roi_id = ct.loc[cell_specimen_id].cell_roi_id
                roi_masks = dataset.roi_masks.copy()  # save this to get approx ROI position if subsequent session is missing the ROI (fails if the first session is the one missing the ROI)
                ax[i] = sf.plot_cell_zoom(dataset.roi_masks, dataset.max_projection, cell_roi_id,
                                          spacex=50, spacey=50, show_mask=True, ax=ax[i])
                ax[i].set_title(experience_level)

                # get change responses and plot on second to last axis
                window = [-1, 1.5]  # window around event
                sdf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True,
                                                       output_sampling_rate=30,
                                                       data_type='events', event_type='changes',
                                                       load_from_file=True)
                cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.is_change == True)]

                ax[n_expts] = utils.plot_mean_trace(cell_data.trace.values, cell_data.trace_timestamps.values[0],
                                                    ylabel=ylabel, legend_label=None, color=color, interval_sec=1,
                                                    xlim_seconds=window, plot_sem=True, ax=ax[n_expts])
                ax[n_expts] = utils.plot_flashes_on_trace(ax[n_expts], cell_data.trace_timestamps.values[0],
                                                          change=True, omitted=False)
                ax[n_expts].set_title('changes')

                # get omission responses and plot on last axis
                if include_omissions:
                    sdf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True,
                                                           output_sampling_rate=30,
                                                           data_type='events', event_type='omissions',
                                                           load_from_file=True)
                    cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.omitted == True)]

                    ax[n_expts + 1] = utils.plot_mean_trace(cell_data.trace.values,
                                                            cell_data.trace_timestamps.values[0],
                                                            ylabel=ylabel, legend_label=None, color=color,
                                                            interval_sec=1,
                                                            xlim_seconds=window, plot_sem=True, ax=ax[n_expts + 1])
                    ax[n_expts + 1] = utils.plot_flashes_on_trace(ax[n_expts + 1],
                                                                  cell_data.trace_timestamps.values[0],
                                                                  change=False, omitted=True)
                    ax[n_expts + 1].set_title('omissions')

            metadata_string = utils.get_metadata_string(dataset.metadata)

            fig.tight_layout()
            fig.suptitle(str(cell_specimen_id) + '_' + metadata_string, x=0.53, y=1.02,
                         horizontalalignment='center', fontsize=16)
        except Exception as e:
            print('problem for cell_specimen_id:', cell_specimen_id, ', ophys_experiment_id:', ophys_experiment_id)
            print(e)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder,
                          str(cell_specimen_id) + '_' + metadata_string + '_' + suffix)
        plt.close()


# examples
if __name__ == '__main__':

    import visual_behavior.data_access.loading as loading
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

    # load cache
    cache_dir = loading.get_platform_analysis_cache_dir()
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir)
    experiments_table = loading.get_platform_paper_experiment_table()

    # load multi_session_df
    df_name = 'omission_response_df'
    conditions = ['cell_specimen_id']
    use_events = True
    filter_events = True

    multi_session_df = loading.get_multi_session_df(cache_dir, df_name, conditions, experiments_table,
                                                    use_events=use_events, filter_events=filter_events)

    # limit to platform paper dataset
    multi_session_df = multi_session_df[multi_session_df.ophys_experiment_id.isin(experiments_table.index.values)]
    # merge with metadata
    multi_session_df = multi_session_df.merge(experiments_table, on='ophys_experiment_id')

    # set project code & df_name to plot
    project_code = 'VisualBehaviorMultiscope'
    df_name = 'omission_response_df'

    # get timestamps for population average
    experiment_id = experiments_table[experiments_table.project_code == project_code].index.values[9]
    timestamps = get_timestamps_for_response_df_type(cache, experiment_id, df_name)

    # plot population average for experience_level
    axes_column = 'cell_type'
    hue_column = 'experience_level'
    palette = utils.get_experience_level_colors()
    xlim_seconds = [-1.8, 2.25]

    df = multi_session_df[multi_session_df.project_code == project_code]
    plot_population_averages_for_conditions(df, df_name, timestamps,
                                            axes_column, hue_column, palette,
                                            use_events=True, filter_events=True, xlim_seconds=xlim_seconds,
                                            horizontal=True, save_dir=None, folder=None)
