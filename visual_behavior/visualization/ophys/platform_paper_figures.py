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
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.ophys.response_analysis.response_processing as rp
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.top': False, 'axes.spines.right': False})  # ticks or white
sns.set_palette('deep')

plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

# basic characterization #########################


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
    cell_types = np.sort(df.cell_type.unique())

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
        data = fraction_responsive[fraction_responsive.cell_type == cell_type]
        for ophys_container_id in data.ophys_container_id.unique():
            ax[i] = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x='experience_level', y='total_cells',
                                  color='gray', join=True, markers='.', scale=0.25, errwidth=0.25, ax=ax[i])
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
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

# population averages across session & within epochs #####################


def plot_population_averages_for_conditions(multi_session_df, data_type, event_type, axes_column, hue_column,
                                            project_code=None, timestamps=None, palette=None, title=None, suptitle=None,
                                            horizontal=True, xlim_seconds=None, interval_sec=1,
                                            save_dir=None, folder=None, suffix='', ax=None):
    if palette is None:
        palette = utils.get_experience_level_colors()

    sdf = multi_session_df.copy()
    if 'trace_timestamps' in sdf.keys():
        timestamps = sdf.trace_timestamps.values[0]
    elif timestamps is not None:
        timestamps = timestamps
    else:
        print('provide timestamps or provide a multi_session_df with a trace_timestamps column')

    if xlim_seconds is None:
        xlim_seconds = [timestamps[0], timestamps[-1]]
    if 'dff' in data_type:
        ylabel = 'dF/F'
    elif 'events' in data_type:
        ylabel = 'population response'
    elif 'pupil' in data_type:
        ylabel = 'pupil width\n(normalized)'
    elif 'running' in data_type:
        ylabel = 'running speed (cm/s)'
    elif 'lick' in data_type:
        ylabel = 'lick rate (licks/s)'
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

    if hue_column == 'experience_level':
        # hue_conditions = ['Familiar', 'Novel 1', 'Novel >1']
        hue_conditions = np.sort(sdf[hue_column].unique())
    else:
        hue_conditions = np.sort(sdf[hue_column].unique())
    if axes_column == 'experience_level':
        # axes_conditions = ['Familiar', 'Novel 1', 'Novel >1']
        axes_conditions = np.sort(sdf[axes_column].unique())
    else:
        axes_conditions = np.sort(sdf[axes_column].unique())
    # if there is only one axis condition, set n conditions for plotting to 2 so it can still iterate
    if len(axes_conditions) == 1:
        n_axes_conditions = 2
    else:
        n_axes_conditions = len(axes_conditions)
    if ax is None:
        format_fig = True
        if horizontal:
            figsize = (4.5 * n_axes_conditions, 4)
            fig, ax = plt.subplots(1, n_axes_conditions, figsize=figsize, sharey=False)
        else:
            figsize = (5, 3.5 * n_axes_conditions)
            fig, ax = plt.subplots(n_axes_conditions, 1, figsize=figsize, sharex=True)
    else:
        format_fig = False
    for i, axis in enumerate(axes_conditions):
        for c, hue in enumerate(hue_conditions):
            # try:
            cdf = sdf[(sdf[axes_column] == axis) & (sdf[hue_column] == hue)]
            traces = cdf.mean_trace.values

            ax[i] = utils.plot_mean_trace(np.asarray(traces), timestamps, ylabel=ylabel,
                                          legend_label=hue, color=palette[c], interval_sec=interval_sec,
                                          xlim_seconds=xlim_seconds, ax=ax[i])
            ax[i] = utils.plot_flashes_on_trace(ax[i], timestamps, change=change, omitted=omitted)
            if omitted:
                omission_color = sns.color_palette()[9]
                ax[i].axvline(x=0, ymin=0, ymax=1, linestyle='--', color=omission_color)
            if title == 'metadata':
                metadata_string = utils.get_container_metadata_string(utils.get_metadata_for_row_of_multi_session_df(cdf))
                ax[i].set_title(metadata_string)
            else:
                if axes_column == 'experience_level':
                    title_colors = utilities.get_experience_level_colors()
                    ax[i].set_title(axis, color=title_colors[i], fontsize=20)
                else:
                    ax[i].set_title(axis)
            if title:  # overwrite title if one is provided
                ax[i].set_title(title)
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
    # ax[0].legend(loc='upper right', fontsize='xx-small') # title='passive', title_fontsize='xx-small')

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


def plot_population_averages_for_cell_types_across_experience(multi_session_df, xlim_seconds=[-1.25, 1.5], xlabel='time (s)',
                                                              ylabel='population average',
                                                              data_type='events', event_type='changes', interval_sec=1,
                                                              save_dir=None, folder=None, suffix=None, ax=None):
    # get important information
    experiments_table = loading.get_platform_paper_experiment_table()
    cell_types = np.sort(experiments_table.cell_type.unique())
    # palette = utilities.get_experience_level_colors()
    palette = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    # palette = [(.2, .2, .2), (.2, .2, .2), (.2, .2, .2)]

    # define plot axes
    axes_column = 'experience_level'
    hue_column = 'experience_level'

    if ax is None:
        format_fig = True
        figsize = (10, 9)
        fig, ax = plt.subplots(3, 3, figsize=figsize, sharey='row', sharex='col')
        ax = ax.ravel()
    else:
        format_fig = False

    for i, cell_type in enumerate(cell_types):
        df = multi_session_df[(multi_session_df.cell_type == cell_type)]
        if format_fig:
            ax[i * 3:(i * 3 + 3)] = plot_population_averages_for_conditions(df, data_type, event_type,
                                                                            axes_column, hue_column,
                                                                            horizontal=True,
                                                                            xlim_seconds=xlim_seconds,
                                                                            interval_sec=interval_sec,
                                                                            palette=palette,
                                                                            ax=ax[i * 3:(i * 3 + 3)])
        else:
            ax[i] = plot_population_averages_for_conditions(df, data_type, event_type,
                                                            axes_column, hue_column, horizontal=True,
                                                            xlim_seconds=xlim_seconds, interval_sec=interval_sec,
                                                            palette=palette, ax=ax[i])

    if format_fig:
        for i in [0, 3, 6]:
            ax[i].set_ylabel(ylabel)
        for i in np.arange(3, 9):
            ax[i].set_title('')
        for i in np.arange(0, 6):
            ax[i].set_xlabel('')
        for i in np.arange(6, 9):
            ax[i].set_xlabel(xlabel)
        fig.tight_layout()
    else:
        for i in range(len(cell_types)):
            ax[i][0].set_ylabel(ylabel)
        for i in np.arange(1, 3):
            for x in range(3):
                ax[i][x].set_title('')
        for i in np.arange(0, 2):
            for x in range(3):
                ax[i][x].set_xlabel('')

    if save_dir:
        fig_title = 'population_average_cell_types_exp_levels' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png', '.pdf'])

    return ax


def plot_population_averages_across_experience(multi_session_df, xlim_seconds=[-1.25, 1.5], xlabel='time (s)', ylabel='population\nresponse',
                                               data_type='events', event_type='changes', interval_sec=1,
                                               save_dir=None, folder=None, suffix=None, ax=None):
    # get important information
    palette = utilities.get_experience_level_colors()

    # define plot axes
    axes_column = 'experience_level'
    hue_column = 'experience_level'

    if ax is None:
        figsize = (12, 3)
        fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True, sharex=True)
        ax = ax.ravel()

    df = multi_session_df.copy()
    ax = plot_population_averages_for_conditions(df, data_type, event_type,
                                                 axes_column, hue_column, horizontal=True,
                                                 xlim_seconds=xlim_seconds, interval_sec=interval_sec,
                                                 palette=palette, ax=ax)
    ax[0].set_ylabel(ylabel)

    if save_dir:
        fig_title = 'population_average_exp_levels' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png', '.pdf'])

    return ax


def plot_mean_response_by_epoch(df, metric='mean_response', horizontal=True, ymin=0, ylabel='mean response', estimator=np.mean,
                                save_dir=None, folder='epochs', max_epoch=6, suptitle=None, suffix='', ax=None):
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

    # add experience epoch column if it doesnt already exist
    if 'experience_epoch' not in df.keys():
        df = utilities.annotate_epoch_df(df)

    cell_types = np.sort(df.cell_type.unique())[::-1]
    experience_levels = np.sort(df.experience_level.unique())

    df = df[df.epoch <= max_epoch]
    max_n_sessions = np.max(df.epoch.unique())

    experience_epoch = np.sort(df.experience_epoch.unique())
    xticks = np.arange(0, len(experience_epoch), 1)

    palette = utils.get_experience_level_colors()
    if ax is None:
        format_fig = True
        if horizontal:
            figsize = (13, 3.5)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=False)
        else:
            figsize = (15, 10)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)
    else:
        format_fig = False

    for i, cell_type in enumerate(cell_types):
        try:
            print(cell_type)
            data = df[df.cell_type == cell_type]
            ax[i] = sns.pointplot(data=data, x='experience_epoch', y=metric, hue='experience_level', hue_order=experience_levels,
                                  order=experience_epoch, palette=palette, ax=ax[i], estimator=estimator)

            if ymin is not None:
                ax[i].set_ylim(ymin=ymin)
            ax[i].set_title(cell_type)
            ax[i].set_ylabel(ylabel)
            ax[i].get_legend().remove()
            ax[i].set_xlim((xticks[0] - 1, xticks[-1] + 1))
            ax[i].set_xticks(xticks[::2])
            xticklabels = [experience_epoch.split(' ')[-1] for experience_epoch in experience_epoch]
            ax[i].set_xticklabels(xticklabels[::2], fontsize=10)
            ax[i].vlines(x=max_n_sessions + 0.5, ymin=0, ymax=1, color='gray', linestyle='--')
            ax[i].vlines(x=max_n_sessions + max_n_sessions + 1.5, ymin=0, ymax=1, color='gray', linestyle='--')
            if horizontal:
                ax[i].set_xlabel('epoch within session', fontsize=14)
            else:
                ax[i].set_xlabel('')
        except Exception as e:
            print(e)
    ax[i].set_xlabel('epoch within session', fontsize=14)
    if format_fig:
        if suptitle is not None:
            plt.suptitle(suptitle, x=0.52, y=1.01, fontsize=18)
        fig.tight_layout()
    if save_dir:
        fig_title = metric + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)
    return ax


def plot_mean_response_by_epoch_for_multiple_conditions(response_df_dict, metric='mean_response', horizontal=True,
                                                        ymin=0, suptitle=None, axes_condition='cell_type',
                                                        save_dir=None, folder='epochs', suffix=''):
    """
    Plots the mean metric value across 10 minute epochs, for two different conditions (multi_session_dfs)
    The two conditions are defined by the multi_session_dfs passed in the response_df_dict,
    which should have two keys, one for the name of each multi_session_df, and the values are the multi_session_dfs
    :param response_df_dict: dictionary containing two dataframes of cell activity with one row per cell_specimen_id / ophys_experiment_id
                            must include columns 'cell_type', 'experience_level', 'epoch', and a column for the metric provided (ex: 'mean_response')
    :param metric: metric value to average over epochs, must be a column of df
    :param save_dir: top level directory to save figure to
    :param folder: folder within save_dir to save figure to; will create folder if it doesnt exist
    :param suffix: string to append at end of saved filename
    :param use_alpha: If True, will plot the two conditions as light and dark versions of experience level colors
                      If False, will plot the two conditions as black and gray lines
    :return:
    """
    import matplotlib.lines as mlines

    df_names = list(response_df_dict.keys())

    colors = sns.color_palette('Paired', len(df_names))

    # get xtick values from one of the dfs
    df = response_df_dict[df_names[0]]
    xticks = [experience_epoch.split(' ')[-1] for experience_epoch in np.sort(df.experience_epoch.unique())]
    n_epochs = np.amax(df.epoch.unique())
    # get info to plot
    axes_conditions = np.sort(df[axes_condition].unique())[::-1]
    experience_epoch = np.sort(df.experience_epoch.unique())

    if horizontal:
        figsize = (4 * len(axes_conditions), 3.5)
        fig, ax = plt.subplots(1, len(axes_conditions), figsize=figsize, sharex=False)
    else:
        figsize = (18, 4 * len(axes_conditions))
        fig, ax = plt.subplots(len(axes_conditions), 1, figsize=figsize, sharex=True)
    for i, axis_value in enumerate(axes_conditions):

        for c, df_name in enumerate(df_names):
            df = response_df_dict[df_name]
            data = df[df[axes_condition] == axis_value]
            ax[i] = sns.pointplot(data=data, x='experience_epoch', y=metric, label=df_name,
                                  order=experience_epoch, color=colors[c], ax=ax[i])
        ax[i].set_ylim(ymin=ymin)
        ax[i].set_title(axis_value)
        ax[i].vlines(x=n_epochs - 0.5, ymin=0, ymax=1, color='gray', linestyle='--')
        ax[i].vlines(x=(n_epochs * 2) + 1.5, ymin=0, ymax=1, color='gray', linestyle='--')

    image = mlines.Line2D([], [], color=colors[0], label='non-change')
    change = mlines.Line2D([], [], color=colors[1], label='change')
    omission = mlines.Line2D([], [], color=colors[2], label='omission')
    ax[i].legend(handles=[image, change, omission], fontsize='x-small')

    xlabel = str(int(60 / n_epochs)) + ' min epoch in session'
    ax[i].set_xlabel(xlabel)
    ax[i].set_xticklabels(xticks, fontsize=9)

    if suptitle is None:
        plt.suptitle(metric + ' over time - ' + df_names[0] + ', ' + df_names[1] + ' - ' + suffix, x=0.52, y=1.02,
                     fontsize=16)
    else:
        plt.suptitle(suffix, x=0.52, y=1.02, fontsize=16)
    fig.tight_layout()
    if save_dir:
        fig_title = metric + '_epochs_' + df_names[0] + '_' + df_names[1] + '_' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)


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

# response metrics ########################


def get_fraction_responsive_cells(multi_session_df, conditions=['cell_type', 'experience_level'], responsiveness_threshold=0.1):
    """
    Computes the fraction of cells for each condition with fraction_significant_p_value_gray_screen > responsiveness_threshold
    :param multi_session_df: dataframe of trial averaged responses for each cell for some set of conditions
    :param conditions: conditions defined by columns in df over which to group to quantify fraction responsive cells
    :param responsiveness_threshold: threshold on fraction_significant_p_value_gray_screen to determine whether a cell is responsive or not
    :return:
    """
    df = multi_session_df.copy()
    total_cells = df.groupby(conditions).count()[['cell_specimen_id']].rename(columns={'cell_specimen_id': 'total_cells'})
    responsive = df[df.fraction_significant_p_value_gray_screen > responsiveness_threshold].copy()
    responsive_cells = responsive.groupby(conditions).count()[['cell_specimen_id']].rename(columns={'cell_specimen_id': 'responsive_cells'})
    fraction = total_cells.merge(responsive_cells, on=conditions, how='left')  # need to use 'left' to prevent dropping of NaN values
    # set sessions with no responsive cells (NaN) to zero
    fraction.loc[fraction[fraction.responsive_cells.isnull()].index.values, 'responsive_cells'] = 0
    fraction['fraction_responsive'] = fraction.responsive_cells / fraction.total_cells
    return fraction


def plot_fraction_responsive_cells(multi_session_df, responsiveness_threshold=0.1, horizontal=True, ylim=(0, 1),
                                   save_dir=None, folder=None, suffix='', format_fig=False, ax=None):
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
    cell_types = np.sort(df.cell_type.unique())

    fraction_responsive = get_fraction_responsive_cells(df, conditions=['cell_type', 'experience_level', 'ophys_container_id', 'ophys_experiment_id'],
                                                        responsiveness_threshold=responsiveness_threshold)
    fraction_responsive = fraction_responsive.reset_index()

    palette = utils.get_experience_level_colors()
    if ax is None:
        if horizontal:
            figsize = (9, 3.5)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False)
        else:
            if format_fig:
                figsize = (3, 8)
                fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
            else:
                figsize = (2, 10)
                fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for i, cell_type in enumerate(cell_types):
        data = fraction_responsive[fraction_responsive.cell_type == cell_type]
        for ophys_container_id in data.ophys_container_id.unique():
            ax[i] = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x='experience_level', y='fraction_responsive',
                                  color='gray', join=True, markers='.', scale=0.25, errwidth=0.25, ax=ax[i])
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
        plt.setp(ax[i].lines, alpha=.3)
        ax[i] = sns.pointplot(data=data, x='experience_level', y='fraction_responsive', hue='experience_level',
                              hue_order=experience_levels, palette=palette, dodge=0, join=False, ax=ax[i])
        ax[i].set_xticklabels(experience_levels, rotation=45, ha='right')
        ax[i].set_ylabel('fraction\nresponsive')
        ax[i].get_legend().remove()
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('')
        if ylim is not None:
            ax[i].set_ylim(0, 1)

    if save_dir:
        fig.subplots_adjust(hspace=0.5)
        fig_title = 'fraction_responsive_cells_' + suffix
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
    cell_types = np.sort(df.cell_type.unique())

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
                                  color='gray', join=True, markers='.', scale=0.25, errwidth=0.25, ax=ax[i])
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


def test_significant_metric_averages(data, metric, column_to_compare='experience_level'):
    """
    run one way anova across experience levels or cell types for a given metric in data,
    based on Alex's stats across experience levels for GLM figures
    data: cell metrics dataframe, each row is one cell_specimen_id in a given ophys_experiment
    metric: column in data representing metric values of interest
    column_to_compare: must be 'experience_level' or 'cell_type'
    """

    from scipy import stats
    import statsmodels.stats.multicomp as mc

    data = data[~data[metric].isnull()].copy()
    if 'Novel +' in data[column_to_compare].unique():
        anova = stats.f_oneway(
            data.query('experience_level == "Familiar"')[metric],
            data.query('experience_level == "Novel"')[metric],
            data.query('experience_level == "Novel +"')[metric])
        mapper = {'Familiar': 0, 'Novel': 1, 'Novel +': 2, }
    elif 'Novel >1' in data[column_to_compare].unique():
        anova = stats.f_oneway(
            data.query('experience_level == "Familiar"')[metric],
            data.query('experience_level == "Novel >1"')[metric],
            data.query('experience_level == "Novel 1"')[metric])
        mapper = {'Familiar': 0, 'Novel 1': 1, 'Novel >1': 2, }
    elif 'Excitatory' in data[column_to_compare].unique():
        anova = stats.f_oneway(
            data.query('cell_type == "Excitatory"')[metric],
            data.query('cell_type == "Sst Inhibitory"')[metric],
            data.query('cell_type == "Vip Inhibitory"')[metric])
        mapper = {'Excitatory': 0, 'Sst Inhibitory': 1, 'Vip Inhibitory': 2, }
    else:
        print('data does not contain experience_level or cell_type')

    comp = mc.MultiComparison(data[metric], data[column_to_compare])
    post_hoc_res = comp.tukeyhsd()
    tukey_table = pd.read_html(post_hoc_res.summary().as_html(), header=0, index_col=0)[0]
    tukey_table = tukey_table.reset_index()

    tukey_table['x1'] = [mapper[str(x)] for x in tukey_table['group1']]
    tukey_table['x2'] = [mapper[str(x)] for x in tukey_table['group2']]
    tukey_table['one_way_anova_p_val'] = anova[1]
    return anova, tukey_table


def add_stats_to_plot(data, metric, colors, ax, ymax=None, column_to_compare='experience_level'):
    """
    add stars to axis indicating across experience level statistics
    x-axis of plots must be experience_levels or cell_types

    data: metrics dataframe, each row is one cell_specimen_id in a given ophys_experiment
    metric: column in data representing metric values of interest
    column_to_compare: must be 'experience_level' or 'cell_type'
    """
    # do anova across experience levels or cell types followed by post-hoc tukey
    anova, tukey = test_significant_metric_averages(data, metric, column_to_compare)

    scale = 0.1
    fontsize = 12

    if ymax is None:
        ytop = ax.get_ylim()[1]
    else:
        ytop = ymax
    y1 = ytop
    y1h = ytop * (1 + scale)
    y2 = ytop * (1 + scale * 2)
    y2h = ytop * (1 + scale * 3)

    if anova.pvalue < 0.05:
        for tindex, row in tukey.iterrows():
            if row.x2 - row.x1 > 1:
                y = y2
                yh = y2h
            else:
                y = y1
                yh = y1h
            if row.reject:
                ax.plot([row.x1, row.x1, row.x2, row.x2], [y, yh, yh, y], 'k-')
                ax.text(np.mean([row.x1, row.x2]), yh, '*', fontsize=fontsize, horizontalalignment='center',
                        verticalalignment='bottom')
            else:
                ax.plot([row.x1, row.x1, row.x2, row.x2], [y, yh, yh, y], 'k-')
                ax.text(np.mean([row.x1, row.x2]), yh, 'ns', fontsize=fontsize, horizontalalignment='center',
                        verticalalignment='bottom')
    else:
        y = y1
        yh = y1h
        ax.plot([0, 0, 1, 1, 1, 2, 2], [y, yh, yh, y, yh, yh, y], 'k-')
        ax.text(.95, ytop * (1 + scale * 1.5), 'ns', color='k', fontsize=fontsize)
    ax.set_ylim(ymax=ytop * (1 + scale * 4))

    return ax, tukey


def get_ci_sem_for_grouped_metric(group):
    """
    Takes grouped dataframe and computes 95% confidence intervals and standard error of the mean
    dataframe should only have one column, corresponding to the metric to compute stats for
    """
    import scipy.stats as st
    # compute standard error of the mean
    sem = st.sem(group.values, nan_policy='omit')
    # compute 95% confidence intervals
    ci = st.norm.interval(alpha=0.95, loc=np.nanmean(group.values), scale=sem)
    ci = [c[0] for c in ci]
    return pd.Series({'CI': ci, 'SEM': sem[0]})


def get_descriptive_stats_for_metric(data, metric, cols_to_groupby):
    """
    group values in data by cols_to_groupby (ex: experience_level), then compute basic stats for metric values.
    stats include mean, std, sem, 95% confidence intervals

    data: dataframe with columns for metric and cols_to_groupby
    metric: string, column of data with numeric values
    cols_to_groupby: list, column(s) in data with categorical values to use for grouping
    """
    # get 95% CI and SEM
    group = data.groupby(cols_to_groupby)[[metric]]
    ci_sem = pd.DataFrame(group.apply(get_ci_sem_for_grouped_metric))

    # get basic descriptive stats
    values = data.groupby(cols_to_groupby)[[metric]].describe()
    # get rid of multi-index with metric name
    values.columns = values.columns.droplevel(0)

    # merge
    values = values.merge(ci_sem, on=cols_to_groupby)
    # add column for metric value
    values['metric'] = metric

    return values


def plot_metric_distribution_by_experience_no_cell_type(metrics_table, metric, event_type, data_type, hue=None, stripplot=False, pointplot=False,
                                                        add_zero_line=False, ylabel=None, ylims=None, save_dir=None, ax=None, suffix=''):
    """
    plot metric distribution across experience levels in metrics_table, with stats across experience levels
    if hue is provided, plots will be split by hue column and stats will be done on hue column differences instead of across experience levels
    plots boxplot by default, can add stripplot (if no hue is provided) or use pointplot instead

    metrics_table: cell metrics table, each row is one cell_specimen_id in one ophys_experiment_id
    metric: column in metrics_table containing metric values, metrics will be plotted using experience level colors unless a hue is provided
    event_type: one of ['changes', 'omissions', 'all']
    data_type: one of ['dff', 'events', 'filtered_events', 'running_speed', 'pupil_width', 'lick_rate']
    hue: column in metrics_table to split metric values by for plotting (ex: 'targeted_structure')
                plots using hue will have 'gray' as palette
    stripplot: Bool, if True, plots each individual cell as swarmplot along with boxplot
                only works when no hue is provided
                if cell_type is 'Excitatory', only shows 25% of cells due to high density
    pointplot: Bool, if True, will use pointplot instead of boxplot and/or stripplot
    ylims: yaxis limits to use; if None, will use +/-1
    save_dir: directory to save to. if None, plot will not be saved
    ax: axes to plot figures on
    """
    data = metrics_table.copy()
    new_experience_levels = utils.get_new_experience_levels()

    if hue:
        if hue == 'targeted_structure':
            hue_order = np.sort(metrics_table[hue].unique())[::-1]
        else:
            hue_order = np.sort(metrics_table[hue].unique())[::-1]
        suffix = '_' + hue + '_' + suffix
    else:
        suffix = '_experience_level' + '_' + suffix
    if (ylims is None) and ('modulation_index' in metric):
        ylims = (-1.1, 1.1)
        ymin = ylims[0]
        ymax = ylims[1]
        # loc = 'lower right'
    elif (ylims is None) and ('response' in metric):
        ymin = 0
        ymax = None
        # loc = 'upper left'
    elif ylims is None:
        print('please provide ylims')
        ymin = 0
        ymax = None
        # loc = 'upper left'
    else:
        ymin = ylims[0]
        ymax = ylims[1]
        # loc = 'upper left'
    order = np.sort(metrics_table['experience_level'].unique())
    colors = utils.get_experience_level_colors()
    if ax is None:
        figsize = (2, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    # stats dataframe to save
    tukey = pd.DataFrame()
    ct_data = data.copy()
    if hue:
        if pointplot:
            ax = sns.pointplot(data=ct_data, y=metric, x='experience_level', order=order, dodge=0.3, join=False,
                               hue=hue, hue_order=hue_order, palette='gray', ax=ax)
        else:
            ax = sns.boxplot(data=ct_data, y=metric, x='experience_level', order=order,
                             width=0.4, hue=hue, hue_order=hue_order, palette='gray', ax=ax)
        ax.legend(fontsize='xx-small', title='')  # , loc=loc)  # bbox_to_anchor=(1,1))
        if ylims:
            ax.set_ylim(ylims)
            # TBD add area or depth comparison stats / stats across hue variable
    else:
        if pointplot:
            ax = sns.pointplot(data=ct_data, x='experience_level', y=metric,
                               palette=colors, ax=ax)
        else:
            ax = sns.boxplot(data=ct_data, x='experience_level', y=metric, width=0.4,
                             palette=colors, ax=ax)
        if stripplot:
            ax = sns.boxplot(data=ct_data, x='experience_level', y=metric, width=0.4,
                             color='white', ax=ax)
            # format to have black lines and transparent box face
            plt.setp(ax.artists, edgecolor='k', facecolor=[0, 0, 0, 0])
            plt.setp(ax.lines, color='k')
            # add strip plot
            ax = sns.stripplot(data=ct_data, size=3, alpha=0.5, jitter=0.2,
                               x='experience_level', y=metric, palette=colors, ax=ax)
        # add stats to plot if only looking at experience levels
        ax, tukey_table = add_stats_to_plot(ct_data, metric, 'white', ax, ymax=ymax)
        # aggregate stats
        tukey_table['metric'] = metric
        tukey = pd.concat([tukey, tukey_table])
        ax.set_ylim(ymin=ymin)
        ax.set_xlim(-0.5, len(order) - 0.5)

        # add line at y=0
        if add_zero_line:
            ax.axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_xticklabels(new_experience_levels, rotation=90,)  # ha='right')
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(metric)

    fig.subplots_adjust(hspace=0.3)
    if save_dir:
        folder = 'metric_distributions'
        filename = event_type + '_' + data_type + '_' + metric + '_distribution' + suffix
        stats_filename = event_type + '_' + data_type + '_' + metric + suffix + '_no_cell_type'
        utils.save_figure(fig, figsize, save_dir, folder, filename)
        try:
            print('saving_stats')
            tukey.to_csv(os.path.join(save_dir, folder, stats_filename + '_tukey.csv'))
            # save descriptive stats
            cols_to_groupby = ['experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, stats_filename + '_values.csv'))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric, hue)
    return ax


def plot_metric_distribution_by_experience(metrics_table, metric, event_type, data_type, hue=None,
                                           stripplot=False, pointplot=False, show_containers=False,
                                           add_zero_line=False, ylabel=None, ylims=None, save_dir=None, ax=None, suffix=''):
    """
    plot metric distribution across experience levels for each cell_type in metrics_table, with stats across experience levels
    if hue is provided, plots will be split by hue column and stats will be done on hue column differences instead of across experience levels
    plots boxplot by default, can add stripplot (if no hue is provided) or use pointplot instead

    metrics_table: cell metrics table, each row is one cell_specimen_id in one ophys_experiment_id
    metric: column in metrics_table containing metric values, metrics will be plotted using experience level colors unless a hue is provided
    event_type: one of ['changes', 'omissions', 'all']
    data_type: one of ['dff', 'events', 'filtered_events', 'running_speed', 'pupil_width', 'lick_rate']
    hue: column in metrics_table to split metric values by for plotting (ex: 'targeted_structure')
                plots using hue will have 'gray' as palette
    stripplot: Bool, if True, plots each individual cell as swarmplot along with boxplot
                only works when no hue is provided
                if cell_type is 'Excitatory', only shows 25% of cells due to high density
    pointplot: Bool, if True, will use pointplot instead of boxplot and/or stripplot
    show_containers: Bool, if True, plot gray lines connecting containers across experience levels
    ylims: yaxis limits to use; if None, will use +/-1
    save_dir: directory to save to. if None, plot will not be saved
    ax: axes to plot figures on
    """
    data = metrics_table.copy()
    # experience_levels = utils.get_experience_levels()
    new_experience_levels = utils.get_new_experience_levels()

    if hue:
        if hue == 'targeted_structure':
            hue_order = np.sort(metrics_table[hue].unique())[::-1]
        else:
            hue_order = np.sort(metrics_table[hue].unique())[::-1]
        suffix = '_' + hue + '_' + suffix
    else:
        suffix = '_experience_level' + '_' + suffix
    if (ylims is None) and ('modulation_index' in metric):
        ylims = (-1.1, 1.1)
        ymin = ylims[0]
        ymax = ylims[1]
        # loc = 'lower right'
    elif (ylims is None) and ('response' in metric):
        ymin = 0
        ymax = None
        # loc = 'upper left'
    elif ylims is None:
        print('please provide ylims')
        ymin = 0
        ymax = None
        # loc = 'upper left'
    else:
        ymin = ylims[0]
        ymax = ylims[1]
        # loc = 'upper left'
    order = np.sort(metrics_table['experience_level'].unique())
    colors = utils.get_experience_level_colors()
    if ax is None:
        figsize = (2, 10)
        fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)
    # stats dataframe to save
    tukey = pd.DataFrame()
    cell_types = utils.get_cell_types()
    for i, cell_type in enumerate(cell_types):
        ct_data = data[data.cell_type == cell_type]
        if hue:
            if pointplot:
                ax[i] = sns.pointplot(data=ct_data, y=metric, x='experience_level', order=order, dodge=0.3, join=False,
                                      hue=hue, hue_order=hue_order, palette='gray', ax=ax[i])
            else:
                ax[i] = sns.boxplot(data=ct_data, y=metric, x='experience_level', order=order,
                                    width=0.4, hue=hue, hue_order=hue_order, palette='gray', ax=ax[i])
            ax[i].legend(fontsize='xx-small', title='')  # , loc=loc)  # bbox_to_anchor=(1,1))
            if ylims:
                ax[i].set_ylim(ylims)
                # TBD add area or depth comparison stats / stats across hue variable
        else:
            if pointplot:
                ax[i] = sns.pointplot(data=ct_data, x='experience_level', y=metric,
                                      palette=colors, ax=ax[i])
            else:
                ax[i] = sns.boxplot(data=ct_data, x='experience_level', y=metric, width=0.4,
                                    palette=colors, ax=ax[i])
            if stripplot:
                ax[i] = sns.boxplot(data=ct_data, x='experience_level', y=metric, width=0.4,
                                    color='white', ax=ax[i])
                # format to have black lines and transparent box face
                plt.setp(ax[i].artists, edgecolor='k', facecolor=[0, 0, 0, 0])
                plt.setp(ax[i].lines, color='k')
                # add strip plot
                if cell_type == 'Excitatory':
                    ct_data = ct_data.reset_index()
                    # get 25% of all data points
                    pct = 0.25
                    n_samples = int(len(ct_data) * pct)
                    print(n_samples, 'is', pct * 100, '% of all', cell_type, 'cells')
                    idx = np.random.choice(np.arange(len(ct_data)), n_samples)
                    # limit to this random subset
                    ct_data = ct_data.loc[idx]
                ax[i] = sns.stripplot(data=ct_data, size=1.5, alpha=0.5, jitter=0.2,
                                      x='experience_level', y=metric, palette=colors, ax=ax[i])
            # add stats to plot if only looking at experience levels
            ax[i], tukey_table = add_stats_to_plot(ct_data, metric, 'white', ax[i], ymax=ymax)
            # aggregate stats
            tukey_table['metric'] = metric
            tukey_table['cell_type'] = cell_type
            tukey = pd.concat([tukey, tukey_table])
            ax[i].set_ylim(ymin=ymin)
            ax[i].set_xlim(-0.5, len(order) - 0.5)

        if show_containers:
            for ophys_container_id in ct_data.ophys_container_id.unique():
                ax[i] = sns.pointplot(data=ct_data[ct_data.ophys_container_id == ophys_container_id], x='experience_level',
                                      y=metric, color='gray', join=True, markers='.', scale=0.25, errwidth=0.25, ax=ax[i])

        # add line at y=0
        if add_zero_line:
            ax[i].axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
        ax[i].set_title(cell_type)
        # ax[i].set_title('')
        ax[i].set_xlabel('')
        ax[i].set_xticklabels(new_experience_levels, rotation=90,)  # ha='right')
        if ylabel:
            ax[i].set_ylabel(ylabel)
        else:
            ax[i].set_ylabel(metric)

    fig.subplots_adjust(hspace=0.3)
    if save_dir:
        folder = 'metric_distributions'
        filename = event_type + '_' + data_type + '_' + metric + '_distribution' + suffix
        stats_filename = event_type + '_' + data_type + '_' + metric + suffix
        utils.save_figure(fig, figsize, save_dir, folder, filename)
        try:
            print('saving_stats')
            # save tukey
            tukey.to_csv(os.path.join(save_dir, folder, stats_filename + '_tukey.csv'))
            # save descriptive stats
            cols_to_groupby = ['cell_type', 'experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, stats_filename + '_values.csv'))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric, hue)
    return ax


def plot_metric_distribution_all_conditions(metrics_table, metric, event_type, data_type, ylabel='metric',
                                            ylims=(0, 1), add_zero_line=True, save_dir=None):
    """
    generates pointplots of the mean +/- CI values of provided metric with experience level on x axis and either area or depth as hue
    plots for entire dataset as well as each project code individually, and saves to a folder called 'metrics_distributions' in save_dir
    function also saves metrics for each set of conditions to a .csv file in the 'metrics_distributions' folder
    metrics_table is the output of visual_behavior.ophys.response_analysis.cell_metrics.get_cell_metrics_for_conditions()

    metrics_table: cell metrics table, each row is one cell_specimen_id in one ophys_experiment_id
    metric: column in metrics_table containing metric values, metrics will be plotted using experience level colors unless a hue is provided
    event_type: one of ['changes', 'omissions', 'all']
    data_type: one of ['dff', 'events', 'filtered_events', 'running_speed', 'pupil_width', 'lick_rate']
    hue: column in metrics_table to split metric values by for plotting (ex: 'targeted_structure')
                plots using hue will have 'gray' as palette
    """

    if event_type is None:
        print('please provide event type for save file prefix')

    # full dataset, average over areas & depths
    plot_metric_distribution_by_experience(metrics_table, metric, stripplot=False, pointplot=True,
                                           add_zero_line=add_zero_line, event_type=event_type, data_type=data_type,
                                           ylabel=ylabel, ylims=ylims, save_dir=save_dir, ax=None)

    # per project code, average over areas & depths
    for project_code in metrics_table.project_code.unique():
        df = metrics_table[metrics_table.project_code == project_code]

        plot_metric_distribution_by_experience(df, metric, stripplot=False, pointplot=True, event_type=event_type, data_type=data_type,
                                               suffix=project_code, add_zero_line=add_zero_line,
                                               ylabel=ylabel, ylims=ylims, save_dir=save_dir, ax=None)

    # full dataset, for each area and depth
    if ('running' in event_type) or ('pupil' in event_type) or ('lick' in event_type):
        # doesnt make sense to look across area and depth for behavior metrics which are only for one session
        pass
    else:
        # only look at VisualBehaviorMultiscope for area depth comparisons
        data = metrics_table[metrics_table.project_code == 'VisualBehaviorMultiscope']
        plot_metric_distribution_by_experience(data, metric, stripplot=False, pointplot=True, add_zero_line=add_zero_line,
                                               event_type=event_type, data_type=data_type, hue='targeted_structure',
                                               ylabel=ylabel, ylims=ylims, save_dir=save_dir, ax=None)

        plot_metric_distribution_by_experience(data, metric, stripplot=False, pointplot=True, add_zero_line=add_zero_line,
                                               event_type=event_type, data_type=data_type, hue='layer',
                                               ylabel=ylabel, ylims=ylims, save_dir=save_dir, ax=None)

        # per project code, for each area and depth
        for project_code in metrics_table.project_code.unique():
            df = metrics_table[metrics_table.project_code == project_code]

            plot_metric_distribution_by_experience(df, metric, stripplot=False, pointplot=True, event_type=event_type, data_type=data_type,
                                                   suffix=project_code, add_zero_line=add_zero_line,
                                                   hue='targeted_structure', ylabel=ylabel, ylims=ylims,
                                                   save_dir=save_dir, ax=None)

            plot_metric_distribution_by_experience(df, metric, stripplot=False, pointplot=True, event_type=event_type, data_type=data_type,
                                                   suffix=project_code, add_zero_line=add_zero_line,
                                                   hue='layer', ylabel=ylabel, ylims=ylims, save_dir=save_dir, ax=None)


def plot_experience_modulation_index(metric_data, event_type, save_dir=None):
    """
    plots experience modulation for some event_type, which is the mean repsonse in familiar vs. novel 1 over the sum,
    and the mean response in novel 1 vs. novel >1 over the sum, giving a value between -1 and 1
    metric_data is the output of visual_behavior.ophys.response_analysis.cell_metrics.compute_experience_modulation_index()

    :param metric_data:
    :param event_type:
    :param save_dir:
    :return:
    """

    data = metric_data[['cell_specimen_id', 'Novel 1 vs. Familiar', 'Novel >1 vs. Familiar', 'cell_type']]
    data = data.melt(id_vars=['cell_specimen_id', 'cell_type'], var_name='comparison',
                     value_vars=['Novel 1 vs. Familiar', 'Novel >1 vs. Familiar'])

    metric = 'value'
    x = 'comparison'
    xorder = np.sort(data[x].unique())

    cell_types = np.sort(data.cell_type.unique())

    # colors = utils.get_experience_level_colors()
    figsize = (1.5, 10)
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
    for i, cell_type in enumerate(cell_types):
        ct_data = data[data.cell_type == cell_type]
    #     ax[i] = sns.barplot(data=ct_data,  x=x, order=xorder, y=metric, dodge=0.5, ax=ax[i])
    #     change_width(ax[i], 0.3)
        ax[i] = sns.pointplot(data=ct_data, order=xorder, join=False,
                              x=x, y=metric, color='gray', ax=ax[i])

        ax[i].axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('')
        ax[i].set_ylim(-0.5, 0.5)
        ax[i].set_xticklabels([x.split('.')[0] + '\n' + x.split('.')[1] for x in xorder], rotation=90, ha='center')
        ax[i].set_ylabel('experience\nmodulation')
    #     ax[i].set_ylim(-1.1, 1.1)
    # fig.tight_layout()
    # fig.suptitle('closest active', x=0.53, y=1.02)
    fig.subplots_adjust(hspace=0.5)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, 'metric_distributions', 'experience_modulation_' + event_type)


def change_width(ax, new_value):
    locs = ax.get_xticks()
    for i, patch in enumerate(ax.patches):
        # current_width = patch.get_width()
        # diff = current_width - new_value

        # change the bar width
        patch.set_width(new_value)

        # recenter the bar
        patch.set_x(locs[i // 4] - (new_value * .5))


# heatmaps ##########################

def plot_cell_response_heatmap(data, timestamps, xlabel='time after change (s)', vmax=0.05,
                               microscope='Multiscope', cbar=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.heatmap(data, cmap='binary', linewidths=0, linecolor='white', square=False,
                     vmin=0, vmax=vmax, robust=True, cbar=cbar,
                     cbar_kws={"drawedges": False, "shrink": 1, "label": 'response'}, ax=ax)

    zero_index = np.where(timestamps == 0)[0][0]
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
    elif 'change' in event_type:
        xlabel = 'time after change (s)'
    else:
        xlabel = 'time (s)'

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
            ax[i].set_xticks(np.arange(0, len(timestamps), 30))  # assuming 30Hz traces
            ax[i].set_xticklabels([int(t) for t in timestamps[::30]])
            # set xlims according to input
            start_index = np.where(timestamps == xlim_seconds[0])[0][0]
            end_index = np.where(timestamps == xlim_seconds[1])[0][0]
            xlims = [start_index, end_index]
            ax[i].set_xlim(xlims)
            ax[i].set_ylabel('')

            if r == len(row_conditions) - 1:
                ax[i].set_xlabel(xlabel)
            else:
                ax[i].set_xlabel('')
            i += 1

    for i in np.arange(0, (len(col_conditions) * len(row_conditions)), len(col_conditions)):
        ax[i].set_ylabel('cells')

    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=1.0, fontsize=18)
    fig.tight_layout()

    if save_dir:
        fig_title = event_type + '_response_heatmap_' + data_type + '_' + col_condition + '_' + row_condition + '_' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)

    return ax

# timeseries plots #################


def addSpan(ax, amin, amax, color='k', alpha=0.3, axtype='x'):
    """
    adds a vertical span to an axis
    """
    if axtype == 'x':
        ax.axvspan(amin, amax, facecolor=color, edgecolor='none', alpha=alpha, linewidth=0)
    if axtype == 'y':
        ax.axhspan(amin, amax, facecolor=color, edgecolor='none', alpha=alpha, linewidth=0)


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
    if xlim is not None:
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
                if color is None:
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

    ln2 = ax.plot(running_timestamps, running_speed, label='running_speed', color=colors[2])
    ax.set_ylabel('running speed\n(cm/s)')
    ax.set_ylim(ymin=-8)

    ax2 = ax.twinx()
    ln3 = ax2.plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color=colors[4])

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

    ax[2].plot(running_timestamps, running_speed, label='running_speed', color=colors[2])
    ax[2].set_ylabel('running\nspeed\n(cm/s)', rotation=0, horizontalalignment='right', verticalalignment='center')
    ax[2].set_ylim(ymin=-8)

    ax[3].plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color=colors[4])
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
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)) + '_' + suffix,
                          formats=['.png', '.pdf'])
    return ax


def sort_trace_csids_by_max_in_window(dff_traces, ophys_timestamps, xlim_seconds):
    traces = dff_traces.copy()
    traces['max'] = np.nan
    for cell_index, cell_specimen_id in enumerate(traces.index.values):
        trace = traces.loc[cell_specimen_id]['dff']
        # limit cell trace to window so yaxes scale properly
        start_ind = np.where(ophys_timestamps < xlim_seconds[0])[0][-1]
        stop_ind = np.where(ophys_timestamps > xlim_seconds[1])[0][0]
        trace = trace[start_ind:stop_ind]
        traces.at[cell_specimen_id, 'dff'] = trace
        traces.at[cell_specimen_id, 'max'] = np.amax(trace)
    traces = traces.sort_values(by='max', ascending=False)
    return traces.index.values


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
    cell_specimen_ids = sort_trace_csids_by_max_in_window(dff_traces, ophys_timestamps, xlim_seconds)
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
        ax[i] = add_stim_color_span(dataset, ax[i], xlim=xlim_seconds, label_changes=label_changes,
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
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)) + '_' + suffix,
                          formats=['.png', '.pdf'])
    return ax

# ### matched cell plots ####


def plot_matched_roi_and_trace(ophys_container_id, cell_specimen_id, limit_to_last_familiar_second_novel=True,
                               use_events=False, filter_events=False, save_figure=True):
    """
    Generates plots characterizing single cell activity in response to stimulus, omissions, and changes.
    Compares across all sessions in a container for each cell, including the ROI mask across days.
    Useful to validate cell matching as well as examine changes in activity profiles over days.
    """
    experiments_table = loading.get_platform_paper_experiment_table()
    if limit_to_last_familiar_second_novel:  # this ensures only one session per experience level
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

    experience_levels = utils.get_experience_levels()
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


# ### behavior plots ####

def plot_behavior_metric_by_experience(stats, metric, title='', ylabel='', ylims=None, best_image=True, show_containers=False,
                                       save_dir=None, folder=None, suffix='', ax=None):
    """
    plots average metric value across experience levels, using experience level colors for average, gray for individual points.

    stats should be a table of behavior metric values loaded using vba.utilities.get_behavior_stats_for_sessions()
    metric is a column of the stats table

    if stats table has a unique row for each image_name in each behavior session, all images will be included in the average,
    unless best_image = True
    if stats table does not have unique images, setting best_image to True will cause an error, as there are no images to filter

    if best_image = True, will sort images by metric value within each experience level and select the top 2 images to plot
    if show_containers = True, will plot a linked gray line for each individual container within the dataset

    returns axis handle
    """
    experience_levels = utils.get_experience_levels()
    new_experience_levels = utils.get_new_experience_levels()
    colors = utils.get_experience_level_colors()

    if ylims is None:
        ymin = 0
        ymax = None
    else:
        ymin = ylims[0]
        ymax = ylims[1]

    if best_image:
        tmp = stats.copy()
        tmp = tmp[tmp.image_name != 'omitted']

        # sort images by metric value within each experience level
        tmp = tmp.groupby(['experience_level', 'image_name']).mean()[[metric]].sort_values(by=['experience_level', metric])

        best_familiar = tmp.loc['Familiar'].index.values[-2:]
        best_novel = tmp.loc['Novel 1'].index.values[-2:]
        best_novel_plus = tmp.loc['Novel >1'].index.values[-2:]

        # get data for images with highest metric value
        familiar_stats = stats[(stats.experience_level == 'Familiar') & (stats.image_name.isin(best_familiar))]
        novel_stats = stats[(stats.experience_level == 'Novel 1') & (stats.image_name.isin(best_novel))]
        novel_plus_stats = stats[(stats.experience_level == 'Novel >1') & (stats.image_name.isin(best_novel_plus))]

        df = pd.concat([familiar_stats, novel_stats, novel_plus_stats])

        data = df.copy()

        suffix = suffix + '_best_image'

    else:
        data = stats.copy()

    colors = utils.get_experience_level_colors()
    experience_levels = utils.get_experience_levels()

    if ax is None:
        figsize = (2, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax = sns.stripplot(data=data, x='experience_level', y=metric, order=experience_levels,
                       orient='v', color='gray', dodge=True, jitter=0.1, size=2, ax=ax, zorder=0)
    if show_containers:
        for ophys_container_id in data.ophys_container_id.unique():
            ax = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x='experience_level', y=metric,
                               order=experience_levels, join=True, orient='v', color='gray',
                               markers='.', scale=0.15, errwidth=0.25, ax=ax)
        suffix = suffix + '_containers'

    ax = sns.pointplot(data=data, x='experience_level', y=metric, order=experience_levels,
                       orient='v', palette=colors, ax=ax)
    ax.set_xticklabels(new_experience_levels, rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    # ax.legend(bbox_to_anchor=(1,1), fontsize='xx-small')

    # add stats to plot if only looking at experience levels
    # stats dataframe to save
    tukey = pd.DataFrame()
    ax, tukey_table = add_stats_to_plot(data, metric, 'white', ax, ymax=ymax)
    # aggregate stats
    tukey_table['metric'] = metric
    tukey = pd.concat([tukey, tukey_table])

    ax.set_ylim(ymin=ymin)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, metric + suffix)
    stats_filename = metric + '_stats' + suffix
    try:
        print('saving_stats')
        tukey.to_csv(os.path.join(save_dir, folder, stats_filename + '_tukey.csv'))
        # save stats
        cols_to_groupby = ['experience_level']
        stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        stats.to_csv(os.path.join(save_dir, folder, stats_filename + '_values.csv'))
    except BaseException:
        print('stats did not save for', metric)


def plot_behavior_metric_across_stages(data, metric, ylabel=None, save_dir=None, folder=None, suffix=''):
    """
    generate boxplot of metric values across behavior stages (gratings flashed, gratings static, familiar, novel)
    with cre line on x-axis and behavior stages as hue
    data: dataframe with one row for each behavior session and columns with metric values
    data must contain 'behavior_stages' column
    """
    cell_types = utils.get_cell_types()
    if ylabel is None:
        ylabel = metric

    # remove passive sessions
    data = data[data.behavior_stage.str.contains('passive') == False]

    behavior_stages = data.behavior_stage.unique()
    color_map = utils.get_behavior_stage_color_map(as_rgb=True)
    colors = [list(color_map[behavior_stage]) for behavior_stage in behavior_stages]
    colors = [[c / 255. for c in color] for color in colors]

    figsize = (7, 3)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(data=data, x='cell_type', y=metric, width=0.8, order=cell_types,
                     hue='behavior_stage', hue_order=behavior_stages, palette=colors, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.legend().remove()
    ax.legend(bbox_to_anchor=(1, 1), fontsize='x-small')

    fig.subplots_adjust(hspace=0.3)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'metric_across_stages_' + metric + suffix)
        # save stats
        stats = data.groupby(['cell_type', 'behavior_stage']).describe()[[metric]]
        stats.to_csv(os.path.join(save_dir, folder, 'metric_across_stages_' + metric + suffix + '_values.csv'))


def plot_days_in_stage(behavior_sessions, stage_column, save_dir=None, folder=None, suffix=None):
    """
    Plot the number of days in each stage, as a boxplot using stage as the hue and cell types on y-axis

    behavior_sessions: behavior sessions table including 'cell_type', 'mouse_id'
    stage_column: column in behavior_sessions to use for grouping of stages, can be 'behavior_stage', 'stimulus_phase', or 'session_type',
    to add 'behavior_stage' column to behavior_sessions, use add_behavior_stage_to_behavior_sessions(behavior_sessions)

    """

    days_in_stage = \
        behavior_sessions.groupby(['mouse_id', stage_column]).count().rename(columns={'equipment_name': 'days_in_stage'})[
            ['days_in_stage']]
    days_in_stage = days_in_stage.reset_index()
    days_in_stage = days_in_stage.merge(behavior_sessions[['mouse_id', 'cell_type', 'project_code']], on='mouse_id')

    behavior_stages = behavior_sessions[stage_column].unique()

    data = days_in_stage.copy()
    if stage_column == 'behavior_stage':
        color_map = utils.get_behavior_stage_color_map(as_rgb=True)
    elif stage_column == 'stimulus_phase':
        color_map = utils.get_stimulus_phase_color_map(as_rgb=True)
    elif stage_column == 'get_session_type_color_map':
        color_map = utils.get_session_type_color_map(as_rgb=True)
    else:
        print('provided stage_column does not have a corresponding colormap')

    colors = [list(color_map[behavior_stage]) for behavior_stage in behavior_stages]
    colors = [[c / 255. for c in color] for color in colors]

    figsize = (7, 3)
    fig, ax = plt.subplots(figsize=figsize)
    # for i, cell_type in enumerate(np.sort(data.cell_type.unique())):
    #     ct_data = data[data.cell_type==cell_type]
    ax = sns.boxplot(data=data, x='cell_type', y='days_in_stage', width=0.8, linewidth=0.8,
                     hue=stage_column, hue_order=behavior_stages, palette=colors, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('days in stage')
    #     ax[i].set_xticklabels(behavior_stages, rotation=90)
    #     ax[i].set_title(cell_type)
    ax.legend().remove()
    ax.legend(bbox_to_anchor=(1, 1), fontsize='x-small')

    fig.subplots_adjust(hspace=0.3)
    if save_dir:
        # save plot
        utils.save_figure(fig, figsize, save_dir, folder, 'days_in_stage' + '_' + stage_column + suffix)
        # save stats
        days_in_stage_stats = data.groupby(['cell_type', stage_column]).describe()
        days_in_stage_stats.to_csv(os.path.join(save_dir, folder, 'days_in_stage_stats.csv'))


def plot_prior_exposures_to_image_set_before_platform_ophys_sessions(platform_experiments, behavior_sessions, save_dir=None, folder=None, suffix='', ax=None):
    """
    Creates a boxplot showing the number of prior exposures to each image set for each experience level (Familiar, Novel, Novel +)
    for the set of mice and sessions in platform_experiments
    Boxplot is distribution of number of prior exposures across mice
    """

    # get the behavior sessions corresponding to the ophys sessions included in platform dataset
    paper_ophys_behavior_sessions = behavior_sessions.loc[platform_experiments.behavior_session_id.unique()]
    exposures = paper_ophys_behavior_sessions.set_index(['experience_level', 'mouse_id'])[['prior_exposures_to_image_set']].reset_index()

    if ax is None:
        figsize = (2.5, 3)
        fig, ax = plt.subplots(figsize=figsize)

    colors = utils.get_experience_level_colors()
    experience_levels = utils.get_experience_levels()

    ax = sns.boxplot(data=exposures, x='experience_level', y='prior_exposures_to_image_set',
                     order=experience_levels, palette=colors, width=0.5, ax=ax)
    ax.set_ylabel('# sessions')
    ax.set_xlabel('')

    stats = exposures.groupby(['experience_level']).describe()[['prior_exposures_to_image_set']]
    stats.columns = stats.columns.droplevel(0)

    xticklabels = utils.get_new_experience_levels()
    ax.set_xticklabels(xticklabels, rotation=90,)
    ax.set_title('total stimulus exposure')

    for i, experience_level in enumerate(experience_levels):
        y = int(np.round(stats.loc[experience_level]['mean'], 0))
        if experience_level == 'Novel 1':
            text = '0'
            y = y + 4
            i = 0.85
        elif experience_level == 'Familiar':
            y = y + 12
            text = str(int(np.round(stats.loc[experience_level]['mean'], 0))) + '+/-' + str(int(np.round(stats.loc[experience_level]['std'], 0)))
        else:
            y = y + 8
            text = str(int(np.round(stats.loc[experience_level]['mean'], 0))) + '+/-' + str(int(np.round(stats.loc[experience_level]['std'], 0)))
        ax.text(i + 0.1, y, text, fontsize=14, rotation='horizontal')

    if save_dir:
        # save plot
        utils.save_figure(fig, figsize, save_dir, folder, 'stimulus_exposures_before_platform_expts_boxplot' + suffix)
        # save stats
        stats = exposures.groupby(['experience_level']).describe()[['prior_exposures_to_image_set']]
        stats.to_csv(os.path.join(save_dir, folder, 'stimulus_exposures_before_platform_expts_stats.csv'))


def plot_prior_exposures_per_cell_type_for_novel_plus(platform_experiments, behavior_sessions, save_dir=None,
                                                      folder=None, suffix='', ax=None):
    """
    Creates a boxplot showing the number of  prior exposures to novel image set for Novel + sessions included in the platform paper
    shows striplot of prior exposures across mice and pointplot of averages plus stats
    """

    cell_types = utils.get_cell_types()

    # get the behavior sessions corresponding to the ophys sessions included in platform dataset
    paper_ophys_behavior_sessions = behavior_sessions.loc[platform_experiments.behavior_session_id.unique()]
    exposures = paper_ophys_behavior_sessions.set_index(['cell_type', 'experience_level', 'mouse_id'])[
        ['prior_exposures_to_image_set']].reset_index()

    #     print(prior_exposures.groupby(['cell_type', 'experience_level']).describe()[['prior_exposures_to_image_set']])

    # limit to Novel+ sessions
    if 'Novel >1' in exposures.experience_level.unique():
        exposures = exposures[exposures.experience_level == 'Novel >1']
    else:
        exposures = exposures[exposures.experience_level == 'Novel +']

    if ax is None:
        figsize = (2.5, 3)
        fig, ax = plt.subplots(figsize=figsize)

    #     ax = sns.boxplot(data=exposures, x='cell_type', y='prior_exposures_to_image_set',
    #                order=cell_types, palette='gray', width=0.5, ax=ax)

    ax = sns.violinplot(data=exposures, x='cell_type', y='prior_exposures_to_image_set', order=cell_types,
                        orient='v', color='white', ax=ax)
    ax = sns.stripplot(data=exposures, x='cell_type', y='prior_exposures_to_image_set', order=cell_types,
                       orient='v', color='gray', dodge=True, size=2, jitter=0.2, ax=ax)

    ax.set_ylabel('# sessions')
    ax.set_xlabel('')

    stats = exposures.groupby(['cell_type']).describe()[['prior_exposures_to_image_set']]
    stats.columns = stats.columns.droplevel(0)

    xticklabels = cell_types
    for i, cell_type in enumerate(cell_types):
        text = str((np.round(stats.loc[cell_type]['mean'], 1))) + '+/-' + str(
            (np.round(stats.loc[cell_type]['std'], 1)))
        xticklabels[i] = xticklabels[i] + '\n(' + text + ')'
    # xticklabels = [experience_level+'\n N = '+str(int(np.round(exposures.loc[experience_level]['mean'],0)))+'+/-'+str(int(np.round(exposures.loc[experience_level]['std'],0))) if experience_level!='Novel 1' else 'Novel 1\nN = 0' for experience_level in experience_levels]
    ax.set_xticklabels(xticklabels, rotation=90, )
    ax.set_title('Novel sessions\nprior to Novel +')
    ax.set_ylim(ymin=0)

    ymax = ax.get_ylim()[1]
    tukey = pd.DataFrame()
    ax, tukey_table = add_stats_to_plot(exposures, 'prior_exposures_to_image_set', 'white', ax, ymax=ymax,
                                        column_to_compare='cell_type')
    # aggregate stats
    tukey_table['metric'] = 'prior_exposures_to_image_set'
    tukey = pd.concat([tukey, tukey_table])

    if save_dir:
        # save plot
        utils.save_figure(fig, figsize, save_dir, folder, 'stimulus_exposures_before_novel_plus' + suffix)
        # save stats
        stats = exposures.groupby(['cell_type', 'experience_level']).describe()[['prior_exposures_to_image_set']]
        stats.to_csv(os.path.join(save_dir, folder, 'stimulus_exposures_before_novel_plus_stats.csv'))


def plot_total_stimulus_exposures(behavior_sessions, save_dir=None, folder=None, suffix='', ax=None):
    """
    Creates a boxplot showing the number of sessions for each experience level (Familiar, Novel, Novel +)
    for the set of mice included in behavior_sessions
    """

    # count number of sessions for each experience level
    exposures = behavior_sessions.groupby(['experience_level', 'mouse_id']).count()[
        ['session_type']].reset_index().rename(columns={'session_type': 'n_sessions'})
    #     print(exposures.groupby(['experience_level']).describe()[['n_sessions']])

    if ax is None:
        figsize = (2.5, 3)
        fig, ax = plt.subplots(figsize=figsize)

    colors = utils.get_experience_level_colors()
    experience_levels = utils.get_experience_levels()
    new_experience_levels = utils.get_new_experience_levels()

    ax = sns.boxplot(data=exposures, x='experience_level', y='n_sessions',
                     order=experience_levels, palette=colors, width=0.5, ax=ax)
    ax.set_ylabel('# sessions')
    ax.set_xlabel('')

    stats = exposures.groupby(['experience_level']).describe()[['n_sessions']]
    stats.columns = stats.columns.droplevel(0)

    xticklabels = new_experience_levels
    # xticklabels = [experience_level+'\n N = '+str(int(np.round(exposures.loc[experience_level]['mean'],0)))+'+/-'+str(int(np.round(exposures.loc[experience_level]['std'],0))) if experience_level!='Novel 1' else 'Novel 1\nN = 0' for experience_level in experience_levels]
    ax.set_xticklabels(xticklabels, rotation=90, )
    ax.set_title('stimulus exposure\nall sessions')

    for i, experience_level in enumerate(experience_levels):
        y = int(np.round(stats.loc[experience_level]['mean'], 0))
        if experience_level == 'Novel 1':
            text = '0'
            y = y + 4
            i = 0.85
        elif experience_level == 'Familiar':
            y = y + 12
            text = str(int(np.round(stats.loc[experience_level]['mean'], 0))) + '+/-' + str(
                int(np.round(stats.loc[experience_level]['std'], 0)))
        else:
            y = y + 8
            text = str(int(np.round(stats.loc[experience_level]['mean'], 0))) + '+/-' + str(
                int(np.round(stats.loc[experience_level]['std'], 0)))
        ax.text(i + 0.1, y, text, fontsize=14, rotation='horizontal')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'total_stimulus_exposures_all_sessions_boxplot' + suffix)
        # save stats
        stats = exposures.groupby(['experience_level']).describe()[['n_sessions']]
        stats.to_csv(os.path.join(save_dir, folder, 'total_stimulus_exposures_all_sessions_stats.csv'))


def plot_stimulus_exposure_prior_to_imaging(behavior_sessions, save_dir=None, folder=None, suffix='', ax=None):
    """
    Creates a boxplot showing the number of sessions for each experience level (Familiar, Novel, Novel +)
    prior to the start of 2P imaging, for the set of mice included in behavior_sessions
    """

    data = behavior_sessions.copy()
    # limit to non-ophys sessions
    data = data[data.session_type.str.contains('OPHYS') == False]
    # count number of sessions of each experience level
    exposures = data.groupby(['experience_level', 'mouse_id']).count()[['session_type']].reset_index().rename(
        columns={'session_type': 'n_sessions'})

    if ax is None:
        figsize = (2.5, 3)
        fig, ax = plt.subplots(figsize=figsize)

    exp = exposures.experience_level.unique()[::-1]

    colors = utils.get_experience_level_colors()
    c = [colors[0], [0.5, 0.5, 0.5]][::-1]

    ax = sns.boxplot(data=exposures, x='experience_level', y='n_sessions',
                     order=exp, palette=c, width=0.5, ax=ax)
    ax.set_ylabel('# sessions')
    ax.set_xlabel('')

    stats = exposures.groupby(['experience_level']).describe()[['n_sessions']]
    stats.columns = stats.columns.droplevel(0)

    ax.set_xticklabels(['Gratings', 'Familiar\nimages'], rotation=90)
    ax.set_title('stimulus exposure\nduring training')

    for i, experience_level in enumerate(exposures.experience_level.unique()[::-1]):
        y = int(np.round(stats.loc[experience_level]['mean'], 0))
        if experience_level == 'Novel 1':
            text = '0'
            y = y + 4
            i = 0.85
        elif experience_level == 'Familiar':
            y = y + 12
            text = str(int(np.round(stats.loc[experience_level]['mean'], 0))) + '+/-' + str(
                int(np.round(stats.loc[experience_level]['std'], 0)))
        else:
            y = y + 4
            text = str(int(np.round(stats.loc[experience_level]['mean'], 0))) + '+/-' + str(
                int(np.round(stats.loc[experience_level]['std'], 0)))
        ax.text(i + 0.1, y, text, fontsize=14, rotation='horizontal')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'stimulus_exposure_prior_to_imaging_boxplot' + suffix)
        # save stats
        stats = exposures.groupby(['experience_level']).describe()[['n_sessions']]
        stats.to_csv(os.path.join(save_dir, folder, 'stimulus_exposure_prior_to_imaging_stats.csv'))


def plot_training_history_for_mice(behavior_sessions, color_column='session_type', color_map=sns.color_palette(),
                                   save_dir=None, folder=None, suffix='', ax=None):
    """
    plots the session sequence for all mice in behavior_sessions table, sorted by total # of sessions per mouse

    sessions are colored by the provided color_column and color_map
    values of color_column must match keys of color_map
    acceptable pairs for color_column and color_map
    color_column = 'session_type' : color_map = utils.get_session_type_color_map()
    color_column = 'stimulus' : color_map = utils.get_stimulus_color_map(as_rgb=True)
    color_column = 'stimulus_phase' : color_map = utils.get_stimulus_phase_color_map(as_rgb=True)

    """
    # group by mice and count n_session per mouse to get the max n_sessions and list of mouse_ids to plot
    n_sessions = \
        behavior_sessions.groupby(['cre_line', 'mouse_id']).count().rename(columns={'equipment_name': 'n_sessions'})[['n_sessions']]
    n_sessions = n_sessions.reset_index()
    n_sessions = n_sessions.sort_values(by=['cre_line', 'n_sessions'])
    max_n_sessions = np.amax(n_sessions.n_sessions.values)
    mouse_ids = n_sessions.mouse_id.values

    # get ytick labels based on number of mice per cre line
    yticklabels = [0]
    for i, cre_line in enumerate(n_sessions.cre_line.unique()):
        yticklabels.append(yticklabels[i] + len(n_sessions[n_sessions.cre_line == cre_line]))

    n_mouse_ids = len(mouse_ids)

    # create an array to fill in with session colors per mouse
    img = np.empty((n_mouse_ids, max_n_sessions, 3))
    img[:] = 256  # make the default value of 256 which is white in RGB space

    # loop through mice
    for mouse, mouse_id in enumerate(mouse_ids):
        # sort session in acquisition date order
        sessions = behavior_sessions[behavior_sessions.mouse_id == mouse_id].sort_values('date_of_acquisition')
        # fill in image array with the color from color_map for the corresponding color_col
        for session, session_id in enumerate(sessions.index.values):
            color_column_value = sessions.loc[session_id][color_column]
            img[mouse, session, :] = color_map[color_column_value]

    # create plot with expt colors image
    if ax is None:
        figsize = (10, n_mouse_ids * 0.1)
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img.astype(int), aspect='auto')
    ax.set_ylim(0, len(mouse_ids))
    ax.invert_yaxis()
    ax.set_yticks(yticklabels)
    ax.set_yticklabels(yticklabels, fontdict={'verticalalignment': 'top'})
    ax.set_xlabel('Session number')
    ax.set_ylabel('Mouse number')
    ax.set_title('Training history')

    # label with cell type
    for i, cre_line in enumerate(n_sessions.cre_line.unique()):
        cell_type = utils.convert_cre_line_to_cell_type(cre_line)
        ax.text(-2, (yticklabels[i] + yticklabels[i + 1]) / 2., cell_type.split(' ')[0], fontsize=16, ha='center',
                va='center', rotation='vertical')

        #     ax.axis('off')

        #     for mouse, mouse_id in enumerate(mouse_ids):
        #         # plot cre line
        #         cre_line = behavior_sessions[behavior_sessions.mouse_id == mouse_id].cre_line.values[0]
        #         if cre_line == 'Gad2-IRES-Cre': # square
        #             ax.text(-1, mouse, '\u25a1', fontsize=16, ha='center', va='center',)
        #         elif cre_line == 'Rbp4-Cre_KL100': # triangle
        #             ax.text(-1, mouse, '\u25b2', fontsize=16, ha='center', va='center', color='black')
        #         else: # circle
        #             ax.text(-1, mouse, '\u25cf', fontsize=16, ha='center', va='center', color='black')
        #         # plot mouse_id
        #         sessions_for_mouse = n_sessions[n_sessions.mouse_id==mouse_id].n_sessions.values[0]
        #         ax.text(sessions_for_mouse, mouse, str(mouse_id), fontsize=17, ha='left', va='center',)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'training_history' + suffix)
    return ax


# examples
if __name__ == '__main__':

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
