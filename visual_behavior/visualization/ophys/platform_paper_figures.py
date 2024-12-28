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
from visual_behavior_glm import GLM_visualization_tools as gvt
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.top': False, 'axes.spines.right': False})  # ticks or white
sns.set_palette('deep')

plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

# plot data for a given session

def plot_max_intensity_projection(dataset, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    max_projection = dataset.max_projection.data
    ax.imshow(max_projection, cmap='gray', vmax=np.percentile(max_projection, 99))
    ax.axis('off')
    return ax


# ophys_container_ids = list(dataset_dict.keys())

# ophys_container_id = ophys_container_ids[0]

def plot_all_planes_all_sessions_for_mouse(dataset_dict, mouse_expts, session_id_for_area_depths=None,
                                           save_dir=None, folder=None, ax=None):
    '''
    For a given mouse, plot all FOVs across all sessions in a grid. If an FOV for a particular container is missing, leave a blank axis
    '''
    # ophys_container_ids = list(dataset_dict.keys())
    mouse_expts = mouse_expts.sort_values(by=['date_of_acquisition', 'targeted_structure', 'imaging_depth'])
    mouse_id = mouse_expts.mouse_id.values[0]
    ophys_container_ids = mouse_expts.ophys_container_id.unique()
    ophys_session_ids = mouse_expts.ophys_session_id.unique()

    if ax is None:
        figsize = (8, 10)
        fig, ax = plt.subplots(len(ophys_container_ids), len(ophys_session_ids), figsize=figsize)
        ax = ax.ravel()

    i = 0
    for c, ophys_container_id in enumerate(ophys_container_ids):
        container_data = mouse_expts[(mouse_expts.ophys_container_id == ophys_container_id)]
        area = container_data.targeted_structure.values[0]
        depth = int(container_data.imaging_depth.values[0])

        for s, ophys_session_id in enumerate(ophys_session_ids):
            # for s, session_type in enumerate(session_types):
            # ophys_experiment_id = container_data[(container_data.session_type==session_type)].index.values[0]
            try:
                ophys_experiment_id = \
                container_data[(container_data.ophys_session_id == ophys_session_id)].index.values[0]
                session_type = \
                container_data[(container_data.ophys_session_id == ophys_session_id)].session_type.values[0]

                dataset = dataset_dict[ophys_container_id][ophys_experiment_id]
                ax[i] = plot_max_intensity_projection(dataset, ax=ax[i])
            except:
                print('could not plot for experiment', ophys_experiment_id, session_type, area, depth)
            if s == 0:
                if session_id_for_area_depths:
                    tmp = mouse_expts[(mouse_expts.ophys_session_id == session_id_for_area_depths) &
                                      (mouse_expts.ophys_container_id == ophys_container_id)]
                    area = tmp.targeted_structure.values[0]
                    depth = int(tmp.imaging_depth.values[0])
                ax[i].text(s=area + ' ' + str(depth), x=-20, y=dataset.max_projection.data.shape[0] / 2,
                           ha='right', va='center', rotation=90, fontsize=8)
            if c == 0:
                ax[i].set_title(str(ophys_session_id) + '\n' + session_type, fontsize=6)
            i += 1

    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    if save_dir:
        cre = dataset.metadata['cre_line'][:3]
        utils.save_figure(fig, figsize, save_dir, folder, str(mouse_id) + '_' + cre + '_max_projection_images')


def aggregate_traces_for_session(dataset_dict, session_metadata, trace_type='dff'):
    '''
    Loop through the fields of view in session_metadata, get traces for each, and combine into a single array

    dataset_dict is a dictionary containing the SDK dataset object for each field of view
        first key is ophys_container_id, second key is ophys_experiment_id
    session_metadata is a subset of an ophys_experiment_table, limited to the experiments from the session of interest
    trace_type is one of ['dff', 'events', 'filtered_events'] to plot

    returns array of traces and the session_metadata for each FOV with the N cells and other useful info added
    '''
    ophys_container_ids = list(dataset_dict.keys())

    # get data for one experiment to start the traces array
    ophys_container_id = ophys_container_ids[0]
    ophys_experiment_id = session_metadata[session_metadata.ophys_container_id == ophys_container_id].index.values[0]
    ophys_session_id = session_metadata.ophys_session_id.values[0]
    dataset = dataset_dict[ophys_container_id][ophys_experiment_id]
    # loop through all FOVs for this session and aggregate the traces
    # area_depth_info = []
    for c, ophys_container_id in enumerate(ophys_container_ids):
        ophys_experiment_id = session_metadata[session_metadata.ophys_container_id == ophys_container_id].index.values[
            0]
        dataset = dataset_dict[ophys_container_id][ophys_experiment_id]
        # get traces
        if trace_type == 'dff':
            traces = dataset.dff_traces['dff'].values
        elif trace_type == 'events':
            traces = dataset.events['events'].values
        elif trace_type == 'filtered_events':
            traces = dataset.events['filtered_events'].values
        # aggregate
        if c == 0:
            all_traces = np.vstack(traces)
        else:
            traces = np.vstack(traces)
            all_traces = np.vstack((all_traces, traces))
        # add useful info to FOV metadata
        session_metadata.loc[ophys_experiment_id, 'n_cells'] = int(traces.shape[0])
        session_metadata.loc[ophys_experiment_id, 'ophys_frame_rate'] = dataset.metadata['ophys_frame_rate']
        session_metadata.loc[ophys_experiment_id, 'trace_type'] = trace_type
        session_metadata.loc[ophys_experiment_id, 'container_order'] = c
    session_metadata = session_metadata.sort_values(by=['targeted_structure', 'imaging_depth'])

    return all_traces, session_metadata


def plot_all_traces_heatmap(all_traces, session_metadata, timestamps=None, cmap='gray_r', save_dir=None, ax=None):
    '''
    Plot heatmap for all traces across multiple fields of view in a session

    all_traces is an array of all traces in the session stacked
    session_metadata is a dataframe containing experiment metadata
    in addition to the n_cells in each plane, and the order in which the FOV traces are stacked
    these can be generated using the function aggregate_traces_for_session

    '''
    if timestamps is not None:
        ophys_frame_rate = np.mean(1 / np.diff(timestamps))
    else:
        ophys_frame_rate = session_metadata.ophys_frame_rate.values[0]  # should be the same for all expts in a session

    if ax is None:
        figsize = (15, 5)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(all_traces, cmap=cmap, vmin=0, vmax=np.percentile(all_traces, 95),
                     cbar_kws={'label': 'dF/F', 'pad': 0.1}, ax=ax)
    ax.set_ylim(0, all_traces.shape[0])
    ax.set_xlim(0, all_traces.shape[1])
    xticks = np.arange(0, all_traces.shape[1], ophys_frame_rate * 60 * 5)  # 11Hz * 60s * 5 mins
    ax.set_xticks(xticks)
    len_mins = (all_traces.shape[1] / ophys_frame_rate) / 60
    xticklabels = [int(t) for t in np.arange(0, len_mins, 5)]
    ax.set_xticklabels(xticklabels)  # tick every 5 mins
    ax.set_xlabel('Time in session (minutes)')

    # put ticks on right side also and keep box around plot
    ax.tick_params(which='both', bottom=True, top=False, right=True, left=True,
                   labelbottom=True, labeltop=False, labelright=False, labelleft=False)
    sns.despine(ax=ax, top=False, right=False, left=False, bottom=False, offset=False, trim=False)

    # label area depths
    # loop through FOVs
    cell_count = 0
    yticks = []
    yticks.append(0)
    for container_order in np.sort(session_metadata.container_order.unique()):
        this_expt_info = session_metadata[session_metadata.container_order == container_order]
        n_cells = this_expt_info.n_cells.values[0]
        # get midpoint of this section for label
        y = cell_count + (n_cells / 2)
        s = this_expt_info.targeted_structure.values[0] + ' ' + str(this_expt_info.imaging_depth.values[0])
        # print(s, n_cells)
        # add label for area depth on left side of plot
        x = -800
        ax.text(s=s, x=x, y=y, rotation=0, ha='right', va='center', fontsize=16)
        # get cell count for ticks
        cell_count = cell_count + n_cells
        yticks.append(cell_count)

    # set yticks in increments of FOV
    ax2 = ax.twinx()
    ax.set_yticks(np.asarray(yticks))
    ax2.set_yticks(np.asarray(yticks))
    ax2.set_ylabel('Cells')
    # flip so zero / VISp on top
    ax.invert_yaxis()
    ax2.invert_yaxis()

    if save_dir:
        ophys_session_id = session_metadata.ophys_session_id.values[0]
        mouse_id = session_metadata.mouse_id.values[0]
        cre = session_metadata.cre_line.values[0][:3]
        trace_type = session_metadata.trace_type.values[0]
        session_type = session_metadata.session_type.values[0]
        filename = str(mouse_id) + '_' + str(
            ophys_session_id) + '_' + session_type + '_' + cre + '_' + trace_type + '_' + cmap
        fig.suptitle(filename, x=0.4, y=1.1, fontsize=16)
        utils.save_figure(fig, figsize, save_dir, 'traces_heatmaps', filename)

    return ax



# basic characterization #########################

def plot_cell_count_by_depth(cells_table, project_code=None, suptitle=None, horiz=True,
                             save_dir=None, folder=None, suffix='', ax=None):
    if project_code == 'VisualBehaviorMultiscope4areasx2d':
        areas = ['VISp', 'VISl', 'VISal', 'VISam']
        colors = sns.color_palette('Paired')[:4]
        hue = True
        bins = 10
    elif project_code == 'VisualBehaviorMultiscope':
        areas = ['VISp', 'VISl']
        colors = sns.color_palette('Paired')[:2]
        hue = True
        bins = 10
    else:
        hue = False
        areas = ['VISp']
        colors = sns.color_palette('Paired')[:1]
        bins = 2

    # bins = len(cells_table.groupby(['binned_depth']).count()) * 2
    bins = 10
    binwidth = 50
    if ax is None:
        if horiz:
            figsize = (12, 3)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
        else:
            figsize = (3, 10)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharey=True)
    for i, cell_type in enumerate(utils.get_cell_types()):
        if hue:
            ax[i] = sns.histplot(data=cells_table[cells_table.cell_type == cell_type], #bins=20,
                                 binwidth=binwidth, discrete=False, binrange=[0, 400],
                                 hue='targeted_structure', y='imaging_depth', hue_order=areas,
                                 palette=colors, multiple='stack', stat='count', ax=ax[i])
            ax[i].get_legend().remove()
        else:
            ax[i] = sns.histplot(data=cells_table[cells_table.cell_type == cell_type],
                                 binwidth=binwidth, discrete=False, binrange=[0, 400],
                                  y='imaging_depth', color=colors[0], stat='count', ax=ax[i])
        title = cell_type + '\n n = ' + str(len(cells_table[cells_table.cell_type == cell_type])) + ' cells, ' + str(
            len(cells_table[cells_table.cell_type == cell_type].mouse_id.unique())) + ' mice'
        ax[i].set_title(title)
        ax[i].invert_yaxis()
        ax[i].set_ylim(400, 0)
        if horiz:
            ax[i].set_xlabel('Cell count')
        else:
            ax[i].set_xlabel('')
        ax[i].set_ylabel('')
    ax[i].set_xlabel('Cell count')
    if hue:
        ax[0].legend(areas[::-1], fontsize='xx-small', bbox_to_anchor=(1,1))
    ax[0].set_ylabel('Imaging depth (um)')
    if horiz:
        plt.subplots_adjust(wspace=0.3)
        plt.suptitle(suptitle, x=0.5, y=1.2)
    else:
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.suptitle(suptitle, x=0.5, y=0.96, fontsize=18)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'cell_count_by_depth_areas' + suffix)
    return ax


def plot_n_cells_per_plane_by_depth(cells_table, suptitle=None, save_dir=None, folder=None, ax=None):

    n_cells = cells_table.groupby(['cell_type', 'binned_depth', 'ophys_experiment_id']).count().rename(columns={'cell_specimen_id':'n_cells'}).reset_index()

    if ax is None:
        figsize = (12, 3)
        fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for i, cell_type in enumerate(utils.get_cell_types()):
        ax[i] = sns.stripplot(data=n_cells[n_cells.cell_type==cell_type],
                              y='binned_depth', x='n_cells', orient='h', ax=ax[i])
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('# Cells per plane')
        ax[i].set_ylabel('')
    ax[0].set_ylabel('Binned depth (um)')
    plt.subplots_adjust(wspace=0.4)
    plt.suptitle(suptitle, x=0.5, y=1.2)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'cell_count_by_depth_areas_'+suptitle)
    return ax


def plot_n_planes_per_depth(experiments_table, suptitle=None, save_dir=None, folder=None, ax=None):

    n_expts = experiments_table.groupby(['cell_type', 'binned_depth']).count().rename(columns={'ophys_session_id':'n_expts'}).reset_index()

    if ax is None:
        figsize = (12, 3)
        fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for i, cell_type in enumerate(utils.get_cell_types()):
        ax[i] = sns.barplot(data=n_expts[n_expts.cell_type==cell_type], y='binned_depth', x='n_expts',
                            orient='h', color='gray', width=0.5, ax=ax[i])
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('# Imaging planes')
        ax[i].set_ylabel('')
    ax[0].set_ylabel('Binned depth (um)')
    plt.subplots_adjust(wspace=0.4)
    plt.suptitle(suptitle, x=0.5, y=1.2)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'cell_count_by_depth_areas_'+suptitle)
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
                                  color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i])
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
        plt.setp(ax[i].lines, alpha=.3)
        ax[i] = sns.pointplot(data=data, x='experience_level', y='total_cells', hue='experience_level',
                              hue_order=experience_levels, palette=palette, dodge=0, linewidth='none', ax=ax[i])
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
                                            project_code=None, timestamps=None, palette=None, sharey=False,
                                            title=None, suptitle=None, xlabel='Time (s)', ylabel='Response',
                                            horizontal=True, xlim_seconds=None, interval_sec=1, legend=True,
                                            save_dir=None, folder=None, suffix='', ax=None):
    '''
    Function to plot a population average response across multiple conditions from a dataframe containing event aligned timeseries,
    where axes_column defines the axes conditions and hue_column defines the colors of traces within each axes condition.
    axes_column and hue_column must be columns of the multi_session_df.
    multi_session_df must contain a column for 'mean_trace' and rows should be individual cells' average responses to a specific condition.
    also works for behavior timeseries, in which case rows are averages across an experiment or subset of an experiment rather than individual cells.

    event_type is one of ['changes', 'omissions', 'images']
    this determines how stimuli will be plotted overlaid with the trace - changes in blue, omissions with dotted line, repeated images in gray

    data_type is one of ['dff', 'events', 'filtered_events', 'running_speed', 'pupil_width', 'lick_rate']

    interval_sec determines the interval of the xtick labels (ex: ticks every 1 second or 0.5 seconds)
    xlim_seconds is the range of x-axis, which must be the same or shorter than the range of the data in the 'mean_response' column of the multi_session_df.
    timestamps can be provided, or inferred from the 'trace_timestamps' column of the multi_session_df.

    event aligned timeseries can be computed using brain_observatory_utilities function 'get_stimulus_response_df' here:
    https://github.com/AllenInstitute/brain_observatory_utilities/blob/main/brain_observatory_utilities/datasets/optical_physiology/data_formatting.py#L441
    Followed by a groupby and mean on the conditions of interest.

    '''

    if palette is None:
        palette = utils.get_experience_level_colors()

    sdf = multi_session_df.copy()

    # get timestamps
    if 'trace_timestamps' in sdf.keys():
        timestamps = sdf.trace_timestamps.values[0]
    elif timestamps is not None:
        timestamps = timestamps
    else:
        print('provide timestamps or provide a multi_session_df with a trace_timestamps column')

    # set formatting options
    if xlim_seconds is None:
        xlim_seconds = [timestamps[0], timestamps[-1]]
    if event_type == 'omissions':
        omitted = True
        change = False
    elif event_type == 'changes':
        omitted = False
        change = True
    else:
        omitted = False
        change = False

    # get conditions to plot
    hue_conditions = np.sort(sdf[hue_column].unique())
    axes_conditions = np.sort(sdf[axes_column].unique())

    # if there is only one axis condition, set n conditions for plotting to 2 so it can still iterate
    if len(axes_conditions) == 1:
        n_axes_conditions = 2
        ax_to_xlabel = 1
    else:
        n_axes_conditions = len(axes_conditions)

    # set plot size depending on what type of data it is
    if data_type in ['dff', 'events', 'filtered_events']:
        if horizontal:
            figsize = (5 * n_axes_conditions, 2.5)
        else:
            figsize = (3, 3 * n_axes_conditions)  # for changes and omissions
    elif data_type in ['running_speed', 'pupil_width', 'lick_rate']:
        if horizontal:
            figsize = (6 * n_axes_conditions, 3)  # for behavior timeseries
        else:
            figsize = (2.5, 3 * n_axes_conditions)  # for image response

    # create axes
    if ax is None:
        format_fig = True
        if horizontal:
            suffix = suffix+'_horiz'
            fig, ax = plt.subplots(1, n_axes_conditions, figsize=figsize, sharey=sharey)
        else:
            fig, ax = plt.subplots(n_axes_conditions, 1, figsize=figsize, sharex=sharey)
    else:
        format_fig = False

    # loop over conditions and plot
    for i, axis in enumerate(axes_conditions):
        for c, hue in enumerate(hue_conditions):
            # try:
            cdf = sdf[(sdf[axes_column] == axis) & (sdf[hue_column] == hue)]
            traces = cdf.mean_trace.values
            # plot average of all traces for this condition
            ax[i] = utils.plot_mean_trace(np.asarray(traces), timestamps, ylabel=ylabel,
                                          legend_label=hue, color=palette[c], interval_sec=interval_sec,
                                          xlim_seconds=xlim_seconds, ax=ax[i])
            # plot stimulus timing overlaid on trace
            ax[i] = utils.plot_flashes_on_trace(ax[i], timestamps, change=change, omitted=omitted)

            # color title by experience level if axes are experience levels
            if axes_column == 'experience_level':
                title_colors = utils.get_experience_level_colors()
                ax[i].set_title(axis, color=title_colors[i], fontsize=20)
            else:
                ax[i].set_title(axis)
            if title:
                ax[i].set_title(title)
            ax[i].set_xlim(xlim_seconds)
            ax[i].set_xlabel(xlabel, fontsize=16)
            ax[i].set_ylabel('')
            ax[i].set_xlabel('')
            ax[i].tick_params(axis='both', which='major', labelsize=14)
    # formatting
    if format_fig:
        if horizontal:
            ax[0].set_ylabel(ylabel)
            if n_axes_conditions == 3:
                ax[1].set_xlabel(xlabel)
            else:
                ax[0].set_xlabel(xlabel)
        else:
            ax[1].set_ylabel(ylabel)
            ax[i].set_xlabel(xlabel)
    if legend:
        ax[i].legend(['active', 'passive'], loc='upper center', fontsize='x-small', bbox_to_anchor=(1.3,1))
    if project_code:
        if suptitle is None:
            suptitle = 'population average - ' + data_type + ' response - ' + project_code[14:]
    if suptitle:
        if horizontal:
            y = 1.1
        else:
            y = 0.97
        plt.suptitle(suptitle, x=0.51, y=y, fontsize=18)

    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.rcParams["savefig.bbox"] = "tight"
    if save_dir:
        plt.rcParams["savefig.bbox"] = "tight"

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
    palette = utils.get_experience_level_colors()
    # palette = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    # palette = [(.2, .2, .2), (.2, .2, .2), (.2, .2, .2)]

    # define plot axes
    axes_column = 'experience_level'
    hue_column = 'experience_level'

    if ax is None:
        format_fig = True
        figsize = (8,6)
        fig, ax = plt.subplots(3, 3, figsize=figsize, sharey='row', sharex='col')
        ax = ax.ravel()
    else:
        format_fig = False

    for i, cell_type in enumerate(cell_types):
        df = multi_session_df[(multi_session_df.cell_type == cell_type)]
        if format_fig:
            ax[i * 3:(i * 3 + 3)] = plot_population_averages_for_conditions(df, data_type, event_type,
                                                                            axes_column, hue_column,
                                                                            horizontal=True, legend=False,
                                                                            xlim_seconds=xlim_seconds,
                                                                            interval_sec=interval_sec,
                                                                            palette=palette,
                                                                            ax=ax[i * 3:(i * 3 + 3)])
        else:
            ax[i] = plot_population_averages_for_conditions(df, data_type, event_type,
                                                            axes_column, hue_column, horizontal=True, legend=False,
                                                            xlim_seconds=xlim_seconds, interval_sec=interval_sec,
                                                            palette=palette, ax=ax[i])
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')

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

    ax[3].set_ylabel(ylabel)
    ax[7].set_xlabel(xlabel)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    if save_dir:
        fig_title = 'population_average_cell_types_exp_levels' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png'])

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


def annotate_epoch_df(epoch_df):
    """
    adds 'experience_epoch' column which is a conjunction of experience level and epoch #
    """

    # add experience epoch column
    def merge_experience_epoch(row):
        # epoch_num = str(int(row.epoch + 1))  # index at 1 not 0
        epoch_num = str(row.epoch)
        if len(epoch_num) == 1:
            epoch_num = '0' + str(epoch_num)
        return 'epoch ' + epoch_num + ' ' + row.experience_level

    epoch_df['experience_epoch'] = epoch_df[['experience_level', 'epoch']].apply(axis=1, func=merge_experience_epoch)

    return epoch_df


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
    # if 'experience_epoch' not in df.keys():
    df = annotate_epoch_df(df)

    cell_types = utils.get_cell_types()
    experience_levels = utils.get_new_experience_levels()

    df = df[df.epoch <= max_epoch]
    max_n_sessions = len(df.epoch.unique())

    experience_epoch = np.sort(df[df.experience_level==experience_levels[0]].experience_epoch.unique())
    experience_epoch = np.sort(df.experience_epoch.unique())
    experience_epoch = np.sort(df.epoch.unique())
    print(experience_epoch)
    # experience_epoch = ['Familiar epoch 1', 'Familiar epoch 2', 'Familiar epoch 3',
    #                      'Familiar epoch 4', 'Familiar epoch 5', 'Familiar epoch 6',
    #                      'Novel epoch 1', 'Novel epoch 2', 'Novel epoch 3',
    #                      'Novel epoch 4', 'Novel epoch 5', 'Novel epoch 6',
    #                      'Novel + epoch 1', 'Novel + epoch 2', 'Novel + epoch 3',
    #                      'Novel + epoch 4', 'Novel + epoch 5', 'Novel + epoch 6']

    # print(experience_epoch)
    xticks = np.arange(0, len(experience_epoch), 1)
    xticklabels = np.arange(0, len(experience_epoch), 1)+1
    # xticklabels = [experience_epoch.split(' ')[1] for experience_epoch in experience_epoch]

    palette = utils.get_experience_level_colors()
    if ax is None:
        format_fig = True
        if horizontal:
            figsize = (15, 3)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=False)
        else:
            figsize = (5, 8)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)
    else:
        format_fig = False

    for i, cell_type in enumerate(cell_types):
        try:
            print(cell_type)
            data = df[df.cell_type == cell_type]
            ax[i] = sns.pointplot(data=data, x='epoch', y=metric, hue='experience_level', hue_order=experience_levels,
                                  order=experience_epoch, palette=palette, ax=ax[i], estimator=estimator)

            if ymin is not None:
                ax[i].set_ylim(ymin=ymin)
            ax[i].set_title(cell_type)
            ax[i].set_ylabel(ylabel)
            ax[i].get_legend().remove()
            ax[i].set_xlim((xticks[0] - 1, xticks[-1] + 1))
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(xticklabels)
            ax[i].vlines(x=max_n_sessions + 0.5, ymin=0, ymax=1, color='gray', linestyle='--')
            ax[i].vlines(x=max_n_sessions + max_n_sessions + 1.5, ymin=0, ymax=1, color='gray', linestyle='--')
            if horizontal:
                ax[i].set_xlabel('epoch within session')
            else:
                ax[i].set_xlabel('')
        except Exception as e:
            print(e)

    ax[i].set_xlabel('epoch within session')
    ax[i].tick_params(axis='both', which='major', labelsize=14)

    if format_fig:
        if suptitle is not None:
            plt.suptitle(suptitle, x=0.52, y=1.01, fontsize=18)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
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


def get_fraction_matched_cells(matched_cells_table, platform_cells_table, conditions=['cell_type', 'ophys_container_id']):
    '''

    Parameters
    ----------
    matched_cells_table: ophys_cells_table limited to cells matched in all 3 sessions
    platform_cells_table: ophys_cells_table limited to containers included in platform dataset
    conditions: columns in cells tables to groupby before quantifying

    Returns
    -------
    table with fraction matched cells per condition
    '''
    total_cells = platform_cells_table.groupby(conditions).count()[['cell_specimen_id']].rename(columns={'cell_specimen_id': 'total_cells'})
    matched_cells = matched_cells_table.groupby(conditions).count()[['cell_specimen_id']].rename(columns={'cell_specimen_id': 'matched_cells'})
    fraction = total_cells.merge(matched_cells, on=conditions, how='left')  # need to use 'left' to prevent dropping of NaN values
    # set sessions with no responsive cells (NaN) to zero
    fraction['fraction_matched'] = fraction.matched_cells / fraction.total_cells
    return fraction


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
                                   ylabel='Fraction responsive', save_dir=None, folder=None, suffix='', ax=None):
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
            suffix = suffix + '_horiz'
            figsize = (9, 2)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False)
        else:
            figsize = (2, 9)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for i, cell_type in enumerate(cell_types):
        data = fraction_responsive[fraction_responsive.cell_type == cell_type]
        for ophys_container_id in data.ophys_container_id.unique():
            ax[i] = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x='experience_level', y='fraction_responsive',
                                  color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i])
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
        plt.setp(ax[i].lines, alpha=.3)
        ax[i] = sns.pointplot(data=data, x='experience_level', y='fraction_responsive', hue='experience_level',
                              hue_order=experience_levels, palette=palette, dodge=0, linewidth='none', ax=ax[i])
        ax[i].set_xticklabels(experience_levels, rotation=90)
        ax[i].set_ylabel('')
        ax[i].get_legend().remove()
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('')
        if ylim is None:
            ax[i].set_ylim(0, 1)
        else:
            ax[i].set_ylim(ylim)
    if horizontal:
        ax[0].set_ylabel(ylabel)
    else:
        ax[1].set_ylabel(ylabel)
    if save_dir:
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        fig_title = 'fraction_responsive_cells_' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)
    return ax


def plot_percent_responsive_cells(multi_session_df, responsiveness_threshold=0.1, horizontal=True, ylim=(0, 100), stats_max=80,
                                   ylabel='% responsive', save_dir=None, folder=None, suffix='', ax=None):
    """
    Plots the fraction of responsive cells across cre lines
    :param multi_session_df: dataframe of trial averaged responses for each cell for some set of conditions
    :param responsiveness_threshold: threshold on fraction_significant_p_value_gray_screen to determine whether a cell is responsive or not
    :param horizontal: Bool, whether to plot axes horizontally or vertically
    :param ylim: ylims of plot
    :param stats_max: value at which to plot statistics labels
    :param ylabel: string label for y axis of plot
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

    metric = 'percent_responsive'
    fraction_responsive[metric] = fraction_responsive['fraction_responsive'] * 100

    palette = utils.get_experience_level_colors()
    if ax is None:
        if horizontal:
            suffix = suffix + '_horiz'
            figsize = (6, 2.5)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=True)
        else:
            figsize = (1.5, 6.5)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)


    tukey = pd.DataFrame()
    for i, cell_type in enumerate(cell_types):
        data = fraction_responsive[fraction_responsive.cell_type == cell_type]
        # data[metric] = data['fraction_responsive']*100.
        print(cell_type, 'includes', len(data.ophys_container_id.unique()), 'containers')
        for ophys_container_id in data.ophys_container_id.unique():
            ax[i] = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x='experience_level', y=metric,
                                  color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i])
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
        plt.setp(ax[i].lines, alpha=.3)
        ax[i] = sns.pointplot(data=data, x='experience_level', y=metric, hue='experience_level',
                              hue_order=experience_levels, palette=palette, dodge=0, markers='.',
                              err_kws={'linewidth': 2}, markersize=5, errorbar=('ci', 95), ax=ax[i])

        ax[i].set_xticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
        [t.set_color(x) for (x, t) in zip(palette, ax[i].xaxis.get_ticklabels())]
        # ax[i].set_xticklabels(experience_levels, rotation=90)
        ax[i].set_ylabel('')
        # ax[i].get_legend().remove()
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('')
        if ylim is None:
            ax[i].set_ylim(0, 100)
        else:
            ax[i].set_ylim(ylim)

        # add stats to plot if only looking at experience levels
        ax[i], tukey_table = add_stats_to_plot_yaxis(data, metric, ax[i], ymax=stats_max)
        # aggregate stats
        tukey_table['metric'] = metric
        tukey_table['cell_type'] = cell_type
        tukey = pd.concat([tukey, tukey_table])

        ax[i].set_xlim((-0.4, 2.4))

    if horizontal:
        ax[0].set_ylabel(ylabel)
    else:
        ax[1].set_ylabel(ylabel)


    if save_dir:
        fig.subplots_adjust(hspace=0.4, wspace=0.2)
        fig_title = 'percent_responsive_cells' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)
        # try:
        print('saving_stats')
        # save tukey
        tukey.to_csv(os.path.join(save_dir, folder, fig_title + '_tukey.csv'))
        # save descriptive stats
        cols_to_groupby = ['cell_type', 'experience_level']
        stats = get_descriptive_stats_for_metric(fraction_responsive, metric, cols_to_groupby)
        stats.to_csv(os.path.join(save_dir, folder, fig_title + '_values.csv'))
        # except BaseException:
        #     print('STATS DID NOT SAVE FOR', metric)
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
                                  color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i])
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
        plt.setp(ax[i].lines, alpha=.3)
        # plot the population average in color
        ax[i] = sns.pointplot(data=data, x='experience_level', y=metric, hue='experience_level',
                              hue_order=experience_levels, palette=palette, dodge=0, linewidth='none', ax=ax[i])
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

    # remove null values
    data = data[~data[metric].isnull()].copy()
    # get conditions to compare
    groups = data[column_to_compare].unique()
    # run anova across groups depending on how many conditions there are
    if len(groups) == 2:
        anova = stats.f_oneway(
            data[data[column_to_compare] == groups[0]][metric],
            data[data[column_to_compare] == groups[1]][metric])
    elif len(groups) == 3:
        anova = stats.f_oneway(
            data[data[column_to_compare] == groups[0]][metric],
            data[data[column_to_compare] == groups[1]][metric],
            data[data[column_to_compare] == groups[2]][metric])
    elif len(groups) == 4:
        anova = stats.f_oneway(
            data[data[column_to_compare] == groups[0]][metric],
            data[data[column_to_compare] == groups[1]][metric],
            data[data[column_to_compare] == groups[2]][metric],
            data[data[column_to_compare] == groups[3]][metric])
    else:
        print('test_significant_metric_average function is not set up for this number of groups to compare')
    # get group index mapper
    mapper = {}
    for i, group in enumerate(groups):
        mapper[str(group)] = i

    if len(data[column_to_compare].unique())>2:
        # create tukey table for multiple comparisons across all pairs that can be compared
        comp = mc.MultiComparison(data[metric], data[column_to_compare])
        post_hoc_res = comp.tukeyhsd()
        tukey_table = pd.read_html(post_hoc_res.summary().as_html(), header=0, index_col=0)[0]
        tukey_table = tukey_table.reset_index()
        tukey_table['x1'] = [mapper[str(x)] for x in tukey_table['group1']]
        tukey_table['x2'] = [mapper[str(x)] for x in tukey_table['group2']]
    elif len(data[column_to_compare].unique())==2:
        tukey_table = pd.DataFrame()
        tukey_table['group1'] = [groups[0]]
        tukey_table['group2'] = [groups[1]]
        tukey_table['x1'] = [0]
        tukey_table['x2'] = [1]

    tukey_table['one_way_anova_p_val'] = anova[1]
    return anova, tukey_table


def add_stats_to_plot_for_hues(data, metric, ax, ymax=None, xorder=None, x='experience_level', hue='layer'):
    """
    add stars to axis indicating statistics across hue values
    x-axis of plots must be experience_levels
    xorder must be a list of values of x in the order that they appear on the plot

    data: metrics dataframe, each row is one cell_specimen_id in a given ophys_experiment
    metric: column in data representing metric values of interest
    column_to_compare: column in data to do stats over (after splitting by x values), such as 'layer' or 'targeted_structure'
    """

    # formatting
    scale = 0.05
    fontsize = 15

    ytop = ax.get_ylim()[1]
    y = ytop
    yh = ytop# * (1 + scale)

    # do anova across experience levels or cell types followed by post-hoc tukey
    for loc, x_value in enumerate(xorder):
        test_data = data[data[x]==x_value]
        hues = data[hue].unique()
        anova, tukey = test_significant_metric_averages(test_data, metric, column_to_compare=hue)

        if anova.pvalue < 0.05: # if something is significant, add some significance bars
            for tindex, row in tukey.iterrows():
                if len(hues)>2: # if more than 2 values, use the multiple corrections result
                    if row.reject:
                        ax.text(loc, yh, '*', fontsize=fontsize, horizontalalignment='center',
                                verticalalignment='bottom', color='k')
                elif len(hues)==2: # if there are only two values, use the p-value from anova
                    if row.one_way_anova_p_val<0.05:
                        ax.text(loc, yh, '*', fontsize=fontsize, horizontalalignment='center',
                                verticalalignment='bottom', color='k')
    ax.set_ylim(ymax=ytop * (1 + (scale * 5))) # 3 works better for behavior plots

    return ax, tukey


def add_stats_to_plot(data, metric, ax, ymax=None, column_to_compare='experience_level',
                      show_ns=False, hue_only=False):
    """
    add stars to axis indicating across experience level statistics
    x-axis of plots must be experience_levels or cell_types

    data: metrics dataframe, each row is one cell_specimen_id in a given ophys_experiment
    metric: column in data representing metric values of interest
    column_to_compare: must be 'experience_level' or 'cell_type'
    hue_only: if the axis only has one x value and the data are differentaited only by hue,
                set this to True so that the stats are plotted in reasonable positions centered around x=0
                (otherwise depends on there being x values)
    show_ns: Bool, whether or not to label non-significant results on the plot
    """
    # do anova across experience levels or cell types followed by post-hoc tukey
    anova, tukey = test_significant_metric_averages(data, metric, column_to_compare)
    if hue_only: # makes things from -0.25 to 0.25
        tukey['x1'] = tukey['x1'] - 1
        tukey['x2'] = tukey['x2'] - 1
        tukey['x1'] = tukey['x1'] / 4
        tukey['x2'] = tukey['x2'] / 4
        dist = 0.25
    else: # x pos is 0, 1, 2
        dist = 1

    scale = 0.05#0.05 # 0.1
    fontsize = 15

    if ymax is None:
        ytop = ax.get_ylim()[1]
    else:
        ytop = ymax
    if ytop > 1:
        second_bar_scale = 12 #yaxis uses 6
    else:
        second_bar_scale = 10 #yaxis uses 4
    y1 = ytop * (1 + scale)
    y1h = ytop * (1 + scale * 2)
    y2 = ytop * (1 + (scale * second_bar_scale))
    y2h = ytop * (1 + (scale * (second_bar_scale+1)))


    top = [ytop]
    scale_factor = 3
    two_away = False
    for tindex, row in tukey.iterrows():
        if (anova.pvalue < 0.05) and (row.reject): # if something is significant, add some significance bars
            label = '*'
            color = 'k'
            alpha = 1
        else:
            if show_ns:
                label = 'ns'
                color = 'k'
                alpha = 1
            else:
                label = ''
                color = 'w'
                alpha = 0
        # if row.x2 - row.x1 > 1:
        if np.abs(row.x2 - row.x1) > dist: # if it is a comparison more than 2 x values away, put the significance bar higher
            y = y2
            yh = y2h
        else: # if they are only one apart, put the bar on the lower level
            y = y1
            yh = y1h
        if len(data[column_to_compare].unique())>2: # if more than 2 values, use the multiple corrections result
            if row.reject:
                # print(row.x1, row.x2, np.abs(row.x2 - row.x1) > dist)
                # original
                # ax.plot([row.x1, row.x1, row.x2, row.x2], [y, yh, yh, y], linestyle='-', color=color, alpha=alpha)
                # from add_stats_yaxis
                ax.plot([row.x1 + 0.1, row.x1 + 0.1, row.x2 - 0.1, row.x2 - 0.1], [y, y, y, y], linestyle='-',
                        color=color, alpha=alpha, clip_on=False)
                # original
                ax.text(np.mean([row.x1, row.x2]), yh, label, fontsize=fontsize, horizontalalignment='center',
                        verticalalignment='bottom')
                top.append(yh)
                if np.abs(row.x2 - row.x1) > dist: # if there are x vals more than one apart, make the y scale bigger to fit the sig bars
                    scale_factor = 6
                    # print(scale_factor)
            else:
                if show_ns:
                    ax.plot([row.x1 + 0.1, row.x1 + 0.1, row.x2 - 0.1, row.x2 - 0.1], [y, y, y, y], linestyle='-',
                            color=color, alpha=alpha, clip_on=False)
                    # ax.plot([row.x1, row.x1, row.x2, row.x2], [y, yh, yh, y], linestyle='-', color=color, alpha=alpha)
                    ax.text(np.mean([row.x1, row.x2]), yh*(1+scale), 'ns', fontsize=fontsize, horizontalalignment='center',
                            verticalalignment='bottom')
                    top.append(yh)
        elif len(data[column_to_compare].unique())==2: #  if there are only two values, use the p-value from anova
            if row.one_way_anova_p_val<0.05:
                # ax.plot([row.x1, row.x1, row.x2, row.x2], [y, yh, yh, y], linestyle='-', color=color, alpha=alpha)
                ax.plot([row.x1 + 0.1, row.x1 + 0.1, row.x2 - 0.1, row.x2 - 0.1], [y, y, y, y], linestyle='-',
                        color=color, alpha=alpha, clip_on=False)
                ax.text(np.mean([row.x1, row.x2]), yh, label, fontsize=fontsize, horizontalalignment='center',
                        verticalalignment='bottom')
                top.append(yh)
            else:
                if show_ns:
                    ax.plot([row.x1 + 0.1, row.x1 + 0.1, row.x2 - 0.1, row.x2 - 0.1], [y, y, y, y], linestyle='-',
                            color=color, alpha=alpha, clip_on=False)
                    # ax.plot([row.x1, row.x1, row.x2, row.x2], [y, yh, yh, y], linestyle='-', color=color, alpha=alpha)
                    ax.text(np.mean([row.x1, row.x2]), yh*(1+scale), 'ns', fontsize=fontsize, horizontalalignment='center',
                            verticalalignment='bottom')
                    top.append(yh)
    # ax.set_ylim(ymax=ytop * (1 + (scale * 7))) # 3 works better for non-behavior plots
    ax.set_ylim(ymax=np.amax(top) * (1 + scale*scale_factor))  # scale factor determined by number of sig points # 3 works better for behavior plots, 2 for regular

    return ax, tukey

def add_stats_to_plot_yaxis(data, metric, ax, ymax=None, column_to_compare='experience_level', hue_only=False):
    """
    add stars to axis indicating across experience level statistics
    y-axis of plots must be experience_levels or cell_types

    data: metrics dataframe, each row is one cell_specimen_id in a given ophys_experiment
    metric: column in data representing metric values of interest
    column_to_compare: must be 'experience_level' or 'cell_type'
    hue_only: if the axis only has one x value and the data are differentiated only by hue,
                set this to True so that the stats are plotted in reasonable positions centered around x=0
                (otherwise depends on there being x values)
    show_ns: Bool, whether or not to label non-significant results on the plot
    """
    # do anova across experience levels or cell types followed by post-hoc tukey
    anova, tukey = test_significant_metric_averages(data, metric, column_to_compare)
    if hue_only: # makes things from -0.25 to 0.25
        tukey['x1'] = tukey['x1'] - 1
        tukey['x2'] = tukey['x2'] - 1
        tukey['x1'] = tukey['x1'] / 4
        tukey['x2'] = tukey['x2'] / 4
        dist = 0.25
    else: # x pos is 0, 1, 2
        dist = 1

    scale = 0.03#0.05 # 0.1
    fontsize = 15

    if ymax is None:
        ytop = ax.get_ylim()[1]
    else:
        ytop = ymax
    if ytop > 1:
        second_bar_scale = 6
    else:
        second_bar_scale = 4
    y1 = ytop * (1 + scale)
    y1h = ytop * (1 + scale * 2)
    y2 = ytop * (1 + (scale * second_bar_scale))
    y2h = ytop * (1 + (scale * (second_bar_scale+1)))

    top = [ytop]
    for tindex, row in tukey.iterrows():
        if (anova.pvalue < 0.05) and (row.reject): # if something is significant, add some significance bars
            label = '*'
            color = 'k'
            alpha = 1
        if np.abs(row.x2 - row.x1) > dist: # if it is a comparison more than 2 x values away, put the significance bar higher
            y = y2
            yh = y2h
        else: # if they are only one apart, put the bar on the lower level
            y = y1
            yh = y1h
        if len(data[column_to_compare].unique())>2: # if more than 2 values, use the multiple corrections result
            if row.reject:
                ax.plot([row.x1+0.1, row.x1+0.1, row.x2-0.1, row.x2-0.1], [y, y, y, y], linestyle='-', color=color, alpha=alpha, clip_on=False)
                # ax.annotate('*', xy=(xh, np.mean([row.x1, row.x2])), xycoords=ax.get_xaxis_transform(), ha="center", va="bottom",
                #                fontsize=15, clip_on=False)
                ax.text(np.mean([row.x1+scale*3, row.x2+scale*3]), yh+scale, label, fontsize=fontsize, horizontalalignment='center',
                        verticalalignment='center', clip_on=False)
                top.append(yh)
        elif len(data[column_to_compare].unique())==2: #  if there are only two values, use the p-value from anova
            if row.one_way_anova_p_val<0.05:
                ax.plot([row.x1-0.1, row.x1-0.1, row.x2-0.1, row.x2-0.1], [y, y, y, y, ], linestyle='-', color=color, alpha=alpha, clip_on=False)
                ax.text(np.mean([row.x1+0.2, row.x2+0.2]), yh+0.04, label, fontsize=fontsize, horizontalalignment='center',
                        verticalalignment='bottom', clip_on=False)
                top.append(yh)

    return ax, tukey


def add_stats_to_plot_xaxis(data, metric, ax, xmax=None, column_to_compare='experience_level',
                      show_ns=False, hue_only=False):
    """
    add stars to axis indicating across experience level statistics
    x-axis of plots must be experience_levels or cell_types

    data: metrics dataframe, each row is one cell_specimen_id in a given ophys_experiment
    metric: column in data representing metric values of interest
    column_to_compare: must be 'experience_level' or 'cell_type'
    hue_only: if the axis only has one x value and the data are differentaited only by hue,
                set this to True so that the stats are plotted in reasonable positions centered around x=0
                (otherwise depends on there being x values)
    show_ns: Bool, whether or not to label non-significant results on the plot
    """
    # do anova across experience levels or cell types followed by post-hoc tukey
    anova, tukey = test_significant_metric_averages(data, metric, column_to_compare)
    if hue_only: # makes things from -0.25 to 0.25
        tukey['x1'] = tukey['x1'] - 1
        tukey['x2'] = tukey['x2'] - 1
        tukey['x1'] = tukey['x1'] / 4
        tukey['x2'] = tukey['x2'] / 4
        dist = 0.25
    else: # x pos is 0, 1, 2
        dist = 1

    scale = 0.03#0.05 # 0.1
    fontsize = 15

    if xmax is None:
        xtop = ax.get_xlim()[1]
    else:
        xtop = xmax
    if xtop > 1:
        second_bar_scale = 6
    else:
        second_bar_scale = 4
    x1 = xtop * (1 + scale)
    x1h = xtop * (1 + scale * 2)
    x2 = xtop * (1 + (scale * second_bar_scale))
    x2h = xtop * (1 + (scale * (second_bar_scale+1)))

    top = [xtop]
    for tindex, row in tukey.iterrows():
        if (anova.pvalue < 0.05) and (row.reject): # if something is significant, add some significance bars
            label = '*'
            color = 'k'
            alpha = 1
        if np.abs(row.x2 - row.x1) > dist: # if it is a comparison more than 2 x values away, put the significance bar higher
            x = x2
            xh = x2h
        else: # if they are only one apart, put the bar on the lower level
            x = x1
            xh = x1h
        if len(data[column_to_compare].unique())>2: # if more than 2 values, use the multiple corrections result
            if row.reject:
                ax.plot([x, x, x, x], [row.x1+0.1, row.x1+0.1, row.x2-0.1, row.x2-0.1], linestyle='-', color=color, alpha=alpha, clip_on=False)
                # ax.annotate('*', xy=(xh, np.mean([row.x1, row.x2])), xycoords=ax.get_xaxis_transform(), ha="center", va="bottom",
                #                fontsize=15, clip_on=False)
                ax.text(xh+scale, np.mean([row.x1+scale*3, row.x2+scale*3]), label, fontsize=fontsize, horizontalalignment='center',
                        verticalalignment='center', clip_on=False)
                top.append(xh)
        elif len(data[column_to_compare].unique())==2: #  if there are only two values, use the p-value from anova
            if row.one_way_anova_p_val<0.05:
                ax.plot([x, x, x, x], [row.x1-0.1, row.x1-0.1, row.x2-0.1, row.x2-0.1], linestyle='-', color=color, alpha=alpha, clip_on=False)
                ax.text(xh+0.04, np.mean([row.x1+0.2, row.x2+0.2]), label, fontsize=fontsize, horizontalalignment='center',
                        verticalalignment='bottom', clip_on=False)
                top.append(xh)

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


def plot_metric_distribution_by_experience_no_cell_type(metrics_table, metric, event_type, data_type, hue=None,
                                                        stripplot=False, pointplot=True, boxplot=False,
                                                        add_zero_line=False, show_ns=False, abbreviate_exp=True,
                                                        show_containers=False, show_mice=False, horiz=False,
                                                        title='', ylabel=None, ylims=None, save_dir=None, ax=None, suffix=''):
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
    abbreviate_exp: Boolean, if True, use single letter experience labels on x axis (F, N, N+) and color by experience
                            if False, print out full name of each experience label on x axis
    show_containers: Boolean, whether or not to plot each individual container's datapoints in gray, with a line joining those for the same FOV
                typically used for physio event types
    show_mice: Boolean, whether or not to plot each individual mouse's datapoints in gray, with a line joining those for the same mouse
                    typically used for behavior event types

    save_dir: directory to save to. if None, plot will not be saved
    ax: axes to plot figures on
    """
    data = metrics_table.copy()
    experience_levels = utils.get_new_experience_levels()

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
    palette = utils.get_experience_level_colors()

    if horiz:
        x = metric
        y = 'experience_level'
    else:
        y = metric
        x = 'experience_level'

    if ax is None:
        if horiz:
            suffix = suffix + '_horiz'
            figsize = (3, 2.25)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            figsize = (2, 3)
            fig, ax = plt.subplots(1, 1, figsize=figsize)

    # stats dataframe to save
    tukey = pd.DataFrame()
    if hue:
        if pointplot:
            ax = sns.pointplot(data=data, y=y, x=x, order=order, dodge=0.3, linewidth='none',
                               markers='.', markersize=5, err_kws={'linewidth': 2}, hue=hue, hue_order=hue_order, palette='gray', ax=ax)

        else:
            ax = sns.boxplot(data=data, y=y, x=x, order=order, cut=0,
                             width=0.4, hue=hue, hue_order=hue_order, palette='gray', ax=ax)
        ax.legend(fontsize='xx-small', title='')  # , loc=loc)  # bbox_to_anchor=(1,1))
            # TBD add area or depth comparison stats / stats across hue variable
    else:
        if show_containers:
            print('table includes', len(data.ophys_container_id.unique()), 'containers')
            for ophys_container_id in data.ophys_container_id.unique():
                ax = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x=x, y=y,
                                   color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax)
        if show_mice:
            print('table includes', len(data.mouse_id.unique()), 'mice')
            for mouse_id in data.mouse_id.unique():
                ax = sns.pointplot(data=data[data.mouse_id == mouse_id], x=x, y=y,
                                   color='gray', linewidth=0.5, markers='.', markersize=1, err_kws={'linewidth': 0.5}, ax=ax)
            plt.setp(ax.collections, alpha=.7)  # for the markers
            plt.setp(ax.lines, alpha=.7)

        if pointplot:
            # ax = sns.pointplot(data=data, x='experience_level', y=metric,
            #                    palette=colors, ax=ax)
            ax = sns.pointplot(data=data, x=x, y=y, hue='experience_level', order=order,
                                  hue_order=experience_levels, palette=palette, dodge=0, linewidth='none',
                                  markers='.', markersize=5, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax)

        else:
            ax = sns.boxplot(data=data, x=x, y=y, width=0.4, order=order,
                             palette=colors, ax=ax)
        if stripplot:
            # add strip plot
            ax = sns.stripplot(data=data, size=3, alpha=0.5, jitter=0.2, order=order,
                               x=x, y=y, color='gray', ax=ax)
        if boxplot:
            ax = sns.boxplot(data=data, x=x, y=y, width=0.4, order=order,
                             palette='dark:white', ax=ax)
            # format to have black lines and transparent box face
            plt.setp(ax.artists, edgecolor='k', facecolor=[0, 0, 0, 0])
            plt.setp(ax.lines, color='k')


        ax.get_legend().remove()
        ax.set_title(title)

        # if ylims and not horiz:
        #     ax.set_ylim(ylims)
        # elif ylims and horiz:
        #     ax.set_xlim(ylims)
        # ymin, ymax = ax.get_ylim()

        if horiz:
            ax.set_xlim(xmin=ymin)
            ax.set_ylim(-0.5, len(order) - 0.5)
            ax, tukey_table = add_stats_to_plot_yaxis(data, metric, ax, ymax=ymax, hue_only=False)
        else:
            ax.set_ylim(ymin=ymin)
            ax.set_xlim(-0.5, len(order) - 0.5)

            # add stats to plot if only looking at experience levels
            ax, tukey_table = add_stats_to_plot(data, metric, ax, ymax=ymax, show_ns=show_ns)
        # aggregate stats
        tukey_table['metric'] = metric
        tukey = pd.concat([tukey, tukey_table])

        # add line at y=0
        if add_zero_line:
            if horiz:
                ax.axvline(x=0, ymin=0, ymax=1, color='gray', linestyle='--')
            else:
                ax.axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
        ax.set_title('')

        if horiz:
            ax.set_ylabel('')
            if ylabel:
                ax.set_xlabel(ylabel)
            else:
                ax.set_xlabel(metric)
            if abbreviate_exp:
                ax.set_yticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
                utils.color_yaxis_labels_by_experience(ax)
            else:
                ax.set_xticks(ax.get_xticks().tolist())
                ax.set_xticklabels(experience_levels, rotation=90, )  # ha='right')
                utils.color_xaxis_labels_by_experience(ax)
            ax.invert_yaxis()
        else:
            ax.set_xlabel('')
            if ylabel:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel(metric)
            if abbreviate_exp:
                ax.set_xticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
                utils.color_xaxis_labels_by_experience(ax)
            else:
                ax.set_xticks(ax.get_xticks().tolist())
                ax.set_xticklabels(experience_levels, rotation=90,)  # ha='right')
                utils.color_xaxis_labels_by_experience(ax)

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
                                           plot_type='pointplot', legend=True, show_containers=False,
                                           add_zero_line=False, show_ns=False, ylabel=None, ylims=None, horiz=True,
                                           abbreviate_exp=True, suptitle=None, save_dir=None, ax=None, suffix=''):
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
    plot_type: type of seaborn plotting function to use, can be one of:
                ['pointplot', 'boxplot', 'violinplot', 'stipplot]
                if 'stripplot' is used, data points will be plotted in color along with a transparent boxplot
    legend: Bool, if True, legend will be plotted on 0th axis
    show_containers: Bool, if True, plot gray lines connecting containers across experience levels
    ylims: yaxis limits to use; if None, will use +/-1
    horiz: Boolean, whether to plot the figure panels stacked horizontally or vertically
    orient: 'h' or 'v', if 'h', and plot_type='violin', will plot violinplots rotated horizontally, and annotate according to orient_annot
    orient_annot: tuple pair of strings to annotate left and right sides of x-axis when orient='h'
    abbreviate_exp: Boolean, if True, use single letter experience labels on x axis (F, N, N+) and color by experience
                            if False, print out full name of each experience label on x axis
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
            hue_order = np.sort(metrics_table[hue].unique())
        suffix = '_' + hue + '_' + plot_type + suffix
    else:
        suffix = '_experience_level_' + plot_type + suffix
    if ('index' in metric) and (ylims is None):
        if plot_type == 'pointplot':
            ylims = [-0.5, 0.5]
        else:
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
        if horiz:
            figsize = (8, 2.25)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=False)
        else:
            figsize = (2, 10)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)
    # stats dataframe to save
    tukey = pd.DataFrame()
    cell_types = utils.get_cell_types()
    for i, cell_type in enumerate(cell_types):
        ct_data = data[data.cell_type == cell_type]

        if show_containers:
            for ophys_container_id in ct_data.ophys_container_id.unique():
                ax[i] = sns.pointplot(data=ct_data[ct_data.ophys_container_id == ophys_container_id], x='experience_level',
                                      y=metric, color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i])

        if hue:
            if plot_type == 'pointplot':
                dodge = 0.1 * float(len(ct_data[hue].unique()))
                ax[i] = sns.pointplot(data=ct_data, y=metric, x='experience_level', order=order, dodge=dodge, linewidth='none',
                                      markers='.', markersize=5, err_kws={'linewidth': 2}, hue=hue, hue_order=hue_order, palette='gray', ax=ax[i])
            elif plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, y=metric, x='experience_level', order=order, fliersize=0,
                                    width=0.4, hue=hue, hue_order=hue_order, palette='gray', ax=ax[i])
            elif plot_type == 'violinplot':
                if len(ct_data[hue].unique())==2:
                    split = True
                else:
                    split = False
                ax[i] = sns.violinplot(data=ct_data, y=metric, x='experience_level', order=order,
                                       hue=hue, hue_order=hue_order, palette='gray', cut=0, inner=None,
                                       split=split, fill=False, ax=ax[i])
                ax2 = ax[i].twinx()
                ax2 = sns.boxplot(data=ct_data, hue=hue, y=metric, x='experience_level',  order=order, palette='dark:white',
                                  hue_order=hue_order, width=0.3, boxprops=dict(alpha=0.8, zorder=2), whis=0, showfliers=False, ax=ax2)
                ax2.get_legend().remove()
                ax2.axis('off')
                ax2.set_ylim(ymin=ymin)
                if ylims is not None:
                    ax2.set_ylim(ylims)
                # for violin in ax[i].collections:
                #     violin.set_alpha(0.5)

            ax[i].set_xlabel('')
            ax[i].set_ylabel('')
            ax[i].get_legend().remove()
            ax[i].set_ylim(ymin=ymin)
            if ylims is not None:
                ax[i].set_ylim(ylims)

            ax[i], tukey_table = add_stats_to_plot_for_hues(ct_data, metric, ax[i],
                                                            xorder=order, x='experience_level', hue=hue)
            # ax[i], tukey_table = add_stats_to_plot(ct_data, metric, ax[i], ymax=ymax)
            # aggregate stats
            tukey_table['hue'] = hue
            tukey_table['metric'] = metric
            tukey_table['cell_type'] = cell_type
            tukey = pd.concat([tukey, tukey_table])
        else:
            if plot_type == 'pointplot':
                ax[i] = sns.pointplot(data=ct_data, x='experience_level', y=metric, palette=colors,
                                      markers='.', markersize=5, err_kws={'linewidth': 2}, ax=ax[i])
            elif plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, x='experience_level', y=metric, width=0.4,
                                    palette=colors, fliersize=0, ax=ax[i])
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, y=metric, x='experience_level', order=order,
                                       palette=colors,  cut=0, ax=ax[i])
                for violin in ax[i].collections:
                    violin.set_alpha(0.75)

            elif plot_type == 'stripplot':
                ax[i] = sns.boxplot(data=ct_data, x='experience_level', y=metric, width=0.4,
                                    palette='dark:white', ax=ax[i])
                # format to have black lines and transparent box face
                plt.setp(ax[i].artists, edgecolor='k', facecolor=[0, 0, 0, 0])
                plt.setp(ax[i].lines, color='k')
                # add strip plot
                if cell_type == 'Excitatory':
                    tmp_data = ct_data.reset_index().copy()
                    # get 25% of all data points
                    pct = 0.25
                    n_samples = float(len(tmp_data) * pct)
                    print(n_samples, 'is', pct * 100., '% of all', cell_type, 'cells')
                    idx = np.random.choice(tmp_data.index.values, n_samples)
                    # limit to this random subset
                    tmp_data = tmp_data.loc[idx]
                else:
                    tmp_data = ct_data.copy()
                ax[i] = sns.stripplot(data=tmp_data, size=1.5, alpha=0.5, jitter=0.2,
                                      x='experience_level', y=metric, palette=colors, ax=ax[i])
            else:
                print('incorrect plot_type provided')

            ax[i].set_ylim(ymin=ymin)
            if ylims is not None:
                ax[i].set_ylim(ylims)
            ax[i].set_xlim(-0.5, len(order) - 0.5)
            # add stats to plot if only looking at experience levels
            # add stats to plot for hues
            ax[i], tukey_table = add_stats_to_plot(ct_data, metric, ax[i], ymax=ymax, show_ns=show_ns)
            # aggregate stats
            tukey_table['metric'] = metric
            tukey_table['cell_type'] = cell_type
            tukey = pd.concat([tukey, tukey_table])
            # set labels
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')

        # add line at y=0
        if add_zero_line:
            ax[i].axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
        # ax[i].set_title(cell_type)
        ax[i].set_title(cell_type+'\n(n = '+str(len(ct_data.cell_specimen_id.unique()))+' cells)', fontsize=16)
        # ax[i].set_title('')
        ax[i].set_xlabel('')

        if abbreviate_exp:
            ax[i].set_xticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
            utils.color_xaxis_labels_by_experience(ax[i])
        else:
            ax[i].set_xticks(ax[i].get_xticks().tolist())
            ax[i].set_xticklabels(new_experience_levels, rotation=90,)  # ha='right')
            utils.color_xaxis_labels_by_experience(ax[i])

        if ylabel:
            if horiz:
                ax[i].set_ylabel('')
            else:
                ax[1].set_ylabel(ylabel)
        else:
            ax[i].set_ylabel(metric)

    if legend and hue:
        ax[i].legend(fontsize='xx-small', title='', bbox_to_anchor=(1.6, 1))

    if horiz:
        ax[0].set_ylabel(ylabel)
        ax[0].set_xlabel('')
        ax[2].set_xlabel('')
    if suptitle:
        if horiz:
            plt.suptitle(suptitle, x=0.52, y=0.98, fontsize=18)
        else:
            plt.suptitle(suptitle, x=0.52, y=0.96, fontsize=18)
    fig.subplots_adjust(hspace=0.4, wspace=0.6)
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
                                            ylims=(0, 1), add_zero_line=True, remove_outliers=False, save_dir=None):
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
        If hue is None, will use experience level colors
    """

    if event_type is None:
        print('please provide event type for save file prefix')

    # full dataset, average over areas & depths
    plot_metric_distribution_by_experience(metrics_table, metric, plot_type='pointplot', legend=False,
                                           add_zero_line=add_zero_line, event_type=event_type, data_type=data_type,
                                           ylabel=ylabel, ylims=ylims, save_dir=save_dir, ax=None)

    # # per project code, average over areas & depths
    # for project_code in metrics_table.project_code.unique():
    #     df = metrics_table[metrics_table.project_code == project_code]
    #     if remove_outliers:
    #         df = df[df[metric]<np.percentile(df[metric].values, 99.7)]
    #     plot_metric_distribution_by_experience(df, metric, plot_type='pointplot', event_type=event_type, data_type=data_type,
    #                                            suffix=project_code, add_zero_line=add_zero_line,
    #                                            ylabel=ylabel, ylims=ylims, save_dir=save_dir, ax=None)

    # full dataset, for each area and depth
    if data_type in ['dff', 'events', 'filtered_events']:
        # if 'index' in metric: # use boxplot for indices that have a standardized range
        #     pointplot = False
        # else:
        #     pointplot = True # pointplots are better for thigns with wide ranges like mean responses

        # only look at VisualBehaviorMultiscope for area depth comparisons
        data = metrics_table[metrics_table.project_code == 'VisualBehaviorMultiscope']
        if 'response' in metric:
            if remove_outliers:
                data = data[data[metric]<np.percentile(data[metric], 99.7)]
            plot_types = ['pointplot']
        else:
            plot_types = ['violinplot', 'pointplot']

        # across areas
        for plot_type in plot_types:
            plot_metric_distribution_by_experience(data, metric, plot_type=plot_type, add_zero_line=add_zero_line,
                                                   event_type=event_type, data_type=data_type, hue='targeted_structure', legend=True,
                                                   ylabel=ylabel, ylims=ylims, save_dir=save_dir, ax=None)
            # across depths
            plot_metric_distribution_by_experience(data, metric, plot_type=plot_type, add_zero_line=add_zero_line,
                                                   event_type=event_type, data_type=data_type, hue='binned_depth', legend=True,
                                                   ylabel=ylabel, ylims=ylims, save_dir=save_dir, ax=None)

        # per project code, for each area and depth
        # for project_code in metrics_table.project_code.unique():
        #     df = metrics_table[metrics_table.project_code == project_code]
        #
        #     plot_metric_distribution_by_experience(df, metric, stripplot=False, pointplot=True, event_type=event_type, data_type=data_type,
        #                                            suffix=project_code, add_zero_line=add_zero_line,
        #                                            hue='targeted_structure', ylabel=ylabel, ylims=ylims,
        #                                            save_dir=save_dir, ax=None)
        #
        #     plot_metric_distribution_by_experience(df, metric, stripplot=False, pointplot=True, event_type=event_type, data_type=data_type,
        #                                            suffix=project_code, add_zero_line=add_zero_line,
        #                                            hue='layer', ylabel=ylabel, ylims=ylims, save_dir=save_dir, ax=None)


def plot_modulation_index_distribution(metrics_table, metric, xlabel=None, xlims=(-1.1, 1.1), horiz=False,
                                       annot=('left', 'right'), abbreviate_exp=True, suptitle=None,
                                       save_dir=None, suffix=''):
    colors = utils.get_experience_level_colors()
    experience_levels = utils.get_experience_levels()

    data = metrics_table.copy()

    ax = None
    if ax is None:
        if horiz:
            figsize = (6, 2.5)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=True)
        else:
            figsize = (2, 6)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)
        ax = ax.ravel()
    # stats dataframe to save
    tukey = pd.DataFrame()
    cell_types = utils.get_cell_types()
    for i, cell_type in enumerate(cell_types):
        ct_data = data[data.cell_type == cell_type]
        ax[i] = sns.violinplot(data=ct_data, x=metric, y='experience_level', order=experience_levels, orient='h',
                               palette=colors, alpha=0.5, cut=0, ax=ax[i])
        # set title
        ax[i].set_title(cell_type)
        # set x and y axis properties
        ax[i].set_ylabel('')
        ax[i].set_xlabel('')
        ax[i].set_xlim(xlims)
        if abbreviate_exp:
            ax[i].set_yticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
            utils.color_yaxis_labels_by_experience(ax[i])
        # set alpha on violinplots
        plt.setp(ax[i].collections, alpha=0.7)
        # line at zero value
        ax[i].axvline(x=0, ymin=0, ymax=1, color='gray', linestyle='--')

        # add stats to plot
        ax[i], tukey_table = add_stats_to_plot_xaxis(ct_data, metric, ax[i], xmax=xlims[1],
                                                         column_to_compare='experience_level')
        tukey_table['metric'] = metric
        tukey_table['cell_type'] = cell_type
        tukey = pd.concat([tukey, tukey_table])

    if horiz:
        ax[0].annotate(annot[0], xy=(-1.2, -0.05), xycoords=ax[0].get_xaxis_transform(), ha="right", va="center",
                       fontsize=10)
        ax[2].annotate(annot[1], xy=(1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="left", va="center",
                       fontsize=10)
        ax[1].set_xlabel(xlabel)
    else:
        ax[2].annotate('pre-change', xy=(-1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="right", va="center",
                       fontsize=10)
        ax[2].annotate('change', xy=(1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="left", va="center",
                       fontsize=10)
        ax[2].set_xlabel(xlabel)

    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=0.98, fontsize=16)

    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    if save_dir:
        if horiz:
            suffix = suffix + '_horiz'
        folder = 'metric_distributions'
        filename = metric + '_distribution' + suffix
        stats_filename = metric + suffix
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
            print('STATS DID NOT SAVE FOR', metric)
    return ax


def plot_experience_modulation_index(metric_data, event_type, hue=None, plot_type='pointplot', ylims=(-1, 1),
                                     suptitle=None, suffix='', include_all_comparisons=True, save_dir=None):
    """
    plots experience modulation for some event_type, which is the mean repsonse in familiar vs. novel 1 over the sum,
    and the mean response in novel 1 vs. novel >1 over the sum, giving a value between -1 and 1
    metric_data is the output of visual_behavior.ophys.response_analysis.cell_metrics.compute_experience_modulation_index()

    :param metric_data:
    :param event_type:
    :param save_dir:
    :return:
    """

    if include_all_comparisons:
        value_vars = ['Novel vs. Familiar', 'Novel + vs. Familiar',  'Novel vs. Novel +']
        data = metric_data[['cell_specimen_id', 'Novel vs. Familiar',  'Novel + vs. Familiar', 'Novel vs. Novel +',
                            'cell_type', 'targeted_structure', 'layer']]
        xorder = ['Novel vs. Familiar', 'Novel + vs. Familiar', 'Novel vs. Novel +']
        fig_width = 2.4
        suffix = suffix+'_all_comparisons'
    else:
        value_vars = ['Novel vs. Familiar', 'Novel + vs. Familiar', ]
        data = metric_data[['cell_specimen_id', 'Novel vs. Familiar', 'Novel + vs. Familiar',
                            'cell_type', 'targeted_structure', 'layer']]
        xorder = ['Novel vs. Familiar', 'Novel + vs. Familiar']
        fig_width = 1.8
        suffix = suffix

    if hue:
        data = data.melt(id_vars=['cell_specimen_id', 'cell_type', hue], var_name='comparison', value_vars=value_vars)
    else:
        data = data.melt(id_vars=['cell_specimen_id', 'cell_type'], var_name='comparison', value_vars=value_vars)

    metric = 'value'
    x = 'comparison'

    cell_types = np.sort(data.cell_type.unique())

    # colors = utils.get_experience_level_colors()
    figsize = (fig_width, 9)
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
    for i, cell_type in enumerate(cell_types):
        ct_data = data[data.cell_type == cell_type]
    #     ax[i] = sns.barplot(data=ct_data,  x=x, order=xorder, y=metric, dodge=0.5, ax=ax[i])
    #     change_width(ax[i], 0.3)
        if hue:
            hue_order = np.sort(ct_data[hue].unique())[::-1]
            suffix = suffix + '_' + hue
            if plot_type == 'pointplot':
                ylims = (-0.5, 0.5)
                dodge = 0.1 * float(len(ct_data[hue].unique()))
                ax[i] = sns.pointplot(data=ct_data, order=xorder, linewidth='none', hue=hue, hue_order=hue_order, dodge=dodge,
                                      x=x, y=metric, palette='gray', ax=ax[i])
            elif plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, order=xorder, hue=hue, hue_order=hue_order, width=0.5, boxprops=dict(alpha=0.8),
                                    x=x, y=metric, palette='gray', ax=ax[i])
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, order=xorder, cut=0, hue=hue, hue_order=hue_order, inner=None,
                                       x=x, y=metric, palette='gray', split=True, fill=False, ax=ax[i])
                ax2 = ax[i].twinx()
                ax2 = sns.boxplot(data=ct_data, order=xorder, hue=hue, hue_order=hue_order, x=x, y=metric, palette='dark:white',
                                    width=0.3, boxprops=dict(alpha=0.8, zorder=2), whis=0, showfliers=False, ax=ax2)
                ax2.get_legend().remove()
                ax2.axis('off')
                ax2.set_ylim(ylims)

            ax[i].get_legend().remove()
            ax[i].set_ylim(ylims)

            # add stats to plot for hues
            tukey = pd.DataFrame()
            ax[i], tukey_table = add_stats_to_plot_for_hues(ct_data, metric, ax[i],
                                                            xorder=xorder, x=x, hue=hue)
            # ax[i], tukey_table = add_stats_to_plot(ct_data, metric, ax[i], ymax=ymax)
            # aggregate stats
            tukey_table['hue'] = hue
            tukey_table['metric'] = metric
            tukey_table['cell_type'] = cell_type
            tukey = pd.concat([tukey, tukey_table])
        else:
            if plot_type == 'pointplot':
                ax[i] = sns.pointplot(data=ct_data, order=xorder, linewidth='none',
                                  x=x, y=metric, color='gray', ax=ax[i])
            elif plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, order=xorder, width=0.5,  boxprops=dict(alpha=0.8),
                                      x=x, y=metric, color='gray', ax=ax[i])
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, order=xorder, cut=0,
                                      x=x, y=metric, color='gray', ax=ax[i])
                for violin in ax[i].collections:
                    violin.set_alpha(0.5)
            ax[i].set_ylim(ylims)

        if not hue:
            ax[i].axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].set_xticklabels([x.split('.')[0] + '\n' + x.split('.')[1] for x in xorder], rotation=90, ha='center')

    if hue:
        ax[0].legend(fontsize='xx-small', title='', bbox_to_anchor=(1.4, 1))
    ax[1].set_ylabel('Experience modulation index')
    if xorder == ['Novel vs. Familiar', 'Novel + vs. Familiar']:
        ax[0].set_ylabel('<- F --- N ->', fontsize=12)
        ax[2].set_ylabel('<- F --- N ->', fontsize=12)

    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=0.98, fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if save_dir:
        filename = 'experience_modulation_' + event_type + '_' + plot_type + suffix
        utils.save_figure(fig, figsize, save_dir, 'metric_distributions', filename)


def plot_experience_modulation_index_annotated(metrics_table, event_type, metric, cells_table,
                                               horiz=False, xlims=(-1.1, 1.1), xlabel='Experience modulation',
                                               suptitle=None, suffix='', save_dir=None, ax=None):
    """
    plots experience modulation for some event_type, which is the mean repsonse in familiar vs. novel 1 over the sum,
    and the mean response in novel 1 vs. novel >1 over the sum, giving a value between -1 and 1
    metric_data is the output of visual_behavior.ophys.response_analysis.cell_metrics.compute_experience_modulation_index()

    :param metric_data:
    :param event_type:
    :param save_dir:
    :return:
    """

    import visual_behavior.ophys.response_analysis.cell_metrics as cm

    exp_mod = cm.compute_experience_modulation_index_new(metrics_table, metric, cells_table)
    exp_mod = exp_mod.drop_duplicates(subset='cell_specimen_id')

    value_vars = ['F N', 'N+ N', 'F N+']
    data = exp_mod.melt(id_vars=['cell_specimen_id', 'cell_type'], var_name='comparison', value_vars=value_vars)
    titles = ['Novel vs. Familiar', 'Novel vs. Novel +', 'Novel + vs. Familiar']

    # if hue:
    #     data = data.melt(id_vars=['cell_specimen_id', 'cell_type', hue], var_name='comparison', value_vars=value_vars)
    # else:
    #     data = data.melt(id_vars=['cell_specimen_id', 'cell_type'], var_name='comparison', value_vars=value_vars)

    metric = 'value'
    x = 'comparison'

    cell_types = np.sort(data.cell_type.unique())

    colors = utils.get_experience_level_colors()
    if ax is None:
        if horiz:
            figsize = (9, 2)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=True)
        else:
            figsize = (2, 6)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)

    tukey = pd.DataFrame()
    for i, comparison in enumerate(value_vars):
        ax[i] = sns.violinplot(data=data[data.comparison == comparison], x='value', y='cell_type', order=cell_types,
                               color='gray', cut=0, inner='box', ax=ax[i],)
                                # inner_kws=dict(box_width=5, whis_width=1, color="gray",))
        for violin in ax[i].collections:
            violin.set_alpha(0.5)
        ax[i].set_xlim(xlims)

        ax[i].axvline(x=0, ymin=0, ymax=1, color='gray', linestyle='--')
        ax[i].set_title(titles[i])
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].set_yticklabels([cell_type[:3] for cell_type in cell_types])

        annot = comparison.split(' ')

        def get_color(c):
            exp = annot[c]
            if exp == 'F':
                color = colors[0]
            elif exp == 'N':
                color = colors[1]
            elif exp == 'N+':
                color = colors[2]
            return color

        c = 0
        ax[i].annotate(annot[c], xy=(-1.2, -0.1), xycoords=ax[i].get_xaxis_transform(), ha="right", va="center",
                       color=get_color(c))
        c = 1
        ax[i].annotate(annot[c], xy=(1.2, -0.1), xycoords=ax[i].get_xaxis_transform(), ha="left", va="center",
                       color=get_color(c))
        # ax[i].set_xticklabels([x.split('.')[0] + '\n' + x.split('.')[1] for x in xorder], rotation=90, ha='center')

        # add stats to plot
        ax[i], tukey_table = add_stats_to_plot_xaxis(data[data.comparison == comparison], 'value', ax[i],
                                                         xmax=xlims[1],
                                                         column_to_compare='cell_type')
        tukey_table['metric'] = metric
        # tukey_table['cell_type'] = cell_type
        tukey = pd.concat([tukey, tukey_table])

    if horiz:

        ax[1].set_xlabel(xlabel)
    else:
        # ax[2].annotate(annot[0], xy=(-1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="right", va="center", fontsize=10)
        # ax[2].annotate(annot[1], xy=(1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="left", va="center", fontsize=10)
        ax[2].set_xlabel(xlabel)

    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=0.98, fontsize=16)

    fig.subplots_adjust(hspace=0.5, wspace=0.6)

    if save_dir:
        if horiz:
            suffix = suffix + '_horiz'
        folder = 'metric_distributions'
        filename = 'experience_modulation_annot_' + event_type + '_' + suffix
        stats_filename = 'experience_modulation_' + event_type + '_' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, filename)
        try:
            print('saving_stats')
            # save tukey
            tukey.to_csv(os.path.join(save_dir, folder, stats_filename + '_tukey.csv'))
            # save descriptive stats
            cols_to_groupby = ['cell_type']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, stats_filename + '_values.csv'))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric)
    return ax


def plot_experience_modulation_index_annotated_by_cell_type(metrics_table, event_type, metric, cells_table,
                                                            xlims=(-1.1, 1.1), xlabel='Experience modulation',
                                                            all_comparisons=True, horiz=False,
                                                            suptitle=None, suffix='', save_dir=None, ax=None):
    """
    plots experience modulation for some event_type, which is the mean repsonse in familiar vs. novel 1 over the sum,
    and the mean response in novel 1 vs. novel >1 over the sum, giving a value between -1 and 1
    metric_data is the output of visual_behavior.ophys.response_analysis.cell_metrics.compute_experience_modulation_index()

    :param metric_data:
    :param event_type:
    :param save_dir:
    :return:
    """

    import visual_behavior.ophys.response_analysis.cell_metrics as cm

    exp_mod = cm.compute_experience_modulation_index_new(metrics_table, metric, cells_table)
    exp_mod = exp_mod.drop_duplicates(subset='cell_specimen_id')

    if all_comparisons:
        value_vars = ['F N', 'N+ N', 'F N+'][::-1]
        titles = ['Novel vs. Familiar', 'Novel vs. Novel +', 'Novel + vs. Familiar']
    else:
        value_vars = ['F N', 'N+ N', ][::-1]
        titles = ['Novel vs. Familiar', 'Novel vs. Novel +', ]
    data = exp_mod.melt(id_vars=['cell_specimen_id', 'cell_type'], var_name='comparison', value_vars=value_vars)

    metric = 'value'
    x = 'comparison'

    cell_types = np.sort(data.cell_type.unique())
    colors = utils.get_experience_level_colors()

    wspace = 0.5
    if ax is None:
        if horiz:
            if 'all_comparisons':
                figsize = (9, 2)
                wspace = 0.6
            else:
                figsize = (9, 2)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=True)
        else:
            figsize = (2, 6)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)

    tukey = pd.DataFrame()
    for i, cell_type in enumerate(cell_types):
        ax[i] = sns.violinplot(data=data[data.cell_type == cell_type],
                               x='value', y='comparison', order=value_vars,
                               color='gray', cut=0, ax=ax[i]) #inner_kws={'marker':'.', 'markersize':5}, ax=ax[i])
        for violin in ax[i].collections:
            violin.set_alpha(0.5)
        ax[i].set_xlim(xlims)

        ax[i].axvline(x=0, ymin=0, ymax=1, color='gray', linestyle='--')
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].set_yticklabels('')

        for x, comparison in enumerate(value_vars):
            annot = comparison.split(' ')

            def get_color(c):
                exp = annot[c]
                if exp == 'F':
                    color = colors[0]
                elif exp == 'N':
                    color = colors[1]
                elif exp == 'N+':
                    color = colors[2]
                return color

            if all_comparisons:
                xy_0 = (-1.25, x / 3 + 0.15)
                xy_1 = (1.4, x / 3 + 0.15)
            else:
                xy_0 = (-1.25, x / 2 + 0.25)
                xy_1 = (1.18, x / 2 + 0.25)
            c = 0
            ax[i].annotate(annot[c], xy=xy_0, xycoords=ax[i].get_xaxis_transform(), ha="right", va="center",
                           color=get_color(c))
            c = 1
            ax[i].annotate(annot[c], xy=xy_1, xycoords=ax[i].get_xaxis_transform(), ha="left", va="center",
                           color=get_color(c))
            # ax[i].set_xticklabels([x.split('.')[0] + '\n' + x.split('.')[1] for x in xorder], rotation=90, ha='center')

        if len(value_vars) > 2:
            # add stats to plot
            ax[i], tukey_table = add_stats_to_plot_xaxis(data[data.cell_type == cell_type], 'value', ax[i],
                                                             xmax=xlims[1],
                                                             column_to_compare='comparison')
            tukey_table['metric'] = metric
            tukey_table['cell_type'] = cell_type
            tukey = pd.concat([tukey, tukey_table])

        ax[i].invert_yaxis()

    if horiz:
        ax[1].set_xlabel(xlabel)
    else:
        # ax[2].annotate(annot[0], xy=(-1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="right", va="center", fontsize=10)
        # ax[2].annotate(annot[1], xy=(1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="left", va="center", fontsize=10)
        ax[2].set_xlabel(xlabel)

    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=0.98, fontsize=16)

    fig.subplots_adjust(hspace=0.5, wspace=wspace)

    if save_dir:
        if horiz:
            suffix = suffix + '_horiz'
        folder = 'metric_distributions'
        filename = 'experience_modulation_annot_by_cell_type_' + event_type + '_' + suffix
        stats_filename = 'experience_modulation_by_cell_type_' + event_type + '_' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, filename)
        try:
            print('saving_stats')
            # save tukey
            tukey.to_csv(os.path.join(save_dir, folder, stats_filename + '_tukey.csv'))
            # save descriptive stats
            cols_to_groupby = ['cell_type']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, stats_filename + '_values.csv'))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric)
    return ax


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
                               microscope='Multiscope', cbar=True, cbar_label='Response', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.heatmap(data, cmap='binary', linewidths=0, square=False,
                     vmin=0, vmax=vmax, robust=True, cbar=cbar,
                     cbar_kws={"drawedges": False, "shrink": 0.7, "label": cbar_label}, ax=ax)

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

    if cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        ax.figure.axes[-1].yaxis.label.set_size(14)


    return ax


def plot_response_heatmaps_for_conditions(multi_session_df, timestamps, data_type, event_type,
                                          row_condition, col_condition, cols_to_sort_by=None, cell_order=None, suptitle=None,
                                          microscope=None, vmax=0.05, xlim_seconds=None, xlabel='time (s)',
                                          match_cells=False, cbar=True, cbar_label='Avg. calcium events',
                                          save_dir=None, folder=None, suffix='', ax=None):
    sdf = multi_session_df.copy()

    if xlim_seconds is None:
        xlim_seconds = (timestamps[0], timestamps[-1])

    row_conditions = np.sort(sdf[row_condition].unique())
    col_conditions = np.sort(sdf[col_condition].unique())

    if ax is None:
        figsize = (2.5 * len(col_conditions), 3 * len(row_conditions))
        fig, ax = plt.subplots(len(row_conditions), len(col_conditions), figsize=figsize, sharex=True)
        ax = ax.ravel()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

    i = 0
    for r, row in enumerate(row_conditions):
        row_sdf = sdf[(sdf[row_condition] == row)]
        for c, col in enumerate(col_conditions):

            if row == 'Excitatory':
                vmax = 0.005
            elif row == 'Vip Inhibitory':
                vmax = 0.01
            elif row == 'Sst Inhibitory':
                vmax = 0.015
            else:
                vmax = 0.02

            tmp = row_sdf[(row_sdf[col_condition] == col)]
            if cols_to_sort_by:
                tmp = tmp.sort_values(by=cols_to_sort_by, ascending=True)
            else:
                if match_cells:

                    cell_order = row_sdf[(row_sdf.experience_level=='Novel') &
                                         (row_sdf.cell_specimen_id.isin(tmp.cell_specimen_id.unique()))].sort_values(by=['cell_type', 'mean_response']).cell_specimen_id.values
                    if cell_order is not None:
                        tmp = tmp.set_index('cell_specimen_id')
                        tmp = tmp.loc[cell_order]
                    else:
                        if c == 0:
                            tmp = tmp.sort_values(by=['cell_type', 'mean_response'], ascending=True)
                            order = tmp.index.values
                        else:
                            tmp = tmp.loc[order]
                else:
                    tmp = tmp.sort_values(by='mean_response', ascending=True)
            data = pd.DataFrame(np.vstack(tmp.mean_trace.values), columns=timestamps)
            n_cells = len(data)

            ax[i] = plot_cell_response_heatmap(data, timestamps, vmax=vmax, xlabel=xlabel, cbar=cbar,
                                               microscope=microscope, cbar_label=cbar_label, ax=ax[i])
            ax[i].set_title(str(row) + '\n' + str(col))
            if col_condition == 'experience_level':
                colors = utils.get_experience_level_colors()
                if r == 0:
                    ax[i].set_title(col, color=colors[c])
                else:
                    ax[i].set_title('')
            # label y with total number of cells
            ax[i].set_yticks([0, n_cells])
            ax[i].set_yticklabels([0, n_cells], fontsize=14)
            if np.sum(np.abs(xlim_seconds)) < 2:
                # set xticks to every 1 second, assuming 30Hz traces
                ax[i].set_xticks(np.arange(0, len(timestamps), 15))  # assuming 30Hz traces
                ax[i].set_xticklabels([t for t in timestamps[::15]], rotation=0, fontsize=14)
            else:
                # set xticks to every 1 second, assuming 30Hz traces
                ax[i].set_xticks(np.arange(0, len(timestamps), 30))  # assuming 30Hz traces
                ax[i].set_xticklabels([t for t in timestamps[::30]], rotation=0, fontsize=14)
            # set xlims according to input
            start_index = np.where(timestamps == xlim_seconds[0])[0][0]
            end_index = np.where(timestamps <= xlim_seconds[1])[0][-1]
            xlims = [start_index, end_index]
            ax[i].set_xlim(xlims)
            ax[i].set_ylabel('')

            if (r == len(row_conditions) - 1) and (c == 1):
                ax[i].set_xlabel(xlabel)
            else:
                ax[i].set_xlabel('')
            sns.despine(ax=ax[i], top=False, right=False, left=False, bottom=False, offset=None, trim=False)
            i += 1

    for c, i in enumerate(np.arange(0, (len(col_conditions) * len(row_conditions)), len(col_conditions))):
        ax[i].set_ylabel(str(row_conditions[c])+'\ncells')

    # if suptitle:
    #     plt.suptitle(suptitle, x=0.52, y=1.0, fontsize=18)

    if save_dir:
        fig_title = event_type + '_response_heatmap_' + data_type + '_' + col_condition + '_' + row_condition + suffix
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


def add_stim_color_span(dataset, ax, xlim=None, color=None, label_changes=True,
                        label_omissions=True, annotate_changes=False, max_alpha=0.5):
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
    stim_table = loading.limit_stimulus_presentations_to_change_detection(stim_table)
    # remove omissions because they dont get labeled
    #     stim_table = stim_table[stim_table.omitted == False].copy()
    # get all images & assign colors (image colors wont be used if a color is provided or if label_changes is True)
    images = np.sort(stim_table[stim_table.omitted == False].image_name.unique())
    image_colors = sns.color_palette("hls", len(images))
    # limit to time window if provided
    if xlim is not None:
        stim_table = stim_table[(stim_table.start_time >= xlim[0]) & (stim_table.end_time <= xlim[1])]
    # loop through stimulus presentations and add a span with appropriate color
    for idx in stim_table.index:
        start_time = stim_table.loc[idx]['start_time']
        end_time = stim_table.loc[idx]['end_time']
        image_name = stim_table.loc[idx]['image_name']
        image_index = stim_table.loc[idx]['image_index']
        if image_name == 'omitted':
            if label_omissions:
                ax.axvline(x=start_time, ymin=0, ymax=1, linestyle='--', color=sns.color_palette()[9])
        else:
            if label_changes:
                if stim_table.loc[idx]['is_change']:  # if its a change, make it blue with higher alpha
                    image_color = sns.color_palette()[0]
                    alpha = max_alpha
                    if annotate_changes:
                        ymin, ymax = ax.get_ylim()
                        ax.annotate(stim_table.loc[idx]['image_name'], xy=(start_time, ymax*1.2), xycoords='data',
                                    fontsize=8,  va='top', clip_on=False, annotation_clip=False)
                        # also show the one before
                        ax.annotate(stim_table.loc[idx-1]['image_name'], xy=(start_time-1.5, ymax * 1.2), xycoords='data',
                                    fontsize=8, va='top', clip_on=False, annotation_clip=False)
                else:  # if its a non-change make it gray with low alpha
                    image_color = 'gray'
                    alpha = max_alpha/2.
            else:
                if color is None:
                    image_color = image_colors[image_index]
                else:
                    image_color = color
            addSpan(ax, start_time, end_time, color=image_color, alpha=alpha)
    return ax


def plot_time_in_minutes(timestamps, ax, interval_duration=5):
    '''
    Takes timestamps, in seconds, convert to minutes, and set xticklabels to show time in the provided interval duration in minutes
    '''
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_xticks(np.arange(0, timestamps[-1], 60 * float(interval_duration)))
    ax.set_xticklabels([int(t) for t in np.arange(0, timestamps[-1] / 60, interval_duration)])  # tick every x mins
    ax.set_xlabel('Time in session (minutes)')

    return ax

def add_stimulus_blocks(stim_table, xlim=None, annotate_blocks=True, ax=None):
    '''
    Function to plot shaded bar across x axis representing each stimulus block within the period of xlim
    Expecting blocks to be either change detection, gray screen, or natural movie one
    if xlim is None, will plot entire session
    if ax is provided, will plot on that axis, otherwise generates a figure to plot
    '''

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 1))

    if xlim is None:
        xlim = [stim_table.start_time.values[0], stim_table.end_time.values[-1]]

    block_change_inds = np.where(stim_table.stimulus_block.diff())
    stimulus_blocks = stim_table.loc[block_change_inds]

    block_change_inds = np.where(stim_table.stimulus_block.diff())
    stimulus_blocks = stim_table.loc[block_change_inds]
    last_block = stimulus_blocks.stimulus_block.max()

    # loop through stimulus blocks and add a span with appropriate color
    for idx in stimulus_blocks.index:
        stimulus_block = stimulus_blocks.loc[idx]['stimulus_block']
        block_name = stimulus_blocks.loc[idx]['stimulus_block_name']
        start_time = stimulus_blocks.loc[idx]['start_time']
        if stimulus_block != last_block:
            end_time = stimulus_blocks[stimulus_blocks.stimulus_block == stimulus_block + 1]['start_time'].values[0]
        else:  # if its the last block, use the very last timestamp as the end
            end_time = stim_table.end_time.values[-1]  # stimulus_blocks.loc[idx]['end_time']

        if 'gray_screen' in block_name:
            color = 'gray'
            name = 'gray\nscreen'
        elif 'change_detection' in block_name:
            color = sns.color_palette()[0]
            name = 'change detection task'
        elif 'movie' in block_name:
            color = sns.color_palette()[9]
            name = 'movie\nclips'

        if annotate_blocks:
            ax.text(s=name, x=start_time + (end_time - start_time) / 2, y=0.5, va='center', ha='center')

        addSpan(ax, start_time, end_time, color=color, alpha=0.5)

    ax.set_yticklabels([])
    ax.set_xlim(xlim)
    ax = plot_time_in_minutes(xlim, ax)

    sns.despine(ax=ax, top=True, right=True, left=True, )
    ax.tick_params(which='both', bottom=True, top=False, right=False, left=False)

    return ax




def plot_behavior_timeseries(dataset, start_time, duration_seconds=20, xlim_seconds=None, save_dir=None, ax=None):
    """
    Plots licking behavior, rewards, running speed, and pupil area for a defined window of time
    """
    if xlim_seconds is None:
        xlim_seconds = [start_time - (duration_seconds / 4.), start_time + float(duration_seconds) * 2]
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

    ln0 = ax.plot(lick_timestamps, licks, '|', label='licks', color=colors[3], markersize=50)
    ln1 = ax.plot(reward_timestamps, rewards, 'o', label='rewards', color=colors[9], markersize=50)

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

    ax = add_stim_color_span(dataset, ax, xlim=xlim_seconds, annotate_changes=True,
                             label_changes=True, label_omissions=True)

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
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)))
    return ax


def plot_behavior_timeseries_stacked(dataset, start_time, fontsize=12,
                                     duration_seconds=20, xlim_seconds=None,
                                     label_changes=True, label_omissions=True,
                                     show_images=True, annotate_yaxis=True,
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

    if xlim_seconds == None:
        # xlim_seconds = [start_time - (duration_seconds / 4.), start_time + duration_seconds * 2]
        xlim_seconds = [start_time, start_time+duration_seconds]

    lick_timestamps = dataset.licks.timestamps.values
    lick_timestamps = lick_timestamps[lick_timestamps > xlim_seconds[0]]
    lick_timestamps = lick_timestamps[lick_timestamps < xlim_seconds[1]]
    licks = np.ones(len(lick_timestamps))
    # licks[:] = -2

    reward_timestamps = dataset.rewards.timestamps.values
    reward_timestamps = reward_timestamps[reward_timestamps > xlim_seconds[0]]
    reward_timestamps = reward_timestamps[reward_timestamps < xlim_seconds[1]]
    rewards = np.ones(len(reward_timestamps))
    # rewards[:] = -4

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
        figsize = (5, 2)
        fig, ax = plt.subplots(4, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [1, 1, 4, 4]})
        ax = ax.ravel()

    colors = sns.color_palette()

    ax[0].plot(lick_timestamps, licks, '|', label='licks', color='gray', markersize=5) #colors[3], markersize=50)
    ax[0].set_yticklabels([])
    ax[0].set_ylabel('licks', rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

    ax[1].plot(reward_timestamps, rewards, '^', label='rewards', color='b', markersize=5) #color=colors[8], markersize=50)
    ax[1].set_yticklabels([])
    ax[1].set_ylabel('rewards', rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

    ax[2].plot(running_timestamps, running_speed, label='running_speed', color='gray')  #color=colors[2])
    ax[2].set_ylabel('running\nspeed\n(cm/s)', rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
    ax[2].set_ylim(ymin=-8)

    ax[3].plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color='gray') # color=colors[4])
    ax[3].set_ylabel('pupil\ndiameter\n(pixels)', rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=fontsize)


    for i in range(4):
        # if (i == 0) & (show_images == True):
        #     ax[i] = add_stim_color_span(dataset, ax[i], xlim=xlim_seconds, annotate_changes=True,
        #                                 label_changes=label_changes, label_omissions=label_omissions)
        # elif label_changes==True | label_omissions==True:
        #     ax[i] = add_stim_color_span(dataset, ax[i], xlim=xlim_seconds, annotate_changes=False,
        #                                 label_changes=label_changes, label_omissions=label_omissions)
        # else:
        #     pass
        ax[i].set_xlim(xlim_seconds)
        if i in [0, 1]: # for licks and rewards
            ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                              labelbottom=False, labeltop=False, labelright=False, labelleft=True, )
        else: # for running and pupil
            if annotate_yaxis: # plot lines for data range instead of ticks
                ymin, ymax = ax[i].get_ylim()
                if i == 2:
                    ymin = 0
                    ymax = int(ymax / 2)
                else:
                    diff = (ymax-ymin)*0.25
                    ymin = int(ymin+diff)
                    ymax = int(ymax-diff)
                ax[i].set_yticks([ymin, ymax])
                ax[i].set_yticklabels([ymin, ymax], va='center', ha='right', fontsize=fontsize - 2)
                ax[i].annotate('', xy=(xlim_seconds[0] - 0.3, ymin), xycoords='data', xytext=(xlim_seconds[0] - 0.3, ymax),
                               fontsize=fontsize, arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0),
                               annotation_clip=False)
                ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                                  labelbottom=False, labeltop=False, labelright=False, labelleft=True, )
        ax[i].spines[['right', 'top']].set_visible(False)
        # sns.despine(ax=ax[i], bottom=True)
    # sns.despine(ax=ax[i], bottom=False)
    ax[i].tick_params(which='both', bottom=True, top=False, right=False, left=False,
                      labelbottom=True, labeltop=False, labelright=False, labelleft=True, labelsize=fontsize-2)

    # label bottom row of plot
    ax[i].set_xlabel('Time in session (seconds)', fontsize=fontsize)

    # ax[i] = plot_time_in_minutes(xlim_seconds, ax[i])

    if save_dir:
        # add title to top row
        metadata_string = utils.get_metadata_string(dataset.metadata)
        plt.suptitle(metadata_string, x=0.5, y=1.1, fontsize=fontsize)

        plt.subplots_adjust(hspace=0)
        folder = 'behavior_timeseries_stacked'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)) + '_' + suffix)
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

    xlim_seconds = [start_time - (duration_seconds / 4.), start_time + float(duration_seconds) * 2]

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

    ax[8].plot(lick_timestamps, licks, '|', label='licks', color='gray', markersize=50)
    ax[8].set_yticklabels([])
    ax[8].set_ylabel('licks', rotation=0, horizontalalignment='right', verticalalignment='center')

    ax[9].plot(reward_timestamps, rewards, 'o', label='rewards', color='gray', markersize=50)
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
    ax[i].set_xlabel('Time in session (seconds)')
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
    First row is the ROI mask in the 3 sessions, second row is the average change response for each session in gray
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
            data_type = 'filtered_events'
        else:
            data_type = 'events'
        ylabel = 'response'
    else:
        data_type = 'dff'
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

                # analysis = ResponseAnalysis(dataset, use_events=use_events, filter_events=filter_events,
                #                             use_extended_stimulus_presentations=False)
                # sdf = analysis.get_response_df(df_name='stimulus_response_df')
                window = [-1, 0.75]
                sdf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True,
                                                       output_sampling_rate=30,
                                                       data_type=data_type, event_type='changes',
                                                       load_from_file=True)
                cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.is_change == True)]

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
        metadata_string = utils.get_metadata_string(dataset.metadata)
        utils.save_figure(fig, figsize, save_dir, folder, str(cell_specimen_id) + '_' + metadata_string + '_' + data_type)
        plt.close()


def plot_matched_roi_and_traces_example(cell_metadata, include_omissions=True,
                                        use_events=False, filter_events=False, save_dir=None, folder=None):
    """
    Plots the ROI masks and cell traces for a cell matched across sessions in a single row
    First 3 panels are ROIs, then change response across sessions, then omission response across sessions if include_omission=True
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


########## behavior plots - figure 1 #############

def plot_behavior_metric_by_experience(stats, metric, title='', ylabel='', ylims=None, best_image=True, show_containers=False,
                                       stripplot=True, pointplot=True, plot_stats=False, show_ns=False, save_dir=None, folder=None, suffix='', ax=None):
    """
    plots average metric value across experience levels, using experience level colors for average, gray for individual points.
    plots a stripplot of all datapoints and a pointplot of means by default. if pointplot is False, a boxplot will be shown.

    stats should be a table of behavior metric values loaded using vba.utilities.get_behavior_stats_for_sessions()
    metric is a column of the stats table

    if stats table has a unique row for each image_name in each behavior session, all images will be included in the average,
    unless best_image = True
    if stats table does not have unique images, setting best_image to True will cause an error, as there are no images to filter

    if best_image = True, will sort images by metric value within each experience level and select the top 2 images to plot
    if show_containers = True, will plot a linked gray line for each individual container within the dataset
    if show_ns = True, indicates whether results are non-significant on plot. otherwise only puts asterisk for significant results
    returns axis handle
    """
    # experience_levels = utils.get_new_experience_levels()
    # new_experience_levels = utils.get_new_experience_levels()
    # colors = utils.get_experience_level_colors()

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
        best_novel = tmp.loc['Novel'].index.values[-2:]
        best_novel_plus = tmp.loc['Novel +'].index.values[-2:]

        # get data for images with highest metric value
        familiar_stats = stats[(stats.experience_level == 'Familiar') & (stats.image_name.isin(best_familiar))]
        novel_stats = stats[(stats.experience_level == 'Novel') & (stats.image_name.isin(best_novel))]
        novel_plus_stats = stats[(stats.experience_level == 'Novel +') & (stats.image_name.isin(best_novel_plus))]

        data = pd.concat([familiar_stats, novel_stats, novel_plus_stats])

        suffix = suffix + '_best_image'

    else:
        data = stats.copy()

    colors = utils.get_experience_level_colors()
    # experience_levels = utils.get_experience_levels()
    experience_levels = np.sort(data.experience_level.unique())

    if ax is None:
        figsize = (2, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if stripplot:
        ax = sns.stripplot(data=data, x='experience_level', y=metric, order=experience_levels,
                       orient='v', color='gray', dodge=True, jitter=0.1, size=2, ax=ax, zorder=0)
    if show_containers:
        for ophys_container_id in data.ophys_container_id.unique():
            ax = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x='experience_level', y=metric,
                               order=experience_levels, linewidth=0.5, orient='v', color='gray',
                               markers='.', markersize=0.15, err_kws={'linewidth': 0.5}, ax=ax)
        suffix = suffix + '_containers'

    if pointplot:
        ax = sns.pointplot(data=data, x='experience_level', y=metric, order=experience_levels,
                       orient='v', palette=colors, ax=ax)
    else:
        ax = sns.boxplot(data=data, x='experience_level', y=metric, order=experience_levels,
                           orient='v', palette=colors, width=0.6, boxprops=dict(alpha=0.8), ax=ax)

    ax.set_xlim(-0.5, len(experience_levels)-0.5)
    ax.set_xticklabels(experience_levels, rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)
    if ylabel is None:
        ylabel = metric
    ax.set_ylabel(ylabel)
    # ax.legend(bbox_to_anchor=(1,1), fontsize='xx-small')

    # add stats to plot if only looking at experience levels
    if plot_stats:
        # stats dataframe to save
        tukey = pd.DataFrame()
        ax, tukey_table = add_stats_to_plot(data, metric, ax, ymax=ymax, show_ns=show_ns)
        # aggregate stats
        tukey_table['metric'] = metric
        tukey = pd.concat([tukey, tukey_table])

    ax.set_ylim(ymin=ymin)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, metric + suffix)
    stats_filename = metric + '_stats' + suffix
    try:
        if plot_stats:
            print('saving_stats')
            tukey.to_csv(os.path.join(save_dir, folder, stats_filename + '_tukey.csv'))
        # save metric values
        cols_to_groupby = ['experience_level']
        stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        stats.to_csv(os.path.join(save_dir, folder, stats_filename + '_values.csv'))
    except BaseException:
        print('stats did not save for', metric)
    return ax


def plot_behavior_metric_by_experience_horiz(stats, metric, title='', xlabel='', xlims=None, best_image=True, show_containers=False,
                                       stripplot=True, pointplot=True, plot_stats=False, show_ns=False, save_dir=None, folder=None, suffix='', ax=None):
    """
    plots average metric value across experience levels, using experience level colors for average, gray for individual points.
    plots a stripplot of all datapoints and a pointplot of means by default. if pointplot is False, a boxplot will be shown.

    stats should be a table of behavior metric values loaded using vba.utilities.get_behavior_stats_for_sessions()
    metric is a column of the stats table

    if stats table has a unique row for each image_name in each behavior session, all images will be included in the average,
    unless best_image = True
    if stats table does not have unique images, setting best_image to True will cause an error, as there are no images to filter

    if best_image = True, will sort images by metric value within each experience level and select the top 2 images to plot
    if show_containers = True, will plot a linked gray line for each individual container within the dataset
    if show_ns = True, indicates whether results are non-significant on plot. otherwise only puts asterisk for significant results
    returns axis handle
    """
    # experience_levels = utils.get_new_experience_levels()
    # new_experience_levels = utils.get_new_experience_levels()
    # colors = utils.get_experience_level_colors()

    if xlims is None:
        xmin = 0
        xmax = None
    else:
        xmin = xlims[0]
        xmax = xlims[1]

    if best_image:
        tmp = stats.copy()
        tmp = tmp[tmp.image_name != 'omitted']

        # sort images by metric value within each experience level
        tmp = tmp.groupby(['experience_level', 'image_name']).mean()[[metric]].sort_values(by=['experience_level', metric])

        best_familiar = tmp.loc['Familiar'].index.values[-2:]
        best_novel = tmp.loc['Novel'].index.values[-2:]
        best_novel_plus = tmp.loc['Novel +'].index.values[-2:]

        # get data for images with highest metric value
        familiar_stats = stats[(stats.experience_level == 'Familiar') & (stats.image_name.isin(best_familiar))]
        novel_stats = stats[(stats.experience_level == 'Novel') & (stats.image_name.isin(best_novel))]
        novel_plus_stats = stats[(stats.experience_level == 'Novel +') & (stats.image_name.isin(best_novel_plus))]

        data = pd.concat([familiar_stats, novel_stats, novel_plus_stats])

        suffix = suffix + '_best_image'

    else:
        data = stats.copy()

    colors = utils.get_experience_level_colors()
    # experience_levels = utils.get_experience_levels()
    experience_levels = np.sort(data.experience_level.unique())

    if ax is None:
        figsize = (3.5, 2)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if stripplot:
        ax = sns.stripplot(data=data, y='experience_level', x=metric, order=experience_levels,
                       orient='h', color='gray', dodge=True, jitter=0.1, size=2, ax=ax, zorder=0)
    if show_containers:
        for ophys_container_id in data.ophys_container_id.unique():
            ax = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], y='experience_level', x=metric,
                               order=experience_levels, linewidth=0.5, orient='h', color='gray',
                               markers='.', markersize=0.15, err_kws={'linewidth': 0.5}, ax=ax)
        suffix = suffix + '_containers'

    if pointplot:
        ax = sns.pointplot(data=data, y='experience_level', x=metric, order=experience_levels,
                       orient='h', palette=colors, ax=ax)
    else:
        ax = sns.boxplot(data=data, y='experience_level', x=metric, order=experience_levels,
                           orient='h', palette=colors, width=0.6, boxprops=dict(alpha=0.8), ax=ax)

    # ax.set_ylim(-0.5, len(experience_levels)-0.5)
    # ax.set_yticklabels(experience_levels, rotation=0)
    # ax.invert_yaxis()
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(title)
    if xlabel is None:
        xlabel = metric
    ax.set_xlabel(xlabel)
    # ax.legend(bbox_to_anchor=(1,1), fontsize='xx-small')

    # add stats to plot if only looking at experience levels
    if plot_stats:
        # stats dataframe to save
        tukey = pd.DataFrame()
        ax, tukey_table = add_stats_to_plot(data, metric, ax, ymax=xmax, show_ns=show_ns)
        # aggregate stats
        tukey_table['metric'] = metric
        tukey = pd.concat([tukey, tukey_table])

    ax.set_xlim(xmin=xmin)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, metric + '_horiz' + suffix)
    stats_filename = metric + '_stats' + suffix
    try:
        if plot_stats:
            print('saving_stats')
            tukey.to_csv(os.path.join(save_dir, folder, stats_filename + '_tukey.csv'))
        # save metric values
        cols_to_groupby = ['experience_level']
        stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        stats.to_csv(os.path.join(save_dir, folder, stats_filename + '_values.csv'))
    except BaseException:
        print('stats did not save for', metric)
    return ax



def plot_behavior_metric_by_cohort(stats, metric, title='', ylabel='', ylims=None, show_containers=False,
                                       stripplot=True, pointplot=True, plot_stats=False, show_ns=False, save_dir=None, folder=None, suffix='', ax=None):
    """
    plots average metric value across project codes
    plots a stripplot of all datapoints and a pointplot of means by default. if pointplot is False, a boxplot will be shown.

    stats should be a table of behavior metric values loaded using vba.utilities.get_behavior_stats_for_sessions()
    metric is a column of the stats table

    if stats table has a unique row for each image_name in each behavior session, all images will be included in the average,
    unless best_image = True
    if stats table does not have unique images, setting best_image to True will cause an error, as there are no images to filter

    if show_containers = True, will plot a linked gray line for each individual container within the dataset
    if show_ns = True, indicates whether results are non-significant on plot. otherwise only puts asterisk for significant results
    returns axis handle
    """
    # experience_levels = utils.get_new_experience_levels()
    # new_experience_levels = utils.get_new_experience_levels()
    # colors = utils.get_experience_level_colors()

    if ylims is None:
        ymin = 0
        ymax = None
    else:
        ymin = ylims[0]
        ymax = ylims[1]

    data = stats.copy()

    # colors = utils.get_experience_level_colors()
    # experience_levels = utils.get_experience_levels()
    c = sns.color_palette()
    colors = [c[0], c[8], c[9]]
    # project_codes = np.sort(data.project_code.unique())
    project_codes = ['VisualBehavior', 'VisualBehaviorTask1B', 'VisualBehaviorMultiscope']
    cohorts = ['Cohort 1', 'Cohort 2', 'Cohort 3']

    if ax is None:
        figsize = (2, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if stripplot:
        ax = sns.stripplot(data=data, x='project_code', y=metric, order=project_codes,
                       orient='v', color='gray', dodge=True, jitter=0.1, size=2, ax=ax, zorder=0)
    if show_containers:
        for ophys_container_id in data.ophys_container_id.unique():
            ax = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x='project_code', y=metric,
                               order=project_codes, linewidth=0.5, orient='v', color='gray',
                               markers='.', markersize=0.15, err_kws={'linewidth': 0.5}, ax=ax)
        suffix = suffix + '_containers'

    if pointplot:
        ax = sns.pointplot(data=data, x='project_code', y=metric, order=project_codes,
                       orient='v', palette=colors, markers='.', ax=ax) # marker_kws={'size':2},
    else:
        ax = sns.boxplot(data=data, x='project_code', y=metric, order=project_codes,
                           orient='v', palette=colors, width=0.6, boxprops=dict(alpha=0.8), ax=ax)

    ax.set_xlim(-0.5, len(project_codes)-0.5)
    ax.set_xticklabels(cohorts, rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('')
    exp_colors = utils.get_experience_level_colors()
    experience_levels = utils.get_experience_levels()
    c = np.where(np.asarray(experience_levels)==title)[0][0]
    ax.set_title(title, color=exp_colors[c])
    if ylabel is None:
        ylabel = metric
    ax.set_ylabel(ylabel)
    # ax.legend(bbox_to_anchor=(1,1), fontsize='xx-small')

    # add stats to plot if only looking at experience levels
    if plot_stats:
        # stats dataframe to save
        tukey = pd.DataFrame()
        ax, tukey_table = add_stats_to_plot(data, metric, ax, ymax=ymax, show_ns=show_ns,
                                                column_to_compare='project_code')
        # aggregate stats
        tukey_table['metric'] = metric
        tukey = pd.concat([tukey, tukey_table])

    ax.set_ylim(ymin=ymin)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, metric + suffix)
    stats_filename = metric + '_stats' + suffix
    try:
        if plot_stats:
            print('saving_stats')
            tukey.to_csv(os.path.join(save_dir, folder, stats_filename + '_tukey.csv'))
        # save metric values
        cols_to_groupby = ['project_code']
        stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        stats.to_csv(os.path.join(save_dir, folder, stats_filename + '_values.csv'))
    except BaseException:
        print('stats did not save for', metric)
    return ax


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
    order = np.sort(data.cell_type.unique())
    ax = sns.boxplot(data=data, x='cell_type', y='days_in_stage', order=order, width=0.8, linewidth=0.8,
                     hue=stage_column, hue_order=behavior_stages, palette=colors, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Days in stage')
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
        figsize = (2, 3)
        fig, ax = plt.subplots(figsize=figsize)

    colors = utils.get_experience_level_colors()
    experience_levels = np.sort(platform_experiments.experience_level.unique())

    ax = sns.boxplot(data=exposures, x='experience_level', y='prior_exposures_to_image_set',
                     order=experience_levels, palette=colors, width=0.5, ax=ax)
    ax.set_ylabel('# sessions')
    ax.set_xlabel('')

    stats = exposures.groupby(['experience_level']).describe()[['prior_exposures_to_image_set']]
    stats.columns = stats.columns.droplevel(0)

    # xticklabels = utils.get_new_experience_levels()
    ax.set_xticklabels(experience_levels, rotation=90)
    ax.set_title('stimulus exposure')

    for i, experience_level in enumerate(experience_levels):
        y = int(np.round(stats.loc[experience_level]['mean'], 0))
        if experience_level == 'Novel':
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

    exposures['prior_exposures_to_image_set'] = exposures['prior_exposures_to_image_set'].astype(int)

    if ax is None:
        figsize = (2.5, 3)
        fig, ax = plt.subplots(figsize=figsize)

    #     ax = sns.boxplot(data=exposures, x='cell_type', y='prior_exposures_to_image_set',
    #                order=cell_types, palette='gray', width=0.5, ax=ax)

    ax = sns.violinplot(data=exposures, x='cell_type', y='prior_exposures_to_image_set', order=cell_types,
                        orient='v', palette='dark:white', ax=ax)
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
    ax, tukey_table = add_stats_to_plot(exposures, 'prior_exposures_to_image_set', ax, ymax=ymax,
                                        show_ns=True, column_to_compare='cell_type')
    # aggregate stats
    tukey_table['metric'] = 'prior_exposures_to_image_set'
    tukey = pd.concat([tukey, tukey_table])

    if save_dir:
        # save plot
        utils.save_figure(fig, figsize, save_dir, folder, 'stimulus_exposures_before_novel_plus' + suffix)
        # save stats
        stats = exposures.groupby(['cell_type', 'experience_level']).describe()[['prior_exposures_to_image_set']]
        stats.to_csv(os.path.join(save_dir, folder, 'stimulus_exposures_before_novel_plus_stats.csv'))


def plot_prior_exposures_to_image_set_before_platform_ophys_sessions_horiz(platform_experiments, behavior_sessions, save_dir=None, folder=None, suffix='', ax=None):
    """
    Creates a boxplot showing the number of prior exposures to each image set for each experience level (Familiar, Novel, Novel +)
    for the set of mice and sessions in platform_experiments
    Boxplot is distribution of number of prior exposures across mice
    """

    # get the behavior sessions corresponding to the ophys sessions included in platform dataset
    paper_ophys_behavior_sessions = behavior_sessions.loc[platform_experiments.behavior_session_id.unique()]
    exposures = paper_ophys_behavior_sessions.set_index(['experience_level', 'mouse_id'])[['prior_exposures_to_image_set']].reset_index()

    if ax is None:
        figsize = (3.5, 2)
        fig, ax = plt.subplots(figsize=figsize)

    colors = utils.get_experience_level_colors()
    experience_levels = np.sort(platform_experiments.experience_level.unique())

    ax = sns.boxplot(data=exposures, y='experience_level', x='prior_exposures_to_image_set', orient='h',
                     order=experience_levels, palette=colors, width=0.5, ax=ax)
    ax.set_xlabel('# sessions')
    ax.set_ylabel('')

    stats = exposures.groupby(['experience_level']).describe()[['prior_exposures_to_image_set']]
    stats.columns = stats.columns.droplevel(0)

    # xticklabels = utils.get_new_experience_levels()
    ax.set_yticklabels(experience_levels, rotation=0)
    ax.set_title('Stimulus exposure')

    for i, experience_level in enumerate(experience_levels):
        x = int(np.round(stats.loc[experience_level]['mean'], 0))
        if experience_level == 'Novel':
            text = '0'
            x = x + 4
            i = 0.85
            pos = 0.3
        elif experience_level == 'Familiar':
            x = x + 12
            pos = 0.45
            text = str(int(np.round(stats.loc[experience_level]['mean'], 0))) + '+/-' + str(int(np.round(stats.loc[experience_level]['std'], 0)))
        else:
            x = x + 8
            pos = 0.15
            text = str(int(np.round(stats.loc[experience_level]['mean'], 0))) + '+/-' + str(int(np.round(stats.loc[experience_level]['std'], 0)))
        ax.text(x, i + pos, text, fontsize=14, rotation='horizontal')

    if save_dir:
        # save plot
        utils.save_figure(fig, figsize, save_dir, folder, 'stimulus_exposures_before_platform_expts_boxplot_horiz' + suffix)
        # save stats
        stats = exposures.groupby(['experience_level']).describe()[['prior_exposures_to_image_set']]
        stats.to_csv(os.path.join(save_dir, folder, 'stimulus_exposures_before_platform_expts_stats.csv'))

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
    ax.set_xticklabels(xticklabels, rotation=90, )
    ax.set_title('stimulus exposure\nall sessions')

    for i, experience_level in enumerate(experience_levels):
        y = int(np.round(stats.loc[experience_level]['mean'], 0))
        if experience_level == 'Novel':
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


def plot_stimulus_exposure_prior_to_imaging(behavior_sessions, column_to_group='behavior_stage',
                                            save_dir=None, folder=None, suffix='', ax=None):
    """
    Creates a boxplot showing the number of sessions for each experience level or session type
    prior to the start of 2P imaging, for the set of mice included in behavior_sessions
    """

    data = behavior_sessions.copy()
    # limit to non-ophys sessions
    data = data[data.session_type.str.contains('OPHYS') == False]
    # count number of sessions of each experience level
    exposures = data.groupby([column_to_group, 'mouse_id']).count()[['session_type']].reset_index().rename(
        columns={'session_type': 'n_sessions'})

    exposures['n_sessions'] = exposures['n_sessions'].astype(int)

    if ax is None:
        figsize = (2.5, 3)
        fig, ax = plt.subplots(figsize=figsize)

    col_values = np.sort(exposures[column_to_group].unique())

    if column_to_group == 'experience_level':
        colors = utils.get_experience_level_colors()
        c = [colors[0], [0.5, 0.5, 0.5]]
    elif  column_to_group == 'behavior_stage':
        colors = utils.get_behavior_stage_color_map()
        c = [colors[col_value] for col_value in col_values]
    elif column_to_group == 'stimulus_type':
        colors = utils.get_stimulus_color_map()
        c = [colors[col_value] for col_value in col_values]

    ax = sns.boxplot(data=exposures, x=column_to_group, y='n_sessions',
                     order=col_values, palette=c, width=0.5, ax=ax)
    ax.set_ylabel('Number of sessions')
    ax.set_xlabel('')

    stats = exposures.groupby(column_to_group).describe()[['n_sessions']]
    stats.columns = stats.columns.droplevel(0)

    # ax.set_xticklabels(['Gratings', 'Familiar\nimages'], rotation=90)
    ax.set_title('Stimulus exposure\nduring training')

    for i, col_value in enumerate(col_values):
        y = np.max(stats.loc[col_value]['75%'])+5
        text = str(int(np.round(stats.loc[col_value]['mean'], 0))) + '+/-' + str(
            int(np.round(stats.loc[col_value]['std'], 0)))
        ax.text(i + 0.1, y, text, fontsize=14, rotation='horizontal')

    ax.set_xticklabels([col_value.replace('_', ' ') for col_value in col_values], rotation=90)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'stimulus_exposure_prior_to_imaging_boxplot_'+ column_to_group + suffix)
        # save stats
        stats = exposures.groupby(column_to_group).describe()[['n_sessions']]
        stats.to_csv(os.path.join(save_dir, folder, 'stimulus_exposure_prior_to_imaging_stats_'+column_to_group+'.csv'))

def plot_training_history_for_mice(behavior_sessions, color_column='session_type', color_map=sns.color_palette(),
                                   group_by_cre_line=True, save_dir=None, folder=None, suffix='', ax=None):
    """
    plots the session sequence for all mice in behavior_sessions table, sorted by total # of sessions per mouse

    sessions are colored by the provided color_column and color_map
    values of color_column must match keys of color_map
    acceptable pairs for color_column and color_map
    color_column = 'session_type' : color_map = utils.get_session_type_color_map()
    color_column = 'stimulus' : color_map = utils.get_stimulus_color_map(as_rgb=True)
    color_column = 'stimulus_phase' : color_map = utils.get_stimulus_phase_color_map(as_rgb=True)

    """

    if group_by_cre_line:
        # group by mice and count n_session per mouse to get the max n_sessions and list of mouse_ids to plot
        n_sessions = behavior_sessions.groupby(['cre_line', 'mouse_id']).count().rename(columns={'equipment_name': 'n_sessions'})[['n_sessions']]
        n_sessions = n_sessions.reset_index()
        n_sessions = n_sessions.sort_values(by=['cre_line', 'n_sessions'])
    else:
        n_sessions =  behavior_sessions.groupby(['mouse_id']).count().rename(columns={'equipment_name': 'n_sessions'})[['n_sessions']]
        n_sessions = n_sessions.reset_index().sort_values(by=['n_sessions'])
    max_n_sessions = np.amax(n_sessions.n_sessions.values)
    mouse_ids = n_sessions.mouse_id.values

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
        figsize = (10, float(n_mouse_ids) * 0.1)
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img.astype(int), aspect='auto')
    ax.set_ylim(0, n_mouse_ids)
    ax.invert_yaxis()

    ax.set_xlabel('Session number')
    ax.set_ylabel('Mouse number')
    ax.set_title('Training history')
    plot_behavior_performance_for_one_mouse

    if group_by_cre_line:
        # set ytick labels based on number of mice per cre line
        yticklabels = [0]
        for i, cre_line in enumerate(n_sessions.cre_line.unique()):
            yticklabels.append(yticklabels[i] + len(n_sessions[n_sessions.cre_line == cre_line]))
        ax.set_yticks(yticklabels)
        ax.set_yticklabels(yticklabels, fontdict={'verticalalignment': 'top'})
        # label with cell type
        for i, cre_line in enumerate(n_sessions.cre_line.unique()):
            cell_type = utils.convert_cre_line_to_cell_type(cre_line)
            ax.text(-2, (yticklabels[i] + yticklabels[i + 1]) / 2., cell_type.split(' ')[0], fontsize=16, ha='center',
                    va='center', rotation='vertical')
        suffix = suffix+'_group_by_cre'
    else:
        ax.set_yticks((0, n_mouse_ids))

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'training_history' + suffix)
    return ax


def plot_ophys_history_for_mice(behavior_sessions, color_column='ophys_stage', color_map=sns.color_palette(),
                                group_by_cre=False, label_rows_by_cre=False, label_with_mouse_id=False,
                                title=None, suffix='', save_dir=None, ax=None):
    """
    plots the ophys session sequence for all mice in behavior_sessions table, sorted by total # of sessions per mouse

    sessions are colored by the provided color_column and color_map
    values of color_column must match keys of color_map
    acceptable pairs for color_column and color_map
    color_column = 'ophys_stage' : color_map = utils.get_ophys_stage_color_map(as_rgb=True)
    color_column = 'stimulus' : color_map = utils.get_stimulus_color_map(as_rgb=True)
    color_column = 'stimulus_phase' : color_map = utils.get_stimulus_phase_color_map(as_rgb=True)

    group_by_cre: Bool, whether or not to order yaxis by cell type
    label_rows_by_cre: Bool, whether or not to put little symbols on each row to indicate the cre line
    label_with_mouse_id: Bool, whether or not to include mouse IDs on y axis
    """
    # group by mice and cre line and count n_session per mouse to get the max n_sessions and list of mouse_ids to plot
    if group_by_cre:
        n_sessions = \
        behavior_sessions.groupby(['cre_line', 'mouse_id']).count().rename(columns={'equipment_name': 'n_sessions'})[
            ['n_sessions']]
        n_sessions = n_sessions.reset_index()
        n_sessions = n_sessions.sort_values(by=['cre_line', 'n_sessions'])
    else:  # order by number of sessions
        n_sessions = behavior_sessions.groupby(['mouse_id']).count().rename(columns={'equipment_name': 'n_sessions'})[
            ['n_sessions']]
        n_sessions = n_sessions.reset_index()
        n_sessions = n_sessions.sort_values(by=['n_sessions'])

    max_n_sessions = np.amax(n_sessions.n_sessions.values)
    mouse_ids = n_sessions.mouse_id.values

    # create an array to fill in with session colors per mouse
    img = np.empty((len(mouse_ids), max_n_sessions, 3))
    img[:] = 256  # make the default value of 256 which is white in RGB space

    # create plot with expt colors image
    if ax is None:
        #         figsize = (6/1.5, n_mouse_ids*0.3/1.5)
        figsize = (6, float(len(mouse_ids)) * 0.3)
        fig, ax = plt.subplots(figsize=figsize)

    # loop through mice and create session colors image
    for mouse, mouse_id in enumerate(mouse_ids):
        # sort session in acquisition date order
        sessions = behavior_sessions[behavior_sessions.mouse_id == mouse_id].sort_values('date_of_acquisition')
        # fill in image array with the color from color_map for the corresponding color_col
        for session, session_id in enumerate(sessions.index.values):
            session_data = sessions.loc[session_id]
            color_column_value = session_data[color_column]
            img[mouse, session, :] = color_map[color_column_value]
            # if its passive, put a P on it
            if session_data.passive == True:
                ax.text(session, mouse, 'P', fontsize=8, ha='center', va='center', )
            # if the session doesnt have ophys at all (i.e. failed QC, put an X on it)
            elif session_data.has_ophys == False:
                ax.text(session, mouse, 'X', fontsize=8, ha='center', va='center', )
            # if the session is in the platform ophys dataset, give it a letter for its experience level
            if session_data.in_dataset and session_data.experience_level == 'Familiar':
                ax.text(session, mouse, 'F', fontsize=12, ha='center', va='center',
                        fontdict={'fontweight': 'bold', 'color': 'white'})
            elif session_data.in_dataset and session_data.experience_level == 'Novel':
                ax.text(session, mouse, 'N', fontsize=12, ha='center', va='center',
                        fontdict={'fontweight': 'bold', 'color': 'white'})
            elif session_data.in_dataset and session_data.experience_level == 'Novel +':
                ax.text(session, mouse, 'N+', fontsize=12, ha='center', va='center',
                        fontdict={'fontweight': 'bold', 'color': 'white'})
                # if its a Novel +, change the color to purple
                img[mouse, session, :] = np.array(
                    [x * 255 for x in list(utils.get_colors_for_session_numbers_GH()[0])]).astype(np.uint8)

            if color_column == 'stimulus_phase':
                # if its not in the final dataset and its not passive, make the color lighter
                if session_data.in_dataset == False and session_data.passive == False:
                    white = (255, 255, 255)
                    img[mouse, session, :] = (
                                color_map[color_column_value] + (white - color_map[color_column_value]) * 0.7)
    # create plot with expt colors image
    if ax is None:
        figsize = (10, len(mouse_ids) * 0.1)
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img.astype(int), aspect='auto')
    ax.set_ylim(-0.5, len(mouse_ids) - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel('Ophys session number')
    ax.set_ylabel('Mouse number')
    if title is None:
        ax.set_title('Ophys session sequence')
    else:
        ax.set_title(title)

    # ytick labels
    if group_by_cre:
        # get ytick labels based on number of mice per cre line
        yticklabels = [0]
        for i, cre_line in enumerate(n_sessions.cre_line.unique()):
            yticklabels.append(yticklabels[i] + len(n_sessions[n_sessions.cre_line == cre_line]))
        # label with cell types
        for i, cre_line in enumerate(n_sessions.cre_line.unique()):
            cell_type = utils.convert_cre_line_to_cell_type(cre_line)
            ax.text(-2, (yticklabels[i] + yticklabels[i + 1]) / 2., cell_type.split(' ')[0], fontsize=16, ha='center',
                    va='center', rotation='vertical')
    elif label_with_mouse_id:
        ax.set_yticks(np.arange(0, len(mouse_ids)))
        ax.set_yticklabels(mouse_ids, fontsize=8)
    else:
        ax.set_yticks([-0.5, len(mouse_ids) - 0.5])
        ax.set_yticklabels([0, len(mouse_ids)])

    if label_rows_by_cre:
        for mouse, mouse_id in enumerate(mouse_ids):
            cre_line = behavior_sessions[behavior_sessions.mouse_id == mouse_id].cre_line.values[0]
            if cre_line == 'Slc17a7-IRES2-Cre':  # square
                ax.text(-1, mouse, '\u25a1', fontsize=8, ha='center', va='center', )
            elif cre_line == 'Sst-IRES-Cre':  # triangle
                ax.text(-1, mouse, '\u25b2', fontsize=8, ha='center', va='center', color='black')
            elif cre_line == 'Vip-IRES-Cre':  # circle
                ax.text(-1, mouse, '\u25cf', fontsize=8, ha='center', va='center', color='black')
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, 'training_history', 'ophys_session_sequence' + suffix)

    return ax


def plot_behavior_performance_for_one_mouse(behavior_stats, mouse_id, metric, method,
                                            hue='stimulus', x='date_of_acquisition',
                                            ylabel=None, use_session_number=True, remove_passive=True,
                                            save_dir=None, folder='behavior_performance', ax=None):
    '''
    plot behavior performance over time for a given behavior metric, for a given mouse

    behavior_stats: DataFrame containing behavior performance statistics, rows are behavior_session_ids,
                    cols must include mosue_id, metric, x, and hue
    mouse_id: str, mouse ID to pull data from for plot
    metric: column in behavior_stats to use for y axis values
    method: str, method of calculating behavior stats, as provided to the function:
                    visual_behavior.utilities.get_behavior_stats_for_sessions()
    hue: column in behavior_stats to use for coloring datapoints, typically "stimulus" or "behavior_stage" or "behastimulus_experience_levelvior_stage"
    x: column in behavior stats to use for sorting and plotting xaxis, typically 'date_of_acquisition'
    ylabel: string to label y axis with, otherwise will use metric value
    yse_session_number: Bool, if True, will plot session numbers as integers on x axis,
                                if False, will show stimulus names / session types on y axis
    remove_passive: Bool, whether or not to remove passive sessions from plot
    '''

    if (hue == 'stimulus') or (hue == 'stimulus_experience_level'):
        color_map = utils.get_stimulus_color_map(as_rgb=False)
    elif hue == 'behavior_stage':
        color_map = utils.get_behavior_stage_color_map(as_rgb=False)
    else:
        color_map = sns.color_palette()

    data = behavior_stats[behavior_stats.mouse_id==mouse_id].sort_values(by=x)
    if remove_passive:
        data = data[data.stimulus.str.contains('passive') == False]

    colors = [color_map[value] for value in data[hue].unique()]
    data[hue] = [hue.replace('_', ' ') for hue in data[hue].values]

    if ax is None:
        figsize = (10,3)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.pointplot(data=data, x=x, y=metric, hue=hue, hue_order=data[hue].unique(), linewidth='none', palette=colors, ax=ax)
    ax.legend(bbox_to_anchor=(1,1), fontsize='x-small')
    ax.set_xticklabels(data[hue].values, rotation=90);
    ax.set_ylim(ymin=0)
    ax.set_title(str(mouse_id)+'\nExample mouse training history')
    if ylabel is None:
        ylabel = metric
    if use_session_number:
        ax.set_xticks(np.arange(0, len(data), 5));
        ax.set_xticklabels(np.arange(0, len(data), 5), rotation=0);
        ax.set_xlabel('Session number')
    ax.set_ylabel(ylabel)
    if save_dir:
        utils.save_figure(fig, figsize, os.path.join(save_dir, folder), 'behavior_performance_over_time_'+method,
                        metric+'_mouse_id_'+str(mouse_id)+'_'+hue)
    return ax

def plot_response_rate_trial_types(data, save_dir=None, suffix='', ax=None):
    '''
    plot response rate for all sessions in data, with trial types on the x-axis
    examples of trial types: change vs non change, or go vs catch, or hit vs FR
    Trial types shown depend on values in 'trial_types' column of input data

    data: dataframe of behavior statistics where every row is one behavior session and the columns are performance metrics
        should include column "response_probability"
    '''
    trial_types = data.trial_type.unique()
    if ax is None:
        figsize = (2,3)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.pointplot(data=data, x='trial_type', y='response_probability',
                    order=trial_types, color='k', ax=ax)
    ax = sns.swarmplot(data=data, x='trial_type', y='response_probability',
                    order=trial_types, color='gray', s=2, ax=ax)
    ax.set_ylabel('Response rate')
    ax.set_xticklabels(trial_types, rotation=45)
    ax.set_xlabel('')
    ax.set_ylim(-0.01, 1)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, 'response_rate', 'response_rate_trial_types' + suffix)
    return ax



def plot_lick_raster_for_trials(trials, title='', save_dir=None, filename=None, suffix='', ax=None):
    # trials = dataset.trials
    # image_set = dataset.metadata.session_type.values[0][-1]
    # mouse_id = str(dataset.metadata.donor_id.values[0])
    if ax is None:
        figsize = (4, 5)
        fig, ax = plt.subplots(figsize=figsize)
    for trial in range(len(trials)):
        trial_data = trials.iloc[trial]
        trial_start = trial_data.start_time - trial_data.change_time
        # plot trial
        if trial_data.go == True:
            color = sns.color_palette()[0]
        else:
            color = 'white'
        # ax.axhspan(trial, trial + 1, -200, 200, palette='dark:white', alpha=0.075*1.5)
        # plot lines in between trials
        ax.vlines(trial_start, trial, trial + 1, color='gray', linewidth=0.5, alpha=0.5)
        # plot line at trial start
        # ax.vlines(0, trial, trial + 1, color='gray', linewidth=1, linestyle='--')
        # plot licks
        lick_times = [(t - trial_data.change_time) for t in trial_data.lick_times]
        if len(lick_times) > 0:
            ax.vlines(lick_times, trial, trial + 1, color='k', linewidth=1)
            # rewarded lick is a different color
            ax.vlines(lick_times[0], trial, trial + 1, color='b', linewidth=1)
        # plot rewards
        # if np.isnan(trial_data.reward_time) == False:
        #     reward_time = trial_data.reward_time - trial_data.change_time
        #     ax.plot(reward_time, trial + 0.5, '^', color='blue', label='reward', markersize=3)
    # plot reward window
    color = sns.color_palette()[0]
    ax.axvspan(0.15, 0.75, facecolor=color, alpha=.4, edgecolor='none')
    ax.grid(False)
    ax.set_ylim(0, len(trials))
    ax.set_xlim([-1, 5])
    ax.set_ylabel('Trial number')
    ax.set_xlabel('Time from change (sec)')
    ax.set_xticks(np.arange(0, 5, 2))
    # ax.set_title('M'+mouse_id+' image set '+image_set, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.invert_yaxis()
    # plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.3)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, 'lick_rasters', filename+suffix)
    return ax

def plot_response_probability_heatmaps_for_cohorts(behavior_sessions, save_dir=None):
    '''
    Plot a heatmaps of response probability across image transitions for Familiar and Novel images
    for each cohort of mice (mice trained on image set A and mice trained on image set B)
    '''
    import visual_behavior.visualization.behavior as behavior

    if 'last_familiar_active' not in behavior_sessions.columns:
        # make fake ophys_container_id column so the below functions will work
        behavior_sessions['ophys_container_id'] = behavior_sessions.mouse_id.values
        # add last familiar and first novel columns
        behavior_sessions = utilities.add_date_string(behavior_sessions)
        behavior_sessions = utilities.add_n_relative_to_first_novel_column(behavior_sessions)
        behavior_sessions = utilities.add_first_novel_column(behavior_sessions)
        behavior_sessions = utilities.add_second_novel_active_column(behavior_sessions)
        behavior_sessions = utilities.add_last_familiar_active_column(behavior_sessions)

    familiar_sessions = behavior_sessions[behavior_sessions.last_familiar_active==True].index.values
    novel_sessions = behavior_sessions[behavior_sessions.first_novel==True].index.values
    print(len(familiar_sessions), len(novel_sessions))

    # get response probability dataframe
    engaged_only = True
    familiar_response_probability = behavior.aggregate_response_probability_across_sessions(familiar_sessions, engaged_only=engaged_only)
    novel_response_probability = behavior.aggregate_response_probability_across_sessions(novel_sessions, engaged_only=engaged_only)

    # add metadata
    familiar_response_probability = familiar_response_probability.merge(behavior_sessions, on='behavior_session_id')
    novel_response_probability = novel_response_probability.merge(behavior_sessions, on='behavior_session_id')

    # make the plot
    figsize = (10,10)
    fig, ax = plt.subplots(2,2, figsize=figsize)
    ax = ax.ravel()

    cmap = 'Greys'
    colors = utils.get_colors_for_session_numbers()

    # A-B mice
    familiar_data = familiar_response_probability[familiar_response_probability.project_code.isin(['VisualBehavior', 'VisualBehaviorMultiscope'])]
    novel_data = novel_response_probability[novel_response_probability.project_code.isin(['VisualBehavior', 'VisualBehaviorMultiscope'])]
    # get matrices
    familiar_response_matrix = behavior.average_response_probability_across_sessions(familiar_data, sort=True)
    novel_response_matrix = behavior.average_response_probability_across_sessions(novel_data, sort=True)

    ax[0] = sns.heatmap(familiar_response_matrix, cmap=cmap, vmin=0, vmax=1, square=True,
                    cbar_kws={'label':'Response probability', 'shrink':0.7}, ax=ax[0])
    ax[0].set_xlabel('Change image')
    ax[0].set_ylabel('Initial image')
    ax[0].set_title('image set A\nFamiliar images', color=colors[3])

    ax[1] = sns.heatmap(novel_response_matrix, cmap=cmap, vmin=0, vmax=1, square=True,
                    cbar_kws={'label':'Response probability', 'shrink':0.7}, ax=ax[1])
    ax[1].set_xlabel('Change image')
    ax[1].set_ylabel('Initial image')
    ax[1].set_title('image set B\nNovel images', color=colors[0])


    # B-A mice
    familiar_data = familiar_response_probability[familiar_response_probability.project_code.isin(['VisualBehaviorTask1B'])]
    novel_data = novel_response_probability[novel_response_probability.project_code.isin(['VisualBehaviorTask1B'])]
    # get matrices
    familiar_response_matrix = behavior.average_response_probability_across_sessions(familiar_data, sort=True)
    novel_response_matrix = behavior.average_response_probability_across_sessions(novel_data, sort=True)

    ax[2] = sns.heatmap(familiar_response_matrix, cmap=cmap, vmin=0, vmax=1, square=True,
                    cbar_kws={'label':'Response probability', 'shrink':0.7}, ax=ax[2])
    ax[2].set_xlabel('Change image')
    ax[2].set_ylabel('Initial image')
    ax[2].set_title('image set B\nFamiliar images', color=colors[0])

    ax[3] = sns.heatmap(novel_response_matrix, cmap=cmap, vmin=0, vmax=1, square=True,
                    cbar_kws={'label':'Response probability', 'shrink':0.7}, ax=ax[3])
    ax[3].set_xlabel('Change image')
    ax[3].set_ylabel('Initial image')
    ax[3].set_title('image set A\nNovel images', color=colors[3])

    fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, 'response_probability', 'response_probability_heatmaps_engaged_only')

####### figure 4 plots ########

def plot_area_depth_modulation(combined_mean_table, ax = None):
    '''plot modulation index by area and layer for each cluster type.
    Parameters
    ----------
    combined_mean_table : dataframe
        dataframe with mean modulation index values for each cluster id'''
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize = (8,4.5))
    plt.rcParams['font.size'] = 14
    color_map = []
    for c, cre in enumerate(combined_mean_table.cre_line.unique()):
        color_map.append(gvt.project_colors()[cre])

    sns.scatterplot(data=combined_mean_table, y='modulation_index_layer', x = 'modulation_index_area', hue = 'cre_line',
                size='size_col', palette = color_map, sizes=(50, 500), alpha=0.7, ax=ax)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    text = legend.get_texts()
    
    for c, cre in enumerate(combined_mean_table.cre_line.unique()):
        for cluster_id in range(1,13):
            cluster_table = combined_mean_table[(combined_mean_table.cre_line==cre) &
                                            (combined_mean_table.cluster_id==cluster_id)]    
            plt.text(cluster_table['modulation_index_area'], cluster_table['modulation_index_layer'], str(cluster_id), ha='right', va='bottom')
    #     ax.legend('')
    plt.plot([0, 0], [-1.2,1.4], '--', color='gray')
    plt.plot([-1.1, 1.1], [0,0], '--', color='gray')
    ax.set_xlabel( '<- LM      V1 ->', fontsize=20)
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_xticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0']) 
    ax.set_ylabel( '<- lower    upper ->', fontsize=20)
    ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.set_ylim([-1.2, 1.2])
    ax.set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0']) 
    text[0].set_text('cell type')
    text[1].set_text('Excitatory')
    text[2].set_text('Sst Inhibitory')
    text[3].set_text('Vip Inhibitory')
    text[4].set_text('cluster size \n(cre proportion)')
    plt.tight_layout()
    
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


