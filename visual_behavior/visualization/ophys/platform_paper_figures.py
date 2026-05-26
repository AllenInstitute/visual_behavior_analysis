"""
Created on Thursday September 23 2021

@author: marinag
"""
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.ophys.response_analysis.response_processing as rp
# Re-export legacy test_significant_metric_averages from platform_paper_stats so
# inline callers in this module (e.g., the heatmap functions) keep working.
from visual_behavior.visualization.ophys.platform_paper_stats import (
    test_significant_metric_averages,
    test_significant_metric_averages_mlm,
    compute_stats,
    insert_stats_metadata,
)
from visual_behavior_glm import GLM_visualization_tools as gvt
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

def _clean_filename(name):
    """
    Collapse runs of underscores to a single underscore and trim boundary
    underscores. Wrap filename constructions with this to be robust to inputs
    like ``suffix=''``, ``suffix='_foo'``, or ``suffix='foo'`` -- which would
    otherwise produce ``foo__bar`` (double underscore), ``foo_`` (trailing
    underscore), or ``foobar`` (missing separator) respectively. Operates on
    each '/' component independently so directory separators are preserved.
    """
    parts = name.split('/')
    parts = [re.sub(r'_{2,}', '_', p).strip('_') for p in parts]
    return '/'.join(parts)


def _norm_suffix(suffix):
    """
    Normalize a user-supplied filename suffix so that non-empty suffixes
    always start with a single underscore. Returns ``''`` for None/empty
    so it can be safely concatenated as ``<base> + suffix``.

    Examples::

        _norm_suffix(None)     -> ''
        _norm_suffix('')       -> ''
        _norm_suffix('v1')     -> '_v1'
        _norm_suffix('_v1')    -> '_v1'
        _norm_suffix('__v1')   -> '_v1'  (excess leading underscores collapsed)
    """
    if not suffix:
        return ''
    return '_' + suffix.lstrip('_')


# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.top': False, 'axes.spines.right': False})  # ticks or white
sns.set_palette('deep')

plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True


def _heatmap_omnibus_p(panel_stats):
    """
    Pull the omnibus p-value out of a stats_table returned by ``compute_stats``.
    The heatmap functions use this only to decide whether to draw a significance
    star on a given cell. The full pairwise table (with model_type, icc,
    mouse_variance, descriptives, etc.) is saved alongside via
    ``insert_stats_metadata`` -- it isn't collapsed away anymore.

    Returns NaN for empty input (no comparison possible).
    """
    if panel_stats is None or len(panel_stats) == 0:
        return np.nan
    return float(panel_stats['omnibus_pvalue'].iloc[0])


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
                
                if s == 0:
                    if session_id_for_area_depths:
                        tmp = mouse_expts[(mouse_expts.ophys_session_id == session_id_for_area_depths) &
                                        (mouse_expts.ophys_container_id == ophys_container_id)]
                        area = tmp.targeted_structure.values[0]
                        depth = int(tmp.imaging_depth.values[0])
                    ax[i].text(s=area + ' ' + str(depth), x=-20, y=dataset.max_projection.data.shape[0] / 2,
                            ha='right', va='center', rotation=90, fontsize=8)
            except:
                print('could not plot for experiment', ophys_experiment_id, session_type, area, depth)
            
            if c == 0:
                ax[i].set_title(str(ophys_session_id) + '\n' + session_type, fontsize=6)
            i += 1

    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    if save_dir:
        cre = dataset.metadata['cre_line'][:3]
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(str(mouse_id) + '_' + cre + '_max_projection_images'))


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
        filename = _clean_filename(str(mouse_id) + '_' + str(
            ophys_session_id) + '_' + session_type + '_' + cre + '_' + trace_type + '_' + cmap)
        fig.suptitle(filename, x=0.4, y=1.1, fontsize=16)
        utils.save_figure(fig, figsize, save_dir, 'traces_heatmaps', _clean_filename(filename))

    return ax



# basic characterization #########################

def plot_cell_count_by_depth(cells_table, project_code=None, suptitle=None, horiz=True,
                             save_dir=None, folder=None, suffix='', ax=None):
    suffix = _norm_suffix(suffix)
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
            figsize = (3, 12)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharey=True)
    for i, cell_type in enumerate(utils.get_cell_types()):
        if hue:
            ax[i] = sns.histplot(data=cells_table[cells_table.cell_type == cell_type], #bins=20,
                                 binwidth=binwidth, discrete=False, binrange=[0, 400],
                                 hue='targeted_structure', y='imaging_depth', hue_order=areas,
                                 palette=colors, multiple='stack', stat='count', ax=ax[i])
            _legend = ax[i].get_legend()

            if _legend: _legend.remove()
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
        plt.subplots_adjust(wspace=0.2)
        plt.suptitle(suptitle, x=0.5, y=1.2)
    else:
        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        plt.suptitle(suptitle, x=0.5, y=0.96, fontsize=18)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('cell_count_by_depth_areas' + suffix))
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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('cell_count_by_depth_areas_'+suptitle))
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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('cell_count_by_depth_areas_'+suptitle))
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
    suffix = _norm_suffix(suffix)
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
                              hue_order=experience_levels, palette=palette, dodge=0, linestyle='none', ax=ax[i])
        ax[i].set_xticklabels(experience_levels, rotation=45)
    #     ax[i].legend(fontsize='xx-small', title='')
        _legend = ax[i].get_legend()

        if _legend: _legend.remove()
        ax[i].set_title(cell_type)
        ax[i].set_ylim(ymin=0)
        ax[i].set_xlabel('')
#         ax[i].set_ylim(0,1)
    if format_fig:
        fig.tight_layout()
    if save_dir:
        fig_title = _clean_filename(df_name.split('-')[0] + '_n_total_cells' + suffix)
        utils.save_figure(fig, figsize, save_dir, 'n_segmented_cells', _clean_filename(fig_title))

# population averages across session & within epochs #####################


def plot_population_averages_for_condition(multi_session_df, data_type, event_type, hue_column,
                                            project_code=None, timestamps=None, palette=None, ylims=None,
                                            title=None, suptitle=None, xlabel='Time (s)', ylabel='Response',
                                            horizontal=True, xlim_seconds=None, interval_sec=1, legend=False,
                                            linewidth=1, save_dir=None, folder=None, suffix='', ax=None):
    '''
    Function to plot a population average response across for a single condition from a dataframe containing event aligned timeseries,
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
    suffix = _norm_suffix(suffix)

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

    # set plot size depending on what type of data it is
    if data_type in ['dff', 'events', 'filtered_events']:
        if horizontal:
            figsize = (5, 2.5)
        else:
            figsize = (3, 3)  # for changes and omissions
    elif data_type in ['running_speed', 'pupil_width', 'lick_rate']:
        if horizontal:
            figsize = (5, 4)  # for behavior timeseries
        else:
            figsize = (2.5, 3)  # for image response

    # create axes
    if ax is None:
        if horizontal:
            suffix = suffix+'_horiz'
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        format_fig = False

    # loop over conditions and plot
    for c, hue in enumerate(hue_conditions):
        # try:
        cdf = sdf[(sdf[hue_column] == hue)]
        traces = cdf.mean_trace.values
        # plot average of all traces for this condition
        ax = utils.plot_mean_trace(np.asarray(traces), timestamps, ylabel=ylabel,
                                        legend_label=hue, color=palette[c], interval_sec=interval_sec,
                                        xlim_seconds=xlim_seconds, linewidth=linewidth, ax=ax)
        # plot stimulus timing overlaid on trace
        ax = utils.plot_flashes_on_trace(ax, timestamps, change=change, omitted=omitted, alpha=0.1)

        # color title by experience level if axes are experience levels
        if title:
            ax.set_title(title)
        ax.set_xlim(xlim_seconds)
        if ylims is not None: 
            ax.set_ylim(ylims)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    if legend:
        if hue_column == 'passive':
            ax.legend(['active', 'passive'], loc='upper center', fontsize='x-small', bbox_to_anchor=(1.3,1))
        else:
            ax.legend(title=hue_column, loc='upper center', fontsize='x-small', title_fontsize='x-small',
                         bbox_to_anchor=(1.2, 1))
    if project_code:
        if suptitle is None:
            suptitle = 'population average - ' + data_type + ' response - ' + project_code[14:]
    if suptitle:
        if horizontal:
            y = 1.1
        else:
            y = 0.95
        plt.suptitle(suptitle, x=0.51, y=y, fontsize=18)
    
    # plt.rcParams["savefig.bbox"] = "tight"
    if save_dir:
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.rcParams["savefig.bbox"] = "tight"

        fig_title = _clean_filename('population_average_' + hue_column + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))

    return ax


def plot_population_averages_for_conditions(multi_session_df, data_type, event_type, axes_column, hue_column,
                                            project_code=None, timestamps=None, palette=None, sharey=False,
                                            title=None, suptitle=None, xlabel='Time (s)', ylabel='Response',
                                            horizontal=True, xlim_seconds=None, interval_sec=1, legend=False,
                                            linewidth=1, save_dir=None, folder=None, suffix='', ax=None):
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
    suffix = _norm_suffix(suffix)

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
            figsize = (3.8, 3 * n_axes_conditions)  # for changes and omissions
    elif data_type in ['running_speed', 'pupil_width', 'lick_rate']:
        if horizontal:
            figsize = (4 * n_axes_conditions, 3)  # for behavior timeseries
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
            ax[i] = utils.plot_mean_trace(np.asarray(traces), timestamps, ylabel=ylabel, alpha=0.3,
                                          legend_label=hue, color=palette[c], interval_sec=interval_sec,
                                          linewidth=linewidth, xlim_seconds=xlim_seconds, ax=ax[i])
        # plot stimulus timing overlaid on trace
        ax[i] = utils.plot_flashes_on_trace(ax[i], timestamps, change=change, omitted=omitted, 
                                            alpha=0.3, linewidth=linewidth)

        # color title by experience level if axes are experience levels
        if axes_column == 'experience_level':
            title_colors = utils.get_experience_level_colors()
            ax[i].set_title(axis, color=title_colors[i], fontsize=20)
        else:
            ax[i].set_title(axis)
        if title:
            ax[i].set_title(title)
        ax[i].set_xlim(xlim_seconds)
        ax[i].set_xlabel(xlabel)
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
        # only pass the Line2D handles (skipping the unlabeled fill_between Polygons from the SEM)
        # so the labels don't get mis-assigned to fill_betweens in artist creation order.
        line_handles = [l for l in ax[i].get_lines() if not l.get_label().startswith('_')]
        if hue_column == 'passive':
            ax[i].legend(line_handles, ['active', 'passive'], loc='upper center',
                         fontsize='x-small', bbox_to_anchor=(1.3, 1))
        else:
            ax[i].legend(handles=line_handles, title=hue_column, loc='upper center',
                         fontsize='x-small', title_fontsize='x-small',
                         bbox_to_anchor=(1.2, 1))
    if project_code:
        if suptitle is None:
            suptitle = 'population average - ' + data_type + ' response - ' + project_code[14:]
    if suptitle:
        if horizontal:
            y = 1.1
        else:
            y = 0.95
        plt.suptitle(suptitle, x=0.51, y=y, fontsize=18)
    
    plt.rcParams["savefig.bbox"] = "tight"
    if save_dir:
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.rcParams["savefig.bbox"] = "tight"

        fig_title = _clean_filename('population_average_' + axes_column + '_' + hue_column + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))

    return ax


def plot_population_averages_for_cell_types_across_experience(multi_session_df, xlim_seconds=[-1.25, 1.5], xlabel='time (s)',
                                                              ylabel='population average',  data_type='events', event_type='changes', interval_sec=1,
                                                              save_dir=None, folder=None, suffix=None, ax=None):
    # get important information
    suffix = _norm_suffix(suffix)
    experiments_table = loading.get_platform_paper_experiment_table()
    cell_types = np.sort(experiments_table.cell_type.unique())
    experience_levels = utils.get_experience_levels()
    palette = utils.get_experience_level_colors()#[:len(experience_levels)]

    # set formatting options
    interval_sec = 1
    dist = xlim_seconds[1]*0.07
    n_cols = len(experience_levels)
    n_rows = len(cell_types)
    small_fontsize = 9

    # define plot axes
    axes_column = 'experience_level'
    hue_column = 'experience_level'

    width = n_cols * np.sum(np.abs(xlim_seconds))
    if event_type == 'changes':
        label = 'Image change'
        label_color = sns.color_palette()[0]
        width = width * 1.2
    elif event_type == 'omissions':
        label = 'Image omission'
        label_color = sns.color_palette()[9]
    else: 
        label = 'Non-change image'
        label_color = sns.color_palette()[7]
        width = width * 1.5

    

    if ax is None:
        format_fig = True
        figsize = (width, 8)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, sharey='row', sharex='col')
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

        if cell_type == 'Excitatory':
            scale = 1.06
        else: 
            scale = 1.15
        # add line and label on yaxis corresponding to 1/3 of max y val
        ymin, ymax = ax[i * 3].get_ylim()
        ax[i * 3].set_ylim(ymin=0, ymax=ymax*scale)
        ymin, ymax = ax[i * 3].get_ylim()
        ax[i * 3].set_yticks([0, np.round(ymax * .3, 3)])
        ax[i * 3].set_yticklabels(['', np.round(ymax * .3, 3)], va='top', fontsize=small_fontsize)
        ax[i * 3].axvline(x=xlim_seconds[0] - dist, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)

        # label y axis with number of cells in first column panels
        n_cells = len(df[(df.cell_type == cell_type)].cell_specimen_id.unique())
        ax[i * 3].set_ylabel(cell_type+'\nn='+str(n_cells)+' cells', rotation=0, fontsize=14, ha='center', va='top', y=0.8)
        
        # annotate time axis and change/omission for excitatory only
        if cell_type == 'Excitatory':
            xmax = 0.5 / (np.abs(xlim_seconds[0]) + xlim_seconds[1])  # 0.5 / of total time
            y_time = (ymax - ymin) * 0.06
            y_label = -(ymax - ymin) * 0.12
            ax[i].axhline(y=-y_time, xmin=0, xmax=xmax, color='k', linewidth=1, clip_on=False)
            ax[i].annotate('0.5 s', xy=(xlim_seconds[0] - 0.1, y_label),
                        xycoords='data', xytext=(xlim_seconds[0] - 0.1 + 0.5, y_label), ha='center', va='top',
                        fontsize=small_fontsize, clip_on=False, annotation_clip=False)
            if xlim_seconds[0] > -0.75:
                x = 1
            else:
                x = 0
            # label image change or image omission on first axis
            ax[i+x].annotate(label, xy=(0.12, -0.1), xycoords=ax[i+x].get_xaxis_transform(), ha="left", va="top",
                            color=label_color, fontsize=small_fontsize+2, clip_on=False)
            ax[i+x].annotate('', xy=(0.01, -0.25), xycoords=ax[i+x].get_xaxis_transform(), xytext=(0.01, 0), fontsize=small_fontsize+2,
                                arrowprops=dict(arrowstyle="<-", color=label_color, lw=1), clip_on=False)

    if format_fig:
        for i in np.arange(3, 9):
            ax[i].set_title('')
        for i in np.arange(0, 6):
            ax[i].set_xlabel('')
    else:
        for i in np.arange(1, 3):
            for x in range(3):
                ax[i][x].set_title('')
        for i in np.arange(0, 2):
            for x in range(3):
                ax[i][x].set_xlabel('')

    for i in np.arange(0, 9):
        sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
        ax[i].tick_params(bottom=False, left=False, right=False, top=False)
    for i in [1, 2, 4, 5, 7, 8]:
        ax[i].axis('off')
    ax[6].set_xticklabels([])

    if save_dir:
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        fig_title = _clean_filename('population_average_cell_types_exp_levels' + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))

    return ax


def plot_population_averages_across_experience(multi_session_df, xlim_seconds=[-1.25, 1.5], xlabel='time (s)', ylabel='population\nresponse',
                                               data_type='events', event_type='changes', interval_sec=1,
                                               save_dir=None, folder=None, suffix=None, ax=None):
    # get important information
    suffix = _norm_suffix(suffix)
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
        fig_title = _clean_filename('population_average_exp_levels' + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))

    return ax


def plot_population_average_across_experience(multi_session_df, xlim_seconds=[-1.25, 1.5], xlabel='time (s)', ylabel='population\nresponse',
                                               data_type='events', event_type='changes', interval_sec=1,
                                               save_dir=None, folder=None, suffix=None, ax=None):
    # get important information
    suffix = _norm_suffix(suffix)
    palette = utilities.get_experience_level_colors()

    # define plot axes
    hue_column = 'experience_level'

    if ax is None:
        figsize = (4, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = ax.ravel()

    df = multi_session_df.copy()
    ax = plot_population_averages_for_conditions(df, data_type, event_type,
                                                 hue_column, horizontal=True,
                                                 xlim_seconds=xlim_seconds, interval_sec=interval_sec,
                                                 palette=palette, ax=ax)
    ax.set_ylabel(ylabel)

    if save_dir:
        fig_title = _clean_filename('population_average_exp_levels' + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))

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


def plot_mean_response_by_epoch(df, metric='mean_response', horizontal=True, ymin=0, ymax=None, 
                                ylabel='mean response', estimator=np.mean, epoch_dur_mins=5,       
                                legend=False, save_dir=None, folder='epochs', max_epoch=6, suptitle=None, palette=None, suffix='', ax=None):
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
    suffix = _norm_suffix(suffix)

    # add experience epoch column if it doesnt already exist
    # if 'experience_epoch' not in df.keys():
    # df = annotate_epoch_df(df)

    cell_types = utils.get_cell_types()
    # experience_levels = utils.get_new_experience_levels()
    # for novel + control
    experience_levels = utils.get_experience_levels()

    df = df[df.epoch <= max_epoch]
    max_n_sessions = len(df.epoch.unique())

    # if epoch is indexed at 0, add 1 for plotting purposes
    if 0 in df.epoch.unique(): 
        df['epoch'] = df['epoch']+1

    experience_epoch = np.sort(df.epoch.unique())
  
    xticks = np.arange(0, len(experience_epoch), 1)
    xticklabels = experience_epoch #np.arange(0, len(experience_epoch), 1)+1
    # xticklabels = [experience_epoch.split(' ')[1] for experience_epoch in experience_epoch]

    if palette is None:
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
            data = df[df.cell_type == cell_type]
            ax[i] = sns.pointplot(data=data, x='epoch', y=metric, hue='experience_level', hue_order=experience_levels,
                                  order=experience_epoch, palette=palette, ax=ax[i], estimator=estimator)

            if ymin is not None:
                ax[i].set_ylim(ymin=ymin)
            if ymax is not None:
                ax[i].set_ylim(ymax=ymax)
            ax[i].set_title(cell_type)
            ax[i].set_ylabel(ylabel)
            _legend = ax[i].get_legend()

            if _legend: _legend.remove()

            ax[i].set_xlim((xticks[0] - 0.5, xticks[-1] + 0.5))
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels((xticklabels)*epoch_dur_mins)
            ax[i].vlines(x=max_n_sessions + 0.5, ymin=0, ymax=1, color='gray', linestyle='--')
            ax[i].vlines(x=max_n_sessions + max_n_sessions + 1.5, ymin=0, ymax=1, color='gray', linestyle='--')
            if horizontal:
                ax[i].set_xlabel('Time (mins)')
                if i != 0: 
                    ax[i].set_ylabel('')
            else:
                ax[i].set_xlabel('')
        except Exception as e:
            print(e)

    ax[i].set_xlabel('Time (mins)')
    ax[i].tick_params(axis='both', which='major', labelsize=14)

    if legend:
        ax[i].legend(fontsize='x-small', bbox_to_anchor=(1,1))

    if format_fig:
        if suptitle is not None:
            plt.suptitle(suptitle, x=0.52, y=1.01, fontsize=18)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    if save_dir:
        fig_title = _clean_filename(metric + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))
    return ax


def plot_mean_response_by_epoch_all_cell_types(df, metric='mean_response', horizontal=True, ymin=0, ymax=None,
                                               ylabel='mean response', estimator=np.mean, epoch_dur_mins=5,
                                               legend=False, save_dir=None, folder='epochs', max_epoch=6,
                                               title=None, suptitle=None, palette=None, suffix='', ax=None):
    """
    Plots the mean metric value across 10 minute epochs within a session, averaged across cell types
    Typically used for plotting behavior changes across sessions, averaged across all mice
    :param df: dataframe of cell activity with one row per cell_specimen_id / ophys_experiment_id
                must include columns 'cell_type', 'experience_level', 'epoch', and a column for the metric provided (ex: 'mean_response')
    :param metric: metric value to average over epochs; must be a column of df
    :param save_dir: top level directory to save figure to
    :param folder: folder within save_dir to save figure to; will create folder if it doesnt exist
    :param suffix: string to append at end of saved filename
    :return:
    """
    suffix = _norm_suffix(suffix)
    # add experience epoch column if it doesnt already exist
    # if 'experience_epoch' not in df.keys():
    # df = annotate_epoch_df(df)

    # experience_levels = utils.get_new_experience_levels()
    # for novel + control
    experience_levels = np.sort(df.experience_level.unique())
    experience_levels = [experience_levels[-1]] + list(experience_levels[:-1])

    df = df[df.epoch <= max_epoch]
    max_n_sessions = len(df.epoch.unique())

    # experience_epoch = np.sort(df[df.experience_level==experience_levels[0]].experience_epoch.unique())
    # experience_epoch = np.sort(df.experience_epoch.unique())
    experience_epoch = np.sort(df.epoch.unique())

    xticks = np.arange(0, len(experience_epoch), 1)
    xticklabels = experience_epoch #np.arange(0, len(experience_epoch), 1)+1
    # xticklabels = [experience_epoch.split(' ')[1] for experience_epoch in experience_epoch]

    if palette is None:
        palette = utils.get_experience_level_colors()

    if ax is None:
        format_fig = True
        if horizontal:
            figsize = (5, 3)
        else:
            figsize = (5, 3)
        fig, ax = plt.subplots(1, 1)
    else:
        format_fig = False

    data = df.copy()
    ax = sns.pointplot(data=data, x='epoch', y=metric, hue='experience_level', hue_order=experience_levels,
                          order=experience_epoch, palette=palette, ax=ax, estimator=estimator)

    if ymin is not None:
        ax.set_ylim(ymin=ymin)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)

    if title is None:
        title = metric.replace('_',' ')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _legend = ax.get_legend()

    if _legend: _legend.remove()
    ax.set_xlim((xticks[0] - 0.5, xticks[-1] + 0.5))
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticklabels+1)*epoch_dur_mins)
    ax.vlines(x=max_n_sessions + 0.5, ymin=0, ymax=1, color='gray', linestyle='--')
    ax.vlines(x=max_n_sessions + max_n_sessions + 1.5, ymin=0, ymax=1, color='gray', linestyle='--')

    if horizontal:
        ax.set_xlabel('Time (mins)')
    else:
        ax.set_xlabel('')
    ax.set_xlabel('Time (mins)')
    ax.tick_params(axis='both', which='major', labelsize=14)

    if legend:
        ax.legend(fontsize='x-small', bbox_to_anchor=(1,1))

    if format_fig:
        if suptitle is not None:
            plt.suptitle(suptitle, x=0.52, y=1.01, fontsize=18)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    if save_dir:
        fig_title = _clean_filename(metric + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))
    return ax
    

def plot_mean_response_by_epoch_for_multiple_conditions(response_df_dict, metric='mean_response', horizontal=True,
                                                        ymin=0, suptitle=None, axes_condition='cell_type', epoch_dur_mins=5, 
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
    suffix = _norm_suffix(suffix)
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

    # xlabel = str(int(60 / n_epochs)) + ' min epoch in session'
    xlabel = 'Time (mins)'
    ax[i].set_xlabel(xlabel)
    ax[i].set_xticklabels(xticks*epoch_dur_mins, fontsize=9)

    if suptitle is None:
        plt.suptitle(metric + ' over time - ' + df_names[0] + ', ' + df_names[1] + ' - ' + suffix, x=0.52, y=1.02,
                     fontsize=16)
    else:
        plt.suptitle(suffix, x=0.52, y=1.02, fontsize=16)
    fig.tight_layout()
    if save_dir:
        fig_title = _clean_filename(metric + '_epochs_' + df_names[0] + '_' + df_names[1] + '_' + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))


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
    suffix = _norm_suffix(suffix)
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
                              hue_order=experience_levels, palette=palette, dodge=0, linestyle='none', ax=ax[i])
        ax[i].set_xticklabels(experience_levels, rotation=90)
        ax[i].set_ylabel('')
        _legend = ax[i].get_legend()

        if _legend: _legend.remove()
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
        fig_title = _clean_filename('fraction_responsive_cells_' + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))
    return ax


def plot_percent_responsive_cells(multi_session_df, responsiveness_threshold=0.1, horizontal=True, ylim=(0, 100), stats_max=80,
                                   ylabel='% responsive', save_dir=None, folder=None, suffix='', ax=None,
                                   use_mlm=True, group_column='mouse_id', event_type='Not specified'):
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
    suffix = _norm_suffix(suffix)
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
            figsize = (8, 2.5)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=True)
        else:
            figsize = (1.5, 8)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)

    combined_stats = pd.DataFrame()
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
        _legend = ax[i].get_legend()
        if _legend:
            _legend.remove()
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('')
        if ylim is None:
            ax[i].set_ylim(0, 100)
        else:
            ax[i].set_ylim(ylim)

        # add stats to plot if only looking at experience levels
        ax[i], panel_stats = add_stats_to_plot_yaxis(data, metric, ax[i], ymax=stats_max,
                                                     use_mlm=use_mlm, group_column=group_column,
                                                     event_type=event_type, cell_type=cell_type)
        combined_stats = pd.concat([combined_stats, panel_stats])

        ax[i].set_xlim((-0.4, 2.4))

    if horizontal:
        ax[0].set_ylabel(ylabel)
    else:
        ax[1].set_ylabel(ylabel)


    if save_dir:
        fig.subplots_adjust(hspace=0.4, wspace=0.2)
        fig_title = _clean_filename('percent_responsive_cells' + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))
        # try:
        print('saving_stats')
        stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
        combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(fig_title + stats_suffix)))
        # save descriptive stats
        cols_to_groupby = ['cell_type', 'experience_level']
        stats = get_descriptive_stats_for_metric(fraction_responsive, metric, cols_to_groupby)
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename(fig_title + '_values.csv')))
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
    suffix = _norm_suffix(suffix)

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
                              hue_order=experience_levels, palette=palette, dodge=0, linestyle='none', ax=ax[i])
        ax[i].set_xticklabels(experience_levels, rotation=45)
        #     ax[i].legend(fontsize='xx-small', title='')
        _legend = ax[i].get_legend()

        if _legend: _legend.remove()
        ax[i].set_title(cell_type)
        ax[i].set_xlabel('')
        if ylim is not None:
            ax[i].set_ylim(ylim)
    if format_fig:
        fig.tight_layout()
        fig_title = _clean_filename(metric + '_across_containers' + suffix)
        plt.suptitle(fig_title, x=0.52, y=1.02, fontsize=16)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))
    return ax


def add_stats_to_plot_for_hues(data, metric, ax, ymax=None, xorder=None, x='experience_level', hue='layer',
                               use_mlm=True, group_column='mouse_id',
                               event_type='Not specified', cell_type='Not specified'):
    """
    add stars to axis indicating statistics across hue values
    x-axis of plots must be experience_levels
    xorder must be a list of values of x in the order that they appear on the plot

    data: metrics dataframe, each row is one cell_specimen_id in a given ophys_experiment
    metric: column in data representing metric values of interest
    column_to_compare: column in data to do stats over (after splitting by x values), such as 'layer' or 'targeted_structure'
    use_mlm: if True (default), use hierarchical mixed linear model with random intercept for group_column;
             falls back automatically to ANOVA/t-test when data is too sparse for MLM.
             If False, use the legacy ANOVA + Tukey HSD path.
    group_column: nesting variable for MLM (e.g., 'mouse_id'). Ignored when use_mlm=False.
    event_type: optional label recorded in the saved stats table.
    """

    # formatting
    scale = 0.05
    fontsize = 15

    ytop = ax.get_ylim()[1]
    y = ytop
    yh = ytop# * (1 + scale)

    stats_table = pd.DataFrame()
    # do hierarchical stats (MLM by default) across hue values within each x value
    for loc, x_value in enumerate(xorder):
        test_data = data[data[x]==x_value]
        hues = test_data[hue].unique()
        if len(hues) >= 2:
            panel_stats = compute_stats(test_data, metric, column_to_compare=hue,
                                        use_mlm=use_mlm, group_column=group_column,
                                        event_type=event_type, cell_type=cell_type)
            # Position `data_subset` in the leading metadata block (right after
            # cell_type). When the call site subsequently inserts `condition`
            # via insert_stats_metadata with the default after='cell_type',
            # condition slots between cell_type and data_subset -- placing
            # data_subset directly after condition in the saved CSV.
            panel_stats = insert_stats_metadata(panel_stats, data_subset=x_value)
            omnibus_pvalue = panel_stats['omnibus_pvalue'].iloc[0] if len(panel_stats) else 1.0
            # gate star drawing on the omnibus, but keep the panel_stats either way
            # so the saved CSV records every comparison that was tested.
            if omnibus_pvalue < 0.05:
                for tindex, row in panel_stats.iterrows():
                    if len(hues) > 2:  # >2 values: use Holm-corrected pairwise reject
                        if row.reject:
                            ax.text(loc, yh, '*', fontsize=fontsize, horizontalalignment='center',
                                    verticalalignment='bottom', color='k')
                    elif len(hues) == 2:  # 2 values: use the omnibus-aligned p
                        if row.one_way_anova_p_val < 0.05:
                            ax.text(loc, yh, '*', fontsize=fontsize, horizontalalignment='center',
                                    verticalalignment='bottom', color='k')
        else:
            # only 1 hue value present at this x — no comparison possible (t-test, ANOVA,
            # and MLM all require >=2 groups). Skip and record nothing for this x.
            print(f"add_stats_to_plot_for_hues: skipping {x}={x_value!r} -- only "
                  f"{len(hues)} hue value(s) present, need >=2 to compare.")
            panel_stats = pd.DataFrame()

        stats_table = pd.concat([stats_table, panel_stats])
    ax.set_ylim(ymax=ytop * (1 + (scale * 5))) # 3 works better for behavior plots

    return ax, stats_table


def add_stats_to_plot_for_hues_along_x(data, metric, ax, yorder=None, y='experience_level', hue='layer',
                                       use_mlm=True, group_column='mouse_id',
                                       event_type='Not specified', cell_type='Not specified'):
    """
    Add significance stars when metric is on the x-axis and categorical groups are on the y-axis.
    Tests are run across hue values within each y category.

    use_mlm: if True (default), use hierarchical mixed linear model; falls back to ANOVA/t-test
             when data is too sparse for MLM. If False, use the legacy ANOVA + Tukey HSD path.
    group_column: nesting variable for MLM (e.g., 'mouse_id').
    event_type: optional label recorded in the saved stats table.
    """

    fontsize = 15
    x_min, x_max = ax.get_xlim()
    x_span = x_max - x_min
    if x_span == 0:
        x_span = 1
    x_star = x_max - (0.03 * x_span)

    stats_table = pd.DataFrame()
    for loc, y_value in enumerate(yorder):
        test_data = data[data[y] == y_value]
        hues = test_data[hue].unique()
        if len(hues) < 2:
            # only 1 hue value present at this y -- no comparison possible (t-test,
            # ANOVA, and MLM all require >=2 groups). Skip and record nothing.
            print(f"add_stats_to_plot_for_hues_along_x: skipping {y}={y_value!r} -- "
                  f"only {len(hues)} hue value(s) present, need >=2 to compare.")
            continue

        panel_stats = compute_stats(test_data, metric, column_to_compare=hue,
                                    use_mlm=use_mlm, group_column=group_column,
                                    event_type=event_type, cell_type=cell_type)
        omnibus_pvalue = panel_stats['omnibus_pvalue'].iloc[0] if len(panel_stats) else 1.0

        has_sig = False
        if omnibus_pvalue < 0.05 and not panel_stats.empty:
            if len(hues) > 2 and 'reject' in panel_stats.columns:
                has_sig = bool(panel_stats['reject'].any())
            elif len(hues) == 2 and 'one_way_anova_p_val' in panel_stats.columns:
                has_sig = bool((panel_stats['one_way_anova_p_val'] < 0.05).any())

        if has_sig:
            ax.text(x_star, loc, '*', fontsize=fontsize, horizontalalignment='center',
                    verticalalignment='center', color='k')

        if not panel_stats.empty:
            panel_stats = panel_stats.copy()
            panel_stats[y] = y_value
            stats_table = pd.concat([stats_table, panel_stats], ignore_index=True)

    return ax, stats_table


def add_stats_to_plot(data, metric, ax, ymax=None, column_to_compare='experience_level',
                      show_ns=False, hue_only=False, behavior=False,
                      use_mlm=True, group_column='mouse_id',
                      event_type='Not specified', cell_type='Not specified'):
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
    use_mlm: if True (default), use hierarchical mixed linear model with random intercept for group_column;
             falls back automatically to ANOVA/t-test when data is too sparse for MLM.
             If False, use the legacy ANOVA + Tukey HSD path.
    group_column: nesting variable for MLM (e.g., 'mouse_id').
    event_type: optional label recorded in the saved stats table.
    """
    # hierarchical stats (MLM by default) across experience levels or cell types
    stats_table = compute_stats(data, metric, column_to_compare,
                                use_mlm=use_mlm, group_column=group_column,
                                event_type=event_type, cell_type=cell_type)
    omnibus_pvalue = stats_table['omnibus_pvalue'].iloc[0] if len(stats_table) else 1.0
    if hue_only: # makes things from -0.25 to 0.25
        stats_table['x1'] = stats_table['x1'] - 1
        stats_table['x2'] = stats_table['x2'] - 1
        stats_table['x1'] = stats_table['x1'] / 4
        stats_table['x2'] = stats_table['x2'] / 4
        dist = 0.25
    else: # x pos is 0, 1, 2
        dist = 1

    scale = 0.05#0.05 # 0.1
    if behavior:
        scale = 0.01
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
    if behavior:
        scale_factor = 5
    two_away = False
    for tindex, row in stats_table.iterrows():
        if (omnibus_pvalue < 0.05) and (row.reject): # if something is significant, add some significance bars
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
                    ax.text(np.mean([row.x1, row.x2]), yh*(1+scale), 'ns', fontsize=fontsize-8, horizontalalignment='center',
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
                    ax.text(np.mean([row.x1, row.x2]), yh*(1+scale), 'ns', fontsize=fontsize-8, horizontalalignment='center',
                            verticalalignment='bottom')
                    top.append(yh)
    # ax.set_ylim(ymax=ytop * (1 + (scale * 7))) # 3 works better for non-behavior plots
    ax.set_ylim(ymax=np.amax(top) * (1 + scale*scale_factor))  # scale factor determined by number of sig points # 3 works better for behavior plots, 2 for regular

    return ax, stats_table


def add_stats_to_plot_yaxis(data, metric, ax, ymax=None, column_to_compare='experience_level', hue_only=False,
                            use_mlm=True, group_column='mouse_id',
                            event_type='Not specified', cell_type='Not specified'):
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
    use_mlm: if True (default), use hierarchical mixed linear model with random intercept for group_column;
             falls back automatically to ANOVA/t-test when data is too sparse for MLM.
             If False, use the legacy ANOVA + Tukey HSD path.
    group_column: nesting variable for MLM (e.g., 'mouse_id').
    event_type: optional label recorded in the saved stats table.
    """
    # do anova across experience levels or cell types followed by post-hoc tukey
    stats_table = compute_stats(data, metric, column_to_compare,
                                use_mlm=use_mlm, group_column=group_column,
                                event_type=event_type, cell_type=cell_type)
    omnibus_pvalue = stats_table['omnibus_pvalue'].iloc[0] if len(stats_table) else 1.0
    if hue_only: # makes things from -0.25 to 0.25
        stats_table['x1'] = stats_table['x1'] - 1
        stats_table['x2'] = stats_table['x2'] - 1
        stats_table['x1'] = stats_table['x1'] / 4
        stats_table['x2'] = stats_table['x2'] / 4
        dist = 0.25
    else: # x pos is 0, 1, 2
        dist = 1

    scale = 0.025 #0.03#0.05 # 0.1
    fontsize = 15
    color = 'k'
    alpha = 1
    label = '*'

    if ymax is None:
        ytop = ax.get_ylim()[1]
    else:
        ytop = ymax
    if ytop > 1:
        second_bar_scale = 6
    else:
        second_bar_scale = 4
    y1 = ytop * (1 + scale)
    y1h = ytop * (1 + scale * 1.5)
    y2 = ytop * (1 + (scale * second_bar_scale))
    y2h = ytop * (1 + (scale * (second_bar_scale+1)))

    top = [ytop]
    for tindex, row in stats_table.iterrows():
        if (omnibus_pvalue < 0.05) and (row.reject): # if something is significant, add some significance bars
            label = '*'
            color = 'k'
            alpha = 1
        else:
            label = ''
            color = 'w'
            alpha = 0
        if np.abs(row.x2 - row.x1) > dist: # if it is a comparison more than 2 x values away, put the significance bar higher
            y = y2
            yh = y2h
        else: # if they are only one apart, put the bar on the lower level
            y = y1
            yh = y1h
        if len(data[column_to_compare].unique())>2: # if more than 2 values, use the multiple corrections result
            if row.reject:
                ax.plot([row.x1+0.1, row.x1+0.1, row.x2-0.1, row.x2-0.1], [y, y, y, y], linestyle='-', color=color, alpha=alpha, clip_on=False)
                ax.text(np.mean([row.x1+scale*3, row.x2+scale*3]), yh+scale, label, fontsize=fontsize, horizontalalignment='center',
                        verticalalignment='center', clip_on=False)
                top.append(yh)
        elif len(data[column_to_compare].unique())==2: #  if there are only two values, use the p-value from anova
            if row.one_way_anova_p_val<0.05:
                ax.plot([row.x1-0.1, row.x1-0.1, row.x2-0.1, row.x2-0.1], [y, y, y, y, ], linestyle='-', color=color, alpha=alpha, clip_on=False)
                ax.text(np.mean([row.x1+0.2, row.x2+0.2]), yh+0.04, label, fontsize=fontsize, horizontalalignment='center',
                        verticalalignment='bottom', clip_on=False)
                top.append(yh)

    return ax, stats_table


def add_stats_to_plot_xaxis(data, metric, ax, xmax=None, column_to_compare='experience_level',
                      show_ns=False, hue_only=False,
                      use_mlm=True, group_column='mouse_id',
                      event_type='Not specified', cell_type='Not specified'):
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
    use_mlm: if True (default), use hierarchical mixed linear model with random intercept for group_column;
             falls back automatically to ANOVA/t-test when data is too sparse for MLM.
             If False, use the legacy ANOVA + Tukey HSD path.
    group_column: nesting variable for MLM (e.g., 'mouse_id').
    event_type: optional label recorded in the saved stats table.
    """
    stats_table = compute_stats(data, metric, column_to_compare,
                                use_mlm=use_mlm, group_column=group_column,
                                event_type=event_type, cell_type=cell_type)
    omnibus_pvalue = stats_table['omnibus_pvalue'].iloc[0] if len(stats_table) else 1.0
    if hue_only: # makes things from -0.25 to 0.25
        stats_table['x1'] = stats_table['x1'] - 1
        stats_table['x2'] = stats_table['x2'] - 1
        stats_table['x1'] = stats_table['x1'] / 4
        stats_table['x2'] = stats_table['x2'] / 4
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
    for tindex, row in stats_table.iterrows():
        if (omnibus_pvalue < 0.05) and (row.reject): # if something is significant, add some significance bars
            label = '*'
            color = 'k'
            alpha = 1
        else:
            # invisible -- mirrors the yaxis/main add_stats_to_plot conventions.
            # Without this else, label/color/alpha could leak from a prior iteration
            # (or be undefined on the first iteration) and produce mis-styled stars.
            label = ''
            color = 'w'
            alpha = 0
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

    return ax, stats_table


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
                                                        title='', ylabel=None, ylims=None, save_dir=None, ax=None, suffix='',
                                                        use_mlm=True, group_column='mouse_id'):
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
    suffix = _norm_suffix(suffix)
    data = metrics_table.copy()
    experience_levels = utils.get_experience_levels()

    if hue:
        if hue == 'targeted_structure':
            hue_order = np.sort(metrics_table[hue].unique())[::-1]
        else:
            hue_order = np.sort(metrics_table[hue].unique())[::-1]
        suffix = '_' + hue + '_' + suffix
    else:
        hue_order = experience_levels
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
            figsize = (2.5, 2.25)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            figsize = (1.75, 2.5)
            fig, ax = plt.subplots(1, 1, figsize=figsize)

    # stats dataframe to save
    combined_stats = pd.DataFrame()
    if hue:
        if pointplot:
            ax = sns.pointplot(data=data, y=y, x=x, order=order, dodge=0.3, linestyle='none',
                               markers='.', markersize=5, err_kws={'linewidth': 2}, hue=hue, hue_order=hue_order, palette='gray', ax=ax)

        else:
            ax = sns.boxplot(data=data, y=y, x=x, order=order, cut=0, notch=True,
                             width=0.4, hue=hue, hue_order=hue_order, palette='gray', ax=ax)
        ax.legend(fontsize='xx-small', title='')  # , loc=loc)  # bbox_to_anchor=(1,1))
            # TBD add area or depth comparison stats / stats across hue variable
    else:
        hue = 'experience_level'
        if show_containers:
            print('table includes', len(data.ophys_container_id.unique()), 'containers')
            for ophys_container_id in data.ophys_container_id.unique():
                ax = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x=x, y=y,
                                   color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax)
        if show_mice:
            print('table includes', len(data.mouse_id.unique()), 'mice')
            for mouse_id in data.mouse_id.unique():
                ax = sns.pointplot(data=data[data.mouse_id == mouse_id], x=x, y=y, order=order,
                                   color='gray', linewidth=0.5, markers='.', markersize=1, err_kws={'linewidth': 0.5}, ax=ax)
            plt.setp(ax.collections, alpha=.7)  # for the markers
            plt.setp(ax.lines, alpha=.7)

        if pointplot:
            # ax = sns.pointplot(data=data, x='experience_level', y=metric,
            #                    palette=colors, ax=ax)
            ax = sns.pointplot(data=data, x=x, y=y, hue=hue, order=order,
                                  hue_order=order, palette=palette, dodge=0, linestyle='None',
                                  markers='.', markersize=8, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax)

        else:
            ax = sns.boxplot(data=data, x=x, y=y, width=0.4, order=order, notch=True,
                             palette=colors, ax=ax)
        if stripplot:
            # add strip plot
            ax = sns.stripplot(data=data, size=3, alpha=0.5, jitter=0.2, order=order,
                               x=x, y=y, color='gray', ax=ax)
        if boxplot:
            ax = sns.boxplot(data=data, x=x, y=y, width=0.4, order=order, notch=True,
                             palette='dark:white', ax=ax)
            # format to have black lines and transparent box face
            plt.setp(ax.artists, edgecolor='k', facecolor=[0, 0, 0, 0])
            plt.setp(ax.lines, color='k')


        _legend = ax.get_legend()
        if _legend:
            _legend.remove()
        ax.set_title(title)

        # if ylims and not horiz:
        #     ax.set_ylim(ylims)
        # elif ylims and horiz:
        #     ax.set_xlim(ylims)
        # ymin, ymax = ax.get_ylim()

        if horiz:
            ax.set_xlim(xmin=ymin)
            ax.set_ylim(-0.5, len(order) - 0.5)
            ax, panel_stats = add_stats_to_plot_yaxis(data, metric, ax, ymax=ymax, hue_only=False,
                                                      use_mlm=use_mlm, group_column=group_column,
                                                      event_type=event_type)
        else:
            ax.set_ylim(ymin=ymin)
            ax.set_xlim(-0.5, len(order) - 0.5)

            # add stats to plot if only looking at experience levels
            ax, panel_stats = add_stats_to_plot(data, metric, ax, ymax=ymax, show_ns=show_ns,
                                                use_mlm=use_mlm, group_column=group_column,
                                                event_type=event_type)
        panel_stats = insert_stats_metadata(panel_stats, condition='experience_level')
        combined_stats = pd.concat([combined_stats, panel_stats])

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
        folder = 'response_metrics'
        filename = _clean_filename(event_type + '_' + data_type + '_' + metric + '_distribution' + suffix)
        stats_filename = _clean_filename(event_type + '_' + data_type + '_' + metric + suffix + '_no_cell_type')
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
            # save descriptive stats
            cols_to_groupby = ['experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric, hue)
    return ax


def plot_metric_distribution_by_experience(metrics_table, metric, event_type, data_type, hue=None,
                                               plot_type='pointplot', legend=True, show_containers=False, estimator=np.mean,
                                               add_zero_line=False, show_ns=False, ylabel=None, ylims=None, horiz=True,
                                               abbreviate_exp=True, suptitle=None, save_dir=None, ax=None, suffix='',
                                               use_mlm=True, group_column='mouse_id'):
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
    suffix = _norm_suffix(suffix)
    data = metrics_table.copy()
    experience_levels = utils.get_experience_levels()

    if hue:
        if hue == 'targeted_structure':
            hue_order = np.sort(metrics_table[hue].unique())[::-1]
        else:
            hue_order = np.sort(metrics_table[hue].unique())
        suffix = '_' + hue + '_' + plot_type + suffix
        hue_colors = sns.color_palette('Greys', len(hue_order))[::-1]
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
            # figsize = (2, 10)
            figsize = (1.5, 8)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)
    # stats dataframe to save
    combined_stats = pd.DataFrame()
    cell_types = utils.get_cell_types()
    for i, cell_type in enumerate(cell_types):
        ct_data = data[data.cell_type == cell_type]

        if show_containers:
            for ophys_container_id in ct_data.ophys_container_id.unique():
                ax[i] = sns.pointplot(data=ct_data[ct_data.ophys_container_id == ophys_container_id], x='experience_level',
                                      y=metric, color='gray',  estimator=estimator,
                                      linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i])

        if hue:
            if plot_type == 'pointplot':
                dodge = 0.1 * len(ct_data[hue].unique())
                ax[i] = sns.pointplot(data=ct_data, y=metric, x='experience_level', order=order, dodge=dodge, linestyle='none',
                                      markers='.', markersize=5, err_kws={'linewidth': 2}, hue=hue, hue_order=hue_order, 
                                      estimator=estimator, palette=hue_colors, ax=ax[i])
            elif plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, y=metric, x='experience_level', order=order, fliersize=0, notch=True,
                                    width=0.4, hue=hue, hue_order=hue_order, palette=hue_colors, ax=ax[i])
                for box in ax[i].collections:
                    box.set_alpha(0.75)
            elif plot_type == 'violinplot':
                if len(ct_data[hue].unique())==2:
                    split = True
                else:
                    split = False
                ax[i] = sns.violinplot(data=ct_data, y=metric, x='experience_level', order=order,
                                       hue=hue, hue_order=hue_order, palette=hue_colors, cut=0, inner=None,
                                       split=split, fill=False, ax=ax[i])
                ax2 = ax[i].twinx()
                ax2 = sns.boxplot(data=ct_data, hue=hue, y=metric, x='experience_level',  order=order, palette='dark:white', notch=True,
                                  hue_order=hue_order, width=0.3, boxprops=dict(alpha=0.8, zorder=2), whis=0, showfliers=False, ax=ax2)
                _legend = ax2.get_legend()

                if _legend: _legend.remove()
                ax2.axis('off')
                ax2.set_ylim(ymin=ymin)
                if ylims is not None:
                    ax2.set_ylim(ylims)
                # for violin in ax[i].collections:
                #     violin.set_alpha(0.5)

            ax[i].set_xlabel('')
            ax[i].set_ylabel(ylabel)
            _legend = ax[i].get_legend()

            if _legend: _legend.remove()
            ax[i].set_ylim(ymin=ymin)
            if ylims is not None:
                ax[i].set_ylim(ylims)

            # ax[i], panel_stats = add_stats_to_plot_for_hues(ct_data, metric, ax[i],
            #                                                 xorder=order, x='experience_level', hue=hue)
            ax[i], panel_stats = add_stats_to_plot(ct_data, metric, ax[i], ymax=ymax,
                                                   use_mlm=use_mlm, group_column=group_column,
                                                   event_type=event_type, cell_type=cell_type)
            panel_stats = insert_stats_metadata(panel_stats, condition='experience_level')
            combined_stats = pd.concat([combined_stats, panel_stats])
        else:
            if plot_type == 'pointplot':
                ax[i] = sns.pointplot(data=ct_data, x='experience_level', y=metric, palette=colors, hue='experience_level',
                                      estimator=estimator, markers='.', markersize=5, err_kws={'linewidth': 2}, ax=ax[i])
            elif plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, x='experience_level', y=metric, width=0.4, hue='experience_level',
                                     notch=True, palette=colors, fliersize=0, ax=ax[i])
                for box in ax[i].collections:
                    box.set_alpha(0.75)
            elif plot_type == 'barplot':
                ax[i] = sns.barplot(data=ct_data, x='experience_level', y=metric, width=0.7, hue='experience_level',
                                     palette=colors, ax=ax[i])
                for bar in ax[i].patches:
                    bar.set_alpha(0.75)
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, y=metric, x='experience_level', order=order, hue='experience_level',
                                       palette=colors,  cut=0, ax=ax[i])
                for violin in ax[i].collections:
                    violin.set_alpha(0.75)

            elif plot_type == 'stripplot':
                ax[i] = sns.boxplot(data=ct_data, x='experience_level', y=metric, width=0.4,
                                    palette='dark:white', ax=ax[i])
                for box in ax[i].collections:
                    box.set_alpha(0.75)
                # format to have black lines and transparent box face
                # plt.setp(ax[i].artists, edgecolor='k', facecolor=[0, 0, 0, 0])
                # plt.setp(ax[i].lines, color='k')
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
            ax[i], panel_stats = add_stats_to_plot(ct_data, metric, ax[i], ymax=ymax, show_ns=show_ns,
                                                   use_mlm=use_mlm, group_column=group_column,
                                                   event_type=event_type, cell_type=cell_type)
            combined_stats = pd.concat([combined_stats, panel_stats])
            # set labels
            ax[i].set_xlabel('')
            if not horiz:
                ax[i].set_ylabel(ylabel)
            else: 
                ax[i].set_ylabel('')

        # add line at y=0
        if add_zero_line:
            ax[i].axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
        # ax[i].set_title(cell_type)
        if data_type not in ['running_speed', 'pupil_width', 'lick_rate']:
            # ax[i].set_title(cell_type+'\n(n = '+str(len(ct_data.cell_specimen_id.unique()))+' cells)', fontsize=16)
            ax[i].set_title('')
        else:
            ax[i].set_title('')

        ax[i].set_xlabel('')

        if abbreviate_exp:
            ax[i].set_xticks(np.arange(0, len(utils.get_abbreviated_experience_levels())))
            ax[i].set_xticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
            utils.color_xaxis_labels_by_experience(ax[i])
        else:
            ax[i].set_xticks(ax[i].get_xticks().tolist())
            ax[i].set_xticklabels(experience_levels, rotation=90,)  # ha='right')
            utils.color_xaxis_labels_by_experience(ax[i])

        if ylabel:
            if not horiz:
                if i == 1: 
                    ax[i].set_ylabel(ylabel+'\n\n'+cell_type)
                else:
                    ax[i].set_ylabel(cell_type)
            else: 
                ax[i].set_ylabel('')
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
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if save_dir:
        folder = 'response_metrics'
        filename = _clean_filename(event_type + '_' + data_type + '_' + metric + '_distribution' + suffix)
        stats_filename = _clean_filename(event_type + '_' + data_type + '_' + metric + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            # save stats: '_mlm.csv' when MLM was used, '_tukey.csv' for the legacy path
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
            # save descriptive stats
            cols_to_groupby = ['cell_type', 'experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
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


def plot_metric_over_repeats(df, metric, x, title='', xlabel=None, ylabel=None, save_dir=None, folder=None, ax=None):
    '''   
    Plot metric value for epochs, stim presentations, or time in session, averaged across mice or sessions
    x value determines bins for averaging (ex: epoch, stimulus presentation number, time bin in session)
    '''
    if ax is None: 
        figsize = (7, 3)
        fig, ax = plt.subplots(figsize=figsize)

    experience_level_colors = utils.get_experience_level_colors()
    experience_levels = utils.get_new_experience_levels()

    ax = sns.pointplot(data=df, x=x, y=metric, hue='experience_level', 
                       linewidth=1, markers='.', markersize=5, err_kws={'linewidth': 1}, estimator=np.mean,
                        palette=experience_level_colors, hue_order=experience_levels, ax=ax)
    ax.legend(bbox_to_anchor=(1,1), fontsize='xx-small', title='')
    ax.set_title(title)
    if xlabel is None: 
        ax.set_xlabel(x.replace('_', ' ').capitalize())
    else: 
        ax.set_xlabel(xlabel)
    if ylabel is None:
        ax.set_ylabel(metric.replace('_', ' ').capitalize())
    else: 
        ax.set_ylabel(ylabel)
    for j, label in enumerate(ax.get_xticklabels()):
        label.set_visible(j % 10 == 0)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metric+'_'+x))
    
    return ax


def plot_metric_over_repeats_for_cell_types(df, metric, x, xlabel=None, ylabel=None, save_dir=None, folder=None):
    figsize = (12, 7)
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
    experience_level_colors = utils.get_experience_level_colors()
    for i, cell_type in enumerate(utils.get_cell_types()):
        cell_type_df = df[df.cell_type==cell_type]
        ax[i] = plot_metric_over_repeats(cell_type_df, metric, x, title='', save_dir=None, folder=None, ax=ax[i])
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].set_title(cell_type)
    if xlabel is None: 
        ax[i].set_xlabel(x.replace('_', ' ').capitalize())
    else: 
        ax[i].set_xlabel(xlabel)
    if ylabel is None:
        ax[1].set_ylabel(metric.replace('_', ' ').capitalize())
    else: 
        ax[1].set_ylabel(ylabel)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metric+'_'+x+'_cell_types'))



def plot_metric_across_stimuli_by_experience_level(stimulus_response_df, metric,
                                                   time_window_min=None, interval=20, show_x_in_minutes=False,
                                                   ylim=None, ylabel=None, title=None, 
                                                   ax=None, save_dir=None, folder=None):
    
    '''
    Take the metric value across stimulus presentations for each session, 
    then average across mice within experience levels and plot 
    Does not group by or average over time bins.
    xticklabels are converted into time based on duration of stim presentations
    
    time_window_min: how many minutes of the session to include in the plot. If None, will include all stimulus presentations.
    interval: how many stimulus presentations to include between xticks. Default is 20, which corresponds to 15 seconds (20*0.75s)
    '''
    df = stimulus_response_df.copy()
    if time_window_min is None:
        time_window_min = 2
    max_stim = (time_window_min*60)/0.75
    df = df[df.stimulus_number<=max_stim+1] 
    
    if ax is None:
        figsize = (7, 3)
        fig, ax = plt.subplots(figsize=figsize)

    experience_levels = utils.get_experience_levels()
    colors = utils.get_experience_level_colors()
    
    for i, experience_level in enumerate(experience_levels):
        subset = df[(df.experience_level == experience_level) ]
        pivot = subset.pivot_table(index=['behavior_session_id'], 
                columns=['stimulus_presentations_id'], 
                values=metric)
        traces = pivot.values
        timestamps = pivot.columns.values 

        trace = np.nanmean(traces, axis=0)
        sem = (np.nanstd(traces, axis=0)) / np.sqrt(float(len(traces)))
        color = colors[i]
        ax.plot(timestamps, trace, label=experience_level, linewidth=1, color=color)
        ax.fill_between(timestamps, trace + sem, trace - sem, alpha=0.4, color=color)
        ax.set_xticks(np.arange(0, max_stim+1, interval))
        if show_x_in_minutes:
            xticklabels = [str(int(x*0.75/60)) for x in ax.get_xticks()]
            ax.set_xlabel('Time (min)')
        else:
            xticklabels = [str(int(x*0.75)) for x in ax.get_xticks()]
            ax.set_xlabel('Time (sec)')
        ax.set_xticklabels(xticklabels)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlim(0, max_stim)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        sns.despine(ax=ax)


    if save_dir is not None and folder is not None:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metric+'_over_time_in_session'))
    
    return ax


def plot_modulation_index_distribution(metrics_table, metric, x_axis_col=None, x_axis_label=None,
                                       label=None, lims=(-1.1, 1.1), horiz=False, plot_type='violinplot',
                                       metric_on_y=True, annot=('left', 'right'), abbreviate_exp=True, suptitle=None,
                                       fill=True, save_dir=None, suffix='', ax=None,
                                       use_mlm=True, group_column='mouse_id', event_type='Not specified'):
    '''
    Plots distribution of metric values split by experience level.

    When x_axis_col is provided:
        horiz=True (default): metric on y-axis, x_axis_col on x-axis, columns = cell types
        horiz=False: metric on x-axis, x_axis_col on y-axis, rows = cell types
            (flipped orientation — useful for compact vertical layouts)

    When x_axis_col is not provided:
        horiz=True: metric on x-axis, experience levels on y-axis, columns = cell types
        horiz=False: metric on x-axis, experience levels on y-axis, rows = cell types
    '''
    suffix = _norm_suffix(suffix)

    data = metrics_table.copy()

    colors = utils.get_experience_level_colors()
    experience_levels = utils.get_experience_levels()

    if x_axis_col:
        if x_axis_col == 'targeted_structure':
            ax_size = 3
            order = np.sort(data[x_axis_col].unique())[::-1]
        else:
            ax_size = 3
            order = np.sort(data[x_axis_col].unique())
        suffix = '_' + x_axis_col + '_' + suffix
        wspace = 0.1
    else:
        suffix = '_experience_level' + suffix
        wspace = 0.3
        order = experience_levels


    if ax is None:
        if horiz:
            if x_axis_col:
                if metric_on_y: 
                    figsize = (ax_size * len(order), 2.5)
                else: 
                    figsize = (ax_size * len(order), len(order))
            else:
                figsize = (8, 2.5)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
        elif not horiz:
            if x_axis_col:
                if metric_on_y: 
                    figsize = (len(order), 8)
                else: 
                    figsize = (2, len(order)*3)
            else:
                figsize = (2, 8)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
        else:
            figsize = (2, 8)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
        ax = ax.ravel()

    # stats dataframe to save
    combined_stats = pd.DataFrame()
    cell_types = utils.get_cell_types()
    for i, cell_type in enumerate(cell_types):
        ct_data = data[data.cell_type == cell_type]

        if x_axis_col:
            if metric_on_y:
                y = metric
                x = x_axis_col
                orient = 'v'
                if 'index' in metric:
                    ax[i].axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
            else: 
                y = x_axis_col
                x = metric
                orient = 'h'
                if 'index' in metric:
                    ax[i].axvline(x=0, ymin=0, ymax=1, color='gray', linestyle='--')

            if plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, x=x, y=y, orient=orient, boxprops=dict(alpha=0.75),
                                    hue='experience_level', hue_order=experience_levels, 
                                    order=order, palette=colors, ax=ax[i], width=0.6, fliersize=0, notch=True)
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, x=x, y=y, orient=orient,
                                hue='experience_level', hue_order=experience_levels,
                                order=order, palette=colors, ax=ax[i], alpha=0.5, fill=fill, linewidth=1, gap=0.1, cut=0,
                                inner='box', inner_kws=dict(box_width=2, whis_width=1, color="k", alpha=0.75))
                ax[i] = sns.pointplot(data=ct_data, x=x, y=y, orient=orient,
                                    hue='experience_level', hue_order=experience_levels, linestyle='none', dodge=0.55,
                                    order=order, color='k', ax=ax[i], zorder=10000,
                                    markers='_', markersize=10, err_kws={'linewidth': 2})
            
            _legend = ax[i].get_legend()
            if _legend: _legend.remove()
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')

            if not metric_on_y:
                ax[i].set_xlim(lims)
                ax[i], panel_stats = add_stats_to_plot_for_hues_along_x(ct_data, metric, ax[i],
                                                            yorder=order, y=x_axis_col, hue='experience_level',
                                                            use_mlm=use_mlm, group_column=group_column,
                                                            event_type=event_type, cell_type=cell_type)
            else:
                ax[i].set_ylim(lims)
                ax[i], panel_stats = add_stats_to_plot_for_hues(ct_data, metric, ax[i],
                                                            xorder=order, x=x_axis_col, hue='experience_level',
                                                            use_mlm=use_mlm, group_column=group_column,
                                                            event_type=event_type, cell_type=cell_type)
            panel_stats = insert_stats_metadata(panel_stats, condition=x_axis_col)
            combined_stats = pd.concat([combined_stats, panel_stats])

        else:
            if metric_on_y: 
                y = metric
                x = 'experience_level'
                if 'index' in metric:
                    ax[i].axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
            else: 
                y = 'experience_level'
                x = metric
                if 'index' in metric:
                    ax[i].axvline(x=0, ymin=0, ymax=1, color='gray', linestyle='--')

            if plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, y=y, x=x, order=experience_levels, boxprops=dict(alpha=0.75),
                                    hue='experience_level', hue_order=experience_levels, legend=False,
                                    palette=colors, ax=ax[i], width=0.6, fliersize=0, notch=True)
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, y=y, x=x, order=experience_levels,
                                        hue='experience_level', hue_order=experience_levels, legend=False,
                                        palette=colors, ax=ax[i], alpha=0.75, fill=fill, linewidth=1, gap=0.1, cut=0,
                                        inner='box', inner_kws=dict(box_width=2, whis_width=1, color="k", alpha=1))
                ax[i] = sns.pointplot(data=ct_data, y=y, x=x, order=experience_levels,
                                        hue='experience_level', hue_order=experience_levels, legend=False,
                                        color='k', ax=ax[i], zorder=10000, linestyle='none',
                                        markers='_', markersize=10, err_kws={'linewidth': 2})            
            if metric_on_y:
                if abbreviate_exp:
                    ax[i].set_xticks(np.arange(0, len(experience_levels)))
                    ax[i].set_xticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
                    utils.color_xaxis_labels_by_experience(ax[i])
                ax[i].set_ylim(lims)
                ax[i], panel_stats = add_stats_to_plot_yaxis(ct_data, metric, ax[i], ymax=lims[1], column_to_compare='experience_level',
                                                             use_mlm=use_mlm, group_column=group_column,
                                                             event_type=event_type, cell_type=cell_type)
                ymin, ymax = ax[i].get_ylim()
                ax[i].set_ylim(ymax=ymax*1.3)
            else:
                if abbreviate_exp:
                    ax[i].set_yticks(np.arange(0, len(experience_levels)))
                    ax[i].set_yticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
                    utils.color_yaxis_labels_by_experience(ax[i])
                ax[i].set_xlim(lims)
                ax[i], panel_stats = add_stats_to_plot_xaxis(ct_data, metric, ax[i], xmax=lims[1], column_to_compare='experience_level',
                                                             use_mlm=use_mlm, group_column=group_column,
                                                             event_type=event_type, cell_type=cell_type)

            panel_stats = insert_stats_metadata(panel_stats, condition='experience_level')
            combined_stats = pd.concat([combined_stats, panel_stats])

        ax[i].set_ylabel('')
        ax[i].set_xlabel('')
        ax[i].set_title(cell_type)
        plt.setp(ax[i].collections, alpha=0.7)

  
    if not horiz: 
        if metric_on_y:
            ax[0].annotate(annot[1], xy=(-0.5, 1.1), xycoords=ax[0].get_xaxis_transform(), ha="right", va="center",
                        fontsize=10)
            ax[2].annotate(annot[0], xy=(-0.5, -0.1), xycoords=ax[2].get_xaxis_transform(), ha="right", va="center",
                        fontsize=10)
            ax[1].set_ylabel(label)
        else:
            ax[2].annotate(annot[0], xy=(lims[0]-0.05, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="right", va="center",
                        fontsize=10)
            ax[2].annotate(annot[1], xy=(lims[1]+0.05, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="left", va="center",
                        fontsize=10)
            ax[2].set_xlabel(label)
    else: 
        if metric_on_y:
            ax[0].annotate(annot[1], xy=(-0.5, 1.1), xycoords=ax[0].get_xaxis_transform(), ha="right", va="center",
                        fontsize=10)
            ax[0].annotate(annot[0], xy=(-0.5, -0.1), xycoords=ax[0].get_xaxis_transform(), ha="right", va="center",
                        fontsize=10)
            ax[0].set_ylabel(label)
        else:
            ax[2].annotate(annot[1], xy=(1.1, -0.1), xycoords=ax[2].get_xaxis_transform(), ha="left", va="center",
                        fontsize=10)
            ax[0].annotate(annot[0], xy=(-1.1, -0.1), xycoords=ax[0].get_xaxis_transform(), ha="right", va="center",
                        fontsize=10)
            ax[1].set_xlabel(label)


    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=0.98, fontsize=16)

    fig.subplots_adjust(hspace=0.3, wspace=wspace)

    if save_dir:
        if horiz:
            suffix = suffix + 'horiz'
        if metric_on_y:
            suffix = suffix + 'yaxis'
        folder = 'response_metrics'
        filename = _clean_filename(metric + '_distribution' + suffix)
        stats_filename = _clean_filename(metric + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
            cols_to_groupby = ['cell_type', 'experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric)
    return ax


def plot_metric_across_cohorts(metrics_table, metric,  ylabel, x_val='binned_depth', plot_type='barplot',
                               event_type='Not specified', save_dir=None, folder=None, ax=None,
                               use_mlm=True, group_column='mouse_id'):
    '''
    Plot metric distributions across cre lines, with a unique axis for each cohort / project code, 
    experience levels as colors, and x-axis defined by x_val (such as 'binned_depth' or 'targeted_structure').
    Will plot stats across exp levels as an asterisk above that x value
    '''

    mdf = metrics_table.copy()
    project_codes = mdf.project_code.unique()
    palette = utils.get_experience_level_colors()
    experience_levels = utils.get_experience_levels()
    cell_types = utils.get_cell_types()

    # get width ratios based on how many x values there are for each condition
    width_ratios = []
    # for c, cell_type in enumerate(cell_types): 
    cell_type = 'Excitatory'
    ct_data = mdf[mdf.cell_type==cell_type]
    for p, project_code in enumerate(project_codes): 
        data = ct_data[(ct_data.project_code==project_code)]
        x_vals = np.sort(data[x_val].unique())
        width_ratios.append(len(x_vals))

    if x_val == 'binned_depth':
        width = 8
    else:
        width = 7
    i = 0 
    if ax is None:
        figsize=(width, 8)
        fig, ax = plt.subplots(3, 3, figsize=figsize, gridspec_kw={'width_ratios':width_ratios})
        ax = ax.ravel()

    combined_stats = pd.DataFrame()
    for c, cell_type in enumerate(cell_types):
        ct_data = mdf[mdf.cell_type==cell_type]
        for p, project_code in enumerate(project_codes):
            data = ct_data[(ct_data.project_code==project_code)]
            x_vals = np.sort(data[x_val].unique())
            if plot_type == 'pointplot': 
                ax[i] = sns.pointplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals,
                                            hue_order=experience_levels, palette=palette, dodge=0.3, linestyle='none',
                                            markers='.', markersize=8, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i])
            elif plot_type == 'barplot': 
                ax[i] = sns.barplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals, width=0.5, alpha=0.75, 
                                                hue_order=experience_levels, palette=palette, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i])
            elif plot_type == 'boxplot': 
                ax[i] = sns.boxplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals, boxprops=dict(alpha=0.75),
                                                hue_order=experience_levels, palette=palette, notch=True,
                                                width=0.5, fliersize=0, ax=ax[i])
                plt.setp(ax[i].collections, alpha=0.75)
            elif plot_type == 'violinplot': 
                ax[i] = sns.violinplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals, 
                                                hue_order=experience_levels, palette=palette, 
                                                width=0.5, fliersize=0, ax=ax[i])
                plt.setp(ax[i].collections, alpha=0.75)
            ax[i].set_ylabel('')
            ax[i].set_xlabel('')
            ax[i].get_legend().remove()
            # label = ax[i].get_yticklabels()
            # label.set_fontsize(8)
            ax[i].tick_params(axis='y', which='major', labelsize=12)
            if c == 0: 
                ax[i].set_title('Cohort '+str(p+1))
            if p == 0: 
                if c == 1: 
                    ax[i].set_ylabel(ylabel+'\n\n'+cell_type)
                else: 
                    ax[i].set_ylabel(cell_type)
            # ax[i], combined_stats = ppf.add_stats_to_plot(data, metric, ax[i])
            ax[i], panel_stats = add_stats_to_plot_for_hues(data, metric, ax[i], event_type=event_type,
                                                            xorder=x_vals, x=x_val, hue='experience_level',
                                                            use_mlm=use_mlm, group_column=group_column,
                                                            cell_type=cell_type)
            # cohort first so it lands before condition; that keeps data_subset
            # (added inside add_stats_to_plot_for_hues) directly after condition.
            panel_stats = insert_stats_metadata(panel_stats, cohort=project_code, condition=x_val)
            combined_stats = pd.concat([combined_stats, panel_stats])
            i+=1
    plt.subplots_adjust(wspace=0.5, hspace=0.35)

    if save_dir:
        if folder is None: 
            folder = 'response_metrics'
        filename = _clean_filename(metric+'_by_cohort_x_'+x_val)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)))
            cols_to_groupby = ['cell_type', 'experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric)
    return ax


def plot_metric_across_cohorts_area_depth(metrics_table, metric,  ylabel, plot_type='barplot',
                               event_type='Not specified', save_dir=None, folder=None, ax=None,
                               use_mlm=True, group_column='mouse_id'):
    '''
    Plot metric distributions across cre lines, with a unique axis for each cohort / project code, experience levels as colors.
    Will first plot values with binned_depth on the x-axis for the first 2 cohorts, then for cohort 3, will plot data split by 'binned_depth' and 'targeted_structure'
    Will plot stats across exp levels as an asterisk above that x value
    '''

    mdf = metrics_table.copy()
    project_codes = mdf.project_code.unique()

    cell_types = utils.get_cell_types()
    experience_levels = utils.get_experience_levels()
    palette = utils.get_experience_level_colors()

    # get width ratios based on how many x values there are for each condition
    width_ratios = []
    for c, cell_type in enumerate(cell_types):
        ct_data = mdf[mdf.cell_type==cell_type]
        for p, project_code in enumerate(project_codes):
            data = ct_data[(ct_data.project_code==project_code)]
            if project_code == 'VisualBehaviorMultiscope':
                x_values = ['binned_depth', 'targeted_structure']
            else:
                x_values = ['binned_depth']
            for x_val in x_values:
                x_vals = np.sort(data[x_val].unique())
                width_ratios.append(len(x_vals))
    if np.sum(width_ratios) > 15:
        fig_width = np.sum(width_ratios)*1.5
    else:
        fig_width = np.sum(width_ratios)*2

    i = 0
    if ax is None:
        figsize=(fig_width, 2)
        fig, ax = plt.subplots(1, 12, figsize=figsize, gridspec_kw={'width_ratios':width_ratios})

    combined_stats = pd.DataFrame()
    for c, cell_type in enumerate(cell_types):
        ct_data = mdf[mdf.cell_type==cell_type]
        for p, project_code in enumerate(project_codes):
            data = ct_data[(ct_data.project_code==project_code)]
            if project_code == 'VisualBehaviorMultiscope':
                x_values = ['binned_depth', 'targeted_structure']
            else:
                x_values = ['binned_depth']
            for x_val in x_values:
                x_vals = np.sort(data[x_val].unique())
                if plot_type == 'pointplot':
                    ax[i] = sns.pointplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals,
                                                hue_order=experience_levels, palette=palette, dodge=0.3, linestyle='none',
                                                markers='.', markersize=8, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i])
                elif plot_type == 'barplot':
                    ax[i] = sns.barplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals, width=0.5, alpha=0.75, 
                                                    hue_order=experience_levels, palette=palette, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i])
                elif plot_type == 'boxplot':
                    ax[i] = sns.boxplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals,
                                                    hue_order=experience_levels, palette=palette,
                                                    width=0.5, fliersize=0, ax=ax[i])
                    plt.setp(ax[i].collections, alpha=0.75)
                elif plot_type == 'violinplot':
                    ax[i] = sns.violinplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals,
                                        hue_order=experience_levels, palette=palette, alpha=0.75, cut=0, width=0.75,
                                        fill=False, linewidth=1.5, gap=0.1, inner='box',
                                        inner_kws=dict(box_width=2, whis_width=1, color="gray", alpha=0.75), ax=ax[i])
                    ax[i] = sns.pointplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals,
                                          hue_order=experience_levels, palette=palette, dodge=0.5, linestyle='none',
                                          markers='.', markersize=5, err_kws={'linewidth': 2}, errorbar=('ci', 95),
                                         zorder=10000, ax=ax[i])

                ax[i].set_ylabel('')
                ax[i].set_xlabel('')
                _legend = ax[i].get_legend()

                if _legend: _legend.remove()
                ax[i].set_title('Cohort '+str(p+1))
                # ax[i], combined_stats = ppf.add_stats_to_plot(data, metric, ax[i])
                ax[i], panel_stats = add_stats_to_plot_for_hues(data, metric, ax[i], event_type=event_type,
                                                                xorder=x_vals, x=x_val, hue='experience_level',
                                                                use_mlm=use_mlm, group_column=group_column,
                                                                cell_type=cell_type)
                # cohort first so it lands before condition; that keeps data_subset
                # (added inside add_stats_to_plot_for_hues) directly after condition.
                panel_stats = insert_stats_metadata(panel_stats, cohort=project_code, condition=x_val)
                combined_stats = pd.concat([combined_stats, panel_stats])
                # , ymax=None, show_ns=False)
                i+=1
    ax[0].set_ylabel(ylabel)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    if save_dir:
        if folder is None: 
            folder = 'response_metrics'
        filename = _clean_filename(metric+'_by_cohort_depth_area')
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)))
            # save descriptive stats
            cols_to_groupby = ['cell_type', 'experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric, 'experience_level')
    return ax


def plot_metric_across_conditions(metrics_table, metric,  title='', xlabel='Imaging depth (um)', x_color='k',
                              x_val='binned_depth', hue='experience_level', plot_type='barplot',
                               event_type='Not specified', save_dir=None, folder=None, ax=None,
                               use_mlm=True, group_column='mouse_id'):
    '''
    Plot metric distributions across cre lines, with a unique axis for each cohort / project code, 
    experience levels as colors, and x-axis defined by x_val (such as 'binned_depth' or 'targeted_structure'), 
    or experience_levels as x-values and hue defined by something else (such as 'targeted_structure').
    Will plot stats across hues as an asterisk above that x value
    '''

    mdf = metrics_table.copy()
    cell_types = utils.get_cell_types()
    experience_levels = utils.get_experience_levels()
    experience_level_colors = utils.get_experience_level_colors()

    if hue == 'experience_level':
        hue_order = experience_levels
        palette = utils.get_experience_level_colors()
    else: 
        hues = np.sort(mdf[hue].unique())
        palette = sns.color_palette('Greys', len(hues)+1)
        if hue == 'targeted_structure':
            hues = hues[::-1]
        palette = palette[1:]
        hue_order = hues
    if x_val == 'experience_level': 
        x_vals = experience_levels
    else:
        x_vals = np.sort(mdf[x_val].unique())

    i = 0 
    if ax is None:
        figsize=(len(x_vals), 8)
        fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
        ax = ax.ravel()
    else: 
        fig = None

    combined_stats = pd.DataFrame()
    for c, cell_type in enumerate(cell_types):
        data = mdf[mdf.cell_type==cell_type]
        if plot_type == 'pointplot':
            ax[i] = sns.pointplot(data=data, x=x_val, y=metric, hue=hue, order=x_vals,
                                        hue_order=hue_order, palette=palette, dodge=0.3, linestyle='none',
                                        markers='.', markersize=8, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i])
        elif plot_type == 'barplot': 
            ax[i] = sns.barplot(data=data, x=x_val, y=metric, hue=hue, order=x_vals, width=0.6, alpha=0.75, 
                                            hue_order=hue_order, palette=palette, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i])
        elif plot_type == 'boxplot': 
            ax[i] = sns.boxplot(data=data, x=x_val, y=metric, hue=hue, order=x_vals, 
                                            hue_order=hue_order, palette=palette, 
                                            width=0.5, fliersize=0, ax=ax[i])
            plt.setp(ax[i].collections, alpha=0.75)
        ax[i].set_ylabel('')
        ax[i].set_xlabel('')
        if hue == 'experience_level':
            ax[i].get_legend().remove()
            if c == 2: 
                ax[i].set_xlabel(xlabel, fontsize=12)
        else: 
            ax[i].get_legend().remove()
            ax[0].legend(bbox_to_anchor=(1,1), fontsize='xx-small', title=xlabel, title_fontsize='xx-small')
        if x_val == 'experience_level': 
            ax[i].set_xticks(np.arange(0, len(experience_levels)))
            ax[i].set_xticklabels(experience_levels)#, color=experience_level_colors)
            for xtick, color in zip(ax[i].get_xticklabels(), experience_level_colors):
                xtick.set_color(color)
        ax[i].tick_params(axis='x', which='major', labelsize=12)
        ax[i].tick_params(axis='y', which='major', labelsize=12)
        if c == 0: 
            ax[i].set_title(title, color=x_color)

        if c == 1: 
            ax[i].set_ylabel(cell_type+'\nCoding score')
        else: 
            ax[i].set_ylabel(cell_type+'\n')

        ax[i], panel_stats = add_stats_to_plot_for_hues(data, metric, ax[i], event_type=event_type,
                                                        xorder=x_vals, x=x_val, hue=hue,
                                                        use_mlm=use_mlm, group_column=group_column,
                                                        cell_type=cell_type)
        panel_stats = insert_stats_metadata(panel_stats, condition=x_val)
        combined_stats = pd.concat([combined_stats, panel_stats])
        # , ymax=None, show_ns=False)
        i+=1
    plt.subplots_adjust(wspace=0.5, hspace=0.3)

    # save stats
    filename = _clean_filename(metric+'_across_'+hue+'_for_'+x_val+'_'+plot_type)
    if save_dir:
        if folder is None: 
            folder = 'response_metrics'
        stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
        combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)))
        cols_to_groupby = ['cell_type', hue, x_val]
        stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + '_values.csv')))

    # save fig
    if save_dir and fig: 
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
    return ax


def plot_experience_modulation_index(metric_data, event_type, hue=None, plot_type='pointplot', ylims=(-1, 1),
                                     suptitle=None, suffix='', include_all_comparisons=True, save_dir=None,
                                     use_mlm=True, group_column='mouse_id'):
    """
    plots experience modulation for some event_type, which is the mean repsonse in familiar vs. novel 1 over the sum,
    and the mean response in novel 1 vs. novel >1 over the sum, giving a value between -1 and 1
    metric_data is the output of visual_behavior.ophys.response_analysis.cell_metrics.compute_experience_modulation_index()

    :param metric_data:
    :param event_type:
    :param save_dir:
    :return:
    """
    suffix = _norm_suffix(suffix)

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
        cols_to_group = ['cell_specimen_id', 'cell_type', 'mouse_id', hue]
        data = data.melt(id_vars=cols_to_group, var_name='comparison', value_vars=value_vars)
    else:
        cols_to_group = ['cell_specimen_id', 'cell_type', 'mouse_id']
        data = data.melt(id_vars=cols_to_group, var_name='comparison', value_vars=value_vars)

    # Rename the melted value column so the saved stats CSV records a meaningful
    # metric name (instead of literally 'value'). This replaces the previous
    # post-hoc panel_stats['metric'] = 'experience_modulation' relabel.
    data = data.rename(columns={'value': 'experience_modulation'})
    metric = 'experience_modulation'
    x = 'comparison'

    cell_types = np.sort(data.cell_type.unique())

    # colors = utils.get_experience_level_colors()
    combined_stats = pd.DataFrame()
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
                ax[i] = sns.pointplot(data=ct_data, order=xorder, linestyle='none', hue=hue, hue_order=hue_order, dodge=dodge,
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
                _legend = ax2.get_legend()

                if _legend: _legend.remove()
                ax2.axis('off')
                ax2.set_ylim(ylims)

            _legend = ax[i].get_legend()


            if _legend: _legend.remove()
            ax[i].set_ylim(ylims)

            # add stats to plot for hues
            ax[i], panel_stats = add_stats_to_plot_for_hues(ct_data, metric, ax[i],
                                                            xorder=xorder, x=x, hue=hue,
                                                            use_mlm=use_mlm, group_column=group_column,
                                                            event_type=event_type, cell_type=cell_type)
            # ax[i], panel_stats = add_stats_to_plot(ct_data, metric, ax[i], ymax=ymax)
            combined_stats = pd.concat([combined_stats, panel_stats])
        else:
            if plot_type == 'pointplot':
                ax[i] = sns.pointplot(data=ct_data, order=xorder, linestyle='none',
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
        if folder is None: 
            folder = 'response_metrics'
        filename = _clean_filename('experience_modulation_' + event_type + '_' + plot_type + suffix)
        utils.save_figure(fig, figsize, save_dir, 'response_metrics', _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            combined_stats.to_csv(os.path.join(save_dir, 'response_metrics', _clean_filename(filename + stats_suffix)))
            # save descriptive stats
            cols_to_groupby = ['cell_type', 'experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, 'response_metrics', _clean_filename(filename + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric, hue)


def plot_experience_modulation_index_annotated(metrics_table, event_type, metric, cells_table,
                                               horiz=False, xlims=(-1.1, 1.1), xlabel='Experience modulation',
                                               suptitle=None, suffix='', save_dir=None, ax=None,
                                               use_mlm=True, group_column='mouse_id'):
    """
    plots experience modulation for some event_type, which is the mean repsonse in familiar vs. novel 1 over the sum,
    and the mean response in novel 1 vs. novel >1 over the sum, giving a value between -1 and 1
    metric_data is the output of visual_behavior.ophys.response_analysis.cell_metrics.compute_experience_modulation_index()

    :param metric_data:
    :param event_type:
    :param save_dir:
    :return:
    """
    suffix = _norm_suffix(suffix)

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

    # Rename the melted value column so the saved stats CSV records a meaningful
    # metric name (instead of literally 'value'). Preserves the caller-supplied
    # `metric` arg in the saved filename via the prefix.
    data = data.rename(columns={'value': metric + '_experience_modulation'})
    metric = metric + '_experience_modulation'
    x = 'comparison'

    cell_types = np.sort(data.cell_type.unique())

    colors = utils.get_experience_level_colors()
    if ax is None:
        if horiz:
            figsize = (8, 2)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=True)
        else:
            figsize = (2, 8)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)

    combined_stats = pd.DataFrame()
    for i, comparison in enumerate(value_vars):
        ax[i] = sns.violinplot(data=data[data.comparison == comparison], x=metric, y='cell_type', order=cell_types,
                               color='gray', cut=0, inner='box', ax=ax[i], alpha=0.25, linewidth=1,
                                inner_kws=dict(box_width=2, whis_width=1, color="k", alpha=0.75))

        ax[i] = sns.pointplot(data=data[data.comparison == comparison], x=metric, y='cell_type', order=cell_types,
                                    color='k', ax=ax[i], zorder=10000, linestyle='none',
                                    markers='|', markersize=15, err_kws={'linewidth': 2})
  
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
        ax[i], panel_stats = add_stats_to_plot_xaxis(data[data.comparison == comparison], metric, ax[i],
                               xmax=xlims[1],
                               column_to_compare='cell_type',
                               use_mlm=use_mlm, group_column=group_column,
                               event_type=event_type)
        combined_stats = pd.concat([combined_stats, panel_stats])

    if horiz:

        ax[1].set_xlabel(xlabel)
    else:
        # ax[2].annotate(annot[0], xy=(-1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="right", va="center", fontsize=10)
        # ax[2].annotate(annot[1], xy=(1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="left", va="center", fontsize=10)
        ax[2].set_xlabel(xlabel)

    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=0.98, fontsize=16)

    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    if save_dir:
        if horiz:
            suffix = suffix + '_horiz'
        folder = 'response_metrics'
        filename = _clean_filename('experience_modulation_annot_' + event_type + '_' + suffix)
        stats_filename = _clean_filename('experience_modulation_' + event_type + '_' + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
            # save descriptive stats
            cols_to_groupby = ['cell_type']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric)
    return ax


def plot_experience_modulation_index_annotated_by_cell_type(metrics_table, event_type, metric, cells_table,
                                                            xlims=(-1.1, 1.1), xlabel='Experience modulation',
                                                            all_comparisons=True, horiz=False,
                                                            suptitle=None, suffix='', save_dir=None, ax=None,
                                                            use_mlm=True, group_column='mouse_id'):
    """
    plots experience modulation for some event_type, which is the mean repsonse in familiar vs. novel 1 over the sum,
    and the mean response in novel 1 vs. novel >1 over the sum, giving a value between -1 and 1
    metric_data is the output of visual_behavior.ophys.response_analysis.cell_metrics.compute_experience_modulation_index()

    :param metric_data:
    :param event_type:
    :param save_dir:
    :return:
    """
    suffix = _norm_suffix(suffix)

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

    # Rename the melted value column so the saved stats CSV records a meaningful
    # metric name (instead of literally 'value'). Preserves the caller-supplied
    # `metric` arg in the saved filename via the prefix.
    data = data.rename(columns={'value': metric + '_experience_modulation'})
    metric = metric + '_experience_modulation'
    x = 'comparison'

    cell_types = np.sort(data.cell_type.unique())
    colors = utils.get_experience_level_colors()

    wspace = 0.5
    if ax is None:
        if horiz:
            if 'all_comparisons':
                figsize = (8, 2)
                wspace = 0.6
            else:
                figsize = (8, 2)
            fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=True)
        else:
            figsize = (2, 8)
            fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)

    combined_stats = pd.DataFrame()
    for i, cell_type in enumerate(cell_types):
        ax[i] = sns.violinplot(data=data[data.cell_type == cell_type],
                               x=metric, y='comparison', order=value_vars,
                               color='gray', cut=0, ax=ax[i], linewidth=1, alpha=0.25, inner='box',
                               inner_kws=dict(box_width=2, whis_width=1, color="k", alpha=0.75))
        ax[i] = sns.pointplot(data=data[data.cell_type == cell_type],
                               x=metric, y='comparison', order=value_vars,
                                color='k', ax=ax[i], zorder=100000, linestyle='none',
                                markers='|', markersize=15, err_kws={'linewidth': 2})
        
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
            ax[i], panel_stats = add_stats_to_plot_xaxis(data[data.cell_type == cell_type], metric, ax[i],
                                                         xmax=xlims[1],
                                                         column_to_compare='comparison',
                                                         use_mlm=use_mlm, group_column=group_column,
                                                         event_type=event_type, cell_type=cell_type)
            combined_stats = pd.concat([combined_stats, panel_stats])

        ax[i].invert_yaxis()

    if horiz:
        ax[1].set_xlabel(xlabel)
    else:
        # ax[2].annotate(annot[0], xy=(-1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="right", va="center", fontsize=10)
        # ax[2].annotate(annot[1], xy=(1.2, -0.05), xycoords=ax[2].get_xaxis_transform(), ha="left", va="center", fontsize=10)
        ax[2].set_xlabel(xlabel)

    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=0.98, fontsize=16)

    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    if save_dir:
        if horiz:
            suffix = suffix + '_horiz'
        folder = 'response_metrics'
        filename = _clean_filename('experience_modulation_annot_by_cell_type_' + event_type + '_' + suffix)
        stats_filename = _clean_filename('experience_modulation_by_cell_type_' + event_type + '_' + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
            # save descriptive stats
            cols_to_groupby = ['cell_type']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric)
    return ax



def plot_experience_modulation_index_depth_heatmap_by_cell_type(metrics_table, event_type, metric, cells_table,
                                                                all_comparisons=True, groupby_col='binned_depth',
                                                                ylabel=None, vmin=-0.5, vmax=0.5,
                                                                suptitle=None, suffix='', save_dir=None):
    """
    One heatmap per cell type. Rows = bins of `groupby_col`, columns = experience comparison, cell color =
    mean experience modulation index. Each column uses its own diverging colormap whose endpoints come from
    utils.get_experience_level_colors() — positive values use the color of the "novel-side" experience
    level in the comparison, negative values use the "familiar-side" color, white at 0.
    Cells annotated with mean and significance stars (* p<0.05) from a one-sample t-test against 0.

    groupby_col: column to bin rows by. Defaults to 'binned_depth' (75/175/275/375 um). Pass any column on
        cells_table — e.g. 'layer', 'targeted_structure', 'area_binned_depth'.
    ylabel: optional y-axis label. Defaults to groupby_col with underscores replaced by spaces.
    """
    suffix = _norm_suffix(suffix)
    import visual_behavior.ophys.response_analysis.cell_metrics as cm
    from scipy import stats as sstats
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib import gridspec

    exp_mod = cm.compute_experience_modulation_index_new(metrics_table, metric, cells_table)
    exp_mod = exp_mod.drop_duplicates(subset='cell_specimen_id')

    if groupby_col not in exp_mod.columns:
        group_map = cells_table.drop_duplicates('cell_specimen_id').set_index('cell_specimen_id')[groupby_col]
        exp_mod[groupby_col] = exp_mod['cell_specimen_id'].map(group_map)
    exp_mod = exp_mod.dropna(subset=[groupby_col])

    if all_comparisons:
        value_vars = ['F N', 'N+ N', 'F N+']
        comparison_labels = ['Novel\nvs. Familiar', 'Novel\nvs. Novel +', 'Novel +\nvs. Familiar']
    else:
        value_vars = ['F N', 'N+ N']
        comparison_labels = ['Novel\nvs. Familiar', 'Novel\nvs. Novel +']

    # build per-column diverging cmaps from experience-level colors.
    # get_experience_level_colors() returns [Familiar=blue, Novel=red, Novel+=purple].
    # modulation index sign convention: positive -> 2nd token of key, negative -> 1st token.
    # 'F N'  -> +N(red),    -F(blue);  'N+ N' -> +N(red), -N+(purple);  'F N+' -> +N+(purple), -F(blue)
    exp_colors = utils.get_experience_level_colors()
    color_for = {'F': exp_colors[0], 'N': exp_colors[1], 'N+': exp_colors[2]}
    cmaps = {}
    for comp in value_vars:
        neg_label, pos_label = comp.split(' ')
        cmaps[comp] = LinearSegmentedColormap.from_list(
            f'expmod_{comp.replace(" ", "_")}',
            [color_for[neg_label], (1.0, 1.0, 1.0), color_for[pos_label]])

    data = exp_mod.melt(id_vars=['cell_specimen_id', 'cell_type', groupby_col],
                        var_name='comparison', value_vars=value_vars).dropna(subset=['value'])

    cell_types = np.sort(data.cell_type.unique())
    group_order = sorted(data[groupby_col].dropna().unique())
    nrows = len(group_order)
    ncols = len(value_vars)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    if ylabel is None:
        ylabel = groupby_col.replace('_', ' ')

    # layout: top row = one heatmap-ax per cell_type (each drawn column-by-column with its own cmap);
    # bottom row = one shared colorbar per comparison spanning across the figure.
    figsize = (3.0 * len(cell_types), 0.55 * nrows + 2.2)
    fig = plt.figure(figsize=figsize)
    # 3-row gridspec: heatmap row, empty spacer row (constant size, ensures the colorbar always sits a
    # consistent distance below the heatmap regardless of nrows), then the colorbar row.
    gs = gridspec.GridSpec(3, len(cell_types), height_ratios=[nrows, 2.0, 0.18 * nrows],
                           hspace=0.0, wspace=0.3)
    heat_axes = [fig.add_subplot(gs[0, i]) for i in range(len(cell_types))]
    cbar_gs = gridspec.GridSpecFromSubplotSpec(1, ncols, subplot_spec=gs[2, :], wspace=0.6)
    cbar_axes = [fig.add_subplot(cbar_gs[0, j]) for j in range(ncols)]

    stats_table = pd.DataFrame()
    for i, cell_type in enumerate(cell_types):
        sub = data[data.cell_type == cell_type]
        ax_i = heat_axes[i]
        for c, comparison in enumerate(value_vars):
            col_vals = np.full((nrows, 1), np.nan)
            for r, d in enumerate(group_order):
                vals = sub.loc[(sub[groupby_col] == d) & (sub.comparison == comparison), 'value'].dropna().values
                if len(vals) < 3:
                    continue
                m = float(np.mean(vals))
                t, p = sstats.ttest_1samp(vals, 0.0)
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                col_vals[r, 0] = m
                # text color: dark on light cells, light on saturated cells
                text_color = 'black' if abs(m) < 0.35 else 'white'
                if stars:
                    ax_i.text(c + 0.5, r + 0.32, stars,
                              ha='center', va='center', fontsize=16, fontweight='bold',
                              color=text_color)
                ax_i.text(c + 0.5, r + 0.55, f'{m:.2f}',
                          ha='center', va='center', fontsize=11, color=text_color)
                ax_i.text(c + 0.5, r + 0.82, f'n={len(vals)}',
                          ha='center', va='center', fontsize=8, color=text_color)
                stats_table = pd.concat([stats_table, pd.DataFrame([{
                    'cell_type': cell_type, 'comparison': comparison, groupby_col: d,
                    'mean': m, 't': t, 'p': p, 'n': len(vals)}])])
            ax_i.imshow(col_vals, cmap=cmaps[comparison], norm=norm, aspect='auto',
                        extent=(c, c + 1, nrows, 0), interpolation='nearest')

        # cell borders
        for r in range(nrows + 1):
            ax_i.axhline(r, color='white', linewidth=1)
        for c in range(ncols + 1):
            ax_i.axvline(c, color='white', linewidth=1)

        ax_i.set_xlim(0, ncols)
        ax_i.set_ylim(nrows, 0)
        ax_i.set_xticks(np.arange(ncols) + 0.5)
        ax_i.set_xticklabels(comparison_labels, fontsize=12, rotation=45, ha='right')
        ax_i.set_yticks(np.arange(nrows) + 0.5)
        ax_i.set_yticklabels([str(d) for d in group_order] if i == 0 else [''] * nrows, fontsize=12)
        ax_i.tick_params(left=(i == 0), bottom=False)
        ax_i.set_title(cell_type, fontsize=14)
        if i == 0:
            ax_i.set_ylabel(ylabel, fontsize=13)
        for side in ('top', 'bottom', 'left', 'right'):
            ax_i.spines[side].set_visible(True)
            ax_i.spines[side].set_color('black')
            ax_i.spines[side].set_linewidth(1.0)

    # per-comparison colorbars along the bottom (shared across cell types).
    # label the positive end with the "novel-side" experience level and the negative end with the
    # "familiar-side" one so the sign convention is clear without reading the cmap.
    full_name = {'F': 'Familiar', 'N': 'Novel', 'N+': 'Novel +'}
    for j, comparison in enumerate(value_vars):
        sm = plt.cm.ScalarMappable(cmap=cmaps[comparison], norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_axes[j], orientation='horizontal')
        cb.ax.tick_params(labelsize=10)
        if j == len(value_vars) // 2:
            cb.set_label('Experience modulation', fontsize=13)
        neg_token, pos_token = comparison.split(' ')
        cb.ax.text(-0.04, 0.5, full_name[neg_token], ha='right', va='center',
                   transform=cb.ax.transAxes, fontsize=11, rotation=90)
        cb.ax.text(1.04, 0.5, full_name[pos_token], ha='left', va='center',
                   transform=cb.ax.transAxes, fontsize=11, rotation=90)

    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=1.0, fontsize=15)

    if save_dir:
        folder = 'response_metrics'
        filename = _clean_filename('experience_modulation_heatmap_by_cell_type_' + groupby_col + '_' + event_type + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + '_ttest.csv')), index=False)
        except BaseException:
            print('STATS TABLE DID NOT SAVE FOR', metric)
    return heat_axes, stats_table


def plot_experience_modulation_index_depth_heatmap_by_comparison(metrics_table, event_type, metric, cells_table,
                                                                 all_comparisons=True, groupby_col='binned_depth',
                                                                 ylabel=None, vmin=-0.5, vmax=0.5,
                                                                 suptitle=None, suffix='', save_dir=None):
    """
    Variant of plot_experience_modulation_index_depth_heatmap_by_cell_type that transposes the layout:
    one subplot per experience-level comparison, with cell types along the x-axis (abbreviated to the
    first 3 letters) and a grouping variable on the y-axis. Each subplot uses its own diverging colormap
    built from utils.get_experience_level_colors() and has its own colorbar shared across cell types.
    Cells annotated with significance stars on top of the mean (* p<0.05, one-sample t-test against 0)
    and group n on the bottom.

    groupby_col: column to bin rows by. Defaults to 'binned_depth' (75/175/275/375 um). Pass any column
        present on cells_table — e.g. 'layer', 'targeted_structure', 'area_binned_depth'.
    ylabel: optional y-axis label. Defaults to groupby_col with underscores replaced by spaces.
    """
    suffix = _norm_suffix(suffix)
    import visual_behavior.ophys.response_analysis.cell_metrics as cm
    from scipy import stats as sstats
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    exp_mod = cm.compute_experience_modulation_index_new(metrics_table, metric, cells_table)
    exp_mod = exp_mod.drop_duplicates(subset='cell_specimen_id')

    if groupby_col not in exp_mod.columns:
        group_map = cells_table.drop_duplicates('cell_specimen_id').set_index('cell_specimen_id')[groupby_col]
        exp_mod[groupby_col] = exp_mod['cell_specimen_id'].map(group_map)
    exp_mod = exp_mod.dropna(subset=[groupby_col])

    if all_comparisons:
        value_vars = ['F N', 'N+ N', 'F N+']
        comparison_labels = ['Novel vs. Familiar', 'Novel vs. Novel +', 'Novel + vs. Familiar']
    else:
        value_vars = ['F N', 'N+ N']
        comparison_labels = ['Novel vs. Familiar', 'Novel vs. Novel +']

    # per-comparison diverging cmaps (same convention as the by_cell_type version).
    # [Familiar=blue, Novel=red, Novel+=purple]; positive -> 2nd token, negative -> 1st token of key.
    exp_colors = utils.get_experience_level_colors()
    color_for = {'F': exp_colors[0], 'N': exp_colors[1], 'N+': exp_colors[2]}
    cmaps = {}
    for comp in value_vars:
        neg_label, pos_label = comp.split(' ')
        cmaps[comp] = LinearSegmentedColormap.from_list(
            f'expmod_{comp.replace(" ", "_")}',
            [color_for[neg_label], (1.0, 1.0, 1.0), color_for[pos_label]])

    data = exp_mod.melt(id_vars=['cell_specimen_id', 'cell_type', groupby_col],
                        var_name='comparison', value_vars=value_vars).dropna(subset=['value'])

    cell_types = list(np.sort(data.cell_type.unique()))
    cell_type_labels = [str(ct)[:3] for ct in cell_types]
    group_order = sorted(data[groupby_col].dropna().unique())
    nrows = len(group_order)
    ncols = len(cell_types)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    if ylabel is None:
        ylabel = groupby_col.replace('_', ' ')

    # full experience-level names per token (for colorbar top/bottom labels)
    full_name = {'F': 'Familiar', 'N': 'Novel', 'N+': 'Novel +'}

    figsize = (3.4 * len(value_vars) + 1.6, 0.55 * nrows + 1.2)
    fig, axes = plt.subplots(1, len(value_vars), figsize=figsize,
                             gridspec_kw={'wspace': 0.35})
    if len(value_vars) == 1:
        axes = [axes]

    stats_table = pd.DataFrame()
    for j, comparison in enumerate(value_vars):
        ax_j = axes[j]
        mat = np.full((nrows, ncols), np.nan)
        sub = data[data.comparison == comparison]
        for r, d in enumerate(group_order):
            for c, ct in enumerate(cell_types):
                vals = sub.loc[(sub[groupby_col] == d) & (sub.cell_type == ct), 'value'].dropna().values
                if len(vals) < 3:
                    continue
                m = float(np.mean(vals))
                t, p = sstats.ttest_1samp(vals, 0.0)
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                mat[r, c] = m
                text_color = 'black' if abs(m) < 0.35 else 'white'
                if stars:
                    ax_j.text(c + 0.5, r + 0.32, stars,
                              ha='center', va='center', fontsize=16, fontweight='bold',
                              color=text_color)
                ax_j.text(c + 0.5, r + 0.55, f'{m:.2f}',
                          ha='center', va='center', fontsize=11, color=text_color)
                ax_j.text(c + 0.5, r + 0.82, f'n={len(vals)}',
                          ha='center', va='center', fontsize=8, color=text_color)
                stats_table = pd.concat([stats_table, pd.DataFrame([{
                    'comparison': comparison, 'cell_type': ct, groupby_col: d,
                    'mean': m, 't': t, 'p': p, 'n': len(vals)}])])

        im = ax_j.imshow(mat, cmap=cmaps[comparison], norm=norm, aspect='auto',
                         extent=(0, ncols, nrows, 0), interpolation='nearest')

        for r in range(nrows + 1):
            ax_j.axhline(r, color='white', linewidth=1)
        for c in range(ncols + 1):
            ax_j.axvline(c, color='white', linewidth=1)

        ax_j.set_xlim(0, ncols)
        ax_j.set_ylim(nrows, 0)
        ax_j.set_xticks(np.arange(ncols) + 0.5)
        ax_j.set_xticklabels(cell_type_labels, fontsize=12)
        ax_j.set_yticks(np.arange(nrows) + 0.5)
        ax_j.set_yticklabels([str(d) for d in group_order], fontsize=12)
        ax_j.tick_params(left=True, bottom=False, labelleft=True)
        ax_j.set_title(comparison_labels[j], fontsize=13, pad=8)
        if j == 0:
            ax_j.set_ylabel(ylabel, fontsize=13)
        for side in ('top', 'bottom', 'left', 'right'):
            ax_j.spines[side].set_visible(True)
            ax_j.spines[side].set_color('black')
            ax_j.spines[side].set_linewidth(1.0)

        # duplicate cell-type labels along the top of each subplot, beneath the title
        ax_top = ax_j.secondary_xaxis('top')
        ax_top.set_xticks(np.arange(ncols) + 0.5)
        ax_top.set_xticklabels(cell_type_labels, fontsize=12)
        ax_top.tick_params(top=False, labeltop=True)

        # colorbar: numeric ticks at vmin/0/vmax, with the comparison's "positive" experience level
        # labeled at the top of the bar and the "negative" one at the bottom
        neg_token, pos_token = comparison.split(' ')
        cb = fig.colorbar(im, ax=ax_j, orientation='vertical', fraction=0.13, pad=0.14,
                          ticks=[vmin, 0, vmax])
        cb.ax.tick_params(labelsize=10)
        if j == len(value_vars) - 1:
            cb.set_label('Experience modulation', fontsize=13)
        cb.ax.set_title(full_name[pos_token], fontsize=11, pad=6)
        cb.ax.set_xlabel(full_name[neg_token], fontsize=11, labelpad=6)

    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=1.02, fontsize=15)
    fig.tight_layout()

    if save_dir:
        folder = 'response_metrics'
        filename = _clean_filename('experience_modulation_heatmap_by_comparison_' + groupby_col + '_' + event_type + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + '_ttest.csv')), index=False)
        except BaseException:
            print('STATS TABLE DID NOT SAVE FOR', metric)
    return axes, stats_table


def plot_metric_heatmap_grid_by_cell_type_and_metric(
        results_pivoted, metric_cols, metric_labels=None,
        groupby_col='binned_depth', exp_col='experience_level',
        ylabel=None, vmax=None, multi_star=False, horiz=False,
        suptitle=None, suffix='', save_dir=None, folder=None,
        use_mlm=True, group_column='mouse_id', event_type='Not specified'):
    """
    Grid of heatmaps: rows = cell types (utils.get_cell_types()), columns = metric_cols (any numeric
    columns of results_pivoted — e.g. coding score columns 'all-images', 'omissions', 'task',
    'behavioral'). Each heatmap has rows = bins of `groupby_col` (e.g. binned_depth, targeted_structure)
    and columns = experience levels (Familiar, Novel, Novel +).

    Each experience-level column in every heatmap uses its own sequential colormap going from white
    to that experience level's color (utils.get_experience_level_colors()), so the color *family*
    encodes experience level and color *intensity* encodes metric value. Each heatmap is normalized
    independently from 0 to its own panel max (override with `vmax` to share a scale).

    Bottom of the figure has a 3-row legend with one mini-colorbar per experience level so it's
    visually clear the three colormaps mean the same quantity at different intensities.

    metric_cols: list of column names in results_pivoted (one heatmap column per metric).
    metric_labels: optional list of subplot titles for each metric column. Defaults to metric_cols.
    groupby_col: column to bin heatmap rows by. Defaults to 'binned_depth'.
    ylabel: optional label for the y-axis (defaults to groupby_col with underscores replaced).
    vmax: shared color limit. If None (default), each metric column shares a vmax across all cell
        type rows (so panels are directly comparable down a column). The per-column max is printed
        below the subplot in gray. Pass a scalar to force a single vmax across the whole figure.
    multi_star: if True, annotate with tiered significance (* p<0.05, ** p<0.01, *** p<0.001).
        Default False — single * for p<0.05 only (matches the convention used in other plots).
    horiz: if True, transposes the outer grid so there's one row per metric (with metric labels on
        the outside-left, colored) and one column per cell type (cell-type names as column titles).
        Colorbars move to the right of each metric row. Default False keeps the original layout
        (cell-types as rows, metrics as columns, cbars at the bottom).
    """
    suffix = _norm_suffix(suffix)
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import gridspec
    from scipy import stats as sstats

    cell_types = utils.get_cell_types()
    exp_levels = utils.get_experience_levels()
    exp_colors = utils.get_experience_level_colors()
    exp_abbrev = utils.get_abbreviated_experience_levels()

    # try to look up feature/coding-score colors so subplot titles can be colored by metric
    try:
        from visual_behavior.dimensionality_reduction.clustering import plotting as cluster_plotting
        feat_colors, feat_labels_dict = cluster_plotting.get_feature_colors_and_labels()
        feature_color_map = {k: feat_colors[ix] for ix, k in enumerate(feat_labels_dict.keys())}
    except Exception:
        feature_color_map = {}

    # white -> experience-level color sequential cmap per experience level
    cmaps = {exp_levels[k]: LinearSegmentedColormap.from_list(
        f'coding_{exp_levels[k].replace(" ", "_").replace("+", "p")}',
        [(1.0, 1.0, 1.0), exp_colors[k]]) for k in range(len(exp_levels))}

    if metric_labels is None:
        metric_labels = list(metric_cols)
    if ylabel is None:
        ylabel = groupby_col.replace('_', ' ').capitalize()

    group_order = sorted(results_pivoted[groupby_col].dropna().unique())
    nrows = len(group_order)
    ncols = len(exp_levels)
    n_cell_types = len(cell_types)
    n_metrics = len(metric_cols)

    # compute per-(cell_type, metric) aggregated matrix, per-panel max, and per-cell stats
    # (one-sample t-test vs 0 across the underlying rows of results_pivoted)
    # tier_stars: turn an ANOVA p-value into a star string honoring `multi_star`
    def tier_stars(p):
        if p >= 0.05 or not np.isfinite(p):
            return ''
        if multi_star:
            return '***' if p < 0.001 else '**' if p < 0.01 else '*'
        return '*'

    mats = {}
    panel_max = {}
    cell_n = {}            # (i, j, r, c) -> n cells contributing to that bin
    col_stars = {}         # (i, j, c) -> star string for the experience-level column
    row_stars = {}         # (i, j, r) -> star string for the groupby_col row
    pairwise_tables = []   # full pairwise output from each compute_stats call, tagged
                           # with the slice it came from; concat'd at the end.
    for i, cell_type in enumerate(cell_types):
        ct_data = results_pivoted[results_pivoted.cell_type == cell_type]
        for j, metric in enumerate(metric_cols):
            agg = (ct_data.groupby([groupby_col, exp_col])[metric].mean()
                   .unstack(exp_col))
            agg = agg.reindex(index=group_order, columns=exp_levels)
            mat = agg.values.astype(float)
            mats[(i, j)] = mat
            finite = mat[np.isfinite(mat)]
            panel_max[(i, j)] = float(finite.max()) if finite.size else 0.0
            # per-cell n (for in-cell annotation)
            for r, d in enumerate(group_order):
                for c, exp in enumerate(exp_levels):
                    vals = ct_data.loc[(ct_data[groupby_col] == d) & (ct_data[exp_col] == exp),
                                       metric].dropna().values
                    cell_n[(i, j, r, c)] = int(len(vals))

            # column-direction stats: within each experience level, hierarchical test
            # (MLM by default) across groupby_col (e.g., "do depths differ for this
            # experience level?")
            for c, exp in enumerate(exp_levels):
                col_subset = ct_data[ct_data[exp_col] == exp].dropna(subset=[metric, groupby_col])
                panel_stats = compute_stats(col_subset, metric, column_to_compare=groupby_col,
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type=event_type, cell_type=cell_type)
                col_stars[(i, j, c)] = tier_stars(_heatmap_omnibus_p(panel_stats))
                # Tag the full pairwise table with which slice this came from.
                # `held_fixed_col` + `held_fixed_value` are generic columns
                # (vs. dynamic-named columns) so every row has values populated
                # in the same place regardless of direction.
                panel_stats = insert_stats_metadata(panel_stats,
                                                   direction='across_groups_within_exp',
                                                   held_fixed_col=exp_col,
                                                   held_fixed_value=exp)
                pairwise_tables.append(panel_stats)

            # row-direction stats: within each groupby_col value, hierarchical test
            # (MLM by default) across experience levels (e.g., "do experience levels
            # differ at this depth?")
            for r, d in enumerate(group_order):
                row_subset = ct_data[ct_data[groupby_col] == d].dropna(subset=[metric, exp_col])
                panel_stats = compute_stats(row_subset, metric, column_to_compare=exp_col,
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type=event_type, cell_type=cell_type)
                row_stars[(i, j, r)] = tier_stars(_heatmap_omnibus_p(panel_stats))
                panel_stats = insert_stats_metadata(panel_stats,
                                                   direction='across_exp_within_group',
                                                   held_fixed_col=groupby_col,
                                                   held_fixed_value=d)
                pairwise_tables.append(panel_stats)
    stats_table = (pd.concat(pairwise_tables, ignore_index=True)
                   if pairwise_tables else pd.DataFrame())

    # cbar label per metric: "Coding score" if "coding" appears in the metric label,
    # otherwise use the metric_label itself.
    cbar_labels = ['Coding score' if 'coding' in str(metric_labels[j]).lower()
                   else str(metric_labels[j]) for j in range(n_metrics)]

    # right-side cbars used for any non-horiz mode (single or multi metric) — vert always now
    # gets one single colorbar set on the right rather than per-column at the bottom.
    right_side_cbar = not horiz

    # vmax handling: each metric column gets its own vmax (max across cell-type rows for that
    # metric). The single right-side cbar is a generic intensity indicator (no numeric labels)
    # so per-column scaling is fine — actual values are read from the cell annotations.
    if vmax is None:
        column_max = {j: max(panel_max[(i, j)] for i in range(n_cell_types))
                      for j in range(n_metrics)}
        norm_per_column = True
    else:
        column_max = {j: float(vmax) for j in range(n_metrics)}
        norm_per_column = False

    # hspace is a fraction of average row height; matplotlib row height scales with nrows here,
    # so we invert the relationship so absolute gap is roughly constant across nrows variants.
    cell_type_hspace = 1.4 / max(nrows, 1)

    if horiz:
        # rows = metrics, cols = cell types. Per-metric-row cbars on the right.
        fig_w = 2.2 * n_cell_types + 2.1
        # Per-row term (0.22 * nrows * n_metrics) scales with the number of y-axis values; the
        # second term is a fixed base for titles/labels. The base is smaller for the low-row
        # case (visual area) so it doesn't look proportionally too tall, while leaving the depth
        # version unchanged.
        fig_h = 0.22 * nrows * n_metrics + (0.55 if nrows <= 2 else 1.0)
        figsize = (fig_w, fig_h)
        fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(1, 2, width_ratios=[1.6, 0.5], wspace=0.4)
        main_gs = gridspec.GridSpecFromSubplotSpec(n_metrics, n_cell_types, subplot_spec=outer[0],
                                                    hspace=cell_type_hspace, wspace=0.3)
    elif n_metrics == 1:
        # vert single-metric: one set of right-side cbars centered on the middle cell-type row
        fig_w = 3.6
        fig_h = 0.45 * nrows * n_cell_types + 1.6
        figsize = (fig_w, fig_h)
        fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1.0], wspace=0.45)
        main_gs = gridspec.GridSpecFromSubplotSpec(n_cell_types, n_metrics, subplot_spec=outer[0],
                                                    hspace=cell_type_hspace, wspace=0.0)
    else:
        # vert multi-metric: heatmap grid on the left + single right-side cbar set
        fig_w = 1.8 * n_metrics + 2.3
        fig_h = 0.45 * nrows * n_cell_types + 1.4
        figsize = (fig_w, fig_h)
        fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(1, 2, width_ratios=[1.5 * n_metrics, 1.2], wspace=0.4)
        main_gs = gridspec.GridSpecFromSubplotSpec(n_cell_types, n_metrics, subplot_spec=outer[0],
                                                    hspace=cell_type_hspace, wspace=0.3)
    # ensure enough left margin for the outside label + yticklabels. Multi-metric vert is
    # wider so it needs more absolute margin to keep cell-type labels off the ylabel.
    if horiz:
        fig.subplots_adjust(left=max(0.10, 1.10 / fig_w))
    elif n_metrics == 1:
        fig.subplots_adjust(left=max(0.10, 1.10 / fig_w))
    else:
        fig.subplots_adjust(left=max(0.10, 1.40 / fig_w))

    heat_axes = np.empty((n_cell_types, n_metrics), dtype=object)
    for i, cell_type in enumerate(cell_types):
        for j, metric in enumerate(metric_cols):
            if horiz:
                ax = fig.add_subplot(main_gs[j, i])
            else:
                ax = fig.add_subplot(main_gs[i, j])
            heat_axes[i, j] = ax
            mat = mats[(i, j)]
            this_vmax = column_max[j]
            if not this_vmax or this_vmax <= 0:
                this_vmax = 1.0

            for c, exp in enumerate(exp_levels):
                col_vals = mat[:, c:c + 1]
                ax.imshow(col_vals, cmap=cmaps[exp], vmin=0, vmax=this_vmax,
                          aspect='auto', extent=(c, c + 1, nrows, 0), interpolation='nearest')
                for r in range(nrows):
                    val = mat[r, c]
                    if not np.isfinite(val):
                        continue
                    frac = val / this_vmax if this_vmax > 0 else 0.0
                    text_color = 'black' if frac < 0.6 else 'white'
                    ax.text(c + 0.5, r + 0.42, f'{val:.2f}',
                            ha='center', va='center', fontsize=8, color=text_color)
                    n = cell_n.get((i, j, r, c), 0)
                    if n:
                        ax.text(c + 0.5, r + 0.72, f'n={n}',
                                ha='center', va='center', fontsize=6, color=text_color)

            # column-direction significance stars (just above each experience-level column).
            # Use a blended transform so x is in data coords but y is in axes fraction — this
            # keeps the stars the same display distance from the top edge regardless of nrows.
            from matplotlib.transforms import blended_transform_factory
            col_trans = blended_transform_factory(ax.transData, ax.transAxes)
            row_trans = blended_transform_factory(ax.transAxes, ax.transData)
            for c in range(ncols):
                s = col_stars.get((i, j, c), '')
                if s:
                    ax.text(c + 0.5, 1.0, s, transform=col_trans,
                            ha='center', va='center',
                            fontsize=12, fontweight='bold', color='k', clip_on=False)
            # row-direction significance stars (just to the right of each row).
            for r in range(nrows):
                s = row_stars.get((i, j, r), '')
                if s:
                    ax.text(1.02, r + 0.5, s, transform=row_trans,
                            ha='left', va='center',
                            fontsize=12, fontweight='bold', color='k', clip_on=False)

            for r in range(nrows + 1):
                ax.axhline(r, color='white', linewidth=0.7)
            for cc in range(ncols + 1):
                ax.axvline(cc, color='white', linewidth=0.7)

            # which grid edges this subplot sits on (depends on outer orientation)
            if horiz:
                is_top_row = (j == 0)
                is_leftmost = (i == 0)
                is_bottom_row = (j == n_metrics - 1)
                is_middle_outer = (j == n_metrics // 2)
            else:
                is_top_row = (i == 0)
                is_leftmost = (j == 0)
                is_bottom_row = (i == n_cell_types - 1)
                is_middle_outer = (i == n_cell_types // 2)

            ax.set_xlim(0, ncols)
            ax.set_ylim(nrows, 0)
            ax.set_xticks(np.arange(ncols) + 0.5)
            if is_bottom_row:
                ax.set_xticklabels(exp_abbrev, fontsize=10)
                for lbl, color in zip(ax.get_xticklabels(), exp_colors):
                    lbl.set_color(color)
            else:
                ax.set_xticklabels([''] * ncols)
            ax.tick_params(bottom=False)

            ax.set_yticks(np.arange(nrows) + 0.5)
            if is_leftmost:
                ax.set_yticklabels([str(d) for d in group_order], fontsize=9)
                # groupby label (e.g. "Binned depth") inside, next to yticks on the middle row
                # of the outer grid only
                if is_middle_outer:
                    ax.set_ylabel(ylabel, fontsize=11)
            else:
                ax.set_yticklabels([''] * nrows)
                ax.tick_params(labelleft=False)
            ax.tick_params(left=is_leftmost)

            # Title on the top row of the outer grid:
            #   vert: metric name (colored by feature)
            #   horiz: cell-type name
            if is_top_row:
                if horiz:
                    ax.set_title(cell_type, fontsize=12, pad=10)
                else:
                    title_color = feature_color_map.get(metric, 'black')
                    ax.set_title(metric_labels[j], fontsize=12, color=title_color, pad=10)

            for side in ('top', 'bottom', 'left', 'right'):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color('black')
                ax.spines[side].set_linewidth(0.8)

    if right_side_cbar:
        # right-side cbars are added later via fig.add_axes(), after heat_axes positions are
        # finalized (so we can size them relative to the middle panel)
        pass
    else:
        # per-column legend at the bottom: 3 stacked mini-cbars under each metric column
        legend_outer_gs = gridspec.GridSpecFromSubplotSpec(1, n_metrics, subplot_spec=outer[2],
                                                           wspace=0.2)
        for j, metric in enumerate(metric_cols):
            col_vmax = column_max[j] if column_max[j] > 0 else 1.0
            col_legend_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=legend_outer_gs[0, j],
                                                             hspace=0.5, wspace=0.0,
                                                             width_ratios=[0.15, 1.0, 0.15])
            for k, exp in enumerate(exp_levels):
                lax = fig.add_subplot(col_legend_gs[k, 1])
                gradient = np.linspace(0.0, col_vmax, 200).reshape(1, -1)
                lax.imshow(gradient, cmap=cmaps[exp], aspect='auto',
                           vmin=0.0, vmax=col_vmax, extent=(0.0, col_vmax, 0, 1))
                lax.set_yticks([])
                if j == 0:
                    lax.set_ylabel(exp_levels[k], rotation=0, ha='right', va='center', fontsize=9,
                                   color=exp_colors[k], labelpad=6)
                if k < 2:
                    lax.set_xticks([])
                else:
                    lax.set_xticks([0.0, col_vmax])
                    lax.set_xticklabels(['0', f'{col_vmax:.2f}'], fontsize=9)
                    lax.tick_params(labelsize=9)
                    lax.set_xlabel(cbar_labels[j], fontsize=10)
                for side in ('top', 'bottom', 'left', 'right'):
                    lax.spines[side].set_visible(True)
                    lax.spines[side].set_color('black')
                    lax.spines[side].set_linewidth(0.5)

    # Outside-left rotated labels, one per row of the outer grid.
    #   vert:  one cell-type name per row (cell types as rows)
    #   horiz: one metric label per row (metrics as rows), colored by feature
    # offset is computed in inches (then converted to figure fraction) so the gap between the
    # label and the yticklabels stays consistent across narrow vs. wide figures.
    fig.canvas.draw()
    abbreviate_cell_type = nrows <= 2
    # multi-metric vert figures are wider, so the cell-type label needs a larger absolute
    # inch offset to stay clear of the ylabel.
    if horiz:
        ct_offset_inches = 0.55
    elif right_side_cbar and n_metrics == 1:
        ct_offset_inches = 0.85
    else:
        ct_offset_inches = 0.80
    if horiz:
        # row j of outer grid = metric j; leftmost subplot is heat_axes[0, j]
        for j_metric, metric in enumerate(metric_cols):
            bbox = heat_axes[0, j_metric].get_position()
            label = metric_labels[j_metric]
            color = feature_color_map.get(metric, 'black')
            x_pos = max(bbox.x0 - ct_offset_inches / fig_w, 0.005)
            fig.text(x_pos,
                     (bbox.y0 + bbox.y1) / 2,
                     label, rotation=90, ha='left', va='center', fontsize=12, color=color)
    else:
        # row i of outer grid = cell type i; leftmost subplot is heat_axes[i, 0]
        for i_ct, cell_type in enumerate(cell_types):
            bbox = heat_axes[i_ct, 0].get_position()
            label = cell_type[:3] if abbreviate_cell_type else cell_type
            x_pos = max(bbox.x0 - ct_offset_inches / fig_w, 0.005)
            fig.text(x_pos,
                     (bbox.y0 + bbox.y1) / 2,
                     label, rotation=90, ha='left', va='center', fontsize=12)

    if right_side_cbar:
        # fixed-size cbars (in inches) placed to the right of the heatmap area.
        # Vert single-metric: one cbar set centered on the middle cell-type row.
        # Horiz: one cbar set per metric row, vertically centered on that metric's row.
        cbar_h_in = 1.0
        cbar_w_in = 0.12
        cbar_gap_in = 0.08
        gap_to_heatmap_in = 0.55  # gap between heatmap right edge and first cbar
        cbar_h_frac = cbar_h_in / fig_h
        cbar_w_frac = cbar_w_in / fig_w
        cbar_gap_frac = cbar_gap_in / fig_w

        if horiz:
            cbar_metric_indices = list(range(n_metrics))
        else:
            cbar_metric_indices = [0]  # single set, uses metric 0's vmax

        for cbar_j in cbar_metric_indices:
            # the cbar is a generic intensity indicator (no numeric scale), so the actual vmax
            # doesn't matter; use 1.0 just to draw the gradient cleanly. Per-panel values live
            # in the cell annotations.
            col_vmax = 1.0
            if horiz:
                # rightmost heatmap in metric row cbar_j: heat_axes[n_cell_types - 1, cbar_j]
                row_bbox = heat_axes[n_cell_types // 2, cbar_j].get_position()
                rightmost_bbox = heat_axes[n_cell_types - 1, cbar_j].get_position()
                ref_bbox = heat_axes[0, cbar_j].get_position()
                cbar_y_center = (ref_bbox.y0 + ref_bbox.y1) / 2
            else:
                ref_bbox = heat_axes[n_cell_types // 2, 0].get_position()
                rightmost_bbox = heat_axes[0, n_metrics - 1].get_position()
                cbar_y_center = (ref_bbox.y0 + ref_bbox.y1) / 2

            cbar_y0 = cbar_y_center - cbar_h_frac / 2
            cbar_x_start = rightmost_bbox.x1 + gap_to_heatmap_in / fig_w
            # no numeric tick labels on the cbar anymore, so no need to pad between the value
            # ticks and the label — anchor the label just to the left of the leftmost cbar.
            label_anchor_x = cbar_x_start

            for k, exp in enumerate(exp_levels):
                x = cbar_x_start + k * (cbar_w_frac + cbar_gap_frac)
                lax = fig.add_axes([x, cbar_y0, cbar_w_frac, cbar_h_frac])
                gradient = np.linspace(0.0, col_vmax, 200).reshape(-1, 1)
                lax.imshow(gradient, cmap=cmaps[exp], aspect='auto', origin='lower',
                           vmin=0.0, vmax=col_vmax, extent=(0, 1, 0.0, col_vmax))
                lax.set_xticks([])
                lax.set_yticks([])
                # experience label rotated 90 above each cbar, colored
                lax.text(0.5, 1.04, exp_levels[k], transform=lax.transAxes,
                         rotation=90, ha='center', va='bottom',
                         fontsize=10, color=exp_colors[k])
                for side in ('top', 'bottom', 'left', 'right'):
                    lax.spines[side].set_visible(True)
                    lax.spines[side].set_color('black')
                    lax.spines[side].set_linewidth(0.5)

            # cbar group label on the LEFT of the leftmost cbar, rotated 90 (bottom-to-top)
            fig.text(label_anchor_x - 0.005, cbar_y0 + cbar_h_frac / 2,
                     cbar_labels[cbar_j], rotation=90,
                     ha='right', va='center', fontsize=10)

    if suptitle:
        plt.suptitle(suptitle, fontsize=14, y=0.99)

    if save_dir:
        if folder is None and event_type == 'coding_score':
            folder = 'coding_scores_and_kernels'
        elif folder is None:
            folder = 'response_metrics'
        filename = 'metric_heatmap_grid_' + groupby_col + suffix
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            # legacy ANOVA stats were saved with the _ttest.csv suffix; keep that
            # name for backward compat when use_mlm=False, use _anova_mlm.csv for the
            # new hierarchical path (consistent across all 4 heatmap functions).
            stats_suffix = '_anova_mlm.csv' if use_mlm else '_ttest.csv'
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)), index=False)
        except BaseException:
            print('STATS TABLE DID NOT SAVE')
    return heat_axes, stats_table


def plot_metric_heatmap_area_and_depth_by_cell_type(
        results_pivoted, metric, metric_label=None,
        area_col='targeted_structure', depth_col='binned_depth',
        area_label='Visual area', depth_label='Imaging depth (um)',
        exp_col='experience_level', aggregate='mean',
        vmax=None, multi_star=False,
        suptitle=None, suffix='', save_dir=None, folder=None,
        use_mlm=True, group_column='mouse_id', event_type='Not specified'):
    """
    Two-row figure with the same panel layout as plot_metric_heatmap_grid_by_cell_type_and_metric
    but groupby_col differs by row:
      - Row 0: bins of `area_col` (default 'targeted_structure')
      - Row 1: bins of `depth_col` (default 'binned_depth')
    Columns are cell types (utils.get_cell_types()).

    All panels share a single vmax (max across both groupings) so colors are directly comparable
    across area and depth. A single shared colorbar set is shown on the right of the figure.
    `metric_label` (or `metric` if not given) becomes the figure suptitle on top.

    aggregate: 'mean' (default) or 'std' — what to compute per (cell_type, grouping_bin,
        experience_level) and display as both the cell color and the annotated value. Note:
        significance stars always come from one-way ANOVAs on the underlying raw values
        (testing whether *means* differ across groups), regardless of `aggregate`. If you
        switch to 'std' the stars still ask "do group means differ?" — ignore them if that's
        not the question you want to answer.
    """
    suffix = _norm_suffix(suffix)
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import gridspec
    from matplotlib.transforms import blended_transform_factory
    from scipy import stats as sstats

    cell_types = utils.get_cell_types()
    exp_levels = utils.get_experience_levels()
    exp_colors = utils.get_experience_level_colors()
    exp_abbrev = utils.get_abbreviated_experience_levels()

    cmaps = {exp_levels[k]: LinearSegmentedColormap.from_list(
        f'mh2_{exp_levels[k].replace(" ", "_").replace("+", "p")}',
        [(1.0, 1.0, 1.0), exp_colors[k]]) for k in range(len(exp_levels))}

    def tier_stars(p):
        if not np.isfinite(p) or p >= 0.05:
            return ''
        if multi_star:
            return '***' if p < 0.001 else '**' if p < 0.01 else '*'
        return '*'

    if metric_label is None:
        metric_label = metric

    groupings = [(area_col, area_label), (depth_col, depth_label)]
    n_groupings = len(groupings)
    n_cell_types = len(cell_types)
    ncols = len(exp_levels)

    # per-grouping ordered row labels
    group_orders = [sorted(results_pivoted[gc].dropna().unique()) for gc, _ in groupings]
    nrows_per_grouping = [len(go) for go in group_orders]
    total_heatmap_rows = sum(nrows_per_grouping)

    # compute matrices, panel maxes, per-cell n, and ANOVA-based stars per (grouping, cell_type)
    mats = {}
    panel_max = {}
    cell_n = {}
    col_stars = {}
    row_stars = {}
    pairwise_tables = []
    for g, (gcol, _) in enumerate(groupings):
        group_order = group_orders[g]
        for i, cell_type in enumerate(cell_types):
            ct_data = results_pivoted[results_pivoted.cell_type == cell_type]
            grouped = ct_data.groupby([gcol, exp_col])[metric]
            if aggregate == 'std':
                agg = grouped.std().unstack(exp_col)
            else:
                agg = grouped.mean().unstack(exp_col)
            agg = agg.reindex(index=group_order, columns=exp_levels)
            mat = agg.values.astype(float)
            mats[(g, i)] = mat
            finite = mat[np.isfinite(mat)]
            panel_max[(g, i)] = float(finite.max()) if finite.size else 0.0

            for r, d in enumerate(group_order):
                for c, exp in enumerate(exp_levels):
                    vals = ct_data.loc[(ct_data[gcol] == d) & (ct_data[exp_col] == exp),
                                       metric].dropna().values
                    cell_n[(g, i, r, c)] = int(len(vals))

            # column-direction: within each experience level, hierarchical test (MLM by
            # default) across the grouping values (e.g., "do depths differ within this
            # experience level?")
            for c, exp in enumerate(exp_levels):
                col_subset = ct_data[ct_data[exp_col] == exp].dropna(subset=[metric, gcol])
                panel_stats = compute_stats(col_subset, metric, column_to_compare=gcol,
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type=event_type, cell_type=cell_type)
                col_stars[(g, i, c)] = tier_stars(_heatmap_omnibus_p(panel_stats))
                panel_stats = insert_stats_metadata(panel_stats, grouping=gcol,
                                                   direction='across_groups_within_exp',
                                                   held_fixed_col=exp_col,
                                                   held_fixed_value=exp)
                pairwise_tables.append(panel_stats)

            # row-direction: within each grouping bin, hierarchical test (MLM by default)
            # across experience levels (e.g., "do experience levels differ at this depth?")
            for r, d in enumerate(group_order):
                row_subset = ct_data[ct_data[gcol] == d].dropna(subset=[metric, exp_col])
                panel_stats = compute_stats(row_subset, metric, column_to_compare=exp_col,
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type=event_type, cell_type=cell_type)
                row_stars[(g, i, r)] = tier_stars(_heatmap_omnibus_p(panel_stats))
                panel_stats = insert_stats_metadata(panel_stats, grouping=gcol,
                                                   direction='across_exp_within_group',
                                                   held_fixed_col=gcol,
                                                   held_fixed_value=d)
                pairwise_tables.append(panel_stats)
    stats_table = (pd.concat(pairwise_tables, ignore_index=True)
                   if pairwise_tables else pd.DataFrame())

    # shared vmax across area + depth panels
    if vmax is None:
        vmax_shared = max(panel_max[(g, i)] for g in range(n_groupings)
                          for i in range(n_cell_types))
        if vmax_shared <= 0:
            vmax_shared = 1.0
    else:
        vmax_shared = float(vmax)

    # figure layout
    fig_w = 2.2 * n_cell_types + 2.1
    # base padding + per-row term that scales with total heatmap rows across both groupings
    fig_h = 0.30 * total_heatmap_rows + 1.4
    figsize = (fig_w, fig_h)
    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.6, 0.5], wspace=0.4)
    # 2-row main grid; rows are sized proportionally to nrows of each grouping so cells in the
    # area row and depth row end up roughly the same height.
    main_gs = gridspec.GridSpecFromSubplotSpec(
        n_groupings, n_cell_types, subplot_spec=outer[0],
        height_ratios=nrows_per_grouping,
        hspace=0.35, wspace=0.3)

    fig.subplots_adjust(left=max(0.10, 1.10 / fig_w))

    heat_axes = np.empty((n_groupings, n_cell_types), dtype=object)
    for g, (gcol, _) in enumerate(groupings):
        nrows_g = nrows_per_grouping[g]
        group_order = group_orders[g]
        for i, cell_type in enumerate(cell_types):
            ax = fig.add_subplot(main_gs[g, i])
            heat_axes[g, i] = ax
            mat = mats[(g, i)]

            for c, exp in enumerate(exp_levels):
                col_vals = mat[:, c:c + 1]
                ax.imshow(col_vals, cmap=cmaps[exp], vmin=0, vmax=vmax_shared,
                          aspect='auto', extent=(c, c + 1, nrows_g, 0), interpolation='nearest')
                for r in range(nrows_g):
                    val = mat[r, c]
                    if not np.isfinite(val):
                        continue
                    frac = val / vmax_shared if vmax_shared > 0 else 0.0
                    text_color = 'black' if frac < 0.6 else 'white'
                    ax.text(c + 0.5, r + 0.42, f'{val:.2f}',
                            ha='center', va='center', fontsize=8, color=text_color)
                    n = cell_n.get((g, i, r, c), 0)
                    if n:
                        ax.text(c + 0.5, r + 0.72, f'n={n}',
                                ha='center', va='center', fontsize=6, color=text_color)

            col_trans = blended_transform_factory(ax.transData, ax.transAxes)
            row_trans = blended_transform_factory(ax.transAxes, ax.transData)
            for c in range(ncols):
                s = col_stars.get((g, i, c), '')
                if s:
                    ax.text(c + 0.5, 1.0, s, transform=col_trans,
                            ha='center', va='center',
                            fontsize=12, fontweight='bold', color='k', clip_on=False)
            for r in range(nrows_g):
                s = row_stars.get((g, i, r), '')
                if s:
                    ax.text(1.02, r + 0.5, s, transform=row_trans,
                            ha='left', va='center',
                            fontsize=12, fontweight='bold', color='k', clip_on=False)

            for r in range(nrows_g + 1):
                ax.axhline(r, color='white', linewidth=0.7)
            for cc in range(ncols + 1):
                ax.axvline(cc, color='white', linewidth=0.7)

            ax.set_xlim(0, ncols)
            ax.set_ylim(nrows_g, 0)
            ax.set_xticks(np.arange(ncols) + 0.5)
            is_bottom_row = (g == n_groupings - 1)
            if is_bottom_row:
                ax.set_xticklabels(exp_abbrev, fontsize=10)
                for lbl, color in zip(ax.get_xticklabels(), exp_colors):
                    lbl.set_color(color)
            else:
                ax.set_xticklabels([''] * ncols)
            ax.tick_params(bottom=False)

            ax.set_yticks(np.arange(nrows_g) + 0.5)
            is_leftmost = (i == 0)
            if is_leftmost:
                ax.set_yticklabels([str(d) for d in group_order], fontsize=9)
            else:
                ax.set_yticklabels([''] * nrows_g)
                ax.tick_params(labelleft=False)
            ax.tick_params(left=is_leftmost)

            # cell-type as column title only on top row
            if g == 0:
                ax.set_title(cell_type, fontsize=12, pad=10)

            for side in ('top', 'bottom', 'left', 'right'):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color('black')
                ax.spines[side].set_linewidth(0.8)

    # outside-left rotated labels per grouping ("Area", "Depth")
    fig.canvas.draw()
    ct_offset_inches = 0.55
    for g, (_, glabel) in enumerate(groupings):
        bbox = heat_axes[g, 0].get_position()
        x_pos = max(bbox.x0 - ct_offset_inches / fig_w, 0.005)
        fig.text(x_pos, (bbox.y0 + bbox.y1) / 2,
                 glabel, rotation=90, ha='left', va='center', fontsize=12)

    # single shared cbar set on the right, vertically centered on the full heatmap area
    cbar_h_in = 0.9
    cbar_w_in = 0.08
    cbar_gap_in = 0.05
    gap_to_heatmap_in = 0.55
    cbar_h_frac = cbar_h_in / fig_h
    cbar_w_frac = cbar_w_in / fig_w
    cbar_gap_frac = cbar_gap_in / fig_w

    top_bbox = heat_axes[0, 0].get_position()
    bot_bbox = heat_axes[n_groupings - 1, 0].get_position()
    total_y_center = (top_bbox.y1 + bot_bbox.y0) / 2
    cbar_y0 = total_y_center - cbar_h_frac / 2
    rightmost_bbox = heat_axes[0, n_cell_types - 1].get_position()
    cbar_x_start = rightmost_bbox.x1 + gap_to_heatmap_in / fig_w

    cbar_label_text = 'Coding score' if 'coding' in str(metric_label).lower() else str(metric_label)

    for k, exp in enumerate(exp_levels):
        x = cbar_x_start + k * (cbar_w_frac + cbar_gap_frac)
        lax = fig.add_axes([x, cbar_y0, cbar_w_frac, cbar_h_frac])
        gradient = np.linspace(0.0, vmax_shared, 200).reshape(-1, 1)
        lax.imshow(gradient, cmap=cmaps[exp], aspect='auto', origin='lower',
                   vmin=0.0, vmax=vmax_shared, extent=(0, 1, 0.0, vmax_shared))
        lax.set_xticks([])
        # experience-level label below each cbar, rotated to read top-to-bottom (same direction
        # as the metric label on the right of the rightmost cbar)
        lax.text(0.5, -0.04, exp_levels[k], transform=lax.transAxes,
                 rotation=270, ha='center', va='top',
                 fontsize=10, color=exp_colors[k])
        if k == 0:
            lax.set_yticks([0.0, vmax_shared])
            lax.set_yticklabels(['0', f'{vmax_shared:.2f}'], fontsize=9)
            lax.tick_params(labelsize=9)
        else:
            lax.set_yticks([])
        if k == len(exp_levels) - 1:
            lax.yaxis.set_label_position('right')
            lax.set_ylabel(cbar_label_text, fontsize=11, rotation=270, labelpad=15)
        for side in ('top', 'bottom', 'left', 'right'):
            lax.spines[side].set_visible(True)
            lax.spines[side].set_color('black')
            lax.spines[side].set_linewidth(0.5)

    # metric label as suptitle on top of figure (replaces the previous y-axis "ylabel")
    fig.suptitle(suptitle if suptitle else metric_label, fontsize=14, x=0.38, y=1.05)

    if save_dir:
        if folder is None:
            folder = 'response_metrics'
        filename = _clean_filename('metric_heatmap_area_and_depth_' + metric + '_' + aggregate + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            stats_suffix = '_anova_mlm.csv' if use_mlm else '_anova.csv'
            # if suffix is None | suffix == '':
            #     stats_suffix = '_' + stats_suffix
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)), index=False)
        except BaseException:
            print('STATS TABLE DID NOT SAVE')

    return heat_axes, stats_table


def plot_bidirectional_metric_heatmap_area_and_depth_by_cell_type(
        results_pivoted, metric, metric_label=None,
        area_col='targeted_structure', depth_col='binned_depth',
        area_label='Visual area', depth_label='Imaging depth (um)',
        exp_col='experience_level',
        cmap='PRGn', vmax=None, multi_star=False,
        suptitle=None, suffix='', save_dir=None, folder=None,
        use_mlm=True, group_column='mouse_id', event_type='Not specified'):
    """
    Variant of plot_metric_heatmap_area_and_depth_by_cell_type for bidirectional indices
    (e.g., change_modulation_index, experience modulation indices — values can be negative or
    positive and 0 is meaningful). Uses a single diverging colormap (default 'PRGn') centered
    at 0 with TwoSlopeNorm and a single shared colorbar on the right, instead of three
    sequential per-experience-level colormaps.

    cmap: a diverging matplotlib colormap name. Default 'PRGn' (purple-white-green).
    vmax: symmetric color limit (vmin = -vmax). If None, defaults to max(|value|) across
        all panels (rounded up slightly).

    See plot_metric_heatmap_area_and_depth_by_cell_type for everything else.
    """
    suffix = _norm_suffix(suffix)
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib import gridspec
    from matplotlib.transforms import blended_transform_factory

    cell_types = utils.get_cell_types()
    exp_levels = utils.get_experience_levels()
    exp_colors = utils.get_experience_level_colors()
    exp_abbrev = utils.get_abbreviated_experience_levels()

    def tier_stars(p):
        if not np.isfinite(p) or p >= 0.05:
            return ''
        if multi_star:
            return '***' if p < 0.001 else '**' if p < 0.01 else '*'
        return '*'

    if metric_label is None:
        metric_label = metric

    groupings = [(area_col, area_label), (depth_col, depth_label)]
    n_groupings = len(groupings)
    n_cell_types = len(cell_types)
    ncols = len(exp_levels)

    group_orders = [sorted(results_pivoted[gc].dropna().unique()) for gc, _ in groupings]
    nrows_per_grouping = [len(go) for go in group_orders]
    total_heatmap_rows = sum(nrows_per_grouping)

    # compute matrices (means), per-cell n, panel absmax, and ANOVA-based stars
    mats = {}
    panel_absmax = {}
    cell_n = {}
    col_stars = {}
    row_stars = {}
    pairwise_tables = []
    for g, (gcol, _) in enumerate(groupings):
        group_order = group_orders[g]
        for i, cell_type in enumerate(cell_types):
            ct_data = results_pivoted[results_pivoted.cell_type == cell_type]
            agg = (ct_data.groupby([gcol, exp_col])[metric].mean()
                   .unstack(exp_col))
            agg = agg.reindex(index=group_order, columns=exp_levels)
            mat = agg.values.astype(float)
            mats[(g, i)] = mat
            finite = mat[np.isfinite(mat)]
            panel_absmax[(g, i)] = float(np.abs(finite).max()) if finite.size else 0.0

            for r, d in enumerate(group_order):
                for c, exp in enumerate(exp_levels):
                    vals = ct_data.loc[(ct_data[gcol] == d) & (ct_data[exp_col] == exp),
                                       metric].dropna().values
                    cell_n[(g, i, r, c)] = int(len(vals))

            for c, exp in enumerate(exp_levels):
                col_subset = ct_data[ct_data[exp_col] == exp].dropna(subset=[metric, gcol])
                panel_stats = compute_stats(col_subset, metric, column_to_compare=gcol,
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type=event_type, cell_type=cell_type)
                col_stars[(g, i, c)] = tier_stars(_heatmap_omnibus_p(panel_stats))
                panel_stats = insert_stats_metadata(panel_stats, grouping=gcol,
                                                   direction='across_groups_within_exp',
                                                   held_fixed_col=exp_col,
                                                   held_fixed_value=exp)
                pairwise_tables.append(panel_stats)

            for r, d in enumerate(group_order):
                row_subset = ct_data[ct_data[gcol] == d].dropna(subset=[metric, exp_col])
                panel_stats = compute_stats(row_subset, metric, column_to_compare=exp_col,
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type=event_type, cell_type=cell_type)
                row_stars[(g, i, r)] = tier_stars(_heatmap_omnibus_p(panel_stats))
                panel_stats = insert_stats_metadata(panel_stats, grouping=gcol,
                                                   direction='across_exp_within_group',
                                                   held_fixed_col=gcol,
                                                   held_fixed_value=d)
                pairwise_tables.append(panel_stats)
    stats_table = (pd.concat(pairwise_tables, ignore_index=True)
                   if pairwise_tables else pd.DataFrame())

    # symmetric vmax across all panels (so 0 stays the diverging-cmap center)
    if vmax is None:
        vmax_shared = max(panel_absmax[(g, i)] for g in range(n_groupings)
                          for i in range(n_cell_types))
        if vmax_shared <= 0:
            vmax_shared = 1.0
    else:
        vmax_shared = float(abs(vmax))
    vmin_shared = -vmax_shared
    norm = TwoSlopeNorm(vmin=vmin_shared, vcenter=0.0, vmax=vmax_shared)

    # figure layout (matches the sequential-cmap version)
    fig_w = 2.2 * n_cell_types + 2.1
    fig_h = 0.30 * total_heatmap_rows + 1.4
    figsize = (fig_w, fig_h)
    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.6, 0.5], wspace=0.4)
    main_gs = gridspec.GridSpecFromSubplotSpec(
        n_groupings, n_cell_types, subplot_spec=outer[0],
        height_ratios=nrows_per_grouping,
        hspace=0.35, wspace=0.3)

    fig.subplots_adjust(left=max(0.10, 1.10 / fig_w))

    heat_axes = np.empty((n_groupings, n_cell_types), dtype=object)
    for g, (gcol, _) in enumerate(groupings):
        nrows_g = nrows_per_grouping[g]
        group_order = group_orders[g]
        for i, cell_type in enumerate(cell_types):
            ax = fig.add_subplot(main_gs[g, i])
            heat_axes[g, i] = ax
            mat = mats[(g, i)]

            # single-cmap imshow over the whole matrix
            ax.imshow(mat, cmap=cmap, norm=norm,
                      aspect='auto', extent=(0, ncols, nrows_g, 0), interpolation='nearest')
            for r in range(nrows_g):
                for c in range(ncols):
                    val = mat[r, c]
                    if not np.isfinite(val):
                        continue
                    # text color: dark on light cells (small magnitudes), light on saturated
                    frac = abs(val) / vmax_shared if vmax_shared > 0 else 0.0
                    text_color = 'black' if frac < 0.55 else 'white'
                    ax.text(c + 0.5, r + 0.42, f'{val:.2f}',
                            ha='center', va='center', fontsize=8, color=text_color)
                    n = cell_n.get((g, i, r, c), 0)
                    if n:
                        ax.text(c + 0.5, r + 0.72, f'n={n}',
                                ha='center', va='center', fontsize=6, color=text_color)

            col_trans = blended_transform_factory(ax.transData, ax.transAxes)
            row_trans = blended_transform_factory(ax.transAxes, ax.transData)
            for c in range(ncols):
                s = col_stars.get((g, i, c), '')
                if s:
                    ax.text(c + 0.5, 1.0, s, transform=col_trans,
                            ha='center', va='center',
                            fontsize=12, fontweight='bold', color='k', clip_on=False)
            for r in range(nrows_g):
                s = row_stars.get((g, i, r), '')
                if s:
                    ax.text(1.02, r + 0.5, s, transform=row_trans,
                            ha='left', va='center',
                            fontsize=12, fontweight='bold', color='k', clip_on=False)

            for r in range(nrows_g + 1):
                ax.axhline(r, color='white', linewidth=0.7)
            for cc in range(ncols + 1):
                ax.axvline(cc, color='white', linewidth=0.7)

            ax.set_xlim(0, ncols)
            ax.set_ylim(nrows_g, 0)
            ax.set_xticks(np.arange(ncols) + 0.5)
            is_bottom_row = (g == n_groupings - 1)
            if is_bottom_row:
                ax.set_xticklabels(exp_abbrev, fontsize=10)
                for lbl, color in zip(ax.get_xticklabels(), exp_colors):
                    lbl.set_color(color)
            else:
                ax.set_xticklabels([''] * ncols)
            ax.tick_params(bottom=False)

            ax.set_yticks(np.arange(nrows_g) + 0.5)
            is_leftmost = (i == 0)
            if is_leftmost:
                ax.set_yticklabels([str(d) for d in group_order], fontsize=9)
            else:
                ax.set_yticklabels([''] * nrows_g)
                ax.tick_params(labelleft=False)
            ax.tick_params(left=is_leftmost)

            if g == 0:
                ax.set_title(cell_type, fontsize=12, pad=10)

            for side in ('top', 'bottom', 'left', 'right'):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color('black')
                ax.spines[side].set_linewidth(0.8)

    # outside-left rotated labels per grouping
    fig.canvas.draw()
    ct_offset_inches = 0.55
    for g, (_, glabel) in enumerate(groupings):
        bbox = heat_axes[g, 0].get_position()
        x_pos = max(bbox.x0 - ct_offset_inches / fig_w, 0.005)
        fig.text(x_pos, (bbox.y0 + bbox.y1) / 2,
                 glabel, rotation=90, ha='left', va='center', fontsize=12)

    # single shared diverging colorbar on the right, vertically centered
    cbar_h_in = 0.9
    cbar_w_in = 0.10
    gap_to_heatmap_in = 0.55
    cbar_h_frac = cbar_h_in / fig_h
    cbar_w_frac = cbar_w_in / fig_w

    top_bbox = heat_axes[0, 0].get_position()
    bot_bbox = heat_axes[n_groupings - 1, 0].get_position()
    cbar_y0 = (top_bbox.y1 + bot_bbox.y0) / 2 - cbar_h_frac / 2
    rightmost_bbox = heat_axes[0, n_cell_types - 1].get_position()
    cbar_x = rightmost_bbox.x1 + gap_to_heatmap_in / fig_w

    lax = fig.add_axes([cbar_x, cbar_y0, cbar_w_frac, cbar_h_frac])
    gradient = np.linspace(vmin_shared, vmax_shared, 200).reshape(-1, 1)
    lax.imshow(gradient, cmap=cmap, norm=norm, aspect='auto', origin='lower',
               extent=(0, 1, vmin_shared, vmax_shared))
    lax.set_xticks([])
    lax.set_yticks([vmin_shared, 0.0, vmax_shared])
    lax.set_yticklabels([f'{vmin_shared:.2f}', '0', f'{vmax_shared:.2f}'], fontsize=9)
    lax.tick_params(labelsize=9)
    lax.yaxis.set_label_position('right')
    cbar_label_text = str(metric_label)
    lax.set_ylabel(cbar_label_text, fontsize=11, rotation=270, labelpad=15)
    for side in ('top', 'bottom', 'left', 'right'):
        lax.spines[side].set_visible(True)
        lax.spines[side].set_color('black')
        lax.spines[side].set_linewidth(0.5)

    fig.suptitle(suptitle if suptitle else metric_label, fontsize=14, x=0.38, y=1.05)

    if save_dir:
        if folder is None:
            folder = 'response_metrics'
        filename = _clean_filename('bidirectional_metric_heatmap_area_and_depth_' + metric + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            stats_suffix = '_anova_mlm.csv' if use_mlm else '_anova.csv'
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)), index=False)
        except BaseException:
            print('STATS TABLE DID NOT SAVE')

    return heat_axes, stats_table


def plot_experience_modulation_heatmap_area_and_depth_by_cell_type(
        metrics_table, event_type, metric, cells_table,
        area_col='targeted_structure', depth_col='binned_depth',
        area_label='Visual area', depth_label='Imaging depth (um)',
        all_comparisons=True, vmax=None, multi_star=False,
        suptitle=None, suffix='', save_dir=None, folder=None,
        use_mlm=True, group_column='mouse_id'):
    """
    Combines plot_experience_modulation_index_depth_heatmap_by_cell_type with the area + depth
    layout of plot_metric_heatmap_area_and_depth_by_cell_type. Two rows of panels:
      - Row 0: bins of `area_col` (default 'targeted_structure')
      - Row 1: bins of `depth_col` (default 'binned_depth')
    Columns are cell types. Each panel's heatmap has rows = grouping bins and columns =
    experience-level comparisons ('F N', 'N+ N', and optionally 'F N+'), with each column
    using its own diverging colormap built from utils.get_experience_level_colors() —
    positive = "novel-side" experience level, negative = "familiar-side", white at 0.

    Shared color scale across all panels (vmin = -vmax, symmetric around 0). Single set of
    3 vertical colorbars on the right (one per comparison) with experience-level endpoint
    labels at top/bottom of each.

    Significance stars come from one-way ANOVAs (same convention as
    plot_metric_heatmap_area_and_depth_by_cell_type):
      - Column stars: within each comparison column, ANOVA across grouping bins.
      - Row stars:    within each grouping bin, ANOVA across the 3 comparison values.

    metrics_table, event_type, metric, cells_table: passed to
        cell_metrics.compute_experience_modulation_index_new (same as the existing
        plot_experience_modulation_index_depth_heatmap_by_cell_type function).
    all_comparisons: include the third 'F N+' (Novel + vs. Familiar) comparison column.
    vmax: symmetric color limit. Defaults to max(|value|) across all panels, clipped to 1.0
        (modulation indices are bounded -1..1).
    """
    suffix = _norm_suffix(suffix)
    import visual_behavior.ophys.response_analysis.cell_metrics as cm
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib import gridspec
    from matplotlib.transforms import blended_transform_factory

    cell_types = utils.get_cell_types()
    exp_colors = utils.get_experience_level_colors()

    if all_comparisons:
        value_vars = ['F N', 'N+ N', 'F N+']
    else:
        value_vars = ['F N', 'N+ N']

    # per-comparison diverging cmaps (same convention as the existing _by_cell_type version):
    # 'F N'  -> neg=F(blue), pos=N(red);  'N+ N' -> neg=N+(purple), pos=N(red);
    # 'F N+' -> neg=F(blue), pos=N+(purple)
    color_for = {'F': exp_colors[0], 'N': exp_colors[1], 'N+': exp_colors[2]}
    full_name = {'F': 'Familiar', 'N': 'Novel', 'N+': 'Novel +'}
    cmaps = {}
    for comp in value_vars:
        neg_label, pos_label = comp.split(' ')
        cmaps[comp] = LinearSegmentedColormap.from_list(
            f'expmod2_{comp.replace(" ", "_").replace("+", "p")}',
            [color_for[neg_label], (1.0, 1.0, 1.0), color_for[pos_label]])

    # compute per-cell experience modulation values and attach area/depth columns if missing
    exp_mod = cm.compute_experience_modulation_index_new(metrics_table, metric, cells_table)
    exp_mod = exp_mod.drop_duplicates(subset='cell_specimen_id')
    for gc in (area_col, depth_col):
        if gc not in exp_mod.columns:
            gmap = cells_table.drop_duplicates('cell_specimen_id').set_index('cell_specimen_id')[gc]
            exp_mod[gc] = exp_mod['cell_specimen_id'].map(gmap)

    def tier_stars(p):
        if not np.isfinite(p) or p >= 0.05:
            return ''
        if multi_star:
            return '***' if p < 0.001 else '**' if p < 0.01 else '*'
        return '*'

    groupings = [(area_col, area_label), (depth_col, depth_label)]
    n_groupings = len(groupings)
    n_cell_types = len(cell_types)
    ncols = len(value_vars)

    group_orders = [sorted(exp_mod[gc].dropna().unique()) for gc, _ in groupings]
    nrows_per_grouping = [len(go) for go in group_orders]
    total_heatmap_rows = sum(nrows_per_grouping)

    # compute matrices, n, panel absmax, and per-cell omnibus stars
    mats = {}
    panel_absmax = {}
    cell_n = {}
    col_stars = {}
    row_stars = {}
    pairwise_tables = []
    for g, (gcol, _) in enumerate(groupings):
        group_order = group_orders[g]
        for i, cell_type in enumerate(cell_types):
            ct_data = exp_mod[exp_mod.cell_type == cell_type].dropna(subset=[gcol])
            mat = np.full((nrows_per_grouping[g], ncols), np.nan)
            for r, d in enumerate(group_order):
                for c, comp in enumerate(value_vars):
                    vals = ct_data.loc[ct_data[gcol] == d, comp].dropna().values
                    cell_n[(g, i, r, c)] = int(len(vals))
                    if len(vals) > 0:
                        mat[r, c] = float(np.mean(vals))
            mats[(g, i)] = mat
            finite = mat[np.isfinite(mat)]
            panel_absmax[(g, i)] = float(np.abs(finite).max()) if finite.size else 0.0

            # column stars: within each comparison column, hierarchical test (MLM by
            # default) across grouping bins. NOTE: col_subset drops all metadata other
            # than the value column and the grouping; since mouse_id is not preserved,
            # this will auto-fall-back to ANOVA. model_type='anova' will be recorded.
            for c, comp in enumerate(value_vars):
                col_subset = ct_data[[comp, gcol]].dropna()
                col_subset = col_subset.rename(columns={comp: 'value'})
                panel_stats = compute_stats(col_subset, 'value', column_to_compare=gcol,
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type=event_type, cell_type=cell_type)
                col_stars[(g, i, c)] = tier_stars(_heatmap_omnibus_p(panel_stats))
                # The held-fixed dimension here is the modulation-index comparison
                # name (e.g., 'F N') -- semantically distinct from column_to_compare
                # which is the grouping (binned_depth/targeted_structure).
                panel_stats = insert_stats_metadata(panel_stats, grouping=gcol,
                                                   direction='across_groups_within_comparison',
                                                   held_fixed_col='comparison',
                                                   held_fixed_value=comp)
                pairwise_tables.append(panel_stats)

            # row stars: within each grouping bin, hierarchical test (MLM by default)
            # across the 3 comparison values. Same caveat: melt strips mouse_id, so
            # MLM falls back to ANOVA on this path.
            for r, d in enumerate(group_order):
                row_subset = ct_data[ct_data[gcol] == d][['cell_specimen_id'] + value_vars]
                row_long = row_subset.melt(id_vars='cell_specimen_id', var_name='comparison',
                                           value_vars=value_vars).dropna(subset=['value'])
                panel_stats = compute_stats(row_long, 'value', column_to_compare='comparison',
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type=event_type, cell_type=cell_type)
                row_stars[(g, i, r)] = tier_stars(_heatmap_omnibus_p(panel_stats))
                panel_stats = insert_stats_metadata(panel_stats, grouping=gcol,
                                                   direction='across_comparisons_within_group',
                                                   held_fixed_col=gcol,
                                                   held_fixed_value=d)
                pairwise_tables.append(panel_stats)
    stats_table = (pd.concat(pairwise_tables, ignore_index=True)
                   if pairwise_tables else pd.DataFrame())

    # symmetric shared vmax (clip to 1 since modulation indices are bounded -1..1)
    if vmax is None:
        vmax_shared = max(panel_absmax[(g, i)] for g in range(n_groupings)
                          for i in range(n_cell_types))
        if vmax_shared <= 0:
            vmax_shared = 1.0
        vmax_shared = min(vmax_shared, 1.0)
    else:
        vmax_shared = float(abs(vmax))
    vmin_shared = -vmax_shared
    norm = TwoSlopeNorm(vmin=vmin_shared, vcenter=0.0, vmax=vmax_shared)

    # layout (same as plot_metric_heatmap_area_and_depth_by_cell_type)
    fig_w = 2.2 * n_cell_types + 2.1
    fig_h = 0.30 * total_heatmap_rows + 1.4
    figsize = (fig_w, fig_h)
    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.6, 0.5], wspace=0.4)
    main_gs = gridspec.GridSpecFromSubplotSpec(
        n_groupings, n_cell_types, subplot_spec=outer[0],
        height_ratios=nrows_per_grouping,
        hspace=0.35, wspace=0.3)
    fig.subplots_adjust(left=max(0.10, 1.10 / fig_w))

    heat_axes = np.empty((n_groupings, n_cell_types), dtype=object)
    for g, (gcol, _) in enumerate(groupings):
        nrows_g = nrows_per_grouping[g]
        group_order = group_orders[g]
        for i, cell_type in enumerate(cell_types):
            ax = fig.add_subplot(main_gs[g, i])
            heat_axes[g, i] = ax
            mat = mats[(g, i)]

            # per-comparison cmap (column-by-column imshow)
            for c, comp in enumerate(value_vars):
                col_vals = mat[:, c:c + 1]
                ax.imshow(col_vals, cmap=cmaps[comp], norm=norm,
                          aspect='auto', extent=(c, c + 1, nrows_g, 0), interpolation='nearest')
                for r in range(nrows_g):
                    val = mat[r, c]
                    if not np.isfinite(val):
                        continue
                    frac = abs(val) / vmax_shared if vmax_shared > 0 else 0.0
                    text_color = 'black' if frac < 0.55 else 'white'
                    ax.text(c + 0.5, r + 0.42, f'{val:.2f}',
                            ha='center', va='center', fontsize=8, color=text_color)
                    n = cell_n.get((g, i, r, c), 0)
                    if n:
                        ax.text(c + 0.5, r + 0.72, f'n={n}',
                                ha='center', va='center', fontsize=6, color=text_color)

            col_trans = blended_transform_factory(ax.transData, ax.transAxes)
            row_trans = blended_transform_factory(ax.transAxes, ax.transData)
            for c in range(ncols):
                s = col_stars.get((g, i, c), '')
                if s:
                    ax.text(c + 0.5, 1.0, s, transform=col_trans,
                            ha='center', va='center',
                            fontsize=12, fontweight='bold', color='k', clip_on=False)
            for r in range(nrows_g):
                s = row_stars.get((g, i, r), '')
                if s:
                    ax.text(1.02, r + 0.5, s, transform=row_trans,
                            ha='left', va='center',
                            fontsize=12, fontweight='bold', color='k', clip_on=False)

            for r in range(nrows_g + 1):
                ax.axhline(r, color='white', linewidth=0.7)
            for cc in range(ncols + 1):
                ax.axvline(cc, color='white', linewidth=0.7)

            ax.set_xlim(0, ncols)
            ax.set_ylim(nrows_g, 0)
            ax.set_xticks(np.arange(ncols) + 0.5)
            is_bottom_row = (g == n_groupings - 1)
            if is_bottom_row:
                # turn 'F N' / 'N+ N' / 'F N+' into 'F vs. N' / 'N+ vs. N' / 'F vs. N+'
                xtick_labels = [c.split(' ')[0] + ' vs. ' + c.split(' ')[1] for c in value_vars]
                ax.set_xticklabels(xtick_labels, fontsize=10, rotation=45, ha='right',
                                   rotation_mode='anchor')
            else:
                ax.set_xticklabels([''] * ncols)
            ax.tick_params(bottom=False)

            ax.set_yticks(np.arange(nrows_g) + 0.5)
            is_leftmost = (i == 0)
            if is_leftmost:
                ax.set_yticklabels([str(d) for d in group_order], fontsize=9)
            else:
                ax.set_yticklabels([''] * nrows_g)
                ax.tick_params(labelleft=False)
            ax.tick_params(left=is_leftmost)

            if g == 0:
                ax.set_title(cell_type, fontsize=12, pad=10)

            for side in ('top', 'bottom', 'left', 'right'):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color('black')
                ax.spines[side].set_linewidth(0.8)

    # outside-left rotated labels per grouping ("Visual area", "Imaging depth")
    fig.canvas.draw()
    ct_offset_inches = 0.55
    for g, (_, glabel) in enumerate(groupings):
        bbox = heat_axes[g, 0].get_position()
        x_pos = max(bbox.x0 - ct_offset_inches / fig_w, 0.005)
        fig.text(x_pos, (bbox.y0 + bbox.y1) / 2,
                 glabel, rotation=90, ha='left', va='center', fontsize=12)

    # right-side cbar set: one vertical cbar per comparison, each with its own diverging cmap.
    cbar_h_in = 0.81
    cbar_w_in = 0.072
    cbar_gap_in = 0.045
    gap_to_heatmap_in = 0.55
    cbar_h_frac = cbar_h_in / fig_h
    cbar_w_frac = cbar_w_in / fig_w
    cbar_gap_frac = cbar_gap_in / fig_w

    top_bbox = heat_axes[0, 0].get_position()
    bot_bbox = heat_axes[n_groupings - 1, 0].get_position()
    cbar_y0 = (top_bbox.y1 + bot_bbox.y0) / 2 - cbar_h_frac / 2
    rightmost_bbox = heat_axes[0, n_cell_types - 1].get_position()
    cbar_x_start = rightmost_bbox.x1 + gap_to_heatmap_in / fig_w
    # capture the original left-edge as the label anchor, then nudge the cbars right so the
    # value tick labels (on the leftmost cbar) have room before the "Experience modulation" label
    label_anchor_x = cbar_x_start
    cbar_x_start += 0.02

    for k, comp in enumerate(value_vars):
        x = cbar_x_start + k * (cbar_w_frac + cbar_gap_frac)
        lax = fig.add_axes([x, cbar_y0, cbar_w_frac, cbar_h_frac])
        gradient = np.linspace(vmin_shared, vmax_shared, 200).reshape(-1, 1)
        lax.imshow(gradient, cmap=cmaps[comp], norm=norm, aspect='auto', origin='lower',
                   extent=(0, 1, vmin_shared, vmax_shared))
        lax.set_xticks([])
        # experience-level endpoint labels at top (positive end) and bottom (negative end),
        # rotated 90 so they read bottom-to-top alongside each cbar's vertical orientation
        neg_token, pos_token = comp.split(' ')
        lax.text(0.5, 1.10, full_name[pos_token], transform=lax.transAxes,
                 ha='center', va='bottom', rotation=90,
                 fontsize=9, color=color_for[pos_token])
        lax.text(0.5, -0.03, full_name[neg_token], transform=lax.transAxes,
                 ha='center', va='top', rotation=90,
                 fontsize=9, color=color_for[neg_token])
        if k == 0:
            lax.set_yticks([vmin_shared, 0, vmax_shared])
            lax.set_yticklabels([f'{vmin_shared:.1f}', '0', f'{vmax_shared:.1f}'], fontsize=8)
            lax.tick_params(labelsize=8)
        else:
            lax.set_yticks([])
        for side in ('top', 'bottom', 'left', 'right'):
            lax.spines[side].set_visible(True)
            lax.spines[side].set_color('black')
            lax.spines[side].set_linewidth(0.5)

    # vertical "Experience modulation" label on the LEFT of the cbar group (just outside the
    # value tick labels of the leftmost cbar). Rotated 90 so it reads bottom-to-top, matching
    # the experience-level endpoint labels.
    fig.text(label_anchor_x - 0.02, cbar_y0 + cbar_h_frac / 2,
             'Experience modulation', rotation=90,
             ha='right', va='center', fontsize=9)

    # map event_type ('images'/'omissions'/'changes') to the singular form used in the title
    event_label_map = {'images': 'image', 'omissions': 'omission', 'changes': 'change',
                       'all-images': 'image'}
    event_label = event_label_map.get(event_type, event_type.rstrip('s'))
    fig.suptitle(suptitle if suptitle else f'Experience modulation - {event_label} response',
                 fontsize=14, x=0.38, y=1.05)

    if save_dir:
        if folder is None:
            folder = 'response_metrics'
        filename = _clean_filename(('experience_modulation_heatmap_area_and_depth_by_cell_type_'
                    + event_type + suffix))
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            stats_suffix = '_anova_mlm.csv' if use_mlm else '_anova.csv'
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)),
                               index=False)
        except BaseException:
            print('STATS TABLE DID NOT SAVE')

    return heat_axes, stats_table


def change_width(ax, new_value):
    locs = ax.get_xticks()
    for i, patch in enumerate(ax.patches):
        # current_width = patch.get_width()
        # diff = current_width - new_value

        # change the bar width
        patch.set_width(new_value)

        # recenter the bar
        patch.set_x(locs[i // 4] - (new_value * .5))


######## activity metrics vs behavior metrics ##########


def plot_correlation_of_behavior_and_cell_metrics(behavior_metrics, cell_metrics,
                                                  behavior_metric, cell_metric, use_median=True,
                                                  save_dir=None, folder=None):
    """
    Plot correlation between behavior metrics and cell metrics.

    Parameters
    ----------
    behavior_metrics : pd.DataFrame
        Table containing behavior metrics with behavior_session_id
    cell_metrics : pd.DataFrame
        Table containing cell metrics
    behavior_metric : str
        Column name for behavior metric
    cell_metric : str
        Column name for cell metric
    use_median : bool
        If True, group and take median; if False, take mean
    save_dir : str, optional
        Directory to save figure
    folder : str, optional
        Subfolder for saving
    """
    metrics = [cell_metric, behavior_metric]

    # Prepare data
    behavior_metrics_prep = behavior_metrics[[behavior_metric, 'behavior_session_id']]
    metric_data = cell_metrics.merge(behavior_metrics_prep, on='behavior_session_id')

    # Group and compute statistics
    if use_median:
        data = metric_data.groupby(['behavior_session_id', 'experience_level', 'cell_type']).median()[metrics].reset_index()
    else:
        data = metric_data.groupby(['behavior_session_id', 'experience_level', 'cell_type']).mean()[metrics].reset_index()

    # Plot
    figsize = (15, 4)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i, cell_type in enumerate(cell_types):
        ax[i] = sns.scatterplot(data=data[data.cell_type == cell_type], x=behavior_metric, y=cell_metric,
                               hue='experience_level', palette=experience_colors, ax=ax[i])
        ax[i].set_title(cell_type)
        ax[i].get_legend().remove()
    ax[i].legend(bbox_to_anchor=(1, 1), fontsize=10, title_fontsize=10)
    plt.subplots_adjust(wspace=0.4)

    if save_dir:
        filename = _clean_filename('correlation_' + cell_metric + '_' + behavior_metric)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))


def get_metric_index_name(metric, exp_level_1='Familiar', exp_level_2='Novel'):
    """Get the name for metric difference index."""
    metric_name = 'delta_' + exp_level_1 + '_' + exp_level_2 + '_' + metric
    return metric_name


def get_difference_in_metric_across_experience_levels(metrics_table, metric,
                                                      groupby=['cell_specimen_id', 'experience_level'],
                                                      exp_level_1='Familiar', exp_level_2='Novel',
                                                      compute_index=True):
    """
    Compute difference in metric values across experience levels.

    Parameters
    ----------
    metrics_table : pd.DataFrame
        Table with metrics and experience level
    metric : str
        Column name of metric to compute difference for
    groupby : list
        Columns to group by
    exp_level_1 : str
        First experience level
    exp_level_2 : str
        Second experience level
    compute_index : bool
        If True, compute normalized index; if False, compute raw difference

    Returns
    -------
    pd.DataFrame
        Table with difference metric column added
    """
    # Group and compute mean
    metrics = metrics_table.groupby(groupby).mean(numeric_only=True)[[metric]]
    metrics = metrics.unstack()
    metrics.columns = metrics.columns.droplevel(0)

    # Compute difference
    metric_name = get_metric_index_name(metric, exp_level_1=exp_level_1, exp_level_2=exp_level_2)
    if compute_index:
        metrics[metric_name] = (metrics[exp_level_2] - metrics[exp_level_1]) / (metrics[exp_level_2] + metrics[exp_level_1])
    else:
        metrics[metric_name] = (metrics[exp_level_2] - metrics[exp_level_1])

    return metrics


def get_change_in_behavior_and_average_cell_metric_across_mice(cell_metrics_table, behavior_metrics_table,
                                                               platform_experiments_table,
                                                               behavior_metric='mean_dprime_engaged',
                                                               cell_metric='mean_response_pref_image',
                                                               avg_diff_per_cell=True):
    """
    Get change in behavior and cell metrics across experience levels for each mouse.

    Parameters
    ----------
    cell_metrics_table : pd.DataFrame
        Table with cell metrics
    behavior_metrics_table : pd.DataFrame
        Table with behavior metrics
    platform_experiments_table : pd.DataFrame
        Platform experiments table with metadata
    behavior_metric : str
        Name of behavior metric column
    cell_metric : str
        Name of cell metric column
    avg_diff_per_cell : bool
        If True, compute difference per cell first then average by mouse
        If False, average metric per experience level first then compute difference per mouse

    Returns
    -------
    pd.DataFrame
        Merged table with difference metrics for each mouse
    """
    if avg_diff_per_cell:
        # Compute difference across experience for each cell, then average within each mouse
        cell_metrics = get_difference_in_metric_across_experience_levels(cell_metrics_table, cell_metric,
                                                                        groupby=['cell_specimen_id', 'experience_level'],
                                                                        exp_level_1='Familiar', exp_level_2='Novel',
                                                                        compute_index=True)
        # Merge with metadata to get mouse ID
        cell_metrics = cell_metrics.reset_index()
        # Load platform cells table to get mouse_id mapping
        platform_cache_dir = loading.get_platform_analysis_cache_dir()
        platform_cells_table = pd.read_csv(os.path.join(platform_cache_dir, 'platform_paper_ophys_cells_table.csv'), index_col=0)
        cell_metrics = cell_metrics.merge(platform_cells_table[['cell_specimen_id', 'mouse_id']], on='cell_specimen_id')
        # Average across cells per mouse
        metric_name = get_metric_index_name(cell_metric)
        cell_metrics = cell_metrics.groupby(['mouse_id']).mean()[[metric_name]]
    else:
        # Get average value per mouse then take the difference across experience levels
        cell_metrics = get_difference_in_metric_across_experience_levels(cell_metrics_table, cell_metric,
                                                                        groupby=['mouse_id', 'experience_level'],
                                                                        exp_level_1='Familiar', exp_level_2='Novel',
                                                                        compute_index=True)

    # Limit behavior metrics to sessions with ophys
    behavior_stats = behavior_metrics_table.copy()
    behavior_stats = behavior_stats[behavior_stats.behavior_session_id.isin(platform_experiments_table.behavior_session_id.unique())]

    # Compute behavior metric difference across experience levels
    behavior_metrics_diff = get_difference_in_metric_across_experience_levels(behavior_stats, behavior_metric,
                                                                             groupby=['mouse_id', 'experience_level'],
                                                                             exp_level_1='Familiar', exp_level_2='Novel',
                                                                             compute_index=False)

    # Merge tables
    metric_data = cell_metrics[[get_metric_index_name(cell_metric)]].merge(
        behavior_metrics_diff[[get_metric_index_name(behavior_metric)]], on='mouse_id')

    # Add metadata
    metric_data = metric_data.merge(platform_experiments_table[['mouse_id', 'cell_type']].drop_duplicates(), on='mouse_id')

    return metric_data


def plot_correlation_of_behavior_and_cell_metrics_delta(
        metric_data, behavior_metric, cell_metric,
        metric_label=None, behavior_label=None, suptitle=None,
        save_dir=None, folder=None):
    
    '''Plot the difference in metrics (F to N) for behavior vs. neural activity.'''
    x = get_metric_index_name(behavior_metric)
    y = get_metric_index_name(cell_metric)

    if metric_label is None:
        metric_label=cell_metric
    if behavior_label is None:
        behavior_label=behavior_metric  
    if suptitle is None:
        suptitle = f'Change in {metric_label} vs. change in {behavior_label}'

    cell_types = utils.get_cell_types()

    figsize = (14, 4)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    for i, cell_type in enumerate(cell_types):
        sns.scatterplot(data=metric_data[metric_data.cell_type == cell_type],
                        x=x, y=y, ax=ax[i])
        n_mice = metric_data[metric_data.cell_type == cell_type]['mouse_id'].nunique()
        ax[i].set_title(cell_type+'\n(n = '+str(n_mice)+' mice)')
        ax[i].set_xlabel(behavior_label+'\nChange from F to N')
        ax[i].set_ylabel(metric_label+'\nChange from F to N')
        ax[i].axhline(y=0, linestyle='--', color='gray', linewidth=1)
        ax[i].axvline(x=0, linestyle='--', color='gray', linewidth=1)
    plt.suptitle(suptitle, x=0.5, y=1.11, fontsize=20)
    plt.subplots_adjust(wspace=0.2)

    if save_dir:
        filename = _clean_filename('difference_' + cell_metric + '_' + behavior_metric)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))



def plot_modulation_index_correlation(data, x_col, y_col, xlabel, ylabel,
                            ax=None, save_dir=None, folder=None, filename=None):
    """
    Plots the correlation between two modulation indices (e.g., experience modulation index and change modulation index)
    for each cell type, including a linear best-fit line and Pearson correlation coefficient.

    Parameters:
    - data: DataFrame containing the modulation indices and cell types.
    - x_col: Column name for the x-axis modulation index.
    - y_col: Column name for the y-axis modulation index.
    - ax: Matplotlib axis to plot on (optional).
    - save_dir: Directory to save the figure (optional).
    - folder: Subfolder within save_dir to save the figure (optional).
    - filename: Name of the file to save the figure (optional).
    """

    import numpy as np
    from scipy.stats import pearsonr

    if ax is None: 
        figsize = (2, 8)
        fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)

    cell_types = utils.get_cell_types()
    for i, cell_type in enumerate(cell_types):
        cell_type_data = data[data.cell_type == cell_type]
        plot_data = cell_type_data[[x_col, y_col]].dropna()

        ax[i] = sns.kdeplot(data=plot_data, x=x_col, y=y_col, ax=ax[i])

        # Add linear best-fit line and Pearson correlation for the plotted x/y values.
        if len(plot_data) >= 2:
            x = plot_data[x_col].to_numpy()
            y = plot_data[y_col].to_numpy()

            if np.std(x) > 0 and np.std(y) > 0:
                r, p = pearsonr(x, y)
                slope, intercept = np.polyfit(x, y, 1)
                x_fit = np.array([x.min(), x.max()])
                y_fit = slope * x_fit + intercept
                p_str = f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"
                ax[i].plot(x_fit, y_fit, color='black', linestyle='--', linewidth=1.5)
                ax[i].set_title(f"{cell_type}\n(r={r:.2f}, p={p_str})")
            else:
                ax[i].set_title(f"{cell_type}\n(r=nan, p=nan)")
        else:
            ax[i].set_title(f"{cell_type}\n(r=nan, p=nan)")

        ax[i].set_ylabel('')

    ax[2].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)

    plt.subplots_adjust(hspace=0.4)
    if save_dir and folder:
        filename = _clean_filename(filename or f"{y_col}_vs_{x_col}_correlation")
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))


def plot_metric_across_exposures(metrics_table, metric, ylabel, plot_type='barplot', 
                               x_val='experience_level', x_vals=None, palette=None, 
                               suptitle=None, save_dir=None, folder=None, ax=None):
    '''
    Plot metric distributions across cre lines, with a unique axis for each cohort / project code, 
    experience levels as colors, and x-axis defined by x_val (such as 'binned_depth' or 'targeted_structure').
    Will plot stats across exp levels as an asterisk above that x value
    '''
    
    mdf = metrics_table.copy()
    cell_types = utils.get_cell_types()

    # x_val='experience_level'
    if x_vals == None: 
        x_vals = np.sort(df['x_val'].unique())
    if palette == None: 
        palette = sns.color_palette()

    i = 0 
    if ax is None:
        figsize=(len(x_vals)*1.35, 2)
        fig, ax = plt.subplots(1, 3, figsize=figsize)

    for c, cell_type in enumerate(cell_types): 
        data = mdf[mdf.cell_type==cell_type]

        if plot_type == 'pointplot': 
            ax[i] = sns.pointplot(data=data, x=x_val, y=metric, order=x_vals, hue=x_val, legend=False,
                                        palette=palette, hue_order=x_vals, dodge=0.3, linestyle='none',
                                        markers='.', markersize=8, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i])
        elif plot_type == 'barplot': 
            ax[i] = sns.barplot(data=data, x=x_val, y=metric, order=x_vals, hue=x_val, legend=False,
                                        palette=palette, hue_order=x_vals, width=0.5, alpha=0.75, 
                                        err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i])
        elif plot_type == 'boxplot': 
            ax[i] = sns.boxplot(data=data, x=x_val, y=metric, order=x_vals, hue=x_val, legend=False,
                                        palette=palette, hue_order=x_vals, width=0.5, fliersize=0, ax=ax[i])
            plt.setp(ax[i].collections, alpha=0.75)
        
        ax[i].set_ylabel('')
        ax[i].set_xlabel('')
        ax[i].set_xticks(np.arange(len(x_vals)))
        ax[i].set_xticklabels(x_vals, rotation=90)
        # ax[i].get_legend().remove()
        ax[i].set_title(cell_type)
        # ax[i], stats_table = ppf.add_stats_to_plot(data, metric, ax[i])
        # ax[i], stats_table = ppf.add_stats_to_plot_for_hues(data, metric, ax[i],
        #                                                 xorder=x_vals, x=x_val, hue='experience_level')
        # , ymax=None, show_ns=False)
        i+=1
    ax[0].set_ylabel(ylabel)

    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    if suptitle: 
        plt.suptitle(suptitle, x=0.5, y=1.15, fontsize=18)

    if save_dir: 
        filename = _clean_filename(metric+'_by_exp_novel_control')
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
    return ax


######### heatmaps ##########################


def plot_cell_response_heatmap(data, timestamps, xlabel='time after change (s)', vmax=0.05,
                               event_type=None, microscope='Multiscope', cbar=True, cbar_label='Response', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.heatmap(data, cmap='binary', linewidths=0, square=False,
                     vmin=0, vmax=vmax, robust=True, cbar=cbar,
                     cbar_kws={"drawedges": False, "shrink": 0.7, "label": cbar_label}, ax=ax)

    zero_index = np.where(timestamps == 0)[0][0]
    if event_type == None:
        color = 'gray'
        linestyle='--'
    elif event_type == 'omissions':
        color = sns.color_palette()[9]
        linestyle='--'
    elif event_type == 'changes':
        color = sns.color_palette()[0]
        linestyle='-'
    else: 
        color='gray'
        linestyle='-'
    ax.vlines(x=zero_index, ymin=0, ymax=len(data), color=color, linestyle=linestyle)

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


def find_cells_without_exactly_three_experience_levels(df, cell_id_col='cell_specimen_id',
                        experience_col='experience_level', expected_n=3):

    """Return cells that do not have exactly `expected_n` rows/experience levels."""

    summary = (df.groupby(cell_id_col)
        .agg(n_rows=(experience_col, 'size'),
            n_unique_experience_levels=(experience_col, 'nunique'),
            experience_levels_present=(experience_col, lambda s: sorted(s.dropna().unique().tolist())),
        ).reset_index())

    bad_cells = summary[(summary['n_rows'] != expected_n)
        | (summary['n_unique_experience_levels'] != expected_n)].copy()

    return bad_cells.sort_values(by=['n_unique_experience_levels', 'n_rows', cell_id_col]).reset_index(drop=True)


def plot_response_heatmaps_for_conditions(multi_session_df, timestamps, data_type, event_type,
                                          row_condition, col_condition, matched_cells_table=None, 
                                          plot_epochs=False, exp_to_match='Familiar',
                                          col_to_sort_by='mean_response', cell_order=None, suptitle=None,
                                          microscope=None, vmax=None, xlim_seconds=None, xlabel='time (s)',
                                          match_cells=False, cbar=True, cbar_label='Avg. calcium events',
                                          save_dir=None, folder=None, suffix='', ax=None):
    suffix = _norm_suffix(suffix)
    sdf = multi_session_df.copy()

    if xlim_seconds is None:
        xlim_seconds = (timestamps[0], timestamps[-1])

    row_conditions = np.sort(sdf[row_condition].unique())
    col_conditions = np.sort(sdf[col_condition].unique())

    if ax is None:
        # figsize = (2.5 * len(col_conditions), 3 * len(row_conditions))
        figsize = (2*len(col_conditions), 8)
        fig, ax = plt.subplots(len(row_conditions), len(col_conditions), figsize=figsize, sharex=True)
        ax = ax.ravel()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

    i = 0
    for r, row in enumerate(row_conditions):
        cre_sdf = sdf[(sdf[row_condition] == row)]

        # Get cell order based on novel session
        if match_cells:
            if matched_cells_table is None:
                print('Please provide the matched cells table to use match_cells=True') 
                matched_cells_this_ct = np.array([])
            else:
                matched_cells_this_ct = matched_cells_table[(matched_cells_table[row_condition] == row)].cell_specimen_id.unique()
            # Limit cre sdf to only matched cells for this cell type
            cre_sdf = cre_sdf[cre_sdf.cell_specimen_id.isin(matched_cells_this_ct)]
            # Find any cells that dont have exactly 3 exp levels and drop them
            drop_cells = find_cells_without_exactly_three_experience_levels(cre_sdf)
            cre_sdf = cre_sdf[cre_sdf.cell_specimen_id.isin(drop_cells.cell_specimen_id.unique())==False]
        
            # get cell order based on mean response in novel session
            # use groupby to get one mean_response per cell, ensuring unique ordering regardless of data structure
            if plot_epochs:
                novel_data = cre_sdf[(cre_sdf.experience_level == exp_to_match) & (cre_sdf.epoch==0)]
            else:
                novel_data = cre_sdf[(cre_sdf.experience_level == exp_to_match)]
            # to ensure that there are no duplicate cells messing with the ordering, average per cell first
            novel_cell_order = novel_data.groupby('cell_specimen_id')[col_to_sort_by].mean().sort_values().index.values

        for c, col in enumerate(col_conditions):
            exp_data = cre_sdf[(cre_sdf[col_condition] == col)]
            exp_data = exp_data.reset_index()

            if match_cells:
                exp_data = exp_data.set_index('cell_specimen_id')
                # use reindex to apply novel session cell order to all experience levels
                exp_data = exp_data.reindex(novel_cell_order)
            else: 
                if col_to_sort_by is not None: 
                    exp_data = exp_data.sort_values(by=col_to_sort_by, ascending=True)

            # if vmax == None: 
            if row == 'Excitatory':
                vmax = 0.0025
            elif row == 'Vip Inhibitory':
                vmax = 0.02
            elif row == 'Sst Inhibitory':
                vmax = 0.015
            else:
                vmax = 0.02

            if plot_epochs:
                vmax = vmax*3

            # turn responses it into a dataframe where columns are timestamps
            if plot_epochs:
                data = exp_data.reset_index().pivot(index=['cell_specimen_id'], columns='epoch', values='mean_response')
            else:
                data = pd.DataFrame(np.vstack(exp_data.mean_trace.values), columns=timestamps, index=exp_data.index.values)
            
            if match_cells:
                data = data.reindex(novel_cell_order)
            
            n_cells = len(data)
    
            ax[i] = plot_cell_response_heatmap(data, timestamps=timestamps, vmax=vmax, xlabel=xlabel, cbar=cbar,
                                               event_type=event_type, cbar_label=cbar_label, ax=ax[i])
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
            if not plot_epochs:
                if np.sum(np.abs(xlim_seconds)) < 2:
                    # set xticks to every 1 second, assuming 30Hz traces
                    ax[i].set_xticks(np.arange(0, len(timestamps), 15))  # assuming 30Hz traces
                    ax[i].set_xticklabels([t for t in timestamps[::15]], rotation=0, fontsize=14)
                else:
                    # set xticks to every 1 second, assuming 30Hz traces
                    ax[i].set_xticks(np.arange(0, len(timestamps), 30))  # assuming 30Hz traces
                    ax[i].set_xticklabels([int(t) for t in timestamps[::30]], rotation=0, fontsize=14)
                # set xlims according to input
                start_index = np.where(timestamps == xlim_seconds[0])[0][0]
                end_index = np.where(timestamps <= xlim_seconds[1])[0][-1]
                xlims = [start_index, end_index]
                ax[i].set_xlim(xlims)
            else:
                ax[i].set_xticks(timestamps[::2]+0.5)
                ax[i].set_xticklabels(timestamps[::2])
            ax[i].set_ylabel('')

            if (r == len(row_conditions) - 1) and (c == 1):
                ax[i].set_xlabel(xlabel)
            else:
                ax[i].set_xlabel('')
            sns.despine(ax=ax[i], top=False, right=False, left=False, bottom=False, offset=None, trim=False)
            i += 1
    for c, i in enumerate(np.arange(0, (len(col_conditions) * len(row_conditions)), len(col_conditions))):
        if match_cells: 
            ax[i].set_ylabel(str(row_conditions[c])+'\nmatched cells')
        else: 
            ax[i].set_ylabel(str(row_conditions[c])+'\ncells')

    # if suptitle:
    #     plt.suptitle(suptitle, x=0.52, y=1.0, fontsize=18)

    if save_dir:
        fig_title = _clean_filename(event_type + '_response_heatmap_' + data_type + '_' + col_condition + '_' + row_condition + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(fig_title))

    return ax

def plot_tuning_curve_heatmaps_for_conditions(multi_session_df, data_type, 
                                          row_condition, col_condition, vmax=None, 
                                          cbar=True, cbar_label='Mean response', title='',
                                          save_dir=None, folder=None, suffix='', ax=None):
    suffix = _norm_suffix(suffix)
    sdf = multi_session_df.copy()

    row_conditions = np.sort(sdf[row_condition].unique())
    col_conditions = np.sort(sdf[col_condition].unique())

    if ax is None:
        figsize = (4 * len(col_conditions), 5 * len(row_conditions))
        fig, ax = plt.subplots(len(row_conditions), len(col_conditions), figsize=figsize)
        ax = ax.ravel()
        
    colors = utils.get_experience_level_colors()

    i = 0
    for r, row in enumerate(row_conditions):
        cre_sdf = sdf[(sdf[row_condition] == row)]
        for c, col in enumerate(col_conditions):

            if row == 'Excitatory':
                vmax = 0.01
            elif row == 'Vip Inhibitory':
                vmax = 0.015
            elif row == 'Sst Inhibitory':
                vmax = 0.015
            else:
                vmax = 0.02

            exp_data = cre_sdf[(cre_sdf[col_condition] == col)]
            df = exp_data.copy()

            images = df.image_name.unique()
            csids = []
            for image in images: 
                tmp = df[(df.pref_stim==True) & (df.image_name==image)].drop_duplicates(subset=['cell_specimen_id'])
                tmp = tmp.sort_values(by='mean_response', ascending=False)
                cells_this_image = list(tmp.cell_specimen_id.values)
                csids = csids+cells_this_image
            pivot = df.pivot_table(index='cell_specimen_id', columns='image_name', values='mean_response')
            pivot = pivot.loc[csids]
            pivot = pivot[images]

            ax[i] = sns.heatmap(pivot.values, cmap='Greys', linewidths=0, linecolor='white', square=False,
                                vmin=0, vmax=vmax, robust=True, cbar=cbar,
                                cbar_kws={"drawedges": False, "shrink": 0.5, "label": cbar_label}, ax=ax[i])
            # make cbar fonts smaller
            # ax[i].figure.axes[-1].yaxis.label.set_size(12)

            if title:
                ax[i].set_title(title, va='bottom', ha='center')
            ax[i].set_xticks(np.arange(0, len(images), 1)+0.5)
            ax[i].set_xticklabels(images, rotation=90)
            ax[i].set_yticks((0, pivot.shape[0]));
            ax[i].set_yticklabels((0, pivot.shape[0]));

            if r == 0: 
                ax[i].set_title(str(col), color=colors[c]);

            sns.despine(ax=ax[i], top=False, right=False, left=False, bottom=False, offset=None, trim=False)
            i += 1
        

    for c, i in enumerate(np.arange(0, (len(col_conditions) * len(row_conditions)), len(col_conditions))):
        ax[i].set_ylabel(str(row_conditions[c])+' cells');

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('tuning_curve_heatmaps_across_experience_'+data_type+suffix))

    return ax


def plot_population_tuning_curve_for_conditions(multi_session_df, data_type, 
                                          row_condition, col_condition, ylabel='Normalized response',
                                          save_dir=None, folder=None, suffix='', ax=None):
    suffix = _norm_suffix(suffix)
    sdf = multi_session_df.copy()

    row_conditions = np.sort(sdf[row_condition].unique())
    col_conditions = np.sort(sdf[col_condition].unique())

    if ax is None:
        figsize = (1 * len(col_conditions), 3 * len(row_conditions))
        fig, ax = plt.subplots(len(row_conditions), 1, figsize=figsize, sharey=True)
        ax = ax.ravel()
        
    colors = utils.get_experience_level_colors()

    i = 0
    for r, row in enumerate(row_conditions):
        cre_sdf = sdf[(sdf[row_condition] == row)]
        for c, col in enumerate(col_conditions):

            if row == 'Excitatory':
                vmax = 0.01
            elif row == 'Vip Inhibitory':
                vmax = 0.015
            elif row == 'Sst Inhibitory':
                vmax = 0.015
            else:
                vmax = 0.02

            exp_data = cre_sdf[(cre_sdf[col_condition] == col)]
            df = exp_data.copy()

            images = df.image_name.unique()
            csids = []
            for image in images: 
                tmp = df[(df.pref_stim==True) & (df.image_name==image)].drop_duplicates(subset=['cell_specimen_id'])
                tmp = tmp.sort_values(by='mean_response', ascending=False)
                cells_this_image = list(tmp.cell_specimen_id.values)
                csids = csids+cells_this_image
            pivot = df.pivot_table(index='cell_specimen_id', columns='image_name', values='mean_response')
            pivot = pivot.loc[csids]
            pivot = pivot[images]

            # gather TCs and plot each cell
            all_tcs = []
            for csid in pivot.index.values:
                tc = pivot.loc[csid]
                tc = tc.values/tc.max()
                tc = np.sort(tc)[::-1]
                all_tcs.append(tc)
                # ax[i].plot(np.arange(0, len(images), 1), tc, color='gray', linewidth=0.5, linestyle='-')

            # average over population
            tc = np.nanmean(all_tcs, axis=0)
            sem = np.nanstd(all_tcs, axis=0) / np.sqrt(len(all_tcs))
            ax[i].plot(np.arange(0, len(images), 1), tc, color=colors[c], linewidth=1.5, linestyle='-', label=col)
            ax[i].fill_between(np.arange(0, len(images), 1), tc + sem, tc - sem, alpha=0.4, color=colors[c])

        ax[i].set_xticks(np.arange(0, len(images), 1))
        ax[i].set_xticklabels(np.arange(0, len(images), 1))
        if i < 2: 
            ax[i].set_xticklabels('')
        # ax[i].set_yticks((0, pivot.shape[0]));
        # ax[i].set_yticklabels((0, pivot.shape[0]));

        ax[i].set_title(str(row));
        ax[i].set_ylim(-0.05, 1.1)
        
        if r == 2:
            ax[i].set_xlabel('Sorted image ID')
        

        sns.despine(ax=ax[i], top=False, right=False, left=False, bottom=False, offset=None, trim=False)
        i += 1


    ax[0].legend(fontsize='xx-small')
    ax[1].set_ylabel(ylabel)
    
    # for c, i in enumerate(np.arange(0, (len(col_conditions) * len(row_conditions)), len(col_conditions))):
    #     ax[i].set_ylabel(str(row_conditions[c]));
    

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('population_tuning_curves_'+data_type+suffix))

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

    sns.despine(ax=ax, top=True, bottom=False, right=True, left=True)
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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metadata_string + '_' + str(int(start_time))))
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
    start_inds = np.where(running_timestamps < xlim_seconds[0])[0]
    start_ind = start_inds[-1] if len(start_inds) > 0 else 0
    stop_inds = np.where(running_timestamps > xlim_seconds[1])[0]
    stop_ind = stop_inds[0] if len(stop_inds) > 0 else len(running_timestamps)
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
    start_inds = np.where(pupil_timestamps < xlim_seconds[0])[0]
    start_ind = start_inds[-1] if len(start_inds) > 0 else 0
    stop_inds = np.where(pupil_timestamps > xlim_seconds[1])[0]
    stop_ind = stop_inds[0] if len(stop_inds) > 0 else len(pupil_timestamps)
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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metadata_string + '_' + str(int(start_time)) + '_' + suffix))
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
    start_inds = np.where(running_timestamps < xlim_seconds[0])[0]
    start_ind = start_inds[-1] if len(start_inds) > 0 else 0
    stop_inds = np.where(running_timestamps > xlim_seconds[1])[0]
    stop_ind = stop_inds[0] if len(stop_inds) > 0 else len(running_timestamps)
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
    start_inds = np.where(pupil_timestamps < xlim_seconds[0])[0]
    start_ind = start_inds[-1] if len(start_inds) > 0 else 0
    stop_inds = np.where(pupil_timestamps > xlim_seconds[1])[0]
    stop_ind = stop_inds[0] if len(stop_inds) > 0 else len(pupil_timestamps)
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

    ax[9].plot(reward_timestamps, rewards, '^', label='rewards', color='blue', markersize=10)
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
        utils.save_figure(fig, figsize, save_dir, folder,
                          _clean_filename(metadata_string + '_' + str(int(start_time)) + '_' + suffix),
                          formats=['.png', '.pdf'])
    return ax

# ### matched cell plots ####


def plot_matched_roi_and_trace(ophys_container_id, cell_specimen_id, limit_to_last_familiar_second_novel=True,
                               use_events=False, filter_events=False, linewidth=1, save_figure=True):
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
                                                  linewidth=linewidth, xlim_seconds=window, plot_sem=True, ax=ax[i + n])

                ax[i + n] = utils.plot_flashes_on_trace(ax[i + n], cell_data.trace_timestamps.values[0], change=True, omitted=False,
                                                        facecolor='gray')
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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(str(cell_specimen_id) + '_' + metadata_string + '_' + data_type))
        plt.close()


def plot_matched_roi_and_traces_example(cell_metadata, include_omissions=True,
                                        use_events=False, filter_events=False, linewidth=1,
                                        save_dir=None, folder=None):
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
                                                    ylabel=ylabel, legend_label=None, color=color, interval_sec=1, linewidth=linewidth,
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
                                                            interval_sec=1, linewidth=linewidth,
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
                          _clean_filename(str(cell_specimen_id) + '_' + metadata_string + '_' + suffix))
        plt.close()


########## behavior plots - figure 1 #############

def plot_behavior_metric_by_experience(stats, metric, title='', ylabel='', ylims=None, best_image=False, show_mice=False,
                                       stripplot=True, pointplot=True, plot_stats=False, show_ns=False,
                                       abbreviate_exp=True, save_dir=None, folder=None, suffix='', ax=None,
                                       use_mlm=True, group_column='mouse_id'):
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
    suffix = _norm_suffix(suffix)
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

    if metric == 'mean_dprime_engaged':
        data = data[data[metric] > 0]

    colors = utils.get_experience_level_colors()
    # experience_levels = utils.get_experience_levels()
    experience_levels = np.sort(data.experience_level.unique())

    if ax is None:
        figsize = (2, 3)
        figsize = (1.75, 2.5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if stripplot:
        ax = sns.stripplot(data=data, x='experience_level', y=metric, order=experience_levels,
                       orient='v', color='gray', dodge=True, jitter=0.1, size=2, ax=ax, zorder=0)
    if show_mice:
        for mouse_id in data.mouse_id.unique():
            ax = sns.pointplot(data=data[data.mouse_id == mouse_id], x='experience_level', y=metric,
                               order=experience_levels, linewidth=0.5, orient='v', color='gray',
                               markers='.', markersize=0.15, err_kws={'linewidth': 0.5}, ax=ax)
        # suffix = suffix + '_show_mice'

    if pointplot:
        ax = sns.pointplot(data=data, x='experience_level', y=metric, order=experience_levels,
                       orient='v', palette=colors, ax=ax,
                       markers='.', markersize=8, err_kws={'linewidth': 2},)
    else:
        ax = sns.boxplot(data=data, x='experience_level', y=metric, order=experience_levels,
                           orient='v', palette=colors, width=0.6, boxprops=dict(alpha=0.8), ax=ax)

    ax.set_xlim(-0.5, len(experience_levels)-0.5)
    if abbreviate_exp:
        ax.set_xticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
        utils.color_xaxis_labels_by_experience(ax)
    else:
        ax.set_xticks(ax.get_xticks().tolist())
        ax.set_xticklabels(experience_levels, rotation=90, )  # ha='right')
        utils.color_xaxis_labels_by_experience(ax)
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
        ax, stats_table = add_stats_to_plot(data, metric, ax, ymax=ymax, show_ns=show_ns, behavior=True,
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type='behavior')
        stats_table = insert_stats_metadata(stats_table, condition='experience_level')

    ax.set_ylim(ymin=ymin)
    plt.subplots_adjust(top=0.96)
    plt.subplots_adjust(hspace=0.3)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metric + suffix))
    stats_filename = _clean_filename(metric + '_stats' + suffix)
    try:
        if plot_stats:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
        # save metric values
        cols_to_groupby = ['experience_level']
        descriptive_stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        descriptive_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
    except BaseException:
        print('stats did not save for', metric)
    return ax


def plot_behavior_metric_by_experience_horiz(stats, metric, title='', xlabel='', xlims=None, best_image=True, show_containers=False,
                                       stripplot=True, pointplot=True, plot_stats=False, show_ns=False, save_dir=None, folder=None, suffix='', ax=None,
                                       use_mlm=True, group_column='mouse_id'):
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
    suffix = _norm_suffix(suffix)
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
    
    if metric == 'mean_dprime_engaged':
        data = data[data[metric] > 0]

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
        # suffix = suffix + '_show_mice'

    if pointplot:
        ax = sns.pointplot(data=data, y='experience_level', x=metric, order=experience_levels,
                        markers='.', markersize=8, err_kws={'linewidth': 2}, orient='h', palette=colors, ax=ax)
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
        ax, stats_table = add_stats_to_plot(data, metric, ax, ymax=xmax, show_ns=show_ns, behavior=True,
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type='behavior')
        stats_table = insert_stats_metadata(stats_table, condition='experience_level')

    ax.set_xlim(xmin=xmin)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metric + '_horiz' + suffix))
    stats_filename = _clean_filename(metric + '_stats' + suffix)
    try:
        if plot_stats:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
        # save metric values
        cols_to_groupby = ['experience_level']
        descriptive_stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        descriptive_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
    except BaseException:
        print('stats did not save for', metric)
    return ax



def plot_behavior_metric_by_cohort(stats, metric, title='', ylabel='', ylims=None, show_containers=False,
                                       stripplot=True, pointplot=True, plot_stats=False, show_ns=False,
                                   save_dir=None, folder=None, suffix='', ax=None,
                                   use_mlm=True, group_column='mouse_id'):
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
    suffix = _norm_suffix(suffix)
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

    if metric == 'mean_dprime_engaged':
        data = data[data[metric] > 0]

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
        # suffix = suffix + '_show_mice'

    if pointplot:
        ax = sns.pointplot(data=data, x='project_code', y=metric, order=project_codes,
                       orient='v', palette=colors, markers='.', ax=ax) # marker_kws={'size':2},
    else:
        ax = sns.boxplot(data=data, x='project_code', y=metric, order=project_codes,
                           orient='v', palette=colors, width=0.6, boxprops=dict(alpha=0.8), ax=ax)

    ax.set_xlim(-0.5, len(project_codes)-0.5)
    ax.set_xticklabels([cohort[-1] for cohort in cohorts], rotation=0)
    ax.set_xlabel('Cohort')
    ax.set_ylabel('')
    exp_colors = utils.get_experience_level_colors()
    experience_levels = utils.get_experience_levels()
    match = np.where(np.asarray(experience_levels) == title)[0]
    if len(match) > 0:
        ax.set_title(title, color=exp_colors[match[0]])
    else:
        ax.set_title(title)
    if ylabel is None:
        ylabel = metric
    ax.set_ylabel(ylabel)
    # ax.legend(bbox_to_anchor=(1,1), fontsize='xx-small')

    # add stats to plot if only looking at experience levels
    if plot_stats:
        # stats dataframe to save
        ax, stats_table = add_stats_to_plot(data, metric, ax, ymax=ymax, show_ns=show_ns, behavior=True,
                                            column_to_compare='project_code',
                                            use_mlm=use_mlm, group_column=group_column,
                                            event_type='behavior')
        stats_table = insert_stats_metadata(stats_table, condition='experience_level')

    ax.set_ylim(ymin=ymin)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metric + suffix))
    stats_filename = _clean_filename(metric + '_stats' + suffix)
    try:
        if plot_stats:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
        # save metric values
        cols_to_groupby = ['project_code']
        descriptive_stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        descriptive_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
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
    suffix = _norm_suffix(suffix)
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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('metric_across_stages_' + metric + suffix))
        # save stats
        stats = data.groupby(['cell_type', 'behavior_stage']).describe()[[metric]]
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename('metric_across_stages_' + metric + suffix + '_values.csv')))


def plot_days_in_stage(behavior_sessions, stage_column, save_dir=None, folder=None, suffix=None):
    """
    Plot the number of days in each stage, as a boxplot using stage as the hue and cell types on y-axis

    behavior_sessions: behavior sessions table including 'cell_type', 'mouse_id'
    stage_column: column in behavior_sessions to use for grouping of stages, can be 'behavior_stage', 'stimulus_phase', or 'session_type',
    to add 'behavior_stage' column to behavior_sessions, use add_behavior_stage_to_behavior_sessions(behavior_sessions)

    """
    suffix = _norm_suffix(suffix)

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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('days_in_stage' + '_' + stage_column + suffix))
        # save stats
        days_in_stage_stats = data.groupby(['cell_type', stage_column]).describe()
        days_in_stage_stats.to_csv(os.path.join(save_dir, folder, _clean_filename('days_in_stage_stats.csv')))


def plot_prior_exposures_to_image_set_before_platform_ophys_sessions(platform_experiments, behavior_sessions, save_dir=None, folder=None, suffix='', ax=None):
    """
    Creates a boxplot showing the number of prior exposures to each image set for each experience level (Familiar, Novel, Novel +)
    for the set of mice and sessions in platform_experiments
    Boxplot is distribution of number of prior exposures across mice
    """
    suffix = _norm_suffix(suffix)

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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('stimulus_exposures_before_platform_expts_boxplot' + suffix))
        # save stats
        stats = exposures.groupby(['experience_level']).describe()[['prior_exposures_to_image_set']]
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename('stimulus_exposures_before_platform_expts_stats.csv')))


def plot_prior_exposures_per_cell_type_for_novel_plus(platform_experiments, behavior_sessions, save_dir=None,
                                                      folder=None, suffix='', ax=None,
                                                      use_mlm=True, group_column='mouse_id'):
    """
    Creates a boxplot showing the number of  prior exposures to novel image set for Novel + sessions included in the platform paper
    shows striplot of prior exposures across mice and pointplot of averages plus stats
    """
    suffix = _norm_suffix(suffix)

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
    ax, stats_table = add_stats_to_plot(exposures, 'prior_exposures_to_image_set', ax, ymax=ymax,
                                        show_ns=True, column_to_compare='cell_type',
                                        use_mlm=use_mlm, group_column=group_column,
                                        event_type='session_metadata')
    stats_table = insert_stats_metadata(stats_table, condition='experience_level')

    if save_dir:
        # save plot
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('stimulus_exposures_before_novel_plus' + suffix))
        # save stats
        print('saving_stats')
        stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
        stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename('stimulus_exposures_before_novel_plus' + stats_suffix)))
        descriptive_stats = exposures.groupby(['cell_type', 'experience_level']).describe()[['prior_exposures_to_image_set']]
        descriptive_stats.to_csv(os.path.join(save_dir, folder, _clean_filename('stimulus_exposures_before_novel_plus_stats.csv')))


def plot_prior_exposures_to_image_set_before_platform_ophys_sessions_horiz(platform_experiments, behavior_sessions, save_dir=None, folder=None, suffix='', ax=None):
    """
    Creates a boxplot showing the number of prior exposures to each image set for each experience level (Familiar, Novel, Novel +)
    for the set of mice and sessions in platform_experiments
    Boxplot is distribution of number of prior exposures across mice
    """
    suffix = _norm_suffix(suffix)

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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('stimulus_exposures_before_platform_expts_boxplot_horiz' + suffix))
        # save stats
        stats = exposures.groupby(['experience_level']).describe()[['prior_exposures_to_image_set']]
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename('stimulus_exposures_before_platform_expts_stats.csv')))

def plot_total_stimulus_exposures(behavior_sessions, save_dir=None, folder=None, suffix='', ax=None):
    """
    Creates a boxplot showing the number of sessions for each experience level (Familiar, Novel, Novel +)
    for the set of mice included in behavior_sessions
    """
    suffix = _norm_suffix(suffix)

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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('total_stimulus_exposures_all_sessions_boxplot' + suffix))
        # save stats
        stats = exposures.groupby(['experience_level']).describe()[['n_sessions']]
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename('total_stimulus_exposures_all_sessions_stats.csv')))


def plot_stimulus_exposure_prior_to_imaging(behavior_sessions, column_to_group='behavior_stage',
                                            save_dir=None, folder=None, suffix='', ax=None):
    """
    Creates a boxplot showing the number of sessions for each experience level or session type
    prior to the start of 2P imaging, for the set of mice included in behavior_sessions
    """
    suffix = _norm_suffix(suffix)

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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('stimulus_exposure_prior_to_imaging_boxplot_'+ column_to_group + suffix))
        # save stats
        stats = exposures.groupby(column_to_group).describe()[['n_sessions']]
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename('stimulus_exposure_prior_to_imaging_stats_'+column_to_group+'.csv')))

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
    suffix = _norm_suffix(suffix)

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
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('training_history' + suffix))
    return ax


def plot_ophys_history_for_mice(behavior_sessions, color_column='ophys_stage', color_map=sns.color_palette(),
                                group_by_cre=False, label_rows_by_cre=False, label_with_mouse_id=False,
                                sort_by_n_sessions=True, title=None, suffix='', save_dir=None, ax=None):
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
    suffix = _norm_suffix(suffix)
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
    if sort_by_n_sessions:
        mouse_ids = n_sessions.mouse_id.values
    else:
        mouse_ids = np.sort(n_sessions.mouse_id.unique())

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
        utils.save_figure(fig, figsize, save_dir, 'training_history', _clean_filename('ophys_session_sequence' + suffix))

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
    ax = sns.pointplot(data=data, x=x, y=metric, hue=hue, hue_order=data[hue].unique(), linestyle='None', palette=colors, ax=ax)
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
        if os.path.exists(os.path.join(save_dir, folder)) == False:
            os.makedirs(os.path.join(save_dir, folder))
        utils.save_figure(fig, figsize, os.path.join(save_dir, folder), 'behavior_performance_over_time_'+method,
                        _clean_filename(metric+'_mouse_id_'+str(mouse_id)+'_'+hue))
    return ax


def plot_response_rate_trial_types(data, save_dir=None, suffix='', ax=None):
    '''
    plot response rate for all sessions in data, with trial types on the x-axis
    examples of trial types: change vs non change, or go vs catch, or hit vs FR
    Trial types shown depend on values in 'trial_types' column of input data

    data: dataframe of behavior statistics where every row is one behavior session and the columns are performance metrics
        should include column "response_probability"
    '''
    suffix = _norm_suffix(suffix)
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
        utils.save_figure(fig, figsize, save_dir, 'response_rate', _clean_filename('response_rate_trial_types' + suffix))
    return ax



def plot_lick_raster_for_trials(trials, title='', legend=False, save_dir=None, filename=None, suffix='', ax=None):
    # trials = dataset.trials
    # image_set = dataset.metadata.session_type.values[0][-1]
    # mouse_id = str(dataset.metadata.donor_id.values[0])
    suffix = _norm_suffix(suffix)
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
        # ax.vlines(trial_start, trial, trial + 1, color='gray', linewidth=0.5, alpha=0.5)
        # plot line at trial start
        # ax.vlines(0, trial, trial + 1, color='gray', linewidth=1, linestyle='--')
        # plot licks
        lick_times = [(t - trial_data.change_time) for t in trial_data.lick_times]
        lick_times = [t for t in lick_times if t<4]
        if len(lick_times) > 0:
            for lick_time in lick_times: 
                ax.plot(lick_time, trial, color='k', label='licks', marker='|', markersize=3, linestyle='None')
            # ax.vlines(lick_time, trial, trial + 1, color='k', linewidth=1, label='licks')
        # plot rewards
        if np.isnan(trial_data.reward_time) == False:
            reward_time = trial_data.reward_time - trial_data.change_time
            ax.plot(reward_time, trial + 0.5, 'o', color='blue', label='rewards', markersize=1)
    # plot reward window
    color = sns.color_palette()[0]
    # ax.axvspan(0, 0.75, facecolor='gray', alpha=.4, edgecolor='none')
    ax.axvspan(0, 0.25, facecolor=color, alpha=.4, edgecolor='none')
    ax.grid(False)
    ax.set_ylim(0, len(trials))
    ax.set_xlim([-1, 5])
    ax.set_ylabel('Trial number')
    ax.set_xlabel('Time from change (sec)')
    ax.set_xticks(np.arange(0, 5, 2))
    ax.set_title(title, fontsize=16)
    ax.invert_yaxis()
    if legend: 
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        leg = ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc='upper left', fontsize='x-small')
        h = leg.legend_handles
        h[0].set_markersize(12)
        h[1].set_markersize(6)
    plt.subplots_adjust(left=0.3)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, 'lick_rasters', _clean_filename(filename+suffix))
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
        utils.save_figure(fig, figsize, save_dir, 'response_probability', _clean_filename('response_probability_heatmaps_engaged_only'))

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
    cache_dir = loading.get_sdk_cache_dir()
    cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=cache_dir, use_static_cache=True)
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


############### GLM coding score / feature fraction plots (figure 4) ###############


def compute_feature_coding_fractions(results_pivoted, run_params, coding_thresh=0.1,
                                     exclude_passive=True):
    """
    For each cre_line, compute the fraction of cells whose coding score for each
    main feature ('all-images', 'omissions', 'behavioral', 'task') exceeds
    `coding_thresh`, restricted to cells that 'code anything' (variance_explained_full
    > run_params['dropout_threshold']).

    Returns a long-form dataframe with columns:
        cre_line, cell_type, feature, fraction, percent, <feature>_ci
    where feature labels are: images, omissions, behavior, task.
    """
    df = results_pivoted.copy()
    if exclude_passive and 'passive' in df.columns:
        df = df.query('not passive').copy()

    df['code_anything'] = df['variance_explained_full'] > run_params['dropout_threshold']
    df['code_images'] = df['code_anything'] & (df['all-images'] > coding_thresh)
    df['code_omissions'] = df['code_anything'] & (df['omissions'] > coding_thresh)
    df['code_behavioral'] = df['code_anything'] & (df['behavioral'] > coding_thresh)
    df['code_task'] = df['code_anything'] & (df['task'] > coding_thresh)

    code_cols = ['code_anything', 'code_images', 'code_omissions', 'code_behavioral', 'code_task']
    summary_df = df.groupby(['cre_line'])[code_cols].mean()
    summary_df['n'] = df.groupby(['cre_line'])[code_cols].count()['code_anything']
    for feat in ['images', 'omissions', 'behavioral', 'task']:
        p = summary_df['code_' + feat]
        summary_df['code_' + feat + '_ci'] = 1.96 * np.sqrt((p * (1 - p)) / summary_df['n'])

    fractions = summary_df.drop(columns=['code_anything', 'n']).copy()
    fractions.columns = [c.split('_')[1] if 'ci' not in c else c.split('_')[1] + '_' + c.split('_')[2]
                         for c in fractions.columns]
    fractions = fractions.rename(columns={'behavioral': 'behavior',
                                          'behavioral_ci': 'behavior_ci'})
    fractions = fractions.reset_index()
    fractions = fractions.melt(id_vars=['cre_line'], var_name='feature', value_name='fraction')
    fractions['cell_type'] = [utils.convert_cre_line_to_cell_type(c) for c in fractions.cre_line.values]
    fractions['percent'] = fractions['fraction'] * 100
    return fractions


def plot_percent_cells_coding_for_features(fractions, save_dir=None, folder='coding_properties',
                                           filename='percent_cells_coding_feature', suffix=''):
    """
    Bar plot of the percent of cells coding for each feature, one panel per cell type.

    Expects the long-form `fractions` dataframe produced by
    `compute_feature_coding_fractions`.
    """
    from visual_behavior.dimensionality_reduction.clustering import plotting
    from visual_behavior.dimensionality_reduction.clustering import processing as processing

    cell_types = utils.get_cell_types()
    features = processing.get_feature_labels_for_clustering()
    feature_colors, _ = plotting.get_feature_colors_and_labels()

    figsize = (8, 2.5)
    fig, ax = plt.subplots(1, len(cell_types), figsize=figsize, sharey=True, sharex=True)
    ax = ax.ravel()

    for i, cell_type in enumerate(cell_types):
        ct_data = fractions[fractions.cell_type == cell_type]
        ax[i] = sns.barplot(data=ct_data, x='feature', y='percent', order=features,
                            palette=feature_colors, width=0.8, alpha=0.75, ax=ax[i])
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].set_ylim(0, 100)
        if i == 0:
            ax[i].set_ylabel('Percent of cells\ncoding for feature')
        ax[i].set_title(cell_type)

        for x, feature in enumerate(features):
            pct = ct_data[ct_data['feature'] == feature].percent.values[0]
            ax[i].text(s=str(np.round(pct, 1)), y=pct, x=x, rotation=0, fontsize=10,
                       color='k', va='bottom', ha='center')

        ax[i].set_xticklabels(features, rotation=45, ha='right', fontsize=14)
        [t.set_color(c) for (c, t) in zip(feature_colors[:len(features)], ax[i].xaxis.get_ticklabels())]

    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, filename + suffix)
    return fig, ax


def compute_maximally_contributing_feature(results_pivoted, cells_table):
    """
    For each cell (averaged across sessions), identify the main feature with the
    largest absolute coding score, then count what fraction of cells in each
    cre line max out on each feature.

    Returns a long-form dataframe with columns: cre_line, max_feature, fraction, percent.
    """
    from visual_behavior.dimensionality_reduction.clustering import plotting
    from visual_behavior.dimensionality_reduction.clustering import processing as processing

    feature_cols = processing.get_features_for_clustering()

    results_avg = results_pivoted.groupby(['cre_line', 'cell_specimen_id']).mean().reset_index()
    df = results_avg.set_index(['cell_specimen_id'])[feature_cols]
    df = np.abs(df)

    max_feature = pd.DataFrame(df.index.values, columns=['cell_specimen_id']).set_index('cell_specimen_id')
    max_feature['max_feature'] = None
    for csid in df.index.values:
        max_feature.loc[csid, 'max_feature'] = df.loc[csid].idxmax()
    max_feature = max_feature.merge(cells_table[['cell_specimen_id', 'cre_line']], on=['cell_specimen_id'])

    feature_counts = plotting.get_fraction_cells_for_column(max_feature, column_to_group='max_feature')
    feature_counts['percent'] = feature_counts.fraction * 100
    return feature_counts


def plot_maximally_contributing_feature_distribution(feature_counts, save_dir=None,
                                                     folder='coding_properties',
                                                     filename='maximally_contributing_feature_dimension',
                                                     suffix=''):
    """
    Stacked bar plot showing the distribution of each cre line's maximally
    contributing GLM feature. `feature_counts` is the output of
    `compute_maximally_contributing_feature`.
    """
    from visual_behavior.dimensionality_reduction.clustering import plotting
    from visual_behavior.dimensionality_reduction.clustering import processing as processing

    feature_colors, _ = plotting.get_feature_colors_and_labels()
    feature_order = processing.get_features_for_clustering()
    # legend ends up showing colors in reverse order
    feature_labels = processing.get_feature_labels_for_clustering()[::-1]

    figsize = (3.5, 2.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.histplot(feature_counts, y='cre_line', hue='max_feature', hue_order=feature_order,
                      weights='percent', multiple='stack', palette=feature_colors,
                      shrink=0.8, alpha=0.75, ax=ax)
    ax.set_yticklabels([utils.get_abbreviated_cell_type(cre_line) for cre_line in utils.get_cre_lines()])
    ax.legend(feature_labels, bbox_to_anchor=(1, 1))
    ax.set_xlabel('Percent of cells')
    ax.set_ylabel('')
    ax.set_title('Maxmially contributing\nfeature distribution')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, filename + suffix)
    return fig, ax


def convert_coding_scores_to_long_form_df(results_pivoted):
    """
    Reshape `results_pivoted` into a long-form dataframe of (absolute) coding
    scores for the four main features, keyed by cell / experiment / cre line /
    experience level. Used by `plot_coding_score_distributions_by_experience`.

    Carries `cell_type` and (if present) `mouse_id` through so they can be used
    for stats grouping / descriptive tables downstream.
    """
    from visual_behavior.dimensionality_reduction.clustering import processing

    features = processing.get_feature_labels_for_clustering()
    metadata = ['cell_specimen_id', 'ophys_experiment_id', 'cre_line', 'experience_level']
    for extra in ('cell_type', 'mouse_id'):
        if extra in results_pivoted.columns and extra not in metadata:
            metadata.append(extra)

    results_melted = results_pivoted.copy()
    results_melted['images'] = results_melted['all-images']
    results_melted['behavior'] = results_melted['behavioral']
    if 'cell_type' not in results_melted.columns:
        results_melted['cell_type'] = [utils.convert_cre_line_to_cell_type(c)
                                       for c in results_melted.cre_line.values]
        metadata.append('cell_type')

    results_melted = results_melted[features + metadata]
    results_melted = results_melted.melt(id_vars=metadata, value_vars=features,
                                         var_name='feature', value_name='coding_score')
    results_melted['coding_score'] = np.abs(results_melted['coding_score'])
    return results_melted


def plot_coding_score_distributions_by_experience(results_melted, cre_lines=None,
                                                  save_dir=None, folder='coding_scores_and_kernels',
                                                  filename='coding_score_distributions_by_experience',
                                                  suffix='', use_mlm=True, group_column='mouse_id',
                                                  event_type='coding_score'):
    """
    Grid of boxplots (rows: cre line, cols: feature) of coding scores by
    experience level, with significance annotations. Expects the long-form
    output of `convert_coding_scores_to_long_form_df`.

    Stats tables are saved alongside the figure when `save_dir` is provided:
      <filename>_mlm.csv (or _tukey.csv) -- pairwise stats across experience
          levels, with metric/event_type/cell_type/feature columns prepended.
      <filename>_values.csv -- descriptive stats grouped by
          (cell_type, feature, experience_level).
    """
    from visual_behavior.dimensionality_reduction.clustering import plotting
    from visual_behavior.dimensionality_reduction.clustering import processing as processing

    if cre_lines is None:
        cre_lines = np.sort(results_melted.cre_line.unique())
    features = processing.get_feature_labels_for_clustering()
    experience_levels = utils.get_experience_levels()
    experience_level_colors = utils.get_experience_level_colors()
    feature_colors, _ = plotting.get_feature_colors_and_labels()

    figsize = (7, 8)
    fig, ax = plt.subplots(len(cre_lines), len(features), figsize=figsize, sharey=True)
    ax = ax.ravel()
    combined_stats = pd.DataFrame()
    i = 0
    for c, cre_line in enumerate(cre_lines):
        cell_type = utils.convert_cre_line_to_cell_type(cre_line)
        for f, feature in enumerate(features):
            sub = results_melted[(results_melted.cre_line == cre_line)
                                 & (results_melted.feature == feature)].copy()

            ax[i] = sns.boxplot(data=sub, x='experience_level', y='coding_score',
                                order=experience_levels, hue='experience_level',
                                hue_order=experience_levels, palette=experience_level_colors,
                                legend=False, linewidth=1.5, width=0.7, fliersize=0,
                                whis=1.5, notch=True,
                                flierprops=dict(markerfacecolor='0.75', markersize=3, marker='_',
                                                linestyle='none', markeredgecolor='0.75'),
                                boxprops={'alpha': 0}, ax=ax[i])
            ax[i] = sns.boxplot(data=sub, x='experience_level', y='coding_score',
                                order=experience_levels, hue='experience_level',
                                hue_order=experience_levels, legend=False,
                                palette=experience_level_colors, notch=True,
                                medianprops={"color": "k", "linewidth": 2},
                                linewidth=1.5, width=0.8, showfliers=False, whis=1.5, dodge=False,
                                boxprops={'alpha': 0.8}, ax=ax[i])

            ax[i].set_ylabel('')
            ax[i].set_xlabel('')
            ax[i].set_xticklabels([])
            if c == 0:
                ax[i].set_title(feature.capitalize() + '\n', color=feature_colors[f])
            if f == 0:
                if c == 1:
                    ax[i].set_ylabel('Coding score\n\n' + cell_type)
                else:
                    ax[i].set_ylabel(cell_type)

            if c == len(cre_lines) - 1:
                exp_level_abbreviations = [
                    el.split(' ')[0][0] if len(el.split(' ')) == 1
                    else el.split(' ')[0][0] + el.split(' ')[1][:2]
                    for el in experience_levels
                ]
                ax[i].set_xticklabels(exp_level_abbreviations)
                plotting.color_xaxis_labels_by_experience(ax[i])

            ax[i].set_ylim(-0.1, 1.1)
            ax[i].set_xlim(-0.75, 2.75)

            ax[i], panel_stats = add_stats_to_plot_yaxis(
                sub, 'coding_score', ax=ax[i], ymax=1.05,
                column_to_compare='experience_level',
                use_mlm=use_mlm, group_column=group_column,
                event_type=event_type, cell_type=cell_type,
            )
            # Attach the per-panel iteration variable (feature) that compute_stats
            # can't see on its own.
            panel_stats = insert_stats_metadata(panel_stats, feature=feature)
            combined_stats = pd.concat([combined_stats, panel_stats])

            ax[i].set_ylim(-0.1, 1.2)
            i += 1

    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, filename + suffix)
        try:
            print('saving_stats')
            stats_suffix = '_mlm.csv' if use_mlm else '_tukey.csv'
            combined_stats.to_csv(
                os.path.join(save_dir, folder, _clean_filename(filename + suffix + stats_suffix))
            )
            cols_to_groupby = ['cell_type', 'feature', 'experience_level']
            descriptive = get_descriptive_stats_for_metric(results_melted, 'coding_score', cols_to_groupby)
            descriptive.to_csv(
                os.path.join(save_dir, folder, _clean_filename(filename + suffix + '_values.csv'))
            )
        except BaseException:
            print('STATS DID NOT SAVE FOR coding_score')
    return fig, ax, combined_stats


def plot_coding_scores_across_conditions_grid(data, x_val='experience_level', hue='targeted_structure',
                                              plot_type='barplot', xlabel='', figsize=None,
                                              event_type='coding_score',
                                              save_dir=None, folder='coding_scores_and_kernels',
                                              filename=None, suffix='',
                                              use_mlm=True, group_column='mouse_id'):
    """
    Grid of coding score bar/box/point plots: 3 rows (cell types) by 4 columns
    (GLM features: image, omission, behavior, task). Each column is one call to
    `plot_metric_across_conditions` (which handles the per-cell-type stats and
    saves its own stats CSVs per feature).

    Used to make the "population averages by area and depth" panels in figure 4
    supplemental, where the grid varies by:
        x_val='experience_level', hue='targeted_structure'  (area as hue)
        x_val='experience_level', hue='binned_depth'        (depth as hue)
        x_val='targeted_structure', hue='experience_level'  (area on x)
        x_val='binned_depth',      hue='experience_level'   (depth on x)

    Stats saving (delegated to `plot_metric_across_conditions`):
        One stats CSV per feature is written to <save_dir>/<folder>/, with
        `event_type='coding_score'` recorded in the table.

    The combined figure is saved at the wrapper level if `save_dir` is provided.
    Returns (fig, ax).
    """
    from visual_behavior.dimensionality_reduction.clustering import plotting

    feature_colors = plotting.get_feature_colors_and_labels()[0]
    feature_specs = [
        ('all-images', 'Image coding', feature_colors[0]),
        ('omissions', 'Omission coding', feature_colors[1]),
        ('behavioral', 'Behavior coding', feature_colors[2]),
        ('task', 'Task coding', feature_colors[3]),
    ]

    if figsize is None:
        # depth (4 bins) needs a wider figure than area (2 areas)
        figsize = (14, 8) if hue == 'binned_depth' or x_val == 'binned_depth' else (12, 8)

    fig, ax = plt.subplots(3, 4, figsize=figsize, sharex=True)

    for col, (metric, title, color) in enumerate(feature_specs):
        ax[:, col] = plot_metric_across_conditions(
            data, metric, title=title, x_color=color,
            xlabel=xlabel, x_val=x_val, hue=hue, plot_type=plot_type,
            event_type=event_type,
            save_dir=save_dir, folder=folder, ax=ax[:, col],
            use_mlm=use_mlm, group_column=group_column,
        )
        # Strip duplicated y-axis labels in non-leftmost columns.
        if col > 0:
            for row in range(3):
                ax[row, col].set_ylabel('')

    # When hue is something other than experience_level, the inner function
    # leaves a legend on each panel; drop everything except the one in the top
    # row to avoid clutter.
    if hue != 'experience_level':
        for col in range(4):
            for row in range(3):
                leg = ax[row, col].get_legend()
                if leg is not None and not (row == 0 and col == 0):
                    leg.remove()

    wspace = 0.3 if (hue == 'binned_depth' or x_val == 'binned_depth') else 0.4
    plt.subplots_adjust(wspace=wspace, hspace=0.3)

    if save_dir:
        if filename is None:
            filename = 'coding_scores_for_' + hue + '_by_' + x_val + '_' + plot_type
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename + suffix))

    return fig, ax


