"""
Created on Thursday September 23 2021

@author: marinag
"""
import warnings
# Harmless third-party import-time warnings from the allensdk -> xarray / requests
# dependency chain. Filtered here, before those heavy imports run below, so they are
# not emitted once per joblib worker process when stats are computed in parallel.
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
warnings.filterwarnings('ignore', message='Unable to find acceptable character detection dependency')
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


# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.top': False, 'axes.spines.right': False})  # ticks or white
sns.set_palette('deep')

plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

# Statistics configuration
# Single switch for the statistical test used by all add_stats_to_plot* functions
# (and the saved-stats CSV filename suffix). Set to True for hierarchical mixed
# linear models (with ANOVA/t-test fallback when data are too sparse), or False
# for the legacy ANOVA + Tukey HSD path. Changing it here applies everywhere.
USE_MLM = True


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


def _stats_suffix_for_table(stats_table):
    """
    Return the saved-stats CSV suffix reflecting the test that was ACTUALLY run,
    read from the per-row ``model_type`` column, rather than blindly from the
    ``USE_MLM`` flag.

    ``compute_stats(use_mlm=True)`` runs the hierarchical MLM but auto-falls back
    to ANOVA / Welch's t-test per panel when the data are too sparse, tagging
    those rows ``model_type='anova'`` instead of ``'mlm'``. A concatenated table
    can therefore be mixed. The naming rule (lean-mlm):

    - any genuine MLM row present -> ``_mlm.csv``
    - otherwise, if a fallback test ran            -> ``_anova.csv``
    - legacy ANOVA+Tukey path (``use_mlm=False``)  -> ``_tukey.csv``
      (that path emits no ``model_type`` column at all)

    This keeps the filename from ever claiming MLM when nothing in the table was
    fit with a mixed model.
    """
    if stats_table is None or 'model_type' not in getattr(stats_table, 'columns', ()):
        # legacy ANOVA + Tukey HSD path emits no model_type column
        return '_tukey.csv'
    kinds = set(stats_table['model_type'].dropna().unique())
    if 'mlm' in kinds:
        return '_mlm.csv'
    if 'anova' in kinds:
        return '_anova.csv'
    # model_type present but only 'none' (no testable groups) -> MLM was the
    # requested/attempted model; keep the mlm name.
    return '_mlm.csv'


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
                                           save_dir=None, folder='imaging_planes', ax=None):
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
                             save_dir=None, folder='dataset_stats', suffix='', ax=None):
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


def plot_n_cells_per_plane_by_depth(cells_table, suptitle=None, save_dir=None, folder='dataset_stats', ax=None):

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


def plot_n_planes_per_depth(experiments_table, suptitle=None, save_dir=None, folder='dataset_stats', ax=None):

    n_expts = experiments_table.groupby(['cell_type', 'binned_depth']).count().rename(columns={'ophys_session_id':'n_expts'}).reset_index()

    if ax is None:
        figsize = (12, 3)
        fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for i, cell_type in enumerate(utils.get_cell_types()):
        ax[i] = sns.barplot(data=n_expts[n_expts.cell_type==cell_type], y='binned_depth', x='n_expts',
                            orient='h', color='gray', width=0.5, ax=ax[i],
                            estimator="mean",
                            errorbar=("ci", 95),
                            n_boot=1000)
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
                                  color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i],
                                  estimator="mean",
                                  errorbar=("ci", 95),
                                  n_boot=1000)
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
        plt.setp(ax[i].lines, alpha=.3)
        ax[i] = sns.pointplot(data=data, x='experience_level', y='total_cells', hue='experience_level',
                              hue_order=experience_levels, palette=palette, dodge=0, linestyle='none', ax=ax[i],
                              estimator="mean",
                              errorbar=("ci", 95),
                              n_boot=1000)
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
                                            linewidth=1, save_dir=None, folder='population_activity', suffix='', ax=None):
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
                                            linewidth=1, save_dir=None, folder='population_activity', suffix='', ax=None):
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
                                                              save_dir=None, folder='population_activity', suffix=None, ax=None):
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
                                               save_dir=None, folder='population_activity', suffix=None, ax=None):
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
                                               save_dir=None, folder='population_activity', suffix=None, ax=None):
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
                                  order=experience_epoch, palette=palette, ax=ax[i], estimator=estimator,
                                  errorbar=("ci", 95),
                                  n_boot=1000)

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
                          linewidth=1.5, order=experience_epoch, palette=palette, ax=ax, estimator=estimator,
                          errorbar=("ci", 95),
                          n_boot=1000)

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
                                  order=experience_epoch, color=colors[c], ax=ax[i],
                                  estimator="mean",
                                  errorbar=("ci", 95),
                                  n_boot=1000)
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
                                   ylabel='Fraction responsive', save_dir=None, folder='response_metrics', suffix='', ax=None):
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
                                  color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i],
                                  estimator="mean",
                                  errorbar=("ci", 95),
                                  n_boot=1000)
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
        plt.setp(ax[i].lines, alpha=.3)
        ax[i] = sns.pointplot(data=data, x='experience_level', y='fraction_responsive', hue='experience_level',
                              hue_order=experience_levels, palette=palette, dodge=0, linestyle='none', ax=ax[i],
                              estimator="mean",
                              errorbar=("ci", 95),
                              n_boot=1000)
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
                                   ylabel='% responsive', save_dir=None, folder='response_metrics', suffix='', ax=None,
                                   group_column='mouse_id', event_type='Not specified'):
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
                                  color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i],
                                  estimator="mean",
                                  errorbar=("ci", 95),
                                  n_boot=1000)
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
        plt.setp(ax[i].lines, alpha=.3)
        ax[i] = sns.pointplot(data=data, x='experience_level', y=metric, hue='experience_level',
                              hue_order=experience_levels, palette=palette, dodge=0, markers='.',
                              err_kws={'linewidth': 2}, markersize=5, errorbar=('ci', 95), ax=ax[i],
                              estimator="mean",
                              n_boot=1000)

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
                                                     group_column=group_column,
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
        stats_suffix = _stats_suffix_for_table(combined_stats)
        combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(fig_title + stats_suffix)))
        # save descriptive stats
        cols_to_groupby = ['cell_type', 'experience_level']
        stats = get_descriptive_stats_for_metric(fraction_responsive, metric, cols_to_groupby)
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename(fig_title + '_values.csv')))
        # except BaseException:
        #     print('STATS DID NOT SAVE FOR', metric)
    return ax


def plot_average_metric_value_for_experience_levels_across_containers(df, metric, ylim=None, horizontal=True,
                                                                      save_dir=None, folder='response_metrics', suffix='', ax=None):
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
                                  color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i],
                                  estimator="mean",
                                  errorbar=("ci", 95),
                                  n_boot=1000)
        plt.setp(ax[i].collections, alpha=.3)  # for the markers
        plt.setp(ax[i].lines, alpha=.3)
        # plot the population average in color
        ax[i] = sns.pointplot(data=data, x='experience_level', y=metric, hue='experience_level',
                              hue_order=experience_levels, palette=palette, dodge=0, linestyle='none', ax=ax[i],
                              estimator="mean",
                              errorbar=("ci", 95),
                              n_boot=1000)
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


def _fit_stat_stars_within_axes(axes, margin_frac=0.03):
    """
    After significance bars/stars are drawn, grow an axis' upper y-limit (if needed) so
    the star *text* artists sit fully inside the axes. Stars are drawn with
    clip_on=False and their rendered glyph height is not accounted for by the data-based
    set_ylim, so without this they can poke above the top spine. Accepts a single Axes
    or an array of Axes. Safe no-op if no renderer / no text is available.
    """
    if hasattr(axes, 'get_ylim'):        # a single Axes
        axes = [axes]
    else:
        axes = [a for a in np.atleast_1d(axes).ravel() if a is not None]
    if not axes:
        return
    fig = axes[0].get_figure()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    for ax in axes:
        if not ax.texts:
            continue
        inv = ax.transData.inverted()
        lo, hi = ax.get_ylim()
        new_hi = hi
        for t in ax.texts:
            try:
                bb = t.get_window_extent(renderer=renderer)
                y_top = inv.transform((bb.x0, bb.y1))[1]
                if np.isfinite(y_top):
                    new_hi = max(new_hi, y_top)
            except Exception:
                pass
        if new_hi > hi:
            ax.set_ylim(top=new_hi + margin_frac * (hi - lo))


def add_stats_to_plot_for_hues(data, metric, ax, ymax=None, xorder=None, x='experience_level', hue='layer',
                               compact_bars=False,
                               group_column='mouse_id',
                               event_type='Not specified', cell_type='Not specified'):
    """
    add stars to axis indicating statistics across hue values
    x-axis of plots must be experience_levels
    xorder must be a list of values of x in the order that they appear on the plot

    data: metrics dataframe, each row is one cell_specimen_id in a given ophys_experiment
    metric: column in data representing metric values of interest
    column_to_compare: column in data to do stats over (after splitting by x values), such as 'layer' or 'targeted_structure'
    compact_bars: Bool. If False (default, legacy) the star sits at the current axis top
             and the axis is then expanded by 25%. If True, the star is placed a small
             fraction of the visible data RANGE above the data and the axis is expanded
             only slightly -- keeping the stars close to the data regardless of scale.
             Opt-in so existing callers are unchanged.
    MLM vs ANOVA/Tukey is set by the module-level ``USE_MLM`` constant: when True, use hierarchical mixed linear model with random intercept for group_column;
             falls back automatically to ANOVA/t-test when data is too sparse for MLM.
             If False, use the legacy ANOVA + Tukey HSD path.
    group_column: nesting variable for MLM (e.g., 'mouse_id'). Ignored when ``USE_MLM`` is False.
    event_type: optional label recorded in the saved stats table.
    """

    # formatting
    scale = 0.05
    fontsize = 15

    # honor an explicit ymax so that on a SHARED y-axis (one add_stats call per panel)
    # the star placement doesn't compound: every panel uses the same reference top
    ytop = ymax if ymax is not None else ax.get_ylim()[1]
    y = ytop - scale
    if compact_bars:
        lo = ax.get_ylim()[0]
        rng = ytop - min(lo, 0.0)
        if rng <= 0:
            rng = ytop if ytop > 0 else 1.0
        yh = ytop + 0.03 * rng        # star just above the data
        final_top = ytop + 0.10 * rng  # small headroom
    else:
        yh = ytop  # * (1 + scale)
        final_top = ytop * (1 + (scale * 5))

    # Build independent stats jobs (one per x_value with >=2 hues), then dispatch
    # through _run_panel_stats_jobs so MLM fits run in parallel across cores.
    # joblib preserves input order, so results match the serial path exactly.
    stat_jobs = []
    job_meta = []  # parallel list of (loc, x_value, n_hues) for star placement after fits
    for loc, x_value in enumerate(xorder):
        test_data = data[data[x] == x_value]
        hues = test_data[hue].unique()
        if len(hues) < 2:
            # only 1 hue value present at this x -- no comparison possible (t-test,
            # ANOVA, and MLM all require >=2 groups). Skip and record nothing.
            print(f"add_stats_to_plot_for_hues: skipping {x}={x_value!r} -- only "
                  f"{len(hues)} hue value(s) present, need >=2 to compare.")
            continue
        stat_jobs.append(dict(
            subset=test_data, metric=metric, column_to_compare=hue,
            use_mlm=USE_MLM, group_column=group_column,
            event_type=event_type, cell_type=cell_type,
            metadata=dict(data_subset=x_value),
        ))
        job_meta.append((loc, x_value, len(hues)))

    job_results = _run_panel_stats_jobs(stat_jobs)

    stats_table = pd.DataFrame()
    for (loc, x_value, n_hues), (omnibus_pvalue, panel_stats) in zip(job_meta, job_results):
        # gate star drawing on the omnibus, but keep the panel_stats either way
        # so the saved CSV records every comparison that was tested.
        if omnibus_pvalue < 0.05:
            for tindex, row in panel_stats.iterrows():
                if n_hues > 2:  # >2 values: use Holm-corrected pairwise reject
                    if row.reject:
                        ax.text(loc, yh, '*', fontsize=fontsize, horizontalalignment='center',
                                verticalalignment='bottom', color='k')
                elif n_hues == 2:  # 2 values: use the omnibus-aligned p
                    if row.one_way_anova_p_val < 0.05:
                        ax.text(loc, yh, '*', fontsize=fontsize, horizontalalignment='center',
                                verticalalignment='bottom', color='k')
        stats_table = pd.concat([stats_table, panel_stats])
    ax.set_ylim(ymax=final_top)
    _fit_stat_stars_within_axes(ax)

    return ax, stats_table


def add_stats_to_plot_for_hues_along_x(data, metric, ax, yorder=None, y='experience_level', hue='layer',
                                       group_column='mouse_id',
                                       event_type='Not specified', cell_type='Not specified'):
    """
    Add significance stars when metric is on the x-axis and categorical groups are on the y-axis.
    Tests are run across hue values within each y category.

    MLM vs ANOVA/Tukey is set by the module-level ``USE_MLM`` constant: when True, use hierarchical mixed linear model; falls back to ANOVA/t-test
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

    # Build independent stats jobs (one per y_value with >=2 hues) and dispatch
    # through _run_panel_stats_jobs for parallel MLM fits. joblib preserves input
    # order so star placement and the stats_table match the serial path exactly.
    stat_jobs = []
    job_meta = []  # parallel list of (loc, y_value, n_hues)
    for loc, y_value in enumerate(yorder):
        test_data = data[data[y] == y_value]
        hues = test_data[hue].unique()
        if len(hues) < 2:
            # only 1 hue value present at this y -- no comparison possible (t-test,
            # ANOVA, and MLM all require >=2 groups). Skip and record nothing.
            print(f"add_stats_to_plot_for_hues_along_x: skipping {y}={y_value!r} -- "
                  f"only {len(hues)} hue value(s) present, need >=2 to compare.")
            continue
        stat_jobs.append(dict(
            subset=test_data, metric=metric, column_to_compare=hue,
            use_mlm=USE_MLM, group_column=group_column,
            event_type=event_type, cell_type=cell_type,
            metadata={},
        ))
        job_meta.append((loc, y_value, len(hues)))

    job_results = _run_panel_stats_jobs(stat_jobs)

    stats_table = pd.DataFrame()
    for (loc, y_value, n_hues), (omnibus_pvalue, panel_stats) in zip(job_meta, job_results):
        has_sig = False
        if omnibus_pvalue < 0.05 and not panel_stats.empty:
            if n_hues > 2 and 'reject' in panel_stats.columns:
                has_sig = bool(panel_stats['reject'].any())
            elif n_hues == 2 and 'one_way_anova_p_val' in panel_stats.columns:
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
                      compact_bars=False,
                      group_column='mouse_id',
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
    compact_bars: Bool. If False (default, legacy behavior) significance bars are placed
                at MULTIPLES of the reference top ``ytop`` -- adjacent comparisons at
                1.05x/1.10x and "far" comparisons at 1.5x/1.55x -- which floats bars far
                above the data when ``ytop`` is large (e.g. percent-scale metrics) or
                when the data doesn't start near 0. If True, bars are placed at small
                ADDITIVE offsets expressed as a fraction of the visible data RANGE, with
                the star sitting right on its bar; this keeps bars just above the data and
                looks consistent across metrics of very different scales. Opt-in so the
                default placement of all existing callers is unchanged.
    MLM vs ANOVA/Tukey is set by the module-level ``USE_MLM`` constant: when True, use hierarchical mixed linear model with random intercept for group_column;
             falls back automatically to ANOVA/t-test when data is too sparse for MLM.
             If False, use the legacy ANOVA + Tukey HSD path.
    group_column: nesting variable for MLM (e.g., 'mouse_id').
    event_type: optional label recorded in the saved stats table.
    """
    # hierarchical stats (MLM by default) across experience levels or cell types
    stats_table = compute_stats(data, metric, column_to_compare,
                                use_mlm=USE_MLM, group_column=group_column,
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
        scale = 0.025
    fontsize = 15
    # compact mode centers the '*' on the bar (the glyph renders high, so va='bottom'
    # would float it well above the bar); legacy keeps the original bottom alignment
    star_va = 'center' if compact_bars else 'bottom'

    if ymax is None:
        ytop = ax.get_ylim()[1]
    else:
        ytop = ymax
    rng = None
    if compact_bars:
        # additive placement: bars sit a small fraction of the visible data RANGE above
        # the data top, with the star essentially on the bar. Scale-independent.
        lo = ax.get_ylim()[0]
        rng = ytop - min(lo, 0.0)
        if rng <= 0:
            rng = ytop if ytop > 0 else 1.0
        pad = 0.05 * rng        # gap above data to the first (adjacent) bar
        step = 0.13 * rng       # vertical separation between adjacent and far bars
        star_gap = 0.02 * rng   # small gap so the star sits just above (not on) the bar
        y1 = ytop + pad
        y1h = y1 + star_gap
        y2 = ytop + pad + step
        y2h = y2 + star_gap
    else:
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
                        verticalalignment=star_va)
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
                        verticalalignment=star_va)
                top.append(yh)
            else:
                if show_ns:
                    ax.plot([row.x1 + 0.1, row.x1 + 0.1, row.x2 - 0.1, row.x2 - 0.1], [y, y, y, y], linestyle='-',
                            color=color, alpha=alpha, clip_on=False)
                    # ax.plot([row.x1, row.x1, row.x2, row.x2], [y, yh, yh, y], linestyle='-', color=color, alpha=alpha)
                    ax.text(np.mean([row.x1, row.x2]), yh*(1+scale), 'ns', fontsize=fontsize-8, horizontalalignment='center',
                            verticalalignment='bottom')
                    top.append(yh)
    if compact_bars:
        # small fixed headroom above the tallest bar (range-relative, not multiplicative)
        ax.set_ylim(top=np.amax(top) + 0.05 * rng)
    else:
        # ax.set_ylim(ymax=ytop * (1 + (scale * 7))) # 3 works better for non-behavior plots
        ax.set_ylim(ymax=np.amax(top) * (1 + scale*scale_factor))  # scale factor determined by number of sig points # 3 works better for behavior plots, 2 for regular
    _fit_stat_stars_within_axes(ax)

    return ax, stats_table


def add_stats_to_plot_yaxis(data, metric, ax, ymax=None, column_to_compare='experience_level', hue_only=False,
                            group_column='mouse_id',
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
    MLM vs ANOVA/Tukey is set by the module-level ``USE_MLM`` constant: when True, use hierarchical mixed linear model with random intercept for group_column;
             falls back automatically to ANOVA/t-test when data is too sparse for MLM.
             If False, use the legacy ANOVA + Tukey HSD path.
    group_column: nesting variable for MLM (e.g., 'mouse_id').
    event_type: optional label recorded in the saved stats table.
    """
    # do anova across experience levels or cell types followed by post-hoc tukey
    stats_table = compute_stats(data, metric, column_to_compare,
                                use_mlm=USE_MLM, group_column=group_column,
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
                      group_column='mouse_id',
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
    MLM vs ANOVA/Tukey is set by the module-level ``USE_MLM`` constant: when True, use hierarchical mixed linear model with random intercept for group_column;
             falls back automatically to ANOVA/t-test when data is too sparse for MLM.
             If False, use the legacy ANOVA + Tukey HSD path.
    group_column: nesting variable for MLM (e.g., 'mouse_id').
    event_type: optional label recorded in the saved stats table.
    """
    stats_table = compute_stats(data, metric, column_to_compare,
                                use_mlm=USE_MLM, group_column=group_column,
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
                                                        title='', ylabel=None, ylims=None, save_dir=None, folder='response_metrics', ax=None, suffix='',
                                                        group_column='mouse_id'):
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
        save_fig = True
    else:
        save_fig = False

    # stats dataframe to save
    combined_stats = pd.DataFrame()
    if hue:
        if pointplot:
            ax = sns.pointplot(data=data, y=y, x=x, order=order, dodge=0.3, linestyle='none',
                               markers='.', markersize=5, err_kws={'linewidth': 2}, hue=hue, hue_order=hue_order, palette='gray', ax=ax,
                               estimator="mean",
                               errorbar=("ci", 95),
                               n_boot=1000)

        else:
            ax = sns.boxplot(data=data, y=y, x=x, order=order, cut=0, notch=True,
                             width=0.4, hue=hue, hue_order=hue_order, palette='gray', ax=ax,
                             whis=1.5)
        ax.legend(fontsize='xx-small', title='')  # , loc=loc)  # bbox_to_anchor=(1,1))
            # TBD add area or depth comparison stats / stats across hue variable
    else:
        hue = 'experience_level'
        if show_containers:
            print('table includes', len(data.ophys_container_id.unique()), 'containers')
            for ophys_container_id in data.ophys_container_id.unique():
                ax = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x=x, y=y,
                                   color='gray', linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax,
                                   estimator="mean",
                                   errorbar=("ci", 95),
                                   n_boot=1000)
        if show_mice:
            print('table includes', len(data.mouse_id.unique()), 'mice')
            for mouse_id in data.mouse_id.unique():
                ax = sns.pointplot(data=data[data.mouse_id == mouse_id], x=x, y=y, order=order,
                                   color='gray', linewidth=0.5, markers='.', markersize=1, err_kws={'linewidth': 0.5}, ax=ax,
                                   estimator="mean",
                                   errorbar=("ci", 95),
                                   n_boot=1000)
            plt.setp(ax.collections, alpha=.7)  # for the markers
            plt.setp(ax.lines, alpha=.7)

        if pointplot:
            # ax = sns.pointplot(data=data, x='experience_level', y=metric,
            #                    palette=colors, ax=ax)
            ax = sns.pointplot(data=data, x=x, y=y, hue=hue, order=order,
                                  hue_order=order, palette=palette, dodge=0, linestyle='None',
                                  markers='.', markersize=8, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax,
                                  estimator="mean",
                                  n_boot=1000)

        else:
            ax = sns.boxplot(data=data, x=x, y=y, width=0.4, order=order, notch=True,
                             palette=colors, ax=ax,
                             whis=1.5)
        if stripplot:
            # add strip plot
            ax = sns.stripplot(data=data, size=3, alpha=0.5, jitter=0.2, order=order,
                               x=x, y=y, color='gray', ax=ax)
        if boxplot:
            ax = sns.boxplot(data=data, x=x, y=y, width=0.4, order=order, notch=True,
                             palette='dark:white', ax=ax,
                             whis=1.5)
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
                                                      group_column=group_column,
                                                      event_type=event_type)
        else:
            ax.set_ylim(ymin=ymin)
            ax.set_xlim(-0.5, len(order) - 0.5)

            # add stats to plot if only looking at experience levels
            ax, panel_stats = add_stats_to_plot(data, metric, ax, ymax=ymax, show_ns=show_ns,
                                                group_column=group_column,
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
        filename = _clean_filename(event_type + '_' + data_type + '_' + metric + '_distribution' + suffix)
        stats_filename = _clean_filename(event_type + '_' + data_type + '_' + metric + suffix + '_no_cell_type')
        if save_fig:
            utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        print('saving_stats')
        stats_suffix = _stats_suffix_for_table(combined_stats)
        combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
        # save descriptive stats
        cols_to_groupby = ['experience_level']
        stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
    return ax


def plot_metric_distribution_by_experience(metrics_table, metric, event_type, data_type, hue=None,
                                               plot_type='pointplot', legend=True, show_containers=False, estimator=np.mean,
                                               add_zero_line=False, show_ns=False, ylabel=None, ylims=None, horiz=True,
                                               abbreviate_exp=True, suptitle=None, save_dir=None, folder='response_metrics', ax=None, suffix='',
                                               group_column='mouse_id'):
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
                                      linewidth=0.5, markers='.', markersize=0.25, err_kws={'linewidth': 0.5}, ax=ax[i],
                                      errorbar=("ci", 95),
                                      n_boot=1000)

        if hue:
            if plot_type == 'pointplot':
                dodge = 0.1 * len(ct_data[hue].unique())
                ax[i] = sns.pointplot(data=ct_data, y=metric, x='experience_level', order=order, dodge=dodge, linestyle='none',
                                      markers='.', markersize=5, err_kws={'linewidth': 2}, hue=hue, hue_order=hue_order, 
                                      estimator=estimator, palette=hue_colors, ax=ax[i],
                                      errorbar=("ci", 95),
                                      n_boot=1000)
            elif plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, y=metric, x='experience_level', order=order, fliersize=0, notch=True,
                                    width=0.4, hue=hue, hue_order=hue_order, palette=hue_colors, ax=ax[i],
                                    whis=1.5)
                for box in ax[i].collections:
                    box.set_alpha(0.75)
            elif plot_type == 'violinplot':
                if len(ct_data[hue].unique())==2:
                    split = True
                else:
                    split = False
                ax[i] = sns.violinplot(data=ct_data, y=metric, x='experience_level', order=order,
                                       hue=hue, hue_order=hue_order, palette=hue_colors, cut=0, inner=None,
                                       split=split, fill=False, ax=ax[i],
                                       density_norm="area",
                                       bw_method="scott")
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
                                                   group_column=group_column,
                                                   event_type=event_type, cell_type=cell_type)
            panel_stats = insert_stats_metadata(panel_stats, condition='experience_level')
            combined_stats = pd.concat([combined_stats, panel_stats])
        else:
            if plot_type == 'pointplot':
                ax[i] = sns.pointplot(data=ct_data, x='experience_level', y=metric, palette=colors, hue='experience_level',
                                      estimator=estimator, markers='.', markersize=5, err_kws={'linewidth': 2}, ax=ax[i],
                                      errorbar=("ci", 95),
                                      n_boot=1000)
            elif plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, x='experience_level', y=metric, width=0.4, hue='experience_level',
                                     notch=True, palette=colors, fliersize=0, ax=ax[i],
                                     whis=1.5)
                for box in ax[i].collections:
                    box.set_alpha(0.75)
            elif plot_type == 'barplot':
                ax[i] = sns.barplot(data=ct_data, x='experience_level', y=metric, width=0.7, hue='experience_level',
                                     palette=colors, ax=ax[i],
                                     estimator="mean",
                                     errorbar=("ci", 95),
                                     n_boot=1000)
                for bar in ax[i].patches:
                    bar.set_alpha(0.75)
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, y=metric, x='experience_level', order=order, hue='experience_level',
                                       palette=colors,  cut=0, ax=ax[i],
                                       inner="box",
                                       density_norm="area",
                                       bw_method="scott")
                for violin in ax[i].collections:
                    violin.set_alpha(0.75)

            elif plot_type == 'stripplot':
                ax[i] = sns.boxplot(data=ct_data, x='experience_level', y=metric, width=0.4,
                                    palette='dark:white', ax=ax[i],
                                    whis=1.5)
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
                                                   group_column=group_column,
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
            ax[i].set_title(cell_type+'\n(n = '+str(len(ct_data.cell_specimen_id.unique()))+' cells)', fontsize=16)
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
        filename = _clean_filename(event_type + '_' + data_type + '_' + metric + '_distribution' + suffix)
        stats_filename = _clean_filename(event_type + '_' + data_type + '_' + metric + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            # save stats: '_mlm.csv' when MLM was used, '_tukey.csv' for the legacy path
            stats_suffix = _stats_suffix_for_table(combined_stats)
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


def plot_metric_over_repeats(df, metric, x, title='', xlabel=None, ylabel=None, save_dir=None, folder='response_metrics', ax=None):
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
                        palette=experience_level_colors, hue_order=experience_levels, ax=ax,
                       errorbar=("ci", 95),
                       n_boot=1000)
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


def plot_rolling_metric_over_time_in_session(rolling_df, metric='rolling_dprime', bin_size_seconds=120,
                                             max_minutes=60, label_every=5, linewidth=1.5, ylabel=None, title='',
                                             save_dir=None, folder='within_session_behavior', suffix='', ax=None):
    '''
    Plot a rolling behavioral performance metric averaged across sessions in equal-width time
    bins over the course of the session, split by experience level.

    Time in session is binned into `bin_size_seconds` bins and plotted in minutes on the x-axis.
    x tick labels are shown as integers (minutes) and only every `label_every`-th bin is labeled,
    so with the defaults (120 s bins, label_every=5) labels appear in 10 minute increments.

    Parameters
    ----------
    rolling_df : pd.DataFrame
        Rolling performance data with columns 'time_in_session' (seconds), `metric`, and
        'experience_level'. E.g. the output of
        utilities.get_stimulus_based_rolling_performance_df_for_dataset joined with experience_level.
    metric : str
        Column to plot on the y-axis (e.g. 'rolling_dprime', 'hit_rate', 'false_alarm_rate',
        'reward_rate'). Default 'rolling_dprime'.
    bin_size_seconds : float
        Width of the time-in-session bins, in seconds. Default 120 (2 min).
    max_minutes : float
        Only include bins below this many minutes into the session. Default 60.
    label_every : int
        Label every Nth x tick; the rest are hidden. Default 5 (=> every 10 min for 120 s bins).
    linewidth : float
        Width of the lines connecting the point estimates. Default 1.5.
    ylabel : str or None
        y-axis label. If None, derived from `metric`.
    save_dir : str or None
        If provided, the figure is saved under save_dir/folder via utils.save_figure.
    folder : str or None
        Sub-folder within save_dir to save into.
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    matplotlib.axes.Axes
    '''
    if ax is None:
        figsize = (5, 3)
        fig, ax = plt.subplots(figsize=figsize)

    data = rolling_df.copy().reset_index(drop=True)
    # bin time in session and express the bin label in minutes
    data['time_bin'] = ((data['time_in_session'] // bin_size_seconds) * bin_size_seconds) / 60.
    data = data[data['time_bin'] < max_minutes]

    # error bars ACROSS MICE (not across flashes): collapse to one value per
    # mouse / experience level / time bin before the pointplot, so the CI reflects
    # between-mouse variability. Requires a 'mouse_id' column; falls back to the
    # per-flash behavior if it is absent.
    if 'mouse_id' in data.columns:
        data = (data.groupby(['mouse_id', 'experience_level', 'time_bin'])[metric]
                .mean().reset_index())

    experience_level_colors = utils.get_experience_level_colors()
    experience_levels = utils.get_new_experience_levels()

    # fix the category order so x positions line up with bin_centers below
    bin_centers = np.sort(data['time_bin'].unique())
    ax = sns.pointplot(data=data, x='time_bin', y=metric, hue='experience_level', order=bin_centers,
                       linewidth=linewidth, markers='.', markersize=5, err_kws={'linewidth': linewidth}, estimator=np.mean,
                       palette=experience_level_colors, hue_order=experience_levels, ax=ax,
                       errorbar=("ci", 95),
                       n_boot=1000)
    ax.legend(fontsize='xx-small', title_fontsize='xx-small', title='')
    ax.set_title(title)
    # integer (minute) x tick labels, labeling only every `label_every`-th bin
    ax.set_xticks(range(len(bin_centers)))
    ax.set_xticklabels([str(int(round(b))) if (j % label_every == 0) else ''
                        for j, b in enumerate(bin_centers)])
    ax.set_xlabel('Time in session (min)')
    if ylabel is None:
        ax.set_ylabel(metric.replace('_', ' ').capitalize())
    else:
        ax.set_ylabel(ylabel)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metric + '_over_time_in_session'+suffix))

    return ax


def plot_metric_in_time_bins_by_experience(rolling_df, metric='rolling_dprime',
                                           bin_size_minutes=15, max_minutes=60,
                                           plot_type='boxplot', ylabel='D-prime', title='',
                                           save_dir=None, folder='within_session_behavior',
                                           suffix='', ax=None):
    '''
    Plot a rolling behavioral metric in equal-width time bins over the course of the session,
    split by experience level and aggregated to one value per mouse per bin (so the spread
    reflects variability ACROSS MICE, not across flashes or sessions).

    Time in session is binned into `bin_size_minutes` bins. For each
    (mouse_id, experience_level, time_bin) the metric is averaged, then plotted one of two ways:

    - plot_type='boxplot' (default): a box per experience level within each time bin showing the
      distribution across mice, with thin semi-transparent boxes and gray lines connecting each
      mouse's values across experience levels within a bin.
    - plot_type='pointplot': mean +/- 95% CI across mice as points connected by lines across
      time bins, one line per experience level (like plot_rolling_metric_over_time_in_session,
      but on the per-mouse, coarsely-binned data).

    Parameters
    ----------
    rolling_df : pd.DataFrame
        Rolling performance data with columns 'time_in_session' (seconds), `metric`,
        'experience_level', and 'mouse_id'. E.g. the output of
        utilities.get_stimulus_based_rolling_performance_df_for_dataset joined with
        'experience_level' and 'mouse_id' from the experiments metadata table.
    metric : str
        Column to plot on the y-axis (e.g. 'rolling_dprime', 'hit_rate', 'false_alarm_rate').
        Default 'rolling_dprime'.
    bin_size_minutes : float
        Width of the time-in-session bins, in minutes. Default 15.
    max_minutes : float
        Only include bins below this many minutes into the session. Default 60.
    plot_type : str
        'boxplot' (boxes across mice + gray per-mouse lines) or 'pointplot' (mean +/- CI across
        mice, connected across time bins). Default 'boxplot'.
    ylabel : str or None
        y-axis label. If None, derived from `metric`.
    title : str
        Axis title. Default ''.
    save_dir : str or None
        If provided, the figure is saved under save_dir/folder via utils.save_figure.
    folder : str or None
        Sub-folder within save_dir to save into.
    suffix : str
        Appended to the saved filename.
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    matplotlib.axes.Axes
    '''
    assert 'mouse_id' in rolling_df.columns, \
        "rolling_df must contain a 'mouse_id' column to aggregate the metric across mice"

    if ax is None:
        figsize = (4, 3)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        figsize = tuple(fig.get_size_inches())

    data = rolling_df.copy()
    # time bin labeled by its left edge in minutes
    data['time_bin_min'] = (data['time_in_session'] // (bin_size_minutes * 60)) * bin_size_minutes
    data = data[data['time_bin_min'] < max_minutes]

    # collapse to one value per mouse per experience level per time bin
    per_mouse = (data.groupby(['mouse_id', 'experience_level', 'time_bin_min'])[metric]
                 .mean().reset_index())

    experience_level_colors = utils.get_experience_level_colors()
    experience_levels = utils.get_new_experience_levels()
    bin_order = sorted(per_mouse['time_bin_min'].unique())
    n_hues = len(experience_levels)

    if plot_type == 'pointplot':
        # mean +/- CI across mice, lines connect across time bins within each experience level
        ax = sns.pointplot(data=per_mouse, x='time_bin_min', y=metric, hue='experience_level',
                      order=bin_order, hue_order=experience_levels, linewidth=1.5, err_kws={'linewidth': 1.5},
                      palette=experience_level_colors, markers='.', markersize=5,
                      estimator=np.mean, ax=ax,
                      errorbar=("ci", 95),
                      n_boot=1000)
        ax.set_ylim(bottom=0)  # performance metrics are typically bounded at 0, so start y-axis there
    elif plot_type == 'boxplot':
        box_width = 0.5  # thinner boxes
        sns.boxplot(data=per_mouse, x='time_bin_min', y=metric, hue='experience_level',
                    order=bin_order, hue_order=experience_levels,
                    palette=experience_level_colors, width=box_width, fliersize=0, ax=ax,
                    whis=1.5)
        # semi-transparent box faces
        for patch in ax.patches:
            r, g, b = patch.get_facecolor()[:3]
            patch.set_facecolor((r, g, b, 0.7))

        # gray lines connecting each mouse across experience levels within each time bin;
        # reproduce seaborn's dodge geometry to place the points at each box center
        def _xpos(i, j):
            return i - box_width / 2 + box_width / (2 * n_hues) + j * box_width / n_hues

        for i, b in enumerate(bin_order):
            bin_df = per_mouse[per_mouse['time_bin_min'] == b].set_index('mouse_id')
            for mouse_id in bin_df.index.unique():
                mvals = (bin_df.loc[[mouse_id]].set_index('experience_level')[metric]
                         .reindex(experience_levels))
                xs = [_xpos(i, j) for j, lvl in enumerate(experience_levels)
                      if not np.isnan(mvals[lvl])]
                ys = [mvals[lvl] for lvl in experience_levels if not np.isnan(mvals[lvl])]
                if len(xs) >= 1:
                    ax.plot(xs, ys, '-', color='gray', lw=0.5, alpha=0.5, marker='o',
                            markersize=2, markerfacecolor='gray', markeredgecolor='none', zorder=2)
    else:
        raise ValueError("plot_type must be 'boxplot' or 'pointplot', got %r" % plot_type)

    # legend outside the axes, upper right, no title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:n_hues], labels[:n_hues], fontsize='x-small', frameon=False,
              bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xticklabels(['%d-%d' % (int(b), int(b) + bin_size_minutes) for b in bin_order])
    ax.set_xlabel('Time in session (min)')
    ax.set_ylabel(metric.replace('_', ' ').capitalize() if ylabel is None else ylabel)
    ax.set_title(title)
    sns.despine()

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder,
                          _clean_filename('%s_%dmin_bins_by_experience_%s%s'
                                          % (metric, bin_size_minutes, plot_type, suffix)))
    return ax


def plot_metric_over_repeats_for_cell_types(df, metric, x, xlabel=None, ylabel=None, save_dir=None, folder='response_metrics'):
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
                                                   ax=None, save_dir=None, folder='response_metrics'):
    
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
        figsize = (6, 3)
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
                                       fill=True, save_dir=None, folder='response_metrics', suffix='', ax=None,
                                       group_column='mouse_id', event_type='Not specified'):
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
                                    order=order, palette=colors, ax=ax[i], width=0.6, fliersize=0, notch=True,
                                    whis=1.5)
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, x=x, y=y, orient=orient,
                                hue='experience_level', hue_order=experience_levels,
                                order=order, palette=colors, ax=ax[i], alpha=0.5, fill=fill, linewidth=1, gap=0.1, cut=0,
                                inner='box', inner_kws=dict(box_width=2, whis_width=1, color="k", alpha=0.75),
                                density_norm="area",
                                bw_method="scott")
                ax[i] = sns.pointplot(data=ct_data, x=x, y=y, orient=orient,
                                    hue='experience_level', hue_order=experience_levels, linestyle='none', dodge=0.55,
                                    order=order, color='k', ax=ax[i], zorder=10000,
                                    markers='_', markersize=10, err_kws={'linewidth': 2},
                                    estimator="mean",
                                    errorbar=("ci", 95),
                                    n_boot=1000)
            
            _legend = ax[i].get_legend()
            if _legend: _legend.remove()
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')

            if not metric_on_y:
                ax[i].set_xlim(lims)
                ax[i], panel_stats = add_stats_to_plot_for_hues_along_x(ct_data, metric, ax[i],
                                                            yorder=order, y=x_axis_col, hue='experience_level',
                                                            group_column=group_column,
                                                            event_type=event_type, cell_type=cell_type)
            else:
                ax[i].set_ylim(lims)
                ax[i], panel_stats = add_stats_to_plot_for_hues(ct_data, metric, ax[i],
                                                            xorder=order, x=x_axis_col, hue='experience_level',
                                                            group_column=group_column,
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
                                    palette=colors, ax=ax[i], width=0.6, fliersize=0, notch=True,
                                    whis=1.5)
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, y=y, x=x, order=experience_levels,
                                        hue='experience_level', hue_order=experience_levels, legend=False,
                                        palette=colors, ax=ax[i], alpha=0.75, fill=fill, linewidth=1, gap=0.1, cut=0,
                                        inner='box', inner_kws=dict(box_width=2, whis_width=1, color="k", alpha=1),
                                        density_norm="area",
                                        bw_method="scott")
                ax[i] = sns.pointplot(data=ct_data, y=y, x=x, order=experience_levels,
                                        hue='experience_level', hue_order=experience_levels, legend=False,
                                        color='k', ax=ax[i], zorder=10000, linestyle='none',
                                        markers='_', markersize=10, err_kws={'linewidth': 2},
                                        estimator="mean",
                                        errorbar=("ci", 95),
                                        n_boot=1000)            
            if metric_on_y:
                if abbreviate_exp:
                    ax[i].set_xticks(np.arange(0, len(experience_levels)))
                    ax[i].set_xticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
                    utils.color_xaxis_labels_by_experience(ax[i])
                ax[i].set_ylim(lims)
                ax[i], panel_stats = add_stats_to_plot_yaxis(ct_data, metric, ax[i], ymax=lims[1], column_to_compare='experience_level',
                                                             group_column=group_column,
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
                                                             group_column=group_column,
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
        filename = _clean_filename(metric + '_distribution' + suffix)
        stats_filename = _clean_filename(metric + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = _stats_suffix_for_table(combined_stats)
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
            cols_to_groupby = ['cell_type', 'experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric)
    return ax


def plot_metric_across_cohorts(metrics_table, metric,  ylabel, x_val='binned_depth', plot_type='barplot',
                               event_type='Not specified', save_dir=None, folder='response_metrics', ax=None,
                               group_column='mouse_id'):
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
                                            markers='.', markersize=8, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i],
                                            estimator="mean",
                                            n_boot=1000)
            elif plot_type == 'barplot': 
                ax[i] = sns.barplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals, width=0.5, alpha=0.75, 
                                                hue_order=experience_levels, palette=palette, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i],
                                                estimator="mean",
                                                n_boot=1000)
            elif plot_type == 'boxplot': 
                ax[i] = sns.boxplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals, boxprops=dict(alpha=0.75),
                                                hue_order=experience_levels, palette=palette, notch=True,
                                                width=0.5, fliersize=0, ax=ax[i],
                                                whis=1.5)
                plt.setp(ax[i].collections, alpha=0.75)
            elif plot_type == 'violinplot': 
                ax[i] = sns.violinplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals, 
                                                hue_order=experience_levels, palette=palette, 
                                                width=0.5, fliersize=0, ax=ax[i],
                                                inner="box",
                                                density_norm="area",
                                                bw_method="scott",
                                                cut=2)
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
                                                            group_column=group_column,
                                                            cell_type=cell_type)
            # cohort first so it lands before condition; that keeps data_subset
            # (added inside add_stats_to_plot_for_hues) directly after condition.
            panel_stats = insert_stats_metadata(panel_stats, cohort=project_code, condition=x_val)
            combined_stats = pd.concat([combined_stats, panel_stats])
            i+=1
    plt.subplots_adjust(wspace=0.5, hspace=0.35)

    if save_dir:
        filename = _clean_filename(metric+'_by_cohort_x_'+x_val)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = _stats_suffix_for_table(combined_stats)
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)))
            cols_to_groupby = ['cell_type', 'experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric)
    return ax


def plot_metric_across_cohorts_area_depth(metrics_table, metric,  ylabel, plot_type='barplot',
                               event_type='Not specified', save_dir=None, folder='response_metrics', ax=None,
                               group_column='mouse_id'):
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
                                                markers='.', markersize=8, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i],
                                                estimator="mean",
                                                n_boot=1000)
                elif plot_type == 'barplot':
                    ax[i] = sns.barplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals, width=0.5, alpha=0.75, 
                                                    hue_order=experience_levels, palette=palette, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i],
                                                    estimator="mean",
                                                    n_boot=1000)
                elif plot_type == 'boxplot':
                    ax[i] = sns.boxplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals,
                                                    hue_order=experience_levels, palette=palette,
                                                    width=0.5, fliersize=0, ax=ax[i],
                                                    whis=1.5)
                    plt.setp(ax[i].collections, alpha=0.75)
                elif plot_type == 'violinplot':
                    ax[i] = sns.violinplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals,
                                        hue_order=experience_levels, palette=palette, alpha=0.75, cut=0, width=0.75,
                                        fill=False, linewidth=1.5, gap=0.1, inner='box',
                                        inner_kws=dict(box_width=2, whis_width=1, color="gray", alpha=0.75), ax=ax[i],
                                        density_norm="area",
                                        bw_method="scott")
                    ax[i] = sns.pointplot(data=data, x=x_val, y=metric, hue='experience_level', order=x_vals,
                                          hue_order=experience_levels, palette=palette, dodge=0.5, linestyle='none',
                                          markers='.', markersize=5, err_kws={'linewidth': 2}, errorbar=('ci', 95),
                                         zorder=10000, ax=ax[i],
                                          estimator="mean",
                                          n_boot=1000)

                ax[i].set_ylabel('')
                ax[i].set_xlabel('')
                _legend = ax[i].get_legend()

                if _legend: _legend.remove()
                ax[i].set_title('Cohort '+str(p+1))
                # ax[i], combined_stats = ppf.add_stats_to_plot(data, metric, ax[i])
                ax[i], panel_stats = add_stats_to_plot_for_hues(data, metric, ax[i], event_type=event_type,
                                                                xorder=x_vals, x=x_val, hue='experience_level',
                                                                group_column=group_column,
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
        filename = _clean_filename(metric+'_by_cohort_depth_area')
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = _stats_suffix_for_table(combined_stats)
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
                               event_type='Not specified', compact_bars=False,
                               save_dir=None, folder='response_metrics', ax=None,
                               group_column='mouse_id'):
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
                                        markers='.', markersize=8, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i],
                                        estimator="mean",
                                        n_boot=1000)
        elif plot_type == 'barplot': 
            ax[i] = sns.barplot(data=data, x=x_val, y=metric, hue=hue, order=x_vals, width=0.6, alpha=0.75, 
                                            hue_order=hue_order, palette=palette, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i],
                                            estimator="mean",
                                            n_boot=1000)
        elif plot_type == 'boxplot': 
            ax[i] = sns.boxplot(data=data, x=x_val, y=metric, hue=hue, order=x_vals,
                                            hue_order=hue_order, palette=palette, notch=True,
                                            width=0.5, showfliers=False, ax=ax[i],
                                            whis=1.5)
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
            ax[i].set_xticklabels(utils.get_abbreviated_experience_levels())  # F, N, N+
            for xtick, color in zip(ax[i].get_xticklabels(), experience_level_colors):
                xtick.set_color(color)
        # ax[i].tick_params(axis='x', which='major', labelsize=12)
        ax[i].tick_params(axis='y', which='major', labelsize=12)
        if c == 0: 
            ax[i].set_title(title, color=x_color, fontsize=12)

        if c == 1: 
            ax[i].set_ylabel(cell_type+'\nCoding score')
        else: 
            ax[i].set_ylabel(cell_type+'\n')

        ax[i], panel_stats = add_stats_to_plot_for_hues(data, metric, ax[i], event_type=event_type,
                                                        xorder=x_vals, x=x_val, hue=hue,
                                                        compact_bars=compact_bars,
                                                        group_column=group_column,
                                                        cell_type=cell_type)
        panel_stats = insert_stats_metadata(panel_stats, condition=x_val)
        combined_stats = pd.concat([combined_stats, panel_stats])
        # , ymax=None, show_ns=False)
        i+=1
    plt.subplots_adjust(wspace=0.5, hspace=0.3)

    # save stats
    filename = _clean_filename(metric+'_across_'+hue+'_for_'+x_val+'_'+plot_type)
    if save_dir:
        stats_suffix = _stats_suffix_for_table(combined_stats)
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
                                     folder='response_metrics', group_column='mouse_id'):
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
                                      x=x, y=metric, palette='gray', ax=ax[i],
                                      estimator="mean",
                                      errorbar=("ci", 95),
                                      n_boot=1000)
            elif plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, order=xorder, hue=hue, hue_order=hue_order, width=0.5, boxprops=dict(alpha=0.8),
                                    x=x, y=metric, palette='gray', ax=ax[i],
                                    whis=1.5)
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, order=xorder, cut=0, hue=hue, hue_order=hue_order, inner=None,
                                       x=x, y=metric, palette='gray', split=True, fill=False, ax=ax[i],
                                       density_norm="area",
                                       bw_method="scott")
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
                                                            group_column=group_column,
                                                            event_type=event_type, cell_type=cell_type)
            # ax[i], panel_stats = add_stats_to_plot(ct_data, metric, ax[i], ymax=ymax)
            combined_stats = pd.concat([combined_stats, panel_stats])
        else:
            if plot_type == 'pointplot':
                ax[i] = sns.pointplot(data=ct_data, order=xorder, linestyle='none',
                                  x=x, y=metric, color='gray', ax=ax[i],
                                  estimator="mean",
                                  errorbar=("ci", 95),
                                  n_boot=1000)
            elif plot_type == 'boxplot':
                ax[i] = sns.boxplot(data=ct_data, order=xorder, width=0.5,  boxprops=dict(alpha=0.8),
                                      x=x, y=metric, color='gray', ax=ax[i],
                                      whis=1.5)
            elif plot_type == 'violinplot':
                ax[i] = sns.violinplot(data=ct_data, order=xorder, cut=0,
                                      x=x, y=metric, color='gray', ax=ax[i],
                                      inner="box",
                                      density_norm="area",
                                      bw_method="scott")
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
        filename = _clean_filename('experience_modulation_' + event_type + '_' + plot_type + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = _stats_suffix_for_table(combined_stats)
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)))
            # save descriptive stats
            cols_to_groupby = ['cell_type', 'experience_level']
            stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR', metric, hue)


def plot_experience_modulation_index_annotated(metrics_table, event_type, metric, cells_table,
                                               horiz=False, xlims=(-1.1, 1.1), xlabel='Experience modulation',
                                               suptitle=None, suffix='', save_dir=None, folder='response_metrics', ax=None,
                                               group_column='mouse_id'):
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
                                inner_kws=dict(box_width=2, whis_width=1, color="k", alpha=0.75),
                               density_norm="area",
                               bw_method="scott")

        ax[i] = sns.pointplot(data=data[data.comparison == comparison], x=metric, y='cell_type', order=cell_types,
                                    color='k', ax=ax[i], zorder=10000, linestyle='none',
                                    markers='|', markersize=15, err_kws={'linewidth': 2},
                                    estimator="mean",
                                    errorbar=("ci", 95),
                                    n_boot=1000)
  
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
                               group_column=group_column,
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
        filename = _clean_filename('experience_modulation_annot_' + event_type + '_' + suffix)
        stats_filename = _clean_filename('experience_modulation_' + event_type + '_' + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = _stats_suffix_for_table(combined_stats)
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
                                                            suptitle=None, suffix='', save_dir=None, folder='response_metrics', ax=None,
                                                            group_column='mouse_id'):
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
                               inner_kws=dict(box_width=2, whis_width=1, color="k", alpha=0.75),
                               density_norm="area",
                               bw_method="scott")
        ax[i] = sns.pointplot(data=data[data.cell_type == cell_type],
                               x=metric, y='comparison', order=value_vars,
                                color='k', ax=ax[i], zorder=100000, linestyle='none',
                                markers='|', markersize=15, err_kws={'linewidth': 2},
                               estimator="mean",
                               errorbar=("ci", 95),
                               n_boot=1000)
        
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
                                                         group_column=group_column,
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
        filename = _clean_filename('experience_modulation_annot_by_cell_type_' + event_type + '_' + suffix)
        stats_filename = _clean_filename('experience_modulation_by_cell_type_' + event_type + '_' + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        try:
            print('saving_stats')
            stats_suffix = _stats_suffix_for_table(combined_stats)
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
                                                                suptitle=None, suffix='', save_dir=None, folder='response_metrics'):
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
                                                                 suptitle=None, suffix='', save_dir=None, folder='response_metrics'):
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
        suptitle=None, suffix='', save_dir=None, folder='response_metrics',
        group_column='mouse_id', event_type='Not specified',
        fig=None, bbox=None, cell_type_hspace_scale=1.4):
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
                                            use_mlm=USE_MLM, group_column=group_column,
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
                                            use_mlm=USE_MLM, group_column=group_column,
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
    # cell_type_hspace_scale lowers/raises the constant to tighten/loosen the gap between
    # cell-type rows within a panel (e.g. when embedding multiple panels into one composite).
    cell_type_hspace = cell_type_hspace_scale / max(nrows, 1)

    # Embedding support: when a fig (and optional bbox sub-rectangle, figure fraction, y from the
    # top) are provided, draw into that figure/region instead of making a new figure, and confine
    # the outer GridSpec to the bbox. Defaults reproduce standalone behavior.
    created_fig = fig is None
    gs_kw = {}
    if bbox is not None:
        gs_kw = dict(left=bbox[0], right=bbox[2], top=1 - bbox[1], bottom=1 - bbox[3])

    if horiz:
        # rows = metrics, cols = cell types. Per-metric-row cbars on the right.
        fig_w = 2.2 * n_cell_types + 2.1
        # Per-row term (0.22 * nrows * n_metrics) scales with the number of y-axis values; the
        # second term is a fixed base for titles/labels. The base is smaller for the low-row
        # case (visual area) so it doesn't look proportionally too tall, while leaving the depth
        # version unchanged.
        fig_h = 0.22 * nrows * n_metrics + (0.55 if nrows <= 2 else 1.0)
        figsize = (fig_w, fig_h)
        if created_fig:
            fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(1, 2, width_ratios=[1.6, 0.5], wspace=0.4, **gs_kw)
        main_gs = gridspec.GridSpecFromSubplotSpec(n_metrics, n_cell_types, subplot_spec=outer[0],
                                                    hspace=cell_type_hspace, wspace=0.3)
    elif n_metrics == 1:
        # vert single-metric: one set of right-side cbars centered on the middle cell-type row
        fig_w = 3.6
        fig_h = 0.45 * nrows * n_cell_types + 1.6
        figsize = (fig_w, fig_h)
        if created_fig:
            fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1.0], wspace=0.45, **gs_kw)
        main_gs = gridspec.GridSpecFromSubplotSpec(n_cell_types, n_metrics, subplot_spec=outer[0],
                                                    hspace=cell_type_hspace, wspace=0.0)
    else:
        # vert multi-metric: heatmap grid on the left + single right-side cbar set
        fig_w = 1.8 * n_metrics + 2.3
        fig_h = 0.45 * nrows * n_cell_types + 1.4
        figsize = (fig_w, fig_h)
        if created_fig:
            fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(1, 2, width_ratios=[1.5 * n_metrics, 1.2], wspace=0.4, **gs_kw)
        main_gs = gridspec.GridSpecFromSubplotSpec(n_cell_types, n_metrics, subplot_spec=outer[0],
                                                    hspace=cell_type_hspace, wspace=0.3)
    # When embedding, the inch->figure-fraction conversions (left margin, label offsets, cbar
    # sizes) must use the ACTUAL composite figure size, not the standalone figsize computed above.
    if not created_fig:
        fig_w, fig_h = fig.get_size_inches()

    # ensure enough left margin for the outside label + yticklabels. Multi-metric vert is
    # wider so it needs more absolute margin to keep cell-type labels off the ylabel.
    # (skip when embedding: subplots_adjust acts on the whole figure and would reflow a composite;
    # the bbox-constrained GridSpec already positions this panel.)
    if created_fig:
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

    if suptitle and created_fig:
        plt.suptitle(suptitle, fontsize=14, y=0.99)

    if save_dir:
        if folder is None and event_type == 'coding_score':
            folder = 'coding_scores_and_kernels'
        elif folder is None:
            folder = 'response_metrics'
        filename = 'metric_heatmap_grid_' + groupby_col + suffix
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        stats_suffix = _stats_suffix_for_table(stats_table)
        stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)), index=False)
    return heat_axes, stats_table


def _panel_stats_job(job):
    """Run one compute_stats call for a heatmap panel and attach caller metadata.

    Defined at module level (so it is picklable) to allow dispatch to joblib worker
    processes by _run_panel_stats_jobs. Returns (omnibus_p, panel_stats).
    """
    panel_stats = compute_stats(
        job['subset'], job['metric'], column_to_compare=job['column_to_compare'],
        use_mlm=job['use_mlm'], group_column=job['group_column'],
        event_type=job['event_type'], cell_type=job['cell_type'])
    omnibus_p = _heatmap_omnibus_p(panel_stats)
    panel_stats = insert_stats_metadata(panel_stats, **job['metadata'])
    return omnibus_p, panel_stats


def _run_panel_stats_jobs(jobs, n_jobs=None):
    """Run a list of independent panel-stats jobs, in parallel when possible.

    Each MLM fit is independent, so jobs are dispatched across processes with joblib;
    joblib preserves input order, so the concatenated result is identical to running
    them serially. Falls back to serial execution if joblib is unavailable, only one
    worker would be used, or the parallel pool fails to start -- so results are the same
    either way, only the speed differs. Set n_jobs to control worker count
    (default: cpu_count - 1).
    """
    if not jobs:
        return []
    try:
        from joblib import Parallel, delayed
    except Exception:
        return [_panel_stats_job(j) for j in jobs]
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    n_jobs = min(n_jobs, len(jobs))
    if n_jobs <= 1:
        return [_panel_stats_job(j) for j in jobs]
    try:
        return Parallel(n_jobs=n_jobs, backend='loky')(delayed(_panel_stats_job)(j) for j in jobs)
    except Exception as e:
        print('parallel stats failed (', e, ') - running serially')
        return [_panel_stats_job(j) for j in jobs]


def plot_metric_heatmap_area_and_depth_by_cell_type(
        results_pivoted, metric, metric_label=None,
        area_col='targeted_structure', depth_col='binned_depth',
        area_label='Visual area', depth_label='Imaging depth (um)',
        exp_col='experience_level', aggregate='mean',
        vmax=None, multi_star=False,
        suptitle=None, suffix='', save_dir=None, folder='response_metrics',
        group_column='mouse_id', event_type='Not specified'):
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

    # compute matrices, panel maxes, per-cell n (cheap, serial), and gather the
    # independent hierarchical-stats jobs to run in parallel below
    mats = {}
    panel_max = {}
    cell_n = {}
    col_stars = {}
    row_stars = {}
    stat_jobs = []
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
                stat_jobs.append(dict(
                    key=('col', g, i, c), subset=col_subset, metric=metric,
                    column_to_compare=gcol, use_mlm=USE_MLM, group_column=group_column,
                    event_type=event_type, cell_type=cell_type,
                    metadata=dict(grouping=gcol, direction='across_groups_within_exp',
                                  held_fixed_col=exp_col, held_fixed_value=exp)))

            # row-direction: within each grouping bin, hierarchical test (MLM by default)
            # across experience levels (e.g., "do experience levels differ at this depth?")
            for r, d in enumerate(group_order):
                row_subset = ct_data[ct_data[gcol] == d].dropna(subset=[metric, exp_col])
                stat_jobs.append(dict(
                    key=('row', g, i, r), subset=row_subset, metric=metric,
                    column_to_compare=exp_col, use_mlm=USE_MLM, group_column=group_column,
                    event_type=event_type, cell_type=cell_type,
                    metadata=dict(grouping=gcol, direction='across_exp_within_group',
                                  held_fixed_col=gcol, held_fixed_value=d)))

    # run the independent stats jobs in parallel (auto serial fallback). joblib preserves
    # input order, so col_stars/row_stars and the concatenated table match the serial version.
    job_results = _run_panel_stats_jobs(stat_jobs)
    pairwise_tables = []
    for job, (omnibus_p, panel_stats) in zip(stat_jobs, job_results):
        kind, g, i, idx = job['key']
        if kind == 'col':
            col_stars[(g, i, idx)] = tier_stars(omnibus_p)
        else:
            row_stars[(g, i, idx)] = tier_stars(omnibus_p)
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
        filename = _clean_filename('metric_heatmap_area_and_depth_' + metric + '_' + aggregate + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        stats_suffix = _stats_suffix_for_table(stats_table)
        stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)), index=False)

    return heat_axes, stats_table


def plot_bidirectional_metric_heatmap_area_and_depth_by_cell_type(
        results_pivoted, metric, metric_label=None,
        area_col='targeted_structure', depth_col='binned_depth',
        area_label='Visual area', depth_label='Imaging depth (um)',
        exp_col='experience_level',
        cmap='PRGn', vmax=None, multi_star=False,
        suptitle=None, suffix='', save_dir=None, folder='response_metrics',
        group_column='mouse_id', event_type='Not specified'):
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

    # compute matrices (means), per-cell n, panel absmax (cheap, serial), and gather the
    # independent hierarchical-stats jobs to run in parallel below
    mats = {}
    panel_absmax = {}
    cell_n = {}
    col_stars = {}
    row_stars = {}
    stat_jobs = []
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
                stat_jobs.append(dict(
                    key=('col', g, i, c), subset=col_subset, metric=metric,
                    column_to_compare=gcol, use_mlm=USE_MLM, group_column=group_column,
                    event_type=event_type, cell_type=cell_type,
                    metadata=dict(grouping=gcol, direction='across_groups_within_exp',
                                  held_fixed_col=exp_col, held_fixed_value=exp)))

            for r, d in enumerate(group_order):
                row_subset = ct_data[ct_data[gcol] == d].dropna(subset=[metric, exp_col])
                stat_jobs.append(dict(
                    key=('row', g, i, r), subset=row_subset, metric=metric,
                    column_to_compare=exp_col, use_mlm=USE_MLM, group_column=group_column,
                    event_type=event_type, cell_type=cell_type,
                    metadata=dict(grouping=gcol, direction='across_exp_within_group',
                                  held_fixed_col=gcol, held_fixed_value=d)))

    # run the independent stats jobs in parallel (auto serial fallback). joblib preserves
    # input order, so col_stars/row_stars and the concatenated table match the serial version.
    job_results = _run_panel_stats_jobs(stat_jobs)
    pairwise_tables = []
    for job, (omnibus_p, panel_stats) in zip(stat_jobs, job_results):
        kind, g, i, idx = job['key']
        if kind == 'col':
            col_stars[(g, i, idx)] = tier_stars(omnibus_p)
        else:
            row_stars[(g, i, idx)] = tier_stars(omnibus_p)
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
        filename = _clean_filename('bidirectional_metric_heatmap_area_and_depth_' + metric + suffix)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        stats_suffix = _stats_suffix_for_table(stats_table)
        stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)), index=False)

    return heat_axes, stats_table


def plot_experience_modulation_heatmap_area_and_depth_by_cell_type(
        metrics_table, event_type, metric, cells_table,
        area_col='targeted_structure', depth_col='binned_depth',
        area_label='Visual area', depth_label='Imaging depth (um)',
        all_comparisons=True, vmax=None, multi_star=False,
        suptitle=None, suffix='', save_dir=None, folder='response_metrics',
        group_column='mouse_id'):
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
                                            use_mlm=USE_MLM, group_column=group_column,
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
                                            use_mlm=USE_MLM, group_column=group_column,
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
        filename = _clean_filename(('experience_modulation_heatmap_area_and_depth_by_cell_type_'
                    + event_type + suffix))
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))
        stats_suffix = _stats_suffix_for_table(stats_table)
        stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + stats_suffix)),
                           index=False)

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
                                                  save_dir=None, folder='physio_behavior_correlation'):
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
        show_fit=False, 
        save_dir=None, folder='physio_behavior_correlation'):

    '''Plot the difference in metrics (F to N) for behavior vs. neural activity.

    show_fit: if True, overlay a dashed best-fit line per panel. Default False because
    these data are typically scattered and a fit line implies a relationship the data
    doesn't support.
    '''
    from scipy.stats import pearsonr

    x = get_metric_index_name(behavior_metric)
    y = get_metric_index_name(cell_metric)

    if metric_label is None:
        metric_label=cell_metric
    if behavior_label is None:
        behavior_label=behavior_metric
    # if suptitle is None:
    #     suptitle = f'Change in {metric_label} vs. change in {behavior_label}'

    cell_types = utils.get_cell_types()

    figsize = (12, 3)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    for i, cell_type in enumerate(cell_types):
        ct_data = metric_data[metric_data.cell_type == cell_type]
        sns.scatterplot(data=ct_data, x=x, y=y, ax=ax[i])
        n_mice = ct_data['mouse_id'].nunique()

        # Pearson r and p across mice (drop NaNs)
        xy = ct_data[[x, y]].dropna()
        if len(xy) >= 2 and xy[x].std() > 0 and xy[y].std() > 0:
            r, p = pearsonr(xy[x].values, xy[y].values)
            p_str = f"{p:.2e}" if p < 1e-3 else f"{p:.2f}"
            r_p_str = f"r = {r:.2f}, p = {p_str}"
            if show_fit:
                slope, intercept = np.polyfit(xy[x].values, xy[y].values, 1)
                xf = np.array([xy[x].min(), xy[x].max()])
                ax[i].plot(xf, slope * xf + intercept, color='black',
                           linestyle='--', linewidth=1.5)
        else:
            r_p_str = "r = nan, p = nan"
        # ax[i].set_title(f'{cell_type} (n = {n_mice} mice)\n{r_p_str}')
        ax[i].set_title(f'{cell_type}\n{r_p_str}')
        ax[i].set_xlabel(behavior_label+'\nChange from F to N')
        ax[i].set_ylabel(metric_label+'\nChange from F to N')
        ax[i].axhline(y=0, linestyle='--', color='gray', linewidth=1)
        ax[i].axvline(x=0, linestyle='--', color='gray', linewidth=1)
    if suptitle is not None: 
        plt.suptitle(suptitle, x=0.5, y=1.2, fontsize=20)
    plt.subplots_adjust(wspace=0.3)

    if save_dir:
        filename = _clean_filename('difference_' + cell_metric + '_' + behavior_metric)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename))



def plot_modulation_index_correlation(data, x_col, y_col, xlabel, ylabel,
                            ax=None, save_dir=None, folder='response_metrics', filename=None):
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
                               suptitle=None, save_dir=None, folder='response_metrics', ax=None):
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
                                        markers='.', markersize=8, err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i],
                                        estimator="mean",
                                        n_boot=1000)
        elif plot_type == 'barplot': 
            ax[i] = sns.barplot(data=data, x=x_val, y=metric, order=x_vals, hue=x_val, legend=False,
                                        palette=palette, hue_order=x_vals, width=0.5, alpha=0.75, 
                                        err_kws={'linewidth': 2}, errorbar=('ci', 95), ax=ax[i],
                                        estimator="mean",
                                        n_boot=1000)
        elif plot_type == 'boxplot': 
            ax[i] = sns.boxplot(data=data, x=x_val, y=metric, order=x_vals, hue=x_val, legend=False,
                                        palette=palette, hue_order=x_vals, width=0.5, fliersize=0, ax=ax[i],
                                        whis=1.5)
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


def collapse_duplicate_cell_rows(df, group_cols, trace_col='mean_trace'):
    """Collapse rows that share the same `group_cols` into a single averaged row.

    A cell_specimen_id can appear more than once at a given matched condition (e.g. a
    container with multiple sessions at the same experience level). For cell-matched
    plotting we need exactly one row per cell per condition, otherwise the heatmap rows
    no longer correspond to the same cell across columns. Rather than dropping these
    cells, average their response trace (and any numeric metric columns) so the cell is
    retained. Non-numeric columns take the first value within each group.

    Returns `df` unchanged if there are no duplicate `group_cols` combinations.
    """
    if not df.duplicated(subset=group_cols).any():
        return df

    has_trace = trace_col in df.columns
    grouped = df.groupby(group_cols, sort=False)

    # mean for numeric metric columns, first value for everything else
    agg = {}
    for col in df.columns:
        if col in group_cols or (has_trace and col == trace_col):
            continue
        agg[col] = 'mean' if pd.api.types.is_numeric_dtype(df[col]) else 'first'
    collapsed = grouped.agg(agg)

    # average the response trace (array per row) across the duplicate rows
    if has_trace:
        collapsed[trace_col] = grouped[trace_col].apply(lambda s: np.mean(np.vstack(s.values), axis=0))

    return collapsed.reset_index()


def plot_response_heatmaps_for_conditions(multi_session_df, timestamps, data_type, event_type,
                                          row_condition, col_condition, matched_cells_table=None, 
                                          plot_epochs=False, exp_to_match='Familiar',
                                          col_to_sort_by='mean_response', cell_order=None, suptitle=None,
                                          microscope=None, vmax=None, xlim_seconds=None, xlabel='time (s)',
                                          match_cells=False, cbar=True, cbar_label='Avg. calcium events',
                                          save_dir=None, folder='population_activity', suffix='', ax=None):
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

            # Collapse cells that appear more than once at the same matched condition
            # (e.g. multiple sessions at one experience level) into a single averaged
            # row, so the cell is retained instead of dropped and the heatmap rows stay
            # aligned to the same cell_specimen_id across columns.
            if plot_epochs:
                collapse_keys = ['cell_specimen_id', col_condition, 'epoch']
            else:
                collapse_keys = ['cell_specimen_id', col_condition]
            n_before = cre_sdf.cell_specimen_id.nunique()
            cre_sdf = collapse_duplicate_cell_rows(cre_sdf, collapse_keys)

            # Keep only cells present in every matched condition so reindexing cannot
            # introduce empty (NaN) rows that would break cell alignment.
            n_levels = len(col_conditions)
            levels_per_cell = cre_sdf.groupby('cell_specimen_id')[col_condition].nunique()
            complete_cells = levels_per_cell[levels_per_cell == n_levels].index
            n_dropped = n_before - len(complete_cells)
            if n_dropped > 0:
                print(f'{row}: dropping {n_dropped} of {n_before} matched cells not present '
                      f'in all {n_levels} {col_condition} conditions')
            cre_sdf = cre_sdf[cre_sdf.cell_specimen_id.isin(complete_cells)]

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
                # guarantee every heatmap column shows the same cell_specimen_id per row
                assert np.array_equal(np.asarray(data.index), np.asarray(novel_cell_order)), \
                    f'Matched-cell order mismatch for {row} / {col}: heatmap rows are not aligned to the matched cell order'
                if not plot_epochs:
                    assert not data.isnull().values.any(), \
                        f'Matched-cell heatmap for {row} / {col} has empty (NaN) rows from missing cells'

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
                                          save_dir=None, folder='image_tuning', suffix='', ax=None):
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
                                          save_dir=None, folder='image_tuning', suffix='', ax=None):
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
                                        save_dir=None, folder='example_cells'):
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
                                       abbreviate_exp=True, save_dir=None, folder='behavior_metrics', suffix='', ax=None,
                                       group_column='mouse_id'):
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
        save_fig = True
    else: 
        save_fig = False

    if stripplot:
        ax = sns.stripplot(data=data, x='experience_level', y=metric, order=experience_levels,
                       orient='v', color='gray', dodge=True, jitter=0.1, size=2, ax=ax, zorder=0)
    if show_mice:
        for mouse_id in data.mouse_id.unique():
            ax = sns.pointplot(data=data[data.mouse_id == mouse_id], x='experience_level', y=metric,
                               order=experience_levels, linewidth=0.5, orient='v', color='gray',
                               markers='.', markersize=0.15, err_kws={'linewidth': 0.5}, ax=ax,
                               estimator="mean",
                               errorbar=("ci", 95),
                               n_boot=1000)
        # suffix = suffix + '_show_mice'

    if pointplot:
        ax = sns.pointplot(data=data, x='experience_level', y=metric, order=experience_levels,
                       orient='v', palette=colors, ax=ax, hue='experience_level', hue_order=experience_levels,legend=False,
                       markers='.', markersize=8, err_kws={'linewidth': 2},estimator="mean",
                       errorbar=("ci", 95),
                       n_boot=1000,
                       )
    else:
        ax = sns.boxplot(data=data, x='experience_level', y=metric, order=experience_levels,
                            hue='experience_level', legend=False, hue_order=experience_levels,
                           orient='v', palette=colors, width=0.6, boxprops=dict(alpha=0.8), ax=ax,
                            whis=1.5)

    ax.set_xlim(-0.5, len(experience_levels)-0.5)
    if abbreviate_exp:
        ax.set_xticks(range(len(experience_levels)))
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
                                            group_column=group_column,
                                            event_type='behavior')
        stats_table = insert_stats_metadata(stats_table, condition='experience_level')

    ax.set_ylim(ymin=ymin)
    plt.subplots_adjust(top=0.96)
    plt.subplots_adjust(hspace=0.3)
    if save_fig and save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metric + suffix))
    stats_filename = _clean_filename(metric + '_stats' + suffix)
    # try:
    if save_dir: 
        if plot_stats:
            print('saving_stats')
            stats_suffix = _stats_suffix_for_table(stats_table)
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
        # save metric values
        cols_to_groupby = ['experience_level']
        descriptive_stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        descriptive_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
        # except BaseException:
        #     print('stats did not save for', metric)
    return ax


def plot_response_rate_by_trial_type(behavior_stats, metric='response_probability', fraction_engaged_thresh=0.7,
                                     title='', ylabel='Response rate', ylims=(-0.01, 1),
                                     save_dir=None, folder='behavior_metrics', suffix='', ax=None,
                                     group_column='mouse_id'):
    """
    plots response rate across trial types (change, non-change, omission, post-omission), split by experience level,
    as a boxplot using experience level colors. Stats are computed across experience levels within each trial type
    (using a hierarchical mixed linear model by default) and saved out alongside the figure.

    behavior_stats: stimulus-based behavior metrics table (e.g. platform_behavior_stats), already limited to the
        sessions of interest. Must contain columns 'fraction_engaged', 'trial_type', 'experience_level',
        the metric column, and group_column.
    metric: column in behavior_stats containing the response rate values (default 'response_probability')
    fraction_engaged_thresh: only sessions with fraction_engaged greater than this value are included
    MLM vs ANOVA/Tukey is set by the module-level ``USE_MLM`` constant: when True, use hierarchical mixed linear model with random intercept for group_column for stats
    group_column: nesting variable for MLM (e.g., 'mouse_id')

    returns axis handle
    """
    suffix = _norm_suffix(suffix)

    # limit to engaged sessions and the trial types of interest, renaming 'could_change' to 'non-change'
    data = behavior_stats.copy()
    # only filter on fraction_engaged when that column is present (the long-form stimulus-based
    # response_rate_df has one row per session x trial type and carries no fraction_engaged column)
    if 'fraction_engaged' in data.columns:
        data = data[data.fraction_engaged > fraction_engaged_thresh]
    data = data[data.trial_type.isin(['change', 'could_change', 'omission', 'post-omission'])]
    data['trial_type'] = ['non-change' if trial_type == 'could_change' else trial_type
                          for trial_type in data.trial_type.values]

    trial_types = data.trial_type.unique()

    if ax is None:
        figsize = (5, 3)
        fig, ax = plt.subplots(figsize=figsize)
        save_fig = True
    else:
        save_fig = False

    ax = sns.boxplot(data=data, x='trial_type', y=metric, hue='experience_level',
                     order=trial_types, palette=utils.get_experience_level_colors(),
                     width=0.6, boxprops=dict(alpha=0.7), ax=ax,
                     whis=1.5)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([str(t)[:1].upper() + str(t)[1:] for t in trial_types], rotation=45)
    ax.set_xlabel('')
    if ylims is not None:
        ax.set_ylim(ylims[0], ylims[1])
    ax.legend(loc='upper right', fontsize='xx-small')
    ax.set_title(title)

    # stats across experience levels within each trial type
    ax, stats_table = add_stats_to_plot_for_hues(data, metric, ax, ymax=None,
                                                 xorder=trial_types, x='trial_type', hue='experience_level',
                                                 group_column=group_column,
                                                 event_type='behavior')
    # data_subset (trial_type) is added inside add_stats_to_plot_for_hues; add condition after cell_type
    # so condition lands directly before data_subset in the saved CSV
    stats_table = insert_stats_metadata(stats_table, condition='experience_level')

    plt.subplots_adjust(top=0.96)
    if save_dir:
        filename = _clean_filename(metric + '_by_trial_type' + suffix)
        if save_fig:
            utils.save_figure(fig, figsize, save_dir, folder, filename)
        print('saving_stats')
        stats_suffix = _stats_suffix_for_table(stats_table)
        stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + '_stats' + stats_suffix)))
        # save descriptive metric values
        cols_to_groupby = ['trial_type', 'experience_level']
        descriptive_stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        descriptive_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(filename + '_stats_values.csv')))
    return ax


def plot_behavior_metric_by_experience_horiz(stats, metric, title='', xlabel='', xlims=None, best_image=True, show_containers=False,
                                       stripplot=True, pointplot=True, plot_stats=False, show_ns=False, save_dir=None, folder='behavior_metrics', suffix='', ax=None,
                                       group_column='mouse_id'):
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
        save_fig = True
    else:
        save_fig = False

    if stripplot:
        ax = sns.stripplot(data=data, y='experience_level', x=metric, order=experience_levels,
                       orient='h', color='gray', dodge=True, jitter=0.1, size=2, ax=ax, zorder=0)
    if show_containers:
        for ophys_container_id in data.ophys_container_id.unique():
            ax = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], y='experience_level', x=metric,
                               order=experience_levels, linewidth=0.5, orient='h', color='gray',
                               markers='.', markersize=0.15, err_kws={'linewidth': 0.5}, ax=ax,
                               estimator="mean",
                               errorbar=("ci", 95),
                               n_boot=1000)
        # suffix = suffix + '_show_mice'

    if pointplot:
        ax = sns.pointplot(data=data, y='experience_level', x=metric, order=experience_levels,
                        markers='.', markersize=8, err_kws={'linewidth': 2}, orient='h', palette=colors, ax=ax,
                        estimator="mean",
                        errorbar=("ci", 95),
                        n_boot=1000)
    else:
        ax = sns.boxplot(data=data, y='experience_level', x=metric, order=experience_levels,
                           orient='h', palette=colors, width=0.6, boxprops=dict(alpha=0.8), ax=ax,
                           whis=1.5)

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
                                            group_column=group_column,
                                            event_type='behavior')
        stats_table = insert_stats_metadata(stats_table, condition='experience_level')

    ax.set_xlim(xmin=xmin)

    if save_fig and save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metric + '_horiz' + suffix))
    stats_filename = _clean_filename(metric + '_stats' + suffix)
    if save_dir:
        if plot_stats:
            print('saving_stats')
            stats_suffix = _stats_suffix_for_table(stats_table)
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
        # save metric values
        cols_to_groupby = ['experience_level']
        descriptive_stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        descriptive_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
    return ax



def plot_behavior_metric_by_cohort(stats, metric, title='', ylabel='', ylims=None, show_containers=False,
                                       stripplot=True, pointplot=True, plot_stats=False, show_ns=False,
                                   save_dir=None, folder='behavior_metrics', suffix='', ax=None,
                                   group_column='mouse_id'):
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
        save_fig = True
    else:
        save_fig = False

    if stripplot:
        ax = sns.stripplot(data=data, x='project_code', y=metric, order=project_codes,
                       orient='v', color='gray', dodge=True, jitter=0.1, size=2, ax=ax, zorder=0)
    if show_containers:
        for ophys_container_id in data.ophys_container_id.unique():
            ax = sns.pointplot(data=data[data.ophys_container_id == ophys_container_id], x='project_code', y=metric,
                               order=project_codes, linewidth=0.5, orient='v', color='gray',
                               markers='.', markersize=0.15, err_kws={'linewidth': 0.5}, ax=ax,
                               estimator="mean",
                               errorbar=("ci", 95),
                               n_boot=1000)
        # suffix = suffix + '_show_mice'

    if pointplot:
        ax = sns.pointplot(data=data, x='project_code', y=metric, order=project_codes,
                       orient='v', palette=colors, markers='.', ax=ax,
                       estimator="mean",
                       errorbar=("ci", 95),
                       n_boot=1000) # marker_kws={'size':2},
    else:
        ax = sns.boxplot(data=data, x='project_code', y=metric, order=project_codes,
                           orient='v', palette=colors, width=0.6, boxprops=dict(alpha=0.8), ax=ax,
                           whis=1.5)

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
                                            group_column=group_column,
                                            event_type='behavior')
        stats_table = insert_stats_metadata(stats_table, condition='experience_level')

    ax.set_ylim(ymin=ymin)

    if save_fig and save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(metric + suffix))
    stats_filename = _clean_filename(metric + '_stats' + suffix)
    if save_dir:
        if plot_stats:
            print('saving_stats')
            stats_suffix = _stats_suffix_for_table(stats_table)
            stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + stats_suffix)))
        # save metric values
        cols_to_groupby = ['project_code']
        descriptive_stats = get_descriptive_stats_for_metric(data, metric, cols_to_groupby)
        descriptive_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(stats_filename + '_values.csv')))
    return ax


def plot_behavior_metric_across_stages(data, metric, ylabel=None, ax=None,
                                       save_dir=None, folder='behavior_metrics', suffix=''):
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

    if ax is None:
        figsize = (7, 3)
        fig, ax = plt.subplots(figsize=figsize)
        save_fig = True
    else:
        save_fig = False
    ax = sns.boxplot(data=data, x='cell_type', y=metric, width=0.8, order=cell_types,
                     hue='behavior_stage', hue_order=behavior_stages, palette=colors, ax=ax,
                     whis=1.5)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.legend().remove()
    ax.legend(bbox_to_anchor=(1, 1), fontsize='x-small')

    if save_fig and save_dir:
        fig.subplots_adjust(hspace=0.3)
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('metric_across_stages_' + metric + suffix))
    if save_dir:
        # save stats
        stats = data.groupby(['cell_type', 'behavior_stage']).describe()[[metric]]
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename('metric_across_stages_' + metric + suffix + '_values.csv')))
    return ax


def plot_days_in_stage(behavior_sessions, stage_column, save_dir=None, folder='training_history', suffix=None, ax=None):
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
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    order = np.sort(data.cell_type.unique())
    ax = sns.boxplot(data=data, x='cell_type', y='days_in_stage', order=order, width=0.8, linewidth=0.8,
                     hue=stage_column, hue_order=behavior_stages, palette=colors, ax=ax,
                     whis=1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Days in stage')
    ax.legend().remove()
    ax.legend(bbox_to_anchor=(1, 1), fontsize='x-small')

    if created_fig:
        fig.subplots_adjust(hspace=0.3)
    if created_fig and save_dir:
        # save plot
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('days_in_stage' + '_' + stage_column + suffix))
        # save stats
        days_in_stage_stats = data.groupby(['cell_type', stage_column]).describe()
        days_in_stage_stats.to_csv(os.path.join(save_dir, folder, _clean_filename('days_in_stage_values.csv')))
    return ax


def plot_prior_exposures_to_image_set_before_platform_ophys_sessions(platform_experiments, behavior_sessions, 
                        title='Stimulus exposure', save_dir=None, folder='stimulus_history', suffix='', ax=None):
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
        save_fig = True
    else:
        save_fig = False

    colors = utils.get_experience_level_colors()
    experience_levels = np.sort(platform_experiments.experience_level.unique())

    ax = sns.boxplot(data=exposures, x='experience_level', y='prior_exposures_to_image_set',
                     order=experience_levels, palette=colors, width=0.5, ax=ax,
                     whis=1.5)
    ax.set_ylabel('# sessions')
    ax.set_xlabel('')

    stats = exposures.groupby(['experience_level']).describe()[['prior_exposures_to_image_set']]
    stats.columns = stats.columns.droplevel(0)

    # xticklabels = utils.get_new_experience_levels()
    ax.set_xticklabels(experience_levels, rotation=90)
    ax.set_title(title)

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

    if save_fig and save_dir:
        # save plot
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('stimulus_exposures_before_platform_expts_boxplot' + suffix))
    if save_dir:
        # save stats
        stats = exposures.groupby(['experience_level']).describe()[['prior_exposures_to_image_set']]
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename('stimulus_exposures_before_platform_expts_values.csv')))


def plot_prior_exposures_per_cell_type_for_novel_plus(platform_experiments, behavior_sessions, save_dir=None,
                                                      folder='stimulus_history', suffix='', ax=None, show_ns=True,
                                                      group_column='mouse_id'):
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
        save_fig = True
    else:
        save_fig = False

    #     ax = sns.boxplot(data=exposures, x='cell_type', y='prior_exposures_to_image_set',
    #                order=cell_types, palette='gray', width=0.5, ax=ax)

    ax = sns.violinplot(data=exposures, x='cell_type', y='prior_exposures_to_image_set', order=cell_types,
                        orient='v', palette='dark:white', ax=ax,
                        inner="box",
                        density_norm="area",
                        bw_method="scott",
                        cut=2)
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
                                        show_ns=show_ns, column_to_compare='cell_type',
                                        group_column=group_column,
                                        event_type='session_metadata')
    stats_table = insert_stats_metadata(stats_table, condition='experience_level')

    if save_fig and save_dir:
        # save plot
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('stimulus_exposures_before_novel_plus' + suffix))
    if save_dir:
        # save stats
        print('saving_stats')
        stats_suffix = _stats_suffix_for_table(stats_table)
        stats_table.to_csv(os.path.join(save_dir, folder, _clean_filename('stimulus_exposures_before_novel_plus' + stats_suffix)))
        descriptive_stats = exposures.groupby(['cell_type', 'experience_level']).describe()[['prior_exposures_to_image_set']]
        descriptive_stats.to_csv(os.path.join(save_dir, folder, _clean_filename('stimulus_exposures_before_novel_plus_values.csv')))


def plot_prior_exposures_to_image_set_before_platform_ophys_sessions_horiz(platform_experiments, behavior_sessions, save_dir=None, folder='stimulus_history', suffix='', ax=None):
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
        save_fig = True
    else:
        save_fig = False

    colors = utils.get_experience_level_colors()
    experience_levels = np.sort(platform_experiments.experience_level.unique())

    ax = sns.boxplot(data=exposures, y='experience_level', x='prior_exposures_to_image_set', orient='h',
                     order=experience_levels, palette=colors, width=0.5, ax=ax,
                     whis=1.5)
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

    if save_fig and save_dir:
        # save plot
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('stimulus_exposures_before_platform_expts_boxplot_horiz' + suffix))
    if save_dir:
        # save stats
        stats = exposures.groupby(['experience_level']).describe()[['prior_exposures_to_image_set']]
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename('stimulus_exposures_before_platform_expts_values.csv')))

def plot_total_stimulus_exposures(behavior_sessions, save_dir=None, folder='stimulus_history', suffix='', ax=None):
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
        save_fig = True
    else:
        save_fig = False

    colors = utils.get_experience_level_colors()
    experience_levels = utils.get_experience_levels()
    new_experience_levels = utils.get_new_experience_levels()

    ax = sns.boxplot(data=exposures, x='experience_level', y='n_sessions',
                     order=experience_levels, palette=colors, width=0.5, ax=ax,
                     whis=1.5)
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

    if save_fig and save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('total_stimulus_exposures_all_sessions_boxplot' + suffix))
    if save_dir:
        # save stats
        stats = exposures.groupby(['experience_level']).describe()[['n_sessions']]
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename('total_stimulus_exposures_all_sessions_values.csv')))


def plot_stimulus_exposure_prior_to_imaging(behavior_sessions, column_to_group='behavior_stage', title='Stimulus exposure\nduring training', 
                                            save_dir=None, folder='stimulus_history', suffix='', ax=None):
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
        save_fig = True
    else:
        save_fig = False

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
                     order=col_values, palette=c, width=0.5, ax=ax,
                     whis=1.5)
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

    if save_fig and save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename('stimulus_exposure_prior_to_imaging_boxplot_'+ column_to_group + suffix))
    if save_dir:
        # save stats
        stats = exposures.groupby(column_to_group).describe()[['n_sessions']]
        stats.to_csv(os.path.join(save_dir, folder, _clean_filename('stimulus_exposure_prior_to_imaging_'+column_to_group+'_values.csv')))

def plot_training_history_for_mice(behavior_sessions, color_column='session_type', color_map=sns.color_palette(),
                                   group_by_cre_line=True, save_dir=None, folder='training_history', suffix='', ax=None):
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
    ax = sns.pointplot(data=data, x=x, y=metric, hue=hue, hue_order=data[hue].unique(), linestyle='None', palette=colors, ax=ax, estimator="mean", errorbar=("ci", 95), n_boot=1000)
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
                    order=trial_types, color='k', ax=ax,
                    estimator="mean",
                    errorbar=("ci", 95),
                    n_boot=1000)
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


def plot_response_probability_heatmaps_for_cohorts(behavior_sessions, save_dir=None, axes=None, cbar_ax=None):
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

    cmap = 'Greys'
    colors = utils.get_colors_for_session_numbers()

    # composite mode: draw only the main-cohort (image set A) Familiar and (image set B) Novel
    # heatmaps into the two provided axes, with a single shared colorbar on the second.
    if axes is not None:
        familiar_data = familiar_response_probability[familiar_response_probability.project_code.isin(['VisualBehavior', 'VisualBehaviorMultiscope'])]
        novel_data = novel_response_probability[novel_response_probability.project_code.isin(['VisualBehavior', 'VisualBehaviorMultiscope'])]
        familiar_response_matrix = behavior.average_response_probability_across_sessions(familiar_data, sort=True)
        novel_response_matrix = behavior.average_response_probability_across_sessions(novel_data, sort=True)
        # both heatmaps drawn without an inline colorbar so they stay the same size; the colorbar
        # goes in its own cbar_ax. Force all image-name tick labels regardless of axis size.
        sns.heatmap(familiar_response_matrix, cmap=cmap, vmin=0, vmax=1, square=True, cbar=False,
                    xticklabels=True, yticklabels=True, ax=axes[0])
        axes[0].set_xlabel('Change image'); axes[0].set_ylabel('Initial image')
        axes[0].set_title('Familiar images', color=colors[3])
        draw_cbar = cbar_ax is not None
        sns.heatmap(novel_response_matrix, cmap=cmap, vmin=0, vmax=1, square=True,
                    cbar=draw_cbar, cbar_ax=cbar_ax,
                    cbar_kws={'label': 'Response probability'} if draw_cbar else None,
                    xticklabels=True, yticklabels=True, ax=axes[1])
        axes[1].set_xlabel('Change image'); axes[1].set_ylabel('Initial image')
        axes[1].set_title('Novel images', color=colors[0])
        for _a in axes:
            _a.tick_params(labelsize=10)
            plt.setp(_a.get_xticklabels(), rotation=90)
            plt.setp(_a.get_yticklabels(), rotation=0)
        return axes

    # make the plot
    figsize = (10,10)
    fig, ax = plt.subplots(2,2, figsize=figsize)
    ax = ax.ravel()

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
                                           filename='percent_cells_coding_feature', suffix='',
                                           fig=None, bbox=None):
    """
    Bar plot of the percent of cells coding for each feature, one panel per cell type.

    Expects the long-form `fractions` dataframe produced by
    `compute_feature_coding_fractions`.

    Embedding: pass `fig` (and an optional figure-fraction `bbox` sub-rectangle)
    to draw into an existing composite figure; in that mode the standalone
    side effects (figure creation, subplots_adjust, save) are skipped.
    """
    from visual_behavior.dimensionality_reduction.clustering import plotting
    from visual_behavior.dimensionality_reduction.clustering import processing as processing

    cell_types = utils.get_cell_types()
    features = processing.get_feature_labels_for_clustering()
    feature_colors, _ = plotting.get_feature_colors_and_labels()

    figsize = (8, 2.5)
    standalone = fig is None
    if standalone:
        fig, ax = plt.subplots(1, len(cell_types), figsize=figsize, sharey=True, sharex=True)
        ax = ax.ravel()
    else:
        ax = utils.placeAxesOnGrid(fig, dim=(1, len(cell_types)),
                                   xspan=(0, 1), yspan=(0, 1),
                                   wspace=0.3, sharey=True, sharex=True, bbox=bbox)
        ax = np.array(ax).ravel()

    for i, cell_type in enumerate(cell_types):
        ct_data = fractions[fractions.cell_type == cell_type]
        ax[i] = sns.barplot(data=ct_data, x='feature', y='percent', order=features,
                            hue='feature', hue_order=features, legend=False,
                            palette=feature_colors, width=0.8, alpha=0.75, ax=ax[i],
                            estimator="mean",
                            errorbar=("ci", 95),
                            n_boot=1000)
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

        ax[i].set_xticks(range(len(features)))
        ax[i].set_xticklabels(features, rotation=45, ha='right', fontsize=14)
        [t.set_color(c) for (c, t) in zip(feature_colors[:len(features)], ax[i].xaxis.get_ticklabels())]

    if standalone:
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
                                                  suffix='', group_column='mouse_id',
                                                  event_type='coding_score',
                                                  fig=None, bbox=None):
    """
    Grid of boxplots (rows: cre line, cols: feature) of coding scores by
    experience level, with significance annotations. Expects the long-form
    output of `convert_coding_scores_to_long_form_df`.

    Embedding: pass `fig` (and an optional figure-fraction `bbox` sub-rectangle)
    to draw into an existing composite figure; in that mode the standalone
    side effects (figure creation, subplots_adjust, save) are skipped.

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
    standalone = fig is None
    if standalone:
        fig, ax = plt.subplots(len(cre_lines), len(features), figsize=figsize, sharey=True)
        ax = ax.ravel()
    else:
        ax = utils.placeAxesOnGrid(fig, dim=(len(cre_lines), len(features)),
                                   xspan=(0, 1), yspan=(0, 1),
                                   wspace=0.3, hspace=0.3, sharey=True, bbox=bbox)
        ax = np.array(ax).ravel()
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
                group_column=group_column,
                event_type=event_type, cell_type=cell_type,
            )
            # Attach the per-panel iteration variable (feature) that compute_stats
            # can't see on its own.
            panel_stats = insert_stats_metadata(panel_stats, feature=feature)
            combined_stats = pd.concat([combined_stats, panel_stats])

            ax[i].set_ylim(-0.1, 1.2)
            i += 1

    if standalone:
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
    if standalone and save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, filename + suffix)
        try:
            print('saving_stats')
            stats_suffix = _stats_suffix_for_table(combined_stats)
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
                                              event_type='coding_score', legend=True, compact_bars=True,
                                              save_dir=None, folder='coding_scores_and_kernels',
                                              filename=None, suffix='',
                                              group_column='mouse_id'):
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
            event_type=event_type, compact_bars=compact_bars,
            save_dir=save_dir, folder=folder, ax=ax[:, col],
            group_column=group_column,
        )
        # Strip duplicated y-axis labels in non-leftmost columns.
        if col > 0:
            for row in range(3):
                ax[row, col].set_ylabel('')

    # The inner function may leave a legend on a per-column panel (usually top-left).
    # Collect the hue handles, clear every panel's legend, then -- if requested --
    # put a single legend on the TOP-RIGHT axis, anchored just outside the axes.
    handles, labels = ax[0, 3].get_legend_handles_labels()
    if not handles:
        for a in ax.ravel():
            h, l = a.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    for a in ax.ravel():
        leg = a.get_legend()
        if leg is not None:
            leg.remove()
    if legend and handles:
        ax[0, 3].legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left',
                        fontsize='xx-small', title=hue.replace('_', ' '),
                        title_fontsize='xx-small')

    wspace = 0.3 if (hue == 'binned_depth' or x_val == 'binned_depth') else 0.4
    plt.subplots_adjust(wspace=wspace, hspace=0.3)

    if save_dir:
        if filename is None:
            filename = 'coding_scores_for_' + hue + '_by_' + x_val + '_' + plot_type
        utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(filename + suffix))

    return fig, ax


# =============================================================================
# GLM coding-score figures (ported from visual_behavior_glm.GLM_visualization_tools)
#
# These reproduce the Figure 4 supplemental coding-score / variance-explained
# panels that were previously made by ``gvt.*`` functions in the visual_behavior_glm
# repo, but re-implemented here so they (a) follow the platform_paper_figures
# conventions (save_dir/folder/suffix, save_figure, _values.csv export) and
# (b) run their statistics through the same MLM stack as the rest of this module
# (``compute_stats`` / ``add_stats_to_plot*``, grouped by ``mouse_id``, gated by the
# module-level ``USE_MLM`` switch).
#
# Input data (built in figure_4_supplemental.ipynb):
#   - results_pivoted : one row per cell per experiment; wide dropout (coding score)
#       columns ('all-images', 'omissions', 'behavioral', 'task', ...), plus
#       'variance_explained_full', 'cre_line', 'experience_level', 'cell_specimen_id',
#       'ophys_experiment_id', and (in the notebook) 'cell_type', 'binned_depth'.
#   - results : long format, one row per cell per dropout, with 'dropout',
#       'adj_fraction_change_from_full', 'cre_line', 'variance_explained_full'.
# =============================================================================

# cre_line -> cell_type mapping used throughout the GLM results tables
_GLM_CRE_TO_CELL_TYPE = {
    'Slc17a7-IRES2-Cre': 'Excitatory',
    'Sst-IRES-Cre': 'Sst Inhibitory',
    'Vip-IRES-Cre': 'Vip Inhibitory',
}


def _coarse_bin_depth(imaging_depth):
    """Bin imaging depth into 'upper' (<250 um) / 'lower' (>=250 um), matching
    gvt.coarse_bin_depth."""
    return 'upper' if imaging_depth < 250 else 'lower'


def _glm_add_cell_type(df):
    """Add a 'cell_type' column from 'cre_line' if it is not already present."""
    if 'cell_type' not in df.columns and 'cre_line' in df.columns:
        df['cell_type'] = df['cre_line'].map(_GLM_CRE_TO_CELL_TYPE)
    return df


def _prepare_glm_metrics_table(results_pivoted, experiment_table=None, include_4x2_data=False,
                               drop_passive=True, include_zero_cells=True):
    """
    Coerce a GLM ``results_pivoted`` dataframe into the platform ``metrics_table``
    shape expected by the stats helpers: one row per cell_specimen_id per
    ophys_experiment_id, with 'cell_type', platform-named 'experience_level', and a
    'mouse_id' column (the MLM grouping variable).

    - drops passive sessions (matches the gvt originals) when ``drop_passive``
    - if ``include_zero_cells`` is False, keeps only cells with
      variance_explained_full > 0.005 (matches the gvt originals)
    - converts experience-level names via utils.convert_experience_level
      ('Novel 1'->'Novel', 'Novel >1'->'Novel +')
    - merges any missing metadata columns (mouse_id, cell_type, targeted_structure,
      binned_depth, imaging_depth, equipment_name) from the platform experiment table
    """
    data = results_pivoted.copy()
    if drop_passive and 'passive' in data.columns:
        data = data[data.passive == False].copy()  # noqa: E712
    if (not include_zero_cells) and ('variance_explained_full' in data.columns):
        data = data[data.variance_explained_full > 0.005].copy()
    if 'experience_level' in data.columns:
        data['experience_level'] = [utils.convert_experience_level(e)
                                    for e in data['experience_level'].values]
    data = _glm_add_cell_type(data)
    # ensure metadata needed for plotting / MLM grouping is present
    wanted = ['mouse_id', 'cell_type', 'targeted_structure', 'binned_depth',
              'imaging_depth', 'equipment_name']
    missing = [c for c in wanted if c not in data.columns]
    if missing and ('ophys_experiment_id' in data.columns):
        if experiment_table is None:
            experiment_table = loading.get_platform_paper_experiment_table(
                include_4x2_data=include_4x2_data)
        et = experiment_table.reset_index()
        merge_cols = ['ophys_experiment_id'] + [c for c in missing if c in et.columns]
        data = data.merge(et[merge_cols], on='ophys_experiment_id', how='left')
        data = _glm_add_cell_type(data)
    return data


def _prepare_glm_dropout_long(results, drop_passive=True, include_zero_cells=True, threshold=0.005):
    """
    Prepare the long-format GLM ``results`` dataframe for the dropout overview plots:
    drop passive, (optionally) threshold on variance_explained_full, add a
    'cell_type' column from 'cre_line', and compute
    ``explained_variance = -1 * adj_fraction_change_from_full`` (the coding score).
    """
    data = results.copy()
    if drop_passive and 'passive' in data.columns:
        data = data[data.passive == False].copy()  # noqa: E712
        
    if include_zero_cells: 
        threshold = 0
    data = data[data.variance_explained_full > threshold].copy()

    data = _glm_add_cell_type(data)
    data['explained_variance'] = -1 * data['adj_fraction_change_from_full']
    return data


def _format_experience_axis(ax, abbreviate=True):
    """Apply the standard platform experience-level x-axis ticks + label coloring."""
    if abbreviate:
        ax.set_xticks(np.arange(len(utils.get_abbreviated_experience_levels())))
        ax.set_xticklabels(utils.get_abbreviated_experience_levels(), rotation=0)
    else:
        ax.set_xticks(np.arange(len(utils.get_experience_levels())))
        ax.set_xticklabels(utils.get_experience_levels(), rotation=90)
    utils.color_xaxis_labels_by_experience(ax)


def _clean_dropout_title(feature):
    """Human-readable panel title for a dropout/coding-score feature name."""
    title = feature.replace('all-images', 'images')
    title = title.replace('omissions_positive', 'excited')
    title = title.replace('omissions_negative', 'inhibited')
    title = title.replace('_', ' ')
    return title


def _get_feature_title_and_color(feature):
    """
    Return a (capitalized title, color) for a GLM coding-score feature, using the
    same feature palette as plot_coding_scores_across_conditions_grid
    (``plotting.get_feature_colors_and_labels()[0]``). Features outside the standard
    image/omission/behavior/task set fall back to a cleaned, capitalized title in
    black.
    """
    from visual_behavior.dimensionality_reduction.clustering import plotting
    feature_colors = plotting.get_feature_colors_and_labels()[0]
    feature_specs = {
        'all-images': ('Image coding', feature_colors[0]),
        'omissions': ('Omission coding', feature_colors[1]),
        'behavioral': ('Behavior coding', feature_colors[2]),
        'task': ('Task coding', feature_colors[3]),
    }
    if feature in feature_specs:
        return feature_specs[feature]
    return (_clean_dropout_title(feature).capitalize(), 'k')


def _get_matched_cells_with_ve(cells_table, results_pivoted, threshold):
    """
    cell_specimen_ids present in ``cells_table`` whose maximum full-model variance
    explained (across sessions) is >= ``threshold``. Same intent as
    gvt.get_matched_cells_with_ve, but joins on 'cell_specimen_id' (always present)
    rather than 'cell_roi_id' (absent from the curated platform cells table).
    """
    matched_ids = set(np.asarray(cells_table.cell_specimen_id.unique()))
    max_ve = results_pivoted.groupby('cell_specimen_id')['variance_explained_full'].max()
    ve_ok = set(max_ve[max_ve >= threshold].index.values)
    return np.array(sorted(matched_ids & ve_ok))


def plot_coding_score_distribution_by_experience(
        results_pivoted,
        dropouts_to_show=['all-images', 'omissions', 'behavioral', 'task'],
        plot_type='pointplot', include_zero_cells=True,
        experiment_table=None, include_4x2_data=False,
        show_matched=True, matched_cells=None, cells_table=None,
        strict_experience_matching=False, strict_matched_cells=None,
        matched_with_variance_explained=False, matched_ve_threshold=0, ve_matched_cells=None,
        add_combined_panel=True, cell_type_order=None,
        ylabel=None, ylims=None, abbreviate_exp=True, suptitle=None,
        save_dir=None, folder='coding_scores_and_kernels', suffix='', ax=None, group_column='mouse_id'):
    """
    Coding score (dropout) across experience levels for each cell type, with stats
    across experience levels (MLM by default, per the ``USE_MLM`` switch).

    Full platform-style re-implementation of ``gvt.plot_population_averages`` (the
    transposed / default layout): one figure per dropout in ``dropouts_to_show`` with
    a panel per cell type (x = experience level, y = coding-score value, colored by
    experience level) plus an optional 'Combined' panel showing all cell types
    together. Subsets of cells are overlaid as gray/navajowhite point plots, exactly
    as in the GLM original:
      - matched cells (present in all experience levels): light gray overlay,
        controlled by ``show_matched`` (default True, matching gvt).
      - strictly matched cells (last familiar + second novel active): navajowhite,
        via ``strict_experience_matching``.
      - cells with full-model variance explained >= ``matched_ve_threshold``:
        navajowhite, via ``matched_with_variance_explained``; stats are then run on
        this subset (matching gvt) rather than all cells.

    Each overlay's cell_specimen_ids can either be passed in precomputed
    (``matched_cells`` / ``strict_matched_cells`` / ``ve_matched_cells``) or computed
    internally from ``cells_table`` (falls back to loading.get_cell_table() if not
    given). Pass ``cells_table`` to avoid the internal load.

    plot_type: 'pointplot' (default) or 'boxplot'.
    Returns the combined stats dataframe across all dropouts.
    """
    suffix = _norm_suffix(suffix)
    data = _prepare_glm_metrics_table(results_pivoted, experiment_table=experiment_table,
                                      include_4x2_data=include_4x2_data,
                                      include_zero_cells=include_zero_cells)
    colors = utils.get_experience_level_colors()
    order = utils.get_experience_levels()
    if cell_type_order is None:
        # platform convention: Excitatory first, Vip last (gvt used the reverse)
        cell_type_order = utils.get_cell_types()
    cell_type_colors = {ct: c for ct, c in zip(utils.get_cell_types(), utils.get_cell_type_colors())}

    # ---- resolve the cell_specimen_id sets used for the gray/navajowhite overlays ----
    _need_cells_table = ((show_matched and matched_cells is None)
                         or (strict_experience_matching and strict_matched_cells is None)
                         or (matched_with_variance_explained and ve_matched_cells is None))
    base_cells_table = None
    if _need_cells_table:
        base_cells_table = cells_table
        if base_cells_table is None:
            base_cells_table = loading.get_cell_table(platform_paper_only=True,
                                                      include_4x2_data=include_4x2_data)
        if 'passive' in base_cells_table.columns:
            base_cells_table = base_cells_table[base_cells_table.passive == False].copy()  # noqa: E712
    if show_matched and matched_cells is None:
        matched_cells = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(
            base_cells_table).cell_specimen_id.unique()
    if strict_experience_matching and strict_matched_cells is None:
        strict_tbl = utilities.limit_to_last_familiar_second_novel_active(base_cells_table)
        strict_tbl = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(strict_tbl)
        strict_matched_cells = strict_tbl.cell_specimen_id.unique()
    if matched_with_variance_explained and ve_matched_cells is None:
        matched_tbl = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(base_cells_table)
        ve_matched_cells = _get_matched_cells_with_ve(matched_tbl, results_pivoted, matched_ve_threshold)

    matched_cells = None if matched_cells is None else np.asarray(matched_cells)
    strict_matched_cells = None if strict_matched_cells is None else np.asarray(strict_matched_cells)
    ve_matched_cells = None if ve_matched_cells is None else np.asarray(ve_matched_cells)

    # filename disambiguation so S2 / S3A / S3B outputs don't collide (matches gvt)
    file_suffix = suffix
    if strict_experience_matching:
        file_suffix = file_suffix + '_strict_matched'
    if matched_with_variance_explained:
        file_suffix = file_suffix + '_matched_with_ve_' + str(matched_ve_threshold)

    n_panels = len(cell_type_order) + (1 if add_combined_panel else 0)
    if ax is not None and len(dropouts_to_show) != 1:
        raise ValueError('pass a single dropout in dropouts_to_show when providing ax')
    combined_stats = pd.DataFrame()
    for feature in dropouts_to_show:
        # coding scores are stored as negative fractions; show as positive magnitude
        data[feature] = data[feature].abs()
        figsize = (2.6 * n_panels, 2.5)
        if ax is None:
            fig, axx = plt.subplots(1, n_panels, figsize=figsize, sharex=False, sharey=True)
            save_fig = True
        else:
            axx = ax
            fig = (axx[0] if hasattr(axx, '__len__') else axx).get_figure()
            figsize = fig.get_size_inches()
            save_fig = False
        if not hasattr(axx, '__len__'):
            axx = [axx]
        stats_data_per_panel = []
        for i, cell_type in enumerate(cell_type_order):
            ct_data = data[data.cell_type == cell_type]
            # all cells, colored by experience level
            if plot_type == 'boxplot':
                axx[i] = sns.boxplot(data=ct_data, x='experience_level', y=feature, order=order,
                                    hue='experience_level', hue_order=order, notch=True, width=0.4,
                                    palette=colors, fliersize=0, ax=axx[i],
                                    whis=1.5)
                for box in axx[i].collections:
                    box.set_alpha(0.75)
            else:
                axx[i] = sns.pointplot(data=ct_data, x='experience_level', y=feature, order=order,
                                      hue='experience_level', hue_order=order, palette=colors,
                                      estimator=np.mean, markers='.', markersize=5,
                                      err_kws={'linewidth': 2}, ax=axx[i],
                                      errorbar=("ci", 95),
                                      n_boot=1000)
            for child in list(axx[i].get_children()):
                child.set_zorder(1000)
            # subset overlays (light gray = matched, navajowhite = strict / VE-matched)
            if matched_cells is not None:
                md = ct_data[ct_data.cell_specimen_id.isin(matched_cells)]
                axx[i] = sns.pointplot(data=md, x='experience_level', y=feature, order=order,
                                      color='lightgray', linestyle='-', ax=axx[i],
                                      estimator="mean",
                                      errorbar=("ci", 95),
                                      n_boot=1000)
            if strict_matched_cells is not None:
                sd = ct_data[ct_data.cell_specimen_id.isin(strict_matched_cells)]
                axx[i] = sns.pointplot(data=sd, x='experience_level', y=feature, order=order,
                                      color='navajowhite', linestyle='-', ax=axx[i],
                                      estimator="mean",
                                      errorbar=("ci", 95),
                                      n_boot=1000)
            if ve_matched_cells is not None:
                vd = ct_data[ct_data.cell_specimen_id.isin(ve_matched_cells)]
                axx[i] = sns.pointplot(data=vd, x='experience_level', y=feature, order=order,
                                      color='navajowhite', linestyle='-', ax=axx[i],
                                      estimator="mean",
                                      errorbar=("ci", 95),
                                      n_boot=1000)
            _legend = axx[i].get_legend()
            if _legend:
                _legend.remove()
            axx[i].set_xlim(-0.5, len(order) - 0.5)
            # collect the data used for stats (VE subset when requested, else all cells);
            # stats are added in a second pass below, after the shared y-axis is set
            if matched_with_variance_explained and ve_matched_cells is not None:
                stats_data_per_panel.append(ct_data[ct_data.cell_specimen_id.isin(ve_matched_cells)])
            else:
                stats_data_per_panel.append(ct_data)
            axx[i].set_xlabel('')
            axx[i].set_ylabel('')
            axx[i].set_title(cell_type, fontsize=14)
            _format_experience_axis(axx[i], abbreviate=abbreviate_exp)
        # combined panel: all cell types overlaid, colored by cell type
        if add_combined_panel:
            j = len(cell_type_order)
            axx[j] = sns.pointplot(data=data, x='experience_level', y=feature, order=order,
                                  hue='cell_type', hue_order=cell_type_order, palette=cell_type_colors,
                                  estimator=np.mean, markers='.', markersize=5,
                                  err_kws={'linewidth': 2}, ax=axx[j],
                                  errorbar=("ci", 95),
                                  n_boot=1000)
            _legend = axx[j].get_legend()
            if _legend:
                _legend.remove()
            axx[j].set_title('Combined', fontsize=14)
            axx[j].set_xlabel('')
            axx[j].set_ylabel('')
            axx[j].set_xlim(-0.5, len(order) - 0.5)
            _format_experience_axis(axx[j], abbreviate=abbreviate_exp)
        # shared y across all panels: floor at 0 (or use ylims), then add the MLM
        # significance bars just above the shared data top
        if ylims is not None:
            axx[0].set_ylim(ylims)
        else:
            axx[0].set_ylim(bottom=0)
        ymax_shared = axx[0].get_ylim()[1]
        needed_tops = []
        for i, cell_type in enumerate(cell_type_order):
            axx[i], panel_stats = add_stats_to_plot(stats_data_per_panel[i], feature, axx[i],
                                                   ymax=ymax_shared, compact_bars=True,
                                                   group_column=group_column,
                                                   event_type=feature, cell_type=cell_type)
            needed_tops.append(axx[i].get_ylim()[1])
            panel_stats = insert_stats_metadata(panel_stats, condition='experience_level')
            combined_stats = pd.concat([combined_stats, panel_stats])
        # Each per-panel add_stats() call resets the (shared) y-limit to fit only its
        # own bars, so the last/shortest panel would clip earlier panels' taller
        # significance bars (drawn with clip_on=False -> they float off the axis).
        # Re-apply the tallest required top to the shared axis so every bar is visible.
        axx[0].set_ylim(top=max(needed_tops))
        # capitalized y-label; capitalized, feature-colored suptitle
        axx[0].set_ylabel(ylabel if ylabel else (_clean_dropout_title(feature).capitalize() + '\ncoding score'))
        feat_title, feat_color = _get_feature_title_and_color(feature)
        # match the suptitle font size to the y-axis label, and sit it just above the panels
        ylabel_fontsize = axx[0].yaxis.label.get_fontsize()
        # suptitle and subplots_adjust act on the whole figure; only apply them when this
        # function owns the figure (standalone). When embedding (ax passed in, save_fig False)
        # skip them so they don't reflow a larger composite figure.
        if save_fig:
            fig.suptitle(suptitle if suptitle else feat_title, x=0.5, y=1.04,
                         fontsize=ylabel_fontsize, color=feat_color)
            fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.82)
        if save_dir:
            base = 'coding_score_' + feature + '_distribution' + file_suffix
            stats_base = 'coding_score_' + feature + file_suffix
            if save_fig:
                utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(base))
            try:
                stats_suffix = _stats_suffix_for_table(combined_stats)
                combined_stats.to_csv(os.path.join(save_dir, folder,
                                                   _clean_filename(stats_base + stats_suffix)))
                stats = get_descriptive_stats_for_metric(data, feature, ['cell_type', 'experience_level'])
                stats.to_csv(os.path.join(save_dir, folder,
                                          _clean_filename(stats_base + '_values.csv')))
            except BaseException:
                print('STATS DID NOT SAVE FOR', feature)
    return axx


def plot_coding_score_distribution_legend(save_dir=None, folder='coding_scores_and_kernels',
                                          include_overlays=True, suffix=''):
    """
    Standalone legend figure to accompany ``plot_coding_score_distribution_by_experience``.

    Those plots strip their own legends to stay compact and use two different color
    schemes -- experience-level colors in the per-cell-type panels and cell-type colors
    in the 'Combined' panel -- so this produces a single figure with both legends side
    by side (plus the gray/navajowhite matched-cell overlay entries when
    ``include_overlays`` is True). Returns (fig, ax).
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    exp_levels = utils.get_experience_levels()
    exp_colors = utils.get_experience_level_colors()
    cell_types = utils.get_cell_types()
    cell_type_colors = utils.get_cell_type_colors()

    exp_handles = [mpatches.Patch(color=c, label=l) for l, c in zip(exp_levels, exp_colors)]
    if include_overlays:
        exp_handles += [
            mlines.Line2D([], [], color='lightgray', marker='o', linestyle='-', label='Matched cells'),
            mlines.Line2D([], [], color='navajowhite', marker='o', linestyle='-',
                          label='Strict / VE-matched cells'),
        ]
    ct_handles = [mpatches.Patch(color=c, label=l) for l, c in zip(cell_types, cell_type_colors)]

    figsize = (6, 2.5)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    for a in ax:
        a.axis('off')
    ax[0].legend(handles=exp_handles, title='Per-cell-type panels', loc='center', frameon=False)
    ax[1].legend(handles=ct_handles, title='Combined panel', loc='center', frameon=False)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder,
                          _clean_filename('coding_score_distribution_legend' + _norm_suffix(suffix)))
    return fig, ax


def _plot_coding_score_by_hue(results_pivoted, hue, hue_order, palette,
                              dropouts_to_show, include_zero_cells, equipment,
                              experiment_table, include_4x2_data, ylabel,
                              save_dir, folder, suffix, group_column, filename_stub,
                              ax=None, cell_type=None):
    """
    Shared implementation for plot_coding_score_by_area / _by_depth: one figure per
    cell type, a panel per dropout, x = experience level, split by ``hue``
    (targeted_structure or coarse depth). Stats compare hue values within each
    experience level via ``add_stats_to_plot_for_hues`` (preserves the gvt
    area-vs-area / depth-vs-depth comparison, but through the platform MLM code).
    Returns the combined stats dataframe.
    """
    suffix = _norm_suffix(suffix)
    data = _prepare_glm_metrics_table(results_pivoted, experiment_table=experiment_table,
                                      include_4x2_data=include_4x2_data,
                                      include_zero_cells=include_zero_cells)
    # equipment filter (mesoscope == 'MESO.1'), matching the gvt originals
    if equipment is not None and 'equipment_name' in data.columns:
        if equipment == 'mesoscope':
            data = data[data.equipment_name == 'MESO.1'].copy()
        else:
            data = data[data.equipment_name != 'MESO.1'].copy()
    order = utils.get_experience_levels()
    cell_types = utils.get_cell_types() if cell_type is None else [cell_type]
    if ax is not None and len(cell_types) != 1:
        raise ValueError('provide a single cell_type when passing ax')

    combined_stats = pd.DataFrame()
    for cell_type_i in cell_types:
        ct_all = data[data.cell_type == cell_type_i]
        figsize = (2.5 * len(dropouts_to_show), 2.5)
        if ax is None:
            fig, axx = plt.subplots(1, len(dropouts_to_show), figsize=figsize, sharey=False)
            save_fig = True
        else:
            axx = ax
            fig = (axx[0] if hasattr(axx, '__len__') else axx).get_figure()
            figsize = fig.get_size_inches()
            save_fig = False
        if not hasattr(axx, '__len__'):
            axx = [axx]
        for index, feature in enumerate(dropouts_to_show):
            ct_data = ct_all.copy()
            ct_data[feature] = ct_data[feature].abs()
            axx[index] = sns.pointplot(data=ct_data, x='experience_level', y=feature, order=order,
                                      hue=hue, hue_order=hue_order, palette=palette,
                                      dodge=0.1 * len(hue_order), linestyle='none',
                                      markers='.', markersize=5, err_kws={'linewidth': 2}, ax=axx[index],
                                      estimator="mean",
                                      errorbar=("ci", 95),
                                      n_boot=1000)
            _legend = axx[index].get_legend()
            if index != len(dropouts_to_show) - 1:
                if _legend:
                    _legend.remove()
            else:
                if _legend:
                    axx[index].legend(title='', fontsize='xx-small', bbox_to_anchor=(1.05, 1))
            feat_title, feat_color = _get_feature_title_and_color(feature)
            axx[index].set_title(feat_title, fontsize=16, color=feat_color)
            axx[index].set_ylim(bottom=0)
            axx[index].set_xlim(-0.5, len(order) - 0.5)
            axx[index], panel_stats = add_stats_to_plot_for_hues(ct_data, feature, axx[index],
                                                                xorder=order, x='experience_level',
                                                                hue=hue, compact_bars=True,
                                                                group_column=group_column,
                                                                event_type=feature, cell_type=cell_type_i)
            panel_stats = insert_stats_metadata(panel_stats, condition=hue)
            combined_stats = pd.concat([combined_stats, panel_stats])
            axx[index].set_xlabel('')
            axx[index].set_ylabel('')
            axx[index].tick_params(axis='y', which='major', labelsize=12)
            _format_experience_axis(axx[index], abbreviate=True)
        axx[0].set_ylabel(ylabel)
        # raise the cell-type suptitle so it clears the panel titles
        fig.suptitle(cell_type_i, fontsize=18, x=0.5, y=1.12)
        fig.subplots_adjust(wspace=0.5, hspace=0.3)
        if save_dir:
            base = filename_stub + '_' + cell_type_i[0:3] + suffix
            if save_fig:
                utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(base))
            try:
                stats_suffix = _stats_suffix_for_table(combined_stats)
                combined_stats.to_csv(os.path.join(save_dir, folder,
                                                   _clean_filename(filename_stub + suffix + stats_suffix)))
            except BaseException:
                print('STATS DID NOT SAVE FOR', filename_stub, cell_type_i)
    return axx


def plot_coding_score_by_area(results_pivoted,
                              dropouts_to_show=['all-images', 'omissions', 'behavioral', 'task'],
                              include_zero_cells=True, equipment='mesoscope', areas=None,
                              experiment_table=None, include_4x2_data=False, ylabel='Coding score',
                              save_dir=None, folder='coding_scores_and_kernels', suffix='',
                              ax=None, cell_type=None, group_column='mouse_id'):
    """
    Coding score across experience levels, split by visual area (targeted_structure),
    one figure per cell type. Platform-style re-implementation of
    ``gvt.plot_population_averages_by_area``. Stats compare areas within each
    experience level (``add_stats_to_plot_for_hues``). Returns combined stats.
    """
    if areas is None:
        areas = ['VISp', 'VISl', 'VISam', 'VISal'] if include_4x2_data else ['VISp', 'VISl']
    palette = {'VISp': 'black', 'VISl': 'gray', 'VISam': 'blue', 'VISal': 'red'}
    palette = {a: palette.get(a, 'gray') for a in areas}
    return _plot_coding_score_by_hue(
        results_pivoted, hue='targeted_structure', hue_order=areas, palette=palette,
        dropouts_to_show=dropouts_to_show, include_zero_cells=include_zero_cells,
        equipment=equipment, experiment_table=experiment_table, include_4x2_data=include_4x2_data,
        ylabel=ylabel, save_dir=save_dir, folder=folder, suffix=suffix,
        group_column=group_column, filename_stub='coding_score_by_area', ax=ax, cell_type=cell_type)


def plot_coding_score_by_depth(results_pivoted,
                               dropouts_to_show=['all-images', 'omissions', 'behavioral', 'task'],
                               include_zero_cells=True, equipment='mesoscope', area=['VISp', 'VISl'],
                               experiment_table=None, include_4x2_data=False, ylabel='Coding score',
                               save_dir=None, folder='coding_scores_and_kernels', suffix='',
                               ax=None, cell_type=None, group_column='mouse_id'):
    """
    Coding score across experience levels, split by coarse cortical depth
    (upper <250 um / lower >=250 um), one figure per cell type. Platform-style
    re-implementation of ``gvt.plot_population_averages_by_depth``. Stats compare
    depths within each experience level (``add_stats_to_plot_for_hues``).
    Returns combined stats.
    """
    suffix = _norm_suffix(suffix)
    data = _prepare_glm_metrics_table(results_pivoted, experiment_table=experiment_table,
                                      include_4x2_data=include_4x2_data,
                                      include_zero_cells=include_zero_cells)
    # equipment filter (mesoscope == 'MESO.1'), matching the gvt original
    if equipment is not None and 'equipment_name' in data.columns:
        if equipment == 'mesoscope':
            data = data[data.equipment_name == 'MESO.1'].copy()
        else:
            data = data[data.equipment_name != 'MESO.1'].copy()
    if area is not None and 'targeted_structure' in data.columns:
        data = data[data.targeted_structure.isin(area)].copy()
    if 'imaging_depth' in data.columns:
        data['coarse_binned_depth'] = [_coarse_bin_depth(d) for d in data['imaging_depth']]
    palette = {'upper': 'black', 'lower': 'gray'}
    # data already filtered/prepped above; pass through the shared plotter without
    # re-filtering equipment or re-merging (use equipment=None, drop already done)
    return _plot_coding_score_by_hue(
        data, hue='coarse_binned_depth', hue_order=['upper', 'lower'], palette=palette,
        dropouts_to_show=dropouts_to_show, include_zero_cells=True, equipment=None,
        experiment_table=experiment_table, include_4x2_data=include_4x2_data, ylabel=ylabel,
        save_dir=save_dir, folder=folder, suffix=suffix, group_column=group_column,
        filename_stub='coding_score_by_depth', ax=ax, cell_type=cell_type)


def plot_variance_explained_by_experience(results_pivoted, plot_type='boxplot', include_zero_cells=True,
                                          experiment_table=None, include_4x2_data=False,
                                          ylabel='Variance explained (%)', ylims=None, abbreviate_exp=True,
                                          suptitle=None, save_dir=None, folder='coding_scores_and_kernels',
                                          suffix='', ax=None, group_column='mouse_id'):
    """
    Full-model variance explained (%) across experience levels for each cell type,
    with stats across experience levels. Platform-style re-implementation of
    ``gvt.var_explained_by_experience``. Returns combined stats.
    """
    suffix = _norm_suffix(suffix)
    data = _prepare_glm_metrics_table(results_pivoted, experiment_table=experiment_table,
                                      include_4x2_data=include_4x2_data,
                                      include_zero_cells=include_zero_cells)
    data['variance_explained_percent'] = data['variance_explained_full'] * 100
    metric = 'variance_explained_percent'
    colors = utils.get_experience_level_colors()
    order = utils.get_experience_levels()
    cell_types = utils.get_cell_types()

    figsize = (8, 2.5)
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=True)
        save_fig = True
    else:
        fig = ax[0].get_figure()
        figsize = fig.get_size_inches()
        save_fig = False
    combined_stats = pd.DataFrame()
    panel_data = []
    for i, cell_type in enumerate(cell_types):
        ct_data = data[data.cell_type == cell_type]
        panel_data.append(ct_data)
        if plot_type == 'pointplot':
            ax[i] = sns.pointplot(data=ct_data, x='experience_level', y=metric, order=order,
                                  hue='experience_level', hue_order=order, palette=colors,
                                  markers='.', markersize=5, err_kws={'linewidth': 2}, ax=ax[i],
                                  estimator="mean",
                                  errorbar=("ci", 95),
                                  n_boot=1000)
        else:
            # showfliers=False (not fliersize=0): fliers must be removed, not just hidden,
            # otherwise they still inflate the autoscaled y-limit and push the stats bars
            # far above the visible boxes
            ax[i] = sns.boxplot(data=ct_data, x='experience_level', y=metric, order=order,
                                hue='experience_level', hue_order=order, palette=colors,
                                showfliers=False, linewidth=1, ax=ax[i],
                                whis=1.5)
        _legend = ax[i].get_legend()
        if _legend:
            _legend.remove()
        ax[i].set_xlim(-0.5, len(order) - 0.5)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].set_title(cell_type, fontsize=14)
        _format_experience_axis(ax[i], abbreviate=abbreviate_exp)
    # shared y across panels: floor at 0 (or use ylims), then place the significance
    # bars just above the shared data top instead of at a fixed ceiling
    if ylims is not None:
        ax[0].set_ylim(ylims)
    else:
        ax[0].set_ylim(bottom=0)
    ymax_shared = ax[0].get_ylim()[1]
    needed_tops = []
    for i, cell_type in enumerate(cell_types):
        ax[i], panel_stats = add_stats_to_plot(panel_data[i], metric, ax[i], ymax=ymax_shared,
                                               compact_bars=True, group_column=group_column,
                                               event_type='variance_explained', cell_type=cell_type)
        needed_tops.append(ax[i].get_ylim()[1])
        panel_stats = insert_stats_metadata(panel_stats, condition='experience_level')
        combined_stats = pd.concat([combined_stats, panel_stats])
    # re-apply the tallest required top so the shared axis isn't clipped to the last
    # (shortest) panel, which would push earlier panels' clip_on=False bars off-axis
    ax[0].set_ylim(top=max(needed_tops))
    ax[0].set_ylabel(ylabel)
    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=1.02, fontsize=18)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if save_dir:
        base = 'variance_explained_by_experience' + suffix
        if save_fig:
            utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(base))
        try:
            stats_suffix = _stats_suffix_for_table(combined_stats)
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(base + stats_suffix)))
            stats = get_descriptive_stats_for_metric(data, metric, ['cell_type', 'experience_level'])
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(base + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR variance_explained_by_experience')
    return ax


def plot_variance_explained_for_matched_cells(results_pivoted, include_4x2_data=False,
                                              experiment_table=None, cells_table=None,
                                              ylabel='Variance explained (%)', ylims=None, abbreviate_exp=True,
                                              figsize=(12, 3.5), suptitle=None, save_dir=None,
                                              folder='coding_scores_and_kernels',
                                              suffix='', ax=None, group_column='mouse_id'):
    """
    Full-model variance explained (%) across experience levels for matched vs
    non-matched cells, one panel per cell type, with stats comparing matched vs
    non-matched within each experience level. Platform-style re-implementation of
    ``gvt.var_explained_matched``. Returns combined stats.
    """
    suffix = _norm_suffix(suffix)
    data = _prepare_glm_metrics_table(results_pivoted, experiment_table=experiment_table,
                                      include_4x2_data=include_4x2_data, include_zero_cells=True)
    data['variance_explained_percent'] = data['variance_explained_full'] * 100
    metric = 'variance_explained_percent'

    # identify cells matched across all experience levels
    if cells_table is None:
        cells_table = loading.get_cell_table(platform_paper_only=True, include_4x2_data=include_4x2_data)
    cells_table = cells_table[cells_table.passive == False].copy()  # noqa: E712
    cells_table = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table)
    matched_cells = cells_table.cell_specimen_id.unique()
    data['matched'] = ['Matched' if csid in np.array(matched_cells) else 'Non-matched'
                       for csid in data['cell_specimen_id']]

    order = utils.get_experience_levels()
    hue_order = ['Matched', 'Non-matched']
    cell_types = utils.get_cell_types()
    cell_type_colors = {ct: c for ct, c in zip(utils.get_cell_types(), utils.get_cell_type_colors())}

    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=True)
        save_fig = True
    else:
        fig = ax[0].get_figure()
        figsize = fig.get_size_inches()
        save_fig = False
    combined_stats = pd.DataFrame()
    panel_data = []
    # First pass: draw all boxplots. With sharey, the shared autoscale only fits every
    # panel's data if no set_ylim runs mid-loop -- so stats are added in a second pass.
    for i, cell_type in enumerate(cell_types):
        ct_data = data[data.cell_type == cell_type]
        panel_data.append(ct_data)
        palette = [cell_type_colors.get(cell_type, 'k'), 'gray']
        ax[i] = sns.boxplot(data=ct_data, x='experience_level', y=metric, order=order,
                            hue='matched', hue_order=hue_order, palette=palette,
                            showfliers=False, linewidth=1, ax=ax[i],
                            whis=1.5)
        if i == len(cell_types) - 1:
            ax[i].legend(title='', fontsize='xx-small', loc='upper right')
        else:
            _legend = ax[i].get_legend()
            if _legend:
                _legend.remove()
        ax[i].set_xlim(-0.5, len(order) - 0.5)
        ax[i].set_title(cell_type, fontsize=14)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        _format_experience_axis(ax[i], abbreviate=abbreviate_exp)
    # all boxplots drawn -> shared autoscale fits every panel; floor at 0, then add stats
    if ylims is not None:
        ax[0].set_ylim(ylims)
    else:
        ax[0].set_ylim(bottom=0)
    ymax_shared = ax[0].get_ylim()[1]
    needed_tops = []
    for i, cell_type in enumerate(cell_types):
        ax[i], panel_stats = add_stats_to_plot_for_hues(panel_data[i], metric, ax[i], ymax=ymax_shared,
                                                        xorder=order,
                                                        x='experience_level', hue='matched',
                                                        compact_bars=True, group_column=group_column,
                                                        event_type='variance_explained', cell_type=cell_type)
        panel_stats = insert_stats_metadata(panel_stats, condition='matched')
        combined_stats = pd.concat([combined_stats, panel_stats])
        needed_tops.append(ax[i].get_ylim()[1])
    ax[0].set_ylim(top=max(needed_tops))
    ax[0].set_ylabel(ylabel)
    if suptitle:
        plt.suptitle(suptitle, x=0.52, y=1.02, fontsize=18)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    if save_dir:
        base = 'variance_explained_matched' + suffix
        if save_fig:
            utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(base))
        try:
            stats_suffix = _stats_suffix_for_table(combined_stats)
            combined_stats.to_csv(os.path.join(save_dir, folder, _clean_filename(base + stats_suffix)))
            stats = get_descriptive_stats_for_metric(data, metric, ['cell_type', 'experience_level', 'matched'])
            stats.to_csv(os.path.join(save_dir, folder, _clean_filename(base + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR variance_explained_matched')
    return ax


def plot_dropout_summary_population(results, dropouts_to_show=['all-images', 'omissions', 'behavioral', 'task'],
                                    plot_type='boxplot', include_zero_cells=True, exclusion_threshold=0.005,
                                    ylabel='Coding Score', xlabel='Withheld component', suptitle=None,
                                    save_dir=None, folder='coding_scores_and_kernels', suffix='',
                                    ax=None, group_column='mouse_id'):
    """
    Population dropout (coding score) summary by cell type for a small set of
    withheld components, aggregated across all cells (no experience-level split).
    Platform-style re-implementation of ``gvt.plot_dropout_summary_population``.

    Descriptive stats only (grouped by cell_type x dropout) -- this is an overview
    plot, matching the original which returns descriptive statistics. Returns the
    descriptive stats dataframe.
    """
    suffix = _norm_suffix(suffix)
    data = _prepare_glm_dropout_long(results, include_zero_cells=include_zero_cells,
                                     threshold=exclusion_threshold)
    # The GLM run may store full omissions/hits/misses dropouts as 'all-*' whenever a
    # 'post-*' split exists; remap the requested names so the columns aren't empty
    # (matches gvt.plot_dropout_summary_population).
    present = set(data.dropout.unique())
    for base_name in ['omissions', 'hits', 'misses', 'passive_change']:
        if base_name in dropouts_to_show:
            # use the full 'all-<x>' dropout when the within-session name is absent, or
            # when a 'post-<x>' split exists (matches gvt) -- as long as 'all-<x>' is there
            need_all = (base_name not in present) or (('post-' + base_name) in present)
            if need_all and (('all-' + base_name) in present):
                dropouts_to_show = ['all-' + base_name if d == base_name else d
                                    for d in dropouts_to_show]
    data = data[data.dropout.isin(dropouts_to_show)].copy()
    cell_type_order = utils.get_cell_types()
    # dict palette keyed by cell type so colors map by hue value (not data order)
    cell_type_colors = {ct: c for ct, c in zip(cell_type_order, utils.get_cell_type_colors())}

    if ax is None:
        figsize = (8, 4)
        fig, ax = plt.subplots(figsize=figsize)
        save_fig = True
    else:
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
        save_fig = False
    if plot_type == 'violinplot':
        ax = sns.violinplot(data=data, x='dropout', y='explained_variance', hue='cell_type',
                            order=dropouts_to_show, hue_order=cell_type_order, palette=cell_type_colors,
                            dodge=True, inner='quartile', cut=0, linewidth=1, ax=ax,
                            density_norm="area",
                            bw_method="scott")
    else:
        ax = sns.boxplot(data=data, x='dropout', y='explained_variance', hue='cell_type',
                         order=dropouts_to_show, hue_order=cell_type_order, palette=cell_type_colors,
                         dodge=True, fliersize=0, width=0.7, ax=ax,
                         whis=1.5)
    ax.set_ylim(0, 1)
    ax.legend(title='', fontsize='xx-small', loc='upper right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(np.arange(len(dropouts_to_show)))
    ax.set_xticklabels([_clean_dropout_title(d) for d in dropouts_to_show], rotation=0)
    if suptitle:
        plt.suptitle(suptitle, fontsize=18)
    descriptive = (get_descriptive_stats_for_metric(data, 'explained_variance', ['cell_type', 'dropout'])
                   if len(data) else pd.DataFrame())
    if save_dir:
        base = 'dropout_summary' + ('' if plot_type == 'violinplot' else '_boxplot') + suffix
        if save_fig:
            utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(base))
        descriptive.to_csv(os.path.join(save_dir, folder, _clean_filename(base + '_values.csv')))
    return ax


def plot_dropout_individual_population(results, run_params=None,
                                       dropouts_to_show=None, plot_type='boxplot',
                                       include_zero_cells=True, exclusion_threshold=0.005,
                                       use_single=False, ylabel='Coding Score', suptitle=None,
                                       figsize=None, save_dir=None, folder='coding_scores_and_kernels',
                                       suffix='', ax=None, group_column='mouse_id'):
    """
    Population dropout (coding score) by cell type for the full set of individual
    withheld components, aggregated across all cells. Platform-style
    re-implementation of ``gvt.plot_dropout_individual_population``.

    Descriptive stats only (grouped by cell_type x dropout). ``use_single`` plots the
    single-component ('single-<name>') dropouts instead. Returns the descriptive
    stats dataframe.
    """
    suffix = _norm_suffix(suffix)
    if suptitle is None:
        suptitle = 'Single feature model fits' if use_single else 'Dropout scores'
    if dropouts_to_show is None:
        dropouts_to_show = ['all-images', 'image0', 'image1', 'image2', 'image3', 'image4',
                            'image5', 'image6', 'image7', '', 'omissions',
                            'behavioral', 'licks', 'pupil', 'running', 
                            'task', 'hits', 'misses', ]
    if include_zero_cells == True: 
        exclusion_threshold = 0
    data = _prepare_glm_dropout_long(results, include_zero_cells=include_zero_cells,
                                     threshold=exclusion_threshold)
    # drop all-/post- variants when the post-* split isn't present (matches gvt)
    # present = set(data.dropout.unique())
    # for base_name in ['omissions', 'hits', 'misses', 'passive_change']:
    #     if ('post-' + base_name) not in present:
    #         dropouts_to_show = [d for d in dropouts_to_show if d not in ['all-' + base_name, 'post-' + base_name]]
    # keep blank separators and any dropouts actually present in the data
    # dropouts_to_show = [d for d in dropouts_to_show if (d == '') or (d in present)]
    if use_single:
        dropouts_to_show = ['single-' + d for d in dropouts_to_show]
        # dropouts_to_show = [d for d in dropouts_to_show if (d == '') or (d in present)]
    data = data[data.dropout.isin(dropouts_to_show)].copy()
    
    cell_type_order = utils.get_cell_types()
    # dict palette keyed by cell type so colors map by hue value (not data order)
    cell_type_colors = {ct: c for ct, c in zip(cell_type_order, utils.get_cell_type_colors())}

    if ax is None:
        if figsize is None:
            # keep both individual-dropout panels compact (override via figsize=)
            figsize = (16, 4.5)
        fig, ax = plt.subplots(figsize=figsize)
        save_fig = True
    else:
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
        save_fig = False
    if plot_type == 'violinplot':
        ax = sns.violinplot(data=data, x='dropout', y='explained_variance', hue='cell_type',
                            order=dropouts_to_show, hue_order=cell_type_order, palette=cell_type_colors,
                            dodge=True, inner='quartile', cut=0, linewidth=0, ax=ax,
                            density_norm="area",
                            bw_method="scott")
        ax.axhline(0, color='k', alpha=0.25)
    else:
        ax = sns.boxplot(data=data, x='dropout', y='explained_variance', hue='cell_type',
                         order=dropouts_to_show, hue_order=cell_type_order, palette=cell_type_colors,
                         dodge=True, fliersize=0, ax=ax,
                         whis=1.5)
    ax.set_ylim(0, 1)
    ax.legend(title='', fontsize='xx-small', loc='upper right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Only component included' if use_single else 'Withheld component')
    if use_single:
        ax.set_xticks(np.arange(len(dropouts_to_show)))
        ax.set_xticklabels([d.replace('single-', '') for d in dropouts_to_show], rotation=90)
    else:
        ax.tick_params(axis='x', rotation=90)
    if suptitle:
        fig.suptitle(suptitle, fontsize=18)
    descriptive = (get_descriptive_stats_for_metric(data, 'explained_variance', ['cell_type', 'dropout'])
                   if len(data) else pd.DataFrame())
    if save_dir:
        single_str = '_single' if use_single else ''
        base = 'dropout_individual' + ('' if plot_type == 'violinplot' else '_boxplot') + single_str + suffix
        if save_fig:
            utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(base))
        descriptive.to_csv(os.path.join(save_dir, folder, _clean_filename(base + '_values.csv')))
    return ax


def plot_fraction_cells_coding_by_experience(
        results_pivoted, features=['all-images', 'omissions', 'behavioral', 'task'],
        coding_threshold=0.1, variance_explained_threshold=None,
        experiment_table=None, include_4x2_data=False,
        ylabel='Fraction of cells\ncoding for', abbreviate_exp=True, suptitle=None,
        save_dir=None, folder='coding_scores_and_kernels', suffix='', ax=None, group_column='mouse_id'):
    """
    Fraction of cells coding for each GLM feature across experience levels, one line
    per cell type with binomial 95% confidence-interval error bars; one panel per
    feature. Documented, platform-style version of the inline "feature coding by
    experience level" plot that previously lived in figure_4_supplemental.ipynb
    (analogous to gvt.plot_fraction_summary_population).

    A cell counts as "coding" for a feature when its full model is above
    ``variance_explained_threshold`` (when provided) AND its (absolute) coding score
    for that feature exceeds ``coding_threshold``. Feature panel titles are
    capitalized and colored by feature; experience-level x labels use the standard
    abbreviated, colored tick labels.

    Returns the per (cell_type, experience_level) summary dataframe (fractions, n, CIs).
    """
    suffix = _norm_suffix(suffix)
    data = _prepare_glm_metrics_table(results_pivoted, experiment_table=experiment_table,
                                      include_4x2_data=include_4x2_data, include_zero_cells=True)
    # "codes anything" gate (optional full-model variance-explained threshold)
    if variance_explained_threshold is not None:
        data['code_anything'] = data['variance_explained_full'] > variance_explained_threshold
    else:
        data['code_anything'] = True
    code_cols = []
    for feature in features:
        col = 'code_' + feature
        data[col] = data['code_anything'] & (data[feature].abs() > coding_threshold)
        code_cols.append(col)

    grp = data.groupby(['cell_type', 'experience_level'])
    summary = grp[code_cols].mean()
    summary['n'] = grp[code_cols].count()[code_cols[0]]
    for col in code_cols:
        summary[col + '_ci'] = 1.96 * np.sqrt((summary[col] * (1 - summary[col])) / summary['n'])

    order = utils.get_experience_levels()
    cell_types = utils.get_cell_types()
    cell_type_colors = {ct: c for ct, c in zip(cell_types, utils.get_cell_type_colors())}

    figsize = (2.7 * len(features), 3)
    if ax is None:
        fig, ax = plt.subplots(1, len(features), figsize=figsize, sharey=True)
        save_fig = True
    else:
        fig = (ax[0] if hasattr(ax, '__len__') else ax).get_figure()
        figsize = fig.get_size_inches()
        save_fig = False
    if not hasattr(ax, '__len__'):
        ax = [ax]
    x = np.arange(len(order))
    for index, feature in enumerate(features):
        col = 'code_' + feature
        for cell_type in cell_types:
            y = [summary.loc[(cell_type, e), col] if (cell_type, e) in summary.index else np.nan
                 for e in order]
            yerr = [summary.loc[(cell_type, e), col + '_ci'] if (cell_type, e) in summary.index else np.nan
                    for e in order]
            ax[index].errorbar(x, y, yerr=yerr, color=cell_type_colors[cell_type], linewidth=3,
                               label=cell_type)
        feat_title, feat_color = _get_feature_title_and_color(feature)
        ax[index].set_title(feat_title, fontsize=16, color=feat_color)
        ax[index].set_xlabel('')
        ax[index].set_ylabel('')
        ax[index].set_xlim(-0.5, len(order) - 0.5)
        ax[index].set_ylim(bottom=0)
        ax[index].set_xticks(x)
        _format_experience_axis(ax[index], abbreviate=abbreviate_exp)
        if index == len(features) - 1:
            ax[index].legend(title='', fontsize='xx-small', bbox_to_anchor=(1.05, 1))
    ax[0].set_ylabel(ylabel)
    # suptitle and subplots_adjust act on the whole figure; only apply them when this function
    # owns the figure (standalone). When embedding (ax passed in, save_fig False) skip them so
    # they don't reflow a larger composite figure.
    if save_fig:
        if suptitle:
            plt.suptitle(suptitle, x=0.52, y=1.04, fontsize=18)
        fig.subplots_adjust(wspace=0.4)
    if save_dir:
        base = 'feature_coding_by_experience_level' + suffix
        if save_fig:
            utils.save_figure(fig, figsize, save_dir, folder, _clean_filename(base))
        try:
            summary.to_csv(os.path.join(save_dir, folder, _clean_filename(base + '_values.csv')))
        except BaseException:
            print('STATS DID NOT SAVE FOR feature_coding_by_experience_level')
    return ax




# ------------------------------------------------------------------
# Single-axis distributions of running modulation / activity-running
# correlation. Used by notebooks/platform_paper_figures/running_activity_correlation_control.ipynb
# ------------------------------------------------------------------

def _draw_box_or_violin_compact(d, x, y, order, hue, hue_order, palette, ax, plot_type='boxplot'):
    """Internal helper for the *_distribution_single_axis plotters."""
    if plot_type == 'boxplot':
        sns.boxplot(
            data=d, x=x, y=y, order=order,
            hue=hue, hue_order=hue_order, palette=palette,
            width=0.6, notch=True, fliersize=0, boxprops=dict(alpha=0.75), ax=ax,
        whis=1.5,
        )
    elif plot_type == 'violinplot':
        sns.violinplot(
            data=d, x=x, y=y, order=order,
            hue=hue, hue_order=hue_order, palette=palette,
            cut=0, linewidth=1, gap=0.1, fill=True, ax=ax,
            inner='box', inner_kws=dict(box_width=2, whis_width=1, color='k', alpha=0.75),
        density_norm="area",
        bw_method="scott",
        )
        plt.setp(ax.collections, alpha=0.7)
    else:
        raise ValueError(f"plot_type must be 'boxplot' or 'violinplot', got {plot_type!r}")


def plot_rmi_distribution_single_axis(metrics_table, metric='running_modulation_all_images',
                                       x_col='cell_type', hue_col='experience_level',
                                       plot_type='boxplot', ylabel='Running\nmodulation',
                                       ylims=(-1.2, 1.25), annot=('Stationary', 'Running'),
                                       save_dir=None, folder='running_modulation', ax=None):
    """Single-axis distribution of `metric` with `x_col` on x and `hue_col` as the within-x split.
    `x_col` and `hue_col` must each be one of {'cell_type', 'experience_level'}.

    Hierarchical stats are computed across hue values within each x value via
    add_stats_to_plot_for_hues. Y-axis is annotated with `annot` (bottom, top) near
    ylims (e.g. 'Stationary' near -1, 'Running' near +1). Pass annot=None to skip.
    """
    assert {x_col, hue_col} <= {'cell_type', 'experience_level'}, \
        "x_col and hue_col must each be 'cell_type' or 'experience_level'"
    assert x_col != hue_col, 'x_col and hue_col must differ'

    cell_types = utils.get_cell_types()
    cell_type_abbrev = [ct[:3] for ct in cell_types]
    exp_levels = utils.get_new_experience_levels()

    if x_col == 'cell_type':
        x_order = cell_types
        x_labels = cell_type_abbrev
        hue_order = exp_levels
        palette = utils.get_experience_level_colors()
    else:
        x_order = exp_levels
        x_labels = exp_levels
        hue_order = cell_types
        palette = utils.get_cell_type_colors()

    d = metrics_table.dropna(subset=[metric]).copy()

    if ax is None:
        figsize = (4, 3)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        figsize = fig.get_size_inches()

    _draw_box_or_violin_compact(d, x_col, metric, x_order, hue_col, hue_order, palette, ax, plot_type)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylims)
    ax.set_xticks(range(len(x_order)))
    xticklabels = ax.set_xticklabels(x_labels)
    if x_col == 'experience_level':
        # color the experience-level tick labels by their experience-level colors
        for ticklabel, color in zip(xticklabels, utils.get_experience_level_colors()):
            ticklabel.set_color(color)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.2, xlim[1] + 0.2)
    ax.legend(title='', frameon=False, fontsize=9, bbox_to_anchor=(1.05, 1.0))

    if annot is not None:
        ax.annotate(annot[1], xy=(-0.05, 0.98), xycoords=ax.transAxes,
                    ha='right', va='top', fontsize=10)
        ax.annotate(annot[0], xy=(-0.05, -0.05), xycoords=ax.transAxes,
                    ha='right', va='bottom', fontsize=10)

    try:
        ax, stats_table = add_stats_to_plot_for_hues(
            d, metric, ax, ymax=ylims[1], xorder=x_order, x=x_col, hue=hue_col,
            event_type='all',
        )
    except Exception as e:
        print(f'stats failed: {e}')
        stats_table = None

    sns.despine()

    if save_dir is not None:
        save_name = _clean_filename(metric + '_single_axis_' + x_col + '_x_' + hue_col + '_hue')
        utils.save_figure(fig, figsize, save_dir, folder, save_name)
        if stats_table is not None:
            os.makedirs(os.path.join(save_dir, folder), exist_ok=True)
            stats_table.to_csv(os.path.join(save_dir, folder, save_name + '_stats.csv'))
    return ax


def plot_correlation_distribution_single_axis(metrics_table, metric='activity_running_corr',
                                               x_col='cell_type', hue_col='experience_level',
                                               plot_type='boxplot',
                                               ylabel='Activity vs.running\ncorrelation (pearson r)',
                                               ylims=None,
                                               save_dir=None, folder='running_correlation', save_name=None, ax=None):
    """Sister of plot_rmi_distribution_single_axis for distributions of correlation values.
    Same layout/stats; no 'Running'/'Stationary' annotation; ylims default to None (autoscale)
    since correlation distributions are typically narrower than +/-1.
    """
    assert {x_col, hue_col} <= {'cell_type', 'experience_level'}, \
        "x_col and hue_col must each be 'cell_type' or 'experience_level'"
    assert x_col != hue_col, 'x_col and hue_col must differ'

    cell_types = utils.get_cell_types()
    cell_type_abbrev = [ct[:3] for ct in cell_types]
    exp_levels = utils.get_new_experience_levels()

    if x_col == 'cell_type':
        x_order, x_labels = cell_types, cell_type_abbrev
        hue_order, palette = exp_levels, utils.get_experience_level_colors()
    else:
        x_order, x_labels = exp_levels, exp_levels
        hue_order, palette = cell_types, utils.get_cell_type_colors()

    d = metrics_table.dropna(subset=[metric]).copy()

    if ax is None:
        figsize = (4, 3)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        figsize = fig.get_size_inches()

    _draw_box_or_violin_compact(d, x_col, metric, x_order, hue_col, hue_order, palette, ax, plot_type)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_xticks(range(len(x_order)))
    xticklabels = ax.set_xticklabels(x_labels)
    if x_col == 'experience_level':
        # color the experience-level tick labels by their experience-level colors
        for ticklabel, color in zip(xticklabels, utils.get_experience_level_colors()):
            ticklabel.set_color(color)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.2, xlim[1] + 0.2)
    ax.legend(title='', frameon=False, fontsize=9, bbox_to_anchor=(1.05, 1.0))

    try:
        ax, stats_table = add_stats_to_plot_for_hues(
            d, metric, ax, ymax=ax.get_ylim()[1], xorder=x_order, x=x_col, hue=hue_col,
            event_type='all',
        )
    except Exception as e:
        print(f'stats failed: {e}')
        stats_table = None

    sns.despine()

    if save_dir is not None:
        if save_name is None:
            save_name = _clean_filename(metric + '_single_axis_' + x_col + '_x_' + hue_col + '_hue')
        utils.save_figure(fig, figsize, save_dir, folder, save_name, formats=['.png', '.pdf'])
        if stats_table is not None:
            os.makedirs(os.path.join(save_dir, folder), exist_ok=True)
            stats_table.to_csv(os.path.join(save_dir, folder, save_name + '_stats.csv'))
    return ax
