"""
Supplemental Figures S3-S5 - Extended behavioral analysis (associated with Paper Figure 2)

This script generates supplemental figures showing detailed behavioral training history,
performance across cohorts, and behavioral timeseries analysis.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

# Data access
from visual_behavior.data_access import loading as loading
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

# Utilities
import visual_behavior.utilities as vbu
import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.behavior

# Visualization modules
import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.visualization.ophys.experiment_summary_figures as df

# ============================================================================
# CONFIGURATION
# ============================================================================
save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'figure_2_supplemental')
folder = 'behavior_metrics'

# ============================================================================
# DATA LOADING
# ============================================================================
# Initialize cache
cache_dir = loading.get_platform_analysis_cache_dir()
cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=cache_dir, use_static_cache=True)

# Get behavior session tables
all_behavior_sessions = cache.get_behavior_session_table()
all_behavior_sessions['mouse_id'] = [int(mouse_id) for mouse_id in all_behavior_sessions.mouse_id.values]
all_behavior_sessions['experience_level'] = [utils.convert_experience_level(experience_level)
                                             for experience_level in all_behavior_sessions.experience_level.values]

# Get ophys experiment tables
all_experiments_table = cache.get_ophys_experiment_table()
all_experiments_table['mouse_id'] = [int(mouse_id) for mouse_id in all_experiments_table.mouse_id.values]
all_experiments_table['experience_level'] = [utils.convert_experience_level(experience_level)
                                            for experience_level in all_experiments_table.experience_level.values]

# Get platform paper specific tables
platform_experiments = loading.get_platform_paper_experiment_table(limit_to_closest_active=True, include_4x2_data=False)
platform_experiments['mouse_id'] = [int(mouse_id) for mouse_id in platform_experiments.mouse_id.values]
platform_experiments['experience_level'] = [utils.convert_experience_level(experience_level)
                                           for experience_level in platform_experiments.experience_level.values]

platform_behavior_sessions = loading.get_platform_paper_behavior_session_table()
platform_behavior_sessions['mouse_id'] = [int(mouse_id) for mouse_id in platform_behavior_sessions.mouse_id.values]
platform_behavior_sessions['experience_level'] = [utils.convert_experience_level(experience_level)
                                                 for experience_level in platform_behavior_sessions.experience_level.values]

# Limit behavior sessions to platform experiment mice
behavior_sessions = platform_behavior_sessions[platform_behavior_sessions.mouse_id.isin(platform_experiments.mouse_id.unique())]

# Add useful columns for filtering
original_behavior_sessions = behavior_sessions.copy()
behavior_sessions['ophys_container_id'] = behavior_sessions.mouse_id.values
behavior_sessions = utilities.add_date_string(behavior_sessions)
behavior_sessions = utilities.add_n_relative_to_first_novel_column(behavior_sessions)
behavior_sessions = utilities.add_first_novel_column(behavior_sessions)
behavior_sessions = utilities.add_second_novel_active_column(behavior_sessions)
behavior_sessions = utilities.add_last_familiar_active_column(behavior_sessions)

# Get utility lists and colormaps
experience_levels = utils.get_experience_levels()
new_experience_levels = utils.get_new_experience_levels()
cre_lines = utils.get_cre_lines()
cell_types = utils.get_cell_types()
experience_colors = utils.get_experience_level_colors()

# Load behavior statistics
method = 'sdk'
behavior_session_ids = behavior_sessions.index.values
original_behavior_stats_sdk, problem_sessions = vbu.get_behavior_stats_for_sessions(behavior_session_ids, behavior_sessions,
                                                method=method, engaged_only=False, per_image=False)
behavior_stats_sdk = original_behavior_stats_sdk.merge(behavior_sessions, on='behavior_session_id')

# Load stimulus-based behavior metrics (engaged only)
method = 'stimulus_based'
original_engaged_behavior_stats_stim, problem_sessions = vbu.get_behavior_stats_for_sessions(behavior_session_ids, behavior_sessions,
                                                method=method, engaged_only=True, per_image=False)
engaged_behavior_stats_stim = original_engaged_behavior_stats_stim.merge(behavior_sessions, on='behavior_session_id')
engaged_behavior_stats_stim = engaged_behavior_stats_stim[engaged_behavior_stats_stim.project_code != 'VisualBehaviorMultiscope4areasx2d']

# Load platform behavior stats for stimulus-based metrics
platform_behavior_stats = pd.read_csv(os.path.join(cache_dir, 'behavior_performance', 'platform_behavior_stats_engaged.csv'), index_col=0)

# ============================================================================
# SUPPLEMENTAL FIGURE S3: Training history and behavior performance
# ============================================================================

def s3_training_history_and_behavior():
    """
    S3 - Training history and behavior performance across stages:
    - Training history for mice
    - Days in stage
    - Prior exposures to image sets
    - Stimulus exposures
    - Individual mouse behavior performance over time
    - Behavior metrics across stages
    - Response rates for trial types
    """

    # Add behavior stage information
    behavior_sessions_local = utilities.add_behavior_stage_to_behavior_sessions(behavior_sessions)

    # Plot training history for mice grouped by cre line
    color_column = 'behavior_stage'
    color_map = utils.get_behavior_stage_color_map(as_rgb=True)
    ppf.plot_training_history_for_mice(behavior_sessions_local, color_column=color_column,
                                       color_map=color_map, group_by_cre_line=True,
                                       save_dir=save_dir, folder='training_history', suffix='_by_cre')

    # Plot training history without cre line grouping
    ppf.plot_training_history_for_mice(behavior_sessions_local, color_column=color_column,
                                       color_map=color_map, group_by_cre_line=False,
                                       save_dir=save_dir, folder='training_history', suffix='')

    # Plot days in each stage
    ppf.plot_days_in_stage(behavior_sessions_local, 'behavior_stage', save_dir, 'training_history',
                          suffix='_behavior_stage')

    # Plot prior exposures to image sets before platform ophys sessions
    ppf.plot_prior_exposures_to_image_set_before_platform_ophys_sessions(platform_experiments,
                                                                    behavior_sessions_local,
                                                                    save_dir=save_dir,
                                                                    folder='training_history',
                                                                    suffix='', ax=None)

    # Plot horizontal version
    ppf.plot_prior_exposures_to_image_set_before_platform_ophys_sessions_horiz(platform_experiments,
                                                                         behavior_sessions_local,
                                                                         save_dir=save_dir,
                                                                         folder='training_history',
                                                                         suffix='_horiz', ax=None)

    # Prior exposures for Familiar and Novel only
    tmp_expts = platform_experiments.copy()
    tmp_expts = tmp_expts[tmp_expts.experience_level.isin(['Familiar', 'Novel'])]
    ppf.plot_prior_exposures_to_image_set_before_platform_ophys_sessions(tmp_expts,
                                                                    behavior_sessions_local,
                                                                    save_dir=save_dir,
                                                                    folder='training_history',
                                                                    suffix='_F_N', ax=None)

    # Prior exposures per cell type for novel plus
    ppf.plot_prior_exposures_per_cell_type_for_novel_plus(platform_experiments,
                                                          behavior_sessions_local,
                                                          save_dir=save_dir,
                                                          folder='training_history',
                                                          suffix='', ax=None)

    # Total stimulus exposures
    ppf.plot_total_stimulus_exposures(behavior_sessions_local, save_dir=save_dir,
                                     folder='training_history', suffix='', ax=None)

    # Stimulus exposure prior to imaging
    ppf.plot_stimulus_exposure_prior_to_imaging(behavior_sessions_local, column_to_group='behavior_stage',
                                               save_dir=save_dir, folder='training_history', suffix='', ax=None)

    ppf.plot_stimulus_exposure_prior_to_imaging(behavior_sessions_local, column_to_group='stimulus_type',
                                               save_dir=save_dir, folder='training_history', suffix='_by_stimulus', ax=None)

    # Example mouse behavior performance over time
    example_mouse_ids = [512458, 456917, 485688, 492395, 533161, 546605, 445002, 489066, 513626]
    mouse_id = example_mouse_ids[0]
    metric = 'dprime_trial_corrected'
    ylabel = 'd-prime'

    ppf.plot_behavior_performance_for_one_mouse(engaged_behavior_stats_stim, mouse_id, metric,
                                               method='stim_based', x='date_of_acquisition',
                                               hue='stimulus', ylabel=ylabel,
                                               save_dir=save_dir, folder='behavior_performance', ax=None)

    ppf.plot_behavior_performance_for_one_mouse(engaged_behavior_stats_stim, mouse_id, metric,
                                               method='stim_based', x='date_of_acquisition',
                                               hue='behavior_stage', use_session_number=True,
                                               ylabel=ylabel, save_dir=save_dir,
                                               folder='behavior_performance', ax=None)

    # Behavior metrics across training stages
    data = behavior_stats_sdk.copy()
    metric = 'mean_dprime'
    data = data[data[metric] > 0]
    ppf.plot_behavior_metric_across_stages(data, metric, ylabel='Mean d-prime',
                                          save_dir=save_dir, folder='behavior_metrics', suffix='_sdk')

    data = behavior_stats_sdk.copy()
    metric = 'max_dprime'
    data = data[data[metric] > 0]
    ppf.plot_behavior_metric_across_stages(data, metric, ylabel='Max d-prime',
                                          save_dir=save_dir, folder='behavior_metrics', suffix='_sdk_max')

    # Fraction trials engaged
    behavior_stats_sdk['fraction_trials_engaged'] = (behavior_stats_sdk.engaged_trial_count_x /
                                                    behavior_stats_sdk.trial_count_x)
    data = behavior_stats_sdk.copy()
    metric = 'fraction_trials_engaged'
    ppf.plot_behavior_metric_across_stages(data, metric, ylabel='Fraction engaged trials',
                                          save_dir=save_dir, folder='behavior_metrics', suffix='_fraction_engaged')

    # Hit rate across stages
    data = behavior_stats_sdk.copy()
    metric = 'mean_hit_rate'
    ppf.plot_behavior_metric_across_stages(data, metric, ylabel='Hit rate',
                                          save_dir=save_dir, folder='behavior_metrics', suffix='_hit_rate')

    # False alarm rate across stages
    data = behavior_stats_sdk.copy()
    metric = 'mean_false_alarm_rate'
    ppf.plot_behavior_metric_across_stages(data, metric, ylabel='False alarm rate',
                                          save_dir=save_dir, folder='behavior_metrics', suffix='_fa_rate')

    # Hit rate (engaged only)
    data = behavior_stats_sdk.copy()
    metric = 'mean_hit_rate_engaged'
    ppf.plot_behavior_metric_across_stages(data, metric, ylabel='Hit rate (engaged)',
                                          save_dir=save_dir, folder='behavior_metrics', suffix='_hit_rate_engaged')


# ============================================================================
# SUPPLEMENTAL FIGURE S4: Cohort-level behavioral performance
# ============================================================================

def s4_cohort_behavioral_performance():
    """
    S4 - Cohort-level behavioral performance:
    - Behavior metrics by cohort
    - Response probability heatmaps for cohorts
    """

    # Plot response rate for go and catch trials using stimulus-based metrics
    stats = engaged_behavior_stats_stim.copy()
    stats = stats[['hit_rate', 'fa_rate', 'behavior_session_id']].melt(id_vars='behavior_session_id')
    stats['response_probability'] = stats['value']
    stats['trial_type'] = stats['variable']
    stats['trial_type'] = [trial_type.split('_')[0] for trial_type in stats.trial_type.values]
    stats = stats.merge(behavior_sessions[['mouse_id', 'experience_level', 'cell_type', 'session_type']],
                       on='behavior_session_id')

    # Platform ophys sessions
    data = stats.copy()
    data = data[data.behavior_session_id.isin(platform_behavior_sessions.index.values)]
    suffix = '_platform_behavior_sessions'
    ppf.plot_response_rate_trial_types(data, save_dir=save_dir, suffix=suffix, ax=None)

    # All ophys sessions
    data = stats.copy()
    data = data[data.session_type.str.contains('OPHYS')]
    suffix = '_platform_ophys_sessions'
    ppf.plot_response_rate_trial_types(data, save_dir=save_dir, suffix=suffix, ax=None)

    # Handoff ready sessions
    data = stats.copy()
    data = data[data.session_type.str.contains('handoff_ready')]
    suffix = '_handoff_ready'
    ppf.plot_response_rate_trial_types(data, save_dir=save_dir, suffix=suffix, ax=None)

    # Plot behavior metrics by cohort - d-prime metrics
    metric = 'max_dprime_engaged'
    stats = behavior_stats_sdk.copy()
    stats = stats[stats[metric] > 0]
    stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]
    stats['ophys_container_id'] = stats['mouse_id']

    project_codes = ['VisualBehavior', 'VisualBehaviorTask1B', 'VisualBehaviorMultiscope']
    figsize = (6, 2.5)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for i, project_code in enumerate(project_codes):
        title = 'Cohort ' + str(i + 1)
        tmp = stats[stats.project_code == project_code]
        suffix = '_stim_based_' + project_code

        ax[i] = ppf.plot_behavior_metric_by_cohort(tmp, metric, title=title, ylabel='D-prime',
                                                   ylims=(0, 3), show_containers=True,
                                                   plot_stats=True, stripplot=True, show_ns=True,
                                                   save_dir=None, folder='behavior_metrics',
                                                   suffix=suffix, ax=ax[i])
        ax[i].set_ylabel('')
    ax[0].set_ylabel('D-prime')
    fig.subplots_adjust(wspace=0.3)
    utils.save_figure(fig, figsize, save_dir, folder, metric + '_cohorts')

    # Response latency by cohort
    metric = 'response_latency_mean'
    ylabel = 'Response latency (s)'
    stats = engaged_behavior_stats_stim.copy()
    stats = stats[stats[metric] > 0]
    stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]
    stats['ophys_container_id'] = stats['mouse_id']

    figsize = (6, 2.5)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for i, project_code in enumerate(project_codes):
        title = 'Cohort ' + str(i + 1)
        tmp = stats[stats.project_code == project_code]
        suffix = '_stim_based_' + project_code

        ax[i] = ppf.plot_behavior_metric_by_cohort(tmp, metric, title=title, ylabel=ylabel,
                                                   ylims=(0, 0.7), show_containers=True,
                                                   plot_stats=True, stripplot=True, show_ns=True,
                                                   save_dir=None, folder='behavior_metrics',
                                                   suffix=suffix, ax=ax[i])
        ax[i].set_ylabel('')
    ax[0].set_ylabel(ylabel)
    fig.subplots_adjust(wspace=0.3)
    utils.save_figure(fig, figsize, save_dir, folder, metric + '_cohorts')

    # Response probability heatmaps for cohorts
    ppf.plot_response_probability_heatmaps_for_cohorts(behavior_sessions, save_dir=save_dir)


# ============================================================================
# SUPPLEMENTAL FIGURE S5: Behavioral timeseries and performance details
# ============================================================================

def s5_behavioral_timeseries_details():
    """
    S5 - Behavioral timeseries details:
    - Metric distributions by experience level for behavior metrics
    - Behavior metrics by experience level (fraction engaged, response latency)
    - Ophys history for mice
    """

    # Distribution of behavior metrics by experience level
    metric = 'dprime_trial_corrected'
    event_type = 'changes'

    stats = engaged_behavior_stats_stim.copy()
    stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]

    ppf.plot_metric_distribution_by_experience(stats, metric, event_type='changes',
                                              ylabel='d-prime', ylims=None, show_mice=False,
                                              pointplot=True, stripplot=True, horiz=True,
                                              show_ns=False, add_zero_line=False,
                                              abbreviate_exp=True, save_dir=save_dir)

    # Behavior metrics by experience level - SDK metrics
    metric = 'max_dprime'
    ylabel = 'd-prime'
    stats = behavior_stats_sdk.copy()
    stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]

    suffix = '_sdk_platform_experiments'
    ppf.plot_behavior_metric_by_experience(stats, metric, title='Behavior performance',
                                          ylabel=ylabel, ylims=[-0.1, 3.3],
                                          best_image=False, show_mice=False, plot_stats=True,
                                          stripplot=True, show_ns=False,
                                          save_dir=save_dir, folder='behavior_metrics', suffix=suffix)

    # Horizontal version
    ppf.plot_behavior_metric_by_experience_horiz(stats, metric, title='Behavior performance',
                                                xlabel='d-prime', xlims=[-0.1, 3.3],
                                                best_image=False, show_containers=False,
                                                plot_stats=False, stripplot=True, show_ns=False,
                                                save_dir=save_dir, folder='behavior_metrics', suffix=suffix)

    # Without Novel+ (Familiar and Novel only)
    metric = 'max_dprime'
    stats = behavior_stats_sdk.copy()
    stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]
    stats = stats[stats.experience_level.isin(['Familiar', 'Novel'])]

    suffix = '_sdk_platform_experiments_F_N'
    ppf.plot_behavior_metric_by_experience(stats, metric, title='Behavior performance',
                                          ylabel='d-prime', ylims=[-0.1, 3.3],
                                          best_image=False, show_mice=False, plot_stats=True,
                                          stripplot=True, show_ns=True,
                                          save_dir=save_dir, folder='behavior_metrics', suffix=suffix)

    # Max dprime engaged
    metric = 'max_dprime_engaged'
    stats = behavior_stats_sdk.copy()
    stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]

    suffix = '_sdk_max_dprime_engaged'
    ppf.plot_behavior_metric_by_experience(stats, metric, title='Behavior performance',
                                          ylabel='d-prime', ylims=[-0.1, 3.3],
                                          best_image=False, show_mice=False, plot_stats=True,
                                          stripplot=True, show_ns=False,
                                          save_dir=save_dir, folder='behavior_metrics', suffix=suffix)

    # Mean dprime engaged
    metric = 'mean_dprime_engaged'
    stats = behavior_stats_sdk.copy()
    stats = stats[stats[metric] > 0]
    stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]

    suffix = '_sdk_mean_dprime_engaged'
    ppf.plot_behavior_metric_by_experience(stats, metric, title='Behavior performance',
                                          ylabel='d-prime', ylims=[-0.1, 3.3],
                                          best_image=False, show_mice=False, plot_stats=True,
                                          stripplot=True, show_ns=False,
                                          save_dir=save_dir, folder='behavior_metrics', suffix=suffix)

    # Fraction engaged and response latency metrics
    metric = 'fraction_engaged'
    stats = engaged_behavior_stats_stim.copy()
    stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]

    ppf.plot_behavior_metric_by_experience(stats, metric, title='Engagement',
                                          ylabel='Fraction engaged', ylims=None,
                                          best_image=False, show_mice=False, plot_stats=True,
                                          stripplot=True, show_ns=False,
                                          save_dir=save_dir, folder='behavior_metrics', suffix='_fraction_engaged')

    metric = 'response_latency'
    stats = engaged_behavior_stats_stim.copy()
    stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]

    ppf.plot_behavior_metric_by_experience(stats, metric, title='Response latency',
                                          ylabel='Response latency (s)', ylims=None,
                                          best_image=False, show_mice=False, plot_stats=True,
                                          stripplot=True, show_ns=False,
                                          save_dir=save_dir, folder='behavior_metrics', suffix='_response_latency')

    # Ophys history for mice (all mice in full dataset)
    behavior_sessions_all = all_behavior_sessions.copy()
    behavior_sessions_all['ophys_container_id'] = behavior_sessions_all.mouse_id.values
    behavior_sessions_all = utilities.add_date_string(behavior_sessions_all)
    behavior_sessions_all = utilities.add_n_relative_to_first_novel_column(behavior_sessions_all)
    behavior_sessions_all = utilities.add_first_novel_column(behavior_sessions_all)
    behavior_sessions_all = utilities.add_passive_flag_to_ophys_experiment_table(behavior_sessions_all)
    behavior_sessions_all = utilities.add_ophys_stage_to_behavior_sessions(all_experiments_table, behavior_sessions_all)

    # Exclude 4x2 sessions
    behavior_sessions_all = behavior_sessions_all[behavior_sessions_all.project_code != 'VisualBehaviorMultiscope4areasx2d']

    color_column = 'ophys_stage'
    color_map = utils.get_ophys_stage_color_map(as_rgb=True)

    sessions = behavior_sessions_all.copy()
    sessions = sessions[sessions.ophys_stage != 'None']

    ppf.plot_ophys_history_for_mice(sessions, color_column=color_column, color_map=color_map,
                                   save_dir=save_dir, label_with_mouse_id=False, suffix='_all_mice')

    # For platform dataset mice only
    behavior_sessions_platform = utilities.add_ophys_stage_to_behavior_sessions(platform_experiments, behavior_sessions)

    sessions = behavior_sessions_platform.copy()
    sessions = sessions[sessions.ophys_stage != 'None']
    sessions = sessions[sessions.mouse_id.isin(platform_experiments.mouse_id.unique())]

    ppf.plot_ophys_history_for_mice(sessions, color_column=color_column, color_map=color_map,
                                   save_dir=save_dir, label_with_mouse_id=False, suffix='_platform_mice')


# ============================================================================
# CUSTOM FUNCTIONS FOR BEHAVIOR-RESPONSE ANALYSIS
# ============================================================================

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
        filename = 'correlation_' + cell_metric + '_' + behavior_metric
        utils.save_figure(fig, figsize, save_dir, folder, filename)


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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("Generating Supplemental Figures S3-S5...")

    # Create output directory if it doesn't exist
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("\nGenerating S3 - Training history and behavior performance...")
    s3_training_history_and_behavior()

    print("\nGenerating S4 - Cohort-level behavioral performance...")
    s4_cohort_behavioral_performance()

    print("\nGenerating S5 - Behavioral timeseries and performance details...")
    s5_behavioral_timeseries_details()

    print("\nSupplemental figures generation complete!")
