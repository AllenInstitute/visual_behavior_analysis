"""
Supplemental Figures S10-S15 - Extended GLM coding analysis (associated with Paper Figure 4).

This script generates supplemental figures for the platform paper, including:
- S10: Alternative coding score computation with dropout summaries
- S11-S13: Extended cell-type specific analysis with kernel heatmaps
- S14-S15: Coding score distributions by area and depth
- Extended feature coding analysis by cell type and experience level
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.utils as utils
from visual_behavior.dimensionality_reduction.clustering import processing
from visual_behavior.dimensionality_reduction.clustering import plotting

import visual_behavior.visualization.ophys.glm_example_plots as gep
import visual_behavior_glm.GLM_visualization_tools as gvt


# ============================================================================
# Configuration
# ============================================================================

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'figure_4_supplemental')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Color mappings
experience_level_colors = utils.get_experience_level_colors()
cre_line_colors = utils.get_cre_line_colors()
experience_levels = utils.get_new_experience_levels()
cell_types = utils.get_cell_types()


# ============================================================================
# Data Loading
# ============================================================================

# Metadata tables
experiments_table = pd.read_csv(
    os.path.join(loading.get_metadata_tables_dir(), 'all_ophys_experiments_table.csv'),
    index_col=0
)
platform_experiments = pd.read_csv(
    os.path.join(loading.get_metadata_tables_dir(), 'platform_paper_ophys_experiments_table.csv'),
    index_col=0
)
platform_cells_table = pd.read_csv(
    os.path.join(loading.get_metadata_tables_dir(), 'platform_paper_ophys_cells_table.csv'),
    index_col=0
)
matched_cells_table = pd.read_csv(
    os.path.join(loading.get_metadata_tables_dir(), 'platform_paper_matched_ophys_cells_table.csv'),
    index_col=0
)

# Define model version and load GLM results
glm_version = '24_events_all_L2_optimize_by_session'

run_params = pd.read_pickle(
    os.path.join(loading.get_glm_results_dir(), glm_version + '_run_params.pkl')
)
run_params['figure_dir'] = os.path.join(save_dir, 'coding_scores_and_kernels')

kernels = run_params['kernels']
all_weights_df = pd.read_hdf(
    os.path.join(loading.get_glm_results_dir(), 'weights_df.h5'),
    key='df'
)
all_results = pd.read_hdf(
    os.path.join(loading.get_glm_results_dir(), 'all_results.h5'),
    key='df'
)
all_results_pivoted = pd.read_hdf(
    os.path.join(loading.get_glm_results_dir(), 'results_pivoted.h5'),
    key='df'
)
all_results_pivoted = processing.set_negative_one_values_to_zero(all_results_pivoted)

# Split out cell_specimen_id and ophys_experiment_id for results table
all_results['cell_specimen_id'] = [
    int(identifier.split('_')[1])
    for identifier in all_results.identifier.values
]
all_results['ophys_experiment_id'] = [
    int(identifier.split('_')[0])
    for identifier in all_results.identifier.values
]

# Get results and weights just for platform paper experiments
results_pivoted = pd.read_hdf(
    os.path.join(loading.get_glm_results_dir(), 'platform_results_pivoted.h5'),
    key='df'
)
results_pivoted = processing.set_negative_one_values_to_zero(results_pivoted)

# Make coding scores positive
results_pivoted[processing.get_features_for_clustering()] = np.abs(
    results_pivoted[processing.get_features_for_clustering()]
)

weights_df = pd.read_hdf(
    os.path.join(loading.get_glm_results_dir(), 'platform_results_weights_df.h5'),
    key='df'
)

# Get lists of matched cells and experiments
matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()

# Get cre_lines and cell types for plot labels
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types_dict = utilities.get_cell_types_dict(cre_lines, platform_experiments)

# Plot kernels, traces and dropouts examples for experiments
gep.plot_kernels_traces_and_dropouts_examples_for_experiment(
    matched_experiments[0], run_params, all_results, results_pivoted, weights_df,
    use_var_exp=False, save_dir=save_dir
)

# Plot model fits and kernels for example cells
dataset_example = loading.get_ophys_dataset(matched_experiments[0], get_extended_stimulus_presentations=False)
frame_rate = 31
image_sdf, omission_sdf, change_sdf = gep.get_stimulus_response_dfs_for_kernel_windows(dataset_example, kernels, frame_rate)
gep.plot_model_fits_and_kernels_for_example_cell(
    matched_experiments[0], matched_cells[0], dataset_example,
    image_sdf, omission_sdf, change_sdf,
    results_pivoted, weights_df, kernels, save_dir=save_dir
)


# ============================================================================
# S10 - Alternative coding score computation
# ============================================================================

# Filter for non-passive sessions
results_pivoted_filtered = results_pivoted.query('not passive').copy()
results = all_results[
    all_results.ophys_experiment_id.isin(platform_experiments.index.unique())
].copy()

# Dropout summary for population
stats_S10A = gvt.plot_dropout_summary_population(all_results, run_params)

# Dropout individual population plots
stats_S10B = gvt.plot_dropout_individual_population(results, run_params, savefig=True)
stats_S10C = gvt.plot_dropout_individual_population(
    results, run_params, use_single=True, savefig=True
)

# Variance explained for matched cells
stats_S10D = gvt.var_explained_matched(results_pivoted_filtered, run_params)


# ============================================================================
# Extended GLM example plots for individual cells
# ============================================================================

# Select example cells for detailed analysis
ophys_experiment_id = 967008471
cell_index = 0
expt_dropouts = results_pivoted[results_pivoted.ophys_experiment_id == ophys_experiment_id]
cell_specimen_id = expt_dropouts.sort_values(by='variance_explained_full', ascending=False).cell_specimen_id.values[cell_index]

# Load dataset and GLM results
dataset = loading.get_ophys_dataset(ophys_experiment_id, get_extended_stimulus_presentations=False)
cell_results_df, expt_results, dropouts = gep.load_glm_model_fit_results(ophys_experiment_id)

# Get stimulus response dfs for kernel windows
frame_rate = 31
image_sdf, omission_sdf, change_sdf = gep.get_stimulus_response_dfs_for_kernel_windows(dataset, kernels, frame_rate)

# Plot extended GLM example plots
gep.plot_image_kernels_and_traces_for_cell(
    cell_specimen_id, dataset,
    image_sdf, omission_sdf, change_sdf,
    weights_df, kernels, save_dir=save_dir
)

gep.plot_all_kernels_and_traces_for_cell(
    cell_specimen_id, dataset,
    image_sdf, omission_sdf, change_sdf,
    weights_df, kernels, show_cell_response=False, save_dir=save_dir
)

gep.plot_coding_scores_for_cell(
    cell_specimen_id, ophys_experiment_id, results_pivoted, save_dir=save_dir
)

gep.plot_coding_score_components_for_cell(
    cell_specimen_id, ophys_experiment_id, results_pivoted, dataset,
    fontsize=14, as_panel=False, save_dir=save_dir
)

gep.plot_stacked_kernels_for_cell(
    cell_specimen_id, dataset, weights_df, kernels,
    save_dir=save_dir
)

gep.plot_model_fits_example_cell(
    cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
    'all-images', twinx=False, include_events=True, include_dff=False,
    save_dir=save_dir
)


# ============================================================================
# S11-S13 - Extended cell-type specific analysis
# ============================================================================

# Kernel heatmaps with dropout by experience level
for kernel in ['image0', 'all-images', 'omissions']:
    if kernel in run_params['kernels']:
        time_vec = np.arange(
            run_params['kernels'][kernel]['offset'],
            run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],
            1/31.
        )
        time_vec = np.round(time_vec, 2)
        if 'image' in kernel:
            time_vec = time_vec[:-1]

        sst_table = weights_df.query('cre_line == "Sst-IRES-Cre"')[[kernel + '_weights', kernel]]
        vip_table = weights_df.query('cre_line == "Vip-IRES-Cre"')[[kernel + '_weights', kernel]]
        slc_table = weights_df.query('cre_line == "Slc17a7-IRES2-Cre"')[[kernel + '_weights', kernel]]

        # Plot with dropout by experience
        gvt.plot_kernel_heatmap_with_dropout_by_experience(
            vip_table, sst_table, slc_table,
            time_vec, kernel, run_params, ncells={}, ax=None, extra='', zlims=None,
            session_filter=['Familiar', 'Novel', 'Novel +'], savefig=True,
            cell_filter='', sort_by_dropout=False, limited=False
        )

        # Plot with dropout for Familiar session
        gvt.plot_kernel_heatmap_with_dropout(
            vip_table, sst_table, slc_table, time_vec, kernel,
            run_params, ncells={}, ax=None, extra='', zlims=None,
            session_filter=['Familiar'], savefig=True
        )

# Kernel weights and coding score heatmaps for all kernels by experience level
for kernel in ['all-images', 'omissions', 'hits', 'misses', 'running', 'pupil', 'licks']:
    if kernel in run_params['kernels']:
        if kernel == 'all-images':
            xlabel = 'Time (s)'
            vmax = 0.004
        elif kernel == 'omissions':
            xlabel = 'Time after omission (s)'
            vmax = 0.002
        elif kernel in ['hits', 'misses']:
            xlabel = 'Time after change (s)'
            vmax = 0.002
        else:
            xlabel = 'Time (s)'
            vmax = 0.002

        gep.plot_weights_and_coding_score_heatmaps_for_experience_levels(
            kernel, weights_df, run_params,
            row_condition='cre_line', col_condition='experience_level',
            vmax=vmax, xlabel=xlabel,
            save_dir=save_dir, folder='kernel_weights'
        )


# ============================================================================
# S14-S15 - Coding score distributions by area and depth
# ============================================================================

# Population averages by depth
summary_depth = gvt.plot_population_averages_by_depth(
    results_pivoted_filtered, run_params,
    dropouts_to_show=['all-images', 'omissions', 'behavioral', 'task'],
    sharey=False, include_zero_cells=True, add_stats=True,
    extra='', equipment="mesoscope", area=['VISp', 'VISl'], savefig=True
)

# Population averages by area with mesoscope equipment filter
summary_area = gvt.plot_population_averages_by_area(
    results_pivoted_filtered, run_params,
    dropouts_to_show=['all-images', 'omissions', 'behavioral', 'task'],
    sharey=False, include_zero_cells=True, add_stats=True,
    extra='', equipment="mesoscope", savefig=True
)

# Test significant differences across cell types
anova_images, tukey_images = gvt.test_significant_across_cell(
    results_pivoted_filtered, 'all-images'
)
anova_omissions, tukey_omissions = gvt.test_significant_across_cell(
    results_pivoted_filtered, 'omissions'
)
anova_behavioral, tukey_behavioral = gvt.test_significant_across_cell(
    results_pivoted_filtered, 'behavioral'
)
anova_task, tukey_task = gvt.test_significant_across_cell(
    results_pivoted_filtered, 'task'
)


# ============================================================================
# Extended feature coding analysis by cell type and experience level
# ============================================================================

# Compute coding fractions with threshold
coding_thresh = 0.1

results_pivoted_filtered['code_anything'] = (
    results_pivoted_filtered['variance_explained_full'] > run_params['dropout_threshold']
)
results_pivoted_filtered['code_images'] = (
    results_pivoted_filtered['code_anything'] & (results_pivoted_filtered['all-images'] > coding_thresh)
)
results_pivoted_filtered['code_omissions'] = (
    results_pivoted_filtered['code_anything'] & (results_pivoted_filtered['omissions'] > coding_thresh)
)
results_pivoted_filtered['code_behavior'] = (
    results_pivoted_filtered['code_anything'] & (results_pivoted_filtered['behavioral'] > coding_thresh)
)
results_pivoted_filtered['code_task'] = (
    results_pivoted_filtered['code_anything'] & (results_pivoted_filtered['task'] > coding_thresh)
)

# Summary statistics by cre_line and experience_level
summary_df = results_pivoted_filtered.groupby(['cre_line', 'experience_level'])[
    ['code_anything', 'code_images', 'code_omissions', 'code_behavior', 'code_task']
].mean()

# Compute confidence intervals
summary_df['n'] = results_pivoted_filtered.groupby(['cre_line', 'experience_level'])[
    ['code_anything', 'code_images', 'code_omissions', 'code_behavior', 'code_task']
].count()['code_anything']

summary_df['code_images_ci'] = 1.96 * np.sqrt(
    (summary_df['code_images'] * (1 - summary_df['code_images'])) / summary_df['n']
)
summary_df['code_omissions_ci'] = 1.96 * np.sqrt(
    (summary_df['code_omissions'] * (1 - summary_df['code_omissions'])) / summary_df['n']
)
summary_df['code_behavior_ci'] = 1.96 * np.sqrt(
    (summary_df['code_behavior'] * (1 - summary_df['code_behavior'])) / summary_df['n']
)
summary_df['code_task_ci'] = 1.96 * np.sqrt(
    (summary_df['code_task'] * (1 - summary_df['code_task'])) / summary_df['n']
)

# Plot feature coding by experience level
coding_groups = ['code_images', 'code_omissions', 'code_behavior', 'code_task']
titles = ['images', 'omissions', 'behavior', 'task']
colors = utils.get_cell_type_colors()

fig, ax = plt.subplots(1, len(coding_groups), figsize=(10.8, 4), sharey=True)
for index, feature in enumerate(coding_groups):
    # Plot three cre-lines with standard colors
    ax[index].plot(
        [0, 1, 2], summary_df.loc['Vip-IRES-Cre'][feature],
        '-', color=colors[2], label='Vip Inhibitory', linewidth=3
    )
    ax[index].plot(
        [0, 1, 2], summary_df.loc['Sst-IRES-Cre'][feature],
        '-', color=colors[1], label='Sst Inhibitory', linewidth=3
    )
    ax[index].plot(
        [0, 1, 2], summary_df.loc['Slc17a7-IRES2-Cre'][feature],
        '-', color=colors[0], label='Excitatory', linewidth=3
    )

    # Add error bars
    ax[index].errorbar(
        [0, 1, 2], summary_df.loc['Vip-IRES-Cre'][feature],
        yerr=summary_df.loc['Vip-IRES-Cre'][feature + '_ci'],
        color=colors[2], linewidth=3
    )
    ax[index].errorbar(
        [0, 1, 2], summary_df.loc['Sst-IRES-Cre'][feature],
        yerr=summary_df.loc['Sst-IRES-Cre'][feature + '_ci'],
        color=colors[1], linewidth=3
    )
    ax[index].errorbar(
        [0, 1, 2], summary_df.loc['Slc17a7-IRES2-Cre'][feature],
        yerr=summary_df.loc['Slc17a7-IRES2-Cre'][feature + '_ci'],
        color=colors[0], linewidth=3
    )

    ax[index].set_title(titles[index], fontsize=20)
    ax[index].set_ylabel('')
    ax[index].set_xlabel('')
    ax[index].set_xticks([0, 1, 2])
    ax[index].set_xticklabels(experience_levels, rotation=90)
    ax[index].tick_params(axis='x', labelsize=16)
    ax[index].tick_params(axis='y', labelsize=16)
    ax[index].spines['top'].set_visible(False)
    ax[index].spines['right'].set_visible(False)
    ax[index].set_xlim(-0.5, 2.5)
    ax[index].set_ylim(bottom=0)
    if index == 3:
        ax[index].legend(bbox_to_anchor=(1, 1))

ax[0].set_ylabel('Fraction of cells\ncoding for', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_coding_by_experience_level.png'), dpi=150, bbox_inches='tight')
plt.close()

# Additional coding statistics by cre_line only (without experience level)
coding_thresh_0 = 0

results_pivoted_cre = results_pivoted_filtered.copy()
results_pivoted_cre['code_images'] = (
    results_pivoted_cre['code_anything'] & (results_pivoted_cre['all-images'] > coding_thresh_0)
)
results_pivoted_cre['code_omissions'] = (
    results_pivoted_cre['code_anything'] & (results_pivoted_cre['omissions'] > coding_thresh_0)
)
results_pivoted_cre['code_behavioral'] = (
    results_pivoted_cre['code_anything'] & (results_pivoted_cre['behavioral'] > coding_thresh_0)
)
results_pivoted_cre['code_task'] = (
    results_pivoted_cre['code_anything'] & (results_pivoted_cre['task'] > coding_thresh_0)
)

summary_df_cre = results_pivoted_cre.groupby(['cre_line'])[
    ['code_anything', 'code_images', 'code_omissions', 'code_behavioral', 'code_task']
].mean()

summary_df_cre['n'] = results_pivoted_cre.groupby(['cre_line'])[
    ['code_anything', 'code_images', 'code_omissions', 'code_behavioral', 'code_task']
].count()['code_anything']

summary_df_cre['code_images_ci'] = 1.96 * np.sqrt(
    (summary_df_cre['code_images'] * (1 - summary_df_cre['code_images'])) / summary_df_cre['n']
)
summary_df_cre['code_omissions_ci'] = 1.96 * np.sqrt(
    (summary_df_cre['code_omissions'] * (1 - summary_df_cre['code_omissions'])) / summary_df_cre['n']
)
summary_df_cre['code_behavioral_ci'] = 1.96 * np.sqrt(
    (summary_df_cre['code_behavioral'] * (1 - summary_df_cre['code_behavioral'])) / summary_df_cre['n']
)
summary_df_cre['code_task_ci'] = 1.96 * np.sqrt(
    (summary_df_cre['code_task'] * (1 - summary_df_cre['code_task'])) / summary_df_cre['n']
)


# ============================================================================
# Extended analysis with melted data for visualization
# ============================================================================

# Prepare melted results for detailed feature coding visualization
results_melted = results_pivoted_filtered.copy()
results_melted['images'] = results_melted['all-images']
results_melted['behavior'] = results_melted['behavioral']

# Get coding scores for main features
features = processing.get_feature_labels_for_clustering()
metadata = ['cell_specimen_id', 'ophys_experiment_id', 'cre_line', 'experience_level']
results_melted = results_melted[features + metadata]

# Transform coding scores for plotting
results_melted = results_melted.melt(
    id_vars=metadata,
    value_vars=features,
    var_name='feature',
    value_name='coding_score'
)

# Take absolute value
results_melted['coding_score'] = np.abs(results_melted['coding_score'])


# ============================================================================
# Extended population analysis plots
# ============================================================================

# Plot dropout summary population CDF
stats_dropout_cdf = gvt.plot_dropout_summary_population_cdf(results_pivoted_filtered, run_params)

# Plot metric and feature coding relationships
# Prepare merged metrics table for visualization
metric_data = results_pivoted_filtered.copy()
metric_data['images'] = metric_data['all-images']
metric_data['behavior'] = metric_data['behavioral']

# Plot relationships between metrics and feature coding
x = 'images'
y = 'all-images'
xlabel = 'Image coding score'
ylabel = 'Image coding score'
title = 'all cells'

plotting.plot_metric_feature_coding_relationship(
    metric_data, x, y, xlabel=xlabel, ylabel=ylabel, title=title,
    color_by_cre=False, save_dir=save_dir
)


print(f"Analysis complete. Supplemental figures generated in {save_dir}")
