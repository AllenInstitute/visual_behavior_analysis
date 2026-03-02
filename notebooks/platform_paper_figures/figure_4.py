"""
Paper Figure 4 - Cell-type specific coding and GLM analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.utils as utils

from visual_behavior.dimensionality_reduction.clustering import plotting
from visual_behavior.dimensionality_reduction.clustering import processing

import visual_behavior.visualization.ophys.glm_example_plots as gep
import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.platform_single_cell_examples as pse

import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_cell_metrics as gcm
import visual_behavior_glm.GLM_event_metrics as gem


# ============================================================================
# Configuration
# ============================================================================

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

experience_level_colors = utils.get_experience_level_colors()
cre_line_colors = utils.get_cre_line_colors()
experience_levels = utils.get_new_experience_levels()
cell_types = utils.get_cell_types()

glm_version = '24_events_all_L2_optimize_by_session'

save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'figure_4')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# ============================================================================
# Data Loading Section
# ============================================================================

# Metadata CSVs
metadata_dir = loading.get_metadata_tables_dir()
experiments_table = pd.read_csv(os.path.join(metadata_dir, 'all_ophys_experiments_table.csv'), index_col=0)
platform_experiments = pd.read_csv(os.path.join(metadata_dir, 'platform_paper_ophys_experiments_table.csv'), index_col=0)
platform_cells_table = pd.read_csv(os.path.join(metadata_dir, 'platform_paper_ophys_cells_table.csv'), index_col=0)
matched_cells_table = pd.read_csv(os.path.join(metadata_dir, 'platform_paper_matched_ophys_cells_table.csv'), index_col=0)

# GLM results loading
glm_results_dir = loading.get_glm_results_dir()
run_params = pd.read_pickle(os.path.join(glm_results_dir, glm_version + '_run_params.pkl'))
run_params['figure_dir'] = os.path.join(save_dir, 'coding_scores_and_kernels')

kernels = run_params['kernels']

all_weights_df = pd.read_hdf(os.path.join(glm_results_dir, 'weights_df.h5'), key='df')
all_results = pd.read_hdf(os.path.join(glm_results_dir, 'all_results.h5'), key='df')
all_results_pivoted = pd.read_hdf(os.path.join(glm_results_dir, 'results_pivoted.h5'), key='df')
all_results_pivoted = processing.set_negative_one_values_to_zero(all_results_pivoted)

# Split out cell_specimen_id and ophys_experiment_id for results table
all_results['cell_specimen_id'] = [int(identifier.split('_')[1]) for identifier in all_results.identifier.values]
all_results['ophys_experiment_id'] = [int(identifier.split('_')[0]) for identifier in all_results.identifier.values]

# Get results and weights just for platform paper experiments
results_pivoted = pd.read_hdf(os.path.join(glm_results_dir, 'platform_results_pivoted.h5'), key='df')
results_pivoted = processing.set_negative_one_values_to_zero(results_pivoted)
# Make coding scores positive
results_pivoted[processing.get_features_for_clustering()] = np.abs(results_pivoted[processing.get_features_for_clustering()])

weights_df = pd.read_hdf(os.path.join(glm_results_dir, 'platform_results_weights_df.h5'), key='df')

# Get lists of matched cells and experiments
matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()

# Get cre_lines and cell types for plot labels
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types_dict = utilities.get_cell_types_dict(cre_lines, platform_experiments)

# Load GLM example cell data for single cell examples
example_cells_dict, example_times_dict = gep.get_glm_example_cells_and_times()

# Load GLM model fit cell results
gep.get_glm_model_fit_cell_results_df(matched_experiments[0])

# Plot GLM example cells figure
gep.plot_glm_example_cells_figure(
    matched_experiments[0], run_params, all_results, results_pivoted, weights_df,
    use_var_exp=True, save_dir=save_dir
)

# Plotting utilities initialization
feature_colors, feature_labels_dict = plotting.get_feature_colors_and_labels()

# Helper functions for visualization
plotting.color_xaxis_labels_by_experience(plt.gca())
plotting.color_yaxis_labels_by_experience(plt.gca())
plotting.get_clean_labels_for_coding_scores_df(results_pivoted, columns=False)
plotting.get_fraction_cells_for_column(results_pivoted, column_to_group='cre_line')
plotting.remap_coding_scores_to_session_colors(results_pivoted[processing.get_features_for_clustering()])

# Plot kernel support visualization
gvt.plot_kernel_support(results_pivoted)


# ============================================================================
# Panel A: Example cell with GLM fit and behavior/physio timeseries
# ============================================================================

# Load behavior and physiology timeseries example
ophys_experiment_id = 808621034
dataset = loading.get_ophys_dataset(ophys_experiment_id)

ppf.plot_behavior_and_physio_timeseries_stacked(
    dataset, start_time=628, duration_seconds=20, save_dir=save_dir, ax=None
)


# ============================================================================
# Panel B: Kernel heatmaps with dropout by experience
# ============================================================================

run_params['fig_kernels_dir'] = os.path.join(save_dir, 'coding_scores_and_kernels')

# Plot kernel heatmaps for key kernels
kernels_to_plot = ['all-images', 'omissions', 'behavioral', 'task']

for kernel in kernels_to_plot:
    # Ensure kernel exists in run_params
    if kernel == 'all-images':
        run_params['kernels'][kernel] = run_params['kernels']['image0'].copy()

    time_vec = np.arange(
        run_params['kernels'][kernel]['offset'],
        run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],
        1 / 31.
    )
    time_vec = np.round(time_vec, 2)
    if 'image' in kernel:
        time_vec = time_vec[:-1]

    # Filter by cre lines
    sst_table = weights_df.query('cre_line == "Sst-IRES-Cre"')[[kernel + '_weights', kernel]]
    vip_table = weights_df.query('cre_line == "Vip-IRES-Cre"')[[kernel + '_weights', kernel]]
    slc_table = weights_df.query('cre_line == "Slc17a7-IRES2-Cre"')[[kernel + '_weights', kernel]]

    gvt.plot_kernel_heatmap_with_dropout_by_experience(
        vip_table, sst_table, slc_table,
        time_vec, kernel, run_params, ncells={}, ax=None, extra='', zlims=None,
        session_filter=['Familiar', 'Novel', 'Novel +'], savefig=True,
        cell_filter='', sort_by_dropout=False, limited=False
    )


# ============================================================================
# Panel C: Weights and coding score heatmaps for experience levels
# ============================================================================

for kernel in kernels_to_plot:
    xlabel = 'Time (s)' if kernel == 'all-images' else 'Time after event (s)'
    gep.plot_weights_and_coding_score_heatmaps_for_experience_levels(
        kernel, weights_df, run_params,
        row_condition='cre_line', col_condition='experience_level',
        vmax=0.004 if kernel == 'all-images' else 0.002,
        xlabel=xlabel,
        save_dir=save_dir, folder='kernel_weights'
    )


# ============================================================================
# Panel D: Variance explained by experience level
# ============================================================================

stats_D = gvt.var_explained_by_experience(results_pivoted, run_params)


# ============================================================================
# Panel E: Feature coding score metrics
# ============================================================================

stats_dropout = gvt.plot_dropout_summary_population(all_results, run_params)

# Compute event metrics
r2 = gcm.compute_event_metrics(results_pivoted, run_params)


# ============================================================================
# Panel F: Coding fractions by cell type and feature
# ============================================================================

var_explained_thresh = run_params['dropout_threshold']
coding_thresh = 0

# Filter to active sessions only
results_pivoted_active = results_pivoted.query('not passive').copy()

# Compute coding fractions
results_pivoted_active['code_anything'] = results_pivoted_active['variance_explained_full'] > run_params['dropout_threshold']
results_pivoted_active['code_images'] = results_pivoted_active['code_anything'] & (results_pivoted_active['all-images'] > coding_thresh)
results_pivoted_active['code_omissions'] = results_pivoted_active['code_anything'] & (results_pivoted_active['omissions'] > coding_thresh)
results_pivoted_active['code_behavioral'] = results_pivoted_active['code_anything'] & (results_pivoted_active['behavioral'] > coding_thresh)
results_pivoted_active['code_task'] = results_pivoted_active['code_anything'] & (results_pivoted_active['task'] > coding_thresh)

summary_df = results_pivoted_active.groupby(['cre_line', 'experience_level'])[
    ['code_anything', 'code_images', 'code_omissions', 'code_behavioral', 'code_task']
].mean()

# Compute confidence intervals
summary_df['n'] = results_pivoted_active.groupby(['cre_line', 'experience_level'])[
    ['code_anything', 'code_images', 'code_omissions', 'code_behavioral', 'code_task']
].count()['code_anything']

summary_df['code_images_ci'] = 1.96 * np.sqrt((summary_df['code_images'] * (1 - summary_df['code_images'])) / summary_df['n'])
summary_df['code_omissions_ci'] = 1.96 * np.sqrt((summary_df['code_omissions'] * (1 - summary_df['code_omissions'])) / summary_df['n'])
summary_df['code_behavioral_ci'] = 1.96 * np.sqrt((summary_df['code_behavioral'] * (1 - summary_df['code_behavioral'])) / summary_df['n'])
summary_df['code_task_ci'] = 1.96 * np.sqrt((summary_df['code_task'] * (1 - summary_df['code_task'])) / summary_df['n'])

# Create barplots for fraction of cells coding each feature
coding_groups = ['code_images', 'code_omissions', 'code_behavioral', 'code_task']
titles = ['images', 'omissions', 'behavioral', 'task']
colors = utils.get_cell_type_colors()

fig, ax = plt.subplots(1, len(coding_groups), figsize=(10.8, 4), sharey=True)
for index, feature in enumerate(coding_groups):
    # Plot three cre-lines in standard colors
    ax[index].plot([0, 1, 2], summary_df.loc['Vip-IRES-Cre'][feature], '-', color=colors[2], label='Vip Inhibitory', linewidth=3)
    ax[index].plot([0, 1, 2], summary_df.loc['Sst-IRES-Cre'][feature], '-', color=colors[1], label='Sst Inhibitory', linewidth=3)
    ax[index].plot([0, 1, 2], summary_df.loc['Slc17a7-IRES2-Cre'][feature], '-', color=colors[0], label='Excitatory', linewidth=3)

    # Add error bars
    ax[index].errorbar([0, 1, 2], summary_df.loc['Vip-IRES-Cre'][feature], yerr=summary_df.loc['Vip-IRES-Cre'][feature + '_ci'], color=colors[2], linewidth=3)
    ax[index].errorbar([0, 1, 2], summary_df.loc['Sst-IRES-Cre'][feature], yerr=summary_df.loc['Sst-IRES-Cre'][feature + '_ci'], color=colors[1], linewidth=3)
    ax[index].errorbar([0, 1, 2], summary_df.loc['Slc17a7-IRES2-Cre'][feature], yerr=summary_df.loc['Slc17a7-IRES2-Cre'][feature + '_ci'], color=colors[0], linewidth=3)

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

ax[0].set_ylabel('Fraction of cells \n coding for ', fontsize=20)
plt.tight_layout()
fig.savefig(os.path.join(save_dir, 'fraction_cells_coding_feature_by_experience.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================================
# Panel G: Barplot - fraction by cell type
# ============================================================================

var_explained_thresh = run_params['dropout_threshold']
coding_thresh = 0

# Compute fractions grouped by cell type
fractions_by_cell = results_pivoted_active.groupby(['cre_line'])[
    ['code_anything', 'code_images', 'code_omissions', 'code_behavioral', 'code_task']
].mean()

fractions_by_cell['n'] = results_pivoted_active.groupby(['cre_line'])[
    ['code_anything', 'code_images', 'code_omissions', 'code_behavioral', 'code_task']
].count()['code_anything']

fractions_by_cell['code_images_ci'] = 1.96 * np.sqrt((fractions_by_cell['code_images'] * (1 - fractions_by_cell['code_images'])) / fractions_by_cell['n'])
fractions_by_cell['code_omissions_ci'] = 1.96 * np.sqrt((fractions_by_cell['code_omissions'] * (1 - fractions_by_cell['code_omissions'])) / fractions_by_cell['n'])
fractions_by_cell['code_behavioral_ci'] = 1.96 * np.sqrt((fractions_by_cell['code_behavioral'] * (1 - fractions_by_cell['code_behavioral'])) / fractions_by_cell['n'])
fractions_by_cell['code_task_ci'] = 1.96 * np.sqrt((fractions_by_cell['code_task'] * (1 - fractions_by_cell['code_task'])) / fractions_by_cell['n'])

fractions = fractions_by_cell.copy()
fractions = fractions.drop(columns=['code_anything', 'n'])
fractions.columns = [column.split('_')[1] if 'ci' not in column else column for column in fractions.columns]
fractions = fractions.rename(columns={'behavioral': 'behavior'})
fractions = fractions.reset_index()
fractions = fractions.melt(id_vars=['cre_line'], var_name='feature', value_name='fraction')
fractions['cell_type'] = [utils.convert_cre_line_to_cell_type(cre_line) for cre_line in fractions.cre_line.values]

features = processing.get_feature_labels_for_clustering()
cell_type_colors = utils.get_cell_type_colors()
cell_types_list = np.sort(fractions.cell_type.unique())

figsize = (5, 2.5)
fig, ax = plt.subplots(figsize=figsize)
ax = sns.barplot(data=fractions, x='feature', y='fraction', order=features,
                 hue='cell_type', hue_order=cell_types_list, palette=cell_type_colors,
                 width=0.7, alpha=0.75, ax=ax)
ax.legend(fontsize='x-small')
ax.set_ylabel('Fraction cells\ncoding for feature')
ax.set_xlabel('')
ax.set_ylim(0, 1)
fig.subplots_adjust(hspace=0.4)
fig.savefig(os.path.join(save_dir, 'fraction_cells_coding_by_cell_type.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================================
# Panel H: Population averages by experience and cell type
# ============================================================================

stats_pop = gvt.plot_population_averages(
    results_pivoted_active, run_params, savefig=True,
    dropouts_to_show=['all-images', 'omissions', 'behavioral', 'task']
)


# ============================================================================
# Panel I: Population averages by area and depth
# ============================================================================

summary_area = gvt.plot_population_averages_by_area(
    results_pivoted_active, run_params,
    dropouts_to_show=['all-images', 'omissions', 'behavioral', 'task'],
    sharey=False, include_zero_cells=True, add_stats=True,
    extra='', equipment="mesoscope", savefig=True
)

summary_depth = gvt.plot_population_averages_by_depth(
    results_pivoted_active, run_params,
    dropouts_to_show=['all-images', 'omissions', 'behavioral', 'task'],
    sharey=False, include_zero_cells=True, add_stats=True,
    extra='', equipment="mesoscope", area=['VISp', 'VISl'], savefig=True
)


# ============================================================================
# GLM kernel visualization - Kernel evaluation by experience
# ============================================================================

# Kernel evaluation for all-images and omissions
gvt.kernel_evaluation(weights_df, run_params, 'all-images', session_filter=['Familiar'])
gvt.kernel_evaluation(weights_df, run_params, 'omissions', session_filter=['Familiar'])
gvt.kernel_evaluation_by_experience(weights_df, run_params, 'all-images', limited=False)
gvt.kernel_evaluation_by_experience(weights_df, run_params, 'omissions', limited=False)

# Kernel comparison across experience levels
gvt.plot_kernel_comparison_by_experience(weights_df, run_params, 'all-images')
gvt.plot_kernel_comparison_by_experience(weights_df, run_params, 'omissions')

# Plot fraction summary
stats_fractions = gvt.plot_fraction_summary_population(results_pivoted_active, run_params)

# Dropout summary with experience
stats_dropout_exp = gvt.plot_dropout_summary_population_with_experience(all_results, run_params)


# ============================================================================
# Single cell GLM examples
# ============================================================================

# Plot behavior timeseries with GLM kernel activations for example experiment
ophys_experiment_id_example = 967008471
dataset_example = loading.get_ophys_dataset(ophys_experiment_id_example, get_extended_stimulus_presentations=False)
start_time_example = 2318
duration_seconds_example = 19.5

gep.plot_behavior_timeseries_and_GLM_kernel_activations(
    dataset_example, start_time_example, duration_seconds_example, save_dir=save_dir
)

# Plot GLM methods with kernels and model fits for example cells
cell_index = 0
expt_dropouts = results_pivoted[results_pivoted.ophys_experiment_id == ophys_experiment_id_example]
cell_specimen_id_1 = expt_dropouts.sort_values(by='variance_explained_full', ascending=False).cell_specimen_id.values[cell_index]
cell_specimen_id_2 = expt_dropouts.sort_values(by='behavioral', ascending=True).cell_specimen_id.values[cell_index]

gep.plot_glm_methods_with_example_cells(
    ophys_experiment_id_example, cell_specimen_id_1, cell_specimen_id_2,
    weights_df, results_pivoted, kernels, save_dir=save_dir
)

# Plot individual example cells with model fits, kernels, and coding scores
for i, ophys_experiment_id in enumerate(list(example_cells_dict.keys())):
    start_time = list(example_times_dict.values())[i][0]

    # Get dataset and GLM results
    dataset = loading.get_ophys_dataset(ophys_experiment_id)
    cell_results_df, expt_results, dropouts = gep.load_glm_model_fit_results(ophys_experiment_id)

    # Define time window parameters
    flashes_between_changes = 12
    four_flash_dur = 6 * 0.75
    xlim_seconds = [start_time, start_time + (flashes_between_changes * 0.75) + four_flash_dur]
    duration_seconds = (flashes_between_changes * 0.75) + four_flash_dur
    suffix = '_' + str(int(xlim_seconds[0]))

    # Get stimulus response dataframes for kernels
    frame_rate = 31
    image_sdf, omission_sdf, change_sdf = gep.get_stimulus_response_dfs_for_kernel_windows(dataset, kernels, frame_rate)

    # Plot GLM features for time window
    gep.plot_glm_features_for_window(dataset, xlim_seconds, save_dir=save_dir, ax=None, suffix=suffix)

    # Plot for each example cell in this experiment
    for cell_specimen_id in list(example_cells_dict.values())[i]:

        # Plot model fits with behavior
        gep.plot_single_cell_example_model_fits_and_behavior(
            dataset, start_time, duration_seconds, [cell_specimen_id],
            cell_results_df, dropouts, expt_results, fontsize=8,
            kernel='all-images', label_csids=True, linewidth=1,
            short_title=False, include_events=True, save_dir=save_dir,
            folder='glm_example_cells', ax=None, suffix=suffix
        )

        # Plot stacked kernels for cell
        gep.plot_stacked_kernels_for_cell(
            cell_specimen_id, dataset, weights_df, kernels,
            save_dir=save_dir, folder='glm_example_cells'
        )

        # Plot coding score components
        gep.plot_coding_score_components_for_cell(
            cell_specimen_id, ophys_experiment_id, results_pivoted, dataset,
            fontsize=14, as_panel=False, save_dir=save_dir, folder='glm_example_cells'
        )

print(f"Figure 4 complete. Outputs saved to {save_dir}")


# ============================================================================
# COMPOSITE FIGURE 4: All panels on a single figure axis
# ============================================================================
# Layout based on published Figure 4 (Encoding model - stimulus and behavior):
#   Row 1 (~45%): A (GLM features/schematic, left ~30%) | B (kernel examples, mid ~30%) |
#                 E (% cells coding barplot, upper right ~20%) | F (coding score distributions, lower right ~40%)
#   Row 2 (~45%): C (model fit example, left ~35%) | D (coding scores bar, mid ~30%) | F cont.
#
# Panels A, B, C are complex multi-panel figures generated by specialized GLM
# visualization functions — left as labeled placeholders. Panels D, E, F are
# data-driven and plotted directly where possible.

def plot_figure_4_composite(results_pivoted_active, summary_df,
                            experience_levels, coding_groups, titles,
                            save_dir=None, folder='composite_figures'):
    """
    Generate a composite Figure 4 with all panels arranged in the published layout.
    Saved as PDF for Illustrator editing.
    """
    figsize = (24, 22)
    fig = plt.figure(figsize=figsize, facecolor='white')

    cell_type_colors = utils.get_cell_type_colors()

    # ---- Panel A: GLM features schematic (placeholder) ----
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.30], yspan=[0, 0.40])
    ax_A.text(0.5, 0.5, 'A — GLM features\nschematic (place manually)',
              transform=ax_A.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    # ---- Panel B: Kernel examples (placeholder) ----
    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.33, 0.62], yspan=[0, 0.40])
    ax_B.text(0.5, 0.5, 'B — Example kernels\n(place manually)',
              transform=ax_B.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_B.axis('off')

    # ---- Panel C: Model fit example (placeholder) ----
    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.30], yspan=[0.44, 0.72])
    ax_C.text(0.5, 0.5, 'C — Model fit example\n(place manually)',
              transform=ax_C.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    # ---- Panel D: Coding scores bar chart (placeholder - generated by gvt) ----
    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.33, 0.62], yspan=[0.44, 0.72])
    ax_D.text(0.5, 0.5, 'D — Coding scores\n(place manually)',
              transform=ax_D.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    # ---- Panel E: % cells coding for each feature by cell type (barplot) ----
    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.66, 1.0], yspan=[0, 0.28])
    try:
        fractions_by_cell = results_pivoted_active.groupby(['cre_line'])[
            ['code_anything', 'code_images', 'code_omissions', 'code_behavioral', 'code_task']
        ].mean()
        fractions = fractions_by_cell.copy()
        fractions = fractions.drop(columns=['code_anything'])
        fractions.columns = [column.split('_')[1] for column in fractions.columns]
        fractions = fractions.rename(columns={'behavioral': 'behavior'})
        fractions = fractions.reset_index()
        fractions = fractions.melt(id_vars=['cre_line'], var_name='feature', value_name='fraction')
        fractions['cell_type'] = [utils.convert_cre_line_to_cell_type(cre_line)
                                   for cre_line in fractions.cre_line.values]
        features_list = processing.get_feature_labels_for_clustering()
        cell_types_list = np.sort(fractions.cell_type.unique())
        sns.barplot(data=fractions, x='feature', y='fraction', order=features_list,
                     hue='cell_type', hue_order=cell_types_list, palette=cell_type_colors,
                     width=0.7, alpha=0.75, ax=ax_E)
        ax_E.legend(fontsize='x-small')
        ax_E.set_ylabel('Fraction cells\ncoding for feature')
        ax_E.set_xlabel('')
        ax_E.set_ylim(0, 1)
    except Exception:
        ax_E.text(0.5, 0.5, 'E — % cells coding\n(see individual panel)',
                  transform=ax_E.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax_E.set_title('E', loc='left', fontweight='bold', fontsize=16)

    # ---- Panel F: Coding score distributions by experience × cell type ----
    ax_F = utils.placeAxesOnGrid(fig, dim=[1, 4], xspan=[0.66, 1.0], yspan=[0.34, 0.72],
                                  wspace=0.3, sharey=True)
    try:
        for index, feature in enumerate(coding_groups):
            ax_F[index].plot([0, 1, 2], summary_df.loc['Vip-IRES-Cre'][feature], '-',
                             color=cell_type_colors[2], label='Vip Inhibitory', linewidth=3)
            ax_F[index].plot([0, 1, 2], summary_df.loc['Sst-IRES-Cre'][feature], '-',
                             color=cell_type_colors[1], label='Sst Inhibitory', linewidth=3)
            ax_F[index].plot([0, 1, 2], summary_df.loc['Slc17a7-IRES2-Cre'][feature], '-',
                             color=cell_type_colors[0], label='Excitatory', linewidth=3)
            ax_F[index].errorbar([0, 1, 2], summary_df.loc['Vip-IRES-Cre'][feature],
                                  yerr=summary_df.loc['Vip-IRES-Cre'][feature + '_ci'],
                                  color=cell_type_colors[2], linewidth=3)
            ax_F[index].errorbar([0, 1, 2], summary_df.loc['Sst-IRES-Cre'][feature],
                                  yerr=summary_df.loc['Sst-IRES-Cre'][feature + '_ci'],
                                  color=cell_type_colors[1], linewidth=3)
            ax_F[index].errorbar([0, 1, 2], summary_df.loc['Slc17a7-IRES2-Cre'][feature],
                                  yerr=summary_df.loc['Slc17a7-IRES2-Cre'][feature + '_ci'],
                                  color=cell_type_colors[0], linewidth=3)
            ax_F[index].set_title(titles[index], fontsize=12)
            ax_F[index].set_xticks([0, 1, 2])
            ax_F[index].set_xticklabels(experience_levels, rotation=90, fontsize=9)
            ax_F[index].spines['top'].set_visible(False)
            ax_F[index].spines['right'].set_visible(False)
            ax_F[index].set_xlim(-0.5, 2.5)
            ax_F[index].set_ylim(bottom=0)
            if index == 3:
                ax_F[index].legend(fontsize='x-small')
        ax_F[0].set_ylabel('Fraction of cells\ncoding for', fontsize=12)
        ax_F[0].set_title('F  ' + titles[0], loc='left', fontweight='bold', fontsize=14)
    except Exception:
        pass

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_4_composite', formats=['.pdf'])
    return fig


# Generate composite figure
coding_groups = ['code_images', 'code_omissions', 'code_behavioral', 'code_task']
titles = ['images', 'omissions', 'behavioral', 'task']

fig4 = plot_figure_4_composite(
    results_pivoted_active, summary_df,
    experience_levels, coding_groups, titles,
    save_dir=save_dir, folder='composite_figures'
)
