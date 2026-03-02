"""
Paper Figure 3 - Population neural responses across experience levels

This script generates Figure 3 from the platform paper, showing population neural
responses to image changes and omissions across experience levels (Familiar, Novel, Novel+),
including single cell examples, population averages, response heatmaps, fraction of
responsive cells, and experience modulation indices.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

import visual_behavior.data_access.utilities as utilities
from visual_behavior.data_access import loading as loading

import visual_behavior.visualization.utils as utils
import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.platform_single_cell_examples as pse
import visual_behavior.ophys.response_analysis.cell_metrics as cm
import visual_behavior.dimensionality_reduction.clustering.processing as processing

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

# ===== Configuration =====
save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'figure_3')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

palette = utils.get_experience_level_colors()
experience_levels = utils.get_new_experience_levels()
experience_level_colors = utils.get_experience_level_colors()
cell_types = utils.get_cell_types()

# ===== Data Loading =====

# Get cache and metadata tables
metadata_dir = loading.get_metadata_tables_dir()
cache_dir = loading.get_platform_analysis_cache_dir()
cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=cache_dir, use_static_cache=True)

# Load metadata CSV files
experiments_table = pd.read_csv(os.path.join(metadata_dir, 'all_ophys_experiments_table.csv'), index_col=0)
platform_experiments = pd.read_csv(os.path.join(metadata_dir, 'platform_paper_ophys_experiments_table.csv'), index_col=0)
platform_cells_table = pd.read_csv(os.path.join(metadata_dir, 'platform_paper_ophys_cells_table.csv'), index_col=0)
matched_cells_table = pd.read_csv(os.path.join(metadata_dir, 'platform_paper_matched_ophys_cells_table.csv'), index_col=0)

# Get lists of matched cells and expts
matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()

# Get cre_lines and cell types for plot labels
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types_dict = utilities.get_cell_types_dict(cre_lines, platform_experiments)

# ===== Load response metrics =====

data_type = 'filtered_events'
session_subset = 'full_session'
inclusion_criteria = 'platform_experiment_table'

filtered_events_metrics_table = processing.generate_merged_table_of_model_free_metrics(
    data_type=data_type, session_subset=session_subset,
    inclusion_criteria=inclusion_criteria, save_dir=metadata_dir)

metrics_table = filtered_events_metrics_table.copy()

# Merge with platform experiments
platform_experiments_temp = loading.get_platform_paper_experiment_table(limit_to_closest_active=True)
metrics_table = metrics_table.merge(platform_experiments_temp.reset_index(),
                                   on=['ophys_experiment_id', 'experience_level'])

# ===== Load multi-session dataframes =====

data_type = 'events'
interpolate = True
output_sampling_rate = 30
inclusion_criteria = 'platform_experiment_table'

# Image changes
event_type = 'all'
conditions = ['cell_specimen_id', 'is_change']
change_mdf = loading.get_multi_session_df_for_conditions(
    data_type, event_type, conditions, inclusion_criteria,
    interpolate=interpolate, output_sampling_rate=output_sampling_rate,
    epoch_duration_mins=None)

change_mdf = change_mdf[change_mdf.is_change == True]
image_mdf = change_mdf[change_mdf.is_change == False]

# Omissions
conditions = ['cell_specimen_id', 'omitted']
omission_mdf = loading.get_multi_session_df_for_conditions(
    data_type, event_type, conditions, inclusion_criteria,
    interpolate=interpolate, output_sampling_rate=output_sampling_rate,
    epoch_duration_mins=None)

omission_mdf = omission_mdf[omission_mdf.omitted == True]

# Per image
conditions = ['cell_specimen_id', 'is_change', 'image_name']
each_image_mdf = loading.get_multi_session_df_for_conditions(
    data_type, event_type, conditions, inclusion_criteria,
    interpolate=interpolate, output_sampling_rate=output_sampling_rate,
    epoch_duration_mins=None)

# Load dataset dict for single cell plotting
ophys_container_ids = matched_cells_table.ophys_container_id.unique()[:4]
dataset_dict = loading.get_dataset_dict_for_containers(ophys_container_ids,
                                                       platform_experiments,
                                                       dataset_dict=None)

# ===== PANEL A: Single cell examples =====

# Example container and cells for matched cell plotting
ophys_container_id = 1018027775
cell_type = 'Excitatory'

# Get high SNR matched cells for container
matched_cells_snr, dff_metrics = pse.get_high_snr_matched_cells_for_container(ophys_container_id, dataset_dict,
                                                                               matched_cells_table,
                                                                               snr_threshold=0.5)

# Plot matched max projections for container
pse.plot_matched_max_projections_for_container(ophys_container_id, platform_experiments,
                                              matched_cells_table, dataset_dict,
                                              cells_to_label=matched_cells_snr, save_dir=save_dir, ax=None)

# Plot single cell example timeseries and behavior for a selected start time
start_time = 513
pse.plot_single_cell_example_timeseries_and_behavior(dataset_dict[ophys_container_id][
                                                     platform_experiments[platform_experiments.ophys_container_id == ophys_container_id].index.values[0]],
                                                    start_time=start_time, duration_seconds=18,
                                                    save_dir=save_dir, ax=None)

# Plot matched ROI outlines for container
pse.plot_matched_roi_outlines_for_container(ophys_container_id, platform_experiments,
                                            matched_cells_table, dataset_dict,
                                            cells_to_label=None, save_dir=save_dir, ax=None)

# Plot matched ROIs and traces for container
start_times = np.asarray([513])
matched_cells_example = matched_cells_table[matched_cells_table.ophys_container_id == ophys_container_id]
matched_cells_example = matched_cells_example.sort_values(by='cell_specimen_id')
matched_cells_example = matched_cells_example.drop_duplicates(subset='cell_specimen_id')
matched_cells_csids = matched_cells_example.cell_specimen_id.values

pse.plot_matched_rois_and_traces_for_container(ophys_container_id, dataset_dict,
                                               matched_cells_table, platform_experiments, fontsize=12,
                                               start_times=start_times, duration_seconds=18,
                                               matched_cells=matched_cells_csids[:5], save_dir=save_dir, suffix='')

# Plot matched traces across sessions
pse.plot_matched_traces_across_sessions(ophys_container_id, dataset_dict,
                                        matched_cells_table, platform_experiments,
                                        change_mdf, omission_mdf, cell_type, skip_behavior=True,
                                        start_times=start_times, duration_seconds=18,
                                        matched_cells=matched_cells_csids[:5], save_dir=save_dir, suffix='')

# Plot reliable example cells - changes
Exc_change_cells = [1086579153, 1086559786, 1086523675, 1086514966, 1086514622, 1120104688,
                    1086558573, 1086527750, 1086517398, 1086577182, 1120108338, 1086510825]

pse.plot_reliable_example_cells(change_mdf, Exc_change_cells[:4], cell_type, event_type='changes',
                                save_dir=save_dir, folder='single_cell_examples', suffix='_selected')

# Plot reliable example cells - omissions
Exc_omission_cells = [1086529704, 1120092455, 1120102262, 1086571711, 1086514606,
                      1086530117, 1120090879, 1086526518]

pse.plot_reliable_example_cells(omission_mdf, Exc_omission_cells[:4], cell_type, event_type='omissions',
                                save_dir=save_dir, folder='single_cell_examples', suffix='_selected')

# ===== PANEL B: Population averages for image changes =====

df = change_mdf.copy()
df = df[df.experience_level.isin(['Familiar', 'Novel'])]

axes_column = 'cell_type'
hue_column = 'experience_level'
xlim_seconds = [-0.25, 0.75]

ax = ppf.plot_population_averages_for_conditions(
    df, data_type, 'changes', axes_column, hue_column,
    horizontal=True, xlabel='Time from image change (s)',
    xlim_seconds=xlim_seconds, interval_sec=0.5, legend=False,
    ylabel='Calcium events', palette=palette, ax=None, title=None, suptitle=None,
    save_dir=save_dir, folder='population_activity', suffix='_changes_F_N')

# ===== PANEL C: Population averages for omissions =====

df = omission_mdf.copy()
df = df[df.experience_level.isin(['Familiar', 'Novel'])]

axes_column = 'cell_type'
hue_column = 'experience_level'
xlim_seconds = [-1., 1.5]

ax = ppf.plot_population_averages_for_conditions(
    df, data_type, 'omissions', axes_column, hue_column,
    horizontal=True, xlabel='Time from omission (s)',
    xlim_seconds=xlim_seconds, interval_sec=1, legend=False,
    ylabel='Calcium events', palette=palette, ax=None, title=None, suptitle=None,
    save_dir=save_dir, folder='population_activity', suffix='_omissions')

# ===== PANEL D: Population averages by cell type across experience =====

df = change_mdf.copy()
xlim_seconds = [-1., 1.5]

ppf.plot_population_averages_for_cell_types_across_experience(
    df, xlim_seconds=xlim_seconds, xlabel='Time from image change (s)',
    ylabel='Calcium events', data_type=data_type, event_type='changes',
    interval_sec=0.5, save_dir=save_dir, folder='population_activity',
    suffix='_changes_stacked', ax=None)

# ===== PANEL E: Response heatmaps for image changes =====

df = change_mdf.copy()
xlim_seconds = [-1., 1.5]

timestamps = df.trace_timestamps.values[0]
row_condition = 'cell_type'
col_condition = 'experience_level'
cols_to_sort_by = ['mean_response']

ppf.plot_response_heatmaps_for_conditions(
    df, timestamps, data_type, 'all', row_condition, col_condition,
    cols_to_sort_by=cols_to_sort_by, suptitle='',
    xlabel='Time from image change (s)',
    microscope=None, vmax=None, xlim_seconds=xlim_seconds,
    match_cells=False, cbar=False,
    save_dir=save_dir, folder='population_activity', suffix='_changes_heatmap', ax=None)

# ===== PANEL F: Fraction of responsive cells =====

folder = 'response_metrics'

# Get single cell example for display
cell_specimen_id = matched_cells[0]
cell_metadata = matched_cells_table[matched_cells_table.cell_specimen_id == cell_specimen_id]

# Plot matched ROI and trace for example cell
ppf.plot_matched_roi_and_trace(cell_metadata.ophys_container_id.values[0], cell_specimen_id,
                               limit_to_last_familiar_second_novel=True,
                               use_events=True, filter_events=False, save_figure=False)

# Plot matched ROI and traces example for cell with omissions
ppf.plot_matched_roi_and_traces_example(cell_metadata, include_omissions=True,
                                        use_events=True, filter_events=False, save_dir=save_dir,
                                        folder='single_cell_examples')

# Sort cells by response in a time window
xlim_seconds = [-1., 1.5]
timestamps = change_mdf.trace_timestamps.values[0]
dff_traces = change_mdf[change_mdf.cell_specimen_id == cell_specimen_id].dff_trace.values[0]
csids_sorted = ppf.sort_trace_csids_by_max_in_window(change_mdf, timestamps, xlim_seconds)

# Plot fraction of responsive cells (original version, not percent)
df = each_image_mdf[each_image_mdf.pref_stim == True]
df = df[df.is_change == True]

ppf.plot_fraction_responsive_cells(
    df, responsiveness_threshold=0.1, ylim=(-0.02, 1),
    ylabel='Fraction of image change responsive cells',
    save_dir=save_dir, folder=folder, suffix='_changes_pref_stim',
    horizontal=False, ax=None)

# ===== PANEL G: Experience modulation index for image changes =====

metric = 'mean_response'
event_type = 'changes'
xlabel = 'Experience modulation'

# Plot non-annotated version
ppf.plot_experience_modulation_index(
    change_mdf, event_type, plot_type='violinplot', ylims=(-1.1, 1.1),
    include_all_comparisons=False, save_dir=save_dir, folder='response_metrics',
    suffix='_changes')

# Plot annotated version
ppf.plot_experience_modulation_index_annotated(
    change_mdf, event_type, metric, platform_cells_table,
    xlabel=xlabel, xlims=(-1.1, 1.1), horiz=True,
    suptitle=None, suffix='', save_dir=save_dir, ax=None)

# ===== PANEL H: Experience modulation index for omissions =====

metric = 'mean_response'
event_type = 'omissions'
xlabel = 'Experience modulation'

ppf.plot_experience_modulation_index_annotated(
    omission_mdf, event_type, metric, platform_cells_table,
    xlabel=xlabel, xlims=(-1.1, 1.1), horiz=True,
    suptitle=None, suffix='', save_dir=save_dir, ax=None)

# ===== PANEL I: Mean response by epoch =====

# Load epoch data for image changes
data_type_epoch = 'events'
conditions_epochs = ['cell_specimen_id', 'is_change', 'epoch']
change_mdf_epochs = loading.get_multi_session_df_for_conditions(
    data_type_epoch, 'all', conditions_epochs, inclusion_criteria,
    interpolate=interpolate, output_sampling_rate=output_sampling_rate,
    epoch_duration_mins=5)

change_mdf_epochs = change_mdf_epochs[change_mdf_epochs.is_change == True]

ppf.plot_mean_response_by_epoch(
    change_mdf_epochs, metric='mean_response', horizontal=True, ymin=0,
    ylabel='Mean change response', estimator=np.mean,
    save_dir=save_dir, folder='epochs', max_epoch=11, suptitle=None,
    suffix='_changes_within_session', ax=None)

# ===== PANEL J: Metric distribution without cell type breakdown =====

ppf.plot_metric_distribution_by_experience_no_cell_type(
    metrics_table, 'mean_response', event_type='changes', data_type=data_type,
    ylabel='Mean change response', ylims=None, save_dir=save_dir, ax=None)

print("Figure 3 generation complete. Figures saved to:", save_dir)


# ============================================================================
# COMPOSITE FIGURE 3: All panels on a single figure axis
# ============================================================================
# Layout based on published Figure 3 (Stimulus novelty alters image change and omission activity):
#   Row 1 (~15%): Panel A (matched ROIs + traces across sessions, full width)
#   Row 2 (~15%): Panels B,C,D (pop avg changes: Exc, Sst, Vip) | Panel E (% change responsive)
#   Row 3 (~18%): Panels F,G,H (single cell change examples: Exc, Sst, Vip)
#   Row 4 (~15%): Panels I,J,K (pop avg omissions: Exc, Sst, Vip) | Panel L (% omission responsive)
#   Row 5 (~15%): Panel M (Vip omission single cell examples)
#
# Panel A and single cell examples (F-H, M) are complex multi-panel figures
# generated by specialized functions — left as labeled placeholders.

def plot_figure_3_composite(change_mdf, omission_mdf, each_image_mdf,
                            platform_cells_table,
                            data_type, palette,
                            save_dir=None, folder='composite_figures'):
    """
    Generate a composite Figure 3 with all panels arranged in the published layout.
    Saved as PDF for Illustrator editing.
    """
    figsize = (24, 32)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # ---- Row 1: Panel A (matched ROIs + traces, full width placeholder) ----
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1.0], yspan=[0, 0.12])
    ax_A.text(0.5, 0.5, 'A — Matched ROIs and traces across sessions\n(place manually)',
              transform=ax_A.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    # ---- Row 2: Panels B,C,D (pop avg changes) + Panel E (% responsive) ----
    ax_BCD = utils.placeAxesOnGrid(fig, dim=[1, 3], xspan=[0, 0.72], yspan=[0.15, 0.28],
                                    wspace=0.3, sharey=True)
    df_changes = change_mdf.copy()
    ppf.plot_population_averages_for_conditions(
        df_changes, data_type, 'changes', 'cell_type', 'experience_level',
        horizontal=True, xlabel='Time from image change (s)',
        xlim_seconds=[-0.25, 0.75], interval_sec=0.5, legend=False,
        ylabel='Calcium events', palette=palette, ax=ax_BCD, title=None, suptitle=None,
        save_dir=None, folder=None, suffix=None)
    if isinstance(ax_BCD, list):
        ax_BCD[0].set_title('B', loc='left', fontweight='bold', fontsize=16)
        if len(ax_BCD) > 1:
            ax_BCD[1].set_title('C', loc='left', fontweight='bold', fontsize=14)
        if len(ax_BCD) > 2:
            ax_BCD[2].set_title('D', loc='left', fontweight='bold', fontsize=14)

    # Panel E: Fraction of change responsive cells
    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.77, 1.0], yspan=[0.15, 0.28])
    try:
        df_E = each_image_mdf[each_image_mdf.pref_stim == True]
        df_E = df_E[df_E.is_change == True]
        ppf.plot_fraction_responsive_cells(
            df_E, responsiveness_threshold=0.1, ylim=(-0.02, 1),
            ylabel='Fraction responsive',
            save_dir=None, folder=None, suffix=None,
            horizontal=False, ax=ax_E)
    except Exception:
        pass
    ax_E.set_title('E', loc='left', fontweight='bold', fontsize=16)

    # ---- Row 3: Panels F,G,H (single cell change examples - placeholders) ----
    ax_F = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.30], yspan=[0.32, 0.50])
    ax_F.text(0.5, 0.5, 'F — Excitatory\nsingle cell examples\n(place manually)',
              transform=ax_F.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax_F.set_title('F', loc='left', fontweight='bold', fontsize=16)
    ax_F.axis('off')

    ax_G = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.33, 0.63], yspan=[0.32, 0.50])
    ax_G.text(0.5, 0.5, 'G — Sst Inhibitory\nsingle cell examples\n(place manually)',
              transform=ax_G.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax_G.set_title('G', loc='left', fontweight='bold', fontsize=16)
    ax_G.axis('off')

    ax_H = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.66, 1.0], yspan=[0.32, 0.50])
    ax_H.text(0.5, 0.5, 'H — Vip Inhibitory\nsingle cell examples\n(place manually)',
              transform=ax_H.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax_H.set_title('H', loc='left', fontweight='bold', fontsize=16)
    ax_H.axis('off')

    # ---- Row 4: Panels I,J,K (pop avg omissions) + Panel L (% omission responsive) ----
    ax_IJK = utils.placeAxesOnGrid(fig, dim=[1, 3], xspan=[0, 0.72], yspan=[0.54, 0.67],
                                    wspace=0.3, sharey=True)
    df_omissions = omission_mdf.copy()
    ppf.plot_population_averages_for_conditions(
        df_omissions, data_type, 'omissions', 'cell_type', 'experience_level',
        horizontal=True, xlabel='Time from omission (s)',
        xlim_seconds=[-1., 1.5], interval_sec=1, legend=False,
        ylabel='Calcium events', palette=palette, ax=ax_IJK, title=None, suptitle=None,
        save_dir=None, folder=None, suffix=None)
    if isinstance(ax_IJK, list):
        ax_IJK[0].set_title('I', loc='left', fontweight='bold', fontsize=16)
        if len(ax_IJK) > 1:
            ax_IJK[1].set_title('J', loc='left', fontweight='bold', fontsize=14)
        if len(ax_IJK) > 2:
            ax_IJK[2].set_title('K', loc='left', fontweight='bold', fontsize=14)

    # Panel L: Fraction of omission responsive cells
    ax_L = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.77, 1.0], yspan=[0.54, 0.67])
    ax_L.text(0.5, 0.5, 'L — % omission\nresponsive cells',
              transform=ax_L.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax_L.set_title('L', loc='left', fontweight='bold', fontsize=16)
    ax_L.axis('off')

    # ---- Row 5: Panel M (Vip omission single cell examples) ----
    ax_M = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.55, 1.0], yspan=[0.72, 0.90])
    ax_M.text(0.5, 0.5, 'M — Vip omission\nsingle cell examples\n(place manually)',
              transform=ax_M.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax_M.set_title('M', loc='left', fontweight='bold', fontsize=16)
    ax_M.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_3_composite', formats=['.pdf'])
    return fig


# Generate composite figure
fig3 = plot_figure_3_composite(
    change_mdf, omission_mdf, each_image_mdf,
    platform_cells_table,
    data_type='filtered_events', palette=palette,
    save_dir=save_dir, folder='composite_figures'
)
