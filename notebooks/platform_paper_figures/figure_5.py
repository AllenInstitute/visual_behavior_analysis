"""
Paper Figure 5 - Functional clustering of neural response types

This script generates Figure 5 showing functional clustering of neural response types.
It includes UMAP visualization, cluster response profiles, coding score heatmaps,
population average responses, response metrics by cell type, and experience modulation analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

# Visual behavior analysis package functions
import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities

# Clustering modules
from visual_behavior.dimensionality_reduction.clustering import plotting
from visual_behavior.dimensionality_reduction.clustering import processing

# Platform paper figure functions
import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.platform_single_cell_examples as pse

# ============================================================================
# CONFIGURATION
# ============================================================================

# Clustering configuration
n_clusters = 14
threshold_percentile = 99.8  # for removing outliers

# Data type configuration
data_type = 'events'
interpolate = True
output_sampling_rate = 30
inclusion_criteria = 'platform_experiment_table'

# Figure saving
save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'figure_5')
os.makedirs(save_dir, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

# Get cache directories
platform_cache_dir = loading.get_platform_analysis_cache_dir()
glm_results_dir = os.path.join(platform_cache_dir, 'glm_results')
clustering_dir = os.path.join(platform_cache_dir, 'clustering')

# Load metadata tables
experiments_table = pd.read_csv(
    os.path.join(platform_cache_dir, 'all_ophys_experiments_table.csv'),
    index_col=0
)
platform_experiments = pd.read_csv(
    os.path.join(platform_cache_dir, 'platform_paper_ophys_experiments_table.csv'),
    index_col=0
)
platform_cells_table = pd.read_csv(
    os.path.join(platform_cache_dir, 'platform_paper_ophys_cells_table.csv'),
    index_col=0
)
matched_cells_table = pd.read_csv(
    os.path.join(platform_cache_dir, 'platform_paper_matched_ophys_cells_table.csv'),
    index_col=0
)

# Get matched cells and experiments
matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()

# Get utility information
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types = utils.get_cell_types()
experience_level_colors = utils.get_experience_level_colors()
cre_line_colors = utils.get_cre_line_colors()

# Load GLM results
glm_version = '24_events_all_L2_optimize_by_session'
results_pivoted = pd.read_hdf(
    os.path.join(glm_results_dir, 'across_session_normalized_platform_results_pivoted.h5'),
    key='df'
)
all_results = pd.read_hdf(
    os.path.join(glm_results_dir, 'all_results.h5'),
    key='df'
)

# Load clustering results
feature_matrix = pd.read_hdf(
    os.path.join(clustering_dir, 'clustering_feature_matrix.h5'),
    key='df'
)
feature_matrix = feature_matrix.rename(columns={'all-images': 'images', 'behavioral': 'behavior'})

cluster_meta = pd.read_hdf(
    os.path.join(clustering_dir, f'cluster_metadata_{n_clusters}_clusters.h5'),
    key='df'
)

# Load UMAP results
umap_df = processing.get_umap_results(feature_matrix, cluster_meta, save_dir=clustering_dir, suffix='')

# ============================================================================
# DATA PROCESSING: LOAD MULTI-SESSION DATAFRAMES
# ============================================================================

# Image change and repeated image responses
conditions = ['cell_specimen_id', 'is_change']
multi_session_df = loading.get_multi_session_df_for_conditions(
    data_type,
    event_type='all',
    conditions=conditions,
    inclusion_criteria=inclusion_criteria,
    interpolate=interpolate,
    output_sampling_rate=output_sampling_rate,
    epoch_duration_mins=None
)

# Separate into change and non-change responses
change_mdf = multi_session_df[multi_session_df.is_change == True]
change_mdf = change_mdf.merge(
    cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']],
    on='cell_specimen_id'
)

image_mdf = multi_session_df[multi_session_df.is_change == False]
image_mdf = image_mdf.merge(
    cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']],
    on='cell_specimen_id'
)

# Omission responses
conditions = ['cell_specimen_id', 'omitted']
omission_mdf = loading.get_multi_session_df_for_conditions(
    data_type,
    event_type='all',
    conditions=conditions,
    inclusion_criteria=inclusion_criteria,
    interpolate=interpolate,
    output_sampling_rate=output_sampling_rate,
    epoch_duration_mins=None
)

omission_mdf = omission_mdf[omission_mdf.omitted == True]
omission_mdf = omission_mdf.merge(
    cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']],
    on='cell_specimen_id'
)

# Remove outliers
image_mdf_clean, image_outliers = processing.remove_outliers(image_mdf, threshold_percentile)
change_mdf_clean, change_outliers = processing.remove_outliers(change_mdf, threshold_percentile)
omission_mdf_clean, omission_outliers = processing.remove_outliers(omission_mdf, threshold_percentile)

# ============================================================================
# PANEL A: UMAP colored by cluster
# ============================================================================

print("Generating Panel A: UMAP for clusters...")
plotting.plot_umap_for_clusters(
    cluster_meta,
    feature_matrix,
    umap_df,
    label_col='cluster_id',
    cre_lines=['all'],
    save_dir=save_dir,
    folder='figure_5_panels'
)

# ============================================================================
# PANEL B: Mean response profiles for each cluster
# ============================================================================

print("Generating Panel B: Cluster means...")
plotting.plot_cluster_means_remapped(
    feature_matrix,
    cluster_meta,
    save_dir=save_dir,
    folder='figure_5_panels'
)

# ============================================================================
# PANEL C: GLM coding scores per cluster (heatmap)
# ============================================================================

print("Generating Panel C: Coding score heatmap...")
cluster_meta_tmp = cluster_meta.copy()
cluster_meta_tmp['cell_index'] = cluster_meta_tmp.index.values
plotting.plot_coding_score_heatmap_remapped(
    cluster_meta_tmp,
    feature_matrix,
    sort_by=None,
    session_colors=True,
    save_dir=save_dir,
    folder='figure_5_panels'
)

# ============================================================================
# PANEL C1: Get cluster order and experience level colors
# ============================================================================

print("Generating cluster order and colors...")

# Get cluster order (for consistent ordering across plots)
cluster_order = plotting.get_cluster_order(cluster_meta)

# Get preference experience level colors for clusters
response_metrics = processing.generate_merged_table_of_coding_score_and_model_free_metrics(
    cluster_meta,
    results_pivoted,
    data_type='events',
    session_subset='full_session',
    inclusion_criteria='platform_experiment_table',
    save_dir=clustering_dir
)

# Replace dominant experience level cluster with correct one
cluster_metrics = processing.get_coding_score_metrics_for_clusters(cluster_meta, results_pivoted)
cluster_metrics = cluster_metrics.rename(columns={
    'dominant_experience_level': 'dominant_experience_level_cluster',
    'dominant_feature': 'dominant_feature_cluster'
})

response_metrics = response_metrics.rename(columns={
    'dominant_experience_level_cluster': 'dominant_experience_level_old',
    'dominant_feature_cluster': 'dominant_feature_old'
})

response_metrics = response_metrics.merge(
    cluster_metrics.reset_index()[['cluster_id', 'dominant_experience_level_cluster', 'dominant_feature_cluster']]
)

pref_experience_colors = plotting.get_pref_experience_level_colors_for_clusters(response_metrics)

# ============================================================================
# PANEL C2: Cell response heatmaps for clusters
# ============================================================================

print("Generating Panel C2: Cell response heatmaps...")

plotting.plot_cell_response_heatmaps_for_clusters(
    image_mdf,
    data_type='events',
    event_type='images',
    vmax=0.0005,
    xlim_seconds=[-0.3, 0.8],
    ax=None
)

# ============================================================================
# PANEL C3: Cluster properties combined
# ============================================================================

print("Generating Panel C3: Cluster properties combined...")

# All cre lines combined
tmp = cluster_meta.copy()
tmp['cre_line'] = 'all'

plotting.plot_cluster_properties_combined(
    tmp,
    feature_matrix,
    'all',
    image_mdf,
    change_mdf,
    omission_mdf,
    save_dir=save_dir
)

# By individual cre line
for cre_line in cre_lines:
    plotting.plot_cluster_properties_combined(
        cluster_meta,
        feature_matrix,
        cre_line,
        image_mdf,
        change_mdf,
        omission_mdf,
        save_dir=save_dir,
        sort_by='cluster_size'
    )

# ============================================================================
# PANEL D: Population average responses split by experience level
# ============================================================================

print("Generating Panel D: Population average responses...")

# Image responses
event_type = 'images'
plotting.plot_population_average_response_for_clusters_as_rows_split(
    image_mdf_clean,
    event_type,
    cluster_order=None,
    save_dir=save_dir,
    ax=None
)

# Change responses
event_type = 'changes'
plotting.plot_population_average_response_for_clusters_as_rows_split(
    change_mdf_clean,
    event_type,
    with_pre_change=True,
    cluster_order=None,
    save_dir=save_dir,
    ax=None
)

# Omission responses
event_type = 'omissions'
plotting.plot_population_average_response_for_clusters_as_rows_split(
    omission_mdf_clean,
    event_type,
    with_pre_change=True,
    cluster_order=None,
    save_dir=save_dir,
    ax=None
)

# ============================================================================
# PANEL E: Response metrics by Cre line
# ============================================================================

print("Generating Panel E: Response metrics...")

# Plot response metrics by Cre line
metric = 'lifetime_sparseness_images'
label = 'Image selectivity'

plotting.plot_response_metrics_boxplot_by_cre(
    response_metrics,
    metric,
    ylabel=label,
    horiz=True,
    line_val=0.5,
    save_dir=save_dir,
    folder='figure_5_panels'
)

plotting.plot_response_metrics_boxplot_by_cre_as_cols(
    response_metrics,
    metric,
    label=label,
    horiz=False,
    line_val=0.5,
    lims=(-0.1, 1.1),
    save_dir=save_dir,
    folder='figure_5_panels'
)

# ============================================================================
# PANEL F: Experience modulation across clusters
# ============================================================================

print("Generating Panel F: Experience modulation...")

# Generate coding score metrics
coding_score_metrics = processing.generate_coding_score_metrics_table(
    cluster_meta,
    results_pivoted,
    save_dir=clustering_dir
)

# Filter to clusters with sufficient cell counts per cre line
n_cells_per_cluster = processing.get_cluster_proportion_stats_for_cell_types(cluster_meta)
small_clusters = n_cells_per_cluster[
    n_cells_per_cluster.fraction_per_cluster < 0.09
].reset_index().set_index(['cre_line', 'cluster_id']).index.values
small_clusters = np.unique(small_clusters)

coding_score_metrics_filtered = coding_score_metrics.copy()
for small_cluster in small_clusters:
    csids = coding_score_metrics[
        (coding_score_metrics.cre_line == small_cluster[0]) &
        (coding_score_metrics.cluster_id == small_cluster[1])
    ].index.values
    coding_score_metrics_filtered = coding_score_metrics_filtered.drop(index=csids)

# Remove non-coding cluster
csids = coding_score_metrics_filtered[coding_score_metrics_filtered.cluster_id == 1].index.values
coding_score_metrics_filtered = coding_score_metrics_filtered.drop(index=csids)

# Plot experience modulation
plotting.plot_experience_modulation(
    coding_score_metrics_filtered,
    metric='experience_modulation',
    save_dir=save_dir,
    folder='figure_5_panels'
)

# ============================================================================
# PANEL G: Single cell examples
# ============================================================================

print("Generating Panel G: Single cell examples...")

# Get cell response dataframes for example cells
cell_specimen_id = matched_cells[0]  # Select first matched cell as example
cell_metadata = platform_cells_table[platform_cells_table.cell_specimen_id == cell_specimen_id]

# Load change and omission response dataframes for example cell
change_mdf_example = change_mdf_clean[change_mdf_clean.cell_specimen_id == cell_specimen_id]
omission_mdf_example = omission_mdf_clean[omission_mdf_clean.cell_specimen_id == cell_specimen_id]

# Plot matched ROI and coding scores
try:
    pse.plot_matched_roi_and_coding_scores(
        cell_metadata,
        platform_experiments,
        save_dir=save_dir
    )
except Exception as e:
    print(f"Warning: Could not generate matched ROI plot: {e}")

# Plot cell change and omission responses
try:
    pse.plot_cell_change_and_omission_responses(
        change_mdf_example,
        omission_mdf_example,
        cell_specimen_id,
        save_dir=save_dir
    )
except Exception as e:
    print(f"Warning: Could not generate cell response plot: {e}")

# ============================================================================
# SUPPLEMENTARY: Additional clustering visualizations
# ============================================================================

print("Generating supplementary clustering figures...")

# UMAP colored by cell type
cmap = utils.get_cell_type_colors()
plotting.plot_umap_for_features_separately(
    cluster_meta,
    feature_matrix,
    umap_df,
    label_col='cell_type',
    cmap=cmap,
    save_dir=save_dir,
    folder='supplementary'
)

# Cluster info (distribution of animals and rigs)
plotting.plot_cluster_info(
    cre_lines,
    cluster_meta,
    save_dir=save_dir,
    folder='supplementary'
)

# Percent cells per cluster
plotting.plot_percent_cells_per_cluster_all_cre(
    cluster_meta,
    col_to_group='cre_line',
    save_dir=save_dir
)

plotting.plot_percent_cells_per_cluster_per_cre(
    cluster_meta,
    cluster_order=None,
    col_to_group='cre_line',
    save_dir=save_dir
)

# Population average responses overlaid
print("Generating overlaid population average responses...")
suptitle = 'All cells'
plotting.plot_population_average_response_for_clusters_as_rows_all_response_types(
    image_mdf_clean,
    change_mdf_clean,
    omission_mdf_clean,
    save_dir=save_dir,
    suptitle=suptitle,
    cluster_order=None,
    ax=None
)

# By Cre line
for cre_line in cre_lines:
    image_mdf_clean_cre = image_mdf_clean[image_mdf_clean.cre_line == cre_line]
    change_mdf_clean_cre = change_mdf_clean[change_mdf_clean.cre_line == cre_line]
    omission_mdf_clean_cre = omission_mdf_clean[omission_mdf_clean.cre_line == cre_line]

    if len(image_mdf_clean_cre) > 0:
        plotting.plot_population_average_response_for_clusters_as_rows_all_response_types(
            image_mdf_clean_cre,
            change_mdf_clean_cre,
            omission_mdf_clean_cre,
            save_dir=save_dir,
            suffix=cre_line[:3],
            suptitle=cre_line[:3],
            cluster_order=None,
            ax=None
        )

# Mean cluster heatmaps
plotting.plot_mean_cluster_heatmaps_remapped(
    feature_matrix,
    cluster_meta,
    sort_by='cluster_id',
    save_dir=save_dir,
    folder='supplementary'
)

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print(f"\nFigure 5 generation complete!")
print(f"Figures saved to: {save_dir}")
print(f"Summary:")
print(f"  - Total cells analyzed: {len(cluster_meta)}")
print(f"  - Number of clusters: {len(cluster_meta.cluster_id.unique())}")
print(f"  - Cre lines: {', '.join(cre_lines)}")


# ============================================================================
# COMPOSITE FIGURE 5: All panels on a single figure axis
# ============================================================================
# Layout based on published Figure 5 (Experience-dependent coding diversity):
#   Row 1 (~25%): Panel A (Sst example cell + coding scores, left ~48%) |
#                 Panel B (Vip example cell + coding scores, right ~48%)
#   Row 2 (~40%): Panel C (clustered coding score heatmap, left ~42%) |
#                 Panel D (avg change responses by cluster, ~20%) |
#                 Panel E (avg omission responses by cluster, ~20%) |
#                 Panel F (% cells per cluster by cell type, ~15%)
#   Row 3 (~30%): Cluster means (left ~42%) | Panel G (summary table, right ~55%)
#
# Panels A, B are complex multi-cell examples — left as placeholders.
# Panels C-G involve clustering-specific visualizations from the plotting module.

def plot_figure_5_composite(cluster_meta, feature_matrix, umap_df,
                            save_dir=None, folder='composite_figures'):
    """
    Generate a composite Figure 5 with all panels arranged in the published layout.
    Saved as PDF for Illustrator editing.
    """
    figsize = (26, 28)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # ---- Row 1: Panels A, B (example cells with coding scores) ----
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.47], yspan=[0, 0.22])
    ax_A.text(0.5, 0.5, 'A — Example Sst cell\ncoding scores across sessions\n(place manually)',
              transform=ax_A.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1.0], yspan=[0, 0.22])
    ax_B.text(0.5, 0.5, 'B — Example Vip cell\ncoding scores across sessions\n(place manually)',
              transform=ax_B.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_B.axis('off')

    # ---- Row 2: Panels C, D, E, F ----
    # Panel C: Clustered coding score heatmap (all cells × features × sessions)
    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.40], yspan=[0.26, 0.62])
    ax_C.text(0.5, 0.5, 'C — Clustered coding score heatmap\n(place manually)',
              transform=ax_C.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    # Panel D: Average image change responses by cluster (cluster rows × experience cols)
    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.43, 0.63], yspan=[0.26, 0.62])
    ax_D.text(0.5, 0.5, 'D — Change responses\nby cluster\n(place manually)',
              transform=ax_D.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    # Panel E: Average omission responses by cluster (cluster rows × experience cols)
    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.66, 0.86], yspan=[0.26, 0.62])
    ax_E.text(0.5, 0.5, 'E — Omission responses\nby cluster\n(place manually)',
              transform=ax_E.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax_E.set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax_E.axis('off')

    # Panel F: % cells per cluster by cell type (stacked bar or pie)
    ax_F = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.88, 1.0], yspan=[0.26, 0.62])
    try:
        # Simple bar chart of cluster sizes
        cluster_sizes = cluster_meta.groupby('cluster_id').size()
        cluster_sizes = cluster_sizes.sort_index()
        ax_F.barh(range(len(cluster_sizes)), cluster_sizes.values, color='gray', alpha=0.7)
        ax_F.set_yticks(range(len(cluster_sizes)))
        ax_F.set_yticklabels(cluster_sizes.index, fontsize=8)
        ax_F.set_xlabel('# cells', fontsize=10)
        ax_F.invert_yaxis()
        ax_F.spines['top'].set_visible(False)
        ax_F.spines['right'].set_visible(False)
    except Exception:
        ax_F.text(0.5, 0.5, 'F — % cells\nper cluster',
                  transform=ax_F.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax_F.set_title('F', loc='left', fontweight='bold', fontsize=16)

    # ---- Row 3: Cluster means + Panel G (summary table) ----
    ax_means = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.40], yspan=[0.66, 0.95])
    ax_means.text(0.5, 0.5, 'Cluster means\n(place manually)',
                  transform=ax_means.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_means.axis('off')

    ax_G = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.43, 1.0], yspan=[0.66, 0.95])
    ax_G.text(0.5, 0.5, 'G — Summary of cluster properties\n(place manually)',
              transform=ax_G.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_G.set_title('G', loc='left', fontweight='bold', fontsize=16)
    ax_G.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_5_composite', formats=['.pdf'])
    return fig


# Generate composite figure
fig5 = plot_figure_5_composite(
    cluster_meta, feature_matrix, umap_df,
    save_dir=save_dir, folder='composite_figures'
)
