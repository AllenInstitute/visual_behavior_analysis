import os
import numpy as np

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities

from visual_behavior.dimensionality_reduction.clustering import plotting
from visual_behavior.dimensionality_reduction.clustering import processing

import seaborn as sns

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

# set critical params for processing and saving
folder = '220527_across_session_norm_10_5_10'
across_session_norm = True
use_signed_weights = False
include_var_exp_full = False

# number of clusters to use
n_clusters_cre = {'Slc17a7-IRES2-Cre': 10,
                  'Sst-IRES-Cre': 5,
                  'Vip-IRES-Cre': 10}

# load experiments table
experiments_table = loading.get_platform_paper_experiment_table()
# limit to closest familiar and novel active
experiments_table = utilities.limit_to_last_familiar_second_novel_active(experiments_table)
experiments_table = utilities.limit_to_containers_with_all_experience_levels(experiments_table)

# load matched cells table
cells_table = loading.get_cell_table()
cells_table = loading.get_matched_cells_table(cells_table)
matched_cells = cells_table.cell_specimen_id.unique()
matched_experiments = cells_table.ophys_experiment_id.unique()

# get cre_lines and cell types for plot labels
cre_lines = np.sort(cells_table.cre_line.unique())
cell_types = utilities.get_cell_types_dict(cre_lines, experiments_table)

# get GLM output, filter and reshape
glm_version = '24_events_all_L2_optimize_by_session'
model_output_type = 'adj_fraction_change_from_full'

base_dir = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/figure_4'
base_dir = os.path.join(base_dir, glm_version)
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# create folder to load and save to
save_dir = os.path.join(base_dir, folder)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

results_pivoted, weights_df, kernels = processing.generate_GLM_outputs(glm_version, experiments_table, cells_table,
                                                                       save_dir,
                                                                       use_signed_weights, across_session_norm,
                                                                       include_var_exp_full)

# unstack dropout scores to get a vector of features x experience levels for each cell
feature_matrix = processing.get_feature_matrix_for_clustering(results_pivoted, glm_version, save_dir=save_dir)
feature_matrix.head()

# get cell metadata for all cells in feature_matrix
cell_metadata = processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)
cell_metadata.head()

# plot feature matrix for each cre line
plotting.plot_feature_matrix_for_cre_lines(feature_matrix, cell_metadata, save_dir=base_dir, folder=folder)

# get Silhouette scores
silhouette_scores = processing.load_silhouette_scores(glm_version, feature_matrix, cell_metadata, save_dir=save_dir)

# plot silhouettes with selected # clusters
plotting.plot_silhouette_scores_n_clusters(silhouette_scores, cell_metadata, n_clusters_cre=n_clusters_cre,
                                           save_dir=base_dir, folder=folder)

# get gap statistic values
gap_statistic = processing.load_gap_statistic(glm_version, feature_matrix, cell_metadata, save_dir=save_dir,
                                              metric='euclidean', shuffle_type='all', k_max=25, n_boots=20)

# plot gap statistic
plotting.plot_gap_statistic(gap_statistic, cre_lines, save_dir=base_dir, folder=folder)

# load eigengap values
eigengap = processing.load_eigengap(glm_version, feature_matrix, cell_metadata, save_dir=save_dir, k_max=25)

# plot eigen gap values
plotting.plot_eigengap_values(eigengap, cre_lines, save_dir=base_dir, folder=folder)

# get coclustering matrices per cre line
coclustering_matrices = processing.get_coclustering_matrix(glm_version, feature_matrix, cell_metadata, n_clusters_cre,
                                                           save_dir, nboot=100)

# get cluster labels per cre line from Agglomerative clustering on co-clustering matrix
cluster_labels = processing.get_cluster_labels(coclustering_matrices, cell_metadata, n_clusters_cre, save_dir,
                                               load=False)

# merge cluster labels with cell metadata, remove small clusters, and add manual sort order
cluster_meta = processing.get_cluster_meta(cluster_labels, cell_metadata, feature_matrix, n_clusters_cre, save_dir,
                                           load=False)

# plot co-clustering matrix and dendrograms sorted by linkage matrix
# for i, cre_line in enumerate(processing.get_cre_lines(cluster_meta)):
#     plotting.plot_coclustering_matrix_and_dendrograms(coclustering_matrices, cre_line, cluster_meta, n_clusters_cre,
#                                                      save_dir=base_dir, folder=folder)

# data = coclustering_matrices[cre_line]
# fig, ax = plt.subplots(figsize=(8,8))
# ax = sns.clustermap(data=data, cmap='Greys', cbar_kws={'label':'co-clustering probability'})


# sort coclustering matrix by cluster ID / cluster size
cre_lines = processing.get_cre_lines(cell_metadata)
for cre_line in cre_lines:
    plotting.plot_coclustering_matrix_sorted_by_cluster_size(coclustering_matrices, cluster_meta, cre_line,
                                                             save_dir=base_dir, folder=folder, ax=None)

# plot average dropout scores for each cluster

# plot each cluster separately and save to single cell examples dir
cell_examples_dir = os.path.join(save_dir, 'matched_cell_examples')
if not os.path.exists(cell_examples_dir):
    os.mkdir(cell_examples_dir)

# plotting.plot_dropout_heatmaps_and_save_to_cell_examples_folders(cluster_meta, feature_matrix, save_dir)

# plot average dropouts for each cre line in cluster size order
sort_col = 'cluster_id'
plotting.plot_dropout_heatmaps_for_clusters(cluster_meta, feature_matrix, sort_col=sort_col, save_dir=base_dir,
                                            folder=folder)

# plot feature matrix sorted by cluster ID
plotting.plot_feature_matrix_sorted(feature_matrix, cluster_meta, sort_col='cluster_id', save_dir=base_dir,
                                    folder=folder)

# umap with cluster labels
label_col = 'cluster_id'
plotting.plot_umap_for_clusters(cluster_meta, feature_matrix, label_col=label_col, save_dir=base_dir, folder=folder)

# Correlations within clusters
plotting.plot_within_cluster_correlations_all_cre(cluster_meta, n_clusters_cre, sort_order=None, save_dir=base_dir,
                                                  folder=folder)

# average dropouts per cre line
plotting.plot_average_dropout_heatmap_for_cre_lines(results_pivoted, cluster_meta, save_dir=base_dir, folder=folder)

# plot std dropouts per cre line
plotting.plot_std_dropout_heatmap_for_cre_lines(results_pivoted, cluster_meta, save_dir=base_dir, folder=folder)

# plot clusters as rows

for cre in cre_lines:
    plotting.plot_clusters_row(cluster_meta, feature_matrix, cre_line,
                                   sort_order=None, save_dir=base_dir, folder=folder, suffix='')

# # plot 100 cells per cluster to examine within cluster variability
# dropouts = results_pivoted.merge(cells_table[['cell_specimen_id', 'experience_level', 'ophys_experiment_id']], on=['cell_specimen_id', 'experience_level'])
# dropouts = dropouts.drop_duplicates(subset=['cell_specimen_id', 'ophys_experiment_id'])
# plotting.plot_random_subset_of_cells_per_cluster(cluster_meta, dropouts, save_dir)


# breakdown by area and depth

# We are going to normalize within each area or depth to account for the large imbalance in N due to Scientifica datasets only being performed in V1 at certain depths, as well as biological variation in cell type specific expression by depth

# plot fraction and number of cells by area and depth
label = 'fraction of cells'
plotting.plot_fraction_cells_by_area_depth(cluster_meta, n_clusters_cre, normalize=True, label=label,
                                           save_dir=base_dir, folder=folder)

label = '# of cells'
plotting.plot_fraction_cells_by_area_depth(cluster_meta, n_clusters_cre, normalize=False, label=label,
                                           save_dir=base_dir, folder=folder)

# compute proportion of cells per area / depth relative to cluster average

# cluster proportions for all cre lines
proportions, stats = processing.get_proportion_cells_rel_cluster_average(cluster_meta, cre_lines,
                                                                         columns_to_groupby=['targeted_structure',
                                                                                             'layer'])

# plot fraction cells per area depth
value_to_plot = 'proportion_cells'
cbar_label = 'proportion relative\n to cluster avg'
suffix = '_size_order'

vmax = np.amax(proportions[value_to_plot].values)

plotting.plot_cell_stats_per_cluster_for_areas_depths(cluster_meta, proportions, n_clusters_cre,
                                                      value_to_plot, cbar_label, cmap='PRGn', vmin=-vmax, vmax=vmax,
                                                      cluster_order=None, save_dir=base_dir, folder=folder,
                                                      suffix=suffix)

# plot fraction of cells in each area and depth belonging to each cluster as a pie chart
plotting.plot_proportion_cells_area_depth_pie_chart(cluster_meta, save_dir=base_dir, folder=folder)


# plot fraction of cells in each area and depth belonging to each cluster as a lollipop graph
plotting.plot_fraction_cells_per_cluster_per_location_vert(cluster_meta, save_dir=base_dir, folder=folder)
plotting.plot_fraction_cells_per_cluster_per_location_horiz(cluster_meta, save_dir=base_dir, folder=folder)



# sort clusters by the fraction of cells in VISp upper
# value_to_plot = 'proportion_cells'
# cbar_label = 'proportion cells\n rel. to cluster avg'
# suffix = '_VISp_upper_sort'
#
# proportions, stats = processing.get_proportion_cells_rel_cluster_average(cluster_meta, cre_lines, groupby_columns=['targeted_structure', 'layer'])
#
# vmax = proportions[value_to_plot].max()
# cluster_order = processing.get_cluster_order_for_metric_location(proportions, cluster_meta,
#                                                                  location='VISp_upper', metric='proportion_cells')
#
# plotting.plot_cell_stats_per_cluster_for_areas_depths(cluster_meta, proportions, n_clusters_cre,
#                                                       value_to_plot, cbar_label, cmap='PRGn', vmin=-vmax, vmax=vmax,
#                                                       cluster_order=cluster_order, save_dir=base_dir, folder=folder,
#                                                       suffix=suffix)


# plot cluster heatmaps, pop avgs and proportion cells

# load multi session dataframe with response traces
multi_session_df = processing.get_multi_session_df_for_omissions()

cluster_mdf = multi_session_df.merge(cluster_meta[['cluster_id']],
                                     on='cell_specimen_id', how='inner')

# area and layer
for cre_line in cre_lines:
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df, cre_line,
                                              columns_to_groupby=['targeted_structure', 'layer'],
                                              sort_order=None, suffix='_area_layer',
                                              save_dir=base_dir, folder=folder, )

# just area
for cre_line in cre_lines:
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df, cre_line,
                                              columns_to_groupby=['targeted_structure'],
                                              sort_order=None, suffix='_area',
                                              save_dir=base_dir, folder=folder, )

# just layer
for cre_line in cre_lines:
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df, cre_line,
                                              columns_to_groupby=['layer'],
                                              sort_order=None, suffix='_layer',
                                              save_dir=base_dir, folder=folder, )

# binned dedpth
for cre_line in cre_lines:
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df, cre_line,
                                              columns_to_groupby=['binned_depth'],
                                              sort_order=None, suffix='_binned_depth',
                                              save_dir=base_dir, folder=folder, )


# plot as columns
for cre_line in cre_lines:
    plotting.plot_clusters_stats_pop_avg_cols(cluster_meta, feature_matrix, multi_session_df, cre_line,
                                     columns_to_groupby=['targeted_structure', 'layer'],
                                     sort_order=None, save_dir=base_dir, folder=folder, suffix='_pop_avg')

