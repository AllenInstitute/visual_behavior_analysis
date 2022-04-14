import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})


from sklearn.cluster import SpectralClustering

import visual_behavior_glm.GLM_analysis_tools as gat

import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities

from visual_behavior.dimensionality_reduction.clustering import processing
from visual_behavior.dimensionality_reduction.clustering import plotting


### set critical params for processing and saving
# folder = '220412_across_session_norm_signed_weights'
folder = '220413_across_session_norm_signed_weights'
across_session_norm = True
use_signed_weights = True

# ### number of clusters for within session norm
# n_clusters_cre = {'Slc17a7-IRES2-Cre': 10,
#                   'Sst-IRES-Cre': 6,
#                   'Vip-IRES-Cre': 12}

### number of clusters for across session norm
n_clusters_cre = {'Slc17a7-IRES2-Cre': 10,
                  'Sst-IRES-Cre': 4,
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

### get GLM output, filter and reshape
glm_version = '24_events_all_L2_optimize_by_session'
model_output_type = 'adj_fraction_change_from_full'

base_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_plots\figure_4'
base_dir = os.path.join(base_dir, glm_version)
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

### create folder to load and save to
save_dir = os.path.join(base_dir, folder)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# get features
features = processing.get_features_for_clustering()

if across_session_norm:

    import visual_behavior_glm.GLM_across_session as gas

    # get across session normalized dropout scores
    df, failed_cells = gas.load_cells(cells='all')
    df = df.set_index('identifier')

    # only use across session values
    within = df[[key for key in df.keys() if '_within' in key] + ['cell_specimen_id', 'ophys_experiment_id',
                                                                  'experience_level']]
    across = df[[key for key in df.keys() if '_across' in key] + ['cell_specimen_id', 'ophys_experiment_id',
                                                                  'experience_level']]
    results_pivoted = across.copy()
    results_pivoted = results_pivoted.rename(
        columns={'omissions_across': 'omissions', 'all-images_across': 'all-images',
                 'behavioral_across': 'behavioral', 'task_across': 'task'})
    print(len(results_pivoted), 'len(results_pivoted)')

    # limit results pivoted to to last familiar and second novel active
    results_pivoted = results_pivoted[results_pivoted.ophys_experiment_id.isin(matched_experiments)]
    results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(matched_cells)]
    print(len(results_pivoted.cell_specimen_id.unique()),
          'cells in results_pivoted after limiting to strictly matched cells')

    # drop duplicates
    results_pivoted = results_pivoted.drop_duplicates(subset=['cell_specimen_id', 'experience_level'])
    print(len(results_pivoted), 'len(results_pivoted) after dropping duplicates')

else:
    # get GLM results from saved file in save_dir or from mongo if file doesnt exist
    results_pivoted = processing.get_glm_results_pivoted_for_clustering(glm_version, model_output_type, save_dir)

if use_signed_weights:
    import visual_behavior_glm.GLM_analysis_tools as gat
    import visual_behavior_glm.GLM_params as glm_params

    # get weights
    run_params = glm_params.load_run_json(glm_version)
    weights_df = gat.build_weights_df(run_params, results_pivoted)

    # modify regressors to include kernel weights as regressors with '_signed' appended
    results_pivoted = gat.append_kernel_excitation(weights_df, results_pivoted)
    # add behavioral results signed by duplicating behavioral dropouts
    results_pivoted['behavioral_signed'] = results_pivoted.behavioral.values

    # limit to features of interest, in this case limit to the 4 feature categories with '_signed' in name
    results_pivoted = processing.limit_results_pivoted_to_features_for_clustering(results_pivoted,
                                                                                  take_absolute_value=False,
                                                                                  use_signed_features=True)

    # now remove 'signed' from column names
    columns = {}
    for column in results_pivoted.columns:
        if 'signed' in column:
            # invert the sign so that "increased coding" is positive
            results_pivoted[column] = results_pivoted[column]*-1
            # get new column name without the '_signed'
            columns[column] = column.split('_')[0]
    results_pivoted = results_pivoted.rename(columns=columns)


    # drop duplicates again just in case
    results_pivoted = results_pivoted.reset_index().drop_duplicates(subset=['cell_specimen_id', 'experience_level'])

# save results_pivoted
results_pivoted.to_hdf(os.path.join(save_dir, glm_version + '_results_pivoted.h5'), key='df')

# get dropout scores just for the features we want to cluster on and take absolute value if not using signed weights
if use_signed_weights:
    take_absolute_value = False
else:
    take_absolute_value = True
dropouts = processing.limit_results_pivoted_to_features_for_clustering(results_pivoted, take_absolute_value)
dropouts.head()

# unstack dropout scores to get a vector of features x experience levels for each cell
feature_matrix = processing.get_feature_matrix_for_clustering(dropouts, glm_version, save_dir=save_dir)
feature_matrix.head()

# get cell metadata for all cells in feature_matrix
cell_metadata = processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)
cell_metadata.head()

### plot feature matrix for each cre line
plotting.plot_feature_matrix_for_cre_lines(feature_matrix, cell_metadata, save_dir=base_dir, folder=folder)

### get Silhouette scores
silhouette_scores = processing.load_silhouette_scores(glm_version, feature_matrix, cell_metadata, save_dir)


### plot silhouettes with selected # clusters
plotting.plot_silhouette_scores_n_clusters(silhouette_scores, cell_metadata, n_clusters_cre=n_clusters_cre,
                                           save_dir=base_dir, folder=folder)

### get coclustering matrices per cre line
coclustering_matrices = processing.get_coclustering_matrix(glm_version, feature_matrix, cell_metadata, n_clusters_cre,
                                                           save_dir, nboot=100)

### get cluster labels per cre line from Agglomerative clustering on co-clustering matrix
cluster_labels = processing.get_cluster_labels(coclustering_matrices, cell_metadata, n_clusters_cre, save_dir,
                                               load=False)

#### merge cluster labels with cell metadata, remove small clusters, and add manual sort order
cluster_meta = processing.get_cluster_meta(cluster_labels, cell_metadata, feature_matrix, n_clusters_cre, save_dir,
                                           load=False)


### plot co-clustering matrix and dendrograms sorted by linkage matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
# for i, cre_line in enumerate(processing.get_cre_lines(cluster_meta)):
#     plotting.plot_coclustering_matrix_and_dendrograms(coclustering_matrices, cre_line, cluster_meta, n_clusters_cre,
#                                                      save_dir=base_dir, folder=folder)

# data = coclustering_matrices[cre_line]
# fig, ax = plt.subplots(figsize=(8,8))
# ax = sns.clustermap(data=data, cmap='Greys', cbar_kws={'label':'co-clustering probability'})


### sort coclustering matrix by cluster ID / cluster size
cre_lines = processing.get_cre_lines(cell_metadata)
for cre_line in cre_lines:
    plotting.plot_coclustering_matrix_sorted_by_cluster_size(coclustering_matrices, cluster_meta, cre_line,
                                                             save_dir=None, folder=folder, ax=None)

### plot average dropout scores for each cluster

### plot each cluster separately and save to single cell examples dir
cell_examples_dir = os.path.join(save_dir, 'matched_cell_examples')
if not os.path.exists(cell_examples_dir):
    os.mkdir(cell_examples_dir)

# plotting.plot_dropout_heatmaps_and_save_to_cell_examples_folders(cluster_meta, feature_matrix, save_dir)

### plot average dropouts for each cre line in cluster size order
sort_col = 'cluster_id'
plotting.plot_dropout_heatmaps_for_clusters(cluster_meta, feature_matrix, sort_col=sort_col, save_dir=base_dir,
                                            folder=folder)

### plot dropout heatmaps in manually sorted order
sort_col = 'manual_sort_order'
plotting.plot_dropout_heatmaps_for_clusters(cluster_meta, feature_matrix, sort_col=sort_col, save_dir=base_dir,
                                            folder=folder)

### plot feature matrix sorted by cluster ID
plotting.plot_feature_matrix_sorted(feature_matrix, cluster_meta, sort_col='cluster_id', save_dir=base_dir,
                                    folder=folder)

plotting.plot_feature_matrix_sorted(feature_matrix, cluster_meta, sort_col='manual_sort_order', save_dir=base_dir,
                                    folder=folder)

### umap with cluster labels
label_col = 'cluster_id'

plotting.plot_umap_for_clusters(cluster_meta, feature_matrix, label_col=label_col, save_dir=base_dir, folder=folder)

### Correlations within clusters
plotting.plot_within_cluster_correlations_all_cre(cluster_meta, n_clusters_cre, sort_order=None, save_dir=base_dir,
                                                  folder=folder)

### average dropouts per cre line
plotting.plot_average_dropout_heatmap_for_cre_lines(dropouts, cluster_meta, save_dir=base_dir, folder=folder)

### plot 100 cells per cluster to examine within cluster variability
# plotting.plot_random_subset_of_cells_per_cluster(cluster_meta, dropouts, save_dir)


### breakdown by area and depth

# We are going to normalize within each area or depth to account for the large imbalance in N due to Scientifica datasets only being performed in V1 at certain depths, as well as biological variation in cell type specific expression by depth

### plot fraction and number of cells by area and depth
label = 'fraction of cells'
plotting.plot_fraction_cells_by_area_depth(cluster_meta, n_clusters_cre, normalize=True, label=label,
                                           save_dir=base_dir, folder=folder)

label = '# of cells'
plotting.plot_fraction_cells_by_area_depth(cluster_meta, n_clusters_cre, normalize=False, label=label,
                                           save_dir=base_dir, folder=folder)

### compute proportion of cells per area / depth relative to cluster average

# frequency = processing.make_frequency_table(cluster_meta[cluster_meta.cre_line==cre_lines[0]],
#                                             groupby_columns = ['targeted_structure', 'layer'], normalize=True)

# fraction_cells_per_cluster = processing.make_frequency_table(cluster_meta[cluster_meta.cre_line==cre_lines[0]],
#                                             groupby_columns = ['cre_line'], normalize=True)

### cluster proportions for one cre line
# cluster_proportions_cre = processing.compute_cluster_proportion_cre(cluster_meta, cre_line)
# cluster_proportions_cre

### cluster proportions for all cre lines
proportions = processing.get_proportion_cells_rel_cluster_average(cluster_meta, cre_lines)

### plot fraction cells per area depth
value_to_plot = 'proportion_cells'
cbar_label = 'proportion relative\n to cluster avg'
suffix = '_full_range'

vmax = proportions[value_to_plot].max()

plotting.plot_cell_stats_per_cluster_for_areas_depths(cluster_meta, proportions, n_clusters_cre,
                                                      value_to_plot, cbar_label, cmap='PRGn', vmin=-vmax, vmax=vmax,
                                                      cluster_order=None, save_dir=base_dir, folder=folder,
                                                      suffix=suffix)

### sort clusters by the fraction of cells in VISp upper
value_to_plot = 'proportion_cells'
cbar_label = 'proportion cells\n rel. to cluster avg'
suffix = '_full_range'

vmax = proportions[value_to_plot].max()
cluster_order = processing.get_cluster_order_for_metric_location(proportions, cluster_meta,
                                                                 location='VISp_upper', metric='proportion_cells')

plotting.plot_cell_stats_per_cluster_for_areas_depths(cluster_meta, proportions, n_clusters_cre,
                                                      value_to_plot, cbar_label, cmap='PRGn', vmin=-vmax, vmax=vmax,
                                                      cluster_order=cluster_order, save_dir=base_dir, folder=folder,
                                                      suffix=suffix)

### plot dropouts and pop avgs in this order

# load multi session dataframe with response traces
df_name = 'omission_response_df'
conditions = ['cell_specimen_id', 'omitted']

data_type = 'events'
event_type = 'omissions'
inclusion_criteria = 'active_only_closest_familiar_and_novel_containers_with_all_levels'

multi_session_df = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria)
print(len(multi_session_df.ophys_experiment_id.unique()))

cluster_mdf = multi_session_df.merge(cluster_meta[['cluster_id']],
                                     on='cell_specimen_id', how='inner')

# input needed for plot functions
fraction_cells = processing.get_fraction_cells_per_area_depth(cluster_meta)

cluster_order = processing.get_cluster_order_for_metric_location(proportions, cluster_meta,
                                                                 location='VISp_upper', metric='proportion_cells')

for cre_line in cre_lines:
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df,
                                              proportions, fraction_cells, cre_line,
                                              sort_order=cluster_order, suffix='_VISp_upper_proportion_rel_avg',
                                              save_dir=base_dir, folder=folder, )


### try for just V1 vs LM or upper vs lower

# areas
area_proportions = processing.get_proportion_cells_rel_cluster_average(cluster_meta, cre_lines,
                                                                       groupby_columns=['targeted_structure'])
area_proportions['location'] = area_proportions.targeted_structure

value_to_plot = 'proportion_cells'
cbar_label = 'proportion cells\n rel. to cluster avg'
suffix = '_proportion_rel_avg_area_sort'
vmax = area_proportions.proportion_cells.max().max()

cluster_order = processing.get_cluster_order_for_metric_location(area_proportions, cluster_meta,
                                                                 location='VISp', metric='proportion_cells')

plotting.plot_cell_stats_per_cluster_for_areas_depths(cluster_meta, area_proportions, n_clusters_cre,
                                                      value_to_plot, cbar_label, cmap='PRGn', vmin=-vmax, vmax=vmax,
                                                      cluster_order=cluster_order, save_dir=base_dir, folder=folder,
                                                      suffix=suffix)

for cre_line in cre_lines:
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df,
                                              area_proportions, fraction_cells, cre_line,
                                              sort_order=cluster_order, suffix='_VISp_upper_proportion_rel_avg',
                                              save_dir=base_dir, folder=folder, )

# depths
layer_proportions = processing.get_proportion_cells_rel_cluster_average(cluster_meta, cre_lines,
                                                                        groupby_columns=['layer'])
layer_proportions['location'] = layer_proportions.layer

value_to_plot = 'proportion_cells'
cbar_label = 'proportion cells\n rel. to cluster avg'
suffix = '_proportion_rel_avg_layer_sort'
vmax = layer_proportions.proportion_cells.max().max()

cluster_order = processing.get_cluster_order_for_metric_location(layer_proportions, cluster_meta,
                                                                 location='upper', metric='proportion_cells')

plotting.plot_cell_stats_per_cluster_for_areas_depths(cluster_meta, layer_proportions, n_clusters_cre,
                                                      value_to_plot, cbar_label, cmap='PRGn', vmin=-vmax, vmax=vmax,
                                                      cluster_order=cluster_order, save_dir=base_dir, folder=folder,
                                                      suffix=suffix)

for cre_line in cre_lines:
    plotting.plot_clusters_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df, cre_line,
                                              sort_order=cluster_order, suffix='_upper_proportion_rel_avg',
                                              save_dir=base_dir, folder=folder, )



