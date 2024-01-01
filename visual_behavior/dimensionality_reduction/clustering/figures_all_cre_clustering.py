import os
import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt

import visual_behavior.visualization.utils as utils

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
from visual_behavior.dimensionality_reduction.clustering import plotting
from visual_behavior.dimensionality_reduction.clustering import processing
from visual_behavior.dimensionality_reduction.clustering import single_cell_plots

import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_analysis_tools as gat

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})


### set paths

base_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_plots\figure_4'

save_dir = os.path.join(base_dir, 'all_cre_clustering_113023')
folder = 'figs'

### get experiment and cell metadata ###

# load experiments table
experiments_table = loading.get_platform_paper_experiment_table(limit_to_closest_active=True)
# limit to closest familiar and novel active
experiments_table = utilities.limit_to_last_familiar_second_novel_active(experiments_table)
experiments_table = utilities.limit_to_containers_with_all_experience_levels(experiments_table)

# load matched cells table
cells_table = loading.get_cell_table()
cells_table = loading.get_matched_cells_table(cells_table)
# update experience level names
cells_table['experience_level'] = [utils.convert_experience_level(experience_level) for experience_level in cells_table.experience_level.values]
matched_cells = cells_table.cell_specimen_id.unique()
matched_experiments = cells_table.ophys_experiment_id.unique()

# get cre_lines and cell types for plot labels
cre_lines = np.sort(cells_table.cre_line.unique())
cell_types = utils.get_cell_types()
cre_line_colors = utils.get_cre_line_colors()

# get other useful info

experience_levels = utils.get_new_experience_levels()
experience_level_colors = utils.get_experience_level_colors()

### get GLM results ###

# get across session normalized dropout scores

import visual_behavior_glm.GLM_across_session as gas

glm_version = '24_events_all_L2_optimize_by_session'

# get across session normalized dropout scores
across_results, failed_cells = gas.load_cells(glm_version, clean_df=True)
across_results = across_results.set_index('identifier')

# only use across session values
across = across_results[[key for key in across_results.keys() if '_across' in key] + ['cell_specimen_id', 'ophys_experiment_id', 'experience_level']]
results_pivoted = across.copy()
# rename across session columns
results_pivoted = results_pivoted.rename(columns={'omissions_across': 'omissions', 'all-images_across': 'all-images',
             'behavioral_across': 'behavioral', 'task_across': 'task'})

# limit to matched cells
results_pivoted = results_pivoted[results_pivoted.ophys_experiment_id.isin(matched_experiments)]
results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(matched_cells)]
# drop duplicates
results_pivoted = results_pivoted.drop_duplicates(subset=['cell_specimen_id', 'experience_level'])
print(len(results_pivoted.cell_specimen_id.unique()), 'cells in results_pivoted after limiting to strictly matched cells')

# limit to features used for clustering
features = processing.get_features_for_clustering()
features = [*features, 'ophys_experiment_id']
results_pivoted = processing.limit_results_pivoted_to_features_for_clustering(results_pivoted, features)
# flip sign so coding scores are positive
results_pivoted = processing.flip_sign_of_dropouts(results_pivoted, processing.get_features_for_clustering(), use_signed_weights=False)
# now drop ophys_experiment_id
results_pivoted = results_pivoted.drop(columns=['ophys_experiment_id'])
# update experience level names
results_pivoted['experience_level'] = [utils.convert_experience_level(experience_level) for experience_level in results_pivoted.experience_level.values]


### cluster on all cre lines together ###

feature_matrix = processing.get_feature_matrix_for_clustering(results_pivoted, glm_version, save_dir=save_dir)

# get metadata for cells that will be clustered
cell_metadata = processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)

n_clusters = 14 # empirically determined, then validated based on within vs. across cluster variance

cluster_meta_save_path = os.path.join(save_dir, 'cluster_meta_n_'+str(n_clusters)+'_clusters.h5')

# if clustering output exists, load it
if os.path.exists(cluster_meta_save_path):
    cluster_meta = pd.read_hdf(cluster_meta_save_path, key='df')
    if 0 in cluster_meta.cluster_id.unique():
        # add one to cluster ID so it starts at 1
        cluster_meta['cluster_id'] = cluster_meta['cluster_id']+1
    # merge in cell metadata
    cell_metadata = processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)
    cell_metadata = cell_metadata.drop(columns=['ophys_experiment_id', 'cre_line'])
    cluster_meta = cluster_meta.merge(cell_metadata.reset_index(), on='cell_specimen_id')
    cluster_meta = cluster_meta.set_index('cell_specimen_id')
# otherwise run it and save it
else:
    # run spectral clustering and get co-clustering matrix
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering()
    X = feature_matrix.values
    m = processing.get_coClust_matrix(X=X, n_clusters=n_clusters, model=sc, nboot=np.arange(100))
    # make co-clustering matrix a dataframe with cell_specimen_ids as indices and columns
    coclustering_df = pd.DataFrame(data=m, index=feature_matrix.index, columns=feature_matrix.index)

    # save co-clustering matrix
    coclust_save_path = os.path.join(save_dir, 'coclustering_matrix_n_'+str(n_clusters)+'_clusters.h5')
    coclustering_df.to_hdf(coclust_save_path, key='df', format='table')

    # run agglomerative clustering on co-clustering matrix to identify cluster labels
    from sklearn.cluster import AgglomerativeClustering
    X = coclustering_df.values
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                    linkage='average')
    labels = cluster.fit_predict(X)
    cell_specimen_ids = coclustering_df.index.values
    # make dictionary with labels for each cell specimen ID in this cre line
    labels_dict = {'labels': labels, 'cell_specimen_id': cell_specimen_ids}
    # turn it into a dataframe
    labels_df = pd.DataFrame(data=labels_dict, columns=['labels', 'cell_specimen_id'])
    # get new cluster_ids based on size of clusters and add to labels_df
    cluster_size_order = labels_df['labels'].value_counts().index.values
    # translate between original labels and new IDS based on cluster size
    labels_df['cluster_id'] = [np.where(cluster_size_order == label)[0][0] for label in labels_df.labels.values]
    # concatenate with df for all cre lines
    cluster_labels = labels_df

    # add metadata to cluster labels
    cell_metadata = processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)

    cluster_meta = cluster_labels[['cell_specimen_id', 'cluster_id', 'labels']].merge(cell_metadata, on='cell_specimen_id')
    cluster_meta = cluster_meta.set_index('cell_specimen_id')
    # annotate & clean cluster metadata
    cluster_meta = processing.clean_cluster_meta(cluster_meta)  # drop cluster IDs with fewer than 5 cells in them
    cluster_meta['original_cluster_id'] = cluster_meta.cluster_id
    # increment cluster ID if needed
    if 0 in cluster_meta.cluster_id.unique():
        # add one to cluster ID so it starts at 1
        cluster_meta['cluster_id'] = cluster_meta['cluster_id']+1

    # plot coclustering matrix - need to hack it since it assumes cre lines
    coclustering_dict = {}
    coclustering_dict['all'] = coclustering_df
    cluster_meta_tmp = cluster_meta.copy()
    cluster_meta_tmp['cre_line'] = 'all'
    plotting.plot_coclustering_matrix_sorted_by_cluster_size(coclustering_dict, cluster_meta_tmp, cre_line='all',
                                                    save_dir=save_dir, folder=folder, ax=None)

    # add within cluster correlation
    cluster_meta = processing.add_within_cluster_corr_to_cluster_meta(feature_matrix, cluster_meta, use_spearmanr=False)

    # plot within cluster correlations distribution
    plotting.plot_within_cluster_correlations(cluster_meta, sort_order=None, spearman=False, suffix='_'+str(n_clusters)+'_clusters',
                                                    save_dir=save_dir, folder=folder, ax=None)

    # save clustering results
    cluster_meta_save_path = os.path.join(save_dir, 'cluster_meta_n_'+str(n_clusters)+'_clusters.h5')
    cluster_data = cluster_meta.reset_index()[['cell_specimen_id', 'ophys_experiment_id', 'cre_line', 'cluster_id', 'labels', 'within_cluster_correlation_p']]
    cluster_data.to_hdf(cluster_meta_save_path, key='df', format='table')


### plot clustring results ###

plotting.plot_cre_line_means_remapped(feature_matrix, cluster_meta, save_dir, folder)

plotting.plot_cluster_means_remapped(feature_matrix, cluster_meta, save_dir=save_dir, folder=folder, ax=None)

# heatmap of coding scores prior to clustering
cluster_meta_tmp = cluster_meta.copy()
cluster_meta_tmp['cell_index'] = cluster_meta_tmp.index.values
plotting.plot_coding_score_heatmap_remapped(cluster_meta_tmp, feature_matrix, sort_by=None, session_colors=True,
                                    save_dir=save_dir, folder=folder, ax=None)


# clustered heatmap
plotting.plot_coding_score_heatmap_remapped(cluster_meta, feature_matrix, sort_by='cluster_id', session_colors=True,
                                    save_dir=save_dir, folder=folder, ax=None)


# average coding sores for clusters
plotting.plot_cluster_means_remapped(feature_matrix, cluster_meta, save_dir=save_dir, folder=folder, ax=None)


### load multi session dfs to plot population average responses ###

data_type = 'events'
interpolate = True
output_sampling_rate = 30
inclusion_criteria = 'platform_experiment_table'

# image and image change multi session dfs
event_type = 'all'
conditions = ['cell_specimen_id', 'is_change']

multi_session_df = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria,
                                                        interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                         epoch_duration_mins=None)

change_mdf = multi_session_df[multi_session_df.is_change==True]
image_mdf = multi_session_df[(multi_session_df.is_change==False)]

# omission mdf
event_type = 'all'
conditions = ['cell_specimen_id', 'omitted']

omission_mdf = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria,
                                                        interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                         epoch_duration_mins=None)
omission_mdf = omission_mdf[omission_mdf.omitted==True]

### plot population average image response for each cluster (averaged over cre lines) ###

# remove outliers
tmp = image_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')
tmp = tmp[(tmp.mean_baseline<np.percentile(tmp.mean_baseline.values, 99.9)) &  (tmp.mean_response<np.percentile(tmp.mean_response.values, 99.8))]

xlim_seconds = [-0.25, 0.75]
timestamps = tmp.trace_timestamps.values[0]

axes_column = 'cluster_id'
hue_column = 'experience_level'

plotting.plot_population_averages_for_clusters(tmp, event_type, axes_column, hue_column,
                                            xlim_seconds=xlim_seconds, interval_sec=0.5,
                                            sharey=True, sharex=False,
                                            ylabel='response', xlabel='time (s)', suptitle=None,
                                            save_dir=save_dir, folder=folder, suffix='_sharey', ax=None);

# as a grid
plotting.plot_population_averages_for_clusters_grid(tmp, event_type, axes_column, hue_column,
                                            xlim_seconds=xlim_seconds, interval_sec=0.5,
                                            sharey=True, sharex=True,
                                            ylabel='response', xlabel='time (s)', suptitle=None,
                                            save_dir=save_dir, folder=folder, suffix='_sharey', ax=None);

# plot with cre lines separately
plotting.plot_population_averages_for_conditions_multi_row(tmp, data_type, 'all', axes_column='cre_line', hue_column='experience_level', row_column='cluster_id',
                                            timestamps=None, xlim_seconds=xlim_seconds, interval_sec=0.5, sharey=False, sharex=False, palette=None,
                                            ylabel='response', xlabel='time (s)', title=None, suptitle=None,
                                            save_dir=save_dir, folder=folder, suffix='_images_clusters', ax=None);


### plot population average change response for each cluster (averaged over cre lines) ###

# remove outliers
tmp = change_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')
tmp = tmp[(tmp.mean_baseline<np.percentile(tmp.mean_baseline.values, 99.9)) &  (tmp.mean_response<np.percentile(tmp.mean_response.values, 99.8))]

xlim_seconds = [-0.9, 0.75]
timestamps = tmp.trace_timestamps.values[0]

# all cre lines averaged
plotting.plot_population_averages_for_clusters(tmp, 'changes', axes_column, hue_column,
                                            xlim_seconds=xlim_seconds, interval_sec=0.5,
                                            sharey=True, sharex=False,
                                            ylabel='response', xlabel='time (s)', suptitle=None,
                                            save_dir=save_dir, folder=folder, suffix='_sharey', ax=None);

# as a grid
plotting.plot_population_averages_for_clusters_grid(tmp, event_type, axes_column, hue_column,
                                            xlim_seconds=xlim_seconds, interval_sec=0.5,
                                            sharey=True, sharex=True,
                                            ylabel='response', xlabel='time (s)', suptitle=None,
                                            save_dir=save_dir, folder=folder, suffix='_sharey', ax=None);

# plot with each cre line separately
plotting.plot_population_averages_for_conditions_multi_row(tmp, data_type, 'changes', axes_column='cre_line', hue_column='experience_level', row_column='cluster_id',
                                            timestamps=None, xlim_seconds=xlim_seconds, interval_sec=0.5, sharey=False, sharex=False, palette=None,
                                            ylabel='response', xlabel='time (s)', title=None, suptitle=None,
                                            save_dir=save_dir, folder=folder, suffix='_changes_clusters_per_cre', ax=None);


# population average omission response for each cluster (averaged over cre line

# remove outliers
tmp = omission_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')
tmp = tmp[(tmp.mean_baseline<np.percentile(tmp.mean_baseline.values, 99.9)) &  (tmp.mean_response<np.percentile(tmp.mean_response.values, 99.8))]

xlim_seconds = [-1, 1.5]
timestamps = tmp.trace_timestamps.values[0]

# cre lines averaged
plotting.plot_population_averages_for_clusters(tmp, 'omissions', axes_column, hue_column,
                                            xlim_seconds=xlim_seconds, interval_sec=1,
                                            sharey=True, sharex=False,
                                            ylabel='response', xlabel='time (s)', suptitle=None,
                                            save_dir=save_dir, folder=folder, suffix='_sharey', ax=None);

# as a grid
plotting.plot_population_averages_for_clusters_grid(tmp, event_type, axes_column, hue_column,
                                            xlim_seconds=xlim_seconds, interval_sec=0.5,
                                            sharey=True, sharex=True,
                                            ylabel='response', xlabel='time (s)', suptitle=None,
                                            save_dir=save_dir, folder=folder, suffix='_sharey', ax=None);

# split by cre line
plotting.plot_population_averages_for_conditions_multi_row(tmp, data_type, 'omissions', axes_column='cre_line', hue_column='experience_level', row_column='cluster_id',
                                            timestamps=None, xlim_seconds=xlim_seconds, interval_sec=0.5, sharey=False, sharex=False, palette=None,
                                            ylabel='response', xlabel='time (s)', title=None, suptitle=None,
                                            save_dir=save_dir, folder=folder, suffix='_omissions_clusters_per_cre', ax=None);


### area / depth distributions by cluster & cre line

# across layers
location = 'layer'
order = ['upper', 'lower']
n_cells_table = processing.get_cluster_proportion_stats_for_locations(cluster_meta, location=location)

metric = 'fraction_cells_location'
xlabel = 'fraction cells'
significance_col = 'bh_significant'

plotting.plot_location_distribution_across_clusters_for_cre_lines(n_cells_table, location=location, order=order,
                                                             metric=metric, xlabel=xlabel,
                                                             significance_col=significance_col,
                                                             ax=None, save_dir=save_dir, folder=folder);


# across depths
location = 'binned_depth'
n_cells_table_depth = processing.get_cluster_proportion_stats_for_locations(cluster_meta, location=location)

metric = 'fraction_cells_location'
xlabel = 'fraction cells'
significance_col = 'bh_significant'

plotting.plot_location_distribution_across_clusters_for_cre_lines(n_cells_table_depth, location=location, order=None,
                                                             metric=metric, xlabel=xlabel,
                                                             significance_col=significance_col,
                                                             ax=None, save_dir=save_dir, folder=folder);

# by area
location = 'targeted_structure'
order = ['VISp', 'VISl']
n_cells_table_depth = processing.get_cluster_proportion_stats_for_locations(cluster_meta, location=location)

metric = 'fraction_cells_location'
xlabel = 'fraction cells'
significance_col = 'bh_significant'

plotting.plot_location_distribution_across_clusters_for_cre_lines(n_cells_table_depth, location=location, order=order,
                                                             metric=metric, xlabel=xlabel,
                                                             significance_col=significance_col,
                                                             sharex=True,
                                                             ax=None, save_dir=save_dir, folder=folder);



### experience modulation vs. layer bias

# get experience modulation & other metrics computed on average coding scores for each cluster
cluster_metrics = processing.get_cluster_metrics(cluster_meta, feature_matrix, results_pivoted)
cluster_metrics.head()

# compute area/depth distributions
location = 'layer'
order = ['upper', 'lower']
n_cells_table = processing.get_cluster_proportion_stats_for_locations(cluster_meta, location=location)

# add layer index metric (upper-lower/sum)
cluster_metrics = processing.add_layer_index_to_cluster_metrics(cluster_metrics, n_cells_table)

# plot exp mod vs. layer index as scatterplot with one point per cre line / cluster, size of point indicating cluster size



### TBD ###


### distribution of clusters over mice / microscopes / etc ###


### average tuning curve per cluster ###


### within session changes per cluster ###









