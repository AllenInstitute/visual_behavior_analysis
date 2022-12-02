#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})


# In[2]:


from sklearn.cluster import SpectralClustering


# In[3]:


import visual_behavior_glm.GLM_analysis_tools as gat


# In[4]:


import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities


# In[5]:


from visual_behavior.dimensionality_reduction.clustering import processing
from visual_behavior.dimensionality_reduction.clustering import plotting


# In[6]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### get experiments and cells tables and limit to closest familiar and novel active

# In[7]:


experiments_table = loading.get_platform_paper_experiment_table()
len(experiments_table)


# In[8]:


# limit to closest familiar and novel active
experiments_table = utilities.limit_to_last_familiar_second_novel_active(experiments_table)
experiments_table = utilities.limit_to_containers_with_all_experience_levels(experiments_table)
len(experiments_table)


# In[9]:


matched_experiments = experiments_table.index.values


# In[10]:


cells_table = loading.get_cell_table()
len(cells_table.cell_specimen_id.unique())


# In[11]:


cells_table = loading.get_matched_cells_table(cells_table)
matched_cells = cells_table.cell_specimen_id.unique()


# In[12]:


# get cre_lines and cell types for plot labels
cre_lines = np.sort(cells_table.cre_line.unique())
cell_types = utilities.get_cell_types_dict(cre_lines, experiments_table)


# ### get GLM output, filter and reshape

# In[13]:


glm_version = '24_events_all_L2_optimize_by_session'
model_output_type = 'adj_fraction_change_from_full'


# In[14]:


base_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_plots\figure_4'
base_dir = os.path.join(base_dir, glm_version)
if not os.path.exists(base_dir):
    os.mkdir(base_dir)


# ### NOTE: create a new folder if you are testing out this notebook or else it will overwrite existing files & plots

# In[15]:


folder = '220411_cluster_proportions'
save_dir = os.path.join(base_dir, folder)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# In[16]:


features = processing.get_features_for_clustering()


# In[17]:


# get GLM results from saved file in save_dir or from mongo if file doesnt exist
results_pivoted = processing.get_glm_results_pivoted_for_clustering(glm_version, model_output_type, save_dir)
results_pivoted.head()


# In[18]:


# get dropout scores just for the features we want to cluster on
dropouts = processing.limit_results_pivoted_to_features_for_clustering(results_pivoted)
dropouts.head()


# In[19]:


# unstack dropout scores to get a vector of features x experience levels for each cell
feature_matrix = processing.get_feature_matrix_for_clustering(dropouts, glm_version, save_dir=save_dir)
feature_matrix.head()


# In[20]:


# get cell metadata for all cells in feature_matrix
cell_metadata = processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)
cell_metadata.head()


# ### plot feature matrix for each cre line

# In[21]:


plotting.plot_feature_matrix_for_cre_lines(feature_matrix, cell_metadata, save_dir=base_dir, folder=folder)


# ### get Silhouette scores

# In[22]:


silhouette_scores = processing.load_silhouette_scores(glm_version, feature_matrix, cell_metadata, save_dir)


# ### select  number of clusters

# In[23]:


n_clusters_cre = {'Slc17a7-IRES2-Cre': 10,
                 'Sst-IRES-Cre': 6, 
                 'Vip-IRES-Cre':12}


# ### plot silhouettes with selected # clusters

# In[24]:


plotting.plot_silhouette_scores_n_clusters(silhouette_scores, cell_metadata, n_clusters_cre=n_clusters_cre, save_dir=base_dir, folder=folder)


# ### get coclustering matrices per cre line

# In[25]:


coclustering_matrices = processing.get_coclustering_matrix(glm_version, feature_matrix, cell_metadata, n_clusters_cre, save_dir, nboot=100)


# ### get cluster labels per cre line from Agglomerative clustering on co-clustering matrix

# In[26]:


cluster_labels = processing.get_cluster_labels(coclustering_matrices, cell_metadata, n_clusters_cre, save_dir, load=False)
cluster_labels.head()


# #### merge cluster labels with cell metadata, remove small clusters, and add manual sort order

# In[27]:


cluster_meta = processing.get_cluster_meta(cluster_labels, cell_metadata, feature_matrix, n_clusters_cre, save_dir, load=False)
cluster_meta.head()


# In[28]:


cluster_meta[['labels', 'cluster_id', 'original_cluster_id', 'manual_sort_order', 'within_cluster_correlation']].head()


# In[ ]:





# ### plot co-clustering matrix and dendrograms sorted by linkage matrix

# In[29]:


from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


# In[30]:


# for i, cre_line in enumerate(processing.get_cre_lines(cluster_meta)):
#     plotting.plot_coclustering_matrix_and_dendrograms(coclustering_matrices, cre_line, cluster_meta, n_clusters_cre, 
#                                                      save_dir=base_dir, folder=folder)


# In[31]:


# data = coclustering_matrices[cre_line]

# fig, ax = plt.subplots(figsize=(8,8))
# ax = sns.clustermap(data=data, cmap='Greys', cbar_kws={'label':'co-clustering probability'})


# ### sort coclustering matrix by cluster ID / cluster size

# In[32]:


cre_lines = processing.get_cre_lines(cell_metadata)
for cre_line in cre_lines: 
    plotting.plot_coclustering_matrix_sorted_by_cluster_size(coclustering_matrices, cluster_meta, cre_line, 
                                                    save_dir=None, folder=folder, ax=None)


# ### plot average dropout scores for each cluster

# ### plot each cluster separately and save to single cell examples dir

# In[33]:


cell_examples_dir = os.path.join(save_dir, 'matched_cell_examples')
if not os.path.exists(cell_examples_dir):
    os.mkdir(cell_examples_dir)


# In[34]:


# plotting.plot_dropout_heatmaps_and_save_to_cell_examples_folders(cluster_meta, feature_matrix, save_dir)


# ### plot average dropouts for each cre line in cluster size order

# In[35]:


sort_col = 'cluster_id'

plotting.plot_dropout_heatmaps_for_clusters(cluster_meta, feature_matrix, sort_col=sort_col, save_dir=base_dir, folder=folder)


# ### plot dropout heatmaps in manually sorted order

# In[36]:


sort_col = 'manual_sort_order'

plotting.plot_dropout_heatmaps_for_clusters(cluster_meta, feature_matrix, sort_col=sort_col, save_dir=base_dir, folder=folder)


# In[ ]:





# ### plot feature matrix sorted by cluster ID

# In[37]:


plotting.plot_feature_matrix_sorted(feature_matrix, cluster_meta, sort_col='cluster_id', save_dir=base_dir, folder=folder)
    


# In[38]:


plotting.plot_feature_matrix_sorted(feature_matrix, cluster_meta, sort_col='manual_sort_order', save_dir=base_dir, folder=folder)


# ### umap with cluster labels

# In[39]:


# for column in ['project_code', 'binned_depth', 'targeted_structure', 'mouse_id']:
label_col = 'cluster_id'

plotting.plot_umap_for_clusters(cluster_meta, feature_matrix, label_col=label_col, save_dir=base_dir, folder=folder)


# ### Correlations within clusters

# In[40]:


plotting.plot_within_cluster_correlations_all_cre(cluster_meta, n_clusters_cre, sort_order=None, save_dir=base_dir, folder=folder)


# ### average dropouts per cre line

# In[42]:


plotting.plot_average_dropout_heatmap_for_cre_lines(dropouts, cluster_meta, save_dir=base_dir, folder=folder)


# In[ ]:





# ### plot 100 cells per cluster to examine within cluster variability

# In[43]:


# plotting.plot_random_subset_of_cells_per_cluster(cluster_meta, dropouts, save_dir)


# ### breakdown by area and depth

# We are going to normalize within each area or depth to account for the large imbalance in N due to Scientifica datasets only being performed in V1 at certain depths, as well as biological variation in cell type specific expression by depth

# ### plot fraction cells by area and depth

# In[44]:


label = 'fraction of cells'
plotting.plot_fraction_cells_by_area_depth(cluster_meta, n_clusters_cre, normalize=True, label=label, 
                                           save_dir=base_dir, folder=folder)


# In[45]:


label = '# of cells'
plotting.plot_fraction_cells_by_area_depth(cluster_meta, n_clusters_cre, normalize=False, label=label, 
                                           save_dir=base_dir, folder=folder)


# ### compute proportion of cells per area / depth relative to cluster average 

# In[72]:


# frequency = processing.make_frequency_table(cluster_meta[cluster_meta.cre_line==cre_lines[0]], 
#                                             groupby_columns = ['targeted_structure', 'layer'], normalize=True)
# frequency


# In[73]:


# fraction_cells_per_cluster = processing.make_frequency_table(cluster_meta[cluster_meta.cre_line==cre_lines[0]], 
#                                             groupby_columns = ['cre_line'], normalize=True)
# fraction_cells_per_cluster


# In[74]:


# # cluster proportions for one cre line
# cluster_proportions_cre = processing.compute_cluster_proportion_cre(cluster_meta, cre_line)
# cluster_proportions_cre


# In[61]:


# for all cre lines
proportions = processing.get_proportion_cells_rel_cluster_average(cluster_meta, cre_lines)
proportions


# In[63]:


value_to_plot = 'proportion_cells'
cbar_label = 'proportion relative\n to cluster avg'
suffix = '_full_range'

vmax = proportions[value_to_plot].max()

plotting.plot_cell_stats_per_cluster_for_areas_depths(cluster_meta, proportions, n_clusters_cre, 
                                             value_to_plot, cbar_label, cmap='PRGn', vmin=-vmax, vmax=vmax, 
                                             cluster_order=None, save_dir=base_dir, folder=folder, suffix=suffix)


# In[ ]:





# ### sort clusters by the fraction of cells in VISp upper using Alexs method

# In[64]:


value_to_plot = 'proportion_cells'
cbar_label = 'proportion cells\n rel. to cluster avg'
suffix = '_full_range'

vmax = proportions[value_to_plot].max()

cluster_order = processing.get_cluster_order_for_metric_location(proportions, cluster_meta, 
                                                                 location='VISp_upper', metric='proportion_cells')  

plotting.plot_cell_stats_per_cluster_for_areas_depths(cluster_meta, proportions, n_clusters_cre, 
                                             value_to_plot, cbar_label, cmap='PRGn', vmin=-vmax, vmax=vmax, 
                                             cluster_order=cluster_order, save_dir=base_dir, folder=folder, suffix=suffix)


# ### plot dropouts and pop avgs in this order

# In[65]:


# load dataframe with response traces
df_name = 'omission_response_df'
conditions = ['cell_specimen_id', 'omitted']

data_type = 'events'
event_type = 'omissions'
inclusion_criteria = 'active_only_closest_familiar_and_novel_containers_with_all_levels'


multi_session_df = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria)
print(len(multi_session_df.ophys_experiment_id.unique()))


cluster_mdf = multi_session_df.merge(cluster_meta[['cluster_id']], 
                                     on='cell_specimen_id', how = 'inner')


# In[66]:


# input needed for plot functions
fraction_cells = processing.get_fraction_cells_per_area_depth(cluster_meta)


# In[68]:


cluster_order = processing.get_cluster_order_for_metric_location(proportions, cluster_meta, 
                                                                 location='VISp_upper', metric='proportion_cells')  

for cre_line in cre_lines: 
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df,
                                              proportions, fraction_cells, cre_line, 
                                              sort_order=cluster_order, suffix='_VISp_upper_proportion_rel_avg', 
                                              save_dir=base_dir, folder=folder, )


# ### try for just V1 vs LM or upper vs lower 

# ### areas

# In[69]:


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
                                                      cluster_order=cluster_order, save_dir=base_dir, folder=folder, suffix=suffix)

for cre_line in cre_lines: 
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df,
                                              area_proportions, fraction_cells, cre_line, 
                                              sort_order=cluster_order, suffix='_VISp_upper_proportion_rel_avg', 
                                              save_dir=base_dir, folder=folder, )


# ### depths

# In[71]:


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
                                                      cluster_order=cluster_order, save_dir=base_dir, folder=folder, suffix=suffix)

for cre_line in cre_lines: 
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df,
                                              layer_proportions, fraction_cells, cre_line, 
                                              sort_order=cluster_order, suffix='_upper_proportion_rel_avg', 
                                              save_dir=base_dir, folder=folder, )


# ### test plot components then plot clusters with other information as additional panels 

# In[ ]:





# In[ ]:





# # next step: adapt below functions to show metric distributions

# ### plot clusters with % cells per area and depth

# In[110]:


fraction_cells = processing.get_fraction_cells_per_area_depth(cluster_meta)
cre_fraction = fraction_cells[fraction_cells.cre_line==cre_line]


# In[111]:


plotting.plot_fraction_cells_per_area_depth(cre_fraction, 1, ax=None)


# ### get relevant data and plot clusters with fractions

# In[112]:


cell_count_stats = processing.get_cell_count_stats(cluster_meta)
cre_counts = cell_count_stats[cell_count_stats.cre_line==cre_line]


# In[115]:


cluster_id = 1
plotting.plot_pct_rel_to_chance_for_cluster(cre_counts, cluster_id, ax=None)


# ### population average trace

# In[116]:


# load dataframe with response traces
df_name = 'omission_response_df'
conditions = ['cell_specimen_id', 'omitted']

data_type = 'events'
event_type = 'omissions'
inclusion_criteria = 'active_only_closest_familiar_and_novel_containers_with_all_levels'


multi_session_df = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria)
print(len(multi_session_df.ophys_experiment_id.unique()))


cluster_mdf = multi_session_df.merge(cluster_meta[['cluster_id']], 
                                     on='cell_specimen_id', how = 'inner')


# In[118]:





# In[119]:


plotting.plot_population_average_response_for_cluster(cluster_mdf, cre_line, cluster_id, change=False, omitted=True, ax=None)


# ### plot clusters, fraction per area depth, and pop avgs as rows

# In[220]:


# get fraction cells relative to chance per cluster per cre_line
cell_count_stats = processing.get_cell_count_stats(cluster_meta)
# get fraction of cells per area/depth per cluster per cre_line 
fraction_cells = processing.get_fraction_cells_per_area_depth(cluster_meta)


# In[221]:


for cre_line in cre_lines: 
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df,
                                              cell_count_stats, fraction_cells, cre_line, 
                                              sort_order=None, suffix='_size_order', 
                                              save_dir=base_dir, folder=folder, )


# ### use manual sort order

# In[222]:


# manual_sort_order = processing.get_manual_sort_order()

# for cre_line in cre_lines: 
#     plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df,
#                                               cell_count_stats, fraction_cells, cre_line, 
#                                               sort_order=manual_sort_order, suffix='_manual_sort', 
#                                               save_dir=base_dir, folder=folder, )


# ### sort by visp upper 

# In[241]:


# get fraction cells relative to chance per cluster per cre_line
cell_count_stats = processing.get_cell_count_stats(cluster_meta)
# get fraction of cells per area/depth per cluster per cre_line 
fraction_cells = processing.get_fraction_cells_per_area_depth(cluster_meta)


# In[242]:


cell_count_stats['location'] = cell_count_stats.targeted_structure+'_'+cell_count_stats.layer
cluster_order = processing.get_cluster_order_for_metric_location(cell_count_stats, cluster_meta, location='VISp_upper', metric='relative_to_random')  

for cre_line in cre_lines: 
    plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df,
                                              cell_count_stats, fraction_cells, cre_line, 
                                              sort_order=cluster_order, suffix='_VISp_upper_sort', 
                                              save_dir=base_dir, folder=folder, )


# ### sort by V1 vs LM order

# In[251]:


cell_count_stats_area = processing.get_cell_count_stats(cluster_meta, conditions_to_groupby = ['targeted_structure'])

value_to_plot = 'relative_to_random'
cbar_label = 'fraction cells\nrel. to chance'
suffix = '_limited_range_area_sort'

cluster_order = processing.get_cluster_order_for_metric_location(cell_count_stats_area, cluster_meta, 
                                                                 location='VISp', metric='relative_to_random')  

plotting.plot_cell_stats_per_cluster_for_areas_depths(cluster_meta, cell_count_stats_area, n_clusters_cre,
                                                      value_to_plot, cbar_label, cmap='PRGn', vmin=-1, vmax=1, 
                                                      cluster_order=cluster_order, save_dir=base_dir, folder=folder, suffix=suffix)


# In[292]:


# cell_count_stats_area['location'] = cell_count_stats_area.targeted_structure
# # need to have both a 'layer' and 'targeted_structure' column for plot to work
# cell_count_stats_area['layer'] = cell_count_stats_area.targeted_structure

# cluster_order = processing.get_cluster_order_for_metric_location(cell_count_stats_area, cluster_meta, 
#                                                                  location='VISp', metric='relative_to_random')  

# for cre_line in cre_lines: 
#     plotting.plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df,
#                                               cell_count_stats_area, fraction_cells, cre_line, 
#                                               sort_order=cluster_order, suffix='_area_sort', 
#                                               save_dir=base_dir, folder=folder, )


# In[ ]:





# In[ ]:





# In[ ]:





# ### plot as columns

# In[227]:


# for cre_line in cre_lines: 
#     plotting.plot_clusters_stats_pop_avg_cols(cluster_meta, feature_matrix, multi_session_df,
#                                               cell_count_stats, fraction_cells, cre_line, 
#                                               sort_order=cluster_order, suffix='_VISp_upper_sort', 
#                                               save_dir=base_dir, folder=folder, )


# ### within cluster correlations sorted by VISp upper

# In[147]:


sort_order = processing.get_cluster_order_for_metric_location(cell_count_stats, cluster_meta, location='VISp_upper', metric='relative_to_random')  

plotting.plot_within_cluster_correlations_all_cre(cluster_meta, n_clusters_cre, sort_order=sort_order, 
                                                  suffix='_VISp_upper_sort', save_dir=base_dir, folder=folder)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Compute selectivity metrics on dropout scores

# ### define stats

# ### compute stats across cells 

# * positive value of exp_mod_direction means stronger coding of pref feature in novel session 
# * positive value of exp_mod_persistence means similar coding in Novel 1 and Novel >1 for pref feature
# * high value of feature_sel_within_session means highly feature selective in strongest exp level
# * low value of feature_sel_within_session means similar strength of coding for multiple features in a given session
# * high value of feature_sel_across sessions means there is a big difference between the strength of coding for the preferred feature in the preferred experience level and a different feature in a different experience level 
# * low value of feature_sel_across sessions means similar coding across two different features in two different sessions
# * cell switching is indicated by low feature_sel_across_sessions and high experience selectivity
# * lack of coding overall would be low exp selectivity and low feature_sel_across_sessions

# In[230]:


cell_stats = pd.DataFrame()
for i, cell_specimen_id in enumerate(cluster_meta.index.values):
    cell_dropouts = dropouts[dropouts.cell_specimen_id==cell_specimen_id].groupby('experience_level').mean()[features]
    stats = processing.get_coding_metrics(index_dropouts=cell_dropouts, index_value=cell_specimen_id, index_name='cell_specimen_id')
    cell_stats = pd.concat([cell_stats, stats], sort=False)
cell_stats = cell_stats.merge(cluster_meta, on='cell_specimen_id')
metrics = stats.keys()[-6:]


# ### average stats across cells per cluster

# In[231]:


avg_cluster_stats = cell_stats.groupby(['cre_line', 'cluster_id']).mean()
avg_cluster_stats = avg_cluster_stats[list(metrics)]
avg_cluster_stats = avg_cluster_stats.reset_index()
n_cells = cell_stats.groupby(['cre_line', 'cluster_id']).count()[['labels']].rename(columns={'labels':'n_cells'})
avg_cluster_stats = avg_cluster_stats.merge(n_cells, on=['cre_line', 'cluster_id'])


# ### compute stats on each cluster directly

# In[232]:


cluster_stats = pd.DataFrame()
for cre_line in cre_lines:
    # get cell specimen ids for this cre line
    cre_cell_specimen_ids = cluster_meta[cluster_meta.cre_line==cre_line].index.values
    # get cluster labels dataframe for this cre line
    cre_cluster_ids = cluster_meta.loc[cre_cell_specimen_ids]
    # get unique cluster labels for this cre line
    cluster_labels = np.sort(cre_cluster_ids.cluster_id.unique())
    # limit dropouts df to cells in this cre line
    feature_matrix_cre = feature_matrix.loc[cre_cell_specimen_ids]
    for i, cluster_id in enumerate(cluster_labels):
        # get cell specimen ids in this cluster in this cre line
        this_cluster_csids = cre_cluster_ids[cre_cluster_ids['cluster_id'] == cluster_id].index.values
        # get dropout scores for cells in this cluster in this cre line
        mean_dropout_df = np.abs(feature_matrix_cre.loc[this_cluster_csids].mean().unstack())
        stats = processing.get_coding_metrics(index_dropouts=mean_dropout_df.T, index_value=cluster_id, index_name='cluster_id')
        fraction_cre = len(this_cluster_csids) / float(len(cre_cell_specimen_ids))
        stats['fraction_cre'] = fraction_cre
        stats['cre_line'] = cre_line
        stats['F_max'] = mean_dropout_df['Familiar'].max()
        stats['N1_max'] = mean_dropout_df['Novel 1'].max()
        stats['N2_max'] = mean_dropout_df['Novel >1'].max()
        cluster_stats = pd.concat([cluster_stats, stats])
cluster_stats = cluster_stats.reset_index()
n_cells = cell_stats.groupby(['cre_line', 'cluster_id']).count()[['labels']].rename(columns={'labels':'n_cells_cluster'})
cluster_stats = cluster_stats.merge(n_cells, on=['cre_line', 'cluster_id'])


# In[ ]:





# ### merge cluster stats with cell counts per cluster

# In[149]:


# create location column merging area and depth
cell_count_stats['location'] = cell_count_stats.targeted_structure+'_'+cell_count_stats.layer
# group & unstack to get fraction relative to random for each location as columns
fraction_cells = cell_count_stats[['cre_line', 'cluster_id', 'location', 'relative_to_random']].groupby(['cre_line', 'cluster_id', 'location']).mean().unstack()
# get rid of multi-index column
fraction_cells.columns = fraction_cells.columns.droplevel()
# merge fraction cells per location with other cluster metrics
cluster_stats = fraction_cells.reset_index().merge(cluster_stats, on=['cre_line', 'cluster_id'])


# In[150]:


cluster_stats.head()


# In[ ]:





# ### plot each relationship on its own fig

# In[152]:


# for col_x in cluster_stats.columns[2:]:
#     for col_y in cluster_stats.columns[2:]:
#         try:
#             figsize=(4,4)
#             fig, ax = plt.subplots(figsize=figsize)
#             ax = sns.scatterplot(data=cluster_stats, x=col_x, y=col_y, size='fraction_cre', sizes=(0, 1000), 
#                                 hue='cre_line', hue_order=cre_lines, palette=colors, ax=ax)
#             ax.axvline(x=0, ymin=-1, ymax=1, linestyle='--', color='gray')
#             ax.axhline(y=0, xmin=-1, xmax=1, linestyle='--', color='gray')
#             ax.get_legend().remove()
#             plt.axis("equal")
#             for i in range(len(cluster_stats[col_x].values)):
#                 xy = (cluster_stats[col_x].values[i], cluster_stats[col_y].values[i])
#                 color_index = cluster_stats['color_index'].values[i]
#                 cluster_id = cluster_stats['cluster_id'].values[i]
#                 plt.annotate(cluster_id, xy, color=colors[color_index])
#             utils.save_figure(fig, figsize, save_dir, 'scatterplots', col_x+'_'+col_y)
#         except:
#             pass


# ### barplots of metric values for clusters

# In[162]:


cluster_order = processing.get_cluster_order_for_metric_location(cell_count_stats, cluster_meta, location='VISp_upper', metric='relative_to_random')  
n_clusters = [n_clusters_cre[cre] for cre in cre_lines]

for metric in cluster_stats.columns[2:]:
    try:
        figsize=(15,2.5)
        fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios':n_clusters}, sharey=True)
        for i, cre_line in enumerate(processing.get_cre_lines(cluster_meta)):
            data = cluster_stats[cluster_stats.cre_line==cre_line]
            ax[i] = sns.barplot(data=data, x='cluster_id', order=cluster_order[cre_line], y=metric, 
                                hue='VISp_upper', palette='PRGn', ax=ax[i], dodge=False)
            ax[i].set_title(processing.get_cell_type_for_cre_line(cluster_meta, cre_line))
            ax[i].get_legend().remove()
        plt.subplots_adjust(wspace=0.4)
        utils.save_figure(fig, figsize, save_dir, 'pointplots_VISp_upper_sort', metric)
    except:
        print('problem for', metric)


# In[ ]:





# ### relationship of across session switching and Familiar session max to % cells in VISp or VISl

# In[176]:


col_x = 'feature_sel_across_sessions'
col_y = 'VISp_upper'

colors = utils.get_cre_line_colors()

figsize=(4,4)
fig, ax = plt.subplots(figsize=figsize)
ax = sns.scatterplot(data=cluster_stats, x=col_x, y=col_y, size='fraction_cre', sizes=(0, 1000), 
                    hue='cre_line', hue_order=cre_lines, palette=colors, ax=ax)
ax.axvline(x=0, ymin=-1, ymax=1, linestyle='--', color='gray')
ax.axhline(y=0, xmin=-1, xmax=1, linestyle='--', color='gray')
ax.get_legend().remove()
plt.axis("equal")
# for i in range(len(cluster_stats[col_x].values)):
#     xy = (cluster_stats[col_x].values[i], cluster_stats[col_y].values[i])
#     color_index = cluster_stats['color_index'].values[i]
#     cluster_id = cluster_stats['cluster_id'].values[i]
#     plt.annotate(cluster_id, xy, color=colors[color_index])
utils.save_figure(fig, figsize, save_dir, 'scatterplots', col_x+'_'+col_y)


# In[177]:


col_x = 'F_max'
col_y = 'VISp_upper'

colors = utils.get_cre_line_colors()

figsize=(4,4)
fig, ax = plt.subplots(figsize=figsize)
ax = sns.scatterplot(data=cluster_stats, x=col_x, y=col_y, size='fraction_cre', sizes=(0, 1000), 
                    hue='cre_line', hue_order=cre_lines, palette=colors, ax=ax)
ax.axvline(x=0, ymin=-1, ymax=1, linestyle='--', color='gray')
ax.axhline(y=0, xmin=-1, xmax=1, linestyle='--', color='gray')
ax.get_legend().remove()
plt.axis("equal")
# for i in range(len(cluster_stats[col_x].values)):
#     xy = (cluster_stats[col_x].values[i], cluster_stats[col_y].values[i])
#     color_index = cluster_stats['color_index'].values[i]
#     cluster_id = cluster_stats['cluster_id'].values[i]
#     plt.annotate(cluster_id, xy, color=colors[color_index])
utils.save_figure(fig, figsize, save_dir, 'scatterplots', col_x+'_'+col_y)


# In[179]:


col_x = 'F_max'
col_y = 'VISl_upper'

colors = utils.get_cre_line_colors()

figsize=(4,4)
fig, ax = plt.subplots(figsize=figsize)
ax = sns.scatterplot(data=cluster_stats, x=col_x, y=col_y, size='fraction_cre', sizes=(0, 1000), 
                    hue='cre_line', hue_order=cre_lines, palette=colors, ax=ax)
ax.axvline(x=0, ymin=-1, ymax=1, linestyle='--', color='gray')
ax.axhline(y=0, xmin=-1, xmax=1, linestyle='--', color='gray')
ax.get_legend().remove()
plt.axis("equal")
# for i in range(len(cluster_stats[col_x].values)):
#     xy = (cluster_stats[col_x].values[i], cluster_stats[col_y].values[i])
#     color_index = cluster_stats['color_index'].values[i]
#     cluster_id = cluster_stats['cluster_id'].values[i]
#     plt.annotate(cluster_id, xy, color=colors[color_index])
utils.save_figure(fig, figsize, save_dir, 'scatterplots', col_x+'_'+col_y)


# In[ ]:





# ### boxplots of metric values for all cells in each cluster

# #### first merge with fraction cells info per cluster

# In[233]:


# create location column merging area and depth
cell_count_stats['location'] = cell_count_stats.targeted_structure+'_'+cell_count_stats.layer
# group & unstack to get fraction relative to random for each location as columns
fraction_cells = cell_count_stats[['cre_line', 'cluster_id', 'location', 'relative_to_random']].groupby(['cre_line', 'cluster_id', 'location']).mean().unstack()
# get rid of multi-index column
fraction_cells.columns = fraction_cells.columns.droplevel()
# merge fraction cells per location with other cluster metrics
cell_stats_loc = fraction_cells.reset_index().merge(cell_stats.reset_index(), on=['cre_line', 'cluster_id'])
cell_stats_loc = cell_stats_loc.set_index('cell_specimen_id')


# In[235]:


cluster_order = processing.get_cluster_order_for_metric_location(cell_count_stats, cluster_meta, location='VISp_upper', metric='relative_to_random')  
n_clusters = [n_clusters_cre[cre] for cre in cre_lines]

for metric in cluster_stats.columns[2:]:
    try:
        figsize=(15,2.5)
        fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios':n_clusters}, sharey=True)
        for i, cre_line in enumerate(processing.get_cre_lines(cluster_meta)):
            data = cell_stats_loc[cell_stats_loc.cre_line==cre_line]
            ax[i] = sns.boxplot(data=data, x='cluster_id', order=cluster_order[cre_line], y=metric, 
                                hue='VISp_upper', palette='PRGn', ax=ax[i], dodge=False)
            ax[i].set_title(processing.get_cell_type_for_cre_line(cluster_meta, cre_line))
            ax[i].get_legend().remove()
        plt.subplots_adjust(wspace=0.4)
        utils.save_figure(fig, figsize, save_dir, 'boxplots_VISp_upper_sort', metric)
    except:
        print('problem for', metric)


# In[ ]:





# ### boxplots using V1/LM or upper/lower

# In[282]:


cell_count_stats_area = processing.get_cell_count_stats(cluster_meta, conditions_to_groupby = ['targeted_structure'])


# In[285]:


# create location column merging area and depth
cell_count_stats_area['location'] = cell_count_stats_area.targeted_structure
# group & unstack to get fraction relative to random for each location as columns
fraction_cells = cell_count_stats_area[['cre_line', 'cluster_id', 'location', 'relative_to_random']].groupby(['cre_line', 'cluster_id', 'location']).mean().unstack()
# get rid of multi-index column
fraction_cells.columns = fraction_cells.columns.droplevel()
# merge fraction cells per location with other cluster metrics
cell_stats_loc = fraction_cells.reset_index().merge(cell_stats.reset_index(), on=['cre_line', 'cluster_id'])
cell_stats_loc = cell_stats_loc.set_index('cell_specimen_id')


# In[288]:


cluster_order = processing.get_cluster_order_for_metric_location(cell_count_stats_area, cluster_meta, 
                                                                 location='VISp', metric='relative_to_random')  
n_clusters = [n_clusters_cre[cre] for cre in cre_lines]

for metric in cluster_stats.columns[2:]:
    try:
        figsize=(15,2.5)
        fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios':n_clusters}, sharey=True)
        for i, cre_line in enumerate(processing.get_cre_lines(cluster_meta)):
            data = cell_stats_loc[cell_stats_loc.cre_line==cre_line]
            ax[i] = sns.boxplot(data=data, x='cluster_id', order=cluster_order[cre_line], y=metric, 
                                hue='VISp', palette='PRGn', ax=ax[i], dodge=False)
            ax[i].set_title(processing.get_cell_type_for_cre_line(cluster_meta, cre_line))
            ax[i].get_legend().remove()
        plt.subplots_adjust(wspace=0.4)
        utils.save_figure(fig, figsize, save_dir, 'boxplots_VISp_sort', metric)
    except:
        print('problem for', metric)


# ### layer

# In[289]:


cell_count_stats_area = processing.get_cell_count_stats(cluster_meta, conditions_to_groupby = ['layer'])


# In[290]:


# create location column merging area and depth
cell_count_stats_area['location'] = cell_count_stats_area.layer
# group & unstack to get fraction relative to random for each location as columns
fraction_cells = cell_count_stats_area[['cre_line', 'cluster_id', 'location', 'relative_to_random']].groupby(['cre_line', 'cluster_id', 'location']).mean().unstack()
# get rid of multi-index column
fraction_cells.columns = fraction_cells.columns.droplevel()
# merge fraction cells per location with other cluster metrics
cell_stats_loc = fraction_cells.reset_index().merge(cell_stats.reset_index(), on=['cre_line', 'cluster_id'])
cell_stats_loc = cell_stats_loc.set_index('cell_specimen_id')


# In[291]:


cluster_order = processing.get_cluster_order_for_metric_location(cell_count_stats_area, cluster_meta, 
                                                                 location='upper', metric='relative_to_random')  
n_clusters = [n_clusters_cre[cre] for cre in cre_lines]

for metric in cluster_stats.columns[2:]:
    try:
        figsize=(15,2.5)
        fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios':n_clusters}, sharey=True)
        for i, cre_line in enumerate(processing.get_cre_lines(cluster_meta)):
            data = cell_stats_loc[cell_stats_loc.cre_line==cre_line]
            ax[i] = sns.boxplot(data=data, x='cluster_id', order=cluster_order[cre_line], y=metric, 
                                hue='upper', palette='PRGn', ax=ax[i], dodge=False)
            ax[i].set_title(processing.get_cell_type_for_cre_line(cluster_meta, cre_line))
            ax[i].get_legend().remove()
        plt.subplots_adjust(wspace=0.4)
        utils.save_figure(fig, figsize, save_dir, 'boxplots_upper_sort', metric)
    except:
        print('problem for', metric)


# In[ ]:





# In[ ]:





# ## Summary plots experimentation

# In[ ]:





# ### plot experience mod values for cluster average

# In[773]:


figsize = (17,5)
fig, ax = plt.subplots(1,3, figsize=figsize)
for i, cre_line in enumerate(cre_lines):
    data = avg_cluster_stats[avg_cluster_stats.cre_line==cre_line]
    data = data[data.cluster_id<10] # TEMPORARY to make sure we dont exceed # clusters in palette
    cluster_order = np.sort(data.cluster_id.unique())
    n_clusters = len(cluster_order)
    print(n_clusters)
    ax[i] = sns.scatterplot(data=data, x='exp_mod_direction', y='exp_mod_persistence', 
                            hue='cluster_id', hue_order=cluster_order, palette=sns.color_palette()[:n_clusters], 
                            size='n_cells', sizes=(0, 1000), ax=ax[i])
    ax[i].axvline(x=0, ymin=-1, ymax=1, linestyle='--', color='gray')
    ax[i].axhline(y=0, xmin=-1, xmax=1, linestyle='--', color='gray')
    ax[i].set_title(cell_types[cre_line]+'\n experience modulation')
    ax[i].legend(fontsize='small', title_fontsize='small', bbox_to_anchor=(1,1))
    ax[i].set_xlabel('direction')
    ax[i].set_ylabel('persistence')
    ax[i].set_xlim(-1.2,1.2)
    ax[i].set_ylim(-1.2,1.2)
fig.tight_layout()
# utils.save_figure(fig, figsize, base_dir, folder, 'experience_modulation_avg_across_cells_in_cluster')


# ### cells and clusters together 

# In[772]:


figsize = (15,5)
fig, ax = plt.subplots(1,3, figsize=figsize)
for i, cre_line in enumerate(cre_lines):
    data = cell_stats[cell_stats.cre_line==cre_line]
    data = data[data.cluster_id<10] # TEMPORARY to make sure we dont exceed # clusters in palette
    cluster_order = np.sort(data.cluster_id.unique())
    n_clusters = len(cluster_order)
    ax[i] = sns.scatterplot(data=data, x='exp_mod_direction', y='exp_mod_persistence', alpha=0.2,
                            hue='cluster_id', hue_order=cluster_order, palette=sns.color_palette()[:n_clusters], ax=ax[i])
    
    data = avg_cluster_stats[avg_cluster_stats.cre_line==cre_line]
    data = data[data.cluster_id<10] # TEMPORARY to make sure we dont exceed # clusters in palette
    cluster_order = np.sort(data.cluster_id.unique())
    n_clusters = len(cluster_order)
    ax[i] = sns.scatterplot(data=data, x='exp_mod_direction', y='exp_mod_persistence', 
                            hue='cluster_id', hue_order=cluster_order, palette=sns.color_palette()[:n_clusters], 
                            size='n_cells', sizes=(0, 1000), ax=ax[i], alpha=0.7)
    
    ax[i].axvline(x=0, ymin=-1, ymax=1, linestyle='--', color='gray')
    ax[i].axhline(y=0, xmin=-1, xmax=1, linestyle='--', color='gray')
    ax[i].set_title(cell_types[cre_line]+'\n experience modulation')
    ax[i].legend(fontsize='small', title_fontsize='small', bbox_to_anchor=(1,1))
    ax[i].set_xlabel('direction')
    ax[i].set_ylabel('persistence')
    ax[i].set_xlim(-1.2,1.2)
    ax[i].set_ylim(-1.2,1.2)
fig.tight_layout()
# utils.save_figure(fig, figsize, base_dir, folder, 'experience_modulation_across_cells_and_clusters')


# In[ ]:





# ### feature selectivity within and across sesions

# In[216]:


# figsize = (15,5)
# fig, ax = plt.subplots(1,3, figsize=figsize)
# for i, cre_line in enumerate(cre_lines):
#     data = cell_stats[cell_stats.cre_line==cre_line]
#     data = data[data.cluster_id<10] # TEMPORARY to make sure we dont exceed # clusters in palette
#     cluster_order = np.sort(data.cluster_id.unique())
#     n_clusters = len(cluster_order)
#     ax[i] = sns.scatterplot(data=data, x='feature_sel_within_session', y='feature_sel_across_sessions', alpha=0.2,
#                             hue='cluster_id', hue_order=cluster_order, palette=sns.color_palette()[:n_clusters], ax=ax[i])
    
#     data = avg_cluster_stats[avg_cluster_stats.cre_line==cre_line]
#     data = data[data.cluster_id<10] # TEMPORARY to make sure we dont exceed # clusters in palette
#     cluster_order = np.sort(data.cluster_id.unique())
#     n_clusters = len(cluster_order)
#     ax[i] = sns.scatterplot(data=data, x='feature_sel_within_session', y='feature_sel_across_sessions', 
#                             hue='cluster_id', hue_order=cluster_order, palette=sns.color_palette()[:n_clusters], 
#                             size='n_cells', sizes=(0, 1000), ax=ax[i], alpha=0.7)
    
# #     ax[i].axvline(x=0, ymin=0, ymax=1, linestyle='--', color='gray')
# #     ax[i].axhline(y=0, xmin=-1, xmax=1, linestyle='--', color='gray')
#     ax[i].set_title(cell_types[cre_line]+'\n feature selectivity')
#     ax[i].legend(fontsize='small', title_fontsize='small', bbox_to_anchor=(1,1))
#     ax[i].set_xlabel('within session')
#     ax[i].set_ylabel('across sessions')
#     ax[i].set_xlim(-0.1,1.1)
#     ax[i].set_ylim(-0.1,1.1)
# fig.tight_layout()
# # utils.save_figure(fig, figsize, base_dir, folder, 'feature_selectivity_across_cells_and_clusters')


# ### feature selectivity across sessions vs. experience selectivity 

# * cell switching is indicated by low feature_sel_across_sessions and high experience selectivity
# 

# In[213]:


figsize = (17,5)
fig, ax = plt.subplots(1,3, figsize=figsize)
for i, cre_line in enumerate(cre_lines):
    data = avg_cluster_stats[avg_cluster_stats.cre_line==cre_line]
    data = data[data.cluster_id<10] # TEMPORARY to make sure we dont exceed # clusters in palette
    cluster_order = np.sort(data.cluster_id.unique())
    n_clusters = len(cluster_order)
    print(n_clusters)
    ax[i] = sns.scatterplot(data=data, x='experience_selectivity', y='feature_sel_across_sessions', 
                            hue='cluster_id', hue_order=cluster_order, palette=sns.color_palette()[:n_clusters], 
                            size='n_cells', sizes=(0, 1000), ax=ax[i])
#     ax[i].axvline(x=0, ymin=-1, ymax=1, linestyle='--', color='gray')
#     ax[i].axhline(y=0, xmin=-1, xmax=1, linestyle='--', color='gray')
    ax[i].set_title(cell_types[cre_line])
    ax[i].legend(fontsize='small', title_fontsize='small', bbox_to_anchor=(1,1))
    ax[i].set_xlabel('experience modulation')
    ax[i].set_ylabel('feature selectivity across sessions')
    ax[i].set_xlim(0, 1.1)
    ax[i].set_ylim(0, 1.1)
fig.tight_layout()
# utils.save_figure(fig, figsize, base_dir, folder, 'feature_sel_exp_mod_avg_across_cells_in_cluster')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### summarize

# In[192]:


n_clusters_per_cre = cell_stats.groupby(['cre_line']).count().rename(columns={'cluster_id':'n_cells_total'})[['n_cells_total']]
n_clusters_per_feature = cell_stats.groupby(['cre_line', 'dominant_feature']).count().rename(columns={'cluster_id':'n_cells'})[['n_cells']]
n_clusters_per_feature = n_clusters_per_feature.reset_index().merge(n_clusters_per_cre, on='cre_line')
n_clusters_per_feature['fraction_cells'] = n_clusters_per_feature['n_cells']/n_clusters_per_feature['n_cells_total']


# In[193]:


colors = utils.get_cre_line_colors()[::-1]

figsize=(4.5,3)
fig, ax = plt.subplots(figsize=figsize)
sns.barplot(data=n_clusters_per_feature, x='dominant_feature', y='fraction_cells', hue='cre_line',
             palette=colors, order=features, hue_order=cre_lines, ax=ax)
ax.legend(fontsize='xx-small', title='', loc='upper right')
ax.set_ylabel('fraction of cells')
ax.set_xlabel('')
ax.set_title('feature preference')
ax.set_xticklabels(features, rotation=45)


# In[194]:


n_clusters_per_cre = cell_stats.groupby(['cre_line']).count().rename(columns={'cluster_id':'n_cells_total'})[['n_cells_total']]
n_clusters_per_feature = cell_stats.groupby(['cre_line', 'dominant_experience_level']).count().rename(columns={'cluster_id':'n_cells'})[['n_cells']]
n_clusters_per_feature = n_clusters_per_feature.reset_index().merge(n_clusters_per_cre, on='cre_line')
n_clusters_per_feature['fraction_cells'] = n_clusters_per_feature['n_cells']/n_clusters_per_feature['n_cells_total']


# In[195]:


colors = utils.get_cre_line_colors()[::-1]
experience_levels = np.sort(cluster_meta.experience_level.unique())

figsize=(4.5,3)
fig, ax = plt.subplots(figsize=figsize)
sns.barplot(data=n_clusters_per_feature, x='dominant_experience_level', y='fraction_cells', hue='cre_line',
             palette=colors, order=experience_levels, hue_order=cre_lines, ax=ax)
ax.legend(fontsize='xx-small', title='', loc='upper right')
ax.set_ylabel('fraction of cells')
ax.set_xlabel('')
ax.set_title('experience level preference')
ax.set_xticklabels(experience_levels, rotation=45);


# ### repeat but per cluster instead of across cells 

# In[189]:


cell_stats = cluster_meta.copy()
for i, cell_specimen_id in enumerate(cell_stats.index.values):
    # get dropout scores per cell 
    cell_dropouts = dropouts[dropouts.cell_specimen_id==cell_specimen_id].groupby('experience_level').mean()[features]
    # get preferred regressor and experience level and save
    dominant_feature = cell_dropouts.stack().idxmax()[1]
    dominant_experience_level = cell_dropouts.stack().idxmax()[0]
    cell_stats.loc[cell_specimen_id, 'dominant_feature'] = dominant_feature
    cell_stats.loc[cell_specimen_id, 'dominant_experience_level'] = dominant_experience_level
    # get selectivity for feature & experience level 
    # feature selectivity is ratio of largest and next largest dropouts for the dominant experience level
    order = np.argsort(cell_dropouts.loc[dominant_experience_level])
    values = cell_dropouts.loc[dominant_experience_level].values[order[::-1]]
    feature_selectivity = (values[0]-(np.mean(values[1:])))/(values[0]+(np.mean(values[1:])))
    # experience selectivity is ratio of largest and next largest dropouts for the dominant feature
    order = np.argsort(cell_dropouts[dominant_feature])
    values = cell_dropouts[dominant_feature].values[order[::-1]]
    experience_selectivity = (values[0]-(np.mean(values[1:])))/(values[0]+(np.mean(values[1:])))
    cell_stats.loc[cell_specimen_id, 'feature_selectivity'] = feature_selectivity
    cell_stats.loc[cell_specimen_id, 'experience_selectivity'] = experience_selectivity


# In[202]:


cluster_stats.keys()


# In[210]:


min_size = 0
max_size = 1000

colors = utils.get_cre_line_colors()

figsize=(4,3)
fig, ax = plt.subplots(figsize=figsize)
sns.scatterplot(data=cluster_stats, x='cre_line', y='feature_selectivity', hue='cre_line',
             palette=colors, hue_order=cell_types, size='fraction_cre', sizes=(min_size, max_size), ax=ax)
ax.legend(fontsize='xx-small', title='', bbox_to_anchor=(1.1,1))
ax.set_ylabel('feature selectivity index')
ax.set_xlabel('')
ax.set_title('feature selectivity')
# ax.set_xticklabels([processing.get_cell_type_for_cre_line(cluster_meta, cre_line) for cre_line in cre_lines], rotation=45)
ax.set_xticklabels([cre_line.split('-')[0] for cre_line in cre_lines[::-1]], rotation=0)

ax.set_ylim(0,1)


# In[212]:


max_size = 0
max_size = 1000

figsize=(4,3)
fig, ax = plt.subplots(figsize=figsize)
sns.scatterplot(data=cluster_stats, x='cre_line', y='experience_selectivity', hue='cre_line', 
             palette=colors, hue_order=cell_types, size='fraction_cre', sizes=(min_size, max_size), ax=ax)
ax.legend(fontsize='x-small', title='', bbox_to_anchor=(1.1,1))
ax.set_ylabel('experience selectivity')
ax.set_xlabel('')
ax.set_title('experience selectivity')
# ax.set_xticklabels([processing.get_cell_type_for_cre_line(cluster_meta, cre_line) for cre_line in cre_lines], rotation=45)
ax.set_xticklabels([cre_line.split('-')[0] for cre_line in cre_lines[::-1]], rotation=0)

ax.set_ylim(0,1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### details stuff / validation

# #### count number of cells in different areas & depths

# In[ ]:


make_frequency_table(cluster_meta, groupby_columns = ['cre_line', 'targeted_structure'], normalize=False)


# There are way more cells in VISp than VISl

# In[ ]:


make_frequency_table(cluster_meta, groupby_columns = ['cre_line', 'layer'], normalize=False)


# There are way more cells in lower layers for Sst and upper layers for Vip

# In[ ]:


make_frequency_table(cluster_meta, groupby_columns = ['cre_line', 'binned_depth'], normalize=False)


# Numbers get pretty small for inhibitory lines when looking at depths in 4 bins

# #### get frequency across areas & layer for one cre line

# In[ ]:


cre_line = cre_lines[1]
print(cre_line)


# In[ ]:


make_frequency_table(cluster_meta[cluster_meta.cre_line==cre_line], 
                     groupby_columns = ['targeted_structure', 'layer'], normalize=False)


# In[ ]:


cre_meta = cluster_meta[cluster_meta.cre_line==cre_line]
make_frequency_table(cre_meta, groupby_columns = ['targeted_structure', 'layer'], normalize=True)


# In[ ]:





# ### plot frequency by area and depth

# In[ ]:


cre_line = cre_lines[1]
cre_meta = cluster_meta[cluster_meta.cre_line==cre_line]
frequency = make_frequency_table(cre_meta, groupby_columns = ['targeted_structure', 'layer'], normalize=True)


# Rows add up to 1

# In[ ]:


fig, ax = plt.subplots(figsize=(6,2.5))
ax = sns.heatmap(frequency, vmin=0, cmap='magma', ax=ax, cbar_kws={'shrink':0.8, 'label':'fraction of cells'})
ax.set_ylim((0, 4))
# ax.set_yticklabels(frequency.index, rotation=0, horizontalalignment='center')
ax.set_xlim(-0.5, len(frequency.columns)+0.5)
ax.set_ylabel('')
ax.set_title(cell_types[cre_line])
fig.tight_layout()


# ### normalizing to cluster size doesnt make sense

# In[ ]:


stats_df = cre_meta[['cluster_id', 'binned_depth']]
frequency_table= stats_df.groupby('cluster_id')['binned_depth'].value_counts(normalize=False).unstack()
frequency_table= frequency_table.fillna(0)
frequency_table


# In[ ]:


stats_df = cre_meta[['cluster_id', 'binned_depth']]
frequency_table= stats_df.groupby('cluster_id')['binned_depth'].value_counts(normalize=True).unstack()
frequency_table= frequency_table.fillna(0)
frequency_table


# In[ ]:


sns.heatmap(frequency_table)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### plots with individual cells per cluster

# In[ ]:


# get dropouts for some specific condition and add to cre meta for plotting
condition = ('all-images', 'Familiar')
metric_data = df[condition]
metric_data = metric_data.rename(columns={('all-images', 'Novel 1'):'metric'})
metric_data = pd.DataFrame(metric_data, columns=['metric'])
metric_meta = cre_meta.merge(metric_data, on='cell_specimen_id')


# In[ ]:


fig, ax = plt.subplots()
ax = sns.scatterplot(data=metric_meta, y='imaging_depth', x='metric', hue='cluster_id', palette='magma_r', ax=ax)
ax.legend(bbox_to_anchor=(1,1), )


# In[ ]:


fig, ax = plt.subplots()
ax = sns.pointplot(data=metric_meta, y='binned_depth', x='metric', hue='cluster_id', 
                   orient='h', join=False, dodge=0.5, palette='magma_r', ax=ax)
ax.legend(bbox_to_anchor=(1,1), )


# In[ ]:


# get dropouts for some specific condition and add to cre meta for plotting
condition = ('behavioral', 'Novel 1')
metric_data = df[condition]
metric_data = metric_data.rename(columns={('all-images', 'Novel 1'):'metric'})
metric_data = pd.DataFrame(metric_data, columns=['metric'])
metric_meta = cre_meta.merge(metric_data, on='cell_specimen_id')


# In[ ]:


fig, ax = plt.subplots()
ax = sns.scatterplot(data=metric_meta, y='imaging_depth', x='metric', hue='cluster_id', palette='magma_r', ax=ax)
ax.legend(bbox_to_anchor=(1,1), )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


area_df = pd.DataFrame(frequency_table_area.unstack(), columns=['fraction']).reset_index()
area_df = area_df.groupby(['cluster_id', 'targeted_structure']).mean().unstack()
area_df.columns = area_df.columns.droplevel()
fig, ax = plt.subplots(figsize=(6,2))
ax = sns.heatmap(area_df.T, vmin=0, cmap='magma', ax=ax, cbar_kws={'shrink':0.8, 'label':'fraction cells\nper area'})
ax.set_ylim((0, 2))
ax.set_yticklabels(area_df.T.index, rotation=0, horizontalalignment='center')
ax.set_ylabel('')
ax.set_xlim(-0.5, len(area_df)+0.5)
fig.tight_layout()


# In[ ]:


depth_df = pd.DataFrame(frequency_table_depth.unstack(), columns=['fraction']).reset_index()
depth_df = depth_df.groupby(['cluster_id', 'binned_depth']).mean().unstack()
depth_df.columns = depth_df.columns.droplevel()

fig, ax = plt.subplots(figsize=(6,2.5))
ax = sns.heatmap(depth_df.T, vmin=0, cmap='magma', ax=ax, cbar_kws={'shrink':0.8, 'label':'fraction cells\nper depth'})
ax.set_ylim((0, 4))
ax.set_yticklabels(depth_df.T.index, rotation=0, horizontalalignment='center')
ax.set_xlim(-0.5, len(depth_df)+0.5)
ax.set_ylabel('')
fig.tight_layout()


# In[ ]:


fig, ax = plt.subplots
sns.barplot(data=area_df, x='cluster_id', y='fraction', hue='targeted_structure')


# In[ ]:


frequency_table_depth


# In[ ]:




