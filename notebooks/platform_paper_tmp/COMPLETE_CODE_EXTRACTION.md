# Complete Code Extraction from Figure 4 Scripts

## SUMMARY OF FILES ANALYZED

1. **251206_figure_4_updated_figures.py** - LATEST MAIN SCRIPT (4094 lines)
2. **241218_figure_4_updated_figures.py** - OLDER VERSION
3. **241218_figure_4_updated_figures_clean_run.py** - OLDER CLEAN RUN
4. **251206_figure_4_prediction_strength_control.py** (4094 lines)
5. **260228_clustering_shuffle_control.py** (744 lines)
6. **240605_familiar_only_clustering_control.py** (349 lines)
7. **240404_figure_4_generate_analysis_files.py** (379 lines)

---

## FILE 1: 251206_figure_4_updated_figures.py (LATEST MAIN)

### ALL IMPORT STATEMENTS
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

%load_ext autoreload
%autoreload 2
%matplotlib inline

import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.ophys.glm_example_plots as gep

from visual_behavior.dimensionality_reduction.clusering import plotting # functions to plot clusters
from visual_behavior.dimensionality_reduction.clustering import processing # function for computing and validating clusters

import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.platform_single_cell_examples as pse

import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_across_session as gas

import scipy.stats as stats
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
import umap
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
```

### CONFIGURATION VARIABLES AND DATA LOADING

```python
# Seaborn context
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

# Get utility information
experience_level_colors = utils.get_experience_level_colors()
cre_line_colors = utils.get_cre_line_colors()
experience_levels = utils.get_new_experience_levels()
cell_types = utils.get_cell_types()

# Cache and save directories
platform_cache_dir = loading.get_platform_analysis_cache_dir()
data_dir = os.path.join(platform_cache_dir, 'clustering')
save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_figures_final\PaperRevision_Neuron\figure_4'

# Load metadata tables
experiments_table = pd.read_csv(os.path.join(platform_cache_dir, 'all_ophys_experiments_table.csv'), index_col=0)
platform_experiments = pd.read_csv(os.path.join(platform_cache_dir, 'platform_paper_ophys_experiments_table.csv'), index_col=0)
platform_cells_table = pd.read_csv(os.path.join(platform_cache_dir, 'platform_paper_ophys_cells_table.csv'), index_col=0)
matched_cells_table = pd.read_csv(os.path.join(platform_cache_dir, 'platform_paper_matched_ophys_cells_table.csv'), index_col=0)

# Get lists of matched cells and experiments
matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()

# Get cre_lines and cell types for plot labels
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types = utils.get_cell_types()

# GLM results loading
glm_version = '24_events_all_L2_optimize_by_session'
run_params = pd.read_pickle(os.path.join(platform_cache_dir, 'glm_results', glm_version+'_run_params.pkl'))
results_pivoted = pd.read_hdf(os.path.join(platform_cache_dir, 'glm_results', 'across_session_normalized_platform_results_pivoted.h5'), key='df')
all_results = pd.read_hdf(os.path.join(platform_cache_dir, 'glm_results', 'all_results.h5'), key='df')

# Clustering results loading
n_clusters = 14
feature_matrix = pd.read_hdf(os.path.join(platform_cache_dir, 'clustering', 'clustering_feature_matrix.h5'), key='df')
feature_matrix = feature_matrix.rename(columns={'all-images':'images', 'behavioral': 'behavior'})
cluster_meta = pd.read_hdf(os.path.join(platform_cache_dir, 'clustering', 'cluster_metadata_'+str(n_clusters)+'_clusters.h5'), key='df')

cluster_order = np.sort(cluster_meta.cluster_id.unique())

# Cell response dataframes - Image and change responses
data_type = 'events'
interpolate = True
output_sampling_rate = 30
inclusion_criteria = 'platform_experiment_table'
event_type = 'all'
conditions = ['cell_specimen_id', 'is_change']
change_suffix = '_'+utils.get_conditions_string(data_type, conditions)+'_'+inclusion_criteria

multi_session_df = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria,
                                                        interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                         epoch_duration_mins=None)

change_mdf = multi_session_df[multi_session_df.is_change==True]
change_mdf = change_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')

image_mdf = multi_session_df[(multi_session_df.is_change==False)]
image_mdf = image_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')

# Omission responses
conditions = ['cell_specimen_id', 'omitted']
change_suffix = '_'+utils.get_conditions_string(data_type, conditions)+'_'+inclusion_criteria
omission_mdf = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria,
                                                        interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                         epoch_duration_mins=None)
omission_mdf = omission_mdf[omission_mdf.omitted==True]
omission_mdf = omission_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')

# Gap statistic parameters
metric = 'euclidean'
shuffle_type = 'all'
k_max = 25
gap_filename = os.path.join(data_dir, 'gap_scores_{}_{}_nb20_unshuffled_to_{}.pkl'.format(metric, glm_version, shuffle_type))
with open(gap_filename, 'rb') as f:
    gap_dict = pickle.load(f)

# Silhouette score parameters
n_clusters_range = np.arange(3, 25)
metric = 'euclidean'
n_boots = 20
silhouette_filename = os.path.join(data_dir, 'silhouette_score_{}_{}_clusters_metric_{}_nboots_{}.pkl'.format(n_clusters_range[0], n_clusters_range[-1], metric, n_boots))
if os.path.exists(silhouette_filename):
    with open(silhouette_filename, 'rb') as f:
        silhouette_scores, silhouette_std = pickle.load(f)
```

### CLUSTERING-RELATED PROCESSING FUNCTION CALLS

```python
# Eigengap loading
eigengap = processing.load_eigengap(glm_version, feature_matrix, cre_line='all', save_dir=data_dir, k_max=25)

# Within-cluster correlation
cluster_meta = processing.add_within_cluster_corr_to_cluster_meta(feature_matrix, cluster_meta, use_spearmanr=spearman)

# UMAP results
umap_df = processing.get_umap_results(feature_matrix, cluster_meta, save_dir=data_dir, suffix='')

# Remove outliers from response dataframes
threshold_percentile = 5
image_mdf_clean, image_outliers = processing.remove_outliers(image_mdf, threshold_percentile)
change_mdf_clean, change_outliers = processing.remove_outliers(change_mdf, threshold_percentile)
omission_mdf_clean, omission_outliers = processing.remove_outliers(omission_mdf, threshold_percentile)

# Generate metrics
metrics = processing.generate_merged_table_of_coding_score_and_model_free_metrics(cluster_meta, results_pivoted,
                                                            data_type='events',
                                                            session_subset='full_session',
                                                            inclusion_criteria='platform_experiment_table',
                                                            save_dir=data_dir)

cluster_metrics = processing.get_coding_score_metrics_for_clusters(cluster_meta, results_pivoted)

# Cluster proportion statistics
n_cells_table = processing.get_cluster_proportion_stats_for_locations(cluster_meta, location=location)
n_cells_mice = processing.get_cluster_proportion_stats_for_locations(tmp, location='mouse_id')
n_cells_exp_level = processing.get_cluster_proportion_stats_for_locations(cluster_meta, location='n_exposures_prior_to_novel+')

# Coding score metrics
coding_score_metrics = processing.generate_coding_score_metrics_table(cluster_meta, results_pivoted,
                                                                      save_dir=data_dir)

n_cells_per_cluster = processing.get_fraction_cells_per_cluster_per_group(cluster_meta, 'cre_line')
coding_metrics = processing.get_coding_score_metrics(cluster_meta, results_pivoted)

# Shuffle dropout scores
reference_df = processing.shuffle_dropout_score(data, shuffle_type=reference_shuffle, separate_cre_lines_for_shuffle=separate_cre_lines_for_shuffle)
```

### PLOTTING FUNCTION CALLS (plotting module)

```python
# Gap and Eigen gap plotting
plotting.plot_gap_statistic(gap_dict, cre_lines=['all'], n_clusters_cre=n_clusters,
                            tag='with_cre_shuffle', save_dir=save_dir, folder='selecting_k_clusters')

plotting.plot_eigengap_values(eigengap, cre_lines=['all'], n_clusters_cre={'all':n_clusters}, save_dir=save_dir, folder='selecting_k_clusters')

plotting.plot_silhouette_scores(silhouette_scores=silhouette_scores, silhouette_std=silhouette_std,
                                n_clusters=n_clusters, save_dir=save_dir, folder='selecting_k_clusters')

# Within-cluster correlation
plotting.plot_within_cluster_correlations(cluster_meta, sort_order=None, spearman=spearman, suffix='_'+str(n_clusters)+'_clusters',
                                         save_dir=save_dir, folder='clustering_results')

# UMAP plotting
plotting.plot_umap_for_clusters(cluster_meta, feature_matrix, umap_df, label_col='cluster_id', cre_lines = ['all'],
                                save_dir=save_dir, folder='cluster_results')

plotting.plot_umap_for_clusters_separately(cluster_meta, feature_matrix, umap_df, label_col='cluster_id', save_dir=save_dir, folder='clustering_results')

plotting.plot_umap_for_features_separately(cluster_meta, feature_matrix, umap_df, label_col='cell_type', cmap=cmap, save_dir=save_dir, folder='clustering_results')

plotting.plot_cluster_info(cre_lines, cluster_meta, save_dir=save_dir, folder='clustering_controls')

# Heatmaps and coding scores
plotting.plot_coding_score_heatmap_remapped(cluster_meta_tmp, feature_matrix, sort_by=None, session_colors=True,
                                            save_dir=save_dir, folder='clustering_results')

plotting.plot_cluster_means_remapped(feature_matrix, cluster_meta, save_dir=save_dir, folder='clustering_results')

plotting.plot_cre_line_means_remapped(feature_matrix, cluster_meta, save_dir=save_dir, folder='clustering_results')

plotting.plot_mean_cluster_heatmaps_remapped(feature_matrix, cluster_meta, sort_by='cluster_id',
                                             save_dir=save_dir, folder='clustering_results')

plotting.plot_coding_score_heatmap_matched(cluster_meta, feature_matrix, sort_by='cluster_id',
                                           save_dir=save_dir, folder='clustering_results')

# Population average response plotting
plotting.plot_population_average_response_for_clusters_as_rows_split(image_mdf_clean, event_type,
                                                                     save_dir=save_dir, folder='clustering_results')

plotting.plot_population_average_response_for_clusters_as_rows_split(change_mdf_clean, event_type, with_pre_change=False,
                                                                     save_dir=save_dir, folder='clustering_results')

plotting.plot_population_average_response_for_clusters_as_rows_split(change_mdf_clean, event_type, with_pre_change=True,
                                                                     save_dir=save_dir, folder='clustering_results')

plotting.plot_population_average_response_for_clusters_as_rows_split(omission_mdf_clean, event_type, with_pre_change=True,
                                                                     save_dir=save_dir, folder='clustering_results')

plotting.plot_population_average_response_for_clusters_as_rows_all_response_types(image_mdf_clean, change_mdf_clean, omission_mdf_clean,
                                                                                  save_dir=save_dir, folder='clustering_results')

plotting.plot_population_average_response_for_clusters_as_rows_all_response_types(image_mdf_clean_cre, change_mdf_clean_cre, omission_mdf_clean_cre,
                                                                                  save_dir=save_dir, folder='clustering_results')

# Cluster properties
plotting.plot_cluster_properties_combined(tmp, feature_matrix, cre_line,
                                         save_dir=save_dir, folder='clustering_results')

plotting.plot_cluster_properties_combined(cluster_meta, feature_matrix, cre_line,
                                         save_dir=save_dir, folder='clustering_results')

# Cell response heatmaps
plotting.plot_cell_response_heatmaps_for_clusters(image_mdf, data_type='events', event_type='images',
                                                 save_dir=save_dir, folder='clustering_results')

# Percent cells per cluster
plotting.plot_percent_cells_per_cluster_all_cre(cluster_meta, horiz=False, col_to_group='cre_line', save_dir=save_dir)
plotting.plot_percent_cells_per_cluster_all_cre(cluster_meta, col_to_group='cre_line', save_dir=save_dir)

plotting.plot_percent_cells_per_cluster_per_cre(cluster_meta, cluster_order=None, col_to_group='cre_line',
                                               save_dir=save_dir, folder='clustering_results')

plotting.plot_percent_cells_per_cluster_per_cre_dominant_feature(tmp, cluster_order=cluster_order,
                                                                 save_dir=save_dir, folder='clustering_results')

plotting.plot_percent_cells_per_cluster_per_cre_and_all(tmp, col_to_group='cre_line', save_dir=save_dir)

plotting.plot_percent_cells_per_cluster_per_cre_dominant_feature_xaxis(tmp, col_to_group='cre_line', match_height=True,
                                                                       save_dir=save_dir, folder='clustering_results')

# Depth distribution plotting
plotting.plot_cluster_depth_distribution_by_cre_line_separately(cluster_meta, location, cluster_order=None,
                                                               save_dir=save_dir, folder='clustering_results')

plotting.plot_cluster_depth_distribution_by_cre_lines(cluster_meta, location, metric, horiz=False,
                                                     save_dir=save_dir, folder='clustering_results')

# Pie charts (NEW in 251206, NOT in 241218)
plotting.plot_proportion_cells_area_depth_pie_chart(cluster_meta, save_dir=None, folder=None)
plotting.plot_proportion_cells_per_depth_pie_chart(cluster_meta, save_dir=None, folder=None)

# Experience modulation
plotting.plot_experience_modulation(coding_score_metrics_filtered, metric='experience_modulation', save_dir=None,
                                    folder='clustering_results')

plotting.plot_experience_modulation(coding_score_metrics_filtered, metric='exp_mod_direction', save_dir=None,
                                    folder='clustering_results')

plotting.plot_experience_modulation(coding_score_metrics_filtered, metric='exp_mod_persistence', save_dir=None,
                                    folder='clustering_results')

# Response metrics boxplots
plotting.plot_response_metrics_boxplot_by_cre_as_cols(response_metrics, metric, label=label, horiz=False, line_val=0.5,
                                                     save_dir=save_dir, folder='clustering_results')

plotting.plot_response_metrics_boxplot_by_cre_as_cols(response_metrics, metric, label=label, horiz=True, line_val=0.5,
                                                     save_dir=save_dir, folder='clustering_results')

plotting.plot_response_metrics_boxplot_by_cre(response_metrics, metric, ylabel=ylabel, horiz=True, line_val=0.5,
                                             save_dir=save_dir, folder='clustering_results')

plotting.plot_response_metrics_boxplot_by_cre_as_cols(response_metrics, metric, label=ylabel, horiz=False, line_val=0.5,
                                                     save_dir=save_dir, folder='clustering_results')

plotting.plot_response_metrics_boxplot_by_cre(response_metrics, metric, ylabel=ylabel, horiz=True, line_val=0,
                                             save_dir=save_dir, folder='clustering_results')

plotting.plot_response_metrics_boxplot_by_cre_as_cols(response_metrics, metric, label=ylabel, horiz=False, line_val=0,
                                                     save_dir=save_dir, folder='clustering_results')

# Tukey HSD comparisons
plotting.plot_tukey_diff_in_means_for_metric(response_metrics, metric, title=title, xlabel=xlabel, group_to_compare=group_to_compare, pref_exp_level=True)

plotting.plot_tukey_diff_in_means_for_metric(response_metrics, metric, title=title, xlabel=xlabel, lims=(-0.1, 1.1),
                                            save_dir=save_dir, folder='clustering_results')

plotting.plot_tukey_diff_in_means_for_metric(response_metrics, metric, horiz=False, lims=(-0.1, 1.1),
                                            save_dir=save_dir, folder='clustering_results')

plotting.plot_tukey_diff_in_means_for_metric(response_metrics, metric, horiz=False, lims=(-1.1, 1.1),
                                            save_dir=save_dir, folder='clustering_results')

plotting.plot_tukey_diff_in_means_for_metric(response_metrics, metric, horiz=False, lims=(-1.1, 1.6),
                                            save_dir=save_dir, folder='clustering_results')

# Color and comparison functions
feature_colors = plotting.get_feature_colors_with_gray()  # NEW in 251206

colors = plotting.get_pref_experience_level_colors_for_clusters(response_metrics)

plotting.plot_difference_of_means_and_universal_CI(tukey_results, group_to_compare=group_to_compare,
                                                  save_dir=save_dir, folder='clustering_results')

plotting.plot_diff_of_means_cre_as_col_clusters_as_rows(cluster_meta, response_metrics, metric='lifetime_sparseness_images',
                                                       save_dir=save_dir, folder='clustering_results')

plotting.plot_cluster_metrics_cre_as_col_clusters_as_rows(cluster_meta, response_metrics,
                                                         save_dir=save_dir, folder='clustering_results')
```

### PLOTTING FUNCTION CALLS (platform_paper_figures ppf module)

```python
# ppf.plot_ophys_history_for_mice - NEW in 251206, NOT in 241218
ppf.plot_ophys_history_for_mice(sessions, color_column=color_column, color_map=color_map, save_dir=None,
                               folder='clustering_results')

# Plot population averages (commented out in current version)
# ax = ppf.plot_population_averages_for_conditions(mdf, 'events', 'omissions', axes_column, hue_column,
#                                                   session_colors=False, ax=None)
```

### PLOTTING FUNCTION CALLS (platform_single_cell_examples pse module)

```python
pse.plot_matched_roi_and_coding_scores(cell_metadata, cell_dropouts, platform_experiments, save_dir=save_dir)

pse.plot_cell_change_and_omission_responses(change_mdf, omission_mdf, cell_specimen_id,
                                           save_dir=save_dir, folder='single_cell_roi_and_coding_scores')
```

### CUSTOM MATPLOTLIB/SEABORN CODE

```python
# Inline plotting code
fig, ax = plt.subplots(figsize=(7,4))
ax = sns.scatterplot(data=ps_scores_df, x='n_clusters', y='prediction_strength',
                     hue='cre_line', palette=sns.color_palette())

fig, ax = plt.subplots(figsize=(8,4))
ax = sns.pointplot(data=ps_scores_df, x='n_clusters', y='prediction_strength',
                   hue='cre_line', palette=sns.color_palette())

fig, ax = plt.subplots(figsize=(7,4))
ax = sns.scatterplot(data=min_scores_df[min_scores_df.n_clusters.isin(range(6,15))],
                     x='mean_ps',  y='sem_ps', hue='n_clusters', palette=sns.color_palette())

fig, ax = plt.subplots(figsize=(18,4))
ax = sns.stripplot(data=ps_scores_df, hue='iteration', palette=sns.color_palette(), dodge=True,
                   x='n_clusters', y='prediction_strength')

fig, ax = plt.subplots()
ax = sns.scatterplot(data=ps_scores_df[ps_scores_df.n_clusters==14],
                     x='prediction_strength', y='cluster_id', hue='cre_line')

# UMAP scatter plots with custom coloring
fig, ax = plt.subplots(figsize=figsize)
ax = sns.scatterplot(data=umap_df, x='x', y='y', s=10, ax=ax, c=colors, vmin=0, vmax=max(colors))

fig, ax = plt.subplots(figsize=figsize)
ax = sns.scatterplot(data=umap_df, x='x', y='y', s=10, ax=ax, c=colors, cmap=cmap, vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(1, 2, figsize=figsize)
ax[0] = sns.scatterplot(data=umap_df, x='x', y='y', s=10, ax=ax[0], c=colors_cell_type, cmap=cmap)
ax[1] = sns.scatterplot(data=umap_df, x='x', y='y', s=10, ax=ax[1], c=colors_cre_line, cmap=cmap)

# Color palette extraction
red = sns.color_palette('Reds_r', 6).as_hex()[0]
blue = sns.color_palette('Blues_r', 6).as_hex()[0]
purple = sns.color_palette('Purples_r', 6).as_hex()[0]

red = sns.color_palette('Reds_r', 6).as_hex()[2]
blue = sns.color_palette('Blues_r', 6).as_hex()[2]
purple = sns.color_palette('Purples_r', 6).as_hex()[2]

# Response trace plots with hspace adjustment
fig, ax = plt.subplots(3, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.3)
```

---

## FILE 2: 251206_figure_4_prediction_strength_control.py

Same imports, configuration variables, and data loading as 251206_figure_4_updated_figures.py (lines 0-199 are identical).

This file appears to be a duplicate for testing prediction strength controls.

---

## FILE 3: 260228_clustering_shuffle_control.py (744 lines)

### IMPORTS
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

%load_ext autoreload
%autoreload 2
%matplotlib inline

import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities

from visual_behavior.dimensionality_reduction.clustering import plotting
from visual_behavior.dimensionality_reduction.clustering import processing

import visual_behavior.visualization.ophys.platform_paper_figures as ppf

import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_across_session as gas

run_examples=False # skip example runs to run the notebook faster
```

### CONFIGURATION AND DATA LOADING
```python
experience_level_colors = utils.get_experience_level_colors()
cre_line_colors = utils.get_cre_line_colors()
experience_levels = utils.get_new_experience_levels()
cell_types = utils.get_cell_types()

# Path to data files
base_path = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache_new'
save_dir = os.path.join(base_path, 'clustering')
data_folder = 'shuffled_files'
folder = 'shuffle_control'
fig_path = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_figures_final/figure_4_tmp'

platform_cache_dir = loading.get_platform_analysis_cache_dir()

# Load metadata tables
experiments_table = pd.read_csv(os.path.join(platform_cache_dir, 'all_ophys_experiments_table.csv'), index_col=0)
platform_experiments = pd.read_csv(os.path.join(platform_cache_dir, 'platform_paper_ophys_experiments_table.csv'), index_col=0)
platform_cells_table = pd.read_csv(os.path.join(platform_cache_dir, 'platform_paper_ophys_cells_table.csv'), index_col=0)
matched_cells_table = pd.read_csv(os.path.join(platform_cache_dir, 'platform_paper_matched_ophys_cells_table.csv'), index_col=0)

matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types = utils.get_cell_types()

# Clustering results
n_clusters = 14
feature_matrix = pd.read_hdf(os.path.join(platform_cache_dir, 'clustering', 'clustering_feature_matrix.h5'), key='df')
feature_matrix = feature_matrix.rename(columns={'all-images':'images', 'behavioral': 'behavior'})
cluster_meta = pd.read_hdf(os.path.join(platform_cache_dir, 'clustering', 'cluster_metadata_'+str(n_clusters)+'_clusters.h5'), key='df')

original_feature_matrix = feature_matrix.copy()
original_cluster_meta = cluster_meta.copy()

# Compute mean cluster coding scores
original_mean_dropout_scores = processing.get_mean_dropout_scores_per_cluster(original_feature_matrix,
                                                  cluster_meta = original_cluster_meta.reset_index(), sort_by_cluster_size=False)

original_mean_dropout_scores_unstacked = processing.get_mean_dropout_scores_per_cluster(original_feature_matrix, stacked=False,
                                                  cluster_meta = original_cluster_meta.reset_index(), sort_by_cluster_size=False)

# Split cell specimen ids by cre lines
cre_ids = {}
cre_lines = np.sort(original_cluster_meta.cre_line.unique())
for cre_line in cre_lines:
    cre_ids[cre_line] = original_cluster_meta[original_cluster_meta.cre_line==cre_line].index.values

# Load shuffled data
shuffle_dir = os.path.join(save_dir, data_folder)
n_boots = np.arange(500)
shuffle_type = 'experience'

filepath = os.path.join(save_dir, folder, 'shuffled_feature_matrices.pkl')
if os.path.exists(filepath):
    print('loading shuffled matrices')
    with open(filepath, 'rb') as f:
        shuffled_feature_matrices = pickle.load(f)
```

### PROCESSING FUNCTION CALLS
```python
processing.get_mean_dropout_scores_per_cluster(original_feature_matrix,
                                                  cluster_meta = original_cluster_meta.reset_index(), sort_by_cluster_size=False)

processing.get_mean_dropout_scores_per_cluster(original_feature_matrix, stacked=False,
                                                  cluster_meta = original_cluster_meta.reset_index(), sort_by_cluster_size=False)
```

### PLOTTING FUNCTION CALLS
```python
plotting.plot_mean_cluster_heatmaps_remapped(original_feature_matrix, cluster_meta_cre, cre_line=cre_line,
                                             annotate=False, session_colors=True, sort_by='cluster_id',
                                             save_dir=fig_path, folder=folder, suffix='_original_clusters')
```

---

## FILE 4: 240605_familiar_only_clustering_control.py (349 lines)

### IMPORTS
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import visual_behavior.visualization.utils as utils
import seaborn as sns
import matplotlib.colors
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

import warnings
warnings.filterwarnings('ignore')

from visual_behavior.dimensionality_reduction.clustering import plotting
from visual_behavior.dimensionality_reduction.clustering import processing

import visual_behavior.visualization.ophys.platform_paper_figures as ppf

import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_across_session as gas

%load_ext autoreload
%autoreload 2
%matplotlib inline

run_examples = False
```

### CONFIGURATION
```python
experience_level_colors = utils.get_experience_level_colors()
cre_line_colors = utils.get_cre_line_colors()
experience_levels = utils.get_new_experience_levels()
cell_types = utils.get_cell_types()

base_path = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache_new'
folder = 'clustering'
save_dir = os.path.join(base_path, folder)
fig_path = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_figures_final'
```

### DATA LOADING AND PROCESSING
```python
cells_table = processing.get_cells_matched_in_3_familiar_active_sessions()
cells_table['identifier'] = [str(cells_table.iloc[row].ophys_experiment_id)+'_'+str(cells_table.iloc[row].cell_specimen_id) for row in range(len(cells_table))]
cells_table = cells_table.set_index('identifier')

version = '24_events_all_L2_optimize_by_session'
run_params, results, results_pivoted, weights_df = gfd.load_analysis_dfs(version)

results_pivoted = results_pivoted.drop(columns='experience_level')
results_pivoted = results_pivoted.merge(cells_table[[ 'experience_level']], on='identifier')

features = processing.get_features_for_clustering()
features = ['all-images', 'omissions', 'task', 'behavioral']
features = [*features, 'ophys_experiment_id']
results_pivoted = processing.limit_results_pivoted_to_features_for_clustering(results_pivoted, features)
results_pivoted = processing.flip_sign_of_dropouts(results_pivoted, processing.get_features_for_clustering(), use_signed_weights=False)
results_pivoted = results_pivoted.drop(columns=['ophys_experiment_id'])

save_dir_familiar = os.path.join(save_dir, 'familiar_only_clustering_without_problem_session')
feature_matrix = processing.get_feature_matrix_for_clustering(results_pivoted, version, save_dir=save_dir_familiar)
feature_matrix = feature_matrix.rename(columns={'all-images':'images', 'behavioral': 'behavior'})

# Gap statistic
from sklearn.cluster import SpectralClustering
metric = 'euclidean'
shuffle_type = 'all'
k_max = 25

gap_filename = os.path.join(save_dir_familiar, 'gap_scores_{}_{}_nb20_unshuffled_to_{}.pkl'.format(metric, version, shuffle_type))
if os.path.exists(gap_filename):
    with open(gap_filename, 'rb') as f:
        gap_df = pickle.load(f)
        print('loaded file')
else:
    sc = SpectralClustering()
    gap_df = processing.compute_gap(clustering=sc, data=feature_matrix, k_max = k_max,
                                    reference_shuffle=shuffle_type, metric=metric, separate_cre_lines=True)
    processing.save_clustering_results(gap_df, gap_filename)

# Co-clustering matrix
n_clusters = 10
coclust_filename = os.path.join(save_dir_familiar, 'coclustering_matrix_n_' + str(n_clusters) + '_clusters.h5')

if os.path.exists(coclust_filename):
    coclustering_df = pd.read_hdf(coclust_filename, key='df')
    print('found and loaded cached coclustering matrix file...')
else:
    print('did not find cached coclustering matrix file, will make one...')
    sc = SpectralClustering()
    X = feature_matrix.values
    m = processing.get_coClust_matrix(X=X, n_clusters=n_clusters, model=sc, nboot=np.arange(150))
    coclustering_df = pd.DataFrame(data=m, index=feature_matrix.index, columns=feature_matrix.index)
    coclust_save_file = os.path.join(save_dir_familiar, 'coclustering_matrix_n_' + str(n_clusters) + '_clusters.h5')
    coclustering_df.to_hdf(coclust_filename, key='df', format='table')

# Cluster metadata
n_clusters = 10
cluster_meta_save_path = os.path.join(save_dir_familiar, 'cluster_meta_n_'+str(n_clusters)+'_clusters.h5')

if os.path.exists(cluster_meta_save_path):
    print('loading results')
    cluster_meta = pd.read_hdf(cluster_meta_save_path, key='df')
    cell_metadata = processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)
    cell_metadata = cell_metadata.drop(columns=['ophys_experiment_id', 'cre_line'])
    cluster_meta = cluster_meta.merge(cell_metadata.reset_index(), on='cell_specimen_id')
    cluster_meta = cluster_meta.set_index('cell_specimen_id')
else:
    from sklearn.cluster import AgglomerativeClustering
    X = coclustering_df.values
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='average')
    labels = cluster.fit_predict(X)
    cell_specimen_ids = coclustering_df.index.values
    labels_dict = {'labels': labels, 'cell_specimen_id': cell_specimen_ids}
    labels_df = pd.DataFrame(data=labels_dict, columns=['labels', 'cell_specimen_id'])
    cluster_size_order = labels_df['labels'].value_counts().index.values
    labels_df['cluster_id'] = [np.where(cluster_size_order == label)[0][0] for label in labels_df.labels.values]
    cluster_labels = labels_df

    cell_metadata = processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)
    cluster_meta = cluster_labels[['cell_specimen_id', 'cluster_id', 'labels']].merge(cell_metadata, on='cell_specimen_id')
    cluster_meta = cluster_meta.set_index('cell_specimen_id')
    cluster_meta['cluster_id'] = cluster_meta['cluster_id'].copy()+1
    cluster_meta = processing.clean_cluster_meta(cluster_meta)
    cluster_meta['original_cluster_id'] = cluster_meta.cluster_id
    cluster_meta = processing.add_within_cluster_corr_to_cluster_meta(feature_matrix, cluster_meta, use_spearmanr=True)

    print('saving results')
    cluster_meta_save_path = os.path.join(save_dir_familiar, 'cluster_meta_n_'+str(n_clusters)+'_clusters.h5')
    cluster_data = cluster_meta.reset_index()[['cell_specimen_id', 'ophys_experiment_id', 'cre_line', 'cluster_id', 'labels', 'within_cluster_correlation']]
    cluster_data.to_hdf(cluster_meta_save_path, key='df', format='table')

# Load or compute response dataframes
import visual_behavior.data_access.loading as loading

data_type = 'events'
interpolate = True
output_sampling_rate = 30
inclusion_criteria = ''
event_type = 'all'
conditions = ['cell_specimen_id', 'is_change']
change_suffix = '_'+utils.get_conditions_string(data_type, conditions)+'_'+inclusion_criteria

multi_session_df = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria,
                                                        interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                         epoch_duration_mins=None)

change_mdf = multi_session_df[multi_session_df.is_change==True]

tmp = multi_session_df.copy()
tmp = tmp[tmp.is_change==False]
tmp = tmp.drop(columns='experience_level').merge(cells_table[['cell_specimen_id', 'ophys_experiment_id', 'experience_level']], on=['ophys_experiment_id', 'cell_specimen_id'])
tmp = tmp.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')
```

### PLOTTING FUNCTION CALLS
```python
# Heatmaps
plotting.plot_cre_line_means_remapped(feature_matrix, cluster_meta, session_colors=False, experience_index=0)

plotting.plot_cluster_means_remapped(feature_matrix, cluster_meta, session_colors=False,
                                     experience_index=0, save_dir=fig_path, folder=folder, ax=None)

plotting.plot_coding_score_heatmap_remapped(cluster_meta, feature_matrix, sort_by='cluster_id', session_colors=False,
                                    experience_index=0, save_dir=fig_path, folder=folder, ax=None)

plotting.plot_fraction_cells_per_cluster_per_cre(cluster_meta, col_to_group='cre_line',
                                                    save_dir=fig_path, folder=folder)

# Cluster heatmaps
plotting.plot_mean_cluster_heatmaps_remapped(feature_matrix, cluster_meta, None, session_colors=False, experience_index=0, save_dir=None, folder=None)

for cre_line in utils.get_cre_lines():
    cluster_meta_cre = cluster_meta[cluster_meta.cre_line==cre_line]
    clusters = cluster_meta_cre.value_counts('cluster_id').index.values
    plotting.plot_mean_cluster_heatmaps_remapped(feature_matrix, cluster_meta_cre,
                                                 cre_line, session_colors=False, sort_by='cluster_size',
                                                 experience_index=0, save_dir=fig_path, folder=folder)

# Population average responses
for cre_line in np.sort(tmp.cre_line.unique()):
    df = tmp[tmp.cre_line==cre_line].copy()
    axes_column = 'cluster_id'
    hue_column = 'experience_level'
    xlim_seconds=[-0.25, 0.75]
    cell_type = processing.get_cell_type_for_cre_line(cre_line)
    ax= plotting.plot_population_averages_for_clusters(df, event_type, axes_column, hue_column, session_colors=False, experience_index=0,legend=True,
                                            xlim_seconds=xlim_seconds, interval_sec=0.5,
                                            sharey=True, sharex=False,
                                            ylabel='response', xlabel='time (s)', suptitle=cell_type,
                                            save_dir=fig_path, folder=folder, suffix='_sharey', ax=None)

# Co-clustering matrix plot
coclustering_dict = {}
coclustering_dict['all'] = coclustering_df
cluster_meta_tmp = cluster_meta.copy()
cluster_meta_tmp['cre_line'] = 'all'
plotting.plot_coclustering_matrix_sorted_by_cluster_size(coclustering_dict, cluster_meta_tmp, cre_line='all',
                                                                 save_dir=fig_path, folder=folder,
                                                                 suffix='_' + str(n_clusters) + '_clusters', ax=None)
```

---

## FILE 5: 240404_figure_4_generate_analysis_files.py (379 lines)

### IMPORTS
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

%load_ext autoreload
%autoreload 2
%matplotlib inline

import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading

from visual_behavior.dimensionality_reduction.clustering import plotting
from visual_behavior.dimensionality_reduction.clustering import processing

import visual_behavior.visualization.ophys.platform_paper_figures as ppf
```

### CONFIGURATION AND DATA LOADING
```python
platform_cache_dir = loading.get_platform_analysis_cache_dir()
clustering_results = os.path.join(platform_cache_dir, 'clustering')

# Load metadata tables
experiments_table = pd.read_csv(os.path.join(platform_cache_dir, 'all_ophys_experiments_table.csv'), index_col=0)
platform_experiments = pd.read_csv(os.path.join(platform_cache_dir, 'platform_paper_ophys_experiments_table.csv'), index_col=0)
platform_cells_table = pd.read_csv(os.path.join(platform_cache_dir, 'platform_paper_ophys_cells_table.csv'), index_col=0)
matched_cells_table = pd.read_csv(os.path.join(platform_cache_dir, 'platform_paper_matched_ophys_cells_table.csv'), index_col=0)

matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types = utils.get_cell_types()

# GLM results
glm_version = '24_events_all_L2_optimize_by_session'
run_params = pd.read_pickle(os.path.join(platform_cache_dir, 'glm_results', glm_version+'_run_params.pkl'))
results_pivoted = pd.read_hdf(os.path.join(platform_cache_dir, 'glm_results', 'across_session_normalized_platform_results_pivoted.h5'), key='df')

# Feature matrix generation
feature_matrix_path = os.path.join(platform_cache_dir, 'clustering', 'clustering_feature_matrix.h5')

if not os.path.exists(feature_matrix_path):
    print("Creating feature matrix...")
    feature_matrix = processing.get_feature_matrix_for_clustering(results_pivoted, glm_version, save_dir=clustering_results)
    feature_matrix.to_hdf(feature_matrix_path, key='df')
    print("Feature matrix created and saved.")
else:
    print("Loading feature matrix...")
    feature_matrix = pd.read_hdf(feature_matrix_path, key='df')
    print("Feature matrix loaded.")

# Cluster selection parameters
expected_n_clusters = 14

# Gap statistic parameters
metric = 'euclidean'
shuffle_type = 'all'
k_max = 25

gap_filename = os.path.join(clustering_results, 'gap_scores_{}_{}_nb20_unshuffled_to_{}.pkl'.format(metric, glm_version, shuffle_type))

if os.path.exists(gap_filename):
    with open(gap_filename, 'rb') as f:
        gap_dict = pickle.load(f)
        print('loaded file')
else:
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering()
    gap_dict = processing.compute_gap(clustering=sc, data=feature_matrix, k_max=k_max,
                                    reference_shuffle=shuffle_type, metric=metric, separate_cre_lines=True)
    processing.save_clustering_results(gap_dict, gap_filename)

# Eigengap
cre_line = 'all'
eigengap_filename = 'eigengap_' + glm_version + '_' + 'kmax' + str(k_max) + '_' + (cre_line) + '.pkl'
eigengap_path = os.path.join(clustering_results, eigengap_filename)

if os.path.exists(eigengap_path):
    print('loading eigengap values scores from', eigengap_path)
    with open(eigengap_path, 'rb') as f:
        eigengap = pickle.load(f)
        f.close()
else:
    eigengap = processing.load_eigengap(glm_version, feature_matrix, cre_line='all', save_dir=clustering_results, k_max=25)

# Silhouette scores
n_clusters_range = np.arange(3, 25)
metric = 'euclidean'
n_boots = 20

silhouette_filename = os.path.join(clustering_results, 'silhouette_score_{}_{}_clusters_metric_{}_nboots_{}.pkl'.format(n_clusters_range[0], n_clusters_range[-1], metric, n_boots))

if os.path.exists(silhouette_filename):
    with open(silhouette_filename, 'rb') as f:
        silhouette_scores, silhouette_std = pickle.load(f)
        print('loaded file')
else:
    from sklearn.cluster import SpectralClustering
    X = feature_matrix.values
    silhouette_scores, silhouette_std = processing.get_silhouette_scores(X=X, model=SpectralClustering(), n_clusters=n_clusters_range, metric=metric, n_boots=n_boots)
    processing.save_clustering_results([silhouette_scores, silhouette_std], silhouette_filename)

# Co-clustering matrix
n_clusters = 14

coclust_filename = os.path.join(clustering_results, 'coclustering_matrix_n_' + str(n_clusters) + '_clusters.h5')
if os.path.exists(coclust_filename):
    coclustering_df = pd.read_hdf(coclust_filename, key='df')
else:
    from sklearn.cluster import SpectralClustering
    X = feature_matrix.values
    coclustering_matrix = processing.get_coClust_matrix(X, model=SpectralClustering, nboot=np.arange(150), n_clusters=n_clusters)
    coclustering_df = pd.DataFrame(coclustering_matrix)
    coclustering_df.to_hdf(coclust_filename, key='df')

# Cluster metadata
cluster_meta_filename = os.path.join(clustering_results, f'cluster_meta_{n_clusters}_clusters.h5')
if os.path.exists(cluster_meta_filename):
    cluster_meta = pd.read_hdf(cluster_meta_filename, key='df')
    print('found and loaded cached cluster_meta file...')
else:
    print('did not find cached cluster_meta file, will make one...')
    cluster_meta = processing.run_hierarchical_clustering_and_save_cluster_meta(coclustering_df, n_clusters, feature_matrix,
                                                            matched_cells_table, cluster_meta_filename)
    from sklearn.cluster import AgglomerativeClustering
    X = coclustering_df.values
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='average')
    labels = cluster.fit_predict(X)

# Within cluster correlation
spearman = True
cluster_meta = processing.add_within_cluster_corr_to_cluster_meta(feature_matrix, cluster_meta, use_spearmanr=spearman)

# UMAP
umap_df = processing.get_umap_results(feature_matrix, cluster_meta, save_dir=clustering_results, suffix='')

# Metrics generation
coding_score_metrics = processing.generate_coding_score_metrics_table(cluster_meta, results_pivoted,
                                                                      save_dir=clustering_results)

metrics = processing.generate_merged_table_of_coding_score_and_model_free_metrics(cluster_meta, results_pivoted,
                                                                 data_type='events',
                                                                 session_subset='full_session',
                                                                 inclusion_criteria='platform_experiment_table',
                                                                 save_dir=clustering_results)
```

### PLOTTING FUNCTION CALLS
```python
plotting.plot_gap_statistic(gap_dict, cre_lines=['all'], n_clusters_cre=expected_n_clusters,
                            tag='with_cre_shuffle', save_dir=None, folder='selecting_k_clusters')

plotting.plot_eigengap_values(eigengap, cre_lines=['all'], n_clusters_cre={'all':expected_n_clusters}, save_dir=clustering_results, folder='selecting_k_clusters')

plotting.plot_silhouette_scores(silhouette_scores=silhouette_scores, silhouette_std=silhouette_std, metric=metric,
                                n_clusters=expected_n_clusters, save_dir=clustering_results, folder='selecting_k_clusters')

plotting.plot_coclustering_matrix_sorted_by_cluster_size(coclustering_dict, cluster_meta_tmp, cre_line='all',
                                                                 suffix='_' + str(n_clusters) + '_clusters',
                                                                 save_dir=None, folder='clustering_results')

plotting.plot_within_cluster_correlations(cluster_meta, sort_order=None, spearman=spearman, suffix='_'+str(n_clusters)+'_clusters',
                                                save_dir=None, folder='clustering_results', ax=None)

plotting.plot_umap_for_clusters(cluster_meta, feature_matrix, umap_df, label_col='cluster_id', cre_lines = ['all'],
                                save_dir=None, folder='cluster_results')

plotting.plot_umap_for_clusters_separately(cluster_meta, feature_matrix, umap_df, label_col='cluster_id', save_dir=None, folder='clustering_results')

cmap = utils.get_cell_type_colors()
plotting.plot_umap_for_features_separately(cluster_meta, feature_matrix, umap_df, label_col='cell_type', cmap=cmap, save_dir=None, folder='clustering_results')
```

---

## UNIQUE FUNCTIONS FOUND IN 251206_figure_4_updated_figures.py (NOT in 241218)

The following functions appear in the latest version (251206) but NOT in the older version (241218):

1. **`plotting.get_feature_colors_with_gray()`** - Color palette function
2. **`plotting.plot_proportion_cells_area_depth_pie_chart(cluster_meta, save_dir=None, folder=None)`** - NEW pie chart
3. **`plotting.plot_proportion_cells_per_depth_pie_chart(cluster_meta, save_dir=None, folder=None)`** - NEW pie chart
4. **`ppf.plot_ophys_history_for_mice(sessions, color_column=color_column, color_map=color_map, save_dir=None, folder=...)`** - NEW platform paper function
5. **`processing.shuffle_dropout_score(data, shuffle_type=reference_shuffle, separate_cre_lines_for_shuffle=separate_cre_lines_for_shuffle)`** - NEW shuffle control function

---

## KEY CONFIGURATION DIRECTORIES

```python
# Main cache directory (varies by system):
platform_cache_dir = loading.get_platform_analysis_cache_dir()

# Data subdirectories:
data_dir = os.path.join(platform_cache_dir, 'clustering')

# Figure saving directories:
save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_figures_final\PaperRevision_Neuron\figure_4'

# Alternative paths used in control scripts:
fig_path = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_figures_final/figure_4_tmp'
base_path = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache_new'
```

---

## DATA FILES LOADED/SAVED

### CSV Files:
- `all_ophys_experiments_table.csv`
- `platform_paper_ophys_experiments_table.csv`
- `platform_paper_ophys_cells_table.csv`
- `platform_paper_matched_ophys_cells_table.csv`

### HDF5 Files (.h5):
- `across_session_normalized_platform_results_pivoted.h5`
- `all_results.h5`
- `clustering_feature_matrix.h5`
- `cluster_metadata_14_clusters.h5`
- `coclustering_matrix_n_14_clusters.h5`

### Pickle Files (.pkl):
- `glm_results/<version>_run_params.pkl`
- `gap_scores_euclidean_24_events_all_L2_optimize_by_session_nb20_unshuffled_to_all.pkl`
- `silhouette_score_3_24_clusters_metric_euclidean_nboots_20.pkl`
- `eigengap_<version>_kmax25_all.pkl`
