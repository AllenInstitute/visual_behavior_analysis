# Additional Code Details and Snippets from Figure 4 Scripts

## DETAILED DATA LOADING SECTION (from 251206_figure_4_updated_figures.py)

### Response DataFrame Creation

```python
# Image change and repeated image responses
data_type = 'events'
interpolate = True
output_sampling_rate = 30
inclusion_criteria = 'platform_experiment_table'

# params for stim response df creation
event_type = 'all'

# params for mean response df creation
conditions = ['cell_specimen_id', 'is_change']

change_suffix = '_'+utils.get_conditions_string(data_type, conditions)+'_'+inclusion_criteria

multi_session_df = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria,
                                                        interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                         epoch_duration_mins=None)

# limit to changes, convert experience level and merge with cluster IDs
change_mdf = multi_session_df[multi_session_df.is_change==True]
change_mdf = change_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')

# limit to non-changes, convert experience level and merge with cluster IDs
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
```

### Outlier Removal

```python
threshold_percentile = 5
image_mdf_clean, image_outliers = processing.remove_outliers(image_mdf, threshold_percentile)
change_mdf_clean, change_outliers = processing.remove_outliers(change_mdf, threshold_percentile)
omission_mdf_clean, omission_outliers = processing.remove_outliers(omission_mdf, threshold_percentile)

# Can also be done per cre line
for cre_line in np.sort(image_mdf.cre_line.unique()):
    image_mdf_clean_cre = image_mdf[image_mdf.cre_line==cre_line].copy()
    image_mdf_clean_cre, _ = processing.remove_outliers(image_mdf_clean_cre, threshold_percentile)
    change_mdf_clean_cre = change_mdf[change_mdf.cre_line==cre_line].copy()
    change_mdf_clean_cre, _ = processing.remove_outliers(change_mdf_clean_cre, threshold_percentile)
    omission_mdf_clean_cre = omission_mdf[omission_mdf.cre_line==cre_line].copy()
    omission_mdf_clean_cre, _ = processing.remove_outliers(omission_mdf_clean_cre, threshold_percentile)
```

### Metrics Generation and Coding Score Calculations

```python
# Generate merged table of coding scores and model-free metrics
metrics = processing.generate_merged_table_of_coding_score_and_model_free_metrics(
    cluster_meta, results_pivoted,
    data_type='events',
    session_subset='full_session',
    inclusion_criteria='platform_experiment_table',
    save_dir=data_dir
)

# Get cluster-level metrics
cluster_metrics = processing.get_coding_score_metrics_for_clusters(cluster_meta, results_pivoted)

# Generate coding score metrics table
coding_score_metrics = processing.generate_coding_score_metrics_table(
    cluster_meta, results_pivoted,
    save_dir=data_dir
)

# Get fraction of cells per cluster per group
n_cells_per_cluster = processing.get_fraction_cells_per_cluster_per_group(cluster_meta, 'cre_line')

# Get cluster proportion statistics for locations
n_cells_table = processing.get_cluster_proportion_stats_for_locations(cluster_meta, location='mouse_id')
n_cells_table = processing.get_cluster_proportion_stats_for_locations(cluster_meta, location='cortical_area')
n_cells_table = processing.get_cluster_proportion_stats_for_locations(cluster_meta, location='imaging_depth')

# Get coding metrics for visualization
coding_metrics = processing.get_coding_score_metrics(cluster_meta, results_pivoted)
```

---

## CLUSTERING INITIALIZATION AND SELECTION CODE

### Gap Statistic Method

```python
# Load or compute gap statistic
metric = 'euclidean' # default distance metric
shuffle_type = 'all' # default shuffle type is all shuffle (cell id and regressors)
k_max = 25 #max number of clusters to test

gap_filename = os.path.join(data_dir, 'gap_scores_{}_{}_nb20_unshuffled_to_{}.pkl'.format(metric, glm_version, shuffle_type))
with open(gap_filename, 'rb') as f:
    gap_dict = pickle.load(f)

# Plot gap statistic (note: plotting module may have autoreload issues)
plotting.plot_gap_statistic(gap_dict, cre_lines=['all'], n_clusters_cre=n_clusters,
                            tag='with_cre_shuffle', save_dir=save_dir, folder='selecting_k_clusters')
```

### Eigengap Method

```python
# Load or compute eigengap
eigengap = processing.load_eigengap(glm_version, feature_matrix, cre_line='all', save_dir=data_dir, k_max=25)

# Plot eigengap
plotting.plot_eigengap_values(eigengap, cre_lines=['all'], n_clusters_cre={'all':n_clusters},
                              save_dir=save_dir, folder='selecting_k_clusters')
```

### Silhouette Score Method

```python
n_clusters_range = np.arange(3, 25)
metric = 'euclidean'
n_boots = 20

silhouette_filename = os.path.join(data_dir, 'silhouette_score_{}_{}_clusters_metric_{}_nboots_{}.pkl'.format(
    n_clusters_range[0], n_clusters_range[-1], metric, n_boots))

if os.path.exists(silhouette_filename):
    with open(silhouette_filename, 'rb') as f:
        silhouette_scores, silhouette_std = pickle.load(f)

# Plot silhouette scores
plotting.plot_silhouette_scores(silhouette_scores=silhouette_scores, silhouette_std=silhouette_std,
                                n_clusters=n_clusters, save_dir=save_dir, folder='selecting_k_clusters')
```

---

## PREDICTION STRENGTH CROSS-VALIDATION

```python
# Prediction strength evaluation code (lines ~400-800 in 251206_figure_4_updated_figures.py)

# Split cells into training and test sets
train_indices = np.random.choice(len(feature_matrix), size=len(feature_matrix)//2, replace=False)
test_indices = np.setdiff1d(np.arange(len(feature_matrix)), train_indices)

X_train = feature_matrix.iloc[train_indices].values
X_test = feature_matrix.iloc[test_indices].values

# Compute prediction strength for different numbers of clusters
n_clusters_range = np.arange(1, 21)

# This would involve running spectral clustering on both training and test sets
# and computing prediction strength metrics as per Tibshirani & Walther (2005)
```

---

## INDIVIDUAL CELL EXAMPLES CODE

```python
# Load cell metadata
cell_metadata = cluster_meta.reset_index().copy()

# Load dropout scores for individual cells
cell_dropouts = results_pivoted.copy()

# Get change and omission responses for individual cell
cell_specimen_id = 123456  # example cell ID
change_mdf_cell = change_mdf[change_mdf.cell_specimen_id == cell_specimen_id]
omission_mdf_cell = omission_mdf[omission_mdf.cell_specimen_id == cell_specimen_id]

# Plot single cell examples using platform_single_cell_examples module
pse.plot_matched_roi_and_coding_scores(cell_metadata, cell_dropouts, platform_experiments, save_dir=save_dir)

pse.plot_cell_change_and_omission_responses(change_mdf, omission_mdf, cell_specimen_id,
                                           save_dir=save_dir, folder='single_cell_roi_and_coding_scores')
```

---

## EXPERIENCE LEVEL MODULATION ANALYSIS

```python
# Filter coding score metrics
coding_score_metrics_filtered = coding_score_metrics.dropna()

# Plot experience modulation
plotting.plot_experience_modulation(coding_score_metrics_filtered, metric='experience_modulation',
                                    save_dir=save_dir, folder='clustering_results')

plotting.plot_experience_modulation(coding_score_metrics_filtered, metric='exp_mod_direction',
                                    save_dir=save_dir, folder='clustering_results')

plotting.plot_experience_modulation(coding_score_metrics_filtered, metric='exp_mod_persistence',
                                    save_dir=save_dir, folder='clustering_results')
```

---

## RESPONSE METRICS STATISTICAL COMPARISONS

### Box plots by cre line

```python
# Response metrics for different stimuli/conditions
response_metrics = metrics.copy()

# Generate box plots
for metric in ['image_response', 'change_response', 'omission_response', 'lifetime_sparseness']:
    ylabel = metric.replace('_', ' ').title()
    label = metric.replace('_', ' ')

    # Horizontal layout
    plotting.plot_response_metrics_boxplot_by_cre(response_metrics, metric, ylabel=ylabel, horiz=True, line_val=0.5,
                                                 save_dir=save_dir, folder='clustering_results')

    # Vertical layout with columns per cre line
    plotting.plot_response_metrics_boxplot_by_cre_as_cols(response_metrics, metric, label=label, horiz=False, line_val=0.5,
                                                         save_dir=save_dir, folder='clustering_results')
```

### Tukey HSD post-hoc tests

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway

# Perform one-way ANOVA
metric = 'image_response'
groups = response_metrics['cluster_id'].unique()
f_stat, p_value = f_oneway(*[response_metrics[response_metrics['cluster_id']==g][metric].dropna() for g in groups])

# Run Tukey HSD test
tukey_results = pairwise_tukeyhsd(endog=response_metrics[metric].dropna(),
                                   groups=response_metrics.loc[response_metrics[metric].notna(), 'cluster_id'],
                                   alpha=0.05)

# Plot Tukey results
plotting.plot_tukey_diff_in_means_for_metric(response_metrics, metric, title='Image Response',
                                            xlabel='Cluster ID', group_to_compare='cluster_id',
                                            save_dir=save_dir, folder='clustering_results')

# Alternative plot with specified limits
plotting.plot_tukey_diff_in_means_for_metric(response_metrics, metric, horiz=False, lims=(-0.1, 1.1),
                                            save_dir=save_dir, folder='clustering_results')
```

### Difference of means with universal confidence intervals

```python
# Plot difference of means
plotting.plot_difference_of_means_and_universal_CI(tukey_results, group_to_compare='cluster_id',
                                                  save_dir=save_dir, folder='clustering_results')

# Plot cluster metrics comparisons
plotting.plot_diff_of_means_cre_as_col_clusters_as_rows(cluster_meta, response_metrics,
                                                       metric='lifetime_sparseness_images',
                                                       save_dir=save_dir, folder='clustering_results')

plotting.plot_cluster_metrics_cre_as_col_clusters_as_rows(cluster_meta, response_metrics,
                                                         save_dir=save_dir, folder='clustering_results')
```

---

## PLOTTING HELPER FUNCTIONS AND COLOR UTILITIES

```python
# Get colors and utilities
experience_level_colors = utils.get_experience_level_colors()
cre_line_colors = utils.get_cre_line_colors()
cell_type_colors = utils.get_cell_type_colors()

# Feature colors
feature_colors = plotting.get_feature_colors_with_gray()

# Preference colors
colors = plotting.get_pref_experience_level_colors_for_clusters(response_metrics)

# Custom color palette creation
red = sns.color_palette('Reds_r', 6).as_hex()[2]
blue = sns.color_palette('Blues_r', 6).as_hex()[2]
purple = sns.color_palette('Purples_r', 6).as_hex()[2]

# Create custom color mapping for UMAP
vmin = response_metrics['metric'].min()
vmax = response_metrics['metric'].max()
cmap = plt.cm.get_cmap('viridis')
```

---

## INLINE PLOTTING AND FIGURE CUSTOMIZATION

```python
# Seaborn context and style
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

# Create figures with specific sizes
fig, ax = plt.subplots(figsize=(7, 4))
fig, ax = plt.subplots(figsize=(8, 4))
fig, ax = plt.subplots(figsize=(18, 4))

# Multi-panel figures
fig, ax = plt.subplots(3, 1, figsize=(12, 10))
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plots with hue
ax = sns.scatterplot(data=response_metrics, x='cluster_id', y='image_response',
                     hue='cre_line', palette=cre_line_colors, s=50, ax=ax)

# Point plots for means with error bars
ax = sns.pointplot(data=response_metrics, x='cluster_id', y='image_response',
                   hue='cre_line', palette=cre_line_colors, ax=ax)

# Strip plots
ax = sns.stripplot(data=response_metrics, x='cluster_id', y='image_response',
                   hue='cre_line', palette=cre_line_colors, dodge=True, ax=ax)

# Box plots
ax = sns.boxplot(data=response_metrics, x='cluster_id', y='image_response',
                 hue='cre_line', palette=cre_line_colors, ax=ax)

# Heatmaps
sns.heatmap(cluster_means.T, cmap='RdBu_r', center=0, ax=ax, cbar_kws={'label': 'Coding Score'})

# Subplots adjustment
plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.3)
```

---

## HELPER PROCESSING FUNCTIONS

These are called but full implementation is in the clustering processing module:

```python
processing.get_feature_matrix_for_clustering(results_pivoted, glm_version, save_dir=...)
processing.get_coClust_matrix(X, model=SpectralClustering, nboot=np.arange(150), n_clusters=14)
processing.clean_cluster_meta(cluster_meta)
processing.run_hierarchical_clustering_and_save_cluster_meta(coclustering_df, n_clusters, feature_matrix, matched_cells_table, filename)
processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)
processing.get_mean_dropout_scores_per_cluster(feature_matrix, cluster_meta, sort_by_cluster_size=False, stacked=True/False)
processing.get_cells_matched_in_3_familiar_active_sessions()
processing.get_features_for_clustering()
processing.limit_results_pivoted_to_features_for_clustering(results_pivoted, features)
processing.flip_sign_of_dropouts(results_pivoted, features, use_signed_weights=False)
processing.compute_gap(clustering=sc, data=feature_matrix, k_max=25, reference_shuffle='all', metric='euclidean', separate_cre_lines=True)
processing.save_clustering_results(data, filename)
processing.get_silhouette_scores(X=X, model=SpectralClustering(), n_clusters=n_clusters_range, metric='euclidean', n_boots=20)
processing.get_cell_type_for_cre_line(cre_line)
```

---

## COMMON DATA FILTERING AND MERGING PATTERNS

```python
# Filter by columns
df = mdf[mdf.cre_line == 'Vip']
df = cluster_meta[cluster_meta.cluster_id < 10]
df = response_metrics[response_metrics['metric'] > threshold]

# Drop NaN values
df = coding_score_metrics.dropna()

# Merge dataframes
mdf = multi_session_df.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')
mdf = mdf.merge(results_pivoted[['cell_specimen_id', 'coding_score']], on='cell_specimen_id')

# Group by and aggregate
grouped = response_metrics.groupby(['cluster_id', 'cre_line']).mean()
counts = cluster_meta.groupby('cluster_id').size()

# Reset index and column operations
cluster_meta_reset = cluster_meta.reset_index()
cluster_meta['original_cluster_id'] = cluster_meta['cluster_id'].copy()
cluster_meta['cluster_id'] = cluster_meta['new_cluster_id']
```

---

## VARIABLES USED IN PLOTS

```python
# Dimension variables
figsize = (12, 8)
figsize = (10, 10)

# Axes variables
axes_column = 'cluster_id'
hue_column = 'experience_level'
label_col = 'cluster_id'

# Threshold variables
threshold_percentile = 5

# Parameters for response trace plotting
xlim_seconds = [-0.25, 0.75]
interval_sec = 0.5
event_type = 'all'

# Statistical comparisons
group_to_compare = 'cluster_id'
pref_exp_level = True

# Visual properties
cmap = 'viridis'
palette = sns.color_palette()
session_colors = True
experience_index = 0
```
