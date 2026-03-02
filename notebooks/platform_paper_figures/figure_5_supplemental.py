"""
Supplemental Figures S16-S22 - Extended clustering analysis (associated with Paper Figure 5)

This script generates supplemental figures for the functional clustering analysis,
including cluster selection validation, individual cluster coding profiles,
clustering validation controls (shuffle analysis, familiar-only clustering),
and prediction strength analysis.

Supplemental figures:
- S16: Cluster number selection (gap statistic, eigengap, silhouette scores)
- S17: Individual cluster coding profiles
- S18: Clustering validation (correlations, co-clustering, shuffle control)
- S19: Additional cluster characterization (coding scores, cre-line means)
- S20-S22: Distribution of clusters across areas, depths, and cell types
- Additional: Familiar-only clustering control and prediction strength analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import scipy.stats as stats
from scipy.special import comb

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.utils as utils

from visual_behavior.dimensionality_reduction.clustering import plotting
from visual_behavior.dimensionality_reduction.clustering import processing

import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.platform_single_cell_examples as pse

import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_across_session as gas


# ============================================================================
# Configuration
# ============================================================================

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

experience_level_colors = utils.get_experience_level_colors()
cre_line_colors = utils.get_cre_line_colors()
experience_levels = utils.get_new_experience_levels()
cell_types = utils.get_cell_types()

glm_version = '24_events_all_L2_optimize_by_session'

# Main clustering parameters
n_clusters = 14
n_clusters_familiar = 10  # For familiar-only clustering control

# Create save directories
save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'figure_5_supplemental')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# ============================================================================
# Data Loading Section
# ============================================================================

# Get platform cache directory
platform_cache_dir = loading.get_platform_analysis_cache_dir()
data_dir = os.path.join(platform_cache_dir, 'clustering')

# Load metadata tables
metadata_dir = loading.get_metadata_tables_dir()
experiments_table = pd.read_csv(os.path.join(metadata_dir, 'all_ophys_experiments_table.csv'), index_col=0)
platform_experiments = pd.read_csv(os.path.join(metadata_dir, 'platform_paper_ophys_experiments_table.csv'), index_col=0)
platform_cells_table = pd.read_csv(os.path.join(metadata_dir, 'platform_paper_ophys_cells_table.csv'), index_col=0)
matched_cells_table = pd.read_csv(os.path.join(metadata_dir, 'platform_paper_matched_ophys_cells_table.csv'), index_col=0)

# Get lists of matched cells and experiments
matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()

# Get cre_lines and cell types for plot labels
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types_dict = utilities.get_cell_types_dict(cre_lines, platform_experiments)

# Load clustering results (main clustering with n_clusters=14)
feature_matrix = pd.read_hdf(os.path.join(data_dir, 'clustering_feature_matrix.h5'), key='df')
feature_matrix = feature_matrix.rename(columns={'all-images': 'images', 'behavioral': 'behavior'})

cluster_meta = pd.read_hdf(
    os.path.join(data_dir, 'cluster_metadata_' + str(n_clusters) + '_clusters.h5'),
    key='df'
)

original_feature_matrix = feature_matrix.copy()
original_cluster_meta = cluster_meta.copy()

cluster_order = np.sort(cluster_meta.cluster_id.unique())


# ============================================================================
# S16: Cluster Number Selection (Gap Statistic, Eigengap, Silhouette)
# ============================================================================

# Gap Statistic
metric = 'euclidean'
shuffle_type = 'all'
k_max = 25

gap_filename = os.path.join(
    data_dir,
    f'gap_scores_{metric}_{glm_version}_nb20_unshuffled_to_{shuffle_type}.pkl'
)
with open(gap_filename, 'rb') as f:
    gap_dict = pickle.load(f)

plotting.plot_gap_statistic(
    gap_dict, cre_lines=['all'], n_clusters_cre=n_clusters,
    tag='with_cre_shuffle', save_dir=save_dir, folder='selecting_k_clusters'
)

# Eigengap values
eigengap = processing.load_eigengap(
    glm_version, feature_matrix, cre_line='all',
    save_dir=data_dir, k_max=25
)

plotting.plot_eigengap_values(
    eigengap, cre_lines=['all'], n_clusters_cre={'all': n_clusters},
    save_dir=save_dir, folder='selecting_k_clusters'
)

# Silhouette scores
n_clusters_range = np.arange(3, 25)
silhouette_filename = os.path.join(
    data_dir,
    f'silhouette_score_{n_clusters_range[0]}_{n_clusters_range[-1]}_clusters_metric_{metric}_nboots_20.pkl'
)
if os.path.exists(silhouette_filename):
    with open(silhouette_filename, 'rb') as f:
        silhouette_scores, silhouette_std = pickle.load(f)

    plotting.plot_silhouette_scores(
        silhouette_scores=silhouette_scores, silhouette_std=silhouette_std,
        n_clusters=n_clusters, save_dir=save_dir, folder='selecting_k_clusters'
    )


# ============================================================================
# S17: Individual Cluster Coding Profiles
# ============================================================================

# Mean cluster heatmaps
plotting.plot_mean_cluster_heatmaps_remapped(
    feature_matrix, cluster_meta, sort_by='cluster_id',
    annotate=False, session_colors=True,
    save_dir=save_dir, folder='clustering_results'
)

# Cluster info
plotting.plot_cluster_info(
    cre_lines, cluster_meta, save_dir=save_dir, folder='clustering_controls'
)

# UMAP for features separately
feature_colors = plotting.get_feature_colors_with_gray()
umap_filename = os.path.join(data_dir, 'umap_df.h5')
if os.path.exists(umap_filename):
    umap_df = pd.read_hdf(umap_filename, key='df')

    plotting.plot_umap_for_features_separately(
        cluster_meta, feature_matrix, umap_df, label_col='cell_type',
        cmap=feature_colors, save_dir=save_dir, folder='clustering_results'
    )

    # UMAP for clusters separately
    plotting.plot_umap_for_clusters_separately(
        cluster_meta, feature_matrix, umap_df, label_col='cluster_id',
        save_dir=save_dir, folder='clustering_results'
    )


# ============================================================================
# S18: Clustering Validation
# ============================================================================

# Within-cluster correlations
plotting.plot_within_cluster_correlations(
    cluster_meta, sort_order=None, spearman=True,
    suffix='_' + str(n_clusters) + '_clusters',
    save_dir=save_dir, folder='clustering_controls'
)

plotting.plot_within_cluster_correlations(
    cluster_meta, sort_order=None, spearman=False,
    suffix='_' + str(n_clusters) + '_clusters',
    save_dir=save_dir, folder='clustering_controls'
)

# Co-clustering matrix
coclustering_filename = os.path.join(
    data_dir, f'coclustering_matrix_n_{n_clusters}_clusters.h5'
)
if os.path.exists(coclustering_filename):
    coclustering_df = pd.read_hdf(coclustering_filename, key='df')
    coclustering_dict = {'all': coclustering_df}
    cluster_meta_tmp = cluster_meta.copy()
    cluster_meta_tmp['cre_line'] = 'all'

    plotting.plot_coclustering_matrix_sorted_by_cluster_size(
        coclustering_dict, cluster_meta_tmp, cre_line='all',
        save_dir=save_dir, folder='clustering_controls',
        suffix='_' + str(n_clusters) + '_clusters'
    )


# ============================================================================
# S18 Continued: Shuffle Control Analysis
# ============================================================================

# Load shuffle control data
shuffle_control_dir = os.path.join(data_dir, 'shuffle_control')

# Load shuffled feature matrices
filepath = os.path.join(shuffle_control_dir, 'shuffled_feature_matrices.pkl')
if os.path.exists(filepath):
    with open(filepath, 'rb') as f:
        shuffled_feature_matrices = pickle.load(f)

    # Plot mean shuffled feature matrix
    plotting.plot_mean_shuffled_feature_matrix(
        shuffled_feature_matrices, original_cluster_meta,
        session_colors=True, experience_index=None,
        save_dir=save_dir, folder='shuffle_control'
    )

# Load shuffled cluster labels and compute mean dropout scores
n_boots = np.arange(500)
filepath = os.path.join(shuffle_control_dir, 'shuffled_labels.pkl')
if os.path.exists(filepath):
    with open(filepath, 'rb') as f:
        shuffled_labels = pickle.load(f)

    # Compute mean dropout scores for original data
    original_mean_dropout_scores = processing.get_mean_dropout_scores_per_cluster(
        original_feature_matrix, cluster_meta=original_cluster_meta.reset_index(),
        sort_by_cluster_size=False
    )

    original_mean_dropout_scores_unstacked = processing.get_mean_dropout_scores_per_cluster(
        original_feature_matrix, stacked=False,
        cluster_meta=original_cluster_meta.reset_index(),
        sort_by_cluster_size=False
    )

    # Load mean dropout scores for shuffled data
    filepath_shuffled_scores = os.path.join(shuffle_control_dir, 'mean_dropout_scores_shuffled.pkl')
    if os.path.exists(filepath_shuffled_scores):
        with open(filepath_shuffled_scores, 'rb') as f:
            nb_mean_dropout_scores_shuffled = pickle.load(f)

    # Load SSE matrix and mapping
    SSE_matrix_filename = os.path.join(shuffle_control_dir, 'SSE_matrix_shuffled_nboot.pkl')
    if os.path.exists(SSE_matrix_filename):
        with open(SSE_matrix_filename, 'rb') as f:
            SSE_matrix_dict = pickle.load(f)

        threshold = 0.15
        SSE_mapping = processing.get_cluster_mapping(SSE_matrix_dict, threshold=threshold)

        # Sort SSE values
        SSE_matrix_sorted = processing.sort_SSE_values(SSE_matrix_dict, SSE_mapping)

        # Get cluster means
        original_cluster_means = plotting.get_cluster_means(original_feature_matrix, original_cluster_meta)

        # Get shuffled cluster means
        nb_mean_dropout_scores_shuffled_unstacked = {}
        for n, n_boot in enumerate(n_boots):
            if n in shuffled_labels:
                cluster_meta_boot = shuffled_labels[n]
                feature_matrix_boot = shuffled_feature_matrices[n]
                mean_dropout_scores_unstacked = processing.get_mean_dropout_scores_per_cluster(
                    feature_matrix_boot, cluster_meta=cluster_meta_boot,
                    stacked=False, max_n_clusters=12
                )
                nb_mean_dropout_scores_shuffled_unstacked[n] = mean_dropout_scores_unstacked

        shuffled_clusters_dict = processing.get_matched_clusters_means_dict(
            SSE_mapping, nb_mean_dropout_scores_shuffled_unstacked,
            metric='mean', shuffle_type=None, cre_line=None
        )

        shuffled_cluster_means = processing.get_shuffled_cluster_means(shuffled_clusters_dict, 12)

        # Plot cluster heatmaps with SSE matrix
        plotting.plot_cluster_heatmaps_with_SSE_matrix(
            original_cluster_means, shuffled_cluster_means, SSE_matrix_sorted,
            session_colors=True, experience_index=None,
            save_dir=save_dir, folder='shuffle_control'
        )

        # Plot matched clusters heatmap
        plotting.plot_matched_clusters_heatmap_remapped(
            shuffled_clusters_dict, abbreviate_features=False,
            abbreviate_experience=True, small_fontsize=False,
            session_colors=True, save_dir=save_dir, folder='shuffle_control'
        )

        # Split shuffled dataframes by cre line
        shuffled_labels_cre = {}
        for cre_line in cre_lines:
            nb_labels_shuffled = {}
            for n_boot in n_boots:
                nb_labels_shuffled[n_boot] = shuffled_labels[n_boot][
                    shuffled_labels[n_boot].cre_line == cre_line
                ]
            shuffled_labels_cre[cre_line] = nb_labels_shuffled

        # Compute cluster probabilities and plot
        cluster_probabilities = processing.compute_probabilities(SSE_mapping, shuffled_labels)
        probability_df = processing.get_cluster_probability_df(
            cluster_probabilities,
            cre_lines=None,
            columns=['cluster_id', 'probability']
        )
        plotting.plot_cluster_probability_for_shuffle(
            probability_df,
            save_dir=save_dir,
            folder='shuffle_control'
        )

        # Compute cluster probabilities per cre line and plot
        cluster_probabilities_cre = {}
        for cre_line in cre_lines:
            cluster_probabilities_cre[cre_line] = processing.compute_probabilities(
                SSE_mapping,
                shuffled_labels_cre[cre_line]
            )

        probability_df_cre = processing.get_cluster_probability_df(
            cluster_probabilities_cre,
            cre_lines=cre_lines,
            columns=['cre_line', 'cluster_id', 'probability']
        )
        plotting.plot_cluster_probability_for_shuffle(
            probability_df_cre,
            save_dir=save_dir,
            folder='shuffle_control',
            cre_lines=cre_lines
        )

        # Compute and plot cluster size differences
        normalize = True
        shuffled_cluster_sizes_cre = {}
        for cre_line in cre_lines:
            shuffled_cluster_sizes_cre[cre_line] = processing.get_cluster_size_variance(
                SSE_mapping,
                shuffled_labels_cre[cre_line],
                normalize=normalize
            )

        original_cluster_sizes_cre = {}
        for cre_line in cre_lines:
            cluster_meta_cre = original_cluster_meta[original_cluster_meta.cre_line == cre_line]
            original_cluster_sizes_cre[cre_line] = cluster_meta_cre.value_counts(
                'cluster_id',
                normalize=normalize
            ).reindex(range(1, n_clusters + 1))
            original_cluster_sizes_cre[cre_line] = original_cluster_sizes_cre[cre_line].replace(
                np.nan, 0
            )

        cluster_size_diff_cre = processing.get_cluster_size_differece_df(
            original_cluster_sizes_cre,
            shuffled_cluster_sizes_cre,
            cre_line=cre_lines
        )
        plotting.plot_cluster_size_difference_for_shuffle(
            cluster_size_diff_cre,
            y='abs_cluster_size_diff',
            save_dir=save_dir,
            folder='shuffle_control'
        )


# ============================================================================
# S19: Additional Cluster Characterization
# ============================================================================

# Coding score heatmap matched
plotting.plot_coding_score_heatmap_matched(
    cluster_meta, feature_matrix, sort_by='cluster_id',
    session_colors=True, experience_index=None,
    save_dir=save_dir, folder='clustering_results'
)

# Cre-line means
plotting.plot_cre_line_means_remapped(
    feature_matrix, cluster_meta, session_colors=True,
    save_dir=save_dir, folder='clustering_results'
)

# Generate response metrics for difference of means analysis
response_metrics = processing.generate_merged_table_of_coding_score_and_model_free_metrics(
    cluster_meta,
    results_pivoted,
    data_type='events',
    session_subset='full_session',
    inclusion_criteria='platform_experiment_table',
    save_dir=os.path.join(platform_cache_dir, 'clustering')
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

# Plot difference of means with universal CI
from statsmodels.stats.multicomp import MultiComparison

metric = 'lifetime_sparseness_images'
title = 'Image selectivity\n(pref. exp. level)'
group_to_compare = 4

colors = plotting.get_pref_experience_level_colors_for_clusters(response_metrics)

pref_exp_response_metrics = response_metrics[
    (response_metrics.experience_level == response_metrics.dominant_experience_level)
]
response_metrics_clean = pref_exp_response_metrics.dropna(subset=[metric])

multi_comp = MultiComparison(response_metrics_clean[metric], response_metrics_clean['cluster_id'])
tukey_results = multi_comp.tukeyhsd()

plotting.plot_difference_of_means_and_universal_CI(
    tukey_results,
    group_to_compare=group_to_compare,
    group_colors=colors,
    color='k',
    title=title,
    xlabel='Difference in means',
    ylabel='Cluster ID',
    ax=None
)

# ============================================================================
# S20-S22: Distribution of Clusters Across Areas and Depths
# ============================================================================

# Cluster depth distributions (separated by cre line)
location = 'binned_depth'
metric = 'n_cells_location'
ylabel = 'Number of cells\nin each depth'

plotting.plot_cluster_depth_distribution_by_cre_line_separately(
    cluster_meta,
    location,
    cluster_order=None,
    metric=metric,
    ylabel=ylabel,
    legend_title='',
    save_dir=save_dir,
    folder='cluster_properties'
)

# Cluster depth distributions (by cre lines, horizontal)
location = 'binned_depth'
metric = 'n_cells_location'
ylabel = 'Number of cells in each depth'

plotting.plot_cluster_depth_distribution_by_cre_lines(
    cluster_meta,
    location,
    metric,
    horiz=True,
    ylabel=ylabel,
    save_dir=save_dir,
    folder='cluster_properties'
)

# Cluster depth distributions (by cre lines, vertical)
location = 'binned_depth'
metric = 'n_cells_location'
ylabel = 'Number of cells in each depth'

plotting.plot_cluster_depth_distribution_by_cre_lines(
    cluster_meta,
    location,
    metric,
    horiz=False,
    ylabel=ylabel,
    save_dir=save_dir,
    folder='cluster_properties'
)

# Fraction cells per cluster per cre-line
plotting.plot_fraction_cells_per_cluster_per_cre(
    cluster_meta, cluster_order=None, col_to_group='cre_line',
    save_dir=save_dir, folder='clustering_results'
)

# Percent cells per cluster with cre lines (all combined and per cre)
plotting.plot_percent_cells_per_cluster_per_cre_and_all(
    cluster_meta,
    col_to_group='cre_line',
    save_dir=save_dir
)

# Percent cells per cluster colored by dominant feature
features = response_metrics[['cell_specimen_id', 'dominant_feature_cluster']].drop_duplicates(
    subset='cell_specimen_id'
)
tmp = cluster_meta.merge(features, on='cell_specimen_id')

plotting.plot_percent_cells_per_cluster_per_cre_dominant_feature(
    tmp,
    cluster_order=cluster_order,
    col_to_group='cre_line',
    save_dir=save_dir
)

plotting.plot_percent_cells_per_cluster_per_cre_dominant_feature(
    tmp,
    cluster_order=cluster_order,
    match_height=True,
    col_to_group='cre_line',
    save_dir=save_dir
)

# Percent cells per cluster with cre on x-axis
plotting.plot_percent_cells_per_cluster_per_cre_dominant_feature_xaxis(
    tmp,
    col_to_group='cre_line',
    match_height=True,
    cluster_order=None,
    save_dir=save_dir
)

# Pie charts for proportion of cells
plotting.plot_proportion_cells_area_depth_pie_chart(
    cluster_meta,
    save_dir=save_dir,
    folder='cluster_properties'
)

plotting.plot_proportion_cells_per_depth_pie_chart(
    cluster_meta,
    save_dir=save_dir,
    folder='cluster_properties'
)

# Distribution metrics across cre-lines
# Prepare different area/depth subsets if available
try:
    # Try plotting tukey differences
    plotting.plot_tukey_diff_in_means_for_metric(
        cluster_meta, feature_matrix, metric='depth',
        save_dir=save_dir, folder='clustering_results'
    )
except:
    pass

# Difference of means by cre line
try:
    plotting.plot_diff_of_means_cre_as_col_clusters_as_rows(
        cluster_meta, feature_matrix,
        save_dir=save_dir, folder='clustering_results'
    )
except:
    pass

# Cluster metrics by cre line
try:
    plotting.plot_cluster_metrics_cre_as_col_clusters_as_rows(
        cluster_meta, feature_matrix,
        save_dir=save_dir, folder='clustering_results'
    )
except:
    pass

# Response metrics by cre line
try:
    plotting.plot_response_metrics_boxplot_by_cre_as_cols(
        cluster_meta, feature_matrix,
        save_dir=save_dir, folder='clustering_results'
    )
except:
    pass


# ============================================================================
# Familiar-Only Clustering Control (S21)
# ============================================================================

# Load familiar-only cells and results
try:
    cells_table_familiar = processing.get_cells_matched_in_3_familiar_active_sessions()
    cells_table_familiar['identifier'] = [
        str(cells_table_familiar.iloc[row].ophys_experiment_id) + '_' +
        str(cells_table_familiar.iloc[row].cell_specimen_id)
        for row in range(len(cells_table_familiar))
    ]
    cells_table_familiar = cells_table_familiar.set_index('identifier')

    # Load familiar results
    run_params_fam, results_fam, results_pivoted_fam, weights_df_fam = gfd.load_analysis_dfs(glm_version)

    # Limit to familiar sessions only
    results_pivoted_fam = results_pivoted_fam.drop(columns='experience_level')
    results_pivoted_fam = results_pivoted_fam.merge(
        cells_table_familiar[['experience_level']], on='identifier'
    )

    # Limit to features for clustering
    features = processing.get_features_for_clustering()
    features_list = ['all-images', 'omissions', 'task', 'behavioral', 'ophys_experiment_id']
    results_pivoted_fam = processing.limit_results_pivoted_to_features_for_clustering(
        results_pivoted_fam, features_list
    )

    # Make coding scores positive
    results_pivoted_fam = processing.flip_sign_of_dropouts(
        results_pivoted_fam, processing.get_features_for_clustering(),
        use_signed_weights=False
    )

    results_pivoted_fam = results_pivoted_fam.drop(columns=['ophys_experiment_id'])

    # Get feature matrix for familiar sessions
    save_dir_familiar = os.path.join(data_dir, 'familiar_only_clustering_without_problem_session')
    if not os.path.exists(save_dir_familiar):
        os.makedirs(save_dir_familiar)

    feature_matrix_familiar = processing.get_feature_matrix_for_clustering(
        results_pivoted_fam, glm_version, save_dir=save_dir_familiar
    )
    feature_matrix_familiar = feature_matrix_familiar.rename(columns={
        'all-images': 'images', 'behavioral': 'behavior'
    })

    # Load or compute familiar clustering
    cluster_meta_familiar_path = os.path.join(
        save_dir_familiar, f'cluster_meta_n_{n_clusters_familiar}_clusters.h5'
    )
    if os.path.exists(cluster_meta_familiar_path):
        cluster_meta_familiar = pd.read_hdf(cluster_meta_familiar_path, key='df')

        # Plot familiar clustering results
        plotting.plot_gap_statistic(
            gap_dict, cre_lines=['all'], n_clusters_cre=n_clusters_familiar,
            tag='familiar_only', save_dir=save_dir, folder='familiar_only_clustering'
        )

        plotting.plot_cre_line_means_remapped(
            feature_matrix_familiar, cluster_meta_familiar,
            session_colors=False, experience_index=0
        )

        plotting.plot_cluster_means_remapped(
            feature_matrix_familiar, cluster_meta_familiar,
            session_colors=False, experience_index=0,
            save_dir=save_dir, folder='familiar_only_clustering'
        )

        plotting.plot_mean_cluster_heatmaps_remapped(
            feature_matrix_familiar, cluster_meta_familiar, None,
            session_colors=False, experience_index=0, save_dir=None, folder=None
        )

        plotting.plot_fraction_cells_per_cluster_per_cre(
            cluster_meta_familiar, col_to_group='cre_line',
            save_dir=save_dir, folder='familiar_only_clustering'
        )

        # Load multi-session data for familiar sessions
        conditions = ['cell_specimen_id', 'is_change']
        multi_session_df_fam = loading.get_multi_session_df_for_conditions(
            data_type,
            event_type='all',
            conditions=conditions,
            inclusion_criteria=inclusion_criteria,
            interpolate=interpolate,
            output_sampling_rate=output_sampling_rate,
            epoch_duration_mins=None
        )

        # Prepare data for familiar-only population averages
        tmp = multi_session_df_fam.copy()
        tmp = tmp[tmp.is_change == False]
        # Add new experience level by merging in cells_table
        tmp = tmp.drop(columns='experience_level').merge(
            cells_table_familiar[['cell_specimen_id', 'ophys_experiment_id', 'experience_level']],
            on=['ophys_experiment_id', 'cell_specimen_id']
        )
        # Add cluster_id by merging in cluster meta
        tmp = tmp.merge(
            cluster_meta_familiar.reset_index()[['cell_specimen_id', 'cluster_id']],
            on='cell_specimen_id'
        )

        # Plot population averages for each cre line
        event_type = 'images'
        for cre_line in np.sort(tmp.cre_line.unique()):
            df = tmp[tmp.cre_line == cre_line].copy()

            axes_column = 'cluster_id'
            hue_column = 'experience_level'
            xlim_seconds = [-0.25, 0.75]
            cell_type = processing.get_cell_type_for_cre_line(cre_line)

            plotting.plot_population_averages_for_clusters(
                df,
                event_type,
                axes_column,
                hue_column,
                session_colors=False,
                experience_index=0,
                legend=True,
                xlim_seconds=xlim_seconds,
                interval_sec=0.5,
                sharey=True,
                sharex=False,
                ylabel='response',
                xlabel='time (s)',
                suptitle=cell_type,
                save_dir=save_dir,
                folder='familiar_only_clustering',
                suffix='_sharey',
                ax=None
            )

except Exception as e:
    print(f"Familiar-only clustering analysis skipped: {e}")


# ============================================================================
# Prediction Strength Analysis (S22)
# ============================================================================

def get_train_test_predict_labels_KMeans(feature_matrix, n_clusters, iteration=42):
    """
    Run clustering on train and test sets, then predict labels in test set
    using training set model (nearest centroid assignment).
    """
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    X = feature_matrix.copy()
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=iteration)

    model_train = KMeans(n_clusters=n_clusters, random_state=iteration, n_init=10)
    model_train.fit(X_train)
    train_labels = model_train.labels_
    train_centers = model_train.cluster_centers_

    test_predictions = model_train.predict(X_test)

    model_test = KMeans(n_clusters=n_clusters, random_state=iteration, n_init=10)
    test_labels = model_test.fit_predict(X_test)

    return X_train, X_test, train_labels, test_labels, test_predictions


def get_prediction_strength(test_labels, test_predictions):
    """
    Computes prediction strength for a clustering.

    Args:
        test_labels: Labels from clustering test set independently.
        test_predictions: Labels from assigning test data to training centroids.

    Returns:
        DataFrame with prediction strengths per cluster.
    """
    unique_test_clusters = np.unique(test_labels)
    k = len(unique_test_clusters)

    if k == 1:
        return pd.DataFrame([[-1, 1.0]], columns=['cluster_number', 'prediction_strength'])

    strengths = []

    for cluster_num in unique_test_clusters:
        indices = np.where(test_labels == cluster_num)[0]
        n_cells_in_test_cluster = len(indices)

        if n_cells_in_test_cluster <= 2:
            strengths.append([cluster_num, 0])
            continue

        pred_in_cluster = test_predictions[indices]
        cluster_ids, counts = np.unique(pred_in_cluster, return_counts=True)

        same_predict_pairs = np.sum(comb(counts, 2))
        total_pairs = comb(n_cells_in_test_cluster, 2)
        strength = same_predict_pairs / total_pairs
        strengths.append([cluster_num, strength])

    return pd.DataFrame(strengths, columns=['cluster_number', 'prediction_strength'])


def get_prediction_strengths(X, n_iterations=5, max_n_clusters=20):
    """
    Compute prediction strength across multiple cluster numbers and iterations.
    """
    range_of_k = range(2, max_n_clusters)
    ps_scores_df = pd.DataFrame()

    for n_clusters in range_of_k:
        for iteration in range(n_iterations):
            X_train, X_test, train_labels, test_labels, test_predictions = \
                get_train_test_predict_labels_KMeans(X, n_clusters=n_clusters, iteration=iteration)
            ps_scores = get_prediction_strength(test_labels, test_predictions)
            ps_scores['n_clusters'] = n_clusters
            ps_scores['iteration'] = iteration
            ps_scores_df = pd.concat([ps_scores_df, ps_scores])

    return ps_scores_df


def get_mean_and_sem_ps_for_k_clusters(group, col='min'):
    """Helper function to compute mean and SEM for prediction strength."""
    sem = stats.sem(group[col].values)
    mean = np.mean(group[col].values)
    n_clusters = group.index.values[0][0]
    return pd.Series([mean, sem])


def get_mean_and_sem_across_iterations(scores_df, col='min'):
    """
    Compute mean and SEM of prediction strength across iterations.

    Args:
        col: 'min' for minimum PS, 'mean' for mean PS
    """
    scores_df_grouped = scores_df.groupby(['n_clusters', 'iteration']).describe()
    try:
        scores_df_grouped = scores_df_grouped.drop(columns=['cluster_number', 'cluster_id'])
    except:
        pass
    scores_df_grouped.columns = scores_df_grouped.columns.droplevel(0)

    scores_df_grouped = scores_df_grouped.groupby(['n_clusters']).apply(
        get_mean_and_sem_ps_for_k_clusters, col=col
    )
    scores_df_grouped.columns = [f'mean_{col}_ps', f'sem_{col}_ps']
    scores_df_grouped['threshold'] = scores_df_grouped[f'mean_{col}_ps'] + scores_df_grouped[f'sem_{col}_ps']
    scores_df_grouped = scores_df_grouped.reset_index()

    return scores_df_grouped


# Run prediction strength analysis
try:
    X = feature_matrix.copy()
    ps_scores_df = get_prediction_strengths(X, n_iterations=5, max_n_clusters=20)

    # Plot minimum prediction strength
    min_scores_df = get_mean_and_sem_across_iterations(ps_scores_df, col='min')

    range_of_k = np.sort(min_scores_df.n_clusters.unique())
    pred_strengths = min_scores_df.sort_values(by=['n_clusters'])['mean_min_ps'].values
    yerr = min_scores_df.sort_values(by=['n_clusters'])['sem_min_ps'].values

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range_of_k, pred_strengths, yerr=yerr)
    ax.set_ylim(0, 1)
    ax.set_xlabel('# Clusters')
    ax.set_ylabel('Minimum Prediction Strength')
    ax.set_title(f'KMeans Prediction Strength (n={len(X)})')
    ax.axhline(y=0.8, xmin=0, xmax=1, linestyle='--', color='gray')

    try:
        selected_k = min_scores_df[min_scores_df['threshold'] > 0.8].sort_values(
            by=['n_clusters']
        )['n_clusters'].values[-1]
        ax.axvline(x=selected_k, ymin=0, ymax=1, linestyle='--', color='gray')
    except:
        pass

    fig.savefig(os.path.join(save_dir, 'prediction_strength_minimum.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot mean prediction strength
    mean_scores_df = get_mean_and_sem_across_iterations(ps_scores_df, col='mean')

    range_of_k = np.sort(mean_scores_df.n_clusters.unique())
    pred_strengths = mean_scores_df.sort_values(by=['n_clusters'])['mean_mean_ps'].values
    yerr = mean_scores_df.sort_values(by=['n_clusters'])['sem_mean_ps'].values

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range_of_k, pred_strengths, yerr=yerr)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('# Clusters')
    ax.set_ylabel('Mean Prediction Strength')
    ax.set_title(f'KMeans Mean Prediction Strength (n={len(X)})')
    ax.axhline(y=0.8, xmin=0, xmax=1, linestyle='--', color='gray')

    try:
        selected_k = mean_scores_df[mean_scores_df['threshold'] > 0.8].sort_values(
            by=['n_clusters']
        )['n_clusters'].values[-1]
        ax.axvline(x=selected_k, ymin=0, ymax=1, linestyle='--', color='gray')
    except:
        pass

    fig.savefig(os.path.join(save_dir, 'prediction_strength_mean.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Scatter plot of all prediction strength values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax = sns.scatterplot(data=ps_scores_df, x='n_clusters', y='prediction_strength',
                        color='k', ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('# Clusters')
    ax.set_ylabel('Prediction Strength')
    ax.set_title(f'Prediction Strength Across Clusters and Iterations (n={len(X)})')
    ax.axhline(y=0.8, xmin=0, xmax=1, linestyle='--', color='gray')
    ax.axhline(y=0.6, xmin=0, xmax=1, linestyle='--', color='gray')

    fig.savefig(os.path.join(save_dir, 'prediction_strength_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Prediction strength analysis skipped: {e}")


print(f"Figure 5 supplemental complete. Outputs saved to {save_dir}")
