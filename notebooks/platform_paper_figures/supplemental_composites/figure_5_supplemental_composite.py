"""
Supplemental Figures S16-S22 Composite Script (associated with Paper Figure 5)

This script generates composite figures for supplemental figures S16-S22,
which show extended clustering analysis including cluster selection validation,
individual cluster coding profiles, clustering validation controls, and
distribution of clusters across cell types, areas, and depths.

S16: Cluster number selection (3 panels)
- A — Gap statistic plot (left)
- B — Eigengap values (center)
- C — Silhouette scores (right)
- figsize=(22, 8), single row

S17: Individual cluster profiles (5 panels)
- Row 1: A — Mean cluster heatmaps (full width)
- Row 2: B — Cluster info | C — UMAP features
- Row 3: D — UMAP clusters | E — Feature colors legend
- figsize=(22, 24)

S18: Clustering validation (6 panels)
- Row 1: A — Within-cluster correlations (Spearman) | B — Within-cluster correlations (Pearson)
- Row 2: C — Co-clustering matrix
- Row 3: D — Shuffled feature matrix | E — Matched clusters heatmap
- Row 4: F — Cluster probability | G — Cluster size differences
- figsize=(24, 30)

S19: Familiar-only clustering (9 panels)
- Row 1: A — Schematic (full width)
- Row 2: B — Gap statistic familiar | C — Cluster means familiar
- Row 3: D — Heatmap familiar | E — Fraction cells per cluster
- Row 4: F-H — Population averages per cre line (1x3)
- Row 5: I — Prediction strength
- figsize=(22, 32)

S20: Excitatory clusters (8 panels)
- Row 1: A — Clustered coding heatmap | B — Cluster means
- Row 2: C — % cells per cluster | D — Image responses
- Row 3: E — Change responses | F — Omission responses
- Row 4: G — Fraction by area | H — Fraction by depth
- figsize=(24, 28)

S21: Sst clusters (same layout as S20)
S22: Vip clusters (same layout as S20)
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
save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'supplemental_composites')
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
# Prediction Strength Helper Functions
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


# ============================================================================
# COMPOSITE FIGURE S16: Cluster Number Selection
# ============================================================================

def plot_figure_S16_composite(save_dir=None, folder='supplemental_composites'):
    """
    Generate composite Figure S16 showing cluster number selection metrics.

    Layout (1 row, 3 panels):
    - A — Gap statistic plot
    - B — Eigengap values
    - C — Silhouette scores
    """
    figsize = (22, 8)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # Panel A: Gap statistic plot
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.32], yspan=[0, 1])
    ax_A.text(0.5, 0.5, 'A — Gap statistic plot\n(place manually)', transform=ax_A.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    # Panel B: Eigengap values
    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.34, 0.66], yspan=[0, 1])
    ax_B.text(0.5, 0.5, 'B — Eigengap values\n(place manually)', transform=ax_B.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_B.axis('off')

    # Panel C: Silhouette scores
    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.68, 1], yspan=[0, 1])
    ax_C.text(0.5, 0.5, 'C — Silhouette scores\n(place manually)', transform=ax_C.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_S16_composite', formats=['.pdf'])
    return fig


# ============================================================================
# COMPOSITE FIGURE S17: Individual Cluster Profiles
# ============================================================================

def plot_figure_S17_composite(save_dir=None, folder='supplemental_composites'):
    """
    Generate composite Figure S17 showing individual cluster coding profiles.

    Layout (3 rows):
    - Row 1 (A): Mean cluster heatmaps (full width)
    - Row 2: B — Cluster info | C — UMAP features
    - Row 3: D — UMAP clusters | E — Feature colors legend
    """
    figsize = (22, 24)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # Row 1 (A): Mean cluster heatmaps
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 0.25])
    ax_A.text(0.5, 0.5, 'A — Mean cluster heatmaps\n(place manually)', transform=ax_A.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    # Row 2: Cluster info (B) and UMAP features (C)
    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.28, 0.52])
    ax_B.text(0.5, 0.5, 'B — Cluster info\n(place manually)', transform=ax_B.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_B.axis('off')

    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0.28, 0.52])
    ax_C.text(0.5, 0.5, 'C — UMAP features\n(place manually)', transform=ax_C.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    # Row 3: UMAP clusters (D) and Feature colors legend (E)
    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.56, 0.80])
    ax_D.text(0.5, 0.5, 'D — UMAP clusters\n(place manually)', transform=ax_D.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0.56, 0.80])
    ax_E.text(0.5, 0.5, 'E — Feature colors legend\n(place manually)', transform=ax_E.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_E.set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax_E.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_S17_composite', formats=['.pdf'])
    return fig


# ============================================================================
# COMPOSITE FIGURE S18: Clustering Validation
# ============================================================================

def plot_figure_S18_composite(save_dir=None, folder='supplemental_composites'):
    """
    Generate composite Figure S18 showing clustering validation metrics.

    Layout (4 rows):
    - Row 1: A — Within-cluster correlations (Spearman) | B — Within-cluster correlations (Pearson)
    - Row 2: C — Co-clustering matrix
    - Row 3: D — Shuffled feature matrix | E — Matched clusters heatmap
    - Row 4: F — Cluster probability | G — Cluster size differences
    """
    figsize = (24, 30)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # Row 1: Spearman (A) and Pearson (B) correlations
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0, 0.20])
    ax_A.text(0.5, 0.5, 'A — Within-cluster correlations\n(Spearman)', transform=ax_A.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0, 0.20])
    ax_B.text(0.5, 0.5, 'B — Within-cluster correlations\n(Pearson)', transform=ax_B.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_B.axis('off')

    # Row 2: Co-clustering matrix (C)
    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0.22, 0.42])
    ax_C.text(0.5, 0.5, 'C — Co-clustering matrix\n(place manually)', transform=ax_C.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    # Row 3: Shuffled feature matrix (D) and Matched clusters heatmap (E)
    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.44, 0.64])
    ax_D.text(0.5, 0.5, 'D — Shuffled feature matrix\n(place manually)', transform=ax_D.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0.44, 0.64])
    ax_E.text(0.5, 0.5, 'E — Matched clusters heatmap\n(place manually)', transform=ax_E.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_E.set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax_E.axis('off')

    # Row 4: Cluster probability (F) and Cluster size differences (G)
    ax_F = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.66, 0.86])
    ax_F.text(0.5, 0.5, 'F — Cluster probability\n(place manually)', transform=ax_F.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_F.set_title('F', loc='left', fontweight='bold', fontsize=16)
    ax_F.axis('off')

    ax_G = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0.66, 0.86])
    ax_G.text(0.5, 0.5, 'G — Cluster size differences\n(place manually)', transform=ax_G.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_G.set_title('G', loc='left', fontweight='bold', fontsize=16)
    ax_G.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_S18_composite', formats=['.pdf'])
    return fig


# ============================================================================
# COMPOSITE FIGURE S19: Familiar-Only Clustering
# ============================================================================

def plot_figure_S19_composite(save_dir=None, folder='supplemental_composites'):
    """
    Generate composite Figure S19 showing familiar-only clustering analysis.

    Layout (5 rows):
    - Row 1: A — Schematic (full width)
    - Row 2: B — Gap statistic familiar | C — Cluster means familiar
    - Row 3: D — Heatmap familiar | E — Fraction cells per cluster
    - Row 4: F-H — Population averages per cre line (1x3)
    - Row 5: I — Prediction strength
    """
    figsize = (22, 32)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # Row 1: Schematic (A)
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 0.12])
    ax_A.text(0.5, 0.5, 'A — Schematic\n(place manually)', transform=ax_A.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    # Row 2: Gap statistic (B) and Cluster means (C)
    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.14, 0.28])
    ax_B.text(0.5, 0.5, 'B — Gap statistic (familiar)\n(place manually)', transform=ax_B.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_B.axis('off')

    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0.14, 0.28])
    ax_C.text(0.5, 0.5, 'C — Cluster means (familiar)\n(place manually)', transform=ax_C.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    # Row 3: Heatmap (D) and Fraction cells (E)
    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.30, 0.48])
    ax_D.text(0.5, 0.5, 'D — Heatmap (familiar)\n(place manually)', transform=ax_D.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0.30, 0.48])
    ax_E.text(0.5, 0.5, 'E — Fraction cells per cluster\n(place manually)', transform=ax_E.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_E.set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax_E.axis('off')

    # Row 4: Population averages (F-H)
    cre_labels = ['F', 'G', 'H']
    for i, (cre_label) in enumerate(cre_labels):
        xstart = i * 0.33
        xend = (i + 1) * 0.33
        ax_cre = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[xstart, xend], yspan=[0.50, 0.68])
        ax_cre.text(0.5, 0.5, f'{cre_label} — Population average\n(cre line {i+1})', transform=ax_cre.transAxes,
                    ha='center', va='center', fontsize=11, color='gray')
        ax_cre.set_title(cre_label, loc='left', fontweight='bold', fontsize=16)
        ax_cre.axis('off')

    # Row 5: Prediction strength (I)
    ax_I = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0.70, 0.90])
    ax_I.text(0.5, 0.5, 'I — Prediction strength\n(place manually)', transform=ax_I.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_I.set_title('I', loc='left', fontweight='bold', fontsize=16)
    ax_I.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_S19_composite', formats=['.pdf'])
    return fig


# ============================================================================
# COMPOSITE FIGURES S20-S22: Cell Type Cluster Composites
# ============================================================================

def _plot_cell_type_cluster_composite(cell_type_label, fig_label, save_dir=None, folder='supplemental_composites'):
    """
    Generate composite figure for cell type specific clustering.

    Layout (4 rows):
    - Row 1: A — Clustered coding heatmap | B — Cluster means
    - Row 2: C — % cells per cluster | D — Image responses
    - Row 3: E — Change responses | F — Omission responses
    - Row 4: G — Fraction by area | H — Fraction by depth
    """
    figsize = (24, 28)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # Row 1: Clustered coding heatmap (A) and Cluster means (B)
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0, 0.20])
    ax_A.text(0.5, 0.5, 'A — Clustered coding heatmap\n(place manually)', transform=ax_A.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0, 0.20])
    ax_B.text(0.5, 0.5, 'B — Cluster means\n(place manually)', transform=ax_B.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_B.axis('off')

    # Row 2: % cells per cluster (C) and Image responses (D)
    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.22, 0.42])
    ax_C.text(0.5, 0.5, 'C — % cells per cluster\n(place manually)', transform=ax_C.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0.22, 0.42])
    ax_D.text(0.5, 0.5, 'D — Image responses\n(place manually)', transform=ax_D.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    # Row 3: Change responses (E) and Omission responses (F)
    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.44, 0.64])
    ax_E.text(0.5, 0.5, 'E — Change responses\n(place manually)', transform=ax_E.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_E.set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax_E.axis('off')

    ax_F = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0.44, 0.64])
    ax_F.text(0.5, 0.5, 'F — Omission responses\n(place manually)', transform=ax_F.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_F.set_title('F', loc='left', fontweight='bold', fontsize=16)
    ax_F.axis('off')

    # Row 4: Fraction by area (G) and Fraction by depth (H)
    ax_G = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.66, 0.86])
    ax_G.text(0.5, 0.5, 'G — Fraction by area\n(place manually)', transform=ax_G.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_G.set_title('G', loc='left', fontweight='bold', fontsize=16)
    ax_G.axis('off')

    ax_H = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1], yspan=[0.66, 0.86])
    ax_H.text(0.5, 0.5, 'H — Fraction by depth\n(place manually)', transform=ax_H.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_H.set_title('H', loc='left', fontweight='bold', fontsize=16)
    ax_H.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, f'figure_{fig_label}_composite', formats=['.pdf'])
    return fig


def plot_figure_S20_composite(save_dir=None, folder='supplemental_composites'):
    """Generate composite Figure S20 for Excitatory clusters."""
    return _plot_cell_type_cluster_composite('Excitatory', 'S20', save_dir, folder)


def plot_figure_S21_composite(save_dir=None, folder='supplemental_composites'):
    """Generate composite Figure S21 for Sst clusters."""
    return _plot_cell_type_cluster_composite('Sst', 'S21', save_dir, folder)


def plot_figure_S22_composite(save_dir=None, folder='supplemental_composites'):
    """Generate composite Figure S22 for Vip clusters."""
    return _plot_cell_type_cluster_composite('Vip', 'S22', save_dir, folder)


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("Generating composite figures S16-S22...")

    # Generate S16 composite (cluster selection)
    fig_S16 = plot_figure_S16_composite(save_dir=save_dir, folder='supplemental_composites')
    print("Figure S16 composite generated and saved.")

    # Generate S17 composite (individual cluster profiles)
    fig_S17 = plot_figure_S17_composite(save_dir=save_dir, folder='supplemental_composites')
    print("Figure S17 composite generated and saved.")

    # Generate S18 composite (clustering validation)
    fig_S18 = plot_figure_S18_composite(save_dir=save_dir, folder='supplemental_composites')
    print("Figure S18 composite generated and saved.")

    # Generate S19 composite (familiar-only clustering)
    fig_S19 = plot_figure_S19_composite(save_dir=save_dir, folder='supplemental_composites')
    print("Figure S19 composite generated and saved.")

    # Generate S20 composite (Excitatory clusters)
    fig_S20 = plot_figure_S20_composite(save_dir=save_dir, folder='supplemental_composites')
    print("Figure S20 composite generated and saved.")

    # Generate S21 composite (Sst clusters)
    fig_S21 = plot_figure_S21_composite(save_dir=save_dir, folder='supplemental_composites')
    print("Figure S21 composite generated and saved.")

    # Generate S22 composite (Vip clusters)
    fig_S22 = plot_figure_S22_composite(save_dir=save_dir, folder='supplemental_composites')
    print("Figure S22 composite generated and saved.")

    print(f"\nAll composite figures saved to: {save_dir}")
