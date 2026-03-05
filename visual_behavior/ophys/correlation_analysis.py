import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


def run_correlation_clustering_analysis(
    dff_traces,
    metadata,
    cluster_threshold_fraction=0.5,
    save_dir=None,
):
    """Run cell-cell correlation and hierarchical clustering analysis on dff_traces.

    Steps performed:
      1. Compute and plot the raw cell-cell correlation matrix.
      2. Run PCA, remove the first principal component, recompute and plot correlations.
      3. Run Ward hierarchical clustering on the raw correlation matrix, plot the
         dendrogram, report cluster sizes, and plot the correlation matrix sorted by
         cluster membership.

    Parameters
    ----------
    dff_traces : pd.DataFrame
        DataFrame with a 'dff' column, as returned by ``dataset.dff_traces``.
    metadata : dict
        Metadata dictionary from the dataset (e.g. ``dataset.metadata``).  Used
        to label plots.  Expected keys: 'ophys_experiment_id', 'cre_line'.
    cluster_threshold_fraction : float, optional
        Fraction of the maximum Ward linkage distance at which to cut the
        dendrogram to assign cluster labels.  Default is 0.5.
    save_dir : str or None, optional
        Directory in which to save correlation matrices as .npy files.  If
        None (default) nothing is saved.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'correlation_matrix'         : ndarray (n_cells x n_cells)
        - 'correlation_matrix_pc1_removed' : ndarray (n_cells x n_cells)
        - 'cluster_labels'             : ndarray (n_cells,), 1-indexed
        - 'n_clusters'                 : int
        - 'cells_per_cluster'          : dict {cluster_id: n_cells}
        - 'linkage_matrix'             : ndarray, Ward linkage matrix Z
        - 'pc1_variance_explained'     : float, fraction (not percent)
    """
    ophys_experiment_id = metadata.get('ophys_experiment_id', '')
    cre_line = metadata.get('cre_line', '')
    title_suffix = f'\nophys_experiment_id: {ophys_experiment_id}, cre_line: {cre_line}'

    # ------------------------------------------------------------------
    # 1. Build dff array and compute raw correlation matrix
    # ------------------------------------------------------------------
    dff_array = np.hstack(dff_traces['dff'].values.reshape(-1, 1))  # (n_cells, n_timepoints)
    correlation_matrix = np.corrcoef(dff_array)

    _plot_correlation_matrix(
        correlation_matrix,
        title='Cell-cell correlations' + title_suffix,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'{ophys_experiment_id}_correlation_matrix.npy')
        np.save(path, correlation_matrix)
        print(f'Saved: {path}')

    # ------------------------------------------------------------------
    # 2. Remove PC1 and recompute correlations
    # ------------------------------------------------------------------
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(dff_array.T)          # (n_timepoints, 1)
    pc1_reconstruction = (pc1_scores @ pca.components_).T  # (n_cells, n_timepoints)
    dff_array_pc1_removed = dff_array - pc1_reconstruction

    pc1_var = pca.explained_variance_ratio_[0]
    print(f'PC1 explains {pc1_var * 100:.1f}% of variance')

    correlation_matrix_pc1_removed = np.corrcoef(dff_array_pc1_removed)

    _plot_correlation_matrix(
        correlation_matrix_pc1_removed,
        title='Cell-cell correlations - PC1 removed' + title_suffix,
    )

    if save_dir is not None:
        path = os.path.join(save_dir, f'{ophys_experiment_id}_correlation_matrix_pc1_removed.npy')
        np.save(path, correlation_matrix_pc1_removed)
        print(f'Saved: {path}')

    # ------------------------------------------------------------------
    # 3. Hierarchical clustering on the raw correlation matrix
    # ------------------------------------------------------------------
    distance_matrix = 1 - correlation_matrix
    np.fill_diagonal(distance_matrix, 0)
    Z = linkage(squareform(distance_matrix), method='ward')

    # Dendrogram
    fig, ax = plt.subplots(figsize=(12, 4))
    dendrogram(Z, ax=ax, color_threshold=0, above_threshold_color='gray', no_labels=True)
    ax.set_xlabel('cell #')
    ax.set_ylabel('Ward linkage distance')
    ax.set_title('Hierarchical clustering dendrogram' + title_suffix)
    plt.tight_layout()
    plt.show()

    # Cut tree and report clusters
    threshold = cluster_threshold_fraction * Z[:, 2].max()
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')
    n_clusters = cluster_labels.max()
    cells_per_cluster = {c: int((cluster_labels == c).sum()) for c in range(1, n_clusters + 1)}

    print(f'\nNumber of clusters (threshold={threshold:.2f}): {n_clusters}')
    for c, n in cells_per_cluster.items():
        print(f'  Cluster {c}: {n} cells')

    # Reordered correlation matrix
    sort_idx = np.argsort(cluster_labels)
    corr_reordered = correlation_matrix[np.ix_(sort_idx, sort_idx)]
    sorted_labels = cluster_labels[sort_idx]
    boundaries = np.where(np.diff(sorted_labels))[0] + 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_reordered, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Pearson correlation')
    for b in boundaries:
        ax.axhline(b - 0.5, color='k', linewidth=0.8, linestyle='--')
        ax.axvline(b - 0.5, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('cell # (sorted by cluster)')
    ax.set_ylabel('cell # (sorted by cluster)')
    ax.set_title(f'Cell-cell correlations - sorted by cluster ({n_clusters} clusters)' + title_suffix)
    plt.tight_layout()
    plt.show()

    if save_dir is not None:
        path = os.path.join(save_dir, f'{ophys_experiment_id}_correlation_matrix_clustered.npy')
        np.save(path, corr_reordered)
        print(f'Saved: {path}')

    return {
        'correlation_matrix': correlation_matrix,
        'correlation_matrix_pc1_removed': correlation_matrix_pc1_removed,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'cells_per_cluster': cells_per_cluster,
        'linkage_matrix': Z,
        'pc1_variance_explained': pc1_var,
    }


def _plot_correlation_matrix(matrix, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Pearson correlation')
    ax.set_xlabel('cell #')
    ax.set_ylabel('cell #')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
