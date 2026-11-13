# Run from figure_5_main.ipynb (after the data-loading cells) with:  %run -i fig5_cluster_structure_plots.py
#   -i runs it in the notebook namespace, so it uses feature_matrix, cluster_meta, etc. and needs the
#   notebook imports (np, pd, plt, sns, os, utils).

# Candidate replacements for Fig 5B: cluster structure shown without a 2D projection
#   Figure 1  left : co-clustering matrix -- probability two cells land in the same cluster across
#                    bootstrap re-clusterings (the matrix the final clusters were cut from)
#             right: cell-cell correlation of coding scores -- raw similarity, no clustering involved
#   Figure 2  dendrograms of the clusters + cluster-mean coding scores in dendrogram order
#             left : from cluster-mean coding scores (Euclidean, average linkage)
#             right: from co-clustering (1 - mean co-clustering probability between clusters)
# Cells sorted by cluster ID (same order as 5C); grey lines mark cluster boundaries.
# Deps: feature_matrix, cluster_meta, data_dir, n_clusters, save_dir (data-loading cells at the top).
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

meta_s = cluster_meta.sort_values('cluster_id', kind='stable')
ids, cl = meta_s.index.values, meta_s.cluster_id.values
cl_order = np.unique(cl)
bounds = np.flatnonzero(np.diff(cl)) + 0.5
centers = [np.flatnonzero(cl == c).mean() for c in cl_order]

cocl = pd.read_hdf(os.path.join(data_dir, f'coclustering_matrix_n_{n_clusters}_clusters.h5'), key='df').loc[ids, ids].values
X = feature_matrix.loc[ids]
flat = X.std(axis=1).values == 0                       # identical scores across features -> correlation undefined
with np.errstate(invalid='ignore', divide='ignore'):
    corr = np.nan_to_num(np.corrcoef(X.values))
print(f'{len(ids)} cells. {flat.sum()} cells have identical coding scores for every feature (e.g. all 0); '
      f'their correlations are undefined and shown as 0. By cluster: {dict(pd.Series(cl[flat]).value_counts().sort_index())}')


def _matrix(ax, M, cmap, vmin, vmax, label, title):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    for b in bounds:
        ax.axhline(b, color='0.5', lw=0.5); ax.axvline(b, color='0.5', lw=0.5)
    ax.set_yticks(centers); ax.set_yticklabels(cl_order, fontsize=10)
    ax.set_xticks(centers); ax.set_xticklabels(cl_order, fontsize=8)
    ax.set_ylabel('Cluster ID'); ax.set_xlabel(f'Cells ({len(M)}), sorted by cluster')
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.7, label=label)


fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
_matrix(axes[0], cocl, 'Greys', 0, 1, 'Probability of co-clustering', 'Co-clustering')
_matrix(axes[1], corr, 'RdBu_r', -1, 1, 'Correlation (Pearson r)', 'Coding-score correlation')
fig.tight_layout()
utils.save_figure(fig, figsize=fig.get_size_inches(), save_dir=save_dir, folder='cluster_structure',
                  fig_title='coclustering_and_correlation_matrices')
plt.show()

# ---- dendrograms ----
centroids = X.groupby(cl).mean()                                   # cluster x feature
Z_feat = linkage(centroids.values, method='average', metric='euclidean')
S = np.array([[cocl[np.ix_(cl == a, cl == b)].mean() for b in cl_order] for a in cl_order])
D = 1 - (S + S.T) / 2
np.fill_diagonal(D, 0)
Z_cocl = linkage(squareform(D, checks=False), method='average')

n = len(cl_order)
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), gridspec_kw=dict(height_ratios=[1, 1.6], hspace=0.05, wspace=0.25))
for col, (Z, title, ylab) in enumerate([(Z_feat, 'From cluster-mean coding scores', 'Euclidean distance'),
                                        (Z_cocl, 'From co-clustering', '1 - co-clustering probability')]):
    d = dendrogram(Z, labels=list(cl_order), ax=axes[0, col], color_threshold=0, above_threshold_color='k',
                   no_labels=True)
    axes[0, col].set_title(title, fontsize=13); axes[0, col].set_ylabel(ylab)
    axes[0, col].set_xticks([]); sns.despine(ax=axes[0, col], bottom=True)
    leaves = d['ivl']
    h = axes[1, col]
    im = h.imshow(centroids.loc[leaves].values.T, cmap='Blues', vmin=0, vmax=centroids.values.max(), aspect='auto',
                  extent=(0, 10 * n, len(centroids.columns), 0))
    axes[0, col].set_xlim(0, 10 * n)
    h.set_xticks(5 + 10 * np.arange(n)); h.set_xticklabels(leaves)
    h.set_yticks(np.arange(len(centroids.columns)) + 0.5)
    h.set_yticklabels([' '.join(map(str, c)) if isinstance(c, tuple) else str(c) for c in centroids.columns]
                      if col == 0 else [], fontsize=9)            # same rows in both; label once
    h.set_xlabel('Cluster ID')
fig.colorbar(im, ax=axes[1, :], shrink=0.8, label='Mean coding score')
utils.save_figure(fig, figsize=fig.get_size_inches(), save_dir=save_dir, folder='cluster_structure',
                  fig_title='cluster_dendrograms')
plt.show()
