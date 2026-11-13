"""
Check whether each functional cluster forms one contiguous region in the Figure 5B UMAP,
to decide if cluster IDs can be labeled at cluster medians on the existing UMAP panels.

Run inside figure_5_main.ipynb after the data-loading cells (needs cluster_meta,
feature_matrix, clustering_dir):

    %run -i check_umap_cluster_contiguity.py

Uses the same cell subset as panel B (cluster 1 dropped, UMAP x < 16).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

EPS = 0.5          # DBSCAN neighborhood radius in UMAP units (axes span ~0-16)
MIN_FRAC = 0.05    # a piece must hold >=5% of the cluster's cells to count as a separate island
K = 25             # neighbors around the median used to test whether a label lands on its own cluster

# --- same subset as panel B ---
u = processing.get_umap_results(feature_matrix, cluster_meta, save_dir=clustering_dir, suffix='')
u = u[['x', 'y']].copy()
u.index = cluster_meta.index.values
u['cluster_id'] = cluster_meta['cluster_id'].values
u = u[(u.cluster_id != 1) & (u.x < 16)]

nn = NearestNeighbors(n_neighbors=K).fit(u[['x', 'y']].values)
rows = []
for cid, g in u.groupby('cluster_id'):
    labels = DBSCAN(eps=EPS, min_samples=5).fit_predict(g[['x', 'y']].values)
    sizes = pd.Series(labels[labels >= 0]).value_counts()
    n_pieces = int((sizes >= MIN_FRAC * len(g)).sum())
    med = g[['x', 'y']].median().values
    _, idx = nn.kneighbors([med])
    rows.append({'cluster_id': cid, 'n_cells': len(g), 'n_pieces': n_pieces,
                 'frac_largest_piece': sizes.max() / len(g) if len(sizes) else 0,
                 'frac_noise': np.mean(labels < 0),
                 'median_purity': np.mean(u.cluster_id.values[idx[0]] == cid)})
summary = pd.DataFrame(rows).set_index('cluster_id').round(2)
summary['label_ok'] = (summary.n_pieces == 1) & (summary.median_purity >= 0.5)
print(summary)

# --- one panel per cluster: cluster in color over all cells in gray, median marked ---
clusters = summary.index.values
ncol = 4
nrow = int(np.ceil(len(clusters) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow), sharex=True, sharey=True)
for ax, cid in zip(axes.flat, clusters):
    g = u[u.cluster_id == cid]
    ax.scatter(u.x, u.y, s=2, color='0.85')
    ax.scatter(g.x, g.y, s=3, color='C3')
    ax.plot(*g[['x', 'y']].median().values, 'kx', ms=10, mew=2)
    s = summary.loc[cid]
    ax.set_title(f'Cluster {cid}: {s.n_pieces} piece(s), purity {s.median_purity:.2f}',
                 fontsize=9, color='k' if s.label_ok else 'r')
for ax in axes.flat[len(clusters):]:
    ax.axis('off')
fig.tight_layout()
plt.show()
