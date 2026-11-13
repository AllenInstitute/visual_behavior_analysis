# Run from figure_5_main.ipynb (after the data-loading cells) with:  %run -i fig5_subsample_reclustering.py
#   -i runs it in the notebook namespace (feature_matrix, cluster_meta, data_dir, n_clusters, save_dir, np, pd, plt, sns, os, utils).

# Subsample re-clustering: are the clusters stable when the DATA change, not just the random start?
# The published co-clustering matrix reruns spectral clustering 100x on the SAME cells; only the random
# initialization varies. Here each run clusters a random SUBSAMPLE_FRAC of cells, drawn without replacement
# (with replacement duplicates cells, which distorts spectral clustering's similarity graph). Co-clustering
# probability for a pair = times clustered together / times both sampled (consensus clustering,
# Monti et al. 2003). Same method otherwise: SpectralClustering(n_clusters=K), then average-linkage
# AgglomerativeClustering on the co-clustering matrix (as in processing.run_hierarchical_clustering...).
#   Figure 1: published vs subsample co-clustering matrix, cells sorted by published cluster
#   Figure 2: per-cluster stability | agreement of each run with published labels (ARI) |
#             consensus clusters re-derived from the subsample matrix vs published clusters
#   Figure 3: dendrogram of published clusters from the subsample matrix
# Runs are cached (save_dir/cluster_structure/subsample_reclustering_runs.pkl); re-running only adds runs,
# so the cell can be interrupted and resumed.
# Deps: feature_matrix, cluster_meta, data_dir, n_clusters, save_dir.
import time
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

N_RUNS = 100
SUBSAMPLE_FRAC = 0.8
K = n_clusters        # same k as the published spectral clustering; clusters with < 5 cells were dropped after

meta_s = cluster_meta.sort_values('cluster_id', kind='stable')
ids, cl = meta_s.index.values, meta_s.cluster_id.values
cl_order = np.unique(cl)
bounds = np.flatnonzero(np.diff(cl)) + 0.5
centers = [np.flatnonzero(cl == c).mean() for c in cl_order]
X = feature_matrix.loc[ids].values
N = len(ids)

# ---- runs (cached) ----
run_dir = os.path.join(save_dir, 'cluster_structure'); os.makedirs(run_dir, exist_ok=True)
run_file = os.path.join(run_dir, f'subsample_reclustering_runs_k{K}_frac{SUBSAMPLE_FRAC}.pkl')
runs = pd.read_pickle(run_file) if os.path.exists(run_file) else []     # list of (sampled idx, labels)
_first = len(runs)
for r in range(len(runs), N_RUNS):
    t0 = time.time()
    idx = np.sort(np.random.default_rng(r).choice(N, int(SUBSAMPLE_FRAC * N), replace=False))
    runs.append((idx, SpectralClustering(n_clusters=K).fit(X[idx]).labels_))
    pd.to_pickle(runs, run_file)
    if r == _first:
        print(f'{time.time() - t0:.0f} s per run -> ~{(N_RUNS - r - 1) * (time.time() - t0) / 60:.0f} min for the rest')
print(f'{len(runs)} subsample runs of {int(SUBSAMPLE_FRAC * N)} / {N} cells, k = {K}')

together = np.zeros((N, N), np.float32)
sampled = np.zeros((N, N), np.float32)
ari = []
for idx, lab in runs[:N_RUNS]:
    ix = np.ix_(idx, idx)
    together[ix] += lab[:, None] == lab[None, :]
    sampled[ix] += 1
    ari.append(adjusted_rand_score(cl[idx], lab))
cocl_sub = together / np.maximum(sampled, 1)
del together, sampled

cocl_pub = pd.read_hdf(os.path.join(data_dir, f'coclustering_matrix_n_{n_clusters}_clusters.h5'), key='df').loc[ids, ids].values

# consensus clusters from the subsample matrix, same procedure as the published labels
cons = AgglomerativeClustering(n_clusters=K, linkage='average').fit_predict(cocl_sub)
ari_cons = adjusted_rand_score(cl, cons)


def _within(M, c):
    w = M[np.ix_(cl == c, cl == c)]
    n = w.shape[0]
    return (w.sum() - np.trace(w)) / max(n * (n - 1), 1)


stab = pd.DataFrame({'published (random starts)': [_within(cocl_pub, c) for c in cl_order],
                     'subsampled data': [_within(cocl_sub, c) for c in cl_order]}, index=cl_order)
print('\nMean within-cluster co-clustering probability:')
print(stab.round(3).to_string())
print(f'\nAgreement with published labels (adjusted Rand index; 1 = identical, 0 = chance):'
      f'\n  single subsample runs: median {np.median(ari):.2f} (range {np.min(ari):.2f}-{np.max(ari):.2f})'
      f'\n  consensus clusters from the subsample matrix: {ari_cons:.2f}')

# ---- Figure 1: matrices ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
for a, M, title in [(axes[0], cocl_pub, 'Published: repeated random starts'),
                    (axes[1], cocl_sub, f'Re-clustered {SUBSAMPLE_FRAC:.0%} subsamples ({len(runs[:N_RUNS])} runs)')]:
    im = a.imshow(M, cmap='Greys', vmin=0, vmax=1, rasterized=True)
    for b in bounds:
        a.axhline(b, color='0.5', lw=0.5); a.axvline(b, color='0.5', lw=0.5)
    a.set_yticks(centers); a.set_yticklabels(cl_order, fontsize=10)
    a.set_xticks(centers); a.set_xticklabels(cl_order, fontsize=8)
    a.set_ylabel('Published cluster ID'); a.set_xlabel(f'Cells ({N}), sorted by published cluster')
    a.set_title(title, fontsize=13)
    plt.colorbar(im, ax=a, shrink=0.7, label='Probability of co-clustering')
fig.tight_layout()
utils.save_figure(fig, figsize=fig.get_size_inches(), save_dir=save_dir, folder='cluster_structure',
                  fig_title='coclustering_published_vs_subsampled')
plt.show()

# ---- Figure 2: stability summaries ----
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.3, 0.8, 1.2]))
stab.plot.bar(ax=axes[0], color=['0.6', 'k'], width=0.8, rot=0)
axes[0].set_ylim(0, 1.25); axes[0].set_yticks([0, 0.5, 1]); axes[0].set_xlabel('Published cluster ID')
axes[0].set_ylabel('Mean within-cluster\nco-clustering probability'); axes[0].legend(frameon=False, fontsize=9, loc='upper center', ncol=2)
axes[0].set_title('Cluster stability')
axes[1].hist(ari, bins=np.linspace(0, 1, 41), color='0.4')
axes[1].axvline(ari_cons, color='r', lw=2, label=f'consensus = {ari_cons:.2f}')
axes[1].set_xlim(0, 1); axes[1].set_xlabel('Adjusted Rand index vs published'); axes[1].set_ylabel('Runs')
axes[1].legend(frameon=False, fontsize=9, loc='upper left'); axes[1].set_title('Agreement per run')
ct = pd.crosstab(cl, cons, normalize='index')                       # rows: published, cols: consensus
ct = ct[ct.idxmax(axis=0).sort_values().index]                      # order columns by best-matching published cluster
im = axes[2].imshow(ct.values, cmap='Greys', vmin=0, vmax=1, aspect='auto')
axes[2].set_yticks(range(len(ct))); axes[2].set_yticklabels(ct.index)
axes[2].set_xticks(range(ct.shape[1])); axes[2].set_xticklabels([])
axes[2].set_ylabel('Published cluster ID'); axes[2].set_xlabel('Consensus cluster (subsampled), ordered by best match')
axes[2].set_title('Where each published cluster\'s cells go')
plt.colorbar(im, ax=axes[2], shrink=0.8, label='Fraction of cells')
for a in axes[:2]:
    sns.despine(ax=a)
fig.tight_layout()
utils.save_figure(fig, figsize=fig.get_size_inches(), save_dir=save_dir, folder='cluster_structure',
                  fig_title='subsample_reclustering_stability')
plt.show()

# ---- Figure 3: dendrogram from the subsample matrix ----
S = np.array([[cocl_sub[np.ix_(cl == a, cl == b)].mean() for b in cl_order] for a in cl_order])
D = 1 - (S + S.T) / 2
np.fill_diagonal(D, 0)
fig, ax = plt.subplots(figsize=(7, 3.5))
dendrogram(linkage(squareform(D, checks=False), method='average'), labels=list(cl_order), ax=ax,
           color_threshold=0, above_threshold_color='k')
ax.set_xlabel('Published cluster ID'); ax.set_ylabel('1 - co-clustering probability')
ax.set_title('Cluster dendrogram from subsampled co-clustering')
sns.despine(ax=ax)
utils.save_figure(fig, figsize=fig.get_size_inches(), save_dir=save_dir, folder='cluster_structure',
                  fig_title='subsample_coclustering_dendrogram')
plt.show()
