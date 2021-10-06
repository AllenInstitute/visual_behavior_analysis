import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from visual_behavior.dimensionality_reduction import get_silhouette_scores


def plot_clustered_dropout_scores(df, labels=None, cmap='RdBu_r', model_output_type='', ax=None,):
    sort_order = np.argsort(labels)
    sorted_labels = labels[sort_order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    ax = sns.heatmap(df.values[sort_order], cmap=cmap, ax=ax, vmin=-1, vmax=1,
                     robust=True, cbar_kws={"drawedges": False, "shrink": 0.7, "label": model_output_type})
    ax.set_ylabel('cells')
    ax.set_xlabel('features')
    ax.set_title('sorted dropout scores')

    ax.set_ylim(0, df.shape[0])
    ax.set_xlim(0, df.shape[1])
    ax.set_xticks(np.arange(0, df.shape[1]))
    ax.set_xticklabels(df.keys(), rotation=90)
    ax.set_xticklabels(df.keys(), rotation=90)

    cluster_divisions = np.where(np.diff(sorted_labels) == 1)[0]
    for y in cluster_divisions:
        ax.hlines(y, xmin=0, xmax=df.shape[1], color='k')
    plt.tight_layout()


def get_elbow_plots(X, K=range(2, 20), ax=None):

    km = [KMeans(n_clusters=k).fit(X) for k in K]  # get Kmeans for different Ks
    centroids = [k.cluster_centers_ for k in km]  # get centroids of clusters

    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]  # compute distance of each datapoint
    # to its centroid, n clusters by n points
    dist = [np.min(D, axis=1) for D in D_k]
    avgWithinSS = [sum(d) / X.shape[0] for d in dist]

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(X)**2) / X.shape[0]
    bss = tss - wcss

    if ax is None:
        fig, ax = plt.figure(2, 1, figsize=(8, 12))

    # elbow curve

    ax[0].plot(K, avgWithinSS, 'k*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')

    ax[1].plot(K, bss / tss * 100, 'k*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained')
    plt.title('Elbow for KMeans clustering')


def plot_silhouette_scores(X, model=KMeans, n_clusters=np.arange(2, 10), metric='euclidean', n_boots=20, ax=None,
                         model_output_type=''):
    silhouette_scores = get_silhouette_scores(X=X, model=model, n_clusters=n_clusters, metric=metric, n_boots=n_boots)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.plot(n_clusters, silhouette_scores)
    ax.set_xlabel('N Clusters')
    ax.set_ylabel('Silhouette score')
    plt.title(model_output_type)
