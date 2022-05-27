import os
import umap
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy.spatial.distance import cdist, pdist

from sklearn.cluster import KMeans

import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading

from visual_behavior_glm import GLM_clustering as glm_clust  # noqa E501

from visual_behavior.dimensionality_reduction.clustering import processing
from visual_behavior.dimensionality_reduction.clustering.processing import get_silhouette_scores, get_cluster_density, get_cre_lines, get_cell_type_for_cre_line

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})


def plot_feature_matrix_for_cre_lines(feature_matrix, cell_metadata, save_dir=None, folder=None):
    """
    plots the feature matrix used for clustering where feature matrix consists of cell_specimen_ids as rows,
    and features x experience levels as columns, for cells matched across experience levels
    :param feature_matrix:
    :param cell_metadata: table with cell_specimen_id as index and metadata as columns
    :param save_dir: directory to save plot
    :param folder:
    :return:
    """
    # check if there are negative values in feature_matrix, if so, use diff cmap and set vmin to -1
    if len(np.where(feature_matrix < 0)[0]) > 0:
        vmin = -1
        cmap = 'RdBu'
    else:
        vmin = 0
        cmap = 'Blues'

    figsize = (15, 5)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i, cre_line in enumerate(get_cre_lines(cell_metadata)):
        data = processing.get_feature_matrix_for_cre_line(feature_matrix, cell_metadata, cre_line)
        ax[i] = sns.heatmap(data.values, cmap=cmap, ax=ax[i], vmin=vmin, vmax=1,
                            robust=True, cbar_kws={"drawedges": False, "shrink": 0.7, "label": 'coding score'})
        ax[i].set_title(get_cell_type_for_cre_line(cell_metadata, cre_line))
        ax[i].set_ylabel('cells')
        ax[i].set_ylim(0, data.shape[0])
        ax[i].set_yticks([0, data.shape[0]])
        ax[i].set_yticklabels((0, data.shape[0]), fontsize=14)
        ax[i].set_ylim(ax[i].get_ylim()[::-1])  # flip y axes so larger clusters are on top
        ax[i].set_xlabel('')
        ax[i].set_xlim(0, data.shape[1])
        ax[i].set_xticks(np.arange(0, data.shape[1]) + 0.5)
        ax[i].set_xticklabels([key[1] + ' -  ' + key[0] for key in list(data.keys())], rotation=90, fontsize=14)

    fig.subplots_adjust(wspace=0.7)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'feature_matrix_unsorted')


def plot_feature_matrix_sorted(feature_matrix, cluster_meta, sort_col='cluster_id', save_dir=None, folder=None, suffix=''):
    """
    plots feature matrix used for clustering sorted by sort_col

    sort_col: column in cluster_meta to sort rows of feature_matrix (cells) by
    """
    # check if there are negative values in feature_matrix, if so, use diff cmap and set vmin to -1
    if len(np.where(feature_matrix < 0)[0]) > 0:
        vmin = -1
        cmap = 'RdBu'
    else:
        vmin = 0
        cmap = 'Blues'

    figsize = (15, 5)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i, cre_line in enumerate(get_cre_lines(cluster_meta)):
        # get cell ids for this cre line in sorted order
        sorted_cluster_meta_cre = cluster_meta[cluster_meta.cre_line == cre_line].sort_values(by=sort_col)
        cell_order = sorted_cluster_meta_cre.index.values
        label_values = sorted_cluster_meta_cre[sort_col].values

        # get data from feature matrix for this set of cells
        data = feature_matrix.loc[cell_order]
        ax[i] = sns.heatmap(data.values, cmap=cmap, ax=ax[i], vmin=vmin, vmax=1,
                            robust=True, cbar_kws={"drawedges": False, "shrink": 0.7, "label": 'coding score'})

        ax[i].set_title(get_cell_type_for_cre_line(cluster_meta, cre_line))
        ax[i].set_ylabel('cells')
        ax[i].set_ylim(0, data.shape[0])
        ax[i].set_yticks([0, data.shape[0]])
        ax[i].set_yticklabels((0, data.shape[0]), fontsize=14)
        ax[i].set_ylim(ax[i].get_ylim()[::-1])  # flip y axes so larger clusters are on top
        ax[i].set_xlabel('')
        ax[i].set_xlim(0, data.shape[1])
        ax[i].set_xticks(np.arange(0, data.shape[1]) + 0.5)
        ax[i].set_xticklabels([key[1] + ' -  ' + key[0] for key in list(data.keys())], rotation=90, fontsize=14)

        # plot a line at the division point between clusters
        cluster_divisions = np.where(np.diff(label_values) == 1)[0]
        for y in cluster_divisions:
            ax[i].hlines(y, xmin=0, xmax=data.shape[1], color='k')

    fig.subplots_adjust(wspace=0.7)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'feature_matrix_sorted_by_' + sort_col + suffix)


# NOTE: the function above and below are doing the same thing and need to be integrated ####


def plot_clustered_dropout_scores(df, labels=None, cmap='Blues', plot_difference=False, ax=None,):
    '''
    params:
    df: pd.DataFrame, pivoted glm output
    labels: array, clustering labels
    cmap: str, matplotlib color map to use, default= "Blues"
    plot_difference: Boolean, default=False, whether to plot dropout scores or difference
        (cluster - population) in dropout scores.
    ax: figure ax

    '''
    sort_order = np.argsort(labels)
    sorted_labels = labels[sort_order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    if plot_difference is True:
        df_diff = df.abs() - df.abs().mean()
        ax = sns.heatmap(df_diff.values[sort_order], cmap=cmap, ax=ax, vmin=0, vmax=1,
                         robust=True,
                         cbar_kws={"drawedges": False, "shrink": 0.8,
                                   "label": 'difference in fraction change in explained variance'})
    elif plot_difference is False:
        ax = sns.heatmap(df.abs().values[sort_order], cmap=cmap, ax=ax, vmin=0, vmax=1,
                         robust=True,
                         cbar_kws={"drawedges": False, "shrink": 0.8,
                                   "label": 'fraction change in explained variance'})
    ax.set_ylabel('cells')
    ax.set_xlabel('features')
    ax.set_title('sorted dropout scores')

    ax.set_ylim(0, df.shape[0])
    ax.set_xlim(0, df.shape[1])
    ax.set_xticks(np.arange(0.5, df.shape[1] + 0.5))
    ax.set_xticklabels([key[1] + ' - ' + key[0] for key in list(df.keys())], rotation=90)
    ax.set_xticklabels(df.keys(), rotation=90)

    ax.set_xlabel('')
    ax2 = ax.twinx()
    ax2.set_yticks([0, len(df)])
    ax2.set_yticklabels([0, len(df)])

    # plot horizontal lines to seperate the clusters
    cluster_divisions = np.where(np.diff(sorted_labels) == 1)[0]
    for y in cluster_divisions:
        ax.hlines(y, xmin=0, xmax=df.shape[1], color='k')

    # set cluster labels
    cluster_divisions = np.hstack([0, cluster_divisions, len(labels)])
    mid_cluster_divisions = []
    for i in range(0, len(cluster_divisions) - 1):
        mid_cluster_divisions.append(((cluster_divisions[i + 1] - cluster_divisions[i]) / 2) + cluster_divisions[i])

    # label cluster ids
    unique_cluster_ids = np.sort(np.unique(labels))
    ax.set_yticks(mid_cluster_divisions)
    ax.set_yticklabels(unique_cluster_ids)
    ax.set_ylabel('cluster ID')

    # separate regressors
    for x in np.arange(0, df.shape[0], 3):
        ax.vlines(x, ymin=0, ymax=df.shape[0], color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig, ax


def get_elbow_plots(X, n_clusters=range(2, 20), ax=None):
    '''
    Computes within cluster density and variance explained for Kmeans method.
    :param X: data
    :param n_clusters: default = range(2, 20)
    :param ax: default=None
    :return:
    '''

    km = [KMeans(n_clusters=k).fit(X) for k in n_clusters]  # get Kmeans for different Ks
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
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))

    # elbow curve

    ax[0].plot(n_clusters, avgWithinSS, 'k*-')
    ax[0].grid(True)
    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('Average within-cluster sum of squares')
    ax[0].set_title('Elbow for KMeans clustering')

    ax[1].plot(n_clusters, bss / tss * 100, 'k*-')
    ax[1].grid(True)
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('Percentage of variance explained')
    ax[1].set_title('Elbow for KMeans clustering')

    return ax


def plot_silhouette_scores(X=None, model=KMeans, silhouette_scores=None, silhouette_std=None,
                           n_clusters=np.arange(2, 10), metric=None, n_boots=20, ax=None,
                           model_output_type=''):

    assert X is not None or silhouette_scores is not None, 'must provide either data to cluster or recomputed scores'
    assert X is None or silhouette_scores is None, 'cannot provide data to cluster and silhouette scores'

    if silhouette_scores is None:
        if metric is None:
            metric = 'euclidean'
        print('no silhouette scores provided, generating silhouette scores')
        silhouette_scores, silhouette_std = get_silhouette_scores(X=X, model=model, n_clusters=n_clusters, metric=metric, n_boots=n_boots)
    elif silhouette_scores is not None:
        if len(silhouette_scores) != len(n_clusters):
            n_clusters = np.arange(1, len(silhouette_scores) + 1)

        if metric is None:
            metric = ''

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(n_clusters, silhouette_scores, 'ko-')
    if silhouette_std is not None:
        ax.errorbar(n_clusters, silhouette_scores, silhouette_std, color='k')
    ax.set_title('{}, {}'.format(model_output_type, metric), fontsize=16)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette score')
    plt.grid()
    plt.tight_layout()

    return ax


def plot_silhouette_scores_n_clusters(silhouette_scores, cell_metadata, n_clusters_cre=None, save_dir=None, folder=None):
    """
    n_clusters_cre is a dictionary with cre lines as keys and the selected number of clusters for that cre line as values
    silhouette scores are plotted with dashed line at the x value for n_clusters_cre
    """
    suffix = ''  # suffix to attach to file name - blank if n_clusters_cre is None, otherwise will add label below
    figsize = (15, 5)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i, cre_line in enumerate(get_cre_lines(cell_metadata)):
        silhouette_scores_cre = silhouette_scores[cre_line]
        ax[i] = plot_silhouette_scores(silhouette_scores=silhouette_scores_cre[0],
                                       silhouette_std=silhouette_scores_cre[1], ax=ax[i])
        ax[i].set_title(get_cell_type_for_cre_line(cell_metadata, cre_line))
        if n_clusters_cre:
            n_clusters = n_clusters_cre[cre_line]
            ax[i].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')
            suffix = '_n_clusters'
    plt.grid()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'silhouette_scores' + suffix)


def plot_coclustering_matrix_and_dendrograms(coclustering_matrices, cre_line, cluster_meta, n_clusters_cre, save_dir=None, folder=None):
    """
    gets linkage matrix for coclustering matrices to create dendrogram, then plots dendrogram and coclustering heatmap accordingly
    coclustering_matrices is a dictionary where keys are cre_lines and values are dataframes of co-clustering probability for all cell pairs
    cluster_meta is a dataframe containing cluster_ids and metadata for all cells
    n_clusters_cre: dictionary with cre_lines as keys and n_clusters to use as values
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    figsize = (6, 8)
    fig, ax = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1, 5]})
    # get coclustering matrix and cell ID labels for this cre line
    coclustering_matrix = coclustering_matrices[cre_line]
    X = coclustering_matrix.values
    cell_specimen_ids = coclustering_matrix.index.values
    labels = cluster_meta.loc[cell_specimen_ids].cluster_id.values
    # get linkage matrix using same parameters as were used for clustering
    Z = linkage(X, method='average', metric='euclidean')
    # create dendrogram from linkage matrix and get dict R to get the leaves
    R = dendrogram(Z, no_plot=True,)  # truncate_mode='level', p=p)
    # the leaves are the original matrix indices reordered according to dendrogram
    leaves = R['leaves']
    print('n_leaves:', len(leaves))

    # reorder the labels corresponding to rows (cell_specimen_id) of coclustering matrix by the leaf order
    leaf_cluster_ids = labels[leaves]  # labels=leaf_cluster_ids,
    # get the unique cluster IDs in the same order as they occur in the leaves of dendrogram
    cluster_order = list(dict.fromkeys(leaf_cluster_ids))
    cluster_order = [int(c) for c in cluster_order]
    # define a leaf label function for what to do when labeling non-singleton clusters

    def llf(xx):
        print(xx)
        return str(cluster_order[xx])
    # now plot the same dendrogram with labels
    dendrogram(Z, labels=leaf_cluster_ids, ax=ax[1])  # truncate_mode='level', p=p,
    ax[1].grid(False)
    ax[1].set_title(get_cell_type_for_cre_line(cluster_meta, cre_line))

    # plot dendrogram with leaves truncated to highest level nodes (p=n_clusters)
    dendrogram(Z, truncate_mode='lastp', p=n_clusters_cre[cre_line], color_threshold=0, ax=ax[0])  # leaf_label_func=llf,truncate_mode='level', p=p,
    # put the order of cluster_ids in the dendrogram in the title
    title = 'cluster order: '
    for c in cluster_order:
        title = title + str(int(c)) + ','
    ax[0].set_title(title)
    print(cluster_order)
    ax[0].grid(False)

    # Sort co-clustering matrix according to dendrogram output and plot
    # reorder cell_specimen_ids (corresponding to rows and cols of coclustering matrix) by the leaf order in the dendrogram
    sorted_cell_specimen_ids = cell_specimen_ids[leaves]
    # sort rows and cols of coclustering matrix by sorted cell_specimen_ids
    sorted_coclustering_matrix = coclustering_matrix.loc[sorted_cell_specimen_ids]
    sorted_coclustering_matrix = sorted_coclustering_matrix[sorted_cell_specimen_ids]
    ax[2] = sns.heatmap(sorted_coclustering_matrix, cmap="Greys", ax=ax[2], square=True,
                        cbar=True, cbar_kws={"drawedges": False, "label": 'probability of co-clustering'},)
    ax[2].set_xticklabels('')
    ax[2].set_yticks((0, sorted_coclustering_matrix.shape[0]))
    ax[2].set_yticklabels((0, sorted_coclustering_matrix.shape[0]))
    ax[2].set_ylabel('cells')
    plt.subplots_adjust(hspace=0.5)
    if save_dir:
        filename = 'coclustering_matrix_sorted_by_dendrogram_' + cre_line.split('-')[0]
        utils.save_figure(fig, figsize, save_dir, folder, filename, formats=['.png'])  # saving to PDF is super slow


def plot_coclustering_matrix_sorted_by_cluster_size(coclustering_matrices, cluster_meta, cre_line,
                                                    save_dir=None, folder=None, ax=None):
    """
    plot co-clustering matrix sorted by cluster size for a given cre_line
    will save plot if save_dir and folder are provided (and ax is None)
    if ax is provided, will plot on provided ax
    """
    coclustering_matrix = coclustering_matrices[cre_line]
    cre_meta = cluster_meta[cluster_meta.cre_line == cre_line]
    cre_meta = cre_meta.sort_values(by='cluster_id')
    sorted_cell_specimen_ids = cre_meta.index.values
    # sort rows and cols of coclustering matrix by sorted cell_specimen_ids
    sorted_coclustering_matrix = coclustering_matrix.loc[sorted_cell_specimen_ids]
    sorted_coclustering_matrix = sorted_coclustering_matrix[sorted_cell_specimen_ids]

    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(sorted_coclustering_matrix, cmap="Greys", ax=ax, square=True,
                     cbar=True, cbar_kws={"drawedges": False, "label": 'probability of\nco-clustering', 'shrink': 0.7, },)
    ax.set_title(get_cell_type_for_cre_line(cluster_meta, cre_line))
    ax.set_title('')
    ax.set_yticks((0, sorted_coclustering_matrix.shape[0]))
    ax.set_yticklabels((0, sorted_coclustering_matrix.shape[0]), fontsize=20)
    ax.set_ylabel('cells', fontsize=20)
    ax.set_xticks((0, sorted_coclustering_matrix.shape[0]))
    ax.set_xticklabels('')
    ax.set_xlabel('')
    sns.despine(ax=ax, bottom=False, top=False, left=False, right=False)
    if save_dir:
        filename = 'coclustering_matrix_sorted_by_cluster_size_' + cre_line.split('-')[0]
        utils.save_figure(fig, figsize, save_dir, folder, filename, formats=['.png'])  # saving to PDF is super slow)
    return ax


def plot_umap_for_clusters(cluster_meta, feature_matrix, label_col='cluster_id', save_dir=None, folder=None):
    """
    plots umap for each cre line, colorized by metadata column provided as label_col

    label_col: column in cluster_meta to colorize points by
    """
    import umap
    figsize = (15, 4)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i, cre_line in enumerate(get_cre_lines(cluster_meta)):
        # get number of unique values in label_col to color by
        cre_meta = cluster_meta[cluster_meta.cre_line == cre_line]
        cre_csids = cre_meta.index.values
        n_cols = len(cre_meta[label_col].unique())
        palette = sns.color_palette('hls', n_cols)
        # get feature matrix for this cre line
        feature_matrix_cre = processing.get_feature_matrix_for_cre_line(feature_matrix, cluster_meta, cre_line)
        # fit umap to dropouts for this cre line
        X = feature_matrix_cre.values
        fit = umap.UMAP()
        u = fit.fit_transform(X)
        labels = [cluster_meta.loc[cell_specimen_id][label_col] for cell_specimen_id in cre_csids]
        umap_df = pd.DataFrame()
        umap_df['x'] = u[:, 0]
        umap_df['y'] = u[:, 1]
        umap_df['labels'] = labels
        ax[i] = sns.scatterplot(data=umap_df, x='x', y='y', hue='labels', size=1, ax=ax[i], palette=palette)
        ax[i].set_xlabel('UMAP 0')
        ax[i].set_ylabel('UMAP 1')
        ax[i].legend(fontsize='x-small', title=label_col, title_fontsize='x-small', bbox_to_anchor=(1, 1))
        ax[i].set_title(get_cell_type_for_cre_line(cluster_meta, cre_line))
    fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'UMAP_' + label_col)


def plot_umap_with_labels(X, labels, ax=None, filename_string=''):
    fit = umap.UMAP()
    u = fit.fit_transform(X)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(u[:, 0], u[:, 1], c=labels)
    plt.tight_layout()
    ax.set_title(filename_string)
    return ax


def plot_cluster_density(df_dropouts=None, labels_list=None, cluster_corrs=None, ax=None):
    if cluster_corrs is None:
        cluster_corrs = get_cluster_density(df_dropouts, labels_list)
    df_labels = pd.DataFrame(columns=['corr', 'labels'])
    for key in cluster_corrs.keys():
        data = np.array([cluster_corrs[key], np.repeat(key, len(cluster_corrs[key]))])
        tmp = pd.DataFrame(data.transpose(), columns=['corr', 'labels'])
        df_labels = df_labels.append(tmp, ignore_index=True)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(max(cluster_corrs.keys()), 6))
    ax.axhline(xmin=-1, xmax=max(cluster_corrs.keys()) + 1, color='grey')
    # sns.violinplot(data=df_labels, x='labels', y='corr', ax=ax)
    sns.boxplot(data=df_labels, x='labels', y='corr', ax=ax)
    return ax


def plot_within_cluster_correlations_for_cre_line(cluster_meta, cre_line, sort_order=None, suffix='_cluster_id_sort',
                                                  save_dir=None, folder=None, ax=None):
    """
    plot distribution of within cluster correlations for a given cre line

    sort_order: can be 'cluster_id' or 'manual_sort_order'
    """

    # get feature_matrix for this cre line
    cluster_meta_cre = cluster_meta[cluster_meta.cre_line == cre_line]
    cluster_ids = np.sort(cluster_meta_cre.cluster_id.unique())
    n_clusters = len(cluster_ids)
    # create fig proportional to number clusters
    if ax is None:
        figsize = (0.5 * n_clusters, 3)
        fig, ax = plt.subplots(figsize=figsize)
    # get xaxis orrder based on provided sort_order
    if sort_order is None:
        order = cluster_ids
    else:
        order = sort_order[cre_line]
    # violin or boxplot of cluster correlation values
    ax = sns.boxplot(data=cluster_meta_cre, x='cluster_id', y='within_cluster_correlation',
                     order=order, ax=ax, color='white', width=0.5)
    # add line at 0
    ax.axhline(y=0, xmin=0, xmax=1, color='grey', linestyle='--')
    ax.set_title(get_cell_type_for_cre_line(cluster_meta, cre_line))
    ax.set_ylabel('correlation')
    ax.set_xlabel('cluster #')
    ax.set_ylim(-1.1, 1.1)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder,
                          'within_cluster_correlations_' + cre_line.split('-')[0] + suffix)
    return ax


def plot_within_cluster_correlations_all_cre(cluster_meta, n_clusters_cre, sort_order=None, suffix='_cluster_id_sort', save_dir=None, folder=None):
    """
    plot distribution of within cluster correlations for all cre lines, sorted by sort_order

    sort_order: dict with cre_lines as keys, order of cluster to sort by as values
    """
    cre_lines = get_cre_lines(cluster_meta)
    n_clusters = [n_clusters_cre[cre] for cre in cre_lines]

    figsize = (12, 3)
    fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': n_clusters})
    for i, cre_line in enumerate(cre_lines):
        ax[i] = plot_within_cluster_correlations_for_cre_line(cluster_meta, cre_line, sort_order, ax=ax[i])

    fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'within_cluster_correlations_' + suffix)


def plot_dropout_heatmap(cluster_meta, feature_matrix, cre_line, cluster_id, small_fontsize=False, ax=None):
    """
    Plots the average dropout score heatmap for a given cluster within a cre line
    Labels plot with original_cluster_id

    cluster_meta: dataframe with cell_specimen_id as index, cluster_id, cre_line and other metadata as columns
    feature_matrix: input to clustering - dataframe of dropout scores for features x experience levels,
                    index must be cell_specimen_id
    cre_line: string, name of cre line this cluster is in
    cluster_id: cluster ID within cre line; must be cluster_id corresponding to cluster size
    """
    # check if there are negative values in feature_matrix, if so, use diff cmap and set vmin to -1
    if len(np.where(feature_matrix < 0)[0]) > 0:
        vmin = -1
        cmap = 'RdBu'
    else:
        vmin = 0
        cmap = 'Blues'
    cre_csids = cluster_meta[(cluster_meta['cre_line'] == cre_line)].index.values
    this_cluster_meta = cluster_meta[(cluster_meta['cluster_id'] == cluster_id) &
                                     (cluster_meta['cre_line'] == cre_line)]
    this_cluster_csids = this_cluster_meta.index.values
    feature_matrix_cre = processing.get_feature_matrix_for_cre_line(feature_matrix, cluster_meta, cre_line)
    mean_dropout_df = feature_matrix_cre.loc[this_cluster_csids].mean().unstack()
    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.heatmap(mean_dropout_df, cmap=cmap, vmin=vmin, vmax=1, ax=ax, cbar=False)
    # fraction is number of cells in this cluster vs all cells in this cre line
    fraction_cre = len(this_cluster_csids) / float(len(cre_csids))
    fraction = np.round(fraction_cre * 100, 1)
    # set title and labels
    ax.set_title('cluster ' + str(cluster_id) + '\n' + str(fraction) + '%, n=' + str(len(this_cluster_csids)),
                 fontsize=12)
    ax.set_yticklabels(mean_dropout_df.index.values, rotation=0, fontsize=14)
    ax.set_xticklabels(mean_dropout_df.columns.values, rotation=90, fontsize=14)
    ax.set_ylim(0, mean_dropout_df.shape[0])
    ax.set_xlabel('')
    if small_fontsize:
        ax = standardize_axes_fontsize(ax)
    return ax


def plot_dropout_heatmaps_and_save_to_cell_examples_folders(cluster_meta, feature_matrix, save_dir):
    """
    loops through each cluster in each cre line and saves average dropout heatmap to the matched_cell_examples folder
    within matched_cell_examples, creates a folder for each cre line and each cluster ID and saves plot there
    """
    cre_lines = get_cre_lines(cluster_meta)
    for cre_line in cre_lines:
        clusters = np.sort(cluster_meta[cluster_meta.cre_line == cre_line].cluster_id.unique())
        for i, cluster_id in enumerate(clusters):
            # plot each cluster separately
            figsize = (3, 2.5)
            fig, ax = plt.subplots(figsize=figsize)
            ax = plot_dropout_heatmap(cluster_meta, feature_matrix, cre_line, cluster_id, ax=ax)
            # save in same folder as matched cell examples for this cre line and cluster ID
            plot_save_dir = os.path.join(save_dir, 'matched_cell_examples', cre_line)
            if not os.path.exists(plot_save_dir):
                os.mkdir(plot_save_dir)
            folder = 'cluster_' + str(int(cluster_id))
            filename = 'average_cluster_heatmap_' + cre_line.split('-')[0] + '_cluster_' + str(cluster_id)
            utils.save_figure(fig, figsize, plot_save_dir, folder, filename)


def plot_dropout_heatmaps_for_clusters(cluster_meta, feature_matrix, sort_col='cluster_id', save_dir=None, folder=None):
    """
    Plot average dropout score heatmap for each cluster in each cre_line
    Plot format is 6 rows x 1-3 columns depending on how many clusters there are

    sort_col: column in cluster_meta to sort clusters by for plotting
                can be 'labels', 'cluster_id', or 'manual_sort_order'
    """

    for cre_line in get_cre_lines(cluster_meta):

        # plot clusters in order of sort_col
        clusters = np.sort(cluster_meta[cluster_meta.cre_line == cre_line][sort_col].unique())
        n_clusters = len(clusters)

        if n_clusters <= 6:
            figsize = (1.75, 14)
            fig, ax = plt.subplots(6, 1, figsize=figsize, sharex=True, sharey=True)
        elif n_clusters <= 12:
            figsize = (4, 14)
            fig, ax = plt.subplots(6, 2, figsize=figsize, sharex=True, sharey=True)
        elif n_clusters <= 18:
            figsize = (5.5, 14)
            fig, ax = plt.subplots(6, 3, figsize=figsize, sharex=True, sharey=True)
        else:
            print('too many clusters to plot, adapt plotting code if needed')
        ax = ax.ravel()

        # loop through clusters in sorted order
        for i, cluster_id in enumerate(clusters):
            # generate and label plot for original cluster_id
            original_cluster_id = cluster_meta[(cluster_meta.cre_line == cre_line) & (
                cluster_meta[sort_col] == cluster_id)].original_cluster_id.unique()[0]
            ax[i] = plot_dropout_heatmap(cluster_meta, feature_matrix, cre_line, original_cluster_id, ax=ax[i])

        plt.suptitle(get_cell_type_for_cre_line(cluster_meta, cre_line), x=0.51, y=.95, fontsize=16)
        plt.subplots_adjust(hspace=0.6, wspace=0.4)
        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder,
                              'cluster_heatmaps_' + sort_col + '_' + cre_line.split('-')[0])


def plot_average_dropout_heatmap_for_cre_lines(dropouts, cluster_meta, save_dir=None, folder=None):
    """
    plot average dropout score for each cre line
    """
    dropouts_meta = dropouts.merge(cluster_meta[['cre_line', 'binned_depth', 'targeted_structure']], on='cell_specimen_id')

    figsize = (12, 2)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(get_cre_lines(cluster_meta)):
        mean_dropouts = dropouts_meta[dropouts_meta.cre_line == cre_line].groupby('experience_level').mean()[processing.get_features_for_clustering()]
        ax[i] = sns.heatmap(mean_dropouts.T, cmap='Blues', vmin=0, vmax=0.75, ax=ax[i], cbar_kws={'shrink': 0.7, 'label': 'coding score'})
        ax[i].set_title(get_cell_type_for_cre_line(cluster_meta, cre_line))
        ax[i].set_ylim(0, 4)
        ax[i].set_xlabel('')
    plt.subplots_adjust(wspace=0.6)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'average_dropout_heatmaps')


def plot_clusters(dropout_df, cluster_df=None, plot_difference=False, mean_response_df=None, save_plots=False, path=None):
    '''
    Plots heatmaps and descriptors of clusters.
    dropout_df: dataframe of dropout scores, n cells by n regressors by experience level
    cluster_df: df with cluster id and cell specimen id columns
    plot_difference: Boolean to plot difference (cluster - population) of mean dropout scores on the heatmap
    mean_response_df: dataframe with mean responseswith cell specimen id, timestamps, and mean_response columns

    :return:
    '''

    # Set up the variables

    cluster_ids = cluster_df['cluster_id'].value_counts().index.values  # sort cluster ids by size
    n_clusters = len(cluster_ids)
    palette = utils.get_cre_line_colors()
    palette_exp = utils.get_experience_level_colors()
    depths = [75, 175, 275, 375]
    areas = ['VISp', 'VISl']

    # get number of animals per cluster
    # grouped_df = cluster_df.groupby('cluster_id')
    # N_mice = grouped_df.agg({"mouse_id": "nunique"})

    # Set up figure
    fig, ax = plt.subplots(5, n_clusters, figsize=(n_clusters * 2, 14),
                           sharex='row',
                           sharey='row',
                           gridspec_kw={'height_ratios': [3, 2, 2, 1, 2]},
                           )
    ax = ax.ravel()
    for i, cluster_id in enumerate(cluster_ids):

        # 1. Mean dropout scores
        this_cluster_ids = cluster_df[cluster_df['cluster_id'] == cluster_id]['cell_specimen_id'].unique()
        mean_dropout_df = dropout_df.loc[this_cluster_ids].mean().unstack()
        if plot_difference is False:
            ax[i] = sns.heatmap(mean_dropout_df,
                                cmap='RdBu',
                                vmin=-1,
                                vmax=1,
                                ax=ax[i],
                                cbar=False, )
        elif plot_difference is True:
            mean_dropout_df_diff = mean_dropout_df.abs() - dropout_df.mean().unstack().abs()
            ax[i] = sns.heatmap(mean_dropout_df_diff,
                                cmap='RdBu',
                                vmin=-1,
                                vmax=1,
                                ax=ax[i],
                                cbar=False, )

        # 2. By cre line
        # % of total cells in this cluster
        within_cluster_df = \
            cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                'cell_type').count()[['cluster_id']]
        n_cells = within_cluster_df.sum().values[0]
        all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('cell_type').count()[['cluster_id']]
        fraction_cre = within_cluster_df / all_df
        fraction_cre.sort_index(inplace=True)
        # add numerical column for sns.barplot
        fraction_cre['cell_type_index'] = np.arange(0, fraction_cre.shape[0])
        fraction = np.round(fraction_cre[['cluster_id']].values * 100, 1)

        ax[i + len(cluster_ids)] = sns.barplot(data=fraction_cre,
                                               y='cluster_id',
                                               x='cell_type_index',
                                               palette=palette,
                                               ax=ax[i + len(cluster_ids)])
        if fraction_cre.shape[0] == 3:
            ax[i + len(cluster_ids)].set_xticklabels(['Exc', 'SST', 'VIP'], rotation=90)
        # else:
        #    ax[i + len(cluster_ids)].set_xticklabels(fraction_cre.index.values, rotation=90)
        ax[i + len(cluster_ids)].set_ylabel('fraction cells\nper class')
        ax[i + len(cluster_ids)].set_xlabel('')
        # ax[i+ len(cluster_ids)].set_title('n mice = ' + str(N_mice.loc[cluster_id].values), fontsize=16)

        # set title and labels
        ax[i].set_title('cluster ' + str(i) + '\n' + str(fraction[0][0]) + '%, n=' + str(n_cells), fontsize=16)
        ax[i].set_yticklabels(mean_dropout_df.index.values, rotation=0)
        ax[i].set_ylim(-0.5, 4.5)
        ax[i].set_xlabel('')

        # 3. Plot by depth
        within_cluster_df = \
            cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                'binned_depth').count()[['cluster_id']]
        all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('binned_depth').count()[['cluster_id']]
        fraction_depth = within_cluster_df / all_df
        fraction_depth.reset_index(inplace=True)
        ax[i + (len(cluster_ids) * 2)] = sns.barplot(data=fraction_depth,
                                                     x='binned_depth',
                                                     y='cluster_id',
                                                     order=depths,
                                                     palette='gray',
                                                     ax=ax[i + (len(cluster_ids) * 2)])
        # set labels
        ax[i + (len(cluster_ids) * 2)].set_xlabel('depth (um)')
        ax[i + (len(cluster_ids) * 2)].set_ylabel('fraction cells\nper depth')
        ax[i + (len(cluster_ids) * 2)].set_xticks(np.arange(0, len(depths)))
        ax[i + (len(cluster_ids) * 2)].set_xticklabels(depths, rotation=90)

        # 4. Plot by area
        within_cluster_df = \
            cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                'targeted_structure').count()[['cluster_id']]
        all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('targeted_structure').count()[['cluster_id']]
        fraction_area = within_cluster_df / all_df
        fraction_area.reset_index(inplace=True, drop=False)

        ax[i + (len(cluster_ids) * 3)] = sns.barplot(data=fraction_area,
                                                     # orient='h',
                                                     x='targeted_structure',
                                                     y='cluster_id',
                                                     order=areas,
                                                     palette='gray',
                                                     ax=ax[i + (len(cluster_ids) * 3)])
        # set labels
        ax[i + (len(cluster_ids) * 3)].set_xlabel('area')
        ax[i + (len(cluster_ids) * 3)].set_ylabel('fraction cells\nper area')
        ax[i + (len(cluster_ids) * 3)].set_xticklabels(areas, rotation=0)

        # plot mean traces

        # axes_column = 'cluster_id'
        hue_column = 'experience_level'
        hue_conditions = np.sort(cluster_df[hue_column].unique())
        timestamps = cluster_df['trace_timestamps'][0]
        xlim_seconds = [-1, 1.5]
        change = False
        omitted = True
        xlabel = 'time (sec)'

        for c, hue in enumerate(hue_conditions):

            traces = cluster_df[(cluster_df['cluster_id'] == cluster_id) &
                                (cluster_df[hue_column] == hue)].mean_trace.values
            for t in range(0, np.shape(traces)[0]):
                traces[t] = signal.resample(traces[t], len(timestamps))
            ax[i + (len(cluster_ids) * 4)] = utils.plot_mean_trace(np.asarray(traces),
                                                                   timestamps, ylabel='response',
                                                                   legend_label=hue,
                                                                   color=palette_exp[c],
                                                                   interval_sec=1,
                                                                   plot_sem=False,
                                                                   xlim_seconds=xlim_seconds,
                                                                   ax=ax[i + (len(cluster_ids) * 4)])
            ax[i + (len(cluster_ids) * 4)] = utils.plot_flashes_on_trace(ax[i + (len(cluster_ids) * 4)], timestamps,
                                                                         change=change, omitted=omitted)
            ax[i + (len(cluster_ids) * 4)].axvline(x=0, ymin=0, ymax=1, linestyle='--', color='gray')
            ax[i + (len(cluster_ids) * 4)].set_title('')
            ax[i + (len(cluster_ids) * 4)].set_xlim(xlim_seconds)
            ax[i + (len(cluster_ids) * 4)].set_xlabel(xlabel)

        if i != 0:
            ax[i + len(cluster_ids)].set_ylabel('')
            ax[i + (len(cluster_ids) * 2)].set_ylabel('')
            ax[i + (len(cluster_ids) * 3)].set_ylabel('')
            ax[i + (len(cluster_ids) * 4)].set_ylabel('')
    return fig
    plt.tight_layout()


def plot_clusters_columns_all_cre_lines(df, df_meta, labels_cre, multi_session_df, save_dir=None):
    """
    plots dropout scores per cluster, per cre line, in a compact format
    df: reshaped dataframe of dropout scores for matched cells
    df_meta: metadata for each cell in df
    labels_cre: dict of cluster labels for each cre line
    multi_session_df: dataframe with trial averaged traces to plot per cluster
    """
    cells_table = loading.get_cell_table()
    cre_lines = np.sort(df_meta.cre_line.unique())[::-1]

    for cre_line in cre_lines:
        cell_type = df_meta[df_meta.cre_line == cre_line].cell_type.unique()[0]

        print(cre_line, cell_type)
        cids = df_meta[df_meta['cre_line'] == cre_line]['cell_specimen_id']
        df_sel = df.loc[cids]
        labels = labels_cre[cre_line]
        cluster_df = pd.DataFrame(index=df_sel.index, columns=['cluster_id'], data=labels)
        cluster_df = cluster_df.merge(cells_table[['cell_specimen_id', 'cell_type', 'cre_line', 'experience_level',
                                                   'binned_depth', 'targeted_structure']], on='cell_specimen_id')
        cluster_df = cluster_df.drop_duplicates(subset='cell_specimen_id')
        cluster_df.reset_index(inplace=True)

        dropout_df = df_sel.copy()

        cluster_mdf = multi_session_df.merge(cluster_df[['cell_specimen_id', 'cluster_id']],
                                             on='cell_specimen_id',
                                             how='inner')

        # Set up the variables

        # cluster_ids = cluster_df['cluster_id'].value_counts().index.values  # sort cluster ids by size
        cluster_ids = cluster_df.groupby(['cluster_id']).count()[['cell_specimen_id']].sort_values(by='cell_specimen_id').index.values
        n_clusters = len(cluster_ids)
        # palette = utils.get_cre_line_colors()
        palette_exp = utils.get_experience_level_colors()
        depths = [75, 175, 275, 375]
        areas = ['VISp', 'VISl']

        # get number of animals per cluster
        # grouped_df = cluster_df.groupby('cluster_id')
        # N_mice = grouped_df.agg({"mouse_id": "nunique"})

        # Set up figure
        n_cols = 4
        figsize = (12, n_clusters * 2,)
        fig, ax = plt.subplots(n_clusters, n_cols, figsize=figsize, sharex='col', gridspec_kw={'width_ratios': [2, 3, 2, 1.25]})
        ax = ax.ravel()
        for i, cluster_id in enumerate(cluster_ids[::-1]):

            # % of total cells in this cluster
            within_cluster_df = cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                'cell_type').count()[['cluster_id']]
            all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('cell_type').count()[['cluster_id']]
            n_cells = within_cluster_df.cluster_id.values[0]
            fraction = within_cluster_df / all_df
            fraction = np.round(fraction.cluster_id.values[0] * 100, 1)

            # 1. Mean dropout scores
            this_cluster_ids = cluster_df[cluster_df['cluster_id'] == cluster_id]['cell_specimen_id'].unique()
            mean_dropout_df = dropout_df.loc[this_cluster_ids].mean().unstack()
            ax[(i * n_cols)] = sns.heatmap(mean_dropout_df,
                                           cmap='RdBu',
                                           vmin=-1,
                                           vmax=1,
                                           ax=ax[(i * n_cols)],
                                           cbar=False, )

            # set title and labels
        #     ax[(i*n_cols)].set_title('cluster ' + str(i) + '\n' + str(fraction[0][0]) + '%, n=' + str(n_cells),  fontsize=14)
            ax[(i * n_cols)].set_title(str(fraction) + '%, n=' + str(n_cells), fontsize=14)
            ax[(i * n_cols)].set_yticks(np.arange(0.5, 4.5))
            ax[(i * n_cols)].set_yticklabels(mean_dropout_df.index.values, rotation=0, fontsize=15)
            ax[(i * n_cols)].set_ylim(0, 4)
            ax[(i * n_cols)].set_xlabel('')
            ax[(i * n_cols)].set_ylabel('cluster ' + str(i), fontsize=15)

            # 3. Plot by depth
            n_col = 2
            within_cluster_df = \
                cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                    'binned_depth').count()[['cluster_id']]
            all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('binned_depth').count()[['cluster_id']]
            fraction_depth = within_cluster_df / all_df
            fraction_depth.reset_index(inplace=True)
            ax[(i * n_cols) + n_col] = sns.barplot(data=fraction_depth, orient='h',
                                                   y='binned_depth',
                                                   x='cluster_id',
                                                   # orient='h',
                                                   palette='gray',
                                                   ax=ax[(i * n_cols) + n_col])
            # set labels
            ax[(i * n_cols) + n_col].set_ylabel('depth (um)', fontsize=15)
            ax[(i * n_cols) + n_col].set_xlabel('')
            ax[(i * n_cols) + n_col].set_ylim(-0.5, len(depths) - 0.5)

            ax[(i * n_cols) + n_col].set_yticks(np.arange(0, len(depths)))
            ax[(i * n_cols) + n_col].set_yticklabels(depths, rotation=0)
            ax[(i * n_cols) + n_col].invert_yaxis()

            # 4. Plot by area
            n_col = 3
            within_cluster_df = \
                cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                    'targeted_structure').count()[['cluster_id']]
            all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('targeted_structure').count()[['cluster_id']]
            fraction_area = within_cluster_df / all_df
            fraction_area.reset_index(inplace=True, drop=False)

            ax[(i * n_cols) + n_col] = sns.barplot(data=fraction_area,
                                                   # orient='h',
                                                   x='targeted_structure',
                                                   y='cluster_id',
                                                   palette='gray',
                                                   ax=ax[(i * n_cols) + n_col])
            # set labels
            ax[(i * n_cols) + n_col].set_xlabel('')
            ax[(i * n_cols) + n_col].set_ylabel('fraction cells\nper area', fontsize=15)
            ax[(i * n_cols) + n_col].set_xticklabels(areas, rotation=0)

            # plot mean traces

            n_col = 1
            # axes_column = 'cluster_id'
            hue_column = 'experience_level'
            hue_conditions = np.sort(cluster_mdf[hue_column].unique())
            timestamps = cluster_mdf['trace_timestamps'][0]
            xlim_seconds = [-1, 1.5]
            change = False
            omitted = True
            # xlabel = 'time (sec)'

            for c, hue in enumerate(hue_conditions):

                traces = cluster_mdf[(cluster_mdf['cluster_id'] == cluster_id) &
                                     (cluster_mdf[hue_column] == hue)].mean_trace.values
                for t in range(0, np.shape(traces)[0]):
                    traces[t] = signal.resample(traces[t], len(timestamps))
                ax[(i * n_cols) + n_col] = utils.plot_mean_trace(np.asarray(traces),
                                                                 timestamps, ylabel='response',
                                                                 legend_label=hue,
                                                                 color=palette_exp[c],
                                                                 interval_sec=1,
                                                                 plot_sem=False,
                                                                 xlim_seconds=xlim_seconds,
                                                                 ax=ax[(i * n_cols) + n_col])
                ax[(i * n_cols) + n_col] = utils.plot_flashes_on_trace(ax[(i * n_cols) + n_col], timestamps,
                                                                       change=change, omitted=omitted)
                ax[(i * n_cols) + n_col].axvline(x=0, ymin=0, ymax=1, linestyle='--', color='gray')
                ax[(i * n_cols) + n_col].set_title('')
                ax[(i * n_cols) + n_col].set_xlim(xlim_seconds)
                ax[(i * n_cols) + n_col].set_xlabel('')
                ax[(i * n_cols) + n_col].set_ylabel('response', fontsize=15)

        ax[(i * n_cols) + 2].set_xlabel('fraction cells\nper depth')
        ax[(i * n_cols) + 3].set_xlabel('area')
        ax[(i * n_cols) + 1].set_xlabel('time (sec)')

        plt.suptitle(cell_type, x=0.54, y=1.01)
        plt.tight_layout()
        if save_dir:
            utils.save_figure(fig, figsize, save_dir, 'clustering', cre_line + '_clusters_column')


def plot_clusters_compact_all_cre_lines(df, df_meta, labels_cre, save_dir=None):
    """
    plots dropout scores per cluster, per cre line, in a compact format
    df: reshaped dataframe of dropout scores for matched cells
    df_meta: metadata for each cell in df
    labels_cre: dict of cluster labels for each cre line
    """
    cells_table = loading.get_cell_table()
    cre_lines = np.sort(df_meta.cre_line.unique())[::-1]

    for cre_line in cre_lines:
        cell_type = df_meta[df_meta.cre_line == cre_line].cell_type.unique()[0]

        print(cre_line, cell_type)
        # cids = df_meta[df_meta['cre_line'] == cre_line]['cell_specimen_id']
        # df_sel = df.loc[cids]
        df_sel = df[cre_line]
        labels = labels_cre[cre_line]
        cluster_df = pd.DataFrame(index=df_sel.index, columns=['cluster_id'], data=labels)
        cluster_df = cluster_df.merge(cells_table[['cell_specimen_id', 'cell_type', 'cre_line', 'experience_level',
                                                   'binned_depth', 'targeted_structure']], on='cell_specimen_id')
        cluster_df = cluster_df.drop_duplicates(subset='cell_specimen_id')
        cluster_df.reset_index(inplace=True)

        dropout_df = df_sel.copy()

        # Set up the variables

        # cluster_ids = cluster_df['cluster_id'].value_counts().index.values  # sort cluster ids by size
        cluster_ids = cluster_df.groupby(['cluster_id']).count()[['cell_specimen_id']].sort_values(by='cell_specimen_id').index.values
        n_clusters = len(cluster_ids)

        # get number of animals per cluster
        # grouped_df = cluster_df.groupby('cluster_id')
        # N_mice = grouped_df.agg({"mouse_id": "nunique"})

        # Set up figure
        n_rows = int(np.ceil(n_clusters / 3.))
        figsize = (5, 1.9 * n_rows)
        fig, ax = plt.subplots(n_rows, 3, figsize=figsize, sharex=True, sharey=True)
        ax = ax.ravel()
        for i, cluster_id in enumerate(cluster_ids[::-1]):

            # % of total cells in this cluster
            within_cluster_df = cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                'cell_type').count()[['cluster_id']]
            all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('cell_type').count()[['cluster_id']]
            n_cells = within_cluster_df.cluster_id.values[0]
            fraction = within_cluster_df / all_df
            fraction = np.round(fraction.cluster_id.values[0] * 100, 1)

            # 1. Mean dropout scores
            this_cluster_ids = cluster_df[cluster_df['cluster_id'] == cluster_id]['cell_specimen_id'].unique()
            mean_dropout_df = dropout_df.loc[this_cluster_ids].mean().unstack()
            ax[i] = sns.heatmap(mean_dropout_df,
                                cmap='RdBu',
                                vmin=-1,
                                vmax=1,
                                ax=ax[i],
                                cbar=False, )

            # set title and labels
            ax[i].set_title('cluster ' + str(i) + '\n' + str(fraction) + '%, n=' + str(n_cells), fontsize=14)
            ax[i].set_yticks(np.arange(0.5, 4.5))
            ax[i].set_yticklabels(mean_dropout_df.index.values, rotation=0, fontsize=15)
            ax[i].set_ylim(0, 4)
            ax[i].set_xlabel('')

        plt.suptitle(cell_type, x=0.5, y=1.01)
        plt.subplots_adjust(wspace=0.4, hspace=0.8)
        if save_dir:
            utils.save_figure(fig, figsize, save_dir, 'clustering', cre_line + '_clusters_compact')


def get_cluster_colors(labels):
    '''
    generates a list of unique colors for each cluster for plots
    :param labels: list of cluster labels, length of N cells
    :return: label_colors: a list of strings for color names
    '''

    color_map = get_cluster_color_map(labels)
    label_colors = []
    for label in labels:
        label_colors.append(color_map[label])
    return label_colors


def get_cluster_color_map(labels):
    '''
    # generates a dictionary of cluster label: color
    :param labels: list of cluster labels, length of N cells
    :return: dictionary color number: color string
    '''
    unique_labels = np.sort(np.unique(labels))
    color_map = {}
    # r = int(random.random() * 256)
    # g = int(random.random() * 256)
    # b = int(random.random() * 256)
    # step = 256 / len(unique_labels)
    for i in unique_labels:
        # r += step
        # g += step
        # b += step
        # r = int(r) % 256
        # g = int(g) % 256
        # b = int(b) % 256
        # color_map[i] = (r, g, b)
        color_map[i] = '#%06X' % random.randint(0, 0xFFFFFF)
    return color_map


def plot_N_clusters_by_cre_line(labels_cre, ax=None, palette=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if palette is None:
        palette = [(1.0, 0.596078431372549, 0.5882352941176471),
                   (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
                   (0.7725490196078432, 0.6901960784313725, 0.8352941176470589)]
    for i, cre_line in enumerate(labels_cre.keys()):
        labels = labels_cre[cre_line]
        (unique, counts) = np.unique(labels, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        frequencies = sorted_counts / sum(counts) * 100
        cumulative_sum = [0]
        for freq in frequencies:
            cumulative_sum.append(cumulative_sum[-1] + freq)
        ax.grid()
        ax.plot(range(0, len(cumulative_sum)), cumulative_sum, 'o-', color=palette[i],
                linewidth=4, markersize=10)
        ax.set_xlabel('Cluster number')
        ax.set_ylabel('Cells per cluster (%)')
    ax.legend(['Excitatory', 'SST inhibitory', 'VIP inhibitory'])

    return ax


def plot_random_subset_of_cells_per_cluster(cluster_meta, dropouts, save_dir=None):
    """
    plots dropout scores for up to 100 randomly selected cells from each cluster for each cre line on a single plot
    helpful to evaluate within cluster variability
    """

    experience_levels = np.sort(cluster_meta.experience_level.unique())
    features = processing.get_features_for_clustering()
    dropouts = dropouts.set_index('cell_specimen_id')

    for cre_line in get_cre_lines(cluster_meta):
        cluster_data = cluster_meta[cluster_meta.cre_line == cre_line].copy()
        cluster_data = cluster_data.reset_index()
        cluster_data = cluster_data.sort_values(by=['targeted_structure', 'binned_depth'])
        cluster_labels = cluster_data[cluster_data.cre_line == cre_line].cluster_id.unique()

        for cluster_id in cluster_labels:
            cluster_csids = cluster_data[cluster_data.cluster_id == cluster_id].cell_specimen_id.values
            print(len(cluster_csids), 'cells in cluster', cluster_id)
            if len(cluster_csids) < 100:
                n_cells = len(cluster_csids)
            else:
                n_cells = 100

            print('selecting a random subset of', n_cells)
            cluster_csids = random.sample(list(cluster_csids), n_cells)

            figsize = (20, 20)
            fig, ax = plt.subplots(10, 10, figsize=figsize, sharex=True, sharey=True)
            ax = ax.ravel()
            for i, cell_specimen_id in enumerate(cluster_csids[:n_cells]):

                cell_info = cluster_meta.loc[cell_specimen_id]['targeted_structure'] + '_' + str(cluster_meta.loc[cell_specimen_id]['imaging_depth'])
                mean_dropouts = dropouts.loc[cell_specimen_id].groupby('experience_level').mean()[features]
                ax[i] = sns.heatmap(mean_dropouts.T, cmap='Blues', vmin=0, vmax=1, ax=ax[i], cbar=False)  # cbar_kws={'shrink':0.7, 'label':model_output_type})
                ax[i].set_title(str(cell_specimen_id) + '_' + cell_info, fontsize=10)
                ax[i].set_ylim(0, 4)
                ax[i].set_yticks(np.arange(0.5, len(features) + 0.5))
                ax[i].set_yticklabels(features, fontsize=14)
                ax[i].set_xlabel('')
            for i in range(90, 100):
                ax[i].set_xticklabels(experience_levels, rotation=90, fontsize=14)
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.3)
            fig.suptitle(get_cell_type_for_cre_line(cluster_meta, cre_line) + ' cluster ' + str(cluster_id), x=0.5, y=1.01)
            filename = cre_line + '_cluster_' + str(cluster_id)
            if save_dir:
                utils.save_figure(fig, figsize, save_dir, 'dropout_heatmaps_per_cluster', filename)


def plot_fraction_cells_by_area_depth(cluster_meta, n_clusters_cre, normalize=True, label='fraction of cells', save_dir=None, folder=None):
    """
    plots either the number (normalize=False) or fraction (normalize=True) of cells per area and depth for each cluster
    fraction of cells is computed by taking the total number of cells for each area and depth
    and computing what fraction of that total belongs in each cluster
    does not explicitly account for sampling bias
    """
    cre_lines = get_cre_lines(cluster_meta)
    n_clusters = [n_clusters_cre[cre] for cre in cre_lines]

    figsize = (24, 2)
    fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': n_clusters})

    for i, cre_line in enumerate(cre_lines):
        cre_meta = cluster_meta[cluster_meta.cre_line == cre_line]
        frequency = processing.make_frequency_table(cre_meta, groupby_columns=['targeted_structure', 'layer'], normalize=normalize)
        frequency = frequency.T  # transpose so that rows are locations and cols are cluster IDs
        ax[i] = sns.heatmap(frequency, vmin=0, cmap='Purples', ax=ax[i], cbar_kws={'shrink': 0.8, 'label': label})
        ax[i].set_ylim((0, len(frequency.index)))
        ax[i].set_xlim(-0.5, len(frequency.columns) + 0.5)
        ax[i].set_ylabel('')
        ax[i].set_title(get_cell_type_for_cre_line(cluster_meta, cre_line))
    plt.subplots_adjust(wspace=0.5)
    if save_dir:
        if normalize:
            filename = 'fraction_cells_per_area_depth'
        else:
            filename = 'number_cells_per_area_depth'
        utils.save_figure(fig, figsize, save_dir, folder, filename)


def plot_cell_stats_per_cluster_for_areas_depths(cluster_meta, cell_count_stats, n_clusters_cre, value_to_plot,
                                                 cbar_label='counts', cmap='Purples', vmin=0, vmax=1,
                                                 cluster_order=None, save_dir=None, folder=None, suffix=None):
    """
    plot heatmap with area/depth combinations as rows and cluster IDs as columns, with values defined by values_to_plot

    cell_count_stats: dataframe with one row for each cre_line, cluster, area, and depth and columns for
                        different cell count values (ex: 'relative_to_random' which is the fraction of cells per loc per cluster relative to chance)
    value_to_plot: column in cell_count_stats to use for heatmap values
    cluster_order: dictionary with keys for cre lines where values are order of cluster IDs to arrange xaxis by
    """
    cre_lines = get_cre_lines(cluster_meta)
    n_clusters = [n_clusters_cre[cre] for cre in cre_lines]
    # get area and depth combos in correct order
    locations = np.sort(cell_count_stats.location.unique())

    figsize = (24, 2.25)
    fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': n_clusters}, sharey=False)

    for i, cre_line in enumerate(cre_lines):

        # pivot stats to get number of cells per area and depth by cluster ID
        data = cell_count_stats[(cell_count_stats.cre_line == cre_line)]
        data = data.pivot(index='cluster_id', columns='location', values=value_to_plot)
        # sort area / depth combos
        data = data[locations]
        # sort clusters
        if cluster_order:
            data = data.loc[cluster_order[cre_line]]
        ax[i] = sns.heatmap(data.T, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax[i],
                            cbar_kws={'shrink': 0.8, 'label': cbar_label})
        ax[i].set_ylim((0, len(data.columns)))
        ax[i].set_xticklabels(data.index, )  # rotation=0, horizontalalignment='center')
        ax[i].set_xlim(0, len(data.index))
        ax[i].set_ylabel('')
        ax[i].set_xlabel('cluster #')
        ax[i].set_title(get_cell_type_for_cre_line(cluster_meta, cre_line))
    # ax[i].set_xlim(ax[i].get_xlim()[::-1])
    plt.subplots_adjust(wspace=0.4)
    if save_dir:
        filename = value_to_plot + '_per_area_depth_' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, filename)


def plot_dropout_heatmaps_for_clusters_sorted(cluster_meta, feature_matrix, cluster_order=None,
                                              save_dir=None, folder=None, sort_type=''):
    """
    Plot average dropout score heatmap for each cluster in each cre_line
    Plot format is 6 rows x 1-3 columns depending on how many clusters there are

    cluster_order: dictionary with keys for cre_lines, values as desired order of clusters for each cre line
    sort_type: string indicating how clusters are sorted to use in saved filename
    """

    for cre_line in get_cre_lines(cluster_meta):

        if cluster_order is None:  # if cluster_order not provided, sort by original cluster_id
            clusters = np.sort(cluster_meta[cluster_meta.cre_line == cre_line].cluster_id.unique())
            sort_type = 'cluster_id'
        else:  # otherwise sort by provided cluster_order for this cre_line
            clusters = cluster_order[cre_line]
        n_clusters = len(clusters)

        if n_clusters <= 6:
            figsize = (1.75, 14)
            fig, ax = plt.subplots(6, 1, figsize=figsize, sharex=True, sharey=True)
        elif n_clusters <= 12:
            figsize = (4, 14)
            fig, ax = plt.subplots(6, 2, figsize=figsize, sharex=True, sharey=True)
        elif n_clusters <= 18:
            figsize = (5.5, 14)
            fig, ax = plt.subplots(6, 3, figsize=figsize, sharex=True, sharey=True)
        else:
            print('too many clusters to plot, adapt plotting code if needed')
        ax = ax.ravel()

        # loop through clusters in sorted order
        for i, cluster_id in enumerate(clusters):
            # generate and label plot for original cluster_id
            #             original_cluster_id = cluster_meta[(cluster_meta.cre_line==cre_line)&(cluster_meta[sort_col]==cluster_id)].original_cluster_id.unique()[0]
            ax[i] = plot_dropout_heatmap(cluster_meta, feature_matrix, cre_line, cluster_id, ax=ax[i])

        plt.suptitle(get_cell_type_for_cre_line(cluster_meta, cre_line), x=0.51, y=.95, fontsize=16)
        plt.subplots_adjust(hspace=0.6, wspace=0.4)
        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder,
                              'cluster_heatmaps_' + sort_type + '_sort_' + cre_line.split('-')[0])


def standardize_axes_fontsize(ax):
    """
    Sets axis titles and x, y labels to font 12
    Sets x and y ticklabels to 10
    :param ax:
    :return: ax
    """
    # set font sizes for ticks and labels
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)
    return ax


def plot_pct_rel_to_chance_for_cluster(cre_counts, cluster_id, ax=None):
    """
    Plots the % of cells in each area and depth relative to chance for a given cluster
    including whether the actual cell count is significantly greater than chance
    y values are depths, hue is for areas

    cre_counts: counts of cells and chance levels per cluster limited to a given cre line
                must include columns: targeted_structure, layer, relative to random, sig_greater
    cluster_id: cluster id to plot
    """
    order = list(np.sort(cre_counts.layer.unique())[::-1])
    hue_order = list(np.sort(cre_counts.targeted_structure.unique())[::-1])

    if ax is None:
        fig, ax = plt.subplots()

    data = cre_counts[cre_counts.cluster_id == cluster_id]
    ax = sns.barplot(data=data, y='layer', x='relative_to_random', order=order, orient='h',
                     hue='targeted_structure', hue_order=hue_order, palette='Greys', ax=ax)
    ax.set_ylim(-0.5, 1.5)

    sig_data = data[data.sig_greater == True]
    for row in range(len(sig_data)):
        row_data = sig_data.iloc[row]
        layer_ind = order.index(row_data.layer)
        area_ind = hue_order.index(row_data.targeted_structure)
        if area_ind == 0:
            y_pos = layer_ind - 0.25
        elif area_ind == 1:
            y_pos = layer_ind + 0.25
        x_pos = row_data.relative_to_random * 1.2
        ax.plot(x_pos, y_pos, marker='*', color='k')

    ax.legend(fontsize='small', title_fontsize='small')
    ax.set_xlabel('% rel. to chance', fontsize=14)
    ax.set_yticklabels(order, fontsize=12)
    ax.set_ylabel('')
    # flip axes so upper layer is on top
    ax.set_ylim(ax.get_ylim()[::-1])
    sns.despine(ax=ax, left=True, bottom=True)
    # make fonts small
    ax = standardize_axes_fontsize(ax)
    return ax


def plot_proportion_cells_for_cluster(cre_proportion, cluster_id, ci_df=None, ax=None):
    """
    Plots the proportion of cells in each area and depth for a given cluster
    x values are areas & depths, y values are proportion cells

    proportion: proportion of cells relative to cluster average per cluster limited to a given cre line
                must include columns: location, proportion_cells, cluster_id, sig_greater
    cluster_id: cluster id to plot
    ci_df: dataframe of confidence intervals per location per cluster
    """

    if 'location' not in cre_proportion.keys():
        print('location column must be in cre_proportion df provided to plot_proportion_cells_per_cluster function')

    locations = list(np.sort(cre_proportion.location.unique()))  # make sure to sort so it matches the data and ci order
    colormap = sns.color_palette("Paired", len(locations))

    if ci_df is None:
        ci = np.zeros(len(locations))
    else:
        # make sure to sort by location so it matches the data
        ci = ci_df[ci_df['cluster_id'] == cluster_id].sort_values('location')['CI'].values

    if ax is None:
        fig, ax = plt.subplots()

    data = cre_proportion[cre_proportion.cluster_id == cluster_id]
    data = data.sort_values('location')  # make sure to sort by location so it matches the ci
    ax = sns.barplot(data=data, x='location', y='proportion_cells', order=locations, yerr=ci, orient='v', palette=colormap, ax=ax)

    ax.set_ylabel('proportion cells\n rel. cluster avg.', fontsize=14)
    ax.set_xticklabels(locations, fontsize=14, rotation=90)
    ax.set_xlabel('')
    # flip axes so V1 and/or upper layer is first
    ax.set_xlim(ax.get_xlim()[::-1])

    # ax.set_xticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', length=0)

    # sns.despine(ax=ax, left=True, bottom=True)
    # make fonts small
    ax = standardize_axes_fontsize(ax)
    return ax


def plot_fraction_cells_per_area_depth(cre_fraction, cluster_id, ax=None):
    """
    Plots the fraction of cells in each area and depth relative to
    the total number of cells in that area & depth for a given cre line

    cre_counts: counts of cells and chance levels per cluster limited to a given cre line
                must include columns: targeted_structure, layer, relative to random, sig_greater
    cluster_id: cluster id to plot
    """
    order = np.sort(cre_fraction.layer.unique())[::-1]
    hue_order = np.sort(cre_fraction.targeted_structure.unique())[::-1]

    if ax is None:
        fig, ax = plt.subplots()

    data = cre_fraction[cre_fraction.cluster_id == cluster_id]
    ax = sns.barplot(data=data, y='layer', x='fraction_cells_per_area_depth', order=order, orient='h',
                     hue='targeted_structure', hue_order=hue_order, palette='Greys', ax=ax)
    ax.set_ylim(-0.5, 1.5)

    ax.legend(fontsize='small', title_fontsize='small')
    ax.set_xlabel('fraction cells', fontsize=14)
    ax.set_yticklabels(order, fontsize=12)
    ax.set_ylabel('')
    # flip axes so upper layer is on top
    ax.set_ylim(ax.get_ylim()[::-1])
    sns.despine(ax=ax, left=True, bottom=True)

    # set font sizes for ticks and labels
    ax = standardize_axes_fontsize(ax)
    return ax


def plot_population_average_response_for_cluster(cluster_mdf, cre_line, cluster_id, change=False, omitted=True,
                                                 ax=None):
    """
    Plots the population average response across experience levels for a give cluster from a given cre line

    cluster_mdf: multi_session_mean_df loaded using get_multi_session_df_for_conditions, with cluster_ids added as a column
    cre_line: string for cre-line to use
    cluster_id: id for cluster to plot
    """
    hue_column = 'experience_level'
    hue_conditions = np.sort(cluster_mdf[hue_column].unique())
    timestamps = cluster_mdf['trace_timestamps'][0]
    xlim_seconds = [-1, 1.5]
    colors = utils.get_experience_level_colors()

    if ax is None:
        fig, ax = plt.subplots()
    for c, hue in enumerate(hue_conditions):
        traces = cluster_mdf[(cluster_mdf.cre_line == cre_line) & (cluster_mdf.cluster_id == cluster_id) & (
            cluster_mdf[hue_column] == hue)].mean_trace.values
        ax = utils.plot_mean_trace(np.asarray(traces), timestamps, ylabel='response',
                                   legend_label=hue, color=colors[c], interval_sec=1,
                                   plot_sem=False, xlim_seconds=xlim_seconds, ax=ax)
    ax = utils.plot_flashes_on_trace(ax, timestamps, change=change, omitted=omitted)
    ax = standardize_axes_fontsize(ax)
    return ax


def plot_clusters_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df, cre_line,
                               sort_order=None, save_dir=None, folder=None, suffix=''):
    """
    For each cluster in a given cre_line, plots dropout heatmaps, fraction cells per area/depth relative to chance,
    fraction cells per cluster per area/depth, and population average omission response.
    Will sort clusters according to sort_order dict if provided

    :param cluster_meta: table of metadata for each cell_specimen_id (rows), including cluster_id for each cell_specimen_id
    :param feature_matrix: dropout scores for matched cells with experience levels x features as cols, cells as rows
    :param multi_session_df: table of cell responses for a set of conditions, from loading.get_multi_session_df_for_conditions()
    :param cre_line: cre line to plot for
    :param sort_order: dictionary with cre_lines as keys, sorted cluster_ids as values
    :param save_dir: directory to save plot to
    :param folder: folder within save_dir to save plot to
    :param suffix: string to be appended to end of filename
    :return:
    """
    # add cluster_id to multi_session_df
    cluster_mdf = multi_session_df.merge(cluster_meta[['cluster_id']],
                                         on='cell_specimen_id', how='inner')
    cluster_ids = np.sort(cluster_meta[cluster_meta.cre_line == cre_line].cluster_id.unique())
    # if order to sort clusters is provided, use it
    if sort_order:
        cluster_ids = sort_order[cre_line]
    n_clusters = len(cluster_ids)

    n_rows = 2
    figsize = (n_clusters * 2.5, n_rows * 2.5)
    fig, ax = plt.subplots(n_rows, n_clusters, figsize=figsize, sharex='row', sharey='row',
                           gridspec_kw={'height_ratios': [1, 0.75]})
    ax = ax.ravel()
    for i, cluster_id in enumerate(cluster_ids):
        # plot mean dropout heatmap for this cluster
        ax[i] = plot_dropout_heatmap(cluster_meta, feature_matrix, cre_line, cluster_id, ax=ax[i])

        # population average for this cluster
        ax[i + (n_clusters * 1)] = plot_population_average_response_for_cluster(cluster_mdf, cre_line, cluster_id,
                                                                                ax=ax[i + (n_clusters * 1)])
        ax[i + (n_clusters * 1)].set_xlabel('time (s)')
        if i > 0:
            ax[i + (n_clusters * 1)].set_ylabel('')

    fig.subplots_adjust(hspace=1.2, wspace=0.6)
    fig.suptitle(get_cell_type_for_cre_line(cluster_meta, cre_line), x=0.52, y=0.98, fontsize=16)
    # fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'clusters_pop_avg_rows_' + cre_line.split('-')[0] + suffix)


def plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df, cre_line, columns_to_groupby=['targeted_structure', 'layer'],
                                     sort_order=None, save_dir=None, folder=None, suffix='', alpha=0.05):
    """
    For each cluster in a given cre_line, plots dropout heatmaps, fraction cells per location (area and/or depth) relative to the cluster average,
    fraction cells per cluster per area/depth, and population average omission response.
    Will sort clusters according to sort_order dict if provided
    Proportion cells per location is computed by glm_clustering code

    :param cluster_meta: table of metadata for each cell_specimen_id (rows), including cluster_id for each cell
    :param feature_matrix: dropout scores for matched cells with experience levels x features as cols, cells as rows
    :param multi_session_df: table of cell responses for a set of conditions, from loading.get_multi_session_df_for_conditions()
    :param cre_line: cre line to plot for
    :param columns_to_groupby: columns in cluster_meta to use when computing proportion cells per location;
                                location is defined as the concatenation of the groups in columns_to_groupby
    :param sort_order: dictionary with cre_lines as keys, sorted cluster_ids as values
    :param save_dir: directory to save plot to
    :param folder: folder within save_dir to save plot to
    :param suffix: string to be appended to end of filename
    :param alpha: int for confidence intervals, default = 0.05, set to None if would like to exclude errorbars
    :return:
    """
    # add cluster_id to multi_session_df
    cluster_mdf = multi_session_df.merge(cluster_meta[['cluster_id']], on='cell_specimen_id', how='inner')
    cluster_ids = np.sort(cluster_meta[cluster_meta.cre_line == cre_line].cluster_id.unique())
    # if order to sort clusters is provided, use it
    if sort_order:
        cluster_ids = sort_order[cre_line]
    n_clusters = len(cluster_ids)

    # if using a numeric value column, like 'binned_depth', convert to a string so downstream code doesnt break
    if 'binned_depth' in columns_to_groupby:
        cluster_meta['binned_depth'] = [str(depth) for depth in cluster_meta.binned_depth.values]

    # location column is a categorical variable (string) that can be a combination of area and depth or just area or depth (or whatever)
    cluster_meta = processing.add_location_column(cluster_meta, columns_to_groupby)

    # limit to this cre line
    cluster_meta_cre = cluster_meta[cluster_meta.cre_line == cre_line].copy()

    # compute CI for this cre line, cluster id and location
    if alpha is not None:
        ci_df = processing.get_CI_for_clusters(cluster_meta, columns_to_groupby, alpha=alpha, type_of_CI='mpc', cre='all')
        ci_df = ci_df[ci_df.cre_line == cre_line]
    else:
        ci_df = None
    # compute proportion cells per location for each cluster and stats using glm_clustering code
    # computes proportions for all cre lines so need to limit to this cre afterwards
    proportions, stats_table = processing.get_proportion_cells_rel_cluster_average(cluster_meta, cluster_meta.cre_line.unique(),
                                                                                   columns_to_groupby)
    cre_proportions = proportions[proportions.cre_line == cre_line]
    cre_stats = stats_table[stats_table.cre_line == cre_line]

    # max confidence interval to know where to plot the significance bar
    y_max = cre_proportions['proportion_cells'].max()
    dh = y_max * 0.2  # extra y space for plotting significance star
    ci_error = ci_df['CI'].max()  # use this max to plot vertical line for stats at the same height for all clusters
    bary = np.array([y_max, y_max]) + ci_error  # height of the vertical line for statistics comparison

    # plot
    n_rows = 3  # 4 if including proportion plots
    figsize = (n_clusters * 2.5, n_rows * 2.5)
    fig, ax = plt.subplots(n_rows, n_clusters, figsize=figsize, sharex='row', sharey='row',
                           gridspec_kw={'height_ratios': [1, 0.75, 1.5]})
    ax = ax.ravel()
    for i, cluster_id in enumerate(cluster_ids):
        # plot mean dropout heatmap for this cluster
        ax[i] = plot_dropout_heatmap(cluster_meta_cre, feature_matrix, cre_line, cluster_id, ax=ax[i])

        # plot population averages per cluster
        ax[i + (n_clusters * 1)] = plot_population_average_response_for_cluster(cluster_mdf, cre_line, cluster_id,
                                                                                ax=ax[i + (n_clusters * 1)])
        ax[i + (n_clusters * 1)].set_xlabel('time (s)')
        if i > 0:
            ax[i + (n_clusters * 1)].set_ylabel('')

        # plot proportion of cells per cluster relative to cluster average, with stats
        # get confidence intervals for this cluster id
        ax[i + (n_clusters * 2)] = plot_proportion_cells_for_cluster(cre_proportions, cluster_id, ci_df=ci_df, ax=ax[i + (n_clusters * 2)])

        # plot significance with bh corrected chi-square test
        this_s = cre_stats.loc[cluster_id]
        if this_s['bh_significant'] == True:
            barx = [int(len(cre_proportions.location.unique())) - 1, 0]
            mid = np.mean(barx) + 0.25
            ax[i + (n_clusters * 2)].plot(barx, bary, color='k')
            if this_s['imq'] < 0.0005:
                text = '***'
            elif this_s['imq'] < 0.005:
                text = '**'
            elif this_s['imq'] < 0.05:
                text = '*'
            ax[i + (n_clusters * 2)].text(mid, bary[0] + dh, text, color='k', fontsize=20)

        if i > 0:
            ax[i + (n_clusters * 2)].set_ylabel('')

    # legend for leftmost plot for pct chance
    # ax[(0 + n_clusters *2)].legend(loc='lower right', fontsize='x-small')

    fig.subplots_adjust(hspace=1.2, wspace=0.6)
    fig.suptitle(get_cell_type_for_cre_line(cluster_meta, cre_line), x=0.51, y=1.01, fontsize=16)
    # fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'clusters_stats_rows_' + cre_line.split('-')[0] + suffix)


def plot_clusters_stats_pop_avg_cols(cluster_meta, feature_matrix, multi_session_df, cell_count_stats, fraction_cells, cre_line,
                                     sort_order=None, save_dir=None, folder=None, suffix=''):
    """
    For each cluster in a given cre_line, plots dropout heatmaps, fraction cells per area/depth relative to chance,
    fraction cells per cluster per area/depth, and population average omission response.
    Will sort clusters according to sort_order dict if provided

    :param cluster_meta: table of metadata for each cell_specimen_id (rows), including cluster_id for each cell
    :param feature_matrix: dropout scores for matched cells with experience levels x features as cols, cells as rows
    :param multi_session_df: table of cell responses for a set of conditions, from loading.get_multi_session_df_for_conditions()
    :param cell_count_stats: table with fraction of cells relative to chance for each area/depth per cluster
    :param fraction_cells: table with fraction of cells in each cluster for each area/depth
    :param cre_line: cre line to plot for
    :param sort_order: dictionary with cre_lines as keys, sorted cluster_ids as values
    :param save_dir: directory to save plot to
    :param folder: folder within save_dir to save plot to
    :param suffix: string to be appended to end of filename
    :return:
    """
    # add cluster_id to multi_session_df
    cluster_mdf = multi_session_df.merge(cluster_meta[['cluster_id']],
                                         on='cell_specimen_id', how='inner')
    # info for this cre line
    # cre_meta = cluster_meta[cluster_meta.cre_line==cre_line]
    # cre_cell_specimen_ids = cluster_meta[cluster_meta.cre_line==cre_line].index.values
    cluster_ids = np.sort(cluster_meta[cluster_meta.cre_line == cre_line].cluster_id.unique())
    # cell counts and fraction for this cre line
    cre_counts = cell_count_stats[cell_count_stats.cre_line == cre_line]
    cre_fraction = fraction_cells[fraction_cells.cre_line == cre_line]

    n_cols = 4
    figsize = (n_cols * 2.75, len(cluster_ids) * 2)
    fig, ax = plt.subplots(len(cluster_ids), n_cols, figsize=figsize, sharex='col',
                           gridspec_kw={'width_ratios': [1, 1.5, 1.5, 1.5]})
    ax = ax.ravel()
    i = 0
    for c, cluster_id in enumerate(cluster_ids):
        # plot mean dropout heatmap for this cluster
        ax[i] = plot_dropout_heatmap(cluster_meta, feature_matrix, cre_line, cluster_id, ax=ax[i])
        i += 1

        ax[i] = plot_population_average_response_for_cluster(cluster_mdf, cre_line, cluster_id, ax=ax[i])
        ax[i].set_xlabel('')
        i += 1

        ax[i] = plot_pct_rel_to_chance_for_cluster(cre_counts, cluster_id, ax=ax[i])
        ax[i].get_legend().remove()
        ax[i].set_xlabel('')
        ax[i].set_xlim(-1, 1)
        i += 1

        ax[i] = plot_fraction_cells_per_area_depth(cre_fraction, cluster_id, ax=ax[i])
        ax[i].get_legend().remove()
        ax[i].set_xlabel('')
        ax[i].set_xlim(0, 0.45)
        i += 1

    # xlabels for bottom plots
    ax[i - 2].set_xlabel('fraction cells', fontsize=14)
    ax[i - 1].set_xlabel('time after omission (s)', fontsize=12)

    ax[i - 3].set_xlabel('% rel. to chance', fontsize=14)
    # legend for top right plot
    ax[2].legend(loc='lower right', fontsize='xx-small')

    fig.subplots_adjust(hspace=0.55, wspace=0.6)
    fig.suptitle(get_cell_type_for_cre_line(cluster_meta, cre_line), x=0.5, y=0.92, fontsize=16)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'clusters_cols_' + cre_line.split('-')[0] + suffix)


def plot_gap_statistic(gap_statistic, cre_lines, save_dir=None, folder=None):
    for cre_line in cre_lines:
        figsize = (10, 3)
        suffix = cre_line
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        x = len(gap_statistic[cre_line][0])
        ax[0].plot(np.arange(1, x + 1), gap_statistic[cre_line][0], 'o-')
        ax[0].set_ylabel('gap value')
        ax[0].grid()
        ax[1].plot(np.arange(1, x + 1), gap_statistic[cre_line][1], 'o-')
        ax[1].plot(np.arange(1, x + 1), gap_statistic[cre_line][2], 'o-')
        ax[1].legend(['reference inertia', 'ondata intertia'])
        ax[1].grid()
        plt.suptitle(cre_line)

        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder, 'eigengap' + suffix)


def plot_eigengap_values(eigenvalues_cre, cre_lines, save_dir=None, folder=None):
    for cre_line in cre_lines:
        eigenvalues = eigenvalues_cre[cre_line]
        suffix = cre_line
        figsize = (7, 7)
        fig, ax = plt.subplots(2, 1, figsize=figsize, sharex='all')

        ax[0].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, '-o')
        ax[0].grid()
        ax[0].set_title(cre_line)
        ax[0].set_ylabel('eigen value (sorted)')
        ax[0].set_xlabel('eigen number')
        ax[0].set_xlim([2, 20])

        ax[1].plot(np.arange(2, len(eigenvalues) + 1), np.diff(eigenvalues), '-o')
        ax[1].set_ylabel('gap value (difference)')
        ax[1].set_xlabel('eigen number')
        ax[1].set_xlim([0, 20])
        ax[1].set_ylim([0, 0.20])
        ax[1].grid()
        plt.tight_layout()

        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder, 'eigengap' + suffix)
