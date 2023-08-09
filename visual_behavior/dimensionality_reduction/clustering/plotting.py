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


def get_abbreviated_experience_levels(experience_levels):
    """
    converts experience level names (ex: 'Novel >1') into short hand versions (ex: 'N>1')
    abbreviated names are returned in the same order as provided in experience_levels
    """
    exp_level_abbreviations = [exp_level.split(' ')[0][0] if len(exp_level.split(' ')) == 1 else exp_level.split(' ')[0][0] + exp_level.split(' ')[1][:2] for exp_level in experience_levels]
    # exp_level_abbreviations = ['F', 'N', 'N+']
    return exp_level_abbreviations


def get_abbreviated_features(features):
    """
    converts GLM feature names (ex: 'Behavioral') into first letter capitalized short hand versions (ex: 'B')
    'all-images' gets converted to 'I' for Images
    abbreviated names are returned in the same order as provided in features
    """
    # get first letter of each feature name and capitalize it
    feature_abbreviations = [feature[0].upper() for feature in features]
    # change the "A" for "all-images" to "I" for "images" instead
    if 'A' in feature_abbreviations:
        feature_abbreviations[feature_abbreviations.index('A')] = 'I'
    return feature_abbreviations


def get_abbreviated_location(location, abbrev_layer=True):
    """
    converts formal area names (ex: 'VISp') into field standards (ex: 'V1')
    and depth strings (ex: 'upper') into the capitalized first letter (ex: 'U')
    only works when location is targeted_structure or depth or both
    """
    if len(location.split('_')) == 2:
        if 'VISl' in location:
            area = 'LM'
        elif 'VISp' in location:
            area = 'V1'
        else:
            area = location.split('_')[0]
        if abbrev_layer:
            if 'lower' in location:
                depth = 'L'
            elif 'upper' in location:
                depth = 'U'
            else:
                depth = location.split('_')[1]
        else:
            depth = location.split('_')[1]
        loc_abbrev = area + '-' + depth
    elif len(location.split('_')) == 1:
        if 'VISl' in location:
            loc = 'LM'
        elif 'VISp' in location:
            loc = 'V1'
        else:
            loc = location
        if abbrev_layer:
            if 'lower' in location:
                loc = 'L'
            elif 'upper' in location:
                loc = 'U'
            else:
                loc = location
        else:
            loc = location
        loc_abbrev = loc

    return loc_abbrev


def plot_feature_matrix_for_cre_lines(feature_matrix, cell_metadata, use_abbreviated_labels=False, save_dir=None, folder=None):
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
        for x in [3, 6, 9]:
            ax[i].axvline(x=x, ymin=0, ymax=data.shape[0], color='gray', linestyle='--', linewidth=1)
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cell_metadata))
        ax[i].set_ylabel('cells')
        ax[i].set_ylim(0, data.shape[0])
        ax[i].set_yticks([0, data.shape[0]])
        ax[i].set_yticklabels((0, data.shape[0]), fontsize=14)
        ax[i].set_ylim(ax[i].get_ylim()[::-1])  # flip y axes so larger clusters are on top
        ax[i].set_xlabel('')
        ax[i].set_xlim(0, data.shape[1])
        ax[i].set_xticks(np.arange(0, data.shape[1]) + 0.5)
        if use_abbreviated_labels:
            xticklabels = [get_abbreviated_experience_levels([key[1]])[0] + ' -  ' + get_abbreviated_features([key[0]])[0].upper() for key in list(data.keys())]
            ax[i].set_xticklabels(xticklabels, rotation=90, fontsize=14)
        else:
            ax[i].set_xticklabels([key[1] + ' -  ' + key[0] for key in list(data.keys())], rotation=90, fontsize=14)

    fig.subplots_adjust(wspace=0.7)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'feature_matrix_unsorted')


def plot_feature_matrix_sorted(feature_matrix, cluster_meta, sort_col='cluster_id', use_abbreviated_labels=False,
                               resort_by_size=False, save_dir=None, folder=None, suffix=''):
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
        cluster_meta_cre = cluster_meta[cluster_meta.cre_line == cre_line]
        # get cell ids for this cre line in sorted order
        if resort_by_size:
            cluster_size_order = cluster_meta_cre['cluster_id'].value_counts().index.values
            cluster_meta_cre['new_cluster_id'] = [np.where(cluster_size_order == label)[0][0] for label in
                                                  cluster_meta_cre.cluster_id.values]
            sort_col = 'new_cluster_id'
        sorted_cluster_meta_cre = cluster_meta_cre.sort_values(by=sort_col)
        cell_order = sorted_cluster_meta_cre.index.values
        label_values = sorted_cluster_meta_cre[sort_col].values

        # get data from feature matrix for this set of cells
        data = feature_matrix.loc[cell_order]
        ax[i] = sns.heatmap(data.values, cmap=cmap, ax=ax[i], vmin=vmin, vmax=1,
                            robust=True, cbar_kws={"drawedges": False, "shrink": 0.7, "label": 'coding score'})

        for x in [3, 6, 9]:
            ax[i].axvline(x=x, ymin=0, ymax=data.shape[0], color='gray', linestyle='--', linewidth=1)
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))
        ax[i].set_ylabel('cells')
        ax[i].set_ylim(0, data.shape[0])
        ax[i].set_yticks([0, data.shape[0]])
        ax[i].set_yticklabels((0, data.shape[0]), fontsize=14)
        ax[i].set_ylim(ax[i].get_ylim()[::-1])  # flip y axes so larger clusters are on top
        ax[i].set_xlabel('')
        ax[i].set_xlim(0, data.shape[1])
        ax[i].set_xticks(np.arange(0, data.shape[1]) + 0.5)
        if use_abbreviated_labels:
            xticklabels = [get_abbreviated_experience_levels([key[1]])[0] + ' -  ' + get_abbreviated_features([key[0]])[0].upper() for key in list(data.keys())]
            ax[i].set_xticklabels(xticklabels, rotation=90, fontsize=14)
        else:
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
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cell_metadata))
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
    ax[1].set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))

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
    ax.set_title(processing.get_cre_line_map(cre_line))
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
        utils.save_figure(fig, figsize, save_dir, folder, filename, formats=['.png', '.pdf'])  # saving to PDF is super slow)
    return ax


def plot_affinity_matrix(feature_matrix, cluster_meta, cre_line, n_clusters_cre, save_dir=None, folder=None, ax=None):
    """
    plot affinity matrix from spectral clustering of dropout scores
    """

    from sklearn.cluster import SpectralClustering

    sc = SpectralClustering()

    n_clusters = n_clusters_cre[cre_line]
    cell_specimen_ids = cluster_meta[cluster_meta.cre_line == cre_line].index.unique()
    X = feature_matrix.loc[cell_specimen_ids].copy()

    sc.n_clusters = n_clusters
    x = sc.fit(X.values)

    sort_order = np.argsort(x.labels_)
    affinity = pd.DataFrame(x.affinity_matrix_)
    affinity = affinity.loc[sort_order]
    affinity = affinity[sort_order]
    figsize = (5, 5)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(affinity, square=True, cmap='Greys', vmin=0, vmax=1, cbar_kws={'label': 'similarity', 'shrink': 0.7}, ax=ax)
    ax.set_xticks((0, len(affinity)))
    ax.set_xticklabels([])
    ax.set_yticks((0, len(affinity)))
    ax.set_yticklabels((0, len(affinity)))
    ax.set_ylabel('cells')
    ax.set_title(processing.get_cell_type_for_cre_line(cre_line, cluster_meta) + '\naffinity matrix')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'affinity_matrix_' + cre_line)
    return ax


def plot_umap_for_clusters(cluster_meta, feature_matrix, label_col='cluster_id', save_dir=None, folder=None):
    """
    plots umap for each cre line, colorized by metadata column provided as label_col

    label_col: column in cluster_meta to colorize points by
    """
#    import umap
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
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))
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
    ax.set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))
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


def plot_dropout_heatmap(cluster_meta, feature_matrix, cre_line, cluster_id, cbar=False,
                         abbreviate_features=False, abbreviate_experience=False,
                         cluster_size_in_title=True, small_fontsize=False, ax=None):
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
    features = processing.get_features_for_clustering()
    mean_dropout_df = mean_dropout_df.loc[features]  # order regressors in a specific order

    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.heatmap(mean_dropout_df, cmap=cmap, vmin=vmin, vmax=1, ax=ax, cbar=cbar, cbar_kws={'label': 'coding score'})
    if cluster_size_in_title:
        # fraction is number of cells in this cluster vs all cells in this cre line
        fraction_cre = len(this_cluster_csids) / float(len(cre_csids))
        fraction = np.round(fraction_cre * 100, 1)
        # set title and labels
        ax.set_title('cluster ' + str(cluster_id) + '\n' + str(fraction) + '%, n=' + str(len(this_cluster_csids)))
    else:
        # title is cre line abbreviation and cluster #
        cell_type = processing.get_cell_type_for_cre_line(cre_line, cluster_meta)
        cell_type_abbreviation = cell_type[:3]
        ax.set_title(cell_type_abbreviation + ' cluster ' + str(cluster_id))
    ax.set_yticks(np.arange(0.5, len(mean_dropout_df.index.values) + 0.5))
    if abbreviate_features:
        # set yticks to abbreviated feature labels
        feature_abbreviations = get_abbreviated_features(mean_dropout_df.index.values)
        ax.set_yticklabels(feature_abbreviations, rotation=0)
    else:
        ax.set_yticklabels(mean_dropout_df.index.values, rotation=0, fontsize=14)
    if abbreviate_experience:
        # set xticks to abbreviated experience level labels
        exp_level_abbreviations = get_abbreviated_experience_levels(mean_dropout_df.columns.values)
        ax.set_xticklabels(exp_level_abbreviations, rotation=90)
    else:
        ax.set_xticklabels(mean_dropout_df.columns.values, rotation=90, fontsize=14)
    ax.set_ylim(0, mean_dropout_df.shape[0])
    # invert y axis so images is always on top
    ax.invert_yaxis()
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

        plt.suptitle(get_cell_type_for_cre_line(cre_line, cluster_meta), x=0.51, y=.95, fontsize=16)
        plt.subplots_adjust(hspace=0.6, wspace=0.4)
        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder,
                              'cluster_heatmaps_' + sort_col + '_' + cre_line.split('-')[0])


def plot_average_dropout_heatmap_for_cre_lines(dropouts, cluster_meta, save_dir=None, folder=None, suffix='', suptitle=None):
    """
    plot average dropout score for each cre line
    """
    dropouts_meta = dropouts.merge(cluster_meta[['cre_line', 'binned_depth', 'targeted_structure']], on='cell_specimen_id')

    figsize = (15, 2)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=False)
    ax = ax.ravel()
    for i, cre_line in enumerate(get_cre_lines(cluster_meta)):
        mean_dropouts = dropouts_meta[dropouts_meta.cre_line == cre_line].groupby('experience_level').mean()[processing.get_features_for_clustering()]
        ax[i] = sns.heatmap(mean_dropouts.T, cmap='Blues', vmin=0, vmax=0.5, ax=ax[i], cbar_kws={'shrink': 0.7, 'label': 'coding score'})
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))
        ax[i].set_ylim(0, 4)
        ax[i].set_ylim(0, 4)
        ax[i].set_xlabel('')
        ax[i].invert_yaxis()
    plt.subplots_adjust(wspace=1.2)
    if suptitle is not None:
        plt.suptitle(suptitle, x=0.5, y=1.3)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'average_dropout_heatmaps' + suffix)


def plot_std_dropout_heatmap_for_cre_lines(dropouts, cluster_meta, save_dir=None, folder=None):
    """
    plot average dropout score for each cre line
    """
    dropouts_meta = dropouts.merge(cluster_meta[['cre_line', 'binned_depth', 'targeted_structure']], on='cell_specimen_id')

    figsize = (15, 2)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=False)
    ax = ax.ravel()
    for i, cre_line in enumerate(get_cre_lines(cluster_meta)):
        mean_dropouts = dropouts_meta[dropouts_meta.cre_line == cre_line].groupby('experience_level').var()[processing.get_features_for_clustering()]
        ax[i] = sns.heatmap(mean_dropouts.T, cmap='Purples', vmin=0, vmax=0.25, ax=ax[i], cbar_kws={'shrink': 0.7, 'label': 'standard\ndeviation'})
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))
        ax[i].set_ylim(0, 4)
        ax[i].set_xlabel('')
        ax[i].invert_yaxis()
    plt.subplots_adjust(wspace=1.2)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'standard_deviation_dropout_heatmaps')


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
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    if palette is None:
        palette = [(1.0, 0.596078431372549, 0.5882352941176471),
                   (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
                   (0.7725490196078432, 0.6901960784313725, 0.8352941176470589)]
    markersize = [14, 10, 10]
    linewidth = [4, 2, 2]
    for i, cre_line in enumerate(labels_cre.keys()):
        labels = labels_cre[cre_line]
        (unique, counts) = np.unique(labels, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        frequencies = sorted_counts / sum(counts) * 100
        cumulative_sum = [0]
        for freq in frequencies:
            cumulative_sum.append(cumulative_sum[-1] + freq)

        ax.plot(range(0, len(cumulative_sum)), cumulative_sum, 'o-', color=palette[i],
                linewidth=linewidth[i], markersize=markersize[i])
        ax.hlines(y=80, xmin=-0.5, xmax=11, linestyle='--', color='Grey')
        ax.set_xlim([-0.5, 11])
        ax.set_xticks(np.arange(0, len(cumulative_sum)))
        ax.set_xlabel('Cluster number')
        ax.set_ylabel('Cumulative % of all cells')
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
            fig.suptitle(get_cell_type_for_cre_line(cre_line, cluster_meta) + ' cluster ' + str(cluster_id), x=0.5, y=1.01)
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
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))
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
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))
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

        plt.suptitle(get_cell_type_for_cre_line(cre_line, cluster_meta), x=0.51, y=.95, fontsize=16)
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


def plot_proportion_cells_for_cluster(cre_proportion, cluster_id, ci_df=None, ax=None, orient='v', small_fontsize=True):
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
    if orient == 'v':
        ax = sns.barplot(data=data, x='location', y='proportion_cells', order=locations, yerr=ci, orient='v', palette=colormap, ax=ax)
        ax.set_ylabel('proportion cells\n rel. cluster avg.', fontsize=14)
        ax.set_xticklabels(locations, rotation=90)
        ax.set_xlabel('')
        # flip axes so V1 and/or upper layer is first
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.tick_params(axis='x', length=0)
    elif orient == 'h':
        ax = sns.barplot(data=data, y='location', x='proportion_cells', order=locations, yerr=ci, orient='h', palette=colormap, ax=ax)
        ax.set_xlabel('proportion cells\n rel. cluster avg.', fontsize=14)
        ax.set_yticklabels(locations, rotation=0)
        ax.set_ylabel('')
        # flip axes so V1 and/or upper layer is first
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.tick_params(axis='y', length=0)

    # ax.set_xticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # sns.despine(ax=ax, left=True, bottom=True)
    # make fonts small
    if small_fontsize:
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
                                                 small_fontsize=True, alpha=0.1, ax=None):
    """
    Plots the population average response across experience levels for a give cluster from a given cre line

    cluster_mdf: multi_session_mean_df loaded using get_multi_session_df_for_conditions, with cluster_ids added as a column
    cre_line: string for cre-line to use
    cluster_id: id for cluster to plot
    """
    hue_column = 'experience_level'
    hue_conditions = np.sort(cluster_mdf[hue_column].unique())
    timestamps = cluster_mdf['trace_timestamps'][0]
    if change:
        xlim_seconds = [-1, 0.75]
    elif omitted:
        xlim_seconds = [-1, 1.5]
    else:
        xlim_seconds = [0.5, 0.75]
    colors = utils.get_experience_level_colors()

    if ax is None:
        fig, ax = plt.subplots()
    for c, hue in enumerate(hue_conditions):
        traces = cluster_mdf[(cluster_mdf.cre_line == cre_line) & (cluster_mdf.cluster_id == cluster_id) & (
            cluster_mdf[hue_column] == hue)].mean_trace.values
        ax = utils.plot_mean_trace(np.asarray(traces), timestamps, ylabel='response',
                                   legend_label=hue, color=colors[c], interval_sec=1,
                                   plot_sem=False, xlim_seconds=xlim_seconds, ax=ax)
    ax = utils.plot_flashes_on_trace(ax, timestamps, change=change, omitted=omitted, alpha=alpha)
    ax.set_xlabel('time (s)')
    if small_fontsize:
        ax = standardize_axes_fontsize(ax)
    return ax


def plot_clusters_row(cluster_meta, feature_matrix, cre_line,
                      sort_order=None, rename_clusters=False, save_dir=None, folder=None, suffix='', abbreviate_experience=True, formats=['.png', '.pdf']):
    """
    For each cluster in a given cre_line, plots dropout heatmaps, fraction cells per area/depth relative to chance,
    fraction cells per cluster per area/depth, and population average omission response.
    Will sort clusters according to sort_order dict if provided

    :param cluster_meta: table of metadata for each cell_specimen_id (rows), including cluster_id for each cell_specimen_id
    :param feature_matrix: dropout scores for matched cells with experience levels x features as cols, cells as rows
    :param cre_line: cre line to plot for
    :param sort_order: dictionary with cre_lines as keys, sorted cluster_ids as values
    :param rename_clusters: bool, if clusters ids are not sorted by their size, use sort order to rename cluster ids
    :param save_dir: directory to save plot to
    :param folder: folder within save_dir to save plot to
    :param suffix: string to be appended to end of filename
    :return:
    """

    cluster_ids = np.sort(cluster_meta[cluster_meta.cre_line == cre_line].cluster_id.unique())
    # if order to sort clusters is provided, use it
    if sort_order:
        if rename_clusters is False:
            cluster_ids = sort_order[cre_line]
        else:
            cluster_remap = {}
            for new_id, old_id in enumerate(sort_order[cre_line]):
                cluster_remap[old_id] = new_id + 1
            cluster_meta['cluster_id'].replace(cluster_remap, inplace=True)

    n_clusters = len(cluster_ids)

    n_rows = 1
    figsize = (n_clusters * 3.5, n_rows * 2)
    fig, ax = plt.subplots(n_rows, n_clusters, figsize=figsize, sharex='row', sharey='row')
    # gridspec_kw={'height_ratios': [1, 0.75]})
    ax = ax.ravel()
    for i, cluster_id in enumerate(cluster_ids):
        # plot mean dropout heatmap for this cluster
        ax[i] = plot_dropout_heatmap(cluster_meta, feature_matrix, cre_line, cluster_id,
                                     abbreviate_experience=abbreviate_experience, abbreviate_features=True, ax=ax[i])

        # # population average for this cluster
        # ax[i + (n_clusters * 1)] = plot_population_average_response_for_cluster(cluster_mdf, cre_line, cluster_id,
        #                                                                         ax=ax[i + (n_clusters * 1)])
        # ax[i + (n_clusters * 1)].set_xlabel('time (s)')
        # if i > 0:
        #     ax[i + (n_clusters * 1)].set_ylabel('')

    fig.subplots_adjust(hspace=1.2, wspace=0.6)
    # fig.suptitle(get_cell_type_for_cre_line(cre_line, cluster_meta), x=0.52, y=1.1, fontsize=16)
    # fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'clusters_row_' + cre_line.split('-')[0] + suffix, formats=formats)


def plot_clusters_column(cluster_meta, feature_matrix, cre_line,
                         sort_order=None, save_dir=None, folder=None, suffix='', formats=['.png', '.pdf']):
    """
    For each cluster in a given cre_line, plots dropout heatmaps, fraction cells per area/depth relative to chance,
    fraction cells per cluster per area/depth, and population average omission response.
    Will sort clusters according to sort_order dict if provided

    :param cluster_meta: table of metadata for each cell_specimen_id (rows), including cluster_id for each cell_specimen_id
    :param feature_matrix: dropout scores for matched cells with experience levels x features as cols, cells as rows
    :param cre_line: cre line to plot for
    :param sort_order: dictionary with cre_lines as keys, sorted cluster_ids as values
    :param save_dir: directory to save plot to
    :param folder: folder within save_dir to save plot to
    :param suffix: string to be appended to end of filename
    :return:
    """

    cluster_ids = np.sort(cluster_meta[cluster_meta.cre_line == cre_line].cluster_id.unique())
    # if order to sort clusters is provided, use it
    if sort_order:
        cluster_ids = sort_order[cre_line]
    n_clusters = len(cluster_ids)

    n_rows = 1
    figsize = ( n_rows * 2, n_clusters * 3.5)
    fig, ax = plt.subplots(n_clusters, n_rows, figsize=figsize, sharex='row', sharey='row')
    # gridspec_kw={'height_ratios': [1, 0.75]})
    ax = ax.ravel()
    for i, cluster_id in enumerate(cluster_ids):
        # plot mean dropout heatmap for this cluster
        ax[i] = plot_dropout_heatmap(cluster_meta, feature_matrix, cre_line, cluster_id,
                                     abbreviate_experience=True, abbreviate_features=True, ax=ax[i])

        # # population average for this cluster
        # ax[i + (n_clusters * 1)] = plot_population_average_response_for_cluster(cluster_mdf, cre_line, cluster_id,
        #                                                                         ax=ax[i + (n_clusters * 1)])
        # ax[i + (n_clusters * 1)].set_xlabel('time (s)')
        # if i > 0:
        #     ax[i + (n_clusters * 1)].set_ylabel('')

    fig.subplots_adjust(hspace=1.2, wspace=0.6)
    # fig.suptitle(get_cell_type_for_cre_line(cre_line, cluster_meta), x=0.52, y=1.1, fontsize=16)
    # fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'clusters_column_' + cre_line.split('-')[0] + suffix, formats=formats)


def plot_clusters_stats_pop_avg_rows(cluster_meta, feature_matrix, multi_session_df, cre_line,
                                     columns_to_groupby=['targeted_structure', 'layer'], change=False, omitted=True,
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
        ax[i] = plot_dropout_heatmap(cluster_meta_cre, feature_matrix, cre_line, cluster_id, small_fontsize=True, ax=ax[i])

        # plot population averages per cluster
        ax[i + (n_clusters * 1)] = plot_population_average_response_for_cluster(cluster_mdf, cre_line, cluster_id, change, omitted,
                                                                                ax=ax[i + (n_clusters * 1)])
        ax[i + (n_clusters * 1)].set_xlabel('time (s)')
        if i > 0:
            ax[i + (n_clusters * 1)].set_ylabel('')

        # plot proportion of cells per cluster relative to cluster average, with stats
        # get confidence intervals for this cluster id
        ax[i + (n_clusters * 2)] = plot_proportion_cells_for_cluster(cre_proportions, cluster_id, ci_df=ci_df, ax=ax[i + (n_clusters * 2)])
        # abbreviate area names
        xticklabels = [ax[i + (n_clusters * 2)].get_xticklabels()[y].get_text() for y in range(len(ax[i + (n_clusters * 2)].get_xticklabels()))]
        abbreviated_xticklabels = [get_abbreviated_location(xticklabel, abbrev_layer=False) for xticklabel in xticklabels]
        ax[i + (n_clusters * 2)].set_xticklabels(abbreviated_xticklabels, rotation=90)

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
    fig.suptitle(get_cell_type_for_cre_line(cre_line, cluster_meta), x=0.51, y=1.01, fontsize=16)
    # fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'clusters_stats_rows_' + cre_line.split('-')[0] + suffix)


def plot_cluster_data(cluster_meta, feature_matrix, cre_line, cluster_id, multi_session_df=None,
                      columns_to_groupby=['targeted_structure', 'layer'], change=False, omitted=True,
                      abbreviate_features=True, abbreviate_experience=True,
                      save_dir=None, ax=None):
    """
    for a given cluster_id,
    plots the cluster average dropout scores as heatmap of matrix of features x experience levels,
    population average for cluster if multi_session_df is provided, otherwise will plot
    overall size of cluster within the cre line as a pie chart,
    and the proportion of cells relative to cluster average as barplot with stats
    """
    # get relevant data
    cre_csids = cluster_meta[(cluster_meta['cre_line'] == cre_line)].index.values
    this_cluster_meta = cluster_meta[(cluster_meta['cluster_id'] == cluster_id) &
                                     (cluster_meta['cre_line'] == cre_line)]
    this_cluster_csids = this_cluster_meta.index.values
#     feature_matrix_cre = processing.get_feature_matrix_for_cre_line(feature_matrix, cluster_meta, cre_line)
#     mean_dropout_df = feature_matrix_cre.loc[this_cluster_csids].mean().unstack()

    # plot
    if ax is None:
        figsize = (12, 2.5)
        fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 1]})
        fig.subplots_adjust(wspace=0.7)

    # cluster average dropout scores heatmap
    if multi_session_df is not None:  # if using pop avg instead of pie chart of cluster size, put cluster size in title
        cluster_size_in_title = True
    else:
        cluster_size_in_title = False
    ax[0] = plot_dropout_heatmap(cluster_meta, feature_matrix, cre_line, cluster_id,
                                 abbreviate_features=abbreviate_features, abbreviate_experience=abbreviate_experience,
                                 small_fontsize=False, cluster_size_in_title=cluster_size_in_title, ax=ax[0])

    # plot population average or pie chart of cluster size
    i = 2  # plot in third column
    if multi_session_df is not None:
        cluster_mdf = multi_session_df.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']],
                                             on='cell_specimen_id', how='inner')
        ax[i] = plot_population_average_response_for_cluster(cluster_mdf, cre_line, cluster_id, change, omitted,
                                                             small_fontsize=False, ax=ax[i])
    else:
        # pie chart of overall cluster size
        # size is ratio of number of cells in this cluster vs all cells in this cre line
        fraction_cre = len(this_cluster_csids) / float(len(cre_csids))
        fraction = np.round(fraction_cre * 100, 1)
        colors = utils.get_cre_line_color_dict()
        ax[i].pie(x=[fraction, 100 - fraction], colors=[colors[cre_line], [0.8, 0.8, 0.8]],
                  startangle=0, radius=1.8, center=(-1, 0))
        ax[i].set_title(str(fraction) + '%', loc='left')
        ax[i].set_xlim(-2, 2)
        ax[i].set_ylim(-2, 2)

    # fraction cells per area depth
    # get proportion cells relative to cluster average
    proportions, stats_table = processing.get_proportion_cells_rel_cluster_average(cluster_meta, cluster_meta.cre_line.unique(), columns_to_groupby)
    # limit to this cre line
    cre_proportions = proportions[proportions.cre_line == cre_line]
    cre_stats = stats_table[stats_table.cre_line == cre_line]
    # get max proportion value regardless of sign
    xlim = cre_proportions.proportion_cells.abs().max()
    barx = [xlim, xlim]  # x location to put stats bar

    # plot proportions
    i = 1  # plot in second column
    ax[i] = plot_proportion_cells_for_cluster(cre_proportions, cluster_id, ci_df=None, ax=ax[i], orient='h', small_fontsize=False)
    yticklabels = [ax[i].get_yticklabels()[y].get_text() for y in range(len(ax[i].get_yticklabels()))]
    abbreviated_yticklabels = [get_abbreviated_location(yticklabel, abbrev_layer=False) for yticklabel in yticklabels]
    ax[i].set_yticklabels(abbreviated_yticklabels)
    ax[i].set_xlabel('proportion')

    # plot significance with bh corrected chi-square test
    this_s = cre_stats.loc[cluster_id]
    if this_s['bh_significant'] == True:
        bary = [int(len(cre_proportions.location.unique())) - 1, 0]  # bar y is length of # locations
        mid = np.mean(bary)  # midpoint is mean of # locations
        ax[i].plot([x * 1.1 for x in barx], bary, color=[0.7, 0.7, 0.7])
        if this_s['imq'] < 0.0005:
            sig_text = '***'
        elif this_s['imq'] < 0.005:
            sig_text = '**'
        elif this_s['imq'] < 0.05:
            sig_text = '*'
        else:
            sig_text = ''
        ax[i].text(barx[0] * 1.2, mid, sig_text, color='k', rotation=90)
    ax[i].set_xlim(-xlim * 1.2, xlim * 1.2)

    if save_dir:
        if multi_session_df is not None:
            suffix = '_pop_avg'
        else:
            suffix = ''
        if len(columns_to_groupby) == 1:
            folder = 'individual_cluster_plots_' + columns_to_groupby[0] + suffix
        elif len(columns_to_groupby) == 2:
            folder = 'individual_cluster_plots_' + columns_to_groupby[0] + '_' + columns_to_groupby[1] + suffix
        else:
            print('function cant handle columns_to_groupby of greater than length 2')
        utils.save_figure(fig, figsize, save_dir, folder, cre_line[:3] + '_cluster_' + str(cluster_id))
    return ax


def plot_clusters_stats_pop_avg_cols(cluster_meta, feature_matrix, multi_session_df, cre_line,
                                     columns_to_groupby=['targeted_structure', 'layer'],
                                     sort_order=None, save_dir=None, folder=None, suffix=''):
    """
    For each cluster in a given cre_line, plots dropout heatmaps,
    proportion cells per area/depth relative to cluster average,
    and population average omission response.
    Will sort clusters according to sort_order dict if provided

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
    :return:
    """

    cluster_ids = np.sort(cluster_meta[cluster_meta.cre_line == cre_line].cluster_id.unique())
    n_cols = 3
    figsize = (n_cols * 4, len(cluster_ids) * 3)
    fig, ax = plt.subplots(len(cluster_ids), n_cols, figsize=figsize, sharex='col',
                           gridspec_kw={'width_ratios': [1, 1.25, 1.25]})
    ax = ax.ravel()
    for c, cluster_id in enumerate(cluster_ids):
        ax[c * 3:(c * 3) + 3] = plot_cluster_data(cluster_meta, feature_matrix, cre_line, cluster_id,
                                                  multi_session_df=multi_session_df,
                                                  columns_to_groupby=columns_to_groupby,
                                                  abbreviate_features=False, abbreviate_experience=False,
                                                  ax=ax[c * 3:(c * 3) + 3])
        ax[(c * 3) + 1].set_xlabel('')
        ax[(c * 3) + 2].set_xlabel('')
        if cre_line == 'Slc17a7-IRES2-Cre':
            ax[(c * 3) + 2].set_ylim(0, 0.025)

    # top and bottom row labels
    ax[1].set_title('proportion relative\nto cluster average')
    ax[(c * 3) + 1].set_xlabel('proportion')
    ax[(c * 3) + 2].set_xlabel('time (sec)')
    ax[2].set_title('cluster average\nomission response')

    fig.subplots_adjust(hspace=0.55, wspace=0.7)
    if cre_line == 'Sst-IRES-Cre':
        y = 0.96
    elif cre_line == 'Vip-IRES-Cre':
        y = 0.915
    else:
        y = 0.92
    fig.suptitle(get_cell_type_for_cre_line(cre_line, cluster_meta), x=0.49, y=y, fontsize=22)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'cluster_stats_cols_' + cre_line.split('-')[0] + suffix)


def plot_gap_statistic(gap_statistic, cre_lines=None, n_clusters_cre=None, tag='', save_dir=None, folder=None):

    if n_clusters_cre is None:
        n_clusters_cre = processing.get_n_clusters_cre()

    if cre_lines is None:
        cre_lines = gap_statistic.keys()

    for cre_line in cre_lines:

        suffix = cre_line + '_' + tag
        n_clusters = n_clusters_cre[cre_line]
        x = len(gap_statistic[cre_line]['gap'])

        figsize = (8, 4)
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        ax[0].plot(np.arange(1, x + 1), gap_statistic[cre_line]['reference_inertia'], 'o-')
        ax[0].plot(np.arange(1, x + 1), gap_statistic[cre_line]['ondata_inertia'], 'o-')
        ax[0].legend(['reference inertia', 'ondata intertia'])
        ax[0].set_ylabel('Natural log of euclidean \ndistance values')
        ax[0].set_xlabel('Number of clusters')
        ax[0].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')

        ax[1].plot(np.arange(1, x + 1), gap_statistic[cre_line]['gap'], 'o-')
        ax[1].set_ylabel('Gap statistic')
        ax[1].set_xlabel('Number of clusters')
        ax[1].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')
        ax[1].set_ylim([0, 0.3])

        title = processing.get_cre_line_map(cre_line)  # get a more interpretable cell type name
        plt.suptitle(title)
        plt.tight_layout()

        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder, 'Gap_' + suffix )


def plot_gap_statistic_with_sem(gap_statistics, cre_lines=None, n_clusters_cre=None, tag='', save_dir=None, folder=None):

    if n_clusters_cre is None:
        n_clusters_cre = processing.get_n_clusters_cre()

    if cre_lines is None:
        cre_lines = gap_statistics.keys()

    for cre_line in cre_lines:

        suffix = cre_line + '_' + tag
        n_clusters = n_clusters_cre[cre_line]
        x = len(gap_statistics[cre_line]['gap'])

        figsize = (10, 4)
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        ax[0].errorbar(x=np.arange(1, x + 1),
                       y=gap_statistics[cre_line]['reference_inertia'],
                       yerr=gap_statistics[cre_line]['reference_sem'],
                       label='reference inertia')
        ax[0].errorbar(x=np.arange(1, x + 1),
                       y=gap_statistics[cre_line]['ondata_inertia'],
                       yerr=gap_statistics[cre_line]['ondata_sem'],
                       label='ondata inertia')
        ax[0].set_ylabel('Natural log of euclidean \ndistance values')
        ax[0].set_xlabel('Number of clusters')
        ax[0].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')

        ax[1].errorbar(x=np.arange(1, x + 1),
                       y=gap_statistics[cre_line]['gap_mean'],
                       yerr=gap_statistics[cre_line]['gap_sem'],
                       )
        ax[1].set_ylabel('Gap statistic')
        ax[1].set_xlabel('Number of clusters')
        ax[1].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')

        title = processing.get_cre_line_map(cre_line)  # get a more interpretable cell type name
        plt.suptitle(title)
        plt.tight_layout()

        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder, 'Gap_' + suffix )


def plot_eigengap_values(eigenvalues_cre, cre_lines, n_clusters_cre=None, save_dir=None, folder=None):

    if n_clusters_cre is None:
        n_clusters_cre = processing.get_n_clusters_cre()

    for cre_line in cre_lines:
        if len(eigenvalues_cre[cre_line]) < 4:  # patchwork her.
            eigenvalues = eigenvalues_cre[cre_line][1]
        else:
            eigenvalues = eigenvalues_cre[cre_line]
        n_clusters = n_clusters_cre[cre_line]
        suffix = cre_line
        title = processing.get_cre_line_map(cre_line)  # get a more interpretable cell type name

        figsize = (10, 4)
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, '-o')
        # ax[0].grid()
        ax[0].set_ylabel('Eigen values \n(sorted)')
        ax[0].set_xlabel('Eigen number')
        ax[0].set_xlim([0, 20])
        ax[0].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')

        ax[1].plot(np.arange(2, len(eigenvalues) + 1), np.diff(eigenvalues), '-o')
        ax[1].set_ylabel('Eigengap value \n(difference)')
        ax[1].set_xlabel('Eigen number')
        ax[1].set_xlim([0, 20])
        ax[1].set_ylim([0, 0.20])
        # ax[1].grid()
        ax[1].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')
        plt.suptitle(title)
        plt.tight_layout()

        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder, 'eigengap' + suffix)


def plot_fraction_cells_per_cluster_per_location_horiz(cluster_meta, save_dir=None, folder=None):
    if 'location' not in cluster_meta.keys():
        cluster_meta = processing.add_location_column(cluster_meta, columns_to_groupby=['targeted_structure', 'layer'])

    n_cells = \
        cluster_meta.groupby(['cre_line', 'location', 'cluster_id']).count().rename(columns={'labels': 'n_cells'})[
            ['n_cells']]
    total_cells = cluster_meta.groupby(['cre_line', 'location']).count().rename(columns={'labels': 'total_cells'})[
        ['total_cells']]
    fraction_cells = n_cells.reset_index().merge(total_cells.reset_index(), on=['cre_line', 'location'])
    fraction_cells['fraction_cells'] = fraction_cells.n_cells / fraction_cells.total_cells

    locations = ['VISp_upper', 'VISl_upper', 'VISp_lower', 'VISl_lower']
    cre_lines = np.sort(fraction_cells.cre_line.unique())[::-1]

    for cre_line in cre_lines:
        cre_data = fraction_cells[fraction_cells.cre_line == cre_line]
        cluster_ids = np.sort(cre_data.cluster_id.unique())
        if cre_line == 'Sst-IRES-Cre':
            figsize = (2 * len(cluster_ids), 2)
        elif cre_line == 'Vip-IRES-Cre':
            figsize = (1.5 * len(cluster_ids), 2)
        else:
            figsize = ((0.75 * 2) * len(cluster_ids), 2)
        fig, ax = plt.subplots(1, 4, figsize=figsize, sharey=True)
        for i, location in enumerate(locations):
            data = cre_data[cre_data.location == location]
            ax[i].vlines(x=data.cluster_id.values, ymin=0, ymax=data.fraction_cells.values)
            ax[i].scatter(x=data.cluster_id.values, y=data.fraction_cells.values, color='k')
            ax[i].set_title(get_abbreviated_location(location, abbrev_layer=False))
            ax[i].set_xlim(0, len(cluster_ids) + 1)
            ax[i].set_xticks(np.arange(1, len(cluster_ids) + 1, 1))
            ax[i].set_xticklabels(np.arange(0, len(cluster_ids), 1) + 1)
            ax[i].set_xlabel('cluster ID')
        ax[0].set_ylabel('fraction cells')
        #         fig.suptitle(processing.get_cell_type_for_cre_line(cre_line, cluster_meta), x=0.52, y=1.02)
        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder, 'fraction_cells_per_cluster_horiz_' + cre_line[:3])


def plot_fraction_cells_per_cluster_per_location_vert(cluster_meta, save_dir=None, folder=None):
    if 'location' not in cluster_meta.keys():
        cluster_meta = processing.add_location_column(cluster_meta,
                                                      columns_to_groupby=['targeted_structure', 'layer'])

    n_cells = \
        cluster_meta.groupby(['cre_line', 'location', 'cluster_id']).count().rename(columns={'labels': 'n_cells'})[
            ['n_cells']]
    total_cells = cluster_meta.groupby(['cre_line', 'location']).count().rename(columns={'labels': 'total_cells'})[
        ['total_cells']]
    fraction_cells = n_cells.reset_index().merge(total_cells.reset_index(), on=['cre_line', 'location'])
    fraction_cells['fraction_cells'] = fraction_cells.n_cells / fraction_cells.total_cells

    locations = ['VISp_upper', 'VISl_upper', 'VISp_lower', 'VISl_lower']
    cre_lines = np.sort(fraction_cells.cre_line.unique())[::-1]

    for cre_line in cre_lines:
        cre_data = fraction_cells[fraction_cells.cre_line == cre_line]
        cluster_ids = np.sort(cre_data.cluster_id.unique())
        if cre_line == 'Sst-IRES-Cre':
            figsize = (8, 1 * len(cluster_ids))
        elif cre_line == 'Vip-IRES-Cre':
            figsize = (8, 0.7 * len(cluster_ids))
        else:
            figsize = (8, 0.75 * len(cluster_ids))
        fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True)
        ax = ax.ravel()
        for i, location in enumerate(locations):
            data = cre_data[cre_data.location == location]
            ax[i].hlines(y=data.cluster_id.values, xmin=0, xmax=data.fraction_cells.values)
            ax[i].scatter(y=data.cluster_id.values, x=data.fraction_cells.values, color='k')
            ax[i].set_title(get_abbreviated_location(location, abbrev_layer=False))
            ax[i].set_ylim(0, len(cluster_ids) + 1)
            ax[i].set_yticks(np.arange(1, len(cluster_ids) + 1, 1))
            ax[i].set_yticklabels(np.arange(0, len(cluster_ids), 1) + 1)
            ax[i].invert_yaxis()
        ax[2].set_xlabel('fraction cells')
        ax[3].set_xlabel('fraction cells')
        ax[0].set_ylabel('cluster ID')
        ax[2].set_ylabel('cluster ID')
        fig.tight_layout()
        fig.suptitle(processing.get_cell_type_for_cre_line(cre_line, cluster_meta), x=0.54, y=1.02)
        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder, 'fraction_cells_per_cluster_vert_' + cre_line[:3])


def plot_proportion_cells_area_depth_pie_chart(cluster_meta, save_dir=None, folder=None):
    colors = sns.color_palette('Greys', 10)

    figsize = (15, 15)
    fig, ax = plt.subplots(3, 4, figsize=figsize)
    ax = ax.ravel()
    i = 0
    for cre_line in get_cre_lines(cluster_meta):
        for area in ['VISp', 'VISl']:
            for layer in ['upper', 'lower']:
                cells = cluster_meta[(cluster_meta.targeted_structure == area) & (cluster_meta.layer == layer) & (cluster_meta.cre_line == cre_line)]

                total = len(cells.index.unique())

                cluster_cells = cells.groupby('cluster_id').count()[['labels']].rename(columns={'labels': 'n_cells'})
                cluster_cells['fraction'] = cluster_cells.n_cells / total
                cluster_cells = cluster_cells.reset_index()

                ax[i].pie(cluster_cells.fraction, labels=cluster_cells.cluster_id, autopct='%.f%%', colors=colors)
                ax[i].set_title(cre_line + '\n' + area + '-' + layer)
                i += 1
    fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'proportion_cells_pie_chart')


def plot_cluster_proportion_pie_legends(save_dir=None, folder=None):
    # cluster types
    cluster_types = processing.get_cluster_types()

    fake_data = pd.DataFrame()
    fake_data['cluster_id'] = np.arange(0, len(cluster_types), 1)
    fake_data['fraction'] = np.repeat(100 / len(cluster_types), len(cluster_types))
    fake_data['cluster_type'] = processing.get_cluster_types()
    fake_data = fake_data.merge(processing.get_cluster_type_color_df(), on='cluster_type')

    figsize = (5, 5)
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts = ax.pie(fake_data.fraction.values, radius=1, wedgeprops=dict(width=0.4, edgecolor='w'),
                           colors=fake_data.cluster_type_color.values, labels=fake_data.cluster_type.values)
    ax.legend(bbox_to_anchor=(1.5, 1), fontsize='small')
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'cluster_proportions_cluster_types_legend')

    # experience_levels
    experience_levels = utils.get_experience_levels()
    exp_colors = utils.get_experience_level_colors()

    fake_data = pd.DataFrame()
    fake_data['cluster_id'] = np.arange(0, len(experience_levels), 1)
    fake_data['fraction'] = np.repeat(100 / len(experience_levels), len(experience_levels))
    fake_data['dominant_experience_level'] = experience_levels
    fake_data['exp_level_colors'] = exp_colors

    figsize = (5, 5)
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts = ax.pie(fake_data.fraction.values, radius=1, wedgeprops=dict(width=0.4, edgecolor='w'),
                           colors=fake_data.exp_level_colors.values, labels=fake_data.dominant_experience_level.values)
    ax.legend(bbox_to_anchor=(1.5, 1), fontsize='small')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'cluster_proportions_exp_levels_legend')


def plot_cluster_proportions_for_location(location_fractions, cre_line, location, ax=None, save_dir=None, folder=None):
    data = location_fractions[(location_fractions.cre_line == cre_line) & (location_fractions.location == location)]
    data = data.sort_values(by='cluster_id')
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots()
    wedges, texts = ax.pie(data.fraction.values, radius=1, colors=data.cluster_type_color.values,
                           labels=data.cluster_id.values, textprops=dict(fontsize=12, color='k'),
                           wedgeprops=dict(width=0.4, edgecolor='w'))

    wedges, texts = ax.pie(data.fraction.values, radius=0.6, colors=data.exp_level_color.values,
                           #                                    autopct=lambda p: '{:.2f}'.format(p/100),
                           labels=np.round(data.fraction.values, 2), rotatelabels=True,
                           labeldistance=1, textprops=dict(fontsize=12, color='w'),
                           wedgeprops=dict(width=0.3, edgecolor='w'))
    ax.set_title(get_abbreviated_location(location, abbrev_layer=False))
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, cre_line[:3] + '_' + location)

#
# def plot_cluster_proportions_for_location(location_fractions, cre_line, location, ax=None, save_dir=None, folder=None):
#
#     data = location_fractions[(location_fractions.cre_line==cre_line)&(location_fractions.location==location)]
#     data = data.sort_values(by=['location', 'cluster_type'])
#     # data = data.sort_values(by='cluster_id')
#     if ax is None:
#         figsize = (5,5)
#         fig, ax = plt.subplots()
#     wedges, texts = ax.pie(data.fraction.values, radius=1, colors=data.cluster_type_color.values,
#                                    labels=data.cluster_id.values, textprops=dict(fontsize=12, color='k'),
#                                    wedgeprops=dict(width=0.4, edgecolor='w'))
#
#     wedges, texts = ax.pie(data.fraction.values, radius=0.6, colors=data.exp_level_color.values,
#                                    labels=np.round(data.fraction.values,2), rotatelabels=True,
#                                    labeldistance=1, textprops=dict(fontsize=12, color='w'),
#                                    wedgeprops=dict(width=0.3, edgecolor='w'))
#     ax.set_title(get_abbreviated_location(location, abbrev_layer=False))
#     if save_dir:
#         utils.save_figure(fig, figsize, plot_save_dir, folder, cre_line[:3]+'_'+location)


def plot_cluster_proportion_donut_for_location(location_fractions, cre_line, location, use_exp_mod_continuous=False,
                                               ax=None, save_dir=None, folder=None):
    """
    Create 2 tiered donut plot with fraction of cells per cluster as wedge size, color of outer ring indicating
    feature preference of each cluster, and color of inner ring indicating either the dominant experience level
    or a continuous measure of experience modulation computed as the difference of novel session to average of F and N>1 over the sum
    :param location_fractions:
    :param cre_line:
    :param location:
    :param ax:
    :param save_dir:
    :param folder:
    :return:
    """
    data = location_fractions[(location_fractions.cre_line == cre_line) & (location_fractions.location == location)]
    data = data.sort_values(by=['location', 'cluster_type'])
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots()
    wedges, texts = ax.pie(data.fraction.values, radius=1, colors=data.cluster_type_color.values,
                           labels=data.cluster_id.values, textprops=dict(fontsize=12, color='k'),
                           wedgeprops=dict(width=0.4, edgecolor='w'))

    if use_exp_mod_continuous:
        # experience modulation as continual variable
        wedges, texts = ax.pie(data.fraction.values, radius=0.6, colors=data.exp_mod_color.values,
                               labels=np.round(data.fraction.values, 2), rotatelabels=True,
                               labeldistance=1, textprops=dict(fontsize=12, color='w'),
                               wedgeprops=dict(width=0.3, edgecolor='w'))

    else:
        # dominant experience level
        wedges, texts = ax.pie(data.fraction.values, radius=0.6, colors=data.exp_level_color.values,
                               labels=np.round(data.fraction.values, 2), rotatelabels=True,
                               labeldistance=1, textprops=dict(fontsize=12, color='w'),
                               wedgeprops=dict(width=0.3, edgecolor='w'))

    # if using dominant experience level, reduce alpha of everything other than Novel 1
    if use_exp_mod_continuous == False:
        for i, exp_level in enumerate(data.dominant_experience_level.values):
            if exp_level != 'Novel 1':
                wedges[i].set_alpha(0.3)

    ax.set_title(get_abbreviated_location(location, abbrev_layer=False))
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, cre_line[:3] + '_' + location)


def plot_cluster_proportion_donuts_all_locations(location_fractions, cluster_metrics, use_exp_mod_continuous=False,
                                                 save_dir=None, folder=None):
    """
    Generate and save donut plots with feature preference and experience level preference for each cluster for all cre lines & locations
    Will either use dominant experience level, or a continuous metric of experience modulation
    computed as the difference of novel session to average of F and N>1 over the sum
    :param location_fractions:
    :param cluster_metrics:
    :param use_exp_mod_continuous:
    :param save_dir:
    :param folder:
    :return:
    """
    locations = ['VISp_upper', 'VISl_upper', 'VISp_lower', 'VISl_lower']
    cre_lines = get_cre_lines(location_fractions)

    for cre_line in cre_lines:
        figsize = (8, 8)
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        ax = ax.ravel()
        for i, location in enumerate(locations):
            plot_cluster_proportion_donut_for_location(location_fractions, cre_line, location, use_exp_mod_continuous=use_exp_mod_continuous,
                                                       ax=ax[i], save_dir=None, folder=None)
            ax[i].set_title(location)
        fig.suptitle(get_cell_type_for_cre_line(cre_line), x=0.5, y=1.1)
        fig.tight_layout()
        if save_dir:
            if use_exp_mod_continuous:
                suffix = '_continuous_exp_mod'
            else:
                suffix = '_pref_exp_level'
            utils.save_figure(fig, figsize, save_dir, folder, 'cluster_proportions_per_location_' + cre_line[:3] + suffix)


def plot_cell_counts_per_location(cluster_meta, save_dir=None, folder=None, ax=None):
    """
    Barplot of the number of cells for each area and depth for each cre line

    :param cluster_meta:
    :param save_dir:
    :param folder:
    :param ax:
    :return:
    """
    if 'location' not in cluster_meta.keys():
        cluster_meta = processing.add_location_column(cluster_meta,
                                                      columns_to_groupby=['targeted_structure', 'layer'])

    n_cells = cluster_meta.groupby(['cre_line', 'location', 'cluster_id']).count().rename(columns={'labels': 'n_cells'})[['n_cells']]
    total_cells = cluster_meta.groupby(['cre_line', 'location']).count().rename(columns={'labels': 'total_cells'})[['total_cells']]
    fraction_cells = n_cells.reset_index().merge(total_cells.reset_index(), on=['cre_line', 'location'])
    fraction_cells['fraction_cells'] = fraction_cells.n_cells / fraction_cells.total_cells

    data = total_cells.reset_index()
    locations = np.sort(data.location.unique())[::-1]
    cre_lines = np.sort(data.cre_line.unique())[::-1]
    if ax is None:
        figsize = (6, 4)
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.barplot(data=data, x='cre_line', y='total_cells', hue='location',
                         order=cre_lines, hue_order=locations, palette=sns.color_palette('Paired'), ax=ax)
        ax.set_xticklabels([get_cell_type_for_cre_line(cre_line, cluster_meta) for cre_line in cre_lines],
                           rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('# cells per location')
        ax.legend(fontsize='small', title_fontsize='small')
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'n_cells_per_location')
    return ax


def plot_number_mice_per_cluster(cluster_meta, save_dir=None, folder=None):
    n_mice = cluster_meta.groupby(['cre_line', 'cluster_id', 'mouse_id']).count().reset_index().groupby(['cre_line', 'cluster_id']).count().rename(columns={'mouse_id': 'n_mice'})[['n_mice']]
    n_mice = n_mice.reset_index()
    cre_lines = processing.get_cre_lines(cluster_meta)

    figsize = (15, 4)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i, cre_line in enumerate(cre_lines[::-1]):
        data = n_mice[n_mice.cre_line == cre_line]
        ax[i] = sns.barplot(data=data, x='cluster_id', y='n_mice', ax=ax[i], palette='Greys')
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))

    fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'n_mice_per_cluster')


def plot_number_clusters_per_mouse(cluster_meta, save_dir=None, folder=None):
    n_clusters = cluster_meta.groupby(['cre_line', 'equipment_name', 'mouse_id', 'cluster_id', ]).count().reset_index().groupby(['cre_line', 'equipment_name', 'mouse_id']).count().rename(columns={'cluster_id': 'n_clusters'})[['n_clusters']]
    n_clusters = n_clusters.reset_index()
    cre_lines = processing.get_cre_lines(cluster_meta)

    # rigs = np.sort(n_clusters.equipment_name.unique())
    figsize = (20, 4)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i, cre_line in enumerate(cre_lines[::-1]):
        data = n_clusters[n_clusters.cre_line == cre_line]
        order = np.sort(data.mouse_id.unique())
        ax[i] = sns.barplot(data=data, x='mouse_id', y='n_clusters', order=order,
                            ax=ax[i],)  # hue='equipment_name', hue_order=rigs)
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))
        ax[i].set_xticklabels(order, rotation=90, fontsize=12)
        ax[i].legend(bbox_to_anchor=(1, 1), fontsize='xx-small')
    fig.tight_layout()
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'n_clusters_per_mouse')


def plot_feature_preference_barplot(cluster_metrics, save_dir=None, folder=None):
    """
    plot fraction of clusters per cre line that prefer each feature
    """
    cre_lines = np.sort(cluster_metrics.cre_line.unique())[::-1]
    palette = utils.get_cre_line_colors()
    order = ['all-images', 'omissions', 'behavioral', 'task']

    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.countplot(data=cluster_metrics, x='dominant_feature', hue='cre_line', order=order,
                       hue_order=cre_lines, palette=palette, ax=ax
                       )
    ax.legend(bbox_to_anchor=(1, 1), fontsize='xx-small', title_fontsize='xx-small')
    ax.set_xticklabels(order, rotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('# clusters')
    ax.set_title('feature preference')
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'feature_preference_barplot')


def plot_exp_level_preference_barplot(cluster_metrics, save_dir=None, folder=None):
    """
    plot fraction of clusters per cre line that prefer each experience level
    """
    cre_lines = np.sort(cluster_metrics.cre_line.unique())[::-1]
    palette = utils.get_cre_line_colors()
    order = ['Familiar', 'Novel 1', 'Novel >1']

    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.countplot(data=cluster_metrics, x='dominant_experience_level', hue='cre_line', order=order,
                       hue_order=cre_lines, palette=palette, ax=ax
                       )
    ax.legend(bbox_to_anchor=(1, 1), fontsize='xx-small', title_fontsize='xx-small')
    ax.set_xticklabels(order, rotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('# clusters')
    ax.set_title('experience level preference')
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'exp_level_preference_barplot')


def plot_cluster_percent_pie_legends(save_dir=None, folder=None):
    """
    Create dummy figures with colorbars & legends for experience level modulation, feature preference, etc.
    :param save_dir:
    :param folder:
    :return:
    """
    # make figure to put everything on
    figsize = (20, 20)
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    ax = ax.ravel()

    # cluster types
    cluster_types = processing.get_cluster_types()

    fake_data = pd.DataFrame()
    fake_data['cluster_id'] = np.arange(0, len(cluster_types), 1)
    fake_data['fraction'] = np.repeat(100 / len(cluster_types), len(cluster_types))
    fake_data['cluster_type'] = cluster_types
    fake_data = fake_data.merge(processing.get_cluster_type_color_df(), on='cluster_type')

    wedges, texts = ax[0].pie(fake_data.fraction.values, radius=1, wedgeprops=dict(width=0.4, edgecolor='w'),
                              colors=fake_data.cluster_type_color.values, labels=fake_data.cluster_type.values)
    ax[0].legend(bbox_to_anchor=(1.5, 1), fontsize='small')

    # experience_levels
    experience_levels = utils.get_experience_levels()
    exp_colors = utils.get_experience_level_colors()

    fake_data = pd.DataFrame()
    fake_data['cluster_id'] = np.arange(0, len(experience_levels), 1)
    fake_data['fraction'] = np.repeat(100 / len(experience_levels), len(experience_levels))
    fake_data['dominant_experience_level'] = experience_levels
    fake_data['exp_level_colors'] = exp_colors

    wedges, texts = ax[1].pie(fake_data.fraction.values, radius=1, wedgeprops=dict(width=0.4, edgecolor='w'),
                              colors=fake_data.exp_level_colors.values,
                              labels=fake_data.dominant_experience_level.values)
    ax[1].legend(bbox_to_anchor=(1.5, 1), fontsize='small')

    # experience_modulation
    experience_modulation_index_range = np.arange(-1, 1, 0.1)
    experience_modulation_index_range = np.asarray([np.round(i, 1) for i in experience_modulation_index_range])
    cmap = plt.get_cmap('RdBu')
    exp_mod_normed = ((experience_modulation_index_range + 1) / 2) * 256
    colors = [cmap(int(np.round(i)))[:3] for i in exp_mod_normed]

    fake_data = pd.DataFrame()
    fake_data['cluster_id'] = experience_modulation_index_range
    fake_data['fraction'] = np.repeat(100 / len(experience_modulation_index_range),
                                      len(experience_modulation_index_range))
    fake_data['exp_mod'] = experience_modulation_index_range
    fake_data['exp_mod_colors'] = colors

    wedges, texts = ax[2].pie(fake_data.fraction.values, radius=1, wedgeprops=dict(width=0.4, edgecolor='w'),
                              colors=fake_data.exp_mod_colors.values, labels=fake_data.exp_mod.values)
    ax[2].legend(bbox_to_anchor=(1.5, 1), fontsize='small')

    experience_modulation_index_range = np.arange(-1, 1, 0.1)
    # make fake image data with our desired range
    n = len(experience_modulation_index_range)
    m = np.zeros((n, n))
    for i in range(n):
        m[i, :] = experience_modulation_index_range
    # plot it with a colorbar
    im = ax[3].imshow(m, cmap='RdBu', vmin=-1, vmax=1)
    fig.colorbar(im, label='experience modulation', shrink=0.7)

    fig.subplots_adjust(hspace=0.9, wspace=0.9)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'cluster_proportions_legends')


def plot_cluster_proportions_barplots(location_fractions, color_column='cluster_type_color', sort_by='cluster_type',
                                      save_dir=None, folder=None):
    """
    Plot fraction of cells per cluster in each area and depth, colorized by color_column input
    color_column can be ['cluster_type_color', 'exp_level_color',  'exp_mod_color']
    clusters in the barplot can be sorted by ['cluster_id', 'cluster_type', 'experience_modulation']
    Plots for all cre lines & locations
    :param location_fractions:
    :param color_column:
    :param save_dir:
    :param folder:
    :return:
    """
    locations = np.sort(location_fractions.location.unique())
    cre_lines = get_cre_lines(location_fractions)

    figsize = (15, 4)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    for i, cre_line in enumerate(cre_lines):
        for x_pos, location in enumerate(locations):
            data = location_fractions[(location_fractions.cre_line == cre_line) & (location_fractions.location == location)]
            data = data.sort_values(by=sort_by, ascending=False)
            cluster_ids = data.cluster_id.values
            next_bar_start = 0
            for c, cluster_id in enumerate(cluster_ids):
                if c == 0:
                    this_value = data[data.cluster_id == cluster_id].fraction.values[0]
                    color = data[data.cluster_id == cluster_id][color_column].values[0]
                    ax[i].bar(x_pos, this_value, bottom=next_bar_start, color=color, edgecolor='w')
                else:
                    this_value = data[data.cluster_id == cluster_id].fraction.values[0]
                    color = data[data.cluster_id == cluster_id][color_column].values[0]
                    ax[i].bar(x_pos, this_value, bottom=next_bar_start, color=color, edgecolor='w')
                next_bar_start = next_bar_start + this_value
        ax[i].set_xticks(np.arange(len(locations)))
        ax[i].set_xticklabels(
            [get_abbreviated_location(location, abbrev_layer=False) for location in locations], rotation=90)
        ax[i].set_title(processing.get_cell_type_for_cre_line(cre_line))
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].invert_yaxis()
    ax[0].set_ylabel('proportion of cells')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'cluster_proportions_barplot_' + color_column)


def plot_cluster_proportions_horizontal_barplots(location_fractions, color_column='cluster_type_color', sort_by='cluster_type',
                                                 save_dir=None, folder=None):
    """
    Plot fraction of cells per cluster in each area and depth, colorized by color_column input
    color_column can be ['cluster_type_color', 'exp_level_color',  'exp_mod_color']
    clusters in the barplot can be sorted by ['cluster_id', 'cluster_type', 'experience_modulation']
    Plots for all cre lines & locations
    :param location_fractions:
    :param color_column:
    :param save_dir:
    :param folder:
    :return:
    """
    locations = np.sort(location_fractions.location.unique())
    cre_lines = get_cre_lines(location_fractions)

    figsize = (15, 4)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharex=False, sharey=False)
    for i, cre_line in enumerate(cre_lines):
        for x_pos, location in enumerate(locations):
            data = location_fractions[
                (location_fractions.cre_line == cre_line) & (location_fractions.location == location)]
            data = data.sort_values(by=sort_by, ascending=False)
            cluster_ids = data.cluster_id.values
            next_bar_start = 0
            for c, cluster_id in enumerate(cluster_ids):
                if c == 0:
                    this_value = data[data.cluster_id == cluster_id].fraction.values[0]
                    color = data[data.cluster_id == cluster_id][color_column].values[0]
                    ax[i].barh(y=x_pos, width=this_value, left=next_bar_start, color=color, edgecolor='w', height=0.6)
                    ax[i].text(s=str(cluster_id), x=next_bar_start, y=x_pos, color='gray', fontsize=12)
                else:
                    this_value = data[data.cluster_id == cluster_id].fraction.values[0]
                    color = data[data.cluster_id == cluster_id][color_column].values[0]
                    ax[i].barh(y=x_pos, width=this_value, left=next_bar_start, color=color, edgecolor='w', height=0.6)
                    ax[i].text(s=str(cluster_id), x=next_bar_start, y=x_pos, color='gray', fontsize=12)
                next_bar_start = next_bar_start + this_value
        ax[i].set_yticks(np.arange(len(locations)))
        ax[i].set_yticklabels(
            [get_abbreviated_location(location, abbrev_layer=False) for location in locations], rotation=0)
        ax[i].set_title(get_cell_type_for_cre_line(cre_line))
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].set_xlabel('proportion of cells')
        # ax[i].invert_yaxis()
    fig.subplots_adjust(wspace=0.6)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'cluster_proportions_horiz_barplot_' + color_column)


def plot_cluster_proportions_treemaps(location_fractions, cluster_meta, color_column='cluster_type_color',
                                      sort_by='cluster_type', save_dir=None, folder=None):
    """
    Plot fraction of cells per area and depth for each cre line as a treemap (i.e. squareified pie chart),
    colorized by color_column which can be ['cluster_type_color', 'exp_level_color',  'exp_mod_color']
    clusters in the plot can be sorted by ['cluster_id', 'cluster_type', 'experience_modulation']
    Plots for all cre lines & locations
    :param location_fractions:
    :param cluster_meta:
    :param color_column:
    :param sort_by:
    :param save_dir:
    :param folder:
    :return:
    """
    import squarify

    # hard code location order so they go in proper anatomical order in a grid
    locations = ['VISp_upper', 'VISl_upper', 'VISp_lower', 'VISl_upper']
    cre_lines = get_cre_lines(location_fractions)

    for c, cre_line in enumerate(cre_lines):
        figsize = (8, 8)
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        ax = ax.ravel()
        for i, location in enumerate(locations):
            data = location_fractions[
                (location_fractions.cre_line == cre_line) & (location_fractions.location == location)]
            data = data.sort_values(by=sort_by, ascending=False)
            ax[i] = squarify.plot(sizes=data['fraction'], label=data['cluster_id'], color=data[color_column], ax=ax[i],
                                  bar_kwargs={'edgecolor': 'w'})
            ax[i].axis('off')
            ax[i].set_title(get_abbreviated_location(location, abbrev_layer=False))
        fig.suptitle(get_cell_type_for_cre_line(cre_line))

        if save_dir:
            utils.save_figure(fig, figsize, save_dir, folder, 'cluster_proportions_treemap' + cre_line[:3] + '_' + color_column)


# plot cluster sizes
def plot_cluster_size_for_cluster(cluster_size_df, cluster_id, stats_table, diff_column, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    color1 = 'gray'

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plot size diff
    ax = sns.barplot(data=cluster_size_df[cluster_size_df['cluster_id'] == cluster_id], x='cluster_id',
                     y=diff_column, color=color1, ax=ax)
    if stats_table is not None:
        y = cluster_size_df[[diff_column, 'cluster_id']].groupby('cluster_id').mean().max()
        if stats_table[stats_table.cluster_id == cluster_id]['significant'].values[0] == True:
            color = 'DarkRed'
            pvalue = round(stats_table[stats_table.cluster_id == cluster_id]['imq'].values[0], 4)
            if pvalue < 0.001:
                marker = r'***'
                x = -0.25
            elif pvalue < 0.01:
                marker = r'**'
                x = -0.2
            elif pvalue <= 0.05:
                marker = r'*'
                x = -0.1

        else:
            x = -0.25
            marker = r'n.s.'
            color = 'Black'

        ax.text(x, y, s=marker, fontsize=24, color=color )

    ax.axhline(0, color='gray')
    ax.set_xlabel('')
    # ax.set_ylim([-0.4, 1])
    ax.set_xlim([-1, 1])
    ax.set_xticklabels('', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    # ax.set_title(f'cluster {cluster_id}', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylabel('size vs. shuffle', color=color1, fontsize=12)
    ax.set_yticklabels(np.round(ax.get_yticks(), 1), color=color1, fontsize=12)

    ax.set_zorder(ax.get_zorder() + 1)
    ax.patch.set_visible(False)

    return ax


def plot_cluster_size(cluster_size_df, cre_line=None, shuffle_type=None, stats_table=None, diff_column='cluster_size_diff',
                      ax=None, figsize=None, save_dir=None, folder=None):
    if cre_line is not None:
        if isinstance(cre_line, str):
            cluster_size_df = cluster_size_df[cluster_size_df.cre_line == cre_line]
        if stats_table is not None:
            stats_table = stats_table[stats_table.cre_line == cre_line]

    # select which shuffle type to plot
    if shuffle_type is not None:
        cluster_size_df = cluster_size_df[cluster_size_df.shuffle_type == shuffle_type]
        if stats_table is not None:
            stats_table = stats_table[stats_table.shuffle_type == shuffle_type]
    cluster_ids = cluster_size_df.cluster_id.unique()
    if figsize is None:
        figsize = (3.5 * len(cluster_ids), 2)
    if ax is None:
        fig, ax = plt.subplots(1, len(cluster_ids), figsize=figsize, sharey='row')
        ax = ax.ravel()

    # plot cluster size first
    for i, cluster_id in enumerate(cluster_ids):
        ax[i] = plot_cluster_size_for_cluster(cluster_size_df, cluster_id, stats_table=stats_table, diff_column=diff_column, ax=ax[i])

    fig.subplots_adjust(hspace=1.2, wspace=0.6)
    plt.suptitle(processing.get_shuffle_label(shuffle_type), x=0.52, y=1.15)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder,
                          f'{shuffle_type}_size_barplot' + cre_line[:3]  )

    return ax


def plot_cluster_size_and_probability_for_cluster(cluster_size_df, shuffle_probability_df, cluster_id, ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    # color1 = 'dimgray'
    color1 = 'gray'
    color2 = 'steelblue'

    # plot probability first

    ax = sns.pointplot(data=shuffle_probability_df[shuffle_probability_df['cluster_id'] == cluster_id],
                       x='cluster_id', y='probability', ax=ax, color=color2, linestyles='', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticklabels('')
    ax.set_ylim([-0.1, 1.1])
    # ax.set_xticks([0.5, 0])
    ax.set_xlim([-1, 1])
    ax.set_xticklabels('')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_yticklabels(np.round(ax.get_yticks(), 1), color=color2, fontsize=12)
    ax.set_ylabel('probability', color=color2, fontsize=12)

    # plot size diff
    ax2 = ax.twinx()
    ax2 = sns.barplot(data=cluster_size_df[cluster_size_df['cluster_id'] == cluster_id], x='cluster_id',
                      y='cluster_size_diff', color=color1, ax=ax2)
    ax2.axhline(0, color='gray')
    ax2.set_xlabel('')
    ax2.set_ylim([-0.4, 1])
    ax2.set_xlim([-1, 1])
    # ax2.set_xticks([1, 0])
    ax2.set_xticklabels('', fontsize=12)
    ax2.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    # ax2.set_title(f'cluster {cluster_id}', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax2.set_ylabel('size vs. shuffle', color=color1, fontsize=12)
    ax2.set_yticklabels(np.round(ax2.get_yticks(), 1), color=color1, fontsize=12)

    ax.set_zorder(ax.get_zorder() + 1)
    ax.patch.set_visible(False)

    return ax


def plot_cluster_size_and_probability(cluster_size_df, shuffle_probability_df, cre_line=None, shuffle_type=None,
                                      ax=None, figsize=None, save_dir=None, folder=None):
    if cre_line is not None:
        if isinstance(cre_line, str):
            cluster_size_df = cluster_size_df[cluster_size_df.cre_line == cre_line]
            shuffle_probability_df = shuffle_probability_df[shuffle_probability_df.cre_line == cre_line]

    # select which shuffle type to plot
    if shuffle_type is not None:
        cluster_size_df = cluster_size_df[cluster_size_df.shuffle_type == shuffle_type]
        shuffle_probability_df = shuffle_probability_df[shuffle_probability_df.shuffle_type == shuffle_type]
    cluster_ids = cluster_size_df.cluster_id.unique()
    if figsize is None:
        figsize = (3.5 * len(cluster_ids), 2)
    if ax is None:
        fig, ax = plt.subplots(1, len(cluster_ids), figsize=figsize, sharey='row')
        ax = ax.ravel()

    # plot cluster size first
    for i, cluster_id in enumerate(cluster_ids):
        ax[i] = plot_cluster_size_and_probability_for_cluster(cluster_size_df, shuffle_probability_df, cluster_id, ax=ax[i])

    fig.subplots_adjust(hspace=1.2, wspace=0.6)
    plt.suptitle(cre_line, x=0.52, y=1.15)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder,
                          f'{shuffle_type}_prob_size' + cre_line[:3]  )

    return ax


def plot_matched_clusters_heatmap(SSE_mapping, mean_dropout_scores_unstacked, metric='mean', shuffle_type=None,
                                  cre_line=None, abbreviate_features=True, abbreviate_experience=True, small_fontsize=False,
                                  cbar=False, save_dir=None, folder=None, figsize=None):
    ''' This function can plot mean (or other metric like std, median, or custom function) of matched shuffle clusters. This is helpful to see
    how well matching worked but it does not show clusters that were not matched with any original clusters.
    INPUT:
    SSE_mapping: dictionary for each n_boot, original cluster_id: matched shuffled cluster_id
    mean_dropout_scores_unstacked: dictionary ofmean unstacked dropout scores  cluster_id: unstached pd.Data'''

    all_clusters_means_dict = processing.get_matched_clusters_means_dict(SSE_mapping,
                                                                         mean_dropout_scores_unstacked,
                                                                         metric=metric, shuffle_type=shuffle_type,
                                                                         cre_line=cre_line)
    cluster_ids = all_clusters_means_dict.keys()
    if figsize is None:
        figsize = (3.5 * len(cluster_ids), 2)

    fig, ax = plt.subplots(1, len(cluster_ids), figsize=figsize, sharex='row', sharey='row')
    ax = ax.ravel()

    for i, cluster_id in enumerate(cluster_ids):
        if all_clusters_means_dict[cluster_id].sum().sum() == 0:
            hm_color = 'Greys'
        else:
            hm_color = 'Blues'
        features = processing.get_features_for_clustering()
        mean_dropout_df = all_clusters_means_dict[cluster_id].loc[features]  # order regressors in a specific order
        ax[i] = sns.heatmap(mean_dropout_df, cmap=hm_color, vmin=0, vmax=1,
                            ax=ax[i], cbar=cbar, cbar_kws={'label': 'coding score'})

        if abbreviate_features:
            # set yticks to abbreviated feature labels
            feature_abbreviations = get_abbreviated_features(mean_dropout_df.index.values)
            ax[i].set_yticklabels(feature_abbreviations, rotation=0)
        else:
            ax[i].set_yticklabels(mean_dropout_df.index.values, rotation=0, fontsize=14)
        if abbreviate_experience:
            # set xticks to abbreviated experience level labels
            exp_level_abbreviations = get_abbreviated_experience_levels(mean_dropout_df.columns.values)
            ax[i].set_xticklabels(exp_level_abbreviations, rotation=90)
        else:
            ax[i].set_xticklabels(mean_dropout_df.columns.values, rotation=90, fontsize=14)
        ax[i].set_ylim(0, mean_dropout_df.shape[0])
        # invert y axis so images is always on top
        ax[i].invert_yaxis()
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].set_title(f'cluster {cluster_id}')
        if small_fontsize:
            ax[i] = standardize_axes_fontsize(ax[i])

    # shuffle_type_dict = {'experience': 'cell id shuffle',
    #                     'experience_within_cell': 'exp label shuffle'}

    # plt.suptitle(cre_line + ' ' + shuffle_type_dict[shuffle_type])
    fig.subplots_adjust(hspace=1.2, wspace=0.6)
    # plt.tight_layout()

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder,
                          f'{metric}_{shuffle_type}dropout_matched_clusters' + cre_line[:3]  )


def plot_unraveled_clusters(feature_matrix, cluster_df, sort_order=None, cre_line=None, save_dir=None, folder='', tag='',
                            ax=None, figsize=(4, 7), rename_columns=False):
    '''function to plot unraveled clusters (not means).
    INPUT:
    feature_matrix: (pd.DataFrame) dataframe of dropout scores with cell_specimen_id as index
    cluster_df: (pd.DataFrame) dataframe with cre_line, cell_specimen_id and cluster_id as columns
    sort_order: either dictionary with cre_line as keys (then provide cre_line) or list/np.array of sorted clusters
                default is None, then uses get_sorted_cluster_ids function to get clusters in descending size order
                from cluster_df
    cre_line: (str) if cluster_df and sort_order contain all cre_lines, you can select which one to plot, default=None
    save_dir: (str) if you wish to save figure, default=None
    save_dir: (str) if you wish to save figure, you can add a folder to the path, default=''
    tag: (str) when saving figure, this is a unique tag that will be added to figure name, default=''
    ax: (obj) if you want to provide axis to plot on, default=None
    figsize: (tuple), if you want to change figsize, default=(4,7)
    rename_columns: (boolean), change experience level columns from Novel 1 and Novel >1 to Novel and Novel+,
                default=False


    Returns:
        '''
    if ax is None:
        figsize = figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if sort_order is None:
        sort_order = processing.get_sorted_cluster_ids(cluster_df)

    if cre_line is not None:
        sort_order = sort_order[cre_line]
        cluster_df = cluster_df[cluster_df.cre_line == cre_line]

    # re name clusters after sorting by cluster size
    sorted_cluster_ids = {}
    for i, j in enumerate(sort_order):
        sorted_cluster_ids[j] = i + 1

    cluster_df = cluster_df.replace({'cluster_id': sorted_cluster_ids})

    if 'cell_specimen_id' in cluster_df.columns:
        cluster_df = cluster_df.set_index('cell_specimen_id')

    if rename_columns is True:
        feature_matrix = feature_matrix.rename(columns={'Novel 1': 'Novel', 'Novel >1': 'Novel+'}, level=1)

    cell_order = cluster_df.sort_values(by=['cluster_id']).index.values
    label_values = cluster_df.sort_values(by=['cluster_id']).cluster_id.values

    data = feature_matrix.loc[cell_order]
    ax = sns.heatmap(data.values, cmap='Blues', ax=ax, vmin=0, vmax=1,
                     robust=True, cbar_kws={"drawedges": False, "shrink": 0.7, "label": 'coding score'})

    cell_type = get_cell_type_for_cre_line(cre_line)
    ax.set_title(cell_type)
    ax.set_ylabel('cells')
    ax.set_ylim(0, data.shape[0])
    ax.set_yticks([0, data.shape[0]])
    ax.set_yticklabels((0, data.shape[0]), fontsize=14)
    ax.set_ylim(ax.get_ylim()[::-1])  # flip y axes so larger clusters are on top
    ax.set_xlabel('')
    ax.set_xlim(0, data.shape[1])
    ax.set_xticks(np.arange(0, data.shape[1]) + 0.5)
    ax.set_xticklabels([key[1] + ' -  ' + key[0] for key in list(data.keys())], rotation=90, fontsize=14)

    cluster_divisions = np.where(np.diff(label_values) == 1)[0]
    for y in cluster_divisions:
        ax.hlines(y, xmin=0, xmax=feature_matrix.shape[1], color='k')

    fig.subplots_adjust(wspace=0.5)
    if save_dir is not None:
        utils.save_figure(fig, figsize, save_dir, folder, f'feature_matrix_sorted_by_cluster_id_{tag}')
