import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from visual_behavior.dimensionality_reduction.clustering.processing import get_silhouette_scores
import umap
import os
from scipy import signal
import visual_behavior.visualization.utils as utils
import random

# figure settings
plt.rcParams['font.size'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16


def plot_clustered_dropout_scores(df, labels=None, cmap='RdBu_r', model_output_type='', ax=None,):
    sort_order = np.argsort(labels)
    sorted_labels = labels[sort_order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    ax = sns.heatmap(df.abs().values[sort_order], cmap='Blues', ax=ax, vmin=0, vmax=1,
                     robust=True,
                     cbar_kws={"drawedges": False, "shrink": 0.8, "label": 'fraction change in explained variance'})
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
        fig, ax = plt.figure(2, 1, figsize=(8, 12))

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


def plot_silhouette_scores(X=None, model=KMeans, silhouette_scores=None, n_clusters=np.arange(2, 10), metric=None, n_boots=20, ax=None,
                           model_output_type=''):
    assert X is not None or silhouette_scores is not None, 'must provide either data to cluster or recomputed scores'
    assert X is None or silhouette_scores is None, 'cannot provide data to cluster and silhouette scores'

    if silhouette_scores is None:
        if metric is None:
            metric = 'euclidean'
        silhouette_scores = get_silhouette_scores(X=X, model=model, n_clusters=n_clusters, metric=metric, n_boots=n_boots)
    elif silhouette_scores is not None:
        if len(silhouette_scores) != len(n_clusters):
            n_clusters = np.arange(0, len(silhouette_scores))

        if metric is None:
            metric = ''

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.plot(n_clusters, silhouette_scores, 'ko-')
    ax.set_title('{}, {}'.format(model_output_type, metric), fontsize=16)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette score')
    plt.grid()
    plt.tight_layout()

    return ax


def plot_umap_with_labels(X, labels, save_plot=False, path=None, ax=None, filename_string=''):
    fit = umap.UMAP()
    u = fit.fit_transform(X)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(u[:, 0], u[:, 1], c=labels)

    if save_plot is True:
        filename = 'UMAP_{}.png'.fomrat(filename_string)
        fig.savefig(os.path.join(path, filename))
    plt.tight_layout()
    ax.set_title(filename_string)
    return fig, ax


def plot_clusters(dropout_df, cluster_df=None, mean_response_df=None, save_plots=False, path=None):
    '''
    Plots heatmaps and descriptors of clusters.
    dropout_df: dataframe of dropout scores, n cells by n regressors by experience level
    cluster_df: df with cluster id and cell specimen id columns
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
    grouped_df = cluster_df.groupby('cluster_id')
    N_mice = grouped_df.agg({"mouse_id": "nunique"})

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
        ax[i] = sns.heatmap(mean_dropout_df,
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
        ax[i].set_title('cluster ' + str(int(cluster_id)) + '\n' + str(fraction) + '%, n=' + str(n_cells)
                        + '\n' + 'n mice = ' + str(N_mice.loc[cluster_id].values),
                        fontsize=16)
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
                                                     # orient='h',
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
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / len(unique_labels)
    for i in unique_labels:
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        color_map[i] = (r, g, b)
        #color_map[i] = '#%06X' % randint(0, 0xFFFFFF)
    return color_map

def plot_N_clusters_by_cre_line(labels_cre, ax=None, palette=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for i, cre_line in enumerate(labels_cre.keys()):
        labels = labels_cre[cre_line]
        (unique, counts) = np.unique(labels, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        frequencies = sorted_counts / sum(counts) * 100
        ax.grid()
        ax.plot(range(0, len(frequencies)), frequencies, 'o-', color=palette[i], LineWidth=4, MarkerSize=10)
        ax.set_xlabel('Cluster number')
        ax.set_ylabel('Number of cells per cluster (%)')
    ax.legend(labels_cre.keys())

    return ax
