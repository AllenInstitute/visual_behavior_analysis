import umap
import random
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
from visual_behavior.dimensionality_reduction.clustering.processing import get_silhouette_scores, get_cluster_density

import seaborn as sns
# figure settings
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})


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


def plot_silhouette_scores(X=None, model=KMeans, silhouette_scores=None, silhouette_std=None,
                           n_clusters=np.arange(2, 10), metric=None, n_boots=20, ax=None,
                           model_output_type=''):

    assert X is not None or silhouette_scores is not None, 'must provide either data to cluster or recomputed scores'
    assert X is None or silhouette_scores is None, 'cannot provide data to cluster and silhouette scores'

    if silhouette_scores is None:
        if metric is None:
            metric = 'euclidean'
        silhouette_scores, silhouette_std = get_silhouette_scores(X=X, model=model, n_clusters=n_clusters, metric=metric, n_boots=n_boots)
    elif silhouette_scores is not None:
        if len(silhouette_scores) != len(n_clusters):
            n_clusters = np.arange(1, len(silhouette_scores)+1)

        if metric is None:
            metric = ''

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.plot(n_clusters, silhouette_scores, 'ko-')
    if silhouette_std is not None:
        ax.errorbar(n_clusters, silhouette_scores, silhouette_std, color='k')
    ax.set_title('{}, {}'.format(model_output_type, metric), fontsize=16)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette score')
    plt.grid()
    plt.tight_layout()

    return ax


def plot_umap_with_labels(X, labels, ax=None, filename_string=''):
    fit = umap.UMAP()
    u = fit.fit_transform(X)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(u[:, 0], u[:, 1], c=labels)
    plt.tight_layout()
    ax.set_title(filename_string)
    return ax


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
