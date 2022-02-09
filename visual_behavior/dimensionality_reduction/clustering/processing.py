from sklearn.metrics import silhouette_score
# from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd
import pickle
import os
import visual_behavior.data_access.loading as loading
from scipy.stats import spearmanr
from scipy.stats import kruskal
from scipy.stats import ttest_ind
from sklearn.metrics import pairwise_distances
# from scipy.stats import ttest_1samp

# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
# cache_dir = loading.get_analysis_cache_dir()
# cache = bpc.from_s3_cache(cache_dir)

# import visual_behavior_glm.GLM_analysis_tools as gat  # to get recent glm results
# import visual_behavior_glm.GLM_params as glm_params


def get_silhouette_scores(X, model=SpectralClustering, n_clusters=np.arange(2, 10), metric='euclidean', n_boots=20):
    '''
    Computes silhouette scores for given n clusters.
    :param X: data, n observations by n features
    :param model: default = SpectralClustering, but you can pass any clustering model object (some models might have unique
                    parameters, so this might cause issues in the future)
    :param n_clusters: an array or list of number of clusters to use.
    :param metric: default = 'euclidean', distance metric to use inner and outer cluster distance
                    (other options in scipy.spatial.distance.pdist)
    :param n_boots: default = 20, number of repeats to average over for each n cluster
    _____________
    :return: silhouette_scores: a list of scores for each n cluster
    '''
    print('size of X = ' + str(np.shape(X)))
    print('NaNs in the array = ' + str(np.sum(X == np.nan)))
    silhouette_scores = []
    silhouette_std = []
    for n_cluster in n_clusters:
        s_tmp = []
        for n_boot in range(0, n_boots):
            model.n_clusters = n_cluster
            md = model.fit(X)
            try:
                labels = md.labels_
            except AttributeError:
                labels = md
            s_tmp.append(silhouette_score(X, labels, metric=metric))
        silhouette_scores.append(np.mean(s_tmp))
        silhouette_std.append(np.std(s_tmp))
        print('n {} clusters mean score = {}'.format(n_cluster, np.mean(s_tmp)))
    return silhouette_scores, silhouette_std


def get_labels_for_coclust_matrix(X, model=SpectralClustering, nboot=np.arange(100), n_clusters=8):
    '''

    :param X: (ndarray) data, n observations by n features
    :param model: (clustering object) default =  SpectralClustering; clustering method to use. Object must be initialized.
    :param nboot: (list or an array) default = 100, number of clustering repeats
    :param n_clusters: (num) default = 8
    ___________
    :return: labels: matrix of labels, n repeats by n observations
    '''
    labels = []
    if n_clusters is not None:
        model.n_clusters = n_clusters
    for _ in nboot:
        md = model.fit(X)
        labels.append(md.labels_)
    return labels


def get_coClust_matrix(X, model=SpectralClustering, nboot=np.arange(150), n_clusters=8):
    '''

    :param X: (ndarray) data, n observations by n features
    :param model: (clustering object) default =  SpectralClustering; clustering method to use. Model must be initialized.
    :param nboot: (list or an array) default = 100, number of clustering repeats
    :param n_clusters: (num) default = 8
    ______________
    returns: coClust_matrix: (ndarray) probability matrix of co-clustering together.
    '''
    labels = get_labels_for_coclust_matrix(X=X,
                                           model=model,
                                           nboot=nboot,
                                           n_clusters=n_clusters)
    coClust_matrix = []
    for i in range(np.shape(labels)[1]):  # get cluster id of this observation
        this_coClust_matrix = []
        for j in nboot:  # find other observations with this cluster id
            id = labels[j][i]
            this_coClust_matrix.append(labels[j] == id)
        coClust_matrix.append(np.sum(this_coClust_matrix, axis=0) / max(nboot))
    return coClust_matrix


def clean_cells_table(cells_table=None, columns=None, add_binned_depth=True):
    '''
    Adds metadata to cells table using ophys_experiment_id. Removes NaNs and duplicates
    :param cells_table: vba loading.get_cell_table()
    :param columns: columns to add, default = ['cre_line', 'imaging_depth', 'targeted_structure']
    :return: cells_table with selected columns
    '''

    if cells_table is None:
        cells_table = loading.get_cell_table()

    if columns is None:
        columns = ['cre_line', 'imaging_depth', 'targeted_structure', 'binned_depth', 'cell_type']

    cells_table = cells_table[columns]

    # drop NaNs
    cells_table.dropna(inplace=True)

    # drop duplicated cells
    cells_table.drop_duplicates('cell_specimen_id', inplace=True)

    # set cell specimen ids as index
    cells_table.set_index('cell_specimen_id', inplace=True)

    if add_binned_depth is True and 'imaging_depth' in cells_table.keys():
        depths = [75, 175, 275, 375]
        cells_table['binned_depth'] = np.nan
        for i, imaging_depth in enumerate(cells_table['imaging_depth']):
            for depth in depths[::-1]:
                if imaging_depth < depth:
                    cells_table['binned_depth'].iloc[i] = depth

    # print number of cells
    print('N cells {}'.format(len(cells_table)))

    return cells_table


def save_clustering_results(data, filename_string='', path=None):
    '''
    for HCP scripts to save output of spectral clustering in a specific folder
    :param data: what to save
    :param filename_string: name of the file, use as descriptive info as possible
    :return:
    '''
    if path is None:
        path = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/summary_plots/glm/SpectralClustering/files'

    if os.path.exists(path) is False:
        os.mkdir(path)

    filename = os.path.join(path, '{}'.format(filename_string))
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def get_cre_line_cell_specimen_ids(df):
    cre_lines = df.cre_line.unique()
    cre_line_ids = {}
    for cre_line in cre_lines:
        ids = df[df.cre_line == cre_line]['cell_specimen_id'].unique()
        cre_line_ids[cre_line] = ids
    return cre_line_ids


def kruskal_by_experience_level(df_pivoted, posthoc=True):
    stats = {}
    f = df_pivoted['Familiar'].values
    n = df_pivoted['Novel 1'].values
    nn = df_pivoted['Novel >1'].values

    k, p = kruskal(f, n, nn)
    stats['KW'] = (k, p)
    if posthoc:
        t, p = ttest_ind(f, n, nan_policy='omit')
        stats['Familiar_vs_Novel'] = (t, p)
        t, p = ttest_ind(f, nn, nan_policy='omit')
        stats['Familiar_vs_Novel>1'] = (t, p)
        t, p = ttest_ind(n, nn, nan_policy='omit')
        stats['Novel_vs_Novel>1'] = (t, p)

    return stats


def pivot_df(df, dropna=True, drop_duplicated_cells=True):
    df_pivoted = df.groupby(['cell_specimen_id', 'experience_level']).mean()
    if dropna is True:
        df_pivoted = df_pivoted.unstack().dropna()
    else:
        df_pivoted = df_pivoted.unstack()

    if drop_duplicated_cells is True:
        if len(df) == len(np.unique(df.index.values)):
            print('No duplicated cells found')
        elif len(df) > len(np.unique(df.index.values)):
            print('found {} duplicated cells. But not removed. This needs to be fixed'.format(len(df) - len(np.unique(df.index.values))))
        elif len(df) < len(np.unique(df.index.values)):
            print('something weird happened!!')

    return df_pivoted


def build_stats_table(metrics_df, metrics_columns=None, dropna=True, pivot=False):
    # check for cre lines
    if 'cre_line' in metrics_df.keys():
        cre_lines = metrics_df['cre_line'].unique()
        cre_line_ids = get_cre_line_cell_specimen_ids(metrics_df)
    else:
        cre_lines = ['all']
        cre_line_ids = metrics_df['cell_specimen_id'].unique()

    # get selected columns
    if metrics_columns is None:
        metrics_columns = ['image_selectivity_index', 'image_selectivity_index_one_vs_all',
                           'lifetime_sparseness', 'fraction_significant_p_value_gray_screen',
                           'fano_factor', 'reliability', 'running_modulation_index']
        if 'hit_miss_index' in metrics_df.keys():
            metrics_columns = [*metrics_columns, 'hit_miss_index']

    # check which columns are in the dataframe
    metrics_columns_corrected = []
    for metric in metrics_columns:
        if metric in metrics_df.keys():
            metrics_columns_corrected.append(metric)

    stats_table = pd.DataFrame(columns=['cre_line', 'comparison', 'statistic', 'metric', 'data'])
    statistics = ('t', 'pvalue')
    for c, cre_line in enumerate(cre_lines):
        # dummy table
        if cre_line == 'all':
            tmp_cre = metrics_df
        else:
            tmp_cre = metrics_df[metrics_df['cell_specimen_id'].isin(cre_line_ids[cre_line])]

        # group df by cell id and experience level
        metrics_df_pivoted = pivot_df(tmp_cre, dropna=dropna)
        for m, metric in enumerate(metrics_columns_corrected):
            stats = kruskal_by_experience_level(metrics_df_pivoted[metric])
            for i, stat in enumerate(statistics):
                for key in stats.keys():
                    data = {'cre_line': cre_line,
                            'comparison': key,
                            'statistic': stat,
                            'metric': metric,
                            'data': stats[key][i], }
                    tmp_table = pd.DataFrame(data, index=[0])
                    stats_table = stats_table.append(tmp_table, ignore_index=True)

    if pivot and stats_table['data'].sum() != 0:
        stats_table = pd.pivot_table(stats_table, columns=['metric'],
                                     index=['cre_line', 'comparison', 'statistic']).unstack()
    elif stats_table['data'].sum() == 0:
        print('Cannot pivot table when all data is NaNs')
    return stats_table


def get_cluster_density(df_dropouts, labels_list, use_spearmanr=False):
    '''
    Computes correlation coefficients for clusters computed on glm dropouts
    '''
    labels = np.unique(labels_list)
    cluster_corrs = {}
    for label in labels:
        cluster_means = df_dropouts[labels_list == label].mean().abs().values
        within_cluster = df_dropouts[labels_list == label].abs()
        corr_coeffs = []
        for cid in within_cluster.index.values:
            if use_spearmanr is False:
                corr_coeffs.append(np.corrcoef(cluster_means, within_cluster.loc[cid].values)[0][1])
            elif use_spearmanr is True:
                corr_coeffs.append(spearmanr(cluster_means, within_cluster.loc[cid].values)[0])
        cluster_corrs[label] = corr_coeffs
    return cluster_corrs


def shuffle_dropout_score(df_dropout, shuffle_type='all'):
    '''
    Shuffles dataframe with dropout scores from GLM.
    shuffle_type: str, default='all', other options= 'experience', 'regressors'
    '''
    df_shuffled = df_dropout.copy()
    regressors = df_dropout.columns.levels[0].values
    experience_levels = df_dropout.columns.levels[1].values
    if shuffle_type == 'all':
        for column in df_dropout.columns:
            df_shuffled[column] = df_dropout[column].sample(frac=1).values

    elif shuffle_type == 'experience':
        print('shuffling data across experience')
        assert np.shape(df_dropout.columns.levels)[
            0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
        for experience_level in experience_levels:
            randomized_cids = df_dropout.sample(frac=1).index.values
            for i, cid in enumerate(randomized_cids):
                for regressor in regressors:
                    df_shuffled.iloc[i][regressor][experience_level] = df_dropout.loc[cid][regressor][experience_level]

    elif shuffle_type == 'regressors':
        print('shuffling data across regressors')
        assert np.shape(df_dropout.columns.levels)[
            0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
        for regressor in regressors:
            randomized_cids = df_dropout.sample(frac=1).index.values
            for i, cid in enumerate(randomized_cids):
                for experience_level in experience_levels:
                    df_shuffled.iloc[i][regressor][experience_level] = df_dropout.loc[cid][regressor][experience_level]
    return df_shuffled


def compute_inertia(a, X, metric = 'euclidean'):
    W = [np.mean(pairwise_distances(X[a == c, :], metric=metric)) for c in np.unique(a)]
    return np.mean(W)


def compute_gap(clustering, data, k_max=5, n_references=20, reference = None, metric='euclidean'):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    if reference is None:
        reference = np.random.rand(*data.shape)*-1
    reference_inertia = []
    for k in range(1, k_max ):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference, metric=metric))
        reference_inertia.append(np.mean(local_inertia))

    ondata_inertia = []
    for k in range(1, k_max ):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(data)
            local_inertia.append(compute_inertia(assignments, data, metric=metric))
        ondata_inertia.append(np.mean(local_inertia))

    gap = np.log(reference_inertia) - np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)
