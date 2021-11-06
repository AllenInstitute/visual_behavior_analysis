from sklearn.metrics import silhouette_score
# from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import numpy as np
import pickle
import os
import visual_behavior.data_access.loading as loading

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
#cache_dir = loading.get_analysis_cache_dir()
#cache = bpc.from_s3_cache(cache_dir)

import visual_behavior_glm.GLM_analysis_tools as gat #to get recent glm results
import visual_behavior_glm.GLM_params as glm_params

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
    silhouette_scores = []
    for n_cluster in n_clusters:
        s_tmp = []
        for n_boot in range(0, n_boots):
            model.n_clusters=n_cluster
            md = model.fit(X)
            try:
                labels = md.labels_
            except AttributeError:
                labels = md
            s_tmp.append(silhouette_score(X, labels, metric=metric))
        silhouette_scores.append(np.mean(s_tmp))
        print('n {} clusters mean score = {}'.format(n_cluster, np.mean(s_tmp)))
    return silhouette_scores

def get_labels_for_coclust_matrix(X, model = SpectralClustering, nboot = np.arange(100), n_clusters = 8):
    '''

    :param X: (ndarray) data, n observations by n features
    :param model: (clustering object) default =  SpectralClustering; clustering method to use.
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

def get_coClust_matrix(X, model = SpectralClustering, nboot = np.arange(100), n_clusters = 8):
    '''

    :param X: (ndarray) data, n observations by n features
    :param model: (clustering object) default =  SpectralClustering; clustering method to use.
    :param nboot: (list or an array) default = 100, number of clustering repeats
    :param n_clusters: (num) default = 8
    ______________
    returns: coClust_matrix: (ndarray) probability matrix of co-clustering together.
    '''
    labels = get_labels_for_coclust_matrix(X = X,
                                           model = model,
                                           nboot = nboot,
                                           n_clusters = n_clusters)
    coClust_matrix = []
    for i in range(np.shape(labels)[1]): # get cluster id of this observation
        this_coClust_matrix = []
        for j in nboot: # find other observations with this cluster id
            id = labels[j][i]
            this_coClust_matrix.append(labels[j] == id)
        coClust_matrix.append(np.sum(this_coClust_matrix, axis=0) / max(nboot))
    return coClust_matrix


def clean_cells_table(cells_table=None, columns = None, add_binned_depth=True):

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


def save_clustering_results(data, filename_string ='', path = None):
    '''
    for HCP scripts to save output of spectral clustering in a specific folder
    :param data: what to save
    :param filename_string: name of the file, use as descriptive info as possible
    :return:
    '''
    if path is None:
        path = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/summary_plots/glm/SpectralClustering/files'
    filename = os.path.join(path, '{}'.format(filename_string))
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    f.close()


