
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import linalg as linalg

from scipy.stats import spearmanr
from scipy.stats import kruskal
from scipy.stats import ttest_ind
from scipy.sparse import csgraph

from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities

import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_params as glm_params
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import multinomial_proportions_confint

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})


# from scipy.stats import ttest_1samp

# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
# cache_dir = loading.get_analysis_cache_dir()
# cache = bpc.from_s3_cache(cache_dir)

# import visual_behavior_glm.GLM_analysis_tools as gat  # to get recent glm results
# import visual_behavior_glm.GLM_params as glm_params


# utilities ###


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


def get_features_for_clustering():
    """
    get GLM features to use for clustering analysis
    """
    features = ['all-images', 'omissions', 'behavioral', 'task']
    return features


def get_cre_lines(cell_metadata):
    """
    get list of cre lines in cell_metadata and sort in reverse order so that Vip is first
    cell_metadata is a table similar to ophys_cells_table from SDK but can have additional columns based on clustering results
    """
    cre_lines = np.sort(cell_metadata.cre_line.unique())[::-1]
    return cre_lines


def get_cell_type_for_cre_line(cell_metadata, cre_line):
    """
    gets cell_type label for a given cre_line using info in cell_metadata table
    cell_metadata is a table similar to ophys_cells_table from SDK but can have additional columns based on clustering results
    """
    cell_type = cell_metadata[cell_metadata.cre_line == cre_line].cell_type.values[0]
    return cell_type


def get_manual_sort_order():
    """
    manually selected cluster order for each cre line
    based on whatever Marina thinks would be useful to sort by
    """
    manual_sort_order = {'Vip-IRES-Cre': [6, 9, 2, 7, 3, 10, 4, 5, 8, 12, 11, 1],
                         'Sst-IRES-Cre': [1, 2, 3, 5, 4, 6],
                         'Slc17a7-IRES2-Cre': [2, 9, 7, 4, 3, 8, 6, 5, 1, 10]}
    return manual_sort_order


def get_manual_sort_order_for_cre_line(cre_line):
    """
    gets order to sort clusters by for a given cre line based on Marina's manual sorting
    """
    manual_sort_order = get_manual_sort_order()
    return manual_sort_order[cre_line]


def add_manual_sort_order_to_cluster_meta(cluster_meta):
    """
    Adds a new cluster_id label called 'manual_sort_order' to cluster_meta (or any df with 'cluster_id')
    manually sorted order is hard coded
    """
    cluster_meta['manual_sort_order'] = None
    for cell_specimen_id in cluster_meta.index.values:
        cre_line = cluster_meta.loc[cell_specimen_id].cre_line
        cluster_id = cluster_meta.loc[cell_specimen_id].cluster_id
        manual_sort = get_manual_sort_order_for_cre_line(cre_line)
        # new ID is the position of this cluster ID in the manually sorted list
        # print(cluster_id, manual_sort)
        manual_cluster_id = np.where(manual_sort == cluster_id)[0][0]
        cluster_meta.at[cell_specimen_id, 'manual_sort_order'] = manual_cluster_id
    return cluster_meta


def get_cre_line_cell_specimen_ids(df_no_cre, df_cre):
    '''
    This function is used to assign correct cre line to cell specimen ids.
    First df (glm_pivoted) is
    Input:
    df_no_cre: pd.DataFrame with cell_specimen_id as index or column but no cre line
    df_cre: pd.DataFrame with columns ['cre_line', 'cell_specimen_id']

    Returns:
        cre_line_ids: dictionary with cre lines and their assossiated cell specimen ids
    '''
    if df_no_cre.index.name == 'cell_specimen_id':
        ids = df_no_cre.index.values
    else:
        ids = df_no_cre['cell_spedimen_id'].values

    df = df_cre[df_cre['cell_specimen_id'].isin(ids)][['cell_specimen_id', 'cre_line']].drop_duplicates(
        'cell_specimen_id')
    cre_lines = df.cre_line.unique()
    cre_line_ids = {}
    for cre_line in cre_lines:
        ids = df[df.cre_line == cre_line]['cell_specimen_id'].unique()
        cre_line_ids[cre_line] = ids
    return cre_line_ids


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


def clean_cluster_meta(cluster_meta):
    """
    drop cluster IDs with fewer than 5 cells in them
    """
    # get clusters with <5 cells
    tmp = cluster_meta.groupby(['cre_line', 'cluster_id']).count()
    locs_to_drop = tmp[tmp.labels < 5].index
    # remove them from cluster_meta
    inds_to_drop = []
    for loc_to_drop in locs_to_drop:
        inds_to_drop_this_loc = list(cluster_meta[(cluster_meta.cre_line == loc_to_drop[0]) & (cluster_meta.cluster_id == loc_to_drop[1])].index.values)
        inds_to_drop = inds_to_drop + inds_to_drop_this_loc
        print('dropping', len(inds_to_drop_this_loc), 'cells for', loc_to_drop)
    cluster_meta = cluster_meta.drop(index=inds_to_drop)
    print(len(inds_to_drop), 'cells dropped total')
    return cluster_meta


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


# loading & processing ###

def get_glm_results_pivoted_for_clustering(glm_version='24_events_all_L2_optimize_by_session',
                                           model_output_type='adj_fraction_change_from_full', save_dir=None):
    """
    loads GLM results pivoted where columns are dropout scores and rows are cells in specific experiments
    filters to limit to cells matched in all experience levels
    loads from file if file exists in save_dir, otherwise generates from mongo and saves to save dir
    """

    results_file_path = os.path.join(save_dir, glm_version + '_results_pivoted.h5')
    if os.path.exists(results_file_path):
        print('loading results_pivoted from', results_file_path)
        results_pivoted = pd.read_hdf(results_file_path, key='df')

    else:  # if it doesnt exist, then load it and save (note this is slow)
        results_pivoted = gat.build_pivoted_results_summary(value_to_use=model_output_type, results_summary=None,
                                                            glm_version=glm_version, cutoff=None)

        # get rid of passive sessions
        results_pivoted = results_pivoted[results_pivoted.passive == False]

        # get cells table and limit to cells matched in closest familiar and novel active
        cells_table = loading.get_cell_table()
        cells_table = utilities.limit_to_last_familiar_second_novel_active(cells_table)
        cells_table = utilities.limit_to_containers_with_all_experience_levels(cells_table)
        cells_table = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table)
        print(len(cells_table.cell_specimen_id.unique()),
              'cells in cells_table after limiting to strictly matched cells')

        # get matched cells and experiments to limit to
        matched_experiments = cells_table.ophys_experiment_id.unique()
        matched_cells = cells_table.cell_specimen_id.unique()

        # limit results pivoted to to last familiar and second novel
        results_pivoted = results_pivoted[results_pivoted.ophys_experiment_id.isin(matched_experiments)]
        results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(matched_cells)]
        print(len(results_pivoted.cell_specimen_id.unique()),
              'cells in results_pivoted after limiting to strictly matched cells')

        if save_dir:
            # save filtered results to save_dir
            results_pivoted.to_hdf(os.path.join(save_dir, glm_version + '_results_pivoted.h5'), key='df')
            print('results_pivoted saved')
    return results_pivoted


def limit_results_pivoted_to_features_for_clustering(results_pivoted, features=None, use_signed_features=False):
    """
    takes matrix of results_pivoted (dropout scores (values) for all cells in each experiment (rows)
    by GLM features (columns) and limits to features to use for clustering,
    then takes absolute value of dropout scores
    features_for_clustering is typically ['all-images', 'omissions', 'behavioral', 'task']
    """

    # glm features to use for clustering
    if features is None:
        features = get_features_for_clustering()
    if use_signed_features:
        features = [feature + '_signed' for feature in features]
    columns = [*features, 'cell_specimen_id', 'experience_level']
    results_pivoted = results_pivoted[columns]

    return results_pivoted


def flip_sign_of_dropouts(results_pivoted, features_to_invert, use_signed_weights=False):
    """
    for features_to_invert, either take the absolute value of the dropout scores so that a reduction in explained variance
    for a given feature is represented as a positive coding score, or invert the sign of the dropouts if using signed weights,
    so features with positive kernel weights have positive coding scores and feature with negative weights have negative coding scores

    :param results_pivoted: table of dropout scores for each cell and experience level
    :param features_to_invert: regressors to apply processing to
    :param use_signed_features: Bool, whether or not sign of weights has been applied to dropouts
    :return:
    """

    if use_signed_weights:
        # invert the sign so that dropout sign reflects sign of weights
        for feature in features_to_invert:
            results_pivoted[feature] = results_pivoted[feature] * -1
    else:
        # make everything positive
        for feature in features_to_invert:
            results_pivoted[feature] = np.abs(results_pivoted[feature])

    return results_pivoted


def generate_GLM_outputs(glm_version, experiments_table, cells_table, glm_output_dir=None, use_signed_weights=False,
                         across_session_norm=False, include_var_exp_full=False):
    """
    generates results_pivoted and weights_df and saves to glm_output_dir
    results_pivoted and weights_df will be limited to the ophys_experiment_ids and cell_specimen_ids present in experiments_table and cells_table
    optional inputs to use across session normalization or signed weights

    :param glm_version: example = '24_events_all_L2_optimize_by_session'
    :param glm_output_dir: directory to save processed data files to for this iteration of the analysis
    :param experiments_table: SDK ophys_experiment table limited to experiments intended for analysis
    :param cells_table: SDK ophys_cell_table limited to cell_specimen_ids intended for analysis
    :param use_signed_weights: whether or not apply the sign of the kernel weights to the dropout scores in results_pivoted
    :param across_session_norm: whether or not to normalize dropout scores by the max overall explained variance across sessions
    :param include_var_exp_full: whether or not to include 'variance_explained_full' column in results_pivoted

    :return:
        results_pivoted: table of dropout scores for all cell_specimen_ids in cells_table, limited to the features used for clustering
                            will output across session normalized or signed dropouts depending on input params
        weights_df: table with model weights for all cell_specimen_ids in cells_table
    """

    if across_session_norm:

        import visual_behavior_glm.GLM_across_session as gas

        # get across session normalized dropout scores
        df, failed_cells = gas.load_cells(cells='all')
        df = df.set_index('identifier')

        # only use across session values
        # within = df[[key for key in df.keys() if '_within' in key] + ['cell_specimen_id', 'ophys_experiment_id',
        #                                                               'experience_level']]
        across = df[[key for key in df.keys() if '_across' in key] + ['cell_specimen_id', 'ophys_experiment_id',
                                                                      'experience_level']]
        results_pivoted = across.copy()
        results_pivoted = results_pivoted.rename(
            columns={'omissions_across': 'omissions', 'all-images_across': 'all-images',
                     'behavioral_across': 'behavioral', 'task_across': 'task'})
        print(len(results_pivoted), 'len(results_pivoted)')

        # limit results pivoted to to last familiar and second novel active
        # requires that cells_table provided as input to this function is already limited to cells of interest
        matched_cells = cells_table.cell_specimen_id.unique()
        matched_experiments = cells_table.ophys_experiment_id.unique()
        results_pivoted = results_pivoted[results_pivoted.ophys_experiment_id.isin(matched_experiments)]
        results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(matched_cells)]
        print(len(results_pivoted.cell_specimen_id.unique()),
              'cells in results_pivoted after limiting to strictly matched cells')

        # drop duplicates
        results_pivoted = results_pivoted.drop_duplicates(subset=['cell_specimen_id', 'experience_level'])
        print(len(results_pivoted), 'len(results_pivoted) after dropping duplicates')

    else:
        # get GLM results from saved file in save_dir or from mongo if file doesnt exist
        # this function limits results_pivoted to matched cells in last familiar second novel active
        model_output_type = 'adj_fraction_change_from_full'
        results_pivoted = get_glm_results_pivoted_for_clustering(glm_version, model_output_type, glm_output_dir)

    # get weights df and save
    run_params = glm_params.load_run_json(glm_version)
    weights_df = gat.build_weights_df(run_params, results_pivoted)
    weights_df.to_hdf(os.path.join(glm_output_dir, glm_version + '_weights_df.h5'), key='df')

    # apply sign of kernel weights to dropouts if desired
    if use_signed_weights:

        # modify regressors to include kernel weights as regressors with '_signed' appended
        results_pivoted = gat.append_kernel_excitation(weights_df, results_pivoted)
        # add behavioral results signed by duplicating behavioral dropouts
        results_pivoted['behavioral_signed'] = results_pivoted.behavioral.values

        # remove redundant unsigned columns
        features = get_features_for_clustering()
        results_pivoted = results_pivoted.drop(columns=features)

        # now remove 'signed' from column names so that the remaining features in are signed but with the regular column name
        columns = {}
        for column in results_pivoted.columns:
            if 'signed' in column:
                # invert the sign so that "increased coding" is positive
                results_pivoted[column] = results_pivoted[column] * -1
                # get new column name without the '_signed'
                columns[column] = column.split('_')[0]
        results_pivoted = results_pivoted.rename(columns=columns)

    # limit dropout scores to the features we want to cluster on
    features = get_features_for_clustering()
    features = [*features, 'ophys_experiment_id']
    if include_var_exp_full:
        features = [*features, 'variance_explained_full']
    results_pivoted = limit_results_pivoted_to_features_for_clustering(results_pivoted, features)

    # take absolute value or flip sign of dropouts so that reduction in explained variance is a positive coding score
    features_to_invert = get_features_for_clustering()  # only invert dropout scores, skip var_exp_full if using
    results_pivoted = flip_sign_of_dropouts(results_pivoted, features_to_invert, use_signed_weights)

    # save processed results_pivoted
    results_pivoted.to_hdf(os.path.join(glm_output_dir, glm_version + '_results_pivoted.h5'), key='df')

    # now drop ophys_experiment_id
    results_pivoted = results_pivoted.drop(columns=['ophys_experiment_id'])

    # get kernels for this version
    run_params = glm_params.load_run_json(glm_version)
    kernels = run_params['kernels']

    print(results_pivoted.keys())

    return results_pivoted, weights_df, kernels


def load_GLM_outputs(glm_version, glm_output_dir):
    """
    loads results_pivoted and weights_df from files in base_dir, or generates them from mongo and save to base_dir
    results_pivoted and weights_df will be limited to the ophys_experiment_ids and cell_specimen_ids present in experiments_table and cells_table
    because this function simply loads the results, any pre-processing applied to results_pivoted (such as across session normalization or signed weights) will be used here

    :param glm_version: example = '24_events_all_L2_optimize_by_session'
    :param glm_output_dir: directory containing GLM output files to load
    :return:
        results_pivoted: table of dropout scores for all cell_specimen_ids in cells_table
        weights_df: table with model weights for all cell_specimen_ids in cells_table
        kernels: dict of kernel params for this model version
    """
    # get GLM kernels and params for this version of the model
    # model_output_type = 'adj_fraction_change_from_full'
    run_params = glm_params.load_run_json(glm_version)
    kernels = run_params['kernels']
    # if glm_output_dir is not None:
    # load GLM results for all cells and sessions from file if it exists otherwise load from mongo
    glm_results_path = os.path.join(glm_output_dir, glm_version + '_results_pivoted.h5')
    if os.path.exists(glm_results_path):
        results_pivoted = pd.read_hdf(glm_results_path, key='df')
    else:
        print('no results_pivoted at', glm_results_path)
        print('please generate before using load_glm_outputs')

    weights_path = os.path.join(glm_output_dir, glm_version + '_weights_df.h5')
    if os.path.exists(weights_path):  # if it exists, load it
        weights_df = pd.read_hdf(weights_path, key='df')
    else:
        print('no weights at', weights_path)
        print('please generate before using load_glm_outputs')

    return results_pivoted, weights_df, kernels


def get_feature_matrix_for_clustering(dropouts, glm_version=None, save_dir=None):
    """
    takes matrix of dropout scores (values) for all cells in each experiment (rows)
    by GLM features (columns) and:
        limits to features to use for clustering
        takes absolute value of dropout scores
        pivots to get multi-index df of features per experience level for each cell (feature_matrix)
    """

    # pivot to get one column per feature per experience level
    feature_matrix = pivot_df(dropouts, dropna=True)
    print(len(feature_matrix))

    # save it
    if save_dir:
        feature_matrix.to_hdf(os.path.join(save_dir, glm_version + '_feature_matrix.h5'), key='df')
    return feature_matrix


def get_cell_metadata_for_feature_matrix(feature_matrix, cells_table):
    """
    get a dataframe of cell metadata for all cells in feature_matrix
    limits ophys_cells_table to cell_specimen_ids present in feature_matrix,
    drops duplicates and sets cell_specimen_id as index
    returns dataframe with cells as indices and metadata as columns
    """
    # get metadata for cells in matched cells df
    cell_metadata = cells_table[cells_table.cell_specimen_id.isin(feature_matrix.index.values)]
    cell_metadata = cell_metadata.drop_duplicates(subset='cell_specimen_id')
    cell_metadata = cell_metadata.set_index('cell_specimen_id')
    print(len(cell_metadata), 'cells in cell_metadata for feature_matrix')
    return cell_metadata


def get_feature_matrix_for_cre_line(feature_matrix, cell_metadata, cre_line):
    """
    limit feature_matrix to cell_specimen_ids for this cre line
    """
    cre_cell_specimen_ids = cell_metadata[cell_metadata['cre_line'] == cre_line].index.values
    feature_matrix_cre = feature_matrix.loc[cre_cell_specimen_ids].copy()
    return feature_matrix_cre


def compute_inertia(a, X, metric='euclidean'):
    W = [np.mean(pairwise_distances(X[a == c, :], metric=metric)) for c in np.unique(a)]
    return np.mean(W)


def compute_gap(clustering, data, k_max=5, n_boots=20, reference=None, metric='euclidean'):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    if reference is None:
        reference = np.random.rand(*data.shape) * -1
    reference_inertia = []
    for k in range(1, k_max ):
        local_inertia = []
        for _ in range(n_boots):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference, metric=metric))
        reference_inertia.append(np.mean(local_inertia))

    ondata_inertia = []
    for k in range(1, k_max ):
        local_inertia = []
        for _ in range(n_boots):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(data)
            local_inertia.append(compute_inertia(assignments, data, metric=metric))
        ondata_inertia.append(np.mean(local_inertia))

    gap = np.log(reference_inertia) - np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)


def get_eigenDecomposition(A, max_n_clusters=25):
    """
    Input:
    A: Affinity matrix from spectral clustering
    max_n_clusters

    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized laplacian matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    # n_components = A.shape[0]
    eigenvalues, eigenvectors = linalg.eigh(L)

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:max_n_clusters]
    nb_clusters = index_largest_gap + 1

    return eigenvalues, eigenvectors, nb_clusters


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


def load_silhouette_scores(glm_version, feature_matrix, cell_metadata, save_dir, n_boots=20, n_clusters=np.arange(2, 30)):
    """
    if silhouette scores file exists in save_dir, load it
    otherwise run spectral clustering n_boots times, for a range of n_clusters
    returns dictionary of silhouette scores for each cre line
    n_boots is the number of times to run clustering for each n_cluster to get variability across runs
    n_clusters is the range of K values for which to compute the silhouette score
    """

    sil_filename = glm_version + '_silhouette_scores.pkl'
    sil_path = os.path.join(save_dir, sil_filename)
    if os.path.exists(sil_path):
        print('loading silhouette scores from', sil_path)
        with open(sil_path, 'rb') as f:
            silhouette_scores = pickle.load(f)
            f.close()
        print('done.')
    else:
        # generate silhouette scores by running spectral clustering n_boots times for each n_clusters
        from sklearn.cluster import SpectralClustering
        silhouette_scores = {}
        for cre_line in get_cre_lines(cell_metadata):
            feature_matrix_cre = get_feature_matrix_for_cre_line(feature_matrix, cell_metadata, cre_line)
            X = feature_matrix_cre.values
            sc = SpectralClustering()
            silhouette_scores_cre, silhouette_std_cre = get_silhouette_scores(X,
                                                                              model=sc,
                                                                              n_clusters=n_clusters,
                                                                              n_boots=n_boots,
                                                                              metric='euclidean')
            silhouette_scores[cre_line] = [silhouette_scores_cre, silhouette_std_cre]
        # save it for next time
        save_clustering_results(silhouette_scores, filename_string=sil_filename, path=save_dir)
    return silhouette_scores


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
    for _ in tqdm(nboot):
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


def get_coclustering_matrix(glm_version, feature_matrix, cell_metadata, n_clusters_cre, save_dir, nboot=100):
    """
    if coclustering matrix file exists in save_dir, load from file
    else, create coclustering matrix by performing spectral clustering nboot number of times and save the matrix

    n_clusters_cre is a dictionary with cre lines as keys and the selected number of clusters for that cre line as values
    returns coclustering_matrices: a dictionary with cre_lines as keys, values as co-clustering matrix per cre line
    """
    coclust_filename = glm_version + '_coClustering_matrix.pkl'
    coclust_path = os.path.join(save_dir, coclust_filename)
    if os.path.exists(coclust_path):
        print('loading file...')
        with open(coclust_path, 'rb') as f:
            coclustering_matrices = pickle.load(f)
            f.close()
            print('done.')
    else:
        coclustering_matrices = {}
        for i, cre_line in enumerate(get_cre_lines(cell_metadata)):
            sc = SpectralClustering()
            feature_matrix_cre = get_feature_matrix_for_cre_line(feature_matrix, cell_metadata, cre_line)
            X = feature_matrix_cre.values
            m = get_coClust_matrix(X=X, n_clusters=n_clusters_cre[cre_line], model=sc, nboot=np.arange(nboot))
            # make co-clustering matrix a dataframe with cell_specimen_ids as indices and columns
            coclustering_df = pd.DataFrame(data=m, index=feature_matrix_cre.index, columns=feature_matrix_cre.index)
            # save to dictionary
            coclustering_matrices[cre_line] = coclustering_df
        save_clustering_results(coclustering_matrices, filename_string=coclust_filename, path=save_dir)
    return coclustering_matrices


def get_cluster_label_file_name(cre_lines, n_clusters_cre, prefix='cluster_labels'):
    """
    create file name with each cre_line and the number of clusters used for that cre line
    """
    cluster_file_name = ''
    # iterate cre lines & build filename
    for i, cre_line in enumerate(cre_lines):
        cluster_file_name = cluster_file_name + '_' + cre_line.split('-')[0] + '_' + str(n_clusters_cre[cre_line])
    cluster_file_name = prefix + cluster_file_name + '.h5'
    return cluster_file_name


def get_cluster_labels(coclustering_matrices, cell_metadata, n_clusters_cre, save_dir=None, load=True):
    """
    if cluster_labels file exists in save_dir, load it, otherwise
    perform agglomerative clustering on co-clustering matrices to get cluster labels based on affinity matrix,
    then determine 'cluster_id' based on the size of each cluster such that cluster_id is in increasing size order
    then save output to save_dir

    coclustering_matrices: a dictionary where keys are cre_lines and values are
                            dataframes of coclustering probabilities for each pair of cells in that cre line
    n_clusters_cre: a dictionary with cre_lines as keys and n_clusters to use for clustering as values
                        n_clusters_cre is selected based on silhouette scores
    returns a dataframe with cluster labels and IDs for each cell_specimen_id in each cre line
    """
    cre_lines = get_cre_lines(cell_metadata)
    cluster_file_name = get_cluster_label_file_name(cre_lines, n_clusters_cre)
    cluster_file_path = os.path.join(save_dir, cluster_file_name)
    if load:
        if os.path.exists(cluster_file_path):
            print('loading cluster labels from', cluster_file_path)
            cluster_labels = pd.read_hdf(cluster_file_path, key='df')
        else:
            get_labels = True
    else:
        get_labels = True
    if get_labels:
        print('generating cluster labels from coclustering matrix')
        from sklearn.cluster import AgglomerativeClustering
        cluster_labels = pd.DataFrame()
        for i, cre_line in enumerate(cre_lines):
            cell_specimen_ids = coclustering_matrices[cre_line].index
            X = coclustering_matrices[cre_line].values
            cluster = AgglomerativeClustering(n_clusters=n_clusters_cre[cre_line], affinity='euclidean',
                                              linkage='average')
            labels = cluster.fit_predict(X)
            # make dictionary with labels for each cell specimen ID in this cre line
            labels_dict = {'labels': labels, 'cell_specimen_id': cell_specimen_ids,
                           'cre_line': [cre_line] * len(cell_specimen_ids)}
            # turn it into a dataframe
            labels_df = pd.DataFrame(data=labels_dict, columns=['labels', 'cell_specimen_id', 'cre_line'])
            # get new cluster_ids based on size of clusters and add to labels_df
            cluster_size_order = labels_df['labels'].value_counts().index.values
            # translate between original labels and new IDS based on cluster size
            labels_df['cluster_id'] = [np.where(cluster_size_order == label)[0][0] for label in labels_df.labels.values]
            # concatenate with df for all cre lines
            cluster_labels = pd.concat([cluster_labels, labels_df])
        # add 1 to cluster labels so they start at 1 instead of 0
        cluster_labels['cluster_id'] = [cluster_id + 1 for cluster_id in cluster_labels.cluster_id.values]
        # save it
        if save_dir:
            print('saving cluster_labels to', cluster_file_path)
            cluster_labels.to_hdf(cluster_file_path, key='df')
    return cluster_labels


def get_cluster_density(df_dropouts, labels_df, label_col='cluster_id', use_spearmanr=False):
    '''
    Computes correlation coefficients for clusters computed on glm dropouts
    df_dropouts: dataframe where rows are unique cell_specimen_ids and columns are dropout scores across experience levels
    labels_df: dataframe with cell_specimen_ids as index and a cluster label as a column
    labels_col: column name for cluster labels to use (ex: 'labels', 'cluster_id')
    '''
    labels = labels_df[label_col].unique()
    cluster_corrs = {}
    for label in labels:
        cluster_csids = labels_df[labels_df[label_col] == label].index.values
        cluster_dropouts = df_dropouts.loc[cluster_csids]
        cluster_means = cluster_dropouts.mean().abs().values
        within_cluster = cluster_dropouts.abs()
        corr_coeffs = []
        for cid in within_cluster.index.values:
            if use_spearmanr is False:
                corr_coeffs.append(np.corrcoef(cluster_means, within_cluster.loc[cid].values)[0][1])
            elif use_spearmanr is True:
                corr_coeffs.append(spearmanr(cluster_means, within_cluster.loc[cid].values)[0])
        cluster_corrs[label] = corr_coeffs
    return cluster_corrs


def add_within_cluster_corr_to_cluster_meta(feature_matrix, cluster_meta, use_spearmanr=False):
    '''
    Computes correlation coefficient of each cell's dropout scores with the average dropout score for it's cluster

    feature_matrix: dataframe where rows are unique cell_specimen_ids and columns are dropout scores across experience levels
    cluster_meta: dataframe with cell_specimen_ids as index and cluster metadata as cols, must include 'cluster_id'
    '''
    print('adding within cluster correlation to cluster_meta')
    for cre_line in get_cre_lines(cluster_meta):
        cluster_meta_cre = cluster_meta[cluster_meta.cre_line == cre_line]
        clusters = cluster_meta_cre['cluster_id'].unique()
        for cluster_id in clusters:
            cluster_csids = cluster_meta_cre[cluster_meta_cre.cluster_id == cluster_id].index.values
            # get average dropout scores for this cluster
            cluster_dropouts = feature_matrix.loc[cluster_csids]
            cluster_mean = cluster_dropouts.mean().abs().values
            # compute correlation of each cell in this cluster to the average
            for cell_specimen_id in cluster_csids:
                if use_spearmanr is False:
                    corr_coeff = np.corrcoef(cluster_mean, feature_matrix.loc[cell_specimen_id].values)[0][1]
                elif use_spearmanr is True:
                    from scipy.stats import spearmanr
                    corr_coeff = spearmanr(cluster_mean, feature_matrix.loc[cell_specimen_id].values)[0]
                # save correlation value to cluster_meta
                cluster_meta.at[cell_specimen_id, 'within_cluster_correlation'] = corr_coeff
    return cluster_meta


def get_cluster_meta(cluster_labels, cell_metadata, feature_matrix, n_clusters_cre, save_dir, load=True):
    """
    merges dataframe with cluster IDs for each cell_specimen_id with dataframe with metadata for each cell_specimen_id
    removes clusters with less than 5 cells
    adds manually sorted cluster order for plotting
    computes within cluster correlation for each cell's dropouts compared to cluster average

    if save_dir, saves cluster_meta
    if load, loads from saved file, unless it doesnt exist, in that case generate and save
    """
    cluster_meta_filename = get_cluster_label_file_name(get_cre_lines(cell_metadata), n_clusters_cre, prefix='cluster_metadata')
    cluster_meta_filepath = os.path.join(save_dir, cluster_meta_filename)
    if load:
        if os.path.exists(cluster_meta_filepath):
            print('loading cluster_meta from', cluster_meta_filepath)
            cluster_meta = pd.read_hdf(cluster_meta_filepath, key='df')
        else:
            generate_data = True
    else:
        generate_data = True

    if generate_data:
        print('generating cluster_meta')
        cluster_meta = cluster_labels[['cell_specimen_id', 'cluster_id', 'labels']].merge(cell_metadata,
                                                                                          on='cell_specimen_id')
        cluster_meta = cluster_meta.set_index('cell_specimen_id')
        # annotate & clean cluster metadata
        cluster_meta = clean_cluster_meta(cluster_meta)  # drop cluster IDs with fewer than 5 cells in them
        cluster_meta['original_cluster_id'] = cluster_meta.cluster_id
        # cluster_meta = add_manual_sort_order_to_cluster_meta(cluster_meta)
        cluster_meta = add_within_cluster_corr_to_cluster_meta(feature_matrix, cluster_meta, use_spearmanr=False)
        # save
        print('saving cluster_meta to', cluster_meta_filepath)
        cluster_meta.to_hdf(cluster_meta_filepath, key='df')
    return cluster_meta


def make_frequency_table(cluster_meta, groupby_columns=['binned_depth'], normalize=True):
    '''
    Computes normalized frequency for depth and area (or whatever groupby columns are provided) per cluster
    Converts groupby columns into a single column label (instead of a multi-index) in output table
    (ex: 'VISp_upper' instead of ('VISp', 'upper')
    Input:
    cluster_meta: pd.DataFrame should include columns ['cluster_id', 'cell_specimen_id', *groupby_columns*]
    groupby_columns: array, includes one or more columns to be used in groupby method

    Returns:
    frequency_table: pd.Dataframe, rows are cluster IDs, columns are locations (ex: area / depths) over which frequencies are computed

    '''

    stats_df = cluster_meta[['cluster_id', *groupby_columns]]
    frequency_table = stats_df.groupby(groupby_columns)['cluster_id'].value_counts(normalize=normalize).unstack()
    frequency_table = frequency_table.fillna(0)
    # transpose to make rows cluster IDs
    frequency_table = frequency_table.T
    # convert multi-index columns to single string, if there are multiple groupby columns
    if len(groupby_columns) == 1:
        pass
    elif len(groupby_columns) == 2:
        columns = [frequency_table.columns[i][0] + '_' + frequency_table.columns[i][1] for i in range(len(frequency_table.columns))]
        frequency_table.columns = columns
    elif len(groupby_columns) == 3:
        columns = [frequency_table.columns[i][0] + '_' + frequency_table.columns[i][1] + '_' + frequency_table.columns[i][1] for i in range(len(frequency_table.columns))]
        frequency_table.columns = columns
    else:
        print('this function only hands 1-3 groupby columns, modify function if grouping by more than 3 things')
    # invert column sort order when using area/depth as groupby_columns - this puts VISp_upper first
    if 'VISp_upper' in frequency_table.columns:
        frequency_table = frequency_table[np.sort(frequency_table.columns)[::-1]]
    return frequency_table


def get_cell_counts_for_conditions(cell_data, conditions_to_groupby=['targeted_structure', 'layer']):
    """
    Computes the number of cells for the provided set of conditions
    cell_data is a table of cell metadata including cell_specimen_id and the columns in conditions_to_groupby
    """
    # count number of cells in provided groups
    cell_counts = cell_data.groupby(conditions_to_groupby).count()
    # get some column in dataframe and rename as n_cells
    col_to_use = cell_counts.keys()[0]
    cell_counts = cell_counts.rename(columns={col_to_use: 'n_cells'})[['n_cells']]
    return cell_counts


def get_n_cells_values_for_condition(group):
    return pd.Series({'random_n_cells': group.n_cells.values,
                      'random_n_cells_mean': group.n_cells.mean()})


def get_random_cell_counts(n_cells_in_cluster, cre_metadata, conditions_to_groupby):
    """
    randomly samples list of cell_specimen_ids in cell_metadata to get a random set of cells,
    equal to the number of cells in a given cluster, this is done iteratively 100x,
    then counts how randomly selected cells are distributed across the conditions_to_groupby (ex: areas and depths).
    the dataframe that is returned contains a list of the number of cells per condition
    that are randomly drawn from the dataset across those 100 iterations, for each of the conditions_to_gropuby,

    n_cells_in_cluster: number of cells to randomly draw on each iteration
    cell_metadata: cell metadata where rows are cell_specimen_id,
                and columns include cluster_id and the conditions_to_gropuby
                should be limited to a specific cre line if the clustering is done on each cre line separately
    conditions_to_groupby: columns in cell_metadata to count randomly drawn cells in
    """
    import random
    n_random_samples = 100
    cre_csids = cre_metadata.index.values
    random_cell_counts = pd.DataFrame()
    for iteration in range(n_random_samples):
        random_csids = random.sample(list(cre_csids), n_cells_in_cluster)
        random_cell_data = cre_metadata.loc[random_csids]
        random_cells = get_cell_counts_for_conditions(random_cell_data, conditions_to_groupby)
        random_cells = random_cells.reset_index()
        random_cells['iteration'] = iteration
        random_cell_counts = pd.concat([random_cell_counts, random_cells])
    random_cell_counts = random_cell_counts.groupby(conditions_to_groupby).apply(get_n_cells_values_for_condition)
    return random_cell_counts


def t_test(group):
    """
    computes p-value for one sample t-test comparing actual number of cells in a given group (typically grouped by area/depth)
    to the number of cells that fall into that same group by random sampling 100x
    :param group:
    :return:
    """
    from scipy import stats
    actual = group.n_cells.values[0]
    random = group.random_n_cells.values[0]
    n_times_greater = len(random[random < actual])  # number of times the actual is greater than random
    n_times_less_than = len(random[random > actual])  # number of times the actual is less than random
    n_times_equal = len(random[random == actual])  # number of times the actual value is equal to random
    t, p_val_greater = stats.ttest_1samp(random, actual, alternative='less')  # actual is greater than distribution mean
    t, p_val_less = stats.ttest_1samp(random, actual, alternative='greater')  # actual is less than distribution mean
    return pd.Series({'n_times_actual_greater_than_random': n_times_greater, 'n_times_actual_less_than_random': n_times_less_than, 'n_times_actual_equal_to_random': n_times_equal,
                      'p_val_actual_greater_than_random_r': np.round(p_val_greater, 5), 'p_val_actual_less_than_random_r': np.round(p_val_less, 5),
                      'p_val_actual_greater_than_random': p_val_greater, 'p_val_actual_less_than_random': p_val_less, })


def get_cell_count_stats(cluster_meta, conditions_to_groupby=['targeted_structure', 'layer']):
    """
    quantify how many cells are in each area and depth per cluster
    compute a random distrubtion of cells in area and depth for a given cluster number given sampling bias in dataset
    count how many times the actual number of cells per area and depth is greater or less than random,
    and associated p-value using 1-sample t-test
    quantify cells per area and depth per cluster as a fraction of the average of random distribution and
    value relative to random distribution (fraction-1) so that + is more than and - is less than random average
    """
    try:
        cluster_meta = cluster_meta.set_index('cell_specimen_id')
    except BaseException:
        pass
    cre_lines = np.sort(cluster_meta.cre_line.unique())

    cell_count_stats = pd.DataFrame()
    # get data just for one cre line
    for cre_line in cre_lines:
        cre_data = cluster_meta[cluster_meta.cre_line == cre_line]
        # get number of cells within each condition for this cre line
        cre_counts = get_cell_counts_for_conditions(cre_data, conditions_to_groupby)
        cre_counts = cre_counts.rename(columns={'n_cells': 'n_cells_cond_cre'})
        cre_csids = cre_data.index.values
        # get data for all clusters for this cre line
        cluster_ids = np.sort(cre_data.cluster_id.unique())
        for cluster_id in cluster_ids:
            cluster_csids = cre_data[cre_data.cluster_id == cluster_id].index.values
            cluster_data = cre_data.loc[cluster_csids]
            n_cells_in_cluster = len(cluster_data)
            # count the number of cells per area and depth
            actual_cell_counts = get_cell_counts_for_conditions(cluster_data, conditions_to_groupby)
            actual_cell_counts.insert(loc=0, column='n_cells_cond_cluster', value=actual_cell_counts.n_cells)
            # get a random distribution of the number of cells per area and depth from this cre line for a given cluster size
            random_cell_counts = get_random_cell_counts(n_cells_in_cluster, cre_data, conditions_to_groupby)
            # get info about overall distribution of cells within this cre line
            cell_counts = cre_counts.merge(actual_cell_counts, on=conditions_to_groupby)
            # add info about overall distribution of cells within this cre line
            cell_counts.insert(loc=0, column='n_cells_total_cre', value=len(cre_csids))
            # fraction of cells per area/depth relative to total cells in cre line
            cell_counts.insert(loc=2, column='fraction_per_cond_cre', value=cell_counts.n_cells_cond_cre / cell_counts.n_cells_total_cre)
            cell_counts.insert(loc=0, column='cre_line', value=cre_line)
            # add cluster and cre line metadata
            cell_counts.insert(loc=4, column='cluster_id', value=cluster_id)
            cell_counts.insert(loc=5, column='n_cells_total_cluster', value=len(cluster_csids))
            # fraction of cells in this area/depth relative to overall size of cluster
            cell_counts.insert(loc=7, column='fraction_per_cond_cluster', value=cell_counts.n_cells_cond_cluster / cell_counts.n_cells_total_cluster)
            # combine actual and random counts
            cell_counts = cell_counts.merge(random_cell_counts, on=conditions_to_groupby)
            # compute fraction of cells per this area/depth in this cluster
            # relative to the number of cells in this area/depth you would get if you randomly sample cells from this cre line
            cell_counts['fraction_of_random'] = cell_counts.n_cells_cond_cluster / cell_counts.random_n_cells_mean
            # subtract 1 so that 0 means same # cells as random, +ve means more frequent than random, -ve means less frequent than random
            cell_counts['relative_to_random'] = cell_counts.fraction_of_random - 1
            cell_counts = cell_counts.reset_index()
            # get p-value describing significance of difference between actual and random distribution for this condition (area/depth)
            stats = cell_counts.groupby(conditions_to_groupby).apply(t_test)
            cell_counts = cell_counts.merge(stats, on=conditions_to_groupby)
            # concat with big df
            cell_count_stats = pd.concat([cell_count_stats, cell_counts])
    # make binary column for significance
    cell_count_stats['sig_greater'] = [True if p_val_actual_greater_than_random < 0.05 else False for p_val_actual_greater_than_random in cell_count_stats.p_val_actual_greater_than_random.values]
    if len(conditions_to_groupby) > 1:
        cell_count_stats['location'] = cell_count_stats[conditions_to_groupby[0]] + '_' + cell_count_stats[conditions_to_groupby[1]]
    elif len(conditions_to_groupby) == 1:
        cell_count_stats['location'] = cell_count_stats[conditions_to_groupby[0]]
    return cell_count_stats


def get_cluster_proportions(df, cre):
    '''
        Returns two tables
        proportion_table contains the proportion of cells in each depth/area found in each cluster, relative to the average proportion across depth/areas for that cluster
        stats_table returns statistical tests on the proportion of cells in each depth/area
        Use 'bh_significant' unless you have a good reason to use the uncorrected tests
    '''
    proportion_table = compute_cluster_proportion_cre(df, cre)
    stats_table = stats(df, cre)
    return proportion_table, stats_table


def compute_cluster_proportion_cre(cluster_meta, cre_line, groupby_columns=['targeted_structure', 'layer']):
    """
    gets proportion of cells per cluster for each area and depth,
    then subtracts the average proportion for each cluster
    """
    frequency = make_frequency_table(cluster_meta[cluster_meta.cre_line == cre_line],
                                     groupby_columns=groupby_columns, normalize=True)

    table = frequency.copy()
    # get average proportion in each cluster
    table['mean'] = table.mean(axis=1)

    # compute proportion in each area relative to cluster average
    depth_areas = table.columns.values[:-1]  # exclude mean column from calculation
    for da in depth_areas:
        table[da] = table[da] - table['mean']
        # drop the mean column
    table = table.drop(columns='mean')

    return table


# def get_proportion_cells_rel_cluster_average(cluster_meta, cre_lines, groupby_columns=['targeted_structure', 'layer']):
#     cluster_proportions = pd.DataFrame()
#     for cre_line in cre_lines:
#         table = compute_cluster_proportion_cre(cluster_meta, cre_line, groupby_columns=groupby_columns)
#         df = pd.DataFrame(table.unstack(), columns=['proportion_cells'])
#         df = df.reset_index()
#         df = df.rename(columns={'level_0': 'location'})
#         df['cre_line'] = cre_line
#         cluster_proportions = pd.concat([cluster_proportions, df])
#     return cluster_proportions


def get_proportion_cells_rel_cluster_average(cluster_meta, cre_lines, columns_to_groupby=['targeted_structure', 'layer']):
    """

    computes the proportion of cells in each location (defined by values of columns_to_groupby) for a given cluster,
    relative to the average proportion of cells across all locations in that cluster, for each cre line;
    as well as the significance of proportions relative to the average using bh corrected chi-square test
    uses visual_behavior_glm functions to compute proportions and stats

    Note that the input is NOT limited to one cre line

    :param cluster_meta: dataframe containing metadata for all cell_specimen_ids, including cluster_id
    :param cre_lines: cre lines to iterate over when creating proportions table
    :param columns_to_groupby: columns in cluster_meta to use when computing proportion cells per location;
                                location is defined as the concatenation of the groups in columns_to_groupby
    :return: cluster_proportions: table with columns for cre line, cluster_id, location, and proportion
                                    proportion is the proportion of cells in location for a given cluster, relative to cluster average
            stats_table: table of pvalues for bh corrected chi-square test to quantify significance of proportion cells relative to cluster average
                                    includes stats for each cre line and couster
    """
    from visual_behavior_glm import GLM_clustering as glm_clust
    cluster_proportions = pd.DataFrame()
    stats_table = pd.DataFrame()
    for cre_line in cre_lines:
        # limit to cre line
        cluster_meta_cre = cluster_meta[cluster_meta.cre_line == cre_line].copy()
        # add location to quantify proportions in
        # location column is a categorical variable (string) that can be a combination of area and depth or just area or depth (or whatever)
        if 'location' not in cluster_meta_cre.keys():
            cluster_meta_cre = add_location_column(cluster_meta_cre, columns_to_groupby)

        # this code gets the proportions across locations and computes significance with corrected chi-square test
        # glm_clust.final will use the location column in cluster_meta_copy to get proportions and stats for those groupings within each cluster
        proportion_table, stats = glm_clust.final(cluster_meta_cre.reset_index(), cre_line)
        stats['cre_line'] = cre_line

        # reformat proportion_table to match format expected by downstream plotting code
        cre_proportions = pd.DataFrame(proportion_table.unstack()).rename(columns={0: 'proportion_cells'})
        cre_proportions = cre_proportions.reset_index()
        cre_proportions['cre_line'] = cre_line

        # add cre line proportions to unified table
        cluster_proportions = pd.concat([cluster_proportions, cre_proportions])
        stats_table = pd.concat([stats_table, stats])

    return cluster_proportions, stats_table

def stats(df, cre):
    '''
        Performs chi-squared tests to asses whether the observed cell counts in each area/depth differ
        significantly from the average for that cluster.
    '''

    from scipy.stats import chisquare

    # compute cell counts in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id', 'coarse_binned_depth_area'])['cell_specimen_id'].count().unstack()
    table = table[['VISp_upper', 'VISp_lower', 'VISl_upper', 'VISl_lower']]
    table = table.fillna(value=0)

    # compute proportion
    depth_areas = table.columns.values
    for da in depth_areas:
        table[da] = table[da] / table[da].sum()

    # get average for each cluster
    table['mean'] = table.mean(axis=1)

    # second table of cell counts in each area/cluster
    table2 = df.query('cre_line == @cre').groupby(['cluster_id', 'coarse_binned_depth_area'])['cell_specimen_id'].count().unstack()
    table2 = table2[['VISp_upper', 'VISp_lower', 'VISl_upper', 'VISl_lower']]
    table2 = table2.fillna(value=0)

    # compute estimated frequency of cells based on average fraction for each cluster
    for da in depth_areas:
        table2[da + '_chance_count'] = table2[da].sum() * table['mean']

    # perform chi-squared test
    for index in table2.index.values:
        f = table2.loc[index][['VISp_upper', 'VISp_lower', 'VISl_upper', 'VISl_lower']].values
        f_expected = table2.loc[index][['VISp_upper_chance_count', 'VISp_lower_chance_count', 'VISl_upper_chance_count', 'VISl_lower_chance_count']].values
        out = chisquare(f, f_expected)
        table2.at[index, 'pvalue'] = out.pvalue
        table2.at[index, 'significant'] = out.pvalue < 0.05

    # Use Benjamini Hochberg Correction for multiple comparisons
    table2 = add_hochberg_correction(table2)
    return table2


def add_hochberg_correction(table):
    '''
        Performs the Benjamini Hochberg correction
    '''
    # Sort table by pvalues
    table = table.sort_values(by='pvalue').reset_index()

    # compute the corrected pvalue based on the rank of each test
    table['imq'] = table.index.values / len(table) * 0.05

    # Find the largest pvalue less than its corrected pvalue
    # all tests above that are significant
    table['bh_significant'] = False
    passing_tests = table[table['pvalue'] < table['imq']]
    if len(passing_tests) > 0:
        last_index = table[table['pvalue'] < table['imq']].tail(1).index.values[0]
        table.at[last_index, 'bh_significant'] = True
        table.at[0:last_index, 'bh_significant'] = True

    # reset order of table and return
    return table.sort_values(by='cluster_id').set_index('cluster_id')


def compute_proportion_cre(df, cre):
    '''
        Computes the proportion of cells in each cluster within each location
    '''
    # Count cells in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id', 'coarse_binned_depth_area'])['cell_specimen_id'].count().unstack()
    table = table[['VISp_upper', 'VISp_lower', 'VISl_upper', 'VISl_lower']]
    table = table.fillna(value=0)

    # compute fraction in each area/cluster
    depth_areas = table.columns.values
    for da in depth_areas:
        table[da] = table[da] / table[da].sum()
    return table


def get_cluster_order_for_metric_location(cell_count_stats, cluster_meta, location='VISp_upper',
                                          metric='relative_to_random'):
    """
    obtain values of metric for the given location and sort clusters within each cre line by that metric value
    location and metric are columns in cell_count_stats to sort the data by
    location is typically a conjunction of targeted_structure and layer
        but can be any set of conditions defined by the column 'location' in cell_count_stats
    metric can be things like the fraction or number of cells in a given location
    cell_count_stats is a table with information about the number and fraction of cells
        for a given set of conditions (such as area & depth) in each cluster in each cre line
    """
    cluster_order = {}
    for cre_line in get_cre_lines(cluster_meta):
        cre_data = cell_count_stats[(cell_count_stats.cre_line == cre_line) &
                                    (cell_count_stats.location == location)]
        cluster_order_cre = cre_data.sort_values(by=metric, ascending=False).cluster_id.values
        cluster_order[cre_line] = cluster_order_cre
    return cluster_order


def get_fraction_cells_per_area_depth(cluster_meta):
    """
    computes the fraction of cells per area/depth in a given cluster
    relative to the total number of cells in that area/depth in that cre line
    i.e. 'out of all the cells in this area and depth, how many are in this cluster? '
    """
    # get fraction cells per area and depth per cluster
    # frequency will be the fraction of cells in each cluster for a given area and depth
    # relative to the total number of cells in that area and depth for that cre line
    n_cells_per_area_depth = cluster_meta.reset_index().groupby(['cre_line', 'targeted_structure', 'layer']).count().rename(columns={'cell_specimen_id': 'n_cells_total'})[['n_cells_total']]
    n_cells_per_area_depth_cluster = cluster_meta.reset_index().groupby(['cre_line', 'cluster_id', 'targeted_structure', 'layer']).count().rename(columns={'cell_specimen_id': 'n_cells_cluster'})[['n_cells_cluster']]
    fraction_cells = n_cells_per_area_depth_cluster.reset_index().merge(n_cells_per_area_depth.reset_index(), on=['cre_line', 'targeted_structure', 'layer'], how='left')
    fraction_cells['fraction_cells_per_area_depth'] = fraction_cells.n_cells_cluster / fraction_cells.n_cells_total
    return fraction_cells


def get_coding_metrics(index_dropouts, index_value, index_name):
    """
    Computes a variety of metrics on dropout scores for some condition (ex: one cell or one cluster)
    including selectivity across features or experience levels, within and across sessions

    index_dropouts can be the dropout scores for a single cell, or average dropouts across cells
    index_dropouts should be in the format of rows = experience levels, cols = regressors
    index_value should be the cluster_id or cell_specimen_id corresponding to the input dropouts
    index_name is the column name corresponding to index_value, i.e. 'cluster_id' or 'cell_specimen_id'
    """
    stats = pd.DataFrame(index=[index_value])
    stats.index.name = index_name
    # get dropout scores per cell
    # get preferred regressor and experience level and save
    dominant_feature = index_dropouts.stack().idxmax()[1]
    dominant_experience_level = index_dropouts.stack().idxmax()[0]
    stats.loc[index_value, 'dominant_feature'] = dominant_feature
    stats.loc[index_value, 'dominant_experience_level'] = dominant_experience_level
    # get selectivity for feature & experience level
    # feature selectivity is ratio of largest dropout vs mean of all other dropous for the dominant experience level
    order = np.argsort(index_dropouts.loc[dominant_experience_level])
    values = index_dropouts.loc[dominant_experience_level].values[order[::-1]]
    feature_selectivity = (values[0] - (np.mean(values[1:]))) / (values[0] + (np.mean(values[1:])))
    # experience selectivity is ratio of largest and average of all other dropouts for the dominant feature
    order = np.argsort(index_dropouts[dominant_feature])
    values = index_dropouts[dominant_feature].values[order[::-1]]
    experience_selectivity = (values[0] - (np.mean(values[1:]))) / (values[0] + (np.mean(values[1:])))
    stats.loc[index_value, 'feature_selectivity'] = feature_selectivity
    stats.loc[index_value, 'experience_selectivity'] = experience_selectivity
    # get experience modulation indices
    row = index_dropouts[dominant_feature]
    # direction of exp mod is whether coding is stronger for familiar or novel
    exp_mod_direction = (row['Novel 1'] - row['Familiar']) / (row['Novel 1'] + row['Familiar'])
    # persistence of exp mod is whether coding stays similar after repetition of novel session
#     exp_mod_persistence = (row['Novel >1']-row['Novel 1'])/(row['Novel >1']+row['Novel 1'])
    exp_mod_persistence = row['Novel >1'] / row['Novel 1']
    stats.loc[index_value, 'exp_mod_direction'] = exp_mod_direction
    stats.loc[index_value, 'exp_mod_persistence'] = exp_mod_persistence
    # within session joint coding index
    # get order and values of features for pref exp level
    order = np.argsort(index_dropouts.loc[dominant_experience_level])
    values = index_dropouts.loc[dominant_experience_level].values[order[::-1]]
#     # joint coding index is first highest feature dropout relative to mean of all other features
#     feature_sel_within_session = (values[0]-(np.mean(values[1:])))/(values[0]+(np.mean(values[1:])))
    # joint coding index is first highest feature dropout relative to next highest
    feature_sel_within_session = (values[0] - values[1]) / (values[0] + values[1])
    # get across session switching index - ratio of pref exp & feature to next max exp & feature, excluding within session or feature comparisons
    # get the max value
    pref_cond_value = index_dropouts.loc[dominant_experience_level][dominant_feature]
    # get the remaining columns after removing the max exp level and feature
    non_pref_conditions = index_dropouts.drop(index=dominant_experience_level).drop(columns=dominant_feature)
    # get the exp level and feature with the next highest value
    next_highest_conditions = non_pref_conditions.stack().idxmax()
    stats.insert(loc=2, column='next_highest_conditions', value=[next_highest_conditions])
    next_experience_level = next_highest_conditions[0]
    next_feature = next_highest_conditions[1]
    next_highest_value = index_dropouts.loc[next_experience_level][next_feature]
    # switching index is ratio of max condition to next max (excluding within session & feature comparisons)
    feature_sel_across_sessions = (pref_cond_value - next_highest_value) / (pref_cond_value + next_highest_value)
    stats.loc[index_value, 'feature_sel_within_session'] = feature_sel_within_session
    stats.loc[index_value, 'feature_sel_across_sessions'] = feature_sel_across_sessions
    return stats


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


def shuffle_dropout_score(df_dropout, shuffle_type='all'):
    '''
    Shuffles dataframe with dropout scores from GLM.
    shuffle_type: str, default='all', other options= 'experience', 'regressors'
    '''
    df_shuffled = df_dropout.copy()
    regressors = df_dropout.columns.levels[0].values
    experience_levels = df_dropout.columns.levels[1].values
    if shuffle_type == 'all':
        print('shuffling all data')
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


def get_CI_for_clusters(cluster_meta, columns_to_groupby=['targeted_structure', 'layer'], alpha=0.05, type_of_CI='mpc', cre = 'all'):
    '''
    Computes CI for cluster sizes using statsmodels.stats.proportion.proportion_confint function
    Input:
        cluster_meta (pd.DataFrame): len of which is number of cells, with column 'cluster_id'
        alpha (int): p value, default = 0.01
        columns_to_groupby: columns in cluster_meta to use when computing proportion cells per location;
                                location is defined as the concatenation of the groups in columns_to_groupby,
                                default = [ 'targeted_structure', 'layer'], do not add cluster-id here
        type_of_CI (str), how CIs are computed, default mpc, mpc=multinomial_proportinal_confident, pc=proportion_confint
    Returns:
        CI_df: pd.DataFrame with columns Low_CI, High_CI, and CI (difference)
    '''

    if 'location' not in cluster_meta.keys():
        cluster_meta = add_location_column(cluster_meta, columns_to_groupby)

    # adding option to select cre line
    if cre == 'all':
        cre_lines = np.sort(cluster_meta['cre_line'].unique())
    else:
        cre_lines = np.array(cre)
    locations = np.sort(cluster_meta['location'].unique())

    CI_df = pd.DataFrame(columns=['location', 'cluster_id', 'cre_line', 'CI_lower', 'CI_upper'])

    if type_of_CI == 'pc':
        for cre_line in cre_lines:
            # subset cre_line
            cluster_meta_cre = cluster_meta[cluster_meta['cre_line'] == cre_line].copy()

            # get groupby totals for all locations
            df_groupedby_totals = cluster_meta_cre.reset_index().groupby('location').count().rename(columns={
                'cell_specimen_id': 'n_cells_total'})[['n_cells_total']]

            # get groupby totals for locations and cluster id
            df_groupedby_per_cluster = \
            cluster_meta_cre.reset_index().groupby(['location', 'cluster_id']).count().rename(columns={
                'cell_specimen_id': 'n_cells_cluster'})[['n_cells_cluster']]

            # get cluster ids for from this dataframe
            cluster_ids = cluster_meta_cre['cluster_id'].unique()

            # iterate over locations
            for location in locations:
                # total N of observation in this location
                n_total = df_groupedby_totals.loc[(location)].values
                for cluster_id in cluster_ids:
                    # compute CI for this cluster/location
                    try:
                        n_cluster = df_groupedby_per_cluster.loc[(location, cluster_id)].values[0]
                        CI = proportion_confint(n_cluster, n_total, alpha=alpha, )
                        data = {'location': location,
                                'cluster_id': cluster_id,
                                'cre_line': cre_line,
                                'CI_lower': CI[0],
                                'CI_upper': CI[1]}
                    # if no such cluster found, there were no cells in this location
                    except KeyError:
                        print(f'{cre_line, location, cluster_id} no cells in this cluster')
                        data = {'location': location,
                                'cluster_id': cluster_id,
                                'cre_line': cre_line,
                                'CI_lower': [0],
                                'CI_upper': [0]}

                    CI_df = CI_df.append(pd.DataFrame(data))

    elif type_of_CI == 'mpc':
        for cre_line in cre_lines:
            # subset cre_line
            cluster_meta_cre = cluster_meta[cluster_meta['cre_line'] == cre_line].copy()

            # get groupby totals for all locations
            df_groupedby_totals = cluster_meta_cre.reset_index().groupby('location').count().rename(columns={
                'cell_specimen_id': 'n_cells_total'})[['n_cells_total']]

            # get groupby totals for locations and cluster id
            df_groupedby_per_cluster = \
                cluster_meta_cre.reset_index().groupby(['location', 'cluster_id']).count().rename(columns={
                    'cell_specimen_id': 'n_cells_cluster'})[['n_cells_cluster']]

            # get cluster ids for from this dataframe
            cluster_ids = cluster_meta_cre['cluster_id'].unique()

            # if cluster ids start with 0 do nothing
            if 0 in cluster_ids:
                e = 0
            else:  # if cluster ids start with 1, subtract 1 for indexing CIs
                e = 1
            for location in locations:
                N = []
                # collect all n per cluster in this location into one array N
                for cluster_id in cluster_ids:
                    try:
                        n_cluster = df_groupedby_per_cluster.loc[(location, cluster_id)].values[0]
                    except: # used to have KeyError here but sometimes its a TypeError when there are no cells in a cluster
                        print(f'{cre_line, location, cluster_id} no cells in this cluster')
                        n_cluster = 0
                    N.append(n_cluster)
                #compute CIs for this location
                CIs = multinomial_proportions_confint(N, alpha=alpha)

                # asign CIs to their cluster
                for cluster_id in cluster_ids:
                    # not sure why, mpc can return 3d or 2d array.
                    # if 3d, first D is usually 1, second D = number of clusters, and third D = 2 CIs
                    if len(CIs.shape) == 3 and CIs.shape[0] == 1:
                        CI = CIs[0][cluster_id - e]
                    elif len(CIs.shape) == 2:
                        CI = CIs[cluster_id - e]
                    data = {'location': location,
                            'cluster_id': cluster_id,
                            'cre_line': cre_line,
                            'CI_lower': [CI[0]],
                            'CI_upper': [CI[1]]}
                    CI_df = CI_df.append(pd.DataFrame(data))

    CI_df['CI'] = CI_df['CI_upper'] - CI_df['CI_lower']
    return CI_df


def add_location_column(cluster_meta, columns_to_groupby=['targeted_structure', 'layer']):
    '''
    function to add location column to a df
    INPUT:
    cluster_meta: (pd.DataFrame) of groupout scores for each cell with other meta data columns
    columns_to_groupby: columns in cluster_meta to use to create new location column
                                location is defined as the concatenation of the column values in columns_to_groupby

    '''
    cluster_meta_copy = cluster_meta.copy()
    if 'layer' not in cluster_meta.keys():
        cluster_meta_copy['layer'] = ['upper' if x < 250 else 'lower' for x in cluster_meta_copy['imaging_depth']]
        print('column layer is not in the dataframe, using 250 as a threshold to create the column')
    if len(columns_to_groupby) == 0:
        print('no columns were provided for grouping location')
    elif len(columns_to_groupby) == 1:
        cluster_meta_copy['location'] = cluster_meta_copy[columns_to_groupby[0]]
    elif len(columns_to_groupby) == 2:
        cluster_meta_copy['location'] = cluster_meta_copy[columns_to_groupby[0]] + '_' + cluster_meta_copy[columns_to_groupby[1]]
    elif len(columns_to_groupby) == 3:
        cluster_meta_copy['location'] = cluster_meta_copy[columns_to_groupby[0]] + '_' + cluster_meta_copy[
            columns_to_groupby[1]]+ '_' + cluster_meta_copy[columns_to_groupby[2]]
    elif len(columns_to_groupby) > 3:
        print('cannot combine more than three columns into a location column')

    return cluster_meta_copy
