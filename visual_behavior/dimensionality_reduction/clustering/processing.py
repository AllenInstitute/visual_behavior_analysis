
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import linalg as linalg

from scipy.stats import spearmanr
from scipy.stats import kruskal
from scipy.stats import ttest_ind
from scipy.stats import sem
from scipy.sparse import csgraph
from scipy.stats import chisquare
from scipy import stats

from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.utils as utils

import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_across_session as gas

import visual_behavior_glm.GLM_clustering as glm_clust
import visual_behavior_glm.GLM_params as glm_params
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import multinomial_proportions_confint
from visual_behavior.dimensionality_reduction.clustering import plotting as plotting


import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})


# from scipy.stats import ttest_1samp

# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
# cache_dir = loading.get_analysis_cache_dir()
# cache = bpc.from_s3_cache(cache_dir)

# import visual_behavior_glm.GLM_analysis_tools as gat  # to get recent glm results
# import visual_behavior_glm.GLM_params as glm_params


# utilities ###

def get_multi_session_df_for_omissions():
    """
    loads multi-session omission response dataframe for events, only closest familiar and novel active
    """
    # load multi session dataframe with response traces
    conditions = ['cell_specimen_id', 'omitted']

    data_type = 'events'
    event_type = 'omissions'
    inclusion_criteria = 'active_only_closest_familiar_and_novel_containers_with_all_levels'

    multi_session_df = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria)
    print(len(multi_session_df.ophys_experiment_id.unique()))
    return multi_session_df


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
    features = ['all-images', 'omissions', 'task', 'behavioral']
    return features


def get_feature_labels_for_clustering():
    """
    get GLM feature names to use for labeling plots
    """
    feature_labels = ['images', 'omissions', 'task', 'behavior']
    return feature_labels


def get_cre_lines(cell_metadata):
    """
    get list of cre lines in cell_metadata and sort in reverse order so that Vip is first
    cell_metadata is a table similar to ophys_cells_table from SDK but can have additional columns based on clustering results
    """
    cre_lines = np.sort(cell_metadata.cre_line.unique())[::-1]
    return cre_lines


def get_cell_type_for_cre_line(cre_line, cell_metadata=None):
    """
    gets cell_type label for a given cre_line using info in cell_metadata table or hardcoded lookup dictionary
    cell_metadata is a table similar to ophys_cells_table from SDK but can have additional columns based on clustering results
    """
    if cell_metadata is not None:
        cell_type = cell_metadata[cell_metadata.cre_line == cre_line].cell_type.values[0]
    else:
        cell_type_dict = {'Slc17a7-IRES2-Cre': 'Excitatory',
                          'Vip-IRES-Cre': 'Vip Inhibitory',
                          'Sst-IRES-Cre': 'Sst Inhibitory'}
        cell_type = cell_type_dict[cre_line]
    return cell_type


def get_cre_line_map(cre_line):

    cre_line_dict = {'Slc17a7-IRES2-Cre': 'Excitatory',
                     'Sst-IRES-Cre': 'SST inhibitory',
                     'Vip-IRES-Cre': 'VIP inhibitory',
                     'all': 'all cells'}

    mapped_cre_line = cre_line_dict[cre_line]
    return mapped_cre_line


def get_experience_map(experience_level):

    experience_dict = {'Familiar': 'Familiar',
                       'Novel 1': 'Novel',
                       'Novel >1': 'Novel+'}

    mapped_exp_level = experience_dict[experience_level]
    return mapped_exp_level


def get_n_clusters_cre():
    ''' Number of clusters used in clustering per cre line'''
    n_clusters_cre = {'Slc17a7-IRES2-Cre': 10,
                      'Sst-IRES-Cre': 5,
                      'Vip-IRES-Cre': 10}
    return n_clusters_cre


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
    locs_to_drop = tmp[tmp.labels <= 5].index
    # remove them from cluster_meta
    inds_to_drop = []
    for loc_to_drop in locs_to_drop:
        inds_to_drop_this_loc = list(cluster_meta[(cluster_meta.cre_line == loc_to_drop[0]) & (cluster_meta.cluster_id == loc_to_drop[1])].index.values)
        inds_to_drop = inds_to_drop + inds_to_drop_this_loc
        print('dropping', len(inds_to_drop_this_loc), 'cells for', loc_to_drop)
    cluster_meta = cluster_meta.drop(index=inds_to_drop)
    print(len(inds_to_drop), 'cells dropped total')
    return cluster_meta


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
            columns_to_groupby[1]] + '_' + cluster_meta_copy[columns_to_groupby[2]]
    elif len(columns_to_groupby) > 3:
        print('cannot combine more than three columns into a location column')

    return cluster_meta_copy


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


# loading & processing of GLM outputs ###

def get_glm_results_pivoted_for_clustering(glm_version='24_events_all_L2_optimize_by_session',
                                           model_output_type='adj_fraction_change_from_full',
                                           across_sessions_normalized=True, save_dir=None):
    """
    loads GLM results pivoted where columns are dropout scores and rows are cells in specific experiments
    filters to limit to cells matched in all experience levels
    loads from file if file exists in save_dir, otherwise generates from mongo and saves to save dir
    """
    if across_sessions_normalized and save_dir:
        results_file_path = os.path.join(save_dir, glm_version + '_across_normalized_results_pivoted.h5')
    elif save_dir:
        results_file_path = os.path.join(save_dir, glm_version + '_results_pivoted.h5')

    if os.path.exists(results_file_path) is False:
        print('loading results_pivoted from', results_file_path)
        results_pivoted = pd.read_hdf(results_file_path, key='df')

    else:  # if it doesnt exist, then load it and save (note this is slow)

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

        if across_sessions_normalized is False:
            results_pivoted = gat.build_pivoted_results_summary(value_to_use=model_output_type,
                                                                results_summary=None,
                                                                glm_version=glm_version, cutoff=None)

            # get rid of passive sessions
            results_pivoted = results_pivoted[results_pivoted.passive == False]
        elif across_sessions_normalized is True:
            results_pivoted, _ = gas.load_cells(glm_version=glm_version)

        # limit results pivoted to to last familiar and second novel
        results_pivoted = results_pivoted[results_pivoted.ophys_experiment_id.isin(matched_experiments)]
        results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(matched_cells)]
        print(len(results_pivoted.cell_specimen_id.unique()),
              'cells in results_pivoted after limiting to strictly matched cells')

        # if save_dir:
        #     # save filtered results to save_dir
        #     results_pivoted.to_hdf(results_file_path, key='df')
        #     print('results_pivoted saved')
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
        df, failed_cells = gas.load_cells(glm_version, clean_df=True)
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
    print(results_pivoted.keys())

    # save processed results_pivoted
    results_pivoted.to_hdf(os.path.join(glm_output_dir, glm_version + '_results_pivoted.h5'), key='df', format='table')

    # now drop ophys_experiment_id
    results_pivoted = results_pivoted.drop(columns=['ophys_experiment_id'])

    # get kernels for this version
    run_params = glm_params.load_run_json(glm_version)
    kernels = run_params['kernels']

    print(results_pivoted.keys())

    return results_pivoted, weights_df, kernels


def load_GLM_outputs(glm_version, base_dir):
    """
    loads results_pivoted and weights_df from files in base_dir
    results_pivoted and weights_df will be limited to the ophys_experiment_ids and cell_specimen_ids present in experiments_table and cells_table
    because this function simply loads the results, any pre-processing applied to results_pivoted (such as across session normalization or signed weights) will be used here

    :param glm_version: example = '24_events_all_L2_optimize_by_session'
    :param base_dir: directory containing output files to load
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
    glm_results_path = os.path.join(base_dir, glm_version + '_results_pivoted.h5')
    if os.path.exists(glm_results_path):
        results_pivoted = pd.read_hdf(glm_results_path, key='df')
    else:
        print('no results_pivoted at', glm_results_path)
        print('please generate before using load_glm_outputs')

    weights_path = os.path.join(base_dir, glm_version + '_weights_df.h5')
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


def get_feature_matrix_for_cre_line(feature_matrix, cell_metadata, cre_line, dropna=True):
    """
    limit feature_matrix to cell_specimen_ids for this cre line
    """
    if cre_line == 'all':
        feature_matrix_cre = feature_matrix.copy()
    else:
        cre_cell_specimen_ids = cell_metadata[cell_metadata['cre_line'] == cre_line].index.values
        feature_matrix_cre = feature_matrix.loc[cre_cell_specimen_ids].copy()

    if dropna is True:
        feature_matrix_cre = feature_matrix_cre.dropna(axis=0)
    return feature_matrix_cre


def normalize_cluster_size(cluster_df):
    '''
    Normalize cluster size in each CRE line and calculate the percentage.

    Args:
    - cluster_df (pd.DataFrame): DataFrame containing original cluster labels.

    Returns:
    - pd.DataFrame: DataFrame with normalized cluster sizes and percentages for each CRE line and cluster ID.
    '''
    # Normalize cluster size in each CRE line
    normalized_values = cluster_df.groupby('cre_line')['cluster_id'].value_counts(normalize=True).values

    # Create DataFrame with each CRE and cluster ID
    grouped_df = cluster_df.groupby(['cre_line','cluster_id']).count().reset_index()

    # Add normalized percentage values
    grouped_df['normalized'] = normalized_values
    grouped_df['percentage'] = grouped_df['normalized'] * 100

    return grouped_df

def prepare_data(cluster_df, rm_unstacked, cluster_id, exp_level, rm_f, cre_line=None):
    '''
    Prepare data based on cluster_id, exp_level, and cre_line.

    Args:
    - cluster_df (pd.DataFrame): DataFrame containing original cluster labels.
    - rm_unstacked (pd.DataFrame): DataFrame containing unstacked response metric data.
    - cluster_id (int): Cluster ID.
    - exp_level (str): Experience level.
    - cre_line (str): CRE line.
    - rm_f (str): Feature to analyze.

    Returns:
    - np.ndarray: Prepared data.
    '''
    if cre_line is None:
        tmp = cluster_df[cluster_df.cluster_id == cluster_id]
    else:
        tmp = cluster_df[(cluster_df.cluster_id == cluster_id) & (cluster_df.cre_line == cre_line)]
    cids = tmp.cell_specimen_id.values
    return rm_unstacked.loc[cids][[rm_f]][[(rm_f, exp_level)]].values

def extract_number(s):
    import re
    # Extract the number from the string using regular expression
    match = re.search(r'\d+', s)
    return int(match.group()) if match else 0


def custom_sort(s):
    # Use the extracted number for sorting
    return extract_number(s)


### selecting # K ###

def compute_inertia(a, X, metric='euclidean'):
    W = [np.mean(pairwise_distances(X[a == c, :], metric=metric)) for c in np.unique(a)]
    return np.mean(W)


def compute_gap(clustering, data, k_max=5, n_boots=20, reference_shuffle='all', metric='euclidean', separate_cre_lines=False):
    '''
    Computes gap statistic between clustered data (ondata inertia) and null hypothesis (reference intertia).

    :param clustering: clustering object that includes "n_clusters" and "fit_predict"
    :param data: an array of data to be clustered (n samples by n features)
    :param k_max: (int) maximum number of clusters to test, starts at 1
    :param n_boots: (int) number of repetitions for computing mean inertias
    :param reference: (str) what type of shuffle to use, shuffle_dropout_scores,
            None is use random normal distribution
    :param metric: (str) type of distance to use, default = 'euclidean'
    :param separate_cre_lines: (bool) whether or not to shuffle within cre lines. Set to True if feature matrix consists of multiple cre lines.
    :return:
    gap: array of gap values that are the difference between two inertias
    reference_inertia: array of log of reference inertia
    ondata_inertia: array of log of ondata inertia
    '''

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    if isinstance(data, pd.core.frame.DataFrame):
        data_array = data.values
    else:
        data_array = data

    gap_statistics = {}
    reference_inertia = []
    reference_sem = []
    gap_mean = []
    gap_sem = []
    for k in range(1, k_max ):
        local_ref_inertia = []
        for _ in range(n_boots):
            # draw random dist or shuffle for every nboot
            if reference_shuffle is None:
                reference = np.random.rand(*data.shape) * -1
            else:
                reference_df = shuffle_dropout_score(data, shuffle_type=reference_shuffle, separate_cre_lines=separate_cre_lines)
                print(len(reference_df))
                reference = reference_df.values

            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_ref_inertia.append(compute_inertia(assignments, reference, metric=metric))
        reference_inertia.append(np.mean(local_ref_inertia))
        reference_sem.append(sem(local_ref_inertia))

    ondata_inertia = []
    ondata_sem = []
    for k in range(1, k_max ):
        local_ondata_inertia = []
        for _ in range(n_boots):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(data_array)
            local_ondata_inertia.append(compute_inertia(assignments, data_array, metric=metric))
        ondata_inertia.append(np.mean(local_ondata_inertia))
        ondata_sem.append(sem(local_ondata_inertia))

        # compute difference before mean
        gap_mean.append(np.mean(np.subtract(np.log(local_ondata_inertia), np.log(local_ref_inertia))))
        gap_sem.append(sem(np.subtract(np.log(local_ondata_inertia), np.log(local_ref_inertia))))

    # maybe plotting error bars with this metric would be helpful but for now I'll leave it
    gap = np.log(reference_inertia) - np.log(ondata_inertia)

    # we potentially do not need all of this info but saving it to plot it for now
    gap_statistics['gap'] = gap
    gap_statistics['reference_inertia'] = np.log(reference_inertia)
    gap_statistics['ondata_inertia'] = np.log(ondata_inertia)
    gap_statistics['reference_sem'] = reference_sem
    gap_statistics['ondata_sem'] = ondata_sem
    gap_statistics['gap_mean'] = gap_mean
    gap_statistics['gap_sem'] = gap_sem

    return gap_statistics


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


def load_gap_statistic(glm_version, feature_matrix, cell_metadata, save_dir=None,
                       metric='euclidean', shuffle_type='all', k_max=25, n_boots=20, separate_cre_lines=False):
    """
       if gap statistic was computed and file exists in save_dir, load it
       otherwise run spectral clustering n_boots times, for a range of 1 to k_max clusters
       returns dictionary of gap_statistic for each cre line
        shuffle_type is an input to shuffle_dropout_score function, which is used as a null hypothesis or reference data
        metric is distance metric used for in compute gap function
       """
    gap_filename = 'gap_cores_' + metric + '_' + glm_version + '_' + shuffle_type + 'kmax' + str(k_max) + '_' + 'nb' + str(n_boots) + '.pkl'
    gap_path = os.path.join(save_dir, gap_filename)
    if os.path.exists(gap_path):
        print('loading gap statistic scores from', gap_path)
        with open(gap_path, 'rb') as f:
            gap_statistic = pickle.load(f)
            f.close()
        print('done.')
    else:
        gap_statistic = {}
        for cre_line in get_cre_lines(cell_metadata):
            feature_matrix_cre = get_feature_matrix_for_cre_line(feature_matrix, cell_metadata, cre_line)
            X = feature_matrix_cre.values
            feature_matrix_cre_shuffled = shuffle_dropout_score(feature_matrix_cre, shuffle_type=shuffle_type, separate_cre_lines=separate_cre_lines)
            reference = feature_matrix_cre_shuffled.values
            # create an instance of Spectral clustering object
            sc = SpectralClustering()
            gap, reference_inertia, ondata_inertia = compute_gap(sc, X, k_max=k_max, reference=reference, metric=metric, n_boots=n_boots)
            gap_statistic[cre_line] = [gap, reference_inertia, ondata_inertia]
        save_clustering_results(gap_statistic, filename_string=gap_filename, path=save_dir)
    return gap_statistic


def load_eigengap(glm_version, feature_matrix, cell_metadata, save_dir=None, k_max=25):
    """
           if eigengap values were computed and file exists in save_dir, load it
           otherwise run get_eigenDecomposition for a range of 1 to k_max clusters
           returns dictionary of eigengap for each cre line = [nb_clusters, eigenvalues, eigenvectors]
           # this doesnt actually take too long, so might not be a huge need to save files besides records
           """
    eigengap_filename = 'eigengap_' + glm_version + '_' + 'kmax' + str(k_max) + '.pkl'
    eigengap_path = os.path.join(save_dir, eigengap_filename)
    if os.path.exists(eigengap_path):
        print('loading eigengap values scores from', eigengap_path)
        with open(eigengap_path, 'rb') as f:
            eigengap = pickle.load(f)
            f.close()
        print('done.')
    else:
        eigengap = {}
        for cre_line in get_cre_lines(cell_metadata):
            feature_matrix_cre = get_feature_matrix_for_cre_line(feature_matrix, cell_metadata, cre_line)
            X = feature_matrix_cre.values
            sc = SpectralClustering(2)  # N of clusters does not impact affinity matrix
            # but you can obtain affinity matrix only after fitting, thus some N of clusters must be provided.
            sc.fit(X)
            A = sc.affinity_matrix_
            eigenvalues, eigenvectors, nb_clusters = get_eigenDecomposition(A, max_n_clusters=k_max)
            eigengap[cre_line] = [nb_clusters, eigenvalues, eigenvectors]
        save_clustering_results(eigengap, filename_string=eigengap_filename, path=save_dir)
    return eigengap


### Clustering analysis ###

def get_labels_for_coclust_matrix(X, model=SpectralClustering, nboot=np.arange(100), n_clusters=8):
    '''

    :param X: (ndarray) data, n observations by n features
    :param model: (clustering object) default =  SpectralClustering; clustering method to use. Object must be initialized.
    :param nboot: (list or an array) default = 100, number of clustering repeats
    :param n_clusters: (num) default = 8
    ___________
    :return: labels: matrix of labels, n repeats by n observations
    '''
    if model is SpectralClustering:
        model = model()
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
    # model = model()
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


### determining cluster labels & assessing cluster correlation ###

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


def get_cluster_density(df_dropouts, labels_df, label_col='cluster_id', use_spearmanr=True):
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


def add_within_cluster_corr_to_cluster_meta(feature_matrix, cluster_meta, use_spearmanr=True):
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
    generate_data = False
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
        cluster_meta = add_within_cluster_corr_to_cluster_meta(feature_matrix, cluster_meta, use_spearmanr=True)
        # save
        print('saving cluster_meta to', cluster_meta_filepath)
        cluster_meta.to_hdf(cluster_meta_filepath, key='df')
    return cluster_meta

### run clustering on all cre lines ###


def run_all_cre_clustering(feature_matrix, cells_table, n_clusters, save_dir, folder):
    cluster_meta_save_path = os.path.join(save_dir, 'cluster_meta_n_' + str(n_clusters) + '_clusters.h5')

    # if clustering output exists, load it
    if os.path.exists(cluster_meta_save_path):
        cluster_meta = pd.read_hdf(cluster_meta_save_path, key='df')
        # merge in cell metadata
        cell_metadata = get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)
        cell_metadata = cell_metadata.drop(columns=['ophys_experiment_id', 'cre_line'])
        cluster_meta = cluster_meta.merge(cell_metadata.reset_index(), on='cell_specimen_id')
        cluster_meta = cluster_meta.set_index('cell_specimen_id')
    # otherwise run it and save it
    else:
        # run spectral clustering and get co-clustering matrix
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering()
        X = feature_matrix.values
        m = get_coClust_matrix(X=X, n_clusters=n_clusters, model=sc, nboot=np.arange(100))
        # make co-clustering matrix a dataframe with cell_specimen_ids as indices and columns
        coclustering_df = pd.DataFrame(data=m, index=feature_matrix.index, columns=feature_matrix.index)

        # save co-clustering matrix
        coclust_save_path = os.path.join(save_dir, 'coclustering_matrix_n_' + str(n_clusters) + '_clusters.h5')
        coclustering_df.to_hdf(coclust_save_path, key='df', format='table')

        # run agglomerative clustering on co-clustering matrix to identify cluster labels
        from sklearn.cluster import AgglomerativeClustering
        X = coclustering_df.values
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                          linkage='average')
        labels = cluster.fit_predict(X)
        cell_specimen_ids = coclustering_df.index.values
        # make dictionary with labels for each cell specimen ID in this cre line
        labels_dict = {'labels': labels, 'cell_specimen_id': cell_specimen_ids}
        # turn it into a dataframe
        labels_df = pd.DataFrame(data=labels_dict, columns=['labels', 'cell_specimen_id'])
        # get new cluster_ids based on size of clusters and add to labels_df
        cluster_size_order = labels_df['labels'].value_counts().index.values
        # translate between original labels and new IDS based on cluster size
        labels_df['cluster_id'] = [np.where(cluster_size_order == label)[0][0] for label in labels_df.labels.values]
        # concatenate with df for all cre lines
        cluster_labels = labels_df

        # add metadata to cluster labels
        cell_metadata = get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)

        cluster_meta = cluster_labels[['cell_specimen_id', 'cluster_id', 'labels']].merge(cell_metadata,
                                                                                          on='cell_specimen_id')
        cluster_meta = cluster_meta.set_index('cell_specimen_id')
        # annotate & clean cluster metadata
        cluster_meta = clean_cluster_meta(cluster_meta)  # drop cluster IDs with fewer than 5 cells in them
        cluster_meta['original_cluster_id'] = cluster_meta.cluster_id

        # plot coclustering matrix - need to hack it since it assumes cre lines
        coclustering_dict = {}
        coclustering_dict['all'] = coclustering_df
        cluster_meta_tmp = cluster_meta.copy()
        cluster_meta_tmp['cre_line'] = 'all'
        plotting.plot_coclustering_matrix_sorted_by_cluster_size(coclustering_dict, cluster_meta_tmp, cre_line='all',
                                                                 save_dir=save_dir, folder=folder,
                                                                 suffix='_' + str(n_clusters) + '_clusters', ax=None)

        # add within cluster correlation
        cluster_meta = add_within_cluster_corr_to_cluster_meta(feature_matrix, cluster_meta,
                                                               use_spearmanr=True)

        # plot within cluster correlations
        plotting.plot_within_cluster_correlations(cluster_meta, sort_order=None, spearman=True,
                                                  suffix='_' + str(n_clusters) + '_clusters',
                                                  save_dir=save_dir, folder=folder, ax=None)

        # # compute the within and acorss cluster correlations for each cell and plot the results for each cluster
        # correlations_df = compute_within_and_across_cluster_correlations(feature_matrix, cluster_meta, save_dir=save_dir)
        # correlations_summary = correlations_df.groupby(['cell_specimen_id']).apply(get_summary_of_within_across_cluster_correlations)
        # correlations_summary.to_hdf(os.path.join(save_dir, 'correlations_summary_'+str(n_clusters)+'.h5'), key='df')
        # plot_within_across_cluster_correlations(correlations_summary, save_dir=save_dir, folder=folder)

        # save clustering results
        cluster_meta_save_path = os.path.join(save_dir, 'cluster_meta_n_' + str(n_clusters) + '_clusters.h5')
        cluster_data = cluster_meta.reset_index()[
            ['cell_specimen_id', 'ophys_experiment_id', 'cre_line', 'cluster_id', 'labels',
             'within_cluster_correlation_p', 'within_cluster_correlation_s']]
        cluster_data.to_hdf(cluster_meta_save_path, key='df', format='table')

    # if cluster_id is zero indexed, add one to it
    if 0 in cluster_meta.cluster_id.unique():
        cluster_meta['cluster_id'] = [cluster_id + 1 for cluster_id in cluster_meta.cluster_id.values]

    return cluster_meta


def compute_within_and_across_cluster_correlations(feature_matrix, cluster_meta, save_dir=None):
    '''
    Computes the pearsons correlation coefficient of each cell's dropout scores with the average dropout score of each cluster
    Creates a dataframe containing the correlation values for each cell_specimen_id with each cluster, including columns for the cell_specimen_id,
    the cluster_id the cell belongs to, the cluster_id being compared, the pearson correlation, and a boolean for whether the cluster being compared is the cell's own cluster

    feature_matrix: dataframe where rows are unique cell_specimen_ids and columns are dropout scores across experience levels
    cluster_meta: dataframe with cell_specimen_ids as index and cluster metadata as cols, must include 'cluster_id'
    Returns
        correlations_df: dataframe with correlation values for each cell compared to each cluster
    '''
    correlation_list = []
    clusters = cluster_meta['cluster_id'].unique()
    n_clusters = len(clusters)
    for cluster_id in clusters:
        cluster_csids = cluster_meta[cluster_meta.cluster_id == cluster_id].index.values
        # get average dropout scores for this cluster
        cluster_dropouts = feature_matrix.loc[cluster_csids]
        cluster_mean = cluster_dropouts.mean().values
        # compute correlation of each cell in this cluster to the average
        for cell_specimen_id in cluster_csids:
            this_cell_coding_scores = feature_matrix.loc[cell_specimen_id].values
            # loop through every other cluster and compute the correlation of this cell's coding scores with the average coding for each cluster
            for cluster_id_to_compare in clusters:
                if cluster_id == cluster_id_to_compare: 
                    within_cluster = True
                else: 
                    within_cluster = False
                cluster_csids = cluster_meta[cluster_meta.cluster_id == cluster_id_to_compare].index.values
                # get average dropout scores for this cluster
                cluster_dropouts = feature_matrix.loc[cluster_csids]
                cluster_mean = cluster_dropouts.mean().values
                corr_coeff_p = np.corrcoef(cluster_mean, this_cell_coding_scores)[0][1]
                correlation_list.append([cell_specimen_id, cluster_id, cluster_id_to_compare, corr_coeff_p, within_cluster, n_clusters])
    correlations_df = pd.DataFrame(correlation_list, columns=['cell_specimen_id', 'cluster_this_cell_belongs_to', 'cluster_id_to_compare', 'pearson_correlation', 'within_cluster', 'total_n_clusters'])
    
    if save_dir: 
        correlations_df.to_hdf(os.path.join(save_dir, 'within_across_cluster_correlations_'+str(n_clusters)+'.h5'), key='df')
    return correlations_df


def get_summary_of_within_across_cluster_correlations(group):
    within_cluster_correlation = group[group.within_cluster==True].pearson_correlation.values[0]
    across_cluster_correlation = np.nanmean(group[group.within_cluster==False].pearson_correlation.values)
    correlation_diff = within_cluster_correlation - across_cluster_correlation
    correlation_ratio = within_cluster_correlation/across_cluster_correlation
    return pd.Series({'cluster_id': int(group.cluster_this_cell_belongs_to.values[0]), 
                    'within_cluster_correlation': within_cluster_correlation,
                    'across_cluster_correlation': across_cluster_correlation,
                    'correlation_diff': correlation_diff, 
                    'correlation_ratio': correlation_ratio})

### older functions for computing area / depth frequencies across clusters, by MG ###

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


def get_location_fractions(cluster_meta):
    """
    for each location, compute percent of cells belonging to each cluster
    """

    cre_lines = np.sort(cluster_meta.cre_line.unique())
    locations = np.sort(cluster_meta.location.unique())[::-1]
    location_fractions = pd.DataFrame()
    for cre_line in cre_lines:
        for location in locations:
            cells = cluster_meta[(cluster_meta.location == location) & (cluster_meta.cre_line == cre_line)]
            total = len(cells)
            cluster_cells = cells.groupby('cluster_id').count()[['labels']].rename(columns={'labels': 'n_cells'})
            fraction = cluster_cells.n_cells / total

            cluster_cells['fraction'] = fraction
            cluster_cells = cluster_cells.reset_index()
            cluster_cells['cre_line'] = cre_line
            cluster_cells['location'] = location
            location_fractions = pd.concat([location_fractions, cluster_cells])

    return location_fractions


def get_fraction_cells_per_cluster_per_group(cluster_meta, col_to_group='mouse_id'):
    """
    counts the number of cells a given condition, and the number of cells in each cluster for that given condition,
    then computes the proportion of cells in each cluster for the given condition
    col_to_group: can be 'cre_line', 'mouse_id', etc
    """
    # get number of cells in each group defined by col_to_group
    n_cells_group = cluster_meta.groupby([col_to_group]).count()[['cluster_id']].reset_index().rename(
        columns={'cluster_id': 'n_cells_group'})
    # get number of cells in each cluster for each group
    n_cells_per_cluster = cluster_meta.groupby([col_to_group, 'cluster_id']).count()[
        ['ophys_experiment_id']].reset_index().rename(columns={'ophys_experiment_id': 'n_cells_cluster'})
    # add n cells per cluster per group to total number of cells per group and compute the fraction per cluster in each group
    n_cells_per_cluster = n_cells_per_cluster.merge(n_cells_group, on=col_to_group)
    n_cells_per_cluster[
        'fraction_per_cluster'] = n_cells_per_cluster.n_cells_cluster / n_cells_per_cluster.n_cells_group

    return n_cells_per_cluster


### area / depth proportion functions based on Alex's GLM repo code ###

# chi_squared test
def get_stats_table(cre_original_cluster_sizes, shuffle_type_cluster_sizes, cre_lines=None,
                    shuffle_types=None, test='chi_squared', add_hochberg_correction=True):
    columns = ['shuffle_type', 'cre_line', 'cluster_id', 'cluster_size',
               'shuffle_mean', test + '_pvalue', 'significant']
    # create dataframe to collect stats
    cluster_statistics_df = pd.DataFrame(columns=columns)

    if shuffle_types is None:
        shuffle_types = shuffle_type_cluster_sizes.keys()
    if cre_lines is None:
        cre_lines = shuffle_type_cluster_sizes[shuffle_types[0]].keys()

    index = 0
    for shuffle_type in shuffle_types:
        for cre_line in cre_lines:
            shuffle_clusters_df = pd.DataFrame(shuffle_type_cluster_sizes[shuffle_type][cre_line]).mean(axis=0)
            # original cluster sizes
            original_df = cre_original_cluster_sizes[cre_line]

            for cluster_id in original_df.index:
                f_observed = [shuffle_clusters_df.loc[cluster_id], original_df.sum() - shuffle_clusters_df.loc[cluster_id]]
                f_expected = [original_df.loc[cluster_id], original_df.sum() - original_df.loc[cluster_id]]

                out = chisquare(f_observed, f_expected)

                data = [shuffle_type, cre_line, cluster_id, original_df.loc[cluster_id],
                        shuffle_clusters_df.loc[cluster_id], out.pvalue, out.pvalue <= 0.05]

                for c, column in enumerate(columns):
                    cluster_statistics_df.at[index, column] = data[c]
                    index = index + 1

    if add_hochberg_correction is True:
        cluster_statistics_df = glm_clust.add_hochberg_correction(cluster_statistics_df, test=test)

    return cluster_statistics_df


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

    taken from vba_glm.glm_clust()
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


def compute_cluster_proportion_stats(cluster_meta, cre_line, location='layer'):
    '''
        Performs chi-squared tests to asses whether the observed cell counts in each location differ
        significantly from the expected number of cells per location based on the overall size of that cluster.

        replicated from glm_clust.stats()
    '''

    from scipy.stats import chisquare

    df = cluster_meta.reset_index()
    cre = cre_line
    locations = np.sort(df[location].unique())
    # compute cell counts in each area/cluster
    table = df.query('cre_line == @cre').groupby(['cluster_id', location])['cell_specimen_id'].count().unstack()
    table = table[locations]
    table = table.fillna(value=0)

    # compute proportion
    for loc in locations:
        table[loc] = table[loc] / table[loc].sum()

    # get average for each cluster
    table['mean'] = table.mean(axis=1)

    # second table of cell counts in each area/cluster
    table2 = df.query('cre_line == @cre').groupby(['cluster_id', location])['cell_specimen_id'].count().unstack()
    table2 = table2[locations]
    table2 = table2.fillna(value=0)

    # compute estimated frequency of cells based on average fraction for each cluster
    chance_columns = []
    for loc in locations:
        chance_columns.append(loc + '_chance_count')
        table2[loc + '_chance_count'] = table2[loc].sum() * table['mean']
    # n_cells you would get if they were proportional to overall size of cluster

    # perform chi-squared test
    for index in table2.index.values:
        f = table2.loc[index][locations].values
        f_expected = table2.loc[index][chance_columns].values
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


### new functions by MG to compute # and % of cells across locations ###

def get_fraction_cells_per_area_depth(cluster_meta):
    """
    computes the fraction of cells per area and depth in a given cluster (hard-coded to use 'targeted_structure' and 'layer')
    relative to the total number of cells in that area and depth in that cre line
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


def get_fraction_cells_per_cluster(cluster_meta):
    """
    creates a dataframe with cre lines and cluster ids as indices
    values are the fraction of cells in each cluster
    relative to the total number of cells in that cre line,
    provided as a single column 'fraction_cells_cluster'
    """
    # get fraction cells per area and depth per cluster
    # frequency will be the fraction of cells in each cluster for a given location
    # relative to the total number of cells in that location for that cre line

    # total n cells in each cre line
    n_cells_per_cre = cluster_meta.reset_index().groupby(['cre_line'])['cell_specimen_id'].count()
    # number of cells in each cluster in each cre line
    n_cells_per_cluster = cluster_meta.reset_index().groupby(['cre_line', 'cluster_id'])['cell_specimen_id'].count()
    n_cells_per_cluster = n_cells_per_cluster.fillna(value=0)
    # n_cells per cluster relative to n cells per cre
    fraction_cells_per_cluster = n_cells_per_cluster / n_cells_per_cre
    # create dataframe and clean it
    fraction_cells_per_cluster = pd.DataFrame(fraction_cells_per_cluster)
    fraction_cells_per_cluster = fraction_cells_per_cluster.rename(columns={'cell_specimen_id': 'fraction_cells_cluster'})
    return fraction_cells_per_cluster


def get_fraction_cells_per_location(cluster_meta, location='layer'):
    """
    creates a dataframe with cre lines & cluster IDs as indices, and locations as columns

    values are the fraction of cells in a given location (i.e. area/depth) in a given cluster
    relative to the total number of cells in that location (i.e. area/depth) in that cre line
    i.e. 'out of all the cells in this area and depth, how many are in this cluster? '

    location can be any categorical column in cluster_meta,
     typically one of: 'layer', 'targeted_structure', 'binned_depth', 'area_binned_depth' etc.

    """
    # get fraction cells per area and depth per cluster
    # frequency will be the fraction of cells in each cluster for a given location
    # relative to the total number of cells in that location for that cre line
    locations = np.sort(cluster_meta[location].unique())
    # get n_cells for each location in a given cre line
    n_cells_per_location = cluster_meta.reset_index().groupby(['cre_line', location])['cell_specimen_id'].count().unstack()
    # get n_cells in each cluster for each location in a given cre linecre line
    n_cells_per_location_per_cluster = cluster_meta.reset_index().groupby(['cre_line', 'cluster_id', location])['cell_specimen_id'].count().unstack()
    n_cells_per_location_per_cluster = n_cells_per_location_per_cluster.fillna(value=0)
    # n_cells in each cluster for a given location divided by total n_cells in that location
    fraction_cells_per_location = n_cells_per_location_per_cluster / n_cells_per_location
    fraction_cells_per_location = fraction_cells_per_location[locations]  # sort by location order for convenience
    return fraction_cells_per_location


def get_fraction_cells_relative_to_cluster_size(cluster_meta, location='layer'):
    """
    computes the fraction of cells in a given location (i.e. area/depth) in a given cluster
    relative to the total number of cells in that location (i.e. area/depth) in that cre line,
    normalized to the overall size of that cluster within that cre line
    i.e. 'how many times more (or less) cells do we see in this location relative to the expected size of the cluster '

    location can be any categorical column in cluster_meta,
     typically one of: 'layer', 'targeted_structure', 'binned_depth', 'area_binned_depth' etc.
    """
    fraction_cells_per_location = get_fraction_cells_per_location(cluster_meta, location)
    fraction_cells_per_cluster = get_fraction_cells_per_cluster(cluster_meta)
    # combine the two so we can devide fraction cells per location by fraction per cluster
    fraction_cells = fraction_cells_per_location.merge(fraction_cells_per_cluster, on=['cre_line', 'cluster_id'])
    # get average of proportions across locations (i.e. cluster average proportion) and add to main df
    fraction_cells_per_location['average_of_locations'] = fraction_cells_per_location.mean(axis=1)
    fraction_cells['average_of_locations'] = fraction_cells_per_location['average_of_locations']
    # compute fraction of cells in each location in each cluster relative to overall fraction of cells per cluster
    locations = np.sort(cluster_meta[location].unique())
    for loc in locations:
        fraction_cells['fraction_of_cluster_size_' + loc] = fraction_cells[loc] / fraction_cells.fraction_cells_cluster
        fraction_cells = fraction_cells.rename(columns={loc: 'fraction_cells_' + loc})
    return fraction_cells


def get_n_cells_per_location_for_clusters(cluster_meta, location='layer'):
    """
    creates table cre line & cluster IDs as indices with unqique columsn for:
        the number of cells in a given location (i.e. area/depth) in a given cluster/cre line
        the total number of cells in that location (i.e. area/depth) in that cre line
        the number of cells in each cluster in each cre line
        the total number of cells in that cre line

    location can be any categorical column in cluster_meta,
     typically one of: 'layer', 'targeted_structure', 'binned_depth', 'area_binned_depth' etc.
    """
    # get fraction cells per area and depth per cluster
    # frequency will be the fraction of cells in each cluster for a given location
    # relative to the total number of cells in that location for that cre line
    locations = np.sort(cluster_meta[location].unique())
    # get number of cells per cluster per cre line
    n_cells_per_cluster = cluster_meta.reset_index().groupby(['cre_line', 'cluster_id']).count()[['cell_specimen_id']]
    n_cells_per_cluster = n_cells_per_cluster.rename(columns={'cell_specimen_id': 'n_cells_cluster'})
    # get total number of cells in each location for each cre line
    n_cells_per_location = cluster_meta.reset_index().groupby(['cre_line', location])['cell_specimen_id'].count().unstack()
    # get number of cells in each location for each cluster in each cre line
    n_cells_per_location_per_cluster = cluster_meta.reset_index().groupby(['cre_line', 'cluster_id', location])['cell_specimen_id'].count().unstack()
    n_cells_per_location_per_cluster = n_cells_per_location_per_cluster.fillna(value=0)
    # relabel to include location in column name
    for loc in locations:
        n_cells_per_location = n_cells_per_location.rename(columns={loc: 'n_cells_total_' + loc})
        n_cells_per_location_per_cluster = n_cells_per_location_per_cluster.rename(columns={loc: 'n_cells_' + loc})
    # create final df with all counts
    n_cells = n_cells_per_location_per_cluster.merge(n_cells_per_cluster, on=['cre_line', 'cluster_id'])
    n_cells = n_cells.reset_index().merge(n_cells_per_location, on='cre_line', how='left')

    n_cells.columns.name = None  # get rid of residual column index name
    n_cells = n_cells.set_index(['cre_line', 'cluster_id'])
    return n_cells

### THIS IS THE FINAL FUNCTION TO USE FOR GETTING AREA / DEPTH DISTRIBUTIONS FOR CLUSTERS ###
#### it compures all the possible metrics, including stats based on Alex's code ###


def get_cluster_proportion_stats_for_locations(cluster_meta, location='layer'):
    """
    creates table with cre line, cluster ID, and locations as indices, with unique columns for:
        the number of cells in a given location (i.e. area/depth) in a given cluster & cre line
        the total number of cells in that location (i.e. area/depth) in that cre line
        the fraction of cells in that location & cluster relative to the total number of cells in that location
        the number of cells in each cluster in each cre line
        the total number of cells in that cre line
        the fraction of cells in each cluster
        the fraction of cells in a given location relative to the overall size of the cluster
        the number of cells in a given location expected by chance based on the size of the cluster & total # cells in that location
        the p-value comparing number of cells in each location to the number expected based on chance for each cluster
        Boolean columns for whether p-val is significant, with and without correction for multiple comparisons

    cluster_meta must have one row per cell_specimen_id, and include columns for 'cre_line', 'cluster_id', and location column
    location can be any categorical column in cluster_meta,
     typically one of: 'layer', 'targeted_structure', 'binned_depth', 'area_binned_depth' etc.
    """

    locations = np.sort(cluster_meta[location].unique())

    # get number of cells in each location for each cluster in each cre line
    n_cells_per_location_per_cluster = cluster_meta.reset_index().groupby(['cre_line', 'cluster_id', location])[
        'cell_specimen_id'].count().to_frame(name='n_cells_location')
    # get total number of cells in each location for each cre line
    # n_cells_per_location = cluster_meta.reset_index().groupby(['cre_line', location])['cell_specimen_id'].count().unstack()
    n_cells_per_location = cluster_meta.reset_index().groupby(['cre_line', location])[
        'cell_specimen_id'].count().to_frame(name='n_cells_location_total')
    # merge them and compute fraction of cells in each cluster for each location
    n_cells_table = n_cells_per_location_per_cluster.copy()
    n_cells_table = n_cells_table.reset_index().merge(n_cells_per_location.reset_index(), on=['cre_line', location],
                                                      how='left')
    n_cells_table['fraction_cells_location'] = n_cells_table['n_cells_location'] / n_cells_table[
        'n_cells_location_total']

    # get number of cells per cluster per cre line
    n_cells_per_cluster = cluster_meta.reset_index().groupby(['cre_line', 'cluster_id']).count()[['cell_specimen_id']]
    n_cells_per_cluster = n_cells_per_cluster.rename(columns={'cell_specimen_id': 'n_cells_cluster'})
    # number of cells per cre line
    n_cells_per_cre = cluster_meta.reset_index().groupby(['cre_line']).count()[['cell_specimen_id']]
    n_cells_per_cre = n_cells_per_cre.rename(columns={'cell_specimen_id': 'n_cells_cre'})
    # fraction cells per cluster per cre line
    n_cells_table = n_cells_table.merge(n_cells_per_cluster.reset_index(), on=['cre_line', 'cluster_id'], how='left')
    n_cells_table = n_cells_table.merge(n_cells_per_cre.reset_index(), on=['cre_line'], how='left')
    n_cells_table['fraction_cells_cluster'] = n_cells_table['n_cells_cluster'] / n_cells_table['n_cells_cre']
    # compute the proportion of cells in a given location for each cluster/cre relative to the proportion of cells in that cluster/cre
    n_cells_table['fraction_of_cluster_size'] = n_cells_table['fraction_cells_location'] / n_cells_table[
        'fraction_cells_cluster']
    # how many cells would you expect per location by chance, assuming fraction of cells in each location is equal to overall size of cluster
    # i.e. null hypothesis that cells are evenly distributed across locations within each cluster (accounting for bias in overall # cells per location)
    n_cells_table['n_cells_chance_location'] = n_cells_table['fraction_cells_cluster'] * n_cells_table[
        'n_cells_location_total']

    # STATS
    # make sure index is properly set before doing stats
    n_cells_table = n_cells_table.set_index(['cre_line', 'cluster_id', location])
    # chi square test for significance of actual cell counts per location/cluster vs chance level based on cluster size
    n_cells_table = add_stats_for_location_comparison_to_cluster_proportion_stats(n_cells_table)
    # # make sure index is properly set before returning
    # n_cells_table = n_cells_table.set_index(['cre_line', 'cluster_id', location])
    return n_cells_table


def add_stats_for_location_comparison_to_cluster_proportion_stats(n_cells_table):
    '''
    Takes in a table with cell counts and proportions for each location, cluster, and cre line combination,
    and performs a chi-square test with Benjamini Hochberg correction to determine if the
    number of cells in each location for a given cluster is significantly different
    than the number expected by chance based on the overall size of the cluster & # cells in each location.
    Chance level for a given location in a given cluster is n_total_cells_for_location * fraction_cells_in_cluster

    input dataframe must contain columns 'n_cells_location', and 'n_cells_chance_location',
        and be a multi-index with [(cre_line, cluster_id, location)] as indices

    returns input table with columns 'pvalue', 'significant', 'imq', and 'bh_significant',
        computed across locations within each cluster & cre line
    '''

    from scipy.stats import chisquare

    cre_lines = n_cells_table.index.get_level_values(level=0).unique().values
    # iterate through clusters for each cre and compute stats for comparison across locations
    for cre_line in cre_lines:
        # get cluster IDs for this cre
        cluster_ids = n_cells_table.loc[(cre_line)].index.get_level_values(level=0).unique().values
        for cluster_id in cluster_ids:
            # get the actual n_cells and chance level n_cells for both locations for this cluster
            f = n_cells_table.loc[(cre_line, cluster_id)]['n_cells_location']
            f_expected = n_cells_table.loc[(cre_line, cluster_id)]['n_cells_chance_location']
            # ask if the actual frequency values are significantly different than expected frequencies
            if len(f) > 1:  # needs to be at least 2 locations to compare
                out = chisquare(f, f_expected)
                n_cells_table.loc[(cre_line, cluster_id), 'pvalue'] = out.pvalue
                n_cells_table.loc[(cre_line, cluster_id), 'significant'] = out.pvalue < 0.05
            else:
                n_cells_table.loc[(cre_line, cluster_id), 'pvalue'] = np.NaN
                n_cells_table.loc[(cre_line, cluster_id), 'significant'] = np.NaN

        # Benjamini Hochberg correction
        # isolate data just for one cre line and remove extra rows per cluster (p-values are the same for each location within a cluster)
        cre_data = n_cells_table.loc[(cre_line)].reset_index().drop_duplicates(subset=['cluster_id'])
        # sort by p-value within this cre line
        cre_data = cre_data.sort_values(by='pvalue')
        # compute the corrected pvalue based on the rank of each test
        cre_data['imq'] = cre_data.index.values / len(cre_data) * 0.05
        # Find the largest pvalue less than its corrected pvalue
        # all tests above that are significant
        cre_data['bh_significant'] = False
        passing_tests = cre_data[cre_data['pvalue'] < cre_data['imq']]
        if len(passing_tests) > 0:
            last_index = cre_data[cre_data['pvalue'] < cre_data['imq']].tail(1).index.values[0]
            cre_data.at[last_index, 'bh_significant'] = True
            cre_data.at[0:last_index, 'bh_significant'] = True
        # reset order of table and return
        cre_data = cre_data.sort_values(by='cluster_id').set_index('cluster_id')
        for cluster_id in cre_data.index.values:
            n_cells_table.loc[(cre_line, cluster_id), 'imq'] = cre_data.loc[cluster_id, 'imq']
            n_cells_table.loc[(cre_line, cluster_id), 'bh_significant'] = cre_data.loc[cluster_id, 'bh_significant']
    return n_cells_table


def get_cre_line_stats_for_locations(cell_metadata, location='layer'):
    """
    creates table with the number and fraction of cells in a given location (i.e. area/depth) in a given cre line

    cell_metadata: dataframe, containing one row per cell_specimen_id
                typically the ophys_cells_table after droping duplicate cells or the cluster_meta table
    location: any categorical column in cell_metadata,
                typically one of: 'layer', 'targeted_structure', 'binned_depth', 'area_binned_depth' etc.
    """
    # get total number of cells in each location for each cre line
    n_cells_per_location = cell_metadata.reset_index().groupby(['cre_line', location])[
        'cell_specimen_id'].count().to_frame(name='n_cells_location')
    # total number of cells per cre line
    n_cells_per_cre = cell_metadata.reset_index().groupby(['cre_line']).count()[['cell_specimen_id']]
    n_cells_per_cre = n_cells_per_cre.rename(columns={'cell_specimen_id': 'n_cells_total'})
    # fraction cells per cluster per cre line
    cre_stats = n_cells_per_location.reset_index().merge(n_cells_per_cre.reset_index(), on=['cre_line'], how='left')
    cre_stats['fraction_cells_location'] = cre_stats['n_cells_location'] / cre_stats['n_cells_total']
    cre_stats = cre_stats.set_index(['cre_line', location])
    return cre_stats

# metrics computed on GLM coding scores for all cells (does not require matching)


def get_coding_metrics_unmatched(index_dropouts, index_value, index_name):
    """
    index_dropouts: dataframe, the dropout scores for a single cell, or average dropouts across cells
                                formatted with rows = experience levels, cols = regressors
    index_value: int, should be the cell_specimen_id corresponding to the input dropouts
    index_name: str, is the column name corresponding to index_value, i.e. 'cell_specimen_id'

    Returns
    stats: dataframe, containing one row for each cell_specimen_id , with metrics as columns
                        metrics are computed within each experience level
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
    stats.loc[index_value, 'max_coding_score'] = index_dropouts.loc[dominant_experience_level][dominant_feature]
    stats.loc[index_value, 'max_image_coding_score'] = index_dropouts.loc[dominant_experience_level]['images']
    for experience_level in index_dropouts.index.values:
        stats.loc[index_value, 'image_coding_' + experience_level] = index_dropouts.loc[experience_level]['images']
    stats.loc[index_value, 'feature_selectivity'] = feature_selectivity
    stats.loc[index_value, 'experience_selectivity'] = experience_selectivity
    # get experience modulation indices
    row = index_dropouts[dominant_feature]
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
    stats.loc[index_value, 'feature_sel_within_session'] = feature_sel_within_session
    return stats


def get_cell_metrics_unmatched_cells(results_pivoted):
    """
    computes metrics on dropout scores for each cell, including experience level and feature selectivity
    does not require cells to be matched, does not add cluster info
    """
    features = get_feature_labels_for_clustering()
    cell_metrics = pd.DataFrame()
    for i, cell_specimen_id in enumerate(results_pivoted.cell_specimen_id.values):
        cell_dropouts = results_pivoted[results_pivoted.cell_specimen_id == cell_specimen_id].groupby('experience_level').mean()[features]
        stats = get_coding_metrics_unmatched(index_dropouts=cell_dropouts, index_value=cell_specimen_id, index_name='cell_specimen_id')
        cell_metrics = pd.concat([cell_metrics, stats], sort=False)
    return cell_metrics


### metrics computed on GLM coding scores for cells matched across sessions ###

def get_coding_metrics(index_dropouts, index_value, index_name):
    """
    Computes a variety of metrics on dropout scores for some condition (ex: one cell or one cluster)
    including selectivity across features or experience levels, within and across sessions

    index_dropouts: dataframe, the dropout scores for a single cell, or average dropouts across cells
                                formatted with rows = experience levels, cols = regressors
    index_value: int, should be the cluster_id or cell_specimen_id corresponding to the input dropouts
    index_name: str, is the column name corresponding to index_value, i.e. 'cluster_id' or 'cell_specimen_id'

    Returns
    stats: dataframe, containing one row for each cell_specimen_id or cluster_id, with metrics as columns
                        metrics can be computed within or across experience levels,
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
    stats.loc[index_value, 'max_coding_score'] = index_dropouts.loc[dominant_experience_level][dominant_feature]
    stats.loc[index_value, 'max_image_coding_score'] = index_dropouts.loc[dominant_experience_level]['all-images']
    for experience_level in index_dropouts.index.values:
        stats.loc[index_value, 'image_coding_' + experience_level] = index_dropouts.loc[experience_level]['all-images']
    stats.loc[index_value, 'feature_selectivity'] = feature_selectivity
    stats.loc[index_value, 'experience_selectivity'] = experience_selectivity
    # get experience modulation indices
    row = index_dropouts[dominant_feature]
    # overall experience modulation is how strong novel session coding is relative to average of Familiar and Novel >1
    # this makes sense under the assumption that Novel images becomes Familiar again rapidly in the Novel >1 session
    # if we wanted consider either novel session, could use novel_dropout = np.amax([row['Novel 1'], row['Novel >1']])
    exp_mod = (row['Novel'] - (np.mean([row['Familiar'], row['Novel +']]))) / (row['Novel'] + (np.mean([row['Familiar'], row['Novel +']])))
    stats.loc[index_value, 'experience_modulation'] = exp_mod
    # direction of exp mod is whether coding is stronger for familiar or novel
    # if we wanted consider either novel session, could use novel_dropout = np.amax([row['Novel 1'], row['Novel >1']])
    exp_mod_direction = (row['Novel'] - row['Familiar']) / (row['Novel'] + row['Familiar'])
    # persistence of exp mod is whether coding stays similar after repetition of novel session
#     exp_mod_persistence = (row['Novel >1']-row['Novel 1'])/(row['Novel >1']+row['Novel 1'])
    exp_mod_persistence = row['Novel +'] / row['Novel']  # make it an index
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


def get_cell_metrics(cluster_meta, results_pivoted):
    """
    computes metrics on dropout scores for each cell, including experience level and feature selectivity
    """
    cell_metrics = pd.DataFrame()
    for i, cell_specimen_id in enumerate(cluster_meta.index.values):
        cell_dropouts = results_pivoted[results_pivoted.cell_specimen_id == cell_specimen_id].groupby('experience_level').mean()[get_features_for_clustering()]
        stats = get_coding_metrics(index_dropouts=cell_dropouts, index_value=cell_specimen_id, index_name='cell_specimen_id')
        cell_metrics = pd.concat([cell_metrics, stats], sort=False)
    cell_metrics = cell_metrics.merge(cluster_meta, on='cell_specimen_id')
    return cell_metrics


def get_coding_score_metrics_per_experience_level(cluster_meta, results_pivoted):
    '''
    Takes coding scores from results_pivoted and computes a few additional pieces of information for each cell,
    including which experience level and feature have the max coding score, a column for the max coding score, and
    a boolean for rows where the experience level is the preferred experience level

    Returns
        coding_scores: dataframe with one row per cell per experience level, and columns for the coding scores for each feature,
                        plus information about the preferred feature and experience level for that cell
    '''
    coding_score_metrics = pd.DataFrame()
    for i, cell_specimen_id in enumerate(cluster_meta.index.values):
        cell_scores = results_pivoted[results_pivoted.cell_specimen_id == cell_specimen_id].reset_index().drop(columns='identifier')
        cell_scores_grid = cell_scores.groupby('experience_level').mean()[get_features_for_clustering()]
        # determine which experience level & feature gives largest coding score
        dominant_feature = cell_scores_grid.stack().idxmax()[1]
        dominant_experience_level = cell_scores_grid.stack().idxmax()[0]
        # create column listing the preferred coding score for this cell
        cell_scores['max_coding_score'] = cell_scores_grid.loc[dominant_experience_level, dominant_feature]
        # create columns listing preferred feature & exp level
        cell_scores['dominant_feature'] = dominant_feature
        cell_scores['dominant_experience_level'] = dominant_experience_level
        # create boolean column for rows where the experience level is the one that has the max coding score
        indices = cell_scores[(cell_scores.experience_level == dominant_experience_level)].index.values
        cell_scores['pref_experience_level'] = False
        cell_scores.at[indices, 'pref_experience_level'] = True
        # move cell specimen id and exp level to front of df
        cell_scores.insert(0, 'cell_specimen_id', cell_scores.pop('cell_specimen_id'))
        cell_scores.insert(1, 'experience_level', cell_scores.pop('experience_level'))
        coding_score_metrics = pd.concat([coding_score_metrics, cell_scores])
    return coding_score_metrics


def get_cluster_metrics(cluster_meta, feature_matrix, results_pivoted):
    """
    Computes metrics for each cluster / cre line, using the average coding score for that cluster
    metrics include experience modulation, feature selectivity, etc.

    returns dataframe with one row per cre line & cluster, with unique columns for metrics
    """
    cre_lines = np.sort(cluster_meta.cre_line.unique())
    cell_metrics = get_cell_metrics(cluster_meta, results_pivoted)
    cluster_metrics = pd.DataFrame()
    for cre_line in cre_lines:
        # get cell specimen ids for this cre line
        cre_cell_specimen_ids = cluster_meta[cluster_meta.cre_line == cre_line].index.values
        # get cluster labels dataframe for this cre line
        cre_cluster_ids = cluster_meta.loc[cre_cell_specimen_ids]
        # get unique cluster labels for this cre line
        cluster_labels = np.sort(cre_cluster_ids.cluster_id.unique())
        # limit dropouts df to cells in this cre line
        feature_matrix_cre = feature_matrix.loc[cre_cell_specimen_ids]
        for i, cluster_id in enumerate(cluster_labels):
            # get cell specimen ids in this cluster in this cre line
            this_cluster_csids = cre_cluster_ids[cre_cluster_ids['cluster_id'] == cluster_id].index.values
            # get dropout scores for cells in this cluster in this cre line
            mean_dropout_df = np.abs(feature_matrix_cre.loc[this_cluster_csids].mean().unstack())
            stats = get_coding_metrics(index_dropouts=mean_dropout_df.T, index_value=cluster_id,
                                       index_name='cluster_id')
            fraction_cre = len(this_cluster_csids) / float(len(cre_cell_specimen_ids))
            stats['fraction_cre'] = fraction_cre
            stats['cre_line'] = cre_line
            stats['F_max'] = mean_dropout_df['Familiar'].max()
            stats['N1_max'] = mean_dropout_df['Novel'].max()
            stats['N2_max'] = mean_dropout_df['Novel +'].max()
            stats['abs_max'] = mean_dropout_df.max().max()
            cluster_metrics = pd.concat([cluster_metrics, stats])
    cluster_metrics = cluster_metrics.reset_index()
    n_cells = cell_metrics.groupby(['cre_line', 'cluster_id']).count()[['labels']].rename(
        columns={'labels': 'n_cells_cluster'})
    cluster_metrics = cluster_metrics.merge(n_cells, on=['cre_line', 'cluster_id'])
    cluster_metrics = cluster_metrics.set_index(['cre_line', 'cluster_id'])
    return cluster_metrics


def get_cluster_metrics_all_cre(cluster_meta, feature_matrix, results_pivoted):
    """
    computes metrics for each cluster, including experience modulation, feature selectivity, etc
    """
    # cell_metrics = get_cell_metrics(cluster_meta, results_pivoted)
    cluster_metrics = pd.DataFrame()
    # get cell specimen ids
    cell_specimen_ids = cluster_meta.index.values
    # get unique cluster labels
    cluster_labels = np.sort(cluster_meta.cluster_id.unique())
    for i, cluster_id in enumerate(cluster_labels):
        # get cell specimen ids in this cluster
        this_cluster_csids = cluster_meta[cluster_meta['cluster_id'] == cluster_id].index.values
        # get dropout scores for cells in this cluster in this cre line
        mean_dropout_df = np.abs(feature_matrix.loc[this_cluster_csids].mean().unstack())
        stats = get_coding_metrics(index_dropouts=mean_dropout_df.T, index_value=cluster_id,
                                   index_name='cluster_id')
        fraction_cells = len(this_cluster_csids) / float(len(cell_specimen_ids))
        stats['fraction_cells'] = fraction_cells
        stats['F_max'] = mean_dropout_df['Familiar'].max()
        stats['N1_max'] = mean_dropout_df['Novel'].max()
        stats['N2_max'] = mean_dropout_df['Novel +'].max()
        stats['abs_max'] = mean_dropout_df.max().max()
        cluster_metrics = pd.concat([cluster_metrics, stats])
    cluster_metrics = cluster_metrics.reset_index()
    print(cluster_metrics.keys())
    n_cells = cluster_meta.groupby(['cluster_id']).count()[['labels']].rename(columns={'labels': 'n_cells_cluster'})
    cluster_metrics = cluster_metrics.merge(n_cells, on=['cluster_id'])
    cluster_metrics = cluster_metrics.set_index(['cluster_id'])
    return cluster_metrics


def add_layer_index_to_cluster_metrics(cluster_metrics, n_cells_table):
    '''
    computes fraction cells upper vs lower over sum from n_cells table then appends that value
    to the cluster_metrics table as 'layer_index'

    n_cells_table: dataframe with cre line, cluter ID and layer as indices, columns for fraction cells in that layer, etc
    cluster_metrics: dataframe with cre line & cluster ID as indices, columns for various metrics computed on cluster

    '''

    layer_index = n_cells_table[['fraction_cells_location']].unstack()
    layer_index.columns = layer_index.columns.droplevel(0)
    layer_index['layer_index'] = (layer_index.upper - layer_index.lower) / (layer_index.upper + layer_index.lower)
    layer_index = layer_index.merge(n_cells_table[['bh_significant', 'significant']], on=['cre_line', 'cluster_id'])
    layer_index = layer_index.reset_index().drop_duplicates(subset=['cre_line', 'cluster_id']).set_index(
        ['cre_line', 'cluster_id'])

    cluster_metrics = cluster_metrics.merge(layer_index[['layer_index']], on=['cre_line', 'cluster_id'])
    fraction = n_cells_table[['fraction_cells_cluster']].reset_index().drop_duplicates(subset=['cre_line', 'cluster_id'])
    cluster_metrics = cluster_metrics.merge(fraction, on=['cre_line', 'cluster_id']).set_index(['cre_line', 'cluster_id'])
    cluster_metrics = cluster_metrics.drop(columns=['layer'])

    return cluster_metrics


def define_cluster_types(cluster_metrics):
    """
    defines clusters as non-coding, mixed coding, image, behavioral, omission or task depending on
    max dropout score, degree of selectivity across sessions, and max feature category
    adds column 'cluster_types' to cluster_metrics
    """
    cluster_types = []
    for i in range(len(cluster_metrics)):
        row = cluster_metrics.iloc[i]
        if row.abs_max < 0.1:
            cluster_type = 'non-coding'
        elif row.feature_sel_across_sessions < 0.2:
            cluster_type = 'mixed coding'
        else:
            cluster_type = row.dominant_feature
        cluster_types.append(cluster_type)
    cluster_metrics['cluster_type'] = cluster_types
    return cluster_metrics


def add_cluster_types(location_fractions, cluster_metrics):
    """
    add column to location_fractions indicating the cluster type for each cluster
    cluster_types include ['all-images', 'omissions', 'behavioral', 'task', 'mixed coding', 'non-coding']
    """
    cluster_metrics = define_cluster_types(cluster_metrics)

    # merge location fractions with metrics per cluster
    cs = cluster_metrics[['cre_line', 'cluster_id', 'cluster_type',
                          'dominant_feature', 'dominant_experience_level',
                          'experience_modulation', 'exp_mod_direction', 'exp_mod_persistence']]
    location_fractions = location_fractions.merge(cs, on=['cre_line', 'cluster_id'])
    return location_fractions


def get_cluster_types():
    return ['all-images', 'omissions', 'behavioral', 'task', 'mixed coding', 'non-coding']


def get_cluster_type_color_df():
    """
    creates a dataframe with columns for cluster_type and corresponding cluster_type_color
    """
    cluster_types = get_cluster_types()

    # get feature colors from gvt
    import visual_behavior_glm.GLM_visualization_tools as gvt
    cluster_type_colors = [
        gvt.project_colors()[cluster_type][:3] if cluster_type in list(gvt.project_colors().keys()) else (0.5, 0.5, 0.5)
        for cluster_type in cluster_types]

    cluster_type_color_df = pd.DataFrame(columns=['cluster_type', 'color'])
    cluster_type_color_df['cluster_type'] = cluster_types
    cluster_type_color_df['cluster_type_color'] = cluster_type_colors

    # make mixed and non-coding shades of gray
    index = cluster_type_color_df[cluster_type_color_df.cluster_type == 'non-coding'].index[0]
    cluster_type_color_df.at[index, 'cluster_type_color'] = (0.8, 0.8, 0.8)
    index = cluster_type_color_df[cluster_type_color_df.cluster_type == 'mixed coding'].index[0]
    cluster_type_color_df.at[index, 'cluster_type_color'] = (0.4, 0.4, 0.4)

    return cluster_type_color_df


def get_experience_level_color_df():
    """
    create dataframe with columns for experience_level and exp_level_color
    """
    import pandas as pd
    import visual_behavior.visualization.utils as utils
    colors = utils.get_experience_level_colors()
    experience_levels = utils.get_experience_levels()
    exp_colors = [colors[np.where(np.asarray(experience_levels) == exp_level)[0][0]] for exp_level in experience_levels]

    exp_level_color_df = pd.DataFrame()
    exp_level_color_df['experience_level'] = experience_levels
    exp_level_color_df['exp_level_color'] = exp_colors
    return exp_level_color_df


def get_cluster_fractions_per_location(cluster_meta, cluster_metrics):
    """
    Compute the fraction of cells in each location that belong to each cluster
    Designate a 'cluster_type', based on metrics of feature and experience selectivity
    Add colors for cluster_types and preferred experience levels for plotting
    This function is used in plotting.plot_cluster_proportions_all_locations()

    :param cluster_meta: table of metadata for each cell_specimen_id, including their cluster_id
    :param cluster_metrics: metrics computed based on average coding scores of each cluster (ex: 'experience_modulation', 'dominant_feature', etc)
    :return:
    """
    if 'location' not in cluster_meta.keys():
        cluster_meta = add_location_column(cluster_meta, columns_to_groupby=['targeted_structure', 'layer'])
    # cre_lines = np.sort(cluster_meta.cre_line.unique())
    # get fraction cells per location belonging to each cluster
    location_fractions = get_location_fractions(cluster_meta)
    # add cluster types
    location_fractions = add_cluster_types(location_fractions, cluster_metrics)
    # merge with cluster type colors
    cluster_type_color_df = get_cluster_type_color_df()
    location_fractions = location_fractions.merge(cluster_type_color_df, on='cluster_type')
    # merge with experience level colors
    exp_level_color_df = get_experience_level_color_df()
    exp_level_color_df = exp_level_color_df.rename(columns={'experience_level': 'dominant_experience_level'})
    location_fractions = location_fractions.merge(exp_level_color_df, on='dominant_experience_level')
    # make exp level color gray for non-coding clusters
    non_coding_inds = location_fractions[location_fractions.cluster_type == 'non-coding'].index
    location_fractions.at[non_coding_inds, 'exp_level_color'] = location_fractions.loc[non_coding_inds, 'cluster_type_color']
    # add experience modulation index with color values in a continuous colormap
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('RdBu')
    exp_mod = location_fractions.experience_modulation.values
    exp_mod_normed = ((exp_mod + 1) / 2) * 256
    colors = [cmap(int(np.round(i)))[:3] for i in exp_mod_normed]
    location_fractions['exp_mod_color'] = colors
    return location_fractions


### functions to merge coding score metrics with model free cell metrics ###


def generate_merged_table_of_model_free_metrics(data_type='filtered_events', session_subset='full_session',
                                                inclusion_criteria='platform_experiment_table', save_dir=None):
    '''
    Load table of cell metrics computed on average image, omission, and change responses,
    rename columns to indicate which type of response the metric is for (ex: 'mean_response_omissions'),
    then merge together all types of table to create a unified table of cell response (i.e. model free) metrics

    if file already exists in save_dir, load it, if not, generate and save it

    input variables are as described in visual_behavior.ophys.response_analysis.cell_metrics.get_cell_metrics_for_conditions()
    '''
    import visual_behavior.ophys.response_analysis.cell_metrics as cm

    if save_dir is not None:
        merged_model_free_metrics_table_filepath = os.path.join(save_dir, 'merged_model_free_metrics_table_' + data_type + '_' + inclusion_criteria + '.csv')
        print('filepath:', merged_model_free_metrics_table_filepath)
    else:
        print('no save_dir provided, cant save or load model free metrics table')
    if (save_dir is not None) and (os.path.exists(merged_model_free_metrics_table_filepath)):
        print('loading model free metrics table')
        model_free_metrics = pd.read_csv(merged_model_free_metrics_table_filepath, index_col=0)
    else:
        # load model free metrics (one row per cell / experience level)

        # all images
        condition = 'images'
        stimuli = 'all_images'
        response_metrics = cm.get_cell_metrics_for_conditions(data_type, condition, stimuli, session_subset, inclusion_criteria)

        # pref image
        condition = 'images'
        stimuli = 'pref_image'
        pref_stim_response_metrics = cm.get_cell_metrics_for_conditions(data_type, condition, stimuli, session_subset, inclusion_criteria)

        # omissions
        condition = 'omissions'
        stimuli = 'all_images'
        omission_response_metrics = cm.get_cell_metrics_for_conditions(data_type, condition, stimuli, session_subset, inclusion_criteria)

        # changes
        condition = 'changes'
        stimuli = 'all_images'
        change_response_metrics = cm.get_cell_metrics_for_conditions(data_type, condition, stimuli, session_subset, inclusion_criteria)

        # merge all tables together

        # rename metrics so its clear what they were computed on (i.e. images vs. omissions vs. changes)
        model_free_metrics = pref_stim_response_metrics.rename(columns={'mean_response': 'mean_response_pref_image',
                                                                        'reliability': 'reliability_pref_image',
                                                                        'fano_factor': 'fano_factor_pref_image',
                                                                        'running_modulation_index': 'running_modulation_pref_image',
                                                                        'fraction_significant_p_value_gray_screen': 'fraction_significant_p_value_gray_screen_pref_image'})

        # select columns to keep
        model_free_metrics = model_free_metrics[['cell_specimen_id', 'experience_level', 'ophys_experiment_id',
                                                'mean_response_pref_image', 'reliability_pref_image', 'fraction_significant_p_value_gray_screen_pref_image',
                                                 'fano_factor_pref_image', 'running_modulation_pref_image']]

        # rename and merge in all image metrics
        tmp = response_metrics.rename(columns={'mean_response': 'mean_response_all_images',
                                               'lifetime_sparseness': 'lifetime_sparseness_images',
                                               'reliability': 'reliability_all_images',
                                               'fano_factor': 'fano_factor_all_images',
                                               'running_modulation_index': 'running_modulation_all_images',
                                               'fraction_significant_p_value_gray_screen': 'fraction_significant_p_value_gray_screen_all_images'})
        model_free_metrics = model_free_metrics.merge(tmp[['cell_specimen_id', 'experience_level', 'ophys_experiment_id',
                                                           'mean_response_all_images', 'lifetime_sparseness_images',
                                                           'reliability_all_images', 'fraction_significant_p_value_gray_screen_all_images',
                                                           'fano_factor_all_images', 'running_modulation_all_images']],
                                                      on=['cell_specimen_id', 'experience_level', 'ophys_experiment_id'])

        # merge in omission metrics for each exp level
        tmp = omission_response_metrics.rename(columns={'mean_response': 'mean_response_omissions',
                                                        # 'post_omitted_response': 'mean_response_post_omissions',
                                                        'reliability': 'reliability_omissions',
                                                        'fano_factor': 'fano_factor_omissions',
                                                        'running_modulation_index': 'running_modulation_omissions',
                                                        'fraction_significant_p_value_gray_screen': 'fraction_significant_p_value_gray_screen_omissions'})
        model_free_metrics = model_free_metrics.merge(tmp[['cell_specimen_id', 'experience_level', 'ophys_experiment_id',
                                                           'mean_response_omissions',  # 'mean_response_post_omissions',
                                                           'reliability_omissions', 'fraction_significant_p_value_gray_screen_omissions',
                                                           'fano_factor_omissions', 'running_modulation_omissions',
                                                           'omission_modulation_index']],
                                                      on=['cell_specimen_id', 'experience_level', 'ophys_experiment_id'])

        # merge in change metrics for each exp level
        tmp = change_response_metrics.rename(columns={'mean_response': 'mean_response_changes',
                                                      'pre_change_response': 'mean_response_pre_change',
                                                      'lifetime_sparseness': 'lifetime_sparseness_changes',
                                                      'reliability': 'reliability_changes',
                                                      'fano_factor': 'fano_factor_changes',
                                                      'running_modulation_index': 'running_modulation_changes',
                                                      'fraction_significant_p_value_gray_screen': 'fraction_significant_p_value_gray_screen_changes'})
        model_free_metrics = model_free_metrics.merge(tmp[['cell_specimen_id', 'experience_level', 'ophys_experiment_id',
                                                           'mean_response_changes', 'mean_response_pre_change',
                                                           'lifetime_sparseness_changes',
                                                           'reliability_changes', 'fraction_significant_p_value_gray_screen_changes',
                                                           'fano_factor_changes', 'running_modulation_changes',
                                                           'change_modulation_index', 'hit_miss_index']],
                                                      on=['cell_specimen_id', 'experience_level', 'ophys_experiment_id'])

        # convert experience level
        model_free_metrics['experience_level'] = [utils.convert_experience_level(experience_level) for experience_level in model_free_metrics.experience_level.values]

        print(len(model_free_metrics.cell_specimen_id.unique()), 'cells in merged response metrics table')

        # save it
        if save_dir is not None:
            model_free_metrics.to_csv(merged_model_free_metrics_table_filepath)
            print('response metrics table saved')

    return model_free_metrics


def generate_coding_score_metrics_per_experience_level_table(cluster_meta, results_pivoted, save_dir=None):
    '''
    compute metrics on coding scores for each experience level, and across experience levels, per cell, then merge tables
    metrics include things like 'dominant_feature' and 'max_image_coding_score', 'experience_modulation_direction' etc

    output is a table with one row per cell / experience level, limited to cells matched across experience levels
    table includes cluster_id from cluster_meta input table

    cluster_meta: dataframe containing one row per cell_specimen_id with the cluster_id assigned by clustering
    results_pivoted: dataframe containing GLM coding scores for features used in clustering
    '''

    # if metrics table already exists, load it, if not, generate and save it
    if save_dir is not None:
        coding_score_metrics_per_exp_level_filepath = os.path.join(save_dir, 'merged_coding_score_metrics_per_exp_level_table.csv')
    else:
        print('no save_dir provided, cant load or save coding score metrics per exp level table')
    if (save_dir is not None) and (os.path.exists(coding_score_metrics_per_exp_level_filepath)):
        print('loading coding score metrics per experience level table')
        metrics = pd.read_csv(coding_score_metrics_per_exp_level_filepath, index_col=0)
    else:
        # get coding scores per exp level from results_pivoted
        coding_scores_per_exp_level = get_coding_score_metrics_per_experience_level(cluster_meta, results_pivoted)
        print(coding_scores_per_exp_level.experience_level.unique())

        # get coding score metrics for each cell (metrics computed across experience levels, one row per cell)
        coding_score_metrics = get_cell_metrics(cluster_meta, results_pivoted)
        coding_score_metrics['experience_level'] = coding_score_metrics.dominant_experience_level

        # merge coding score metrics per experience level with select coding score metrics per cell

        # add columns with the dominant feature and exp level determined based on the cluster each cell belongs to
        # get cluster based dominant feature & exp from coding score metrics table
        coding_score_metrics['dominant_feature_cluster'] = coding_score_metrics.dominant_feature
        coding_score_metrics['dominant_experience_level_cluster'] = coding_score_metrics.dominant_experience_level

        # merge coding scores tables
        merged_metrics = coding_scores_per_exp_level.merge(coding_score_metrics.reset_index()[['cell_specimen_id', 'dominant_feature_cluster',
                                                                                               'dominant_experience_level_cluster']], on='cell_specimen_id', how='left')
        print(merged_metrics.experience_level.unique())
        # merge in metadata for each cell if cluster_meta is provided
        merged_metrics = merged_metrics.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cre_line', 'cluster_id', 'targeted_structure',
                                                                          'imaging_depth', 'binned_depth', 'area_binned_depth', 'project_code',
                                                                          'mouse_id' ]], on='cell_specimen_id', how='left')

        print(len(merged_metrics.cell_specimen_id.unique()), 'cells in merged coding score metrics per exp level table')
        if save_dir is not None:
            merged_metrics.to_csv(coding_score_metrics_per_exp_level_filepath)
            print('metrics table saved')

    return merged_metrics


def generate_merged_table_of_coding_score_and_model_free_metrics(cluster_meta, results_pivoted,
                                                                 data_type='filtered_events',
                                                                 session_subset='full_session',
                                                                 inclusion_criteria='platform_experiment_table',
                                                                 save_dir=None):
    '''

    generate table of coding score metrics per cell / experience level and merge with
    table of model-free metrics computed on cell responses for each cell / experience level,
    plus cluster metadata (i.e. cluster ID assigned to each cell_specimen_id)

    resulting merged table has one row per cell_specimen_id & experience_level and
    should be limited to cells used in clustering

    cluster_meta: dataframe containing one row per cell_specimen_id with the cluster_id assigned by clustering
    results_pivoted: dataframe containing GLM coding scores for features used in clustering
    save_dir: directory to save or load merged metrics table from

    other input variables are as described in visual_behavior.ophys.response_analysis.cell_metrics.get_cell_metrics_for_conditions()

    '''

    # if metrics table already exists, load it, if not, generate and save it
    if save_dir is not None:
        filename = 'merged_coding_score_and_model_free_metrics_table_' + data_type + '_' + inclusion_criteria + '.csv'
        merged_metrics_table_filepath = os.path.join(save_dir, filename)
    else:
        print('no save_dir provided, cant load or save metrics table')
    if (save_dir is not None) and (os.path.exists(merged_metrics_table_filepath)):
        print('loading coding score and model free metrics table')
        metrics = pd.read_csv(merged_metrics_table_filepath, index_col=0)
    else:

        model_free_metrics = generate_merged_table_of_model_free_metrics(data_type, session_subset, inclusion_criteria, save_dir)

        coding_score_per_experience_metrics = generate_coding_score_metrics_per_experience_level_table(cluster_meta, results_pivoted, save_dir)

        metrics = coding_score_per_experience_metrics.merge(model_free_metrics, on=['cell_specimen_id', 'experience_level'], how='left')

        print(len(metrics.cell_specimen_id.unique()), 'cells in merged coding score and model free metrics table')
        if save_dir is not None:
            metrics.to_csv(merged_metrics_table_filepath)
            print('merged metrics table saved')

    return metrics


### stats for comparison of coding scores across experience levels (and/or shuffle control) ###

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
    '''needs doc string'''
    # check for cre lines
    if 'cre_line' in metrics_df.keys():
        cre_lines = metrics_df['cre_line'].unique()
        if len(cre_lines) == 1:
            cre_line_ids = get_cre_line_cell_specimen_ids(metrics_df, metrics_df[cre_lines[0]])
        else:
            print('DOUBLE CHECK THAT THE CRE LINE SELECTION IS CORRECT IN THE CODE')
            cre_line_ids = metrics_df['cell_specimen_id'].unique()
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


def select_cre_cids(cids, cre_line):
    '''
    get cell specimen ids for a given cre line
    cids: list of cids
    cre_line: str, cre line to get cell specimen ids for
    returns: list of cell specimen ids for the given cre line
    '''
    cells_table = loading.get_cell_table()
    all_cre_line_ids = cells_table[cells_table['cre_line'] == cre_line]['cell_specimen_id'].values
    cre_line_ids = np.intersect1d(cids, all_cre_line_ids)
    return cre_line_ids

def shuffle_dropout_score(df_dropout, shuffle_type='all', separate_cre_lines=False):
    '''
    Shuffles dataframe with dropout scores from GLM.
    shuffle_type: str, default='all', other options= 'experience', 'regressors', 'experience_within_cell',
                        'full_experience'

    Returns:
        df_shuffled (pd. Dataframe) of shuffled dropout scores
    '''
    if separate_cre_lines:
        print('shuffle_dropout_score function: Feature matrix consists of multiple cre lines, going to shuffle dropout scores within cre lines, and combined them afterwards.')
        # adding this if statement to amke sure that even if dropouts cores are clustered with all cre lines, the shuffle only happens within cre lines
        # there is better ways of doing it, but for now, this is the easiest way to do it ira 2021-07-20
        cre_lines = ['Slc17a7-IRES2-Cre', 'Sst-IRES-Cre', 'Vip-IRES-Cre']
        print('using within cre line shuffle to create reference data for gap statistic. Cre lines are hardcoded [Slc17a7-IRES2-Cre, Sst-IRES-Cre, Vip-IRES-Cre]')
        
        df_shuffled = pd.DataFrame()
        for cre_line in cre_lines:
            cids = select_cre_cids(df_dropout.index.values, cre_line)
            print(len(cids))
            df_cre = df_dropout.loc[cids].copy()
            df_shuffled_cre= df_cre.copy()
            regressors = df_cre.columns.levels[0].values
            experience_levels = df_cre.columns.levels[1].values

            if shuffle_type == 'all':
                print('shuffling all data')
                for column in df_cre.columns:
                    df_shuffled_cre[column] = df_cre[column].sample(frac=1).values

            elif shuffle_type == 'experience':
                print('shuffling data across experience')
                assert np.shape(df_cre.columns.levels)[
                    0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
                for experience_level in experience_levels:
                    randomized_cids = df_cre.sample(frac=1).index.values
                    for i, cid in enumerate(randomized_cids):
                        for regressor in regressors:
                            df_shuffled_cre.iloc[i][(regressor, experience_level)] = df_cre.loc[cid][(regressor, experience_level)]

            elif shuffle_type == 'regressors':
                print('shuffling data across regressors')
                assert np.shape(df_cre.columns.levels)[
                    0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
                for regressor in regressors:
                    randomized_cids = df_cre.sample(frac=1).index.values
                    for i, cid in enumerate(randomized_cids):
                        for experience_level in experience_levels:
                            df_shuffled_cre.iloc[i][(regressor, experience_level)] = df_cre.loc[cid][(regressor, experience_level)]

            elif shuffle_type == 'experience_within_cell':
                print('shuffling data across experience within each cell')
                cids = df_cre.index.values
                experience_level_shuffled = experience_levels.copy()
                for cid in cids:
                    np.random.shuffle(experience_level_shuffled)
                    for j, experience_level in enumerate(experience_level_shuffled):
                        for regressor in regressors:
                            df_shuffled_cre.loc[cid][(regressor, experience_levels[j])] = df_cre.loc[cid][(regressor,
                                                                                                    experience_level)]
            elif shuffle_type == 'full_experience':
                print('shuffling data across experience fully (cell id and experience level)')
                assert np.shape(df_cre.columns.levels)[
                    0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
                # Shuffle cell ids first
                for experience_level in experience_levels:
                    randomized_cids = df_cre.sample(frac=1).index.values
                    for i, cid in enumerate(randomized_cids):
                        for regressor in regressors:
                            df_shuffled_cre.iloc[i][(regressor, experience_level)] = df_cre.loc[cid][
                            (   regressor, experience_level)]
                    # Shuffle experience labels
                    df_shuffled_cre_again = df_shuffled_cre.copy(deep=True)
                    cids = df_shuffled_cre.index.values
                    experience_level_shuffled = experience_levels.copy()
                    for cid in cids:
                        np.random.shuffle(experience_level_shuffled)
                        for j, experience_level in enumerate(experience_level_shuffled):
                            for regressor in regressors:
                                df_shuffled_cre_again.loc[cid][(regressor, experience_levels[j])] = df_shuffled_cre.loc[cid][(regressor,
                                                                                                            experience_level)]

                    df_shuffled_cre = df_shuffled_cre_again.copy()
            
            df_shuffled = df_shuffled.append(df_shuffled_cre)
        else:
            print('no such shuffle type..')
            df_shuffled = None
 # get cids for each cre line
            # 
    else:
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
                        df_shuffled.iloc[i][(regressor, experience_level)] = df_dropout.loc[cid][(regressor, experience_level)]

        elif shuffle_type == 'regressors':
            print('shuffling data across regressors')
            assert np.shape(df_dropout.columns.levels)[
                0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
            for regressor in regressors:
                randomized_cids = df_dropout.sample(frac=1).index.values
                for i, cid in enumerate(randomized_cids):
                    for experience_level in experience_levels:
                        df_shuffled.iloc[i][(regressor, experience_level)] = df_dropout.loc[cid][(regressor, experience_level)]

        elif shuffle_type == 'experience_within_cell':
            print('shuffling data across experience within each cell')
            cids = df_dropout.index.values
            experience_level_shuffled = experience_levels.copy()
            for cid in cids:
                np.random.shuffle(experience_level_shuffled)
                for j, experience_level in enumerate(experience_level_shuffled):
                    for regressor in regressors:
                        df_shuffled.loc[cid][(regressor, experience_levels[j])] = df_dropout.loc[cid][(regressor,
                                                                                                    experience_level)]
        elif shuffle_type == 'full_experience':
            print('shuffling data across experience fully (cell id and experience level)')
            assert np.shape(df_dropout.columns.levels)[
                0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
            # Shuffle cell ids first
            for experience_level in experience_levels:
                randomized_cids = df_dropout.sample(frac=1).index.values
                for i, cid in enumerate(randomized_cids):
                    for regressor in regressors:
                        df_shuffled.iloc[i][(regressor, experience_level)] = df_dropout.loc[cid][
                            (regressor, experience_level)]
            # Shuffle experience labels
            df_shuffled_again = df_shuffled.copy(deep=True)
            cids = df_shuffled.index.values
            experience_level_shuffled = experience_levels.copy()
            for cid in cids:
                np.random.shuffle(experience_level_shuffled)
                for j, experience_level in enumerate(experience_level_shuffled):
                    for regressor in regressors:
                        df_shuffled_again.loc[cid][(regressor, experience_levels[j])] = df_shuffled.loc[cid][(regressor,
                                                                                                            experience_level)]

            df_shuffled = df_shuffled_again.copy()

        else:
            print('no such shuffle type..')
            df_shuffled = None
    return df_shuffled


def get_CI_for_clusters(cluster_meta, columns_to_groupby=['targeted_structure', 'layer'], alpha=0.05, type_of_CI='mpc', cre='all'):
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
                        # print(f'{cre_line, location, cluster_id} no cells in this cluster')
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
                    except:  # noqa E501 # used to have KeyError here but sometimes its a TypeError when there are no cells in a cluster
                        # print(f'{cre_line, location, cluster_id} no cells in this cluster')
                        n_cluster = 0
                    N.append(n_cluster)
                # compute CIs for this location
                CIs = multinomial_proportions_confint(N, alpha=alpha)

                # assign CIs to their cluster
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


#### shuffle control functions ###

def get_cluster_mapping(matrix, threshold=1, ):
    '''
    find clusters with most similar SSE, create a dictionaty of cluster maps.

    Input:
    matrix: (np.array) SSE matrix (n clusters by n clusters), can be any other matrix (correlation, etc)
    threshold: (int) a value, abov which the matrix will not find an optimally similar cluster

    Returns:
    cluster_mapping_comparisons: (dict) {original cluster id: matched cluster id}

    '''
    comparisons = matrix.keys()

    # comparison dictionary
    cluster_mapping_comparisons = {}
    for i, comparison in enumerate(comparisons):
        tmp_matrix = matrix[comparison]

        # cluster
        cluster_mapping = {}
        # tmp_matrix is n_clusters x n_clusters with SSE values saying how well matched clusters are to shuffled clusters
        # index is original cluster id, column is matched cluster id
        for original_cluster_index in range(0, len(tmp_matrix)):

            # if min less than threshold
            if min(tmp_matrix[original_cluster_index]) < threshold:
                # the matched cluster is the one with the lowest SSE value
                matched_cluster_index = tmp_matrix[original_cluster_index].index(min(tmp_matrix[original_cluster_index])) + 1
                # add one because this is an index and cluster IDs start at 1
                cluster_mapping[original_cluster_index + 1] = matched_cluster_index

            # else there is no cluster, set matched cluster ID to -1
            else:
                cluster_mapping[original_cluster_index + 1] = -1

        cluster_mapping_comparisons[comparison] = cluster_mapping

    return cluster_mapping_comparisons


def get_mapped_SSE_values(matrix, threshold=1, ):
    '''
    collect SSE values for plotting for each cluster id
    Input:
    matrix: (np.array) SSE matrix (n clusters by n clusters), can be any other matrix (correlation, etc)
    threshold: (int) a value, abov which the matrix will not find an optimally similar cluster. This is necessary because 
        min values do not always reflect a true match, but rather a match that is better than all other matches.

    Returns:
    cluster_mapping_SSE,  (dict)

    '''
    n_boots = matrix.keys()
    cluster_mapping_SSE = {}
    for cluster_id in range(0, len(matrix[0])):
        sse_nb = []
        for i, n_boot in enumerate(n_boots):
            tmp_matrix = matrix[n_boot]
            # if min less than threshold
            if min(tmp_matrix[cluster_id]) < threshold:
                # the matched cluster is the one with the lowest SSE value
                min_SSE = min(tmp_matrix[cluster_id])
                sse_nb.append(min_SSE)
            else:
                sse_nb.append(np.nan)

        cluster_mapping_SSE[cluster_id + 1] = sse_nb

    return cluster_mapping_SSE


def get_mean_dropout_scores_per_cluster(dropout_df, cluster_df=None, labels=None, stacked=True, sort=False, max_n_clusters=None):
    '''
    INPUT:
    dropout_df: (pd.DataFrame) of GLM dropout scores (cell_specimen_ids by regressors x experience)
    cluster_df: (pd.DataFrame), either provide this df, must contain columns 'cluster_id', 'cell_specimen_id'
    labels: (list, np.array) or provide this array, list or array of int indicating cells' cluster ids,
                            if provided, len(labels)==len(dropout_df)
    sort: boolean, if True, sorts clusters by size, if False, clusters are not sorted

    Provide either cluster_df or labels. Cluster df must be df with 'cluster_id' and 'cell_specimen_id' columns,
    labels can be np array or list of cluster ids

    if unstacked, returns dictionary of DataFrames for each cluster. This version is helpful for plotting
    if stacked, returns a DataFrame of mean dropout scores per cluster, this version is great for computing corr or SSE
    (rows are droopout scores, columns are cluster ids)

    Clusters are sorted by size.
    '''
    if sort and isinstance(cluster_df, pd.core.frame.DataFrame):
        cluster_ids = cluster_df['cluster_id'].value_counts().index.values  # sort cluster ids by size
    elif sort is False and isinstance(cluster_df, pd.core.frame.DataFrame):
        cluster_ids = cluster_df['cluster_id'].unique()  # sorts numerically not by cluster size

    if sort is False and max_n_clusters is not None:
        cluster_ids = np.arange(1, max_n_clusters + 1)

    elif labels is not None:
        cluster_df = pd.DataFrame(data={'cluster_id': labels, 'cell_specimen_id': dropout_df.index.values})
        cluster_ids = cluster_df['cluster_id'].value_counts().index.values  # sort cluster ids by size

    # new cluster ids will start with 1, and they will be sorted by cluster size
    mean_cluster = {}
    for i, cluster_id in enumerate(cluster_ids):
        this_cluster_ids = cluster_df[cluster_df['cluster_id'] == cluster_id]['cell_specimen_id'].unique()
        if stacked is True:
            mean_dropout_df = dropout_df.loc[this_cluster_ids].mean()
            mean_cluster[cluster_id] = mean_dropout_df.values
        elif stacked is False:
            mean_dropout_df = dropout_df.loc[this_cluster_ids].mean().unstack()
            mean_cluster[cluster_id] = mean_dropout_df

    if stacked is True:
        return (pd.DataFrame(mean_cluster))
    elif stacked is False:
        return (mean_cluster)


def compute_SSE(mean_dropout_df_original, mean_dropout_df_compare):
    '''
    Computes SSE value between original and shuffled clusters
    mean dropout_df should be computed with get_mean_dropout_scores_per_cluster function, stacked=True
    returns:
    SSE_matrix, rows are original clusters, columns are compared clusters
    '''
    SSE_matrix = []
    for cluster_original in mean_dropout_df_original.keys():  # cluster ids are columns
        x = mean_dropout_df_original[cluster_original].values
        row = []
        for cluster_compare in mean_dropout_df_compare.keys():
            y = mean_dropout_df_compare[cluster_compare].values
            sse = np.sum((np.abs(x) - np.abs(y)) ** 2)
            row.append(sse)
        SSE_matrix.append(row)

    return SSE_matrix


def compute_cluster_sse(feature_matrix):
    '''
    Computes Sum of Squared Error between each cell in feature matrix and the mean, to measure variability within clusters.

    INPUT: 
    feature_matrix (pd.DataFrame): Dropout scores, rows are cell specimen ids.

    Returns:
    SSE (np.ndarray): Array of SSE values between each cell and their mean.
    '''
    mean_values = feature_matrix.mean().values
    SSE = np.sum(np.subtract(feature_matrix.values, mean_values)**2, axis=1)

    return SSE


def get_sse_df(feature_matrix, cluster_df, columns=['cluster_id', 'cre_line'], metric='sse'):
    '''
    Calculate the sum of squared errors (SSE) for each cluster in each cre_line. Create a dataframe to plot the results.
    This allows to compare the variability within clusters across different cre_lines.

    Args:
    - feature_matrix (pd.DataFrame): DataFrame containing the features for each cell.
    - cluster_df (pd.DataFrame): DataFrame with columns ['cre_line', 'cluster_id'] and cell specimen id as an index.
    - columns (list): List of columns for the output DataFrame.
    - metric (str): Metric to compute, currently supports only 'sse'. Option to add other metrics

    Returns:
    - pd.DataFrame: DataFrame containing the SSE values for each cluster in each cre_line.
    '''

    variability_df = pd.DataFrame(columns=columns)  # initialize empty dataframe
    cre_lines = cluster_df['cre_line'].unique()

    columns = [*columns, metric]

    if 'cell_specimen_id' in cluster_df.columns:
        cluster_df.set_index('cell_specimen_id', inplace=True)

    for cre_line in cre_lines:
        cre_cluster_df = cluster_df[cluster_df['cre_line'] == cre_line]
        cre_cell_ids = cre_cluster_df.index.values
        cre_feature_matrix = feature_matrix.loc[cre_cell_ids]
        cluster_ids = np.sort(cre_cluster_df['cluster_id'].values)
        for cluster_id in cluster_ids:  # compute for each cluster id
            cluster_cids = cre_cluster_df[cre_cluster_df['cluster_id'] == cluster_id].index.values
            cluster_feature_matrix = cre_feature_matrix.loc[cluster_cids]
            if metric == 'sse':
                values = compute_cluster_sse(cluster_feature_matrix)

            variability_df = variability_df.append(pd.DataFrame({'cre_line': [cre_line] * len(values),
                                                                 'cluster_id': [cluster_id] * len(values),
                                                                 metric: values}, index=np.arange(len(values))),
                                                   ignore_index=True)

        # compute for mean for clustered values
        index = len(variability_df)
        variability_df.at[index, 'cre_line'] = cre_line
        variability_df.at[index, 'cluster_id'] = 'clustered_mean'
        variability_df.at[index, metric] = variability_df[variability_df['cre_line'] == cre_line][metric].mean()

        # compute mean of unclustered values for each cre_line
        if metric == 'sse':
            value = np.mean(compute_cluster_sse(cre_feature_matrix))
        index = len(variability_df)
        variability_df.at[index, 'cre_line'] = cre_line
        variability_df.at[index, 'cluster_id'] = 'unclustered_mean'
        variability_df.at[index, metric] = value

    return variability_df


def get_sse_df_with_clustered_column(feature_matrix, cluster_df, columns=['cluster_id', 'cre_line', 'clustered'], metric='sse'):
    '''
    Calculate the sum of squared errors (SSE) for each cluster in each cre_line.
    This function has a column 'clustered' which indicates if the SSE is calculated based on clustered mean or not.
    Having this column helps plotting the results in slightly different way than get_sse_df function.

    Args:
    - feature_matrix (pd.DataFrame): DataFrame containing the features for each cell.
    - cluster_df (pd.DataFrame): DataFrame with columns ['cre_line', 'cluster_id'] and cell specimen id as an index.
    - columns (list): List of columns for the output DataFrame.
    - metric (str): Metric to compute, currently supports only 'sse'.

    Returns:
    - pd.DataFrame: DataFrame containing the SSE values for each cluster in each cre_line.
    '''

    variability_df = pd.DataFrame(columns=columns)
    cre_lines = np.sort(cluster_df['cre_line'].unique())

    columns = [*columns, metric]

    if 'cell_specimen_id' in cluster_df.columns:
        cluster_df.set_index('cell_specimen_id', inplace=True)

    for cre_line in cre_lines:
        cre_cluster_df = cluster_df[cluster_df['cre_line'] == cre_line]
        cre_cell_ids = cre_cluster_df.index.values
        cre_feature_matrix = feature_matrix.loc[cre_cell_ids]

        cluster_ids = np.sort(cre_cluster_df['cluster_id'].values)
        for cluster_id in cluster_ids:  # clustered
            cluster_cids = cre_cluster_df[cre_cluster_df['cluster_id'] == cluster_id].index.values
            cluster_feature_matrix = cre_feature_matrix.loc[cluster_cids]
            if metric == 'sse':
                values = compute_cluster_sse(cluster_feature_matrix)

            variability_df = variability_df.append(pd.DataFrame({'cre_line': [cre_line] * len(values),
                                                                 'cluster_id': [cluster_id] * len(values),
                                                                 'clustered': [True] * len(values),
                                                                 metric: values}, index=np.arange(len(values))),
                                                   ignore_index=True)

        if metric == 'sse':  # not clustered
            values = compute_cluster_sse(cre_feature_matrix)
        variability_df = variability_df.append(pd.DataFrame({'cre_line': [cre_line] * len(values),
                                                             'cluster_id': [np.nan] * len(values),
                                                             'clustered': [False] * len(values),
                                                             metric: values}, index=np.arange(len(values))),
                                               ignore_index=True)

    return variability_df


def get_sorted_cluster_ids(cluster_df):
    '''
    Maybe a reduntant function, but used to quickly get cluster ids that are sorted by size
    inputs:
    cluster_df (pd.DataFrame) dataframe with column 'cluster_id' of rows = cells
    returns:
    sorted_cluster_ids (np.array) of cluster ids sorted by size for plotting
    '''
    sorted_cluster_ids = cluster_df.value_counts('cluster_id').index.values
    return sorted_cluster_ids


# compute difference between original cluster sizes and shuffled cluster sizes. Create a df.

def get_cluster_size_differece_df(cre_original_cluster_sizes, shuffle_type_cluster_sizes, cre_line=None,
                                  columns=['cre_line', 'cluster_id', 'shuffle_type', 'n_boot', 'cluster_size_diff',
                                           'abs_cluster_size_diff', 'og_cluster_size', 'shuffle_size'],):
    ''' in long format, either normalized or not normalized difference of shuffled clusters and original clusters.
    This function is not finished but it works for now.

    INPUT:
    cre_original_cluster_sizes: dictionary cre_line: cluter_id: size
    shuffle_type_cluster_sizes: dictionary shuffle_type: cre_line: cluster_id: n_boot: cluster size
    cre_line: str of cre line if you wish to use only one
    columns: list of str/columns for output df
    normalize: boolean, wether to  normalize cluster size (og-shuffled)/(og+shuffled) or just use raw diff

    OUTPUT:
    cluster_size_difference_df: pd.DataFrame with specified columns

    '''
    if cre_line is None:
        cre_lines = [None]
    else:
        cre_lines = cre_line
    shuffle_types = shuffle_type_cluster_sizes.keys()
    # create empty df with columns to collect
    cluster_size_difference_df = pd.DataFrame(columns=columns)

    for shuffle_type in shuffle_types:
        for cre_line in cre_lines:

            # number of clusters to iterate over in this cre line
            if cre_line is not None:
                cluster_ids = shuffle_type_cluster_sizes[shuffle_type][cre_line].keys()
            else:
                cluster_ids = shuffle_type_cluster_sizes[shuffle_type].keys()

            for cluster_id in cluster_ids:
                if cre_line is not None:
                    og_size = cre_original_cluster_sizes[cre_line][cluster_id]
                    shuffled_sizes = shuffle_type_cluster_sizes[shuffle_type][cre_line][cluster_id]
                else:
                    og_size = cre_original_cluster_sizes[cluster_id]
                    shuffled_sizes = shuffle_type_cluster_sizes[shuffle_type][cluster_id]
                cluster_size_diff = np.subtract(og_size, shuffled_sizes) / np.add(og_size, shuffled_sizes)
                abs_cluster_size_diff = np.subtract(og_size, shuffled_sizes)

                # this part needs to be optimized. There should be a better way of adding values to dictionary without iteration
                data = []
                for n_boot, value in enumerate(cluster_size_diff):
                    data.append([cre_line, cluster_id, shuffle_type, n_boot, value, abs_cluster_size_diff[n_boot], og_size, shuffled_sizes[n_boot]])
                nb_df = pd.DataFrame(data, columns=columns)
                cluster_size_difference_df = cluster_size_difference_df.append(nb_df, ignore_index=True)

    return cluster_size_difference_df


def get_cluster_probability_df(shuffle_type_probabilities, cre_line=None,
                               columns=['cre_line', 'cluster_id', 'shuffle_type', 'probability']):
    ''' in long format, probabilities of shuffled clusters.
    This function is not finished but it works for now.'''

    shuffle_types = shuffle_type_probabilities.keys()
    # create empty df with columns to collect
    shuffle_type_probability_df = pd.DataFrame(columns=columns)

    for shuffle_type in shuffle_types:
        if cre_line is None:
            cre_lines = [None]
        else:
            cre_lines = shuffle_type_probabilities[shuffle_type].keys()
        for cre_line in cre_lines:

            # number of clusters to iterate over in this cre line
            if cre_line is not None:
                cluster_ids = shuffle_type_probabilities[shuffle_type][cre_line].keys()
            else:
                cluster_ids = shuffle_type_probabilities[shuffle_type].keys()
            for cluster_id in cluster_ids:
                if cre_line is not None:
                    value = shuffle_type_probabilities[shuffle_type][cre_line][cluster_id]
                else:
                    value = shuffle_type_probabilities[shuffle_type][cluster_id]
                # cluster_df = pd.DataFrame(data=[cre_line, cluster_id, shuffle_type, value], columns = columns)
                cluster_df = pd.DataFrame({'cre_line': cre_line, 'cluster_id': cluster_id, 'shuffle_type': shuffle_type,
                                           'probability': value},
                                          index=[0])

                shuffle_type_probability_df = shuffle_type_probability_df.append(cluster_df, ignore_index=True)

    return shuffle_type_probability_df


def get_matched_cluster_labels(SSE_mapping):
    '''
    this function reformats SSE_mapping dictionary from {n_boot: {original id: matched id}}
    to matched_clusters dictionary of {original id: [matched id nb1, nb2, etc.]}
    '''
    original_cluster_ids = SSE_mapping[0].keys()  # this is just using the first set of mappings to get original cluster IDs
    n_boots = SSE_mapping.keys()
    matched_clusters = {}
    for cluster_id in original_cluster_ids:
        matched_ids = []
        for n_boot in n_boots:  # for every shuffle iteration, collect the matched ID for each original cluster id
            matched_id = SSE_mapping[n_boot][cluster_id]
            matched_ids.append(matched_id)
        matched_clusters[cluster_id] = matched_ids  # list of all matched IDs for each original cluster
    return matched_clusters  # dictionary of original cluster IDs and their matched IDs across all shuffle iterations


def get_cluster_size_variance(SSE_mapping, cluster_df_shuffled, normalize=False, use_nan=False, adjust_to_expected_N=False):
    original_cluster_ids = SSE_mapping[0].keys()
    matched_ids = get_matched_cluster_labels(SSE_mapping)  # gets list of all matched IDs across shuffle iterations for
    # each original cluster, in the order of shuffle iterations

    # cluster_df_shuffled is a dictionary with the cluster labels for each shuffle iteration
    n_boots = cluster_df_shuffled.keys()
    all_cluster_sizes = {}
    for original_cluster_id in original_cluster_ids:
        cluster_sizes = []
        for n_boot in n_boots:
            # count how many cells there are in each cluster for each shuffle iteration
            shuffled_cluster_sizes_this_boot = cluster_df_shuffled[n_boot].value_counts('cluster_id', normalize=normalize)
            # this will be a list of cluster IDs and their sizes for a single shuffle iteration
            matched_id_this_nb = matched_ids[original_cluster_id][n_boot]
            # now get the size of the matched cluster in this iteration
            if adjust_to_expected_N is True:
                expected_N = shuffled_cluster_sizes_this_boot.sum()  # total number of cells used for clustering
                matched_cluster_ids = np.array(list(SSE_mapping[n_boot].values()))  # which ids where matched on this nboot
                matched_cluster_ids = matched_cluster_ids[matched_cluster_ids > 0]  # exclude -1
                observed_N = shuffled_cluster_sizes_this_boot.loc[matched_cluster_ids].sum()  # compute sum of matched cluster sizes as observed N
                cluster_proportions = shuffled_cluster_sizes_this_boot / observed_N   # proportion of cells that ended up being matched
                shuffled_cluster_sizes_this_boot = cluster_proportions * expected_N  # adjust values to make sure sum observed = sum expected

            if matched_id_this_nb != -1:
                # get number of cells per cluster for the cluster IDs in the shuffles using the matched ID for THIS shuffle iteration

                try:
                    cluster_sizes.append(shuffled_cluster_sizes_this_boot.loc[matched_id_this_nb])
                except KeyError:
                    # print(shuffled_cluster_sizes_this_boot)
                    break
            else:
                if use_nan is True:
                    cluster_sizes.append(np.nan)
                elif use_nan is False:
                    cluster_sizes.append(0)

        # aggregate matched cluster sizes acoss iterations for each original cluster ID
        all_cluster_sizes[original_cluster_id] = cluster_sizes

    return all_cluster_sizes


def compute_probabilities(SSE_mapping):
    ''' Computes probability of shuffle clusters to be mapped to an original cluster for N repetitions.
    INPUT:
    SSE_mapping (dict), nested dictionary of matched clusters. Keys = cluster ids, values = dictionary where keys
        are n_boot (shuffle number) and values are orignal cluster id: matched cluster id.

    Returns:
        cluster_probabilities (dict), where keys are cluster ids and values are probabilities of matching this cluster
        during N number of shuffled (or nboots) from 0 to 1.

    '''
    cluster_ids = SSE_mapping[0].keys()
    cluster_count = count_cluster_frequency(SSE_mapping)
    cluster_probabilities = {}
    for cluster_id in cluster_ids:
        cluster_probabilities[cluster_id] = np.sum(cluster_count[cluster_id]) / len(cluster_count[cluster_id])
    return cluster_probabilities


def count_cluster_frequency(SSE_mapping):
    cluster_ids = SSE_mapping[0].keys()
    n_boots = SSE_mapping.keys()
    cluster_count = {}
    for cluster_id in cluster_ids:
        boolean_count = []
        for n_boot in n_boots:
            matched_id = SSE_mapping[n_boot][cluster_id]
            if matched_id != -1:
                boolean_count.append(True)
            else:
                boolean_count.append(False)
        cluster_count[cluster_id] = boolean_count
    return cluster_count


def get_matched_clusters_means_dict(SSE_mapping, mean_dropout_scores_unstacked, metric='mean', shuffle_type=None,
                                    cre_line=None):
    ''' This function can plot mean (or other metric like std, median, or custom function) of matched shuffle clusters. This is helpful to see
    how well matching worked but it does not show clusters that were not matched with any original clusters.
    INPUT:
    SSE_mapping: dictionary for each n_boot, original cluster_id: matched shuffled cluster_id
    mean_dropout_scores_unstacked: dictionary ofmean unstacked dropout scores  cluster_id: unstached pd.Data'''

    if shuffle_type is not None:
        SSE_mapping = SSE_mapping[shuffle_type]
        mean_dropout_scores_unstacked = mean_dropout_scores_unstacked[shuffle_type]

    if cre_line is not None:
        SSE_mapping = SSE_mapping[cre_line]
        mean_dropout_scores_unstacked = mean_dropout_scores_unstacked[cre_line]

    # set up variables
    n_boots = SSE_mapping.keys()
    cluster_ids = SSE_mapping[0].keys()
    columns = mean_dropout_scores_unstacked[0][1].columns

    all_clusters_means_dict = {}
    for cluster_id in cluster_ids:
        # get dropout scores for matched clusters across each shuffle iteration
        all_matched_cluster_df = pd.DataFrame(columns=columns)
        for n_boot in n_boots:
            matched_cluster_id = SSE_mapping[n_boot][cluster_id]
            if matched_cluster_id != -1:
                all_matched_cluster_df = all_matched_cluster_df.append(
                    mean_dropout_scores_unstacked[n_boot][matched_cluster_id])

        all_matched_cluster_df = all_matched_cluster_df.reset_index().rename(columns={'index': 'regressor'})

        # create dummy df for unmatched clusters
        if cluster_id == 1:  # is this a typo? should it be -1 for unmatched clusters?
            dummy_df = all_matched_cluster_df.groupby('regressor').mean().copy()
            dummy_df[dummy_df > 0] = 0
        # compute metrics
        if len(all_matched_cluster_df) >= 4:  # must be at least 4 matched clusters
            if metric == 'mean':
                all_clusters_means_dict[cluster_id] = all_matched_cluster_df.groupby('regressor').mean()
            elif metric == 'std':
                all_clusters_means_dict[cluster_id] = all_matched_cluster_df.groupby('regressor').std()
            elif metric == 'median':
                all_clusters_means_dict[cluster_id] = all_matched_cluster_df.groupby('regressor').median()
            else:
                all_clusters_means_dict[cluster_id] = all_matched_cluster_df.groupby('regressor').apply(metric)
        # elif all_matched_cluster_df.shape[0] == 4:
        #    all_clusters_means_dict[cluster_id] = all_matched_cluster_df
        else:
            all_clusters_means_dict[cluster_id] = dummy_df

    return all_clusters_means_dict


def get_corr_for_matched_clusters_dict(SSE_mapping, mean_shuffled_dropout_scores,
                                       mean_original_dropout_scores=None,
                                       cre_line=None, shuffle_type=None,
                                       use_spearman=True, ):
    ''' function to create a dictionary of within cluster correlations between original clusters
    and shuffled clusters. If mean_original_dropout_scores = None (default), computes correlation among the shuffles only.
     defaults to spearman r.'''
    # set up the variables
    if shuffle_type is not None:
        SSE_mapping = SSE_mapping[shuffle_type]
        mean_shuffled_dropout_scores = mean_shuffled_dropout_scores[shuffle_type]

    if cre_line is not None:
        SSE_mapping = SSE_mapping[cre_line]
        mean_shuffled_dropout_scores = mean_shuffled_dropout_scores[cre_line]
        if mean_original_dropout_scores is not None:
            mean_original_dropout_scores = mean_original_dropout_scores[cre_line]

    n_boots = SSE_mapping.keys()
    cluster_ids = SSE_mapping[0].keys()

    cluster_corr_dict = {}
    for cluster_id in cluster_ids:
        # X is a list of mean dropout scores of matched cluster per shuffle
        X = []

        if mean_original_dropout_scores is not None:
            Y = mean_original_dropout_scores[cluster_id].values

        # print(f'finding matched clusters for cluster id {cluster_id}')
        for n_boot in n_boots:
            matched_cluster_id = SSE_mapping[n_boot][cluster_id]
            if matched_cluster_id != -1:
                X.append(mean_shuffled_dropout_scores[n_boot][matched_cluster_id].values)

        # corr is a np.array of corr rs for all matched clusters (with original or one another)
        corr = np.array([])
        # compute corr values just for shuffled clusters
        if mean_original_dropout_scores is None and np.size(X) != 0:
            if use_spearman is True:
                # print(f'computing spearman r between {len(X)} matched clusters')
                if len(X) > 1:
                    corr = spearmanr(X, axis=1)[0]
                else:
                    corr = spearmanr(X)[0]
            else:
                # print(f'computing pearson r between {len(X)} matched clusters')
                corr = np.corrcoef(X)

            # use half of identity matrix only
            if np.shape(X)[0] > 1 and np.size(corr) > 1:
                indices = np.triu_indices_from(corr, k=1)
                corr = corr[indices]

        # compute correlation values of original with shuffled clusters
        elif np.size(X) != 0:
            if use_spearman is True:
                # print(f'computing spearman r between original and {len(X)} matched clusters')
                # use only one row of X corr with Y, exclude autocorr
                if np.size(X) > 12:
                    corr = spearmanr(X, Y, axis=1)[0][-1][:-1]
                else:  # if only one cluster is matched
                    corr = spearmanr(X, Y, axis=1)[0]
            else:
                # print(f'computing pearson r between original and {len(X)} matched clusters')
                if np.size(X) > 12:
                    corr = np.corrcoef(X, Y)[-1][:-1]
                else:  # if only one cluster is matched
                    corr = np.corrcoef(X, Y)

        # add mean and std values to cluster_corr_dict
        if np.size(corr) != 0:
            cluster_corr_dict[cluster_id] = [np.mean(corr), np.std(corr)]
        else:
            # print(f'did not find any matched clusters for cluster id {cluster_id}')
            cluster_corr_dict[cluster_id] = [np.nan, np.nan]

    return cluster_corr_dict


def get_shuffle_label(shuffle_type):
    shuffle_type_dict = {}
    shuffle_type_dict['experience'] = 'cell ID'
    shuffle_type_dict['experience_within_cell'] = 'exp labels'
    shuffle_type_dict['full_experience'] = 'both'
    shuffle_type_dict['regressors'] = 'regress'

    return shuffle_type_dict[shuffle_type]


def compute_sse(feature_matrix):
    '''
    Computes Sum of Squared Error between each cell in feature matrix and the mean.

    INPUT:
    feature_matrix (pd.DataFrame) dropout scores, rows are cell specimen ids

    Returns:
    SSE (list) of sse values between each cell and their mean.
    '''

    mean_values = feature_matrix.mean().values

    SSE = np.sum(np.subtract(feature_matrix.values, mean_values) ** 2, axis=1)

    return SSE


def get_variability_df(feature_matrix, cluster_df, columns=['cluster_id', 'cre_line', 'clustered'], metric='sse'):
    '''
    INPUT:
    feature_matrix:
    cluster_df: (pd.DataFrame) dataframe with columns ['cre_line', 'cluster_id'] and cell specimen id as an index
    metric: (string)

    Returns:
    variability_df
    '''

    variability_df = pd.DataFrame(columns=columns)
    cre_lines = np.sort(get_cre_lines(cluster_df))

    columns = [*columns, metric]

    if 'cell_specimen_id' in cluster_df.keys():
        cluster_df.set_index('cell_specimen_id', inplace=True)

    for cre_line in cre_lines:
        print(cre_line)
        cre_cluster_df = cluster_df[cluster_df.cre_line == cre_line]
        cre_cell_ids = cre_cluster_df.index.values
        cre_feature_matrix = feature_matrix.loc[cre_cell_ids]

        cluster_ids = np.sort(cre_cluster_df['cluster_id'].values)
        # compute values for each cluster id
        for cluster_id in cluster_ids:
            cluster_cids = cre_cluster_df[cre_cluster_df.cluster_id == cluster_id].index.values
            cluster_feature_matrix = cre_feature_matrix.loc[cluster_cids]
            if metric == 'sse':
                values = compute_sse(cluster_feature_matrix)

            variability_df = variability_df.append(pd.DataFrame({'cre_line': [cre_line] * len(values),
                                                                 'cluster_id': [cluster_id] * len(values),
                                                                 'clustered': [True] * len(values),
                                                                 metric: values}, index=np.arange(len(values))),
                                                   ignore_index=True)

        if metric == 'sse':
            values = compute_sse(cre_feature_matrix)
        variability_df = variability_df.append(pd.DataFrame({'cre_line': [cre_line] * len(values),
                                                             'cluster_id': [np.nan] * len(values),
                                                             'clustered': [False] * len(values),
                                                             metric: values}, index=np.arange(len(values))),
                                               ignore_index=True)

    return variability_df


def create_and_save_cluster_meta(coclustering_df, n_clusters, feature_matrix, cells_table_matched, cluster_meta_filename=None):
    '''
    INPUT:
    coclustering_df: (pd.DataFrame) of coclustering results
    n_clusters: (int) number of clusters
    feature_matrix: (pd.DataFrame) of dropout scores
    cells_table_matched: (pd.DataFrame) of cell metadata
    cluster_meta_filename: (str) filename to save the cluster meta file
    '''
    from sklearn.cluster import AgglomerativeClustering
    X = coclustering_df.values
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='average')
    labels = cluster.fit_predict(X)
    cell_specimen_ids = coclustering_df.index.values

    # make dictionary with labels for each cell specimen ID in this cre line
    labels_dict = {'labels': labels, 'cell_specimen_id': cell_specimen_ids}

    # turn it into a dataframe
    labels_df = pd.DataFrame(data=labels_dict, columns=['labels', 'cell_specimen_id'])

    # get new cluster_ids based on size of clusters and add to labels_df
    cluster_size_order = labels_df['labels'].value_counts().index.values

    # translate between original labels and new IDs based on cluster size
    labels_df['cluster_id'] = [np.where(cluster_size_order == label)[0][0] for label in labels_df.labels.values]

    # concatenate with df for all cre lines
    cluster_labels = labels_df.copy()
    cluster_labels['cluster_id'] = cluster_labels['cluster_id'].copy() + 1

    # add metadata to cluster labels
    cell_metadata = get_cell_metadata_for_feature_matrix(feature_matrix, cells_table_matched)

    cluster_meta = cluster_labels[['cell_specimen_id', 'cluster_id', 'labels']].merge(cell_metadata, on='cell_specimen_id')
    cluster_meta = cluster_meta.set_index('cell_specimen_id')

    # annotate & clean cluster metadata
    cluster_meta = clean_cluster_meta(cluster_meta)  # drop cluster IDs with fewer than 5 cells in them

    # save cluster meta file
    if cluster_meta_filename is not None:
        cluster_meta.to_hdf(cluster_meta_filename, key='df')
        print('Created and saved cluster_meta file.')
    return cluster_meta

# Functions for response metric plots across clusters


def add_significance(sample1, sample2, test='2ttest'):

    from scipy import stats
    if len(sample1) != 0 and len(sample2) != 0:

        nan_mask = np.isnan(sample1)
        # Remove NaN values from the array
        sample1_without_nans = sample1[~nan_mask]

        nan_mask = np.isnan(sample2)
        # Remove NaN values from the array
        sample2_without_nans = sample2[~nan_mask]

        if test == '2ttest':

            # Perform two-sample t-test (parametric)
            t, p = stats.ttest_ind(sample1_without_nans, sample2_without_nans)

        elif test == 'MW':

            # Perform Mann-Whitney U test (non-parametric)
            t, p = stats.mannwhitneyu(sample1_without_nans, sample2_without_nans)

        elif test == 'W':
            try:
                diff = np.squeeze(sample1) - np.squeeze(sample2)
                res = stats.wilcoxon(diff)  # diff)
                t = res.statistic
                p = res.pvalue
            except:
                t = np.nan
                p = np.nan
        else:
            print('Test was not recognized')
            t = np.nan
            p = np.nan
    else:
        t = np.nan
        p = np.nan

    return t, p
