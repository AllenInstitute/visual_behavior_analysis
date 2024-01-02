import os
import numpy as np
import argparse
import pickle as pkl
import pandas as pd
from visual_behavior.dimensionality_reduction import clustering as vba_clust
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering as ac
import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.utils as utils
import seaborn as sns

if __name__ == '__main__':

    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_type', type=str, help='type of shuffle to perform (experience, experience_within_cell, regressors, all')
    parser.add_argument('--n_boot', type=int, help='run number')
    args = parser.parse_args()
    
    # get shuffle type and n_boot
    shuffle_type = args.shuffle_type
    n_boot = args.n_boot


    base_dir = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/figure_4/all_cre_clustering_082823_n_14'
    glm_version = '24_events_all_L2_optimize_by_session'
    folder = '240101_shuffle_test'

    save_dir = os.path.join(base_dir, folder)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # load feature matrix
    filename = '24_events_all_L2_optimize_by_session_feature_matrix.h5'
    df = pd.read_hdf(os.path.join(base_dir, filename), key='df')

    # load meta data
    meta_filename = 'cluster_meta_n_14_clusters.h5'
    df_meta = pd.read_hdf(os.path.join(base_dir, meta_filename), key='df')
    # get cre lines
    cre_lines = np.sort(df_meta.cre_line.unique())

    # create a dictionary of splitting feature matrix by cre line
    cre_line_dfs = {}
    ids =[]
    cre_lines_values = []
    for cre_line in cre_lines:
        cids = df_meta[df_meta['cre_line'] == cre_line].cell_specimen_id.values
        df_cre = df.loc[cids].copy()
        cre_line_dfs[cre_line] = df_cre
        ids.append(cids)
        cre_lines_values.append([cre_line]* len(cids))
    ids = np.concatenate(ids)
    cre_lines_values = np.concatenate(cre_lines_values)

    # Optimal number of clusters
    n_clusters = 12

    # create clustering object
    sc = SpectralClustering()

    # create file name for this shuffle
    nb_filename ='all_cells_{}_nb{}'.format(shuffle_type, n_boot) # nb shuffled dataset
    nb_full_name = os.path.join(save_dir, 'files', nb_filename+'.h5')

    # 1.shuffle feature matrix within cre line and merge it
    # if that shuffle already exist, load it
    start=True
    for cre_line in cre_lines:
        feature_matrix = cre_line_dfs[cre_line]
        if os.path.exists(nb_full_name):
            print(f'found shuffled feature matrix {n_boot}..')
            shuffled_feature_matrix = pd.read_hdf(nb_full_name, key='df')
        else:
            shuffled_feature_matrix_cre = vba_clust.shuffle_dropout_score(feature_matrix, shuffle_type=shuffle_type)
            if start:
                shuffled_feature_matrix = shuffled_feature_matrix_cre.copy()
                start=False
            else:
                shuffled_feature_matrix = shuffled_feature_matrix.append(shuffled_feature_matrix_cre)
    shuffled_feature_matrix.to_hdf(nb_full_name, key='df')

    # get feature matrix as numpy array
    X = shuffled_feature_matrix.values

    # 2. Compute coclustering matrix
    nb_full_name = os.path.join(save_dir, 'files', nb_filename+'_coClustering_matrix.pkl')
    if os.path.exists(nb_full_name):
        with open(nb_full_name, 'rb') as f:
            coclust_matrix = pkl.load(f)
            f.close()
            print('done loading coclust matrix..')
    else:
        coclust_matrix = vba_clust.get_coClust_matrix(X , model=sc, n_clusters=n_clusters)
        vba_clust.save_clustering_results(coclust_matrix, nb_full_name)
        print('saved coclustering matrix..')

    # 3. Plot coclustering matrix
    fig = sns.clustermap(coclust_matrix , cmap='Blues')
    fig_name = nb_filename + '_coClustering_Matrix'
    fig.fig.suptitle(fig_name)
    utils.save_figure(fig.fig, figsize=(25, 20), save_dir=save_dir, folder='plots',
                          fig_title=fig_name, formats=['.png'])

    # fit
    nb_full_name = os.path.join(save_dir, 'files', nb_filename+'_cluster_labels.h5')
    if os.path.exists(nb_full_name):
        cluster_df = pd.read_hdf(nb_full_name, key='clustered_df')
    else:
        cluster = ac(n_clusters=n_clusters, affinity='euclidean', linkage='average')
        labels = cluster.fit_predict(coclust_matrix)
        cluster_ids = labels +1
        print(len(ids), len(cre_lines_values), len(cluster_ids), len(labels))
        data = {'cell_specimen_id': ids, 'cre_line': cre_lines_values,
                    'cluster_id': cluster_ids, 'labels': labels}
        cluster_df = pd.DataFrame(data=data)
        cluster_df.to_hdf(nb_full_name, key='clustered_df')



    # 3. Plot clusters
    for cre_line in cre_lines:
        tmp = cluster_df[cluster_df['cre_line'] == cre_line]
        sort_order_array = tmp.value_counts('cluster_id').index.values # this might need upgrade in pandas package if you get an arror
        sort_order = {cre_line: sort_order_array} # sort order must be a dictionary with cre line as a key
        # make sure that clustered_df (or meta dataframe) in indexed with cell specimen ids
        vba_clust.plot_clusters_row(tmp.set_index('cell_specimen_id'),
                                              shuffled_feature_matrix,
                                              cre_line,
                                              sort_order=sort_order,
                                              save_dir=save_dir,
                                              folder='plots',
                                              suffix=f'_{shuffle_type}_nb{n_boot}',
                                              formats=['.png', '.pdf'])



