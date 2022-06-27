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
    parser.add_argument('--n_boots', type=int, help='number of run')
    args = parser.parse_args()

    base_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/figure_4/'
    glm_version = '24_events_all_L2_optimize_by_session'
    folder = '220627_shuffle_test'

    save_dir = os.path.join(base_dir, glm_version, folder)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    filename = '24_events_all_L2_optimize_by_session_feature_matrix.h5'
    df = pd.read_hdf(os.path.join(save_dir, 'files', filename), key='df')
    meta_filename = 'cluster_metadata_Vip_10_Sst_5_Slc17a7_10.h5'
    df_meta = pd.read_hdf(os.path.join(save_dir, 'files', meta_filename), key='df')
    cre_lines = np.sort(df_meta.cre_line.unique())
    shuffle_type = args.shuffle_type
    n_boots = args.n_boots

    # create a dictionary of splitting cre lines
    cre_line_dfs = {}
    for cre_line in cre_lines:
        cids = df_meta[df_meta['cre_line'] == cre_line].index.values
        df_cre = df.loc[cids].copy()
        cre_line_dfs[cre_line] = df_cre

    # Optimal number of clusters
    n_clusters_cre_dict = {}
    n_clusters_cre = [10, 5, 10]
    for c, cre_line in enumerate(cre_lines):
        n_clusters_cre_dict[cre_line] = n_clusters_cre[c]

    for cre_line in cre_lines:
        feature_matrix = cre_line_dfs[cre_line]
        for n_boot in range(n_boots):

            # create clustering object
            sc = SpectralClustering()

            # create file name for this shuffle
            nb_filename ='{}_{}_nb{}'.format(cre_line, shuffle_type, n_boot) # nb shuffled dataset
            nb_full_name = os.path.join(save_dir, nb_filename+'.h5')

            # 1.shuffle feature matrix
            # if that shuffle already exist, load it
            if os.path.exists(nb_full_name):
                shuffled_feature_matrix = pd.read_hdf(nb_full_name, key='df')
            else:
                shuffled_feature_matrix = vba_clust.shuffle_dropout_score(feature_matrix, shuffle_type=shuffle_type)
                shuffled_feature_matrix.to_hdf(nb_full_name, key='df')

            X = shuffled_feature_matrix.values

            # 2. Compute coclustering matrix
            nb_full_name = os.path.join(save_dir, 'files', nb_filename+'_coClustering_matrix')
            if os.path.exists(nb_full_name):
                with open(nb_full_name, 'rb') as f:
                    coclust_matrix = pkl.load(f)
                    f.close()
                    print('done loading coclust matrix..')
            else:

                coclust_matrix = vba_clust.get_coClust_matrix(X, model=sc, n_clusters=n_clusters_cre_dict[cre_line])
                vba_clust.save_clustering_results(coclust_matrix, nb_full_name)
                print('saved coclustering matrix..')

            # 3. Plot coclustering matrix
            fig = sns.clustermap(coclust_matrix , cmap='Blues')
            fig_name = nb_full_name + '_coClustering_Matrix'
            fig.fig.suptitle(fig_name)
            utils.save_figure(fig.fig, figsize=(25, 20), save_dir=save_dir, folder='plots',
                          fig_title=fig_name, formats=['.png'])

            # fit
            nb_full_name = os.path.join(save_dir, 'files', nb_filename+'_cluster_labels.h5')
            if os.path.exists(nb_full_name):
                clustered_df = pd.read_hdf(nb_full_name, key='clustered_df')
            else:
                ids = feature_matrix.index.values
                cluster = ac(n_clusters=n_clusters_cre_dict[cre_line], affinity='euclidean', linkage='average')
                labels = cluster.fit_predict(X)
                cluster_ids = labels +1
                data = {'cell_specimen_id': ids, 'cre_line': [cre_line] * len(ids),
                        'cluster_id': cluster_ids, 'labels': labels}
                clustered_df = pd.DataFrame(data=data)
                clustered_df.to_hdf(nb_full_name, key='clustered_df')


            # 3. Plot clusters
            sort_order = clustered_df.value_counts('cluster_id').index.values
            fig, ax = vba_clust.plot_clusters_row(clustered_df, shuffled_feature_matrix, cre_line,
                                    sort_order=sort_order, save_dir=save_dir, folder='plots',
                                    suffix = nb_filename+'_clusters')



