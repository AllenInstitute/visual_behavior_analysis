import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from visual_behavior.dimensionality_reduction import clustering as vba_clust
from sklearn.cluster import SpectralClustering

if __name__ == '__main__':

    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, help='distance type')
    parser.add_argument('--method', type=str, help='method to assign labels used in spectral clustering (Kmeans, Discrete)')
    parser.add_argument('--n_clusters', type=int, help='number of clusters')
    parser.add_argument('--n_boots', type=int, help='number of repeats for sil score')
    #parser.add_argument('--df', help = 'glm dropout values, n_cells by n_features')
    args = parser.parse_args()
    data_file = '24_events_all_L2_optimize_by_session_last_familiar_second_novel_active.h5'

    file_path = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/summary_plots/glm/SpectralClustering/files/'
    fig_path = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/summary_plots/glm/SpectralClustering/plots/'
    file_path = file_path+data_file[:-3]
    fig_path = fig_path+data_file[:-3]

    cre_lines = ['Slc17a7-IRES2-Cre', 'Sst-IRES-Cre', 'Vip-IRES-Cre']

    df = pd.read_hdf(path_or_buf=os.path.join(file_path, data_file),key = 'df')
    for cre_line in cre_lines:
        filename = os.path.join(file_path, '{}_cre_ids.pkl'.format(data_file[:-3]))
        with open(filename, 'rb') as f:
            cre_ids = pkl.load(f)
            cids = cre_ids[cre_line]
            X = df.loc[cids].values
            f.close()
        sc = SpectralClustering(assign_labels=args.method)
        n_clusters = np.arange(2, args.n_clusters)

        filename_string = 'SpectralClustering_{}_{}_nc{}_nb{}_{}_{}'.format(args.method, args.metric, args.n_clusters,
                                                                            args.n_boots, data_file[:-3], cre_line)
        print(np.sum(X==np.nan))
        print(n_clusters)
        print(args.n_boots)
        print(args.metric)
        sil_scores, sil_std = vba_clust.get_silhouette_scores(X = X,
                                           model = sc,
                                           n_clusters = n_clusters,
                                           metric=args.metric,
                                           n_boots=args.n_boots)
        print('finished computing sil scores for '+ args.method + ' ' + args.metric)
        vba_clust.save_clustering_results([sil_scores, sil_std], filename_string, path = file_path)
    print('saved..')

