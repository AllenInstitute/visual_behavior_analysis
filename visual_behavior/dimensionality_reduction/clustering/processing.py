from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import numpy as np


def get_silhouette_kmeans(X, n_clusters=np.arange(2,10), metric='euclidean', n_boots=20):
    s = []
    for n_cluster in n_clusters:

        s_tmp = []
        for n_boot in range(1, n_boots):
            kmeans = KMeans(n_clusters=n_cluster).fit(X)
            labels = kmeans.labels_
            s_tmp.append(silhouette_score(X, labels, metric=metric))
        s.append(np.mean(s_tmp))
        print('n {} clusters mean score = {}'.format(n_cluster, np.mean(s_tmp)))

    return s


def get_silhouette_spec(X, n_clusters=np.arange(2,10), metric='euclidean', n_boots=20):
    s = []
    for n_cluster in n_clusters:

        s_tmp = []
        for n_boot in range(1, n_boots):
            # print(n_cluster, n_boot)
            sc = SpectralClustering(n_clusters=n_cluster)
            sc.fit_predict(X)
            labels = sc.labels_
            s_tmp.append(silhouette_score(X, labels, metric=metric))
        s.append(np.mean(s_tmp))
        print('n {} clusters mean score = {}'.format(n_cluster, np.mean(s_tmp)))

    return s