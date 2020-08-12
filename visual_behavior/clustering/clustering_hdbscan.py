"""
HDBSCAN clustering of umap embedding

Created on Mon Aug 10 11:01:25 2020
@author: farzaneh
"""

import hdbscan # First install the package on the terminal: pip install hdbscan


#%% load umap embedding run on glm coeffs (fraction change in variance after dropping out each coefficient)

save_dir = os.path.join('/allen', 'programs', 'braintv', 'workgroups', 'nc-ophys', 'visual_behavior', 'clustering', 'UMAP_output')
file_name = '200810_UMAP_2D_GLM_fraction_change_in_explained_variance'
filepath = os.path.join(save_dir, f'{file_name}.npy')

embedding = np.load(filepath)

# data = embedding


#%% Run HDBSCAN clustering

clusterer = hdbscan.HDBSCAN() #(min_cluster_size=9, gen_min_span_tree=True)
clusterer.fit(embedding) # umap_df[['x','y']].to_numpy()
# cluster_labels = clusterer.fit_predict(embedding)
clusterer.labels_
clusterer.probabilities_

np.unique(clusterer.labels_)
[np.mean(clusterer.labels_ == np.unique(clusterer.labels_)[i]) for i in range(len(np.unique(clusterer.labels_)))]

# np.sum(clusterer.probabilities_ < 1)


#%% save the clusterer

import pickle

f = open(os.path.join(save_dir, f'{file_name}_hdbscan.pkl'), 'wb')
pickle.dump(clusterer, f)        
f.close()

# Read it
# f = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/clustering/UMAP_output/200810_UMAP_2D_GLM_fraction_change_in_explained_variance_hdbscan.pkl'
# pkl = open(f, 'rb')
# clusterer = pickle.load(pkl)
# use "clusterer.labels_" to get cluster ids


#%% Scatter plot and mark each cluster

color_palette = sns.color_palette('deep', 8) # ('Paired', 12)

cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]

cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]

plt.scatter(*embedding.T, s=5, linewidth=0, c=cluster_member_colors, alpha=0.25)


# color_palette = ['g', 'r']
# cluster_colors = [color_palette[x] for x in clusterer.labels_]
# cluster_member_colors = cluster_colors

#%%
# clusterer.condensed_tree_
# clusterer.condensed_tree_.plot()



