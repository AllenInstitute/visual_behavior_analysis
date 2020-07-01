"""
Vars needed here are set in umap_set_vars.py

Here we run umap and pca on the data matrix created in umap_set_vars.py.

After this script run umap_plots.py to make plots.


Created on Mon Jun 29 11:31:25 2020
@author: farzaneh

"""


################################################################################################    
################################################################################################        
#%% Run PCA on all_sess_ns_fof_this_cre
################################################################################################    
################################################################################################    

from sklearn.decomposition import PCA
varexpmax = .99 # 1 # .9

pc_all_cre = []
pca_variance_all_cre = []

for icre in range(len(cre_lines)): # icre=0
    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
    print(f'Running PCA on {cre}, matrix size: {np.shape(all_sess_ns_fof_thisCre)}')

    x_train_pc, pca = doPCA(all_sess_ns_fof_thisCre, varexpmax=varexpmax, doplot=1)
    pca_variance = pca.explained_variance_ratio_
#     x_train_pc.shape

    pc_all_cre.append(x_train_pc)
    pca_variance_all_cre.append(pca_variance)
    
    
################################################################################################    
################################################################################################    
#%% Run umap on all_sess_ns_fof_thisCre
################################################################################################
################################################################################################

import umap    
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

ncomp = 2 # 3 # number of umap components

embedding_all_cre = []

for icre in range(len(cre_lines)): # icre = 2    
    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
    print(f'Running UMAP on {cre}')

    sp = 2
    neigh = 7
    embedding = umap.UMAP(spread= sp, n_neighbors = neigh, n_components = ncomp).fit_transform(all_sess_ns_fof_thisCre)
    print(f'embedding size: {embedding.shape}')
    embedding_all_cre.append(embedding)
    
# embedding_all_cre_3d = copy.deepcopy(embedding_all_cre)

    
    
###########################################################################
#%% After this script, run umap_plot.py to make plots for umap/pca analysis.    
###########################################################################







from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

ncomp = 2 # 3 # number of umap components

embedding_all_cre = []

for icre in [0]: #range(len(cre_lines)): # icre = 2
    
    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
    print(f'Running UMAP on {cre}')

#     sp = 2
#     neigh = 7
    
    neigh_vals = np.concatenate(([50], np.arange(200, int(all_sess_ns_fof_thisCre.shape[0]/5), 500)))
#     neigh_vals = list(range(2,20))
    
    # creating dataframe of BIC score output
    best_clust = []
    best_type = []
    umap_neigh = []
    spreads = []
    bic_list = []

    # Iterating through UMAP configurations
    
    embedding_neigh_sp = []
    
    for neigh in neigh_vals:
        for sp in range(1,10):
            
            print(f'neighbors: {neigh}, spread: {sp}')
            
            embedding = umap.UMAP(spread= sp, n_neighbors = neigh, n_components = ncomp).fit_transform(all_sess_ns_fof_thisCre)
            print(f'embedding size: {embedding.shape}')
#             embedding_all_cre.append(embedding)
            
            embedding_neigh_sp.append(embedding)

    
    
            
        
            # Iterating through GMM configurations, 5 fold validation
            
            X = embedding
            kf = KFold(5)
            bic_array = []
            lowest_bic = np.infty
            bic = []
            n_components_range = range(2, 20)
            cv_types = ['spherical', 'tied', 'diag', 'full']
            
            for cv_type in cv_types:
                for n_components in n_components_range:
                    print(f'cv_type: {cv_type}, n_components: {n_components}')
                    
                    bic_temp = []
                    for train_index, test_index in kf.split(X):
                        
                        X_train = X[train_index]
                        X_test = X[test_index]
                        gmm = mixture.GaussianMixture(n_components=n_components,
                                                      covariance_type=cv_type)
                        gmm.fit(X_train)
                        bic_temp.append(gmm.bic(X_test))
                        
                    bic.append(np.average(bic_temp))
                    bic_array.append(bic_temp)

                    if bic[-1] < lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = gmm
                        best_cv = cv_type
                        
#             color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
            clf = best_gmm
            bars = []
            bic = np.array(bic)
            std_bars = np.std(bic_array, axis = 1)
            
            best_clust.append(best_gmm.n_components)
            best_type.append(best_gmm.covariance_type)
            umap_neigh.append(neigh)
            spreads.append(sp)
            bic_list.append(np.min(bic))

    

emb = np.reshape(embedding_neigh_sp, (9, 18, np.shape(embedding_neigh_sp)[1], np.shape(embedding_neigh_sp)[2]), order='F')
emb.shape

bic_listr = np.reshape(bic_list, (9, 18), order='F')
bic_listr.shape

dest_ct = f'/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/'
np_corr_fold = os.path.join(dest_ct, f'umap')
np_corr_file = os.path.join(np_corr_fold, f'umap_sp_neigh.h5')

if not os.path.exists(np_corr_fold):
    os.makedirs(np_corr_fold)
    
df_grid = pd.DataFrame([best_clust, best_type, umap_neigh, spreads]).T
df_grid.columns = ['n_best_cluster', 'best_gmm_type', 'umap_neighbor_config', 'umap_spread_config']

f = open(np_corr_file, 'wb')
pickle.dump(df_grid, f)        
pickle.dump(emb, f)    
pickle.dump(bic_listr, f)    
f.close()

# f = np_corr_file
# pkl = open(f, 'rb')
# this_sess = pickle.load(pkl)
# this_sess_emb = pickle.load(pkl)
# this_sess_bic = pickle.load(pkl)


spread = 6 #7
neigh = 12
embedding_all_cre = [emb[spread,neigh]]

embedding_all_cre = [emb[-1,-1]]
plot_scatter_fo([cre_lines[0]], embedding_all_cre, color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp)


#%%

df_grid = pd.DataFrame([best_clust, best_type, umap_neigh, spreads]).T
df_grid.columns = ['n_best_cluster', 'best_gmm_type', 'umap_neighbor_config', 'umap_spread_config']

df_grid

sns.countplot(df_grid['best_gmm_type'])
plt.title('Covariance Types per Best GMM:\n All Projects', fontsize = 20)
sns.despine(left=True, bottom=True, right=True)
plt.show()


sns.countplot(df_grid['n_best_cluster'])
plt.title('# Cluster Fit per Best GMM):\n All Projects', fontsize = 20)
sns.despine(left=True, bottom=True, right=True)
plt.show()


df_grid2 = df_grid[df_grid['best_gmm_type']=='full']

sns.countplot(data = df_grid2, x = 'n_best_cluster')
plt.title('# Cluster Fit per Best GMM (Full Covariance):\n All Projects', fontsize = 20)
sns.despine(left=True, bottom=True, right=True)
plt.show()



##### kde plots
sns.kdeplot(df_grid2['umap_neighbor_config'], df_grid2['umap_spread_config'], cmap = "Blues", shade = True)
plt.title('Stability of spread/n_neighbor configs:\n All Projects: Full Covariance', fontsize = 20)
plt.show()


df_grid.n_best_cluster.unique()


sns.kdeplot(df_grid['umap_neighbor_config'], df_grid['n_best_cluster'], cmap = "Blues", shade = True)
plt.title('Stability of # Clusters by Neighbor Config:\nAll Projects', fontsize = 20)
plt.show()

sns.kdeplot(df_grid['umap_spread_config'], df_grid['n_best_cluster'], cmap = "Blues", shade = True)
plt.title('Heatmap of Best GMMs using Correlation Covariance')
plt.title('Stability of # Clusters by Spread Config:\nAll Projects', fontsize = 20)
plt.show()




# run UMAP on your choice of configuration
embedding = umap.UMAP(spread= 5, n_neighbors = 15, n_components = 2, n_epochs = 500).fit_transform(all_sess_ns_fof_all_cre[icre])
X = embedding

plot_scatter_fo([cre_lines[0]], [X], color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp)

# df['umap-1'] = embedding[:,0]
# df['umap-2'] = embedding[:,1]
# df['umap-3'] = embedding[:,2]
# X = df[['umap-1', 'umap-2', 'umap-3']]


# iterate GMM (4 covariance types and # of clusters between 2-20) with 5 fold validation

kf = KFold(5)
bic_array = []
lowest_bic = np.infty
bic = []
n_components_range = range(2, 20)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        bic_temp =[]
        for train_index, test_index in kf.split(X):
            X_train = X[train_index]
            X_test = X[test_index]
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X_train)
            bic_temp.append(gmm.bic(X_test))
        bic.append(np.average(bic_temp))

        bic_array.append(bic_temp)
        

        
        
import itertools        

# plot output to visualize BIC scores per GMM (most of this I pulled from someone elses code online)

bic = np.average(bic_array, axis =1)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
bars = []
std_bars = np.std(bic_array, axis = 1)
start = 0

# Plot the BIC scores
plt.figure(figsize=(12, 6))
spl = plt.subplot(1, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    end = ((i+1)*len(n_components_range))
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):(i + 1) * len(n_components_range)], width=.2, color=color, yerr = std_bars[start:end]))
    start = end

plt.xticks(n_components_range)
plt.ylim([bic.min() * 0.99, bic.max()*1.01])
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)
#plt.savefig('GMM_BIC_scores.svg', format='svg', dpi=330)

plt.title('BIC Scores per GMM Configuration: 5 Fold Cross Validated\nAll Projects (Spread = 9, neighbors = 17)', fontsize = 20)
plt.show()




gmm = mixture.GaussianMixture(n_components=7, covariance_type='full')
gmm.fit(X)

labels = gmm.predict(X)

colors = list(plt.get_cmap('tab10').colors)
color_list = []
for n in labels:
    color_list.append(colors[n])
# X['color'] = color_list
# df['color'] = color_list
x =X[:,0] #['umap-1']
y =X[:,1] #['umap-2']
# z =X['umap-3']

fig = plt.figure()
ax3D = fig.add_subplot(111) #, projection='3d')
ax3D.scatter(x, y, s=10, c=color_list, marker='o')
# ax3D.scatter(x, y, z, s=10, c=X['color'], marker='o')

#ax3D.w_xaxis.set_ticks(np.arange(-5, 15, 5))
#ax3D.w_xaxis.set_label('UMAP Axis 1 (Arbitrary Units)')
ax3D.set_xlabel('UMAP Axis 1 (Arbitrary Units)')
ax3D.set_ylabel('UMAP Axis 2')
# ax3D.set_zlabel('UMAP Axis 3 ')


plt.title('All Projects UMAP Clusters', fontsize = 16)
plt.show()
