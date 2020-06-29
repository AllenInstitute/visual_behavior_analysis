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