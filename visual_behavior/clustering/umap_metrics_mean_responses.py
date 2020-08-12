"""
Running umap on an input data that includes metrics (quantification) of image, image change, and omission responses (before and after each event).

The metrics are set in the following notebook:
200713_create_metrics_array_for_clustering.ipynb (inside visual_behavior_analysis/notebooks)

Created on Mon Jul 13 13:52:25 2020
@author: farzaneh
"""

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

import visual_behavior.data_access.loading as loading
# import visual_behavior.ophys.response_analysis.utilities as ut



experiments_table = loading.get_filtered_ophys_experiment_table() 



#%% Load previously saved data
# there is an easy function to get the default path for the 'visual_behavior_production_analysis' folder:
# loading.get_analysis_cache_dir()

multi_session_metrics_df = pd.read_hdf(os.path.join(loading.get_analysis_cache_dir(), 'multi_session_summary_dfs', 'response_metrics_feature_matrix.h5'), key='df')

unstacked_metrics = pd.read_hdf(os.path.join(loading.get_analysis_cache_dir(), 'multi_session_summary_dfs', 'response_metrics_feature_matrix_unstacked.h5'), key='df')
###### remember: 
# change_im 8 to 15 are for B sessions.
# im 9 to 16 are for B sessions.
# im 8 is the same as omitted.


#%% Get the metrics array without all the metadata columns
# i assume this array is what goes into PCA or UMAP
metrics_array = unstacked_metrics.drop(columns=experiments_table.columns)
# drop im8 columns, since im8 is the same as omitted
metrics_array = metrics_array.drop(columns=['im8_post', 'im8_pre'])
metrics_array = metrics_array.set_index(['cell_specimen_id','experiment_id','session_id'])
metrics_array.values 
metrics_array.values.shape # size: (24661, 70) : all_cells x columns


#%% Get just the metadata columns (corresponding to the rows in metrics_array)
# we can then use this metadata to label points in umap or split by cre line, etc
expts = experiments_table.reset_index().rename(columns={'ophys_experiment_id':'experiment_id'})
metadata_df = metrics_array.reset_index()[['cell_specimen_id','experiment_id','session_id']]
metadata_df = metadata_df.merge(expts, on='experiment_id')
metadata_df


#%% Further take care of metadata_df
'''
cre = experiments_table_2an.iloc[iexp]['cre_line']
date = experiments_table_2an.iloc[iexp]['date_of_acquisition'][:10]
stage = experiments_table_2an.iloc[iexp]['session_type']
area = experiments_table_2an.iloc[iexp]['targeted_structure']
depth = int(experiments_table_2an.iloc[iexp]['imaging_depth'])


session_name = str(experiments_table_2an.iloc[iexp]['session_name'])
uind = [m.start() for m in re.finditer('_', session_name)]
mid = session_name[uind[0]+1 : uind[1]]
if len(mid)<6: # a different nomenclature for session_name was used for session_id: 935559843
    mouse_id = int(session_name[uind[-1]+1 :])
else:
    mouse_id = int(mid)

all_sess_allN_allO.at[all_sess_allN_allO['mouse_id'].values==6, 'mouse_id'] = 453988    
'''


#%% Remove NaNs so umap can take metrics_array as input

# loop over each row (each cell), and remove NaN columns (they are coming from im8-15 (corresponding to B sessions)... which we dont need to do since we are keeping session stages in metadata)

inp = metrics_array.values

nc = sum(np.isfinite(inp[0])) # number of columns after removing nan values # 38
inpv = np.full((inp.shape[0], nc), np.nan)
for icell in range(inp.shape[0]): # icell = 0
    inpv[icell] = inp[icell, np.isfinite(inp[icell])]

inp.shape, inpv.shape # cells x 38

all_sess_ns_fof_thisCre = inpv
cre_lines = ['all']   
    
    
# embedding = umap.UMAP(spread= 5, n_neighbors = 15, n_components = 2, n_epochs = 500).fit_transform(all_sess_ns_fof_thisCre)
x = embedding[:,0]
y = embedding[:,1]
plt.scatter(x,y, s=2)

# embedding_all_cre = [embedding]
# cre_lines = ['all cre']
# plot_scatter_fo(cre_lines, embedding_all_cre, color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp)


ncells = len(metadata_df['cre_line'].values)
cres = np.array([metadata_df['cre_line'].values[i][:3] for i in range(ncells)])


####################################################################################    
#%% Run umap codes
####################################################################################

from multiscope_fn.def_paths import *    
import datetime, copy 

all_sess_ns_fof_all_cre = [all_sess_ns_fof_thisCre]    

# embedding_neigh_sp0 = copy.deepcopy(embedding_neigh_sp)
# np.shape(embedding_neigh_sp0)


    

#%% do clustering on the best looking embedding

#%% Get those points that are clustered separately and see how their traces look
# clust1 = np.logical_and(y>15 , x<5)
# clust2 = np.logical_and(y>15 , x<5)

# x.shape
# inpv.shape

# metadata_df




#%% analyze B1 sessions separately from A sessions
#%% do umap on individual cre lines
#%% only do umap on pooled images/ image changes (not individual images)

                 


######################################################################
######################################################################
#%% SVM weights
######################################################################
######################################################################

# load the latest svm file '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/SVM/all_sess_svm_gray_omit_20200811_124712.h5'
# it has weights saved for sameNumNeur = 0
        
data = np.vstack(all_sess['av_w_data_new'].values)
data0 = data+0
data = data[~np.isnan(data0[:,0])]


cres2 = np.array([all_sess['cre'].values[i][:3] for i in range(len(all_sess['cre'].values))])
a = []
# a = all_sess['av_w_data_new'].iloc[0]
for i in np.arange(0, len(all_sess)):
    if ~np.isnan(all_sess['av_w_data_new'].iloc[i][0]).any() and ~np.isnan(all_sess['n_neurons'].iloc[i]):
        if np.ndim(all_sess['av_w_data_new'].iloc[i])==2: # now you fixed svm_main_post but the code was run such that if n_neurons=1, then av_w_data will have 1 dimension, so we have to fix it here.
#             all_sess['av_w_data_new'].iloc[i] = all_sess['av_w_data_new'].iloc[i][np.newaxis,:]        
#             print(all_sess['n_neurons'].iloc[i])
            nn = np.full((all_sess['n_neurons'].iloc[i], 1), cres2[i])#.squeeze()
#             print(nn.shape)
            a.append(nn)

#         a = np.concatenate((a, all_sess['av_w_data_new'].iloc[i]), axis=0)
cres = np.concatenate(a).squeeze()
cres.shape
        
# cres2 = [np.full((all_sess['av_w_data_new'].iloc[i].shape[0], 1), cres[i]).squeeze() for i in range(len(cres))]


for i in range(len(cres)):
	all_sess['av_w_data_new'].values[i]

import umap    
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

ncomp = 20 # 3 # number of umap components
embedding = umap.UMAP(n_components = ncomp).fit_transform(data)

print(f'embedding size: {embedding.shape}')
    


import hdbscan # First install the package on the terminal: pip install hdbscan

clusterer = hdbscan.HDBSCAN() #(min_cluster_size=9, gen_min_span_tree=True)
clusterer.fit(embedding) # umap_df[['x','y']].to_numpy()
# cluster_labels = clusterer.fit_predict(embedding)



color_metric = [[cres]] # if umap was run on a matrix that included all cre lines                 
color_labs = ['cre']
dosavefig = 0


amp = color_metric[0][0] # cres    
clab = color_labs[0]

                 
lab_analysis='UMAP'
cut_axes = 0

same_norm_fo = 0
fign_subp = 'cre_'

now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
dir_now = 'umap_cluster'
fmt = '.pdf'


plot_scatter_fo(cre_lines, embedding_all_cre, color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp)



