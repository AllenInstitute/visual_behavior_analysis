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
# x = embedding[:,0]
# y = embedding[:,1]
# plt.scatter(x,y, s=2)

# embedding_all_cre = [embedding]
# cre_lines = ['all cre']
# plot_scatter_fo(cre_lines, embedding_all_cre, color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp)


ncells = len(metadata_df['cre_line'].values)
cres = np.array([metadata_df['cre_line'].values[i][:3] for i in range(ncells)])


####################################################################################    
#%% Run umap codes
####################################################################################

from def_paths import *    
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

                 



