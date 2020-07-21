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


#%% Umap ran on different values of n_neighbors and spread; make scatter plots for each parameter

neigh_vals = np.concatenate(([10,50], np.arange(200, int(all_sess_ns_fof_thisCre.shape[0]/10), 500)))
min_dist_vals = [.1, .3, .7]

np.shape(embedding_neigh_sp) # (len_minDistVals x len_neigh) x num_samps x num_comps

emb = np.reshape(embedding_neigh_sp, (len(min_dist_vals), len(neigh_vals), np.shape(embedding_neigh_sp)[1], np.shape(embedding_neigh_sp)[2]), order='F')
emb.shape
# x = emb[ispr, ineigh, :, 0]
# y = emb[ispr, ineigh, :, 1]

##### save
'''
import pickle
f = open('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/umap/embedding_neigh_sp0.pkl', 'wb')
pickle.dump(np.array(embedding_neigh_sp0), f)        
f.close()


pkl = open('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/umap/embedding_neigh_sp0.pkl', 'rb')
this_sess = pickle.load(pkl)
this_sess.shape
'''

#%% Set vars for making the scatter plot

dosavefig = 1

color_metric = [[cres]]                
color_labs = ['cre']

amp = color_metric[0][0] # cres    
clab = color_labs[0]

                 
lab_analysis='UMAP'
cut_axes = 0

same_norm_fo = 0
fign_subp = 'cre_'

now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
dir_now = 'umap_cluster'
fmt = '.pdf'



#%% Plot each embedding ran with a particular umap parameter                 
# color by cre

fig = plt.figure(figsize=(21,7)) #(figsize=(10,3)) (14,5)
# plt.suptitle(f'{cre[:3]}', y=.995, fontsize=22) 

for isp in range(np.shape(embedding_neigh_sp)[0]): # isp=0 
#     fig = plt.figure(figsize=(21,7)) #(figsize=(10,3)) (14,5)
    xy = embedding_neigh_sp[isp]
    
    x = xy[:, 0]
    y = xy[:, 1]
    z = xy[:, 2]

#     ax = fig.add_subplot(len(min_dist_vals), len(neigh_vals), isp+1)
    ax = fig.add_subplot(len(min_dist_vals), len(neigh_vals), isp+1, projection='3d')    
    
    fig_ax = [fig, ax]
    
    scatter_umap(x, y, amp, clab, fig_ax, lab_analysis, cut_axes, same_norm_fo, 0, fign_subp, pc_exp_var=np.nan, whatSess = '_AallBall', fgn='', dim=[3,z])

    
plt.subplots_adjust(wspace=0.4, hspace=0.4)

if dosavefig:
    cre_short = 'allCre_allUMAPconfig'
    fgn_umap = f'{lab_analysis}_scatter_{fign_subp}allNeurSess'
#         if useSDK:
#             fgn_umap = f'{fgn_umap}_sdk_'
    nam = '%s_%s%s_%s_%s' %(cre_short, fgn_umap, '_AallBall', '', now)
    fign = os.path.join(dir0, dir_now, nam+fmt)     

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


    
    
    
    

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

                 



