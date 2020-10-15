#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save the list of sessions, experiments, and the basic metadata for those sessions as a pickle file. 
This pickle file will be loaded in svm_init_images_pbs.py and svm_main_images_pbs.py

Created on Thu Oct 9 12:19:43 2020

@author: farzaneh
"""

#%%
import os
import numpy as np
import pandas as pd
import pickle

import visual_behavior.data_access.loading as loading
from set_metadata_basic import *



####################################################################################################
####################################################################################################
####################################################################################################
#%% Set list of sessions for a given session stage, using concatenated stimulus_response_dfs.
####################################################################################################
####################################################################################################
####################################################################################################

#%% Stimulus response data from the project and stage below will be loaded to set list of sessions
# this is to be used for svm_images analysis.
project_codes = ['VisualBehaviorMultiscope']
session_numbers = [6] #4


#%% Set "stimulus_response_data" and "stim_presentations" dataframes   #this can be slow for larger amounts of data. limiting to non-mesoscope production codes is faster.

#%time 
stim_response_dfs = loading.get_concatenated_stimulus_response_dfs(project_codes, session_numbers) # stimulus_response_data has cell response data and all experiment metadata



#%% Set list_all_sessions_valid and list_all_experiments (for all sessions at stage 4)

list_all_sessions_valid = stim_response_dfs['ophys_session_id'].unique()
list_all_experiments_valid = [stim_response_dfs[stim_response_dfs['ophys_session_id']==si]['ophys_experiment_id'].unique() for si in list_all_sessions_valid]
list_all_experiments_valid = np.array(list_all_experiments_valid, dtype=object)
# [list_all_experiments_valid[i].shape for i in range(len(list_all_experiments_valid))]
print(list_all_sessions_valid.shape)

dict_se = {'list_all_sessions_valid': list_all_sessions_valid, 'list_all_experiments_valid': list_all_experiments_valid} 




#%% Set metadata_basic, a df that includes basic metadata for all the 8 experiments of all sessions in list_all_sessions_valid; metadata include: 'session_id', 'experiment_id', 'area', 'depth', 'valid'

metadata_basic = set_metadata_basic(list_all_sessions_valid, list_all_experiments_valid)
print(metadata_basic)




#%% save sessions and experiments in a pkl file

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

dir_svm = os.path.join(dir_server_me, 'SVM')
# filen = os.path.join(dir_svm, f'sessions_experiments_{project_codes}_{session_numbers}')
a = '_'.join(project_codes)
b = '_'.join([str(session_numbers[i]) for i in range(len(session_numbers))])
filen = os.path.join(dir_svm, f'metadata_basic_{a}_{b}')
print(filen)

# save to a pickle file
f = open(filen, 'wb')
pickle.dump(dict_se, f)   
pickle.dump(metadata_basic, f)   
f.close()





####################################################################################################
####################################################################################################
####################################################################################################
#%% Set list of sessions out of experiments_table, for a given cre line
# We need below for slc, because stim_response_dfs could not be saved for it!
####################################################################################################
####################################################################################################
####################################################################################################

cre_full = 'Vip-IRES-Cre' # 'Slc17a7-IRES2-Cre' # 'Sst-IRES-Cre' # 

cre2ana = cre_full[:3].lower() #'slc' 


#%% Set metadata_basic, and use it to set list_all_sessions_valid

experiments_table = loading.get_filtered_ophys_experiment_table() # functions to load cell response data and stimulus presentations metadata for a set of sessions
experiments_table.shape

##### get experiments_table for Slc mice, mesoscope data
a = experiments_table[experiments_table['project_code']=='VisualBehaviorMultiscope']
experiments_table_slc = a[a['cre_line']==cre_full]
experiments_table_slc = experiments_table_slc.reset_index('ophys_experiment_id')
experiments_table_slc.shape

##### set list_all_sessions_valid and list_all_experiments_valid, so we can set metadata_basic
list_all_sessions_valid = experiments_table_slc['ophys_session_id'].unique()
# list_all_sessions_valid.shape

list_all_experiments_valid = [experiments_table_slc[experiments_table_slc['ophys_session_id']==si]['ophys_experiment_id'].unique() for si in list_all_sessions_valid]
list_all_experiments_valid = np.array(list_all_experiments_valid, dtype=object)
# [list_all_experiments_valid[i].shape for i in range(len(list_all_experiments_valid))]
print(list_all_sessions_valid.shape)

##### set metadata_basic
from set_metadata_basic import *
metadata_basic = set_metadata_basic(list_all_sessions_valid, list_all_experiments_valid)
print(metadata_basic)



#%% save metadata in a pkl file

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

dir_svm = os.path.join(dir_server_me, 'SVM')
# filen = os.path.join(dir_svm, f'sessions_experiments_{project_codes}_{session_numbers}')
a = '_'.join(project_codes)
b = cre2ana
filen = os.path.join(dir_svm, f'metadata_basic_{a}_{b}')
print(filen)


# save to a pickle file
f = open(filen, 'wb')
# pickle.dump(dict_se, f)   
pickle.dump(metadata_basic, f)   
f.close()

