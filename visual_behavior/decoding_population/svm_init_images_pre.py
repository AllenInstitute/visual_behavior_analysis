#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save the list of sessions, experiments, and the basic metadata for those sessions as a pickle file. 
This pickle file will be loaded in svm_init_images_pbs.py and svm_main_images_pbs.py

Created on Thu Oct 9 12:19:43 2020

@author: farzaneh
"""

#%% Stimulus response data from the project and stage below will be loaded to set list of sessions
# this is to be used for svm_images analysis.
project_codes = ['VisualBehaviorMultiscope']
session_numbers = [6] #4


#%%
import os
import numpy as np
import pandas as pd
import pickle

import visual_behavior.data_access.loading as loading


#%% Function to set metadata_basic, a df that includes basic metadata for all the 8 experiments of all sessions in list_all_sessions_valid; metadata include: 'session_id', 'experiment_id', 'area', 'depth', 'valid'
def set_metadata_basic(list_all_sessions_valid):
    
#     list_all_sessions_valid = stimulus_response_data['ophys_session_id'].unique()
    
    import ophysextractor
    from ophysextractor.datasets.lims_ophys_session import LimsOphysSession
    from ophysextractor.utils.util import mongo, get_psql_dict_cursor

    DB = mongo.qc.metrics 

    metadata_basic = pd.DataFrame()
    for isess in range(len(list_all_sessions_valid)):
        session_id = list_all_sessions_valid[isess]
        Session_obj = LimsOphysSession(lims_id=session_id)

        # get all the 8 experiments ids for this session
        experiment_ids_this_session = np.sort(np.array(Session_obj.data_pointer['ophys_experiment_ids']))

        # experiment_ids_this_session.shape, list_all_experiments_valid[isess].shape
        valid_allexp_this_session = np.in1d(experiment_ids_this_session, list_all_experiments_valid[isess])

        metadata_now = pd.DataFrame([], columns=['session_id', 'experiment_id', 'area', 'depth', 'valid'])
        for i in range(len(experiment_ids_this_session)): # ophys_experiment_id = experiment_ids_this_session[0]
            experiment_id = experiment_ids_this_session[i]

            # We have to get depth from Mouse-seeks database
            db_cursor = DB.find({"lims_id":int(experiment_id)})
            depth = db_cursor[0]['lims_ophys_experiment']['depth']
            area = db_cursor[0]['lims_ophys_experiment']['area']

            valid = valid_allexp_this_session[i]

            metadata_now.at[i, :] = session_id, experiment_id, area, depth, valid

        ## Important: here we sort metadata_basic by area and depth
        metadata_now = metadata_now.sort_values(by=['area', 'depth'])

        metadata_basic = pd.concat([metadata_basic, metadata_now])

#     len(metadata_basic)
#     metadata_basic

    return(metadata_basic)





#%% functions to load cell response data and stimulus presentations metadata for a set of sessions
# experiments_table = loading.get_filtered_ophys_experiment_table()


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

metadata_basic = set_metadata_basic(list_all_sessions_valid)
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
