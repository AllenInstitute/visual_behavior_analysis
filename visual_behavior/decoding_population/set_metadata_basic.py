#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#%% Function to set metadata_basic, a df that includes basic metadata for all the 8 experiments of all sessions in list_all_sessions_valid; metadata include: 'session_id', 'experiment_id', 'area', 'depth', 'valid'

This function gets called in svm_init_images_pre.py.


Created on Wed Oct 14 23:07:43 2020
@author: farzaneh
"""

# import sys
# sys.setrecursionlimit(1500)

def set_metadata_basic(list_all_sessions_valid, list_all_experiments_valid):    
#     list_all_sessions_valid = stimulus_response_data['ophys_session_id'].unique()
    
    import pandas as pd
    import numpy as np
    
    import ophysextractor
    from ophysextractor.datasets.lims_ophys_session import LimsOphysSession
    from ophysextractor.utils.util import mongo, get_psql_dict_cursor

    DB = mongo.qc.metrics 

    metadata_basic = pd.DataFrame()
    for isess in range(len(list_all_sessions_valid)):
        print(isess)
        session_id = list_all_sessions_valid[isess]
        
        try:
            Session_obj = LimsOphysSession(lims_id=session_id)
            cont = True
        except Exception as E:
            cont = False
            print(f'Note: metadata cannot be set for isess {isess}, session_id {session_id}!!')
            print(E)


        if cont:
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