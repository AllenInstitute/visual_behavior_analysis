#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#%% Function to set a subset of trials df that includes only those rows (flashes) in trials df that also exist in stim df

This function gets called in svm_images_main_pbs.py


Created on Sun Nov 1 17:50:43 2020
@author: farzaneh
"""

import numpy as np

def set_trialsdf_existing_in_stimdf(stim_response_df, trials_response_df):
    
    # set a subset of trials df that includes only those rows (flashes) in trials df that also exist in stim df
    # note: you need to set use_extended_stimulus_presentations=True so stim_response_df has start_time as a column.
    
    # trialsdf_existing_in_stimdf has the same size as scdf, and so can be used to link trials df and stimulus df, hence to get certain rows out of scdf (eg hit or aborted trials, etc)
    
#     dataset = loading.get_ophys_dataset(ophys_experiment_id)
#     analysis = ResponseAnalysis(dataset, use_extended_stimulus_presentations=True) # False # use_extended_stimulus_presentations flag is set to False, meaning that only the main stimulus metadata will be present (image name, whether it is a change or omitted, and a few other things). If you need other columns (like engagement_state or anything from the behavior strategy model), you have to set that to True
#     trials_response_df = analysis.get_response_df(df_name='trials_response_df')
#     stim_response_df = analysis.get_response_df(df_name='stimulus_response_df')


    scdf = stim_response_df[stim_response_df['change']==True]
    sdf_start = scdf['start_time'].values
    tdf_start = trials_response_df['start_time'].values
    tdf_stop = trials_response_df['stop_time'].values
#     print(sdf_start.shape, tdf_start.shape, tdf_stop.shape)

    # add the last element of tdf_stop to tdf_start, so we can catch the last trial too
    tdf_start = np.concatenate((tdf_start, [tdf_stop[-1]]))
#     print(sdf_start.shape, tdf_start.shape, tdf_stop.shape)

    # identify the row of trials df that belongs to each element of stim df
    stimdf_ind_in_trdf = np.full((len(sdf_start)), np.nan) # same size as scdf (ie stim df for change trials)
    for i in range(len(sdf_start)):
        stimdf_ind_in_trdf[i] = np.argwhere((sdf_start[i] - tdf_start)<0).flatten()[0] - 1

    # set trialsdf_existing_in_stimdf: includes a subset of rows in trials df that exist in stim df
    trialsdf_existing_in_stimdf = trials_response_df.iloc[stimdf_ind_in_trdf] # same size as scdf
    print(trialsdf_existing_in_stimdf.shape, scdf.shape)

    return trialsdf_existing_in_stimdf
