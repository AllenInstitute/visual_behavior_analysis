#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function is called in svm_images_init_pbs.py. We run svm_images_init_pbs.py in command line (cluster: slurm), so at the end of the function below we argeparse the inputs to this function.

This function sets svm vars and calls svm_images_main_pbs.py


Created on Wed Jul  7 14:24:17 2021
@author: farzaneh
"""

import os
import numpy as np
import pandas as pd
import pickle
import sys
import visual_behavior.data_access.loading as loading
from svm_images_main_pbs import *


def svm_images_main_pre_pbs(isess, project_codes, use_events, to_decode, trial_type, svm_blocks, engagement_pupil_running, use_spont_omitFrMinus1, use_balanced_trials):

    # if socket.gethostname() != 'ibs-farzaneh-ux2': # it's not known what node on the cluster will run your code, so all i can do is to say if it is not your pc .. but obviously this will be come problematic if running the code on a computer other than your pc or the cluster    
#     print(project_codes)

    #%% Get the input arguments passed here from pbstools (in svm_init_images script)

#     isess = int(sys.argv[1])
#     use_events = int(sys.argv[2]) #bool(sys.argv[2]): note: bool('False') is True!!!
#     to_decode = str(sys.argv[3])
#     trial_type = str(sys.argv[4])
#     svm_blocks = int(sys.argv[5]) 
#     engagement_pupil_running = int(sys.argv[6]) 
#     use_spont_omitFrMinus1 = int(sys.argv[7]) 
#     use_balanced_trials = int(sys.argv[8]) 

    # trial_type = 'images' # 'omissions', 'images', 'changes' # what trials to use for SVM analysis # the population activity of these trials at time time_win will be used to decode the image identity of flashes that occurred at their time 0 (if to_decode='current') or 750ms before (if to_decode='previous').
    # to_decode = 'previous' # 'current' (default): decode current image.    'previous': decode previous image.    'next': decode next image. # remember for omissions, you cant do "current", bc there is no current image, it has to be previous or next!
    # use_events = True # False # whether to run the analysis on detected events (inferred spikes) or dff traces.
    # svm_blocks = -1 #np.nan # 2 #np.nan # -1: divide trials based on engagement # 2 # number of trial blocks to divide the session to, and run svm on. # set to np.nan to run svm analysis on the whole session
    # engagement_pupil_running = 1 # np.nan or 0,1,2 for engagement, pupil, running: which metric to use to define engagement. only effective if svm_blocks=-1

    print('\n\n======== Analyzing session index %d ========\n' %(isess))



    #%% Set SVM vars # NOTE: Pay special attention to the following vars before running the SVM:

    # project_codes = ['VisualBehaviorMultiscope'] # session_numbers = [4]
#     project_codes = ['VisualBehavior']

    time_win = [-.5, 1.5] #[-.5, .75] # [-.3, 0] # timewindow (relative to trial onset) to run svm; this will be used to set frames_svm # analyze image-evoked responses
    # time_trace goes from -.5 to .65sec in the image-aligned traces.

    kfold = 5 #2 #10 # KFold divides all the samples in  groups of samples, called folds (if , this is equivalent to the Leave One Out strategy), of equal sizes (if possible). The prediction function is learned using  folds, and the fold left out is used for test.
    numSamples = 50 # numSamples = 2
    same_num_neuron_all_planes = 0 # if 1, use the same number of neurons for all planes to train svm
    saveResults = 1 # 0 #


    #%% Print some params

    print(f'Analyzing {project_codes} data')
    if trial_type=='changes_vs_nochanges':
        print(f'Decoding changs from no-changes, activity at {time_win} ms.')
    elif trial_type=='hits_vs_misses':
        print(f'Decoding hits from misses, activity at {time_win} ms.')    
    elif trial_type=='baseline_vs_nobaseline':
        print(f'Decoding *baseline* activity from activity at *each time point* at {time_win} ms.')
    else:    
        print(f'Decoding *{to_decode}* image from *{trial_type}* activity at {time_win} s.')
    print(f'kfold = {kfold}')    
    if use_events:
        print(f'Using events.')
    else:
        print(f'Using DF/F.')
    if svm_blocks==-100:
        print(f'Running SVM on the whole session.')
    elif svm_blocks==-1:
        print(f'Dividing trials based on engagement state.')
        if engagement_pupil_running==0:
            print(f"Using 'engagement' to define engagement state.")
        elif engagement_pupil_running==1:
            print(f"Using pupil to define engagement state.")        
        elif engagement_pupil_running==2:
            print(f"Using running to define engagement state.")                
    elif svm_blocks==-101:
        print(f'Running SVM only on engaged trials.')        
    else:    
        print(f'Doing block-by-block analysis.')
    if use_balanced_trials==1:
        print(f'Using equal number of trials from each class for the training and testing datasets.')
    else:    
        print(f'Not using equal number of trials from each class for the training and testing datasets.')



    #%%
    dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

    # make svm dir to save analysis results
    dir_svm = os.path.join(dir_server_me, 'SVM')

    if same_num_neuron_all_planes:
        dir_svm = os.path.join(dir_svm, 'same_n_neurons_all_planes')

    if svm_blocks!=-100:
        dir_svm = os.path.join(dir_svm, f'trial_blocks')

    if not os.path.exists(dir_svm):
        os.makedirs(dir_svm)




    #%% Use below for VIP and SST, since concatenated_stimulus_response_dfs exists for them.
    '''
    #%% Set session_numbers from filen; NOTE: below will work only if a single element exists in session_numbers

    sn = os.path.basename(filen)[len('metadata_basic_'):].find('_'); 
    session_numbers = [int(os.path.basename(filen)[len('metadata_basic_'):][sn+1:])]

    print(f'The following sessions will be analyzed: {project_codes}, {session_numbers}')

    #%% Set the project and stage for sessions to be loaded # Note: it would be nice to pass the following two vars as python args in the python job (pbs tools); except I'm not sure how to pass an array as an argument.
    # we need these to set "stim_response_data" and "stim_presentations" dataframes
    # r1 = filen.find('metadata_basic_'); r2 = filen.find(']_'); 
    # project_codes = [filen[r1+3:r2-1]]
    # session_numbers = [int(filen[filen.rfind('_[')+2:-1])] # may not work if multiple sessions are being passed as session_numbers



    #%% Set "stim_response_data" and "stim_presentations" dataframes

    stim_response_dfs = loading.get_concatenated_stimulus_response_dfs(project_codes, session_numbers) # stim_response_data has cell response data and all experiment metadata
    experiments_table = loading.get_filtered_ophys_experiment_table() # functions to load cell response data and stimulus presentations metadata for a set of sessions

    # merge stim_response_dfs with experiments_table
    stim_response_data = stim_response_dfs.merge(experiments_table, on=['ophys_experiment_id', 'ophys_session_id'])
    # stim_response_data.keys()

    stim_presentations = loading.get_concatenated_stimulus_presentations(project_codes, session_numbers) # stim_presentations has stimulus presentations metadata for the same set of experiments
    # stim_presentations.keys()




    #%% Load the pickle file that includes the list of sessions and metadata

    pkl = open(filen, 'rb')
    dict_se = pickle.load(pkl)
    metadata_basic = pickle.load(pkl)

    list_all_sessions_valid = dict_se['list_all_sessions_valid']
    list_all_experiments_valid = dict_se['list_all_experiments_valid']




    #%% Set the session to be analyzed, and its metada

    session_id = int(list_all_sessions_valid[isess])
    data_list = metadata_basic[metadata_basic['session_id'].values==session_id]


    #%% Set session_data and session_trials for the current session. # these two dataframes are linked by stimulus_presentations_id and ophys_session_id

    session_data = stim_response_data[stim_response_data['ophys_session_id']==session_id].copy() # get cell response data for this session
    session_trials = stim_presentations[stim_presentations['ophys_session_id']==session_id].copy() # get stimulus presentations data for this session

    df_data = session_data
    '''




    #%% We need below for slc, because stim_response_dfs could not be saved for it!

    #%% Load metadata_basic, and use it to set list_all_sessions_valid
    # note metadata is set in code: svm_init_images_pre
    '''
    dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
    dir_svm = os.path.join(dir_server_me, 'SVM')
    a = '_'.join(project_codes)
    filen = os.path.join(dir_svm, f'metadata_basic_{a}_slc')
    print(filen)
    '''

    # import visual_behavior.data_access.loading as loading

    # data release sessions
    # experiments_table = loading.get_filtered_ophys_experiment_table()
    experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
    experiments_table = experiments_table.reset_index('ophys_experiment_id')

    # get those rows of experiments_table that are for a specific project code
    metadata_valid = experiments_table[experiments_table['project_code']==project_codes]
#     metadata_valid = experiments_table[experiments_table['project_code']==project_codes[0]] #'VisualBehaviorMultiscope'] # multiscope sessions


    # Use the new list of sessions that are de-crosstalked and will be released in March 2021
    # metadata_meso_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/meso_decrosstalk/meso_experiments_in_release.csv'
    # metadata_valid = pd.read_csv(metadata_meso_dir)

    list_all_sessions_valid = metadata_valid['ophys_session_id'].unique()


    print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')

    # get the list of 8 experiments for all sessions
    experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
    experiments_table = experiments_table.reset_index('ophys_experiment_id')

    metadata_all = experiments_table[experiments_table['ophys_session_id'].isin(list_all_sessions_valid)==True] # metadata_all = experiments_table[np.in1d(experiments_table['ophys_session_id'].values, list_all_sessions_valid)]
    metadata_all.shape
    metadata_all.shape[0]/8



    # set the list of experiments for each session in list_all_sessions_valid
    if project_codes == 'VisualBehavior':
        list_all_experiments = metadata_all['ophys_experiment_id'].values

    else:
        try:
            list_all_experiments = np.reshape(metadata_all['ophys_experiment_id'].values, (8, len(list_all_sessions_valid)), order='F').T    

        except Exception as E:
            print(E)

            list_all_experiments = []
            for sess in list_all_sessions_valid: # sess = list_all_sessions_valid[0]
                es = metadata_all[metadata_all['ophys_session_id']==sess]['ophys_experiment_id'].values
                list_all_experiments.append(es)
            list_all_experiments = np.array(list_all_experiments)

            list_all_experiments.shape

            b = np.array([len(list_all_experiments[i]) for i in range(len(list_all_experiments))])
            no8 = list_all_sessions_valid[b!=8]
            if len(no8)>0:
                print(f'The following sessions dont have all the 8 experiments, excluding them! {no8}')    
                list_all_sessions_valid = list_all_sessions_valid[b==8]
                list_all_experiments = list_all_experiments[b==8]

                print(list_all_sessions_valid.shape, list_all_experiments.shape)

            list_all_experiments = np.vstack(list_all_experiments)

        list_all_experiments = np.sort(list_all_experiments) # sorts across axis 1, ie experiments within each session (does not change the order of sessions!)

    print(list_all_experiments.shape)




    '''
    pkl = open(filen, 'rb')
    metadata_basic = pickle.load(pkl)
    metadata_basic



    #%% Reset list_all_sessions_valid using metadata_basic
    # note for some sessions we cant set metadata_basic because the following code fails: Session_obj = LimsOphysSession(lims_id=session_id)
    # so, we use metadata_basic to reset list_all_sessions_valid

    list_all_sessions_valid_now = metadata_basic[metadata_basic['valid']]['session_id'].unique()
    # list_all_sessions_valid_now.shape, list_all_sessions_valid.shape
    list_all_sessions_valid = list_all_sessions_valid_now
    print(list_all_sessions_valid.shape)
    '''


    #%% Set experiments_table to be merged with stim response df later

    # import visual_behavior.data_access.loading as loading
    from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

    experiments_table = loading.get_filtered_ophys_experiment_table()

    # only keep certain columns of experiments_table, before merging it with stim_response_df
    experiments_table_sub = experiments_table.reset_index('ophys_experiment_id')
    experiments_table_sub = experiments_table_sub.loc[:, ['ophys_experiment_id', 'ophys_session_id', 'cre_line', 'session_name', 'date_of_acquisition', 'session_type', 'exposure_number']]

    '''
    # make sure all experiments in metadata exist in experiment_table
    es = experiments_table.index
    esmeta = metadata_basic[metadata_basic['valid']]['experiment_id'].unique()
    if np.in1d(es, esmeta).sum() == len(esmeta):
        print(f'All experiments in metadata df exist in experiment_table, as expected')
    else:
        print(f'Some experiments in metadata df do NOT exist in experiment_table; very uncanny!')
    '''



    ############################################################################################    
    ####################### Set vars for the session to be analyzed #######################
    ############################################################################################

    #%% Set the session to be analyzed, and its metada

    session_id = list_all_sessions_valid[isess] #session_id = int(list_all_sessions_valid[isess]) # experiment_ids = list_all_experiments_valid[isess]
    experiment_ids = list_all_experiments[isess] # we want to have the list of all experiments for each session regardless of whethere they were valid or not... this way we can find the same plane across all sessions.

    # Note: we cant use metadata from allensdk because it doesnt include all experiments of a session, it only includes the valid experiments, so it wont allow us to set the metadata for all experiments.
    # session_id = sessions_ctDone[isess]
    metadata_now = metadata_valid[metadata_valid['ophys_session_id']==session_id]
    experiment_ids_valid = metadata_now['ophys_experiment_id'].values # the problem is that this wont give us the list of all experiments for a session ... it only includes the valid experiments.
    experiment_ids_valid = np.sort(experiment_ids_valid)
    print(f'\n{sum(np.in1d(experiment_ids, experiment_ids_valid)==False)} of the experiments are invalid!\n')

    #cnt_sess = cnt_sess + 1
    print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess, len(list_all_sessions_valid)))
    # print('%d: session %d out of %d sessions' %(session_id, cnt_sess+1, len(list_all_sessions_valid)))



    #%% Set data_list: metadata for this session including a valid column

    data_list = metadata_all[metadata_all['ophys_session_id']==session_id]
    data_list['valid'] = False
    data_list.loc[data_list['ophys_experiment_id'].isin(experiment_ids_valid), 'valid'] = True
    # sort by area and depth
    data_list = data_list.sort_values(by=['targeted_structure', 'imaging_depth'])

    if project_codes == 'VisualBehavior':
        experiment_ids_this_session = [experiment_ids]
    else:
        experiment_ids_this_session = experiment_ids



    '''
    # isess = 0
    session_id = int(list_all_sessions_valid[isess])
    data_list = metadata_basic[metadata_basic['session_id'].values==session_id]
    experiment_ids_this_session = data_list['experiment_id'].values
    '''


    #%% Set stimulus_response_df_allexp, which includes stimulus_response_df (plus some metadata columns) for all experiments of a session

    # from set_trialsdf_existing_in_stimdf import *

    # set columns of stim_response_df to keep
    if trial_type != 'hits_vs_misses':
        c = ['stimulus_presentations_id', 'cell_specimen_id', 'trace', 'trace_timestamps', 'image_index', 'image_name', 'change', 'engagement_state', 'mean_running_speed', 'mean_pupil_area'] 
    else:
        c = ['trials_id', 'cell_specimen_id', 'trace', 'trace_timestamps', 'hit', 'false_alarm', 'miss', 'stimulus_change', 'aborted', 'go', 'catch', 'auto_rewarded', 'correct_reject', 'initial_image_name', 'change_image_name', 'engagement_state', 'mean_running_speed']

    stimulus_response_df_allexp = pd.DataFrame()

    for ophys_experiment_id in experiment_ids_this_session: # ophys_experiment_id = experiment_ids_this_session[0]

        if sum(np.in1d(experiment_ids_valid, int(ophys_experiment_id)))>0: # make sure lims_id is among the experiments in the data release
    #     if data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['valid']:

            valid = True

            dataset = loading.get_ophys_dataset(ophys_experiment_id)
    #         analysis = ResponseAnalysis(dataset, use_extended_stimulus_presentations=True) # False # use_extended_stimulus_presentations flag is set to False, meaning that only the main stimulus metadata will be present (image name, whether it is a change or omitted, and a few other things). If you need other columns (like engagement_state or anything from the behavior strategy model), you have to set that to True        
            analysis = ResponseAnalysis(dataset, use_extended_stimulus_presentations=True, use_events=use_events)                

            if trial_type == 'hits_vs_misses':
                trials_response_df = analysis.trials_response_df #(use_extended_stimulus_presentations=True)

                #### add engagement state from trials_df to trials_response_df
                trials_df = dataset.extended_trials

                for itrid in trials_df.index.values: # itrid = trials_df.index.values[4]
                    thistr = trials_response_df['trials_id']==itrid
                    if sum(thistr)>0:# it's not an aborted trial
                        trials_response_df.loc[thistr, 'engagement_state'] = trials_df.iloc[itrid]['engagement_state']


                stim_response_df = trials_response_df # so it matches the naming of the rest of the code down here
    #             c = stim_response_df.keys().values # get all the columns
            else:
                stim_response_df = analysis.get_response_df(df_name='stimulus_response_df')

    #         dataset.running_speed['speed'].values
    #         dataset.eye_tracking['pupil_area'] # dataset.eye_tracking['time']

    #         dataset.extended_stimulus_presentations['mean_running_speed'].values
    #         dataset.extended_stimulus_presentations['mean_pupil_area']



            # work on this
    #         if svm_blocks==-1 and np.isnan(np.unique(stim_response_df['engagement_state'].values)):
    #             sys.exit(f'Session {session_id} does not have engagement state!')

            ########### 
            if 0: #trial_type == 'changes':
                # if desired, set a subset of stimulus df that only includes image changes with a given trial condition (hit, autorewarded, etc)
                # trials_response_df.keys()  # 'hit', 'false_alarm', 'miss', 'correct_reject', 'aborted', 'go', 'catch', 'auto_rewarded', 'stimulus_change'

                scdf = stim_response_df[stim_response_df['change']==True]
    #             trials_response_df = analysis.get_response_df(df_name='trials_response_df')

                # get a subset of trials df that includes only those rows (flashes) in trials df that also exist in stim df # note: you need to set use_extended_stimulus_presentations=True so stim_response_df has start_time as a column.
                trialsdf_existing_in_stimdf = set_trialsdf_existing_in_stimdf(stim_response_df, trials_response_df) # trialsdf_existing_in_stimdf has the same size as scdf, and so can be used to link trials df and stimulus df, hence to get certain rows out of scdf (eg hit or aborted trials, etc)
                # trialsdf_existing_in_stimdf.shape, scdf.shape, trials_response_df.shape

                # only get those stimulus df rows that are go trials; seems like all stim df rows are go trials.
    #             stim_response_df_new = scdf[(trialsdf_existing_in_stimdf['go']==True).values]

                # only get those stimulus df rows that are not aborted and not auto_rewarded (we use trialsdf_existing_in_stimdf to get this information)
                a = np.logical_and((trialsdf_existing_in_stimdf['auto_rewarded']==False), (trialsdf_existing_in_stimdf['aborted']==False))
                stim_response_df_new = scdf.iloc[a.values]
                stim_response_df_new.shape

                # only get those stimulus df rows that are hit trials
    #             stim_response_df_new = scdf[(trialsdf_existing_in_stimdf['hit']==True).values]

                # reset stim_response_df
                stim_response_df = stim_response_df_new    
            ############################################    

            stim_response_df0 = stim_response_df # stim_response_df.keys()       
            # only get certain columns of stim_response_df
            stim_response_df = stim_response_df0.loc[:,c]

        else:

            valid = False
            stim_response_df = pd.DataFrame([np.full((len(c)), np.nan)], columns=c) 

        stim_response_df['ophys_session_id'] = session_id #data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['session_id']
        stim_response_df['ophys_experiment_id'] = ophys_experiment_id # data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['experiment_id']
        stim_response_df['area'] = metadata_all[metadata_all['ophys_experiment_id']==ophys_experiment_id].iloc[0]['targeted_structure']
        stim_response_df['depth'] = metadata_all[metadata_all['ophys_experiment_id']==ophys_experiment_id].iloc[0]['imaging_depth']
        stim_response_df['valid'] = valid 

    #     stim_response_df['area'] = data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['area']
    #     stim_response_df['depth'] = data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['depth']
    #     stim_response_df['valid'] = data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['valid']    


        #### set the final stim_response_data_this_exp

        stim_response_data_this_exp = stim_response_df.merge(experiments_table_sub, on=['ophys_experiment_id', 'ophys_session_id']) # add to stim_response_df columns with info on cre, date, etc        
        # reorder columns, so metadata columns come first then stimulus and cell data
        stim_response_data_this_exp = stim_response_data_this_exp.iloc[:, np.concatenate((np.arange(len(c), stim_response_data_this_exp.shape[-1]),    np.arange(0, len(c))))]

        #### keep data from all experiments    
        stimulus_response_df_allexp = pd.concat([stimulus_response_df_allexp, stim_response_data_this_exp])

    stimulus_response_df_allexp
    stimulus_response_df_allexp.keys()

    df_data = stimulus_response_df_allexp
    session_trials = np.nan # we need it as an input to the function


    ### some tests related to images changes in stim df vs. trials df  
    # trials_response_df[~np.in1d(range(len(trials_response_df)), stimdf_ind_in_trdf)]['trial_type'].unique()
    # trials_response_df[~np.in1d(range(len(trials_response_df)), stimdf_ind_in_trdf)]['catch'].unique()

    # trials_response_df.keys()  # 'hit', 'false_alarm', 'miss', 'correct_reject', 'aborted', 'go', 'catch', 'auto_rewarded', 'stimulus_change'
    # stimdf_ind_in_trdf.shape


    # (trialsdf_existing_in_stimdf['catch']==True).sum()
    # a = trialsdf_existing_in_stimdf[trialsdf_existing_in_stimdf['go']==False]
    # a['stimulus_change']
    # a['lick_times']
    # a['auto_rewarded']

    # b = scdf[(trialsdf_existing_in_stimdf['go']==False).values]
    # b['licked']
    # b['rewarded']
    # b['start_time']

    '''
    trials_response_df[trials_response_df['auto_rewarded']==True]['catch'] # all autorewarded are marked as neither catch or go.

    # find the change cases where animal licked but no reward is delivered and see if they are catch or not, if not whats wrong? (seems like stim df does not have catches, but has one of these weird cases)
    d = np.logical_and(np.logical_and((stim_response_df['licked']==1) , (stim_response_df['rewarded']==0)) , (stim_response_df['change']==True))
    dd = np.argwhere(d).flatten()
    stim_response_df.iloc[dd]
    stim_response_df[dd[1]]
    '''


    #%% Set frames_svm

    # Note: trials df has a much longer time_trace (goes up to 4.97) compared to stimulus df (goes up to .71), so frames_svm ends up being 1 element longer for trials df (ie when decoding hits from misses) compared to stimulus df (ie when decoding the images)

    # trace_time = np.array([-0.46631438, -0.3730515 , -0.27978863, -0.18652575, -0.09326288,
    #         0.        ,  0.09326288,  0.18652575,  0.27978863,  0.3730515 ,
    #         0.46631438,  0.55957726,  0.65284013])

    # trace_time = array([-0.48482431, -0.45250269, -0.42018107, -0.38785944, -0.35553782,
    #        -0.3232162 , -0.29089458, -0.25857296, -0.22625134, -0.19392972,
    #        -0.1616081 , -0.12928648, -0.09696486, -0.06464324, -0.03232162,
    #         0.        ,  0.03232162,  0.06464324,  0.09696486,  0.12928648,
    #         0.1616081 ,  0.19392972,  0.22625134,  0.25857296,  0.29089458,
    #         0.3232162 ,  0.35553782,  0.38785944,  0.42018107,  0.45250269,
    #         0.48482431,  0.51714593,  0.54946755,  0.58178917,  0.61411079,
    #         0.64643241,  0.67875403,  0.71107565,  0.74339727])

    trace_time = df_data.iloc[0]['trace_timestamps']
    # set samps_bef and samps_aft: on the image-aligned traces, samps_bef frames were before the image, and samps_aft-1 frames were after the image
    samps_bef = np.argwhere(trace_time==0)[0][0] # 5
    samps_aft = len(trace_time)-samps_bef #8 
    print(samps_bef, samps_aft)

    r1 = np.argwhere((trace_time-time_win[0])>=0)[0][0]
    r2 = np.argwhere((trace_time-time_win[1])>=0)
    if len(r2)>0:
        r2 = r2[0][0]
    else:
        r2 = len(trace_time)

    frames_svm = range(r1, r2)-samps_bef # range(-10, 30) # range(-1,1) # run svm on these frames relative to trial (image/omission) onset.
    print(frames_svm)

    # samps_bef (=40) frames before omission ; index: 0:39
    # omission frame ; index: 40
    # samps_aft - 1 (=39) frames after omission ; index: 41:79





    #%%
    cols_basic = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_trials', 'n_neurons', 'cell_specimen_id', 'frame_dur', 'samps_bef', 'samps_aft']) #, 'flash_omit_dur_all', 'flash_omit_dur_fr_all'])
    cols_svm_0 = ['frames_svm', 'to_decode', 'use_balanced_trials', 'thAct', 'numSamples', 'softNorm', 'kfold', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs', 
           'image_labels', 'image_indices', 'image_indices_previous_flash', 'image_indices_next_flash',
           'num_classes', 'iblock', 'trials_blocks', 'engagement_pupil_running', 'pupil_running_values' , 
           'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
           'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
           'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs']
    if same_num_neuron_all_planes:        
        cols_svm = np.concatenate((cols_svm_0, ['inds_subselected_neurons_all', 'population_sizes_to_try', 'numShufflesN']))

    else:    
        cols_svm = np.concatenate((cols_svm_0, ['testTrInds_allSamps_allFrs', 'Ytest_allSamps_allFrs', 'Ytest_hat_allSampsFrs_allFrs']))




    #%% Run the SVM function

    print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess, len(list_all_sessions_valid)))

    # Use below if you set session_data and session_trials above: for VIP and SST
    svm_images_main_pbs(session_id, data_list, experiment_ids_valid, df_data, session_trials, trial_type, dir_svm, kfold, frames_svm, numSamples, saveResults, cols_basic, cols_svm, project_codes, to_decode, svm_blocks, engagement_pupil_running, use_events, same_num_neuron_all_planes, use_balanced_trials, use_spont_omitFrMinus1)

    # svm_main_images_pbs(data_list, df_data, session_trials, trial_type, dir_svm, kfold, frames_svm, numSamples, saveResults, cols_basic, cols_svm, to_decode, svm_blocks, use_events, same_num_neuron_all_planes)

    # Use below if you set stimulus_response_df_allexp above: for Slc (can be also used for other cell types too; but we need it for Slc, as the concatenated dfs could not be set for it)
    # svm_main_images_pbs(data_list, stimulus_response_df_allexp, dir_svm, frames_svm, numSamples, saveResults, cols_basic, cols_svm, same_num_neuron_all_planes=0)





######################################################
######################################################
#%% For SLURM
######################################################
######################################################

import argparse

if __name__ == "__main__":
    
    # define args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--isess', type=int)
    parser.add_argument('--project_codes', type=str)    
    parser.add_argument('--use_events', type=int)    
    parser.add_argument('--to_decode', type=str)
    parser.add_argument('--trial_type', type=str)
    parser.add_argument('--svm_blocks', type=int)
    parser.add_argument('--engagement_pupil_running', type=int) 
    parser.add_argument('--use_spont_omitFrMinus1', type=int) 
    parser.add_argument('--use_balanced_trials', type=int) 
    
    args = parser.parse_args()


    
    
    # call the function
    svm_images_main_pre_pbs(
        args.isess, 
        args.project_codes, 
        args.use_events, 
        args.to_decode, 
        args.trial_type, 
        args.svm_blocks, 
        args.engagement_pupil_running, 
        args.use_spont_omitFrMinus1, 
        args.use_balanced_trials)


