#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run this script to create all_sess, a dataframe in which the svm results of all sessions are concatenated. We need this to make plots for the svm (images) analysis.
This script calls the function: svm_main_images_post

SVM files are already saved (by running the svm_image_main_pbs code).

Follow this script by svm_images_plots_setVars_blocks.py to make plots.
If making plots for the block-block analysis, follow it by svm_images_plots_setVars_blocks.py to make plots.


Created on Tue Oct 13 20:48:43 2020
@author: farzaneh
"""

#%%
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import visual_behavior.data_access.loading as loading
from svm_images_main_post import *

# import warnings
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) # ignore some warnings!


    
#%% Get SVM output for each cell type, and each frames_svm.

svm_blocks = np.nan # np.nan # -1: divide trials based on engagement #2 # number of trial blocks to divide the session to, and run svm on. # set to np.nan to run svm analysis on the whole session
use_events = True #False # whether to run the analysis on detected events (inferred spikes) or dff traces.

to_decode = 'current' #'next' # 'current' (default): decode current image.    'previous': decode previous image.    'next': decode next image.
trial_type = 'hits_vs_misses' #'changes_vs_nochanges' #'omissions' # 'omissions', 'images', 'changes' # what trials to use for SVM analysis # the population activity of these trials at time time_win will be used to decode the image identity of flashes that occurred at their time 0 (if to_decode='current') or 750ms before (if to_decode='previous').
use_balanced_trials = 1 # if 1, use same number of trials for each class; only applicable when we have 2 classes (binary classification).

time_win = [0, .55] # 'frames_svm' # set time_win to a string (any string) to use frames_svm as the window of quantification. # time window relative to trial onset to quantify image signal. Over this window class accuracy traces will be averaged.

if trial_type=='hits_vs_misses':
    frames_svm_all = [np.arange(-5,9)]
else:
    frames_svm_all = [np.arange(-5,8)] #[[-3,-2,-1], [0,1,2,3,4,5]]

same_num_neuron_all_planes = 0

saveResults = 1
cre2ana_all = 'slc', 'sst', 'vip'
# session_numbers = [6] # ophys session stage corresponding to project_codes that we will load.


#%%
project_codes = ['VisualBehaviorMultiscope'] # ['VisualBehaviorMultiscope', 'VisualBehaviorTask1B', 'VisualBehavior', 'VisualBehaviorMultiscope4areasx2d']
num_planes = 8
analysis_dates = [''] # ['2020507'] #set to [''] if you want to load the latest file. # the date on which the svm files were saved.
frame_dur = np.array([.093])
# trace_time = np.array([-0.46631438, -0.3730515 , -0.27978863, -0.18652575, -0.09326288, 0.,  0.09326288,  0.18652575,  0.27978863,  0.3730515 , 0.46631438,  0.55957726,  0.65284013])
# samps_bef = 5 #np.argwhere(trace_time==0)[0][0] # 
# samps_aft = 8 #len(trace_time)-samps_bef #


frames_svm = frames_svm_all[0] #[-3,-2,-1] # [0,1,2,3,4,5] #  which svm files to load (this will be used to set svm file name) # (frames_svm also gets saved in svm_vars if svm could be run successfully on a session)
#     print(f'Analyzing: {cre2ana}, {frames_svm}')



#%% Set the directory

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
dir_svm = os.path.join(dir_server_me, 'SVM')

#             a = '_'.join(project_codes)
#             b = cre2ana  # b = '_'.join([str(session_numbers[i]) for i in range(len(session_numbers))])
#             filen = os.path.join(dir_svm, f'metadata_basic_{a}_{b}')
#             print(filen)

if same_num_neuron_all_planes:
    dir_svm = os.path.join(dir_svm, 'same_n_neurons_all_planes')

if ~np.isnan(svm_blocks):
    dir_svm = os.path.join(dir_svm, f'trial_blocks')



#%% Use March 2021 data release sessions

#%% Set the list of sessions
# experiments_table = loading.get_filtered_ophys_experiment_table()
experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')
metadata_valid = experiments_table[experiments_table['project_code']=='VisualBehaviorMultiscope'] # multiscope sessions

list_all_sessions_valid = metadata_valid['ophys_session_id'].unique()
print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')


#%% Get the list of 8 experiments for all sessions
experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')

metadata_all = experiments_table[experiments_table['ophys_session_id'].isin(list_all_sessions_valid)==True] # metadata_all = experiments_table[np.in1d(experiments_table['ophys_session_id'].values, list_all_sessions_valid)]
metadata_all.shape
metadata_all.shape[0]/8

# set the list of experiments for each session in list_all_sessions_valid
list_all_experiments = np.reshape(metadata_all['ophys_experiment_id'].values, (8, len(list_all_sessions_valid)), order='F').T    


    


#%% Set the columns in dataframe all_sess

cols0 = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_trials', 'n_neurons', 'frame_dur', 'meanX_allFrs', 'stdX_allFrs', 'av_train_data', 'av_test_data', 'av_test_shfl', 'av_test_chance', 'sd_train_data', 'sd_test_data', 'sd_test_shfl', 'sd_test_chance', 'peak_amp_trainTestShflChance'])
if same_num_neuron_all_planes:
    cols = np.concatenate((cols0, ['population_sizes_to_try'])) 
else:
    cols = np.concatenate((cols0, ['av_w_data', 'av_b_data']))

if ~np.isnan(svm_blocks):
    if svm_blocks==-1: # divide trials into blocks based on the engagement state
        svmb = 2
    else:
        svmb = svm_blocks
    br = range(svmb)    
else:
    br = [np.nan]

    
for iblock in br: # iblock=0; iblock=np.nan
#     for icre in np.arange(0, len(cre2ana_all)): # icre=0

    #%% Set vars for the analysis

#         cre2ana = cre2ana_all[icre]
#         print(f'\n=============== Setting all_sess for {cre2ana} ===============\n')


    #%% Load the pickle file that includes the list of sessions and metadata
    '''
    pkl = open(filen, 'rb')
    # dict_se = pickle.load(pkl)
    metadata_basic = pickle.load(pkl)

    list_all_sessions_valid = metadata_basic[metadata_basic['valid']]['session_id'].unique()
    print(f'number of {cre2ana} sessions: {list_all_sessions_valid.shape[0]}')
    # list_all_sessions_valid = dict_se['list_all_sessions_valid']
    # list_all_experiments_valid = dict_se['list_all_experiments_valid']
    # print(f'number of sessions: {len(list_all_sessions_valid)}')
    '''


    #%% Set all_sess: includes svm results from all sessions of a given cre line, all concatenated

    all_sess = pd.DataFrame([], columns=cols)

    for isess in np.arange(0, len(list_all_sessions_valid)):  # isess=0

        session_id = int(list_all_sessions_valid[isess])
#         data_list = metadata_basic[metadata_basic['session_id'].values==session_id]

        experiment_ids = list_all_experiments[isess] # we want to have the list of all experiments for each session regardless of whethere they were valid or not... this way we can find the same plane across all sessions.

        # Note: we cant use metadata from allensdk because it doesnt include all experiments of a session, it only includes the valid experiments, so it wont allow us to set the metadata for all experiments.
        # session_id = sessions_ctDone[isess]
        metadata_now = metadata_valid[metadata_valid['ophys_session_id']==session_id]
        experiment_ids_valid = metadata_now['ophys_experiment_id'].values # the problem is that this wont give us the list of all experiments for a session ... it only includes the valid experiments.
        experiment_ids_valid = np.sort(experiment_ids_valid)
        print(f'\n{sum(np.in1d(experiment_ids, experiment_ids_valid)==False)} of the experiments are invalid!\n')

        print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess, len(list_all_sessions_valid)))



        #%% Set data_list: metadata for this session including a valid column

        data_list = metadata_all[metadata_all['ophys_session_id']==session_id]
        data_list['valid'] = False
        data_list.loc[data_list['ophys_experiment_id'].isin(experiment_ids_valid), 'valid'] = True
        # sort by area and depth
        data_list = data_list.sort_values(by=['targeted_structure', 'imaging_depth'])


        #%% Run the main function
        
        this_sess = svm_images_main_post(session_id, data_list, svm_blocks, iblock, dir_svm, frames_svm, time_win, trial_type, to_decode, same_num_neuron_all_planes, cols, analysis_dates, use_events, doPlots=0)

        all_sess = all_sess.append(this_sess) 

    print(len(all_sess))    
    all_sess



    ##################################################################
    #%% Save all_sess
    ################################################################## 

    # Set input_vars dataframe so you know what vars were used in your analysis
    cols2 = np.array(['frames_svm', 'time_win', 'use_events', 'trial_type', 'to_decode', 'same_num_neuron_all_planes', 'project_codes'])
    input_vars = pd.DataFrame([], columns=cols2)
    input_vars.at[0, cols2] =  frames_svm, time_win, use_events, trial_type, to_decode, same_num_neuron_all_planes, project_codes

#         cols2 = np.array(['iblock', 'frames_svm', 'cre2ana', 'time_win', 'trial_type', 'to_decode', 'same_num_neuron_all_planes', 'project_codes'])
#         input_vars = pd.DataFrame([], columns=cols2)
#         input_vars.at[0, cols2] =  iblock, frames_svm, cre2ana, time_win, trial_type, to_decode, same_num_neuron_all_planes, project_codes


    # Set the name of the h5 file for saving all_sess
    e = 'events_' if use_events else ''
    if trial_type=='changes_vs_nochanges': # change, then no-change will be concatenated
        svmn = f'{e}svm_decode_changes_from_nochanges' # 'svm_gray_omit'
    elif trial_type=='hits_vs_misses': # 
        svmn = f'{e}svm_decode_hits_from_misses'                
    else:
        svmn = f'{e}svm_decode_{to_decode}_image_from_{trial_type}' # 'svm_gray_omit'
    
    if use_balanced_trials:
        svmn = f'{svmn}_equalTrs'
        
    now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

    if same_num_neuron_all_planes:
        name = 'all_sess_%s_sameNumNeuronsAllPlanes' %(svmn) 
    else:
        name = 'all_sess_%s' %(svmn) 

    b = '_'.join(project_codes)
    a = f'frames{frames_svm[0]}to{frames_svm[-1]}'
#     a = f'{cre2ana}_frames{frames_svm[0]}to{frames_svm[-1]}'  # b = '_'.join([str(session_numbers[i]) for i in range(len(session_numbers))])

    if ~np.isnan(svm_blocks):
        if svm_blocks==-1: # divide trials into blocks based on the engagement state
            word = 'engaged'
        else:
            word = 'block'

        a = f'{a}_{word}{iblock}'

    name = f'{name}_{a}_{b}_{now}'
#     print(name)


    if saveResults:
        print('\n\nSaving all_sess.h5 file')
        allSessName = os.path.join(dir_svm, name + '.h5') # os.path.join(d, svmn+os.path.basename(pnevFileName))
        print(allSessName)

        # save to a h5 file                    
        all_sess.to_hdf(allSessName, key='all_sess', mode='w')    

        # save input_vars to the h5 file
        input_vars.to_hdf(allSessName, key='input_vars', mode='a')





#%% rename svm files to add frames_svm to the file name
'''
name = '(.*)_%s_2020.' %(svmn)
svmName ,_ = all_sess_set_h5_fileName(name, dir_svm, all_files=1)

for i in range(len(svmName)):

    oldname = svmName[i]
    b = os.path.basename(oldname)
    ind = b.find('_2020')
    newname = os.path.join(dir_svm, f'{b[:ind]}_frames0to5{b[ind:]}')
    print(newname)
    print(oldname)

    os.rename(oldname, newname)
'''    




