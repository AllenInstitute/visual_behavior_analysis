#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
After SVM analysis is already done and the results are already saved (by running the svm_image_main_pbs code), 
run this script to create all_sess, a dataframe in which the svm results of all sessions are concatenated. We need this to make plots for the svm (images) analysis.

This script calls the function: svm_images_plots_main.py

After this script, run svm_images_plots_setVars.py to make plots.

Note: run this script for individual project codes to save their all_sess files.


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
import sys

import visual_behavior.data_access.loading as loading
from visual_behavior.data_access import utilities
from svm_images_plots_main import *

# import warnings
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) # ignore some warnings!


#%% Get SVM output for each cell type, and each frames_svm.


#%% Set vars

project_codes = ['VisualBehaviorMultiscope'] # ['VisualBehaviorMultiscope'] # ['VisualBehaviorMultiscope', 'VisualBehaviorTask1B', 'VisualBehavior', 'VisualBehaviorMultiscope4areasx2d']

to_decode = 'current' #'next' # 'current' (default): decode current image.    'previous': decode previous image.    'next': decode next image.
trial_type = 'changes' #'changes' #'baseline_vs_nobaseline' #'hits_vs_misses' #'changes_vs_nochanges' #'omissions' # 'omissions', 'images', 'changes' # what trials to use for SVM analysis # the population activity of these trials at time time_win will be used to decode the image identity of flashes that occurred at their time 0 (if to_decode='current') or 750ms before (if to_decode='previous'). # 'baseline_vs_nobaseline' # decode activity at each frame vs. baseline (ie the frame before omission unless use_spont_omitFrMinus1 = 1 (see below))

use_events = True #False # whether to run the analysis on detected events (inferred spikes) or dff traces.
svm_blocks = np.nan #-101 #np.nan # -1: divide trials based on engagement #2 # number of trial blocks to divide the session to, and run svm on. # set to np.nan to run svm analysis on the whole session

use_matched_cells = 0 #123 # 0: analyze all cells of an experiment. 123: analyze cells matched in all experience levels (familiar, novel 1 , novel >1); 12: cells matched in familiar and novel 1.  23: cells matched in novel 1 and novel >1.  13: cells matched in familiar and novel >1

# use ['20210722'] for vb-ms, 'current', 'changes', events, svm_blocks=-1, no matched cells
analysis_dates = [''] #['20210722'] #['20210708'] # ['2020507'] #set to [''] if you want to load the latest file. # the date on which the svm files were saved.


# set window for quantifying the responses
if use_events:
    time_win = [0, .4]
else:
    time_win = [0, .55] # 'frames_svm' # set time_win to a string (any string) to use frames_svm as the window of quantification. # time window relative to trial onset to quantify image signal. Over this window class accuracy traces will be averaged.
if trial_type == 'baseline_vs_nobaseline':
    time_win = [0, .75] # we use the entire time after omission because the omission responses slowly go up


same_num_neuron_all_planes = 0
saveResults = 1

# engagement_pupil_running = 0 # applicable when svm_blocks=-1 # np.nan or 0,1,2 for engagement, pupil, running: which metric to use to define engagement? 


#%%    
    
use_spont_omitFrMinus1 = 1 # applicable when trial_type='baseline_vs_nobaseline' # if 0, classify omissions against randomly picked spontanoues frames (the initial gray screen); if 1, classify omissions against the frame right before the omission 
    
# Note: in the cases below, we are decoding baseline from no-baseline , hits from misses, and changes from no-changes, respectively. 
# since we are not decoding images, to_decode really doesnt mean anything (when decoding images, to_decode means current/next/previous image) and we just set it to 'current' so it takes some value.
if trial_type == 'baseline_vs_nobaseline' or trial_type == 'hits_vs_misses' or trial_type == 'changes_vs_nochanges':
    to_decode = 'current'
    use_balanced_trials = 1 #only effective when num_classes=2 #if 1, use same number of trials for each class; only applicable when we have 2 classes (binary classification).
else: # decoding images
    use_balanced_trials = 0 #1 #only effective when num_classes=2 #if 1, use same number of trials for each class; only applicable when we have 2 classes (binary classification).
    

    
# Note: trials df has a much longer time_trace (goes up to 4.97) compared to stimulus df (goes up to .71), so frames_svm ends up being 1 element longer for trials df (ie when decoding hits from misses) compared to stimulus df (ie when decoding the images)    
if project_codes != ['VisualBehaviorMultiscope']:
    frames_svm_all = [np.arange(-15,23)] #[[-3,-2,-1], [0,1,2,3,4,5]]
else:
    # NOTE: some mesoscope sessions were recorded at a different frame rate (9hz, frame duration: 0.109ms vs. 10.7hz, frame duration: 0.093ms), as a result their frames_svm are -4 to 6 instad of -5 to 7.
    # example mesoscope sessions with this problem: 1050231786 1050597678 1051107431
    frames_svm_all = [np.arange(-5,8)]
    
if trial_type=='hits_vs_misses':
    frames_svm_all[0] = np.concatenate((frames_svm_all[0], [frames_svm_all[0][-1] + 1]))  # frames_svm_all[0][-1] = frames_svm_all[0][-1] + 1

frames_svm = frames_svm_all[0] #[-3,-2,-1] # [0,1,2,3,4,5] #  which svm files to load (this will be used to set svm file name) # (frames_svm also gets saved in svm_vars if svm could be run successfully on a session)
# print(f'Analyzing: {cre2ana}, {frames_svm}')
    
doPlots = 0
num_planes = 8

if project_codes != ['VisualBehaviorMultiscope']:
    frame_dur = np.array([.032])
else:
    frame_dur = np.array([.093])
# trace_time = np.array([-0.46631438, -0.3730515 , -0.27978863, -0.18652575, -0.09326288, 0.,  0.09326288,  0.18652575,  0.27978863,  0.3730515 , 0.46631438,  0.55957726,  0.65284013])
# trace_time = array([-0.48482431, -0.45250269, -0.42018107, -0.38785944, -0.35553782,
#        -0.3232162 , -0.29089458, -0.25857296, -0.22625134, -0.19392972,
#        -0.1616081 , -0.12928648, -0.09696486, -0.06464324, -0.03232162,
#         0.        ,  0.03232162,  0.06464324,  0.09696486,  0.12928648,
#         0.1616081 ,  0.19392972,  0.22625134,  0.25857296,  0.29089458,
#         0.3232162 ,  0.35553782,  0.38785944,  0.42018107,  0.45250269,
#         0.48482431,  0.51714593,  0.54946755,  0.58178917,  0.61411079,
#         0.64643241,  0.67875403,  0.71107565,  0.74339727])
# samps_bef = 5 #np.argwhere(trace_time==0)[0][0] # 
# samps_aft = 8 #len(trace_time)-samps_bef #




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

#%% Use only those project codes that will be used for the paper
experiments_table = loading.get_platform_paper_experiment_table()
# len(experiments_table)
# experiments_table.keys()
# experiments_table.project_code.unique()

experiments_table = experiments_table.reset_index('ophys_experiment_id')

# get those rows of experiments_table that are for a specific project code
metadata_valid = experiments_table[experiments_table['project_code']==project_codes[0]]
metadata_valid = metadata_valid.sort_values('ophys_session_id')

# Use the new list of sessions that are de-crosstalked and will be released in March 2021
list_all_sessions_valid = metadata_valid['ophys_session_id'].unique()
# list_all_sessions_valid = sessions_ctDone

print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')




################################################################################################
if use_matched_cells!=0: 

    #%% get matched cells to redeine list of valid sessions

    #%% New method: match across the following 3 experience levels: last familiar active; Novel 1; second novel active
    # from Marina's notebook: visual_behavior_analysis/notebooks/211008_validate_filtering_criteria_and_check_exposure_number.ipynb

    cells_table = loading.get_cell_table()
    cells_table.shape

    # limit to cells matched in all 3 experience levels, only considering last Familiar and second Novel active
    df = utilities.limit_to_last_familiar_second_novel_active(cells_table) # important that this goes first
    df = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(df)

    # Sanity checks:        
    # count numbers of expts and cells¶
    # there should be the same number of experiments for each experience level, and a similar number of cells        
    # now the number of cell_specimen_ids AND the number of experiment_ids are the same across all 3 experience levels, because we have limited to only the last Familiar and first Novel active sessions
    # this is the most conservative set of experiments and cells - matched cells across experience levels, only considering the most recent Familiar and Novel >1 sessions        

#     utilities.count_mice_expts_containers_cells(df)
    print(utilities.count_mice_expts_containers_cells(df)['n_cell_specimen_id'])

    '''
    # there are only 3 experience levels per container
    df.groupby(['ophys_container_id', 'experience_level']).count().reset_index().groupby(['ophys_container_id']).count().experience_level.unique()        
    # there should only be 1 experiment per experience level
    df.groupby(['ophys_container_id', 'experience_level', 'ophys_experiment_id']).count().reset_index().groupby(['ophys_container_id', 'experience_level']).count().ophys_experiment_id.unique()
    '''        

    list_all_sessions_valid_matched = df[df['project_code']==project_codes[0]]['ophys_session_id'].unique() # note that if you get ophys experiments it has to be a multiplication of 3. (not ophys sessions.)
    list_all_sessions_valid_matched = np.sort(list_all_sessions_valid_matched)


    
    
    
    # TEMPORARY SOLUTION: remove this session; 
    # SOLUTION: switch to sdk branch rc/2.13.2 to fix : dataset.running_speed has issues.
    print(len(list_all_sessions_valid_matched))
    list_all_sessions_valid_matched = list_all_sessions_valid_matched[list_all_sessions_valid_matched!=795625712] # it's a novel 1 session, VB project
    print(len(list_all_sessions_valid_matched))

    
    
    
    
    b = len(list_all_sessions_valid_matched) / len(list_all_sessions_valid)
    print(f'{len(list_all_sessions_valid_matched)}/{len(list_all_sessions_valid)}, {b*100:.0f}% of {project_codes} sessions have matched cells in the 3 experience levels.')

    
    ### redefine list all sessions valid if using matched cells
    list_all_sessions_valid = list_all_sessions_valid_matched
    matched_cells = df
    
    
print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')

################################################################################################




####################################################################################################################
#### Set metadata_all : includes metadata for all the 8 experiments of a session, even if they are failed.
####################################################################################################################

# get the list of all the 8 experiments for each session; we need this to set the depth and area for all experiments, since we need the metadata for all the 8 experiments for our analysis even if we dont analyze some experiments. this is because we want to know which experiment belongs to which plane. we later make plots for each plot individually.
experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True) # , release_data_only=False
experiments_table = experiments_table.reset_index('ophys_experiment_id')
# experiments_table.shape

metadata_all = experiments_table[experiments_table['ophys_session_id'].isin(list_all_sessions_valid)==True] # metadata_all = experiments_table[np.in1d(experiments_table['ophys_session_id'].values, list_all_sessions_valid)]
metadata_all = metadata_all.sort_values('ophys_session_id')
# metadata_all.shape
# metadata_all.shape[0]/8



# set the list of experiments for each session in list_all_sessions_valid
if project_codes != ['VisualBehaviorMultiscope']:
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




#%% Set the columns in dataframe all_sess

cols0 = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'experience_level', 'area', 'depth', 'n_trials', 'n_neurons', 'cell_specimen_id', 'frame_dur', 'meanX_allFrs', 'stdX_allFrs', 'av_train_data', 'av_test_data', 'av_test_shfl', 'av_test_chance', 'sd_train_data', 'sd_test_data', 'sd_test_shfl', 'sd_test_chance', 'peak_amp_trainTestShflChance', 'bl_pre0'])
if same_num_neuron_all_planes:
    cols = np.concatenate((cols0, ['population_sizes_to_try'])) 
else:
    cols = np.concatenate((cols0, ['av_w_data', 'av_b_data']))

if svm_blocks==-101:
    br = [np.nan]    
elif ~np.isnan(svm_blocks):
    if svm_blocks==-1: # divide trials into blocks based on the engagement state
        svmb = 2
    else:
        svmb = svm_blocks
    br = range(svmb)    
else:
    br = [np.nan]
print(br)    




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

    for isess in np.arange(0, len(list_all_sessions_valid)):  # isess=0 # isess=38

        session_id = int(list_all_sessions_valid[isess])
#         data_list = metadata_basic[metadata_basic['session_id'].values==session_id]

        print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess, len(list_all_sessions_valid)))

        experiment_ids = list_all_experiments[isess] # we want to have the list of all experiments for each session regardless of whethere they were valid or not... this way we can find the same plane across all sessions.

        # Note: we cant use metadata from allensdk because it doesnt include all experiments of a session, it only includes the valid experiments, so it wont allow us to set the metadata for all experiments.
        # session_id = sessions_ctDone[isess]
        metadata_now = metadata_valid[metadata_valid['ophys_session_id']==session_id]        

        # If using matched cells, reset experiment ids valid:
        # Because some planes (experiments) are not passed for last familiar active or second novel active, after matching cells, some planes (experiments) of a mesoscope session become invalid, so we need to reset experiment_ids_valid.
        # For VB (or task1B) projects though, if an experiment and its session become invalid at the same time, so no need to reset the valid experiments.    
        if use_matched_cells!=0:
            experiment_ids_valid = matched_cells[matched_cells['ophys_session_id']==session_id]['ophys_experiment_id'].unique()
        else:
            experiment_ids_valid = metadata_now['ophys_experiment_id'].values # the problem is that this wont give us the list of all experiments for a session ... it only includes the valid experiments.

        experiment_ids_valid = np.sort(experiment_ids_valid)

        if use_matched_cells!=0:
            v = metadata_now['ophys_experiment_id'].values
            print(f"Before matching cells across experience levels {sum(np.in1d(experiment_ids, v)==False)} experiments were invalid; but now:")
        print(f'\n{sum(np.in1d(experiment_ids, experiment_ids_valid)==False)} of the experiments are invalid!\n')


        #%% Set data_list: metadata for this session including a valid column

        data_list = metadata_all[metadata_all['ophys_session_id']==session_id]
        data_list['valid'] = False
        data_list.loc[data_list['ophys_experiment_id'].isin(experiment_ids_valid), 'valid'] = True
        # sort by area and depth
        data_list = data_list.sort_values(by=['targeted_structure', 'imaging_depth'])


        #%% Run the main function
        
        this_sess = svm_images_plots_main(session_id, data_list, svm_blocks, iblock, dir_svm, frames_svm, time_win, trial_type, to_decode, same_num_neuron_all_planes, cols, analysis_dates, project_codes, use_spont_omitFrMinus1, use_events, doPlots, use_matched_cells)

        
        all_sess = all_sess.append(this_sess) 

    
    #######################################
    print(f'\n\nFinal: length of all_sess: {len(all_sess)}\n\n')
    
    # check if there are any nan values in all_sess, so you can investigate why!
    if project_codes == ['VisualBehaviorMultiscope']:
        
        ns = int(len(all_sess)/8) # number of sessioins
        d = np.reshape(all_sess['date'].values, (8, ns), order='F') # 8 x 58
        dr = np.reshape(d, (8*ns, 1), order='F').squeeze() # 464
        ei = np.array([isinstance(dr[i], float) for i in range(len(dr))]) # are there any nan (float) dates? # 464
        eir = np.reshape(ei, (8, ns), order='F') # 8 x 58
        n_valid_exp_per_sess = np.sum(eir, axis=0)
        alleinv = n_valid_exp_per_sess == 8 # 58 # find sessions with all experiments invalid (sum will be 8, if all the experiment dates were of type float, ie invalid)
#         alleinv
        aa = alleinv.any()
        
        ss = np.reshape(all_sess['session_id'].values, (8, ns), order='F')[0,:] # 58
        si = ss[np.array(alleinv)] # these sessions have all experiments as invalid; investigate them!
        if aa==True:
            print(f'{len(si)} sessions have all experiments as invalid:\n\n{si}! investigate them!')        
        
    else:
        l = np.array([len(all_sess['date'].values[i]) for i in range(len(all_sess))])
        aa = (l != 10).any()
        
    # if there are nan rows in all_sess, exit the analysis to investigate why those experiments dont exist!    
    if aa:
        sys.exit(f'There are some invalid rows in all_sess! examine those experiments!')
    else:
        print(f'\nAll rows in all_sess are valid! Save all_sess file!\n')

#     all_sess

        
        
        
        
        
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
        svmn = f'{e}svm_decode_changes_from_nochanges'
    elif trial_type=='hits_vs_misses':
        svmn = f'{e}svm_decode_hits_from_misses'        
    elif trial_type == 'baseline_vs_nobaseline':
        svmn = f'{e}svm_decode_baseline_from_nobaseline' #f'{e}svm_gray_omit' #svm_decode_baseline_from_nobaseline
        if use_spont_omitFrMinus1==0:
            svmn = svmn + '_spontFrs'
    else:
        svmn = f'{e}svm_decode_{to_decode}_image_from_{trial_type}' # 'svm_gray_omit'
        

    if use_matched_cells==123:
        svmn = svmn + '_matched_cells_FN1N2' #Familiar, N1, N+1
    elif use_matched_cells==12:
        svmn = svmn + '_matched_cells_FN1'
    elif use_matched_cells==23:
        svmn = svmn + '_matched_cells_N1Nn'
    elif use_matched_cells==13:
        svmn = svmn + '_matched_cells_FNn'        

        
    if svm_blocks==-101: # run svm analysis only on engaged trials; redifine df_data only including the engaged rows
        svmn = f'{svmn}_only_engaged'
        
    if use_balanced_trials:
        svmn = f'{svmn}_equalTrs'
        
    print(svmn)

    
    now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

    if same_num_neuron_all_planes:
        name = 'all_sess_%s_sameNumNeuronsAllPlanes' %(svmn) 
    else:
        name = 'all_sess_%s' %(svmn) 

    b = '_'.join(project_codes)
    a = f'frames{frames_svm[0]}to{frames_svm[-1]}'
#     a = f'{cre2ana}_frames{frames_svm[0]}to{frames_svm[-1]}'  # b = '_'.join([str(session_numbers[i]) for i in range(len(session_numbers))])

    if ~np.isnan(svm_blocks) and svm_blocks!=-101:
        if svm_blocks==-1: # divide trials into blocks based on the engagement state
            word = 'engaged'
        else:
            word = 'block'

        a = f'{a}_{word}{iblock}'

    name = f'{name}_{a}_{b}_{now}'
    print(name)


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




