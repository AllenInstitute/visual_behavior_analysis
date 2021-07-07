#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run svm_init_images_pre.py to save the pickle file which includes session and metadata information. This pickle file will be loaded here.

Created on Thu Oct 9 12:29:43 2020
@author: farzaneh
"""

import numpy as np
import visual_behavior.data_access.loading as loading
import numpy as np
import os
# import pickle


#%% Define vars for svm_images analysis

to_decode = 'current' # 'current' (default): decode current image.    'previous': decode previous image.    'next': decode next image.     # remember for omissions, you cant do "current", bc there is no current image, it has to be previous or next!
trial_type = 'omissions_baseline' #'omissions' #'omissions_baseline' #'hits_vs_misses' #'changes_vs_nochanges' # 'images_omissions', 'images', 'changes', 'omissions', 'changes_vs_nochanges', 'hits_vs_misses' # what trials to use for SVM analysis # the population activity of these trials at time time_win will be used to decode the image identity of flashes that occurred at their time 0 (if to_decode='current') or 750ms before (if to_decode='previous').   # if 'changes_vs_nochanges', we will decode image changes from no changes; in this case set to_decode to 'current', but it doesnt really matter. # 'omissions_baseline' # decode activity at each frame vs. baseline (ie the frame before omission unless use_spont_omitFrMinus1 = 1 (see below))
#### NOTE: svm codes will result in decoding 9 classes (8 images + omissions) when to_decode='previous' and trial_type='images'. (it wont happen when to_decode='current' because above we only include images for trial_type='images'; it also wont happen when trial_type='omissions' or 'changes', because changes and omissions are not preceded by omissions (although rarely we do see double omissions))
# if you want to also decode omissions (in addition to the 8 images) when to_decode='current', you should set trial_type='images_omissions'; HOWEVER, I dont think it's a good idea to mix image and omission aligned traces because omission aligned traces may have prediction/error signal, so it wont be easy to interpret the results bc we wont know if the decoding reflects image-evoked or image-prediciton/error related signals.

use_events = 1 #0 #1 # whether to run the analysis on detected events (inferred spikes) or dff traces.
svm_blocks = -100 #-1 #-100 # 2 # -101: run the analysis only on engaged trials # -1: divide trials based on engagement # 2 # number of trial blocks to divide the session to, and run svm on. # set to -100 to run svm analysis on the whole session
use_balanced_trials = 1 #1 #only effective when num_classes=2 #if 1, use same number of trials for each class; only applicable when we have 2 classes (binary classification).

engagement_pupil_running = 1 # np.nan or 0,1,2 for engagement, pupil, running: which metric to use to define engagement? only effective if svm_blocks=-1
use_spont_omitFrMinus1 = 1 # applicable when trial_type='omissions_baseline' # if 0, classify omissions against randomly picked spontanoues frames (the initial gray screen); if 1, classify omissions against the frame right before the omission 


# cre2ana = 'slc' # slc, sst, vip # will be used if session_numbers[0]<0 (we will use dataset and responseAnalysis (instead of the concatenated dfs) to set stim_response_df)
# project_codes = ['VisualBehaviorMultiscope'] # ['VisualBehaviorMultiscope', 'VisualBehaviorTask1B', 'VisualBehavior', 'VisualBehaviorMultiscope4areasx2d']
project_codes = ['VisualBehavior']
# project_codes = ['VisualBehaviorMultiscope' , 'VisualBehavior']



#%% Use March 2021 data release sessions

# experiments_table = loading.get_filtered_ophys_experiment_table()
experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')
# metadata_valid = experiments_table[experiments_table['project_code']=='VisualBehaviorMultiscope'] # multiscope sessions

# get those rows of experiments_table that are for a specific project code
metadata_valid = experiments_table[experiments_table['project_code'].isin(project_codes)]


# Use the new list of sessions that are de-crosstalked and will be released in March 2021
# metadata_meso_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/meso_decrosstalk/meso_experiments_in_release.csv'
# metadata_valid = pd.read_csv(metadata_meso_dir)

sessions_ctDone = metadata_valid['ophys_session_id'].unique()
# sessions_ctDone.shape

list_all_sessions_valid = sessions_ctDone
print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')


#%% Get the list of all 8 experiments for each session, regardless of being valid

experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')

metadata_all = experiments_table[experiments_table['ophys_session_id'].isin(list_all_sessions_valid)==True] # metadata_all = experiments_table[np.in1d(experiments_table['ophys_session_id'].values, list_all_sessions_valid)]
metadata_all.shape
metadata_all.shape[0]/8


# set the list of experiments for each session in list_all_sessions_valid
if project_codes == ['VisualBehavior']:
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




#%%
'''
# below is not needed anymore; just set session_numbers to a negative value so we load stim response df for cre lines (not the concat version that does not exist for all mice)
session_numbers = [-10] #[4] # for slc, set to a value <0 (because for slc the concatenated stim response dfs could not be saved, so we have to load a different metadata file below)


#%% Load the pickle file which includes session and metadata information.

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
dir_svm = os.path.join(dir_server_me, 'SVM')
a = '_'.join(project_codes)
if session_numbers[0]<0:
    b = cre2ana
else:
    b = '_'.join([str(session_numbers[i]) for i in range(len(session_numbers))])
filen = os.path.join(dir_svm, f'metadata_basic_{a}_{b}')
print(filen)


### load the pickle file
if session_numbers[0]>0:
    pkl = open(filen, 'rb')
    dict_se = pickle.load(pkl)
    # metadata_basic = pickle.load(pkl)

    list_all_sessions_valid = dict_se['list_all_sessions_valid']
else:
    pkl = open(filen, 'rb')
    metadata_basic = pickle.load(pkl)
    list_all_sessions_valid = metadata_basic[metadata_basic['valid']]['session_id'].unique()
'''



    
#%% Set vars for the cluster

import sys
from pbstools import PythonJob # flake8: noqa: E999
#sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
#import platform
#if platform.system() == 'Linux':
#    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')


python_file = r"/home/farzaneh.najafi/analysis_codes/visual_behavior_analysis/visual_behavior/decoding_population/svm_images_main_pbs.py" #r"/home/farzaneh.najafi/analysis_codes/multiscope_fn/svm_main_pbs.py"


#%%

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ClusterJobs/SVMJobs'
job_settings = {'queue': 'braintv',
                'mem': '24g', #24g
                'walltime': '120:00:00', # 48
                'ppn': 4} #,
#                'jobdir': jobdir,
#                }
job_settings.update({
                'outfile':os.path.join(jobdir, '$PBS_JOBID.out'),
                'errfile':os.path.join(jobdir, '$PBS_JOBID.err'),
                'email': 'farzaneh.najafi@alleninstitute.org',
                'email_options': 'a'
                })




#%% Loop through valid sessions to perform the analysis

# example active session: isess = -2; session_id = 914639324; isess = -30; session_id = 981845703
cnt_sess = -1     
for isess in range(len(list_all_sessions_valid)): # [0,1]: # isess = -35 # session_id = list_all_sessions_valid[0] #[num_valid_exps_each_sess == 8][0]
    
    session_id = int(list_all_sessions_valid[isess])

    
    cnt_sess = cnt_sess + 1    
    print('\n\n======================== %d: session %d out of %d sessions ========================\n' %
          (session_id, cnt_sess, len(list_all_sessions_valid)))    

    
    '''
    #%% Get session imaging data and session trial data
    # get cell response data for this session
    session_data = stimulus_response_data[stimulus_response_data['ophys_session_id']==session_id].copy()

    # get stimulus presentations data for this session
    session_trials = stim_presentations[stim_presentations['ophys_session_id']==session_id].copy()
    # these two dataframes are linked by stimulus_presentations_id and ophys_session_id
            
    python_arg1 = '%s ' %session_data
    python_arg2 = '%s' %session_trials
    '''

    python_arg1 = '%s ' %isess
    python_arg2 = '%s ' %use_events # cast it as boolean!!!
    python_arg3 = '%s ' %to_decode
    python_arg4 = '%s ' %trial_type
    python_arg5 = '%s ' %svm_blocks    
    python_arg6 = '%s ' %engagement_pupil_running
    python_arg7 = '%s ' %use_spont_omitFrMinus1
    python_arg8 = '%s' %use_balanced_trials
    
    print(python_arg1 + python_arg2 + python_arg3 + python_arg4 + python_arg5 + python_arg6 + python_arg7 + python_arg8)
    
    
    #%%    
    jobname = 'SVM'
    job_settings['jobname'] = '%s:%s' %(jobname, session_id)
    
    
    PythonJob(
        python_file,
        python_executable = '/home/farzaneh.najafi/anaconda3/envs/visbeh/bin/python',
        python_args = python_arg1 + python_arg2 + python_arg3 + python_arg4 + python_arg5 + python_arg6 + python_arg7 + python_arg8,
#         python_args = str(session_data) + str(session_trials), # session_id experiment_ids validity_log_all dir_svm frames_svm numSamples saveResults use_ct_traces same_num_neuron_all_planes',
#         python_args = isess, # session_id experiment_ids validity_log_all dir_svm frames_svm numSamples saveResults use_ct_traces same_num_neuron_all_planes',
        conda_env = None,
#        jobname = 'process_{}'.format(session_id),
        **job_settings
    ).run(dryrun=False)
   
    
    

    
    
    

    
###############################################################################################
##### Save a dataframe entries to MONGO and retrieve them later
##### code from Doug: https://gist.github.com/dougollerenshaw/b2ddb9f5e26f33059001f21bae49ea2b
###############################################################################################
'''
#%% save the dataframe to mongo (make every row a 'document' or entry)¶
# This might be slow for a big table, but you only need to do it once

project_codes = ['VisualBehaviorMultiscope'] # ['VisualBehaviorMultiscope', 'VisualBehaviorTask1B', 'VisualBehavior', 'VisualBehaviorMultiscope4areasx2d']
session_numbers = [4]

import numpy as np
import visual_behavior.data_access.loading as loading
stim_response_dfs = loading.get_concatenated_stimulus_response_dfs(project_codes, session_numbers) # stim_response_data has cell response data and all experiment metadata

df = stim_response_dfs
print(df.shape)


from visual_behavior import database

conn = database.Database('visual_behavior_data')
collection_name = f'{project_codes}_{session_numbers}' # 'test_table'
collection = conn['ophys_population_analysis'][collection_name]

print(collection_name)


# if kernel stops, use below to find the last entry (or 1 to the last just to be safe), and resume running the code below to save the remaining data to mongo
# alle = collection.find({})
# print(alle.count())
# print(alle.count() / df.shape[0])
# alle[alle.count()-2]

# 1202985

# for idx,row in df.iterrows():
for idx in np.arange(alle.count()-2, len(df)):  
    row = df.iloc[idx]
    # run every document through the 'clean_and_timestamp' function.
    # this function ensures that all datatypes are good for mongo, then adds an additional field with the current timestamp
    document_to_save = database.clean_and_timestamp(row.to_dict()) 
    database.update_or_create(
        collection  = collection, # this is the collection you defined above. it's the mongo equivalent of a table
        document = document_to_save, # save the document in mongo. Think of every document as a row in your table
        keys_to_check = ['ophys_experiment_id', 'ophys_session_id', 'stimulus_presentations_id', 'cell_specimen_id'], # this will check to make sure that duplicate documents don't exist
    )
    
# close the connection when you're done with it
conn.close()
'''