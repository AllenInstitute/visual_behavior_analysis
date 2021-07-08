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

# Note: the variable names 'to_decode' and 'trial_type' are confusing. The names really only make sense when we are decoding images (ie when trial_type is images/changes/omissions), in which case they mean we are decoding to_decode image (eg current image) from trial_type (eg images); otherwise, to_decode is useless (we just default it to 'current') and trial_type indicates what was decoded from what (eg hits_vs_misses)
to_decode = 'current' # 'current' (default): decode current image.    'previous': decode previous image.    'next': decode next image.     # remember for omissions, you cant do "current", bc there is no current image, it has to be previous or next!
trial_type = 'baseline_vs_nobaseline' #'omissions' #'baseline_vs_nobaseline' #'hits_vs_misses' #'changes_vs_nochanges' # 'images_omissions', 'images', 'changes', 'omissions' # what trials to use for SVM analysis # the population activity of these trials at time time_win will be used to decode the image identity of flashes that occurred at their time 0 (if to_decode='current') or 750ms before (if to_decode='previous').   # if 'changes_vs_nochanges', we will decode image changes from no changes; in this case set to_decode to 'current', but it doesnt really matter. # 'baseline_vs_nobaseline' # decode activity at each frame vs. baseline (ie the frame before omission unless use_spont_omitFrMinus1 = 1 (see below))
#### NOTE: svm codes will result in decoding 9 classes (8 images + omissions) when to_decode='previous' and trial_type='images'. (it wont happen when to_decode='current' because above we only include images for trial_type='images'; it also wont happen when trial_type='omissions' or 'changes', because changes and omissions are not preceded by omissions (although rarely we do see double omissions))
# if you want to also decode omissions (in addition to the 8 images) when to_decode='current', you should set trial_type='images_omissions'; HOWEVER, I dont think it's a good idea to mix image and omission aligned traces because omission aligned traces may have prediction/error signal, so it wont be easy to interpret the results bc we wont know if the decoding reflects image-evoked or image-prediciton/error related signals.

use_events = 1 #0 #1 # whether to run the analysis on detected events (inferred spikes) or dff traces.
svm_blocks = -100 #-1 #-100 # 2 # -101: run the analysis only on engaged trials # -1: divide trials based on engagement # 2 # number of trial blocks to divide the session to, and run svm on. # set to -100 to run svm analysis on the whole session

engagement_pupil_running = 0 # applicable when svm_blocks=-1 # np.nan or 0,1,2 for engagement, pupil, running: which metric to use to define engagement? 
use_spont_omitFrMinus1 = 1 # applicable when trial_type='baseline_vs_nobaseline' # if 0, classify omissions against randomly picked spontanoues frames (the initial gray screen); if 1, classify omissions against the frame right before the omission 

project_codes = 'VisualBehaviorMultiscope' #'VisualBehavior' # has to only include 1 project
# project_codes : ['VisualBehaviorMultiscope', 'VisualBehaviorTask1B', 'VisualBehavior', 'VisualBehaviorMultiscope4areasx2d']


# Note: in the cases below, we are decoding baseline from no-baseline , hits from misses, and changes from no-changes, respectively. 
# since we are not decoding images, to_decode really doesnt mean anything (when decoding images, to_decode means current/next/previous image) and we just set it to 'current' so it takes some value.
if trial_type == 'baseline_vs_nobaseline' or trial_type == 'hits_vs_misses' or trial_type == 'changes_vs_nochanges':
    to_decode = 'current'
    use_balanced_trials = 1 #only effective when num_classes=2 #if 1, use same number of trials for each class; only applicable when we have 2 classes (binary classification).
else: # decoding images
    use_balanced_trials = 0 #1 #only effective when num_classes=2 #if 1, use same number of trials for each class; only applicable when we have 2 classes (binary classification).
    

    
    
#%% Use March 2021 data release sessions

# experiments_table = loading.get_filtered_ophys_experiment_table()
experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')
# metadata_valid = experiments_table[experiments_table['project_code']=='VisualBehaviorMultiscope'] # multiscope sessions

# get those rows of experiments_table that are for a specific project code
metadata_valid = experiments_table[experiments_table['project_code'].isin([project_codes])]


# Use the new list of sessions that are de-crosstalked and will be released in March 2021
# metadata_meso_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/meso_decrosstalk/meso_experiments_in_release.csv'
# metadata_valid = pd.read_csv(metadata_meso_dir)

sessions_ctDone = metadata_valid['ophys_session_id'].unique()
# sessions_ctDone.shape

list_all_sessions_valid = sessions_ctDone
print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')


    
    
    
#%% Define slurm vars   

from simple_slurm import Slurm

# define the job record output folder
stdout_location = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ClusterJobs/SVMJobs' #os.path.join(os.path.expanduser("~"), 'job_records')

# make the job record location if it doesn't already exist
os.mkdir(stdout_location) if not os.path.exists(stdout_location) else None

    
# define the conda environment
conda_environment = 'visbeh'

jobname0 = 'SVM'
jobname = f'{jobname0}'
# jobname = f'{jobname0}:{session_id}'


python_file = r"/home/farzaneh.najafi/analysis_codes/visual_behavior_analysis/visual_behavior/decoding_population/svm_images_main_pre_pbs.py" #r"/home/farzaneh.najafi/analysis_codes/multiscope_fn/svm_main_pbs.py"




# build the python path
# this assumes that the environments are saved in the user's home directory in a folder called 'anaconda3'
# this will be user setup dependent.
python_path = os.path.join(
    os.path.expanduser("~"), 
    'anaconda3', 
    'envs', 
    conda_environment,
    'bin',
    'python'
)    
    
    
print(python_path)


# instantiate a Slurm object    
slurm = Slurm(
#     array = range(len(list_all_sessions_valid)),
    cpus_per_task = 4, #4
    job_name = jobname,
    output = f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    partition = 'braintv',
    mem = '24g', #'24g'
    time = '120:00:00'
    )
# c = 1

print(slurm)

# call the `sbatch` command to run the jobs
# slurm.sbatch('python ../demo_python_scripts/simple_demo.py ' + Slurm.SLURM_ARRAY_TASK_ID)






#%% Loop through valid sessions to perform the analysis

# example active session: isess = -2; session_id = 914639324; isess = -30; session_id = 981845703
cnt_sess = -1     
for isess in range(len(list_all_sessions_valid)): # [0,1]: # isess = -35 # session_id = list_all_sessions_valid[0] #[num_valid_exps_each_sess == 8][0]
    
    session_id = int(list_all_sessions_valid[isess])

    
    cnt_sess = cnt_sess + 1    
    print('\n\n======================== %d: session %d out of %d sessions ========================\n' %
          (session_id, cnt_sess, len(list_all_sessions_valid)))    

    
    
    
    
    ##### call the `sbatch` command to run the jobs
#     slurm.sbatch(f'{python_path} {python_file} --isess {isess} --use_events {use_events} --to_decode {to_decode} --trial_type {trial_type} --svm_blocks {svm_blocks} --engagement_pupil_running {engagement_pupil_running} --use_spont_omitFrMinus1 {use_spont_omitFrMinus1} --use_balanced_trials {use_balanced_trials}') #  + Slurm.SLURM_ARRAY_TASK_ID

    slurm.sbatch('{} {} --isess {} --project_codes {} --use_events {} --to_decode {} --trial_type {} --svm_blocks {} --engagement_pupil_running {} --use_spont_omitFrMinus1 {} --use_balanced_trials {}'.format(
        python_path,
        python_file,
        isess,
        project_codes,
        use_events,
        to_decode,
        trial_type,
        svm_blocks,
        engagement_pupil_running,
        use_spont_omitFrMinus1,
        use_balanced_trials,
            )
                )
    
#     slurm.sbatch('{} ../demo_python_scripts/plotting_demo.py --frequency {} --save-loc {}'.format(
#             python_path,
#             frequency,
#             plot_save_location,
#         )    
    
    

    
    
    

    
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