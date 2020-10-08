#%% Set list of sessions to be analyzed

import os
import itertools
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

import visual_behavior.data_access.loading as loading


#%% functions to load cell response data and stimulus presentations metadata for a set of sessions
experiments_table = loading.get_filtered_ophys_experiment_table()


#%% Set "stimulus_response_data" dataframe
%time 
#this can be slow for larger amounts of data. limiting to non-mesoscope production codes is faster.
project_codes = ['VisualBehaviorMultiscope']
session_numbers = [4]

# stimulus_response_data has cell response data and all experiment metadata
stim_response_dfs = loading.get_concatenated_stimulus_response_dfs(project_codes, session_numbers)
stimulus_response_data = stim_response_dfs.merge(experiments_table, on=['ophys_experiment_id', 'ophys_session_id'])
# stimulus_response_data.keys()

# stim_presentations has stimulus presentations metadata for the same set of experiments
stim_presentations = loading.get_concatenated_stimulus_presentations(project_codes, session_numbers)
# stim_presentations.keys()




#%% Set list_all_sessions_valid and list_all_experiments (for all sessions at stage 4)

# #%% lets get the VIP mice
# stimulus_response_data = stimulus_response_data[stimulus_response_data.cre_line=='Vip-IRES-Cre']

# # number of sessions
# s = stimulus_response_data['ophys_session_id'].unique()
# s.shape

list_all_sessions_valid = stimulus_response_data['ophys_session_id'].unique()
# list_all_experiments = [stimulus_response_data[stimulus_response_data['ophys_session_id']==si]['ophys_experiment_id'].unique() for si in list_all_sessions_valid]
# list_all_experiments = np.array(list_all_experiments, dtype=object)
# # [a[i].shape for i in range(len(a))]
print(list_all_sessions_valid.shape)




    
#%% Set vars for the cluster

import sys
from pbstools import PythonJob # flake8: noqa: E999
#sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
#import platform
#if platform.system() == 'Linux':
#    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')


python_file = r"/home/farzaneh.najafi/analysis_codes/visual_behavior_analysis/visual_behavior/decoding_population/svm_main_images_pbs.py" #r"/home/farzaneh.najafi/analysis_codes/multiscope_fn/svm_main_pbs.py"


#%%

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ClusterJobs/SVMJobs'
job_settings = {'queue': 'braintv',
                'mem': '24g',
                'walltime': '48:00:00',
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

#sess_no_omission = []
cnt_sess = -1     

#cols = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'meanX_allFrs', 'stdX_allFrs', 'av_train_data_new', 'av_test_data_new', 'av_test_shfl_new', 'av_test_chance_new', 'sd_train_data_new', 'sd_test_data_new', 'sd_test_shfl_new', 'sd_test_chance_new'])
#all_sess = pd.DataFrame([], columns=cols)

for isess in range(len(list_all_sessions_valid)): # [0,1]: # isess = -5 # session_id = list_all_sessions_valid[0] #[num_valid_exps_each_sess == 8][0]
    
    session_id = int(list_all_sessions_valid[isess])

    
    cnt_sess = cnt_sess + 1    
    print('\n\n======================== %d: session %d out of %d sessions ========================\n' %
          (session_id, cnt_sess+1, len(list_all_sessions_valid)))    

    
    
    #%% Get session imaging data and session trial data
    # get cell response data for this session
    session_data = stimulus_response_data[stimulus_response_data['ophys_session_id']==session_id].copy()

    # get stimulus presentations data for this session
    session_trials = stim_presentations[stim_presentations['ophys_session_id']==session_id].copy()
    # these two dataframes are linked by stimulus_presentations_id and ophys_session_id
    
        
    
    #%%    
    jobname = 'SVM'
    job_settings['jobname'] = '%s:%s' %(jobname, session_id)
    
    
    PythonJob(
        python_file,
        python_executable = '/home/farzaneh.najafi/anaconda3/envs/visbeh/bin/python',
        python_args = str(session_data) + str(session_trials), # session_id experiment_ids validity_log_all dir_svm frames_svm numSamples saveResults use_ct_traces same_num_neuron_all_planes',
#        python_args = isess, # session_id experiment_ids validity_log_all dir_svm frames_svm numSamples saveResults use_ct_traces same_num_neuron_all_planes',
        conda_env = None,
#        jobname = 'process_{}'.format(session_id),
        **job_settings
    ).run(dryrun=False)
   
    
    

