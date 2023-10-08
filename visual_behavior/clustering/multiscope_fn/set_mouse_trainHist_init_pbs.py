#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:24:19 2019

@author: farzaneh
"""


#%% NOTE:
'''
remember to add to the end of python script that you are calling here the following:
(if it is a function, you need to call the function at the end of python_file)

print(len(sys.argv))
print(sys.argv)

session_id = sys.argv[1]
experiment_ids = sys.argv[2] 
validity_log_all = sys.argv[3]
dir_svm = sys.argv[4]
frames_svm = sys.argv[5]
numSamples = sys.argv[6]
saveResults = sys.argv[7]
same_num_neuron_all_planes = sys.argv[8]


svm_main(session_id, experiment_ids, validity_log_all, dir_svm, frames_svm, numSamples, saveResults, same_num_neuron_all_planes)
'''
    
    
#%% Set vars for the cluster

import os
import sys
from pbstools import PythonJob # flake8: noqa: E999
#sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
#import platform
#if platform.system() == 'Linux':
#    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')

python_file = r"/home/farzaneh.najafi/analysis_codes/visual_behavior_analysis/visual_behavior/clustering/multiscope_fn/set_mouse_trainHist_all2.py"
# r"/home/farzaneh.najafi/analysis_codes/multiscope_fn/set_mouse_trainHist_all2.py"


#%%

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ClusterJobs'
job_settings = {'queue': 'braintv',
                'mem': '4g',
                'walltime': '2:00:00',
                'ppn': 2} #,
#                'jobdir': jobdir,
#                }
job_settings.update({
                'outfile':os.path.join(jobdir, '$PBS_JOBID.out'),
                'errfile':os.path.join(jobdir, '$PBS_JOBID.err'),
                'email': 'farzaneh.najafi@alleninstitute.org',
                'email_options': 'a'
                })

#for experiment_id in experiment_ids:
#    PythonJob(
#        python_file,
#        python_executable='/home/farzaneh.najafi/anaconda3/envs/visbeh/bin/python',
#        python_args=experiment_id, #cache_dir, plot_roi_validation,
#        conda_env=None,
#        jobname='process_{}'.format(experiment_id),
#        **job_settings
#    ).run(dryrun=False)
    


    
#%% 
#import numpy as np    
#
#all_mice_id = np.array([392241, 409296, 411922, 412035, 414146, 429956, 431151, 435431,
#       438912, 440631, 447934, 448366, 449653, 451787, 453911, 456915])    

    
#%%    
jobname = 'mouse_train_hist'
job_settings['jobname'] = jobname


PythonJob(
    python_file,
    python_executable = '/home/farzaneh.najafi/anaconda3/envs/visbeh/bin/python',
#    python_args = all_mice_id,
    conda_env = None,
#        jobname = 'process_{}'.format(session_id),
    **job_settings
).run(dryrun=False)
   
    
    

    
    
