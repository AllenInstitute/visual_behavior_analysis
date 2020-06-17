#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:24:19 2019

@author: farzaneh
"""

import socket
import os
import pickle

if socket.gethostname() == 'OSXLT1JHD5.local': # allen mac
    dirAna = "/Users/farzaneh.najafi/Documents/analysis_codes/"
#    dir0 = '/Users/farzaneh.najafi/OneDrive - Allen Institute/Analysis'

elif socket.gethostname() == 'ibs-farzaneh-ux2': # allen pc
    dirAna = "/home/farzaneh/Documents/analysis_codes/"
#    dir0 = '/home/farzaneh/OneDrive/Analysis'

elif socket.gethostname() == 'hpc-login.corp.alleninstitute.org': # hpc server
    dirAna = "/home/farzaneh.najafi/analysis_codes/"
    
    
dirMs = os.path.join(dirAna, 'multiscope_fn')

os.chdir(dirMs)

from def_funs import *
from def_funs_general import *
from svm_funs import *

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'


#%% Load vars related to list_all_sessions_valid 
# (read the pickle file saved in the script: set_valid_sessions.py)

dir_valid_sess = os.path.join(dir_server_me, 'ValidSessions')
validSessName = os.path.join(dir_valid_sess, 'valid_sessions' + '.pkl')
print(validSessName)
    
    
pkl = open(validSessName, 'rb')
dictNow = pickle.load(pkl)

for k in list(dictNow.keys()):
    exec(k + '= dictNow[k]')

#['list_all_sessions0',
# 'list_sessions_date0',
# 'list_sessions_experiments0',
# 'validity_log_all',
# 'list_all_sessions_valid',
# 'list_all_experiments_valid']   
  
    
#%%
########################################################################################################
################################# Analysis starts here  ################################################
########################################################################################################

#%% Set SVM vars

frames_after_omission = 5 #30 # run svm on how many frames after omission
numSamples = 3 #10 #50
saveResults = 1


kfold = 10
regType = 'l2'

norm_to_max_svm = 1 # normalize each neuron trace by its max
doPlots = 0 # svm regularization plots

softNorm = 1 # soft normalziation : neurons with sd<thAct wont have too large values after normalization
thAct = 5e-4 # will be used for soft normalization #1e-5 
# you did soft normalization when using inferred spikes ... i'm not sure if it makes sense here (using df/f trace); but we should take care of non-active neurons: 
# Set NsExcluded : Identify neurons that did not fire in any of the trials (during ep) and then exclude them. Otherwise they cause problem for feature normalization.

smallestC = 0   
shuffleTrs = False # set to 0 so for each iteration of numSamples, all frames are trained and tested on the same trials# If 1 shuffle trials to break any dependencies on the sequence of trails 
# note about shuffleTrs: if you wanted to actually take effect, you should copy the relevant part from the svm code (svm_omissions) to
# here before you go through nAftOmit frames ALSO do it for numSamples so for each sample different set of testing and training 
# trials are used, however, still all nAftOmit frames use the same set ... 
#... in the current version, the svm is trained and tested on different trials 
# I don't think it matters at all, so I am not going to change it!
cbest0 = np.nan
useEqualTrNums = np.nan
cbestKnown = 0 #cbest = np.nan

    
# make svm dir
dir_svm = os.path.join(dir_server_me, 'SVM')
if not os.path.exists(dir_svm):
    os.makedirs(dir_svm)
                    
                    
#%% Set initial vars

# %matplotlib inline
#get_ipython().magic(u'matplotlib inline')

# To align on omissions, get 40 frames before omission and 39 frames after omission
samps_bef = 40 
samps_aft = 40

num_planes = 8
    
now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

cols = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'traces_aveTrs_time_ns', 'traces_aveNs_time_trs', 'peak_amp_eachN_traceMed', 'peak_timing_eachN_traceMed', 'peak_amp_trs_ns', 'peak_timing_trs_ns', 'peak_h1', 'peak_h2', 'auc_peak_h1_h2'])

cols_svm = ['thAct', 'numSamples', 'softNorm', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs', 
       'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
       'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
       'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs',
       'testTrInds_allSamps_allFrs', 'Ytest_allSamps_allFrs', 'Ytest_hat_allSampsFrs_allFrs']
       


#%%

import sys
#sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
#import platform
#if platform.system() == 'Linux':
#    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob # flake8: noqa: E999



#%% 
python_file = r"/home/farzaneh.najafi/analysis_codes/multiscope_fn/svm_set_vars0.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ClusterJobs/SVMJobs'

job_settings = {'queue': 'braintv',
                'mem': '24g',
                'walltime': '48:00:00',
                'ppn': 4,
                'jobdir': jobdir,
                }



#%% Loop through valid sessions to perform the analysis

sess_no_omission = []
cnt_sess = -1     

for isess in range(len(list_all_sessions_valid)):  # isess = -1 # session_id = list_all_sessions_valid[0] #[num_valid_exps_each_sess == 8][0]
    PythonJob(
        python_file,
        python_executable='/home/farzaneh.najafi/anaconda3/envs/visbeh/bin/python',
        python_args=isess, #cache_dir, plot_roi_validation,
        conda_env=None,
        jobname='process_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)
  
	             
