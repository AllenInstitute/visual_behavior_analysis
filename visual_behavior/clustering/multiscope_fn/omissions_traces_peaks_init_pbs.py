#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:24:19 2019

@author: farzaneh
"""

use_ct_traces = 1 # if 0, we go with dff traces saved in analysis_dir (visual behavior production analysis); if 1, we go with crosstalk corrected dff traces on rd-storage


#%% Change directory to analysis directory

import re
import socket
import os
import pickle
import numpy as np

if socket.gethostname() == 'OSXLT1JHD5.local': # allen mac
    dirAna = "/Users/farzaneh.najafi/Documents/analysis_codes/"
    dir0 = '/Users/farzaneh.najafi/OneDrive - Allen Institute/Analysis'

elif socket.gethostname() == 'ibs-farzaneh-ux2': # allen pc
    dirAna = "/home/farzaneh/Documents/analysis_codes/"
    dir0 = '/home/farzaneh/OneDrive/Analysis'

elif socket.gethostname() == 'hpc-login.corp.alleninstitute.org': # hpc server
    dirAna = "/home/farzaneh.najafi/analysis_codes/"        


dirMs = os.path.join(dirAna, 'multiscope_fn')

os.chdir(dirMs)


#%% Load vars related to list_all_sessions_valid 
# (read the pickle file saved in the script: set_valid_sessions.py)

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

# validSessName = os.path.join(dir_valid_sess, 'valid_sessions' + '.pkl')
dir_valid_sess = os.path.join(dir_server_me, 'ValidSessions')

regex = re.compile('valid_sessions_(.*)' + '.pkl')
l = os.listdir(dir_valid_sess)
files = [string for string in l if re.match(regex, string)]

# NOTE: # get the latest file (make sure this is correct)
files = files[-1]
validSessName = os.path.join(dir_valid_sess, files)

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
# 'list_all_experiments_valid',
# 'list_all_experiments']  
    


#%% crosstalk-corrected sessions
if use_ct_traces:
    '''
    # Later when we call load_session_data_new, it will load the ct traces if use_ct_traces is 1; otherwise it will seet dataset which loads the original dff traces using vb codes.

    # Get a list of sessions ready for post-cross talk analysis with the following code:
    import visual_behavior.ophys.mesoscope.utils as mu
    import logging
    lims_done, lims_notdone, meso_data = mu.get_lims_done_sessions()
    lims_done = lims_done.session_id.drop_duplicates()
    lims_done.values
    '''

    sessions_ctDone = np.array([839514418,  841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 868688430, 869117575])
    
    # old ct:    
#     np.array([839208243, 839514418, 840490733, 841303580, 841682738, 841778484,
#                                842023261, 842364341, 842623907, 843059122, 843871999, 844469521,
#                                845444695, 846652517, 846871218, 847758278, 848401585, 848983781,
#                                849304162, 850667270, 850894918, 851428829, 851740017, 852070825,
#                                852794141, 853177377, 853416532, 854060305, 856201876, 857040020,
#                                863815473, 864458864, 865024413, 865854762, 866197765, 867027875,
#                                868688430, 869117575, 870352564, 870762788, 871526950, 871906231,
#                                872592724, 873720614, 874070091, 874616920, 875259383, 876303107,
#                                880498009, 882674040, 884451806, 885303356, 886806800, 888009781,
#                                889944877, 906299056, 906521029])

    print(sessions_ctDone.shape)

    # Copy ct folders to allen server (nc-ophys/Farzaneh):
    # ssh to the analysis computer by running: ssh -L 8888:ibs-is-analysis-ux1.corp.alleninstitute.org:8889 farzaneh.najafi@ibs-is-analysis-ux1.corp.alleninstitute.org
    # then on the analysis computer, in your folder (Farzaneh), run the notebook below
    # copy_crosstalkFolders_analysisPC_to_allen.py

    # or, on the analysis computer, in the terminal do:
    # cp -rv /media/rd-storage/Z/MesoscopeAnalysis/session_839208243 /allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk;

    # now take those valid sessions that are ct-corrected
    list_all_sessions_valid_ct = list_all_sessions_valid[np.in1d(
        list_all_sessions_valid, sessions_ctDone)]
    list_all_experiments_valid_ct = np.array(list_all_experiments_valid)[
        np.in1d(list_all_sessions_valid, sessions_ctDone)]
    list_all_experiments_ct = np.array(list_all_experiments)[
        np.in1d(list_all_sessions_valid, sessions_ctDone)]

    print(f'{len(list_all_sessions_valid_ct)}: Number of de-crosstalked sessions for analysis')
    list_all_experiments_valid_ct.shape
    list_all_experiments_ct.shape

    # redefine these vars
    list_all_sessions_valid = list_all_sessions_valid_ct
    list_all_experiments_valid = list_all_experiments_valid_ct
    list_all_experiments = list_all_experiments_ct




#%%
########################################################################################################
################################# Set vars for the cluster #############################################
########################################################################################################    
    
#%% Set vars for the cluster

std_dir = 'omitCorrs' #'SVMJobs' # folder name to save job std out/err files. # jobs will be saved here: '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ClusterJobs'
python_file = r"/home/farzaneh.najafi/analysis_codes/multiscope_fn/omissions_traces_peaks_pbs.py" # function to call below


#%%
import sys
from pbstools import PythonJob # flake8: noqa: E999
#sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
#import platform
#if platform.system() == 'Linux':
#    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')


#%% The params below are for each one of the sessions that we run below

jobdir = os.path.join('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ClusterJobs', std_dir)
job_settings = {'queue': 'braintv',
                'mem': '80g',
                'walltime': '10:00:00',
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
#cnt_sess = -1     

for isess in range(len(list_all_sessions_valid)):  # [64,66,70,74, 8,72]: #[8, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]: # isess = -1 # session_id = list_all_sessions_valid[0] #[num_valid_exps_each_sess == 8][0]
    
    jobname = 'session'
    job_settings['jobname'] = '%s_%d' %(jobname, isess)
    
  #  input_vars = session_id #[session_id, experiment_ids, validity_log_all, dir_svm, frames_after_omission, numSamples, saveResults]
    
    PythonJob(
        python_file,
        python_executable = '/home/farzaneh.najafi/anaconda3/envs/visbeh/bin/python',
        python_args = isess,
        conda_env = None,
#        jobname = 'process_{}'.format(session_id),
        **job_settings
    ).run(dryrun=False)
       
    

    
