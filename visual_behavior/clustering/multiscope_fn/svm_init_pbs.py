#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the first script to be run for doing SVM analysis.

To perform SVM (do this on the cluster), it will call "svm_main_pbs.py".

To do a quick test of the code on the cluster: 
change the mem and isess range in this script. Also numSamples and frames_svm in svm_main_pbs.py


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


#%%
use_ct_traces = 1 # if 0, we go with dff traces saved in analysis_dir (visual behavior production analysis); if 1, we go with crosstalk corrected dff traces on rd-storage
same_num_neuron_all_planes = 1 # 0 # if 1, use the same number of neurons for all planes to train svm

# vars below will be set at the end of svm_main_pbs.py
'''
use_np_corr = 1 # will be used when use_ct_traces=1; if use_np_corr=1, we will load the manually neuropil corrected traces; if 0, we will load the soma traces.
use_common_vb_roi = 1 # only those ct dff ROIs that exist in vb rois will be used.

# Set SVM vars
frames_svm = range(-10, 30) # range(-1,1) # frames_after_omission = 30 # 5 # run svm on how many frames after omission
numSamples = 50 # 2 #10 #
saveResults = 1 # 0 #

# frames_after_omission = 30
'''

#%%
import socket
import os
import pickle
import numpy as np

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

if socket.gethostname() == 'OSXLT1JHD5.local': # allen mac
    dirAna = "/Users/farzaneh.najafi/Documents/analysis_codes/"
#    dir0 = '/Users/farzaneh.najafi/OneDrive - Allen Institute/Analysis'

elif socket.gethostname() == 'ibs-farzaneh-ux2': # allen pc
    dirAna = "/home/farzaneh/Documents/analysis_codes/"
#    dir0 = '/home/farzaneh/OneDrive/Analysis'

elif socket.gethostname() == 'hpc-login.corp.alleninstitute.org': # hpc server
    dirAna = "/home/farzaneh.najafi/analysis_codes/"        
    
# dirMs = os.path.join(dirAna, 'multiscope_fn')    
# os.chdir(dirMs)


# make svm dir to save analysis results
dir_svm = os.path.join(dir_server_me, 'SVM')
if same_num_neuron_all_planes:
    dir_svm = os.path.join(dir_svm, 'same_num_neurons_all_planes')
if not os.path.exists(dir_svm):
    os.makedirs(dir_svm)
        
#from svm_main import *
#from svm_plots_init import *

            
#%% Load vars related to list_all_sessions_valid 
# (read the pickle file saved in the script: set_valid_sessions.py)

dir_valid_sess = os.path.join(dir_server_me, 'ValidSessions') 
#validSessName = os.path.join(dir_valid_sess, 'valid_sessions' + '.pkl')

import re    
regex = re.compile('valid_sessions_(.*)' + '.pkl')
l = os.listdir(dir_valid_sess) 
files = [string for string in l if re.match(regex, string)]

### NOTE: # get the latest file (make sure this is correct)
files = files[-1]
validSessName = os.path.join(dir_valid_sess, files)

print(validSessName)
    
    
pkl = open(validSessName, 'rb')
dictNow = pickle.load(pkl)

for k in list(dictNow.keys()):
    exec(k + '= dictNow[k]')

pkl.close()

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
    # Get a list of sessions ready for post-cross talk analysis with the following code:
    import visual_behavior.ophys.mesoscope.utils as mu
    import logging
    lims_done, lims_notdone, meso_data = mu.get_lims_done_sessions()
    lims_done = lims_done.session_id.drop_duplicates()
    lims_done.values
    '''

    # remove these sessions; because: 
    # ct dff files dont exist (843871999)
    # one depth is nan, so dataset cannot be set: 958772311
    # dataset.cell_specimen_table couldnt be set (error in load_session_data_new): 986767503
    
    # this is resolved by using loading dataset: there is no common ROI between VB and CT traces: 843871999, 882674040, 884451806, 914728054, 944888114, 952430817, 971922380, 974486549, 976167513, 976382032, 977760370, 978201478, 981705001, 982566889, 986130604, 988768058, 988903485, 989267296, 990139534, 990464099, 991958444, 993253587, 
    
    sessions_ctDone = np.array([839514418, 840490733, 841303580, 841682738, 841778484, 842023261, 842364341, 842623907, 844469521, 845235947, 846871218, 847758278, 848401585, 849304162, 850667270, 850894918, 852794141, 853416532, 854060305, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 871906231, 872592724, 873247524, 874616920, 875259383, 876303107, 880498009, 880709154, 882674040, 884451806, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 904771513, 906299056, 906521029, 906968227, 907177554, 907753304, 907991198, 908441202, 911719666, 913564409, 914161594, 914639324, 914728054, 915306390, 916650386, 917498735, 918889065, 919041767, 919888953, 920695792, 921636320, 921922878, 922564930, 923705570, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931687751, 933439847, 933604359, 935559843, 937162622, 937682841, 938140092, 938898514, 939526443, 940145217, 940775208, 941676716, 946015345, 947199653, 947358663, 948042811, 948252173, 949217880, 950031363, 951410079, 952430817, 954954402, 955775716, 957020350, 958105827, 959458018, 971632311, 971922380, 973384292, 973701907, 974167263, 974486549, 975452945, 976167513, 976382032, 977760370, 978201478, 980062339, 981705001, 981845703, 982566889, 985609503, 985888070, 986130604, 987352048, 988768058, 988903485, 989267296, 990139534, 990464099, 991639544, 991958444, 992393325, 993253587, 993420347, 993738515, 993962221, 1000439105, 1002120640, 1005374186])

#     sessions_ctDone = np.array([839514418, 840490733, 841303580, 841682738, 841778484, 842023261, 842364341, 842623907, 844469521, 845235947, 846871218, 847758278, 848401585, 849304162, 850667270, 850894918, 852794141, 853416532, 854060305, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 871906231, 872592724, 873247524, 874616920, 875259383, 876303107, 880498009, 880709154, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 904771513, 906299056, 906521029, 906968227, 907177554, 907753304, 907991198, 908441202, 911719666, 913564409, 914161594, 914639324, 915306390, 916650386, 917498735, 918889065, 919041767, 919888953, 920695792, 921636320, 921922878, 922564930, 923705570, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931687751, 933439847, 933604359, 935559843, 937162622, 937682841, 938140092, 938898514, 939526443, 940145217, 940775208, 941676716, 944888114, 946015345, 947199653, 947358663, 948042811, 948252173, 949217880, 950031363, 951410079, 954954402, 955775716, 957020350, 958105827, 959458018, 971632311, 991639544, 
#                                 973384292, 973701907, 974167263, 975452945, 980062339, 981845703, 985609503, 985888070, 986767503, 987352048, 
#                                 992393325, 993420347, 993738515, 993962221, 1000439105, 1002120640, 1005374186])        

#     old ct:        
#     [839514418,  841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 868688430, 869117575]    
#     [839208243, 839514418, 840490733, 841303580, 841682738, 841778484, 842023261, 842364341, 842623907, 843059122, 843871999, 844469521, 845444695, 846652517, 846871218, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 851740017, 852070825, 852794141, 853177377, 853416532, 854060305, 856201876, 857040020, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 871906231, 872592724, 873720614, 874070091, 874616920, 875259383, 876303107, 880498009, 882674040, 884451806, 885303356, 886806800, 888009781, 889944877, 906299056, 906521029]


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
#     print(list_all_experiments_valid_ct.shape)
#     print(list_all_experiments_ct.shape)

    # redefine these vars
    list_all_sessions_valid = list_all_sessions_valid_ct
    list_all_experiments_valid = list_all_experiments_valid_ct
    list_all_experiments = list_all_experiments_ct
    

  
    
    
    
#%% Set vars for the cluster

import sys
from pbstools import PythonJob # flake8: noqa: E999
#sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
#import platform
#if platform.system() == 'Linux':
#    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')


python_file = r"/home/farzaneh.najafi/analysis_codes/visual_behavior_analysis/visual_behavior/clustering/multiscope_fn/svm_main_pbs.py" #r"/home/farzaneh.najafi/analysis_codes/multiscope_fn/svm_main_pbs.py"


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

#for experiment_id in experiment_ids:
#    PythonJob(
#        python_file,
#        python_executable='/home/farzaneh.najafi/anaconda3/envs/visbeh/bin/python',
#        python_args=experiment_id, #cache_dir, plot_roi_validation,
#        conda_env=None,
#        jobname='process_{}'.format(experiment_id),
#        **job_settings
#    ).run(dryrun=False)
    





    
#%% Loop through valid sessions to perform the analysis

#sess_no_omission = []
cnt_sess = -1     

#cols = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'meanX_allFrs', 'stdX_allFrs', 'av_train_data_new', 'av_test_data_new', 'av_test_shfl_new', 'av_test_chance_new', 'sd_train_data_new', 'sd_test_data_new', 'sd_test_shfl_new', 'sd_test_chance_new'])
#all_sess = pd.DataFrame([], columns=cols)

for isess in range(len(list_all_sessions_valid)): # [0,1]: # session_id = list_all_sessions_valid[0] #[num_valid_exps_each_sess == 8][0]
    
    # none of the below is really need (unless we want to print the part below when calling the function on the cluster); we can just directly call PythonJob below.
    
    # isess = -5
    session_id = int(list_all_sessions_valid[isess])
    experiment_ids = list_all_experiments_valid[isess]
#    experiment_ids = list_all_experiments_valid[np.squeeze(np.argwhere(np.in1d(list_all_sessions_valid, session_id)))]
    
    cnt_sess = cnt_sess + 1    
    sess_date = validity_log_all[validity_log_all['session_id'].values ==
                                 session_id]['date'].iloc[0]
    print('\n\n======================== %d: session %d out of %d sessions, date: %s ========================\n' %
          (session_id, cnt_sess+1, len(list_all_sessions_valid), sess_date))    

#    print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess+1, len(list_all_sessions_valid)))
#    print('%d: session %d out of %d sessions' %(session_id, cnt_sess+1, len(list_all_sessions_valid)))


    #%% Run the SVM analysis (do this on the cluster)
    
#    svm_main(session_id, experiment_ids, validity_log_all, dir_svm, frames_svm, numSamples, saveResults, same_num_neuron_all_planes)
#    svm_vars = pd.read_hdf(svmName, key='svm_vars')
        
    
    
    
    
    #%%    
    jobname = 'SVM'
    job_settings['jobname'] = '%s:%s' %(jobname, session_id)
    
    
    PythonJob(
        python_file,
        python_executable = '/home/farzaneh.najafi/anaconda3/envs/visbeh/bin/python',
        python_args = isess, # session_id experiment_ids validity_log_all dir_svm frames_svm numSamples saveResults use_ct_traces same_num_neuron_all_planes',
        conda_env = None,
#        jobname = 'process_{}'.format(session_id),
        **job_settings
    ).run(dryrun=False)
   
    
    

              
    
    
