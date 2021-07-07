#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USE this code for the correlation analysis.
USE omissions_traces_peaks_init.py for the population-average analysis.


Created on Wed Jul 31 13:24:19 2019

@author: farzaneh
"""

import pandas as pd
import numpy as np

import visual_behavior.data_access.loading as loading

# data release sessions
# experiments_table = loading.get_filtered_ophys_experiment_table()
experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
metadata_valid = experiments_table[experiments_table['project_code']=='VisualBehaviorMultiscope'] # multiscope sessions


# Use the new list of sessions that are de-crosstalked and will be released in March 2021
# metadata_meso_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/meso_decrosstalk/meso_experiments_in_release.csv'
# metadata_valid = pd.read_csv(metadata_meso_dir)

aa = metadata_valid['ophys_session_id'].unique()
# aa.shape

sessions_ctDone = aa

# aa = [951410079,  952430817,  954954402,  955775716,  957020350,
#         958105827,  958772311,  959458018,  985609503,  988768058,
#         989267296,  990139534,  990464099,  991639544, 1048122684,
#        1048363441, 1049240847, 1050231786, 1050597678, 1051107431,
#        1051319542,  937682841,  938898514,  940145217,  940775208,
#         945124131,  946015345,  948252173,  929686773,  931326814,
#         933439847,  935559843,  937162622,  938140092,  939526443,
#        1052096166, 1052330675, 1052512524, 1056065360, 1056238781,
#         880709154,  882386411,  883619540,  884613038,  885557130,
#         886367984,  887031077,  888171877,  846871218,  847758278,
#         848401585,  848983781,  850894918,  853177377,  880498009,
#         881094781,  882060185,  882674040,  884451806,  886130638,
#         886806800,  888009781,  889944877,  941676716,  944888114,
#         947199653,  948042811,  949217880,  950031363,  991958444,
#         992393325,  993253587,  993420347,  993738515,  993962221,
#         870352564,  871526950,  872592724,  873247524,  875259383,
#         876303107,  971632311,  971922380,  973384292,  973701907,
#         974167263,  974486549,  902884228,  903813946,  904771513,
#         906521029,  907177554,  907991198,  866197765,  867027875,
#         868688430,  870762788,  873720614,  874616920,  975452945,
#         976167513,  976382032,  977760370,  978201478,  980062339,
#         981845703,  983913570,  986130604,  916650386,  917498735,
#         919041767,  919888953,  920695792,  921922878,  923705570,
#         925478114,  927787876,  849304162,  850667270,  852794141,
#         853416532,  854060305,  855711263,  857040020,  903621170,
#         904418381,  906299056,  906968227,  907753304,  908441202,
#         911719666,  913564409,  914161594,  914639324,  915306390]

# common sessions between the list above and previously used (see the part use_ct_traces):
# cc = sessions_ctDone[np.in1d(sessions_ctDone, aa)]
# cc.shape
# [846871218, 847758278, 848401585, 849304162, 850667270, 850894918,
#        852794141, 853416532, 854060305, 855711263, 866197765, 867027875,
#        868688430, 870352564, 870762788, 871526950, 872592724, 873247524,
#        874616920, 875259383, 876303107, 880498009, 880709154, 882674040,
#        884451806, 886130638, 886806800, 888009781, 889944877, 902884228,
#        903621170, 903813946, 904418381, 904771513, 906299056, 906521029,
#        906968227, 907177554, 907753304, 907991198, 908441202, 911719666,
#        913564409, 914161594, 914639324, 915306390, 916650386, 917498735,
#        919041767, 919888953, 920695792, 921922878, 923705570, 925478114,
#        927787876, 929686773, 933439847, 935559843, 937162622, 937682841,
#        938140092, 938898514, 939526443, 940145217, 940775208, 941676716,
#        946015345, 947199653, 948042811, 948252173, 949217880, 950031363,
#        951410079, 952430817, 954954402, 955775716, 957020350, 958105827,
#        959458018, 971632311, 971922380, 973384292, 973701907, 974167263,
#        974486549, 975452945, 976167513, 976382032, 977760370, 978201478,
#        980062339, 981845703, 985609503, 986130604, 988768058, 989267296,
#        990139534, 990464099, 991639544, 991958444, 992393325, 993253587,
#        993420347, 993738515, 993962221]


# sessions_ctDone = cc




#%%
# print(sessions_ctDone.shape)

list_all_sessions_valid = sessions_ctDone
print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')

# get the list of 8 experiments for all sessions
experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')

metadata_all = experiments_table[experiments_table['ophys_session_id'].isin(list_all_sessions_valid)==True] # metadata_all = experiments_table[np.in1d(experiments_table['ophys_session_id'].values, list_all_sessions_valid)]
metadata_all.shape
metadata_all.shape[0]/8
# cache = loading.get_visual_behavior_cache()
# experiments = cache.get_experiment_table()    
# experiments_table = experiments


# set the list of experiments for each session in list_all_sessions_valid
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


list_all_experiments = np.sort(list_all_experiments)
print(list_all_experiments.shape)


    
#########################################################################################################
#%%
use_ct_traces = 1 # if 0, we go with dff traces saved in analysis_dir (visual behavior production analysis); if 1, we go with crosstalk corrected dff traces on rd-storage


#%% Change directory to analysis directory

import re
import socket
import os
import pickle
import numpy as np

if socket.gethostname() == 'OSXLT1JHD5.local': # allen mac
    dirAna = "/Users/farzaneh.najafi/Documents/analysis_codes/"
#     dir0 = '/Users/farzaneh.najafi/OneDrive - Allen Institute/Analysis'

elif socket.gethostname() == 'ibs-farzaneh-ux2': # allen pc
    dirAna = "/home/farzaneh/Documents/analysis_codes/"
#     dir0 = '/home/farzaneh/OneDrive/Analysis'

elif socket.gethostname() == 'hpc-login.corp.alleninstitute.org': # hpc server
    dirAna = "/home/farzaneh.najafi/analysis_codes/"        


# dirMs = os.path.join(dirAna, 'multiscope_fn')
# os.chdir(dirMs)


# below is not needed anymore since we are using data release sessions

#%% Load vars related to list_all_sessions_valid 
# (read the pickle file saved in the script: set_valid_sessions.py)

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

'''
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

pkl.close()

#['list_all_sessions0',
# 'list_sessions_date0',
# 'list_sessions_experiments0',
# 'validity_log_all',
# 'list_all_sessions_valid',
# 'list_all_experiments_valid',
# 'list_all_experiments']  
'''


#%% crosstalk-corrected sessions
if 0: #use_ct_traces: # lims now contains the decrosstalked traces (2/11/2021)
    '''
    # Later when we call load_session_data_new, it will load the ct traces if use_ct_traces is 1; otherwise it will seet dataset which loads the original dff traces using vb codes.

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

    # used for nature methods and nature communications submissions    
#     sessions_ctDone = np.array([839514418, 840490733, 841303580, 841682738, 841778484, 842023261, 842364341, 842623907, 844469521, 845235947, 846871218, 847758278, 848401585, 849304162, 850667270, 850894918, 852794141, 853416532, 854060305, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 871906231, 872592724, 873247524, 874616920, 875259383, 876303107, 880498009, 880709154, 882674040, 884451806, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 904771513, 906299056, 906521029, 906968227, 907177554, 907753304, 907991198, 908441202, 911719666, 913564409, 914161594, 914639324, 914728054, 915306390, 916650386, 917498735, 918889065, 919041767, 919888953, 920695792, 921636320, 921922878, 922564930, 923705570, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931687751, 933439847, 933604359, 935559843, 937162622, 937682841, 938140092, 938898514, 939526443, 940145217, 940775208, 941676716, 946015345, 947199653, 947358663, 948042811, 948252173, 949217880, 950031363, 951410079, 952430817, 954954402, 955775716, 957020350, 958105827, 959458018, 971632311, 971922380, 973384292, 973701907, 974167263, 974486549, 975452945, 976167513, 976382032, 977760370, 978201478, 980062339, 981705001, 981845703, 982566889, 985609503, 985888070, 986130604, 987352048, 988768058, 988903485, 989267296, 990139534, 990464099, 991639544, 991958444, 992393325, 993253587, 993420347, 993738515, 993962221, 1000439105, 1002120640, 1005374186])
    
#     sessions_ctDone = np.array([839514418,  841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 868688430, 869117575])
    
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

    
    list_all_sessions_valid = sessions_ctDone
    
    i = np.in1d(experiments_table['ophys_session_id'].values, list_all_sessions_valid)
    metadata_all = experiments_table[i]
    list_all_experiments = np.reshape(experiments_table[i]['ophys_experiment_id'].values, (8, len(sessions_ctDone)), order='F').T    
    list_all_experiments = np.sort(list_all_experiments)
    
    print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')
    
    
    
    # Copy ct folders to allen server (nc-ophys/Farzaneh):
    # ssh to the analysis computer by running: ssh -L 8888:ibs-is-analysis-ux1.corp.alleninstitute.org:8889 farzaneh.najafi@ibs-is-analysis-ux1.corp.alleninstitute.org
    # then on the analysis computer, in your folder (Farzaneh), run the notebook below
    # copy_crosstalkFolders_analysisPC_to_allen.py

    # or, on the analysis computer, in the terminal do:
    # cp -rv /media/rd-storage/Z/MesoscopeAnalysis/session_839208243 /allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk;

    # now take those valid sessions that are ct-corrected
    list_all_sessions_valid_ct = list_all_sessions_valid[np.in1d(list_all_sessions_valid, sessions_ctDone)]
    list_all_experiments_valid_ct = np.array(list_all_experiments_valid)[np.in1d(list_all_sessions_valid, sessions_ctDone)]
    list_all_experiments_ct = np.array(list_all_experiments)[np.in1d(list_all_sessions_valid, sessions_ctDone)]

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
python_file = r"/home/farzaneh.najafi/analysis_codes/visual_behavior_analysis/visual_behavior/clustering/multiscope_fn/omissions_traces_peaks_pbs.py" # function to call below


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
       
    

    
