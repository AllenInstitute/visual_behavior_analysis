#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the first script to be run for doing SVM analysis.

To perform SVM (do this on the cluster; here this part is commented out), it will call "svm_main" (for this, you will need to run svm_init_pbs on the cluster which calls svm_main_pbs).

To set vars for plotting svm results, it will call "svm_main_post". The output (all_sess), includes for each experiment the average and st error of class accuracies across CV samples.
all_sess will be saved to /allen/programs/braintv/workgroups/nc-ophys/Farzaneh/SVM/same_num_neurons_all_planes/all_sess_svm_gray_omit*

Follow this script by svm_plots_setVars.py to make plots.

Created on Wed Jul 31 13:24:19 2019
@author: farzaneh
"""

use_spont_omitFrMinus1 = 0 # if 0, classify omissions against randomly picked spontanoues frames (the initial gray screen); if 1, classify omissions against the frame right before the omission

use_ct_traces = 1 # if 0, we go with dff traces saved in analysis_dir (visual behavior production analysis); if 1, we go with crosstalk corrected dff traces on rd-storage
use_np_corr = 1 # will be used when use_ct_traces=1; if use_np_corr=1, we will load the manually neuropil corrected traces; if 0, we will load the soma traces.
use_common_vb_roi = 1 # only those ct dff ROIs that exist in vb rois will be used.

#%% Set SVM vars
same_num_neuron_all_planes = 1 #0 # if 1, use the same number of neurons for all planes to train svm

frames_svm = range(-16, 24) # range(-10, 30) # range(-1,1) # frames_after_omission = 30 # 5 # run svm on how many frames after omission
numSamples = 50 # 2 #10 #
saveResults = 1 # 0
#frames_after_omission = 30

#%% Set vars for SVM_main_post

mean_notPeak = 1  # set to 1, so mean is computed; it is preferred because SST and SLC responses don't increase after omission
# if 1, to quantify omission-evoked response, compute average trace during peak_win, instead of the peak amplitude.
# if 1, we will still use the timing of the "peak" for peak timing measures ...
peak_win = [0, .5] #[0, .75] # you should totally not go beyond 0.75!!! # go with 0.75ms if you don't want the responses to be contaminated by the next flash
# [0,2] # flashes are 250ms and gray screen is 500ms # sec # time window to find peak activity, seconds relative to omission onset
# [-.75, 0] is too large for flash responses... (unlike omit responses, they peak fast!
# [-.75, -.4] #[-.75, 0] # window (relative to omission) for computing flash-evoked responses (750ms includes flash and gray)
flash_win = [0, .35] #[-.75, 0] # previous value: # [-.75, -.25]
flash_win_vip = [-.25, .1] #[-1, -.25] # previous value (for flash_win of all cre lines): # [-.75, -.25] # 
flash_win_timing = [-.85, -.5]

bl_percentile = 10 #20  # for peak measurements, we subtract pre-omit baseline from the peak. bl_percentile determines what pre-omit values will be used.      
doShift_again = 0

analysis_dates = [''] # ['2020507'] #set to [''] if you want to load the latest file. # the date on which the svm files were saved.
doCorrs = 0 # we set it to 0 so we can set distinct_areas later # -1 #0 #1 # if -1, only get the traces, dont compute peaks, mean, etc. # if 0, compute omit-aligned trace median, peaks, etc. If 1, compute corr coeff between neuron pairs in each layer of v1 and lm


#%%
import pickle
import numpy as np

from def_paths import * 

dir_svm = os.path.join(dir_server_me, 'SVM')
if same_num_neuron_all_planes:
    dir_svm = os.path.join(dir_svm, 'same_num_neurons_all_planes')
if not os.path.exists(dir_svm):
    os.makedirs(dir_svm)

        
from svm_main import *
from svm_main_post import *

            
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

print(list_all_sessions_valid.shape)

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

    # remove these sessions; because: 
    # ct dff files dont exist (843871999)
    # one depth is nan, so dataset cannot be set: 958772311
    # dataset.cell_specimen_table couldnt be set (error in load_session_data_new): 986767503
    
    # this is resolved by using loading dataset: there is no common ROI between VB and CT traces: 843871999, 882674040, 884451806, 914728054, 944888114, 952430817, 971922380, 974486549, 976167513, 976382032, 977760370, 978201478, 981705001, 982566889, 986130604, 988768058, 988903485, 989267296, 990139534, 990464099, 991958444, 993253587, 
    
    sessions_ctDone = np.array([839514418, 840490733, 841303580, 841682738, 841778484, 842023261, 842364341, 842623907, 844469521, 845235947, 846871218, 847758278, 848401585, 849304162, 850667270, 850894918, 852794141, 853416532, 854060305, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 871906231, 872592724, 873247524, 874616920, 875259383, 876303107, 880498009, 880709154, 882674040, 884451806, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 904771513, 906299056, 906521029, 906968227, 907177554, 907753304, 907991198, 908441202, 911719666, 913564409, 914161594, 914639324, 914728054, 915306390, 916650386, 917498735, 918889065, 919041767, 919888953, 920695792, 921636320, 921922878, 922564930, 923705570, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931687751, 933439847, 933604359, 935559843, 937162622, 937682841, 938140092, 938898514, 939526443, 940145217, 940775208, 941676716, 946015345, 947199653, 947358663, 948042811, 948252173, 949217880, 950031363, 951410079, 952430817, 954954402, 955775716, 957020350, 958105827, 959458018, 971632311, 971922380, 973384292, 973701907, 974167263, 974486549, 975452945, 976167513, 976382032, 977760370, 978201478, 980062339, 981705001, 981845703, 982566889, 985609503, 985888070, 986130604, 987352048, 988768058, 988903485, 989267296, 990139534, 990464099, 991639544, 991958444, 992393325, 993253587, 993420347, 993738515, 993962221, 1000439105, 1002120640, 1005374186])

#     sessions_ctDone = np.array([839514418, 841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 872592724, 873720614, 874616920, 875259383, 876303107, 880498009, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931326814, 931687751, 940145217, 940775208, 941676716, 944888114, 946015345, 947199653, 948252173, 950031363, 952430817, 954954402, 955775716, 958105827, 971922380])
#         [839208243, 839514418, 840490733, 841303580, 841682738, 841778484,
#                                842023261, 842364341, 842623907, 843059122, 843871999, 844469521,
#                                845444695, 846652517, 846871218, 847758278, 848401585, 848983781,
#                                849304162, 850667270, 850894918, 851428829, 851740017, 852070825,
#                                852794141, 853177377, 853416532, 854060305, 856201876, 857040020,
#                                863815473, 864458864, 865024413, 865854762, 866197765, 867027875,
#                                868688430, 869117575, 870352564, 870762788, 871526950, 871906231,
#                                872592724, 873720614, 874070091, 874616920, 875259383, 876303107,
#                                880498009, 882674040, 884451806, 885303356, 886806800, 888009781, 
#                                889944877, 906299056, 906521029]) # 

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

    print(list_all_sessions_valid_ct.shape)
#     print(list_all_experiments_valid_ct.shape)
#     print(list_all_experiments_ct.shape)

    # redefine these vars
    list_all_sessions_valid = list_all_sessions_valid_ct
    list_all_experiments_valid = list_all_experiments_valid_ct
    list_all_experiments = list_all_experiments_ct
    
    
#%% Loop through valid sessions to perform the analysis

#sess_no_omission = []
cnt_sess = -1     

cols0 = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'frame_dur', 'flash_omit_dur_all', 'flash_omit_dur_fr_all', 'meanX_allFrs_new', 'stdX_allFrs_new', 'av_train_data_new', 'av_test_data_new', 'av_test_shfl_new', 'av_test_chance_new', 'sd_train_data_new', 'sd_test_data_new', 'sd_test_shfl_new', 'sd_test_chance_new', 'peak_amp_omit_trainTestShflChance', 'peak_timing_omit_trainTestShflChance', 'peak_amp_flash_trainTestShflChance', 'peak_timing_flash_trainTestShflChance'])
if same_num_neuron_all_planes:
    cols = np.concatenate((cols0, ['population_sizes_to_try'])) 
else:
    cols = np.concatenate((cols0, ['av_w_data_new', 'av_b_data_new']))
#     cols = cols0
all_sess = pd.DataFrame([], columns=cols)

for isess in range(len(list_all_sessions_valid)):   # session_id = list_all_sessions_valid[0] #[num_valid_exps_each_sess == 8][0]
    # isess = 15 # 
    # isess = np.argwhere(list_all_sessions_valid==851428829).squeeze()
    session_id = int(list_all_sessions_valid[isess])
#    experiment_ids = list_all_experiments_valid[np.squeeze(np.argwhere(np.in1d(list_all_sessions_valid, session_id)))]
    
    cnt_sess = cnt_sess + 1    
    sess_date = validity_log_all[validity_log_all['session_id'].values ==
                                 session_id]['date'].iloc[0]
    print('\n\n======================== %d: session %d out of %d sessions, date: %s ========================\n' %
          (session_id, cnt_sess+1, len(list_all_sessions_valid), sess_date))    
#     print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess+1, len(list_all_sessions_valid)))
#    print('%d: session %d out of %d sessions' %(session_id, cnt_sess+1, len(list_all_sessions_valid)))


    #%% Run the SVM analysis (do this on the cluster)

#     experiment_ids = list_all_experiments_valid[isess] # uncomment this    
#    svm_main(session_id, experiment_ids, validity_log_all, dir_svm, frames_svm, numSamples, saveResults, use_ct_traces, same_num_neuron_all_planes)
#    svm_vars = pd.read_hdf(svmName, key='svm_vars')
    
    
    #%% Call function to set vars for plotting svm results
    
    # For the new run of SVM, you have to modify the following for frames_after_omission
    
    experiment_ids = list_all_experiments[isess] # we want to have the list of all experiments for each session regardless of whethere they were valid or not... this way we can find the same plane across all sessions.
    
    this_sess = svm_main_post(session_id, experiment_ids, validity_log_all, dir_svm, frames_svm, all_sess, same_num_neuron_all_planes, use_ct_traces, use_np_corr, use_common_vb_roi, mean_notPeak, peak_win, flash_win, flash_win_vip, flash_win_timing, bl_percentile, cols, doShift_again, analysis_dates, use_spont_omitFrMinus1, doPlots=0)
    
    all_sess = all_sess.append(this_sess) 
    

    
##################################################################    
##################################################################
#%% Save all_sess

samps_bef = -frames_svm[0] 
samps_aft = frames_svm[-1]+1 

# set this so you know what input vars you ran the script with
cols = np.array(['samps_bef', 'samps_aft', 'same_num_neuron_all_planes', 'use_ct_traces', 'mean_notPeak', 'peak_win', 'flash_win', 'flash_win_timing', 'bl_percentile', 'doShift_again'])
input_vars = pd.DataFrame([], columns=cols)
input_vars.at[0, cols] =  samps_bef, samps_aft, same_num_neuron_all_planes, use_ct_traces, mean_notPeak, peak_win, flash_win, flash_win_timing, bl_percentile, doShift_again

svmn = 'svm_gray_omit'
if use_spont_omitFrMinus1==0:
    svmn = svmn + '_spontFrs'    

now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

if same_num_neuron_all_planes:
    name = 'all_sess_%s_sameNumNeuronsAllPlanes_%s' %(svmn, now) 
else:
    name = 'all_sess_%s_%s' %(svmn, now) 
    
    
if saveResults:
    print('\n\nSaving all_sess.h5 file')
    allSessName = os.path.join(dir_svm, name + '.h5') # os.path.join(d, svmn+os.path.basename(pnevFileName))
    print(allSessName)

    # save to a h5 file                    
    all_sess.to_hdf(allSessName, key='all_sess', mode='w')    
    
    # save input_vars to the h5 file
    input_vars.to_hdf(allSessName, key='input_vars', mode='a')
    
                  