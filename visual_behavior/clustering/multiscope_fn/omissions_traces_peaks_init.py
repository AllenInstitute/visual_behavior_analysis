#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use this code for the population-average analysis, to save all_sess, and follow it by "omissions_traces_peaks_plots_setVars.py" to make plots.
Use omissions_traces_peaks_init_pbs.py to run the analysis for the correlation analysis. After the files are saved, run "omissions_traces_peaks_plots_setVars.py" to make plots.



# This is the initial script to be called for most analyses. It creates df/f traces 
# for the session, also aligned on omission.

Before running this script we need to run "set_valid_sessions.py" to get the list of valid session.

This funciton calls omissions_traces_peaks.py: it creates a pandas table (all_sess) that
includes vars related to median traces across trials, aligned on omission, and their peak amplitude.
It saves the results in an h5 file named "all_sess_omit_traces_peaks"

# Note: if you are going to follow this script by "omissions_traces_peaks_plots_setVars.py", you need to
# save the updated all_sess, alse the updated "mouse_trainHist_all2". 
# You do this by: 1) updating the list of all_mice_id in set_mouse_trainHist_all2.py, and 
# 2) running set_mouse_trainHist_init_pbs.py on the cluster (which calls set_mouse_trainHist_all2.py).

Created on Wed Jul 31 13:24:19 2019
@author: farzaneh
"""
# get_ipython().magic(u'matplotlib inline')

##%% Set initial vars

import re
from omissions_traces_peaks import *
from def_funs_general import *
from def_funs import *
import pickle
import os
import socket
import datetime

#%%
controlSingleBeam_oddFrPlanes = [0, []] #[1, [0,2,5,7]] # if 1st element is 1, make control data to remove the simultaneous aspect of dual beam mesoscope (ie the coupled planes) to see if the correlation results require the high temporal resolution (11Hz) of dual beam vs. 5Hz of single beam mesoscope # 2nd element: odd_fr_planes = [0,2,5,7] # if the plane index is among these, we will take df/f from odd frame indices         

saveResults = 1  # save the all_sess pandas at the end

doCorrs = 0 #-1 #0 #1 # if -1, only get the traces, dont compute peaks, mean, etc. # if 0, compute omit-aligned trace median, peaks, etc. If 1, compute corr coeff between neuron pairs in each layer of v1 and lm
# note: when doCorrs=1, run this code on the cluster: omissions_traces_peaks_init_pbs.py (this will call omissions_traces_peaks_pbs.py) 
num_shfl_corr = 2 #50 # set to 0 if you dont want to compute corrs for shuffled data # shuffle trials, then compute corrcoeff... this serves as control to evaluate the values of corrcoeff of actual data    
subtractSigCorrs = 1 # if 1, compute average response to each image, and subtract it out from all trials of that image. Then compute correlations; ie remove signal correlations.

use_ct_traces = 1 # if 0, we go with dff traces saved in analysis_dir (visual behavior production analysis); if 1, we go with crosstalk corrected dff traces on rd-storage
use_np_corr = 1 # will be used when use_ct_traces=1; if use_np_corr=1, we will load the manually neuropil corrected traces; if 0, we will load the soma traces.

# To align on omissions, get 40 frames before omission and 39 frames after omission
samps_bef = 16 #40 # 40 is a lot especially for correlation analysis which takes a long time; the advantage of 40 thouth is that bl_preOmit and bl_preFlash will be more accurate as they use 10th percentile of frames before omission (or flash), so more frames means better accuracy.
samps_aft = 24 #40


# remember the following two were 1, for autoencoder analuysis you changed them to 0.
doScale = 1 #1 # 0  # Scale the traces by baseline std
doShift = 1 #1 # 0  # Shift the y values so baseline mean is at 0 at time 0
doShift_again = 0 # 0 # this is a second shift just to make sure the pre-omit activity has baseline at 0. (it is after we normalize the traces by baseline ave and sd... but because the traces are median of trials (or computed from the initial gray screen activity), and the baseline is mean of trials, the traces wont end up at baseline of 0)
bl_gray_screen = 1 # 1 # if 1, baseline will be computed on the initial gray screen at the beginning of the session
# if 0, get the lowest 20 percentile values of the median trace preceding omission and then compute their mean and sd
# this value will only affect the traces, but not the peak quantification, because for peak quantification we subtract the local baseline
# which is computed as 10th percentile of activity preceding each omission

# THE following is not a problem anymore! as peak timing is not being computed on SST/SLC!
# NOTE: not sure how much to trust peak measures (amplitude and timing) for non-VIP lines...
# they dont really have a peak, and you have the note below in the omissions_traces_peaks.py code:
# if the mouse is NOT a VIP line, instead of computing the peak, we compute the trough: we invert the trace, and then compute the max. (this is because in VIP mice the response after omission goes up, but not in SST or Slc)
mean_notPeak = 1  # set to 1, so mean is computed; it is preferred because SST and SLC responses don't increase after omission
# if 1, to quantify omission-evoked response, compute average trace during peak_win, instead of the peak amplitude.
# if 1, we will still use the timing of the "peak" for peak timing measures ...
# Note sure of the following: if mean_notPeak is 1, it makes sense to go with a conservative (short) peak_win so the results are not contaminated by the responses to the flash that follows the omission
# set mean_notPeak to 0 if you want to measure trough (on traces with decreased response after the omission)

# you should totally not go beyond 0.75!!! # go with 0.75ms if you don't want the responses to be contaminated by the next flash
peak_win = [0, .5] #[0, .75] # this should be named omit_win
# [0,2] # flashes are 250ms and gray screen is 500ms # sec # time window to find peak activity, seconds relative to omission onset
# [-.75, 0] is too large for flash responses... (unlike omit responses, they peak fast!
# [-.75, -.4] #[-.75, 0] # window (relative to omission) for computing flash-evoked responses (750ms includes flash and gray)
flash_win = [0, .35] #[0, .75] # now using flash-aligned traces; previously flash responses were computed on omission-aligned traces; problem: not all flashes happen every .75sec # [-.75, 0] # previous value: # [-.75, -.25] # 
flash_win_vip = [-.25, .1] #[-.25, .5] # on flash-aligned traces # [-1, -.25] # previous value (for flash_win of all cre lines): # [-.75, -.25] # 
flash_win_timing = [-.875, -.25] # this is not needed anymore bc we are using different values for vip vs others, so we just use flash_win and flash_win_vip to compute timing. # previous value: #[-.85, -.5] 
# we use flash_win_timing, so it can take care of vip responses too, but maybe start using different windows for non-vip and vip lines; those windows will be used for both timing and amplitude computation. This whole thing is because vip responses in familiar imagesets precedes the flash occurrence. So for the first novel session we need to use a similar window as in non-vip lines, but for all the other sessions we need to use flash_win_vip to find the amp and timing of vip responses. 
# one solution is to go with flash_win (for non-vip and vip, B1 session) and flash_win_vip (for vip sessions except B1); use those same windows for both amp and timing computation. 
# the other solution is what currently we are using which is a wider window for flash_timing so it takes care of vip responses; and a flash_win that is non-optimal for vip responses (bc it doesnt precede the image; although now that i extended it to 0 (previously it was [-.75, -.25]) it will include vip ramping activity too, so perhaps non-optimality is less of a problem).

bl_percentile = 10 #20  # for peak measurements, we subtract pre-omit baseline from the peak. bl_percentile determines what pre-omit values will be used.      

norm_to_max = 0  # normalize each neuron trace by its max
# In this case doScale is automatically set to zero, because:
# remember whether you are normalizing to max peak activity or not will not matter if you normalize by baseline SD. because if you normalize to max peak activity, baseline SD will also get changed by that same amount, so after normalizing to baseseline SD the effect of normalizing to max peak activity will be gone.

# if 1, use peak measures computed on the trial-median traces (less noisy); if 0, use peak measures computed on individual trials.
trace_median = 1


doPlots = 0  # some plots (not the main summary ones at the end)... set to 0
doROC = 0  # if 1, do ROC analysis: for each experiment, find, across all neurons, how separable the distributions of trial-averaged peak amplitudes are for half-1 omissions vs. half-2 omissions
#dosavefig = 1

use_common_vb_roi = 1 # only those ct dff ROIs that exist in vb rois will be used.



if norm_to_max:
    doScale = 0

if doROC:
    from sklearn.metrics import roc_curve, auc



#%% Load vars related to list_all_sessions_valid 
# (read the pickle file saved in the script: set_valid_sessions.py)

# below is not needed anymore since we are using data release sessions

import pickle

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

# validSessName = os.path.join(dir_valid_sess, 'valid_sessions' + '.pkl')
'''
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
'''

validity_log_all = [] # defining it here because we commented above. The right thing is to just remove it from the function input, but for now leaving it...
    
    
###############################################
# set sessions and get the metadata
###############################################

# Use the new list of sessions that are de-crosstalked and will be released in March 2021
import pandas as pd
import numpy as np
import visual_behavior.data_access.loading as loading

# metadata_meso_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/meso_decrosstalk/meso_experiments_in_release.csv'
# metadata_valid = pd.read_csv(metadata_meso_dir)
experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')
metadata_valid = experiments_table[experiments_table['project_code']=='VisualBehaviorMultiscope'] # multiscope sessions

sessions_ctDone = metadata_valid['ophys_session_id'].unique()
print(sessions_ctDone.shape)


#%% remove some problematic sessions

# sess2rmv = 873720614


list_all_sessions_valid = sessions_ctDone

print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')


# get the list of 8 experiments for all sessions
experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')

metadata_all = experiments_table[np.in1d(experiments_table['ophys_session_id'].values, list_all_sessions_valid)]


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




    
    
#%% crosstalk-corrected sessions

if 0: #use_ct_traces: # lims now contains decrosstalked traces (2/11/2021)
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
    
    
    

#%% Initiate all_sess panda table

cols_basic = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'frame_dur', 'flash_omit_dur_all', 'flash_omit_dur_fr_all'])

if doCorrs==1:
    colsa = np.array(['cc_cols_area12', 'cc_cols_areaSame',
                     'cc_a12', 'cc_a11', 'cc_a22', 'p_a12', 'p_a11', 'p_a22',
                     'cc_a12_shfl', 'cc_a11_shfl', 'cc_a22_shfl', 'p_a12_shfl', 'p_a11_shfl', 'p_a22_shfl'])
elif doCorrs==0:
    colsa = np.array(['traces_aveTrs_time_ns', 'traces_aveNs_time_trs', 'traces_aveTrs_time_ns_f', 'traces_aveNs_time_trs_f',
                     'peak_amp_eachN_traceMed', 'peak_timing_eachN_traceMed', 'peak_amp_trs_ns', 'peak_timing_trs_ns',
                     'peak_h1', 'peak_h2', 'auc_peak_h1_h2', 'peak_amp_eachN_traceMed_flash', 'peak_timing_eachN_traceMed_flash'])
elif doCorrs==-1:
    colsa = np.array(['valid', 'local_fluo_traces', 'local_time_traces', 'roi_ids', 'local_fluo_allOmitt', 'list_flashes', 'list_omitted', 'running_speed', 'licks', 'rewards'])
    
cols = np.concatenate((cols_basic, colsa))

cols_this_sess_l = np.array(['valid', 'local_fluo_traces', 'local_time_traces', 'roi_ids', 'local_fluo_allOmitt', 'list_flashes', 'list_omitted', 'running_speed', 'licks', 'rewards'])


#%% problematic sessions: (5/16/20) ... related to segmentation.
# 923469780 experiment 924211430
# 947199653 experiment 7: 947798785
# 958772311 experiment 0: 959388788
# 

all_sess = pd.DataFrame([], columns=cols) # size: 8*len(num_sessions)


#%% Loop through valid sessions to perform the analysis

#sess_no_omission = []

# VIP: isess = 20
# SST: isess = 50
# SLC: isess = 76 ; isess = 147
# [152]: #isess = 27 # session_id = list_all_sessions_valid[0] #[num_valid_exps_each_sess == 8][0]
# mouse_all=[]

cnt_sess = -1

for isess in np.arange(0, len(list_all_sessions_valid)): # isess=0 # 24

    session_id = list_all_sessions_valid[isess] #session_id = int(list_all_sessions_valid[isess]) # experiment_ids = list_all_experiments_valid[isess]
    experiment_ids = list_all_experiments[isess] # we want to have the list of all experiments for each session regardless of whethere they were valid or not... this way we can find the same plane across all sessions.

    # Note: we cant use metadata from allensdk because it doesnt include all experiments of a session, it only includes the valid experiments, so it wont allow us to set the metadata for all experiments.
    # session_id = sessions_ctDone[isess]
    metadata_now = metadata_valid[metadata_valid['ophys_session_id']==session_id]
    experiment_ids_valid = metadata_now['ophys_experiment_id'].values # the problem is that this wont give us the list of all experiments for a session ... it only includes the valid experiments.
    experiment_ids_valid = np.sort(experiment_ids_valid)
    print(f'\n{sum(np.in1d(experiment_ids, experiment_ids_valid)==False)} of the experiments are invalid!\n')


    #cnt_sess = cnt_sess + 1
    print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess, len(list_all_sessions_valid)))
    # print('%d: session %d out of %d sessions' %(session_id, cnt_sess+1, len(list_all_sessions_valid)))


    
    #%% Call the main function

    this_sess = omissions_traces_peaks(metadata_all, session_id, experiment_ids, experiment_ids_valid, validity_log_all, norm_to_max, mean_notPeak, peak_win, flash_win, flash_win_timing, flash_win_vip, bl_percentile, num_shfl_corr, trace_median,
       doScale, doShift, doShift_again, bl_gray_screen, samps_bef, samps_aft, doCorrs, subtractSigCorrs, saveResults, cols, cols_basic, colsa, cols_this_sess_l, use_ct_traces, use_np_corr, use_common_vb_roi, controlSingleBeam_oddFrPlanes, doPlots, doROC)

    
#     mouse_all.append(this_sess)
    all_sess = all_sess.append(this_sess)

print(len(all_sess))    
    
    
    
# %% Save the output of the analysis
#####################################################################################
#####################################################################################
#####################################################################################

# Set below to 1 when you need to save the results # if doCorrs, we save values for each session (because we run it on the cluster). Or if you dont wish to run on the cluster, we keep values of each session, append them, and them save "all_sess" var.
if 1:
    
    #%% Set pandas dataframe input_vars, to save it below
    # set this so you know what input vars you ran the script with
    cols = np.array(['list_all_sessions_valid', 'list_all_experiments', 'norm_to_max', 'mean_notPeak', 'peak_win', 'flash_win', 'flash_win_timing', 'flash_win_vip', 'bl_percentile', 'num_shfl_corr', 'trace_median',
                     'samps_bef', 'samps_aft', 'doScale', 'doShift', 'doShift_again', 'bl_gray_screen', 'subtractSigCorrs', 'controlSingleBeam_oddFrPlanes'])
    input_vars = pd.DataFrame([], columns=cols)

    input_vars.at[0, cols] = list_all_sessions_valid, list_all_experiments, norm_to_max, mean_notPeak, peak_win, flash_win, flash_win_timing, flash_win_vip, bl_percentile, num_shfl_corr, trace_median, samps_bef, samps_aft, doScale, doShift, doShift_again, bl_gray_screen, subtractSigCorrs, controlSingleBeam_oddFrPlanes
    # input_vars.iloc[0]

    
    #%% Save all_sess
    # directory: /allen/programs/braintv/workgroups/nc-ophys/Farzaneh/omit_traces_peaks

    analysis_name = 'omit_traces_peaks'
    dir_now = os.path.join(dir_server_me, analysis_name)
    # create a folder: omit_traces_peaks
    if not os.path.exists(dir_now):
        os.makedirs(dir_now)

    if doCorrs == 1:
        namespec = '_corr'
    elif doCorrs == -1:
        namespec = '_allTraces'
    else:
        namespec = ''

    if doShift_again == 1:
        namespec = namespec + '_blShiftAgain'
    else:
        namespec = namespec + ''

    
    if use_np_corr==1:
        namespec = namespec + '_np_corr_dff'
    else:
        namespec = namespec + '_soma_dff'

        
    now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

    
    if saveResults:

        print('Saving .h5 file')

        if doCorrs == 0:
            name = 'all_sess_%s%s_%s' % (analysis_name, namespec, now)
            allSessName = os.path.join(dir_now, name + '.h5')
            print(allSessName)
            
            # save all_sess to the h5 file
            all_sess.to_hdf(allSessName, key='all_sess', mode='w')
            # save input_vars to the h5 file
            input_vars.to_hdf(allSessName, key='input_vars', mode='a')

            
        if doCorrs == -1:
            name = 'all_sess_%s%s_%s' % (analysis_name, namespec, now)            
            # save to a pickle file
            allSessName = os.path.join(dir_now , name + '.pkl') 
            print(allSessName)

            f = open(allSessName, 'wb')
            pickle.dump(all_sess, f)    
            pickle.dump(input_vars, f)
            f.close()
            
            
        # all_sess is too big, so break down the columns, and save multiple h5 files.
        if doCorrs == 1:

            nn = '_info'
            name = 'all_sess_%s%s%s_%s' % (analysis_name, namespec, nn, now)
            allSessName = os.path.join(dir_now, name + '.pkl')
            print(allSessName)

            f = open(allSessName, 'wb')
            pickle.dump(input_vars, f)
            # pickle.dump(input_vars, f) # this didnt work. I guess we cant save multiple files to a pkl file
            f.close()

            '''
            nn = '_info'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'cc_cols_area12', 'cc_cols_areaSame']]
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file
#            print(a.shape)
            input_vars.to_hdf(allSessName, key='input_vars', mode='a') # save input_vars to the h5 file            

            nn = '_c12'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['cc_a12']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file

            nn = '_p12'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['p_a12']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file

            nn = '_c11'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['cc_a11']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file

            nn = '_p11'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['p_a11']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file

            nn = '_c22'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['cc_a22']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file

            nn = '_p22'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['p_a22']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file


            ##### shfl vars #####
            
            nn = '_c12_shfl'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['cc_a12_shfl']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file

            nn = '_p12_shfl'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['p_a12_shfl']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file

            nn = '_c11_shfl'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['cc_a11_shfl']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file

            nn = '_p11_shfl'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['p_a11_shfl']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file

            nn = '_c22_shfl'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['cc_a22_shfl']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file

            nn = '_p22_shfl'
            name = 'all_sess_%s%s%s_%s' %(analysis_name, namespec, nn, now) 
            allSessName = os.path.join(dir_now , name + '.h5') 
            print(allSessName)
            a = all_sess.loc[:,['p_a22_shfl']]
#            print(a.shape)
            a.to_hdf(allSessName, key='all_sess', mode='w') # save all_sess to the h5 file
            '''

#    df= all_sess
#    chunksize = 100
#    for i in range(len(df)):
#        chunk = df.iloc[(i*chunksize):min((i+1)*chunksize,len(df))]
#        try:
#           df.to_hdf('....','df',mode='w')
#        except (Exception) as e:
#           print(e)
#           print(df)
#           print(df.info())
#

    # %%
    #all_sess = pd.read_hdf('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/all_sess_omit_traces_peaks_20190826_130759.h5', key='all_sess')
    #input_vars = pd.read_hdf('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/all_sess_omit_traces_peaks_20190826_130759.h5', key='input_vars')
