#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The 1st script to run to make plots (assuming the analysis is done and files are saved.)

This script loads all_sess that is set by function svm_main_post (called in svm_init).
It sets all_sess_2an, which is a subset of all_sess that only includes desired sessions for analysis (A, or B, etc).

Its output is a pandas table "svm_this_plane_allsess" which includes svm variables for each mouse, (for each plane, for each area, and for each depth) and will be used for plotting.

Follow this script by "svm_plots_eachMouse" to make plots for each mouse.
Or, by "svm_plots_setVars_sumMice.py" to set vars for making average plots across mice (for each cre line).

Note, this script is similar to part of omissions_traces_peaks_plots_setVars_ave.py.
    The variable svm_this_plane_allsess here is basically a combination of the following 2 vars in omissions_traces_peaks_plots_setVars_ave.py:
        trace_peak_allMice   :   includes traces and peak measures for all planes of all sessions, for each mouse.
        pooled_trace_peak_allMice   : includes layer- and area-pooled vars, for each mouse.

This script also sets some of the session-averaged vars per mouse (in var "svm_this_plane_allsess") , which will be used in eachMouse plots; 
    # note: those vars are also set in svm_plots_setVars_sumMice.py (in var "svm_this_plane_allsess_sessAvSd". I didn't take them out of here, bc below will be used for making eachMouse plots. Codes in svm_plots_setVars_sumMice.py are compatible with omissions code, and will be used for making sumMice plots.


Created on Fri Aug  9 19:12:00 2019
@author: farzaneh
"""

#%%
use_spont_omitFrMinus1 = 0 # if 0, classify omissions against randomly picked spontanoues frames (the initial gray screen); if 1, classify omissions against the frame right before the omission

frames_svm = range(-16, 24) # 30 #frames_after_omission = 30 # 5 # run svm on how many frames after omission
same_num_neuron_all_planes = 1 # if 1, use the same number of neurons for all planes to train svm

all_ABtransit_AbefB_Aall = 3 # 1 # 0: analyze all sessions;  1: analyze AB transitions;  2: analyze only A sessions (before any B);  3: analyze all A sessions (before or after B)    
only_1st_transit = 1 # relevant only if all_ABtransit_AbefB_Aall=1 # if 1, only include data from the 1st A-->B transition even if a mouse has more than one (safer, since in the subsequent transitions, B has been already introduced, so they are not like the 1st A-->B transition)
dosavefig = 1

svmn = 'svm_gray_omit'
if use_spont_omitFrMinus1==0:
    svmn = svmn + '_spontFrs'    

doCorrs = 0


#%%
from def_paths import * 

dir_svm = os.path.join(dir_server_me, 'SVM')
if same_num_neuron_all_planes:
    dir_svm = os.path.join(dir_svm, 'same_num_neurons_all_planes')

from def_funs import *
from def_funs_general import *
from set_mousetrainhist_allsess2an_whatsess import *
# from set_timetrace_flashesind_allmice_areas import *


#%% Read all_see h5 file made in svm_main_post (by calling svm_init)
    
name = 'all_sess_%s_.' %(svmn) 
allSessName, h5_files = all_sess_set_h5_fileName(name, dir_svm)

## Load all_sess dataframe
all_sess = pd.read_hdf(allSessName, key='all_sess') #'svm_vars')


## Load input_vars
input_vars = pd.read_hdf(allSessName, key='input_vars')     ## Load input_vars dataframe    


#%% Take care of sessions with unusual frame duration!!!    
# make sure below works!

frame_durs = all_sess['frame_dur'].values
rmv = np.logical_or(frame_durs < .089, frame_durs > .1)
if sum(rmv)>0:
    print('take a look at these sessions, and add them to set_sessions_valid, line 178 to be excluded.')
    sys.exit(f'\n\nRemoving {sum(rmv)} sessions with unexpected frame duration!!\n\n')
    all_sess = all_sess[~rmv]

#%%    
print(np.shape(all_sess))
all_sess.iloc[:2]



#%%
samps_bef = input_vars['samps_bef'].iloc[0]
samps_aft = input_vars['samps_aft'].iloc[0]
same_num_neuron_all_planes = input_vars['same_num_neuron_all_planes'].iloc[0]
use_ct_traces = input_vars['use_ct_traces'].iloc[0] 
mean_notPeak = input_vars['mean_notPeak'].iloc[0]
peak_win = input_vars['peak_win'].iloc[0]
flash_win = input_vars['flash_win'].iloc[0]
flash_win_timing = input_vars['flash_win_timing'].iloc[0]
bl_percentile = input_vars['bl_percentile'].iloc[0]
doShift_again = input_vars['doShift_again'].iloc[0]

# frames_svm = range(-samps_bef, samps_aft)


#%% Initialize variables

if type(frames_svm)==int: # First type you ran svm analysis: SVM was run on 30 frames after omission and 0 frames before omission (each frame after omission was compared with a gray frame (frame -1 relative to omission))
    samps_bef = 0
    samps_aft = frames_svm #frames_after_omission # 30 # frames_after_omission in svm_main # we trained the classifier for 30 frames after omission    
    
else:        
    samps_bef = -frames_svm[0] 
    samps_aft = frames_svm[-1]+1 
    
numFrames = samps_bef + samps_aft
frames_svm = np.array(frames_svm)

dir_now = svmn #'omit_across_sess'
fmt = '.pdf'
if not os.path.exists(os.path.join(dir0, dir_now)):
    os.makedirs(os.path.join(dir0, dir_now))
            
now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")


#%% Set vars related to plotting

get_ipython().magic(u'matplotlib inline') # %matplotlib inline

num_depth = 4

cols_each = colorOrder(num_planes)
cols_area = ['b', 'k']    # first plot V1 (in black, index 1) then LM (in blue, index 0)   
cols_depth = ['b', 'c', 'g', 'r'] #colorOrder(num_depth) #a.shape[0]) # depth1_area2    
alph = .3 # for plots, transparency of errorbars
bb = (.92, .7) # (.9, .7)
xlim = [-1.2, 2.25] # [-13, 24]

fgn = 'ClassAccur'
if same_num_neuron_all_planes:
    fgn = fgn + '_sameNumNeursAllPlanes'
fgn = fgn + '_omitMean' + str(peak_win)
fgn = fgn + '_flashMean' + str(flash_win)
fgn = fgn + '_flashTiming' + str(flash_win_timing)
    
ylabel='% Classification accuracy'


#%% Set a number of useful variables

# Set time for the interpolated traces (every 3ms time points); 
# Set the time of flashes and grays for the imaging traces;
# Set some useful vars which you will need for most subsequent analaysis: # all_mice_id, stages_sessionN, mouse_trainHist_all, mouseCre_numSess
# Set plane indeces for each area (V1 and LM)
# %run -i 'set_timetrace_flashesind_allmice_areas.py'
time_trace, time_trace_new, xmjn, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, flashes_win_trace_index_unq, grays_win_trace_index_unq, all_mice_id, stages_sessionN, mouse_trainHist_all, mouseCre_numSess, distinct_areas, i_areas, inds_v1, inds_lm = \
    set_timetrace_flashesind_allmice_areas(samps_bef, samps_aft, frame_dur, doCorrs, all_sess)


# Set sessions for analysis based on their stage (A, B)
all_sess_2an, whatSess = set_mousetrainhist_allsess2an_whatsess(all_sess, dir_server_me, all_mice_id, all_ABtransit_AbefB_Aall, only_1st_transit)



#%% Set svm vars for each plane across all sessions (for each mouse)

columns0 = ['mouse_id', 'cre', 'cre_exp', 'session_stages', 'session_labs', 'num_sessions_valid', 'area_this_plane_allsess_allp', 'depth_this_plane_allsess_allp', \
            'area', 'depth', 'plane', \
            'n_neurons_this_plane_allsess_allp', 'n_omissions_this_plane_allsess_allp', \
            'av_meanX_avSess_eachP', 'sd_meanX_avSess_eachP', \
            'av_test_data_this_plane_allsess_allp', 'av_test_shfl_this_plane_allsess_allp', \
            'av_test_data_avSess_eachP', 'sd_test_data_avSess_eachP', 'av_test_shfl_avSess_eachP', 'sd_test_shfl_avSess_eachP', \
            'av_train_data_avSess_eachP', 'sd_train_data_avSess_eachP', \
            'av_n_neurons_avSess_eachP', 'sd_n_neurons_avSess_eachP', 'av_n_omissions_avSess_eachP', 'sd_n_omissions_avSess_eachP', \
            'av_test_data_pooled_sesss_planes_eachArea', 'av_test_shfl_pooled_sesss_planes_eachArea', 'meanX_pooled_sesss_planes_eachArea', \
            'num_sessions_valid_eachArea', 'distinct_areas', 'cre_pooled_sesss_planes_eachArea', \
            'av_test_data_pooled_sesss_areas_eachDepth', 'av_test_shfl_pooled_sesss_areas_eachDepth', 'meanX_pooled_sesss_areas_eachDepth', \
            'cre_pooled_sesss_areas_eachDepth', 'depth_pooled_sesss_areas_eachDepth', \
            'peak_amp_omit_trTsShCh_this_plane_allsess_allp', 'peak_timing_omit_trTsShCh_this_plane_allsess_allp', 'peak_amp_flash_trTsShCh_this_plane_allsess_allp', 'peak_timing_flash_trTsShCh_this_plane_allsess_allp', \
            'av_peak_amp_omit_trTsShCh_avSess_eachP', 'sd_peak_amp_omit_trTsShCh_avSess_eachP', 'av_peak_timing_omit_trTsShCh_avSess_eachP', 'sd_peak_timing_omit_trTsShCh_avSess_eachP', 'av_peak_amp_flash_trTsShCh_avSess_eachP', 'sd_peak_amp_flash_trTsShCh_avSess_eachP', 'av_peak_timing_flash_trTsShCh_avSess_eachP', 'sd_peak_timing_flash_trTsShCh_avSess_eachP', \
            'peak_amp_omit_trTsShCh_pooled_sesss_planes_eachArea', 'peak_timing_omit_trTsShCh_pooled_sesss_planes_eachArea', 'peak_amp_flash_trTsShCh_pooled_sesss_planes_eachArea', 'peak_timing_flash_trTsShCh_pooled_sesss_planes_eachArea', \
            'peak_amp_omit_trTsShCh_pooled_sesss_areas_eachDepth', 'peak_timing_omit_trTsShCh_pooled_sesss_areas_eachDepth', 'peak_amp_flash_trTsShCh_pooled_sesss_areas_eachDepth', 'peak_timing_flash_trTsShCh_pooled_sesss_areas_eachDepth', \
            ]

if same_num_neuron_all_planes:
    columns = np.concatenate((columns0, ['n_neurons_svm_trained_this_plane_allsess_allp', 'av_n_neurons_svm_trained_avSess_eachP', 'sd_n_neurons_svm_trained_avSess_eachP'])) 
else:
    columns = columns0

svm_this_plane_allsess = pd.DataFrame([], columns=columns)

        
# loop over mice based on the order in all_mice_id
# Note: if you want to use all sessions (not just those in all_sess_2n), replace all_sess_2n with all_sess below, also uncomment the currerntly commented out defintion of session_stages
for im in range(len(all_mice_id)): # im=0
    
    mouse_id = all_mice_id[im]
    
    if sum((all_sess_2an['mouse_id']==mouse_id).values) <= 0:
        print(f'mouse {im} doesnt have data!')
        
    else: #sum((all_sess_2an['mouse_id']==mouse_id).values) > 0: # make sure there is data for this mouse, for the specific session: A, B, etc that you care about
        print(f'there is data for mouse {im}')
    
        all_sess_2an_this_mouse = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]
#         cre = all_sess_2an_this_mouse['cre'].iloc[0]

        cre = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['cre'].iloc[0]
        cre = cre[:cre.find('-')] # remove the IRES-Cre part

        session_stages = all_sess_2an_this_mouse['stage'].values[np.arange(0, all_sess_2an_this_mouse.shape[0], num_planes)]
#         session_stages = mouse_trainHist_all[mouse_trainHist_all['mouse_id']==mouse_id]['stage'].values
        num_sessions = len(session_stages)     # some planes for some sessions could be nan (because they had fewer neurons than 3 (svm_min_neurs)), so below we will get the accurate number of sessions for each plane separately
#         print(cre, num_sessions)

        ######## Set session labels (stage names are too long)
    #    session_beg_inds = np.concatenate(([0], np.cumsum(num_trs_each_sess[:-1])+1))
        session_labs = []
        for ibeg in range(num_sessions):
            a = str(session_stages[ibeg]); 
            en = np.argwhere([a[ich]=='_' for ich in range(len(a))])[-1].squeeze() + 6
            txt = str(session_stages[ibeg][15:]) # session_stages[ibeg][8:en]
            txt = txt[0:min(3, len(txt))]
            session_labs.append(txt)
                    
        #### replicate cre to the number of sessions ####
        cre_exp = np.full((num_planes*num_sessions), cre)

        areas = all_sess_2an_this_mouse['area'] # pooled: (8*num_sessions)
        depths = all_sess_2an_this_mouse['depth'] # pooled: (8*num_sessions)
        planes = all_sess_2an_this_mouse.index.values # pooled: (8*num_sessions)
        
        
        ######### Get data from a given plane across all sessions #########
        # remember y is a pandas series; do y.values
        
        # area
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['area']#.values # num_sess
        np.shape(y.values)
        area_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 1)  # 8 is number of planes; nFrames_upsampled is the length of the upsampled time array   # Get data from a given plane across all sessions

        # depth
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['depth']#.values # num_sess
        depth_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 1)  # 8 is number of planes; nFrames_upsampled is the length of the upsampled time array   # Get data from a given plane across all sessions

        # num_neurons
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['n_neurons']#.values
        n_neurons_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 1) 
    #    print(n_neurons_this_plane_allsess_allp)

        # population size used for svm training
        if same_num_neuron_all_planes:
            y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['population_sizes_to_try']#.values # 8 * number_of_session_for_the_mouse
            n_neurons_svm_trained_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x nFrames_upsampled)  
    #        print(n_neurons_svm_trained_this_plane_allsess_allp)    

        # num_omissions
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['n_omissions']#.values
        n_omissions_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 1) 

        
        # peak_amp_omit_trainTestShflChance
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_amp_omit_trainTestShflChance'] # 8 * number_of_session_for_the_mouse
        peak_amp_omit_trTsShCh_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 4) # 4 for train, Test, Shfl, and Chance        
        
        # peak_timing_omit_trainTestShflChance
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_timing_omit_trainTestShflChance'] # 8 * number_of_session_for_the_mouse
        peak_timing_omit_trTsShCh_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 4)         

        # peak_amp_flash_trainTestShflChance
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_amp_flash_trainTestShflChance'] # 8 * number_of_session_for_the_mouse
        peak_amp_flash_trTsShCh_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 4)         
        
        # peak_timing_flash_trainTestShflChance
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_timing_flash_trainTestShflChance'] # 8 * number_of_session_for_the_mouse
        peak_timing_flash_trTsShCh_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 4)         
        
        
        # av_test_data
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['av_test_data_new']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated
        av_test_data_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x nFrames_upsampled)  # 8 is number of planes; nFrames_upsampled is the length of the upsampled time array   # Get data from a given plane across all sessions
    #    print(im, av_test_data_this_plane_allsess_allp.shape)

        # av_test_shfl
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['av_test_shfl_new']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated
        av_test_shfl_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x nFrames_upsampled)  

        # av_train_data
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['av_train_data_new']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated
        av_train_data_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x nFrames_upsampled)  # 8 is number of planes; nFrames_upsampled is the length of the upsampled time array   # Get data from a given plane across all sessions
    #    print(im, av_test_data_this_plane_allsess_allp.shape)

        # meanX
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['meanX_allFrs_new']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated x n_neurons
        # remember, below also takes average across neurons
        meanX_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions, takeAve=1) # size: (8 x num_sessions x nFrames_upsampled)  # 8 is number of planes; nFrames_upsampled is the length of the upsampled time array   # Get data from a given plane across all sessions


        # some planes for some sessions could be nan (because they had fewer neurons than 3 (svm_min_neurs)), so lets get the accurate number of sessions for each plane separately
        num_sessions_valid = np.sum(~np.isnan(n_neurons_this_plane_allsess_allp), axis=1).squeeze() # 8 (shows for each plane the number of non nan sessions)


        
        ############################################################
        ######### For each plane, average vars across days #########
        ############################################################
        # note: vars below are also set in svm_plots_setVars_sumMice.py. I didn't take them out of here, bc below will be used for making eachMouse plots. Codes in svm_plots_setVars_sumMice.py are compatible with omissions code, and will be used for making sumMice plots.
            
        # depth
        a = depth_this_plane_allsess_allp # size: (8 x num_sessions x 1)
        av_depth_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
        sd_depth_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane

        # num_neurons
        a = n_neurons_this_plane_allsess_allp # size: (8 x num_sessions x 1)
        av_n_neurons_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
        sd_n_neurons_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane

        # population size used for svm training
        if same_num_neuron_all_planes:
            a = n_neurons_svm_trained_this_plane_allsess_allp # size: (8 x num_sessions x 1)
            av_n_neurons_svm_trained_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
            sd_n_neurons_svm_trained_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane

        # num_omissions
        a = n_omissions_this_plane_allsess_allp # size: (8 x num_sessions x 1)
        av_n_omissions_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
        sd_n_omissions_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane
        
        # av_test_data
        a = av_test_data_this_plane_allsess_allp # size: (8 x num_sessions x nFrames_upsampled)
        av_test_data_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
        sd_test_data_avSess_eachP = np.nanstd(a,axis=1) / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x nFrames_upsampled # st error across sessions, for each plane

        # av_test_shfl
        a = av_test_shfl_this_plane_allsess_allp # size: (8 x num_sessions x nFrames_upsampled)
        av_test_shfl_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
        sd_test_shfl_avSess_eachP = np.nanstd(a,axis=1) / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x nFrames_upsampled # st error across sessions, for each plane

        # av_train_data
        a = av_train_data_this_plane_allsess_allp # size: (8 x num_sessions x nFrames_upsampled)
        av_train_data_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
        sd_train_data_avSess_eachP = np.nanstd(a,axis=1) / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x nFrames_upsampled # st error across sessions, for each plane

        # meanX_allFrs
        a = meanX_this_plane_allsess_allp # size: (8 x num_sessions x nFrames_upsampled)
        av_meanX_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
        sd_meanX_avSess_eachP = np.nanstd(a,axis=1) / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x nFrames_upsampled # st error across sessions, for each plane

        # peak_amp_omit_trTsShCh
        a = peak_amp_omit_trTsShCh_this_plane_allsess_allp # size: (8 x num_sessions x 4)
        av_peak_amp_omit_trTsShCh_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes x 4
        sd_peak_amp_omit_trTsShCh_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x 4 # st error across sessions, for each plane

        # peak_timing_omit_trTsShCh
        a = peak_timing_omit_trTsShCh_this_plane_allsess_allp # size: (8 x num_sessions x 4)
        av_peak_timing_omit_trTsShCh_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes x 4
        sd_peak_timing_omit_trTsShCh_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x 4 # st error across sessions, for each plane

        # peak_amp_flash_trTsShCh
        a = peak_amp_flash_trTsShCh_this_plane_allsess_allp # size: (8 x num_sessions x 4)
        av_peak_amp_flash_trTsShCh_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes x 4
        sd_peak_amp_flash_trTsShCh_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x 4 # st error across sessions, for each plane

        # peak_timing_flash_trTsShCh
        a = peak_timing_flash_trTsShCh_this_plane_allsess_allp # size: (8 x num_sessions x 4)
        av_peak_timing_flash_trTsShCh_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes x 4
        sd_peak_timing_flash_trTsShCh_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x 4 # st error across sessions, for each plane

        
        #######################################################################
        ######### For each area, pool data across sessions and depths #########            
        #######################################################################
        # 2 x (4 x num_sessions) 
        
        '''
        # Note: use the codes below if you want to use the same codes as in "omissions_traces_peaks_plots_setVars_ave.py"

        area = all_sess_2an_this_mouse['area'].values

        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_amp_omit_trainTestShflChance']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated; 
        t = np.vstack(y) # (8*num_sessions) x num_frames # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time

        trace_pooled_sesss_planes_eachArea, distinct_areas, i_areas = pool_sesss_planes_eachArea(area, t)
        
        # sanity check that above is the same as what we set below
#         np.sum(~np.equal(trace_pooled_sesss_planes_eachArea, peak_amp_omit_trTsShCh_pooled_sesss_planes_eachArea))
        '''
        # NOTE: the reshape below (using order='F') moves in the following order:
        # it first orders data from the 1st area in the following order
        # data from all 4 depths for session 1, area 1
        # data from all 4 depths for session 2, area 1
        # data from all 4 depths for session 3, area 1 .... (then the same thing repeats for the 2nd area)
        #
        # data from all 4 depths for session 1, area 2
        # data from all 4 depths for session 2, area 2
        # data from all 4 depths for session 3, area 2

        # you used to not use order='F' in the reshape below, so it was 1st all sessions for one depth, then all sessions for the other depth, etc
        # but on Apr 13 2020 you added order='F' to make it consistent with omissions plots, which uses function pool_sesss_planes_eachArea.

        a = area_this_plane_allsess_allp # 8 x num_sessions x 1
        a = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # (8 x num_sessions) x 1 # do this to make sure things are exactly the same as how you do for the arrays below (reshape is confusing in python!!)
        distinct_areas, i_areas = np.unique(a, return_inverse=True)
    #    print(distinct_areas)
        if len(distinct_areas) > 2:
            sys.exit('There cannot be more than two areas!! Fix the area names!')


        area = all_sess_2an_this_mouse['area'].values
        
        t = cre_exp # (8*num_sessions)
        cre_pooled_sesss_planes_eachArea,_,_ = pool_sesss_planes_eachArea(area, t) # 2 x (4 x num_sessions) 
        
        # av_test_data, each area
        a = av_test_data_this_plane_allsess_allp # 8 x num_sessions x nFrames_upsampled       
        av_test_data_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x nFrames_upsampled        
        # below has size: num_areas x (num_layers_per_area x num_sessions) x nFrames_upsampled
        # so, 2 x (4 x num_sessions) x nFrames_upsampled
        av_test_data_pooled_sesss_planes_eachArea = np.array([av_test_data_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x nFrames_upsampled

        # av_test_shfl      
        a = av_test_shfl_this_plane_allsess_allp     
        av_test_shfl_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x nFrames_upsampled        
        # below has size: num_areas x (num_layers_per_area x num_sessions) x nFrames_upsampled
        # so, 2 x (4 x num_sessions) x nFrames_upsampled    
        av_test_shfl_pooled_sesss_planes_eachArea = np.array([av_test_shfl_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x nFrames_upsampled

        # meanX_allFrs
        a = meanX_this_plane_allsess_allp     
        meanX_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x nFrames_upsampled        
        # below has size: num_areas x (num_layers_per_area x num_sessions) x nFrames_upsampled
        # so, 2 x (4 x num_sessions) x nFrames_upsampled    
        meanX_pooled_sesss_planes_eachArea = np.array([meanX_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x nFrames_upsampled

        # peak_amp_omit_trTsShCh
        a = peak_amp_omit_trTsShCh_this_plane_allsess_allp # 8 x num_sessions x 4
        peak_amp_omit_trTsShCh_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x 4
        # below has size: num_areas x (num_layers_per_area x num_sessions) x 4
        # so, 2 x (4 x num_sessions) x 4
        peak_amp_omit_trTsShCh_pooled_sesss_planes_eachArea = np.array([peak_amp_omit_trTsShCh_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x 4

        # peak_timing_omit_trTsShCh            
        a = peak_timing_omit_trTsShCh_this_plane_allsess_allp # 8 x num_sessions x 4
        peak_timing_omit_trTsShCh_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x 4
        # below has size: num_areas x (num_layers_per_area x num_sessions) x 4
        # so, 2 x (4 x num_sessions) x 4
        peak_timing_omit_trTsShCh_pooled_sesss_planes_eachArea = np.array([peak_timing_omit_trTsShCh_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x 4

        # peak_amp_flash_trTsShCh            
        a = peak_amp_flash_trTsShCh_this_plane_allsess_allp # 8 x num_sessions x 4
        peak_amp_flash_trTsShCh_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x 4
        # below has size: num_areas x (num_layers_per_area x num_sessions) x 4
        # so, 2 x (4 x num_sessions) x 4
        peak_amp_flash_trTsShCh_pooled_sesss_planes_eachArea = np.array([peak_amp_flash_trTsShCh_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x 4

        # peak_timing_flash_trTsShCh            
        a = peak_timing_flash_trTsShCh_this_plane_allsess_allp # 8 x num_sessions x 4
        peak_timing_flash_trTsShCh_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x 4
        # below has size: num_areas x (num_layers_per_area x num_sessions) x 4
        # so, 2 x (4 x num_sessions) x 4
        peak_timing_flash_trTsShCh_pooled_sesss_planes_eachArea = np.array([peak_timing_flash_trTsShCh_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x 4

        # num_sessions_valid_eachArea
        # note: we are redefining "i_areas" below.
        a = area_this_plane_allsess_allp[:,0,0] # 8
        distinct_areas, i_areas = np.unique(a, return_inverse=True)
        num_sessions_valid_eachArea = np.array([num_sessions_valid[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x 4  # area x depth_per_area

        
        ###############################################################################
        ######### For each layer (depth), pool data across sessions and areas #########            
        ###############################################################################

        ### Note: "xxx_pooled_sesss_areas_eachDepth" vars below have size: # 4depths x (2areas x num_sess) 
        # (2areas x num_sess) are pooled in this way: 
        # first all areas of session 1, then all areas of session 2, etc
        
        # the first 4 depths are for one area and the 2nd four depths are for the other area, but lets code it, just in case:
    #    depth1_area2 = 4 # index of the 1st depth of the 2nd area
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['area'].values
        distinct_areas, i_areas = np.unique(y, return_inverse=True)
        depth1_area2 = np.argwhere(i_areas).squeeze()[0] # 4
        if depth1_area2!=4:
            sys.exit('Because the 8 planes are in the following order: area 1 (4 planes), then area 2 (another 4 planes), we expect depth1_area2 to be 4! What is wrong?!')

        
        planes_allsess = planes # (8*num_sessions)
        depth = depths.values

        t = cre_exp # (8*num_sessions)        
        cre_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess)
        
        t = depth
        depth_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, num_depth=4) # 4 x (2 x num_sessions) # each row is for a depth and shows values from the two areas of all the sessions
        
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['av_test_data_new']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
        t = np.vstack(y) # (8*num_sessions) x num_frames # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time        
        av_test_data_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2
        
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['av_test_shfl_new']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
        t = np.vstack(y) # (8*num_sessions) x num_frames # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time        
        av_test_shfl_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2

        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_amp_omit_trainTestShflChance']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
        t = np.vstack(y) # (8*num_sessions) x 4 # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time        
        peak_amp_omit_trTsShCh_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x 4(train,test,shuffle,chance) # 4 is the number of distinct depths: depth1_area2

        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_timing_omit_trainTestShflChance']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
        t = np.vstack(y) # (8*num_sessions) x 4 # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time        
        peak_timing_omit_trTsShCh_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x 4(train,test,shuffle,chance) # 4 is the number of distinct depths: depth1_area2

        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_amp_flash_trainTestShflChance']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
        t = np.vstack(y) # (8*num_sessions) x 4 # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time        
        peak_amp_flash_trTsShCh_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x 4(train,test,shuffle,chance) # 4 is the number of distinct depths: depth1_area2

        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_timing_flash_trainTestShflChance']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
        t = np.vstack(y) # (8*num_sessions) x 4 # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time        
        peak_timing_flash_trTsShCh_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x 4(train,test,shuffle,chance) # 4 is the number of distinct depths: depth1_area2
        
        # for mouse 440631, code below gives error bc one of the sessions doesnt have valid neurons, and is just a vector of nans (1 dimensional) 
        # so, we go with the old method for setting meanX_pooled_sesss_areas_eachDepth
        # the only difference would be that now (2areas x num_sess) are pooled in this way: 
        # first all sessions of one area, then all sessions of the other area are concatenated.
        # but it doesnt matter bc we use meanX_pooled_sesss_areas_eachDepth only for eachMouse plots, where we average across areas and sessions (so the order of areas and sessions doesnt matter.)
        '''
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['meanX_allFrs_new']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
        # take average across neurons: # (8*num_sessions) x num_frames 
        t = np.array([np.nanmean(y.values[inn], axis=1) for inn in range(len(y.values))])
        meanX_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2
        '''
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id].index.values # (pooled: num_planes x num_sessions)    
        p = np.reshape(y, (num_planes, num_sessions), order='F') # num_planes x num_sessions
        
        # meanX
        meanX_pooled_sesss_areas_eachDepth = []
        for idepth in range(depth1_area2): # idepth = 0
            # merge data with the same depth from 2 areas
            b = np.logical_or(p==idepth, p==idepth + depth1_area2)
            a = np.array([meanX_this_plane_allsess_allp[b]]).squeeze() # (2 x num_sess) x nFrames_upsampled
            meanX_pooled_sesss_areas_eachDepth.append(a)
        meanX_pooled_sesss_areas_eachDepth = np.array(meanX_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2
        
        
        '''
        ### original codes; here, "xxx_pooled_sesss_areas_eachDepth" : 4depths x (2areas x num_sess) 
        # (2areas x num_sess) are pooled in this way: 
        # first all sessions of one area, then all sessions of the other area are concatenated.
        
        ### Note: I am using instead the above codes to make the vars similar to omissions vars
        
        # plane indeces
        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id].index.values # (pooled: num_planes x num_sessions)    
        planes_allsess = np.reshape(y, (num_planes, num_sessions), order='F') # num_planes x num_sessions

        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['depth'].values # (pooled: num_planes x num_sessions)    
        depths_allsess = np.reshape(y, (num_planes, num_sessions), order='F') # num_planes x num_sessions

        # all of the following have the same first 2 dimensions: # num_planes x num_sessions
    #    av_test_data_this_plane_allsess_allp 
    #    area_this_plane_allsess_allp
    #    planes_allsess
    #    depths_allsess
            
        
        # peak_amp_omit_trTsShCh
        peak_amp_omit_trTsShCh_pooled_sesss_areas_eachDepth = []
        for idepth in range(depth1_area2): # idepth = 0
            # merge data with the same depth across 2 areas 
            b = np.logical_or(planes_allsess==idepth, planes_allsess==idepth + depth1_area2) # num_planes x num_sessions
            # 1st dimention of "a" below includes pooled sessions and areas (2 x num_sess) in this way: first all sessions of one area, then all sessions of the other area.
            a = np.array([peak_amp_omit_trTsShCh_this_plane_allsess_allp[b]]).squeeze() # (2 x num_sess) x 4(train,test,shuffle,chance)
            peak_amp_omit_trTsShCh_pooled_sesss_areas_eachDepth.append(a)
        peak_amp_omit_trTsShCh_pooled_sesss_areas_eachDepth = np.array(peak_amp_omit_trTsShCh_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x 4(train,test,shuffle,chance) # 4 is the number of distinct depths: depth1_area2
        
        # peak_timing_omit_trTsShCh                    
        peak_timing_omit_trTsShCh_pooled_sesss_areas_eachDepth = []
        for idepth in range(depth1_area2): # idepth = 0
            # merge data with the same depth across 2 areas 
            b = np.logical_or(planes_allsess==idepth, planes_allsess==idepth + depth1_area2)
            a = np.array([peak_timing_omit_trTsShCh_this_plane_allsess_allp[b]]).squeeze() # (2 x num_sess) x 4(train,test,shuffle,chance)
            peak_timing_omit_trTsShCh_pooled_sesss_areas_eachDepth.append(a)
        peak_timing_omit_trTsShCh_pooled_sesss_areas_eachDepth = np.array(peak_timing_omit_trTsShCh_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x 4(train,test,shuffle,chance) # 4 is the number of distinct depths: depth1_area2
        
        # peak_amp_flash_trTsShCh                    
        peak_amp_flash_trTsShCh_pooled_sesss_areas_eachDepth = []
        for idepth in range(depth1_area2): # idepth = 0
            # merge data with the same depth across 2 areas 
            b = np.logical_or(planes_allsess==idepth, planes_allsess==idepth + depth1_area2)
            a = np.array([peak_amp_flash_trTsShCh_this_plane_allsess_allp[b]]).squeeze() # (2 x num_sess) x 4(train,test,shuffle,chance)
            peak_amp_flash_trTsShCh_pooled_sesss_areas_eachDepth.append(a)
        peak_amp_flash_trTsShCh_pooled_sesss_areas_eachDepth = np.array(peak_amp_flash_trTsShCh_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x 4(train,test,shuffle,chance) # 4 is the number of distinct depths: depth1_area2
        
        # peak_timing_flash_trTsShCh                    
        peak_timing_flash_trTsShCh_pooled_sesss_areas_eachDepth = []
        for idepth in range(depth1_area2): # idepth = 0
            # merge data with the same depth across 2 areas 
            b = np.logical_or(planes_allsess==idepth, planes_allsess==idepth + depth1_area2)
            a = np.array([peak_timing_flash_trTsShCh_this_plane_allsess_allp[b]]).squeeze() # (2 x num_sess) x 4(train,test,shuffle,chance)
            peak_timing_flash_trTsShCh_pooled_sesss_areas_eachDepth.append(a)
        peak_timing_flash_trTsShCh_pooled_sesss_areas_eachDepth = np.array(peak_timing_flash_trTsShCh_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x 4(train,test,shuffle,chance) # 4 is the number of distinct depths: depth1_area2
        
        
        
        # av_test_data        
        av_test_data_pooled_sesss_areas_eachDepth = []
        for idepth in range(depth1_area2): # idepth = 0
            # merge data with the same depth across 2 areas 
            b = np.logical_or(planes_allsess==idepth, planes_allsess==idepth + depth1_area2)
            a = np.array([av_test_data_this_plane_allsess_allp[b]]).squeeze() # (2 x num_sess) x nFrames_upsampled
            av_test_data_pooled_sesss_areas_eachDepth.append(a)
        av_test_data_pooled_sesss_areas_eachDepth = np.array(av_test_data_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2

        # av_test_shfl
        av_test_shfl_pooled_sesss_areas_eachDepth = []
        for idepth in range(depth1_area2): # idepth = 0
            # merge data with the same depth from 2 areas
            b = np.logical_or(planes_allsess==idepth, planes_allsess==idepth + depth1_area2)
            a = np.array([av_test_shfl_this_plane_allsess_allp[b]]).squeeze() # (2 x num_sess) x nFrames_upsampled
            av_test_shfl_pooled_sesss_areas_eachDepth.append(a)
        av_test_shfl_pooled_sesss_areas_eachDepth = np.array(av_test_shfl_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2

        # meanX
        meanX_pooled_sesss_areas_eachDepth = []
        for idepth in range(depth1_area2): # idepth = 0
            # merge data with the same depth from 2 areas
            b = np.logical_or(planes_allsess==idepth, planes_allsess==idepth + depth1_area2)
            a = np.array([meanX_this_plane_allsess_allp[b]]).squeeze() # (2 x num_sess) x nFrames_upsampled
            meanX_pooled_sesss_areas_eachDepth.append(a)
        meanX_pooled_sesss_areas_eachDepth = np.array(meanX_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2
        '''

        '''
    #    [depth_this_plane_allsess_allp[4], area_this_plane_allsess_allp[4]]
    #    [depth_this_plane_allsess_allp[0], area_this_plane_allsess_allp[0]]
        av_test_data_this_plane_allsess_allp[0] # depth 0, LM 
        av_test_data_this_plane_allsess_allp[4] # depth 0, V1

    #    [depth_this_plane_allsess_allp[4+1], area_this_plane_allsess_allp[4+1]]
    #    [depth_this_plane_allsess_allp[1], area_this_plane_allsess_allp[1]]
        av_test_data_this_plane_allsess_allp[0+1] # depth 1, LM 
        av_test_data_this_plane_allsess_allp[4+1] # depth 1, V1
        '''    


        ###############################################################
        # areas.values, depth.values, plane, 
        svm_this_plane_allsess.at[im, columns0] = \
                   mouse_id, cre, cre_exp, session_stages, session_labs, num_sessions_valid, area_this_plane_allsess_allp, depth_this_plane_allsess_allp, \
                   areas.values, depths.values, planes, \
                   n_neurons_this_plane_allsess_allp, n_omissions_this_plane_allsess_allp, \
                   av_meanX_avSess_eachP, sd_meanX_avSess_eachP, \
                   av_test_data_this_plane_allsess_allp, av_test_shfl_this_plane_allsess_allp,\
                   av_test_data_avSess_eachP, sd_test_data_avSess_eachP, av_test_shfl_avSess_eachP, sd_test_shfl_avSess_eachP, av_train_data_avSess_eachP, sd_train_data_avSess_eachP,\
                   av_n_neurons_avSess_eachP, sd_n_neurons_avSess_eachP, av_n_omissions_avSess_eachP, sd_n_omissions_avSess_eachP,\
                   av_test_data_pooled_sesss_planes_eachArea, av_test_shfl_pooled_sesss_planes_eachArea, meanX_pooled_sesss_planes_eachArea, \
                   num_sessions_valid_eachArea, distinct_areas, cre_pooled_sesss_planes_eachArea, \
                   av_test_data_pooled_sesss_areas_eachDepth, av_test_shfl_pooled_sesss_areas_eachDepth, meanX_pooled_sesss_areas_eachDepth, \
                   cre_pooled_sesss_areas_eachDepth, depth_pooled_sesss_areas_eachDepth, \
                   peak_amp_omit_trTsShCh_this_plane_allsess_allp, \
                   peak_timing_omit_trTsShCh_this_plane_allsess_allp,\
                   peak_amp_flash_trTsShCh_this_plane_allsess_allp,\
                   peak_timing_flash_trTsShCh_this_plane_allsess_allp,\
                   av_peak_amp_omit_trTsShCh_avSess_eachP, sd_peak_amp_omit_trTsShCh_avSess_eachP,\
                   av_peak_timing_omit_trTsShCh_avSess_eachP, sd_peak_timing_omit_trTsShCh_avSess_eachP,\
                   av_peak_amp_flash_trTsShCh_avSess_eachP, sd_peak_amp_flash_trTsShCh_avSess_eachP,\
                   av_peak_timing_flash_trTsShCh_avSess_eachP, sd_peak_timing_flash_trTsShCh_avSess_eachP,\
                   peak_amp_omit_trTsShCh_pooled_sesss_planes_eachArea,\
                   peak_timing_omit_trTsShCh_pooled_sesss_planes_eachArea,\
                   peak_amp_flash_trTsShCh_pooled_sesss_planes_eachArea,\
                   peak_timing_flash_trTsShCh_pooled_sesss_planes_eachArea,\
                   peak_amp_omit_trTsShCh_pooled_sesss_areas_eachDepth,\
                   peak_timing_omit_trTsShCh_pooled_sesss_areas_eachDepth,\
                   peak_amp_flash_trTsShCh_pooled_sesss_areas_eachDepth,\
                   peak_timing_flash_trTsShCh_pooled_sesss_areas_eachDepth


        if same_num_neuron_all_planes:
            svm_this_plane_allsess.at[im, ['n_neurons_svm_trained_this_plane_allsess_allp']] = [n_neurons_svm_trained_this_plane_allsess_allp]
            svm_this_plane_allsess.at[im, ['av_n_neurons_svm_trained_avSess_eachP']] = [av_n_neurons_svm_trained_avSess_eachP]
            svm_this_plane_allsess.at[im, ['sd_n_neurons_svm_trained_avSess_eachP']] = [sd_n_neurons_svm_trained_avSess_eachP]



#%%
#####################################################################################
####################################### PLOTS #######################################
#####################################################################################

# Follow this script by "svm_plots_eachMouse" to make plots for each mouse.

# Follow this script by "svm_plots_setVars_sumMice.py" to set vars for making average plots across mice (for each cre line).


                
