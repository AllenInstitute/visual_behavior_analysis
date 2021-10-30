#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The 1st script to run to make plots. Before this script, "svm_images_plots_init" must be run to save all_sess dataframe.

This script loads all_sess that is set by function svm_main_post (called in svm_images_init).
It sets all_sess_2an, which is a subset of all_sess that only includes desired sessions for analysis (A, or B, etc).

Its output is a pandas table "svm_this_plane_allsess" which includes svm variables for each mouse, (for each plane, for each area, and for each depth) and will be used for plotting.

This script will call the following scripts to make all the plots related to svm analysis:
"svm_images_plots_eachMouse" to make plots for each mouse.
"svm_images_plots_setVars_sumMice.py" and "svm_images_plots_sumMice.py" which set vars and make average plots across mice (for each cre line).
"svm_images_plots_compare_ophys_stages.py" to make comparison of the decoding performance across ophys stages (or experience levels if using matched cells).

Note, this script is similar to part of omissions_traces_peaks_plots_setVars_ave.py.
    The variable svm_this_plane_allsess here is basically a combination of the following 2 vars in omissions_traces_peaks_plots_setVars_ave.py:
        trace_peak_allMice   :   includes traces and peak measures for all planes of all sessions, for each mouse.
        pooled_trace_peak_allMice   : includes layer- and area-pooled vars, for each mouse.

This script also sets some of the session-averaged vars per mouse (in var "svm_this_plane_allsess") , which will be used in eachMouse plots; 
    # note: those vars are also set in svm_plots_setVars_sumMice.py (in var "svm_this_plane_allsess_sessAvSd". I didn't take them out of here, bc below will be used for making eachMouse plots. Codes in svm_plots_setVars_sumMice.py are compatible with omissions code, and will be used for making sumMice plots.


NOTE: This code is the new version that replaced the old "svm_images_plots_setVars.py". It works for both when the svm was run on the whole session or when it was run on multiple blocks during the session.
It will create plots for each stage (ophy1, 2, etc). If you want average plots across stages, you need to modify line 60 of code "svm_images_plots_setVars_sumMice_blocks.py" so we dont get only a given stage out of svm_this_plane_allsess0


Created on Tue Oct  20 13:56:00 2020
@author: farzaneh
"""

'''
Relevant parts of different scripts, for quantifying response amplitudes:

# average across cv samples # size perClassErrorTrain_data_allFrs: nSamples x nFrames
av_train_data = 100-np.nanmean(perClassErrorTrain_data_allFrs, axis=0) # numFrames


# concatenate CA traces for training, testing, shfl, and chance data, each a column in the matrix below; to compute their peaks all at once.
CA_traces = np.vstack((av_train_data, av_test_data, av_test_shfl, av_test_chance)).T # times x 4
#             print(CA_traces.shape)

peak_amp_trainTestShflChance = np.nanmean(CA_traces[time_win_frames], axis=0)


# peak_amp_trainTestShflChance
y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_amp_trainTestShflChance'] # 8 * number_of_session_for_the_mouse
peak_amp_trTsShCh_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 4) # 4 for train, Test, Shfl, and Chance        


aa = svm_this_plane_allsess['peak_amp_trTsShCh_this_plane_allsess_allp'].values # num_all_mice; # size of each element : 8 x num_sessions x 4
aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))]) # aar.shape # num_all_mice; # size of each element : (8 x num_sessions) x 4
a0 = np.concatenate((aar)) # (8*sum(num_sess_per_mouse)) x 4 # 8 planes of 1 session, then 8 planes of next session, and so on 
pa_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes), a0.shape[-1]), order='F') # 8 x sum(num_sess_per_mouse) x 4 # each row is one plane, all sessions        


pa_all = svm_allMice_sessPooled['peak_amp_allPlanes'].values[0]  # 8 x pooledSessNum x 4 (trTsShCh)

top = np.nanmean(pa_all[:, cre_all[0,:]==cre], axis=1)
top_sd = np.nanstd(pa_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(sum(cre_all[0,:]==cre))        


# testing data
ax1.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt=fmt_now, markersize=3, capsize=3, label=lab1, color=cols_area_now[1])
ax1.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt=fmt_now, markersize=3, capsize=3, label=lab2, color=cols_area_now[0])

# shuffled data
ax1.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt=fmt_now, markersize=3, capsize=3, color='gainsboro') # label='V1',  # gray
ax1.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt=fmt_now, markersize=3, capsize=3, color='gainsboro') # label='LM', # skyblue

'''

import os
import numpy as np
import pandas as pd
import re
import datetime
import copy
import sys
import numpy.ma as ma
import seaborn as sns

from general_funs import *

import visual_behavior.data_access.loading as loading
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
    
dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
dir0 = '/home/farzaneh/OneDrive/Analysis'


#%% Set the following vars

project_codes = ['VisualBehavior'], ['VisualBehaviorTask1B'], ['VisualBehaviorMultiscope'] # pooled: ['VisualBehavior'], ['VisualBehaviorMultiscope'] # ['VisualBehaviorMultiscope'] # ['VisualBehaviorMultiscope', 'VisualBehaviorTask1B', 'VisualBehavior', 'VisualBehaviorMultiscope4areasx2d']

to_decode = 'current' # 'current': decode current image. # 'previous': decode previous image. # 'next': decode next image.
trial_type = 'changes' # 'baseline_vs_nobaseline' # 'omissions' # 'changes' # 'hits_vs_misses' # 'changes_vs_nochanges' # 'images'# what trials to use for SVM analysis # the population activity of these trials at time time_win will be used to decode the image identity of flashes that occurred at their time 0 (if to_decode='current') or 750ms before (if to_decode='previous'). # eg 'omissions' means to use omission-aligned traces # 'baseline_vs_nobaseline' # decode activity at each frame vs. baseline (ie the frame before omission unless use_spont_omitFrMinus1 = 1 (see below))
# Note: when trial_type is 'hits_vs_misses' or 'changes_vs_nochanges', to_decode will be 'current' and wont really make sense.
# in all other cases, we decode "to_decode" image from "trial_type", e.g. we decode 'current' image from 'changes' (ie change-aligned traces)

use_events = True # True # False #whether to run the analysis on detected events (inferred spikes) or dff traces.
svm_blocks = np.nan #-101 #-1: divide trials based on engagement # -101: use only engaged epochs for svm analysis # number of trial blocks to divide the session to, and run svm on. # set to np.nan to run svm analysis on the whole session

use_matched_cells = 123 # 0: analyze all cells of an experiment. 123: analyze cells matched in all experience levels (familiar, novel 1 , novel >1); 12: cells matched in familiar and novel 1.  23: cells matched in novel 1 and novel >1.  13: cells matched in familiar and novel >1

engagement_pupil_running = 0 # applicable when svm_blocks=-1 # np.nan or 0,1,2 for engagement, pupil, running: which metric to use to define engagement? 
baseline_subtract = 0 #1 # subtract the baseline (CA average during baseline, ie before time 0) from the evoked CA (classification accuracy)

summary_which_comparison = [3, 4, 6] #'all' # an array of session numbers: [3, 4, 6] or the following strings: # 'novelty' # 'engagement' # 'all' # determins sessions to use for plotting summary of ophys stages in svm_images_plots_compare_ophys_stages.py # 'novelty' will use [1,3,4,6] # 'engagement' will use [1,2,3] # 'all' will use [1,2,3,4,5,6]
dosavefig = 1 # 0
fmt = '.pdf' # '.png' # '.svg'



#%% The following vars we usually don't change, but you can if you want to.

plot_svm_weights = 0
plot_single_mouse = 0 # if 1, make plots for each mouse

# the following 3 vars will be used for plots to correlate classification accuracy with behavioral strategy, and only if trial_type =='changes_vs_nochanges' or trial_type =='hits_vs_misses'
pool_all_stages = True #True
fit_reg = False # lmplots: add the fitted line to the catter plot or not
superimpose_all_cre = False # plot all cre lines on the same figure
plot_testing_shufl = 0 # if 0 correlate testing data with strategy dropout index; if 1, correlated shuffled data with strategy dropout index

use_spont_omitFrMinus1 = 1 # applicable when trial_type='baseline_vs_nobaseline' # if 0, classify omissions against randomly picked spontanoues frames (the initial gray screen); if 1, classify omissions against the frame right before the omission 

# Note: in the cases below, we are decoding baseline from no-baseline , hits from misses, and changes from no-changes, respectively. 
# since we are not decoding images, to_decode really doesnt mean anything (when decoding images, to_decode means current/next/previous image) and we just set it to 'current' so it takes some value.
if trial_type == 'baseline_vs_nobaseline' or trial_type == 'hits_vs_misses' or trial_type == 'changes_vs_nochanges':
    to_decode = 'current'
    use_balanced_trials = 1 #only effective when num_classes=2 #if 1, use same number of trials for each class; only applicable when we have 2 classes (binary classification).
else: # decoding images
    use_balanced_trials = 0 #1 #only effective when num_classes=2 #if 1, use same number of trials for each class; only applicable when we have 2 classes (binary classification).
    

# NOTE: below should match the plots_init code. or better you can load    
'''
if use_events:
    time_win = [0, .4]
else:
    time_win = [0, .55] # 'frames_svm' # set time_win to a string (any string) to use frames_svm as the window of quantification. # time window relative to trial onset to quantify image signal. Over this window class accuracy traces will be averaged.
if trial_type == 'baseline_vs_nobaseline':
    time_win = [0, .75] # we use the entire time after omission because the omission responses slowly go up
'''

same_num_neuron_all_planes = 0 #1 # if 1, use the same number of neurons for all planes to train svm

# cre2ana_all = 'slc', 'sst', 'vip'
use_same_experiments_dff_events = False #True # use the same set of experiments for both events and dff analysis (Note: for this to work you need to get both ea_evs and ev_dff; for this run the code until line ~300 twice once setting use_events to True and once to False.)




#%% Initialize variables

e = 'events_' if use_events else ''  

if trial_type=='changes_vs_nochanges': # change, then no-change will be concatenated
    svmn = f'{e}svm_decode_changes_from_nochanges'
elif trial_type=='hits_vs_misses':
    svmn = f'{e}svm_decode_hits_from_misses'        
elif trial_type == 'baseline_vs_nobaseline':
    svmn = f'{e}svm_decode_baseline_from_nobaseline' #f'{e}svm_gray_omit' #svm_decode_baseline_from_nobaseline
    if use_spont_omitFrMinus1==0:
        svmn = svmn + '_spontFrs'
else:
    svmn = f'{e}svm_decode_{to_decode}_image_from_{trial_type}' # 'svm_gray_omit'


if use_matched_cells==123:
    svmn = svmn + '_matched_cells_FN1N2' #Familiar, N1, N+1
elif use_matched_cells==12:
    svmn = svmn + '_matched_cells_FN1'
elif use_matched_cells==23:
    svmn = svmn + '_matched_cells_N1Nn'
elif use_matched_cells==13:
    svmn = svmn + '_matched_cells_FNn'        


if svm_blocks==-101: # run svm analysis only on engaged trials; redifine df_data only including the engaged rows
    svmn = f'{svmn}_only_engaged'    
    
if use_balanced_trials:
    svmn = f'{svmn}_equalTrs'

print(svmn)

    
dir_now = svmn #'omit_across_sess'
if use_events:
    dir_now = 'svm_' + svmn[len('events_svm_'):] + '_events' # so all the folders start with svm

if svm_blocks==-1:
    dir_now = os.path.join(dir_now, f'engaged_disengaged_blocks')
elif svm_blocks==-101:
    dir_now = os.path.join(dir_now, f'only_engaged')
elif ~np.isnan(svm_blocks):
    dir_now = os.path.join(dir_now, f'trial_blocks')
        
dd = os.path.join(dir0, 'svm', dir_now)
if not os.path.exists(dd):
    os.makedirs(dd)

now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")



#%% Set vars related to plotting

get_ipython().magic(u'matplotlib inline') # %matplotlib inline

cols_area = ['b', 'k']    # first plot V1 (in black, index 1) then LM (in blue, index 0)   
cols_depth = ['b', 'c', 'g', 'r'] #colorOrder(num_depth) #a.shape[0]) # depth1_area2    
alph = .3 # for plots, transparency of errorbars
bb = (.92, .7) # (.9, .7)

ylabel = '% Classification accuracy'

    
    
##############################################################################
##############################################################################
##############################################################################
#%% Load and concatenate all_sess from all cre lines and all sessions
##############################################################################
##############################################################################
##############################################################################

#%%
dir_svm = os.path.join(dir_server_me, 'SVM')
if same_num_neuron_all_planes:
    dir_svm = os.path.join(dir_svm, 'same_num_neurons_all_planes')
if ~np.isnan(svm_blocks):
    dir_svm = os.path.join(dir_svm, f'trial_blocks')


    

#%%
if svm_blocks==-101:
    br = [np.nan]    
elif ~np.isnan(svm_blocks):
    if svm_blocks==-1: # divide trials into blocks based on the engagement state
        svmb = 2
    else:
        svmb = svm_blocks
    br = range(svmb)    
else:
    br = [np.nan]
print(br)

project_codes_all = copy.deepcopy(project_codes)


#####################################################################################
#%% Follow this script by "svm_images_plots_setVars_sumMice.py" to set vars for making average plots across mice (for each cre line).
#####################################################################################

if len(project_codes_all)==1:

    #%% Set frames_svm, samps_bef and samps_aft
    # Note: trials df has a much longer time_trace (goes up to 4.97) compared to stimulus df (goes up to .71), so frames_svm ends up being 1 element longer for trials df (ie when decoding hits from misses) compared to stimulus df (ie when decoding the images)    
    if project_codes != ['VisualBehaviorMultiscope']:
        frames_svm = np.arange(-15,23)
        num_planes = 1
        num_depth = 1
    else:
        frames_svm = np.arange(-5,8)
        num_planes = 8
        num_depth = 4

    if trial_type=='hits_vs_misses': # frames_svm[-1] = frames_svm[-1] + 1
        frames_svm = np.concatenate((frames_svm, [frames_svm[-1] + 1]))

    # numFrames = samps_bef + samps_aft
    frames_svm = np.array(frames_svm)

    samps_bef = -frames_svm[0] # np.argwhere(trace_time==0)[0][0] # 5
    samps_aft = len(frames_svm)-samps_bef #8 
    print(samps_bef, samps_aft)

    # if trial_type=='hits_vs_misses':
    #     samps_aft = samps_aft + 1

    
    #%%    
    cols_each = colorOrder(num_planes)
    
    

    #%% Set svm vars for each plane across all sessions (for each mouse)
    if project_codes != ['VisualBehaviorMultiscope']: # remove area/layer pooled columns
        columns0 = ['mouse_id', 'cre', 'mouse_id_exp', 'cre_exp', 'block', 'session_ids', 'experience_levels', 'session_stages', 'session_labs', 'num_sessions_valid', 'area_this_plane_allsess_allp', 'depth_this_plane_allsess_allp', \
                    'area', 'depth', 'plane', \
                    'n_neurons_this_plane_allsess_allp', 'n_trials_this_plane_allsess_allp', \
                    'av_meanX_avSess_eachP', 'sd_meanX_avSess_eachP', \
                    'av_test_data_this_plane_allsess_allp', 'av_test_shfl_this_plane_allsess_allp', \
                    'av_test_data_avSess_eachP', 'sd_test_data_avSess_eachP', 'av_test_shfl_avSess_eachP', 'sd_test_shfl_avSess_eachP', \
                    'av_train_data_avSess_eachP', 'sd_train_data_avSess_eachP', \
                    'av_n_neurons_avSess_eachP', 'sd_n_neurons_avSess_eachP', 'av_n_trials_avSess_eachP', 'sd_n_trials_avSess_eachP', \
                    'distinct_areas', \
                    'peak_amp_trTsShCh_this_plane_allsess_allp', \
                    'av_peak_amp_trTsShCh_avSess_eachP', 'sd_peak_amp_trTsShCh_avSess_eachP', \
                    ]    

    else:
        columns0 = ['mouse_id', 'cre', 'mouse_id_exp', 'cre_exp', 'block', 'session_ids', 'experience_levels', 'session_stages', 'session_labs', 'num_sessions_valid', 'area_this_plane_allsess_allp', 'depth_this_plane_allsess_allp', \
                    'area', 'depth', 'plane', \
                    'n_neurons_this_plane_allsess_allp', 'n_trials_this_plane_allsess_allp', \
                    'av_meanX_avSess_eachP', 'sd_meanX_avSess_eachP', \
                    'av_test_data_this_plane_allsess_allp', 'av_test_shfl_this_plane_allsess_allp', \
                    'av_test_data_avSess_eachP', 'sd_test_data_avSess_eachP', 'av_test_shfl_avSess_eachP', 'sd_test_shfl_avSess_eachP', \
                    'av_train_data_avSess_eachP', 'sd_train_data_avSess_eachP', \
                    'av_n_neurons_avSess_eachP', 'sd_n_neurons_avSess_eachP', 'av_n_trials_avSess_eachP', 'sd_n_trials_avSess_eachP', \
                    'av_test_data_pooled_sesss_planes_eachArea', 'av_test_shfl_pooled_sesss_planes_eachArea', 'meanX_pooled_sesss_planes_eachArea', \
                    'num_sessions_valid_eachArea', 'distinct_areas', 'cre_pooled_sesss_planes_eachArea', \
                    'av_test_data_pooled_sesss_areas_eachDepth', 'av_test_shfl_pooled_sesss_areas_eachDepth', 'meanX_pooled_sesss_areas_eachDepth', \
                    'cre_pooled_sesss_areas_eachDepth', 'depth_pooled_sesss_areas_eachDepth', \
                    'peak_amp_trTsShCh_this_plane_allsess_allp', \
                    'av_peak_amp_trTsShCh_avSess_eachP', 'sd_peak_amp_trTsShCh_avSess_eachP', \
                    'peak_amp_trTsShCh_pooled_sesss_planes_eachArea', \
                    'peak_amp_trTsShCh_pooled_sesss_areas_eachDepth', \
                    ]

    if same_num_neuron_all_planes:
        columns = np.concatenate((columns0, ['n_neurons_svm_trained_this_plane_allsess_allp', 'av_n_neurons_svm_trained_avSess_eachP', 'sd_n_neurons_svm_trained_avSess_eachP'])) 
    else:
        columns = columns0
    
    
    #####################################
    #%% Set svm_this_plane_allsess
    #####################################

    
    ######### set svm_this_plane_allsess

    exec(open('svm_images_plots_setVars2.py').read())     
    
    svm_this_plane_allsess0 = copy.deepcopy(svm_this_plane_allsess)


    ######### set svm_allMice_sessPooled and svm_allMice_sessAvSd
    # note: svm_allMice_sessPooled will be used to set summary_vars_all, which is a key paramter in svm_images_plots_compare_ophys_stages.py)    
    exec(open('svm_images_plots_setVars_sumMice.py').read()) 

    svm_allMice_sessPooled0 = copy.deepcopy(svm_allMice_sessPooled)
    svm_allMice_sessAvSd0 = copy.deepcopy(svm_allMice_sessAvSd)

    
    ######### set vars to make mouse-averaged plots for each ophys stage
    # also set summary_vars_all (a df that includes response amplitude, computed from svm_allMice_sessPooled) which will be used in svm_images_plots_compare_ophys_stages
    exec(open('svm_images_plots_setVars_sumMice2.py').read()) 

    # make mouse-averaged plots
    exec(open('svm_images_plots_sumMice.py').read()) 


    ######### compare quantifications across ophys stages
    exec(open('svm_images_plots_compare_ophys_stages.py').read())



    ############################################################
    #%% Plot svm weight distributions

    if plot_svm_weights and project_codes == ['VisualBehaviorMultiscope']:

        ifr = [4,6]
        plt.figure(figsize = (10,6))
        icre = 0
        for cren in ['Slc17a7-IRES2-Cre', 'Sst-IRES-Cre', 'Vip-IRES-Cre']:
            icre = icre+1
            bb = all_sess[all_sess['cre']==cren]; 
            c = np.concatenate((bb['av_w_data'].values), axis=(1)) # (8, 730, 13)
        #     cc = c[:,:,ifr] # frame after image onset
            cc = c[:,:,ifr[1]] - c[:,:,ifr[0]]

            a = cc.flatten() # pool all the 8 decoders
        #     a = np.nanmean(cc, axis=0); # get average of the 8 decoders
        #     a = cc[di] # only take one decoder

            print(bb.shape, a.shape)

            # plot weights
            plt.subplot(2, 3, icre)    
            plt.hist(a, 50) #, density=True); 
            plt.xlabel('svm weights'); plt.ylabel('number of neurons'); 
            plt.title(f'{cren[:3]}, mean={np.nanmean(a):.4f}')    

            # plot absolute weights
            plt.subplot(2, 3, icre+3)
            plt.hist(np.abs(a), 50) #, density=True); 
            plt.xlabel('svm absolute weights'); plt.ylabel('number of neurons'); 
            plt.title(f'{cren[:3]}, mean={np.nanmean(np.abs(a)):.4f}')

            plt.subplots_adjust(wspace=0.5, hspace=0.5)

            plt.suptitle(f'frame {ifr}, 8 decoders pooled')
            plt.suptitle(f'Average of 8 decoders')
        #     plt.suptitle(f'Decoder {di}')




    ### plot distributions for individual decoders (the 8 decoders)
    '''
    ifr = 4 #6
    for di in range(8):
        plt.figure(figsize = (10,6))
        icre = 0
        for cren in ['Slc17a7-IRES2-Cre', 'Sst-IRES-Cre', 'Vip-IRES-Cre']:
            icre = icre+1
            bb = all_sess[all_sess['cre']==cren]; 
            c = np.concatenate((bb['av_w_data'].values), axis=(1)) # (8, 730, 13)
            cc = c[:,:,ifr] # frame after image onset
    #         cc = c[:,:,6] - c[:,:,4]

            a = cc.flatten() # pool all the 8 decoders
    #         a = np.nanmean(cc, axis=0); # get average of the 8 decoders
            a = cc[di] # only take one decoder

            print(bb.shape, a.shape)

            # plot weights
            plt.subplot(2, 3, icre)    
            plt.hist(a, 50) #, density=True); 
            plt.xlabel('svm weights'); plt.ylabel('number of neurons'); 
            plt.title(f'{cren[:3]}, mean={np.nanmean(a):.4f}')    

            # plot absolute weights
            plt.subplot(2, 3, icre+3)
            plt.hist(np.abs(a), 50) #, density=True); 
            plt.xlabel('svm absolute weights'); plt.ylabel('number of neurons'); 
            plt.title(f'{cren[:3]}, mean={np.nanmean(np.abs(a)):.4f}')

            plt.subplots_adjust(wspace=0.5, hspace=0.5)

        plt.suptitle(f'frame {ifr}, 8 decoders pooled')
        plt.suptitle(f'Average of 8 decoders')
        plt.suptitle(f'Decoder {di}')
    '''    

    
    

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################


else: # pooling data across multiple project codes    
    
    svm_allMice_sessPooled_allprojects = []
    svm_allMice_sessAvSd_allprojects = []
    
    for project_codes in project_codes_all:
        
        print(f'\nRunning project {project_codes}\n')
        
        #%% Set frames_svm, samps_bef and samps_aft
        # Note: trials df has a much longer time_trace (goes up to 4.97) compared to stimulus df (goes up to .71), so frames_svm ends up being 1 element longer for trials df (ie when decoding hits from misses) compared to stimulus df (ie when decoding the images)    
        if project_codes != ['VisualBehaviorMultiscope']:
            frames_svm = np.arange(-15,23)
            num_planes = 1
            num_depth = 1
        else:
            frames_svm = np.arange(-5,8)
            num_planes = 8
            num_depth = 4

        if trial_type=='hits_vs_misses': # frames_svm[-1] = frames_svm[-1] + 1
            frames_svm = np.concatenate((frames_svm, [frames_svm[-1] + 1]))

        frames_svm = np.array(frames_svm)

        samps_bef = -frames_svm[0] # np.argwhere(trace_time==0)[0][0] # 5
        samps_aft = len(frames_svm)-samps_bef #8 
        print(samps_bef, samps_aft)


        #%%    
        cols_each = colorOrder(num_planes)


        
        #%% Set svm vars for each plane across all sessions (for each mouse)
        if project_codes != ['VisualBehaviorMultiscope']: # remove area/layer pooled columns
            columns0 = ['mouse_id', 'cre', 'mouse_id_exp', 'cre_exp', 'block', 'session_ids', 'experience_levels', 'session_stages', 'session_labs', 'num_sessions_valid', 'area_this_plane_allsess_allp', 'depth_this_plane_allsess_allp', \
                        'area', 'depth', 'plane', \
                        'n_neurons_this_plane_allsess_allp', 'n_trials_this_plane_allsess_allp', \
                        'av_meanX_avSess_eachP', 'sd_meanX_avSess_eachP', \
                        'av_test_data_this_plane_allsess_allp', 'av_test_shfl_this_plane_allsess_allp', \
                        'av_test_data_avSess_eachP', 'sd_test_data_avSess_eachP', 'av_test_shfl_avSess_eachP', 'sd_test_shfl_avSess_eachP', \
                        'av_train_data_avSess_eachP', 'sd_train_data_avSess_eachP', \
                        'av_n_neurons_avSess_eachP', 'sd_n_neurons_avSess_eachP', 'av_n_trials_avSess_eachP', 'sd_n_trials_avSess_eachP', \
                        'distinct_areas', \
                        'peak_amp_trTsShCh_this_plane_allsess_allp', \
                        'av_peak_amp_trTsShCh_avSess_eachP', 'sd_peak_amp_trTsShCh_avSess_eachP', \
                        ]    

        else:
            columns0 = ['mouse_id', 'cre', 'mouse_id_exp', 'cre_exp', 'block', 'session_ids', 'experience_levels', 'session_stages', 'session_labs', 'num_sessions_valid', 'area_this_plane_allsess_allp', 'depth_this_plane_allsess_allp', \
                        'area', 'depth', 'plane', \
                        'n_neurons_this_plane_allsess_allp', 'n_trials_this_plane_allsess_allp', \
                        'av_meanX_avSess_eachP', 'sd_meanX_avSess_eachP', \
                        'av_test_data_this_plane_allsess_allp', 'av_test_shfl_this_plane_allsess_allp', \
                        'av_test_data_avSess_eachP', 'sd_test_data_avSess_eachP', 'av_test_shfl_avSess_eachP', 'sd_test_shfl_avSess_eachP', \
                        'av_train_data_avSess_eachP', 'sd_train_data_avSess_eachP', \
                        'av_n_neurons_avSess_eachP', 'sd_n_neurons_avSess_eachP', 'av_n_trials_avSess_eachP', 'sd_n_trials_avSess_eachP', \
                        'av_test_data_pooled_sesss_planes_eachArea', 'av_test_shfl_pooled_sesss_planes_eachArea', 'meanX_pooled_sesss_planes_eachArea', \
                        'num_sessions_valid_eachArea', 'distinct_areas', 'cre_pooled_sesss_planes_eachArea', \
                        'av_test_data_pooled_sesss_areas_eachDepth', 'av_test_shfl_pooled_sesss_areas_eachDepth', 'meanX_pooled_sesss_areas_eachDepth', \
                        'cre_pooled_sesss_areas_eachDepth', 'depth_pooled_sesss_areas_eachDepth', \
                        'peak_amp_trTsShCh_this_plane_allsess_allp', \
                        'av_peak_amp_trTsShCh_avSess_eachP', 'sd_peak_amp_trTsShCh_avSess_eachP', \
                        'peak_amp_trTsShCh_pooled_sesss_planes_eachArea', \
                        'peak_amp_trTsShCh_pooled_sesss_areas_eachDepth', \
                        ]

        if same_num_neuron_all_planes:
            columns = np.concatenate((columns0, ['n_neurons_svm_trained_this_plane_allsess_allp', 'av_n_neurons_svm_trained_avSess_eachP', 'sd_n_neurons_svm_trained_avSess_eachP'])) 
        else:
            columns = columns0        
        
        #####################################
        #%% Set svm_this_plane_allsess
        #####################################

        exec(open('svm_images_plots_setVars2.py').read())     
        
        svm_this_plane_allsess0 = copy.deepcopy(svm_this_plane_allsess)

        
        ######### set svm_allMice_sessPooled and svm_allMice_sessAvSd
        # note: svm_allMice_sessPooled will be used to set summary_vars_all, which is a key paramter in svm_images_plots_compare_ophys_stages.py)
        exec(open('svm_images_plots_setVars_sumMice.py').read()) 

        
        svm_allMice_sessPooled_allprojects.append(svm_allMice_sessPooled)
        svm_allMice_sessAvSd_allprojects.append(svm_allMice_sessAvSd)


    #####################################
    #%% Pool data from both project codes
    #####################################

    # svm_allMice_sessPooled0_vb.iloc[0]['av_test_shfl_allPlanes'].shape
    # (1, 34, 38)
    # svm_allMice_sessPooled0_vbm.iloc[0]['av_test_shfl_allPlanes'].shape
    # (8, 21, 13)

    # svm_allMice_sessPooled0_vb.iloc[0]['peak_amp_allPlanes'].shape
    # (1, 34, 4)
    # svm_allMice_sessPooled0_vbm.iloc[0]['peak_amp_allPlanes'].shape
    # (8, 21, 4)



    #%% turn the vars into a single long vector, independent of the area/depth

    # for each project code: concatenate data from all planes and sessions    
    pa_allpr = []
    for ipc in range(len(project_codes_all)):
        
#         svm_allMice_sessPooled_allprojects[ipc].iloc[0]['peak_amp_allPlanes'].shape
        pa_now = np.vstack(svm_allMice_sessPooled_allprojects[ipc].iloc[0]['peak_amp_allPlanes'])
        pa_now.shape

        pa_allpr.append(pa_now)
        

    # now concatenate data from all projects
    pa_pooled_projects = np.concatenate((pa_allpr), axis=0) # pooled_session_planes_projects x 4
    pa_pooled_projects.shape

    
    
    ######################################################
    #%% set vars and plot traces and quantifications for each ophys stage
    
    # for each project code: set vars to make mouse-averaged plots
    summary_vars_allpr = []
    for ipc in range(len(project_codes_all)):
        
        svm_allMice_sessPooled0 = svm_allMice_sessPooled_allprojects[ipc]
        svm_allMice_sessAvSd0 = svm_allMice_sessAvSd_allprojects[ipc]

        project_codes = project_codes_all[ipc]
        
        # sets summary_vars_all, a dataframe that includes response amplitude (computed from svm_allMice_sessPooled); it will be used in svm_images_plots_compare_ophys_stages.py
        # if len(project_codes_all)==1, it calls svm_images_plots_sumMice.py to make mouse-averaged plots for each ophys stage.        
        exec(open('svm_images_plots_setVars_sumMice2.py').read()) 

        summary_vars_allpr.append(summary_vars_all)
        
        
    # make mouse-averaged plots
#     exec(open('svm_images_plots_sumMice.py').read()) 



    ################## make svm_df which is a proper pandas table including svm response amplitude results for each experiment ##################
    # svm_df will be used to make summary plots across experience levels
    
    exec(open('svm_images_plots_setVars_sumMice3_svmdf.py').read())

              
    ################## compare quantifications across ophys experience levels ##################
    
    exec(open('svm_images_plots_compare_ophys_experience_levels.py').read())
    

    ################## compare quantifications across ophys stages ##################
    
    # pool summary_vars_all of the 2 projects
    
    summary_vars_all = pd.concat((summary_vars_allpr))
    summary_vars_all.shape
    
    exec(open('svm_images_plots_compare_ophys_stages.py').read())
    
    
    
    
    