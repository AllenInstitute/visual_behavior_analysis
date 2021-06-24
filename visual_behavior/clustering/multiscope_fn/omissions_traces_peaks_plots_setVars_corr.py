#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run "omissions_traces_peaks_plots_setVars.py" to set the variable "all_sess_2an" needed here.

Follow this code by "omissions_traces_peaks_plots_setVars_corr_eachCre.py"

This script contains all the functions (plotting, etc), and also
sets dataframe "corr_trace_peak_allMice", which includes corr and p traces and their quantification for each layer pair, for each mouse:
    # cc traces (session-averaged) for each layer pair: eg. cc12_sessAv_44
    # cc quantification (session-averaged) of peaks (after omission and flash) for each layer pair: eg. cc12_peak_amp_omit_sessAv
    
cc traces and cc quantification are done in three different ways:
    # cc12_sessAv_44, cc12_peak_amp_omit_sessAv: corr between each layer and another layer (size: 4x4)
    # cc12_pooled_sessAv, cc12_peak_amp_omit_sessAv_pooled: corr between each layer and neurons in all other layers  (size: 4)    
    # cc12_pooledAll_sessAv, cc12_peak_amp_omit_sessAv_pooledAll: corr between all neuron pairs (regardless of their layer) (size: 1)


Remember: baselines are not subtracted from the traces; but when we quantify the peak we subtract the baseline if sameBl_allLayerPairs is set to 1. Remember: baselines are not subtracted from the traces; but when we quantify the peak we subtract the baseline if sameBl_allLayerPairs is set to 1.


Created on Mon Sep 23 15:58:29 2019
@author: farzaneh
"""

# from omissions_traces_peaks_plots_setVars_corrs_funs import *

#%%
sameBl_allLayerPairs = 1 #1 # if 1, when quantifying cc, shift the baseline of cc trace of all 16 area1-area2 layer combinations so their baseline is at 0. Remember: baselines are not subtracted from the traces; but when we quantify the peak we subtract the baseline if sameBl_allLayerPairs is set to 1. # note: it makese sense to set it to 1 bc there is this difference in baseline of correlation plots which i dont understand why.
use_spearman_p = 1 # if 1, we use spearman p to show fraction of significant neuron pairs, if 0, we use the manually computed p (one-sample ttest between cc_shlf distribution and the actual cc) to quantify fraction of neuron pairs with significant cc values.
do_single_mouse_plots = 0 #1 # make cc traces and peaks plots of session-averaged data, for each mouse

peak_win = [0, .75] #[0, .5] #[0, .75] # this should be named omit_win
flash_win = np.array([0, .5])-.75 #np.array([0, .35])-.75 #[0, .75] # now using flash-aligned traces; previously flash responses were computed on omission-aligned traces; problem: not all flashes happen every .75sec # [-.75, 0] # previous value: # [-.75, -.25] # 
flash_win_vip = flash_win-.25 #[-.25, .5] # on flash-aligned traces # [-1, -.25] # previous value (for flash_win of all cre lines): # [-.75, -.25] # 

'''
######## NOTE: in the new method (implemented 04/29/2020) we compute list_times from peak_win (for omission responses); then we use flash-omit interval (flash_omit_dur_fr_all) to compute list_times_flash from list_times; Then we use flash_win_vip_shift to compute list_times_flash_vip from list_times_flash.
### although, it really didn't make a difference really from when you defined flash_win as peak_win -.75 ... because fo_dur_fr_med is 8 frames which is about 750ms.
# we set these vars in omissions_traces_peaks_init.py, but we dont used them until in this script. se we can just redifine them here.
peak_win = [0, .6] # [0, .75] # you should totally not go beyond 0.75!!! # go with 0.75ms if you don't want the responses to be contaminated by the next flash
flash_win_vip_shift = .25 # sec; shift flash_win this much earlier to compute flash responses on vip B1 sessions.
# flash_win = [-.75, -.15] # [-.75, -.25] #[-.75, -.4] #[-.75, 0] # window (relative to omission) for computing flash-evoked responses (750ms includes flash and gray)    
# flash_win_vip = [-1, -.4] # [-1, -.25] # previous value (for flash_win of all cre lines): # [-.75, -.25] # 
'''

# fw0 = (peak_win[0] - (fo_dur_fr_med * frame_dur)).squeeze()
# flash_win = [fw0, fw0 + np.diff(peak_win).squeeze()]

alpha_sig = .05 # significance level when computing fraction of pairs with significant corrcoeff
# bl_percentile = 10 #20  # for peak measurements, we subtract pre-omit baseline from the peak. bl_percentile determines what pre-omit values will be used.      

#%%
xnow = time_trace # time_trace_new


#%% set xlim
#### NOTE: when we subtract signal (to compute noise corrs), we do it for 1 image before and 1 image after omission, so it makes sense to choose xlim such that it only covers this range and not beyond.
#xlim = [-1.2, 1.5]
#xlim = [-1.2, 2.25] # [-13, 24]
frame_dur_new = np.diff(xnow)[0]
r = np.round(np.array(xlim) / frame_dur_new).astype(int)
r[0] = np.max([-samps_bef, r[0]])
r[1] = np.min([samps_aft-1, r[1]])
xlim_frs = np.arange(int(np.argwhere(xnow==0).squeeze()) + r[0], int(np.argwhere(xnow==0).squeeze()) + r[1] + 1)
    

jet_cm = ['b', 'c', 'g', 'r'] # colorOrder(nlines=num_depth)


#%%
num_frs = samps_bef + samps_aft

fo_dur_fr_med = np.nanmedian(np.hstack(all_sess_2an['flash_omit_dur_fr_all'].values)).astype(int)
# flash_win_vip_shift = .25 # sec; shift flash_win this much earlier to compute flash responses on vip B1 sessions.

'''
######## NOTE: in the new method (implemented 04/29/2020) we compute list_times from peak_win (for omission responses); then we use flash-omit interval (flash_omit_dur_fr_all) to compute list_times_flash from list_times; Then we use flash_win_vip_shift to compute list_times_flash_vip from list_times_flash.
### although, it really didn't make a difference really from when you defined flash_win as peak_win -.75 ... because fo_dur_fr_med is 8 frames which is about 750ms.
# set time windows, in frame units, for computing flash and omission evoked responese
# list_times, list_times_flash, list_times_flash_vip = set_frame_window_flash_omit(peak_win, flash_win, flash_win_vip, samps_bef, frame_dur)
list_times, _, _ = set_frame_window_flash_omit(peak_win, np.nan, np.nan, samps_bef, frame_dur)
list_times_flash = list_times - fo_dur_fr_med
list_times_flash_vip = list_times_flash - np.round(flash_win_vip_shift/frame_dur).astype(int) # we shift the window 250ms earlier to capture vip responses
'''

# 08/04/2020: going back to using set_frame_window_flash_omit to set the list_times, because I want to use different time windows for omissions and images.
list_times, list_times_flash, list_times_flash_vip = set_frame_window_flash_omit(peak_win, flash_win, flash_win_vip, samps_bef, frame_dur)


    
    
    
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
"""
Defines functions needed in omissions_traces_peaks_plots_setVars_corrs.py and other correlation plotting scripts.
"""
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from def_funs import * 

plt.rcParams["axes.grid"] = False

#%% Define flash_win_final (depending on cre and sessions stage)

def set_flash_win_final(cre, session_novel, flash_win, flash_win_vip):
    
    if np.logical_and(cre.find('Vip')==0 , session_novel) or cre.find('Vip')==-1: # VIP B1, SST, SLC: window after images (response follows the image)
        flash_win_final = flash_win # [0, .75]

    else: # VIP (non B1,familiar sessions): window precedes the image (pre-stimulus ramping activity)
        flash_win_final = flash_win_vip # [-.25, .5] 
        

    return flash_win_final




#%% Quantify corr after flash, and after omission (to get the heatmap of corrs!)

def quant_cc(cc_sessAv, fo_dur, list_times, list_times_flash, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs): # cc_sessAv = cc11_sessAv

    if sameBl_allLayerPairs:
        # Note: instead of defining a single flash_index for all sessions, I would use flash_omit_dur_fr_all (which now we are saving in this_sess (and all_sess)) for each session individually.
#         mxgp = np.max(fo_dur) # .75 #

        if np.isnan(fo_dur).all():
            mxgp = .77
        else:
            if np.max(fo_dur)>.77:
                print(f'\n\nWARNING: {sum(fo_dur>.77)} flash-omission intervals are above .77sec; max = {np.max(fo_dur)} sec \n\n')
            mxgp = np.max(fo_dur[fo_dur<=.77]) # i had to comment above because in mouse_id: 440631, session_id: 849304162, we have a flash that is about 5sec apart from the omission! remove this!

        '''
        # on 5/11/2020 I changed np.floor (below) to np.round, to make things consistent with set_frame_window_flash_omit... also i think it is more accurate to use round than floor!
        flash_index = samps_bef - np.round(mxgp / frame_dur).astype(int) # not quite accurate, if you want to be quite accurate you should align the traces on flashes!
        bl_index_pre_flash = np.arange(0,flash_index)

        bl_index_pre_omit = np.arange(0,samps_bef) # you will need for response amplitude quantification, even if you dont use it below.
        
        bl_preOmit = np.percentile(cc_sessAv[bl_index_pre_omit], bl_percentile, axis=0) # 16 (layer combinations of the two areas)
        bl_preFlash = np.percentile(cc_sessAv[bl_index_pre_flash], bl_percentile, axis=0) # 16 (layer combinations of the two areas)
        '''
        
        # average all values below the 50th percentile of the entire trace
        '''
        th = np.percentile(cc_sessAv, 50, axis=0)
        if np.ndim(cc_sessAv)>1:
            bl_preOmit = [np.mean(cc_sessAv[cc_sessAv[:, l] <= th[l] , l]) for l in range(len(th))]
        else:
            bl_preOmit = np.mean(cc_sessAv[cc_sessAv <= th])
        bl_preFlash = bl_preOmit
        '''
        
        # newest method: take frames right before the flash onset (or shifted by .25sec in case of VIP, familiar) and average them
        
        b0_relOmit = np.round((flash_win_final0) / frame_dur).astype(int)
        stp = np.round(-.75 / frame_dur).astype(int)
        # below is better: image and omission frames than the commented one down here, which is image-1 and omission-1.
        bl_index_pre_omit = np.sort(np.arange(samps_bef + b0_relOmit , 0 , stp))
#         bl_index_pre_omit = np.sort(np.arange(samps_bef-1 + b0_relOmit , 0 , stp))
#         bl_index_pre_omit

        bl_preOmit = np.mean(cc_sessAv[bl_index_pre_omit], axis=0) # 16
        bl_preFlash = bl_preOmit
        
        
        
    # compute mean during peak_win to quantify cc after omission, for each layer pair
    peak_amp_omit = np.nanmean(cc_sessAv[list_times], axis=0) # 16        
    peak_amp_omit0 = peak_amp_omit + 0
    if sameBl_allLayerPairs: 
        peak_amp_omit = peak_amp_omit0 - bl_preOmit

    # compute mean during flash_win to quantify cc after flash, for each layer pair
    peak_amp_flash = np.nanmean(cc_sessAv[list_times_flash], axis=0) # 16
    peak_amp_flash0 = peak_amp_flash + 0
    if sameBl_allLayerPairs:
        peak_amp_flash = peak_amp_flash0 - bl_preFlash


    ##################################################    
    # now reshape to 4x4 (area by area matrix of cc)
    if np.ndim(cc_sessAv)==1:
        peak_amp_omit44 = peak_amp_omit
        peak_amp_flash44 = peak_amp_flash
    
    else:
        if cc_sessAv.shape[1]==16: # across area correlations
            peak_amp_omit44 = np.reshape(peak_amp_omit, (num_depth, num_depth), order='C') # 4 x 4
            peak_amp_flash44 = np.reshape(peak_amp_flash, (num_depth, num_depth), order='C') # 4 x 4
            
        elif cc_sessAv.shape[1]==10: # within area correlations; # create a nan 4x4 matrix and fill in the upper triangular with the values computed above.
            indsr, indsc = np.triu_indices(num_depth)
    
            a = np.full((num_depth, num_depth), np.nan)
            a[indsr,indsc] = peak_amp_omit
            peak_amp_omit44 = a
    
            a = np.full((num_depth, num_depth), np.nan)
            a[indsr,indsc] = peak_amp_flash
            peak_amp_flash44 = a
            
        else:
            peak_amp_omit44 = peak_amp_omit
            peak_amp_flash44 = peak_amp_flash

    
    return peak_amp_omit44, peak_amp_flash44


# column orders in cc and p dataframes:
    # each column is corrcoeff between a layer of area1 and and a layer of area2; 
    # area1: LM; area 2: V1
    # eg. 'l32': corrcoeff between layer3, LM and layer2, V1.
    # rows are area1 (LM) layers; colums are area2 (V1) layers

# in cols_a1a2: rows are area1 (LM) layers; colums are area2 (V1) layers
# cols = cc_a1_a2.columns.values
# cols_a1a2 = np.reshape(cols, (4,4), order='C')
#            cols_a1a2 = [['l00', 'l01', 'l02', 'l03'],
#                        ['l10', 'l11', 'l12', 'l13'],
#                        ['l20', 'l21', 'l22', 'l23'],
#                        ['l30', 'l31', 'l32', 'l33']]


#%% Reshape corr traces across layers within the same area to 4x4 matrices
# input: cc11_sessAv: 80 x 10
# output: cc11_sessAv_44: 80 x 4 x 4

def traces_44_layerSame(cc11_sessAv, num_depth):
    indsr, indsc = np.triu_indices(num_depth)
    a = np.full((cc11_sessAv.shape[0], num_depth, num_depth), np.nan)
    a[:,indsr,indsc] = cc11_sessAv
    cc11_sessAv_44 = a            
    # sanity check:
#        plt.plot(a[:,1,2]); plt.plot(cc11_sessAv[:,5])
    
    return cc11_sessAv_44



#%% Function to plot corr traces: 4 subplots for each V1 layer. Each subplot shows cc between a given V1 layer and all LM layers.    
# plots are made for cc, cc_shf, p 
    
def plot_traces_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, fgn, areas, depths, title, xlim_frs, jet_cm, alph, mouseid=np.nan, dosavefig=0):
        
#     flash_win_final = set_flash_win_final(cre, session_novel, flash_win, flash_win_vip)
    
    num_depth = int(len(depths)/len(areas))
    
    # each layer of V1 with all layers of LM
    plt.figure(figsize=(10,11))    
#    title = '%s (%d mice)' %(cre, thisCre_numMice)    
    st = '%s - %s ' %(areas[1], areas[0]) # '%s each layer- %s all layers' %(areas[0], areas[1]
    plt.suptitle('%s\n%s' %(title,st), y=1, fontsize=18)
    
    # subplots of cc
    gs = gridspec.GridSpec(num_depth,1) #, width_ratios=[3, 1]) 
    gs.update(left=0.02, right=0.22, hspace=.6) #, hspace=1)
    # subplots of cc_shfl
    gs2 = gridspec.GridSpec(num_depth,1)
    gs2.update(left=0.48, right=0.68, hspace=0.6) #, hspace=1)
    # subplots of p
    gs3 = gridspec.GridSpec(num_depth,1)
    gs3.update(left=0.78, right=0.98, hspace=0.6) #, hspace=1)

    
    
    ########## gs: cc ##########   
    
    yl = 'Spearman cc' #'CC'
    lims0 = [np.nanmin(top0[xlim_frs] - top_sd0[xlim_frs]) , np.nanmax(top0[xlim_frs] + top_sd0[xlim_frs])]
    
    for i2 in range(num_depth): # loop through V1 layers (subplots)
#        plt.subplot(2,2,i2+1)
        ax1 = plt.subplot(gs[i2,0])        
        h = []
        for i1 in range(num_depth): # loop through LM layers (curves on each subplot)
            h0 = plt.plot(time_trace, top0[:,i1,i2], color=jet_cm[i1], label='%s %dum' %(areas[0], depths[i1]))[0]
            plt.fill_between(time_trace, top0[:,i1,i2] - top_sd0[:,i1,i2], top0[:,i1,i2] + top_sd0[:,i1,i2], alpha=alph,  edgecolor=jet_cm[i1], facecolor=jet_cm[i1])
            h.append(h0)
            plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
#            if i2==0:
#            plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, fontsize=12, handlelength=1)
            
            if i1==(num_depth-1):
                plot_flashLines_ticks_legend(lims0, h, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab=yl, xmjn=xmjn)
            if i2!=(num_depth-1):
                plt.xlabel('')
#        plot_flashLines_ticks_legend(lims0, h, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab='CC', xmjn=xmjn)                
        plt.xlim(xlim)        
        if i2==0: # mark peak_win: the window over which the response quantification (peak or mean) was computed 
            plt.hlines(lims0[1], peak_win[0], peak_win[1], color='gray')
            # excluding below because depending on the cre and session type flash quantification varies; if all_ABtransit_AbefB_Aall=2 or 3, then it is all A, but with other values of all_ABtransit_AbefB_Aall, it is not possible to have one single mark for flash quantification as it will be a mixture of A and B sessions.
#             plt.hlines(lims0[1], flash_win_final[0], flash_win_final[1], color='green')



    ########## gs2: cc_shfl ##########    

    yl = 'Spearman cc_shfl' #'CC_shfl'
    lims0 = [np.nanmin(top1[xlim_frs] - top_sd1[xlim_frs]) , np.nanmax(top1[xlim_frs] + top_sd1[xlim_frs])]
 
    for i2 in range(num_depth): # loop through V1 layers
#        plt.subplot(2,2,i2+1)
        ax1 = plt.subplot(gs2[i2,0])        
        h = []
        for i1 in range(num_depth): # loop through LM layers
            h0 = plt.plot(time_trace, top1[:,i1,i2], color=jet_cm[i1], label='%s %dum' %(areas[0], depths[i1]))[0]
            plt.fill_between(time_trace, top1[:,i1,i2] - top_sd1[:,i1,i2], top1[:,i1,i2] + top_sd1[:,i1,i2], alpha=alph,  edgecolor=jet_cm[i1], facecolor=jet_cm[i1])
            h.append(h0)
            plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
#            if i2==0:
#            plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, fontsize=12, handlelength=1)
            
            if i1==(num_depth-1):
                plot_flashLines_ticks_legend(lims0, [], flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab=yl, xmjn=xmjn)
            if i2!=(num_depth-1):
                plt.xlabel('')
        plt.xlim(xlim)

 
    
    ########## gs3: p ##########

    lims0 = [np.nanmin(top2[xlim_frs] - top_sd2[xlim_frs]) , np.nanmax(top2[xlim_frs] + top_sd2[xlim_frs])]
 
    for i2 in range(num_depth): # loop through V1 layers
        ax1 = plt.subplot(gs3[i2,0])        
        h = []
        for i1 in range(num_depth): # loop through LM layers
            h0 = plt.plot(time_trace, top2[:,i1,i2], color=jet_cm[i1], label='%s %dum' %(areas[0], depths[i1]))[0]
            plt.fill_between(time_trace, top2[:,i1,i2] - top_sd2[:,i1,i2], top2[:,i1,i2] + top_sd2[:,i1,i2], alpha=alph,  edgecolor=jet_cm[i1], facecolor=jet_cm[i1])
            h.append(h0)
            plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
#            if i2==0:
#            plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, fontsize=12, handlelength=1)
            
            if i1==(num_depth-1):
                plot_flashLines_ticks_legend(lims0, h, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab='Fract sig neuron pairs', xmjn=xmjn)
            if i2!=(num_depth-1):
                plt.xlabel('')
        plt.xlim(xlim)    


    #############################
#    dir_now = 'corr_omit_flash'        
    if dosavefig:
        if np.isnan(mouseid): # plotting average across mice
            nam = '%s_aveMice_%s%s_traces_%s' %(cre, fgn, whatSess, now)        
        else: # mouse_id is provided; plotting single mouse data
            nam = '%s_mouse%d_aveSess_%s%s_traces_%s' %(cre, mouseid, fgn, whatSess, now)        
        fign = os.path.join(dir0, dir_now, nam+fmt)        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



#%% Function to plot omit/ flash peak quantification: 4 subplots for each V1 layer. Each subplot shows cc quantification between a given V1 layer and all LM layers.    
# plots are made for flash and omit evoked responses.
# plots are made for cc, cc_shf, p 

def plot_peak_cc_ccShfl_p(f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, fgn, areas, depths, title, jet_cm, alph, mouseid=np.nan, dosavefig=0):
          
    num_depth = int(len(depth_sessAv_allMice_thisCre[icn]) / len(areas))
    
    # each layer of V1 with all layers of LM
    plt.figure(figsize=(8,11))
#    title = '%s (%d mice)' %(cre, thisCre_numMice)
    st = '%s - %s ' %(areas[1], areas[0]) # '%s each layer- %s all layers' %(areas[0], areas[1]
    plt.suptitle('%s\n%s' %(title,st), y=1.1, fontsize=18)
    
    # subplots of cc
    gs = gridspec.GridSpec(2,num_depth) #, width_ratios=[3, 1]) 
#     gs.update(left=0.0, right=0.25, hspace=.6, wspace=1)
    gs.update(top=1, bottom=0.75, hspace=1, wspace=1)
    # subplots of cc_shfl
    gs2 = gridspec.GridSpec(2,num_depth)
#     gs2.update(left=0.38, right=0.63, hspace=0.6, wspace=1)
    gs2.update(top=0.62, bottom=0.37, hspace=1, wspace=1)
    # subplots of p
    gs3 = gridspec.GridSpec(2,num_depth)
#     gs3.update(left=0.75, right=1, hspace=0.6, wspace=1)
    gs3.update(top=0.25, bottom=0, hspace=1, wspace=1)

    x = np.arange(num_depth)
    
    
    ##################################
    ############# gs: cc #############   
    ##################################
    yl = 'cc amplitude' #'CC'
                
    lims0_f = np.squeeze([np.nanmin(f_top0 - f_top_sd0) , np.nanmax(f_top0 + f_top_sd0)])
    lims0_o = np.squeeze([np.nanmin(o_top0 - o_top_sd0) , np.nanmax(o_top0 + o_top_sd0)])
    lims0_f[0] = lims0_f[0] - np.diff(lims0_f) / 30.
    lims0_f[1] = lims0_f[1] + np.diff(lims0_f) / 30.
    lims0_o[0] = lims0_o[0] - np.diff(lims0_o) / 30.
    lims0_o[1] = lims0_o[1] + np.diff(lims0_o) / 30.      

    for i2 in range(num_depth): # loop through V1 layers

        ############ flash ############
#        plt.subplot(2,2,i2+1)
        ax1 = plt.subplot(gs[0,i2])        

        # plot all LM layers (and V1 layer i2)
        ax1.errorbar(x, f_top0[:,i2], yerr=f_top_sd0[:,i2], fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

#         plt.hlines(0, 0, len(x)-1, linestyle=':') # superimpose shuffle on actual data
        ax1.set_xticks(x)
        xticklabs = depths[:num_depth].astype(int) # '%s %dum' %(areas[0], depths[i1])
#         if i2==num_depth-1:
        plt.xlabel('%s depth (um)' %areas[0], fontsize=12)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
        if ~np.isnan(lims0_f).any():
            plt.ylim(lims0_f)
        if i2==0:
            plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
        ylim0 = lims0_f #plt.gca().get_ylim()
#         text_y = ylim0[1] + np.diff(ylim0)/3
        text_y = .5 #ylim0[0] + np.diff(ylim0)/3.3
        if i2==0:
#             plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#             plt.text(-3.8, text_y, 'Flash', fontsize=15, rotation= 'vertical') # 3.2
            plt.text(-1, text_y, 'Image', fontsize=15, rotation='vertical', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


        ############ omission ############
#        plt.subplot(2,2,i2+1)
        ax1 = plt.subplot(gs[1,i2])        

        # plot all LM layers (and V1 layer i2)
        ax1.errorbar(x, o_top0[:,i2], yerr=o_top_sd0[:,i2], fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

#         plt.hlines(0, 0, len(x)-1, linestyle=':') # superimpose shuffle on actual data
        ax1.set_xticks(x)
        xticklabs = depths[:num_depth].astype(int) # '%s %dum' %(areas[0], depths[i1])
#         if i2==num_depth-1:
        plt.xlabel('%s depth (um)' %areas[0], fontsize=12)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
        if ~np.isnan(lims0_o).any():
            plt.ylim(lims0_o)        
        if i2==0:    
            plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
        ylim0 = lims0_o #plt.gca().get_ylim()        
#         text_y = ylim0[1] + np.diff(ylim0)/3
        text_y = .5 #ylim0[0] + np.diff(ylim0)/4.5
        if i2==0:
#             plt.text(-.7, text_y, 'Omission', fontsize=15)
            plt.text(-1, text_y, 'Omission', fontsize=15, rotation='vertical', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)



    ##################################
    ########## gs2: cc_shfl ##########   
    ##################################    
    yl = 'cc_shfl amplitude' #'CC'
             
    lims0_f = np.squeeze([np.nanmin(f_top1 - f_top_sd1) , np.nanmax(f_top1 + f_top_sd1)])
    lims0_o = np.squeeze([np.nanmin(o_top1 - o_top_sd1) , np.nanmax(o_top1 + o_top_sd1)])

    # superimpose shuffle on actual data
    lims0_f = np.squeeze([np.nanmin(np.concatenate((f_top0 - f_top_sd0, f_top1 - f_top_sd1))) , np.nanmax(np.concatenate((f_top0 + f_top_sd0, f_top1 + f_top_sd1)))])
    lims0_o = np.squeeze([np.nanmin(np.concatenate((o_top0 - o_top_sd0, o_top1 - o_top_sd1))) , np.nanmax(np.concatenate((o_top0 + o_top_sd0, o_top1 + o_top_sd1)))])

    lims0_f[0] = lims0_f[0] - np.diff(lims0_f) / 30.
    lims0_f[1] = lims0_f[1] + np.diff(lims0_f) / 30.  
    lims0_o[0] = lims0_o[0] - np.diff(lims0_o) / 30.
    lims0_o[1] = lims0_o[1] + np.diff(lims0_o) / 30.      
    
    for i2 in range(num_depth): # loop through V1 layers

        ############ flash ############
#        plt.subplot(2,2,i2+1)
        ax1 = plt.subplot(gs2[0,i2])        
        ax1 = plt.subplot(gs[0,i2]) # superimpose shuffle on actual data
        
        # plot all LM layers (and V1 layer i2)
        ax1.errorbar(x, f_top1[:,i2], yerr=f_top_sd1[:,i2], fmt='o', markersize=3, capsize=3, color='gray') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

#         plt.hlines(0, 0, len(x)-1, linestyle=':')  # superimpose shuffle on actual data
        ax1.set_xticks(x)
        xticklabs = depths[:num_depth].astype(int) # '%s %dum' %(areas[0], depths[i1])
#         if i2==num_depth-1:
        plt.xlabel('%s depth (um)' %areas[0], fontsize=12)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)     
        if ~np.isnan(lims0_f).any():
            plt.ylim(lims0_f)
#         if i2==0:    
#             plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
        ylim0 = lims0_f #plt.gca().get_ylim()
#         text_y = ylim0[1] + np.diff(ylim0)/3
        text_y = .5 #ylim0[0] + np.diff(ylim0)/3.3
        if i2==0:
#             plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#             plt.text(-4.8, text_y, 'Flash', fontsize=15, rotation='vertical')
            plt.text(-1, text_y, 'Image', fontsize=15, rotation='vertical', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


        ############ omission ############
#        plt.subplot(2,2,i2+1)
        ax1 = plt.subplot(gs2[1,i2])        
        ax1 = plt.subplot(gs[1,i2])  # superimpose shuffle on actual data

        # plot all LM layers (and V1 layer i2)
        ax1.errorbar(x, o_top1[:,i2], yerr=o_top_sd1[:,i2], fmt='o', markersize=3, capsize=3, color='gray') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

#         plt.hlines(0, 0, len(x)-1, linestyle=':')  # superimpose shuffle on actual data
        ax1.set_xticks(x)
        xticklabs = depths[:num_depth].astype(int) # '%s %dum' %(areas[0], depths[i1])
#         if i2==num_depth-1:
        plt.xlabel('%s depth (um)' %areas[0], fontsize=12)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)      
        if ~np.isnan(lims0_o).any():
            plt.ylim(lims0_o)        
#         if i2==0:    
#             plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
        ylim0 = lims0_o #plt.gca().get_ylim()        
#         text_y = ylim0[1] + np.diff(ylim0)/3
        text_y = .5 #ylim0[0] + np.diff(ylim0)/4.5
        if i2==0:
#             plt.text(-.7, text_y, 'Omission', fontsize=15)
#             plt.text(-4.8, text_y, 'Omission', fontsize=15, rotation='vertical')
            plt.text(-1, text_y, 'Omission', fontsize=15, rotation='vertical', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)



    ##################################
    ########## gs3: p ##########   
    ##################################    
    yl = 'Fract sig neur pairs' #'CC'
                
    lims0_f = np.squeeze([np.nanmin(f_top2 - f_top_sd2) , np.nanmax(f_top2 + f_top_sd2)])
    lims0_o = np.squeeze([np.nanmin(o_top2 - o_top_sd2) , np.nanmax(o_top2 + o_top_sd2)])
    lims0_f[0] = lims0_f[0] - np.diff(lims0_f) / 30.
    lims0_f[1] = lims0_f[1] + np.diff(lims0_f) / 30.  
    lims0_o[0] = lims0_o[0] - np.diff(lims0_o) / 30.
    lims0_o[1] = lims0_o[1] + np.diff(lims0_o) / 30.      
    
    for i2 in range(num_depth): # loop through V1 layers

        ############ flash ############
#        plt.subplot(2,2,i2+1)
        ax1 = plt.subplot(gs3[0,i2])        

        # plot all LM layers (and V1 layer i2)
        ax1.errorbar(x, f_top2[:,i2], yerr=f_top_sd2[:,i2], fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

        plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        xticklabs = depths[:num_depth].astype(int) # '%s %dum' %(areas[0], depths[i1])
#         if i2==num_depth-1:
        plt.xlabel('%s depth (um)' %areas[0], fontsize=12)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)          
        if ~np.isnan(lims0_f).any():
            plt.ylim(lims0_f)
        if i2==0:    
            plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
        ylim0 = lims0_f #plt.gca().get_ylim()
#         text_y = ylim0[0] + np.diff(ylim0)/3
        text_y = .5 #ylim0[0] + np.diff(ylim0)/3.3
        if i2==0:
#             plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#             plt.text(-4.8, text_y, 'Flash', fontsize=15, rotation='vertical')
            plt.text(-1, text_y, 'Image', fontsize=15, rotation='vertical', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


        ############ omission ############
#        plt.subplot(2,2,i2+1)
        ax1 = plt.subplot(gs3[1,i2])        

        # plot all LM layers (and V1 layer i2)
        ax1.errorbar(x, o_top2[:,i2], yerr=o_top_sd2[:,i2], fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

        plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        xticklabs = depths[:num_depth].astype(int) # '%s %dum' %(areas[0], depths[i1])
#         if i2==num_depth-1:
        plt.xlabel('%s depth (um)' %areas[0], fontsize=12)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
        if ~np.isnan(lims0_o).any():
            plt.ylim(lims0_o)       
        if i2==0:    
            plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
        ylim0 = lims0_o #plt.gca().get_ylim()        
#         text_y = ylim0[1] + np.diff(ylim0)/3
        text_y = .5 #ylim0[0] + np.diff(ylim0)/4.5
        if i2==0:
#             plt.text(-.7, text_y, 'Omission', fontsize=15)
#             plt.text(-4.8, text_y, 'Omission', fontsize=15, rotation='vertical')
            plt.text(-1, text_y, 'Omission', fontsize=15, rotation='vertical', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)

     #############################
#    dir_now = 'corr_omit_flash'        
    if dosavefig:
        if np.isnan(mouseid): # plotting average across mice
            nam = '%s_aveMice_%s%s_peak_%s' %(cre, fgn, whatSess, now)        
        else: # mouse_id is provided; plotting single mouse data
            nam = '%s_mouse%d_aveSess_%s%s_peak_%s' %(cre, mouseid, fgn, whatSess, now)        
        fign = os.path.join(dir0, dir_now, nam+fmt)        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



        
#%% Function to plot heatmaps for omit/ flash peak quantification: 4x4 heatmaps for cc quantification of each layer pair.
# plots are made for flash and omit evoked responses.
# plots are made for cc, cc_shf, p 

def plot_heatmap_peak_cc_ccShfl_p(f_top0, o_top0, f_top1, o_top1, f_top2, o_top2, fgn, areas, depths, title, mouseid=np.nan, dosavefig=0):
    
    num_depth = int(len(depths)/len(areas))
    
    def plot_colorbar_set_xy_ticks(ax1, im, lims0_fo, yl, depths, num_depth):
        plt.clim(lims0_fo[0], lims0_fo[1])
        cb = plt.colorbar(im, fraction=.046)#
        cb.ax.tick_params(labelsize=10) 
        cb.set_label(yl, labelpad=20, size=12)

        ax1.set_xticks(x)
        xticklabs = depths[num_depth:].astype(int) # x axis: v1
        ax1.set_xticklabels(xticklabs, rotation=45)
        plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
        plt.xlim([-.5, len(x)-.5])

        ax1.set_yticks(x)
        yticklabs = depths[:num_depth].astype(int) # y axis: lm
        ax1.set_yticklabels(yticklabs, rotation=45)
        plt.ylabel('%s depth (um)' %areas[0], fontsize=12)
        plt.ylim([len(x)-.5, -.5])
#         plt.ylim([-.5, len(x)-.5]) # this will reverse the y axis

        ax1.tick_params(labelsize=10)
        
        
    plt.figure(figsize=(8,11))
    #    title = '%s (%d mice)' %(cre, thisCre_numMice)
    st = '%s - %s ' %(areas[1], areas[0]) # '%s each layer- %s all layers' %(areas[0], areas[1]
    plt.suptitle('%s\n%s' %(title,st), y=1.1, fontsize=18)

    # subplots of cc
    gs = gridspec.GridSpec(1,2)
    gs.update(top=1, bottom=0.75, hspace=1, wspace=1)
    # subplots of cc_shfl
    gs2 = gridspec.GridSpec(1,2)
    gs2.update(top=0.67, bottom=0.42, hspace=1, wspace=1)
    # subplots of p
    gs3 = gridspec.GridSpec(1,2)
    gs3.update(top=0.34, bottom=.09, hspace=1, wspace=1)

    x = np.arange(num_depth)


    ##################################
    ########## gs: cc ##########   
    ##################################    

    yl = 'cc amplitude'
    lims0_fo = np.squeeze([np.nanmin([f_top0, o_top0]) , np.nanmax([f_top0, o_top0])])
#     lims0_fo = [-.01, .03] # [-.002, .006] # [-.01, .02] # # use a customized range of colors (colorbar axis)

    ###### Flash ######
    ax1 = plt.subplot(gs[0])    

    im = plt.imshow(f_top0);
    plt.title('Image', y=1, fontsize=13)
    plot_colorbar_set_xy_ticks(ax1, im, lims0_fo, yl, depths, num_depth)
    plt.rcParams["axes.grid"] = False
    

    ###### Omission ######
    ax1 = plt.subplot(gs[1])    

    im = plt.imshow(o_top0);
    plt.title('Omission', y=1, fontsize=13)
    plot_colorbar_set_xy_ticks(ax1, im, lims0_fo, yl, depths, num_depth)
    plt.rcParams["axes.grid"] = False
    


    ##################################
    ########## gs2: cc_shfl ##########   
    ##################################    

    yl = 'cc_shfl amplitude'
    lims0_fo = np.squeeze([np.nanmin([f_top1, o_top1]) , np.nanmax([f_top1, o_top1])])


    ###### Flash ######
    ax1 = plt.subplot(gs2[0])    
    plt.title('Image', y=1, fontsize=13)

    im = plt.imshow(f_top1);
    plot_colorbar_set_xy_ticks(ax1, im, lims0_fo, yl, depths, num_depth)
    

    ###### Omission ######
    ax1 = plt.subplot(gs2[1])    
    plt.title('Omission', y=1, fontsize=13)

    im = plt.imshow(o_top1);
    plot_colorbar_set_xy_ticks(ax1, im, lims0_fo, yl, depths, num_depth)

    
    
    ##################################
    ########## gs3: p ##########   
    ##################################    
    yl = 'Fract sig neur pairs'
    lims0_fo = np.squeeze([np.nanmin([f_top2, o_top2]) , np.nanmax([f_top2, o_top2])])


    ###### Flash ######
    ax1 = plt.subplot(gs3[0])    
    plt.title('Image', y=1, fontsize=13)

    im = plt.imshow(f_top2);
    plot_colorbar_set_xy_ticks(ax1, im, lims0_fo, yl, depths, num_depth)
    

    ###### Omission ######
    ax1 = plt.subplot(gs3[1])    
    plt.title('Omission', y=1, fontsize=13)

    im = plt.imshow(o_top2);
    plot_colorbar_set_xy_ticks(ax1, im, lims0_fo, yl, depths, num_depth)


    #############################
#     dir_now = 'corr_omit_flash'        
    if dosavefig:
        if np.isnan(mouseid): # plotting average across mice
            nam = '%s_aveMice_heatmap_%s%s_peak_%s' %(cre, fgn, whatSess, now)        
        else: # mouse_id is provided; plotting single mouse data
            nam = '%s_mouse%d_heatmap_aveSess_%s%s_peak_%s' %(cre, mouseid, fgn, whatSess, now)        
        fign = os.path.join(dir0, dir_now, nam+fmt)        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

        
        
        
        

#%% Function for pooled and pooledAll vars: plot corr traces and peaks 
# plots are made for cc, cc_shf, p 

def plot_traces_peak_pooled_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, top0a, top_sd0a, top1a, top_sd1a, top2a, top_sd2a, \
    f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, \
    f_top0a, f_top_sd0a, o_top0a, o_top_sd0a, f_top1a, f_top_sd1a, o_top1a, o_top_sd1a, f_top2a, f_top_sd2a, o_top2a, o_top_sd2a, \
    fgn, areas, depths, title, xlim_frs, jet_cm, alph, mouseid=np.nan, dosavefig=0):
      
#     flash_win_final = set_flash_win_final(cre, session_novel, flash_win, flash_win_vip)
    num_depth = int(len(depths)/len(areas))
    
    ##%%
    # each layer of V1 with all layers of LM
    plt.figure(figsize=(11,11))    
#    title = '%s (%d mice)' %(cre, thisCre_numMice)    
    st = '%s - %s ' %(areas[1], areas[0]) # '%s each layer- %s all layers' %(areas[0], areas[1]
    plt.suptitle('%s\n%s' %(title,st), y=1, fontsize=18)
    
    # pooled
    # subplots of v1(each layer)-lm(all layers) traces: cc, cc_shfl, p
    gs = gridspec.GridSpec(3,1) #, width_ratios=[3, 1]) 
    gs.update(left=0, right=0.2, hspace=.6) #, hspace=1)
    # subplots of v1(each layer)-lm(all layers) peaks: cc, cc_shfl, p
    gs2 = gridspec.GridSpec(3,2)
    gs2.update(left=0.27, right=0.47, hspace=0.6) #, hspace=1)
    
    # pooledAll
    # subplots of v1(all layers)-lm(all layers) traces: cc, cc_shfl, p
    gs3 = gridspec.GridSpec(3,1) #, width_ratios=[3, 1]) 
    gs3.update(left=0.6, right=0.8, hspace=.6) #, hspace=1)
    # subplots of v1(all layers)-lm(all layers) peaks: cc, cc_shfl, p
    gs4 = gridspec.GridSpec(3,2)
    gs4.update(left=0.87, right=.97, hspace=0.6) #, hspace=1)
    
    
    ############################################### 
    ########## gs and gs3: subplot1: cc traces ##########   
    ############################################### 
    yl = 'Spearman cc' #'CC'
    lims0 = [np.nanmin(top0[xlim_frs] - top_sd0[xlim_frs]) , np.nanmax(top0[xlim_frs] + top_sd0[xlim_frs])]
    
    h = []
    ax1 = plt.subplot(gs[0,0])        

    for i2 in range(num_depth): # loop through V1 layers (subplots) (corr between each layer of V1 and layer layers of LM)  
        h0 = plt.plot(time_trace, top0[:,i2], color=jet_cm[i2], label='%dum' %(depths[i2+num_depth]))[0] #  label='%s %dum' %(areas[1], depths[i2]))[0]
        plt.fill_between(time_trace, top0[:,i2] - top_sd0[:,i2], top0[:,i2] + top_sd0[:,i2], alpha=alph,  edgecolor=jet_cm[i2], facecolor=jet_cm[i2])
        h.append(h0)
        plt.title('%s (each), %s (all)' %(areas[1], areas[0]), y=1, fontsize=13)            
    
    plt.legend(loc='upper left', bbox_to_anchor=(-.5, 1.5), frameon=False, fontsize=12, handlelength=1, ncol=2, columnspacing=.7, handletextpad=.1) #mode='expand')    
    plot_flashLines_ticks_legend(lims0, [], flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab=yl, xmjn=xmjn)
    # mark peak_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims0[1], peak_win[0], peak_win[1], color='gray')
#     plt.hlines(lims0[1], flash_win_final[0], flash_win_final[1], color='green')
    plt.xlim(xlim);


    #### pooledAll 
    lims0 = [np.nanmin(top0a[xlim_frs] - top_sd0a[xlim_frs]) , np.nanmax(top0a[xlim_frs] + top_sd0a[xlim_frs])]
    
    ax1 = plt.subplot(gs3[0,0])        
    h0 = plt.plot(time_trace, top0a, color='k')[0] #  label='%s %dum' %(areas[1], depths[i2]))[0]
    plt.fill_between(time_trace, top0a - top_sd0a, top0a + top_sd0a, alpha=alph,  edgecolor='k', facecolor='k')
    plt.title('%s (all), %s (all)' %(areas[1], areas[0]), y=1, fontsize=13)            
    
    plot_flashLines_ticks_legend(lims0, [], flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab=yl, xmjn=xmjn)
    # mark peak_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims0[1], peak_win[0], peak_win[1], color='gray')
#     plt.hlines(lims0[1], flash_win_final[0], flash_win_final[1], color='green')
    plt.xlim(xlim);
 

    ###############################################     
    ########## gs and gs3: subplot2: p traces ##########
    ############################################### 
    yl = 'Fract sig neuron pairs'
    lims0 = [np.nanmin(top2[xlim_frs] - top_sd2[xlim_frs]) , np.nanmax(top2[xlim_frs] + top_sd2[xlim_frs])]
 
    h = []
    ax1 = plt.subplot(gs[1,0])        

    for i2 in range(num_depth): # loop through V1 layers (subplots) (corr between each layer of V1 and all layers of LM)  
        h0 = plt.plot(time_trace, top2[:,i2], color=jet_cm[i2], label='%dum' %(depths[i2+num_depth]))[0] #  label='%s %dum' %(areas[1], depths[i2]))[0]
        plt.fill_between(time_trace, top2[:,i2] - top_sd2[:,i2], top2[:,i2] + top_sd2[:,i2], alpha=alph,  edgecolor=jet_cm[i2], facecolor=jet_cm[i2])
        h.append(h0)
        plt.title('%s (each), %s (all)' %(areas[1], areas[0]), y=1, fontsize=13)            
    
#    plt.legend(loc='right', bbox_to_anchor=(0, .7), frameon=False, fontsize=12, handlelength=1)
    plot_flashLines_ticks_legend(lims0, [], flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab=yl, xmjn=xmjn)
    # mark peak_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims0[1], peak_win[0], peak_win[1], color='gray')
#     plt.hlines(lims0[1], flash_win_final[0], flash_win_final[1], color='green')
    plt.xlim(xlim)        
    
    
    #### pooledAll 
    lims0 = [np.nanmin(top2a[xlim_frs] - top_sd2a[xlim_frs]) , np.nanmax(top2a[xlim_frs] + top_sd2a[xlim_frs])]
    
    ax1 = plt.subplot(gs3[1,0])        
    h0 = plt.plot(time_trace, top2a, color='k')[0] #  label='%s %dum' %(areas[1], depths[i2]))[0]
    plt.fill_between(time_trace, top2a - top_sd2a, top2a + top_sd2a, alpha=alph,  edgecolor='k', facecolor='k')
    plt.title('%s (all), %s (all)' %(areas[1], areas[0]), y=1, fontsize=13)            
    
    plot_flashLines_ticks_legend(lims0, [], flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab=yl, xmjn=xmjn)
    # mark peak_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims0[1], peak_win[0], peak_win[1], color='gray')
#     plt.hlines(lims0[1], flash_win_final[0], flash_win_final[1], color='green')
    plt.xlim(xlim);

    
    ############################################### 
    ########## gs and gs3: subplot3: cc_shfl ##########    
    ############################################### 
    '''
    yl = 'Spearman cc_shfl' #'CC_shfl'
    lims0 = [np.nanmin(top1[xlim_frs] - top_sd1[xlim_frs]) , np.nanmax(top1[xlim_frs] + top_sd1[xlim_frs])]
 
    h = []
    ax1 = plt.subplot(gs[2,0])        

    for i2 in range(num_depth): # loop through V1 layers (subplots) (corr between each layer of V1 and layer layers of LM)  
        h0 = plt.plot(time_trace, top1[:,i2], color=jet_cm[i2], label='%dum' %(depths[i2+num_depth]))[0] #  label='%s %dum' %(areas[1], depths[i2]))[0]
        plt.fill_between(time_trace, top1[:,i2] - top_sd1[:,i2], top1[:,i2] + top_sd1[:,i2], alpha=alph,  edgecolor=jet_cm[i2], facecolor=jet_cm[i2])
        h.append(h0)
        plt.title('%s (each), %s (all)' %(areas[1], areas[0]), y=1, fontsize=13)            
    
    plot_flashLines_ticks_legend(lims0, h, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab=yl, xmjn=xmjn)
    # mark peak_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims0[1], peak_win[0], peak_win[1], color='gray')
    plt.hlines(lims0[1], flash_win_final[0], flash_win_final[1], color='green')
    plt.xlim(xlim)        


    #### pooledAll 
    lims0 = [np.nanmin(top1a[xlim_frs] - top_sd1a[xlim_frs]) , np.nanmax(top1a[xlim_frs] + top_sd1a[xlim_frs])]
    
    ax1 = plt.subplot(gs3[2,0])        
    h0 = plt.plot(time_trace, top1a, color='k')[0] #  label='%s %dum' %(areas[1], depths[i2]))[0]
    plt.fill_between(time_trace, top1a - top_sd1a, top1a + top_sd1a, alpha=alph,  edgecolor='k', facecolor='k')
    plt.title('%s (all), %s (all)' %(areas[1], areas[0]), y=1, fontsize=13)            
    
    plot_flashLines_ticks_legend(lims0, [], flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab=yl, xmjn=xmjn)
    # mark peak_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims0[1], peak_win[0], peak_win[1], color='gray')
    plt.hlines(lims0[1], flash_win_final[0], flash_win_final[1], color='green')
    plt.xlim(xlim);

    '''

    #############################################################################################################################################
    #############################################################################################################################################
    
    ###############################################   
    ######## gs2 and gs4: subplot 1: cc quantification #######
    ###############################################   
    yl = 'cc amplitude' #'CC'

    x = np.arange(num_depth) # for errorbars                
#    lims0_f = [np.nanmin(f_top0 - f_top_sd0) , np.nanmax(f_top0 + f_top_sd0)]
#    lims0_o = [np.nanmin(o_top0 - o_top_sd0) , np.nanmax(o_top0 + o_top_sd0)]
    lims0_fo = np.squeeze([np.nanmin([f_top0 - f_top_sd0, o_top0 - o_top_sd0]) , np.nanmax([f_top0 + f_top_sd0, o_top0 + o_top_sd0])])
    lims0_fo[0] = lims0_fo[0] - np.diff(lims0_fo) / 30.
    lims0_fo[1] = lims0_fo[1] + np.diff(lims0_fo) / 30.    
    

    ############ flash ############
    ax1 = plt.subplot(gs2[0,0])        

    # plot corr quantification of each V1 layer with all LM layers
    ax1.errorbar(x, f_top0, yerr=f_top_sd0, fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    xticklabs = depths[num_depth:].astype(int) # '%s %dum' %(areas[0], depths[i1])
    plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.xaxis.set_label_coords(1.1, -.25)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.tick_params(labelsize=10)
    plt.xlim([-.5, len(x)-.5])
#    plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
    if ~np.isnan(lims0_fo).any():
        plt.gca().set_ylim(lims0_fo)
#    plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    ylim0 = lims0_fo #plt.gca().get_ylim()
    text_y = ylim0[1] + np.diff(ylim0)/10
#    if i2==0:
    plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)


    ############ omission ############
    ax1 = plt.subplot(gs2[0,1])        

    # plot corr quantification of each V1 layer with all LM layers
    ax1.errorbar(x, o_top0, yerr=o_top_sd0, fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    xticklabs = depths[num_depth:].astype(int) # '%s %dum' %(areas[0], depths[i1])
#    plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.set_yticklabels('')#, rotation=45)
    ax1.tick_params(labelsize=10)
    plt.xlim([-.7, len(x)-.5])
#        plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)      
    if ~np.isnan(lims0_fo).any():
        plt.gca().set_ylim(lims0_fo)        
#        plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    ylim0 = lims0_fo #plt.gca().get_ylim()        
    text_y = ylim0[1] + np.diff(ylim0)/10
#    if i2==0:
    plt.text(-.7, text_y, 'Omission', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)


    ########## pooledAll
    x = np.arange(2) # flash, omission
    lims0_fo = np.squeeze([np.nanmin([f_top0a - f_top_sd0a, o_top0a - o_top_sd0a]) , np.nanmax([f_top0a + f_top_sd0a, o_top0a + o_top_sd0a])])
    lims0_fo[0] = lims0_fo[0] - np.diff(lims0_fo) / 30.
    lims0_fo[1] = lims0_fo[1] + np.diff(lims0_fo) / 30.    

    ax1 = plt.subplot(gs4[0,0])
    # plot corr quantification of each V1 layer with all LM layers
    ax1.errorbar(x, [f_top0a, o_top0a], yerr=[f_top_sd0a, o_top_sd0a], fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    xticklabs = 'Flash', 'Omis' #depths[num_depth:].astype(int) # '%s %dum' %(areas[0], depths[i1])
#    plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.tick_params(labelsize=10)
    plt.xlim([-.5, len(x)-.5])
#    plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
    if ~np.isnan(lims0_fo).any():
        plt.gca().set_ylim(lims0_fo)
#    plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    ylim0 = lims0_fo #plt.gca().get_ylim()
    text_y = ylim0[1] + np.diff(ylim0)/10
#    if i2==0:
#    plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)



    ###############################################   
    ######## gs2 and gs4: subplot 2: p quantification #######
    ###############################################   
    yl = 'p amplitude' #'CC'

    x = np.arange(num_depth) # for errorbars                
#    lims0_f = [np.nanmin(f_top0 - f_top_sd0) , np.nanmax(f_top0 + f_top_sd0)]
#    lims0_o = [np.nanmin(o_top0 - o_top_sd0) , np.nanmax(o_top0 + o_top_sd0)]
    lims0_fo = np.squeeze([np.nanmin([f_top2 - f_top_sd2, o_top2 - o_top_sd2]) , np.nanmax([f_top2 + f_top_sd2, o_top2 + o_top_sd2])])
    lims0_fo[0] = lims0_fo[0] - np.diff(lims0_fo) / 30.
    lims0_fo[1] = lims0_fo[1] + np.diff(lims0_fo) / 30.    
    

    ############ flash ############
    ax1 = plt.subplot(gs2[1,0])        

    # plot corr quantification of each V1 layer with all LM layers
    ax1.errorbar(x, f_top2, yerr=f_top_sd2, fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    xticklabs = depths[num_depth:].astype(int) # '%s %dum' %(areas[0], depths[i1])
    plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.xaxis.set_label_coords(1.1, -.25)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.tick_params(labelsize=10)
    plt.xlim([-.5, len(x)-.5])
#    plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
    if ~np.isnan(lims0_fo).any():
        plt.gca().set_ylim(lims0_fo)
#    plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    ylim0 = lims0_fo #plt.gca().get_ylim()
    text_y = ylim0[1] + np.diff(ylim0)/10
#    if i2==0:
    plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)


    ############ omission ############
    ax1 = plt.subplot(gs2[1,1])        

    # plot corr quantification of each V1 layer with all LM layers
    ax1.errorbar(x, o_top2, yerr=o_top_sd2, fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    xticklabs = depths[num_depth:].astype(int) # '%s %dum' %(areas[0], depths[i1])
#    plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.set_yticklabels('')#, rotation=45)
    ax1.tick_params(labelsize=10)
    plt.xlim([-.7, len(x)-.5])
#        plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
    if ~np.isnan(lims0_fo).any():
        plt.gca().set_ylim(lims0_fo)        
#        plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    ylim0 = lims0_fo #plt.gca().get_ylim()        
    text_y = ylim0[1] + np.diff(ylim0)/10
#    if i2==0:
    plt.text(-.7, text_y, 'Omission', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)



    ########## pooledAll
    x = np.arange(2) # flash, omission
    lims0_fo = np.squeeze([np.nanmin([f_top2a - f_top_sd2a, o_top2a - o_top_sd2a]) , np.nanmax([f_top2a + f_top_sd2a, o_top2a + o_top_sd2a])])
    lims0_fo[0] = lims0_fo[0] - np.diff(lims0_fo) / 30.
    lims0_fo[1] = lims0_fo[1] + np.diff(lims0_fo) / 30.    

    ax1 = plt.subplot(gs4[1,0])        
    # plot corr quantification of each V1 layer with all LM layers
    ax1.errorbar(x, [f_top2a, o_top2a], yerr=[f_top_sd2a, o_top_sd2a], fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    xticklabs = 'Flash', 'Omis' #depths[num_depth:].astype(int) # '%s %dum' %(areas[0], depths[i1])
#    plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.tick_params(labelsize=10)
    plt.xlim([-.5, len(x)-.5])
#    plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
    if ~np.isnan(lims0_fo).any():
        plt.gca().set_ylim(lims0_fo)
#    plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    ylim0 = lims0_fo #plt.gca().get_ylim()
    text_y = ylim0[1] + np.diff(ylim0)/10
#    if i2==0:
#    plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)



    ###############################################   
    ######## gs2 and gs4: subplot 3: cc_shfl quantification #######
    ###############################################   
    '''
    yl = 'cc_shfl amplitude' #'CC'

    x = np.arange(num_depth) # for errorbars                
#    lims0_f = [np.nanmin(f_top0 - f_top_sd0) , np.nanmax(f_top0 + f_top_sd0)]
#    lims0_o = [np.nanmin(o_top0 - o_top_sd0) , np.nanmax(o_top0 + o_top_sd0)]
    lims0_fo = np.squeeze([np.nanmin([f_top1 - f_top_sd1, o_top1 - o_top_sd1]) , np.nanmax([f_top1 + f_top_sd1, o_top1 + o_top_sd1])])
    lims0_fo[0] = lims0_fo[0] - np.diff(lims0_fo) / 30.
    lims0_fo[1] = lims0_fo[1] + np.diff(lims0_fo) / 30.    
    

    ############ flash ############
    ax1 = plt.subplot(gs2[1,0])        

    # plot corr quantification of each V1 layer with all LM layers
    ax1.errorbar(x, f_top1, yerr=f_top_sd1, fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    xticklabs = depths[num_depth:].astype(int) # '%s %dum' %(areas[0], depths[i1])
    plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.xaxis.set_label_coords(1.1, -.25)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.tick_params(labelsize=10)
    plt.xlim([-.5, len(x)-.5])
#    plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
    plt.gca().set_ylim(lims0_fo)
#    plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    ylim0 = lims0_fo #plt.gca().get_ylim()
    text_y = ylim0[1] + np.diff(ylim0)/10
#    if i2==0:
    plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)


    ############ omission ############
    ax1 = plt.subplot(gs2[1,1])        

    # plot corr quantification of each V1 layer with all LM layers
    ax1.errorbar(x, o_top1, yerr=o_top_sd1, fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    xticklabs = depths[num_depth:].astype(int) # '%s %dum' %(areas[0], depths[i1])
#    plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.set_yticklabels('')#, rotation=45)
    ax1.tick_params(labelsize=10)
    plt.xlim([-.7, len(x)-.5])
#        plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
    plt.gca().set_ylim(lims0_fo)        
#        plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    ylim0 = lims0_fo #plt.gca().get_ylim()        
    text_y = ylim0[1] + np.diff(ylim0)/10
#    if i2==0:
    plt.text(-.7, text_y, 'Omission', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)


    ########## pooledAll
    x = np.arange(2) # flash, omission
    lims0_fo = np.squeeze([np.nanmin([f_top1a - f_top_sd1a, o_top1a - o_top_sd1a]) , np.nanmax([f_top1a + f_top_sd1a, o_top1a + o_top_sd1a])])
    lims0_fo[0] = lims0_fo[0] - np.diff(lims0_fo) / 30.
    lims0_fo[1] = lims0_fo[1] + np.diff(lims0_fo) / 30.    

    ax1 = plt.subplot(gs4[1,0])        
    # plot corr quantification of each V1 layer with all LM layers
    ax1.errorbar(x, [f_top1a, o_top1a], yerr=[f_top_sd1a, o_top_sd1a], fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    xticklabs = 'Flash', 'Omis' #depths[num_depth:].astype(int) # '%s %dum' %(areas[0], depths[i1])
#    plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.tick_params(labelsize=10)
    plt.xlim([-.5, len(x)-.5])
#    plt.title('%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
    plt.gca().set_ylim(lims0_fo)
#    plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    ylim0 = lims0_fo #plt.gca().get_ylim()
    text_y = ylim0[1] + np.diff(ylim0)/10
#    if i2==0:
#    plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)
    '''


    #############################
#    dir_now = 'corr_omit_flash'        
    if dosavefig:
        if np.isnan(mouseid): # plotting average across mice
            nam = '%s_aveMice_pooledCorr_%s%s_%s' %(cre, fgn, whatSess, now)        
        else: # mouse_id is provided; plotting single mouse data
            nam = '%s_mouse%d_pooledCorr_aveSess_%s%s_%s' %(cre, mouseid, fgn, whatSess, now)        
        fign = os.path.join(dir0, dir_now, nam+fmt)        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


        
        
#%% Plot superimposed pooledAll traces for cc12, cc11, cc22
# also make errorbar plots for flash (cc12,cc11,cc22) and omission (cc12,cc11,cc22).

def plot_superimposed_pooled_cc_ccShfl_p(time_trace, top0a, top_sd0a, top1a, top_sd1a, top2a, top_sd2a, \
    f_top0a, f_top_sd0a, o_top0a, o_top_sd0a, f_top1a, f_top_sd1a, o_top1a, o_top_sd1a, f_top2a, f_top_sd2a, o_top2a, o_top_sd2a, \
    fgn, areas, yl, title, xlim_frs, mouseid=np.nan, dosavefig=0):

#     flash_win_final = set_flash_win_final(cre, session_novel, flash_win, flash_win_vip)
    
    st12 = areas[1]+'-'+areas[0]    # V1-LM
    st22 = areas[1]+'-'+areas[1]    # V1-V1
    st11 = areas[0]+'-'+areas[0]    # LM-LM
    
    jet_cm0 = colorOrder(nlines=3)


    plt.figure(figsize=(11,4))
    plt.suptitle('%s' %(title), y=1.1, fontsize=18)
    
    gs = gridspec.GridSpec(1,1) #, width_ratios=[3, 1]) 
    gs.update(left=0.2, right=0.45) #, wspace=.6) #, hspace=1)
    gs2 = gridspec.GridSpec(1,2)
    gs2.update(left=0.65, right=0.8, wspace=0.6) #, hspace=1)

    
    #### pooledAll; cc trace; superimposed v1-lm, v1-v1, and lm-lm 
    mn = np.nanmin([top0a[xlim_frs] - top_sd0a[xlim_frs], top1a[xlim_frs] - top_sd1a[xlim_frs], top2a[xlim_frs] - top_sd2a[xlim_frs]])
    mx = np.nanmax([top0a[xlim_frs] + top_sd0a[xlim_frs], top1a[xlim_frs] + top_sd1a[xlim_frs], top2a[xlim_frs] + top_sd2a[xlim_frs]])
    lims0 = [mn , mx]

    ax1 = plt.subplot(gs[0,0])        

    h = []
    # V1-LM
    h0 = plt.plot(time_trace, top0a, color=jet_cm0[0], label=st12)[0] #  label='%s %dum' %(areas[1], depths[i2]))[0]
    plt.fill_between(time_trace, top0a - top_sd0a, top0a + top_sd0a, alpha=alph, edgecolor=jet_cm0[0], facecolor=jet_cm0[0])
    h.append(h0)
    # V1-V1
    h0 = plt.plot(time_trace, top2a, color=jet_cm0[1], label=st22)[0] #  label='%s %dum' %(areas[1], depths[i2]))[0]
    plt.fill_between(time_trace, top2a - top_sd2a, top2a + top_sd2a, alpha=alph, edgecolor=jet_cm0[1], facecolor=jet_cm0[1])
    h.append(h0)
    # LM-LM
    h0 = plt.plot(time_trace, top1a, color=jet_cm0[2], label=st11)[0] #  label='%s %dum' %(areas[1], depths[i2]))[0]
    plt.fill_between(time_trace, top1a - top_sd1a, top1a + top_sd1a, alpha=alph, edgecolor=jet_cm0[2], facecolor=jet_cm0[2])
    h.append(h0)

    # plt.title('%s (all), %s (all)' %(areas[1], areas[0]), y=1, fontsize=13)            
    plot_flashLines_ticks_legend(lims0, h, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab=yl, xmjn=xmjn)
    # mark peak_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims0[1], peak_win[0], peak_win[1], color='gray')
#     plt.hlines(lims0[1], flash_win_final[0], flash_win_final[1], color='green')
    plt.xlim(xlim);


    # yl = 'cc amplitude'

    ########## pooledAll; cc quantification ##########

    x = np.arange(3) # flash, omission
    xticklabs = st12, st22, st11 # 'V1-LM', 'V1-V1', 'LM-LM' #depths[num_depth:].astype(int) # '%s %dum' %(areas[0], depths[i1])

    f_mn = np.nanmin([f_top0a - f_top_sd0a, f_top1a - f_top_sd1a, f_top2a - f_top_sd2a])
    f_mx = np.nanmax([f_top0a + f_top_sd0a, f_top1a + f_top_sd1a, f_top2a + f_top_sd2a])
    o_mn = np.nanmin([o_top0a - o_top_sd0a, o_top1a - o_top_sd1a, o_top2a - o_top_sd2a])
    o_mx = np.nanmax([o_top0a + o_top_sd0a, o_top1a + o_top_sd1a, o_top2a + o_top_sd2a])

    lims0_fo = np.squeeze([np.nanmin([f_mn, o_mn]) , np.nanmax([f_mx, o_mx])])
    lims0_fo[0] = lims0_fo[0] - np.diff(lims0_fo) / 30.
    lims0_fo[1] = lims0_fo[1] + np.diff(lims0_fo) / 30.    

    #### Flash
    ax1 = plt.subplot(gs2[0,0])

    ax1.errorbar(x, [f_top0a, f_top2a, f_top1a], yerr=[f_top_sd0a, f_top_sd2a, f_top_sd1a], fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    # plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.tick_params(labelsize=10)
    plt.xlim([-.5, len(x)-.5])
    plt.title('Flash') #'%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
    plt.ylim(lims0_fo)
    plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    # ylim0 = lims0_fo #plt.gca().get_ylim()
    # text_y = ylim0[1] + np.diff(ylim0)/10
    #    if i2==0:
    # plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
    #        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
    #        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)


    #### Omission
    ax1 = plt.subplot(gs2[0,1])

    ax1.errorbar(x, [o_top0a, o_top2a, o_top1a], yerr=[o_top_sd0a, o_top_sd2a, o_top_sd1a], fmt='o', markersize=3, capsize=3, color='k') # jet_cm[i1] # , label='%s %dum' %(areas[0], depths[i1])

    plt.hlines(0, 0, len(x)-1, linestyle=':')
    ax1.set_xticks(x)
    # plt.xlabel('%s depth (um)' %areas[1], fontsize=12)
    ax1.set_xticklabels(xticklabs, rotation=45)
    ax1.tick_params(labelsize=10) # axis='x',
    plt.xlim([-.5, len(x)-.5])
    plt.title('Omission') #'%s, %d um' %(areas[1], depths[i2+num_depth]), y=1, fontsize=13)            
    plt.ylim(lims0_fo)
    # use below for a black background (I guess)
#     ax1.tick_params(axis='y', labelcolor='w',labelsize=1) #labelleft='off') #) #
    #    plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
    # ylim0 = lims0_fo #plt.gca().get_ylim()
    # text_y = ylim0[1] + np.diff(ylim0)/10
    #    if i2==0:
    # plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)


        
    #############################
#    dir_now = 'corr_omit_flash'        
    if dosavefig:
        if np.isnan(mouseid): # plotting average across mice
            nam = '%s_aveMice_pooledCorr_%s%s_%s' %(cre, fgn, whatSess, now)        
        else: # mouse_id is provided; plotting single mouse data
            nam = '%s_mouse%d_pooledCorr_aveSess_%s%s_%s' %(cre, mouseid, fgn, whatSess, now)        
        fign = os.path.join(dir0, dir_now, nam+fmt)        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

    
    
    
    
    
    
    

###########################################################################
###########################################################################
###########################################################################
#%% Analysis starts here        
###########################################################################
###########################################################################
###########################################################################

#%% Set dataframe "corr_trace_peak_allMice", which includes corr and p traces and their quantification for each layer pair, for each mouse.
# vars (cc and p for actual and shuffled data): 
    # session-averaged traces for each layer pair: eg. cc12_sessAv_44
    # session-averaged quantification of peaks (after omission and flash) for each layer pair: eg. cc12_peak_amp_omit_sessAv
    
# below we set traces and cc quantification in three different ways:
# 1. corr between each layer and another layer (size: 4x4)
# 2. corr between each layer and neurons in all other layers  (size: 4)    
# 3. corr between all neuron pairs (regardless of their layer) (size: 1)
    
cols = ['mouse_id', 'cre', 'session_stages', 'session_labs', 'areas', 'depths', \
        'num_valid_sess_12_44', 'num_valid_sess_11_44', 'num_valid_sess_22_44', \
        'cc12_sessAv_44', 'cc12_sessSd_44', 'cc11_sessAv_44', 'cc11_sessSd_44', 'cc22_sessAv_44', 'cc22_sessSd_44', \
        'p12_sessAv_44', 'p12_sessSd_44', 'p11_sessAv_44', 'p11_sessSd_44', 'p22_sessAv_44', 'p22_sessSd_44', \
        'cc12_sessAv_44_shfl', 'cc12_sessSd_44_shfl', 'cc11_sessAv_44_shfl', 'cc11_sessSd_44_shfl', 'cc22_sessAv_44_shfl', 'cc22_sessSd_44_shfl', \
        'p12_sessAv_44_shfl', 'p12_sessSd_44_shfl', 'p11_sessAv_44_shfl', 'p11_sessSd_44_shfl', 'p22_sessAv_44_shfl', 'p22_sessSd_44_shfl', \
        'cc12_peak_amp_omit_sessAv', 'cc12_peak_amp_omit_sessSd', 'cc12_peak_amp_flash_sessAv', 'cc12_peak_amp_flash_sessSd', \
        'cc11_peak_amp_omit_sessAv', 'cc11_peak_amp_omit_sessSd', 'cc11_peak_amp_flash_sessAv', 'cc11_peak_amp_flash_sessSd', \
        'cc22_peak_amp_omit_sessAv', 'cc22_peak_amp_omit_sessSd', 'cc22_peak_amp_flash_sessAv', 'cc22_peak_amp_flash_sessSd', \
        'p12_peak_amp_omit_sessAv', 'p12_peak_amp_omit_sessSd', 'p12_peak_amp_flash_sessAv', 'p12_peak_amp_flash_sessSd', \
        'p11_peak_amp_omit_sessAv', 'p11_peak_amp_omit_sessSd', 'p11_peak_amp_flash_sessAv', 'p11_peak_amp_flash_sessSd', \
        'p22_peak_amp_omit_sessAv', 'p22_peak_amp_omit_sessSd', 'p22_peak_amp_flash_sessAv', 'p22_peak_amp_flash_sessSd', \
        'cc12_peak_amp_omit_sessAv_shfl', 'cc12_peak_amp_omit_sessSd_shfl', 'cc12_peak_amp_flash_sessAv_shfl', 'cc12_peak_amp_flash_sessSd_shfl', \
        'cc11_peak_amp_omit_sessAv_shfl', 'cc11_peak_amp_omit_sessSd_shfl', 'cc11_peak_amp_flash_sessAv_shfl', 'cc11_peak_amp_flash_sessSd_shfl', \
        'cc22_peak_amp_omit_sessAv_shfl', 'cc22_peak_amp_omit_sessSd_shfl', 'cc22_peak_amp_flash_sessAv_shfl', 'cc22_peak_amp_flash_sessSd_shfl', \
        'p12_peak_amp_omit_sessAv_shfl', 'p12_peak_amp_omit_sessSd_shfl', 'p12_peak_amp_flash_sessAv_shfl', 'p12_peak_amp_flash_sessSd_shfl', \
        'p11_peak_amp_omit_sessAv_shfl', 'p11_peak_amp_omit_sessSd_shfl', 'p11_peak_amp_flash_sessAv_shfl', 'p11_peak_amp_flash_sessSd_shfl', \
        'p22_peak_amp_omit_sessAv_shfl', 'p22_peak_amp_omit_sessSd_shfl', 'p22_peak_amp_flash_sessAv_shfl', 'p22_peak_amp_flash_sessSd_shfl',\
        'num_valid_sess_pooledLayers_12', 'num_valid_sess_pooledLayers_11', 'num_valid_sess_pooledLayers_22', \
        'num_valid_sess_pooledAllLayers_12', 'num_valid_sess_pooledAllLayers_11', 'num_valid_sess_pooledAllLayers_22', \
        'cc12_pooled_sessAv', 'cc12_pooled_sessSd', 'cc11_pooled_sessAv', 'cc11_pooled_sessSd', 'cc22_pooled_sessAv', 'cc22_pooled_sessSd',	\
        'p12_pooled_sessAv', 'p12_pooled_sessSd', 'p11_pooled_sessAv', 'p11_pooled_sessSd', 'p22_pooled_sessAv', 'p22_pooled_sessSd', \
        'p12_pooled_sessAv_shfl', 'p12_pooled_sessSd_shfl', 'p11_pooled_sessAv_shfl', 'p11_pooled_sessSd_shfl', 'p22_pooled_sessAv_shfl', 'p22_pooled_sessSd_shfl', \
        'cc12_pooledAll_sessAv', 'cc12_pooledAll_sessSd', 'cc11_pooledAll_sessAv', 'cc11_pooledAll_sessSd', 'cc22_pooledAll_sessAv', 'cc22_pooledAll_sessSd', \
        'p12_pooledAll_sessAv', 'p12_pooledAll_sessSd', 'p11_pooledAll_sessAv', 'p11_pooledAll_sessSd', 'p22_pooledAll_sessAv', 'p22_pooledAll_sessSd', \
        'p12_pooledAll_sessAv_shfl', 'p12_pooledAll_sessSd_shfl', 'p11_pooledAll_sessAv_shfl', 'p11_pooledAll_sessSd_shfl', 'p22_pooledAll_sessAv_shfl', 'p22_pooledAll_sessSd_shfl', \
        'cc12_peak_amp_omit_sessAv_pooled', 'cc12_peak_amp_omit_sessSd_pooled', 'cc12_peak_amp_flash_sessAv_pooled', 'cc12_peak_amp_flash_sessSd_pooled', \
        'cc11_peak_amp_omit_sessAv_pooled', 'cc11_peak_amp_omit_sessSd_pooled', 'cc11_peak_amp_flash_sessAv_pooled', 'cc11_peak_amp_flash_sessSd_pooled', \
        'cc22_peak_amp_omit_sessAv_pooled', 'cc22_peak_amp_omit_sessSd_pooled', 'cc22_peak_amp_flash_sessAv_pooled', 'cc22_peak_amp_flash_sessSd_pooled', \
        'p12_peak_amp_omit_sessAv_pooled', 'p12_peak_amp_omit_sessSd_pooled', 'p12_peak_amp_flash_sessAv_pooled', 'p12_peak_amp_flash_sessSd_pooled', \
        'p11_peak_amp_omit_sessAv_pooled', 'p11_peak_amp_omit_sessSd_pooled', 'p11_peak_amp_flash_sessAv_pooled', 'p11_peak_amp_flash_sessSd_pooled', \
        'p22_peak_amp_omit_sessAv_pooled', 'p22_peak_amp_omit_sessSd_pooled', 'p22_peak_amp_flash_sessAv_pooled', 'p22_peak_amp_flash_sessSd_pooled', \
        'p12_peak_amp_omit_sessAv_pooled_shfl', 'p12_peak_amp_omit_sessSd_pooled_shfl', 'p12_peak_amp_flash_sessAv_pooled_shfl', 'p12_peak_amp_flash_sessSd_pooled_shfl', \
        'p11_peak_amp_omit_sessAv_pooled_shfl', 'p11_peak_amp_omit_sessSd_pooled_shfl', 'p11_peak_amp_flash_sessAv_pooled_shfl', 'p11_peak_amp_flash_sessSd_pooled_shfl', \
        'p22_peak_amp_omit_sessAv_pooled_shfl', 'p22_peak_amp_omit_sessSd_pooled_shfl', 'p22_peak_amp_flash_sessAv_pooled_shfl', 'p22_peak_amp_flash_sessSd_pooled_shfl', \
        'cc12_peak_amp_omit_sessAv_pooledAll', 'cc12_peak_amp_omit_sessSd_pooledAll', 'cc12_peak_amp_flash_sessAv_pooledAll', 'cc12_peak_amp_flash_sessSd_pooledAll', \
        'cc11_peak_amp_omit_sessAv_pooledAll', 'cc11_peak_amp_omit_sessSd_pooledAll', 'cc11_peak_amp_flash_sessAv_pooledAll', 'cc11_peak_amp_flash_sessSd_pooledAll', \
        'cc22_peak_amp_omit_sessAv_pooledAll',  'cc22_peak_amp_omit_sessSd_pooledAll', 'cc22_peak_amp_flash_sessAv_pooledAll', 'cc22_peak_amp_flash_sessSd_pooledAll', \
        'p12_peak_amp_omit_sessAv_pooledAll', 'p12_peak_amp_omit_sessSd_pooledAll', 'p12_peak_amp_flash_sessAv_pooledAll', 'p12_peak_amp_flash_sessSd_pooledAll', \
        'p11_peak_amp_omit_sessAv_pooledAll', 'p11_peak_amp_omit_sessSd_pooledAll', 'p11_peak_amp_flash_sessAv_pooledAll', 'p11_peak_amp_flash_sessSd_pooledAll', \
        'p22_peak_amp_omit_sessAv_pooledAll', 'p22_peak_amp_omit_sessSd_pooledAll', 'p22_peak_amp_flash_sessAv_pooledAll', 'p22_peak_amp_flash_sessSd_pooledAll', \
        'p12_peak_amp_omit_sessAv_pooledAll_shfl', 'p12_peak_amp_omit_sessSd_pooledAll_shfl', 'p12_peak_amp_flash_sessAv_pooledAll_shfl', 'p12_peak_amp_flash_sessSd_pooledAll_shfl', \
        'p11_peak_amp_omit_sessAv_pooledAll_shfl', 'p11_peak_amp_omit_sessSd_pooledAll_shfl', 'p11_peak_amp_flash_sessAv_pooledAll_shfl', 'p11_peak_amp_flash_sessSd_pooledAll_shfl', \
        'p22_peak_amp_omit_sessAv_pooledAll_shfl', 'p22_peak_amp_omit_sessSd_pooledAll_shfl', 'p22_peak_amp_flash_sessAv_pooledAll_shfl', 'p22_peak_amp_flash_sessSd_pooledAll_shfl']
                
corr_trace_peak_allMice = pd.DataFrame([], columns = cols)


#%%
for im in range(len(all_mice_id)): # im=0
    
    #%%
    print('_______________ mouse %d _______________' %im)
    
    mouse_id = all_mice_id[im]
    
    if sum((all_sess_2an['mouse_id']==mouse_id).values) > 0: # make sure there is data for this mouse

        # remember if doCorrs=1, each row of all_sess is for one session (not one experiment!)
        all_sess_2an_this_mouse = all_sess_2an[all_sess_2an['mouse_id']==mouse_id] # num_sessions 
        print(len(all_sess_2an_this_mouse))
#        print(all_sess_2an_this_mouse['n_omissions'].values)
        
        # Remove sessions without omission trials from all_sess_2an_this_mouse
        
        # I dont think the statement below applies anymore, bc in omissions_traces_peaks.py we set n_omissions for this_sess_allExp from non-nan sessions. 
        # the following is correct, but the problem is that we got n_omissions from the 1st experiment, which if invalid, will have nan
        # for n_omissions ... this has happened to session_id: 880709154
        sess_no_omit = np.argwhere(np.isnan(all_sess_2an_this_mouse['n_omissions'].values.astype(float))).flatten()
        if len(sess_no_omit) > 0:
            
            sess2rmv = all_sess_2an_this_mouse['session_id'].values[sess_no_omit]
            print('Removing %d session(s) with no omissions: %s' %(len(sess_no_omit), str(sess2rmv)))
            sys.exit('Check below!')
            # remove sessions without omission trials from all_sess_2an_this_mouse
            d = all_sess_2an_this_mouse
            d = d.set_index('session_id')
            d = d.drop(sess2rmv, axis=0)
            
            all_sess_2an_this_mouse = d
       
        
        cre = all_sess_2an_this_mouse['cre'].iloc[0]
        cre = cre[:cre.find('-')] # remove the IRES-Cre part
                    
        session_dates = all_sess_2an_this_mouse['date'].values
        session_stages = all_sess_2an_this_mouse['stage'].values
        num_sessions = len(session_stages)
        
        areas = all_sess_2an_this_mouse['area'].iloc[0] #.astype(str) # (8)
        depths = np.round(np.mean(np.vstack(all_sess_2an_this_mouse['depth'].values), axis=0).astype(float)) # (8) # average across sessions
#        planes = all_sess_2an_this_mouse.index.values # (8*num_sessions)
        
        ######## Set session labels (stage names are too long)
    #    session_beg_inds = np.concatenate(([0], np.cumsum(num_trs_each_sess[:-1])+1))
        session_labs = []
        for ibeg in range(num_sessions): # ibeg=0
            a = str(session_stages[ibeg]); 
            en = np.argwhere([a[ich]=='_' for ich in range(len(a))])[-1].squeeze() + 6
            txt = str(session_stages[ibeg][15:]) # session_stages[ibeg][8:en]
            txt = txt[0:min(3, len(txt))]
            session_labs.append(txt)

   
        layerPairs_a12 = all_sess_2an_this_mouse['cc_cols_area12'].iloc[0] # 'l03': the 1st number indicate area 1 (LM) layer ; the 2nd number indicates area 2 (V1) layer
        layerPairs_aSame = all_sess_2an_this_mouse['cc_cols_areaSame'].iloc[0] # 'l03': correlation between layer 0 and 3 of the same area
                
        
        
        
        ###########################################################################
        ###########################################################################
        ###########################################################################        
        ###########################################################################
        ###########################################################################
        ###########################################################################                
        #%% Loop through each session, and compute average of corrs across all neuron pairs; also load shuffled data      
        # area1: LM
        # area 2: V1
        ###########################################################################
        ###########################################################################
        ###########################################################################        
        ###########################################################################
        ###########################################################################
        ###########################################################################        
        
        cc12_aveNPairs_allSess = np.full((num_sessions, num_frs, len(layerPairs_a12)), np.nan) # size: num_sess x num_frames x 16 (num_layerPairs between area 1 and 2)
        p12_aveNPairs_allSess = np.full((num_sessions, num_frs, len(layerPairs_a12)), np.nan)

        cc11_aveNPairs_allSess = np.full((num_sessions, num_frs, len(layerPairs_aSame)), np.nan) # size: num_sess x num_frames x 10 (num_layerPairs between area 1 and 1)
        p11_aveNPairs_allSess = np.full((num_sessions, num_frs, len(layerPairs_aSame)), np.nan)
        
        cc22_aveNPairs_allSess = np.full((num_sessions, num_frs, len(layerPairs_aSame)), np.nan) # size: num_sess x num_frames x 10 (num_layerPairs between area 2 and 2)
        p22_aveNPairs_allSess = np.full((num_sessions, num_frs, len(layerPairs_aSame)), np.nan)

        cc12_aveNPairs_allSess_shfl = np.full((num_sessions, num_frs, len(layerPairs_a12)), np.nan) # size: num_sess x num_frames x 16 (num_layerPairs between area 1 and 2)
        p12_aveNPairs_allSess_shfl = np.full((num_sessions, num_frs, len(layerPairs_a12)), np.nan)

        cc11_aveNPairs_allSess_shfl = np.full((num_sessions, num_frs, len(layerPairs_aSame)), np.nan) # size: num_sess x num_frames x 10 (num_layerPairs between area 1 and 1)
        p11_aveNPairs_allSess_shfl = np.full((num_sessions, num_frs, len(layerPairs_aSame)), np.nan)
        
        cc22_aveNPairs_allSess_shfl = np.full((num_sessions, num_frs, len(layerPairs_aSame)), np.nan) # size: num_sess x num_frames x 10 (num_layerPairs between area 2 and 2)
        p22_aveNPairs_allSess_shfl = np.full((num_sessions, num_frs, len(layerPairs_aSame)), np.nan)
        
        cc12_aveNPairsPooled_allSess = np.full((num_sessions, num_frs, num_depth), np.nan) # size: num_sess x num_frames x 4 # cc of each V1 layer with all neuron pairs of LM
        cc12_aveNPairsPooledAll_allSess = np.full((num_sessions, num_frs), np.nan) # size: num_sess x num_frames x 4 # grand average of all neuron pairs across the two areas, regardless of their layer
        cc11_aveNPairsPooled_allSess = np.full((num_sessions, num_frs, num_depth), np.nan) # size: num_sess x num_frames x 4 # cc of each LM layer with all neuron pairs of LM
        cc11_aveNPairsPooledAll_allSess = np.full((num_sessions, num_frs), np.nan) # size: num_sess x num_frames x 4 # grand average of all neuron pairs within LM, regardless of their layer        
        cc22_aveNPairsPooled_allSess = np.full((num_sessions, num_frs, num_depth), np.nan) # size: num_sess x num_frames x 4 # cc of each V1 layer with all neuron pairs of V1
        cc22_aveNPairsPooledAll_allSess = np.full((num_sessions, num_frs), np.nan) # size: num_sess x num_frames x 4 # grand average of all neuron pairs within V1, regardless of their layer
        # to get the equivalent of above pooled vars, we need to run the original code again on the cluster!
        p12_aveNPairsPooled_allSess = np.full((num_sessions, num_frs, num_depth), np.nan) # size: num_sess x num_frames x 4 # cc of each V1 layer with all neuron pairs of LM
        p12_aveNPairsPooledAll_allSess = np.full((num_sessions, num_frs), np.nan) # size: num_sess x num_frames x 4 # grand average of all neuron pairs across the two areas, regardless of their layer
        p11_aveNPairsPooled_allSess = np.full((num_sessions, num_frs, num_depth), np.nan) # size: num_sess x num_frames x 4 # cc of each LM layer with all neuron pairs of LM
        p11_aveNPairsPooledAll_allSess = np.full((num_sessions, num_frs), np.nan) # size: num_sess x num_frames x 4 # grand average of all neuron pairs within LM, regardless of their layer        
        p22_aveNPairsPooled_allSess = np.full((num_sessions, num_frs, num_depth), np.nan) # size: num_sess x num_frames x 4 # cc of each V1 layer with all neuron pairs of V1
        p22_aveNPairsPooledAll_allSess = np.full((num_sessions, num_frs), np.nan) # size: num_sess x num_frames x 4 # grand average of all neuron pairs within V1, regardless of their layer
        
        p12_aveNPairsPooled_allSess_shfl = np.full((num_sessions, num_frs, num_depth), np.nan) # size: num_sess x num_frames x 4 # cc of each V1 layer with all neuron pairs of LM
        p12_aveNPairsPooledAll_allSess_shfl = np.full((num_sessions, num_frs), np.nan) # size: num_sess x num_frames x 4 # grand average of all neuron pairs across the two areas, regardless of their layer
        p11_aveNPairsPooled_allSess_shfl = np.full((num_sessions, num_frs, num_depth), np.nan) # size: num_sess x num_frames x 4 # cc of each LM layer with all neuron pairs of LM
        p11_aveNPairsPooledAll_allSess_shfl = np.full((num_sessions, num_frs), np.nan) # size: num_sess x num_frames x 4 # grand average of all neuron pairs within LM, regardless of their layer        
        p22_aveNPairsPooled_allSess_shfl = np.full((num_sessions, num_frs, num_depth), np.nan) # size: num_sess x num_frames x 4 # cc of each V1 layer with all neuron pairs of V1
        p22_aveNPairsPooledAll_allSess_shfl = np.full((num_sessions, num_frs), np.nan) # size: num_sess x num_frames x 4 # grand average of all neuron pairs within V1, regardless of their layer
        
        flash_omit_dur_all_allSess = []
        flash_omit_dur_fr_all_allSess = []
        
        # get cc and p data for each session
        for ise in range(num_sessions): # ise=0 # loop through all sessions of this mouse

            flash_omit_dur_all_allSess.append(all_sess_2an_this_mouse['flash_omit_dur_all'].iloc[ise])
            flash_omit_dur_fr_all_allSess.append(all_sess_2an_this_mouse['flash_omit_dur_fr_all'].iloc[ise])
            
            n_neurons = all_sess_2an_this_mouse['n_neurons'].iloc[ise] # num_planes; number of neurons per plane.
            n_neurs = np.reshape(n_neurons, (2, num_depth)) # 2 x num_depth; number of neurons for each area (each layer).
            
            cc_a12 = all_sess_2an_this_mouse['cc_a12'].iloc[ise] # size: 16 (num_layerPairs between area 1 and 2); # each element: num_frs x (nu1*nu2)   
            p_a12 = all_sess_2an_this_mouse['p_a12'].iloc[ise]

            cc_a11 = all_sess_2an_this_mouse['cc_a11'].iloc[ise] # size: 10 (num_layerPairs between area 1 and 1); # each element: num_frs x (nu1*nu2)   
            p_a11 = all_sess_2an_this_mouse['p_a11'].iloc[ise]
            
            cc_a22 = all_sess_2an_this_mouse['cc_a22'].iloc[ise] # size: 10 (num_layerPairs between area 2 and 2); # each element: num_frs x (nu1*nu2)   
            p_a22 = all_sess_2an_this_mouse['p_a22'].iloc[ise]
                
            # shfl p (same dimensions as actual data (cc and p))             
            p_a12_shfl = all_sess_2an_this_mouse['p_a12_shfl'].iloc[ise]  # size: 16 (num_layerPairs between area 1 and 2); # each element: num_frs x (nu1*nu2)
            p_a11_shfl = all_sess_2an_this_mouse['p_a11_shfl'].iloc[ise]            
            p_a22_shfl = all_sess_2an_this_mouse['p_a22_shfl'].iloc[ise]
            
            # shfl cc (remember these are already averaged across neuron pairs and shuffles)
            # all we do below is that we turn it to nan if number of neurons is below a threshold
            cc12_aveNPairs_shfl = all_sess_2an_this_mouse['cc_a12_shfl'].iloc[ise] # size: num_frs x 16   
            cc11_aveNPairs_shfl = all_sess_2an_this_mouse['cc_a11_shfl'].iloc[ise] # size: num_frs x 10
            cc22_aveNPairs_shfl = all_sess_2an_this_mouse['cc_a22_shfl'].iloc[ise] # size: num_frs x 10

            
            
            ######### Loop through each layer combination, and compute average of corrs across all neuron pairs.
            cc12_aveNPairs = np.full((num_frs, len(layerPairs_a12)), np.nan) # size: num_frames x 16
            p12_aveNPairs = np.full((num_frs, len(layerPairs_a12)), np.nan) # size: num_frames x 16

            cc11_aveNPairs = np.full((num_frs, len(layerPairs_aSame)), np.nan) # size: num_frames x 10
            p11_aveNPairs = np.full((num_frs, len(layerPairs_aSame)), np.nan) # size: num_frames x 10

            cc22_aveNPairs = np.full((num_frs, len(layerPairs_aSame)), np.nan) # size: num_frames x 10
            p22_aveNPairs = np.full((num_frs, len(layerPairs_aSame)), np.nan) # size: num_frames x 10

            p12_aveNPairs_shfl = np.full((num_frs, len(layerPairs_a12)), np.nan) # size: num_frames x 16
            p11_aveNPairs_shfl = np.full((num_frs, len(layerPairs_aSame)), np.nan) # size: num_frames x 10
            p22_aveNPairs_shfl = np.full((num_frs, len(layerPairs_aSame)), np.nan) # size: num_frames x 10
            
            ################
            ##### cc12 #####
            ################
            for ilc in range(len(layerPairs_a12)): # ilc=0
                layer_a12 = [int(layerPairs_a12[ilc][1]), int(layerPairs_a12[ilc][2])] # what layers from each area are included in the layerPairs
                if ((n_neurs[0, layer_a12[0]] >= th_neurons) + (n_neurs[1, layer_a12[1]] >= th_neurons))==2 :  # make sure both layers have >th_neurons neurons
                    # ave cc across pairs
                    cc12_aveNPairs[:,ilc] = np.nanmean(cc_a12[ilc], axis=1) # num_frames
                    # fraction of neuron pairs with significant corrcoeff values (ie their corrcoef is higher than an uncorrelated dataset)
                    p12_aveNPairs[:,ilc] = np.mean(p_a12[ilc]<=alpha_sig, axis=1) # num_frames
                    # same as above, but significance is computed by running ttest_1sample between the actuall cc and the shuffled cc (above p comes from spearman package)
                    p12_aveNPairs_shfl[:,ilc] = np.mean(p_a12_shfl[ilc]<=alpha_sig, axis=1) # num_frames
                    
                else:
                    cc12_aveNPairs_shfl[:,ilc] = np.full((num_frs), np.nan)
                    print('Too few neurons in layer pair %s:\n%s' %(str(layer_a12), str(n_neurs)))
#             plt.imshow(cc12_aveNPairs, aspect='auto')
            cc12_aveNPairs_allSess[ise] = cc12_aveNPairs
            p12_aveNPairs_allSess[ise] = p12_aveNPairs
            cc12_aveNPairs_allSess_shfl[ise] = cc12_aveNPairs_shfl
            p12_aveNPairs_allSess_shfl[ise] = p12_aveNPairs_shfl


            
            ##### pool neuron pairs: v1 each layer and LM all layers         
            # you can do the same for within v1 and within LM corrs, but you first have to make the corr matrices symmetric.
            cc_Veach_Lall = []
            p_Veach_Lall = []
            p_Veach_Lall_shfl = []
            cc12_aveNPairsPooled = np.full((num_frs, num_depth), np.nan) # average cc of each layer of V1 with all neuron pairs in all layers of LM
            p12_aveNPairsPooled = np.full((num_frs, num_depth), np.nan)
            p12_aveNPairsPooled_shfl = np.full((num_frs, num_depth), np.nan)
            
            for vl in range(num_depth): # vl=0
                cc_Veach_Lall0 = cc_a12[vl]
                p_Veach_Lall0 = p_a12[vl]
                p_Veach_Lall0_shfl = p_a12_shfl[vl]
                
                for ll in np.arange(vl+num_depth, len(layerPairs_a12), num_depth): # each V1 layer with all LM layers
                    cc_Veach_Lall0 = np.column_stack((cc_Veach_Lall0, cc_a12[ll]))
                    p_Veach_Lall0 = np.column_stack((p_Veach_Lall0, p_a12[ll]))
                    p_Veach_Lall0_shfl = np.column_stack((p_Veach_Lall0_shfl, p_a12_shfl[ll]))
                print(cc_Veach_Lall0.shape) # num_frs x num_pooled_neuron_pairs
                
                cc12_aveNPairsPooled[:,vl] = np.nanmean(cc_Veach_Lall0, axis=1) # num_frs x 4 # average across pooled neuron pairs
                if np.sum(~np.isnan(p_Veach_Lall0))>0:
                    p12_aveNPairsPooled[:,vl] = np.mean(p_Veach_Lall0<=alpha_sig, axis=1)
                    p12_aveNPairsPooled_shfl[:,vl] = np.mean(p_Veach_Lall0_shfl<=alpha_sig, axis=1)
                
                cc_Veach_Lall.append(cc_Veach_Lall0)
                p_Veach_Lall.append(p_Veach_Lall0)
                p_Veach_Lall_shfl.append(p_Veach_Lall0_shfl)
                
            cc12_aveNPairsPooled_allSess[ise] = cc12_aveNPairsPooled    
            p12_aveNPairsPooled_allSess[ise] = p12_aveNPairsPooled    
            p12_aveNPairsPooled_allSess_shfl[ise] = p12_aveNPairsPooled_shfl
            
            
            
            
            ##### pool neuron pairs from all layers, to get a grand cc average between v1 and LM, regardless of layer (to compare it between flash and omission)
            cc12_aveNPairsPooledAll_allSess[ise] = np.nanmean(np.column_stack((cc_Veach_Lall)), axis=1) # grand average of cc of all neurons pairs across two areas regardless of their layer 
            p12_aveNPairsPooledAll_allSess[ise] = np.mean(np.column_stack((p_Veach_Lall))<=alpha_sig, axis=1)
            p12_aveNPairsPooledAll_allSess_shfl[ise] = np.mean(np.column_stack((p_Veach_Lall_shfl))<=alpha_sig, axis=1)
            
            
            
            
            
            ################
            ##### cc11 #####
            ################
            for ilc in range(len(layerPairs_aSame)): # ilc=0
                layer_aSame = [int(layerPairs_aSame[ilc][1]), int(layerPairs_aSame[ilc][2])] # what layers from each area are included in the layerPairs
                if ((n_neurs[0, layer_aSame[0]] >= th_neurons) + (n_neurs[0, layer_aSame[1]] >= th_neurons))==2 :  # make sure both layers have >th_neurons neurons
                    # ave cc across pairs
                    cc11_aveNPairs[:,ilc] = np.nanmean(cc_a11[ilc], axis=1) # num_frames
                    # fraction of pairs that have sig p
                    p11_aveNPairs[:,ilc] = np.mean(p_a11[ilc]<=.05, axis=1) # num_frames
                    # same as above, but significance is computed by running ttest_1sample between the actuall cc and the shuffled cc (above p comes from spearman package)
                    p11_aveNPairs_shfl[:,ilc] = np.mean(p_a11_shfl[ilc]<=alpha_sig, axis=1) # num_frames
                    
                else:
                    cc11_aveNPairs_shfl[:,ilc] = np.full((num_frs), np.nan)                    
                    print('Too few neurons in layer pair %s: %s' %(str(layer_aSame), str(n_neurs[0])))
#             plt.imshow(cc11_aveNPairs, aspect='auto')        
            cc11_aveNPairs_allSess[ise] = cc11_aveNPairs
            p11_aveNPairs_allSess[ise] = p11_aveNPairs
            cc11_aveNPairs_allSess_shfl[ise] = cc11_aveNPairs_shfl
            p11_aveNPairs_allSess_shfl[ise] = p11_aveNPairs_shfl

            

            ##### pool neuron pairs: LM each layer and all the other layers        
            cc_Veach_Lall = []
            p_Veach_Lall = []
            p_Veach_Lall_shfl = []
            cc11_aveNPairsPooled = np.full((num_frs, num_depth), np.nan) # average cc of each layer of LM with all neuron pairs in all other layers (of LM)
            p11_aveNPairsPooled = np.full((num_frs, num_depth), np.nan)
            p11_aveNPairsPooled_shfl = np.full((num_frs, num_depth), np.nan)
            
            # you should hard code this... but for now lets do this (they way to hard code is to look at layerPairs_aSame, turn it to a pair of numbers, reverse their order, and get the appropriate ones.)                        
            l0 = np.column_stack((cc_a11[0:4])) # pairs consisting of neurons in layer 0 and neurons in all other layers
            cc11_aveNPairsPooled[:,0] = np.nanmean(l0, axis=1) # num_frs x 4 # average across pooled neuron pairs
            cc_Veach_Lall.append(l0)
            
            l0 = np.column_stack((p_a11[0:4])) # pairs consisting of neurons in layer 0 and neurons in all other layers
            if np.sum(~np.isnan(l0))>0:
                p11_aveNPairsPooled[:,0] = np.mean(l0<=alpha_sig, axis=1) # num_frs x 4 # average across pooled neuron pairs
            p_Veach_Lall.append(l0)
            
            l0 = np.column_stack((p_a11_shfl[0:4])) # pairs consisting of neurons in layer 0 and neurons in all other layers
            if np.sum(~np.isnan(l0))>0:
                p11_aveNPairsPooled_shfl[:,0] = np.mean(l0<=alpha_sig, axis=1) # num_frs x 4 # average across pooled neuron pairs
            p_Veach_Lall_shfl.append(l0)
#            print(l0.shape)
            
    
            l1 = np.column_stack((cc_a11[1], cc_a11[4], cc_a11[5], cc_a11[6])) # pairs consisting of neurons in layer 1 and neurons in all other layers
            cc11_aveNPairsPooled[:,1] = np.nanmean(l1, axis=1)
            cc_Veach_Lall.append(l1)

            l1 = np.column_stack((p_a11[1], p_a11[4], p_a11[5], p_a11[6])) # pairs consisting of neurons in layer 1 and neurons in all other layers
            if np.sum(~np.isnan(l1))>0:
                p11_aveNPairsPooled[:,1] = np.mean(l1<=alpha_sig, axis=1)
            p_Veach_Lall.append(l1)

            l1 = np.column_stack((p_a11_shfl[1], p_a11_shfl[4], p_a11_shfl[5], p_a11_shfl[6])) # pairs consisting of neurons in layer 1 and neurons in all other layers
            if np.sum(~np.isnan(l1))>0:
                p11_aveNPairsPooled_shfl[:,1] = np.mean(l1<=alpha_sig, axis=1)
            p_Veach_Lall_shfl.append(l1)
#            print(l1.shape)
            
    
            l2 = np.column_stack((cc_a11[2], cc_a11[5], cc_a11[7], cc_a11[8])) # pairs consisting of neurons in layer 2 and neurons in all other layers
            cc11_aveNPairsPooled[:,2] = np.nanmean(l2, axis=1)
            cc_Veach_Lall.append(l2)

            l2 = np.column_stack((p_a11[2], p_a11[5], p_a11[7], p_a11[8])) # pairs consisting of neurons in layer 2 and neurons in all other layers
            if np.sum(~np.isnan(l2))>0:
                p11_aveNPairsPooled[:,2] = np.mean(l2<=alpha_sig, axis=1)
            p_Veach_Lall.append(l2)

            l2 = np.column_stack((p_a11_shfl[2], p_a11_shfl[5], p_a11_shfl[7], p_a11_shfl[8])) # pairs consisting of neurons in layer 2 and neurons in all other layers
            if np.sum(~np.isnan(l2))>0:
                p11_aveNPairsPooled_shfl[:,2] = np.mean(l2<=alpha_sig, axis=1)
            p_Veach_Lall_shfl.append(l2)
#            print(l2.shape)


            l3 = np.column_stack((cc_a11[3], cc_a11[6], cc_a11[8], cc_a11[9])) # pairs consisting of neurons in layer 3 and neurons in all other layers
            cc11_aveNPairsPooled[:,3] = np.nanmean(l3, axis=1)
            cc_Veach_Lall.append(l3)

            l3 = np.column_stack((p_a11[3], p_a11[6], p_a11[8], p_a11[9])) # pairs consisting of neurons in layer 3 and neurons in all other layers
            if np.sum(~np.isnan(l3))>0:
                p11_aveNPairsPooled[:,3] = np.mean(l3<=alpha_sig, axis=1)
            p_Veach_Lall.append(l3)

            l3 = np.column_stack((p_a11_shfl[3], p_a11_shfl[6], p_a11_shfl[8], p_a11_shfl[9])) # pairs consisting of neurons in layer 3 and neurons in all other layers
            if np.sum(~np.isnan(l3))>0:
                p11_aveNPairsPooled_shfl[:,3] = np.mean(l3<=alpha_sig, axis=1)
            p_Veach_Lall_shfl.append(l3)
#            print(l3.shape)


            cc11_aveNPairsPooled_allSess[ise] = cc11_aveNPairsPooled    
            p11_aveNPairsPooled_allSess[ise] = p11_aveNPairsPooled    
            p11_aveNPairsPooled_allSess_shfl[ise] = p11_aveNPairsPooled_shfl    

            
            ##### pool neuron pairs from all layers, to get a grand cc average between v1 and LM, regardless of layer (to compare it between flash and omission)
            cc11_aveNPairsPooledAll_allSess[ise] = np.nanmean(np.column_stack((cc_Veach_Lall)), axis=1) # grand average of cc of all neurons pairs within LM regardless of their layer 
            p11_aveNPairsPooledAll_allSess[ise] = np.mean(np.column_stack((p_Veach_Lall))<=alpha_sig, axis=1)
            p11_aveNPairsPooledAll_allSess_shfl[ise] = np.mean(np.column_stack((p_Veach_Lall_shfl))<=alpha_sig, axis=1)

            
            
            ################
            ##### cc22 #####
            ################
            for ilc in range(len(layerPairs_aSame)): # ilc=0
                layer_aSame = [int(layerPairs_aSame[ilc][1]), int(layerPairs_aSame[ilc][2])] # what layers from each area are included in the layerPairs
                if ((n_neurs[1, layer_aSame[0]] >= th_neurons) + (n_neurs[1, layer_aSame[1]] >= th_neurons))==2 :  # make sure both layers have >th_neurons neurons
                    # ave cc across pairs
                    cc22_aveNPairs[:,ilc] = np.nanmean(cc_a22[ilc], axis=1) # num_frames
                    # fraction of pairs that have sig p
                    p22_aveNPairs[:,ilc] = np.mean(p_a22[ilc]<=alpha_sig, axis=1) # num_frames
                    # same as above, but significance is computed by running ttest_1sample between the actuall cc and the shuffled cc (above p comes from spearman package)
                    p22_aveNPairs_shfl[:,ilc] = np.mean(p_a22_shfl[ilc]<=alpha_sig, axis=1) # num_frames
                    
                else:
                    cc22_aveNPairs_shfl[:,ilc] = np.full((num_frs), np.nan)                                        
                    print('Too few neurons in layer pair %s: %s' %(str(layer_aSame), str(n_neurs[1])))
#             plt.imshow(cc22_aveNPairs, aspect='auto')        
            cc22_aveNPairs_allSess[ise] = cc22_aveNPairs
            p22_aveNPairs_allSess[ise] = p22_aveNPairs
            cc22_aveNPairs_allSess_shfl[ise] = cc22_aveNPairs_shfl
            p22_aveNPairs_allSess_shfl[ise] = p22_aveNPairs_shfl


            ##### pool neuron pairs: V1 each layer and all the other layers        
            cc_Veach_Lall = []
            p_Veach_Lall = []
            p_Veach_Lall_shfl = []
            cc22_aveNPairsPooled = np.full((num_frs, num_depth), np.nan) # average cc of each layer of LM with all neuron pairs in all other layers (of LM)
            p22_aveNPairsPooled = np.full((num_frs, num_depth), np.nan)
            p22_aveNPairsPooled_shfl = np.full((num_frs, num_depth), np.nan)
            
            # you should hard code this... but for now lets do this (they way to hard code is to look at layerPairs_aSame, turn it to a pair of numbers, reverse their order, and get the appropriate ones.)                        
            l0 = np.column_stack((cc_a22[0:4])) # pairs consisting of neurons in layer 0 and neurons in all other layers
            cc22_aveNPairsPooled[:,0] = np.nanmean(l0, axis=1) # num_frs x 4 # average across pooled neuron pairs
            cc_Veach_Lall.append(l0)
            
            l0 = np.column_stack((p_a22[0:4])) # pairs consisting of neurons in layer 0 and neurons in all other layers
            if np.sum(~np.isnan(l0))>0:
                p22_aveNPairsPooled[:,0] = np.mean(l0<=alpha_sig, axis=1) # num_frs x 4 # average across pooled neuron pairs
            p_Veach_Lall.append(l0)
            
            l0 = np.column_stack((p_a22_shfl[0:4])) # pairs consisting of neurons in layer 0 and neurons in all other layers
            if np.sum(~np.isnan(l0))>0:
                p22_aveNPairsPooled_shfl[:,0] = np.mean(l0<=alpha_sig, axis=1) # num_frs x 4 # average across pooled neuron pairs
            p_Veach_Lall_shfl.append(l0)
#            print(l0.shape)
            
    
            l1 = np.column_stack((cc_a22[1], cc_a22[4], cc_a22[5], cc_a22[6])) # pairs consisting of neurons in layer 1 and neurons in all other layers
            cc22_aveNPairsPooled[:,1] = np.nanmean(l1, axis=1)
            cc_Veach_Lall.append(l1)

            l1 = np.column_stack((p_a22[1], p_a22[4], p_a22[5], p_a22[6])) # pairs consisting of neurons in layer 1 and neurons in all other layers
            if np.sum(~np.isnan(l1))>0:
                p22_aveNPairsPooled[:,1] = np.mean(l1<=alpha_sig, axis=1)
            p_Veach_Lall.append(l1)

            l1 = np.column_stack((p_a22_shfl[1], p_a22_shfl[4], p_a22_shfl[5], p_a22_shfl[6])) # pairs consisting of neurons in layer 1 and neurons in all other layers
            if np.sum(~np.isnan(l1))>0:
                p22_aveNPairsPooled_shfl[:,1] = np.mean(l1<=alpha_sig, axis=1)
            p_Veach_Lall_shfl.append(l1)
#            print(l1.shape)
            
    
            l2 = np.column_stack((cc_a22[2], cc_a22[5], cc_a22[7], cc_a22[8])) # pairs consisting of neurons in layer 2 and neurons in all other layers
            cc22_aveNPairsPooled[:,2] = np.nanmean(l2, axis=1)
            cc_Veach_Lall.append(l2)

            l2 = np.column_stack((p_a22[2], p_a22[5], p_a22[7], p_a22[8])) # pairs consisting of neurons in layer 2 and neurons in all other layers
            if np.sum(~np.isnan(l2))>0:
                p22_aveNPairsPooled[:,2] = np.mean(l2<=alpha_sig, axis=1)
            p_Veach_Lall.append(l2)

            l2 = np.column_stack((p_a22_shfl[2], p_a22_shfl[5], p_a22_shfl[7], p_a22_shfl[8])) # pairs consisting of neurons in layer 2 and neurons in all other layers
            if np.sum(~np.isnan(l2))>0:
                p22_aveNPairsPooled_shfl[:,2] = np.mean(l2<=alpha_sig, axis=1)
            p_Veach_Lall_shfl.append(l2)
#            print(l2.shape)


            l3 = np.column_stack((cc_a22[3], cc_a22[6], cc_a22[8], cc_a22[9])) # pairs consisting of neurons in layer 3 and neurons in all other layers
            cc22_aveNPairsPooled[:,3] = np.nanmean(l3, axis=1)
            cc_Veach_Lall.append(l3)

            l3 = np.column_stack((p_a22[3], p_a22[6], p_a22[8], p_a22[9])) # pairs consisting of neurons in layer 3 and neurons in all other layers
            if np.sum(~np.isnan(l3))>0:
                p22_aveNPairsPooled[:,3] = np.mean(l3<=alpha_sig, axis=1)
            p_Veach_Lall.append(l3)

            l3 = np.column_stack((p_a22_shfl[3], p_a22_shfl[6], p_a22_shfl[8], p_a22_shfl[9])) # pairs consisting of neurons in layer 3 and neurons in all other layers
            if np.sum(~np.isnan(l3))>0:
                p22_aveNPairsPooled_shfl[:,3] = np.mean(l3<=alpha_sig, axis=1)
            p_Veach_Lall_shfl.append(l3)
#            print(l3.shape)

            cc22_aveNPairsPooled_allSess[ise] = cc22_aveNPairsPooled    
            p22_aveNPairsPooled_allSess[ise] = p22_aveNPairsPooled    
            p22_aveNPairsPooled_allSess_shfl[ise] = p22_aveNPairsPooled_shfl    

            
            
            ##### pool neuron pairs from all layers, to get a grand cc average within V1, regardless of layer (to compare it between flash and omission)
            cc22_aveNPairsPooledAll_allSess[ise] = np.nanmean(np.column_stack((cc_Veach_Lall)), axis=1) # grand average of cc of all neurons pairs within LM regardless of their layer 
            p22_aveNPairsPooledAll_allSess[ise] = np.mean(np.column_stack((p_Veach_Lall))<=alpha_sig, axis=1)
            p22_aveNPairsPooledAll_allSess_shfl[ise] = np.mean(np.column_stack((p_Veach_Lall_shfl))<=alpha_sig, axis=1)


        '''
        ##%% For each session, compute average cc of each layer with all other layers
        # another way to do this is to pool cc of neuron pairs across all layers (cc_a12, pool across np.arange(0,16,4))
        # this will dominated with depth 1 or so because they have more neurons.... also you will need to run the code on cluster again to get the shuffled values
        
        for ise in range(num_sessions): # ise=0 # loop through all sessions of this mouse
            cc12_aveNPairs_allSess_44 = np.reshape(cc12_aveNPairs_allSess[ise], (num_frs,num_depth, num_depth), order='C') # 80 x 4 x 4
            # average of each layer of V1 with all LM layers
            a = np.nanmean(cc12_aveNPairs_allSess_44, axis=2) # num_frs x 4 (V1 layers) # average across LM layers (ie average each column)
        '''
            
        
        
        
        
        #%% Average across sessions for each layer pair
        # remember: baselines are not subtracted from the traces; but when we quantify the peak we subtract the baseline if sameBl_allLayerPairs is set to 1.

        ##### cc12 #####
        num_valid_sess_12 = num_sessions - np.sum(np.isnan(cc12_aveNPairs_allSess[:,0,:]), axis=0)
        print(num_valid_sess_12)
        cc12_sessAv = np.nanmean(cc12_aveNPairs_allSess, axis=0) # num_frames x 16
        cc12_sessSd = np.nanstd(cc12_aveNPairs_allSess, axis=0) / np.sqrt(num_valid_sess_12)
        p12_sessAv = np.nanmean(p12_aveNPairs_allSess, axis=0) # num_frames x 16
        p12_sessSd = np.nanstd(p12_aveNPairs_allSess, axis=0) / np.sqrt(num_valid_sess_12)

        cc12_sessAv_shfl = np.nanmean(cc12_aveNPairs_allSess_shfl, axis=0) # num_frames x 16
        cc12_sessSd_shfl = np.nanstd(cc12_aveNPairs_allSess_shfl, axis=0) / np.sqrt(num_valid_sess_12)
        p12_sessAv_shfl = np.nanmean(p12_aveNPairs_allSess_shfl, axis=0) # num_frames x 16
        p12_sessSd_shfl = np.nanstd(p12_aveNPairs_allSess_shfl, axis=0) / np.sqrt(num_valid_sess_12)

        # pooled
        nv_pooled_12 = num_sessions - np.sum(np.isnan(cc12_aveNPairsPooled_allSess[:,0,:]), axis=0)
        cc12_pooled_sessAv = np.nanmean(cc12_aveNPairsPooled_allSess, axis=0) # num_frames x 4
        cc12_pooled_sessSd = np.nanstd(cc12_aveNPairsPooled_allSess, axis=0) / np.sqrt(nv_pooled_12)
        p12_pooled_sessAv = np.nanmean(p12_aveNPairsPooled_allSess, axis=0) # num_frames x 4
        p12_pooled_sessSd = np.nanstd(p12_aveNPairsPooled_allSess, axis=0) / np.sqrt(nv_pooled_12)
        p12_pooled_sessAv_shfl = np.nanmean(p12_aveNPairsPooled_allSess_shfl, axis=0) # num_frames x 4
        p12_pooled_sessSd_shfl = np.nanstd(p12_aveNPairsPooled_allSess_shfl, axis=0) / np.sqrt(nv_pooled_12)

        nv_pooledAll_12 = num_sessions - np.sum(np.isnan(cc12_aveNPairsPooledAll_allSess[:,0]), axis=0)
        cc12_pooledAll_sessAv = np.nanmean(cc12_aveNPairsPooledAll_allSess, axis=0) # num_frames
        cc12_pooledAll_sessSd = np.nanstd(cc12_aveNPairsPooledAll_allSess, axis=0)/ np.sqrt(nv_pooledAll_12)
        p12_pooledAll_sessAv = np.nanmean(p12_aveNPairsPooledAll_allSess, axis=0) # num_frames
        p12_pooledAll_sessSd = np.nanstd(p12_aveNPairsPooledAll_allSess, axis=0)/ np.sqrt(nv_pooledAll_12)
        p12_pooledAll_sessAv_shfl = np.nanmean(p12_aveNPairsPooledAll_allSess_shfl, axis=0) # num_frames
        p12_pooledAll_sessSd_shfl = np.nanstd(p12_aveNPairsPooledAll_allSess_shfl, axis=0)/ np.sqrt(nv_pooledAll_12)


        ##### cc11 #####
        num_valid_sess_11 = num_sessions - np.sum(np.isnan(cc11_aveNPairs_allSess[:,0,:]), axis=0)
        print(num_valid_sess_11)
        cc11_sessAv = np.nanmean(cc11_aveNPairs_allSess, axis=0) # num_frames x 10
        cc11_sessSd = np.nanstd(cc11_aveNPairs_allSess, axis=0) / np.sqrt(num_valid_sess_11)
        p11_sessAv = np.nanmean(p11_aveNPairs_allSess, axis=0) # num_frames x 10
        p11_sessSd = np.nanstd(p11_aveNPairs_allSess, axis=0) / np.sqrt(num_valid_sess_11)

        cc11_sessAv_shfl = np.nanmean(cc11_aveNPairs_allSess_shfl, axis=0) # num_frames x 10
        cc11_sessSd_shfl = np.nanstd(cc11_aveNPairs_allSess_shfl, axis=0) / np.sqrt(num_valid_sess_11)
        p11_sessAv_shfl = np.nanmean(p11_aveNPairs_allSess_shfl, axis=0) # num_frames x 10
        p11_sessSd_shfl = np.nanstd(p11_aveNPairs_allSess_shfl, axis=0) / np.sqrt(num_valid_sess_11)
        
        # pooled
        nv_pooled_11 = num_sessions - np.sum(np.isnan(cc11_aveNPairsPooled_allSess[:,0,:]), axis=0)
        cc11_pooled_sessAv = np.nanmean(cc11_aveNPairsPooled_allSess, axis=0) # num_frames x 4
        cc11_pooled_sessSd = np.nanstd(cc11_aveNPairsPooled_allSess, axis=0) / np.sqrt(nv_pooled_11)
        p11_pooled_sessAv = np.nanmean(p11_aveNPairsPooled_allSess, axis=0) # num_frames x 4
        p11_pooled_sessSd = np.nanstd(p11_aveNPairsPooled_allSess, axis=0) / np.sqrt(nv_pooled_11)
        p11_pooled_sessAv_shfl = np.nanmean(p11_aveNPairsPooled_allSess_shfl, axis=0) # num_frames x 4
        p11_pooled_sessSd_shfl = np.nanstd(p11_aveNPairsPooled_allSess_shfl, axis=0) / np.sqrt(nv_pooled_11)
        
        nv_pooledAll_11 = num_sessions - np.sum(np.isnan(cc11_aveNPairsPooledAll_allSess[:,0]), axis=0)
        cc11_pooledAll_sessAv = np.nanmean(cc11_aveNPairsPooledAll_allSess, axis=0) # num_frames
        cc11_pooledAll_sessSd = np.nanstd(cc11_aveNPairsPooledAll_allSess, axis=0)/ np.sqrt(nv_pooledAll_11)
        p11_pooledAll_sessAv = np.nanmean(p11_aveNPairsPooledAll_allSess, axis=0) # num_frames
        p11_pooledAll_sessSd = np.nanstd(p11_aveNPairsPooledAll_allSess, axis=0)/ np.sqrt(nv_pooledAll_11)
        p11_pooledAll_sessAv_shfl = np.nanmean(p11_aveNPairsPooledAll_allSess_shfl, axis=0) # num_frames
        p11_pooledAll_sessSd_shfl = np.nanstd(p11_aveNPairsPooledAll_allSess_shfl, axis=0)/ np.sqrt(nv_pooledAll_11)

        
        ##### cc22 #####
        num_valid_sess_22 = num_sessions - np.sum(np.isnan(cc22_aveNPairs_allSess[:,0,:]), axis=0)
        print(num_valid_sess_22)
        cc22_sessAv = np.nanmean(cc22_aveNPairs_allSess, axis=0) # num_frames x 10
        cc22_sessSd = np.nanstd(cc22_aveNPairs_allSess, axis=0) / np.sqrt(num_valid_sess_22)
        p22_sessAv = np.nanmean(p22_aveNPairs_allSess, axis=0) # num_frames x 10
        p22_sessSd = np.nanstd(p22_aveNPairs_allSess, axis=0) / np.sqrt(num_valid_sess_22)

        cc22_sessAv_shfl = np.nanmean(cc22_aveNPairs_allSess_shfl, axis=0) # num_frames x 10
        cc22_sessSd_shfl = np.nanstd(cc22_aveNPairs_allSess_shfl, axis=0) / np.sqrt(num_valid_sess_22)
        p22_sessAv_shfl = np.nanmean(p22_aveNPairs_allSess_shfl, axis=0) # num_frames x 10
        p22_sessSd_shfl = np.nanstd(p22_aveNPairs_allSess_shfl, axis=0) / np.sqrt(num_valid_sess_22)

        # pooled
        nv_pooled_22 = num_sessions - np.sum(np.isnan(cc22_aveNPairsPooled_allSess[:,0,:]), axis=0)
        cc22_pooled_sessAv = np.nanmean(cc22_aveNPairsPooled_allSess, axis=0) # num_frames x 4
        cc22_pooled_sessSd = np.nanstd(cc22_aveNPairsPooled_allSess, axis=0) / np.sqrt(nv_pooled_22)
        p22_pooled_sessAv = np.nanmean(p22_aveNPairsPooled_allSess, axis=0) # num_frames x 4
        p22_pooled_sessSd = np.nanstd(p22_aveNPairsPooled_allSess, axis=0) / np.sqrt(nv_pooled_22)
        p22_pooled_sessAv_shfl = np.nanmean(p22_aveNPairsPooled_allSess_shfl, axis=0) # num_frames x 4
        p22_pooled_sessSd_shfl = np.nanstd(p22_aveNPairsPooled_allSess_shfl, axis=0) / np.sqrt(nv_pooled_22)
        
        nv_pooledAll_22 = num_sessions - np.sum(np.isnan(cc22_aveNPairsPooledAll_allSess[:,0]), axis=0)
        cc22_pooledAll_sessAv = np.nanmean(cc22_aveNPairsPooledAll_allSess, axis=0) # num_frames
        cc22_pooledAll_sessSd = np.nanstd(cc22_aveNPairsPooledAll_allSess, axis=0)/ np.sqrt(nv_pooledAll_22)
        p22_pooledAll_sessAv = np.nanmean(p22_aveNPairsPooledAll_allSess, axis=0) # num_frames
        p22_pooledAll_sessSd = np.nanstd(p22_aveNPairsPooledAll_allSess, axis=0)/ np.sqrt(nv_pooledAll_22)
        p22_pooledAll_sessAv_shfl = np.nanmean(p22_aveNPairsPooledAll_allSess_shfl, axis=0) # num_frames
        p22_pooledAll_sessSd_shfl = np.nanstd(p22_aveNPairsPooledAll_allSess_shfl, axis=0)/ np.sqrt(nv_pooledAll_22)
        
#         plt.imshow(cc12_sessAv, aspect='auto')
#         plt.imshow(p11_sessAv, aspect='auto')        
#         plt.imshow(p22_sessAv, aspect='auto')
        
#         plt.subplot(121); plt.plot(time_trace, cc12_sessAv); plt.subplot(122); plt.plot(time_trace, p12_sessAv); plt.subplots_adjust(wspace=.5)

#        plt.figure()
#        plt.plot(time_trace, cc12_sessAv)
#        plot_flashLines_ticks_legend([], [], flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
#        plt.title([mouse_id, cre, num_sessions])
        
        
        #%% Reshape traces to 4 x 4
        
        # in cols_a1a2: 
        # rows are area1 (LM) layers; colums are area2 (V1) layers
        # cols = cc_a1_a2.columns.values
        # cols_a1a2 = np.reshape(cols, (4,4), order='C')
        #            cols_a1a2 = [['l00', 'l01', 'l02', 'l03'],
        #                        ['l10', 'l11', 'l12', 'l13'],
        #                        ['l20', 'l21', 'l22', 'l23'],
        #                        ['l30', 'l31', 'l32', 'l33']]
        
        # remember: baselines are not subtracted from the traces; but when we quantify the peak we subtract the baseline if sameBl_allLayerPairs is set to 1.
        
        num_valid_sess_12_44 = np.reshape(num_valid_sess_12, (num_depth, num_depth), order='C') # 4 x 4
        num_valid_sess_11_44 = traces_44_layerSame(num_valid_sess_11, num_depth)[0] # 4 x 4
        num_valid_sess_22_44 = traces_44_layerSame(num_valid_sess_22, num_depth)[0] # 4 x 4
        
        
        # cc12
        cc12_sessAv_44 = np.reshape(cc12_sessAv, (num_frs,num_depth, num_depth), order='C') # 80 x 4 x 4
        cc12_sessSd_44 = np.reshape(cc12_sessSd, (num_frs,num_depth, num_depth), order='C') # 80 x 4 x 4
        p12_sessAv_44 = np.reshape(p12_sessAv, (num_frs,num_depth, num_depth), order='C') # 80 x 4 x 4
        p12_sessSd_44 = np.reshape(p12_sessSd, (num_frs,num_depth, num_depth), order='C') # 80 x 4 x 4

        # cc11
        cc11_sessAv_44 = traces_44_layerSame(cc11_sessAv, num_depth) # 80 x 4 x 4
        cc11_sessSd_44 = traces_44_layerSame(cc11_sessSd, num_depth)
        p11_sessAv_44 = traces_44_layerSame(p11_sessAv, num_depth)
        p11_sessSd_44 = traces_44_layerSame(p11_sessSd, num_depth)        

        # cc22
        cc22_sessAv_44 = traces_44_layerSame(cc22_sessAv, num_depth) # 80 x 4 x 4
        cc22_sessSd_44 = traces_44_layerSame(cc22_sessSd, num_depth)
        p22_sessAv_44 = traces_44_layerSame(p22_sessAv, num_depth)
        p22_sessSd_44 = traces_44_layerSame(p22_sessSd, num_depth)
        
        
        # cc12_shfl
        cc12_sessAv_44_shfl = np.reshape(cc12_sessAv_shfl, (num_frs,num_depth, num_depth), order='C') # 80 x 4 x 4
        cc12_sessSd_44_shfl = np.reshape(cc12_sessSd_shfl, (num_frs,num_depth, num_depth), order='C') # 80 x 4 x 4
        p12_sessAv_44_shfl = np.reshape(p12_sessAv_shfl, (num_frs,num_depth, num_depth), order='C') # 80 x 4 x 4
        p12_sessSd_44_shfl = np.reshape(p12_sessSd_shfl, (num_frs,num_depth, num_depth), order='C') # 80 x 4 x 4

        # cc11_shfl
        cc11_sessAv_44_shfl = traces_44_layerSame(cc11_sessAv_shfl, num_depth) # 80 x 4 x 4
        cc11_sessSd_44_shfl = traces_44_layerSame(cc11_sessSd_shfl, num_depth)
        p11_sessAv_44_shfl = traces_44_layerSame(p11_sessAv_shfl, num_depth)
        p11_sessSd_44_shfl = traces_44_layerSame(p11_sessSd_shfl, num_depth)        

        # cc22_shfl
        cc22_sessAv_44_shfl = traces_44_layerSame(cc22_sessAv_shfl, num_depth) # 80 x 4 x 4
        cc22_sessSd_44_shfl = traces_44_layerSame(cc22_sessSd_shfl, num_depth)
        p22_sessAv_44_shfl = traces_44_layerSame(p22_sessAv_shfl, num_depth)
        p22_sessSd_44_shfl = traces_44_layerSame(p22_sessSd_shfl, num_depth)        
        
        # plot cc trace for LM layer 3, V1 layer 2
#        plt.plot(time_trace, cc12_sessAv[:, 14])
#        plt.plot(time_trace, cc12_sessAv_44[:, 3,2])

        # sanity check:
#        plt.plot(cc_sessAv[:,14]) # layerPairs[14] = 'l32' # int(layerPairs[13][1:]) = 32
#        plt.plot(cc_sessAv_a12[:,3,2]) 
                
        
        
        
        ###########################################################################
        ###########################################################################
        ###########################################################################        
        ###########################################################################
        ###########################################################################
        ###########################################################################        
        #%% Quantify corrcoeff (and p) after flash, and omission (to get the heatmap of corrs!) also reshape the arrays to 4 x 4
        # remember: when you call "quant_cc" if "sameBl_allLayerPairs" is 1, it will first subtract the baseline from each trace, and then will compute the peak.
        ###########################################################################
        ###########################################################################
        ###########################################################################
        ###########################################################################
        ###########################################################################        
        ###########################################################################
        
        # quantify peak on each session ... then show the average and sd across sessions 
        cc12_peak_amp_omit_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)
        cc12_peak_amp_flash_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)
        cc11_peak_amp_omit_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)
        cc11_peak_amp_flash_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)
        cc22_peak_amp_omit_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)
        cc22_peak_amp_flash_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)

        p12_peak_amp_omit_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)
        p12_peak_amp_flash_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)
        p11_peak_amp_omit_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)
        p11_peak_amp_flash_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)
        p22_peak_amp_omit_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)
        p22_peak_amp_flash_eachSess = np.full((num_sessions, num_depth, num_depth), np.nan)

        cc12_peak_amp_omit_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        cc12_peak_amp_flash_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        cc11_peak_amp_omit_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        cc11_peak_amp_flash_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        cc22_peak_amp_omit_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        cc22_peak_amp_flash_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)

        p12_peak_amp_omit_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        p12_peak_amp_flash_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        p11_peak_amp_omit_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        p11_peak_amp_flash_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        p22_peak_amp_omit_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        p22_peak_amp_flash_eachSess_shfl = np.full((num_sessions, num_depth, num_depth), np.nan)
        
        # pooled
        cc12_peak_amp_omit_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)
        cc12_peak_amp_flash_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)
        cc11_peak_amp_omit_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)
        cc11_peak_amp_flash_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)
        cc22_peak_amp_omit_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)
        cc22_peak_amp_flash_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)

        p12_peak_amp_omit_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)
        p12_peak_amp_flash_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)
        p11_peak_amp_omit_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)
        p11_peak_amp_flash_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)
        p22_peak_amp_omit_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)
        p22_peak_amp_flash_eachSess_pooled = np.full((num_sessions, num_depth), np.nan)

        p12_peak_amp_omit_eachSess_pooled_shfl = np.full((num_sessions, num_depth), np.nan)
        p12_peak_amp_flash_eachSess_pooled_shfl = np.full((num_sessions, num_depth), np.nan)
        p11_peak_amp_omit_eachSess_pooled_shfl = np.full((num_sessions, num_depth), np.nan)
        p11_peak_amp_flash_eachSess_pooled_shfl = np.full((num_sessions, num_depth), np.nan)
        p22_peak_amp_omit_eachSess_pooled_shfl = np.full((num_sessions, num_depth), np.nan)
        p22_peak_amp_flash_eachSess_pooled_shfl = np.full((num_sessions, num_depth), np.nan)
        
        # pooledAll
        cc12_peak_amp_omit_eachSess_pooledAll = np.full((num_sessions), np.nan)
        cc12_peak_amp_flash_eachSess_pooledAll = np.full((num_sessions), np.nan)
        cc11_peak_amp_omit_eachSess_pooledAll = np.full((num_sessions), np.nan)
        cc11_peak_amp_flash_eachSess_pooledAll = np.full((num_sessions), np.nan)
        cc22_peak_amp_omit_eachSess_pooledAll = np.full((num_sessions), np.nan)
        cc22_peak_amp_flash_eachSess_pooledAll = np.full((num_sessions), np.nan)

        p12_peak_amp_omit_eachSess_pooledAll = np.full((num_sessions), np.nan)
        p12_peak_amp_flash_eachSess_pooledAll = np.full((num_sessions), np.nan)
        p11_peak_amp_omit_eachSess_pooledAll = np.full((num_sessions), np.nan)
        p11_peak_amp_flash_eachSess_pooledAll = np.full((num_sessions), np.nan)
        p22_peak_amp_omit_eachSess_pooledAll = np.full((num_sessions), np.nan)
        p22_peak_amp_flash_eachSess_pooledAll = np.full((num_sessions), np.nan)

        p12_peak_amp_omit_eachSess_pooledAll_shfl = np.full((num_sessions), np.nan)
        p12_peak_amp_flash_eachSess_pooledAll_shfl = np.full((num_sessions), np.nan)
        p11_peak_amp_omit_eachSess_pooledAll_shfl = np.full((num_sessions), np.nan)
        p11_peak_amp_flash_eachSess_pooledAll_shfl = np.full((num_sessions), np.nan)
        p22_peak_amp_omit_eachSess_pooledAll_shfl = np.full((num_sessions), np.nan)
        p22_peak_amp_flash_eachSess_pooledAll_shfl = np.full((num_sessions), np.nan)
        
        for ise in range(num_sessions): # ise=0 # loop through all sessions of this mouse
            
            fo_dur = flash_omit_dur_all_allSess[ise] # flash-omission interval of all trials. We use max of this to compute bl_index_pre_flash.
            
            #%% Set flash_win: use a different window for computing flash responses if it is VIP, and B1 session (1st novel session)
            date = session_dates[ise]
            session_novel = is_session_novel(dir_server_me, mouse_id, date) # first determine if the session is the 1st novel session or not

            # vip, except for B1, needs a different timewindow for computing image-triggered average (because its response precedes the image).
            # sst, slc, and vip B1 sessions all need a timewindow that immediately follows the images.
            if np.logical_and(cre.find('Vip')==1 , session_novel) or cre.find('Vip')==-1: # VIP B1, SST, SLC: window after images (response follows the image)
                list_times_flash_final = list_times_flash # [8,  9, 10, 11, 12, 13]

            else: # VIP (non B1, familiar sessions): window precedes the image (pre-stimulus ramping activity)
                list_times_flash_final = list_times_flash_vip # [ 5,  6,  7,  8,  9, 10]
            
            
            flash_win_final = set_flash_win_final(cre, session_novel, flash_win, flash_win_vip)
            flash_win_final0 = flash_win_final[0] + .75 # we add .75 because here flash_win_final is relative to omission (unlike in omission_xx_init code, where it is relative to flash)

            cc12_peak_amp_omit_eachSess[ise], cc12_peak_amp_flash_eachSess[ise] = quant_cc(cc12_aveNPairs_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            cc11_peak_amp_omit_eachSess[ise], cc11_peak_amp_flash_eachSess[ise] = quant_cc(cc11_aveNPairs_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            cc22_peak_amp_omit_eachSess[ise], cc22_peak_amp_flash_eachSess[ise] = quant_cc(cc22_aveNPairs_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 

            p12_peak_amp_omit_eachSess[ise], p12_peak_amp_flash_eachSess[ise] = quant_cc(p12_aveNPairs_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p11_peak_amp_omit_eachSess[ise], p11_peak_amp_flash_eachSess[ise] = quant_cc(p11_aveNPairs_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p22_peak_amp_omit_eachSess[ise], p22_peak_amp_flash_eachSess[ise] = quant_cc(p22_aveNPairs_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 

            cc12_peak_amp_omit_eachSess_shfl[ise], cc12_peak_amp_flash_eachSess_shfl[ise] = quant_cc(cc12_aveNPairs_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            cc11_peak_amp_omit_eachSess_shfl[ise], cc11_peak_amp_flash_eachSess_shfl[ise] = quant_cc(cc11_aveNPairs_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            cc22_peak_amp_omit_eachSess_shfl[ise], cc22_peak_amp_flash_eachSess_shfl[ise] = quant_cc(cc22_aveNPairs_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 

            p12_peak_amp_omit_eachSess_shfl[ise], p12_peak_amp_flash_eachSess_shfl[ise] = quant_cc(p12_aveNPairs_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p11_peak_amp_omit_eachSess_shfl[ise], p11_peak_amp_flash_eachSess_shfl[ise] = quant_cc(p11_aveNPairs_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p22_peak_amp_omit_eachSess_shfl[ise], p22_peak_amp_flash_eachSess_shfl[ise] = quant_cc(p22_aveNPairs_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            
            ### pooled (corr of neuron pairs in one layer with all other neuron pairs)
            cc12_peak_amp_omit_eachSess_pooled[ise], cc12_peak_amp_flash_eachSess_pooled[ise] = quant_cc(cc12_aveNPairsPooled_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            cc11_peak_amp_omit_eachSess_pooled[ise], cc11_peak_amp_flash_eachSess_pooled[ise] = quant_cc(cc11_aveNPairsPooled_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            cc22_peak_amp_omit_eachSess_pooled[ise], cc22_peak_amp_flash_eachSess_pooled[ise] = quant_cc(cc22_aveNPairsPooled_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            
            p12_peak_amp_omit_eachSess_pooled[ise], p12_peak_amp_flash_eachSess_pooled[ise] = quant_cc(p12_aveNPairsPooled_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p11_peak_amp_omit_eachSess_pooled[ise], p11_peak_amp_flash_eachSess_pooled[ise] = quant_cc(p11_aveNPairsPooled_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p22_peak_amp_omit_eachSess_pooled[ise], p22_peak_amp_flash_eachSess_pooled[ise] = quant_cc(p22_aveNPairsPooled_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 

            p12_peak_amp_omit_eachSess_pooled_shfl[ise], p12_peak_amp_flash_eachSess_pooled_shfl[ise] = quant_cc(p12_aveNPairsPooled_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p11_peak_amp_omit_eachSess_pooled_shfl[ise], p11_peak_amp_flash_eachSess_pooled_shfl[ise] = quant_cc(p11_aveNPairsPooled_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p22_peak_amp_omit_eachSess_pooled_shfl[ise], p22_peak_amp_flash_eachSess_pooled_shfl[ise] = quant_cc(p22_aveNPairsPooled_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 

            ### pooledAll (corr of all neuron pairs in all layers with each other)
            cc12_peak_amp_omit_eachSess_pooledAll[ise], cc12_peak_amp_flash_eachSess_pooledAll[ise] = quant_cc(cc12_aveNPairsPooledAll_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            cc11_peak_amp_omit_eachSess_pooledAll[ise], cc11_peak_amp_flash_eachSess_pooledAll[ise] = quant_cc(cc11_aveNPairsPooledAll_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            cc22_peak_amp_omit_eachSess_pooledAll[ise], cc22_peak_amp_flash_eachSess_pooledAll[ise] = quant_cc(cc22_aveNPairsPooledAll_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            
            p12_peak_amp_omit_eachSess_pooledAll[ise], p12_peak_amp_flash_eachSess_pooledAll[ise] = quant_cc(p12_aveNPairsPooledAll_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p11_peak_amp_omit_eachSess_pooledAll[ise], p11_peak_amp_flash_eachSess_pooledAll[ise] = quant_cc(p11_aveNPairsPooledAll_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p22_peak_amp_omit_eachSess_pooledAll[ise], p22_peak_amp_flash_eachSess_pooledAll[ise] = quant_cc(p22_aveNPairsPooledAll_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 

            p12_peak_amp_omit_eachSess_pooledAll_shfl[ise], p12_peak_amp_flash_eachSess_pooledAll_shfl[ise] = quant_cc(p12_aveNPairsPooledAll_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p11_peak_amp_omit_eachSess_pooledAll_shfl[ise], p11_peak_amp_flash_eachSess_pooledAll_shfl[ise] = quant_cc(p11_aveNPairsPooledAll_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 
            p22_peak_amp_omit_eachSess_pooledAll_shfl[ise], p22_peak_amp_flash_eachSess_pooledAll_shfl[ise] = quant_cc(p22_aveNPairsPooledAll_allSess_shfl[ise], fo_dur, list_times, list_times_flash_final, samps_bef, flash_win_final0, frame_dur, num_depth, sameBl_allLayerPairs) 



        #%% Set average and standard error of peak values (computed above) across sessions
        
        ##### omission-evoked
        cc12_peak_amp_omit_sessAv = np.nanmean(cc12_peak_amp_omit_eachSess, axis=0) # 4 x 4
        cc12_peak_amp_omit_sessSd = np.nanstd(cc12_peak_amp_omit_eachSess, axis=0) / np.sqrt(num_valid_sess_12_44)

        cc11_peak_amp_omit_sessAv = np.nanmean(cc11_peak_amp_omit_eachSess, axis=0)
        cc11_peak_amp_omit_sessSd = np.nanstd(cc11_peak_amp_omit_eachSess, axis=0) / np.sqrt(num_valid_sess_11_44)

        cc22_peak_amp_omit_sessAv = np.nanmean(cc22_peak_amp_omit_eachSess, axis=0)
        cc22_peak_amp_omit_sessSd = np.nanstd(cc22_peak_amp_omit_eachSess, axis=0) / np.sqrt(num_valid_sess_22_44)
        
        p12_peak_amp_omit_sessAv = np.nanmean(p12_peak_amp_omit_eachSess, axis=0)
        p12_peak_amp_omit_sessSd = np.nanstd(p12_peak_amp_omit_eachSess, axis=0) / np.sqrt(num_valid_sess_12_44)

        p11_peak_amp_omit_sessAv = np.nanmean(p11_peak_amp_omit_eachSess, axis=0)
        p11_peak_amp_omit_sessSd = np.nanstd(p11_peak_amp_omit_eachSess, axis=0) / np.sqrt(num_valid_sess_11_44)

        p22_peak_amp_omit_sessAv = np.nanmean(p22_peak_amp_omit_eachSess, axis=0)
        p22_peak_amp_omit_sessSd = np.nanstd(p22_peak_amp_omit_eachSess, axis=0) / np.sqrt(num_valid_sess_22_44)        

        
        cc12_peak_amp_omit_sessAv_shfl = np.nanmean(cc12_peak_amp_omit_eachSess_shfl, axis=0)
        cc12_peak_amp_omit_sessSd_shfl = np.nanstd(cc12_peak_amp_omit_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_12_44)

        cc11_peak_amp_omit_sessAv_shfl = np.nanmean(cc11_peak_amp_omit_eachSess_shfl, axis=0)
        cc11_peak_amp_omit_sessSd_shfl = np.nanstd(cc11_peak_amp_omit_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_11_44)

        cc22_peak_amp_omit_sessAv_shfl = np.nanmean(cc22_peak_amp_omit_eachSess_shfl, axis=0)
        cc22_peak_amp_omit_sessSd_shfl = np.nanstd(cc22_peak_amp_omit_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_22_44)
        
        p12_peak_amp_omit_sessAv_shfl = np.nanmean(p12_peak_amp_omit_eachSess_shfl, axis=0)
        p12_peak_amp_omit_sessSd_shfl = np.nanstd(p12_peak_amp_omit_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_12_44)

        p11_peak_amp_omit_sessAv_shfl = np.nanmean(p11_peak_amp_omit_eachSess_shfl, axis=0)
        p11_peak_amp_omit_sessSd_shfl = np.nanstd(p11_peak_amp_omit_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_11_44)

        p22_peak_amp_omit_sessAv_shfl = np.nanmean(p22_peak_amp_omit_eachSess_shfl, axis=0)
        p22_peak_amp_omit_sessSd_shfl = np.nanstd(p22_peak_amp_omit_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_22_44)        
        
        
        # pooled
        cc12_peak_amp_omit_sessAv_pooled = np.nanmean(cc12_peak_amp_omit_eachSess_pooled, axis=0) # 4
        cc12_peak_amp_omit_sessSd_pooled = np.nanstd(cc12_peak_amp_omit_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_12)

        cc11_peak_amp_omit_sessAv_pooled = np.nanmean(cc11_peak_amp_omit_eachSess_pooled, axis=0)
        cc11_peak_amp_omit_sessSd_pooled = np.nanstd(cc11_peak_amp_omit_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_11)

        cc22_peak_amp_omit_sessAv_pooled = np.nanmean(cc22_peak_amp_omit_eachSess_pooled, axis=0)
        cc22_peak_amp_omit_sessSd_pooled = np.nanstd(cc22_peak_amp_omit_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_22)
        
        p12_peak_amp_omit_sessAv_pooled = np.nanmean(p12_peak_amp_omit_eachSess_pooled, axis=0)
        p12_peak_amp_omit_sessSd_pooled = np.nanstd(p12_peak_amp_omit_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_12)

        p11_peak_amp_omit_sessAv_pooled = np.nanmean(p11_peak_amp_omit_eachSess_pooled, axis=0)
        p11_peak_amp_omit_sessSd_pooled = np.nanstd(p11_peak_amp_omit_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_11)

        p22_peak_amp_omit_sessAv_pooled = np.nanmean(p22_peak_amp_omit_eachSess_pooled, axis=0)
        p22_peak_amp_omit_sessSd_pooled = np.nanstd(p22_peak_amp_omit_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_22)        

        # shfl
        p12_peak_amp_omit_sessAv_pooled_shfl = np.nanmean(p12_peak_amp_omit_eachSess_pooled_shfl, axis=0)
        p12_peak_amp_omit_sessSd_pooled_shfl = np.nanstd(p12_peak_amp_omit_eachSess_pooled_shfl, axis=0) / np.sqrt(nv_pooled_12)

        p11_peak_amp_omit_sessAv_pooled_shfl = np.nanmean(p11_peak_amp_omit_eachSess_pooled_shfl, axis=0)
        p11_peak_amp_omit_sessSd_pooled_shfl = np.nanstd(p11_peak_amp_omit_eachSess_pooled_shfl, axis=0) / np.sqrt(nv_pooled_11)

        p22_peak_amp_omit_sessAv_pooled_shfl = np.nanmean(p22_peak_amp_omit_eachSess_pooled_shfl, axis=0)
        p22_peak_amp_omit_sessSd_pooled_shfl = np.nanstd(p22_peak_amp_omit_eachSess_pooled_shfl, axis=0) / np.sqrt(nv_pooled_22)    


        # pooledAll
        cc12_peak_amp_omit_sessAv_pooledAll = np.nanmean(cc12_peak_amp_omit_eachSess_pooledAll, axis=0) # 1
        cc12_peak_amp_omit_sessSd_pooledAll = np.nanstd(cc12_peak_amp_omit_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_12)

        cc11_peak_amp_omit_sessAv_pooledAll = np.nanmean(cc11_peak_amp_omit_eachSess_pooledAll, axis=0)
        cc11_peak_amp_omit_sessSd_pooledAll = np.nanstd(cc11_peak_amp_omit_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_11)

        cc22_peak_amp_omit_sessAv_pooledAll = np.nanmean(cc22_peak_amp_omit_eachSess_pooledAll, axis=0)
        cc22_peak_amp_omit_sessSd_pooledAll = np.nanstd(cc22_peak_amp_omit_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_22)
        
        p12_peak_amp_omit_sessAv_pooledAll = np.nanmean(p12_peak_amp_omit_eachSess_pooledAll, axis=0)
        p12_peak_amp_omit_sessSd_pooledAll = np.nanstd(p12_peak_amp_omit_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_12)

        p11_peak_amp_omit_sessAv_pooledAll = np.nanmean(p11_peak_amp_omit_eachSess_pooledAll, axis=0)
        p11_peak_amp_omit_sessSd_pooledAll = np.nanstd(p11_peak_amp_omit_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_11)

        p22_peak_amp_omit_sessAv_pooledAll = np.nanmean(p22_peak_amp_omit_eachSess_pooledAll, axis=0)
        p22_peak_amp_omit_sessSd_pooledAll = np.nanstd(p22_peak_amp_omit_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_22)        

        # shfl
        p12_peak_amp_omit_sessAv_pooledAll_shfl = np.nanmean(p12_peak_amp_omit_eachSess_pooledAll_shfl, axis=0)
        p12_peak_amp_omit_sessSd_pooledAll_shfl = np.nanstd(p12_peak_amp_omit_eachSess_pooledAll_shfl, axis=0) / np.sqrt(nv_pooledAll_12)

        p11_peak_amp_omit_sessAv_pooledAll_shfl = np.nanmean(p11_peak_amp_omit_eachSess_pooledAll_shfl, axis=0)
        p11_peak_amp_omit_sessSd_pooledAll_shfl = np.nanstd(p11_peak_amp_omit_eachSess_pooledAll_shfl, axis=0) / np.sqrt(nv_pooledAll_11)

        p22_peak_amp_omit_sessAv_pooledAll_shfl = np.nanmean(p22_peak_amp_omit_eachSess_pooledAll_shfl, axis=0)
        p22_peak_amp_omit_sessSd_pooledAll_shfl = np.nanstd(p22_peak_amp_omit_eachSess_pooledAll_shfl, axis=0) / np.sqrt(nv_pooledAll_22)    
        


        ##### flash-evoked
        cc12_peak_amp_flash_sessAv = np.nanmean(cc12_peak_amp_flash_eachSess, axis=0) # 4 x 4
        cc12_peak_amp_flash_sessSd = np.nanstd(cc12_peak_amp_flash_eachSess, axis=0) / np.sqrt(num_valid_sess_12_44)

        cc11_peak_amp_flash_sessAv = np.nanmean(cc11_peak_amp_flash_eachSess, axis=0)
        cc11_peak_amp_flash_sessSd = np.nanstd(cc11_peak_amp_flash_eachSess, axis=0) / np.sqrt(num_valid_sess_11_44)

        cc22_peak_amp_flash_sessAv = np.nanmean(cc22_peak_amp_flash_eachSess, axis=0)
        cc22_peak_amp_flash_sessSd = np.nanstd(cc22_peak_amp_flash_eachSess, axis=0) / np.sqrt(num_valid_sess_22_44)
        
        p12_peak_amp_flash_sessAv = np.nanmean(p12_peak_amp_flash_eachSess, axis=0)
        p12_peak_amp_flash_sessSd = np.nanstd(p12_peak_amp_flash_eachSess, axis=0) / np.sqrt(num_valid_sess_12_44)

        p11_peak_amp_flash_sessAv = np.nanmean(p11_peak_amp_flash_eachSess, axis=0)
        p11_peak_amp_flash_sessSd = np.nanstd(p11_peak_amp_flash_eachSess, axis=0) / np.sqrt(num_valid_sess_11_44)

        p22_peak_amp_flash_sessAv = np.nanmean(p22_peak_amp_flash_eachSess, axis=0)
        p22_peak_amp_flash_sessSd = np.nanstd(p22_peak_amp_flash_eachSess, axis=0) / np.sqrt(num_valid_sess_22_44)        

        
        cc12_peak_amp_flash_sessAv_shfl = np.nanmean(cc12_peak_amp_flash_eachSess_shfl, axis=0)
        cc12_peak_amp_flash_sessSd_shfl = np.nanstd(cc12_peak_amp_flash_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_12_44)

        cc11_peak_amp_flash_sessAv_shfl = np.nanmean(cc11_peak_amp_flash_eachSess_shfl, axis=0)
        cc11_peak_amp_flash_sessSd_shfl = np.nanstd(cc11_peak_amp_flash_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_11_44)

        cc22_peak_amp_flash_sessAv_shfl = np.nanmean(cc22_peak_amp_flash_eachSess_shfl, axis=0)
        cc22_peak_amp_flash_sessSd_shfl = np.nanstd(cc22_peak_amp_flash_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_22_44)
        
        p12_peak_amp_flash_sessAv_shfl = np.nanmean(p12_peak_amp_flash_eachSess_shfl, axis=0)
        p12_peak_amp_flash_sessSd_shfl = np.nanstd(p12_peak_amp_flash_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_12_44)

        p11_peak_amp_flash_sessAv_shfl = np.nanmean(p11_peak_amp_flash_eachSess_shfl, axis=0)
        p11_peak_amp_flash_sessSd_shfl = np.nanstd(p11_peak_amp_flash_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_11_44)

        p22_peak_amp_flash_sessAv_shfl = np.nanmean(p22_peak_amp_flash_eachSess_shfl, axis=0)
        p22_peak_amp_flash_sessSd_shfl = np.nanstd(p22_peak_amp_flash_eachSess_shfl, axis=0) / np.sqrt(num_valid_sess_22_44)  
        
        
        # pooled
        cc12_peak_amp_flash_sessAv_pooled = np.nanmean(cc12_peak_amp_flash_eachSess_pooled, axis=0) # 4
        cc12_peak_amp_flash_sessSd_pooled = np.nanstd(cc12_peak_amp_flash_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_12)

        cc11_peak_amp_flash_sessAv_pooled = np.nanmean(cc11_peak_amp_flash_eachSess_pooled, axis=0)
        cc11_peak_amp_flash_sessSd_pooled = np.nanstd(cc11_peak_amp_flash_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_11)

        cc22_peak_amp_flash_sessAv_pooled = np.nanmean(cc22_peak_amp_flash_eachSess_pooled, axis=0)
        cc22_peak_amp_flash_sessSd_pooled = np.nanstd(cc22_peak_amp_flash_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_22)
        
        p12_peak_amp_flash_sessAv_pooled = np.nanmean(p12_peak_amp_flash_eachSess_pooled, axis=0)
        p12_peak_amp_flash_sessSd_pooled = np.nanstd(p12_peak_amp_flash_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_12)

        p11_peak_amp_flash_sessAv_pooled = np.nanmean(p11_peak_amp_flash_eachSess_pooled, axis=0)
        p11_peak_amp_flash_sessSd_pooled = np.nanstd(p11_peak_amp_flash_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_11)

        p22_peak_amp_flash_sessAv_pooled = np.nanmean(p22_peak_amp_flash_eachSess_pooled, axis=0)
        p22_peak_amp_flash_sessSd_pooled = np.nanstd(p22_peak_amp_flash_eachSess_pooled, axis=0) / np.sqrt(nv_pooled_22)        

        # shfl
        p12_peak_amp_flash_sessAv_pooled_shfl = np.nanmean(p12_peak_amp_flash_eachSess_pooled_shfl, axis=0)
        p12_peak_amp_flash_sessSd_pooled_shfl = np.nanstd(p12_peak_amp_flash_eachSess_pooled_shfl, axis=0) / np.sqrt(nv_pooled_12)

        p11_peak_amp_flash_sessAv_pooled_shfl = np.nanmean(p11_peak_amp_flash_eachSess_pooled_shfl, axis=0)
        p11_peak_amp_flash_sessSd_pooled_shfl = np.nanstd(p11_peak_amp_flash_eachSess_pooled_shfl, axis=0) / np.sqrt(nv_pooled_11)

        p22_peak_amp_flash_sessAv_pooled_shfl = np.nanmean(p22_peak_amp_flash_eachSess_pooled_shfl, axis=0)
        p22_peak_amp_flash_sessSd_pooled_shfl = np.nanstd(p22_peak_amp_flash_eachSess_pooled_shfl, axis=0) / np.sqrt(nv_pooled_22)    


        # pooledAll
        cc12_peak_amp_flash_sessAv_pooledAll = np.nanmean(cc12_peak_amp_flash_eachSess_pooledAll, axis=0) # 1
        cc12_peak_amp_flash_sessSd_pooledAll = np.nanstd(cc12_peak_amp_flash_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_12)

        cc11_peak_amp_flash_sessAv_pooledAll = np.nanmean(cc11_peak_amp_flash_eachSess_pooledAll, axis=0)
        cc11_peak_amp_flash_sessSd_pooledAll = np.nanstd(cc11_peak_amp_flash_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_11)

        cc22_peak_amp_flash_sessAv_pooledAll = np.nanmean(cc22_peak_amp_flash_eachSess_pooledAll, axis=0)
        cc22_peak_amp_flash_sessSd_pooledAll = np.nanstd(cc22_peak_amp_flash_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_22)
        
        p12_peak_amp_flash_sessAv_pooledAll = np.nanmean(p12_peak_amp_flash_eachSess_pooledAll, axis=0)
        p12_peak_amp_flash_sessSd_pooledAll = np.nanstd(p12_peak_amp_flash_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_12)

        p11_peak_amp_flash_sessAv_pooledAll = np.nanmean(p11_peak_amp_flash_eachSess_pooledAll, axis=0)
        p11_peak_amp_flash_sessSd_pooledAll = np.nanstd(p11_peak_amp_flash_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_11)

        p22_peak_amp_flash_sessAv_pooledAll = np.nanmean(p22_peak_amp_flash_eachSess_pooledAll, axis=0)
        p22_peak_amp_flash_sessSd_pooledAll = np.nanstd(p22_peak_amp_flash_eachSess_pooledAll, axis=0) / np.sqrt(nv_pooledAll_22)        

        # shfl
        p12_peak_amp_flash_sessAv_pooledAll_shfl = np.nanmean(p12_peak_amp_flash_eachSess_pooledAll_shfl, axis=0)
        p12_peak_amp_flash_sessSd_pooledAll_shfl = np.nanstd(p12_peak_amp_flash_eachSess_pooledAll_shfl, axis=0) / np.sqrt(nv_pooledAll_12)

        p11_peak_amp_flash_sessAv_pooledAll_shfl = np.nanmean(p11_peak_amp_flash_eachSess_pooledAll_shfl, axis=0)
        p11_peak_amp_flash_sessSd_pooledAll_shfl = np.nanstd(p11_peak_amp_flash_eachSess_pooledAll_shfl, axis=0) / np.sqrt(nv_pooledAll_11)

        p22_peak_amp_flash_sessAv_pooledAll_shfl = np.nanmean(p22_peak_amp_flash_eachSess_pooledAll_shfl, axis=0)
        p22_peak_amp_flash_sessSd_pooledAll_shfl = np.nanstd(p22_peak_amp_flash_eachSess_pooledAll_shfl, axis=0) / np.sqrt(nv_pooledAll_22)    
        
        
        ######### outputs: 4 x 4 (quantification of corrlation in neural activities evoked by omission (or flash) between different layers)
        # I prefer above, bc it is done on individual sessions, and allows us to get sd across sessions. below is done on average sessions.
        '''
        cc12_peak_amp_omit, cc12_peak_amp_flash = quant_cc(cc12_sessAv, sameBl_allLayerPairs) 
        cc11_peak_amp_omit, cc11_peak_amp_flash = quant_cc(cc11_sessAv, sameBl_allLayerPairs)
        cc22_peak_amp_omit, cc22_peak_amp_flash = quant_cc(cc22_sessAv, sameBl_allLayerPairs)

        p12_peak_amp_omit, p12_peak_amp_flash = quant_cc(p12_sessAv, sameBl_allLayerPairs)
        p11_peak_amp_omit, p11_peak_amp_flash = quant_cc(p11_sessAv, sameBl_allLayerPairs)
        p22_peak_amp_omit, p22_peak_amp_flash = quant_cc(p22_sessAv, sameBl_allLayerPairs)
        
        cc12_peak_amp_omit_shfl, cc12_peak_amp_flash_shfl = quant_cc(cc12_sessAv_shfl, sameBl_allLayerPairs) 
        cc11_peak_amp_omit_shfl, cc11_peak_amp_flash_shfl = quant_cc(cc11_sessAv_shfl, sameBl_allLayerPairs)
        cc22_peak_amp_omit_shfl, cc22_peak_amp_flash_shfl = quant_cc(cc22_sessAv_shfl, sameBl_allLayerPairs)

        p12_peak_amp_omit_shfl, p12_peak_amp_flash_shfl = quant_cc(p12_sessAv_shfl, sameBl_allLayerPairs)
        p11_peak_amp_omit_shfl, p11_peak_amp_flash_shfl = quant_cc(p11_sessAv_shfl, sameBl_allLayerPairs)
        p22_peak_amp_omit_shfl, p22_peak_amp_flash_shfl = quant_cc(p22_sessAv_shfl, sameBl_allLayerPairs)
        
        # remember: when you call "quant_cc" if "sameBl_allLayerPairs" is 1, it will first subtract the baseline from each trace, and then will compute the peak.
        # so a is not the same as cc12_peak_amp_omit
#         a = np.nanmean(cc12_peak_amp_omit_eachSess, axis=0)
#         plt.figure(); plt.subplot(121); plt.imshow(a); plt.colorbar(); plt.subplot(122); plt.imshow(cc12_peak_amp_omit); plt.colorbar()
        '''        

        
        ### traces: plot time traces of corr of area1-area2, for each layer pair
        '''
        colors = colorOrder(nlines=num_depth)
        # each layer of V1 with all layers of LM
        plt.figure(figsize=(10,6))
        plt.suptitle('V1 each layer- LM all layers', y=1)
        for i2 in range(4): # loop through V1 layers
            plt.subplot(2,2,i2+1)
            for i1 in range(4): # loop through LM layers
                plt.plot(time_trace, cc12_sessAv_44[:,i1,i2], color=colors[i1], label='LM L%d' %i1)
                plt.title('V1 layer %d' %i2, y=1, fontsize=13)
                if i2==0:
                    plt.legend(loc='center left', bbox_to_anchor=(.9, .7), frameon=False, fontsize=12, handlelength=.4)
            plot_flashLines_ticks_legend([], [], flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, ylab='CC', xmjn=xmjn)
        plt.subplots_adjust(wspace=.5, hspace=.6)
        '''
        
        ### heatmap: plot quantification of corr of area1-area2, for each layer pair
        '''
        plt.figure(figsize=(8,4))
        plt.suptitle([mouse_id, cre, num_sessions])
        
        plt.subplot(121)
        plt.imshow(cc12_peak_amp_omit)
        plt.colorbar(label='cc')
#        plt.clim(cmin, cmax)
        plt.title('omit-evoked')
        plt.xlabel('V1')
        plt.ylabel('LM')
        
        plt.subplot(122)
        plt.imshow(cc12_peak_amp_flash)
        plt.colorbar(label='cc')
#        plt.clim(cmin, cmax)
        plt.title('flash-evoked')
        plt.xlabel('V1')
        plt.ylabel('LM')
        
        plt.subplots_adjust(wspace=.7) #, hspace=.8)
        '''
        
        
        ###########################################################################
        ###########################################################################
        ###########################################################################        
        ###########################################################################
        ###########################################################################
        ###########################################################################        
        #%% Keep vars in a dataframe
        ###########################################################################
        ###########################################################################
        ###########################################################################        
        ###########################################################################
        ###########################################################################
        ###########################################################################
        
        # cc12_sessAv_44
        # cc12_sessSd_44
        # cc12_peak_amp_omit_sessAv    
        # cc12_peak_amp_omit_sessSd
        # num_valid_sess_12_44
        
        corr_trace_peak_allMice.at[im, ['mouse_id', 'cre', 'session_stages', 'session_labs']] = mouse_id, cre, session_stages, session_labs
        corr_trace_peak_allMice.at[im, 'areas'] = areas
        corr_trace_peak_allMice.at[im, 'depths'] = depths

        corr_trace_peak_allMice.at[im, 'num_valid_sess_12_44'] = num_valid_sess_12_44
        corr_trace_peak_allMice.at[im, 'num_valid_sess_11_44'] = num_valid_sess_11_44        
        corr_trace_peak_allMice.at[im, 'num_valid_sess_22_44'] = num_valid_sess_22_44
        
        # pooled
        corr_trace_peak_allMice.at[im, 'num_valid_sess_pooledLayers_12'] = nv_pooled_12
        corr_trace_peak_allMice.at[im, 'num_valid_sess_pooledLayers_11'] = nv_pooled_11
        corr_trace_peak_allMice.at[im, 'num_valid_sess_pooledLayers_22'] = nv_pooled_22
        corr_trace_peak_allMice.at[im, 'num_valid_sess_pooledAllLayers_12'] = nv_pooledAll_12
        corr_trace_peak_allMice.at[im, 'num_valid_sess_pooledAllLayers_11'] = nv_pooledAll_11
        corr_trace_peak_allMice.at[im, 'num_valid_sess_pooledAllLayers_22'] = nv_pooledAll_22
        
        
        ########## Traces of cc and p ##########
        corr_trace_peak_allMice.at[im, 'cc12_sessAv_44'] = cc12_sessAv_44
        corr_trace_peak_allMice.at[im, 'cc12_sessSd_44'] = cc12_sessSd_44
        corr_trace_peak_allMice.at[im, 'cc11_sessAv_44'] = cc11_sessAv_44
        corr_trace_peak_allMice.at[im, 'cc11_sessSd_44'] = cc11_sessSd_44
        corr_trace_peak_allMice.at[im, 'cc22_sessAv_44'] = cc22_sessAv_44
        corr_trace_peak_allMice.at[im, 'cc22_sessSd_44'] = cc22_sessSd_44

        corr_trace_peak_allMice.at[im, 'p12_sessAv_44'] = p12_sessAv_44
        corr_trace_peak_allMice.at[im, 'p12_sessSd_44'] = p12_sessSd_44        
        corr_trace_peak_allMice.at[im, 'p11_sessAv_44'] = p11_sessAv_44
        corr_trace_peak_allMice.at[im, 'p11_sessSd_44'] = p11_sessSd_44        
        corr_trace_peak_allMice.at[im, 'p22_sessAv_44'] = p22_sessAv_44
        corr_trace_peak_allMice.at[im, 'p22_sessSd_44'] = p22_sessSd_44

        corr_trace_peak_allMice.at[im, 'cc12_sessAv_44_shfl'] = cc12_sessAv_44_shfl
        corr_trace_peak_allMice.at[im, 'cc12_sessSd_44_shfl'] = cc12_sessSd_44_shfl
        corr_trace_peak_allMice.at[im, 'cc11_sessAv_44_shfl'] = cc11_sessAv_44_shfl
        corr_trace_peak_allMice.at[im, 'cc11_sessSd_44_shfl'] = cc11_sessSd_44_shfl
        corr_trace_peak_allMice.at[im, 'cc22_sessAv_44_shfl'] = cc22_sessAv_44_shfl
        corr_trace_peak_allMice.at[im, 'cc22_sessSd_44_shfl'] = cc22_sessSd_44_shfl

        corr_trace_peak_allMice.at[im, 'p12_sessAv_44_shfl'] = p12_sessAv_44_shfl
        corr_trace_peak_allMice.at[im, 'p12_sessSd_44_shfl'] = p12_sessSd_44_shfl        
        corr_trace_peak_allMice.at[im, 'p11_sessAv_44_shfl'] = p11_sessAv_44_shfl
        corr_trace_peak_allMice.at[im, 'p11_sessSd_44_shfl'] = p11_sessSd_44_shfl        
        corr_trace_peak_allMice.at[im, 'p22_sessAv_44_shfl'] = p22_sessAv_44_shfl
        corr_trace_peak_allMice.at[im, 'p22_sessSd_44_shfl'] = p22_sessSd_44_shfl

        # pooled
        corr_trace_peak_allMice.at[im, 'cc12_pooled_sessAv'] = cc12_pooled_sessAv
        corr_trace_peak_allMice.at[im, 'cc12_pooled_sessSd'] = cc12_pooled_sessSd
        corr_trace_peak_allMice.at[im, 'cc11_pooled_sessAv'] = cc11_pooled_sessAv
        corr_trace_peak_allMice.at[im, 'cc11_pooled_sessSd'] = cc11_pooled_sessSd
        corr_trace_peak_allMice.at[im, 'cc22_pooled_sessAv'] = cc22_pooled_sessAv
        corr_trace_peak_allMice.at[im, 'cc22_pooled_sessSd'] = cc22_pooled_sessSd

        corr_trace_peak_allMice.at[im, 'p12_pooled_sessAv'] = p12_pooled_sessAv
        corr_trace_peak_allMice.at[im, 'p12_pooled_sessSd'] = p12_pooled_sessSd        
        corr_trace_peak_allMice.at[im, 'p11_pooled_sessAv'] = p11_pooled_sessAv
        corr_trace_peak_allMice.at[im, 'p11_pooled_sessSd'] = p11_pooled_sessSd        
        corr_trace_peak_allMice.at[im, 'p22_pooled_sessAv'] = p22_pooled_sessAv
        corr_trace_peak_allMice.at[im, 'p22_pooled_sessSd'] = p22_pooled_sessSd

        corr_trace_peak_allMice.at[im, 'p12_pooled_sessAv_shfl'] = p12_pooled_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p12_pooled_sessSd_shfl'] = p12_pooled_sessSd_shfl        
        corr_trace_peak_allMice.at[im, 'p11_pooled_sessAv_shfl'] = p11_pooled_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p11_pooled_sessSd_shfl'] = p11_pooled_sessSd_shfl        
        corr_trace_peak_allMice.at[im, 'p22_pooled_sessAv_shfl'] = p22_pooled_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p22_pooled_sessSd_shfl'] = p22_pooled_sessSd_shfl
        
        # pooledAll
        corr_trace_peak_allMice.at[im, 'cc12_pooledAll_sessAv'] = cc12_pooledAll_sessAv
        corr_trace_peak_allMice.at[im, 'cc12_pooledAll_sessSd'] = cc12_pooledAll_sessSd
        corr_trace_peak_allMice.at[im, 'cc11_pooledAll_sessAv'] = cc11_pooledAll_sessAv
        corr_trace_peak_allMice.at[im, 'cc11_pooledAll_sessSd'] = cc11_pooledAll_sessSd
        corr_trace_peak_allMice.at[im, 'cc22_pooledAll_sessAv'] = cc22_pooledAll_sessAv
        corr_trace_peak_allMice.at[im, 'cc22_pooledAll_sessSd'] = cc22_pooledAll_sessSd

        corr_trace_peak_allMice.at[im, 'p12_pooledAll_sessAv'] = p12_pooledAll_sessAv
        corr_trace_peak_allMice.at[im, 'p12_pooledAll_sessSd'] = p12_pooledAll_sessSd        
        corr_trace_peak_allMice.at[im, 'p11_pooledAll_sessAv'] = p11_pooledAll_sessAv
        corr_trace_peak_allMice.at[im, 'p11_pooledAll_sessSd'] = p11_pooledAll_sessSd        
        corr_trace_peak_allMice.at[im, 'p22_pooledAll_sessAv'] = p22_pooledAll_sessAv
        corr_trace_peak_allMice.at[im, 'p22_pooledAll_sessSd'] = p22_pooledAll_sessSd

        corr_trace_peak_allMice.at[im, 'p12_pooledAll_sessAv_shfl'] = p12_pooledAll_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p12_pooledAll_sessSd_shfl'] = p12_pooledAll_sessSd_shfl        
        corr_trace_peak_allMice.at[im, 'p11_pooledAll_sessAv_shfl'] = p11_pooledAll_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p11_pooledAll_sessSd_shfl'] = p11_pooledAll_sessSd_shfl        
        corr_trace_peak_allMice.at[im, 'p22_pooledAll_sessAv_shfl'] = p22_pooledAll_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p22_pooledAll_sessSd_shfl'] = p22_pooledAll_sessSd_shfl

        
        
        ########## Peak quantification vars (omit and flash evoked) ##########
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_omit_sessAv'] = cc12_peak_amp_omit_sessAv
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_omit_sessSd'] = cc12_peak_amp_omit_sessSd        
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_flash_sessAv'] = cc12_peak_amp_flash_sessAv
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_flash_sessSd'] = cc12_peak_amp_flash_sessSd

        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_omit_sessAv'] = cc11_peak_amp_omit_sessAv
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_omit_sessSd'] = cc11_peak_amp_omit_sessSd        
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_flash_sessAv'] = cc11_peak_amp_flash_sessAv
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_flash_sessSd'] = cc11_peak_amp_flash_sessSd

        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_omit_sessAv'] = cc22_peak_amp_omit_sessAv
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_omit_sessSd'] = cc22_peak_amp_omit_sessSd        
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_flash_sessAv'] = cc22_peak_amp_flash_sessAv
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_flash_sessSd'] = cc22_peak_amp_flash_sessSd

        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessAv'] = p12_peak_amp_omit_sessAv
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessSd'] = p12_peak_amp_omit_sessSd        
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessAv'] = p12_peak_amp_flash_sessAv
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessSd'] = p12_peak_amp_flash_sessSd

        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessAv'] = p11_peak_amp_omit_sessAv
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessSd'] = p11_peak_amp_omit_sessSd        
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessAv'] = p11_peak_amp_flash_sessAv
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessSd'] = p11_peak_amp_flash_sessSd

        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessAv'] = p22_peak_amp_omit_sessAv
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessSd'] = p22_peak_amp_omit_sessSd        
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessAv'] = p22_peak_amp_flash_sessAv
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessSd'] = p22_peak_amp_flash_sessSd
        
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_omit_sessAv_shfl'] = cc12_peak_amp_omit_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_omit_sessSd_shfl'] = cc12_peak_amp_omit_sessSd_shfl        
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_flash_sessAv_shfl'] = cc12_peak_amp_flash_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_flash_sessSd_shfl'] = cc12_peak_amp_flash_sessSd_shfl

        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_omit_sessAv_shfl'] = cc11_peak_amp_omit_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_omit_sessSd_shfl'] = cc11_peak_amp_omit_sessSd_shfl        
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_flash_sessAv_shfl'] = cc11_peak_amp_flash_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_flash_sessSd_shfl'] = cc11_peak_amp_flash_sessSd_shfl

        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_omit_sessAv_shfl'] = cc22_peak_amp_omit_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_omit_sessSd_shfl'] = cc22_peak_amp_omit_sessSd_shfl        
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_flash_sessAv_shfl'] = cc22_peak_amp_flash_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_flash_sessSd_shfl'] = cc22_peak_amp_flash_sessSd_shfl

        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessAv_shfl'] = p12_peak_amp_omit_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessSd_shfl'] = p12_peak_amp_omit_sessSd_shfl        
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessAv_shfl'] = p12_peak_amp_flash_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessSd_shfl'] = p12_peak_amp_flash_sessSd_shfl

        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessAv_shfl'] = p11_peak_amp_omit_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessSd_shfl'] = p11_peak_amp_omit_sessSd_shfl        
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessAv_shfl'] = p11_peak_amp_flash_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessSd_shfl'] = p11_peak_amp_flash_sessSd_shfl

        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessAv_shfl'] = p22_peak_amp_omit_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessSd_shfl'] = p22_peak_amp_omit_sessSd_shfl        
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessAv_shfl'] = p22_peak_amp_flash_sessAv_shfl
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessSd_shfl'] = p22_peak_amp_flash_sessSd_shfl
        
        # pooled        
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_omit_sessAv_pooled'] = cc12_peak_amp_omit_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_omit_sessSd_pooled'] = cc12_peak_amp_omit_sessSd_pooled        
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_flash_sessAv_pooled'] = cc12_peak_amp_flash_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_flash_sessSd_pooled'] = cc12_peak_amp_flash_sessSd_pooled

        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_omit_sessAv_pooled'] = cc11_peak_amp_omit_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_omit_sessSd_pooled'] = cc11_peak_amp_omit_sessSd_pooled        
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_flash_sessAv_pooled'] = cc11_peak_amp_flash_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_flash_sessSd_pooled'] = cc11_peak_amp_flash_sessSd_pooled

        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_omit_sessAv_pooled'] = cc22_peak_amp_omit_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_omit_sessSd_pooled'] = cc22_peak_amp_omit_sessSd_pooled        
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_flash_sessAv_pooled'] = cc22_peak_amp_flash_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_flash_sessSd_pooled'] = cc22_peak_amp_flash_sessSd_pooled

        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessAv_pooled'] = p12_peak_amp_omit_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessSd_pooled'] = p12_peak_amp_omit_sessSd_pooled        
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessAv_pooled'] = p12_peak_amp_flash_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessSd_pooled'] = p12_peak_amp_flash_sessSd_pooled

        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessAv_pooled'] = p11_peak_amp_omit_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessSd_pooled'] = p11_peak_amp_omit_sessSd_pooled        
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessAv_pooled'] = p11_peak_amp_flash_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessSd_pooled'] = p11_peak_amp_flash_sessSd_pooled

        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessAv_pooled'] = p22_peak_amp_omit_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessSd_pooled'] = p22_peak_amp_omit_sessSd_pooled        
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessAv_pooled'] = p22_peak_amp_flash_sessAv_pooled
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessSd_pooled'] = p22_peak_amp_flash_sessSd_pooled
 
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessAv_pooled_shfl'] = p12_peak_amp_omit_sessAv_pooled_shfl
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessSd_pooled_shfl'] = p12_peak_amp_omit_sessSd_pooled_shfl        
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessAv_pooled_shfl'] = p12_peak_amp_flash_sessAv_pooled_shfl
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessSd_pooled_shfl'] = p12_peak_amp_flash_sessSd_pooled_shfl

        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessAv_pooled_shfl'] = p11_peak_amp_omit_sessAv_pooled_shfl
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessSd_pooled_shfl'] = p11_peak_amp_omit_sessSd_pooled_shfl        
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessAv_pooled_shfl'] = p11_peak_amp_flash_sessAv_pooled_shfl
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessSd_pooled_shfl'] = p11_peak_amp_flash_sessSd_pooled_shfl

        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessAv_pooled_shfl'] = p22_peak_amp_omit_sessAv_pooled_shfl
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessSd_pooled_shfl'] = p22_peak_amp_omit_sessSd_pooled_shfl        
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessAv_pooled_shfl'] = p22_peak_amp_flash_sessAv_pooled_shfl
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessSd_pooled_shfl'] = p22_peak_amp_flash_sessSd_pooled_shfl

        # pooledAll        
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_omit_sessAv_pooledAll'] = cc12_peak_amp_omit_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_omit_sessSd_pooledAll'] = cc12_peak_amp_omit_sessSd_pooledAll        
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_flash_sessAv_pooledAll'] = cc12_peak_amp_flash_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'cc12_peak_amp_flash_sessSd_pooledAll'] = cc12_peak_amp_flash_sessSd_pooledAll

        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_omit_sessAv_pooledAll'] = cc11_peak_amp_omit_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_omit_sessSd_pooledAll'] = cc11_peak_amp_omit_sessSd_pooledAll        
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_flash_sessAv_pooledAll'] = cc11_peak_amp_flash_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'cc11_peak_amp_flash_sessSd_pooledAll'] = cc11_peak_amp_flash_sessSd_pooledAll

        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_omit_sessAv_pooledAll'] = cc22_peak_amp_omit_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_omit_sessSd_pooledAll'] = cc22_peak_amp_omit_sessSd_pooledAll        
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_flash_sessAv_pooledAll'] = cc22_peak_amp_flash_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'cc22_peak_amp_flash_sessSd_pooledAll'] = cc22_peak_amp_flash_sessSd_pooledAll

        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessAv_pooledAll'] = p12_peak_amp_omit_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessSd_pooledAll'] = p12_peak_amp_omit_sessSd_pooledAll        
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessAv_pooledAll'] = p12_peak_amp_flash_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessSd_pooledAll'] = p12_peak_amp_flash_sessSd_pooledAll

        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessAv_pooledAll'] = p11_peak_amp_omit_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessSd_pooledAll'] = p11_peak_amp_omit_sessSd_pooledAll        
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessAv_pooledAll'] = p11_peak_amp_flash_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessSd_pooledAll'] = p11_peak_amp_flash_sessSd_pooledAll

        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessAv_pooledAll'] = p22_peak_amp_omit_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessSd_pooledAll'] = p22_peak_amp_omit_sessSd_pooledAll        
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessAv_pooledAll'] = p22_peak_amp_flash_sessAv_pooledAll
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessSd_pooledAll'] = p22_peak_amp_flash_sessSd_pooledAll
 
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessAv_pooledAll_shfl'] = p12_peak_amp_omit_sessAv_pooledAll_shfl
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_omit_sessSd_pooledAll_shfl'] = p12_peak_amp_omit_sessSd_pooledAll_shfl        
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessAv_pooledAll_shfl'] = p12_peak_amp_flash_sessAv_pooledAll_shfl
        corr_trace_peak_allMice.at[im, 'p12_peak_amp_flash_sessSd_pooledAll_shfl'] = p12_peak_amp_flash_sessSd_pooledAll_shfl

        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessAv_pooledAll_shfl'] = p11_peak_amp_omit_sessAv_pooledAll_shfl
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_omit_sessSd_pooledAll_shfl'] = p11_peak_amp_omit_sessSd_pooledAll_shfl        
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessAv_pooledAll_shfl'] = p11_peak_amp_flash_sessAv_pooledAll_shfl
        corr_trace_peak_allMice.at[im, 'p11_peak_amp_flash_sessSd_pooledAll_shfl'] = p11_peak_amp_flash_sessSd_pooledAll_shfl

        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessAv_pooledAll_shfl'] = p22_peak_amp_omit_sessAv_pooledAll_shfl
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_omit_sessSd_pooledAll_shfl'] = p22_peak_amp_omit_sessSd_pooledAll_shfl        
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessAv_pooledAll_shfl'] = p22_peak_amp_flash_sessAv_pooledAll_shfl
        corr_trace_peak_allMice.at[im, 'p22_peak_amp_flash_sessSd_pooledAll_shfl'] = p22_peak_amp_flash_sessSd_pooledAll_shfl

        
        
#%%        
print(len(corr_trace_peak_allMice))
corr_trace_peak_allMice #.iloc[:2]
#corr_trace_peak_allMice.iloc[0]['cc12_sessSd_44'].shape





#%% 
###############################################################################
######## Set vars to summarize data across mice of the same cre line ##########
###############################################################################
###############################################################################

# Call the function below
# exec(open("omissions_traces_peaks_plots_setVars_corr_eachCre.py").read())    
    
    
