#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines functions needed in omissions_traces_peaks_plots_setVars_corrs.py and other correlation plotting scripts.

Created on Wed Apr 29 11:50:29 2020
@author: farzaneh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from def_funs import * 

#%% Define flash_win_final (depending on cre and sessions stage)

def set_flash_win_final(cre, session_novel, flash_win, flash_win_vip):
    
    if np.logical_and(cre.find('Vip')==1 , session_novel) or cre.find('Vip')==-1: # VIP B1, SST, SLC: window after images (response follows the image)
        flash_win_final = flash_win # [0, .75]

    else: # VIP (non B1,familiar sessions): window precedes the image (pre-stimulus ramping activity)
        flash_win_final = flash_win_vip # [-.25, .5] 
        

    return flash_win_final


#%% Quantify corr after flash, and after omission (to get the heatmap of corrs!)

def quant_cc(cc_sessAv, fo_dur, list_times, list_times_flash, samps_bef, bl_percentile, frame_dur, num_depth, sameBl_allLayerPairs): # cc_sessAv = cc11_sessAv
        
    if sameBl_allLayerPairs:
        # Note: instead of defining a single flash_index for all sessions, I would use flash_omit_dur_fr_all (which now we are saving in this_sess (and all_sess)) for each session individually.
        mxgp = np.max(fo_dur) # .75 #
        # on 5/11/2020 I changed np.floor (below) to np.round, to make things consistent with set_frame_window_flash_omit... also i think it is more accurate to use round than floor!
        flash_index = samps_bef - np.round(mxgp / frame_dur).astype(int) # not quite accurate, if you want to be quite accurate you should align the traces on flashes!
        bl_index_pre_flash = np.arange(0,flash_index)

        bl_index_pre_omit = np.arange(0,samps_bef) # you will need for response amplitude quantification, even if you dont use it below.
        
        bl_preOmit = np.percentile(cc_sessAv[bl_index_pre_omit], bl_percentile, axis=0) # 16 (layer combinations of the two areas)
        bl_preFlash = np.percentile(cc_sessAv[bl_index_pre_flash], bl_percentile, axis=0) # 16 (layer combinations of the two areas)
        
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
        plt.ylim(lims0_f)
        if i2==0:
            plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
        ylim0 = lims0_f #plt.gca().get_ylim()
#         text_y = ylim0[1] + np.diff(ylim0)/3
        text_y = .5 #ylim0[0] + np.diff(ylim0)/3.3
        if i2==0:
#             plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#             plt.text(-3.8, text_y, 'Flash', fontsize=15, rotation= 'vertical') # 3.2
            plt.text(-1, text_y, 'Flash', fontsize=15, rotation='vertical', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
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
        plt.ylim(lims0_f)
        if i2==0:    
            plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
        ylim0 = lims0_f #plt.gca().get_ylim()
#         text_y = ylim0[1] + np.diff(ylim0)/3
        text_y = .5 #ylim0[0] + np.diff(ylim0)/3.3
        if i2==0:
#             plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#             plt.text(-4.8, text_y, 'Flash', fontsize=15, rotation='vertical')
            plt.text(-1, text_y, 'Flash', fontsize=15, rotation='vertical', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
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
        plt.ylim(lims0_f)
        if i2==0:    
            plt.ylabel(yl, fontsize=12)#, rotation=0, labelpad=35)
        ylim0 = lims0_f #plt.gca().get_ylim()
#         text_y = ylim0[0] + np.diff(ylim0)/3
        text_y = .5 #ylim0[0] + np.diff(ylim0)/3.3
        if i2==0:
#             plt.text(.3, text_y, 'Flash', fontsize=15) # 3.2
#             plt.text(-4.8, text_y, 'Flash', fontsize=15, rotation='vertical')
            plt.text(-1, text_y, 'Flash', fontsize=15, rotation='vertical', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
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


    ###### Flash ######
    ax1 = plt.subplot(gs[0])    

    im = plt.imshow(f_top0);
    plt.title('Flash', y=1, fontsize=13)
    plot_colorbar_set_xy_ticks(ax1, im, lims0_fo, yl, depths, num_depth)
    

    ###### Omission ######
    ax1 = plt.subplot(gs[1])    

    im = plt.imshow(o_top0);
    plt.title('Omission', y=1, fontsize=13)
    plot_colorbar_set_xy_ticks(ax1, im, lims0_fo, yl, depths, num_depth)



    ##################################
    ########## gs2: cc_shfl ##########   
    ##################################    

    yl = 'cc_shfl amplitude'
    lims0_fo = np.squeeze([np.nanmin([f_top1, o_top1]) , np.nanmax([f_top1, o_top1])])


    ###### Flash ######
    ax1 = plt.subplot(gs2[0])    
    plt.title('Flash', y=1, fontsize=13)

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
    plt.title('Flash', y=1, fontsize=13)

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

