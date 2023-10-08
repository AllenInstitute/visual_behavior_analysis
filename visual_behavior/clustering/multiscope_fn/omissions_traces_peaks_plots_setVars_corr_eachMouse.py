#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gets called in omissions_traces_peaks_plots_setVars_corr.py
Plot each mouse of a cre line

Created on Tue Oct  8 18:22:20 2019
@author: farzaneh
"""

sess = session_labs_allMice_thisCre[icn]
title = '%s (id: %d, %d sessions %s)' %(cre, mouse_id_allMice_thisCre[icn], len(sess), sess)

flash_win_final = set_flash_win_final(cre, session_novel, flash_win, flash_win_vip)


#%% Traces: 4 subplots for each V1 layer. Each subplot shows cc between a given V1 layer and all LM layers.    

if plot_cc_traces:
    ###################################
    ########## area1 - area2 ##########
    ###################################

    fgn = 'cc12'
    areas = 'LM', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    top0 = cc12_sessAv_allMice_thisCre[icn]
    top_sd0 = cc12_sessSd_allMice_thisCre[icn]

    top1 = cc12_sessAv_allMice_thisCre_shfl[icn]
    top_sd1 = cc12_sessSd_allMice_thisCre_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p12_sessAv_allMice_thisCre[icn]
        top_sd2 = p12_sessSd_allMice_thisCre[icn]
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p12_sessAv_allMice_thisCre_shfl[icn]
        top_sd2 = p12_sessSd_allMice_thisCre_shfl[icn]    


    plot_traces_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, fgn, areas, depth_sessAv_allMice_thisCre[icn], title, xlim_frs, jet_cm, alph, mouse_id_allMice_thisCre[icn], dosavefig)


    ###################################
    ########## area1 - area1 ##########
    ###################################

    fgn = 'cc11'
    areas = 'LM', 'LM' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    top0 = cc11_sessAv_allMice_thisCre[icn]
    top_sd0 = cc11_sessSd_allMice_thisCre[icn]

    top1 = cc11_sessAv_allMice_thisCre_shfl[icn]
    top_sd1 = cc11_sessSd_allMice_thisCre_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p11_sessAv_allMice_thisCre[icn]
        top_sd2 = p11_sessSd_allMice_thisCre[icn]
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p11_sessAv_allMice_thisCre_shfl[icn]
        top_sd2 = p11_sessSd_allMice_thisCre_shfl[icn]    

    # replace the lower part of the matrix (that are nan) with the symmetric values on the upper side... so we get to see all cc traces in all subplots    
    for i2 in range(num_depth):
        for i1 in np.arange(i2+1, num_depth):
            top0[:,i1,i2] = top0[:,i2,i1]
            top1[:,i1,i2] = top1[:,i2,i1]            
            top2[:,i1,i2] = top2[:,i2,i1]
            top_sd0[:,i1,i2] = top_sd0[:,i2,i1]
            top_sd1[:,i1,i2] = top_sd1[:,i2,i1]            
            top_sd2[:,i1,i2] = top_sd2[:,i2,i1]


    plot_traces_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, fgn, areas, depth_sessAv_allMice_thisCre[icn], title, xlim_frs, jet_cm, alph, mouse_id_allMice_thisCre[icn], dosavefig)


    ###################################
    ########## area2 - area2 ##########
    ###################################

    fgn = 'cc22'
    areas = 'V1', 'V1'

    top0 = cc22_sessAv_allMice_thisCre[icn]
    top_sd0 = cc22_sessSd_allMice_thisCre[icn]

    top1 = cc22_sessAv_allMice_thisCre_shfl[icn]
    top_sd1 = cc22_sessSd_allMice_thisCre_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p22_sessAv_allMice_thisCre[icn]
        top_sd2 = p22_sessSd_allMice_thisCre[icn]
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p22_sessAv_allMice_thisCre_shfl[icn]
        top_sd2 = p22_sessSd_allMice_thisCre_shfl[icn]    

    # replace the lower part of the matrix (that are nan) with the symmetric values on the upper side... so we get to see all cc traces in all subplots    
    for i2 in range(num_depth):
        for i1 in np.arange(i2+1, num_depth):
            top0[:,i1,i2] = top0[:,i2,i1]
            top1[:,i1,i2] = top1[:,i2,i1]            
            top2[:,i1,i2] = top2[:,i2,i1]
            top_sd0[:,i1,i2] = top_sd0[:,i2,i1]
            top_sd1[:,i1,i2] = top_sd1[:,i2,i1]            
            top_sd2[:,i1,i2] = top_sd2[:,i2,i1]


    plot_traces_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, fgn, areas, depth_sessAv_allMice_thisCre[icn], title, xlim_frs, jet_cm, alph, mouse_id_allMice_thisCre[icn], dosavefig)




#%% CC quantification: 4 subplots for each V1 layer. Each subplot shows cc quantification between a given V1 layer and all LM layers.    

if plot_cc_peaks:
    ###################################
    ########## area1 - area2 ##########
    ###################################

    fgn = 'cc12'
    areas = 'LM', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    f_top0 = cc12_peak_sessAv_allMice_thisCre_flash[icn]
    f_top_sd0 = cc12_peak_sessSd_allMice_thisCre_flash[icn]
    o_top0 = cc12_peak_sessAv_allMice_thisCre_omit[icn]
    o_top_sd0 = cc12_peak_sessSd_allMice_thisCre_omit[icn]

    f_top1 = cc12_peak_sessAv_allMice_thisCre_flash_shfl[icn]
    f_top_sd1 = cc12_peak_sessSd_allMice_thisCre_flash_shfl[icn]
    o_top1 = cc12_peak_sessAv_allMice_thisCre_omit_shfl[icn]
    o_top_sd1 = cc12_peak_sessSd_allMice_thisCre_omit_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p12_peak_sessAv_allMice_thisCre_flash[icn]
        f_top_sd2 = p12_peak_sessSd_allMice_thisCre_flash[icn]
        o_top2 = p12_peak_sessAv_allMice_thisCre_omit[icn]
        o_top_sd2 = p12_peak_sessSd_allMice_thisCre_omit[icn]
    else:
        f_top2 = p12_peak_sessAv_allMice_thisCre_flash_shfl[icn]
        f_top_sd2 = p12_peak_sessSd_allMice_thisCre_flash_shfl[icn]
        o_top2 = p12_peak_sessAv_allMice_thisCre_omit_shfl[icn]
        o_top_sd2 = p12_peak_sessSd_allMice_thisCre_omit_shfl[icn]


    plot_peak_cc_ccShfl_p(f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, fgn, areas, depth_sessAv_allMice_thisCre[icn], title, jet_cm, alph, mouse_id_allMice_thisCre[icn], dosavefig)



    ###################################
    ########## area1 - area1 ##########
    ###################################

    fgn = 'cc11'
    areas = 'LM', 'LM' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    f_top0 = cc11_peak_sessAv_allMice_thisCre_flash[icn]
    f_top_sd0 = cc11_peak_sessSd_allMice_thisCre_flash[icn]
    o_top0 = cc11_peak_sessAv_allMice_thisCre_omit[icn]
    o_top_sd0 = cc11_peak_sessSd_allMice_thisCre_omit[icn]

    f_top1 = cc11_peak_sessAv_allMice_thisCre_flash_shfl[icn]
    f_top_sd1 = cc11_peak_sessSd_allMice_thisCre_flash_shfl[icn]
    o_top1 = cc11_peak_sessAv_allMice_thisCre_omit_shfl[icn]
    o_top_sd1 = cc11_peak_sessSd_allMice_thisCre_omit_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p11_peak_sessAv_allMice_thisCre_flash[icn]
        f_top_sd2 = p11_peak_sessSd_allMice_thisCre_flash[icn]
        o_top2 = p11_peak_sessAv_allMice_thisCre_omit[icn]
        o_top_sd2 = p11_peak_sessSd_allMice_thisCre_omit[icn]
    else:
        f_top2 = p11_peak_sessAv_allMice_thisCre_flash_shfl[icn]
        f_top_sd2 = p11_peak_sessSd_allMice_thisCre_flash_shfl[icn]
        o_top2 = p11_peak_sessAv_allMice_thisCre_omit_shfl[icn]
        o_top_sd2 = p11_peak_sessSd_allMice_thisCre_omit_shfl[icn]

    # replace the lower part of the matrix (that are nan) with the symmetric values on the upper side... so we get to see all cc traces in all subplots    
    for i2 in range(num_depth):
        for i1 in np.arange(i2+1, num_depth):
            f_top0[i1,i2] = f_top0[i2,i1]
            f_top1[i1,i2] = f_top1[i2,i1]            
            f_top2[i1,i2] = f_top2[i2,i1]
            f_top_sd0[i1,i2] = f_top_sd0[i2,i1]
            f_top_sd1[i1,i2] = f_top_sd1[i2,i1]            
            f_top_sd2[i1,i2] = f_top_sd2[i2,i1]            
            o_top0[i1,i2] = o_top0[i2,i1]
            o_top1[i1,i2] = o_top1[i2,i1]            
            o_top2[i1,i2] = o_top2[i2,i1]
            o_top_sd0[i1,i2] = o_top_sd0[i2,i1]
            o_top_sd1[i1,i2] = o_top_sd1[i2,i1]            
            o_top_sd2[i1,i2] = o_top_sd2[i2,i1]


    plot_peak_cc_ccShfl_p(f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, fgn, areas, depth_sessAv_allMice_thisCre[icn], title, jet_cm, alph, mouse_id_allMice_thisCre[icn], dosavefig)



    ###################################
    ########## area2 - area2 ##########
    ###################################

    fgn = 'cc22'
    areas = 'V1', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    f_top0 = cc22_peak_sessAv_allMice_thisCre_flash[icn]
    f_top_sd0 = cc22_peak_sessSd_allMice_thisCre_flash[icn]
    o_top0 = cc22_peak_sessAv_allMice_thisCre_omit[icn]
    o_top_sd0 = cc22_peak_sessSd_allMice_thisCre_omit[icn]

    f_top1 = cc22_peak_sessAv_allMice_thisCre_flash_shfl[icn]
    f_top_sd1 = cc22_peak_sessSd_allMice_thisCre_flash_shfl[icn]
    o_top1 = cc22_peak_sessAv_allMice_thisCre_omit_shfl[icn]
    o_top_sd1 = cc22_peak_sessSd_allMice_thisCre_omit_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p22_peak_sessAv_allMice_thisCre_flash[icn]
        f_top_sd2 = p22_peak_sessSd_allMice_thisCre_flash[icn]
        o_top2 = p22_peak_sessAv_allMice_thisCre_omit[icn]
        o_top_sd2 = p22_peak_sessSd_allMice_thisCre_omit[icn]
    else:
        f_top2 = p22_peak_sessAv_allMice_thisCre_flash_shfl[icn]
        f_top_sd2 = p22_peak_sessSd_allMice_thisCre_flash_shfl[icn]
        o_top2 = p22_peak_sessAv_allMice_thisCre_omit_shfl[icn]
        o_top_sd2 = p22_peak_sessSd_allMice_thisCre_omit_shfl[icn]

    # replace the lower part of the matrix (that are nan) with the symmetric values on the upper side... so we get to see all cc traces in all subplots    
    for i2 in range(num_depth):
        for i1 in np.arange(i2+1, num_depth):
            f_top0[i1,i2] = f_top0[i2,i1]
            f_top1[i1,i2] = f_top1[i2,i1]            
            f_top2[i1,i2] = f_top2[i2,i1]
            f_top_sd0[i1,i2] = f_top_sd0[i2,i1]
            f_top_sd1[i1,i2] = f_top_sd1[i2,i1]            
            f_top_sd2[i1,i2] = f_top_sd2[i2,i1]            
            o_top0[i1,i2] = o_top0[i2,i1]
            o_top1[i1,i2] = o_top1[i2,i1]            
            o_top2[i1,i2] = o_top2[i2,i1]
            o_top_sd0[i1,i2] = o_top_sd0[i2,i1]
            o_top_sd1[i1,i2] = o_top_sd1[i2,i1]            
            o_top_sd2[i1,i2] = o_top_sd2[i2,i1]


    plot_peak_cc_ccShfl_p(f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, fgn, areas, depth_sessAv_allMice_thisCre[icn], title, jet_cm, alph, mouse_id_allMice_thisCre[icn], dosavefig)



#%% (Single-mouse plots): Plot pooled-layer corrs (pooled (each layer - all layers) and pooledAll (across areas, or within an area).)
#########################################################################################################

#%% Traces and cc magnitude quantifications

if plot_cc_pooled:
    
    ############################################################################################
    ########## area1 - area2 (each V1 layer with all LM layers; also all v1-lm pairs) ##########
    ############################################################################################

    fgn = 'cc12'
    areas = 'LM', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    # pooled; traces
    top0 = cc12_sessAv_pooled_allMice_thisCre[icn]
    top_sd0 = cc12_sessSd_pooled_allMice_thisCre[icn]

    top1 = [] #cc12_sessAv_pooled_allMice_thisCre_shfl[icn]
    top_sd1 = [] #cc12_sessSd_pooled_allMice_thisCre_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p12_sessAv_pooled_allMice_thisCre[icn]
        top_sd2 = p12_sessSd_pooled_allMice_thisCre[icn]
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p12_sessAv_pooled_allMice_thisCre_shfl[icn]
        top_sd2 = p12_sessSd_pooled_allMice_thisCre_shfl[icn]    


    # pooled; peaks
    f_top0 = cc12_peak_sessAv_pooled_allMice_thisCre_flash[icn]
    f_top_sd0 = cc12_peak_sessSd_pooled_allMice_thisCre_flash[icn]
    o_top0 = cc12_peak_sessAv_pooled_allMice_thisCre_omit[icn]
    o_top_sd0 = cc12_peak_sessSd_pooled_allMice_thisCre_omit[icn]

    f_top1 = [] #cc12_peak_sessAv_pooled_allMice_thisCre_flash_shfl[icn]
    f_top_sd1 = [] #cc12_peak_sessSd_pooled_allMice_thisCre_flash_shfl[icn]
    o_top1 = [] #cc12_peak_sessAv_pooled_allMice_thisCre_omit_shfl[icn]
    o_top_sd1 = [] #cc12_peak_sessSd_pooled_allMice_thisCre_omit_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p12_peak_sessAv_pooled_allMice_thisCre_flash[icn]
        f_top_sd2 = p12_peak_sessSd_pooled_allMice_thisCre_flash[icn]
        o_top2 = p12_peak_sessAv_pooled_allMice_thisCre_omit[icn]
        o_top_sd2 = p12_peak_sessSd_pooled_allMice_thisCre_omit[icn]
    else:
        f_top2 = p12_peak_sessAv_pooled_allMice_thisCre_flash_shfl[icn]
        f_top_sd2 = p12_peak_sessSd_pooled_allMice_thisCre_flash_shfl[icn]
        o_top2 = p12_peak_sessAv_pooled_allMice_thisCre_omit_shfl[icn]
        o_top_sd2 = p12_peak_sessSd_pooled_allMice_thisCre_omit_shfl[icn]


    ######## pooledAll; traces
    top0a = cc12_sessAv_pooledAll_allMice_thisCre[icn]
    top_sd0a = cc12_sessSd_pooledAll_allMice_thisCre[icn]

    top1a = [] #cc12_sessAv_pooledAll_allMice_thisCre_shfl[icn]
    top_sd1a = [] #cc12_sessSd_pooledAll_allMice_thisCre_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        top2a = p12_sessAv_pooledAll_allMice_thisCre[icn]
        top_sd2a = p12_sessSd_pooledAll_allMice_thisCre[icn]
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2a = p12_sessAv_pooledAll_allMice_thisCre_shfl[icn]
        top_sd2a = p12_sessSd_pooledAll_allMice_thisCre_shfl[icn]    


    # pooledAll; peaks
    f_top0a = cc12_peak_sessAv_pooledAll_allMice_thisCre_flash[icn]
    f_top_sd0a = cc12_peak_sessSd_pooledAll_allMice_thisCre_flash[icn]
    o_top0a = cc12_peak_sessAv_pooledAll_allMice_thisCre_omit[icn]
    o_top_sd0a = cc12_peak_sessSd_pooledAll_allMice_thisCre_omit[icn]

    f_top1a = [] #cc12_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl[icn]
    f_top_sd1a = [] #cc12_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl[icn]
    o_top1a = [] #cc12_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl[icn]
    o_top_sd1a = [] #cc12_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        f_top2a = p12_peak_sessAv_pooledAll_allMice_thisCre_flash[icn]
        f_top_sd2a = p12_peak_sessSd_pooledAll_allMice_thisCre_flash[icn]
        o_top2a = p12_peak_sessAv_pooledAll_allMice_thisCre_omit[icn]
        o_top_sd2a = p12_peak_sessSd_pooledAll_allMice_thisCre_omit[icn]
    else:
        f_top2a = p12_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl[icn]
        f_top_sd2a = p12_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl[icn]
        o_top2a = p12_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl[icn]
        o_top_sd2a = p12_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl[icn]



    plot_traces_peak_pooled_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, top0a, top_sd0a, top1a, top_sd1a, top2a, top_sd2a, \
        f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, \
        f_top0a, f_top_sd0a, o_top0a, o_top_sd0a, f_top1a, f_top_sd1a, o_top1a, o_top_sd1a, f_top2a, f_top_sd2a, o_top2a, o_top_sd2a, \
        fgn, areas, depth_sessAv_allMice_thisCre[icn], title, xlim_frs, jet_cm, alph, mouse_id_allMice_thisCre[icn], dosavefig)




    ############################################################################################
    ########## area1 - area1 (each LM layer with all other LM layers; also all lm-lm pairs) ##########
    ############################################################################################

    fgn = 'cc11'
    areas = 'LM', 'LM' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    # pooled; traces
    top0 = cc11_sessAv_pooled_allMice_thisCre[icn]
    top_sd0 = cc11_sessSd_pooled_allMice_thisCre[icn]

    top1 = [] #cc11_sessAv_pooled_allMice_thisCre_shfl[icn]
    top_sd1 = [] #cc11_sessSd_pooled_allMice_thisCre_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p11_sessAv_pooled_allMice_thisCre[icn]
        top_sd2 = p11_sessSd_pooled_allMice_thisCre[icn]
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p11_sessAv_pooled_allMice_thisCre_shfl[icn]
        top_sd2 = p11_sessSd_pooled_allMice_thisCre_shfl[icn]    


    # pooled; peaks
    f_top0 = cc11_peak_sessAv_pooled_allMice_thisCre_flash[icn]
    f_top_sd0 = cc11_peak_sessSd_pooled_allMice_thisCre_flash[icn]
    o_top0 = cc11_peak_sessAv_pooled_allMice_thisCre_omit[icn]
    o_top_sd0 = cc11_peak_sessSd_pooled_allMice_thisCre_omit[icn]

    f_top1 = [] #cc11_peak_sessAv_pooled_allMice_thisCre_flash_shfl[icn]
    f_top_sd1 = [] #cc11_peak_sessSd_pooled_allMice_thisCre_flash_shfl[icn]
    o_top1 = [] #cc11_peak_sessAv_pooled_allMice_thisCre_omit_shfl[icn]
    o_top_sd1 = [] #cc11_peak_sessSd_pooled_allMice_thisCre_omit_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p11_peak_sessAv_pooled_allMice_thisCre_flash[icn]
        f_top_sd2 = p11_peak_sessSd_pooled_allMice_thisCre_flash[icn]
        o_top2 = p11_peak_sessAv_pooled_allMice_thisCre_omit[icn]
        o_top_sd2 = p11_peak_sessSd_pooled_allMice_thisCre_omit[icn]
    else:
        f_top2 = p11_peak_sessAv_pooled_allMice_thisCre_flash_shfl[icn]
        f_top_sd2 = p11_peak_sessSd_pooled_allMice_thisCre_flash_shfl[icn]
        o_top2 = p11_peak_sessAv_pooled_allMice_thisCre_omit_shfl[icn]
        o_top_sd2 = p11_peak_sessSd_pooled_allMice_thisCre_omit_shfl[icn]


    ######## pooledAll; traces
    top0a = cc11_sessAv_pooledAll_allMice_thisCre[icn]
    top_sd0a = cc11_sessSd_pooledAll_allMice_thisCre[icn]

    top1a = [] #cc11_sessAv_pooledAll_allMice_thisCre_shfl[icn]
    top_sd1a = [] #cc11_sessSd_pooledAll_allMice_thisCre_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        top2a = p11_sessAv_pooledAll_allMice_thisCre[icn]
        top_sd2a = p11_sessSd_pooledAll_allMice_thisCre[icn]
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2a = p11_sessAv_pooledAll_allMice_thisCre_shfl[icn]
        top_sd2a = p11_sessSd_pooledAll_allMice_thisCre_shfl[icn]    


    # pooledAll; peaks
    f_top0a = cc11_peak_sessAv_pooledAll_allMice_thisCre_flash[icn]
    f_top_sd0a = cc11_peak_sessSd_pooledAll_allMice_thisCre_flash[icn]
    o_top0a = cc11_peak_sessAv_pooledAll_allMice_thisCre_omit[icn]
    o_top_sd0a = cc11_peak_sessSd_pooledAll_allMice_thisCre_omit[icn]

    f_top1a = [] #cc11_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl[icn]
    f_top_sd1a = [] #cc11_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl[icn]
    o_top1a = [] #cc11_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl[icn]
    o_top_sd1a = [] #cc11_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        f_top2a = p11_peak_sessAv_pooledAll_allMice_thisCre_flash[icn]
        f_top_sd2a = p11_peak_sessSd_pooledAll_allMice_thisCre_flash[icn]
        o_top2a = p11_peak_sessAv_pooledAll_allMice_thisCre_omit[icn]
        o_top_sd2a = p11_peak_sessSd_pooledAll_allMice_thisCre_omit[icn]
    else:
        f_top2a = p11_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl[icn]
        f_top_sd2a = p11_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl[icn]
        o_top2a = p11_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl[icn]
        o_top_sd2a = p11_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl[icn]



    plot_traces_peak_pooled_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, top0a, top_sd0a, top1a, top_sd1a, top2a, top_sd2a, \
        f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, \
        f_top0a, f_top_sd0a, o_top0a, o_top_sd0a, f_top1a, f_top_sd1a, o_top1a, o_top_sd1a, f_top2a, f_top_sd2a, o_top2a, o_top_sd2a, \
        fgn, areas, depth_sessAv_allMice_thisCre[icn], title, xlim_frs, jet_cm, alph, mouse_id_allMice_thisCre[icn], dosavefig)



    ############################################################################################
    ########## area2 - area2 (each V1 layer with all other V1 layers; also all v1-v1 pairs) ##########
    ############################################################################################

    fgn = 'cc22'
    areas = 'V1', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    # pooled; traces
    top0 = cc22_sessAv_pooled_allMice_thisCre[icn]
    top_sd0 = cc22_sessSd_pooled_allMice_thisCre[icn]

    top1 = [] #cc22_sessAv_pooled_allMice_thisCre_shfl[icn]
    top_sd1 = [] #cc22_sessSd_pooled_allMice_thisCre_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p22_sessAv_pooled_allMice_thisCre[icn]
        top_sd2 = p22_sessSd_pooled_allMice_thisCre[icn]
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p22_sessAv_pooled_allMice_thisCre_shfl[icn]
        top_sd2 = p22_sessSd_pooled_allMice_thisCre_shfl[icn]    


    # pooled; peaks
    f_top0 = cc22_peak_sessAv_pooled_allMice_thisCre_flash[icn]
    f_top_sd0 = cc22_peak_sessSd_pooled_allMice_thisCre_flash[icn]
    o_top0 = cc22_peak_sessAv_pooled_allMice_thisCre_omit[icn]
    o_top_sd0 = cc22_peak_sessSd_pooled_allMice_thisCre_omit[icn]

    f_top1 = [] #cc22_peak_sessAv_pooled_allMice_thisCre_flash_shfl[icn]
    f_top_sd1 = [] #cc22_peak_sessSd_pooled_allMice_thisCre_flash_shfl[icn]
    o_top1 = [] #cc22_peak_sessAv_pooled_allMice_thisCre_omit_shfl[icn]
    o_top_sd1 = [] #cc22_peak_sessSd_pooled_allMice_thisCre_omit_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p22_peak_sessAv_pooled_allMice_thisCre_flash[icn]
        f_top_sd2 = p22_peak_sessSd_pooled_allMice_thisCre_flash[icn]
        o_top2 = p22_peak_sessAv_pooled_allMice_thisCre_omit[icn]
        o_top_sd2 = p22_peak_sessSd_pooled_allMice_thisCre_omit[icn]
    else:
        f_top2 = p22_peak_sessAv_pooled_allMice_thisCre_flash_shfl[icn]
        f_top_sd2 = p22_peak_sessSd_pooled_allMice_thisCre_flash_shfl[icn]
        o_top2 = p22_peak_sessAv_pooled_allMice_thisCre_omit_shfl[icn]
        o_top_sd2 = p22_peak_sessSd_pooled_allMice_thisCre_omit_shfl[icn]


    ######## pooledAll; traces
    top0a = cc22_sessAv_pooledAll_allMice_thisCre[icn]
    top_sd0a = cc22_sessSd_pooledAll_allMice_thisCre[icn]

    top1a = [] #cc22_sessAv_pooledAll_allMice_thisCre_shfl[icn]
    top_sd1a = [] #cc22_sessSd_pooledAll_allMice_thisCre_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        top2a = p22_sessAv_pooledAll_allMice_thisCre[icn]
        top_sd2a = p22_sessSd_pooledAll_allMice_thisCre[icn]
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2a = p22_sessAv_pooledAll_allMice_thisCre_shfl[icn]
        top_sd2a = p22_sessSd_pooledAll_allMice_thisCre_shfl[icn]    


    # pooledAll; peaks
    f_top0a = cc22_peak_sessAv_pooledAll_allMice_thisCre_flash[icn]
    f_top_sd0a = cc22_peak_sessSd_pooledAll_allMice_thisCre_flash[icn]
    o_top0a = cc22_peak_sessAv_pooledAll_allMice_thisCre_omit[icn]
    o_top_sd0a = cc22_peak_sessSd_pooledAll_allMice_thisCre_omit[icn]

    f_top1a = [] #cc22_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl[icn]
    f_top_sd1a = [] #cc22_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl[icn]
    o_top1a = [] #cc22_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl[icn]
    o_top_sd1a = [] #cc22_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl[icn]

    if use_spearman_p: # p that came out of spearman corr package
        f_top2a = p22_peak_sessAv_pooledAll_allMice_thisCre_flash[icn]
        f_top_sd2a = p22_peak_sessSd_pooledAll_allMice_thisCre_flash[icn]
        o_top2a = p22_peak_sessAv_pooledAll_allMice_thisCre_omit[icn]
        o_top_sd2a = p22_peak_sessSd_pooledAll_allMice_thisCre_omit[icn]
    else:
        f_top2a = p22_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl[icn]
        f_top_sd2a = p22_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl[icn]
        o_top2a = p22_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl[icn]
        o_top_sd2a = p22_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl[icn]



    plot_traces_peak_pooled_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, top0a, top_sd0a, top1a, top_sd1a, top2a, top_sd2a, \
        f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, \
        f_top0a, f_top_sd0a, o_top0a, o_top_sd0a, f_top1a, f_top_sd1a, o_top1a, o_top_sd1a, f_top2a, f_top_sd2a, o_top2a, o_top_sd2a, \
        fgn, areas, depth_sessAv_allMice_thisCre[icn], title, xlim_frs, jet_cm, alph, mouse_id_allMice_thisCre[icn], dosavefig)
