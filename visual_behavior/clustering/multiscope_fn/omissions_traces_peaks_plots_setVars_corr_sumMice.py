#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gets called in omissions_traces_peaks_plots_setVars_corr_eachCre.py
Plot summary of mice all of a cre line

Created on Tue Oct  8 18:30:23 2019
@author: farzaneh
"""

title = '%s (%d mice)' %(cre, thisCre_numMice)

#%%
######################## Plots of mice-averaged data, for each cre line ########################

#%% Traces: 4 subplots for each V1 layer. Each subplot shows cc between a given V1 layer and all LM layers.    

if plot_cc_traces:

    ###################################
    ########## area1 - area2 ##########
    ###################################

    fgn = 'cc12'
    areas = 'LM', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    top0 = cc12_sessAv_avMice
    top_sd0 = cc12_sessAv_sdMice    

    top1 = cc12_sessAv_avMice_shfl
    top_sd1 = cc12_sessAv_sdMice_shfl    

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p12_sessAv_avMice
        top_sd2 = p12_sessAv_sdMice
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p12_sessAv_avMice_shfl
        top_sd2 = p12_sessAv_sdMice_shfl    


    plot_traces_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, fgn, areas, depth_sessAv_avMice, title, xlim_frs, jet_cm, alph, mouseid=np.nan, dosavefig=dosavefig)



    ###################################
    ########## area1 - area1 ##########
    ###################################

    fgn = 'cc11'
    areas = 'LM', 'LM'

    top0 = cc11_sessAv_avMice
    top_sd0 = cc11_sessAv_sdMice    

    top1 = cc11_sessAv_avMice_shfl
    top_sd1 = cc11_sessAv_sdMice_shfl    

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p11_sessAv_avMice
        top_sd2 = p11_sessAv_sdMice
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p11_sessAv_avMice_shfl
        top_sd2 = p11_sessAv_sdMice_shfl    

    # replace the lower part of the matrix (that are nan) with the symmetric values on the upper side... so we get to see all cc traces in all subplots    
    for i2 in range(num_depth):
        for i1 in np.arange(i2+1, num_depth):
            top0[:,i1,i2] = top0[:,i2,i1]
            top1[:,i1,i2] = top1[:,i2,i1]            
            top2[:,i1,i2] = top2[:,i2,i1]
            top_sd0[:,i1,i2] = top_sd0[:,i2,i1]
            top_sd1[:,i1,i2] = top_sd1[:,i2,i1]            
            top_sd2[:,i1,i2] = top_sd2[:,i2,i1]


    plot_traces_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, fgn, areas, depth_sessAv_avMice, title, xlim_frs, jet_cm, alph, mouseid=np.nan, dosavefig=dosavefig)



    ###################################
    ########## area2 - area2 ##########
    ###################################

    fgn = 'cc22'
    areas = 'V1', 'V1'

    top0 = cc22_sessAv_avMice
    top_sd0 = cc22_sessAv_sdMice    

    top1 = cc22_sessAv_avMice_shfl
    top_sd1 = cc22_sessAv_sdMice_shfl    

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p22_sessAv_avMice
        top_sd2 = p22_sessAv_sdMice
    else:    # p that I computed by doing ttest of cc_shfl and cc
        top2 = p22_sessAv_avMice_shfl
        top_sd2 = p22_sessAv_sdMice_shfl    

    # replace the lower part of the matrix (that are nan) with the symmetric values on the upper side... so we get to see all cc traces in all subplots    
    for i2 in range(num_depth):
        for i1 in np.arange(i2+1, num_depth):
            top0[:,i1,i2] = top0[:,i2,i1]
            top1[:,i1,i2] = top1[:,i2,i1]            
            top2[:,i1,i2] = top2[:,i2,i1]
            top_sd0[:,i1,i2] = top_sd0[:,i2,i1]
            top_sd1[:,i1,i2] = top_sd1[:,i2,i1]            
            top_sd2[:,i1,i2] = top_sd2[:,i2,i1]


    plot_traces_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, fgn, areas, depth_sessAv_avMice, title, xlim_frs, jet_cm, alph, mouseid=np.nan, dosavefig=dosavefig)





#%% CC quantification: 4 subplots for each V1 layer. Each subplot shows cc quantification between a given V1 layer and all LM layers.    

if plot_cc_peaks:

    ###################################
    ########## area1 - area2 ##########
    ###################################

    fgn = 'cc12'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    
    areas = 'LM', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    f_top0 = cc12_peak_avMice_flash
    f_top_sd0 = cc12_peak_sdMice_flash
    o_top0 = cc12_peak_avMice_omit
    o_top_sd0 = cc12_peak_sdMice_omit

    f_top1 = cc12_peak_avMice_flash_shfl
    f_top_sd1 = cc12_peak_sdMice_flash_shfl
    o_top1 = cc12_peak_avMice_omit_shfl
    o_top_sd1 = cc12_peak_sdMice_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p12_peak_avMice_flash
        f_top_sd2 = p12_peak_sdMice_flash
        o_top2 = p12_peak_avMice_omit
        o_top_sd2 = p12_peak_sdMice_omit
    else:
        f_top2 = p12_peak_avMice_flash_shfl
        f_top_sd2 = p12_peak_sdMice_flash_shfl
        o_top2 = p12_peak_avMice_omit_shfl
        o_top_sd2 = p12_peak_sdMice_omit_shfl


    plot_peak_cc_ccShfl_p(f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, fgn, areas, depth_sessAv_avMice, title, jet_cm, alph, mouseid=np.nan, dosavefig=dosavefig)



    ###################################
    ########## area1 - area1 ##########
    ###################################

    fgn = 'cc11'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    
    areas = 'LM', 'LM' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    f_top0 = cc11_peak_avMice_flash
    f_top_sd0 = cc11_peak_sdMice_flash
    o_top0 = cc11_peak_avMice_omit
    o_top_sd0 = cc11_peak_sdMice_omit

    f_top1 = cc11_peak_avMice_flash_shfl
    f_top_sd1 = cc11_peak_sdMice_flash_shfl
    o_top1 = cc11_peak_avMice_omit_shfl
    o_top_sd1 = cc11_peak_sdMice_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p11_peak_avMice_flash
        f_top_sd2 = p11_peak_sdMice_flash
        o_top2 = p11_peak_avMice_omit
        o_top_sd2 = p11_peak_sdMice_omit
    else:
        f_top2 = p11_peak_avMice_flash_shfl
        f_top_sd2 = p11_peak_sdMice_flash_shfl
        o_top2 = p11_peak_avMice_omit_shfl
        o_top_sd2 = p11_peak_sdMice_omit_shfl

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


    plot_peak_cc_ccShfl_p(f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, fgn, areas, depth_sessAv_avMice, title, jet_cm, alph, mouseid=np.nan, dosavefig=dosavefig)



    ###################################
    ########## area2 - area2 ##########
    ###################################

    fgn = 'cc22'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    
    areas = 'V1', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    f_top0 = cc22_peak_avMice_flash
    f_top_sd0 = cc22_peak_sdMice_flash
    o_top0 = cc22_peak_avMice_omit
    o_top_sd0 = cc22_peak_sdMice_omit

    f_top1 = cc22_peak_avMice_flash_shfl
    f_top_sd1 = cc22_peak_sdMice_flash_shfl
    o_top1 = cc22_peak_avMice_omit_shfl
    o_top_sd1 = cc22_peak_sdMice_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p22_peak_avMice_flash
        f_top_sd2 = p22_peak_sdMice_flash
        o_top2 = p22_peak_avMice_omit
        o_top_sd2 = p22_peak_sdMice_omit
    else:
        f_top2 = p22_peak_avMice_flash_shfl
        f_top_sd2 = p22_peak_sdMice_flash_shfl
        o_top2 = p22_peak_avMice_omit_shfl
        o_top_sd2 = p22_peak_sdMice_omit_shfl

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


    plot_peak_cc_ccShfl_p(f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, fgn, areas, depth_sessAv_avMice, title, jet_cm, alph, mouseid=np.nan, dosavefig=dosavefig)



    
#%% Plot heatmaps for cc quantification: 4x4 heatmaps for cc quantification of each layer pair; for flash and omit evoked responses.

if plot_cc_peaks_heatmaps:

    ###################################
    ########## area1 - area2 ##########
    ###################################

    fgn = 'cc12'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    
    areas = 'LM', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    f_top0 = cc12_peak_avMice_flash
    o_top0 = cc12_peak_avMice_omit

    f_top1 = cc12_peak_avMice_flash_shfl
    o_top1 = cc12_peak_avMice_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p12_peak_avMice_flash
        o_top2 = p12_peak_avMice_omit
    else:
        f_top2 = p12_peak_avMice_flash_shfl
        o_top2 = p12_peak_avMice_omit_shfl


    plot_heatmap_peak_cc_ccShfl_p(f_top0, o_top0, f_top1, o_top1, f_top2, o_top2, fgn, areas, depth_sessAv_avMice, title, mouseid=np.nan, dosavefig=dosavefig)



    ###################################
    ########## area1 - area1 ##########
    ###################################

    fgn = 'cc11'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    
    areas = 'LM', 'LM' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    f_top0 = cc11_peak_avMice_flash
    o_top0 = cc11_peak_avMice_omit

    f_top1 = cc11_peak_avMice_flash_shfl
    o_top1 = cc11_peak_avMice_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p11_peak_avMice_flash
        o_top2 = p11_peak_avMice_omit
    else:
        f_top2 = p11_peak_avMice_flash_shfl
        o_top2 = p11_peak_avMice_omit_shfl

    # replace the lower part of the matrix (that are nan) with the symmetric values on the upper side... so we get to see all cc traces in all subplots    
    for i2 in range(num_depth):
        for i1 in np.arange(i2+1, num_depth):
            f_top0[i1,i2] = f_top0[i2,i1]
            f_top1[i1,i2] = f_top1[i2,i1]            
            f_top2[i1,i2] = f_top2[i2,i1]
            o_top0[i1,i2] = o_top0[i2,i1]
            o_top1[i1,i2] = o_top1[i2,i1]            
            o_top2[i1,i2] = o_top2[i2,i1]


    plot_heatmap_peak_cc_ccShfl_p(f_top0, o_top0, f_top1, o_top1, f_top2, o_top2, fgn, areas, depth_sessAv_avMice, title, mouseid=np.nan, dosavefig=dosavefig)



    ###################################
    ########## area2 - area2 ##########
    ###################################

    fgn = 'cc22'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    
    areas = 'V1', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    f_top0 = cc22_peak_avMice_flash
    o_top0 = cc22_peak_avMice_omit

    f_top1 = cc22_peak_avMice_flash_shfl
    o_top1 = cc22_peak_avMice_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p22_peak_avMice_flash
        o_top2 = p22_peak_avMice_omit
    else:
        f_top2 = p22_peak_avMice_flash_shfl
        o_top2 = p22_peak_avMice_omit_shfl

    # replace the lower part of the matrix (that are nan) with the symmetric values on the upper side... so we get to see all cc traces in all subplots    
    for i2 in range(num_depth):
        for i1 in np.arange(i2+1, num_depth):
            f_top0[i1,i2] = f_top0[i2,i1]
            f_top1[i1,i2] = f_top1[i2,i1]            
            f_top2[i1,i2] = f_top2[i2,i1]
            o_top0[i1,i2] = o_top0[i2,i1]
            o_top1[i1,i2] = o_top1[i2,i1]            
            o_top2[i1,i2] = o_top2[i2,i1]


    plot_heatmap_peak_cc_ccShfl_p(f_top0, o_top0, f_top1, o_top1, f_top2, o_top2, fgn, areas, depth_sessAv_avMice, title, mouseid=np.nan, dosavefig=dosavefig)


        
        

#%% Plot pooled-layer corrs (pooled (each layer - all layers) and pooledAll (across areas, or within an area).)
# (Mice-averaged plots) 
#########################################################################################################

#%% Traces and cc magnitude quantifications

if plot_cc_pooled:

    ############################################################################################
    ########## area1 - area2 (each V1 layer with all LM layers; also all v1-lm pairs) ##########
    ############################################################################################

    fgn = 'cc12'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    
    areas = 'LM', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    # pooled; traces
    top0 = cc12_sessAv_pooled_avMice
    top_sd0 = cc12_sessAv_pooled_sdMice

    top1 = [] #cc12_sessAv_pooled_avMice_shfl
    top_sd1 = [] #cc12_sessAv_pooled_sdMice_shfl

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p12_sessAv_pooled_avMice
        top_sd2 = p12_sessAv_pooled_sdMice
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p12_sessAv_pooled_avMice_shfl
        top_sd2 = p12_sessAv_pooled_sdMice_shfl    


    # pooled; peaks
    f_top0 = cc12_peak_avMice_pooled_flash
    f_top_sd0 = cc12_peak_sdMice_pooled_flash
    o_top0 = cc12_peak_avMice_pooled_omit
    o_top_sd0 = cc12_peak_sdMice_pooled_omit

    f_top1 = [] #cc12_peak_avMice_pooled_flash_shfl
    f_top_sd1 = [] #cc12_peak_sdMice_pooled_flash_shfl
    o_top1 = [] #cc12_peak_avMice_pooled_omit_shfl
    o_top_sd1 = [] #cc12_peak_sdMice_pooled_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p12_peak_avMice_pooled_flash
        f_top_sd2 = p12_peak_sdMice_pooled_flash
        o_top2 = p12_peak_avMice_pooled_omit
        o_top_sd2 = p12_peak_sdMice_pooled_omit
    else:
        f_top2 = p12_peak_avMice_pooled_flash_shfl
        f_top_sd2 = p12_peak_sdMice_pooled_flash_shfl
        o_top2 = p12_peak_avMice_pooled_omit_shfl
        o_top_sd2 = p12_peak_sdMice_pooled_omit_shfl


    ######## pooledAll; traces
    top0a = cc12_sessAv_pooledAll_avMice
    top_sd0a = cc12_sessAv_pooledAll_sdMice

    top1a = [] #cc12_sessAv_pooledAll_avMice_shfl
    top_sd1a = [] #cc12_sessAv_pooledAll_sdMice_shfl

    if use_spearman_p: # p that came out of spearman corr package
        top2a = p12_sessAv_pooledAll_avMice
        top_sd2a = p12_sessAv_pooledAll_sdMice
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2a = p12_sessAv_pooledAll_avMice_shfl
        top_sd2a = p12_sessAv_pooledAll_sdMice_shfl    


    # pooledAll; peaks
    f_top0a = cc12_peak_avMice_pooledAll_flash
    f_top_sd0a = cc12_peak_sdMice_pooledAll_flash
    o_top0a = cc12_peak_avMice_pooledAll_omit
    o_top_sd0a = cc12_peak_sdMice_pooledAll_omit

    f_top1a = [] #cc12_peak_avMice_pooledAll_flash_shfl
    f_top_sd1a = [] #cc12_peak_sdMice_pooledAll_flash_shfl
    o_top1a = [] #cc12_peak_avMice_pooledAll_omit_shfl
    o_top_sd1a = [] #cc12_peak_sdMice_pooledAll_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2a = p12_peak_avMice_pooledAll_flash
        f_top_sd2a = p12_peak_sdMice_pooledAll_flash
        o_top2a = p12_peak_avMice_pooledAll_omit
        o_top_sd2a = p12_peak_sdMice_pooledAll_omit
    else:
        f_top2a = p12_peak_avMice_pooledAll_flash_shfl
        f_top_sd2a = p12_peak_sdMice_pooledAll_flash_shfl
        o_top2a = p12_peak_avMice_pooledAll_omit_shfl
        o_top_sd2a = p12_peak_sdMice_pooledAll_omit_shfl



    plot_traces_peak_pooled_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, top0a, top_sd0a, top1a, top_sd1a, top2a, top_sd2a, \
        f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, \
        f_top0a, f_top_sd0a, o_top0a, o_top_sd0a, f_top1a, f_top_sd1a, o_top1a, o_top_sd1a, f_top2a, f_top_sd2a, o_top2a, o_top_sd2a, \
        fgn, areas, depth_sessAv_avMice, title, xlim_frs, jet_cm, alph, mouseid=np.nan, dosavefig=dosavefig)



    ############################################################################################
    ########## area1 - area1 (each LM layer with all LM layers; also all lm-lm pairs) ##########
    ############################################################################################

    fgn = 'cc11'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    
    areas = 'LM', 'LM' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    # pooled; traces
    top0 = cc11_sessAv_pooled_avMice
    top_sd0 = cc11_sessAv_pooled_sdMice

    top1 = [] #cc11_sessAv_pooled_avMice_shfl
    top_sd1 = [] #cc11_sessAv_pooled_sdMice_shfl

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p11_sessAv_pooled_avMice
        top_sd2 = p11_sessAv_pooled_sdMice
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p11_sessAv_pooled_avMice_shfl
        top_sd2 = p11_sessAv_pooled_sdMice_shfl    


    # pooled; peaks
    f_top0 = cc11_peak_avMice_pooled_flash
    f_top_sd0 = cc11_peak_sdMice_pooled_flash
    o_top0 = cc11_peak_avMice_pooled_omit
    o_top_sd0 = cc11_peak_sdMice_pooled_omit

    f_top1 = [] #cc11_peak_avMice_pooled_flash_shfl
    f_top_sd1 = [] #cc11_peak_sdMice_pooled_flash_shfl
    o_top1 = [] #cc11_peak_avMice_pooled_omit_shfl
    o_top_sd1 = [] #cc11_peak_sdMice_pooled_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p11_peak_avMice_pooled_flash
        f_top_sd2 = p11_peak_sdMice_pooled_flash
        o_top2 = p11_peak_avMice_pooled_omit
        o_top_sd2 = p11_peak_sdMice_pooled_omit
    else:
        f_top2 = p11_peak_avMice_pooled_flash_shfl
        f_top_sd2 = p11_peak_sdMice_pooled_flash_shfl
        o_top2 = p11_peak_avMice_pooled_omit_shfl
        o_top_sd2 = p11_peak_sdMice_pooled_omit_shfl


    ######## pooledAll; traces
    top0a = cc11_sessAv_pooledAll_avMice
    top_sd0a = cc11_sessAv_pooledAll_sdMice

    top1a = [] #cc11_sessAv_pooledAll_avMice_shfl
    top_sd1a = [] #cc11_sessAv_pooledAll_sdMice_shfl

    if use_spearman_p: # p that came out of spearman corr package
        top2a = p11_sessAv_pooledAll_avMice
        top_sd2a = p11_sessAv_pooledAll_sdMice
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2a = p11_sessAv_pooledAll_avMice_shfl
        top_sd2a = p11_sessAv_pooledAll_sdMice_shfl    


    # pooledAll; peaks
    f_top0a = cc11_peak_avMice_pooledAll_flash
    f_top_sd0a = cc11_peak_sdMice_pooledAll_flash
    o_top0a = cc11_peak_avMice_pooledAll_omit
    o_top_sd0a = cc11_peak_sdMice_pooledAll_omit

    f_top1a = [] #cc11_peak_avMice_pooledAll_flash_shfl
    f_top_sd1a = [] #cc11_peak_sdMice_pooledAll_flash_shfl
    o_top1a = [] #cc11_peak_avMice_pooledAll_omit_shfl
    o_top_sd1a = [] #cc11_peak_sdMice_pooledAll_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2a = p11_peak_avMice_pooledAll_flash
        f_top_sd2a = p11_peak_sdMice_pooledAll_flash
        o_top2a = p11_peak_avMice_pooledAll_omit
        o_top_sd2a = p11_peak_sdMice_pooledAll_omit
    else:
        f_top2a = p11_peak_avMice_pooledAll_flash_shfl
        f_top_sd2a = p11_peak_sdMice_pooledAll_flash_shfl
        o_top2a = p11_peak_avMice_pooledAll_omit_shfl
        o_top_sd2a = p11_peak_sdMice_pooledAll_omit_shfl



    plot_traces_peak_pooled_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, top0a, top_sd0a, top1a, top_sd1a, top2a, top_sd2a, \
        f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, \
        f_top0a, f_top_sd0a, o_top0a, o_top_sd0a, f_top1a, f_top_sd1a, o_top1a, o_top_sd1a, f_top2a, f_top_sd2a, o_top2a, o_top_sd2a, \
        fgn, areas, depth_sessAv_avMice, title, xlim_frs, jet_cm, alph, mouseid=np.nan, dosavefig=dosavefig)



    ############################################################################################
    ########## area2 - area2 (each V1 layer with all V1 layers; also all v1-v1 pairs) ##########
    ############################################################################################

    fgn = 'cc22'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    
    areas = 'V1', 'V1' # np.unique(corr_trace_peak_allMice['areas'].iloc[im])

    # pooled; traces
    top0 = cc22_sessAv_pooled_avMice
    top_sd0 = cc22_sessAv_pooled_sdMice

    top1 = [] #cc22_sessAv_pooled_avMice_shfl
    top_sd1 = [] #cc22_sessAv_pooled_sdMice_shfl

    if use_spearman_p: # p that came out of spearman corr package
        top2 = p22_sessAv_pooled_avMice
        top_sd2 = p22_sessAv_pooled_sdMice
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2 = p22_sessAv_pooled_avMice_shfl
        top_sd2 = p22_sessAv_pooled_sdMice_shfl    


    # pooled; peaks
    f_top0 = cc22_peak_avMice_pooled_flash
    f_top_sd0 = cc22_peak_sdMice_pooled_flash
    o_top0 = cc22_peak_avMice_pooled_omit
    o_top_sd0 = cc22_peak_sdMice_pooled_omit

    f_top1 = [] #cc22_peak_avMice_pooled_flash_shfl
    f_top_sd1 = [] #cc22_peak_sdMice_pooled_flash_shfl
    o_top1 = [] #cc22_peak_avMice_pooled_omit_shfl
    o_top_sd1 = [] #cc22_peak_sdMice_pooled_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2 = p22_peak_avMice_pooled_flash
        f_top_sd2 = p22_peak_sdMice_pooled_flash
        o_top2 = p22_peak_avMice_pooled_omit
        o_top_sd2 = p22_peak_sdMice_pooled_omit
    else:
        f_top2 = p22_peak_avMice_pooled_flash_shfl
        f_top_sd2 = p22_peak_sdMice_pooled_flash_shfl
        o_top2 = p22_peak_avMice_pooled_omit_shfl
        o_top_sd2 = p22_peak_sdMice_pooled_omit_shfl


    ######## pooledAll; traces
    top0a = cc22_sessAv_pooledAll_avMice
    top_sd0a = cc22_sessAv_pooledAll_sdMice

    top1a = [] #cc22_sessAv_pooledAll_avMice_shfl
    top_sd1a = [] #cc22_sessAv_pooledAll_sdMice_shfl

    if use_spearman_p: # p that came out of spearman corr package
        top2a = p22_sessAv_pooledAll_avMice
        top_sd2a = p22_sessAv_pooledAll_sdMice
    else: # p that I computed by doing ttest of cc_shfl and cc
        top2a = p22_sessAv_pooledAll_avMice_shfl
        top_sd2a = p22_sessAv_pooledAll_sdMice_shfl    


    # pooledAll; peaks
    f_top0a = cc22_peak_avMice_pooledAll_flash
    f_top_sd0a = cc22_peak_sdMice_pooledAll_flash
    o_top0a = cc22_peak_avMice_pooledAll_omit
    o_top_sd0a = cc22_peak_sdMice_pooledAll_omit

    f_top1a = [] #cc22_peak_avMice_pooledAll_flash_shfl
    f_top_sd1a = [] #cc22_peak_sdMice_pooledAll_flash_shfl
    o_top1a = [] #cc22_peak_avMice_pooledAll_omit_shfl
    o_top_sd1a = [] #cc22_peak_sdMice_pooledAll_omit_shfl

    if use_spearman_p: # p that came out of spearman corr package
        f_top2a = p22_peak_avMice_pooledAll_flash
        f_top_sd2a = p22_peak_sdMice_pooledAll_flash
        o_top2a = p22_peak_avMice_pooledAll_omit
        o_top_sd2a = p22_peak_sdMice_pooledAll_omit
    else:
        f_top2a = p22_peak_avMice_pooledAll_flash_shfl
        f_top_sd2a = p22_peak_sdMice_pooledAll_flash_shfl
        o_top2a = p22_peak_avMice_pooledAll_omit_shfl
        o_top_sd2a = p22_peak_sdMice_pooledAll_omit_shfl



    plot_traces_peak_pooled_cc_ccShfl_p(time_trace, top0, top_sd0, top1, top_sd1, top2, top_sd2, top0a, top_sd0a, top1a, top_sd1a, top2a, top_sd2a, \
        f_top0, f_top_sd0, o_top0, o_top_sd0, f_top1, f_top_sd1, o_top1, o_top_sd1, f_top2, f_top_sd2, o_top2, o_top_sd2, \
        f_top0a, f_top_sd0a, o_top0a, o_top_sd0a, f_top1a, f_top_sd1a, o_top1a, o_top_sd1a, f_top2a, f_top_sd2a, o_top2a, o_top_sd2a, \
        fgn, areas, depth_sessAv_avMice, title, xlim_frs, jet_cm, alph, mouseid=np.nan, dosavefig=dosavefig)



    
    
    
#%% Plot superimposed pooledAll traces for cc12, cc11, cc22
# also make errorbar plots for flash (cc12,cc11,cc22) and omission (cc12,cc11,cc22).

if plot_cc_pooled_superimposed:

    areas = 'LM', 'V1'

    ########################################################################
    ########## cc traces and peaks ##################################
    ########################################################################

    yl = 'Spearman cc'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    
    fgn = 'ccAll'

    # pooledAll; traces
    top0a = cc12_sessAv_pooledAll_avMice
    top_sd0a = cc12_sessAv_pooledAll_sdMice

    top1a = cc11_sessAv_pooledAll_avMice
    top_sd1a = cc11_sessAv_pooledAll_sdMice

    top2a = cc22_sessAv_pooledAll_avMice
    top_sd2a = cc22_sessAv_pooledAll_sdMice


    # pooledAll; peaks
    # V1-LM
    f_top0a = cc12_peak_avMice_pooledAll_flash
    f_top_sd0a = cc12_peak_sdMice_pooledAll_flash
    o_top0a = cc12_peak_avMice_pooledAll_omit
    o_top_sd0a = cc12_peak_sdMice_pooledAll_omit
    # LM
    f_top1a = cc11_peak_avMice_pooledAll_flash
    f_top_sd1a = cc11_peak_sdMice_pooledAll_flash
    o_top1a = cc11_peak_avMice_pooledAll_omit
    o_top_sd1a = cc11_peak_sdMice_pooledAll_omit
    # V1
    f_top2a = cc22_peak_avMice_pooledAll_flash
    f_top_sd2a = cc22_peak_sdMice_pooledAll_flash
    o_top2a = cc22_peak_avMice_pooledAll_omit
    o_top_sd2a = cc22_peak_sdMice_pooledAll_omit


    plot_superimposed_pooled_cc_ccShfl_p(time_trace, top0a, top_sd0a, top1a, top_sd1a, top2a, top_sd2a, \
        f_top0a, f_top_sd0a, o_top0a, o_top_sd0a, f_top1a, f_top_sd1a, o_top1a, o_top_sd1a, f_top2a, f_top_sd2a, o_top2a, o_top_sd2a, \
        fgn, areas, yl, title, xlim_frs, mouseid=np.nan, dosavefig=dosavefig)



    ########################################################################
    ########## p (fraction neuron pairs with significant cc ##################################
    ########################################################################

    yl = 'Fract sig neuron pairs'
    fgn = 'pAll'
    if sameBl_allLayerPairs==1:
        fgn = fgn + '_sameBl'
    

    # pooledAll; traces
    top0a = p12_sessAv_pooledAll_avMice
    top_sd0a = p12_sessAv_pooledAll_sdMice

    top1a = p11_sessAv_pooledAll_avMice
    top_sd1a = p11_sessAv_pooledAll_sdMice

    top2a = p22_sessAv_pooledAll_avMice
    top_sd2a = p22_sessAv_pooledAll_sdMice


    # pooledAll; peaks
    # V1-LM
    f_top0a = p12_peak_avMice_pooledAll_flash
    f_top_sd0a = p12_peak_sdMice_pooledAll_flash
    o_top0a = p12_peak_avMice_pooledAll_omit
    o_top_sd0a = p12_peak_sdMice_pooledAll_omit
    # LM
    f_top1a = p11_peak_avMice_pooledAll_flash
    f_top_sd1a = p11_peak_sdMice_pooledAll_flash
    o_top1a = p11_peak_avMice_pooledAll_omit
    o_top_sd1a = p11_peak_sdMice_pooledAll_omit
    # V1
    f_top2a = p22_peak_avMice_pooledAll_flash
    f_top_sd2a = p22_peak_sdMice_pooledAll_flash
    o_top2a = p22_peak_avMice_pooledAll_omit
    o_top_sd2a = p22_peak_sdMice_pooledAll_omit


    plot_superimposed_pooled_cc_ccShfl_p(time_trace, top0a, top_sd0a, top1a, top_sd1a, top2a, top_sd2a, \
        f_top0a, f_top_sd0a, o_top0a, o_top_sd0a, f_top1a, f_top_sd1a, o_top1a, o_top_sd1a, f_top2a, f_top_sd2a, o_top2a, o_top_sd2a, \
        fgn, areas, yl, title, xlim_frs, mouseid=np.nan, dosavefig=dosavefig)

