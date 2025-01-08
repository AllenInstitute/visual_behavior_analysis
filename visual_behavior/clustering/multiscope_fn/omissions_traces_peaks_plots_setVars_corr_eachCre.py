#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vars are set in "omissions_traces_peaks_plots_setVars_corr.py"

Here we set vars to make summary plots for mice in each cre line (for individual mouse, also summary across mice)

This script calls  "omissions_traces_peaks_plots_setVars_corr_sumMice.py" to make summary plots for mice.
It may also run omissions_traces_peaks_plots_setVars_corr_eachMouse.py to make plots for each mouse, if do_single_mouse_plots=1 (set in omissions_traces_peaks_plots_setVars_corr.py)

Note that the plotting functions are defined in "omissions_traces_peaks_plots_setVars_corr.py"

Figures are saved in folder 'corr_omit_flash' (inside OneDrive/Analysis/)

I created a notes file ("omissions_traces_peaks_plots_setVars_corr_sumMice_summaryCodeNotes.py") that explains what variables give us the average cc traces and cc quantifications. The file is located in the same folder as the scripts.


Created on Mon Sep 23 15:58:29 2019
@author: farzaneh
"""

'''
NOTE: the reason p12_peak_sessAv_allMice_thisCre_flash might be negative is that it shows the difference in fraction significant neurons during stimulus window vs. baseline

'''

# vars below apply to average-mouse plots
plot_cc_traces = 0
plot_cc_peaks = 0
plot_cc_peaks_heatmaps = 1
plot_cc_pooled = 0
plot_cc_pooled_superimposed = 0


#%% 
###############################################################################
######## Set vars to summarize data across mice of the same cre line ##########
###############################################################################
###############################################################################

#%% 

cre_all = corr_trace_peak_allMice['cre'].values # len_mice_withData
cre_lines = np.unique(cre_all)

p12f_allCre = [] # p value of ttest between actual and shuffled peak values
p11f_allCre = []
p22f_allCre = []
p12o_allCre = []
p11o_allCre = []
p22o_allCre = []

# loop through each cre line
for icre in range(len(cre_lines)): # icre = 0
    
    cre = cre_lines[icre]    
    thisCre_numMice = sum(cre_all==cre) # number of mice that are of line cre
#     b = np.vstack(cc12_peak_all_omit.iloc[cre_all==cre])

    
    #%% Get the cc and p vars for all mice of the same cre line
    
    mouse_id_allMice_thisCre = np.full((thisCre_numMice), 0) 
    depth_sessAv_allMice_thisCre = np.full((thisCre_numMice, num_planes), np.nan) 
    
    ############ traces ############
    cc12_sessAv_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan) 
    cc11_sessAv_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)
    cc22_sessAv_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)
    
    p12_sessAv_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)    
    p11_sessAv_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)    
    p22_sessAv_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)
    
    cc12_sessAv_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan) 
    cc11_sessAv_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)
    cc22_sessAv_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)
    
    p12_sessAv_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)    
    p11_sessAv_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)    
    p22_sessAv_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)

    #### sd
    cc12_sessSd_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan) 
    cc11_sessSd_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)
    cc22_sessSd_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)
    
    p12_sessSd_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)    
    p11_sessSd_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)    
    p22_sessSd_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)
    
    cc12_sessSd_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan) 
    cc11_sessSd_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)
    cc22_sessSd_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)
    
    p12_sessSd_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)    
    p11_sessSd_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)    
    p22_sessSd_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth, num_depth), np.nan)


    # pooled
    cc12_sessAv_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan) 
    cc11_sessAv_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan)
    cc22_sessAv_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan)
    
    p12_sessAv_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan)    
    p11_sessAv_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan)    
    p22_sessAv_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan)
    
    p12_sessAv_pooled_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth), np.nan)    
    p11_sessAv_pooled_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth), np.nan)    
    p22_sessAv_pooled_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth), np.nan)

    #### pooled, sd
    cc12_sessSd_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan) 
    cc11_sessSd_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan)
    cc22_sessSd_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan)
    
    p12_sessSd_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan)    
    p11_sessSd_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan)    
    p22_sessSd_pooled_allMice_thisCre = np.full((thisCre_numMice, num_frs, num_depth), np.nan)
    
    p12_sessSd_pooled_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth), np.nan)    
    p11_sessSd_pooled_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth), np.nan)    
    p22_sessSd_pooled_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs, num_depth), np.nan)


    # pooledAll
    cc12_sessAv_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan) 
    cc11_sessAv_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan)
    cc22_sessAv_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan)
    
    p12_sessAv_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan)    
    p11_sessAv_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan)    
    p22_sessAv_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan)
        
    p12_sessAv_pooledAll_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs), np.nan)    
    p11_sessAv_pooledAll_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs), np.nan)    
    p22_sessAv_pooledAll_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs), np.nan)

    #### pooledAll, sd
    cc12_sessSd_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan) 
    cc11_sessSd_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan)
    cc22_sessSd_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan)
    
    p12_sessSd_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan)    
    p11_sessSd_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan)    
    p22_sessSd_pooledAll_allMice_thisCre = np.full((thisCre_numMice, num_frs), np.nan)
    
    p12_sessSd_pooledAll_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs), np.nan)    
    p11_sessSd_pooledAll_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs), np.nan)    
    p22_sessSd_pooledAll_allMice_thisCre_shfl = np.full((thisCre_numMice, num_frs), np.nan)


    ############ cc quantification ############
    cc12_peak_sessAv_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc12_peak_sessAv_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc11_peak_sessAv_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc11_peak_sessAv_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc22_peak_sessAv_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc22_peak_sessAv_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    
    p12_peak_sessAv_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p12_peak_sessAv_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)    
    p11_peak_sessAv_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p11_peak_sessAv_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)    
    p22_peak_sessAv_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p22_peak_sessAv_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    
    cc12_peak_sessAv_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc12_peak_sessAv_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc11_peak_sessAv_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc11_peak_sessAv_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc22_peak_sessAv_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc22_peak_sessAv_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    
    p12_peak_sessAv_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p12_peak_sessAv_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)    
    p11_peak_sessAv_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p11_peak_sessAv_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)    
    p22_peak_sessAv_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p22_peak_sessAv_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)

    #### sd
    cc12_peak_sessSd_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc12_peak_sessSd_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc11_peak_sessSd_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc11_peak_sessSd_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc22_peak_sessSd_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc22_peak_sessSd_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    
    p12_peak_sessSd_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p12_peak_sessSd_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)    
    p11_peak_sessSd_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p11_peak_sessSd_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)    
    p22_peak_sessSd_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p22_peak_sessSd_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    
    cc12_peak_sessSd_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc12_peak_sessSd_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc11_peak_sessSd_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc11_peak_sessSd_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc22_peak_sessSd_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    cc22_peak_sessSd_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    
    p12_peak_sessSd_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p12_peak_sessSd_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)    
    p11_peak_sessSd_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p11_peak_sessSd_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)    
    p22_peak_sessSd_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    p22_peak_sessSd_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth, num_depth), np.nan)
    
    
    # pooled
    cc12_peak_sessAv_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    cc12_peak_sessAv_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)
    cc11_peak_sessAv_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    cc11_peak_sessAv_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)
    cc22_peak_sessAv_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    cc22_peak_sessAv_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)
    
    p12_peak_sessAv_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    p12_peak_sessAv_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)    
    p11_peak_sessAv_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    p11_peak_sessAv_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)    
    p22_peak_sessAv_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    p22_peak_sessAv_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)
        
    p12_peak_sessAv_pooled_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth), np.nan)
    p12_peak_sessAv_pooled_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth), np.nan)    
    p11_peak_sessAv_pooled_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth), np.nan)
    p11_peak_sessAv_pooled_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth), np.nan)    
    p22_peak_sessAv_pooled_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth), np.nan)
    p22_peak_sessAv_pooled_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth), np.nan)

    #### sd
    cc12_peak_sessSd_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    cc12_peak_sessSd_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)
    cc11_peak_sessSd_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    cc11_peak_sessSd_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)
    cc22_peak_sessSd_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    cc22_peak_sessSd_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)
    
    p12_peak_sessSd_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    p12_peak_sessSd_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)    
    p11_peak_sessSd_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    p11_peak_sessSd_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)    
    p22_peak_sessSd_pooled_allMice_thisCre_omit = np.full((thisCre_numMice, num_depth), np.nan)
    p22_peak_sessSd_pooled_allMice_thisCre_flash = np.full((thisCre_numMice, num_depth), np.nan)
        
    p12_peak_sessSd_pooled_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth), np.nan)
    p12_peak_sessSd_pooled_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth), np.nan)    
    p11_peak_sessSd_pooled_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth), np.nan)
    p11_peak_sessSd_pooled_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth), np.nan)    
    p22_peak_sessSd_pooled_allMice_thisCre_omit_shfl = np.full((thisCre_numMice, num_depth), np.nan)
    p22_peak_sessSd_pooled_allMice_thisCre_flash_shfl = np.full((thisCre_numMice, num_depth), np.nan)
        
    
    # pooledAll
    cc12_peak_sessAv_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    cc12_peak_sessAv_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)
    cc11_peak_sessAv_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    cc11_peak_sessAv_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)
    cc22_peak_sessAv_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    cc22_peak_sessAv_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)
    
    p12_peak_sessAv_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    p12_peak_sessAv_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)    
    p11_peak_sessAv_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    p11_peak_sessAv_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)    
    p22_peak_sessAv_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    p22_peak_sessAv_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)
        
    p12_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl = np.full((thisCre_numMice), np.nan)
    p12_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl = np.full((thisCre_numMice), np.nan)    
    p11_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl = np.full((thisCre_numMice), np.nan)
    p11_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl = np.full((thisCre_numMice), np.nan)    
    p22_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl = np.full((thisCre_numMice), np.nan)
    p22_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl = np.full((thisCre_numMice), np.nan)

    #### sd
    cc12_peak_sessSd_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    cc12_peak_sessSd_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)
    cc11_peak_sessSd_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    cc11_peak_sessSd_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)
    cc22_peak_sessSd_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    cc22_peak_sessSd_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)
    
    p12_peak_sessSd_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    p12_peak_sessSd_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)    
    p11_peak_sessSd_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    p11_peak_sessSd_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)    
    p22_peak_sessSd_pooledAll_allMice_thisCre_omit = np.full((thisCre_numMice), np.nan)
    p22_peak_sessSd_pooledAll_allMice_thisCre_flash = np.full((thisCre_numMice), np.nan)
        
    p12_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl = np.full((thisCre_numMice), np.nan)
    p12_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl = np.full((thisCre_numMice), np.nan)    
    p11_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl = np.full((thisCre_numMice), np.nan)
    p11_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl = np.full((thisCre_numMice), np.nan)    
    p22_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl = np.full((thisCre_numMice), np.nan)
    p22_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl = np.full((thisCre_numMice), np.nan)
    
    
    ### loop through mice that belong to cre line "cre"
    session_labs_allMice_thisCre = []     
    icn = -1
    for im in np.argwhere(cre_all==cre).flatten(): # im = np.argwhere(cre_all==cre).flatten()[0] # find the index in corr_trace_peak_allMice for each mouse in a given cre line 
        icn = icn+1
        
        mouse_id_allMice_thisCre[icn] = corr_trace_peak_allMice['mouse_id'].iloc[im]        
        session_labs_allMice_thisCre.append(corr_trace_peak_allMice['session_labs'].iloc[im])
        
        depth_sessAv_allMice_thisCre[icn] = corr_trace_peak_allMice['depths'].iloc[im]        
        
        ############ traces
        cc12_sessAv_allMice_thisCre[icn] = corr_trace_peak_allMice['cc12_sessAv_44'].iloc[im]        
        cc11_sessAv_allMice_thisCre[icn] = corr_trace_peak_allMice['cc11_sessAv_44'].iloc[im]
        cc22_sessAv_allMice_thisCre[icn] = corr_trace_peak_allMice['cc22_sessAv_44'].iloc[im]

        p12_sessAv_allMice_thisCre[icn] = corr_trace_peak_allMice['p12_sessAv_44'].iloc[im]        
        p11_sessAv_allMice_thisCre[icn] = corr_trace_peak_allMice['p11_sessAv_44'].iloc[im]
        p22_sessAv_allMice_thisCre[icn] = corr_trace_peak_allMice['p22_sessAv_44'].iloc[im]

        cc12_sessAv_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['cc12_sessAv_44_shfl'].iloc[im]        
        cc11_sessAv_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['cc11_sessAv_44_shfl'].iloc[im]
        cc22_sessAv_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['cc22_sessAv_44_shfl'].iloc[im]

        p12_sessAv_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p12_sessAv_44_shfl'].iloc[im]        
        p11_sessAv_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p11_sessAv_44_shfl'].iloc[im]
        p22_sessAv_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p22_sessAv_44_shfl'].iloc[im]

        # pooled
        cc12_sessAv_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['cc12_pooled_sessAv'].iloc[im]        
        cc11_sessAv_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['cc11_pooled_sessAv'].iloc[im]
        cc22_sessAv_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['cc22_pooled_sessAv'].iloc[im]

        p12_sessAv_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['p12_pooled_sessAv'].iloc[im]        
        p11_sessAv_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['p11_pooled_sessAv'].iloc[im]
        p22_sessAv_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['p22_pooled_sessAv'].iloc[im]

        p12_sessAv_pooled_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p12_pooled_sessAv_shfl'].iloc[im]        
        p11_sessAv_pooled_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p11_pooled_sessAv_shfl'].iloc[im]
        p22_sessAv_pooled_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p22_pooled_sessAv_shfl'].iloc[im]
        
        # pooledAll
        cc12_sessAv_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['cc12_pooledAll_sessAv'].iloc[im]        
        cc11_sessAv_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['cc11_pooledAll_sessAv'].iloc[im]
        cc22_sessAv_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['cc22_pooledAll_sessAv'].iloc[im]

        p12_sessAv_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['p12_pooledAll_sessAv'].iloc[im]        
        p11_sessAv_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['p11_pooledAll_sessAv'].iloc[im]
        p22_sessAv_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['p22_pooledAll_sessAv'].iloc[im]

        p12_sessAv_pooledAll_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p12_pooledAll_sessAv_shfl'].iloc[im]        
        p11_sessAv_pooledAll_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p11_pooledAll_sessAv_shfl'].iloc[im]
        p22_sessAv_pooledAll_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p22_pooledAll_sessAv_shfl'].iloc[im]
        
        
        
        #### sd
        cc12_sessSd_allMice_thisCre[icn] = corr_trace_peak_allMice['cc12_sessSd_44'].iloc[im]        
        cc11_sessSd_allMice_thisCre[icn] = corr_trace_peak_allMice['cc11_sessSd_44'].iloc[im]
        cc22_sessSd_allMice_thisCre[icn] = corr_trace_peak_allMice['cc22_sessSd_44'].iloc[im]

        p12_sessSd_allMice_thisCre[icn] = corr_trace_peak_allMice['p12_sessSd_44'].iloc[im]        
        p11_sessSd_allMice_thisCre[icn] = corr_trace_peak_allMice['p11_sessSd_44'].iloc[im]
        p22_sessSd_allMice_thisCre[icn] = corr_trace_peak_allMice['p22_sessSd_44'].iloc[im]

        cc12_sessSd_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['cc12_sessSd_44_shfl'].iloc[im]        
        cc11_sessSd_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['cc11_sessSd_44_shfl'].iloc[im]
        cc22_sessSd_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['cc22_sessSd_44_shfl'].iloc[im]

        p12_sessSd_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p12_sessSd_44_shfl'].iloc[im]        
        p11_sessSd_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p11_sessSd_44_shfl'].iloc[im]
        p22_sessSd_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p22_sessSd_44_shfl'].iloc[im]

        # pooled
        cc12_sessSd_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['cc12_pooled_sessSd'].iloc[im]        
        cc11_sessSd_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['cc11_pooled_sessSd'].iloc[im]
        cc22_sessSd_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['cc22_pooled_sessSd'].iloc[im]

        p12_sessSd_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['p12_pooled_sessSd'].iloc[im]        
        p11_sessSd_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['p11_pooled_sessSd'].iloc[im]
        p22_sessSd_pooled_allMice_thisCre[icn] = corr_trace_peak_allMice['p22_pooled_sessSd'].iloc[im]

        p12_sessSd_pooled_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p12_pooled_sessSd_shfl'].iloc[im]        
        p11_sessSd_pooled_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p11_pooled_sessSd_shfl'].iloc[im]
        p22_sessSd_pooled_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p22_pooled_sessSd_shfl'].iloc[im]
        
        # pooledAll
        cc12_sessSd_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['cc12_pooledAll_sessSd'].iloc[im]        
        cc11_sessSd_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['cc11_pooledAll_sessSd'].iloc[im]
        cc22_sessSd_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['cc22_pooledAll_sessSd'].iloc[im]

        p12_sessSd_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['p12_pooledAll_sessSd'].iloc[im]        
        p11_sessSd_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['p11_pooledAll_sessSd'].iloc[im]
        p22_sessSd_pooledAll_allMice_thisCre[icn] = corr_trace_peak_allMice['p22_pooledAll_sessSd'].iloc[im]

        p12_sessSd_pooledAll_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p12_pooledAll_sessSd_shfl'].iloc[im]        
        p11_sessSd_pooledAll_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p11_pooledAll_sessSd_shfl'].iloc[im]
        p22_sessSd_pooledAll_allMice_thisCre_shfl[icn] = corr_trace_peak_allMice['p22_pooledAll_sessSd_shfl'].iloc[im]
        
        
        
        ############ cc magnitude quantification
        cc12_peak_sessAv_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc12_peak_amp_omit_sessAv'].iloc[im]
        cc12_peak_sessAv_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc12_peak_amp_flash_sessAv'].iloc[im]
        cc11_peak_sessAv_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc11_peak_amp_omit_sessAv'].iloc[im]
        cc11_peak_sessAv_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc11_peak_amp_flash_sessAv'].iloc[im]
        cc22_peak_sessAv_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc22_peak_amp_omit_sessAv'].iloc[im]
        cc22_peak_sessAv_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc22_peak_amp_flash_sessAv'].iloc[im]
        
        p12_peak_sessAv_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessAv'].iloc[im]
        p12_peak_sessAv_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessAv'].iloc[im]        
        p11_peak_sessAv_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessAv'].iloc[im]
        p11_peak_sessAv_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessAv'].iloc[im]
        p22_peak_sessAv_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessAv'].iloc[im]
        p22_peak_sessAv_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessAv'].iloc[im]
        
        
        cc12_peak_sessAv_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['cc12_peak_amp_omit_sessAv_shfl'].iloc[im]
        cc12_peak_sessAv_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['cc12_peak_amp_flash_sessAv_shfl'].iloc[im]
        cc11_peak_sessAv_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['cc11_peak_amp_omit_sessAv_shfl'].iloc[im]
        cc11_peak_sessAv_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['cc11_peak_amp_flash_sessAv_shfl'].iloc[im]
        cc22_peak_sessAv_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['cc22_peak_amp_omit_sessAv_shfl'].iloc[im]
        cc22_peak_sessAv_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['cc22_peak_amp_flash_sessAv_shfl'].iloc[im]
        
        p12_peak_sessAv_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessAv_shfl'].iloc[im]
        p12_peak_sessAv_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessAv_shfl'].iloc[im]        
        p11_peak_sessAv_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessAv_shfl'].iloc[im]
        p11_peak_sessAv_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessAv_shfl'].iloc[im]
        p22_peak_sessAv_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessAv_shfl'].iloc[im]
        p22_peak_sessAv_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessAv_shfl'].iloc[im]
        

        # pooled
        cc12_peak_sessAv_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc12_peak_amp_omit_sessAv_pooled'].iloc[im]
        cc12_peak_sessAv_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc12_peak_amp_flash_sessAv_pooled'].iloc[im]
        cc11_peak_sessAv_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc11_peak_amp_omit_sessAv_pooled'].iloc[im]
        cc11_peak_sessAv_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc11_peak_amp_flash_sessAv_pooled'].iloc[im]
        cc22_peak_sessAv_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc22_peak_amp_omit_sessAv_pooled'].iloc[im]
        cc22_peak_sessAv_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc22_peak_amp_flash_sessAv_pooled'].iloc[im]
        
        p12_peak_sessAv_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessAv_pooled'].iloc[im]
        p12_peak_sessAv_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessAv_pooled'].iloc[im]        
        p11_peak_sessAv_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessAv_pooled'].iloc[im]
        p11_peak_sessAv_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessAv_pooled'].iloc[im]
        p22_peak_sessAv_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessAv_pooled'].iloc[im]
        p22_peak_sessAv_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessAv_pooled'].iloc[im]        
        
        p12_peak_sessAv_pooled_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessAv_pooled_shfl'].iloc[im]
        p12_peak_sessAv_pooled_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessAv_pooled_shfl'].iloc[im]        
        p11_peak_sessAv_pooled_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessAv_pooled_shfl'].iloc[im]
        p11_peak_sessAv_pooled_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessAv_pooled_shfl'].iloc[im]
        p22_peak_sessAv_pooled_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessAv_pooled_shfl'].iloc[im]
        p22_peak_sessAv_pooled_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessAv_pooled_shfl'].iloc[im]

        # pooledAll
        cc12_peak_sessAv_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc12_peak_amp_omit_sessAv_pooledAll'].iloc[im]
        cc12_peak_sessAv_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc12_peak_amp_flash_sessAv_pooledAll'].iloc[im]
        cc11_peak_sessAv_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc11_peak_amp_omit_sessAv_pooledAll'].iloc[im]
        cc11_peak_sessAv_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc11_peak_amp_flash_sessAv_pooledAll'].iloc[im]
        cc22_peak_sessAv_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc22_peak_amp_omit_sessAv_pooledAll'].iloc[im]
        cc22_peak_sessAv_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc22_peak_amp_flash_sessAv_pooledAll'].iloc[im]
        
        p12_peak_sessAv_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessAv_pooledAll'].iloc[im]
        p12_peak_sessAv_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessAv_pooledAll'].iloc[im]        
        p11_peak_sessAv_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessAv_pooledAll'].iloc[im]
        p11_peak_sessAv_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessAv_pooledAll'].iloc[im]
        p22_peak_sessAv_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessAv_pooledAll'].iloc[im]
        p22_peak_sessAv_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessAv_pooledAll'].iloc[im]        
        
        p12_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessAv_pooledAll_shfl'].iloc[im]
        p12_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessAv_pooledAll_shfl'].iloc[im]        
        p11_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessAv_pooledAll_shfl'].iloc[im]
        p11_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessAv_pooledAll_shfl'].iloc[im]
        p22_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessAv_pooledAll_shfl'].iloc[im]
        p22_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessAv_pooledAll_shfl'].iloc[im]
                
        
        #### sd
        cc12_peak_sessSd_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc12_peak_amp_omit_sessSd'].iloc[im]
        cc12_peak_sessSd_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc12_peak_amp_flash_sessSd'].iloc[im]
        cc11_peak_sessSd_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc11_peak_amp_omit_sessSd'].iloc[im]
        cc11_peak_sessSd_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc11_peak_amp_flash_sessSd'].iloc[im]
        cc22_peak_sessSd_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc22_peak_amp_omit_sessSd'].iloc[im]
        cc22_peak_sessSd_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc22_peak_amp_flash_sessSd'].iloc[im]
        
        p12_peak_sessSd_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessSd'].iloc[im]
        p12_peak_sessSd_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessSd'].iloc[im]        
        p11_peak_sessSd_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessSd'].iloc[im]
        p11_peak_sessSd_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessSd'].iloc[im]
        p22_peak_sessSd_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessSd'].iloc[im]
        p22_peak_sessSd_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessSd'].iloc[im]
        
        
        cc12_peak_sessSd_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['cc12_peak_amp_omit_sessSd_shfl'].iloc[im]
        cc12_peak_sessSd_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['cc12_peak_amp_flash_sessSd_shfl'].iloc[im]
        cc11_peak_sessSd_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['cc11_peak_amp_omit_sessSd_shfl'].iloc[im]
        cc11_peak_sessSd_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['cc11_peak_amp_flash_sessSd_shfl'].iloc[im]
        cc22_peak_sessSd_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['cc22_peak_amp_omit_sessSd_shfl'].iloc[im]
        cc22_peak_sessSd_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['cc22_peak_amp_flash_sessSd_shfl'].iloc[im]
        
        p12_peak_sessSd_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessSd_shfl'].iloc[im]
        p12_peak_sessSd_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessSd_shfl'].iloc[im]        
        p11_peak_sessSd_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessSd_shfl'].iloc[im]
        p11_peak_sessSd_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessSd_shfl'].iloc[im]
        p22_peak_sessSd_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessSd_shfl'].iloc[im]
        p22_peak_sessSd_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessSd_shfl'].iloc[im]

        # pooled
        cc12_peak_sessSd_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc12_peak_amp_omit_sessSd_pooled'].iloc[im]
        cc12_peak_sessSd_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc12_peak_amp_flash_sessSd_pooled'].iloc[im]
        cc11_peak_sessSd_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc11_peak_amp_omit_sessSd_pooled'].iloc[im]
        cc11_peak_sessSd_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc11_peak_amp_flash_sessSd_pooled'].iloc[im]
        cc22_peak_sessSd_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc22_peak_amp_omit_sessSd_pooled'].iloc[im]
        cc22_peak_sessSd_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc22_peak_amp_flash_sessSd_pooled'].iloc[im]
        
        p12_peak_sessSd_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessSd_pooled'].iloc[im]
        p12_peak_sessSd_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessSd_pooled'].iloc[im]        
        p11_peak_sessSd_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessSd_pooled'].iloc[im]
        p11_peak_sessSd_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessSd_pooled'].iloc[im]
        p22_peak_sessSd_pooled_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessSd_pooled'].iloc[im]
        p22_peak_sessSd_pooled_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessSd_pooled'].iloc[im]        
        
        p12_peak_sessSd_pooled_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessSd_pooled_shfl'].iloc[im]
        p12_peak_sessSd_pooled_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessSd_pooled_shfl'].iloc[im]        
        p11_peak_sessSd_pooled_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessSd_pooled_shfl'].iloc[im]
        p11_peak_sessSd_pooled_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessSd_pooled_shfl'].iloc[im]
        p22_peak_sessSd_pooled_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessSd_pooled_shfl'].iloc[im]
        p22_peak_sessSd_pooled_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessSd_pooled_shfl'].iloc[im]

        # pooledAll
        cc12_peak_sessSd_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc12_peak_amp_omit_sessSd_pooledAll'].iloc[im]
        cc12_peak_sessSd_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc12_peak_amp_flash_sessSd_pooledAll'].iloc[im]
        cc11_peak_sessSd_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc11_peak_amp_omit_sessSd_pooledAll'].iloc[im]
        cc11_peak_sessSd_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc11_peak_amp_flash_sessSd_pooledAll'].iloc[im]
        cc22_peak_sessSd_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc22_peak_amp_omit_sessSd_pooledAll'].iloc[im]
        cc22_peak_sessSd_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['cc22_peak_amp_flash_sessSd_pooledAll'].iloc[im]
        
        p12_peak_sessSd_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessSd_pooledAll'].iloc[im]
        p12_peak_sessSd_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessSd_pooledAll'].iloc[im]        
        p11_peak_sessSd_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessSd_pooledAll'].iloc[im]
        p11_peak_sessSd_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessSd_pooledAll'].iloc[im]
        p22_peak_sessSd_pooledAll_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessSd_pooledAll'].iloc[im]
        p22_peak_sessSd_pooledAll_allMice_thisCre_flash[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessSd_pooledAll'].iloc[im]        
        
        p12_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_omit_sessSd_pooledAll_shfl'].iloc[im]
        p12_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p12_peak_amp_flash_sessSd_pooledAll_shfl'].iloc[im]        
        p11_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_omit_sessSd_pooledAll_shfl'].iloc[im]
        p11_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p11_peak_amp_flash_sessSd_pooledAll_shfl'].iloc[im]
        p22_peak_sessSd_pooledAll_allMice_thisCre_omit_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_omit_sessSd_pooledAll_shfl'].iloc[im]
        p22_peak_sessSd_pooledAll_allMice_thisCre_flash_shfl[icn] = corr_trace_peak_allMice['p22_peak_amp_flash_sessSd_pooledAll_shfl'].iloc[im]
        

        
        #%%
        ######################## Plots of session-averaged data, for each mouse ########################        
        # one figure for: 
        # corr of all layer combinations
        #
        # another figure for the following:
        # corr of 1 layer with all other neurons in other layers (pooled)
        # corr of all neurons in all layers with each other (pooledAll)
        
        if do_single_mouse_plots:
            exec(open("./omissions_traces_peaks_plots_setVars_corr_eachMouse.py").read())


            
    ###########################################################################    
    ###########################################################################        
    #%% Two-sided ttest between actual and shuffled correlations
    ###########################################################################    
    ###########################################################################    

    # indeces: [LM, V1]; eg p[:,0] is p between each depth of LM, and depth 0 of V1.
    _, p12f = st.ttest_ind(cc12_peak_sessAv_allMice_thisCre_flash, cc12_peak_sessAv_allMice_thisCre_flash_shfl, nan_policy='omit')
    _, p11f = st.ttest_ind(cc11_peak_sessAv_allMice_thisCre_flash, cc11_peak_sessAv_allMice_thisCre_flash_shfl, nan_policy='omit')
    _, p22f = st.ttest_ind(cc22_peak_sessAv_allMice_thisCre_flash, cc22_peak_sessAv_allMice_thisCre_flash_shfl, nan_policy='omit')

    _, p12o = st.ttest_ind(cc12_peak_sessAv_allMice_thisCre_omit, cc12_peak_sessAv_allMice_thisCre_omit_shfl, nan_policy='omit')
    _, p11o = st.ttest_ind(cc11_peak_sessAv_allMice_thisCre_omit, cc11_peak_sessAv_allMice_thisCre_omit_shfl, nan_policy='omit')
    _, p22o = st.ttest_ind(cc22_peak_sessAv_allMice_thisCre_omit, cc22_peak_sessAv_allMice_thisCre_omit_shfl, nan_policy='omit')
    
    p12f_allCre.append(p12f.data) 
    p11f_allCre.append(p11f.data)
    p22f_allCre.append(p22f.data)
    
    p12o_allCre.append(p12o.data) 
    p11o_allCre.append(p11o.data)
    p22o_allCre.append(p22o.data)
    
    
    
    #%%
    ###########################################################################    
    ###########################################################################    
    #%% Average and sd across mice (for each cre line)
    ###########################################################################
    ###########################################################################    
    
    
    #%%    
    
    depth_sessAv_avMice = np.mean(depth_sessAv_allMice_thisCre, axis=0)
 
    numValidMice12 = np.sum(~np.isnan(cc12_peak_sessAv_allMice_thisCre_omit), axis=0)
    numValidMice11 = np.sum(~np.isnan(cc11_peak_sessAv_allMice_thisCre_omit), axis=0)
    numValidMice22 = np.sum(~np.isnan(cc22_peak_sessAv_allMice_thisCre_omit), axis=0)
    
    numValidMice12_pooled = np.sum(~np.isnan(cc12_peak_sessAv_pooled_allMice_thisCre_omit), axis=0)
    numValidMice11_pooled = np.sum(~np.isnan(cc11_peak_sessAv_pooled_allMice_thisCre_omit), axis=0)
    numValidMice22_pooled = np.sum(~np.isnan(cc22_peak_sessAv_pooled_allMice_thisCre_omit), axis=0)

    numValidMice12_pooledAll = np.sum(~np.isnan(cc12_peak_sessAv_pooledAll_allMice_thisCre_omit), axis=0)
    numValidMice11_pooledAll = np.sum(~np.isnan(cc11_peak_sessAv_pooledAll_allMice_thisCre_omit), axis=0)
    numValidMice22_pooledAll = np.sum(~np.isnan(cc22_peak_sessAv_pooledAll_allMice_thisCre_omit), axis=0)
    
    
    ##### traces #####
    cc12_sessAv_avMice = np.nanmean(cc12_sessAv_allMice_thisCre, axis=0) # 80x4x4    
    cc11_sessAv_avMice = np.nanmean(cc11_sessAv_allMice_thisCre, axis=0) # 80x4x4
    cc22_sessAv_avMice = np.nanmean(cc22_sessAv_allMice_thisCre, axis=0) # 80x4x4

    p12_sessAv_avMice = np.nanmean(p12_sessAv_allMice_thisCre, axis=0)
    p11_sessAv_avMice = np.nanmean(p11_sessAv_allMice_thisCre, axis=0)    
    p22_sessAv_avMice = np.nanmean(p22_sessAv_allMice_thisCre, axis=0)
    
    cc12_sessAv_avMice_shfl = np.nanmean(cc12_sessAv_allMice_thisCre_shfl, axis=0) # 80x4x4    
    cc11_sessAv_avMice_shfl = np.nanmean(cc11_sessAv_allMice_thisCre_shfl, axis=0) # 80x4x4
    cc22_sessAv_avMice_shfl = np.nanmean(cc22_sessAv_allMice_thisCre_shfl, axis=0) # 80x4x4

    p12_sessAv_avMice_shfl = np.nanmean(p12_sessAv_allMice_thisCre_shfl, axis=0)
    p11_sessAv_avMice_shfl = np.nanmean(p11_sessAv_allMice_thisCre_shfl, axis=0)    
    p22_sessAv_avMice_shfl = np.nanmean(p22_sessAv_allMice_thisCre_shfl, axis=0)

    # sd
    cc12_sessAv_sdMice = np.nanstd(cc12_sessAv_allMice_thisCre, axis=0) / np.sqrt(numValidMice12) # 80x4x4    
    cc11_sessAv_sdMice = np.nanstd(cc11_sessAv_allMice_thisCre, axis=0) / np.sqrt(numValidMice11) # 80x4x4
    cc22_sessAv_sdMice = np.nanstd(cc22_sessAv_allMice_thisCre, axis=0) / np.sqrt(numValidMice22) # 80x4x4

    p12_sessAv_sdMice = np.nanstd(p12_sessAv_allMice_thisCre, axis=0) / np.sqrt(numValidMice12)
    p11_sessAv_sdMice = np.nanstd(p11_sessAv_allMice_thisCre, axis=0) / np.sqrt(numValidMice11)    
    p22_sessAv_sdMice = np.nanstd(p22_sessAv_allMice_thisCre, axis=0) / np.sqrt(numValidMice22)
    
    cc12_sessAv_sdMice_shfl = np.nanstd(cc12_sessAv_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice12) # 80x4x4    
    cc11_sessAv_sdMice_shfl = np.nanstd(cc11_sessAv_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice11) # 80x4x4
    cc22_sessAv_sdMice_shfl = np.nanstd(cc22_sessAv_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice22) # 80x4x4

    p12_sessAv_sdMice_shfl = np.nanstd(p12_sessAv_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice12)
    p11_sessAv_sdMice_shfl = np.nanstd(p11_sessAv_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice11)    
    p22_sessAv_sdMice_shfl = np.nanstd(p22_sessAv_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice22)


    ### pooled
    cc12_sessAv_pooled_avMice = np.nanmean(cc12_sessAv_pooled_allMice_thisCre, axis=0) # 80x4    
    cc11_sessAv_pooled_avMice = np.nanmean(cc11_sessAv_pooled_allMice_thisCre, axis=0) # 80x4
    cc22_sessAv_pooled_avMice = np.nanmean(cc22_sessAv_pooled_allMice_thisCre, axis=0) # 80x4

    p12_sessAv_pooled_avMice = np.nanmean(p12_sessAv_pooled_allMice_thisCre, axis=0)
    p11_sessAv_pooled_avMice = np.nanmean(p11_sessAv_pooled_allMice_thisCre, axis=0)    
    p22_sessAv_pooled_avMice = np.nanmean(p22_sessAv_pooled_allMice_thisCre, axis=0)
    
#    cc12_sessAv_pooled_avMice_shfl = np.nanmean(cc12_sessAv_pooled_allMice_thisCre_shfl, axis=0) # 80x4    
#    cc11_sessAv_pooled_avMice_shfl = np.nanmean(cc11_sessAv_pooled_allMice_thisCre_shfl, axis=0) # 80x4
#    cc22_sessAv_pooled_avMice_shfl = np.nanmean(cc22_sessAv_pooled_allMice_thisCre_shfl, axis=0) # 80x4

    p12_sessAv_pooled_avMice_shfl = np.nanmean(p12_sessAv_pooled_allMice_thisCre_shfl, axis=0)
    p11_sessAv_pooled_avMice_shfl = np.nanmean(p11_sessAv_pooled_allMice_thisCre_shfl, axis=0)    
    p22_sessAv_pooled_avMice_shfl = np.nanmean(p22_sessAv_pooled_allMice_thisCre_shfl, axis=0)

    # sd
    cc12_sessAv_pooled_sdMice = np.nanstd(cc12_sessAv_pooled_allMice_thisCre, axis=0) / np.sqrt(numValidMice12_pooled) # 80x4    
    cc11_sessAv_pooled_sdMice = np.nanstd(cc11_sessAv_pooled_allMice_thisCre, axis=0) / np.sqrt(numValidMice11_pooled) # 80x4
    cc22_sessAv_pooled_sdMice = np.nanstd(cc22_sessAv_pooled_allMice_thisCre, axis=0) / np.sqrt(numValidMice22_pooled) # 80x4

    p12_sessAv_pooled_sdMice = np.nanstd(p12_sessAv_pooled_allMice_thisCre, axis=0) / np.sqrt(numValidMice12_pooled)
    p11_sessAv_pooled_sdMice = np.nanstd(p11_sessAv_pooled_allMice_thisCre, axis=0) / np.sqrt(numValidMice11_pooled)    
    p22_sessAv_pooled_sdMice = np.nanstd(p22_sessAv_pooled_allMice_thisCre, axis=0) / np.sqrt(numValidMice22_pooled)
    
#    cc12_sessAv_pooled_sdMice_shfl = np.nanstd(cc12_sessAv_pooled_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice12_pooled) # 80x4
#    cc11_sessAv_pooled_sdMice_shfl = np.nanstd(cc11_sessAv_pooled_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice11_pooled) # 80x4
#    cc22_sessAv_pooled_sdMice_shfl = np.nanstd(cc22_sessAv_pooled_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice22_pooled) # 80x4

    p12_sessAv_pooled_sdMice_shfl = np.nanstd(p12_sessAv_pooled_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice12_pooled)
    p11_sessAv_pooled_sdMice_shfl = np.nanstd(p11_sessAv_pooled_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice11_pooled)    
    p22_sessAv_pooled_sdMice_shfl = np.nanstd(p22_sessAv_pooled_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice22_pooled)


    ### pooledAll
    cc12_sessAv_pooledAll_avMice = np.nanmean(cc12_sessAv_pooledAll_allMice_thisCre, axis=0) # 80    
    cc11_sessAv_pooledAll_avMice = np.nanmean(cc11_sessAv_pooledAll_allMice_thisCre, axis=0) # 80
    cc22_sessAv_pooledAll_avMice = np.nanmean(cc22_sessAv_pooledAll_allMice_thisCre, axis=0) # 80

    p12_sessAv_pooledAll_avMice = np.nanmean(p12_sessAv_pooledAll_allMice_thisCre, axis=0)
    p11_sessAv_pooledAll_avMice = np.nanmean(p11_sessAv_pooledAll_allMice_thisCre, axis=0)    
    p22_sessAv_pooledAll_avMice = np.nanmean(p22_sessAv_pooledAll_allMice_thisCre, axis=0)
    
#    cc12_sessAv_pooledAll_avMice_shfl = np.nanmean(cc12_sessAv_pooledAll_allMice_thisCre_shfl, axis=0) # 80    
#    cc11_sessAv_pooledAll_avMice_shfl = np.nanmean(cc11_sessAv_pooledAll_allMice_thisCre_shfl, axis=0) # 80
#    cc22_sessAv_pooledAll_avMice_shfl = np.nanmean(cc22_sessAv_pooledAll_allMice_thisCre_shfl, axis=0) # 80

    p12_sessAv_pooledAll_avMice_shfl = np.nanmean(p12_sessAv_pooledAll_allMice_thisCre_shfl, axis=0)
    p11_sessAv_pooledAll_avMice_shfl = np.nanmean(p11_sessAv_pooledAll_allMice_thisCre_shfl, axis=0)    
    p22_sessAv_pooledAll_avMice_shfl = np.nanmean(p22_sessAv_pooledAll_allMice_thisCre_shfl, axis=0)

    # sd
    cc12_sessAv_pooledAll_sdMice = np.nanstd(cc12_sessAv_pooledAll_allMice_thisCre, axis=0) / np.sqrt(numValidMice12_pooledAll) # 80    
    cc11_sessAv_pooledAll_sdMice = np.nanstd(cc11_sessAv_pooledAll_allMice_thisCre, axis=0) / np.sqrt(numValidMice11_pooledAll) # 80
    cc22_sessAv_pooledAll_sdMice = np.nanstd(cc22_sessAv_pooledAll_allMice_thisCre, axis=0) / np.sqrt(numValidMice22_pooledAll) # 80

    p12_sessAv_pooledAll_sdMice = np.nanstd(p12_sessAv_pooledAll_allMice_thisCre, axis=0) / np.sqrt(numValidMice12_pooledAll)
    p11_sessAv_pooledAll_sdMice = np.nanstd(p11_sessAv_pooledAll_allMice_thisCre, axis=0) / np.sqrt(numValidMice11_pooledAll)    
    p22_sessAv_pooledAll_sdMice = np.nanstd(p22_sessAv_pooledAll_allMice_thisCre, axis=0) / np.sqrt(numValidMice22_pooledAll)
    
#    cc12_sessAv_pooledAll_sdMice_shfl = np.nanstd(cc12_sessAv_pooledAll_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice12_pooledAll) # 80
#    cc11_sessAv_pooledAll_sdMice_shfl = np.nanstd(cc11_sessAv_pooledAll_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice11_pooledAll) # 80
#    cc22_sessAv_pooledAll_sdMice_shfl = np.nanstd(cc22_sessAv_pooledAll_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice22_pooledAll) # 80

    p12_sessAv_pooledAll_sdMice_shfl = np.nanstd(p12_sessAv_pooledAll_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice12_pooledAll)
    p11_sessAv_pooledAll_sdMice_shfl = np.nanstd(p11_sessAv_pooledAll_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice11_pooledAll)    
    p22_sessAv_pooledAll_sdMice_shfl = np.nanstd(p22_sessAv_pooledAll_allMice_thisCre_shfl, axis=0) / np.sqrt(numValidMice22_pooledAll)




    ##### peak quantification #####   
    cc12_peak_avMice_omit = np.nanmean(cc12_peak_sessAv_allMice_thisCre_omit, axis=0) # 4x4 
    cc12_peak_avMice_flash = np.nanmean(cc12_peak_sessAv_allMice_thisCre_flash, axis=0) # 4x4
    cc11_peak_avMice_omit = np.nanmean(cc11_peak_sessAv_allMice_thisCre_omit, axis=0) # 4x4 
    cc11_peak_avMice_flash = np.nanmean(cc11_peak_sessAv_allMice_thisCre_flash, axis=0) # 4x4
    cc22_peak_avMice_omit = np.nanmean(cc22_peak_sessAv_allMice_thisCre_omit, axis=0) # 4x4 
    cc22_peak_avMice_flash = np.nanmean(cc22_peak_sessAv_allMice_thisCre_flash, axis=0) # 4x4

    p12_peak_avMice_omit = np.nanmean(p12_peak_sessAv_allMice_thisCre_omit, axis=0) # 4x4 
    p12_peak_avMice_flash = np.nanmean(p12_peak_sessAv_allMice_thisCre_flash, axis=0) # 4x4
    p11_peak_avMice_omit = np.nanmean(p11_peak_sessAv_allMice_thisCre_omit, axis=0) # 4x4 
    p11_peak_avMice_flash = np.nanmean(p11_peak_sessAv_allMice_thisCre_flash, axis=0) # 4x4
    p22_peak_avMice_omit = np.nanmean(p22_peak_sessAv_allMice_thisCre_omit, axis=0) # 4x4 
    p22_peak_avMice_flash = np.nanmean(p22_peak_sessAv_allMice_thisCre_flash, axis=0) # 4x4
    
    cc12_peak_avMice_omit_shfl = np.nanmean(cc12_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) # 4x4 
    cc12_peak_avMice_flash_shfl = np.nanmean(cc12_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) # 4x4
    cc11_peak_avMice_omit_shfl = np.nanmean(cc11_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) # 4x4 
    cc11_peak_avMice_flash_shfl = np.nanmean(cc11_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) # 4x4
    cc22_peak_avMice_omit_shfl = np.nanmean(cc22_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) # 4x4 
    cc22_peak_avMice_flash_shfl = np.nanmean(cc22_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) # 4x4

    p12_peak_avMice_omit_shfl = np.nanmean(p12_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) # 4x4 
    p12_peak_avMice_flash_shfl = np.nanmean(p12_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) # 4x4
    p11_peak_avMice_omit_shfl = np.nanmean(p11_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) # 4x4 
    p11_peak_avMice_flash_shfl = np.nanmean(p11_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) # 4x4
    p22_peak_avMice_omit_shfl = np.nanmean(p22_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) # 4x4 
    p22_peak_avMice_flash_shfl = np.nanmean(p22_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) # 4x4

    # sd
    cc12_peak_sdMice_omit = np.nanstd(cc12_peak_sessAv_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice12) # 4x4 
    cc12_peak_sdMice_flash = np.nanstd(cc12_peak_sessAv_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice12) # 4x4
    cc11_peak_sdMice_omit = np.nanstd(cc11_peak_sessAv_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice11) # 4x4 
    cc11_peak_sdMice_flash = np.nanstd(cc11_peak_sessAv_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice11) # 4x4
    cc22_peak_sdMice_omit = np.nanstd(cc22_peak_sessAv_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice22) # 4x4 
    cc22_peak_sdMice_flash = np.nanstd(cc22_peak_sessAv_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice22) # 4x4

    p12_peak_sdMice_omit = np.nanstd(p12_peak_sessAv_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice12) # 4x4 
    p12_peak_sdMice_flash = np.nanstd(p12_peak_sessAv_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice12) # 4x4
    p11_peak_sdMice_omit = np.nanstd(p11_peak_sessAv_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice11) # 4x4 
    p11_peak_sdMice_flash = np.nanstd(p11_peak_sessAv_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice11) # 4x4
    p22_peak_sdMice_omit = np.nanstd(p22_peak_sessAv_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice22) # 4x4 
    p22_peak_sdMice_flash = np.nanstd(p22_peak_sessAv_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice22) # 4x4
    
    cc12_peak_sdMice_omit_shfl = np.nanstd(cc12_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice12) # 4x4 
    cc12_peak_sdMice_flash_shfl = np.nanstd(cc12_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice12) # 4x4
    cc11_peak_sdMice_omit_shfl = np.nanstd(cc11_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice11) # 4x4 
    cc11_peak_sdMice_flash_shfl = np.nanstd(cc11_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice11) # 4x4
    cc22_peak_sdMice_omit_shfl = np.nanstd(cc22_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice22) # 4x4 
    cc22_peak_sdMice_flash_shfl = np.nanstd(cc22_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice22) # 4x4

    p12_peak_sdMice_omit_shfl = np.nanstd(p12_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice12) # 4x4 
    p12_peak_sdMice_flash_shfl = np.nanstd(p12_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice12) # 4x4
    p11_peak_sdMice_omit_shfl = np.nanstd(p11_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice11) # 4x4 
    p11_peak_sdMice_flash_shfl = np.nanstd(p11_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice11) # 4x4
    p22_peak_sdMice_omit_shfl = np.nanstd(p22_peak_sessAv_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice22) # 4x4 
    p22_peak_sdMice_flash_shfl = np.nanstd(p22_peak_sessAv_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice22) # 4x4
    
    
    
    #### pooled
    cc12_peak_avMice_pooled_omit = np.nanmean(cc12_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) # 4 
    cc12_peak_avMice_pooled_flash = np.nanmean(cc12_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) # 4
    cc11_peak_avMice_pooled_omit = np.nanmean(cc11_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) # 4 
    cc11_peak_avMice_pooled_flash = np.nanmean(cc11_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) # 4
    cc22_peak_avMice_pooled_omit = np.nanmean(cc22_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) # 4 
    cc22_peak_avMice_pooled_flash = np.nanmean(cc22_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) # 4

    p12_peak_avMice_pooled_omit = np.nanmean(p12_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) # 4 
    p12_peak_avMice_pooled_flash = np.nanmean(p12_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) # 4
    p11_peak_avMice_pooled_omit = np.nanmean(p11_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) # 4 
    p11_peak_avMice_pooled_flash = np.nanmean(p11_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) # 4
    p22_peak_avMice_pooled_omit = np.nanmean(p22_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) # 4 
    p22_peak_avMice_pooled_flash = np.nanmean(p22_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) # 4
    
#    cc12_peak_avMice_pooled_omit_shfl = np.nanmean(cc12_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) # 4 
#    cc12_peak_avMice_pooled_flash_shfl = np.nanmean(cc12_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) # 4
#    cc11_peak_avMice_pooled_omit_shfl = np.nanmean(cc11_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) # 4 
#    cc11_peak_avMice_pooled_flash_shfl = np.nanmean(cc11_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) # 4
#    cc22_peak_avMice_pooled_omit_shfl = np.nanmean(cc22_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) # 4 
#    cc22_peak_avMice_pooled_flash_shfl = np.nanmean(cc22_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) # 4

    p12_peak_avMice_pooled_omit_shfl = np.nanmean(p12_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) # 4 
    p12_peak_avMice_pooled_flash_shfl = np.nanmean(p12_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) # 4
    p11_peak_avMice_pooled_omit_shfl = np.nanmean(p11_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) # 4 
    p11_peak_avMice_pooled_flash_shfl = np.nanmean(p11_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) # 4
    p22_peak_avMice_pooled_omit_shfl = np.nanmean(p22_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) # 4 
    p22_peak_avMice_pooled_flash_shfl = np.nanmean(p22_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) # 4

    # sd
    cc12_peak_sdMice_pooled_omit = np.nanstd(cc12_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice12_pooled) # 4 
    cc12_peak_sdMice_pooled_flash = np.nanstd(cc12_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice12_pooled) # 4
    cc11_peak_sdMice_pooled_omit = np.nanstd(cc11_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice11_pooled) # 4 
    cc11_peak_sdMice_pooled_flash = np.nanstd(cc11_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice11_pooled) # 4
    cc22_peak_sdMice_pooled_omit = np.nanstd(cc22_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice22_pooled) # 4 
    cc22_peak_sdMice_pooled_flash = np.nanstd(cc22_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice22_pooled) # 4

    p12_peak_sdMice_pooled_omit = np.nanstd(p12_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice12_pooled) # 4 
    p12_peak_sdMice_pooled_flash = np.nanstd(p12_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice12_pooled) # 4
    p11_peak_sdMice_pooled_omit = np.nanstd(p11_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice11_pooled) # 4 
    p11_peak_sdMice_pooled_flash = np.nanstd(p11_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice11_pooled) # 4
    p22_peak_sdMice_pooled_omit = np.nanstd(p22_peak_sessAv_pooled_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice22_pooled) # 4 
    p22_peak_sdMice_pooled_flash = np.nanstd(p22_peak_sessAv_pooled_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice22_pooled) # 4
    
#    cc12_peak_sdMice_pooled_omit_shfl = np.nanstd(cc12_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice12_pooled) # 4 
#    cc12_peak_sdMice_pooled_flash_shfl = np.nanstd(cc12_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice12_pooled) # 4
#    cc11_peak_sdMice_pooled_omit_shfl = np.nanstd(cc11_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice11_pooled) # 4 
#    cc11_peak_sdMice_pooled_flash_shfl = np.nanstd(cc11_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice11_pooled) # 4
#    cc22_peak_sdMice_pooled_omit_shfl = np.nanstd(cc22_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice22_pooled) # 4 
#    cc22_peak_sdMice_pooled_flash_shfl = np.nanstd(cc22_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice22_pooled) # 4

    p12_peak_sdMice_pooled_omit_shfl = np.nanstd(p12_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice12_pooled) # 4 
    p12_peak_sdMice_pooled_flash_shfl = np.nanstd(p12_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice12_pooled) # 4
    p11_peak_sdMice_pooled_omit_shfl = np.nanstd(p11_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice11_pooled) # 4 
    p11_peak_sdMice_pooled_flash_shfl = np.nanstd(p11_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice11_pooled) # 4
    p22_peak_sdMice_pooled_omit_shfl = np.nanstd(p22_peak_sessAv_pooled_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice22_pooled) # 4 
    p22_peak_sdMice_pooled_flash_shfl = np.nanstd(p22_peak_sessAv_pooled_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice22_pooled) # 4

    
    #### pooledAll
    cc12_peak_avMice_pooledAll_omit = np.nanmean(cc12_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) # 4 
    cc12_peak_avMice_pooledAll_flash = np.nanmean(cc12_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) # 4
    cc11_peak_avMice_pooledAll_omit = np.nanmean(cc11_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) # 4 
    cc11_peak_avMice_pooledAll_flash = np.nanmean(cc11_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) # 4
    cc22_peak_avMice_pooledAll_omit = np.nanmean(cc22_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) # 4 
    cc22_peak_avMice_pooledAll_flash = np.nanmean(cc22_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) # 4

    p12_peak_avMice_pooledAll_omit = np.nanmean(p12_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) # 4 
    p12_peak_avMice_pooledAll_flash = np.nanmean(p12_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) # 4
    p11_peak_avMice_pooledAll_omit = np.nanmean(p11_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) # 4 
    p11_peak_avMice_pooledAll_flash = np.nanmean(p11_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) # 4
    p22_peak_avMice_pooledAll_omit = np.nanmean(p22_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) # 4 
    p22_peak_avMice_pooledAll_flash = np.nanmean(p22_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) # 4
    
#    cc12_peak_avMice_pooledAll_omit_shfl = np.nanmean(cc12_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) # 4 
#    cc12_peak_avMice_pooledAll_flash_shfl = np.nanmean(cc12_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) # 4
#    cc11_peak_avMice_pooledAll_omit_shfl = np.nanmean(cc11_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) # 4 
#    cc11_peak_avMice_pooledAll_flash_shfl = np.nanmean(cc11_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) # 4
#    cc22_peak_avMice_pooledAll_omit_shfl = np.nanmean(cc22_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) # 4 
#    cc22_peak_avMice_pooledAll_flash_shfl = np.nanmean(cc22_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) # 4

    p12_peak_avMice_pooledAll_omit_shfl = np.nanmean(p12_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) # 4 
    p12_peak_avMice_pooledAll_flash_shfl = np.nanmean(p12_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) # 4
    p11_peak_avMice_pooledAll_omit_shfl = np.nanmean(p11_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) # 4 
    p11_peak_avMice_pooledAll_flash_shfl = np.nanmean(p11_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) # 4
    p22_peak_avMice_pooledAll_omit_shfl = np.nanmean(p22_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) # 4 
    p22_peak_avMice_pooledAll_flash_shfl = np.nanmean(p22_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) # 4

    # sd
    cc12_peak_sdMice_pooledAll_omit = np.nanstd(cc12_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice12_pooledAll) # 4 
    cc12_peak_sdMice_pooledAll_flash = np.nanstd(cc12_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice12_pooledAll) # 4
    cc11_peak_sdMice_pooledAll_omit = np.nanstd(cc11_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice11_pooledAll) # 4 
    cc11_peak_sdMice_pooledAll_flash = np.nanstd(cc11_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice11_pooledAll) # 4
    cc22_peak_sdMice_pooledAll_omit = np.nanstd(cc22_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice22_pooledAll) # 4 
    cc22_peak_sdMice_pooledAll_flash = np.nanstd(cc22_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice22_pooledAll) # 4

    p12_peak_sdMice_pooledAll_omit = np.nanstd(p12_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice12_pooledAll) # 4 
    p12_peak_sdMice_pooledAll_flash = np.nanstd(p12_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice12_pooledAll) # 4
    p11_peak_sdMice_pooledAll_omit = np.nanstd(p11_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice11_pooledAll) # 4 
    p11_peak_sdMice_pooledAll_flash = np.nanstd(p11_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice11_pooledAll) # 4
    p22_peak_sdMice_pooledAll_omit = np.nanstd(p22_peak_sessAv_pooledAll_allMice_thisCre_omit, axis=0) / np.sqrt(numValidMice22_pooledAll) # 4 
    p22_peak_sdMice_pooledAll_flash = np.nanstd(p22_peak_sessAv_pooledAll_allMice_thisCre_flash, axis=0) / np.sqrt(numValidMice22_pooledAll) # 4
    
#    cc12_peak_sdMice_pooledAll_omit_shfl = np.nanstd(cc12_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice12_pooledAll) # 4 
#    cc12_peak_sdMice_pooledAll_flash_shfl = np.nanstd(cc12_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice12_pooledAll) # 4
#    cc11_peak_sdMice_pooledAll_omit_shfl = np.nanstd(cc11_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice11_pooledAll) # 4 
#    cc11_peak_sdMice_pooledAll_flash_shfl = np.nanstd(cc11_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice11_pooledAll) # 4
#    cc22_peak_sdMice_pooledAll_omit_shfl = np.nanstd(cc22_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice22_pooledAll) # 4 
#    cc22_peak_sdMice_pooledAll_flash_shfl = np.nanstd(cc22_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice22_pooledAll) # 4

    p12_peak_sdMice_pooledAll_omit_shfl = np.nanstd(p12_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice12_pooledAll) # 4 
    p12_peak_sdMice_pooledAll_flash_shfl = np.nanstd(p12_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice12_pooledAll) # 4
    p11_peak_sdMice_pooledAll_omit_shfl = np.nanstd(p11_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice11_pooledAll) # 4 
    p11_peak_sdMice_pooledAll_flash_shfl = np.nanstd(p11_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice11_pooledAll) # 4
    p22_peak_sdMice_pooledAll_omit_shfl = np.nanstd(p22_peak_sessAv_pooledAll_allMice_thisCre_omit_shfl, axis=0) / np.sqrt(numValidMice22_pooledAll) # 4 
    p22_peak_sdMice_pooledAll_flash_shfl = np.nanstd(p22_peak_sessAv_pooledAll_allMice_thisCre_flash_shfl, axis=0) / np.sqrt(numValidMice22_pooledAll) # 4

    '''
    # CONSIDER THIS: I dont think I need to do it!
    # here it perhaps makes sense to normalize each mouse by its max cc value, otherwise one mouse may dominate the average!

    # do scale and shift
    m = [np.max(cc12_sessAv_allMice_thisCre[i]) for i in range(len(cc12_sessAv_allMice_thisCre))]
    if sameBl_allLayerPairs:
        bl_preOmit = np.percentile(cc_sessAv[bl_index_pre_omit,:], bl_percentile, axis=0) # 16 (layer combinations of the two areas)
        bl_preFlash = np.percentile(cc_sessAv[bl_index_pre_flash,:], bl_percentile, axis=0) # 16 (layer combinations of the two areas)
    '''

    
    #%%
    ######################## Plots of mice-averaged data, for each cre line ########################
    
    exec(open("./omissions_traces_peaks_plots_setVars_corr_sumMice.py").read())
    

    
    
    
    #%%
    ####################################################################################
    ####################################################################################
    ### ANOVA on response quantifications (image and omission evoked response amplitudes)
    ####################################################################################
    ####################################################################################    
    
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
        
#     cc12_peak_sessAv_allMice_thisCre_flash  # mice x depth x depth
#     p12_peak_sessAv_allMice_thisCre_flash  # mice x depth x depth

    # 2 x 8 x sessions
#     aa_all = [pa_all[:, cre_all[0,:]==cre], paf_all[:, cre_all[0,:]==cre]] # 2 x 8 x sessions
    
    
    area_pairs = 'LM-V1', 'V1-V1', 'LM-LM'
    flts = cc12_peak_sessAv_allMice_thisCre_flash, cc22_peak_sessAv_allMice_thisCre_flash, cc11_peak_sessAv_allMice_thisCre_flash
    omts = cc12_peak_sessAv_allMice_thisCre_omit, cc22_peak_sessAv_allMice_thisCre_omit, cc11_peak_sessAv_allMice_thisCre_omit

    cols = ['cre', 'area_pair', 'depth_pair', 'value']
    a_data = pd.DataFrame([], columns = cols)    
    i_at = -1
    for iap in range(len(area_pairs)): # iap=0

        # flash: pool across depth pairs for each mouse    
        # first 4 elements: V1 depth 0 with LM depth 0-3. Next 4 elements: V1 depth 1 with LM depth 0-3; and so on
        ff = np.reshape(flts[iap], (flts[iap].shape[0], num_depth*num_depth), order = 'F')
        ff.shape # mice x 16

        # flash: pool across depth pairs for each mouse    
        oo = np.reshape(omts[iap], (omts[iap].shape[0], num_depth*num_depth), order = 'F')
        oo.shape # mice x 16


        # now concat omit and flash, also turn the size to 2 x 16 x sess
        aa_all = [oo.T, ff.T]
        np.shape(aa_all) # 2 x 16 x session


        ##############################################################################################
        # set the data frame to be used for ANOVA        
#         print(f'\n\n==============================\n========== {cre} --- {area_pairs[iap]} ==========\n==============================\n\n')

        aan = 'omission', 'image'

        for of in range(2): # omission, image # of=0
#             print(f'\n========== {aan[of]} ==========')
            aa = aa_all[of]
            ####
#             cols = ['area_pair', 'depth_pair', 'event', 'value']
#             a_data = pd.DataFrame([], columns = cols)
#             i_at = -1
            for i_depth_pair in range(aa.shape[0]): # i_depth_pair=0
                a_now = aa[i_depth_pair,:][~np.isnan(aa[i_depth_pair])]
                for i_a in range(len(a_now)): # i_a=0
                    i_at = i_at+1
                    a_data.at[i_at, 'cre'] = cre 
                    a_data.at[i_at, 'area_pair'] = area_pairs[iap] 
                    a_data.at[i_at, 'depth_pair'] = i_depth_pair
                    a_data.at[i_at, 'event'] = aan[of]
                    a_data.at[i_at, 'value'] = a_now[i_a]

            a_data.at[:,'value'] = a_data['value'].values.astype(float)
#             a_data



    #######################################################
    #######################################################
    # Do anova and tukey for each area: compare depths
    #######################################################
    #######################################################

    for ars in area_pairs: # ars = 'LM-V1' # ars = 'V1-V1' #, 'LM-LM'
        for of in range(2): # omission, image # of=0
            print(f'\n========== {cre} -- {ars} -- {aan[of]} ==========')
            print(ars)
            a_data_now = a_data[np.logical_and(a_data['area_pair'].values==ars , a_data['event'].values==aan[of])]
#                 a_data_now = a_data

            ### ANOVA
            # https://reneshbedre.github.io/blog/anova.html        
            # https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/
            # https://help.xlstat.com/s/article/how-to-interpret-contradictory-results-between-anova-and-multiple-pairwise-comparisons?language=es
            # https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/

            ######## ANOVA: compare depths for a given area
            # Ordinary Least Squares (OLS) model
            # 2-way
            # C(Genotype):C(years) represent interaction term        
    #         model = ols('value ~ C(area) + C(depth) + C(area):C(depth)', data=a_data).fit()
            # 1-way
            model = ols('value ~ C(depth_pair)', data=a_data_now).fit() 
    #         anova_table = sm.stats.anova_lm(model, typ=3)
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)        
            print('\n')
            # scipy anova: same result as above
    #                 a = aa[inds_v1,:]
    #                 fvalue, pvalue = st.f_oneway(a[0][~np.isnan(a[0])], a[1][~np.isnan(a[1])], a[2][~np.isnan(a[2])], a[3][~np.isnan(a[3])])
    #                 print(fvalue, pvalue)


            ######### TUKEY HSD: compare depths for a given area #########        

    #         v = a_data['value']
    #         f = a_data['depth']
            v = a_data_now['value'] #a_data['value']
            f = a_data_now['depth_pair'] #a_data['depth']

            MultiComp = MultiComparison(v, f)
            mcs = MultiComp.tukeyhsd().summary()
#                 print(mcs) # Show all pair-wise comparisons

            # show only significant rows
            a = np.array(mcs.data)
            # columns names:
            # ['group1' 'group2' 'meandiff' 'p-adj' 'lower' 'upper' 'reject']
            aa = a[1:,:] # remove the column names
            print(sum(aa[:,-1]=='True'))
            sigpairs = np.concatenate((a[0][np.newaxis,:], aa[aa[:,-1]=='True', :]), axis=0)
            print(sigpairs)

            # get all pair indices of the significant pairs
            pair_inds_sig = sigpairs[1:,:2]
            a3 = pair_inds_sig.flatten() # flattens horizontally; ie 1st row of pair_inds_sig, then 2nd row of it, etc
            ints = [int(a3[iint]) for iint in range(len(a3))] # convert to int (they are strings in pair_inds_sig)
            pair_inds_sig = np.reshape(ints, pair_inds_sig.shape) # reshape it back to n x 2; same as pair_inds_sig except it's int not str
            
            # convert pair inds to depth inds
            depth_inds_sig = convert_pair_index_to_area_depth_index(ints, num_depth=4)
            
            depth_inds_sig_ineraction_pairs = np.reshape(depth_inds_sig, (int(depth_inds_sig.shape[0]/2), 4))
            
            for ir in np.arange(0, depth_inds_sig.shape[0], 2):
                a = f'{depth_inds_sig[ir]} vs {depth_inds_sig[ir+1]}'
                print(a)
                
                
                
            
            