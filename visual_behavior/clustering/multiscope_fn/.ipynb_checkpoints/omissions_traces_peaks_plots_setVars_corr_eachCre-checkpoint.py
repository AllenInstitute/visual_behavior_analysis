#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vars are set in "omissions_traces_peaks_plots_setVars_corr.py"

Here we get vars to make summary plots for mice in each cre line (for individual mouse, also summary across mice)

This function calls  "omissions_traces_peaks_plots_setVars_corr_sumMice.py" to make summary plots for mice.
We can also call omissions_traces_peaks_plots_setVars_corr_eachMouse.py to make plots for each mouse. (currently commenetd out.)


Created on Mon Sep 23 15:58:29 2019
@author: farzaneh
"""

# vars below apply to average-mice plots
plot_cc_traces = 1 #1
plot_cc_peaks = 1 #1
plot_cc_peaks_heatmaps = 1
plot_cc_pooled = 1 #1
plot_cc_pooled_superimposed = 1 #0


#%% 
###############################################################################
######## Set vars to summarize data across mice of the same cre line ##########
###############################################################################
###############################################################################

#%% 

cre_all = corr_trace_peak_allMice['cre'].values # len_mice_withData
cre_lines = np.unique(cre_all)

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
    

    