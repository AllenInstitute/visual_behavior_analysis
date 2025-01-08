"""
Run "svm_plots_setVars.py" to set vars needed here, mainly svm_this_plane_allsess.

This script sets vars that will be used for making summary plots across mice.

Follow this script by "svm_plots_sumMice" to make summary plots across mice (each cre line).

Note, this script is similar to part of omissions_traces_peaks_plots_setVars_ave.py, which is related to setting vars for making summary plots across mice.
Here, we set vars that are similar to the following vars in script omissions_traces_peaks_plots_setVars_ave.py:
    svm_allMice_sessPooled (similar to trace_peak_allMice_sessPooled): includes vars pooled across all sessions of all mice (for each plane, each area, and each depth); # use this to make summary mouse plots, by averaging across pooled sessions of all mice.
    svm_allMice_sessAvSd (similar to trace_peak_allMice_sessAvSd): includes vars pooled across all sessions of all mice (for each plane, each area, and each depth); # use this to make summary mouse plots, by averaging across mice (each mouse is here averaged across sessions).
    (note: some of the vars in "svm_allMice_sessAvSd" are also set in svm_plots_setVars.py (in var "svm_this_plane_allsess") . I didn't take them out of there, bc they will be used for making eachMouse plots. "svm_allMice_sessAvSd" is compatible with omissions code, and will be used for making sumMice plots.)
    
If all_ABtransit_AbefB_Aall is 1 (analyzing A to B transition), we compute average and standard error traces and peaks, across mice for each cre line. Done for each plane separately (and each session).
    # final vars will have size:  num_cre x (8 x num_sessions).
    # remember num_sessions = 2 (last A and first B).

Note: for ABtransit, here, in this script, we group data for each cre line. 
      but for non-ABtransit, we will group data in "svm_plots_sumMice.py"... here we only pool data across all sessions of all mouse.


Note: you can run the codes for no

Created on Tue Apr 10 11:56:05 2020
@author: farzaneh
    
"""

#%% ##########################################################################################
############### Set vars to make average plots across mice of the same cell line #############
################ (each plane individually, no pooling across areas or layers) ###########################
##############################################################################################

# Remember each row of svm_this_plane_allsess is for each mouse; below we combine data from all mice

#%% Pool all sessions of all mice
# pandas series "svm_allMice_sessPooled" is created that includes all the important vars below.
# use this to make summary mouse plots, by averaging across pooled sessions of all mice.

a0 = np.concatenate((svm_this_plane_allsess['cre_exp']).values) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
cre_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        

aa = svm_this_plane_allsess['area_this_plane_allsess_allp'].values
#    aa.shape # num_all_mice; # size of each element : 8 x num_sessions x 1
# For each element of aa, make array sizes similar to those in omissions_traces_peaks_plots_setVars_ave.py:
# concatenate planes of all sessions (1st, all planes of session 1, then all planes of session 2, etc.)
aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))])
#    aar.shape # num_all_mice; # size of each element : (8 x num_sessions) x 1
a0 = np.concatenate((aar)) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
area_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        


aa = svm_this_plane_allsess['depth_this_plane_allsess_allp'].values # num_all_mice; # size of each element : 8 x num_sessions x 1
aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))]) # aar.shape # num_all_mice; # size of each element : (8 x num_sessions) x 1
a0 = np.concatenate((aar)) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
depth_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is


session_labs_all = np.unique(np.concatenate((svm_this_plane_allsess['session_labs']).values))


########## individual planes ##########

aa = svm_this_plane_allsess['av_test_data_this_plane_allsess_allp'].values # num_all_mice; # size of each element : 8 x num_sessions x n_frs
aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))]) # aar.shape # num_all_mice; # size of each element : (8 x num_sessions) x n_frs
a0 = np.concatenate((aar)) # (8*sum(num_sess_per_mouse)) x n_frs # 8 planes of 1 session, then 8 planes of next session, and so on 
ts_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes), a0.shape[-1]), order='F') # 8 x sum(num_sess_per_mouse) x n_frs # each row is one plane, all sessions        
#        aa = np.reshape(np.concatenate((svm_this_plane_allsess['plane'].values)), (num_planes, int(a0.shape[0]/num_planes)), order='F')

aa = svm_this_plane_allsess['av_test_shfl_this_plane_allsess_allp'].values # num_all_mice; # size of each element : 8 x num_sessions x n_frs
aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))]) # aar.shape # num_all_mice; # size of each element : (8 x num_sessions) x n_frs
a0 = np.concatenate((aar)) # (8*sum(num_sess_per_mouse)) x n_frs # 8 planes of 1 session, then 8 planes of next session, and so on 
sh_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes), a0.shape[-1]), order='F') # 8 x sum(num_sess_per_mouse) x n_frs # each row is one plane, all sessions        

aa = svm_this_plane_allsess['peak_amp_omit_trTsShCh_this_plane_allsess_allp'].values # num_all_mice; # size of each element : 8 x num_sessions x 4
aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))]) # aar.shape # num_all_mice; # size of each element : (8 x num_sessions) x 4
a0 = np.concatenate((aar)) # (8*sum(num_sess_per_mouse)) x 4 # 8 planes of 1 session, then 8 planes of next session, and so on 
pa_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes), a0.shape[-1]), order='F') # 8 x sum(num_sess_per_mouse) x 4 # each row is one plane, all sessions        

aa = svm_this_plane_allsess['peak_timing_omit_trTsShCh_this_plane_allsess_allp'].values # num_all_mice; # size of each element : 8 x num_sessions x 4
aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))]) # aar.shape # num_all_mice; # size of each element : (8 x num_sessions) x 4
a0 = np.concatenate((aar)) # (8*sum(num_sess_per_mouse)) x 4 # 8 planes of 1 session, then 8 planes of next session, and so on 
pt_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes), a0.shape[-1]), order='F') # 8 x sum(num_sess_per_mouse) x 4 # each row is one plane, all sessions        

aa = svm_this_plane_allsess['peak_amp_flash_trTsShCh_this_plane_allsess_allp'].values # num_all_mice; # size of each element : 8 x num_sessions x 4
aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))]) # aar.shape # num_all_mice; # size of each element : (8 x num_sessions) x 4
a0 = np.concatenate((aar)) # (8*sum(num_sess_per_mouse)) x 4 # 8 planes of 1 session, then 8 planes of next session, and so on 
paf_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes), a0.shape[-1]), order='F') # 8 x sum(num_sess_per_mouse) x 4 # each row is one plane, all sessions        

aa = svm_this_plane_allsess['peak_timing_flash_trTsShCh_this_plane_allsess_allp'].values # num_all_mice; # size of each element : 8 x num_sessions x 4
aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))]) # aar.shape # num_all_mice; # size of each element : (8 x num_sessions) x 4
a0 = np.concatenate((aar)) # (8*sum(num_sess_per_mouse)) x 4 # 8 planes of 1 session, then 8 planes of next session, and so on 
ptf_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes), a0.shape[-1]), order='F') # 8 x sum(num_sess_per_mouse) x 4 # each row is one plane, all sessions        


########## Each area: layers pooled ########## 

cre_all_eachArea = np.concatenate((svm_this_plane_allsess['cre_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse))
ts_all_eachArea = np.concatenate((svm_this_plane_allsess['av_test_data_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse)) x 1270
sh_all_eachArea = np.concatenate((svm_this_plane_allsess['av_test_shfl_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse)) x 1270    
pao_all_eachArea = np.concatenate((svm_this_plane_allsess['peak_amp_omit_trTsShCh_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse)) x 4(train,test,shuffle,chance)
pto_all_eachArea = np.concatenate((svm_this_plane_allsess['peak_timing_omit_trTsShCh_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse))
paf_all_eachArea = np.concatenate((svm_this_plane_allsess['peak_amp_flash_trTsShCh_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse))
ptf_all_eachArea = np.concatenate((svm_this_plane_allsess['peak_timing_flash_trTsShCh_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse))


########## Each layer: areas pooled ########## svm_this_plane_allsess['trace_pooled_eachDepth']

cre_all_eachDepth = np.concatenate((svm_this_plane_allsess['cre_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))
ts_all_eachDepth = np.concatenate((svm_this_plane_allsess['av_test_data_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse)) x 1270
sh_all_eachDepth = np.concatenate((svm_this_plane_allsess['av_test_shfl_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse)) x 1270    
pao_all_eachDepth = np.concatenate((svm_this_plane_allsess['peak_amp_omit_trTsShCh_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse)) x 4(train,test,shuffle,chance)
pto_all_eachDepth = np.concatenate((svm_this_plane_allsess['peak_timing_omit_trTsShCh_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))
paf_all_eachDepth = np.concatenate((svm_this_plane_allsess['peak_amp_flash_trTsShCh_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))
ptf_all_eachDepth = np.concatenate((svm_this_plane_allsess['peak_timing_flash_trTsShCh_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))

depth_all_eachDepth = np.concatenate((svm_this_plane_allsess['depth_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))    


########### Make a pandas series to keep the pooled session data of all mice ########### 

indexes = ['cre_allPlanes', 'area_allPlanes', 'depth_allPlanes', 'session_labs', \
           'av_test_data_allPlanes', 'av_test_shfl_allPlanes', 'peak_amp_omit_allPlanes', 'peak_timing_omit_allPlanes', 'peak_amp_flash_allPlanes', 'peak_timing_flash_allPlanes',\
           'cre_eachArea', 'av_test_data_eachArea', 'av_test_shfl_eachArea', 'peak_amp_omit_eachArea', 'peak_timing_omit_eachArea', 'peak_amp_flash_eachArea', 'peak_timing_flash_eachArea',\
           'cre_eachDepth', 'depth_eachDepth', 'av_test_data_eachDepth', 'av_test_shfl_eachDepth', 'peak_amp_omit_eachDepth', 'peak_timing_omit_eachDepth', 'peak_amp_flash_eachDepth', 'peak_timing_flash_eachDepth']

svm_allMice_sessPooled = pd.Series([cre_all, area_all, depth_all, session_labs_all, \
   ts_all, sh_all, pa_all, pt_all, paf_all, ptf_all,\
   cre_all_eachArea, ts_all_eachArea, sh_all_eachArea, pao_all_eachArea, pto_all_eachArea, paf_all_eachArea, ptf_all_eachArea,\
   cre_all_eachDepth, depth_all_eachDepth, ts_all_eachDepth, sh_all_eachDepth, pao_all_eachDepth, pto_all_eachDepth, paf_all_eachDepth, ptf_all_eachDepth],\
   index=indexes)

print(svm_allMice_sessPooled)
#     t_all[:, cre_exp_all[0,:]=='Sst'].shape    



######################################################################
######################################################################
######################################################################
#%% Compute session-averaged data for each mouse
# pandas dataFrame "svm_allMice_sessAvSd" is created to keep the important vars below
# use this to make summary mouse plots, by averaging across mice (each mouse is here averaged across sessions).
######################################################################
######################################################################
######################################################################

cols0 = ['mouse_id', 'cre', 'session_stages', 'session_labs', 'area', 'depth', 'plane', \
        'av_depth_avSess_eachP', 'sd_depth_avSess_eachP', 'av_n_neurons_avSess_eachP', 'sd_n_neurons_avSess_eachP', \
        'av_n_omissions_avSess_eachP', 'sd_n_omissions_avSess_eachP', 'av_test_data_avSess_eachP', 'sd_test_data_avSess_eachP', 'av_test_shfl_avSess_eachP', 'sd_test_shfl_avSess_eachP', \
        'av_peak_amp_omit_trTsShCh_avSess_eachP', 'sd_peak_amp_omit_trTsShCh_avSess_eachP', 'av_peak_timing_omit_trTsShCh_avSess_eachP', 'sd_peak_timing_omit_trTsShCh_avSess_eachP', \
        'av_peak_amp_flash_trTsShCh_avSess_eachP', 'sd_peak_amp_flash_trTsShCh_avSess_eachP', 'av_peak_timing_flash_trTsShCh_avSess_eachP', 'sd_peak_timing_flash_trTsShCh_avSess_eachP', \
        'num_sessLayers_valid_eachArea', 'av_test_data_pooledSessPlanes_eachArea', 'sd_test_data_pooledSessPlanes_eachArea', 'av_test_shfl_pooledSessPlanes_eachArea', 'sd_test_shfl_pooledSessPlanes_eachArea', \
        'av_pao_trTsShCh_pooledSessPlanes_eachArea', 'sd_pao_trTsShCh_pooledSessPlanes_eachArea', 'av_pto_trTsShCh_pooledSessPlanes_eachArea', 'sd_pto_trTsShCh_pooledSessPlanes_eachArea', \
        'av_paf_trTsShCh_pooledSessPlanes_eachArea', 'sd_paf_trTsShCh_pooledSessPlanes_eachArea', 'av_ptf_trTsShCh_pooledSessPlanes_eachArea', 'sd_ptf_trTsShCh_pooledSessPlanes_eachArea', \
        'num_sessAreas_valid_eachDepth', 'depth_ave', \
        'av_test_data_pooledSessAreas_eachDepth', 'sd_test_data_pooledSessAreas_eachDepth', 'av_test_shfl_pooledSessAreas_eachDepth', 'sd_test_shfl_pooledSessAreas_eachDepth', \
        'av_pao_trTsShCh_pooledSessAreas_eachDepth', 'sd_pao_trTsShCh_pooledSessAreas_eachDepth', 'av_pto_trTsShCh_pooledSessAreas_eachDepth', 'sd_pto_trTsShCh_pooledSessAreas_eachDepth', \
        'av_paf_trTsShCh_pooledSessAreas_eachDepth', 'sd_paf_trTsShCh_pooledSessAreas_eachDepth', 'av_ptf_trTsShCh_pooledSessAreas_eachDepth', 'sd_ptf_trTsShCh_pooledSessAreas_eachDepth']

if same_num_neuron_all_planes:
    cols = np.concatenate((cols0, ['av_n_neurons_svm_trained_avSess_eachP', 'sd_n_neurons_svm_trained_avSess_eachP'])) 
else:
    cols = cols0


svm_allMice_sessAvSd = pd.DataFrame([], columns=cols)


for im in range(len(svm_this_plane_allsess)): # im=5

    #%%
    mouse_id = svm_this_plane_allsess.iloc[im]['mouse_id']
    cre = svm_this_plane_allsess.iloc[im]['cre']
    session_stages = svm_this_plane_allsess.iloc[im]['session_stages']
    session_labs = svm_this_plane_allsess.iloc[im]['session_labs']
    num_sessions = len(session_stages)


    ################################################################
    #%% Session-averaged, each plane ; omission-aligned traces
    ################################################################
    ######### For each plane, average vars across days #########        
    # note: some of the vars below are also set in svm_plots_setVars.py (in var "svm_this_plane_allsess") . I didn't take them out of there, bc they will be used for making eachMouse plots. Below is compatible with omissions code, and will be used for making sumMice plots.

    num_sessions_valid = svm_this_plane_allsess.iloc[im]['num_sessions_valid']

    a0 = svm_this_plane_allsess.iloc[im]['area']
    areas = np.reshape(a0, (num_planes, num_sessions), order='F') # 8 x num_sessions # each row is one plane, all sessions
    a0 = svm_this_plane_allsess.iloc[im]['depth']
    depths = np.reshape(a0, (num_planes, num_sessions), order='F') # 8 x num_sessions # each row is one plane, all sessions
    a0 = svm_this_plane_allsess.iloc[im]['plane']
    planes = np.reshape(a0, (num_planes, num_sessions), order='F') # 8 x num_sessions # each row is one plane, all sessions

    # depth
    a = svm_this_plane_allsess.iloc[im]['depth_this_plane_allsess_allp'] # size: (8 x num_sessions x 1)
    av_depth_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
    sd_depth_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane

    # num_neurons
    a = svm_this_plane_allsess.iloc[im]['n_neurons_this_plane_allsess_allp'] # size: (8 x num_sessions x 1)
    av_n_neurons_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
    sd_n_neurons_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane

    # population size used for svm training
    if same_num_neuron_all_planes:
        a = svm_this_plane_allsess.iloc[im]['n_neurons_svm_trained_this_plane_allsess_allp'] # size: (8 x num_sessions x 1)
        av_n_neurons_svm_trained_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
        sd_n_neurons_svm_trained_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane

    # num_omissions
    a = svm_this_plane_allsess.iloc[im]['n_omissions_this_plane_allsess_allp'] # size: (8 x num_sessions x 1)
    av_n_omissions_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
    sd_n_omissions_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane


    # av_test_data
    a = svm_this_plane_allsess.iloc[im]['av_test_data_this_plane_allsess_allp'] # size: (8 x num_sessions x nFrames_upsampled)
    av_test_data_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
    sd_test_data_avSess_eachP = np.nanstd(a,axis=1) / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x nFrames_upsampled # st error across sessions, for each plane

    # av_test_shfl
    a = svm_this_plane_allsess.iloc[im]['av_test_shfl_this_plane_allsess_allp'] # size: (8 x num_sessions x nFrames_upsampled)
    av_test_shfl_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
    sd_test_shfl_avSess_eachP = np.nanstd(a,axis=1) / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x nFrames_upsampled # st error across sessions, for each plane


    # peak_amp_omit_trTsShCh
    a = svm_this_plane_allsess.iloc[im]['peak_amp_omit_trTsShCh_this_plane_allsess_allp'] # size: (8 x num_sessions x 4)
    av_peak_amp_omit_trTsShCh_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes x 4
    sd_peak_amp_omit_trTsShCh_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x 4 # st error across sessions, for each plane

    # peak_timing_omit_trTsShCh
    a = svm_this_plane_allsess.iloc[im]['peak_timing_omit_trTsShCh_this_plane_allsess_allp'] # size: (8 x num_sessions x 4)
    av_peak_timing_omit_trTsShCh_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes x 4
    sd_peak_timing_omit_trTsShCh_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x 4 # st error across sessions, for each plane

    # peak_amp_flash_trTsShCh
    a = svm_this_plane_allsess.iloc[im]['peak_amp_flash_trTsShCh_this_plane_allsess_allp'] # size: (8 x num_sessions x 4)
    av_peak_amp_flash_trTsShCh_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes x 4
    sd_peak_amp_flash_trTsShCh_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x 4 # st error across sessions, for each plane

    # peak_timing_flash_trTsShCh
    a = svm_this_plane_allsess.iloc[im]['peak_timing_flash_trTsShCh_this_plane_allsess_allp'] # size: (8 x num_sessions x 4)
    av_peak_timing_flash_trTsShCh_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes x 4
    sd_peak_timing_flash_trTsShCh_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid)[:, np.newaxis] # num_planes x 4 # st error across sessions, for each plane


    ################################################################
    #%% Data per area: pooled across sessions and layers for each area
    # Question 1: is there a difference between areas (V1 vs LM) # for each mouse pool across sessions and layers (for each area)

    # Get the total number of valid sessions and layers for each area        
    num_sessions_valid_eachArea = svm_this_plane_allsess.iloc[im]['num_sessions_valid_eachArea']
    num_sessLayers_valid_eachArea = np.sum(num_sessions_valid_eachArea, axis=1)

    # Average across layers and sessions for each area
    # test_data
    a = svm_this_plane_allsess.iloc[im]['av_test_data_pooled_sesss_planes_eachArea'] # 2 x (4 x num_sessions) x nFrames_upsampled
    av_test_data_pooledSessPlanes_eachArea = np.nanmean(a, axis=1) # 2 x nFrames_upsampled 
    sd_test_data_pooledSessPlanes_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea)[:,np.newaxis] # 2 x nFrames_upsampled 

    # test_shfl
    a = svm_this_plane_allsess.iloc[im]['av_test_shfl_pooled_sesss_planes_eachArea'] # 2 x (4 x num_sessions) x nFrames_upsampled
    av_test_shfl_pooledSessPlanes_eachArea = np.nanmean(a, axis=1) # 2 x nFrames_upsampled 
    sd_test_shfl_pooledSessPlanes_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea)[:,np.newaxis] # 2 x nFrames_upsampled 


    # peak_amp
    a = svm_this_plane_allsess.iloc[im]['peak_amp_omit_trTsShCh_pooled_sesss_planes_eachArea'] # 2 x (4 x num_sessions) x 4(train, test, shuffle, chance)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_pao_trTsShCh_pooledSessPlanes_eachArea = np.nanmean(a, axis=1) # 2 x 4 
    sd_pao_trTsShCh_pooledSessPlanes_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea)[:,np.newaxis] # 2 x 4(train, test, shuffle, chance) 

    # peak_timing
    a = svm_this_plane_allsess.iloc[im]['peak_timing_omit_trTsShCh_pooled_sesss_planes_eachArea'] # 2 x (4 x num_sessions)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_pto_trTsShCh_pooledSessPlanes_eachArea = np.nanmean(a, axis=1) # 2 x 4 
    sd_pto_trTsShCh_pooledSessPlanes_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea)[:,np.newaxis] # 2 x 4(train, test, shuffle, chance) 

    # peak_amp_flash
    a = svm_this_plane_allsess.iloc[im]['peak_amp_flash_trTsShCh_pooled_sesss_planes_eachArea'] # 2 x (4 x num_sessions)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_paf_trTsShCh_pooledSessPlanes_eachArea = np.nanmean(a, axis=1) # 2 x 4 
    sd_paf_trTsShCh_pooledSessPlanes_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea)[:,np.newaxis] # 2 x 4(train, test, shuffle, chance) 

    # peak_timing_flash
    a = svm_this_plane_allsess.iloc[im]['peak_timing_flash_trTsShCh_pooled_sesss_planes_eachArea'] # 2 x (4 x num_sessions)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_ptf_trTsShCh_pooledSessPlanes_eachArea = np.nanmean(a, axis=1) # 2 x 4 
    sd_ptf_trTsShCh_pooledSessPlanes_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea)[:,np.newaxis] # 2 x 4(train, test, shuffle, chance) 



    ################################################################
    #%% Data per depth: pooled across sessions and areas for each depth
    # Question 2: is there a difference between superficial and deep layers # for each mouse pool across sessions and area (for each layer)... also pool 2 superifical (deep) layers into 1         

    # set average depth across sessions and areas            
    d = svm_this_plane_allsess.iloc[im]['depth_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions)
    depth_ave = np.mean(d, axis=1) # average across areas and sessions
#    b, _,_ = pool_sesss_planes_eachArea(area, area)

    # Get the total number of valid sessions and layers for each depth
    num_sessAreas_valid_eachDepth = np.sum(num_sessions_valid_eachArea, axis=0)

    # Average across areas and sessions for each depth
    # test_data
    a = svm_this_plane_allsess.iloc[im]['av_test_data_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions) x nFrames_upsampled
    av_test_data_pooledSessAreas_eachDepth = np.nanmean(a, axis=1) # 4 x nFrames_upsampled 
    sd_test_data_pooledSessAreas_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth)[:,np.newaxis] # 4 x nFrames_upsampled 

    # test_shfl
    a = svm_this_plane_allsess.iloc[im]['av_test_shfl_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions) x nFrames_upsampled
    av_test_shfl_pooledSessAreas_eachDepth = np.nanmean(a, axis=1) # 4 x nFrames_upsampled 
    sd_test_shfl_pooledSessAreas_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth)[:,np.newaxis] # 4 x nFrames_upsampled 


    # peak_amp
    a = svm_this_plane_allsess.iloc[im]['peak_amp_omit_trTsShCh_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions) x 4(train, test, shuffle, chance)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_pao_trTsShCh_pooledSessAreas_eachDepth = np.nanmean(a, axis=1) # 4 x 4 
    sd_pao_trTsShCh_pooledSessAreas_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth)[:,np.newaxis] # 4 x 4(train, test, shuffle, chance) 

    # peak_timing
    a = svm_this_plane_allsess.iloc[im]['peak_timing_omit_trTsShCh_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_pto_trTsShCh_pooledSessAreas_eachDepth = np.nanmean(a, axis=1) # 4 x 4 
    sd_pto_trTsShCh_pooledSessAreas_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth)[:,np.newaxis] # 4 x 4(train, test, shuffle, chance) 

    # peak_amp_flash
    a = svm_this_plane_allsess.iloc[im]['peak_amp_flash_trTsShCh_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_paf_trTsShCh_pooledSessAreas_eachDepth = np.nanmean(a, axis=1) # 4 x 4 
    sd_paf_trTsShCh_pooledSessAreas_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth)[:,np.newaxis] # 4 x 4(train, test, shuffle, chance) 

    # peak_timing_flash
    a = svm_this_plane_allsess.iloc[im]['peak_timing_flash_trTsShCh_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_ptf_trTsShCh_pooledSessAreas_eachDepth = np.nanmean(a, axis=1) # 4 x 4 
    sd_ptf_trTsShCh_pooledSessAreas_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth)[:,np.newaxis] # 4 x 4(train, test, shuffle, chance) 


    #%% Keep vars (session averaged, each plane, also eachArea, eachDepth vars) for all mice

    svm_allMice_sessAvSd.at[im,cols0] = \
        mouse_id, cre, session_stages, session_labs, areas, depths, planes, \
        av_depth_avSess_eachP, sd_depth_avSess_eachP, av_n_neurons_avSess_eachP, sd_n_neurons_avSess_eachP, \
        av_n_omissions_avSess_eachP, sd_n_omissions_avSess_eachP, av_test_data_avSess_eachP, sd_test_data_avSess_eachP, av_test_shfl_avSess_eachP, sd_test_shfl_avSess_eachP, \
        av_peak_amp_omit_trTsShCh_avSess_eachP, sd_peak_amp_omit_trTsShCh_avSess_eachP, av_peak_timing_omit_trTsShCh_avSess_eachP, sd_peak_timing_omit_trTsShCh_avSess_eachP, \
        av_peak_amp_flash_trTsShCh_avSess_eachP, sd_peak_amp_flash_trTsShCh_avSess_eachP, av_peak_timing_flash_trTsShCh_avSess_eachP, sd_peak_timing_flash_trTsShCh_avSess_eachP, \
        num_sessLayers_valid_eachArea, av_test_data_pooledSessPlanes_eachArea, sd_test_data_pooledSessPlanes_eachArea, av_test_shfl_pooledSessPlanes_eachArea, sd_test_shfl_pooledSessPlanes_eachArea, \
        av_pao_trTsShCh_pooledSessPlanes_eachArea, sd_pao_trTsShCh_pooledSessPlanes_eachArea, av_pto_trTsShCh_pooledSessPlanes_eachArea, sd_pto_trTsShCh_pooledSessPlanes_eachArea, \
        av_paf_trTsShCh_pooledSessPlanes_eachArea, sd_paf_trTsShCh_pooledSessPlanes_eachArea, av_ptf_trTsShCh_pooledSessPlanes_eachArea, sd_ptf_trTsShCh_pooledSessPlanes_eachArea, \
        num_sessAreas_valid_eachDepth, depth_ave, \
        av_test_data_pooledSessAreas_eachDepth, sd_test_data_pooledSessAreas_eachDepth, av_test_shfl_pooledSessAreas_eachDepth, sd_test_shfl_pooledSessAreas_eachDepth, \
        av_pao_trTsShCh_pooledSessAreas_eachDepth, sd_pao_trTsShCh_pooledSessAreas_eachDepth, av_pto_trTsShCh_pooledSessAreas_eachDepth, sd_pto_trTsShCh_pooledSessAreas_eachDepth, \
        av_paf_trTsShCh_pooledSessAreas_eachDepth, sd_paf_trTsShCh_pooledSessAreas_eachDepth, av_ptf_trTsShCh_pooledSessAreas_eachDepth, sd_ptf_trTsShCh_pooledSessAreas_eachDepth

    if same_num_neuron_all_planes:
        svm_allMice_sessAvSd.at[im, ['av_n_neurons_svm_trained_avSess_eachP']] = [av_n_neurons_svm_trained_avSess_eachP]
        svm_allMice_sessAvSd.at[im, ['sd_n_neurons_svm_trained_avSess_eachP']] = [sd_n_neurons_svm_trained_avSess_eachP]

print(svm_allMice_sessAvSd.shape)




#################################################################
#################################################################
#################################################################
#%% A-B transition
#################################################################
#################################################################
#################################################################

#%% Take average across mice of the same cre line, do it for all the columns in svm_this_plane_allsess
        
if all_ABtransit_AbefB_Aall==1: # take average across mice for A sessions, and B sessions, separately.
    # note: ABtransit is a special case because all mice have the same number of sessions, ie 2 (A and B).
    
    # groupby below doesnt work in jupyterlab (apparently due to memory issues)!!
    '''       
    cre_group_ave = svm_this_plane_allsess.groupby('cre').apply(lambda x: np.mean(x, axis=0)) # np.mean # for this to work, all mice need to have the same number of sessions.
    print(cre_group_ave)
    print(cre_group_ave.columns)
    
    
    all_cres = cre_group_ave.index.values
    num_each_cre = svm_this_plane_allsess.groupby('cre').apply(len)
    #print(all_cres)
    print(num_each_cre)
    #print(cre_group_ave)
    print([cre_group_ave.iloc[0]['trace'].shape, cre_group_ave.iloc[0]['depth'].shape ,cre_group_ave.iloc[0]['peak_amp'].shape, cre_group_ave.iloc[0]['peak_timing'].shape])
    
    
    ############ set the traces averaged across mice for each cre line. Done for each plane separately (and each session) ############
    # we take advantage of pandas groupby (average function) that we made above
    depth_allCre = []
    trace_ave_allCre = []
    peak_amp_ave_allCre = []
    peak_timing_ave_allCre = []
    peak_amp_flash_ave_allCre = []
    peak_timing_flash_ave_allCre = []
    
    for cre_now in all_cres: # cre_now = all_cres[0]
        # average depth across mice (for each plane and each session)
        d = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['depth'].values).squeeze() # (num_sess*num_planes)
        # average trace across mice (for each plane and each session)
        # each row of t is one plane (of one session), and average across mice that are in the same cre group
        t = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['trace'].values) # (num_sess*num_planes) x numFrames        
        pa = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['peak_amp'].values).squeeze() # (num_sess*num_planes)
        pt = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['peak_timing'].values).squeeze() # (num_sess*num_planes)
        paf = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['peak_amp_flash'].values).squeeze() # (num_sess*num_planes)
        ptf = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['peak_timing_flash'].values).squeeze() # (num_sess*num_planes)
        
        depth_allCre.append(d)
        trace_ave_allCre.append(t)    
        peak_amp_ave_allCre.append(pa)    
        peak_timing_ave_allCre.append(pt)    
        peak_amp_flash_ave_allCre.append(paf)    
        peak_timing_flash_ave_allCre.append(ptf)    
        
    depth_allCre = np.array(depth_allCre) # num_cre x (num_sess*num_planes)
    trace_ave_allCre = np.array(trace_ave_allCre) # num_cre x (num_sess*num_planes) x num_frs
    peak_amp_ave_allCre = np.array(peak_amp_ave_allCre) # num_cre x (num_sess*num_planes)
    peak_timing_ave_allCre = np.array(peak_timing_ave_allCre) # num_cre x (num_sess*num_planes)
    peak_amp_flash_ave_allCre = np.array(peak_amp_flash_ave_allCre) # num_cre x (num_sess*num_planes)
    peak_timing_flash_ave_allCre = np.array(peak_timing_flash_ave_allCre) # num_cre x (num_sess*num_planes)
    #depth_all = np.reshape(depth_allCre, (len(all_cres), num_planes, num_sessions), order='F') # num_cre x num_planes x num_sess
    '''    

    # if groupby (above) works, uncomment all below (the average computing section), except for the line below that defines "cres".
    cres = svm_this_plane_allsess['cre'].values
    all_cres = np.unique(cres) # cre_unq
    num_each_cre = np.array([np.sum(cres==all_cres[i]) for i in range(len(all_cres))])

    
    #%% Set the average and standard error traces: across mice for each cre line. Done for each plane separately (and each session) ############
    # (use this if the pandas groupby above doesnt work)

    depth_ave_allCre = []
    av_test_data_ave_allCre = []
    av_test_shfl_ave_allCre = []
    peak_amp_omit_trTsShCh_ave_allCre = []
    peak_timing_omit_trTsShCh_ave_allCre = []
    peak_amp_flash_trTsShCh_ave_allCre = []
    peak_timing_flash_trTsShCh_ave_allCre = []
    peak_amp_mod_omit_trTsShCh_ave_allCre = [] # relative modulation in response amplitude and timing from A to B 
    peak_timing_mod_omit_trTsShCh_ave_allCre = []
    peak_amp_mod_flash_trTsShCh_ave_allCre = []
    peak_timing_mod_flash_trTsShCh_ave_allCre = []
    
    depth_sd_allCre = []
    av_test_data_sd_allCre = []
    av_test_shfl_sd_allCre = []
    peak_amp_omit_trTsShCh_sd_allCre = []
    peak_timing_omit_trTsShCh_sd_allCre = []
    peak_amp_flash_trTsShCh_sd_allCre = []
    peak_timing_flash_trTsShCh_sd_allCre = []
    peak_amp_mod_omit_trTsShCh_sd_allCre = []
    peak_timing_mod_omit_trTsShCh_sd_allCre = []
    peak_amp_mod_flash_trTsShCh_sd_allCre = []
    peak_timing_mod_flash_trTsShCh_sd_allCre = []

        
    # loop through each cre line
    ### remember: num_sessions will be 2, because it's A-B transition analysis.
    for icre in range(len(all_cres)): # icre = 0
        cre_now = all_cres[icre] 
        cre_i = np.in1d(cres, cre_now) #cre_unq[icre]) # group cre lines manually
#         num_mice_this_cre = sum(cre_i)
        
    
        ###### depth ######
        aa = svm_this_plane_allsess.iloc[cre_i]['depth_this_plane_allsess_allp'].values
    #    aa.shape # num_mice_this_cre; # size of each element : 8 x num_sessions x 1
        # For each element of aa, make array sizes similar to those in omissions_traces_peaks_plots_setVars_ave.py:
        # concatenate planes of all sessions (1st, all planes of session 1, then all planes of session 2, etc.)
        aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))])
    #    aar.shape # num_mice_this_cre x (8 x num_sessions) x 1
        y_thisCre_allMice = np.transpose(aar, (1,0,2))
    #    y_thisCre_allMice.shape # (8 x num_sessions) x num_mice_this_cre  # 32x3
        # compute mean across mice (of the same cre line) for each plane     
        d = np.nanmean(y_thisCre_allMice, axis=1).squeeze()  # (8 x num_sessions)
        ds = np.nanstd(y_thisCre_allMice, axis=1).squeeze() / np.sqrt(sum(cre_i))  # (8 x num_sessions)

        
        ###### av_test_data ######
        aa = svm_this_plane_allsess.iloc[cre_i]['av_test_data_this_plane_allsess_allp'].values 
    #    aa.shape # num_mice_this_cre; # size of each element : 8 x num_sessions x nFrames_upsampled
        # For each element of aa, make array sizes similar to those in omissions_traces_peaks_plots_setVars_ave.py:
        # concatenate planes of all sessions (1st, all planes of session 1, then all planes of session 2, etc.)
        aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))])
    #    aar.shape # num_mice_this_cre x (8 x num_sessions) x nFrames_upsampled
        y_thisCre_allMice = np.transpose(aar, (1,0,2))
    #    y_thisCre_allMice.shape # (8 x num_sessions) x num_mice_this_cre x num_frs  # 32x3x1270  
        # compute mean across mice (of the same cre line) for each plane     
        ts = np.nanmean(y_thisCre_allMice, axis=1)  # (8 x num_sessions) x num_frs    
        tss = np.nanstd(y_thisCre_allMice, axis=1) / np.sqrt(sum(cre_i))  # (8 x num_sessions) x num_frs        
        

        ###### av_test_shfl ######
        aa = svm_this_plane_allsess.iloc[cre_i]['av_test_shfl_this_plane_allsess_allp'].values 
        aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))])
        y_thisCre_allMice = np.transpose(aar, (1,0,2)) # y_thisCre_allMice.shape # (8 x num_sessions) x num_mice_this_cre x num_frs  # 32x3x1270  
        sh = np.nanmean(y_thisCre_allMice, axis=1)  # (8 x num_sessions) x num_frs    
        shs = np.nanstd(y_thisCre_allMice, axis=1) / np.sqrt(sum(cre_i)) # (8 x num_sessions) x num_frs            

        
        ###### peak amp omit ######
        aa = svm_this_plane_allsess.iloc[cre_i]['peak_amp_omit_trTsShCh_this_plane_allsess_allp'].values 
    #    aa.shape, aa[0].shape # num_mice_this_cre; # size of each element : 8 x num_sessions x 4 (train,test,shuffle,chance)
        # For each element of aa, make array sizes similar to those in omissions_traces_peaks_plots_setVars_ave.py:
        # concatenate planes of all sessions (1st, all planes of session 1, then all planes of session 2, etc.)
        aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))])
    #    aar.shape # num_mice_this_cre x (8 x num_sessions) x 4
        y_thisCre_allMice = np.transpose(aar, (1,0,2))
    #    y_thisCre_allMice.shape # (8 x num_sessions) x num_mice_this_cre x 4  # 32x3x4
        # compute mean across mice (of the same cre line) for each plane     
        pa = np.nanmean(y_thisCre_allMice, axis=1)  # (8 x num_sessions) x 4
        pas = np.nanstd(y_thisCre_allMice, axis=1) / np.sqrt(sum(cre_i)) # (8 x num_sessions) x 4        
        
        # compute relative change in resp amp from A to B sessions.
        As = y_thisCre_allMice[:num_planes]
        Bs = y_thisCre_allMice[num_planes:]
        ab_mod = (Bs - As) / (Bs + As) # 8 x num_mice_this_cre x 4 
        # compute mean across mice (of the same cre line) for each plane     
        pa_mod = np.nanmean(ab_mod, axis=1) # 8 x 4
        pa_mods = np.nanmean(ab_mod, axis=1) / np.sqrt(sum(cre_i)) # 8 x 4
    
    
        ###### peak timing omit ######
        aa = svm_this_plane_allsess.iloc[cre_i]['peak_timing_omit_trTsShCh_this_plane_allsess_allp'].values 
        aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))])
        y_thisCre_allMice = np.transpose(aar, (1,0,2))
        pt = np.nanmean(y_thisCre_allMice, axis=1)  # (8 x num_sessions) x 4  # compute mean across mice (of the same cre line) for each plane
        pts = np.nanstd(y_thisCre_allMice, axis=1) / np.sqrt(sum(cre_i))  # (8 x num_sessions) x 4  # compute std across mice (of the same cre line) for each plane

        # compute relative change in resp amp from A to B sessions.
        As = y_thisCre_allMice[:num_planes]
        Bs = y_thisCre_allMice[num_planes:]
        ab_mod = (Bs - As) / (Bs + As) # 8 x num_mice_this_cre x 4 
        # compute mean across mice (of the same cre line) for each plane     
        pt_mod = np.nanmean(ab_mod, axis=1)  # 8 x 4
        pt_mods = np.nanmean(ab_mod, axis=1) / np.sqrt(sum(cre_i)) # 8 x 4
        
        
        ###### peak amp flash ######
        aa = svm_this_plane_allsess.iloc[cre_i]['peak_amp_flash_trTsShCh_this_plane_allsess_allp'].values 
        aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))])
        y_thisCre_allMice = np.transpose(aar, (1,0,2))
        paf = np.nanmean(y_thisCre_allMice, axis=1)  # (8 x num_sessions) x 4    # compute mean across mice (of the same cre line) for each plane
        pafs = np.nanstd(y_thisCre_allMice, axis=1) / np.sqrt(sum(cre_i))  # (8 x num_sessions) x 4    # compute std across mice (of the same cre line) for each plane
    
        # compute relative change in resp amp from A to B sessions.
        As = y_thisCre_allMice[:num_planes]
        Bs = y_thisCre_allMice[num_planes:]
        ab_mod = (Bs - As) / (Bs + As) # 8 x num_mice_this_cre x 4 
        # compute mean across mice (of the same cre line) for each plane     
        paf_mod = np.nanmean(ab_mod, axis=1)  # 8 x 4
        paf_mods = np.nanmean(ab_mod, axis=1) / np.sqrt(sum(cre_i)) # 8 x 4
        
        
        ###### peak timing flash ######
        aa = svm_this_plane_allsess.iloc[cre_i]['peak_timing_flash_trTsShCh_this_plane_allsess_allp'].values 
        aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))])
        y_thisCre_allMice = np.transpose(aar, (1,0,2))
        ptf = np.nanmean(y_thisCre_allMice, axis=1)  # (8 x num_sessions) x 4    # compute mean across mice (of the same cre line) for each plane
        ptfs = np.nanstd(y_thisCre_allMice, axis=1) / np.sqrt(sum(cre_i))  # (8 x num_sessions) x 4    # compute std across mice (of the same cre line) for each plane
    
        # compute relative change in resp amp from A to B sessions.
        As = y_thisCre_allMice[:num_planes]
        Bs = y_thisCre_allMice[num_planes:]
        ab_mod = (Bs - As) / (Bs + As) # 8 x num_mice_this_cre x 4 
        # compute mean across mice (of the same cre line) for each plane     
        ptf_mod = np.nanmean(ab_mod, axis=1)  # 8 x 4
        ptf_mods = np.nanmean(ab_mod, axis=1) / np.sqrt(sum(cre_i)) # 8 x 4        
        
        
        depth_ave_allCre.append(d)     
        av_test_data_ave_allCre.append(ts)    
        av_test_shfl_ave_allCre.append(sh)
        peak_amp_omit_trTsShCh_ave_allCre.append(pa)    
        peak_timing_omit_trTsShCh_ave_allCre.append(pt)    
        peak_amp_flash_trTsShCh_ave_allCre.append(paf)    
        peak_timing_flash_trTsShCh_ave_allCre.append(ptf)

        peak_amp_mod_omit_trTsShCh_ave_allCre.append(pa_mod)    
        peak_timing_mod_omit_trTsShCh_ave_allCre.append(pt_mod)    
        peak_amp_mod_flash_trTsShCh_ave_allCre.append(paf_mod)    
        peak_timing_mod_flash_trTsShCh_ave_allCre.append(ptf_mod)

        depth_sd_allCre.append(ds)     
        av_test_data_sd_allCre.append(tss)    
        av_test_shfl_sd_allCre.append(shs)
        peak_amp_omit_trTsShCh_sd_allCre.append(pas)    
        peak_timing_omit_trTsShCh_sd_allCre.append(pts)    
        peak_amp_flash_trTsShCh_sd_allCre.append(pafs)    
        peak_timing_flash_trTsShCh_sd_allCre.append(ptfs)

        peak_amp_mod_omit_trTsShCh_sd_allCre.append(pa_mods)    
        peak_timing_mod_omit_trTsShCh_sd_allCre.append(pt_mods)    
        peak_amp_mod_flash_trTsShCh_sd_allCre.append(paf_mods)    
        peak_timing_mod_flash_trTsShCh_sd_allCre.append(ptf_mods)
        
    depth_ave_allCre = np.array(depth_ave_allCre) # num_cre x (8 x num_sessions)
    av_test_data_ave_allCre = np.array(av_test_data_ave_allCre) # num_cre x (8 x num_sessions) x num_frs
    av_test_shfl_ave_allCre = np.array(av_test_shfl_ave_allCre) # num_cre x (8 x num_sessions) x num_frs
    peak_amp_omit_trTsShCh_ave_allCre = np.array(peak_amp_omit_trTsShCh_ave_allCre) # num_cre x (8 x num_sessions) x 4
    peak_timing_omit_trTsShCh_ave_allCre = np.array(peak_timing_omit_trTsShCh_ave_allCre) # num_cre x (8 x num_sessions) x 4
    peak_amp_flash_trTsShCh_ave_allCre = np.array(peak_amp_flash_trTsShCh_ave_allCre) # num_cre x (8 x num_sessions) x 4
    peak_timing_flash_trTsShCh_ave_allCre = np.array(peak_timing_flash_trTsShCh_ave_allCre) # num_cre x (8 x num_sessions) x 4
            
    peak_amp_mod_omit_trTsShCh_ave_allCre = np.array(peak_amp_mod_omit_trTsShCh_ave_allCre) # num_cre x 8 x 4    
    peak_timing_mod_omit_trTsShCh_ave_allCre = np.array(peak_timing_mod_omit_trTsShCh_ave_allCre)    
    peak_amp_mod_flash_trTsShCh_ave_allCre = np.array(peak_amp_mod_flash_trTsShCh_ave_allCre)    
    peak_timing_mod_flash_trTsShCh_ave_allCre = np.array(peak_timing_mod_flash_trTsShCh_ave_allCre)

    depth_sd_allCre = np.array(depth_sd_allCre) # num_cre x (8 x num_sessions)
    av_test_data_sd_allCre = np.array(av_test_data_sd_allCre) # num_cre x (8 x num_sessions) x num_frs
    av_test_shfl_sd_allCre = np.array(av_test_shfl_sd_allCre) # num_cre x (8 x num_sessions) x num_frs
    peak_amp_omit_trTsShCh_sd_allCre = np.array(peak_amp_omit_trTsShCh_sd_allCre) # num_cre x (8 x num_sessions) x 4
    peak_timing_omit_trTsShCh_sd_allCre = np.array(peak_timing_omit_trTsShCh_sd_allCre) # num_cre x (8 x num_sessions) x 4
    peak_amp_flash_trTsShCh_sd_allCre = np.array(peak_amp_flash_trTsShCh_sd_allCre) # num_cre x (8 x num_sessions) x 4
    peak_timing_flash_trTsShCh_sd_allCre = np.array(peak_timing_flash_trTsShCh_sd_allCre) # num_cre x (8 x num_sessions) x 4
    
    peak_amp_mod_omit_trTsShCh_sd_allCre = np.array(peak_amp_mod_omit_trTsShCh_sd_allCre) # num_cre x 8 x 4        
    peak_timing_mod_omit_trTsShCh_sd_allCre = np.array(peak_timing_mod_omit_trTsShCh_sd_allCre)    
    peak_amp_mod_flash_trTsShCh_sd_allCre = np.array(peak_amp_mod_flash_trTsShCh_sd_allCre)    
    peak_timing_mod_flash_trTsShCh_sd_allCre = np.array(peak_timing_mod_flash_trTsShCh_sd_allCre)
    
    
    #%% Interpolate the traces every 3ms (instead of the original one that is 93ms)
    # (it doesnt really help ... as we are getting the time trace based on the frames of omittion aligned traces; also the flash/gray times are coming based on that... so it is nothing like we are adding more time resolution by interpolating!)
    

    
    
