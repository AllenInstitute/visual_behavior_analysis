"""
Set svm_allMice_sessPooled and svm_allMice_sessAvSd. (note: svm_allMice_sessPooled will be used to set summary_vars_all, which is a key paramter in svm_images_plots_compare_ophys_stages.py)

Run "svm_images_plots_setVars.py" to set vars needed here, mainly svm_this_plane_allsess.

This script sets vars that will be used for making summary plots across mice.

Follow this script by "svm_images_plots_setVars_sumMice2.py" to set summary_vars_all (which will be used in svm_images_plots_compare_ophys_stages.py), and make summary plots across mice (if len(project_codes_all)==1).


Note, this script is similar to part of omissions_traces_peaks_plots_setVars_ave.py, which is related to setting vars for making summary plots across mice.
Here, we set vars that are similar to the following vars in script omissions_traces_peaks_plots_setVars_ave.py:
    svm_allMice_sessPooled (similar to trace_peak_allMice_sessPooled): includes vars pooled across all sessions of all mice (for each plane, each area, and each depth); # use this to make summary mouse plots, by averaging across pooled sessions of all mice.
    svm_allMice_sessAvSd (similar to trace_peak_allMice_sessAvSd): includes vars pooled across all sessions of all mice (for each plane, each area, and each depth); # use this to make summary mouse plots, by averaging across mice (each mouse is here averaged across sessions).
    (note: some of the vars in "svm_allMice_sessAvSd" are also set in svm_plots_setVars.py (in var "svm_this_plane_allsess") . I didn't take them out of there, bc they will be used for making eachMouse plots. "svm_allMice_sessAvSd" is compatible with omissions code, and will be used for making sumMice plots.)
    
    
Created on Wed Oct 21 15:14:05 2020
@author: farzaneh
    
"""

print(svm_this_plane_allsess0.shape)

stages_all = (np.array([svm_this_plane_allsess0['session_labs'].values[i][0][0] for i in range(svm_this_plane_allsess0.shape[0])])).astype(int)
blocks_all = svm_this_plane_allsess0['block'].values # weird: 70 block 0s, and 69 block 1s ... you need to figure out why some experiments have failed!


#%% Loop through svm_this_plane_allsess0 and get those rows of it that belong to a given session stage and block; for each one set svm_allMice_sessPooled and svm_allMice_sessAvSd

####### Set svm_allMice_sessPooled #######
if project_codes != ['VisualBehaviorMultiscope']: # remove area/layer pooled columns
    indexes = ['cre_allPlanes', 'mouse_id_allPlanes', 'session_ids', 'area_allPlanes', 'depth_allPlanes', 'block_all', 'experience_levels', 'session_labs', \
               'av_test_data_allPlanes', 'av_test_shfl_allPlanes', 'peak_amp_allPlanes']
else:
    indexes = ['cre_allPlanes', 'mouse_id_allPlanes', 'session_ids', 'area_allPlanes', 'depth_allPlanes', 'block_all', 'experience_levels', 'session_labs', \
               'av_test_data_allPlanes', 'av_test_shfl_allPlanes', 'peak_amp_allPlanes', \
               'cre_eachArea', 'av_test_data_eachArea', 'av_test_shfl_eachArea', 'peak_amp_eachArea', \
               'cre_eachDepth', 'depth_eachDepth', 'av_test_data_eachDepth', 'av_test_shfl_eachDepth', 'peak_amp_eachDepth']

svm_allMice_sessPooled = pd.DataFrame([], columns=indexes)


####### Set svm_allMice_sessAvSd #######
if project_codes != ['VisualBehaviorMultiscope']: # remove area/layer pooled columns
    cols0 = ['mouse_id', 'cre', 'block', 'session_ids', 'session_stages', 'session_labs', 'area', 'depth', 'plane', \
            'av_depth_avSess_eachP', 'sd_depth_avSess_eachP', 'av_n_neurons_avSess_eachP', 'sd_n_neurons_avSess_eachP', \
            'av_n_trials_avSess_eachP', 'sd_n_trials_avSess_eachP', 'av_test_data_avSess_eachP', 'sd_test_data_avSess_eachP', 'av_test_shfl_avSess_eachP', 'sd_test_shfl_avSess_eachP', \
            'av_peak_amp_trTsShCh_avSess_eachP', 'sd_peak_amp_trTsShCh_avSess_eachP']

else:
    cols0 = ['mouse_id', 'cre', 'block', 'session_ids', 'session_stages', 'session_labs', 'area', 'depth', 'plane', \
            'av_depth_avSess_eachP', 'sd_depth_avSess_eachP', 'av_n_neurons_avSess_eachP', 'sd_n_neurons_avSess_eachP', \
            'av_n_trials_avSess_eachP', 'sd_n_trials_avSess_eachP', 'av_test_data_avSess_eachP', 'sd_test_data_avSess_eachP', 'av_test_shfl_avSess_eachP', 'sd_test_shfl_avSess_eachP', \
            'av_peak_amp_trTsShCh_avSess_eachP', 'sd_peak_amp_trTsShCh_avSess_eachP', \
            'num_sessLayers_valid_eachArea', 'av_test_data_pooledSessPlanes_eachArea', 'sd_test_data_pooledSessPlanes_eachArea', 'av_test_shfl_pooledSessPlanes_eachArea', 'sd_test_shfl_pooledSessPlanes_eachArea', \
            'av_pao_trTsShCh_pooledSessPlanes_eachArea', 'sd_pao_trTsShCh_pooledSessPlanes_eachArea', \
            'num_sessAreas_valid_eachDepth', 'depth_ave', \
            'av_test_data_pooledSessAreas_eachDepth', 'sd_test_data_pooledSessAreas_eachDepth', 'av_test_shfl_pooledSessAreas_eachDepth', 'sd_test_shfl_pooledSessAreas_eachDepth', \
            'av_pao_trTsShCh_pooledSessAreas_eachDepth', 'sd_pao_trTsShCh_pooledSessAreas_eachDepth']

if same_num_neuron_all_planes:
    cols = np.concatenate((cols0, ['av_n_neurons_svm_trained_avSess_eachP', 'sd_n_neurons_svm_trained_avSess_eachP'])) 
else:
    cols = cols0
svm_allMice_sessAvSd = pd.DataFrame([], columns=cols)


if ~np.isnan(svm_blocks) and svm_blocks!=-101:
    br = np.unique(blocks_all)
else:
    br = [np.nan]
    
    
    
    
########################################################
cntall = 0
cntall2 = 0

for istage in np.unique(stages_all): # istage=1
    for iblock in br: # iblock=0 ; iblock=np.nan
        
        if ~np.isnan(svm_blocks) and svm_blocks!=-101: # block by block analysis
            svm_this_plane_allsess = svm_this_plane_allsess0[np.logical_and(stages_all==istage , blocks_all==iblock)]
        else:
            svm_this_plane_allsess = svm_this_plane_allsess0[stages_all==istage]
            
            
        cntall = cntall + 1
        
        #%% ##########################################################################################
        ############### Set svm_allMice_sessPooled to make average plots of pooled data across mice sessions #############
        ################ (each plane individually, no pooling across areas or layers) ###########################
        ##############################################################################################

        # Remember each row of svm_this_plane_allsess is for each mouse; below we combine data from all mice

        #%% Pool all sessions of all mice
        # pandas series "svm_allMice_sessPooled" is created that includes all the important vars below.
        # use this to make summary mouse plots, by averaging across pooled sessions of all mice.

        a0 = np.concatenate((svm_this_plane_allsess['cre_exp']).values) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
        cre_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        


        a0 = np.concatenate((svm_this_plane_allsess['mouse_id_exp']).values) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
        mouse_id_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        

        
        a0 = np.concatenate((svm_this_plane_allsess['session_ids']).values)
        session_ids_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        
        
        
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
        block_all = np.unique(svm_this_plane_allsess['block'].values)
        
        experience_levels_all = np.concatenate((svm_this_plane_allsess['experience_levels']).values) # num_sessions
#         experience_levels_all = np.unique(np.concatenate((svm_this_plane_allsess['experience_levels']).values))
        
        print(session_labs_all, np.unique(experience_levels_all))
        

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

        aa = svm_this_plane_allsess['peak_amp_trTsShCh_this_plane_allsess_allp'].values # num_all_mice; # size of each element : 8 x num_sessions x 4
        aar = np.array([np.reshape(aa[ic], (aa[ic].shape[0] * aa[ic].shape[1] , aa[ic].shape[2]), order='F') for ic in range(len(aa))]) # aar.shape # num_all_mice; # size of each element : (8 x num_sessions) x 4
        a0 = np.concatenate((aar)) # (8*sum(num_sess_per_mouse)) x 4 # 8 planes of 1 session, then 8 planes of next session, and so on 
        pa_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes), a0.shape[-1]), order='F') # 8 x sum(num_sess_per_mouse) x 4 # each row is one plane, all sessions        


        
        if project_codes == ['VisualBehaviorMultiscope']:
            ########## Each area: layers pooled ########## 

            cre_all_eachArea = np.concatenate((svm_this_plane_allsess['cre_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse))
            ts_all_eachArea = np.concatenate((svm_this_plane_allsess['av_test_data_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse)) x 1270
            sh_all_eachArea = np.concatenate((svm_this_plane_allsess['av_test_shfl_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse)) x 1270    
            pao_all_eachArea = np.concatenate((svm_this_plane_allsess['peak_amp_trTsShCh_pooled_sesss_planes_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse)) x 4(train,test,shuffle,chance)


            ########## Each layer: areas pooled ########## svm_this_plane_allsess['trace_pooled_eachDepth']

            cre_all_eachDepth = np.concatenate((svm_this_plane_allsess['cre_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))
            ts_all_eachDepth = np.concatenate((svm_this_plane_allsess['av_test_data_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse)) x 1270
            sh_all_eachDepth = np.concatenate((svm_this_plane_allsess['av_test_shfl_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse)) x 1270    
            pao_all_eachDepth = np.concatenate((svm_this_plane_allsess['peak_amp_trTsShCh_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse)) x 4(train,test,shuffle,chance)

            depth_all_eachDepth = np.concatenate((svm_this_plane_allsess['depth_pooled_sesss_areas_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))    


            '''        
            ########### Make a pandas series to keep the pooled session data of all mice ########### 

            svm_allMice_sessPooled = pd.Series([cre_all, mouse_id_all, area_all, depth_all, block_all, session_labs_all, \
               ts_all, sh_all, pa_all, \
               cre_all_eachArea, ts_all_eachArea, sh_all_eachArea, pao_all_eachArea, \
               cre_all_eachDepth, depth_all_eachDepth, ts_all_eachDepth, sh_all_eachDepth, pao_all_eachDepth],\
               index=indexes)
            '''

            svm_allMice_sessPooled.at[cntall, indexes] = cre_all, mouse_id_all, session_ids_all, area_all, depth_all, block_all, experience_levels_all, session_labs_all, \
               ts_all, sh_all, pa_all, \
               cre_all_eachArea, ts_all_eachArea, sh_all_eachArea, pao_all_eachArea, \
               cre_all_eachDepth, depth_all_eachDepth, ts_all_eachDepth, sh_all_eachDepth, pao_all_eachDepth

        else:

            arrvars = cre_all, mouse_id_all, session_ids_all, area_all, depth_all, block_all, experience_levels_all, session_labs_all, ts_all, sh_all, pa_all
            for iar in range(len(indexes)):
                svm_allMice_sessPooled.at[cntall, indexes[iar]] = arrvars[iar]
                
            # not sure why but below fails...
#             svm_allMice_sessPooled.at[cntall, indexes] = cre_all, mouse_id_all, session_ids_all, \
#                 area_all, depth_all, block_all, session_labs_all, ts_all, sh_all, pa_all
            

        print(svm_allMice_sessPooled.shape)
#         print(svm_allMice_sessPooled)




        ######################################################################
        ######################################################################
        ######################################################################
        #%% Set svm_allMice_sessAvSd: session-averaged data for each mouse; note we dont use svm_allMice_sessAvSd in svm_images_plots_compare_ophys_stages.py
        # pandas dataFrame "svm_allMice_sessAvSd" is created to keep the important vars below
        # use this to make summary mouse plots, by averaging across mice (each mouse is here averaged across sessions).
        ######################################################################
        ######################################################################
        ######################################################################

        for im in range(len(svm_this_plane_allsess)): # im=0

            cntall2 = cntall2 + 1
            
            #%%
            mouse_id = svm_this_plane_allsess.iloc[im]['mouse_id']
            cre = svm_this_plane_allsess.iloc[im]['cre']
            session_stages = svm_this_plane_allsess.iloc[im]['session_stages']
            session_labs = svm_this_plane_allsess.iloc[im]['session_labs']
            num_sessions = len(session_stages)
            block = svm_this_plane_allsess.iloc[im]['block']
            session_ids_now = svm_this_plane_allsess.iloc[im]['session_ids']

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
            a = svm_this_plane_allsess.iloc[im]['n_trials_this_plane_allsess_allp'] # size: (8 x num_sessions x 1)
            av_n_trials_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
            sd_n_trials_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane


            if project_codes != ['VisualBehaviorMultiscope']:
                sqnv = np.sqrt(num_sessions_valid)
            else:
                sqnv = np.sqrt(num_sessions_valid)[:, np.newaxis]
            
            # av_test_data
            a = svm_this_plane_allsess.iloc[im]['av_test_data_this_plane_allsess_allp'] # size: (8 x num_sessions x nFrames_upsampled)
            av_test_data_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
            sd_test_data_avSess_eachP = np.nanstd(a,axis=1) / sqnv # num_planes x nFrames_upsampled # st error across sessions, for each plane

            # av_test_shfl
            a = svm_this_plane_allsess.iloc[im]['av_test_shfl_this_plane_allsess_allp'] # size: (8 x num_sessions x nFrames_upsampled)
            av_test_shfl_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
            sd_test_shfl_avSess_eachP = np.nanstd(a,axis=1) / sqnv # num_planes x nFrames_upsampled # st error across sessions, for each plane


            # peak_amp_trTsShCh
            a = svm_this_plane_allsess.iloc[im]['peak_amp_trTsShCh_this_plane_allsess_allp'] # size: (8 x num_sessions x 4)
            av_peak_amp_trTsShCh_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes x 4
            sd_peak_amp_trTsShCh_avSess_eachP = np.nanstd(a,axis=1).squeeze() / sqnv # num_planes x 4 # st error across sessions, for each plane


            if project_codes == ['VisualBehaviorMultiscope']:
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
                a = svm_this_plane_allsess.iloc[im]['peak_amp_trTsShCh_pooled_sesss_planes_eachArea'] # 2 x (4 x num_sessions) x 4(train, test, shuffle, chance)
            #        num_sessLayers_valid_eachArea = a.shape[1]
                av_pao_trTsShCh_pooledSessPlanes_eachArea = np.nanmean(a, axis=1) # 2 x 4 
                sd_pao_trTsShCh_pooledSessPlanes_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea)[:,np.newaxis] # 2 x 4(train, test, shuffle, chance) 


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
                a = svm_this_plane_allsess.iloc[im]['peak_amp_trTsShCh_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions) x 4(train, test, shuffle, chance)
            #        num_sessLayers_valid_eachArea = a.shape[1]
                av_pao_trTsShCh_pooledSessAreas_eachDepth = np.nanmean(a, axis=1) # 4 x 4 
                sd_pao_trTsShCh_pooledSessAreas_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth)[:,np.newaxis] # 4 x 4(train, test, shuffle, chance) 


                #%% Keep vars (session averaged, each plane, also eachArea, eachDepth vars) for all mice

                svm_allMice_sessAvSd.at[cntall2, cols0] = \
                    mouse_id, cre, block, session_ids_now, session_stages, session_labs, areas, depths, planes, \
                    av_depth_avSess_eachP, sd_depth_avSess_eachP, av_n_neurons_avSess_eachP, sd_n_neurons_avSess_eachP, \
                    av_n_trials_avSess_eachP, sd_n_trials_avSess_eachP, av_test_data_avSess_eachP, sd_test_data_avSess_eachP, av_test_shfl_avSess_eachP, sd_test_shfl_avSess_eachP, \
                    av_peak_amp_trTsShCh_avSess_eachP, sd_peak_amp_trTsShCh_avSess_eachP, \
                    num_sessLayers_valid_eachArea, av_test_data_pooledSessPlanes_eachArea, sd_test_data_pooledSessPlanes_eachArea, av_test_shfl_pooledSessPlanes_eachArea, sd_test_shfl_pooledSessPlanes_eachArea, \
                    av_pao_trTsShCh_pooledSessPlanes_eachArea, sd_pao_trTsShCh_pooledSessPlanes_eachArea, \
                    num_sessAreas_valid_eachDepth, depth_ave, \
                    av_test_data_pooledSessAreas_eachDepth, sd_test_data_pooledSessAreas_eachDepth, av_test_shfl_pooledSessAreas_eachDepth, sd_test_shfl_pooledSessAreas_eachDepth, \
                    av_pao_trTsShCh_pooledSessAreas_eachDepth, sd_pao_trTsShCh_pooledSessAreas_eachDepth

            else:
                
                #%% Keep vars (session averaged, each plane, also eachArea, eachDepth vars) for all mice

                svm_allMice_sessAvSd.at[cntall2, cols0] = \
                    mouse_id, cre, block, session_ids_now, session_stages, session_labs, areas, depths, planes, \
                    av_depth_avSess_eachP, sd_depth_avSess_eachP, av_n_neurons_avSess_eachP, sd_n_neurons_avSess_eachP, \
                    av_n_trials_avSess_eachP, sd_n_trials_avSess_eachP, av_test_data_avSess_eachP, sd_test_data_avSess_eachP, av_test_shfl_avSess_eachP, sd_test_shfl_avSess_eachP, \
                    av_peak_amp_trTsShCh_avSess_eachP, sd_peak_amp_trTsShCh_avSess_eachP                
                
                
            if same_num_neuron_all_planes:
                svm_allMice_sessAvSd.at[cntall2, ['av_n_neurons_svm_trained_avSess_eachP']] = [av_n_neurons_svm_trained_avSess_eachP]
                svm_allMice_sessAvSd.at[cntall2, ['sd_n_neurons_svm_trained_avSess_eachP']] = [sd_n_neurons_svm_trained_avSess_eachP]

        print(svm_allMice_sessAvSd.shape)
#         svm_allMice_sessAvSd



