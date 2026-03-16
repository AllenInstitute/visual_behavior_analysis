"""
Gets called in svm_images_plots_setVars.py

Sets summary_vars_all, a dataframe that includes response amplitude (computed from svm_allMice_sessPooled); it will be used in svm_images_plots_compare_ophys_stages.py

If len(project_codes_all)==1, it calls svm_images_plots_sumMice.py to make mouse-averaged plots for each ophys stage.

Vars needed here are set in "svm_images_plots_setVars_sumMice.py"


Created on Wed Oct 21 15:22:05 2020
@author: farzaneh

"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import seaborn
import sys

from general_funs import *

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)        


#%%
stages_allp = (np.array([svm_allMice_sessPooled0['session_labs'].values[i][0][0] for i in range(svm_allMice_sessPooled0.shape[0])])).astype(int)
if ~np.isnan(svm_blocks) and svm_blocks!=-101:
    blocks_allp = (np.array([svm_allMice_sessPooled0['block_all'].values[i][0] for i in range(svm_allMice_sessPooled0.shape[0])])).astype(int)
else:
    blocks_allp = (np.array([svm_allMice_sessPooled0['block_all'].values[i][0] for i in range(svm_allMice_sessPooled0.shape[0])]))
    
    
stages_alla = (np.array([svm_allMice_sessAvSd0['session_labs'].values[i][0][0] for i in range(svm_allMice_sessAvSd0.shape[0])])).astype(int)
blocks_alla = svm_allMice_sessAvSd0['block'].values

cre_all = svm_allMice_sessPooled0['cre_allPlanes'].values[0]
cre_lines = np.unique(cre_all)
cre_lineso = copy.deepcopy(cre_lines)


if ~np.isnan(svm_blocks) and svm_blocks!=-101:
    if svm_blocks==-1: # divide trials into blocks based on the engagement state
        svmb = 2
    else:
        svmb = svm_blocks
    br = range(svmb)    
else:
    br = [np.nan]
    

if baseline_subtract: # subtract the baseline (CA average during baseline, ie before time 0) from the evoked CA (classification accuracy)
    ylabs = '% Class accuracy rel. baseline' #'Amplitude'
else:
    ylabs = '% Classification accuracy' #'Amplitude'    
    
    
    
#%% Make sure for each mouse we have data from both blocks
'''
for istage in np.unique(stages_all): # istage=1
    for iblock in br: # iblock=0 ; iblock=np.nan
        if ~np.isnan(svm_blocks):
            svm_this_plane_allsess = svm_this_plane_allsess0[np.logical_and(stages_all==istage , blocks_all==iblock)]
        else:
            svm_this_plane_allsess = svm_this_plane_allsess0[stages_all==istage]
'''

if ~np.isnan(svm_blocks) and svm_blocks!=-101: # svm was run on blocks 

    for istage in np.unique(stages_all): # istage=1
        a = svm_allMice_sessPooled0[stages_allp==istage]
        if len(a) != len(br):
            print(f'data from both blocks dont exist for stage {istage}!')
#             sys.exit(f'data from both blocks dont exist for stage {istage}!')

        else:
            mice_blocks = []
            for iblock in br:
                b = np.unique(a[a['block_all']==iblock]['mouse_id_allPlanes'].values[0])
                mice_blocks.append(b)

            mice_blocks = np.array(mice_blocks)
            
            d0 = np.setdiff1d(mice_blocks[0], mice_blocks[1]) # mouse exists in block0 but not block1
            d1 = np.setdiff1d(mice_blocks[1], mice_blocks[0]) # mouse exists in block1 but not block0
            
            if len(d0)>0: # ~np.equal(mice_blocks[0], mice_blocks[1]).all()
                print(f'Stage {istage}, mouse {d0} is missing from block 1!')                
                print('\tsetting this mouse values in block0 to nan')
                this_stage_this_block = a.iloc[0] # block 0
                this_mouse_ind = this_stage_this_block['mouse_id_allPlanes'][0,:]==d0 # mouse_id for each session (of plane 0)

#                 aa = this_stage_this_block['av_test_data_allPlanes'] # num_planes x num_sessions x num_frames
#                 aa[:, this_mouse_ind,:] = np.nan
#                 aa = this_stage_this_block['av_test_shfl_allPlanes'] # num_planes x num_sessions x num_frames
#                 aa[:, this_mouse_ind,:] = np.nan
                
                svm_allMice_sessPooled0[stages_allp==istage].iloc[0]['av_test_data_allPlanes'][:, this_mouse_ind,:] = np.nan
                svm_allMice_sessPooled0[stages_allp==istage].iloc[0]['av_test_shfl_allPlanes'][:, this_mouse_ind,:] = np.nan
                
                # remember you also need to do the same thing for svm_allMice_sessAvSd0, but we are not using it below so i'll pass for now!
                
            if len(d1)>0:
                print(f'Stage {istage}, mouse {d1} is missing from block 0!')
                print('\tsetting this mouse values in block1 to nan')
                this_stage_this_block = a.iloc[1] # block 1
                this_mouse_ind = this_stage_this_block['mouse_id_allPlanes'][0,:]==d0 # mouse_id for each session (of plane 0)

#                 aa = this_stage_this_block['av_test_data_allPlanes'] # num_planes x num_sessions x num_frames
#                 aa[:, this_mouse_ind,:] = np.nan
#                 aa = this_stage_this_block['av_test_shfl_allPlanes'] # num_planes x num_sessions x num_frames
#                 aa[:, this_mouse_ind,:] = np.nan

                svm_allMice_sessPooled0[stages_allp==istage].iloc[1]['av_test_data_allPlanes'][:, this_mouse_ind,:] = np.nan
                svm_allMice_sessPooled0[stages_allp==istage].iloc[1]['av_test_shfl_allPlanes'][:, this_mouse_ind,:] = np.nan
                
            if len(br)>2:
                sys.exit('above needs work; setdiff1d works on 2 arrays; perhaps loop through arrays!')


                
##########################################################################################
##########################################################################################
#%% SUMMARY ACROSS MICE 
##########################################################################################
##########################################################################################
############# Set summary_vars_all, a dataframe that includes response amplitude (computed from svm_allMice_sessPooled); it will be used in svm_images_plots_compare_ophys_stages.py
############# Also plot traces averaged across mice of the same cre line ################
##########################################################################################    
##########################################################################################

cols_sum = ['stage', 'experience_level', 'cre', 'block', 'depth_ave', 'resp_amp']
summary_vars_all = pd.DataFrame([], columns=cols_sum)

xgap_areas = .3
linestyle_all = ['solid', 'dashed'] # for block 0 and 1
fmt_all = ['o', 'x']
cols_area_all = [cols_area , ['skyblue', 'dimgray']]

cntn = 0
for istage in np.unique(stages_all): # istage=1
    for cre in ['Slc17a7', 'Sst', 'Vip']: # cre = 'Slc17a7'
        
        ###############################         
        plt.figure(figsize=(8,11))  

        ##### traces ##### 
        gs1 = gridspec.GridSpec(1,2) #(3, 2)#, width_ratios=[3, 1]) 
        gs1.update(bottom=.73, top=0.9, left=0.05, right=0.95, wspace=.8)
        #
        gs2 = gridspec.GridSpec(1,2) #(3, 2)#, width_ratios=[3, 1]) 
        gs2.update(bottom=.45, top=0.62, left=0.05, right=0.95, wspace=.8)    

        ##### Response measures; each depth ##### 
        # subplots of flash responses
        gs3 = gridspec.GridSpec(1,2) #, width_ratios=[3, 1]) 
        gs3.update(bottom=.22, top=0.35, left=0.05, right=0.55, wspace=.55, hspace=.5)

        ##### Response measures; compare areas (pooled depths) ##### 
        # subplots of flash responses
        gs5 = gridspec.GridSpec(1,2) #, width_ratios=[3, 1]) 
        gs5.update(bottom=.22, top=0.35, left=0.7, right=0.95, wspace=.55, hspace=.5)

#         plt.suptitle('%s, %d mice, %d total sessions: %s' %(cre, thisCre_mice_num, thisCre_pooledSessNum, session_labs_all), fontsize=14)


        #%%
        h1v1 = []
        h1lm = []
        h1a = []
        h1d = []
        lims_v1lm = []
        
        for iblock in br: # np.unique(blocks_all): # iblock=0 ; iblock=np.nan

            if ~np.isnan(svm_blocks) and svm_blocks!=-101:
                svm_allMice_sessPooled = svm_allMice_sessPooled0[np.logical_and(stages_allp==istage , blocks_allp==iblock)]
                svm_allMice_sessAvSd = svm_allMice_sessAvSd0[np.logical_and(stages_alla==istage , blocks_alla==iblock)]
    #             svm_allMice_sessPooled = eval(f'svm_allMice_sessPooled_block{iblock}')
    #             svm_allMice_sessAvSd = eval(f'svm_allMice_sessAvSd_block{iblock}')
    
                linestyle_now = linestyle_all[iblock]
                cols_area_now = cols_area_all[iblock]
                fmt_now = fmt_all[iblock]
                
            else:
                svm_allMice_sessPooled = svm_allMice_sessPooled0[stages_allp==istage]
                svm_allMice_sessAvSd = svm_allMice_sessAvSd0[stages_alla==istage]
                
                linestyle_now = linestyle_all[0]
                cols_area_now = cols_area_all[0]
                fmt_now = fmt_all[0]
                

            ###############################
            if len(svm_allMice_sessPooled)==0: # eg when dividing blocks by engagement metric (svm_blocks=-1), we wont have data for passive sessions.
                thisCre_pooledSessNum = 0
                print(f'\nThere is no data for {cre}, stage {istage} block {iblock}!\n')
                
            else: # there is data

                cre_all = svm_allMice_sessPooled['cre_allPlanes'].values[0]
                cre_lines = np.unique(cre_all)
                mouse_id_all = svm_allMice_sessPooled['mouse_id_allPlanes'].values[0]

    #             cre = 'Sst' #cre_lines[icre]    
                thisCre_pooledSessNum = sum(cre_all[0,:]==cre) # number of pooled sessions for this cre line
                thisCre_mice = mouse_id_all[0, cre_all[0,:]==cre]
                thisCre_mice_num = len(np.unique(thisCre_mice))


                if svm_blocks==-1:
                    lab_b = f'e{iblock}' # engagement
                else:
                    lab_b = f'b{iblock}' # block


                if thisCre_pooledSessNum < 1:
                    print(f'There are no sessions for stage {istage}, mouse {cre}, block {iblock}')

                else:
                    ###############################
                    #%% Set vars
                    a_all = svm_allMice_sessPooled['area_allPlanes'].values[0]  # 8 x pooledSessNum 
                    d_all = svm_allMice_sessPooled['depth_allPlanes'].values[0]  # 8 x pooledSessNum 
                    session_labs_all = svm_allMice_sessPooled['session_labs'].values[0]    

                    ts_all = svm_allMice_sessPooled['av_test_data_allPlanes'].values[0]  # 8 x pooledSessNum x 80
                    sh_all = svm_allMice_sessPooled['av_test_shfl_allPlanes'].values[0]  # 8 x pooledSessNum x 80

                    pa_all = svm_allMice_sessPooled['peak_amp_allPlanes'].values[0]  # 8 x pooledSessNum x 4 (trTsShCh)


                    if project_codes == ['VisualBehaviorMultiscope']:
                        cre_eachArea = svm_allMice_sessPooled['cre_eachArea'].values[0] # 2 x (4*sum(num_sess_per_mouse))
                        ts_eachArea = svm_allMice_sessPooled['av_test_data_eachArea'].values[0] # 2 x (4*sum(num_sess_per_mouse)) x 80
                        sh_eachArea = svm_allMice_sessPooled['av_test_shfl_eachArea'].values[0] # 2 x (4*sum(num_sess_per_mouse)) x 80    
                        pa_eachArea = svm_allMice_sessPooled['peak_amp_eachArea'].values[0] # 2 x (4*sum(num_sess_per_mouse)) x 4 (trTsShCh)

                        cre_eachDepth = svm_allMice_sessPooled['cre_eachDepth'].values[0] # 4 x (2*sum(num_sess_per_mouse))
                        depth_eachDepth = svm_allMice_sessPooled['depth_eachDepth'].values[0] # 4 x (2*sum(num_sess_per_mouse))
                        ts_eachDepth = svm_allMice_sessPooled['av_test_data_eachDepth'].values[0] # 4 x (2*sum(num_sess_per_mouse)) x 80
                        sh_eachDepth = svm_allMice_sessPooled['av_test_shfl_eachDepth'].values[0] # 4 x (2*sum(num_sess_per_mouse)) x 80    

                        depth_ave = np.mean(depth_eachDepth[:, cre_eachDepth[0,:]==cre], axis=1).astype(float) # 4 # average across areas and sessions

                    else:
                        depth_ave = np.mean(d_all[:,cre_all[0,:]==cre], axis=1).astype(float) # 4 # average across areas and sessions


                    #################################
                    #%% Keep summary vars for all ophys stages

                    cntn = cntn+1
    #                 depth_ave = np.mean(depth_eachDepth[:, cre_eachDepth[0,:]==cre], axis=1).astype(float) # 4 # average across areas and sessions
                    pallnow = pa_all[:, cre_all[0,:]==cre]

                    summary_vars_all.at[cntn, cols_sum] = istage, exp_level, cre, iblock, depth_ave, pallnow



                    ####################################################################################
                    ####################################################################################                
                    #%% Plot session-averaged traces for each plane; 2 subplots: 1 per area
                    ####################################################################################
                    ####################################################################################

                    #%%
                    # Average across all pooled sessions for each plane # remember some experiments might be nan (because they were not valid, we keep all 8 experiments to make indexing easy)
                    a = ts_all[:, cre_all[0,:]==cre]  # 8 x thisCre_pooledSessNum x 80
                    av_ts_eachPlane = np.nanmean(a, axis=1) # 8 x 80
                    sd_ts_eachPlane = np.nanstd(a, axis=1) / np.sqrt(a.shape[1]) # 8 x 80

                    a = sh_all[:, cre_all[0,:]==cre]  # 8 x thisCre_pooledSessNum x 80
                    av_sh_eachPlane = np.nanmean(a, axis=1) # 8 x 80
                    sd_sh_eachPlane = np.nanstd(a, axis=1) / np.sqrt(a.shape[1]) # 8 x 80

                    areas = a_all[:, cre_all[0,:]==cre]  # 8 x thisCre_pooledSessNum
                    depths = d_all[:, cre_all[0,:]==cre]  # 8 x thisCre_pooledSessNum

                    lims = np.array([np.nanmin(av_sh_eachPlane - sd_sh_eachPlane), np.nanmax(av_ts_eachPlane + sd_ts_eachPlane)])
                    lims[0] = lims[0] - np.diff(lims) / 20.
                    lims[1] = lims[1] + np.diff(lims) / 20.

                    lims_v1lm.append(lims)



                    ################################################################
                    ##############%% Plot averages of all mice in each cell line ###################
                    ################################################################

                    plots_make_finalize = [1,0] # make the plot

                    if len(project_codes_all)==1:
                        exec(open('svm_images_plots_sumMice.py').read())



        ################################################################
        ################################################################            
        if thisCre_pooledSessNum > 0: # done with both blocks of a given ophys stage; now add labels etc to the plot
            
            plots_make_finalize = [0,1]

            if len(project_codes_all)==1:
                exec(open('svm_images_plots_sumMice.py').read())
            
            


            
            
            
            
            
