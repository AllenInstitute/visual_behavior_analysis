"""
Vars needed here are set in "svm_images_plots_setVars_sumMice.py"
Make summary plots across mice (each cre line) for the svm analysis.

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
if ~np.isnan(svm_blocks):
    blocks_allp = (np.array([svm_allMice_sessPooled0['block_all'].values[i][0] for i in range(svm_allMice_sessPooled0.shape[0])])).astype(int)
else:
    blocks_allp = (np.array([svm_allMice_sessPooled0['block_all'].values[i][0] for i in range(svm_allMice_sessPooled0.shape[0])]))
    
    
stages_alla = (np.array([svm_allMice_sessAvSd0['session_labs'].values[i][0][0] for i in range(svm_allMice_sessAvSd0.shape[0])])).astype(int)
blocks_alla = svm_allMice_sessAvSd0['block'].values

cre_all = svm_allMice_sessPooled0['cre_allPlanes'].values[0]
cre_lines = np.unique(cre_all)
cre_lineso = copy.deepcopy(cre_lines)


if ~np.isnan(svm_blocks):
    if svm_blocks==-1: # divide trials into blocks based on the engagement state
        svmb = 2
    else:
        svmb = svm_blocks

    br = range(svmb)    

else:
    br = [np.nan]
    

#%% Make sure for each mouse we have data from both blocks
'''
for istage in np.unique(stages_all): # istage=1
    for iblock in br: # iblock=0 ; iblock=np.nan
        if ~np.isnan(svm_blocks):
            svm_this_plane_allsess = svm_this_plane_allsess0[np.logical_and(stages_all==istage , blocks_all==iblock)]
        else:
            svm_this_plane_allsess = svm_this_plane_allsess0[stages_all==istage]
'''

if ~np.isnan(svm_blocks):

    for istage in np.unique(stages_all): # istage=4
        a = svm_allMice_sessPooled0[stages_allp==istage]
        if len(a) != svm_blocks:
            print(f'data from both blocks dont exist for stage {istage}!')
#             sys.exit(f'data from both blocks dont exist for stage {istage}!')

        else:
            mice_blocks = []
            for iblock in range(svm_blocks):
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
                
            if svm_blocks>2:
                sys.exit('above needs work; setdiff1d works on 2 arrays; perhaps loop through arrays!')


                

#%% SUMMARY ACROSS MICE 
##########################################################################################
##########################################################################################
############# Plot traces averaged across mice of the same cre line ################
##########################################################################################    
##########################################################################################

xgap_areas = .3
linestyle_all = ['solid', 'dashed'] # for block 0 and 1
fmt_all = ['o', 'x']
cols_area_all = [cols_area , ['skyblue', 'dimgray']]

for istage in np.unique(stages_all): # istage=1

    #%% Plot averages of all mice in each cell line

    for cre in ['Slc17a7', 'Sst', 'Vip']: # cre = 'Slc17a7'
#     for icre in range(len(cre_lines)): # icre=0
        
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

            if ~np.isnan(svm_blocks):
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

                cre_eachArea = svm_allMice_sessPooled['cre_eachArea'].values[0] # 2 x (4*sum(num_sess_per_mouse))
                ts_eachArea = svm_allMice_sessPooled['av_test_data_eachArea'].values[0] # 2 x (4*sum(num_sess_per_mouse)) x 80
                sh_eachArea = svm_allMice_sessPooled['av_test_shfl_eachArea'].values[0] # 2 x (4*sum(num_sess_per_mouse)) x 80    
                pa_eachArea = svm_allMice_sessPooled['peak_amp_eachArea'].values[0] # 2 x (4*sum(num_sess_per_mouse)) x 4 (trTsShCh)

                cre_eachDepth = svm_allMice_sessPooled['cre_eachDepth'].values[0] # 4 x (2*sum(num_sess_per_mouse))
                depth_eachDepth = svm_allMice_sessPooled['depth_eachDepth'].values[0] # 4 x (2*sum(num_sess_per_mouse))
                ts_eachDepth = svm_allMice_sessPooled['av_test_data_eachDepth'].values[0] # 4 x (2*sum(num_sess_per_mouse)) x 80
                sh_eachDepth = svm_allMice_sessPooled['av_test_shfl_eachDepth'].values[0] # 4 x (2*sum(num_sess_per_mouse)) x 80    


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
                ############################### V1 ###############################
    #             h1v1 = []            
                for iplane in inds_v1: # iplane = inds_v1[0]

                    ax = plt.subplot(gs1[0]) # plt.subplot(2,1,2)

                    area = areas[iplane] # num_sess 
                    depth = depths[iplane] # num_sess
                    lab = '%dum' %(np.mean(depth))
                    if ~np.isnan(svm_blocks):
                        lab = f'{lab},{lab_b}'

                    h1_0 = plt.plot(time_trace, av_ts_eachPlane[iplane], color=cols_depth[iplane-num_planes], label=(lab), markersize=3.5, linestyle=linestyle_now)[0]
                    h2 = plt.plot(time_trace, av_sh_eachPlane[iplane], color='gray', label='shfl', markersize=3.5)[0]

                    # errorbars (standard error across cv samples)   
                    '''
                    plt.fill_between(time_trace, av_ts_eachPlane[iplane] - sd_ts_eachPlane[iplane], av_ts_eachPlane[iplane] + sd_ts_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane-num_planes], facecolor=cols_depth[iplane-num_planes])
                    plt.fill_between(time_trace, av_sh_eachPlane[iplane] - sd_sh_eachPlane[iplane], av_sh_eachPlane[iplane] + sd_sh_eachPlane[iplane], alpha=alph, edgecolor='gray', facecolor='gray')
                    '''
                    h1v1.append(h1_0)



                ################################################################
                ############################### LM ###############################
    #             h1lm = []
                for iplane in inds_lm: #range(4): # iplane = 0
                    ax = plt.subplot(gs1[1]) # plt.subplot(2,1,1)

                    area = areas[iplane] # num_sess 
                    depth = depths[iplane] # num_sess             
                    lab = '%dum' %(np.mean(depth))
                    if ~np.isnan(svm_blocks):
                        lab = f'{lab},{lab_b}'

                    h1_0 = plt.plot(time_trace, av_ts_eachPlane[iplane], color=cols_depth[iplane], label=(lab), markersize=3.5, linestyle=linestyle_now)[0]            
                    h2 = plt.plot(time_trace, av_sh_eachPlane[iplane], color='gray', label='shfl', markersize=3.5)[0]            

                    # errorbars (standard error across cv samples)   
                    '''
                    plt.fill_between(time_trace, av_ts_eachPlane[iplane] - sd_ts_eachPlane[iplane], av_ts_eachPlane[iplane] + sd_ts_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane], facecolor=cols_depth[iplane])            
                    plt.fill_between(time_trace, av_sh_eachPlane[iplane] - sd_sh_eachPlane[iplane], av_sh_eachPlane[iplane] + sd_sh_eachPlane[iplane], alpha=alph, edgecolor='gray', facecolor='gray')
                    '''
                    h1lm.append(h1_0)



                ################################################################
                #%% Plot traces per area (pooled across sessions and layers for each area)
                # Question 1: is there a difference between areas (V1 vs LM) # for each mouse pool across sessions and layers (for each area)
                ################################################################

                # trace: average across all pooled sessions for each plane    
                a = ts_eachArea[:, cre_eachArea[0,:]==cre]  # 2 x (4*thisCre_pooledSessNum) x 80
                av_ts_pooled_eachArea = np.nanmean(a, axis=1) # 2 x 80
                sd_ts_pooled_eachArea = np.nanstd(a, axis=1) / np.sqrt(a.shape[1]) # 2 x 80

                a = sh_eachArea[:, cre_eachArea[0,:]==cre]  # 2 x (4*thisCre_pooledSessNum) x 80
                av_sh_pooled_eachArea = np.nanmean(a, axis=1) # 2 x 80
                sd_sh_pooled_eachArea = np.nanstd(a, axis=1) / np.sqrt(a.shape[1]) # 2 x 80

                ##################
                ax = plt.subplot(gs2[0])
                #    ax = plt.subplot(gs1[2,0])
                plt.title('Sessions and layers pooled', fontsize=13, y=1)

    #             h1a = []
                for iarea in [1,0]: # first plot V1 then LM  #range(a.shape[0]):
                    lab = '%s' %(distinct_areas[iarea])
                    if ~np.isnan(svm_blocks):
                        lab = f'{lab},{lab_b}'

                    h1_0 = plt.plot(time_trace, av_ts_pooled_eachArea[iarea], color = cols_area_now[iarea], label=(lab), markersize=3.5, linestyle=linestyle_now)[0]        
                    h2 = plt.plot(time_trace, av_sh_pooled_eachArea[iarea], color='gray', label='shfl', markersize=3.5)[0]

                    '''
                    plt.fill_between(time_trace, av_ts_pooled_eachArea[iarea] - sd_ts_pooled_eachArea[iarea] , \
                                     av_ts_pooled_eachArea[iarea] + sd_ts_pooled_eachArea[iarea], alpha=alph, edgecolor=cols_area_now[iarea], facecolor=cols_area_now[iarea])
                    plt.fill_between(time_trace, av_sh_pooled_eachArea[iarea] - sd_sh_pooled_eachArea[iarea] , \
                                     av_sh_pooled_eachArea[iarea] + sd_sh_pooled_eachArea[iarea], alpha=alph, edgecolor='gray', facecolor='gray')
                    '''
                    h1a.append(h1_0)
                '''
                lims = np.array([np.nanmin(av_sh_pooled_eachArea - sd_sh_pooled_eachArea), \
                                 np.nanmax(av_ts_pooled_eachArea + sd_ts_pooled_eachArea)])
                lims[0] = lims[0] - np.diff(lims) / 20.
                lims[1] = lims[1] + np.diff(lims) / 20.
                '''
                lims = []


                ################################################################
                #%% Plot traces per depth (pooled across sessions and areas for each depth)
                # Question 2: is there a difference between superficial and deep layers # for each mouse pool across sessions and area (for each layer)... also pool 2 superifical (deep) layers into 1 
                ################################################################

                # trace: average across all pooled sessions for each plane    
                a = ts_eachDepth[:, cre_eachDepth[0,:]==cre]  # 4 x (2*thisCre_pooledSessNum) x 80
                av_ts_pooled_eachDepth = np.nanmean(a, axis=1) # 4 x 80
                sd_ts_pooled_eachDepth = np.nanstd(a, axis=1) / np.sqrt(a.shape[1]) # 4 x 80

                a = sh_eachDepth[:, cre_eachDepth[0,:]==cre]  # 4 x (2*thisCre_pooledSessNum) x 80
                av_sh_pooled_eachDepth = np.nanmean(a, axis=1) # 4 x 80
                sd_sh_pooled_eachDepth = np.nanstd(a, axis=1) / np.sqrt(a.shape[1]) # 4 x 80

                depth_ave = np.mean(depth_eachDepth[:, cre_eachDepth[0,:]==cre], axis=1).astype(float) # 4 # average across areas and sessions


                ##################
                ax = plt.subplot(gs2[1])
            #    ax = plt.subplot(gs1[2,1])
                plt.title('Sessions and areas pooled', fontsize=13, y=1)

    #             h1d = []
                for idepth in range(a.shape[0]):
                    lab = '%d um' %(depth_ave[idepth])
                    if ~np.isnan(svm_blocks):
                        lab = f'{lab},{lab_b}'

                    h1_0 = plt.plot(time_trace, av_ts_pooled_eachDepth[idepth], color = cols_depth[idepth], label=(lab), markersize=3.5, linestyle=linestyle_now)[0]        
                    h2 = plt.plot(time_trace, av_sh_pooled_eachDepth[idepth], color='gray', label='shfl', markersize=3.5)[0] # gray
                    '''
                    plt.fill_between(time_trace, av_ts_pooled_eachDepth[idepth] - sd_ts_pooled_eachDepth[idepth] , \
                                     av_ts_pooled_eachDepth[idepth] + sd_ts_pooled_eachDepth[idepth], alpha=alph, edgecolor=cols_depth[idepth], facecolor=cols_depth[idepth])            
                    plt.fill_between(time_trace, av_sh_pooled_eachDepth[idepth] - sd_sh_pooled_eachDepth[idepth] , \
                                     av_sh_pooled_eachDepth[idepth] + sd_sh_pooled_eachDepth[idepth], alpha=alph, edgecolor='gray', facecolor='gray')
                    '''
                    h1d.append(h1_0)

                '''
                lims = np.array([np.nanmin(av_sh_pooled_eachDepth - sd_sh_pooled_eachDepth), \
                                 np.nanmax(av_ts_pooled_eachDepth + sd_ts_pooled_eachDepth)])
                lims[0] = lims[0] - np.diff(lims) / 20.
                lims[1] = lims[1] + np.diff(lims) / 20.
                '''
                lims = []


                ####################################################################################
                ####################################################################################
                #%% Plot response amplitude
                # per depth; two areas in two subplots       
                ####################################################################################
                ####################################################################################

                ## remember svm peak quantification vars are for training, testing, shuffle, and chance data: trTsShCh

                xlabs = 'Depth (um)'
                x = np.arange(num_depth)
                xticklabs = np.round(depth_ave).astype(int)


                ########################################################
                #%% Image responses

                # left plot: omission, response amplitude
                ylabs = '%Classification accuracy' #'Amplitude'
            #        x = np.arange(num_depth)
                lab1 = 'V1'
                if ~np.isnan(svm_blocks):
                    lab1 = f'{lab1},{lab_b}'
                lab2 = 'LM'
                if ~np.isnan(svm_blocks):
                    lab2 = f'{lab2},{lab_b}'

                top = np.nanmean(pa_all[:, cre_all[0,:]==cre], axis=1)
                top_sd = np.nanstd(pa_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(sum(cre_all[0,:]==cre))        

                ax1 = plt.subplot(gs3[0])

                # testing data
                ax1.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt=fmt_now, markersize=3, capsize=3, label=lab1, color=cols_area_now[1])
                ax1.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt=fmt_now, markersize=3, capsize=3, label=lab2, color=cols_area_now[0])

                # shuffled data
                ax1.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt=fmt_now, markersize=3, capsize=3, color='gainsboro') # label='V1',  # gray
                ax1.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt=fmt_now, markersize=3, capsize=3, color='gainsboro') # label='LM', # skyblue

#                 plt.hlines(0, 0, len(x)-1, linestyle=':')
                ax1.set_xticks(x)
                ax1.set_xticklabels(xticklabs, rotation=45)
                ax1.tick_params(labelsize=10)
                plt.xlim([-.5, len(x)-.5])
                plt.xlabel(xlabs, fontsize=12)
            #        plt.ylim(lims0)
            #        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
            #        plt.title('%s' %(ylabs), fontsize=12)
            #        plt.title('Flash', fontsize=13.5, y=1)
                ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
                plt.ylabel(ylabs, fontsize=12) #, labelpad=35) # , rotation=0
            #        plt.text(3.2, text_y, 'Omission', fontsize=15)
            #     ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 

    #             plt.legend(loc='center left', bbox_to_anchor=(.97,.8), frameon=False, handlelength=1, fontsize=12)

            #        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
                plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
                seaborn.despine()#left=True, bottom=True, right=False, top=False)




                ####################################################################################
                ####################################################################################
                #%% Plot response amplitude
                # pooled layers; compare two areas
                ####################################################################################
                ####################################################################################

                xlabs = 'Area'
                x = np.arange(len(distinct_areas))
                xticklabs = distinct_areas[[1,0]] # so we plot V1 first, then LM


                ########################################################
                #%% Image responses

                # left: omission, response amplitude
                ylabs = '%Classification accuracy'
            #        xlabs = 'Area'
            #        x = np.arange(len(distinct_areas))
            #        xticklabs = xticklabs
                lab = 'data'
    #             lab = f'{lab},{lab_b}'
                if ~np.isnan(svm_blocks):
                    lab = f'{lab_b}'

                top = np.nanmean(pa_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]]
                top_sd = np.nanstd(pa_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] / np.sqrt(sum(cre_eachArea[0,:]==cre))   

                ax1 = plt.subplot(gs5[0])  

                # testing data
                ax1.errorbar(x, top[:,1], yerr=top_sd[:,1], fmt=fmt_now, markersize=3, capsize=3, color=cols_area_now[1], label=lab)

                # shuffled data
                ax1.errorbar(x, top[:,2], yerr=top_sd[:,2], fmt=fmt_now, markersize=3, capsize=3, color='gainsboro') # gray  # , label='shfl'

#                 plt.hlines(0, 0, len(x)-1, linestyle=':')
                ax1.set_xticks(x)
                ax1.set_xticklabels(xticklabs, rotation=45)
                ax1.tick_params(labelsize=10)
                plt.xlim([-.5, len(x)-.5])
                plt.xlabel(xlabs, fontsize=12)
            #        plt.ylim(lims0)
            #        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
            #        plt.title('%s' %(ylabs), fontsize=12)
            #        plt.title('Flash', fontsize=13.5, y=1)
                ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
                plt.ylabel(ylabs, fontsize=12) #, labelpad=30) # , rotation=0
            #        plt.text(3.2, text_y, 'Omission', fontsize=15)
            #        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 

    #             plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)

                plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
                seaborn.despine()#left=True, bottom=True, right=False, top=False)




        ################################################################
        ################################################################            
        if thisCre_pooledSessNum > 0:
            plt.suptitle('%s, %d mice, %d total sessions: %s' %(cre, thisCre_mice_num, thisCre_pooledSessNum, session_labs_all), fontsize=14)


            #%% Add title, lines, ticks and legend to all plots

            plt.subplot(gs1[0])
            a = areas[[inds_v1[0], inds_lm[0]], 0]
    #         lims = lims_v1lm
            lims = [np.min(np.min(lims_v1lm, axis=1)), np.max(np.max(lims_v1lm, axis=1))]
            plot_flashLines_ticks_legend(lims, h1v1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, bbox_to_anchor=bb, ylab=ylabel)
            plt.xlim(xlim);
            plt.title(a[0], fontsize=13, y=1); # np.unique(area)
            # mark time_win: the window over which the response quantification (peak or mean) was computed 
            lims = plt.gca().get_ylim();
            plt.hlines(lims[1], time_win[0], time_win[1], color='gray')



            plt.subplot(gs1[1])            
    #         lims = lims_v1lm
            plot_flashLines_ticks_legend(lims, h1lm, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, ylab=ylabel)
        #    plt.ylabel('')
            plt.xlim(xlim);
            plt.title(a[1], fontsize=13, y=1)
            # mark time_win: the window over which the response quantification (peak or mean) was computed 
            lims = plt.gca().get_ylim();
            plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
        #         plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')



            # add flash lines, ticks and legend
            plt.subplot(gs2[0])
            lims = []
            plot_flashLines_ticks_legend(lims, h1a, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, bbox_to_anchor=bb, ylab=ylabel)
            #    plt.ylabel('')
            plt.xlim(xlim);
            # mark time_win: the window over which the response quantification (peak or mean) was computed 
            lims = plt.gca().get_ylim();
            plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
        #         plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')


            plt.subplot(gs2[1])
            lims = []
            plot_flashLines_ticks_legend(lims, h1d, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, bbox_to_anchor=bb, ylab=ylabel)
        #    plt.ylabel('')
            plt.xlim(xlim);
        #     plt.ylim([10, 55]); # use the same y axis scale for all cell types
            # mark time_win: the window over which the response quantification (peak or mean) was computed 
            lims = plt.gca().get_ylim();        
            plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
        #         plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')



            plt.subplot(gs3[0])
            plt.legend(loc='center left', bbox_to_anchor=(.97,.8), frameon=False, handlelength=1, fontsize=12)



            plt.subplot(gs5[0])
            plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)        



            ################################################################
            ################################################################
            #%%
            if dosavefig:
                # set figure name
    #             snn = [str(sn) for sn in session_numbers]
    #             snn = '_'.join(snn)
                snn = istage
                whatSess = f'_ophys{snn}'

    #             fgn = f'_frames{frames_svm[0]}to{frames_svm[-1]}{whatSess}'
                fgn = f'{whatSess}'
                if same_num_neuron_all_planes:
                    fgn = fgn + '_sameNumNeursAllPlanes'
                #         if ~np.isnan(svm_blocks):
                #             fgn = fgn + f'_block{iblock}'

                if svm_blocks==-1:
                    word = 'engagement'
                else:
                    word = 'blocks'
                fgn = f'{fgn}_{word}_frames{frames_svm[0]}to{frames_svm[-1]}'                        
                fgn = fgn + '_ClassAccur'

                nam = f'{cre[:3]}_aveMice_aveSessPooled{fgn}_{now}'
                fign = os.path.join(dir0, dir_now, nam+fmt)

                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



