"""
Vars needed here are set in "svm_images_plots_setVars_sumMice.py"
Make summary plots across mice (each cre line) for the svm analysis.


Created on Wed Oct 21 15:22:05 2020
@author: farzaneh

"""

#%% SUMMARY ACROSS MICE 
##########################################################################################
##########################################################################################
############# Plot traces averaged across mice of the same cre line ################
##########################################################################################    
##########################################################################################

#svm_allMice_sessPooled
#svm_allMice_sessAvSd


#%% Plot image-aligned traces

cre_all = svm_allMice_sessPooled['cre_allPlanes']
cre_lines = np.unique(cre_all)
mouse_id_all = svm_allMice_sessPooled['mouse_id_allPlanes']
a_all = svm_allMice_sessPooled['area_allPlanes']  # 8 x pooledSessNum 
d_all = svm_allMice_sessPooled['depth_allPlanes']  # 8 x pooledSessNum 
session_labs_all = svm_allMice_sessPooled['session_labs']    

ts_all = svm_allMice_sessPooled['av_test_data_allPlanes']  # 8 x pooledSessNum x 80
sh_all = svm_allMice_sessPooled['av_test_shfl_allPlanes']  # 8 x pooledSessNum x 80

pa_all = svm_allMice_sessPooled['peak_amp_allPlanes']  # 8 x pooledSessNum x 4 (trTsShCh)

cre_eachArea = svm_allMice_sessPooled['cre_eachArea'] # 2 x (4*sum(num_sess_per_mouse))
ts_eachArea = svm_allMice_sessPooled['av_test_data_eachArea'] # 2 x (4*sum(num_sess_per_mouse)) x 80
sh_eachArea = svm_allMice_sessPooled['av_test_shfl_eachArea'] # 2 x (4*sum(num_sess_per_mouse)) x 80    
pa_eachArea = svm_allMice_sessPooled['peak_amp_eachArea'] # 2 x (4*sum(num_sess_per_mouse)) x 4 (trTsShCh)

cre_eachDepth = svm_allMice_sessPooled['cre_eachDepth'] # 4 x (2*sum(num_sess_per_mouse))
depth_eachDepth = svm_allMice_sessPooled['depth_eachDepth'] # 4 x (2*sum(num_sess_per_mouse))
ts_eachDepth = svm_allMice_sessPooled['av_test_data_eachDepth'] # 4 x (2*sum(num_sess_per_mouse)) x 80
sh_eachDepth = svm_allMice_sessPooled['av_test_shfl_eachDepth'] # 4 x (2*sum(num_sess_per_mouse)) x 80    


xgap_areas = .3


#%% Plot averages of all mice in each cell line

for icre in range(len(cre_lines)): # icre=0

    cre = cre_lines[icre]    
    thisCre_pooledSessNum = sum(cre_all[0,:]==cre) # number of pooled sessions for this cre line
    thisCre_mice = mouse_id_all[0, cre_all[0,:]==cre]
    thisCre_mice_num = len(np.unique(thisCre_mice))
    
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


    plt.suptitle('%s, %d mice, %d total sessions: %s' %(cre, thisCre_mice_num, thisCre_pooledSessNum, session_labs_all), fontsize=14)


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

    ############################### V1 ###############################
    h1 = []
    for iplane in inds_v1: # iplane = inds_v1[0]

        ax = plt.subplot(gs1[0]) # plt.subplot(2,1,2)

        area = areas[iplane] # num_sess 
        depth = depths[iplane] # num_sess         
        lab = '%dum' %(np.mean(depth))

        h1_0 = plt.plot(time_trace, av_ts_eachPlane[iplane],color=cols_depth[iplane-num_planes], label=(lab), markersize=3.5)[0]
        h2 = plt.plot(time_trace, av_sh_eachPlane[iplane], color='gray', label='shfl', markersize=3.5)[0]

        # errorbars (standard error across cv samples)   
        plt.fill_between(time_trace, av_ts_eachPlane[iplane] - sd_ts_eachPlane[iplane], av_ts_eachPlane[iplane] + sd_ts_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane-num_planes], facecolor=cols_depth[iplane-num_planes])
        plt.fill_between(time_trace, av_sh_eachPlane[iplane] - sd_sh_eachPlane[iplane], av_sh_eachPlane[iplane] + sd_sh_eachPlane[iplane], alpha=alph, edgecolor='gray', facecolor='gray')

        h1.append(h1_0)


    # add to plots flash lines, ticks and legend
    plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, bbox_to_anchor=bb, ylab=ylabel)
    plt.xlim(xlim);
    plt.title(np.unique(area)[0], fontsize=13, y=1);
    # mark time_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims[1], time_win[0], time_win[1], color='gray')


    ############################### LM ###############################
    h1 = []
    for iplane in inds_lm: #range(4): # iplane = 0
        ax = plt.subplot(gs1[1]) # plt.subplot(2,1,1)

        area = areas[iplane] # num_sess 
        depth = depths[iplane] # num_sess             
        lab = '%dum' %(np.mean(depth))

        h1_0 = plt.plot(time_trace, av_ts_eachPlane[iplane], color=cols_depth[iplane], label=(lab), markersize=3.5)[0]            
        h2 = plt.plot(time_trace, av_sh_eachPlane[iplane], color='gray', label='shfl', markersize=3.5)[0]            

        # errorbars (standard error across cv samples)   
        plt.fill_between(time_trace, av_ts_eachPlane[iplane] - sd_ts_eachPlane[iplane], av_ts_eachPlane[iplane] + sd_ts_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane], facecolor=cols_depth[iplane])            
        plt.fill_between(time_trace, av_sh_eachPlane[iplane] - sd_sh_eachPlane[iplane], av_sh_eachPlane[iplane] + sd_sh_eachPlane[iplane], alpha=alph, edgecolor='gray', facecolor='gray')

        h1.append(h1_0)


    # add to plots flash lines, ticks and legend
    plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, ylab=ylabel)
#    plt.ylabel('')
    plt.xlim(xlim);
    plt.title(np.unique(area)[0], fontsize=13, y=1)
    # mark time_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
#         plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')



    ################################################################
    #%% Plot traces per area (pooled across sessions and layers for each area)
    # Question 1: is there a difference between areas (V1 vs LM) # for each mouse pool across sessions and layers (for each area)

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

    h1 = []
    for iarea in [1,0]: # first plot V1 then LM  #range(a.shape[0]):
        lab = '%s' %(distinct_areas[iarea])
        h1_0 = plt.plot(time_trace, av_ts_pooled_eachArea[iarea], color = cols_area[iarea], label=(lab), markersize=3.5)[0]        
        h2 = plt.plot(time_trace, av_sh_pooled_eachArea[iarea], color='gray', label='shfl', markersize=3.5)[0]

        plt.fill_between(time_trace, av_ts_pooled_eachArea[iarea] - sd_ts_pooled_eachArea[iarea] , \
                         av_ts_pooled_eachArea[iarea] + sd_ts_pooled_eachArea[iarea], alpha=alph, edgecolor=cols_area[iarea], facecolor=cols_area[iarea])
        plt.fill_between(time_trace, av_sh_pooled_eachArea[iarea] - sd_sh_pooled_eachArea[iarea] , \
                         av_sh_pooled_eachArea[iarea] + sd_sh_pooled_eachArea[iarea], alpha=alph, edgecolor='gray', facecolor='gray')

        h1.append(h1_0)

    lims = np.array([np.nanmin(av_sh_pooled_eachArea - sd_sh_pooled_eachArea), \
                     np.nanmax(av_ts_pooled_eachArea + sd_ts_pooled_eachArea)])
    lims[0] = lims[0] - np.diff(lims) / 20.
    lims[1] = lims[1] + np.diff(lims) / 20.

    # add to plots flash lines, ticks and legend        
    plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, bbox_to_anchor=bb, ylab=ylabel)
    #    plt.ylabel('')
    plt.xlim(xlim);
    # mark time_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
#         plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')



    ################################################################
    #%% Plot traces per depth (pooled across sessions and areas for each depth)
    # Question 2: is there a difference between superficial and deep layers # for each mouse pool across sessions and area (for each layer)... also pool 2 superifical (deep) layers into 1 

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

    h1 = []
    for idepth in range(a.shape[0]):
        lab = '%d um' %(depth_ave[idepth])
        h1_0 = plt.plot(time_trace, av_ts_pooled_eachDepth[idepth], color = cols_depth[idepth], label=(lab), markersize=3.5)[0]        
        h2 = plt.plot(time_trace, av_sh_pooled_eachDepth[idepth], color='gray', label='shfl', markersize=3.5)[0]

        plt.fill_between(time_trace, av_ts_pooled_eachDepth[idepth] - sd_ts_pooled_eachDepth[idepth] , \
                         av_ts_pooled_eachDepth[idepth] + sd_ts_pooled_eachDepth[idepth], alpha=alph, edgecolor=cols_depth[idepth], facecolor=cols_depth[idepth])            
        plt.fill_between(time_trace, av_sh_pooled_eachDepth[idepth] - sd_sh_pooled_eachDepth[idepth] , \
                         av_sh_pooled_eachDepth[idepth] + sd_sh_pooled_eachDepth[idepth], alpha=alph, edgecolor='gray', facecolor='gray')

        h1.append(h1_0)


    lims = np.array([np.nanmin(av_sh_pooled_eachDepth - sd_sh_pooled_eachDepth), \
                     np.nanmax(av_ts_pooled_eachDepth + sd_ts_pooled_eachDepth)])
    lims[0] = lims[0] - np.diff(lims) / 20.
    lims[1] = lims[1] + np.diff(lims) / 20.

    # add to plots flash lines, ticks and legend        
    plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, bbox_to_anchor=bb, ylab=ylabel)
#    plt.ylabel('')
    plt.xlim(xlim);
    # mark time_win: the window over which the response quantification (peak or mean) was computed 
    plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
#         plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')



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

    top = np.nanmean(pa_all[:, cre_all[0,:]==cre], axis=1)
    top_sd = np.nanstd(pa_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(sum(cre_all[0,:]==cre))        

    ax1 = plt.subplot(gs3[0])

    # testing data
    ax1.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
    ax1.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

    # shuffled data
    ax1.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt='o', markersize=3, capsize=3, color='gray') # label='V1', 
    ax1.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt='o', markersize=3, capsize=3, color='skyblue') # label='LM', 

    plt.hlines(0, 0, len(x)-1, linestyle=':')
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
    plt.legend(loc='center left', bbox_to_anchor=(.97,.8), frameon=False, handlelength=1, fontsize=12)
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

    top = np.nanmean(pa_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]]
    top_sd = np.nanstd(pa_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] / np.sqrt(sum(cre_eachArea[0,:]==cre))   

    ax1 = plt.subplot(gs5[0])  

    # testing data
    ax1.errorbar(x, top[:,1], yerr=top_sd[:,1], fmt='o', markersize=3, capsize=3, color=cols_area[1], label='data')

    # shuffled data
    ax1.errorbar(x, top[:,2], yerr=top_sd[:,2], fmt='o', markersize=3, capsize=3, color='gray', label='shfl')

    plt.hlines(0, 0, len(x)-1, linestyle=':')
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
    plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)

    


    #%%
    if dosavefig:
        nam = f'{cre[:3]}_aveMice_aveSessPooled{fgn}_{now}'
        fign = os.path.join(dir0, dir_now, nam+fmt)
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


