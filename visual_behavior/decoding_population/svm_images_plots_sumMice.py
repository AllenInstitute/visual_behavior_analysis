"""
Gets called in svm_images_plots_setVars.py.

Vars needed here are set in "svm_images_plots_setVars_sumMice.py"
Make summary plots across mice (each cre line) for the svm analysis.

Created on Wed Oct 21 15:22:05 2020
@author: farzaneh

"""


if plots_make_finalize[0] == 1:

    ################################################################
    ############################### V1 ###############################
    #             h1v1 = []   
    if ~np.isnan(inds_v1[0]):
        for iplane in inds_v1: # iplane=inds_v1[0] # iplane=0

            ax = plt.subplot(gs1[0]) # plt.subplot(2,1,2)

            area = areas[iplane] # num_sess 
            if project_codes == ['VisualBehaviorMultiscope']:
                depth = depths[iplane] # num_sess
            else:
                depth = depth_ave[0]
            lab = '%dum' %(np.mean(depth))
            if ~np.isnan(svm_blocks) and svm_blocks!=-101:
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
    if ~np.isnan(inds_lm[0]):
        for iplane in inds_lm: #range(4): # iplane = 0
            ax = plt.subplot(gs1[1]) # plt.subplot(2,1,1)

            area = areas[iplane] # num_sess 
            if project_codes == ['VisualBehaviorMultiscope']:
                depth = depths[iplane] # num_sess
            else:
                depth = depth_ave[0]
            lab = '%dum' %(np.mean(depth))
            if ~np.isnan(svm_blocks) and svm_blocks!=-101:
                lab = f'{lab},{lab_b}'

            h1_0 = plt.plot(time_trace, av_ts_eachPlane[iplane], color=cols_depth[iplane], label=(lab), markersize=3.5, linestyle=linestyle_now)[0]            
            h2 = plt.plot(time_trace, av_sh_eachPlane[iplane], color='gray', label='shfl', markersize=3.5)[0]            

            # errorbars (standard error across cv samples)   
            '''
            plt.fill_between(time_trace, av_ts_eachPlane[iplane] - sd_ts_eachPlane[iplane], av_ts_eachPlane[iplane] + sd_ts_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane], facecolor=cols_depth[iplane])            
            plt.fill_between(time_trace, av_sh_eachPlane[iplane] - sd_sh_eachPlane[iplane], av_sh_eachPlane[iplane] + sd_sh_eachPlane[iplane], alpha=alph, edgecolor='gray', facecolor='gray')
            '''
            h1lm.append(h1_0)




    if project_codes == ['VisualBehaviorMultiscope']:
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
            if ~np.isnan(svm_blocks) and svm_blocks!=-101:
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
            if ~np.isnan(svm_blocks) and svm_blocks!=-101:
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
    #                     if baseline_subtract: # subtract the baseline (CA average during baseline, ie before time 0) from the evoked CA (classification accuracy)
    #                         ylabs = '% Class accuracy rel. baseline' #'Amplitude'
    #                     else:
    #                         ylabs = '% Classification accuracy' #'Amplitude'
    #        x = np.arange(num_depth)
        lab1 = 'V1'
        if ~np.isnan(svm_blocks) and svm_blocks!=-101:
            lab1 = f'{lab1},{lab_b}'
        lab2 = 'LM'
        if ~np.isnan(svm_blocks) and svm_blocks!=-101:
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
    #                     if baseline_subtract: # subtract the baseline (CA average during baseline, ie before time 0) from the evoked CA (classification accuracy)
    #                         ylabs = '% Class accuracy rel. baseline' #'Amplitude'
    #                     else:
    #                         ylabs = '% Classification accuracy'
    #                     xlabs = 'Area'
    #                     x = np.arange(len(distinct_areas))
    #                     xticklabs = xticklabs
        lab = 'data'
    #                 lab = f'{lab},{lab_b}'
        if ~np.isnan(svm_blocks) and svm_blocks!=-101:
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
    #                 plt.ylim(lims0)
    #                 ax1.set_ylabel('%s' %(ylabs), fontsize=12)
    #                 plt.title('%s' %(ylabs), fontsize=12)
    #                 plt.title('Flash', fontsize=13.5, y=1)
        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
        plt.ylabel(ylabs, fontsize=12) #, labelpad=30) # , rotation=0
    #                 plt.text(3.2, text_y, 'Omission', fontsize=15)
    #                 ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 

    #                 plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)

        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)



            
            
            
            
            
            

################################################################
################################################################            
# if thisCre_pooledSessNum > 0: # done with both blocks of a given ophys stage

if plots_make_finalize[1] == 1:

    plt.suptitle('%s, %d mice, %d total sessions: %s' %(cre, thisCre_mice_num, thisCre_pooledSessNum, session_labs_all), fontsize=14)


    #%% Add title, lines, ticks and legend to all plots

    if project_codes == ['VisualBehaviorMultiscope']:
        a = areas[[inds_v1[0], inds_lm[0]], 0]
    else:
        a = areas_title

    plt.subplot(gs1[0])
#             lims = lims_v1lm
    lims = [np.min(np.min(lims_v1lm, axis=1)), np.max(np.max(lims_v1lm, axis=1))]
    # i commented below bc for hits_vs_misses it was plotting a weird yellow box, and i didnt spend enough time to figure out why it happens only for hits_vs_misses
    plot_flashLines_ticks_legend(lims, h1v1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, bbox_to_anchor=bb, ylab=ylabel)
    plt.xlim(xlim);
    plt.title(a[0], fontsize=13, y=1); # np.unique(area)
    # mark time_win: the window over which the response quantification (peak or mean) was computed 
    lims = plt.gca().get_ylim();
    plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
#             plt.ylim([9.9, 60.1])


    if ~np.isnan(inds_lm[0]):
        plt.subplot(gs1[1])            
#         lims = lims_v1lm
        # i commented below bc for hits_vs_misses it was plotting a weird yellow box, and i didnt spend enough time to figure out why it happens only for hits_vs_misses
        plot_flashLines_ticks_legend(lims, h1lm, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, ylab=ylabel)
    #    plt.ylabel('')
        plt.xlim(xlim);
        plt.title(a[1], fontsize=13, y=1)
        # mark time_win: the window over which the response quantification (peak or mean) was computed 
        lims = plt.gca().get_ylim();
        plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
#             plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')



    if project_codes == ['VisualBehaviorMultiscope']:
        # add flash lines, ticks and legend
        plt.subplot(gs2[0])
        lims = []
        # i commented below bc for hits_vs_misses it was plotting a weird yellow box, and i didnt spend enough time to figure out why it happens only for hits_vs_misses
        plot_flashLines_ticks_legend(lims, h1a, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, bbox_to_anchor=bb, ylab=ylabel)
        #    plt.ylabel('')
        plt.xlim(xlim);
        # mark time_win: the window over which the response quantification (peak or mean) was computed 
        lims = plt.gca().get_ylim();
        plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
#             plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')


        plt.subplot(gs2[1])
        lims = []
        # i commented below bc for hits_vs_misses it was plotting a weird yellow box, and i didnt spend enough time to figure out why it happens only for hits_vs_misses
        plot_flashLines_ticks_legend(lims, h1d, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, xmjn=xmjn, bbox_to_anchor=bb, ylab=ylabel)
    #    plt.ylabel('')
        plt.xlim(xlim);
    #     plt.ylim([10, 55]); # use the same y axis scale for all cell types
        # mark time_win: the window over which the response quantification (peak or mean) was computed 
        lims = plt.gca().get_ylim();        
        plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
#             plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')



        plt.subplot(gs3[0])
        plt.legend(loc='center left', bbox_to_anchor=(.97,.8), frameon=False, handlelength=1, fontsize=12)



        plt.subplot(gs5[0])
        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)        



    ################################################################
    ################################################################
    #%%
    if dosavefig:
        # set figure name
#                 snn = [str(sn) for sn in session_numbers]
#                 snn = '_'.join(snn)
        snn = istage
        whatSess = f'_ophys{snn}'

#             fgn = f'_frames{frames_svm[0]}to{frames_svm[-1]}{whatSess}'
        fgn = f'{whatSess}'
        if same_num_neuron_all_planes:
            fgn = fgn + '_sameNumNeursAllPlanes'
        #         if ~np.isnan(svm_blocks):
        #             fgn = fgn + f'_block{iblock}'

        if svm_blocks==-1:
            word = 'engaged_disengaged_blocks_'
        elif svm_blocks==-101:
            word = 'only_engaged_'
        elif ~np.isnan(svm_blocks):
            word = 'blocks_'
        else:
            word = ''

        if use_events:
            word = word + 'events'

        fgn = f'{fgn}_{word}_frames{frames_svm[0]}to{frames_svm[-1]}'                        
        fgn = fgn + '_ClassAccur'
#         if project_codes == ['VisualBehavior']:
        fgn = f'{fgn}_{project_codes[0]}'

        nam = f'{cre[:3]}_aveMice_aveSessPooled{fgn}_{now}'
        fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)
        print(fign)


        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



