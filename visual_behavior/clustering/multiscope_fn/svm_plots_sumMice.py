"""
Vars needed here are set in "svm_plots_setVars_sumMice.py"
Make summary plots across mice (each cre line) for the svm analysis.

(Note: ABtransit codes right now do not make layer-pooled (per area) plots. To do so, just use svm_allMice_sessPooled same as when ABtransit is 0, except make plots for A, B separately, instead of averaged sessions)

Created on Tue Apr 14 21:56:05 2020
@author: farzaneh

"""

#%% SUMMARY ACROSS MICE 
##########################################################################################
##########################################################################################
############# Plot traces averaged across mice of the same cre line ################
##########################################################################################    
##########################################################################################
            
#%%
if all_ABtransit_AbefB_Aall==1: # plot average across mice for A sessions, and B sessions, separately.

    #%% Decide if you want to plot interpolated or original traces
    #(because I am using linear interpolation it makes no difference really!)

    ts_av_now = av_test_data_ave_allCre # num_cre x (8 x num_sessions) x num_frs
    ts_sd_now = av_test_data_sd_allCre
    # shfl
    sh_av_now = av_test_shfl_ave_allCre # num_cre x (8 x num_sessions) x num_frs
    sh_sd_now = av_test_shfl_sd_allCre
    
    xnow = time_trace_new
    #xnow = time_trace    
    # use frames instead of time
    #xnow = np.unique(np.concatenate((np.arange(0, -(samps_bef+1), -1), np.arange(0, samps_aft))))

    num_sessions = int(ts_av_now.shape[1] / num_planes)
    
    
    #%% Plot the mouse-averaged traces for the two sessions (A, B), done separately for each cre line and each plane
    
#    xlim = [-1.2, 2.25] # [-13, 24]
    frame_dur_new = np.diff(xnow)[0]
    r = np.round(np.array(xlim) / frame_dur_new).astype(int)
    xlim_frs = np.arange(int(np.argwhere(xnow==0).squeeze()) + r[0], int(np.argwhere(xnow==0).squeeze()) + r[1] + 1)
    
    #num_depth = 4
    bbox_to_anchor = (.92, .72)
    
    jet_cm = colorOrder(nlines=num_sessions)
    
    # set x tick marks                
    xmj = np.unique(np.concatenate((np.arange(0, time_trace[0], -1), np.arange(0, time_trace[-1], 1))))
    xmn = np.arange(.25, time_trace[-1], .5)
    xmjn = [xmj, xmn]
    
    # set ylim: same for all plots of all cre lines
    lims0 = [np.min(ts_av_now[:,:,xlim_frs] - ts_sd_now[:,:,xlim_frs]), np.max(ts_av_now[:,:,xlim_frs] + ts_sd_now[:,:,xlim_frs])]
    
    for icre in range(len(all_cres)): # icre = 0
        
        cre = all_cres[icre]
        
        
        fa = plt.figure(figsize=(12, 15)) # (12, 5)
        plt.suptitle('%s, %d mice' %(cre, num_each_cre[icre]), y=1.1, fontsize=18) # .94    
        
        ##### classification accuracy plots #####
        # subplots of traces, V1
        gs = gridspec.GridSpec(1, num_depth) #, width_ratios=[3, 1]) 
        gs.update(bottom=0.87, top=.98, wspace=.6)        
#         gs.update(bottom=0.63, top=0.95, wspace=.6) #, hspace=1)
        
        # subplots of traces, LM
        gs2 = gridspec.GridSpec(1, num_depth)
        gs2.update(bottom=0.71, top=0.82, wspace=0.6)        
#         gs2.update(bottom=0.05, top=0.37, wspace=0.6) #, hspace=1)
    
        
        ##### Response measures; each depth ##### 
        # subplots of flash responses
        # 4 subplots: resp amp, V1 and LM; then resp timing, V1 and LM
        gs3 = gridspec.GridSpec(1,4) #, width_ratios=[3, 1]) 
        gs3.update(bottom=.52, top=0.63, wspace=.65, hspace=.5) #, left=0.03, right=0.65

        # subplots of omission responses
        # resp amp, V1 and LM; then resp timing, V1 and LM
        gs4 = gridspec.GridSpec(1,4)
        gs4.update(bottom=.37, top=0.48, wspace=.65, hspace=.5) #, left=0.03, right=0.65
        


        ##### Response Modulation (change) measures; each depth ##### 
        # subplots of flash responses; 2 subplots: resp amp, resp timing (each includes both v1 and lm)
        gs5 = gridspec.GridSpec(1,4) #, width_ratios=[3, 1]) 
        gs5.update(bottom=.16, top=.28, wspace=.55, hspace=.5) # , left=0.05, right=0.55
#        gs5.update(bottom=.17, top=0.3, left=0.05, right=0.43, wspace=.55, hspace=.5)

        # subplots of omission responses; 2 subplots: resp amp, resp timing (each includes both v1 and lm)
        gs6 = gridspec.GridSpec(1,4)
        gs6.update(bottom=0, top=0.12, wspace=.55, hspace=.5) # , left=0.05, right=0.55
#        gs6.update(bottom=.17, top=0.3, left=0.57, right=0.95, wspace=.55, hspace=.5)


        '''
        ##### Response measures; compare areas (pooled depths) ##### 
        # subplots of flash responses
        # 4 subplots: resp amp, resp timing
        gs5 = gridspec.GridSpec(1,2) #, width_ratios=[3, 1]) 
        gs5.update(bottom=.22, top=0.35, left=0.72, right=0.97, wspace=.55, hspace=.5)

        # subplots of omission responses
        # resp amp, resp timing
        gs6 = gridspec.GridSpec(1,2)
        gs6.update(bottom=.05, top=0.18, left=0.72, right=0.97, wspace=.55, hspace=.5)
        '''

        ############################################################
        ########## Plot classification accuracies ##########
        ############################################################        
        
        lims0 = [np.min(ts_av_now[icre,:,xlim_frs] - ts_sd_now[icre,:,xlim_frs]), np.max(ts_av_now[icre,:,xlim_frs] + ts_sd_now[icre,:,xlim_frs])]  # set ylim: same for all plots of the same cre line
    
    
        ############ Subplot 1: area V1 ############ (planes 4:8)
        for iplane in inds_v1: # iplane = inds_v1[0] #np.arange(num_depth, num_planes): #num_planes): 
    
            area = svm_this_plane_allsess.iloc[0]['area'][iplane] # take it from the 1st mouse, this info is the same for all mice
            onePlane_allSess = np.arange(iplane, np.shape(ts_av_now)[1], num_planes) # 2, this plane index for all sessions         
            depth = depth_ave_allCre[icre, onePlane_allSess] # 2 # depth_ave_allCre: num_cre x (8 x num_sessions)
            
            area = np.unique(area)[0]
            depth = np.mean(depth) # average across days        
            
            top = ts_av_now[icre, onePlane_allSess] # num_sess x num_frs
            top_sd = ts_sd_now[icre, onePlane_allSess] # num_sess x num_frs
            # shfl
            topsh = sh_av_now[icre, onePlane_allSess] # num_sess x num_frs
            topsh_sd = sh_sd_now[icre, onePlane_allSess] # num_sess x num_frs
            
    
            ax1 = plt.subplot(gs[iplane-num_depth])        
            
            h = []
            for isess in range(num_sessions): # loop through A and B sessions             
                h0 = plt.plot(xnow, top.T[:,isess], color=jet_cm[isess], label=session_labs[isess])[0]
                plt.plot(xnow, topsh.T[:,isess], color='gray')

                # errorbars (standard error across mice)   
                plt.fill_between(xnow, top.T[:,isess] - top_sd.T[:,isess], top.T[:,isess] + top_sd.T[:,isess], alpha=alph,  edgecolor=jet_cm[isess], facecolor=jet_cm[isess])
                plt.fill_between(xnow, topsh.T[:,isess] - topsh_sd.T[:,isess], topsh.T[:,isess] + topsh_sd.T[:,isess], alpha=alph,  edgecolor=jet_cm[isess], facecolor=jet_cm[isess])

                h.append(h0)
                
    #        mn = np.min(top.flatten())
    #        mx = np.max(top.flatten())       
    #        lims = [mn, mx]         
            ylabn = ylabel if iplane==num_depth else ''
     
            lims = np.array([np.nanmin(top[:,xlim_frs] - top_sd[:,xlim_frs]), np.nanmax(top[:,xlim_frs] + top_sd[:,xlim_frs])])
            lims[0] = lims[0] - np.diff(lims) / 20.
            lims[1] = lims[1] + np.diff(lims) / 20.
                    
            plot_flashLines_ticks_legend(lims0, h, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor, ylab=ylabel, xmjn=xmjn)
    
            plt.title('%s, %dum' %(area, depth), fontsize=13.5, y=1)
            plt.xlim(xlim)
            plt.xlabel('')
    
    #        plt.vlines(0, mn, mx, 'r', linestyle='-') #, cmap=colorsm)        
    #        plt.vlines(flashes_win_trace_index_unq, mn, mx, linestyle=':')
    #        plt.xlabel('Frame (~10Hz)', fontsize=12)
    #        if iplane==num_depth:
    #            plt.ylabel('DF/F %s' %(ylab), fontsize=12)            
    ##        if iplane==0: #num_planes-1:
    #        plt.legend(h, session_labs, loc='center left', bbox_to_anchor=bbox_to_anchor, frameon=False, fontsize=12, handlelength=.4)        
    #        ax1.tick_params(labelsize=10)
    #        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    #        seaborn.despine()#left=True, bottom=True, right=False, top=False)
    
    
    
    
        ############ Subplot 2: area LM ############ (planes 0:3)
        for iplane in inds_lm: # iplane = inds_lm[0] #range(num_depth): #num_planes): # iplane = 0
    
            area = svm_this_plane_allsess.iloc[0]['area'][iplane] # take it from the 1st mouse, this info is the same for all mice
            onePlane_allSess = np.arange(iplane, np.shape(ts_av_now)[1], num_planes) # 2, this plane index for all sessions    
            depth = depth_ave_allCre[icre, onePlane_allSess] # 2 # depth_ave_allCre: num_cre x (8 x num_sessions)
            
            area = np.unique(area)[0]
            depth = np.mean(depth)        
            
            top = ts_av_now[icre, onePlane_allSess] # num_sess x num_frs
            top_sd = ts_sd_now[icre, onePlane_allSess] # num_sess x num_frs
            # shfl
            topsh = sh_av_now[icre, onePlane_allSess] # num_sess x num_frs
            topsh_sd = sh_sd_now[icre, onePlane_allSess] # num_sess x num_frs    
    
    
            ax2 = plt.subplot(gs2[iplane])     
               
            h = []
            for isess in range(num_sessions): # loop through A and B sessions             
                h0 = plt.plot(xnow, top.T[:,isess], color=jet_cm[isess], label=session_labs[isess])[0]
                plt.plot(xnow, topsh.T[:,isess], color='gray')

                # errorbars (standard error across mice)   
                plt.fill_between(xnow, top.T[:,isess] - top_sd.T[:,isess], top.T[:,isess] + top_sd.T[:,isess], alpha=alph,  edgecolor=jet_cm[isess], facecolor=jet_cm[isess])
                plt.fill_between(xnow, topsh.T[:,isess] - topsh_sd.T[:,isess], topsh.T[:,isess] + topsh_sd.T[:,isess], alpha=alph,  edgecolor=jet_cm[isess], facecolor=jet_cm[isess])

                h.append(h0)
                
    #        mn = np.min(top.flatten())
    #        mx = np.max(top.flatten())
    #        lims = [mn, mx]                 
            ylabn = ylabel if iplane==0 else ''
    
            lims = np.array([np.nanmin(top[:,xlim_frs] - top_sd[:,xlim_frs]), np.nanmax(top[:,xlim_frs] + top_sd[:,xlim_frs])])
            lims[0] = lims[0] - np.diff(lims) / 20.
            lims[1] = lims[1] + np.diff(lims) / 20.
    
            plot_flashLines_ticks_legend(lims0, h, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor, ylab=ylabel, xmjn=xmjn)
    
            plt.title('%s, %dum' %(area, depth), fontsize=13.5, y=1)
            plt.xlim(xlim)        
    
    #        plt.vlines(0, mn, mx, 'r', linestyle='-') #, cmap=colorsm)        
    #        plt.vlines(flashes_win_trace_index_unq, mn, mx, linestyle=':')
    #        plt.xlabel('Frame (~10Hz)', fontsize=12)
    #        plt.legend(h, session_labs, loc='center left', bbox_to_anchor=bbox_to_anchor, frameon=False, fontsize=12, handlelength=.4)        
    #        ax2.tick_params(labelsize=10)
    #        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    #        seaborn.despine()#left=True, bottom=True, right=False, top=False)
                        
        
        #%%     
        ##########################################################################################
        ##########################################################################################
        ############# Plot peak amplitude for omission and flash evoked responses ################
        ##########################################################################################    
        ##########################################################################################
        
        ####################################################################
        ############ Subplot gs3 & gs4: flash- & omission-evoked responses ################### 
        ### 4 subplots: resp amp, V1 and LM; then resp timing, V1 and LM ###
        ####################################################################
        
        xlabs = 'Depth (um)'
        x = np.arange(num_depth)
        xgap_areas = .2 #.3

        depth = depth_ave_allCre[icre]  # (8 x num_sessions)        

        inds_v1_lm = inds_v1, inds_lm
        labvl = 'V1 ', 'LM '
    
        
        ########################################################################
        #%% Flash-evoked responses
        ########################################################################
        
        ###### Flash: Response amplitude, V1, then LM ######

        y_ave_now = peak_amp_flash_trTsShCh_ave_allCre # num_cre x (8 x num_sessions) x 4(trTsShCh)
        y_sd_now = peak_amp_flash_trTsShCh_sd_allCre

        top = y_ave_now[icre] # (8 x num_sessions) x 4(trTsShCh)
        top_sd = y_sd_now[icre] 

        # same ylim for all layers and areas
        mn = np.min(top[:,[1,2]] - top_sd[:,[1,2]])
        mx = np.max(top[:,[1,2]] + top_sd[:,[1,2]])
        r = mx - mn
        lims0 = [mn-r/20., mx+r/20.]  # set ylim: same for all plots of the same cre line
        
        ylabs0 = 'Resp amplitude'        
        
        for ivl in range(len(labvl)):
            
            inds_area_allSess = np.array([np.arange(inds_v1_lm[ivl][iplane], y_ave_now.shape[1], num_planes) for iplane in range(num_depth)])
    #         inds_area_allSess: # 4 x num_sessions ; each column is the indeces of v1 planes for a given session (along the 2nd dim of y_ave_now).
    #         inds_v1 + num_planes
            indsA = inds_area_allSess[:,0] # depth indeces for V1, for session A
            indsB = inds_area_allSess[:,1] # depth indeces for V1, for session B    

            depth_area_eachSess = np.array([depth[inds_area_allSess[idepth]] for idepth in range(num_depth)]) # 4 x num_sessions
            depth_ave = np.mean(depth_area_eachSess, axis=1) # average across sessions, depth of each plane
            xticklabs = np.round(depth_ave).astype(int)
            ylabs = labvl[ivl] + ylabs0

            # flash, response amplitude; V1, then LM; 4 depths, A,B sessions
            ax1 = plt.subplot(gs3[ivl])

            ### testing data # plot testing, and shuffle data, ie indeces 1 and 2 in top    # ind_ts_sh = [1,2] (train, test, shuffle, chance)
            # session A
            ax1.errorbar(x, top[indsA, 1], yerr=top_sd[indsA, 1], fmt='o', markersize=3, capsize=3, label='A', color='b')
            # session B
            ax1.errorbar(x+xgap_areas, top[indsB, 1], yerr=top_sd[indsB, 1], fmt='o', markersize=3, capsize=3, label='B', color='r')

            ### shuffled data
            # session A
            ax1.errorbar(x, top[indsA, 2], yerr=top_sd[indsA, 2], fmt='o', markersize=3, capsize=3, color='skyblue')
            # session B
            ax1.errorbar(x+xgap_areas, top[indsB, 2], yerr=top_sd[indsB, 2], fmt='o', markersize=3, capsize=3, color='mistyrose')

            plt.hlines(0, 0, len(x)-1, linestyle=':')
            ax1.set_xticks(x)
            ax1.set_xticklabels(xticklabs, rotation=45)
            ax1.tick_params(labelsize=10)
            plt.xlim([-.5, len(x)-.5])
    #        plt.xlabel('Depth', fontsize=12)
            if ~np.isnan(lims0).any():
                plt.ylim(lims0)
    #        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
            plt.title('%s' %(ylabs), fontsize=13) #12
    #        plt.title('Flash', fontsize=13.5, y=1)
            ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
            if ivl==0:
                plt.ylabel('Flash', fontsize=15, rotation=0, labelpad=35)
    #        plt.text(3.2, text_y, 'Flash', fontsize=15)
            if ivl==0:
#                 ax1.legend(loc=3, bbox_to_anchor=(-.1, 1.2, 1, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
                plt.legend(loc='center left', bbox_to_anchor=(.97,.8), frameon=False, handlelength=1, fontsize=12)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)

            
            
        ################## Flash: Response timing, V1, then LM ##################

        y_ave_now = peak_timing_flash_trTsShCh_ave_allCre # num_cre x (8 x num_sessions) x 4(trTsShCh)
        y_sd_now = peak_timing_flash_trTsShCh_sd_allCre

        top = y_ave_now[icre] # (8 x num_sessions) x 4(trTsShCh)
        top_sd = y_sd_now[icre] 

        # same ylim for all layers and areas
        mn = np.min(top[:,[1,2]] - top_sd[:,[1,2]])
        mx = np.max(top[:,[1,2]] + top_sd[:,[1,2]])
        r = mx - mn
        lims0 = [mn-r/20., mx+r/20.]  # set ylim: same for all plots of the same cre line

        
        ylabs0 = 'Resp timing'        
        
        for ivl in range(len(labvl)):
            
            inds_area_allSess = np.array([np.arange(inds_v1_lm[ivl][iplane], y_ave_now.shape[1], num_planes) for iplane in range(num_depth)])
    #         inds_area_allSess: # 4 x num_sessions ; each column is the indeces of v1 planes for a given session (along the 2nd dim of y_ave_now).
    #         inds_v1 + num_planes
            indsA = inds_area_allSess[:,0] # depth indeces for V1, for session A
            indsB = inds_area_allSess[:,1] # depth indeces for V1, for session B    

            depth_area_eachSess = np.array([depth[inds_area_allSess[idepth]] for idepth in range(num_depth)]) # 4 x num_sessions
            depth_ave = np.mean(depth_area_eachSess, axis=1) # average across sessions, depth of each plane
            xticklabs = np.round(depth_ave).astype(int)
            ylabs = labvl[ivl] + ylabs0

            # flash, response amplitude; V1, then LM; 4 depths, A,B sessions
            ax1 = plt.subplot(gs3[2+ivl])

            ### testing data # plot testing, and shuffle data, ie indeces 1 and 2 in top    # ind_ts_sh = [1,2] (train, test, shuffle, chance)
            # session A
            ax1.errorbar(x, top[indsA, 1], yerr=top_sd[indsA, 1], fmt='o', markersize=3, capsize=3, label='A', color='b')
            # session B
            ax1.errorbar(x+xgap_areas, top[indsB, 1], yerr=top_sd[indsB, 1], fmt='o', markersize=3, capsize=3, label='B', color='r')

            ### shuffled data
            # session A
#             ax1.errorbar(x, top[indsA, 2], yerr=top_sd[indsA, 2], fmt='o', markersize=3, capsize=3, color='skyblue')
            # session B
#             ax1.errorbar(x+xgap_areas, top[indsB, 2], yerr=top_sd[indsB, 2], fmt='o', markersize=3, capsize=3, color='mistyrose')

            plt.hlines(0, 0, len(x)-1, linestyle=':')
            ax1.set_xticks(x)
            ax1.set_xticklabels(xticklabs, rotation=45)
            ax1.tick_params(labelsize=10)
            plt.xlim([-.5, len(x)-.5])
    #        plt.xlabel('Depth', fontsize=12)
            if ~np.isnan(lims0).any():
                plt.ylim(lims0)
    #        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
            plt.title('%s' %(ylabs), fontsize=13) #12
    #        plt.title('Flash', fontsize=13.5, y=1)
            ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
#             plt.ylabel('Flash', fontsize=15, rotation=0, labelpad=35)
    #        plt.text(3.2, text_y, 'Flash', fontsize=15)
#             ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
    #        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)

            
    
        ########################################################################
        #%% Omission-evoked responses
        ########################################################################
        
        ###### Omission: Response amplitude, V1, then LM ######

        y_ave_now = peak_amp_omit_trTsShCh_ave_allCre # num_cre x (8 x num_sessions) x 4(trTsShCh)
        y_sd_now = peak_amp_omit_trTsShCh_sd_allCre

        top = y_ave_now[icre] # (8 x num_sessions) x 4(trTsShCh)
        top_sd = y_sd_now[icre] 

        # same ylim for all layers and areas
        mn = np.min(top[:,[1,2]] - top_sd[:,[1,2]])
        mx = np.max(top[:,[1,2]] + top_sd[:,[1,2]])
        r = mx - mn
        lims0 = [mn-r/20., mx+r/20.]  # set ylim: same for all plots of the same cre line

        
        ylabs0 = 'Resp amplitude'        
        
        for ivl in range(len(labvl)):
            
            inds_area_allSess = np.array([np.arange(inds_v1_lm[ivl][iplane], y_ave_now.shape[1], num_planes) for iplane in range(num_depth)])
    #         inds_area_allSess: # 4 x num_sessions ; each column is the indeces of v1 planes for a given session (along the 2nd dim of y_ave_now).
    #         inds_v1 + num_planes
            indsA = inds_area_allSess[:,0] # depth indeces for V1, for session A
            indsB = inds_area_allSess[:,1] # depth indeces for V1, for session B    

            depth_area_eachSess = np.array([depth[inds_area_allSess[idepth]] for idepth in range(num_depth)]) # 4 x num_sessions
            depth_ave = np.mean(depth_area_eachSess, axis=1) # average across sessions, depth of each plane
            xticklabs = np.round(depth_ave).astype(int)
            ylabs = labvl[ivl] + ylabs0

            # omission, response amplitude; V1, then LM; 4 depths, A,B sessions
            ax1 = plt.subplot(gs4[ivl])

            ### testing data # plot testing, and shuffle data, ie indeces 1 and 2 in top    # ind_ts_sh = [1,2] (train, test, shuffle, chance)
            # session A
            ax1.errorbar(x, top[indsA, 1], yerr=top_sd[indsA, 1], fmt='o', markersize=3, capsize=3, label='A', color='b')
            # session B
            ax1.errorbar(x+xgap_areas, top[indsB, 1], yerr=top_sd[indsB, 1], fmt='o', markersize=3, capsize=3, label='B', color='r')

            ### shuffled data
            # session A
            ax1.errorbar(x, top[indsA, 2], yerr=top_sd[indsA, 2], fmt='o', markersize=3, capsize=3, label='A', color='skyblue')
            # session B
            ax1.errorbar(x+xgap_areas, top[indsB, 2], yerr=top_sd[indsB, 2], fmt='o', markersize=3, capsize=3, label='B', color='mistyrose')

            plt.hlines(0, 0, len(x)-1, linestyle=':')
            ax1.set_xticks(x)
            ax1.set_xticklabels(xticklabs, rotation=45)
            ax1.tick_params(labelsize=10)
            plt.xlim([-.5, len(x)-.5])
            plt.xlabel('Depth', fontsize=12)
            if ~np.isnan(lims0).any():    
                plt.ylim(lims0)
    #        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
#             plt.title('%s' %(ylabs), fontsize=13) #12
    #        plt.title('Omission', fontsize=13.5, y=1)
            ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
            if ivl==0:
                plt.ylabel('Omission', fontsize=15, rotation=0, labelpad=35)
    #        plt.text(3.2, text_y, 'Omission', fontsize=15)
    #        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
    #        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)

            
            
        ################## Omission: Response timing, V1, then LM ##################

        y_ave_now = peak_timing_omit_trTsShCh_ave_allCre # num_cre x (8 x num_sessions) x 4(trTsShCh)
        y_sd_now = peak_timing_omit_trTsShCh_sd_allCre

        top = y_ave_now[icre] # (8 x num_sessions) x 4(trTsShCh)
        top_sd = y_sd_now[icre] 

        # same ylim for all layers and areas
        mn = np.min(top[:,[1,2]] - top_sd[:,[1,2]])
        mx = np.max(top[:,[1,2]] + top_sd[:,[1,2]])
        r = mx - mn
        lims0 = [mn-r/20., mx+r/20.]  # set ylim: same for all plots of the same cre line

        
        ylabs0 = 'Resp timing'        
        
        for ivl in range(len(labvl)):
            
            inds_area_allSess = np.array([np.arange(inds_v1_lm[ivl][iplane], y_ave_now.shape[1], num_planes) for iplane in range(num_depth)])
    #         inds_area_allSess: # 4 x num_sessions ; each column is the indeces of v1 planes for a given session (along the 2nd dim of y_ave_now).
    #         inds_v1 + num_planes
            indsA = inds_area_allSess[:,0] # depth indeces for V1, for session A
            indsB = inds_area_allSess[:,1] # depth indeces for V1, for session B    

            depth_area_eachSess = np.array([depth[inds_area_allSess[idepth]] for idepth in range(num_depth)]) # 4 x num_sessions
            depth_ave = np.mean(depth_area_eachSess, axis=1) # average across sessions, depth of each plane
            xticklabs = np.round(depth_ave).astype(int)
            ylabs = labvl[ivl] + ylabs0

            # omission, response amplitude; V1, then LM; 4 depths, A,B sessions
            ax1 = plt.subplot(gs4[2+ivl])

            ### testing data # plot testing, and shuffle data, ie indeces 1 and 2 in top    # ind_ts_sh = [1,2] (train, test, shuffle, chance)
            # session A
            ax1.errorbar(x, top[indsA, 1], yerr=top_sd[indsA, 1], fmt='o', markersize=3, capsize=3, label='A', color='b')
            # session B
            ax1.errorbar(x+xgap_areas, top[indsB, 1], yerr=top_sd[indsB, 1], fmt='o', markersize=3, capsize=3, label='B', color='r')

            ### shuffled data
            # session A
#             ax1.errorbar(x, top[indsA, 2], yerr=top_sd[indsA, 2], fmt='o', markersize=3, capsize=3, label='A', color='skyblue')
            # session B
#             ax1.errorbar(x+xgap_areas, top[indsB, 2], yerr=top_sd[indsB, 2], fmt='o', markersize=3, capsize=3, label='B', color='mistyrose')

            plt.hlines(0, 0, len(x)-1, linestyle=':')
            ax1.set_xticks(x)
            ax1.set_xticklabels(xticklabs, rotation=45)
            ax1.tick_params(labelsize=10)
            plt.xlim([-.5, len(x)-.5])
            plt.xlabel('Depth', fontsize=12)
            if ~np.isnan(lims0).any():
                plt.ylim(lims0)
    #        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
#             plt.title('%s' %(ylabs), fontsize=13) #12
    #        plt.title('Omission', fontsize=13.5, y=1)
            ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
#             plt.ylabel('Omission', fontsize=15, rotation=0, labelpad=35)
    #        plt.text(3.2, text_y, 'Omission', fontsize=15)
    #        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
    #        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)


            
        ####################################################################################
        ############## Plot relative change of response amplitude and timing from A to B ##############
        ####################################################################################
        # add 4 subplots; 2 for resp amp (flash, omit), 2 for resp timing (flash, omit)
        # each shows resp modulation for 4 depths.
                
        # depth_ave_allCre[icre].shape  # (8 x num_sessions)
        d = np.reshape(depth_ave_allCre[icre], (num_planes, num_sessions), order='F') # 8 x num_sessions
        depth_ave = np.mean(d, axis=1) # 8 # average across A,B sessions

        # same ylim for all layers and areas
        mn = np.min(top[:,[1,2]] - top_sd[:,[1,2]])
        mx = np.max(top[:,[1,2]] + top_sd[:,[1,2]])
        r = mx - mn
        lims0 = [mn-r/20., mx+r/20.]  # set ylim: same for all plots of the same cre line
        
        xlabs = 'Depth (um)'
        x = np.arange(num_depth)
        xticklabs = np.round(depth_ave).astype(int)
            
        
        ###########################################
        #%% Flash-evoked responses
        ###########################################
        
        ################# left plot: flash, response amplitude #######################

        ylabs = 'Resp amplitude'
        
        y_ave_now = peak_amp_mod_flash_trTsShCh_ave_allCre # num_cre x 8 x 4    
        y_sd_now = peak_amp_mod_flash_trTsShCh_sd_allCre

        top = y_ave_now[icre] # 8 x 4(trTsShCh)
        top_sd = y_sd_now[icre]         
        
        ax1 = plt.subplot(gs5[0])
        
        # testing data
        ax1.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax1.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

        # shuffled data
        ax1.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt='o', markersize=3, capsize=3, label='V1', color='gray')
        ax1.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt='o', markersize=3, capsize=3, label='LM', color='skyblue')
        
        plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Flash', fontsize=13.5, y=1)
        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
        plt.ylabel('Flash', fontsize=15, rotation=0, labelpad=35)
#        plt.text(3.2, text_y, 'Flash', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)

        

        ################# right plot: flash, response timing #######################

        ylabs = 'Peak timing (s)'
        
        y_ave_now = peak_timing_mod_flash_trTsShCh_ave_allCre # num_cre x 8 x 4    
        y_sd_now = peak_timing_mod_flash_trTsShCh_sd_allCre

        top = y_ave_now[icre] # 8 x 4(trTsShCh)
        top_sd = y_sd_now[icre] 
                
        ax2 = plt.subplot(gs5[1])
        
        # testing data
        ax2.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax2.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

        # shuffled data : we don't plot timing for shuffled data, bc they don't have a response!
#         ax2.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt='o', markersize=3, capsize=3, label='V1', color='gray')
#         ax2.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt='o', markersize=3, capsize=3, label='LM', color='skyblue')
        
#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Flash', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Flash', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)
                
            
        
        ###########################################
        #%% Omission-evoked responses
        ###########################################
        
        ################# left plot: omission, response amplitude #######################

        ylabs = 'Resp amplitude'
        
        y_ave_now = peak_amp_mod_omit_trTsShCh_ave_allCre # num_cre x 8 x 4    
        y_sd_now = peak_amp_mod_omit_trTsShCh_sd_allCre

        top = y_ave_now[icre] # 8 x 4(trTsShCh)
        top_sd = y_sd_now[icre]

        ax1 = plt.subplot(gs6[0])
        
        # testing data
        ax1.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax1.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

        # shuffled data
        ax1.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt='o', markersize=3, capsize=3, label='V1', color='gray')
        ax1.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt='o', markersize=3, capsize=3, label='LM', color='skyblue')

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
        plt.ylabel('Omission', fontsize=15, rotation=0, labelpad=35)
#        plt.text(3.2, text_y, 'Omission', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)
            
            
            
        ################# right plot: omission, response timing #######################

        ylabs = 'Peak timing'
        
        y_ave_now = peak_timing_mod_omit_trTsShCh_ave_allCre # num_cre x 8 x 4    
        y_sd_now = peak_timing_mod_omit_trTsShCh_sd_allCre

        top = y_ave_now[icre] # 8 x 4(trTsShCh)
        top_sd = y_sd_now[icre]

        ax1 = plt.subplot(gs6[1])
        
        # testing data
        ax2.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax2.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

        # shuffled data
#         ax2.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt='o', markersize=3, capsize=3, label='V1', color='gray')
#         ax2.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt='o', markersize=3, capsize=3, label='LM', color='skyblue')
        
#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.xlabel(xlabs, fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
#        plt.title('%s' %(ylabs), fontsize=12)
#        plt.title('Flash', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Flash', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


            

        if dosavefig:
            nam = '%s_aveMice%s_%s_%s' %(cre, whatSess, fgn, now)
            fign = os.path.join(dir0, dir_now, nam+fmt)        
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
        
        

        
        
        
        
        
        
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################            
#%% Not A-B transitions
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################            
#svm_allMice_sessPooled
#svm_allMice_sessAvSd

else: 

    #%% Plot omission-aligned traces
    
    cre_all = svm_allMice_sessPooled['cre_allPlanes']
    cre_lines = np.unique(cre_all)
    a_all = svm_allMice_sessPooled['area_allPlanes']  # 8 x pooledSessNum 
    d_all = svm_allMice_sessPooled['depth_allPlanes']  # 8 x pooledSessNum 
    session_labs_all = svm_allMice_sessPooled['session_labs']    

    ts_all = svm_allMice_sessPooled['av_test_data_allPlanes']  # 8 x pooledSessNum x 80
    sh_all = svm_allMice_sessPooled['av_test_shfl_allPlanes']  # 8 x pooledSessNum x 80
    
    pa_all = svm_allMice_sessPooled['peak_amp_omit_allPlanes']  # 8 x pooledSessNum x 4 (trTsShCh)
    pt_all = svm_allMice_sessPooled['peak_timing_omit_allPlanes']  # 8 x pooledSessNum x 4     
    paf_all = svm_allMice_sessPooled['peak_amp_flash_allPlanes']  # 8 x pooledSessNum x 4     
    ptf_all = svm_allMice_sessPooled['peak_timing_flash_allPlanes']  # 8 x pooledSessNum x 4     
    
    cre_eachArea = svm_allMice_sessPooled['cre_eachArea'] # 2 x (4*sum(num_sess_per_mouse))
    ts_eachArea = svm_allMice_sessPooled['av_test_data_eachArea'] # 2 x (4*sum(num_sess_per_mouse)) x 80
    sh_eachArea = svm_allMice_sessPooled['av_test_shfl_eachArea'] # 2 x (4*sum(num_sess_per_mouse)) x 80    
    pa_eachArea = svm_allMice_sessPooled['peak_amp_omit_eachArea'] # 2 x (4*sum(num_sess_per_mouse)) x 4 (trTsShCh)
    pt_eachArea = svm_allMice_sessPooled['peak_timing_omit_eachArea'] # 2 x (4*sum(num_sess_per_mouse)) x 4 (trTsShCh)
    paf_eachArea = svm_allMice_sessPooled['peak_amp_flash_eachArea'] # 2 x (4*sum(num_sess_per_mouse)) x 4 (trTsShCh)
    ptf_eachArea = svm_allMice_sessPooled['peak_timing_flash_eachArea'] # 2 x (4*sum(num_sess_per_mouse)) x 4 (trTsShCh)
    
    cre_eachDepth = svm_allMice_sessPooled['cre_eachDepth'] # 4 x (2*sum(num_sess_per_mouse))
    depth_eachDepth = svm_allMice_sessPooled['depth_eachDepth'] # 4 x (2*sum(num_sess_per_mouse))
    ts_eachDepth = svm_allMice_sessPooled['av_test_data_eachDepth'] # 4 x (2*sum(num_sess_per_mouse)) x 80
    sh_eachDepth = svm_allMice_sessPooled['av_test_shfl_eachDepth'] # 4 x (2*sum(num_sess_per_mouse)) x 80    
#    pa_eachDepth = svm_allMice_sessPooled['peak_amp_omit_eachDepth'] # 4 x (2*sum(num_sess_per_mouse)) x 4 (trTsShCh)
#    pt_eachDepth = svm_allMice_sessPooled['peak_timing_omit_eachDepth'] # 4 x (2*sum(num_sess_per_mouse)) x 4 (trTsShCh)
#    paf_eachDepth = svm_allMice_sessPooled['peak_amp_flash_eachDepth'] # 4 x (2*sum(num_sess_per_mouse)) x 4 (trTsShCh)
#    ptf_eachDepth = svm_allMice_sessPooled['peak_timing_flash_eachDepth'] # 4 x (2*sum(num_sess_per_mouse)) x 4 (trTsShCh)
    

    xgap_areas = .3
    
    
    #%% Plot averages of all mice in each cell line
    
    for icre in range(len(cre_lines)): # icre=0
        
        cre = cre_lines[icre]    
        thisCre_pooledSessNum = sum(cre_all[0,:]==cre) # number of pooled sessions for this cre line

        ###############################         
        plt.figure(figsize=(8,11))  
        
        ##### traces ##### 
    #    gs1 = gridspec.GridSpec(2,1) #(3, 2)#, width_ratios=[3, 1]) 
    #    gs1.update(bottom=.6, top=0.9, left=0.05, right=0.5, hspace=.5)    
        gs1 = gridspec.GridSpec(1,2) #(3, 2)#, width_ratios=[3, 1]) 
        gs1.update(bottom=.73, top=0.9, left=0.05, right=0.95, wspace=.8)
        #
        gs2 = gridspec.GridSpec(1,2) #(3, 2)#, width_ratios=[3, 1]) 
        gs2.update(bottom=.45, top=0.62, left=0.05, right=0.95, wspace=.8)    
        
        ##### Response measures; each depth ##### 
        # subplots of flash responses
        gs3 = gridspec.GridSpec(1,2) #, width_ratios=[3, 1]) 
        gs3.update(bottom=.22, top=0.35, left=0.05, right=0.55, wspace=.55, hspace=.5)
#        gs3.update(bottom=.17, top=0.3, left=0.05, right=0.43, wspace=.55, hspace=.5)
        # subplots of omission responses
        gs4 = gridspec.GridSpec(1,2)
        gs4.update(bottom=.05, top=0.18, left=0.05, right=0.55, wspace=.55, hspace=.5)
#        gs4.update(bottom=.17, top=0.3, left=0.57, right=0.95, wspace=.55, hspace=.5)
        
        ##### Response measures; compare areas (pooled depths) ##### 
        # subplots of flash responses
        gs5 = gridspec.GridSpec(1,2) #, width_ratios=[3, 1]) 
        gs5.update(bottom=.22, top=0.35, left=0.7, right=0.95, wspace=.55, hspace=.5)
#        gs3.update(bottom=.17, top=0.3, left=0.05, right=0.43, wspace=.55, hspace=.5)
        # subplots of omission responses
        gs6 = gridspec.GridSpec(1,2)
        gs6.update(bottom=.05, top=0.18, left=0.7, right=0.95, wspace=.55, hspace=.5)
#        gs4.update(bottom=.17, top=0.3, left=0.57, right=0.95, wspace=.55, hspace=.5)

        
        plt.suptitle('%s, %d total sessions: %s' %(cre, thisCre_pooledSessNum, session_labs_all), fontsize=14)
        
        
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
        for iplane in inds_v1: #np.arange(4,num_planes): #num_planes):
            
            ax = plt.subplot(gs1[0]) # plt.subplot(2,1,2)
    
            area = areas[iplane] # num_sess 
            depth = depths[iplane] # num_sess         
            lab = '%dum' %(np.mean(depth))
            
            h1_0 = plt.plot(time_trace_new, av_ts_eachPlane[iplane],color=cols_depth[iplane-num_planes], label=(lab), markersize=3.5)[0]
            h2 = plt.plot(time_trace_new, av_sh_eachPlane[iplane], color='gray', label='shfl', markersize=3.5)[0]
            
            # errorbars (standard error across cv samples)   
            plt.fill_between(time_trace_new, av_ts_eachPlane[iplane] - sd_ts_eachPlane[iplane], av_ts_eachPlane[iplane] + sd_ts_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane-num_planes], facecolor=cols_depth[iplane-num_planes])
            plt.fill_between(time_trace_new, av_sh_eachPlane[iplane] - sd_sh_eachPlane[iplane], av_sh_eachPlane[iplane] + sd_sh_eachPlane[iplane], alpha=alph, edgecolor='gray', facecolor='gray')
            
            h1.append(h1_0)


        # add to plots flash lines, ticks and legend
        plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
        plt.xlim(xlim);
        plt.title(np.unique(area)[0], fontsize=13, y=1)
        # mark peak_win: the window over which the response quantification (peak or mean) was computed 
        plt.hlines(lims[1], peak_win[0], peak_win[1], color='gray')
        plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')
#        plt.axvspan(peak_win[0], peak_win[1], alpha=0.3, facecolor='gray')
#        plt.axvspan(flash_win[0], flash_win[1], alpha=0.3, facecolor='green')


        ############################### LM ###############################
        h1 = []
        for iplane in inds_lm: #range(4): # iplane = 0
            ax = plt.subplot(gs1[1]) # plt.subplot(2,1,1)
    
            area = areas[iplane] # num_sess 
            depth = depths[iplane] # num_sess             
            lab = '%dum' %(np.mean(depth))
            
            h1_0 = plt.plot(time_trace_new, av_ts_eachPlane[iplane], color=cols_depth[iplane], label=(lab), markersize=3.5)[0]            
            h2 = plt.plot(time_trace_new, av_sh_eachPlane[iplane], color='gray', label='shfl', markersize=3.5)[0]            

            # errorbars (standard error across cv samples)   
            plt.fill_between(time_trace_new, av_ts_eachPlane[iplane] - sd_ts_eachPlane[iplane], av_ts_eachPlane[iplane] + sd_ts_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane], facecolor=cols_depth[iplane])            
            plt.fill_between(time_trace_new, av_sh_eachPlane[iplane] - sd_sh_eachPlane[iplane], av_sh_eachPlane[iplane] + sd_sh_eachPlane[iplane], alpha=alph, edgecolor='gray', facecolor='gray')

            h1.append(h1_0)
            
            
        # add to plots flash lines, ticks and legend
        plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, ylab=ylabel, xmjn=xmjn)
    #    plt.ylabel('')
        plt.xlim(xlim);
        plt.title(np.unique(area)[0], fontsize=13, y=1)
        # mark peak_win: the window over which the response quantification (peak or mean) was computed 
        plt.hlines(lims[1], peak_win[0], peak_win[1], color='gray')
        plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')
            

            
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
            h1_0 = plt.plot(time_trace_new, av_ts_pooled_eachArea[iarea], color = cols_area[iarea], label=(lab), markersize=3.5)[0]        
            h2 = plt.plot(time_trace_new, av_sh_pooled_eachArea[iarea], color='gray', label='shfl', markersize=3.5)[0]
            
            plt.fill_between(time_trace_new, av_ts_pooled_eachArea[iarea] - sd_ts_pooled_eachArea[iarea] , \
                             av_ts_pooled_eachArea[iarea] + sd_ts_pooled_eachArea[iarea], alpha=alph, edgecolor=cols_area[iarea], facecolor=cols_area[iarea])
            plt.fill_between(time_trace_new, av_sh_pooled_eachArea[iarea] - sd_sh_pooled_eachArea[iarea] , \
                             av_sh_pooled_eachArea[iarea] + sd_sh_pooled_eachArea[iarea], alpha=alph, edgecolor='gray', facecolor='gray')

            h1.append(h1_0)
        
        lims = np.array([np.nanmin(av_sh_pooled_eachArea - sd_sh_pooled_eachArea), \
                         np.nanmax(av_ts_pooled_eachArea + sd_ts_pooled_eachArea)])
        lims[0] = lims[0] - np.diff(lims) / 20.
        lims[1] = lims[1] + np.diff(lims) / 20.

        # add to plots flash lines, ticks and legend        
        plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
        #    plt.ylabel('')
        plt.xlim(xlim);
        # mark peak_win: the window over which the response quantification (peak or mean) was computed 
        plt.hlines(lims[1], peak_win[0], peak_win[1], color='gray')
        plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')
            
            
            
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
            h1_0 = plt.plot(time_trace_new, av_ts_pooled_eachDepth[idepth], color = cols_depth[idepth], label=(lab), markersize=3.5)[0]        
            h2 = plt.plot(time_trace_new, av_sh_pooled_eachDepth[idepth], color='gray', label='shfl', markersize=3.5)[0]
        
            plt.fill_between(time_trace_new, av_ts_pooled_eachDepth[idepth] - sd_ts_pooled_eachDepth[idepth] , \
                             av_ts_pooled_eachDepth[idepth] + sd_ts_pooled_eachDepth[idepth], alpha=alph, edgecolor=cols_depth[idepth], facecolor=cols_depth[idepth])            
            plt.fill_between(time_trace_new, av_sh_pooled_eachDepth[idepth] - sd_sh_pooled_eachDepth[idepth] , \
                             av_sh_pooled_eachDepth[idepth] + sd_sh_pooled_eachDepth[idepth], alpha=alph, edgecolor='gray', facecolor='gray')
            
            h1.append(h1_0)
        
        
        lims = np.array([np.nanmin(av_sh_pooled_eachDepth - sd_sh_pooled_eachDepth), \
                         np.nanmax(av_ts_pooled_eachDepth + sd_ts_pooled_eachDepth)])
        lims[0] = lims[0] - np.diff(lims) / 20.
        lims[1] = lims[1] + np.diff(lims) / 20.

        # add to plots flash lines, ticks and legend        
        plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
    #    plt.ylabel('')
        plt.xlim(xlim);
        # mark peak_win: the window over which the response quantification (peak or mean) was computed 
        plt.hlines(lims[1], peak_win[0], peak_win[1], color='gray')
        plt.hlines(lims[1], flash_win[0], flash_win[1], color='green')
        


        ####################################################################################
        ####################################################################################
        #%% Plot response amplitude and timing measures
        # per depth; two areas in two subplots       
        ####################################################################################
        ####################################################################################
        
        ## remember svm peak quantification vars are for training, testing, shuffle, and chance data: trTsShCh
        
        xlabs = 'Depth (um)'
        x = np.arange(num_depth)
        xticklabs = np.round(depth_ave).astype(int)
            
        
        #%% Flash-evoked responses

        # left plot: flash, response amplitude
        ylabs = 'Resp amplitude'
        
        top = np.nanmean(paf_all[:, cre_all[0,:]==cre], axis=1) # 8 x 4(train, test, shuffle, chance)
        top_sd = np.nanstd(paf_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(sum(cre_all[0,:]==cre))       
        
        # plot testing, and shuffle data, ie indeces 1 and 2 in top
#         ind_ts_sh = [1,2] (train, test, shuffle, chance)
        
        ax1 = plt.subplot(gs3[0])
        
        # testing data
        ax1.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax1.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

        # shuffled data
        ax1.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt='o', markersize=3, capsize=3, label='V1', color='gray')
        ax1.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt='o', markersize=3, capsize=3, label='LM', color='skyblue')
        
        plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Flash', fontsize=13.5, y=1)
        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
        plt.ylabel('Flash', fontsize=15, rotation=0, labelpad=35)
#        plt.text(3.2, text_y, 'Flash', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


        
        
        # right plot: flash, response timing
        ylabs = 'Peak timing (s)'
        
        top = np.nanmean(ptf_all[:, cre_all[0,:]==cre], axis=1)
        top_sd = np.nanstd(ptf_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(sum(cre_all[0,:]==cre))       

        ax2 = plt.subplot(gs3[1])
        
        # testing data
        ax2.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax2.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

        # shuffled data : we don't plot timing for shuffled data, bc they don't have a response!
#         ax2.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt='o', markersize=3, capsize=3, label='V1', color='gray')
#         ax2.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt='o', markersize=3, capsize=3, label='LM', color='skyblue')
        
#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Flash', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Flash', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)
                
            
            
        ########################################################
        #%% Omission-evoked responses

        # left plot: omission, response amplitude
        ylabs = 'Resp amplitude'
#        x = np.arange(num_depth)
        
        top = np.nanmean(pa_all[:, cre_all[0,:]==cre], axis=1)
        top_sd = np.nanstd(pa_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(sum(cre_all[0,:]==cre))        

        ax1 = plt.subplot(gs4[0])
        
        # testing data
        ax1.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax1.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

        # shuffled data
        ax1.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt='o', markersize=3, capsize=3, label='V1', color='gray')
        ax1.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt='o', markersize=3, capsize=3, label='LM', color='skyblue')

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
        plt.ylabel('Omission', fontsize=15, rotation=0, labelpad=35)
#        plt.text(3.2, text_y, 'Omission', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)
            
            
            
        # right plot: omission, response timing
        ylabs = 'Peak timing (s)'
        
        top = np.nanmean(pt_all[:, cre_all[0,:]==cre], axis=1)
        top_sd = np.nanstd(pt_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(sum(cre_all[0,:]==cre))        

        ax2 = plt.subplot(gs4[1])
        
        # testing data
        ax2.errorbar(x, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax2.errorbar(x + xgap_areas, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

        # shuffled data
#         ax2.errorbar(x, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt='o', markersize=3, capsize=3, label='V1', color='gray')
#         ax2.errorbar(x + xgap_areas, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt='o', markersize=3, capsize=3, label='LM', color='skyblue')
        
#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.xlabel(xlabs, fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
#        plt.title('%s' %(ylabs), fontsize=12)
#        plt.title('Flash', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Flash', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


            

        ####################################################################################
        ####################################################################################
        #%% Plot response amplitude and timing measures
        # pooled layers; compare two areas
        ####################################################################################
        ####################################################################################

        xlabs = 'Area'
        x = np.arange(len(distinct_areas))
        xticklabs = distinct_areas[[1,0]] # so we plot V1 first, then LM
        
        
        #%% Flash-evoked responses
        
        # left plot: flash response amplitude
        ylabs = 'Amplitude'
        
        top = np.nanmean(paf_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] # 2 x 4(train, test, shuffle, chance)
        top_sd = np.nanstd(paf_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] / np.sqrt(sum(cre_eachArea[0,:]==cre))       

        ax1 = plt.subplot(gs5[0])         

        # testing data
        ax1.errorbar(x, top[:,1], yerr=top_sd[:,1], fmt='o', markersize=3, capsize=3, color=cols_area[1])

        # shuffled data
        ax1.errorbar(x, top[:,2], yerr=top_sd[:,2], fmt='o', markersize=3, capsize=3, color='gray')

        plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Flash', fontsize=13.5, y=1)
        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
#        plt.ylabel('Flash', fontsize=15, rotation=0, labelpad=30)
#        plt.text(3.2, text_y, 'Flash', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


        
        # right plot: flash response timing
        ylabs = 'Timing (s)'
        
        top = np.nanmean(ptf_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]]
        top_sd = np.nanstd(ptf_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] / np.sqrt(sum(cre_eachArea[0,:]==cre))        

        ax2 = plt.subplot(gs5[1])
        
        # testing data
        ax2.errorbar(x, top[:,1], yerr=top_sd[:,1], fmt='o', markersize=3, capsize=3, color=cols_area[1])

        # shuffled data
#         ax2.errorbar(x, top[:,2], yerr=top_sd[:,2], fmt='o', markersize=3, capsize=3, color='gray')
        
#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Flash', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Flash', fontsize=15)
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)
            
            
            
        ########################################################
        #%% Omission-evoked responses

        # left: omission, response amplitude
        ylabs = 'Amplitude'
#        xlabs = 'Area'
#        x = np.arange(len(distinct_areas))
#        xticklabs = xticklabs
        
        top = np.nanmean(pa_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]]
        top_sd = np.nanstd(pa_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] / np.sqrt(sum(cre_eachArea[0,:]==cre))   

        ax1 = plt.subplot(gs6[0])  
        
        # testing data
        ax1.errorbar(x, top[:,1], yerr=top_sd[:,1], fmt='o', markersize=3, capsize=3, color=cols_area[1])

        # shuffled data
        ax1.errorbar(x, top[:,2], yerr=top_sd[:,2], fmt='o', markersize=3, capsize=3, color='gray')

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
#        plt.ylabel('Omission', fontsize=15, rotation=0, labelpad=30)
#        plt.text(3.2, text_y, 'Omission', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


        
        # right: omission, response timing
        ylabs = 'Timing (s)'
        
        top = np.nanmean(pt_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]]
        top_sd = np.nanstd(pt_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] / np.sqrt(sum(cre_eachArea[0,:]==cre))    

        ax2 = plt.subplot(gs6[1])
        
        # testing data
        ax2.errorbar(x, top[:,1], yerr=top_sd[:,1], fmt='o', markersize=3, capsize=3, color=cols_area[1])

        # shuffled data
#         ax2.errorbar(x, top[:,2], yerr=top_sd[:,2], fmt='o', markersize=3, capsize=3, color='gray')

        
#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.xlabel(xlabs, fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
#        plt.title('%s' %(ylabs), fontsize=12)
#        plt.title('Flash', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Flash', fontsize=15)
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)

        
        
        #%%
        if dosavefig:
            nam = '%s_aveMice%s_aveSessPooled_%s_%s' %(cre, whatSess, fgn, now)            
            fign = os.path.join(dir0, dir_now, nam+fmt)
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
            

            