#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vars needed here are set in omissions_traces_peaks_plots_setVars_ave.py

(Note: ABtransit codes right now do not make layer-pooled (per area) plots. To do so, just use svm_allMice_sessPooled same as when ABtransit is 0, except make plots for A, B separately, instead of averaged sessions)

Created on Thu Sep 12 18:37:08 2019
@author: farzaneh
"""

#%% SUMMARY ACROSS MICE 
##########################################################################################
##########################################################################################
############# Plot traces averaged across mice of the same cre line ################
##########################################################################################    
##########################################################################################

#%%
if all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==1: # plot average across mice for A sessions, and B sessions, separately.

    #%% Decide if you want to plot interpolated or original traces
    #(because I am using linear interpolation it makes no difference really!)
    
    trace_now = trace_ave_allCre_new
    #trace_now = trace_ave_allCre
    trace_sd_now = trace_sd_allCre_new
    #trace_sd_now = trace_sd_allCre
    xnow = time_trace_new
    #xnow = time_trace
    
    # use frames instead of time
    #xnow = np.unique(np.concatenate((np.arange(0, -(samps_bef+1), -1), np.arange(0, samps_aft))))


    #%% Plot the mouse-averaged traces for the two sessions (A, B), done separately for each cre line and each plane
    
#    xlim = [-1.2, 2.25] # [-13, 24]
    frame_dur_new = np.diff(xnow)[0]
    r = np.round(np.array(xlim) / frame_dur_new).astype(int)
    xlim_frs = np.arange(int(np.argwhere(xnow==0).squeeze()) + r[0], min(trace_now.shape[-1], int(np.argwhere(xnow==0).squeeze()) + r[1] + 1))
    
    #num_depth = 4
    bbox_to_anchor = (.92, .72)
    
    jet_cm = colorOrder(nlines=num_sessions)
    
    # set x tick marks                
    xmj = np.unique(np.concatenate((np.arange(0, time_trace[0], -1), np.arange(0, time_trace[-1], 1))))
    xmn = np.arange(.25, time_trace[-1], .5)
    xmjn = [xmj, xmn]
    
    # set ylim: same for all plots of all cre lines
    lims0 = [np.min(trace_now[:,:,xlim_frs] - trace_sd_now[:,:,xlim_frs]), np.max(trace_now[:,:,xlim_frs] + trace_sd_now[:,:,xlim_frs])]
    
    
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
        fa = plt.figure(figsize=(12, 5)) # 2* len(session_stages)
        plt.suptitle('%s, %d mice' %(cre, num_each_cre[icre]), y=1.1, fontsize=18) # .94    
        
        # subplots of peak amp (neuron averaged, per trial) and omit-aligned traces
        gs = gridspec.GridSpec(1, num_depth) #, width_ratios=[3, 1]) 
        gs.update(bottom=0.63, top=0.95, wspace=.6) #, hspace=1)
        # subplots of peak amp/ timing (trial-median)
        gs2 = gridspec.GridSpec(1, num_depth)
        gs2.update(bottom=0.05, top=0.37, wspace=0.6) #, hspace=1)
        '''
        
        ############################################################
        ########## Plot population averages ##########
        ############################################################        

        lims0 = [np.min(trace_now[icre,:,xlim_frs] - trace_sd_now[icre,:,xlim_frs]), np.max(trace_now[icre,:,xlim_frs] + trace_sd_now[icre,:,xlim_frs])]
            
            
        ############ Subplot 1: area V1 ############ (planes 4:8)
        for iplane in inds_v1: #np.arange(num_depth, num_planes): #num_planes): # iplane = 0
    
            area = trace_peak_allMice.iloc[0]['area'][iplane] # take it from the 1st mouse, this info is the same for all mice
            onePlane_allSess = np.arange(iplane, np.shape(trace_ave_allCre)[1], num_planes)     
            depth = depth_ave_allCre[icre, onePlane_allSess]
            
            area = np.unique(area)[0]
            depth = np.mean(depth)        
            
            top = trace_now[icre, onePlane_allSess] # num_sess x num_frs
            top_sd = trace_sd_now[icre, onePlane_allSess] # num_sess x num_frs
    
    
    
            ax1 = plt.subplot(gs[iplane-num_depth])        
            
            h = []
            for isess in range(num_sessions):            
                h0 = plt.plot(xnow, top.T[:,isess], color=jet_cm[isess], label=session_labs[isess])[0]
                h.append(h0)
                # errorbars (standard error across mice)   
                plt.fill_between(xnow, top.T[:,isess] - top_sd.T[:,isess], top.T[:,isess] + top_sd.T[:,isess], alpha=alph,  edgecolor=jet_cm[isess], facecolor=jet_cm[isess])
                
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
        for iplane in inds_lm: #range(num_depth): #num_planes): # iplane = 0
    
            area = trace_peak_allMice.iloc[0]['area'][iplane] # take it from the 1st mouse, this info is the same for all mice
            onePlane_allSess = np.arange(iplane, np.shape(trace_ave_allCre)[1], num_planes)
            depth = depth_ave_allCre[icre, onePlane_allSess]
            
            area = np.unique(area)[0]
            depth = np.mean(depth)
            
            top = trace_now[icre, onePlane_allSess] # num_sess x num_frs
            top_sd = trace_sd_now[icre, onePlane_allSess] # num_sess x num_frs
    
    
    
            ax2 = plt.subplot(gs2[iplane])     
               
            h = []
            for isess in range(num_sessions):
                h0 = plt.plot(xnow, top.T[:,isess], color=jet_cm[isess], label=session_labs[isess])[0]
                h.append(h0)
                # errorbars (standard error across mice)   
                plt.fill_between(xnow, top.T[:,isess] - top_sd.T[:,isess], top.T[:,isess] + top_sd.T[:,isess], alpha=alph, edgecolor=jet_cm[isess], facecolor=jet_cm[isess])
                
                
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
        xgap_areas = .2 # .3

        depth = depth_ave_allCre[icre]  # (8 x num_sessions)        

        inds_v1_lm = inds_v1, inds_lm
        labvl = 'V1 ', 'LM '
    
                
        ########################################################################
        #%% Image-evoked responses
        ########################################################################
        
        ###### Image: Response amplitude, V1, then LM ######

        y_ave_now = peak_amp_flash_ave_allCre # num_cre x (8 x num_sessions) 
        y_sd_now = peak_amp_flash_sd_allCre

        top = y_ave_now[icre] # (8 x num_sessions) 
        top_sd = y_sd_now[icre] 

        # same ylim for all layers and areas
        mn = np.min(top - top_sd)
        mx = np.max(top + top_sd)
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
            ax1.errorbar(x, top[indsA], yerr=top_sd[indsA], fmt='o', markersize=3, capsize=3, label='A', color='b')
            # session B
            ax1.errorbar(x+xgap_areas, top[indsB], yerr=top_sd[indsB], fmt='o', markersize=3, capsize=3, label='B', color='r')

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
    #        plt.title('Image', fontsize=13.5, y=1)
            ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
            if ivl==0:
                plt.ylabel('Image', fontsize=15, rotation=0, labelpad=35)
    #        plt.text(3.2, text_y, 'Image', fontsize=15)
            if ivl==0:
#                 ax1.legend(loc=3, bbox_to_anchor=(-.1, 1.2, 1, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
                plt.legend(loc='center left', bbox_to_anchor=(.97,.8), frameon=False, handlelength=1, fontsize=12)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)

            
                
        ################## Image: Response timing, V1, then LM ##################

        y_ave_now = peak_timing_flash_ave_allCre # num_cre x (8 x num_sessions) 
        y_sd_now = peak_timing_flash_sd_allCre

        top = y_ave_now[icre] # (8 x num_sessions) 
        top_sd = y_sd_now[icre] 

        # same ylim for all layers and areas
        mn = np.min(top - top_sd)
        mx = np.max(top + top_sd)
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
            ax1.errorbar(x, top[indsA], yerr=top_sd[indsA], fmt='o', markersize=3, capsize=3, label='A', color='b')
            # session B
            ax1.errorbar(x+xgap_areas, top[indsB], yerr=top_sd[indsB], fmt='o', markersize=3, capsize=3, label='B', color='r')

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
    #        plt.title('Image', fontsize=13.5, y=1)
            ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
#             plt.ylabel('Image', fontsize=15, rotation=0, labelpad=35)
    #        plt.text(3.2, text_y, 'Image', fontsize=15)
#             ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
    #        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)

            
    
        ########################################################################
        #%% Omission-evoked responses
        ########################################################################
        
        ###### Omission: Response amplitude, V1, then LM ######

        y_ave_now = peak_amp_omit_ave_allCre # num_cre x (8 x num_sessions) 
        y_sd_now = peak_amp_omit_sd_allCre

        top = y_ave_now[icre] # (8 x num_sessions) 
        top_sd = y_sd_now[icre] 

        # same ylim for all layers and areas
        mn = np.min(top - top_sd)
        mx = np.max(top + top_sd)
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
            ax1.errorbar(x, top[indsA], yerr=top_sd[indsA], fmt='o', markersize=3, capsize=3, label='A', color='b')
            # session B
            ax1.errorbar(x+xgap_areas, top[indsB], yerr=top_sd[indsB], fmt='o', markersize=3, capsize=3, label='B', color='r')

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

        y_ave_now = peak_timing_omit_ave_allCre # num_cre x (8 x num_sessions) 
        y_sd_now = peak_timing_omit_sd_allCre

        top = y_ave_now[icre] # (8 x num_sessions) 
        top_sd = y_sd_now[icre] 

        # same ylim for all layers and areas
        mn = np.min(top - top_sd)
        mx = np.max(top + top_sd)
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
            ax1.errorbar(x, top[indsA], yerr=top_sd[indsA], fmt='o', markersize=3, capsize=3, label='A', color='b')
            # session B
            ax1.errorbar(x+xgap_areas, top[indsB], yerr=top_sd[indsB], fmt='o', markersize=3, capsize=3, label='B', color='r')

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
#         d = np.reshape(depth_ave_allCre[icre], (num_planes, num_sessions), order='F') # 8 x num_sessions
#         depth_ave = np.mean(d, axis=1) # 8 # average across A,B sessions
        depth_ave = np.mean(np.reshape(depth_ave_allCre[icre], (num_depth, 2, num_sessions), order='F'), axis=(1,2))
                                                                                             
        # same ylim for all layers and areas
        mn = np.min(top - top_sd)
        mx = np.max(top + top_sd)
        r = mx - mn
        lims0 = [mn-r/20., mx+r/20.]  # set ylim: same for all plots of the same cre line
        
        xlabs = 'Depth (um)'
        x = np.arange(num_depth)
        xticklabs = np.round(depth_ave).astype(int)
            
        
        ###########################################
        #%% Image-evoked responses
        ###########################################
        
        ################# left plot: image, response amplitude #######################

        ylabs = 'Resp amplitude'
        
        y_ave_now = peak_amp_mod_flash_ave_allCre # num_cre x 8 x 4    
        y_sd_now = peak_amp_mod_flash_sd_allCre

        top = y_ave_now[icre] # 8 
        top_sd = y_sd_now[icre]         
        
        ax1 = plt.subplot(gs5[0])
        
        # testing data
        ax1.errorbar(x, top[inds_v1], yerr=top_sd[inds_v1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax1.errorbar(x + xgap_areas, top[inds_lm], yerr=top_sd[inds_lm], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

        plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Image', fontsize=13.5, y=1)
        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
        plt.ylabel('Image', fontsize=15, rotation=0, labelpad=35)
#        plt.text(3.2, text_y, 'Image', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)

        

        ################# right plot: flash, response timing #######################

        ylabs = 'Peak timing (s)'
        
        y_ave_now = peak_timing_mod_flash_ave_allCre # num_cre x 8 x 4    
        y_sd_now = peak_timing_mod_flash_sd_allCre

        top = y_ave_now[icre] # 8 
        top_sd = y_sd_now[icre] 
                
        ax2 = plt.subplot(gs5[1])
        
        # testing data
        ax2.errorbar(x, top[inds_v1], yerr=top_sd[inds_v1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax2.errorbar(x + xgap_areas, top[inds_lm], yerr=top_sd[inds_lm], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Image', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Image', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)
                
            
        
        ###########################################
        #%% Omission-evoked responses
        ###########################################
        
        ################# left plot: omission, response amplitude #######################

        ylabs = 'Resp amplitude'
        
        y_ave_now = peak_amp_mod_omit_ave_allCre # num_cre x 8 x 4    
        y_sd_now = peak_amp_mod_omit_sd_allCre

        top = y_ave_now[icre] # 8 
        top_sd = y_sd_now[icre]

        ax1 = plt.subplot(gs6[0])
        
        # testing data
        ax1.errorbar(x, top[inds_v1], yerr=top_sd[inds_v1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax1.errorbar(x + xgap_areas, top[inds_lm], yerr=top_sd[inds_lm], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

        plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.xlabel(xlabs, fontsize=12)
#        plt.ylim(lims0)
#        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
#        plt.title('%s' %(ylabs), fontsize=12)
#        plt.title('Image', fontsize=13.5, y=1)
        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
        plt.ylabel('Omission', fontsize=15, rotation=0, labelpad=35)
#        plt.text(3.2, text_y, 'Omission', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)
            
            
            
        ################# right plot: omission, response timing #######################

        ylabs = 'Peak timing'
        
        y_ave_now = peak_timing_mod_omit_ave_allCre # num_cre x 8 x 4    
        y_sd_now = peak_timing_mod_omit_sd_allCre

        top = y_ave_now[icre] # 8 
        top_sd = y_sd_now[icre]

        ax1 = plt.subplot(gs6[1])
        
        # testing data
        ax2.errorbar(x, top[inds_v1], yerr=top_sd[inds_v1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax2.errorbar(x + xgap_areas, top[inds_lm], yerr=top_sd[inds_lm], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.xlabel(xlabs, fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
#        plt.title('%s' %(ylabs), fontsize=12)
#        plt.title('Image', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Image', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)

    
    
        if dosavefig:
            nam = '%s_aveMice%s_%s_%s' %(cre, whatSess, fgn, now)
            fign = os.path.join(dir0, dir_now, nam+fmt)        
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


    
    ### Old codes
    """
    #%%     
    ##########################################################################################
    ##########################################################################################
    ############# Plot peak amplitude for omission and flash evoked responses ################
    ##########################################################################################    
    ##########################################################################################
    
    plotPeakTiming = 0 # if 0, plot peak amp; if 1, plot peak timing
    
    same_ylim_all_layers_areas = 1 # if 0, use the same ylim for the same depth of both areas (but different ylims for different depths)
    
    if plotPeakTiming:
        ylabs = 'Peak timing' #
        amp_time = 'peakTime' # needed for figure name 
    else:    
        ylabs = 'Resp amplitude' 
        amp_time = 'peakAmp' # needed for figure name #
    
                    
    for icre in range(len(all_cres)): # icre = 0
        
        cre = all_cres[icre]    
        
        fa = plt.figure(figsize=(8, 11)) # 2* len(session_stages)
        plt.suptitle('%s, %d mice' %(cre, num_each_cre[icre]), y=.99, fontsize=18) # .94    
        
        # subplots of flash responses
        gs = gridspec.GridSpec(num_depth,2) #, width_ratios=[3, 1]) 
        gs.update(left=0.05, right=0.4, wspace=.6, hspace=.5)
        # subplots of omission responses
        gs2 = gridspec.GridSpec(num_depth,2)
        gs2.update(left=0.6, right=0.95, wspace=.6, hspace=.5)
        
    
        
        ############################################################
        ############ Subplot gs2: omission-evoked responses ############ 
        ###############################################################
        if plotPeakTiming:
            y_ave_now = peak_timing_ave_allCre #peak_amp_ave_allCre
            y_sd_now = peak_timing_sd_allCre #peak_amp_sd_allCre
        else:
            y_ave_now = peak_amp_ave_allCre
            y_sd_now = peak_amp_sd_allCre
            
        # same ylim for all layers and areas
    #    lims0 = [np.min(y_ave_now[icre] - y_sd_now[icre]), np.max(y_ave_now[icre] + y_sd_now[icre])]  # set ylim: same for all plots of the same cre line
    
        
        ### Left column: area V1 (planes 4:8)
        for iplane in inds_v1: #np.arange(num_depth, num_planes): #num_planes): # iplane = 0
            
            area = trace_peak_allMice.iloc[0]['area'][iplane] # take it from the 1st mouse, this info is the same for all mice
            onePlane_allSess = np.arange(iplane, np.shape(trace_ave_allCre)[1], num_planes)
            depth = depth_ave_allCre[icre, onePlane_allSess]
            
            area = np.unique(area)[0]
            depth = np.mean(depth)        
    
            # use same ylim for the same depth of both areas 
            if same_ylim_all_layers_areas:
                onePlane_allSess_allAreas = np.concatenate((onePlane_allSess, onePlane_allSess - num_depth)) # do +num_depth if iplane is 0:3 (ie for area LM)
                lims0 = np.squeeze([np.min(y_ave_now[icre, onePlane_allSess_allAreas] - y_sd_now[icre, onePlane_allSess_allAreas]), \
                         np.max(y_ave_now[icre, onePlane_allSess_allAreas] + y_sd_now[icre, onePlane_allSess_allAreas])])  # set ylim: same for all plots of the same cre line
                lims0[0] = lims0[0] - np.diff(lims0) / 20.
                lims0[1] = lims0[1] + np.diff(lims0) / 20.
            
            
            top = y_ave_now[icre, onePlane_allSess] # num_sess
            top_sd = y_sd_now[icre, onePlane_allSess] # num_sess
    
    
            ax1 = plt.subplot(gs2[iplane-num_depth, 0]) 
               
            ax1.errorbar(range(num_sessions), top, yerr=top_sd, fmt='o', markersize=3, capsize=3)
            
            plt.hlines(0, 0, num_sessions-1, linestyle=':')
            ax1.set_xticks(range(num_sessions))
            ax1.set_xticklabels(session_labs, rotation=45)
            ax1.tick_params(labelsize=10)
            plt.xlim([-.5, num_sessions-.5])
            plt.ylim(lims0)
            ax1.set_ylabel('%s\n(%d um)' %(ylabs, depth), fontsize=12)
            if iplane==4:
                plt.title('%s' %area, fontsize=13.5, y=1)
                ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
                plt.text(1.2, text_y, 'Omission', fontsize=15)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)
             
    
    
        ### Right column: area LM (planes 0:3)
        for iplane in inds_lm: #range(num_depth): #num_planes): # iplane = 0
    
            area = trace_peak_allMice.iloc[0]['area'][iplane] # take it from the 1st mouse, this info is the same for all mice
            onePlane_allSess = np.arange(iplane, np.shape(trace_ave_allCre)[1], num_planes)
            depth = depth_ave_allCre[icre, onePlane_allSess]
            
            area = np.unique(area)[0]
            depth = np.mean(depth)        
            
            # use same ylim for the same depth of both areas 
            if same_ylim_all_layers_areas:
                onePlane_allSess_allAreas = np.concatenate((onePlane_allSess, onePlane_allSess - num_depth)) # do +num_depth if iplane is 0:3 (ie for area LM)
                lims0 = np.squeeze([np.min(y_ave_now[icre, onePlane_allSess_allAreas] - y_sd_now[icre, onePlane_allSess_allAreas]), \
                         np.max(y_ave_now[icre, onePlane_allSess_allAreas] + y_sd_now[icre, onePlane_allSess_allAreas])])  # set ylim: same for all plots of the same cre line
                lims0[0] = lims0[0] - np.diff(lims0) / 20.
                lims0[1] = lims0[1] + np.diff(lims0) / 20.
                
                
            top = y_ave_now[icre, onePlane_allSess] # num_sess
            top_sd = y_sd_now[icre, onePlane_allSess] # num_sess
    
            
            ax2 = plt.subplot(gs2[iplane, 1])
               
            ax2.errorbar(range(num_sessions), top, yerr=top_sd, fmt='o', markersize=3, capsize=3)
            
            plt.hlines(0, 0, num_sessions-1, linestyle=':')
            ax2.set_xticks(range(num_sessions))
            ax2.set_xticklabels(session_labs, rotation=45)
            ax2.tick_params(labelsize=10)
            plt.xlim([-.5, num_sessions-.5])
            plt.ylim(lims0)
    #        ax2.set_ylabel('Resp amp\n%d um' %depth, fontsize=12)
            if iplane==0:
                plt.title('%s' %area, fontsize=13.5, y=1)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)
    
            
        #%% Plot change in peak amplitude from A to B sessions (B minus A)
        
        ### This part still needs work... it will give good summary plots for the two areas (all depth on the x axis)
        
        '''
#        TO DO:
#            1. load the allsess file that gives the plots that make more sense in terms of quantifications!
#            2. edit the definition of diffAB_sd below... you have to compute sd on diff traces!
#            3. finish the plot
            
        d = np.reshape(depth_ave_allCre, (depth_ave_allCre.shape[0], num_planes, 2), order='F') # num_cres x 8 x 2 (2: A and B sessions: 2 sessions)       
        d_ave = np.mean(d, axis=2) # num_cre x 8 (average of depth for each plane across the two sessions)
        d_aveAreas = np.mean(np.reshape(d_ave[icre], (num_depth, 2), order='F'), axis=1).astype(int) # 4 # average of depth across the two areas
        
        a = np.reshape(y_ave_now, (y_ave_now.shape[0], num_planes, 2), order='F') # num_cres x 8 x 2
        diffAB_ave = np.diff(a, axis=2).squeeze() # difference from A to B # num_cre x 8
        a = np.reshape(y_sd_now, (y_ave_now.shape[0], num_planes, 2), order='F') # num_cres x 8 x 2
        diffAB_sd = np.diff(a, axis=2).squeeze() # difference from A to B # num_cre x 8

        lims0 = np.squeeze([np.min(y_ave_now[icre, onePlane_allSess_allAreas] - y_sd_now[icre, onePlane_allSess_allAreas]), \
                 np.max(y_ave_now[icre, onePlane_allSess_allAreas] + y_sd_now[icre, onePlane_allSess_allAreas])])  # set ylim: same for all plots of the same cre line
        lims0[0] = lims0[0] - np.diff(lims0) / 20.
        lims0[1] = lims0[1] + np.diff(lims0) / 20.
        
#         slc = np.reshape(aa[0], (num_depth, 2), order='F')
#         sst = np.reshape(aa[1], (num_depth, 2), order='F')
#         vip = np.reshape(aa[2], (num_depth, 2), order='F')

        top = diffAB_ave[icre, inds_v1] # num_sess
        top_sd = diffAB_sd[icre, inds_v1] # num_sess

        ax1 = plt.subplot(1,1,1) #(gs2[iplane-num_depth, 0]) 

        ax1.errorbar(range(num_depth), top, yerr=top_sd, fmt='o', markersize=3, capsize=3)

        plt.hlines(0, 0, num_depth-1, linestyle=':')
        ax1.set_xticks(range(num_depth))
        ax1.set_xticklabels(d_aveAreas, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, num_depth-.5])
        plt.ylim(lims0)
        
        ax1.set_ylabel('%s\n(%d um)' %(ylabs, depth), fontsize=12)
        if iplane==4:
            plt.title('%s' %area, fontsize=13.5, y=1)
            ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
            plt.text(1.2, text_y, 'Omission', fontsize=15)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)
        '''

        #%%    
        ############################################################
        ############ Subplot gs: flash-evoked responses ############ 
        ############################################################
        if plotPeakTiming:
            y_ave_now = peak_timing_flash_ave_allCre #peak_amp_ave_allCre
            y_sd_now = peak_timing_flash_sd_allCre #peak_amp_sd_allCre
        else:
            y_ave_now = peak_amp_flash_ave_allCre
            y_sd_now = peak_amp_flash_sd_allCre
            
        # same ylim for all layers and areas
    #    lims0 = [np.min(y_ave_now[icre] - y_sd_now[icre]), np.max(y_ave_now[icre] + y_sd_now[icre])]  # set ylim: same for all plots of the same cre line
    
        
        ### Left column: area V1 (planes 4:8)
        for iplane in np.arange(num_depth, num_planes): #num_planes): # iplane = 0
            
            area = trace_peak_allMice.iloc[0]['area'][iplane] # take it from the 1st mouse, this info is the same for all mice
            onePlane_allSess = np.arange(iplane, np.shape(trace_ave_allCre)[1], num_planes)
            depth = depth_ave_allCre[icre, onePlane_allSess]
            
            area = np.unique(area)[0]
            depth = np.mean(depth)        
    
            # use same ylim for the same depth of both areas 
            if same_ylim_all_layers_areas:
                onePlane_allSess_allAreas = np.concatenate((onePlane_allSess, onePlane_allSess - num_depth)) # do +num_depth if iplane is 0:3 (ie for area LM)
                lims0 = np.squeeze([np.min(y_ave_now[icre, onePlane_allSess_allAreas] - y_sd_now[icre, onePlane_allSess_allAreas]), \
                         np.max(y_ave_now[icre, onePlane_allSess_allAreas] + y_sd_now[icre, onePlane_allSess_allAreas])])  # set ylim: same for all plots of the same cre line
                lims0[0] = lims0[0] - np.diff(lims0) / 20.
                lims0[1] = lims0[1] + np.diff(lims0) / 20.
            
            
            top = y_ave_now[icre, onePlane_allSess] # num_sess
            top_sd = y_sd_now[icre, onePlane_allSess] # num_sess
    
    
            ax1 = plt.subplot(gs[iplane-num_depth, 0]) 
               
            ax1.errorbar(range(num_sessions), top, yerr=top_sd, fmt='o', markersize=3, capsize=3)
            
            plt.hlines(0, 0, num_sessions-1, linestyle=':')
            ax1.set_xticks(range(num_sessions))
            ax1.set_xticklabels(session_labs, rotation=45)
            ax1.tick_params(labelsize=10)
            plt.xlim([-.5, num_sessions-.5])
            plt.ylim(lims0)
            ax1.set_ylabel('%s\n(%d um)' %(ylabs, depth), fontsize=12)
            if iplane==4:
                plt.title('%s' %area, fontsize=13.5, y=1)
                ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
                plt.text(1.2, text_y, 'Image', fontsize=15)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)
             
    
    
        ### Right column: area LM (planes 0:3)
        for iplane in range(num_depth): #num_planes): # iplane = 0
    
            area = trace_peak_allMice.iloc[0]['area'][iplane] # take it from the 1st mouse, this info is the same for all mice
            onePlane_allSess = np.arange(iplane, np.shape(trace_ave_allCre)[1], num_planes)
            depth = depth_ave_allCre[icre, onePlane_allSess]
            
            area = np.unique(area)[0]
            depth = np.mean(depth)        
            
            # use same ylim for the same depth of both areas 
            if same_ylim_all_layers_areas:
                onePlane_allSess_allAreas = np.concatenate((onePlane_allSess, onePlane_allSess - num_depth)) # do +num_depth if iplane is 0:3 (ie for area LM)
                lims0 = np.squeeze([np.min(y_ave_now[icre, onePlane_allSess_allAreas] - y_sd_now[icre, onePlane_allSess_allAreas]), \
                         np.max(y_ave_now[icre, onePlane_allSess_allAreas] + y_sd_now[icre, onePlane_allSess_allAreas])])  # set ylim: same for all plots of the same cre line
                lims0[0] = lims0[0] - np.diff(lims0) / 20.
                lims0[1] = lims0[1] + np.diff(lims0) / 20.
                
                
            top = y_ave_now[icre, onePlane_allSess] # num_sess
            top_sd = y_sd_now[icre, onePlane_allSess] # num_sess
    
    
            ax2 = plt.subplot(gs[iplane, 1])     
               
            ax2.errorbar(range(num_sessions), top, yerr=top_sd, fmt='o', markersize=3, capsize=3)
            
            plt.hlines(0, 0, num_sessions-1, linestyle=':')
            ax2.set_xticks(range(num_sessions))
            ax2.set_xticklabels(session_labs, rotation=45)
            ax2.tick_params(labelsize=10)
            plt.xlim([-.5, num_sessions-.5])
            plt.ylim(lims0)
    #        ax2.set_ylabel('Resp amp\n%d um' %depth, fontsize=12)
            if iplane==0:
                plt.title('%s' %area, fontsize=13.5, y=1)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)
        
    
        #%%        
        if dosavefig:
            nam = '%s_aveMice%s_%s_all_planes_%s_omit_flash_%s' %(cre, whatSess, fgn, amp_time, now)
            fign = os.path.join(dir0, dir_now, nam+fmt)        
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
        
    """
            
            
            

#%%
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################                        
#%% Not A-B transitions
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################            
#trace_peak_allMice_sessPooled
#trace_peak_allMice_sessAvSd

else: 

    #%% Plot omission-aligned traces
    
    cre_all = trace_peak_allMice_sessPooled['cre_allPlanes']
    cre_lines = np.unique(cre_all)
    a_all = trace_peak_allMice_sessPooled['area_allPlanes']  # 8 x pooledSessNum 
    d_all = trace_peak_allMice_sessPooled['depth_allPlanes']  # 8 x pooledSessNum 
    session_labs_all = trace_peak_allMice_sessPooled['session_labs']    
    t_all = trace_peak_allMice_sessPooled['trace_allPlanes']  # 8 x pooledSessNum x 80
    pa_all = trace_peak_allMice_sessPooled['peak_amp_allPlanes']  # 8 x pooledSessNum     
    pt_all = trace_peak_allMice_sessPooled['peak_timing_allPlanes']  # 8 x pooledSessNum     
    paf_all = trace_peak_allMice_sessPooled['peak_amp_flash_allPlanes']  # 8 x pooledSessNum     
    ptf_all = trace_peak_allMice_sessPooled['peak_timing_flash_allPlanes']  # 8 x pooledSessNum     
    
    cre_eachArea = trace_peak_allMice_sessPooled['cre_eachArea'] # 2 x (4*sum(num_sess_per_mouse))
    t_eachArea = trace_peak_allMice_sessPooled['trace_eachArea'] # 2 x (4*sum(num_sess_per_mouse)) x 80
    pa_eachArea = trace_peak_allMice_sessPooled['peak_amp_eachArea'] # 2 x (4*sum(num_sess_per_mouse))
    pt_eachArea = trace_peak_allMice_sessPooled['peak_timing_eachArea'] # 2 x (4*sum(num_sess_per_mouse))
    paf_eachArea = trace_peak_allMice_sessPooled['peak_amp_flash_eachArea'] # 2 x (4*sum(num_sess_per_mouse))
    ptf_eachArea = trace_peak_allMice_sessPooled['peak_timing_flash_eachArea'] # 2 x (4*sum(num_sess_per_mouse))
    
    cre_eachDepth = trace_peak_allMice_sessPooled['cre_eachDepth'] # 4 x (2*sum(num_sess_per_mouse))
    t_eachDepth = trace_peak_allMice_sessPooled['trace_eachDepth'] # 4 x (2*sum(num_sess_per_mouse)) x 80
    depth_eachDepth = trace_peak_allMice_sessPooled['depth_eachDepth'] # 4 x (2*sum(num_sess_per_mouse)) x 80
#    pa_eachDepth = trace_peak_allMice_sessPooled['peak_amp_eachDepth'] # 4 x (2*sum(num_sess_per_mouse))
#    pt_eachDepth = trace_peak_allMice_sessPooled['peak_timing_eachDepth'] # 4 x (2*sum(num_sess_per_mouse))
#    paf_eachDepth = trace_peak_allMice_sessPooled['peak_amp_flash_eachDepth'] # 4 x (2*sum(num_sess_per_mouse))
#    ptf_eachDepth = trace_peak_allMice_sessPooled['peak_timing_flash_eachDepth'] # 4 x (2*sum(num_sess_per_mouse))
    

    xgap_areas = .3
    
    cols = ['cre', 'event', 'depth', 'p-value', 'ave_resp_V1_LM']
    p_areas_each_depth = pd.DataFrame([], columns = cols)
    ittest = -1

    #%% Plot averages of all mice in each cell line
    
    for icre in range(len(cre_lines)):
        
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
        a = t_all[:, cre_all[0,:]==cre]  # 8 x thisCre_pooledSessNum x 80
        av_trace_eachPlane = np.nanmean(a, axis=1) # 8 x 80
        sd_trace_eachPlane = np.nanstd(a, axis=1) / np.sqrt(np.sum(~np.isnan(a[:,:,0]), axis=1))[:,np.newaxis] #(a.shape[1]) # 8 x 80
    
        areas = a_all[:, cre_all[0,:]==cre]  # 8 x thisCre_pooledSessNum
        depths = d_all[:, cre_all[0,:]==cre]  # 8 x thisCre_pooledSessNum

        lims = np.array([np.nanmin(av_trace_eachPlane - sd_trace_eachPlane), np.nanmax(av_trace_eachPlane + sd_trace_eachPlane)])
        lims[0] = lims[0] - np.diff(lims) / 20.
        lims[1] = lims[1] + np.diff(lims) / 20.
        
        ############################### V1 ###############################
        h1 = []
        for iplane in inds_v1: #np.arange(4,num_planes): #num_planes):
            
            ax = plt.subplot(gs1[0]) # plt.subplot(2,1,2)
    
            area = areas[iplane] # num_sess 
            depth = depths[iplane] # num_sess         
            lab = '%dum' %(np.mean(depth))
            
            h1_0 = plt.plot(time_trace, av_trace_eachPlane[iplane],color=cols_depth[iplane-num_planes], label=(lab), markersize=3.5)[0]
            # errorbars (standard error across cv samples)   
            plt.fill_between(time_trace, av_trace_eachPlane[iplane] - sd_trace_eachPlane[iplane], av_trace_eachPlane[iplane] + sd_trace_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane-num_planes], facecolor=cols_depth[iplane-num_planes])
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
            
            h1_0 = plt.plot(time_trace, av_trace_eachPlane[iplane], color=cols_depth[iplane], label=(lab), markersize=3.5)[0]            
            # errorbars (standard error across cv samples)   
            plt.fill_between(time_trace, av_trace_eachPlane[iplane] - sd_trace_eachPlane[iplane], av_trace_eachPlane[iplane] + sd_trace_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane], facecolor=cols_depth[iplane])            
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
        a = t_eachArea[:, cre_eachArea[0,:]==cre]  # 2 x (4*thisCre_pooledSessNum) x 80
        av_trace_pooled_eachArea = np.nanmean(a, axis=1) # 2 x 80
        sd_trace_pooled_eachArea = np.nanstd(a, axis=1) / np.sqrt(np.sum(~np.isnan(a[:,:,0]), axis=1))[:,np.newaxis] #(a.shape[1]) # 2 x 80
        
        ##################
        ax = plt.subplot(gs2[0])
        #    ax = plt.subplot(gs1[2,0])
        plt.title('Sessions and layers pooled', fontsize=13, y=1)
        
        h1 = []
        for iarea in [1,0]: # first plot V1 then LM  #range(a.shape[0]):
            lab = '%s' %(distinct_areas[iarea])
            h1_0 = plt.plot(time_trace, av_trace_pooled_eachArea[iarea], color = cols_area[iarea], label=(lab), markersize=3.5)[0]        
            plt.fill_between(time_trace, av_trace_pooled_eachArea[iarea] - sd_trace_pooled_eachArea[iarea] , \
                             av_trace_pooled_eachArea[iarea] + sd_trace_pooled_eachArea[iarea], alpha=alph, edgecolor=cols_area[iarea], facecolor=cols_area[iarea])
            h1.append(h1_0)
        
        lims = np.array([np.nanmin(av_trace_pooled_eachArea - sd_trace_pooled_eachArea), \
                         np.nanmax(av_trace_pooled_eachArea + sd_trace_pooled_eachArea)])
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
        a = t_eachDepth[:, cre_eachDepth[0,:]==cre]  # 4 x (2*thisCre_pooledSessNum) x 80
        av_trace_pooled_eachDepth = np.nanmean(a, axis=1) # 4 x 80
        sd_trace_pooled_eachDepth = np.nanstd(a, axis=1) / np.sqrt(np.sum(~np.isnan(a[:,:,0]), axis=1))[:,np.newaxis] #(a.shape[1]) # 4 x 80
    
        depth_ave = np.mean(depth_eachDepth[:, cre_eachDepth[0,:]==cre], axis=1).astype(float) # average across areas and sessions
        
        ##################
        ax = plt.subplot(gs2[1])
    #    ax = plt.subplot(gs1[2,1])
        plt.title('Sessions and areas pooled', fontsize=13, y=1)
        
        h1 = []
        for idepth in range(a.shape[0]):
            lab = '%d um' %(depth_ave[idepth])
            h1_0 = plt.plot(time_trace, av_trace_pooled_eachDepth[idepth], color = cols_depth[idepth], label=(lab), markersize=3.5)[0]        
            plt.fill_between(time_trace, av_trace_pooled_eachDepth[idepth] - sd_trace_pooled_eachDepth[idepth] , \
                             av_trace_pooled_eachDepth[idepth] + sd_trace_pooled_eachDepth[idepth], alpha=alph, edgecolor=cols_depth[idepth], facecolor=cols_depth[idepth])            
            h1.append(h1_0)
        
        
        lims = np.array([np.nanmin(av_trace_pooled_eachDepth - sd_trace_pooled_eachDepth), \
                         np.nanmax(av_trace_pooled_eachDepth + sd_trace_pooled_eachDepth)])
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
        ####################################################################################
        #%% Plot response amplitude and timing measures
        # per depth
        ####################################################################################
        ####################################################################################
        ####################################################################################

        xlabs = 'Depth (um)'
        x = np.arange(num_depth)
        xticklabs = np.round(depth_ave).astype(int)
            
        top = np.concatenate((np.nanmean(paf_all[:, cre_all[0,:]==cre], axis=1), np.nanmean(pa_all[:, cre_all[0,:]==cre], axis=1)))
        aa = paf_all[:, cre_all[0,:]==cre]
        pafn = np.sum(~np.isnan(aa), axis=1)
        aa = pa_all[:, cre_all[0,:]==cre]
        pan = np.sum(~np.isnan(aa), axis=1)        
        top_sd = np.concatenate((np.nanstd(paf_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(pafn), np.nanstd(pa_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(pan)))        
        
        # same ylim for images and omissions
        mn = np.min(top - top_sd)
        mx = np.max(top + top_sd)
        r = mx - mn
        lims0 = [mn-r/20., mx+r/20.]  # set ylim: same for all plots of the same cre line
        
        

        
        ####################################################################################
        ####################################################################################
        ### ANOVA on response quantifications (image and omission evoked response amplitudes)
        ####################################################################################
        ####################################################################################    

        exec(open("anova_tukey_ttest_resp_amplitude.py").read())
                                            
                
        ########################################################
        ########################################################
        ########################################################
        #########%% Image-evoked responses ########
        ########################################################
        
        ######## left plot: flash, response amplitude
        ylabs = 'Resp amplitude'
        
        top = np.nanmean(paf_all[:, cre_all[0,:]==cre], axis=1)
        top_sd = np.nanstd(paf_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(pafn) # some are nan; you should divide it by the number of valid experiments!       

        ax1 = plt.subplot(gs3[0])         
        ax1.errorbar(x, top[inds_v1], yerr=top_sd[inds_v1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax1.errorbar(x + xgap_areas, top[inds_lm], yerr=top_sd[inds_lm], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

#         plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
        if same_y_fo:
            plt.ylim(lims0)
#        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Image', fontsize=13.5, y=1)
        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
        plt.ylabel('Image', fontsize=15, rotation=0, labelpad=35)
#        plt.text(3.2, text_y, 'Image', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)

        
        ######### right plot: image, response timing
        ylabs = 'Peak timing (s)'
        
        top = np.nanmean(ptf_all[:, cre_all[0,:]==cre], axis=1)
        top_sd = np.nanstd(ptf_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(pafn)                

        ax2 = plt.subplot(gs3[1])
        ax2.errorbar(x, top[inds_v1], yerr=top_sd[inds_v1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax2.errorbar(x + xgap_areas, top[inds_lm], yerr=top_sd[inds_lm], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])
        
#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Image', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Image', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)
        

        
        ########################################################
        #%% Omission-evoked responses

        ######### left plot: omission, response amplitude
        ylabs = 'Resp amplitude'
#        x = np.arange(num_depth)
        
        top = np.nanmean(pa_all[:, cre_all[0,:]==cre], axis=1)
        top_sd = np.nanstd(pa_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(pan)                

        ax1 = plt.subplot(gs4[0])         
        ax1.errorbar(x, top[inds_v1], yerr=top_sd[inds_v1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax1.errorbar(x + xgap_areas, top[inds_lm], yerr=top_sd[inds_lm], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])

#         plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.xlabel(xlabs, fontsize=12)
        if same_y_fo:
            plt.ylim(lims0)
#        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
#        plt.title('%s' %(ylabs), fontsize=12)
#        plt.title('Image', fontsize=13.5, y=1)
        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
        plt.ylabel('Omission', fontsize=15, rotation=0, labelpad=35)
#        plt.text(3.2, text_y, 'Omission', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


        
        ######### right plot: omission, response timing
        ylabs = 'Peak timing (s)'
        
        top = np.nanmean(pt_all[:, cre_all[0,:]==cre], axis=1)
        top_sd = np.nanstd(pt_all[:, cre_all[0,:]==cre], axis=1) / np.sqrt(pan)                

        ax2 = plt.subplot(gs4[1])
        ax2.errorbar(x, top[inds_v1], yerr=top_sd[inds_v1], fmt='o', markersize=3, capsize=3, label='V1', color=cols_area[1])
        ax2.errorbar(x + xgap_areas, top[inds_lm], yerr=top_sd[inds_lm], fmt='o', markersize=3, capsize=3, label='LM', color=cols_area[0])
        
#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.xlabel(xlabs, fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
#        plt.title('%s' %(ylabs), fontsize=12)
#        plt.title('Image', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Image', fontsize=15)
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
        
        aa = paf_eachArea[:, cre_eachArea[0,:]==cre]
        pafan = np.sum(~np.isnan(aa), axis=1)
        aa = pa_eachArea[:, cre_eachArea[0,:]==cre]
        paan = np.sum(~np.isnan(aa), axis=1)
        
        
        #%% Image-evoked responses
        
        # left plot: flash response amplitude
        ylabs = 'Amplitude'
        
        top = np.nanmean(paf_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]]
        top_sd = np.nanstd(paf_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] / np.sqrt(pafan) #sum(cre_eachArea[0,:]==cre))      

        ax1 = plt.subplot(gs5[0])         
        ax1.errorbar(x, top, yerr=top_sd, fmt='o', markersize=3, capsize=3, color=cols_area[1])

#         plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Image', fontsize=13.5, y=1)
        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
#        plt.ylabel('Image', fontsize=15, rotation=0, labelpad=30)
#        plt.text(3.2, text_y, 'Image', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


        
        # right plot: flash response timing
        ylabs = 'Timing (s)'
        
        top = np.nanmean(ptf_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]]
        top_sd = np.nanstd(ptf_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] / np.sqrt(pafan) #sum(cre_eachArea[0,:]==cre))        

        ax2 = plt.subplot(gs5[1])
        ax2.errorbar(x, top, yerr=top_sd, fmt='o', markersize=3, capsize=3, color=cols_area[1])
        
#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
#        plt.xlabel('Depth', fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
        plt.title('%s' %(ylabs), fontsize=13) #12
#        plt.title('Image', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Image', fontsize=15)
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)

        
        ########################################################
        #%% Omission-evoked responses

        # left: response amplitude
        ylabs = 'Amplitude'
#        xlabs = 'Area'
#        x = np.arange(len(distinct_areas))
#        xticklabs = xticklabs
        
        top = np.nanmean(pa_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]]
        top_sd = np.nanstd(pa_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] / np.sqrt(paan) #sum(cre_eachArea[0,:]==cre))            

        ax1 = plt.subplot(gs6[0])         
        ax1.errorbar(x, top, yerr=top_sd, fmt='o', markersize=3, capsize=3, color=cols_area[1])

#         plt.hlines(0, 0, len(x)-1, linestyle=':')
        ax1.set_xticks(x)
        ax1.set_xticklabels(xticklabs, rotation=45)
        ax1.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.xlabel(xlabs, fontsize=12)
#        plt.ylim(lims0)
#        ax1.set_ylabel('%s' %(ylabs), fontsize=12)
#        plt.title('%s' %(ylabs), fontsize=12)
#        plt.title('Image', fontsize=13.5, y=1)
        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3
#        plt.ylabel('Omission', fontsize=15, rotation=0, labelpad=30)
#        plt.text(3.2, text_y, 'Omission', fontsize=15)
#        ax1.legend(loc=3, bbox_to_anchor=(-.1, 1, 2, .1), ncol=2, frameon=False, mode='expand', borderaxespad=0, fontsize=12, handletextpad=.5) # handlelength=1, 
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)


        
        # right: response timing
        ylabs = 'Timing (s)'
        
        top = np.nanmean(pt_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]]
        top_sd = np.nanstd(pt_eachArea[:, cre_eachArea[0,:]==cre], axis=1)[[1,0]] / np.sqrt(paan) #sum(cre_eachArea[0,:]==cre))             

        ax2 = plt.subplot(gs6[1])
        ax2.errorbar(x, top, yerr=top_sd, fmt='o', markersize=3, capsize=3, color=cols_area[1])
        
#        plt.hlines(0, 0, num_sessions-1, linestyle=':')
        ax2.set_xticks(x)
        ax2.set_xticklabels(xticklabs, rotation=45)
        ax2.tick_params(labelsize=10)
        plt.xlim([-.5, len(x)-.5])
        plt.xlabel(xlabs, fontsize=12)
#        plt.ylim(lims0)
#        ax2.set_ylabel('%s' %(ylabs), fontsize=12)
#        plt.title('%s' %(ylabs), fontsize=12)
#        plt.title('Image', fontsize=13.5, y=1)
#        ylim = plt.gca().get_ylim(); text_y = ylim[1] + np.diff(ylim)/3.
#        plt.text(1.2, text_y, 'Image', fontsize=15)
#        plt.legend(loc='center left', bbox_to_anchor=bb, frameon=False, handlelength=1, fontsize=12)
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
        seaborn.despine()#left=True, bottom=True, right=False, top=False)

        
        
        #%%
        if dosavefig:
            nam = '%s_aveMice%s_aveSessPooled_%s_%s' %(cre, whatSess, fgn, now)
            fign = os.path.join(dir0, dir_now, nam+fmt)        
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
            

