#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make plots for individual mice.

Vars are set in omissions_traces_peaks_plots_setVars_ave.py

To get summary of mice plots run the following:
omissions_traces_peaks_plots_sumMice.py


Created on Thu Sep 12 18:35:34 2019
@author: farzaneh
"""

        
#%% Single-mouse plots
########################################################################################################
########################################################################################################
#### Plots omission-evoked peak responses (median across neurons) for each trial in the session. #######
#### A figure is made for each mouse, and each plane. Each figure includes subplots for each session. ##
########################################################################################################      
########################################################################################################
        
for im in range(len(all_mice_id)): # im=9
    
    #%%
    mouse_id = all_mice_id[im]

    if sum((all_sess_2an['mouse_id']==mouse_id).values) > 0: # make sure there is data for this mouse

        print(f'There is data for mouse {im}')
        
        all_sess_2an_this_mouse = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]
        peak_trace_this_mouse = trace_peak_med_iqr_0[trace_peak_med_iqr_0['mouse_id']==mouse_id]
        
        cre = all_sess_2an_this_mouse['cre'].iloc[0]
        cre = cre[:cre.find('-')] # remove the IRES-Cre part
                    
        session_stages = all_sess_2an_this_mouse['stage'].values[np.arange(0, all_sess_2an_this_mouse.shape[0], num_planes)]
#        session_stages = mouse_trainHist_all[mouse_trainHist_all['mouse_id']==mouse_id]['stage'].values
        num_sessions = len(session_stages)
        
        areas = all_sess_2an_this_mouse['area'] # (8*num_sessions)
        depths = all_sess_2an_this_mouse['depth'] # (8*num_sessions)
        planes = all_sess_2an_this_mouse.index.values # (8*num_sessions)
        
        ######## Traces ########
        y = np.array(np.squeeze(peak_trace_this_mouse['med_peak_eachTr'])) # 8 x trials: if the mouse has only 1 session; # if more than one session: size: total number of planes (8*num_sessions), and then each element has size (trials,) 
        y_trace = np.array(np.squeeze(peak_trace_this_mouse['med_trace'])) # total number of planes (8*num_sessions) x time 
    
        
        ######## Get the trial-median peak data for each neuron (peak computed on individual trials, then, median computed across trials)
        # remember: the first 8 are for the first session, the next 8 are for the next session..
        peak_amp_ns = np.array(np.squeeze(peak_trace_this_mouse['med_peak_amp_eachN'])) # size: total number of planes (8*num_sessions), and then each element has size (neurons,) 
        peak_timing_ns = np.array(np.squeeze(peak_trace_this_mouse['med_peak_timing_eachN']))
    
        # for each session, compute med and iqr across neurons  # or plot hist of peak of all sessions    
        peak_amp_med_ns, peak_amp_25q_ns, peak_amp_75q_ns = med_perc_each_exp(peak_amp_ns) # total number of planes (8*num_sessions); each element: med of peak amp across neurons in that plane
        peak_timing_med_ns, peak_timing_25q_ns, peak_timing_75q_ns = med_perc_each_exp(peak_timing_ns) # total number of planes (8*num_sessions); each element: med of peak timing across neurons in that plane
        
        
        ######## Peak amplitude, computed on trial-median *traces* ########    # get the peak measures computed on trial-median traces (above is computed on individual trials, but then median is taken across trials; the measure here is less noisy as the peaks are computed on the meidan of the traces.)
        peak_amp_ns_traceMed = all_sess_2an_this_mouse['peak_amp_eachN_traceMed'].values # size: total number of planes (8*num_sessions), and then each element has size (neurons,) 
        peak_timing_ns_traceMed = all_sess_2an_this_mouse['peak_timing_eachN_traceMed'].values
    
        # for each session, compute med and iqr across neurons  # or plot hist of peak of all sessions    
        peak_amp_med_ns_traceMed, peak_amp_25q_ns_traceMed, peak_amp_75q_ns_traceMed = med_perc_each_exp(peak_amp_ns_traceMed) # total number of planes (8*num_sessions); each element: med of peak amp across neurons in that plane
        peak_timing_med_ns_traceMed, peak_timing_25q_ns_traceMed, peak_timing_75q_ns_traceMed = med_perc_each_exp(peak_timing_ns_traceMed*frame_dur) # total number of planes (8*num_sessions); each element: med of peak timing across neurons in that plane

        
        ######## FLASH-evoked responses: Peak amplitude, computed on trial-median *traces* ########    # get the peak measures computed on trial-median traces (above is computed on individual trials, but then median is taken across trials; the measure here is less noisy as the peaks are computed on the meidan of the traces.)
        peak_amp_ns_traceMed_flash = all_sess_2an_this_mouse['peak_amp_eachN_traceMed_flash'].values # size: total number of planes (8*num_sessions), and then each element has size (neurons,) 
        peak_timing_ns_traceMed_flash = all_sess_2an_this_mouse['peak_timing_eachN_traceMed_flash'].values
    
        # for each session, compute med and iqr across neurons  # or plot hist of peak of all sessions    
        peak_amp_med_ns_traceMed_flash, peak_amp_25q_ns_traceMed_flash, peak_amp_75q_ns_traceMed_flash = med_perc_each_exp(peak_amp_ns_traceMed_flash) # total number of planes (8*num_sessions); each element: med of peak amp across neurons in that plane
        peak_timing_med_ns_traceMed_flash, peak_timing_25q_ns_traceMed_flash, peak_timing_75q_ns_traceMed_flash = med_perc_each_exp(peak_timing_ns_traceMed_flash*frame_dur) # total number of planes (8*num_sessions); each element: med of peak timing across neurons in that plane
                
        ######## Set session labels (stage names are too long)
    #    session_beg_inds = np.concatenate(([0], np.cumsum(num_trs_each_sess[:-1])+1))
        session_labs = []
        for ibeg in range(num_sessions):
            a = str(session_stages[ibeg]); 
            en = np.argwhere([a[ich]=='_' for ich in range(len(a))])[-1].squeeze() + 6
            txt = str(session_stages[ibeg][15:]) # session_stages[ibeg][8:en]
            txt = txt[0:min(3, len(txt))]
            session_labs.append(txt)
        
        jet_cm = colorOrder(nlines=len(session_stages)) # ax2.set_color_cycle(jet_cm)

        
        #%% ################################################
        ################## Start plotting ##################
        ####################################################
        
        fa = plt.figure(figsize=(np.max([8, 2*6]), 2.1*num_planes)) # 2* len(session_stages)
        plt.suptitle('%s, mouse %d' %(cre, mouse_id), y=.95, fontsize=18) # .94    
        # subplots of peak amp (neuron averaged, per trial) and omit-aligned traces
        gs = gridspec.GridSpec(num_planes, 2, width_ratios=[3, 1]) 
        gs.update(left=0.05, right=0.68, wspace=.2, hspace=1)
        # subplots of peak amp/ timing (trial-median)
        gs2 = gridspec.GridSpec(num_planes, 2)
        gs2.update(left=0.77, right=0.98, wspace=0.8, hspace=1)
        
    
        # Get data from a given plane across all sessions
        for iplane in range(num_planes): # iplane=0
             
            area = areas.at[iplane] # num_sess
            depth = depths.at[iplane] # num_sess
            if np.ndim(area)==0: # happens when len(session_stages)==1
                area = [area]
                depth = [depth]        
            
            ##### get data from a given plane across all sessions
            onePlane_allSess = np.arange(iplane, y.shape[0], num_planes)            
            
            y_this_plane_allsess = y[onePlane_allSess] # num_sess; each element: num_trs  # np.shape(y_this_plane_allsess)        
            y_trace_this_plane_allsess = y_trace[onePlane_allSess] # num_sess x time        
    #        y_trace_this_plane_allsess = np.vstack(y_trace_this_plane_allsess)
    
            peak_amp_med_ns_this_plane_allsess = peak_amp_med_ns[onePlane_allSess]
            peak_amp_25q_ns_this_plane_allsess = peak_amp_25q_ns[onePlane_allSess]
            peak_amp_75q_ns_this_plane_allsess = peak_amp_75q_ns[onePlane_allSess]
            
            peak_timing_med_ns_this_plane_allsess = peak_timing_med_ns[onePlane_allSess]
            peak_timing_25q_ns_this_plane_allsess = peak_timing_25q_ns[onePlane_allSess]
            peak_timing_75q_ns_this_plane_allsess = peak_timing_75q_ns[onePlane_allSess]        
            
            peak_amp_med_ns_traceMed_this_plane_allsess = peak_amp_med_ns_traceMed[onePlane_allSess]
            peak_amp_25q_ns_traceMed_this_plane_allsess = peak_amp_25q_ns_traceMed[onePlane_allSess]
            peak_amp_75q_ns_traceMed_this_plane_allsess = peak_amp_75q_ns_traceMed[onePlane_allSess]
            
            peak_timing_med_ns_traceMed_this_plane_allsess = peak_timing_med_ns_traceMed[onePlane_allSess]
            peak_timing_25q_ns_traceMed_this_plane_allsess = peak_timing_25q_ns_traceMed[onePlane_allSess]
            peak_timing_75q_ns_traceMed_this_plane_allsess = peak_timing_75q_ns_traceMed[onePlane_allSess]  
            
            # FLASH
            peak_amp_med_ns_traceMed_this_plane_allsess_flash = peak_amp_med_ns_traceMed_flash[onePlane_allSess]
            peak_amp_25q_ns_traceMed_this_plane_allsess_flash = peak_amp_25q_ns_traceMed_flash[onePlane_allSess]
            peak_amp_75q_ns_traceMed_this_plane_allsess_flash = peak_amp_75q_ns_traceMed_flash[onePlane_allSess]
            
            peak_timing_med_ns_traceMed_this_plane_allsess_flash = peak_timing_med_ns_traceMed_flash[onePlane_allSess]
            peak_timing_25q_ns_traceMed_this_plane_allsess_flash = peak_timing_25q_ns_traceMed_flash[onePlane_allSess]
            peak_timing_75q_ns_traceMed_this_plane_allsess_flash = peak_timing_75q_ns_traceMed_flash[onePlane_allSess]  

            
            #####
            num_trs_each_sess = [len(y_this_plane_allsess[i]) for i in range(num_sessions)]
            y_all = np.concatenate((y_this_plane_allsess)) # all sess data concatenated
            mn = np.nanmin(y_all)
            mx = np.nanmax(y_all)
            r = (mx-mn)/50.
            
            
            
            #%% Plot; all sessions concatenated in each subplot; makes subplots for each plane
            # For each mouse and each experiment (plane), show max amplitude of neural response at peak_win (0-2sec) following the omission trials vs. 
            # trial number ... do this for each session.... and show all sessions concatenated on the x axis.
            
            ####### Plot y_all : all sessions concatenated #######
            
            plt.figure(fa.number)        
    #        ax = plt.subplot(num_planes,1,iplane+1)
            ax = plt.subplot(gs[iplane,0])
            
            plt.plot(y_all)
            plt.vlines(np.cumsum(num_trs_each_sess)+1, mn-r, mx+r)
            plt.title('plane %d, area %s, depth %dum' %(iplane, np.unique(area)[0], np.mean(depth)), fontsize=13.5, y=1)
            if iplane==3:
                plt.ylabel('peak amp %s, omit-evoked response ' %(ylab_short), fontsize=12)
            if iplane==0: #num_planes-1:
                plt.xlabel('trial', fontsize=12)            
            ax.tick_params(labelsize=10)
            plt.gca().grid(False)
            
            session_beg_inds = np.concatenate(([0], np.cumsum(num_trs_each_sess[:-1])+1))
    #        session_labs = []
            txt_y = mx # mn-r*2 # mn-r*5
            for ibeg in range(len(session_beg_inds)):
    #            a = str(session_stages[ibeg]); 
    #            en = np.argwhere([a[ich]=='_' for ich in range(len(a))])[-1].squeeze() + 6
    #            txt = str(session_stages[ibeg][15:]) # session_stages[ibeg][8:en]
    #            session_labs.append(txt)
                txt = session_labs[ibeg]
                plt.text(session_beg_inds[ibeg], txt_y, txt, fontsize=10) #, transform=plt.gca().transAxes)
    
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)
    
    
            #%% Plot omit-aligned traces (averaged across trials, and median across neurons for each session)
            
            ####### Plot y_trace_this_plane_allsess ####### 
            
            ax2 = plt.subplot(gs[iplane,1])
            
            h = []
            for isess in range(y_trace_this_plane_allsess.shape[0]):
                h0 = plt.plot(time_trace, y_trace_this_plane_allsess.T[:,isess], color=jet_cm[isess])[0]
                h.append(h0)

            mn = np.min(y_trace_this_plane_allsess.flatten())
            mx = np.max(y_trace_this_plane_allsess.flatten())        
            plt.vlines(0, mn, mx, 'r', linestyle='-') #, cmap=colorsm)        
    #        plt.vlines(samps_bef, mn, mx, 'r', linestyle='-') #, cmap=colorsm)        
            plt.vlines(flashes_win_trace_index_unq_time, mn, mx, linestyle=':')
            
            # mark peak_win: the window over which the response quantification (peak or mean) was computed 
            plt.axvspan(peak_win[0], peak_win[1], alpha=0.3, facecolor='gray')
            plt.axvspan(flash_win[0], flash_win[1], alpha=0.3, facecolor='y')
                    
            
            if iplane==3:
                plt.ylabel('DF/F %s' %(ylab_short), fontsize=12)
                plt.xlabel('Time (s)', fontsize=12)
            if iplane==0: #num_planes-1:
                plt.xlabel('Time (s)', fontsize=12)
                plt.legend(h, session_labs, loc='center left', bbox_to_anchor=(.95, .7), frameon=False, fontsize=12, handlelength=.4)        
            ax2.tick_params(labelsize=10)
    #        makeNicePlots(ax1,0,1)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)
#            plt.xlim([-3,1])
            
        
            #%% Plot peak amp/timing (med/iqr across neurons. each neuron was already med and iqr across trials)
            
            ####### OMISSION: Plot peak_amp_med_ns_traceMed_this_plane_allsess & peak_timing_med_ns_traceMed_this_plane_allsess #######
            if trace_median: # use peak measures computed on the trial-median traces (less noisy)
                med = peak_amp_med_ns_traceMed_this_plane_allsess
                q25 = peak_amp_25q_ns_traceMed_this_plane_allsess
                q75 = peak_amp_75q_ns_traceMed_this_plane_allsess
            else:
                med = peak_amp_med_ns_this_plane_allsess
                q25 = peak_amp_25q_ns_this_plane_allsess
                q75 = peak_amp_75q_ns_this_plane_allsess
    
            l = med - q25
            u = q75 - med
    #        ax = axs[0]
            ax = plt.subplot(gs2[iplane,0])

            ax.errorbar(range(num_sessions), med, yerr=[l,u], fmt='o', markersize=3, capsize=3)

            plt.hlines(0, 0, num_sessions-1, linestyle=':')
            ax.set_xticks(range(num_sessions))
            ax.set_xticklabels(session_labs, rotation=45)
            ax.tick_params(labelsize=10)
            plt.xlim([-.5, num_sessions-.5])
            if iplane==0:
                plt.title('Omission', fontsize=13.5, y=1)
            if iplane==3:
                ax.set_ylabel('%s (med,iqr of Ns), (each N: %s)' %(lab_peakMean, lab_med), fontsize=12)
            plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
            seaborn.despine()#left=True, bottom=True, right=False, top=False)
                        
            

            ####### FLASHE-EVOKED RESPONSES: Plot peak_amp_med_ns_traceMed_this_plane_allsess & peak_timing_med_ns_traceMed_this_plane_allsess #######
            if plot_flash==1:
                if trace_median: # use peak measures computed on the trial-median traces (less noisy)
                    med = peak_amp_med_ns_traceMed_this_plane_allsess_flash
                    q25 = peak_amp_25q_ns_traceMed_this_plane_allsess_flash
                    q75 = peak_amp_75q_ns_traceMed_this_plane_allsess_flash
                else:
                    med = peak_amp_med_ns_this_plane_allsess
                    q25 = peak_amp_25q_ns_this_plane_allsess
                    q75 = peak_amp_75q_ns_this_plane_allsess
        
                l = med - q25
                u = q75 - med
        #        ax = axs[0]
                ax = plt.subplot(gs2[iplane,1])
    
                ax.errorbar(range(num_sessions), med, yerr=[l,u], fmt='o', markersize=3, capsize=3)
    
                plt.hlines(0, 0, num_sessions-1, linestyle=':')
                ax.set_xticks(range(num_sessions))
                ax.set_xticklabels(session_labs, rotation=45)
                ax.tick_params(labelsize=10)
                plt.xlim([-.5, num_sessions-.5])
                if iplane==0:
                    plt.title('Flash', fontsize=13.5, y=1)
                if iplane==3:
                    ax.set_ylabel('Flash: %s (med,iqr of Ns), (each N: %s)' %(lab_peakMean, lab_med), fontsize=12)
                plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
                seaborn.despine()#left=True, bottom=True, right=False, top=False)
 
    
    
            ############# OMISSION: Plot peak timing #############
            else: 
                if trace_median: # use peak measures computed on the trial-median traces (less noisy)
                    med = peak_timing_med_ns_traceMed_this_plane_allsess
                    q25 = peak_timing_25q_ns_traceMed_this_plane_allsess
                    q75 = peak_timing_75q_ns_traceMed_this_plane_allsess
                else:
                    med = peak_timing_med_ns_this_plane_allsess
                    q25 = peak_timing_25q_ns_this_plane_allsess
                    q75 = peak_timing_75q_ns_this_plane_allsess
                
                l = med - q25
                u = q75 - med
        #        ax = axs[1]
                ax = plt.subplot(gs2[iplane,1])
    
                ax.errorbar(range(num_sessions), med, yerr=[l,u], fmt='o', markersize=3, capsize=3)
    
                ax.set_xticks(range(num_sessions))
                ax.tick_params(labelsize=10)
                ax.set_xticklabels(session_labs, rotation=45, fontsize=8.8)
                plt.xlim([-.5, num_sessions-.5])
                if iplane==0:
                    plt.title('Omission', fontsize=13.5, y=1)
                if iplane==3:
                    ax.set_ylabel('Peak time (s) rel. omit (med,iqr of Ns), (each N: %s)' %(lab_med), fontsize=12)    
        #        figp.suptitle('plane %d, area %s, depth %dum' %(iplane, np.unique(area), np.mean(depth)), fontsize=13.5, y=1.1)
        #        figp.subplots_adjust(wspace=.7) #, hspace=.8)                     
                plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
                seaborn.despine()#left=True, bottom=True, right=False, top=False)
            
            
            
            #%% Make subplots for each session; loop over sessions (for a given plane)
            """
            # make individual figures for each plane; each figure includes subplots for every session.
            fe = plt.figure(figsize=(8,3*len(session_stages)))
            if len(session_stages)>1:
                plt.suptitle('%s, mouse %d, plane %d' %(cre, mouse_id, iplane), fontsize=18)
    
            for isess in range(len(session_stages)):
                y_plane = y_this_plane_allsess[isess]
                # np.shape(y_plane)
                
                ax = plt.subplot(len(session_stages),1,isess+1)            
                plt.plot(y_plane)    
    
                xlab = session_stages[isess] #[np.array(iplane)/8]
                plt.xlabel(xlab, fontsize=12)
                if isess==0:
                    plt.ylabel('peak amp (norm. to baseline sd) \n omit-evoked response ', fontsize=12)
                ax.tick_params(labelsize=10)
                if len(session_stages)>1:
                    plt.title('area %s, depth %d' %(area[isess], depth[isess]), fontsize=15)
                else:
                    plt.title('area %s, depth %d \n plane %d, area %s, depth %dum' %(area[isess], depth[isess], iplane, np.unique(area), np.mean(depth)), fontsize=15)                
    #            plt.suptitle('mouse %d, %s, \nplane %d, area %s, depth %d' %(mouse_id, cre, iplane, area[isess], depth[isess]), fontsize=15)
                plt.ylim([mn-r , mx+r])
                
            plt.figure(fe.number)
            plt.subplots_adjust(wspace=.5, hspace=.8)                
    #        set_xticklabels(x_ticks, rotation=0, fontsize=8)
        
            if dosavefig:
    #            plt.figure(fe.number)
                nam = '%s_mouse%d_plane%d' %(cre, mouse_id, iplane)
                fign = os.path.join(dir0, dir_now, nam+fmt)        
                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
            """
            
        #%%    
        if dosavefig:
    #        plt.figure(fa.number)
            nam = '%s_mouse%d%s_%s_all_planes_%s' %(cre, mouse_id, whatSess, fgn, now)
            fign = os.path.join(dir0, dir_now, nam+fmt)        
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    






#%% Single-mouse plots
########################################################################################################
########################################################################################################
######################## Average data across sessions, for each plane, and each area. ##################
######################## Also plot averages that are pooled across layers or areas. ####################           
########################################################################################################
########################################################################################################            
                
for im in range(len(trace_peak_allMice)): # im=5
    
    #%%
    mouse_id = trace_peak_allMice.iloc[im]['mouse_id']
    cre = trace_peak_allMice.iloc[im]['cre']
    session_stages = trace_peak_allMice.iloc[im]['session_stages']
    session_labs = trace_peak_allMice.iloc[im]['session_labs']
    n_omissions = trace_peak_allMice.iloc[im]['n_omissions']
    
    if sum(n_omissions)==0:
        print(f'No omissions for mouse {im}! skipping the summary plot!')
    else:
        plt.figure(figsize=(8,11))  
    #    gs1 = gridspec.GridSpec(2,1) #(3, 2)#, width_ratios=[3, 1]) 
    #    gs1.update(bottom=.6, top=0.9, left=0.05, right=0.5, hspace=.5)

        gs1 = gridspec.GridSpec(1,2) #(3, 2)#, width_ratios=[3, 1]) 
        gs1.update(bottom=.73, top=0.9, left=0.05, right=0.95, wspace=.8)

        gs2 = gridspec.GridSpec(1,2) #(3, 2)#, width_ratios=[3, 1]) 
        gs2.update(bottom=.45, top=0.62, left=0.05, right=0.95, wspace=.8)

        plt.suptitle('%s, mouse %d\n%d sessions: %s' %(cre, mouse_id, len(session_stages), session_labs), fontsize=14)


        #%% Plot omission-aligned traces, session-averaged, each plane    

        # trace
        av_trace_eachPlane = trace_peak_allMice_sessAvSd.iloc[im]['av_trace_allPlanes'] # 8 x num_sessions x 80
        sd_trace_eachPlane = trace_peak_allMice_sessAvSd.iloc[im]['sd_trace_allPlanes']

        areas = trace_peak_allMice_sessAvSd.iloc[im]['area'] # 8 x num_sessions # each row is one plane, all sessions
        depths = trace_peak_allMice_sessAvSd.iloc[im]['depth']    

        lims = np.array([np.nanmin(av_trace_eachPlane - sd_trace_eachPlane), np.nanmax(av_trace_eachPlane + sd_trace_eachPlane)])
        lims[0] = lims[0] - np.diff(lims) / 20.
        lims[1] = lims[1] + np.diff(lims) / 20.


        ############################### LM ###############################
        h1 = []
        for iplane in inds_lm: #range(4): # iplane = 0
            ax = plt.subplot(gs1[1]) # plt.subplot(2,1,1)

            area = areas[iplane] # num_sess 
            depth = depths[iplane] # num_sess             
            lab = '%dum' %(np.mean(depth))
    #        lab = '%s, ~%dum' %(np.unique(area)[0], np.mean(depth))

            h1_0 = plt.plot(time_trace, av_trace_eachPlane[iplane], color=cols_depth[iplane], label=(lab), markersize=3.5)[0]            
            # errorbars (standard error across cv samples)   
            plt.fill_between(time_trace, av_trace_eachPlane[iplane] - sd_trace_eachPlane[iplane], av_trace_eachPlane[iplane] + sd_trace_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane], facecolor=cols_depth[iplane])            
            h1.append(h1_0)

        # add to plots flash lines, ticks and legend
        plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
    #    plt.ylabel('')
        plt.xlim(xlim);
        plt.title(np.unique(area)[0], fontsize=13, y=1)


        ############################### V1 ###############################
        h1 = []
        for iplane in inds_v1: #np.arange(4,num_planes): #num_planes):

            ax = plt.subplot(gs1[0]) # plt.subplot(2,1,2)

            area = areas[iplane] # num_sess 
            depth = depths[iplane] # num_sess         
            lab = '%dum' %(np.mean(depth))
    #        lab = '%s, ~%dum' %(np.unique(area)[0], np.mean(depth))

            h1_0 = plt.plot(time_trace, av_trace_eachPlane[iplane],color=cols_depth[iplane-num_planes], label=(lab), markersize=3.5)[0]
            # errorbars (standard error across cv samples)   
            plt.fill_between(time_trace, av_trace_eachPlane[iplane] - sd_trace_eachPlane[iplane], av_trace_eachPlane[iplane] + sd_trace_eachPlane[iplane], alpha=alph, edgecolor=cols_depth[iplane-num_planes], facecolor=cols_depth[iplane-num_planes])
            h1.append(h1_0)

        # add to plots flash lines, ticks and legend
        plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
        plt.xlim(xlim);
        plt.title(np.unique(area)[0], fontsize=13, y=1)


        #%% Plot results per area (pooled across sessions and layers for each area)
        # Question 1: is there a difference between areas (V1 vs LM) # for each mouse pool across sessions and layers (for each area)

        # trace
        av_trace_pooled_eachArea = trace_peak_allMice_sessAvSd.iloc[im]['av_trace_eachArea']
        sd_trace_pooled_eachArea = trace_peak_allMice_sessAvSd.iloc[im]['sd_trace_eachArea']

        ##################
        ax = plt.subplot(gs2[0])
        #    ax = plt.subplot(gs1[2,0])
        plt.title('Sessions and layers pooled', fontsize=13, y=1)

        h1 = []
        for iarea in [1,0]: # first plot V1 then LM  #range(a.shape[0]):
            lab = '%s, n=%d' %(distinct_areas[iarea], num_sessLayers_valid_eachArea)
            h1_0 = plt.plot(time_trace, av_trace_pooled_eachArea[iarea], color = cols_area[iarea], label=(lab), markersize=3.5)[0]        
            plt.fill_between(time_trace, av_trace_pooled_eachArea[iarea] - sd_trace_pooled_eachArea[iarea] , \
                             av_trace_pooled_eachArea[iarea] + sd_trace_pooled_eachArea[iarea], alpha=alph, edgecolor=cols_area[iarea], facecolor=cols_area[iarea])
            h1.append(h1_0)

        # add to plots flash lines, ticks and legend
        lims = np.array([np.nanmin(av_trace_pooled_eachArea - sd_trace_pooled_eachArea), \
                         np.nanmax(av_trace_pooled_eachArea + sd_trace_pooled_eachArea)])
        lims[0] = lims[0] - np.diff(lims) / 20.
        lims[1] = lims[1] + np.diff(lims) / 20.

        plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
        #    plt.ylabel('')
        plt.xlim(xlim);


        #%% Plot results per depth (pooled across sessions and areas for each depth)
        # Question 2: is there a difference between superficial and deep layers # for each mouse pool across sessions and area (for each layer)... also pool 2 superifical (deep) layers into 1 

        # set average depth across sessions and areas            
        d = pooled_trace_peak_allMice.iloc[im]['depth_pooled_eachDepth']
        depth_ave = np.mean(d, axis=1) # average across areas and sessions
    #    b, _,_ = pool_sesss_planes_eachArea(area, area)

        # Average across areas and sessions for each depth
        # trace
        av_trace_pooled_eachDepth = trace_peak_allMice_sessAvSd.iloc[im]['av_trace_eachDepth']
        sd_trace_pooled_eachDepth = trace_peak_allMice_sessAvSd.iloc[im]['sd_trace_eachDepth']

        ##################
        ax = plt.subplot(gs2[1])
    #    ax = plt.subplot(gs1[2,1])
        plt.title('Sessions and areas pooled', fontsize=13, y=1)

        h1 = []
        for idepth in range(num_depth):
            lab = '%d um, n=%d' %(depth_ave[idepth], num_sessAreas_valid_eachDepth)
            h1_0 = plt.plot(time_trace, av_trace_pooled_eachDepth[idepth], color = cols_depth[idepth], label=(lab), markersize=3.5)[0]        
            plt.fill_between(time_trace, av_trace_pooled_eachDepth[idepth] - sd_trace_pooled_eachDepth[idepth] , \
                             av_trace_pooled_eachDepth[idepth] + sd_trace_pooled_eachDepth[idepth], alpha=alph, edgecolor=cols_depth[idepth], facecolor=cols_depth[idepth])

            h1.append(h1_0)

        # add to plots flash lines, ticks and legend
        lims = np.array([np.nanmin(av_trace_pooled_eachDepth - sd_trace_pooled_eachDepth), \
                         np.nanmax(av_trace_pooled_eachDepth + sd_trace_pooled_eachDepth)])
        lims[0] = lims[0] - np.diff(lims) / 20.
        lims[1] = lims[1] + np.diff(lims) / 20.

        plot_flashLines_ticks_legend(lims, h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
    #    plt.ylabel('')
        plt.xlim(xlim);


        #%%    
        if dosavefig:
            nam = '%s_mouse%d%s_aveSessPooled_%s_all_planes_%s' %(cre, mouse_id, whatSess, fgn, now)
            fign = os.path.join(dir0, dir_now, nam+fmt)        
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    





#%% 
##########################################################################################
##########################################################################################
############# For each session plot trial-averaged and neuron-averaged ###################
####################### traces for every individual plane ################################ 
##########################################################################################    
##########################################################################################

if plot_eachSess_trAve_neurAve:
    for isess in np.arange(0, len(mouse_trainHist_all)):
        
        mouse_id = mouse_trainHist_all.iloc[isess]['mouse_id']
        session_id = mouse_trainHist_all.iloc[isess]['session_id']   
#         print([isess, session_id])
    #    all_sess_now = all_sess_2an[all_sess_2an['mouse_id'] == mouse_id]
        all_exp_now = all_sess_2an[all_sess_2an['session_id'] == session_id] # find all the experiments corresponding to this session    

        if len(all_exp_now)>0:

            cre = all_exp_now.iloc[0]['cre']
            cre = cre[:cre.find('-')] # remove the IRES-Cre part
            stage = str(all_exp_now.iloc[0]['stage'])[15:]
            experiment_ids = all_exp_now['experiment_id'].values     

            plt.figure(figsize=(11,10))        
            plt.suptitle('%s, mouse %d, session %d, %s' %(cre, mouse_id, session_id, stage), y=.995, fontsize=18)
        #    gs = gridspec.GridSpec(4, 4) #, width_ratios=[3, 1])

            # now loop over the experiments 
            for iplane in range(len(experiment_ids)): # iplane = 0

                area = all_exp_now['area'].iloc[iplane]
                depth = all_exp_now['depth'].iloc[iplane]
                experiment_id = all_exp_now['experiment_id'].iloc[iplane]
                
                traces_aveTrs_time_ns = all_exp_now.iloc[iplane]['traces_aveTrs_time_ns']
                traces_aveNs_time_trs = all_exp_now.iloc[iplane]['traces_aveNs_time_trs']


                # plot trial-averaged and neuron-average traces
                if (~np.isnan(traces_aveTrs_time_ns)).all(): # make sure the experiment is valid
            #        r = int(np.floor(iplane/4))+1
                    c = np.remainder(iplane,4)

                    if iplane<4:
                        rt = 0
                        rn = 2
                    else:
                        rt = 5
                        rn = 7

                    # show trial-averaged, all neurons
            #        ax1 = plt.subplot(gs[r*2-2, c])
                    ax1 = plt.subplot2grid((9, 4), (rt, c), rowspan=2)

                    plt.plot(traces_aveTrs_time_ns, 'gray', zorder=1, linewidth=.5) # show trial-averaged, all neurons


                    avD = np.median(traces_aveTrs_time_ns, axis=-1)
                    sdD = st.iqr(traces_aveTrs_time_ns, axis=-1) # = q75-q25
                    q25 = np.percentile(traces_aveTrs_time_ns, 25, axis=-1) # in average and std case, this term is like av-sd
                    q75 = np.percentile(traces_aveTrs_time_ns, 75, axis=-1) # in average and std case, this term is like av+sd


                    plt.fill_between(range(samps_bef+samps_aft), q25, q75, alpha=0.3, edgecolor='none', facecolor='r', zorder=10)

                    plt.plot(avD, 'red', marker='.', markersize=2, linewidth=.7, zorder=2) # average across neurons (each trace is trial-average)
                    mn = min(traces_aveTrs_time_ns.flatten())
                    mx = max(traces_aveTrs_time_ns.flatten())
                    plt.vlines(flashes_win_trace_index_unq+samps_bef, mn, mx, color='black', linestyle='dashed', linewidth=.7, zorder=3) # showing when the flashes happened for the 1st omission ... it is not easy to show it on the trial-averaged trace. as the timing of flashes varies a bit across all omission-aligned traces                  
    #                 plt.vlines(flashes_win_trace_index_all[np.random.permutation(num_omissions)[0]], mn, mx, color='black', linestyle='dashed', linewidth=.7, zorder=3) # showing when the flashes happened for the 1st omission ... it is not easy to show it on the trial-averaged trace. as the timing of flashes varies a bit across all omission-aligned traces  
                    plt.vlines(samps_bef, mn, mx, 'red', linestyle=':', zorder=4) #, cmap=colorsm)
                    plt.title('%s,%d um\n%s' %(area, depth, experiment_id), fontsize=13.5, y=1) # plt.title('%s,%d um\nplane %d' %(area, depth, iplane), fontsize=13.5, y=1)
                    ax1.tick_params(labelsize=10)
                    if np.logical_or(iplane==0, iplane==4):            
            #            plt.xlabel('Frame (~10Hz)', fontsize=12)
                        plt.ylabel('DF/F %s\nTrial-ave, each N' %(ylab_short), fontsize=12)


                    # show neuron-averaged, all trials
            #        ax2 = plt.subplot(gs[r*2-1, c])
                    ax2 = plt.subplot2grid((9, 4), (rn, c), rowspan=2)

                    avD = np.median(traces_aveNs_time_trs, axis=-1)
                    sdD = st.iqr(traces_aveNs_time_trs, axis=-1)
                    q25 = np.percentile(traces_aveNs_time_trs, 25, axis=-1)
                    q75 = np.percentile(traces_aveNs_time_trs, 75, axis=-1)

                    plt.fill_between(range(samps_bef+samps_aft), q25, q75, alpha=0.3, edgecolor='none', facecolor='r', zorder=10)

                    plt.plot(traces_aveNs_time_trs, 'gray', zorder=1, linewidth=.5)
                    plt.plot(avD, 'red', marker='.', markersize=2, linewidth=.7, zorder=2) # average across trials (each trace is neuron-average)
                    mn = min(traces_aveNs_time_trs.flatten())
                    mx = max(traces_aveNs_time_trs.flatten())                
                    plt.vlines(flashes_win_trace_index_unq+samps_bef, mn, mx, color='black', linestyle='dashed', linewidth=.7, zorder=3) # showing when the flashes happened for the 1st omission ... it is not easy to show it on the trial-averaged trace. as the timing of flashes varies a bit across all omission-aligned traces 
                    plt.vlines(samps_bef, mn, mx, 'red', linestyle=':', zorder=4) #, cmap=colorsm)
                    ax2.tick_params(labelsize=10)
                    if np.logical_or(iplane==0, iplane==4):
                        plt.xlabel('Frame (~10Hz)', fontsize=12)
            #            plt.ylabel('DF/F %s\nNeuron-averaged, each trial' %(ylab), fontsize=12)            
                        plt.ylabel('Neuron-ave\neach Trial', fontsize=12)

                    plt.subplots_adjust(wspace=.25, hspace=.9) 



                    if dosavefig:
                #        plt.figure(fa.number)
                        nam = '%s_mouse%d_session%d_%s_aveTrsNs_%s' %(cre, mouse_id, session_id, fgn, now)
                        fign = os.path.join(dir0, dir_now, nam+fmt)        
                        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
        



