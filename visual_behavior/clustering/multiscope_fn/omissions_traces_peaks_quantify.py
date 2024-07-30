#%% Quantify omission and flash evoked responses.

from def_funs import *

def omissions_traces_peaks_quantify(traces_aveTrs_time_ns, bl_preOmit, bl_preFlash, mean_notPeak, cre, peak_win, flash_win, flash_win_timing, flash_index, samps_bef, frame_dur, doShift_again, doPeakPerTrial, doROC, doPlots=0, index=0, local_fluo_allOmitt=0, num_omissions=0):

    if np.ndim(traces_aveTrs_time_ns)>1:
        num_neurons = traces_aveTrs_time_ns.shape[1]
    else:
        num_neurons = 1
    
    #%% Set time windows, in frame units, for computing flash and omission evoked responese
    list_times, list_times_flash, list_times_flash_timing = set_frame_window_flash_omit(peak_win, flash_win, flash_win_timing, samps_bef, frame_dur)

    '''
    # the method below is more accurate because instead of going with .75ms before peak_win to define flash_win, we use flash_omit_dur_fr_all; but you computed it and noticed that in both cases list_times_flash is 8 frames before list_times... so i'm not bothering with below... if you want to do it, you have to input flash_omit_dur_fr_all and also if it is vip,B1, so you can choose which list_times_flash to use. (in corrs code I'm using the code below.; for svm (this code), not.)
    list_times, _, _ = set_frame_window_flash_omit(peak_win, np.nan, np.nan, samps_bef, frame_dur)    
    
    fo_dur_fr_med = np.median(flash_omit_dur_fr_all).astype(int)
    flash_win_vip_shift = .25 # sec; shift flash_win this much earlier to compute flash responses on vip B1 sessions.

    list_times_flash = list_times - fo_dur_fr_med
    list_times_flash_vip = list_times_flash - np.round(flash_win_vip_shift/frame_dur).astype(int) # we shift the window 250ms earlier to capture vip responses
    '''

    #%% Compute omission-evoked response on trial-median traces ... to avoid computing peak on individual noisy trials which are perhaps too noisy

    ### NOTE: this part needs work at some point. 
    # instead of going with cell lines, you should compute the 
    # mean activity after omission. Then see if it is above baseline (or some other threshold).
    # if yes, call it incrased activity, and attemt to compute peak amp/timing. 
    # if no, call it decreased activity, and do not compute peak amp/ timing.

    # this method will allow for proper computation of the peak (timing especially) in case
    # some neurons of a cell line have increased activity and some have decreased activity.

    ###### response amplitude : either max (peak) or mean during peak_win. Mean is preferred because SST and SLC responses don't increase after omission ######
    # for each trial-median neuron trace find the peak during the peak_win window
    if mean_notPeak: # compute average trace during peak_win
        peak_amp_eachN_traceMed = np.nanmean(traces_aveTrs_time_ns[list_times], axis=0) # neurons

    else: # compute peak
        if cre.find('Vip')==-1: # SLC, SST # if the mouse is NOT a VIP line, instead of computing the peak, we compute the trough: we invert the trace, and then compute the max. (this is because in VIP mice the response after omission goes up, but not in SST or Slc)
            if index==0:
                print('Peak amplitude cannot be computed for SST or SLC lines, because their response after omission does not increase, setting to NaN!')
            peak_amp_eachN_traceMed = np.full((num_neurons), np.nan)
            ###### NOT GOOD!!! we cant really do the following! maybe some SST and SLC neurons incrase activity or dont respond after omission
            # we cant just inverse their trace
    #                        print('BE CAUTIOUS: to compute peak amplitude, we are inverting the trace assuming it is inhibited by omission!')
    #                        peak_amp_eachN_traceMed = -np.nanmax(-traces_aveTrs_time_ns[list_times], axis=0) # neurons
        else: # VIP, easy, just get the max
            peak_amp_eachN_traceMed = np.nanmax(traces_aveTrs_time_ns[list_times], axis=0) # neurons
    peak_amp_eachN_traceMed0 = peak_amp_eachN_traceMed + 0


    ###### peak timing : wont be accurate, because not always the response goes up after omission (ie peak cant be computed always) ######
    if cre.find('Vip')==-1: # SLC, SST   
        if index==0:
            print('Peak timing cannot be computed for SST or SLC lines, because their response after omission does not increase, setting to NaN!')
        peak_timing_eachN_traceMed = np.full((num_neurons), np.nan)
    #                        print('BE CAUTIOUS: to compute peak timing, we are inverting the trace assuming it is inhibited by omission! Be cautious!')
    #                    peak_timing_eachN_traceMed = np.nanargmax(-traces_aveTrs_time_ns[list_times], axis=0)# neurons
    else: # VIP, easy, just get the argmax
        peak_timing_eachN_traceMed = np.nanargmax(traces_aveTrs_time_ns[list_times], axis=0) # neurons
    # compute timing relative to omission time
    peak_timing_eachN_traceMed = peak_timing_eachN_traceMed + (list_times[0] - samps_bef)


    #%%
    ##### Meausre the trough (this quantity wont make sense for vip mice, bc their activity goes up after omissions!)            
    # If the omission response goes down (relative to baseline), then we care about the trough not peak, so inverse the trace to compute the trough               

    # one way is to invert the trace along a horizontal line that equals the baseline ... 
    # since we shifted the traces so baseline is at 0, we can just inverse the traces (ie change their sign)... remember for this you have to define baseline as median, if you are computing trough on the median traces!

    # below did not work ... because in the 1st B session of VIP meouse, a similar thing to SST sessions happens, 
    # ie there is no response after omission ... and the code below messes up the effect there, as it  
    '''               
    if mean_notPeak==0:
        for ineur in range(num_neurons): 
            # If the omission response goes down (relative to baseline), then we care about the trough not peak, so inverse the trace to compute the trough               
            # we compute mean trace in a small window (of size ~750ms) following the omission to see if it is less than baseline or higher
            # if less, we invert the trace, and then compute the max.
            if np.nanmean(traces_aveTrs_time_ns[np.arange(samps_bef, samps_bef+8)], axis=0)[ineur] < bl[ineur]: # peak_amp_eachN_traceMed[ineur]
                # we find the max on the inverse trace, but the value we assign to peak amp is the actual value (before the inverse)
                peak_amp_eachN_traceMed[ineur] = -np.nanmax(-traces_aveTrs_time_ns[list_times, ineur], axis=0) # neurons
                peak_timing_eachN_traceMed[ineur] = np.nanargmax(-traces_aveTrs_time_ns[list_times, ineur], axis=0)# neurons
    '''


    #%% Subtract baseline of the omit-aligned trace from its peak to compute the relative change in peak amplitude.
    # compute how much the trace max (or mean trace) (during peak_win) has changed relative to
    # the min (or 10th percentile) activity right before omission

    if doShift_again==0:
        peak_amp_eachN_traceMed = peak_amp_eachN_traceMed0 - bl_preOmit

    # below should have baseline at 0
    #                plt.plot(traces_aveTrs_time_ns[:,3]-bl_preOmit[3])                

#     this_sess.at[index, ['peak_amp_eachN_traceMed']] = [peak_amp_eachN_traceMed] # neurons
#     this_sess.at[index, ['peak_timing_eachN_traceMed']] = [peak_timing_eachN_traceMed] # neurons



    
    #########################################################################
    #########################################################################
    #########################################################################
    #%% FLASHES: Do the same as above except for flash-evoked responses: compute response amp on trial-median traces
    #########################################################################
    #########################################################################
    #########################################################################
    
    ###### response amplitude : either max (peak) or mean during peak_win. Mean is preferred because SST and SLC responses don't increase after omission ######
    # for each trial-median neuron trace find the peak during the peak_win window
    if mean_notPeak: # compute average trace during peak_win
        peak_amp_eachN_traceMed_flash = np.nanmean(traces_aveTrs_time_ns[list_times_flash], axis=0) # neurons
    else: # compute peak (unlike omissions, we know all cell lines respond to flash, so we just get the max)
        peak_amp_eachN_traceMed_flash = np.nanmax(traces_aveTrs_time_ns[list_times_flash], axis=0) # neurons
    peak_amp_eachN_traceMed_flash0 = peak_amp_eachN_traceMed_flash + 0

    #%% Subtract baseline of the omit-aligned trace from its peak to compute the relative change in peak amplitude.
    # compute how much the trace max (or mean trace) (during peak_win) has changed relative to
    # the min (or 10th percentile) activity right before omission

    if doShift_again==0:
        peak_amp_eachN_traceMed_flash = peak_amp_eachN_traceMed_flash0 - bl_preOmit
        # I think below makes more sense: to subtract bl_preFlash; (although bc both bl_preFlash and bl_preOmit are 10th pecentile of baseline (one from points before flash, the other from points after flash), the 2 values should not be different.)
#         peak_amp_eachN_traceMed_flash = peak_amp_eachN_traceMed_flash0 - bl_preFlash


    

    ###### peak timing : wont be accurate, because not always the response goes up after omission (ie peak cant be computed always) ######
    # unlike omissions, we know all lines respond to flash, so we just get the argmax regadless of cell line
    peak_timing_eachN_traceMed_flash = np.nanargmax(traces_aveTrs_time_ns[list_times_flash_timing], axis=0) # neurons
    # compute timing relative to flash time (remember flash_index is not quite accurate, I think); to be accurate you should align on the flash, instead of getting the flash timing from omission aligned traces.
    peak_timing_eachN_traceMed_flash = peak_timing_eachN_traceMed_flash + (list_times_flash_timing[0] - flash_index)



    #%%
#     this_sess.at[index, ['peak_amp_eachN_traceMed_flash']] = [peak_amp_eachN_traceMed_flash] # neurons
#     this_sess.at[index, ['peak_timing_eachN_traceMed_flash']] = [peak_timing_eachN_traceMed_flash] # neurons





    #%% Find amplitude of peak response on each omission trial (noisier compared to peak_amp_eachN_traceMed which is computed on trial-median trace)
    
    if doPeakPerTrial:
        peak_allTrsNs = np.full((num_omissions, num_neurons), np.nan) # trials x neurons
        peak_timing_allTrsNs = np.full((num_omissions, num_neurons), np.nan) # trials x neurons
        
        for itrial in range(num_omissions):
            a = local_fluo_allOmitt[:,:,itrial] # time x neurons                
            '''
            # Jerome coded list_times like below. I prefer to use a different method: 
            # convert peak_win to frames and use that to decide which frames to use for finding the peak
            # it's because local_time_allOmitt will be negative if the omission onset is slightly after the frame onset. 
            # And it will be positive if the omission onset is slightly before the frame onset.
            # so if we use the code below, for an omission that happened only 1 ms after frame 0, 
            # we will use the following frame as the onset frame in peak_win 
            # (because when an omission happens after frame onset, local_time_allOmitt will become negative, and the code below looks for the 1st positive value.)
            #
            # however, it makes more sense to just define peak_win relative to the frame of omission
            #(ie the frame in which more than ~45ms (0.09326/2 = 0.04663 sec) of the omission occurred ... 
            # this is because we are using min(abs(local_time_traces - indiv_time)) to find the frame of omission, so 
            # we may call a frame, frame of omission, although the omission started ~45ms (ie half a frame time) earlier)
            # 
            # using this method, we do still run into the problem that in some trials we will be looking for peak response a bit late, 
            # i.e. in the frame that follows the omission onset by 45ms, while 
            # in some other trials we will look for the peak a bit early, 
            # i.e. in the frame that precedes the omission onset by 45ms 
            # but I think this inherent problem will be less if I convert peak_win to frames to find peaks, as opposed to 
            # using the times (like the code below) 

            om_time = local_time_allOmitt[:,itrial] # time
            list_times = np.argwhere(np.all([om_time > peak_win[0], om_time < peak_win[1]], axis=0)).flatten()
            '''

            # for each trial and each neuron find amplitude of max response during [0,2] sec after omission trial    
            if mean_notPeak: # if 1, to quantify omission-evoked response, compute average trace during peak_win, instead of the peak amplitude.
                peak_allTrsNs[itrial,:] = np.nanmean(a[list_times,:], axis=0) # neurons
            else:
                peak_allTrsNs[itrial,:] = np.nanmax(a[list_times,:], axis=0) # neurons
        #                peak_allTrsNs.append(np.nanmax(a[:,list_times], axis = 1))

            # for each trial and each neuron find the timing (frame number) of the max response during [0,2] sec after omission trial    
            # compute peak timing relative to frame of omission
            peak_timing_allTrsNs[itrial,:] = np.nanargmax(a[list_times,:], axis=0) # neurons
        #                peak_timing_allTrsNs[itrial,:] = list_times[0] + np.nanargmax(a[list_times,:], axis=0) # neurons # peak timing relative to the trace onset

        ### some tests :
        #            plt.hist(peak_allTrsNs[:,0])
        #            plt.hist(local_fluo_allOmitt[samps_bef+10, 0])
        #            np.median(peak_allTrsNs[:,0])
        #            np.median(local_fluo_allOmitt[samps_bef+10, 0])


#         this_sess.at[index, ['peak_amp_trs_ns']] = [peak_allTrsNs] # trials x neurons
#         this_sess.at[index, ['peak_timing_trs_ns']] = [peak_timing_allTrsNs] # trials x neurons

    else:
        peak_allTrsNs = np.nan
        peak_timing_allTrsNs = np.nan




    #%% Average traces across omissions: in the first and second halves of the session
    # Set vars to get average of the omissions in the first and second halves of the session
    # I mean the first half of the omissions vs the second half.

    if doROC:
        mt = np.ceil(num_omissions / float(2)).astype(int)
        h1 = np.arange(0, mt) # trials in the 1st half of the session 
        h2 = np.arange(mt, num_omissions)  # trials in the 2nd half of the session

        # average across omission trials
        om_av_half1 = np.nanmean(local_fluo_allOmitt[:,:,h1], axis=2) # time x neurons
        om_av_half2 = np.nanmean(local_fluo_allOmitt[:,:,h2], axis=2) # time x neurons

        #   time 0 (omissions) values:
        #   local_fluo_allOmitt[:,samps_bef,:]
        #   local_time_allOmitt[:,samps_bef]

        # get average time trace for the 1st and 2nd halves of the session
        om_time_half1 = np.nanmean(local_time_allOmitt[:,h1], axis=1) # time
        om_time_half2 = np.nanmean(local_time_allOmitt[:,h2], axis=1) # time

        # average time trace for all trials
        om_time = np.nanmean(local_time_allOmitt, axis=1) # time


        #%% Plot average of the omissions in the first and second halves of the session

        if doPlots:
            ## average of the omissions in the first and second halves of the session
            plt.figure()
            plt.plot(om_time_half1, om_av_half1, 'b')
            plt.plot(om_time_half2, om_av_half2, 'r')
            plt.axvline(0)
            plt.title(depth)

            ## half2 - half1        
            plt.figure()
            # individual neurons
            plt.plot(om_time, om_av_half2 - om_av_half1, 'k')
            # average across neurons
            plt.plot(om_time, np.nanmean(om_av_half2 - om_av_half1, axis=1), 'r')
            plt.axvline(0)
            plt.title([depth, area])
            plt.ylim([-1,1])


        #%% Find the peak amplitude for each neuron on trial-averaged traces of the 1st half of the session

        # find peak on window: peak_win = [0, 2] # sec relative to omission onset (will be applied to om_time vector)
        list_times = np.argwhere(np.all([om_time > peak_win[0], om_time < peak_win[1]], axis=0)).flatten()

        # for each neuron find amplitude of max response during [0,2] sec after omission trial    
        # 1st half of the session
        peak_om_av_h1 = np.nanmax(om_av_half1[list_times,:], axis = 0) # neurons
        # 2nd half of the session
        peak_om_av_h2 = np.nanmax(om_av_half2[list_times,:], axis = 0) # neurons

#         this_sess.at[index, ['peak_h1', 'peak_h2']] = peak_om_av_h1, peak_om_av_h2

        # plot a histogram of peaks for 1st and 2nd halves
        if doPlots:
            plt.figure()
            plt.hist(peak_om_av_h1, label='h1')    
            plt.hist(peak_om_av_h2, label='h2')    
            plt.legend()


        #%% Compute AUC: for each experiment, find, across all neurons, how separable the distributions of 
        # trial-averaged peak amplitudes are for half-1 omissions vs. half-2 omissions.

        ######### it makes sense to do this also for each neuron (ie compute AUC for each neuron across all omission trials!)
        # We assign 0 to h1 and 1 to h2, so:
        # h1 > h2 means AUC < 0.5
        # h1 < h2 means AUC > 0.5
        y_true = np.concatenate(([np.zeros(np.shape(peak_om_av_h1)), np.ones(np.shape(peak_om_av_h1))]), axis = 0)
        y_score = np.concatenate(([peak_om_av_h1, peak_om_av_h2]), axis = 0)

        fpr, tpr, thresholds = roc_curve(y_true, y_score)    
        auc_peak_h1_h2 = auc(fpr, tpr)

#         this_sess.at[index, 'auc_peak_h1_h2'] = auc_peak_h1_h2  
        
    
    else:
        
        peak_om_av_h1 = np.nan
        peak_om_av_h2 = np.nan
        auc_peak_h1_h2 = np.nan
        
        
    
    return peak_amp_eachN_traceMed, peak_timing_eachN_traceMed, \
        peak_amp_eachN_traceMed_flash, peak_timing_eachN_traceMed_flash, \
        peak_allTrsNs, peak_timing_allTrsNs, \
        peak_om_av_h1, peak_om_av_h2, auc_peak_h1_h2  

