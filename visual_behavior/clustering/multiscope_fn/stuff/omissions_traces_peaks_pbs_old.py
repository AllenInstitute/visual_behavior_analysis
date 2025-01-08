#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Called by the script omissions_traces_peaks_init.py.
The function here creates a pandas table (all_sess) that
includes vars related to median traces across trials, aligned on omission, and their peak amplitude.

Created on Wed May  8 15:25:18 2019
@author: Farzaneh
"""

from def_funs import *
from def_funs_general import *
                
                
#%%
def omissions_traces_peaks_pbs(session_id, experiment_ids, validity_log_all, norm_to_max, mean_notPeak, peak_win, flash_win, trace_median, doScale, doShift, doShift_again, bl_gray_screen, samps_bef, samps_aft, doCorrs, saveResults, doPlots=0, doROC=0):    
    
    #%% Set initial vars
        
    # Use a different window to compute flash timing: the idea is to be able to capture the timing
    # also for VIP A sessions, in which the response rises before the flash and peaks right at flash time.
    # Also to get an accurate response magnitude measure for VIP A sessions, try: flash_win = [-1, -.65] 
    flash_win_timing = [-.85, -.5]
    bl_percentile = 10 #20  # for peak measurements, we subtract pre-omit baseline from the peak. bl_percentile determines what pre-omit values will be used.      
    corr_doShfl =  1 #1 # if 1, compute shuffled corrcoeff too.
    num_shfl_corr = 50 # shuffle trials, then compute corrcoeff... this serves as control to evaluate the values of corrcoeff of actual data    
        
    '''
    doCorrs = 1 # if 0, compute omit-aligned trace median, peaks, etc. If 1, compute corr coeff between neuron pairs in each layer of v1 and lm 
    doShift_again = 1 #0 # this is a second shift just to make sure the pre-omit activity has baseline at 0. (it is after we normalize the traces by baseline ave and sd... but because the traces are median of trials (or computed from the initial gray screen activity), and the baseline is mean of trials, the traces wont end up at baseline of 0)                                      
    
    flash_win = [-.75, 0] # window (relative to omission) for computing flash-evoked responses (750ms includes flash and gray)    
    bl_gray_screen = 1 # if 1, baseline will be computed on the initial gray screen at the beginning of the session
    # if 0, get the lowest 20 percentile values of the median trace preceding omission and then compute their mean and sd
    
    norm_to_max = 0 # normalize each neuron trace by its max
    ### In this case doScale is automatically set to zero, because:
    # remember whether you are normalizing to max peak activity or not will not matter if you normalize by baseline SD. because if you normalize to max peak activity, baseline SD will also get changed by that same amount, so after normalizing to baseseline SD the effect of normalizing to max peak activity will be gone.
    
    mean_notPeak = 0 # if 1, to quantify omission-evoked response, compute average trace during peak_win, instead of the peak amplitude.
    # if 1, we will still use the timing of the "peak" for peak timing measures ... 
    # Note sure of the following: if mean_notPeak is 1, it makes sense to go with a conservative (short) peak_win so the results are not contaminated by the responses to the flash that follows the omission    
    # set mean_notPeak to 0 if you want to measure trough (on traces with decreased response after the omission)
    
    peak_win = [0, 2] #.725] # go with 0.725ms if you don't want the responses to be contaminated by the next flash
    # [0,2] # flashes are 250ms and gray screen is 500ms # sec # time window to find peak activity, seconds relative to omission onset
    
    trace_median = 1 # if 1, use peak measures computed on the trial-median traces (less noisy); if 0, use peak measures computed on individual trials.
        
    doScale = 1 # Scale the traces by baseline std
    doShift = 1 # Shift the y values so baseline mean is at 0 at time 0    
    
    doPlots = 0 # some plots (not the main summary ones at the end)... set to 0
    doROC = 0 # if 1, do ROC analysis: for each experiment, find, across all neurons, how separable the distributions of trial-averaged peak amplitudes are for half-1 omissions vs. half-2 omissions
    #dosavefig = 1
    
    # To align on omissions, get 40 frames before omission and 39 frames after omission
    samps_bef = 40 
    samps_aft = 40
    
    saveResults = 1    
    '''
    
#    #%%              
#    if norm_to_max:
#        doScale = 0
#    
#    ### set this so you know what input vars you ran the script with
#    cols = np.array(['norm_to_max', 'mean_notPeak', 'peak_win', 'trace_median', 'samps_bef', 'samps_aft', 'doScale', 'doShift'])
#    input_vars = pd.DataFrame([], columns=cols)
#    
#    input_vars.at[0, cols] = norm_to_max, mean_notPeak, peak_win, trace_median, samps_bef, samps_aft, doScale, doShift
#    #input_vars.iloc[0]
#    
#        
#    #%% Initiate all_sess and this_sess panda tables
    if doCorrs: 
        cols = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', \
                 'n_omissions', 'n_neurons']) #, 'cc_a1_a2', 'cc_a1_a1', 'cc_a2_a2', 'p_a1_a2', 'p_a1_a1', 'p_a2_a2'])
    else:
        cols = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', \
                         'n_omissions', 'n_neurons', \
                         'traces_aveTrs_time_ns', 'traces_aveNs_time_trs', \
                         'peak_amp_eachN_traceMed', 'peak_timing_eachN_traceMed', 'peak_amp_trs_ns', 'peak_timing_trs_ns', \
                         'peak_h1', 'peak_h2', 'auc_peak_h1_h2', 'peak_amp_eachN_traceMed_flash', 'peak_timing_eachN_traceMed_flash'])

#    all_sess = pd.DataFrame([], columns=cols)
#    
#    
#    #%% Loop through valid sessions to perform the analysis
#    
##    sess_no_omission = []
#    cnt_sess = -1     
#    
#    for isess in range(len(list_all_sessions_valid)):  # isess = 0 # session_id = list_all_sessions_valid[0] #[num_valid_exps_each_sess == 8][0]
#        
#        session_id = int(list_all_sessions_valid[isess])
#    #    experiment_ids = list_all_experiments_valid[isess]
#        experiment_ids = list_all_experiments[isess] # we want to have the list of all experiments for each session regardless of whethere they were valid or not... this way we can find the same plane across all sessions.
#    #    experiment_ids = list_all_experiments_valid[np.squeeze(np.argwhere(np.in1d(list_all_sessions_valid, session_id)))]
#        
#        cnt_sess = cnt_sess + 1    
#        print('%d: session %d out of %d sessions' %(session_id, cnt_sess+1, len(list_all_sessions_valid)))
#    
#        
    #%% Load some important variables from the experiment
    
#    [whole_data, data_list, table_stim] = load_session_data(session_id) # data_list is similar to whole_data but sorted by area and depth
    [whole_data, data_list, table_stim, behav_data] = load_session_data_new(session_id, experiment_ids, use_ct_traces)

    # start time of omissions (in sec)
    list_omitted = table_stim[table_stim['omitted']==True]['start_time']
    list_omitted0 = list_omitted + 0
    
    
    #%%
    
    if len(list_omitted)==0:
        print('session %d has no omissions!' %session_id)
##        sys.exit('Exiting analysis as there are no omission trials in session %d!' %session_id)
##            sess_no_omission.append(session_id)
#        
#    else:
    ######################################################################
    ######################################################################
    #%% Set the omission-aligned traces
    # go through each experiment and if it is valid, we do a bunch of computations.
    ######################################################################
    ######################################################################
    #%%
    exp_ids = list(whole_data.keys())
    mouse = whole_data[exp_ids[0]]['mouse']
    date = whole_data[exp_ids[0]]['experiment_date']    
    cre = whole_data[exp_ids[0]]['cre'] # it will be the same for all lims_ids (all planes in a session)
    stage = whole_data[exp_ids[0]]['stage']
    
    # start time of all flashes (in sec)
    list_flashes = table_stim['start_time'].values #[table_stim['omitted']==False]['start_time']
    
    
    #%% Check for "weird" cases: repeated omissions (based on start times) or consecutive omissions
    #s = np.sort(np.diff(list_omitted))
    #si = np.argsort(np.diff(list_omitted))
    
    d = np.diff(list_omitted0)
    # If there are any repeated omissions, remove one of them.
    rep_om = np.argwhere(d == 0).flatten()
    list_omitted.iloc[rep_om] = np.nan
    
    # consecutive omission: would be interesting to study them but for now remove them all.    
    a = np.argwhere(np.logical_and(d>0 , d<.8)).flatten()
    consec_om = np.sort(np.concatenate((a, a+1), axis=0)) # remove any flash that is preceded or followed by another omitted flash
    list_omitted.iloc[consec_om] = np.nan
    
    # remove the values set to nan
    #list_omitted.drop(list_omitted.index[consec_om, rep_om])
    list_omitted = list_omitted[~np.isnan(list_omitted.values)]                
    #        len(list_omitted0), len(list_omitted)
    #        list_omitted0, ist_omitted
    
    
    #%% Loop through the 8 planes of each session
    
    # initiate the pandas tabledatetime.datetime.fromtimestamp(1548449865.568)
    this_sess = pd.DataFrame([], columns = cols)
    this_sess_l = pd.DataFrame([], columns = ['valid', 'local_fluo_allOmitt'])
    
    #        local_time_traces_all = []                
    #        depth_all = []
    #        area_all = []
    
    '''
    # plot the entire trace for each session
    
    get_ipython().magic(u'matplotlib inline') # %matplotlib inline
    
    dir_now = 'traces_entire_session'
    if not os.path.exists(os.path.join(dir0, dir_now)):
        os.makedirs(os.path.join(dir0, dir_now))
    nam = '%s_mouse%d_sess%d_traceAveNs_allFrs' %(cre, mouse, session_id)
    
    plt.figure(figsize=(8,11)) # show all traces of all neurons for each plane individually, to see how the activity changes in a session!
    '''
    
    for index, lims_id in enumerate(data_list['lims_id']): 
        '''
        for il in [5]: #range(num_planes):
            index = il
            lims_id = data_list['lims_id'].iloc[il]
        '''            
        '''
        ll = list(enumerate(data_list['lims_id'])); 
        l = ll[0]; # first plane
        index = l[0]; # plane index
        lims_id = l[1] # experiment id 
        '''
        depth = whole_data[lims_id]['imaging_depth']
        area = whole_data[lims_id]['targeted_structure']
    
        this_sess.at[index, cols[range(8)]] = session_id, lims_id, mouse, date, cre, stage, area, depth
    
    
        #%%
    
        # samps_bef (=40) frames before omission ; index: 0:39
        # omission frame ; index: 40
        # samps_aft - 1 (=39) frames after omission ; index: 41:79
            
        # Do the analysis only if the experiment is valid
        if validity_log_all.iloc[validity_log_all.lims_id.values == int(lims_id)]['valid'].bool() == False:
            # num_neurons = 1
            local_fluo_allOmitt = np.full((samps_bef + samps_aft, 1, len(list_omitted)), np.nan) # just a nan array as if there was 1 neuron, so this experiment index is not empty in this_sess_l.
            this_sess_l.at[index, 'local_fluo_allOmitt'] = local_fluo_allOmitt
            this_sess_l.at[index, 'valid'] = 0
            print('Skipping invalid experiment %d, index %d' %(int(lims_id), index))
    #                this_sess.at[index, :] = np.nan # check this works.
            
        else:
            # Get traces of all neurons for the entire session
            local_fluo_traces = whole_data[lims_id]['fluo_traces'] # neurons x frames
            local_time_traces = whole_data[lims_id]['time_trace']  # frame times in sec. Volume rate is 10 Hz. Are these the time of frame onsets?? (I think yes... double checking with Jerome/ Marina.)
            num_neurons = local_fluo_traces.shape[0]
    
            # Convert peak_win to frames (new way to code list_times)
            frame_dur = np.mean(np.diff(local_time_traces)) # difference in sec between frames
            # compute peak_win_frames, ie window to look for peak in frame units
            peak_win_frames = samps_bef + np.floor(peak_win / frame_dur).astype(int)
            list_times = np.arange(peak_win_frames[0], peak_win_frames[-1]+1) # [40, 41, 42, 43, 44, 45, 46, 47, 48]
            # for omit-evoked peak timing, compute it relative to samps_bef (which is the index of omission)
            
            # same as above, now for flash-evoked responses :compute peak_win_frames, ie window to look for peak in frame units
            peak_win_frames_flash = samps_bef + np.floor(flash_win / frame_dur).astype(int)
            list_times_flash = np.arange(peak_win_frames_flash[0], peak_win_frames_flash[-1]) # [31, 32, 33, 34]  #, 35, 36, 37, 38, 39]
            # window for measuring the timing of flash-evoked responses
            peak_win_frames_flash_timing = samps_bef + np.floor(flash_win_timing / frame_dur).astype(int)
            list_times_flash_timing = np.arange(peak_win_frames_flash_timing[0], peak_win_frames_flash_timing[-1]) # [30, 31, 32, 33]
            # for flash-evoked peak timing, compute it relative to flash index: 31 
            # although below is not quite accurate, if you want to be quite accurate you should align the traces on flashes!
            flash_index = samps_bef + np.floor(-.75 / frame_dur).astype(int)
            
    #                local_time_traces_all.append(local_time_traces)
    #            depth_all.append(depth)
    #            area_all.append(area)
            
            
            #%% Plot the trace of all neurons (for the entire session) and mark the flash onsets
            '''                
            plt.subplot(8,1,index+1)                
    #                y = local_fluo_traces.T # individual neurons
            y = np.nanmean(local_fluo_traces, axis=0) # average of neurons
            plt.plot(local_time_traces, y)  
            mn = 0; mx = np.max(y)-3*np.std(y)
            plt.vlines(list_flashes, mn, mx)
            plt.title('%s, %dum' %(area, depth), fontsize=12);
            plt.subplots_adjust(hspace=.5)
            plt.gca().tick_params(labelsize=10)
            if index!=7:
                plt.gca().set_xticklabels('')                                        
    #                plt.xlim([0, 30]); # 5*60 / frame_dur # 5 minutes gray
    #                plt.xlim([local_time_traces[-1]-620, local_time_traces[-1]-300]); # 5*60 / frame_dur # 5 minutes gray
            '''
            
            '''           
            # Remember to remove 2 indents to make this work
            # plot the entire trace for each session (after the subplots are made for all planes)           
            # Save the figure
    #        nam = '%s_mouse%d_sess%d_exp%s' %(cre, mouse, session_id, lims_id)
            fign = os.path.join(dir0, dir_now, nam+'.pdf')
            plt.savefig(fign, bbox_inches='tight')
            '''
            
            
            #%% Align on omission trials:
            # samps_bef (=40) frames before omission ; index: 0:39
            # omission frame ; index: 40
            # samps_aft - 1 (=39) frames after omission ; index: 41:79
                    
            # Keep a matrix of omission-aligned traces for all neurons and all omission trials    
            local_fluo_allOmitt = np.full((samps_bef + samps_aft, num_neurons, len(list_omitted)), np.nan) # time x neurons x omissions_trials 
            local_time_allOmitt = np.full((samps_bef + samps_aft, len(list_omitted)), np.nan) # time x omissions_trials
    
            # Loop over omitted trials to align traces on omissions
            flashes_win_trace_all = []
            flashes_win_trace_index_all = []
            num_omissions = 0 # trial number
            
            for iomit in range(len(list_omitted)): # # indiv_time in list_omitted: # 
                
                indiv_time = list_omitted.iloc[iomit]
                
                local_index = np.argmin(np.abs(local_time_traces - indiv_time)) # the index of omission on local_time_traces               
                
                be = local_index - samps_bef
                af = local_index + samps_aft
                
                if ~np.logical_and(be >= 0 , af <= local_fluo_traces.shape[1]): # make sure the omission is at least samps_bef frames after trace beigining and samps_aft before trace end. 
                    print('Omission %d at time %f cannot be analyzed: %d timepoints before it and %d timepoints after it!' %(iomit, indiv_time, be, af)) 
                    
                try:
                    # Align on omission
                    local_fluo_allOmitt[:,:, num_omissions] = local_fluo_traces[:, be:af].T # frame x neurons x omissions_trials (10Hz)
                    # local_time_allOmitt is frame onset relative to omission onset. (assuming that the times in local_time_traces are frame onsets.... still checking on this, but it seems to be the rising edge, hence frame onset time!)
                    # so local_time_allOmitt will be positive if frame onset is after omission onset.
                    #
                    # local_time_allOmitt shows the time of the frame that is closest to omission relative to omission.
                    # (note, the closest frame to omission is called omission frame).
                    # eg if local_time_allOmitt is -0.04sec, it means, the omission frame starts -.04sec before omission.
                    # or if local_time_allOmitt is +0.04sec, it means, the omission frame starts +.04sec before omission.
                    #
                    # Note: the following two quantities are very similar (though time_orig is closer to the truth!): 
                    # time_orig  and time_trace (defined below)
                    # time_orig = np.mean(local_time_allOmitt, axis=1)                            
                    # time_trace, time_trace_new = upsample_time_imaging(samps_bef, samps_aft, 31.) # set time for the interpolated traces (every 3ms time points)
                    #
                    # so we dont save local_time_allOmitt, although it will be more accurate to use it than time_trace, but the
                    # difference is very minimum
                    local_time_allOmitt[:, num_omissions] = local_time_traces[be:af] - indiv_time # frame x omissions_trials
                    
                    
                    
                    # Identify the flashes (time and index on local_time_allOmitt) that happened within local_time_allOmitt                    
                    flashes_relative_timing = list_flashes - indiv_time # timing of all flashes in the session relative to the current omission
                    # the time of flashes that happened within the omit-aligned trace
                    a = np.logical_and(flashes_relative_timing >= local_time_allOmitt[0, num_omissions], flashes_relative_timing <= local_time_allOmitt[-1, num_omissions]) # which flashes are within local_time
                    flashes_win_trace = flashes_relative_timing[a] 
                    # now find the index of flashes_win_trace on local_time_allOmitt
                    a = [np.argmin(np.abs(local_time_allOmitt[:, num_omissions] - flashes_win_trace[i])) for i in range(len(flashes_win_trace))]
                    flashes_win_trace_index = a # the index of omission on local_time_traces
                    '''
                    plt.plot(local_time_allOmitt[:,num_omissions], local_fluo_allOmitt[:,:,num_omissions])
                    plt.vlines(flashes_win_trace, -.4, .4)
    
                    plt.plot(local_fluo_allOmitt[:,:,num_omissions])
                    plt.vlines(flashes_win_trace_index, -.4, .4)
                    '''
                    num_omissions = num_omissions + 1
                    flashes_win_trace_all.append(flashes_win_trace)
                    flashes_win_trace_index_all.append(flashes_win_trace_index)
                    
    #                    time 0 (omissions) values:
    #                    local_fluo_allOmitt[samps_bef,:,:]
    #                    local_time_allOmitt[samps_bef,:]
                    
                except Exception as e:
                    print(indiv_time, num_omissions, be, af, local_index)
#                        print(e)
                                        
                                            
            # remove the last nan rows from the traces, which happen if some of the omissions are not used for alignment (due to being too early or too late in the session) 
            if len(list_omitted)-num_omissions > 0:
                local_fluo_allOmitt = local_fluo_allOmitt[:,:,0:-(len(list_omitted)-num_omissions)]
                local_time_allOmitt = local_time_allOmitt[:,0:-(len(list_omitted)-num_omissions)]
    
    
            local_fluo_allOmitt0_orig = local_fluo_allOmitt + 0      
#                local_fluo_allOmitt = local_fluo_allOmitt0_orig 
                                
            
            this_sess.at[index, ['n_omissions', 'n_neurons']] = num_omissions, num_neurons
            print('===== plane %d: %d neurons; %d trials =====' %(index, num_neurons, num_omissions))
            
    
            #%%#############################################################
            ###### The following two variables are the key variables: ######
            ###### (they are ready to be used)                        ######
            ###### calcium traces aligned on omissions:               ######
            ###### local_fluo_allOmitt  (frames x units x trials)     ######
            ###### and their time traces:                             ######
            ###### local_time_allOmitt  (frames x trials)             ######
            ################################################################
            
            #%% Normalize each neuron trace by its max (so if on a day a neuron has low FR in general, it will not be read as non responsive!)
            ### In this case doScale is automatically set to zero, because:
            # Remember whether you are normalizing to max peak activity or not will not matter if you normalize by baseline SD (next section). because if you normalize to max peak activity, baseline SD will also get changed by that same amount, so after normalizing to baseseline SD the effect of normalizing to max peak activity will be gone.
            
            '''
            for iu in range(num_neurons):
                a = local_fluo_allOmitt[:,iu,:]
                plt.figure()
                plt.plot(a) # all trials
                plt.plot(np.mean(a,axis=1)) # average across trials
                
                for itr in range(num_omissions):
                    plt.plot(a[:,itr])
            
            # entire session                    
            a = local_fluo_allOmitt[:,iu,:].flatten()
            aa = a / np.max(a)
            plt.plot(a)
            plt.plot(aa)
            '''
            if norm_to_max: 
                # not sure if it makes more sense to compute the max on the entire trace of the session, or only on omission-aligned traces
                
                # compute max on the entire trace of the session:
    #                    aa_mx = np.max(local_fluo_traces, axis=1)
                
                # compute max using the omission-aligned traces
                # concatenate traces of all trials for each neuron
                a = np.transpose(local_fluo_allOmitt, (0,2,1)) # frames x trials x units            
                aa = np.reshape(a, (a.shape[0]*a.shape[1], a.shape[2])) # (frames x trials) x units            
                # compute max activity of each neruon during the session 
                aa_mx = np.max(aa, axis=0) # neurons
                
                # normalize each neuron trace by its max
                b = a / aa_mx
                local_fluo_allOmitt = np.transpose(b, (0,2,1)) # frames x units x trials
            
            
            #%% Define baseline (frames of spontanteous activity)
            
            bl_index_pre_omit = np.arange(0,samps_bef) # you will need for response amplitude quantification, even if you dont use it below.
            bl_index_pre_flash = np.arange(0,flash_index)
            
            if bl_gray_screen==1:
                # the only problem with this method is that, we wont necessarily get the same starting point for 
                # A, B omit-aligned traces. Especially for VIP, pre-omit in A is always lower than B. Which is an 
                # interesting finding!
                
                # use the last half of the gray screen at the beginning of the session
                # list_flashes[0]/2 : list_flashes[0]-10  
                ff = np.argwhere(list_flashes > 100)[0] # first flash that happens after 100sec. remember we expect the first flash to be at time ~300sec... but sometimes there is a first flash (or omission) at time 9sec, which makes no sense... in these cases use the timing of the second flash to compute spont frames
                if ff==1:
                    print('UNCANNY: first flash should happen at ~300sec, not at %dsec! ' %list_flashes[0])
                elif ff>1:
                    sys.exit('REALLY?? the first few flashes happened before the expected 5min gray screen!! Investigate this!!!')
                b = np.round((list_flashes[ff]/2.) / frame_dur).astype(int)
                e = np.round((list_flashes[ff]-10) / frame_dur).astype(int)                
                spont_frames = np.arange(b,e) # ~1544
                
                baseline_trace = local_fluo_traces[:, spont_frames].T # frames x neurons
                # compute mean and sd across frames of baseline_trace
                bl_ave = np.mean(baseline_trace, axis=0) # units
                bl_sd = np.std(baseline_trace, axis=0) # units                    
                bl_ave = bl_ave[None,:,None]
                bl_sd = bl_sd[None,:,None]
                
            else:
                               
                # Try this new method: 
                # get the lowest 20 percentile values of the median trace preceding omission and then compute their mean and sd
                baseline_trace0 = local_fluo_allOmitt[bl_index_pre_omit,:,:] # frames x neurons
                n_p = int(samps_bef*.2) # take 20% of baseline frames
                # sort baseline frames, and then take the lowest 20% of them... this will define baseline frames
                baseline_trace = np.sort(baseline_trace0, axis=0)[:n_p, :,:] # 8 frames x units x trials
                # compute mean and sd across frames of baseline_trace                    
                bl_ave = np.mean(baseline_trace, axis=0) # units x trials
                bl_sd = np.std(baseline_trace, axis=0) # units x trials
                bl_ave = bl_ave[None,:,:]
                bl_sd = bl_sd[None,:,:]
                
                # Old method: use frames preceding omission as baseline (not a good idea! bc flashes are so frequent, that the pre-omit activity is not really spont!)
                '''
                baseline_trace = local_fluo_allOmitt[bl_index_pre_omit,:,:] # frames x neurons
                # compute mean and sd across frames of baseline_trace                    
                bl_ave = np.mean(baseline_trace, axis=0) # units x trials
                bl_sd = np.std(baseline_trace, axis=0) # units x trials
    #                    bl_ave = bl_ave[None,:,:]
    #                    bl_sd = bl_sd[None,:,:]                                
                '''
                # the problem with below, ie using percentile, is that, i don't know how to compute an equivalent of bl_sd for it. 
                # use the 10th percentile of trace preceding the omission; compute it on triral-median trace to reduce noise                                      
                '''
                baseline_trace = np.median(local_fluo_allOmitt[bl_index_pre_omit,:,:], axis=2) # frames x neurons 
                bl_ave = np.percentile(baseline_trace, 10, axis=0) # neurons 
                '''                         
                
                
            #%% Normalize the average trace by baseline std (ie pre-omitted fluctuations)
            
    #                local_fluo_allOmitt0_orig = local_fluo_allOmitt + 0
    #                local_fluo_allOmitt = local_fluo_allOmitt0_orig + 0
            
            if doShift:  # shift the y values so baseline mean is at 0 at time 0            
                local_fluo_allOmitt = local_fluo_allOmitt - bl_ave
                        
            if doScale:  # scale the traces by baseline std
                local_fluo_allOmitt = local_fluo_allOmitt / bl_sd
                
            
            #%% Keep local_fluo_allOmitt for all planes
            
            this_sess_l.at[index, 'local_fluo_allOmitt'] = local_fluo_allOmitt
            this_sess_l.at[index, 'valid'] = 1
#                local_fluo_allOmitt_allPlanes.append(local_fluo_allOmitt)
    
    
            #%% 
            ######################################################################
            ######################################################################
            ######################################################################
            ######################################################################
                                            
            #%% Set omit-aligned median traces, compute peaks, etc.                
            
            if doCorrs==0:                    
                #%% Keep trial-averaged traces
                
    #                traces_aveTrs_time_ns0 = np.median(local_fluo_allOmitt0_orig, axis=2)
                
                ### NOTE: tested on Slc mouse 451787, session_id 885557130:
                ### There is a big difference between mean and median ... median seems to be a quite better measure!!
                ### So I am going with median and iqr!
                                
                traces_aveTrs_time_ns = np.median(local_fluo_allOmitt, axis=2) # time x neurons # Average across trials            
                
                traces_aveNs_time_trs = np.median(local_fluo_allOmitt, axis=1) # time x trials  # Average across neurons
                # currently not saving the following ...             
    #            traces_sdTrs_time_ns = st.iqr(local_fluo_allOmitt, axis=2) # time x neurons # Std across trials            
    #            traces_sdNs_time_trs = st.iqr(local_fluo_allOmitt, axis=1) # time x trials  # Std across neurons
                traces_aveTrs_time_ns0 = traces_aveTrs_time_ns + 0
    
    
                #%% Compute baseline of (normalized) traces (as 10th percentile pre-omit activity)
                # Also, perhaps do a second shift just to make sure the pre-omit activity has baseline at 0.
                
                # you may decide not to do it, if you want to keep the baseline information.
                # or you may decide to do it, if you want to make the comparison of traces independent of their baseline changes.
                
                # we will anyway do it, when it comes to the peak amplitude computation.
                
                ##### Remember: if you use median across trials (instead of mean) to compute aveTrs traces, the baseline will not be anymore at 0 
                # even if you have shifted it to 0... because baseline is computed on the mean of each trial during bl_index_pre_omit!
                # so to compute peak on aveTrs traces, we will again shift the traces below so their baseline is at 0.
                
                # try 10th or 20th percentile
                bl_preOmit = np.percentile(traces_aveTrs_time_ns[bl_index_pre_omit,:], bl_percentile, axis=0) # neurons # use the 10th percentile
                bl_preFlash = np.percentile(traces_aveTrs_time_ns[bl_index_pre_flash,:], bl_percentile, axis=0) # neurons
    #                bl_preOmit = np.mean(traces_aveTrs_time_ns[bl_index_pre_omit,:], axis=0) # use average                   
    
                # I think I should only use the following for the computation of the peak
                # because for peak amplitude I want to see how much the trace changed... but 
                # I rather not change the trace: traces_aveTrs_time_ns, and save it the way it is, only
                # spontaneous activity (during the initial 5min gray screen) is subtraced.
                
                if doShift_again: # this is a second shift just to make sure the pre-omit activity has baseline at 0.                                       
    #                    traces_aveTrs_time_ns = traces_aveTrs_time_ns0 + 0                    
                    traces_aveTrs_time_ns = traces_aveTrs_time_ns0 - bl_preOmit
                                                         
    
    
                #%%                    
                this_sess.at[index, 'traces_aveTrs_time_ns'] = traces_aveTrs_time_ns # the time of flashes that happened within the omit-aligned trace
                this_sess.at[index, 'traces_aveNs_time_trs'] = traces_aveNs_time_trs # the time of flashes that happened within the omit-aligned trace
                
    
                #%% Plot omit-aligned traces for each neuron / trial ; also make average plots
                
                if doPlots: # 1
                    
                    get_ipython().magic(u'matplotlib inline') # %matplotlib inline
    
                    # show trial-averaged and neuron-average traces
                    plt.figure(figsize=(4.5,6))
                    
                    plt.suptitle('%s, mouse %d' %(cre, mouse), y=.98, fontsize=18)
                    # show trial-averaged, all neurons
                    plt.subplot(211)
                    plt.plot(traces_aveTrs_time_ns, 'gray') # show trial-averaged, all neurons
                    
                    avD = np.median(traces_aveTrs_time_ns, axis=-1)
                    sdD = st.iqr(traces_aveTrs_time_ns, axis=-1) # = q75-q25
                    q25 = np.percentile(traces_aveTrs_time_ns, 25, axis=-1) # in average and std case, this term is like av-sd
                    q75 = np.percentile(traces_aveTrs_time_ns, 75, axis=-1) # in average and std case, this term is like av+sd
                    
                    plt.fill_between(range(samps_bef+samps_aft), q25, q75, alpha=0.3, edgecolor='r', facecolor='r', zorder=10)
    
                    plt.plot(avD, 'r', marker='.') # average across neurons (each trace is trial-average)
                    mn = min(traces_aveTrs_time_ns.flatten())
                    mx = max(traces_aveTrs_time_ns.flatten())
                    plt.vlines(flashes_win_trace_index_all[np.random.permutation(num_omissions)[0]], mn, mx, color='black', linestyle='dashed') # showing when the flashes happened for the 1st omission ... it is not easy to show it on the trial-averaged trace. as the timing of flashes varies a bit across all omission-aligned traces  
                    plt.title('Trial-averaged, each neuron', fontsize=13.5)
                    plt.xlabel('Frame (~10Hz)', fontsize=12)
                    plt.ylabel('DF/F') # %s' %(ylab), fontsize=12)
                    
                    
                    # show neuron-averaged, all trials
                    plt.subplot(212)
                    plt.plot(traces_aveNs_time_trs, 'gray')
                    
                    avD = np.median(traces_aveNs_time_trs, axis=-1)
                    sdD = st.iqr(traces_aveNs_time_trs, axis=-1)
                    q25 = np.percentile(traces_aveNs_time_trs, 25, axis=-1)
                    q75 = np.percentile(traces_aveNs_time_trs, 75, axis=-1)
                    
                    plt.fill_between(range(samps_bef+samps_aft), q25, q75, alpha=0.3, edgecolor='r', facecolor='r', zorder=10)
                    
                    plt.plot(avD, 'r', marker='.') # average across trials (each trace is neuron-average)
                    mn = min(traces_aveNs_time_trs.flatten()) #(traces_allTrsNs[:,0])
                    mx = max(traces_aveNs_time_trs.flatten())                
                    plt.vlines(flashes_win_trace_index_all[np.random.permutation(num_omissions)[0]], mn, mx, color='black', linestyle='dashed') # showing when the flashes happened for the 1st omission ... it is not easy to show it on the trial-averaged trace. as the timing of flashes varies a bit across all omission-aligned traces 
                    plt.title('Neuron-averaged, each trial', fontsize=13.5)
                                    
                    plt.subplots_adjust(hspace=.55) # wspace=.25, 
                    
                    
                    '''
                    # plots of each neuron (all trials)  # time x neurons x trials
                    for iu in range(num_neurons): # iu = 2
                        trac = local_fluo_allOmitt[:,iu,:] # time x trials  # one neuron, all trials                  
                        plt.figure()
                        plt.plot(trac, color='gray') # 1 neuron, all trials
                        plt.plot(np.mean(trac, axis=1), 'r') # average across trials
                    
                    
                    # plots of each trial (all neurons)
                    for itr in range(num_omissions):
    #                    itr = np.random.permutation(num_omissions)[0]
                        ave_allNs_1tr = np.mean(local_fluo_allOmitt[:,:,itr], axis=-1) # neuron-averaged trace for trial itr
    #                    iu = np.random.permutation(num_neurons)[0]
                        trace = local_fluo_allOmitt[:,:,itr] # all neuron traces, 1 trial
                        mn = min(trace.flatten())
                        mx = max(trace.flatten())
                        
                        plt.figure()
                        plt.plot(trace, 'gray') # all neuron traces, 1 trial
                        plt.plot(ave_allNs_1tr, 'r') # neuron-averaged trace for trial itr
                        plt.vlines(flashes_win_trace_index_all[itr], mn, mx)
                    '''
    
                     
                    #####                
    #                a = np.mean(local_fluo_allOmitt, axis=2) # average across trials: time x neurons
    #                aav = np.mean(a, axis=1) 
    #
    #                plt.figure()                
    #                plt.plot(a, 'gray') # each neuron
    #                plt.plot(aav, color='red', marker='.') # average across neurons        
    #                plt.vlines(flashes_win_trace_index_all[0], -.05, .05) # showing when the flashes happened for the 1st omission ... it is not easy to show it on the trial-averaged trace. as the timing of flashes varies a bit across all omission-aligned traces 
    
    #                a_norm1 = a
    #                aav_norm1 = np.mean(a_norm1, axis=1)
    #                plt.plot(a_norm1,'green')
    #                plt.plot(aav_norm1, 'magenta')
                                    
                    
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
                    peak_amp_eachN_traceMed = np.mean(traces_aveTrs_time_ns[list_times, :], axis=0) # neurons
                    
                else: # compute peak
                    if cre.find('Vip')==-1: # SLC, SST # if the mouse is NOT a VIP line, instead of computing the peak, we compute the trough: we invert the trace, and then compute the max. (this is because in VIP mice the response after omission goes up, but not in SST or Slc)
                        if index==0:
                            print('Peak amplitude cannot be computed for SST or SLC lines, because their response after omission does not increase, setting to NaN!')
                        peak_amp_eachN_traceMed = np.full((num_neurons), np.nan)
                        ###### NOT GOOD!!! we cant really do the following! maybe some SST and SLC neurons incrase activity or dont respond after omission
                        # we cant just inverse their trace
    #                        print('BE CAUTIOUS: to compute peak amplitude, we are inverting the trace assuming it is inhibited by omission!')
    #                        peak_amp_eachN_traceMed = -np.max(-traces_aveTrs_time_ns[list_times, :], axis=0) # neurons
                    else: # VIP, easy, just get the max
                        peak_amp_eachN_traceMed = np.max(traces_aveTrs_time_ns[list_times, :], axis=0) # neurons
                peak_amp_eachN_traceMed0 = peak_amp_eachN_traceMed + 0
                
    
                ###### peak timing : wont be accurate, because not always the response goes up after omission (ie peak cant be computed always) ######
                if cre.find('Vip')==-1: # SLC, SST   
                    if index==0:
                        print('Peak timing cannot be computed for SST or SLC lines, because their response after omission does not increase, setting to NaN!')
                    peak_timing_eachN_traceMed = np.full((num_neurons), np.nan)
    #                        print('BE CAUTIOUS: to compute peak timing, we are inverting the trace assuming it is inhibited by omission! Be cautious!')
    #                    peak_timing_eachN_traceMed = np.argmax(-traces_aveTrs_time_ns[list_times, :], axis=0)# neurons
                else: # VIP, easy, just get the argmax
                    peak_timing_eachN_traceMed = np.argmax(traces_aveTrs_time_ns[list_times, :], axis=0) # neurons
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
                        if np.mean(traces_aveTrs_time_ns[np.arange(samps_bef, samps_bef+8), :], axis=0)[ineur] < bl[ineur]: # peak_amp_eachN_traceMed[ineur]
                            # we find the max on the inverse trace, but the value we assign to peak amp is the actual value (before the inverse)
                            peak_amp_eachN_traceMed[ineur] = -np.max(-traces_aveTrs_time_ns[list_times, ineur], axis=0) # neurons
                            peak_timing_eachN_traceMed[ineur] = np.argmax(-traces_aveTrs_time_ns[list_times, ineur], axis=0)# neurons
                '''
                                    
        
                #%% Subtract baseline of the omit-aligned trace from its peak to compute the relative change in peak amplitude.
                # compute how much the trace max (or mean trace) (during peak_win) has changed relative to
                # the min (or 10th percentile) activity right before omission
                
                if doShift_again==0:
                    peak_amp_eachN_traceMed = peak_amp_eachN_traceMed0 - bl_preOmit
                
                # below should have baseline at 0
    #                plt.plot(traces_aveTrs_time_ns[:,3]-bl_preOmit[3])                
                
                this_sess.at[index, ['peak_amp_eachN_traceMed']] = [peak_amp_eachN_traceMed] # neurons
                this_sess.at[index, ['peak_timing_eachN_traceMed']] = [peak_timing_eachN_traceMed] # neurons
        
    
    
    
                #%% FLASHES: Do the same as above except for flash-evoked responses: compute response amp on trial-median traces
                
                ###### response amplitude : either max (peak) or mean during peak_win. Mean is preferred because SST and SLC responses don't increase after omission ######
                # for each trial-median neuron trace find the peak during the peak_win window
                if mean_notPeak: # compute average trace during peak_win
                    peak_amp_eachN_traceMed_flash = np.mean(traces_aveTrs_time_ns[list_times_flash, :], axis=0) # neurons
                else: # compute peak (unlike omissions, we know all cell lines respond to flash, so we just get the max)
                    peak_amp_eachN_traceMed_flash = np.max(traces_aveTrs_time_ns[list_times_flash, :], axis=0) # neurons
                peak_amp_eachN_traceMed_flash0 = peak_amp_eachN_traceMed_flash + 0
                
    
                ###### peak timing : wont be accurate, because not always the response goes up after omission (ie peak cant be computed always) ######
                # unlike omissions, we know all lines respond to flash, so we just get the argmax regadless of cell line
                peak_timing_eachN_traceMed_flash = np.argmax(traces_aveTrs_time_ns[list_times_flash_timing, :], axis=0) # neurons
                # compute timing relative to flash time (remember flash_index is not quite accurate, I think); to be accurate you should align on the flash, instead of getting the flash timing from omission aligned traces.
                peak_timing_eachN_traceMed_flash = peak_timing_eachN_traceMed_flash + (list_times_flash_timing[0] - flash_index)
                
        
                #%% Subtract baseline of the omit-aligned trace from its peak to compute the relative change in peak amplitude.
                # compute how much the trace max (or mean trace) (during peak_win) has changed relative to
                # the min (or 10th percentile) activity right before omission
                
                if doShift_again==0:
                    peak_amp_eachN_traceMed_flash = peak_amp_eachN_traceMed_flash0 - bl_preOmit
    #                    peak_amp_eachN_traceMed_flash = peak_amp_eachN_traceMed_flash0 - bl_preFlash
                
    
                #%%
                this_sess.at[index, ['peak_amp_eachN_traceMed_flash']] = [peak_amp_eachN_traceMed_flash] # neurons
                this_sess.at[index, ['peak_timing_eachN_traceMed_flash']] = [peak_timing_eachN_traceMed_flash] # neurons
    
    
    
    
                
                #%% Find amplitude of peak response on each omission trial (noisier compared to peak_amp_eachN_traceMed which is computed on trial-median trace)
                            
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
                        peak_allTrsNs[itrial,:] = np.mean(a[list_times,:], axis=0) # neurons
                    else:
                        peak_allTrsNs[itrial,:] = np.max(a[list_times,:], axis=0) # neurons
    #                peak_allTrsNs.append(np.max(a[:,list_times], axis = 1))
                    
                    # for each trial and each neuron find the timing (frame number) of the max response during [0,2] sec after omission trial    
                    # compute peak timing relative to frame of omission
                    peak_timing_allTrsNs[itrial,:] = np.argmax(a[list_times,:], axis=0) # neurons
    #                peak_timing_allTrsNs[itrial,:] = list_times[0] + np.argmax(a[list_times,:], axis=0) # neurons # peak timing relative to the trace onset
    
                ### some tests :
    #            plt.hist(peak_allTrsNs[:,0])
    #            plt.hist(local_fluo_allOmitt[samps_bef+10, 0, :])
    #            np.median(peak_allTrsNs[:,0])
    #            np.median(local_fluo_allOmitt[samps_bef+10, 0, :])
                    
                
                this_sess.at[index, ['peak_amp_trs_ns']] = [peak_allTrsNs] # trials x neurons
                this_sess.at[index, ['peak_timing_trs_ns']] = [peak_timing_allTrsNs] # trials x neurons
                
                
                
                
                
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
                        plt.plot(om_time, np.mean(om_av_half2 - om_av_half1, axis=1), 'r')
                        plt.axvline(0)
                        plt.title([depth, area])
                        plt.ylim([-1,1])
                
                    
                    #%% Find the peak amplitude for each neuron on trial-averaged traces of the 1st half of the session
                            
                    # find peak on window: peak_win = [0, 2] # sec relative to omission onset (will be applied to om_time vector)
                    list_times = np.argwhere(np.all([om_time > peak_win[0], om_time < peak_win[1]], axis=0)).flatten()
                    
                    # for each neuron find amplitude of max response during [0,2] sec after omission trial    
                    # 1st half of the session
                    peak_om_av_h1 = np.max(om_av_half1[list_times,:], axis = 0) # neurons
                    # 2nd half of the session
                    peak_om_av_h2 = np.max(om_av_half2[list_times,:], axis = 0) # neurons
                    
                    this_sess.at[index, ['peak_h1', 'peak_h2']] = peak_om_av_h1, peak_om_av_h2
                    
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
                    
                    this_sess.at[index, 'auc_peak_h1_h2'] = auc_peak_h1_h2    


    
    #%% 
    #######################################################################
    #######################################################################
    
    #%% Once we have the omit-aligned traces of all experiments, run correlations between neuron paris in different areas
    
    #####
    colsn = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', \
                      'cc_cols_area12', 'cc_cols_areaSame', \
                      'cc_a12', 'cc_a11', 'cc_a22', 'p_a12', 'p_a11', 'p_a22', \
                      'cc_a12_shfl', 'cc_a11_shfl', 'cc_a22_shfl', 'p_a12_shfl', 'p_a11_shfl', 'p_a22_shfl'])
    this_sess_allExp = pd.DataFrame([], columns = colsn)

    this_sess_allExp.at[0, 'session_id'] = this_sess['session_id'].values[0]

    this_sess_allExp.at[0, 'mouse_id'] = this_sess['mouse_id'].values[0]
    this_sess_allExp.at[0, 'date'] = this_sess['date'].values[0]
    this_sess_allExp.at[0, 'cre'] = this_sess['cre'].values[0]
    this_sess_allExp.at[0, 'stage'] = this_sess['stage'].values[0]

    this_sess_allExp.at[0, 'n_omissions'] = this_sess[~np.isnan(this_sess['n_omissions'].values.astype(float))]['n_omissions'].values[0] # if an experiment is invalid its n_omissions will be nan, so we make sure we get a non-nan value  #this_sess['n_omissions'].values[0]

    this_sess_allExp.at[0, 'experiment_id'] = this_sess['experiment_id'].values
    this_sess_allExp.at[0, 'area'] = this_sess['area'].values
    this_sess_allExp.at[0, 'depth'] = this_sess['depth'].values
    this_sess_allExp.at[0, 'n_neurons'] = this_sess['n_neurons'].values


    if np.logical_and(len(list_omitted)>0, doCorrs):                
        #%% Compute correlation coefficient (also try this without shifting and scaling the traces!)
        # between a layer of area1 and and a layer of area2
        # do this on all pairs of neurons
        
        # compute correlation coeffiecient of frame ifr in omission-align traces (eg 3 frames before omission begins) 
        # between neuron iu1 (in area1, layer d1) and neuron iu2 (in area2, layer d2)
        # basically, asking how the activity of neurons covaries eg 3 frames before omission
        # eg if all neurons predict the omission, then their activities will covary.
        # we want to see if the occurance of flash or omission, changes the coactivation pattern of 
        # neurons across specific layers of the 2 areas.
        
        print('Computing pairwise correaltions...')
        
        num_depth = int(num_planes/2)
        inds_lm = np.arange(num_depth) # we can get these from the following vars: distinct_areas and i_areas[range(num_planes)]
        inds_v1 = np.arange(num_depth, num_planes)
#            print(inds_lm); print(inds_v1); print(np.shape(this_sess_l))
        
        # area1: LM
        # area 2: V1
        local_fluo_allOmitt_a1 = this_sess_l.iloc[inds_lm]['local_fluo_allOmitt'].values # 4
        local_fluo_allOmitt_a2 = this_sess_l.iloc[inds_v1]['local_fluo_allOmitt'].values # 4
        
        valid_a1 = this_sess_l.iloc[inds_lm]['valid'].values # 4
        valid_a2 = this_sess_l.iloc[inds_v1]['valid'].values # 4
        
        nfrs = local_fluo_allOmitt_a1[0].shape[0]
        n_neurs_a1 = [local_fluo_allOmitt_a1[iexp].shape[1] for iexp in range(len(local_fluo_allOmitt_a1))] # number of neurons for each depth of area 1
        n_neurs_a2 = [local_fluo_allOmitt_a2[iexp].shape[1] for iexp in range(len(local_fluo_allOmitt_a2))] # number of neurons for each depth of area 2
        print(n_neurs_a1, n_neurs_a2)
        
        
        ########## corr of area1 - area2 ##########
        ###########################################        
        # each column is corrcoeff between a layer of area1 and and a layer of area2; 
        # area1: LM; area 2: V1
        # eg. 'l32': corrcoeff between layer3, LM and layer2, V1.
        
        # in cols_a1a2: rows are area1 (LM) layers; colums are area2 (V1) layers
        '''
#            cols = cc_a1_a2.columns.values
#            cols_a1a2 = np.reshape(cols, (4,4), order='C')
        cols_a1a2 = [['l00', 'l01', 'l02', 'l03'],
                    ['l10', 'l11', 'l12', 'l13'],
                    ['l20', 'l21', 'l22', 'l23'],
                    ['l30', 'l31', 'l32', 'l33']]
        '''            
        cols = ['l00','l01','l02','l03',\
                 'l10','l11','l12','l13',\
                 'l20','l21','l22','l23',\
                 'l30','l31','l32','l33']
        
        
#        c12_a12 = np.full((nfrs, len(cols), np.max([n_neurs_a1, n_neurs_a2])*np.max([n_neurs_a1, n_neurs_a2])), np.nan) # frames x layerPairs x all_neuronPairs
#        p12_a12 = np.full((nfrs, len(cols), np.max([n_neurs_a1, n_neurs_a2])*np.max([n_neurs_a1, n_neurs_a2])), np.nan)
        c12_a12 = []
        p12_a12 = []
        c12_a12_shfl = []
        p12_a12_shfl = []
        pw = -1 # column index: a combination of layers between area1 and area2
        for d1 in range(num_depth): # d1=0
            area1 = local_fluo_allOmitt_a1[d1]
            nu1 = area1.shape[1]
            v1 = valid_a1[d1]
            
            for d2 in range(num_depth): # d2=0
                print('area 1 layer %d - area 2 layer %d' %(d1, d2))
                pw = pw+1
                area2 = local_fluo_allOmitt_a2[d2]
                nu2 = area2.shape[1]
                v2 = valid_a2[d2]

                # now that we picked a layer combination (between area 1 and area2),
                # loop through frames: for each frame, go through all neuron pairs to compute corrcoeff.
                c_a12 = np.full((nfrs, nu1*nu2), np.nan)
                p_a12 = np.full((nfrs, nu1*nu2), np.nan)
                if corr_doShfl:
                    c_a12_shfl = np.full((nfrs, nu1*nu2, num_shfl_corr), np.nan)
#                    p_a12_shfl = np.full((nfrs, nu1*nu2, num_shfl_corr), np.nan)        
                
                if np.logical_and(v1+v2==2, nu1+nu2>2): # make sure both planes are valid (ie the traces are non-NaN), otherwise spearman wont work! Also make sure there is more than a single neuron pair, otherwise cs will have a non-square shape below!  
                    for ifr in range(nfrs): # ifr=0 
#                        print('frame %d' %ifr)
                        # spearman r (does not assume normal distribution for inputs);  also provides the p value.
                        t1 = area1[ifr, :, :] # units x trials
                        t2 = area2[ifr, :, :] # units x trials
                                            
    #                    if np.isnan(t1).all() + np.isnan(t2).all()!=True: # make sure the plane is valid (ie the traces are non-NaN), otherwise spearman wont work! 
                        cs, ps = st.spearmanr(t1.T, t2.T) # (nu1+nu2) x (nu1+nu2) # how the activity of neurons covaries across trials  (input: each column: a variable)
            
                        cs_across = cs[:nu1, nu1:] # nu1 x nu2   # area1-area2 corr
                        ps_across = ps[:nu1, nu1:] # nu1 x nu2   # area1-area2 corr
    
                        c_a12[ifr,:] = cs_across.flatten() # 80 x (nu1 x nu2)
                        p_a12[ifr,:] = ps_across.flatten() # 80 x (nu1 x nu2)
    
                        ####################### Shuffled #######################
                        # shuffle trials, then compute corrcoefftween neurons... this serves as control to evaluate the values of corrcoeff of actual data (I'm worried they might be affected by how active neurons are... this will control for that)!
                        # although if variation across trials is different for different frames (eg after omission there is less variability in the response of a neuron across trials, compared to before omission)
                        # , then shuffling will affect different frames differently... (eg after omission because of the lower variability, shufflign will less change the corrcoeff compared to actual data, but before omission, bc of the higher variability across trials, shuffling will change corrcoef more compared to actual data.)
                        # I guess so!
                        if corr_doShfl:
                            for ish in range(num_shfl_corr): # for each frame, we get corrcoeff between trial-shuffled traces. we do this num_shfl times. 
                                print('\tshuffle %d' %ish)
                                t1_shfl = t1[:, rnd.permutation(t1.shape[1])] # import numpy.random as rnd
                                t2_shfl = t2[:, rnd.permutation(t2.shape[1])]
                            
        #                        if np.isnan(t1).all() + np.isnan(t2).all()!=True: # make sure the plane is valid (ie the traces are non-NaN), otherwise spearman wont work! 
                                cs, ps = st.spearmanr(t1_shfl.T, t2_shfl.T) # (nu1+nu2) x (nu1+nu2) # how the activity of neurons covaries across trials  (input: each column: a variable)
                                
                                cs_across = cs[:nu1, nu1:] # nu1 x nu2   # area1-area2 corr
#                                ps_across = ps[:nu1, nu1:] # nu1 x nu2   # area1-area2 corr
                                
                                c_a12_shfl[ifr,:,ish] = cs_across.flatten() # 80 x (nu1 x nu2) # NOTE: you can just sum this, since you dont actually keep the arrays! and just get their average!
#                                p_a12_shfl[ifr,:,ish] = ps_across.flatten() # 80 x (nu1 x nu2)

                c12_a12.append(c_a12) # 16 (layer pairs, in the order of cols); each element: 80 * (nu1*nu2) (nu1 and nu2 are the number of neurons in that layer pair)
                p12_a12.append(p_a12)
                if corr_doShfl:
                    c12_a12_shfl.append(c_a12_shfl) # 16 (layer pairs, in the order of cols); each element: 80 * (nu1*nu2) * num_shfl (nu1 and nu2 are the number of neurons in that layer pair)
#                    p12_a12_shfl.append(p_a12_shfl)

#        plt.imshow(c12_a12[0], aspect='auto')
                
                    
#                    c = [] # pair of neurons (iu1 x iu2) for d1 and d2
#                    for iu1 in range(nu1): # iu1 = 0
#                        for iu2 in range(nu2): # iu2 = 0
#                            trace1 = area1[ifr, iu1, :] # trials # area 1 
#                            trace2 = area2[ifr, iu2, :] # trials # area 2
#                            # compute correlation coeffiecient of frame ifr in omission-align traces (eg 3 frames before omission begins) 
#                            # between neuron iu1 (in area1, layer d1) and neuron iu2 (in area2, layer d2)
#                            c.append(np.corrcoef(trace1, trace2)[0,1]) # (nu1 x nu2) 
#                    c = np.array(c)
#                    cc_a1_a2.at[ifr, cols[pw]] = c # c has size: (nu1 x nu2)
                    
            

        ########## corr of area1 - area1 ##########
        ###########################################
        # len(colsas): 10
        colsas = ['l00','l01','l02','l03',\
                        'l11','l12','l13',\
                              'l22','l23',\
                                    'l33']  
#        c12_a11 = np.full((nfrs, len(colsas), np.max(n_neurs_a1)*np.max(n_neurs_a1)), np.nan) # frames x layerPairs x all_neuronPairs
#        p12_a11 = np.full((nfrs, len(colsas), np.max(n_neurs_a1)*np.max(n_neurs_a1)), np.nan)
        c12_a11 = []
        p12_a11 = []
        c12_a11_shfl = []
        p12_a11_shfl = []
        pw = -1 # column index: a combination of layers between area1 and area2
        for d1 in range(num_depth): # d1=0
            area1_d1 = local_fluo_allOmitt_a1[d1]
            nu1 = area1_d1.shape[1]
            v1 = valid_a1[d1]
            
            for d2 in np.arange(d1, num_depth): # d2=0 # d2=d1:4
                print('area 1 layer %d - area 1 layer %d' %(d1, d2))
                pw = pw+1
                area1_d2 = local_fluo_allOmitt_a1[d2]
                nu2 = area1_d2.shape[1]
                v2 = valid_a1[d2]

                # now that we picked a layer combination (between area 1 and area2),
                # loop through frames: for each frame, go through all neuron pairs to compute corrcoeff.
                c_a11 = np.full((nfrs, nu1*nu2), np.nan)
                p_a11 = np.full((nfrs, nu1*nu2), np.nan)
                if corr_doShfl:
                    c_a11_shfl = np.full((nfrs, nu1*nu2, num_shfl_corr), np.nan)
#                    p_a11_shfl = np.full((nfrs, nu1*nu2, num_shfl_corr), np.nan)        
                
                if np.logical_and(v1+v2==2, nu1+nu2>2): # make sure both planes are valid (ie the traces are non-NaN), otherwise spearman wont work! Also make sure there is more than a single neuron pair, otherwise cs will have a non-square shape below!  
                    for ifr in range(nfrs): # ifr=0 
#                        print('frame %d' %ifr)
                        # spearman r (does not assume normal distribution for inputs);  also provides the p value.
                        t1 = area1_d1[ifr, :, :] # units x trials
                        t2 = area1_d2[ifr, :, :] # units x trials
                        
#                        if np.logical_and(np.isnan(t1).all() + np.isnan(t2).all()!=True, nu1+nu2>2): # make sure the plane is valid (ie the traces are non-NaN), otherwise spearman wont work! 
                        cs, ps = st.spearmanr(t1.T, t2.T) # (nu1+nu2) x (nu1+nu2) # how the activity of neurons covaries across trials  (input: each column: a variable)
            
                        cs_across = cs[:nu1, nu1:] # nu1 x nu2   # area1_d1 - area1_d2 corr
                        ps_across = ps[:nu1, nu1:] # nu1 x nu2   # area1_d1 - area1_d2 corr           
                        if d1==d2: # take only the upper triangle when d1 and d2 are the same (because it is the same pair of neurons; also the diagonal is corr of a neuron with itself, we dont want it!)
                            ii = np.tril_indices_from(cs_across)
                            cs_across[ii] = np.nan
                            ps_across[ii] = np.nan
#                        plt.imshow(cs_across)
                            # Remove nans to save space! you have to change the size of c_a11 defined above to: 
                            # i2 = np.tril_indices(nu1)
                            # new_size = nu1*nu2 - len(i2[0])
#                            cnow = cs_across.flatten()
#                            cnow = cnow[~np.isnan(cnow)]
#                            pnow = ps_across.flatten()
#                            pnow = pnow[~np.isnan(pnow)]

                        c_a11[ifr,:] = cs_across.flatten() # 80 x (nu1 x nu2)
                        p_a11[ifr,:] = ps_across.flatten() # 80 x (nu1 x nu2)                                     

                        ####################### Shuffled #######################
                        # shuffle trials, then compute corrcoefftween neurons... this serves as control to evaluate the values of corrcoeff of actual data (I'm worried they might be affected by how active neurons are... this will control for that)!
                        if corr_doShfl:
                            for ish in range(num_shfl_corr): # for each frame, we get corrcoeff between trial-shuffled traces. we do this num_shfl times. 
                                print('\tshuffle %d' %ish)
                                t1_shfl = t1[:, rnd.permutation(t1.shape[1])] # import numpy.random as rnd
                                t2_shfl = t2[:, rnd.permutation(t2.shape[1])]
    
        #                        if np.logical_and(np.isnan(t1).all() + np.isnan(t2).all()!=True, nu1+nu2>2): # make sure the plane is valid (ie the traces are non-NaN), otherwise spearman wont work! 
                                cs, ps = st.spearmanr(t1_shfl.T, t2_shfl.T) # (nu1+nu2) x (nu1+nu2) # how the activity of neurons covaries across trials  (input: each column: a variable)
                    
                                cs_across = cs[:nu1, nu1:] # nu1 x nu2   # area1_d1 - area1_d2 corr
#                                ps_across = ps[:nu1, nu1:] # nu1 x nu2   # area1_d1 - area1_d2 corr           
                                if d1==d2: # take only the upper triangle when d1 and d2 are the same (because it is the same pair of neurons; also the diagonal is corr of a neuron with itself, we dont want it!)
                                    ii = np.tril_indices_from(cs_across)
                                    cs_across[ii] = np.nan
#                                    ps_across[ii] = np.nan
    #                        plt.imshow(cs_across)
                                    # Remove nans to save space!
    #                                cnow = cs_across.flatten()
    #                                cnow = cnow[~np.isnan(cnow)]
    #                                pnow = ps_across.flatten()
    #                                pnow = pnow[~np.isnan(pnow)]
    
                                c_a11_shfl[ifr,:,ish] = cs_across.flatten() # 80 x (nu1 x nu2)
#                                p_a11_shfl[ifr,:,ish] = ps_across.flatten() # 80 x (nu1 x nu2)


                c12_a11.append(c_a11) # 10 (layer pairs, in the order of cols); each element: 80 * (nu1*nu2) (nu1 and nu2 are the number of neurons in that layer pair)
                p12_a11.append(p_a11)
                if corr_doShfl:
                    c12_a11_shfl.append(c_a11_shfl) # 10 (layer pairs, in the order of cols); each element: 80 * (nu1*nu2) * num_shfl (nu1 and nu2 are the number of neurons in that layer pair)
#                    p12_a11_shfl.append(p_a11_shfl)


        ########## corr of area2 - area2 ##########
        ###########################################
#        colsas = ['l00','l01','l02','l03',\
#                        'l11','l12','l13',\
#                              'l22','l23',\
#                                    'l33']        
#        c12_a22 = np.full((nfrs, len(colsas), np.max(n_neurs_a2)*np.max(n_neurs_a2)), np.nan) # frames x layerPairs x all_neuronPairs
#        p12_a22 = np.full((nfrs, len(colsas), np.max(n_neurs_a2)*np.max(n_neurs_a2)), np.nan)
        c12_a22 = []
        p12_a22 = []
        c12_a22_shfl = []
        p12_a22_shfl = []        
        pw = -1 # column index: a combination of layers between area1 and area2
        for d1 in range(num_depth): # d1=0
            area2_d1 = local_fluo_allOmitt_a2[d1]
            nu1 = area2_d1.shape[1]
            v1 = valid_a2[d1]
            
            for d2 in np.arange(d1, num_depth): # d2=1
                print('area 2 layer %d - area 2 layer %d' %(d1, d2))
                pw = pw+1
                area2_d2 = local_fluo_allOmitt_a2[d2]
                nu2 = area2_d2.shape[1]
                v2 = valid_a2[d2]
                
                # now that we picked a layer combination (between area 1 and area2),
                # loop through frames: for each frame, go through all neuron pairs to compute corrcoeff.
                c_a22 = np.full((nfrs, nu1*nu2), np.nan)
                p_a22 = np.full((nfrs, nu1*nu2), np.nan)      
                if corr_doShfl:
                    c_a22_shfl = np.full((nfrs, nu1*nu2, num_shfl_corr), np.nan)
#                    p_a22_shfl = np.full((nfrs, nu1*nu2, num_shfl_corr), np.nan)        

                if np.logical_and(v1+v2==2, nu1+nu2>2): # make sure both planes are valid (ie the traces are non-NaN), otherwise spearman wont work! Also make sure there is more than a single neuron pair, otherwise cs will have a non-square shape below!                
                    for ifr in range(nfrs): # ifr = 0 
#                        print('frame %d' %ifr)
                        # spearman r (does not assume normal distribution for inputs);  also provides the p value.
                        t1 = area2_d1[ifr, :, :] # units x trials
                        t2 = area2_d2[ifr, :, :] # units x trials
                        
#                        if np.logical_and(np.isnan(t1).all() + np.isnan(t2).all()!=True, nu1+nu2>2): # make sure the plane is valid (ie the traces are non-NaN), otherwise spearman wont work! 
                        cs, ps = st.spearmanr(t1.T, t2.T) # (nu1+nu2) x (nu1+nu2) # how the activity of neurons covaries across trials  (input: each column: a variable)
            
                        cs_across = cs[:nu1, nu1:] # nu1 x nu2   # area1_d1 - area1_d2 corr            
                        ps_across = ps[:nu1, nu1:] # nu1 x nu2   # area1_d1 - area1_d2 corr            
                        if d1==d2: # take only the upper triangle when d1 and d2 are the same (because it is the same pair of neurons; also the diagonal is corr of a neuron with itself, we dont want it!)
                            ii = np.tril_indices_from(cs_across)
                            cs_across[ii] = np.nan
                            ps_across[ii] = np.nan                        
#                        plt.imshow(cs_across) 
                            # Remove nans to save space!
#                            cnow = cs_across.flatten()
#                            cnow = cnow[~np.isnan(cnow)]
#                            pnow = ps_across.flatten()
#                            pnow = pnow[~np.isnan(pnow)]                            

                        c_a22[ifr,:] = cs_across.flatten() # 80 x (nu1 x nu2)
                        p_a22[ifr,:] = ps_across.flatten() # 80 x (nu1 x nu2)
                        
                        
                        ####################### Shuffled #######################
                        # shuffle trials, then compute corrcoefftween neurons... this serves as control to evaluate the values of corrcoeff of actual data (I'm worried they might be affected by how active neurons are... this will control for that)!
                        if corr_doShfl:
                            for ish in range(num_shfl_corr): # for each frame, we get corrcoeff between trial-shuffled traces. we do this num_shfl times. 
                                print('\tshuffle %d' %ish)
                                t1_shfl = t1[:, rnd.permutation(t1.shape[1])] # import numpy.random as rnd
                                t2_shfl = t2[:, rnd.permutation(t2.shape[1])]
    
        #                        if np.logical_and(np.isnan(t1).all() + np.isnan(t2).all()!=True, nu1+nu2>2): # make sure the plane is valid (ie the traces are non-NaN), otherwise spearman wont work! 
                                cs, ps = st.spearmanr(t1_shfl.T, t2_shfl.T) # (nu1+nu2) x (nu1+nu2) # how the activity of neurons covaries across trials  (input: each column: a variable)
                    
                                cs_across = cs[:nu1, nu1:] # nu1 x nu2   # area1_d1 - area1_d2 corr
#                                ps_across = ps[:nu1, nu1:] # nu1 x nu2   # area1_d1 - area1_d2 corr           
                                if d1==d2: # take only the upper triangle when d1 and d2 are the same (because it is the same pair of neurons; also the diagonal is corr of a neuron with itself, we dont want it!)
                                    ii = np.tril_indices_from(cs_across)
                                    cs_across[ii] = np.nan
#                                    ps_across[ii] = np.nan
        #                        plt.imshow(cs_across)
                                    # Remove nans to save space!
    #                                cnow = cs_across.flatten()
    #                                cnow = cnow[~np.isnan(cnow)]
    #                                pnow = ps_across.flatten()
    #                                pnow = pnow[~np.isnan(pnow)]
                                    
                                c_a22_shfl[ifr,:,ish] = cs_across.flatten() # 80 x (nu1 x nu2)
#                                p_a22_shfl[ifr,:,ish] = ps_across.flatten() # 80 x (nu1 x nu2)
                        
                c12_a22.append(c_a22) # 10 (layer pairs, in the order of cols); each element: 80 * (nu1*nu2) (nu1 and nu2 are the number of neurons in that layer pair)
                p12_a22.append(p_a22)
                if corr_doShfl:
                    c12_a22_shfl.append(c_a22_shfl) # 10 (layer pairs, in the order of cols); each element: 80 * (nu1*nu2) * num_shfl (nu1 and nu2 are the number of neurons in that layer pair)
#                    p12_a22_shfl.append(p_a22_shfl)
        
        
        
        ############ Compute average of shuffled cc across neuron pairs and shuffles
        # area1-area2                
        c12_a12_shfl_av = np.full((nfrs, len(cols)), np.nan) # 80x16
        for ilc in range(len(cols)): # loop through each layer pair
            c12_a12_shfl_av[:,ilc] = np.nanmean(c12_a12_shfl[ilc], axis=(1,2)) # nfrs # average across n pairs and shfles

        # area1-area1
        c12_a11_shfl_av = np.full((nfrs, len(colsas)), np.nan) # 80x10
        for ilc in range(len(colsas)): # loop through each layer pair
            c12_a11_shfl_av[:,ilc] = np.nanmean(c12_a11_shfl[ilc], axis=(1,2)) # nfrs # average across n pairs and shfles
        
        # area2-area2
        c12_a22_shfl_av = np.full((nfrs, len(colsas)), np.nan) # 80x10
        for ilc in range(len(colsas)): # loop through each layer pair
            c12_a22_shfl_av[:,ilc] = np.nanmean(c12_a22_shfl[ilc], axis=(1,2)) # nfrs # average across n pairs and shfles

        
        ############# Set p values: is the cc of each neuron pair significant (compared to its shuffle distribution) #############
        # area1-area2        
        p12_a12_shflm = []
        for ilc in range(len(cols)): # loop through each layer pair
            # is the cc of each neuron pair significant (compared to the shuffle dist)
            p = np.full(c12_a12[ilc].shape, np.nan)
            for inp in range(c12_a12[ilc].shape[1]): # loop over all n pairs
                [_, p[:,inp]] = st.ttest_1samp(c12_a12_shfl[ilc][:,inp], c12_a12[ilc][:,inp], axis=1) # nfrs # whether cc is sig (compared to shuffled dist of cc), for each frame
            p12_a12_shflm.append(p)
        
        # area1-area1
        p12_a11_shflm = []
        for ilc in range(len(colsas)): # loop through each layer pair
            # is the cc of each neuron pair significant (compared to the shuffle dist)
            p = np.full(c12_a11[ilc].shape, np.nan)
            for inp in range(c12_a11[ilc].shape[1]): # loop over all n pairs
                [_, p[:,inp]] = st.ttest_1samp(c12_a11_shfl[ilc][:,inp], c12_a11[ilc][:,inp], axis=1) # nfrs # whether cc is sig (compared to shuffled dist of cc), for each frame
            p12_a11_shflm.append(p)

        # area2-area2        
        p12_a22_shflm = []
        for ilc in range(len(colsas)): # loop through each layer pair
            # is the cc of each neuron pair significant (compared to the shuffle dist)
            p = np.full(c12_a22[ilc].shape, np.nan)
            for inp in range(c12_a22[ilc].shape[1]): # loop over all n pairs
                [_, p[:,inp]] = st.ttest_1samp(c12_a22_shfl[ilc][:,inp], c12_a22[ilc][:,inp], axis=1) # nfrs # whether cc is sig (compared to shuffled dist of cc), for each frame
            p12_a22_shflm.append(p)


        
        ###########
        this_sess_allExp.at[0, 'cc_cols_area12'] = cols
        this_sess_allExp.at[0, 'cc_cols_areaSame'] = colsas
        this_sess_allExp.at[0, 'cc_a12'] = c12_a12
        this_sess_allExp.at[0, 'cc_a11'] = c12_a11        
        this_sess_allExp.at[0, 'cc_a22'] = c12_a22        
        this_sess_allExp.at[0, 'p_a12'] = p12_a12
        this_sess_allExp.at[0, 'p_a11'] = p12_a11
        this_sess_allExp.at[0, 'p_a22'] = p12_a22

        this_sess_allExp.at[0, 'cc_a12_shfl'] = c12_a12_shfl_av # to save space we only save the averaged values! if space was not an issue, save the whole array: c12_a12_shfl
        this_sess_allExp.at[0, 'cc_a11_shfl'] = c12_a11_shfl_av        
        this_sess_allExp.at[0, 'cc_a22_shfl'] = c12_a22_shfl_av        
        this_sess_allExp.at[0, 'p_a12_shfl'] = p12_a12_shflm # it makes more sense to save the values computed manually above, than those coming out of spearman (for shuffled)... I also dont think we need to save the p values above... as i dont know what spearman p values are ... but it would be nice to compare the two.
        this_sess_allExp.at[0, 'p_a11_shfl'] = p12_a11_shflm
        this_sess_allExp.at[0, 'p_a22_shfl'] = p12_a22_shflm
        
    if doCorrs:
        this_sess = this_sess_allExp
        
        
        
        
        
    #%% Save the output of the analysis (for each session)
    #####################################################################################
    #####################################################################################
    #####################################################################################
    
    if doCorrs: # if doCorrs, we save values for each session (because we run it on the cluster), otherwise, we keep values of each session, append them, and them save "all_sess" var.
        
        save1file = 1 # for large files, set to 0, so different vars are saved in different h5 files.
        
        #%% Set pandas dataframe input_vars, to save it below
        
        ### set this so you know what input vars you ran the script with
        cols = np.array(['norm_to_max', 'mean_notPeak', 'peak_win', 'flash_win', 'trace_median', 'samps_bef', 'samps_aft', 'doScale', 'doShift', 'doShift_again', 'bl_gray_screen'])
        input_vars = pd.DataFrame([], columns=cols)
        
        input_vars.at[0, cols] = norm_to_max, mean_notPeak, peak_win, flash_win, trace_median, samps_bef, samps_aft, doScale, doShift, doShift_again, bl_gray_screen
        #input_vars.iloc[0]
        
        
        #%% Save this_sess
        # directory: /allen/programs/braintv/workgroups/nc-ophys/Farzaneh/omit_traces_peaks
        
        analysis_name = 'omit_traces_peaks'
        dir_now = os.path.join(dir_server_me, analysis_name)
        # create a folder: omit_traces_peaks
        if not os.path.exists(dir_now):
            os.makedirs(dir_now)
        
        #### Set the analysis file name
        if doCorrs==1:
            namespec = '_corr'
        else:
            namespec = ''
        
        if doShift_again==1:
            namespec = namespec + '_blShiftAgain'
        else:
            namespec = namespec + ''
            
        now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
        
        
        #cre_now = cre[:cre.find('-')]
        #name_this_sess = '%s_m-%d_s-%d' %(cre_now, mouse, session_id) # mouse, session: m, s
        name_this_sess = 'm-%d_s-%d' %(mouse, session_id) # mouse, session: m, s
        name = 'this_sess_%s%s_%s_%s' %(analysis_name, namespec, name_this_sess, now) 
        
            
        if saveResults:
            print('Saving .pkl file')
            if save1file:                
                '''
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
            
                # Save to a h5 file                    
                this_sess.to_hdf(allSessName, key='this_sess', mode='w')
                # save input_vars
                input_vars.to_hdf(allSessName, key='input_vars', mode='a')    
                '''

                # save to a pickle file
                allSessName = os.path.join(dir_now , name + '.pkl') 
                print(allSessName)
                
                f = open(allSessName, 'wb')
                pickle.dump(this_sess, f)    
                pickle.dump(input_vars, f)
                f.close()
    
            else: # with large dataset we need to do this! create a folder for each session and save multiple h5 files in it (each file contains some vars of this_sess dataframe)
                
                #### create a folder for each session ####
                named = 'this_sess_%s%s_s-%d' %(analysis_name, namespec, session_id)
                dir_now = os.path.join(dir_server_me, analysis_name, named)
                # create a folder: omit_traces_peaks
                if not os.path.exists(dir_now):
                    os.makedirs(dir_now)
                
                #### save vars in a h5 file ####                                                  
                nn = '_info'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'cc_cols_area12', 'cc_cols_areaSame']]
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    #            print(a.shape)
                input_vars.to_hdf(allSessName, key='input_vars', mode='a') # save input_vars to the h5 file            
    
                nn = '_c12'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['cc_a12']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
                nn = '_p12'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['p_a12']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
                nn = '_c11'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['cc_a11']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
                nn = '_p11'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['p_a11']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
                nn = '_c22'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['cc_a22']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
                nn = '_p22'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['p_a22']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
    
                ##### shfl vars #####
                
                nn = '_c12_shfl'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['cc_a12_shfl']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
                nn = '_p12_shfl'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['p_a12_shfl']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
                nn = '_c11_shfl'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['cc_a11_shfl']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
                nn = '_p11_shfl'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['p_a11_shfl']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
                nn = '_c22_shfl'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['cc_a22_shfl']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file
    
                nn = '_p22_shfl'
                name = 'this_sess_%s%s%s_%s_%s' %(analysis_name, namespec, nn, name_this_sess, now) 
                allSessName = os.path.join(dir_now , name + '.h5') 
                print(allSessName)
                a = this_sess.loc[:,['p_a22_shfl']]
    #            print(a.shape)
                a.to_hdf(allSessName, key='this_sess', mode='w') # save this_sess to the h5 file                
                
                
    #%% Return the output after we go through all experiments
    # ie after the following for loop is done: for index, lims_id in enumerate(data_list['lims_id'])
    # so we will have this_sess as long as a session has omissions, but even if some of its experiments are not valid.
    
#    if doCorrs==0:
    return this_sess
#    else:
#        return this_sess_allExp, cc_a1_a2, cc_a1_a1, cc_a2_a2, p_a1_a2, p_a1_a1, p_a2_a2
        
        
        

    #%% Concatanate values for all sessions, once done with all planes of a session
        
#    all_sess = all_sess.append(this_sess)
#    all_sess = all_sess.rename(columns={"mouse":"mouse_id"})
#    all_sess = all_sess.rename(columns={"peak_timing_eachN_trMed":"peak_timing_eachN_traceMed", "peak_amp_eachN_trMed":"peak_amp_eachN_traceMed"})




#%% 
#print(sess_no_omission)
# list_all_sessions = list_all_sessions[~np.in1d(list_all_sessions, sess_no_omission)]
'''
# plt.hist(all_sess.auc_peak_h1_h2.values) #, bins=60)
# all_sess.auc_peak_h1_h2
print(np.mean(all_sess.auc_peak_h1_h2.values))
# fraction of experiments where h1 > h2
print(np.mean(all_sess.auc_peak_h1_h2.values < .5))
# fraction of experiments where h1 < h2 
print(np.mean(all_sess.auc_peak_h1_h2.values > .5))
print(np.shape(all_sess.auc_peak_h1_h2))
'''


##%% Save all_sess
#
#svmn = 'omit_traces_peaks'
#now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
#name = 'all_sess_%s_%s' %(svmn, now) 
#    
#    
#if saveResults:
#    print('Saving .h5 file')
#    allSessName = os.path.join(dir_server_me , name + '.h5') # os.path.join(d, svmn+os.path.basename(pnevFileName))
#    print(allSessName)
#
#    # Save to a h5 file                    
#    all_sess.to_hdf(allSessName, key='all_sess', mode='w')
#    # save input_vars
#    input_vars.to_hdf(allSessName, key='input_vars', mode='a')    
#    
#
##%%
##all_sess = pd.read_hdf('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/all_sess_omit_traces_peaks_20190826_130759.h5', key='all_sess')
##input_vars = pd.read_hdf('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/all_sess_omit_traces_peaks_20190826_130759.h5', key='input_vars')
#    
#    











#%% Set vars to submit call the function above (when submitting jobs to the cluster)
###########################################################################################        
###########################################################################################
########################################################################################### 
###########################################################################################
###########################################################################################

#%%
use_ct_traces = 1 # if 0, we go with dff traces saved in analysis_dir (visual behavior production analysis); if 1, we go with crosstalk corrected dff traces on rd-storage

saveResults = 1 # save the all_sess pandas at the end

doCorrs = 1 # if 0, compute omit-aligned trace median, peaks, etc. If 1, compute corr coeff between neuron pairs in each layer of v1 and lm 
doShift_again = 0 #0 # this is a second shift just to make sure the pre-omit activity has baseline at 0. (it is after we normalize the traces by baseline ave and sd... but because the traces are median of trials (or computed from the initial gray screen activity), and the baseline is mean of trials, the traces wont end up at baseline of 0)                                      
bl_gray_screen = 1 #1 # if 1, baseline will be computed on the initial gray screen at the beginning of the session
# if 0, get the lowest 20 percentile values of the median trace preceding omission and then compute their mean and sd
# this value will only affect the traces, but not the peak quantification, because for peak quantification we subtract the local baseline
# which is computed as 10th percentile of activity preceding each omission
    
# THE following is not a problem anymore! as peak timing is not being computed on SST/SLC!
# NOTE: not sure how much to trust peak measures (amplitude and timing) for non-VIP lines... 
# they dont really have a peak, and you have the note below in the omissions_traces_peaks.py code:
# if the mouse is NOT a VIP line, instead of computing the peak, we compute the trough: we invert the trace, and then compute the max. (this is because in VIP mice the response after omission goes up, but not in SST or Slc)
mean_notPeak = 1 # set to 1, so mean is computed; it is preferred because SST and SLC responses don't increase after omission 
# if 1, to quantify omission-evoked response, compute average trace during peak_win, instead of the peak amplitude.
# if 1, we will still use the timing of the "peak" for peak timing measures ... 
# Note sure of the following: if mean_notPeak is 1, it makes sense to go with a conservative (short) peak_win so the results are not contaminated by the responses to the flash that follows the omission    
# set mean_notPeak to 0 if you want to measure trough (on traces with decreased response after the omission)

peak_win = [0, .75] # you should totally not go beyond 0.75!!! # go with 0.75ms if you don't want the responses to be contaminated by the next flash
# [0,2] # flashes are 250ms and gray screen is 500ms # sec # time window to find peak activity, seconds relative to omission onset
# [-.75, 0] is too large for flash responses... (unlike omit responses, they peak fast!
flash_win = [-.75, -.25] #[-.75, -.4] #[-.75, 0] # window (relative to omission) for computing flash-evoked responses (750ms includes flash and gray)    

norm_to_max = 0 # normalize each neuron trace by its max
### In this case doScale is automatically set to zero, because:
# remember whether you are normalizing to max peak activity or not will not matter if you normalize by baseline SD. because if you normalize to max peak activity, baseline SD will also get changed by that same amount, so after normalizing to baseseline SD the effect of normalizing to max peak activity will be gone.

trace_median = 1 # if 1, use peak measures computed on the trial-median traces (less noisy); if 0, use peak measures computed on individual trials.
    
doScale = 1 # Scale the traces by baseline std
doShift = 1 # Shift the y values so baseline mean is at 0 at time 0    

doPlots = 0 # some plots (not the main summary ones at the end)... set to 0
doROC = 0 # if 1, do ROC analysis: for each experiment, find, across all neurons, how separable the distributions of trial-averaged peak amplitudes are for half-1 omissions vs. half-2 omissions
#dosavefig = 1

# To align on omissions, get 40 frames before omission and 39 frames after omission
samps_bef = 40 
samps_aft = 40


if norm_to_max:
    doScale = 0

if doROC:
    from sklearn.metrics import roc_curve, auc



#%% Load vars related to list_all_sessions_valid 
# (read the pickle file saved in the script: set_valid_sessions.py)

import pickle

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

# validSessName = os.path.join(dir_valid_sess, 'valid_sessions' + '.pkl')

dir_valid_sess = os.path.join(dir_server_me, 'ValidSessions')

regex = re.compile('valid_sessions_(.*)' + '.pkl')
l = os.listdir(dir_valid_sess)
files = [string for string in l if re.match(regex, string)]

# NOTE: # get the latest file (make sure this is correct)
files = files[-1]
validSessName = os.path.join(dir_valid_sess, files)

print(validSessName)
    
    
pkl = open(validSessName, 'rb')
dictNow = pickle.load(pkl)

for k in list(dictNow.keys()):
    exec(k + '= dictNow[k]')

#['list_all_sessions0',
# 'list_sessions_date0',
# 'list_sessions_experiments0',
# 'validity_log_all',
# 'list_all_sessions_valid',
# 'list_all_experiments_valid',
# 'list_all_experiments']  
    

#%% crosstalk-corrected sessions
if use_ct_traces:
    '''
    # Later when we call load_session_data_new, it will load the ct traces if use_ct_traces is 1; otherwise it will seet dataset which loads the original dff traces using vb codes.

    # Get a list of sessions ready for post-cross talk analysis with the following code:
    import visual_behavior.ophys.mesoscope.utils as mu
    import logging
    lims_done, lims_notdone, meso_data = mu.get_lims_done_sessions()
    lims_done = lims_done.session_id.drop_duplicates()
    lims_done.values
    '''

    sessions_ctDone = np.array([839208243, 839514418, 840490733, 841303580, 841682738, 841778484,
                               842023261, 842364341, 842623907, 843059122, 843871999, 844469521,
                               845444695, 846652517, 846871218, 847758278, 848401585, 848983781,
                               849304162, 850667270, 850894918, 851428829, 851740017, 852070825,
                               852794141, 853177377, 853416532, 854060305, 856201876, 857040020,
                               863815473, 864458864, 865024413, 865854762, 866197765, 867027875,
                               868688430, 869117575, 870352564, 870762788, 871526950, 871906231,
                               872592724, 873720614, 874070091, 874616920, 875259383, 876303107,
                               880498009, 882674040, 884451806, 885303356, 886806800, 888009781,
                               889944877, 906299056, 906521029])

    print(sessions_ctDone.shape)

    # Copy ct folders to allen server (nc-ophys/Farzaneh):
    # ssh to the analysis computer by running: ssh -L 8888:ibs-is-analysis-ux1.corp.alleninstitute.org:8889 farzaneh.najafi@ibs-is-analysis-ux1.corp.alleninstitute.org
    # then on the analysis computer, in your folder (Farzaneh), run the notebook below
    # copy_crosstalkFolders_analysisPC_to_allen.py

    # or, on the analysis computer, in the terminal do:
    # cp -rv /media/rd-storage/Z/MesoscopeAnalysis/session_839208243 /allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk;

    # now take those valid sessions that are ct-corrected
    list_all_sessions_valid_ct = list_all_sessions_valid[np.in1d(
        list_all_sessions_valid, sessions_ctDone)]
    list_all_experiments_valid_ct = np.array(list_all_experiments_valid)[
        np.in1d(list_all_sessions_valid, sessions_ctDone)]
    list_all_experiments_ct = np.array(list_all_experiments)[
        np.in1d(list_all_sessions_valid, sessions_ctDone)]

    print(list_all_sessions_valid_ct.shape)
    list_all_experiments_valid_ct.shape
    list_all_experiments_ct.shape

    # redefine these vars
    list_all_sessions_valid = list_all_sessions_valid_ct
    list_all_experiments_valid = list_all_experiments_valid_ct
    list_all_experiments = list_all_experiments_ct


    
#%%
#from omissions_traces_peaks_pbs import *
    
    
#%%
#isess = 0
isess = int(sys.argv[1])
#print(isess)
#print(type(isess))


#%% Run the main function
    
session_id = int(list_all_sessions_valid[isess])
#    experiment_ids = list_all_experiments_valid[isess]
experiment_ids = list_all_experiments[isess] # we want to have the list of all experiments for each session regardless of whethere they were valid or not... this way we can find the same plane across all sessions.

#cnt_sess = cnt_sess + 1
print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess+1, len(list_all_sessions_valid)))
#print('%d: session %d out of %d sessions' %(session_id, cnt_sess+1, len(list_all_sessions_valid)))


# call the main function
this_sess = omissions_traces_peaks_pbs(session_id, experiment_ids, validity_log_all, norm_to_max, mean_notPeak, peak_win, flash_win, trace_median, doScale, doShift, doShift_again, bl_gray_screen, samps_bef, samps_aft, doCorrs, saveResults, doPlots, doROC)


    



