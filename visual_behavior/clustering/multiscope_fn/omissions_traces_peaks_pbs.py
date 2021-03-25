#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Called by the script omissions_traces_peaks_init.py.
The function here creates a pandas table (all_sess) that
includes vars related to median traces across trials, aligned on omission, and their peak amplitude.

To make plots, after this function is run, run the script: omissions_traces_peaks_plots_setVars.py

# Note: if you are going to follow this script by "omissions_traces_peaks_plots_setVars.py", 
# in addition to saving updated all_sess to file, you also need to save to file the updated "mouse_trainHist_all2". 
# You do this by: 1) updating the list of all_mice_id in set_mouse_trainHist_all2.py, and 
# 2) running set_mouse_trainHist_init_pbs.py on the cluster (which calls set_mouse_trainHist_all2.py).

Created on Wed May  8 15:25:18 2019
@author: Farzaneh
"""

# %load_ext autoreload
# %autoreload 2


from def_funs import *
from def_funs_general import *
from omissions_traces_peaks_quantify import *
from omissions_traces_peaks_quantify_flashAl import *
                
#%%
def omissions_traces_peaks(metadata_all, session_id, experiment_ids, experiment_ids_valid, validity_log_all, norm_to_max, mean_notPeak, peak_win, flash_win, flash_win_timing, flash_win_vip, bl_percentile, num_shfl_corr, trace_median, doScale, doShift, doShift_again, bl_gray_screen, samps_bef, samps_aft, doCorrs, subtractSigCorrs, saveResults, cols, cols_basic, colsa, cols_this_sess_l, use_ct_traces=1, use_np_corr=1, use_common_vb_roi=1, controlSingleBeam_oddFrPlanes=[0], doPlots=0, doROC=0):    
    
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

    #%% Set initial vars
        
    if num_shfl_corr==0:
        corr_doShfl =  0 #1 # if 1, compute shuffled corrcoeff too.
    else:
        corr_doShfl = 1

    doPeakPerTrial = 1 # if 1, find amplitude of peak response on each omission trial (noisier compared to peak_amp_eachN_traceMed which is computed on trial-median trace)
    
    # define columns for dataframes
    colsn = np.concatenate((cols_basic, colsa)) # colsn = cols
    if doCorrs==1:
        cols = cols_basic

    # Use a different window to compute flash timing: the idea is to be able to capture the timing
    # also for VIP A sessions, in which the response rises before the flash and peaks right at flash time.
    # Also to get an accurate response magnitude measure for VIP A sessions, try: flash_win = [-1, -.65] 
#     flash_win_timing = [-.85, -.5]
#     bl_percentile = 10 #20  # for peak measurements, we subtract pre-omit baseline from the peak. bl_percentile determines what pre-omit values will be used.      
#     num_shfl_corr = 50 # shuffle trials, then compute corrcoeff... this serves as control to evaluate the values of corrcoeff of actual data    
    
#    if norm_to_max:
#        doScale = 0


    ################################################################################
    ######################################################################
    ######################################################################
    #%% Load the traces and metadata for the session
    ######################################################################
    ######################################################################
    ################################################################################

    # load the traces, behavioral and metadata for the session
    [whole_data, data_list, table_stim, behav_data] = load_session_data_new(metadata_all, session_id, experiment_ids, use_ct_traces, use_np_corr, use_common_vb_roi)    

#    [whole_data, data_list, table_stim] = load_session_data(session_id) # data_list is similar to whole_data but sorted by area and depth



    if any(np.isnan(data_list['depth'].values.astype(float))):
        session_exclude = 1
        sys.exit(f'\n\nSession {session_id} cannot be analyzed. At least one depth is NaN, because "dataset" could not be set, likely due to segmentation problems and the absence of some related files!')

        
    else:
        #####################################################################################
        #%% Set metadata: mouse, cre, stage, etc.
        #####################################################################################
        
        exp_ids = list(whole_data.keys())
        mouse = whole_data[exp_ids[0]]['mouse']
        date = whole_data[exp_ids[0]]['experiment_date']    
        cre = whole_data[exp_ids[0]]['cre'] # it will be the same for all lims_ids (all planes in a session)
        stage = whole_data[exp_ids[0]]['stage']

        print(f'\n {cre} , {stage}\n\n')
    #     this_sess = mouse
    #     return this_sess

        #%% Set flash_win: use a different window for computing flash responses if it is VIP, and B1 session (1st novel session)

        session_novel = is_session_novel(dir_server_me, mouse, date) # first determine if the session is the 1st novel session or not

        # vip, except for B1, needs a different timewindow for computing image-triggered average (because its response precedes the image).
        # sst, slc, and vip B1 sessions all need a timewindow that immediately follows the images.
        if np.logical_and(cre.find('Vip')==0 , session_novel) or cre.find('Vip')==-1: # VIP B1, SST, SLC: window after images (response follows the image)
            flash_win_final = flash_win # [0, .75]

        else: # VIP (non B1,familiar sessions): window precedes the image (pre-stimulus ramping activity)
            flash_win_final = flash_win_vip # [-.25, .5] 
        

        #####################################################################################
        #%% Set stimulus and behavioral variables
        #####################################################################################

        #%% Set omissions and flashes
        list_omitted = table_stim[table_stim['omitted']==True]['start_time'] # start time of omissions (in sec)
        list_omitted0 = list_omitted + 0

        if len(list_omitted)==0:
            print('session %d has no omissions!' %session_id)
    ##        sys.exit('Exiting analysis as there are no omission trials in session %d!' %session_id)
    ##            sess_no_omission.append(session_id)
    #        
    #    else:

        #%%
        # start time of all flashes (in sec)
        list_flashesOmissions = table_stim['start_time'].values #[table_stim['omitted']==False]['start_time']
        list_flashes = table_stim[table_stim['omitted']==False]['start_time']


        # compute flash and gray durations from table_stim
        '''
        flashdurs = table_stim['end_time'].values - table_stim['start_time'].values
        flash_dur_med = np.median(flashdurs)

        # use both flash and omissions to set flashgraydurs
        flashgraydurs, i = np.unique(np.diff(list_flashesOmissions), return_inverse=True)

        # use only flashes to set flashgraydurs
        flashgraydurs, i = np.unique(np.diff(list_flashes), return_inverse=True)

        flashgray_dur_med = np.median(flashgraydurs)    
        gray_dur_med = flashgray_dur_med - flash_dur_med

        print(flash_dur_med, gray_dur_med, flashgray_dur_med)
        '''

        #%% Check for repeated omission entries (based on start times) or consecutive omissions
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
        if len(consec_om)>0:
            print(f'\nThere are {len(consec_om)} consecutive omissions in session {session_id}; removing them...')

        # remove the values set to nan
        #list_omitted.drop(list_omitted.index[consec_om, rep_om])
        list_omitted = list_omitted[~np.isnan(list_omitted.values)]                
        #        len(list_omitted0), len(list_omitted)
        #        list_omitted0, ist_omitted


        
        #%% Set dataframe this_sess_l        
        
        this_sess_l = pd.DataFrame([], columns = cols_this_sess_l) # colsa # index = range(num_planes), 
#         this_sess_l = pd.DataFrame([], columns = ['valid', 'local_fluo_allOmitt', 'list_flashes', 'list_omitted', 'running_speed', 'licks', 'rewards'])
#         this_sess_l = pd.DataFrame([], columns = ['valid', 'local_fluo_allOmitt', 'local_fluo_flashBefOmitt', 'local_fluo_traces', 'local_time_traces', 'list_flashes', 'list_omitted', 'running_speed', 'licks', 'rewards'])

        # the following are at the session level, even though we are saving them at the experiment level.
        for ind in range(num_planes): # ind=0
            this_sess_l.at[ind, 'list_flashes'] = [list_flashes] # flash onset
            this_sess_l.at[ind, 'list_omitted'] = [list_omitted] # omission onset
            this_sess_l.at[ind, 'running_speed'] = behav_data['running_speed']
            this_sess_l.at[ind, 'licks'] = behav_data['licks'] 
            this_sess_l.at[ind, 'rewards'] = behav_data['rewards']                        
        #'running_speed', 'licks', 'rewards', 'd_prime', 'hit_rate', 'catch_rate'


        #%% Set vars (time window and image names/indeces) for removing signal so correlations are not affected by signal, and instead reflect noise corrletions.
        # for each image type, we will compute average response across all trials of that image, and will subtract this average response from individual trials. This will be done for each neuron separately.
        '''
        if np.logical_and(doCorrs==1, subtractSigCorrs==1):

            # flash_win_images = np.array(flash_win_final) + .75 # vip,B1: # [-.25, .5] ; rest: [0, .75]
            flash_win_frames = np.floor(flash_win_final / frame_dur).astype(int)

            samps_bef_images = flash_win_frames[0]
            samps_aft_images = flash_win_frames[-1]-1 # assuming that np.arage for list_times definition goes up to peak_win_frames[-1] (not peak_win_frames[-1]+1); read your note for function "set_frame_window_flash_omit" in def_funs, copied here: # note: list_times goes upto peak_win_frames[-1]+1 but we dont add +1 for flash (see below)... omissions peak late, so we do it; but perhaps it makes sense to be consistent and just remove +1 from omissions too!        # you removed it on 04/20/2020

            # Note: for session_id: 852794141, np.unique(np.diff(list_flashesOmissions)) gives values ranging from 0.71725 to 0.78398. It is weird as we expect flashes to occur every 0.75 sec.
            img_arr = table_stim['image_name'].values
            img_names, iimg = np.unique(img_arr, return_inverse=True)    
        '''


        
        #%% Loop through the 8 planes of each session

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

        ################################################################################
        ######################################################################
        ######################################################################
        #%% Go through each experiment and if it is valid, we do a number of analyses.
        ######################################################################
        ######################################################################
        ################################################################################

        # initiate the pandas dataframe this_sess
        this_sess = pd.DataFrame([], columns = cols) # index = range(num_planes), 
                
        for index, lims_id in enumerate(data_list['lims_id']): 
            
            print(f'\n\n=========== Analyzing experiment_id: {lims_id} ===========\n\n')
            
            '''
            for il in [0]: #range(num_planes):
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
            
            this_sess.at[index, ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth']] = session_id, lims_id, mouse, date, cre, stage, area, depth
            this_sess_l.at[index, 'valid'] = 1 # we will reset it to 0 if the experiment has been invalid (based on the code set_valid_sessions) or there are no neurons for this plane.

#             return this_sess

            #%% Get traces of all neurons for the entire session

            # samps_bef (=40) frames before omission ; index: 0:39
            # omission frame ; index: 40
            # samps_aft - 1 (=39) frames after omission ; index: 41:79

            # Do the analysis only if the experiment is valid
            ### NOTE: commenting below and instead using the list of experiments in March 2021 data release to identify valid experiments
#             if validity_log_all.iloc[validity_log_all.lims_id.values == int(lims_id)]['valid'].bool() == False:
                
            if sum(np.in1d(experiment_ids_valid, int(lims_id)))==0: # make sure lims_id is among the experiments in the data release
                
                num_frs_control = samps_bef + samps_aft
                if controlSingleBeam_oddFrPlanes[0]==1:
                    num_frs_control = int(num_frs_control/2)
                    
                # num_neurons = 1
                local_fluo_allOmitt = np.full((num_frs_control, 1, len(list_omitted)), np.nan) # just a nan array as if there was 1 neuron, so this experiment index is not empty in this_sess_l.
                this_sess_l.at[index, 'local_fluo_allOmitt'] = local_fluo_allOmitt
                this_sess_l.at[index, 'valid'] = 0
                print('Skipping invalid experiment %d, index %d' %(int(lims_id), index))
        #                this_sess.at[index, :] = np.nan # check this works.

            else:

                # Get traces of all neurons for the entire session
                local_fluo_traces = whole_data[lims_id]['fluo_traces'] # neurons x frames
                # added below after using data release sessions; the traces that we get from allensdk dataset object are 1 dimensional, so we make them 2 dim
                if np.ndim(local_fluo_traces)==1 and local_fluo_traces.shape[0]>0:
                    local_fluo_traces = np.vstack(whole_data[lims_id]['fluo_traces'])
                    
                local_time_traces = whole_data[lims_id]['time_trace']  # frame times in sec. Volume rate is 10 Hz. Are these the time of frame onsets?? (I think yes... double checking with Jerome/ Marina.) # dataset.timestamps['ophys_frames'][0]             
                roi_ids = whole_data[lims_id]['roi_ids']
                
                
                this_sess_l.at[index, 'local_fluo_traces'] = local_fluo_traces
                this_sess_l.at[index, 'local_time_traces'] = local_time_traces
                this_sess_l.at[index, 'roi_ids'] = roi_ids
                
                
                frame_dur = np.mean(np.diff(local_time_traces)) # difference in sec between frames
                print(f'Frame duration {frame_dur:.3f} ms')
                if np.logical_or(frame_dur < .09, frame_dur > .1):
                    print(f'\n\nWARNING:Frame duration is unexpected!! {frame_dur}ms\n\n')
        #                local_time_traces_all.append(local_time_traces)
        #            depth_all.append(depth)
        #            area_all.append(area)

                num_neurons = local_fluo_traces.shape[0]

                if num_neurons==0: # some of the de-crosstalked planes don't have any neurons.
                    this_sess_l.at[index, 'valid'] = 0
                    print('0 neurons! skipping invalid experiment %d, index %d' %(int(lims_id), index))

                elif len(list_omitted)==0:
                    num_omissions = 0
                    this_sess.at[index, ['n_omissions', 'n_neurons', 'frame_dur']] = num_omissions, num_neurons, frame_dur
                             
                    
                    
                    
            if doCorrs==-1: # in case an experiment is invalid or there are no omissions we make sure we set this_sess
                this_sess = this_sess.iloc[:,range(len(cols_basic))].join(this_sess_l) # len(cols_basic) = 13

            
            ################################################################################
            ################################################################################
            # If the session is valid, start the analysis:
            ################################################################################
            ################################################################################
            
            if this_sess_l.at[index, 'valid'] == 1 and len(list_omitted)>0:

                #%% Plot the trace of all neurons (for the entire session) and mark the flash onsets
                '''    
                get_ipython().magic(u'matplotlib inline')
                plt.subplot(8,1,index+1)                
                # y = local_fluo_traces.T # individual neurons
                y = np.nanmean(local_fluo_traces, axis=0) # average of neurons
                plt.plot(local_time_traces, y)  
                mn = 0; mx = np.max(y)-3*np.std(y)
                plt.vlines(list_flashesOmissions, mn, mx)
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


                #%% Remove signal from local_fluo_traces so when we later compute correlations, they are not affected by signal, and instead reflect noise corrletions.
                # for each image type, we will compute average response across all trials of that image, and will subtract this average response from individual trials. This will be done for each neuron separately.
                # Note: you originally coded this; the idea was to remove the signal from the entire session trace, and then do omition alignment; the problem is that it is hard to find the time window for computing signal given the continuous stream of flashes, also the difference in timing of image vs omission responses, and vip B1 vs other sessions.
                '''
                if np.logical_and(doCorrs==1, subtractSigCorrs==1):

                    local_fluo_traces0 = local_fluo_traces + 0 # save a copy before subtracting out the average responses.

                    # get the timestamps of all images that belong to a given image name 
                    for img_ind in range(len(img_names)):
                        times_this_image = list_flashesOmissions[iimg==img_ind] # start time (sec) of flashes that are of image type img_names[img_ind]  

                        # align traces on each image (so you can get averages of each image type)
                        [local_fluo_this_image, local_time_this_image] = align_trace_on_event(local_fluo_traces, local_time_traces, samps_bef_images, samps_aft_images, times_this_image)

                        # average across all images of type img_ind  # local_fluo_this_image: frames x units x trials
                        fluo_ave_this_image = np.nanmean(local_fluo_this_image, axis=-1) # frames x neurons

                    #     time_ave_this_image = np.nanmean(local_time_this_image, axis=-1) # frames
                    #     plt.plot(time_ave_this_image, np.mean(fluo_ave_this_image, axis=1))

                        # subtract fluo_ave_this_image from all flashes that had that image
                        for iflash in range(len(times_this_image)): # # indiv_time in list_omitted: # 

                            # find the image onsets in frame units (times_this_image on local_fluo_traces)
                            indiv_time = times_this_image[iflash] 
                            local_index = np.argmin(np.abs(local_time_traces - indiv_time)) # the index of omission on local_time_traces               

                            be = local_index - samps_bef_images
                            af = local_index + samps_aft_images

                            # get a chuck of trace after this image, then subtract from this chunck, the average response: fluo_ave_this_image
                            local_fluo_traces[:, be:af] = local_fluo_traces[:, be:af] - fluo_ave_this_image
                '''


                ################################################################################
                ################################################################################
                ################################################################################
                #%% Align on omission trials:
                # samps_bef (=40) frames before omission ; index: 0:39
                # omission frame ; index: 40
                # samps_aft - 1 (=39) frames after omission ; index: 41:79
                ################################################################################
                ################################################################################
                ################################################################################
                
                # Keep a matrix of omission-aligned traces for all neurons and all omission trials    
                local_fluo_allOmitt = np.full((samps_bef + samps_aft, num_neurons, len(list_omitted)), np.nan) # time x neurons x omissions_trials 
                local_time_allOmitt = np.full((samps_bef + samps_aft, len(list_omitted)), np.nan) # time x omissions_trials
                # flash-aligned traces (using the flash before omission)
                local_fluo_flashBefOmitt = np.full((samps_bef + samps_aft, num_neurons, len(list_omitted)), np.nan) # time x neurons x omissions_trials 
                local_time_flashBefOmitt = np.full((samps_bef + samps_aft, len(list_omitted)), np.nan) # time x omissions_trials

                # Loop over omitted trials to align traces on omissions
                flashes_win_trace_all = []
                flashes_win_trace_index_all = []
                image_names_surr_omit_all = []
                flash_omit_dur_all = []
                flash_omit_dur_fr_all = []
                num_omissions = 0 # trial number

                for iomit in range(len(list_omitted)): # indiv_time in list_omitted: # iomit=0

                    indiv_time = list_omitted.iloc[iomit]

                    local_index = np.argmin(np.abs(local_time_traces - indiv_time)) # the index of omission on local_time_traces               

                    be = local_index - samps_bef
                    af = local_index + samps_aft

                    if ~np.logical_and(be >= 0 , af <= local_fluo_traces.shape[1]): # make sure the omission is at least samps_bef frames after trace beigining and samps_aft before trace end. 
                        print('Omission %d at time %f cannot be analyzed: %d timepoints before it and %d timepoints after it!' %(iomit, indiv_time, be, af)) 

                    try:
                        ######### Align on omission
                        local_fluo_allOmitt[:,:, iomit] = local_fluo_traces[:, be:af].T # frame x neurons x omissions_trials (10Hz)
                        # local_time_allOmitt is frame onset relative to omission onset. (assuming that the times in local_time_traces are frame onsets.... still checking on this, but it seems to be the rising edge, hence frame onset time!)
                        # so local_time_allOmitt will be positive if frame onset is after omission onset.

                        # local_time_allOmitt shows the time of the frame that is closest to omission relative to omission.
                        # (note, the closest frame to omission is called omission frame).
                        # eg if local_time_allOmitt is -0.04sec, it means, the omission frame starts -.04sec before omission.
                        # or if local_time_allOmitt is +0.04sec, it means, the omission frame starts +.04sec before omission.

                        # Note: the following two quantities are very similar (though time_orig is closer to the truth!): 
                        # time_orig  and time_trace (defined below)
                        # time_orig = np.mean(local_time_allOmitt, axis=1)                            
                        # time_trace, time_trace_new = upsample_time_imaging(samps_bef, samps_aft, 31.) # set time for the interpolated traces (every 3ms time points)

                        # so we dont save local_time_allOmitt, although it will be more accurate to use it than time_trace, but the
                        # difference is very minimum
                        local_time_allOmitt[:, iomit] = local_time_traces[be:af] - indiv_time # frame x omissions_trials



                        ######### Identify the flashes (time and index on local_time_allOmitt) that happened within local_time_allOmitt                    
                        flashes_relative_timing = list_flashesOmissions - indiv_time # timing of all flashes in the session relative to the current omission
                        # the time of flashes that happened within the omit-aligned trace
                        a = np.logical_and(flashes_relative_timing >= local_time_allOmitt[0, iomit], flashes_relative_timing <= local_time_allOmitt[-1, iomit]) # which flashes are within local_time
                        flashes_win_trace = flashes_relative_timing[a] 
                        # now find the index of flashes_win_trace on local_time_allOmitt
                        a2 = [np.argmin(np.abs(local_time_allOmitt[:, iomit] - flashes_win_trace[i])) for i in range(len(flashes_win_trace))]
                        flashes_win_trace_index = a2 # the index of omission on local_time_traces

                        # image names for 1 flash before and 2 flashes after omission 
                        image_names_surr_omit0 = table_stim[a]['image_name'].values[np.argwhere(flashes_win_trace==0).squeeze()-1: np.argwhere(flashes_win_trace==0).squeeze()+3]
                        image_names_surr_omit = image_names_surr_omit0[image_names_surr_omit0!='omitted'] # remove 'omitted'
                        # 'im', 'omitted', 'im', 'im' : normal sequence; but somettimes: there are two alternating omissions: 'im', 'omitted', 'im', 'omitted'
                        
    #                     if len(image_names_surr_omit) < 3:
    #                         print(f'There are two alternating omissions: {image_names_surr_omit0}')

                        # set to nan the omission trace if it was not preceded by another image in the repetitive structure that we expect to see.
                        if len(image_names_surr_omit) == 0: # session_id: 839514418; there is a last omission after 5.23sec of previous image
                            print(f'Omission {iomit} is uncanny! no images around it! so removing it!')
                            local_fluo_allOmitt[:,:, iomit] = np.nan
                            
                        elif len(image_names_surr_omit) == 1: # session_id: 47758278; there is a last omission after 1.48sec of previous image and no images after that!
                            print(f'Omission {iomit} is uncanny! no images after it! so removing it!')
                            local_fluo_allOmitt[:,:, iomit] = np.nan
                          
                        elif len(image_names_surr_omit)>0 and len(np.unique(image_names_surr_omit[[0,1]]))>1: # the 2nd image after omission could be a different type, but the first after omission should be the same as the one before omission.
                            print('image after omission is different from image before omission! uncanny!') # sys.exit
                            local_fluo_allOmitt[:,:, iomit] = np.nan
                        else:
                            
                            '''
                            plt.plot(local_time_allOmitt[:,iomit], local_fluo_allOmitt[:,:,iomit])
                            plt.vlines(flashes_win_trace, -.4, .4)

                            plt.figure()
                            plt.plot(local_fluo_allOmitt[:,:,iomit])
                            plt.vlines(flashes_win_trace_index, -.4, .4)
                            '''

                            #################################################################################
                            ######### Align traces on the flash preceding the omission (I used to just take .75sec before omission, but due to frame-drop/unknown issues flashes dont exactly happen every 750ms (revealed by np.unique(np.diff(list_flashesOmissions)))) #########
                            #################################################################################
                            indiv_timef = list_flashesOmissions[np.argwhere(flashes_relative_timing==0).squeeze()-1]
                            local_indexf = np.argmin(np.abs(local_time_traces - indiv_timef)) # the index of omission on local_time_traces               

                            bef = local_indexf - samps_bef
                            aff = local_indexf + samps_aft

                            # align on the flash preceding the omission
                            local_fluo_flashBefOmitt[:,:, iomit] = local_fluo_traces[:, bef:aff].T # frame x neurons x omissions_trials (10Hz)
                            local_time_flashBefOmitt[:, iomit] = local_time_traces[bef:aff] - indiv_timef # frame x omissions_trials

                            '''
                            plt.plot(np.nanmean(local_time_allOmitt, axis=1), np.nanmean(local_fluo_allOmitt, axis=(1,2)))
                            plt.vlines(flashes_win_trace, -.4, .4)

                            plt.plot(np.nanmean(local_time_flashBefOmitt, axis=1), np.nanmean(local_fluo_flashBefOmitt, axis=(1,2)))
                            plt.vlines(flashes_win_trace, -.4, .4)
                            '''

                            ######### 
    #                         num_omissions = num_omissions + 1
                            flash_omit_dur = indiv_time - indiv_timef # sec # this is to check how much the values deviate from the expected .75sec
                            flash_omit_dur_fr = local_index - local_indexf # in frames; same as above

                            flashes_win_trace_all.append(flashes_win_trace)
                            flashes_win_trace_index_all.append(flashes_win_trace_index)
                            image_names_surr_omit_all.append(image_names_surr_omit)
                            flash_omit_dur_all.append(flash_omit_dur)
                            flash_omit_dur_fr_all.append(flash_omit_dur_fr)

        #                    time 0 (omissions) values:
        #                    local_fluo_allOmitt[samps_bef,:,:]
        #                    local_time_allOmitt[samps_bef,:]


                    except Exception as e:
                        print(f'Omission alignment failed; omission index: {iomit}, omission frame: {local_index}, starting and ending frames: {be, af}') # indiv_time, 
                        print(e)


                flash_omit_dur_all = np.array(flash_omit_dur_all)
                flash_omit_dur_fr_all = np.array(flash_omit_dur_fr_all)

                # plot duration between the last flash before omission and omission. we want to see how much it varies across trials. to see how problematic it is to use omit-aligned traces for computing flash responses.
                '''
                get_ipython().magic(u'matplotlib inline')
                plt.plot(np.sort(flash_omit_dur_fr_all), label='frames'); plt.plot(10*np.sort(flash_omit_dur_all), label='duration*10sec'); plt.legend(); plt.ylabel('flash-omission gap, sorted'); plt.xlabel('omission')
                print(np.mean(np.array(flash_omit_dur_fr_all)==7)) # the majority are 8 frames apart.
                '''

                # remove the last nan rows from the traces, which happen if some of the omissions are not used for alignment (due to being too early or too late in the session) 
                # change this to removing nan rows and define num_omiss
                # find nan rows
                mask_valid_trs = np.nansum(local_fluo_allOmitt, axis=(0,1))!=0
                if sum(mask_valid_trs==False) > 0:
                    local_fluo_allOmitt = local_fluo_allOmitt[:,:,mask_valid_trs]
                    local_time_allOmitt = local_time_allOmitt[:,mask_valid_trs]
                    local_fluo_flashBefOmitt = local_fluo_flashBefOmitt[:,:,mask_valid_trs]
                    local_time_flashBefOmitt = local_time_flashBefOmitt[:,mask_valid_trs]
                 
                num_omissions = sum(mask_valid_trs)
                
#                 if len(list_omitted)-num_omissions > 0:
#                     local_fluo_allOmitt = local_fluo_allOmitt[:,:,0:-(len(list_omitted)-num_omissions)]
#                     local_time_allOmitt = local_time_allOmitt[:,0:-(len(list_omitted)-num_omissions)]
#                     local_fluo_flashBefOmitt = local_fluo_flashBefOmitt[:,:,0:-(len(list_omitted)-num_omissions)]
#                     local_time_flashBefOmitt = local_time_flashBefOmitt[:,0:-(len(list_omitted)-num_omissions)]

                ##### Note: you need to take care of local_fluo_flashBefOmitt below if norm_to_max, doShift, or doScale are set to 1.


                local_fluo_allOmitt0_orig = local_fluo_allOmitt + 0   
                local_fluo_flashBefOmitt_orig = local_fluo_flashBefOmitt + 0
    #                local_fluo_allOmitt = local_fluo_allOmitt0_orig 


                this_sess.at[index, ['n_omissions', 'n_neurons', 'frame_dur', 'flash_omit_dur_all', 'flash_omit_dur_fr_all']] = num_omissions, num_neurons, frame_dur, flash_omit_dur_all, flash_omit_dur_fr_all
                print('===== plane %d: %d neurons; %d trials =====' %(index, num_neurons, num_omissions))

#                 this_sess_l.at[index, 'local_fluo_allOmitt'] = local_fluo_allOmitt                
#                 this_sess_l.at[index, 'local_fluo_flashBefOmitt'] = local_fluo_flashBefOmitt
                
                #%%#############################################################
                ###### The following two variables are the key variables: ######
                ###### (they are ready to be used; unless you want to shift and scale them below) 
                ###### local_fluo_allOmitt  (frames x units x trials)     ###### # calcium traces aligned on omissions
                ###### local_time_allOmitt  (frames x trials)             ###### # and their time traces
                ################################################################


                
                

                ################################################################################
                ##############################################################
                ##############################################################                
                #### Make control data to remove the simultaneous aspect of dual beam mesoscope (ie the coupled planes)
                # by resampling [the dual-beam mesoscope] data to resemble the single-beam mesoscope data in the following way:
                # for each plane, we take either odd or even frame indices of the original df/f. Going through 4 uncoupled planes first, and the other 4 uncoupled planes next. As if the 8 planes are imaged consecutively (as opposed to 2 planes simultaneously.)
                # remember: one of the planes in the coupled plane group must take even frame indices, and the other plane must take odd frame indices; because we want to remove the simultaneous aspect of dual beam mesoscope.
                ##############################################################
                ##############################################################
                ################################################################################
                
                if controlSingleBeam_oddFrPlanes[0]==1:
                    ### Decide for which planes you are going to take even (or odd) frame indices of df/f    # index:0-3: LM depths; index: 4-7: V1 depths
                    # remember: one of the planes in the coupled plane group takes even frame indices, and the other plane takes odd frame indices; because we want to remove the simultaneous aspect of dual beam mesoscope.
                    # coupled planes: 0-3 ; 1-2 ; 4-7 ; 5-6
#                     odd_fr_planes = np.array([0,2,5,7]) # if the index is among these, we will take df/f from odd frame indices         
                    odd_fr_planes = np.array(controlSingleBeam_oddFrPlanes[1])
                                             
                    even_fr_planes = np.arange(0,8)[~np.in1d(np.arange(0,8), odd_fr_planes)]
                    num_frs = local_fluo_allOmitt.shape[0]
                    even_inds = np.arange(0,num_frs,2)
                    odd_inds = np.arange(1,num_frs,2)


                    if np.in1d(index, odd_fr_planes): # if the index is among these, we will take df/f from odd frame indices                                
                        local_fluo_allOmitt_resamp = local_fluo_allOmitt[odd_inds] # take df/f from odd frame indices
                    else:
                        local_fluo_allOmitt_resamp = local_fluo_allOmitt[even_inds] # take df/f from even frame indices
                
                    local_fluo_allOmitt = local_fluo_allOmitt_resamp

                
                ################################################################################
                ##############################################################
                ##############################################################
                ############ Take care of centering and scaling the traces if desired.
                ##############################################################
                ##############################################################
                ################################################################################
                
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

                    # flash-aligned traces
                    a = np.transpose(local_fluo_flashBefOmitt, (0,2,1)) # frames x trials x units                            
                    b = a / aa_mx
                    local_fluo_flashBefOmitt = np.transpose(b, (0,2,1)) # frames x units x trials


                    
                ################################################################################
                ################################################################################
                
                #%% Define baseline (frames of spontanteous activity)
                # Note: bl_gray_screen is relevant for scaling and shifting df/f traces (the default is not to do so);
                # but when quantifying flash/omission responses, we always use 10th percentile of midian traces of omit-aligned traces to compute baseline and subtract it off quantifications (within those windows).

                bl_index_pre_omit = np.arange(0,samps_bef) # you will need for response amplitude quantification, even if you dont use it below.
                bl_index_pre_flash = np.arange(0,samps_bef) # we are computing flash responses on flash aligned traces.
                # note about note below: i think even when you are computing flash responses out of omission traces, you should use above for bl computation.... it makes more sense to use the same baseline for both
                # note: below will be useful when we compute flash responses from omission aligned traces (.75sec before omission).
                # Flash_index: for flash-evoked peak timing, compute it relative to flash index: 31 
                # although below is not quite accurate, if you want to be quite accurate you should align the traces on flashes!
    #             flash_index = samps_bef + np.round(-.75 / frame_dur).astype(int) # changed from floor to round: 5/16/20
    #             bl_index_pre_flash = np.arange(0,flash_index)

                if bl_gray_screen==1:
            
                    # the only problem with this method is that, we wont necessarily get the same starting point for 
                    # A, B omit-aligned traces. Especially for VIP, pre-omit in A is always lower than B. Which is an 
                    # interesting finding!

                    # use the last half of the gray screen at the beginning of the session
                    # list_flashesOmissions[0]/2 : list_flashesOmissions[0]-10  
                    ff = np.argwhere(list_flashesOmissions > 100)[0] # first flash that happens after 100sec. remember we expect the first flash to be at time ~300sec... but sometimes there is a first flash (or omission) at time 9sec, which makes no sense... in these cases use the timing of the second flash to compute spont frames
                    if ff==1:
                        print('UNCANNY: first flash should happen at ~300sec, not at %dsec! ' %list_flashesOmissions[0])
                    elif ff>1:
                        sys.exit('REALLY?? the first few flashes happened before the expected 5min gray screen!! Investigate this!!!')
                    b = np.round((list_flashesOmissions[ff]/2.) / frame_dur).astype(int)
                    e = np.round((list_flashesOmissions[ff]-10) / frame_dur).astype(int)                
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

                ################################################################################
                #%% Center and scale (normalize) the average trace by baseline mean and std (ie pre-omitted fluctuations)

        #                local_fluo_allOmitt0_orig = local_fluo_allOmitt + 0
        #                local_fluo_allOmitt = local_fluo_allOmitt0_orig + 0            
                if doShift:  # shift the y values so baseline mean is at 0 at time 0            
                    local_fluo_allOmitt = local_fluo_allOmitt - bl_ave
                    local_fluo_flashBefOmitt = local_fluo_flashBefOmitt - bl_ave

                if doScale:  # scale the traces by baseline std
                    local_fluo_allOmitt = local_fluo_allOmitt / bl_sd
                    local_fluo_flashBefOmitt = local_fluo_flashBefOmitt / bl_sd


                    
                ################################################################################
                ########################################################################
                ########################################################################
                ########################################################################
                #%% Remove signal from local_fluo_allOmitt so when we later compute correlations, they are not affected by signal, and instead reflect noise corrletions.
                # for each image type, in omit-aligned trace, we will compute average response across all trials of that image, and will subtract this average response from individual trials. This will be done for each neuron separately.
                # we know that the image before and after an omission will be the same type, so there will be 8 types of omission-aligned traces.
                ########################################################################
                ########################################################################
                ########################################################################
                ################################################################################
                
                if np.logical_and(doCorrs==1, subtractSigCorrs==1):

                    ###### find image types within local_fluo_allOmitt (this applies to the 1 image before and 1 image after omission, both are always of the same type.)
                    # I believe you need to compute below only for one of the experiments, as img_arr is a session property.
                    img_arr = []
                    for iomit in range(local_fluo_allOmitt.shape[2]):
                        image_names_surr_omit = image_names_surr_omit_all[iomit]
                        # double check that the images before and after omission are the same
                        if len(np.unique(image_names_surr_omit[[0,1]]))>1:
                            sys.exit('Images before and after omission are NOT the same!')

                        # find the types of omit-aligned traces based on their image types
                        img_arr.append(image_names_surr_omit_all[iomit][0]) # get the first image

                    img_arr = np.array(img_arr)
                    img_names, iimg = np.unique(img_arr, return_inverse=True)    
                    num_trials_perImageType = np.array([sum(iimg==img_ind) for img_ind in range(len(img_names))])
                    if index==0:
                        print(f'Number of omission trials per image type: {num_trials_perImageType}')
    #                 sum(num_trials_perImageType) == local_fluo_allOmitt.shape[2]


                    ###### subtract out average response to a given image from all trials of that image, on local_fluo_allOmitt
                    # take all omit-aligned trials that have a particular image type; then take their average, and subtract the average from each individual trial (of that type)
                    local_fluo_allOmitt_noSigSub = local_fluo_allOmitt + 0 # keep a copy
                    local_fluo_flashBefOmitt_noSigSub = local_fluo_flashBefOmitt + 0 # keep a copy

                    for img_ind in range(len(img_names)): # img_ind = 0
                        #### omission-aligned traces
                        omit_al_this_image = local_fluo_allOmitt[:,:,iimg==img_ind] # frames x units x trials_this_image
                        omit_al_this_image_ave = np.nanmean(omit_al_this_image, axis=-1) # frames x units # average across all trials for a given image type
                        omit_al_this_image_ave_subtracted = omit_al_this_image - omit_al_this_image_ave[:,:,np.newaxis] # frames x units x trials_this_image # for each neuron, subtract out signal (ie its average across trials with a give image), so when we compute correlations it will be noise correlations.

                        # redefine local_fluo_allOmitt trials that have image type img_ind
                        local_fluo_allOmitt[:,:,iimg==img_ind] = omit_al_this_image_ave_subtracted 


                        #### flash-aligned traces (you set this so we can run correlation on flash-aligned traces, but for now we are going to use omission-aligned traces only, and quantify flash cc from those... it wont be very accurate as not all flashes happen every .75sec.)
                        flash_al_this_image = local_fluo_flashBefOmitt[:,:,iimg==img_ind] # frames x units x trials_this_image
                        flash_al_this_image_ave = np.nanmean(flash_al_this_image, axis=-1) # frames x units # average across all trials for a given image type
                        flash_al_this_image_ave_subtracted = flash_al_this_image - flash_al_this_image_ave[:,:,np.newaxis] # frames x units x trials_this_image # for each neuron, subtract out signal (ie its average across trials with a give image), so when we compute correlations it will be noise correlations.

                        # redefine local_fluo_allOmitt trials that have image type img_ind
                        local_fluo_flashBefOmitt[:,:,iimg==img_ind] = flash_al_this_image_ave_subtracted 

                        '''
                        # average
                        plt.plot(np.mean(local_time_allOmitt, axis=-1), omit_al_this_image_ave[:,0]);
                        plt.vlines(flashes_win_trace, -.4, .4);
                        # original
                        plt.plot(np.mean(local_time_allOmitt, axis=-1), omit_al_this_image[:,0,0]);
                        plt.vlines(flashes_win_trace, -.4, .4);
                        # average subtracted
                        plt.plot(np.mean(local_time_allOmitt, axis=-1), omit_al_this_image_ave_subtracted[:,0,0]);
                        plt.vlines(flashes_win_trace, -.4, .4);

                        ##### flash aligned traces
                        # average
                        plt.plot(np.mean(local_time_flashBefOmitt, axis=-1), flash_al_this_image_ave[:,0]);
                        plt.vlines(flashes_win_trace, -.4, .4);
                        # original
                        plt.plot(np.mean(local_time_flashBefOmitt, axis=-1), flash_al_this_image[:,0,0]);
                        plt.vlines(flashes_win_trace, -.4, .4);
                        # average subtracted
                        plt.plot(np.mean(local_time_flashBefOmitt, axis=-1), flash_al_this_image_ave_subtracted[:,0,0]);
                        plt.vlines(flashes_win_trace, -.4, .4);

                        '''



                #%% Keep local_fluo_allOmitt for all planes

                this_sess_l.at[index, 'local_fluo_allOmitt'] = local_fluo_allOmitt
                '''
                this_sess_l.at[index, 'local_fluo_flashBefOmitt'] = local_fluo_flashBefOmitt
                this_sess_l.at[index, 'valid'] = 1
                this_sess_l.at[index, 'local_fluo_traces'] = local_fluo_traces
                this_sess_l.at[index, 'local_time_traces'] = local_time_traces
                '''

    #            local_fluo_allOmitt_allPlanes.append(local_fluo_allOmitt)
                if doCorrs==-1:
                    this_sess = this_sess.iloc[:,range(len(cols_basic))].join(this_sess_l) # len(cols_basic) = 13



                #%%
                ################################################################################
                ######################################################################
                ######################################################################            
                #%% Set median of omission-aligned traces (across trials). Quantify the response amplitudes and timings.
                ######################################################################
                ######################################################################            
                ################################################################################
                
                if doCorrs==0:

                    #%% Set median traces (across trials, also neurons)

    #                 traces_aveTrs_time_ns0 = np.median(local_fluo_allOmitt0_orig, axis=2)

                    ### UPDATED NOTE: 08/06/2020: I decided to go with mean (across trials, neurons) instead of median: it actually revealed that in the 3rd plane of LM (especially in LM) the mean response of Slc is slightly going up after omissions (we know some neurons are like this among slc).
                    # Also, when we plot summary mice data, we take average across sessions of mice, so it makes sense that we also take average across trials and neurons (and not mean).    
                    #
                    ### NOTE: tested on Slc mouse 451787, session_id 885557130:
                    ### There is a big difference between mean and median ... median seems to be a quite better measure!!
                    ### So I am going with median and iqr!

                    traces_aveTrs_time_ns = np.nanmean(local_fluo_allOmitt, axis=2) # time x neurons # Median across trials            
                    traces_aveNs_time_trs = np.nanmean(local_fluo_allOmitt, axis=1) # time x trials  # Median across neurons
                    traces_aveTrs_time_ns0 = traces_aveTrs_time_ns + 0
                    # currently not saving the following ...             
                    #            traces_sdTrs_time_ns = st.iqr(local_fluo_allOmitt, axis=2) # time x neurons # Std across trials            
                    #            traces_sdNs_time_trs = st.iqr(local_fluo_allOmitt, axis=1) # time x trials  # Std across neurons


                    ######### flash-aligned traces  
                    traces_aveTrs_time_ns_f = np.nanmean(local_fluo_flashBefOmitt, axis=2) # time x neurons # Median across trials
                    traces_aveNs_time_trs_f = np.nanmean(local_fluo_flashBefOmitt, axis=1) # time x trials  # Median across neurons
                    traces_aveTrs_time_ns0_f = traces_aveTrs_time_ns_f + 0

                    #%% Compute baseline of (normalized) traces (as 10th percentile pre-omit activity)
                    # Also, perhaps do a second shift just to make sure the pre-omit activity has baseline at 0.

                    # you may decide not to do it, if you want to keep the baseline information.
                    # or you may decide to do it, if you want to make the comparison of traces independent of their baseline changes.

                    # we will anyway do it, when it comes to the peak amplitude computation.

                    ##### Remember: if you use median across trials (instead of mean) to compute aveTrs traces, the baseline will not be anymore at 0 
                    # even if you have shifted it to 0... because baseline is computed on the mean of each trial during bl_index_pre_omit!
                    # so to compute peak on aveTrs traces, we will again shift the traces below so their baseline is at 0.

                    # defined in line 658
#                     bl_index_pre_omit = np.arange(0,samps_bef) # you will need for response amplitude quantification, even if you dont use it below.
#                     bl_index_pre_flash = np.arange(0,samps_bef) # we are computing flash responses on flash aligned traces.
                    
                    '''
                    ### the 2 methods below lower the baseline value by a lot.
                    
                    # try 10th or 20th percentile                    
                    bl_preOmit = np.percentile(traces_aveTrs_time_ns[bl_index_pre_omit,:], bl_percentile, axis=0) # neurons # use the 10th percentile
                    bl_preFlash = np.percentile(traces_aveTrs_time_ns_f[bl_index_pre_flash,:], bl_percentile, axis=0) # neurons
                    # below was used when we were computing flash-evoked responses from omission aligned traces, and bl_index_pre_flash was defined differently.
    #                 bl_preFlash = np.percentile(traces_aveTrs_time_ns[bl_index_pre_flash,:], bl_percentile, axis=0) # neurons
    #                 bl_preOmit = np.mean(traces_aveTrs_time_ns[bl_index_pre_omit,:], axis=0) # use average                   
                    print(np.mean(bl_preOmit))
                    
                    # new method
                    # get the lowest 10 percentile values of the median trace preceding omission and then compute their mean and sd
                    baseline_trace0 = traces_aveTrs_time_ns[bl_index_pre_omit] # frames x neurons
                    n_p = int(samps_bef*.5) # take 20% of baseline frames
                    # sort baseline frames, and then take the lowest 20% of them... this will define baseline frames
                    baseline_trace = np.sort(baseline_trace0, axis=0)[:n_p] # baseline_frames x neurons 
                    # compute mean and sd across frames of baseline_trace                    
                    bl_preOmit = np.mean(baseline_trace, axis=0) # neurons
                    bl_preFlash = bl_preOmit
                    print(np.mean(bl_preOmit))
                    '''
                    
                    # newest method: take frames right before the flash onset (or shifted by .25sec in case of VIP, familiar) and average them
                    b0_relOmit = np.round((flash_win_final[0]) / frame_dur).astype(int)
                    stp = np.round(-.75 / frame_dur).astype(int)
                    bl_index_pre_omit = np.sort(np.arange(samps_bef-1 + b0_relOmit , 0 , stp))
                    # Note below might be better (according to corr analysis): image and omission frames than image-1 and omission-1.
#                     bl_index_pre_omit = np.sort(np.arange(samps_bef + b0_relOmit , 0 , stp))                    
#                     bl_index_pre_omit
                    
                    bl_preOmit = np.mean(traces_aveTrs_time_ns[bl_index_pre_omit,:], axis=0) # neurons # use the 10th percentile
                    bl_preFlash = bl_preOmit
#                     print(np.mean(bl_preOmit))
    

                    # I think I should only use the following for the computation of the peak
                    # because for peak amplitude I want to see how much the trace changed... but 
                    # I rather not change the trace: traces_aveTrs_time_ns, and save it the way it is, only
                    # spontaneous activity (during the initial 5min gray screen) is subtraced.

                    if doShift_again: # this is a second shift just to make sure the pre-omit activity has baseline at 0.                                       
                    #                    traces_aveTrs_time_ns = traces_aveTrs_time_ns0 + 0                    
                        traces_aveTrs_time_ns = traces_aveTrs_time_ns0 - bl_preOmit
                        traces_aveTrs_time_ns_f = traces_aveTrs_time_ns0_f - bl_preFlash #bl_preOmit                    


                    #%%                    
                    this_sess.at[index, 'traces_aveTrs_time_ns'] = traces_aveTrs_time_ns
                    this_sess.at[index, 'traces_aveNs_time_trs'] = traces_aveNs_time_trs
                    this_sess.at[index, 'traces_aveTrs_time_ns_f'] = traces_aveTrs_time_ns_f
                    this_sess.at[index, 'traces_aveNs_time_trs_f'] = traces_aveNs_time_trs_f



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
                        # itr = np.random.permutation(num_omissions)[0]
                            ave_allNs_1tr = np.mean(local_fluo_allOmitt[:,:,itr], axis=-1) # neuron-averaged trace for trial itr
                            # iu = np.random.permutation(num_neurons)[0]
                            trace = local_fluo_allOmitt[:,:,itr] # all neuron traces, 1 trial
                            mn = min(trace.flatten())
                            mx = max(trace.flatten())

                            plt.figure()
                            plt.plot(trace, 'gray') # all neuron traces, 1 trial
                            plt.plot(ave_allNs_1tr, 'r') # neuron-averaged trace for trial itr
                            plt.vlines(flashes_win_trace_index_all[itr], mn, mx)
                        '''



                    ################################################################################
                    ######################################################################
                    ######################################################################            
                    #%% Quanity flash and omission evoked responses: use flash-aligned traces for computing flash responses
                    ######################################################################
                    ######################################################################            
                    ################################################################################
                    
                    peak_amp_eachN_traceMed, peak_timing_eachN_traceMed, peak_amp_eachN_traceMed_flash, peak_timing_eachN_traceMed_flash, \
                        peak_allTrsNs, peak_timing_allTrsNs, peak_om_av_h1, peak_om_av_h2, auc_peak_h1_h2 = \
                                omissions_traces_peaks_quantify_flashAl(traces_aveTrs_time_ns, traces_aveTrs_time_ns_f, bl_preOmit, bl_preFlash, mean_notPeak, cre, peak_win, flash_win_final, samps_bef, frame_dur, doShift_again, doPeakPerTrial, doROC, doPlots, index, local_fluo_allOmitt, num_omissions) # note now that we are using a different window for VIP,B1 thatn the rest of the sessions, we use the same window for computing flash timing as computing flash amplitude.
                    '''                              
                    # below was used when flash evoked responses were computed out of omission aligned traces
                    peak_amp_eachN_traceMed, peak_timing_eachN_traceMed, peak_amp_eachN_traceMed_flash, peak_timing_eachN_traceMed_flash, \
                        peak_allTrsNs, peak_timing_allTrsNs, peak_om_av_h1, peak_om_av_h2, auc_peak_h1_h2 = \
                                omissions_traces_peaks_quantify(traces_aveTrs_time_ns, bl_preOmit, bl_preFlash, mean_notPeak, cre, peak_win, flash_win_final, flash_win_final, flash_index, samps_bef, frame_dur, doShift_again, doPeakPerTrial, doROC, doPlots, index, local_fluo_allOmitt, num_omissions) # note now that we are using a different window for VIP,B1 than the rest of the sessions, we use the same window for computing flash timing as computing flash amplitude.
                                omissions_traces_peaks_quantify(traces_aveTrs_time_ns, bl_preOmit, bl_preFlash, mean_notPeak, cre, peak_win, flash_win, flash_win_timing, flash_index, samps_bef, frame_dur, doShift_again, doPeakPerTrial, doROC, doPlots, index, local_fluo_allOmitt, num_omissions)
                    '''


                    this_sess.at[index, ['peak_amp_eachN_traceMed']] = [peak_amp_eachN_traceMed] # neurons
                    this_sess.at[index, ['peak_timing_eachN_traceMed']] = [peak_timing_eachN_traceMed] # neurons

                    this_sess.at[index, ['peak_amp_eachN_traceMed_flash']] = [peak_amp_eachN_traceMed_flash] # neurons
                    this_sess.at[index, ['peak_timing_eachN_traceMed_flash']] = [peak_timing_eachN_traceMed_flash] # neurons

                    this_sess.at[index, ['peak_amp_trs_ns']] = [peak_allTrsNs] # trials x neurons
                    this_sess.at[index, ['peak_timing_trs_ns']] = [peak_timing_allTrsNs] # trials x neurons

                    this_sess.at[index, ['peak_h1', 'peak_h2']] = peak_om_av_h1, peak_om_av_h2
                    this_sess.at[index, 'auc_peak_h1_h2'] = auc_peak_h1_h2  




        #%% Correlation analysis starts here.
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################    
        #%% Once we have the omit-aligned traces of all experiments, run correlations between neuron paris in different areas
        #######################################################################
        #######################################################################
        #######################################################################
        #######################################################################

    #     colsn = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'frame_dur', 'flash_omit_dur_all', 'flash_omit_dur_fr_all', \
    #                       'cc_cols_area12', 'cc_cols_areaSame', \
    #                       'cc_a12', 'cc_a11', 'cc_a22', 'p_a12', 'p_a11', 'p_a22', \
    #                       'cc_a12_shfl', 'cc_a11_shfl', 'cc_a22_shfl', 'p_a12_shfl', 'p_a11_shfl', 'p_a22_shfl'])

        this_sess_allExp = pd.DataFrame([], columns = colsn)

        this_sess_allExp.at[0, 'session_id'] = this_sess['session_id'].values[0]

        this_sess_allExp.at[0, 'mouse_id'] = this_sess['mouse_id'].values[0]
        this_sess_allExp.at[0, 'date'] = this_sess['date'].values[0]
        this_sess_allExp.at[0, 'cre'] = this_sess['cre'].values[0]
        this_sess_allExp.at[0, 'stage'] = this_sess['stage'].values[0]

        this_sess_allExp.at[0, 'flash_omit_dur_all'] = this_sess['flash_omit_dur_all'].values[0]
        this_sess_allExp.at[0, 'flash_omit_dur_fr_all'] = this_sess['flash_omit_dur_fr_all'].values[0]

        if len(list_omitted)>0:
            this_sess_allExp.at[0, 'n_omissions'] = this_sess[~np.isnan(this_sess['n_omissions'].values.astype(float))]['n_omissions'].values[0] # if an experiment is invalid its n_omissions will be nan, so we make sure we get a non-nan value  #this_sess['n_omissions'].values[0]
        else:
            this_sess_allExp.at[0, 'n_omissions'] = np.nan

        this_sess_allExp.at[0, 'experiment_id'] = this_sess['experiment_id'].values
        this_sess_allExp.at[0, 'area'] = this_sess['area'].values
        this_sess_allExp.at[0, 'depth'] = this_sess['depth'].values
        this_sess_allExp.at[0, 'n_neurons'] = this_sess['n_neurons'].values
        this_sess_allExp.at[0, 'frame_dur'] = this_sess['frame_dur'].values


        if np.logical_and(len(list_omitted)>0, doCorrs==1):
            
            #%% Compute correlation coefficient (also try this without shifting and scaling the traces!)
            # between a layer of area1 and and a layer of area2
            # do this on all pairs of neurons

            # compute correlation coeffiecient of frame ifr in omission-align traces (eg 3 frames before omission begins) 
            # between neuron iu1 (in area1, layer d1) and neuron iu2 (in area2, layer d2)
            # basically, asking how the activity of neurons covaries eg 3 frames before omission
            # eg if all neurons predict the omission, then their activities will covary.
            # we want to see if the occurance of flash or omission, changes the coactivation pattern of 
            # neurons across specific layers of the 2 areas.

            print('Computing pairwise correlations...')

            num_depth = int(num_planes/2)
            inds_lm = np.arange(num_depth) # we can get these from the following vars: distinct_areas and i_areas[range(num_planes)]
            inds_v1 = np.arange(num_depth, num_planes)
    #            print(inds_lm); print(inds_v1); print(np.shape(this_sess_l))

            # area1: LM
            # area 2: V1
            local_fluo_allOmitt_a1 = this_sess_l.iloc[inds_lm]['local_fluo_allOmitt'].values # 4
            local_fluo_allOmitt_a2 = this_sess_l.iloc[inds_v1]['local_fluo_allOmitt'].values # 4

            '''
            # also run correlation on flash-aligned traces
            local_fluo_flashBefOmitt_a1 = this_sess_l.iloc[inds_lm]['local_fluo_flashBefOmitt'].values # 4
            local_fluo_flashBefOmitt_a2 = this_sess_l.iloc[inds_v1]['local_fluo_flashBefOmitt'].values # 4
            '''

            '''
            # you decided to change samps_bef and samps_aft in the init code to 16 and 24, instead of doing it here; see below.
            ##### run correlation on frames [-16, 24] relative to omission (or flash). we dont need to run it on so many frames: [-40, 40] that exist in the omission/flash aligned traces.
            # new shape: 40 x units x trials (instead of 80 frames, we have 40 frames now.)
            samps_bef_corr = 16 # 10 : like svm
            samps_aft_corr = 24 # 31 : like svm
            bb = samps_bef - samps_bef_corr
            aa = samps_bef + samps_aft_corr

            local_fluo_allOmitt_a1 = [local_fluo_allOmitt_a1[ipl][bb:aa] for ipl in range(len( local_fluo_allOmitt_a1))]
            local_fluo_allOmitt_a2 = [local_fluo_allOmitt_a2[ipl][bb:aa] for ipl in range(len( local_fluo_allOmitt_a2))]
            # the following two are equal
    #         local_fluo_allOmitt_a10[3][samps_bef,2,1] , local_fluo_allOmitt_a1[3][samps_bef_corr,2,1]        
            '''

            valid_a1 = this_sess_l.iloc[inds_lm]['valid'].values # 4
            valid_a2 = this_sess_l.iloc[inds_v1]['valid'].values # 4

            nfrs = samps_bef + samps_aft #local_fluo_allOmitt_a1[0].shape[0]
            if controlSingleBeam_oddFrPlanes[0]==1:
                nfrs = int(nfrs/2)
                
    #         n_neurs_a1 = [local_fluo_allOmitt_a1[iexp].shape[1] for iexp in range(len(local_fluo_allOmitt_a1))] # number of neurons for each depth of area 1
    #         n_neurs_a2 = [local_fluo_allOmitt_a2[iexp].shape[1] for iexp in range(len(local_fluo_allOmitt_a2))] # number of neurons for each depth of area 2
            # since de-crosstalking may give planes with 0 neurons, we need to run the following instead of above:
            n_neurs_a1 = np.full(num_depth, 0)
            n_neurs_a2 = np.full(num_depth, 0)
            for iexp in range(len(local_fluo_allOmitt_a1)):
                if valid_a1[iexp]==1:
                    n_neurs_a1[iexp] = local_fluo_allOmitt_a1[iexp].shape[1] # number of neurons for each depth of area 1
                if valid_a2[iexp]==1:
                    n_neurs_a2[iexp] = local_fluo_allOmitt_a2[iexp].shape[1] # number of neurons for each depth of area 1

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
                nu1 = n_neurs_a1[d1] #area1.shape[1]
                v1 = valid_a1[d1]

                for d2 in range(num_depth): # d2=0
                    print('area 1 layer %d - area 2 layer %d' %(d1, d2))
                    pw = pw+1
                    area2 = local_fluo_allOmitt_a2[d2]
                    nu2 = n_neurs_a2[d2] #area2.shape[1]
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
                            # shuffle trials, then compute corrcoeff between neurons... this serves as control to evaluate the values of corrcoeff of actual data (I'm worried they might be affected by how active neurons are... this will control for that)!
                            # although if variation across trials is different for different frames (eg after omission there is less variability in the response of a neuron across trials, compared to before omission)
                            # , then shuffling will affect different frames differently... (eg after omission because of the lower variability, shuffling will less change the corrcoeff compared to actual data, but before omission, bc of the higher variability across trials, shuffling will change corrcoef more compared to actual data.)
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
                nu1 = n_neurs_a1[d1] #area1_d1.shape[1]
                v1 = valid_a1[d1]

                for d2 in np.arange(d1, num_depth): # d2=0 # d2=d1:4
                    print('area 1 layer %d - area 1 layer %d' %(d1, d2))
                    pw = pw+1
                    area1_d2 = local_fluo_allOmitt_a1[d2]
                    nu2 = n_neurs_a1[d2] #area1_d2.shape[1]
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
                nu1 = n_neurs_a2[d1] #area2_d1.shape[1]
                v1 = valid_a2[d1]

                for d2 in np.arange(d1, num_depth): # d2=1
                    print('area 2 layer %d - area 2 layer %d' %(d1, d2))
                    pw = pw+1
                    area2_d2 = local_fluo_allOmitt_a2[d2]
                    nu2 = n_neurs_a2[d2] #area2_d2.shape[1]
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
            if corr_doShfl:
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

            else:
                c12_a12_shfl_av = np.nan
                c12_a11_shfl_av = np.nan        
                c12_a22_shfl_av = np.nan        
                p12_a12_shflm = np.nan
                p12_a11_shflm = np.nan
                p12_a22_shflm = np.nan            


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

        if doCorrs==1:
            this_sess = this_sess_allExp


            
        #####################################################################################
        #####################################################################################
        #%% Save the output of the analysis (for each session)
        #####################################################################################
        #####################################################################################

        if doCorrs==1: # if doCorrs, we save values for each session (because we run it on the cluster), otherwise, we keep values of each session, append them, and them save "all_sess" var.

            save1file = 1 # for large files, set to 0, so different vars are saved in different h5 files.

            #%% Set pandas dataframe input_vars, to save it below

            ### set this so you know what input vars you ran the script with
            cols = np.array(['norm_to_max', 'mean_notPeak', 'peak_win', 'flash_win', 'flash_win_timing', 'flash_win_vip', 'bl_percentile', 'num_shfl_corr', 'trace_median',
                         'samps_bef', 'samps_aft', 'doScale', 'doShift', 'doShift_again', 'bl_gray_screen', 'subtractSigCorrs', 'controlSingleBeam_oddFrPlanes'])

            input_vars = pd.DataFrame([], columns=cols)
            input_vars.at[0, cols] = norm_to_max, mean_notPeak, peak_win, flash_win, flash_win_timing, flash_win_vip, bl_percentile, num_shfl_corr, trace_median, samps_bef, samps_aft, doScale, doShift, doShift_again, bl_gray_screen, subtractSigCorrs, controlSingleBeam_oddFrPlanes
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
                if controlSingleBeam_oddFrPlanes[0]==1:
                    namespec = namespec + '_controlSingleBeam'
            else:
                namespec = ''

            if subtractSigCorrs==0:
                namespec = namespec + '_withSigCorrs'                

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

        
        
        







#%% Set vars to call the function above (when submitting jobs to the cluster)
###########################################################################################        
###########################################################################################
########################################################################################### 
###########################################################################################
###########################################################################################

#%%
controlSingleBeam_oddFrPlanes = [1, [0,2,5,7]] # [0, []] #if 1st element is 1, make control data to remove the simultaneous aspect of dual beam mesoscope (ie the coupled planes) to see if the correlation results require the high temporal resolution (11Hz) of dual beam vs. 5Hz of single beam mesoscope # 2nd element: odd_fr_planes = [0,2,5,7] # if the plane index is among these, we will take df/f from odd frame indices         

saveResults = 1  # save the all_sess pandas at the end

doCorrs = 1  #-1 #0 #1 # if -1, only get the traces, dont compute peaks, mean, etc. # if 0, compute omit-aligned trace median, peaks, etc. If 1, compute corr coeff between neuron pairs in each layer of v1 and lm
# note: when doCorrs=1, run this code on the cluster: omissions_traces_peaks_init_pbs.py (this will call omissions_traces_peaks_pbs.py) 
num_shfl_corr = 2 #50 # set to 0 if you dont want to compute corrs for shuffled data # shuffle trials, then compute corrcoeff... this serves as control to evaluate the values of corrcoeff of actual data    
subtractSigCorrs = 1 # if 1, compute average response to each image, and subtract it out from all trials of that image. Then compute correlations; ie remove signal correlations.

# To align on omissions, get 40 frames before omission and 39 frames after omission
samps_bef = 16 #40 # 40 is a lot especially for correlation analysis which takes a long time; the advantage of 40 thouth is that bl_preOmit and bl_preFlash will be more accurate as they use 10th percentile of frames before omission (or flash), so more frames means better accuracy.
samps_aft = 24 #40

use_ct_traces = 1 # if 0, we go with dff traces saved in analysis_dir (visual behavior production analysis); if 1, we go with crosstalk corrected dff traces on rd-storage
use_np_corr = 1 # will be used when use_ct_traces=1; if use_np_corr=1, we will load the manually neuropil corrected traces; if 0, we will load the soma traces.

# remember the following two were 1, for autoencoder analuysis you changed them to 0.
doScale = 0  # 1 # Scale the traces by baseline std
doShift = 0  # 1 # Shift the y values so baseline mean is at 0 at time 0
doShift_again = 0 # 0 # this is a second shift just to make sure the pre-omit activity has baseline at 0. (it is after we normalize the traces by baseline ave and sd... but because the traces are median of trials (or computed from the initial gray screen activity), and the baseline is mean of trials, the traces wont end up at baseline of 0)
bl_gray_screen = 1 # 1 # if 1, baseline will be computed on the initial gray screen at the beginning of the session
# if 0, get the lowest 20 percentile values of the median trace preceding omission and then compute their mean and sd
# this value will only affect the traces, but not the peak quantification, because for peak quantification we subtract the local baseline
# which is computed as 10th percentile of activity preceding each omission

# THE following is not a problem anymore! as peak timing is not being computed on SST/SLC!
# NOTE: not sure how much to trust peak measures (amplitude and timing) for non-VIP lines...
# they dont really have a peak, and you have the note below in the omissions_traces_peaks.py code:
# if the mouse is NOT a VIP line, instead of computing the peak, we compute the trough: we invert the trace, and then compute the max. (this is because in VIP mice the response after omission goes up, but not in SST or Slc)
mean_notPeak = 1  # set to 1, so mean is computed; it is preferred because SST and SLC responses don't increase after omission
# if 1, to quantify omission-evoked response, compute average trace during peak_win, instead of the peak amplitude.
# if 1, we will still use the timing of the "peak" for peak timing measures ...
# Note sure of the following: if mean_notPeak is 1, it makes sense to go with a conservative (short) peak_win so the results are not contaminated by the responses to the flash that follows the omission
# set mean_notPeak to 0 if you want to measure trough (on traces with decreased response after the omission)

# you should totally not go beyond 0.75!!! # go with 0.75ms if you don't want the responses to be contaminated by the next flash
peak_win = [0, .5] #[0, .75] # this should be named omit_win
# [0,2] # flashes are 250ms and gray screen is 500ms # sec # time window to find peak activity, seconds relative to omission onset
# [-.75, 0] is too large for flash responses... (unlike omit responses, they peak fast!
# [-.75, -.4] #[-.75, 0] # window (relative to omission) for computing flash-evoked responses (750ms includes flash and gray)
flash_win = [0, .35] #[0, .75] # now using flash-aligned traces; previously flash responses were computed on omission-aligned traces; problem: not all flashes happen every .75sec # [-.75, 0] # previous value: # [-.75, -.25] # 
flash_win_vip = [-.25, .1] #[-.25, .5] # on flash-aligned traces # [-1, -.25] # previous value (for flash_win of all cre lines): # [-.75, -.25] # 
flash_win_timing = [-.875, -.25] # this is not needed anymore bc we are using different values for vip vs others, so we just use flash_win and flash_win_vip to compute timing. # previous value: #[-.85, -.5] 
# we use flash_win_timing, so it can take care of vip responses too, but maybe start using different windows for non-vip and vip lines; those windows will be used for both timing and amplitude computation. This whole thing is because vip responses in familiar imagesets precedes the flash occurrence. So for the first novel session we need to use a similar window as in non-vip lines, but for all the other sessions we need to use flash_win_vip to find the amp and timing of vip responses. 
# one solution is to go with flash_win (for non-vip and vip, B1 session) and flash_win_vip (for vip sessions except B1); use those same windows for both amp and timing computation. 
# the other solution is what currently we are using which is a wider window for flash_timing so it takes care of vip responses; and a flash_win that is non-optimal for vip responses (bc it doesnt precede the image; although now that i extended it to 0 (previously it was [-.75, -.25]) it will include vip ramping activity too, so perhaps non-optimality is less of a problem).

bl_percentile = 10 #20  # for peak measurements, we subtract pre-omit baseline from the peak. bl_percentile determines what pre-omit values will be used.      

norm_to_max = 0  # normalize each neuron trace by its max
# In this case doScale is automatically set to zero, because:
# remember whether you are normalizing to max peak activity or not will not matter if you normalize by baseline SD. because if you normalize to max peak activity, baseline SD will also get changed by that same amount, so after normalizing to baseseline SD the effect of normalizing to max peak activity will be gone.

# if 1, use peak measures computed on the trial-median traces (less noisy); if 0, use peak measures computed on individual trials.
trace_median = 1


doPlots = 0  # some plots (not the main summary ones at the end)... set to 0
doROC = 0  # if 1, do ROC analysis: for each experiment, find, across all neurons, how separable the distributions of trial-averaged peak amplitudes are for half-1 omissions vs. half-2 omissions
#dosavefig = 1

use_common_vb_roi = 1 # only those ct dff ROIs that exist in vb rois will be used.



if norm_to_max:
    doScale = 0

if doROC:
    from sklearn.metrics import roc_curve, auc



#%% Load vars related to list_all_sessions_valid 
# (read the pickle file saved in the script: set_valid_sessions.py)

# below is not needed anymore since we are using data release sessions

import pickle

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

# validSessName = os.path.join(dir_valid_sess, 'valid_sessions' + '.pkl')
'''
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
'''

validity_log_all = [] # defining it here because we commented above. The right thing is to just remove it from the function input, but for now leaving it...
    
    
###############################################
# set sessions and get the metadata
###############################################

# Use the new list of sessions that are de-crosstalked and will be released in March 2021
import pandas as pd
import numpy as np
import visual_behavior.data_access.loading as loading

# metadata_meso_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/meso_decrosstalk/meso_experiments_in_release.csv'
# metadata_valid = pd.read_csv(metadata_meso_dir)
experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')
metadata_valid = experiments_table[experiments_table['project_code']=='VisualBehaviorMultiscope'] # multiscope sessions

sessions_ctDone = metadata_valid['ophys_session_id'].unique()
print(sessions_ctDone.shape)


list_all_sessions_valid = sessions_ctDone

print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')


# get the list of 8 experiments for all sessions
experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')

metadata_all = experiments_table[np.in1d(experiments_table['ophys_session_id'].values, list_all_sessions_valid)]


# set the list of experiments for each session in list_all_sessions_valid
try:
    list_all_experiments = np.reshape(metadata_all['ophys_experiment_id'].values, (8, len(list_all_sessions_valid)), order='F').T    

except Exception as E:
    print(E)
    
    list_all_experiments = []
    for sess in list_all_sessions_valid: # sess = list_all_sessions_valid[0]
        es = metadata_all[metadata_all['ophys_session_id']==sess]['ophys_experiment_id'].values
        list_all_experiments.append(es)
    list_all_experiments = np.array(list_all_experiments)

    list_all_experiments.shape

    b = np.array([len(list_all_experiments[i]) for i in range(len(list_all_experiments))])
    no8 = list_all_sessions_valid[b!=8]
    if len(no8)>0:
        print(f'The following sessions dont have all the 8 experiments, excluding them! {no8}')    
        list_all_sessions_valid = list_all_sessions_valid[b==8]
        list_all_experiments = list_all_experiments[b==8]

        print(list_all_sessions_valid.shape, list_all_experiments.shape)

    list_all_experiments = np.vstack(list_all_experiments)


list_all_experiments = np.sort(list_all_experiments)
print(list_all_experiments.shape)




    
    
#%% crosstalk-corrected sessions

if 0: #use_ct_traces: # lims now contains decrosstalked traces (2/11/2021)
    '''
    # Later when we call load_session_data_new, it will load the ct traces if use_ct_traces is 1; otherwise it will seet dataset which loads the original dff traces using vb codes.

    # Get a list of sessions ready for post-cross talk analysis with the following code:
    import visual_behavior.ophys.mesoscope.utils as mu
    import logging
    lims_done, lims_notdone, meso_data = mu.get_lims_done_sessions()
    lims_done = lims_done.session_id.drop_duplicates()
    lims_done.values
    '''

    # remove these sessions; because: 
    # ct dff files dont exist (843871999)
    # one depth is nan, so dataset cannot be set: 958772311
    # dataset.cell_specimen_table couldnt be set (error in load_session_data_new): 986767503
    
    # this is resolved by using loading dataset: there is no common ROI between VB and CT traces: 843871999, 882674040, 884451806, 914728054, 944888114, 952430817, 971922380, 974486549, 976167513, 976382032, 977760370, 978201478, 981705001, 982566889, 986130604, 988768058, 988903485, 989267296, 990139534, 990464099, 991958444, 993253587, 
    
    # used for nature methods and nature communications submissions    
#     sessions_ctDone = np.array([839514418, 840490733, 841303580, 841682738, 841778484, 842023261, 842364341, 842623907, 844469521, 845235947, 846871218, 847758278, 848401585, 849304162, 850667270, 850894918, 852794141, 853416532, 854060305, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 871906231, 872592724, 873247524, 874616920, 875259383, 876303107, 880498009, 880709154, 882674040, 884451806, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 904771513, 906299056, 906521029, 906968227, 907177554, 907753304, 907991198, 908441202, 911719666, 913564409, 914161594, 914639324, 914728054, 915306390, 916650386, 917498735, 918889065, 919041767, 919888953, 920695792, 921636320, 921922878, 922564930, 923705570, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931687751, 933439847, 933604359, 935559843, 937162622, 937682841, 938140092, 938898514, 939526443, 940145217, 940775208, 941676716, 946015345, 947199653, 947358663, 948042811, 948252173, 949217880, 950031363, 951410079, 952430817, 954954402, 955775716, 957020350, 958105827, 959458018, 971632311, 971922380, 973384292, 973701907, 974167263, 974486549, 975452945, 976167513, 976382032, 977760370, 978201478, 980062339, 981705001, 981845703, 982566889, 985609503, 985888070, 986130604, 987352048, 988768058, 988903485, 989267296, 990139534, 990464099, 991639544, 991958444, 992393325, 993253587, 993420347, 993738515, 993962221, 1000439105, 1002120640, 1005374186])
    
#     sessions_ctDone = np.array([839514418,  841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 868688430, 869117575])
    
    # old ct:    
#     np.array([839208243, 839514418, 840490733, 841303580, 841682738, 841778484,
#                                842023261, 842364341, 842623907, 843059122, 843871999, 844469521,
#                                845444695, 846652517, 846871218, 847758278, 848401585, 848983781,
#                                849304162, 850667270, 850894918, 851428829, 851740017, 852070825,
#                                852794141, 853177377, 853416532, 854060305, 856201876, 857040020,
#                                863815473, 864458864, 865024413, 865854762, 866197765, 867027875,
#                                868688430, 869117575, 870352564, 870762788, 871526950, 871906231,
#                                872592724, 873720614, 874070091, 874616920, 875259383, 876303107,
#                                880498009, 882674040, 884451806, 885303356, 886806800, 888009781,
#                                889944877, 906299056, 906521029])


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

    print(f'{len(list_all_sessions_valid_ct)}: Number of de-crosstalked sessions for analysis')
    list_all_experiments_valid_ct.shape
    list_all_experiments_ct.shape

    # redefine these vars
    list_all_sessions_valid = list_all_sessions_valid_ct
    list_all_experiments_valid = list_all_experiments_valid_ct
    list_all_experiments = list_all_experiments_ct
    
    
    

#%% Initiate all_sess panda table

cols_basic = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'frame_dur', 'flash_omit_dur_all', 'flash_omit_dur_fr_all'])

if doCorrs==1:
    colsa = np.array(['cc_cols_area12', 'cc_cols_areaSame',
                     'cc_a12', 'cc_a11', 'cc_a22', 'p_a12', 'p_a11', 'p_a22',
                     'cc_a12_shfl', 'cc_a11_shfl', 'cc_a22_shfl', 'p_a12_shfl', 'p_a11_shfl', 'p_a22_shfl'])
elif doCorrs==0:
    colsa = np.array(['traces_aveTrs_time_ns', 'traces_aveNs_time_trs', 'traces_aveTrs_time_ns_f', 'traces_aveNs_time_trs_f',
                     'peak_amp_eachN_traceMed', 'peak_timing_eachN_traceMed', 'peak_amp_trs_ns', 'peak_timing_trs_ns',
                     'peak_h1', 'peak_h2', 'auc_peak_h1_h2', 'peak_amp_eachN_traceMed_flash', 'peak_timing_eachN_traceMed_flash'])
elif doCorrs==-1:
    colsa = np.array(['valid', 'local_fluo_traces', 'local_time_traces', 'roi_ids', 'local_fluo_allOmitt', 'list_flashes', 'list_omitted', 'running_speed', 'licks', 'rewards'])
    
cols = np.concatenate((cols_basic, colsa))

cols_this_sess_l = np.array(['valid', 'local_fluo_traces', 'local_time_traces', 'roi_ids', 'local_fluo_allOmitt', 'list_flashes', 'list_omitted', 'running_speed', 'licks', 'rewards'])


#%%
#from omissions_traces_peaks_pbs import *
    
    

#####################################################################################
#####################################################################################
#%%
isess = int(sys.argv[1])
#isess = 0
#print(isess)
#print(type(isess))


#%% Run the main function

session_id = list_all_sessions_valid[isess] #session_id = int(list_all_sessions_valid[isess]) # experiment_ids = list_all_experiments_valid[isess]
experiment_ids = list_all_experiments[isess] # we want to have the list of all experiments for each session regardless of whethere they were valid or not... this way we can find the same plane across all sessions.

# Note: we cant use metadata from allensdk because it doesnt include all experiments of a session, it only includes the valid experiments, so it wont allow us to set the metadata for all experiments.
# session_id = sessions_ctDone[isess]
metadata_now = metadata_valid[metadata_valid['ophys_session_id']==session_id]
experiment_ids_valid = metadata_now['ophys_experiment_id'].values # the problem is that this wont give us the list of all experiments for a session ... it only includes the valid experiments.
experiment_ids_valid = np.sort(experiment_ids_valid)
print(f'\n{sum(np.in1d(experiment_ids, experiment_ids_valid)==False)} of the experiments are invalid!\n')


#cnt_sess = cnt_sess + 1
print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess, len(list_all_sessions_valid)))
# print('%d: session %d out of %d sessions' %(session_id, cnt_sess+1, len(list_all_sessions_valid)))




#%% Call the main function

this_sess = omissions_traces_peaks(metadata_all, session_id, experiment_ids, experiment_ids_valid, validity_log_all, norm_to_max, mean_notPeak, peak_win, flash_win, flash_win_timing, flash_win_vip, bl_percentile, num_shfl_corr, trace_median,
   doScale, doShift, doShift_again, bl_gray_screen, samps_bef, samps_aft, doCorrs, subtractSigCorrs, saveResults, cols, cols_basic, colsa, cols_this_sess_l, use_ct_traces, use_np_corr, use_common_vb_roi, controlSingleBeam_oddFrPlanes, doPlots, doROC)

# this_sess = omissions_traces_peaks_pbs(session_id, experiment_ids, validity_log_all, norm_to_max, mean_notPeak, peak_win, flash_win, flash_win_timing, flash_win_vip, bl_percentile, num_shfl_corr, trace_median,
#        doScale, doShift, doShift_again, bl_gray_screen, samps_bef, samps_aft, doCorrs, subtractSigCorrs, saveResults, cols, use_ct_traces, doPlots, doROC)
    




