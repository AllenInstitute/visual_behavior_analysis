#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run svm_init.py to set vars needed here.

Performs SVM analysis on a particular experiment of a session.
Saves the results in: dir_svm = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
    if using same_num_neuron_all_planes, results will be saved in: dir_svm/ 'same_num_neurons_all_planes'


Created on Fri Aug  2 15:24:17 2019
@author: farzaneh
"""

from def_funs import *
from def_funs_general import *
from svm_funs import *

    
#%%
def svm_main_pbs(session_id, experiment_ids, validity_log_all, dir_svm, frames_svm, numSamples, saveResults, cols_basic, cols_svm, use_ct_traces, use_np_corr, use_common_vb_roi, use_spont_omitFrMinus1, same_num_neuron_all_planes=0):

    
    #%% Set SVM vars
    
#    same_num_neuron_all_planes = 1 # if 1, use the same number of neurons for all planes to train svm
    svm_min_neurs = 3 # min population size for training svm

#    frames_svm = np.arange(-10, 30) #30 # run svm on how what frames relative to omission
#    numSamples = 3 #10 #50
#    saveResults = 1    
    
    svmn = 'svm_gray_omit'
    if use_spont_omitFrMinus1==0:
        svmn = svmn + '_spontFrs'
        
    kfold = 10
    regType = 'l2'
    
    norm_to_max_svm = 1 # normalize each neuron trace by its max
    doPlots = 0 # svm regularization plots
    
    softNorm = 1 # soft normalziation : neurons with sd<thAct wont have too large values after normalization
    thAct = 5e-4 # will be used for soft normalization #1e-5 
    # you did soft normalization when using inferred spikes ... i'm not sure if it makes sense here (using df/f trace); but we should take care of non-active neurons: 
    # Set NsExcluded : Identify neurons that did not fire in any of the trials (during ep) and then exclude them. Otherwise they cause problem for feature normalization.
    
    smallestC = 0   
    shuffleTrs = False # set to 0 so for each iteration of numSamples, all frames are trained and tested on the same trials# If 1 shuffle trials to break any dependencies on the sequence of trails 
    # note about shuffleTrs: if you wanted to actually take effect, you should copy the relevant part from the svm code (svm_omissions) to
    # here before you go through nAftOmit frames ALSO do it for numSamples so for each sample different set of testing and training 
    # trials are used, however, still all nAftOmit frames use the same set ... 
    #... in the current version, the svm is trained and tested on different trials 
    # I don't think it matters at all, so I am not going to change it!
    cbest0 = np.nan
    useEqualTrNums = np.nan
    cbestKnown = 0 #cbest = np.nan
    
                        
    #%% Set initial vars
    
    # %matplotlib inline
    #get_ipython().magic(u'matplotlib inline')
    
    # To align on omissions, get 40 frames before omission and 39 frames after omission
    samps_bef = 40 
    samps_aft = 40
    
#    num_planes = 8
        
    svm_total_frs = len(frames_svm)
    
    now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

    cols = cols_basic
#     cols = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'traces_aveTrs_time_ns', 'traces_aveNs_time_trs', 'peak_amp_eachN_traceMed', 'peak_timing_eachN_traceMed', 'peak_amp_trs_ns', 'peak_timing_trs_ns', 'peak_h1', 'peak_h2', 'auc_peak_h1_h2'])
    
#     if same_num_neuron_all_planes:        
#         cols_svm = ['frames_svm', 'thAct', 'numSamples', 'softNorm', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs', 
#                'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
#                'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
#                'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs',
#                'inds_subselected_neurons_all', 'population_sizes_to_try', 'numShufflesN']
        
#     else:    
#         cols_svm = ['frames_svm', 'thAct', 'numSamples', 'softNorm', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs', 
#                'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
#                'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
#                'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs',
#                'testTrInds_allSamps_allFrs', 'Ytest_allSamps_allFrs', 'Ytest_hat_allSampsFrs_allFrs']
           
        
    
    #%% Load some important variables from the experiment
    
#    [whole_data, data_list, table_stim] = load_session_data(session_id) # data_list is similar to whole_data but sorted by area and depth
    [whole_data, data_list, table_stim, behav_data] = load_session_data_new(session_id, experiment_ids, use_ct_traces, use_np_corr, use_common_vb_roi)
    
    
    #%% Set number of neurons for all planes
    # Do this if training svm on the same number of neurons on all planes
    
    if same_num_neuron_all_planes==1:
        
        num_neurons_all_planes = [] # for invalid experiments it will be nan.
    
        for index, lims_id in enumerate(data_list['lims_id']):    
            '''
            for il in [7]: #range(num_planes):
                index = il
                lims_id = data_list['lims_id'].iloc[il]
            '''
            ##%% Abort the analysis if the experiment is invalid        
            if validity_log_all.iloc[validity_log_all.lims_id.values == int(lims_id)]['valid'].bool() == False: # we dont realy need this. because in svm_init, we only use valid experiment ids... so data_list here has only the valid experiment ids.
                num_neurons = np.nan
            else:            
                local_fluo_traces = whole_data[lims_id]['fluo_traces'] # neurons x frames
                num_neurons = local_fluo_traces.shape[0]    
            
            num_neurons_all_planes.append(num_neurons)
                    
        min_num_neurons_all_planes = np.nanmin(num_neurons_all_planes)
        nN_trainSVM = np.max([svm_min_neurs, min_num_neurons_all_planes]) # use the same number of neurons for all planes to train svm
        
        # Train SVM on the following population sizes
        population_sizes_to_try = [nN_trainSVM] # [x+1 for x in range(X0.shape[0])] : if you want to try all possible population sizes

        print('Number of neurons per plane: %s' %str(sorted(num_neurons_all_planes)))
        print('Training SVM on the following population sizes for all planes: %s neurons.' %str(population_sizes_to_try))
        
    


    #%%
    exp_ids = list(whole_data.keys())
    mouse = whole_data[exp_ids[0]]['mouse']
    date = whole_data[exp_ids[0]]['experiment_date']    
    cre = whole_data[exp_ids[0]]['cre'] # it will be the same for all lims_ids (all planes in a session)
    stage = whole_data[exp_ids[0]]['stage']
    
#     this_sess = mouse
#     return this_sess


    #%% Set omissions
    
    list_omitted = table_stim[table_stim['omitted']==True]['start_time'] # start time of omissions (in sec)
    list_omitted0 = list_omitted + 0
    
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
    
    
    
    #%% Set flash_win: use a different window for computing flash responses if it is VIP, and B1 session (1st novel session)
    '''
    session_novel = is_session_novel(dir_server_me, mouse, date) # first determine if the session is the 1st novel session or not

    # vip, except for B1, needs a different timewindow for computing image-triggered average (because its response precedes the image).
    # sst, slc, and vip B1 sessions all need a timewindow that immediately follows the images.
    if np.logical_and(cre.find('Vip')==1 , session_novel): # VIP, B1
        flash_win_final = flash_win_vip # [-.25, .5] 

    else: # SST, SLC, VIP (non B1)
        flash_win_final = flash_win # [0, .75]
    '''
    
    
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
    
    # initiate the pandas tabledatetime.datetime.fromtimestamp(1548449865.568)
    this_sess = pd.DataFrame([], columns = cols)
    this_sess_l = pd.DataFrame([], columns = ['valid', 'local_fluo_allOmitt', 'local_fluo_flashBefOmitt', 'local_fluo_traces', 'local_time_traces', 'list_flashes', 'list_omitted', 'running_speed', 'licks', 'rewards'])
    
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
    
        this_sess.at[index, cols[range(8)]] = session_id, lims_id, mouse, date, cre, stage, area, depth
    
#         return this_sess
    
        #%% Get traces of all neurons for the entire session
    
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
            local_time_traces = whole_data[lims_id]['time_trace']  # frame times in sec. Volume rate is 10 Hz. Are these the time of frame onsets?? (I think yes... double checking with Jerome/ Marina.) # dataset.timestamps['ophys_frames'][0]             
            
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
            
            elif len(list_omitted)>0: # only run the analysis if there are omissions in the session. #else: # 
                #%% Plot the trace of all neurons (for the entire session) and mark the flash onsets
                '''                
                plt.subplot(8,1,index+1)                
        #                y = local_fluo_traces.T # individual neurons
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

                ###################################################
                #%% Get the 5min gray screen frames at the begining of each session
                ###################################################
                flash1_index = np.argmin(np.abs(local_time_traces - list_flashesOmissions[0]))
                trace_start_index = 100 # dont pick any frames before 100 (ie the first 10sec of the session) to avoid potential problems with dff computation at the begining of the session or some potential surprise signal in vip at the begining of the session
                spont_frames = local_fluo_traces[:, trace_start_index: flash1_index] # neurons x frames


                ################################################################################
                ################################################################################
                #%% Align on omission trials:
                # samps_bef (=40) frames before omission ; index: 0:39
                # omission frame ; index: 40
                # samps_aft - 1 (=39) frames after omission ; index: 41:79

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
                        local_fluo_allOmitt[:,:, num_omissions] = local_fluo_traces[:, be:af].T # frame x neurons x omissions_trials (10Hz)
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
                        local_time_allOmitt[:, num_omissions] = local_time_traces[be:af] - indiv_time # frame x omissions_trials



                        ######### Identify the flashes (time and index on local_time_allOmitt) that happened within local_time_allOmitt                    
                        flashes_relative_timing = list_flashesOmissions - indiv_time # timing of all flashes in the session relative to the current omission
                        # the time of flashes that happened within the omit-aligned trace
                        a = np.logical_and(flashes_relative_timing >= local_time_allOmitt[0, num_omissions], flashes_relative_timing <= local_time_allOmitt[-1, num_omissions]) # which flashes are within local_time
                        flashes_win_trace = flashes_relative_timing[a] 
                        # now find the index of flashes_win_trace on local_time_allOmitt
                        a2 = [np.argmin(np.abs(local_time_allOmitt[:, num_omissions] - flashes_win_trace[i])) for i in range(len(flashes_win_trace))]
                        flashes_win_trace_index = a2 # the index of omission on local_time_traces

                        # image names for 1 flash before and 2 flashes after omission 
                        image_names_surr_omit0 = table_stim[a]['image_name'].values[np.argwhere(flashes_win_trace==0).squeeze()-1: np.argwhere(flashes_win_trace==0).squeeze()+3]
                        image_names_surr_omit = image_names_surr_omit0[image_names_surr_omit0!='omitted'] # remove 'omitted'
                        # 'im', 'omitted', 'im', 'im' : normal sequence; but somettimes: there are two alternating omissions: 'im', 'omitted', 'im', 'omitted'
                        if len(image_names_surr_omit) < 3:
                            print(f'There are two alternating omissions: {image_names_surr_omit0}')
                        if len(np.unique(image_names_surr_omit[[0,1]]))>1: # the 2nd image after omission could be a different type, but the first after omission should be the same as the one before omission.
                            print('image after omission is different from image before omission! uncanny!') # sys.exit

                        '''
                        plt.plot(local_time_allOmitt[:,num_omissions], local_fluo_allOmitt[:,:,num_omissions])
                        plt.vlines(flashes_win_trace, -.4, .4)

                        plt.plot(local_fluo_allOmitt[:,:,num_omissions])
                        plt.vlines(flashes_win_trace_index, -.4, .4)
                        '''

                        ######### Align traces on the flash preceding the omission (I used to just take .75sec before omission, but due to frame-drop/unknown issues flashes dont exactly happen every 750ms (revealed by np.unique(np.diff(list_flashesOmissions))))
                        indiv_timef = list_flashesOmissions[np.argwhere(flashes_relative_timing==0).squeeze()-1]
                        local_indexf = np.argmin(np.abs(local_time_traces - indiv_timef)) # the index of omission on local_time_traces               

                        bef = local_indexf - samps_bef
                        aff = local_indexf + samps_aft

                        # align on the flash preceding the omission
                        local_fluo_flashBefOmitt[:,:, num_omissions] = local_fluo_traces[:, bef:aff].T # frame x neurons x omissions_trials (10Hz)
                        local_time_flashBefOmitt[:, num_omissions] = local_time_traces[bef:aff] - indiv_timef # frame x omissions_trials

                        '''
                        plt.plot(np.nanmean(local_time_allOmitt, axis=1), np.nanmean(local_fluo_allOmitt, axis=(1,2)))
                        plt.vlines(flashes_win_trace, -.4, .4)

                        plt.plot(np.nanmean(local_time_flashBefOmitt, axis=1), np.nanmean(local_fluo_flashBefOmitt, axis=(1,2)))
                        plt.vlines(flashes_win_trace, -.4, .4)
                        '''

                        ######### 
                        num_omissions = num_omissions + 1
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
                        print(indiv_time, num_omissions, be, af, local_index)
    #                     print(e)


                flash_omit_dur_all = np.array(flash_omit_dur_all)
                flash_omit_dur_fr_all = np.array(flash_omit_dur_fr_all)

                # plot duration between the last flash before omission and omission. we want to see how much it varies across trials. to see how problematic it is to use omit-aligned traces for computing flash responses.
                '''
                get_ipython().magic(u'matplotlib inline')
                plt.plot(np.sort(flash_omit_dur_fr_all), label='frames'); plt.plot(10*np.sort(flash_omit_dur_all), label='duration*10sec'); plt.legend(); plt.ylabel('flash-omission gap, sorted'); plt.xlabel('omission')
                print(np.mean(np.array(flash_omit_dur_fr_all)==7)) # the majority are 8 frames apart.
                '''

                # remove the last nan rows from the traces, which happen if some of the omissions are not used for alignment (due to being too early or too late in the session) 
                if len(list_omitted)-num_omissions > 0:
                    local_fluo_allOmitt = local_fluo_allOmitt[:,:,0:-(len(list_omitted)-num_omissions)]
                    local_time_allOmitt = local_time_allOmitt[:,0:-(len(list_omitted)-num_omissions)]
                    local_fluo_flashBefOmitt = local_fluo_flashBefOmitt[:,:,0:-(len(list_omitted)-num_omissions)]
                    local_time_flashBefOmitt = local_time_flashBefOmitt[:,0:-(len(list_omitted)-num_omissions)]
                    ##### Note: you need to take care of local_fluo_flashBefOmitt below if norm_to_max, doShift, or doScale are set to 1.


                local_fluo_allOmitt0_orig = local_fluo_allOmitt + 0   
                local_fluo_flashBefOmitt_orig = local_fluo_flashBefOmitt + 0
    #                local_fluo_allOmitt = local_fluo_allOmitt0_orig 


                this_sess.at[index, ['n_omissions', 'n_neurons', 'frame_dur', 'flash_omit_dur_all', 'flash_omit_dur_fr_all']] = num_omissions, num_neurons, frame_dur, flash_omit_dur_all, flash_omit_dur_fr_all
                print('===== plane %d: %d neurons; %d trials =====' %(index, num_neurons, num_omissions))



            #%%#############################################################
            ###### The following two variables are the key variables: ######
            ###### (they are ready to be used)                        ######
            ###### calcium traces aligned on omissions:               ######
            ###### local_fluo_allOmitt  (frames x units x trials)     ######
            ###### and their time traces:                             ######
            ###### local_time_allOmitt  (frames x trials)             ######
            ################################################################


            #%%
            if num_omissions < 10:
                print('Skipping ... too few trials to do SVM training! omissions=%d' %(num_omissions))

            elif np.logical_and(same_num_neuron_all_planes , num_neurons < svm_min_neurs):
                print('Skipping ... too few neurons to do SVM training! neurons=%d' %(num_neurons))

            else:
                #%% Normalize each neuron trace by its max (so if on a day a neuron has low FR in general, it will not be read as non responsive!)

                if norm_to_max_svm==1:
                    # compute max on the entire trace of the session:
                    aa_mx = np.max(local_fluo_traces, axis=1) # neurons

                    a = np.transpose(local_fluo_allOmitt, (0,2,1)) # frames x trials x units
                    b = a / aa_mx
                    local_fluo_allOmitt = np.transpose(b, (0,2,1)) # frames x units x trials


                #%% Starting to set variables for the SVM analysis                

                meanX_allFrs = np.full((svm_total_frs, num_neurons), np.nan)
                stdX_allFrs = np.full((svm_total_frs, num_neurons), np.nan)        

                if same_num_neuron_all_planes:
                    nnt = num_neurons #X_svm[0]
                    numShufflesN = np.ceil(nnt/float(nN_trainSVM)).astype(int) # if you are selecting only 1 neuron out of 500 neurons, you will do this 500 times to get a selection of all neurons. On the other hand if you are selecting 400 neurons out of 500 neurons, you will do this only twice.

                    perClassErrorTrain_data_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, svm_total_frs), np.nan)        
                    perClassErrorTest_data_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, svm_total_frs), np.nan)
                    perClassErrorTest_shfl_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, svm_total_frs), np.nan)
                    perClassErrorTest_chance_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, svm_total_frs), np.nan)
                    if nN_trainSVM==1:
                        w_data_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, nN_trainSVM, svm_total_frs), np.nan).squeeze() # squeeze helps if num_neurons=1 
                    else:
                        w_data_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, nN_trainSVM, svm_total_frs), np.nan) # squeeze helps if num_neurons=1 
                    b_data_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, svm_total_frs), np.nan)

                    cbest_allFrs = np.full((len(population_sizes_to_try), numShufflesN, svm_total_frs), np.nan)


                ##########################
                else:
                    perClassErrorTrain_data_allFrs = np.full((numSamples, svm_total_frs), np.nan)        
                    perClassErrorTest_data_allFrs = np.full((numSamples, svm_total_frs), np.nan)
                    perClassErrorTest_shfl_allFrs = np.full((numSamples, svm_total_frs), np.nan)
                    perClassErrorTest_chance_allFrs = np.full((numSamples, svm_total_frs), np.nan)
                    if num_neurons==1:
                        w_data_allFrs = np.full((numSamples, num_neurons, svm_total_frs), np.nan).squeeze() # squeeze helps if num_neurons=1 
                    else:
                        w_data_allFrs = np.full((numSamples, num_neurons, svm_total_frs), np.nan) # squeeze helps if num_neurons=1 
                    b_data_allFrs = np.full((numSamples, svm_total_frs), np.nan)

                    cbest_allFrs = np.full(svm_total_frs, np.nan)



                numTrials = 2*local_fluo_allOmitt.shape[2]   # numDataPoints = X_svm.shape[1] # trials             
                len_test = numTrials - int((kfold-1.)/kfold*numTrials) # number of testing trials   
                numDataPoints = numTrials

                if len_test==1:
                    Ytest_hat_allSampsFrs_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan).squeeze() # squeeze helps if len_test=1
                else:
                    Ytest_hat_allSampsFrs_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan)
                Ytest_allSamps_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan)         
                testTrInds_allSamps_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan)

                ifr = -1


                #%% Use SVM to classify population activity on the gray-screen frame right before the omission vs. the activity on frame + nAftOmit after omission 

                for nAftOmit in frames_svm: #range(frames_after_omission): # nAftOmit = frames_svm[0]
                    ifr = ifr+1
                    print('\n================ Running SVM on frame %d relative to omission ================\n' %nAftOmit)
                
                    # x
                    if use_spont_omitFrMinus1==0: # if 0, classify omissions against spontanoues frames (the initial gray screen); if 1, classify omissions against the frame right before the omission
                        rand_spont_frs_num_omit = rnd.permutation(spont_frames.shape[1])[:num_omissions] # pick num_omissions random spontanous frames
                        g = spont_frames[:,rand_spont_frs_num_omit] # units x trials
                    elif use_spont_omitFrMinus1==1:
                        g = local_fluo_allOmitt[samps_bef - 1,:,:] # units x trials ; neural activity in the frame before the omission (gray screen)
                        rand_spont_frs_num_omit = np.nan
                        
                    m = local_fluo_allOmitt[samps_bef + nAftOmit,:,:] # units x trials ; neural activity on the frame of omission

                    # y
                    g_y = np.zeros(g.shape[1])
                    m_y = np.ones(m.shape[1])

                    # now set the x matrix for svm
                    X_svm = np.concatenate((g, m), axis=1) # units x (gray + omission)
                    Y_svm = np.concatenate((g_y, m_y)) # trials (gray + omission)


                    #%% Z score (make each neuron have mean 0 and std 1 across all trials)

                    # mean and std of each neuron across all trials (trials here mean both gray screen frames and omission frames)
                    m = np.mean(X_svm, axis=1)
                    s = np.std(X_svm, axis=1)

                    meanX_allFrs[ifr,:] = m # frs x neurons
                    stdX_allFrs[ifr,:] = s     

                    # soft normalziation : neurons with sd<thAct wont have too large values after normalization                    
                    if softNorm==1:
                        s = s + thAct     

                    X_svm = ((X_svm.T - m) / s).T # units x trials


                    #%% ############################ Run SVM analysis #############################

                    if same_num_neuron_all_planes:
                        perClassErrorTrain_nN_all, perClassErrorTest_nN_all, wAllC_nN_all, bAllC_nN_all, cbestAllFrs_nN_all, cvect, \
                                perClassErrorTest_shfl_nN_all, perClassErrorTest_chance_nN_all, inds_subselected_neurons_all, numShufflesN \
                        = set_best_c_diffNumNeurons(X_svm,Y_svm,regType,kfold,numDataPoints,numSamples,population_sizes_to_try,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest0,\
                                     fr2an=np.nan, shflTrLabs=0, X_svm_incorr=0, Y_svm_incorr=0, mnHRLR_acrossDays=np.nan)

                    else:
                        perClassErrorTrain, perClassErrorTest, wAllC, bAllC, cbestAll, cbest, cvect,\
                                perClassErrorTestShfl, perClassErrorTestChance, testTrInds_allSamps, Ytest_allSamps, Ytest_hat_allSampsFrs0, trsnow_allSamps \
                        = set_best_c(X_svm,Y_svm,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest0,\
                                     fr2an=np.nan, shflTrLabs=0, X_svm_incorr=0, Y_svm_incorr=0, mnHRLR_acrossDays=np.nan) # outputs have size the number of shuffles in setbestc (shuffles per c value)


                    ########## Set the SVM output at best C (for all samples)

                    if same_num_neuron_all_planes:
                        for i_pop_size in range(len(population_sizes_to_try)):

                            for inN in range(numShufflesN):
                                # each element of perClassErrorTrain_nN_all is for a neural population of a given size. 
                                # perClassErrorTrain_nN_all[0] has size: # numShufflesN x nSamples x nCvals
                                perClassErrorTrain = perClassErrorTrain_nN_all[i_pop_size][inN] 
                                perClassErrorTest = perClassErrorTest_nN_all[i_pop_size][inN]
                                perClassErrorTestShfl = perClassErrorTest_shfl_nN_all[i_pop_size][inN]
                                perClassErrorTestChance = perClassErrorTest_chance_nN_all[i_pop_size][inN]
                                wAllC = wAllC_nN_all[i_pop_size][inN] # each element of wAllC_nN_all has size: numShufflesN x nSamples x nCvals x nNerons_used_for_training
                                bAllC = bAllC_nN_all[i_pop_size][inN]
                                cbest = cbestAllFrs_nN_all[i_pop_size][inN]

                                indBestC = np.squeeze([0 if cbestKnown else np.in1d(cvect, cbest)])

                                perClassErrorTrain_data = perClassErrorTrain[:,indBestC].squeeze()
                                perClassErrorTest_data = perClassErrorTest[:,indBestC].squeeze()
                                perClassErrorTest_shfl = perClassErrorTestShfl[:,indBestC].squeeze()
                                perClassErrorTest_chance = perClassErrorTestChance[:,indBestC].squeeze()
                                w_data = wAllC[:,indBestC,:].squeeze() # samps x neurons
                                b_data = bAllC[:,indBestC].squeeze()


                                ########## keep SVM vars for all frames after omission 
                                cbest_allFrs[i_pop_size,inN,ifr] = cbest

                                perClassErrorTrain_data_allFrs[i_pop_size,inN,:,ifr] = perClassErrorTrain_data # numSamps        
                                perClassErrorTest_data_allFrs[i_pop_size,inN,:,ifr] = perClassErrorTest_data
                                perClassErrorTest_shfl_allFrs[i_pop_size,inN,:,ifr] = perClassErrorTest_shfl
                                perClassErrorTest_chance_allFrs[i_pop_size,inN,:,ifr] = perClassErrorTest_chance
                                if num_neurons==1:
                                    w_data_allFrs[i_pop_size,inN,:,ifr] = w_data # numSamps x neurons
                                else:
                                    w_data_allFrs[i_pop_size,inN,:,:,ifr] = w_data # numSamps x neurons
                                b_data_allFrs[i_pop_size,inN,:,ifr] = b_data            


                    ################################################
                    else:   

                        indBestC = np.squeeze([0 if cbestKnown else np.in1d(cvect, cbest)])
            #                for ifr in range(nFrs):   
        #                if cbestKnown:
        #                    indBestC = 0
        #                else:
        #                    indBestC = np.in1d(cvect, cbest)

                        perClassErrorTrain_data = perClassErrorTrain[:,indBestC].squeeze()
                        perClassErrorTest_data = perClassErrorTest[:,indBestC].squeeze()
                        perClassErrorTest_shfl = perClassErrorTestShfl[:,indBestC].squeeze()
                        perClassErrorTest_chance = perClassErrorTestChance[:,indBestC].squeeze()
                        w_data = wAllC[:,indBestC,:].squeeze() # samps x neurons
                        b_data = bAllC[:,indBestC].squeeze()
                        Ytest_hat_allSampsFrs = Ytest_hat_allSampsFrs0[:,indBestC,:].squeeze()


                        ########## keep SVM vars for all frames after omission 
                        cbest_allFrs[ifr] = cbest

                        perClassErrorTrain_data_allFrs[:,ifr] = perClassErrorTrain_data # numSamps        
                        perClassErrorTest_data_allFrs[:,ifr] = perClassErrorTest_data
                        perClassErrorTest_shfl_allFrs[:,ifr] = perClassErrorTest_shfl
                        perClassErrorTest_chance_allFrs[:,ifr] = perClassErrorTest_chance
                        if num_neurons==1:
                            w_data_allFrs[:,ifr] = w_data # numSamps x neurons
                        else:
                            w_data_allFrs[:,:,ifr] = w_data # numSamps x neurons
                        b_data_allFrs[:,ifr] = b_data            
                        Ytest_hat_allSampsFrs_allFrs[:,:,ifr] = Ytest_hat_allSampsFrs # numSamps x numTestTrs

                        Ytest_allSamps_allFrs[:,:,ifr] = Ytest_allSamps # numSamps x numTestTrs                                        
                        testTrInds_allSamps_allFrs[:,:,ifr] = testTrInds_allSamps  # numSamps x numTestTrs
            #                    trsnow_allSamps_allFrs = trsnow_allSamps                    

                        '''
                        a_train = np.mean(perClassErrorTrain, axis=0)
                        a_test = np.mean(perClassErrorTest, axis=0)
                        a_shfl = np.mean(perClassErrorTestShfl, axis=0)
                        a_chance = np.mean(perClassErrorTestChance, axis=0)

                        plt.figure()
                        plt.plot(a_train, color='k', label='train') 
                        plt.plot(a_test, color='r', label='test') 
                        plt.plot(a_shfl, color='y', label='shfl')
                        plt.plot(a_chance, color='b', label='chance')
                        plt.legend(loc='center left', bbox_to_anchor=(1, .7))
                        plt.ylabel('% Classification error')
                        plt.xlabel('C values')
                        '''                       
                        #### Sanity checks ####
                        # the following two are the same:
            #                (abs(Ytest_hat_allSampsFrs[0] - Y_svm[testTrInds_allSamps[0].astype(int)])).mean()
            #                perClassErrorTest_data[0]

                        # the following two are the same:
            #                Ytest_allSamps[2]
            #                Y_svm[testTrInds_allSamps[2].astype(int)]




                #%% Save SVM vars (for each experiment separately)
                ####################################################################################################################################                

                #%%                
                svm_vars = pd.DataFrame([], columns = np.concatenate((cols_basic, cols_svm)))

                # experiment info
                svm_vars.at[index, cols_basic] = this_sess.iloc[index, :] 

                # svm output
                if same_num_neuron_all_planes:
                    svm_vars.at[index, cols_svm] = frames_svm, thAct, numSamples, softNorm, regType, cvect, meanX_allFrs, stdX_allFrs, \
                        rand_spont_frs_num_omit, cbest_allFrs, w_data_allFrs, b_data_allFrs, \
                        perClassErrorTrain_data_allFrs, perClassErrorTest_data_allFrs, \
                        perClassErrorTest_shfl_allFrs, perClassErrorTest_chance_allFrs, \
                        inds_subselected_neurons_all, population_sizes_to_try, numShufflesN

                else:
                    svm_vars.at[index, cols_svm] = frames_svm, thAct, numSamples, softNorm, regType, cvect, meanX_allFrs, stdX_allFrs, \
                        rand_spont_frs_num_omit, cbest_allFrs, w_data_allFrs, b_data_allFrs, \
                        perClassErrorTrain_data_allFrs, perClassErrorTest_data_allFrs, \
                        perClassErrorTest_shfl_allFrs, perClassErrorTest_chance_allFrs, \
                        testTrInds_allSamps_allFrs, Ytest_allSamps_allFrs, Ytest_hat_allSampsFrs_allFrs


                #%% Save SVM results

                cre_now = cre[:cre.find('-')]
                # mouse, session, experiment: m, s, e
                if same_num_neuron_all_planes:
                    name = '%s_m-%d_s-%d_e-%s_%s_sameNumNeuronsAllPlanes_%s' %(cre_now, mouse, session_id, lims_id, svmn, now)
                else:
                    name = '%s_m-%d_s-%d_e-%s_%s_%s' %(cre_now, mouse, session_id, lims_id, svmn, now) 

                if saveResults:
                    print('Saving .h5 file')
                    svmName = os.path.join(dir_svm, name + '.h5') # os.path.join(d, svmn+os.path.basename(pnevFileName))
                    print(svmName)
                    
                    # Save to a h5 file                    
                    svm_vars.to_hdf(svmName, key='svm_vars', mode='w')


        #%%                    
                    '''
                    # save arrays to h5:
                    with h5py.File(svmName, 'w') as f:                        
                        f.create_dataset('svm_vars', data=svm_vars)
        #                    for k, v in svm_vars.items(): # a = list(svm_vars.items()); k = a[0][0]; v = a[0][1]
        #                        f.create_dataset(k, data=v)                      
                    f.close()
                    '''

                    # read h5 file                    
        #                    svm_vars = pd.read_hdf(svmName, key='svm_vars')                    
                    '''
                    f = h5py.File(svmName, 'r')                    
                    for k, v in f.items(): # a = list(f.items()); k = a[0][0]; v = np.asarray(a[0][1])
                        exec(k + ' = np.asarray(v)')                        
                    f.close()
                    '''                                         
        #                    scio.savemat(svmName, save_dict)


                ## Make dictionaries ... (I couldnt save them as h5 file)
                '''
                # define sess_info (similar to this_sess but as a dict ... to save it with other SVM vars below:

                k = this_sess.columns
                v = this_sess.iloc[index,:]                
                sess_info = dict()                
                for i in range(10):                
                    sess_info[k[i]] = v[i]

                # define SVM vars as a dictionary
                svm_vars = {'thAct':thAct, 'numSamples':numSamples, 'softNorm':softNorm, 'regType':regType, 'cvect':cvect, #'smallestC':smallestC, 'cbestAll':cbestAll, 'cbest':cbest,
                       'meanX_allFrs':meanX_allFrs, 'stdX_allFrs':stdX_allFrs, #'eventI_ds':eventI_ds, 
                       'cbest_allFrs':cbest_allFrs, 'w_data_allFrs':w_data_allFrs, 'b_data_allFrs':b_data_allFrs,
                       'perClassErrorTrain_data_allFrs':perClassErrorTrain_data_allFrs,
                       'perClassErrorTest_data_allFrs':perClassErrorTest_data_allFrs,
                       'perClassErrorTest_shfl_allFrs':perClassErrorTest_shfl_allFrs,
                       'perClassErrorTest_chance_allFrs':perClassErrorTest_chance_allFrs,
                       'testTrInds_allSamps_allFrs':testTrInds_allSamps_allFrs, 
                       'Ytest_allSamps_allFrs':Ytest_allSamps_allFrs,
                       'Ytest_hat_allSampsFrs_allFrs':Ytest_hat_allSampsFrs_allFrs}
                '''                        

            
#%%
"""
get_ipython().magic(u'matplotlib inline')

# read the svm_vars file
a = pd.read_hdf(svmName, key='svm_vars')
   
# make plots
a_train = np.mean(a.iloc[0]['perClassErrorTrain_data_allFrs'], axis=0)
a_test = np.mean(a.iloc[0]['perClassErrorTest_data_allFrs'], axis=0)
a_shfl = np.mean(a.iloc[0]['perClassErrorTest_shfl_allFrs'], axis=0)
a_chance = np.mean(a.iloc[0]['perClassErrorTest_chance_allFrs'], axis=0)

plt.figure()
plt.plot(a_train, color='k', label='train') 
plt.plot(a_test, color='r', label='test') 
plt.plot(a_shfl, color='y', label='shfl')
plt.plot(a_chance, color='b', label='chance')
plt.legend(loc='center left', bbox_to_anchor=(1, .7))                
plt.ylabel('% Classification error')
plt.xlabel('Frame after omission')  
"""








#%% For the cluster:

# if socket.gethostname() != 'ibs-farzaneh-ux2': # it's not known what node on the cluster will run your code, so all i can do is to say if it is not your pc .. but obviously this will be come problematic if running the code on a computer other than your pc or the cluster
#%% Set SVM vars

use_ct_traces = 1 # if 0, we go with dff traces saved in analysis_dir (visual behavior production analysis); if 1, we go with crosstalk corrected dff traces on rd-storage
use_np_corr = 1 # will be used when use_ct_traces=1; if use_np_corr=1, we will load the manually neuropil corrected traces; if 0, we will load the soma traces.
use_common_vb_roi = 1 # only those ct dff ROIs that exist in vb rois will be used.

same_num_neuron_all_planes = 1 # 0 #if 1, use the same number of neurons for all planes to train svm

frames_svm = range(-16, 24) # range(-10, 30) # range(-1,1) # frames_after_omission = 30 # 5 # run svm on how many frames after omission
numSamples = 50 # 2 # 10 #
saveResults = 1 # 0 #

use_spont_omitFrMinus1 = 0 # if 0, classify omissions against randomly picked spontanoues frames (the initial gray screen); if 1, classify omissions against the frame right before the omission


#%%
import pickle

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

# make svm dir to save analysis results
dir_svm = os.path.join(dir_server_me, 'SVM')
if same_num_neuron_all_planes:
    dir_svm = os.path.join(dir_svm, 'same_num_neurons_all_planes')
if not os.path.exists(dir_svm):
    os.makedirs(dir_svm)


#%% Load vars related to list_all_sessions_valid 
# (read the pickle file saved in the script: set_valid_sessions.py)

dir_valid_sess = os.path.join(dir_server_me, 'ValidSessions')
#validSessName = os.path.join(dir_valid_sess, 'valid_sessions' + '.pkl')

import re    
regex = re.compile('valid_sessions_(.*)' + '.pkl')
l = os.listdir(dir_valid_sess) 
files = [string for string in l if re.match(regex, string)]

### NOTE: # get the latest file (make sure this is correct)
files = files[-1]
validSessName = os.path.join(dir_valid_sess, files)

print(validSessName)
    
    
pkl = open(validSessName, 'rb')
dictNow = pickle.load(pkl)

for k in list(dictNow.keys()):
    exec(k + '= dictNow[k]')

pkl.close()

print(list_all_sessions_valid.shape)



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

    # remove these sessions; because: 
    # ct dff files dont exist (843871999)
    # one depth is nan, so dataset cannot be set: 958772311
    # dataset.cell_specimen_table couldnt be set (error in load_session_data_new): 986767503
    
    # this is resolved by using loading dataset: there is no common ROI between VB and CT traces: 843871999, 882674040, 884451806, 914728054, 944888114, 952430817, 971922380, 974486549, 976167513, 976382032, 977760370, 978201478, 981705001, 982566889, 986130604, 988768058, 988903485, 989267296, 990139534, 990464099, 991958444, 993253587, 
    
    sessions_ctDone = np.array([839514418, 840490733, 841303580, 841682738, 841778484, 842023261, 842364341, 842623907, 844469521, 845235947, 846871218, 847758278, 848401585, 849304162, 850667270, 850894918, 852794141, 853416532, 854060305, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 871906231, 872592724, 873247524, 874616920, 875259383, 876303107, 880498009, 880709154, 882674040, 884451806, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 904771513, 906299056, 906521029, 906968227, 907177554, 907753304, 907991198, 908441202, 911719666, 913564409, 914161594, 914639324, 914728054, 915306390, 916650386, 917498735, 918889065, 919041767, 919888953, 920695792, 921636320, 921922878, 922564930, 923705570, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931687751, 933439847, 933604359, 935559843, 937162622, 937682841, 938140092, 938898514, 939526443, 940145217, 940775208, 941676716, 946015345, 947199653, 947358663, 948042811, 948252173, 949217880, 950031363, 951410079, 952430817, 954954402, 955775716, 957020350, 958105827, 959458018, 971632311, 971922380, 973384292, 973701907, 974167263, 974486549, 975452945, 976167513, 976382032, 977760370, 978201478, 980062339, 981705001, 981845703, 982566889, 985609503, 985888070, 986130604, 987352048, 988768058, 988903485, 989267296, 990139534, 990464099, 991639544, 991958444, 992393325, 993253587, 993420347, 993738515, 993962221, 1000439105, 1002120640, 1005374186])

#     sessions_ctDone = np.array([839514418, 840490733, 841303580, 841682738, 841778484, 842023261, 842364341, 842623907, 844469521, 845235947, 846871218, 847758278, 848401585, 849304162, 850667270, 850894918, 852794141, 853416532, 854060305, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 871906231, 872592724, 873247524, 874616920, 875259383, 876303107, 880498009, 880709154, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 904771513, 906299056, 906521029, 906968227, 907177554, 907753304, 907991198, 908441202, 911719666, 913564409, 914161594, 914639324, 915306390, 916650386, 917498735, 918889065, 919041767, 919888953, 920695792, 921636320, 921922878, 922564930, 923705570, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931687751, 933439847, 933604359, 935559843, 937162622, 937682841, 938140092, 938898514, 939526443, 940145217, 940775208, 941676716, 944888114, 946015345, 947199653, 947358663, 948042811, 948252173, 949217880, 950031363, 951410079, 954954402, 955775716, 957020350, 958105827, 959458018, 971632311, 991639544, 
#                                 973384292, 973701907, 974167263, 975452945, 980062339, 981845703, 985609503, 985888070, 986767503, 987352048, 
#                                 992393325, 993420347, 993738515, 993962221, 1000439105, 1002120640, 1005374186])        

#     old ct:        
#     [839514418,  841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 868688430, 869117575]    
#     [839208243, 839514418, 840490733, 841303580, 841682738, 841778484, 842023261, 842364341, 842623907, 843059122, 843871999, 844469521, 845444695, 846652517, 846871218, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 851740017, 852070825, 852794141, 853177377, 853416532, 854060305, 856201876, 857040020, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 871906231, 872592724, 873720614, 874070091, 874616920, 875259383, 876303107, 880498009, 882674040, 884451806, 885303356, 886806800, 888009781, 889944877, 906299056, 906521029]


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
#     print(list_all_experiments_valid_ct.shape)
#     print(list_all_experiments_ct.shape)

    # redefine these vars
    list_all_sessions_valid = list_all_sessions_valid_ct
    list_all_experiments_valid = list_all_experiments_valid_ct
    list_all_experiments = list_all_experiments_ct
    
        

cols_basic = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'frame_dur', 'flash_omit_dur_all', 'flash_omit_dur_fr_all'])
if same_num_neuron_all_planes:        
    cols_svm = ['frames_svm', 'thAct', 'numSamples', 'softNorm', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs', 
           'rand_spont_frs_num_omit', 'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
           'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
           'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs',
           'inds_subselected_neurons_all', 'population_sizes_to_try', 'numShufflesN']

else:    
    cols_svm = ['frames_svm', 'thAct', 'numSamples', 'softNorm', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs', 
           'rand_spont_frs_num_omit', 'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
           'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
           'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs',
           'testTrInds_allSamps_allFrs', 'Ytest_allSamps_allFrs', 'Ytest_hat_allSampsFrs_allFrs']

        
        
#%%
isess = int(sys.argv[1])
#print(isess)
#print(type(isess))

session_id = int(list_all_sessions_valid[isess])
experiment_ids = list_all_experiments_valid[isess]
#    experiment_ids = list_all_experiments_valid[np.squeeze(np.argwhere(np.in1d(list_all_sessions_valid, session_id)))]

#cnt_sess = cnt_sess + 1
print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess+1, len(list_all_sessions_valid)))
#print('%d: session %d out of %d sessions' %(session_id, cnt_sess+1, len(list_all_sessions_valid)))




#%%
 
svm_main_pbs(session_id, experiment_ids, validity_log_all, dir_svm, frames_svm, numSamples, saveResults, cols_basic, cols_svm, use_ct_traces, use_np_corr, use_common_vb_roi, use_spont_omitFrMinus1, same_num_neuron_all_planes)



