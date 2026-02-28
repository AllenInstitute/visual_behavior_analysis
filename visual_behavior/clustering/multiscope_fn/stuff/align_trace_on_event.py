#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#%% Align imaging traces on omissions or flashes

# local_fluo_traces # neurons x frames
# local_time_traces # frame times in sec. Volume rate is 10 Hz. Are these the time of frame onsets?? (I think yes... double checking with Jerome/ Marina.) # dataset.timestamps['ophys_frames'][0]             
# list_omitted (or list_flashes) # pandas series showing the time (in sec) of omissions (or flashes)
# samps_bef (=40) frames before omission ; index: 0:39
    # omission frame ; index: 40
    # samps_aft - 1 (=39) frames after omission ; index: 41:79

    
# example call to the function
# samps_bef = 16 # 40 # 40 is a lot especially for correlation analysis which takes a long time; the advantage of 40 thouth is that bl_preOmit and bl_preFlash will be more accurate as they use 10th percentile of frames before omission (or flash), so more frames means better accuracy.
# samps_aft = 24 # 40 # 

# [local_fluo_allOmitt, local_time_allOmitt] = align_trace_on_event(local_fluo_traces, local_time_traces, samps_bef, samps_aft, list_omitted)
    
Created on Fri Jul  31 18:39:49 2020
@author: farzaneh
"""


def align_trace_on_event(local_fluo_traces, local_time_traces, samps_bef, samps_aft, list_omitted):

    num_neurons = local_fluo_traces.shape[0]
    
    # Keep a matrix of omission-aligned traces for all neurons and all omission trials    
    local_fluo_allOmitt = np.full((samps_bef + samps_aft, num_neurons, len(list_omitted)), np.nan) # time x neurons x omissions_trials 
    local_time_allOmitt = np.full((samps_bef + samps_aft, len(list_omitted)), np.nan) # time x omissions_trials

    # Loop over omitted trials to align traces on omissions
    num_omissions = 0 # trial number
    for iomit in range(len(list_omitted)): # # indiv_time in list_omitted: # 

        indiv_time = list_omitted[iomit] # list_omitted.iloc[iomit]
        local_index = np.argmin(np.abs(local_time_traces - indiv_time)) # the index of omission on local_time_traces               

        be = local_index - samps_bef
        af = local_index + samps_aft

        if ~np.logical_and(be >= 0 , af <= local_fluo_traces.shape[1]): # make sure the omission is at least samps_bef frames after trace beigining and samps_aft before trace end. 
            print('Event %d at time %f cannot be analyzed: %d timepoints before it and %d timepoints after it!' %(iomit, indiv_time, be, af)) 

        try: # Align on omission
            local_fluo_allOmitt[:,:, iomit] = local_fluo_traces[:, be:af].T # frame x neurons x omissions_trials (10Hz)
            local_time_allOmitt[:, iomit] = local_time_traces[be:af] - indiv_time # frame x omissions_trials
            
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

#             num_omissions = num_omissions + 1
            
        except Exception as e:
            print(f'Omission alignment failed; omission index: {iomit}, omission frame: {local_index}, starting and ending frames: {be, af}') # indiv_time, 
#             print(indiv_time, num_omissions, be, af, local_index)
#             print(e)


    # remove the last nan rows from the traces, which happen if some of the omissions are not used for alignment (due to being too early or too late in the session) 
    # find nan rows
    mask_valid_trs = np.nansum(local_fluo_allOmitt, axis=(0,1))!=0
    if sum(mask_valid_trs==False) > 0:
        local_fluo_allOmitt = local_fluo_allOmitt[:,:,mask_valid_trs]
        local_time_allOmitt = local_time_allOmitt[:,mask_valid_trs]

    num_omissions = sum(mask_valid_trs)

#     if len(list_omitted)-num_omissions > 0:
#         local_fluo_allOmitt = local_fluo_allOmitt[:,:,0:-(len(list_omitted)-num_omissions)]
#         local_time_allOmitt = local_time_allOmitt[:,0:-(len(list_omitted)-num_omissions)]

        
    return local_fluo_allOmitt, local_time_allOmitt