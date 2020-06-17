#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here, we set all_sess_now and other vars needed to do umap/clustering analysis.

To set vars needed her, run omissions_traces_peaks_plots_setVars.py with "doCorrs = -1" and "useSDK = 1" to load allsess. 
(Note: all_sess is created by the script load_behavior_ophys_dataset_fn.py)

After this script, run umap_run.py to run umap.

Created on Wed Jun 10 16:05:25 2020
@author: farzaneh
"""


peak_win = [0, .75] # this should be named omit_win
flash_win = [-.75, -.25] # flash responses are computed on omission-aligned traces; # [-.75, 0] 


##################################################################################
#%% Set all_sess_now
##################################################################################

if useSDK==0:
    # remove sessions without any omissions
    # identify sessions with no omissions
    o = all_sess_2an['n_omissions'].values.astype(float) # if n_omissions is nan, that's because the experiment is invalid
    sess_no_omit = np.unique(all_sess_2an[o==0]['session_id'].values)

    # remove sessions with no omissions
    all_sess_no_omit = all_sess_2an[~np.in1d(all_sess_2an['session_id'].values, sess_no_omit)]
    all_sess_no_omit.shape[0]/8.


    # remove invalid experiments
    o = all_sess_no_omit['valid'].values # if valid is 0, that's because the experiment is invalid or there are no neurons in that plane.
    print(f'Removing {sum(o==0)} invalid experiments!')
    all_sess_now = all_sess_no_omit[o==1]
    print(len(all_sess_now))


else:
    #%% all_sess_now: it is like all_sess_allN_allO, but also has traces_fut: omit-aligned traces

    # all_sess = all_sess_allN_allO
    cols0 = all_sess_2an.columns
    cols = np.concatenate((cols0, ['n_neurons', 'n_omissions', 'local_fluo_allOmitt']))

    ##### Set all_sess_thisCre, which is like all_sess_now, but also has omit-aligned traces
    all_sess_now = pd.DataFrame([], columns=cols)
    n_frames_issue_expid_sampbef = []

    for iexp in np.arange(0, all_sess_2an.shape[0]): # iexp = 0 

            all_sess_now.at[iexp, cols0] = all_sess_2an.iloc[iexp] #.iloc[cre_all==cre]

            experiment_id = all_sess_2an.iloc[iexp]['experiment_id']
            omission_response_xr = all_sess_2an.iloc[iexp]['omission_response_xr']

            if type(omission_response_xr)==float:
                n_neurons = 0

            else:
                n_neurons = omission_response_xr['trace_id'].values.shape[0]
                n_omissions = omission_response_xr['trial_id'].values.shape[0] 
        #         time_trace = omission_response_xr['eventlocked_timestamps'].values
                traces_fut = np.transpose(omission_response_xr['eventlocked_traces'].values, (0,2,1)) # frames x neurons x trials
                n_frames = traces_fut.shape[0]
        #         the cell_ids defined below is same as cell_ids.astype(int)[:n_neurons] when cell_ids is defined out of omission_response_df
        #         cell_ids = omission_response_xr['trace_id'].values

                if n_frames < (samps_bef+samps_aft):
                    print(f'WARNING: number of frames: {n_frames}')            
                    sbef = np.argwhere(time_trace==0)[0,0] # if sampes_bef is as expected, then we will simply fix the trace by adding nans for the last missing frame, so all the omit-aligned traces have the same number of frames.
                    n_frames_issue_expid_sampbef.append([experiment_id, sbef])

                    if (sbef==samps_bef) and (n_frames == samps_bef+samps_aft-1):
                        print('Samps_bef is fine, so adding nan for the last frame')
                        traces_fut = np.concatenate((traces_fut, np.full((1, n_neurons, n_omissions), np.nan)), axis=0)

                '''
                n_neurons = all_sess_2an.iloc[iexp]['n_neurons']
                n_omissions = all_sess_2an.iloc[iexp]['n_omissions']
                n_frames = all_sess_2an.iloc[iexp]['traces_allN_allO'].shape[1]
                traces_allN_allO = all_sess_2an.iloc[iexp]['traces_allN_allO']

                traces_temp = np.reshape(traces_allN_allO, (n_neurons, n_omissions, n_frames), order='F')
                traces_fut = np.transpose(traces_temp, (2,0,1)) # frames x units x trials
        #         traces_fut.shape        
                # sanity check for reshape
        #         cells = all_sess_2an.iloc[iexp]['cell_specimen_id']
        #         icell = 18
        #         plt.plot(np.mean(traces_allN_allO[all_sess_2an.iloc[iexp]['cell_specimen_id']==cells[icell]], axis=0)) 
        #         plt.plot(np.mean(traces_allN_allO_fut[:,icell], axis=1))
                '''

                all_sess_now.at[iexp, ['n_neurons', 'n_omissions', 'local_fluo_allOmitt']] = n_neurons, n_omissions, traces_fut    

    print(np.shape(all_sess_now))


    ### Remove experiments (rows) without any neurons
    exp_noNeur = np.argwhere(np.isnan(all_sess_now['n_neurons'].values.astype(float))).flatten()
    all_sess_now = all_sess_now.drop(exp_noNeur)
    print('\nExperiments with no neurons removed!')
    print(np.shape(all_sess_now))




##################################################################################
#%% Set trial-averaged (omission-aligned) traces for all mice (all experiments, all neurons concatenated) of a specific cre line
##################################################################################

#%%
flash_ind = abs(flashes_win_trace_index_unq[flashes_win_trace_index_unq<0][-1].astype(int))

# set samps_bef_now and samps_aft_now, which are the indeces on the original omit-aligned traces in order to take one image before and one image after omission
samps_bef_now = samps_bef - flash_ind # image before omission
samps_aft_now = samps_bef + flash_ind*2 # omission and the image after

bl_index_pre_omit = np.arange(0,samps_bef) # we will need it for response amplitude quantification.


#%%
cre_all = all_sess_now['cre'].values # len_mice_withData
cre_lines = np.unique(cre_all)

# loop through each cre line
all_sess_ns_fof_all_cre = [] # size: number of distinct cre lines; # includes image before omission, omission, and image after omission (24 frames) # use samps_bef_now to find omission time.
bl_preOmit_all_cre = []
all_sess_ns_all_cre = [] # 40 frames; use samps_bef to find omission time.

for icre in range(len(cre_lines)): # icre = 0
    
    cre = cre_lines[icre]    
    thisCre_numMice = sum(cre_all==cre) # number of mice that are of line cre
    print(f'\n~{thisCre_numMice/8.} sessions for {cre} mice')

    all_sess_thisCre = all_sess_now.iloc[cre_all==cre]    
#     print(all_sess_thisCre.shape)


    ############### set area
    n = all_sess_thisCre['n_neurons'].values
    
    a = all_sess_thisCre['area'].values.astype('str') # each element is frames x units x trials    
    aa = [np.full((n[i]), a[i]) for i in range(len(a))] # replicate the area value of each experiment to the number of neurons in that experiment
    all_sess_areas = np.concatenate((aa)) # neurons_allExp_thisCre
    
    ############### set depth    
    a = all_sess_thisCre['depth'].values.astype('str') # each element is frames x units x trials    
    aa = [np.full((n[i]), a[i]) for i in range(len(a))] # replicate the area value of each experiment to the number of neurons in that experiment
    all_sess_areas = np.concatenate((aa)) # neurons_allExp_thisCre

    
    ############### set traces ###############
    # note: local_fluo_allOmitt for invalid experiments will be a nan trace as if they had one neuron
    a = all_sess_thisCre['local_fluo_allOmitt'].values # each element is frames x units x trials
#     a.shape
    
    ############### take the average of omission-aligned traces across trials; also transpose so the traces are neurons x frames
    aa = [np.nanmean(a[i], axis=2).T for i in range(len(a))] # each element is neurons x frames
#     np.shape(aa)

    ############### concantenate trial-averaged neuron traces from all experiments
    all_sess_ns = np.vstack(aa) # neurons_allExp_thisCre x frames
#     all_sess_ns.shape
#     plt.plot(np.nanmean(all_sess_ns, axis=0))

    ############### compute baseline (on trial-averaged traces); we need it for quantifying flash/omission evoked responses.
#     bl_index_pre_flash = np.arange(0,samps_bef) # we are computing flash responses on flash aligned traces.
    bl_preOmit = np.percentile(all_sess_ns.T[bl_index_pre_omit,:], bl_percentile, axis=0) # neurons # use the 10th percentile
#     bl_preFlash = np.percentile(all_sess_ns.T[bl_index_pre_flash,:], bl_percentile, axis=0) # neurons

    ############### take one image before and one image after omission
    all_sess_ns_fof_thisCre = all_sess_ns[:, samps_bef_now:samps_aft_now] # df/f contains a sequence of image, omission and image
    print(f'{all_sess_ns_fof_thisCre.shape}: size of neurons_allExp_thisCre') # neurons_allExp_thisCre x 24(frames)
#     plt.plot(np.nanmean(all_sess_ns_fof_thisCre, axis=0))
    
    # keep arrays for all cres
    all_sess_ns_fof_all_cre.append(all_sess_ns_fof_thisCre) # each element is # neurons_allExp_thisCre x 24(frames)
    bl_preOmit_all_cre.append(bl_preOmit) # each element is # neurons_allExp_thisCre
    all_sess_ns_all_cre.append(all_sess_ns) # each element is # neurons_allExp_thisCre x frames
    
    
 
#%% Compute flash and omission-evoked responses on trial-averaged traces in all_sess_ns_fof_thisCre (relative to baseline)

from omissions_traces_peaks_quantify import *

samps_bef_newTrace = flash_ind # samps_bef - samps_bef_now # index of samps_bef on the new omit-aligned traces which only include 1 image before and 2 images after the omission
# peak_win = [0, .75] # this should be named omit_win
# flash_win = [-.75, -.25] # flash responses are computed on omission-aligned traces; # [-.75, 0] 

flash_index = samps_bef_newTrace + np.round(-.75 / frame_dur).astype(int)
# flash_index = samps_bef_now + np.round(-.75 / frame_dur).astype(int)

peak_amp_eachN_traceMed_all_cre = []
peak_timing_eachN_traceMed_all_cre = []
peak_amp_eachN_traceMed_flash_all_cre = []
peak_timing_eachN_traceMed_flash_all_cre = []

for icre in range(len(cre_lines)): # icre = 0

    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
    bl_preOmit = bl_preOmit_all_cre[icre] # neurons_allExp_thisCre
    bl_preFlash = bl_preOmit

    # the input traces below need to be frames x neurons
    # output size: # neurons_allExp_thisCre
    # flash-evoked responses are computed on omission aligned traces.
    peak_amp_eachN_traceMed, peak_timing_eachN_traceMed, peak_amp_eachN_traceMed_flash, peak_timing_eachN_traceMed_flash, \
        peak_allTrsNs, peak_timing_allTrsNs, peak_om_av_h1, peak_om_av_h2, auc_peak_h1_h2 = \
                omissions_traces_peaks_quantify \
    (all_sess_ns_fof_thisCre.T, bl_preOmit, bl_preFlash, mean_notPeak, cre, peak_win, flash_win, flash_win_timing, flash_index, samps_bef_newTrace, frame_dur, doShift_again, doPeakPerTrial=0, doROC=0, doPlots=0, index=0, local_fluo_allOmitt=0, num_omissions=0)

    
    peak_amp_eachN_traceMed_all_cre.append(peak_amp_eachN_traceMed)
    peak_timing_eachN_traceMed_all_cre.append(peak_timing_eachN_traceMed)
    peak_amp_eachN_traceMed_flash_all_cre.append(peak_amp_eachN_traceMed_flash)
    peak_timing_eachN_traceMed_flash_all_cre.append(peak_timing_eachN_traceMed_flash)

    