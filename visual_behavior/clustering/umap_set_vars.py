#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To set vars needed here, run omissions_traces_peaks_plots_setVars.py with "doCorrs = -1" and "useSDK = 1" to load allsess. 
(Note: all_sess is created and saved by the script load_behavior_ophys_dataset_fn.py. Script omissions_traces_peaks_plots_setVars.py loads them)

Here, we set dataframe all_sess_now, also some arrays for distinct cre lines (all_sess_ns_fof_all_cre (trial averaged traces), image/omission response amplitude, area/depth)  
these vars will be needed to do umap/clustering analysis.

After this script, run umap_run.py to run umap/pca.


Created on Wed Jun 10 16:05:25 2020
@author: farzaneh
"""

peak_win = [0, .75] # this should be named omit_win
flash_win = [-.75, -.25] # flash responses are computed on omission-aligned traces; # [-.75, 0] 


##################################################################################
#%% Set dataframe all_sess_now
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


else: # use allenSDK
    #%% all_sess_now: it is like all_sess_allN_allO, but also has traces_fut: omit-aligned traces

    # all_sess = all_sess_allN_allO
    cols0 = all_sess_2an.columns
    cols = np.concatenate((cols0, ['plane', 'n_neurons', 'n_omissions', 'local_fluo_allOmitt']))

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

    all_sess_now0 = copy.deepcopy(all_sess_now)
    print(np.shape(all_sess_now0))
    
    

    #%% Remove experiments (rows) without any neurons
    # remember "drop" uses the value of index to remove a row, not based on its location
    
    exp_noNeur = np.argwhere(np.isnan(all_sess_now0['n_neurons'].values.astype(float))).flatten()
    all_sess_now = all_sess_now0.drop(exp_noNeur, axis=0)
    print('\nExperiments with no neurons removed!')
    print(np.shape(all_sess_now))

    

    #%% Sort all_sess_now by session_id, area and depth    

    all_sess_now = all_sess_now.sort_values(by=['session_id','area', 'depth'])

    
    #%% Add plane_index to the columns of all_sess_now
    # NOTE: below is incorrect
    # the problem with the method below is that sometimes depths of the same planes are different for v1 and lm.
    # the right way to find plane_index is to use the order of experiment_ids

    '''    
    sess_ids = np.unique(all_sess_now['session_id'].values)
    areas = np.unique(all_sess_now['area'].values) # ['VISl', 'VISp']
    
    cols = all_sess_now.columns.tolist()
    cols_newp = np.concatenate((cols[:np.argwhere(np.in1d(cols,'depth')).squeeze()+1], ['plane_index'], cols[np.argwhere(np.in1d(cols,'depth')).squeeze()+1:]))
    all_sess_nowp = pd.DataFrame([], columns=cols_newp)

    for i in sess_ids:
        this_sess = all_sess_now.iloc[all_sess_now['session_id'].values==i]
        
        ud, plane_ind = np.unique(this_sess['depth'], return_inverse=True)
#         for arn in areas: # arn = 'VISl'
#         this_sess_this_area = this_sess[this_sess['area']==arn]
#         ud, plane_ind = np.unique(this_sess_this_area['depth'], return_inverse=True)

        if len(plane_ind) < num_depth: # if not all 8 experiments exist (due to qc failure), we wont be able to set plane_index!!
            print(i)
            plane_ind = np.full(len(plane_ind), np.nan)
        
#         if len(plane_ind)>4:
#             print(np.diff(ud))            
            
        # add plane_index to this_sess
        this_sess_nowp = this_sess_this_area
        this_sess_nowp.at[:, 'plane_index'] = plane_ind
        this_sess_nowp = this_sess_nowp[cols_newp]

        all_sess_nowp = all_sess_nowp.append(this_sess_nowp)

    #     this_sess_nowp = pd.DataFrame([], columns=cols_newp)
    #     for iexp in range(this_sess.shape[0]):
    #         this_sess_nowp.at[iexp, all_sess_now.columns] = this_sess.iloc[iexp]



    #%% Remove sessions that have <4 experiments (because we couldnt set their plane_index)
    # note: this is a temporary thing, once we get all the experiments from loading (even the failed ones), we wont need to do this part anymore.
    # note we cant use drop anymore, because all_sess_nowp is sorted (, so indeces are not in numerical order anymore), also we already removed some indeces (when removing no neuron experiments). Remember "drop" uses the value of index to remove a row, not based on its location.
    
    exp_missing = np.argwhere(np.isnan(all_sess_nowp['plane_index'].values.astype(float))).flatten()
    print(f'\nRemoving {len(exp_missing)} experiments that have <4 experiments, so we cant set their plane_index.\n')
    mask = np.full((all_sess_nowp.shape[0]), True)
    mask[exp_missing] = False

    all_sess_nowp2 = all_sess_nowp[mask]

    print(all_sess_nowp.shape)
    print(all_sess_nowp2.shape)


    #%% Finally, reset all_sess_now
    
    all_sess_now = all_sess_nowp2 
    '''
    
    print(f'\n\nFinal size of all_sess_now: {all_sess_now.shape}\n\n')

          
          
          
    
##################################################################################
#%% Set trial-averaged (omission-aligned) traces (also area and depth) for all mice (all experiments, all neurons concatenated) of a specific cre line
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

# all_sess_ns_all_cre = [] # 40 frames; use samps_bef to find omission time.
all_sess_ns_fof_all_cre = [] # size: number of distinct cre lines; # includes 1 image before omission, omission, and 1 image after omission (24 frames) # use samps_bef_now to find omission time.
bl_preOmit_all_cre = []
area_all_cre = []
depth_all_cre = []
depth_categ_all_cre = []
# plane_all_cre = []

for icre in range(len(cre_lines)): # icre = 0
    
    cre = cre_lines[icre]    
    thisCre_numMice = sum(cre_all==cre) # number of mice that are of line cre
    print(f'\n~{thisCre_numMice/8.} sessions for {cre} mice')

    all_sess_thisCre = all_sess_now.iloc[cre_all==cre]    
#     print(all_sess_thisCre.shape)


    ############### set area ###############
    
    n = all_sess_thisCre['n_neurons'].values
    
    a = all_sess_thisCre['area'].values.astype('str') # each element is frames x units x trials    
    aa = [np.full((n[i]), a[i]) for i in range(len(a))] # replicate the area value of each experiment to the number of neurons in that experiment
    all_sess_areas = np.concatenate((aa)) # neurons_allExp_thisCre
    
    ############### set depth ###############   
    
    a = all_sess_thisCre['depth'].values.astype('int') # each element is frames x units x trials    
    aa = [np.full((n[i]), a[i]) for i in range(len(a))] # replicate the area value of each experiment to the number of neurons in that experiment
    all_sess_depths = np.concatenate((aa)) # neurons_allExp_thisCre

    ############### set depth category ###############   
    
    all_sess_depth_categ = np.full((len(all_sess_depths)), 1)
#     np.diff(np.unique(all_sess_depths))
    all_sess_depth_categ[all_sess_depths<100] = 0
    all_sess_depth_categ[all_sess_depths>350] = 2
    
    
    ############### set plane ###############  
    # we dont have this yet.
    
#     a = all_sess_thisCre['plane_index'].values.astype('int') # each element is frames x units x trials    
#     aa = [np.full((n[i]), a[i]) for i in range(len(a))] # replicate the area value of each experiment to the number of neurons in that experiment
#     all_sess_planes = np.concatenate((aa)) # neurons_allExp_thisCre

    
    
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
    
    
    
    ############### keep arrays for all cres ###############
    
#     all_sess_ns_all_cre.append(all_sess_ns) # each element is # neurons_allExp_thisCre x frames    
    all_sess_ns_fof_all_cre.append(all_sess_ns_fof_thisCre) # each element is # neurons_allExp_thisCre x 24(frames)
    bl_preOmit_all_cre.append(bl_preOmit) # each element is # neurons_allExp_thisCre
    area_all_cre.append(all_sess_areas)
    depth_all_cre.append(all_sess_depths)
#     plane_all_cre.append(all_sess_planes)
    depth_categ_all_cre.append(all_sess_depth_categ)

    
    
################################################################################## 
#%% Compute flash and omission-evoked responses on trial-averaged traces in all_sess_ns_fof_thisCre (relative to baseline)
##################################################################################

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



    
###########################################################################
#%% After this script, run umap_run.py to run umap/pca.
###########################################################################