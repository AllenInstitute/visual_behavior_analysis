#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vars needed here are set in "omissions_traces_peaks_plots_setVars.py"

This script sets the following pandas dataframes:

trace_peak_allMice   :   includes traces and peak measures for all planes of all sessions, for each mouse.
pooled_trace_peak_allMice   : includes layer- and area-pooled vars, for each mouse. 

Also it sets vars for making summary plots across mice for each cre line:
Here, we set the following vars:
trace_peak_allMice_sessPooled : includes vars pooled across all sessions of all mice (for each plane, each area, and each depth); use this to make summary mouse plots, by averaging across pooled sessions of all mice.
trace_peak_allMice_sessAvSd : includes vars averaged across all sessions of all mice (for each plane, each area, and each depth); use this to make summary mouse plots, by averaging across mice (each mouse is here averaged across sessions).

If all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1 is 1 (analyzing A to B transition), we compute average and standard error traces and peaks, across mice for each cre line. Done for each plane separately (and each session).
    # final vars will have size:  num_cre x (8 x num_sessions).
    # remember num_sessions = 2 (last A and first B).


Follow this script by the following scripts to make plots:
omissions_traces_peaks_plots_eachMouse.py    # single mouse plots
omissions_traces_peaks_plots_sumMice.py      # summary of mice plots


Created on Mon Sep 23 16:01:05 2019
@author: farzaneh
"""

#%% Compute across-neuron med and iqr of DF/F omit-aligned traces (already median-ed across trials)
### also compute across-neuron median and iqr of peak measures
# this dataframe will later change into a much more useful dataframe name "trace_peak_allMice"... you wont need this one anymore after "trace_peak_allMice" is made.
# this dataframe is kind of confusing ... pretty much its only useful var is "med_trace".
    
doPlots = 0
len_trace = samps_bef + samps_aft
trace_peak_med_iqr_0 = pd.DataFrame([], columns=['mouse_id', 'med_peak_eachTr', 'iqr_peak_eachTr', \
    'med_peak_amp_eachN', 'iqr_peak_amp_eachN', 'med_peak_timing_eachN', 'iqr_peak_timing_eachN', 'med_trace', 'iqr_trace'])

# loop over mice based on the order in all_mice_id
for im in range(len(all_mice_id)): # im=1

    mouse_id = all_mice_id[im]
    all_sess_2an_this_mouse = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]
    
    # NOTE: 08/06/2020: I decided to go with mean (across trials, neurons) instead of median: it actually revealed that in the 3rd plane of LM (especially in LM) the mean response of Slc is slightly going up after omissions (we know some neurons are like this among slc).
    # Also, when we plot summary mice data, we take average across sessions of mice, so it makes sense that we also take average across trials and neurons (and not mean).        

    #################### #################### #################### #################### 
    #### Traces: take med and iqr across neurons; traces are already trial-median ####   
    #################### #################### #################### #################### 
    a = all_sess_2an_this_mouse['traces_aveTrs_time_ns'].values # 8 * number_of_session_for_the_mouse  ; each experiment: time x neurons
    med_trace = []
    iqr_trace = []
    for ipl in range(len(a)):
        if np.sum(~np.isnan(a[ipl])) > 0: # experiment is valid
            m = np.mean(a[ipl], axis=1)
            iq = st.iqr(a[ipl], axis=1)
        else:
            m = np.full((len_trace), np.nan)
            iq = np.full((len_trace), np.nan) # np.array([np.nan])
        med_trace.append(m)
        iqr_trace.append(iq)
        
# I dont understand why you are not using below!!! instead of the dumb code above! perhaps just delete the code above, and use the code below!!!
#    med_trace = [np.median(a[ipl], axis=1) for ipl in range(len(a))] # (8 * number_of_session_for_the_mouse) x 80 (ie time)
#    iqr_trace = [st.iqr(a[ipl], axis=1) for ipl in range(len(a))]        

    
    #################### #################### #################### ##############################################
    ####### For each trial, compute median and iqr of peak amplitude across all neurons (each experiment) #######
    #################### #################### #################### ##############################################
    a = all_sess_2an_this_mouse['peak_amp_trs_ns'].values # 8 * number_of_session_for_the_mouse  ; each experiment: trials x neurons    
    med_peak_eachTr = []
    iqr_peak_eachTr = []
    for ipl in range(len(a)):
        if np.sum(~np.isnan(a[ipl])) > 0: # experiment is valid
            m = np.mean(a[ipl], axis=1)
            iq = st.iqr(a[ipl], axis=1)
        else:
            m = np.array([np.nan])
            iq = np.array([np.nan])
        med_peak_eachTr.append(m)
        iqr_peak_eachTr.append(iq)                

#    med_peak_eachTr = [np.median(a[ipl], axis=1) for ipl in range(len(a))] # 8 * number_of_session_for_the_mouse  ; each experiment: trials
#    iqr_peak_eachTr = [st.iqr(a[ipl], axis=1) for ipl in range(len(a))] # the first 8 are for the first session, the next 8 are for the next session..


    
    '''
    # distribution of peak amplitudes across neurons is very skewed (not normal), so we go with median and iqr
    if doPlots:
    #    plt.figure(figsize=(8,6))
        plt.suptitle('mouse %d\npeak amp, each Trial' %(mouse_id), fontsize=15, y=1)
        for ipl in range(8):
    #        ax = plt.subplot(2,4,ipl+1)
            plt.figure()
            ax = plt.gca()
            for itr in range(a[ipl].shape[0]):
                plt.hist(a[ipl][itr]) #, alpha=.5)
                ax.tick_params(labelsize=10)
            plt.subplots_adjust(wspace=.4, hspace=.5)
            plt.title('plane %d' %ipl, fontsize=13)
            if ipl==0:
                plt.xlabel('peak amplitude', fontsize=12)
                plt.ylabel('# neurons', fontsize=12)
    '''



    ################################################ ################################################ ################################################ ################################################ 
    ###################### The quantities below are computed on individual trials, and we don't really end up using them!! ################################################ 
    ################################################ ################################################ ################################################ ################################################ 
    ######################## peak measures are computed on individual trials, below we compute the median across trials ##############
    #### the quantities resemble     peak_amp_eachN_traceMed    and    peak_timing_eachN_traceMed
    #### however those are computed on trial-median traces
    
    #### peak amplitude, each neuron ####
    # For each neuron, compute median and iqr of peak amplitude across all trial (each experiment)
    a = all_sess_2an_this_mouse['peak_amp_trs_ns'].values # 8 * number_of_session_for_the_mouse  ; each experiment: trials x neurons    
    med_peak_amp_eachN = []
    iqr_peak_amp_eachN = []
    for ipl in range(len(a)):
        if np.sum(~np.isnan(a[ipl])) > 0: # experiment is valid
            m = np.mean(a[ipl], axis=0)
            iq = st.iqr(a[ipl], axis=0)
        else:
            m = np.array([np.nan])
            iq = np.array([np.nan])
        med_peak_amp_eachN.append(m)
        iqr_peak_amp_eachN.append(iq)
#    med_peak_amp_eachN = [np.median(a[ipl], axis=0) for ipl in range(len(a))] # 8 * number_of_session_for_the_mouse  ; each experiment: neurons
#    iqr_peak_amp_eachN = [st.iqr(a[ipl], axis=0) for ipl in range(len(a))] # the first 8 are for the first session, the next 8 are for the next session..

    # distribution of peak amplitudes across trials is very skewed (not normal)
    '''
    if doPlots:
    #    plt.figure(figsize=(4,20))
        plt.suptitle('mouse %d\npeak amp, each N' %(mouse_id), fontsize=15, y=1)
        for ipl in range(8):
    #        ax = plt.subplot(2,4,ipl+1)
    #        ax = plt.subplot(8,1,ipl+1)
            plt.figure()
            ax = plt.gca()
            for ineu in range(a[ipl].shape[1]):
                plt.hist(a[ipl][:,ineu]) #, alpha=.5)
                ax.tick_params(labelsize=10)
            plt.subplots_adjust(wspace=.4, hspace=.5)
            plt.title('plane %d' %ipl, fontsize=13)
            if ipl==0:
                plt.xlabel('peak amplitude', fontsize=12)
                plt.ylabel('# trials', fontsize=12)
    '''


    #### peak timing, each neuron ####
    # For each neuron, compute median and iqr of peak amplitude across all trial (each experiment)
    a = all_sess_2an_this_mouse['peak_timing_trs_ns'].values # 8 * number_of_session_for_the_mouse  ; each experiment: trials x neurons
    med_peak_timing_eachN = []
    iqr_peak_timing_eachN = []
    for ipl in range(len(a)):
        if np.sum(~np.isnan(a[ipl])) > 0: # experiment is valid
            m = np.mean(a[ipl], axis=0)
            iq = st.iqr(a[ipl], axis=0)
        else:
            m = np.array([np.nan])
            iq = np.array([np.nan])
        med_peak_timing_eachN.append(m)
        iqr_peak_timing_eachN.append(iq)
#    med_peak_timing_eachN = [np.median(a[ipl], axis=0) for ipl in range(len(a))] # 8 * number_of_session_for_the_mouse  ; each experiment: neurons
#    iqr_peak_timing_eachN = [st.iqr(a[ipl], axis=0) for ipl in range(len(a))] # the first 8 are for the first session, the next 8 are for the next session..

    # distribution of peak timing across trials is kind of uniform ... or sometimes high on the smallest and largest values (not normal)
    '''
    if doPlots:
        plt.figure(figsize=(8,6))
        plt.suptitle('mouse %d\npeak time, each N' %(mouse_id), fontsize=15, y=1)
        for ipl in range(8):
            ax = plt.subplot(2,4,ipl+1)
            for ineu in range(a[ipl].shape[1]):
                plt.hist(a[ipl][:,ineu]) #, alpha=.5)
                ax.tick_params(labelsize=10)
            plt.subplots_adjust(wspace=.4, hspace=.5)
            plt.title('plane %d' %ipl, fontsize=13)
            if ipl==0:
                plt.xlabel('peak frame rel. omit (~10Hz)', fontsize=12)
                plt.ylabel('# trials', fontsize=12)

    
    # distribution of peak timing across trials is kind of uniform ... or sometimes high on the smallest and largest values (not normal)
    
    if doPlots:
        plt.figure(figsize=(8,6))
        plt.suptitle('mouse %d\npeak time, each N' %(mouse_id), fontsize=15, y=1)
        for ipl in range(8):
            ax = plt.subplot(2,4,ipl+1)
            for ineu in range(a[ipl].shape[1]):
                plt.hist(a[ipl][:,ineu]) #, alpha=.5)
                ax.tick_params(labelsize=10)
            plt.subplots_adjust(wspace=.4, hspace=.5)
            plt.title('plane %d' %ipl, fontsize=13)
            if ipl==0:
                plt.xlabel('peak frame rel. omit (~10Hz)', fontsize=12)
                plt.ylabel('# trials', fontsize=12)
    '''
    
    
    trace_peak_med_iqr_0.at[im, ['mouse_id', 'med_peak_eachTr', 'iqr_peak_eachTr', 'med_peak_amp_eachN', 'iqr_peak_amp_eachN', \
                                        'med_peak_timing_eachN', 'iqr_peak_timing_eachN', 'med_trace', 'iqr_trace']] = mouse_id, med_peak_eachTr, iqr_peak_eachTr, med_peak_amp_eachN, iqr_peak_amp_eachN, med_peak_timing_eachN, iqr_peak_timing_eachN, med_trace, iqr_trace
            

        
##################################################################################################
##################################################################################################        
#%% Set pandas dataframe "trace_peak_allMice" that includes traces and peak measures for all planes of all sessions, for each mouse
# size of each variable is: (8*num_sessions)
##################################################################################################
##################################################################################################
    
cols = ['mouse_id', 'cre', 'cre_exp', 'session_ids', 'experiment_ids', 'session_stages', 'session_labs', 'area', 'depth', 'plane', 'n_neurons', 'n_omissions', 'trace', 'peak_amp', 'peak_timing', 'peak_amp_flash', 'peak_timing_flash']
trace_peak_allMice = pd.DataFrame([], columns = cols)

for im in range(len(all_mice_id)): # im=0
    
    ##%%
    mouse_id = all_mice_id[im]

    if sum((all_sess_2an['mouse_id']==mouse_id).values) > 0: # make sure there is data for this mouse
        print(f'There is data for mouse {im}')

        all_sess_2an_this_mouse = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]
        peak_trace_this_mouse = trace_peak_med_iqr_0[trace_peak_med_iqr_0['mouse_id']==mouse_id]
        
        cre = all_sess_2an_this_mouse['cre'].iloc[0]
        cre = cre[:cre.find('-')] # remove the IRES-Cre part

        session_ids = all_sess_2an_this_mouse['session_id'].values[np.arange(0, all_sess_2an_this_mouse.shape[0], num_planes)]
        num_sessions = len(session_ids)
        
        experiment_ids = all_sess_2an_this_mouse['experiment_id'].values # (8*num_sessions)
#         experiment_ids = np.reshape(experiment_ids, (num_planes, num_sessions)) # 8 x num_sessions
        
        session_stages = all_sess_2an_this_mouse['stage'].values[np.arange(0, all_sess_2an_this_mouse.shape[0], num_planes)]
#        session_stages = mouse_trainHist_all[mouse_trainHist_all['mouse_id']==mouse_id]['stage'].values
#         num_sessions = len(session_stages)
        
        areas = all_sess_2an_this_mouse['area'] # (8*num_sessions)
        depths = all_sess_2an_this_mouse['depth'] # (8*num_sessions)
        planes = all_sess_2an_this_mouse.index.values # (8*num_sessions)
#         print(len(depths))

        #### replicate cre to the number of sessions ####
        cre_exp = np.full((num_planes*num_sessions), cre) # (8*num_sessions)
        
        n_neurons = all_sess_2an_this_mouse['n_neurons'].values # (8*num_sessions)
        n_omissions = all_sess_2an_this_mouse['n_omissions'].values # (8*num_sessions)        
        
        
        ######## Traces ########
        y = np.array(np.squeeze(peak_trace_this_mouse['med_peak_eachTr'])) # 8 x trials: if the mouse has only 1 session; # if more than one session: size: total number of planes (8*num_sessions), and then each element has size (trials,) 
        y_trace = np.array(np.squeeze(peak_trace_this_mouse['med_trace'])) # total number of planes (8*num_sessions) x time 
        
        # turn to nan those planes with fewer than th_neurons neurons
        y[n_neurons<th_neurons] = np.full((y[n_neurons<th_neurons].shape), np.nan)
        y_trace[n_neurons<th_neurons] = np.full((y_trace[n_neurons<th_neurons].shape), np.nan)
        
        ######## Get the trial-median peak data for each neuron (peak computed on individual trials, then, median computed across trials)
        # remember: the first 8 are for the first session, the next 8 are for the next session..
        '''
        peak_amp_ns = np.array(np.squeeze(peak_trace_this_mouse['med_peak_amp_eachN'])) # size: total number of planes (8*num_sessions), and then each element has size (neurons,) 
        peak_timing_ns = np.array(np.squeeze(peak_trace_this_mouse['med_peak_timing_eachN']))
    
        # for each session, compute med and iqr across neurons  # or plot hist of peak of all sessions    
        peak_amp_med_ns, peak_amp_25q_ns, peak_amp_75q_ns = med_perc_each_exp(peak_amp_ns) # total number of planes (8*num_sessions); each element: med of peak amp across neurons in that plane
        peak_timing_med_ns, peak_timing_25q_ns, peak_timing_75q_ns = med_perc_each_exp(peak_timing_ns) # total number of planes (8*num_sessions); each element: med of peak timing across neurons in that plane
        '''
        
        
        ######## Peak amplitude, computed on trial-median *traces* ########    # get the peak measures computed on trial-median traces (above is computed on individual trials, but then median is taken across trials; the measure here is less noisy as the peaks are computed on the meidan of the traces.)
        peak_amp_ns_traceMed = all_sess_2an_this_mouse['peak_amp_eachN_traceMed'].values # size: total number of planes (8*num_sessions), and then each element has size (neurons,) 
        peak_timing_ns_traceMed = all_sess_2an_this_mouse['peak_timing_eachN_traceMed'].values
    
        # for each session, compute med and iqr across neurons  # or plot hist of peak of all sessions    
        peak_amp_med_ns_traceMed, peak_amp_25q_ns_traceMed, peak_amp_75q_ns_traceMed = med_perc_each_exp(peak_amp_ns_traceMed) # total number of planes (8*num_sessions); each element: med of peak amp across neurons in that plane
        peak_timing_med_ns_traceMed, peak_timing_25q_ns_traceMed, peak_timing_75q_ns_traceMed = med_perc_each_exp(peak_timing_ns_traceMed*frame_dur) # total number of planes (8*num_sessions); each element: med of peak timing across neurons in that plane

        # turn to nan those planes with fewer than th_neurons neurons
        peak_amp_med_ns_traceMed[n_neurons<th_neurons] = np.full((peak_amp_med_ns_traceMed[n_neurons<th_neurons].shape), np.nan)
        peak_timing_med_ns_traceMed[n_neurons<th_neurons] = np.full((peak_timing_med_ns_traceMed[n_neurons<th_neurons].shape), np.nan)

        
        ######## Image-evoked responses: Peak amplitude, computed on trial-median *traces* ########    # get the peak measures computed on trial-median traces (above is computed on individual trials, but then median is taken across trials; the measure here is less noisy as the peaks are computed on the meidan of the traces.)
        peak_amp_ns_traceMed_flash = all_sess_2an_this_mouse['peak_amp_eachN_traceMed_flash'].values # size: total number of planes (8*num_sessions), and then each element has size (neurons,) 
        peak_timing_ns_traceMed_flash = all_sess_2an_this_mouse['peak_timing_eachN_traceMed_flash'].values
    
        # for each session, compute med and iqr across neurons  # or plot hist of peak of all sessions    
        peak_amp_med_ns_traceMed_flash, peak_amp_25q_ns_traceMed_flash, peak_amp_75q_ns_traceMed_flash = med_perc_each_exp(peak_amp_ns_traceMed_flash) # total number of planes (8*num_sessions); each element: med of peak amp across neurons in that plane
        peak_timing_med_ns_traceMed_flash, peak_timing_25q_ns_traceMed_flash, peak_timing_75q_ns_traceMed_flash = med_perc_each_exp(peak_timing_ns_traceMed_flash*frame_dur) # total number of planes (8*num_sessions); each element: med of peak timing across neurons in that plane
                
        # turn to nan those planes with fewer than th_neurons neurons
        peak_amp_med_ns_traceMed_flash[n_neurons<th_neurons] = np.full((peak_amp_med_ns_traceMed_flash[n_neurons<th_neurons].shape), np.nan)
        peak_timing_med_ns_traceMed_flash[n_neurons<th_neurons] = np.full((peak_timing_med_ns_traceMed_flash[n_neurons<th_neurons].shape), np.nan)

        
        ######## Set session labels (stage names are too long)
    #    session_beg_inds = np.concatenate(([0], np.cumsum(num_trs_each_sess[:-1])+1))
        session_labs = []
        for ibeg in range(num_sessions):
            a = str(session_stages[ibeg]); 
            en = np.argwhere([a[ich]=='_' for ich in range(len(a))])[-1].squeeze() + 6
            txt = str(session_stages[ibeg][15:]) # session_stages[ibeg][8:en]
            txt = txt[0:min(3, len(txt))]
            session_labs.append(txt)
        
#         print(np.sum(np.isnan(peak_amp_med_ns_traceMed)), np.sum(np.isnan(peak_amp_med_ns_traceMed_flash)))    

        #######%% Keep vars
        trace_peak_allMice.at[im, ['mouse_id', 'cre', 'cre_exp', 'session_ids', 'experiment_ids', 'session_stages', 'session_labs']] = mouse_id, cre, cre_exp, session_ids, experiment_ids, session_stages, session_labs
        trace_peak_allMice.at[im, ['area', 'depth', 'plane']] = areas.values, depths.values, planes
        trace_peak_allMice.at[im, ['n_neurons', 'n_omissions']] = n_neurons, n_omissions
        trace_peak_allMice.at[im, 'trace'] = y_trace
        trace_peak_allMice.at[im, 'peak_amp'] = peak_amp_med_ns_traceMed
        trace_peak_allMice.at[im, 'peak_timing'] = peak_timing_med_ns_traceMed
        trace_peak_allMice.at[im, 'peak_amp_flash'] = peak_amp_med_ns_traceMed_flash
        trace_peak_allMice.at[im, 'peak_timing_flash'] = peak_timing_med_ns_traceMed_flash

        
#%% Remember each row of trace_peak_allMice is for each mouse        
print(len(trace_peak_allMice)) 

if all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1 == 1: 
    session_labs = trace_peak_allMice.iloc[0]['session_labs'] # take it from the 1st mouse, this info is the same for all mice
    num_sessions = len(session_labs) #np.shape(trace_peak_allMice['session_stages'].iloc[icre])[0]
else:
    a = trace_peak_allMice['plane'].values
    num_sess_per_mouse = np.array([len(a[i])/8 for i in range(len(a))]) # number of sessions for each mouse
    print(num_sess_per_mouse, sum(num_sess_per_mouse), sum(num_sess_per_mouse)*8, len(all_sess_2an))
    
#trace_peak_allMice.columns
print(trace_peak_allMice.iloc[0])


##%% Pool data across layers ... also across areas , for each mouse; keep vars in a panda dataframe#
# Compare results across areas, also across depths (layers) #######################################
#
# Set pooled_trace_peak_allMice, which is similar to what we set above (trace_peak_allMice) #######
# but it includes vars that are pooled across layers or areas ########
###################################################################################################
###################################################################################################

cols = ['cre_pooled_eachArea', 'distinct_areas', 'trace_pooled_eachArea', 'peak_amp_pooled_eachArea', 'peak_timing_pooled_eachArea', 'peak_amp_flash_pooled_eachArea', 'peak_timing_flash_pooled_eachArea',\
        'cre_pooled_eachDepth', 'depth_pooled_eachDepth', 'trace_pooled_eachDepth', 'peak_amp_pooled_eachDepth', 'peak_timing_pooled_eachDepth', 'peak_amp_flash_pooled_eachDepth', 'peak_timing_flash_pooled_eachDepth']
pooled_trace_peak_allMice = pd.DataFrame([], columns=cols)

for im in range(len(trace_peak_allMice)): # im = 0

    ########## For each area, pool data across sessions and layers ##########    
    t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time
    pa = trace_peak_allMice.iloc[im]['peak_amp'] # (8*num_sessions)
    pt = trace_peak_allMice.iloc[im]['peak_timing'] # (8*num_sessions)
    paf = trace_peak_allMice.iloc[im]['peak_amp_flash'] # (8*num_sessions)
    ptf = trace_peak_allMice.iloc[im]['peak_timing_flash'] # (8*num_sessions)
    c = trace_peak_allMice.iloc[im]['cre_exp'] # (8*num_sessions)

    area = trace_peak_allMice.iloc[im]['area'] # (8*num_sessions)
    depth = trace_peak_allMice.iloc[im]['depth']
    
    trace_pooled_sesss_planes_eachArea, distinct_areas, i_areas = pool_sesss_planes_eachArea(area, t) # 2 x (4 x num_sessions) x nFrames_upsampled
    peak_amp_pooled_sesss_planes_eachArea,_,_ = pool_sesss_planes_eachArea(area, pa) # 2 x (4 x num_sessions) 
    peak_timing_pooled_sesss_planes_eachArea,_,_ = pool_sesss_planes_eachArea(area, pt) # 2 x (4 x num_sessions) 
    peak_amp_flash_pooled_sesss_planes_eachArea,_,_ = pool_sesss_planes_eachArea(area, paf) # 2 x (4 x num_sessions) 
    peak_timing_flash_pooled_sesss_planes_eachArea,_,_ = pool_sesss_planes_eachArea(area, ptf) # 2 x (4 x num_sessions) 
    cre_pooled_sesss_planes_eachArea,_,_ = pool_sesss_planes_eachArea(area, c) # 2 x (4 x num_sessions) 

    
    ########## For each layer (depth), pool data across sessions and areas ##########    
    planes_allsess = trace_peak_allMice.iloc[im]['plane'] # (8*num_sessions)
    
    depth1_area2 = np.argwhere(i_areas).squeeze()[0] # 4
    if depth1_area2!=4:
        sys.exit('Because the 8 planes are in the following order: area 1 (4 planes), then area 2 (another 4 planes), we expect depth1_area2 to be 4! What is wrong?!')
    
    trace_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2
    peak_amp_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, pa, depth1_area2) # 4 x (2 x num_sess)
    peak_timing_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, pt, depth1_area2) # 4 x (2 x num_sess)
    peak_amp_flash_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, paf, depth1_area2) # 4 x (2 x num_sess)
    peak_timing_flash_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, ptf, depth1_area2) # 4 x (2 x num_sess)
    cre_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, c, depth1_area2) # 4 x (2 x num_sess)
    depth_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, depth, num_depth=4) # 4 x (2 x num_sessions) # each row is for a depth and shows values from the two areas of all the sessions
    
    #### collect values for each mouse
    pooled_trace_peak_allMice.at[im,:] = cre_pooled_sesss_planes_eachArea, distinct_areas, trace_pooled_sesss_planes_eachArea, peak_amp_pooled_sesss_planes_eachArea, peak_timing_pooled_sesss_planes_eachArea, peak_amp_flash_pooled_sesss_planes_eachArea, peak_timing_flash_pooled_sesss_planes_eachArea,\
        cre_pooled_sesss_areas_eachDepth, depth_pooled_sesss_areas_eachDepth, trace_pooled_sesss_areas_eachDepth, peak_amp_pooled_sesss_areas_eachDepth, peak_timing_pooled_sesss_areas_eachDepth, peak_amp_flash_pooled_sesss_areas_eachDepth, peak_timing_flash_pooled_sesss_areas_eachDepth
    

    
    
    
    
    
##########################################################################################
##########################################################################################
##########################################################################################
#%% ##########################################################################################
############### Set vars to make average plots across mice of the same cell line #############
################ (each plane individually, no pooling across areas or layers) ###########################
##############################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
# Remember each row of trace_peak_allMice is for each mouse; below we combine data from all mice of the same cre line            
##%% Average vars across mice for each cre line (done for each plane separately)


#%% Take average across mice of the same cre line, do it for all the columns in trace_peak_allMice
        
if all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==1: # take average across mice for A sessions, and B sessions, separately.
    
    # groupby below doesnt work in jupyterlab (apparently due to memory issues)!!
    '''       
    cre_group_ave = trace_peak_allMice.groupby('cre').apply(lambda x: np.mean(x, axis=0)) # np.mean # for this to work, all mice need to have the same number of sessions.
    print(cre_group_ave)
    print(cre_group_ave.columns)
    
    
    all_cres = cre_group_ave.index.values
    num_each_cre = trace_peak_allMice.groupby('cre').apply(len)
    #print(all_cres)
    print(num_each_cre)
    #print(cre_group_ave)
    print([cre_group_ave.iloc[0]['trace'].shape, cre_group_ave.iloc[0]['depth'].shape ,cre_group_ave.iloc[0]['peak_amp'].shape, cre_group_ave.iloc[0]['peak_timing'].shape])
    
    
    ############ set the traces averaged across mice for each cre line. Done for each plane separately (and each session) ############
    # we take advantage of pandas groupby (average function) that we made above
    depth_allCre = []
    trace_ave_allCre = []
    peak_amp_omit_ave_allCre = []
    peak_timing_omit_ave_allCre = []
    peak_amp_flash_ave_allCre = []
    peak_timing_flash_ave_allCre = []
    
    for cre_now in all_cres: # cre_now = all_cres[0]
        # average depth across mice (for each plane and each session)
        d = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['depth'].values).squeeze() # (num_sess*num_planes)
        # average trace across mice (for each plane and each session)
        # each row of t is one plane (of one session), and average across mice that are in the same cre group
        t = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['trace'].values) # (num_sess*num_planes) x numFrames        
        pa = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['peak_amp'].values).squeeze() # (num_sess*num_planes)
        pt = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['peak_timing'].values).squeeze() # (num_sess*num_planes)
        paf = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['peak_amp_flash'].values).squeeze() # (num_sess*num_planes)
        ptf = np.vstack(cre_group_ave[cre_group_ave.index==cre_now]['peak_timing_flash'].values).squeeze() # (num_sess*num_planes)
        
        depth_allCre.append(d)
        trace_ave_allCre.append(t)    
        peak_amp_omit_ave_allCre.append(pa)    
        peak_timing_omit_ave_allCre.append(pt)    
        peak_amp_flash_ave_allCre.append(paf)    
        peak_timing_flash_ave_allCre.append(ptf)    
        
    depth_allCre = np.array(depth_allCre) # num_cre x (num_sess*num_planes)
    trace_ave_allCre = np.array(trace_ave_allCre) # num_cre x (num_sess*num_planes) x num_frs
    peak_amp_omit_ave_allCre = np.array(peak_amp_omit_ave_allCre) # num_cre x (num_sess*num_planes)
    peak_timing_omit_ave_allCre = np.array(peak_timing_omit_ave_allCre) # num_cre x (num_sess*num_planes)
    peak_amp_flash_ave_allCre = np.array(peak_amp_flash_ave_allCre) # num_cre x (num_sess*num_planes)
    peak_timing_flash_ave_allCre = np.array(peak_timing_flash_ave_allCre) # num_cre x (num_sess*num_planes)
    #depth_all = np.reshape(depth_allCre, (len(all_cres), num_planes, num_sessions), order='F') # num_cre x num_planes x num_sess
    '''    

    # if groupby (above) works, uncomment all below (the average computing section), except for the line below that defines "cres".
    cres = trace_peak_allMice['cre'].values
    all_cres = np.unique(cres) # cre_unq
    num_each_cre = np.array([np.sum(cres==all_cres[i]) for i in range(len(all_cres))])

    
    #%% Set the average and standard error traces: across mice for each cre line. Done for each plane separately (and each session) ############
    # (use this if the pandas groupby above doesnt work)

    depth_ave_allCre = []    
    trace_ave_allCre = []
    peak_amp_omit_ave_allCre = []
    peak_timing_omit_ave_allCre = []
    peak_amp_flash_ave_allCre = []
    peak_timing_flash_ave_allCre = []
    peak_amp_mod_omit_ave_allCre = [] # relative modulation in response amplitude and timing from A to B 
    peak_timing_mod_omit_ave_allCre = []
    peak_amp_mod_flash_ave_allCre = []
    peak_timing_mod_flash_ave_allCre = []
    
    depth_sd_allCre = []
    trace_sd_allCre = []
    peak_amp_omit_sd_allCre = []
    peak_timing_omit_sd_allCre = []
    peak_amp_flash_sd_allCre = []
    peak_timing_flash_sd_allCre = []    
    peak_amp_mod_omit_sd_allCre = []
    peak_timing_mod_omit_sd_allCre = []
    peak_amp_mod_flash_sd_allCre = []
    peak_timing_mod_flash_sd_allCre = []
    
    # loop through each cre line
    for icre in range(len(all_cres)): # icre = 0
        cre_now = all_cres[icre] # because we are indexing all_cres, trace_ave_allCre and the vars computed above (depth_ave_allCre, and trace_ave_allCre) will have the same order of cre lines.
        cre_i = np.in1d(cres, cre_now) #cre_unq[icre]) # group cre lines manually

        ###### depth ######
        aa = trace_peak_allMice.iloc[cre_i]['depth'].values 
#         aa.shape # num_mice_this_cre        
        y_thisCre_allMice = np.vstack(aa).T # data from the mice are concatenated vertically: the 1st 16 rows are for one mouse, and the next 16 rows are for the other mouse
#         y_thisCre_allMice.shape # (num_sessions*num_planes) x num_mice_this_cre    
        # compute std across mice (of the same cre line) for each plane
        d = np.nanmean(y_thisCre_allMice.astype(float), axis=1).squeeze()  # (num_sessions*num_planes)
        ds = np.nanstd(y_thisCre_allMice.astype(float), axis=1).squeeze() / np.sqrt(sum(cre_i))  # (8 x num_sessions)
    
    
        ###### trace ######
        aa = trace_peak_allMice.iloc[cre_i]['trace'].values 
    #    aa.shape # num_mice_this_cre
        trace_thisCre_allMice0 = np.vstack(aa) # data from the mice are concatenated vertically: the 1st 16 rows are for one mouse, and the next 16 rows are for the other mouse
    #    trace_thisCre_allMice0.shape # (num_sessions*num_planes* num_mice_this_cre) x num_frs # 32x80 
        trace_thisCre_allMice = np.reshape(trace_thisCre_allMice0, (num_sessions*num_planes, sum(cre_i), \
                trace_thisCre_allMice0.shape[1]), order='F') # 16x2x80  # (num_sessions*num_planes)* num_mice_this_cre x num_frs 
        
        # compute mean across mice (of the same cre line) for each plane     
        t = np.nanmean(trace_thisCre_allMice, axis=1)  # (num_sessions*num_planes) x num_frs    
        # compute std across mice (of the same cre line) for each plane     
        ts = np.nanstd(trace_thisCre_allMice, axis=1) / np.sqrt(sum(cre_i)) # (num_sessions*num_planes) x num_frs    
        
        
        # below: check to make sure reshaping worked the way it should!!!!
    #    example_plane = 7
    #    a = trace_thisCre_allMice0[np.arange(example_plane, num_sessions*num_planes*sum(cre_i), num_sessions*num_planes),:]
    #    a.shape
    #    b = np.nanmean(a, axis=0) 
    #    np.equal(b, trace_ave[example_plane])

    
        ###### peak amp ######
        aa = trace_peak_allMice.iloc[cre_i]['peak_amp'].values 
    #    aa.shape # num_mice_this_cre
        y_thisCre_allMice = np.vstack(aa).T # data from the mice are concatenated vertically: the 1st 16 rows are for one mouse, and the next 16 rows are for the other mouse
    #    y_thisCre_allMice.shape # (num_sessions*num_planes) x num_mice_this_cre    
        # compute mean across mice (of the same cre line) for each plane     
        pa = np.nanmean(y_thisCre_allMice, axis=1).squeeze()  # (num_sessions*num_planes)
        pas = np.nanstd(y_thisCre_allMice, axis=1).squeeze() / np.sqrt(sum(cre_i)) # (num_sessions*num_planes)

        # compute relative change in resp amp from A to B sessions.
        As = y_thisCre_allMice[:num_planes]
        Bs = y_thisCre_allMice[num_planes:]
        ab_mod = (Bs - As) / (Bs + As) # 8 x num_mice_this_cre  
        # compute mean across mice (of the same cre line) for each plane     
        pa_mod = np.nanmean(ab_mod, axis=1) # 8 
        pa_mods = np.nanmean(ab_mod, axis=1) / np.sqrt(sum(cre_i)) # 8 
        
    
        ###### peak timing ######
        aa = trace_peak_allMice.iloc[cre_i]['peak_timing'].values # aa.shape # num_mice_this_cre
        y_thisCre_allMice = np.vstack(aa).T # data from the mice are concatenated vertically: the 1st 16 rows are for one mouse, and the next 16 rows are for the other mouse  # y_thisCre_allMice.shape # (num_sessions*num_planes) x num_mice_this_cre    
        pt = np.nanmean(y_thisCre_allMice, axis=1).squeeze()  # (num_sessions*num_planes)     # compute std across mice (of the same cre line) for each plane
        pts = np.nanstd(y_thisCre_allMice, axis=1).squeeze() / np.sqrt(sum(cre_i)) # (num_sessions*num_planes)     # compute std across mice (of the same cre line) for each plane
    
        # compute relative change in resp amp from A to B sessions.
        As = y_thisCre_allMice[:num_planes]
        Bs = y_thisCre_allMice[num_planes:]
        ab_mod = (Bs - As) / (Bs + As) # 8 x num_mice_this_cre  
        # compute mean across mice (of the same cre line) for each plane     
        pt_mod = np.nanmean(ab_mod, axis=1)  # 8 
        pt_mods = np.nanmean(ab_mod, axis=1) / np.sqrt(sum(cre_i)) # 8 

        
        ###### peak amp flash ######
        aa = trace_peak_allMice.iloc[cre_i]['peak_amp_flash'].values # aa.shape # num_mice_this_cre
        y_thisCre_allMice = np.vstack(aa).T # data from the mice are concatenated vertically: the 1st 16 rows are for one mouse, and the next 16 rows are for the other mouse  # y_thisCre_allMice.shape # (num_sessions*num_planes) x num_mice_this_cre    
        paf = np.nanmean(y_thisCre_allMice, axis=1).squeeze()  # (num_sessions*num_planes)     # compute std across mice (of the same cre line) for each plane
        pafs = np.nanstd(y_thisCre_allMice, axis=1).squeeze() / np.sqrt(sum(cre_i)) # (num_sessions*num_planes)     # compute std across mice (of the same cre line) for each plane

        # compute relative change in resp amp from A to B sessions.
        As = y_thisCre_allMice[:num_planes]
        Bs = y_thisCre_allMice[num_planes:]
        ab_mod = (Bs - As) / (Bs + As) # 8 x num_mice_this_cre  
        # compute mean across mice (of the same cre line) for each plane     
        paf_mod = np.nanmean(ab_mod, axis=1)  # 8 
        paf_mods = np.nanmean(ab_mod, axis=1) / np.sqrt(sum(cre_i)) # 8 

        
        ###### peak timing flash ######
        aa = trace_peak_allMice.iloc[cre_i]['peak_timing_flash'].values # aa.shape # num_mice_this_cre
        y_thisCre_allMice = np.vstack(aa).T # data from the mice are concatenated vertically: the 1st 16 rows are for one mouse, and the next 16 rows are for the other mouse  # y_thisCre_allMice.shape # (num_sessions*num_planes) x num_mice_this_cre    
        ptf = np.nanmean(y_thisCre_allMice, axis=1).squeeze()  # (num_sessions*num_planes)     # compute std across mice (of the same cre line) for each plane
        ptfs = np.nanstd(y_thisCre_allMice, axis=1).squeeze() / np.sqrt(sum(cre_i)) # (num_sessions*num_planes)     # compute std across mice (of the same cre line) for each plane    

        # compute relative change in resp amp from A to B sessions.
        As = y_thisCre_allMice[:num_planes]
        Bs = y_thisCre_allMice[num_planes:]
        ab_mod = (Bs - As) / (Bs + As) # 8 x num_mice_this_cre  
        # compute mean across mice (of the same cre line) for each plane     
        ptf_mod = np.nanmean(ab_mod, axis=1)  # 8 
        ptf_mods = np.nanmean(ab_mod, axis=1) / np.sqrt(sum(cre_i)) # 8         

        
        
        depth_ave_allCre.append(d)     
        trace_ave_allCre.append(t)    
        peak_amp_omit_ave_allCre.append(pa)    
        peak_timing_omit_ave_allCre.append(pt)    
        peak_amp_flash_ave_allCre.append(paf)    
        peak_timing_flash_ave_allCre.append(ptf)    

        peak_amp_mod_omit_ave_allCre.append(pa_mod)    
        peak_timing_mod_omit_ave_allCre.append(pt_mod)    
        peak_amp_mod_flash_ave_allCre.append(paf_mod)    
        peak_timing_mod_flash_ave_allCre.append(ptf_mod)
        
        depth_sd_allCre.append(ds)
        trace_sd_allCre.append(ts)    
        peak_amp_omit_sd_allCre.append(pas)    
        peak_timing_omit_sd_allCre.append(pts)    
        peak_amp_flash_sd_allCre.append(pafs)    
        peak_timing_flash_sd_allCre.append(ptfs)    

        peak_amp_mod_omit_sd_allCre.append(pa_mods)    
        peak_timing_mod_omit_sd_allCre.append(pt_mods)    
        peak_amp_mod_flash_sd_allCre.append(paf_mods)    
        peak_timing_mod_flash_sd_allCre.append(ptf_mods)

        
        
    depth_ave_allCre = np.array(depth_ave_allCre) # num_cre x (num_sess*num_planes) x num_frs        
    trace_ave_allCre = np.array(trace_ave_allCre) # num_cre x (num_sess*num_planes) x num_frs
    peak_amp_omit_ave_allCre = np.array(peak_amp_omit_ave_allCre) # num_cre x (num_sess*num_planes)
    peak_timing_omit_ave_allCre = np.array(peak_timing_omit_ave_allCre) # num_cre x (num_sess*num_planes)
    peak_amp_flash_ave_allCre = np.array(peak_amp_flash_ave_allCre) # num_cre x (num_sess*num_planes)
    peak_timing_flash_ave_allCre = np.array(peak_timing_flash_ave_allCre) # num_cre x (num_sess*num_planes)
    
    peak_amp_mod_omit_ave_allCre = np.array(peak_amp_mod_omit_ave_allCre) # num_cre x 8
    peak_timing_mod_omit_ave_allCre = np.array(peak_timing_mod_omit_ave_allCre)    
    peak_amp_mod_flash_ave_allCre = np.array(peak_amp_mod_flash_ave_allCre)    
    peak_timing_mod_flash_ave_allCre = np.array(peak_timing_mod_flash_ave_allCre)

    depth_sd_allCre = np.array(depth_sd_allCre) # num_cre x (8 x num_sessions)    
    trace_sd_allCre = np.array(trace_sd_allCre) # num_cre x (num_sess*num_planes) x num_frs
    peak_amp_omit_sd_allCre = np.array(peak_amp_omit_sd_allCre) # num_cre x (num_sess*num_planes)
    peak_timing_omit_sd_allCre = np.array(peak_timing_omit_sd_allCre) # num_cre x (num_sess*num_planes)
    peak_amp_flash_sd_allCre = np.array(peak_amp_flash_sd_allCre) # num_cre x (num_sess*num_planes)
    peak_timing_flash_sd_allCre = np.array(peak_timing_flash_sd_allCre) # num_cre x (num_sess*num_planes)
    
    peak_amp_mod_omit_sd_allCre = np.array(peak_amp_mod_omit_sd_allCre) # num_cre x 8         
    peak_timing_mod_omit_sd_allCre = np.array(peak_timing_mod_omit_sd_allCre)    
    peak_amp_mod_flash_sd_allCre = np.array(peak_amp_mod_flash_sd_allCre)    
    peak_timing_mod_flash_sd_allCre = np.array(peak_timing_mod_flash_sd_allCre)    
        

    # Compute standard error across days
    #g_se = trace_peak_allMice.loc[:,['cre','trace']].groupby('cre').agg(np.std, ddof=1) # apply(lambda x: np.std(x, axis=0)) # 
    #/np.sqrt(num_each_cre)
    
    
    #%% Interpolate the traces every 3ms (instead of the original one that is 93ms)
    # (it doesnt really help ... as we are getting the time trace based on the frames of omittion aligned traces; also the flash/gray times are coming based on that... so it is nothing like we are adding more time resolution by interpolating!)
    
    kind = 'linear' #'quadratic' #     
    #time_trace, time_trace_new = upsample_time_imaging(samps_bef, samps_aft, 31.) # set time for the interpolated traces (every 3ms time points)
    trace_ave_allCre_new = np.full((len(all_cres), trace_ave_allCre.shape[1], len(time_trace_new)), np.nan)
    trace_sd_allCre_new = np.full((len(all_cres), trace_ave_allCre.shape[1], len(time_trace_new)), np.nan)
    
    for icre in range(len(all_cres)):
        for ips in range(trace_ave_allCre.shape[1]):
            trace_ave_allCre_new[icre, ips], _, _ = interp_imaging(trace_ave_allCre[icre, ips], samps_bef, samps_aft, kind)
            trace_sd_allCre_new[icre, ips], _, _ = interp_imaging(trace_sd_allCre[icre, ips], samps_bef, samps_aft, kind)
              
                
                
                
                

################################################################################                
################################################################################
################################################################################
################################################################################                
################################################################################
################################################################################
#%%
# else: # not A-B transition

#%% Pool all sessions of all mice
# pandas series "trace_peak_allMice_sessPooled" is created that includes all the important vars below.
# use this to make summary mouse plots, by averaging across pooled sessions of all mice.

a0 = np.concatenate((trace_peak_allMice['cre_exp']).values) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
cre_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        

a0 = np.concatenate((trace_peak_allMice['area']).values) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
area_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        

a0 = np.concatenate((trace_peak_allMice['depth']).values) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
depth_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        

session_labs_all = np.unique(np.concatenate((trace_peak_allMice['session_labs']).values))

########## individual planes ##########

a0 = np.concatenate((trace_peak_allMice['trace'].values)) # (8*sum(num_sess_per_mouse)) x 80 # 8 planes of 1 session, then 8 planes of next session, and so on 
t_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes), a0.shape[-1]), order='F') # 8 x sum(num_sess_per_mouse) x 80 # each row is one plane, all sessions        
#        aa = np.reshape(np.concatenate((trace_peak_allMice['plane'].values)), (num_planes, int(a0.shape[0]/num_planes)), order='F')

a0 = np.concatenate((trace_peak_allMice['peak_amp']).values) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
pa_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        

a0 = np.concatenate((trace_peak_allMice['peak_timing']).values) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
pt_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        

a0 = np.concatenate((trace_peak_allMice['peak_amp_flash']).values) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
paf_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        

a0 = np.concatenate((trace_peak_allMice['peak_timing_flash']).values) # (8*sum(num_sess_per_mouse)) # 8 planes of 1 session, then 8 planes of next session, and so on 
ptf_all = np.reshape(a0, (num_planes, int(a0.shape[0]/num_planes)), order='F') # 8 x sum(num_sess_per_mouse) # each row is one plane, all sessions        


########## Each area: layers pooled ########## 

cre_all_eachArea = np.concatenate((pooled_trace_peak_allMice['cre_pooled_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse))
t_all_eachArea = np.concatenate((pooled_trace_peak_allMice['trace_pooled_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse)) x 80
pa_all_eachArea = np.concatenate((pooled_trace_peak_allMice['peak_amp_pooled_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse))
pt_all_eachArea = np.concatenate((pooled_trace_peak_allMice['peak_timing_pooled_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse))
paf_all_eachArea = np.concatenate((pooled_trace_peak_allMice['peak_amp_flash_pooled_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse))
ptf_all_eachArea = np.concatenate((pooled_trace_peak_allMice['peak_timing_flash_pooled_eachArea'].values), axis=1) # 2 x (4*sum(num_sess_per_mouse))


########## Each layer: areas pooled ########## pooled_trace_peak_allMice['trace_pooled_eachDepth']

cre_all_eachDepth = np.concatenate((pooled_trace_peak_allMice['cre_pooled_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))
t_all_eachDepth = np.concatenate((pooled_trace_peak_allMice['trace_pooled_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse)) x 80
pa_all_eachDepth = np.concatenate((pooled_trace_peak_allMice['peak_amp_pooled_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))
pt_all_eachDepth = np.concatenate((pooled_trace_peak_allMice['peak_timing_pooled_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))
paf_all_eachDepth = np.concatenate((pooled_trace_peak_allMice['peak_amp_flash_pooled_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))
ptf_all_eachDepth = np.concatenate((pooled_trace_peak_allMice['peak_timing_flash_pooled_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))

depth_all_eachDepth = np.concatenate((pooled_trace_peak_allMice['depth_pooled_eachDepth'].values), axis=1) # 4 x (2*sum(num_sess_per_mouse))    


########### Make a pandas series to keep the pooled session data of all mice ########### 

indexes = ['cre_allPlanes', 'area_allPlanes', 'depth_allPlanes', 'session_labs', 'trace_allPlanes', 'peak_amp_allPlanes', 'peak_timing_allPlanes', 'peak_amp_flash_allPlanes', 'peak_timing_flash_allPlanes',\
           'cre_eachArea', 'trace_eachArea', 'peak_amp_eachArea', 'peak_timing_eachArea', 'peak_amp_flash_eachArea', 'peak_timing_flash_eachArea',\
           'cre_eachDepth', 'depth_eachDepth', 'trace_eachDepth', 'peak_amp_eachDepth', 'peak_timing_eachDepth', 'peak_amp_flash_eachDepth', 'peak_timing_flash_eachDepth']

trace_peak_allMice_sessPooled = pd.Series([cre_all, area_all, depth_all, session_labs_all, t_all, pa_all, pt_all, paf_all, ptf_all,\
   cre_all_eachArea, t_all_eachArea, pa_all_eachArea, pt_all_eachArea, paf_all_eachArea, ptf_all_eachArea,\
   cre_all_eachDepth, depth_all_eachDepth, t_all_eachDepth, pa_all_eachDepth, pt_all_eachDepth, paf_all_eachDepth, ptf_all_eachDepth],\
    index=indexes)

# print(trace_peak_allMice_sessPooled)
#    t_all[:, cre_exp_all[0,:]=='Sst'].shape


######################################################################
######################################################################
######################################################################
#%% Compute session-averaged data for each mouse
# pandas dataFrame "trace_peak_allMice_sessAvSd" is created to keep the important vars below
# use this to make summary mouse plots, by averaging across mice (each mouse is here averaged across sessions).
######################################################################
######################################################################
######################################################################

cols = ['mouse_id', 'cre', 'session_stages', 'session_labs', 'area', 'depth', 'plane', 'depth_ave', \
    'av_trace_allPlanes', 'av_peak_amp_allPlanes', 'av_peak_timing_allPlanes', 'av_peak_amp_flash_allPlanes', 'av_peak_timing_flash_allPlanes', \
    'sd_trace_allPlanes', 'sd_peak_amp_allPlanes', 'sd_peak_timing_allPlanes', 'sd_peak_amp_flash_allPlanes', 'sd_peak_timing_flash_allPlanes', \
    'av_trace_eachArea', 'av_peak_amp_eachArea', 'av_peak_timing_eachArea', 'av_peak_amp_flash_eachArea', 'av_peak_timing_flash_eachArea', \
    'sd_trace_eachArea', 'sd_peak_amp_eachArea', 'sd_peak_timing_eachArea', 'sd_peak_amp_flash_eachArea', 'sd_peak_timing_flash_eachArea', \
    'av_trace_eachDepth', 'av_peak_amp_eachDepth', 'av_peak_timing_eachDepth', 'av_peak_amp_flash_eachDepth', 'av_peak_timing_flash_eachDepth', \
    'sd_trace_eachDepth', 'sd_peak_amp_eachDepth', 'sd_peak_timing_eachDepth', 'sd_peak_amp_flash_eachDepth', 'sd_peak_timing_flash_eachDepth']

trace_peak_allMice_sessAvSd = pd.DataFrame([], columns=cols)


for im in range(len(trace_peak_allMice)): # im=5

    #%%
    mouse_id = trace_peak_allMice.iloc[im]['mouse_id']
    cre = trace_peak_allMice.iloc[im]['cre']
    session_stages = trace_peak_allMice.iloc[im]['session_stages']
    session_labs = trace_peak_allMice.iloc[im]['session_labs']
    num_sessions = len(session_stages)


    ################################################################
    #%% Session-averaged, each plane ; omission-aligned traces

    a0 = trace_peak_allMice.iloc[im]['area']
    areas = np.reshape(a0, (num_planes, num_sessions), order='F') # 8 x num_sessions # each row is one plane, all sessions
    a0 = trace_peak_allMice.iloc[im]['depth']
    depths = np.reshape(a0, (num_planes, num_sessions), order='F') # 8 x num_sessions # each row is one plane, all sessions
    a0 = trace_peak_allMice.iloc[im]['plane']
    planes = np.reshape(a0, (num_planes, num_sessions), order='F') # 8 x num_sessions # each row is one plane, all sessions

    # Average across all sessions for each plane        
    # trace
    a0 = trace_peak_allMice.iloc[im]['trace'] # (8 x num_sessions) x nFrames_upsampled
    a = np.reshape(a0, (num_planes, num_sessions, a0.shape[-1]), order='F') # 8 x num_sessions x nFrames_upsampled # each row is one plane, all sessions
    av_trace_eachPlane = np.nanmean(a, axis=1)  # 8 x nFrames_upsampled
    sd_trace_eachPlane = np.nanstd(a, axis=1) / np.sqrt(num_sessions)  # 8 x nFrames_upsampled    
    # make sure the stupid python reshape worked the way it should!!
#    b = trace_peak_allMice.iloc[im]['plane'] # (8 x num_sessions)
#    bb = np.reshape(b, (num_planes, num_sessions), order='F') # 8 x num_sessions

    # peak_amp
    a0 = trace_peak_allMice.iloc[im]['peak_amp'] # (8 x num_sessions)
    a = np.reshape(a0, (num_planes, num_sessions), order='F') # 8 x num_sessions # each row is one plane, all sessions
    av_pa_eachPlane = np.nanmean(a, axis=1)  # 8 x nFrames_upsampled
    sd_pa_eachPlane = np.nanstd(a, axis=1) / np.sqrt(num_sessions)  # 8

    # peak_timing
    a0 = trace_peak_allMice.iloc[im]['peak_timing'] # (8 x num_sessions)
    a = np.reshape(a0, (num_planes, num_sessions), order='F') # 8 x num_sessions # each row is one plane, all sessions
    av_pt_eachPlane = np.nanmean(a, axis=1)  # 8 x nFrames_upsampled
    sd_pt_eachPlane = np.nanstd(a, axis=1) / np.sqrt(num_sessions)  # 8


    # peak_amp_flash
    a0 = trace_peak_allMice.iloc[im]['peak_amp_flash'] # (8 x num_sessions)
    a = np.reshape(a0, (num_planes, num_sessions), order='F') # 8 x num_sessions # each row is one plane, all sessions
    av_paf_eachPlane = np.nanmean(a, axis=1)  # 8 x nFrames_upsampled
    sd_paf_eachPlane = np.nanstd(a, axis=1) / np.sqrt(num_sessions)  # 8 

    # peak_timing_flash
    a0 = trace_peak_allMice.iloc[im]['peak_timing_flash'] # (8 x num_sessions)
    a = np.reshape(a0, (num_planes, num_sessions), order='F') # 8 x num_sessions # each row is one plane, all sessions
    av_ptf_eachPlane = np.nanmean(a, axis=1)  # 8 x nFrames_upsampled
    sd_ptf_eachPlane = np.nanstd(a, axis=1) / np.sqrt(num_sessions)  # 8


    ################################################################
    #%% Data per area: pooled across sessions and layers for each area
    # Question 1: is there a difference between areas (V1 vs LM) # for each mouse pool across sessions and layers (for each area)

    # Average across layers and sessions for each area
    # trace
    a = pooled_trace_peak_allMice.iloc[im]['trace_pooled_eachArea'] # 2 x (4 x num_sessions) x nFrames_upsampled
    num_sessLayers_valid_eachArea = a.shape[1]
    # note: "num_sessions_valid_eachArea" is more properly set for svm codes:
    # Get the total number of valid sessions and layers for each area        
#         num_sessions_valid_eachArea = svm_this_plane_allsess.iloc[im]['num_sessions_valid_eachArea']
#         num_sessLayers_valid_eachArea = np.sum(num_sessions_valid_eachArea, axis=1)

    av_trace_pooled_eachArea = np.nanmean(a, axis=1) # 2 x nFrames_upsampled 
    sd_trace_pooled_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea) # 2 x nFrames_upsampled 


    # peak_amp
    a = pooled_trace_peak_allMice.iloc[im]['peak_amp_pooled_eachArea'] # 2 x (4 x num_sessions)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_pa_pooled_eachArea = np.nanmean(a, axis=1) # 2 x nFrames_upsampled 
    sd_pa_pooled_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea) # 2

    # peak_timing
    a = pooled_trace_peak_allMice.iloc[im]['peak_timing_pooled_eachArea'] # 2 x (4 x num_sessions)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_pt_pooled_eachArea = np.nanmean(a, axis=1) # 2 x nFrames_upsampled 
    sd_pt_pooled_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea) # 2

    # peak_amp_flash
    a = pooled_trace_peak_allMice.iloc[im]['peak_amp_flash_pooled_eachArea'] # 2 x (4 x num_sessions)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_paf_pooled_eachArea = np.nanmean(a, axis=1) # 2 x nFrames_upsampled 
    sd_paf_pooled_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea) # 2

    # peak_timing_flash
    a = pooled_trace_peak_allMice.iloc[im]['peak_timing_flash_pooled_eachArea'] # 2 x (4 x num_sessions)
#        num_sessLayers_valid_eachArea = a.shape[1]
    av_ptf_pooled_eachArea = np.nanmean(a, axis=1) # 2 x nFrames_upsampled 
    sd_ptf_pooled_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea) # 2


    ################################################################
    #%% Data per depth: pooled across sessions and areas for each depth
    # Question 2: is there a difference between superficial and deep layers # for each mouse pool across sessions and area (for each layer)... also pool 2 superifical (deep) layers into 1 

    # set average depth across sessions and areas            
    d = pooled_trace_peak_allMice.iloc[im]['depth_pooled_eachDepth']
    depth_ave = np.mean(d, axis=1) # average across areas and sessions
#    b, _,_ = pool_sesss_planes_eachArea(area, area)

    # Average across areas and sessions for each depth
    # trace
    a = pooled_trace_peak_allMice.iloc[im]['trace_pooled_eachDepth'] # 4 x (2 x num_sessions) x nFrames_upsampled
    num_sessAreas_valid_eachDepth = a.shape[1]
    av_trace_pooled_eachDepth = np.nanmean(a, axis=1) # 4 x nFrames_upsampled 
    sd_trace_pooled_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth) # 4 x nFrames_upsampled 

    # peak_amp
    a = pooled_trace_peak_allMice.iloc[im]['peak_amp_pooled_eachDepth'] # 4 x (2 x num_sessions)
#        num_sessAreas_valid_eachDepth = a.shape[1]
    av_pa_pooled_eachDepth = np.nanmean(a, axis=1) # 4 x nFrames_upsampled 
    sd_pa_pooled_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth) # 4

    # peak_timing
    a = pooled_trace_peak_allMice.iloc[im]['peak_timing_pooled_eachDepth'] # 4 x (2 x num_sessions)
#        num_sessAreas_valid_eachDepth = a.shape[1]
    av_pt_pooled_eachDepth = np.nanmean(a, axis=1) # 4 x nFrames_upsampled 
    sd_pt_pooled_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth) # 4

    # peak_amp_flash
    a = pooled_trace_peak_allMice.iloc[im]['peak_amp_flash_pooled_eachDepth'] # 4 x (2 x num_sessions)
#        num_sessAreas_valid_eachDepth = a.shape[1]
    av_paf_pooled_eachDepth = np.nanmean(a, axis=1) # 4 x nFrames_upsampled 
    sd_paf_pooled_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth) # 4

    # peak_timing_flash
    a = pooled_trace_peak_allMice.iloc[im]['peak_timing_flash_pooled_eachDepth'] # 4 x (2 x num_sessions)
#        num_sessAreas_valid_eachDepth = a.shape[1]
    av_ptf_pooled_eachDepth = np.nanmean(a, axis=1) # 4 x nFrames_upsampled 
    sd_ptf_pooled_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth) # 4


    #%% Keep vars (session averaged, each plane, also eachArea, eachDepth vars) for all mice

    trace_peak_allMice_sessAvSd.at[im,:] = \
        mouse_id, cre, session_stages, session_labs, areas, depths, planes, depth_ave, \
        av_trace_eachPlane, av_pa_eachPlane, av_pt_eachPlane, av_paf_eachPlane, av_ptf_eachPlane, \
        sd_trace_eachPlane, sd_pa_eachPlane, sd_pt_eachPlane, sd_paf_eachPlane, sd_ptf_eachPlane, \
        av_trace_pooled_eachArea, av_pa_pooled_eachArea, av_pt_pooled_eachArea, av_paf_pooled_eachArea, av_ptf_pooled_eachArea, \
        sd_trace_pooled_eachArea, sd_pa_pooled_eachArea, sd_pt_pooled_eachArea, sd_paf_pooled_eachArea, sd_ptf_pooled_eachArea, \
        av_trace_pooled_eachDepth, av_pa_pooled_eachDepth, av_pt_pooled_eachDepth, av_paf_pooled_eachDepth, av_ptf_pooled_eachDepth, \
        sd_trace_pooled_eachDepth, sd_pa_pooled_eachDepth, sd_pt_pooled_eachDepth, sd_paf_pooled_eachDepth, sd_ptf_pooled_eachDepth
            


        
#%% Lets understand how pandas groupby works
        
'''
# group cre lines manually
cres = trace_peak_allMice['cre'].values
#np.unique(cres)
Ssts = np.in1d(cres, 'Sst')
Vips = np.in1d(cres, 'Vip')
Slcs = np.in1d(cres, 'Slc17a7')
    
aa = trace_peak_allMice.iloc[Slcs]['trace'].values
aa.shape
aa2 = np.vstack(aa)
aa2.shape # 32x80 # i think data from the mice are concatenated vertically... the first 16 rows are for one mouse, and the next 16 rows are for the other mouse
b = np.nanmean(aa2[[3,16+3],:], axis=0)
b.shape
b



# group cre lines using groupby
cre_group_ave = trace_peak_allMice.groupby('cre').apply(lambda x: np.mean(x, axis=0)) # np.mean
#cre_group_ave
bb = cre_group_ave[cre_group_ave.index=='Slc17a7']['trace'].values
bb2 = np.vstack(bb)
#bb2.shape # 16x80 # i think each row is one plane (of one session), and average across mice that are in the same cre group
#bb2[3] 



# confirm that the two are the same:
np.equal(b, bb2[3])
''';


#%% #### PLOTTING STARTS HERE ####

# single mouse plots
#omissions_traces_peaks_plots_eachMouse.py

# summary of mice plots
#omissions_traces_peaks_plots_sumMice.py
