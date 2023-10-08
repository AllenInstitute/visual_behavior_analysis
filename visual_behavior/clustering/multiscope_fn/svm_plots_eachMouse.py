"""
Vars needed here are set in svm_plots_setvars.
This script uses pandas table "svm_this_plane_allsess" to make svm related plots for each mouse.
Figures will be saved in "/home/farzaneh/OneDrive/Analysis/svm_gray_omit/"

# Follow this script by "svm_plots_setVars_sumMice.py" to set vars for making average plots across mice (for each cre line).


Created on Thu Apr  9 01:09:00 2020
@author: farzaneh
"""

#####################################################################################
####################################### PLOTS #######################################
#####################################################################################

#%% For each plane, and each area, plot session-averaged data. We do this for each mouse
    
cols = colorOrder(num_planes)
bb = (.92, .7) # (.9, .7)
    
for im in range(len(svm_this_plane_allsess)): # im=0
    
    #%%    
    mouse_id = svm_this_plane_allsess.iloc[im]['mouse_id']    
    cre = svm_this_plane_allsess.iloc[im]['cre']
    session_stages = svm_this_plane_allsess.iloc[im]['session_stages']
    
    area_this_plane_allsess_allp = svm_this_plane_allsess.iloc[im]['area_this_plane_allsess_allp'] # size: (8 x num_sessions x 1)
    depth_this_plane_allsess_allp = svm_this_plane_allsess.iloc[im]['depth_this_plane_allsess_allp'] # size: (8 x num_sessions x 1)
    
    av_n_neurons_avSess_eachP = svm_this_plane_allsess.iloc[im]['av_n_neurons_avSess_eachP'] # size: (8)
    sd_n_neurons_avSess_eachP = svm_this_plane_allsess.iloc[im]['sd_n_neurons_avSess_eachP'] # size: (8)

    av_n_omissions_avSess_eachP = svm_this_plane_allsess.iloc[im]['av_n_omissions_avSess_eachP'] # size: (8)
    sd_n_omissions_avSess_eachP = svm_this_plane_allsess.iloc[im]['sd_n_omissions_avSess_eachP'] # size: (8)
        
    av_test_data_avSess_eachP = svm_this_plane_allsess.iloc[im]['av_test_data_avSess_eachP'] # size: (8 x nFrames_upsampled)
    sd_test_data_avSess_eachP = svm_this_plane_allsess.iloc[im]['sd_test_data_avSess_eachP'] # size: (8 x nFrames_upsampled)
    
    av_test_shfl_avSess_eachP = svm_this_plane_allsess.iloc[im]['av_test_shfl_avSess_eachP'] # size: (8 x nFrames_upsampled)
    sd_test_shfl_avSess_eachP = svm_this_plane_allsess.iloc[im]['sd_test_shfl_avSess_eachP'] # size: (8 x nFrames_upsampled)
    
    av_train_data_avSess_eachP = svm_this_plane_allsess.iloc[im]['av_train_data_avSess_eachP'] # size: (8 x nFrames_upsampled)
    sd_train_data_avSess_eachP = svm_this_plane_allsess.iloc[im]['sd_train_data_avSess_eachP'] # size: (8 x nFrames_upsampled)

    av_meanX_avSess_eachP = svm_this_plane_allsess.iloc[im]['av_meanX_avSess_eachP'] # size: (8 x nFrames_upsampled)
    

    if same_num_neuron_all_planes:
        av_n_neurons_svm_trained_avSess_eachP = svm_this_plane_allsess.iloc[im]['av_n_neurons_svm_trained_avSess_eachP'] # size: (8)
        sd_n_neurons_svm_trained_avSess_eachP = svm_this_plane_allsess.iloc[im]['sd_n_neurons_svm_trained_avSess_eachP'] # size: (8)
        

    n_omission_session = av_n_omissions_avSess_eachP[np.isnan(av_n_omissions_avSess_eachP)==False][0]
      
    ##### set session labels (stage names are too long) #    session_beg_inds = np.concatenate(([0], np.cumsum(num_trs_each_sess[:-1])+1))
    num_sessions = len(session_stages)
    session_labs = []
    for ibeg in range(num_sessions):
        a = str(session_stages[ibeg]); 
        en = np.argwhere([a[ich]=='_' for ich in range(len(a))])[-1].squeeze() + 6
        txt = str(session_stages[ibeg][15:]) # session_stages[ibeg][8:en]
        txt = txt[0:min(3, len(txt))]
        session_labs.append(txt)

        
    #%% 
    plt.figure(figsize=(8,11))  
    # subplots of peak amp (neuron averaged, per trial) and omit-aligned traces
    gs1 = gridspec.GridSpec(2,1) #(3, 2)#, width_ratios=[3, 1]) 
    gs1.update(bottom=.57, top=0.92, left=0.05, right=0.4, hspace=.5)
#    gs1.update(bottom=.37, top=0.9, left=0.05, right=0.5, hspace=.5)
#    gs1.update(hspace=.5, wspace=.8)
    gs2 = gridspec.GridSpec(1,2) #(3, 2)#, width_ratios=[3, 1]) 
    gs2.update(bottom=.31, top=0.48, left=0.05, right=0.95, wspace=.8)
#    gs2.update(bottom=.05, top=0.25, left=0.05, right=0.95, wspace=.5)

    gs3 = gridspec.GridSpec(1,2) #(3, 2)#, width_ratios=[3, 1]) 
    gs3.update(bottom=.05, top=0.22, left=0.05, right=0.95, wspace=.8)

    plt.suptitle('%s, mouse %d, ~%d omissions\n%d sessions: %s' %(cre, mouse_id, n_omission_session, len(session_stages), \
                                                                  session_labs), fontsize=14)
#    gs1.update(left=0.05, right=0.68, wspace=.2, hspace=1)
    # subplots of peak amp/ timing (trial-median)
#    gs2 = gridspec.GridSpec(2, 1)
#    gs2.update(left=0.77, right=0.98, wspace=0.8, hspace=1)
    
    
    #%% Plot SVM classification accuracy (indicates magnitude of population signal about omission vs. gray at each frame) 
    # Make 2 subplots, one for each area    
    
    ############################### 1st area ###############################
    h1 = []
    for iplane in range(4): #num_planes):
        
#        plt.subplot(2,1,1)
        ax = plt.subplot(gs1[0,0])
        
        area = area_this_plane_allsess_allp[iplane] # num_sess 
        depth = depth_this_plane_allsess_allp[iplane] # num_sess 
        # all_sess[all_sess.mouse_id==mouse_id].area.at[iplane] # num_sess
        # all_sess[all_sess.mouse_id==mouse_id].depth.at[iplane] # num_sess
#        if np.ndim(area)==0: # happens when len(session_stages)==1
#            area = [area]
#            depth = [depth]         
                    
        if np.isnan(av_n_neurons_avSess_eachP[iplane])==False: # for invalid experiments, this will be nan
            if same_num_neuron_all_planes:
                lab = '%s, ~%dum, nNeurs ~%d, trained ~%d' %(np.unique(area)[0], np.mean(depth), av_n_neurons_avSess_eachP[iplane], av_n_neurons_svm_trained_avSess_eachP[iplane])
            else:
                lab = '%s, ~%dum, nNeurs ~%d' %(np.unique(area)[0], np.mean(depth), av_n_neurons_avSess_eachP[iplane])
            
            h1_0 = plt.plot(time_trace_new, av_test_data_avSess_eachP[iplane], color=cols[iplane], label=(lab), markersize=3.5)[0]
            h2 = plt.plot(time_trace_new, av_test_shfl_avSess_eachP[iplane], color='gray', label='shfl', markersize=3.5)[0]
#            plt.plot(time_trace_new, av_train_data_avSess_eachP[iplane], color=cols[iplane], label=(lab), markersize=3.5)[0]
            
            # errorbars (standard error across cv samples)   
            plt.fill_between(time_trace_new, av_test_data_avSess_eachP[iplane] - sd_test_data_avSess_eachP[iplane], av_test_data_avSess_eachP[iplane] + sd_test_data_avSess_eachP[iplane], alpha=alph, edgecolor=cols[iplane], facecolor=cols[iplane])
            plt.fill_between(time_trace_new, av_test_shfl_avSess_eachP[iplane] - sd_test_shfl_avSess_eachP[iplane], av_test_shfl_avSess_eachP[iplane] + sd_test_shfl_avSess_eachP[iplane], alpha=alph, edgecolor='y', facecolor='y')
            
            h1.append(h1_0)
    h1.append(h2)  # check if it works!
#    h1 = np.array(h1)    
        
    
    lims = np.array([np.nanmin(np.concatenate((av_test_shfl_avSess_eachP - sd_test_shfl_avSess_eachP , av_test_data_avSess_eachP - sd_test_data_avSess_eachP))), \
                     np.nanmax(av_test_data_avSess_eachP + sd_test_data_avSess_eachP)])
    lims[0] = lims[0] - np.diff(lims) / 20.
    lims[1] = lims[1] + np.diff(lims) / 20.

    # set handles for legend    
    H = h1 # check if it works!
#    if len(h1)==2:
#        H = [h1[0], h1[1], h2]
#    elif len(h1)==3:
#        H = [h1[0], h1[1], h1[2], h2]
#    elif len(h1)==4:
#        H = [h1[0], h1[1], h1[2], h1[3], h2]
        
    # add to plots flash lines, ticks and legend
    plot_flashLines_ticks_legend(lims, H, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace)
#    plt.ylabel('')
    
    

    ############################### 2nd area ###############################
    h1 = []
    for iplane in np.arange(4,num_planes): #num_planes):
        
#        plt.subplot(2,1,2)
        ax = plt.subplot(gs1[1,0])

        area = area_this_plane_allsess_allp[iplane] # num_sess 
        depth = depth_this_plane_allsess_allp[iplane] # num_sess         
#        area = all_sess[all_sess.mouse_id==mouse_id].area.at[iplane] # num_sess
#        depth = all_sess[all_sess.mouse_id==mouse_id].depth.at[iplane] # num_sess
        if np.ndim(area)==0: # happens when len(session_stages)==1
            area = [area]
            depth = [depth]

        if np.isnan(av_n_neurons_avSess_eachP[iplane])==False: # for invalid experiments, this will be nan
            if same_num_neuron_all_planes:
                lab = '%s, ~%dum, nNeurs ~%d, trained ~%d' %(np.unique(area)[0], np.mean(depth), av_n_neurons_avSess_eachP[iplane], av_n_neurons_svm_trained_avSess_eachP[iplane])
            else:
                lab = '%s, ~%dum, nNeurs ~%d' %(np.unique(area)[0], np.mean(depth), av_n_neurons_avSess_eachP[iplane])
            
            h1_0 = plt.plot(time_trace_new, av_test_data_avSess_eachP[iplane],color=cols[iplane], label=(lab), markersize=3.5)[0]
            h2 = plt.plot(time_trace_new, av_test_shfl_avSess_eachP[iplane], color='gray', label='shfl', markersize=3.5)[0]
#            plt.plot(time_trace_new, av_train_data_avSess_eachP[iplane], color=cols[iplane], label=(lab), markersize=3.5)[0]
            
            # errorbars (standard error across cv samples)   
            plt.fill_between(time_trace_new, av_test_data_avSess_eachP[iplane] - sd_test_data_avSess_eachP[iplane], av_test_data_avSess_eachP[iplane] + sd_test_data_avSess_eachP[iplane], alpha=alph, edgecolor=cols[iplane], facecolor=cols[iplane])
            plt.fill_between(time_trace_new, av_test_shfl_avSess_eachP[iplane] - sd_test_shfl_avSess_eachP[iplane], av_test_shfl_avSess_eachP[iplane] + sd_test_shfl_avSess_eachP[iplane], alpha=alph, edgecolor='y', facecolor='y')
            
            h1.append(h1_0)
    h1.append(h2)
#    h1 = np.array(h1)    


    # set handles for legend    
    H = h1 # check if it works!        
#    if len(h1)==2:
#        H = [h1[0], h1[1], h2]
#    elif len(h1)==3:
#        H = [h1[0], h1[1], h1[2], h2]
#    elif len(h1)==4:
#        H = [h1[0], h1[1], h1[2], h1[3], h2]
    
    # add to plots flash lines, ticks and legend
    plot_flashLines_ticks_legend(lims, H, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace)

#    plt.title('%s, mouse %d, ~%d omissions' %(cre, mouse_id, av_n_omissions_avSess_eachP[0]), fontsize=13.5)
#    plt.subplots_adjust(hspace=.5)
    

    ##################################################################
    ##################################################################    
    #%% Plot results per area (pooled across sessions and layers for each area)
    # Question 1: is there a difference between areas (V1 vs LM) # for each mouse pool across sessions and layers (for each area)
    ##################################################################
    ##################################################################    

    # Get the total number of valid sessions and layers for each area        
    num_sessions_valid_eachArea = svm_this_plane_allsess.iloc[im]['num_sessions_valid_eachArea']
    num_sessLayers_valid_eachArea = np.sum(num_sessions_valid_eachArea, axis=1)
    
    # Average across layers and sessions for each area
    # test_data
    a = svm_this_plane_allsess.iloc[im]['av_test_data_pooled_sesss_planes_eachArea'] # 2 x (4 x num_sessions) x nFrames_upsampled
    av_test_data_pooledSessPlanes_eachArea = np.nanmean(a, axis=1) # 2 x nFrames_upsampled 
    sd_test_data_pooledSessPlanes_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea)[:,np.newaxis] # 2 x nFrames_upsampled 

    # test_shfl
    a = svm_this_plane_allsess.iloc[im]['av_test_shfl_pooled_sesss_planes_eachArea'] # 2 x (4 x num_sessions) x nFrames_upsampled
    av_test_shfl_pooledSessPlanes_eachArea = np.nanmean(a, axis=1) # 2 x nFrames_upsampled 
    sd_test_shfl_pooledSessPlanes_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea)[:,np.newaxis] # 2 x nFrames_upsampled 

    
    ##################
    colsa = ['b', 'k']    # first plot V1 (in black, index 1) then LM (in blue, index 0)
    ax = plt.subplot(gs2[0])
#    ax = plt.subplot(gs1[2,0])
    plt.title('Sessions and layers pooled', fontsize=13, y=1)
    
    h1 = []
    for iarea in [1,0]: # first plot V1 then LM  #range(a.shape[0]):
        lab = '%s, n=%d' %(distinct_areas[iarea], num_sessLayers_valid_eachArea[iarea])
        h1_0 = plt.plot(time_trace_new, av_test_data_pooledSessPlanes_eachArea[iarea], color = colsa[iarea], label=(lab), markersize=3.5)[0]
        h2 = plt.plot(time_trace_new, av_test_shfl_pooledSessPlanes_eachArea[iarea], color='gray', label='shfl', markersize=3.5)[0]
        
        plt.fill_between(time_trace_new, av_test_data_pooledSessPlanes_eachArea[iarea] - sd_test_data_pooledSessPlanes_eachArea[iarea] , \
                         av_test_data_pooledSessPlanes_eachArea[iarea] + sd_test_data_pooledSessPlanes_eachArea[iarea], alpha=alph, edgecolor=colsa[iarea], facecolor=colsa[iarea])
        
        h1.append(h1_0)
    h1.append(h2)

    # add to plots flash lines, ticks and legend
    lims = np.array([np.nanmin(np.concatenate((av_test_data_pooledSessPlanes_eachArea - sd_test_data_pooledSessPlanes_eachArea , \
                                               av_test_shfl_pooledSessPlanes_eachArea - sd_test_shfl_pooledSessPlanes_eachArea))), \
                     np.nanmax(av_test_data_pooledSessPlanes_eachArea + sd_test_data_pooledSessPlanes_eachArea)])
    lims[0] = lims[0] - np.diff(lims) / 20.
    lims[1] = lims[1] + np.diff(lims) / 20.
    
    H = h1 # check if this works!
#    H = [h1[0], h1[1], h2]

    plot_flashLines_ticks_legend(lims, H, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb)
#    plt.ylabel('')
    
    # Sanity check: above plots should be the same as below
#    plt.plot(time_trace_new, np.mean(av_test_data_avSess_eachP[:4], axis=0), color = colsa[0])
#    plt.plot(time_trace_new, np.mean(av_test_data_avSess_eachP[4:], axis=0), color = colsa[1])



    ##################################################################
    ##################################################################    
    #%% Plot results per depth (pooled across sessions and areas for each depth)
    # Question 2: is there a difference between superficial and deep layers # for each mouse pool across sessions and area (for each layer)... also pool 2 superifical (deep) layers into 1 
    ##################################################################
    ##################################################################    

    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['depth'].values # (pooled: num_planes x num_sessions)    
    depths_allsess = np.reshape(y, (num_planes, num_sessions), order='F') # num_planes x num_sessions


    # Get the total number of valid sessions and layers for each depth
    num_sessAreas_valid_eachDepth = np.sum(num_sessions_valid_eachArea, axis=0)
    
    # Average across areas and sessions for each depth
    # test_data
    a = svm_this_plane_allsess.iloc[im]['av_test_data_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions) x nFrames_upsampled
    av_test_data_pooledSessAreas_eachDepth = np.nanmean(a, axis=1) # 4 x nFrames_upsampled 
    sd_test_data_pooledSessAreas_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth)[:,np.newaxis] # 4 x nFrames_upsampled 

    # test_shfl
    a = svm_this_plane_allsess.iloc[im]['av_test_shfl_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions) x nFrames_upsampled
    av_test_shfl_pooledSessAreas_eachDepth = np.nanmean(a, axis=1) # 4 x nFrames_upsampled 
    sd_test_shfl_pooledSessAreas_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth)[:,np.newaxis] # 4 x nFrames_upsampled 
    
    
    ##################
    colsd = colorOrder(a.shape[0]) # depth1_area2    
    ax = plt.subplot(gs2[1])
#    ax = plt.subplot(gs1[2,1])
    plt.title('Sessions and areas pooled', fontsize=13, y=1)
    
    h1 = []
    for idepth in range(a.shape[0]):
        lab = '%d um, n=%d' %(depths_allsess[idepth,0], num_sessAreas_valid_eachDepth[idepth])
        h1_0 = plt.plot(time_trace_new, av_test_data_pooledSessAreas_eachDepth[idepth], color = colsd[idepth], label=(lab), markersize=3.5)[0]
        h2 = plt.plot(time_trace_new, av_test_shfl_pooledSessAreas_eachDepth[idepth], color='gray', label='shfl', markersize=3.5)[0]
        
        plt.fill_between(time_trace_new, av_test_data_pooledSessAreas_eachDepth[idepth] - sd_test_data_pooledSessAreas_eachDepth[idepth] , \
                         av_test_data_pooledSessAreas_eachDepth[idepth] + sd_test_data_pooledSessAreas_eachDepth[idepth], alpha=alph, edgecolor=colsd[idepth], facecolor=colsd[idepth])
        
        h1.append(h1_0)
    h1.append(h2)
    
    # add to plots flash lines, ticks and legend
    lims = np.array([np.nanmin(np.concatenate((av_test_data_pooledSessAreas_eachDepth - sd_test_data_pooledSessAreas_eachDepth , \
                                               av_test_shfl_pooledSessAreas_eachDepth - sd_test_shfl_pooledSessAreas_eachDepth))), \
                     np.nanmax(av_test_data_pooledSessAreas_eachDepth + sd_test_data_pooledSessAreas_eachDepth)])
    lims[0] = lims[0] - np.diff(lims) / 20.
    lims[1] = lims[1] + np.diff(lims) / 20.
    
    H = h1 # check if this works!
#    H = [h1[0], h1[1], h1[2], h1[3], h2]
    
    plot_flashLines_ticks_legend(lims, H, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb)
#    plt.ylabel('')
    
    # Sanity check: above plots should be the same as below (or close, as the arrays below are first averaged across sessions, then here we averaged across areas ... depending on how many sessions were valid the results may vary with above)
#    plt.plot(time_trace_new, np.mean(av_test_data_avSess_eachP[[0,4]], axis=0), color = colsd[0])
#    plt.plot(time_trace_new, np.mean(av_test_data_avSess_eachP[[1,5]], axis=0), color = colsd[1])
#    plt.plot(time_trace_new, np.mean(av_test_data_avSess_eachP[[2,6]], axis=0), color = colsd[2])
#    plt.plot(time_trace_new, np.mean(av_test_data_avSess_eachP[[3,7]], axis=0), color = colsd[3])
    
    

    ##################################################################
    ##################################################################    
    ##################################################################
    ##################################################################    
    ##################################################################
    ##################################################################    
    
    #%% meanX (average DF/F across trials (half gray (frame omission-1), half non gray (frames_svm)) for each neuron... 
    # these values were subtracted from x_svm to z score the matrix before running SVM).

    ##################################################################
    ##################################################################        
    #%% Plot results per area (pooled across sessions and layers for each area)
    # Question 1: is there a difference between areas (V1 vs LM) # for each mouse pool across sessions and layers (for each area)
    ##################################################################
    ##################################################################    

    # Get the total number of valid sessions and layers for each area        
    num_sessions_valid_eachArea = svm_this_plane_allsess.iloc[im]['num_sessions_valid_eachArea']
    num_sessLayers_valid_eachArea = np.sum(num_sessions_valid_eachArea, axis=1)
    
    # Average across layers and sessions for each area
    # meanX
    a = svm_this_plane_allsess.iloc[im]['meanX_pooled_sesss_planes_eachArea'] # 2 x (4 x num_sessions) x nFrames_upsampled
    av_meanX_pooledSessPlanes_eachArea = np.nanmean(a, axis=1) # 2 x nFrames_upsampled 
    sd_meanX_pooledSessPlanes_eachArea = np.nanstd(a, axis=1) / np.sqrt(num_sessLayers_valid_eachArea)[:,np.newaxis] # 2 x nFrames_upsampled 

    
    ##################
    colsa = ['b', 'k']    
    ax = plt.subplot(gs3[0])
#    ax = plt.subplot(gs1[2,0])
    plt.title('Sessions and layers pooled', fontsize=13, y=1)
    
    h1 = []
    for iarea in range(a.shape[0]):
        lab = '%s, n=%d' %(distinct_areas[iarea], num_sessLayers_valid_eachArea[iarea])
        h1_0 = plt.plot(time_trace_new, av_meanX_pooledSessPlanes_eachArea[iarea], color = colsa[iarea], label=(lab), markersize=3.5)[0]
        
        plt.fill_between(time_trace_new, av_meanX_pooledSessPlanes_eachArea[iarea] - sd_meanX_pooledSessPlanes_eachArea[iarea] , \
                         av_meanX_pooledSessPlanes_eachArea[iarea] + sd_meanX_pooledSessPlanes_eachArea[iarea], alpha=alph, edgecolor=colsa[iarea], facecolor=colsa[iarea])
        
        h1.append(h1_0)
    h1.append(h2)
    
    # add to plots flash lines, ticks and legend
    lims = np.array([np.nanmin(av_meanX_pooledSessPlanes_eachArea - sd_meanX_pooledSessPlanes_eachArea), \
                     np.nanmax(av_meanX_pooledSessPlanes_eachArea + sd_meanX_pooledSessPlanes_eachArea)])
    lims[0] = lims[0] - np.diff(lims) / 20.
    lims[1] = lims[1] + np.diff(lims) / 20.
    
    H = h1 # check if this works!
#    H = [h1[0], h1[1], h2]

    plot_flashLines_ticks_legend(lims, H, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb)
    plt.ylabel('DF/F')
    
    # Sanity check: above plots should be the same as below
#    plt.plot(time_trace_new, np.mean(av_meanX_avSess_eachP[:4], axis=0), color = colsa[0])
#    plt.plot(time_trace_new, np.mean(av_meanX_avSess_eachP[4:], axis=0), color = colsa[1])


    ##################################################################
    ##################################################################    
    #%% Plot results per depth (pooled across sessions and areas for each depth)
    # Question 2: is there a difference between superficial and deep layers # for each mouse pool across sessions and area (for each layer)... also pool 2 superifical (deep) layers into 1 
    ##################################################################
    ##################################################################    
    
    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['depth'].values # (pooled: num_planes x num_sessions)    
    depths_allsess = np.reshape(y, (num_planes, num_sessions), order='F') # num_planes x num_sessions


    # Get the total number of valid sessions and layers for each depth
    num_sessAreas_valid_eachDepth = np.sum(num_sessions_valid_eachArea, axis=0)
    
    # Average across areas and sessions for each depth
    # test_data
    a = svm_this_plane_allsess.iloc[im]['meanX_pooled_sesss_areas_eachDepth'] # 4 x (2 x num_sessions) x nFrames_upsampled
    av_meanX_pooledSessAreas_eachDepth = np.nanmean(a, axis=1) # 4 x nFrames_upsampled 
    sd_meanX_pooledSessAreas_eachDepth = np.nanstd(a, axis=1) / np.sqrt(num_sessAreas_valid_eachDepth)[:,np.newaxis] # 4 x nFrames_upsampled 
    
    
    ##################
    colsd = colorOrder(a.shape[0]) # depth1_area2    
    ax = plt.subplot(gs3[1])
#    ax = plt.subplot(gs1[2,1])
    plt.title('Sessions and areas pooled', fontsize=13, y=1)
    
    h1 = []
    for idepth in range(a.shape[0]):
        lab = '%d um, n=%d' %(depths_allsess[idepth,0], num_sessAreas_valid_eachDepth[idepth])
        h1_0 = plt.plot(time_trace_new, av_meanX_pooledSessAreas_eachDepth[idepth], color = colsd[idepth], label=(lab), markersize=3.5)[0]
        
        plt.fill_between(time_trace_new, av_meanX_pooledSessAreas_eachDepth[idepth] - sd_meanX_pooledSessAreas_eachDepth[idepth] , \
                         av_meanX_pooledSessAreas_eachDepth[idepth] + sd_meanX_pooledSessAreas_eachDepth[idepth], alpha=alph, edgecolor=colsd[idepth], facecolor=colsd[idepth])
        
        h1.append(h1_0)
    h1.append(h2)
    
    # add to plots flash lines, ticks and legend
    lims = np.array([np.nanmin(av_meanX_pooledSessAreas_eachDepth - sd_meanX_pooledSessAreas_eachDepth), \
                     np.nanmax(av_meanX_pooledSessAreas_eachDepth + sd_meanX_pooledSessAreas_eachDepth)])
    lims[0] = lims[0] - np.diff(lims) / 20.
    lims[1] = lims[1] + np.diff(lims) / 20.
    
    H = h1 # check if this works!
#    H = [h1[0], h1[1], h1[2], h1[3], h2]
    
    plot_flashLines_ticks_legend(lims, H, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb)
    plt.ylabel('DF/F')
    
    # Sanity check: above plots should be the same as below (or close, as the arrays below are first averaged across sessions, then here we averaged across areas ... depending on how many sessions were valid the results may vary with above)
#    plt.plot(time_trace_new, np.mean(av_meanX_avSess_eachP[[0,4]], axis=0), color = colsd[0])
#    plt.plot(time_trace_new, np.mean(av_meanX_avSess_eachP[[1,5]], axis=0), color = colsd[1])
#    plt.plot(time_trace_new, np.mean(av_meanX_avSess_eachP[[2,6]], axis=0), color = colsd[2])
#    plt.plot(time_trace_new, np.mean(av_meanX_avSess_eachP[[3,7]], axis=0), color = colsd[3])
    
    

    
    #%% Save the figure for each mouse
    
    if dosavefig:
#        plt.figure(fa.number)
        if same_num_neuron_all_planes:
            nam = '%s_mouse%d%s_%s_sameNumNeuronsAllPlanes_%s' %(cre, mouse_id, whatSess, fgn, now)
        else:
            nam = '%s_mouse%d%s_%s_%s' %(cre, mouse_id, whatSess, fgn, now)

        fign = os.path.join(dir0, dir_now, nam+fmt)        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)        







    
#%% SVM plots

#'session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'meanX_allFrs', 'stdX_allFrs', 
#'av_train_data_new', 'av_test_data_new', 'av_test_shfl_new', 'av_test_chance_new', 'sd_train_data_new', 'sd_test_data_new', 'sd_test_shfl_new', 'sd_test_chance_new'

"""
plt.figure()  

h0 = plt.plot(time_trace, av_train_data,'ko', time_trace_new, av_train_data_new,'k-', label='train', markersize=3.5);
h1 = plt.plot(time_trace, av_test_data,'ro', time_trace_new, av_test_data_new,'r-', label='test', markersize=3.5);
h2 = plt.plot(time_trace, av_test_shfl,'yo', time_trace_new, av_test_shfl_new,'y-', label='shfl', markersize=3.5);
h3 = plt.plot(time_trace, av_test_chance,'bo', time_trace_new, av_test_chance_new,'b-',  label='chance', markersize=3.5);

# errorbars (standard error across cv samples)   
alph = .3
plt.fill_between(time_trace_new, av_train_data_new - sd_train_data_new, av_train_data_new + sd_train_data_new, alpha=alph, edgecolor='k', facecolor='k')
plt.fill_between(time_trace_new, av_test_data_new - sd_test_data_new, av_test_data_new + sd_test_data_new, alpha=alph, edgecolor='r', facecolor='r')
plt.fill_between(time_trace_new, av_test_shfl_new - sd_test_shfl_new, av_test_shfl_new + sd_test_shfl_new, alpha=alph, edgecolor='y', facecolor='y')
plt.fill_between(time_trace_new, av_test_chance_new - sd_test_chance_new, av_test_chance_new + sd_test_chance_new, alpha=alph, edgecolor='b', facecolor='b')


mn = 45; mx = 80
# mark omission onset
plt.vlines([0], mn, mx, color='k', linestyle='--')
# mark the onset of flashes
plt.vlines(flashes_win_trace_index_unq_time, mn, mx, color='y', linestyle='-.')
# mark the onset of grays
plt.vlines(grays_win_trace_index_unq_time, mn, mx, color='gray', linestyle=':')


ax = plt.gca()
xmj = np.arange(0, x[-1], .5)
ax.set_xticks(xmj); # plt.xticks(np.arange(0,x[-1],.25)); #, fontsize=10)
ax.set_xticklabels(xmj, rotation=45)       
xmn = np.arange(.25, x[-1], .5)
ax.xaxis.set_minor_locator(ticker.FixedLocator(xmn))
ax.tick_params(labelsize=10, length=6, width=2, which='major')
ax.tick_params(labelsize=10, length=5, width=1, which='minor')
#    plt.xticklabels(np.arange(0,x[-1],.25))


plt.legend(handles=[h0[1],h1[1],h2[1],h3[1]], loc='center left', bbox_to_anchor=(1, .7), frameon=False, handlelength=.5, fontsize=12)
plt.ylabel('% Classification accuracy', fontsize=12)
plt.xlabel('Time after omission (sec)', fontsize=12)  

plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
seaborn.despine()#left=True, bottom=True, right=False, top=False)


## Plot with x axis in frame units
'''
plt.figure()
plt.plot(av_train_data, color='k', label='train') 
plt.plot(av_test_data, color='r', label='test') 
plt.plot(av_test_shfl, color='y', label='shfl')
plt.plot(av_test_chance, color='b', label='chance')
plt.legend(loc='center left', bbox_to_anchor=(1, .7))                

mn = 45; mx = 80
# mark omission onset
plt.vlines([0], mn, mx, color='k', linestyle='--')
# mark the onset of flashes
plt.vlines(flashes_win_trace_index_unq, mn, mx, color='y', linestyle='-.')
# mark the onset of grays
plt.vlines(grays_win_trace_index_unq, mn, mx, color='gray', linestyle=':')
    
plt.legend(handles=[h0[1],h1[1],h2[1],h3[1]], loc='center left', bbox_to_anchor=(1, .7), frameon=False, handlelength=.5, fontsize=12)
plt.ylabel('% Classification accuracy', fontsize=12)
plt.xlabel('Frame after omission (sec)', fontsize=12)      

"""




