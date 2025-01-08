#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function is called in svm_init.

It loads the svm file that was saved for each experiment, and returns a pandas dataframe (this_sess), which includes a number of columns, including average and st error of class accuracies across CV samples; also quantification of omission and flash evoked responses.
svm_init combines this_sess for all sessions into a single pandas table (all_sess), and saves it at /allen/programs/braintv/workgroups/nc-ophys/Farzaneh/SVM/same_num_neurons_all_planes/all_sess_svm_gray_omit*

Follow this function by svm_plots_setVars.py to make plots.

Created on Wed Aug  7 10:42:49 2019
@author: farzaneh
"""

## First run svm_init, then run below

from def_funs import *
from omissions_traces_peaks_quantify import *
# import re

def svm_main_post(session_id, experiment_ids, validity_log_all, dir_svm, frames_svm, all_sess, same_num_neuron_all_planes, use_ct_traces, use_np_corr, use_common_vb_roi, mean_notPeak, peak_win, flash_win, flash_win_vip, flash_win_timing, bl_percentile, cols, doShift_again, analysis_dates, use_spont_omitFrMinus1, doPlots=0):
    
    #%%    
    if type(frames_svm)==int: # First type you ran svm analysis: SVM was run on 30 frames after omission and 0 frames before omission (each frame after omission was compared with a gray frame (frame -1 relative to omission))
        samps_bef = 0
        samps_aft = frames_svm #frames_after_omission # 30 # frames_after_omission in svm_main # we trained the classifier for 30 frames after omission    
        
    else:        
        samps_bef = -frames_svm[0] 
        samps_aft = frames_svm[-1]+1 
        
    numFrames = samps_bef + samps_aft
    
    
    #%%
    svmn = 'svm_gray_omit'
    
    if use_spont_omitFrMinus1==0:
        svmn = svmn + '_spontFrs'    
    if same_num_neuron_all_planes:
        svmn = svmn + '_sameNumNeuronsAllPlanes'
        
#     frame_dur = np.array([0.093]) # sec (~10.7 Hz; each pair of planes that are recorded simultaneously have time resolution frame_dur)    
    frames_svm = np.array(frames_svm)
    doPeakPerTrial = 0
    doROC = 0

    '''
    doShift_again = 0
    doPeakPerTrial = 0
    doROC = 0
    mean_notPeak = 1  # set to 1, so mean is computed; it is preferred because SST and SLC responses don't increase after omission
    # if 1, to quantify omission-evoked response, compute average trace during peak_win, instead of the peak amplitude.
    # if 1, we will still use the timing of the "peak" for peak timing measures ...
    peak_win = [0, .75] # you should totally not go beyond 0.75!!! # go with 0.75ms if you don't want the responses to be contaminated by the next flash
    # [0,2] # flashes are 250ms and gray screen is 500ms # sec # time window to find peak activity, seconds relative to omission onset
    # [-.75, 0] is too large for flash responses... (unlike omit responses, they peak fast!
    # [-.75, -.4] #[-.75, 0] # window (relative to omission) for computing flash-evoked responses (750ms includes flash and gray)
    flash_win = [-.75, -.25]

    flash_win_timing = [-.85, -.5]
    bl_percentile = 10 #20  # for peak measurements, we subtract pre-omit baseline from the peak. bl_percentile determines what pre-omit values will be used.      
    '''
    
    
    #%% Load some important variables from the experiment
    
    #    [whole_data, data_list, table_stim] = load_session_data(session_id) # data_list is similar to whole_data but sorted by area and depth
    [whole_data, data_list, table_stim, behav_data] = load_session_data_new(session_id, experiment_ids, use_ct_traces, use_np_corr, use_common_vb_roi)
    
    
    #%%
    exp_ids = list(whole_data.keys())
    mouse = whole_data[exp_ids[0]]['mouse']
    date = whole_data[exp_ids[0]]['experiment_date']    
    cre = whole_data[exp_ids[0]]['cre'] # it will be the same for all lims_ids (all planes in a session)
    stage = whole_data[exp_ids[0]]['stage']
    
    print(cre, '\n', date)
    
    
    #%% Loop through the 8 planes of each session
    
    # the same "cols" is defined in svm_init.py which calls this script    
#     cols0 = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'meanX_allFrs_new', 'stdX_allFrs_new', 'av_train_data_new', 'av_test_data_new', 'av_test_shfl_new', 'av_test_chance_new', 'sd_train_data_new', 'sd_test_data_new', 'sd_test_shfl_new', 'sd_test_chance_new', 'peak_amp_omit_trainTestShflChance', 'peak_timing_omit_trainTestShflChance', 'peak_amp_flash_trainTestShflChance', 'peak_timing_flash_trainTestShflChance'])
#     if same_num_neuron_all_planes:
#         cols = np.concatenate((cols0, ['population_sizes_to_try'])) 
#     else:
#         cols = cols0
            #        cols = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'meanX_allFrs', 'stdX_allFrs', 'av_train_data_new', 'av_test_data_new', 'av_test_shfl_new', 'av_test_chance_new', 'sd_train_data_new', 'sd_test_data_new', 'sd_test_shfl_new', 'sd_test_chance_new', 'population_sizes_to_try'])
    
    this_sess = pd.DataFrame([], columns = cols)
    
    for index, lims_id in enumerate(data_list['lims_id']): 
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
        
        print('\n======================== Analyzing experiment %s, plane %d/%d ========================\n' %(lims_id, index+1, num_planes))
        
        depth = whole_data[lims_id]['imaging_depth']
        area = whole_data[lims_id]['targeted_structure']
    
        this_sess.at[index, cols[range(8)]] = session_id, lims_id, mouse, date, cre, stage, area, depth
    

        #%% Abort the analysis if the experiment is invalid
        
        # the comment below is not anymore true, bc the input to this function is list_all_experiments (not list_all_experiments_valid):
        # we dont realy need this. because in svm_init, we only use valid experiment ids... so data_list here has only the valid experiment ids.
        
        if validity_log_all.iloc[validity_log_all.lims_id.values == int(lims_id)]['valid'].bool() == False: 
            print('Skipping invalid experiment %d' %int(lims_id))
#            this_sess.at[index, :] = np.nan # check this works.
#            sys.exit()
            set_to_nan = 1  # set svm vars to nan      
    
        else:
            #%% Set the h5 filename containing SVM vars
            
            cre_now = cre[:cre.find('-')]

            # mouse, session, experiment: m, s, e
            if analysis_dates[0] == '':
                name = '%s_m-%d_s-%d_e-%s_%s_.' %(cre_now, mouse, session_id, lims_id, svmn)
            else:
                name = '%s_m-%d_s-%d_e-%s_%s_%s_.' %(cre_now, mouse, session_id, lims_id, svmn, analysis_dates[0])
            
            svmName ,_ = all_sess_set_h5_fileName(name, dir_svm, all_files=0)
            
                        
            if len(svmName)==0:
                print(name)
                print('\nSVM h5 file does not exist! uncanny! (most likely due to either too few neurons or absence of omissions!)')
                set_to_nan = 1
#                 sys.exit()
            else:
                set_to_nan = 0
            

            ########################################################################################################    
            ########################################################################################################    
            ########################################################################################################
            ########################################################################################################                

            if set_to_nan==0:
                
                #%% Load svm dataframe and set SVM vars
                
                svm_vars = pd.read_hdf(svmName, key='svm_vars')
                
                '''
                   'session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage',
                   'area', 'depth', 'n_omissions', 'n_neurons', 'thAct', 'numSamples',
                   'softNorm', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs',
                   'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
                   'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
                   'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs',
                   'testTrInds_allSamps_allFrs', 'Ytest_allSamps_allFrs',
                   'Ytest_hat_allSampsFrs_allFrs'
                '''
                
                n_omissions = svm_vars.iloc[0]['n_omissions']
                n_neurons = svm_vars.iloc[0]['n_neurons']
                print(f'{n_omissions} omissions; {n_neurons} neurons')
                
                frame_dur = svm_vars.iloc[0]['frame_dur']                
                print(f'Frame duration {frame_dur} ms')
                if np.logical_or(frame_dur < .089, frame_dur > .1):
                    print(f'\n\nFrame duration is unexpected!! {frame_dur}ms\n\n')

                flash_omit_dur_all = svm_vars.iloc[0]['flash_omit_dur_all']
                flash_omit_dur_fr_all = svm_vars.iloc[0]['flash_omit_dur_fr_all']
                
                meanX_allFrs = svm_vars.iloc[0]['meanX_allFrs'] # nFrames x nNeurons
                stdX_allFrs = svm_vars.iloc[0]['stdX_allFrs']
            
                cbest_allFrs = svm_vars.iloc[0]['cbest_allFrs'] # nFrames
                numSamples = svm_vars.iloc[0]['numSamples'] # 50
                # if same_num_neuron_all_planes, each element of the arrays below is for a population of a given size (population_sizes_to_try)
                # and perClassErrorTrain_data_allFrs[0] has size has size: # numShufflesN x nSamples x nCval
                perClassErrorTrain_data_allFrs = svm_vars.iloc[0]['perClassErrorTrain_data_allFrs'] # 50 x 30 (numSamps x nFrames)
                perClassErrorTest_data_allFrs = svm_vars.iloc[0]['perClassErrorTest_data_allFrs']
                perClassErrorTest_shfl_allFrs = svm_vars.iloc[0]['perClassErrorTest_shfl_allFrs']
                perClassErrorTest_chance_allFrs = svm_vars.iloc[0]['perClassErrorTest_chance_allFrs']
                
                w_data_allFrs = svm_vars.iloc[0]['w_data_allFrs'] # numSamps x nNeurons x nFrames
                b_data_allFrs = svm_vars.iloc[0]['b_data_allFrs'] # numSamps x nFrames
                
#                frames_svm = svm_vars.iloc[0]['frames_svm']
                
            #    plt.plot(cbest_allFrs)
            #    plt.plot(sorted(cbest_allFrs)[:-2])
                
                
                ########################################################################################################    
                ########################################################################################################    
                ########################################################################################################
                ########################################################################################################                                
                #%% Average and st error of class accuracies across cross-validation samples
                ########################################################################################################    
                ########################################################################################################    
                ########################################################################################################
                ########################################################################################################                
                
                if same_num_neuron_all_planes: # size perClassErrorTrain_data_allFrs: num_all_pop_sizes x numShufflesN x nSamples x nFrames

                    population_sizes_to_try = svm_vars.iloc[0]['population_sizes_to_try']
                    numShufflesN = np.shape(perClassErrorTrain_data_allFrs[0])[0]
#                    numShufflesN = svm_vars.iloc[0]['numShufflesN']
                    av_train_data_all = np.full((len(population_sizes_to_try), numShufflesN, numFrames), np.nan)
                    sd_train_data_all = np.full((len(population_sizes_to_try), numShufflesN, numFrames), np.nan)               
                    av_test_data_all = np.full((len(population_sizes_to_try), numShufflesN, numFrames), np.nan)
                    sd_test_data_all = np.full((len(population_sizes_to_try), numShufflesN, numFrames), np.nan)                    
                    av_test_shfl_all = np.full((len(population_sizes_to_try), numShufflesN, numFrames), np.nan)
                    sd_test_shfl_all = np.full((len(population_sizes_to_try), numShufflesN, numFrames), np.nan)                    
                    av_test_chance_all = np.full((len(population_sizes_to_try), numShufflesN, numFrames), np.nan)
                    sd_test_chance_all = np.full((len(population_sizes_to_try), numShufflesN, numFrames), np.nan)

                    for i_pop_size in range(len(population_sizes_to_try)):
                        for inN in range(numShufflesN):
                            # For each neuron subsample, compute average and st error of class accuracies across CV samples
                            av_train_data_all[i_pop_size][inN] = 100-np.nanmean(perClassErrorTrain_data_allFrs[i_pop_size][inN], axis=0) # numFrames
                            sd_train_data_all[i_pop_size][inN] = np.nanstd(perClassErrorTrain_data_allFrs[i_pop_size][inN], axis=0) / np.sqrt(numSamples)  
                        
                            av_test_data_all[i_pop_size][inN] = 100-np.nanmean(perClassErrorTest_data_allFrs[i_pop_size][inN], axis=0) # numFrames
                            sd_test_data_all[i_pop_size][inN] = np.nanstd(perClassErrorTest_data_allFrs[i_pop_size][inN], axis=0) / np.sqrt(numSamples)  
                            
                            av_test_shfl_all[i_pop_size][inN] = 100-np.nanmean(perClassErrorTest_shfl_allFrs[i_pop_size][inN], axis=0) # numFrames
                            sd_test_shfl_all[i_pop_size][inN] = np.nanstd(perClassErrorTest_shfl_allFrs[i_pop_size][inN], axis=0) / np.sqrt(numSamples)  
                            
                            av_test_chance_all[i_pop_size][inN] = 100-np.nanmean(perClassErrorTest_chance_allFrs[i_pop_size][inN], axis=0) # numFrames
                            sd_test_chance_all[i_pop_size][inN] = np.nanstd(perClassErrorTest_chance_allFrs[i_pop_size][inN], axis=0) / np.sqrt(numSamples)  
                    
                    ######### Average across *neuron* subsamples (already averaged across cv samples) #########
                    av_train_data = np.mean(av_train_data_all, axis=1).squeeze() # numFrames
                    sd_train_data = np.std(av_train_data_all, axis=1).squeeze() / np.sqrt(numShufflesN)
                    av_test_data = np.mean(av_test_data_all, axis=1).squeeze() # numFrames
                    sd_test_data = np.std(av_test_data_all, axis=1).squeeze() / np.sqrt(numShufflesN)                    
                    av_test_shfl = np.mean(av_test_shfl_all, axis=1).squeeze() # numFrames
                    sd_test_shfl = np.std(av_test_shfl_all, axis=1).squeeze() / np.sqrt(numShufflesN)
                    av_test_chance = np.mean(av_test_chance_all, axis=1).squeeze() # numFrames
                    sd_test_chance = np.std(av_test_chance_all, axis=1).squeeze() / np.sqrt(numShufflesN)

                    # average across trials and neuron subsamples
                    av_w_data = np.nanmean(w_data_allFrs, axis=(0,1,2)) # nN_trainSVM x nFrames
                    av_b_data = np.nanmean(b_data_allFrs, axis=(0,1,2)) # nFrames
                
                    if np.ndim(av_w_data)==1: #n_neurons==1: # make sure all experiments have dimensions nNeurons x nFrames
                        av_w_data = av_w_data[np.newaxis,:]
                    
                    
                else: # average across cv samples # size perClassErrorTrain_data_allFrs: nSamples x nFrames

                    av_train_data = 100-np.nanmean(perClassErrorTrain_data_allFrs, axis=0) # numFrames
                    sd_train_data = np.nanstd(perClassErrorTrain_data_allFrs, axis=0) / np.sqrt(numSamples)  
                
                    av_test_data = 100-np.nanmean(perClassErrorTest_data_allFrs, axis=0) # numFrames
                    sd_test_data = np.nanstd(perClassErrorTest_data_allFrs, axis=0) / np.sqrt(numSamples)  
                    
                    av_test_shfl = 100-np.nanmean(perClassErrorTest_shfl_allFrs, axis=0) # numFrames
                    sd_test_shfl = np.nanstd(perClassErrorTest_shfl_allFrs, axis=0) / np.sqrt(numSamples)  
                    
                    av_test_chance = 100-np.nanmean(perClassErrorTest_chance_allFrs, axis=0) # numFrames
                    sd_test_chance = np.nanstd(perClassErrorTest_chance_allFrs, axis=0) / np.sqrt(numSamples)  
            
                    av_w_data = np.nanmean(w_data_allFrs, axis=0) # nNeurons x nFrames
                    av_b_data = np.nanmean(b_data_allFrs, axis=0) # nFrames
                
                    if n_neurons==1: # make sure all experiments have dimensions nNeurons x nFrames
                        av_w_data = av_w_data[np.newaxis,:]
                        
                        
                ########################################################################################################    
                ########################################################################################################    
                ########################################################################################################
                ########################################################################################################                            
                #%% Quantify flash and omission evoked responses
                ########################################################################################################    
                ########################################################################################################    
                ########################################################################################################
                ########################################################################################################                
                
                # Flash_index: for flash-evoked peak timing, compute it relative to flash index: 31 
                # although below is not quite accurate, if you want to be quite accurate you should align the traces on flashes!
                # on 5/11/2020 I changed np.floor (below) to np.round, to make things consistent with set_frame_window_flash_omit... also i think it is more accurate to use round than floor!
                flash_index = samps_bef + np.round(-.75 / frame_dur).astype(int)
                ### try this more accurate measure of flash_index:
                # below is commented because it seems omission times are all logged 1 frame off, so flash-omission durations are 734ms (instead of 750ms), so it is more accurate to go with the code above!
#                 flash_index = samps_bef - np.median(flash_omit_dur_fr_all)
                
                bl_index_pre_omit = np.arange(0,samps_bef) # you will need for response amplitude quantification, even if you dont use it below.
                bl_index_pre_flash = np.arange(0,flash_index)
                
                
                # concatenate CA traces for training, testing, shfl, and chance data, each a column in the matrix below; to compute their peaks all at once.
                CA_traces = np.vstack((av_train_data, av_test_data, av_test_shfl, av_test_chance)).T # times x 4
                
                bl_preOmit = np.percentile(CA_traces[bl_index_pre_omit], bl_percentile, axis=0) # neurons # use the 10th percentile
                bl_preFlash = np.percentile(CA_traces[bl_index_pre_flash], bl_percentile, axis=0) # neurons



                #%% Set flash_win: use a different window for computing flash responses if it is VIP, and B1 session (1st novel session)    
                session_novel = is_session_novel(dir_server_me, mouse, date) # first determine if the session is the 1st novel session or not

                # vip, except for B1, needs a different timewindow for computing image-triggered average (because its response precedes the image).
                # sst, slc, and vip B1 sessions all need a timewindow that immediately follows the images.
                if np.logical_and(cre.find('Vip')==1 , session_novel) or cre.find('Vip')==-1: # VIP B1, SST, SLC: window after images (response follows the image)
                    flash_win_final = flash_win # [0, .75]

                else: # VIP (non B1, familiar sessions): window precedes the image (pre-stimulus ramping activity)
                    flash_win_final = flash_win_vip # [-.25, .5] 
                
#                 if np.logical_and(cre.find('Vip')==1 , session_novel): # VIP, B1
#                     flash_win_final = flash_win_vip # [-.25, .5] 

#                 else: # SST, SLC, VIP (non B1)
#                     flash_win_final = flash_win # [0, .75]


                peak_amp_omit_trainTestShflChance, peak_timing_omit_trainTestShflChance, \
                    peak_amp_flash_trainTestShflChance, peak_timing_flash_trainTestShflChance, peak_allTrsNs, peak_timing_allTrsNs, peak_om_av_h1, peak_om_av_h2, auc_peak_h1_h2 = \
                        omissions_traces_peaks_quantify(CA_traces, bl_preOmit, bl_preFlash, mean_notPeak, cre, peak_win, flash_win_final, flash_win_final, flash_index, samps_bef, frame_dur, doShift_again, doPeakPerTrial, doROC, doPlots, index)
                    
                
#                 peak_amp_omit_trainTestShflChance, peak_timing_omit_trainTestShflChance, \
#                     peak_amp_flash_trainTestShflChance, peak_timing_flash_trainTestShflChance, peak_allTrsNs, peak_timing_allTrsNs, peak_om_av_h1, peak_om_av_h2, auc_peak_h1_h2 = \
#                         omissions_traces_peaks_quantify(CA_traces, bl_preOmit, bl_preFlash, mean_notPeak, cre, peak_win, flash_win, flash_win_timing, flash_index, samps_bef, frame_dur, doShift_again, doPeakPerTrial, doROC, doPlots, index)
                
                
                this_sess.at[index, ['peak_amp_omit_trainTestShflChance']] = [peak_amp_omit_trainTestShflChance] # 4
                this_sess.at[index, ['peak_timing_omit_trainTestShflChance']] = [peak_timing_omit_trainTestShflChance] # 4
                this_sess.at[index, ['peak_amp_flash_trainTestShflChance']] = [peak_amp_flash_trainTestShflChance] # 4
                this_sess.at[index, ['peak_timing_flash_trainTestShflChance']] = [peak_timing_flash_trainTestShflChance] # 4

                
            
            
            
                #%% Interpolate the class accur traces every 3ms (instead of the original one that is 93ms)
                
                kind = 'linear' #'quadratic' # 
            
                av_train_data_new, xnew, x = interp_imaging(av_train_data, samps_bef, samps_aft, kind) # x and x_new: time (sec) 
                av_test_data_new, _, _ = interp_imaging(av_test_data, samps_bef, samps_aft, kind)
                av_test_shfl_new, _, _ = interp_imaging(av_test_shfl, samps_bef, samps_aft, kind)
                av_test_chance_new, _, _ = interp_imaging(av_test_chance, samps_bef, samps_aft, kind)
            
                sd_train_data_new, xnew, x = interp_imaging(sd_train_data, samps_bef, samps_aft, kind)
                sd_test_data_new, _, _ = interp_imaging(sd_test_data, samps_bef, samps_aft, kind)
                sd_test_shfl_new, _, _ = interp_imaging(sd_test_shfl, samps_bef, samps_aft, kind)
                sd_test_chance_new, _, _ = interp_imaging(sd_test_chance, samps_bef, samps_aft, kind)
                
                av_w_data_new, _, _ = interp_imaging(av_w_data, samps_bef, samps_aft, kind)
                av_b_data_new, _, _ = interp_imaging(av_b_data, samps_bef, samps_aft, kind)
                
                meanX_allFrs_new = np.full((len(xnew), n_neurons), np.nan)
                stdX_allFrs_new = np.full((len(xnew), n_neurons), np.nan)
                for ine in range(meanX_allFrs.shape[1]):
                    meanX_allFrs_new[:, ine], _, _ = interp_imaging(meanX_allFrs[:,ine], samps_bef, samps_aft, kind)
                    stdX_allFrs_new[:, ine], _, _ = interp_imaging(stdX_allFrs[:,ine], samps_bef, samps_aft, kind)
                
                
                
#                this_sess.at[index, ['n_omissions', 'n_neurons', 'meanX_allFrs', 'stdX_allFrs']] = [n_omissions, n_neurons, meanX_allFrs, stdX_allFrs] 
                this_sess.at[index, ['n_omissions', 'n_neurons', 'frame_dur', 'flash_omit_dur_all', 'flash_omit_dur_fr_all', 'meanX_allFrs_new', 'stdX_allFrs_new']] = [n_omissions, n_neurons, frame_dur, flash_omit_dur_all, flash_omit_dur_fr_all, meanX_allFrs_new, stdX_allFrs_new] 
                this_sess.at[index, ['av_train_data_new', 'av_test_data_new', 'av_test_shfl_new', 'av_test_chance_new', 'sd_train_data_new', 'sd_test_data_new', 'sd_test_shfl_new', 'sd_test_chance_new']] = [av_train_data_new, av_test_data_new, av_test_shfl_new, av_test_chance_new, sd_train_data_new, sd_test_data_new, sd_test_shfl_new, sd_test_chance_new]
                
                if same_num_neuron_all_planes:
                    this_sess.at[index, ['population_sizes_to_try']] = population_sizes_to_try
                else:
                    this_sess.at[index, ['av_w_data_new', 'av_b_data_new']] = [av_w_data_new, av_b_data_new]


                    
                    
                ########################################################################################################    
                ########################################################################################################    
                ########################################################################################################
                ########################################################################################################                
                #%% Plots
                ########################################################################################################    
                ########################################################################################################    
                ########################################################################################################
                ########################################################################################################                
                #%%
                if doPlots==1:
                    get_ipython().magic(u'matplotlib inline')                
                    flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, flashes_win_trace_index_unq, grays_win_trace_index_unq = \
                        flash_gray_onset_relOmit(samps_bef, samps_aft, frame_dur)

                            
                #%% Plot the mean and std of each neuron (across trials: half omission, half gray)
                
                if doPlots:
                    plt.figure(figsize=(4.8,6))
                    # mean
                    plt.subplot(211)
                    plt.plot(frames_svm, meanX_allFrs)
                    plt.title('meanX (half omit & half gray trials), eachN')
                    
                    mn = np.min(meanX_allFrs)
                    mx = np.max(meanX_allFrs)
                    plt.vlines([0], mn, mx, color='k', linestyle='--')     # mark omission onset
                    plt.vlines(flashes_win_trace_index_unq, mn, mx, color='y', linestyle='-.')    # mark the onset of flashes
                    plt.vlines(grays_win_trace_index_unq, mn, mx, color='gray', linestyle=':')    # mark the onset of grays
                
                    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
                    seaborn.despine()    
                    
                    
                    ## std
                    plt.subplot(212)
                    plt.plot(frames_svm, stdX_allFrs)
                    plt.title('stdX (half omit & half gray trials), eachN')
                    plt.subplots_adjust(hspace=.55)
                
                    mn = np.min(stdX_allFrs)
                    mx = np.max(stdX_allFrs)
                    plt.vlines([0], mn, mx, color='k', linestyle='--')     # mark omission onset
                    plt.vlines(flashes_win_trace_index_unq, mn, mx, color='y', linestyle='-.')    # mark the onset of flashes
                    plt.vlines(grays_win_trace_index_unq, mn, mx, color='gray', linestyle=':')    # mark the onset of grays
                    plt.xlabel('Frames rel. omission')
                    
                    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
                    seaborn.despine()
                
                
                #%% Make plots of classification accuracy 
                
                ### MAKE SURE to print number of neurons and number of trials in your plots
                # some experiments (superficial layers) have only 1 neuron!!
                
                ### different planes have different number of neurons... so we cannot directly compare svm performance across planes!!!
                # unless we subselect neurons!!
                
                if doPlots:
                    plt.figure()  
                
                    h0 = plt.plot(x, av_train_data,'ko', xnew, av_train_data_new,'k-', label='train', markersize=3.5);
                    h1 = plt.plot(x, av_test_data,'ro', xnew, av_test_data_new,'r-', label='test', markersize=3.5);
                    h2 = plt.plot(x, av_test_shfl,'yo', xnew, av_test_shfl_new,'y-', label='shfl', markersize=3.5);
                    h3 = plt.plot(x, av_test_chance,'bo', xnew, av_test_chance_new,'b-',  label='chance', markersize=3.5);
                    
                    # errorbars (standard error across cv samples)   
                    alph = .3
                    plt.fill_between(xnew, av_train_data_new - sd_train_data_new, av_train_data_new + sd_train_data_new, alpha=alph, edgecolor='k', facecolor='k')
                    plt.fill_between(xnew, av_test_data_new - sd_test_data_new, av_test_data_new + sd_test_data_new, alpha=alph, edgecolor='r', facecolor='r')
                    plt.fill_between(xnew, av_test_shfl_new - sd_test_shfl_new, av_test_shfl_new + sd_test_shfl_new, alpha=alph, edgecolor='y', facecolor='y')
                    plt.fill_between(xnew, av_test_chance_new - sd_test_chance_new, av_test_chance_new + sd_test_chance_new, alpha=alph, edgecolor='b', facecolor='b')
                
                
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
                    '''

        #%%    
#        all_sess = all_sess.append(this_sess) # all_sess is initiated in svm_init.py           
    
        if set_to_nan==1:
            x, xnew = upsample_time_imaging(samps_bef, samps_aft, 31.)
    
            av_train_data_new = np.full((len(xnew)), np.nan)
            av_test_data_new = np.full((len(xnew)), np.nan)
            av_test_shfl_new = np.full((len(xnew)), np.nan)
            av_test_chance_new = np.full((len(xnew)), np.nan)
            sd_train_data_new = np.full((len(xnew)), np.nan)
            sd_test_data_new = np.full((len(xnew)), np.nan)
            sd_test_shfl_new = np.full((len(xnew)), np.nan)
            sd_test_chance_new = np.full((len(xnew)), np.nan)
            
            meanX_allFrs_new = np.full((len(xnew)), np.nan) # np.full((len(xnew), n_neurons), np.nan)
            stdX_allFrs_new = np.full((len(xnew)), np.nan) # np.full((len(xnew), n_neurons), np.nan)            
            
            av_w_data_new = np.full((1, len(xnew)), np.nan)
            av_b_data_new = np.full((len(xnew)), np.nan)
            
            this_sess.at[index, ['meanX_allFrs_new', 'stdX_allFrs_new', 'av_train_data_new', 'av_test_data_new', 'av_test_shfl_new', 'av_test_chance_new', 'sd_train_data_new', 'sd_test_data_new', 'sd_test_shfl_new', 'sd_test_chance_new']] = \
                [meanX_allFrs_new, stdX_allFrs_new, av_train_data_new, av_test_data_new, av_test_shfl_new, av_test_chance_new, sd_train_data_new, sd_test_data_new, sd_test_shfl_new, sd_test_chance_new]

            this_sess.at[index, ['peak_amp_omit_trainTestShflChance']] = [np.full((4), np.nan)] # 4
            this_sess.at[index, ['peak_timing_omit_trainTestShflChance']] = [np.full((4), np.nan)] # 4
            this_sess.at[index, ['peak_amp_flash_trainTestShflChance']] = [np.full((4), np.nan)] # 4
            this_sess.at[index, ['peak_timing_flash_trainTestShflChance']] = [np.full((4), np.nan)] # 4
            
            if same_num_neuron_all_planes:
                this_sess.at[index, ['population_sizes_to_try']] = np.nan
            else:
                this_sess.at[index, ['av_w_data_new', 'av_b_data_new']] = [av_w_data_new, av_b_data_new]


    return this_sess            
                
                



            