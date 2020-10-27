#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function is called in svm_init_images.

It loads the svm file that was saved for each experiment, and returns a pandas dataframe (this_sess), which includes a number of columns, including average and st error of class accuracies across CV samples; also quantification of the image signal.
svm_init combines this_sess for all sessions into a single pandas table (all_sess), and saves it at /allen/programs/braintv/workgroups/nc-ophys/Farzaneh/SVM/same_num_neurons_all_planes/all_sess_svm_images*

Follow this function by svm_images_plots_setVars.py to make plots.

Created on Tue Oct  13 20:53:49 2019
@author: farzaneh
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from general_funs import *

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def svm_main_images_post(session_id, data_list, dir_svm, frames_svm, time_win, trial_type, to_decode, same_num_neuron_all_planes, cols, analysis_dates, doPlots=0):
    
    num_classes = 8 # decoding 8 images
    num_planes = 8
    svmn = f'svm_decode_{to_decode}_image_from_{trial_type}' # 'svm_images' # 'svm_gray_omit'
    
#     if same_num_neuron_all_planes:
#         svmn = svmn + '_sameNumNeuronsAllPlanes'
        
    exp_ids = data_list['experiment_id'].values
#     frame_dur = np.array([0.093]) # sec (~10.7 Hz; each pair of planes that are recorded simultaneously have time resolution frame_dur)    
    
    frames_svm = np.array(frames_svm)
    numFrames = len(frames_svm)
    
    
    #%% Loop through the 8 planes of each session
    
    this_sess = pd.DataFrame([], columns = cols)
    
    for index, lims_id in enumerate(exp_ids): 

        '''
        for il in [5]: #range(num_planes):
            index = il
            lims_id = exp_ids[il]
        '''            
        '''
        ll = list(enumerate(exp_ids)); 
        l = ll[0]; # first plane
        index = l[0]; # plane index
        lims_id = l[1] # experiment id 
        '''
        
        print('\n======================== Analyzing experiment %s, plane %d/%d ========================\n' %(lims_id, index, num_planes))
        
        area = data_list.iloc[index]['area']
        depth = int(data_list.iloc[index]['depth'])
        valid = data_list.iloc[index]['valid']
    
        this_sess.at[index, ['session_id', 'experiment_id', 'area', 'depth']] = session_id, lims_id, area, depth
        

        if valid==False: 
            print('Skipping invalid experiment %d' %int(lims_id))
#            this_sess.at[index, :] = np.nan # check this works.
#            sys.exit()
            set_to_nan = 1  # set svm vars to nan      
    
        else:
            #%% Set the h5 filename containing SVM vars
            
#             cre_now = cre[:cre.find('-')]

            # mouse, session, experiment: m, s, e
            if analysis_dates[0] == '':
                nown = '.'
#                 name = '(.*)_s-%d_e-%s_%s_frames%dto%d_.' %(session_id, lims_id, svmn, frames_svm[0], frames_svm[-1])
            else:
                nown = f'{analysis_dates[0]}_.'
#                 name = '(.*)_s-%d_e-%s_%s_frames%dto%d_%s_.' %(session_id, lims_id, svmn, frames_svm[0], frames_svm[-1], analysis_dates[0])
            
            if same_num_neuron_all_planes:
                nown = 'sameNumNeuronsAllPlanes_' + nown
            name = f'(.*)_s-{session_id}_e-{lims_id}_{svmn}_frames{frames_svm[0]}to{frames_svm[-1]}_{nown}'
            
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
               'area', 'depth', 'n_trials', 'n_neurons', 'thAct', 'numSamples',
               'softNorm', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs',
               'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
               'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
               'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs',
               'testTrInds_allSamps_allFrs', 'Ytest_allSamps_allFrs',
               'Ytest_hat_allSampsFrs_allFrs'
            '''

            cre = svm_vars.iloc[0]['cre']
            date = svm_vars.iloc[0]['date']
            mouse = svm_vars.iloc[0]['mouse_id']
            stage = svm_vars.iloc[0]['stage']

            this_sess.at[index, ['mouse_id', 'date', 'cre', 'stage']] = mouse, date, cre, stage
            

            n_trials = svm_vars.iloc[0]['n_trials']
            n_neurons = svm_vars.iloc[0]['n_neurons']
            print(f'{n_trials} trials; {n_neurons} neurons')

            if np.in1d(svm_vars.keys(), 'frame_dur').any(): # svm code was run without setting frame_dur at first.
                frame_dur = svm_vars.iloc[0]['frame_dur']                
                print(f'Frame duration {frame_dur} ms')
                frame_dur = np.round(frame_dur, 3)
#                 if np.logical_or(frame_dur < .089, frame_dur > .1):
#                     sys.exit(f'\n\nFrame duration is unexpected!! {frame_dur}ms\n\n')
            else: # for mesoscope data
                frame_dur = np.array([0.093]) # sec # mesoscope time resolution (4 depth, 2 areas) (~10.7 Hz; each pair of planes that are recorded simultaneously have time resolution frame_dur)



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

            w_data_allFrs = svm_vars.iloc[0]['w_data_allFrs'] # numSamps x nNeurons x nFrames, if n_classes=2; # samps x classes x neurons x nFrames, if n_classes>2
            b_data_allFrs = svm_vars.iloc[0]['b_data_allFrs'] # numSamps x nFrames, if n_classes=2; # samps x classes x nFrames, if n_classes>2

            frames_svm = svm_vars.iloc[0]['frames_svm']
#             numFrames = len(frames_svm)

        #    plt.plot(cbest_allFrs)
        #    plt.plot(sorted(cbest_allFrs)[:-2])


            ########################################################################################################
            ########################################################################################################                                
            #%% Average and st error of class accuracies across cross-validation samples
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

            else: # average across cv samples # size perClassErrorTrain_data_allFrs: nSamples x nFrames

                av_train_data = 100-np.nanmean(perClassErrorTrain_data_allFrs, axis=0) # numFrames
                sd_train_data = np.nanstd(perClassErrorTrain_data_allFrs, axis=0) / np.sqrt(numSamples)  

                av_test_data = 100-np.nanmean(perClassErrorTest_data_allFrs, axis=0) # numFrames
                sd_test_data = np.nanstd(perClassErrorTest_data_allFrs, axis=0) / np.sqrt(numSamples)  

                av_test_shfl = 100-np.nanmean(perClassErrorTest_shfl_allFrs, axis=0) # numFrames
                sd_test_shfl = np.nanstd(perClassErrorTest_shfl_allFrs, axis=0) / np.sqrt(numSamples)  

                av_test_chance = 100-np.nanmean(perClassErrorTest_chance_allFrs, axis=0) # numFrames
                sd_test_chance = np.nanstd(perClassErrorTest_chance_allFrs, axis=0) / np.sqrt(numSamples)  

                av_w_data = np.nanmean(w_data_allFrs, axis=0) # nNeurons x nFrames, if n_classes=2; # classes x neurons x nFrames, if n_classes>2
                av_b_data = np.nanmean(b_data_allFrs, axis=0) # nFrames, if n_classes=2; # classes x nFrames, if n_classes>2

                if n_neurons==1: # make sure all experiments have dimensions nNeurons x nFrames
                    av_w_data = av_w_data[np.newaxis,:]


            ########################################################################################################
            ########################################################################################################                            
            #%% Quantify image signal by averaging the class accuracy traces in the window time_win
            ########################################################################################################    
            ########################################################################################################    

            # in the svm code used for flash/omission vs. gray, we would compute the baseline and subtract it out.
            # here, we are just taking the average of the 750ms time points after the image

            # NOTE: if you end up using events in the decoding analysis, you should add the baseline parts of the code back. 
            
            if np.in1d([-1,1], np.sign(frames_svm)).all(): # frames_svm includes both negative and positive values; ie svm was trained on some frames before and some frames after the trial onset.
                samps_bef_here = -frames_svm[0] # relative to svm output traces (eg av_train_data)
            else:
                samps_bef_here = 0
                
            if type(time_win)==str: # time_win = 'frames_svm' means : use frames_svm to quantify image signal
                time_win_frames = frames_svm + samps_bef_here # these indices will be applied to svm trace outputs.
            else:
                time_win_frames = set_frame_window_flash_omit(time_win, samps_bef_here, frame_dur)
                time_win_frames[np.in1d(time_win_frames, frames_svm + samps_bef_here)]                
            
            print(f'frames_svm: {frames_svm}')
            print(f'time_win_frames for quantification: {time_win_frames}')
            

            # concatenate CA traces for training, testing, shfl, and chance data, each a column in the matrix below; to compute their peaks all at once.
            CA_traces = np.vstack((av_train_data, av_test_data, av_test_shfl, av_test_chance)).T # times x 4
#             print(CA_traces.shape)

            peak_amp_trainTestShflChance = np.nanmean(CA_traces[time_win_frames], axis=0)


            this_sess.at[index, ['peak_amp_trainTestShflChance']] = [peak_amp_trainTestShflChance] # 4


            this_sess.at[index, ['n_trials', 'n_neurons', 'frame_dur', 'meanX_allFrs', 'stdX_allFrs']] = [n_trials, n_neurons, frame_dur, meanX_allFrs, stdX_allFrs] 
            this_sess.at[index, ['av_train_data', 'av_test_data', 'av_test_shfl', 'av_test_chance', 'sd_train_data', 'sd_test_data', 'sd_test_shfl', 'sd_test_chance']] = [av_train_data, av_test_data, av_test_shfl, av_test_chance, sd_train_data, sd_test_data, sd_test_shfl, sd_test_chance]

            if same_num_neuron_all_planes:
                this_sess.at[index, ['population_sizes_to_try']] = population_sizes_to_try
            else:
                this_sess.at[index, 'av_w_data'] = av_w_data
                this_sess.at[index, 'av_b_data'] = av_b_data
#                     this_sess.at[index, ['av_w_data', 'av_b_data']] = [av_w_data, av_b_data]



            ########################################################################################################
            ########################################################################################################                
            #%% Plots
            ########################################################################################################    
            ########################################################################################################    
            #%%
            if doPlots==1:
                get_ipython().magic(u'matplotlib inline')                
#                     flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, flashes_win_trace_index_unq, grays_win_trace_index_unq = \
#                         flash_gray_onset_relOmit(samps_bef, samps_aft, frame_dur)


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
#                     plt.vlines(flashes_win_trace_index_unq, mn, mx, color='y', linestyle='-.')    # mark the onset of flashes
#                     plt.vlines(grays_win_trace_index_unq, mn, mx, color='gray', linestyle=':')    # mark the onset of grays

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
#                     plt.vlines(flashes_win_trace_index_unq, mn, mx, color='y', linestyle='-.')    # mark the onset of flashes
#                     plt.vlines(grays_win_trace_index_unq, mn, mx, color='gray', linestyle=':')    # mark the onset of grays
                plt.xlabel('Frames rel. image onset')

                plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
                seaborn.despine()


            #%% Make plots of classification accuracy 

            ### MAKE SURE to print number of neurons and number of trials in your plots
            # some experiments (superficial layers) have only 1 neuron!!

            ### different planes have different number of neurons... so we cannot directly compare svm performance across planes!!!
            # unless we subselect neurons!!

            if doPlots:

                if (np.sign(frames_svm)==-1).all(): # when frames_svm is all negative
                    x = np.arange(frames_svm[0] * frame_dur, frames_svm[-1] * frame_dur, frame_dur)
                else: 
#                     samps_bef_here = -frames_svm[0] # relative to svm output traces (eg av_train_data)
                    samps_aft_here = frames_svm[-1]+1 

                    samps_bef_time = (samps_bef_here+1) * frame_dur # 1 is added bc below we do np.arange(0,-samps_bef), so we get upto one value below samps_bef
                    samps_aft_time = samps_aft_here * frame_dur # frames_after_omission in svm_main # we trained the classifier until 30 frames after omission                    
                    
                    x = np.unique(np.concatenate((np.arange(0, -samps_bef_time, -frame_dur)[0:samps_bef_here+1], np.arange(0, samps_aft_time, frame_dur)[0:samps_aft_here])))
                                    
                import seaborn
                import matplotlib.ticker as ticker                    

                ###
                plt.figure()  

                h0 = plt.plot(x, av_train_data,'ko-', label='train', markersize=3.5);
                h1 = plt.plot(x, av_test_data,'ro-', label='test', markersize=3.5);
                h2 = plt.plot(x, av_test_shfl,'yo-', label='shfl', markersize=3.5);
                h3 = plt.plot(x, av_test_chance,'bo-',  label='chance', markersize=3.5);

                # errorbars (standard error across cv samples)   
                alph = .3
                plt.fill_between(x, av_train_data - sd_train_data, av_train_data + sd_train_data, alpha=alph, edgecolor='k', facecolor='k')
                plt.fill_between(x, av_test_data - sd_test_data, av_test_data + sd_test_data, alpha=alph, edgecolor='r', facecolor='r')
                plt.fill_between(x, av_test_shfl - sd_test_shfl, av_test_shfl + sd_test_shfl, alpha=alph, edgecolor='y', facecolor='y')
                plt.fill_between(x, av_test_chance - sd_test_chance, av_test_chance + sd_test_chance, alpha=alph, edgecolor='b', facecolor='b')


#                 mn = 45; mx = 80
                # mark omission onset
#                 plt.vlines([0], mn, mx, color='k', linestyle='--')
                # mark the onset of flashes
#                     plt.vlines(flashes_win_trace_index_unq_time, mn, mx, color='y', linestyle='-.')
                # mark the onset of grays
#                     plt.vlines(grays_win_trace_index_unq_time, mn, mx, color='gray', linestyle=':')


                ax = plt.gca()
                xmj = np.round(np.arange(x[0], x[-1], .1), 3)
                ax.set_xticks(xmj); # plt.xticks(np.arange(0,x[-1],.25)); #, fontsize=10)
                ax.set_xticklabels(xmj, rotation=45)       
                xmn = np.round(np.arange(x[0]+.5, x[-1], .1), 3)
                ax.xaxis.set_minor_locator(ticker.FixedLocator(xmn))
                ax.tick_params(labelsize=10, length=6, width=2, which='major')
                ax.tick_params(labelsize=10, length=5, width=1, which='minor')
            #    plt.xticklabels(np.arange(0,x[-1],.25))


                plt.legend(handles=[h0[0],h1[0],h2[0],h3[0]], loc='center left', bbox_to_anchor=(1, .7), frameon=False, handlelength=.5, fontsize=12)
                plt.ylabel('% Classification accuracy', fontsize=12)
                plt.xlabel('Time rel. image onset (sec)', fontsize=12)  
                plt.title(f'{cre[:3]}, {area}, {depth}')

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
#             x, xnew = upsample_time_imaging(samps_bef, samps_aft, 31.)
    
            n_neurons = 1 # just an arbitrary number so we can set the arrays below
    
            av_train_data = np.full((numFrames), np.nan)
            av_test_data = np.full((numFrames), np.nan)
            av_test_shfl = np.full((numFrames), np.nan)
            av_test_chance = np.full((numFrames), np.nan)
            sd_train_data = np.full((numFrames), np.nan)
            sd_test_data = np.full((numFrames), np.nan)
            sd_test_shfl = np.full((numFrames), np.nan)
            sd_test_chance = np.full((numFrames), np.nan)
            
            meanX_allFrs = np.full((numFrames, n_neurons), np.nan) # np.full((numFrames), np.nan) # 
            stdX_allFrs = np.full((numFrames, n_neurons), np.nan) # np.full((numFrames), np.nan) #             
            
            if num_classes>2:
                av_w_data = np.full((num_classes, n_neurons, numFrames), np.nan)
                av_b_data = np.full((num_classes, numFrames), np.nan)
            else:
                av_w_data = np.full((n_neurons, numFrames), np.nan)
                av_b_data = np.full((numFrames), np.nan)

            
            this_sess.at[index, ['meanX_allFrs', 'stdX_allFrs']] = [meanX_allFrs, stdX_allFrs]
            this_sess.at[index, ['av_train_data', 'av_test_data', 'av_test_shfl', 'av_test_chance', 'sd_train_data', 'sd_test_data', 'sd_test_shfl', 'sd_test_chance']] = \
                [av_train_data, av_test_data, av_test_shfl, av_test_chance, sd_train_data, sd_test_data, sd_test_shfl, sd_test_chance]
            this_sess.at[index, ['peak_amp_trainTestShflChance']] = [np.full((4), np.nan)] # 4
            
            if same_num_neuron_all_planes:
                this_sess.at[index, ['population_sizes_to_try']] = np.nan
            else:
                this_sess.at[index, 'av_w_data'] = av_w_data
                this_sess.at[index, 'av_b_data'] = av_b_data


    return this_sess            
                
                



            