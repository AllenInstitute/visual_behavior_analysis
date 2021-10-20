#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gets called in svm_images_main_pre_pbs.py

Performs SVM analysis and saves the results in: dir_svm = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'.
If using same_num_neuron_all_planes, results will be saved in: dir_svm/ 'same_num_neurons_all_planes'

Once the analysis is done and svm results are saved, run svm_images_plots_init.py to set vars for making plots.


Created on Fri Aug  2 15:24:17 2019
@author: farzaneh
"""

#%%
import os
import datetime
import re
import pandas as pd
import copy
import sys
import numpy.random as rnd
from svm_funs import *
# get_ipython().magic(u'matplotlib inline') # %matplotlib inline


#%%
def svm_images_main_pbs(session_id, data_list, experiment_ids_valid, df_data, session_trials, trial_type, dir_svm, kfold, frames_svm, numSamples, saveResults, cols_basic, cols_svm, project_codes, to_decode='current', svm_blocks=-100, engagement_pupil_running = np.nan, use_events=False, same_num_neuron_all_planes=0, use_balanced_trials=0, use_spont_omitFrMinus1=1, use_matched_cells=0):
    
    def svm_run_save(traces_fut_now, image_labels_now, iblock_trials_blocks, svm_blocks, now, engagement_pupil_running, pupil_running_values, use_balanced_trials, project_codes): #, same_num_neuron_all_planes, norm_to_max_svm, svm_total_frs, n_neurons, numSamples, num_classes, samps_bef, regType, kfold, cre, saveResults):
        '''
        traces_fut_now = traces_fut_0
        image_labels_now = image_labels_0
        
        iblock_trials_blocks = [np.nan, []]
        
        iblock_trials_blocks = [iblock, trials_blocks]
        '''
        
        iblock = iblock_trials_blocks[0]
        trials_blocks = iblock_trials_blocks[1]
        
        #%% Normalize each neuron trace by its max (so if on a day a neuron has low FR in general, it will not be read as non responsive!)
        # we dont need this because we are z scoring the traces.
        
        traces_fut_orig = copy.deepcopy(traces_fut_now)

        if 0: #norm_to_max_svm==1:
            print('Normalizing each neuron trace by its max.')
            # compute max on the entire trace of the session:
            aa_mx = [np.max(traces_fut_orig[:,i,:].flatten()) for i in range(n_neurons)] # neurons
            if (np.array(aa_mx)==0).any():
                print(aa_mx)
                print('\n\nSome neurons have max=0; dividing will lead to NaN! STOP!\n\n')
            else:
                a = np.transpose(traces_fut_orig, (0,2,1)) # frames x trials x units
                b = a / aa_mx
                traces_fut_now = np.transpose(b, (0,2,1)) # frames x units x trials



        #%% Starting to set variables for the SVM analysis                

        meanX_allFrs = np.full((svm_total_frs, n_neurons), np.nan)
        stdX_allFrs = np.full((svm_total_frs, n_neurons), np.nan)        

        if same_num_neuron_all_planes:
            nnt = n_neurons #X_svm[0]
            numShufflesN = np.ceil(nnt/float(nN_trainSVM)).astype(int) # if you are selecting only 1 neuron out of 500 neurons, you will do this 500 times to get a selection of all neurons. On the other hand if you are selecting 400 neurons out of 500 neurons, you will do this only twice.

            perClassErrorTrain_data_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, svm_total_frs), np.nan)        
            perClassErrorTest_data_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, svm_total_frs), np.nan)
            perClassErrorTest_shfl_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, svm_total_frs), np.nan)
            perClassErrorTest_chance_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, svm_total_frs), np.nan)
            if nN_trainSVM==1:
                if num_classes==2:
                    w_data_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, nN_trainSVM, svm_total_frs), np.nan).squeeze() # squeeze helps if n_neurons=1 
    #                 else:                    
            else:
                if num_classes==2:
                    w_data_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, nN_trainSVM, svm_total_frs), np.nan) # squeeze helps if n_neurons=1 
    #                 else:                    
            if num_classes==2:
                b_data_allFrs = np.full((len(population_sizes_to_try), numShufflesN, numSamples, svm_total_frs), np.nan)
    #             else:                

            cbest_allFrs = np.full((len(population_sizes_to_try), numShufflesN, svm_total_frs), np.nan)


        ##########################
        else:
            perClassErrorTrain_data_allFrs = np.full((numSamples, svm_total_frs), np.nan)        
            perClassErrorTest_data_allFrs = np.full((numSamples, svm_total_frs), np.nan)
            perClassErrorTest_shfl_allFrs = np.full((numSamples, svm_total_frs), np.nan)
            perClassErrorTest_chance_allFrs = np.full((numSamples, svm_total_frs), np.nan)
            if n_neurons==1:
                if num_classes==2:
                    w_data_allFrs = np.full((numSamples, n_neurons, svm_total_frs), np.nan).squeeze() # squeeze helps if n_neurons=1 
                else: 
                    w_data_allFrs = np.full((numSamples, num_classes, n_neurons, svm_total_frs), np.nan).squeeze()
            else:
                if num_classes==2:
                    w_data_allFrs = np.full((numSamples, n_neurons, svm_total_frs), np.nan) # squeeze helps if n_neurons=1 
                else:
                    w_data_allFrs = np.full((numSamples, num_classes, n_neurons, svm_total_frs), np.nan)

            if num_classes==2:
                b_data_allFrs = np.full((numSamples, svm_total_frs), np.nan)
            else:
                b_data_allFrs = np.full((numSamples, num_classes, svm_total_frs), np.nan)

            cbest_allFrs = np.full(svm_total_frs, np.nan)


        
        ##########################
        if trial_type == 'baseline_vs_nobaseline': # decode activity at each frame vs. baseline
            numTrials = 2*traces_fut_now.shape[2]
        else:
            numTrials = traces_fut_now.shape[2]  # numDataPoints = X_svm.shape[1] # trials             

        if num_classes==2 and use_balanced_trials==1:
            numO = sum(image_labels_now==0) 
            len_test = 2*len(np.arange(int((kfold-1.)/kfold*numO), numO))
        else:
            len_test = numTrials - int((kfold-1.)/kfold*numTrials) # number of testing trials   

        numDataPoints = numTrials
        print(f'\n{len_test} testing trials in SVM. {numTrials} total number of trials.\n')

        if len_test==1:
            Ytest_hat_allSampsFrs_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan).squeeze() # squeeze helps if len_test=1
        else:
            Ytest_hat_allSampsFrs_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan)
        Ytest_allSamps_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan)         
        testTrInds_allSamps_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan)



        #%% Use SVM to classify population activity on the gray-screen frame right before the omission vs. the activity on frame + nAftOmit after omission 

        ifr = -1
        for nAftOmit in frames_svm: # nAftOmit = frames_svm[0]

            ifr = ifr+1
            if np.isnan(iblock):
                print('\n===== Running SVM on frame %d relative to trial onset =====\n' %nAftOmit)
            else:
                print(f'\n===== BLOCK {iblock} : running SVM on frame {nAftOmit} relative to trial onset =====\n')

                
            if trial_type == 'baseline_vs_nobaseline': # decode activity at each frame vs. baseline
                # x
                if use_spont_omitFrMinus1==0: # if 0, classify omissions against spontanoues frames (the initial gray screen); if 1, classify omissions against the frame right before the omission
                    rand_spont_frs_num_omit = rnd.permutation(spont_frames.shape[1])[:num_omissions] # pick num_omissions random spontanous frames
                    g = spont_frames[:,rand_spont_frs_num_omit] # units x trials
                elif use_spont_omitFrMinus1==1:
                    g = traces_fut_now[samps_bef - 1,:,:] # units x trials ; neural activity in the frame before the omission (gray screen)
                    rand_spont_frs_num_omit = np.nan

                m = traces_fut_now[samps_bef + nAftOmit,:,:] # units x trials ; neural activity on the frame of omission

                # now set the x matrix for svm
                X_svm = np.concatenate((g, m), axis=1) # units x (gray + omission)
                
            
            else:

                # x
                m = traces_fut_now[samps_bef + nAftOmit,:,:] # units x trials ; neural activity on the frame of omission

                # now set the x matrix for svm
                X_svm = m # units x trials

            
            # y
            Y_svm = image_labels_now # trials

            print(X_svm.shape, Y_svm.shape)

            
            

            #%% Z score (make each neuron have mean 0 and std 1 across all trials)

            print('Z scoring acitivity traces!')

            # mean and std of each neuron across all trials (trials here mean both gray screen frames and omission frames)
            m = np.mean(X_svm, axis=1)
            s = np.std(X_svm, axis=1)

            meanX_allFrs[ifr,:] = m # frs x neurons
            stdX_allFrs[ifr,:] = s     

            # soft normalziation : neurons with sd<thAct wont have too large values after normalization                    
            if softNorm==1:
                # Gamal: soft normalization is needed for numerical stability (i.e., to avoid division by zero). If you are worried about introducing inaccuracies you could try to lower the threshold. You can base the threshold on the typical values you get for std. Maybe make the threshold 1e-6 lower than that; ie. if a typical std value is x then set the threshold at x*1e-6
                print(f'Using soft normalization. {thAct} will be added to std!')
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
                             fr2an=np.nan, shflTrLabs=0, X_svm_incorr=0, Y_svm_incorr=0, mnHRLR_acrossDays=np.nan, use_balanced_trials=use_balanced_trials) # outputs have size the number of shuffles in setbestc (shuffles per c value)



            ########## Set the SVM output at best C (for all samples) and keep SVM vars for all frames
#             print(np.shape(wAllC))
#             print(np.shape(w_data_allFrs))
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
                        if n_neurons==1:
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

                perClassErrorTrain_data = perClassErrorTrain[:,indBestC].squeeze() # samps
                perClassErrorTest_data = perClassErrorTest[:,indBestC].squeeze()
                perClassErrorTest_shfl = perClassErrorTestShfl[:,indBestC].squeeze()
                perClassErrorTest_chance = perClassErrorTestChance[:,indBestC].squeeze()
                w_data = wAllC[:,indBestC].squeeze() # samps x neurons, if n_classes=2; # samps x classes x neurons, if n_classes>2
                b_data = bAllC[:,indBestC].squeeze() # samps, if n_classes=2; # samps x classes, if n_classes>2
                Ytest_hat_allSampsFrs = Ytest_hat_allSampsFrs0[:,indBestC,:].squeeze()


                ########## keep SVM vars for all frames after omission 
                cbest_allFrs[ifr] = cbest

                perClassErrorTrain_data_allFrs[:,ifr] = perClassErrorTrain_data # numSamps        
                perClassErrorTest_data_allFrs[:,ifr] = perClassErrorTest_data
                perClassErrorTest_shfl_allFrs[:,ifr] = perClassErrorTest_shfl
                perClassErrorTest_chance_allFrs[:,ifr] = perClassErrorTest_chance
                if n_neurons==1:
                    if num_classes==2:
                        w_data_allFrs[:,ifr] = w_data # numSamps x neurons, if n_classes=2; # samps x classes x neurons, if n_classes>2
                    else:
                        w_data_allFrs[:,:,ifr] = w_data
                else:
                    if num_classes==2:
                        w_data_allFrs[:,:,ifr] = w_data # numSamps x neurons, if n_classes=2; # samps x classes x neurons, if n_classes>2
                    else:
                        w_data_allFrs[:,:,:,ifr] = w_data

                if num_classes==2:
                    b_data_allFrs[:,ifr] = b_data
                else:
                    b_data_allFrs[:,:,ifr] = b_data

                Ytest_hat_allSampsFrs_allFrs[:,:,ifr] = Ytest_hat_allSampsFrs # numSamps x numTestTrs

                Ytest_allSamps_allFrs[:,:,ifr] = Ytest_allSamps # numSamps x numTestTrs                                        
                testTrInds_allSamps_allFrs[:,:,ifr] = testTrInds_allSamps  # numSamps x numTestTrs
    #                    trsnow_allSamps_allFrs = trsnow_allSamps                    

                '''
                a_train = np.mean(perClassErrorTrain, axis=0)
                a_test = np.mean(perClassErrorTest, axis=0)
                a_shfl = np.mean(perClassErrorTestShfl, axis=0)
                a_chance = np.mean(perClassErrorTestChance, axis=0)

                import matplotlib.pyplot as plt
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



        ####################################################################################################################################                
        #%% Save SVM vars (for each experiment separately)
        ####################################################################################################################################                

        #%% Set svm dataframe

        svm_vars = pd.DataFrame([], columns = np.concatenate((cols_basic, cols_svm)))
#         svm_vars.at[index, cols_basic] = this_sess.iloc[index, :] # experiment info
        svm_vars.at[index, cols_basic] = this_sess.loc[index, :]

        # svm output
        if same_num_neuron_all_planes:
            svm_vars.at[index, cols_svm] = frames_svm, to_decode, use_balanced_trials, thAct, numSamples, softNorm, kfold, regType, cvect, meanX_allFrs, stdX_allFrs, \
                image_labels_now, image_indices, image_indices_previous_flash, image_indices_next_flash, \
                num_classes, iblock, trials_blocks, engagement_pupil_running, pupil_running_values, \
                cbest_allFrs, w_data_allFrs, b_data_allFrs, \
                perClassErrorTrain_data_allFrs, perClassErrorTest_data_allFrs, \
                perClassErrorTest_shfl_allFrs, perClassErrorTest_chance_allFrs, \
                inds_subselected_neurons_all, population_sizes_to_try, numShufflesN

        else:
            svm_vars.at[index, cols_svm] = frames_svm, to_decode, use_balanced_trials, thAct, numSamples, softNorm, kfold, regType, cvect, meanX_allFrs, stdX_allFrs, \
                image_labels_now, image_indices, image_indices_previous_flash, image_indices_next_flash, \
                num_classes, iblock, trials_blocks, engagement_pupil_running, pupil_running_values, \
                cbest_allFrs, w_data_allFrs, b_data_allFrs, \
                perClassErrorTrain_data_allFrs, perClassErrorTest_data_allFrs, \
                perClassErrorTest_shfl_allFrs, perClassErrorTest_chance_allFrs, \
                testTrInds_allSamps_allFrs, Ytest_allSamps_allFrs, Ytest_hat_allSampsFrs_allFrs



        #%% Save SVM results

        cre_now = cre[:cre.find('-')]
        # mouse, session, experiment: m, s, e
        ending = ''
        if same_num_neuron_all_planes:
            ending = 'sameNumNeuronsAllPlanes_'

        if ~np.isnan(iblock): # svm ran on the trial blocks
            if svm_blocks==-1: # divide trials into blocks based on the engagement state
                word = 'engaged'
            else:
                word = 'block'
            ending = f'{ending}{word}{iblock}_'

            
        name = f'{cre_now}_m-{mouse}_s-{session_id}_e-{lims_id}_{svmn}_frames{frames_svm[0]}to{frames_svm[-1]}_{ending}'

#         if project_codes == 'VisualBehavior':
        name = f'{name}{project_codes}_'
        
        name = f'{name}{now}'
        
        
        if saveResults:    
            svmName = os.path.join(dir_svm, name + '.h5') # os.path.join(d, svmn+os.path.basename(pnevFileName))
            print(svmName)

            print('Saving .h5 file')                
            svm_vars.to_hdf(svmName, key='svm_vars', mode='w')


        return svmName
    
    
    
    
    ####################################################################################################
    ####################################################################################################    
    ####################################################################################################
    ####################################################################################################
    #%% Set SVM vars
    ####################################################################################################
    ####################################################################################################    
    ####################################################################################################
    ####################################################################################################
    
    svm_min_neurs = 3 # min population size for training svm

#    same_num_neuron_all_planes = 1 # if 1, use the same number of neurons for all planes to train svm
#    frames_svm = np.arange(-10, 30) #30 # run svm on how what frames relative to omission
#    numSamples = 3 #10 #50
#    saveResults = 1    
    
    e = 'events_' if use_events else ''  
    
    if trial_type=='changes_vs_nochanges': # change, then no-change will be concatenated
        svmn = f'{e}svm_decode_changes_from_nochanges'
    elif trial_type=='hits_vs_misses':
        svmn = f'{e}svm_decode_hits_from_misses'        
    elif trial_type == 'baseline_vs_nobaseline':
        svmn = f'{e}svm_decode_baseline_from_nobaseline' #f'{e}svm_gray_omit' #svm_decode_baseline_from_nobaseline
        if use_spont_omitFrMinus1==0:
            svmn = svmn + '_spontFrs'
    else:
        svmn = f'{e}svm_decode_{to_decode}_image_from_{trial_type}' # 'svm_gray_omit'
    
    if use_matched_cells==123:
        svmn = svmn + '_matched_cells_FN1N2' #'_matched_cells_FN1Nn' #Familiar, N1, N+1
    elif use_matched_cells==12:
        svmn = svmn + '_matched_cells_FN1'
    elif use_matched_cells==23:
        svmn = svmn + '_matched_cells_N1Nn'
    elif use_matched_cells==13:
        svmn = svmn + '_matched_cells_FNn'        
    
    print(svmn)
        
    
#     kfold = 5 #2 #10 # KFold divides all the samples in  groups of samples, called folds (if , this is equivalent to the Leave One Out strategy), of equal sizes (if possible). The prediction function is learned using  folds, and the fold left out is used for test.
    regType = 'l2'
    
    norm_to_max_svm = 1 # normalize each neuron trace by its max # NOTE: no need to do this as we are z scoring neurons!
    doPlots = 0 # svm regularization plots
    
    softNorm = 1 # soft normalziation : neurons with sd<thAct wont have too large values after normalization
    thAct = 5e-4 # will be used for soft normalization #1e-5 
    # you did soft normalization when using inferred spikes ... i'm not sure if it makes sense here (using df/f trace); but we should take care of non-active neurons: 
    # Set NsExcluded : Identify neurons that did not fire in any of the trials (during ep) and then exclude them. Otherwise they cause problem for feature normalization.
    # Gamal: soft normalization is needed for numerical stability (i.e., to avoid division by zero). If you are worried about introducing inaccuracies you could try to lower the threshold. You can base the threshold on the typical values you get for std. Maybe make the threshold 1e-6 lower than that; ie. if a typical std value is x then set the threshold at x*1e-6
    
    smallestC = 0   
    shuffleTrs = False # set to 0 so for each iteration of numSamples, all frames are trained and tested on the same trials# If 1 shuffle trials to break any dependencies on the sequence of trials 
    # note about shuffleTrs: if you wanted to actually take effect, you should copy the relevant part from the svm code (svm_omissions) to
    # here before you go through nAftOmit frames ALSO do it for numSamples so for each sample different set of testing and training 
    # trials are used, however, still all nAftOmit frames use the same set ... 
    #... in the current version, the svm is trained and tested on different trials 
    # I don't think it matters at all, so I am not going to change it!
    cbest0 = np.nan
    useEqualTrNums = np.nan
    cbestKnown = 0 #cbest = np.nan
    
        
    #%% Set initial vars
    
    svm_total_frs = len(frames_svm)    
    now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")    
    cols = cols_basic
#    num_planes = 8           
        


    #%% If desired, run svm analysis only on engaged trials; redifine df_data only including the engaged rows    
    if svm_blocks==-101:
        print('\nOnly using engaged trials for SVM analysis.\n')

        df_data = df_data[df_data['engagement_state']=='engaged']            
        svmn = f'{svmn}_only_engaged'
        
        
    
    #%%
    if trial_type=='hits_vs_misses':
        
        image_data = pd.concat((df_data[df_data['hit']==1.0], df_data[df_data['miss']==1.0]))
        
        tr_id = 'trials_id'
        
        #%% Set the vector of image indices for each trial (current flash) (at time 0 of trial_data.iloc[0]['trace_timestamps'])
        u, u_i = np.unique(image_data[tr_id].values, return_index=True) # u_i is the index of the first occurrence of unique values in image_data[tr_id]
        image_trials = image_data.iloc[u_i,:] # image_trials is the same as image_data, except it includes only data from one neuron (yet all trials) # get the first row for each stimulus_presentations_id (multiple rows belong to the same stimulus_presentations_id, but different cells) 
        
        image_indices = image_trials['initial_image_name'].values
        image_indices_previous_flash = image_indices
        image_indices_next_flash = image_trials['change_image_name'].values
        
        hit_sum = sum(image_trials['hit']==1)
        miss_sum = sum(image_trials['miss']==1)
        print(f"\n{hit_sum} hit trials ; {miss_sum} miss trials\n")
                
        a = np.full((hit_sum), 1)
        b = np.full((miss_sum), 0)
        image_labels = np.concatenate((a,b)) # whether it was a hit or not

        
#         # set the engagement vector for hits and misses.
#         hit_eng = image_trials[image_trials['hit']==1]['engagement_state']
#         miss_eng = image_trials[image_trials['miss']==1]['engagement_state']                
#         image_trials_engagement = np.concatenate((hit_eng, miss_eng)) # whether it was engaged or not
            
            
        
    else:    
        
        tr_id = 'stimulus_presentations_id'
        
        #%% Set cell responses and trial types, using dataframe stimulus_response_df_allexp
        # NOTE: image_index indicates the image identity. It's NOT about the index of elements in an array!

        #%% Set image_indices of the current and previous flash (including all stimuli in the session) (image_indices is the image index of the flash that occurred at time 0; all_stim_indices_previous_flash is the image index of the flash that occurred 750ms before.)    
        # note: we do this on df_data which includes all images and omissions. We cannot do this after we exclude some trials (eg omissions), because then image index of the previous flash wont be simply a shift of image index of the current flash (becasue image index of the current flash doesnt have some stimuli anymore)!
        u, u_i = np.unique(df_data[tr_id].values, return_index=True)
        all_stim_trials = df_data.iloc[u_i,:] # images and omissions # get the first row for each stimulus_presentations_id (multiple rows belong to the same stimulus_presentations_id, but different cells) 
        all_stim_indices = all_stim_trials['image_index'].values
        
        if all_stim_indices[0]==8:
            print(f'\nThe very first flash is an omission. Not expected!\n') # if trial_type is omissions, we will take care of it below by setting image_indices_next_flash to nan
            
        all_stim_indices_previous_flash = np.concatenate(([np.nan], all_stim_indices))[:-1] # shift all_stim_indices 1 row down, and add nan to the begining of it
        all_stim_indices_next_flash = np.concatenate((all_stim_indices[1:], [np.nan])) # shift all_stim_indices 1 row up, and add nan to the end of it

    #     all_stim_indices.shape, all_stim_indices_previous_flash.shape, all_stim_indices_next_flash.shape
    #     np.equal(all_stim_indices[:-1], all_stim_indices_previous_flash[1:]).sum(), all_stim_indices[:-1].shape
    #     np.equal(all_stim_indices[1:], all_stim_indices_next_flash[:-1]).sum(), all_stim_indices[:-1].shape


        #%% Set data only including trials that will be used for analysis. Also set image index of the previous flash
        if trial_type=='images_omissions': # all images and omissions # i dont think it's a good idea to mix image and omission aligned traces because omission aligned traces may have prediction/error signal, so it wont be easy to interpret the results bc we wont know if the decoding reflects image-evoked or image-prediciton/error related signals.
            image_data = df_data
            image_indices_previous_flash = all_stim_indices_previous_flash
            image_indices_next_flash = all_stim_indices_next_flash

        elif trial_type=='omissions' or trial_type=='baseline_vs_nobaseline': # omissions:
            image_data = df_data[df_data['image_name']=='omitted']
            image_indices_previous_flash = all_stim_indices_previous_flash[all_stim_indices==8]
            image_indices_next_flash = all_stim_indices_next_flash[all_stim_indices==8]

            # list each omission with its previous and next image labels
#             a = np.argwhere(all_stim_indices==8).flatten()
            # b and c are the same
#             b = [[all_stim_indices[a[i]-1], all_stim_indices[a[i]], all_stim_indices[a[i]+1]] for i in range(len(a))]
#             b
#             c = [[image_indices_previous_flash[i], all_stim_indices[a[i]], image_indices_next_flash[i]] for i in range(len(a))]
#             c
        
            if all_stim_indices[0]==8:
                image_indices_next_flash[0] = np.nan # so it wont be analyzed bc image_labels[0] will now be nan.
                
            do = sum(image_indices_previous_flash==8)
            if do>0:
                print(f'Number of double omissions: {do}')
                print(f'\nRemoving {do*2} double-omission trials from analysis!\n')
                aa = np.argwhere(image_indices_next_flash==8).flatten()
                bb = np.argwhere(image_indices_previous_flash==8).flatten()
                image_indices_next_flash[aa] = np.nan
                image_indices_next_flash[bb] = np.nan
#                 c = [[image_indices_previous_flash[i], all_stim_indices[a[i]], image_indices_next_flash[i]] for i in range(len(a))]
#                 c                
                
            a = image_indices_previous_flash - image_indices_next_flash
            a = a[~np.isnan(a)]
            if a.any():
                print(f'UNCANNY: in {sum(a!=0)} cases, the images before and after the omission are different!!')

        elif trial_type=='images': # all images excluding omissions:
            image_data = df_data[df_data['image_name']!='omitted']
            image_indices_previous_flash = all_stim_indices_previous_flash[all_stim_indices!=8]
            image_indices_next_flash = all_stim_indices_next_flash[all_stim_indices!=8]

        elif trial_type=='changes' or trial_type=='changes_vs_nochanges': # image changes:
            image_data = df_data[df_data['change']==True]
            image_indices_previous_flash = all_stim_indices_previous_flash[all_stim_trials['change']==True]
            image_indices_next_flash = all_stim_indices_next_flash[all_stim_trials['change']==True]


        if trial_type=='changes_vs_nochanges': # decode image change vs no image change
            #image_data = df_data[df_data['change']==True] # change
            image_data2 = df_data[df_data['image_name']!='omitted'] # all images for now; below we will only get the image preceding the change



        #%% Set the vector of image indices for each trial (current flash) (at time 0 of trial_data.iloc[0]['trace_timestamps'])
        u, u_i = np.unique(image_data[tr_id].values, return_index=True) # u_i is the index of the first occurrence of unique values in image_data[tr_id]
        image_trials = image_data.iloc[u_i,:] # image_trials is the same as image_data, except it includes only data from one neuron (yet all trials) # get the first row for each stimulus_presentations_id (multiple rows belong to the same stimulus_presentations_id, but different cells) 
        image_indices = image_trials['image_index'].values

        if trial_type=='changes_vs_nochanges': # decode image change vs no image change
            u, u_i = np.unique(image_data2[tr_id].values, return_index=True) # u_i is the index of the first occurrence of unique values in image_data[tr_id]
            image_trials2 = image_data2.iloc[u_i,:] # image_trials is the same as image_data, except it includes only data from one neuron (yet all trials) # get the first row for each stimulus_presentations_id (multiple rows belong to the same stimulus_presentations_id, but different cells) 

            rel2image_change = - 1
            # set image_data2 only for images right preceding the image change
            image_data2_n = image_data2[image_data2[tr_id].isin(image_trials[tr_id].values + rel2image_change)]

            # find those rows of image_trials2 that belong to the image right preceding the image change        
    #         image_trials2_n = image_trials2[image_trials2[tr_id].isin(image_trials[tr_id].values + rel2image_change)]
    #         if (image_trials2_n.shape == image_trials.shape) == False:
    #             print('doesnt make sense! imagetrials and imagetrials2 must be the same shape')



        #%% Set the vector of image labels which will be used for decoding
        if trial_type=='changes_vs_nochanges' or trial_type == 'baseline_vs_nobaseline': # change, then no-change will be concatenated
            a = np.full((len(image_indices)), 0)
            b = np.full((len(image_indices)), 1)
            image_labels = np.concatenate((a,b))
                    
        else:

            if to_decode == 'current':
                image_labels = image_indices
            elif to_decode == 'previous':
                image_labels = image_indices_previous_flash
            elif to_decode == 'next':
                image_labels = image_indices_next_flash 


    image_trials0 = copy.deepcopy(image_trials)            
    image_labels0 = copy.deepcopy(image_labels)
    #len(image_labels0)
    print(f'Unique image labels: {np.unique(image_labels0)}')


    
    
    #### NOTE: codes above will result in decoding 9 classes (8 images + omissions) when to_decode='previous' and trial_type='images'. (it wont happen when to_decode='current' because above we only include images for trial_type='images'; it also wont happen when trial_type='omissions' or 'changes', because changes and omissions are not preceded by omissions (although rarely we do see double omissions))
    # if you want to also decode omissions (in addition to the 8 images) when to_decode='current', you should change trial_type='images' to trial_type='images_omissions'
    

    if trial_type=='changes': # image changes:
        no_change = np.argwhere((image_indices_previous_flash - image_indices)==0)
        if len(no_change)>0:
            sys.exit('Supposed to be an image change, but actually no image change occurred!')

            
    
    #%% If session_data and session_trials are given given as input:
    '''
#     #%% Set cell responses and trial types, using dataframes stim_response_data and stim_presentations

#     # get cell response data for this session
#     session_data = stim_response_data[stim_response_data['ophys_session_id']==session_id].copy()

#     # get stimulus presentations data for this session
#     session_trials = stim_presentations[stim_presentations['ophys_session_id']==session_id].copy()
#     # these two dataframes are linked by stimulus_presentations_id and ophys_session_id
    
    # get image trials
    image_trials = session_trials[session_trials['omitted']==False]
#     image_trials.shape

    # get cell response data for those trials
    image_stimulus_presentation_ids = image_trials.stimulus_presentations_id.values
    image_data = session_data[session_data.stimulus_presentations_id.isin(image_stimulus_presentation_ids)] # n_trials
#     image_data.shape

    # NOTE: no need to sort image_data below! because we loop through experiments in data_list which is already sorted; also we find the corresponding experiment in image_data by finding the right experiment id 
    # sort image_data by area and depth; this is mainly for consistency with my previous codes; also so that we analyze experiments in a good order, instead of randomly.
#     image_data = image_data.sort_values(by=['targeted_structure', 'imaging_depth'])

    
    # set the vector of image indices for each trial (flash) (at time 0 of trial_data.iloc[0]['trace_timestamps'])
    image_indices = image_trials['image_index'].values
#     image_indices.shape
    '''
    
    
    
        
    #%%
    exp_ids = data_list['ophys_experiment_id'].values
#     exp_ids = image_data['ophys_experiment_id'].unique() # get all the 8 experiments of a given session

    date = str(image_data.iloc[0]['date_of_acquisition'])[:10]
    cre = image_data.iloc[0]['cre_line']    
    stage = image_data.iloc[0]['session_type']
    experience_level = image_data.iloc[0]['experience_level']
    
    # set mouse_id ('external_donor_name')
    mouse = int(data_list['mouse_id'].iloc[0])
    
    # old method:
    '''
    session_name = str(image_data.iloc[0]['session_name'])
    uind = [m.start() for m in re.finditer('_', session_name)]
    mid = session_name[uind[0]+1 : uind[1]]
    if len(mid)<6: # a different nomenclature for session_name was used for session_id: 935559843
        mouse = int(session_name[uind[-1]+1 :])
    else:
        mouse = int(mid)
    
    if mouse==6:
        print('this guy has an invalid id! fixing it!')
        mouse = 453988
    '''    

        
        
    #%% Set number of neurons for all planes
    # Do this if training svm on the same number of neurons on all planes
    
    if same_num_neuron_all_planes==1:        
        n_neurons_all_planes = [] # for invalid experiments it will be nan.
        for index, lims_id in enumerate(exp_ids):    
            '''
            for il in [7]: #range(num_planes):
                index = il
                lims_id = exp_ids[il]
            '''
            # get data for a single experiment
            image_data_this_exp = image_data[image_data['ophys_experiment_id']==lims_id] # (neurons x trials) # all neurons for trial 1, then all neurons for trial 2, etc            
            
            if trial_type=='changes_vs_nochanges':
                # image preceding image change
                image_data_this_exp2 = image_data2_n[image_data2_n['ophys_experiment_id']==lims_id] # (neurons x trials) # all neurons for trial 1, then all neurons for trial 2, etc            
                
            n_neurons = image_data_this_exp['cell_specimen_id'].unique().shape[0]        
            
            n_neurons_all_planes.append(n_neurons)
                    
        min_n_neurons_all_planes = np.nanmin(n_neurons_all_planes)
        nN_trainSVM = np.max([svm_min_neurs, min_n_neurons_all_planes]) # use the same number of neurons for all planes to train svm
        
        # Train SVM on the following population sizes
        population_sizes_to_try = [nN_trainSVM] # [x+1 for x in range(X0.shape[0])] : if you want to try all possible population sizes

        print('Number of neurons per plane: %s' %str(sorted(n_neurons_all_planes)))
        print('Training SVM on the following population sizes for all planes: %s neurons.' %str(population_sizes_to_try))

        
        
        
        
    #%% Loop through the 8 planes of each session
    
    # initiate the pandas tabledatetime.datetime.fromtimestamp(1548449865.568)
    this_sess = pd.DataFrame([], columns = cols)
    
    for index, lims_id in enumerate(exp_ids):  

        '''
        ll = list(enumerate(exp_ids))
        l = ll[0]; # first plane
        index = l[0]; # plane index
        lims_id = l[1] # experiment id 
        '''
        '''
        for il in [2]: #range(num_planes):
            index = il
            lims_id = exp_ids[il]
        '''            
        
        print(f'\n\n------------- Analyzing {cre[:3]}, experiment_id: {lims_id} -------------\n\n')
        
        area = data_list.iloc[index]['targeted_structure'] #area
        depth = int(data_list.iloc[index]['imaging_depth']) #depth
        valid = sum(np.in1d(experiment_ids_valid, int(lims_id)))>0 #data_list.iloc[index]['valid']

        this_sess.at[index, cols[range(9)]] = session_id, lims_id, mouse, date, cre, stage, experience_level, area, depth #, valid
        
        if valid==False:
            print(f'Experiment {index} is not valid; skipping analysis...')
            
        elif valid: # do the analysis only if the experiment is valid

            # get data for a single experiment
            image_data_this_exp = image_data[image_data['ophys_experiment_id']==lims_id]
            
            if trial_type=='changes_vs_nochanges':
                # image preceding image change
                image_data_this_exp2 = image_data2_n[image_data2_n['ophys_experiment_id']==lims_id] # (neurons x trials) # all neurons for trial 1, then all neurons for trial 2, etc            

    #         image_data_this_exp.shape # (neurons x trials) # all neurons for trial 1, then all neurons for trial 2, etc
        
            '''
            area = image_data_this_exp.iloc[0]['targeted_structure']
            depth = int(image_data_this_exp.iloc[0]['imaging_depth'])
            print(area, depth)

            this_sess.at[index, cols[range(8)]] = session_id, lims_id, mouse, date, cre, stage, area, depth
            '''
        
            #%% Compute frame duration
            trace_time = image_data_this_exp['trace_timestamps'].iloc[0]
            samps_bef = np.argwhere(trace_time==0)[0][0]
            samps_aft = len(trace_time)-samps_bef
            
            frame_dur = np.mean(np.diff(trace_time)) # difference in sec between frames
            print(f'Frame duration {frame_dur:.3f} ms')
            
            # for mesoscope data:
        #     if np.logical_or(frame_dur < .09, frame_dur > .1):
        #         print(f'\n\nWARNING:Frame duration is unexpected!! {frame_dur}ms\n\n')


            #%% Set image-aligned traces in the format: frames x units x trials
            # create traces_fut, ie the traces in shape f x u x t: frames x units x trials, for each experiment of a given session
            n_frames = image_data_this_exp['trace'].values[0].shape[0]
            n_neurons = image_data_this_exp['cell_specimen_id'].unique().shape[0]
            n_trials = image_data_this_exp[tr_id].unique().shape[0]
            print(f'n_frames, n_neurons, n_trials: {n_frames, n_neurons, n_trials}')

            cell_specimen_id = image_data_this_exp['cell_specimen_id'].unique()
            
            # image_data_this_exp['trace'].values.shape # (neurons x trials) # all neurons for trial 1, then all neurons for trial 2, etc
            traces = np.concatenate((image_data_this_exp['trace'].values)) # (frames x neurons x trials)
            traces_fut = np.reshape(traces, (n_frames,  n_neurons, n_trials), order='F')
#             traces_fut.shape # frames x neurons x trials

            
            if trial_type=='changes_vs_nochanges':
                # image preceding image change
                traces2 = np.concatenate((image_data_this_exp2['trace'].values)) # (frames x neurons x trials)
                traces_fut2 = np.reshape(traces2, (n_frames,  n_neurons, n_trials), order='F')
                
                #plt.figure(); plt.plot(np.nanmean(traces_fut2, axis=(1,2))); plt.plot(np.nanmean(traces_fut, axis=(1,2)), color='r')
                
                # concatenate change and no-change trials
                traces_fut = np.concatenate((traces_fut, traces_fut2), axis=2)
                #traces_fut.shape

            
            '''
            aa_mx = np.array([np.max(traces_fut[:,i,:].flatten()) for i in range(traces_fut.shape[1])]) # neurons
            if (aa_mx==0).any():
#                 print(aa_mx)
                print('\n\nSome neurons have max=0; excluding them!\n\n')
                traces_fut = traces_fut[:, aa_mx!=0, :]
        
            n_neurons = traces_fut.shape[1]
        
            print(f'n_frames, n_neurons, n_trials: {n_frames, n_neurons, n_trials}')
            '''
            
            #%% Take care of nan values in image_labels (it happens if we are decoding the previous or next image and trial_type is images), and remove them frome traces and image_labels
            if to_decode != 'current':
                masknan = ~np.isnan(image_labels0)
                # remove the nan from labels and traces
                image_labels = image_labels0[masknan]
                traces_fut = traces_fut[:,:,masknan]
                image_trials = image_trials0[masknan]
            else:
                image_labels = copy.deepcopy(image_labels0)


            num_classes = len(np.unique(image_labels)) # number of classes to be classified by the svm

            print(f'Number of classes in SVM: {num_classes}')

            
            
            ######################################################
            # use balanced trials; same number of trials for each class 
            if num_classes==2 and use_balanced_trials==1: # if 1, use same number of trials for each class
                
                hrn = (image_labels==1).sum() # hit
                lrn = (image_labels==0).sum() # miss
                if hrn > lrn:
                    print('\nSubselecting hit trials so both classes have the same number of trials!\n')
                elif lrn > hrn:
                    print('\nSubselecting miss trials so both classes have the same number of trials!\n')
                else:
                    print('\nTrials are balanced; no need for subselecting\n')
                    
                if lrn != hrn:
                    mn_n_trs = np.min([sum(image_labels==0) , sum(image_labels==1)])
                    ind_mn_n_trs = np.argmin([sum(image_labels==0) , sum(image_labels==1)]) # class with fewer trials
                    # which class has more trials; this is the class that we want to resample
                    ind_mx_n_trs = np.argmax([sum(image_labels==0) , sum(image_labels==1)])

                    # get random order of trials from the class with more trials
        #             rnd.permutation(sum(image_labels==ind_mx_n_trs)) # then get the first mn_n_trs trials from the class with more trials
                    tr_inds = rnd.permutation(sum(image_labels==ind_mx_n_trs))[:mn_n_trs]

                    traces_larger_class = traces_fut[:,:,image_labels==ind_mx_n_trs][:,:,tr_inds]
                    labels_larger_class = image_labels[image_labels==ind_mx_n_trs][tr_inds]

                    # get the traces and labels for the smaller class
                    traces_smaller_class = traces_fut[:,:,image_labels==ind_mn_n_trs]
                    labels_smaller_class = image_labels[image_labels==ind_mn_n_trs]

                    # concatenate the 2 classes
                    traces_fut = np.concatenate((traces_smaller_class, traces_larger_class), axis=2)
                    image_labels = np.concatenate((labels_smaller_class, labels_larger_class))

                    print(traces_fut.shape , image_labels.shape)
            
#             import matplotlib.pyplot as plt
#             m = np.mean(traces_fut, axis=(1,2))
#             plt.plot(trace_time[samps_bef+frames_svm[0] : samps_bef+frames_svm[-1]], m[samps_bef+frames_svm[0] : samps_bef+frames_svm[-1]])

            ######################################################
            # reset n_trials
            n_trials = len(image_labels) #traces_fut.shape[2] # note: when trial_type='baseline_vs_nobaseline', traces_fut includes half the trials at this point; so we go with image_labels to get the number of trials.
            print(n_trials)
            
            # double check reshape above worked fine.
            '''
            c_all = []
            for itr in image_data_this_exp[tr_id].unique():
                this_data = image_data_this_exp[image_data_this_exp[tr_id]==itr]
                c = np.vstack(this_data['trace'].values)
            #     c.shape # neurons x frames
                c_all.append(c) # trials x neurons x frames

            c_all = np.array(c_all)
            c_all.shape

            np.equal(traces_fut[:,-2,102], c_all[102,-2,:]).sum()
            '''

            # image_indices = []
            # for itr in image_data_this_exp[tr_id].unique():
            #     image_index = image_trials[image_trials[tr_id]==itr]['image_index'].values[0]
            #     image_indices.append(image_index)

            
            
            this_sess.at[index, ['n_trials', 'n_neurons', 'cell_specimen_id', 'frame_dur', 'samps_bef', 'samps_aft']] = n_trials, n_neurons, cell_specimen_id, frame_dur, samps_bef, samps_aft
            print('===== plane %d: %d neurons; %d trials =====' %(index, n_neurons, n_trials))

            
            
            ######################################################
            #%% Do not continue the analysis if neurons or trials are too few, or if there are not enough unique classes in the data            
            
            continue_analysis = True
            
            if n_neurons==0: # some of the de-crosstalked planes don't have any neurons.
                this_sess.at[index, 'valid'] = False
                continue_analysis = False
                print('0 neurons! skipping invalid experiment %d, index %d' %(int(lims_id), index))
    
            if n_trials < 10:
                continue_analysis = False
                print('Skipping; too few trials to do SVM training! omissions=%d' %(n_trials))

            if n_neurons < svm_min_neurs: #np.logical_and(same_num_neuron_all_planes , n_neurons < svm_min_neurs):
                continue_analysis = False
                print('Skipping; too few neurons to do SVM training! neurons=%d' %(n_neurons))

            if num_classes < 2:
                continue_analysis = False
                print('Skipping; too few classes to do SVM training! number of classes=%d' %(num_classes))


                
            ######################## If everything is fine, continue with the SVM analysis ########################
            
            if continue_analysis:
                
                traces_fut_0 = copy.deepcopy(traces_fut)
                image_labels_0 = copy.deepcopy(image_labels)
                print(traces_fut_0.shape , image_labels_0.shape)
                
                pupil_running_values = np.nan

                ###############################################################################
                ###############################################################################                
                #### run svm analysis on the whole session
                if svm_blocks == -100 or svm_blocks == -101:
                    
                    svmName = svm_run_save(traces_fut_0, image_labels_0, [np.nan, []], svm_blocks, now, engagement_pupil_running, pupil_running_values, use_balanced_trials, project_codes) #, same_num_neuron_all_planes, norm_to_max_svm, svm_total_frs, n_neurons, numShufflesN, numSamples, num_classes, samps_bef, regType, kfold, cre, saveResults)
                    

                #### run svm analysis only on engaged trials
#                 if svm_blocks==-101:
#                     # only take the engaged trials
#                     engaged = image_trials['engagement_state'].values
#                     engaged[engaged=='engaged'] = 1
#                     engaged[engaged=='disengaged'] = 0
                    
    
                ###############################################################################
                ###############################################################################
                #### divide trials based on the engagement state
                elif svm_blocks == -1:
                    
                    #### engagement_state; based on lick rate and reward rate
                    if engagement_pupil_running == 0:
                        engaged = image_trials['engagement_state'].values
                        engaged[engaged=='engaged'] = 1
                        engaged[engaged=='disengaged'] = 0
                    
                    #### running speed or mean_pupil_area; trials above the median are called engaged
                    else:
                        if engagement_pupil_running == 1:
                            metric = image_trials['mean_pupil_area'].values
                    
                        elif engagement_pupil_running == 2:
                            metric = image_trials['mean_running_speed'].values
            
                        th_eng = np.nanmedian(metric)
                        engaged = metric+0
                        engaged[metric>=th_eng] = 1
                        engaged[metric<th_eng] = 0

                        pupil_running_values = metric
                        
#                     import matplotlib.pyplot as plt
#                     plt.plot(metric); plt.hlines(th_eng, 0, len(metric))
#                     plt.plot(engaged)
                    
    
                    # make sure there is enough trials in each engagement state
                    if sum(engaged==1)<10:
                        sys.exit(f'Aborting the analysis; too few ({sum(engaged==1)}) engaged trials!')
                        
                    if sum(engaged==0)<10:
                        sys.exit(f'Aborting the analysis; too few ({sum(engaged==0)}) disengaged trials!')
                        
                        
            
                    ### set trial_blocks: block0 is engaged0, block1 is engaged1
                    blocks_int = np.unique(engaged[~np.isnan(engaged.astype(float))]).astype(int)
                    trials_blocks = []
                    for iblock in blocks_int: 
                        trials_blocks.append(np.argwhere(engaged==iblock).flatten())
#                     print(trials_blocks); print(len(trials_blocks[0]), len(trials_blocks[1]))        
                    
    
                    ############## Loop through each trial block to run the SVM analysis ##############                    
                    for iblock in blocks_int: # iblock=0
                            
                        traces_fut_now = traces_fut_0[:,:,trials_blocks[iblock]]
                        image_labels_now = image_labels_0[trials_blocks[iblock]]
                        
                        if trial_type == 'baseline_vs_nobaseline': # decode activity at each frame vs. baseline
                            a = np.full((len(image_labels_now)), 0)
                            b = np.full((len(image_labels_now)), 1)
                            image_labels_now = np.concatenate((a,b))

                        print(traces_fut_now.shape , image_labels_now.shape)
                        print(np.unique(image_labels_now))
                        num_classes = len(np.unique(image_labels_now))
                        
                        svmName = svm_run_save(traces_fut_now, image_labels_now, [iblock, trials_blocks], svm_blocks, now, engagement_pupil_running, pupil_running_values, use_balanced_trials, project_codes) #, same_num_neuron_all_planes, norm_to_max_svm, svm_total_frs, n_neurons, numSamples, num_classes, samps_bef, regType, kfold, cre, saveResults)
                        
        
        
        
                ########################################################################
                ########################################################################
                #### do the block-by-block analysis: train SVM on blocks of trials; block 0: 1st half of the session; block 1: 2nd half of the session.
                else:
                    # divide the trials into a given number of blocks and take care of the last trial in the last block
                    a = np.arange(0, n_trials, int(n_trials/svm_blocks))
                    if len(a) < svm_blocks+1:
                        a = np.concatenate((a, [n_trials]))
                    if len(a) == svm_blocks+1:
                        a[-1] = n_trials
                    print(a)
                    print('Number of trials per block:')
                    print(np.diff(a))
                    
                    # set the range of trials for each block
                    trials_blocks = []
                    for iblock in range(len(a)-1): 
                        trials_blocks.append(np.arange(a[iblock], a[iblock+1]))
#                     print(trials_blocks)
                        
                    ############## Loop through each trial block to run the SVM analysis ##############
                    
                    for iblock in range(len(a)-1): # iblock=0        
                            
                        traces_fut = traces_fut_0[:,:,trials_blocks[iblock]]
                        image_labels = image_labels_0[trials_blocks[iblock]]
                        print(traces_fut.shape , image_labels.shape)
                        
                        svmName = svm_run_save(traces_fut, image_labels, [iblock, trials_blocks], svm_blocks, now, engagement_pupil_running, pupil_running_values, use_balanced_trials, project_codes) #, same_num_neuron_all_planes, norm_to_max_svm, svm_total_frs, n_neurons, numSamples, num_classes, samps_bef, regType, kfold, cre, saveResults)
                    
                
                
                

            
#%% Plot classification error for each frame of frames_svm
"""
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# read the svm_vars file
a = pd.read_hdf(svmName, key='svm_vars')
   
# make plots
a_train = 100-np.mean(a.iloc[0]['perClassErrorTrain_data_allFrs'], axis=0)
a_test = 100-np.mean(a.iloc[0]['perClassErrorTest_data_allFrs'], axis=0)
a_shfl = 100-np.mean(a.iloc[0]['perClassErrorTest_shfl_allFrs'], axis=0)
a_chance = 100-np.mean(a.iloc[0]['perClassErrorTest_chance_allFrs'], axis=0)

x = trace_time[samps_bef+frames_svm]

plt.figure()
plt.plot(x, a_train, color='k', marker='.', label='train') 
plt.plot(x, a_test, color='r', marker='.', label='test') 
plt.plot(x, a_shfl, color='y', marker='.', label='shfl')
plt.plot(x, a_chance, color='b', marker='.', label='chance')

plt.legend(loc='center left', bbox_to_anchor=(1, .7))                
plt.ylabel('% Classification accuracy')
plt.xlabel('Time after trial onset (sec)');
plt.vlines(0, np.min(a_chance), np.max(a_train))


"""



