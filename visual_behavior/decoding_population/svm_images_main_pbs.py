#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is adapted from svm_main_pbs.py to decode images (multi-class classification). [svm_main_pbs.py was to decode the state at each time point from the spontaneous activity; used for finding flash and omission signal.]

Run svm_init_images_pbs.py to set vars needed here.

Performs SVM analysis on a particular experiment of a session.
Saves the results in: dir_svm = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
    if using same_num_neuron_all_planes, results will be saved in: dir_svm/ 'same_n_neurons_all_planes'


Created on Wed Oct  7 16:01:17 2020
@author: farzaneh
"""

#%%
import datetime
import re
import pandas as pd
import copy
from svm_funs import *
# get_ipython().magic(u'matplotlib inline') # %matplotlib inline


#%%
def svm_main_images_pbs(data_list, experiment_ids_valid, df_data, session_trials, trial_type, dir_svm, kfold, frames_svm, numSamples, saveResults, cols_basic, cols_svm, to_decode='current', svm_blocks= np.nan, engagement_pupil_running = np.nan, use_events=False, same_num_neuron_all_planes=0):
    
    def svm_run_save(traces_fut_now, image_labels_now, iblock_trials_blocks, svm_blocks, now, engagement_pupil_running, pupil_running_values): #, same_num_neuron_all_planes, norm_to_max_svm, svm_total_frs, n_neurons, numSamples, num_classes, samps_bef, regType, kfold, cre, saveResults):
        '''
        #traces_fut_now = traces_fut_0
        #image_labels_now = image_labels_0
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



        numTrials = traces_fut_now.shape[2]  # numDataPoints = X_svm.shape[1] # trials             
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

            # x
            m = traces_fut_now[samps_bef + nAftOmit,:,:] # units x trials ; neural activity on the frame of omission

            # y
            m_y = image_labels_now

            # now set the x matrix for svm
            X_svm = m # units x trials
            Y_svm = m_y # trials

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
                             fr2an=np.nan, shflTrLabs=0, X_svm_incorr=0, Y_svm_incorr=0, mnHRLR_acrossDays=np.nan) # outputs have size the number of shuffles in setbestc (shuffles per c value)



            ########## Set the SVM output at best C (for all samples) and keep SVM vars for all frames

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
            svm_vars.at[index, cols_svm] = frames_svm, to_decode, thAct, numSamples, softNorm, kfold, regType, cvect, meanX_allFrs, stdX_allFrs, \
                image_labels_now, image_indices, image_indices_previous_flash, image_indices_next_flash, \
                num_classes, iblock, trials_blocks, engagement_pupil_running, pupil_running_values, \
                cbest_allFrs, w_data_allFrs, b_data_allFrs, \
                perClassErrorTrain_data_allFrs, perClassErrorTest_data_allFrs, \
                perClassErrorTest_shfl_allFrs, perClassErrorTest_chance_allFrs, \
                inds_subselected_neurons_all, population_sizes_to_try, numShufflesN

        else:
            svm_vars.at[index, cols_svm] = frames_svm, to_decode, thAct, numSamples, softNorm, kfold, regType, cvect, meanX_allFrs, stdX_allFrs, \
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
            
        name = f'{cre_now}_m-{mouse}_s-{session_id}_e-{lims_id}_{svmn}_frames{frames_svm[0]}to{frames_svm[-1]}_{ending}{now}'
                    
        if saveResults:    
            svmName = os.path.join(dir_svm, name + '.h5') # os.path.join(d, svmn+os.path.basename(pnevFileName))
            print(svmName)

            print('Saving .h5 file')                
            svm_vars.to_hdf(svmName, key='svm_vars', mode='w')



    
    ####################################################################################################
    ####################################################################################################    
    ####################################################################################################
    ####################################################################################################
    #%% Set SVM vars
    
    svm_min_neurs = 3 # min population size for training svm

#    same_num_neuron_all_planes = 1 # if 1, use the same number of neurons for all planes to train svm
#    frames_svm = np.arange(-10, 30) #30 # run svm on how what frames relative to omission
#    numSamples = 3 #10 #50
#    saveResults = 1    
    
    e = 'events_' if use_events else ''  
    
    if trial_type=='changes_vs_nochanges': # change, then no-change will be concatenated
        svmn = f'{e}svm_decode_changes_from_nochanges' # 'svm_gray_omit'
    else:
        svmn = f'{e}svm_decode_{to_decode}_image_from_{trial_type}' # 'svm_gray_omit'
        
#     kfold = 5 #2 #10 # KFold divides all the samples in  groups of samples, called folds (if , this is equivalent to the Leave One Out strategy), of equal sizes (if possible). The prediction function is learned using  folds, and the fold left out is used for test.
    regType = 'l2'
    
    norm_to_max_svm = 1 # normalize each neuron trace by its max # NOTE: no need to do this as we are z scoring neurons!
    doPlots = 0 # svm regularization plots
    
    softNorm = 1 # soft normalziation : neurons with sd<thAct wont have too large values after normalization
    thAct = 5e-4 # will be used for soft normalization #1e-5 
    # you did soft normalization when using inferred spikes ... i'm not sure if it makes sense here (using df/f trace); but we should take care of non-active neurons: 
    # Set NsExcluded : Identify neurons that did not fire in any of the trials (during ep) and then exclude them. Otherwise they cause problem for feature normalization.
    
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
        

    
    
    #%% Set cell responses and trial types, using dataframe stimulus_response_df_allexp
    # NOTE: image_index indicates the image identity. It's NOT about the index of elements in an array!
    
    #%% Set image_indices of the current and previous flash (including all stimuli in the session) (image_indices is the image index of the flash that occurred at time 0; all_stim_indices_previous_flash is the image index of the flash that occurred 750ms before.)    
    # note: we do this on df_data which includes all images and omissions. We cannot do this after we exclude some trials (eg omissions), because then image index of the previous flash wont be simply a shift of image index of the current flash (becasue image index of the current flash doesnt have some stimuli anymore)!
    u, u_i = np.unique(df_data['stimulus_presentations_id'].values, return_index=True)
    all_stim_trials = df_data.iloc[u_i,:] # images and omissions # get the first row for each stimulus_presentations_id (multiple rows belong to the same stimulus_presentations_id, but different cells) 
    all_stim_indices = all_stim_trials['image_index'].values

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
    
    elif trial_type=='omissions': # omissions:
        image_data = df_data[df_data['image_name']=='omitted']
        image_indices_previous_flash = all_stim_indices_previous_flash[all_stim_indices==8]
        image_indices_next_flash = all_stim_indices_next_flash[all_stim_indices==8]
        do = sum(image_indices_previous_flash==8)
        if do>0:
            print(f'Number of double omissions: {do}')
        if (image_indices_previous_flash - image_indices_next_flash).any():
            print(f'UNEXPECTED: the images before and after the omission are different!!')
            
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
    u, u_i = np.unique(image_data['stimulus_presentations_id'].values, return_index=True) # u_i is the index of the first occurrence of unique values in image_data['stimulus_presentations_id']
    image_trials = image_data.iloc[u_i,:] # image_trials is the same as image_data, except it includes only data from one neuron (yet all trials) # get the first row for each stimulus_presentations_id (multiple rows belong to the same stimulus_presentations_id, but different cells) 
    image_indices = image_trials['image_index'].values

    if trial_type=='changes_vs_nochanges': # decode image change vs no image change
        u, u_i = np.unique(image_data2['stimulus_presentations_id'].values, return_index=True) # u_i is the index of the first occurrence of unique values in image_data['stimulus_presentations_id']
        image_trials2 = image_data2.iloc[u_i,:] # image_trials is the same as image_data, except it includes only data from one neuron (yet all trials) # get the first row for each stimulus_presentations_id (multiple rows belong to the same stimulus_presentations_id, but different cells) 

        rel2image_change = - 1
        # set image_data2 only for images right preceding the image change
        image_data2_n = image_data2[image_data2['stimulus_presentations_id'].isin(image_trials['stimulus_presentations_id'].values + rel2image_change)]
        
        # find those rows of image_trials2 that belong to the image right preceding the image change        
#         image_trials2_n = image_trials2[image_trials2['stimulus_presentations_id'].isin(image_trials['stimulus_presentations_id'].values + rel2image_change)]
#         if (image_trials2_n.shape == image_trials.shape) == False:
#             print('doesnt make sense! imagetrials and imagetrials2 must be the same shape')
    
    
    
    #%% Set the vector of image labels which will be used for decoding
    if trial_type=='changes_vs_nochanges': # change, then no-change will be concatenated
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

    date = image_data.iloc[0]['date_of_acquisition'][:10]
    cre = image_data.iloc[0]['cre_line']    
    stage = image_data.iloc[0]['session_type']
    
    # set mouse_id ('external_donor_name')
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

        print(f'\n\n=========== Analyzing {cre[:3]}, experiment_id: {lims_id} ===========\n\n')

        '''
        for il in [2]: #range(num_planes):
            index = il
            lims_id = exp_ids[il]
        '''            
        '''
        ll = list(enumerate(exp_ids))
        l = ll[0]; # first plane
        index = l[0]; # plane index
        lims_id = l[1] # experiment id 
        '''
        
        area = data_list.iloc[index]['targeted_structure'] #area
        depth = int(data_list.iloc[index]['imaging_depth']) #depth
        valid = sum(np.in1d(experiment_ids_valid, int(lims_id)))>0 #data_list.iloc[index]['valid']

        this_sess.at[index, cols[range(8)]] = session_id, lims_id, mouse, date, cre, stage, area, depth #, valid
        
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
            n_trials = image_data_this_exp['stimulus_presentations_id'].unique().shape[0]
            print(f'n_frames, n_neurons, n_trials: {n_frames, n_neurons, n_trials}')

            # image_data_this_exp['trace'].values.shape # (neurons x trials) # all neurons for trial 1, then all neurons for trial 2, etc
            traces = np.concatenate((image_data_this_exp['trace'].values)) # (frames x neurons x trials)
            traces_fut = np.reshape(traces, (n_frames,  n_neurons, n_trials), order='F')
#             traces_fut.shape # frames x neurons x trials

            
            if trial_type=='changes_vs_nochanges':
                # image preceding image change
                traces2 = np.concatenate((image_data_this_exp2['trace'].values)) # (frames x neurons x trials)
                traces_fut2 = np.reshape(traces2, (n_frames,  n_neurons, n_trials), order='F')
                
                plt.figure(); plt.plot(np.nanmean(traces_fut2, axis=(1,2))); plt.plot(np.nanmean(traces_fut, axis=(1,2)), color='r')
                
                # concatenate change and no-change trials
                traces_fut = np.concatenate((traces_fut, traces_fut2), axis=2)
                traces_fut.shape

            
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
                # reset n_trials
                n_trials = traces_fut.shape[2] 
            else:
                image_labels = copy.deepcopy(image_labels0)
        
            num_classes = len(np.unique(image_labels)) # number of classes to be classified by the svm
            print(f'Number of classes in SVM: {num_classes}')

            
            # double check reshape above worked fine.
            '''
            c_all = []
            for itr in image_data_this_exp['stimulus_presentations_id'].unique():
                this_data = image_data_this_exp[image_data_this_exp['stimulus_presentations_id']==itr]
                c = np.vstack(this_data['trace'].values)
            #     c.shape # neurons x frames
                c_all.append(c) # trials x neurons x frames

            c_all = np.array(c_all)
            c_all.shape

            np.equal(traces_fut[:,-2,102], c_all[102,-2,:]).sum()
            '''

            # image_indices = []
            # for itr in image_data_this_exp['stimulus_presentations_id'].unique():
            #     image_index = image_trials[image_trials['stimulus_presentations_id']==itr]['image_index'].values[0]
            #     image_indices.append(image_index)

            
            
            this_sess.at[index, ['n_trials', 'n_neurons', 'frame_dur', 'samps_bef', 'samps_aft']] = n_trials, n_neurons, frame_dur, samps_bef, samps_aft
            print('===== plane %d: %d neurons; %d trials =====' %(index, n_neurons, n_trials))

            
            
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
                
                #### run svm analysis on the whole session
                if np.isnan(svm_blocks):
                    svm_run_save(traces_fut_0, image_labels_0, [np.nan, []], svm_blocks, now, engagement_pupil_running, pupil_running_values) #, same_num_neuron_all_planes, norm_to_max_svm, svm_total_frs, n_neurons, numShufflesN, numSamples, num_classes, samps_bef, regType, kfold, cre, saveResults)
                    
                    
                
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
                    blocks_int = np.unique(engaged[~np.isnan(engaged)]).astype(int)
                    trials_blocks = []
                    for iblock in blocks_int: 
                        trials_blocks.append(np.argwhere(engaged==iblock).flatten())
#                     print(trials_blocks)    
                    
                    ############## Loop through each trial block to run the SVM analysis ##############
                    
                    for iblock in blocks_int: # iblock=0
                            
                        traces_fut_now = traces_fut_0[:,:,trials_blocks[iblock]]
                        image_labels_now = image_labels_0[trials_blocks[iblock]]
                        print(traces_fut_now.shape , image_labels_now.shape)
                        
                        svm_run_save(traces_fut_now, image_labels_now, [iblock, trials_blocks], svm_blocks, now, engagement_pupil_running, pupil_running_values) #, same_num_neuron_all_planes, norm_to_max_svm, svm_total_frs, n_neurons, numSamples, num_classes, samps_bef, regType, kfold, cre, saveResults)
                        
        
        
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
                        
                        svm_run_save(traces_fut, image_labels, [iblock, trials_blocks], svm_blocks, now, engagement_pupil_running, pupil_running_values) #, same_num_neuron_all_planes, norm_to_max_svm, svm_total_frs, n_neurons, numSamples, num_classes, samps_bef, regType, kfold, cre, saveResults)
                    
                
                
                

            
#%% Plot classification error for each frame of frames_svm
"""
import matplotlib.pyplot as plt
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
plt.xlabel('Frame after trial onset')  
"""







################################################################################################################################################################################################################################################
#%% To run on the cluster
################################################################################################################################################################################################################################################

# if socket.gethostname() != 'ibs-farzaneh-ux2': # it's not known what node on the cluster will run your code, so all i can do is to say if it is not your pc .. but obviously this will be come problematic if running the code on a computer other than your pc or the cluster

import os
import numpy as np
import pandas as pd
import pickle
import sys
import visual_behavior.data_access.loading as loading


    
    
#%% Get the input arguments passed here from pbstools (in svm_init_images script)

isess = int(sys.argv[1])
use_events = bool(sys.argv[2])
to_decode = str(sys.argv[3])
trial_type = str(sys.argv[4])
svm_blocks = int(sys.argv[5]) 
engagement_pupil_running = int(sys.argv[6]) 

# trial_type = 'images' # 'omissions', 'images', 'changes' # what trials to use for SVM analysis # the population activity of these trials at time time_win will be used to decode the image identity of flashes that occurred at their time 0 (if to_decode='current') or 750ms before (if to_decode='previous').
# to_decode = 'previous' # 'current' (default): decode current image.    'previous': decode previous image.    'next': decode next image. # remember for omissions, you cant do "current", bc there is no current image, it has to be previous or next!
# use_events = True # False # whether to run the analysis on detected events (inferred spikes) or dff traces.
# svm_blocks = -1 #np.nan # 2 #np.nan # -1: divide trials based on engagement # 2 # number of trial blocks to divide the session to, and run svm on. # set to np.nan to run svm analysis on the whole session
# engagement_pupil_running = 1 # np.nan or 0,1,2 for engagement, pupil, running: which metric to use to define engagement. only effective if svm_blocks=-1

print('\n\n======== Analyzing session index %d ========\n' %(isess))



#%% Set SVM vars # NOTE: Pay special attention to the following vars before running the SVM:

time_win = [-.5, .75] # [-.3, 0] # timewindow (relative to trial onset) to run svm; this will be used to set frames_svm # analyze image-evoked responses
# time_trace goes from -.5 to .65sec in the image-aligned traces.

kfold = 5 #2 #10 # KFold divides all the samples in  groups of samples, called folds (if , this is equivalent to the Leave One Out strategy), of equal sizes (if possible). The prediction function is learned using  folds, and the fold left out is used for test.
numSamples = 50 # numSamples = 2
same_num_neuron_all_planes = 0 # if 1, use the same number of neurons for all planes to train svm
saveResults = 1 # 0 #

project_codes = ['VisualBehaviorMultiscope'] # session_numbers = [4]

print(f'Decoding *{to_decode}* image from *{trial_type}* activity at {time_win} ms!')
print(f'kfold = {kfold}')    
if use_events:
    print(f'Using events')
else:
    print(f'Using DF/F')
if np.isnan(svm_blocks):
    print(f'Running SVM on the whole session.')
elif svm_blocks==-1:
    print(f'Dividing trials based on engagement state.')
else:    
    print(f'Doing block-by-block analysis.')
    
    

#%%
dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

# make svm dir to save analysis results
dir_svm = os.path.join(dir_server_me, 'SVM')

if same_num_neuron_all_planes:
    dir_svm = os.path.join(dir_svm, 'same_n_neurons_all_planes')

if ~np.isnan(svm_blocks):
    dir_svm = os.path.join(dir_svm, f'trial_blocks')
                
if not os.path.exists(dir_svm):
    os.makedirs(dir_svm)


    

#%% Use below for VIP and SST, since concatenated_stimulus_response_dfs exists for them.
'''
#%% Set session_numbers from filen; NOTE: below will work only if a single element exists in session_numbers

sn = os.path.basename(filen)[len('metadata_basic_'):].find('_'); 
session_numbers = [int(os.path.basename(filen)[len('metadata_basic_'):][sn+1:])]

print(f'The following sessions will be analyzed: {project_codes}, {session_numbers}')

#%% Set the project and stage for sessions to be loaded # Note: it would be nice to pass the following two vars as python args in the python job (pbs tools); except I'm not sure how to pass an array as an argument.
# we need these to set "stim_response_data" and "stim_presentations" dataframes
# r1 = filen.find('metadata_basic_'); r2 = filen.find(']_'); 
# project_codes = [filen[r1+3:r2-1]]
# session_numbers = [int(filen[filen.rfind('_[')+2:-1])] # may not work if multiple sessions are being passed as session_numbers



#%% Set "stim_response_data" and "stim_presentations" dataframes

stim_response_dfs = loading.get_concatenated_stimulus_response_dfs(project_codes, session_numbers) # stim_response_data has cell response data and all experiment metadata
experiments_table = loading.get_filtered_ophys_experiment_table() # functions to load cell response data and stimulus presentations metadata for a set of sessions

# merge stim_response_dfs with experiments_table
stim_response_data = stim_response_dfs.merge(experiments_table, on=['ophys_experiment_id', 'ophys_session_id'])
# stim_response_data.keys()

stim_presentations = loading.get_concatenated_stimulus_presentations(project_codes, session_numbers) # stim_presentations has stimulus presentations metadata for the same set of experiments
# stim_presentations.keys()




#%% Load the pickle file that includes the list of sessions and metadata

pkl = open(filen, 'rb')
dict_se = pickle.load(pkl)
metadata_basic = pickle.load(pkl)

list_all_sessions_valid = dict_se['list_all_sessions_valid']
list_all_experiments_valid = dict_se['list_all_experiments_valid']
    



#%% Set the session to be analyzed, and its metada

session_id = int(list_all_sessions_valid[isess])
data_list = metadata_basic[metadata_basic['session_id'].values==session_id]


#%% Set session_data and session_trials for the current session. # these two dataframes are linked by stimulus_presentations_id and ophys_session_id

session_data = stim_response_data[stim_response_data['ophys_session_id']==session_id].copy() # get cell response data for this session
session_trials = stim_presentations[stim_presentations['ophys_session_id']==session_id].copy() # get stimulus presentations data for this session

df_data = session_data
'''
        



#%% We need below for slc, because stim_response_dfs could not be saved for it!

#%% Load metadata_basic, and use it to set list_all_sessions_valid
# note metadata is set in code: svm_init_images_pre
'''
dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
dir_svm = os.path.join(dir_server_me, 'SVM')
a = '_'.join(project_codes)
filen = os.path.join(dir_svm, f'metadata_basic_{a}_slc')
print(filen)
'''

# import visual_behavior.data_access.loading as loading

# data release sessions
# experiments_table = loading.get_filtered_ophys_experiment_table()
experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')
metadata_valid = experiments_table[experiments_table['project_code']=='VisualBehaviorMultiscope'] # multiscope sessions


# Use the new list of sessions that are de-crosstalked and will be released in March 2021
# metadata_meso_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/meso_decrosstalk/meso_experiments_in_release.csv'
# metadata_valid = pd.read_csv(metadata_meso_dir)

list_all_sessions_valid = metadata_valid['ophys_session_id'].unique()


print(f'{len(list_all_sessions_valid)}: Number of de-crosstalked sessions for analysis')

# get the list of 8 experiments for all sessions
experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
experiments_table = experiments_table.reset_index('ophys_experiment_id')

metadata_all = experiments_table[experiments_table['ophys_session_id'].isin(list_all_sessions_valid)==True] # metadata_all = experiments_table[np.in1d(experiments_table['ophys_session_id'].values, list_all_sessions_valid)]
metadata_all.shape
metadata_all.shape[0]/8



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




'''
pkl = open(filen, 'rb')
metadata_basic = pickle.load(pkl)
metadata_basic



#%% Reset list_all_sessions_valid using metadata_basic
# note for some sessions we cant set metadata_basic because the following code fails: Session_obj = LimsOphysSession(lims_id=session_id)
# so, we use metadata_basic to reset list_all_sessions_valid

list_all_sessions_valid_now = metadata_basic[metadata_basic['valid']]['session_id'].unique()
# list_all_sessions_valid_now.shape, list_all_sessions_valid.shape
list_all_sessions_valid = list_all_sessions_valid_now
print(list_all_sessions_valid.shape)
'''


#%% Set experiments_table to be merged with stim response df later

# import visual_behavior.data_access.loading as loading
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

experiments_table = loading.get_filtered_ophys_experiment_table()

# only keep certain columns of experiments_table, before merging it with stim_response_df
experiments_table_sub = experiments_table.reset_index('ophys_experiment_id')
experiments_table_sub = experiments_table_sub.loc[:, ['ophys_experiment_id', 'ophys_session_id', 'cre_line', 'session_name', 'date_of_acquisition', 'session_type', 'exposure_number']]

'''
# make sure all experiments in metadata exist in experiment_table
es = experiments_table.index
esmeta = metadata_basic[metadata_basic['valid']]['experiment_id'].unique()
if np.in1d(es, esmeta).sum() == len(esmeta):
    print(f'All experiments in metadata df exist in experiment_table, as expected')
else:
    print(f'Some experiments in metadata df do NOT exist in experiment_table; very uncanny!')
'''
    

    
############################################################################################    
####################### Set vars for the sessions to be analyzed #######################
############################################################################################

#%% Set the session to be analyzed, and its metada

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



#%% Set data_list: metadata for this session including a valid column

data_list = metadata_all[metadata_all['ophys_session_id']==session_id]
data_list['valid'] = False
data_list.loc[data_list['ophys_experiment_id'].isin(experiment_ids_valid), 'valid'] = True
# sort by area and depth
data_list = data_list.sort_values(by=['targeted_structure', 'imaging_depth'])

experiment_ids_this_session = experiment_ids



'''
# isess = 0
session_id = int(list_all_sessions_valid[isess])
data_list = metadata_basic[metadata_basic['session_id'].values==session_id]
experiment_ids_this_session = data_list['experiment_id'].values
'''

    
#%% Set stimulus_response_df_allexp, which includes stimulus_response_df (plus some metadata columns) for all experiments of a session

# from set_trialsdf_existing_in_stimdf import *

# set columns of stim_response_df to keep
c = ['stimulus_presentations_id', 'cell_specimen_id', 'trace', 'trace_timestamps', 'image_index', 'image_name', 'change', 'engagement_state', 'mean_running_speed', 'mean_pupil_area'] 

stimulus_response_df_allexp = pd.DataFrame()

for ophys_experiment_id in experiment_ids_this_session: # ophys_experiment_id = experiment_ids_this_session[0]

    if sum(np.in1d(experiment_ids_valid, int(ophys_experiment_id)))>0: # make sure lims_id is among the experiments in the data release
#     if data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['valid']:
            
        valid = True
    
        dataset = loading.get_ophys_dataset(ophys_experiment_id)
#         analysis = ResponseAnalysis(dataset, use_extended_stimulus_presentations=True) # False # use_extended_stimulus_presentations flag is set to False, meaning that only the main stimulus metadata will be present (image name, whether it is a change or omitted, and a few other things). If you need other columns (like engagement_state or anything from the behavior strategy model), you have to set that to True        
        analysis = ResponseAnalysis(dataset, use_extended_stimulus_presentations=True, use_events=use_events)        
        stim_response_df = analysis.get_response_df(df_name='stimulus_response_df')

#         dataset.running_speed['speed'].values
#         dataset.eye_tracking['pupil_area'] # dataset.eye_tracking['time']

#         dataset.extended_stimulus_presentations['mean_running_speed'].values
#         dataset.extended_stimulus_presentations['mean_pupil_area']
        

        
        # work on this
#         if svm_blocks==-1 and np.isnan(np.unique(stim_response_df['engagement_state'].values)):
#             sys.exit(f'Session {session_id} does not have engagement state!')
            
        ########### 
        if 0: #trial_type == 'changes':
            # if desired, set a subset of stimulus df that only includes image changes with a given trial condition (hit, autorewarded, etc)
            # trials_response_df.keys()  # 'hit', 'false_alarm', 'miss', 'correct_reject', 'aborted', 'go', 'catch', 'auto_rewarded', 'stimulus_change'
    
            scdf = stim_response_df[stim_response_df['change']==True]
#             trials_response_df = analysis.get_response_df(df_name='trials_response_df')

            # get a subset of trials df that includes only those rows (flashes) in trials df that also exist in stim df # note: you need to set use_extended_stimulus_presentations=True so stim_response_df has start_time as a column.
            trialsdf_existing_in_stimdf = set_trialsdf_existing_in_stimdf(stim_response_df, trials_response_df) # trialsdf_existing_in_stimdf has the same size as scdf, and so can be used to link trials df and stimulus df, hence to get certain rows out of scdf (eg hit or aborted trials, etc)
            # trialsdf_existing_in_stimdf.shape, scdf.shape, trials_response_df.shape

            # only get those stimulus df rows that are go trials; seems like all stim df rows are go trials.
#             stim_response_df_new = scdf[(trialsdf_existing_in_stimdf['go']==True).values]
            
            # only get those stimulus df rows that are not aborted and not auto_rewarded (we use trialsdf_existing_in_stimdf to get this information)
            a = np.logical_and((trialsdf_existing_in_stimdf['auto_rewarded']==False), (trialsdf_existing_in_stimdf['aborted']==False))
            stim_response_df_new = scdf.iloc[a.values]
            stim_response_df_new.shape

            # only get those stimulus df rows that are hit trials
#             stim_response_df_new = scdf[(trialsdf_existing_in_stimdf['hit']==True).values]

            # reset stim_response_df
            stim_response_df = stim_response_df_new    
        ###########    
            
        stim_response_df0 = stim_response_df # stim_response_df.keys()        
        stim_response_df = stim_response_df0.loc[:,c]
        
    else:
        
        valid = False
        stim_response_df = pd.DataFrame([np.full((len(c)), np.nan)], columns=c) 
        
    stim_response_df['ophys_session_id'] = session_id #data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['session_id']
    stim_response_df['ophys_experiment_id'] = ophys_experiment_id # data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['experiment_id']
    stim_response_df['area'] = metadata_all[metadata_all['ophys_experiment_id']==ophys_experiment_id].iloc[0]['targeted_structure']
    stim_response_df['depth'] = metadata_all[metadata_all['ophys_experiment_id']==ophys_experiment_id].iloc[0]['imaging_depth']
    stim_response_df['valid'] = valid 
    
#     stim_response_df['area'] = data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['area']
#     stim_response_df['depth'] = data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['depth']
#     stim_response_df['valid'] = data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['valid']    
    
    
    #### set the final stim_response_data_this_exp
    
    stim_response_data_this_exp = stim_response_df.merge(experiments_table_sub, on=['ophys_experiment_id', 'ophys_session_id']) # add to stim_response_df columns with info on cre, date, etc        
    # reorder columns, so metadata columns come first then stimulus and cell data
    stim_response_data_this_exp = stim_response_data_this_exp.iloc[:, np.concatenate((np.arange(len(c), stim_response_data_this_exp.shape[-1]), np.arange(0, len(c))))]
        
    #### keep data from all experiments    
    stimulus_response_df_allexp = pd.concat([stimulus_response_df_allexp, stim_response_data_this_exp])
    
stimulus_response_df_allexp
stimulus_response_df_allexp.keys()

df_data = stimulus_response_df_allexp
session_trials = np.nan # we need it as an input to the function


### some tests related to images changes in stim df vs. trials df  
# trials_response_df[~np.in1d(range(len(trials_response_df)), stimdf_ind_in_trdf)]['trial_type'].unique()
# trials_response_df[~np.in1d(range(len(trials_response_df)), stimdf_ind_in_trdf)]['catch'].unique()

# trials_response_df.keys()  # 'hit', 'false_alarm', 'miss', 'correct_reject', 'aborted', 'go', 'catch', 'auto_rewarded', 'stimulus_change'
# stimdf_ind_in_trdf.shape


# (trialsdf_existing_in_stimdf['catch']==True).sum()
# a = trialsdf_existing_in_stimdf[trialsdf_existing_in_stimdf['go']==False]
# a['stimulus_change']
# a['lick_times']
# a['auto_rewarded']

# b = scdf[(trialsdf_existing_in_stimdf['go']==False).values]
# b['licked']
# b['rewarded']
# b['start_time']

'''
trials_response_df[trials_response_df['auto_rewarded']==True]['catch'] # all autorewarded are marked as neither catch or go.

# find the change cases where animal licked but no reward is delivered and see if they are catch or not, if not whats wrong? (seems like stim df does not have catches, but has one of these weird cases)
d = np.logical_and(np.logical_and((stim_response_df['licked']==1) , (stim_response_df['rewarded']==0)) , (stim_response_df['change']==True))
dd = np.argwhere(d).flatten()
stim_response_df.iloc[dd]
stim_response_df[dd[1]]
'''


#%% Set frames_svm

# trace_time = np.array([-0.46631438, -0.3730515 , -0.27978863, -0.18652575, -0.09326288,
#         0.        ,  0.09326288,  0.18652575,  0.27978863,  0.3730515 ,
#         0.46631438,  0.55957726,  0.65284013])

trace_time = df_data.iloc[0]['trace_timestamps']
# set samps_bef and samps_aft: on the image-aligned traces, samps_bef frames were before the image, and samps_aft-1 frames were after the image
samps_bef = np.argwhere(trace_time==0)[0][0] # 5
samps_aft = len(trace_time)-samps_bef #8 
print(samps_bef, samps_aft)

r1 = np.argwhere((trace_time-time_win[0])>=0)[0][0]
r2 = np.argwhere((trace_time-time_win[1])>=0)
if len(r2)>0:
    r2 = r2[0][0]
else:
    r2 = len(trace_time)
             
frames_svm = range(r1, r2)-samps_bef # range(-10, 30) # range(-1,1) # run svm on these frames relative to trial (image/omission) onset.
# print(frames_svm)

# samps_bef (=40) frames before omission ; index: 0:39
# omission frame ; index: 40
# samps_aft - 1 (=39) frames after omission ; index: 41:79





#%%
cols_basic = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_trials', 'n_neurons', 'frame_dur', 'samps_bef', 'samps_aft']) #, 'flash_omit_dur_all', 'flash_omit_dur_fr_all'])
cols_svm_0 = ['frames_svm', 'to_decode', 'thAct', 'numSamples', 'softNorm', 'kfold', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs', 
       'image_labels', 'image_indices', 'image_indices_previous_flash', 'image_indices_next_flash',
       'num_classes', 'iblock', 'trials_blocks', 'engagement_pupil_running', 'pupil_running_values' , 
       'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
       'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
       'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs']
if same_num_neuron_all_planes:        
    cols_svm = np.concatenate((cols_svm_0, ['inds_subselected_neurons_all', 'population_sizes_to_try', 'numShufflesN']))

else:    
    cols_svm = np.concatenate((cols_svm_0, ['testTrInds_allSamps_allFrs', 'Ytest_allSamps_allFrs', 'Ytest_hat_allSampsFrs_allFrs']))

    


#%% Run the SVM function

print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess, len(list_all_sessions_valid)))

# Use below if you set session_data and session_trials above: for VIP and SST
svm_main_images_pbs(data_list, experiment_ids_valid, df_data, session_trials, trial_type, dir_svm, kfold, frames_svm, numSamples, saveResults, cols_basic, cols_svm, to_decode, svm_blocks, engagement_pupil_running, use_events, same_num_neuron_all_planes)
# svm_main_images_pbs(data_list, df_data, session_trials, trial_type, dir_svm, kfold, frames_svm, numSamples, saveResults, cols_basic, cols_svm, to_decode, svm_blocks, use_events, same_num_neuron_all_planes)

# Use below if you set stimulus_response_df_allexp above: for Slc (can be also used for other cell types too; but we need it for Slc, as the concatenated dfs could not be set for it)
# svm_main_images_pbs(data_list, stimulus_response_df_allexp, dir_svm, frames_svm, numSamples, saveResults, cols_basic, cols_svm, same_num_neuron_all_planes=0)


