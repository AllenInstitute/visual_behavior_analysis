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
from svm_funs import *
# get_ipython().magic(u'matplotlib inline') # %matplotlib inline


#%%
def svm_main_images_pbs(data_list, df_data, session_trials, dir_svm, frames_svm, numSamples, saveResults, cols_basic, cols_svm, labels2ana=0, same_num_neuron_all_planes=0):

    
    #%% Set SVM vars
    
    svm_min_neurs = 3 # min population size for training svm

#    same_num_neuron_all_planes = 1 # if 1, use the same number of neurons for all planes to train svm
#    frames_svm = np.arange(-10, 30) #30 # run svm on how what frames relative to omission
#    numSamples = 3 #10 #50
#    saveResults = 1    
    
    svmn = 'svm_images' # 'svm_gray_omit'
        
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
    
    svm_total_frs = len(frames_svm)    
    now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")    
    cols = cols_basic
#    num_planes = 8           
        

    
    
    
    #%% If stimulus_response_df_allexp is given as input:
    #%% Set cell responses and trial types, using dataframe stimulus_response_df_allexp
    
#     session_data = stimulus_response_df_allexp
    
    image_data = df_data[df_data['image_name']!='omitted']

    #%% set the vector of image indices for each trial (flash) (at time 0 of trial_data.iloc[0]['trace_timestamps'])
    u, u_i = np.unique(image_data['stimulus_presentations_id'].values, return_index=True)
    image_trials = image_data.iloc[u_i,:] # get the first row for each stimulus_presentations_id (multiple rows belong to the same stimulus_presentations_id, but different cells) 
    image_indices = image_trials['image_index'].values
    # set image_index of the previous flash (image_indices is the image index of the flash that occurred at time 0; image_indices_previous_flash is the image index of the flash that occurred 750ms before.)
    image_indices_previous_flash = np.concatenate(([np.nan], image_indices))[:-1]
#     image_indices.shape, image_indices_previous_flash.shape
#     np.equal(image_indices[:-1], image_indices_previous_flash[1:]).sum()
    
    if labels2ana == 0:
        image_labels = image_indices
    elif labels2ana == -1:
        image_labels = image_indices_previous_flash
        
        
    
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
    exp_ids = data_list['experiment_id'].values
#     exp_ids = image_data['ophys_experiment_id'].unique() # get all the 8 experiments of a given session

    date = image_data.iloc[0]['date_of_acquisition'][:10]
    cre = image_data.iloc[0]['cre_line']    
    stage = image_data.iloc[0]['session_type']
    
    
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

    '''
    # plot the entire trace for each session
    
    get_ipython().magic(u'matplotlib inline') # %matplotlib inline
    
    dir_now = 'traces_entire_session'
    if not os.path.exists(os.path.join(dir0, dir_now)):
        os.makedirs(os.path.join(dir0, dir_now))
    nam = '%s_mouse%d_sess%d_traceAveNs_allFrs' %(cre, mouse, session_id)
    
    plt.figure(figsize=(8,11)) # show all traces of all neurons for each plane individually, to see how the activity changes in a session!
    '''
    
    for index, lims_id in enumerate(exp_ids): 

        print(f'\n\n=========== Analyzing {cre[:3]}, experiment_id: {lims_id} ===========\n\n')

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
        
        area = data_list.iloc[index]['area']
        depth = int(data_list.iloc[index]['depth'])
        valid = data_list.iloc[index]['valid']

        this_sess.at[index, cols[range(8)]] = session_id, lims_id, mouse, date, cre, stage, area, depth #, valid
        
        
        if valid: # do the analysis only if the experiment is valid

            # get data for a single experiment
            image_data_this_exp = image_data[image_data['ophys_experiment_id']==lims_id]
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
            print(n_frames, n_neurons, n_trials)

            # image_data_this_exp['trace'].values.shape # (neurons x trials) # all neurons for trial 1, then all neurons for trial 2, etc
            traces = np.concatenate((image_data_this_exp['trace'].values)) # (frames x neurons x trials)
            traces_fut = np.reshape(traces, (n_frames,  n_neurons, n_trials), order='F')
    #         traces_fut.shape # frames x neurons x trials

    
        
            #%% Take care of nan values in image_labels (it happens if we are decoding the previous or next image), and remove them frome traces and image_labels
            if labels2ana != 0:
                masknan = ~np.isnan(image_labels)
                # remove the nan from labels and traces
                image_labels = image_labels[masknan]
                traces_fut = traces_fut[:,:,masknan]
                # reset n_trials
                n_trials = traces_fut.shape[2] 
                
        

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


            if n_neurons==0: # some of the de-crosstalked planes don't have any neurons.
                this_sess.at[index, 'valid'] = False
                print('0 neurons! skipping invalid experiment %d, index %d' %(int(lims_id), index))

            elif len(image_labels)>0: # only run the analysis if there are omissions in the session. #else: # 
                this_sess.at[index, ['n_trials', 'n_neurons', 'frame_dur']] = n_trials, n_neurons, frame_dur
                print('===== plane %d: %d neurons; %d trials =====' %(index, n_neurons, n_trials))



            #%%#############################################################
            ###### The following two variables are the key variables: ######
            ###### (they are ready to be used)                        ######
            ###### calcium traces aligned on trials:                  ######
            ###### traces_fut  (frames x units x trials)              ######
            ###### and their time traces:                             ######
            ###### local_time_allOmitt  (frames x trials)             ######
            ################################################################


            #%%
            if n_trials < 10:
                sys.exit('Too few trials to do SVM training! omissions=%d' %(n_trials))

            if np.logical_and(same_num_neuron_all_planes , n_neurons < svm_min_neurs):
                sys.exit('Too few neurons to do SVM training! neurons=%d' %(n_neurons))


            #%% Normalize each neuron trace by its max (so if on a day a neuron has low FR in general, it will not be read as non responsive!)

            if norm_to_max_svm==1:
                print('Normalizing each neuron trace by its max.')
                # compute max on the entire trace of the session:
                aa_mx = [np.max(traces_fut[:,i,:].flatten()) for i in range(n_neurons)] # neurons

                a = np.transpose(traces_fut, (0,2,1)) # frames x trials x units
                b = a / aa_mx
                traces_fut = np.transpose(b, (0,2,1)) # frames x units x trials



            #%% Starting to set variables for the SVM analysis                

            num_classes = len(np.unique(image_indices)) # number of classes to be classified by the svm

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


                
            numTrials = traces_fut.shape[2]  # numDataPoints = X_svm.shape[1] # trials             
            len_test = numTrials - int((kfold-1.)/kfold*numTrials) # number of testing trials   
            numDataPoints = numTrials

            if len_test==1:
                Ytest_hat_allSampsFrs_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan).squeeze() # squeeze helps if len_test=1
            else:
                Ytest_hat_allSampsFrs_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan)
            Ytest_allSamps_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan)         
            testTrInds_allSamps_allFrs = np.full((numSamples, len_test, svm_total_frs), np.nan)



            #%% Use SVM to classify population activity on the gray-screen frame right before the omission vs. the activity on frame + nAftOmit after omission 

            ifr = -1
            for nAftOmit in frames_svm: # nAftOmit=frames_svm[0]; ifr = 3
                ifr = ifr+1
                print('\n================ Running SVM on frame %d relative to trial onset ================\n' %nAftOmit)

                # x
                m = traces_fut[samps_bef + nAftOmit,:,:] # units x trials ; neural activity on the frame of omission

                # y
                m_y = image_labels

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

            #%%
            svm_vars = pd.DataFrame([], columns = np.concatenate((cols_basic, cols_svm)))

            # experiment info
            svm_vars.at[index, cols_basic] = this_sess.iloc[index, :] 

            # svm output
            if same_num_neuron_all_planes:
                svm_vars.at[index, cols_svm] = frames_svm, thAct, numSamples, softNorm, regType, cvect, meanX_allFrs, stdX_allFrs, \
                    cbest_allFrs, w_data_allFrs, b_data_allFrs, \
                    perClassErrorTrain_data_allFrs, perClassErrorTest_data_allFrs, \
                    perClassErrorTest_shfl_allFrs, perClassErrorTest_chance_allFrs, \
                    inds_subselected_neurons_all, population_sizes_to_try, numShufflesN

            else:
                svm_vars.at[index, cols_svm] = frames_svm, thAct, numSamples, softNorm, regType, cvect, meanX_allFrs, stdX_allFrs, \
                    cbest_allFrs, w_data_allFrs, b_data_allFrs, \
                    perClassErrorTrain_data_allFrs, perClassErrorTest_data_allFrs, \
                    perClassErrorTest_shfl_allFrs, perClassErrorTest_chance_allFrs, \
                    testTrInds_allSamps_allFrs, Ytest_allSamps_allFrs, Ytest_hat_allSampsFrs_allFrs


            #%% Save SVM results

            cre_now = cre[:cre.find('-')]
            # mouse, session, experiment: m, s, e
            if same_num_neuron_all_planes:
#                 name = '%s_m-%d_s-%d_e-%s_%s_sameNumNeuronsAllPlanes_%s' %(cre_now, mouse, session_id, lims_id, svmn, now)
                name = '%s_m-%d_s-%d_e-%s_%s_frames%dto%d_sameNumNeuronsAllPlanes_%s' %(cre_now, mouse, session_id, lims_id, svmn, frames_svm[0], frames_svm[-1], now) 
            else:
#                 name = '%s_m-%d_s-%d_e-%s_%s_%s' %(cre_now, mouse, session_id, lims_id, svmn, now) 
                name = '%s_m-%d_s-%d_e-%s_%s_frames%dto%d_%s' %(cre_now, mouse, session_id, lims_id, svmn, frames_svm[0], frames_svm[-1], now) 
                
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


#%% Set SVM vars

labels2ana = 0 # 0 (default): decode current image.    -1: decode previous image.    1: decode next image.
same_num_neuron_all_planes = 0 # if 1, use the same number of neurons for all planes to train svm
numSamples = 50 # 2
saveResults = 1 # 0 #

time_win = [-.3, 0] #[0, .5] # timewindow (relative to trial onset) to run svm; this will be used to set frames_svm # analyze image-evoked responses

project_codes = ['VisualBehaviorMultiscope'] # session_numbers = [4]



#%%
dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

# make svm dir to save analysis results
dir_svm = os.path.join(dir_server_me, 'SVM')
if same_num_neuron_all_planes:
    dir_svm = os.path.join(dir_svm, 'same_n_neurons_all_planes')
if not os.path.exists(dir_svm):
    os.makedirs(dir_svm)


    
    
    
#%% Get the input arguments passed here from pbstools (in svm_init_images script)

isess = int(sys.argv[1])
filen = str(sys.argv[2])




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



#%% Set the session to be analyzed, and its metada

# isess = 0
session_id = int(list_all_sessions_valid[isess])
data_list = metadata_basic[metadata_basic['session_id'].values==session_id]
experiment_ids_this_session = data_list['experiment_id'].values



#%% Set stimulus_response_df_allexp, which includes stimulus_response_df (plus some metadata columns) for all experiments of a session

CHECK THIS:
NOTE: below you added exposure_number, make sure it appears as a column in stim_response_data_this_exp
(exposure_number is in experiments_table)

# import visual_behavior.data_access.loading as loading
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

experiments_table = loading.get_filtered_ophys_experiment_table()
c = ['stimulus_presentations_id', 'cell_specimen_id', 'trace', 'trace_timestamps', 'image_index', 'image_name']

stimulus_response_df_allexp = pd.DataFrame()
for ophys_experiment_id in experiment_ids_this_session: # ophys_experiment_id = experiment_ids_this_session[1]

    if data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['valid']:
        dataset = loading.get_ophys_dataset(ophys_experiment_id)
        analysis = ResponseAnalysis(dataset, use_extended_stimulus_presentations=False) # use_extended_stimulus_presentations flag is set to False, meaning that only the main stimulus metadata will be present (image name, whether it is a change or omitted, and a few other things). If you need other columns (like engagement_state or anything from the behavior strategy model), you have to set that to True
        stim_response_df = analysis.get_response_df(df_name='stimulus_response_df')
        stim_response_df0 = stim_response_df
    #     stim_response_df.keys()        
        stim_response_df = stim_response_df0.loc[:,c]
    else:
        stim_response_df = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=['stimulus_presentations_id', 'cell_specimen_id', 'trace', 'trace_timestamps', 'image_index', 'image_name']) 
        
    stim_response_df['ophys_session_id'] = data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['session_id']
    stim_response_df['ophys_experiment_id'] = ophys_experiment_id # data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['experiment_id']
    stim_response_df['area'] = data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['area']
    stim_response_df['depth'] = data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['depth']
    stim_response_df['valid'] = data_list[data_list['experiment_id']==ophys_experiment_id].iloc[0]['valid']    
    
    
    #### set the final stim_response_data_this_exp
    
    # add to stim_response_df columns with info on cre, date, etc 
    e = experiments_table.reset_index('ophys_experiment_id')
    # only keep certain columns of experiments_table, before merging it with stim_response_df
    e = e.loc[:, ['ophys_experiment_id', 'ophys_session_id', 'cre_line', 'session_name', 'date_of_acquisition', 'session_type', 'exposure_number']]
    stim_response_data_this_exp = stim_response_df.merge(e, on=['ophys_experiment_id', 'ophys_session_id'])    
    # reorder columns
    stim_response_data_this_exp = stim_response_data_this_exp.iloc[:, np.concatenate((np.arange(len(c), stim_response_data_this_exp.shape[-1]), np.arange(0, len(c))))]
        
    #### keep data from all experiments    
    stimulus_response_df_allexp = pd.concat([stimulus_response_df_allexp, stim_response_data_this_exp])
    
stimulus_response_df_allexp
stimulus_response_df_allexp.keys()

df_data = stimulus_response_df_allexp
session_trials = np.nan # we need it as an input to the function






#%% Set frames_svm

trace_time = df_data.iloc[0]['trace_timestamps']

# trace_time = np.array([-0.46631438, -0.3730515 , -0.27978863, -0.18652575, -0.09326288,
#         0.        ,  0.09326288,  0.18652575,  0.27978863,  0.3730515 ,
#         0.46631438,  0.55957726,  0.65284013])

r1 = np.argwhere((trace_time-time_win[0])>=0)[0][0]
r2 = np.argwhere((trace_time-time_win[1])>=0)[0][0]

# set samps_bef and samps_aft: on the image-aligned traces, samps_bef frames were before the image, and samps_aft-1 frames were after the image
samps_bef = np.argwhere(trace_time==0)[0][0] # 5
samps_aft = len(trace_time)-samps_bef #8 

frames_svm = range(r1, r2)-samps_bef # range(-10, 30) # range(-1,1) # run svm on these frames relative to trial (image/omission) onset.
# print(frames_svm)

# samps_bef (=40) frames before omission ; index: 0:39
# omission frame ; index: 40
# samps_aft - 1 (=39) frames after omission ; index: 41:79





#%%
cols_basic = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_trials', 'n_neurons', 'frame_dur']) #, 'flash_omit_dur_all', 'flash_omit_dur_fr_all'])
if same_num_neuron_all_planes:        
    cols_svm = ['frames_svm', 'thAct', 'numSamples', 'softNorm', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs', 
           'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
           'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
           'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs',
           'inds_subselected_neurons_all', 'population_sizes_to_try', 'numShufflesN']

else:    
    cols_svm = ['frames_svm', 'thAct', 'numSamples', 'softNorm', 'regType', 'cvect', 'meanX_allFrs', 'stdX_allFrs', 
           'cbest_allFrs', 'w_data_allFrs', 'b_data_allFrs',
           'perClassErrorTrain_data_allFrs', 'perClassErrorTest_data_allFrs',
           'perClassErrorTest_shfl_allFrs', 'perClassErrorTest_chance_allFrs',
           'testTrInds_allSamps_allFrs', 'Ytest_allSamps_allFrs', 'Ytest_hat_allSampsFrs_allFrs']

    
    


#%% Run the SVM function

print('\n\n======================== Analyzing session %d, %d/%d ========================\n' %(session_id, isess+1, len(list_all_sessions_valid)))

# Use below if you set session_data and session_trials above: for VIP and SST
svm_main_images_pbs(data_list, df_data, session_trials, dir_svm, frames_svm, numSamples, saveResults, cols_basic, cols_svm, labels2ana=0, same_num_neuron_all_planes=0)

# Use below if you set stimulus_response_df_allexp above: for Slc (can be also used for other cell types too; but we need it for Slc, as the concatenated dfs could not be set for it)
# svm_main_images_pbs(data_list, stimulus_response_df_allexp, dir_svm, frames_svm, numSamples, saveResults, cols_basic, cols_svm, same_num_neuron_all_planes=0)





