#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:12:43 2019

@author: farzaneh
"""

import numpy as np
    

#%%
"""
Fits SVM using XTrain, and returns percent class loss for XTrain and XTest
"""

def linearSVM(XTrain, YTrain, XTest, YTest, **options):
    
    import numpy as np
    from sklearn import svm
    linear_svm = [];
    # Create a classifier: a support vector classifier
    if options.get('l1'):
        l1 = options.get('l1');
        #print 'running l1 svm classification\r' 
        linear_svm = svm.LinearSVC(C = l1, loss='squared_hinge', penalty='l1', dual=False)
    elif options.get('l2'):
        l2 = options.get('l2');        
        #print 'running l2 svm classification\r' 
        linear_svm = svm.LinearSVC(C = l2, loss='squared_hinge', penalty='l2', dual=True)
        
    linear_svm.fit(XTrain, np.squeeze(YTrain))    

    #%%
    def perClassError(Y, Yhat):
        import numpy as np
        perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr
    
    perClassErrorTest = perClassError(YTest, linear_svm.predict(XTest))
    perClassErrorTrain = perClassError(YTrain, linear_svm.predict(XTrain))
    
    #%%
    class summaryClass:
        perClassErrorTrain = []
        perClassErrorTest = []
        model = []
        XTest = []
        XTrain = []
        YTest = []
        YTrain = []
        
    summary = summaryClass()
    summary.perClassErrorTrain = perClassErrorTrain
    summary.perClassErrorTest = perClassErrorTest
    summary.model = linear_svm
    summary.XTest = XTest
    summary.XTrain = XTrain
    summary.YTest = YTest
    summary.YTrain = YTrain
    
    return summary





#%%
    
"""
crossValidateModel: divides data into training and test datasets. Calls linearSVM.py, which does linear SVM 
using XTrain, and returns percent class loss for XTrain and XTest.

"""
# summary,_ =  crossValidateModel(X[ifr,:,:].transpose(), Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
def crossValidateModel(X, Y, modelFn, **options):
    import numpy as np
    import numpy.random as rng
#    from linearSVM import linearSVM
#    from linearSVR import linearSVR
    
    if options.get('kfold'):
        kfold = options.get('kfold')
    else:
        kfold = 10;

    if np.logical_or(options.get('shflTrs'), options.get('shflTrs')==0):
        shflTrs = options.get('shflTrs')
    else:
        shflTrs = True
#    print shflTrs
        
#    Y = np.squeeze(np.array(Y).astype(int)); # commented so it works for svr too.
        
    if X.shape[0]>len(Y):
        numObservations = len(Y);
        numFeatures = len(X)/numObservations;
        X = np.reshape(np.array(X.astype('float')), (numObservations, numFeatures), order='F');
    
    numObservations, numFeatures = X.shape # trials x neurons
    
    
    ## %%%%%
    cls = [0]
    while len(cls)<2: # make sure both classes exist in YTrain    
        if shflTrs==1: # shuffle trials to break any dependencies on the sequence of trails; Also since we take the first 90% of trials as training and the last 10% as testing, for each run of this code we want to make sure we use different sets of trials as testing and training.
            print('shuffling trials in crossValidateModel')
            shfl = rng.permutation(np.arange(0, numObservations))
            Ys = Y[shfl]
            Xs = X[shfl, :]
            testTrInds = shfl[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)] # index of testing trials (that will be used in svm below)
        else:
            shfl = np.arange(0, numObservations)
            Ys = Y
            Xs = X
            
        ## %%%%% divide data to training and testing sets
        YTrain = Ys[np.arange(0, int((kfold-1.) / kfold * numObservations))] # Take the first 90% of trials as training set       
        cls = np.unique(YTrain)        
#        print cls
    
    
    if len(cls)==2:        
        YTest = Ys[np.arange(int((kfold-1.)/kfold*numObservations), numObservations)] # Take the last 10% of trials as testing set
    
        XTrain = Xs[np.arange(0, int((kfold-1.)/kfold*numObservations)), :]
        XTest = Xs[np.arange(int((kfold-1.)/kfold*numObservations), numObservations), :]
    
    
        # Fit the classifier
        results = modelFn(XTrain, YTrain, XTest, YTest, **options)
        
        return results, shfl # shfl includes the index of trials in X whose first 90% are used for training and the last 10% are used for testing.





#%% Function to run SVM  (when X is frames x units x trials)
# Remember each numSamples will have a different set of training and testing dataset, however for each numSamples, the same set of testing/training dataset
# will be used for all frames and all values of c (unless shuffleTrs is 1, in which case different frames and c values will have different training/testing datasets.)

def set_best_c(X,Y,regType,kfold,numDataPoints,numSamples,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest=np.nan,fr2an=np.nan, shflTrLabs=0, X_svm_incorr=0, Y_svm_incorr=0, mnHRLR_acrossDays=np.nan):
#    X = X_svm
#    Y = Y_svm
#    cbest=np.nan
#    fr2an=np.nan
#    X_svm_incorr=0
#    Y_svm_incorr=0
#    mnHRLR_acrossDays=np.nan
    
    # numSamples = 10; # number of iterations for finding the best c (inverse of regularization parameter)
    # if you don't want to regularize, go with a very high cbest and don't run the section below.
    # cbest = 10**6    
    # regType = 'l1'
    # kfold = 10;

            
#    import numpy as np
    import numpy.random as rng

#    from imaging_decisionMaking_exc_inh.utils.lassoClassifier.crossValidateModel import crossValidateModel
#    from imaging_decisionMaking_exc_inh.utils.lassoClassifier.linearSVM import linearSVM
#    from crossValidateModel import crossValidateModel
#    from linearSVM import linearSVM
    
    def perClassError(Y, Yhat):
        perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr


    #%%
#    if type(X_svm_incorr)==int:
#        testIncorr = 0
#    else:
#        testIncorr = 1
#        
#    if testIncorr:
#        len_test_incorr = len(Y_svm_incorr)
#        
#    # frames to do SVM analysis on
#    if np.isnan(fr2an).all(): # run SVM on all frames
#        frs = range(X.shape[0])
#    else:
#        frs = fr2an        
    
    
    #%% NOW:

#    numSamples = 10 #50
#    
#    kfold = 10
#    regType = 'l2'
#    cbest = np.nan
#    numDataPoints = X_svm.shape[1] # trials 
#
#    smallestC = 0   
#    shuffleTrs = False # set to 0 so for each iteration of numSamples, all frames are trained and tested on the same trials# If 1 shuffle trials to break any dependencies on the sequence of trails 
#
#    shflTrLabs = 0
#    
    
    #%%    
    # set range of c (regularization parameters) to check    
    if np.isnan(cbest).all(): # we need to set cbest
        bestcProvided = False        
        if regType == 'l1':
            print('\n-------------- Running l1 svm classification --------------\r') 
            # cvect = 10**(np.arange(-4, 6,0.2))/numTrials;
            cvect = 10**(np.arange(-4, 6,0.2)) / numDataPoints
        elif regType == 'l2':
            print('\n-------------- Running l2 svm classification --------------\r') 
            cvect = 10**(np.arange(-6, 6,0.2)) / numDataPoints          
        nCvals = len(cvect)
#        print('try the following regularization values: \n', cvect
        # formattedList = ['%.2f' % member for member in cvect]
        # print('try the following regularization values = \n', formattedList        
    else: # bestc is provided and we want to fit svm on shuffled trial labels
        bestcProvided = True           
        nCvals = 1 # cbest is already provided
       
        
    #%%
#    smallestC = 0 # if 1: smallest c whose CV error falls below 1 se of min CV error will be used as optimal C; if 0: c that gives min CV error will be used as optimal c.
    if smallestC==1:
        print('bestc = smallest c whose cv error is less than 1se of min cv error')
    else:
        print('bestc = c that gives min cv error')
    #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
        
    
    #%%    
    ##############
#    hrn = (Y==1).sum()
#    lrn = (Y==0).sum()
#
#    if ~np.isnan(mnHRLR_acrossDays): # we will make sure for all days SVM is trained on similar number of trials, ie mnHRLR_acrossDays trials of HR and mnHRLR_acrossDays trials of LR.
#        trsn = mnHRLR_acrossDays        
#        numTrials = 2*mnHRLR_acrossDays 
#        print('using %d HR and %d LR trials for SVM training, same across all sessions' %(mnHRLR_acrossDays, mnHRLR_acrossDays))
#    else:        
#        if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!        
#            trsn = min(lrn,hrn)
#            if hrn > lrn:
#                print('Subselecting HR trials so both classes have the same number of trials!')
#                numTrials = lrn*2
#            elif lrn > hrn:
#                print('Subselecting LR trials so both classes have the same number of trials!')
#                numTrials = hrn*2            
#        else:
#            numTrials = X.shape[2]
    
    
    #%%
    
    numTrials = X.shape[1]
    print('FINAL: %d trials; %d neurons' %(numTrials, X.shape[0]))
    
    len_test = numTrials - int((kfold-1.)/kfold*numTrials) # number of testing trials   
            
    X0 = X + 0 # units x trials
    Y0 = Y + 0
    
    
    #%%    
    ########################################################################################################################################################################
    ########################################################################################################################################################################
#    nFrs = np.shape(X)[0]
    wAllC = np.ones((numSamples, nCvals, X.shape[0])) + np.nan
    bAllC = np.ones((numSamples, nCvals)) + np.nan
    
    perClassErrorTrain = np.ones((numSamples, nCvals)) + np.nan
    perClassErrorTest = np.ones((numSamples, nCvals)) + np.nan    
    perClassErrorTest_shfl = np.ones((numSamples, nCvals)) + np.nan
    perClassErrorTest_chance = np.ones((numSamples, nCvals)) + np.nan
    
    testTrInds_allSamps = np.full((numSamples, len_test), np.nan)              
    Ytest_allSamps = np.full((numSamples, len_test), np.nan)              
    Ytest_hat_allSampsFrs = np.full((numSamples, nCvals, len_test), np.nan)        
#    testTrInds_outOfY0_allSamps = np.full((numSamples, len_test), np.nan)
    trsnow_allSamps = np.full((numSamples, numTrials), np.nan)              
#    eqy = np.full((X0.shape[0], numSamples), np.nan)
    
    # incorr:
#    if testIncorr:
#        perClassErrorTest_incorr = np.ones((numSamples, nCvals)) + np.nan
#        perClassErrorTest_shfl_incorr = np.ones((numSamples, nCvals)) + np.nan
#        perClassErrorTest_chance_incorr = np.ones((numSamples, nCvals)) + np.nan        
#        Ytest_hat_allSampsFrs_incorr = np.full((numSamples, nCvals, len_test_incorr), np.nan)        
       
    
    #%%    
    ########################################## Train SVM numSamples times to get numSamples cross-validated datasets.    
    for s in range(numSamples): # s = 0       
        print('Iteration %d' %(s))
        
        ############ Make sure both classes have the same number of trials when training the classifier
        # set trsnow: # index of trials (out of Y0) after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)
        
#        if ~np.isnan(mnHRLR_acrossDays): 
#            randtrs_hr = np.argwhere(Y0==1)[rng.permutation(hrn)[0:trsn]].squeeze() # subselect mnHRLR_acrossDays trials from HR
#            randtrs_lr = np.argwhere(Y0==0)[rng.permutation(lrn)[0:trsn]].squeeze() # subselect mnHRLR_acrossDays trials from LR
#            trsnow = np.sort(np.concatenate((randtrs_hr , randtrs_lr))) # index of trials after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)
#
#            X = X0[:,:,trsnow] # trsnow : index of trials (out of X0 and Y0) that are used to set X and Y
#            Y = Y0[trsnow]
#                
#        else:
#            if useEqualTrNums and hrn!=lrn: # if the HR and LR trials numbers are not the same, pick equal number of trials of the 2 classes!
#                if hrn > lrn:
#                    randtrs = np.argwhere(Y0==1)[rng.permutation(hrn)[0:trsn]].squeeze()
#                    trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==0).squeeze()))) # index of trials after picking random hr (or lr) in order to make sure both classes have the same number in the final Y (on which svm was run)
#                elif lrn > hrn:
#                    randtrs = np.argwhere(Y0==0)[rng.permutation(lrn)[0:trsn]].squeeze() # random sample of the class with more trials
#                    trsnow = np.sort(np.concatenate((randtrs , np.argwhere(Y0==1).squeeze()))) # all trials of the class with fewer trials + the random sample set above for the other class
#    
#                X = X0[:,:,trsnow] # trsnow : index of trials (out of X0 and Y0) that are used to set X and Y
#                Y = Y0[trsnow]
#    
#            else: # include all trials
#                trsnow = np.arange(0, len(Y0))
        
        trsnow = np.arange(0, len(Y0))
        X = X0[:,trsnow] # trsnow : index of trials (out of X0 and Y0) that are used to set X and Y
        Y = Y0[trsnow]

        trsnow_allSamps[s,:] = trsnow
#        numTrials, numNeurons = X.shape[2], X.shape[1]
#            print('FINAL: %d trials; %d neurons' %(numTrials, numNeurons)                        
            
        ######################## Setting chance Y: same length as Y for testing data, and with equal number of classes 0 and 1.
#        no = Y.shape[0]
#        len_test = numTrials - int((kfold-1.)/kfold*numTrials)    
        permIxs = rng.permutation(len_test) # needed to set perClassErrorTest_shfl   
    
        Y_chance = np.zeros(len_test)
        if rng.rand()>.5:
            b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
        else:
            b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
        Y_chance[b] = 1

#        if testIncorr:
#            permIxs_incorr = rng.permutation(len_test_incorr)
#            Y_chance_incorr = np.zeros(len_test_incorr)
#            if rng.rand()>.5:
#                b = rng.permutation(len_test_incorr)[0:np.floor(len_test_incorr/float(2)).astype(int)]
#            else:
#                b = rng.permutation(len_test_incorr)[0:np.ceil(len_test_incorr/float(2)).astype(int)]
#            Y_chance_incorr[b] = 1            
            
            
        ####################### Set the chance Y for training SVM on shuffled trial labels
        if shflTrLabs: # shuffle trial classes in Y
            Y = np.zeros(numTrials) # Y_chance0
            if rng.rand()>.5:
                b = rng.permutation(numTrials)[0:np.floor(numTrials/float(2)).astype(int)]
            else:
                b = rng.permutation(numTrials)[0:np.ceil(numTrials/float(2)).astype(int)]
            Y[b] = 1

        
        ######################## Shuffle trial orders, so the training and testing datasets are different for each numSamples (we only do this if shuffleTrs is 0, so crossValidateModel does not shuffle trials, so we have to do it here, otherwise all numSamples will have the same set of testing and training datasets.)
        ######################## REMEMBER : YOU ARE CHANGING THE ORDER OF TRIALS HERE!!!
        ########################
        ########################
        if shuffleTrs==0: # shuffle trials here (instead of inside crossValidateModel) to break any dependencies on the sequence of trails 
            
#            Ybefshfl = Y            
            shfl = rng.permutation(np.arange(0, numTrials)) # shfl: new order of trials ... shuffled indeces of Y... the last 1/10th indeces will be testing trials.
            
            Y = Y[shfl] 
            X = X[:,shfl]             
            
            # Ytest_allSamps[s,:] : Y that will be used as testing trials in this sample
            Ytest_allSamps[s,:] = Y[np.arange(numTrials-len_test, numTrials)] # the last 1/10th of Y (after applying shfl labels to it)
            testTrInds = shfl[np.arange(numTrials-len_test, numTrials)] # indeces to be applied on trsnow in order to get the trials (index out of Y0) that were used as testing trs; eg stimrate[trsnow[testTrInds]] is the stimrate of testing trials
#            testTrInds_outOfY0 = trsnow[testTrInds] # index of testing trials out of Y0 (not Y!) (that will be used in svm below)
             ######## IMPORTANT: Ybefshfl[testTrInds] is same as Y0[trsnow[testTrInds]] and same as Y[np.arange(numTrials-len_test, numTrials)] and same as summary.YTest computed below ########

            testTrInds_allSamps[s,:] = testTrInds            
            
#            print(np.equal(Y0[testTrInds], Ytest_allSamps[s])) # sanity check. must be True
#            testTrInds_outOfY0_allSamps[s,:] = testTrInds_outOfY0            
        else:
            testTrInds_allSamps = np.nan # for now, but to set it correctly: testTrInds will be set in crossValidateModel.py, you just need to output it from crossValidateModel
            Ytest_allSamps[s,:] = np.nan  

        
        
        
        ########################## Start training SVM ##########################
        ########################
        ########################
        ########################
#        for ifr in frs: # train SVM on each frame            
        if bestcProvided:
            cvect = [cbest] #[cbest[ifr]]
    
#            print('\tFrame %d' %(ifr)  
        #%%######################## Loop over different values of regularization
        for i in range(nCvals): # i = 0 # train SVM using different values of regularization parameter
            if regType == 'l1':                               
                summary,_ =  crossValidateModel(X.transpose(), Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
                
            elif regType == 'l2':
                summary,_ =  crossValidateModel(X.transpose(), Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = shuffleTrs)
                    
            '''
            ###### below is the gist of the codes for svm training (done in linearSVM.py) ######
            
            import sklearn.svm as svm                        
            
            linear_svm = svm.LinearSVC(C = cbest_allExc[12], loss='squared_hinge', penalty='l2', dual=True) # get c for a frame, eg frame 12.
            linear_svm.fit(X_svm[12,:,:].transpose(), np.squeeze(Y_svm)) # x should be in trials x units
            
            linear_svm.predict(XTest)
            linear_svm.coef_
            linear_svm.intercept_
            
            ###
            # def perClassError(Y, Yhat):
            #    import numpy as np
            #    perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
            #    return perClassEr
            
            # perClassErrorTest = perClassError(YTest, linear_svm.predict(XTest)); 
            
            # summary.model = linear_svm;
            '''
                
            wAllC[s,i,:] = np.squeeze(summary.model.coef_) # weights of all neurons for each value of c and each shuffle
            bAllC[s,i] = np.squeeze(summary.model.intercept_)
    
            # classification errors                    
            perClassErrorTrain[s,i] = summary.perClassErrorTrain
            perClassErrorTest[s,i] = summary.perClassErrorTest # perClassError(YTest, linear_svm.predict(XTest));
            
            # Testing correct shuffled data: same decoder trained on correct trials, but use shuffled trial labels to compute class error
            Ytest_hat = summary.model.predict(summary.XTest) # prediction of trial label for each trial
            perClassErrorTest_shfl[s,i] = perClassError(summary.YTest[permIxs], Ytest_hat) # fraction of incorrect predicted trial labels
            perClassErrorTest_chance[s,i] = perClassError(Y_chance, Ytest_hat)
            Ytest_hat_allSampsFrs[s,i,:] = Ytest_hat
            
            # Incorrect trials
#                if testIncorr:
#                    Ytest_hat_incorr = summary.model.predict(X_svm_incorr[ifr,:,:].transpose()) # prediction of trial label for each trial
#                    perClassErrorTest_incorr[s,i] = perClassError(Y_svm_incorr, Ytest_hat_incorr) # fraction of incorrect predicted trial labels
#                    perClassErrorTest_shfl_incorr[s,i] = perClassError(Y_svm_incorr[permIxs_incorr], Ytest_hat_incorr) # fraction of incorrect predicted trial labels
#                    perClassErrorTest_chance_incorr[s,i] = perClassError(Y_chance_incorr, Ytest_hat_incorr)
#                    Ytest_hat_allSampsFrs_incorr[s,i,:] = Ytest_hat_incorr
            
            
            ########## sanity check ##########
            """
            trsnow = trsnow_allSamps[s].astype(int)
            testTrInds = testTrInds_allSamps[s].astype(int)
            testTrInds_outOfY0 = trsnow[testTrInds]
            xx = X0[ifr][:,testTrInds_outOfY0]        
            yy = Y0[testTrInds_outOfY0]
            
            ww = wAllC[s,i,:,ifr]
#                normw = sci.linalg.norm(ww)   # numSamps x numFrames
#                ww = ww / normw                 
            
            bb = bAllC[s,i,ifr] 
            
            # Project population activity of each frame onto the decoder of frame ifr
            yhat = np.dot(ww, xx) + bb # testingFrs x testing trials                
            th = 0
            yhat[yhat<th] = 0 # testingFrs x testing trials
            yhat[yhat>th] = 1
                            
            d = yhat - yy  # testing Frs x nTesting Trials # difference between actual and predicted y
            c = np.mean(abs(d), axis=-1) * 100

            eqy[ifr, s] = np.equal(c, perClassErrorTest[s,i,ifr])                
            
            if eqy[ifr, s]==0:
                print(np.mean(np.equal(xx.T, summary.XTest))
                print(np.mean(np.equal(yy, summary.YTest))
                print(np.mean(np.equal(yhat, Ytest_hat))
                print(ifr, s
                print(c, perClassErrorTest[s,i,ifr]
                sys.exit('Error!') 
            """


    #%%
    ######################### Find bestc for each frame, and plot the c path 
    if bestcProvided: 
        cbestAllFrs = cbest
        cbestFrs = cbest
        
    else:
        print('--------------- Identifying best c ---------------')
#        cbestFrs = np.full((X.shape[0]), np.nan)  
#        cbestAllFrs = np.full((X.shape[0]), np.nan)  
        
#        for ifr in frs: #range(X.shape[0]):    
        #######%% Compute average of class errors across numSamples        
        meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:], axis = 0);
        semPerClassErrorTrain = np.std(perClassErrorTrain[:,:], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest = np.mean(perClassErrorTest[:,:], axis = 0);
        semPerClassErrorTest = np.std(perClassErrorTest[:,:], axis = 0)/np.sqrt(numSamples);
        
        meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:], axis = 0);
#            semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:], axis = 0)/np.sqrt(numSamples);            
        meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:], axis = 0);
#            semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:], axis = 0)/np.sqrt(numSamples);
        
        
        #######%% Identify best c                
        # Use all range of c... it may end up a value at which all weights are 0.
        ix = np.argmin(meanPerClassErrorTest)
        if smallestC==1:
            cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])];
            cbest = cbest[0]; # best regularization term based on minError+SE criteria
            cbestAll = cbest
        else:
            cbestAll = cvect[ix]
#        print('\tFrame %d: %f' %(ifr,cbestAll))
        print('\t%f' %(cbestAll))
#        cbestAllFrs[ifr] = cbestAll
        cbestAllFrs = cbestAll
        
        
        
        ####### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
        if regType == 'l1': # in l2, we don't really have 0 weights!
            sys.exit('Needs work! below wAllC has to be for 1 frame') 
            
            a = abs(wAllC)>eps # non-zero weights
            b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
            c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
            cvectnow = cvect[c1stnon0:]
            
            meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:], axis = 0);
            semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:], axis = 0)/np.sqrt(numSamples);
            ix = np.argmin(meanPerClassErrorTestnow)
            if smallestC==1:
                cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])];
                cbest = cbest[0]; # best regularization term based on minError+SE criteria    
            else:
                cbest = cvectnow[ix]
            
            print('best c (at least 1 non-0 weight) = ', cbest)
        else:
            cbest = cbestAll
                
#        cbestFrs[ifr] = cbest
        cbestFrs = cbest
        
        
        ########%% Set the decoder and class errors at best c (for data)
        """
        # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
        # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
        indBestC = np.in1d(cvect, cbest)
        
        w_bestc_data = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
        b_bestc_data = bAllC[:,indBestC,ifr]
        
        classErr_bestC_train_data = perClassErrorTrain[:,indBestC,ifr].squeeze()
        
        classErr_bestC_test_data = perClassErrorTest[:,indBestC,ifr].squeeze()
        classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
        classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,ifr].squeeze()
        """
        
        
        #%%
        ########### Plot C path    
        if doPlots:
            import matplotlib.pyplot as plt
    #        print('Best c (inverse of regularization parameter) = %.2f' %cbest
            plt.figure()
#            plt.subplot(1,2,1)
            plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
            plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
        #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
        #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
            
            plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
            plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
            plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
            plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'cv-shfl')            
        
            plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
            
            plt.xlim([cvect[1], cvect[-1]])
            plt.xscale('log')
            plt.xlabel('c (inverse of regularization parameter)')
            plt.ylabel('classification error (%)')
            plt.legend(loc='center left', bbox_to_anchor=(1, .7))
            
#            plt.title('Frame %d' %(ifr))
            plt.tight_layout()
    
    
    
    #%%
    ##############

#    if testIncorr:
#        return perClassErrorTrain, perClassErrorTest, wAllC, bAllC, cbestAllFrs, cbestFrs, cvect, perClassErrorTest_shfl, perClassErrorTest_chance, testTrInds_allSamps, Ytest_allSamps, Ytest_hat_allSampsFrs, trsnow_allSamps, perClassErrorTest_incorr, perClassErrorTest_shfl_incorr, perClassErrorTest_chance_incorr, Ytest_hat_allSampsFrs_incorr 
#    else:
    return perClassErrorTrain, perClassErrorTest, wAllC, bAllC, cbestAllFrs, cbestFrs, cvect, perClassErrorTest_shfl, perClassErrorTest_chance, testTrInds_allSamps, Ytest_allSamps, Ytest_hat_allSampsFrs, trsnow_allSamps














#%% Function to run SVM  (when X is frames x units x trials)
# Remember each numSamples will have a different set of training and testing dataset, however for each numSamples, the same set of testing/training dataset
# will be used for all frames and all values of c (unless shuffleTrs is 1, in which case different frames and c values will have different training/testing datasets.)

def set_best_c_diffNumNeurons(X,Y,regType,kfold,numDataPoints,numSamples,population_sizes_to_try,doPlots,useEqualTrNums,smallestC,shuffleTrs,cbest=np.nan,fr2an=np.nan, shflTrLabs=0, X_svm_incorr=0, Y_svm_incorr=0, mnHRLR_acrossDays=np.nan):
    
    '''
    X = X_svm
    Y = Y_svm
    cbest = np.nan
    fr2an = np.nan
    shflTrLabs = 0
    X_svm_incorr = 0
    Y_svm_incorr = 0
    mnHRLR_acrossDays = np.nan
    '''
    # numSamples = 10 # number of iterations for finding the best c (inverse of regularization parameter)
    # if you don't want to regularize, go with a very high cbest and don't run the section below.
    # cbest = 10**6    
    # regType = 'l1'
    # kfold = 10;


    #%%            
#    import numpy as np
    import numpy.random as rng
    
    def perClassError(Y, Yhat):
        perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr

    
    #%% Set c values (regularization parameter)
    
    if np.isnan(cbest).all(): # we need to set cbest
        bestcProvided = False        
        if regType == 'l1':
            print('\n-------------- Running l1 svm classification --------------\r') 
            cvect = 10**(np.arange(-4, 6,0.2)) / numDataPoints
        elif regType == 'l2':
            print('\n-------------- Running l2 svm classification --------------\r') 
            cvect = 10**(np.arange(-6, 6,0.2)) / numDataPoints          
        nCvals = len(cvect)
#        print('try the following regularization values: \n', cvect)
    else: # bestc is provided and we want to fit svm on shuffled trial labels
        bestcProvided = True           
        nCvals = 1 # cbest is already provided
       
        
    #%%
#    smallestC = 0 
    # if 1: the smallest c whose CV error falls below 1 se of min CV error will be used as optimal C
    # if 0: c that gives min CV error will be used as optimal c.

    if smallestC==1:
        print('bestc = smallest c whose cv error is less than 1se of min cv error')
    else:
        print('bestc = c that gives min cv error')
    #I think we should go with min c as the bestc... at least we know it gives the best cv error... and it seems like it has nothing to do with whether the decoder generalizes to other data or not.
            
    
    #%%    
    numTrials = X.shape[1]
    print('FINAL: %d trials; %d neurons' %(numTrials, X.shape[0]))
    
    len_test = numTrials - int((kfold-1.)/kfold*numTrials) # number of testing trials   
            
    X0 = X + 0 # units x trials
    Y0 = Y + 0
    
    
    #%%    
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    
    #%% Loop over population size, ie number of neurons to use in SVM training # nN = 1:totalNumNeurons     
    # right now, we are not changing the number of neurons in the population...
    # but the codes are ready to assess how changing the population size affects classification accuracy.

    nnt = X0.shape[0] #X_svm0.shape[1] # total number of neurons    
#    population_sizes_to_try = [nN_trainSVM] # [x+1 for x in range(X0.shape[0])] : if you want to try all possible population sizes
    
    # all lists below have length len(population_sizes_to_try); each element has size
    # numShufflesN x nSamples x nFrs
        # wAllC_nN_all : each cell has size: numShufflesN x nSamples x nNerons_used_for_training x nFrs
    # eg. perClassErrorTest_shfl_nN_all[nN][inN] = (numSamples x nfr) shows when we subselected nN+1 neurons out of all neurons, 
    # in the inN-th iteration, we got numSamples of classAccuracy values for each of the nfr frames (coming from numSamples of subselecting trials).
    
    # Each element of the following arrays includes svm related vars for a population of a given size (nN)    
    wAllC_nN_all = []
    bAllC_nN_all = []
    perClassErrorTrain_nN_all = []
    perClassErrorTest_nN_all = []
    perClassErrorTest_shfl_nN_all = []
    perClassErrorTest_chance_nN_all = []
    inds_subselected_neurons_nN_all = []
    
#    cbest = 10**6  # if you don't want to regularize, go with a very high cbest
        
    for nN in population_sizes_to_try:  # nN = nN_trainSVM
    
        # Select nN random neurons from X, do this numShufflesN times
        numShufflesN = np.ceil(nnt/float(nN)).astype(int) # if you are selecting only 1 neuron out of 500 neurons, you will do this 500 times to get a selection of all neurons. On the other hand if you are selecting 400 neurons out of 500 neurons, you will do this only twice.
        print('Subsampling %d out of %d neurons %d times....' %(nN, nnt, numShufflesN))
        
        # CHECK THE SIZE OF ARRAYS BELOW
        wAllC = np.full((numShufflesN, numSamples, nCvals, nN), np.nan)
        bAllC = np.full((numShufflesN, numSamples, nCvals), np.nan)
        perClassErrorTrain = np.full((numShufflesN, numSamples, nCvals), np.nan)
        perClassErrorTest = np.full((numShufflesN, numSamples, nCvals), np.nan)
        perClassErrorTest_shfl = np.full((numShufflesN, numSamples, nCvals), np.nan)
        perClassErrorTest_chance = np.full((numShufflesN, numSamples, nCvals), np.nan)
        
        inds_subselected_neurons = np.full((numShufflesN, nN), np.nan) # index of neurons in X0 that were used for training svm
        
        # you have to add numShufflesN to the arrays below if you actually want to set them here
#        testTrInds_allSamps = np.full((numSamples, len_test), np.nan)              
#        Ytest_allSamps = np.full((numSamples, len_test), np.nan)              
#        Ytest_hat_allSampsFrs = np.full((numSamples, nCvals, len_test), np.nan)        
#        trsnow_allSamps = np.full((numSamples, numTrials), np.nan)              
        
        
        #%% Loop over subselecting nN number of neurons out of all neurons
        ######## (subsampling neurons) ########
        
        for inN in range(numShufflesN):   # eg. for 20 iterations subselect 10 neuron out of 200 neurons # inN = 0
            print('\tIteration %d/%d of subsampling neurons' %(inN+1, numShufflesN))
            
            if nN==1: # when only training SVM with one neuron, go through neurons from begining to end, no need to do subselection!            
                Xnow = X0[[inN],:]
            else: # subselect nN number of neurons (so neuron orders get shuffled here!)
                inds = rng.permutation(nnt)
                inds = inds[range(nN)]
                Xnow = X0[inds,:]
                inds_subselected_neurons[inN] = inds # index of neurons in X0 that were used for training svm
                
    
            #%%################# Loop over subselecting testing/training trials... to get numSamples cross-validated datasets ####################     
            ######## (subsampling trials) ########

            print('\tSubsampling testing/training trials %d times....' %(numSamples))
            
            for s in range(numSamples): # s = 0       
                print('\t\tIteration %d' %(s))
                        
                trsnow = np.arange(0, len(Y0))
                X = Xnow[:,trsnow] # trsnow : index of trials (out of Xnow and Y0) that are used to set X and Y
                Y = Y0[trsnow]
        
#                trsnow_allSamps[s,:] = trsnow
        #        numTrials, numNeurons = X.shape[2], X.shape[1]
        #            print('FINAL: %d trials; %d neurons' %(numTrials, numNeurons)                        
                    
                ######################## Setting chance Y: same length as Y for testing data, and with equal number of classes 0 and 1.
        #        no = Y.shape[0]
        #        len_test = numTrials - int((kfold-1.)/kfold*numTrials)    
                permIxs = rng.permutation(len_test) # needed to set perClassErrorTest_shfl   
            
                Y_chance = np.zeros(len_test)
                if rng.rand()>.5:
                    b = rng.permutation(len_test)[0:np.floor(len_test/float(2)).astype(int)]
                else:
                    b = rng.permutation(len_test)[0:np.ceil(len_test/float(2)).astype(int)]
                Y_chance[b] = 1
        
                    
                ####################### Set the chance Y for training SVM on shuffled trial labels
                if shflTrLabs: # shuffle trial classes in Y
                    Y = np.zeros(numTrials) # Y_chance0
                    if rng.rand()>.5:
                        b = rng.permutation(numTrials)[0:np.floor(numTrials/float(2)).astype(int)]
                    else:
                        b = rng.permutation(numTrials)[0:np.ceil(numTrials/float(2)).astype(int)]
                    Y[b] = 1
        
                
                ######################## Shuffle trial orders, so the training and testing datasets are different for each numSamples (we only do this if shuffleTrs is 0, so crossValidateModel does not shuffle trials, so we have to do it here, otherwise all numSamples will have the same set of testing and training datasets.)
                ######################## REMEMBER : YOU ARE CHANGING THE ORDER OF TRIALS HERE!!! ########################        
                if shuffleTrs==0: # shuffle trials here (instead of inside crossValidateModel) to break any dependencies on the sequence of trails 
                    
        #            Ybefshfl = Y            
                    shfl = rng.permutation(np.arange(0, numTrials)) # shfl: new order of trials ... shuffled indeces of Y... the last 1/10th indeces will be testing trials.
                    
                    Y = Y[shfl] 
                    X = X[:,shfl]             
                    
                    # Ytest_allSamps[s,:] : Y that will be used as testing trials in this sample
#                    Ytest_allSamps[s,:] = Y[np.arange(numTrials-len_test, numTrials)] # the last 1/10th of Y (after applying shfl labels to it)
#                    testTrInds = shfl[np.arange(numTrials-len_test, numTrials)] # indeces to be applied on trsnow in order to get the trials (index out of Y0) that were used as testing trs; eg stimrate[trsnow[testTrInds]] is the stimrate of testing trials
        #            testTrInds_outOfY0 = trsnow[testTrInds] # index of testing trials out of Y0 (not Y!) (that will be used in svm below)
                     ######## IMPORTANT: Ybefshfl[testTrInds] is same as Y0[trsnow[testTrInds]] and same as Y[np.arange(numTrials-len_test, numTrials)] and same as summary.YTest computed below ########
        
#                    testTrInds_allSamps[s,:] = testTrInds            
                    
        #            print(np.equal(Y0[testTrInds], Ytest_allSamps[s])) # sanity check. must be True
        #            testTrInds_outOfY0_allSamps[s,:] = testTrInds_outOfY0            
#                else:
#                    testTrInds_allSamps = np.nan # for now, but to set it correctly: testTrInds will be set in crossValidateModel.py, you just need to output it from crossValidateModel
#                    Ytest_allSamps[s,:] = np.nan  
        
                
        #        for ifr in frs: # train SVM on each frame            
                if bestcProvided:
                    cvect = [cbest] #[cbest[ifr]]    
        #            print('\tFrame %d' %(ifr)  
                
                
                #%%######################## Loop over different values of regularization ########################
                ####### Start training SVM here #######
                
                print('\t\t\tRunning SVM on %d C values...' %nCvals)
                for i in range(nCvals): # i = 0 # train SVM using different values of regularization parameter
                    if regType == 'l1':                               
                        summary,_ =  crossValidateModel(X.transpose(), Y, linearSVM, kfold = kfold, l1 = cvect[i], shflTrs = shuffleTrs)
                        
                    elif regType == 'l2':
                        summary,_ =  crossValidateModel(X.transpose(), Y, linearSVM, kfold = kfold, l2 = cvect[i], shflTrs = shuffleTrs)
                            
                        
                    wAllC[inN,s,i,:] = np.squeeze(summary.model.coef_) # weights of all neurons for each value of c and each shuffle
                    bAllC[inN,s,i] = np.squeeze(summary.model.intercept_)
            
                    # classification errors                    
                    perClassErrorTrain[inN,s,i] = summary.perClassErrorTrain
                    perClassErrorTest[inN,s,i] = summary.perClassErrorTest # perClassError(YTest, linear_svm.predict(XTest));
                    
                    # Testing correct shuffled data: same decoder trained on correct trials, but use shuffled trial labels to compute class error
                    Ytest_hat = summary.model.predict(summary.XTest) # prediction of trial label for each trial
                    perClassErrorTest_shfl[inN,s,i] = perClassError(summary.YTest[permIxs], Ytest_hat) # fraction of incorrect predicted trial labels
                    perClassErrorTest_chance[inN,s,i] = perClassError(Y_chance, Ytest_hat)
#                    Ytest_hat_allSampsFrs[s,i,:] = Ytest_hat
                                
                    
                    ###### below is the gist of the codes for svm training (used in linearSVM.py) ######
                    '''
                    import sklearn.svm as svm                        
                    
                    linear_svm = svm.LinearSVC(C = cbest_allExc[12], loss='squared_hinge', penalty='l2', dual=True) # get c for a frame, eg frame 12.
                    linear_svm.fit(X_svm[12,:,:].transpose(), np.squeeze(Y_svm)) # x should be in trials x units
                    
                    linear_svm.predict(XTest)
                    linear_svm.coef_
                    linear_svm.intercept_
                    
                    ###
                    # def perClassError(Y, Yhat):
                    #    import numpy as np
                    #    perClassEr = np.sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
                    #    return perClassEr
                    
                    # perClassErrorTest = perClassError(YTest, linear_svm.predict(XTest)); 
                    
                    # summary.model = linear_svm
                    '''
                    
                    
                    ########## sanity checks ##########
                    """
                    trsnow = trsnow_allSamps[s].astype(int)
                    testTrInds = testTrInds_allSamps[s].astype(int)
                    testTrInds_outOfY0 = trsnow[testTrInds]
                    xx = X0[ifr][:,testTrInds_outOfY0]        
                    yy = Y0[testTrInds_outOfY0]
                    
                    ww = wAllC[s,i,:,ifr]
        #                normw = sci.linalg.norm(ww)   # numSamps x numFrames
        #                ww = ww / normw                 
                    
                    bb = bAllC[s,i,ifr] 
                    
                    # Project population activity of each frame onto the decoder of frame ifr
                    yhat = np.dot(ww, xx) + bb # testingFrs x testing trials                
                    th = 0
                    yhat[yhat<th] = 0 # testingFrs x testing trials
                    yhat[yhat>th] = 1
                                    
                    d = yhat - yy  # testing Frs x nTesting Trials # difference between actual and predicted y
                    c = np.mean(abs(d), axis=-1) * 100
        
                    eqy[ifr, s] = np.equal(c, perClassErrorTest[s,i,ifr])                
                    
                    if eqy[ifr, s]==0:
                        print(np.mean(np.equal(xx.T, summary.XTest))
                        print(np.mean(np.equal(yy, summary.YTest))
                        print(np.mean(np.equal(yhat, Ytest_hat))
                        print(ifr, s
                        print(c, perClassErrorTest[s,i,ifr]
                        sys.exit('Error!') 
                    """
                    
                    
        #%% Each element of the following arrays includes svm related vars for a population of a given size (population_sizes_to_try)
        
        wAllC_nN_all.append(wAllC) # each element has size: numShufflesN x nSamples x nCvals x nNerons_used_for_training
        bAllC_nN_all.append(bAllC) # each element has size: # numShufflesN x nSamples x nCvals
        perClassErrorTrain_nN_all.append(perClassErrorTrain) # each element has size: # numShufflesN x nSamples x nCvals
        perClassErrorTest_nN_all.append(perClassErrorTest)                        
        perClassErrorTest_shfl_nN_all.append(perClassErrorTest_shfl)
        perClassErrorTest_chance_nN_all.append(perClassErrorTest_chance)
        inds_subselected_neurons_nN_all.append(inds_subselected_neurons) # each element has size: numShufflesN x nNerons_used_for_training
        
        # eg. perClassErrorTest_shfl_nN_all[nN][inN] = (numSamples x nfr) shows when we subselected nN+1 neurons out of all neurons, 
        # in the inN-th iteration, we got numSamples of classAccuracy values for each of the nfr frames (coming from numSamples of subselecting trials).
       
        

    #%% Find bestc for each frame, and plot the c path 
    
#    numShufflesN, numSamples, nCvals, nN
#    numShufflesN, numSamples, nCvals
        
    if bestcProvided: 
        cbestAllFrs = cbest
#        cbestFrs = cbest
        
    else:
        print('--------------- Identifying best c ---------------')        
        cbestAllFrs_nN_all = [] 
        # cbestAllFrs_nN_all: each element is for a population of a given size; 
        # cbestAllFrs_nN_all[0]: each element is the best c (the c that gave min error on average across numSamples) for training svm on a given neuron subselect
        
        for i_pop_size in range(len(population_sizes_to_try)):  # nN = nN_trainSVM            
            cbestAllFrs_all = np.full((numShufflesN), np.nan)            

            for inN in range(numShufflesN): # inN = 0                
                perClassErrorTrain = perClassErrorTrain_nN_all[i_pop_size][inN] # each element of perClassErrorTrain_nN_all has size: # numShufflesN x nSamples x nCvals
                perClassErrorTest = perClassErrorTest_nN_all[i_pop_size][inN]
                perClassErrorTest_shfl = perClassErrorTest_shfl_nN_all[i_pop_size][inN]
                perClassErrorTest_chance = perClassErrorTest_chance_nN_all[i_pop_size][inN]
                
        #        cbestFrs = np.full((X.shape[0]), np.nan)  
        #        cbestAllFrs = np.full((X.shape[0]), np.nan)                  
        #        for ifr in frs: #range(X.shape[0]):    
                #######%% Compute average of class errors across numSamples        
                meanPerClassErrorTrain = np.mean(perClassErrorTrain[:,:], axis = 0)
                semPerClassErrorTrain = np.std(perClassErrorTrain[:,:], axis = 0)/np.sqrt(numSamples)
                
                meanPerClassErrorTest = np.mean(perClassErrorTest[:,:], axis = 0)
                semPerClassErrorTest = np.std(perClassErrorTest[:,:], axis = 0)/np.sqrt(numSamples)
                
                meanPerClassErrorTest_shfl = np.mean(perClassErrorTest_shfl[:,:], axis = 0)
        #            semPerClassErrorTest_shfl = np.std(perClassErrorTest_shfl[:,:], axis = 0)/np.sqrt(numSamples)
                meanPerClassErrorTest_chance = np.mean(perClassErrorTest_chance[:,:], axis = 0)
        #            semPerClassErrorTest_chance = np.std(perClassErrorTest_chance[:,:], axis = 0)/np.sqrt(numSamples)
                
                
                ####### Identify best c #######               
                # Use all range of c... it may end up a value at which all weights are 0.
                ix = np.argmin(meanPerClassErrorTest)
                if smallestC==1:
                    cbest = cvect[meanPerClassErrorTest <= (meanPerClassErrorTest[ix]+semPerClassErrorTest[ix])]
                    cbest = cbest[0] # best regularization term based on minError+SE criteria
    #                cbestAll = cbest
                else:
                    cbest = cvect[ix]
    #                cbestAll = cvect[ix]
        #        print('\tFrame %d: %f' %(ifr,cbestAll))
                print('Best C for neuron subselect %d = %f' %(inN, cbest))
        #        cbestAllFrs[ifr] = cbestAll
                cbestAllFrs = cbest#All
                
                cbestAllFrs_all[inN] = cbestAllFrs # each element has size: numShufflesN
                            
                ####### Make sure at bestc at least one weight is non-zero (ie pick bestc from only those values of c that give non-0 average weights.)
                '''
                if regType == 'l1': # in l2, we don't really have 0 weights!
                    sys.exit('Needs work! below wAllC has to be for 1 frame') 
                    
                    a = abs(wAllC)>eps # non-zero weights
                    b = np.mean(a, axis=(0,2,3)) # Fraction of non-zero weights (averaged across shuffles)
                    c1stnon0 = np.argwhere(b)[0].squeeze() # first element of c with at least 1 non-0 w in 1 shuffle
                    cvectnow = cvect[c1stnon0:]
                    
                    meanPerClassErrorTestnow = np.mean(perClassErrorTest[:,c1stnon0:], axis = 0);
                    semPerClassErrorTestnow = np.std(perClassErrorTest[:,c1stnon0:], axis = 0)/np.sqrt(numSamples)
                    ix = np.argmin(meanPerClassErrorTestnow)
                    if smallestC==1:
                        cbest = cvectnow[meanPerClassErrorTestnow <= (meanPerClassErrorTestnow[ix]+semPerClassErrorTestnow[ix])]
                        cbest = cbest[0] # best regularization term based on minError+SE criteria    
                    else:
                        cbest = cvectnow[ix]
                    
                    print('best c (at least 1 non-0 weight) = ', cbest)
                else:
                    cbest = cbestAll
                        
        #        cbestFrs[ifr] = cbest
                cbestFrs = cbest
                '''
                
                ########%% Set the decoder and class errors at best c (for data)
                """
                # you don't need to again train classifier on data bc you already got it above when you found bestc. You just need to do it for shuffled. ... [you already have access to test/train error as well as b and w of training SVM with bestc.)]
                # we just get the values of perClassErrorTrain and perClassErrorTest at cbest (we already computed these values above when training on all values of c)
                indBestC = np.in1d(cvect, cbest)
                
                w_bestc_data = wAllC[:,indBestC,:,ifr].squeeze() # numSamps x neurons
                b_bestc_data = bAllC[:,indBestC,ifr]
                
                classErr_bestC_train_data = perClassErrorTrain[:,indBestC,ifr].squeeze()
                
                classErr_bestC_test_data = perClassErrorTest[:,indBestC,ifr].squeeze()
                classErr_bestC_test_shfl = perClassErrorTest_shfl[:,indBestC,ifr].squeeze()
                classErr_bestC_test_chance = perClassErrorTest_chance[:,indBestC,ifr].squeeze()
                """
                
                
                #%% Plot C path    
                
                if doPlots:
                    import matplotlib.pyplot as plt
            #        print('Best c (inverse of regularization parameter) = %.2f' %cbest
                    plt.figure()
        #            plt.subplot(1,2,1)
                    plt.fill_between(cvect, meanPerClassErrorTrain-semPerClassErrorTrain, meanPerClassErrorTrain+ semPerClassErrorTrain, alpha=0.5, edgecolor='k', facecolor='k')
                    plt.fill_between(cvect, meanPerClassErrorTest-semPerClassErrorTest, meanPerClassErrorTest+ semPerClassErrorTest, alpha=0.5, edgecolor='r', facecolor='r')
                #    plt.fill_between(cvect, meanPerClassErrorTest_chance-semPerClassErrorTest_chance, meanPerClassErrorTest_chance+ semPerClassErrorTest_chance, alpha=0.5, edgecolor='b', facecolor='b')        
                #    plt.fill_between(cvect, meanPerClassErrorTest_shfl-semPerClassErrorTest_shfl, meanPerClassErrorTest_shfl+ semPerClassErrorTest_shfl, alpha=0.5, edgecolor='y', facecolor='y')        
                    
                    plt.plot(cvect, meanPerClassErrorTrain, 'k', label = 'training')
                    plt.plot(cvect, meanPerClassErrorTest, 'r', label = 'validation')
                    plt.plot(cvect, meanPerClassErrorTest_chance, 'b', label = 'cv-chance')       
                    plt.plot(cvect, meanPerClassErrorTest_shfl, 'y', label = 'cv-shfl')            
                
                    plt.plot(cvect[cvect==cbest], meanPerClassErrorTest[cvect==cbest], 'bo')
                    
                    plt.xlim([cvect[1], cvect[-1]])
                    plt.xscale('log')
                    plt.xlabel('c (inverse of regularization parameter)')
                    plt.ylabel('classification error (%)')
                    plt.legend(loc='center left', bbox_to_anchor=(1, .7))
                    
        #            plt.title('Frame %d' %(ifr))
                    plt.tight_layout()
                
        
        #%%                
        cbestAllFrs_nN_all.append(cbestAllFrs_all)
            
        
                
    #%%    
    return perClassErrorTrain_nN_all, perClassErrorTest_nN_all, wAllC_nN_all, bAllC_nN_all, \
        cbestAllFrs_nN_all, cvect, \
        perClassErrorTest_shfl_nN_all, perClassErrorTest_chance_nN_all,\
        inds_subselected_neurons_nN_all, numShufflesN
        







