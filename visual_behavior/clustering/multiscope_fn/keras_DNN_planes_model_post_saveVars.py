#%% Gets called in keras_DNN_planes_model_setVars (fit_dnn_model = 0)
    
from def_funs import *
from keras_funs import *
    
def keras_DNN_planes_model_post_saveVars(traces_x0, traces_y0, traces_y0_evs, inds_final_all, x_ind, y_ind, len_win, kfold, dir_now):
        
    #%% Set loss, r_square, y_prediction variables for each neuron and save them

    get_ipython().magic(u'matplotlib inline')

    n_r_sq_rand = 50 # number of times to computed explained variance for a random vector
    
    loss_bestModel_all = np.full(traces_y0.shape[0], np.nan)
    val_loss_bestModel_all = np.full(traces_y0.shape[0], np.nan)

    err_mx_all = np.full(traces_y0.shape[0], np.nan)
    val_err_mx_all = np.full(traces_y0.shape[0], np.nan)
    
    r_sq_train_all = np.full(traces_y0.shape[0], np.nan)
    r_sq_test_all = np.full(traces_y0.shape[0], np.nan)
    r_sq_alltrace_all = np.full(traces_y0.shape[0], np.nan)
    r_sq_rand_all = np.full((traces_y0.shape[0], n_r_sq_rand), np.nan)
    
    pmw_rand_test = np.full(traces_y0.shape[0], np.nan)
    pt1_rand_test = np.full(traces_y0.shape[0], np.nan)
    
    y_pred_train_all = [[]]*traces_y0.shape[0]    
    y_pred_test_all = [[]]*traces_y0.shape[0]
    y_pred_alltrace_all = [[]]*traces_y0.shape[0]

    this_fold = os.path.basename(dir_now)
    traces_x00 = traces_x0+0
    
    for neuron_y in np.arange(0, traces_y0.shape[0]): # traces_y0.shape[0]  # neuron_y = 0

        print('\n======================== Plane %d to %d; loading data for neuron %d/%d in y trace ========================\n' %(x_ind, y_ind, neuron_y, traces_y0.shape[0]))
        
#         try:
        gc.collect()

        # for each neuron, load the best autoencoder model (the one that was saved by modelcheckpoint, which gave the min val_loss)
        aname = 'plane%dto%d_neuronY_%03d_model_ep_' %(x_ind, y_ind, neuron_y)
        [modelName, h5_files] = all_sess_set_h5_fileName(aname, dir_now, all_files=0)    
        
        if modelName!='': # if it is '', then this neuron did not have enough training/testing data, and no model was fit on it.
            autoencoder = load_model(modelName)
#             K.clear_session()

            # load the training, testing, etc data for this neuron
            dname = 'neuronY_%03d_model_data' %neuron_y
            [dataName, h5_files] = all_sess_set_h5_fileName(dname, dir_now, all_files=0)            
    #         if dataName!='':
            model_data = pd.read_hdf(dataName, key='model_ae')
            #    f = h5py.File(dirmov, 'r')    #mov = f[list(f.keys())[0]]
            #    mov = f.get('data') 

            # set some model params
        #    model_hist = model_data.iloc[0]['model_hist']
        #    model_hist_params = model_hist.params    
        #    model_hist_hist = model_hist.history
            model_hist_params = model_data.iloc[0]['model_hist_params']    
            model_hist_hist = model_data.iloc[0]['model_hist_hist']
            loss_batches = np.array(model_data.iloc[0]['loss_batches'])    

            batch_size = model_hist_params['batch_size']
        #    epochs = model_hist_params['epochs']


            ####### if doing within-plane prediction, remove neuron_y from traces_x0
            if x_ind==y_ind: 
                print('Within plane prediction, hence, removing neuron_y from x population:')
                no_ny0 = np.arange(0, traces_x00.shape[0])
                no_ny = no_ny0[~np.in1d(no_ny0, neuron_y)]

                traces_x0 = traces_x00[no_ny,:]
                print(traces_x0.shape)


            ####### set traing and testing datasets (x and y) for the model
            a0 = model_data.iloc[0]['train_data_inds']
            a1 = model_data.iloc[0]['test_data_inds']
            set_train_test_inds = [a0, a1]

            [x_train, x_test, y_train, y_test, _, _] = set_x_y_train_test(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, kfold, set_train_test_inds, plotPlots=0)


            ####### set the control distribution of r_sqr for the testing data of each neuron: predict a random vector for x_test    
            # create a random vector, 50 times, and compute r2 for the prediction of this random vector from x_test data    
            r_sq_rand_eachN = []
            for irand in range(n_r_sq_rand):
                y_true = rnd.normal(y_test.mean(), y_test.std(), y_test.shape)    
                [r_sq_rand, y_pred_rand] = r2_yPred(x_test, y_true, autoencoder, loss_provided=[0])
        #        print('explained variance for y_rand: ', r_sq_rand)
                r_sq_rand_eachN.append(r_sq_rand)    


            # set max error: when there is no prediction, so max error will be sum(y^2).
            err_mx = set_max_error(y_train)
            val_err_mx = set_max_error(y_test)       
            print('\nmax error, training, testing data:')
            print(err_mx, val_err_mx)


            #### loss and val_loss for all epochs, computed during model fitting
            # set loss    
            loss = np.array(model_hist_hist['loss']) # this value is affected by dropout and batchNorm layers; use model.evaluate to get the actuall loss value.
            val_loss = np.array(model_hist_hist['val_loss'])
            ep = np.arange(0, loss.shape[0]) #np.array(model_hist.epoch)    
        #    mae = np.array(model_hist_hist['mae'])
        #    val_mae = np.array(model_hist_hist['val_mae'])

            # set loss values for all batches of each epoch    
            len_train = x_train.shape[0]
            [loss_eachEpoch_allBatches, n_batches_per_epoch, epochs_completed] = set_loss_allBatches(loss_batches, batch_size, len_train)    
            # training data: compute mean and sem of loss across batches, for each epoch        
            loss_eachEpoch_aveBatches = np.mean(loss_eachEpoch_allBatches, axis=0) # meanErrorTrain # average of loss across batches in each epoch
            loss_eachEpoch_semBatches = np.std(loss_eachEpoch_allBatches, axis=0) / np.sqrt(n_batches_per_epoch) # semErrorTrain

            # plot loss vs. epoch
            if 0:
                plot_loss_sem(loss, val_loss, loss_eachEpoch_aveBatches, loss_eachEpoch_semBatches, ep, err_mx, val_err_mx, plotNorm=0)
        #    plot_loss(loss, val_loss, ep, err_mx, val_err_mx, plotNorm=0)

            # find the best model across epochs, i.e. the one with minimum val_loss; set loss and val_loss at that epoch.
            [ind_bestModel, loss_bestModel, val_loss_bestModel] = set_bestModel_loss(loss, val_loss, err_mx, val_err_mx)
        #    val_loss_bestModel_norm = val_loss_bestModel / val_err_mx_all
        #    loss_bestModel_norm = loss_bestModel / err_mx_all # this should be the same as loss_bestModel if the input was z scored. (because err_mx_all will be 1)


            ##### evaluate model (note: the best model is being used here)
            # (note: for training data, model.evaulate is the same as the manual method; but,
            # both methods are different from model.history loss values. (there is nodifference for val_loss though); 
            # as per the forum below this is because model.history loss value is affected by dropout and batchNorm layers) : 
            # https://github.com/keras-team/keras/issues/6977
            x_true = x_train
            y_true = y_train

            loss_bestModel_eval = autoencoder.evaluate(x_true, y_true)[0] # , batch_size=batch_size # x_true.shape[0] # Note: the first metric is loss
            print('bestModel loss_evaluate: ', loss_bestModel_eval)
            print('Norm bestModel loss_evaluate: ', loss_bestModel_eval / err_mx)

        #    autoencoder.metrics_names
            '''
            y_pred = autoencoder.predict(x_true)    
            loss_bestModel_man =  mean_sqr_err(y_true, y_pred)
            print(np.mean(loss_bestModel_man))

            print(loss_bestModel)
        #    print(val_loss_bestModel)
            '''


            ###### compute r_squared (fraction explained variance), and y_prediction given the model
            #x_true = xx_alltrace[:,np.newaxis,:,:]
            #y_true = yy_alltrace

            [r_sq_train, y_pred_train] = r2_yPred(x_train, y_train, autoencoder, loss_provided=[0]) # loss_provided=[1, loss_bestModel_eval]
            [r_sq_test, y_pred_test] = r2_yPred(x_test, y_test, autoencoder, loss_provided=[0]) # loss_provided=[1, val_loss_bestModel]


            ###### compute r_squared and y_prediction for the entire session trace
            [xx_alltrace, yy_alltrace] = set_xx_yy_alltrace(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, plotPlots=0) # NOTE: xx_alltrace includes all x neurons, however, depending on what neuron_y it is predicting, it will be # differet. Because, it is z scored based on the mean and sd of the following trace, which varies across neuron_ys:# traces_x0[:,inds_final_all[neuron_y]]

            x_true = xx_alltrace[:,np.newaxis,:,:]
            y_true = yy_alltrace
            [r_sq_alltrace, y_pred_alltrace] = r2_yPred(x_true, y_true, autoencoder, loss_provided=[0])


            ###### use mann-whitney u test and 1-sample ttest for each neuron to see if r2 of testing data is different from when the best model of that neuron predicted random vectors
            _, pmw = sci.stats.mannwhitneyu(r_sq_rand_eachN, [r_sq_test])
            _, pt1 = sci.stats.ttest_1samp(r_sq_rand_eachN, [r_sq_test])


            ####### keep values for all neuron_ys
            loss_bestModel_all[neuron_y] = loss_bestModel_eval 
            val_loss_bestModel_all[neuron_y] = val_loss_bestModel 

            err_mx_all[neuron_y] = err_mx     
            val_err_mx_all[neuron_y] = val_err_mx     

            r_sq_train_all[neuron_y] = r_sq_train 
            r_sq_test_all[neuron_y] = r_sq_test 
            r_sq_alltrace_all[neuron_y] = r_sq_alltrace             
            r_sq_rand_all[neuron_y] = r_sq_rand_eachN 

            pmw_rand_test[neuron_y] = pmw 
            pt1_rand_test[neuron_y] = pt1 

            y_pred_train_all[neuron_y] = y_pred_train             
            y_pred_test_all[neuron_y] = y_pred_test     
            y_pred_alltrace_all[neuron_y] = y_pred_alltrace 

            gc.collect()                   
            K.clear_session()
            # delete vars
        #    del autoencoder, model_data
        #    sys.getsizeof(model_data)
        #    import psutil
        #    psutil.Process().memory_info().rss / 2**20    


#         except Exception as e:
#             print(e)
#             print('This neuron must have had very few training samples, hence lack of a saved model!')
        
    '''    
    loss_bestModel_all = np.array(loss_bestModel_all)
    val_loss_bestModel_all = np.array(val_loss_bestModel_all)

    err_mx_all = np.array(err_mx_all)
    val_err_mx_all = np.array(val_err_mx_all)

    r_sq_train_all = np.array(r_sq_train_all)
    r_sq_test_all = np.array(r_sq_test_all)
    r_sq_alltrace_all = np.array(r_sq_alltrace_all)
    r_sq_rand_all = np.array(r_sq_rand_all) # num_neurons x 50 
    
    y_pred_train_all = np.array(y_pred_train_all)
    y_pred_test_all = np.array(y_pred_test_all)    
    y_pred_alltrace_all = np.array(y_pred_alltrace_all)
    
    pmw_rand_test = np.array(pmw_rand_test)
    pt1_rand_test = np.array(pt1_rand_test)
    '''

    #%% Save variables for all y neurons as .npz file

    name = '%s_vars' %this_fold
    fname = os.path.join(dir_server_me, analysis_name , name + '.npz') 
    print(fname)

    np.savez(fname, loss_bestModel_all = loss_bestModel_all, val_loss_bestModel_all = val_loss_bestModel_all, 
             r_sq_train_all = r_sq_train_all, r_sq_test_all = r_sq_test_all, r_sq_alltrace_all = r_sq_alltrace_all, r_sq_rand_all = r_sq_rand_all, 
             y_pred_train_all = y_pred_train_all, y_pred_test_all = y_pred_test_all, y_pred_alltrace_all = y_pred_alltrace_all,
             pmw_rand_test = pmw_rand_test, pt1_rand_test = pt1_rand_test, 
             err_mx_all = err_mx_all, val_err_mx_all = val_err_mx_all)




    #%% Run the following to get plots:
    # Documents/analysis_codes/multiscope_fn/keras_DNN_planes_model_post_plots