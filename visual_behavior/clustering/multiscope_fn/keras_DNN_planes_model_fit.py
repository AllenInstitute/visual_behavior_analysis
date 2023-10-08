# This function gets called in keras_DNN_planes_model_setVars

from def_funs import *

def keras_DNN_planes_model_fit(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, kfold, x_ind, y_ind, th_train, th_test, plotPlots=0):

    #%% Set traing and testing datasets (x and y) for the model
    # extract the same part of the traces_y0_evs for a given y neuron from traces_x

    set_train_test_inds = 1 # set to a list: [train_inds, test_inds] if training and testing inds are already set (do this after the model data are saved and loaded)
    [x_train, x_test, y_train, y_test, train_data_inds, test_data_inds] = set_x_y_train_test(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, kfold, set_train_test_inds, plotPlots=0)


    #%% If doing within-plane prediction, remove neuron_y from x_train and x_test
    if x_ind==y_ind: 
        print('Within-plane prediction, hence, removing neuron_y from x population:')
        no_ny0 = np.arange(0, x_train.shape[2])
        no_ny = no_ny0[~np.in1d(no_ny0, neuron_y)]

        x_train = x_train[:,:,no_ny,:]
        x_test = x_test[:,:,no_ny,:]                
        print(x_train.shape, x_test.shape)


    #%% Only run the model if there is enough training and testing samples

    if len(train_data_inds)<th_train or len(test_data_inds)<th_test:

        print('Too few samples to run the model; %d training; %d testing' %(len(train_data_inds), len(test_data_inds)))
        tooFewSamps_x_y_neur.append([x_ind, y_ind, neuron_y])
        model_ae = np.nan
        
    else:                
        #%% Set max error: when there is no prediction, so max error will be sum(y^2).

        err_mx = set_max_error(y_train)
        val_err_mx = set_max_error(y_test)

        print('\nmax error, training, testing data:')
        print(err_mx, val_err_mx)


        #%%

        batch_size = max(80, int(x_train.shape[0] / 200)) #500 #50 #20 # # int(np.shape(x_train)[0] / 10) # 500 #1000 # 256
        print('\nbatch_size: %d, training samples: %d, iterations ~%.0f' %(batch_size, x_train.shape[0], x_train.shape[0]/batch_size))

        epochs = 5000 # 500 #50 # 

        shuffle = True # whether to shuffle the training data before each epoch
        activation_fun = 'elu' # 'relu' # 'relu'
        out_activation_fun = 'linear'
        opt = 'SGD' #'ADAM' #'adadelta' #'RMSprop' #                        
#                lr = .001
#                lr = tf.keras.experimental.CosineDecay(.001, n_gradient_steps)
#                opt = tf.optimizers.RMSprop(learning_rate=lr) # 0.001 # , rho=0.9        
#                opt = tf.optimizers.Adam(learning_rate=lr)


        #%% Deep autoencoder

    #    x_nfeatures = x_train.shape[1]
        x_nfeatures = x_train.shape[2] * x_train.shape[3]
        y_nfeatures = y_train.shape[1]
        print('Number of features for x and y: ', x_nfeatures, y_nfeatures)

    #    layer1_dim = int(np.floor((x_nfeatures - 10) / 10.)*10) # x_nfeatures -10 : int version of it.
    #    print('Layer1 dimension: ', layer1_dim)

        n_gradient_steps = int(np.ceil(x_train.shape[0] / batch_size) * epochs)
        print('Number of gradient steps: %d' %(n_gradient_steps))


        #%%

        # for latent_size in latent_size_all[-1]:
        for cval in [.15]: #cvect:
        # for batch_size in batch_size_all:

    #        print('batch size: %d' %batch_size)
            # print('latent size: %d' %latent_size)

            #%%
            '''
            # cval1 = 0 #1e-5
            # cval2 = 0 #1e-2 #1e-3
            cval = 1e-2 # cvect[ix]

            input_data = Input(shape = (x_nfeatures,))

            encoded = Dense(10, activation=activation_fun , activity_regularizer=regularizers.l2(cval))(input_data) # , activity_regularizer=regularizers.l1_l2(cval1, cval2)  # 10e-8
            decoded = Dense(50, activation=activation_fun)(encoded)
            decoded = Dense(100, activation=activation_fun)(decoded)

            decoded = Dense(y_nfeatures, activation=out_activation_fun)(decoded)

            autoencoder = Model(input_data, decoded)
            autoencoder.summary()
            '''

            #%%

    #        , activity_regularizer=regularizers.l2(cval)
    #        cval = .3 #.1 #.1 #.001 #.1    
    #        latent_dim = int((x_train0.shape[1] * x_train0.shape[2]/2 - y_nfeatures) / 2.)#500 #1000
    #        print(latent_dim)

            autoencoder = Sequential()

    #        autoencoder.add(Dense(y_train.shape[1], activation=out_activation_fun, input_shape=(x_train.shape[1],)))

    #        autoencoder.add(Dropout(cval, input_shape=(x_train.shape[1],)))

            autoencoder.add(Dropout(.5, input_shape=(1, x_train.shape[2], x_train.shape[3])))

            autoencoder.add(Dense(int(len_win/2), activation=activation_fun))
    #        autoencoder.add(Dense(5, activation=activation_fun, input_shape=(1, x_train0.shape[1], x_train0.shape[2])))

    #        autoencoder.add(BatchNormalization())
            autoencoder.add(Flatten())
    #        autoencoder.add(Dropout(cval))
            autoencoder.add(BatchNormalization())        


            latent_dim = int(autoencoder.layers[-1].output_shape[-1]/2)
#                    print(latent_dim)
            autoencoder.add(Dense(latent_dim, activation=activation_fun))
            autoencoder.add(Dropout(.5))
            autoencoder.add(BatchNormalization())


            latent_dim = int(autoencoder.layers[-1].output_shape[-1]/2)
#                    print(latent_dim)
            autoencoder.add(Dense(latent_dim, activation=activation_fun))                
            autoencoder.add(Dropout(.5))
            autoencoder.add(BatchNormalization())

            '''
            latent_dim = int(autoencoder.layers[-1].output_shape[-1]/2.5)
            print(latent_dim)
            autoencoder.add(Dense(latent_dim, activation=activation_fun))
            autoencoder.add(Dropout(cval))
            autoencoder.add(BatchNormalization())


            latent_dim = int(autoencoder.layers[-1].output_shape[-1]/3)
            print(latent_dim)
            autoencoder.add(Dense(latent_dim, activation=activation_fun))
            autoencoder.add(Dropout(cval))
            autoencoder.add(BatchNormalization())


            latent_dim = int(autoencoder.layers[-1].output_shape[-1]/3.5)
            print(latent_dim)
            autoencoder.add(Dense(latent_dim, activation=activation_fun))
            autoencoder.add(Dropout(cval))
            autoencoder.add(BatchNormalization())

            '''        
    #        latent_dim = int(autoencoder.layers[-1].output_shape[-1]/4)
    #        print(latent_dim)
    #        autoencoder.add(Dense(latent_dim, activation=activation_fun))
    #        autoencoder.add(Dropout(cval))
    #        autoencoder.add(BatchNormalization())


            autoencoder.add(Dense(y_train.shape[1], activation=out_activation_fun))

            autoencoder.summary()


            #%% Set x and y for cnn
            '''
            a = np.reshape(x_train, (x_train.shape[0], 10, 39), order='F')
            aa = np.reshape(x_test, (x_test.shape[0], 10, 39), order='F')
            b = np.reshape(y_train, (x_train.shape[0], 10, 39), order='F')
            bb = np.reshape(y_test, (x_test.shape[0], 10, 39), order='F')

            x_train = a #np.reshape(x_trainf, (x_trainf.shape[0], x_trainf.shape[1], 1)) # (batch, steps, channels)
            y_train = b #np.reshape(y_trainf, (y_trainf.shape[0], y_trainf.shape[1], 1))

            x_test = aa #np.reshape(x_testf, (x_testf.shape[0], x_testf.shape[1], 1))
            y_test = bb #np.reshape(y_testf, (y_testf.shape[0], y_testf.shape[1], 1))

            x_train.shape


            #%% Convolutional neural net        

            cval = 0 #1e-3 #.1        

            autoencoder = Sequential()

    #        autoencoder.add(Dropout(cval, input_shape=(x_nfeatures,1)))
    #        autoencoder.add(Conv1D(16, 3, padding='same', activation=activation_fun))
            autoencoder.add(Conv1D(16, 3, padding='same', activation=activation_fun, input_shape=(x_nfeatures, 1))) # (sequence_length, features)
    #        autoencoder.add(Conv1D(16, 3, padding='same', activation=activation_fun, input_shape=(10, 39), activity_regularizer=regularizers.l2(cval)))
            autoencoder.add(MaxPooling1D(2, padding='same'))
            autoencoder.add(Conv1D(8, 3, padding='same', activation=activation_fun))
            autoencoder.add(MaxPooling1D(2, padding='same'))
    #        autoencoder.add(Conv1D(8, 3, padding='same', activation=activation_fun))

            autoencoder.add(Conv1D(8, 3, padding='same', activation=activation_fun))
            autoencoder.add(UpSampling1D(2))
            autoencoder.add(Conv1D(16, 3, padding='same', activation=activation_fun))
            autoencoder.add(UpSampling1D(2))
            autoencoder.add(Conv1D(1, 3, activation=out_activation_fun))
    #        autoencoder.add(Conv1D(39, 3, activation=out_activation_fun)) # padding='same', 

            autoencoder.summary()
            '''

            #%% Change the number of hidden layers
            """
            depth_size_ind_all = np.arange(1, int(np.ceil((layer1_dim-latent_size)/10.))+2) # the last element is used to set a network that directly jumps to the hiddlen layer with size laten_size (with no preceding layers).

            loss_all1 = []
            val_loss_all1 = []
            hidden_layers_all1 = []

            for depth_size_ind in depth_size_ind_all: # depth_size_ind = 1

                hidden_layer_dims = np.arange(layer1_dim, latent_size, max(latent_size - layer1_dim, -10*depth_size_ind)) # decrease dims by 10 at every layer.
                if depth_size_ind==depth_size_ind_all[-1]: # the last round will be a network with only the middle layer.
                    hidden_layer_dims = hidden_layer_dims[1:]
                hidden_layers = np.concatenate((hidden_layer_dims, [latent_size]))
                # print(hidden_layers)

                # input layer
                input_data = Input(shape = (x_nfeatures,)) # encoded = Dense(layer1_dim, activation=activation_fun)(input_data)''
                encoded = input_data

                # all encoding hidden layers
                for hdl in hidden_layer_dims:            
                    encoded = Dense(hdl, activation=activation_fun)(encoded)

                # middle layer
                encoded = Dense(latent_size, activation=activation_fun)(encoded) 
                decoded = encoded

                # all decoding hidden layers
                for hdl in hidden_layer_dims[::-1]:            
                    decoded = Dense(hdl, activation=activation_fun)(decoded)

                # last decoding layer
                decoded = Dense(y_nfeatures, activation=out_activation_fun)(decoded) # decoded = Dense(layer1_dim, activation=activation_fun)(decoded)

                autoencoder = Model(input_data, decoded)

                autoencoder.summary()


                #%%
                # input_data = Input(shape = (x_nfeatures,))
                '''
                # encoded = Dense(500, activation=activation_fun, activity_regularizer=regularizers.l1(10e-8))(input_data) # 128 # Dropout(0.2 # kernel_constraint=maxnorm(3) # , activity_regularizer=regularizers.l1(10e-5) 
                # encoded = Dense(500, activation=activation_fun)(input_data) # 128 # Dropout(0.2 # kernel_constraint=maxnorm(3) # , activity_regularizer=regularizers.l1(10e-5) 
                # encoded = Dense(100, activation=activation_fun)(encoded)

                # encoded1 = Dense(100, activation=activation_fun)(input_data)
                encoded1.trainable = False
                # encoded2 = Dense(50, activation=activation_fun)(encoded1)
                encoded2.trainable = False
                # encoded_stop_grad = Lambda(lambda x: K.stop_gradient(x))(encoded)
                decoded1 = Dense(100, activation=activation_fun)(encoded2)#_stop_grad)

                #encoded = Dense(5, activation=activation_fun)(encoded) # 32
                # decoded = Dense(100, activation=activation_fun)(encoded1) # this one
                # decoded = Dense(500, activation=activation_fun)(decoded)  # 128
                decoded1 = Dense(y_nfeatures, activation=out_activation_fun)(decoded1) # sigmoid

                autoencoder = Model(input_data, decoded1)
                '''
            """    

            #%% Compile the model

            # def mean_err(y_true, y_pred):
            #     return K.sum(K.abs(y_pred - y_true)) # K.abs(y_pred - y_true)

            autoencoder.compile(optimizer=opt, # 'RMSprop', #'ADAM',#'RMSprop', # 'SGD'
                                loss='mse', #'mae', 'mse' #'mean_squared_error', #'msle' # mean_err : NOTE: do not use '' when using a custom function.
                                metrics=['mae']) #, 'hinge']) # adadelta # binary_crossentropy # 'mae', 'acc' # mean_squared_logarithmic_error, 'categorical_accuracy'


            #%%
            '''        
            x_train = x_train0[:,np.newaxis,:,:]
            x_test = x_test0[:,np.newaxis,:,:]
            print(x_train.shape)

            y_train = y_train0
            y_test = y_test0
            print(y_train.shape)

    #        get_ipython().magic(u'matplotlib qt')
            plt.figure(); plt.plot(y_train)
            plt.figure(); plt.plot(y_test)

            plt.figure(); plt.plot(np.mean(x_train, axis=(1,2,3)))
            plt.figure(); plt.plot(np.mean(x_test, axis=(1,2,3)))
            '''


            #%% Train the model

            losshistory = LossHistory()
            # save keras models
    #        weights.{epoch:02d}-{val_loss:.2f}.hdf5
            aname = 'plane%dto%d_neuronY_%03d_model_ep_' %(x_ind, y_ind, neuron_y)
            dir_this = '%s{epoch:03d}.hdf5' %aname
            checkpointer = callbacks.ModelCheckpoint(filepath=os.path.join(dir_ae, dir_this), verbose=1, save_best_only=True) # ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    #        saver = CustomSaver()

            es = callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=.00001, patience=100, verbose=1) # (monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

            autoencoder.fit(x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            callbacks=[losshistory, checkpointer, es],
                            validation_data=(x_test, y_test))


            #%% Get model history, contains loss values    

            model_hist = autoencoder.history
            loss_batches = np.array(losshistory.losses)

            model_hist_params = model_hist.params
            model_hist_hist = model_hist.history


            #%% Save loss (the attribute "history" does not get saved in model.save in keras)
            # also save x and y, (train and test)
    #        traces_x0 , inds_final_all , traces_y0_evs

            cols = np.array(['model_hist_params', 'model_hist_hist', 'loss_batches', 'train_data_inds', 'test_data_inds'])
            model_ae = pd.DataFrame([], columns=cols)

            model_ae.at[0, cols] = model_hist_params, model_hist_hist, loss_batches, train_data_inds, test_data_inds

            # save data that was modeled
            name = 'neuronY_%03d_model_data' %(neuron_y) 
            aem = os.path.join(dir_now , name + '.h5') 
            print(aem)            

            model_ae.to_hdf(aem, key='model_ae', mode='w') 
    #        loss_batches.to_hdf(aem, key='loss_batches', mode='a') # save input_vars to the h5 file            


            #%% Move the best model from /tmp directory to DNN_allTrace_timeWindow_planes directory

            [modelName, h5_files] = all_sess_set_h5_fileName(aname, dir_ae, all_files=0)    
    #        autoencoder = load_model(modelName)

            shutil.move(modelName, dir_now)

    #        a = pd.read_hdf('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/DNN_allTrace_timeWindow_planes/neuronY_004_model_data.h5')

        gc.collect()
        K.clear_session()


    #%%
    
    return model_ae


#%%
########################################################################################        
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
"""
#%% Get loss values

model_hist_hist = model_ae.iloc[0,'model_hist_hist']

#### omission-evoked responses    
loss = model_hist_hist['loss']
val_loss = model_hist_hist['val_loss']

mae = model_hist_hist['mae'] # mean_squared_logarithmic_error
val_mae = model_hist_hist['val_mae'] # val_mean_squared_logarithmic_error            

ep = np.arange(0, loss.shape[0])
# ep = range(epochs) # model_hist.epoch # 
# model_hist = autoencoder.history        
# model_hist.params

\
#### flash-evoked responses    
# loss_f = model_hist_hist['loss']
# val_loss_f = model_hist_hist['val_loss']

# msle_f = model_hist_hist['msle'] # mean_squared_logarithmic_error
# val_msle_f = model_hist_hist['val_msle'] # val_mean_squared_logarithmic_error

# print(np.shape(xy_train[0]), np.shape(xy_train[1]))    
# print(np.shape(xy_test[0]), np.shape(xy_test[1]))                

print('loss: ', loss[-1], val_loss[-1])
print('norm loss: ', loss[-1] / err_mx, val_loss[-1] / val_err_mx)
print('err_mx: ', err_mx, val_err_mx)


#%% Get loss values for all batches of each epoch

[loss_eachEpoch_allBatches, n_batches_per_epoch, epochs_completed] = set_loss_allBatches(loss_batches, batch_size)

# training data: compute mean and sem of loss across batches, for each epoch        
loss_eachEpoch_aveBatches = np.mean(loss_eachEpoch_allBatches, axis=0) # meanErrorTrain # average of loss across batches in each epoch
loss_eachEpoch_semBatches = np.std(loss_eachEpoch_allBatches, axis=0) / np.sqrt(n_batches_per_epoch) # semErrorTrain
# the following two are almost the same: loss_eachEpoch_aveBatches - loss (not 100% the same because keras seems to compute loss per epoch using a running average... based on a keras github comment)
#        plt.plot(loss_eachEpoch_aveBatches - loss)

'''
xe = range(loss_eachEpoch_aveBatches.shape[0]) # epochs_completed
plt.figure()

plt.plot(xe, loss_eachEpoch_aveBatches, 'k')
plt.fill_between(xe, loss_eachEpoch_aveBatches-loss_eachEpoch_semBatches , loss_eachEpoch_aveBatches+loss_eachEpoch_semBatches, color='k', alpha=.3)
plt.plot(xe, loss, 'r')
'''

#%% Keep the values for different depth sizes for a given latent_size

loss_all.append(loss)
val_loss_all.append(val_loss)

err_mx_all.append(err_mx)
val_err_mx_all.append(val_err_mx)

autoencoder_all.append(autoencoder)

x_train_all.append(x_train)
x_test_all.append(x_test)
y_train_all.append(y_train)
y_test_all.append(y_test)

# loss_all1.append(loss)
# val_loss_all1.append(val_loss)
# hidden_layers_all1.append(hidden_layers)


#%% Plot the loss values

if 1: # doLossPlot:            
#            loss = loss_all[-1,:]
#            val_loss = val_loss_all[-1,:]
#            plot_loss(loss, val_loss, ep, err_mx, val_err_mx, plotNorm=0)
    plot_loss_sem(loss, val_loss, loss_eachEpoch_aveBatches, loss_eachEpoch_semBatches, ep, err_mx, val_err_mx, plotNorm=0)


#%% Keep values for all latent sizes

# loss_all.append(loss_all1)
# val_loss_all.append(val_loss_all1)
# hidden_layers_all.append(hidden_layers_all1)

#autoencoder_last = autoencoder
"""
