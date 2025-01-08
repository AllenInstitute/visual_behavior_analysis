#%% Make plots for the DNN analysis.
# no need to run other scripts before running this script.
# vars are saved in keras_DNN_planes_model_post_saveVars.py (fit_dnn_model = 0)
# Load saved model vars, and make plots

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

#%%

x_planes = [0,1,4,5]
y_planes = [0,1,4,5] #[1,4,5]

savefigs = 1
sigLevel = .05
fmt = 'pdf'

#%% Set vars

%load_ext autoreload
%run -i 'omissions_traces_peaks_init.py'

from keras.models import Model, Sequential, load_model
import keras.backend as K

analysis_name = 'DNN_allTrace_timeWindow_planes'
n_beg_end_rmv = 60 # number of frames to exclude from the begining and end of the session (because we usually see big rises in the trace, perhaps due to df/f computation?)
len_win = 20 #10 # length of the consecutive frames that will be included in the feature space for each neuron        
len_ne = len_win #20 # number of frames before and after each event that are taken to create traces_events.
th_ag = 10 #8 # threshold to apply on erfc (output of evaluate_components) to find events on the trace; the higher the more strict on what we call an event.
kfold = 45

analysis_name_n = 'DNN_planes_popActivity' # folder for saving figures
col_true = 'gray'
col_pred = 'c' #'g'
samples_per_plot = 1000 #10000 # each page of pdf will include this many frames


#%% Set image, omission, lick, reward, running times (these are the same for all experiments)

flash_times = this_sess.iloc[0]['list_flashes'].values 
omission_times = this_sess.iloc[0]['list_omitted'].values         

lick_times = this_sess.iloc[0]['licks']['time'].values
reward_times = this_sess.iloc[0]['rewards']['time'].values

running_times = this_sess.iloc[0]['running_speed']['time'].values
running_speed = this_sess.iloc[0]['running_speed']['running_speed'].values

                
#%% Set folder for saving figures

# x_ind = x_planes[0] 
# y_ind = y_planes[0] 

for x_ind in x_planes: #x_planes: #[1]: #np.arange(0,num_depth): # x_ind = x_planes[0]
    for y_ind in y_planes: #[x_ind] #y_planes: #[4,5,6]: #np.arange(4,8): # y_ind = y_planes[0]

        print(x_ind, y_ind)

#         if y_ind!=x_ind: # remove this once you saved vars ... now you dont have within plane data

        print('======================== Plane %d to %d; saving figures ======================== ' %(x_ind, y_ind))

        plp = 'plane%dto%d_' %(x_ind, y_ind)
        this_fold = 'isess%d_plane_%dto%d' %(isess, x_ind, y_ind) 
        dir_now = os.path.join(dir_server_me, analysis_name, this_fold)

        dir_planePair = os.path.join(dir0, analysis_name_n, this_fold)
        if not os.path.exists(dir_planePair):
            print('creating folder')
            os.makedirs(dir_planePair)
            
        list_figs = os.listdir(dir_planePair)
        nowStr = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

        get_ipython().magic(u'matplotlib inline')


        #%% Load the numpy file (.npz) that includes model vars for all neurons

        name = '%s_vars' %this_fold
        fname = os.path.join(dir_server_me, analysis_name , name + '.npz') 
        print(fname)

        nf = np.load(fname, allow_pickle=True) #list(nf.files) , sorted(nf.files)

        loss_bestModel_all = nf['loss_bestModel_all']
        val_loss_bestModel_all = nf['val_loss_bestModel_all']

        r_sq_train_all = nf['r_sq_train_all']
        r_sq_test_all = nf['r_sq_test_all']
        r_sq_alltrace_all = nf['r_sq_alltrace_all'] 
        r_sq_rand_all = nf['r_sq_rand_all'] # num_neurons x 50 

        y_pred_train_all = nf['y_pred_train_all']
        y_pred_test_all = nf['y_pred_test_all']    
        y_pred_alltrace_all = nf['y_pred_alltrace_all']

        pmw_rand_test = nf['pmw_rand_test']
        pt1_rand_test = nf['pt1_rand_test']

        err_mx_all = nf['err_mx_all']
        val_err_mx_all = nf['val_err_mx_all']

        print('\nNumber of neurons in y trace:', len(val_err_mx_all))

        
        # set NaN neurons (neurons without a model because they didnt have enough training/ testing datapoints)        
        nan_neurons = np.isnan(r_sq_test_all)
        

        #%% Set traces_x0 and traces_y0; these are traces read from this_sess_l.iloc[ind]['local_fluo_traces']; after removing n_beg_end_rmv frames from their begining and end.

        [traces_x0, traces_y0] = set_traces_x0_y0(this_sess, x_ind, y_ind, n_beg_end_rmv, plotPlots=0) 

        
        #%% Set traces_events (active parts of Y traces are found. Y traces are are made by concatenating the active parts. X traces are made from those same time points (during which Y is active).)

        [traces_y0_evs, inds_final_all] = set_traces_evs(traces_y0, th_ag, len_ne, doPlots=0)
        
        
        #%% Sort neurons based on their testing r2

        get_ipython().magic(u'matplotlib inline')
        
        a = r_sq_rand_all #[~nan_neurons]
        b = r_sq_test_all #[~nan_neurons]
        c = r_sq_train_all #[~nan_neurons]
        d = r_sq_alltrace_all #[~nan_neurons]        
                
        r2_sorted_all = np.argsort(b) # last element will be index of neuron with nan r2
        r2_sorted_vals = np.sort(b)        
        r2_sorted = r2_sorted_all[~np.isnan(r2_sorted_vals)] # exclude nans
        rank_neurons = np.array([len(r2_sorted) - np.argwhere(np.in1d(r2_sorted, neuron_y)).squeeze() for neuron_y in np.arange(0, len(r2_sorted_all))]) # rank of the neurons in the sorted array (high to low values); best neuron is rank 1. 
        
        plt.figure()
        plt.hlines(0, 0, len(b))
        plt.plot(r2_sorted_vals, '.')

        
        
        ####################################################################
        ####################################################################
        ####################################################################
        
        #%% Compute fraction of neurons with significant variance explained
        # mann-whitney u test or 1-sample ttest used for each neuron to see if r2 of testing data is different from when the best model of that neuron predicted random vectors

        '''
        # print some usefull quantities
        # average r2 for neurons with positive r2
        np.mean(r_sq_test_all[r_sq_test_all>0])
        # fraction of neurons with positive r2 that are not significant
        np.mean(np.logical_and(r_sq_test_all>0 , pmw_rand_test > sigLevel))
        # fraction of neurons with positive r2 that are also significant (from random)
        np.mean(np.logical_and(r_sq_test_all>0 , pmw_rand_test <= sigLevel))
        '''
        
        pos_r_sq_test_fractNs = np.mean(r_sq_test_all>0)        
        sigFract_mw = np.mean(pmw_rand_test <= sigLevel)
        sigFract_t1 = np.mean(pt1_rand_test <= sigLevel)

        plt.figure()
        plt.plot(pmw_rand_test, 'k', label='Mann-Whitney')
        plt.plot(pt1_rand_test, 'b', label='ttest 1samp')
        plt.xlabel('Neurons')
        plt.ylabel('p value of variance explained\n(actual vs. random)')
        plt.legend(loc='center left', bbox_to_anchor=[1,.7], frameon=False, handlelength=1, fontsize=12)

        plt.title('\nFraction of neurons with positive explained variance:%.2f\nsignificant explained variance:Mann-Whitney u test:%.2f\n1sample ttest:%.2f' %(pos_r_sq_test_fractNs, sigFract_mw, sigFract_t1))

        if savefigs:
            aname = f"r2_sig_fractNeurons_{plp}"
            save_fig_if_notexist(aname, dir_planePair, nowStr, fmt='pdf')
            
                
        #%% Comute ROC area, comparing r2 distributions of testing and random-control data

        a = r_sq_rand_all[~nan_neurons].flatten()
        b = r_sq_test_all[~nan_neurons]
        #c = r_sq_train_all

        # We assign 0 to test and 1 to train, so:
        # test > train means AUC < 0.5
        # test < train means AUC > 0.5
        y_true = np.concatenate(([np.zeros(np.shape(a)), np.ones(np.shape(b))]), axis = 0)
        y_score = np.concatenate(([a, b]), axis = 0)

        fpr, tpr, thresholds = roc_curve(y_true, y_score)    
        auc_r2 = auc(fpr, tpr)

        print('AUC, random vs. testing explained variance:')
        print(auc_r2)


        #%% Plot loss and explained variance for all neurons

        get_ipython().magic('matplotlib inline')

        ### loss values
        plot_loss(loss_bestModel_all, val_loss_bestModel_all, np.nan, err_mx_all, val_err_mx_all, plotNorm=0, xlab='y neuron')


        ### normalized loss values
        plot_loss(loss_bestModel_all, val_loss_bestModel_all, np.nan, err_mx_all, val_err_mx_all, plotNorm=1, xlab='y neuron')


        ### explained variance
        h, lab = plot_loss(r_sq_train_all, r_sq_test_all, np.nan, 1, 1, plotNorm=0, xlab='y neuron', ylab='explained variance')

        h2 = plt.plot(np.mean(r_sq_rand_all, axis=1), 'gray', label='random')
        plt.plot(r_sq_alltrace_all, 'm', label='allTrace')        
        plt.legend(loc='center left', bbox_to_anchor=[1,.7], frameon=False, handlelength=1, fontsize=12)

        # mark sig neurons
        a = max(r_sq_train_all) * (pmw_rand_test <= sigLevel).astype(int)
        a[a==0] = np.nan
        plt.plot(a, 'r*', markersize=5)
        plt.hlines(0, 0, len(val_err_mx_all), 'r')
        
        makeNicePlots(plt.gca())
#         plt.show()

        if savefigs:
            aname = f"r2_eachN_randTestTrain_{plp}"
            save_fig_if_notexist(aname, dir_planePair, nowStr, fmt='pdf')
            
        # h.append(h2)
        # lab.append(lab2)
        # plt.gca().legend(handles=h, labels=lab, loc='center left', bbox_to_anchor=[1,.7], frameon=False, handlelength=1, fontsize=12)

        #plot_loss(np.mean(r_sq_rand_all, axis=1), r_sq_test_all, np.nan, 1, 1, plotNorm=0, xlab='y neuron', ylab='explained variance')

        '''
        plt.figure()
        plt.subplot(211)
        plt.plot(loss_bestModel_all)
        plt.plot(val_loss_bestModel_all)

        plt.subplot(212)
        plt.plot(loss_bestModel_all / err_mx_all)
        plt.plot(val_loss_bestModel_all / val_err_mx_all)

        plt.subplots_adjust(hspace=.8)
        '''


        #%% Plot histogram of control (random), testing and training data

        a = r_sq_rand_all[~nan_neurons].flatten()
        b = r_sq_test_all[~nan_neurons]
        c = r_sq_train_all[~nan_neurons]
        d = r_sq_alltrace_all[~nan_neurons]         

        topall = a,b,c,d

        labs = 'rand', 'test', 'train', 'allTrace'
        colors = 'gray', 'b', 'k', 'm' #'c', 'k','b'   
        linestyles = 'solid', 'solid', 'solid', 'solid' #(0, (4,4)) #(0, (4,3.5)) 'dashed', 'solid', 'dashed' # . For example, (0, (3, 10, 1, 15)) means (3pt line, 10pt space, 1pt line, 15pt space) with no offset.
        xlab = 'Explained variance'
        ylab ='Fraction neurons'
        tit_txt = 'posFractNs %.2f, sigFractNs %.2f, aucRandTest %.2f' %(pos_r_sq_test_fractNs, sigFract_mw, auc_r2)
        doSmooth = 0 #3
        nbins = 50

        plothist(topall, nbins, doSmooth, colors, linestyles, labs, xlab, ylab, tit_txt)
        #plt.xlim([-.75, .8])
        yl = plt.gca().get_ylim()

        plt.vlines(0, yl[0], yl[1], 'r')

        if savefigs:
            aname = f"r2_hist_randTestTrain_{plp}"
            save_fig_if_notexist(aname, dir_planePair, nowStr, fmt='pdf')


        ####################################################################################
        ############################ representative neuron plots ############################
        ####################################################################################

        #%% Pick a representative neuron to make plots for

        rank2p = 1 # what ranking of neuron (in terms of r2 value) to pick. 1: the best; 2: the second best, etc neuron
        ind_best_neuron = r2_sorted[-rank2p]

        print(ind_best_neuron)
        print(r_sq_test_all[ind_best_neuron], r_sq_train_all[ind_best_neuron], d[ind_best_neuron], r_sq_rand_all[ind_best_neuron].mean())
        neuron_y = ind_best_neuron


        #%% Set traces_x0_evs and traces_y0_evs_thisNeuron
        # final (z scored) input matrices (traces_evs for x population activity and y neuron, ie traces_x0_evs and traces_y0_evs_thisNeuron) to traces_neurons_times_features, which sets xx0 and yy0.    

        [traces_x0_evs, traces_y0_evs_thisNeuron, scaler_x, scaler_y] = set_traces_x0_y0_evs_final(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, plotPlots=0, setcorrs=0)

        
        #%% Plot traces_x0 (df/f traces of all neurons in plane x_ind)

        # add a constant to each neuron, so we can see all the traces along the y axis
        traces_x0_plt = traces_x0 + 2 * np.ones(traces_x0.shape) * np.arange(0, traces_x0.shape[0])[:,np.newaxis]

        plt.figure(figsize=(int(traces_x0.shape[1]*2e-4), traces_x0.shape[0]))
        plt.plot(traces_x0_plt.T);
        plt.xlabel('Frames (10Hz)', fontsize=12)
        plt.title('traces_x0')
        makeNicePlots(plt.gca())

        if savefigs:
            aname = f"traces_x0_{plp}"
            save_fig_if_notexist(aname, dir_planePair, nowStr, fmt='pdf')


        #%% Plot traces_y0 for neuron_y

        plt.figure(figsize=(int(traces_x0.shape[1]*2e-4), 1))
        plt.plot(traces_y0[neuron_y])
        plt.xlabel('Frames (10Hz)', fontsize=12)
        plt.title('traces_y0, neuron %d' %ind_best_neuron)
        makeNicePlots(plt.gca())

        if savefigs:
            aname = f"traces_y0_neuronY_{neuron_y}_{plp}" # 'traces_y0_neuronY_%d_' %neuron_y +plp+nowStr+'.'+'pdf'
            save_fig_if_notexist(aname, dir_planePair, nowStr, fmt='pdf')


        #%% Plot traces_x0_evs 

        traces_x0_plt = traces_x0_evs + 10 * np.ones(traces_x0_evs.shape) * np.arange(0, traces_x0_evs.shape[0])[:,np.newaxis]

        plt.figure(figsize=(int(traces_x0.shape[1]*2e-4), traces_x0.shape[0]))
        plt.plot(traces_x0_plt.T);
        plt.xlabel('Frames (10Hz)', fontsize=12)
        plt.title('traces_x0_evs')
        makeNicePlots(plt.gca())

        if savefigs:
            aname = f"traces_x0_evs_neuronY_{neuron_y}_{plp}"
            save_fig_if_notexist(aname, dir_planePair, nowStr, fmt='pdf')


        #%% Plot traces_y0_evs for neuron_y

        plt.figure(figsize=(int(traces_x0.shape[1]*2e-4), 1))
        plt.plot(traces_y0_evs[neuron_y])
        plt.xlabel('Frames (10Hz)', fontsize=12)
        plt.title('traces_y0_evs, neuron %d' %ind_best_neuron)
        makeNicePlots(plt.gca())

        if savefigs:
            aname = f"traces_y0_evs_neuronY_{neuron_y}_{plp}"
            save_fig_if_notexist(aname, dir_planePair, nowStr, fmt='pdf')

        
        #%%
        ####################################################################################
        ############################ Plot true and predicted traces of all neurons ############################
        ####################################################################################

#         get_ipython().magic(u'matplotlib qt')

        for neuron_y in np.arange(0, traces_y0.shape[0]): # [ind_best_neuron]: # neuron_y=ind_best_neuron

            print('\nn======================== Plane %d to %d; saving figures for neuron %d/%d in y trace ======================== ' %(x_ind, y_ind, neuron_y, traces_y0.shape[0]))
            
            if np.isnan(r_sq_train_all[neuron_y]):
                print(f'skipping neuron {neuron_y} because it is nan!')
            else:
                # NOTE: xx_alltrace includes all x neurons, however, depending on what neuron_y it is predicting, it will be 
                # differet. Because, it is z scored based on the mean and sd of the following trace, which varies across neuron_ys:
                # traces_x0[:,inds_final_all[neuron_y]]            
                [xx_alltrace, yy_alltrace] = set_xx_yy_alltrace(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, plotPlots=0)            
            #    [xx_alltrace, yy_alltrace] = set_x_y_train_test(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, kfold, set_train_test_inds=0, plotPlots=0, set_xx_alltrace=1)
                # xx_alltrace.shape[0] == traces_x0.shape[1] - len_win
                # t_y = len_win/2
                # yy_alltrace = z_scored_traces_y0[t_y: le-t_y]
                # xx_alltrace, yy_alltrace are made from z scored inputs; so yy_alltrace is z scored traces_y0


                ########### load the best model for neuron_y ###########        
                '''
                aname = 'plane%dto%d_neuronY_%03d_model_ep_' %(x_ind, y_ind, neuron_y)
                [modelName, h5_files] = all_sess_set_h5_fileName(aname, dir_now, all_files=0)    
                autoencoder = load_model(modelName)
                '''


                ########### load training and testing indeces for neuron_y ###########        
                dname = 'neuronY_%03d_model_data' %neuron_y
                [dataName, h5_files] = all_sess_set_h5_fileName(dname, dir_now, all_files=0)            
                model_data = pd.read_hdf(dataName, key='model_ae')
                train_data_inds = model_data.iloc[0]['train_data_inds']
                test_data_inds = model_data.iloc[0]['test_data_inds']


                ########### set training and testing data for ind_best_neuron ###########        
                set_train_test_inds = [train_data_inds, test_data_inds]
                [x_train, x_test, y_train, y_test, _, _] = set_x_y_train_test(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, kfold, set_train_test_inds, plotPlots=0)


                ########### find the corresponding train_data_inds on trace yy_alltrace ############    
                train_data_inds_on_yy_alltrace, train_data_inds_on_traces_y0 = map_inds(train_data_inds, len_win, inds_final_all, neuron_y)
                test_data_inds_on_yy_alltrace, test_data_inds_on_traces_y0 = map_inds(test_data_inds, len_win, inds_final_all, neuron_y)

                # double check this ... see if train_data_inds_on_traces_y0 applied on traces_y0 are indeed like y_train        
                '''
                plt.figure()
                plt.plot(y_train, 'b')    
                plt.plot(yy_alltrace[train_data_inds_on_yy_alltrace], 'r')
                '''
            #    plt.plot(traces_y0[neuron_y, train_data_inds_on_traces_y0], 'r')

                                
                #%% Set the time trace for yy_alltrace        

                traces_y0_time = this_sess.iloc[y_ind]['local_time_traces'][n_beg_end_rmv:-n_beg_end_rmv] # same size as traces_y0
                t_y = int(len_win/2)
                le = traces_y0_time.shape[0]
                yy_alltrace_time = traces_y0_time[t_y: le-t_y] # same size as yy_alltrace
#                 traces_y0 = traces_y[:, n_beg_end_rmv:-n_beg_end_rmv]
#                 yy_alltrace = z_scored_traces_y0[t_y: le-t_y]


                #%% Find flash, omission, lick and reward times on yy_alltrace_time (basically convert time to frame)
    
                flashes_win_yyalltrace_index = convert_time_to_frame(yy_alltrace_time, flash_times) # index of flash times on yy_alltrace_time
                omissions_win_yyalltrace_index = convert_time_to_frame(yy_alltrace_time, omission_times)
                licks_win_yyalltrace_index = convert_time_to_frame(yy_alltrace_time, lick_times)
                rewards_win_yyalltrace_index = convert_time_to_frame(yy_alltrace_time, reward_times)                

                
                #%% Take care of running trace: (running this takes a bit time)
                # align its timing on yy_alltrace_time
                # turn it from stimulus-computer time resolution to imaging-computer time resolution, so it can be plotted on the same timescale as imaging.
                running_speed_duringImaging_downsamp = running_align_on_imaging(running_times, running_speed, yy_alltrace_time)
                
                plt.figure()
#                 plt.plot(running_time, running_speed)                
#                 plt.plot(running_times_duringImaging, running_speed_duringImaging)
#                 plt.plot(yy_alltrace_time[running_times_win_yyalltrace_index], running_speed_duringImaging)
                plt.plot(yy_alltrace_time, running_speed_duringImaging_downsamp)                
                # df/f trace 
                plt.plot(yy_alltrace_time, yy_alltrace)




                ###########################################
                ################### Plots #################
                ###########################################
                
                ########################################################################
                ############ plot yy_alltrace and compare it with the prediction (the entire trace of the session) ############
                '''
                x_true = xx_alltrace[:,np.newaxis,:,:]
                y_true = yy_alltrace
                [r_sq_alltrace, y_pred_alltrace] = r2_yPred(x_true, y_true, autoencoder, loss_provided=[0])
    #                 print('explained variance for yy_alltrace: ', r_sq_alltrace)
                '''

                y_pred_alltrace = y_pred_alltrace_all[neuron_y]
                r_sq_alltrace = r_sq_alltrace_all[neuron_y]

                aname = f"trace_pred_allTrace_neuronY_rank{rank_neurons[neuron_y]}_ind{neuron_y}_{plp}"
                fign = os.path.join(dir_planePair, f"{aname}{nowStr}.pdf")            
                titl = 'neuron %d, entire session, expained_var %.2f' %(neuron_y, r_sq_alltrace)
                y_min = min(yy_alltrace)
                y_max = max(yy_alltrace)
                r_y = (y_max - y_min)
                lent = yy_alltrace.shape[0]
                
                
                # scale running to fit it on the figure
                runmx = np.max(running_speed_duringImaging_downsamp)
#                 runmn = np.min(running_speed_duringImaging_downsamp)
#                 run_bl = np.min(running_speed_duringImaging_downsamp)
                run_scale = (running_speed_duringImaging_downsamp / runmx) + (y_min + r_y/1.5)
#                 plt.plot(runn)
                
                pdf = matplotlib.backends.backend_pdf.PdfPages(fign)
                for i in range(int(lent / samples_per_plot) + 1): # i=5
                    r0 = i * samples_per_plot
                    r1 = (i + 1) * samples_per_plot

                    y_true = yy_alltrace[r0:r1]
                    y_pred = y_pred_alltrace[r0:r1]
                    
                    ymn = min(y_true)
                    ymx = max(y_true)
                    ry = (ymx - ymn)/10
                    
                    # set training and testing indeces for this chunk of data
                    tri = train_data_inds_on_yy_alltrace[np.logical_and(train_data_inds_on_yy_alltrace >= r0, train_data_inds_on_yy_alltrace < r1)] - r0
                    tsi = test_data_inds_on_yy_alltrace[np.logical_and(test_data_inds_on_yy_alltrace >= r0, test_data_inds_on_yy_alltrace < r1)] - r0
                    
                    # set flash-onset indeces for this chunk of data (note: if a flash starts at the end of a chunk, then it will continue to the next chunck (page) but you wont show it on the next chunk because you are only taking the flash onset indeces)
                    foi = flashes_win_yyalltrace_index[np.logical_and(flashes_win_yyalltrace_index >= r0, flashes_win_yyalltrace_index < r1)] - r0
                    # omissions
                    ooi = omissions_win_yyalltrace_index[np.logical_and(omissions_win_yyalltrace_index >= r0, omissions_win_yyalltrace_index < r1)] - r0
                    # licks
                    loi = licks_win_yyalltrace_index[np.logical_and(licks_win_yyalltrace_index >= r0, licks_win_yyalltrace_index < r1)] - r0
                    # rewards
                    roi = rewards_win_yyalltrace_index[np.logical_and(rewards_win_yyalltrace_index >= r0, rewards_win_yyalltrace_index < r1)] - r0
                    
                    # running
                    run = run_scale[r0:r1]

                    
                    f = plt.figure(figsize=(20, 10))
                    plt.plot(y_true, color=col_true, label='true') 
                    plt.plot(y_pred, color=col_pred, label='predicted')    
                    # running
                    plt.plot(run, color='b', label='running')    
                    
                    # mark training and testing frames on yy_alltrace  
                    plt.plot(tri, np.ones(tri.shape), 'k.', label='training frames') #(train_data_inds_on_traces_y0, np.ones(train_data_inds_on_traces_y0.shape), 'r.')
                    plt.plot(tsi, np.ones(tsi.shape), 'r.', label='testing frames')

                    # mark flashs (the whole duration, in frame units) for this chunck of data
                    for ifl in range(len(foi)):
                        if ifl==0:
                            plt.axvspan(foi[ifl], foi[ifl] + flash_dur/frame_dur, alpha=.5, facecolor='y', label='images')
                        else:
                            plt.axvspan(foi[ifl], foi[ifl] + flash_dur/frame_dur, alpha=.5, facecolor='y')
                    # omissions
                    for ifl in range(len(ooi)):
                        if ifl==0:                        
                            plt.axvspan(ooi[ifl], ooi[ifl] + flash_dur/frame_dur, alpha=.5, facecolor='r', label='omissions')
                        else:
                            plt.axvspan(ooi[ifl], ooi[ifl] + flash_dur/frame_dur, alpha=.5, facecolor='r')
                        
                    # licks
                    for ifl in range(len(loi)):
                        if ifl==0:
                            plt.plot(loi[ifl], ymx-ry, '.g', label='licks')
                        else:
                            plt.plot(loi[ifl], ymx-ry, '.g')
                    # rewards
                    for ifl in range(len(roi)):
                        if ifl==0:
                            plt.plot(roi[ifl], ymx-ry, '.m', label='rewards', markersize=15)
                        else:
                            plt.plot(roi[ifl], ymx-ry, '.m', markersize=15)
                        
                    plt.ylim(y_min, y_max)
                    plt.xticks(np.arange(0,samples_per_plot, samples_per_plot/5), np.arange(r0, r1, samples_per_plot/5).astype(int))
                    plt.legend(loc='upper left', frameon=False, handlelength=1, fontsize=12) #'center left' , bbox_to_anchor=(1,.7)
                    plt.title(titl)
                    makeNicePlots(plt.gca())

                    if savefigs:
                        regex = re.compile(f"{aname}(.*).{fmt}")
                        files = [string for string in list_figs if re.match(regex, string)]

                        if len(files)>0: #os.path.isfile(os.path.join(dir_planePair, files)):
                            files = files[-1] # take the last file
                            print(f"Figure exists: {files}")

                        else:
                            pdf.savefig(f, bbox_inches='tight')

                pdf.close()


                # plot the entire trace in one signle pdf file (instead of doing pages like above) 
                '''
                plt.figure(figsize=(int(traces_x0.shape[1]*3e-4), 3))            
                plt.plot(yy_alltrace, color=col_true, label='true') # traces_y0[neuron_y]
                plt.plot(y_pred_alltrace, color=col_pred, label='predicted')
                # mark training and testing frames on yy_alltrace  
                plt.plot(train_data_inds_on_yy_alltrace, np.ones(train_data_inds_on_yy_alltrace.shape), 'k.', label='training frames') # (train_data_inds_on_traces_y0, np.ones(train_data_inds_on_traces_y0.shape), 'r.')
                plt.plot(test_data_inds_on_yy_alltrace, np.ones(test_data_inds_on_yy_alltrace.shape), 'r.', label='testing frames')
                # mark flashes
                for i in range(len(flashes_win_yyalltrace_index)):
                    plt.axvspan(flashes_win_yyalltrace_index[i], flashes_win_yyalltrace_index[i] + flash_dur/frame_dur, alpha=.5, facecolor='y')
                plt.legend(loc='center left', bbox_to_anchor=(1,.7), frameon=False, handlelength=1, fontsize=12)
                plt.title('neuron %d, entire session, expained_var %.2f' %(neuron_y, r_sq_alltrace))
                makeNicePlots(plt.gca())

                if savefigs:
                    fign = os.path.join(dir_planePair, 'trace_pred_allTrace0_neuronY_%d_' %neuron_y + plp+nowStr+'.'+'pdf')
                    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
                '''


                ########################################################################
                ############ plot y_train and compare it with the prediction ############ 
                ########################################################################

                y_train_pred = y_pred_train_all[neuron_y] #autoencoder.predict(x_train)
                print('explained variance for y_train: ', r_sq_train_all[neuron_y])
            #    loss_bestModel_all[neuron_y] # mean_sqr_err(y_train_pred, y_train)

                aname = f"trace_pred_train_neuronY_rank{rank_neurons[neuron_y]}_ind{neuron_y}_{plp}"
                fign = os.path.join(dir_planePair, f"{aname}{nowStr}.pdf")            
                titl = 'neuron %d, training data, expained_var %.2f' %(neuron_y, r_sq_train_all[neuron_y])
                y_min = min(y_train)
                y_max = max(y_train)
                lent = y_train.shape[0]

                if lent > samples_per_plot:
                    pdf = matplotlib.backends.backend_pdf.PdfPages(fign)

                    for i in range(int(lent / samples_per_plot) + 1):
                        r0 = i * samples_per_plot
                        r1 = (i + 1) * samples_per_plot

                        y_true = y_train[r0:r1]
                        y_pred = y_train_pred[r0:r1]

                        f = plt.figure(figsize=(20, 10))
                        plt.plot(y_true, color=col_true, label='true') 
                        plt.plot(y_pred, color=col_pred, label='predicted')    

                        plt.ylim(y_min, y_max)
                        plt.xticks(np.arange(0,samples_per_plot, samples_per_plot/5), np.arange(r0, r1, samples_per_plot/5).astype(int))
                        plt.legend(loc='center left', bbox_to_anchor=(1,.7), frameon=False, handlelength=1, fontsize=12)
                        plt.title(titl)
                        makeNicePlots(plt.gca())

                        if savefigs:
                            regex = re.compile(f'{aname}(.*).{fmt}')
                            files = [string for string in list_figs if re.match(regex, string)]

                            if len(files)>0: 
                                files = files[-1] # take the last file
                                print(f"Figure exists: {files}")                    
                            else:
                                pdf.savefig(f)

                    pdf.close()

                else: 

                    plt.figure(figsize=(int(traces_x0.shape[1]*3e-4), 3))

                    plt.plot(y_train, color=col_true, label='true') # traces_y0[neuron_y]
                    plt.plot(y_train_pred, color=col_pred, label='predicted')
                    plt.legend(loc='center left', bbox_to_anchor=(1,.7), frameon=False, handlelength=1, fontsize=12)
                    plt.title(titl)
                    makeNicePlots(plt.gca())

                    if savefigs:
                        save_fig_if_notexist(aname, dir_planePair, nowStr, fmt='pdf')



                ########################################################################
                ############ plot y_test and compare it with the prediction ############
                ########################################################################            
                y_test_pred = y_pred_test_all[neuron_y] #autoencoder.predict(x_test)
                print('explained variance for y_test: ', r_sq_test_all[neuron_y])
            #    loss_bestModel_all[neuron_y] # mean_sqr_err(y_test_pred, y_test)

                aname = f"trace_pred_test_neuronY_rank{rank_neurons[neuron_y]}_ind{neuron_y}_{plp}"
                fign = os.path.join(dir_planePair, f"{aname}{nowStr}.pdf")            
                titl = 'neuron %d, testing data, expained_var %.2f' %(neuron_y, r_sq_test_all[neuron_y])
                y_min = min(y_test)
                y_max = max(y_test)
                lent = y_test.shape[0]

                if lent > samples_per_plot:
                    pdf = matplotlib.backends.backend_pdf.PdfPages(fign)

                    for i in range(int(lent / samples_per_plot) + 1):
                        r0 = i * samples_per_plot
                        r1 = (i + 1) * samples_per_plot

                        y_true = y_test[r0:r1]
                        y_pred = y_test_pred[r0:r1]

                        f = plt.figure(figsize=(20, 10))
                        plt.plot(y_true, color=col_true, label='true') 
                        plt.plot(y_pred, color=col_pred, label='predicted')    

                        plt.ylim(y_min, y_max)
                        plt.xticks(np.arange(0,samples_per_plot, samples_per_plot/5), np.arange(r0, r1, samples_per_plot/5).astype(int))
                        plt.legend(loc='center left', bbox_to_anchor=(1,.7), frameon=False, handlelength=1, fontsize=12)
                        plt.title(titl)
                        makeNicePlots(plt.gca())

                        if savefigs:
                            regex = re.compile(f'{aname}(.*).{fmt}')
                            files = [string for string in list_figs if re.match(regex, string)]
                            if len(files)>0: 
                                files = files[-1] # take the last file
                                print(f"Figure exists: {files}")                    
                            else:
                                pdf.savefig(f)
                    pdf.close()

                else:         
                    plt.figure(figsize=(int(traces_x0.shape[1]*3e-4), 3))

                    plt.plot(y_test, color=col_true, label='true') # traces_y0[neuron_y]
                    plt.plot(y_test_pred, color=col_pred, label='predicted')    
                    plt.legend(loc='center left', bbox_to_anchor=(1,.7), frameon=False, handlelength=1, fontsize=12)
                    plt.title(titl)
                    makeNicePlots(plt.gca())

                    if savefigs:
                        save_fig_if_notexist(aname, dir_planePair, nowStr, fmt='pdf')



                ####### plot histogram of r2 for random data, also mark training and testing r2
                '''
                plt.figure()
                plothist(r_sq_rand_all[neuron_y]) #, nbins=50, doSmooth=5)
                plt.vlines(r_sq_test_all[neuron_y], 0, .01, 'b')
                plt.vlines(r_sq_train_all[neuron_y], 0, .01, 'k')
                '''

                gc.collect()                   
                K.clear_session()

    
###########################################################################################
###########################################################################################

sys.exit('End here!')

#%%

#y_true = y_test

plt.figure(figsize=(10,4))
plt.plot(y_pred, 'r', label='predicted')
plt.plot(y_true, 'b', label='actual')

plt.legend()
plt.title('average across feature space')
plt.xlabel('observations')


#%% Simple reconstruction: here we only predict y_train using x_train, without adding the testing trials and further reshaping the output 
# Reconstruct the input (decoded); also, plot average of the testing dataset


# NOTE: on the average trace, y_pred is more noisy than x_test!!!
    
mean_actual = np.mean(y_true, axis=1) 
mean_decoded = np.mean(y_pred, axis=1) 

#get_ipython().magic(u'matplotlib qt')

#### plot average trace across feature space
plt.figure(figsize=(10,4))
plt.plot(mean_decoded, 'r', label='predicted')
plt.plot(mean_actual, 'b', label='actual')
#plt.plot(np.argwhere(evs[neuron_y]), np.full((np.argwhere(evs[neuron_y])).shape, 5),'y.', label='events'); # max(traces[neuron_y]) 

plt.legend()
plt.title('average across feature space')
plt.xlabel('observations')




#%%
###############################
####################################
########################################
############################################
#################################################
########################################################

#%%
    
loss_all = np.array(loss_all)
val_loss_all = np.array(val_loss_all)
# hidden_layers_all = np.array(hidden_layers_all)
val_err_mx_all = np.array(val_err_mx_all)

print(loss_all.shape) # num_subsamp x epochs
print(val_loss_all.shape) # num_subsamp x epochs


#%% Find the best model across epochs, i.e. the one with minimum val_loss; set loss and val_loss at that epoch.
    
[ind_bestModel, loss_bestModel, val_loss_bestModel] = set_bestModel_loss(loss_all, val_loss_all, err_mx_all=1, val_err_mx_all=1)


#%% Plot loss for different parameter values   

x = np.arange(iters_neur_subsamp) #cvect # x = batch_size_all
xlab = 'Iteration (subsampling neurons)' #'regularization values' # xlab = 'batch size'

#r = range(len(x)) 
plt.figure(figsize=(6,6))    

plt.subplot(211)
plt.title('Raw loss')
plt.plot(x, loss_bestModel, 'b.-', label='training')
plt.plot(x, val_loss_bestModel, 'r.-', label='training')
plt.xlabel(xlab)
plt.ylabel('Loss')
plt.legend(loc=0, frameon=False)

plt.subplot(212)
plt.title('Norm loss')
plt.plot(x, loss_bestModel_norm, 'b.-')
plt.plot(x, val_loss_bestModel_norm, 'r.-')
#plt.plot(x, loss_bestModel_norm, 'g.-')
plt.xlabel(xlab)
plt.ylabel('Loss')

plt.subplots_adjust(hspace=1)

'''    
#plt.plot(x[r], loss_all[:,-1][r] / err_mx, 'b.-')
#plt.plot(x[r], val_loss_all[:,-1][r] / val_err_mx, 'r.-')    
plt.plot(x[r], loss_all[:,-1][r], 'b.-')
plt.plot(x[r], val_loss_all[:,-1][r], 'r.-')    
plt.xlabel(xlab)
#plt.xscale('log')
'''


#%% Plot loss for different number of hidden layers (depth size), for each latent size
'''
for ils in range(len(latent_size_all)): # ils = 0

    latent_size = np.array(latent_size_all[ils])
    loss_all1 = np.array(loss_all[ils])
    val_loss_all1 = np.array(val_loss_all[ils])
    hidden_layers_all1 = np.array(hidden_layers_all[ils])
    
    # xv = latent_size_all
    # xv = depth_size_ind_all[::-1]
    xv = [len(x) for x in hidden_layers_all1[::-1]]
    print(list(hidden_layers_all1[::-1]))
    print('\n')
    
    plt.figure()    
    plt.plot(xv, loss_all1[:,-1], 'b.-')
    plt.plot(xv, val_loss_all1[:,-1], 'r.-')    
    plt.title('latent_size: %d' %(latent_size))
    plt.xlabel('# encoding layers')



    
#%% Find the optimal regularization value

smallestC = 0 # just pick the reg value that gives the min error.

# take the loss value at the last epoch
meanErrorTrain = loss_all[:, -1] 
meanErrorTest = val_loss_all[:, -1]
  
  
# Identify best c                 # Use all range of c... it may end up a value at which all weights are 0.
ix = np.argmin(meanErrorTest)

if smallestC==0:
    cbest = cvect[ix]
else:    # here, we are picking the largest lambda (smallest C)
    cbest = cvect[meanErrorTest <= (meanErrorTest[ix]+semErrorTest[ix])]
    cbest = cbest[-1] #[0] # best regularization term based on minError+SE criteria
    
print('\t%f' %(cbest))
'''





#%% Simple reconstruction: here we only predict y_train using x_train, without adding the testing trials and further reshaping the output 
# Reconstruct the input (decoded); also, plot average of the testing dataset

# use the non-shuffled versions of x and y, so we see the calcium events

x_true = x_train
y_true = y_train

x_true = x_test #x_train0[:,np.newaxis,:,:]
y_true = y_test #y_train0

#x_true = xx_alltrace[:,np.newaxis,:,:]
#y_true = yy_alltrace


decoded = autoencoder.predict(x_true)
print(decoded.shape)


#%% Compute loss manually and compare with keras loss; also plot loss for each neuron
# note if you use regularization, keras loss will be different from manual loss:
# see this : https://stackoverflow.com/questions/49903706/keras-predict-gives-different-error-than-evaluate-loss-different-from-metrics
# It turns out that the loss is supposed to be different than the metric, because of the regularization. Using regularization, the loss is higher (in my case), because the regularization increases loss when the nodes are not as active as specified. The metrics don't take this into account, and therefore return a different value, which equals what one would get when manually computing the error.

err_eachNeuron = mean_sqr_err(y_true, decoded)
err = np.mean(err_eachNeuron)
print('loss(manual, keras):', err, loss[-1])

if len(err_eachNeuron)>1:
    plt.figure()
    plt.plot(err_eachNeuron)
    plt.xlabel('Neuron')
    plt.ylabel('Loss')

#d = (recover_orig_ytrain - recover_decoded_ytrain)**2
#print(np.nanmean(d))


#%% NOTE: on the average trace, decoded is more noisy than x_test!!!
    
mean_actual = np.mean(y_true, axis=1) 
mean_decoded = np.mean(decoded, axis=1) 

#get_ipython().magic(u'matplotlib qt')

#### plot average trace across feature space
plt.figure(figsize=(10,4))
plt.plot(mean_decoded, 'r', label='decoded')
plt.plot(mean_actual, 'b', label='actual')
#plt.plot(np.argwhere(evs[neuron_y]), np.full((np.argwhere(evs[neuron_y])).shape, 5),'y.', label='events'); # max(traces[neuron_y]) 

plt.legend()
plt.title('average across feature space')
plt.xlabel('observations')


#%%
#### plot trace for individual features
ifeat = rnd.permutation(y_true.shape[1])[0]

plt.figure()
plt.subplot(211)
plt.plot(y_true[:, ifeat], 'b')
#plt.title('y_true')
#plt.plot(np.argwhere(evs[neuron_y]), np.full((np.argwhere(evs[neuron_y])).shape, 1),'g.', label='events'); # max(traces[neuron_y]) 

#plt.figure()
plt.subplot(212)
plt.plot(decoded[:, ifeat], 'r')
#plt.title('y_pred')
#plt.plot(np.argwhere(evs[neuron_y]), np.full((np.argwhere(evs[neuron_y])).shape, 1),'g.', label='events'); # max(traces[neuron_y]) 
plt.xlabel('observations')



#%% Plot decoded and compare with actual: plot individual observations, all features

get_ipython().magic(u'matplotlib inline')   

for isamp in rnd.permutation(y_true.shape[0]):
    plt.figure(figsize=(10,4))
    plt.plot(y_true[isamp, :], 'b')
    plt.plot(decoded[isamp, :], 'r')
    plt.pause(.01)
#    plt.cla()


#%% Plot decoded and compare with actual: plot individual features, all observations

get_ipython().magic(u'matplotlib inline')   

for ineur in rnd.permutation(y_true.shape[1]):
    plt.figure(figsize=(10,4))
    plt.plot(y_true[:, ineur], 'b')
    plt.plot(decoded[:, ineur], 'r')
    plt.pause(.01)



#%%
#################################################################
#################################################################
#################################################################
#################################################################

#%% Reconstruct the output trace from the model
# here we use both training and testing datasets ... also we reshape the final trace to match the original input dimensions.
    
trial_inds = 0 # set to 1 if train_data_inds and test_data_inds are trial indeces

decoded_ytrain = autoencoder.predict(x_train)
decoded_ytest = autoencoder.predict(x_test)

# both y train and y test
recover_decoded= recover_orig_traces(decoded_ytrain, decoded_ytest, train_data_inds, test_data_inds, trial_inds)
print(recover_decoded.shape)

recover_orig = recover_orig_traces(y_train, y_test, train_data_inds, test_data_inds, trial_inds)
print(recover_orig.shape)



##### recover ytest
# set ytrain to nan
ynow = np.full(y_train.shape, np.nan)
recover_decoded_ytest = recover_orig_traces(ynow, decoded_ytest, train_data_inds, test_data_inds, trial_inds)
print(recover_decoded_ytest.shape)

recover_orig_ytest = recover_orig_traces(ynow, y_test, train_data_inds, test_data_inds, trial_inds)
print(recover_orig_ytest.shape)


##### recover ytrain
# set ytest to nan
ynow = np.full(y_test.shape, np.nan)
recover_decoded_ytrain = recover_orig_traces(decoded_ytrain, ynow, train_data_inds, test_data_inds, trial_inds)
print(recover_decoded_ytrain.shape)

recover_orig_ytrain = recover_orig_traces(y_train, ynow, train_data_inds, test_data_inds, trial_inds)
print(recover_orig_ytrain.shape)


recover_orig_ytrain0 = recover_orig_ytrain + 0
recover_orig_ytest0 = recover_orig_ytest + 0
recover_decoded_ytrain0 = recover_decoded_ytrain + 0
recover_decoded_ytest0 = recover_decoded_ytest + 0


#%% Plot the original and reconstructed traces; averaged across neurons (and trials, if the traces are set that way.)

avax = tuple(np.arange(1, np.ndim(recover_orig_ytrain)).flatten()) # (1,2)

plt.figure(figsize=(10,7))

### train
plt.subplot(221)
p = np.nanmean(recover_orig_ytrain, axis=avax)
plt.plot(p, 'g', label='original')

p = np.nanmean(recover_decoded_ytrain, axis=avax)
plt.plot(p, 'r', label='reconstructed')

plt.title('y_train')#, fontsize=10)
#plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 1), frameon=False)
plt.xlabel('frame since image onset', fontsize=10)


### test
plt.subplot(222)
p = np.nanmean(recover_orig_ytest, axis=avax)
plt.plot(p, 'g', label='original')

p = np.nanmean(recover_decoded_ytest, axis=avax)
plt.plot(p, 'r', label='reconstructed')

plt.title('y_test')#, fontsize=10)
plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 1), frameon=False)
plt.xlabel('frame since image onset', fontsize=10)



######## shift and scale the original and the recovered traces

### train, scaled
plt.subplot(223)
# p = np.nanmean(traces0[frames2ana], axis=avax)
p = np.nanmean(recover_orig_ytrain, axis=avax)
p = p-p[0]
p = p / max(p)
plt.plot(p, 'g', label='original')

p = np.nanmean(recover_decoded_ytrain, axis=avax)
p = p-p[0]
p = p / max(p)
plt.plot(p, 'r', label='reconstructed')

plt.title('y_train, shifted and scaled', fontsize=10)
# plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 1), frameon=False)
plt.xlabel('frame since image onset', fontsize=10)


### test, scaled
plt.subplot(224)
# p = np.nanmean(traces0[frames2ana], axis=avax)
p = np.nanmean(recover_orig_ytest, axis=avax)
p = p-p[0]
p = p / max(p)
plt.plot(p, 'g', label='original')

p = np.nanmean(recover_decoded_ytest, axis=avax)
p = p-p[0]
p = p / max(p)
plt.plot(p, 'r', label='reconstructed')

plt.title('y_test, shifted and scaled', fontsize=10)
#plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 1), frameon=False)
plt.xlabel('frame since image onset', fontsize=10)


plt.subplots_adjust(wspace=.2, hspace=.5)


'''
plt.figure()
plt.subplot(221)
plt.plot(np.nanmean(traces0[frames2ana], axis=avax), 'b')
plt.plot(np.nanmean(recover_orig_ytrain, axis=avax), 'g')
plt.plot(np.nanmean(recover_decoded_ytrain, axis=avax), 'r')
plt.title('train')

plt.subplot(222)
plt.plot(np.nanmean(traces0[frames2ana], axis=avax), 'b')
plt.plot(np.nanmean(recover_orig_ytest, axis=avax), 'g')
plt.plot(np.nanmean(recover_decoded_ytest, axis=avax), 'r')
plt.subplots_adjust(wspace=1)
plt.title('test')
'''


#%% Look at single-trial reconstruction for individual neurons

# decide how many neurons (columns) and trials (rows) you want to plot; we will pick them randomly.
nn2p = 20 # number of neurons to plot
nt2p = 5 # number of trials to plot
if np.ndim(recover_orig_ytrain0)==2:
    nt2p = 1
    
# only take non-nan trials
if np.ndim(recover_orig_ytrain0)==3:
    recover_orig_ytrain = recover_orig_ytrain0[:,:,train_data_inds]
    recover_orig_ytest = recover_orig_ytest0[:,:,test_data_inds]
    recover_decoded_ytrain = recover_decoded_ytrain0[:,:,train_data_inds]
    recover_decoded_ytest = recover_decoded_ytest0[:,:,test_data_inds]
else:
    recover_orig_ytrain = recover_orig_ytrain0[:,:,np.newaxis]
    recover_orig_ytest = recover_orig_ytest0[:,:,np.newaxis]
    recover_decoded_ytrain = recover_decoded_ytrain0[:,:,np.newaxis]
    recover_decoded_ytest = recover_decoded_ytest0[:,:,np.newaxis]    

for scale_shift in [0]: #[0,1]: # scale_shift = 1

    ### Training data ### 
    
    nc = rnd.permutation(recover_orig_ytrain.shape[1])[:nn2p] # neurons
    nr = rnd.permutation(recover_orig_ytrain.shape[2])[:nt2p] # trials
    allsub = len(nc) * len(nr)
    
    plt.figure(figsize=(1.5*nn2p, 1.5*nt2p))

    titl = 'Training data_scaled & shifted' if scale_shift else 'Training data'
    plt.suptitle(titl, fontsize=13)

    iic = -1
    for ic in nc: # loop over neurons
        iic = iic+1
        iir = -1
        for ir in nr: # loop over trials
            iir = iir+1
            
            nsu = iic + iir*len(nc) + 1
            
            plt.subplot(len(nr), len(nc), nsu)
            
            p = recover_orig_ytrain[:, ic, ir]
            if scale_shift:
                p = p-p[0]
                p = p / max(p)
            plt.plot(p, 'g', label='original')
#            plt.xticks([], [])
            
            p = recover_decoded_ytrain[:, ic, ir]
            if scale_shift:
                p = p-p[0]
                p = p / max(p)
            plt.plot(p, 'r', label='reconst.')
#            plt.axis('off')
            plt.gca().tick_params(axis='both', which='major', labelsize=9)
            plt.gca().grid(False)
            
            if nsu==1:    #            plt.title('y_train')#, fontsize=10)
                plt.legend(fontsize=10, loc='top left', bbox_to_anchor=(-.2, 1.2), frameon=False)
            if nsu==allsub-(len(nc)-1):
                plt.xlabel('frame since image onset', fontsize=10)
                plt.ylabel('DF/F', fontsize=10)
                
                plt.subplots_adjust(wspace=.5, hspace=.5)
            
    
    
    
    ### Testing data ### 
    
    # below is commented so we plot the modeled trace of the same neurons plotted above but for testing data 
#    nc = rnd.permutation(recover_orig_ytest.shape[1])[:nn2p] # neurons
    nr = rnd.permutation(recover_orig_ytest.shape[2])[:nt2p] # trials
    allsub = len(nc) * len(nr)    # allsub = recover_orig_ytest.shape[1] * recover_orig_ytest.shape[2]
    
    plt.figure(figsize=(1.5*nn2p, 1.5*nt2p))
    titl = 'Testing data_scaled & shifted' if scale_shift else 'Testing data'
    plt.suptitle(titl, fontsize=13)
    
    iic = -1
    for ic in nc: # loop over neurons
        iic = iic+1
        iir = -1
        for ir in nr: # loop over trials
            iir = iir+1
            
            nsu = iic + iir*len(nc) + 1
            
            plt.subplot(len(nr), len(nc), nsu)
            
            p = recover_orig_ytest[:, ic, ir]
            if scale_shift:
                p = p-p[0]
                p = p / max(p)
            plt.plot(p, 'g', label='original')
    #        plt.axis('off')
            
            p = recover_decoded_ytest[:, ic, ir]
            if scale_shift:
                p = p-p[0]
                p = p / max(p)        
            plt.plot(p, 'r', label='reconst.')
#            plt.axis('off')
            plt.gca().tick_params(axis='both', which='major', labelsize=9)
            plt.gca().grid(False)
            
            if nsu==1:    #            plt.title('y_test')#, fontsize=10)
                plt.legend(fontsize=10, loc='top left', bbox_to_anchor=(-.2, 1.2), frameon=False)
            if nsu==allsub-(len(nc)-1):
                plt.xlabel('frame since image onset', fontsize=10)
                plt.ylabel('DF/F', fontsize=10)
                
            plt.subplots_adjust(wspace=.5, hspace=.5)
            





    

#%% Plot histogram of loss and val_loss across neurons
'''
#a = loss_bestModel_all / err_mx_all
#b = val_loss_bestModel_all / val_err_mx_all
#lab = 'loss value'

a = r_sq_train_all
b = r_sq_test_all

plot_hist(a,b)


a = r_sq_test_all
b = r_sq_rand_all
plot_hist(a,b)


print('Fraction neurons with r_sqr>0:\n', np.mean(r_sq_test_all>0))
print('Average r_sqr of neurons with r_sqr>0:\n', np.mean(r_sq_test_all[r_sq_test_all>0]))

if savefigs:           
#    dd = alN + corrn + 'numNeurons_days_' + days[0][0:6] + '-to-' + days[-1][0:6] + '_' + nowStr            
    fign = os.path.join(dir_planePair, 'r2_hist_trainTest_'+nowStr+'.'+'pdf')
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)
'''


#%%
'''
loss_bestModel_all = np.concatenate((loss_bestModel_all0, loss_bestModel_all))
val_loss_bestModel_all = np.concatenate((val_loss_bestModel_all0, val_loss_bestModel_all))

err_mx_all = np.concatenate((err_mx_all0, err_mx_all))
val_err_mx_all = np.concatenate((val_err_mx_all0, val_err_mx_all))

r_sq_train_all = np.concatenate((r_sq_train_all0, r_sq_train_all))
r_sq_test_all = np.concatenate((r_sq_test_all0, r_sq_test_all))

y_pred_train_all = np.concatenate((y_pred_train_all0, y_pred_train_all))
y_pred_test_all = np.concatenate((y_pred_test_all0, y_pred_test_all))    

r_sq_rand_all = np.concatenate((r_sq_rand_all0, r_sq_rand_all)) # num_neurons x 50 
    
pmw_rand_test = np.concatenate((pmw_rand_test0, pmw_rand_test))
pt1_rand_test = np.concatenate((pt1_rand_test0, pt1_rand_test))
'''


#%% Shuffle samples in x and y (model fit does it, but i would like to do it here too)
# we are keeping the order of frames in win the same! ... 
'''
#shfl_ord = rnd.permutation(num_samps)
#xx = xx0[shfl_ord] # samples x neurons x 10 
#yy = yy0[shfl_ord] # samples x neurons
#xx.shape, yy.shape

shfl_ord_train = rnd.permutation(x_train0.shape[0])

x_train00 = x_train0[shfl_ord_train] # num_samps x neurons x len_win
y_train = y_train0[shfl_ord_train]


shfl_ord_test = rnd.permutation(x_test0.shape[0])

x_test00 = x_test0[shfl_ord_test]
y_test = y_test0[shfl_ord_test]

print(np.shape(x_train00) , np.shape(x_test00))  # samples x neurons x 10 
print(np.shape(y_train) , np.shape(y_test))  # samples x neurons


#%% Concatenate neurons and time axes for x data

# each row includes includes 10 frames from neuron 1, then 10 frames from neuron 2, etc
x_train = np.reshape(x_train00, (x_train0.shape[0], x_train0.shape[1]*x_train0.shape[2])) # num_samps x (neurons x 10 frames)
x_test = np.reshape(x_test00, (x_test0.shape[0], x_test0.shape[1]*x_test0.shape[2]))

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#for isamp in range(x_train.shape[0]):
#    x_train[isamp]
'''    



#%% subsample neurons in y if it has fewer neurons than x.
'''
n_neurs_subsamp = np.min((traces_x.shape[0], traces_y.shape[0]))
print('Subsampling %d neurons' %n_neurs_subsamp)

#x_rand_neurs = rnd.permutation(x_train0.shape[1])[:n_neurs_subsamp]
y_rand_neurs = rnd.permutation(traces.shape[0])[:n_neurs_subsamp]

#x_train = x_train0[:, x_rand_neurs]
traces_y = traces_y[y_rand_neurs, :]

#traces_y = traces
traces_y.shape
'''

#%% Set samples (training and testing datasets)
# eg. frames 0-9 of neuron 1, then frames 0-9 of neuron 2, etc... all concatentanated. This array is sample 1.
# frames 1-10 of neuron 1, then frames 1-10 of neuron 2, etc .... this array will be sample 2....
# we randomly choose 10% of these samples of testing dataset, and 90% of them as training dataset.
'''
len_training_frs = 10 # Get 1 sec of the traces, as one training data point. # 10Hz, so 1sec will be 10 frames
# eg. frames 0-9 of neuron 1, then frames 0-9 of neuron 2, etc... all concatentanated. This array is sample 1.
# frames 1-10 of neuron 1, then frames 1-10 of neuron 2, etc .... this array will be sample 2....
# we randomly choose 10% of these samples of testing dataset, and 90% of them as training dataset.

num_samps = traces_x.shape[1] - len_training_frs # + 1 # total number of training and testing data points
nfeatures = traces_x.shape[0] * len_training_frs # length of each sample (data point) # (num_neurons x 10)

dataNeurons_all = np.full((num_samps, nfeatures), np.nan)

for isamp in range(num_samps): # isamp = 0

    dataNeurons0 = traces_x[:, isamp: isamp + len_training_frs] # neurons x 10 frames

    dataNeurons_all[isamp,:] = dataNeurons0.flatten() # includes 10 frames from neuron 1, then 10 frames from neuron 2, etc
    
print(dataNeurons_all.shape) # num_samps x (neurons x 10 frames)


#%%
#### Now assign 10% random samps as testing dataset

# assign testing and training datasets
r = rnd.permutation(num_samps)
test_data_inds = r[:int(kfold/100 * num_samps)+1]
train_data_inds = r[int(kfold/100 * num_samps)+1:]
print(len(test_data_inds) , len(train_data_inds))

x_train = dataNeurons_all[train_data_inds, :]
x_test = dataNeurons_all[test_data_inds, :]
print(np.shape(x_train) , np.shape(x_test))  # num_samps x (neurons x 10 frames)
'''





 
#%% Alternative way of setting a deep autoencoder (allows setting dropouts)
'''
autoencoder = Sequential()

#autoencoder.add(Dropout(0, input_shape=(x_nfeatures,)))
#autoencoder.add(Dense(1000, activation=activation_fun))
# autoencoder.add(Dense(1000, activation=activation_fun, input_dim=x_nfeatures)) # 
autoencoder.add(Dense(500, activation=activation_fun, input_dim=x_nfeatures)) # 
#autoencoder.add(Dropout(0.2))
autoencoder.add(Dense(100, activation=activation_fun))
autoencoder.add(Dense(50, activation=activation_fun))
# autoencoder.add(Dense(5, activation=activation_fun))
# autoencoder.add(Dense(50, activation=activation_fun))
autoencoder.add(Dense(100, activation=activation_fun))
autoencoder.add(Dense(500, activation=activation_fun))
# autoencoder.add(Dense(1000, activation=activation_fun))
autoencoder.add(Dense(y_nfeatures, activation=out_activation_fun))
'''


#%%
"""
input_data = Input(shape = (x_nfeatures,))
# encoded = Dense(500, activation=activation_fun, activity_regularizer=regularizers.l1(10e-8))(input_data) # 128 # Dropout(0.2 # kernel_constraint=maxnorm(3) # , activity_regularizer=regularizers.l1(10e-5) 
# encoded = Dense(500, activation=activation_fun)(input_data) # 128 # Dropout(0.2 # kernel_constraint=maxnorm(3) # , activity_regularizer=regularizers.l1(10e-5) 
# encoded = Dense(100, activation=activation_fun)(encoded)
encoded1 = Dense(100, activation=activation_fun)(input_data)
encoded2 = Dense(latent_size, activation=activation_fun)(encoded1) # 50
#encoded = Dense(5, activation=activation_fun)(encoded) # 32
decoded1 = Dense(100, activation=activation_fun)(encoded2)
# decoded = Dense(500, activation=activation_fun)(decoded)  # 128
decoded2 = Dense(y_nfeatures, activation=out_activation_fun)(decoded1) # sigmoid

autoencoder = Model(input_data, decoded2)
"""



#%% number of parameters of a conv layer
'''
p: number of parameters of a conv layer
w: number of weights of a conv layer
b: number of biases of a conv layer

p = w + b

p = (k^2 x n_input_channels x n_kernels) + b
'''


#%%
'''
Model: "sequential_33"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_146 (Conv1D)          (None, 10, 16)            1888      # 3*39*16+16
_________________________________________________________________
max_pooling1d_57 (MaxPooling (None, 5, 16)             0         
_________________________________________________________________
conv1d_147 (Conv1D)          (None, 5, 8)              392       # 3*16*8+8
_________________________________________________________________
max_pooling1d_58 (MaxPooling (None, 3, 8)              0         
_________________________________________________________________
conv1d_148 (Conv1D)          (None, 3, 8)              200       
_________________________________________________________________
up_sampling1d_57 (UpSampling (None, 6, 8)              0         
_________________________________________________________________
conv1d_149 (Conv1D)          (None, 6, 16)             400       
_________________________________________________________________
up_sampling1d_58 (UpSampling (None, 12, 16)            0         
_________________________________________________________________
conv1d_150 (Conv1D)          (None, 10, 39)            1911      # 3*16*39+39
=================================================================
Total params: 4,791
Trainable params: 4,791
Non-trainable params: 0


#%%
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_130 (Conv1D)          (None, 390, 16)           64        # 3*1*16+16
_________________________________________________________________
max_pooling1d_51 (MaxPooling (None, 195, 16)           0         
_________________________________________________________________
conv1d_131 (Conv1D)          (None, 195, 8)            392       # 3*16*8+8
_________________________________________________________________
max_pooling1d_52 (MaxPooling (None, 98, 8)             0         
_________________________________________________________________
conv1d_132 (Conv1D)          (None, 98, 8)             200       
_________________________________________________________________
up_sampling1d_51 (UpSampling (None, 196, 8)            0         
_________________________________________________________________
conv1d_133 (Conv1D)          (None, 196, 16)           400       
_________________________________________________________________
up_sampling1d_52 (UpSampling (None, 392, 16)           0         
_________________________________________________________________
conv1d_134 (Conv1D)          (None, 390, 1)            49        
=================================================================
Total params: 1,105
Trainable params: 1,105
Non-trainable params: 0


#%%
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         (None, 1850)              0         
_________________________________________________________________
dense_25 (Dense)             (None, 1024)              1895424   
_________________________________________________________________
dense_26 (Dense)             (None, 64)                65600     
_________________________________________________________________
dense_27 (Dense)             (None, 32)                2080      
_________________________________________________________________
dense_28 (Dense)             (None, 64)                2112      
_________________________________________________________________
dense_29 (Dense)             (None, 1024)              66560     
_________________________________________________________________
dense_30 (Dense)             (None, 1850)              1896250   
=================================================================
Total params: 3,928,026
Trainable params: 3,928,026
Non-trainable params: 0
'''



#%% Add nan and transpose recover_decoded_ytrain so it matches "local_fluo_traces"

#local_fluo_traces.shape
#recover_decoded_ytrain.shape

# add the 10 (len_win) missing values to the end of the reconsturcted traces. (they are missing bc we used a window of size len_win to form them.)
# also transpose the trace so it matches local_fluo_traces : neurons x times
recover_decoded_ytrain_n = np.concatenate((recover_decoded_ytrain, np.full((len_win, recover_decoded_ytrain.shape[1]), np.nan))).T # neurons x times
recover_decoded_ytrain_n.shape # neurons x times



decoded_ytrain_omit_align, local_time_allOmitt = align_on_omit(recover_decoded_ytrain_n, local_time_traces)
decoded_ytrain_omit_align.shape


local_time_traces # average across planes


#%%
# it is the same for pairs of planes (bc they are recorded simultaneously)
a = this_sess_l.iloc[range(8)]['local_time_traces']
times = np.vstack((a.values)) # 8 x times
times.shape

# list_flashes and list_omitted are the same across all planes.
a = this_sess_l.iloc[range(8)]['list_flashes']
flash_times = np.vstack((a.values)) # 8 x num_flashes
flash_times.shape

a = this_sess_l.iloc[range(8)]['list_omitted']
omit_times = np.vstack((a.values)) # 8 x num_omissions
omit_times.shape



#%% Align on omission trials:
# samps_bef (=40) frames before omission ; index: 0:39
# omission frame ; index: 40
# samps_aft - 1 (=39) frames after omission ; index: 41:79
        

def align_on_omit(local_fluo_traces, local_time_traces):

    num_neurons = local_fluo_traces.shape[0]
    
    # Keep a matrix of omission-aligned traces for all neurons and all omission trials    
    local_fluo_allOmitt = np.full((samps_bef + samps_aft, num_neurons, len(list_omitted)), np.nan) # time x neurons x omissions_trials 
    local_time_allOmitt = np.full((samps_bef + samps_aft, len(list_omitted)), np.nan) # time x omissions_trials
    
    # Loop over omitted trials to align traces on omissions
    flashes_win_trace_all = []
    flashes_win_trace_index_all = []
    num_omissions = 0 # trial number
    
    for iomit in range(len(list_omitted)): # # indiv_time in list_omitted: # iomit = 0
        
        indiv_time = list_omitted.iloc[iomit]
        
        local_index = np.argmin(np.abs(local_time_traces - indiv_time)) # the index of omission on local_time_traces               
        
        be = local_index - samps_bef
        af = local_index + samps_aft
        
        if ~np.logical_and(be >= 0 , af <= local_fluo_traces.shape[1]): # make sure the omission is at least samps_bef frames after trace beigining and samps_aft before trace end. 
            print('Omission %d at time %f cannot be analyzed: %d timepoints before it and %d timepoints after it!' %(iomit, indiv_time, be, af)) 
            
        try:
            # Align on omission
            local_fluo_allOmitt[:,:, num_omissions] = local_fluo_traces[:, be:af].T # frame x neurons x omissions_trials (10Hz)
            # local_time_allOmitt is frame onset relative to omission onset. (assuming that the times in local_time_traces are frame onsets.... still checking on this, but it seems to be the rising edge, hence frame onset time!)
            # so local_time_allOmitt will be positive if frame onset is after omission onset.
            #
            # local_time_allOmitt shows the time of the frame that is closest to omission relative to omission.
            # (note, the closest frame to omission is called omission frame).
            # eg if local_time_allOmitt is -0.04sec, it means, the omission frame starts -.04sec before omission.
            # or if local_time_allOmitt is +0.04sec, it means, the omission frame starts +.04sec before omission.
            #
            # Note: the following two quantities are very similar (though time_orig is closer to the truth!): 
            # time_orig  and time_trace (defined below)
            # time_orig = np.mean(local_time_allOmitt, axis=1)                            
            # time_trace, time_trace_new = upsample_time_imaging(samps_bef, samps_aft, 31.) # set time for the interpolated traces (every 3ms time points)
            #
            # so we dont save local_time_allOmitt, although it will be more accurate to use it than time_trace, but the
            # difference is very minimum
            local_time_allOmitt[:, num_omissions] = local_time_traces[be:af] - indiv_time # frame x omissions_trials
            
            
            
            # Identify the flashes (time and index on local_time_allOmitt) that happened within local_time_allOmitt                    
            '''
            flashes_relative_timing = list_flashes - indiv_time # timing of all flashes in the session relative to the current omission
            # the time of flashes that happened within the omit-aligned trace
            a = np.logical_and(flashes_relative_timing >= local_time_allOmitt[0, num_omissions], flashes_relative_timing <= local_time_allOmitt[-1, num_omissions]) # which flashes are within local_time
            flashes_win_trace = flashes_relative_timing[a] 
            # now find the index of flashes_win_trace on local_time_allOmitt
            a = [np.argmin(np.abs(local_time_allOmitt[:, num_omissions] - flashes_win_trace[i])) for i in range(len(flashes_win_trace))]
            flashes_win_trace_index = a # the index of omission on local_time_traces
            '''
            plt.plot(local_time_allOmitt[:,num_omissions], local_fluo_allOmitt[:,:,num_omissions])
            plt.vlines(flashes_win_trace, -.4, .4)
    
            plt.plot(local_fluo_allOmitt[:,:,num_omissions])
            plt.vlines(flashes_win_trace_index, -.4, .4)
            '''
            num_omissions = num_omissions + 1
            flashes_win_trace_all.append(flashes_win_trace)
            flashes_win_trace_index_all.append(flashes_win_trace_index)
            '''
    #                    time 0 (omissions) values:
    #                    local_fluo_allOmitt[samps_bef,:,:]
    #                    local_time_allOmitt[samps_bef,:]
            
        except Exception as e:
            print(indiv_time, num_omissions, be, af, local_index)
    #                        print(e)
                                
                                    
    # remove the last nan rows from the traces, which happen if some of the omissions are not used for alignment (due to being too early or too late in the session) 
    if len(list_omitted)-num_omissions > 0:
        local_fluo_allOmitt = local_fluo_allOmitt[:,:,0:-(len(list_omitted)-num_omissions)]
        local_time_allOmitt = local_time_allOmitt[:,0:-(len(list_omitted)-num_omissions)]
    

    return local_fluo_allOmitt, local_time_allOmitt












#%%
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################

#%% Train omission-aligned traces

################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################

#%% Set frames to be analyzed: after omission or flash?

# frames2ana = np.arange(samps_bef, samps_bef+7) # Omission-evoked response # frames from omission until the following flash 
frames2ana = np.arange(samps_bef-7, samps_bef) # Flash-evoked response


#%% Subsample neurons in both x and y to the minimum number of neurons across planes. Then run the autoencoder model.

loss_all = []
val_loss_all = []
# hidden_layers_all = []
err_mx_all = []
val_err_mx_all = []
autoencoder_all = []
x_train_all = []
x_test_all = []
y_train_all = []
y_test_all = []

# depth_size = int(max((layer1_dim-latent_size)/10. , 1))
# (layer1_dim-latent_size) / depth_size

'''
# numDataPoints = x_train.shape[0]
# cvect = 10**(np.arange(-6, 6, 0.2))/numDataPoints    
# cvect = 10**(np.arange(-15, -5, 0.2))
cvect = 10**(np.arange(-30, -5, .5))
cvect = 10**(np.arange(-10, 10, .5))
cvect = 10**(np.arange(-7, -2, .2))
# cvect = 10**(np.arange(-20, -7, 0.2))
cvect = 10**(np.arange(-30, -5, .2))
cvect = 10**(np.arange(-15, -2, .8))
cvect = 10**(np.arange(-8, 1, .95))
cvect = 10**(np.arange(-15, 2, .2))
len(cvect)

# drapout fraction
# cvect = np.arange(0,1.1,.1) 
# len(cvect)

batch_size_all = np.arange(5, np.shape(x_train)[0], 50)    
latent_size_all = np.arange(5, layer1_dim, 5)
'''

#%% Set the index of X and Y traces for the autoencoder: plane index in this_sess_l

x_ind = 2 #2 #1 # LM
y_ind = 5 #num_depth + x_ind # V1


#%%    
for isubsamp in range(iters_neur_subsamp): # isubsamp = 0
    
    #%% Set the training and testing trials
    
    num_samps = ntrs #local_fluo_allOmitt_a1[0].shape[2] # num trials  # set 10% of trials as testing dataset (it will in fact be these all frames of these trials as testing dataset)
#    num_samps = len(frames2ana) * ntrs # num observations (ie frames2ana x trials)  # first concatenate frames of all trials; then assign testing dataset
    
    r = rnd.permutation(num_samps)
    train_data_inds = r[int(kfold/100 * num_samps)+1:]
    test_data_inds = r[:int(kfold/100 * num_samps)+1]    
    print(len(test_data_inds) , len(train_data_inds))
        

    #%% Preprocess data; loop through x_ind and y_ind, to set x and y traces for the autoencoder
    
    # norm_to_max_svm = 0
    # doZscore = 0
    # zEachFrame = 0 # if 0, normalize all frames to the same value. if 1, normalize each frame separately. # makes sense to set it to 0. 
    
    
    xy_train = []
    xy_test = []
    
    for xy_ind in [x_ind, y_ind]:
        
        # traces = local_fluo_traces # neurons x times  # entire session trace
        # traces = local_fluo_allOmitt # frames x units x trials # omission-aligned traces
        traces = this_sess_l.iloc[xy_ind]['local_fluo_allOmitt'] # frames x units x trials # omission-aligned traces
    
    
        print(traces.shape)
        traces00 = traces+0 # keep a copy of it


        #%% Take only those frames that you want to use in the model
        
        traces0 = traces00[frames2ana,:,:]
        
        # make sure what you want to train the autoencoder on, looks reasonable!!!
        # show average stimulus-aligned response (averaged across neurons and trials) for each plane
        plt.figure(figsize=(3,2))
        plt.plot(np.mean(traces0, axis=(1,2)))
    
    
        #%% Preprocess data : omission-aligned traces
        # NOTE: this should not be done here. we need to do scaling on training dataset! see: 
#        https://datascience.stackexchange.com/questions/27615/should-we-apply-normalization-to-test-data-as-well
        
        traces = traces0
        
        '''        
        ###%% Normalize each neuron trace by its max (so if on a day a neuron has low FR in general, it will not be read as non responsive!)
        # Note: because we are using z-scored data, norm to max wont make a difference.
        if norm_to_max_svm==1:
            
            T1, N1, C1 = traces0.shape
            X_svm_N = np.reshape(traces0.transpose(0 ,2 ,1), (T1*C1, N1), order = 'F') # (frames x trials) x units        
            traces_max = np.max(X_svm_N, axis=0) # neurons      # compute max on the entire trace of the session:          
            traces = np.transpose((np.transpose(traces0, (0,2,1)) / traces_max), (0,2,1)) 
            # plt.plot(traces_max)
            # plt.plot(np.mean(traces, axis=(1,2)))
            
            
        
        ###%% Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 
        
        if doZscore==1:
    
            ###### Normalize all frames to the same value. (problem for svm (decoding choice): some frames FRs may span a much larger range than other frames, hence svm solution will become slow)
            if zEachFrame==0:
                T1, N1, C1 = traces0.shape
                X_svm_N = np.reshape(traces0.transpose(0 ,2 ,1), (T1*C1, N1), order = 'F') # (frames x trials) x units
                    
                meanX_svm = np.mean(X_svm_N, axis = 0)
                stdX_svm = np.std(X_svm_N, axis = 0)
                
                if softNorm==1: # soft normalziation : neurons with sd<thAct wont have too large values after normalization
                    stdX_svm = stdX_svm + thAct                 
                    
                # normalize X
                X_svm_N = (X_svm_N - meanX_svm) / stdX_svm
                traces = np.reshape(X_svm_N, (T1, C1, N1), order = 'F').transpose(0 ,2 ,1)
            
            
            ###### Normalize each frame separately. (we do soft normalization)
            # Note: I don't think it is right to normalize each frame individually for the analysis here, because:
            # here we want to know if one population can predict the other population activity. For this we care 
            # about the frame-to-frame variation in neural activity; but if we normalize each frame individually,
            # we will remove this analysis-relevant information.
            if zEachFrame==1:
                X_svm_N = np.full(np.shape(traces0), np.nan)
                meanX_fr = []
                stdX_fr = []
            
                for ifr in range(np.shape(traces0)[0]):    
                    xf = traces0[ifr,:,:]    
            
                    m = np.mean(xf, axis=1) # across trials
                    s = np.std(xf, axis=1)   
            
                    if softNorm==1: # soft normalziation : neurons with sd<thAct wont have too large values after normalization
                        s = s+thAct     
            
                    X_svm_N[ifr,:,:] = ((xf.T - m) / s).T
            
                    meanX_fr.append(m) # frs x neurons
                    stdX_fr.append(s)       
            
                meanX_fr = np.array(meanX_fr) # frames x neurons
                stdX_fr = np.array(stdX_fr) # frames x neurons
            
                traces = X_svm_N # frames x units x trials    
        '''    
        
        
        #%% Set 10% of the trials as testing, and 90% as training dataset
        # make sure you use the same trials for x and y traces, so we set them before this loop.
        
        train0 = traces[:,:,train_data_inds] # frames x units x trials
        test0 = traces[:,:,test_data_inds]
        print(np.shape(train0) , np.shape(test0))  # frames x units x trials
    
    
        #%% Set the observations x features matrix: for each neuron, concatenate frames of all trials.
    
        a = train0 #[frames2ana] # frames x units x trials     
        # observations x features
        train = np.reshape(a.transpose(0,2,1), (a.shape[0]*a.shape[2], a.shape[1]), order='F') # (frames x trials) x units
    
    
        a = test0 #[frames2ana] # frames x units x trials  # a2 = np.transpose(a, (0,2,1)) # frames x trials x units
        # observations x features
        test = np.reshape(a.transpose(0,2,1), (a.shape[0]*a.shape[2], a.shape[1]), order='F') # (frames x trials) x units
        
        
        #%% First concatenate frames of all trials; then assign testing dataset. NOTE: we should not do this, because a testing sample can easily be a frame right after a training sample; so they will be correlated. This makes the model learn improperly becuase it will learn this correlation instead.
        '''
        a = traces[frames2ana]
        # observations x features
        tracesn = np.reshape(a.transpose(0,2,1), (a.shape[0]*a.shape[2], a.shape[1]), order='F') # (frames x trials) x units
        tracesn0 = tracesn + 0
        
    
        #%% Do PCA on the data before fitting the model
        
        # tracesn, _ = doPCA(tracesn)
        
        #%%
        train = tracesn[train_data_inds, :] 
        test = tracesn[test_data_inds, :]
        '''
        
        #%% Keep a copy before standardization
        
        if doZscore:
            train00 = train + 0
            test00 = test + 0
    
            #%% Do standardization (scaling and shifting): compute mean and std on training data; and use the same values to standardize testing data
            # z = (x-mean) / std
            
            scaler = StandardScaler()
            train = scaler.fit_transform(train00)
#            plt.plot(scaler.mean_)
#            plt.plot(scaler.scale_)            
            test = scaler.transform(test00)


        #%%
        ########
#        print(np.shape(train) , np.shape(test))  # (frames x trials) x units
    
        ######
        xy_train.append(train)
        xy_test.append(test)    
    
    
    
    x_train = xy_train[0] # (frames x trials) x units
    y_train = xy_train[1] 
    
    x_test = xy_test[0] # (frames x trials) x units
    y_test = xy_test[1]
        
    print('___________')
    print(np.shape(xy_train[0]), np.shape(xy_train[1]))    
    print(np.shape(xy_test[0]), np.shape(xy_test[1]))    
    
    
    # compare original and normalized traces.
    if isubsamp==0:
        plt.figure()
        plt.plot(np.mean(traces0, axis=(0,2)), 'b')
        plt.plot(np.mean(traces, axis=(0,2)), 'r')
        plt.xlabel('neurons')
        plt.title('frame and trial averaged')
    
    # compare x and y traces.
    plt.figure()
    plt.suptitle('neurons averaged')
    plt.subplot(211)
    plt.plot(np.mean(x_train, axis=(1)), 'b') # average across neurons
    plt.plot(np.mean(y_train, axis=(1)), 'r')
    plt.title('training')
    
    plt.subplot(212)
    plt.plot(np.mean(x_test, axis=(1)), 'b') # average across neurons
    plt.plot(np.mean(y_test, axis=(1)), 'r')
    plt.title('testing')
    plt.xlabel('observations (frames of all trials)')
    plt.subplots_adjust(hspace=1)
            

    
    #### keep a copy of x_train and y_train
    x_train0 = x_train + 0
    y_train0 = y_train + 0
    
    x_test0 = x_test + 0
    y_test0 = y_test + 0
    
    print('\nORIGINAL:')
    print('Training data:')    
    print('x:', x_train0.shape)
    print('y:', y_train0.shape)
    
    print('\nTesting data:')    
    print('x:', x_test0.shape)
    print('y:', y_test0.shape)
    
    
    
    #%% Subsample neurons
    
    if subSampNeurons:
        n_neurs_subsamp = np.min(np.concatenate((n_neurs_a1, n_neurs_a2)))
        print('Subsampling %d neurons' %n_neurs_subsamp)
        
        x_rand_neurs = rnd.permutation(x_train0.shape[1])[:n_neurs_subsamp]
        y_rand_neurs = rnd.permutation(y_train0.shape[1])[:n_neurs_subsamp]
        
        x_train = x_train0[:, x_rand_neurs]
        y_train = y_train0[:, y_rand_neurs]
        
        x_test = x_test0[:, x_rand_neurs]
        y_test = y_test0[:, y_rand_neurs]
        
        
        print('NEURON SUBSAMPLED:')
        print('Training data:')    
        print('x:', x_train.shape)
        print('y:', y_train.shape)
        
        print('\nTesting data:')    
        print('x:', x_test.shape)
        print('y:', y_test.shape)
        

    #%% Control: use a random vector as y_test, to check model performance
    
#    y_test = rnd.normal(0, 1, y_test.shape)

    # y_test = rnd.rand(y_test.shape[0], y_test.shape[1])
    # shuffle frames .. perhaps do it differently for each neuron ... but remember bc of calcium correlation, we are still not fully getting a random vector
    # y_test = y_test[rnd.permutation,:] 
    
    
    #%% Toy example: y = x*w ... check model performance
    '''
    w = rnd.rand(x_train.shape[1])
    plt.plot(w)

    y_train = x_train*w
    plt.plot(y_train)

#    scaler = StandardScaler()
#    y_train = scaler.fit_transform(y_train)
#    plt.plot(y_train)
    
    y_test = x_test*(w+.01)
    plt.plot(y_test)
    '''
    
    #%% Do PCA on the data before fitting the model
    
    if do_pca_data:
        varMax = .9 # take only the PC components that explain up to 90% of variance in the data
        
        x_train_pc, x_pca = doPCA(x_train, varexpmax=varMax)
        y_train_pc, y_pca = doPCA(y_train, varexpmax=varMax)
        
        # project testing data onto the PC space found on training data
        x_test_pc = doPCA_transform(x_test, x_pca, varexpmax=varMax, doplot=0)
        y_test_pc = doPCA_transform(y_test, y_pca, varexpmax=varMax, doplot=0)
        
        
        #%% See what pca did to training and testing data
        
        # superimpose x_train0 and x_train_pc (averaged across trials and units)
        def xo_xr_comp(x_train0, x_test0, x_train_pc, x_test_pc):
        
            # reshape x_train0 to frames x units x trials    
            ynow = np.full(x_test0.shape, np.nan)
            xo = recover_orig_traces(x_train0, ynow, train_data_inds, test_data_inds)
            print(xo.shape)
            
            # reshape x_train_pc to frames x units x trials
            ynow = np.full(x_test_pc.shape, np.nan)
            xr = recover_orig_traces(x_train_pc, ynow, train_data_inds, test_data_inds)
            print(xr.shape)
            
            # compare xtrain and its projection onto the pc spaces (averaged across trials and neurons/pcs)
            plt.figure
            plt.plot(np.nanmean(xo, axis=(1,2)), 'b')
            plt.plot(np.nanmean(xr, axis=(1,2)), 'r')
            plt.xlabel('frames since trial onset')
            plt.show()
        
        
        # compare original vs. dimension reduced data; once look at x data, then look at y data.
        #xo_xr_comp(x_train0, x_test0, x_train_pc, x_test_pc)
        #xo_xr_comp(y_train0, y_test0, y_train_pc, y_test_pc)
        
        # compare training and testing data, once for original data, then for dimension reduced data
        xo_xr_comp(x_train, x_test, y_train, y_test)
        xo_xr_comp(x_train_pc, x_test_pc, y_train_pc, y_test_pc)    
        
    
        #%% Use projections of x_train and y_train on the PC space as training dataset
        
        x_train = x_train_pc
        y_train = y_train_pc
        
        x_test = x_test_pc
        y_test = y_test_pc
    
        print('\n\nFINAL:')
        print('Training data:')    
        print('x:', x_train.shape)
        print('y:', y_train.shape)
        
        print('\nTesting data:')    
        print('x:', x_test.shape)
        print('y:', y_test.shape)
    
    
    #%% Use original x_train and y_train as training dataset
        
    #else:
    #    
    #    x_train = x_train0
    #    y_train = y_train0
    #    
    #    x_test = x_test0
    #    y_test = y_test0
    
    
    #%% Keep a copy of x_train and y_train
    
    x_trainf = x_train + 0
    y_trainf = y_train + 0
    
    x_testf = x_test + 0
    y_testf = y_test + 0

    
    