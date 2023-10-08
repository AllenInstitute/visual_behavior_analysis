###############################################
###############################################
################ main codes ###################
###############################################
###############################################


#%% Set traces_x0 and traces_y0; these are traces read from this_sess.iloc[ind]['local_fluo_traces']; after removing n_beg_end_rmv frames from their begining and end. 

[traces_x0, traces_y0] = set_traces_x0_y0(this_sess, x_ind, y_ind, n_beg_end_rmv, plotPlots=0) 


#%% Create traces_events
# active parts of Y traces are found. 
# Y traces are are made by concatenating the active parts.
# X traces are made from those same time points (during which Y is active).

## set traces_evs (for the y trace), ie traces that are made by extracting the active parts of the input trace
# the idea is that to help with learning the events, take parts of the trace that have events        
#        len_ne = len_win #20 # number of frames before and after each event that are taken to create traces_events.
#        th_ag = 10 #8 # the higher the more strict on what we call an event.                
[traces_y0_evs, inds_final_all] = set_traces_evs(traces_y0, th_ag, len_ne, doPlots=0)


#%% Set traing and testing datasets (x and y) for the model
# extract the same part of the traces_y0_evs for a given y neuron from traces_x

set_train_test_inds = 1 # set to a list: [train_inds, test_inds] if training and testing inds are already set (do this after the model data are saved and loaded)
[x_train, x_test, y_train, y_test, train_data_inds, test_data_inds] = set_x_y_train_test(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, kfold, set_train_test_inds, plotPlots=0)




#%% Set traing and testing datasets (x and y) for the model

def set_x_y_train_test(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, kfold, set_train_test_inds=1, plotPlots=0):
    
    ######%% Set final (z scored) input matrices (traces_evs for x population activity and y neuron, ie traces_x0_evs and traces_y0_evs_thisNeuron) to traces_neurons_times_features, which sets xx0 and yy0.    
    [traces_x0_evs, traces_y0_evs_thisNeuron, scaler_x, scaler_y] = set_traces_x0_y0_evs_final(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, plotPlots=0, setcorrs=0)
    
    
    ######%% From traces_events create the input matrix for the model: samples x neurons x time_window
    # set samples so time and neurons are both in the feature space: have 1sec of each neuron activity is in the feature space, ie concatenate 10sec of all neurons to get the features.
    # the idea is that a window in x traces predicts the middle time point of the y traces    
    [xx0, yy0] = traces_neurons_times_features(traces_x0_evs, traces_y0_evs_thisNeuron, len_win, plotfigs=plotPlots)    
    num_samps = xx0.shape[0] #traces_x0_evs.shape[1] - len_win

    
    ######%% Assign testing and training indeces; Note: these indeces belong to xx0 and yy0.                
    if type(set_train_test_inds)==int: # set the training and testing indeces on xx0 and yy0
        doPlots = 0 # set to 1 to see a nice plot of the trace, with training and testing indeces marked.
        [train_data_inds, test_data_inds] = set_train_test_data_inds_wind(num_samps, len_win, kfold, xx0, yy0, doPlots=doPlots)
        
    else:
        train_data_inds = set_train_test_inds[0]
        test_data_inds = set_train_test_inds[1]
                
    
    ######%% Now assign kfold% random samples as testing dataset
    
    x_train0 = xx0[train_data_inds]
    x_test0 = xx0[test_data_inds]
    
    y_train0 = yy0[train_data_inds]
    y_test0 = yy0[test_data_inds]
    
    print('\nx, train, test:')
    print(np.shape(x_train0) , np.shape(x_test0))  # samples x neurons x 10 
    print('y, train, test:')
    print(np.shape(y_train0) , np.shape(y_test0))  # samples x neurons
      




###############################################
###############################################
#### Main variables and how they are driven ###
###############################################
###############################################

traces_y = this_sess.iloc[xy_ind]['local_fluo_traces']
traces_y0 = traces_y[:, n_beg_end_rmv:-n_beg_end_rmv]

# [traces_y0_evs, inds_final_all] = set_traces_evs(traces_y0, th_ag, len_ne, doPlots=1)
traces_y0_evs = traces_y0[iu][inds_final] 
# note: the following two have the same size:
# inds_final_all[neuron_y] and 
# traces_y0_evs[neuron_y]

traces_y0_evs_thisNeuron = traces_y0_evs[neuron_y][np.newaxis,:] # 1 x active_frames_of_neuronY


# [xx0, yy0] = traces_neurons_times_features(traces_x0_evs, traces_y0_evs_thisNeuron, len_win, plotfigs=plotPlots)    
# xx0.shape[0] == traces_x0_evs.shape[1] - len_win
# yy_alltrace = z_scored_traces_y0[t_y: le-t_y]
# xx_alltrace, yy_alltrace are made from z scored inputs; so yy_alltrace is z scored traces_y0
t_y = int(len_win/2)
le = traces_y0_evs_thisNeuron.shape[1]
yy0 = traces_y0_evs_thisNeuron[:, t_y: le-t_y].T # frames x neurons
# note the following two have the same size:
# inds_final_all[neuron_y].shape[0] and
# yy0.shape[0] + len_win


# [train_data_inds, test_data_inds] = set_train_test_data_inds_wind(num_samps, len_win, kfold, xx0, yy0, doPlots=1)

        
    