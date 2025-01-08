# This is the first script that needs to be run to either:
# fit the model or save all-neuron vars after model fitting.

# First run omissions_traces_peaks_init.py to get the df/f traces.
#(set doCorrs = -1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:17:12 2019
@author: farzaneh
"""

# remember you used isess = 152 as the initial prototype session to develop your DNN codes!

fit_dnn_model = 0 # if 1, model the activity of each neuron in traces_y0_evs using the population activity in traces_x0_evs  # if 0, model is fit, load and save vars for all neurons

import numpy as np
x_planes = [6,7] #np.arange(0,8) #[0] # num_planes
y_planes = np.arange(0,8) #[0,1,4,5] #[2] # num_planes


#%%
%load_ext autoreload
%run -i 'omissions_traces_peaks_init.py'

from def_funs import *
from keras_funs import *
from keras_DNN_planes_model_fit import *
from keras_DNN_planes_model_post_saveVars import *

get_ipython().magic(u'matplotlib inline')   
#get_ipython().magic(u'matplotlib qt')


#%% Set vars for the model
 
kfold = 45
len_win = 20 #10 # length of the consecutive frames that will be included in the feature space for each neuron        
len_ne = len_win #20 # number of frames before and after each event that are taken to create traces_events.
th_ag = 10 #8 # threshold to apply on erfc (output of evaluate_components) to find events on the trace; the higher the more strict on what we call an event.

th_train = 10 # minimum number of training and testing samples to run the model
th_test = 10 
n_beg_end_rmv = 60 # number of frames to exclude from the begining and end of the session (because we usually see big rises in the trace, perhaps due to df/f computation?)

analysis_name = 'DNN_allTrace_timeWindow_planes'

plotPlots = 0
softNorm = 0
dir_ae = '/tmp/autoencoder'
if not os.path.exists(dir_ae):
    os.makedirs(dir_ae)


#%% Get the traces

# this_sess
num_depth = int(num_planes/2)
inds_lm = np.arange(num_depth) # we can get these from the following vars: distinct_areas and i_areas[range(num_planes)]
inds_v1 = np.arange(num_depth, num_planes)
#            print(inds_lm); print(inds_v1); print(np.shape(this_sess))

# area1: LM
# area 2: V1
local_fluo_allOmitt_a1 = this_sess.iloc[inds_lm]['local_fluo_allOmitt'].values # 4
local_fluo_allOmitt_a2 = this_sess.iloc[inds_v1]['local_fluo_allOmitt'].values # 4

valid_a1 = this_sess.iloc[inds_lm]['valid'].values # 4
valid_a2 = this_sess.iloc[inds_v1]['valid'].values # 4

nfrs = local_fluo_allOmitt_a1[0].shape[0]
ntrs = local_fluo_allOmitt_a1[0].shape[2]
n_neurs_a1 = [local_fluo_allOmitt_a1[iexp].shape[1] for iexp in range(len(local_fluo_allOmitt_a1))] # number of neurons for each depth of area 1
n_neurs_a2 = [local_fluo_allOmitt_a2[iexp].shape[1] for iexp in range(len(local_fluo_allOmitt_a2))] # number of neurons for each depth of area 2

print(n_neurs_a1, ': LM #neurons')
print(n_neurs_a2, ': V1 #neurons')
print('Number of trials: %d' %ntrs)



#%% Main variables and how they are driven:    
'''
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

'''

#%%

#x_ind = x_planes[0] 
#y_ind = y_planes[0] 
#
#print('Analyzing plane %d prediction of plane %d' %(x_ind, y_ind))

# concantenate neurons from a number of planes to form the x trace.
# eg. train a model so neurons in 7 planes predict the activity in 1 plane
#y_ind = 5  # recorded simultaneously with plane 7. exclude 7 from x_planes, bc of potential cross talk.  
#x_planes = [1] #[0,1,2,3] #, 5,6] #np.arange(1,8) # 1:


#%%

for x_ind in x_planes: #x_planes: #[1]: #np.arange(0,num_depth): # x_ind = 0
    for y_ind in y_planes: #[x_ind] #y_planes: #[4,5,6]: #np.arange(4,8): # y_ind = 0
               
#         if y_ind==x_ind: # remove this once you saved vars ... now you dont have within plane data
        gc.collect()

        this_fold = 'isess%d_plane_%dto%d' %(isess, x_ind, y_ind) 
        dir_now = os.path.join(dir_server_me, analysis_name, this_fold)

        if not os.path.exists(dir_now):
            os.makedirs(dir_now)


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


        #%%
        if fit_dnn_model==1:

            #%% Model the activity of each neuron in traces_y0_evs using the population activity in traces_x0_evs

            tooFewSamps_x_y_neur = [] # keep record of x and y planes, also the neuron_y index which had too few training/testing samples to run the model

            for neuron_y in np.arange(0, traces_y0.shape[0]): # neuron_y = 5

                print('\nn======================== Plane %d to %d; modeling neuron %d/%d in y trace ======================== ' %(x_ind, y_ind, neuron_y, traces_y0.shape[0]))

                model_ae = keras_DNN_planes_model_fit(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, kfold, x_ind, y_ind, th_train, th_test, plotPlots=0)


        else: # model is already fit, load and save vars for all neurons

            #%% Set loss, r_square, y_prediction variables for each neuron and save them

            keras_DNN_planes_model_post_saveVars(traces_x0, traces_y0, traces_y0_evs, inds_final_all, x_ind, y_ind, len_win, kfold, dir_now)
