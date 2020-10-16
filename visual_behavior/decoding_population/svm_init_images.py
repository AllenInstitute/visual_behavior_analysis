#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run this script to create all_sess, in which the svm results of all sessions are concatenated. We need this to make plots for the svm (images) analysis.

SVM files are already saved (by running the svm_main_image_pbs) code.


Created on Tue Oct 13 20:48:43 2020

@author: farzaneh
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from svm_main_images_post import *


#%% ignore some warnings!
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


#%%
cre2ana = 'slc'
# session_numbers = [6] # ophys session stage corresponding to project_codes that we will load.
same_num_neuron_all_planes = 0
saveResults = 1

time_win = [0, .6] # time window relative to trial onset to quantify image signal. Over this window class accuracy traces will be averaged.
frames_svm = np.array([0,1,2,3,4,5]) # this must match the value set in svm_main_images_pbs. (it gets saved in svm_vars if svm could be run successfully on a session; but we set it here too since svm_main_images_post needs to set the values of invalid experimetns to nan, and for that it needs to know the number of frames (which is len(frames_svm))


project_codes = ['VisualBehaviorMultiscope'] # ['VisualBehaviorMultiscope', 'VisualBehaviorTask1B', 'VisualBehavior', 'VisualBehaviorMultiscope4areasx2d']


#%%
num_planes = 8
analysis_dates = [''] # ['2020507'] #set to [''] if you want to load the latest file. # the date on which the svm files were saved.


#%% Load the pickle file which includes session and metadata information.

dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
dir_svm = os.path.join(dir_server_me, 'SVM')
a = '_'.join(project_codes)
b = cre2ana  # b = '_'.join([str(session_numbers[i]) for i in range(len(session_numbers))])
filen = os.path.join(dir_svm, f'metadata_basic_{a}_{b}')
print(filen)



#%% Load the pickle file that includes the list of sessions and metadata

pkl = open(filen, 'rb')
# dict_se = pickle.load(pkl)
metadata_basic = pickle.load(pkl)

list_all_sessions_valid = metadata_basic[metadata_basic['valid']]['session_id'].unique()
print(f'number of {cre2ana} sessions: {list_all_sessions_valid.shape[0]}')
# list_all_sessions_valid = dict_se['list_all_sessions_valid']
# list_all_experiments_valid = dict_se['list_all_experiments_valid']
# print(f'number of sessions: {len(list_all_sessions_valid)}')



#%% Set the columns in dataframe all_sess

cols0 = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_trials', 'n_neurons', 'frame_dur', 'meanX_allFrs', 'stdX_allFrs', 'av_train_data', 'av_test_data', 'av_test_shfl', 'av_test_chance', 'sd_train_data', 'sd_test_data', 'sd_test_shfl', 'sd_test_chance', 'peak_amp_trainTestShflChance'])
if same_num_neuron_all_planes:
    cols = np.concatenate((cols0, ['population_sizes_to_try'])) 
else:
    cols = np.concatenate((cols0, ['av_w_data', 'av_b_data']))
    
    

#%% Set the current session to be analyzed, and its metada

all_sess = pd.DataFrame([], columns=cols)

for isess in np.arange(0, len(list_all_sessions_valid)):  # isess = 0
    session_id = int(list_all_sessions_valid[isess])
    data_list = metadata_basic[metadata_basic['session_id'].values==session_id]

    this_sess = svm_main_images_post(session_id, data_list, dir_svm, frames_svm, same_num_neuron_all_planes, time_win, cols, analysis_dates, doPlots=0)

    all_sess = all_sess.append(this_sess) 
    
print(len(all_sess))    
all_sess

    
    
##################################################################    
##################################################################
#%% Save all_sess

# set this so you know what input vars you ran the script with
cols = np.array(['time_win', 'same_num_neuron_all_planes', 'project_codes', 'session_numbers'])
input_vars = pd.DataFrame([], columns=cols)
input_vars.at[0, cols] =  time_win, same_num_neuron_all_planes, project_codes, session_numbers

# set the name of the h5 file for saving data
svmn = 'svm_images'
now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

if same_num_neuron_all_planes:
    name = 'all_sess_%s_sameNumNeuronsAllPlanes' %(svmn) 
else:
    name = 'all_sess_%s' %(svmn) 

a = '_'.join(project_codes)
b = '_'.join([str(session_numbers[i]) for i in range(len(session_numbers))])
name = f'{name}_{a}_{b}_{now}'
# print(name)

    
if saveResults:
    print('\n\nSaving all_sess.h5 file')
    allSessName = os.path.join(dir_svm, name + '.h5') # os.path.join(d, svmn+os.path.basename(pnevFileName))
    print(allSessName)

    # save to a h5 file                    
    all_sess.to_hdf(allSessName, key='all_sess', mode='w')    
    
    # save input_vars to the h5 file
    input_vars.to_hdf(allSessName, key='input_vars', mode='a')
    
                  
        
        
        
#%% rename svm files to add frames_svm to the file name
'''
name = '(.*)_%s_2020.' %(svmn)
svmName ,_ = all_sess_set_h5_fileName(name, dir_svm, all_files=1)
            
for i in range(len(svmName)):
    
    oldname = svmName[i]
    b = os.path.basename(oldname)
    ind = b.find('_2020')
    newname = os.path.join(dir_svm, f'{b[:ind]}_frames0to5{b[ind:]}')
    print(newname)
    print(oldname)
    
    os.rename(oldname, newname)
'''    