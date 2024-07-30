"""
Load all_sess_dff_stim which has the dff traces as well as stimulus information and some metadata for all mesoscope experiments.
all_sess_dff_stim is set in all_sess_dff_stim.py

This is made for Akul Gupta, for the RNN analysis.

Created on Tue Jul 7 17:06:25 2020
@author: farzaneh
"""

#%% Load the pkl file on the server that contains all_sess_df_stim      
    
import pickle
import os
import numpy as np


dir_now = os.path.join(os.sep, 'allen', 'programs', 'braintv', 'workgroups', 'nc-ophys', 'Farzaneh')


#%% original dff                       
name = 'all_sess_dff_stim_20200707_160837'


#%% de-cross-talked dff
name = 'all_sess_omit_traces_peaks_allTraces_20200721_225720'
dir_now = os.path.join(os.sep, dir_now, 'omit_traces_peaks')
# dff: 'local_fluo_traces'
# dff_times: 'local_time_traces'
# 'roi_ids'




allSessName = os.path.join(dir_now , name + '.pkl') # save to a pickle file
print(allSessName)

pkl = open(allSessName, 'rb')
all_sess_now = pickle.load(pkl) # all_sess_dff_stim
print(all_sess_now.keys())


#%% Remove sessions with less than 8 experiments

sess_ids = np.unique(all_sess_now['session_id'].values)

# set session_ids that have <8 experiments
sess_ids_missing_exp = []
for sess_id in sess_ids:
    this_sess = all_sess_now.iloc[all_sess_now['session_id'].values==sess_id]
    
    if len(this_sess) < 8:
        sess_ids_missing_exp.append(sess_id)
        
mask = np.full((all_sess_now.shape[0]), True)
exp_missing = np.in1d(all_sess_now['session_id'].values, sess_ids_missing_exp)
mask[exp_missing] = False

all_sess_now2 = all_sess_now[mask]

print(all_sess_now.shape)
print(all_sess_now2.shape)


#%% Finally, reset all_sess_now

all_sess_now = all_sess_now2 



#%% Example experiments:

all_sess_now.iloc[:2]

# Some important columns of all_sess_now:
#
# cre (cell type: slc, sst, vip)
# mouse_id (each mouse has a unique id)
# session_id (each day is 1 session)
# experiment_id (each depth is 1 experiment; 8 experiments per session)
# stage (familiar active (1A,3A), familiar passive (2A); novel active (4B,6B); passive active (5B))
# area (visl: LM ; visp: V1)
# depth (in micrometer)
# dff (fluorescence trace (DF/F))
# dff_times (timestamps of the fluorescence trace) 
# stimulus_info: pandas dataframe (each row is for one image (or omission) presentation); it contains the following important information:
# (note stimulus_info is the same for all experiment_ids in 1 session (because the experiment_ids are just the 8 planes))
#       image_index (8 is omitted; the rest are the indices of the 8 images shown to mouse); 
#       start_time and stop_time: onset and offset of image presentation


#%% 

import matplotlib.pyplot as plt
all_sess_now.iloc[0]['dff'].shape

plt.plot(all_sess_now.iloc[0]['dff'][2])
plt.xlim([2.97e4, 2.98e4])
plt.ylim([-5,2]);


plt.plot(all_sess_now.iloc[2]['dff'][10])

all_sess_now.iloc[2]['cell_specimen_ids'][10]


## 
a = all_sess_now.iloc[38:]['dff'].values
aa = np.vstack([a[i][:,:40000] for i in range(len(a))])
aa.shape

plt.plot(np.mean(aa, axis=0))


all_sess_now.iloc[41:]

np.unique(all_sess_now.iloc[:41]['experiment_id'])