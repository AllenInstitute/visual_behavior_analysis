#!/usr/bin/env python
# coding: utf-8

# #### Using master branch on AllenSDK and master branch on VBA (current as of 5/18/20)

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})


#%%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


#### get experiments to analyze

#%% The VBA data_access module provides useful functions for identifying and loading experiments to analyze. 

import visual_behavior.data_access.loading as loading


#%% The get_filtered_ophys_experiment_table() function returns a table describing passing ophys experiments from relevant project codes 

experiments_table = loading.get_filtered_ophys_experiment_table() # include_failed_data=False
experiments_table.keys()
experiments_table.head()
np.shape(experiments_table)

# aribtrary_index = 699
# ophys_experiment_id = experiments_table.index.values[aribtrary_index]
# experiments_table.iloc[aribtrary_index]

'''
experiments_table.index.name
'ophys_experiment_id'

list(experiments_table.keys())
['ophys_session_id',
 'behavior_session_id',
 'container_id',
 'project_code',
 'container_workflow_state',
 'experiment_workflow_state',
 'session_name',
 'session_type',
 'equipment_name',
 'date_of_acquisition',
 'isi_experiment_id',
 'specimen_id',
 'sex',
 'age_in_days',
 'full_genotype',
 'reporter_line',
 'driver_line',
 'imaging_depth',
 'targeted_structure',
 'published_at',
 'super_container_id',
 'cre_line',
 'session_tags',
 'failure_tags',
 'exposure_number',
 'model_outputs_available',
 'location']
'''


#%% Get some experiment
aribtrary_index = 699
ophys_experiment_id = experiments_table.index.values[aribtrary_index]
experiments_table.iloc[aribtrary_index]
# ophys_experiment_id = experiments_table.index[0]
# experiments_table.iloc[experiments_table.index == ophys_experiment_id]




#%% Get SDK dataset through VBA loading function
# The get_ophys_dataset function in data_access.loading returns an AllenSDK session object for a single imaging plane.

# help(loading.get_ophys_dataset)
# help(loading.BehaviorOphysDataset)

# this function gets an SDK session object then does a bunch of reformatting to fix things
dataset = loading.get_ophys_dataset(ophys_experiment_id, include_invalid_rois=False)
# attrs = dataset.__dir__()
'''   
 'analysis_folder',
 'analysis_dir',
 'cell_specimen_table',
 'cell_indices',
 'cell_specimen_ids',
 'roi_masks',
 'corrected_fluorescence_traces',
 'dff_traces',
 'timestamps',
 'ophys_timestamps',
 'metadata',
 'metadata_string',
 'licks',
 'rewards',
 'running_speed',
 'stimulus_presentations',
 'extended_stimulus_presentations',
 'trials',
 'get_cell_specimen_id_for_cell_index',
 'get_cell_index_for_cell_specimen_id',
 'from_lims',
 'from_nwb_path',
 'ophys_experiment_id',
 'max_projection',
 'stimulus_timestamps',
 'running_data_df',
 'stimulus_templates',
 'task_parameters',
 'average_projection',
 'motion_correction',
 'segmentation_mask_image',
 'eye_tracking',
 'cache_clear',
 '''
# the dataset object has attributes for relevant datastreams including: max_projection, dff_traces, running_speed, stimulus_presentations, and eye_tracking 
dataset.dff_traces.head()
dataset.ophys_timestamps

dataset.dff_traces.shape # neurons

# max projection
plt.imshow(dataset.max_projection, cmap='gray')

# dff traces
cell_specimen_ids = dataset.cell_specimen_ids
cell_specimen_id = cell_specimen_ids[5]
plt.plot(dataset.ophys_timestamps, dataset.dff_traces.loc[cell_specimen_id, 'dff'])
plt.xlabel('time (seconds)')
plt.ylabel('dF/F')

# running speed
plt.plot(dataset.stimulus_timestamps, dataset.running_speed['speed'])
plt.xlabel('time (seconds)')
plt.ylabel('run speed (cm/s)')

# eye tracking
plt.plot(dataset.eye_tracking.time, dataset.eye_tracking.pupil_width)
plt.xlabel('time (seconds)')
plt.ylabel('pupil width (pixels?)')

# basic stimulus information
dataset.stimulus_presentations.head(10)
np.shape(dataset.stimulus_presentations)

# stimulus information with many extra columns
dataset.extended_stimulus_presentations.head(10)

# behavioral trial information
dataset.trials.head(10)
np.shape(dataset.trials)

# Trial definition:
# trials means behavioral trials, as in the time of stimulus change
# also catch trials as defined by the behavior software
# so non-changes pulled from the same distribution as change times



################################################################################
#%% ResponseAnalysis class provides access to time aligned cell responses for trials, stimuli, and omissions
# VBA also has useful functionality for creating data frames with cell traces aligned to the time of stimulus presentations, omissions, or behavioral trials. 
################################################################################
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

help(ResponseAnalysis)

analysis = ResponseAnalysis(dataset) 


#%%
#### Get cell traces for all stimulus presentations 

stim_response_df = analysis.get_response_df(df_name='stimulus_response_df')
stim_response_df.keys() 
stim_response_df.head()
np.shape(stim_response_df)

'''
trace = stim_response_df[stim_response_df.cell_specimen_id==cell_specimen_id].trace.values
trace.shape
(4802,)
np.shape(dataset.stimulus_presentations)
(4803, 10)
sum(dataset.stimulus_presentations['omitted']==False)
4601
'''

##### plot single trial response for some cell

cell_specimen_id = stim_response_df.cell_specimen_id.unique()[0]

trace = stim_response_df[stim_response_df.cell_specimen_id==cell_specimen_id].trace.values[0]
times = stim_response_df[stim_response_df.cell_specimen_id==cell_specimen_id].trace_timestamps.values[0]

plt.plot(times, trace)
plt.xlabel('time after stim onset (s)')
plt.ylabel('dF/F')

##### plot trial averaged trace for some image

image_name = stim_response_df.image_name.unique()[6]
mean_trace = stim_response_df[(stim_response_df.cell_specimen_id==cell_specimen_id)&
                         (stim_response_df.image_name==image_name)].trace.mean()

plt.plot(times, mean_trace)
plt.xlabel('time after stim onset (sec)')
plt.ylabel('dF/F')


##### responsiveness

# Response dataframes include other useful columns including  p_value_gray_screen  which compares the mean response for each trial to a shuffled distribution of values from the 5 min gray screen periods at the beginning and end of the session.

cell_data = stim_response_df[(stim_response_df.cell_specimen_id==cell_specimen_id)]
fraction_responsive = cell_data.p_value_gray_screen.mean()
print('this cell had a significant response for',fraction_responsive,'of all image presentations')

#  p_value_omission  compares the mean response for each trial to a shuffled distribution of all omission responses
#  p_value_stimulus  compares the mean response for each trial to a shuffled distribution of all other stimulus responses




#%%
#### Get cell responses around change times for behavioral trials

trials_response_df = analysis.get_response_df(df_name='trials_response_df')

trials_response_df.keys()
trials_response_df.head()
np.shape(trials_response_df)

'''
trials_response_df[trials_response_df.cell_specimen_id==cell_specimen_id].trace.values.shape
(405,)
np.shape(dataset.trials)
(406, 22)
sum(dataset.stimulus_presentations['change'])
356
'''

# popuation average response to image change
times = trials_response_df.trace_timestamps.values[0]
plt.plot(times, trials_response_df.trace.mean())
plt.title('population average change response')
plt.xlabel('time after change (s)')
plt.ylabel('dF/F')





#### Get omission triggered responses 

omission_response_df = analysis.get_response_df(df_name='omission_response_df')

omission_response_df.keys()
omission_response_df.head()
np.shape(omission_response_df)

# popuation average response
times = omission_response_df.trace_timestamps.values[0]
plt.plot(times, omission_response_df.trace.mean())
plt.title('population average omission response')
plt.xlabel('time after change (s)')
plt.ylabel('dF/F')


#### Get running behavior for omissions 

run_speed_df = analysis.get_response_df(df_name='omission_run_speed_df')

run_speed_df.head()
np.shape(run_speed_df)

# running speed averaged across all omissions
times = run_speed_df.trace_timestamps.values[0]
plt.plot(times, run_speed_df.trace.mean())
plt.title('average omission triggered running behavior')
plt.xlabel('time after omission (s)')
plt.ylabel('run speed (cm/s)')





### get trial averaged response dataframe for some set of conditions 
# VBA  response_analysis.utilities  has a function for averaging across trials for a given set of conditions: the  get_mean_df()  function

import visual_behavior.ophys.response_analysis.utilities as utilities

help(utilities.get_mean_df) # needs documentation...

#
conditions = ['cell_specimen_id', 'image_name'] # conditions to groupby before averaging
mean_df = utilities.get_mean_df(stim_response_df, conditions=conditions, flashes=True)

mean_df.head()


# The resulting dataframe includes useful columns such as  pref_stim  which indicates the stimulus that evoked the maximal response across conditions for that cell, or   fraction_significant_p_value_gray_screen  which tells you the fraction of trials for the given condition that had a significant p_value compared to the gray screen periods, or  mean_responses  which has an array of the mean response value for all trials of a given condition (useful for things like computing variability or noise correlations)

#### plot mean trace for a cells preferred stimulus



# demo new plotting functions 



