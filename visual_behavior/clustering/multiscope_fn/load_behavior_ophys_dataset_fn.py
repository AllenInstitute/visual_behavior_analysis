#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using AllenSDK to set allSess which has the omission-aligned traces as well as some metadata details.

Created on Tue May 26 11:17:25 2020
@author: farzaneh
"""

saveResults = 1 # save allsess file (in a pkl file named "all_sess_omit_traces_peaks_allTraces_sdk...")
bl_percentile = 10


#%%
from def_funs import *
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
from visual_behavior.ophys.response_analysis.response_processing import get_omission_response_xr
# help(ResponseAnalysis)

# omission_response_xr = get_omission_response_xr(dataset, use_events=False, frame_rate=None)


#%% The VBA data_access module provides useful functions for identifying and loading experiments to analyze. 

import visual_behavior.data_access.loading as loading


#%% The get_filtered_ophys_experiment_table() function returns a table describing passing ophys experiments from relevant project codes 

experiments_table = loading.get_filtered_ophys_experiment_table() # include_failed_data=False
# experiments_table.keys()
# experiments_table.head()
np.shape(experiments_table)


#%% Set some essential parameters

# get omission_response_df for an arbitrary experiment, so we can set samps_bef etc below
ophys_experiment_id = experiments_table.index[0]     # experiments_table_2an.iloc[experiments_table_2an.index == ophys_experiment_id]
dataset = loading.get_ophys_dataset(ophys_experiment_id, include_invalid_rois=False)
analysis = ResponseAnalysis(dataset) 
omission_response_df = analysis.get_response_df(df_name='omission_response_df')

n_frames = len(omission_response_df['trace'].iloc[0])
time_trace = omission_response_df['trace_timestamps'].iloc[0]
samps_bef = np.argwhere(time_trace==0)[0,0]
samps_aft = n_frames - samps_bef
print(samps_bef, samps_aft)

'''
bl_index_pre_omit = np.arange(0, samps_bef) # we will need it for response amplitude quantification.

# frame_dur = np.mean(np.diff(omission_response_df['trace_timestamps'].iloc[0]))

flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, flashes_win_trace_index_unq, grays_win_trace_index_unq = \
    flash_gray_onset_relOmit(samps_bef, samps_aft, frame_dur)

flash_ind = abs(flashes_win_trace_index_unq[flashes_win_trace_index_unq<0][-1].astype(int))

# take one image before and one image after omission
samps_bef_now = samps_bef - flash_ind # image before omission
samps_aft_now = samps_bef + flash_ind*2 # omission and the image after
'''

#%% Get VisualBehaviorMultiscope experiments

a = experiments_table['project_code'].values
all_project_codes = np.unique(a) # ['VisualBehavior', 'VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d', 'VisualBehaviorTask1B']
experiments_table_2an = experiments_table[a=='VisualBehaviorMultiscope']
experiments_table_2an.shape
# add experiment_id to the columns
experiments_table_2an.insert(loc=1, column='experiment_id', value=experiments_table_2an.index.values)


#%% For each experiment set trial-average omission-aligned traces.

cols = ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'frame_dur', 'stim_response_df', 'image_names_inds'] 
# cols = ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'frame_dur', 'omission_response_xr', 'trials_response_df', 'image_names_inds', 'run_speed_times', 'run_speed_traces'] 
all_sess_allN_allO = pd.DataFrame([], columns=cols) # size: 8*len(num_sessions)

errors_iexp = []
errors_log = []

for iexp in np.arange(0, experiments_table_2an.shape[0]): # iexp = 0 
        
    experiment_id = experiments_table_2an.index[iexp]     # experiments_table_2an.iloc[experiments_table_2an.index == ophys_experiment_id]
    session_id = experiments_table_2an.iloc[iexp]['ophys_session_id']
#     session_id = experiments_table[experiments_table.index.values==ophys_experiment_id].iloc[0]['ophys_session_id']

    print(f'\n\n------------ Setting vars for experiment_id {experiment_id} , session_id {session_id} , ({iexp} / {experiments_table_2an.shape[0]}) ------------\n\n')
    
    cre = experiments_table_2an.iloc[iexp]['cre_line']
    date = experiments_table_2an.iloc[iexp]['date_of_acquisition'][:10]
    stage = experiments_table_2an.iloc[iexp]['session_type']
    area = experiments_table_2an.iloc[iexp]['targeted_structure']
    depth = int(experiments_table_2an.iloc[iexp]['imaging_depth'])
    
    
    session_name = str(experiments_table_2an.iloc[iexp]['session_name'])
    uind = [m.start() for m in re.finditer('_', session_name)]
    mid = session_name[uind[0]+1 : uind[1]]
    if len(mid)<6: # a different nomenclature for session_name was used for session_id: 935559843
        mouse_id = int(session_name[uind[-1]+1 :])
    else:
        mouse_id = int(mid)
        
        
    try:
        # get SDK dataset through VBA loading function # this function gets an SDK session object then does a bunch of reformatting to fix things
        dataset = loading.get_ophys_dataset(experiment_id, include_invalid_rois=False)
        frame_dur = np.mean(np.diff(dataset.ophys_timestamps))
        n_neurons = dataset.dff_traces.shape[0]

        '''
        if n_neurons > 0: # get 3d traces: frames x trials x units
            omission_response_xr = get_omission_response_xr(dataset, use_events=False, frame_rate=None)
        #     n_omissions = omission_response_xr['trial_id'].values.shape[0] 
        #     n_neurons = omission_response_xr['trace_id'].values.shape[0]
        #     time_trace = omission_response_xr['eventlocked_timestamps'].values
        #     traces_ftu = omission_response_xr['eventlocked_traces'].values # frames x trials x neurons
            # the cell_ids defined below is same as cell_ids.astype(int)[:n_neurons] when cell_ids is defined out of omission_response_df
        #     cell_ids = omission_response_xr['trace_id'].values
        else:
            omission_response_xr = np.nan
        '''

            
        #%% ResponseAnalysis class provides access to time aligned cell responses for trials, stimuli, and omissions
        # VBA also has useful functionality for creating data frames with cell traces aligned to the time of stimulus presentations, omissions, or behavioral trials. 

        analysis = ResponseAnalysis(dataset)  
        
        
        #%% Keep image names
        
        stim_response_df = analysis.get_response_df(df_name='stimulus_response_df')
#         np.shape(stim_response_df) # (neurons x flashes) x 19 (columns)
    
        cell0 = stim_response_df['cell_specimen_id'].values[0] # get flashes for one of the cells
        s = stim_response_df[stim_response_df['cell_specimen_id'].values==cell0]
        image_names_inds = s[['stimulus_presentations_id', 'image_index', 'image_name']]


        #%% Get image-change aligned traces
        '''
        trials_response_df = analysis.get_response_df(df_name='trials_response_df')
        '''
        
        #%% Get running speed for omission-aligned traces 
        '''
        run_speed_df = analysis.get_response_df(df_name='omission_run_speed_df')
#         run_speed_df.head()
#         np.shape(run_speed_df)

        # running speed averaged across all omissions
        run_speed_times = run_speed_df['trace_timestamps'].values[0] # times (60Hz)
        run_speed_traces = np.vstack(run_speed_df['trace']).T # times x omissions (remember the last omission is missing due to some issues in responseAnalysis)
        
#         plt.plot(run_speed_times, run_speed_df.trace.mean())
#         plt.title('average omission triggered running behavior')
#         plt.xlabel('time after omission (s)')
#         plt.ylabel('run speed (cm/s)')
        '''
    
        
        #%% Get pupil area for omission-aligned traces 
        # this part didnt work; wrote marina/doug about it:
        '''
        pupil_area_df = analysis.get_response_df(df_name='omission_pupil_area_df')
        
        pupil_area_df.head()
        np.shape(pupil_area_df)

        # pupil area averaged across all omissions
        pupil_times = pupil_area_df['trace_timestamps'].values[0] # times (60Hz)
        pupil_traces = np.vstack(pupil_area_df['trace']).T # times x neurons
        
#         plt.plot(run_times, run_speed_df.trace.mean())
#         plt.title('average omission triggered running behavior')
#         plt.xlabel('time after omission (s)')
#         plt.ylabel('run speed (cm/s)')
        '''
        
    
        '''
        # get omission triggered responses 
        omission_response_df = analysis.get_response_df(df_name='omission_response_df')

        cell_ids = omission_response_df['cell_specimen_id'].values.astype(float)
        if np.isnan(cell_ids).any(): # cell matching has not been run, just copy cell roi ids to cell specimen ids
            print(f'\n\n Warning! experiment {experiment_id} does not have cell_specimen_id; using cell_roi_id instead....\n\n')
            dataset.dff_traces.index = dataset.dff_traces.cell_roi_id
            analysis = ResponseAnalysis(dataset) 
            omission_response_df = analysis.get_response_df(df_name='omission_response_df')


        frame_dur = np.mean(np.diff(omission_response_df['trace_timestamps'].iloc[iexp]))
    #     n_neurons = len(np.unique(omission_response_df['cell_specimen_id']))
        n_omissions = int(omission_response_df.shape[0] / n_neurons)
    #     sum(dataset.stimulus_presentations['omitted']==True)
    #     sum(stim_response_df['omitted']) / n_neurons

    #     a = omission_response_df[['cell_specimen_id', 'trace', 'trace_timestamps']]
        traces_allN_allO = np.vstack(omission_response_df['trace'].values) # all neurons and all omissions x frames (107)  # all neurons, omission 1, then all neurons omission 2, etc.... all neuron traces for all omissions are concatenated.
        '''

        all_sess_allN_allO.at[iexp, ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', \
                                     'frame_dur', 'stim_response_df', 'image_names_inds']] = \
            session_id ,  experiment_id ,  mouse_id ,  date ,  cre ,  stage ,  area ,  depth ,  frame_dur ,  stim_response_df, image_names_inds
        
#         all_sess_allN_allO.at[iexp, ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', \
#                                      'frame_dur', 'omission_response_xr', 'trials_response_df', 'image_names_inds', 'run_speed_times', 'run_speed_traces']] = \
#             session_id ,  experiment_id ,  mouse_id ,  date ,  cre ,  stage ,  area ,  depth ,  frame_dur ,  omission_response_xr, trials_response_df, image_names_inds, run_speed_times, run_speed_traces




    #     all_sess_allN_allO.at[iexp, ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', \
    #                                  'n_omissions', 'n_neurons', 'frame_dur', 'cell_specimen_id', 'traces_allN_allO']] = \
    #         session_id ,  experiment_id ,  mouse_id ,  date ,  cre ,  stage ,  area ,  depth ,  n_omissions ,  n_neurons ,  frame_dur ,  cell_ids , traces_allN_allO

    except Exception as e:
        print(e)
        errors_iexp.append(iexp)
        errors_log.append(e)

        

all_sess_allN_allO.at[all_sess_allN_allO['mouse_id'].values==6, 'mouse_id'] = 453988

    
#%% Save the results

if saveResults:

    # set pandas dataframe input_vars, to save it below
    # set this so you know what vars the script was run with
    cols = np.array(['bl_percentile', 'samps_bef', 'samps_aft'])
    input_vars = pd.DataFrame([], columns=cols)
    input_vars.at[0, cols] = bl_percentile, samps_bef, samps_aft


    analysis_name = 'omit_traces_peaks'
    dir_now = os.path.join(dir_server_me, analysis_name)
    if not os.path.exists(dir_now): # create a folder: omit_traces_peaks
        os.makedirs(dir_now)

    namespec = '_allTraces_sdk'
    now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")


    print('\nSaving allsess file')

    name = 'all_sess_%s%s_%s' % (analysis_name, namespec, now)            
    allSessName = os.path.join(dir_now , name + '.pkl') # save to a pickle file
    print(allSessName)

    
    #### save the file ####
    f = open(allSessName, 'wb')
    pickle.dump(all_sess_allN_allO, f)    
    pickle.dump(input_vars, f)
    pickle.dump(errors_iexp, f)
    pickle.dump(errors_log, f)
    
    f.close()

    
    
    
# relevant keys of trials_response_df:
'''
cell_specimen_id
trace
trace_timestamps  

lick_times  
reward_time 
initial_image_name                                                         im035
change_image_name

start_time                                                               546.125
stop_time                                                                553.414
trial_length                                                             7.28932
behavioral_response_time                                                 549.578
change_frame                                                               32432
change_time                                                              549.174
behavioral_response_latency  

hit                                                                         True
false_alarm                                                                False
miss                                                                       False
stimulus_change                                                             True
aborted                                                                    False
go                                                                          True
catch                                                                      False
auto_rewarded                                                              False
correct_reject       

trial_type
'''



