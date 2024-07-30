#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using AllenSDK to do umap/clustering analysis.

Created on Tue May 26 11:17:25 2020
@author: farzaneh
"""

saveResults = 1 # save allsess file (in a pkl file named "all_sess_omit_traces_peaks_allTraces_sdk...")
bl_percentile = 10


#%%

from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
from visual_behavior.ophys.response_analysis.response_processing import get_omission_response_xr
omission_response_xr = get_omission_response_xr(dataset, use_events=False, frame_rate=None)

    
# help(ResponseAnalysis)
from def_funs import *


#%% Set some essential parameters

# get omission_response_df for an arbitrary experiment, so we can set samps_bef etc below
ophys_experiment_id = experiments_table.index[0]     # experiments_table_2an.iloc[experiments_table_2an.index == ophys_experiment_id]
dataset = loading.get_ophys_dataset(ophys_experiment_id, include_invalid_rois=False)
analysis = ResponseAnalysis(dataset) 
omission_response_df = analysis.get_response_df(df_name='omission_response_df')

n_frames = len(omission_response_df['trace'].iloc[0])
samps_bef = np.argwhere(omission_response_df['trace_timestamps'].iloc[0]==0)[0,0]
samps_aft = n_frames - samps_bef
print(samps_bef, samps_aft)


bl_index_pre_omit = np.arange(0, samps_bef) # we will need it for response amplitude quantification.

# frame_dur = np.mean(np.diff(omission_response_df['trace_timestamps'].iloc[0]))

flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, flashes_win_trace_index_unq, grays_win_trace_index_unq = \
    flash_gray_onset_relOmit(samps_bef, samps_aft, frame_dur)

flash_ind = abs(flashes_win_trace_index_unq[flashes_win_trace_index_unq<0][-1].astype(int))

# take one image before and one image after omission
samps_bef_now = samps_bef - flash_ind # image before omission
samps_aft_now = samps_bef + flash_ind*2 # omission and the image after


#%% Get VisualBehaviorMultiscope experiments

a = experiments_table['project_code'].values
all_project_codes = np.unique(a) # ['VisualBehavior', 'VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d', 'VisualBehaviorTask1B']
experiments_table_2an = experiments_table[a=='VisualBehaviorMultiscope']
experiments_table_2an.shape
# add experiment_id to the columns
experiments_table_2an.insert(loc=1, column='experiment_id', value=experiments_table_2an.index.values)


#%% For each experiment set trial-average omission-aligned traces.

cols = ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'frame_dur', 'cell_specimen_id', 'traces_allN_allO'] 
all_sess_allN_allO = pd.DataFrame([], columns=cols) # size: 8*len(num_sessions)

for iexp in np.arange(0, experiments_table_2an.shape[0]): # iexp = 0 
        
    experiment_id = experiments_table_2an.index[iexp]     # experiments_table_2an.iloc[experiments_table_2an.index == ophys_experiment_id]
    session_id = experiments_table_2an.iloc[iexp]['ophys_session_id']

    print(f'\n\n------------ Setting vars for experiment_id {experiment_id} , session_id {session_id} , ({iexp} / {experiments_table_2an.shape[0]}) ------------\n\n')
    
    cre = experiments_table_2an.iloc[iexp]['cre_line']
    date = experiments_table_2an.iloc[0]['date_of_acquisition'][:10]
    stage = experiments_table_2an.iloc[0]['session_type']
    area = experiments_table_2an.iloc[0]['targeted_structure']
    depth = int(experiments_table_2an.iloc[0]['imaging_depth'])
    
    
    session_name = str(experiments_table_2an.iloc[0]['session_name'])
    uind = [m.start() for m in re.finditer('_', session_name)]
    mouse_id = session_name[uind[0]+1 : uind[1]]
    
    
    # get SDK dataset through VBA loading function # this function gets an SDK session object then does a bunch of reformatting to fix things
    dataset = loading.get_ophys_dataset(experiment_id, include_invalid_rois=False)
    frame_dur = np.mean(np.diff(dataset.ophys_timestamps))
    n_neurons = dataset.dff_traces.shape[0]

    
    # get 3d traces: frames x trials x units
    omission_response_xr = get_omission_response_xr(dataset, use_events=False, frame_rate=None)
#     n_omissions = omission_response_xr['trial_id'].values.shape[0] 
#     n_neurons = omission_response_xr['trace_id'].values.shape[0]
#     time = omission_response_xr['eventlocked_timestamps'].values
#     trace_ftu = omission_response_xr['eventlocked_traces'].values # frames x trials x neurons

    
    '''
    # ResponseAnalysis class provides access to time aligned cell responses for trials, stimuli, and omissions
    # VBA also has useful functionality for creating data frames with cell traces aligned to the time of stimulus presentations, omissions, or behavioral trials. 
    analysis = ResponseAnalysis(dataset)  

    # get omission triggered responses 
    omission_response_df = analysis.get_response_df(df_name='omission_response_df')
        
    cell_ids = omission_response_df['cell_specimen_id'].values.astype(float)
    if np.isnan(cell_ids).any(): # cell matching has not been run, just copy cell roi ids to cell specimen ids
        print(f'\n\n Warning! experiment {experiment_id} does not have cell_specimen_id; using cell_roi_id instead....\n\n')
        dataset.dff_traces.index = dataset.dff_traces.cell_roi_id
        analysis = ResponseAnalysis(dataset) 
        omission_response_df = analysis.get_response_df(df_name='omission_response_df')

        
    frame_dur = np.mean(np.diff(omission_response_df['trace_timestamps'].iloc[0]))
#     n_neurons = len(np.unique(omission_response_df['cell_specimen_id']))
    n_omissions = int(omission_response_df.shape[0] / n_neurons)
#     sum(dataset.stimulus_presentations['omitted']==True)
#     sum(stim_response_df['omitted']) / n_neurons
    
#     a = omission_response_df[['cell_specimen_id', 'trace', 'trace_timestamps']]
    traces_allN_allO = np.vstack(omission_response_df['trace'].values) # all neurons and all omissions x frames (107)  # all neurons, omission 1, then all neurons omission 2, etc.... all neuron traces for all omissions are concatenated.
    '''
  
    all_sess_allN_allO.at[iexp, ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', \
                                 'frame_dur', 'omission_response_xr']] = \
        session_id ,  experiment_id ,  mouse_id ,  date ,  cre ,  stage ,  area ,  depth ,  n_omissions ,  n_neurons ,  frame_dur ,  cell_ids , traces_allN_allO

#     all_sess_allN_allO.at[iexp, ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', \
#                                  'n_omissions', 'n_neurons', 'frame_dur', 'cell_specimen_id', 'traces_allN_allO']] = \
#         session_id ,  experiment_id ,  mouse_id ,  date ,  cre ,  stage ,  area ,  depth ,  n_omissions ,  n_neurons ,  frame_dur ,  cell_ids , traces_allN_allO


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


    print('Saving allsess file')

    name = 'all_sess_%s%s_%s' % (analysis_name, namespec, now)            
    # save to a pickle file
    allSessName = os.path.join(dir_now , name + '.pkl') 
    print(allSessName)

    #### save the file ####
    f = open(allSessName, 'wb')
    pickle.dump(all_sess_allN_allO, f)    
    pickle.dump(input_vars, f)
    f.close()

    
    
    
##################################################################
#%% Go through each cre line, ...
##################################################################

cre_all = all_sess_allN_allO['cre'].values
cre_lines = np.unique(cre_all)

all_sess_now = all_sess_allN_allO
cols0 = all_sess_now.columns
cols = np.concatenate((cols0, ['traces_fut']))

for icre in range(len(cre_lines)): # icre = 0
    
    cre = cre_lines[icre]    
    thisCre_numMice = sum(cre_all==cre) # number of mice that are of line cre
    print(f'\n~{thisCre_numMice/8.} sessions for {cre} mice')

    all_sess_thisCre = pd.DataFrame([], columns=cols)
    all_sess_thisCre[cols0] = all_sess_now.iloc[cre_all==cre]    
    
    for iexp in range(len(all_sess_thisCre)): # iexp = 0
        
        experiment_id = all_sess_thisCre.iloc[iexp]['experiment_id']
        
        n_neurons = all_sess_thisCre.iloc[iexp]['n_neurons']
        n_omissions = all_sess_thisCre.iloc[iexp]['n_omissions']
        n_frames = all_sess_thisCre.iloc[iexp]['traces_allN_allO'].shape[1]
        traces_allN_allO = all_sess_thisCre.iloc[iexp]['traces_allN_allO']
        
        traces_temp = np.reshape(traces_allN_allO, (n_neurons, n_omissions, n_frames), order='F')
        traces_fut = np.transpose(traces_temp, (2,0,1)) # frames x units x trials
#         traces_fut.shape
        
        # sanity check for reshape
#         cells = all_sess_thisCre.iloc[iexp]['cell_specimen_id']
#         icell = 18
#         plt.plot(np.mean(traces_allN_allO[all_sess_thisCre.iloc[iexp]['cell_specimen_id']==cells[icell]], axis=0)) 
#         plt.plot(np.mean(traces_allN_allO_fut[:,icell], axis=1))
        
        all_sess_thisCre.at[all_sess_thisCre.index[iexp],'traces_fut'] = traces_fut    
        
        
        
    # note: local_fluo_allOmitt for invalid experiments will be a nan trace as if they had one neuron
    a = all_sess_thisCre['traces_fut'].values # each element is frames x units x trials
#     a.shape

    ############### take the average of omission-aligned traces across trials; also transpose so the traces are neurons x frames
    aa = np.array([np.nanmean(a[i], axis=2).T for i in range(len(a))]) # each element is neurons x frames
#     np.shape(aa[0])

    ############### concantenate trial-averaged neuron traces from all sessions
    all_sess_ns = np.vstack(aa) # neurons_allSessions_thisCre x 24(frames)
#     all_sess_ns.shape
#     plt.plot(np.nanmean(all_sess_ns, axis=0))

    
    for i in range(aa.shape[0]):
        print(aa[i].shape[1])
        


    # compute baseline (on trial-averaged traces); we need it for quantifying flash/omission evoked responses.    
    bl_preOmit = np.percentile(traces_allN_aveO[:,bl_index_pre_omit], bl_percentile, axis=1) # neurons # use the 10th percentile    

    # take one image before and one image after omission
    traces_allN_aveO_fof = traces_allN_aveO[:, samps_bef_now: samps_aft_now] # neurons x frames (24)

