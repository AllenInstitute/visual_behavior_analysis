#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using AllenSDK to set all_sess_dff_stim which has the dff traces as well as stimulus information and some metadata for each mesoscope experiment.

After this script run all_sess_dff_stim_load.py to load the pandas saved here.

This is made for Akul Gupta, for the RNN analysis.

Created on Mon Jul 6 19:52:25 2020
@author: farzaneh
"""

saveResults = 1 # save allsess file (in a pkl file named "all_sess_omit_traces_peaks_allTraces_sdk...")

from def_funs import *
import visual_behavior.data_access.loading as loading  # VBA data_access module provides useful functions for identifying and loading experiments to analyze. 


#%% The get_filtered_ophys_experiment_table() function returns a table describing passing ophys experiments from relevant project codes 

experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True) # 
print(np.shape(experiments_table))


#%% Get VisualBehaviorMultiscope experiments

a = experiments_table['project_code'].values
all_project_codes = np.unique(a) # ['VisualBehavior', 'VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d', 'VisualBehaviorTask1B']
experiments_table_2an = experiments_table[a=='VisualBehaviorMultiscope']

# add experiment_id to the columns
experiments_table_2an.insert(loc=1, column='experiment_id', value=experiments_table_2an.index.values)
print(experiments_table_2an.shape)


#%% For each experiment set dff traces, and stimulus data.

cols = ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'frame_dur', 'n_neurons', \
        'cell_specimen_ids', 'dff', 'dff_times', 'stimulus_info'] 

all_sess_dff_stim = pd.DataFrame([], columns=cols) # size: 8*len(num_sessions)

errors_iexp = []
errors_log = []

for iexp in np.arange(0, experiments_table_2an.shape[0]): # iexp = 0 
        
    experiment_id = experiments_table_2an.index[iexp]     # experiments_table_2an.iloc[experiments_table_2an.index == ophys_experiment_id]
    session_id = experiments_table_2an.iloc[iexp]['ophys_session_id']
    
    cre = experiments_table_2an.iloc[iexp]['cre_line']
    date = experiments_table_2an.iloc[iexp]['date_of_acquisition'][:10]
    stage = experiments_table_2an.iloc[iexp]['session_type']
    area = experiments_table_2an.iloc[iexp]['targeted_structure']
    depth = int(experiments_table_2an.iloc[iexp]['imaging_depth'])
    
    print(f'\n\n------------ Setting vars for experiment_id {experiment_id} , session_id {session_id} , ({iexp} / {experiments_table_2an.shape[0]}) , {cre[:3]} ------------\n\n')
    
    session_name = str(experiments_table_2an.iloc[iexp]['session_name'])        
        
    try:
        
        # set mouse_id
        uind = [m.start() for m in re.finditer('_', session_name)]
        mid = session_name[uind[0]+1 : uind[1]]
        if len(mid)<6: # a different nomenclature for session_name was used for session_id: 935559843
            mouse_id = int(session_name[uind[-1]+1 :])
        else:
            mouse_id = int(mid)
        
        # get SDK dataset through VBA loading function # this function gets an SDK session object then does a bunch of reformatting to fix things
        dataset = loading.get_ophys_dataset(experiment_id, include_invalid_rois=False)

        frame_dur = np.mean(np.diff(dataset.ophys_timestamps))
        n_neurons = dataset.dff_traces.shape[0]

        dff = np.vstack(dataset.dff_traces['dff']) # neurons x frames        
        dff_times = dataset.ophys_timestamps # frames

        cell_specimen_ids = dataset.cell_specimen_ids # neurons

#         dataset.dff_traces.keys() #['cell_roi_id', 'dff']
#         dataset.metadata

        
        # Stimulus information
        stimulus_info = dataset.stimulus_presentations.loc[:,['image_index', 'image_name', 'omitted', 'start_time', 'stop_time']]
        stimulus_info.shape
        
        
        
        all_sess_dff_stim.at[iexp, ['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'frame_dur', 'n_neurons', \
                               'cell_specimen_ids', 'dff', 'dff_times', 'stimulus_info']] = \
            session_id ,  experiment_id ,  mouse_id ,  date ,  cre ,  stage ,  area ,  depth ,  frame_dur ,  n_neurons, cell_specimen_ids , dff , dff_times , stimulus_info


    except Exception as e:
        print(e)
        errors_iexp.append(iexp)
        errors_log.append(e)

        

all_sess_dff_stim.at[all_sess_dff_stim['mouse_id'].values==6, 'mouse_id'] = 453988

# keep a copy of the dataframe before removing any experiments
all_sess_now0 = copy.deepcopy(all_sess_dff_stim)
print(np.shape(all_sess_now0))



#%% Remove experiments (rows) without any neurons
# remember "drop" uses the value of index to remove a row, not based on its location

exp_noNeur = np.argwhere(np.isnan(all_sess_now0['n_neurons'].values.astype(float))).flatten()
if len(exp_noNeur)>0:
    all_sess_now = all_sess_now0.drop(exp_noNeur, axis=0)
    print('\nExperiments with no neurons removed!')
    print(np.shape(all_sess_now))
else:
    all_sess_now = all_sess_now0
    
print(np.shape(all_sess_now))    
    
    

#%% Remove sessions with less than 8 experiments
# this part is commented so we save all experiments; but when we load it, we should apply this part if needed.
'''
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
'''
    
    
    
    
        
#%% Sort all_sess_now by session_id, area and depth    

all_sess_dff_stim = all_sess_now.sort_values(by=['session_id','area', 'depth'])

    

    
#%% Save the results

if saveResults:

    analysis_name = ''
    dir_now = dir_server_me 
    
    namespec = 'dff_stim'
    now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")


    print('\nSaving allsess file')

    name = 'all_sess_%s%s_%s' % (analysis_name, namespec, now)            
    allSessName = os.path.join(dir_now , name + '.pkl') # save to a pickle file
    print(allSessName)

    
    #### save all_sess_dff_stim to allSessName ####
    f = open(allSessName, 'wb')

    pickle.dump(all_sess_dff_stim, f)    
    pickle.dump(errors_iexp, f)
    pickle.dump(errors_log, f)
    
    f.close()

    
    


#%% After this script run all_sess_dff_stim_load.py to load the pandas saved here.

