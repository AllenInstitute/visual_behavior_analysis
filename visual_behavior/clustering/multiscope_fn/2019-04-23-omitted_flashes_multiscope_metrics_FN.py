#%% [markdown]
# ### Import external packages

#%%
import os
import h5py 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import scipy.stats as st

#%% [markdown]
# ### Import internal packages

#%%
from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis 
from visual_behavior.visualization.ophys import experiment_summary_figures as esf
from visual_behavior.visualization.ophys import summary_figures as sf
import visual_behavior.ophys.response_analysis.utilities as ut
from visual_behavior.ophys.io.lims_database import LimsDatabase

#%%
# FN:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#%%
import ophysextractor
from ophysextractor.datasets.lims_ophys_session import LimsOphysSession
from ophysextractor.datasets.lims_ophys_experiment import LimsOphysExperiment
from ophysextractor.datasets.motion_corr_physio import MotionCorrPhysio
from ophysextractor.utils.util import mongo, get_psql_dict_cursor
from visual_behavior.ophys.io.convert_level_1_to_level_2 import get_segmentation_dir, get_lims_data, get_roi_locations, get_roi_metrics
import visual_behavior.utilities as vbut

#%% [markdown]
# ### Define local functions

#%%
def load_session_data(session_id):
    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'

    # Get list of experiments using ophysextractor
    Session_obj = LimsOphysSession(lims_id=session_id)
    list_mesoscope_exp = Session_obj.data_pointer['ophys_experiment_ids']

    whole_data={}
    DB = mongo.qc.metrics 

    for indiv_id in list_mesoscope_exp: 
        indiv_data = {}
        
        try: 
            dataset = VisualBehaviorOphysDataset(indiv_id, cache_dir=cache_dir)
        except Exception as e: 
            if 'roi_metrics.h5' in str(e):
                ophys_data = convert_level_1_to_level_2(indiv_id, cache_dir)
                dataset = VisualBehaviorOphysDataset(indiv_id, cache_dir=cache_dir)
            else:
                raise(e)
        #print('Cannot find data, converting '+str(indiv_id))
        #ophys_data = convert_level_1_to_level_2(indiv_id, cache_dir)
        #dataset = VisualBehaviorOphysDataset(indiv_id, cache_dir=cache_dir)
        indiv_data['fluo_traces'] = dataset.dff_traces
        indiv_data['time_trace'] =  dataset.timestamps['ophys_frames'][0]
        local_meta = dataset.get_metadata()
        indiv_data['targeted_structure'] = local_meta['targeted_structure'].values[0]
        indiv_data['mouse'] = local_meta['donor_id'].values[0]
        indiv_data['stage'] = local_meta['stage'].values[0]
        indiv_data['cre'] = local_meta['cre_line'].values[0]
        indiv_data['experiment_date'] = local_meta['experiment_date'].values[0]
        indiv_data['session_id'] = session_id
        trials = dataset.get_all_trials()
        hit_rate, catch_rate, d_prime = vbut.get_response_rates(trials)
        
        # the following are computed over sliding_window=100
        indiv_data['d_prime'] = d_prime
        indiv_data['hit_rate'] = hit_rate
        indiv_data['catch_rate'] = catch_rate

        # we have to get depth from Mouse-seeks database
        db_cursor = DB.find({"lims_id":indiv_id})

        local_depth = db_cursor[0]['lims_ophys_experiment']['depth']
        indiv_data['imaging_depth'] = local_depth
        whole_data[str(indiv_id)]= indiv_data
    
    data_list = pd.DataFrame([], columns=['lims_id', 'area', 'depth'])

    for index,lims_ids in enumerate(whole_data.keys()):
        depth = whole_data[lims_ids]['imaging_depth']
        area = whole_data[lims_ids]['targeted_structure']
        local_exp = pd.DataFrame([[lims_ids, area, depth]], columns=['lims_id', 'area', 'depth'])
        data_list = data_list.append(local_exp)  
    data_list = data_list.sort_values(by=['area', 'depth'])
    
    experiment_id = list_mesoscope_exp[1]
    dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
    table_stim = dataset.stimulus_table
    
    return [whole_data, data_list, table_stim]


#%%

def get_frame_number_from_time(dataset,time_to_look):
    time_trace =  dataset.timestamps['ophys_frames'][0]
    local_index = np.argmin(np.abs(time_trace-time_to_look))

    return local_index

def get_corr_movie_pointer(experiment_id):
    local_data =  LimsOphysExperiment(lims_id = experiment_id)    
    local_movie_data = MotionCorrPhysio(local_data) 
    data_pointer = local_movie_data.data_pointer

    return data_pointer

def get_frame(data_pointer, frame_number):
    # we have to get depth from Mouse-seeks database

    return data_pointer[frame_number,:,:]

def average_frames(data_pointer, list_frames):
    for index, indiv_frame in enumerate(list_frames):
        if index == 0:
            local_img = get_frame(data_pointer, indiv_frame).astype('float')
        else:
            local_img = local_img + get_frame(data_pointer, indiv_frame).astype('float')
    
    local_img = local_img/len(list_frames)

    return local_img

def get_triggered_averaged_movie(experiment_id, list_ref_frames, nb_frames_before, nb_frames_after):

    list_of_final_frames = np.arange(-nb_frames_before, nb_frames_after, 1)
    data_pointer = get_corr_movie_pointer(experiment_id)
    local_movie = np.zeros([len(list_of_final_frames), data_pointer.shape[1],  data_pointer.shape[2]])

    for index, local_index in enumerate(list_of_final_frames):
        to_average = list_ref_frames+local_index
        local_average = average_frames(data_pointer, to_average)
        local_movie[index, :, :] = local_average

    return local_movie


#%%
def plot_omitted_depth_area(whole_data, data_list, list_omitted):
    # plot for each session, and each plane, omitted-aligned traces (averaged across all omitted flashes).
    # traces for each ROI in the plane is shown in gray, and the average across all ROIs is shown in red. 
    
    fig1 = plt.figure(figsize=(30,20))

    # there are 8 planes... below we loop over data from each plane.
    for index,lims_ids in enumerate(data_list['lims_id']):
        local_fluo_traces = whole_data[lims_ids]['fluo_traces'] # roi x time
        local_time_traces = whole_data[lims_ids]['time_trace']  # time is in sec. Volume rate is 10 Hz.
        stamps_bef = 40
        stamps_aft = 40
        index_cell = 0
        scratch = 0
        nb_roi = local_fluo_traces.shape[0]
        all_averages = np.zeros([nb_roi, stamps_aft+stamps_bef])
        nb_times = len(list_omitted)

        plt.subplot(2,4,index+1)         
        
        # loop over ROIs in each plane
        for index_cell in range(nb_roi):   
            average_time = np.zeros([stamps_aft+stamps_bef])
            local_fluo = np.zeros([stamps_aft+stamps_bef])
            
            # loop over omitted stimuli in order to set the average trace across omitted stimuli
            for indiv_time in list_omitted:    
                local_index = np.argmin(np.abs(local_time_traces - indiv_time))
                average_time = average_time + local_time_traces[local_index - stamps_bef  :  local_index + stamps_aft] - indiv_time
                local_fluo = local_fluo + local_fluo_traces[index_cell,local_index - stamps_bef : local_index + stamps_aft]
            
            # compute average omission-aligned fluorescent trace across all omitted stimuli
            local_fluo = local_fluo/nb_times
            average_time = average_time/nb_times

            # shift the y values so all traces have y value = 0 at time 0. (jerome: align at time zero)
            Index_zero = np.argmin(np.abs(average_time))
            local_fluo = local_fluo - local_fluo[Index_zero]

            # normalize the average trace by baseline std (ie pre-omitted fluctuations)
            index_pre = np.where(average_time<0)
            std_norm = np.std(local_fluo[index_pre])
            local_fluo = local_fluo/std_norm

            # shift the y values so baseline mean is at 0 at time 0.
            #Index_zero = np.argmin(np.abs(average_time))
            local_fluo = local_fluo - np.mean(local_fluo[index_pre])

            all_averages[index_cell,:] = local_fluo
            
            # plot (gray) the average omitted-aligned trace for each ROI
            plt.plot(average_time, local_fluo, 'gray')
        
        # plot (red) average omitted-aligned traces across all ROIs
        plt.plot(average_time, np.mean(all_averages,axis=0),'r')
        
        if index+1==5:
            plt.xlabel("Relative time to omitted flashes (s)")
            plt.ylabel("Normalized response to pre-omitted period")
        plt.ylim((-5,20))
        plt.title(whole_data[lims_ids]['targeted_structure'] + " - " + str(whole_data[lims_ids]['imaging_depth'])+' um')
    
    return fig1


#%%
def plot_omitted_session(session_id):
    [whole_data, data_list, table_stim] = load_session_data(session_id)
    list_omitted = table_stim[table_stim['omitted']==True]['start_time']
    fig = plot_omitted_depth_area(whole_data, data_list, list_omitted)
    plt.suptitle(str(session_id))
    return fig


#%%
def get_all_omitted_peaks(session_id):
    [whole_data, data_list, table_stim] = load_session_data(session_id)
    list_omitted = table_stim[table_stim['omitted']==True]['start_time']
    data = get_omitted_peaks(whole_data, data_list, list_omitted)
    return data


#%%
def get_all_multiscope_exp():
    local_db = mongo.db.ophys_session_log
    db_cursor = local_db.find({"project_code":{"$in":["VisualBehaviorMultiscope","MesoscopeDevelopment"]}})
    list_sesssions_id = []
    for indiv_cursor in db_cursor:
        try: 
            # We check the session is well constructed
            # Get list of experiments using ophysextractor
            Session_obj = LimsOphysSession(lims_id=indiv_cursor['id'])
            list_mesoscope_exp = Session_obj.data_pointer['ophys_experiment_ids']

            list_sesssions_id.append(indiv_cursor['id'])
        except: 
            print(str(indiv_cursor['id'])+' has issues')

    return(list_sesssions_id)


#%%
def get_meta_session(session_id):
    from ophysextractor.projects import (BehaviorProject, EphysProject, IsiProject,
                                     OphysProject)
    from ophysextractor.datasets import ChangeDetection
    proj = OphysProject(session_id,'VisualBehaviorMultiscope')
    cd_object = ChangeDetection(proj.ophys_session, _load_history = False)
    
    return cd_object.get_full_summary_stats()


#%%
def get_omitted_peaks(whole_data, data_list, list_omitted):
    
    omitted_peak = []
    depth = []
    area= []
    lims_id = []
    mouse = []
    stage = []
    cre = []
    d_prime = []
    experiment_date = []
    session_id = []
    
    # first we go through each plane, then each cell within a plane.
    # for each cell find peak response to omission on the omission-trial-averaged trace that has been normalized to 
    # baseline sd and shifted so baseline mean is at 0
    for index,lims_ids in enumerate(data_list['lims_id']):
        local_fluo_traces = whole_data[lims_ids]['fluo_traces']
        local_time_traces = whole_data[lims_ids]['time_trace']
        stamps_bef = 40
        stamps_aft = 40
        index_cell = 0
        scratch = 0
        nb_roi = local_fluo_traces.shape[0]
        all_averages = np.zeros([nb_roi, stamps_aft+stamps_bef])
        nb_times = len(list_omitted)

        for index_cell in range(nb_roi):   
            average_time = np.zeros([stamps_aft+stamps_bef])
            local_fluo = np.zeros([stamps_aft+stamps_bef])

            for indiv_time in list_omitted:    
                local_index = np.argmin(np.abs(local_time_traces - indiv_time))
                average_time = average_time + local_time_traces[local_index-stamps_bef:local_index+stamps_aft]-indiv_time
                local_fluo = local_fluo+local_fluo_traces[index_cell,local_index-stamps_bef: local_index+stamps_aft]

            local_fluo = local_fluo/nb_times
            average_time = average_time/nb_times

            # align at time zero
            Index_zero = np.argmin(np.abs(average_time))
            local_fluo = local_fluo-local_fluo[Index_zero]

            # normalize pre-omitted fluctuations
            index_pre = np.where(average_time<0)
            std_norm = np.std(local_fluo[index_pre])
            local_fluo = local_fluo/std_norm

            # align at time zero
            #Index_zero = np.argmin(np.abs(average_time))
            local_fluo = local_fluo - np.mean(local_fluo[index_pre])

            all_averages[index_cell,:] = local_fluo
            
            
            # windows to look for peak 
            bottom = 0
            top = 2
            list_times = np.where(np.all([average_time > bottom, average_time < top], axis=0))
            local_peak = np.max(local_fluo[list_times])

            omitted_peak.append(local_peak)
            depth.append(whole_data[lims_ids]['imaging_depth'])
            area.append(whole_data[lims_ids]['targeted_structure'])
            mouse.append(whole_data[lims_ids]['mouse'])
            stage.append(whole_data[lims_ids]['stage'])
            cre.append(whole_data[lims_ids]['cre'])
            session_id.append(whole_data[lims_ids]['session_id'])
            experiment_date.append(whole_data[lims_ids]['experiment_date'])
            d_prime.append(whole_data[lims_ids]['d_prime'])
            lims_id.append(lims_ids)
    
    all_cell_properties = pd.DataFrame(data = {'d_prime': d_prime, 'lims_id':lims_id, 'session_id':session_id,\
                                               'depth':depth, 'targeted_structure':area,\
                                               'omitted_peak_relative_change':omitted_peak,\
                                               'mouse':mouse, 'stage':stage, 'cre':cre,\
                                               'experiment_date':experiment_date})   
    
    return all_cell_properties   


#%%
X = get_all_omitted_peaks(814281477)

print(X)


#%% [markdown]
# ### Run local analysis

#%% Get the ID of all mesoscope sessions

list_all_sessions = get_all_multiscope_exp()
'''
for session_id in list_all_sessions:
    print session_id
    [whole_data, data_list, table_stim] = load_session_data(session_id)
    
    whole_data_all.append(whole_data)
    data_list_all.append(data_list)
    table_stim_all.append(table_stim)
'''

#%% Set all_pd which includes data (peak response to average omission trials, as well as some metadata).
# Organization is as follows: 
# data from all sessions. Within each session, data from all planes. Within each plane, data from all neurons exist. 

initiate = True
for indiv_id in list_all_sessions:
    
    try:
        # local_pd will include the following:
#        'omitted_peak_relative_change':omitted_peak, 
#        'd_prime': d_prime, 
#        'lims_id':lims_id, 'session_id':session_id, 
#        'depth':depth, 
#        'targeted_structure':area, 
#        'mouse':mouse, 'stage':stage, 'cre':cre, 'experiment_date':experiment_date 
        
        local_pd = get_all_omitted_peaks(indiv_id) # this includes data from all planes and all neurons (for one session)
        
        if initiate:
            all_pd = local_pd
            initiate = False
        else:
            all_pd = all_pd.append(local_pd)
        
    except Exception as e: # FN: apparently these are good, but their analysis file is not yet saved! (you are running the convert code so you should have their analysis dir soon.)
        print(e)        
        print("Issues with "+str(indiv_id))

print(all_pd)
print np.shape(all_pd)
print all_pd.iloc[0]

print np.unique(all_pd.session_id.values).shape, 'sessions are in all_pd'

print np.unique(all_pd['cre'].values)
sum(all_pd['cre'].str.find('Vip')!=-1)
sum(all_pd['cre'].str.find('Sst')!=-1)

print np.unique(all_pd['stage'].values)
print np.unique(all_pd['experiment_date'].values)
print np.unique(all_pd['mouse'].values)


#%% Select sessions out of all_pd that you want to plot
    
# Select one CRE line:
all_pd_local = all_pd[all_pd['cre'].isin(['Vip-IRES-Cre'])]

print np.shape(all_pd_local)


# Select session types (active, passive, image A, B, etc):
#local_pd = all_pd_local[~all_pd_local['stage'].isin(['OPHYS_5_images_B_passive','OPHYS_0_images_A_habituation','OPHYS_2_images_A_passive'])]
local_pd = all_pd_local[all_pd_local['stage'].isin(['OPHYS_2_images_A_passive', 'OPHYS_5_images_B_passive'])]
#all_pd_local = all_pd_local[all_pd_local['stage'].isin(['OPHYS_6_images_B','OPHYS_4_images_B'])]

list_sessions_id = local_pd['session_id'].unique()

# sessions of interest (cre line and session type):
#print local_pd
print np.shape(local_pd) 
print list_sessions_id


#%% Plot for each area, neuron-averaged peak response to omissions vs. depth

plt.figure()
date_all = []

# go through sessions defined above (ie VIP and only A,passive or B,passive)
for indiv_id in list_sessions_id: # indiv_id = list_sessions_id[0]
    
    # Get data for one session, from all planes and all neurons 
    local_pd = all_pd_local[all_pd_local['session_id']==indiv_id]
    print np.shape(local_pd) 
    dateNow = np.unique(local_pd['experiment_date'].values)

    # Make plots for each area
    list_area = local_pd['targeted_structure'].unique()    
    
    # Go through neurons of a given area
    for index, uni_area in enumerate(list_area): 
            
        local_pd_area = local_pd[local_pd['targeted_structure']==uni_area]
        print(local_pd_area['stage'].unique())

        list_depth = local_pd_area['depth'].unique()

        list_depth = np.sort(list_depth)
        
        list_mean = []
        
        # Go through neurons of a given plane (depth) and compute their average peak response to omissions
#        h_all = []
        for indiv_depth in list_depth:
            a = local_pd_area[local_pd_area['depth']==indiv_depth]['omitted_peak_relative_change']
            print np.shape(a) # number of neurons in each plane
            local_mean = np.mean(a)
            list_mean.append(local_mean)
         
        # plot for each area, neuron-averaged peak response to omissions vs. depth            
        plt.subplot(1,2,index+1)
#        plt.plot(list_depth, list_mean/ list_mean[0], label = local_pd_area['d_prime'].unique())          
        
        h = plt.plot(list_depth, np.array(list_mean), label=dateNow)# / np.array(list_mean[0])) #, label = local_pd_area['d_prime'].unique())          

        # normalize the peak response by the response at the most superficial layer
#        h = plt.plot(list_depth, np.array(list_mean) / np.array(list_mean[0]), label=dateNow) #, label = local_pd_area['d_prime'].unique())          
#        plt.ylim([0,2])
        
        #plt.scatter(local_pd_area['depth'], local_pd_area['omitted_peak_relative_change']/list_mean[0], label=indiv_id)
        plt.title(uni_area)
#        h_all.append(h)

#plt.legend(h_all[0], date_all)        
plt.legend(loc='center left', bbox_to_anchor=(1,.7), frameon=False)


#%%
all_pd['experiment_date'].unique()


#%% FN

session_id = 814281477

[whole_data, data_list, table_stim] = load_session_data(session_id)
list_omitted = table_stim[table_stim['omitted']==True]['start_time']

#data = get_omitted_peaks(whole_data, data_list, list_omitted)

plot_omitted_depth_area(whole_data, data_list, list_omitted)



