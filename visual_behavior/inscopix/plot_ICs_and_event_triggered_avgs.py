import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import warnings
import cv2
import time
from skimage import io as skio
from skimage.measure import block_reduce
import gc

import seaborn as sns

from PIL import Image

from ipywidgets import interact

from sync import Dataset
from visual_behavior.inscopix import utilities as iu
import visual_behavior.plotting as vbp
import visual_behavior.utilities as vbu
import imaging_behavior.core.utilities as ut
import imaging_behavior.plotting.utilities as pu
import imaging_behavior.plotting.plotting_functions as pf


def get_filename_dict(path,mouse_id):

    filename_dict = iu.find_filenames(path,mouse_id)

    return filename_dict

def extract_sync_data(filename_dict):
    sync_data = Dataset(filename_dict['sync_file'])
    sample_freq = sync_data.meta_data['ni_daq']['counter_output_freq']

    behavior_frame_times = sync_data.get_falling_edges('behavior_vsync')/sample_freq
    behavior_frame_times = behavior_frame_times[1:]
    f_frame_times = sync_data.get_rising_edges('fluorescence_camera')/sample_freq

    return behavior_frame_times,f_frame_times

def load_pkl(filename_dict):
    pkl=filename_dict['behavior_pkl']
    pkl_data=pd.read_pickle(pkl)
    stim_df=pd.DataFrame(pkl_data['triallog'])

    return stim_df

def get_mouse_id(path):
    splits = path.split("_")
    for split in splits:
        if split.startswith("M"):
            return split

def load_IC_masks(filename_dict):
    IC_masks = {}
    for ii,filename in enumerate(filename_dict['IC_list']):
        IC_masks[filename.split('IC')[1].split('.tif')[0]] = skio.imread(filename)

    return IC_masks

def get_event_frames(stim_df,behavior_frame_times,f_frame_times):
    ## find all event times

    visual_events=[]
    for i,row in stim_df.iterrows():
    #     print row['stim_type']
        if row['stim_type']=='visual':
            behavior_frame_times[row['startframe']]+0.035
            visual_events.append(ut.find_nearest_index(behavior_frame_times[row['startframe']]+0.035,f_frame_times))
            
    auditory_events=[]
    for i,row in stim_df.iterrows():
    #     print row['stim_type']
        if row['stim_type']=='auditory':
            behavior_frame_times[row['startframe']]+0.035
            auditory_events.append(ut.find_nearest_index(behavior_frame_times[row['startframe']]+0.035,f_frame_times))
            
    visual_auditory_events=[]
    for i,row in stim_df.iterrows():
    #     print row['stim_type']
        if row['stim_type']=='auditory_and_visual':
            behavior_frame_times[row['startframe']]+0.035
            visual_auditory_events.append(ut.find_nearest_index(behavior_frame_times[row['startframe']]+0.035,f_frame_times))

    return visual_events,auditory_events,visual_auditory_events

def make_IC_plots(path,events,traces,IC_masks,figsize=(9,6)):
    visual_events=events[0]
    auditory_events=events[1]
    visual_auditory_events=events[2]

    savepath=os.path.join(path,"IC_plots")
    iu.mkdir(savepath)

    pb = vbu.Progress_Bar_Text(len(IC_masks.keys()))

    fig=plt.figure(figsize=figsize)
    for ii,IC_num in enumerate(np.sort(IC_masks.keys())):
    #     if ii==1:
    #         break
        fig.clf()
        ax_IC_mask = vbp.placeAxesOnGrid(fig,xspan=(0,0.45),yspan=(0,0.5))
        ax_IC_ts = vbp.placeAxesOnGrid(fig,xspan=(0.575,1),yspan=(0,0.40))
        ax_ev_trig_avg = []
        ax_ev_trig_avg.append(vbp.placeAxesOnGrid(fig,xspan=(0,0.3),yspan=(0.57,1)))
        ax_ev_trig_avg.append(vbp.placeAxesOnGrid(fig,xspan=(0.356,0.656),yspan=(0.57,1)))
        ax_ev_trig_avg.append(vbp.placeAxesOnGrid(fig,xspan=(0.7,1),yspan=(0.57,1)))
        
    #     ax_IC_mask.imshow(IC_masks[IC_num],cmap='gray')
        pf.show_image(
            IC_masks[IC_num],
            ax=ax_IC_mask,
            cmin=np.min(IC_masks[IC_num]),
            cmax=np.max(IC_masks[IC_num]),
            colorbar=True,
            cmap='gray',
            hide_ticks=True
        )
        ax_IC_mask.axis('off')
        ax_IC_mask.set_title('IC Mask',fontweight="bold")
        
        trace = traces['IC trace {} (s.d.)'.format(int(IC_num))].values
        ax_IC_ts.plot(traces['Time (s)']/60.,trace)
        ax_IC_ts.set_ylabel('z-scored activity')
        ax_IC_ts.set_xlabel('time (min)')
        ax_IC_ts.set_ylim(-5,20)
        ax_IC_ts.set_title('full timeseries',fontweight="bold")
        
        dat_vis=ut.event_triggered_average(trace,mask=None,events=visual_events,frame_before=20,frame_after=100,sampling_rate=20,
                                                       output='f',progressbar=False)
        dat_aud=ut.event_triggered_average(trace,mask=None,events=auditory_events,frame_before=20,frame_after=100,sampling_rate=20,
                                                       output='f',progressbar=False)
        dat_vis_aud=ut.event_triggered_average(trace,mask=None,events=visual_auditory_events,frame_before=20,frame_after=100,sampling_rate=20,
                                                       output='f',progressbar=False)
        
        pf.plot_event_triggered_timeseries(dat_vis,ax=ax_ev_trig_avg[0])
        pf.plot_event_triggered_timeseries(dat_aud,ax=ax_ev_trig_avg[1])
        pf.plot_event_triggered_timeseries(dat_vis_aud,ax=ax_ev_trig_avg[2])
        
        titles = ['visual triggered avg','auditory triggered avg','vis/aud triggered avg']
        for jj in range(3):
            ax_ev_trig_avg[jj].set_title(titles[jj],fontweight="bold")
            ax_ev_trig_avg[jj].set_xlabel('time from stim (s)')
            ax_ev_trig_avg[jj].set_ylim(-5,20)
            ax_ev_trig_avg[jj].axvline(0,alpha=0.5,linewidth=3,color='red',zorder=-1)
        ax_ev_trig_avg[0].set_ylabel('z-scored activity')
        
        fig.suptitle('IC Number {}'.format(int(IC_num)),fontweight="bold")
        plt.subplots_adjust(top=0.90)
            
        vbp.save_figure(fig,os.path.join(savepath,'IC{}'.format(IC_num)),formats=['.png'],figsize=figsize)
        
        gc.collect

        pb.update()


def run(path):

    mouse_id = get_mouse_id(path)
    print "gathering filenames for mouse {}".format(mouse_id)
    filename_dict = get_filename_dict(path,mouse_id)
    print "extracting sync data"
    behavior_frame_times,f_frame_times = extract_sync_data(filename_dict)
    print "opening event log"
    stim_df = load_pkl(filename_dict)
    print "opening traces"
    traces = iu.open_traces(filename_dict['traces'])
    print "opening IC mask files"
    IC_masks = load_IC_masks(filename_dict)
    event_frames = get_event_frames(stim_df,behavior_frame_times,f_frame_times)
    print "making IC plots"
    make_IC_plots(path,event_frames,traces,IC_masks)

if __name__ == '__main__':
    print "can't run as main"

