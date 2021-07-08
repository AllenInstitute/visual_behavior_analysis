# %load_ext autoreload

# from importlib import reload
# import x
# reload(x)

#get_ipython().magic(u'matplotlib inline')   

import numpy as np

frame_dur = np.array([0.093]) # sec # mesoscope time resolution (4 depth, 2 areas) (~10.7 Hz; each pair of planes that are recorded simultaneously have time resolution frame_dur)
flash_dur = .25
gray_dur = .5
num_planes = 8

if 'control_single_beam' in locals() and control_single_beam==1: # data resampled as if it was recorded with single beam scope
    frame_dur = frame_dur*2

#%% Use below to stop matplotlib.font_manager debug messages in log file
# https://stackoverflow.com/questions/58320567/matplotlib-font-manager-debug-messages-in-log-file

import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename='logfile.log', # level=logging.DEBUG
                    format='%(levelname)s:%(name)s:%(message)s)')

def objective(x):
    obj=model(x)
    logger.debug('objective = {}'.format(obj))

    return obj


#%%
import numpy as np
import socket
import os
import h5py 
import pandas as pd
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('whitegrid')
import scipy as sci
import scipy.stats as st
from IPython.display import display
import datetime
import copy
import sys
import re
import numpy.random as rnd
import pickle
import shutil
import gc
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from matplotlib import cm # colormap


#%%
try:
    from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
except Exception as E:
    print('cound not import visual_behavior.ophys.io.convert_level_1_to_level_2')
    print(E)
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis 
from visual_behavior.visualization.ophys import experiment_summary_figures as esf
from visual_behavior.visualization.ophys import summary_figures as sf
import visual_behavior.ophys.response_analysis.utilities as ut
from visual_behavior.ophys.io.lims_database import LimsDatabase


#%%
import seaborn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


#%%
import ophysextractor
from ophysextractor.datasets.lims_ophys_session import LimsOphysSession
from ophysextractor.datasets.lims_ophys_experiment import LimsOphysExperiment
from ophysextractor.datasets.motion_corr_physio import MotionCorrPhysio
from ophysextractor.utils.util import mongo, get_psql_dict_cursor
from visual_behavior.ophys.io.convert_level_1_to_level_2 import get_segmentation_dir, get_lims_data, get_roi_locations, get_roi_metrics
import visual_behavior.utilities as vbut


#%%
# from def_paths import * 
'''
if socket.gethostname() == 'OSXLT1JHD5.local': # allen mac
    dirAna = "/Users/farzaneh.najafi/Documents/analysis_codes/"
    dir0 = '/Users/farzaneh.najafi/OneDrive - Allen Institute/Analysis'

elif socket.gethostname() == 'ibs-farzaneh-ux2': # allen pc
    dirAna = "/home/farzaneh/Documents/analysis_codes/"
    dir0 = '/home/farzaneh/OneDrive/Analysis'
    
elif socket.gethostname() == 'ibs-is-analysis-ux1': # analysis computer
    dirAna = "/home/farzaneh.najafi/analysis_codes/"
    dir0 = '/home/farzaneh.najafi/OneDrive/Analysis' # you need to create this!
'''

if sys.platform=='win32':
    dir_server_me = 'Z:\\braintv\\workgroups\\nc-ophys\\Farzaneh'
else:
    # /allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
    dir_server_me = os.path.join(os.sep, 'allen', 'programs', 'braintv', 'workgroups', 'nc-ophys', 'Farzaneh')

#dir_svm = os.path.join(dir_server_me, 'SVM')
#if same_num_neuron_all_planes:
#    dir_svm = os.path.join(dir_svm, 'same_num_neurons_all_planes')


#%% Find index of list "ophys_sess" in the main list "stages" (they are both list of strings)
'''
# you don't need the following, all you need to do is:
np.in1d(np.array(stages), np.array(ophys_sess))

def index_list_in_listM(stages, ophys_sess):
    
    stage_now = []
    for j in range(len(stages)):
        this_stage = [stages[j].find(ophys_sess[i]) for i in range(len(ophys_sess))]
        stage_now.append(any(np.in1d(this_stage, 0)))
#    len(stage_now), sum(stage_now)    
    stage_now = np.array(stage_now)
    
    return stage_now
'''


#%% Rename files
'''
for i in allSessName:
    d,p = os.path.split(i)
    ii = os.path.basename(i)
    nn = ii.find('_')
    ii = 'this'+ii[nn:]
    de = os.path.join(d, ii)
    os.rename(i, de)
'''
'''
import re
a,allSessName = all_sess_set_h5_fileName('(.*)events_svm_gray', '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/SVM', all_files=0)

for i in allSessName: # i = allSessName[0]
    d,p = os.path.split(i)
    ii = os.path.basename(i)
    
    nn = ii.find('events_svm_gray_omit')
    nn2 = nn+len('events_svm_gray_omit')
    
    ii = ii[:nn]+'events_svm_decode_baseline_from_nobaseline'+ii[nn2:]
    de = os.path.join(d, ii)
    
    i = os.path.join('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/SVM', i)
    de = os.path.join('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/SVM', de)
    
    print(i)
    print(de)
    
    os.rename(i, de)
'''



#%% Set figure labels and fgn (it will be used for setting fign (the final figure named to be saved))

def set_figname_labels(peak_win, flash_win, norm_to_max, doShift, doScale, doShift_again, bl_gray_screen, mean_notPeak, trace_median):
    if norm_to_max:
        ylab = '(norm. to max sess amp)'
        ylab_short = '(norm max)'
        fgn = 'normMax' # for figure name
    #     fgn = fgn0 + 'normMax' # for figure name

    else:
        # Note: bl_gray_screen is for when the df/f traces were scaled and shifted (the default is not to do so);
        # but when quantifying flash/omission responses, we always use 10th percentile of midian traces of omit-aligned traces to compute baseline and subtract it off quantifications (within those windows).
        if np.logical_and(doShift, doScale):
            if bl_gray_screen: # baseline was computed on the initial gray screen at the beginning of the session
                fgn0 = 'blGray_'
            else:
                fgn0 = 'blP10' # baseline was computed on the 10th percentile of trace during samps_bef frames preceding the omissions

            ylab =  '(norm. to baseline sd)'
            ylab_short =  '(norm blSD)'
            fgn = fgn0 + 'normBlSd_'

        else:
            fgn0 = ''
            ylab = ''
            ylab_short = ''
            fgn = fgn0        
    
    if ylab=='':
        ylabel = 'DF/F'
    else:
        ylabel = 'DF/F %s' %(ylab) #if iplane==num_depth else ''


    if mean_notPeak:
        fgn = fgn + 'omitMean' + str(peak_win)
        lab_peakMean = 'Mean response'
    else:
        fgn = fgn + 'omitPeak' + str(peak_win)
        lab_peakMean = 'Peak amp'

    fgn = fgn + '_flashMean' + str(flash_win)
    # fgn = fgn + '_flashTiming' + str(flash_win_timing)

    if doShift_again:
        fgn = fgn + '_blShiftAgain'


    if trace_median: # use peak measures computed on the trial-median traces (less noisy)
        lab_med = 'trace med'
    else:
        lab_med = 'med of Trs'


    return fgn, ylabel, ylab_short, lab_peakMean, lab_med



#%% Compute median and 25th, 75th percentile across neurons for each experiment

def med_perc_each_exp(all_exp_traces):    

    # all_exp_traces has size e, where e = number of experiments
    # all_exp_traces[i] has size n, where n = number of neurons
    
    # output variables will have size e, where e = number of experiments

    # NOTE: 08/06/2020: I decided to go with mean (across trials, neurons) instead of median: it actually revealed that in the 3rd plane of LM (especially in LM) the mean response of Slc is slightly going up after omissions (we know some neurons are like this among slc).
    # Also, when we plot summary mice data, we take average across sessions of mice, so it makes sense that we also take average across trials and neurons (and not mean).    
    
    med_ns = np.array([np.mean(all_exp_traces[i]) for i in range(len(all_exp_traces))])
    q25_ns = np.array([np.percentile(all_exp_traces[i], 25) for i in range(len(all_exp_traces))]) # total number of planes (8*num_sessions); each element: med of peak timing across neurons in that plane
    q75_ns = np.array([np.percentile(all_exp_traces[i], 75) for i in range(len(all_exp_traces))]) # total number of planes (8*num_sessions); each element: med of peak timing across neurons in that plane
    
    return med_ns, q25_ns, q75_ns 


#%% Determine if a session is the 1st novel session or not
# match the session date ("date") with all the dates in mouse_trainHist_all2, to find the stage of the current and the previous sessions.

# NOTE: below you call a session novel if its BA: A and preceded by B, or AB: B and preceded by A. 
# It perhaps makes more sense to only call the 1st B session (after A sessions) (or the 1st A session after B sessions) a novel session. 
# for this from mouse training history figure out the date of the 1st novel session, and then see if the current experiment date matches that.
# to figure out the date of the 1st novel session, look for the 1st B after a row of A (or 1st A after a row of B); you have these pieces of code in set_mousetrainhist_allsess2an_whatsess.py
def is_session_novel(dir_server_me, mouse, date):

    # set file name: mouse_trainHist_all2
    analysis_name2 = 'mouse_trainHist_all2'
    name = 'all_sess_%s_.' %(analysis_name2) 
    allSessName2, h5_files = all_sess_set_h5_fileName(analysis_name2, dir_server_me)
    # print(allSessName2)

    # load mouse_trainHist_all2
    mouse_trainHist_all2 = pd.read_hdf(allSessName2) #, key='all_sess') #'svm_vars')        ## Load all_sess dataframe


    trainHist_this_mouse = mouse_trainHist_all2[mouse_trainHist_all2['mouse_id']==mouse]
    stages = list(trainHist_this_mouse['stage'].values)

    # All ophys_A sessions (all A, habit, passive)
    name = 'OPHYS\w+A'  # name = 'OPHYS\_[0-9]\w+A'
    regex = re.compile(name) # + '.h5')
    ophys_sessA = [string for string in stages if re.findall(regex, string)] # string=stages[-1]
    #    ophys_sessA, len(ophys_sessA)

    # All ophys_B sessions
    name = 'OPHYS\w+B'  # name = 'OPHYS\_[0-9]\w+A'
    regex = re.compile(name) # + '.h5')
    ophys_sessB = [string for string in stages if re.findall(regex, string)] # string=stages[-1]
#    ophys_sessB, len(ophys_sessB)


    ### match the session date ("date") with all the dates in mouse_trainHist_all2, so you can find the stage of the current and the previous sessions.
    d = trainHist_this_mouse['date'].values
    this_mouse_sess = np.argwhere(np.in1d(d, date)).squeeze()

    stage_this_sess = trainHist_this_mouse.iloc[this_mouse_sess]['stage'] # is this B
    stage_prev_sess = trainHist_this_mouse.iloc[this_mouse_sess-1]['stage'] # is this A

    # is current session B, and the previous session A?
    AB = np.logical_and(np.in1d(ophys_sessB, stage_this_sess).any(), np.in1d(ophys_sessA, stage_prev_sess).any())
    BA = np.logical_and(np.in1d(ophys_sessA, stage_this_sess).any(), np.in1d(ophys_sessB, stage_prev_sess).any())

    if any([AB,BA]):
        session_novel = True
    else:
        session_novel = False
        
#     print(trainHist_this_mouse)
#     print(f'Current data: {date}; session_novel: {session_novel}')
    
    return session_novel
        

#%%
# Set time for the interpolated traces (every 3ms time points); 
# Set the time of flashes and grays for the imaging traces;
# Set some useful vars which you will need for most subsequent analaysis: # all_mice_id, stages_sessionN, mouse_trainHist_all, mouseCre_numSess
# Set plane indeces for each area (V1 and LM)

def set_timetrace_flashesind_allmice_areas(samps_bef, samps_aft, frame_dur, doCorrs, all_sess):
    
    #%% Set the time trace (original and upsampled)

    # Note: the following two quantities are very similar (though time_orig is closer to the truth!): 
    # time_orig  and time_trace (defined below)
    # time_orig = np.mean(local_time_allOmitt, axis=1)

    time_trace, time_trace_new = upsample_time_imaging(samps_bef, samps_aft, 31.) # set time for the interpolated traces (every 3ms time points)
    #xnow = np.unique(np.concatenate((np.arange(0, -(samps_bef+1), -1), np.arange(0, samps_aft))))

    # set x tick marks                
    xmj = np.unique(np.concatenate((np.arange(0, time_trace[0], -1), np.arange(0, time_trace[-1], 1))))
    xmn = np.arange(-.5, time_trace[-1], 1)
#     xmn = np.arange(.25, time_trace[-1], .5)
    xmjn = [xmj, xmn]


    #%% Set the time of flashes and grays for the imaging traces

    ### NOTE: you should remove 0 from flashes_win_trace_index_unq_time
    ### because at time 0, there is no flash, there is omission!!

    flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, flashes_win_trace_index_unq, grays_win_trace_index_unq = \
        flash_gray_onset_relOmit(samps_bef, samps_aft, frame_dur)


    #%% Set some useful vars which you will need for most subsequent analaysis:
    # all_mice_id, stages_sessionN, mouse_trainHist_all, mouseCre_numSess

    all_mice_id, stages_sessionN, mouse_trainHist_all, mouseCre_numSess = all_sess_sum_stages_trainHist(all_sess, doCorrs)


    #%% Set plane indeces for each area (V1 and LM)

    if doCorrs==1:
        distinct_areas, i_areas = np.unique(all_sess[all_sess['mouse_id']==all_mice_id[0]]['area'].iloc[0], return_inverse=True) # take this info from any mouse ... it doesn't 
    else:
        distinct_areas, i_areas = np.unique(all_sess[all_sess['mouse_id']==all_mice_id[0]]['area'].values, return_inverse=True) # take this info from any mouse ... it doesn't matter... we got with mouse 0
    i_areas = i_areas[range(num_planes)]

    # distinct_areas = np.array(['VISl', 'VISp'])
    # i_areas = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # v1 : shows as VISp
    v1_ai = np.argwhere([distinct_areas[i].find('p')!=-1 for i in range(len(distinct_areas))]).squeeze()
    # LM : shown as VISl or LM
    lm_ai = np.argwhere([(distinct_areas[i].find('l')!=-1 or distinct_areas[i].find('L')!=-1) for i in range(len(distinct_areas))]).squeeze()

    inds_v1 = np.argwhere(i_areas==v1_ai).squeeze()
    inds_lm = np.argwhere(i_areas==lm_ai).squeeze()

    print('V1 plane indeces: ', inds_v1)
    print('LM plane indeces: ', inds_lm)
    #inds_lm = np.arange(num_depth) # we can get these from the following vars: distinct_areas and i_areas[range(num_planes)]
    #inds_v1 = np.arange(num_depth, num_planes)


    return time_trace, time_trace_new, xmjn, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, flashes_win_trace_index_unq, grays_win_trace_index_unq, all_mice_id, stages_sessionN, mouse_trainHist_all, mouseCre_numSess, distinct_areas, i_areas, inds_v1, inds_lm




#%% Set time windows, in frame units, for computing flash and omission evoked responese
# the output includes frame indices that are relative to "trace" begining; i.e. index on the trace whose time 0 is trace[samps_bef]. 

def set_frame_window_flash_omit(peak_win, flash_win, flash_win_timing, samps_bef, frame_dur):
    
    # np.floor was used here. On 04/29/2020 I changed it all to np.round, bc it is more accurate.
    # just define one of these below ... no need to do it 3 times! for now leaving it like this.
    
    # Compute peak_win_frames, ie window to look for peak in frame units
    peak_win_frames = samps_bef + np.round(peak_win / frame_dur).astype(int) # convert peak_win to frames (new way to code list_times)
    peak_win_frames[-1] = peak_win_frames[0] + np.round(np.diff(peak_win) / frame_dur).astype(int) # we redefine the upper limit of the window, otherwise the same window duration can lead to different upper limit values due to the division and flooring, and turning into int.
    list_times = np.arange(peak_win_frames[0], peak_win_frames[-1]) #+1) # [40, 41, 42, 43, 44, 45, 46, 47, 48]
    # for omit-evoked peak timing, compute it relative to samps_bef (which is the index of omission)

    # note: list_times goes upto peak_win_frames[-1]+1 but we dont add +1 for flash (see below)... omissions peak late, so we do it; but perhaps it makes sense to be consistent and just remove +1 from omissions too!
        # you removed it on 04/20/2020
        
        
    # same as above, now for flash-evoked responses :compute peak_win_frames, ie window to look for peak in frame units
    if ~np.isnan(flash_win).any():
        peak_win_frames_flash = samps_bef + np.round(flash_win / frame_dur).astype(int)
        peak_win_frames_flash[-1] = peak_win_frames_flash[0] + np.round(np.diff(flash_win) / frame_dur).astype(int) # we redefine the upper limit of the window, otherwise the same window duration can lead to different upper limit values due to the division and flooring, and turning into int.
        list_times_flash = np.arange(peak_win_frames_flash[0], peak_win_frames_flash[-1]) # [31, 32, 33, 34]  #, 35, 36, 37, 38, 39]
    else:
        list_times_flash = np.nan
        
        
    # window for measuring the timing of flash-evoked responses
    if ~np.isnan(flash_win_timing).any():
        peak_win_frames_flash_timing = samps_bef + np.round(flash_win_timing / frame_dur).astype(int)
        peak_win_frames_flash_timing[-1] = peak_win_frames_flash_timing[0] + np.round(np.diff(flash_win_timing) / frame_dur).astype(int) # we redefine the upper limit of the window, otherwise the same window duration can lead to different upper limit values due to the division and flooring, and turning into int.
        list_times_flash_timing = np.arange(peak_win_frames_flash_timing[0], peak_win_frames_flash_timing[-1]) # [30, 31, 32, 33]
    else:
        list_times_flash_timing = np.nan
        

    return list_times, list_times_flash, list_times_flash_timing

'''
#%% Set time windows, in frame units, for computing flash and omission evoked responese

def set_frame_window_flash_omit_new(peak_win, samps_bef, frame_dur):
    
    # np.floor was used here. On 04/29/2020 I changed it all to np.round, bc it is more accurate.
    
    # Compute peak_win_frames, ie window to look for peak in frame units
    peak_win_frames = samps_bef + np.round(peak_win / frame_dur).astype(int) # convert peak_win to frames (new way to code list_times)
    peak_win_frames[-1] = peak_win_frames[0] + np.round(np.diff(peak_win) / frame_dur).astype(int) # we redefine the upper limit of the window, otherwise the same window duration can lead to different upper limit values due to the division and flooring, and turning into int.
    list_times = np.arange(peak_win_frames[0], peak_win_frames[-1]) #+1) # [40, 41, 42, 43, 44, 45, 46, 47, 48]
    # for omit-evoked peak timing, compute it relative to samps_bef (which is the index of omission)

    return list_times
'''



def running_align_on_imaging(running_times, running_speed, yy_alltrace_time, doplots=0):
    # Take care of running trace: 
    # align its timing on yy_alltrace_time
    # turn it from stimulus-computer time resolution to imaging-computer time resolution, so it can be plotted on the same timescale as imaging.

    # set the running time, speed, and stimulus computer time resolution (which recorded the running)
    stim_computer_time_res = np.mean(np.diff(running_times))
    frdur = np.mean(np.diff(yy_alltrace_time)) # same as frame_dur
    runningSamps_per_imagingSamps = int(np.ceil(frdur / stim_computer_time_res)) # compare time resolution of imaging vs stimulus computer
    print(f'Max running samples per imaging samples: {runningSamps_per_imagingSamps}')

    # first set the begining and end indeces (of running_times) to take the part of running array that was recorded at the same time as imaging 
    running_times_firstInd_durImg = np.argmin(np.abs(running_times - yy_alltrace_time[0]))                
    running_times_lastInd_durImg = np.argmin(np.abs(running_times - yy_alltrace_time[-1])) # on what index of running_times, the last frame of imaging happened.
    # when i got running_times_win_yyalltrace_index, i noticed that the following was totally not needed: 
    # adjust indeces for the higher sampling rate of running vs imaging (n=runningSamps_per_imagingSamps) 
#                 running_times_firstInd_durImg = running_times_firstInd_durImg - (runningSamps_per_imagingSamps - 2) # I subtract 1 to be more conservative and not include running values that perhaps were recorded before imaging # another 1 is subtracted bc this index intself will also be included
#                 running_times_lastInd_durImg = running_times_lastInd_durImg + runningSamps_per_imagingSamps - 2 # I subtract 1 to be more conservative and not include running values that perhaps were recorded after imaging
    # now use the above indeces to take the part of running array that was recorded at the same time as imaging 
    running_times_duringImaging = running_times[running_times_firstInd_durImg : running_times_lastInd_durImg+1]
    running_speed_duringImaging = running_speed[running_times_firstInd_durImg : running_times_lastInd_durImg+1]

    # now convert the running times to imaging frame indeces in yy_alltrace_time
    # this is a bit slow
    running_times_win_yyalltrace_index = convert_time_to_frame(yy_alltrace_time, running_times_duringImaging) # find the frame number in yy_alltrace_time during which each element in running_times_duringImaging happened.

    # now downsample running elements to frame resolution
    # frame-bin index for each running time value 
    bins = np.arange(max(running_times_win_yyalltrace_index)+1+1)
    hist_inds = np.digitize(running_times_win_yyalltrace_index, bins)
    # average running elements within each bin (frame): downsample running # the following is a bit slow
    running_speed_duringImaging_downsamp = np.array([np.mean(running_speed_duringImaging[hist_inds==ihi]) for ihi in np.arange(1,len(bins))])
    if len(running_speed_duringImaging_downsamp) != len(yy_alltrace_time):
        sys.exit('something wrong!')

    # as a test see how many running elements occurred within 1 imaging frame
    hist, bin_edges = np.histogram(running_times_win_yyalltrace_index, bins) 
    print(f'{np.unique(hist[1:-1])}: number of running points that was recorded within an imaging frame (excluding the 1st and last frames)\n')
    print(f'{hist[0]}: number of running points that was recorded within the 1st imaging frame\n')
    print(f'{hist[-1]}: number of running points that was recorded within the last imaging frame\n')
#                 plt.plot(bin_edges[0:-1]+1/2., hist)

    ##### Final result: now running trace is on the same time scale as the imaging trace, ie:
    # each element of running_speed_duringImaging_downsamp corresponds to yy_alltrace_time

    if doplots:
        plt.plot(running_time, running_speed)                
        plt.plot(running_times_duringImaging, running_speed_duringImaging)
        plt.plot(yy_alltrace_time[running_times_win_yyalltrace_index], running_speed_duringImaging)
        plt.plot(yy_alltrace_time, running_speed_duringImaging_downsamp)                
        # df/f trace 
        plt.plot(yy_alltrace_time, yy_alltrace)


    return running_speed_duringImaging_downsamp

    #### Old method 
    '''
    running_win_yyalltrace_index = convert_time_to_frame(yy_alltrace_time, running_times)

    # find the first running element that corresponds to frame 0 of imaging (remember running recording may have started earlier than imaging)
    running_ind_fist = np.argwhere(running_win_yyalltrace_index == 1).squeeze()[0] - runningSamps_per_imagingSamps
    # find the last running element that corresponds to the last frame of imaging (remember running recording may have ended later than imaging)
    running_ind_last = np.argwhere(running_win_yyalltrace_index == np.unique(running_win_yyalltrace_index)[-2]).squeeze()[-1] + runningSamps_per_imagingSamps
    # take only the portion of running speed and time trace that was recorded during imaging
    running_win_yyalltrace_index_duringImaging = running_win_yyalltrace_index[running_ind_fist: running_ind_last+1]
    running_speed_duringImaging = running_speed[running_ind_fist: running_ind_last+1]

    plt.plot(running_time, running_speed)                
    plt.plot(running_win_yyalltrace_index, running_speed)
    plt.plot(running_win_yyalltrace_index_duringImaging, running_speed_duringImaging)

    # average running elements within a frame: downsample running
#                 hist, bin_edges = np.histogram(running_win_yyalltrace_index_duringImaging, bins=np.arange(max(running_win_yyalltrace_index_duringImaging)+1+1))                
#                 plt.plot(bin_edges[0:-1]+1/2., hist)
    # frame-bin index for each running time value 
    bins = np.arange(max(running_win_yyalltrace_index_duringImaging)+1+1)
    hist_inds = np.digitize(running_win_yyalltrace_index_duringImaging, bins)
    running_speed_duringImaging_downsamp = np.array([np.mean(running_speed_duringImaging[hist_inds==ihi]) for ihi in np.arange(1,len(bins))])
    running_win_yyalltrace_index_duringImaging_downsamp = bins[:-1]
    '''


#%% save the figure if it does not exist

def save_fig_if_notexist(aname, dir_planePair, nowStr, fmt='pdf'):    
    # aname is the main part of the figure name (not including the nowStr (ie the date)); eg :
    # figname = 'traces_x0_plane0to0_200320-100429.pdf'
    # aname = 'traces_x0_plane0to0_'
    
    list_figs = os.listdir(dir_planePair) # list of files in dir_planePair

    regex = re.compile(f'{aname}(.*).{fmt}')
    files = [string for string in list_figs if re.match(regex, string)]

    if len(files)>0: #os.path.isfile(os.path.join(dir_planePair, files)):
        files = files[-1] # take the last file
        print(f"Figure exists: {files}")
    else:
        fign = os.path.join(dir_planePair, f'{aname}{nowStr}.pdf')
        print(f'Saving figure:\n{fign}')
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)


#%% plot histogram for one or multiple arrays

def plothist(topall, nbins=20, doSmooth=0, colors = ('c','b','k'), linestyles=('-','-','-'), labs=('','',''), xlab='', ylab='', tit_txt=''):
    '''
    topall = a,b,c    
    tit_txt = 'sigFractNs %.2f, aucRandTest %.2f' %(sigFract_mw, auc_r2)

    labs = 'rand', 'test', 'train'
    colors = 'gray', 'k','b'    
    linestyles = 'dotted', 'solid', 'loosely dashed' 
    xlab = 'Explained variance'
    ylab ='Fraction neurons'
    doSmooth = 3
    nbins = 50
    '''
    
    if type(topall)==tuple:
        topc = np.concatenate((topall))
    else:
        topc = topall
        topall = topc,
        
    r = np.max(topc) - np.min(topc)
    binEvery = r/float(nbins)
    # set bins
    bn = np.arange(np.min(topc), np.max(topc), binEvery)
    bn[-1] = np.max(topc)#+binEvery/10. # unlike digitize, histogram doesn't count the right most value
    
    plt.subplots() #plt.subplot(h1) #(gs[0,0:2])    
    for itp in range(len(topall)):        
        hist, bin_edges = np.histogram(topall[itp], bins=bn)
        
        hist = hist/float(np.sum(hist))    
    #        if plotCumsum:
    #            hist = np.cumsum(hist)
        if doSmooth!=0:        
            hist = smooth(hist, doSmooth)
        # plot the center of bins
        plt.plot(bin_edges[0:-1]+binEvery/2., hist, color=colors[itp], label=labs[itp], linestyle=linestyles[itp]) # plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[0], alpha=.4, label=lab1)    
    
    fs = 12
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, fontsize=fs)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(tit_txt)
    
    makeNicePlots(plt.gca())
    
    
#%%
def zscore(traces, softNorm=False):
    # or use :
#    from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     train = scaler.fit_transform(traces_x00.T)
#        plt.plot(scaler.mean_)
#        plt.plot(scaler.scale_)

    m = np.mean(traces, axis=1) # neurons
    s = np.std(traces, axis=1) # neurons
    if softNorm==1:
        s = s + thAct     
    traces = ((traces.T - m) / s).T # neurons x times
    
    return traces, m, s


#%%
def evaluate_components(traces, N=5, robust_std=False):
    
    # traces: neurons x frames
    
    """ Define a metric and order components according to the probabilty if some "exceptional events" (like a spike). Suvh probability is defined as the likeihood of observing the actual trace value over N samples given an estimated noise distribution. 
    The function first estimates the noise distribution by considering the dispersion around the mode. This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349. 
    Then, the probavility of having N consecutive eventsis estimated. This probability is used to order the components. 

    Parameters
    ----------
    traces: ndarray
        Fluorescence traces 

    N: int
        N number of consecutive events


    Returns
    -------
    idx_components: ndarray
        the components ordered according to the fitness

    fitness: ndarray


    erfc: ndarray
        probability at each time step of observing the N consequtive actual trace values given the distribution of noise


    Created on Tue Aug 23 09:40:37 2016    
    @author: Andrea G with small modifications from farznaj

    """

#    import scipy  #    import numpy   #    from scipy.stats import norm   

#    import numpy as np
#    import scipy.stats as st

    
    T=np.shape(traces)[-1]
   # import pdb
   # pdb.set_trace()
    md = mode_robust(traces, axis=1)
    ff1 = traces - md[:, None]
    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:
        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, el in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1**2, 1) / Ns)
#


    # compute z value
    z = (traces - md[:, None]) / (3 * sd_r[:, None])
    # probability of observing values larger or equal to z given notmal
    # distribution with mean md and std sd_r
    erf = 1 - st.norm.cdf(z)
    # use logarithm so that multiplication becomes sum
    erf = np.log(erf)
    filt = np.ones(N)
    # moving sum
    erfc = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='full'), axis=1, arr=erf)
    erfc = erfc[:,:T]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    ordered = np.argsort(fitness)

    idx_components = ordered  # [::-1]# selec only portion of components
#    fitness = fitness[idx_components] % FN commented bc we want the indexing to match C and YrA.
#    erfc = erfc[idx_components] % FN commented bc we want the indexing to match C and YrA.

    return idx_components, fitness, erfc




def mode_robust(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    if axis is not None:
        fnc = lambda x: mode_robust(x, dtype=dtype)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:
                
                #            wMin = data[-1] - data[0]
                wMin = np.inf
                N = data.size / 2 + data.size % 2
                N = int(N)
                for i in range(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i

                return _hsm(data[j:j + N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode


#%% Find length of gaps between events

def find_event_gaps(evs):
    # evs: boolean; neurons x frames
    # evs indicates if there was an event. (it can be 1 for several continuous frames too)
    # eg: evs = (traces >= th_ag) # neurons x frames
    
    d = np.diff(evs.astype(int), axis=1) # neurons x frames
    begs = np.array(np.nonzero(d==1)) # 2 x sum(d==1) # first row is row index (ie neuron) in d; second row is column index (ie frame) in d.
    ends = np.array(np.nonzero(d==-1)) # 2 x sum(d==-1)
    #np.shape(begs)
    #np.shape(ends)
    
    ##### Note: begs_evs does not include the first event; so index i in gap_evs corresponds to index i in begs_evs and ends_evs.
    ##### however, gaps are truely gaps, ie number of frames without an event that span the interval between two frames.
    gap_evs_all = [] # # includes the gap before the 1st event and the gap before the last event too, in addition to inter-event gaps        
    gap_evs = [] # includes only gap between events 
    begs_evs = [] # index of event onsets, excluding the 1st events. (in fact these are 1 index before the actual event onset; since they are computed on the difference trace ('d'))
    ends_evs = [] # index of event offsets, excluding the last event. 
    bgap_evs = [] # number of frames before the first event
    egap_evs = [] # number of frames after the last event

    for iu in range(evs.shape[0]): 
#         print(iu)
        # sum(begs[0]==0)
    
        if sum(evs[iu]) > 0: # make sure there are events in the trace of unit iu
            begs_this_n = begs[1,begs[0]==iu] # indeces belong to "d" (the difference trace)
            ends_this_n = ends[1,ends[0]==iu]

            # gap between event onsets will be begs(next event) - ends(current event)

            if evs[iu, 0]==False and evs[iu, -1]==False: # normal case
                #len(begs_this_n) == len(ends_this_n): 
                b = begs_this_n[1:]
                e = ends_this_n[:-1]

                bgap = [begs_this_n[0] + 1] # after how many frames the first event happened
                egap = [evs.shape[1] - ends_this_n[-1] - 1] # how many frames with no event exist after the last event

            elif evs[iu, 0]==True and evs[iu, -1]==False: # first event was already going when the recording started.
                #len(begs_this_n)+1 == len(ends_this_n): 
                b = begs_this_n
                e = ends_this_n[:-1]

                bgap = []
                egap = [evs.shape[1] - ends_this_n[-1] - 1]

            elif evs[iu, 0]==False and evs[iu, -1]==True: # last event was still going on when the recording ended.
                #len(begs_this_n) == len(ends_this_n)+1:
                b = begs_this_n[1:]
                e = ends_this_n

                bgap = [begs_this_n[0] + 1]
                egap = []

            elif evs[iu, 0]==True and evs[iu, -1]==True: # first event and last event were happening when the recording started and ended.
    #            print('cool :)')
                b = begs_this_n
                e = ends_this_n

                bgap = []
                egap = []

            else:
                sys.exit('doesnt make sense! plot d to debug')


            gap_this_n = b - e
            gap_this = np.concatenate((bgap, gap_this_n, egap)).astype(int) # includes all gaps, before the 1st event, between events, and after the last event.
        
        
        else: # there are no events in this neuron; set everything to nan
            gap_this = np.nan
            gap_this_n = np.nan
            b = np.nan
            e = np.nan
            bgap = np.nan
            egap = np.nan

            
        gap_evs_all.append(gap_this) # includes the gap before the 1st event and the gap before the last event too.        
        gap_evs.append(gap_this_n) # only includes gaps between events: b - e      
        begs_evs.append(b)
        ends_evs.append(e)
        bgap_evs.append(bgap)
        egap_evs.append(egap)

    
    gap_evs_all = np.array(gap_evs_all) # size: number of neurons; each element shows the gap between events for a given neuron
    gap_evs = np.array(gap_evs)
    begs_evs = np.array(begs_evs)
    ends_evs = np.array(ends_evs)
    bgap_evs = np.array(bgap_evs)
    egap_evs = np.array(egap_evs)
    
    return gap_evs_all, begs_evs, ends_evs, gap_evs, bgap_evs, egap_evs, begs, ends


#%% shift and scale traces so they can be superimposed and easily compared.
# Note: this code needs more work to take care of cases where max(trace) is 0 or inf.
    
def supimpose(tt):
    # tt: neurons x frames
    p = np.nanmean(tt, axis=0)
    p = p-p[0]
#    p = p / min(max(p),40) 
    p = p / max(p) # this needs work... take care of 0, also inf, 
    return p


#%%  Pool all sessions and layers for each area, do this for each mouse

def pool_sesss_planes_eachArea(area, y):
    # area: ['LM', 'LM', 'LM', 'LM', 'VISp', 'VISp', 'VISp', 'VISp', 'LM', 'LM', 'LM', 'LM', 'VISp', 'VISp', 'VISp', 'VISp'] # (8*num_sessions)
    # y : (8*num_sessions) x time  or   # (8*num_sessions) 

    # set area indeces
#         area = trace_peak_allMice.iloc[im]['area'] # (8*num_sessions)
    distinct_areas, i_areas = np.unique(area, return_inverse=True)
    #    print(distinct_areas)
    if len(distinct_areas) > 2:
        sys.exit('There should not be more than two areas!! Fix the area names!')

    # below has size: num_areas x (num_layers_per_area x num_sessions) x nFrames_upsampled
    # so, 2 x (4 x num_sessions) x nFrames_upsampled
    # For each area, first take all the layers of session 1, then take all the layers of session 2
    y_pooled_sesss_planes_eachArea = np.array([y[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x nFrames_upsampled

    return y_pooled_sesss_planes_eachArea, distinct_areas, i_areas


#%% For each layer (depth), pool data across sessions and areas

def pool_sesss_areas_eachDepth(planes_allsess, y, num_depth=4):
    # planes_allsess: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] # (8*num_sessions)
    # y: (8*num_sessions) x time   or   (8*num_sessions) 
    # num_depth: 4
    
    y_pooled_sesss_areas_eachDepth = []
    for idepth in range(num_depth): # idepth = 0
        # merge data with the same depth across 2 areas 
        # For each depth, first take all the areas of session 1, then take all the areas of session 2
        b = np.logical_or(planes_allsess==idepth, planes_allsess==idepth + num_depth)
        a = y[b] #np.array([t[b]]).squeeze() # (2 x num_sess) x nFrames_upsampled
        y_pooled_sesss_areas_eachDepth.append(a)
    y_pooled_sesss_areas_eachDepth = np.array(y_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2

    return y_pooled_sesss_areas_eachDepth

    
#%% Add to the plots the following: flash/ gray screen lines , proper tick marks, and legend

def plot_flashLines_ticks_legend(lims, H, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, x='', bbox_to_anchor=(1, .7), ylab='% Classification accuracy', xmjn='', xlab='Time after omission (sec)', omit_aligned=0):
#     h1 = plt.plot()
#     plot_flashLines_ticks_legend([], h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)        
    
    if len(lims)!=0: # set lims=[] when lims is not known.
        mn = lims[0] # np.round(np.nanmin(av_test_shfl_avSess_eachP)) # 45
        mx = lims[1] # np.round(np.nanmax(av_test_data_avSess_eachP)) # 80
        #    rg = (mx - mn) / 10.
    else:
        mn = plt.gca().get_ylim()[0];
        mx = plt.gca().get_ylim()[1];
        
    # mark omission onset
    plt.vlines([0], mn, mx, color='k', linestyle='--')
    
    # mark flash duration with a shaded area
    ### NOTE: you should remove 0 from flashes_win_trace_index_unq_time
    ### because at time 0, there is no flash, there is omission!!    
    flashes_win_trace_index_unq_time0 = flashes_win_trace_index_unq_time
    
    flash_dur = .25 #np.unique(grays_win_trace_index_unq_time - flashes_win_trace_index_unq_time0)

    if omit_aligned:
        omit_ind = np.argwhere(flashes_win_trace_index_unq_time0==0).squeeze()
    #    flashes_win_trace_index_unq_time = np.delete(flashes_win_trace_index_unq_time, omit_ind)
        flashes_win_trace_index_unq_time_new = np.delete(flashes_win_trace_index_unq_time0, omit_ind)
    else:
        flashes_win_trace_index_unq_time_new = flashes_win_trace_index_unq_time0
    
    for i in range(len(flashes_win_trace_index_unq_time_new)):
        plt.axvspan(flashes_win_trace_index_unq_time_new[i], flashes_win_trace_index_unq_time_new[i] + flash_dur, alpha=0.2, facecolor='y')
    
    '''
    # mark the onset of flashes
    plt.vlines(flashes_win_trace_index_unq_time_new, mn, mx, color='y', linestyle='-.', linewidth=.7)
    # mark the onset of grays
    plt.vlines(grays_win_trace_index_unq_time, mn, mx, color='gray', linestyle=':', linewidth=.7)
    '''
    if xmjn=='':    
        xmj = np.unique(np.concatenate((np.arange(0, x[0], -.5), np.arange(0, x[-1], .5))))
        xmn = np.arange(.25, x[-1], .5)
    else:
        xmj = xmjn[0]
        xmn = xmjn[1]
    
    ax = plt.gca()
    
    ax.set_xticks(xmj); # plt.xticks(np.arange(0,x[-1],.25)); #, fontsize=10)
    ax.set_xticklabels(xmj, rotation=45)       
    
    ax.xaxis.set_minor_locator(ticker.FixedLocator(xmn))
    ax.tick_params(labelsize=10, length=6, width=2, which='major')
    ax.tick_params(labelsize=10, length=5, width=1, which='minor')
    #    plt.xticklabels(np.arange(0,x[-1],.25))
    
    if len(H)!=0:        
        plt.legend(handles=H, loc='center left', bbox_to_anchor=bbox_to_anchor, frameon=False, handlelength=1, fontsize=12)
    plt.ylabel(ylab, fontsize=12)
    plt.xlabel(xlab, fontsize=12)  
    if len(lims)!=0:
        if ~np.isnan(lims).any():
            plt.ylim(lims)
#    plt.legend(handles=[h1[0], h1[1], h1[2], h1[3], h1[4], h1[5], h1[6], h1[7], h2], loc='center left', bbox_to_anchor=(1, .7), frameon=False, handlelength=1, fontsize=12)
    
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)


    
#%% Look for a file in a directory; if desired, sort by modification time, and only return the latest file.

# example inputs:
#    name = 'all_sess_%s_.' %(analysis_name) 
#    name = 'Yneuron%d_model_' %neuron_y
    
# Set the name of all_sess h5 files made in *_init scripts which ran the analysis of interest, named analysis_name, and exist in dir_now

def all_sess_set_h5_fileName(name, dir_now, all_files=0):

    regex = re.compile(name) # + '.h5')
#    regex = re.compile(aname + '(.*).hdf5') # note: "aname" does not include the directory path; it's just the file name.

    l = os.listdir(dir_now)
     
    h5_files = [string for string in l if re.match(regex, string)] # string=l[0]
    
    if len(h5_files)==0:
        print('Error: no h5 file exists!!!')
        allSessName = ''
        print(regex)
        
    if all_files==0: # only get the latest file, otherwise get all file names
        # Get the modification times of the existing analysis folders
        modifTimes = [os.path.getmtime(os.path.join(dir_now, h5_files[i])) for i in range(len(h5_files))]
        
        # Find all the old analysis folders                               
        if len(modifTimes) > 1:
            h5_files = np.array(h5_files)[np.argsort(modifTimes).squeeze()]
            print(h5_files)
        
        
        if len(h5_files)==0:
            print('h5 file does not exist! (run svm_init to call svm_plots_init and save all_sess)')
        elif len(h5_files)>1:
            print('More than 1 h5 file exists! Using the latest file')
            allSessName = os.path.join(dir_now, h5_files[-1])            
        else:
            allSessName = os.path.join(dir_now, h5_files[0])            
        print(allSessName)
    
    else:
        allSessName = []
        for i in range(len(h5_files)):
            allSessName.append(os.path.join(dir_now, h5_files[i]))
        print(allSessName)

    # read hdf file    
#    all_sess = pd.read_hdf(allSessName, key='all_sess') #'svm_vars')        ## Load all_sess dataframe
#    input_vars = pd.read_hdf(allSessName, key='input_vars')     ## Load input_vars dataframe

    return allSessName, h5_files



#%% THIS IS GREAT BUT IT TAKES SOME TIME: Get the sequence of stages for each mouse by making a sql query from LIMS.... this will also show all training days before ophys sessions.
'''
def set_mouse_trainHist_all2(all_mice_id, saveResults=1):
    
    from get_mouse_pkls import *

    mouse_trainHist_all2 = pd.DataFrame([], columns=['mouse_id', 'date', 'stage'])
    
    for mouse_id in all_mice_id:
        print(mouse_id)
        # mouse_id = 449653    
        pkl_all_sess = get_mouse_pkls(mouse_id) # make a sql query from LIMS.
         
        # For each mouse go through all pkl files to get the stages
        mouse_trainHist2 = pd.DataFrame([], columns=['mouse_id', 'date', 'stage']) #list()
        for i in range(len(pkl_all_sess)):
            print('pickle file %d / %d' %(i, len(pkl_all_sess)))
            pkl = pd.read_pickle(pkl_all_sess[i]['pkl_path'])
            try:
                stage = pkl['items']['behavior']['params']['stage']
            except:
                stage = ''            
    
            try:
                date = str(pkl['start_time'].date())
            except:
                date = str(pkl['startdatetime'].date())            
    #            import datetime
    #            datetime.datetime.fromtimestamp(1548449865.568)
    
    
            mouse_trainHist2.at[i, ['mouse_id', 'date', 'stage']] = mouse_id, date, stage
            
        mouse_trainHist_all2 = mouse_trainHist_all2.append(mouse_trainHist2)

        
        #### save the variable
        if saveResults:
            print('Saving .h5 file')
            dir_Farz = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
            name = 'mouse_trainHist_all2'
            Name = os.path.join(dir_Farz, name + '.h5') # os.path.join(d, svmn+os.path.basename(pnevFileName))
            print(Name)
        
            # Save to a h5 file                    
            mouse_trainHist_all2.to_hdf(Name, key='mouse_trainHist_all2', mode='w')    
        
        return mouse_trainHist_all2
'''    


#%% After all_sess is set, run this function to set the following useful vars which you will need for most subsequent analaysis
# all_mice_id, stages_sessionN, mouse_trainHist_all, mouseCre_numSess

def all_sess_sum_stages_trainHist(all_sess, doCorrs):    
    #%% Show the number of sessions for each stage 
     
    # all_sess[all_sess.stage == 'OPHYS_5_images_A_passive']
    #list(np.unique(all_sess.stage))
    
    [all_stages, stages_inds, unique_inverse, unique_counts] = np.unique(all_sess.stage, return_index=True, return_inverse=True, return_counts=True)
    #[sum(all_sess.untitled0stage == all_stages[i]) for i in range(len(all_stages))]
    stages_sessionN = pd.DataFrame(unique_counts, index=[np.unique(all_sess.stage)], columns=['n_sessions'])
    print(stages_sessionN)
    
    # number of neurons per experiment
    # all_sess.n_neurons
    
    
    #%% For all analysis sessions, show mouse_id, date and stage ... (this shows the sequence of stages for each mouse)
    
    all_mice_id = np.unique(all_sess.mouse_id) # remember np.unique also sorts the values! so the following two are NOT the same: all_mice_id[0] and all_sess.iloc[0].mouse_id
    # how much data do you have from each mouse?
    
    mouse_trainHist_all = pd.DataFrame([], columns=['mouse_id', 'date', 'session_id', 'stage']) # total number of sessions x 3
    
    for cnt, i in enumerate(all_mice_id):
        m = all_sess[all_sess.mouse_id==i].mouse_id.values
        s = all_sess[all_sess.mouse_id==i].stage.values
        d = all_sess[all_sess.mouse_id==i].date.values
        ss = all_sess[all_sess.mouse_id==i].session_id.values
    
        # take data from the 1st experiment of each session (because date and stage are common across all experiments of each session)
        # NOTE: below doesnt work when docorr=1, because each row of all_sess is for a session and includes all the 8 experiments 

        mouse_trainHist = pd.DataFrame([], columns=['mouse_id', 'date', 'session_id', 'stage'])

        if doCorrs==0:
            r = np.arange(0,len(m),8)
        else:
            r = np.arange(0,len(m))
            
        for ii in r:
            mouse_trainHist.at[ii, ['mouse_id', 'date', 'session_id', 'stage']] = m[ii] , d[ii] , ss[ii] , s[ii]  # m.iloc[np.arange(0,len(m),8)].values , d.iloc[np.arange(0,len(d),8)].values , s.iloc[np.arange(0,len(s),8)].values
        
        mouse_trainHist_all = mouse_trainHist_all.append(mouse_trainHist)
    
    print(mouse_trainHist_all)
    
    '''
    mouse_trainHist = pd.DataFrame([], index=all_mice_id, columns=['date', 'stage'])
    for cnt, i in enumerate(all_mice_id):
        s = all_sess[all_sess.mouse_id==i].stage
        d = all_sess[all_sess.mouse_id==i].date
        mouse_trainHist.at[i, ['date','stage']] = d.iloc[np.arange(0,len(d),8)].values , s.iloc[np.arange(0,len(s),8)].values
    '''
    
    
    #%% THIS IS GREAT BUT IT TAKES SOME TIME: Get the sequence of stages for each mouse by making a sql query from LIMS.... this will also show all training days before ophys sessions.
    
    if 0:
        mouse_trainHist_all2 = set_mouse_trainHist_all2(all_mice_id, saveResults=1)                
    
    
    #%% For each mouse_id, show the cre line and the number of sessions
    
    if doCorrs==0:
        num_sessions_per_mouse = np.ceil(np.array([sum(all_sess.mouse_id == all_mice_id[i]) for i in range(len(all_mice_id))]) / 8.)
    else:
        num_sessions_per_mouse = np.ceil(np.array([sum(all_sess.mouse_id == all_mice_id[i]) for i in range(len(all_mice_id))]))
        
    mouseCre_numSess = pd.DataFrame([], columns=['cre', 'num_sessions'])
    for i in range(len(all_mice_id)):
        mouse_id = all_mice_id[i]
        mouseCre_numSess.at[mouse_id, ['cre','num_sessions']] = all_sess[all_sess.mouse_id == all_mice_id[i]].cre.iloc[0] , num_sessions_per_mouse[i]
    
    #print(mouseCre_numSess)
    print(mouseCre_numSess.sort_values(by='cre'))
    # show total number of sessions per cre line
    print(mouseCre_numSess.groupby('cre').num_sessions.sum())
    
    
    return all_mice_id, stages_sessionN, mouse_trainHist_all, mouseCre_numSess


#%% Get data from a given plane across all sessions (assuming that the same order of planes exist in all sessions, ie for instance plane n is always the ith experiment in all sessions.)

def set_y_this_plane_allsess(y, num_sessions, takeAve=0): # set takeAve to 1 for meanX_allFrs (so we average across neurons)
    
    
    # PERHAPS just forget about all of this below and do a reshape, you can double check reshape by doing the same thing on plane, area, depth 
    
    
    # NOTE: you are using every 8th index to find data from the same plane, but I really think
    # you should use .at[iplane] instead. Because when you set all_sess, you use index of planes as the
    # index of all_sess data frame.... this will mean that y (the input to this function) will have to be
    # a data frame (you are currently passing it as a numpy array)
    
#    y_this_plane_allsess_allp = pd.DataFrame([], columns=['area', 'depth', 'y_this_plane'])
    y_this_plane_allsess_allp = []
    # len(y_this_plane_allsess_allp) = num_planes (8) # each element is for a given plane
    # len(y_this_plane_allsess_allp[0]) = num_sessions # each element is for a given sessions 
    # len(y_this_plane_allsess_allp[0][0]) = 930 (30 frames of svm*31 (interpolations)) # each element is y for a frame 

    for iplane in range(num_planes): # iplane=0 

#        area = all_sess[all_sess.mouse_id==mouse_id].area.at[iplane] # num_sess
#        depth = all_sess[all_sess.mouse_id==mouse_id].depth.at[iplane] # num_sess
#        if np.ndim(area)==0: # happens when len(session_stages)==1
#            area = [area]
#            depth = [depth]        
        
        ##### get data from a given plane across all sessions
        '''
        onePlane_allSess = np.arange(iplane, y.shape[0], num_planes)                    
        y_this_plane_allsess = y[onePlane_allSess] # num_sess; each element: num_frames  # np.shape(y_this_plane_allsess)        
        '''
        y_this_plane_allsess = y.at[iplane] # num_sess        
        # If y_this_plane_allsess is not a vector, take average across the 2nd dimension         
#        if np.shape(y_this_plane_allsess)!=() and type(y_this_plane_allsess[0]) != str:
#            # we need the following in case there are invalid sessions which will be nan            
#            takeAve = 1
#        else:
#            takeAve = 0
        

        # when the mouse had only one session, we need the following to take care of the shape of y_this_plane_allsess
        if np.shape(y_this_plane_allsess)==(): # type(y_this_plane_allsess) == int: --> this didnt work bc populatin_sizes_to_try is of type numpy.int64
            if type(y_this_plane_allsess) == str:
                y_this_plane_allsess = np.array([y_this_plane_allsess])
            else:
                y_this_plane_allsess = np.array([np.float(y_this_plane_allsess)])
                
        if num_sessions < 2:
            y_this_plane_allsess = y_this_plane_allsess[np.newaxis,:]            
#        print(np.shape(y_this_plane_allsess), y_this_plane_allsess.dtype)
        

        # If y_this_plane_allsess is not a vector, take average across the 2nd dimension         
        if takeAve==1:
            nonNan_sess = np.array([np.sum(~np.isnan(y_this_plane_allsess[isess])) for isess in range(num_sessions)])            
            len_trace = np.shape(y.values[0])[0] #len(y.values[0]) #y[0].shape
#            nonNan_sess_ind0 = np.argwhere(nonNan_sess > 0)
#            if len(nonNan_sess_ind0) > 0:
#                nonNan_sess_ind = nonNan_sess_ind0[0][0] # .squeeze()
#                if np.ndim(y_this_plane_allsess[nonNan_sess_ind]) > 1: # eg in meanX, the dimenstions are frames x neurons            
            aa = np.full((num_sessions, len_trace), np.nan) # []
            for isess in range(num_sessions): # isess = 0
                if nonNan_sess[isess] > 0:
                    aa[isess] = np.mean(y_this_plane_allsess[isess], axis = 1)
#                else:
#                    a = np.full((len_trace), np.nan)
#                    a = np.full((y_this_plane_allsess[nonNan_sess_ind].shape[0]), np.nan)
                    
#                aa.append(a)
            y_this_plane_allsess = aa #np.array(aa)
                
        '''
        y_this_plane_allsess = np.vstack(y_this_plane_allsess).astype(np.float) # num_sess x 930 # to change dtype from object to regular 2d array        
        '''
        y_this_plane_allsess = np.vstack(y_this_plane_allsess) # num_sess x nFrames
#        print(np.shape(y_this_plane_allsess), y_this_plane_allsess.dtype)
        
        
        y_this_plane_allsess_allp.append(y_this_plane_allsess) # .squeeze()
#        y_this_plane_allsess_allp.at[iplane] = area, depth, y_this_plane_allsess
 
    y_this_plane_allsess_allp = np.array(y_this_plane_allsess_allp)

    return y_this_plane_allsess_allp




#%% Align on omission trials

# local_fluo_traces # neurons x frames
# local_time_traces # frame times in sec. Volume rate is 10 Hz. Are these the time of frame onsets?? (I think yes... double checking with Jerome/ Marina.) # dataset.timestamps['ophys_frames'][0]             
# samps_bef (=40) frames before omission ; index: 0:39
    # omission frame ; index: 40
    # samps_aft - 1 (=39) frames after omission ; index: 41:79

def align_trace_on_event(local_fluo_traces, local_time_traces, samps_bef, samps_aft, list_omitted):
    
    num_neurons = local_fluo_traces.shape[0]
    
    # Keep a matrix of omission-aligned traces for all neurons and all omission trials    
    local_fluo_allOmitt = np.full((samps_bef + samps_aft, num_neurons, len(list_omitted)), np.nan) # time x neurons x omissions_trials 
    local_time_allOmitt = np.full((samps_bef + samps_aft, len(list_omitted)), np.nan) # time x omissions_trials

    # Loop over omitted trials to align traces on omissions
    num_omissions = 0 # trial number
    for iomit in range(len(list_omitted)): # # indiv_time in list_omitted: # 

        indiv_time = list_omitted[iomit] # list_omitted.iloc[iomit]
        local_index = np.argmin(np.abs(local_time_traces - indiv_time)) # the index of omission on local_time_traces               

        be = local_index - samps_bef
        af = local_index + samps_aft

        if ~np.logical_and(be >= 0 , af <= local_fluo_traces.shape[1]): # make sure the omission is at least samps_bef frames after trace beigining and samps_aft before trace end. 
            print('Event %d at time %f cannot be analyzed: %d timepoints before it and %d timepoints after it!' %(iomit, indiv_time, be, af)) 

        try: # Align on omission
            local_fluo_allOmitt[:,:, iomit] = local_fluo_traces[:, be:af].T # frame x neurons x omissions_trials (10Hz)
            local_time_allOmitt[:, iomit] = local_time_traces[be:af] - indiv_time # frame x omissions_trials
            
            # local_time_allOmitt is frame onset relative to omission onset. (assuming that the times in local_time_traces are frame onsets.... still checking on this, but it seems to be the rising edge, hence frame onset time!)
            # so local_time_allOmitt will be positive if frame onset is after omission onset.
            #
            # local_time_allOmitt shows the time of the frame that is closest to omission relative to omission.
            # (note, the closest frame to omission is called omission frame).
            # eg if local_time_allOmitt is -0.04sec, it means, the omission frame starts -.04sec before omission.
            # or if local_time_allOmitt is +0.04sec, it means, the omission frame starts +.04sec before omission.
            #
            # Note: the following two quantities are very similar (though time_orig is closer to the truth!): 
            # time_orig  and time_trace (defined below)
            # time_orig = np.mean(local_time_allOmitt, axis=1)                            
            # time_trace, time_trace_new = upsample_time_imaging(samps_bef, samps_aft, 31.) # set time for the interpolated traces (every 3ms time points)
            #
            # so we dont save local_time_allOmitt, although it will be more accurate to use it than time_trace, but the
            # difference is very minimum

#             num_omissions = num_omissions + 1
            
        except Exception as e:
            print(f'Omission alignment failed; omission index: {iomit}, omission frame: {local_index}, starting and ending frames: {be, af}') # indiv_time, 
#             print(indiv_time, num_omissions, be, af, local_index)
#             print(e)


    # remove the last nan rows from the traces, which happen if some of the omissions are not used for alignment (due to being too early or too late in the session) 
    # find nan rows
    mask_valid_trs = np.nansum(local_fluo_allOmitt, axis=(0,1))!=0
    if sum(mask_valid_trs==False) > 0:
        local_fluo_allOmitt = local_fluo_allOmitt[:,:,mask_valid_trs]
        local_time_allOmitt = local_time_allOmitt[:,mask_valid_trs]

    num_omissions = sum(mask_valid_trs)

#     if len(list_omitted)-num_omissions > 0:
#         local_fluo_allOmitt = local_fluo_allOmitt[:,:,0:-(len(list_omitted)-num_omissions)]
#         local_time_allOmitt = local_time_allOmitt[:,0:-(len(list_omitted)-num_omissions)]

        
    return local_fluo_allOmitt, local_time_allOmitt



#%% Set time (sec) array for imaging data, original and upsampled (to be used for interpolatolation of imaging data) 

def upsample_time_imaging(samps_bef, samps_aft, mult=31.):
    
    stp = frame_dur / mult
    # To align on omissions, get 40 frames before omission and 39 frames after omission
    samps_bef_time = (samps_bef+1) * frame_dur # 1 is added bc below we do np.arange(0,-samps_bef), so we get upto one value below samps_bef
    samps_aft_time = samps_aft * frame_dur # frames_after_omission in svm_main # we trained the classifier until 30 frames after omission
    
    x = np.unique(np.concatenate((np.arange(0, -samps_bef_time, -frame_dur)[0:samps_bef+1], np.arange(0, samps_aft_time, frame_dur)[0:samps_aft])))
    xnew = np.unique(np.concatenate((np.arange(0, -samps_bef_time, -stp)[0:int((samps_bef+1)*mult)], np.arange(0, samps_aft_time, stp)[0:int(samps_aft*mult)]))) #     xnew = np.unique(np.concatenate((np.arange(0, -samps_bef_time, -stp), np.arange(0, samps_aft_time, stp))))   
#    x = np.arange(0, frame_dur*len_y, frame_dur) # time (sec) # np.arange(0, len(y))        
#    xnew = np.arange(0, frame_dur*len_y, stp) # np.arange(0, len(y), stp)
    
    return x, xnew

    
#%% interpolate traces that have imaging time resolution (93ms) to 3ms (it is otherwise very hard to mark where exactly flashes happened.)

def interp_imaging(y, samps_bef, samps_aft, kind='linear'):
    
    from scipy import interpolate

    x, xnew = upsample_time_imaging(samps_bef, samps_aft, 31.)
    
    f = interpolate.interp1d(x, y, kind=kind, fill_value='extrapolate')
    ynew = f(xnew)

    return ynew, xnew, x


#%% For traces that are aligned on omission, this funciton will give us the time (and frame number) of flashes and gray screens

def flash_gray_onset_relOmit(samps_bef, samps_aft, frame_dur, flash_dur=.25, gray_dur=.5):
    
    # To align on omissions, get 40 frames before omission and 39 frames after omission
    samps_bef_time = (samps_bef+1) * frame_dur # 1 is added bc below we do np.arange(0,-samps_bef), so we get upto one value below samps_bef
    samps_aft_time = samps_aft * frame_dur # frames_after_omission in svm_main # we trained the classifier until 30 frames after omission
    
    flash_gray_dur = flash_dur + gray_dur # .75 # sec (.25 flash + .5 gray)
    
    # times (sec) of flash onset, when aligned on omission (0 is the onset of omission)
#    flashes_win_trace_index_unq_time = np.unique(np.concatenate((np.arange(samps_bef_time, 0, -flash_gray_dur), [gray_dur], \
#                                                            np.arange(gray_dur, samps_aft_time, flash_gray_dur))))
    ### NOTE: you should remove 0 from flashes_win_trace_index_unq_time
    ### because at time 0, there is no flash, there is omission!!
    flashes_win_trace_index_unq_time = np.unique(np.concatenate((np.arange(0, -samps_bef_time, -flash_gray_dur), \
                                                            np.arange(0, samps_aft_time, flash_gray_dur))))
    
    # times (sec) of gray onset, when aligned on omission (0 is the onset of omission)
    grays_win_trace_index_unq_time = np.unique(np.concatenate((np.arange(0+flash_dur, -samps_bef_time, -flash_gray_dur), \
                                                            np.arange(0+flash_dur, samps_aft_time, flash_gray_dur))))
        
    # same as above, except in frame
    # below will give us [0, 8, 16, 24, 32]: these are the frames of flash onset, when aligned on omission (0 is the onset of omission)
    flashes_win_trace_index_unq = flashes_win_trace_index_unq_time / frame_dur
    # below will give us [2.7, 10.8, 18.8 , 26.9]: these are the frames of flash onset, when aligned on omission (0 is the onset of omission)
    grays_win_trace_index_unq = grays_win_trace_index_unq_time / frame_dur

    return flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, flashes_win_trace_index_unq, grays_win_trace_index_unq #= flash_gray_onset_relOmit(samps_bef, samps_aft, frame_dur)


#%% Play the movie

def movie_play(dirmov, frames2show=[0,500], maxPercentile=99):
    # file: motion_corrected_video.h5
    #dirmov = '/allen/programs/braintv/production/visualbehavior/prod0/specimen_837581585/ophys_session_880709154/ophys_experiment_881003491/processed/motion_corrected_video.h5'
    
    # maxPercentile: use this to increase the contrast. Change the max pixel intensity to this values. 
    
#    %matplotlib qt
    get_ipython().magic(u'matplotlib qt')    
    
    # read the h5py file
    f = h5py.File(dirmov, 'r')    #mov = f[list(f.keys())[0]]
    mov = f.get('data') 
#    type(mov), np.shape(mov)    
    
    # plot average projection
#    a = np.mean(mov[frs], axis=0)    
#    plt.imshow(a)
    
    frs = np.arange(frames2show[0], frames2show[1])
            
    # use this to increase the contrast
    vmax = np.percentile(mov[frs,:,:].flatten(), maxPercentile)
    
    # plot the movie
    fig, ax = plt.subplots(1,1)
    # plot the 1st frames
    myobj = plt.imshow(mov[frames2show[0]],'gray', aspect='auto', vmin=0, vmax=vmax)
    # plot the rest of the frames
    for i in frs[1:]: #mov.shape[0]): :
    #    ax.imshow(mov[i],'gray', aspect='auto')#, extent = [0,1,0,1])     
        myobj.set_data(mov[i])
        plt.pause(.05)     #    plt.show()     #    plt.draw()
        plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    #    seaborn.despine()#left=True, bottom=True, right=False, top=False)
        

#%%
def get_stage_mongo(session_id):

    DB = mongo.db.ophys_session_log 
    db_cursor_sess = DB.find({"id":session_id})

   
    ######### get stage from mongo
    a = list(db_cursor_sess[0]['name'])
    try:
        ai = (np.argwhere([a[i]=='_' for i in range(len(a))]).flatten()[-1]).astype(int)
        stage_mongo = str(db_cursor_sess[0]['name'][ai+1:])
    except Exception as e: # some sessions have unconventional names on mongo; e.g.  990940005 (they dont seem to have QC); assing "test" to their stage name! 
        stage_mongo = 'test'
        print(e)        
        print('Session_id %s: weird stage name exists; lets call it "test"!' %str(session_id))
        
        
    return stage_mongo


#%% 
def is_session_valid(session_id, list_mesoscope_exp=[], exp_date_analyze=[]): 
    # exp_date_analyze: set to [] to analyze session_id regardless of its experiment data.
    # otherwise run the code below only if the session was recorded after exp_date_analyze
    
    session_id = int(session_id)
    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'


    DB = mongo.db.ophys_session_log 
    db_cursor_sess = DB.find({"id":session_id})


    #########  get workflow_state from mongo
    workflow_state = pd.DataFrame()
    for i in range(8): # i=0
        workflow_state.at[i, 'workflow_state'] = db_cursor_sess[0]['ophys_experiments'][i]['workflow_state']

   
    ######### get stage from mongo
    stage_mongo = get_stage_mongo(session_id)
#    a = list(db_cursor_sess[0]['name'])
#    ai = (np.argwhere([a[i]=='_' for i in range(len(a))]).flatten()[-1]).astype(int)
#    stage_mongo = str(db_cursor_sess[0]['name'][ai+1:])
        

    ######### get date from mongo        
#    DB = mongo.qc.metrics 
    f = 0
    try:
        exp_date = db_cursor_sess[0]['date_of_acquisition'].strftime('%Y-%m-%d')          
        # Get date from lims query:
        # Session_obj.data_pointer['date_of_acquisition'].strftime('%Y-%m-%d')
        
#        db_cursor_sess = DB.find({"lims_id":session_id})
#        exp_date = db_cursor_sess[0]['lims_ophys_session']['date_of_acquisition'].strftime('%Y-%m-%d')
#        e00 = ''
    except Exception as e00:
        exp_date = np.nan
        print(session_id, e00, stage_mongo)        
#    db_cursor_sess[0]['name']
    

    # you should add mouse_id to validity_log: db_cursor_sess[0]['external_specimen_name']
    validity_log = pd.DataFrame([], columns=['session_id', 'lims_id', 'date', 'stage_metadata', 'stage_mongo', 'valid', 'log'])    

    if exp_date_analyze==[]:
        exp_date_analyze = 0
        
    if int(datetime.datetime.strptime(exp_date, '%Y-%m-%d').strftime('%Y%m%d')) > exp_date_analyze:
            
        # Get list of experiments using ophysextractor       
        if list_mesoscope_exp==[]:
            Session_obj = LimsOphysSession(lims_id=session_id)
            list_mesoscope_exp = Session_obj.data_pointer['ophys_experiment_ids']
            
        ##########################################################     
        ######### Go through experiments in each session #########        
        ##########################################################     
        DB = mongo.qc.metrics
        cnt = -1
        
        for indiv_id in list_mesoscope_exp: # indiv_id = list_mesoscope_exp[0] 
            
            print('\t%d' %indiv_id) 
            indiv_id = int(indiv_id)
            
            f = 0        
            db_cursor = DB.find({"lims_id":indiv_id})
            
            indiv_data = {}
            cnt = cnt+1        
    
    
            ############## set dataset       
            try:
                dataset = VisualBehaviorOphysDataset(indiv_id, cache_dir=cache_dir) # some sessions don't have roi_id, so dff_traces cannot be set
                e0 = ''
                
            except Exception as e:       
                e0 = e.args # 'cannot set dataset'
                print('\t session %d, experiment %d: %s' %(session_id, indiv_id, e0))
                session_valid = False #session_valid.at[indiv_id] = False
                f = 1
            
    
            ############## get dff_traces from dataset               
            try:
                if e0 == '':
                    indiv_data['fluo_traces'] = dataset.dff_traces
                    e1 = ''
                else:
                    e1 = 'cannot set dataset.dff_traces'
            except Exception as e:
                e1 = e.args
                session_valid = False #session_valid.at[indiv_id] = False
                f = 1
                if cnt==0:
                    print('\t session %d, experiment %d: %s' %(session_id, indiv_id, e1))
    
    
            ############## get stage variable from dataset_metadata           
            try:
                if e0 == '':
                    local_meta = dataset.get_metadata()
                    stage_metadata = local_meta['stage'].values[0]
    #                exp_date = local_meta['experiment_date'].values[0]                
    #                indiv_data['imaging_depth'] = db_cursor[0]['lims_ophys_experiment']['depth']
                    e2 = ''
                else:
                    stage_metadata = ''
                    e2 = 'cannot set dataset.get_metadata'
    #                exp_date = np.nan
            except Exception as e2:
                session_valid = False #session_valid.at[indiv_id] = False
    #            exp_date = np.nan
                f = 1
                if cnt==0:
                    print('\t session %d, experiment %d: %s' %(session_id, indiv_id, e2))
                    
            
            
            ############## get depth from mongo
            try:
                indiv_data['imaging_depth'] = db_cursor[0]['lims_ophys_experiment']['depth']
                e3 = ''
            except Exception as e:
                e3 = e.args
                session_valid = False #session_valid.at[indiv_id] = False
                f = 1
                if cnt==0:
                    print('\t session %d, experiment %d: %s' %(session_id, indiv_id, e3))
                                            

    
            ##### Take care of mouse-seeks qc... if it was set to failed, set the session to invalid!
            if workflow_state.iloc[cnt].values=='failed':
                e4 = 'failed workflow_state'
                session_valid = False
                f = 1
            else:
                e4 = ''
                
                
            ###########################        
            if f==0:    
                session_valid = True #session_valid.at[indiv_id] = True
                e0 = ''; e1 = ''; e2 = ''; e3 = ''; e4 = ''
            
    
            etot = [e0,e1,e2,e3,e4]
            validity_log.at[cnt, ['session_id', 'lims_id', 'date', 'stage_metadata', 'stage_mongo', 'valid', 'log']] = session_id, indiv_id, exp_date, stage_metadata, stage_mongo, session_valid, etot
    

    validity_log = validity_log.merge(workflow_state, left_index=True, right_index=True)
    
        
    return validity_log



#%% Set ROI ids of neurons in crosstalk-corrected dff traces that you created    

def set_roiIds_manDff(dir_ica, indiv_id):
    
#     dir_ica = f'/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk/session_{session_id}'
#     indiv_id = experiment_id
    
    import allensdk.core.json_utilities as ju
    
    ############ ROI ids in traces_out file. ############
    tnam = os.path.join(dir_ica, f'traces_out_{indiv_id}.h5')
    tfo = h5py.File(tnam, 'r') # <KeysViewHDF5 ['data', 'mixing_matrix', 'mixing_matrix_adjusted', 'roi_names']>
    traces_rois = np.array(tfo['roi_names']) # roi_names in traces_out file

    
    ############ valid ROI ids ############
    ############ load valid_ct file 
    regex = re.compile(f'ica_traces_(.*)') # valid_{839716139}.json        
    list_files = os.listdir(dir_ica) # list of files in session_xxxx
    ica_traces_folders = [string for string in list_files if re.match(regex, string)]
#         print(ica_traces_folders)

    # find the ica_traces folder that belongs to experiment indiv_id; the correct file name will have an output other than -1 when we run str.find()
    ica_traces_ind = np.argwhere([-1!=ica_traces_folders[ifo].find(str(indiv_id)) for ifo in range(len(ica_traces_folders))]).squeeze()
    ica_traces_dir = ica_traces_folders[ica_traces_ind]
#         print(ica_traces_dir)

    # ica traces : json filename
    # valid_json_dir = os.path.join(dir_ica, ica_traces_dir, f'{indiv_id}_valid.json')
    valid_json_dir = os.path.join(dir_ica, ica_traces_dir, f'{indiv_id}_valid_ct.json')
#     print(f'\n{valid_json_dir}')


    rois = ju.read(valid_json_dir)
    # set the roi ids for true ROIs in the json file
    ra = np.array(list(rois.keys())) # all ROIs
    rv = np.array(list(rois.values())) # array of False and True
    roi_ids_ct = ra[rv].astype(int) # only valid ROIs
#     print(len(ra), len(roi_ids_ct))
    # print(roi_ids_ct)
#     print(f'{np.mean(rv)} : fraction valid_ct ROIs')

    
    roi_ids_ct_dff = traces_rois[np.in1d(traces_rois, roi_ids_ct)]
#     common_rois = np.in1d(traces_rois, roi_ids_ct) # those rois of traces_out that also exist in valid_ct file.
#     print(sum(common_rois))
    # if the number of rois in valid_ct and traces_out is the same common_rois will be the same as "rv".


    return roi_ids_ct_dff # ,roi_ids_ct, traces_rois
                

    
#%%
def load_session_data_new(metadata_all, session_id, list_mesoscope_exp, use_ct_traces=1, use_np_corr=1, use_common_vb_roi=1):
    
    # if use_np_corr=1, we will load the manually neuropil corrected traces; if 0, we will load the soma traces.
    # if use_common_vb_roi=1, only those ct dff ROIs that exist in vb rois will be used.    
    # list_mesoscope_exp = experiment_ids
    
    import visual_behavior.data_access.loading as loading
    from datetime import datetime
    
    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'

    r_fact = .7 # manual neuropil correction: np_corr = soma_trace - r_fact*np_trace
    
    # Get list of experiments using ophysextractor
    '''
    Session_obj = LimsOphysSession(lims_id=session_id)
    list_mesoscope_exp = Session_obj.data_pointer['ophys_experiment_ids']
    '''

    ########################################################################
    #### set whole_data; contains df/f and time traces, also experiment metadata ####
    ########################################################################
    
    whole_data ={}
    DB = mongo.qc.metrics 

    # Go through data from each plane (experiment)
    for indiv_id in list_mesoscope_exp: # indiv_id = list_mesoscope_exp[0] # indiv_id = experiment_ids[0]
        
        print(f'\n============ experiment_id:{indiv_id} ============\n')
        indiv_id = int(indiv_id)
        
        ##### Set whole_data #### 
        indiv_data = {}
        indiv_data['session_id'] = session_id
        
        try: 
#             dataset = VisualBehaviorOphysDataset(indiv_id, cache_dir=cache_dir) # vb            
#             roi_ids_vb = dataset.cell_specimen_table['cell_roi_id'].values

            dataset = loading.get_ophys_dataset(indiv_id, include_invalid_rois=False) # allen sdk
    
            indiv_id_with_dataset = indiv_id
            
            try:
                table_stim = dataset.stimulus_presentations
            except Exception as e:
                print(f'Cannot get dataset.stimulus_presentations; will try again on for a different experiment!')
                print(e)
                
            ###
            if 0: #use_ct_traces: # now (2/10/2021 decrosstalked traces are part of lims; so we get them from dataset object)
                print('Using crosstalk-corrected dff traces.')
                ######## read the crosstalk-corrected dff traces ########
                dir_ica = f'/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk/session_{session_id}'
                
                # load manually neuropil corrected trace
                r_dff_dir = os.path.join(dir_ica, f'neuropil_correction_r{r_fact}_dff_{indiv_id}.h5')
                # load soma trace
                t_dff_dir = os.path.join(dir_ica, f'traces_out_dff_{indiv_id}.h5')
                
                if use_np_corr==1:
                    dir_ica_sess = r_dff_dir
                else: # use soma traces
                    dir_ica_sess = t_dff_dir
                                    
                f = h5py.File(dir_ica_sess, 'r') 
                dff_traces0 = np.asarray(f.get('data')).T # neurons x frames            

                roi_ids_ct_dff = set_roiIds_manDff(dir_ica, indiv_id) # set ROI ids for the neurons in crosstalk-corrected dff traces that you created                
    
                # old dff files
#                 dir_ica_sess = os.path.join(dir_ica, f'{indiv_id}_dff_ct.h5') # _dff.h5'
#                 print(dir_ica_sess)

#                 f = h5py.File(dir_ica_sess, 'r') 
#                 dff_traces0 = np.asarray(f.get('data')) # neurons x frames            
#                 roi_ids_ct_dff  = np.array(f.get('roi_names'))    
    
                
                ### take care of nan neurons
                nan_rois = np.isnan(dff_traces0[:,0])
                if sum(nan_rois)>0:
                    print(f'Removing {sum(nan_rois)} NaN ROIs...')
                    dff_traces0 = dff_traces0[~nan_rois]
                    roi_ids_ct_dff = roi_ids_ct_dff[~nan_rois]
                
                ######## make sure crosstalk traces have the same set of ROIs as those created by the convert code ########
                if use_common_vb_roi==1:
                    '''
                    # set the json file name
                    regex = re.compile(f'ica_traces_(.*)') # valid_{839716139}.json        
                    list_files = os.listdir(dir_ica) # list of files in session_xxxx
                    ica_traces_folders = [string for string in list_files if re.match(regex, string)]
                    # find the ica_traces folder that belongs to experiment indiv_id; the correct file name will have an output other than -1 when we run str.find()
                    ica_traces_ind = np.argwhere([-1!=ica_traces_folders[ifo].find(str(indiv_id)) for ifo in range(len(ica_traces_folders))]).squeeze()
                    ica_traces_dir = ica_traces_folders[ica_traces_ind]
                    # json filename:
                    valid_json_dir = os.path.join(dir_ica, ica_traces_dir, f'valid_{indiv_id}.json')

                    # read the valid_xx.json file to get the list of valid ROIs    
                    import allensdk.core.json_utilities as ju
                    rois_valid = ju.read(valid_json_dir)
                    rois = rois_valid["signal"]
        #             print(sum(rois.values())) # sum of true rois      
                    # set the roi ids for true ROIs in the json file
                    ra = np.array(list(rois.keys()))
                    rv = np.array(list(rois.values()))
                    roi_ids_ct_dff = ra[rv].astype(int)
        #             print(len(roi_ids_ct_dff))
                    '''

                    ####### get VB roi set
                    roi_ids_vb = dataset.roi_metrics['id'].values
        #             print(len(roi_ids_vb))

                    # get only those ct rois that exist in dataset roi 
                    commonROIs = np.in1d(roi_ids_ct_dff, roi_ids_vb)
                    print(f'{len(roi_ids_vb)} VB ROIs. {len(roi_ids_ct_dff)} CT ROIs. {sum(commonROIs)} final common ROIs.')
                    print(f'{sum(commonROIs)/len(roi_ids_vb):.2f}: fraction VB ROIs that exist in the final ROI set!\n')
    #                 roi_ids_ct_dff[commonROIs]
    
                    if sum(commonROIs)==0: # this problem happens when roi_ids that we got above are indeed cell_specimen_ids; to fix this we get the roi_ids from allenSDK
                        
                        print(f'ATTN: 0 common ROIs; so setting ROI ids using loading!')                
                        print(f'session_id: {session_id}; experiment_id: {indiv_id}')
                    
                        dataset2 = loading.get_ophys_dataset(indiv_id, include_invalid_rois=False)
                        roi_ids_vb = dataset2.cell_specimen_table['cell_roi_id'].values

                        # get only those ct rois that exist in dataset roi 
                        commonROIs = np.in1d(roi_ids_ct_dff, roi_ids_vb)
                        print(f'{len(roi_ids_vb)} VB ROIs. {len(roi_ids_ct_dff)} CT ROIs. {sum(commonROIs)} final common ROIs.')
                        print(f'{sum(commonROIs)/len(roi_ids_vb):.2f}: fraction VB ROIs that exist in the final ROI set!\n')
                
#                         print(f'session_id: {session_id}; experiment_id: {indiv_id}')
#                         print(f'roi_ids_vb:\n{roi_ids_vb}')
#                         print(f'\nroi_ids_ct_dff:\n{roi_ids_ct_dff}')
        
                    ###### set final dff_traces; only includes rois common between VB and cross-talk.
                    dff_traces = dff_traces0[commonROIs]
                    roi_ids_final = roi_ids_ct_dff[commonROIs]

                    
                else: # dont worry about using common rois with vb; use all rois that exist in ct traces.
                    dff_traces = dff_traces0
                    roi_ids_final = roi_ids_ct_dff

            
                indiv_data['fluo_traces'] = dff_traces
                indiv_data['roi_ids'] = roi_ids_final
                
    #             plt.figure(figsize=(15,10)); plt.subplot(211); plt.plot(dff_traces.T); plt.subplot(212); plt.plot(dataset.dff_traces.T);

    
            ###########################################################################
            ############### use dff traces in lims; get them from dataset object ###############
            else:
#                 indiv_data['fluo_traces'] = dataset.dff_traces
                indiv_data['fluo_traces'] = dataset.dff_traces['dff'].values
                indiv_data['roi_ids'] = dataset.dff_traces['cell_roi_id'].values
                
                
        except Exception as e:
            print('session: %d' %session_id)
            print('experiment: %d' %indiv_id)
            print('\t%s' %e)
#            x = e.args
#            print(x[0])
#            if 'roi_metrics.h5' in str(e):
#                ophys_data = convert_level_1_to_level_2(indiv_id, cache_dir)
#                dataset = VisualBehaviorOphysDataset(indiv_id, cache_dir=cache_dir)
#            else:
#                raise(e)

        
        try:
            indiv_data['time_trace'] =  dataset.ophys_timestamps #dataset.timestamps['ophys_frames'][0]
        except Exception as e:
            print('session: %d' %session_id)
            print('experiment: %d' %indiv_id)
            print('\t%s' %e)

            
        try:
            # set metadata using allensdk
#             experiment_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
    
            local_meta = metadata_all[metadata_all['ophys_experiment_id']==indiv_id]
            
            indiv_data['targeted_structure'] = local_meta['targeted_structure'].values[0]
            indiv_data['mouse'] = local_meta['mouse_id'].values[0]      
            indiv_data['stage'] = local_meta['session_type'].values[0]         
            indiv_data['cre'] = local_meta['cre_line'].values[0]
            
            oldformat = local_meta['date_of_acquisition'].values[0]
            datetimeobject = datetime.strptime(oldformat,'%Y-%m-%d %H:%M:%S.%f')
            indiv_data['experiment_date'] = datetimeobject.strftime('%Y-%m-%d')
            
            
            # set metadata using vb dataset object
            '''
            dataset = VisualBehaviorOphysDataset(indiv_id, cache_dir=cache_dir) # vb            
            local_meta = dataset.get_metadata() # vb
            
            indiv_data['targeted_structure'] = local_meta['targeted_structure'].values[0]
            indiv_data['mouse'] = local_meta['donor_id'].values[0]      # mouse_id  
            # You can also get stage from mongo: 
            # db_cursor = DB.find({"lims_id":session_id})
            # db_cursor[0]['change_detection']['stage']
            indiv_data['stage'] = local_meta['stage'].values[0]   # session_type      
            indiv_data['cre'] = local_meta['cre_line'].values[0]
            indiv_data['experiment_date'] = local_meta['experiment_date'].values[0]
            '''
            
        except Exception as e:
            print('session: %d' %session_id)
            print('experiment: %d' %indiv_id)
            print('\t%s' %e)

            
            
        # We have to get depth from Mouse-seeks database
        db_cursor = DB.find({"lims_id":indiv_id})
        indiv_data['imaging_depth'] = db_cursor[0]['lims_ophys_experiment']['depth']       

        
        whole_data[str(indiv_id)] = indiv_data
    
    
    
    ########################################################################    
    ##### Set data_list : contains area and depth for each experiment, sorted by area and depth ####
    ########################################################################
    data_list = pd.DataFrame([], columns=['lims_id', 'area', 'depth'])
    session_exclude = 0 

    for index,lims_ids in enumerate(whole_data.keys()):
        try:
            depth = whole_data[lims_ids]['imaging_depth']
            area = whole_data[lims_ids]['targeted_structure']
            
        except Exception as e:
            depth = np.nan
            area = np.nan
            session_exclude = 1 # we sort by area and depth below, so we need to get these vars for all experiments, or we have to exclude the session!
            
        local_exp = pd.DataFrame([[lims_ids, area, depth]], columns=['lims_id', 'area', 'depth'])
        data_list = data_list.append(local_exp)  
        
        
    ### Important: here we sort data_list by area and depth, so the order of experiment is different in data_list vs whole_data 
    # data_list is similar to whole_data but sorted by area and depth        
    if session_exclude == 0:
        data_list = data_list.sort_values(by=['area', 'depth'])
    else:
        data_list.loc[:,'depth']=np.nan
    
    
    
    ########################################################################    
    #### Set table_stim & behavioral parameters; these are all at the session level ####
    ########################################################################    
    
    #### NOTE: you should use allensdk for below as well!!!
    
    ####### vb
    '''
    # we dont need to reset dataset, we can just go with the dataset from the last experiment, generated above!
    dataset = VisualBehaviorOphysDataset(indiv_id_with_dataset, cache_dir=cache_dir) # indiv_id_with_dataset for sure has dataset. some experiments may have failed. 

    ### stimulus parameters
    table_stim = dataset.stimulus_table
    '''
    
    ####### use allensdk instead of above
#     dataset = loading.get_ophys_dataset(indiv_id_with_dataset, include_invalid_rois=False)
#     table_stim = dataset.stimulus_presentations
    
    
    
    ### behavioral parameters
    ### vb
    behav_data = {}
    '''
    behav_data['running_speed'] = dataset.running_speed
    behav_data['licks'] = dataset.licks
    behav_data['rewards'] = dataset.rewards

    trials = dataset.get_all_trials()
    hit_rate, catch_rate, d_prime = vbut.get_response_rates(trials)
    behav_data['d_prime'] = d_prime
    behav_data['hit_rate'] = hit_rate
    behav_data['catch_rate'] = catch_rate            
    '''
    
    ### allensdk
    behav_data['running_speed'] = dataset.running_speed['speed']
    try:
        behav_data['licks'] = dataset.rewards['timestamps']
    except Exception as e:
        print(e)
        behav_data['licks'] = np.nan
        
    try:
        behav_data['rewards'] = dataset.licks['timestamps']    
    except Exception as e:
        print(e)
        behav_data['rewards'] = np.nan
        
    # below needs work
#     behavior_data = dataset.get_rolling_performance_df()
#     behavior_data['rolling_dprime']
#     behavior_data['hit_rate']
    
    
    return [whole_data, data_list, table_stim, behav_data]




#%% old code (not using anymore)
def load_session_data(session_id):
    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'

    # Get list of experiments using ophysextractor
    Session_obj = LimsOphysSession(lims_id=session_id)
    list_mesoscope_exp = Session_obj.data_pointer['ophys_experiment_ids']

    whole_data ={}
    DB = mongo.qc.metrics 

    # Go through data from each plane (experiment)
    for indiv_id in list_mesoscope_exp: #  indiv_id = list_mesoscope_exp[0]
        
        indiv_id = int(indiv_id)
        
        ##### Set whole_data #### 
        indiv_data = {}
        indiv_data['session_id'] = session_id
        
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
        # You can also get stage from mongo: 
        # db_cursor = DB.find({"lims_id":session_id})
        # db_cursor[0]['change_detection']['stage']
        indiv_data['stage'] = local_meta['stage'].values[0]         
        indiv_data['cre'] = local_meta['cre_line'].values[0]
        indiv_data['experiment_date'] = local_meta['experiment_date'].values[0]
        
        trials = dataset.get_all_trials()
        hit_rate, catch_rate, d_prime = vbut.get_response_rates(trials)
        indiv_data['d_prime'] = d_prime
        indiv_data['hit_rate'] = hit_rate
        indiv_data['catch_rate'] = catch_rate


        # we have to get depth from Mouse-seeks database
        db_cursor = DB.find({"lims_id":indiv_id})
        local_depth = db_cursor[0]['lims_ophys_experiment']['depth']       
        indiv_data['imaging_depth'] = local_depth
        
        
        whole_data[str(indiv_id)] = indiv_data
    
    
    
    ##### Set data_list ####
    data_list = pd.DataFrame([], columns=['lims_id', 'area', 'depth'])

    for index,lims_ids in enumerate(whole_data.keys()):
        depth = whole_data[lims_ids]['imaging_depth']
        area = whole_data[lims_ids]['targeted_structure']
        
        local_exp = pd.DataFrame([[lims_ids, area, depth]], columns=['lims_id', 'area', 'depth'])
        data_list = data_list.append(local_exp)  
        
    ### Important: here we sort data_list by area and depth, so the order of experiment is different in data_list vs whole_data 
    # data_list is similar to whole_data but sorted by area and depth                
    data_list = data_list.sort_values(by=['area', 'depth'])
    
    
    #### Set table_stim ####
    experiment_id = list_mesoscope_exp[1]
    dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
    
    table_stim = dataset.stimulus_table
    
    
    return [whole_data, data_list, table_stim]


#%%
def convert_time_to_frame(frameTrace, timeTrace): # find the frameTrace indeces at which timeTrace elements occurred.
    frame_inds = np.array([np.argmin(np.abs(frameTrace - timeTrace[i])) for i in range(len(timeTrace))])

    return frame_inds
                
                
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
    
    return fig1load_session_data


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
def get_all_multiscope_exp(projects=["VisualBehaviorMultiscope" , "MesoscopeDevelopment"]):
    
    local_db = mongo.db.ophys_session_log
    db_cursor = local_db.find({"project_code":{"$in":projects}}) #,"MesoscopeDevelopment"]}}) # VisualBehaviorMultiscope ; MesoscopeDevelopment
    
    print('%d sessions found for projects %s\n' %(db_cursor.count(), projects))
    
    # Sam:
    # "VisualBehaviorMultiscope" : production quality, started almost early March
    # "MesoscopeDevelopment" : older experiments
    
    # For Scientifica experiments use:
#    db_cursor = local_db.find({"project_code":{"$in":["VisualBehavior"]}})
    
    # list of all session_ids
#    [db_cursor[i]['id'] for i in range(db_cursor.count())]
        
    list_sessions_id = []
    list_sessions_date = []
    list_sessions_experiments = []
    
    cnt = 1
    for indiv_cursor in db_cursor: # indiv_cursor = db_cursor[290]
        print('----- session %d out of %d -----' %(cnt, db_cursor.count()))
        cnt = cnt+1
        try: 
            # We check the session is well constructed :
            # Get list of experiments using ophysextractor
            Session_obj = LimsOphysSession(lims_id = indiv_cursor['id'])
            list_mesoscope_exp = Session_obj.data_pointer['ophys_experiment_ids']

#            Session_obj.data_pointer['date_of_acquisition'].strftime('%Y-%m-%d')
            db_cursor_sess = local_db.find({"id":indiv_cursor['id']})
            exp_date = db_cursor_sess[0]['date_of_acquisition'].strftime('%Y-%m-%d')

            list_sessions_id.append(indiv_cursor['id'])
            list_sessions_date.append(exp_date)
            list_sessions_experiments.append([list_mesoscope_exp])
            
        except Exception as e:
            print(e)
#            print(str(indiv_cursor['id'])+' has issues')

    print('%d out of %d sessions have data and ophys_experiment_ids\n' %(len(list_sessions_id), db_cursor.count()))
    print('sessions span dates %s to %s\n' %(list_sessions_date[0], list_sessions_date[-1]))

    return(list_sessions_id, list_sessions_date, list_sessions_experiments)



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
                local_index = np.argmin(np.abs(local_time_traces-indiv_time))
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
            local_fluo = local_fluo-np.mean(local_fluo[index_pre])

            all_averages[index_cell,:]=local_fluo
            
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
    
    all_cell_properties = pd.DataFrame(data = {'d_prime': d_prime, 'lims_id':lims_id, 'session_id':session_id, 'depth':depth, 'targeted_structure':area, 'omitted_peak_relative_change':omitted_peak
        , 'mouse':mouse, 'stage':stage, 'cre':cre, 'experiment_date':experiment_date })   
    return all_cell_properties   


#%%
"""
X = get_all_omitted_peaks(814281477)


#%%
print(X)

#%% [markdown]
# ### Run local analysis

#%%
list_all_sessions = get_all_multiscope_exp()


#%%
initiate = True
for indiv_id in list_all_sessions:
    try: 
        local_pd = get_all_omitted_peaks(indiv_id)
        if initiate:
            all_pd = local_pd
            initiate = False
        else:
            all_pd = all_pd.append(local_pd)
        
    except Exception as e: 
        print(e)        
        print("Issues with "+str(indiv_id))

print(all_pd)



#%%
    
all_pd_local = all_pd[all_pd['cre'].isin(['Vip-IRES-Cre'])]
#local_pd = all_pd_local[~all_pd_local['stage'].isin(['OPHYS_5_images_B_passive','OPHYS_0_images_A_habituation','OPHYS_2_images_A_passive'])]
local_pd = all_pd_local[all_pd_local['stage'].isin(['OPHYS_2_images_A_passive', 'OPHYS_5_images_B_passive'])]
#all_pd_local = all_pd_local[all_pd_local['stage'].isin(['OPHYS_6_images_B','OPHYS_4_images_B'])]

list_sessions_id = local_pd['session_id'].unique()
plt.figure()
for indiv_id in list_sessions_id:
    local_pd = all_pd_local[all_pd_local['session_id']==indiv_id]
    list_area = local_pd['targeted_structure'].unique()
    for index,uni_area in enumerate(list_area): 
        plt.subplot(1,2,index+1)
        local_pd_area = local_pd[local_pd['targeted_structure']==uni_area]
        print(local_pd_area['stage'].unique())

        list_depth = local_pd_area['depth'].unique()

        list_depth = np.sort(list_depth)
        list_mean = []
        for indiv_depth in list_depth:
            local_mean = np.mean(local_pd_area[local_pd_area['depth']==indiv_depth]['omitted_peak_relative_change'])
            list_mean.append(local_mean)
#        plt.plot(list_depth, list_mean/ list_mean[0], label = local_pd_area['d_prime'].unique())          
        plt.plot(list_depth, np.array(list_mean) / np.array(list_mean[0])) #, label = local_pd_area['d_prime'].unique())          
        #plt.scatter(local_pd_area['depth'], local_pd_area['omitted_peak_relative_change']/list_mean[0], label=indiv_id)
        plt.title(uni_area)
        plt.ylim([0,2])

plt.legend()

#%%


#%%
all_pd['experiment_date'].unique()


#%% FN

session_id = 814281477

[whole_data, data_list, table_stim] = load_session_data(session_id)
list_omitted = table_stim[table_stim['omitted']==True]['start_time']

#data = get_omitted_peaks(whole_data, data_list, list_omitted)

plot_omitted_depth_area(whole_data, data_list, list_omitted)

"""





#%% Codes from postdoc defFuns.m

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################

#%% Function to only show left and bottom axes of plots, make tick directions outward, remove every other tick label if requested.

def makeNicePlots(ax, rmv2ndXtickLabel=0, rmv2ndYtickLabel=0):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    # Make tick directions outward    
    ax.tick_params(direction='out')    
    # Tweak spacing between subplots to prevent labels from overlapping
    #plt.subplots_adjust(hspace=0.5)
#    ymin, ymax = ax.get_ylim()

    # Remove every other tick label
    if rmv2ndXtickLabel:
        [label.set_visible(False) for label in ax.xaxis.get_ticklabels()[::2]]
        
    if rmv2ndYtickLabel:
        [label.set_visible(False) for label in ax.yaxis.get_ticklabels()[::2]]
#        a = np.array(ax.yaxis.get_ticklabels())[np.arange(0,len(ax.yaxis.get_ticklabels()),2).astype(int).flatten()]
#        [label.set_visible(False) for label in a]
    
    plt.grid(False)
        
    ax.tick_params(labelsize=12)

    # gap between tick labeles and axis
#    ax.tick_params(axis='x', pad=30)

#    plt.xticks(x, labels, rotation='vertical')
    #ax.xaxis.label.set_color('red')    
#    plt.gca().spines['left'].set_color('white')
    #plt.gca().yaxis.set_visible(False)


#%% Smooth (got from internet)
    
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


#%% Plot histogram of verctors a and b on axes h1 and h2

def histerrbar(h1,h2,a,b,binEvery,p,lab,colors = ['g','k'],ylab='Fraction',lab1='exc',lab2='inh',plotCumsum=0,doSmooth=0, addErrBar=0):
#    import matplotlib.gridspec as gridspec    
#    r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
#    binEvery = r/float(10)
#    _, p = stats.ttest_ind(a, b, nan_policy='omit')
#    plt.figure(figsize=(5,3))    
#    gs = gridspec.GridSpec(2, 4)#, width_ratios=[2, 1]) 
#    h1 = gs[0,0:2]
#    h2 = gs[0,2:3]
#    lab1 = 'exc'
#    lab2 = 'inh'
#    colors = ['g','k']
    
    fs = 12
    ################### hist
    # set bins
    bn = np.arange(np.min(np.concatenate((a,b))), np.max(np.concatenate((a,b))), binEvery)
    bn[-1] = np.max([np.max(a),np.max(b)])#+binEvery/10. # unlike digitize, histogram doesn't count the right most value
    
    
    #######################################
    ############# plt hist of a ###########
    #######################################
    hist, bin_edges = np.histogram(a, bins=bn)
    hist = hist/float(np.sum(hist))    
    if plotCumsum:
        hist = np.cumsum(hist)
    if doSmooth!=0:        
        hist = smooth(hist, doSmooth)
    ax1 = plt.subplot(h1) #(gs[0,0:2])
    # plot the center of bins
    plt.plot(bin_edges[0:-1]+binEvery/2., hist, color=colors[0], label=lab1)    #    plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[0], alpha=.4, label=lab1)    
#    plt.xscale('log')
    
    if addErrBar:
        # set errorbars defined as :  np.sqrt(n*(1-n/N))/N if y axis is fraction; and sqrt(n(1-n/N)) if y axis is numbers. (Peter Latham defined errorbars this way!)
        ind = np.digitize(a, bins=bn)
        N = float(len(a))
        
        errbr = np.full((len(bn)-1), np.nan)
        n_allbins = []
        for indnum in np.arange(1, len(bn)):
            n = np.sum(ind==indnum)
            n_allbins.append(n)
#            errbr[indnum-1] = np.sqrt((n/N)/N)
            errbr[indnum-1] = np.sqrt(n*(1-n/N))/N
            #errbr[indnum-1] = np.sqrt(n/N)/N
#            errbr[indnum-1] = np.sqrt((n/N)*(1-n/N))
            
        plt.fill_between(bin_edges[0:-1]+binEvery/2., hist-errbr, hist+errbr, alpha=.5, facecolor=colors[0], edgecolor=colors[0])
#        plt.plot(bin_edges[0:-1]+binEvery/2., hist+errbr, color=colors[0])#, label=lab1)
#        plt.plot(bin_edges[0:-1]+binEvery/2., hist-errbr, color=colors[0])#, label=lab1)
    

    #####################################
    ########### plot his of b ########### 
    #####################################
    hist, bin_edges = np.histogram(b, bins=bn)
    hist = hist/float(np.sum(hist));     #d = stats.mode(np.diff(bin_edges))[0]/float(2)
    if plotCumsum:
        hist = np.cumsum(hist)
    if doSmooth!=0:
        hist = smooth(hist, doSmooth)
    plt.plot(bin_edges[0:-1]+binEvery/2., hist, color=colors[1], label=lab2)        #    plt.bar(bin_edges[0:-1], hist, binEvery, color=colors[1], alpha=.4, label=lab2)
#    plt.xscale('log')
    
    
    if addErrBar:
        # set errorbars defined as :  np.sqrt(n*(1-n/N))/N if y axis is fraction; and sqrt(n(1-n/N)) if y axis is numbers. (Peter Latham defined errorbars this way!)
        ind = np.digitize(b, bins=bn)
        N = float(len(b))
        
        errbr = np.full((len(bn)-1), np.nan)
        n_allbins = []
        for indnum in np.arange(1, len(bn)):
            n = np.sum(ind==indnum)
            n_allbins.append(n)
#            errbr[indnum-1] = np.sqrt((n/N)/N)
            errbr[indnum-1] = np.sqrt(n*(1-n/N))/N            
            #errbr[indnum-1] = np.sqrt(n/N)/N
#            errbr[indnum-1] = np.sqrt((n/N)*(1-n/N))
            
#        plt.plot(bin_edges[0:-1]+binEvery/2., hist, color=colors[1], label=lab2)
        plt.fill_between(bin_edges[0:-1]+binEvery/2., hist-errbr, hist+errbr, alpha=.5, facecolor=colors[1], edgecolor=colors[1])
#        plt.plot(bin_edges[0:-1]+binEvery/2., hist+errbr, color=colors[1])#, label=lab2)
#        plt.plot(bin_edges[0:-1]+binEvery/2., hist-errbr, color=colors[1])#, label=lab2)    
    
    
    # set labels, etc
    yl = plt.gca().get_ylim()
    ry = np.diff(yl)
    plt.ylim([yl[0]-ry/20 , yl[1]])   
    #
    xl = plt.gca().get_xlim()
    rx = np.diff(xl)
    
    plt.xlim([xl[0]-rx/20 , xl[1]])   # comment this if you want to plot in log scale
            
    plt.legend(loc=0, frameon=False, fontsize=fs)
    plt.ylabel(ylab, fontsize=fs) #('Prob (all days & N shuffs at bestc)')
#    plt.title('mean diff= %.3f, p=%.3f' %(np.mean(a)-np.mean(b), p))
    plt.title('mean diff= %.3f' %(np.mean(a)-np.mean(b)), fontsize=fs)
    #plt.xlim([-.5,.5])
    plt.xlabel(lab, fontsize=fs)
#    makeNicePlots(ax1,0,1)
    makeNicePlots(ax1,0,0)

    
    ################ errorbar: mean and st error
    ax2 = plt.subplot(h2) #(gs[0,2:3])
    plt.errorbar([0,1], [a.mean(),b.mean()], [a.std()/np.sqrt(len(a)), b.std()/np.sqrt(len(b))], marker='o',color='k', fmt='.')
    plt.xlim([-1,2])
#    plt.title('%.3f, %.3f' %(a.mean(), b.mean()))
    plt.xticks([0,1], (lab1, lab2), rotation='vertical', fontsize=fs)
    plt.ylabel(lab, fontsize=fs)
    plt.title('p=%.3f' %(p), fontsize=fs)
    makeNicePlots(ax2,0,1)
#    plt.tick_params
    yl = plt.gca().get_ylim()
    r = np.diff(yl)
    plt.ylim([yl[0], yl[1]+r/10.])
    
    plt.subplots_adjust(wspace=1, hspace=.5)
    return ax1,ax2



#%% hist 2d, natalia
'''    
def plot_pixel_hist2d(x, y, xlabel=None, ylabel=None, save_fig = False, save_path = None):

    fig = plt.figure(figsize=(5,5))

    H, xedges, yedges = np.histogram2d(x, y, bins=(30, 30))

    H = H.T

    plt.imshow(H, interpolation='nearest', origin='low',

              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', norm=LogNorm())

    plt.colorbar()

 

    slope, offset, r_value, p_value, std_err = linregress(x, y)

    fit_fn = np.poly1d([slope, offset])

 

    plt.plot(x, fit_fn(x), '--k')

    plt.rcParams.update({'font.size': 20})

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title('%s    R2=%.2f'%(fit_fn, r_value**2), fontsize= 16)

    if save_fig :

        plt.savefig(save_path, bbox_inches='tight')

   

    return fig, slope, offset, r_value
'''

#%% Plot histogram of loss values for training and testing data
'''
def plot_hist(a,b):
    
    get_ipython().magic(u'matplotlib qt')
    
#    a = r_sq_train_all
#    b = r_sq_test_all
    lab = 'explained variance'
    
    #import matplotlib.gridspec as gridspec    
    r = np.max(np.concatenate((a,b))) - np.min(np.concatenate((a,b)))
    binEvery = r/float(10)
    _, p = st.ttest_ind(a, b, nan_policy='omit') # stats
    plt.figure(figsize=(8,5))    
    gs = gridspec.GridSpec(2, 4)#, width_ratios=[2, 1]) 
    h1 = gs[0,0:2]
    h2 = gs[0,2:3]
    lab1 = 'training'
    lab2 = 'testing'
    colors = ['k','g']
    ylab ='Fraction neurons'
        
    #get_ipython().magic(u'matplotlib inline')
    
    [ax1, ax2] = histerrbar(h1,h2,a,b,binEvery,p,lab,colors,ylab,lab1,lab2) #='exc',lab2='inh',plotCumsum=0,doSmooth=0, addErrBar=0)
    
    ax1.set_title('Fraction neurons with r2>0 = %.2f\nAverage r2 of neurons with r2>0 = %.2f' 
           %(np.mean(b>0), np.mean(b[b>0])), fontsize=11)
'''


#%%
##########################################################################
##########################################################################
#################### functions used for the DNN model ####################
##########################################################################
##########################################################################
##########################################################################


#%% Recover the original traces from training and testing datasets

def recover_orig_traces(train, test, train_data_inds, test_data_inds, trial_inds=1, recover_y=1):
    
    # when train_data_inds and test_data_inds are trial indeces, we need to turn them to frame indeces (by multiplying them by len(frames2ana)), so they correspond to x_train indeces.
    if trial_inds:
        # convert train_data_inds into "frames" indeces
        train_frame_inds = np.array([np.arange(itr*len(frames2ana), itr*len(frames2ana)+len(frames2ana)) for itr in train_data_inds]).flatten()
        # convert test_data_inds into "frames" indeces
        test_frame_inds = np.array([np.arange(itr*len(frames2ana), itr*len(frames2ana)+len(frames2ana)) for itr in test_data_inds]).flatten()
    
    else: # use below if train and test data_inds were found on frames of all trials (instead of trials), (which is wrong, because it will not lead to proper model training due to correlation across continuous frames and hence correlation of training and testing dataset)
        train_frame_inds = train_data_inds
        test_frame_inds = test_data_inds

    
    si = np.argsort(np.concatenate((train_frame_inds, test_frame_inds))) # indeces that would sort the array
    
    concat_train_test = np.concatenate((train, test), axis=0)
    sorted_tt = concat_train_test[si,:] # (frames x trials) x units # when np.ndim(sorted_tt)=3
    # sorted_tt.shape
    # np.equal(tracesn, sorted_tt).sum()
    
    if np.ndim(sorted_tt)==2: # entire session data
        if recover_y:
            recover_traces = sorted_tt # frames x units
        else:# sorted_tt # frames x (neurons x 10) # because x was made by concatenating a window of 10 frames across for all neurons
            recover_traces = sorted_tt[:, np.arange(0, sorted_tt.shape[1], len_win)] # frames x units 
    
    elif np.ndim(sorted_tt)==3: # frames, units, trial data
        # print(sorted_tt.shape[0] / len(frames2ana))
        # aa = np.reshape(sorted_tt, (len(frames2ana), traces0.shape[2], traces0.shape[1]), order='F') # frames x trials x units
        aa = np.reshape(sorted_tt, (len(frames2ana), int(sorted_tt.shape[0] / len(frames2ana)), train.shape[1]), order='F') 
        recover_traces = np.transpose(aa, (0,2,1)) # frames x units x trials
        # a3 = np.equal(traces[frames2ana], recover_traces)
    
    return(recover_traces)


#%% Define max error: when there is no prediction, so max error will be sum(y^2).
# see the function: r_squared
    
def set_max_error(y_train):        
    
    # compute sum of y sqaured, then average it across neurons and observations 
    # the values below must be 1 if using z score y_train and y_test, because std = 2d_norm / sqrt(n). 
    err_mx = (np.linalg.norm(y_train)**2) / np.prod(y_train.shape)    
    
    return err_mx

    
    # a = np.sqrt(np.square(y_train).sum()) # same as Frobenius norm
    # al = np.sqrt(np.square(np.log(y_train)).sum()) # squared log error
    
    #### 2d norm and std relation: std = 2d_norm / sqrt(n)
    ## note: this is true if mean(y)= 0, otherwise std : sqrt(sum((y-y_ave)^2) / n)
    # 2d norm : sqrt(sum(y^2))
    # std : sqrt(sum(y^2) / n)
    ### so, for centered data:
    # std = 2d_norm / sqrt(n)    

    # below is wrong! because we compute loss by taking mean across all obersations and neurons.... compute norm for each observation, and then average across observations to get the averaged max error.
#    err_mx = np.mean(np.linalg.norm(y_train, axis=1)**2) 
#    val_err_mx = np.mean(np.linalg.norm(y_test, axis=1)**2) 


#%% Define variance explained

def r_squared(y_train, loss_bestModel_eval):        
    
    # compute sum of y sqaured, then average it across neurons and observations 
    # the values below must be 1 if using z score y_train and y_test, because std = 2d_norm / sqrt(n). 
#    err_mx = (np.linalg.norm(y_train)**2) / np.prod(y_train.shape)
    
    # Note: it makes more sense to compute sum_squared_total, because:
    # R_squared (proportion of variance in y that is predictable from x)
    ####################################################################
    # R_squared = SS_explained / SS_total = 1 - (SS_residual / SS_total)
    #################################################################### 
    # SS_total = SS_explained + SS_residual
    #
    # SS_total = sum(y - mean_y)^2
    # SS_explained = sum(y_hat - mean_y)^2
    # SS_residual = sum(y - y_hat)^2
    #
    # Note:
    # loss = SS_residual / n
    #
    # y_pred = autoencoder.predict(x_true)
    # loss = mean_sqr_err(y_true, y_pred)
    
    ss_total_mean = (np.linalg.norm(y_train - np.mean(y_train))**2) / np.prod(y_train.shape)
     
    r_sq = 1 - (loss_bestModel_eval / ss_total_mean)
    
    
    return r_sq


#%% Plot loss for training and validation data
    
def plot_loss(loss, val_loss, ep=np.nan, err_mx=1, val_err_mx=1, plotNorm=0, xlab='epochs', ylab='loss'):
    
    if np.isnan(ep).all():
        ep = range(len(loss))
        
#    plt.figure(figsize=(6,6))
    plt.figure(figsize=(6,3))    
    # plt.suptitle(('latent size: %d' %latent_size))
#            plt.subplot(211)
    
    if plotNorm: # plot errors normalized to max error
        lab0 = 'Norm training loss'
        lab1 = 'Norm validation loss'
        h0 = plt.plot(ep, loss / err_mx, 'g-', label=lab0)
        h1 = plt.plot(ep, val_loss / val_err_mx, 'g:', label=lab1)            
    else:
        lab0 = 'Training loss'
        lab1 = 'Validation loss'        
        h0 = plt.plot(ep, loss, 'b-', label=lab0)
        h1 = plt.plot(ep, val_loss, 'b:', label=lab1)
    h = [h1,h1]
    lab = [lab0, lab1]
    
    if type(err_mx) == np.float64: # err_mx is a single element; display the loss for the last epoch
        plt.title('Training: %.3f, norm: %.3f\nTesting: %.3f, norm: %.3f' %(loss[-1], loss[-1]/err_mx, 
                                                                        val_loss[-1], val_loss[-1]/val_err_mx))
    else: # show in the title average of loss across all x values
        plt.title('Training: %.3f, norm: %.3f\nTesting: %.3f, norm: %.3f' %(np.nanmean(loss), np.nanmean(loss/err_mx), 
                                                                        np.nanmean(val_loss), np.nanmean(val_loss/val_err_mx)))

    plt.legend(loc='center left', bbox_to_anchor=(1, .7), fontsize=10)
    # plt.show()
    plt.xlabel(xlab, fontsize=11)
    plt.ylabel(ylab, fontsize=11)
    
    '''
    lab = 'MAE' # 'MSLE'
    plt.subplot(212)
    plt.plot(ep, mae, 'r-', label='Training %s' %lab)
    plt.plot(ep, val_mae, 'r:', label='Validation %s' %lab)
    
    plt.title(lab)
    plt.legend(loc='top right', fontsize=10)
    # plt.show()
    plt.xlabel('epochs', fontsize=11)
    
    plt.subplots_adjust(hspace=.5)
    '''
    return h, lab


#%% Just like plot_loss, but we also plot the sem of loss across batches for each epoch
        
def plot_loss_sem(loss, val_loss, loss_eachEpoch_aveBatches, loss_eachEpoch_semBatches, ep=np.nan, err_mx=1, val_err_mx=1, plotNorm=0):
    
    if np.isnan(ep).all():
        ep = range(len(loss))
        
    plt.figure(figsize=(6,3))    
    
    if plotNorm: # plot errors normalized to max error
        plt.plot(ep, loss / err_mx, 'g-', label='Norm training loss')
        plt.plot(ep, val_loss / val_err_mx, 'g:', label='Norm validation loss')            
        
    else:
#        plt.plot(ep, loss, 'b-', label='Training loss')
        plt.plot(ep, loss_eachEpoch_aveBatches, 'k')
        plt.fill_between(ep, loss_eachEpoch_aveBatches-loss_eachEpoch_semBatches , loss_eachEpoch_aveBatches+loss_eachEpoch_semBatches, color='k', alpha=.3)
        plt.plot(ep, loss, 'b')

        plt.plot(ep, val_loss, 'b:', label='Validation loss')

        
    
    plt.title('Train: loss %.3f, norm: %.3f\nTest: loss %.3f, norm: %.3f' %(loss[-1], loss[-1] / err_mx, val_loss[-1], val_loss[-1] / val_err_mx))
    plt.legend(loc='center left', bbox_to_anchor=(1, .7), fontsize=10)
    # plt.show()
    plt.xlabel('epochs', fontsize=11)


    
#%% Compute loss manually and compare with keras loss; also plot loss for each neuron
# note if you use regularization, keras loss will be different from manual loss:
# see this : https://stackoverflow.com/questions/49903706/keras-predict-gives-different-error-than-evaluate-loss-different-from-metrics
# It turns out that the loss is supposed to be different than the metric, because of the regularization. Using regularization, the loss is higher (in my case), because the regularization increases loss when the nodes are not as active as specified. The metrics don't take this into account, and therefore return a different value, which equals what one would get when manually computing the error.

def mean_sqr_err(y_true, y_pred):
    
    d = (y_pred - y_true)**2
    e = np.mean(d, axis=0)  # average across observations  

#    y_true = y_test
#    y_pred = autoencoder.predict(x_test)
#    err_eachNeuron = mean_sqr_err(y_true, y_pred)

    return e # loss for each neuron
#    return np.mean(e)    
    # return K.mean(K.square(y_pred - y_true)) # K.abs(y_pred - y_true)    


#%% Find the best model across epochs, i.e. the one with minimum val_loss; set loss and val_loss at that epoch.

def set_bestModel_loss(loss_all, val_loss_all, err_mx_all=1, val_err_mx_all=1):
    
#    ax = np.ndim(loss_all)-1
    if np.ndim(loss_all)==1:
        dim = 1
        loss_all = np.array([loss_all])
        val_loss_all = np.array([val_loss_all])
        err_mx_all = np.array([err_mx_all])
        val_err_mx_all = np.array([val_err_mx_all])

    ax = 1
        
    ind_bestModel = np.argmin(val_loss_all, axis=ax) # min of val_loss across epochs for each neuron subsample
    print('Epoch index of best model:', ind_bestModel)
    if len(ind_bestModel)>1:
        plt.plot(ind_bestModel)
        plt.xlabel('Iteration (subsampling neurons)')
        plt.ylabel('Epoch of min val_loss')
    
    
    val_loss_bestModel = np.min(val_loss_all, axis=ax)
    loss_bestModel = np.array([loss_all[insamp, ind_bestModel[insamp]] for insamp in range(np.shape(loss_all)[0])]) # iters_neur_subsamp
    
    val_loss_bestModel_norm = val_loss_bestModel / val_err_mx_all
    loss_bestModel_norm = loss_bestModel / err_mx_all # this should be the same as loss_bestModel if the input was z scored. (because err_mx_all will be 1)
    
    #val_loss_bestModel.shape, val_loss_bestModel_norm.shape
    
    # print average across neuron subsamples
    print('bestModel loss(train,test): ', np.mean(loss_bestModel), np.mean(val_loss_bestModel))
    print('Norm bestModel loss(train,test): ', np.mean(loss_bestModel_norm), np.mean(val_loss_bestModel_norm))

    if dim==1:
        ind_bestModel = ind_bestModel.squeeze()
        loss_bestModel = loss_bestModel.squeeze()
        val_loss_bestModel = val_loss_bestModel.squeeze()
    
    return ind_bestModel, loss_bestModel, val_loss_bestModel
        

#%% Get loss values for all batches of each epoch

def set_loss_allBatches(loss_batches, batch_size, len_train):
    
#     len_train = x_train.shape[0]
#     loss_batches = np.array(history.losses)
    n_batches_per_epoch = int(np.ceil(len_train / batch_size))
    epochs_completed = int(loss_batches.shape[0] / n_batches_per_epoch)
    
    loss_eachEpoch_allBatches = np.reshape(loss_batches, (n_batches_per_epoch, epochs_completed), order='F') # n_batches_per_epoch x epochs

    return loss_eachEpoch_allBatches, n_batches_per_epoch, epochs_completed


#%% Do PCA on data, and project data onto the top PCs that explain 90% of variance in the data

def doPCA_transform(x, pca, varexpmax=.90, doplot=1):
    ##### project x onto pc space
    t = pca.transform(x)
    t.shape

    ##### take only those columns of dimensionality-reduced x whose corresponding pcs explain upto 90% of the variance    
    # explained variance by each pc
    pca_variance = pca.explained_variance_ratio_
    pcmax = int(np.argwhere(np.cumsum(pca_variance) >= varexpmax)[0])
    
    if doplot:
        plt.figure()
        plt.plot(np.cumsum(pca_variance), '.', label='cumsum var')
        plt.plot(pca_variance, '.', label='var')
        plt.xlabel('pc component')
        plt.ylabel('varience explained')
        plt.hlines(varexpmax, 0, pca.n_components_)    
        plt.title('%d / %d components\n%.2f of pcs explain %.2f variance' %(pcmax+1, pca.n_components_, (pcmax+1) / x.shape[1] , varexpmax))
        plt.legend()
#         print('%.2f of pcs are needed to explain %.2f variance' %((pcmax+1) / x.shape[1] , varexpmax))
    
    # take only those columns of dimensionality-reduced x whose corresponding pcs explain upto 90% of the variance
    x_train_pc = t[:, :pcmax+1]
    print(x_train_pc.shape)
    
    # plt.plot(np.mean(x, axis=1), 'b')
    # plt.plot(np.mean(t, axis=1), 'r')
    
    # get the pc space
    x_pcs = pca.components_ # n_components x n_features # Principal axes in feature space, representing the directions of maximum variance in the data. The components are sorted by explained_variance_.
    x_pcs.shape
    
    # plt.figure()    
    # plt.plot(x_pcs[0])
    # plt.plot(x_pcs[1])

    return x_train_pc



def doPCA(x, varexpmax=.90, doplot=1):
    
    from sklearn.decomposition import PCA
    
    ##### fit pca to data
    pca = PCA()
    pca.fit(x)
    
    x_train_pc = doPCA_transform(x, pca, varexpmax=varexpmax, doplot=1)
    
    return x_train_pc, pca

    '''
    # singular values
    n = np.linalg.norm(t, axis=0)
    # the following two quantities are the same
    plt.plot(pca.singular_values_)
    plt.plot(n, 'r')
    
    
    # reshape t to frames x units x trials
    ynow = np.full(x_test.shape, np.nan)
    tr = recover_orig_traces(t, ynow, train_data_inds, test_data_inds)
    print(tr.shape)
    
    # reshape x to frames x units x trials
    xr = recover_orig_traces(x, ynow, train_data_inds, test_data_inds)
    print(xr.shape)
    
    # compare xtrain and its projection onto the pc spaces (averaged across trials and neurons/pcs)
    plt.plot(np.nanmean(xr, axis=(1,2)), 'b')
    plt.plot(np.nanmean(tr, axis=(1,2)), 'r')
    
    
    # confirm that transform worked properly
    # it = pca.inverse_transform(t)
    # it.shape
    # plt.plot(np.mean(x, axis=0), 'b')
    # plt.plot(np.mean(it, axis=0), 'r')
    '''


#%% Train the entire trace of a session: 
# x trace size: frames x (neurons x time_window)
# y trace size: frames x neurons

# a time_window in x predicts the frame at the middle of the time_window in y.
  
#################################################################
#################################################################
#################################################################
#################################################################

#%% From traces_events create the input matrix for the model: samples x neurons x time_window
# set samples so time and neurons are both in the feature space: have 1sec of each neuron activity is in the feature space, ie concatenate 10sec of all neurons to get the features.
# the idea is that a window in x traces predicts the middle time point of the y traces

def traces_neurons_times_features(traces_x0_evs, traces_y0_evs_thisNeuron, len_win, plotfigs=0):
    
    num_samps = traces_x0_evs.shape[1] - len_win
    
    xx0 = [] # frames x neurons x 10  # frames are like samples
    for t_x in range(num_samps):
        a = traces_x0_evs[:, t_x : t_x+len_win] # neurons x 10
        xx0.append(a)  # frames x neurons x 10  # frames are like samples
    
    xx0 = np.array(xx0) # frames x neurons x 10  # frames are samples
    xx0_2d = np.reshape(xx0, (xx0.shape[0], xx0.shape[1]*xx0.shape[2])) # frames x (neurons x 10) 
    
    if plotfigs:
#        get_ipython().magic(u'matplotlib qt')
        #plt.figure(); plt.plot(np.mean(xx0, axis=(1,2))) # all samples, averaged across features
        plt.figure(); plt.plot(xx0[:, rnd.permutation(xx0.shape[1])[0], 0]); # all samples, one neuron (one frame of the window)
        plt.title('X trace: one neuron (one feature)'); plt.xlabel('observations (frames)')
    
        plt.figure(); plt.plot(np.mean(xx0, axis=(1)).T); # look at the window, for all samples, averaged across neurons
        plt.title('X trace: neuron-averaged time window; all observations'); plt.xlabel('frames (time window used in features)')    
    #    plt.figure(); plt.plot(np.mean(xx0, axis=(0,1)), 'k', linewidth=3) # look at the window; averaged across samples and neurons
    #    plt.title('X trace: neuron-and-observation averaged time window'); plt.xlabel('frames (time window used in features)')    
        plt.figure(); plt.plot(np.mean(xx0, axis=(1))[rnd.permutation(xx0.shape[0])[0]]) # look at the window, for one sample, averaged across neurons
        plt.title('X trace: one observation, neuron-averaged'); plt.xlabel('frames (time window used in features)')
        
        plt.figure(); plt.plot(xx0_2d[rnd.permutation(xx0.shape[0])[0]]); 
        plt.title('X: one sample, time window from all neurons'); plt.xlabel('features')

    
    ##%% take the frame at the middle of the window for y traces
    
    t_y = int(len_win/2)
    le = traces_y0_evs_thisNeuron.shape[1]
    yy0 = traces_y0_evs_thisNeuron[:, t_y: le-t_y].T # frames x neurons
    
    if plotfigs:
        #plt.plot(np.mean(yy0, axis=(1))[-100:-1])
        plt.figure(); plt.plot(yy0[:,rnd.permutation(yy0.shape[1])[0]]) # all samples, one neuron
        plt.title('Y trace: one neuron (one feature)'); plt.xlabel('observations (frames)')

    print('\nx, y design matrix (samples x neurons x time_window) shape:')
    print(xx0.shape , yy0.shape)
    
    return xx0, yy0


#%% Assign training and testing dataset ... (to be used on xx0 and yy0, which correspond to traces_evs ... xx0 is samples x (neurons x time_window))
    
def set_train_test_data_inds_wind(num_samps, len_win, kfold, xx0=0, yy0=0, doPlots=0):
    
    # pick a random time point in the session, then take a chunk of data starting at this random point as the testind dataset. Assing the rest of the data as training data, excluding len_win-1 frames surrounding the testing data.
    len_test = int(kfold/100 * num_samps)+1
    print('\nlen_test %d' %len_test)
    
    # Get testing data from all over the session, randomly select 10 chunks of data as testing
    n_testing_blocks = max(1, int(len_test / 500)) # we want to have 500 frames in each chuck of testing data #2 #10
    
    if n_testing_blocks==1:
        
        test_frame1 = rnd.permutation(num_samps-len_test)[0] # num_samps-len_test is the last time point that the testing dataset of length len_test can start on.
        print('Testing frame 1st, last, len_trace:\n%d %d %d' %(test_frame1, test_frame1 + len_test, num_samps))
        
        test_data_inds = np.arange(test_frame1, test_frame1 + len_test)
        train_data_inds = np.concatenate((np.arange(0, test_frame1-len_win+1), np.arange(test_frame1+len_test-1+len_win, num_samps)))
        # if test_frame1 is < len_win, equation below wont be true.
#        if len(test_data_inds)+len(train_data_inds) != num_samps - (len_win-1)*2:
#            sys.exit('Error! numbers dont match!') 
        
        # set random observations as testing and training ... the problem is that a testing frame may immediately follow a training frame, hence be correlated with that ... so testing and training wont be quite independent and the model will cheat.
        '''
        num_samps = ntrs #local_fluo_allOmitt_a1[0].shape[2] # num trials  # set 10% of trials as testing dataset (it will in fact be these all frames of these trials as testing dataset)
        #    num_samps = len(frames2ana) * ntrs # num observations (ie frames2ana x trials)  # first concatenate frames of all trials; then assign testing dataset
        
        r = rnd.permutation(num_samps)
        train_data_inds = r[int(kfold/100 * num_samps)+1:]
        test_data_inds = r[:int(kfold/100 * num_samps)+1]    
        print(len(test_data_inds) , len(train_data_inds))
        '''                
        if doPlots:
            # plot xx and yy; also show traing and testing chunks
            plt.figure()
            ax1 = plt.subplot(211)
            tt = xx0
            xfeat = rnd.permutation(tt.shape[1])[0]
            plt.plot(tt[:, xfeat, 0]); plt.title('x: 1 neuron (1 frame of the window; all samples)'); plt.xlabel('Observations (frames)') # all samples, one neuron (one frame of the window)
            # mark training frames
            plt.plot(train_data_inds, np.full(train_data_inds.shape, 4), '.', markersize=1) 
            for i in range(n_testing_blocks):
                # testing blocks
                plt.axvspan(test_data_inds[0], test_data_inds[-1], alpha=0.2, facecolor='y')
                # the window surrounding testing blocks that is excluded from training dataset (to avoid any overlap)
                plt.axvspan(test_data_inds[0]-len_win, test_data_inds[0], alpha=0.2, facecolor='g')
                plt.axvspan(test_data_inds[-1], test_data_inds[-1]+len_win, alpha=0.2, facecolor='g')
                
                
            plt.subplot(212, sharex=ax1)
            tt = yy0
#            xfeat = rnd.permutation(tt.shape[1])[0]
            plt.plot(tt[:,0]); plt.title('y: 1 neuron (all samples)'); plt.xlabel('Observations (frames)') # all samples, one neuron
            # mark training frames
            plt.plot(train_data_inds, np.full(train_data_inds.shape, 4), '.', markersize=1) 
            for i in range(n_testing_blocks):
                # testing blocks
                plt.axvspan(test_data_inds[0], test_data_inds[-1], alpha=0.2, facecolor='y')
                # the window surrounding testing blocks that is excluded from training dataset (to avoid any overlap)
                plt.axvspan(test_data_inds[0]-len_win, test_data_inds[0], alpha=0.2, facecolor='g')
                plt.axvspan(test_data_inds[-1], test_data_inds[-1]+len_win, alpha=0.2, facecolor='g')
            
        
    else:
    
        len_test_block = int(len_test/n_testing_blocks)
        print(len_test_block)
        # divide the session into a number of blocks (n_testing_blocks)
        sess_blocks = np.arange(0, num_samps-len_test_block, (num_samps-len_test_block)/n_testing_blocks).astype(int)
        # set last_test_frame_block: last allowed frame to be picked as a testing frame in a block of size np.min(np.diff(sess_blocks))
        last_test_frame_block = np.min(np.diff(sess_blocks)) - len_test_block # this is the latest frame that can be test_frame1 in a block. (if we go after this frame, we wont be able to get the entire frames of len_test_block)
        # from each block pick a random number as the 1st frame of the testing data in that block, then take len_test_block frames after that.
        test_data_inds_blocks = []
        for isb in range(n_testing_blocks): # isb=0
            test_frame1 = rnd.permutation(last_test_frame_block+1)[0] + sess_blocks[isb]
            test_data_inds_blocks.append(np.arange(test_frame1, test_frame1 + len_test_block))
        
        test_data_inds_blocks = np.array(test_data_inds_blocks)
        print('test_data_inds_blocks (start, end): ', test_data_inds_blocks[:,0], test_data_inds_blocks[:,-1])
        # concatenate the testing blocks to set the final testing indeces
        test_data_inds = np.concatenate((test_data_inds_blocks))
        #plt.plot(test_data_inds)
        len_test_final = len(test_data_inds)
        
        # the following should be true to make sure testing blocks are non overalpping with each other
        #(test_frame1+ len_test_block)    <     sess_blocks + last_test_frame_block
        
                
        ######### set training indeces
        # exclude len_win-1 frames before and after each testing block
        # set the first and last frames surrounding each testing block that need to be excluded
        fr_b = test_data_inds_blocks[:,0]-(len_win-1)
        fr_e = test_data_inds_blocks[:,-1]+(len_win-1)
        # set all the testin and their surrounding frames that need be excluded from training dataset 
        test_win_blocks = []
        for isb in range(n_testing_blocks):
            test_win_blocks.append(np.arange(fr_b[isb] , fr_e[isb]+1))
        test_win_blocks = np.array(test_win_blocks)
        test_win = np.concatenate((test_win_blocks))
                
        a = np.arange(0, num_samps)
        train_data_inds = a[~np.in1d(a, test_win)]
        
        # plot xx and yy; also show traing and testing chunks
        if doPlots:
            plt.figure()
            ax1 = plt.subplot(211)
            tt = xx0
            xfeat = rnd.permutation(tt.shape[1])[0]
            plt.plot(tt[:, xfeat, 0]); plt.title('x: 1 neuron (1 frame of the window; all samples)'); plt.xlabel('Observations (frames)') # all samples, one neuron (one frame of the window)
            # mark training frames
            plt.plot(train_data_inds, np.full(train_data_inds.shape, 4), '.', markersize=1) 
            for i in range(n_testing_blocks):
                # testing blocks
                plt.axvspan(test_data_inds_blocks[i,0], test_data_inds_blocks[i,-1], alpha=0.2, facecolor='y')
                # the window surrounding testing blocks that is excluded from training dataset (to avoid any overlap)
                plt.axvspan(fr_b[i], test_data_inds_blocks[i,0], alpha=0.2, facecolor='g')
                plt.axvspan(test_data_inds_blocks[i,-1], fr_e[i], alpha=0.2, facecolor='g')
                
                
            plt.subplot(212, sharex=ax1)
            tt = yy0
            plt.plot(tt[:,0]); plt.title('y: 1 neuron (all samples)'); plt.xlabel('Observations (frames)') # all samples, one neuron
            # mark training frames
            plt.plot(train_data_inds, np.full(train_data_inds.shape, 4), '.', markersize=1) 
            for i in range(n_testing_blocks):
                # testing blocks
                plt.axvspan(test_data_inds_blocks[i,0], test_data_inds_blocks[i,-1], alpha=0.2, facecolor='y')
                # the window surrounding testing blocks that is excluded from training dataset (to avoid any overlap)
                plt.axvspan(fr_b[i], test_data_inds_blocks[i,0], alpha=0.2, facecolor='g')
                plt.axvspan(test_data_inds_blocks[i,-1], fr_e[i], alpha=0.2, facecolor='g')
        
    
    print('length training and testing datasets; sum; excluded frames:\n', len(train_data_inds), len(test_data_inds), len(test_data_inds) + len(train_data_inds), num_samps-(len(test_data_inds) + len(train_data_inds)))
    print('number of frames originally: ', num_samps)
    # sanity check
    print(np.equal(num_samps , len(test_data_inds) + len(train_data_inds) + n_testing_blocks*2*(len_win-1)))

    return train_data_inds, test_data_inds



#%% Set final (z scored) input matrices (traces_evs for x population activity and y neuron, ie traces_x0_evs and traces_y0_evs_thisNeuron) to traces_neurons_times_features, which sets xx0 and yy0.

def set_traces_x0_y0_evs_final(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, plotPlots=0, setcorrs=0):
    
#    traces_y0_evs[neuron_y] = traces[iu][inds_final]

    # set traces that only include the active parts of x and y traces      
    traces_x0_evs = traces_x0[:,inds_final_all[neuron_y]] # x_neurons x active_frames_of_neuronY
    traces_y0_evs_thisNeuron = traces_y0_evs[neuron_y][np.newaxis,:] # 1 x active_frames_of_neuronY
    
    print('\ntraces_original shape:')
    print(traces_x0.shape, traces_y0.shape)
    print('traces_events shape:')
    print(traces_x0_evs.shape, traces_y0_evs_thisNeuron.shape)


    ##### set correlation coefficients 
    if setcorrs:
        # compare correlation of average x trace with the y trace, for the original traces vs. the newly synthesized traces
        print('\nx-y corrcoef, average traces_original:')
        print(np.corrcoef(np.mean(traces_x0, axis=0), np.mean(traces_y0, axis=0))[0,1])
    
        print('x-y corrcoef, average traces_events') # how correlated are x neurons with y_neuron at times that y_neuron is active 
        print(np.corrcoef(np.mean(traces_x0_evs, axis=0), traces_y0_evs_thisNeuron)[0,1])
    
        # compute correlation between individual x and y neurons; then take an average across corr coefs 
        cs = np.corrcoef(traces_x0, traces_y0) # sum_x_y_neurons x sum_x_y_neurons    
        nu1 = traces_x0.shape[0]
        cs_allx_ally = cs[:nu1, nu1:] # n_x_neurons x n_y_neurons # correlation of each x neuron with each y neuron
        # average across x_neurons
        cc_orig = np.mean(cs_allx_ally)
        print('\nx-y average corrcoef, traces_original:')
        print(cc_orig)
        
    #    cs, ps = st.spearmanr(t1.T, t2.T) # (nu1+nu2) x (nu1+nu2) # how the activity of neurons covaries across trials  (input: each column: a variable)
    #    ps_across = ps[:nu1, nu1:] # nu1 x nu2   # area1-area2 corr
        cs = np.corrcoef(traces_x0_evs, traces_y0_evs_thisNeuron) # sum_x_y_neurons x sum_x_y_neurons    
        nu1 = traces_x0_evs.shape[0]
        cs_allx_activey = cs[:nu1, nu1:] # n_x_neurons # correlation of each neuron in x with the y_neuron
        # average across x_neurons
        cc_evs = np.mean(cs_allx_activey)
        print('x-y average corrcoef, traces_events')
        print(cc_evs)
    
    
    ######%% Plot traces_events for x and y    
    if plotPlots:
        plt.figure(figsize=(8,15)); 
        plt.subplot(311); plt.plot(traces_x0_evs.T); plt.title('traces_x0_evs, each neuron', fontsize=12)
        plt.subplot(312); plt.plot(np.mean(traces_x0_evs, axis=0)); plt.title('traces_x0_evs, neuron-averaged', fontsize=12)
        plt.subplot(313); plt.plot(traces_y0_evs_thisNeuron.T); plt.title('traces_y0_evs, neuron %d' %neuron_y, fontsize=12)
        plt.subplots_adjust(hspace=.8)
    
    
    ######%% Keep a copy of traces_events    
    traces_x00 = traces_x0_evs + 0
    traces_y00 = traces_y0_evs_thisNeuron + 0
    
    
    ######%% Z score the x and y traces_events    
    scaler_x = StandardScaler()    
    scaler_y = StandardScaler()

    traces_x0_evs = scaler_x.fit_transform(traces_x00.T).T
    traces_y0_evs_thisNeuron = scaler_y.fit_transform(traces_y00.T).T
    #test = scaler.transform(test00)
    #scaler_y.mean_
    
    #[traces_x0_evs, m_x, s_x] = zscore(traces_x00, softNorm=False)
    #[traces_y0_evs_thisNeuron, m_y, s_y] = zscore(traces_y00, softNorm=False)
    
    #; either do it here, before training and testing are assinged. Or do it on training and apply the same standardization to testing.
    # if you do it here, each neuron will be standardized to its mean and std of all frames.
    # if you do it on training dataset, each feature (1 frame within the time window) will be normalized across frames.
    
    return traces_x0_evs, traces_y0_evs_thisNeuron, scaler_x, scaler_y
    
    
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
        
    
    if plotPlots:
        x_train0_2d = np.reshape(x_train0, (x_train0.shape[0], x_train0.shape[1]*x_train0.shape[2])) # frames x (neurons x 10)         
 
        ##### x
        tt = x_train0
        #tt = x_test0
        neur = rnd.permutation(tt.shape[1])[0]
        plt.figure(); plt.plot(tt[:, neur, 0]); plt.title('x_train: 1 neuron (1 frame of the window; all samples)'); plt.xlabel('Observations (frames)') # all samples, one neuron (one frame of the window)
        plt.figure(); plt.plot(np.mean(tt, axis=(1))[rnd.permutation(tt.shape[0])[0]]); plt.title('x_train: neuron-averaged time window; 1 sample'); plt.xlabel('frames') # look at the window, for one sample, averaged across neurons
        
        plt.figure(); plt.plot(x_train0_2d[rnd.permutation(x_train0_2d .shape[0])[0]]); 
        plt.title('x_train: one sample, time window from all neurons'); plt.xlabel('features')
        
        x_test0_2d = np.reshape(x_test0, (x_test0.shape[0], x_test0.shape[1]*x_test0.shape[2])) # frames x (neurons x 10) 
        
        
        ##### y
        tt = y_train0
        #tt = y_test0
        plt.figure(); plt.plot(tt); plt.title('y_train: 1 neuron (all samples)'); plt.xlabel('Observations (frames)') # all samples, one neuron
    
    
    ######%% z score training data; apply the same standardization to testing data
    ''' 
    scaler_x = StandardScaler()
    x_train0z = scaler_x.fit_transform(x_train0_2d) # frames x (neurons x 10)
    x_train0z = np.reshape(x_train0z, (x_train0.shape[0], x_train0.shape[1], x_train0.shape[2]))
    x_train0z.shape
    
    scaler_y = StandardScaler()
    y_train0z = scaler_y.fit_transform(y_train0)
    y_train0z.shape
    
    
    # apply the same standardization to testing data
    x_test0z = scaler_x.transform(x_test0_2d)
    x_test0z = np.reshape(x_test0z, (x_test0.shape[0], x_test0.shape[1], x_test0.shape[2]))
    x_test0z.shape
    
    y_test0z = scaler_y.transform(y_test0)
    y_test0z.shape
    
    
    #scaler_y.mean_
    
    #[traces_x0_evs, m_x, s_x] = zscore(traces_x00, softNorm=False)
    #[traces_y0_evs_thisNeuron, m_y, s_y] = zscore(traces_y00, softNorm=False)
    
    y_train0 = y_train0z+0
    y_test0 = y_test0z+0
    '''
    
    ######%% Define final training and testing datasets for the model
    
    x_train = x_train0[:,np.newaxis,:,:] # samples x 1 x neurons x time_window
    x_test = x_test0[:,np.newaxis,:,:]
    
    y_train = y_train0 # samples x 1
    y_test = y_test0
    
    print('\nFinal: x_train, x_test:\n', x_train.shape, x_test.shape)
    print('Final: y_train, y_test:\n', y_train.shape, y_test.shape)
    
    
    if plotPlots:
        xfeat = rnd.permutation(x_train0.shape[1])[0]
        #get_ipython().magic(u'matplotlib qt')
        plt.figure(); plt.suptitle('training data') 
        #ax1 = plt.subplot(211); plt.plot(np.mean(x_train, axis=(1,2,3))); plt.title('x_train, feature averaged')
        ax1 = plt.subplot(211); plt.plot(x_train[:,0,xfeat,0]); plt.title('x_train, feature averaged')
        plt.subplot(212, sharex=ax1); plt.plot(y_train); plt.title('y_train; neuron %d' %neuron_y)
        
        plt.figure(); plt.suptitle('testing data')
        #ax1 = plt.subplot(211); plt.plot(np.mean(x_test, axis=(1,2,3))); plt.title('x_test, feature averaged')
        ax1 = plt.subplot(211); plt.plot(x_test[:,0,xfeat,0]); plt.title('x_test, feature averaged')
        plt.subplot(212, sharex=ax1); plt.plot(y_test); plt.title('y_test; neuron %d' %neuron_y)
    
    
    return x_train, x_test, y_train, y_test, train_data_inds, test_data_inds


#%% Set xx_alltrace and yy_alltrace, which include all frames of the session, and can be used in the model; they have size: samples x neurons x time_window
    
def set_xx_yy_alltrace(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, len_win, plotPlots=0):
    
    # set final (z scored) input matrices (traces_evs for x population activity and y neuron, ie traces_x0_evs and traces_y0_evs_thisNeuron) to traces_neurons_times_features, which sets xx0 and yy0.    
    [traces_x0_evs, traces_y0_evs_thisNeuron, scaler_x, scaler_y] = set_traces_x0_y0_evs_final(traces_x0, traces_y0, traces_y0_evs, inds_final_all, neuron_y, plotPlots=0, setcorrs=0)
    
    # xx_alltrace, same as xx0 but for the entire session (not just the active parts of the trace) 
    # (design matrix from which we set training and testing data; of size samples x neurons x time_window)

    # NOTE: traces_x0z includes all x neurons, however, depending on what neuron_y it is predicting, it will be 
    # differet. Because, it is z scored based on the mean and sd of the following trace, which varies across neuron_ys:
    # traces_x0[:,inds_final_all[neuron_y]]
            
    # apply the same standardization that was used for traces_events to the original traces traces_x0 and traces_y0
    traces_x0z = scaler_x.transform(traces_x0.T).T
    traces_y0z = scaler_y.transform(traces_y0.T).T
    traces_y0z_thisNeuron = traces_y0z[neuron_y][np.newaxis,:]
    
#    print(traces_x0z.shape, traces_y0z_thisNeuron.shape)
    #[traces_x0, m_x, s_x] = zscore(traces_x0, softNorm=False)
    #[traces_y0, m_y, s_y] = zscore(traces_y0, softNorm=False)
    
    
    # Plot original traces (z scored just like traces_events) for x and y
    if plotPlots:
        plt.figure(figsize=(8,15)); 
        plt.subplot(311); plt.plot(traces_x0z.T); plt.title('traces_x0, each neuron', fontsize=12)
        plt.subplot(312); plt.plot(np.mean(traces_x0z, axis=0)); plt.title('traces_x0_evs, neuron-averaged', fontsize=12)
        plt.subplot(313); plt.plot(traces_y0z_thisNeuron.T); plt.title('traces_y0, neuron %d' %neuron_y, fontsize=12)
        plt.subplots_adjust(hspace=.8)
    
    # Sanity check: below should be same as traces_events plots that we made above
    '''
    plt.figure(figsize=(8,15)); 
    plt.subplot(311); plt.plot(traces_x0z[:,inds_final_all[neuron_y]].T); plt.title('traces_x0, each neuron', fontsize=12)
    plt.subplot(312); plt.plot(np.mean(traces_x0z[:,inds_final_all[neuron_y]], axis=0)); plt.title('traces_x0_evs, neuron-averaged', fontsize=12)
    plt.subplot(313); plt.plot(traces_y0z[neuron_y][inds_final_all[neuron_y]].T); plt.title('traces_y0, neuron %d' %neuron_y, fontsize=12)
    plt.subplots_adjust(hspace=.8)
    '''
    
    ######%% From original traces create the input matrix for the model: samples x neurons x time_window
    # NOTE: xx_alltrace includes all x neurons, however, depending on what neuron_y it is predicting, it will be 
    # differet. Because, it is z scored based on the mean and sd of the following trace, which varies across neuron_ys:
    # traces_x0[:,inds_final_all[neuron_y]]
    
    [xx_alltrace, yy_alltrace] = traces_neurons_times_features(traces_x0z, traces_y0z_thisNeuron, len_win, plotfigs=0)
    # xx_alltrace.shape[0] == traces_x0.shape[1] - len_win
    # t_y = len_win/2
    # yy_alltrace = z_scored_traces_y0[t_y: le-t_y] 


    return xx_alltrace, yy_alltrace
    
    
#%% Set traces_evs (for the y trace), ie traces that are made by extracting the active parts of the input trace
# the idea is that to help with learning the events, take parts of the trace that have events

def set_traces_evs(traces_y0, th_ag, len_ne, doPlots=1):

#    len_ne = len_win #20 # number of frames before and after each event that are taken to create traces_events.    
#    th_ag = 10 #8 # the higher the more strict on what we call an event.
    
    ###########################################################################################
    ############# Andrea Giovannucci's method of identifying "exceptional" events. ############
    ###########################################################################################
    [idx_components, fitness, erfc] = evaluate_components(traces_y0, N=5, robust_std=False)
    erfc = -erfc
    
    #erfc.shape
    
    '''
    p1 = traces_y0.T
    p2 = -erfc.T

#    p1 = supimpose(traces_y0)
#    p2 = supimpose(-erfc)

    plt.figure()
    plt.plot(p1,'b');
    plt.plot(p2,'r');
    '''
    
    # seems like I can apply a threshold of ~10 on erfc to find "events"
    evs = (erfc >= th_ag) # neurons x frames
    #traces_y0_evs = np.full(evs.shape, np.nan)
    #traces_y0_evs[evs] = traces_y0[evs]
    
    # take a look at the fraction of time points that have calcium events.
    evs_num = np.sum(evs,axis=1) # number of time points with high df/f values for each neuron
    evs_fract = np.mean(evs,axis=1)
    
    if doPlots:
        get_ipython().magic(u'matplotlib inline')
        
        plt.figure(figsize=(4,3)); 
        plt.plot(evs_fract); plt.xlabel('Neuron in Y trace'); plt.title('Fraction of time points with high ("active") DF/F')
        makeNicePlots(plt.gca())
    
    ##################################################################
    ############ find gaps between events for each neuron ############
    ##################################################################    
    [gap_evs_all, begs_evs, ends_evs, gap_evs, bgap_evs, egap_evs, begs, ends] = find_event_gaps(evs)
    # np.equal(begs_evs[neuron_y],   ends_evs[neuron_y] + gap_evs[neuron_y])
    # begs[1,begs[0]==neuron_y]

    ##### Note: begs_evs does not include the first event; so index i in gap_evs corresponds to index i in begs_evs and ends_evs.
    ##### however, gaps are truely gaps, ie number of frames without an event that span the interval between two frames.
#    gap_evs_all # includes the gap before the 1st event and the gap before the last event too, in addition to inter-event gaps        
#    gap_evs # includes only gap between events 
#    begs_evs # index of event onsets, excluding the 1st events. (in fact these are 1 index before the actual event onset; since they are computed on the difference trace ('d'))
#    ends_evs # index of event offsets, excluding the last event. 
#    bgap_evs # number of frames before the first event
#    egap_evs # number of frames after the last event    
    
        
    #############################################################################################################
    ############ set traces_evs, ie a trace that contains mostly the active parts of the input trace ############
    #############################################################################################################    
    traces_y0_evs = []
    inds_final_all = []
    
    for iu in range(traces_y0.shape[0]):
        
        if sum(evs[iu])>0:
            # loop through each event, and take 20 frames before until 20 frames after the event ... we want to have a good sample of non-events too for training the model.
            #bnow = begs[1, begs[0]==iu]
            #enow = ends[1, ends[0]==iu]
            enow = ends_evs[iu]
            bnow = begs_evs[iu]
            e_aft = []
            b_bef = []
            for ig in range(len(bnow)):
            #    ev_sur.append(np.arange(bnow[ig]-len_ne, enow[ig]+len_ne+1))
                e_aft.append(np.arange(enow[ig], min(evs.shape[1], enow[ig]+len_ne)))
                b_bef.append(np.arange(bnow[ig]+1-len_ne, min(evs.shape[1], bnow[ig]+2))) # +2 because bnow[ig] is actually the frame preceding the event onset, so +1 here, and another +1 because range doesn't include the last element. # anyway the details dont matter bc we will do unique later.

            e_aft = np.array(e_aft) # n_gap x len_ne
            b_bef = np.array(b_bef)

            if len(e_aft)>1:
                e_aft_u = np.hstack(e_aft)
            else:
                e_aft_u = []

            if len(b_bef)>1:            
                b_bef_u = np.hstack(b_bef)
            else:
                b_bef_u = []
            #e_aft.shape, b_bef_u.shape

            # below sets frames that cover the duration of all events, but excludes the first and last event
            ev_dur = []
            for ig in range(len(bnow)-1):
                ev_dur.append(np.arange(bnow[ig], enow[ig+1]))

            ev_dur = np.array(ev_dur)

            if len(ev_dur)>1:            
                ev_dur_u = np.hstack(ev_dur)
            else:
                ev_dur_u = []
            #ev_dur_u.shape


            evs_inds = np.argwhere(evs[iu]).flatten() # includes ALL events.   
            # now take care of the 1st and the last event
            #bgap = [begs_this_n[0] + 1] # after how many frames the first event happened
            #egap = [evs.shape[1] - ends_this_n[-1] - 1] # how many frames with no event exist after the last event
            if len(bgap_evs[iu])>0:
                # get len_ne frames before the 1st event
                ind1 = np.arange(np.array(bgap_evs[iu])-len_ne, bgap_evs[iu])
                # if the 1st event is immediately followed by more events, add those to ind1, because they dont appear in any of the other vars that we are concatenating below.
                if len(ends_evs[iu])>1:
                    ii = np.argwhere(np.in1d(evs_inds, ends_evs[iu][0])).squeeze()
                    ind1 = np.concatenate((ind1, evs_inds[:ii]))            
            else: # first event was already going when the recording started; add these events to ind1
                jj = np.argwhere(np.in1d(evs_inds, ends_evs[iu][0])).squeeze()
    #            jj = ends_evs[iu][0]
                ind1 = evs_inds[:jj+1]


            if len(egap_evs[iu])>0:
                # get len_ne frames after the last event
                indl = np.arange(evs.shape[1]-np.array(egap_evs[iu])-1, min(evs.shape[1], evs.shape[1]-np.array(egap_evs[iu])+len_ne))
                # if the last event is immediately preceded by more events, add those to indl, because they dont appear in any of the other vars that we are concatenating below.
                if len(begs_evs[iu])>1:
                    ii = np.argwhere(np.in1d(evs_inds, 1+begs_evs[iu][-1])).squeeze() # find the fist event of the last event bout
                    indl = np.concatenate((evs_inds[ii:], indl))
            else: # last event was already going when the recording ended; add these events to ind1
                jj = np.argwhere(np.in1d(evs_inds, begs_evs[iu][-1]+1)).squeeze()
                indl = evs_inds[jj:]

            #ind1, indl

            inds_final = np.unique(np.concatenate((e_aft_u, b_bef_u, ev_dur_u, ind1, indl))).astype(int)

            if np.in1d(evs_inds, inds_final).all()==False: # all evs_inds must exist in inds_final, otherwise something is wrong!
                if np.array([len(e_aft)>1, len(b_bef)>1, len(ev_dur)>1]).all()==False: # there was only one event bout in the trace
                    inds_final = np.unique(np.concatenate((inds_final, evs_inds))).astype(int)
                else:
                    print(np.in1d(evs_inds, inds_final))
                    sys.exit('error in neuron %d! some of the events dont exist in inds_final! all events must exist in inds_final!' %iu)
        #    inds_final.shape
            inds_final = inds_final[inds_final>=0] # to avoid the negative values that can happen due to taking 20 frames before an event.
            
            traces_y0_evs_now = traces_y0[iu][inds_final]
        
        
        else: # there are no events in the neuron; assign a nan vector of length 10 to the following vars
            inds_final = np.full((10,), np.nan)
            traces_y0_evs_now = np.full((10,), np.nan)        
        
        #plt.figure(); plt.plot(inds_final)
        inds_final_all.append(inds_final)
        traces_y0_evs.append(traces_y0_evs_now) # neurons


    inds_final_all = np.array(inds_final_all)
    traces_y0_evs = np.array(traces_y0_evs) # neurons

    
    
    #################################################################################################
    ##################### make plots of traces_events for a random y_neuron #########################
    #################################################################################################
    if doPlots:
        get_ipython().magic(u'matplotlib inline')
        neuron_y = rnd.permutation(traces_y0.shape[0])[0] # 10 # 4
        if sum(evs[neuron_y])==0:
            neuron_y = rnd.permutation(traces_y0.shape[0])[0]

        doscl=0 # set to 1 if not dealing with df/f , so you can better see the plots 
        if doscl:
            traces_y0c = traces_y0 / np.max(traces_y0, axis=0)
            traces_y0_evsc = [traces_y0_evs[iu] / np.max(traces_y0_evs[iu]) for iu in range(len(traces_y0_evs))]

#        print(neuron_y)
        evs_inds = np.argwhere(evs[neuron_y]).flatten()

        
        # plot the entire trace and mark the extracted events
        plt.figure(); plt.suptitle('Y Neuron, %d' %neuron_y)
        plt.subplot(211)
        plt.plot(traces_y0[neuron_y],'b', label='df/f'); 
        plt.plot(evs_inds, np.full(evs_inds.shape, max(traces_y0[neuron_y])),'g.', label='events'); # max(traces_y0[neuron_y]) 
        plt.plot(inds_final_all[neuron_y], np.ones(inds_final_all[neuron_y].shape)*(max(traces_y0[neuron_y])*.9), 'r.', label='extracted frames')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, fontsize=12)
        makeNicePlots(plt.gca())
        
        plt.subplot(212)
        plt.plot(erfc[neuron_y],'r', label='-erfc'); 
        plt.plot(evs[neuron_y].astype(int)*10,'g', label='events'); 
        plt.plot(traces_y0c[neuron_y],'b', label='df/f'); 
        plt.hlines(th_ag, 0, erfc.shape[1])
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, fontsize=12)
        makeNicePlots(plt.gca())
        
        # now plot the extracted chunk of trace which includes events!!
        #iu = 10
        evs_inds_evs = np.argwhere(evs[neuron_y][inds_final_all[neuron_y]])
        plt.figure(); plt.suptitle('Y Neuron, %d' %neuron_y)
        plt.subplot(211)
        plt.plot(traces_y0_evsc[neuron_y],'b', label='df/f'); 
        plt.plot(evs_inds_evs, np.full(evs_inds_evs.shape, 1),'g.', label='events'); # max(traces_y0[neuron_y]) 
        makeNicePlots(plt.gca())
        
        plt.subplot(212)
        plt.plot(traces_y0_evsc[neuron_y],'b', label='df/f'); 
        plt.plot(erfc[neuron_y][inds_final_all[neuron_y]],'r', label='-erfc'); 
        plt.plot(evs[neuron_y][inds_final_all[neuron_y]].astype(int)*10,'g', label='events'); 
        plt.hlines(th_ag, 0, traces_y0_evsc[neuron_y].shape)
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, fontsize=12)
        makeNicePlots(plt.gca())
    
    return traces_y0_evs, inds_final_all


 
#%% Set traces_x0 and traces_y0; these are traces read from this_sess_l.iloc[ind]['local_fluo_traces']; after removing n_beg_end_rmv frames from their begining and end. 

def set_traces_x0_y0(this_sess_l, x_ind, y_ind, n_beg_end_rmv=0, plotPlots=0):

    ######## Set traces_x (neurons from multiple planes concatendated; activity of each neuron standardized)    
    traces_x = this_sess_l.iloc[x_ind]['local_fluo_traces']
    traces_y = this_sess_l.iloc[y_ind]['local_fluo_traces']

    print('Size of the very original traces:')
    print(traces_x.shape, traces_y.shape)
    
    if plotPlots:        
        plt.figure(); plt.plot(np.mean(traces_x, axis=0)); plt.title('X trace; neuron-averaged')
        plt.figure(); plt.plot(np.mean(traces_y, axis=0)); plt.title('Y trace; neuron-averaged')

#        traces = np.concatenate((a.values)) # neurons concatenated from 7 planes x times    
    #m = np.mean(traces, axis=1) # neurons
    #s = np.std(traces, axis=1) # neurons
    #if softNorm==1:
    #    s = s + thAct     
    #traces = ((traces.T - m) / s).T # neurons x times

    
    ######## Exclude the 1st and the last 60 frames of the session ... sometimes we see the trace rises at the beginning and at the end... it might be related to how df/f is computed
#        n_beg_end_rmv = 60 # number of frames to exclude from the begining and end of the session (because we usually see big rises in the trace, perhaps due to df/f computation?)        
    if n_beg_end_rmv>0:
        traces_x = traces_x[:, n_beg_end_rmv:-n_beg_end_rmv]
        traces_y = traces_y[:, n_beg_end_rmv:-n_beg_end_rmv]
    
    print('Size of the traces after excluding %d frames from the begining and end:' %n_beg_end_rmv)
    print(traces_x.shape, traces_y.shape)
    
    
    ######## Keep a copy of the original traces    
    traces_x0 = traces_x + 0
    traces_y0 = traces_y + 0
    
    return traces_x0, traces_y0   
      
        
#%% Compute y_prediction and explained variance

def r2_yPred(x_true, y_true, autoencoder, loss_provided=[0]):
    
    # loss_provided = [0]
    # loss_provided = [1, loss]
    
    # compute y_pred    
    y_pred = autoencoder.predict(x_true)    
    
    if loss_provided[0]==1: # loss is provided
        loss_now = loss_provided[1] #loss_bestModel_eval
    else: # loss is not provided; compute it
        loss_now = np.mean(mean_sqr_err(y_true, y_pred)) 
    # compute variance explained
    r_sq = r_squared(y_true, loss_now)
    
    return r_sq, y_pred


#%% Find the corresponding train_data_inds on trace yy_alltrace 

def map_inds(train_data_inds, len_win, inds_final_all, neuron_y, doPlots=0):
    # train_data_inds are on yy0, find their indeces on traces_y0_evs_thisNeuron (yy0 is made from traces_y0_evs_thisNeuron)
    t_y = int(len_win/2)  # yy0 = traces_y0_evs[:, t_y: traces_y.shape[1]-t_y].T
    train_data_inds_on_traces_y0_evs = train_data_inds + t_y
    # traces_y0_evs are made by applying inds_final indeces on traces_y0; i.e. traces_y0_evs = traces_y0[iu][inds_final]
    # so train_data_inds_on_traces_y0_evs onto inds_final, to find the indeces of traces_y0 that were chosen as train_data_inds
    train_data_inds_on_traces_y0 = inds_final_all[neuron_y][train_data_inds_on_traces_y0_evs] 
    # yy_alltrace is traces_y0[len_win/2:], so we subtract len_win/2 from train_data_inds_on_traces_y0 to get the corresponding indeces on yy_alltrace
    train_data_inds_on_yy_alltrace = train_data_inds_on_traces_y0 - t_y
                
    return train_data_inds_on_yy_alltrace, train_data_inds_on_traces_y0




#%% Convert pair index to area, depth index (eg pair 5 to 1st area depth 1 and 2nd area depth 1 ; or pair 11 to 1st area depth 3 and 2nd area depth 2); based on the following configuration: (note: this is used for pair-wise interactions)
# columns and rows are depths of each area:
# 0,4,8,12
# 1,5,9,13
# 2,6,10,14
# 3,7,11,15

# eg:
'''
pair_inds = [3, 5, 11, 8]
area12_depths = convert_pair_index_to_area_depth_index(pair_inds, num_depth=4)
print(area12_depths) # 1st column shows area 1 depth, 2nd column shows area 2 depth
# [[3. 0.]
#  [1. 1.]
#  [3. 2.]
#  [0. 2.]]
'''

def convert_pair_index_to_area_depth_index(pair_inds, num_depth=4):
    
    pair_inds = np.array(pair_inds)
    a = np.arange(0, num_depth*num_depth+1, num_depth)
    area12_depths = np.full((len(pair_inds),2), np.nan) # 1st column shows area 1 depth, 2nd column shows area 2 depth
    for pi in range(len(pair_inds)): # go through rows (area 1)
        pair_index = pair_inds[pi]
        # find the depth for area 1 (rows of pair_inds)
        area12_depths[pi, 0] = np.mod(pair_index, num_depth)
        
        # find the depth for area 2 (columns of pair_inds)
        hist, bin_edges = np.histogram(pair_index, a)
        area12_depths[pi, 1] = np.argwhere(hist)[0][0]

    area12_depths = area12_depths.astype(int)
    
    return area12_depths # 1st column shows area 1 depth, 2nd column shows area 2 depth




#%%
'''
def set_frame_window_flash_omit(time_win, samps_bef, frame_dur):
    # Convert a time window (relative to image/omission onset) to frames units. 
    # samps_bef: number of frames before image/omission
    # time_win = [0, .75]
    
    frames_win = samps_bef + np.round(time_win / frame_dur).astype(int) # convert peak_win to frames (new way to code list_times)
    frames_win[-1] = frames_win[0] + np.round(np.diff(time_win) / frame_dur).astype(int) # we redefine the upper limit of the window, otherwise the same window duration can lead to different upper limit values due to the division and flooring, and turning into int.
    list_times = np.arange(frames_win[0], frames_win[-1]) #+1) # [40, 41, 42, 43, 44, 45, 46, 47, 48]
    # for omit-evoked peak timing, compute it relative to samps_bef (which is the index of omission)
        
    return list_times
'''


#%
# h5
'''
f = h5py.File(np_corr_file, 'w')
dset = f.create_dataset("data", data=neuropil_correction_man) 
f.close()
'''

'''
f = h5py.File(dir_ica_sess, 'r') 
dff_traces0 = np.asarray(f.get('data')) # neurons x frames            
'''


# pandas
'''
all_sess = pd.read_hdf(allSessName, key='all_sess')
'''


# pickle
'''
f = open(np_corr_file, 'wb')
pickle.dump(df_grid, f)        
f.close()
'''

'''
f = np_corr_file
pkl = open(f, 'rb')
this_sess = pickle.load(pkl)
'''



############
#%% Example uses of exec and eval
'''
svm_allMice_sessPooled_block_name = f'svm_allMice_sessPooled_block{iblock}'

exec(svm_allMice_sessPooled_block_name + " = svm_allMice_sessPooled")

svm_allMice_sessPooled = eval(f'svm_allMice_sessPooled_block{iblock}')
'''




