#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First script that we need to run (for post analysis) after omissions_traces_peaks.py is run and the main vars are saved. 

############## NOTE ##############
This script loads mouse_trainHist_all2.h5 file, which shows the entire training history of a mouse
We have to fist save this file by running, on the cluster, set_mouse_trainHist_init_pbs.py which calls set_mouse_trainHist_all2.py 
Remember to fist update the list of mice in all_mice_id on set_mouse_trainHist_all2.py.
##################################

Vars needed here are loaded from the already saved h5 files (created by the function omissions_traces_peaks.py) 
(if doCorrs=0, the h5 filename is "all_sess_omit_traces_peaks")

If doCorrs=0, the loaded var is all_sess, a pandas table that includes vars related to median traces across trials, aligned on omission, and their peak amplitude.
If doCorrs=1, we load vars for each session from the server (this_sess), and then here we concatenate them into all_sess.
doCorrs=-1, eg. for umap analysis.

The main thing this script does is that it sets all_sess_2an which includes sessions to be analyzed given their imageset (A,B)

Follow this script by either of the following two to set vars for plotting.
if doCorrs:
    omissions_traces_peaks_plots_setVars_corr.py    
else:
    omissions_traces_peaks_plots_setVars_ave.py    

If doing umap analysis, "all_sess" is set in load_behavior_ophys_dataset_fn.py. "all_sess" will be loaded from server. 
Follow this script by umap_set_vars_run.py


Created on Mon Aug 26 12:23:25 2019
@author: farzaneh
"""

#### also add A1!

#NOTE:
#check for len(n_omissions)
#also have a threshold for the number of neurons you are taking into your analysis (whether it is to take their median, or to compute their corr)


#%%
doCorrs = 1 #1 # if 0, compute omit-aligned trace median, peaks, etc. If 1, compute corr coeff between neuron pairs in each layer of v1 and lm. If -1, only get the omisstion-aligned traces, dont compute peaks, mean, etc.  
analysis_dates = ['20210215'] #['20210309'] #['20210211'] #['20201119'] #['20200731'] #(subtractSigCorrs=1) #['20200804'] #(subtractSigCorrs=0) #['20200508_23'] #['20200424'] # will be used if doCorrs=1; the dates that correlation outputs (pkl files) were saved; we will only load pkl files saved on these dates. # normally it will be only 1 date, but in case the analysis lasted more than a day.  
# note: analysis_dates must not include the entire date_time (eg '20200508_233842'), because the code below assumes it is followed by some wildcard characters.
control_single_beam = 0 #1 #0 # if 1, make control data to remove the simultaneous aspect of dual beam mesoscope (ie the coupled planes) to see if the correlation results require the high temporal resolution (11Hz) of dual beam vs. 5Hz of single beam mesoscope 

subtractSigCorrs = 1 # only applicable to doCorrs=1; # if 1, the corr files were saved for the case that signal correlations were subtracted.

all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1 = 3 #5 #3 # a number between 0 to 6 # 0: analyze all sessions;  1: analyze AB transitions;  2: analyze only A sessions (before any B);  3: analyze all A sessions (before or after B);   4: analyze all B sessions (before or after A)    ;   5: analyze the first B session;   6: analyze all A and B sessions except for B1
only_1st_transit = 1 # relevant only if all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1=1 # if 1, only include data from the 1st A-->B transition even if a mouse has more than one (safer, since in the subsequent transitions, B has been already introduced, so they are not like the 1st A-->B transition)

th_neurons = 3 #3 # 5 #minimum number of neurons (for a plane), in order to use that plane in analysis. (for all cre lines)
same_y_fo = 1 # in summary mice plots, use the same ylim for image- and omission-evoked responses.
    
doShift_again = 0 #1 #0 # whether the all_sess_omit_traces_peaks file was saved for doShift_again or not; # this is a second shift just to make sure the pre-omit activity has baseline at 0. (it is after we normalize the traces by baseline ave and sd... but because the traces are median of trials (or computed from the initial gray screen activity), and the baseline is mean of trials, the traces wont end up at baseline of 0)                                      

dosavefig = 0 #1

#bl_gray_screen = 1 # this should be saved in all_sess, if not set it here. if 1, we used the initial gray screen at the beginning of the session to compute baseline
plot_flash = 1 # if 1, plot the amplitude of flash-evoked responses; if 0, plot the timing of omission-evoked responses. (in either case peak amp of omissions would be plotted!)
plot_eachSess_trAve_neurAve = 0 #1 # 0# if 1, for each session plot trial-averaged and neuron-averaged traces for every individual plane 


useSDK = 0 # always 0; it's an old thing. # load the all_sess file created by using allenSDK in script load_behavior_ophys_dataset_fn.py.
use_np_corr = 1 # will be used when use_ct_traces=1; if use_np_corr=1, we will load the manually neuropil corrected traces; if 0, we will load the soma traces.


#%%
from def_paths import * 
from def_funs import *
from def_funs_general import *
from set_mousetrainhist_allsess2an_whatsess import *
# from set_timetrace_flashesind_allmice_areas import *

dir_now = os.path.join(dir_server_me, 'omit_traces_peaks')
#if not os.path.exists(dir_now):
#    os.makedirs(dir_now)


##############################################################################
##############################################################################
#%% Set all_sess h5 file name
##############################################################################    
##############################################################################    
'''
analysis_name = 'omit_traces_peaks'
if doShift_again==1:
    namespec = '_blShiftAgain'
else:
    namespec =''
name = 'all_sess%s_%s_.' %(namespec, analysis_name) 
#name = 'all_sess_%s_.' %(analysis_name) 
'''

analysis_name = 'omit_traces_peaks'

if doCorrs==1:
    namespec = '_corr'
    if control_single_beam==1:
        namespec = namespec + '_controlSingleBeam'
elif doCorrs == -1:
    namespec = '_allTraces'
else:
    namespec = ''

if subtractSigCorrs==0:
    namespec = namespec + '_withSigCorrs'
    
if useSDK:
    namespec = namespec + '_sdk'
    
if doShift_again==1:
    namespec = namespec + '_blShiftAgain'
else:
    namespec = namespec + ''
    
if doCorrs==0:
    if use_np_corr==1:
        namespec = namespec + '_np_corr_dff'
    else:
        namespec = namespec + '_soma_dff'

    
if (doCorrs == 0) or (doCorrs == -1):
#    name = 'all_sess_%s%s_.' %(analysis_name, namespec) 
    name = 'all_sess_%s%s_[0-9].' %(analysis_name, namespec)
    all_files = 0
    allSessName, h5_files = all_sess_set_h5_fileName(name, dir_now, all_files)
    print(f'\nh5 file that will be used for analysis:\n {allSessName}')
    
else:          
#    namespec = namespec + 'm-%d_s-%d' %(mouse, session_id) # mouse, session: m, s
#    name = 'all_sess_%s%s_.' %(analysis_name, namespec) 
    name0 = 'this_sess_%s%s_.' %(analysis_name, namespec) 
    all_files = 1



##############################################################################    
##############################################################################    
#%% Load all_sess h5 file name
##############################################################################    
##############################################################################    

print(f'Loading all_sess dataframe...')

if doCorrs == 0:
    all_sess = pd.read_hdf(allSessName, key='all_sess') #'svm_vars')        ## Load all_sess dataframe
    input_vars = pd.read_hdf(allSessName, key='input_vars')     ## Load input_vars dataframe    

if doCorrs == -1: # pkl file was saved
    f = allSessName
    pkl = open(f, 'rb')
    all_sess = pickle.load(pkl)
    input_vars = pickle.load(pkl)

elif doCorrs == 1: # load pkl files (each one is for one session)

    l = os.listdir(dir_now) 
    
    # get pkl files names for each session    
#     named = 'this_sess_%s%s_m-(.*).pkl' %(analysis_name, namespec) # named = 'this_sess_(.*).pkl'
    pkl_files = []
    for iad in range(len(analysis_dates)):
#         named = 'this_sess_%s%s_m-(.*)_%s_(.*).pkl' %(analysis_name, namespec, analysis_dates[iad]) # named = 'this_sess_(.*).pkl'
        named = 'this_sess_%s%s_m-(.*)_%s(.*).pkl' %(analysis_name, namespec, analysis_dates[iad]) # named = 'this_sess_(.*).pkl'
        regex = re.compile(named) # + '.h5')
        pkl_files.append([string for string in l if re.match(regex, string)]) # each folder is for one session
    pkl_files = np.concatenate((pkl_files))
    print(len(pkl_files))

    # use below for sessions that are pkl files!
    cols = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', 'flash_omit_dur_all', 'flash_omit_dur_fr_all', \
                      'cc_cols_area12', 'cc_cols_areaSame', \
                      'cc_a12', 'cc_a11', 'cc_a22', 'p_a12', 'p_a11', 'p_a22', \
                      'cc_a12_shfl', 'cc_a11_shfl', 'cc_a22_shfl', 'p_a12_shfl', 'p_a11_shfl', 'p_a22_shfl'])   
    all_sess = pd.DataFrame([], columns=cols)
    
    # go through each session folder
    for i in range(len(pkl_files)): # i=0
        print(pkl_files[i])
        f = os.path.join(dir_now, pkl_files[i])

        pkl = open(f, 'rb')
        this_sess = pickle.load(pkl)
        if i==0: # input_vars form all sessions are the same, so we just load the one from the 1st session.
            input_vars = pickle.load(pkl)
        
        # data from all sessions
        all_sess = all_sess.append(this_sess) 
    print(f'Final number of sessions: {len(all_sess)}')
    
    
print(np.shape(all_sess))




#%% Take care of sessions with unusual frame duration!!!    
# make sure below works!

# frame_durs = np.concatenate((all_sess['frame_dur'].values)) # wont work if there are invalid experiments, they will be nan for frame duration
frame_durs = np.array([all_sess.iloc[i]['frame_dur'] for i in range(len(all_sess))])
if doCorrs==1:
    frame_durs = frame_durs[:,0] # take it only for one of the experiments

rmv = np.logical_or(frame_durs < .093, frame_durs > .094)
if sum(rmv)>0:
    s = np.unique(all_sess['session_id'].values[rmv])
    print(f'{len(s)} sessions have unusual frame duration. Take a look at these sessions, and add them to set_sessions_valid, line 178 to be excluded.')
    print(f"session ids: {s}")
    print(f'frame durations: {frame_durs[rmv]}')

    print(f'\n\nRemoving {len(s)} sessions with unexpected frame duration!!\n\n')
    all_sess = all_sess[~all_sess['session_id'].isin(s)]
#     all_sess = all_sess[~rmv]

        
#%%    
print(np.shape(all_sess))
all_sess

#len(input_vars), input_vars.iloc[0]

a = np.hstack(all_sess['n_neurons'].values); aa = sum(~np.isnan(a.astype(float)))
print(f'Final number of valid experiments (with >0 neurons): {aa}/{len(a)}')


##############################################################################    
##############################################################################    
#%% Set the vars in input_vars
### Note: the windows below will be used when doCorrs=0. For doCorrs=1, windows will be set in omissions_traces_peaks_plots_setVars_corr.py
##############################################################################    
##############################################################################    

if useSDK==1:
    samps_bef = input_vars['samps_bef'].iloc[0]
    samps_aft = input_vars['samps_aft'].iloc[0]
    bl_percentile = input_vars['bl_percentile'].iloc[0]
    
    # we define these here so we can run the code set_figname_labels below
    peak_win = [0, .75] # this should be named omit_win
    flash_win = [-.75, -.25]
    flash_win_timing = [-.875, -.25] # this is not needed anymore bc we are using different values     
    doScale = 0  # 1 # Scale the traces by baseline std
    doShift = 0  # 1 # Shift the y values so baseline mean is at 0 at time 0
    doShift_again = 0 # 0 # this is a second shift just to make sure the pre-omit activity has 
    bl_gray_screen = 1
    mean_notPeak = 1  # set to 1, so mean is computed; it is preferred because SST and SLC responses don't increase after omission
    norm_to_max = 0  # normalize each neuron trace by its max
    trace_median = 1

    
else:

    samps_bef = input_vars['samps_bef'].iloc[0]
    samps_aft = input_vars['samps_aft'].iloc[0]
    
    if (doCorrs==1) and (control_single_beam==1): # to resemble single-beam data, we took alternating frames, so samps_bef and samps_aft are now half the original values.
        samps_bef = int(samps_bef/2)
        samps_aft = int(samps_aft/2)

    norm_to_max = input_vars['norm_to_max'].iloc[0]
    mean_notPeak = input_vars['mean_notPeak'].iloc[0]
    trace_median = input_vars['trace_median'].iloc[0]
    doScale = input_vars['doScale'].iloc[0]
    doShift = input_vars['doShift'].iloc[0]
    bl_gray_screen = input_vars['bl_gray_screen'].iloc[0] # it may not exist if it is one of the older runs

    ############### Note: the windows below will be used when doCorrs=0. For doCorrs=1, windows will be redefined in omissions_traces_peaks_plots_setVars_corr.py
    peak_win = input_vars['peak_win'].iloc[0]

    '''
    flash_win = [-.75, -.15] # [-.75, -.25] #[-.75, -.4] #[-.75, 0] # window (relative to omission) for computing flash-evoked responses (750ms includes flash and gray)    
    flash_win_vip = [-1, -.4] # [-1, -.25] # previous value (for flash_win of all cre lines): # [-.75, -.25] # 
    bl_percentile = 10 #20  # for peak measurements, we subtract pre-omit baseline from the peak. bl_percentile determines what pre-omit values will be used.      
    '''
    flash_win = input_vars['flash_win'].iloc[0] # it may not exist if it is one of the older runs
    flash_win_vip = input_vars['flash_win_vip'].iloc[0] # it may not exist if it is one of the older runs
    flash_win_timing = input_vars['flash_win_timing'].iloc[0]

    bl_percentile = input_vars['bl_percentile'].iloc[0]
    num_shfl_corr = input_vars['num_shfl_corr'].iloc[0]
    doShift_again = input_vars['doShift_again'].iloc[0]
    subtractSigCorrs = input_vars['subtractSigCorrs'].iloc[0]




##############################################################################    
##############################################################################    
#%% Set a number of useful variables
##############################################################################    
##############################################################################    

# Set time for the interpolated traces (every 3ms time points); 
# Set the time of flashes and grays for the imaging traces;
# Set some useful vars which you will need for most subsequent analaysis: # all_mice_id, stages_sessionN, mouse_trainHist_all, mouseCre_numSess
# Set plane indeces for each area (V1 and LM)
# %run -i 'set_timetrace_flashesind_allmice_areas.py'
time_trace, time_trace_new, xmjn, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, \
    flashes_win_trace_index_unq, grays_win_trace_index_unq, all_mice_id, stages_sessionN, \
    mouse_trainHist_all, mouseCre_numSess, distinct_areas, i_areas, inds_v1, inds_lm = \
        set_timetrace_flashesind_allmice_areas(samps_bef, samps_aft, frame_dur, doCorrs, all_sess)




#%% Set sessions for analysis based on the desired stage (A, B, transition, etc)

print(f"\n\nIMPORTANT NOTE: make sure 'mouse_trainHist_all2' file is up to date; To do so, set_mouse_trainHist_all2.py needs to have the updated list of all_mice_id, and set_mouse_trainHist_init_pbs.py (which calls the xxxall2 function) needs to be run on the cluster.")
all_sess_2an, whatSess = set_mousetrainhist_allsess2an_whatsess(all_sess, dir_server_me, all_mice_id, all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1, only_1st_transit)


print(f'\n')
if all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==0:
    print(f'Analyze all sessions')  
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==1: 
    print(f'Analyze only A-->B transitions (ie consecutive A,B sessions)')
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==2: 
    print(f'Analyze A sessions before B')  
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==3: 
    print(f'Analyze all A sessions before or after B sessions')
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==4:       
    print(f'Analyze all B sessions before or after A sessions')
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==5:       
    print(f'Analyze the first B sessions')
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==6:       
    print(f'Analyze all A and B sessions except for the first B session')
        
        
if type(all_sess.iloc[0]['experiment_id'])=='str' or type(all_sess.iloc[0]['experiment_id'])==str: # svm analysis: each row of all_sess is one experiment, so we divide the numbers below by 8 to reflect number of sessions.
    print(f'\nFinal number of sessions for analysis: {len(all_sess_2an)/num_planes}')
else:
    print(f'\nFinal number of sessions for analysis: {len(all_sess_2an)}')    

    
##############################################################################    
##############################################################################    
#%% Set vars related to plotting
##############################################################################    
##############################################################################    

get_ipython().magic(u'matplotlib inline') # %matplotlib inline

num_depth = 4

cols_each = colorOrder(num_planes)
cols_area = ['b', 'k']    # first plot V1 (in black, index 1) then LM (in blue, index 0)   
cols_depth = ['b', 'c', 'g', 'r'] #colorOrder(num_depth) #a.shape[0]) # depth1_area2    
alph = .3
bb = (.92, .7) # (.9, .7)
xlim = [-1.2, 2.25] # [-13, 24]



#%% Set figure labels and fgn (it will be used for setting fign (the final figure named to be saved))

fgn, ylabel, ylab_short, lab_peakMean, lab_med = set_figname_labels(peak_win, flash_win, norm_to_max, doShift, doScale, doShift_again, bl_gray_screen, mean_notPeak, trace_median)


    

# Path for saving figures
if doCorrs==1:
    dir_now = 'correlations_omit_flash'
    if control_single_beam==1:
        dir_now = os.path.join(dir_now, 'control_single_beam')                
elif doCorrs==0:
    dir_now = 'averages_omit_across_sess'
elif doCorrs==-1:
    dir_now = 'umap_cluster'

    
if all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==0:
    dir_now = os.path.join(dir_now, 'A_B_all')
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==1: 
    if only_1st_transit:
        dir_now = os.path.join(dir_now, 'AB_transit_first')
    else:
        dir_now = os.path.join(dir_now, 'AB_transit')
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==2: 
    dir_now = os.path.join(dir_now, 'A_beforeB')
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==3: 
    dir_now = os.path.join(dir_now, 'A_all')
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==4:       
    dir_now = os.path.join(dir_now, 'B_all')
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==5:       
    dir_now = os.path.join(dir_now, 'B_first')
elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==6:       
    dir_now = os.path.join(dir_now, 'A_B_all_notB1')

        
fmt = '.pdf'
if not os.path.exists(os.path.join(dir0, dir_now)):
    os.makedirs(os.path.join(dir0, dir_now))
            
now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")




    
    
#%% Follow this script by either of the following codes:
    
'''
if doCorrs:
#     from omissions_traces_peaks_plots_setVars_corr import *
    omissions_traces_peaks_plots_setVars_corr.py
    
else:
    omissions_traces_peaks_plots_setVars_ave.py    
'''



         

#%% Old: loading corr files, when corr results were saved in multiple h5 files.    

'''
######## Load input_vars
nn = '_info'
name = 'all_sess_%s%s%s_(.*).pkl' %(analysis_name, namespec, nn) 
regex = re.compile(name) # + '.h5')
pkl_input_vars = [string for string in l if re.match(regex, string)] # each folder is for one session

allSessName = os.path.join(dir_now , pkl_input_vars[0]) 
print(allSessName)

f = os.path.join(dir_now, allSessName)
pkl = open(f, 'rb')
input_vars = pickle.load(pkl)
print(input_vars)


# load this_sess var for each session, and concatenate them to make the var "all_sess"
named = 'this_sess_%s%s_s-.' %(analysis_name, namespec)
regex = re.compile(named) # + '.h5')
l = os.listdir(dir_now) 
h5_folders = [string for string in l if re.match(regex, string)] # each folder is for one session


cols = np.array(['session_id', 'experiment_id', 'mouse_id', 'date', 'cre', 'stage', 'area', 'depth', 'n_omissions', 'n_neurons', \
                  'cc_cols_area12', 'cc_cols_areaSame', \
                  'cc_a12', 'cc_a11', 'cc_a22', 'p_a12', 'p_a11', 'p_a22', \
                  'cc_a12_shfl', 'cc_a11_shfl', 'cc_a22_shfl', 'p_a12_shfl', 'p_a11_shfl', 'p_a22_shfl'])   
all_sess = pd.DataFrame([], columns=cols)


# go through each session folder
for i in range(len(h5_folders)): # i=0
    dir_now_s = os.path.join(dir_now, h5_folders[i]) 
    allSessName, h5_files = all_sess_set_h5_fileName(name0, dir_now_s, all_files)

    # load each h5 file
    nn = '_info_m'
    name = 'this_sess_%s%s%s.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_info = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe
    input_vars = pd.read_hdf(allSessName[ind], key='input_vars')     ## Load input_vars dataframe    

    nn = '_c12_m'
    name = 'this_sess_%s%s%s.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_c12 = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    nn = '_c11_m'
    name = 'this_sess_%s%s%s.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_c11 = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    nn = '_c22_m'
    name = 'this_sess_%s%s%s.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_c22 = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    nn = '_p12_m'
    name = 'this_sess_%s%s%s.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_p12 = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    nn = '_p11_m'
    name = 'this_sess_%s%s%s.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_p11 = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    nn = '_p22_m'
    name = 'this_sess_%s%s%s.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_p22 = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe


    nn = '_c12_shfl'
    name = 'this_sess_%s%s%s_.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_c12_shfl = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    nn = '_c11_shfl'
    name = 'this_sess_%s%s%s_.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_c11_shfl = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    nn = '_c22_shfl'
    name = 'this_sess_%s%s%s_.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_c22_shfl = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    nn = '_p12_shfl'
    name = 'this_sess_%s%s%s_.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_p12_shfl = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    nn = '_p11_shfl'
    name = 'this_sess_%s%s%s_.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_p11_shfl = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    nn = '_p22_shfl'
    name = 'this_sess_%s%s%s_.' %(analysis_name, namespec, nn) 
    regex = re.compile(name) # + '.h5')
    ind = np.array([i for i in range(len(h5_files)) if re.match(regex, h5_files[i])]).squeeze()
    all_sess_p22_shfl = pd.read_hdf(allSessName[ind], key='this_sess') #'svm_vars')        ## Load all_sess dataframe

    # concatenate all dataframes above
    this_sess = pd.concat([all_sess_info, all_sess_c12, all_sess_c11, all_sess_c22, all_sess_p12, all_sess_p11, all_sess_p22,\
                  all_sess_c12_shfl, all_sess_c11_shfl, all_sess_c22_shfl, all_sess_p12_shfl, all_sess_p11_shfl, all_sess_p22_shfl], axis=1)
    #all_sess.shape; all_sess.iloc[0]


    # data from all sessions
    all_sess = all_sess.append(this_sess) 
    ''';        
            
