#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:23:43 2019

@author: farzaneh
"""

def set_frame_window_flash_omit(time_win, samps_bef, frame_dur):
    # Convert a time window (relative to trial onset) to frames units, relative to "trace" begining; i.e. index on the trace whose time 0 is trace[samps_bef]. 
    # samps_bef: number of frames before image/omission
    
    import numpy as np
    
    frames_win = samps_bef + np.round(time_win / frame_dur).astype(int) # convert peak_win to frames (new way to code time_win_frames)
    frames_win[-1] = frames_win[0] + np.round(np.diff(time_win) / frame_dur).astype(int) # we redefine the upper limit of the window, otherwise the same window duration can lead to different upper limit values due to the division and flooring, and turning into int.
    time_win_frames = np.arange(frames_win[0], frames_win[-1]) #+1) # [40, 41, 42, 43, 44, 45, 46, 47, 48]
    # for omit-evoked peak timing, compute it relative to samps_bef (which is the index of omission)
        
    return time_win_frames



def all_sess_set_h5_fileName(name, dir_now, all_files=0):
    # Look for a file in a directory; if desired, sort by modification time, and only return the latest file.
    # example inputs:
    #    name = 'all_sess_%s_.' %(analysis_name) 
    #    name = 'Yneuron%d_model_' %neuron_y
    
    import re
    import os
    import numpy as np
    
    regex = re.compile(name) # + '.h5')
#    regex = re.compile(aname + '(.*).hdf5') # note: "aname" does not include the directory path; it's just the file name.

    l = os.listdir(dir_now)
     
    h5_files = [string for string in l if re.match(regex, string)] # string=l[0]
    
    if len(h5_files)==0:
        print('Error: no h5 file exists!!!')
        allSessName = ''
        
    if all_files==0: # only get the latest file, otherwise get all file names
        # Get the modification times of the existing analysis folders
        modifTimes = [os.path.getmtime(os.path.join(dir_now, h5_files[i])) for i in range(len(h5_files))]
        
        # Find all the old analysis folders                               
        if len(modifTimes) > 1:
            h5_files = np.array(h5_files)[np.argsort(modifTimes).squeeze()]
            print(h5_files)
        
        
        if len(h5_files)==0:
            print('all_see h5 file does not exist! (run svm_init to call svm_plots_init and save all_sess)')
        elif len(h5_files)>1:
            print('More than 1 h5 file exists! Loading the latest file')
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




def colorOrder(nlines=30):
    # change color order of lines in matplotlib to a given colormap
    
    from numpy import linspace
    from matplotlib import cm
    cmtype = cm.jet # jet; what kind of colormap?
    
    start = 0.0
    stop = 1.0
    number_of_lines = nlines #len(days)
    cm_subsection = linspace(start, stop, number_of_lines) 
    colorsm = [ cmtype(x) for x in cm_subsection ]
    
    #% Change color order to jet 
#    from cycler import cycler
#    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    
#    a = plt.scatter(y, y2, c=np.arange(len(y)), cmap=cm.jet, edgecolors='face')#, label='class accuracy (% correct testing trials)')
            
    return colorsm

