#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:23:43 2019

@author: farzaneh
"""
'''
from importlib import reload
import x
reload(x)
'''

import numpy as np
import matplotlib.pyplot as plt

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
            print(name)
            print('\nall_sess h5 file does not exist! (run svm_init to call svm_plots_init and save all_sess)')
        elif len(h5_files)>1:
            print('\nMore than 1 h5 file exists! Using the latest file')
            allSessName = os.path.join(dir_now, h5_files[-1])            
        else:
            allSessName = os.path.join(dir_now, h5_files[0])            
        print(allSessName)
        print(f'\n')
        
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



def flash_gray_onset_relOmit(samps_bef, samps_aft, frame_dur, flash_dur=.25, gray_dur=.5):
    #%% For traces that are aligned on omission, this funciton will give us the time (and frame number) of flashes and gray screens
    
    import numpy as np
    
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



def plot_flashLines_ticks_legend(lims, H, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, x='', xmjn='', bbox_to_anchor=(1, .7), ylab='% Classification accuracy', xlab='Time rel. trial onset (sec)', omit_aligned=0):
    ### NOTE: there is also this same code in def_funs
    
    #%% Add to the plots the following: flash/ gray screen lines , proper tick marks, and legend
#     h1 = plt.plot()
#     plot_flashLines_ticks_legend([], h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)        

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    import seaborn

    
    if len(lims)!=0: # set lims=[] when lims is not known.
        mn = lims[0] # np.round(np.nanmin(av_test_shfl_avSess_eachP)) # 45
        mx = lims[1] # np.round(np.nanmax(av_test_data_avSess_eachP)) # 80
        #    rg = (mx - mn) / 10.
    else:
        mn = plt.gca().get_ylim()[0];
        mx = plt.gca().get_ylim()[1];
        
    # mark omission onset
    plt.vlines([0], mn, mx, color='k', linestyle='--')
    
    # mark flash duration with a shaded area ### NOTE: you should remove 0 from flashes_win_trace_index_unq_time because at time 0, there is no flash, there is omission!!    
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
#         xmj = np.round(np.unique(np.concatenate((np.arange(0, time_trace[0], -.05), np.arange(0, time_trace[-1], .05)))), 2)
        xmj = np.round(np.unique(np.concatenate((np.arange(0, -1, -.05), np.arange(0, 1, .05)))), 2)
        xmn = [] #np.arange(-.5, time_trace[-1], 1)        
    else:
        xmj = xmjn[0]
        xmn = xmjn[1]
    
    ax = plt.gca()
    
    ax.set_xticks(xmj); # plt.xticks(np.arange(0,x[-1],.25)); #, fontsize=10)
    ax.set_xticklabels(xmj, rotation=45, ha='right')       
    
    ax.xaxis.set_minor_locator(ticker.FixedLocator(xmn))
    ax.tick_params(labelsize=10, length=6, width=2, which='major')
    ax.tick_params(labelsize=10, length=5, width=1, which='minor')
    #    plt.xticklabels(np.arange(0,x[-1],.25))
    
    if len(H)!=0:        
        plt.legend(handles=H, loc='center left', bbox_to_anchor=bbox_to_anchor, frameon=False, handlelength=1, fontsize=12)
    plt.ylabel(ylab, fontsize=12);
    plt.xlabel(xlab, fontsize=12);  
    if len(lims)!=0:
        plt.ylim(lims);
#    plt.legend(handles=[h1[0], h1[1], h1[2], h1[3], h1[4], h1[5], h1[6], h1[7], h2], loc='center left', bbox_to_anchor=(1, .7), frameon=False, handlelength=1, fontsize=12)
    
    plt.grid(False) #    plt.box(on=None) #    plt.axis(True)
    seaborn.despine()#left=True, bottom=True, right=False, top=False)

    
    
    

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
    