#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:59:56 2019

@author: farzaneh
"""

import datetime


#%% Change color order to jet 

def colorOrder(nlines=30):
    ##%% Define a colormap   
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


#%%
# https://stackoverflow.com/questions/16006572/plotting-different-colors-in-matplotlib
# ax2 = plt.subplot(gs[iplane,1])
# jet_cm = colorOrder(nlines=len(session_stages))
# ax2.set_color_cycle(jet_cm)



