#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dirAna: directory containing analysis files
dir0: directory for saving figures
"""

import socket
import os

if socket.gethostname() == 'OSXLT1JHD5.local': # allen mac
    dirAna = "/Users/farzaneh.najafi/Documents/analysis_codes/"
    dir0 = '/Users/farzaneh.najafi/OneDrive - Allen Institute/Analysis'

elif socket.gethostname() == 'ibs-farzaneh-ux2': # allen pc
    dirAna = "/home/farzaneh/Documents/analysis_codes/"
    dir0 = '/home/farzaneh/OneDrive/Analysis'

elif socket.gethostname() == 'hpc-login.corp.alleninstitute.org': # hpc server
    dirAna = "/home/farzaneh.najafi/analysis_codes/"        
    dir0 = '/home/farzaneh.najafi/OneDrive/Analysis' # you need to create this!

elif socket.gethostname() == 'W10DTMJ02D7Z8': # jerome's windows machine
    dirAna = 'C:\\Users\\farzaneh.najafi\\Documents\\analysis_codes\\'
    dir0 = "C:\\Users\\farzaneh.najafi\\OneDrive - Allen Institute\\Analysis"
        
# dirMs = os.path.join(dirAna, 'multiscope_fn')
dirMs = os.path.join(dirAna, 'visual_behavior_analysis', 'visual_behavior', 'clustering', 'multiscope_fn')
os.chdir(dirMs)


# Path to save analysis results
dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
