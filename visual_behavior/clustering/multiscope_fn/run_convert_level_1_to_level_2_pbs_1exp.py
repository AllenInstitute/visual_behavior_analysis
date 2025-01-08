#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:55:22 2019

@author: farzaneh
"""

import sys
#sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
#import platform
#if platform.system() == 'Linux':
#    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob # flake8: noqa: E999

cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'
plot_roi_validation = False
# VisualBehaviorMultiscope , all valid experiments
experiment_ids = [841624564] #, 783477276, 783477281, 783477287, 783477293, 783477300, 783477307,
 


#%% 
python_file = r"/home/farzaneh.najafi/analysis_codes/visual_behavior_analysis/visual_behavior/ophys/io/convert_level_1_to_level_2.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ClusterJobs/JobRecords1'

job_settings = {'queue': 'braintv',
                'mem': '24g',
                'walltime': '36:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/farzaneh.najafi/anaconda3/envs/visbeh/bin/python',
        python_args=experiment_id, #cache_dir, plot_roi_validation,
        conda_env=None,
        jobname='process_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)
    
