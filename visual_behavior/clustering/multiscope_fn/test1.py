#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:07:02 2019

@author: farzaneh
"""


omitted_flash_frame_log:
pkl['items']['behavior']['omitted_flash_frame_log']



#%%
core_data.keys()
Out[108]: 
['running',
 'licks',
 'log',
 'visual_stimuli',
 'time',
 'trials',
 'rewards',
 'image_set',
 'omitted_stimuli',
 'metadata']

core_data['omitted_stimuli']

    
#%%
vars(type(dataset)).keys()    
sorted(list(vars(type(dataset)).keys()))


#%%
lims_id = experiment_id
cache_dir = cache_dir


#%%
for key in pkl:
    print(key)
   
#%%    
pkl.keys()

['unpickleable',
 'items',
 'start_time',
 'script',
 'threads',
 'stop_time',
 'platform_info']


a = pkl['items'];
aa = a['behavior']; aa


['ai',
 'trial_count',
 'ao',
 'lick_sensors',
 'rewards_dispensed',
 'volume_dispensed',
 'encoders',
 'behavior_path',
 'config_path',
 'auto_update',
 'trial_log',
 'window',
 'params',
 'config',
 'rewards',
 'unpickleable',
 'nidaq_tasks',
 'intervalsms',
 'behavior_text',
 'update_count',
 'omitted_flash_frame_log',
 'custom_output_path',
 'items',
 'stimuli',
 'cl_params',
 'sync_pulse']


#%%
# useful ones:
aa['trial_log']
aa['rewards']



