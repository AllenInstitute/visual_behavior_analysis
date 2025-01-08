#!/usr/bin/env python
# coding: utf-8

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

get_ipython().magic(u'matplotlib inline')


#%%
import visual_behavior.ophys.response_analysis.utilities as ut
import visual_behavior.visualization.ophys.population_summary_figures as psf
import visual_behavior.visualization.ophys.experiment_summary_figures as esf
import visual_behavior.visualization.ophys.summary_figures as sf

#import sys
#sys.path.append('~/Documents/analysis_codes/multiscope_fn') # "/Users/farzaneh.najafi/Documents/analysis_codes/multiscope_fn/def_funs.py")
#from def_funs import *
#os.chdir("~/Documents/analysis_codes/multiscope_fn")
#execfile("def_funs.py")

cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'


#%%
from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
# previous convert code:
#from visual_behavior.ophys.io.convert_level_1_to_level_2_old import convert_level_1_to_level_2


#%%
#experiment_id = 851428829  #850894918 #845783018# 834279496 # 841624576

# mesoscope experiments:
import ophysextractor
from ophysextractor.datasets.lims_ophys_session import LimsOphysSession

    #import visual_behavior.ophys.mesoscope.mesoscope as ms
    #meso_data = ms.get_all_mesoscope_data()
    
    
'''
Id
    833812106 
Name
    20190307_431252_4imagesB 
Parent session
    20190228_431252_1imagesA ID: 830272668 
Ophys Experiments
    20190307_431252_4imagesB ID: 834279496 
Specimen
    Slc17a7-IRES2-Cre;Camk2a-tTA;Ai93-431252 
External specimen name
    431252 
Project
    VisualBehavior 
Targeted structure
    VISp 
Imaging Depth (um)
    375 
Stimulus name
    [n/a] 
Date of acquisition
    03/07/2019 10:26 
Rig ID
    CAM2P.4 
Storage directory
    Linux: /allen/programs/braintv/production/visualbehavior/prod0/specimen_784057626/ophys_session_833812106/
    Windows: \\allen\programs\braintv\production\visualbehavior\prod0\specimen_784057626\ophys_session_833812106\
'''
    
    
#%%
### EXAMPLE VALID SESSION: 885557130

for session_id in [958772311]: #[982722967]: #[886130638, 886367984, 886806800, 887031077, 888009781, 888171877, 889944877]: #[786144371, 788253110, 789092007, 790000697, 790910226, 791329292, 791807371, 794843265, 807249534, 810278787]: # session_id = 869117575
    
    Session_obj = LimsOphysSession(lims_id=session_id)
    list_mesoscope_exp = Session_obj.data_pointer['ophys_experiment_ids']    
    print(list_mesoscope_exp)

#    i = 0
    for i in np.arange(0,8):  # 
        print('_________________________')
        print(session_id, 'experiment', i)

        experiment_id = list_mesoscope_exp[i]     
       
#        experiment_id = 924211430 #946513788 #903485705
 
        print(experiment_id)
        
        ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir, plot_roi_validation=False)
    
#    validity_log = is_session_valid(session_id)    
#    print validity_log
    
    
    #%% 
    for i in np.arange(0,8):  # 
        try:
#           i=7
            print('_________________________')
            print(session_id, 'experiment', i)

            experiment_id = list_mesoscope_exp[i]     
#            lims_id = experiment_id
            ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir, plot_roi_validation=False)

        except Exception as e:
            print(e)


#%%
"""
lims_id = list_mesoscope_exp[i]


lims_data = get_lims_data(lims_id)
analysis_dir = get_analysis_dir(lims_data, cache_dir=None, cache_on_lims_data=True)
timestamps = get_timestamps(lims_data, analysis_dir)
timestamps_stimulus = get_timestamps_stimulus(timestamps)
pkl = get_pkl(lims_data)
core_data = get_core_data(pkl, timestamps_stimulus)

metadata['stage'] = core_data['metadata']['stage']
"""



# In[7]:

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset


# In[8]:

dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)

# look what is inside dataset:
# vars(type(dataset)).keys()    
# sorted(list(vars(type(dataset)).keys()))


# In[9]:

dataset.metadata


# In[10]:

from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis 


# In[11]:

analysis = ResponseAnalysis(dataset, overwrite_analysis_files=False)


# In[12]:

tdf = analysis.trial_response_df.copy()


# In[13]:

fdf = analysis.flash_response_df.copy()


# In[14]:

omf = analysis.omitted_flash_response_df.copy()


# ### figures

# In[15]:

esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)


# In[16]:

esf.plot_roi_masks(dataset, save=True)


# In[17]:

sf.plot_image_change_response(analysis, 1, save=False)


# In[18]:

sf.plot_cell_summary_figure(analysis, 1, save=False, show=True, cache_dir=cache_dir)


# In[ ]:

for cell_order, cell_index in enumerate(fdf.cell.unique()):
    sf.plot_image_change_response(analysis, cell_index, cell_order, save=True)
    sf.plot_cell_summary_figure(analysis, cell_index, save=True, show=False, cache_dir=cache_dir)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




