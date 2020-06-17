#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

get_ipython().magic(u'matplotlib inline')


# In[2]:


import visual_behavior.ophys.response_analysis.utilities as ut

import visual_behavior.visualization.ophys.population_summary_figures as psf
import visual_behavior.visualization.ophys.experiment_summary_figures as esf
import visual_behavior.visualization.ophys.summary_figures as sf


# In[3]:


cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'


# In[4]:


from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2


# In[5]:


experiment_id = 834279496


# In[6]:


ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir) #, plot_roi_validation=False);


# In[7]:


from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset


# In[8]:


dataset= VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)


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




