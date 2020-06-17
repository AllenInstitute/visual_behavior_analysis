# Copy crosstalk folders to allen server (nc-ophys/Farzaneh):
# on the analysis computer: 
# cp -rv /media/rd-storage/Z/MesoscopeAnalysis/session_839208243 /allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk; 

# Only copy dff files:

#%%
import numpy as np
import shutil
import re
import os

#%% Get a list of sessions ready for post-cross talk analysis with the following code:
'''
import visual_behavior.ophys.mesoscope.utils as mu
import logging
lims_done, lims_notdone, meso_data = mu.get_lims_done_sessions()
lims_done = lims_done.session_id.drop_duplicates()
lims_done.values

# import visual_behavior.ophys.mesoscope.mesoscope as ms
# meso_data = ms.get_all_mesoscope_data() 

'''


#%%
# sessions_ctDone = np.array([839208243, 839514418, 840490733, 841303580, 841682738, 841778484, 842023261, 842364341, 842623907, 843871999, 844469521, 846871218, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 854060305, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 871906231, 872592724, 873720614, 874070091, 874616920])
sessions_ctDone = lims_done.values
sessions_ctDone.shape

for isct in range(len(sessions_ctDone)): # isct=0

    dir_ica = f'/media/rd-storage/Z/MesoscopeAnalysis/session_{sessions_ctDone[isct]}'
    list_files = os.listdir(dir_ica) # list of files in dir_planePair
    
    regex = re.compile(f'(.*)_dff.h5')
    files = [string for string in list_files if re.match(regex, string)]
    
    for ifi in range(len(files)): # loop over dff files
        source_ct = f'{dir_ica}/{files[ifi]}'
        dest_ct = f'/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk/session_{sessions_ctDone[isct]}/'
        print(f'copying {source_ct}')
        shutil.copy(source_ct, dest_ct)    

        # copy valid json files
        regex = re.compile(f'{dest_ct}/ica_traces_(.*)') # valid_{839716139}.json        
        files = [string for string in list_files if re.match(regex, string)]
        dest_ct_validjson = 
