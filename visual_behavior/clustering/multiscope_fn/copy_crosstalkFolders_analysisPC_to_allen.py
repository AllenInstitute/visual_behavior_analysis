#!/usr/bin/env python
# coding: utf-8

# In[27]:


import visual_behavior.ophys.mesoscope.utils as mu
import logging

# lims_sessions = mu.get_lims_done_sessions()
lims_sessions = mu.get_ica_done_sessions()

lims_sessions
# lims_done, lims_notdone, meso_data = mu.get_lims_done_sessions()
# lims_done = lims_done.session_id.drop_duplicates()
# lims_done.values


# In[28]:


import numpy as np
import os
import re
import shutil


# In[4]:


sessions_ctDone = np.array(lims_sessions)
print(sessions_ctDone.shape)
sessions_ctDone

# sessions that passed data integrity test
'''
sessions_ctDone = [839514418, 841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 872592724, 873720614, 874616920, 875259383, 876303107, 880498009, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931326814, 931687751, 940145217, 940775208, 941676716, 944888114, 946015345, 947199653, 948252173, 950031363, 952430817, 954954402, 955775716, 958105827, 971922380]
#[839514418,  841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 868688430, 869117575]
print(len(sessions_ctDone))
'''


# In[29]:





# In[30]:


# for ica_traces, and ica_neuropil, copy out.h5 and valid.json files

for isct in range(len(sessions_ctDone)): # isct=0

    dir_ica = f'/media/rd-storage/Z/MesoscopeAnalysis/session_{sessions_ctDone[isct]}'
    dest_ct = f'/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk/session_{sessions_ctDone[isct]}/'


    if os.path.exists(dest_ct):
        print(f'Session {sessions_ctDone[isct]} files are already copied; skipping...')
    else:
        print(f'\n\nSession {isct}/{len(sessions_ctDone)}: {dir_ica}')
        list_files = os.listdir(dir_ica) # list of files in session_xxxx


        ##### Go to ica_traces and ica_neuropil folder and copy out.h5 files, also valid.json files.
        for trace_type in ['traces', 'neuropil']: 
            regex = re.compile(f'ica_{trace_type}_(.*)') # valid_{839716139}.json        
            ica_traces_folders = [string for string in list_files if re.match(regex, string)]
    #         print(ica_traces_folders)


            # cd to each ica_traces folder (each plane pair folder), and copy the valid json file there to your allen directory    
            for ifo in range(len(ica_traces_folders)):
                # get the list of valid json files for this plane pair
                dir_ica_traces = os.path.join(dir_ica, ica_traces_folders[ifo])
                list_files2 = os.listdir(dir_ica_traces) # list of files in ica_traces
        #         print(list_files2)
                # valid json
                regex = re.compile(f'(.*)_valid.json')
                jsonfiles = [string for string in list_files2 if re.match(regex, string)]
        #         print(jsonfiles)
                # out file
                regex = re.compile(f'(.*)_out.h5')
                outfiles = [string for string in list_files2 if re.match(regex, string)]
    #             print(outfiles)


                # copy valid json files to your directory
                dest_ct_validjson = os.path.join(dest_ct, ica_traces_folders[ifo])
                print(f'\ndestination folder: {dest_ct_validjson}')
                if not os.path.exists(dest_ct_validjson):
                    os.makedirs(dest_ct_validjson)

                for ijs in range(len(jsonfiles)):
                    source_ct_validjson = os.path.join(dir_ica_traces, jsonfiles[ijs])
        #             print(source_ct_validjson)

                    print(f'copying {source_ct_validjson}')        
                    shutil.copy(source_ct_validjson, dest_ct_validjson)


                for ijs in range(len(outfiles)):
                    source_ct_out = os.path.join(dir_ica_traces, outfiles[ijs])
        #             print(source_ct_out)

                    print(f'copying {source_ct_out}')        
                    shutil.copy(source_ct_out, dest_ct_validjson)
        
print('Done with copying all out.h5 and valid.json files of all sessions!')         


# In[23]:


# loop over ica sessions; copy dff ica files and valid json files into your directory on allen server

for isct in range(len(sessions_ctDone)): # isct=0
    dir_ica = f'/media/rd-storage/Z/MesoscopeAnalysis/session_{sessions_ctDone[isct]}'
    dest_ct = f'/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk/session_{sessions_ctDone[isct]}/'
    
    if os.path.exists(dest_ct):
        print(f'Session {sessions_ctDone[isct]} files are already copied; skipping...')

    else:
        print(f'\n\nSession {isct}/{len(sessions_ctDone)}: {dir_ica}')
        list_files = os.listdir(dir_ica) # list of files in session_xxxx

        regex = re.compile(f'(.*)_dff_ct.h5')
        dfffiles = [string for string in list_files if re.match(regex, string)]
    #     print(dfffiles)

        
        #### copy dff files for each session
        print(f'\ndesination folder: {dest_ct}')
        if not os.path.exists(dest_ct):
            os.makedirs(dest_ct)

        for ifi in range(len(dfffiles)): # loop over dff files
            source_ct = os.path.join(dir_ica, dfffiles[ifi])

            print(f'copying {source_ct}')
            shutil.copy(source_ct, dest_ct)    

        '''
        ### copy valid json files that are inside each ica_traces folders (plane pair folders) of each session
        regex = re.compile(f'ica_traces_(.*)') # valid_{839716139}.json        
        ica_traces_folders = [string for string in list_files if re.match(regex, string)]
        # print(ica_traces_folders)

        # cd to each ica_traces folder (each plane pair folder), and copy the valid json file there to your allen directory    
        for ifo in range(len(ica_traces_folders)):
            # get the list of valid json files for this plane pair
            dir_ica_traces = os.path.join(dir_ica, ica_traces_folders[ifo])
            list_files2 = os.listdir(dir_ica_traces) # list of files in ica_traces
            regex = re.compile(f'valid_(.*).json')
            jsonfiles = [string for string in list_files2 if re.match(regex, string)]
        #     print(jsonfiles)

            # copy valid json files to your directory
            dest_ct_validjson = os.path.join(dest_ct, ica_traces_folders[ifo])
            print(f'\ndestination folder: {dest_ct_validjson}')
            if not os.path.exists(dest_ct_validjson):
                os.makedirs(dest_ct_validjson)


            for ijs in range(len(jsonfiles)):
                source_ct_validjson = os.path.join(dir_ica_traces, jsonfiles[ijs])
    #             print(source_ct_validjson)

                print(f'copying {source_ct_validjson}')        
                shutil.copy(source_ct_validjson, dest_ct_validjson)
        '''
        
        
print('Done with copying all sessions!')        


# In[ ]:





# In[ ]:




