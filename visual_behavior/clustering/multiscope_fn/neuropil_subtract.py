#!/usr/bin/env python
# coding: utf-8

# In[1]:


import visual_behavior.ophys.mesoscope.utils as mu

# lims_sessions = mu.get_ica_done_sessions()

# list on 06/26/2020
lims_sessions = [782107878, 786144371, 788253110, 789092007, 839514418, 841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 872592724, 873720614, 874070091, 874616920, 875259383, 876303107, 880498009, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931326814, 931687751, 940145217, 940775208, 941676716, 944888114, 946015345, 947199653, 948252173, 950031363, 952430817, 954954402, 955775716, 958105827, 971922380]


# In[2]:


print(len(lims_sessions))
print(lims_sessions)


# In[20]:


import numpy as np
import os
import re
import shutil
import h5py
import matplotlib.pyplot as plt


# In[4]:


sessions_ctDone = np.array(lims_sessions)
print(sessions_ctDone.shape)
sessions_ctDone

'''
# sessions that passed data integrity test
sessions_ctDone = [839514418, 841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 867027875, 868688430, 869117575, 870352564, 870762788, 871526950, 872592724, 873720614, 874616920, 875259383, 876303107, 880498009, 886130638, 886806800, 888009781, 889944877, 902884228, 903621170, 903813946, 904418381, 925478114, 926488384, 927787876, 928414538, 929255311, 929686773, 931326814, 931687751, 940145217, 940775208, 941676716, 944888114, 946015345, 947199653, 948252173, 950031363, 952430817, 954954402, 955775716, 958105827, 971922380]
#[839514418,  841778484, 842623907, 844469521, 847758278, 848401585, 848983781, 849304162, 850667270, 850894918, 851428829, 852070825, 852794141, 853416532, 855711263, 863815473, 864458864, 865024413, 865854762, 866197765, 868688430, 869117575]
'''


# In[67]:


# copy the out.h5 file of each experiment to the session folder, also rename it as 'traces_out_{exp_id}' (do the same with 'neuropil' traces)
# we do this so we can easily handle both the np and traces files from the session folder!
        
for isct in range(len(sessions_ctDone)): # isct=0

    dest_ct = f'/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk/session_{sessions_ctDone[isct]}/'

    
    print(f'\n\nSession {isct}/{len(sessions_ctDone)}: {dest_ct}')
    list_files = os.listdir(dest_ct) # list of files in session_xxxx


    ##### Go to ica_traces and ica_neuropil folder and get the name of out.h5 files, also valid.json files.
    for trace_type in ['traces', 'neuropil']: 
        
        regex = re.compile(f'ica_{trace_type}_(.*)') # valid_{839716139}.json        
        ica_traces_folders = [string for string in list_files if re.match(regex, string)]
#         print(ica_traces_folders)


        # set experiment ids
        exp_ids = []
        for ipp in range(len(ica_traces_folders)):
            f = ica_traces_folders[ipp]
            # print(f)
            tokens = [m.start() for m in re.finditer('_', f)]
            exp0 = f[tokens[1]+1 : tokens[2]]
            exp1 = f[tokens[2]+1:]

            exp_ids.append(exp0)
            exp_ids.append(exp1)
#         print(exp_ids)    

        

        # create neuropil_corrected_manual folder in the session folder
        for exp_id in exp_ids: 
            np_corr_fold = os.path.join(dest_ct, f'neuropil_corrected_manual_{exp_id}')
            
#             print(f'\nnp_corr folder: {np_corr_fold}')
            if not os.path.exists(np_corr_fold):
                os.makedirs(np_corr_fold)

        
        
        # cd to each ica_traces folder (each plane pair folder)
        # copy the out.h5 file of each experiment to the session folder, also rename it as 'traces_out_{exp_id}' (do the same with 'neuropil' traces)
        # we do this so we can easily handle both the np and traces files from the session folder!
        
        for ifo in range(len(ica_traces_folders)):
            
            # get the list of valid json files for this plane pair
            dir_ica_traces = os.path.join(dest_ct, ica_traces_folders[ifo])
            
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


            # get the name of out and valid json files
            dest_ct_validjson = os.path.join(dest_ct, ica_traces_folders[ifo])
#             print(f'\ndestination folder: {dest_ct_validjson}')

#             for ijs in range(len(jsonfiles)):
#                 dest_ct_validjson = os.path.join(dir_ica_traces, jsonfiles[ijs])
#                 print(dest_ct_validjson)


            # copy the out.h5 file of each experiment to the session folder, also rename it as 'traces_out_{exp_id}' (same with 'neuropil')
            # we do this so we can easily handle both the np and traces files from the session folder!
            for ijs in range(len(outfiles)):
                dest_ct_out = os.path.join(dir_ica_traces, outfiles[ijs])
#                 print(dest_ct_out)

                exp_id_now = outfiles[ijs][:outfiles[ijs].find('_')]
#                 print(exp_id_now)
                        

                if os.path.exists(os.path.join(dest_ct, f'{trace_type}_out_{exp_id_now}')): # remove the files you created which dont have .h5 in their name
                    os.remove(os.path.join(dest_ct, f'{trace_type}_out_{exp_id_now}')) 

                # copy 
                dest_good_name = os.path.join(dest_ct, f'{trace_type}_out_{exp_id_now}.h5')
                
                print(f'copying {dest_good_name}')
                shutil.copy(dest_ct_out, dest_good_name)
                
            
            


# In[23]:


# Go inside each session folder, for each experiment get the np and traces files, do the subtraction, save the new trace as h5 file inside np_corr folder
# save the manually np corrected trace as "neuropil_correction_ct_man.h5"

for isct in range(len(sessions_ctDone)): # isct=0

    for r_fact in [.7, 1]: #.7 # what fraction of np signal to subtract from soma.

        dest_ct = f'/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk/session_{sessions_ctDone[isct]}/'


        print(f'\n\nSession {isct}/{len(sessions_ctDone)}: {dest_ct}')
        list_files = os.listdir(dest_ct) # list of files in session_xxxx


        # set experiment ids
        regex = re.compile(f'ica_traces_(.*)') # valid_{839716139}.json        
        ica_traces_folders = [string for string in list_files if re.match(regex, string)]
    #     print(ica_traces_folders)

        exp_ids = []
        for ipp in range(len(ica_traces_folders)):
            f = ica_traces_folders[ipp]
            # print(f)
            tokens = [m.start() for m in re.finditer('_', f)]
            exp0 = f[tokens[1]+1 : tokens[2]]
            exp1 = f[tokens[2]+1:]

            exp_ids.append(exp0)
            exp_ids.append(exp1)
        print(exp_ids)    


        for exp_id in exp_ids:
            tnam = os.path.join(dest_ct, f'traces_out_{exp_id}.h5')
            nnam = os.path.join(dest_ct, f'neuropil_out_{exp_id}.h5')
    #         print(tnam, nnam)

            # load the traces and np files
            tf = np.array(h5py.File(tnam, 'r')['data'])
            nf = np.array(h5py.File(nnam, 'r')['data'])                    
            print(tf.shape, nf.shape)


            # subtract np from soma trace
            # signal traces
            neuropil_correction_man = tf[0] - r_fact*nf[0]
            # cross-talk traces
            neuropil_correction_ct_man = tf[1] - r_fact*nf[1]


            # set the directory and file names
            np_corr_fold = os.path.join(dest_ct, f'neuropil_corrected_manual_{exp_id}')

            np_corr_file = os.path.join(np_corr_fold, f'neuropil_correction_r{r_fact}.h5')
            np_corr_ct_file = os.path.join(np_corr_fold, f'neuropil_correction_ct_r{r_fact}.h5')
    #         print(np_corr_file)

            # save the np corr file to its directory
            f = h5py.File(np_corr_file, 'w')
            dset = f.create_dataset("data", data=neuropil_correction_man) 
            f.close()

            fc = h5py.File(np_corr_ct_file, 'w')        
            dset = fc.create_dataset("data", data=neuropil_correction_ct_man)        
            fc.close()


            # make plots

            plt.figure()
            ra = np.random.randint(0,neuropil_correction_man.shape[1])

            plt.subplot(121)
            plt.plot(neuropil_correction_man.T)
            plt.xlim([ra, ra+500]);
            plt.title(r_fact)

            plt.subplot(122)            
            plt.plot(tf[0].T)
            plt.xlim([ra, ra+500]);            
            plt.title('soma trace')
            
        


# In[ ]:


# read the np subtracted files and plot them

for isct in range(len(sessions_ctDone)): # isct=0

    dest_ct = f'/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ICA_crossTalk/session_{sessions_ctDone[isct]}/'


    print(f'\n\nSession {isct}/{len(sessions_ctDone)}: {dest_ct}')
    list_files = os.listdir(dest_ct) # list of files in session_xxxx


    # set experiment ids
    regex = re.compile(f'ica_traces_(.*)') # valid_{839716139}.json        
    ica_traces_folders = [string for string in list_files if re.match(regex, string)]
#     print(ica_traces_folders)

    exp_ids = []
    for ipp in range(len(ica_traces_folders)):
        f = ica_traces_folders[ipp]
        # print(f)
        tokens = [m.start() for m in re.finditer('_', f)]
        exp0 = f[tokens[1]+1 : tokens[2]]
        exp1 = f[tokens[2]+1:]

        exp_ids.append(exp0)
        exp_ids.append(exp1)
    print(exp_ids)    

    
    # read the np_corr files for each experiment; compare the np subtracted files for r_fact=1 vs. 0.7
    for exp_id in exp_ids:
        
        
        tnam = os.path.join(dest_ct, f'traces_out_{exp_id}.h5')
        nnam = os.path.join(dest_ct, f'neuropil_out_{exp_id}.h5')

        # load the traces and np files
        tf = np.array(h5py.File(tnam, 'r')['data'])
        nf = np.array(h5py.File(nnam, 'r')['data'])                    
        
        
#         r_all = []

        plt.figure(figsize=(11,8))
        plt.suptitle(f'sess{sessions_ctDone[isct]} - exp{exp_id}')
        ra = np.random.randint(0, tf[0].shape[1])        

        plt.subplot(411)            
        plt.plot(tf[0].T)
        plt.title('soma trace')
        plt.xlim([ra, ra+500]);            

        plt.subplot(412)            
        plt.plot(nf[0].T)
        plt.title('np trace')
        plt.xlim([ra, ra+500]);            
        
        isp = 3
        for r_fact in [.7, 1]: #.7 # what fraction of np signal to subtract from soma.
        
            np_corr_fold = os.path.join(dest_ct, f'neuropil_corrected_manual_{exp_id}')

            np_corr_file = os.path.join(np_corr_fold, f'neuropil_correction_r{r_fact}.h5')
#             np_corr_ct_file = os.path.join(np_corr_fold, f'neuropil_correction_ct_r{r_fact}.h5')

            f = h5py.File(np_corr_file, 'r')
            r = np.array(f['data']).T            
#             r_all.append(r)
           
    
            # make plots
            plt.subplot(4,1,isp)
            plt.plot(r)

            plt.xlim([ra, ra+500]);
            plt.title(r_fact)
            isp = isp+1
            
        plt.subplots_adjust(hspace=0.7)            

        # save the figure
        fign = os.path.join(np_corr_fold, f'neuropil_correction.pdf')
        print(f'Saving figure:\n{fign}')
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)


# In[ ]:




