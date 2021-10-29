#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gets called in svm_images_plots_setVars.py (Read the comments in that script for more info.)
Set svm_this_plane_allsess.

Created on Tue Oct  20 13:56:00 2020
@author: farzaneh
"""

#####################################################################
#####################################################################
#%% Define functions to set vars needed for plotting

def set_y_this_plane_allsess(y, num_sessions, takeAve=0): # set takeAve to 1 for meanX_allFrs (so we average across neurons)

    #%% Get data from a given plane across all sessions (assuming that the same order of planes exist in all sessions, ie for instance plane n is always the ith experiment in all sessions.)    
    # PERHAPS just forget about all of this below and do a reshape, you can double check reshape by doing the same thing on plane, area, depth 

    y_this_plane_allsess_allp = []
    for iplane in range(num_planes): # iplane=0 

        ##### get data from a given plane across all sessions
        y_this_plane_allsess = y.at[iplane] # num_sess        

        # when the mouse had only one session, we need the following to take care of the shape of y_this_plane_allsess
        if np.shape(y_this_plane_allsess)==(): # type(y_this_plane_allsess) == int: --> this didnt work bc populatin_sizes_to_try is of type numpy.int64
            if type(y_this_plane_allsess) == str:
                y_this_plane_allsess = np.array([y_this_plane_allsess])
            else:
                y_this_plane_allsess = np.array([np.float(y_this_plane_allsess)])

        if num_sessions < 2:
            y_this_plane_allsess = y_this_plane_allsess[np.newaxis,:]            
#             print(np.shape(y_this_plane_allsess), y_this_plane_allsess.dtype)


        # If y_this_plane_allsess is not a vector, take average across the 2nd dimension         
        if takeAve==1:
            nonNan_sess = np.array([np.sum(~np.isnan(y_this_plane_allsess[isess])) for isess in range(num_sessions)])            
            len_trace = np.shape(y.values[0])[0] #len(y.values[0]) #y[0].shape
            aa = np.full((num_sessions, len_trace), np.nan) # []
            for isess in range(num_sessions): # isess = 0
                if nonNan_sess[isess] > 0:
                    aa[isess] = np.mean(y_this_plane_allsess[isess], axis = 1)
            y_this_plane_allsess = aa #np.array(aa)

        y_this_plane_allsess = np.vstack(y_this_plane_allsess) # num_sess x nFrames
#        print(np.shape(y_this_plane_allsess), y_this_plane_allsess.dtype)


        y_this_plane_allsess_allp.append(y_this_plane_allsess) # .squeeze()
#        y_this_plane_allsess_allp.at[iplane] = area, depth, y_this_plane_allsess

    y_this_plane_allsess_allp = np.array(y_this_plane_allsess_allp)

    return y_this_plane_allsess_allp




def pool_sesss_planes_eachArea(area, y):
    #%%  Pool all sessions and layers for each area, do this for each mouse

    # area: ['LM', 'LM', 'LM', 'LM', 'VISp', 'VISp', 'VISp', 'VISp', 'LM', 'LM', 'LM', 'LM', 'VISp', 'VISp', 'VISp', 'VISp'] # (8*num_sessions)
    # y : (8*num_sessions) x time  or   # (8*num_sessions) 

    # set area indeces
#         area = trace_peak_allMice.iloc[im]['area'] # (8*num_sessions)
    distinct_areas, i_areas = np.unique(area, return_inverse=True)
    #    print(distinct_areas)
    if len(distinct_areas) > 2:
        sys.exit('There should not be more than two areas!! Fix the area names!')

    # below has size: num_areas x (num_layers_per_area x num_sessions) x nFrames_upsampled
    # so, 2 x (4 x num_sessions) x nFrames_upsampled
    # For each area, first take all the layers of session 1, then take all the layers of session 2
    y_pooled_sesss_planes_eachArea = np.array([y[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x nFrames_upsampled

    return y_pooled_sesss_planes_eachArea, distinct_areas, i_areas





def pool_sesss_areas_eachDepth(planes_allsess, y, num_depth=4):
    #%% For each layer (depth), pool data across sessions and areas

    # planes_allsess: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] # (8*num_sessions)
    # y: (8*num_sessions) x time   or   (8*num_sessions) 
    # num_depth: 4

    y_pooled_sesss_areas_eachDepth = []
    for idepth in range(num_depth): # idepth = 0
        # merge data with the same depth across 2 areas 
        # For each depth, first take all the areas of session 1, then take all the areas of session 2
        b = np.logical_or(planes_allsess==idepth, planes_allsess==idepth + num_depth)
        a = y[b] #np.array([t[b]]).squeeze() # (2 x num_sess) x nFrames_upsampled
        y_pooled_sesss_areas_eachDepth.append(a)
    y_pooled_sesss_areas_eachDepth = np.array(y_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2

    return y_pooled_sesss_areas_eachDepth




##############################################################################################
##############################################################################################


svm_this_plane_allsess = pd.DataFrame([], columns=columns)
cntall = 0 # index of svm_this_plane_allsess df; not really important if it starts at 1 or 0.
    
for iblock in br: # iblock=0 ; iblock=np.nan
    
#     svm_allMice_sessPooled_block_name = f'svm_allMice_sessPooled_block{iblock}'
#     svm_allMice_sessAvSd_block_name = f'svm_allMice_sessAvSd_block{iblock}'

#     allSessName_block_name = f'allSessName_block{iblock}' # not sure if we need this anymore
    
    ##############################################################################
    ##############################################################################    
    #%% Read all_see h5 file made in svm_main_post (by calling svm_init)    
    # all cre lines, a given block; (a given frames_svm; latest files saved)
    ##############################################################################
    ##############################################################################

#     a = f'(.*)_frames{frames_svm[0]}to{frames_svm[-1]}'    # {cre2ana}

    allSessName = []
#     for cre2ana in cre2ana_all: # cre2ana = cre2ana_all[0]
        
    b = '_'.join(project_codes)
    a = f'frames{frames_svm[0]}to{frames_svm[-1]}'
#     a = f'{cre2ana}_frames{frames_svm[0]}to{frames_svm[-1]}'

    if ~np.isnan(svm_blocks) and svm_blocks!=-101:
        if svm_blocks==-1: # divide trials into blocks based on the engagement state
            word = 'engaged'
        else:
            word = 'block'

        a = f'{a}_{word}{iblock}'

    name = f'all_sess_{svmn}_{a}_{b}_.'
#     name = f'all_sess_{svmn}_{a}_(.*){analysis_date}_.' # analysis_date = 20210108
    # print(name)

    allSessName_thisCre, h5_files = all_sess_set_h5_fileName(name, dir_svm, all_files=0) # all_files=1
    allSessName.append(allSessName_thisCre)
              
    print(f'\n{len(allSessName_thisCre)} all_sess files found!\n')
    
#     exec(allSessName_block_name + " = allSessName") 

    
    #%% Set block number, use the last allsess file ... this needs work.

    # a = h5_files[0].find('block')
    # if a!=-1:
    #     iblock = h5_files[0][a+5]
    #     print(f'Analyzing block {iblock}')
    # else:
    #     iblock = np.nan



    #%% Load all_sess dataframe for all cre lines, for a given block

    all_sess = pd.DataFrame()
    for ia in range(len(allSessName)):
        print(f'Loading: {allSessName[ia]}')
        all_sess_now = pd.read_hdf(allSessName[ia], key='all_sess') #'svm_vars')
        all_sess = pd.concat([all_sess, all_sess_now])
#     all_sess
#     all_sess.keys()    
#     print(len(all_sess))

    all_sess0 = copy.deepcopy(all_sess)
    
    
    #######################################################################################
    #%% Remove mesoscope sessions with unusual frame duration; note this is a temporary solution. We will later interpolated their traces.
    #######################################################################################
    print(all_sess0.shape)
    a0 = all_sess0['session_id'].isin([1050231786, 1050597678, 1051107431])
    if sum(a0)>0:
        print(f'\n\n\n Removing {sum(a0)} mesoscope experiments with unusual frame duration; note this is a temporary solution\n\n\n')
        all_sess0 = all_sess0[~a0]    
    print(all_sess0.shape)
    
    #######################################################################################
    session_ids = all_sess0['session_id'].unique()
    print(f'\n {session_ids.shape[0]} sessions found!')

    input_vars = pd.read_hdf(allSessName[ia], key='input_vars')
    time_win = input_vars.iloc[0]['time_win']
    print(f'\n{time_win}: time window that was used for response quantification.\n')


    
    #%% Fix nan values of mouse_ids/date/stage/cre (some invalid experiments of a session have nan for mouse_id; this will cause problem later; here we replace them by the mouse_id of a valid experiment of that same session)

    for sess in session_ids: # sess = session_ids[0]
        this_sess = all_sess0['session_id']==sess
        
        mouse_id = all_sess0[this_sess]['mouse_id'].values
        nans = np.array([type(m)==float for m in mouse_id]) # nans are float
        v = np.argwhere(nans==False).flatten() # valid experiments

        date = all_sess0[this_sess]['date'].values
        stage = all_sess0[this_sess]['stage'].values
        cre = all_sess0[this_sess]['cre'].values
        exp_level = all_sess0[this_sess]['experience_level'].values

        if len(v)>0:    
            v = v[0] # take the mouse_id, date and stage from the 1st valid experiment.
            all_sess0.loc[this_sess, 'mouse_id'] = mouse_id[v]
            all_sess0.loc[this_sess, 'date'] = date[v]
            all_sess0.loc[this_sess, 'stage'] = stage[v]
            all_sess0.loc[this_sess, 'cre'] = cre[v]
            all_sess0.loc[this_sess, 'experience_level'] = exp_level[v]

        else:
            print(f'Session {sess}: all experiments are invalid')


#     all_sess0[['cre', 'stage']].groupby(['stage', 'cre']).count()
    all_sess0[['cre', 'stage', 'experience_level', 'mouse_id', 'session_id', 'experiment_id']].groupby(['experience_level', 'cre']).count()



    #%% Set the stage and experience level for each session in all_sess

    session_stage_df = pd.DataFrame([], columns=['session_id', 'stage', 'experience_level'])
    stages_all_sess = []
    explevel_all_sess = []
    for isess in range(len(session_ids)): # sess = session_ids[0]
        sess = session_ids[isess]
        stages = all_sess0[all_sess0['session_id']==sess]['stage'].values
        explev = all_sess0[all_sess0['session_id']==sess]['experience_level'].values

        ts = [type(sg) for sg in stages]
        su = stages[np.in1d(ts, str)] # take the non-nan experiments for the stage value
        eu = explev[np.in1d(ts, str)]
        if len(su)>0: # not all experiments are nan.
            sun = su[0]
            eun = eu[0]
        else:
            sun = ''
            eun = ''

        session_stage_df.at[isess,:] = [sess, sun, eun]
        stages_all_sess.append(sun)
        explevel_all_sess.append(eun)

    stages_all_sess = np.array(stages_all_sess)    
    explevel_all_sess = np.array(explevel_all_sess)
    
    session_stage_df.groupby(['stage']).count()
    session_stage_df.groupby(['experience_level']).count()

    
    
    #%% Number of invalid sessions
    
    a = sum(stages_all_sess=='')    
    print(f"{a} sessions ({a*8} experiments, if mesoscope) have all experiments as NaN! These need further evaluation for why their svm files dont exist!")    

 
    
    #%% Number of invalid experiments
    
    ea = [all_sess0['av_test_data'].values[i][0] for i in range(len(all_sess0))] # the 1st element of av_test_data for each experiment; if nan, then invalid experiment.
    en_ids = all_sess0['experiment_id'][np.isnan(ea)] # experiment id of invalid experiments
#     valid_exps = ~np.isnan(ea)
    
    print(f"{sum(np.isnan(ea))} / {len(all_sess0)} experiments are NaN!\nTheir experiment ids:\n{en_ids}")

    if use_events:
        ea_evs = copy.deepcopy(ea)
    else:
        ea_dff = copy.deepcopy(ea)
        
        
    if use_same_experiments_dff_events:
        if ('ea_evs' in locals() and 'ea_dff' in locals())==False:
            sys.exit('Set both ea_evs and ea_dff before proceeding!')

            
            
    ##############################################################################
    ##############################################################################
    # Run this part after both ea_evs and ea_dff are set.
    
    all_sess00 = copy.deepcopy(all_sess0)
    
    #%% Use the same set of experiments for both events and dff analysis

    if use_same_experiments_dff_events:
        
        ea_ids = all_sess00['experiment_id'].values    

        #### set valid exps in both dff and events analyses
        # valid exps in dff
        valid_exps_dff = ~np.isnan(ea_dff)
        # valid exps in events
        valid_exps_evs = ~np.isnan(ea_evs)
        # get the common valid exps
        valid_exps_dff_evs = np.logical_and(valid_exps_dff, valid_exps_evs)
        # get the common valid exps ids
        ea_ids_valid_common = ea_ids[valid_exps_dff_evs]


        #%% Add a valid column to all_sess00, and only set those rows to valid that are valid in both dff and events analysis

        all_sess00['valid'] = False

        all_sess00.loc[valid_exps_dff_evs, 'valid'] = True
    #     all_sess00.loc[valid_exps, 'valid'] = True
    #     all_sess00.loc[valid_exps_dff, 'valid'] = True
    #     all_sess00.loc[valid_exps_evs, 'valid'] = True



        #%% Set svm class accuracy vars to nan for all invalid experiments; then run it on both events and dff ; and comapre the figures; also add stuff to the code to take care of common exp finding.

        c = ['av_train_data', 'av_test_data', 'av_test_shfl',
            'av_test_chance', 'sd_train_data', 'sd_test_data', 'sd_test_shfl',
            'sd_test_chance']

        a = all_sess00[all_sess00['valid']==False][c]
        for i in range(len(a)):
            a.iloc[i] = [np.full((len(frames_svm)), np.nan)] 
        
        
        #%% IMPORTANT: resetting the value of all_sess0
        
        all_sess0.loc[all_sess00['valid']==False, c] = a


        ### the new number of invalid experiments
        ea = [all_sess0['av_test_data'].values[i][0] for i in range(len(all_sess0))] # the 1st element of av_test_data for each experiment; if nan, then invalid experiment.
        en_ids = all_sess0['experiment_id'][np.isnan(ea)] # experiment id of invalid experiments
    #     valid_exps = ~np.isnan(ea)

        print(f"{sum(np.isnan(ea))} / {len(all_sess0)} experiments are NaN!\nTheir experiment ids:\n{en_ids}")


        
    ##############################################################################
    #%% Set the exposure_number (or better to call "retake_number") for each experiment
    # this part is all commented and moved to the end of this page; grab it if you ever need it!
    ##############################################################################
    
    
    ##############################################################################
    #%% Correlate classification accuracy with behavioral strategy
    
    if trial_type =='changes_vs_nochanges' or trial_type =='hits_vs_misses' or trial_type == 'baseline_vs_nobaseline':
        
        exec(open('svm_images_plots_corr_beh_strategy.py').read()) 
        
    ##############################################################################
    
    
    
    
    
    
    
    ############################################################################################################################################################
    ############################################################################################################################################################
    ############################################################################################################################################################
    # Now get those experiments that belong to a given ophys stage, and make plots for them.
    ############################################################################################################################################################
    ############################################################################################################################################################
    ############################################################################################################################################################
    #%%
    for session_numbers in [[1],[2],[3],[4],[5],[6]]: # 1 to 6 # ophys session stage corresponding to project_codes that we will make plots for.
#         session_numbers = [1]

        #%% If analyzing novel sessions, only take sessions that include the 1st presentation of the novel session (ie the ones without a retake of session ophys-3)
#         if np.in1d(4, session_numbers): # novel sessions
#             exposure_number_2an = 0 # # only take those rows of all_sess that are the 1st exposure of the mouse to the novel session
#         else:
#             exposure_number_2an = np.nan

        '''
        ###### NOTE: here we redefine all_sess ######
        #%% If analyzing novel sessions, remove the 2nd retakes; only analyze the 1st exposures.
        if exposure_number_2an is not np.nan: # if novel session:
            print(all_sess0.shape)
            ### BELOW does not work! because we get exposure_number from experiment_table, but somehow some of the experiments in all_sess are missing from exp_table; so we need to make exposure_number the same size as all_sess and perhaps using some large value for the missing experiments so they will get filtered out below. Or, we can set to nan the extra rows in all_sess which are missing from exp_table (note we cannot remove them because we want to keep the 8 experiments of each session!).
            all_sess = all_sess0[exposure_number==exposure_number_2an]
            print(f'Only analyzing the 1st exposure to the novel session: {len(all_sess)} sessions!')
        else:
            all_sess = copy.deepcopy(all_sess0)
            print(f'Not analyzing novel session 1, so including all sessions with any exposure_number!')
        print(all_sess.shape)
        '''

#         print(f'\nFor now including all sessions with any exposure_number!\n')
#         all_sess = copy.deepcopy(all_sess0)


        ##############################################################################
        ##############################################################################
        #%% Get those session ids (those rows of all_sess) that belong to a given stage (eg. ophys 1, identified by session_numbers)
        ##############################################################################
        ##############################################################################
    
        l = stages_all_sess
        stage_inds_all = []
        
        for ise in range(len(session_numbers)): # ise=0
            
            print(f'\n\n')
            s = session_numbers[ise]
            name = f'OPHYS_{s}_.'

            # return index of the rows in all_sess that belong to a given stage 
            regex = re.compile(name)
            stage_inds = [i for i in range(len(l)) if type(re.match(regex, l[i]))!=type(None)]
            print(f'{len(stage_inds)} sessions are in stage {name}')

            stage_inds_all.append(stage_inds)

        if len(stage_inds)==0:
            print(f'\tskipping this stage') # no sessions in {name}

        else:
            stage_inds_all = np.concatenate((stage_inds_all))    
        #     print(f'Total of {len(stage_inds_all)} sessions in stage {name}')

            session_ids_this_stage = session_ids[stage_inds_all]

            # find stage values that are nan and change them to ''
            # l = all_sess['stage'].values
            # nans = np.array([type(ll)==float for ll in l])
            # l[nans] = ''


            #%% Set all_sess only for a given stage

            # print(all_sess0.shape)
    #         a = np.logical_and(all_sess0['valid'], np.in1d(all_sess0['session_id'], session_ids_this_stage))
            a = np.in1d(all_sess0['session_id'], session_ids_this_stage)

            all_sess = all_sess0[a]

            print(f'Final size of all_sess: {all_sess.shape}')
            print(f"Existing cre lines: {all_sess['cre'].unique()}")


            #%% Define all_sess_2an, ie the final all_sess to be analyzed

            all_sess_2an = all_sess





            #%% Take care of sessions with unusual frame duration!!!    
            # make sure below works!
            '''
            frame_durs = all_sess['frame_dur'].values
            rmv = np.logical_or(frame_durs < .089, frame_durs > .1)
            if sum(rmv)>0:
                print('take a look at these sessions, and add them to set_sessions_valid, line 178 to be excluded.')
                sys.exit(f'\n\nRemoving {sum(rmv)} sessions with unexpected frame duration!!\n\n')
                all_sess = all_sess[~rmv]

            #%%    
            print(np.shape(all_sess))
            all_sess.iloc[:2]
            '''

            ## Load input_vars
            '''
            f = allSessName[0] # take input_vars from the 1st cre lines
            input_vars = pd.read_hdf(f, key='input_vars')     ## Load input_vars dataframe    

            samps_bef = input_vars['samps_bef'].iloc[0]
            samps_aft = input_vars['samps_aft'].iloc[0]
            same_num_neuron_all_planes = input_vars['same_num_neuron_all_planes'].iloc[0]
            time_win = input_vars['time_win'].iloc[0]
            # bl_percentile = input_vars['bl_percentile'].iloc[0]
            '''



            ######################################################################################################
            #%% Set a number of useful variables
            ######################################################################################################

            #%%
            frame_dur = all_sess['frame_dur'].mode().values[0]
            print(f'frame duration: {frame_dur}')

            if type(time_win)==str:
                time_win = (frames_svm*frame_dur)[[0,-1]]

            # set the entire time_trace for the flash-aligned traces, on which we applied frames_svm to get svm results.
            samps_bef_time = (samps_bef+1) * frame_dur # 1 is added bc below we do np.arange(0,-samps_bef), so we get upto one value below samps_bef
            samps_aft_time = samps_aft * frame_dur # frames_after_omission in svm_main # we trained the classifier until 30 frames after omission    
            time_trace0 = np.unique(np.concatenate((np.arange(0, -samps_bef_time, -frame_dur)[0:samps_bef+1], np.arange(0, samps_aft_time, frame_dur)[0:samps_aft])))

            # set trace_time corresponding to svm traces
            rt = np.arange(samps_bef+frames_svm[0] , min(len(time_trace0), samps_bef+frames_svm[-1]+1))
#             rt = samps_bef+frames_svm
            time_trace = time_trace0[rt]

    
            xlim = [time_trace[0], time_trace[-1]] #[-1.2, 2.25] # [-13, 24]


            # set x tick marks
            xmj = np.round(np.unique(np.concatenate((np.arange(0, time_trace[0], -.1), np.arange(0, time_trace[-1], .1)))), 2)
            xmn = [] #np.arange(-.5, time_trace[-1], 1)
            xmjn = [xmj, xmn]


            # set the time of flashes and grays for the imaging traces ### note: you should remove 0 from flashes_win_trace_index_unq_time,  because at time 0, there is no flash, there is omission!!
            flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, flashes_win_trace_index_unq, grays_win_trace_index_unq = \
                flash_gray_onset_relOmit(samps_bef, samps_aft, frame_dur)


            #%%
            all_mice_id = np.sort(all_sess['mouse_id'].unique())
            # remove nans: this should be for sessions whose all experiments are invalid
            all_mice_id = all_mice_id[[~np.isnan(all_mice_id[i]) for i in range(len(all_mice_id))]]
            
            print(f'There are {len(all_mice_id)} mice in this stage.')

            '''
            mouse_trainHist_all = pd.DataFrame([], columns=['mouse_id', 'date', 'session_id', 'stage']) # total number of sessions x 3
            for cnt, i in enumerate(all_mice_id):
                m = all_sess[all_sess.mouse_id==i].mouse_id.values
                s = all_sess[all_sess.mouse_id==i].stage.values
                d = all_sess[all_sess.mouse_id==i].date.values
                ss = all_sess[all_sess.mouse_id==i].session_id.values

                # take data from the 1st experiment of each session (because date and stage are common across all experiments of each session)
                mouse_trainHist = pd.DataFrame([], columns=['mouse_id', 'date', 'session_id', 'stage'])
                for ii in np.arange(0,len(m),8):
                    mouse_trainHist.at[ii, ['mouse_id', 'date', 'session_id', 'stage']] = m[ii] , d[ii] , ss[ii] , s[ii]  # m.iloc[np.arange(0,len(m),8)].values , d.iloc[np.arange(0,len(d),8)].values , s.iloc[np.arange(0,len(s),8)].values

                mouse_trainHist_all = mouse_trainHist_all.append(mouse_trainHist)

            print(mouse_trainHist_all)
            '''


            #%% Set plane indices for each area (V1 and LM)
            # I think you can just move this section all the way up, outside the loop.
            
            distinct_areas, i_areas = np.unique(all_sess[all_sess['mouse_id']==all_mice_id[0]]['area'].values, return_inverse=True) # take this info from any mouse ... it doesn't matter... we go with mouse 0
            i_areas = i_areas[range(num_planes)]
            # distinct_areas = np.array(['VISl', 'VISp'])
            # i_areas = np.array([0, 0, 0, 0, 1, 1, 1, 1])

            if len(distinct_areas)>1: # project_codes == ['VisualBehaviorMultiscope'] or 
                # v1 : shown as VISp
                v1_ai = np.argwhere([distinct_areas[i].find('p')!=-1 for i in range(len(distinct_areas))]).squeeze()
                # LM : shown as VISl or LM
                lm_ai = np.argwhere([(distinct_areas[i].find('l')!=-1 or distinct_areas[i].find('L')!=-1) for i in range(len(distinct_areas))]).squeeze()

                inds_v1 = np.argwhere(i_areas==v1_ai).squeeze()
                inds_lm = np.argwhere(i_areas==lm_ai).squeeze()
            
            else: # scientifica when there is only 1 area
                if distinct_areas[0] == 'VISp':
                    inds_v1 = [0]
                    inds_lm = [np.nan]
                    areas_title = ['VISp']
                    
                elif distinct_areas[0] == 'VISl':
                    inds_lm = [0]
                    inds_v1 = [np.nan]
                    areas_title = ['VISl']
                
#             print('V1 plane indeces: ', inds_v1) # [4, 5, 6, 7] 
#             print('LM plane indeces: ', inds_lm) # [0, 1, 2, 3]
            #inds_lm = np.arange(num_depth) # we can get these from the following vars: distinct_areas and i_areas[range(num_planes)]
            #inds_v1 = np.arange(num_depth, num_planes)



            ##############################################################################################
            ##############################################################################################
            ##############################################################################################
            ##############################################################################################


            #%% Set svm vars for each plane across all sessions (for each mouse)
            '''
            columns0 = ['mouse_id', 'cre', 'mouse_id_exp', 'cre_exp', 'block', 'session_stages', 'session_labs', 'num_sessions_valid', 'area_this_plane_allsess_allp', 'depth_this_plane_allsess_allp', \
                        'area', 'depth', 'plane', \
                        'n_neurons_this_plane_allsess_allp', 'n_trials_this_plane_allsess_allp', \
                        'av_meanX_avSess_eachP', 'sd_meanX_avSess_eachP', \
                        'av_test_data_this_plane_allsess_allp', 'av_test_shfl_this_plane_allsess_allp', \
                        'av_test_data_avSess_eachP', 'sd_test_data_avSess_eachP', 'av_test_shfl_avSess_eachP', 'sd_test_shfl_avSess_eachP', \
                        'av_train_data_avSess_eachP', 'sd_train_data_avSess_eachP', \
                        'av_n_neurons_avSess_eachP', 'sd_n_neurons_avSess_eachP', 'av_n_trials_avSess_eachP', 'sd_n_trials_avSess_eachP', \
                        'av_test_data_pooled_sesss_planes_eachArea', 'av_test_shfl_pooled_sesss_planes_eachArea', 'meanX_pooled_sesss_planes_eachArea', \
                        'num_sessions_valid_eachArea', 'distinct_areas', 'cre_pooled_sesss_planes_eachArea', \
                        'av_test_data_pooled_sesss_areas_eachDepth', 'av_test_shfl_pooled_sesss_areas_eachDepth', 'meanX_pooled_sesss_areas_eachDepth', \
                        'cre_pooled_sesss_areas_eachDepth', 'depth_pooled_sesss_areas_eachDepth', \
                        'peak_amp_trTsShCh_this_plane_allsess_allp', \
                        'av_peak_amp_trTsShCh_avSess_eachP', 'sd_peak_amp_trTsShCh_avSess_eachP', \
                        'peak_amp_trTsShCh_pooled_sesss_planes_eachArea', \
                        'peak_amp_trTsShCh_pooled_sesss_areas_eachDepth', \
                        ]

            if same_num_neuron_all_planes:
                columns = np.concatenate((columns0, ['n_neurons_svm_trained_this_plane_allsess_allp', 'av_n_neurons_svm_trained_avSess_eachP', 'sd_n_neurons_svm_trained_avSess_eachP'])) 
            else:
                columns = columns0

            svm_this_plane_allsess = pd.DataFrame([], columns=columns)
            '''

            # for a given stage, if there are sessions in all_sess, now loop over mice based on the order in all_mice_id
            # Note: if you want to use all sessions (not just those in all_sess_2n), replace all_sess_2n with all_sess below, also uncomment the currerntly commented out defintion of session_stages

            ### in svm_this_plane_allsess, each row is for one stage, includes data from all sessions that belong to that stage, also from all planes; in the following format: 1st session of that given stage: all planes; then 2nd session of that given stage: all planes; and so on. 

            for im in range(len(all_mice_id)): # im=0

                cntall = cntall+1
                mouse_id = all_mice_id[im]
                
                if sum((all_sess_2an['mouse_id']==mouse_id).values) <= 0:
                    print(f'mouse {im} doesnt have data!')

                else: #sum((all_sess_2an['mouse_id']==mouse_id).values) > 0: # make sure there is data for this mouse, for the specific session: A, B, etc that you care about
                    print(f'\nanalyzing mouse {mouse_id}')
    #                 svm_this_plane_allsess.at[cntall, 'block'] = iblock


                    ################################################

                    all_sess_2an_this_mouse = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]
#                     cre = all_sess_2an_this_mouse['cre'].iloc[0]

                    cre = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['cre'].iloc[0]
                    cre = cre[:cre.find('-')] # remove the IRES-Cre part

                    session_ids_now = all_sess_2an_this_mouse['session_id'].values

                    session_stages = all_sess_2an_this_mouse['stage'].values[np.arange(0, all_sess_2an_this_mouse.shape[0], num_planes)]
            #         session_stages = mouse_trainHist_all[mouse_trainHist_all['mouse_id']==mouse_id]['stage'].values
                    num_sessions = len(session_stages)     # some planes for some sessions could be nan (because they had fewer neurons than 3 (svm_min_neurs)), so below we will get the accurate number of sessions for each plane separately
            #         print(cre, num_sessions)
                
                    experience_levels = all_sess_2an_this_mouse['experience_level'].values[np.arange(0, all_sess_2an_this_mouse.shape[0], num_planes)]
                
                    ######## Set session labels (stage names are too long)
                #    session_beg_inds = np.concatenate(([0], np.cumsum(num_trs_each_sess[:-1])+1))
                    session_labs = []
                    for ibeg in range(num_sessions):
                        a = str(session_stages[ibeg])
                        us = np.argwhere([a[ich]=='_' for ich in range(len(a))]).flatten() # index of underscores
                        en = us[-1].squeeze() + 6
                        txt = str(session_stages[ibeg][15:]) # session_stages[ibeg][8:en]
                        txt = txt[0:min(3, len(txt))]

                        sn = a[us[0]+1: us[1]+1] # stage number
                        txt = sn + txt            
                        session_labs.append(txt)


                    #### replicate cre to the number of sessions ####
                    cre_exp = np.full((num_planes*num_sessions), cre)
                    mouse_id_exp = np.full((num_planes*num_sessions), mouse_id)

                    areas = all_sess_2an_this_mouse['area'] # pooled: (8*num_sessions)
                    depths = all_sess_2an_this_mouse['depth'] # pooled: (8*num_sessions)
                    planes = all_sess_2an_this_mouse.index.values # pooled: (8*num_sessions)


                    ######### Get data from a given plane across all sessions #########
                    # remember y is a pandas series; do y.values

                    # area
                    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['area']#.values # num_sess
                    np.shape(y.values)
                    area_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 1)  # 8 is number of planes; nFrames_upsampled is the length of the upsampled time array   # Get data from a given plane across all sessions

                    # depth
                    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['depth']#.values # num_sess
                    depth_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 1)  # 8 is number of planes; nFrames_upsampled is the length of the upsampled time array   # Get data from a given plane across all sessions

                    # num_neurons
                    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['n_neurons']#.values
                    n_neurons_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 1) 
                #    print(n_neurons_this_plane_allsess_allp)

                    # population size used for svm training
                    if same_num_neuron_all_planes:
                        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['population_sizes_to_try']#.values # 8 * number_of_session_for_the_mouse
                        n_neurons_svm_trained_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x nFrames_upsampled)  
                #        print(n_neurons_svm_trained_this_plane_allsess_allp)    

                    # num_trials
                    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['n_trials']#.values
                    n_trials_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 1) 


                    # peak_amp_trainTestShflChance
                    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_amp_trainTestShflChance'] # 8 * number_of_session_for_the_mouse
                    peak_amp_trTsShCh_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 4) # 4 for train, Test, Shfl, and Chance        
                    
                    # baseline 
                    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['bl_pre0'] # 8 * number_of_session_for_the_mouse
                    bl_pre0_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x 4) # 4 for train, Test, Shfl, and Chance        
                    
                    if baseline_subtract: # subtract the baseline (CA average during baseline, ie before time 0) from the evoked CA (classification accuracy)
                        peak_amp_trTsShCh_this_plane_allsess_allp = peak_amp_trTsShCh_this_plane_allsess_allp - bl_pre0_this_plane_allsess_allp
                        if im==0:
                            print(f'Subtracting out baseline to quantify class accuracy.')
                    
                    
                    # av_test_data
                    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['av_test_data']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated
                    av_test_data_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x nFrames_upsampled)  # 8 is number of planes; nFrames_upsampled is the length of the upsampled time array   # Get data from a given plane across all sessions
                #    print(im, av_test_data_this_plane_allsess_allp.shape)

                    # av_test_shfl
                    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['av_test_shfl']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated
                    av_test_shfl_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x nFrames_upsampled)  

                    # av_train_data
                    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['av_train_data']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated
                    av_train_data_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions) # size: (8 x num_sessions x nFrames_upsampled)  # 8 is number of planes; nFrames_upsampled is the length of the upsampled time array   # Get data from a given plane across all sessions
                #    print(im, av_test_data_this_plane_allsess_allp.shape)

                    # meanX
                    y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['meanX_allFrs']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated x n_neurons
                    # remember, below also takes average across neurons
                    meanX_this_plane_allsess_allp = set_y_this_plane_allsess(y, num_sessions, takeAve=1) # size: (8 x num_sessions x nFrames_upsampled)  # 8 is number of planes; nFrames_upsampled is the length of the upsampled time array   # Get data from a given plane across all sessions


                    # some planes for some sessions could be nan (because they had fewer neurons than 3 (svm_min_neurs)), so lets get the accurate number of sessions for each plane separately
                    num_sessions_valid = np.sum(~np.isnan(n_neurons_this_plane_allsess_allp), axis=1).squeeze() # 8 (shows for each plane the number of non nan sessions)



                    ############################################################
                    ######### For each plane, average vars across days #########
                    ############################################################
                    # note: vars below are also set in svm_plots_setVars_sumMice.py. I didn't take them out of here, bc below will be used for making eachMouse plots. Codes in svm_plots_setVars_sumMice.py are compatible with omissions code, and will be used for making sumMice plots.

                    # depth
                    a = depth_this_plane_allsess_allp # size: (8 x num_sessions x 1)
                    av_depth_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
                    sd_depth_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane

                    # num_neurons
                    a = n_neurons_this_plane_allsess_allp # size: (8 x num_sessions x 1)
                    av_n_neurons_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
                    sd_n_neurons_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane

                    # population size used for svm training
                    if same_num_neuron_all_planes:
                        a = n_neurons_svm_trained_this_plane_allsess_allp # size: (8 x num_sessions x 1)
                        av_n_neurons_svm_trained_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
                        sd_n_neurons_svm_trained_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane

                    # num_trials
                    a = n_trials_this_plane_allsess_allp # size: (8 x num_sessions x 1)
                    av_n_trials_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes 
                    sd_n_trials_avSess_eachP = np.nanstd(a,axis=1).squeeze() / np.sqrt(num_sessions_valid) # num_planes # st error across sessions, for each plane

                    
                    if project_codes != ['VisualBehaviorMultiscope']:
                        sqnv = np.sqrt(num_sessions_valid)
                    else:
                        sqnv = np.sqrt(num_sessions_valid)[:, np.newaxis]
                    
                    # av_test_data
                    a = av_test_data_this_plane_allsess_allp # size: (8 x num_sessions x nFrames_upsampled)
                    av_test_data_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
                    sd_test_data_avSess_eachP = np.nanstd(a,axis=1) / sqnv # num_planes x nFrames_upsampled # st error across sessions, for each plane

                    # av_test_shfl
                    a = av_test_shfl_this_plane_allsess_allp # size: (8 x num_sessions x nFrames_upsampled)
                    av_test_shfl_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
                    sd_test_shfl_avSess_eachP = np.nanstd(a,axis=1) / sqnv # num_planes x nFrames_upsampled # st error across sessions, for each plane

                    # av_train_data
                    a = av_train_data_this_plane_allsess_allp # size: (8 x num_sessions x nFrames_upsampled)
                    av_train_data_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
                    sd_train_data_avSess_eachP = np.nanstd(a,axis=1) / sqnv # num_planes x nFrames_upsampled # st error across sessions, for each plane

                    # meanX_allFrs
                    a = meanX_this_plane_allsess_allp # size: (8 x num_sessions x nFrames_upsampled)
                    av_meanX_avSess_eachP = np.nanmean(a,axis=1) # num_planes x nFrames_upsampled (interpolated times) 
                    sd_meanX_avSess_eachP = np.nanstd(a,axis=1) / sqnv # num_planes x nFrames_upsampled # st error across sessions, for each plane

                    # peak_amp_trTsShCh
                    a = peak_amp_trTsShCh_this_plane_allsess_allp # size: (8 x num_sessions x 4)
                    av_peak_amp_trTsShCh_avSess_eachP = np.nanmean(a,axis=1).squeeze() # num_planes x 4
                    sd_peak_amp_trTsShCh_avSess_eachP = np.nanstd(a,axis=1).squeeze() / sqnv # num_planes x 4 # st error across sessions, for each plane


                    #######################################################################
                    ######### For each area, pool data across sessions and depths #########            
                    #######################################################################
                    # 2 x (4 x num_sessions) 
                    
                    if project_codes == ['VisualBehaviorMultiscope']:

                        '''
                        # Note: use the codes below if you want to use the same codes as in "omissions_traces_peaks_plots_setVars_ave.py"

                        area = all_sess_2an_this_mouse['area'].values

                        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_amp_trainTestShflChance']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated; 
                        t = np.vstack(y) # (8*num_sessions) x num_frames # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time

                        trace_pooled_sesss_planes_eachArea, distinct_areas, i_areas = pool_sesss_planes_eachArea(area, t)

                        # sanity check that above is the same as what we set below
                #         np.sum(~np.equal(trace_pooled_sesss_planes_eachArea, peak_amp_trTsShCh_pooled_sesss_planes_eachArea))
                        '''
                        # NOTE: the reshape below (using order='F') moves in the following order:
                        # it first orders data from the 1st area in the following order
                        # data from all 4 depths for session 1, area 1
                        # data from all 4 depths for session 2, area 1
                        # data from all 4 depths for session 3, area 1 .... (then the same thing repeats for the 2nd area)
                        #
                        # data from all 4 depths for session 1, area 2
                        # data from all 4 depths for session 2, area 2
                        # data from all 4 depths for session 3, area 2

                        # you used to not use order='F' in the reshape below, so it was 1st all sessions for one depth, then all sessions for the other depth, etc
                        # but on Apr 13 2020 you added order='F' to make it consistent with omissions plots, which uses function pool_sesss_planes_eachArea.

                        a = area_this_plane_allsess_allp # 8 x num_sessions x 1
                        a = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # (8 x num_sessions) x 1 # do this to make sure things are exactly the same as how you do for the arrays below (reshape is confusing in python!!)
                        distinct_areas, i_areas = np.unique(a, return_inverse=True)
                    #    print(distinct_areas)
                        if len(distinct_areas) > 2:
                            sys.exit('There cannot be more than two areas!! Fix the area names!')


                        area = all_sess_2an_this_mouse['area'].values

                        t = cre_exp # (8*num_sessions)
                        cre_pooled_sesss_planes_eachArea,_,_ = pool_sesss_planes_eachArea(area, t) # 2 x (4 x num_sessions) 

                        # av_test_data, each area
                        a = av_test_data_this_plane_allsess_allp # 8 x num_sessions x nFrames_upsampled       
                        av_test_data_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x nFrames_upsampled        
                        # below has size: num_areas x (num_layers_per_area x num_sessions) x nFrames_upsampled
                        # so, 2 x (4 x num_sessions) x nFrames_upsampled
                        av_test_data_pooled_sesss_planes_eachArea = np.array([av_test_data_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x nFrames_upsampled

                        # av_test_shfl      
                        a = av_test_shfl_this_plane_allsess_allp     
                        av_test_shfl_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x nFrames_upsampled        
                        # below has size: num_areas x (num_layers_per_area x num_sessions) x nFrames_upsampled
                        # so, 2 x (4 x num_sessions) x nFrames_upsampled    
                        av_test_shfl_pooled_sesss_planes_eachArea = np.array([av_test_shfl_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x nFrames_upsampled

                        # meanX_allFrs
                        a = meanX_this_plane_allsess_allp     
                        meanX_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x nFrames_upsampled        
                        # below has size: num_areas x (num_layers_per_area x num_sessions) x nFrames_upsampled
                        # so, 2 x (4 x num_sessions) x nFrames_upsampled    
                        meanX_pooled_sesss_planes_eachArea = np.array([meanX_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x nFrames_upsampled

                        # peak_amp_trTsShCh
                        a = peak_amp_trTsShCh_this_plane_allsess_allp # 8 x num_sessions x 4
                        peak_amp_trTsShCh_pooled_sesss_planes = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]), order='F') # size: (8 x num_sessions) x 4
                        # below has size: num_areas x (num_layers_per_area x num_sessions) x 4
                        # so, 2 x (4 x num_sessions) x 4
                        peak_amp_trTsShCh_pooled_sesss_planes_eachArea = np.array([peak_amp_trTsShCh_pooled_sesss_planes[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x (8/2 x num_sessions) x 4

                        # num_sessions_valid_eachArea
                        # note: we are redefining "i_areas" below.
                        a = area_this_plane_allsess_allp[:,0,0] # 8
                        distinct_areas, i_areas = np.unique(a, return_inverse=True)
                        num_sessions_valid_eachArea = np.array([num_sessions_valid[i_areas == ida] for ida in range(len(distinct_areas))]) # 2 x 4  # area x depth_per_area


                        ###############################################################################
                        ######### For each layer (depth), pool data across sessions and areas #########            
                        ###############################################################################

                        ### Note: "xxx_pooled_sesss_areas_eachDepth" vars below have size: # 4depths x (2areas x num_sess) 
                        # (2areas x num_sess) are pooled in this way: 
                        # first all areas of session 1, then all areas of session 2, etc

                        # the first 4 depths are for one area and the 2nd four depths are for the other area, but lets code it, just in case:
                    #    depth1_area2 = 4 # index of the 1st depth of the 2nd area
                        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['area'].values
                        distinct_areas, i_areas = np.unique(y, return_inverse=True)
                        depth1_area2 = np.argwhere(i_areas).squeeze()[0] # 4 # index of the 1st depth of the 2nd area
                        if depth1_area2!=4:
                            sys.exit('Because the 8 planes are in the following order: area 1 (4 planes), then area 2 (another 4 planes), we expect depth1_area2 to be 4! What is wrong?!')


                        planes_allsess = planes # (8*num_sessions)
                        depth = depths.values

                        t = cre_exp # (8*num_sessions)        
                        cre_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess)

                        t = depth
                        depth_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, num_depth=4) # 4 x (2 x num_sessions) # each row is for a depth and shows values from the two areas of all the sessions

                        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['av_test_data']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
                        t = np.vstack(y) # (8*num_sessions) x num_frames # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time        
                        av_test_data_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2

                        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['av_test_shfl']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
                        t = np.vstack(y) # (8*num_sessions) x num_frames # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time        
                        av_test_shfl_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2

                        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['peak_amp_trainTestShflChance']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
                        t = np.vstack(y) # (8*num_sessions) x 4 # this is similar to the following var in the omissions code:  t = trace_peak_allMice.iloc[im]['trace'] # (8*num_sessions) x time        
                        peak_amp_trTsShCh_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x 4(train,test,shuffle,chance) # 4 is the number of distinct depths: depth1_area2

                        # for mouse 440631, code below gives error bc one of the sessions doesnt have valid neurons, and is just a vector of nans (1 dimensional) 
                        # so, we go with the old method for setting meanX_pooled_sesss_areas_eachDepth
                        # the only difference would be that now (2areas x num_sess) are pooled in this way: 
                        # first all sessions of one area, then all sessions of the other area are concatenated.
                        # but it doesnt matter bc we use meanX_pooled_sesss_areas_eachDepth only for eachMouse plots, where we average across areas and sessions (so the order of areas and sessions doesnt matter.)
                        '''
                        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id]['meanX_allFrs']#.values # 8 * number_of_session_for_the_mouse; each experiment: nFrames_interpolated        
                        # take average across neurons: # (8*num_sessions) x num_frames 
                        t = np.array([np.nanmean(y.values[inn], axis=1) for inn in range(len(y.values))])
                        meanX_pooled_sesss_areas_eachDepth = pool_sesss_areas_eachDepth(planes_allsess, t, depth1_area2) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2
                        '''
                        y = all_sess_2an[all_sess_2an['mouse_id']==mouse_id].index.values # (pooled: num_planes x num_sessions)    
                        p = np.reshape(y, (num_planes, num_sessions), order='F') # num_planes x num_sessions

                        # meanX
                        meanX_pooled_sesss_areas_eachDepth = []
                        for idepth in range(depth1_area2): # idepth = 0
                            # merge data with the same depth from 2 areas
                            b = np.logical_or(p==idepth, p==idepth + depth1_area2)
                            a = np.array([meanX_this_plane_allsess_allp[b]]).squeeze() # (2 x num_sess) x nFrames_upsampled
                            meanX_pooled_sesss_areas_eachDepth.append(a)
                        meanX_pooled_sesss_areas_eachDepth = np.array(meanX_pooled_sesss_areas_eachDepth) # 4 x (2 x num_sess) x nFrames_upsampled # 4 is the number of distinct depths: depth1_area2



                        ###############################################################
                        # areas.values, depth.values, plane, 
                        svm_this_plane_allsess.at[cntall, columns0] = \
                                   mouse_id, cre, mouse_id_exp, cre_exp, iblock, session_ids_now, experience_levels, session_stages, session_labs, num_sessions_valid, area_this_plane_allsess_allp, depth_this_plane_allsess_allp, \
                                   areas.values, depths.values, planes, \
                                   n_neurons_this_plane_allsess_allp, n_trials_this_plane_allsess_allp, \
                                   av_meanX_avSess_eachP, sd_meanX_avSess_eachP, \
                                   av_test_data_this_plane_allsess_allp, av_test_shfl_this_plane_allsess_allp,\
                                   av_test_data_avSess_eachP, sd_test_data_avSess_eachP, av_test_shfl_avSess_eachP, sd_test_shfl_avSess_eachP, av_train_data_avSess_eachP, sd_train_data_avSess_eachP,\
                                   av_n_neurons_avSess_eachP, sd_n_neurons_avSess_eachP, av_n_trials_avSess_eachP, sd_n_trials_avSess_eachP,\
                                   av_test_data_pooled_sesss_planes_eachArea, av_test_shfl_pooled_sesss_planes_eachArea, meanX_pooled_sesss_planes_eachArea, \
                                   num_sessions_valid_eachArea, distinct_areas, cre_pooled_sesss_planes_eachArea, \
                                   av_test_data_pooled_sesss_areas_eachDepth, av_test_shfl_pooled_sesss_areas_eachDepth, meanX_pooled_sesss_areas_eachDepth, \
                                   cre_pooled_sesss_areas_eachDepth, depth_pooled_sesss_areas_eachDepth, \
                                   peak_amp_trTsShCh_this_plane_allsess_allp, \
                                   av_peak_amp_trTsShCh_avSess_eachP, sd_peak_amp_trTsShCh_avSess_eachP,\
                                   peak_amp_trTsShCh_pooled_sesss_planes_eachArea,\
                                   peak_amp_trTsShCh_pooled_sesss_areas_eachDepth

                    
                    else: # scientifica data (visual behavior); doesnt include any area/layer pooled data
                        ###############################################################
                        # areas.values, depth.values, plane, 
                        svm_this_plane_allsess.at[cntall, columns0] = \
                                   mouse_id, cre, mouse_id_exp, cre_exp, iblock, session_ids_now, experience_levels, session_stages, session_labs, num_sessions_valid, area_this_plane_allsess_allp, depth_this_plane_allsess_allp, \
                                   areas.values, depths.values, planes, \
                                   n_neurons_this_plane_allsess_allp, n_trials_this_plane_allsess_allp, \
                                   av_meanX_avSess_eachP, sd_meanX_avSess_eachP, \
                                   av_test_data_this_plane_allsess_allp, av_test_shfl_this_plane_allsess_allp,\
                                   av_test_data_avSess_eachP, sd_test_data_avSess_eachP, av_test_shfl_avSess_eachP, sd_test_shfl_avSess_eachP, av_train_data_avSess_eachP, sd_train_data_avSess_eachP,\
                                   av_n_neurons_avSess_eachP, sd_n_neurons_avSess_eachP, av_n_trials_avSess_eachP, sd_n_trials_avSess_eachP,\
                                   distinct_areas, \
                                   peak_amp_trTsShCh_this_plane_allsess_allp, \
                                   av_peak_amp_trTsShCh_avSess_eachP, sd_peak_amp_trTsShCh_avSess_eachP
                        
                        
                        
                    if same_num_neuron_all_planes:
                        svm_this_plane_allsess.at[cntall, ['n_neurons_svm_trained_this_plane_allsess_allp']] = [n_neurons_svm_trained_this_plane_allsess_allp]
                        svm_this_plane_allsess.at[cntall, ['av_n_neurons_svm_trained_avSess_eachP']] = [av_n_neurons_svm_trained_avSess_eachP]
                        svm_this_plane_allsess.at[cntall, ['sd_n_neurons_svm_trained_avSess_eachP']] = [sd_n_neurons_svm_trained_avSess_eachP]


                # done with setting svm_this_plane_allsess for each mouse of a given ophys session and a given block (for iblock; for isession; for imouse)
#                 print(svm_this_plane_allsess.shape) 
#                 print(f'\n')
            
            
            # done with setting svm_this_plane_allsess for all mice of a given ophys session and a given block (for iblock; for isession; for imouse)
            print(svm_this_plane_allsess.shape) 
            print(f'\n')
#             svm_this_plane_allsess

            #%%
            #####################################################################################
            ####################################### PLOTS #######################################
            #####################################################################################
            '''
            if np.isnan(svm_blocks):

                # Follow this script by "svm_images_plots_eachMouse" to make plots for each mouse.
                if plot_single_mouse:
                    exec(open('svm_images_plots_eachMouse.py').read())

                # Follow this script by "svm_images_plots_setVars_sumMice.py" to set vars for making average plots across mice (for each cre line).
                exec(open('svm_images_plots_setVars_sumMice.py').read())
                exec(open('svm_images_plots_sumMice.py').read())

                # assign the vars sessPooled and sessAvSd to var names specific to each block
    #             exec(svm_allMice_sessPooled_block_name + " = svm_allMice_sessPooled")
    #             exec(svm_allMice_sessAvSd_block_name + " = svm_allMice_sessAvSd")

        #         svm_allMice_sessPooled = eval(f'svm_allMice_sessPooled_block{iblock}')

                # svm_allMice_sessPooled_block0
                # svm_allMice_sessAvSd_block0
            '''



print(svm_this_plane_allsess.shape)
svm_this_plane_allsess









#%% Set the exposure_number (or better to call "retake_number") for each experiment
# this part is all commented and moved at the end of this page; grab it if you ever need it!
"""
experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)

estbl = experiments_table.index
es = all_sess0['experiment_id'].values

#     try:
#         exposure_number = experiments_table[np.in1d(estbl, es)]['session_type_exposure_number'].values
#         # print(f'exposure numbers: {np.unique(exposure_number)}')
#     except Exception as E:
#         print('exposure_number does not exist in experiments_table!! For now it is fine because we are not really using exposure_number')
#         print(E)


#%% Are all experiments in all_sess also in experiment_table; this should always be the case, unless experiment_table changes later bc of some qc related thing!

a = sum(~np.in1d(es, estbl))
if a!=0: # print(f'All experiments in all_sess exist in experiment_table, as expected')
    sys.exit(f'\n{a} experiments in all_sess do NOT exist in experiment_table; uncanny!')


    ### make exposure_number the same size as all_sess and use some large value for the missing experiments so they will get filtered out below. 
    # turn exposure_number to a df so we know which element belongs to which experiment, add additional rows for the experiments in all_sess that are not in exposure_number, and set their value to a ridiculus value like 100.

    # i dont think the method below is useful, bc still we cant apply exposure_number onto all_sess ...
    '''
    ### set svm outputs of all_sess experiments missing from exp_table to nan
    print(f'NOTE: Turning their SVM outputs to nan!\n')

#     'meanX_allFrs', 'stdX_allFrs', 'av_train_data', 'av_test_data', 'av_test_shfl',
#     'av_test_chance', 'sd_train_data', 'sd_test_data', 'sd_test_shfl',
#     'sd_test_chance', 'peak_amp_trainTestShflChance', 'av_w_data', 'av_b_data'

#     all_sess_missing_from_estable = all_sess0[~np.in1d(es, estbl)]
#     aa = np.argwhere(~np.in1d(es, estbl))

    all_sess0.at[~np.in1d(es, estbl), 'av_train_data'] = [np.full((all_sess0.iloc[0]['av_test_data'].shape), np.nan)]    
    all_sess0.at[~np.in1d(es, estbl), 'av_test_data'] = [np.full((all_sess0.iloc[0]['av_test_data'].shape), np.nan)]
    all_sess0.at[~np.in1d(es, estbl), 'av_test_shfl'] = [np.full((all_sess0.iloc[0]['av_test_shfl'].shape), np.nan)]
    all_sess0.at[~np.in1d(es, estbl), 'av_test_chance'] = [np.full((all_sess0.iloc[0]['av_test_shfl'].shape), np.nan)]
    all_sess0.at[~np.in1d(es, estbl), 'peak_amp_trainTestShflChance'] = [np.full((all_sess0.iloc[0]['peak_amp_trainTestShflChance'].shape), np.nan)]    
    all_sess0.at[~np.in1d(es, estbl), 'meanX_allFrs'] = [np.full((all_sess0.iloc[0]['meanX_allFrs'].shape), np.nan)]        
    '''
##############################################################################
##############################################################################
"""