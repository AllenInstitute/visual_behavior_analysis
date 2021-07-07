# Set sessions for analysis based on the desired stage (A, B, transition, etc)

from def_funs import *

def set_mousetrainhist_allsess2an_whatsess(all_sess, dir_server_me, all_mice_id, all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1, only_1st_transit):

    #%% Load mouse_trainHist_all2.h5 file, which shows the entire training history of a mouse
    # we get the var by running, on the cluster, set_mouse_trainHist_init_pbs.py which calls set_mouse_trainHist_all2.py 

    # Remember: 
    # mouse_trainHist_all2 may have multiple rows per day! and it seems some sessions may lack "stage" name.... this is
    # because multiple pkl paths existed ... but really only is real (i guess)... i saw this for im=0...
    # anyway it should not affect our analysis below (you can get rid of repetitions if you want).

    # set file name: mouse_trainHist_all2
    analysis_name2 = 'mouse_trainHist_all2'
    name = 'all_sess_%s_.' %(analysis_name2) 
    allSessName2, h5_files = all_sess_set_h5_fileName(analysis_name2, dir_server_me)
    print(allSessName2)

    # load mouse_trainHist_all2
    mouse_trainHist_all2 = pd.read_hdf(allSessName2) #, key='all_sess') #'svm_vars')        ## Load all_sess dataframe

    #mouse_trainHist_all2,
    #len(mouse_trainHist_all2), 
    #mouse_trainHist_all2.iloc[0]

    # np.unique(mouse_trainHist_all2['mouse_id'].values)
    # np.unique(all_sess['mouse_id'].values)


    #%% Set mouse_trainHist_all_inds; it is same as mouse_trainHist_all2, except it also has a column (stage_index) that shows 
    # the index of each stage (you should mark A_habit differently from A!, right now both are called A!)
        # 0 : Ah
        # 1 : A
        # 2 : Ap
        # 4 : B
        # 5 : Bp
        # 9 is like nan ... not A_habit, A, B, A_passive, or B_passive

    # the idea is to have an easy way to find consecutive A, B sessions.... or only A sessions, etc.

    cols = ['mouse_id', 'date', 'stage', 'stage_index']
    mouse_trainHist_all_inds = pd.DataFrame([], columns=cols)

    for im in range(len(all_mice_id)): # im = 14

        mouse_id = all_mice_id[im]

        this_trainHist = pd.DataFrame([], columns=cols)

        # Get the sequence of all the behavioral stages for a mouse
        trainHist_now = mouse_trainHist_all2[mouse_trainHist_all2['mouse_id']==mouse_id]
        '''
        # Sort trainHist_now by date, and name it trainHist_sorted
        sdi = np.argsort(trainHist_now['date'].values) # seems like they are all in order of date... but lets anyway sort it to make sure
        # datetime.datetime.strptime(trainHist_now['date'].values[0], '%Y-%m-%d')
        trainHist_sorted = trainHist_now.iloc[sdi]
        '''
        trainHist_sorted = trainHist_now
        stages = list(trainHist_sorted['stage'].values)
        # list(np.unique(stages))

        # to get the ones that end in habituation or passive (anything other than A) do either of thte below: 
        #name = 'OPHYS\w+A\w+[a-zA-Z]'
        #name = 'OPHYS\w+A\B'
        # to get those that end in A:
        #name = 'OPHYS\w+A\Z'


        # Set an array, that is similar to stages, but it assigns indeces to each stage
        stages_num = np.full((np.shape(stages)), 9).astype(int) # 9 is like nan ... not A, B, A_passive, or B_passive


        # All ophys_A sessions (all A, habit, passive)
        name = 'OPHYS\w+A'  # name = 'OPHYS\_[0-9]\w+A'
        regex = re.compile(name) # + '.h5')
        ophys_sessA = [string for string in stages if re.findall(regex, string)] # string=stages[-1]
    #    ophys_sessA, len(ophys_sessA)


        # A_habit
        name = 'OPHYS\w+A_habit'
        regex = re.compile(name) # + '.h5')
        ophys_sessAh = [string for string in stages if re.findall(regex, string)] # string=stages[-1]
    #    ophys_sessAh, len(ophys_sessAh)
        # indeces of the above sessions in the list stages
        stage_Ah = np.in1d(np.array(stages), np.array(ophys_sessAh))
    #    stage_Ah, len(stage_Ah), sum(stage_Ah)
        # lets assign 0 to A_h
        stages_num[stage_Ah] = 0


        # A_passive
        name = 'OPHYS\w+A_passive'
        regex = re.compile(name) # + '.h5')
        ophys_sessAp = [string for string in stages if re.findall(regex, string)] # string=stages[-1]
    #    ophys_sessAp, len(ophys_sessAp)
        # indeces of the above sessions in the list stages
        stage_Ap = np.in1d(np.array(stages), np.array(ophys_sessAp))
    #    stage_Ap, len(stage_Ap), sum(stage_Ap)
        # lets assign 2 to A_P
        stages_num[stage_Ap] = 2 


        # A but not A_passive
        A_other = np.concatenate((np.array(ophys_sessAp), np.array(ophys_sessAh))) #np.array(ophys_sessAp)
        ophys_sessA_noOther = list(np.array(ophys_sessA)[~np.in1d(np.array(ophys_sessA), A_other)])
        # indeces of the above sessions in the list stages
        stage_A_noOther = np.in1d(np.array(stages), np.array(ophys_sessA_noOther))
    #    stage_A_noOther, len(stage_A_noOther), sum(stage_A_noOther)
        # lets assign 1 to A_noOther
        stages_num[stage_A_noOther] = 1 


        # All ophys_B sessions
        name = 'OPHYS\w+B'  # name = 'OPHYS\_[0-9]\w+A'
        regex = re.compile(name) # + '.h5')
        ophys_sessB = [string for string in stages if re.findall(regex, string)] # string=stages[-1]
    #    ophys_sessB, len(ophys_sessB)


        # ophys_B_passive sessions
        name = 'OPHYS\w+B_passive'  # name = 'OPHYS\_[0-9]\w+A'
        regex = re.compile(name) # + '.h5')
        ophys_sessBp = [string for string in stages if re.findall(regex, string)] # string=stages[-1]
    #    ophys_sessBp, len(ophys_sessBp)
        # indeces of the above sessions in the list stages
        stage_Bp = np.in1d(np.array(stages), np.array(ophys_sessBp))
    #    stage_Bp, len(stage_Bp), sum(stage_Bp)
        # lets assign 5 to A_P
        stages_num[stage_Bp] = 5


        # B but not B_passive
        if len(ophys_sessB) > 0:
            ophys_sessB_noP = list(np.array(ophys_sessB)[~np.array(np.in1d(np.array(ophys_sessB), np.array(ophys_sessBp)))])
            # indeces of the above sessions in the list stages
            stage_B_noP = np.in1d(np.array(stages), np.array(ophys_sessB_noP))
        #    stage_B_noP, len(stage_B_noP), sum(stage_B_noP)
            # lets assign 4 to B_noP
            stages_num[stage_B_noP] = 4 


        this_trainHist = trainHist_sorted
        this_trainHist.at[:,'stage_index'] = stages_num


        ##### dataframe for all mice
        mouse_trainHist_all_inds = mouse_trainHist_all_inds.append(this_trainHist)

        
        
        

    ###
    for im in range(len(all_mice_id)): # im=0    
        mouse_id = all_mice_id[im]
        print('-------------------- mouse %d --------------------' %mouse_id)
        print(mouse_trainHist_all_inds[mouse_trainHist_all_inds['mouse_id']==mouse_id]['stage_index'].values)        
        print(mouse_trainHist_all_inds[mouse_trainHist_all_inds['mouse_id']==mouse_id])
        
    # 3A --> 4B
    # 6B --> 3A
    # 3A --> 4B
    # 5B_passive --> 3A
    # 3A --> 5B_passive
    # 6B --> blank!!! --> 1A


    
    
    
    #%% Set all_sess_transit_AB: a subset of all_sess that includes data only for consecutive A-B transitions    
    # Also set all_sess_transit_A: a subset of all_sess that includes data only for A sessions (preceding any B session).    
        # 0 : Ah
        # 1 : A
        # 2 : Ap
        # 4 : B
        # 5 : Bp
        # 9 is like nan ... not A_habit, A, B, A_passive, or B_passive

    # lets find a transition : eg. 
    # A to B (1 to 4)
    # all A to all B (1/2 to 4/5)
    # 
    # also try B to A:
    # B to A (4 to 1)
    # all B to all A (4/5 to 1/2)

    
    each_row_is_one_exp = type(all_sess.iloc[0]['experiment_id'])=='str' or type(all_sess.iloc[0]['experiment_id'])==str

    all_sess_transit_AB = pd.DataFrame([]) # similar to all_sess, except it only includes data for consecutive A-B transitions
    all_sess_A_befB = pd.DataFrame([]) # only A sessions (A, Ap) that happened before a B session.
    all_sess_A_all = pd.DataFrame([]) # all A (A, Ap), regardless of whethere they happened before or after B sessions.
    all_sess_B_all = pd.DataFrame([]) # all B (B, Bp), regardless of whethere they happened before or after A sessions.
    all_sess_B_first = pd.DataFrame([]) # first B (B or Bp)
    all_sess_AB_all_NotB1 = pd.DataFrame([]) # all A and B (A, Ap, B, Bp), except for the 1st B session.
    
    for im in range(len(all_mice_id)): # im=0

        print(f'\n===========================================================\n')
        mouse_id = all_mice_id[im]

        trainHist_this_mouse = mouse_trainHist_all_inds[mouse_trainHist_all_inds['mouse_id']==mouse_id]
        all_sess_this_mouse = all_sess[all_sess['mouse_id']== mouse_id]

        strm = all_sess_this_mouse['date'].values # imaging data exists for these dates
    #    all_sess_this_mouse['stage'].values

        # make a single number by putting next together all stage indeces 
        a = trainHist_this_mouse['stage_index'].values   
        train_hist_code = ''.join(a.astype(str)) # eg : '999999999999999999999999999999999111121112145491'

        print(f'Mouse {mouse_id} training history (0:Ah , 1:A , 2:Ap , 4:B , 5:Bp , 9:rest):')
        print(f'{train_hist_code}\n')
        
#         print(f'Training:\n')
#         print(trainHist_this_mouse)
        
        print(f'Imaging data:\n')
        iso = np.argsort(all_sess_this_mouse['date'].values)
        print(f"{all_sess_this_mouse[['stage', 'date']].iloc[iso]}\n")
        
        
        ########################## Set all_sess for all A sessions ##########################    
        ########## (A, Ap, before and after B sessions would be fine, as long as it is A!) ########
        ##########################################################################################
        print('----------- All A sessions -------------')
    #    A_Ap_inds = [1,2]    
    #    all_0 = [m.start() for m in re.finditer('0', train_hist_code)]
        all_1 = [m.start() for m in re.finditer('1', train_hist_code)]
        all_2 = [m.start() for m in re.finditer('2', train_hist_code)]

        all_A_inds = np.sort(np.concatenate(([all_1, all_2])))
    #    if len(all_A_inds)==0:
    #        print(trainHist_this_mouse['stage'].values)
    #        print(train_hist_code)
    #        print(train_hist_code[all_A_inds]) # should be all 1 or 2
    #        print('mouse %d does not have any A sessions!!' %mouse_id)
        # get all stages between first A and first B (should include A, Ap)
    #    trainHist_this_mouse['stage'].values[firstA:firstB] # should be all A (including habit and passive sessions)
    #    trainHist_this_mouse['stage'].values[A_inds]
        print(trainHist_this_mouse.iloc[all_A_inds])

        strs = trainHist_this_mouse.iloc[all_A_inds]['date'].values
        a = np.in1d(strm, strs) # rows in all_sess_this_mouse that have the desired stage transitions 
        # remember strs may not exist in all_sess ... either because convert code is not run yet, or because the session is not valid
        if each_row_is_one_exp:
            print(f'\nImaging data includes {sum(a)/num_planes} total A sessions.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!
        else:
            print(f'\nImaging data includes {sum(a)} total A sessions.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!
        all_sess_A_all = all_sess_A_all.append(all_sess_this_mouse[a])



        ########################## Set all_sess for all B sessions ##########################    
        ########## (B, Bp, before and after A sessions would be fine, as long as it is B!) ########
        ##########################################################################################
        print('----------- All B sessions -------------')
    #    A_Ap_inds = [1,2]    
    #    all_0 = [m.start() for m in re.finditer('0', train_hist_code)]
        all_1 = [m.start() for m in re.finditer('4', train_hist_code)]
        all_2 = [m.start() for m in re.finditer('5', train_hist_code)]

        all_B_inds = np.sort(np.concatenate(([all_1, all_2])))
    #    if len(all_A_inds)==0:
    #        print(trainHist_this_mouse['stage'].values)
    #        print(train_hist_code)
    #        print(train_hist_code[all_A_inds]) # should be all 1 or 2
    #        print('mouse %d does not have any A sessions!!' %mouse_id)
        # get all stages between first A and first B (should include A, Ap)
    #    trainHist_this_mouse['stage'].values[firstA:firstB] # should be all A (including habit and passive sessions)
    #    trainHist_this_mouse['stage'].values[A_inds]
        print(trainHist_this_mouse.iloc[all_B_inds])

        strs = trainHist_this_mouse.iloc[all_B_inds]['date'].values
        a = np.in1d(strm, strs) # rows in all_sess_this_mouse that have the desired stage transitions 
        # remember strs may not exist in all_sess ... either because convert code is not run yet, or because the session is not valid
        if each_row_is_one_exp:
            print(f'\nImaging data includes {sum(a)/num_planes} total B sessions.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!
        else:
            print(f'\nImaging data includes {sum(a)} total B sessions.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!
        all_sess_B_all = all_sess_B_all.append(all_sess_this_mouse[a])


        
        ########################## Set all_sess for the 1st B session ##########################    
        ########## (B or Bp ########
        ##########################################################################################
        print('----------- 1st B session -------------')
#         first_b = train_hist_code.find('4')
        all_1 = [m.start() for m in re.finditer('4', train_hist_code)]
        all_2 = [m.start() for m in re.finditer('5', train_hist_code)]

        all_B_inds = np.sort(np.concatenate(([all_1, all_2])))
        first_b = all_B_inds[0]
        
        all_B_inds = [first_b]
    #    if len(all_A_inds)==0:
    #        print(trainHist_this_mouse['stage'].values)
    #        print(train_hist_code)
    #        print(train_hist_code[all_A_inds]) # should be all 1 or 2
    #        print('mouse %d does not have any A sessions!!' %mouse_id)
        # get all stages between first A and first B (should include A, Ap)
    #    trainHist_this_mouse['stage'].values[firstA:firstB] # should be all A (including habit and passive sessions)
    #    trainHist_this_mouse['stage'].values[A_inds]
        print(trainHist_this_mouse.iloc[all_B_inds])

        strs = trainHist_this_mouse.iloc[all_B_inds]['date'].values
        a = np.in1d(strm, strs) # rows in all_sess_this_mouse that have the desired stage transitions 
        # remember strs may not exist in all_sess ... either because convert code is not run yet, or because the session is not valid
        if each_row_is_one_exp:
            print(f'\nImaging data includes {sum(a)/num_planes} B1 sessions.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!            
        else:
            print(f'\nImaging data includes {sum(a)} B1 sessions.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!
        all_sess_B_first = all_sess_B_first.append(all_sess_this_mouse[a])
        
        

        
        ########################## Set all_sess for all A and B sessions except the 1st B session ##########################    
        ########## (A, Ap, B, Bp, except B1) ########
        ##########################################################################################
        print('----------- All A and B sessions except the 1st B session -------------')

        # A
        all_1 = [m.start() for m in re.finditer('1', train_hist_code)]
        all_2 = [m.start() for m in re.finditer('2', train_hist_code)]

        all_A_inds = np.sort(np.concatenate(([all_1, all_2])))

        # B
        all_1 = [m.start() for m in re.finditer('4', train_hist_code)]
        all_2 = [m.start() for m in re.finditer('5', train_hist_code)]

        all_B_inds = np.sort(np.concatenate(([all_1, all_2])))        
        all_B_inds = all_B_inds[1:] # exclude the first B session
        
        # all A and B
        all_B_inds = np.sort(np.concatenate(([all_A_inds, all_B_inds])))
        
    #    if len(all_A_inds)==0:
    #        print(trainHist_this_mouse['stage'].values)
    #        print(train_hist_code)
    #        print(train_hist_code[all_A_inds]) # should be all 1 or 2
    #        print('mouse %d does not have any A sessions!!' %mouse_id)
        # get all stages between first A and first B (should include A, Ap)
    #    trainHist_this_mouse['stage'].values[firstA:firstB] # should be all A (including habit and passive sessions)
    #    trainHist_this_mouse['stage'].values[A_inds]
        print(trainHist_this_mouse.iloc[all_B_inds])

        strs = trainHist_this_mouse.iloc[all_B_inds]['date'].values
        a = np.in1d(strm, strs) # rows in all_sess_this_mouse that have the desired stage transitions 
        # remember strs may not exist in all_sess ... either because convert code is not run yet, or because the session is not valid
        if each_row_is_one_exp:
            print(f'\nImaging data includes {sum(a)/num_planes} total A & B sessions excluding B1.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!
        else:
            print(f'\nImaging data includes {sum(a)} total A & B sessions excluding B1.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!            
        all_sess_AB_all_NotB1 = all_sess_AB_all_NotB1.append(all_sess_this_mouse[a])

        
        
        
        ################## Set all_sess for all A sessions before any B session ##################    
        ##########################################################################################
        print('----------- A sessions before any B -------------')
        # find the first "4", take all 1 and 2 before it.
        # note: sometimes multiple A-B transitions exists, below you are just taking only those A that preced the first B! 
        firstA = train_hist_code.find('1')
        firstB = train_hist_code.find('4')
        if firstB==-1: # there was no B session, so we take all sessions after firstA
            firstB = len(train_hist_code)
        A_inds = np.arange(firstA, firstB)
        # get all stages between first A and first B (should include A, Ap ... in a weird case (like mouse 453911) it may include Ah too!)
    #    train_hist_code[firstA:firstB] # should be all 1 or 2
    #    trainHist_this_mouse['stage'].values[firstA:firstB] # should be all A (including habit and passive sessions)
    #    trainHist_this_mouse['stage'].values[A_inds]
        print(trainHist_this_mouse.iloc[A_inds])

        strs = trainHist_this_mouse.iloc[A_inds]['date'].values
        a = np.in1d(strm, strs) # rows in all_sess_this_mouse that have the desired stage transitions 
        # remember strs may not exist in all_sess ... either because convert code is not run yet, or because the session is not valid
        if each_row_is_one_exp:
            print(f'\nImaging data includes {sum(a)/num_planes} A sessions before any B session.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!
        else:
            print(f'\nImaging data includes {sum(a)} A sessions before any B session.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!
        all_sess_A_befB = all_sess_A_befB.append(all_sess_this_mouse[a])



        ################## Set all_sess for only A-->B transitions (A_last, B_first sessions) ##################
        ##########################################################################################
        print('----------- A-->B transitions -------------')
        # all A to all B (1/2 to 4/5)
        # look up train_hist_code for 14, 15, 24, 25
        # 1 : A    # 2 : Ap    # 4 : B    # 5 : Bp
        # find indeces of a given transition in "train_hist_code"    
        t_14 = [m.start() for m in re.finditer('14', train_hist_code)]
        t_15 = [m.start() for m in re.finditer('15', train_hist_code)]
        t_24 = [m.start() for m in re.finditer('24', train_hist_code)]
        t_25 = [m.start() for m in re.finditer('25', train_hist_code)]
        print(f'Number of AB, ABp, ApB, ApBp transitions:')
        print(len(t_14), len(t_15), len(t_24), len(t_25))

        # remember a mouse may have multiple transitions! .... you should decide how you want to average them; the two transitions have different histories!
        if only_1st_transit:
            t_14 = [t_14[0]] if len(t_14)>0 else t_14
            t_15 = [t_15[0]] if len(t_15)>0 else t_15
            t_24 = [t_24[0]] if len(t_24)>0 else t_24
            t_25 = [t_25[0]] if len(t_25)>0 else t_25

        if len(t_14)==0:
            print('mouse %d does not have any A_active to B transition sessions!!' %mouse_id)

        # Go through each transition of type A-->B 
        # for now, lets not include transitions to/from passive sessions
        for i in range(len(t_14)): # i=0

            transit_inds = np.arange(t_14[i], t_14[i]+2) # take A_last and the one after (which is B_first)        
            print(trainHist_this_mouse.iloc[transit_inds])                

           # Find rows in all_sess that match the transition dates of mouse_trainHist_all_inds        
            strs = trainHist_this_mouse.iloc[transit_inds]['date'].values
            a = np.in1d(strm, strs) # rows in all_sess_this_mouse that have the desired stage transitions 
            # remember strs may not exist in all_sess ... either because convert code is not run yet, or because the session is not valid
            if sum(a)>0: 
                if each_row_is_one_exp:
                    print(f'\nImaging data includes {sum(a)/num_planes} A-B transitions.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!
                else:   
                    print(f'\nImaging data includes {sum(a)-1} A-B transitions.\n') # for each transition, we expect to have 16 True elements in a. (8 for session A planes, 8 for session B planes); if it is only 8, then it means one of the sessions is not valid!
            else:
                print(f'\nImaging data includes no A-B transitions.\n')
    #        print(all_sess_this_mouse[a].loc[0].iloc[0], all_sess_this_mouse[a].loc[0].iloc[1])
    
            ### NOTE: double check below
            if each_row_is_one_exp: # each row of all_sess is for 1 experiment
                if ~(sum(a) >= 2*num_planes): # we need at least 2 sessions! (hence data from 16 planes) when studying transitions
                    print('At least one of the sessions is not valid, hence excluding!')
                else:
                    all_sess_transit_AB = all_sess_transit_AB.append(all_sess_this_mouse[a])
            else:            
                all_sess_transit_AB = all_sess_transit_AB.append(all_sess_this_mouse[a])

                
                
                
    print(f'======================== Summary ==========================')            

    if each_row_is_one_exp: # svm analysis: each row of all_sess is one experiment, so we divide the numbers below by 8 to reflect number of sessions.
        print('\nnumber of all A sessions (before or after B): %d' %(len(all_sess_A_all)/num_planes))
        print('number of all B sessions (before or after A): %d' %(len(all_sess_B_all)/num_planes))        
        print('number of first B sessions: %d' %(len(all_sess_B_first)/num_planes))        
        print('number of all A and B sessions, except for B1: %d' %(len(all_sess_AB_all_NotB1)/num_planes))                
        print('number of A sessions (before any B): %d' %(len(all_sess_A_befB)/num_planes))
        print('number of A-B sessions: %d' %(len(all_sess_transit_AB)/num_planes))
        
    else: # each row of all_sess includes all 8 experiments; omissions_peak analysis.
        print('\nnumber of all A sessions (before or after B): %d' %len(all_sess_A_all))
        print('number of all B sessions (before or after A): %d' %len(all_sess_B_all))        
        print('number of first B sessions: %d' %len(all_sess_B_first))        
        print('number of all A and B sessions, except for B1: %d' %len(all_sess_AB_all_NotB1))
        print('number of A sessions (before any B): %d' %len(all_sess_A_befB))
        print('number of A-B sessions: %d' %len(all_sess_transit_AB))

        
    #### sanity check:
    a = all_sess['stage'].values

    aa = np.array([a[i].find('_A') for i in range(len(a))])
    aaa = sum(aa!=-1) == len(all_sess_A_all)
    if aaa==False:
        sys.exit("Something doesn't make sense! number of A sessions dont match up between all_sess_A_all computed above, and all_sess['stage']")
        
    aa = np.array([a[i].find('_B') for i in range(len(a))])
    aaa = sum(aa!=-1) == len(all_sess_B_all)
    if aaa==False:
        sys.exit("Something doesn't make sense! number of B sessions dont match up between all_sess_B_all computed above, and all_sess['stage']")
        
        

    #%% Decide which all_sess you want to plot below

    # all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1 = 2 #1

    if all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==0:
        all_sess_2an = all_sess # includes all sessions  
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==1: 
        all_sess_2an = all_sess_transit_AB # includes only A-->B transitions (ie consecutive A,B sessions)
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==2: 
        all_sess_2an = all_sess_A_befB # includes A sessions before B  
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==3: 
        all_sess_2an = all_sess_A_all # includes all A sessions before or after B
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==4: 
        all_sess_2an = all_sess_B_all # includes all B sessions before or after A
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==5: 
        all_sess_2an = all_sess_B_first # includes the 1st B session
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==6: 
        all_sess_2an = all_sess_AB_all_NotB1 # includes all A and B sessions except for B1
        

    # figure name to be saved
    if all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==0:       
        whatSess = '_AallBall'
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==1:
        if only_1st_transit==1:
            whatSess = '_ABtransFirst'  
        elif only_1st_transit==0:
            whatSess = '_ABtransAll'
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==2:       
        whatSess = '_AbefB'
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==3:       
        whatSess = '_Aall'
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==4:       
        whatSess = '_Ball'
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==5:       
        whatSess = '_Bfirst'
    elif all_ABtransit_AbefB_Aall_Ball_Bfirst_ABallButB1==6:       
        whatSess = '_ABallButB1'
    
    
    
    
    return all_sess_2an, whatSess