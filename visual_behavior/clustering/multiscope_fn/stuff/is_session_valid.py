"""
Created on Thu Aug  1 15:07:27 2019
@author: farzaneh
"""

def is_session_valid(session_id, list_mesoscope_exp=[], exp_date_analyze=[]): 
    # exp_date_analyze: set to [] to analyze session_id regardless of its experiment data.
    # otherwise run the code below only if the session was recorded after exp_date_analyze
    
    session_id = int(session_id)
    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'


    DB = mongo.db.ophys_session_log 
    db_cursor_sess = DB.find({"id":session_id})


    #########  get workflow_state from mongo
    workflow_state = pd.DataFrame()
    for i in range(8): # i=0
        workflow_state.at[i, 'workflow_state'] = db_cursor_sess[0]['ophys_experiments'][i]['workflow_state']

   
    ######### get stage from mongo
    stage_mongo = get_stage_mongo(session_id)
#    a = list(db_cursor_sess[0]['name'])
#    ai = (np.argwhere([a[i]=='_' for i in range(len(a))]).flatten()[-1]).astype(int)
#    stage_mongo = str(db_cursor_sess[0]['name'][ai+1:])
        

    ######### get date from mongo        
#    DB = mongo.qc.metrics 
    f = 0
    try:
        exp_date = db_cursor_sess[0]['date_of_acquisition'].strftime('%Y-%m-%d')          
        # Get date from lims query:
        # Session_obj.data_pointer['date_of_acquisition'].strftime('%Y-%m-%d')
        
#        db_cursor_sess = DB.find({"lims_id":session_id})
#        exp_date = db_cursor_sess[0]['lims_ophys_session']['date_of_acquisition'].strftime('%Y-%m-%d')
#        e00 = ''
    except Exception as e00:
        exp_date = np.nan
        print(session_id, e00, stage_mongo)        
#    db_cursor_sess[0]['name']
    


    validity_log = pd.DataFrame([], columns=['session_id', 'lims_id', 'date', 'stage_metadata', 'stage_mongo', 'valid', 'log'])    

    if exp_date_analyze==[]:
        exp_date_analyze = 0
        
    if int(datetime.datetime.strptime(exp_date, '%Y-%m-%d').strftime('%Y%m%d')) > exp_date_analyze:
            
        # Get list of experiments using ophysextractor       
        if list_mesoscope_exp==[]:
            Session_obj = LimsOphysSession(lims_id=session_id)
            list_mesoscope_exp = Session_obj.data_pointer['ophys_experiment_ids']
            
        ##########################################################     
        ######### Go through experiments in each session #########        
        ##########################################################     
        DB = mongo.qc.metrics
        cnt = -1
        
        for indiv_id in list_mesoscope_exp: # indiv_id = list_mesoscope_exp[0] 
            
            print('\t%d' %indiv_id) 
            indiv_id = int(indiv_id)
            
            f = 0        
            db_cursor = DB.find({"lims_id":indiv_id})
            
            indiv_data = {}
            cnt = cnt+1        
    
    
            ############## set dataset       
            try:
                dataset = VisualBehaviorOphysDataset(indiv_id, cache_dir=cache_dir) # some sessions don't have roi_id, so dff_traces cannot be set
                e0 = ''
                
            except Exception as e:       
                e0 = e.args # 'cannot set dataset'
                print('\t session %d, experiment %d: %s' %(session_id, indiv_id, e0))
                session_valid = False #session_valid.at[indiv_id] = False
                f = 1
            
    
            ############## get dff_traces from dataset               
            try:
                if e0 == '':
                    indiv_data['fluo_traces'] = dataset.dff_traces
                    e1 = ''
                else:
                    e1 = 'cannot set dataset.dff_traces'
            except Exception as e1:
                session_valid = False #session_valid.at[indiv_id] = False
                f = 1
                if cnt==0:
                    print('\t session %d, experiment %d: %s' %(session_id, indiv_id, e1))
    
    
            ############## get stage variable from dataset_metadata           
            try:
                if e0 == '':
                    local_meta = dataset.get_metadata()
                    stage_metadata = local_meta['stage'].values[0]
    #                exp_date = local_meta['experiment_date'].values[0]                
    #                indiv_data['imaging_depth'] = db_cursor[0]['lims_ophys_experiment']['depth']
                    e2 = ''
                else:
                    stage_metadata = ''
                    e2 = 'cannot set dataset.get_metadata'
    #                exp_date = np.nan
            except Exception as e2:
                session_valid = False #session_valid.at[indiv_id] = False
    #            exp_date = np.nan
                f = 1
                if cnt==0:
                    print('\t session %d, experiment %d: %s' %(session_id, indiv_id, e2))
                    
            
            
            ############## get depth from mongo
            try:
                indiv_data['imaging_depth'] = db_cursor[0]['lims_ophys_experiment']['depth']
                e3 = ''
            except Exception as e:
                e3 = e.args
                session_valid = False #session_valid.at[indiv_id] = False
                f = 1
                if cnt==0:
                    print('\t session %d, experiment %d: %s' %(session_id, indiv_id, e3))
                                            

    
            ##### Take care of mouse-seeks qc... if it was set to failed, set the session to invalid!
            if workflow_state.iloc[cnt].values=='failed':
                e4 = 'failed workflow_state'
                session_valid = False
                f = 1
            else:
                e4 = ''
                
                
            ###########################        
            if f==0:    
                session_valid = True #session_valid.at[indiv_id] = True
                e0 = ''; e1 = ''; e2 = ''; e3 = ''; e4 = ''
            
    
            etot = [e0,e1,e2,e3,e4]
            validity_log.at[cnt, ['session_id', 'lims_id', 'date', 'stage_metadata', 'stage_mongo', 'valid', 'log']] = session_id, indiv_id, exp_date, stage_metadata, stage_mongo, session_valid, etot
    

    validity_log = validity_log.merge(workflow_state, left_index=True, right_index=True)
    
        
    return validity_log

 