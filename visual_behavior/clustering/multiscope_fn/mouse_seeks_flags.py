
records = fetch_records(project = 'meso') # record of all failed visBehMes experiments
    
intensity_df = all_fails(records)

    
#%%
from ophysextractor.utils.util import mongo, get_psql_dict_cursor
import pandas as pd


### LIMS query to get the list of failed experiments from a particular project 
def fetch_records(project = None):
    #pulls information for visual coding or visual behavior experiments and returns as a database
    if project == 'vb':
        # All failed visual behavior experiments
        all_fail_query = '''
        SELECT os.id, os.date_of_acquisition , users.login AS operator
        FROM ophys_sessions os
        JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
        JOIN projects p ON os.project_id = p.id
        JOIN users ON users.id = os.operator_id
        WHERE p.code = 'VisualBehavior'
        '''
    if project == 'vc':
        # All failed c600 experiments
        all_fail_query = '''
        SELECT os.id, os.date_of_acquisition , users.login AS operator, tag.name AS fail_tag
        FROM ophys_sessions os
        JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
        JOIN projects p ON os.project_id = p.id
        JOIN users ON users.id = os.operator_id
        JOIN ophys_experiment_tags_ophys_experiments tid ON tid.ophys_experiment_id = oe.id
        JOIN ophys_experiment_tags tag ON tid.ophys_experiment_tag_id = tag.id
        WHERE p.code = 'c600'
        '''
    if project == 'meso':
        all_fail_query = '''
        SELECT os.id, os.date_of_acquisition , users.login AS operator
        FROM ophys_sessions os
        JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
        JOIN projects p ON os.project_id = p.id
        JOIN users ON users.id = os.operator_id
        WHERE p.code = 'VisualBehaviorMultiscope'
            AND oe.workflow_state = 'failed'
        '''
    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(all_fail_query)
    records = lims_cursor.fetchall()
    return(records)
    
    

#%%
def all_fails(records = None):
    failure_tags = []
    failure_sessions = []
    failure_experiments = []
    operator = []
    rig = []
    date = []
    genotype = []
    mouse= []
    
    for record in records:
            ms_r = mongo.db.ophys_session_log.find({u'id': record[u'id']}).next()
            
            failure_tags_temp = []            
            for iexp in range(ms_r[u'ophys_experiments']): # iexp = 0                       
                if ms_r[u'ophys_experiments'][iexp][u'workflow_state'] != 'failed':
                    print(ms_r[u'ophys_experiments'][iexp]['id'])
                    
                    try:
                        failure_sessions.append(ms_r[u'id'])
                        failure_experiments.append(ms_r[u'ophys_experiments'][iexp]['id'])                        
                        operator.append(ms_r[u'operator'])
                        rig.append(ms_r[u'rig'])
                        date.append(ms_r[u'date_of_acquisition'])
                        genotype.append(ms_r[u'genotype'])
                        mouse.append(ms_r[u'external_specimen_name'])
                        if ms_r[u'failure_tags'] >0:
                            failure_tags_temp.append(ms_r[u'failure_tags'])
                    except:
                        pass
                    
                    # check overrides
                    if len(failure_tags_temp) == 0:
                        try:
                            if len(ms_r[u'overrides'][0][u'metric']) > 0:
                                failure_tags_temp.append(ms_r[u'overrides'][0][u'metric']) 
                        except:
                            pass
                    
                    # check flags
                    if len(failure_tags_temp) == 0:
                        try:
                            if ms_r[u'failure_tags'] == 0:       
                                failure_tags_temp.append([ms_r[u'flags'][0][u'metric']])
                        except:
                            pass
                    
                    #
                    flat_list = []
                    for sublist in failure_tags_temp:
                        for item in sublist:
                            flat_list.append(item)
                    
                    if len(flat_list) == 0:
                        failure_tags.append([u'no_mouseseeks_qc'])
                    else:
                        failure_tags.append(flat_list)


        #Create dataframe from lists
    intensity_df = pd.DataFrame()
    intensity_df['sessions_id'] = failure_sessions
    intensity_df['tag'] = failure_tags
    intensity_df['rig'] = rig
    intensity_df['operator'] = operator
    intensity_df['date'] = date
    intensity_df['genotype']= genotype
    intensity_df['mouse'] = mouse     

    return(intensity_df)
    