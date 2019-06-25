#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 20:20:17 2019

@author: farzaneh
"""

from ophysextractor.utils.util import get_psql_dict_cursor

#%%
def get_extract_json_file(experiment_id):
    
    QUERY='''
    SELECT storage_directory || filename 
    FROM well_known_files 
    WHERE well_known_file_type_id = 
      (SELECT id FROM well_known_file_types WHERE name = 'OphysExtractedTracesInputJson') 
    AND attachable_id = '{0}';
    '''

    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(QUERY.format(experiment_id))
    
    return(lims_cursor.fetchall()) #    return(lims_cursor.fetchone())
    
    
#%%
'''    
experiment_id = 888876939 #881003498 #887386953 #888876939 
f = get_extract_json_file(experiment_id)
extract_json_file_dir = f[0]['?column?']
print(extract_json_file_dir )
'''

#%%
'''
experiment_id = 887386953    
[RealDictRow([('?column?',
               '/allen/programs/braintv/production/neuralcoding/prod0/specimen_807248992/ophys_session_888009781/ophys_experiment_888876939/OPHYS_EXTRACT_TRACES_QUEUE_888876939_input.json')])]
    
experiment_id = 881003498    
 '/allen/programs/braintv/production/visualbehavior/prod0/specimen_837581585/ophys_session_880709154/ophys_experiment_881003498/processed/881003498_input_extract_traces.json'    
 
''' 