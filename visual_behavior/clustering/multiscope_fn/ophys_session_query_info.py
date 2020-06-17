#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:44:42 2019
@author: farzaneh
"""

from ophysextractor.utils.util import mongo, get_psql_dict_cursor

def ophys_session_query_info(session_id):

    OPHYS_SESSION_QRY = '''    
        SELECT os.id,    
            os.name,    
            os.storage_directory,    
            sp.external_specimen_name,    
            TRIM(TRAILING '-' FROM TRIM(TRAILING sp.external_specimen_name FROM sp.name)) AS genotype,    
            os.date_of_acquisition,    
            u.login AS operator,    
            e.name AS rig,    
            os.parent_session_id,    
            os.workflow_state,    
            p.code AS project,    
            os.stimulus_name,    
            isi.id AS isi_experiment_id,    
            ARRAY_AGG(oe.id ORDER BY oe.id) AS ophys_experiment_ids,    
            im1.jp2 AS vasculature_image,    
            os.vasculature_image_id
    
        FROM ophys_sessions os    
            LEFT JOIN ophys_experiments oe on oe.ophys_session_id = os.id    
            JOIN specimens sp ON sp.id = os.specimen_id    
            LEFT JOIN equipment e ON e.id = os.equipment_id    
            LEFT JOIN users u ON u.id = os.operator_id    
            JOIN projects p ON p.id = os.project_id    
            LEFT JOIN isi_experiments isi ON isi.id = os.isi_experiment_id    
            LEFT JOIN images im1 ON os.vasculature_image_id = im1.id
    
        WHERE os.id = '{0}'
    
        GROUP BY os.id, sp.external_specimen_name, sp.name, u.login, e.name, p.code, isi.id, im1.jp2    
        '''
        
    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(OPHYS_SESSION_QRY.format(session_id))
    records = lims_cursor.fetchall()
    return(records)


#%%
session_id = 841303580    
ophys_session = ophys_session_query_info(session_id)
ophys_session

