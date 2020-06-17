#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
From Sam Seid:
        
Created on Mon Jun  3 16:44:49 2019
@author: farzaneh
"""

from ophysextractor.utils.util import get_psql_dict_cursor

def get_mouse_pkls(mouse_id):
    #Pull pkl files for all behavior sessions associated with a mouse, regardless of whether there is ophys associated
    #Input mouse id
    #Output list of dictionaries, each containing LIMS ID, mouse ID, and the storage directory to the pkl file
    QUERY = '''
    SELECT bs.id, d.external_donor_name AS mouse_id,
        wkf.storage_directory || '' || wkf.filename AS pkl_path, s.name AS mouse_name
    FROM behavior_sessions AS bs

    JOIN donors d ON d.id = bs.donor_id
    JOIN well_known_files wkf ON wkf.attachable_id = bs.id
    JOIN well_known_file_types wkft ON wkf.well_known_file_type_id = wkft.id
    JOIN specimens s ON d.external_donor_name = s.external_specimen_name

    WHERE d.external_donor_name = '{0}'
        AND wkft.name = 'StimulusPickle'

    ORDER BY bs.created_at
    '''
    
    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(QUERY.format(mouse_id))
    return(lims_cursor.fetchall())
    
    
