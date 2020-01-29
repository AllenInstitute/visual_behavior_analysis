from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
import visual_behavior.ophys.io.convert_level_1_to_level_2 as convert
from allensdk.internal.api import PostgresQueryMixin
from psycopg2 import connect, extras 


from pathlib import Path
import pandas as pd
import numpy as np

import os

get_psql_dict_cursor = convert.get_psql_dict_cursor
get_lims_data = convert.get_lims_data


######################## DATABASE QUERIES ##############


######### EXPERIMENT & CONTAINER BASICS


def get_lims_experiment_info(experiment_id):
    experiment_id = int(experiment_id)
    mixin = PostgresQueryMixin()
    #build query
    query = '''
    select 

    oe.id as experiment_id,
    oe.workflow_state,

    container.visual_behavior_experiment_container_id as container_id,

    os.date_of_acquisition,
    os.stimulus_name as stage_name,
    os.foraging_id,
    oe.workflow_state,

    specimens.name as mouse_info,
    specimens.donor_id as mouse_donor_id,
    structures.acronym as targeted_structure,
    imaging_depths.depth,
    equipment.name as rig

    from 
    ophys_experiments_visual_behavior_experiment_containers container
    join ophys_experiments oe on oe.id = container.ophys_experiment_id
    join ophys_sessions os on os.id = oe.ophys_session_id
    join specimens on specimens.id = os.specimen_id
    join structures on structures.id = oe.targeted_structure_id
    join imaging_depths on imaging_depths.id = oe.imaging_depth_id
    join equipment on equipment.id = os.equipment_id

    where oe.id = {}'''.format(experiment_id)
    
    lims_experiment_info =  mixin.select(query)

    return lims_experiment_info




######### SEGMENTATION AND CELL SPECIMENS ####################
def get_lims_cell_segmentation_run_info(experiment_id):
    """Queries LIMS via AllenSDK PostgresQuery function to retrieve information on all segmentations run in the 
        ophys_cell_segmenatation_runs table for a given experiment
    
    Returns:
        dataframe -- dataframe with the following columns:
            id {int}:  9 digit segmentation run id
            run_number {int}: segmentation run number
            ophys_experiment_id{int}: 9 digit ophys experiment id
            current{boolean}: True/False True: most current segmentation run; False: not the most current segmentation run
            created_at{timestamp}:
            updated_at{timestamp}:
    """
    mixin = PostgresQueryMixin()
    query ='''
    select *
    FROM ophys_cell_segmentation_runs
    WHERE ophys_experiment_id = {} '''.format(experiment_id)
    return mixin.select(query)


def get_objectlisttxt_location(segmentation_run_id):
    """use SQL and the LIMS well known file system to get the location information for the objectlist.txt file
        for a given cell segmentation run 
    
    Arguments:
        segmentation_run_id {int} -- 9 digit segmentation run id
    
    Returns:
        list -- list with storage directory and filename
    """
    
    QUERY = '''
    SELECT wkf.storage_directory, wkf.filename 
    FROM well_known_files wkf
    JOIN well_known_file_types wkft on wkf.well_known_file_type_id = wkft.id
    JOIN ophys_cell_segmentation_runs ocsr on wkf.attachable_id = ocsr.id
    WHERE wkft.name = 'OphysSegmentationObjects'
    AND wkf.attachable_type = 'OphysCellSegmentationRun'
    AND ocsr.id = {0}
    '''  
    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(QUERY.format(segmentation_run_id))
    objecttxt_info = (lims_cursor.fetchall())
    return objecttxt_info


def get_lims_cell_rois_table(experiment_id):
    """Queries LIMS via AllenSDK PostgresQuery function to retrieve everything in the cell_rois table for a given experiment
    
    Arguments:
        experiment_id {int} -- 9 digit unique identifier for an ophys experiment
    
    Returns:
        dataframe -- returns dataframe with the following columns:
            id:
            cell_specimen_id:
            ophys_experiment_id:
            x:
            y:
            width:
            height:
            valid_roi: boolean(true/false), whether the roi passes or fails roi filtering
            mask_matrix: boolean mask of roi
            max_correction_up:
            max_correction_down:
            max_correction_right:
            max_correction_left:
            mask_image_plane:
            ophys_cell_segmentation_run_id:

    """
    #query from AllenSDK

    mixin = PostgresQueryMixin()
    query ='''select cell_rois.*

    from 

    ophys_experiments oe
    join cell_rois on oe.id = cell_rois.ophys_experiment_id

    where oe.id = {}'''.format(experiment_id)
    lims_cell_rois_table =  mixin.select(query)
    return lims_cell_rois_table


def get_failed_roi_exclusion_labels(experiment_id):
    """Queries LIMS  roi_exclusion_labels table via AllenSDK PostgresQuery function to retrieve and build a 
         table of all failed ROIS for a particular experiment, and their exclusion labels.
         
         Failed rois will be listed multiple times/in multiple rows depending upon how many exclusion 
         labels they have.  
    
    Arguments:
        experiment_id {int} -- [9 digit unique identifier for the experiment]
    
    Returns:
        dataframe -- returns a dataframe with the following columns:
            ophys_experiment_id: 9 digit unique identifier for the experiment
            cell_roi_id:unique identifier for each roi (created after segmentation, before cell matching)
            cell_specimen_id: unique identifier for each roi (created after cell matching, so could be blank for some experiments/rois depending on processing step)
            valid_roi: boolean true/false, should be false for all entries, since dataframe should only list failed/invalid rois
            exclusion_label_name: label/tag for why the roi was deemed invalid
    """
    #query from AllenSDK
    experiment_id = int(experiment_id)
    mixin = PostgresQueryMixin()
    #build query
    query = '''
    select 

    oe.id as ophys_experiment_id,
    cell_rois.id as cell_roi_id,
    cell_rois.valid_roi,
    cell_rois.cell_specimen_id,
    el.name as exclusion_label_name

    from 

    ophys_experiments oe
    join cell_rois on oe.id = cell_rois.ophys_experiment_id
    join cell_rois_roi_exclusion_labels crel on crel.cell_roi_id = cell_rois.id
    join roi_exclusion_labels el on el.id = crel.roi_exclusion_label_id 

    where oe.id = {}'''.format(experiment_id)

    failed_roi_exclusion_labels =  mixin.select(query)
    return failed_roi_exclusion_labels





######### CONTAINER
 
def get_lims_container_info(container_id):
    container_id = int(container_id)

    mixin = PostgresQueryMixin()
    #build query
    query = '''
    SELECT
    container.visual_behavior_experiment_container_id as container_id,
    oe.id as ophys_experiment_id,
    os.stimulus_name as stage_name,
    os.foraging_id,
    oe.workflow_state,
    specimens.name as mouse_info,
    specimens.donor_id as mouse_donor_id,
    structures.acronym as targeted_structure,
    imaging_depths.depth,
    equipment.name as rig,
    os.date_of_acquisition
    
    FROM 
    ophys_experiments_visual_behavior_experiment_containers container
    join ophys_experiments oe on oe.id = container.ophys_experiment_id
    join ophys_sessions os on os.id = oe.ophys_session_id
    join structures on structures.id = oe.targeted_structure_id
    join imaging_depths on imaging_depths.id = oe.imaging_depth_id
    join specimens on specimens.id = os.specimen_id
    join equipment on equipment.id = os.equipment_id
    
    where 
    container.visual_behavior_experiment_container_id ={}'''.format(container_id)

    lims_container_info =  mixin.select(query)
    return lims_container_info









######## MTRAIN QUERY
def get_stage_name_from_mtrain_sqldb(df_with_foraging_id):
    foraging_ids = df_with_foraging_id['foraging_id'][~pd.isnull(df_with_foraging_id['foraging_id'])]
    mtrain_api = PostgresQueryMixin(dbname="mtrain", user="mtrainreader", host="prodmtrain1", password="mtrainro", port=5432)
    query = """
            SELECT
		    stages.name as stage_name, 
		    bs.id as foraging_id
		    FROM behavior_sessions bs
		    LEFT JOIN states ON states.id = bs.state_id
		    LEFT JOIN stages ON stages.id = states.stage_id
            WHERE bs.id IN ({})
        """.format(",".join(["'{}'".format(x) for x in foraging_ids]))
    mtrain_response = pd.read_sql(query, mtrain_api.get_connection())
    df_with_foraging_id = df_with_foraging_id.merge(mtrain_response, on='foraging_id', how='left')
    df_with_foraging_id = df_with_foraging_id.rename(columns={"stage_name_x":"stage_name_lims", "stage_name_y":"stage_name_mtrain"})
    return df_with_foraging_id




def get_current_segmentation_run_id(experiment_id):
    """gets the id for the current cell segmentation run for a given experiment. 
        Queries LIMS via AllenSDK PostgresQuery function.
    
    Arguments:
        experiment_id {int} -- 9 digit experiment id
    
    Returns:
        int -- current cell segmentation run id
    """
    
    segmentation_run_table = get_lims_cell_segmentation_run_info(experiment_id)
    current_segmentation_run_id = segmentation_run_table.loc[segmentation_run_table["current"]==True, ["id"][0]][0]
    return current_segmentation_run_id









def load_current_objectlisttxt_file(experiment_id):
    """loads the objectlist.txt file for the current segmentation run, then "cleans" the column names and returns a dataframe
    
    Arguments:
        experiment_id {[int]} -- 9 digit unique identifier for the experiment
    
    Returns:
        dataframe -- dataframe with the following columns: (from http://confluence.corp.alleninstitute.org/display/IT/Ophys+Segmentation)
            trace_index:The index to the corresponding trace plot computed  (order of the computed traces in file _somaneuropiltraces.h5)
            center_x: The x coordinate of the centroid of the object in image pixels
            center_y:The y coordinate of the centroid of the object in image pixels
            frame_of_max_intensity_masks_file: The frame the object mask is in maxInt_masks2.tif
            frame_of_enhanced_movie: The frame in the movie enhimgseq.tif that best shows the object
            layer_of_max_intensity_file: The layer of the maxInt file where the object can be seen
            bbox_min_x: coordinates delineating a bounding box that contains the object, in image pixels (upper left corner)
            bbox_min_y: coordinates delineating a bounding box that contains the object, in image pixels (upper left corner)
            bbox_max_x: coordinates delineating a bounding box that contains the object, in image pixels (bottom right corner)
            bbox_max_y: coordinates delineating a bounding box that contains the object, in image pixels (bottom right corner)
            area: Total area of the segmented object
            ellipseness: The "ellipticalness" of the object, i.e. length of long axis divided by length of short axis
            compactness: Compactness :  perimeter^2 divided by area
            exclude_code: A non-zero value indicates the object should be excluded from further analysis.  Based on measurements in objectlist.txt
                        0 = not excluded
                        1 = doublet cell
                        2 = boundary cell
                        Others = classified as not complete soma, apical dendrite, ....
            mean_intensity: Correlates with delta F/F.  Mean brightness of the object
            mean_enhanced_intensity: Mean enhanced brightness of the object
            max_intensity: Max brightness of the object
            max_enhanced_intensity: Max enhanced brightness of the object
            intensity_ratio: (max_enhanced_intensity - mean_enhanced_intensity) / mean_enhanced_intensity, for detecting dendrite objects
            soma_minus_np_mean: mean of (soma trace – its neuropil trace)
            soma_minus_np_std: 1-sided stdv of (soma trace – its neuropil trace)
            sig_active_frames_2_5:# frames with significant detected activity (spiking)   : Sum ( soma_trace > (np_trace + Snpoffsetmean+ 2.5 * Snpoffsetstdv)   trace_38_soma.png  See example traces attached.
            sig_active_frames_4: # frames with significant detected activity (spiking)   : Sum ( soma_trace > (np_trace + Snpoffsetmean+ 4.0 * Snpoffsetstdv)
            overlap_count: 	Number of other objects the object overlaps with
            percent_area_overlap: the percentage of total object area that overlaps with other objects
            overlap_obj0_index: The index of the first object with which this object overlaps
            overlap_obj1_index: The index of the second object with which this object overlaps
            soma_obj0_overlap_trace_corr: trace correlation coefficient between soma and overlap soma0  (-1.0:  excluded cell,  0.0 : NA)
            soma_obj1_overlap_trace_corr: trace correlation coefficient between soma and overlap soma1
    """
    current_segmentation_run_id = get_current_segmentation_run_id(experiment_id)
    objectlist_location_info = get_objectlisttxt_location(current_segmentation_run_id)
    objectlist_path = objectlist_location_info[0]['storage_directory']
    objectlist_file = objectlist_location_info[0]["filename"]
    full_name = os.path.join(objectlist_path, objectlist_file).replace('/allen','//allen') #works with windows and linux filepaths
    objectlist_dataframe = pd.read_csv(full_name)
    objectlist_dataframe = clean_objectlist_col_labels(objectlist_dataframe) #"clean" columns names to be more meaningful
    return objectlist_dataframe
