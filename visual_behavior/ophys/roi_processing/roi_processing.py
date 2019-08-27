from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
import visual_behavior.ophys.io.convert_level_1_to_level_2 as convert
from allensdk.internal.api import PostgresQueryMixin
from psycopg2 import connect, extras 



import pandas as pd
import numpy as np
from pathlib import Path
import os

get_psql_dict_cursor = convert.get_psql_dict_cursor
get_lims_data = convert.get_lims_data


######################## RETRIEVING DATA ##############
 ########  From LIMS 


def get_masks_from_lims_cell_rois_table(experiment_id):
    # make roi_dict with ids as keys and roi_mask_array
    lims_data = get_lims_data(experiment_id)
    lims_cell_rois_table = get_lims_cell_rois_table(lims_data['lims_id'].values[0])
    
    h = lims_cell_rois_table["height"]
    w = lims_cell_rois_table["width"]
    cell_specimen_ids = lims_cell_rois_table["cell_specimen_id"]
    
    roi_masks = {}
    for i, id in enumerate(cell_specimen_ids):
        m = roi_metrics[roi_metrics.id == id].iloc[0]
        mask = np.asarray(m['mask'])
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        binary_mask[int(m.y):int(m.y) + int(m.height), int(m.x):int(m.x) + int(m.width)] = mask
        roi_masks[int(id)] = binary_mask
    return roi_masks


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

def load_current_objectlisttxt_file(experiment_id):
    """loads the objectlist.txt file for the current segmentation run, then "cleans" the column names and returns a dataframe
    
    Arguments:
        experiment_id {[int]} -- 9 digit unique identifier for the experiment
    
    Returns:
        dataframe -- dataframe with the following columns:
            trace_index:
            center_x:
            center_y:
            frame_of_max_intensity_masks_file:
            frame_of_enhanced_movie:
            layer_of_max_intensity_file:
            bbox_min_x:
            bbox_min_y:
            bbox_max_x:
            bbox_max_y:
            area:
            ellipseness:
            compactness:
            exclude_code:
            mean_intensity:
            mean_enhanced_intensity:
            max_intensity:
            max_enhanced_intensity:
            intensity_ratio:
            soma_minus_np_mean:
            soma_minus_np_std:
            sig_active_frames_2_5:
            sig_active_frames_4:
            overlap_count:
            percent_area_overlap:
            overlap_obj0_index:
            overlap_obj1_index:
            soma_obj0_overlap_trace_corr:
            soma_obj1_overlap_trace_corr:
    """
    current_segmentation_run_id = get_current_segmentation_run_id(experiment_id)
    objectlist_location_info = get_objectlisttxt_location(current_segmentation_run_id)
    objectlist_path = objectlist_location_info[0]['storage_directory']
    objectlist_file = objectlist_location_info[0]["filename"]
    full_name = os.path.join(objectlist_path, objectlist_file).replace('/allen','//allen') #works with windows and linux filepaths
    objectlist_dataframe = pd.read_csv(full_name)
    objectlist_dataframe = clean_objectlist_col_labels(objectlist_dataframe) #"clean" columns names to be more meaningful
    return objectlist_dataframe


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

def roi_locations_from_cell_rois_table(experiment_id):
    """takes the lims_cell_rois_table and pares it down to just what's relevent to join with the roi metrics table. 
       Renames columns to maintain continuity between the tables. 
    
    Arguments:
        lims_cell_rois_table {dataframe} -- dataframe from LIMS with roi location information
    
    Returns:
        dataframe -- pared down dataframe
    """
    #get the cell_rois_table for that experiment
    lims_data = get_lims_data(experiment_id)
    exp_cell_rois_table = get_lims_cell_rois_table(lims_data['lims_id'].values[0])
    
    #select only the relevent columns
    roi_locations = exp_cell_rois_table[["id", "x", "y", "width", "height", "valid_roi", "mask_matrix"]]
    
    #rename columns to be able to merge them with the roi metrics
    roi_locations = roi_locations.rename(columns={"valid_roi":"valid", "mask_matrix":"mask"})
    return roi_locations



def get_roi_metrics(experiment_id):
    """[summary]
    
    Arguments:
        lims_data {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # objectlist.txt contains metrics associated with segmentation masks
    segmentation_dir = get_segmentation_dir(experiment_id)
    roi_metrics = pd.read_csv(os.path.join(segmentation_dir, 'objectlist.txt'))
    
    # get roi_locations and add unfiltered cell index
    roi_locations = roi_locations_from_cell_rois_table(lims_data)
    roi_names = np.sort(roi_locations.id.values)
    roi_locations['unfiltered_cell_index'] = [np.where(roi_names == id)[0][0] for id in roi_locations.id.values]
    # add cell ids to roi_metrics from roi_locations
    roi_metrics = add_roi_ids_to_roi_metrics(roi_metrics, roi_locations)
    # merge roi_metrics and roi_locations
    roi_metrics['id'] = roi_metrics.roi_id.values
    roi_metrics = pd.merge(roi_metrics, roi_locations, on='id')
    unfiltered_roi_metrics = roi_metrics
    # remove invalid roi_metrics
    roi_metrics = roi_metrics[roi_metrics.valid == True]
    # hack to get rid of cases with 2 rois at the same location
    for roi_id in roi_metrics.roi_id.values:
        roi_data = roi_metrics[roi_metrics.roi_id == roi_id]
        if len(roi_data) > 1:
            ind = roi_data.index
            roi_metrics = roi_metrics.drop(index=ind.values)
    # add filtered cell index
    cell_index = [np.where(np.sort(roi_metrics.roi_id.values) == id)[0][0] for id in
                  roi_metrics.roi_id.values]
    roi_metrics['cell_index'] = cell_index
    return roi_metrics, unfiltered_roi_metrics




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



def clean_objectlist_col_labels(objectlist_dataframe):
    """take the roi metrics from the objectlist.txt file and renames them to be more explicit and descriptive.  
        -removes single blank space at the beginning of column names
        -enforced naming scheme(no capitolization, added _)
        -renamed columns to be more descriptive/reflect contents of column
    
    Arguments:
        objectlist_dataframe {pandas dataframe} -- [roi metrics dataframe or dataframe generated from the objectlist.txt file]
    
    Returns:
        [pandas dataframe] -- [same dataframe with same information but with more informative column names
    """
    
    objectlist_dataframe = objectlist_dataframe.rename(index = str, columns = {' traceindex' :"trace_index", 
                                                        ' cx':'center_x',
                                                        ' cy':'center_y',
                                                        ' mask2Frame':'frame_of_max_intensity_masks_file',
                                                        ' frame':'frame_of_enhanced_movie',
                                                        ' object':'layer_of_max_intensity_file',
                                                        ' minx':'bbox_min_x',
                                                        ' miny':'bbox_min_y',
                                                        ' maxx':'bbox_max_x',
                                                        ' maxy':'bbox_max_y',
                                                        ' area':'area',
                                                        ' shape0':'ellipseness',
                                                        ' shape1':"compactness",
                                                        ' eXcluded':"exclude_code",
                                                        ' meanInt0':"mean_intensity",
                                                        ' meanInt1':"mean_enhanced_intensity",
                                                        ' maxInt0':"max_intensity",
                                                        ' maxInt1':"max_enhanced_intensity",
                                                        ' maxMeanRatio':"intensity_ratio",
                                                        ' snpoffsetmean':"soma_minus_np_mean",
                                                        ' snpoffsetstdv':"soma_minus_np_std",
                                                        ' act2':"sig_active_frames_2_5",
                                                        ' act3':"sig_active_frames_4",
                                                        ' OvlpCount':"overlap_count",
                                                        ' OvlpAreaPer':"percent_area_overlap",
                                                        ' OvlpObj0':"overlap_obj0_index",
                                                        ' OvlpObj1':"overlap_obj1_index",
                                                        ' corcoef0':"soma_obj0_overlap_trace_corr",
                                                        ' corcoef1':"soma_obj1_overlap_trace_corr"})
    return objectlist_dataframe


########  From SDK 

def get_session_from_sdk(experiment_id):
    """Use LIMS API to return session object
    
    Arguments:
        experiment_id {int} -- 9 digit experiment ID (same as LIMS ID)
    
    Returns:
        session object -- session object from SDK
    """
    api = BehaviorOphysLimsApi(experiment_id)
    session = BehaviorOphysSession(api)
    return session

def get_sdk_cell_specimen_table(experiment_id):
    """returns cell specimen table using the SDK LIMS API
    
    Arguments:
        experiment_id {int} -- 9 digit experiment id
    
    Returns:
        cell specimen table {pandas dataframe} -- dataframe with the following columns:
                    cell_specimen_id(index):
                    cell_roi_id:
                    height:
                    image_mask:
                    mask_image_plane:
                    max_correction_down:
                    max_correction_left:
                    max_correction_right:
                    max_correction_up:
                    valid_roi:
                    width:
                    bbox_min_x:
                    bbox_min_y:

    """
    session = get_session_from_sdk(experiment_id)
    cell_specimen_table = session.cell_specimen_table
    return cell_specimen_table


def get_binary_mask_for_cell_specimen_id(experiment_id, cell_specimen_id):
    """[summary]
    
    Arguments:
        sdk_cell_specimen_table {[type]} -- [description]
        cell_specimen_id {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    cell_specimen_table = get_sdk_cell_specimen_table(experiment_id)

    ##retrieve the mask & the x & y shift
        #cell_specimen_id is the index col so thats why use .index
        #get the mask
    roi_mask = cell_specimen_table.loc[cell_specimen_table.index==cell_specimen_id,"image_mask"].values[0]
    
        #get the x & y shift, x & y cols are the x & y min for roi bounding box (upper left corner)
    x_shift = int(cell_specimen_table.loc[cell_specimen_table.index==cell_specimen_id,"x"]) 
    y_shift = int(cell_specimen_table.loc[cell_specimen_table.index==cell_specimen_id,"y"])
    
    ##shift the roi mask to the correct position by using np roll
    shifted_roi=np.roll(roi_mask,(x_shift,y_shift),axis=(1,0))
    
    ##change the background from true/false to nan & 1, so area not masked will be transparent for plotting over ave proj
        #make a new mask (0,1) by copying the shifted mask
    new_mask = shifted_roi.copy()
    new_mask = new_mask.astype(int)
        #create an all 0s(binary) mask of the same shape
    binary_mask = np.zeros(new_mask.shape)
        #turn all 0s to nans
    binary_mask[:] = np.nan 
        #where new mask is one, change binary mask to 1, so now all nans and 1s where roi is
    binary_mask[new_mask==1] = 1
    return binary_mask



def gen_multi_roi_mask(experiment_id, roi_list):
    """[summary]
    
    Arguments:
        experiment_id {[type]} -- [description]
        roi_list {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    cell_specimen_table = get_sdk_cell_specimen_table(experiment_id)
    starting_mask = cell_specimen_table.loc[cell_specimen_table.index==roi_list[0],"image_mask"].values[0]
    #make a general mask of the correcte shape to be used for multiple rois
    multi_roi_mask = np.zeros(starting_mask.shape)
    multi_roi_mask[:] = np.nan
    
    for roi in roi_list:
        #create the binary mask for that roi
        roi_mask = get_binary_mask_for_cell_specimen_id(sdk_cell_specimen_table, roi)
        #make a 3d array with the roi mask and the starting mask
        masks_3d = np.array([multi_roi_mask,roi_mask])
        combo_mask = np.nansum(masks_3d, axis = 0)
        multi_roi_mask = combo_mask
    
    binary_mask = np.zeros(multi_roi_mask.shape)
    binary_mask[:]=np.nan
    binary_mask[multi_roi_mask==1]=1
    return binary_mask


def gen_roi_exclusion_category_masks(experiment_id):
    roi_metrics = rois.loc[rois["experiment_id"]==str(experiment_id), ["bbox_min_x", "bbox_min_y", "exclusion_labels"]]
    cell_specimen_table = cell_specimen_table.rename(columns = {"y":"bbox_min_y", "x":"bbox_min_x"})
    #merging will reset the index & you'll lose the cell_spec_id so need to make it a col that's not the index
    cell_specimen_table["cell_specimen_id"]=cell_specimen_table.index 
    merged_cell_specimen_table = pd.merge(cell_specimen_table, roi_metrics, how ="left", on=["bbox_min_x", "bbox_min_y"])
    exp_roi_failtags = gen_failtag_roi_df(merged_cell_specimen_table)  