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


######################## RETRIEVING DATA ##############
 ########  From LIMS 



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
    """takes the lims_cell_rois_table and pares it down to just what's relevent to join with the objectlist.txt table. 
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
    
    #rename columns 
    roi_locations = clean_roi_locations_column_labels(roi_locations)
    return roi_locations


def clean_roi_locations_column_labels(roi_locations_dataframe):
    """takes some column labels from the roi_locations dataframe and  renames them to be more explicit and descriptive, and to match the column labels
        from the objectlist dataframe.
    
    Arguments:
        roi_locations_dataframe {dataframe} -- dataframe with roi id and location information
    
    Returns:
        dataframe -- [description]
    """
    roi_locations_dataframe = roi_locations_dataframe.rename(columns={"id":"cell_roi_id",
                                                            "mask_matrix":"roi_mask",
                                                            "x":"bbox_min_x",
                                                            "y":"bbox_min_y"})
    return roi_locations_dataframe




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


def gen_roi_exclusion_labels_lists(experiment_id):
    """[summary]
    
    Arguments:
        experiment_id {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    roi_exclusion_table = get_failed_roi_exclusion_labels(experiment_id)
    roi_exclusion_table = roi_exclusion_table[["cell_roi_id", "exclusion_label_name"]]
    exclusion_list_per_invalid_roi = roi_exclusion_table.groupby(["cell_roi_id"]).agg(lambda x: tuple(x)).applymap(list).reset_index()
    return exclusion_list_per_invalid_roi


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


def get_sdk_max_projection(experiment_id):
    session = roi.get_session_from_sdk(experiment_id)
    max_projection = session.max_projection
    return max_projection

def get_sdk_ave_projection(experiment_id):
    session = roi.get_session_from_sdk(experiment_id)
    ave_projection = session.average_projection
    return ave_projection



def get_sdk_cell_specimen_table(experiment_id):
    """returns cell specimen table using the SDK LIMS API
    
    Arguments:
        experiment_id {int} -- 9 digit experiment id
    
    Returns:
        cell specimen table {pandas dataframe} -- dataframe with the following columns:
                    cell_specimen_id(index):
                    cell_roi_id:
                    height:
                    image_mask: roi mask put within the motion corrected 2photon FOV (always the upper left corner of the image)
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
    cell_specimen_table = clean_cell_specimen_table_column_labels(cell_specimen_table) #rename columns to be consistent with roi_locations_dataframe and objectlist
    return cell_specimen_table




def clean_cell_specimen_table_column_labels(cell_specimen_table):
    """takes some column names from the cell specimen dataframe and renames them to be more explicit and descriptive, and to match the column labels
        from the objectlist dataframe and the roi locations dataframe.
    
    Arguments:
        cell_specimen_table {dataframe} -- dataframe with various metrics
    
    Returns:
        dataframe -- dataframe with some column names/labels changed
    """
    cell_specimen_table = cell_specimen_table.rename(columns={"x":"bbox_min_x",
                                                            "y":"bbox_min_y"})
    cell_specimen_table["cell_specimen_id"] =  cell_specimen_table.index #make cell_specimen_id a column so it can be merged on
    return cell_specimen_table

def dataframe_shift_fov_mask(row):
    """acts on a datframe row- for use in specifically in roi_metrics_dataframe
        applies the np.roll to move an image mask, on to every row in a dataframe, row by row
    
    Arguments:
        row {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return np.roll(row["image_mask"], (row["bbox_min_x"], row["bbox_min_y"]),axis=(1,0))

def dataframe_binarize_mask(row):
    """acts on a datframe row- for use in specifically in roi_metrics_dataframe.
        applies the change-mask_from_bool_to_binary on a dataframe row

    
    Arguments:
        row {[type]} -- dataframe row
    
    Returns:
        r -- [description]
    """
    return change_mask_from_bool_to_binary(row["shifted_image_mask"])

def gen_roi_metrics_dataframe(experiment_id, shift = False):
    """[summary]
    
    Arguments:
        experiment_id {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    objectlist_df = load_current_objectlisttxt_file(experiment_id)  
    roi_locations_df = roi_locations_from_cell_rois_table(experiment_id)
    cell_specimen_table = get_sdk_cell_specimen_table(experiment_id)
    failed_roi_exclusion_labels = gen_roi_exclusion_labels_lists(experiment_id)
    

    obj_loc_df = pd.merge(objectlist_df, roi_locations_df, how = "outer", on= ["bbox_min_x", "bbox_min_y"])
    roi_metrics_df = pd.merge(obj_loc_df, cell_specimen_table, how = "outer", on= ["cell_roi_id", "bbox_min_x", "bbox_min_y", "valid_roi"])
    roi_metrics_df = pd.merge(roi_metrics_df, failed_roi_exclusion_labels, how = "outer", on="cell_roi_id")
    roi_metrics_df["experiment_id"]= experiment_id

    if shift== True:
        roi_metrics_df["shifted_image_mask"]= roi_metrics_df.apply(dataframe_shift_fov_mask, axis=1)
        # roi_metrics_df["shifted_image_mask"]=roi_metrics_df.apply(dataframe_binarize_mask, axis=1)

    return roi_metrics_df

def determine_roi_id_type(experiment_id, roi_id, print_stmnts = False):
    """[summary]
    
    Arguments:
        experiment_id {[type]} -- [description]
        roi_id {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    roi_metrics = gen_roi_metrics_dataframe(experiment_id)
    is_cell_specimen_id = roi_id in pd.Series(roi_metrics["cell_specimen_id"]).values
    is_cell_roi_id = roi_id in pd.Series(roi_metrics["cell_roi_id"]).values
    if is_cell_specimen_id == True:
        if print_stmnts == True:
            print("cell_specimen_id")
        return "cell_specimen_id"
    elif is_cell_roi_id == True:
        if print_stmnts == True:
            print("cell_roi_id")
        return "cell_roi_id"
    else:
        print("roi id not listed in experiment dataframe")


##################Manipulating ROI Masks 

def get_transparent_roi_FOV_mask(experiment_id, roi_id, id_type = False):
    if id_type ==False: 
        roi_id_type = determine_roi_id_type(experiment_id, roi_id)
    else: 
        roi_id_type = id_type
    
    roi_metrics = gen_roi_metrics_dataframe(experiment_id)
    
    unshifted_roi_image_mask = roi_metrics.loc[roi_metrics[roi_id_type]==roi_id,"image_mask"].values[0]
    shifted_roi = shift_roi_FOV_mask_to_roi_FOV_location(roi_metrics, unshifted_roi_image_mask, roi_id, roi_id_type) #shift ROI to correction location in FOV
    binary_mask = change_mask_from_bool_to_binary(shifted_roi) #change mask from bool to binary
    return binary_mask


def shift_roi_FOV_mask_to_roi_FOV_location(roi_metrics, unshifted_roi_image_mask, roi_id, roi_id_type):
    """Takes the image mask and rolls it such that the ROI is in the correction location within the FOV
    
    Arguments:
        roi_mask {[type]} -- [description]
        cell_specimen_table {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    #get the x & y shift, x & y cols are the x & y min for roi bounding box (upper left corner)
    x_shift = int(roi_metrics.loc[roi_metrics[roi_id_type]==roi_id, "bbox_min_x"]) 
    y_shift = int(roi_metrics.loc[roi_metrics[roi_id_type]==roi_id, "bbox_min_y"])
    ##shift the roi mask to the correct position by using np roll
    shifted_roi_image_mask=np.roll(unshifted_roi_image_mask,(x_shift,y_shift),axis=(1,0))
    return shifted_roi_image_mask



def change_mask_from_bool_to_binary(mask):
    """change a mask from bool to binary so the backgroun is 0s and transparent, which allows it to be plotted over another image
        such as a max intensity projection or an average projection
    
    Arguments:
        mask {array} -- array of True & False
    
    Returns:
        array -- array/matrix of 1's & 0s
    """
    ##change the background from true/false to nan & 1, so area not masked will be transparent for plotting over ave proj
        #make a new mask (0,1) by copying the shifted mask
    new_mask = mask.copy()
    new_mask = new_mask.astype(int)
        #create an all 0s(binary) mask of the same shape
    binary_mask = np.zeros(new_mask.shape)
        #turn all 0s to nans
    binary_mask[:] = np.nan 
        #where new mask is one, change binary mask to 1, so now all nans and 1s where roi is
    binary_mask[new_mask==1] = 1
    return binary_mask

def gen_blank_mask_of_FOV_dimensions(experiment_id):
    """generates a transparent mask the same dimensions as the microscope FOV after motion correction 
        & cell segmentation
    
    Arguments:
        experiment_id {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    roi_metrics = gen_roi_metrics_dataframe(experiment_id)
    #make a general mask of the correcte shape to be used for multiple rois
    first_mask_index = roi_metrics["image_mask"].first_valid_index()
    FOV_dimension = roi_metrics["image_mask"][first_mask_index].shape
    blank_mask = np.zeros(FOV_dimension)
    blank_mask[:] = np.nan
    return blank_mask


def gen_multi_roi_mask(experiment_id, roi_id_list, mask_type = "binary"):
    roi_metrics = gen_roi_metrics_dataframe(experiment_id, shift= True)
    id_type = determine_roi_id_type(experiment_id, roi_id_list[0])
    roi_list_df = roi_metrics[roi_metrics[id_type].isin(roi_id_list)]
    roi_masks = roi_list_df["shifted_image_mask"].values
    bool_multi_roi_mask = np.sum(roi_masks,0)
    binary_multi_roi_mask = change_mask_from_bool_to_binary(bool_multi_roi_mask)
    if mask_type =="binary":
        return binary_multi_roi_mask
    elif mask_type =="bool":
        return bool_multi_roi_mask
    else:
        print("please specify 'bool' or 'binary' for mask_type")


##################################  CONTAINER LEVEL FUNCTIONS

def get_lims_container_info(container_id):
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

def get_stage_name_from_mtrain_sqldb(lims_container_info_df):
    foraging_ids = lims_container_info_df['foraging_id'][~pd.isnull(lims_container_info_df['foraging_id'])]
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
    lims_container_info_df = lims_container_info_df.merge(mtrain_response, on='foraging_id', how='left')
    lims_container_info_df = lims_container_info_df.rename(columns={"stage_name_x":"stage_name_lims", "stage_name_y":"stage_name_mtrain"})
    return lims_container_info_df

def get_6digit_mouse_id(lims_container_info_df):
    mouse_id = lims_container_info_df["mouse_info"][0][-6:]
    mouse_id = int(mouse_id)
    lims_container_info_df["mouse_id"] = mouse_id
    return lims_container_info_df


def full_geno(lims_container_info_df):
    full_geno = lims_container_info_df["mouse_info"][0][:-7]
    lims_container_info_df["full_genotype"]= full_geno
    return lims_container_info_df


def calc_retake_number(lims_container_info_df, stage_name_source= "mtrain"):
    
    #using the stage name from LIMS database or mtrain? LIMS DB needs updating to be accurate 9/24/19
    if stage_name_source== "mtrain":
        stage_name_column = "stage_name_mtrain"
    elif stage_name_source == "lims":
        stage_name_column = "stage_name_lims"
    else: 
        print ("please enter a valid stage name source, either 'mtrain' or 'lims'")

    stage_gb = lims_container_info_df.groupby(stage_name_column)
    unique_stages = lims_container_info_df[stage_name_column][~pd.isnull(lims_container_info_df[stage_name_column])].unique()
    for stage_name in unique_stages:
        # Iterate through the sessions sorted by date and save the index to the row
        sessions_this_stage = stage_gb.get_group(stage_name).sort_values('date_of_acquisition')
        for ind_enum, (ind_row, row) in enumerate(sessions_this_stage.iterrows()):
            lims_container_info_df.at[ind_row, 'retake_number'] = ind_enum
    return lims_container_info_df


def gen_container_manifest(container_id):
    lims_container_info_df = get_lims_container_info(container_id)
    lims_container_info_df = get_6digit_mouse_id(lims_container_info_df)
    lims_container_info_df = full_geno(lims_container_info_df)
    lims_container_info_df = get_stage_name_from_mtrain_sqldb(lims_container_info_df)
    lims_container_info_df = calc_retake_number(lims_container_info_df)
    lims_container_info_df = lims_container_info_df.drop(["mouse_info", "foraging_id"], axis = 1)
    return lims_container_info_df














# def dataframe_mouseinfo_split(row):
#     return row["mouse_info"].split('-')[:-1]

# def get_6digit_mouse_id(lims_container_info_df):
#     lims_container_info_df['mouse_id'] = lims_container_info_df.apply(dataframe_mouseinfo_split, axis =1)
#     lims_container_info_df["mouse_id"]= lims_container_info_df["mouse_id"].astype(int)
#     return lims_container_info_df





def get_container_roi_metrics(container_id):
    container_manifest= gen_container_manifest(container_id)
    passed_experiments = container_manifest.loc[container_manifest["workflow_state"]=="passed", "ophys_experiment_id"].values
    container_roi_metrics_df = gen_roi_metrics_dataframe(passed_experiments[0])
    container_roi_metrics_df["stage_name"]= container_manifest["stage_name_mtrain"][0]
    container_roi_metrics_df["full_genotype"] = container_manifest["full_genotype"][0]

    for experiment_id in passed_experiments[1:]:
        experiment_roi_metrics_df = gen_roi_metrics_dataframe(experiment_id)
        experiment_roi_metrics_df["stage_name"] = container_manifest.loc[container_manifest["ophys_experiment_id"]==experiment_id, "stage_name_mtrain"].values[0]
        experiment_roi_metrics_df["full_genotype"] = container_manifest.loc[container_manifest["ophys_experiment_id"]==experiment_id, "full_genotype"].values[0]
        container_roi_metrics_df =container_roi_metrics_df.append(experiment_roi_metrics_df)
        container_roi_metrics_df = container_roi_metrics_df.reset_index(drop=True)
    return container_roi_metrics_df

 
def for_manifest_get_container_roi_metrics(manifest, container_id):
    container_df = manifest.loc[manifest["container_id"]==container_id]
    container_df = container_df.reset_index(drop=True)

    container_roi_metrics_df = gen_roi_metrics_dataframe(container_df["ophys_experiment_id"].values[0])
    container_roi_metrics_df["stage_name"]= container_df["stage_name"][0]
    container_roi_metrics_df["full_genotype"] = container_df["full_genotype"][0]
    container_roi_metrics_df["valid_cell_matching"]= container_df["valid_cell_matching"][0]
    
    for experiment_id in container_df["ophys_experiment_id"].values[1:]:
        experiment_roi_metrics_df = gen_roi_metrics_dataframe(experiment_id)
        experiment_roi_metrics_df["stage_name"] = container_df.loc[container_df["ophys_experiment_id"]==experiment_id, "stage_name"].values[0]
        experiment_roi_metrics_df["full_genotype"] = container_df.loc[container_df["ophys_experiment_id"]==experiment_id, "full_genotype"].values[0]
        experiment_roi_metrics_df["valid_cell_matching"] = container_df.loc[container_df["ophys_experiment_id"]==experiment_id, "valid_cell_matching"].values[0]
        container_roi_metrics_df =container_roi_metrics_df.append(experiment_roi_metrics_df)
        container_roi_metrics_df = container_roi_metrics_df.reset_index(drop=True)
    return container_roi_metrics_df

