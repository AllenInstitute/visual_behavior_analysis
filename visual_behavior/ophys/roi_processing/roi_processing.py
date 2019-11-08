from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.internal.api import PostgresQueryMixin
from psycopg2 import connect, extras 


import visual_behavior.ophys.roi_processing.cell_matching_report as matching
import visual_behavior.ophys.roi_processing.segmentation_report as seg
import visual_behavior.ophys.roi_processing.data_loading as load

from pathlib import Path
import pandas as pd
import numpy as np

import os


######################## EXPERIMENT LEVEL INFORMATION ##############
def experiment_info_df(experiment_id):
    """"manifest" styple information about a specific experiment
     
     Arguments:
         experiment_id {[type]} -- [description]
    
    Returns:
        dataframe -- dataframe with the following columns: 
                                                "experiment_id", 
                                                "container_id", 
                                                "workflow_state",
                                                "stage_name_lims", 
                                                "full_genotype", 
                                                "targeted_structure", 
                                                "depth", 
                                                "mouse_id", 
                                                "mouse_donor_id",
                                                "date_of_acquisition",
                                                "rig",
                                                "stage_name_mtrain"
    """
    lims_experiment_info = load.get_lims_experiment_info(experiment_id)
    lims_experiment_info = get_6digit_mouse_id(lims_experiment_info)
    lims_experiment_info = full_geno(lims_experiment_info)
    lims_experiment_info = load.get_stage_name_from_mtrain_sqldb(lims_experiment_info)
    lims_experiment_info = lims_experiment_info.drop(["mouse_info", "foraging_id"], axis =1)

    lims_experiment_info = lims_experiment_info[["experiment_id", 
                                                "container_id", 
                                                "workflow_state",
                                                "stage_name_lims", 
                                                "full_genotype", 
                                                "targeted_structure", 
                                                "depth", 
                                                "mouse_id", 
                                                "mouse_donor_id",
                                                "date_of_acquisition",
                                                "rig",
                                                "stage_name_mtrain"]]
    return lims_experiment_info  



def exp_roi_metrics_dataframe(experiment_id, mask_shift = False, include_failed_rois = True):
    """creates a dataframe with metrics about the rois/cells associated with a particular experiment id
    
    Arguments:
        experiment_id {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    objectlist_df = load.load_current_objectlisttxt_file(experiment_id)  
    roi_locations_df = load.roi_locations_from_cell_rois_table(experiment_id)
    cell_specimen_table = load.get_sdk_cell_specimen_table(experiment_id)
    failed_roi_exclusion_labels = load.gen_roi_exclusion_labels_lists(experiment_id)
    

    obj_loc_df = pd.merge(objectlist_df, roi_locations_df, how = "outer", on= ["bbox_min_x", "bbox_min_y"])
    roi_metrics_df = pd.merge(obj_loc_df, cell_specimen_table, how = "outer", on= ["cell_roi_id", "bbox_min_x", "bbox_min_y", "valid_roi"])
    roi_metrics_df = pd.merge(roi_metrics_df, failed_roi_exclusion_labels, how = "outer", on="cell_roi_id")
    roi_metrics_df = roi_metrics_df.dropna(subset=["cell_specimen_id"])
    roi_metrics_df.cell_specimen_id= roi_metrics_df.cell_specimen_id.astype('int32', copy=False)
    roi_metrics_df.loc[:,"experiment_id"]= experiment_id
    

    #delete duplicated cell specimen IDs
    duplicates = roi_metrics_df[["cell_specimen_id"]].copy()
    duplicates.loc[:,"duplicated"]=duplicates.duplicated()
    dup_indexes = duplicates.loc[duplicates["duplicated"]==True].index.values.tolist()
    roi_metrics_df.drop(roi_metrics_df.index[dup_indexes], inplace=True)

    if mask_shift== True:
        roi_metrics_df["shifted_image_mask"]= roi_metrics_df.apply(dataframe_shift_fov_mask, axis=1)

    if include_failed_rois ==False:
        roi_metrics_df = roi_metrics_df.loc[croi_metrics_df["valid_roi"]==True]

    #reset index since potentially filtered out some cells with the dropping of duplicates or failed rois
    roi_metrics_df = roi_metrics_df.reset_index(drop=True)
    
    return roi_metrics_df


def roi_metrics_for_experiment_list(experiment_list, include_failed_rois = True):
    """get roi metrics for a list of experiments
    
    Arguments:
        experiment_list {[type]} -- [description]
    
    Keyword Arguments:
        include_failed_rois {bool} -- [description] (default: {True})
    
    Returns:
        [type] -- [description]
    """
    roi_metrics = pd.DataFrame()
    experiments_info_df = pd.DataFrame()
    
    for experiment_id in experiment_list:
        experiment_roi_metrics_df = exp_roi_metrics_dataframe(experiment_id, include_failed_rois = include_failed_rois)
        
        exp_info_df = experiment_info_df(experiment_id)
        
        roi_metrics =roi_metrics.append(experiment_roi_metrics_df, sort=True)
        experiments_info_df = experiments_info_df.append(exp_info_df, sort=True)
        
    roi_metrics = roi_metrics.reset_index(drop = True)
    roi_metrics= matching.get_number_exps_rois_matched(roi_metrics)

    experiments_info_df = experiments_info_df.reset_index(drop=True)

    master_df = pd.merge(roi_metrics, experiments_info_df, how="inner", on="experiment_id")

    return master_df




######################## CONTAINER LEVEL INFORMATION ##############


def gen_container_manifest(container_id, 
                        include_ffield_test= False, 
                        include_failed_sessions = False):

    lims_container_info_df = load.get_lims_container_info(container_id)
    lims_container_info_df = get_6digit_mouse_id(lims_container_info_df)
    lims_container_info_df = full_geno(lims_container_info_df)
    lims_container_info_df = load.get_stage_name_from_mtrain_sqldb(lims_container_info_df)
    lims_container_info_df = calc_retake_number(lims_container_info_df)
    lims_container_info_df = lims_container_info_df.drop(["mouse_info", "foraging_id"], axis = 1)
    lims_container_info_df = lims_container_info_df.rename(columns={"ophys_experiment_id":"experiment_id"})
    
    # ## remove full field & receptive field test imaging sessions from the dataframe and don't generate metrics for them
    if include_ffield_test== False:
        lims_container_info_df = lims_container_info_df.dropna(subset=["stage_name_mtrain"])
        lims_container_info_df = lims_container_info_df.loc[lims_container_info_df["stage_name_lims"]!= "OPHYS_7_receptive_field_mapping"]
        lims_container_info_df = lims_container_info_df.loc[lims_container_info_df["stage_name_mtrain"]!= "OPHYS_7_receptive_field_mapping"]
    
    if include_failed_sessions ==False:
        lims_container_info_df= lims_container_info_df.loc[lims_container_info_df["workflow_state"]=="passed"]
    
    lims_container_info_df = lims_container_info_df.reset_index(drop=True)
    
    return lims_container_info_df


def get_container_roi_metrics(container_id, 
                            include_ffield_test= False, 
                            include_failed_sessions = False, 
                            include_failed_rois = True):
    """[summary]
    
    Arguments:
        container_id {[type]} -- [description]
    
    Keyword Arguments:
        include_ffield_test {bool} -- [include metrics for the "fulll field test imaging session] (default: {False})
        include_failed_sessions {bool} -- [include metrics for imaging sessions that failed in manifest] (default: {False})
        include_failed_rois {bool} -- [include rois that are invalid] (default: {True})
    
    Returns:
        [type] -- [description]
    """

    container_manifest = gen_container_manifest(container_id, 
                                                include_ffield_test= include_ffield_test, 
                                                include_failed_sessions = include_failed_sessions)
    
    experiments_list = container_manifest["experiment_id"].values.tolist()

    
    container_roi_metrics_df = pd.DataFrame()

    for experiment_id in experiments_list:
        experiment_roi_metrics_df = exp_roi_metrics_dataframe(experiment_id)
        experiment_roi_metrics_df.loc[:,"stage_name"] = container_manifest.loc[container_manifest["experiment_id"]==experiment_id, "stage_name_mtrain"].values[0]
        experiment_roi_metrics_df.loc[:,"full_genotype"] = container_manifest.loc[container_manifest["experiment_id"]==experiment_id, "full_genotype"].values[0]
        container_roi_metrics_df =container_roi_metrics_df.append(experiment_roi_metrics_df, sort=True)
        container_roi_metrics_df.loc[:,"container_id"]= container_id
        container_roi_metrics_df = container_roi_metrics_df.reset_index(drop=True)
        
    ###If cell_specimen_ids duplicated for a single exp id drop them here
    duplicates = container_roi_metrics_df[["cell_specimen_id", "experiment_id"]].copy()
    duplicates.loc[:,"duplicated"]=duplicates.duplicated()
    dup_indexes = duplicates.loc[duplicates["duplicated"]==True].index.values.tolist()
    container_roi_metrics_df.drop(container_roi_metrics_df.index[dup_indexes], inplace=True)

    if include_failed_rois == False:
        container_roi_metrics_df = container_roi_metrics_df.loc[container_roi_metrics_df["valid_roi"]==True]

    container_roi_metrics_df= matching.get_number_exps_rois_matched(container_roi_metrics_df)
    container_roi_metrics_df = container_roi_metrics_df.reset_index(drop = True)
    
    return container_roi_metrics_df

 
def for_manifest_get_container_roi_metrics(manifest, container_id):
    container_df = manifest.loc[manifest["container_id"]==container_id]
    container_df = container_df.reset_index(drop=True)
    
    #must be list not array for lims query to work
    experiments_list = container_df["experiment_id"].values
    experiments_list = experiments_list.tolist()

    container_roi_metrics_df = exp_roi_metrics_dataframe(experiments_list[0])
    container_roi_metrics_df["stage_name"]= container_df["stage_name"][0]
    container_roi_metrics_df["full_genotype"] = container_df["full_genotype"][0]
    container_roi_metrics_df["valid_cell_matching"]= container_df["valid_cell_matching"][0]

    
    
    for experiment_id in experiments_list[1:]:
        experiment_roi_metrics_df = exp_roi_metrics_dataframe(experiment_id)
        experiment_roi_metrics_df["stage_name"] = container_df.loc[container_df["experiment_id"]==experiment_id, "stage_name"].values[0]
        experiment_roi_metrics_df["full_genotype"] = container_df.loc[container_df["experiment_id"]==experiment_id, "full_genotype"].values[0]
        experiment_roi_metrics_df["valid_cell_matching"] = container_df.loc[container_df["experiment_id"]==experiment_id, "valid_cell_matching"].values[0]
        container_roi_metrics_df =container_roi_metrics_df.append(experiment_roi_metrics_df, sort=True)
        container_roi_metrics_df = container_roi_metrics_df.reset_index(drop=True)
    return container_roi_metrics_df





######################## ROI LEVEL INFORMATION ##############

def get_transparent_roi_FOV_mask(experiment_id, roi_id, id_type = False):
    if id_type ==False: 
        roi_id_type = determine_roi_id_type(experiment_id, roi_id)
    else: 
        roi_id_type = id_type
    
    roi_metrics = exp_roi_metrics_dataframe(experiment_id)
    
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

def dataframe_shift_fov_mask(row):
    """acts on a datframe row- for use in specifically in roi_metrics_dataframe
        applies the np.roll to move an image mask, on to every row in a dataframe, row by row
    
    Arguments:
        row {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return np.roll(row["image_mask"], (row["bbox_min_x"], row["bbox_min_y"]),axis=(1,0))


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

def dataframe_binarize_mask(row):
    """acts on a datframe row- for use in specifically in roi_metrics_dataframe.
        applies the change-mask_from_bool_to_binary on a dataframe row

    
    Arguments:
        row {[type]} -- dataframe row
    
    Returns:
        r -- [description]
    """
    return change_mask_from_bool_to_binary(row["shifted_image_mask"])

def gen_blank_mask_of_FOV_dimensions(experiment_id):
    """generates a transparent mask the same dimensions as the microscope FOV after motion correction 
        & cell segmentation
    
    Arguments:
        experiment_id {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    roi_metrics = exp_roi_metrics_dataframe(experiment_id)
    #make a general mask of the correcte shape to be used for multiple rois
    first_mask_index = roi_metrics["image_mask"].first_valid_index()
    FOV_dimension = roi_metrics["image_mask"][first_mask_index].shape
    blank_mask = np.zeros(FOV_dimension)
    blank_mask[:] = np.nan
    return blank_mask


def gen_multi_roi_mask(experiment_id, roi_id_list, mask_type = "binary"):
    roi_metrics = exp_roi_metrics_dataframe(experiment_id, mask_shift= True)
    id_type = roi_id_type_from_df(roi_metrics, roi_id_list[0])
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

def multi_roi_mask_from_df(roi_metrics_df, roi_id_list, mask_type="binary"):
    id_type = roi_id_type_from_df(roi_metrics_df, roi_id_list[0])
    roi_list_df = roi_metrics_df[roi_metrics_df[id_type].isin(roi_id_list)]
    roi_masks = roi_list_df["shifted_image_mask"].values
    bool_multi_roi_mask = np.sum(roi_masks,0)
    binary_multi_roi_mask = change_mask_from_bool_to_binary(bool_multi_roi_mask)
    if mask_type =="binary":
        return binary_multi_roi_mask
    elif mask_type =="bool":
        return bool_multi_roi_mask
    else:
        print("please specify 'bool' or 'binary' for mask_type")






######################## UTILITIES FUNCTIONS ##############
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

def stage_num(row):
    return row["stage_name"][6]

def get_stage_num(dataframe):
    dataframe.loc[:,"stage_num"] = dataframe.apply(stage_num, axis=1)
    return dataframe

def roi_id_type_from_df(roi_metrics_df, roi_id, print_stmnts = False):
    is_cell_specimen_id = roi_id in pd.Series(roi_metrics_df["cell_specimen_id"]).values
    is_cell_roi_id = roi_id in pd.Series(roi_metrics_df["cell_roi_id"]).values
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

def determine_roi_id_type(experiment_id, roi_id, print_stmnts = False):
    """if you provide either an roi_id or a cell_specimen_id it will determine the id type
    
    Arguments:
        experiment_id {[type]} -- [description]
        roi_id {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    roi_metrics = exp_roi_metrics_dataframe(experiment_id)
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
