
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

import visual_behavior.ophys.roi_processing.cell_matching_report as matching
import visual_behavior.ophys.roi_processing.roi_processing as roi

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

exclusion_labels_list = ["apical_dendrite", "bad_shape", "boundary", "demix_error", "duplicate", "empty_neropil_mask",
"empty_roi_mask", "low_signal", "motion_border", "small_size", "union", "zero_pixels"]

roi_id_types_list = ["cell_specimen_id", "cell_roi_id"]

##################################  EXPERIMENT LEVEL FUNCTIONS
def gen_exclusion_label_roi_df(experiment_id):
    exclusion_labels_df = roi.get_failed_roi_exclusion_labels(experiment_id)
    excl_label_roi_df = pd.DataFrame(columns=["label","cell_specimen_ids", "cell_roi_ids"])

    for ex_label in exclusion_labels_list:
        label = ex_label
        cell_spec_ids = exclusion_labels_df.loc[exclusion_labels_df["exclusion_label_name"]==label, "cell_specimen_id"].values
        cell_r_ids = exclusion_labels_df.loc[exclusion_labels_df["exclusion_label_name"]==label, "cell_roi_id"].values
        df = pd.DataFrame({"label":label,"cell_specimen_ids":[cell_spec_ids], "cell_roi_ids":[cell_r_ids]})
        excl_label_roi_df = excl_label_roi_df.append(df)
    return excl_label_roi_df


def gen_roi_validity_df(experiment_id):
    cell_specimen_table = roi.get_sdk_cell_specimen_table(experiment_id)
    validity_df = pd.DataFrame(columns=["label","cell_specimen_ids", "cell_roi_ids"])
    boolean = [True, False]
    for b in boolean:
        cell_spec_ids = cell_specimen_table.loc[cell_specimen_table["valid_roi"]==b, "cell_specimen_id"].values
        cell_roi_ids = cell_specimen_table.loc[cell_specimen_table["valid_roi"]==b, "cell_specimen_id"].values
        df = pd.DataFrame({"label":b,"cell_specimen_ids":[cell_spec_ids], "cell_roi_ids":[cell_roi_ids]})
        validity_df = validity_df.append(df)
        validity_df["label"] = np.where(validity_df["label"], "all_valid", "all_invalid")
    return validity_df


def gen_seg_labels_df(experiment_id):
    exclusion_df = gen_exclusion_label_roi_df(experiment_id)
    roi_validity_df = gen_roi_validity_df(experiment_id)
    seg_labels_df = roi_validity_df.append(exclusion_df)
    seg_labels_df = seg_labels_df.reset_index(drop=True)
    seg_labels_df["number_rois"]=seg_labels_df['cell_specimen_ids'].str.len()
    return seg_labels_df


# def gen_label_masks_df(experiment_id):
#     seg_labels_df = gen_seg_labels_df(experiment_id)
#     mask_df = pd.DataFrame(columns=["label", "multi_roi_mask"])
#     blank_mask = roi.gen_blank_mask_of_FOV_dimensions(experiment_id)
#     for label in seg_labels_df["label"]:
#         lab = label
#         print(lab)
#         label_df = seg_labels_df.loc[seg_labels_df["label"]==lab]
#         if label_df["number_rois"].values[0]== 0:
#             label_mask = blank_mask
#         else:
#             label_mask = roi.gen_multi_roi_mask(experiment_id, seg_labels_df.loc[seg_labels_df["label"]==lab, "cell_roi_ids"].values[0])
#         df = pd.DataFrame({"label":[lab],"multi_roi_mask":[label_mask]})
#         mask_df = mask_df.append(df)
#     return mask_df


def gen_label_masks_df(experiment_id, print = False):
    seg_labels_df = gen_seg_labels_df(experiment_id)
    roi_metrics = roi.exp_roi_metrics_dataframe(experiment_id, shift= True)
    blank_mask = roi.gen_blank_mask_of_FOV_dimensions(experiment_id)

    mask_df = pd.DataFrame(columns=["label", "multi_roi_mask"])
    for label in seg_labels_df["label"]:
        lab = label
        if print == True:
            print(lab)
        label_df = seg_labels_df.loc[seg_labels_df["label"]==lab]
        if label_df["number_rois"].values[0]== 0:
            label_mask = blank_mask
        else:
            label_mask = roi.multi_roi_mask_from_df(roi_metrics, seg_labels_df.loc[seg_labels_df["label"]==lab, "cell_roi_ids"].values[0])
        df = pd.DataFrame({"label":[lab],"multi_roi_mask":[label_mask]})
        mask_df = mask_df.append(df)
    return mask_df


def gen_segmentation_dataframe(experiment_id):
    seg_labels_df = gen_seg_labels_df(experiment_id)
    label_mask_df = gen_label_masks_df(experiment_id)
    segmentation_df = pd.merge(seg_labels_df, label_mask_df, how = "outer", on ="label")
    return segmentation_df
    
def gen_exp_seg_validity_summary_df(experiment_id, masks = True):
    roi_metrics = roi.exp_roi_metrics_dataframe(experiment_id, shift= True)
    blank_mask = roi.gen_blank_mask_of_FOV_dimensions(experiment_id)
    
    valid_csi = roi_metrics.loc[roi_metrics["valid_roi"]==True, "cell_specimen_id"].values
    invalid_csi = roi_metrics.loc[roi_metrics["valid_roi"]==False, "cell_specimen_id"].values

    valid_count = len(valid_csi)
    invalid_count = len(invalid_csi)

    if masks == False:
        exp_seg_validity_summary = pd.DataFrame({"experiment_id":experiment_id, 
                                    "valid_count": valid_count, "invalid_count":invalid_count, 
                                    "valid_csis": [valid_csi], "invalid_csis": [invalid_csi]})

    else:  
        if valid_count ==0:
            valid_mask = blank_mask
        else: 
            valid_mask = roi.multi_roi_mask_from_df(roi_metrics, valid_csi)

        if invalid_count ==0:
            invalid_mask = blank_mask
        else: 
            invalid_mask = roi.multi_roi_mask_from_df(roi_metrics, invalid_csi)

        exp_seg_validity_summary = pd.DataFrame({"experiment_id":experiment_id, 
                                    "valid_count": valid_count, "invalid_count":invalid_count, 
                                    "valid_csis": [valid_csi], "invalid_csis": [invalid_csi],
                                    "valid_mask": [valid_mask], "invalid_mask": [invalid_mask]})
    
    return exp_seg_validity_summary

def gen_container_seg_validitity_summary_df(container_id, masks = True):
    container_manifest = roi.gen_container_manifest(container_id)
    
    experiment_id_list = container_manifest["experiment_id"].unique()
    experiment_id_list = experiment_id_list.tolist()

    container_seg_validity_df = pd.DataFrame()

    for experiment_id in experiment_id_list:
        exp_seg_validity_summary = gen_exp_seg_validity_summary_df(experiment_id, masks = masks)

        container_seg_validity_df = container_seg_validity_df.append(exp_seg_validity_summary)
    
    container_seg_validity_df.loc[:,"container_id"]= container_id
    container_seg_validity_df = container_seg_validity_df.reset_index(drop = True)

    return container_seg_validity_df







# def plot_exclusion_label_masks(experiment_id, segmentation_df):
#     ave_proj = roi.get_sdk_ave_projection(experiment_id)
#     max_proj = roi.get_sdk_max_projection(experiment_id)
#     for label in exclsuion_labels_list:
#         lab = label
#         label_df = segmentation_df.loc[segmentation_df["label"]==lab]



##################################  CONTAINER LEVEL FUNCTIONS














##################################  MANIFEST LEVEL FUNCTIONS 



