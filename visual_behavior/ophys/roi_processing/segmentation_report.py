
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
import visual_behavior.ophys.roi_processing.roi_processing as roi
from allensdk.internal.api import PostgresQueryMixin
from psycopg2 import connect, extras

import matplotlib.pyplot as plt


import scipy.stats as stats

import SimpleITK as sitk
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

exclusion_labels_list = ["apical_dendrite", "bad_shape", "boundary", "demix_error", "duplicate", "empty_neropil_mask",
"empty_roi_mask", "low_signal", "motion_border", "small_size", "union", "zero_pixels"]

roi_id_types_list = ["cell_specimen_id", "cell_roi_id"]

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


def gen_label_masks_df(experiment_id):
    seg_labels_df = gen_seg_labels_df(experiment_id)
    mask_df = pd.DataFrame(columns=["label", "multi_roi_mask"])
    blank_mask = roi.gen_blank_mask_of_FOV_dimensions(experiment_id)
    for label in seg_labels_df["label"]:
        lab = label
        print(lab)
        label_df = seg_labels_df.loc[seg_labels_df["label"]==lab]
        if label_df["number_rois"].values[0]== 0:
            label_mask = blank_mask
        else:
            label_mask = roi.gen_multi_roi_mask(experiment_id, seg_labels_df.loc[seg_labels_df["label"]==lab, "cell_roi_ids"].values[0])
        df = pd.DataFrame({"label":[lab],"multi_roi_mask":[label_mask]})
        mask_df = mask_df.append(df)
    return mask_df


def gen_segmentation_dataframe(experiment_id):
    seg_labels_df = gen_seg_labels_df(experiment_id)
    label_mask_df = gen_label_masks_df(experiment_id)
    segmentation_df = pd.merge(seg_labels_df, label_mask_df, how = "outer", on ="label")
    return segmentation_df


def gen_roi_exclusion_masks_dict(experiment_id):
    exclusion_rois_dict = gen_exclusion_label_cell_roi_id_dict(experiment_id)
    exclusion_masks_dict = {}
    for label in exclusion_labels_list:
        exclusion_masks_dict[label] = roi.gen_multi_roi_mask(experiment_id, exclusion_rois_dict[label])
    return exclusion_masks_dict


def gen_roi_exclusion_category_masks(experiment_id):
    roi_metrics = rois.loc[rois["experiment_id"]==str(experiment_id), ["bbox_min_x", "bbox_min_y", "exclusion_labels"]]
    cell_specimen_table = cell_specimen_table.rename(columns = {"y":"bbox_min_y", "x":"bbox_min_x"})
    #merging will reset the index & you'll lose the cell_spec_id so need to make it a col that's not the index
    cell_specimen_table["cell_specimen_id"]=cell_specimen_table.index 
    merged_cell_specimen_table = pd.merge(cell_specimen_table, roi_metrics, how ="left", on=["bbox_min_x", "bbox_min_y"])
    exp_roi_failtags = gen_failtag_roi_df(merged_cell_specimen_table) 

