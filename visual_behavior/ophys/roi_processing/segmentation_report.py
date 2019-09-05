
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


def gen_exclusion_label_roi_dict(experiment_id):
    roi_id_types = ["cell_specimen_id", "cell_roi_id"]
    exclusion_labels_df = roi.get_failed_roi_exclusion_labels(experiment_id)
    exclusion_labels_roi_dict = {}
    
    for label in exclusion_labels_list:
        id_type_dict = {}
        for id_type in roi_id_types:
            id_type_dict[id_type]= exclusion_labels_df.loc[exclusion_labels_df["exclusion_label_name"]==label, id_type].values
            exclusion_labels_roi_dict[label] = id_type_dict
    
    all_invalid_id_type_dict = {}
    for id_type in roi_id_types:
        all_invalid_id_type_dict[id_type]=exclusion_labels_roi_dict[id_type].unique()
    
    exclusion_labels_roi_dict["all_invalid"] = id_type_dict
    return exclusion_labels_roi_dict


def roi_exclusion_label_dict_to_df(exclusion_dict, roi_id_type):
    exclsuion_df = pd.DataFrame(list(exclusion_roi_dict.items()))
    exclsuion_df =exclsuion_df.rename (columns = {0: "label", 1:roi_id_type})

def gen_segmentation_report_dataframe (experiment_id)


def gen_roi_exclusion_masks_dict(experiment_id):
    exclusion_rois_dict = gen_exclusion_label_cell_roi_id_dict(experiment_id)
    exclusion_masks_dict = {}
    for label in exclusion_labels_list:
        exclusion_masks_dict[label] = roi.gen_multi_roi_mask(experiment_id, exclusion_rois_dict[label])
    return exclusion_masks_dict


# def get_masks_from_lims_cell_rois_table(experiment_id):
#     # make roi_dict with ids as keys and roi_mask_array
#     lims_data = roi.get_lims_data(experiment_id)
#     lims_cell_rois_table = roi.get_lims_cell_rois_table(lims_data['lims_id'].values[0])
    
#     h = lims_cell_rois_table["height"]
#     w = lims_cell_rois_table["width"]
#     cell_specimen_ids = lims_cell_rois_table["cell_specimen_id"]
    
#     roi_masks = {}
#     for i, id in enumerate(cell_specimen_ids):
#         m = roi_metrics[roi_metrics.id == id].iloc[0]
#         mask = np.asarray(m['mask'])
#         binary_mask = np.zeros((h, w), dtype=np.uint8)
#         binary_mask[int(m.y):int(m.y) + int(m.height), int(m.x):int(m.x) + int(m.width)] = mask
#         roi_masks[int(id)] = binary_mask
#     return roi_masks



def gen_roi_exclusion_category_masks(experiment_id):
    roi_metrics = rois.loc[rois["experiment_id"]==str(experiment_id), ["bbox_min_x", "bbox_min_y", "exclusion_labels"]]
    cell_specimen_table = cell_specimen_table.rename(columns = {"y":"bbox_min_y", "x":"bbox_min_x"})
    #merging will reset the index & you'll lose the cell_spec_id so need to make it a col that's not the index
    cell_specimen_table["cell_specimen_id"]=cell_specimen_table.index 
    merged_cell_specimen_table = pd.merge(cell_specimen_table, roi_metrics, how ="left", on=["bbox_min_x", "bbox_min_y"])
    exp_roi_failtags = gen_failtag_roi_df(merged_cell_specimen_table) 



