
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

def get_unique_exclusion_labels():
     mixin = PostgresQueryMixin()
     query ='''
     Select 
     DISTINCT name
     From
     roi_exclusion_labels'''.format(experiment_id)
     return mixin.select(query)


def exclusion_label_roi_dict(experiment_id):
    exclusion_labels_df = roi.get_failed_roi_exclusion_labels(experiment_id)




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



def gen_roi_exclusion_category_masks(experiment_id):
    roi_metrics = rois.loc[rois["experiment_id"]==str(experiment_id), ["bbox_min_x", "bbox_min_y", "exclusion_labels"]]
    cell_specimen_table = cell_specimen_table.rename(columns = {"y":"bbox_min_y", "x":"bbox_min_x"})
    #merging will reset the index & you'll lose the cell_spec_id so need to make it a col that's not the index
    cell_specimen_table["cell_specimen_id"]=cell_specimen_table.index 
    merged_cell_specimen_table = pd.merge(cell_specimen_table, roi_metrics, how ="left", on=["bbox_min_x", "bbox_min_y"])
    exp_roi_failtags = gen_failtag_roi_df(merged_cell_specimen_table) 



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