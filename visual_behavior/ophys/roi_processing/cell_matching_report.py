from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

import visual_behavior.ophys.roi_processing.segmentation_report as seg
import visual_behavior.ophys.roi_processing.roi_processing as roi


import matplotlib.pyplot as plt


import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

##################################  USING A MANIFEST 


def get_number_exps_rois_matched(dataframe):
    dataframe["number_exps_roi_matched"] = dataframe.groupby("cell_specimen_id")["cell_specimen_id"].transform('count')
    return dataframe

def remove_unmatched_rois(dataframe):
    matched_dataframe = dataframe.loc[dataframe["number_exps_roi_matched"]!=1]
    return matched_dataframe


def from_manifest_container_matched_roi_metrics(manifest, container_id):
    container_roi_metrics = roi.for_manifest_get_container_roi_metrics(manifest, container_id)
    container_roi_metrics = get_number_exps_rois_matched(container_roi_metrics)
    container_roi_metrics = remove_unmatched_rois(container_roi_metrics)
    container_roi_metrics.set_index("cell_specimen_id", inplace = True)
    container_roi_metrics.sort_index(inplace=True)
   
    return container_roi_metrics

def from_manifest_container_matched_roi_morphology_metrics(manifest, container_id):
    container_matched_metrics = from_manifest_container_matched_roi_metrics(manifest, container_id)
    morphology_metrics_columns = ["number_exps_roi_matched", "experiment_id", 'area', 'ellipseness', 'compactness',
       'mean_intensity', 'max_intensity', 'mean_enhanced_intensity','valid_roi','exclusion_label_name', 'stage_name', 'valid_cell_matching']
    
    matched_roi_morphology_metrics = container_matched_metrics[morphology_metrics_columns]
    return matched_roi_morphology_metrics

# def plot_cell_zoom(roi_mask, max_projection, cell_specimen_id, spacex=10, spacey=10, show_mask=False, ax=None):
#     m = roi_mask 
#     (y, x) = np.where(m == 1)
#     xmin = np.min(x)
#     xmax = np.max(x)
#     ymin = np.min(y)
#     ymax = np.max(y)
#     mask = np.empty(m.shape)
#     mask[:] = np.nan
#     mask[y, x] = 1
#     if ax is None:
#         fig, ax = plt.subplots()
#     ax.imshow(max_projection, cmap='gray', vmin=0, vmax=np.amax(max_projection))
#     if show_mask:
#         ax.imshow(mask, cmap='jet', alpha=0.3, vmin=0, vmax=1)
#     ax.set_xlim(xmin - spacex, xmax + spacex)
#     ax.set_ylim(ymin - spacey, ymax + spacey)
#     ax.set_title('cell ' + str(cell_specimen_id))
#     ax.grid(False)
#     ax.axis('off')
#     return ax