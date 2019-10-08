from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

import visual_behavior.ophys.roi_processing.segmentation_report as seg
import visual_behavior.ophys.roi_processing.roi_processing as roi


import matplotlib.pyplot as plt


import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import json
import os

##################################  USING A MANIFEST 


def get_number_exps_rois_matched(dataframe):
    dataframe["number_exps_roi_matched"] = dataframe.groupby("cell_specimen_id")["cell_specimen_id"].transform('count')
    return dataframe

def remove_unmatched_rois(dataframe):
    matched_dataframe = dataframe.loc[dataframe["number_exps_roi_matched"]!=1]
    return matched_dataframe

def container_matched_roi_metrics(container_id, remove_unmatched = True, include_ffield_test =False, include_failed = False):
    """Gets matched ROI metrics listed as "passed" in workflow state. 
    
    Arguments:
        container_id {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    ##whether to include the full field test imaging sessions
    if include_ffield_test==True and include_failed == False:
        container_manifest = roi.gen_container_manifest(container_id, include_ffield_test=True)
        container_roi_metrics = roi.get_container_roi_metrics(container_id, include_ffield_test= True)
    
    elif include_ffield_test==False and include_failed== True:
        container_manifest== roi.gen_container_manifest(container_id, include_failed=True)
        container_roi_metrics = roi.get_container_roi_metrics(container_id, include_failed = True)
    
    else: 
        container_manifest = roi.gen_container_manifest(container_id)
        container_roi_metrics = roi.get_container_roi_metrics(container_id)
    
    container_roi_metrics = get_number_exps_rois_matched(container_roi_metrics)
    container_roi_metrics["num_exps_matched"] = container_roi_metrics['cell_specimen_id'].value_counts()
    
    
    if remove_unmatched == True:
        container_roi_metrics = remove_unmatched_rois(container_roi_metrics)
    
    container_roi_metrics["cell_specimen_id"].astype(int)
    container_roi_metrics = container_roi_metrics.sort_values(["number_exps_roi_matched", "cell_specimen_id"])
    container_roi_metrics = container_roi_metrics.reset_index(drop=True)
    container_roi_metrics["container_id"]= container_id

    #issue with 


    return container_roi_metrics

def container_matched_roi_morphology_metrics(container_id):
    container_matched_metrics = container_matched_roi_metrics(container_id)
    morphology_metrics_columns = ["cell_specimen_id","number_exps_roi_matched", "experiment_id", 
                                'area', 'ellipseness', 'compactness',
                                'mean_intensity', 'max_intensity', 'mean_enhanced_intensity', 'intensity_ratio',
                                'soma_minus_np_mean','soma_minus_np_std',
                                'valid_roi','exclusion_label_name', 'stage_name']
                                
    container_matched_morph_metrics = container_matched_metrics[morphology_metrics_columns]
    container_matched_morph_metrics["container_id"]= container_id
    container_matched_morph_metrics = container_matched_morph_metrics.reset_index(drop=True)
    return container_matched_morph_metrics

def get_experiment_pairs(group):
    pairs = itertools.combinations(group['experiment_id'], 2)
    return pairs


def get_matched_cells_experiment_pairs_df(matched_roi_morphology_metrics_df):
    morph_gb = matched_roi_morphology_metrics_df.groupby(by="cell_specimen_id")
    columns = ["cell_specimen_id", "exp1", "exp2"]
    cell_match_exp_pairs = pd.DataFrame(columns = columns)
    
    for cell_specimen_id in matched_roi_morphology_metrics_df["cell_specimen_id"].unique():
        pairs = list(pd.DataFrame(morph_gb.apply(get_experiment_pairs)).loc[cell_specimen_id].values[0])
        pairs = np.asarray(pairs)
        pairs_df = pd.DataFrame({"exp1":pairs[:,0], "exp2":pairs[:,1]})
        pairs_df["cell_specimen_id"] = cell_specimen_id
        pairs_df = pairs_df[["cell_specimen_id", "exp1", "exp2"]]
        cell_match_exp_pairs = cell_match_exp_pairs.append(pairs_df)
    cell_match_exp_pairs= cell_match_exp_pairs.reset_index(drop=True)
    return cell_match_exp_pairs
    
def get_cell_matched_exp_pairs_morph_metrics(matched_roi_morphology_metrics_df, cell_match_exp_pairs_df):
    """[summary]
    
    Arguments:
        matched_roi_morphology_metrics_df {[type]} -- [description]
        cell_match_exp_pairs_df {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    exp_pairs = pd.merge(cell_match_exp_pairs_df, matched_roi_morphology_metrics_df, left_on=["cell_specimen_id", "exp1"], right_on=["cell_specimen_id", "experiment_id"], how = "inner")
    exp_pairs = exp_pairs.drop(["number_exps_roi_matched", "experiment_id"], axis=1)
    exp_pairs = exp_pairs.rename(columns={"area":'area_exp1', 
                                       'ellipseness':'ellipseness_exp1',
                                       'compactness':'compactness_exp1', 
                                       'mean_intensity':'mean_intensity_exp1', 
                                       'max_intensity':'max_intensity_exp1',
                                       'mean_enhanced_intensity':'mean_enhanced_intensity_exp1', 
                                       'valid_roi':'valid_roi_exp1', 
                                       'exclusion_label_name':'exclusion_label_name_exp1',
                                       'stage_name':'stage_name_exp1'})

    exp_pairs = pd.merge(exp_pairs, matched_roi_morphology_metrics_df,  left_on=["cell_specimen_id", "exp2"], right_on=["cell_specimen_id", "experiment_id"], how = "inner")
    exp_pairs = exp_pairs.drop(["number_exps_roi_matched", "experiment_id"], axis=1)
    exp_pairs = exp_pairs.rename(columns={"area":'area_exp2', 
                                       'ellipseness':'ellipseness_exp2',
                                       'compactness':'compactness_exp2', 
                                       'mean_intensity':'mean_intensity_exp2', 
                                       'max_intensity':'max_intensity_exp2',
                                       'mean_enhanced_intensity':'mean_enhanced_intensity_exp2', 
                                       'valid_roi':'valid_roi_exp2', 
                                       'exclusion_label_name':'exclusion_label_name_exp2',
                                       'stage_name':'stage_name_exp2'})
    exp_pairs["both_rois_valid"]= exp_pairs[["valid_roi_exp1", "valid_roi_exp2"]].all(axis="columns")
    return exp_pairs

 
def gen_experiment_pair_morph_metrics(container_id):
    matched_roi_morphology_metrics_df = container_matched_roi_morphology_metrics(container_id)
    experiment_pairs = get_matched_cells_experiment_pairs_df(matched_roi_morphology_metrics_df)
    exp_pair_morph_metrics = get_cell_matched_exp_pairs_morph_metrics(matched_roi_morphology_metrics_df, experiment_pairs)
    morph_metrics_pairs = get_cell_matched_exp_pairs_morph_metrics(matched_roi_morphology_metrics_df, experiment_pairs)
    exp_pair_morph_metrics.loc[:,"container_id"]= container_id
    return exp_pair_morph_metrics

        

def experiment_match_summary_df(container_id, masks = False):
    num_exps_matched_list = [1,2,3,4,5,6]
    matched_morph_metrics= container_matched_roi_metrics(container_id, remove_unmatched = False)
    
    ####IF YOU DONT WANT TO GENERATE THE MASKS
    if masks== False:
        exp_match_summary_df = pd.DataFrame(columns=["experiment_id","num_exps_matched", "total_seg_cells", "total_valid_seg_cells",
                                                "matched_count","valid_count", "invalid_count", 
                                                "all_cell_specimen_ids", "valid_cell_specimen_ids", "invalid_cell_specimen_ids"])
       
        #must be list not array for lims query to work
        experiments_list= matched_morph_metrics["experiment_id"].unique()
        experiments_list = experiments_list.tolist()
        

        for experiment_id in experiments_list:
            # print(experiment_id)
            total_seg_cells = len(matched_morph_metrics.loc[matched_morph_metrics["experiment_id"]==experiment_id, "cell_specimen_id"].unique())
            total_valid_seg_cells = len(matched_morph_metrics.loc[(matched_morph_metrics["experiment_id"]==experiment_id) & 
                                                                (matched_morph_metrics["valid_roi"]==True), "cell_specimen_id"].unique())
            
            for num_exps in num_exps_matched_list:
                # print(num_exps)
               
                #Get cell specimen IDs
                matched_cell_spec_ids = matched_morph_metrics.loc[(matched_morph_metrics["number_exps_roi_matched"]==num_exps) &  
                                                                (matched_morph_metrics["experiment_id"]==experiment_id), "cell_specimen_id"].values
                
                valid_cell_spec_ids = matched_morph_metrics.loc[(matched_morph_metrics["number_exps_roi_matched"]==num_exps) &  
                                                                (matched_morph_metrics["experiment_id"]==experiment_id) &
                                                                (matched_morph_metrics["valid_roi"]==True), 
                                                                "cell_specimen_id"].values
                
                invalid_cell_spec_ids = matched_morph_metrics.loc[(matched_morph_metrics["number_exps_roi_matched"]==num_exps) &  
                                                                (matched_morph_metrics["experiment_id"]==experiment_id) &
                                                                (matched_morph_metrics["valid_roi"]==False), 
                                                                "cell_specimen_id"].values
                #get counts                                                 
                matched_count = len(matched_cell_spec_ids)
                valid_count = len(valid_cell_spec_ids)
                invalid_count = len(invalid_cell_spec_ids)

                df = pd.DataFrame({"experiment_id":experiment_id,  "total_seg_cells": total_seg_cells, "total_valid_seg_cells":total_valid_seg_cells,"num_exps_matched":num_exps, 
                                    "matched_count": matched_count, "valid_count":valid_count, "invalid_count":invalid_count, 
                                    "all_cell_specimen_ids":[matched_cell_spec_ids], "valid_cell_specimen_ids":[valid_cell_spec_ids], "invalid_cell_specimen_ids":[invalid_cell_spec_ids]})
                
                exp_match_summary_df = exp_match_summary_df.append(df) 
                
        exp_match_summary_df["container_id"]= container_id
        exp_match_summary_df = exp_match_summary_df.reset_index(drop=True)
        
        exp_match_summary_df= exp_match_summary_df[["container_id", "experiment_id", "total_seg_cells", "total_valid_seg_cells",  "num_exps_matched", "matched_count", "valid_count", "invalid_count", 
                                                "all_cell_specimen_ids", "valid_cell_specimen_ids", "invalid_cell_specimen_ids"]]
        
    
    
    ###IF YOU DO WANT TO GENERATE THE MASKS- THIS TAKES A TON OF TIME AND IS NOT EFFICIENT
    else:
        exp_match_summary_df = pd.DataFrame(columns=["experiment_id","num_exps_matched",
                                                "matched_count","valid_count", "invalid_count", 
                                                "all_cell_specimen_ids", "valid_cell_specimen_ids", "invalid_cell_specimen_ids", 
                                                "all_cells_mask", "valid_cells_mask", "invalid_cells_mask"])
    
        #must be list not array for lims query to work
        experiments_list= matched_morph_metrics["experiment_id"].unique()
        experiments_list = experiments_list.tolist()
        
        for experiment_id in experiments_list:
            # print(experiment_id)
            total_seg_cells = len(matched_morph_metrics.loc[matched_morph_metrics["experiment_id"]==experiment_id, "cell_specimen_id"].unique())
            total_valid_seg_cells = len(matched_morph_metrics.loc[(matched_morph_metrics["experiment_id"]==experiment_id) & 
                                                                (matched_morph_metrics["valid_roi"]==True), "cell_specimen_id"].unique())
            
            
            blank_mask = roi.gen_blank_mask_of_FOV_dimensions(experiment_id)
            roi_metrics = roi.gen_roi_metrics_dataframe(experiment_id, shift= True)
            
            for num_exps in num_exps_matched_list:
                
                # print(num_exps)
               
                #Get cell specimen IDs
                matched_cell_spec_ids = matched_morph_metrics.loc[(matched_morph_metrics["number_exps_roi_matched"]==num_exps) &  
                                                                (matched_morph_metrics["experiment_id"]==experiment_id), "cell_specimen_id"].values
                
                valid_cell_spec_ids = matched_morph_metrics.loc[(matched_morph_metrics["number_exps_roi_matched"]==num_exps) &  
                                                                (matched_morph_metrics["experiment_id"]==experiment_id) &
                                                                (matched_morph_metrics["valid_roi"]==True), 
                                                                "cell_specimen_id"].values
                
                invalid_cell_spec_ids = matched_morph_metrics.loc[(matched_morph_metrics["number_exps_roi_matched"]==num_exps) &  
                                                                (matched_morph_metrics["experiment_id"]==experiment_id) &
                                                                (matched_morph_metrics["valid_roi"]==False), 
                                                                "cell_specimen_id"].values
                #get counts                                                 
                matched_count = len(matched_cell_spec_ids)
                valid_count = len(valid_cell_spec_ids)
                invalid_count = len(invalid_cell_spec_ids)

                ####make masks
                mask_df = pd.DataFrame(columns=["num_exps_matched", "all_cells_mask", "valid_cells_mask", "invalid_cells_mask" ])
                
                if matched_count == 0:
                    all_cells_mask = blank_mask
                    valid_cells_mask = blank_mask
                    invalid_cells_mask = blank_mask
                
                elif invalid_count == 0 and valid_count !=0:
                    all_cells_mask = roi.multi_roi_mask_from_df(roi_metrics, matched_cell_spec_ids)
                    valid_cells_mask = roi.multi_roi_mask_from_df(roi_metrics, valid_cell_spec_ids)
                    invalid_cells_mask = blank_mask
                
                elif valid_count == 0 and invalid_count !=0:
                    all_cells_mask = roi.multi_roi_mask_from_df(roi_metrics, matched_cell_spec_ids)
                    valid_cells_mask = blank_mask
                    invalid_cells_mask = roi.multi_roi_mask_from_df(roi_metrics, invalid_cell_spec_ids)
                
                else:
                    all_cells_mask = roi.multi_roi_mask_from_df(roi_metrics, matched_cell_spec_ids)
                    valid_cells_mask = roi.multi_roi_mask_from_df(roi_metrics, valid_cell_spec_ids)
                    invalid_cells_mask = roi.multi_roi_mask_from_df(roi_metrics, invalid_cell_spec_ids)
        
                df = pd.DataFrame({"experiment_id":experiment_id,  "total_seg_cells": total_seg_cells, "total_valid_seg_cells":total_valid_seg_cells,"num_exps_matched":num_exps, 
                                    "matched_count": matched_count, "valid_count":valid_count, "invalid_count":invalid_count, 
                                    "all_cell_specimen_ids":[matched_cell_spec_ids], "valid_cell_specimen_ids":[valid_cell_spec_ids], "invalid_cell_specimen_ids":[invalid_cell_spec_ids],
                                    "all_cells_mask":[all_cells_mask], "valid_cells_mask":[valid_cells_mask], "invalid_cells_mask":[invalid_cells_mask]})
                
                exp_match_summary_df = exp_match_summary_df.append(df)
        
        exp_match_summary_df["container_id"]= container_id
        exp_match_summary_df = exp_match_summary_df.reset_index(drop=True)               
                
        exp_match_summary_df= exp_match_summary_df[["container_id","experiment_id", "total_seg_cells", "total_valid_seg_cells","num_exps_matched", "matched_count", "valid_count", "invalid_count", 
                                                    "all_cell_specimen_ids", "valid_cell_specimen_ids", "invalid_cell_specimen_ids", 
                                                    "all_cells_mask", "valid_cells_mask", "invalid_cells_mask"]]
    
    return exp_match_summary_df



# def gen_num_exp_masks_df(matched_count, valid_count, invalid_count):
#     num_exps_matched_list = [1,2,3,4,5,6]
#     df =experiment_match_summary_df(container_id)
    
#     experiments_list= df["experiment_id"].unique()
#     experiments_list = experiments_list.tolist()
    
#     for experiment_id in experiments_list:
#         blank_mask = roi.gen_blank_mask_of_FOV_dimensions(experiment_id)
#         roi_metrics = gen_roi_metrics_dataframe(experiment_id, shift= True)
        
#         for num_exps in num_exps_matched_list:
             
#              if matched_count == 0:
#                 all_cells_mask = blank_mask
#                 valid_cells_mask = blank_mask
#                 invalid_cells_mask = blank_mask
                
#             elif invalid_count == 0 and valid_count !=0:
#                 all_cells_mask = roi.multi_roi_mask_from_cell_list(matched_cell_spec_ids)
#                 valid_cells_mask = roi.multi_roi_mask_from_cell_list(valid_cell_spec_ids)
#                 invalid_cells_mask = blank_mask
                
#             elif valid_count == 0 and invalid_count !=0:
#                 all_cells_mask = roi.multi_roi_mask_from_cell_list(matched_cell_spec_ids)
#                 valid_cells_mask = blank_mask
#                 invalid_cells_mask = roi.multi_roi_mask_from_cell_list(invalid_cell_spec_ids)
                
#             else:
#                 all_cells_mask = roi.multi_roi_mask_from_cell_list(matched_cell_spec_ids)
#                 valid_cells_mask = roi.multi_roi_mask_from_cell_list(valid_cell_spec_ids)
#                 invalid_cells_mask = roi.multi_roi_mask_from_cell_list(invalid_cell_spec_ids)

#     roi_list_df = roi_metrics[roi_metrics[id_type].isin(roi_id_list)]
#     roi_masks = roi_list_df["shifted_image_mask"].values
#     binary_multi_roi_mask = roi.change_mask_from_bool_to_binary(bool_multi_roi_mask)
