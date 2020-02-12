import visual_behavior.visualization.qc.data_loading as load

import pandas as pd
import numpy as np

def ophys_experiment_info_df(ophys_experiment_id):
    """manifest style information about a specific ophys experiment
    
    Arguments:
        ophys_experiment_id {[type]} -- [description]
    
    Returns:
        dataframe -- dataframe with the following columns:
                                                "ophys_experiment_id",
                                                "ophys_session_id"
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
    
    experiment_info_df = load.get_lims_experiment_info(ophys_experiment_id)
    experiment_info_df = split_mouse_info_column(experiment_info_df)
    experiment_info_df = load.get_mtrain_stage_name(experiment_info_df)
    experiment_info_df = experiment_info_df.drop(["mouse_info", "foraging_id"], axis=1)
    return experiment_info_df


def split_mouse_info_column(dataframe):
    """takes a lims info dataframe with the column "mouse_info" and splits it
        to separate the mouse_id and the full genotype
    
    Arguments:
        dataframe {[type]} -- dataframe (experiment, session or container level)
                                with the column "mouse_info" 
    
    Returns:
        dataframe -- returns same dataframe but with these columns added:
                        "mouse_id": 6 digit mouse id
                        "full_geno": full genotype of the mouse 
    """

    dataframe["mouse_id"] = int(dataframe["mouse_info"][0][-6:])
    dataframe["full_geno"] = dataframe["mouse_info"][0][:-7]
    return dataframe
   
