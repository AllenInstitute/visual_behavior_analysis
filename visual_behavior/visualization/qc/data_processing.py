import visual_behavior.visualization.qc.data_loading as load

import pandas as pd
import numpy as np


####### EXPERIMENT LEVEL ####### # NOQA: E402
def ophys_experiment_info_df(ophys_experiment_id):
    """manifest style information about a specific
        ophys experiment

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


def gen_roi_validity_masks(ophys_experiment_id):

    cell_table = load.get_lims_cell_rois_table(ophys_experiment_id)
    cell_table = shift_image_masks(cell_table)


def ophys_experiment_segmentation_summary_df(ophys_experiment_id):
    """for a given experiment, uses the cell_rois_table from lims
        to get the total number segmented rois, as well as number
        valid and invalid rois

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                            "ophys_experiment_id",
                            "total_rois",
                            "valid_count",
                            "invalid_count",
                            "valid_percent",
                            "invalid_percent"
    """

    cell_table = load.get_lims_cell_rois_table(ophys_experiment_id)

    total_rois = len(cell_table)
    number_valid = len(cell_table.loc[cell_table["valid_roi"] == True])
    number_invalid = len(cell_table.loc[cell_table["valid_roi"] == False])

    seg_summary_df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                                   "total_rois": total_rois,
                                   "valid_count": number_valid,
                                   "invalid_count": number_invalid},
                                  index=[0])
    seg_summary_df["valid_percent"] = seg_summary_df["valid_count"] / seg_summary_df["total_rois"]
    seg_summary_df["invalid_percent"] = seg_summary_df["invalid_count"] / seg_summary_df["total_rois"]
    return seg_summary_df


####### CONTAINER LEVEL ####### # NOQA: E402


def ophys_container_info_df(ophys_container_id):
    container_info_df = load.get_lims_container_info(ophys_container_id)
    container_info_df = split_mouse_info_column(container_info_df)
    container_info_df = load.get_mtrain_stage_name(container_info_df)
    container_info_df = container_info_df.drop(["mouse_info", "foraging_id"], axis=1)


# def ophys_container_segmentation_summary_df(ophys_container_id):


def calc_retake_number(container_dataframe, stage_name_column="stage_name_mtrain"):
    stage_gb = container_dataframe.groupby(stage_name_column)
    unique_stages = container_dataframe[stage_name_column][~pd.isnull(container_dataframe[stage_name_column])].unique()
    for stage_name in unique_stages:
        # Iterate through the sessions sorted by date and save the index to the row
        sessions_this_stage = stage_gb.get_group(stage_name).sort_values('date_of_acquisition')
        for ind_enum, (ind_row, row) in enumerate(sessions_this_stage.iterrows()):
            container_dataframe.at[ind_row, 'retake_number'] = ind_enum
    return container_dataframe


####### UTILITIES ####### # NOQA: E402


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


def stage_num(row):
    return row["stage_name"][6]


def get_stage_num(dataframe):
    dataframe.loc[:, "stage_num"] = dataframe.apply(stage_num, axis=1)
    return dataframe


def shift_image_masks(dataframe):
    """takes a dataframe with cell specimen or roi information, and specifically
        the columns "image_mask", "x"(bbox_min_x), "y"(bbox_min_y) and shifts the
        image masks so they reflect where the ROI/Cell is within the imaging
        FOV

    Arguments:
        dataframe {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    dataframe["shifted_image_mask"] = dataframe.apply(shift_mask_by_row, axis=1)
    return dataframe


def shift_mask_by_row(row):
    """acts on a datframe row- for use in specifically in roi_metrics_dataframe
        applies the np.roll to move an image mask, on to every row in a dataframe,
        row by row

    Arguments:
        row {[type]} -- row of the dataframe

    Returns:
        [type] -- [description]
    """
    return np.roll(row["image_mask"], (row["x"], row["y"]), axis=(1, 0))


def remove_invalid_rois(dataframe):
    """takes a cell/roi level dataframe with column "valid_roi"
        and removes invalid rois

    Arguments:
        dataframe {[type]} -- [description]

    Returns:
        dataframe -- dataframe with invalid rois removed and index reset
    """
    dataframe = dataframe.loc[dataframe["valid_roi"] == True]
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def remove_unpassed_experiments(dataframe):
    """takes a container level dataframe with experiments as rows
        and removes all unpassed experiments.

    Arguments:
        dataframe {[type]} -- [description]

    Returns:
        dataframe -- dataframe with unpassed experiments removed and index reset
    """
    dataframe = dataframe.loc[dataframe["workflow_state"] == "passed"]
    dataframe = dataframe.reset_index(drop=True)
    return dataframe
