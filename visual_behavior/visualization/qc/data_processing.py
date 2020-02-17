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
    """ queries LIMS and Mtrain databases and compiles container information

    Arguments:
        ophys_container_id {int} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:

    """
    container_info_df = load.get_lims_container_info(ophys_container_id)
    container_info_df = split_mouse_info_column(container_info_df)
    container_info_df = load.get_mtrain_stage_name(container_info_df)
    container_info_df = container_info_df.drop(["mouse_info", "foraging_id"], axis=1)
    return container_info_df


def ophys_container_segmentation_summary_df(ophys_container_id,
                                            rmv_unpassed_experiments=True):
    """[summary]

    Arguments:
        ophys_container_id {int} -- [description]

    Keyword Arguments:
        rmv_unpassed_experiments {bool} -- [description] (default: {True})

    Returns:
        dataframe -- dataframe with the following columns:
                        "ophys_experiment_id",
                        "total_rois",
                        "valid_count",
                        "invalid_count",
                        "valid_percent",
                        "invalid_percent"

    """

    container_info_df = ophys_container_info_df(ophys_container_id)

    if rmv_unpassed_experiments == True:
        container_info_df = remove_unpassed_experiments(container_info_df)
    elif rmv_unpassed_experiments == False:
        pass

    container_seg_summary = pd.DataFrame()
    for exp_id in container_info_df["ophys_experiment_id"].unique():
        exp_seg_summary = ophys_experiment_segmentation_summary_df(exp_id)
        container_seg_summary = container_seg_summary.append(exp_seg_summary)

    container_seg_summary = container_seg_summary.reset_index(drop=True)
    return container_seg_summary


def melted_container_segmentation_summary_df(container_seg_summary_df):
    """takes the segmentation summary df and manipulates/reorders it so
        it can be used for bar plots with a column for hue


    Arguments:
        container_seg_summary_df {dataframe} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                        "ophys_experiment_id": 9 digit ophys experiment id

                        "total_rois": total number of segmented rois

                        "valid_invalid": "valid" or "invalid" to indicate
                                            the rois were valid or invalid

                        "roi_count": number of rois

                        "roi_percent": percentage of the total number of
                                        segmented rois
    """

    count_df = container_seg_summary_df[["ophys_experiment_id",
                                         "total_rois",
                                         "valid_count",
                                         "invalid_count"]].copy()

    perc_df = container_seg_summary_df[["ophys_experiment_id",
                                        "valid_percent",
                                        "invalid_percent"]].copy()

    melted_count = pd.melt(count_df, id_vars=["ophys_experiment_id", "total_rois"],
                                    var_name="valid_invalid",
                                    value_name="roi_count")

    melted_percent = pd.melt(perc_df, id_vars=["ophys_experiment_id"],
                                        var_name="valid_invalid",
                                        value_name="roi_count")

    melted_count['valid_invalid'] = melted_count['valid_invalid'].map({'valid_count': "valid",
                                                                       'invalid_count': "invalid"})

    melted_percent['valid_invalid'] = melted_percent['valid_invalid'].map({'valid_percent': "valid",
                                                                           'invalid_percent': "invalid"})

    merged_melted_df = pd.merge(melted_count, melted_percent, how="left", on=["ophys_experiment_id", "valid_invalid"])
    merged_melted_df = merged_melted_df.sort_values("ophys_experiment_id")
    return merged_melted_df


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


####### ROI MASKS ####### # NOQA: E402


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


def shift_mask_by_row(row, col="image_mask"):
    """acts on a datframe row and applies the np.roll to
        move an image mask, on to every row in a dataframe,
        row by row

    Arguments:
        row {[type]} -- [description]

    Keyword Arguments:
        col {str} -- the column to apply the roll to
                    if using LIMS cell table use "mask_matrix"
                    if using SDK session object cell_specimen_table use "image_mask"
                    (default: {"image_mask"})

    Returns:
        [type] -- [description]
    """
    if col == "mask_matrix":
        pass
    elif col == "image_mask":
        pass
    else:
        print("please enter a valid col, either 'mask_matrix' or 'image_mask'")
    return np.roll(row[col], (row["x"], row["y"]), axis=(1, 0))


def gen_multi_mask_bool(shifted_image_masks_array):
    """takes a 3d array with the shifted image masks (z,x,y) and sums them
        over z such that the 2d array it produces has all the rois

    Arguments:
        shifted_image_masks_array {array} -- [description]

    Returns:
        array -- [description]
    """

    multi_mask_bool = np.sum(shifted_image_masks_array, 0)
    return multi_mask_bool


def change_mask_from_bool_to_binary(mask):
    """change a mask from bool to binary so the backgroun is 0s and transparent,
        which allows it to be plotted over another image such as a max intensity
        projection or an average projection

    Arguments:
        mask {array} -- array of True & False

    Returns:
        array -- array/matrix of 1's & 0s
    """
    # change the background from true/false to nan & 1, so area not masked will be transparent for plotting over ave proj
    # make a new mask (0,1) by copying the shifted mask
    new_mask = mask.copy()
    new_mask = new_mask.astype(int)
    # create an all 0s(binary) mask of the same shape
    binary_mask = np.zeros(new_mask.shape)
    # turn all 0s to nans
    binary_mask[:] = np.nan
    # where new mask is one, change binary mask to 1, so now all nans and 1s where roi is
    binary_mask[new_mask == 1] = 1
    return binary_mask


def gen_transparent_multi_roi_mask(shifted_image_masks_array):
    """takes and array with shifted image masks the size of the imaging FOV
        and summs them to create a bool mask, then changes the bool mask
        to a binary mask so it's transparent everywhere there isn't an ROI

    Arguments:
        shifted_image_masks_array {array} -- 3d array of all the shifted image masks
                                            that are the same size as the FOV

    Returns:
        2d array -- 2d array the size of the imaging FOV for a single
                    ophys experiment FOV
    """
    multi_mask_bool = gen_multi_mask_bool(shifted_image_masks_array)
    multi_mask_binary = change_mask_from_bool_to_binary(multi_mask_bool)
    return multi_mask_binary


def gen_transparent_validity_masks(ophys_experiment_id):
    """uses the sdk cell_specimen_table and returns a dataframe with
        a single transparent mask for the all the valid cells
        and another single mask for all the the invalid cells

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        dataframe -- a dataframe with the following columns:
                    "ophys_experiment_id",
                    "valid_rois",
                    "transparent_mask"
    """
    sdk_cell_table = load.get_sdk_cell_specimen_table(ophys_experiment_id)
    sdk_cell_table = shift_image_masks(sdk_cell_table)

    validity_masks_df = pd.DataFrame()
    for TF in sdk_cell_table["valid_roi"].unique():
        shifted_image_masks_array = sdk_cell_table.loc[sdk_cell_table["valid_roi"] == TF, "shifted_image_mask"].values
        transparent_mask = gen_transparent_multi_roi_mask(shifted_image_masks_array)

        temp_df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                                "valid_rois": TF,
                                "transparent_mask": [transparent_mask]},
                               index=[0])
        validity_masks_df = validity_masks_df.append(temp_df)
    validity_masks_df = validity_masks_df.reset_index(drop=True)

    return validity_masks_df
