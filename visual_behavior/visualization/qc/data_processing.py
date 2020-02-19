import visual_behavior.visualization.qc.data_loading as load

import pandas as pd
import numpy as np
import itertools

# csid = cell_specimen_id


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
                                                "experiment_workflow_state",
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


# CONTAINER LEVEL #######     NOQA: E402


def ophys_container_info_df(ophys_container_id):
    """ queries LIMS and Mtrain databases and compiles container information

    Arguments:
        ophys_container_id {int} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "container_id":
                    "ophys_experiment_id":
                    "ophys_session_id":
                    "stage_name_lims":
                    "

    """
    container_info_df = load.get_lims_container_info(ophys_container_id)
    container_info_df = split_mouse_info_column(container_info_df)
    container_info_df = load.get_mtrain_stage_name(container_info_df)
    container_info_df = container_info_df.drop(["mouse_info", "foraging_id"], axis=1)
    return container_info_df


def calc_retake_number(container_dataframe, stage_name_column="stage_name_mtrain"):
    """calculates the retake number for each stage name

    Arguments:
        container_dataframe {dataframe} -- dataframe with info about the container
                                            that includes a column with stage name
                                            either from lims or mtrain

    Keyword Arguments:
        stage_name_column {str} -- the column that has the stage name
                                     (default: {"stage_name_mtrain"})

    Returns:
        dataframe -- the original dataframe plus the column "retake_number" with an
                        integer as the retake number for that stage name
    """
    stage_gb = container_dataframe.groupby(stage_name_column)
    unique_stages = container_dataframe[stage_name_column][~pd.isnull(container_dataframe[stage_name_column])].unique()
    for stage_name in unique_stages:
        # Iterate through the sessions sorted by date and save the index to the row
        sessions_this_stage = stage_gb.get_group(stage_name).sort_values('date_of_acquisition')
        for ind_enum, (ind_row, row) in enumerate(sessions_this_stage.iterrows()):
            container_dataframe.at[ind_row, 'retake_number'] = ind_enum
    return container_dataframe


def sort_dataframe_by_stage_name(dataframe, stage_name_column="stage_name_lims"):
    dataframe = dataframe.sort_values(stage_name_column)
    return dataframe


def stage_name_ordered_list(dataframe, stage_name_column="stage_name_lims"):
    sorted_df = sort_dataframe_by_stage_name(dataframe, stage_name_column=stage_name_column)
    stage_name_list = sorted_df[stage_name_column].unique()
    return stage_name_list


# SEGMENTATION ####   NOQA: E402


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


def melted_container_segmentation_summary_df(ophys_container_id,
                                             rmv_unpassed_experiments=True):
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
    container_seg_summary = ophys_container_segmentation_summary_df(ophys_container_id,
                                                                    rmv_unpassed_experiments=rmv_unpassed_experiments)

    count_df = container_seg_summary[["ophys_experiment_id",
                                      "total_rois",
                                      "valid_count",
                                      "invalid_count"]].copy()

    perc_df = container_seg_summary[["ophys_experiment_id",
                                     "valid_percent",
                                     "invalid_percent"]].copy()

    melted_count = pd.melt(count_df, id_vars=["ophys_experiment_id"],
                           var_name="roi_category",
                           value_name="roi_count")

    melted_percent = pd.melt(perc_df, id_vars=["ophys_experiment_id"],
                             var_name="valid_invalid",
                             value_name="roi_percent")

    melted_count['valid_invalid'] = melted_count['valid_invalid'].map({'valid_count': "valid_rois",
                                                                       'invalid_count': "invalid_rois",
                                                                       'total_rois': 'total_rois'})

    melted_percent['valid_invalid'] = melted_percent['valid_invalid'].map({'valid_percent': "valid",
                                                                           'invalid_percent': "invalid"})

    merged_melted_df = pd.merge(melted_count, melted_percent, how="left", on=["ophys_experiment_id", "valid_invalid"])
    merged_melted_df = merged_melted_df.sort_values("ophys_experiment_id")
    return merged_melted_df


def container_segmentation_barplots_df(ophys_container_id, stage_name_column="stage_name_lims"):
    """

    Arguments:
        ophys_container_id {[type]} -- [description]

    Keyword Arguments:
        stage_name_column {str} -- [description] (default: {"stage_name_lims"})

    Returns:
        dataframe -- dataframe with the following columns:
                    "container_id"
                    "ophys_experiment_id"
                    "stage_name_lims"
                    "total_rois"
                    valid_invalid"
                    "roi_count"
                    "roi_percent"
    """
    container_info_df = ophys_container_info_df(ophys_container_id)
    stage_name_df = container_info_df[["container_id", "ophys_experiment_id", stage_name_column]].copy()

    melted_cont_seg_sum = melted_container_segmentation_summary_df(ophys_container_id,
                                                                   rmv_unpassed_experiments=True)

    seg_sum_with_stage = pd.merge(stage_name_df, melted_cont_seg_sum, how="right", on="ophys_experiment_id")

    seg_sum_with_stage = sort_dataframe_by_stage_name(seg_sum_with_stage,
                                                      stage_name_column=stage_name_column)
    return seg_sum_with_stage


### CELL MATCHING ###

def get_lims_cell_roi_tables_for_container(ophys_container_id):
    """returns all the cell_specimen_ids and valid/invalid status for rois
        for all PASSED ophys_experiment_id's in a container

    Arguments:
        ophys_container_id {int} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                        "ophys_experiment_id"
                        "stage_name_lims"
                        "cell_specimen_id"
                        "valid_roi"
                        "container_id"
    """
    container_info_df = ophys_container_info_df(ophys_container_id)
    passed_container = remove_unpassed_experiments(container_info_df)

    stage_name_df = passed_container[["ophys_experiment_id", "stage_name_lims"]].copy()

    container_cell_table = pd.DataFrame()

    for ophys_experiment_id in passed_container["ophys_experiment_id"].unique():
        exp_cell_table = load.get_lims_cell_rois_table(ophys_experiment_id)

        exp_cell_table = exp_cell_table[["ophys_experiment_id",
                                         "cell_specimen_id",
                                         "valid_roi"]]
        container_cell_table = container_cell_table.append(exp_cell_table)
    container_cell_table.loc[:, "container_id"] = ophys_container_id
    container_cell_table = container_cell_table.reset_index(drop=True)
    container_cell_table = pd.merge(container_cell_table, stage_name_df, how="left", on="ophys_experiment_id")
    return container_cell_table


def get_valid_container_cell_roi_table(ophys_container_id):
    """uses the LIMS cell_roi_table and returns all the valid
        cell_specimen_ids and for all PASSED ophys_experiment_id's
        in a container

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                        "ophys_experiment_id"
                        "stage_name_lims"
                        "cell_specimen_id"
                        "valid_roi"
                        "container_id"
    """
    container_csid_table = get_lims_cell_roi_tables_for_container(ophys_container_id)
    valid_container_csid_df = remove_invalid_rois(container_csid_table)
    valid_container_csid_df = valid_container_csid_df.dropna(subset=['cell_specimen_id'])
    return valid_container_csid_df


def cell_specimen_id_matches_in_dataframe(dataframe):
    """takes a dataframe that contains the column "cell_specimen_id"
        and returns the number of times that cell_specimen_id appears
        in that dataframe

    Arguments:
        dataframe {[type]} -- [description]

    Returns:
        dataframe -- returns original  dataframe plus the column "match_count"
    """
    matches = dataframe["cell_specimen_id"].value_counts().to_frame().reset_index()
    matches = matches.rename(columns={"index": "cell_specimen_id", "cell_specimen_id": "match_count"})
    dataframe = pd.merge(dataframe, matches, how="left", on="cell_specimen_id")
    return dataframe

def container_valid_csid_count_by_number_exps_matched_to(ophys_container_id):
    """compiles all the valid cell_specimen_ids for a container and then computes
        how many unique cell_specimen_id s were matched across the 7 experiments
        in an experiment container

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "num_exp_matched"
                    "csid_count"
                    "csid_percent"
    """
    valid_container_csid_df = get_valid_container_cell_roi_table(ophys_container_id)
    valid_container_csid_df = cell_specimen_id_matches_in_dataframe(valid_container_csid_df)
    total_valid_csids_in_container = len(valid_container_csid_df["cell_specimen_id"].unique())
    num_exps_matched_to_df = get_csid_count_by_number_exps_matched_to(valid_container_csid_df)
    num_exps_matched_to_df["csid_percent"] = num_exps_matched_to_df["csid_count"] / total_valid_csids_in_container
    return num_exps_matched_to_df


def get_csid_count_by_number_exps_matched_to(dataframe):
    """takes in a dataframe with at least the columns: "cell_specimen_id" and
        "match_count" and returns the number of cell_specimen_id's that match
        1-7 experiments.

    Arguments:
        dataframe {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the folling columns:
                    "num_exp_matched": how many experiments were matched to (1-7 for how many experiments are in a container)
                    "csid_count": number of unique cell specimen ids
    """
    num_matched_df = pd.DataFrame()
    num_exps_matched_list = [1, 2, 3, 4, 5, 6, 7]
    for num_matched in num_exps_matched_list:
        #number of unique cell specimen ids that matched a given number of experiments
        csid_count = len(dataframe.loc[dataframe["match_count"]==num_matched, "cell_specimen_id"].unique())
        temp_df = pd.DataFrame({"num_exp_matched": num_matched, "csid_count": csid_count}, index=[0])
        num_matched_df = num_matched_df.append(temp_df)
    num_matched_df = num_matched_df.reset_index(drop=True)
    return num_matched_df



def container_experiment_pairs_valid_cell_matching(ophys_container_id):
    """takes a container and for the PASSED experiments, gets the VALID
        cell_specimen_ids. Then uses itertools to list all combinations
        of experiments in the container and for each experiment pair:
                -number of valid cell_specimen_ids
                -number of matched cell_specimen_ids
                -percent of match cell_specimen_ids

    Arguments:
        ophys_container_id {int} -- [description]

    Returns:
        pandas dataframe -- produces a dataframe with the following columns:
                                "exp1": experiment id for first experiment in experiment pair
                                "exp1_stage_num": the stage number of the stage name for experiment 1
                                                of the experiment pair
                                "exp2": experiment id for the second experiment in the experiment pair
                                "exp2_stage_name": the stage number of the stage name for experiment 2 of the experiment pair
                                "matched_count": the number of valid cells matched between the experiment pair
                                "perc_matched": the percentage of the valid cells that were matched (matched count /total valid count)
                                "total_valid_count": total number of valid cells for the experiment pair
    """
    # csid = cell_specimen_id
    valid_container_csid_df = get_valid_container_cell_roi_table(ophys_container_id)

    experiments_list = valid_container_csid_df["ophys_experiment_id"].unique().tolist()
    exp_combos = list(itertools.combinations(experiments_list, 2))

    container_df = pd.DataFrame()
    for combo in exp_combos:
        combo = np.asarray(combo)
        # subset valid_container_csid_df to get just just the data for the experiments in the experiment pair
        exp_pair_df = valid_container_csid_df.loc[valid_container_csid_df["ophys_experiment_id"].isin(combo)].copy()

        exp1 = combo[0]
        exp2 = combo[1]

        exp1_stage_name = valid_container_csid_df.loc[valid_container_csid_df["ophys_experiment_id"] == exp1, "stage_name_lims"].unique()[0]
        exp2_stage_name = valid_container_csid_df.loc[valid_container_csid_df["ophys_experiment_id"] == exp2, "stage_name_lims"].unique()[0]

        csids_in_exp_pair = len(exp_pair_df["cell_specimen_id"].unique())  # all the csids across both experiments
        exp_pair_df = cell_specimen_id_matches_in_dataframe(exp_pair_df)  # get number of times a csid appears & add to exp_pair_df
        matches_btw_pair = len(exp_pair_df.loc[exp_pair_df["match_count"] == 2, "cell_specimen_id"].unique())

        if csids_in_exp_pair == 0:
            perc_cells_matched = np.nan
        else:
            perc_cells_matched = np.round(matches_btw_pair / csids_in_exp_pair, 2)

        combo_df = pd.DataFrame({"exp1": exp1,
                                 "exp1_stage_name": exp1_stage_name,
                                 "exp2": exp2,
                                 "exp2_stage_name": exp2_stage_name,
                                 "total_valid_count": csids_in_exp_pair,
                                 "matched_count": matches_btw_pair,
                                 "perc_matched": perc_cells_matched}, index=[0])

        container_df = container_df.append(combo_df, sort=True)
    container_df.loc[:, "container_id"] = int(ophys_container_id)
    container_df = container_df.reset_index(drop=True)
    return container_df


def container_cell_matching_percent_heatmap_df(ophys_container_id):
    """takes the container_experiment_pairs_cell_matching dataframe and
        pivots it so it can be used to easily plot a heatmap of the % of valid
        cell_specimen_id s matched between experiment pairs


    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe where both rows are exp1_stage_name and columns are
                    exp2_stage_name and the values are the % of valid cell_specimen_id s
                    that were matched between the experiment pairs
    """

    container_exp_pair_matching_df = container_experiment_pairs_valid_cell_matching(ophys_container_id)
    valid_container_csid_df = get_valid_container_cell_roi_table(ophys_container_id)
    stage_order = stage_name_ordered_list(valid_container_csid_df, stage_name_column="stage_name_lims")
    pivot_perc = container_exp_pair_matching_df.pivot_table(index="exp1_stage_name", columns="exp2_stage_name", values="perc_matched")
    pivot_perc = pivot_perc.reindex(stage_order, axis=1)
    pivot_perc = pivot_perc.reindex(stage_order)
    return pivot_perc


def container_cell_matching_count_heatmap_df(ophys_container_id):
    """takes the container_experiment_pairs_cell_matching dataframe and
        pivots it so it can be used to easily plot a heatmap of the # of valid
        cell_specimen_id s matched between experiment pairs

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe where both rows are exp1_stage_name and columns are
                    exp2_stage_name and the values are the # of valid cell_specimen_id s
                    that were matched between the experiment pairs
    """
    container_exp_pair_matching_df = container_experiment_pairs_valid_cell_matching(ophys_container_id)
    valid_container_csid_df = get_valid_container_cell_roi_table(ophys_container_id)
    stage_order = stage_name_ordered_list(valid_container_csid_df, stage_name_column="stage_name_lims")
    pivot_count = container_exp_pair_matching_df.pivot_table(index="exp1_stage_name", columns="exp2_stage_name", values="matched_count")
    pivot_count = pivot_count.reindex(stage_order, axis=1)
    pivot_count = pivot_count.reindex(stage_order)
    return pivot_count

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
    dataframe = dataframe.loc[dataframe["experiment_workflow_state"] == "passed"]
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
