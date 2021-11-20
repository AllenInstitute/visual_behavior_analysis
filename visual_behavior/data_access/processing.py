from visual_behavior.data_access import loading as data_loading

import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
from functools import reduce

# csid = cell_specimen_id

# FUNCTIONS TO MANIPULATE DATA FOR ANALSYSIS #


# BASIC INFO/DATAFRAMES (EXP, SESSION & CONTAINER)

def ophys_experiment_info_df(ophys_experiment_id):
    """manifest style information about a specific
        ophys experiment

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                                                "ophys_experiment_id",
                                                "ophys_session_id"
                                                "ophys_container_id",
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

    experiment_info_df = data_loading.get_lims_experiment_info(ophys_experiment_id)
    experiment_info_df = split_mouse_info_column(experiment_info_df)
    experiment_info_df = data_loading.get_mtrain_stage_name(experiment_info_df)
    experiment_info_df = experiment_info_df.drop(["mouse_info", "foraging_id"], axis=1)
    return experiment_info_df


def ophys_container_info_df(ophys_container_id):
    """ queries LIMS and Mtrain databases and compiles container information

    Arguments:
        ophys_container_id {int} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "ophys_container_id":
                    "container_workflow_state"
                    "ophys_experiment_id":
                    "ophys_session_id":
                    "stage_name_lims":
                    "experiment_workflow_state"
                    "mouse_donor_id"
                    "targeted_structure"
                    "depth"
                    "rig"
                    "date_of_acquisition"
                    "retake_number"
                    "mouse_id"
                    "full_geno"
                    "stage_name_mtrain"

    """
    container_info_df = data_loading.get_lims_container_info(ophys_container_id)
    container_info_df = calc_retake_number(container_info_df, stage_name_column="stage_name_lims")
    container_info_df = split_mouse_info_column(container_info_df)
    container_info_df = data_loading.get_mtrain_stage_name(container_info_df)
    container_info_df = container_info_df.drop(["mouse_info", "foraging_id"], axis=1)
    return container_info_df


def passed_experiment_info_for_container(ophys_container_id):
    """returns "manifest style" container information but
    filters out all unpassed ophys_experiments

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "ophys_container_id":
                    "container_workflow_state"
                    "ophys_experiment_id":
                    "ophys_session_id":
                    "stage_name_lims":
                    "experiment_workflow_state"
                    "mouse_donor_id"
                    "targeted_structure"
                    "depth"
                    "rig"
                    "date_of_acquisition"
                    "retake_number"
                    "mouse_id"
                    "full_geno"
                    "stage_name_mtrain"
    """
    container_info_df = ophys_container_info_df(ophys_container_id)
    passed_exp_container = remove_unpassed_experiments(container_info_df)
    return passed_exp_container


def experiment_order_and_stage_for_container(ophys_container_id):
    """gets all passed experiments for a container and then sorts them by the
        acquisition date and returns a datafram ewith experiment id and stage name(from lims)
        in acquisition date order

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "ophys_experiment_id"
                    "stage_name_lims"
    """
    container_exp_order_and_stage = passed_experiment_info_for_container(ophys_container_id).sort_values('date_of_acquisition').reset_index(drop=True)[["ophys_experiment_id", "stage_name_lims"]].copy()
    return container_exp_order_and_stage


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


# SEGMENTATION (EXP & CONTAINER)

def segmentation_summary_for_experiment(ophys_experiment_id):
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

    cell_table = data_loading.get_lims_cell_rois_table(ophys_experiment_id)

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


def segmentation_summary_for_container(ophys_container_id):
    """[summary]

    Arguments:
        ophys_container_id {int} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                        "ophys_experiment_id",
                        "total_rois",
                        "valid_count",
                        "invalid_count",
                        "valid_percent",
                        "invalid_percent"
    """

    container_info_df = passed_experiment_info_for_container(ophys_container_id)
    container_seg_summary = pd.DataFrame()
    for ophys_experiment_id in container_info_df["ophys_experiment_id"].unique():
        exp_seg_summary = segmentation_summary_for_experiment(ophys_experiment_id)
        container_seg_summary = container_seg_summary.append(exp_seg_summary)

    container_seg_summary = container_seg_summary.reset_index(drop=True)
    return container_seg_summary


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


def segmentation_validity_count_for_container(ophys_container_id):
    container_seg_summary = segmentation_summary_for_container(ophys_container_id)
    count_df = container_seg_summary[["ophys_experiment_id",
                                      "total_rois",
                                      "valid_count",
                                      "invalid_count"]].copy()

    melted_count = pd.melt(count_df, id_vars=["ophys_experiment_id"],
                           var_name="roi_category",
                           value_name="roi_count")

    return melted_count


def segmentation_validity_percent_for_container(ophys_container_id):
    container_seg_summary = segmentation_summary_for_container(ophys_container_id)
    perc_df = container_seg_summary[["ophys_experiment_id",
                                     "valid_percent",
                                     "invalid_percent"]].copy()
    melted_percent = pd.melt(perc_df, id_vars=["ophys_experiment_id"],
                             var_name="valid_invalid",
                             value_name="roi_percent")
    return melted_percent


# CELL MATCHING (CONTAINER)

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
                        "ophys_container_id"
    """
    passed_container = passed_experiment_info_for_container(ophys_container_id)

    stage_name_df = passed_container[["ophys_experiment_id", "stage_name_lims"]].copy()

    container_cell_table = pd.DataFrame()

    for ophys_experiment_id in passed_container["ophys_experiment_id"].unique():
        exp_cell_table = data_loading.get_lims_cell_rois_table(ophys_experiment_id)

        exp_cell_table = exp_cell_table[["ophys_experiment_id",
                                         "cell_specimen_id",
                                         "valid_roi"]]
        container_cell_table = container_cell_table.append(exp_cell_table)
    container_cell_table.loc[:, "ophys_container_id"] = ophys_container_id
    container_cell_table = container_cell_table.drop_duplicates()
    container_cell_table = container_cell_table.reset_index(drop=True)
    container_cell_table = pd.merge(container_cell_table, stage_name_df, how="left", on="ophys_experiment_id")
    return container_cell_table


def get_valid_csids_from_lims_for_container(ophys_container_id):
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


def get_valid_csids_from_dataset_for_container(ophys_container_id):
    """gets valid cell specimen IDs from the SDK dataset object for passed experiments in a container

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                        "ophys_experiment_id"
                        "stage_name"
                        "cell_specimen_id"
                        "valid_roi"
                        "container_id"
    """

    experiments_table = data_loading.get_filtered_ophys_experiment_table()
    container_expts = experiments_table[experiments_table.container_id == ophys_container_id]
    valid_cells_df = pd.DataFrame()
    for ophys_experiment_id in container_expts.index.values:
        dataset = data_loading.get_ophys_dataset(ophys_experiment_id)
        ct = dataset.cell_specimen_table.copy()
        ct = ct.reset_index()
        ct = ct[['cell_specimen_id', 'valid_roi']]
        ct['ophys_experiment_id'] = ophys_experiment_id
        ct['container_id'] = ophys_container_id
        ct['stage_name'] = experiments_table.loc[ophys_experiment_id].session_type
        valid_cells_df = pd.concat([valid_cells_df, ct])
    return valid_cells_df


def get_valid_and_invalid_csids_from_lims_for_container(ophys_container_id):
    """uses the LIMS cell_roi_table and returns all
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
    container_csid_table = container_csid_table.dropna(subset=['cell_specimen_id'])
    return container_csid_table

# does this go in data.validation?


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
    valid_container_csid_df = get_valid_csids_from_lims_for_container(ophys_container_id)
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
        # number of unique cell specimen ids that matched a given number of experiments
        csid_count = len(dataframe.loc[dataframe["match_count"] == num_matched, "cell_specimen_id"].unique())
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
                                "exp1": experiment id for first experiment in
                                        experiment pair

                                "exp1_stage_num": the stage number of the stage name
                                                for experiment 1 of the experiment pair

                                "exp2": experiment id for the second experiment
                                        in the experiment pair

                                "exp2_stage_name": the stage number of the stage name
                                                    for experiment 2 of the experiment pair

                                "matched_count": the number of valid cells matched
                                                between the experiment pair

                                "perc_matched": the percentage of the valid cells that were
                                                matched (matched count /total valid count)

                                "total_valid_count": total number of valid cells for
                                                    the experiment pair
    """
    # csid = cell_specimen_id
    valid_container_csid_df = get_valid_csids_from_lims_for_container(ophys_container_id)
    # valid_container_csid_df = get_valid_csids_from_dataset_for_container(ophys_container_id)
    valid_container_csid_df = valid_container_csid_df.rename(columns={'stage_name_lims': 'stage_name'})

    experiments_list = np.sort(valid_container_csid_df["ophys_experiment_id"].unique().tolist())
    exp_combos = list(itertools.product(experiments_list, repeat=2))

    container_df = pd.DataFrame()
    for combo in exp_combos:
        combo = np.asarray(combo)
        # subset valid_container_csid_df to get just just the data for the experiments in the experiment pair
        exp_pair_df = valid_container_csid_df.loc[valid_container_csid_df["ophys_experiment_id"].isin(combo)].copy()

        exp1 = combo[0]
        exp2 = combo[1]

        unique_exps_in_combo = len(np.unique(combo))

        exp1_stage_name = valid_container_csid_df.loc[valid_container_csid_df["ophys_experiment_id"] == exp1,
                                                      "stage_name"].unique()[0]

        exp2_stage_name = valid_container_csid_df.loc[valid_container_csid_df["ophys_experiment_id"] == exp2,
                                                      "stage_name"].unique()[0]

        csids_in_exp_pair = len(exp_pair_df["cell_specimen_id"].unique())  # all the csids across both experiments
        exp_pair_df = cell_specimen_id_matches_in_dataframe(exp_pair_df)  # get number of times a csid appears & add to exp_pair_df

        matches_btw_pair = len(exp_pair_df.loc[exp_pair_df["match_count"] == unique_exps_in_combo,
                                               "cell_specimen_id"].unique())

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
    valid_container_csid_df = get_valid_csids_from_lims_for_container(ophys_container_id)
    # stage_order = stage_name_ordered_list(valid_container_csid_df, stage_name_column="stage_name_lims")
    # put in order that the data was collected in
    expt_order = np.sort(valid_container_csid_df.ophys_experiment_id.unique())
    stage_order = [valid_container_csid_df[valid_container_csid_df.ophys_experiment_id == expt].stage_name_lims.values[0] for expt
                   in expt_order]
    stage_order = ['None' if stage == np.nan else stage for stage in stage_order]
    # pivot_perc = container_exp_pair_matching_df.pivot_table(index="exp1_stage_name", columns="exp2_stage_name", values="perc_matched")
    # pivot_perc = pivot_perc.reindex(stage_order, axis=1)
    # pivot_perc = pivot_perc.reindex(stage_order)
    pivot_perc = container_exp_pair_matching_df.pivot_table(index="exp1", columns="exp2",
                                                            values="perc_matched")
    pivot_perc = pivot_perc.reindex(expt_order, axis=1)
    pivot_perc = pivot_perc.reindex(expt_order)
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
    valid_container_csid_df = get_valid_csids_from_lims_for_container(ophys_container_id)
    # stage_order = stage_name_ordered_list(valid_container_csid_df, stage_name_column="stage_name_lims")
    expt_order = np.sort(valid_container_csid_df.ophys_experiment_id.unique())
    # pivot_count = container_exp_pair_matching_df.pivot_table(index="exp1_stage_name", columns="exp2_stage_name", values="matched_count")
    # pivot_count = pivot_count.reindex(stage_order, axis=1)
    # pivot_count = pivot_count.reindex(stage_order)
    pivot_perc = container_exp_pair_matching_df.pivot_table(index="exp1", columns="exp2",
                                                            values="matched_count")
    pivot_perc = pivot_perc.reindex(expt_order, axis=1)
    pivot_perc = pivot_perc.reindex(expt_order)
    return pivot_perc


# ROI PROCESSING (EXP AND CONTAINER, SEG & CELL MATCHING, DFF)


def shift_image_masks(dataframe):
    """takes a dataframe with cell specimen or roi information, and specifically
        the columns "image_mask", "x"(bbox_min_x), "y"(bbox_min_y) and shifts the
        image masks so they reflect where the ROI/Cell is within the imaging
        FOV, adding the shifted FOV as a new column "roi_mask" to preserve the original masks

    Arguments:
        dataframe {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    dataframe["roi_mask"] = None
    dataframe.loc[:, "roi_mask"] = dataframe.apply(shift_mask_by_row, axis=1)
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
    sdk_cell_table = data_loading.get_sdk_cell_specimen_table(ophys_experiment_id)
    sdk_cell_table = shift_image_masks(sdk_cell_table)

    validity_masks_df = pd.DataFrame()
    for TF in sdk_cell_table["valid_roi"].unique():
        shifted_image_masks_array = sdk_cell_table.loc[sdk_cell_table["valid_roi"] == TF, "image_mask"].values
        transparent_mask = gen_transparent_multi_roi_mask(shifted_image_masks_array)

        temp_df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                                "valid_rois": TF,
                                "transparent_mask": [transparent_mask]},
                               index=[0])
        validity_masks_df = validity_masks_df.append(temp_df)
    validity_masks_df = validity_masks_df.reset_index(drop=True)
    return validity_masks_df


def valid_sdk_dff_traces_experiment(ophys_experiment_id):
    """gets the dff traces from the sdk  session object
        and then filters out the invalid rois  from the
        sdk cell_specimen_table and returns only valid
        cell_specimen_ids and dff traces

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "cell_specimen_id"
                    "dff"
                    "ophys_experiment_id
    """
    dff_traces = data_loading.get_sdk_dff_traces(ophys_experiment_id)
    cell_specimen_table = data_loading.get_sdk_cell_specimen_table(ophys_experiment_id)
    merged_dfs = dff_traces.merge(cell_specimen_table, left_index=True, right_index=True)
    merged_dfs["cell_specimen_id"] = merged_dfs.index
    merged_dfs = remove_invalid_rois(merged_dfs)
    merged_dfs = merged_dfs[["cell_specimen_id", "dff"]].copy()
    merged_dfs.loc[:, "ophys_experiment_id"] = ophys_experiment_id
    return merged_dfs


def valid_sdk_dff_traces_container(ophys_container_id):
    """uses the sdk to get all the valid cell specimen ids
        for each experiment in a container and their dff traces

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "cell_specimen_id"
                    "dff"
                    "ophys_experiment_id"
                    "ophys_container_id"
    """
    container_passed_exps = passed_experiment_info_for_container(ophys_container_id).sort_values('date_of_acquisition').reset_index(drop=True)
    container_valid_dff_traces = pd.DataFrame()
    for ophys_experiment_id in container_passed_exps["ophys_experiment_id"].unique():
        experiment_dff_traces = valid_sdk_dff_traces_experiment(ophys_experiment_id)
        container_valid_dff_traces = container_valid_dff_traces.append(experiment_dff_traces)
    container_valid_dff_traces.loc[:, "ophys_container_id"] = ophys_container_id
    container_valid_dff_traces = container_valid_dff_traces.reset_index(drop=True)
    return container_valid_dff_traces


def dff_robust_noise(dff_trace):
    """Robust estimate of std of noise in df/f

    Arguments:
        dff_trace {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    sigma_MAD_conversion_factor = 1.4826

    dff_trace = np.asarray(dff_trace)
    # first pass removing big pos peaks
    dff_trace = dff_trace[dff_trace < 1.5 * np.abs(dff_trace.min())]
    MAD = np.median(np.abs(dff_trace - np.median(dff_trace)))  # MAD = median absolute deviation
    robust_standard_deviation = sigma_MAD_conversion_factor * MAD

    # second pass removing remaining pos and neg peaks
    dff_trace = dff_trace[np.abs(dff_trace - np.median(dff_trace)) < 2.5 * robust_standard_deviation]
    MAD = np.median(np.abs(dff_trace - np.median(dff_trace)))
    robust_standard_deviation = sigma_MAD_conversion_factor * MAD
    return robust_standard_deviation


def dff_robust_signal(dff_trace, robust_standard_deviation):
    """ median deviation

    Arguments:
        dff_trace {[type]} -- [description]
        robust_standard_deviation {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    dff_trace = np.asarray(dff_trace)
    median_deviation = np.median(dff_trace[(dff_trace - np.median(dff_trace)) > robust_standard_deviation])
    return median_deviation


def compute_robust_snr_on_dataframe(dataframe):
    """takes a dataframe with a "dff" column that has the dff trace array
        for a cell_specimen_id and for noise uses Robust estimate of std for signal
        uses median deviation, and for robust snr the robust signal / robust noise

    Arguments:
        dataframe {[type]} -- [description]

    Returns:
        dataframe -- input dataframe but with the following columns added:
                        "robust_noise"
                        "robust_signal"
                        "robust_snr"
    """
    if 'dff' in dataframe.columns:
        column = 'dff'
    elif 'filtered_events' in dataframe.columns:
        column = 'filtered_events'
    dataframe["robust_noise"] = dataframe[column].apply(lambda x: dff_robust_noise(x))
    dataframe["robust_signal"] = dataframe.apply(lambda x: dff_robust_signal(x[column], x["robust_noise"]), axis=1 )
    dataframe["robust_snr"] = dataframe["robust_signal"] / dataframe["robust_noise"]
    return dataframe


def experiment_csid_snr_table(ophys_experiment_id):
    """gets the valid cell_specimen_id dff traces from the sdk dff_traces
        object and computes a robust estimate of noise(robust std), signal(median deviation)
        and snr(robust signal/robust noise)  for all cell_specimen_ids
    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "cell_specimen_id":
                    "ophys_experiment_id":
                    "dff":
                    "robust_noise":
                    "robust_signal":
                    "robust_snr":
                    "snr_zscore":
    """
    exp_dff_traces = valid_sdk_dff_traces_experiment(ophys_experiment_id)
    exp_dff_traces["ophys_experiment_id"] = ophys_experiment_id
    exp_dff_traces = compute_robust_snr_on_dataframe(exp_dff_traces)
    exp_dff_traces["snr_zscore"] = np.abs(stats.zscore(exp_dff_traces["robust_snr"]))
    return exp_dff_traces


def experiment_mean_robust_snr_for_all_csids(ophys_experiment_id, rmv_outliers=False):
    """takes the mean

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Keyword Arguments:
        rmv_outliers {bool} -- [description] (default: {True})

    Returns:
        float -- [description]
    """
    exp_csid_snr_table = experiment_csid_snr_table(ophys_experiment_id)
    if rmv_outliers == True:
        exp_csid_snr_table = remove_outliers(exp_csid_snr_table, "snr_zscore")
    mean_rsnr_csids = np.mean(exp_csid_snr_table["robust_snr"])
    return mean_rsnr_csids


def experiment_median_robust_snr_all_csids(ophys_experiment_id, rmv_outliers=False):
    """[summary]

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Keyword Arguments:
        rmv_outliers {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    exp_csid_snr_table = experiment_csid_snr_table(ophys_experiment_id)
    if rmv_outliers == True:
        exp_csid_snr_table = remove_outliers(exp_csid_snr_table, "snr_zscore")
    median_rsnr_csids = np.median(exp_csid_snr_table["robust_snr"])
    return median_rsnr_csids


def remove_outliers(dataframe, zscore_column):
    dataframe = dataframe.loc[dataframe[zscore_column] < 3].copy().reset_index(drop=True)
    return dataframe


def container_csid_snr_table(ophys_container_id):
    """gets all the passed ophys_experiment_ids for a container
        and then compiles all the experiment_csid_snr_table s in to
        one master dataframe

    Arguments:
        ophys_container_id {int} -- 9 digit ophys container id

    Returns:
        dataframe --  dataframe with the following columns:
                    "cell_specimen_id":
                    "ophys_experiment_id":
                    "dff":
                    "robust_noise":
                    "robust_signal":
                    "robust_snr":
                    "snr_zscore":
    """
    container_df = (passed_experiment_info_for_container(ophys_container_id)).sort_values('date_of_acquisition').reset_index(drop=True)
    container_csid_snr_df = pd.DataFrame()
    for ophys_experiment_id in container_df["ophys_experiment_id"].unique():
        experiment_csid_snr_df = experiment_csid_snr_table(ophys_experiment_id)
        container_csid_snr_df = container_csid_snr_df.append(experiment_csid_snr_df)
    container_csid_snr_df = container_csid_snr_df.reset_index(drop=True)
    container_csid_snr_df.loc[:, "ophys_container_id"] = ophys_container_id
    return container_csid_snr_df


def container_snr_summary_table(ophys_container_id):
    """[summary]

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "ophys_experiment_id"
                    "median_rsnr_all_csids"
                    "ophys_container_id"
    """
    container_passed_exps = passed_experiment_info_for_container(ophys_container_id).sort_values('date_of_acquisition').reset_index(drop=True)
    container_summary_df = pd.DataFrame()
    for ophys_experiment_id in container_passed_exps["ophys_experiment_id"].unique():
        exp_median_csid_snr = experiment_median_robust_snr_all_csids(ophys_experiment_id, rmv_outliers=False)
        exp_median_csid_snr_df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                                               "median_rsnr_all_csids": exp_median_csid_snr}, index=[0])
        container_summary_df = container_summary_df.append(exp_median_csid_snr_df)
    container_summary_df = container_summary_df.reset_index(drop=True)
    container_summary_df.loc[:, "ophys_container_id"] = ophys_container_id
    return container_summary_df


# PHYSIO FOV AND INTENSITY (EXP AND CONTAINER)


def get_experiment_average_intensity_timeseries(ophys_experiment_id):
    """uses the LIMS wkf system to get the filepath for the
        motion_corrected_movie.h5 file Then loads the file
        using h5py package.

        subsets the motion corrected movie by taking every
        500th frame, then taking just the inner portion of that frame
        so as not to have any border motion artifacts
        and then averaging that frame in x&y to get
        a single average intensity number for that frame

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        average_intensity -- array of average frame intensities
        frame_numbers -- array of frame numbers the frame intensitites
                            were calculated for
    """
    motion_corrected_movie_array = data_loading.load_motion_corrected_movie(ophys_experiment_id)

    subset = motion_corrected_movie_array[::500, 100:400, 50:400]

    # takes the mean across both x & y to get a single number for the frame
    average_intensity = np.mean(subset, axis=(1, 2))

    frame_numbers = np.arange(0, motion_corrected_movie_array.shape[0])
    frame_numbers = frame_numbers[::500]

    return average_intensity, frame_numbers


def experiment_average_FOV_from_motion_corrected_movie(ophys_experiment_id):
    """takes every 500th frame o fthe motion corrected movie and averages
        it together to create an average FOV for the experiment

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:s
        2d array   -- 2d array thats the average of the motion corrected FOV
    """
    motion_corrected_movie_array = data_loading.load_motion_corrected_movie(ophys_experiment_id)
    subset = motion_corrected_movie_array[::500, :, :]
    average_FOV = np.mean(subset, axis=0)
    return average_FOV


def experiment_max_FOV_from_motion_corrected_movie(ophys_experiment_id):
    """takes every 500th frame o fthe motion corrected movie and takes
        the max to create an maximum intensity FOV for the experiment

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        2d array   -- 2d array thats the average of the motion corrected FOV
    """
    motion_corrected_movie_array = data_loading.load_motion_corrected_movie(ophys_experiment_id)
    subset = motion_corrected_movie_array[::500, :, :]
    max_FOV = np.amax(subset, axis=0)
    return max_FOV


def experiment_intensity_mean_and_std(ophys_experiment_id):
    """Takes the average intensity timeseries from the motion corrected movie
        (already downsampled to be every 500th frame) and gets
        the mean and standard deviation of that time series
    Arguments:
        experiment_average_intensity_timeseries {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "ophys_experiment_id"
                    "intensity_mean"
                    "intensity_std"
    """
    experiment_average_intensity_timeseries = get_experiment_average_intensity_timeseries(ophys_experiment_id)[0]
    intensity_mean = np.mean(experiment_average_intensity_timeseries)
    intensity_std = np.std(experiment_average_intensity_timeseries)
    df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                       "intensity_mean": intensity_mean,
                       "intensity_std": intensity_std},
                      index=[0])

    return df


def container_intensity_mean_and_std(ophys_container_id):
    """Takes the average intensity timeseries from the motion corrected movie
        (already downsampled to be every 500th frame) and gets
        the mean and standard deviation of that time series

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "ophys_experiment_id"
                    "intensity_mean"
                    "intensity_std"
                    "ophys_container_id"
    """

    container_df = passed_experiment_info_for_container(ophys_container_id).sort_values('stage_name_lims').reset_index(drop=True)
    container_ave_intensity = pd.DataFrame()
    for ophys_experiment_id in container_df["ophys_experiment_id"].unique():
        exp_df = experiment_intensity_mean_and_std(ophys_experiment_id)
        container_ave_intensity = container_ave_intensity.append(exp_df)
    container_ave_intensity.loc[:, "ophys_container_id"] = ophys_container_id
    container_ave_intensity = container_ave_intensity.reset_index(drop=True)
    return container_ave_intensity


def experiment_FOV_information(ophys_experiment_id):
    """[summary]

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "ophys_experiment_id",
                    "FOV_intensity_mean",
                    "FOV_intensity_std",
                    "FOV_mean_div_std",
                    "pmt_gain",
                    "median_rsnr_all_csids"
    """
    intensity_df = experiment_intensity_mean_and_std(ophys_experiment_id)
    intensity_df["median_rsnr_all_csids"] = experiment_median_robust_snr_all_csids(ophys_experiment_id)
    intensity_df["pmt_gain"] = data_loading.get_pmt_gain_for_experiment(ophys_experiment_id)

    return intensity_df


def container_pmt_settings(ophys_container_id):
    """[summary]

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "ophys_experiment_id"
                    "pmt_gain"
                    "ophys_container_id"
    """
    container_df = passed_experiment_info_for_container(ophys_container_id).sort_values('stage_name_lims').reset_index(drop=True)
    container_pmt_df = pd.DataFrame()
    for ophys_experiment_id in container_df["ophys_experiment_id"].unique():
        exp_pmt = data_loading.get_pmt_gain_for_experiment(ophys_experiment_id)
        exp_pmt_df = pd.DataFrame({"ophys_experiment_id": ophys_experiment_id,
                                   "pmt_gain": exp_pmt}, index=[0])
        container_pmt_df = container_pmt_df.append(exp_pmt_df)
    container_pmt_df = container_pmt_df.dropna()
    container_pmt_df.pmt_gain = container_pmt_df.pmt_gain.astype(int)
    container_pmt_df.loc[:, "ophys_container_id"] = ophys_container_id
    container_pmt_df = container_pmt_df.reset_index(drop=True)
    return container_pmt_df


def container_FOV_information(ophys_container_id):
    """[summary]

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        dataframe -- dataframe with the following columns:
                    "ophys_experiment_id"
                    "pmt_gain"
                    "ophys_conatiner_id"
                    "intensity_mean"
                    "intensity_std"
                    "median_rsnr_all_csids"
    """
    pmt_settings = container_pmt_settings(ophys_container_id)
    intensity_info = container_intensity_mean_and_std(ophys_container_id)
    snr_summary = container_snr_summary_table(ophys_container_id)
    dfs = [pmt_settings, intensity_info, snr_summary]
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=["ophys_experiment_id", "ophys_container_id"]), dfs)
    return merged_df


def add_dff_stats_to_specimen_table(session):
    '''
    merges statistics about df/f to the cell specimen table
    input is session object
    returns the cell_specimen_table with additional columns describing dff trace stats
    '''
    dff_stats = []
    for idx, row in session.cell_specimen_table.reset_index().iterrows():
        roi_id = row['cell_roi_id']
        dff = session.dff_traces.query('cell_roi_id == @roi_id')['dff'].iloc[0]
        dff_stats_single = pd.Series(dff).describe().to_dict()
        dff_stats_single.update({
            'cell_roi_id': roi_id
        })
        dff_stats.append(dff_stats_single)
    dff_stats = pd.DataFrame(dff_stats)
    df = session.cell_specimen_table.reset_index().merge(
        dff_stats,
        left_on='cell_roi_id',
        right_on='cell_roi_id'
    )
    if 'cell_specimen_id' not in df.columns:
        df['cell_specimen_id'] = None
    return df


def add_hits_to_licks(licks, trials):
    """
    Adds a column to the licks table to indicate which licks were hits, resulting in a reward
    :param licks: BehaviorOphysSession or BehaviorSession licks dataframe
    :param trials: BehaviorOphysSession or BehaviorSession trials dataframe
    :return: annotated licks dataframe
    """
    licks['hit'] = False
    hit_licks = [lick_time[0] for lick_time in trials[trials.hit == True].lick_times.values]
    indices = licks[licks.timestamps.isin(hit_licks)].index.values
    licks.at[indices, 'hit'] = True
    return licks


def add_bouts_to_licks(licks):
    """
    Adds columns to licks dataframe for inter flash lick difference and whether a lick was in a bout, given the median inter lick interval
    :param licks: BehaviorOphysSession or BehaviorSession licks dataframe
    :return: annotated licks dataframe
    """
    licks['inter_flash_lick_diff'] = [licks.iloc[row].timestamps - licks.iloc[row - 1].timestamps if row != 0 else np.nan for row in range(len(licks))]
    lick_times = licks.timestamps.values
    median_inter_lick_interval = np.median(np.diff(np.hstack(list(lick_times))))
    licks['lick_in_bout'] = False
    indices = licks[licks.inter_flash_lick_diff < median_inter_lick_interval * 3].index
    licks.at[indices, 'lick_in_bout'] = True
    return licks


def get_multi_session_df(experiments, df_name, conditions, use_events=False, filter_events=False,
                         use_extended_stimulus_presentations=False):
    """
    Code to generate a concatenated multi session dataframe with cell responses averaged over a set of conditions.
    This functionality also exists in scripts for the cluster where it will create the dataframes for all cre lines and session types automatically.
    This is a standalone version that runs only on the set of experiments provided (ex: just on one cre line)
    :param experiments: experiments_table filtered to include only experiments you want to include in the concatenated df
    :param df_name: which ResponseAnalysis dataframe to use; 'trials_response_df', 'omission_response_df', or 'stimulus_response_df'
    :param conditions: conditions over which to group by and average for each cell, must be a column in the response dataframe;
                        ex: ['cell_specimen_id', 'image_name'] or ['cell_specimen_id', 'engagement_state', 'hit']
    :param use_events: Boolean; whether or not to use detected events
    :param filter_events: Boolean; whether or not to use filtered events
    :param use_extended_stimulus_presentations: Boolean; whether or not to merge response df with extended_stimulus_presentations or the default SDK version of stimulus_presentations
    :return:
    """
    import visual_behavior.ophys.response_analysis.utilities as ut
    from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

    if 'stimulus' in df_name:
        flashes = True
        omitted = False
        get_pref_stim = True
    elif 'omission' in df_name:
        flashes = False
        omitted = True
        get_pref_stim = False
    elif 'trials' in df_name:
        flashes = False
        omitted = False
        get_pref_stim = True
    else:
        print('unable to set params for', df_name)
    if ('run_speed' in df_name) or ('pupil_area' in df_name):
        get_pref_stim = False
    if ('engaged' in conditions) or ('engagement_state' in conditions):
        use_extended_stimulus_presentations = True

    mega_mdf = pd.DataFrame()
    for i, experiment_id in enumerate(experiments.index):
        print('processing experiment', i, 'out of', str(len(experiments.index)))
        try:
            print(experiment_id)
            dataset = data_loading.get_ophys_dataset(experiment_id)
            analysis = ResponseAnalysis(dataset, use_events=use_events, filter_events=filter_events,
                                        use_extended_stimulus_presentations=use_extended_stimulus_presentations)
            df = analysis.get_response_df(df_name)
            df['ophys_experiment_id'] = experiment_id
            if 'passive' in dataset.metadata['session_type']:
                df['lick_on_next_flash'] = False
                df['engaged'] = False
                df['engagement_state'] = 'disengaged'
            if 'running' in conditions:
                df['running'] = [True if mean_running_speed > 2 else False for mean_running_speed in df.mean_running_speed.values]
            if 'large_pupil' in conditions:
                if 'mean_pupil_area' in df.keys():
                    df = df[df.mean_pupil_area.isnull() == False]
                    if len(df) > 100:
                        median_pupil_area = df.mean_pupil_area.median()
                        df['large_pupil'] = [True if mean_pupil_area > median_pupil_area else False for mean_pupil_area in
                                             df.mean_pupil_area.values]
            mdf = ut.get_mean_df(df, analysis, conditions=conditions, get_pref_stim=get_pref_stim,
                                 flashes=flashes, omitted=omitted, get_reliability=True,
                                 exclude_omitted_from_pref_stim=True)
            if 'correlation_values' in mdf.keys():
                mdf = mdf.drop(columns=['correlation_values'])
            mdf['ophys_experiment_id'] = experiment_id
            print('mean df created for', experiment_id)
            mega_mdf = pd.concat([mega_mdf, mdf])
        except Exception as e:  # flake8: noqa: E722
            print(e)
            print('problem for', experiment_id)

    if 'level_0' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='level_0')
    if 'index' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='index')

    print('finished creating multi_session_df')
    return mega_mdf


def zscore_pupil_data(eye_tracking):
    """
    removes likely blinks from all columns of eye_tracking attribute of VisualBehaviorOphysDataset object
    adds a pupil radius column based on pupil area
    and z-scores pupil radius and pupil width
    :param eye_tracking:
    :return:
    """
    import scipy

    eye = eye_tracking.copy()

    # Compute pupil radius
    eye['pupil_radius'] = np.sqrt(eye['pupil_area']*(1/np.pi))

    # Remove likely blinks and interpolate
    eye.loc[eye['likely_blink'],:] = np.nan
    eye = eye.interpolate()

    eye['pupil_radius_zscored'] = scipy.stats.zscore(eye['pupil_radius'], nan_policy='omit')
    eye['pupil_width_zscored'] = scipy.stats.zscore(eye['pupil_width'], nan_policy='omit')
    eye['pupil_area_zscored'] = scipy.stats.zscore(eye['pupil_area'], nan_policy='omit')
    return eye
