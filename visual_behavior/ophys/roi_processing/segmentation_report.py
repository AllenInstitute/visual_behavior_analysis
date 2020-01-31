import visual_behavior.ophys.roi_processing.roi_processing as roi
import visual_behavior.ophys.roi_processing.data_loading as load

import pandas as pd
import numpy as np


exclusion_labels_list = ["apical_dendrite", "bad_shape", "boundary", "demix_error", "duplicate", "empty_neropil_mask",
                         "empty_roi_mask", "low_signal", "motion_border", "small_size", "union", "zero_pixels"]

roi_id_types_list = ["cell_specimen_id", "cell_roi_id"]



####### EXPERIMENT ####### # NOQA: E402


def experiment_segmentation_dataframe(experiment_id, generate_masks=False):
    """gatherers basic segmentation information like number or rois and roi failure tag counts for a particular experiment and returns it as a dataframe

    Arguments:
        experiment_id {[type]} -- [description]

    Returns:
        dataframe -- pandas dataframe with the following columns:
                    "label":  the category that the subsequent columns/metrics are being computed for.
                    Is currently all the valid rois, all the invalid rois, and then the exclusion labels
                                "all_valid"
                                "all_invalid",
                                "apical_dendrite",
                                "bad_shape",
                                "boundary",
                                "demix_error",
                                "duplicated",
                                "empty_neuropil_mask",
                                "empty_roi_mask",
                                "low_signal"
                                "motion_border"
                                "small_size"
                                "union"
                                "zero_pixels"
                    "cell_specimen_ids": array of all the cell specimen ids for the given label
                    "cell_roi_ids": array of all the cell roi ids for the given label
                    "roi_count": the number of rois with that label
                    "multi_roi_mask": a transparent mask that has all the rois for that given label

    """

    if generate_masks == True:
        seg_labels_df = gen_seg_labels_df(experiment_id)
        label_mask_df = gen_label_masks_df(experiment_id)
        segmentation_df = pd.merge(seg_labels_df, label_mask_df, how="outer", on="label")
        segmentation_df.loc[:, "experiment_id"] = int(experiment_id)
        segmentation_df = segmentation_df[["experiment_id",
                                           "label",
                                           "roi_count",
                                           "percent_of_all_rois",
                                           "cell_specimen_ids",
                                           "cell_roi_ids",
                                           "multi_roi_mask"]]
    else:
        segmentation_df = gen_seg_labels_df(experiment_id)
        segmentation_df.loc[:, "experiment_id"] = int(experiment_id)
        segmentation_df = segmentation_df[["experiment_id",
                                           "label",
                                           "roi_count",
                                           "percent_of_all_rois",
                                           "cell_specimen_ids",
                                           "cell_roi_ids"]]
    return segmentation_df


def gen_seg_labels_df(experiment_id):
    """combines the roi validity dataframe(all the validy and invalid cells) and the exclusion label dataframe (lists all the exclusion labels
        and how many rois are associated with each)

    Arguments:
        experiment_id {[type]} -- [description]

    Returns:
        dataframe -- pandas dataframe with the following columns:
                    "label":  the category that the subsequent columns/metrics are being computed for.
                                "all_valid",
                                "all_invalid",
                                "apical_dendrite",
                                "bad_shape",
                                "boundary",
                                "demix_error",
                                "duplicated",
                                "empty_neuropil_mask",
                                "empty_roi_mask",
                                "low_signal",
                                "motion_border",
                                "small_size",
                                "union",
                                "zero_pixels",
                    "cell_specimen_ids": array of all the cell specimen ids for the given label
                    "cell_roi_ids": array of all the cell roi ids for the given label
                    "roi_count": the number of rois with that label
    """
    exclusion_df = gen_exclusion_label_roi_df(experiment_id)
    roi_validity_df = gen_roi_validity_df(experiment_id)
    seg_labels_df = roi_validity_df.append(exclusion_df)
    seg_labels_df = seg_labels_df.reset_index(drop=True)
    seg_labels_df["roi_count"] = seg_labels_df['cell_specimen_ids'].str.len()
    seg_labels_df = seg_labels_df.reset_index(drop=True)

    total_seg_rois = seg_labels_df.loc[seg_labels_df["label"] == "all_valid", "roi_count"].values[0] + seg_labels_df.loc[seg_labels_df["label"] == "all_invalid", "roi_count"].values[0]
    seg_labels_df.loc[:, "percent_of_all_rois"] = np.round(seg_labels_df["roi_count"] / total_seg_rois, 2)
    return seg_labels_df


def gen_exp_seg_validity_summary_df(experiment_id, masks=True):
    roi_metrics = roi.exp_roi_metrics_dataframe(experiment_id, mask_shift=True)
    blank_mask = roi.gen_blank_mask_of_FOV_dimensions(experiment_id)

    valid_csi = roi_metrics.loc[roi_metrics["valid_roi"] == True, "cell_specimen_id"].values
    invalid_csi = roi_metrics.loc[roi_metrics["valid_roi"] == False, "cell_specimen_id"].values

    valid_count = len(valid_csi)
    invalid_count = len(invalid_csi)

    if masks == False:
        exp_seg_validity_summary = pd.DataFrame({"experiment_id": experiment_id,
                                                 "valid_count": valid_count, "invalid_count": invalid_count,
                                                 "valid_csis": [valid_csi], "invalid_csis": [invalid_csi]})

    else:
        if valid_count == 0:
            valid_mask = blank_mask
        else:
            valid_mask = roi.multi_roi_mask_from_df(roi_metrics, valid_csi)

        if invalid_count == 0:
            invalid_mask = blank_mask
        else:
            invalid_mask = roi.multi_roi_mask_from_df(roi_metrics, invalid_csi)

        exp_seg_validity_summary = pd.DataFrame({"experiment_id": experiment_id,
                                                 "valid_count": valid_count, "invalid_count": invalid_count,
                                                 "valid_csis": [valid_csi], "invalid_csis": [invalid_csi],
                                                 "valid_mask": [valid_mask], "invalid_mask": [invalid_mask]})

    return exp_seg_validity_summary


def gen_roi_validity_df(experiment_id):
    """dataframe that lists all valid rois, and all invalid rois (those with an associated failure tag)

    Arguments:
        experiment_id {[type]} -- [description]

    Returns:
        Dataframe -- dataframe with the following columns:
                        "label": generic label:
                                    "all_valid": valid cells for the experiment
                                    "all_invalid": all invalid cells for the experiment
                        "cell_specimen_ids: a list of the cell specimen ids that have that specific label (either all valid or all invalid)
                        "cell_roi_ids": a list of the cell roi ids that have that specific label (either all valid or all invalid)
    """
    cell_specimen_table = load.get_sdk_cell_specimen_table(experiment_id)
    validity_df = pd.DataFrame(columns=["label", "cell_specimen_ids", "cell_roi_ids"])
    boolean = [True, False]
    for b in boolean:
        cell_spec_ids = cell_specimen_table.loc[cell_specimen_table["valid_roi"] == b, "cell_specimen_id"].values
        cell_roi_ids = cell_specimen_table.loc[cell_specimen_table["valid_roi"] == b, "cell_specimen_id"].values
        df = pd.DataFrame({"label": b, "cell_specimen_ids": [cell_spec_ids], "cell_roi_ids": [cell_roi_ids]})
        validity_df = validity_df.append(df)
        validity_df["label"] = np.where(validity_df["label"], "all_valid", "all_invalid")
    return validity_df


def gen_exclusion_label_roi_df(experiment_id):
    """dataframe of roi exclusion labels/failure tags and the roi ids associated with those tags

    Arguments:
        experiment_id {[type]} -- [description]

    Returns:
        Dataframe -- dataframe with the following columns:
                    "label": exclusion labels :
                                "apical_dendrite"
                                "bad_shape"
                                "boundary"
                                "demix_error"
                                "duplicate"
                                "empty_neuropil_mask"
                                "empty_roi_mask"
                                "low_signal"
                                "motion_border"
                                "small_size"
                                "union"
                                "zero pixels"
                    "cell_specimen_ids": a list of the cell specimen ids that have that specific label
                    "cell_roi_ids": a list of the cell roi ids that have that specific label
    """
    exclusion_labels_df = load.get_failed_roi_exclusion_labels(experiment_id)
    excl_label_roi_df = pd.DataFrame()

    for ex_label in exclusion_labels_list:
        cell_spec_ids = exclusion_labels_df.loc[exclusion_labels_df["exclusion_label_name"] == ex_label, "cell_specimen_id"].values
        cell_r_ids = exclusion_labels_df.loc[exclusion_labels_df["exclusion_label_name"] == ex_label, "cell_roi_id"].values
        df = pd.DataFrame({"label": ex_label, "cell_specimen_ids": [cell_spec_ids], "cell_roi_ids": [cell_r_ids]})
        excl_label_roi_df = excl_label_roi_df.append(df)
    return excl_label_roi_df


def gen_label_masks_df(experiment_id, print_lab=False):
    seg_labels_df = gen_seg_labels_df(experiment_id)
    roi_metrics = roi.exp_roi_metrics_dataframe(experiment_id, mask_shift=True)
    blank_mask = roi.gen_blank_mask_of_FOV_dimensions(experiment_id)

    mask_df = pd.DataFrame(columns=["label", "multi_roi_mask"])
    for label in seg_labels_df["label"]:
        lab = label
        if print_lab == True:
            print(lab)
        label_df = seg_labels_df.loc[seg_labels_df["label"] == lab]
        if label_df["roi_count"].values[0] == 0:
            label_mask = blank_mask
        else:
            label_mask = roi.multi_roi_mask_from_df(roi_metrics, seg_labels_df.loc[seg_labels_df["label"] == lab, "cell_specimen_ids"].values[0])
        df = pd.DataFrame({"label": [lab], "multi_roi_mask": [label_mask]})
        mask_df = mask_df.append(df)
    return mask_df




####### CONTAINER ####### # NOQA: E402


def container_segmentation_dataframe(container_id,
                                     generate_masks=False,
                                     include_ffield_test=False,
                                     include_failed_sessions=False,
                                     print_exp=False):

    container_manifest = roi.gen_container_manifest(container_id,
                                                    include_ffield_test=include_ffield_test,
                                                    include_failed_sessions=include_failed_sessions)

    stage_names = container_manifest[["experiment_id", "stage_name_lims"]].copy()
    experiments_list = container_manifest["experiment_id"].values.tolist()

    container_seg_df = pd.DataFrame()
    for experiment_id in experiments_list:
        if print_exp == True:
            print("experiment_id: " + str(experiment_id))
        experiment_seg_df = experiment_segmentation_dataframe(experiment_id, generate_masks == generate_masks)
        container_seg_df = container_seg_df.append(experiment_seg_df)

    container_seg_df = container_seg_df.reset_index(drop=True)
    container_seg_df.loc[:, "container_id"] = int(container_id)
    container_seg_df = pd.merge(container_seg_df, stage_names, how="left", on="experiment_id")
    # order of columns
    if generate_masks == True:
        container_seg_df = container_seg_df[["container_id",
                                             "experiment_id",
                                             "stage_name_lims",
                                             "label",
                                             "roi_count",
                                             "percent_of_all_rois",
                                             "cell_specimen_ids",
                                             "cell_roi_ids",
                                             "multi_roi_mask"]]

    else:
        container_seg_df = container_seg_df[["container_id",
                                             "experiment_id",
                                             "stage_name_lims",
                                             "label",
                                             "roi_count",
                                             "percent_of_all_rois",
                                             "cell_specimen_ids",
                                             "cell_roi_ids"]]
    return container_seg_df


def gen_container_seg_validitity_summary_df(container_id, masks=True):
    container_manifest = roi.gen_container_manifest(container_id)

    experiment_id_list = container_manifest["experiment_id"].unique()
    experiment_id_list = experiment_id_list.tolist()

    container_seg_validity_df = pd.DataFrame()

    for experiment_id in experiment_id_list:
        exp_seg_validity_summary = gen_exp_seg_validity_summary_df(experiment_id, masks=masks)

        container_seg_validity_df = container_seg_validity_df.append(exp_seg_validity_summary)

    container_seg_validity_df.loc[:, "container_id"] = container_id
    container_seg_validity_df = container_seg_validity_df.reset_index(drop=True)

    return container_seg_validity_df
