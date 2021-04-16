from visual_behavior.data_access import utilities as utils


def correct_general_info_filepaths(general_info_df):
    storage_directory_columns_list = ['experiment_storage_directory',
                                      'session_storage_directory',
                                      'container_storage_directory']

    for directory in storage_directory_columns_list:
        general_info_df = utils.correct_dataframe_filepath(general_info_df, directory)

    return general_info_df


def get_filepath_from_realdict_object(realdict_object):
    """takes a RealDictRow object returned when loading well known files
       from lims and parses it to return the filepath to the well known file.

    Args:
        wkf_realdict_object ([type]): [description]

    Returns:
        filepath: [description]
    """
    filepath = realdict_object['filepath'][0]
    filepath = utils.correct_filepath(filepath)
    return filepath


def update_objectlist_column_labels(objectlist_df):
    """take the roi metrics from the objectlist.txt file and renames
       them to be more explicit and descriptive.
        -removes single blank space at the beginning of column names
        -enforced naming scheme(no capitolization, added _)
        -renamed columns to be more descriptive/reflect contents of column

    Arguments:
        objectlist_dataframe {dataframe} -- dataframe from the objectlist.txt
                                            file containing various roi metrics

    Returns:
        dataframe -- same dataframe with same information
                     but with more informative column names
    """

    objectlist_df = objectlist_df.rename(index=str, columns={' traceindex': "trace_index",
                                                             ' cx': 'center_x',
                                                             ' cy': 'center_y',
                                                             ' mask2Frame': 'frame_of_max_intensity_masks_file',
                                                             ' frame': 'frame_of_enhanced_movie',
                                                             ' object': 'layer_of_max_intensity_file',
                                                             ' minx': 'bbox_min_x',
                                                             ' miny': 'bbox_min_y',
                                                             ' maxx': 'bbox_max_x',
                                                             ' maxy': 'bbox_max_y',
                                                             ' area': 'area',
                                                             ' shape0': 'ellipseness',
                                                             ' shape1': "compactness",
                                                             ' eXcluded': "exclude_code",
                                                             ' meanInt0': "mean_intensity",
                                                             ' meanInt1': "mean_enhanced_intensity",
                                                             ' maxInt0': "max_intensity",
                                                             ' maxInt1': "max_enhanced_intensity",
                                                             ' maxMeanRatio': "intensity_ratio",
                                                             ' snpoffsetmean': "soma_minus_np_mean",
                                                             ' snpoffsetstdv': "soma_minus_np_std",
                                                             ' act2': "sig_active_frames_2_5",
                                                             ' act3': "sig_active_frames_4",
                                                             ' OvlpCount': "overlap_count",
                                                             ' OvlpAreaPer': "percent_area_overlap",
                                                             ' OvlpObj0': "overlap_obj0_index",
                                                             ' OvlpObj1': "overlap_obj1_index",
                                                             ' corcoef0': "soma_obj0_overlap_trace_corr",
                                                             ' corcoef1': "soma_obj1_overlap_trace_corr"})
    return objectlist_df
