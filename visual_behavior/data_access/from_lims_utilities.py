from visual_behavior.data_access import from_lims
from visual_behavior.data_access import utilities as utils
import visual_behavior.data_access.pre_post_conditions as conditions


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
        -enforced naming scheme(no camel case, added _)
        -renamed columns to be more descriptive/reflect contents of column

    Arguments:
        objectlist_dataframe {dataframe} -- dataframe from the objectlist.txt
                                            file containing various roi metrics

    Returns:
        dataframe -- same dataframe with same information
                     but with more informative column names
    """
    objectlist_df = objectlist_df.rename(index=str,
                                         columns={' traceindex':    'trace_index',                         # noqa: E241
                                                  ' cx':            'center_x',                            # noqa: E241
                                                  ' cy':            'center_y',                            # noqa: E241
                                                  ' mask2Frame':    'frame_of_max_intensity_masks_file',   # noqa: E241
                                                  ' frame':         'frame_of_enhanced_movie',             # noqa: E241
                                                  ' object':        'layer_of_max_intensity_file',         # noqa: E241
                                                  ' minx':          'bbox_min_x',                          # noqa: E241
                                                  ' miny':          'bbox_min_y',                          # noqa: E241
                                                  ' maxx':          'bbox_max_x',                          # noqa: E241
                                                  ' maxy':          'bbox_max_y',                          # noqa: E241
                                                  ' area':          'area',                                # noqa: E241
                                                  ' shape0':        'ellipseness',                         # noqa: E241
                                                  ' shape1':        'compactness',                         # noqa: E241
                                                  ' eXcluded':      'exclude_code',                        # noqa: E241
                                                  ' meanInt0':      'mean_intensity',                      # noqa: E241
                                                  ' meanInt1':      'mean_enhanced_intensity',             # noqa: E241
                                                  ' maxInt0':       'max_intensity',                       # noqa: E241
                                                  ' maxInt1':       'max_enhanced_intensity',              # noqa: E241
                                                  ' maxMeanRatio':  'intensity_ratio',                     # noqa: E241
                                                  ' snpoffsetmean': 'soma_minus_np_mean',                  # noqa: E241
                                                  ' snpoffsetstdv': 'soma_minus_np_std',                   # noqa: E241
                                                  ' act2':          'sig_active_frames_2_5',               # noqa: E241
                                                  ' act3':          'sig_active_frames_4',                 # noqa: E241
                                                  ' OvlpCount':     'overlap_count',                       # noqa: E241
                                                  ' OvlpAreaPer':   'percent_area_overlap',                # noqa: E241
                                                  ' OvlpObj0':      'overlap_obj0_index',                  # noqa: E241
                                                  ' OvlpObj1':      'overlap_obj1_index',                  # noqa: E241
                                                  ' corcoef0':      'soma_obj0_overlap_trace_corr',        # noqa: E241
                                                  ' corcoef1':      'soma_obj1_overlap_trace_corr'})       # noqa: E241
    return objectlist_df


MICROSCOPE_TYPE_EQUIPMENT_NAMES_DICT = {
    "Nikon":       ["CAM2P.1", "CAM2P.2"],                     # noqa: E241
    "Scientifica": ["CAM2P.3, CAM2P.4, CAM2P.5, CAM2P.6"],
    "Mesoscope":   ["MESO.1", "MESO.2"]}                       # noqa: E241


def get_microscope_equipment_name(ophys_session_id):
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    equipment_name = from_lims.get_general_info_for_ophys_session_id(ophys_session_id)["equipment_name"][0]
    return equipment_name


def get_microscope_type(ophys_session_id):
    equipment_name = get_microscope_equipment_name(ophys_session_id)

    for key, value in MICROSCOPE_TYPE_EQUIPMENT_NAMES_DICT.items():
        if equipment_name in value:
            return key
    return "Cannot find microscope type for {}".format(equipment_name)
