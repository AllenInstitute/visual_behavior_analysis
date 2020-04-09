from allensdk.internal.api import PostgresQueryMixin
from visual_behavior.translator.allensdk_sessions import sdk_utils
import visual_behavior.ophys.io.convert_level_1_to_level_2 as convert
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc

import os
import h5py  # for loading motion corrected movie
import numpy as np
import pandas as pd
import configparser as configp  # for parsing scientifica ini files


lims_dbname = os.environ["LIMS_DBNAME"]
lims_user = os.environ["LIMS_USER"]
lims_host = os.environ["LIMS_HOST"]
lims_password = os.environ["LIMS_PASSWORD"]
lims_port = os.environ["LIMS_PORT"]

mtrain_dbname = os.environ["MTRAIN_DBNAME"]
mtrain_user = os.environ["MTRAIN_USER"]
mtrain_host = os.environ["MTRAIN_HOST"]
mtrain_password = os.environ["MTRAIN_PASSWORD"]
mtrain_port = os.environ["MTRAIN_PORT"]

lims_engine = PostgresQueryMixin(dbname=lims_dbname,
                                 user=lims_user,
                                 host=lims_host,
                                 password=lims_password,
                                 port=lims_port)

mtrain_engine = PostgresQueryMixin(dbname=mtrain_dbname,
                                   user=mtrain_user,
                                   host=mtrain_host,
                                   password=mtrain_password,
                                   port=mtrain_port)

get_psql_dict_cursor = convert.get_psql_dict_cursor  # to load well-known files
config = configp.ConfigParser()

# function inputs
# ophys_experiment_id
# ophys_session_id
# behavior_session_id
# ophys_container_id


#  GENERAL STUFF

def get_super_container_plots_dir():
    return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/super_container_plots'


def get_container_plots_dir():
    return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/container_plots'


def get_session_plots_dir():
    return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/session_plots'


def get_experiment_plots_dir():
    return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/experiment_plots'


#  FROM MANIFEST

def get_qc_manifest_path():
    """Get path to default manifest file for QC"""
    manifest_path = "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/2020_cache/production_cache/manifest.json"
    return manifest_path


def get_qc_cache(manifest_path=None):
    """Get cache using default QC manifest path"""
    if manifest_path is None:
        manifest_path = get_qc_manifest_path()
    cache = bpc.from_lims(manifest=get_qc_manifest_path())
    return cache


def get_filtered_ophys_sessions_table():
    """Get ophys sessions that meet the criteria in sdk_utils.get_filtered_sessions_table(), including that the container
    must be 'complete' or 'container_qc', and experiments associated with the session are passing. In addition,
    add container_id and container_workflow_state to table.

        Arguments:
            None

        Returns:
            filtered_sessions -- filtered version of ophys_session_table from QC cache
        """
    cache = get_qc_cache()
    sessions = sdk_utils.get_filtered_sessions_table(cache, include_multiscope=True, require_full_container=False,
                                                     require_exp_pass=False)
    sessions = sessions.reset_index()
    experiments = get_filtered_ophys_experiment_table()
    filtered_sessions = sessions.merge(experiments[['ophys_session_id', 'container_id', 'container_workflow_state']],
                                       right_on='ophys_session_id', left_on='ophys_session_id')
    return filtered_sessions


def get_filtered_ophys_experiment_table(include_failed_data=False):
    """Get ophys experiments that meet the criteria in sdk_utils.get_filtered_sessions_table(), including that the container
        must be 'complete' or 'container_qc', and experiments are passing. Includes MultiScope data.

            Arguments:
                include_failed_data: Boolean, if False, only include passing behavior experiments from containers that were not failed.
                                If True, return all experiments including those from failed containers and receptive field mapping experiments.

            Returns:
                filtered_experiments -- filtered version of ophys_experiment_table from QC cache
            """
    cache = get_qc_cache()
    experiments = cache.get_experiment_table()
    experiments = experiments.reset_index()
    experiments['super_container_id'] = experiments['specimen_id'].values

    experiments = reformat_experiments_table(experiments)
    if include_failed_data:
        filtered_experiments = experiments[experiments.experiment_workflow_state.isin(['passed', 'failed'])]
    else:
        # sessions = sdk_utils.get_filtered_sessions_table(cache, include_multiscope=True, require_full_container=False, require_exp_pass=False)
        # sessions = sessions.reset_index()
        # filtered_experiments = experiments[experiments.ophys_session_id.isin(sessions.ophys_session_id.unique())]
        filtered_experiments = experiments.copy()
        filtered_experiments = filtered_experiments[filtered_experiments.experiment_workflow_state == 'passed']
        filtered_experiments = filtered_experiments[filtered_experiments.container_workflow_state != 'failed']  # include containers in holding

    return filtered_experiments


def add_mouse_seeks_fail_tags_to_experiments_table(experiments):
    mouse_seeks_report_file_base = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    report_file = 'ophys_session_log_031820.xlsx'
    vb_report_path = os.path.join(mouse_seeks_report_file_base, report_file)
    vb_report_df = pd.read_excel(vb_report_path)

    def clean_columns(columns):
        return [c.lower().replace(' ', '_') for c in columns]

    vb_report_df.columns = clean_columns(vb_report_df.columns)
    vb_report_df = vb_report_df.rename(columns={'session_id': 'ophys_session_id'})
    # merge fail tags into all_experiments manifest
    experiments = experiments.merge(vb_report_df[['ophys_session_id', 'session_tags', 'failure_tags']],
                                    right_on='ophys_session_id', left_on='ophys_session_id')
    return experiments


# def reformat_fail_tags_for_multiscope(experiments):
#     experiments['fail_tags'] = experiments['failure_tags']
#     meso_expts = experiments[
#         experiments.project_code.isin(['VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d'])]
#     experiments.at[meso_expts.index.values, 'fail_tags'] = None
#
#     for session_id in meso_expts.ophys_session_id.unique():
#         experiment_ids = meso_expts[meso_expts.ophys_session_id == session_id].ophys_experiment_id.values
#         for i, experiment_id in enumerate(experiment_ids):
#             expt_data = experiments[experiments.ophys_experiment_id == experiment_id]
#             index = expt_data.index.values[0]
#             if type(expt_data.failure_tags.values[0]) == str:
#                 fail_tags = expt_data.failure_tags.values[0].split(',')
#                 tags_this_expt = []
#                 for fail_tag in fail_tags:
#                     if (fail_tag[-1].isdigit()) and (fail_tag[-1] == str(int(expt_data.container_index.values[0]))):
#                         tags_this_expt.append(fail_tag)
#                     elif (fail_tag[-1].isdigit() == False):
#                         tags_this_expt.append(fail_tag)
#                         experiments.at[index, 'fail_tags'] = tags_this_expt
#     return experiments


def add_location_to_expts(expts):
    expts['location'] = [expts.loc[x]['cre_line'].split('-')[0] + '_' + expts.loc[x]['targeted_structure'] + '_' + str(int(expts.loc[x]['imaging_depth'])) for x in expts.index.values]
    return expts


def get_exposure_number_for_group(group):
    order = np.argsort(group['date_of_acquisition'].values)
    group['exposure_number'] = order
    return group


def add_exposure_number_to_experiments_table(experiments):
    experiments = experiments.groupby(['super_container_id', 'container_id', 'session_type']).apply(get_exposure_number_for_group)
    return experiments


def reformat_experiments_table(experiments):
    experiments = experiments.reset_index()
    # clean up cre_line naming
    experiments['cre_line'] = [driver_line[1] if driver_line[0] == 'Camk2a-tTA' else driver_line[0] for driver_line in
                               experiments.driver_line.values]
    experiments = experiments[experiments.cre_line != 'Cux2-CreERT2']  # why is this here?
    # replace session types that are NaN with string None
    experiments.at[experiments[experiments.session_type.isnull()].index.values, 'session_type'] = 'None'
    # experiments = add_container_index_to_experiments_table(experiments)
    # experiments = revise_container_id_for_multiscope(experiments)
    experiments = add_mouse_seeks_fail_tags_to_experiments_table(experiments)
    # experiments = reformat_fail_tags_for_multiscope(experiments)
    experiments = add_exposure_number_to_experiments_table(experiments)
    if 'level_0' in experiments.columns:
        experiments = experiments.drop(columns='level_0')
    if 'index' in experiments.columns:
        experiments = experiments.drop(columns='index')
    experiments = add_location_to_expts(experiments)
    return experiments


def get_filtered_ophys_container_ids():
    """Get container_ids that meet the criteria in sdk_utils.get_filtered_sessions_table(), including that the container
    must be 'complete' or 'container_qc', and experiments associated with the container are passing. """
    experiments = get_filtered_ophys_experiment_table()
    filtered_container_ids = np.sort(experiments.container_id.unique())
    return filtered_container_ids


def get_ophys_session_ids_for_ophys_container_id(ophys_container_id):
    """Get ophys_session_ids belonging to a given ophys_container_id. ophys container must meet the criteria in
    sdk_utils.get_filtered_sessions_table()

            Arguments:
                ophys_container_id -- must be in get_filtered_ophys_container_ids()

            Returns:
                ophys_session_ids -- list of ophys_session_ids that meet filtering criteria
            """
    # sessions = get_filtered_ophys_sessions_table()
    # ophys_session_ids = np.sort(sessions[(sessions.container_id == ophys_container_id)].ophys_session_id.values)
    # temporary fix - need to make get_filtered_ophys_sessions_table work properly
    experiments = get_filtered_ophys_experiment_table()
    ophys_session_ids = np.sort(experiments[(experiments.container_id == ophys_container_id)].ophys_session_id.values)
    return ophys_session_ids


def get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id):
    """Get ophys_experiment_ids belonging to a given ophys_container_id. ophys container must meet the criteria in
        sdk_utils.get_filtered_sessions_table()

                Arguments:
                    ophys_container_id -- must be in get_filtered_ophys_container_ids()

                Returns:
                    ophys_experiment_ids -- list of ophys_experiment_ids that meet filtering criteria
                """
    experiments = get_filtered_ophys_experiment_table()
    ophys_experiment_ids = np.sort(experiments[(experiments.container_id == ophys_container_id)].ophys_experiment_id.values)
    return ophys_experiment_ids


def get_session_type_for_ophys_experiment_id(ophys_experiment_id):
    experiments = get_filtered_ophys_experiment_table()
    session_type = experiments[experiments.ophys_experiment_id == ophys_experiment_id].session_type.values[0]
    return session_type


def get_session_type_for_ophys_session_id(ophys_session_id):
    # sessions = get_filtered_ophys_sessions_table()
    # session_type = sessions[sessions.ophys_session_id == ophys_session_id].session_type.values[0]
    # temporary fix!!
    experiments = get_filtered_ophys_experiment_table()
    session_type = experiments[experiments.ophys_session_id == ophys_session_id].session_type.values[0]
    return session_type


def get_ophys_experiment_id_for_ophys_session_id(ophys_session_id):
    experiments = get_filtered_ophys_experiment_table()
    ophys_experiment_id = experiments[(experiments.ophys_session_id == ophys_session_id)].ophys_experiment_id.values[0]
    return ophys_experiment_id


def get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id):
    lims_experiment_info = get_lims_experiment_info(ophys_experiment_id)
    ophys_session_id = lims_experiment_info["ophys_session_id"][0]
    return ophys_session_id

#  FROM SDK


def get_sdk_session_obj(ophys_experiment_id, include_invalid_rois=False):
    """Use LIMS API from SDK to return session object

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID
        include_invalid_rois {Boolean} -- if True, return all ROIs including invalid. If False, filter out invalid ROIs

    Returns:
        session object -- session object from SDK
    """
    api = BehaviorOphysLimsApi(ophys_experiment_id)
    session = BehaviorOphysSession(api)
    if include_invalid_rois == False:
        dff_traces = session.dff_traces
        valid_cells = session.cell_specimen_table[session.cell_specimen_table.valid_roi == True].cell_roi_id.values
        session.dff_traces = dff_traces[dff_traces.cell_roi_id.isin(valid_cells)]
    return session


def get_sdk_max_projection(ophys_experiment_id):
    """ uses SDK to return 2d max projection image of the microscope field of view


    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        image -- can be visualized via plt.imshow(max_projection)
    """
    session = get_sdk_session_obj(ophys_experiment_id)
    max_projection = session.max_projection
    return max_projection


def get_sdk_ave_projection(ophys_experiment_id):
    """uses SDK to return 2d image of the 2-photon microscope filed of view, averaged
        across the experiment.

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        image -- can be visualized via plt.imshow(ave_projection)
    """
    session = get_sdk_session_obj(ophys_experiment_id)
    ave_projection = session.average_projection
    return ave_projection


def get_sdk_segmentation_mask_image(ophys_experiment_id):
    """uses SDK to return an array containing the masks of all cell ROIS

    Arguments:
       ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
       array -- a 2D boolean array
                visualized via plt.imshow(seg_mask_image)
    """
    session = get_sdk_session_obj(ophys_experiment_id)
    seg_mask_image = session.segmentation_mask_image.data
    return seg_mask_image


def get_sdk_roi_masks(ophys_experiment_id):
    """uses sdk to return a dictionary with individual ROI
        masks for each cell specimen ID.

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        dictonary -- keys are cell specimen ids(ints)
                    values are 2d numpy arrays(binary array
                    the size of the motion corrected 2photon
                    FOV where 1's are the ROI/Cell mask).
                    specific cell masks can be visualized via
                    plt.imshow(roi_masks[cell_specimen_id])
    """

    session = get_sdk_session_obj(ophys_experiment_id)
    roi_masks = session.get_roi_masks
    return roi_masks


def get_valid_segmentation_mask(ophys_experiment_id):
    session = get_sdk_session_obj(ophys_experiment_id)
    ct = session.cell_specimen_table
    valid_cell_specimen_ids = ct[ct.valid_roi == True].index.values
    roi_masks = session.get_roi_masks()
    valid_roi_masks = roi_masks[roi_masks.cell_specimen_id.isin(valid_cell_specimen_ids)].data
    valid_segmentation_mask = np.sum(valid_roi_masks, axis=0)
    valid_segmentation_mask[valid_segmentation_mask > 0] = 1
    return valid_segmentation_mask


def get_sdk_cell_specimen_table(ophys_experiment_id):
    """[summary]

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        Dataframe -- dataframe with the following columns:
                    "cell_specimen_id": index
                    "cell_roi_id"
                    "height"
                    "image_mask"
                    "mask_image_plane"
                    "max_correction_down"
                    "max_correction_left"
                    "max_correction_right"
                    "max_correction_up"
                    "valid_roi"
                    "width"
                    "x"
                    "y"
    """
    session = get_sdk_session_obj(ophys_experiment_id)
    cell_specimen_table = session.cell_specimen_table
    return cell_specimen_table


def get_sdk_dff_traces(ophys_experiment_id):
    session = get_sdk_session_obj(ophys_experiment_id)
    dff_traces = session.dff_traces
    return dff_traces


def get_sdk_dff_traces_array(ophys_experiment_id):
    dff_traces = get_sdk_dff_traces(ophys_experiment_id)
    dff_traces_array = np.vstack(dff_traces.dff.values)
    return dff_traces_array


def get_sdk_running_speed(ophys_session_id):
    session = get_sdk_session_obj(get_ophys_experiment_id_for_ophys_session_id(ophys_session_id))
    running_speed = session.running_data_df['speed']
    return running_speed


def get_sdk_trials(ophys_session_id):
    session = get_sdk_session_obj(get_ophys_experiment_id_for_ophys_session_id(ophys_session_id))
    trials = session.trials.reset_index()
    return trials


#  FROM LIMS DATABASE


# EXPERIMENT LEVEL


def get_lims_experiment_info(ophys_experiment_id):
    """uses an sqlite query to retrieve ophys experiment information
        from the lims2 database

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        table -- table with the following columns:
                    "ophys_experiment_id":
                    "experiment_workflow_state":
                    "ophys_session_id":
                    "ophys_container_id":
                    "date_of_acquisition":
                    "stage_name_lims":
                    "foraging_id":
                    "mouse_info":
                    "mouse_donor_id":
                    "targeted_structure":
                    "depth":
                    "rig":

    """
    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    # build query
    query = '''
    select

    oe.id as ophys_experiment_id,
    oe.workflow_state,
    oe.ophys_session_id,

    container.visual_behavior_experiment_container_id as ophys_container_id,

    os.date_of_acquisition,
    os.stimulus_name as stage_name_lims,
    os.foraging_id,
    oe.workflow_state as experiment_workflow_state,

    specimens.name as mouse_info,
    specimens.donor_id as mouse_donor_id,
    structures.acronym as targeted_structure,
    imaging_depths.depth,
    equipment.name as rig

    from
    ophys_experiments_visual_behavior_experiment_containers container
    join ophys_experiments oe on oe.id = container.ophys_experiment_id
    join ophys_sessions os on os.id = oe.ophys_session_id
    join specimens on specimens.id = os.specimen_id
    join structures on structures.id = oe.targeted_structure_id
    join imaging_depths on imaging_depths.id = oe.imaging_depth_id
    join equipment on equipment.id = os.equipment_id

    where oe.id = {}'''.format(ophys_experiment_id)

    lims_experiment_info = mixin.select(query)

    return lims_experiment_info


def get_current_segmentation_run_id(ophys_experiment_id):
    """gets the id for the current cell segmentation run for a given experiment.
        Queries LIMS via AllenSDK PostgresQuery function.

    Arguments:
       ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        int -- current cell segmentation run id
    """

    segmentation_run_table = get_lims_cell_segmentation_run_info(ophys_experiment_id)
    current_segmentation_run_id = segmentation_run_table.loc[segmentation_run_table["current"] == True, ["id"][0]][0]
    return current_segmentation_run_id


def get_lims_cell_segmentation_run_info(ophys_experiment_id):
    """Queries LIMS via AllenSDK PostgresQuery function to retrieve information on all segmentations run in the
        ophys_cell_segmenatation_runs table for a given experiment

    Arguments:
         ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        dataframe --  dataframe with the following columns:
                        id {int}:  9 digit segmentation run id
                        run_number {int}: segmentation run number
                        ophys_experiment_id{int}: 9 digit ophys experiment id
                        current{boolean}: True/False
                                    True: most current segmentation run
                                    False: not the most current segmentation run
                        created_at{timestamp}:
                        updated_at{timestamp}:
    """

    mixin = lims_engine
    query = '''
    select *
    FROM ophys_cell_segmentation_runs
    WHERE ophys_experiment_id = {} '''.format(ophys_experiment_id)
    return mixin.select(query)


def get_lims_cell_rois_table(ophys_experiment_id):
    """Queries LIMS via AllenSDK PostgresQuery function to retrieve
        everything in the cell_rois table for a given experiment

    Arguments:
        experiment_id {int} -- 9 digit unique identifier for an ophys experiment

    Returns:
        dataframe -- returns dataframe with the following columns:
            id: a temporary id assigned before cell matching has occured. Same as cell_roi_id in
                the objectlist.txt file

            cell_specimen_id: a permanent id that is assigned after cell matching has occured.
                            this id can be found in multiple experiments in a container if a
                            cell is matched across experiments.
                            experiments that fail qc are not assigned cell_specimen_id s
            ophys_experiment_id:
            x: roi bounding box min x or "bbox_min_x" in objectlist.txt file
            y: roi bounding box min y or "bbox_min_y" in objectlist.txt file
            width:
            height:
            valid_roi: boolean(true/false), whether the roi passes or fails roi filtering
            mask_matrix: boolean mask of just roi
            max_correction_up:
            max_correction_down:
            max_correction_right:
            max_correction_left:
            mask_image_plane:
            ophys_cell_segmentation_run_id:

    """
    # query from AllenSDK

    mixin = lims_engine
    query = '''select cell_rois.*

    from

    ophys_experiments oe
    join cell_rois on oe.id = cell_rois.ophys_experiment_id

    where oe.id = {}'''.format(ophys_experiment_id)
    lims_cell_rois_table = mixin.select(query)
    return lims_cell_rois_table


# CONTAINER  LEVEL


def get_lims_container_info(ophys_container_id):
    """"uses an sqlite query to retrieve container level information
        from the lims2 database. Each row is an experiment within the container.

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
       table -- table with the following columns:
                    "ophys_container_id":
                    "container_workflow_state":
                    "ophys_experiment_id":
                    "ophys_session_id":
                    "stage_name_lims":
                    "foraging_id":
                    "experiment_workflow_state":
                    "mouse_info":
                    "mouse_donor_id":
                    "targeted_structure":
                    "depth":
                    "rig":
                    "date_of_acquisition":
    """
    ophys_container_id = int(ophys_container_id)

    mixin = lims_engine
    # build query
    query = '''
    SELECT
    container.visual_behavior_experiment_container_id as ophys_container_id,
    vbec.workflow_state as container_workflow_state,
    oe.id as ophys_experiment_id,
    oe.ophys_session_id,
    os.stimulus_name as stage_name_lims,
    os.foraging_id,
    oe.workflow_state as experiment_workflow_state,
    specimens.name as mouse_info,
    specimens.donor_id as mouse_donor_id,
    structures.acronym as targeted_structure,
    imaging_depths.depth,
    equipment.name as rig,
    os.date_of_acquisition

    FROM
    ophys_experiments_visual_behavior_experiment_containers container
    join visual_behavior_experiment_containers vbec on vbec.id = container.visual_behavior_experiment_container_id
    join ophys_experiments oe on oe.id = container.ophys_experiment_id
    join ophys_sessions os on os.id = oe.ophys_session_id
    join structures on structures.id = oe.targeted_structure_id
    join imaging_depths on imaging_depths.id = oe.imaging_depth_id
    join specimens on specimens.id = os.specimen_id
    join equipment on equipment.id = os.equipment_id

    where
    container.visual_behavior_experiment_container_id ={}'''.format(ophys_container_id)

    lims_container_info = mixin.select(query)
    return lims_container_info


# FROM LIMS WELL KNOWN FILES


def get_timeseries_ini_wkf_info(ophys_session_id):
    """use SQL and the LIMS well known file system to get the timeseries_XYT.ini file
        for a given ophys session *from a Scientifica rig*

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session id

    Returns:

    """

    QUERY = '''
    SELECT wkf.storage_directory || wkf.filename
    FROM well_known_files wkf
    JOIN well_known_file_types wkft ON wkft.id=wkf.well_known_file_type_id
    JOIN specimens sp ON sp.id=wkf.attachable_id
    JOIN ophys_sessions os ON os.specimen_id=sp.id
    WHERE wkft.name = 'SciVivoMetadata'
    AND wkf.storage_directory LIKE '%ophys_session_{0}%'
    AND os.id = {0}

    '''.format(ophys_session_id)

    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    timeseries_ini_wkf_info = (lims_cursor.fetchall())
    return timeseries_ini_wkf_info


def get_timeseries_ini_location(ophys_session_id):
    """use SQL and the LIMS well known file system to get info for the timeseries_XYT.ini file
        for a given ophys session, and then parses that information to get the filepath

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session id

    Returns:
        filepath -- [description]
    """
    timeseries_ini_wkf_info = get_timeseries_ini_wkf_info(ophys_session_id)
    timeseries_ini_path = timeseries_ini_wkf_info[0]['?column?']  # idk why it's ?column? but it is :(
    timeseries_ini_path = timeseries_ini_path.replace('/allen', '//allen')  # works with windows and linux filepaths
    return timeseries_ini_path


def pmt_gain_from_timeseries_ini(timeseries_ini_path):
    """parses the timeseries ini file and extracts the pmt gain setting

    Arguments:
        timeseries_ini_path {[type]} -- [description]

    Returns:
        int -- int of the pmt gain
    """
    config.read(timeseries_ini_path)
    pmt_gain = int(float(config['_']['PMT.2']))
    return pmt_gain


def get_pmt_gain_for_session(ophys_session_id):
    """finds the timeseries ini file for a given ophys session
        on a Scientifica rig, parses the file and returns the
        pmt gain setting for that session

    Arguments:
        ophys_session_id {int} -- [description]

    Returns:
        int -- pmt gain setting
    """
    try:
        timeseries_ini_path = get_timeseries_ini_location(ophys_session_id)
        pmt_gain = pmt_gain_from_timeseries_ini(timeseries_ini_path)
    except IndexError:
        ophys_experiment_id = get_ophys_experiment_id_for_ophys_session_id(ophys_session_id)
        print("lims query did not return timeseries_XYT.ini location for session_id: " + str(ophys_session_id) + ", experiment_id: " + str(ophys_experiment_id))
        pmt_gain = np.nan
    return pmt_gain


def get_pmt_gain_for_experiment(ophys_experiment_id):
    """finds the timeseries ini file for  the ophys_session_id
        associated with an ophys_experiment_id  from a Scientifica
        rig, parses the file and returns the
        pmt gain setting for that session

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        int -- pmt gain setting
    """
    ophys_session_id = get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    pmt_gain = get_pmt_gain_for_session(ophys_session_id)
    return pmt_gain


def get_motion_corrected_movie_h5_wkf_info(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get the
        "motion_corrected_movie.h5" information for a given
        ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        [type] -- [description]
    """

    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 886523092 AND
     attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    motion_corrected_movie_h5_wkf_info = (lims_cursor.fetchall())
    return motion_corrected_movie_h5_wkf_info


def get_motion_corrected_movie_h5_location(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get info for the
        "motion_corrected_movie.h5" file for a ophys_experiment_id,
        and then parses that information to get the filepath

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        filepath -- [description]
    """
    motion_corrected_movie_h5_wkf_info = get_motion_corrected_movie_h5_wkf_info(ophys_experiment_id)
    motion_corrected_movie_h5_path = motion_corrected_movie_h5_wkf_info[0]['?column?']  # idk why it's ?column? but it is :(
    motion_corrected_movie_h5_path = motion_corrected_movie_h5_path.replace('/allen', '//allen')  # works with windows and linux filepaths
    return motion_corrected_movie_h5_path


def load_motion_corrected_movie(ophys_experiment_id):
    """uses well known file system to get motion_corrected_movie.h5
        filepath and then loads the h5 file with h5py function.
        Gets the motion corrected movie array in the h5 from the only
        datastream/key 'data' and returns it.

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        HDF5 dataset -- 3d array-like  (z, y, x) dimensions
                        z: timeseries/frame number
                        y: single frame y axis
                        x: single frame x axis
    """
    motion_corrected_movie_h5_path = get_motion_corrected_movie_h5_location(ophys_experiment_id)
    motion_corrected_movie_h5 = h5py.File(motion_corrected_movie_h5_path, 'r')
    motion_corrected_movie = motion_corrected_movie_h5['data']

    return motion_corrected_movie


def get_rigid_motion_transform_csv_wkf_info(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get the
        "rigid_motion_transform.csv" information for a given
        ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        [type] -- [description]
    """
    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 514167000 AND
     attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    rigid_motion_transform_csv_wkf_info = (lims_cursor.fetchall())
    return rigid_motion_transform_csv_wkf_info


def get_rigid_motion_transform_csv_location(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get info for the
        rigid_motion_transform.csv" file for a ophys_experiment_id,
        and then parses that information to get the filepath

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        filepath -- [description]
    """
    rigid_motion_transform_csv_wkf_info = get_rigid_motion_transform_csv_wkf_info(ophys_experiment_id)
    rigid_motion_transform_csv_path = rigid_motion_transform_csv_wkf_info[0]['?column?']  # idk why it's ?column? but it is :(
    rigid_motion_transform_csv_path = rigid_motion_transform_csv_path.replace('/allen', '//allen')  # works with windows and linux filepaths
    return rigid_motion_transform_csv_path


def load_rigid_motion_transform_csv(ophys_experiment_id):
    """use SQL and the LIMS well known file system to locate
        and load the rigid_motion_transform.csv file for
        a given ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        dataframe -- dataframe with the following columns:
                        "framenumber":
                                  "x":
                                  "y":
                        "correlation":
                           "kalman_x":
                           "kalman_y":
    """
    rigid_motion_transform_csv_path = get_rigid_motion_transform_csv_location(ophys_experiment_id)
    rigid_motion_transform_df = pd.read_csv(rigid_motion_transform_csv_path)
    return rigid_motion_transform_df


# FROM MTRAIN DATABASE


def get_mtrain_stage_name(dataframe):
    foraging_ids = dataframe['foraging_id'][~pd.isnull(dataframe['foraging_id'])]
    query = """
            SELECT
            stages.name as stage_name,
            bs.id as foraging_id
            FROM behavior_sessions bs
            LEFT JOIN states ON states.id = bs.state_id
            LEFT JOIN stages ON stages.id = states.stage_id
            WHERE bs.id IN ({})
        """.format(",".join(["'{}'".format(x) for x in foraging_ids]))
    mtrain_response = pd.read_sql(query, mtrain_engine.get_connection())
    dataframe = dataframe.merge(mtrain_response, on='foraging_id', how='left')
    dataframe = dataframe.rename(columns={"stage_name": "stage_name_mtrain"})
    return dataframe


def build_container_df():
    '''
    build dataframe with one row per container
    '''

    table = get_filtered_ophys_experiment_table().sort_values(by='date_of_acquisition', ascending=False).reset_index()
    container_ids = table['container_id'].unique()
    list_of_dicts = []
    for container_id in container_ids:
        subset = table.query('container_id == @container_id').sort_values(by='date_of_acquisition', ascending=True).drop_duplicates('ophys_session_id').reset_index()
        temp_dict = {
            'container_id': container_id,
            'container_workflow_state': table.query('container_id == @container_id')['container_workflow_state'].unique()[0],
            'first_acquistion_date': subset['date_of_acquisition'].min().split(' ')[0],
            'project_code': subset['project_code'].unique()[0],
            'driver_line': subset['driver_line'][0],
            'cre_line': subset['cre_line'][0],
            'targeted_structure': subset['targeted_structure'].unique()[0],
            'imaging_depth': subset['imaging_depth'].unique()[0],
            'exposure_number': subset['exposure_number'][0],
            'equipment_name': subset['equipment_name'].unique(),
            'specimen_id': subset['specimen_id'].unique()[0],
            'sex': subset['sex'].unique()[0],
            'age_in_days': subset['age_in_days'].min(),
        }
        for idx, row in subset.iterrows():
            temp_dict.update({'session_{}'.format(idx): '{} {}'.format(row['session_type'], row['ophys_experiment_id'])})

        list_of_dicts.append(temp_dict)

    return pd.DataFrame(list_of_dicts).sort_values(by='container_id', ascending=False)
