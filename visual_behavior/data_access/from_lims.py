import os
import warnings
import from_lims_utilities as utils
from allensdk.internal.api import PostgresQueryMixin


# Accessing Lims Database
try:
    lims_dbname = os.environ["LIMS_DBNAME"]
    lims_user = os.environ["LIMS_USER"]
    lims_host = os.environ["LIMS_HOST"]
    lims_password = os.environ["LIMS_PASSWORD"]
    lims_port = os.environ["LIMS_PORT"]

    lims_engine = PostgresQueryMixin(
        dbname=lims_dbname,
        user=lims_user,
        host=lims_host,
        password=lims_password,
        port=lims_port)

except Exception as e:
    warn_string = 'failed to set up LIMS/mtrain credentials\n{}\n\ninternal AIBS users should set up environment variables appropriately\nfunctions requiring database access will fail'.format(
        e)
    warnings.warn(warn_string)

# building querys
mixin = lims_engine

# ID TYPES

# mouse related IDS


def get_donor_id_for_specimen_id(specimen_id):
    specimen_id = int(specimen_id)
    query = '''
    SELECT donor_id
    FROM specimens
    WHERE id = {}
    '''.format(specimen_id)
    donor_ids = mixin.select(query)
    return donor_ids


def get_specimen_id_for_donor_id(donor_id):
    donor_id = int(donor_id)
    query = '''
    SELECT id
    FROM specimens
    WHERE donor_id = {}
    '''.format(donor_id)
    specimen_ids = mixin.select(query)
    return specimen_ids


# for ophys_experimnt_id
def get_current_segmentation_run_id_for_ophys_experiment_id(ophys_experiment_id):
    """gets the id for the current cell segmentation run for a given experiment.
        Queries LIMS via AllenSDK PostgresQuery function.

    Parameters
    ----------
    ophys_experiment_id : int
        9 digit ophys experiment id

    Returns
    -------
    int
        current cell segmentation run id
    """
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    SELECT
    id

    FROM
    ophys_cell_segmentation_runs

    WHERE
    current = true
    AND ophys_experiment_id = {}
    '''.format(ophys_experiment_id)
    current_run_id = mixin.select(query)
    current_run_id = int(current_run_id["id"][0])
    return current_run_id


def get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id):
    """uses an sqlite to query lims2 database. Gets the ophys_session_id
       for a given ophys_experiment_id from the ophys_experiments table
       in lims.
    """
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    SELECT
    ophys_session_id

    FROM
    ophys_experiments

    WHERE
    id = {}
    '''.format(ophys_experiment_id)
    ophys_session_id = mixin.select(query)
    ophys_session_id = ophys_session_id["ophys_session_id"][0]
    return ophys_session_id


def get_behavior_session_id_for_ophys_experiment_id(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    SELECT
    behavior.id

    FROM
    behavior_sessions behavior

    JOIN
    ophys_experiments oe on oe.ophys_session_id = behavior.ophys_session_id

    WHERE
    oe.id = {}
    '''.format(ophys_experiment_id)
    behavior_session_id = mixin.select(query)
    return behavior_session_id


def get_ophys_container_id_for_ophys_experiment_id(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    SELECT
    visual_behavior_experiment_container_id

    FROM
    ophys_experiments_visual_behavior_experiment_containers

    WHERE
    ophys_experiment_id = {}
    '''.format(ophys_experiment_id)
    ophys_container_id = mixin.select(query)
    ophys_container_id = ophys_container_id.rename(columns={"visual_behavior_experiment_container_id": "ophys_container_id"})
    return ophys_container_id

# def get_supercontainer_id_for_ophys_experiment_id(ophys_experiment_id):


def get_all_ids_for_ophys_experiment_id(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    SELECT
    experiments.id as ophys_experiment_id,
    experiments.ophys_session_id,
    behavior.id as behavior_session_id,
    container.visual_behavior_experiment_container_id as ophys_container_id,
    sessions.visual_behavior_supercontainer_id as supercontainer_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior on behavior.ophys_session_id = experiments.ophys_session_id
    JOIN ophys_experiments_visual_behavior_experiment_containers container on container.ophys_experiment_id = experiments.id
    JOIN ophys_sessions sessions on sessions.id = experiments.ophys_session_id

    WHERE
    experiments.id = {}
    '''.format(ophys_experiment_id)
    all_ids = mixin.select(query)
    return all_ids


# for ophys_session_id

def get_ophys_experiment_ids_for_ophys_session_id(ophys_session_id):
    """uses an sqlite to query lims2 database. Gets the ophys_experiment_id
       for a given ophys_session_id from the ophys_experiments table in lims.

       note: number of experiments per session will vary depending on the
       rig it was collected on and the project it was collected for.
    """
    ophys_session_id = int(ophys_session_id)
    query = '''
    SELECT
    id

    FROM
    ophys_experiments

    WHERE
    ophys_session_id = {}
    '''.format(ophys_session_id)
    ophys_experiment_ids = mixin.select(query)
    ophys_experiment_ids = ophys_experiment_ids.rename(columns={"id": "ophys_experiment_id"})
    return ophys_experiment_ids


def get_behavior_session_id_for_ophys_session_id(ophys_session_id):
    ophys_session_id = int(ophys_session_id)
    query = '''
    SELECT
    id

    FROM
    behavior_sessions

    WHERE
    ophys_session_id = {}
    '''.format(ophys_session_id)
    behavior_session_id = mixin.select(query)
    behavior_session_id = behavior_session_id.rename(columns={"id": "behavior_session_id"})
    return behavior_session_id


def get_ophys_container_ids_for_ophys_session_id(ophys_session_id):
    ophys_session_id = int(ophys_session_id)
    query = '''
    SELECT
    container.visual_behavior_experiment_container_id

    FROM
    ophys_experiments_visual_behavior_experiment_containers container

    JOIN
    ophys_experiments oe on oe.id = container.ophys_experiment_id

    WHERE
    oe.ophys_session_id = {}
    '''.format(ophys_session_id)
    ophys_container_ids = mixin.select(query)
    return ophys_container_ids


def get_supercontainer_id_for_ophys_session_id(ophys_session_id):
    ophys_session_id = int(ophys_session_id)
    query = '''
    SELECT
    visual_behavior_supercontainer_id

    FROM
    ophys_sessions

    WHERE id = {}
    '''.format(ophys_session_id)
    supercontainer_id = mixin.select(query)
    return supercontainer_id


# def get_all_ids_for_ophys_session_id(ophys_session_id):

# for behavior_session_id

def get_ophys_experiment_ids_for_behavior_session_id(behavior_session_id):
    behavior_session_id = int(behavior_session_id)
    query = '''
    SELECT
    oe.id

    FROM
    behavior_sessions behavior

    JOIN
    ophys_experiments oe on oe.ophys_session_id = behavior.ophys_session_id

    WHERE
    behavior.id = {}
    '''.format(behavior_session_id)
    ophys_experiment_ids = mixin.select(query)
    return ophys_experiment_ids


def get_ophys_session_id_for_behavior_session_id(behavior_session_id):
    """uses an sqlite to query lims2 database. Gets the ophys_session_id
       for a given behavior_session_id from the behavior_sessions table in
       lims.

       note: not all behavior_sessions have associated ophys_sessions. Only
       behavior_sessions where ophys imaging took place will have
       ophys_session_id's
    """
    behavior_session_id = int(behavior_session_id)
    query = '''
    SELECT
    ophys_session_id

    FROM
    behavior_sessions

    WHERE id = {}'''.format(behavior_session_id)
    ophys_session_id = mixin.select(query)
    return ophys_session_id

# def get_ophys_container_id_for_behavior_session_id(behavior_session_id):
# def get_all_ids_for_ophys_behavior_session_id(ophys_behavior_session_id):
# def get_supercontainer_id_for_behavior_session_id(ophys_behavior_session_id):

# for ophys_container_id


def get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id):
    ophys_container_id = int(ophys_container_id)
    query = '''
    SELECT
    ophys_experiment_id

    FROM
    ophys_experiments_visual_behavior_experiment_containers

    WHERE
    visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)
    ophys_experiment_ids = mixin.select(query)
    return ophys_experiment_ids


def get_ophys_session_ids_for_ophys_container_id(ophys_container_id):
    ophys_container_id = int(ophys_container_id)
    query = '''
    SELECT
    oe.ophys_session_id

    FROM
    ophys_experiments_visual_behavior_experiment_containers container

    JOIN
    ophys_experiments oe on oe.id = container.ophys_experiment_id

    WHERE
    container.visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)
    ophys_session_ids = mixin.select(query)
    return ophys_session_ids


# def get_all_ids_for_ophys_container_id(ophys_container_id):
# def get_supercontainer_id_for_ophys_container_id(ophys_container_id):

# for supercontainer_id
# def get_ophys_experiment_ids_for_supercontainer_id(ophys_supercontainer_id):
# def get_ophys_session_ids_for_supercontainer_id(ophys_supercontainer_id):
# def get_behavior_session_id_for_supercontainer_id(behavior_session_id):
# def get_ophys_container_ids_for_supercontainer_id(ophys_supercontainer_id):


### WELL KNOWN FILE FILEPATHS ###

def get_timeseries_ini_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the 
       timeseries_XYT.ini file for a given ophys session.
       Notes: only Scientifca microscopes prodice timeseries.ini files

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session id

    Returns:

    """
    ophys_session_id = int(ophys_session_id)
    mixin = lims_engine
    mixin = lims_engine
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
    RealDict_object = mixin.select(query)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_dff_traces_filepath(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    # build query
    query = '''
    SELECT wkf.storage_directory || wkf.filename AS dff_file
    FROM ophys_experiments oe
    JOIN well_known_files wkf ON wkf.attachable_id = oe.id
    JOIN well_known_file_types wkft
    ON wkft.id = wkf.well_known_file_type_id
    WHERE wkft.name = 'OphysDffTraceFile'
    AND oe.id = {}'''.format(ophys_experiment_id)
    RealDict_object = mixin.select(query)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_rigid_motion_transform_filepath(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    # build query
    query = '''
    SELECT
    wkf.storage_directory || wkf.filename AS transform_file

    FROM
    ophys_experiments oe

    JOIN well_known_files wkf
    ON wkf.attachable_id = oe.id

    JOIN well_known_file_types wkft
    ON wkft.id = wkf.well_known_file_type_id

    WHERE
    w`kf.attachable_type = 'OphysExperiment'
    AND
    wkft.name = 'OphysMotionXyOffsetData'
    AND oe.id = {}
    '''.format(ophys_experiment_id)
    RealDict_object = mixin.select(query)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_objectlist_filepath(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get the location 
       information for the objectlist.txt file for a given cell 
       segmentation run

    Arguments:
        segmentation_run_id {int} -- 9 digit segmentation run id

    Returns:
        list -- list with storage directory and filename
    """
    current_segmentation_run_id = int(get_current_segmentation_run_id_for_ophys_experiment_id(ophys_experiment_id))
    mixin = lims_engine
    query = '''
    SELECT wkf.storage_directory, wkf.filename
    FROM well_known_files wkf
    JOIN well_known_file_types wkft on wkf.well_known_file_type_id = wkft.id
    JOIN ophys_cell_segmentation_runs ocsr on wkf.attachable_id = ocsr.id
    WHERE wkft.name = 'OphysSegmentationObjects'
    AND wkf.attachable_type = 'OphysCellSegmentationRun'
    AND ocsr.id = {0}'''.format(current_segmentation_run_id)
    RealDict_object = mixin.select(query)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_demixed_traces_filepath(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    query = """
    SELECT wkf.storage_directory || wkf.filename AS demix_file
    FROM ophys_experiments oe
    JOIN well_known_files wkf ON wkf.attachable_id = oe.id
    JOIN well_known_file_types wkft
    ON wkft.id = wkf.well_known_file_type_id
    WHERE wkf.attachable_type = 'OphysExperiment'
    AND wkft.name = 'DemixedTracesFile'
    AND oe.id = {};""".format(ophys_experiment_id)
    RealDict_object = mixin.select(query)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_neuropil_traces_filepath(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    query = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 514173078 AND
    attachable_id = {0}'''.format(ophys_experiment_id)
    RealDict_object = mixin.select(query)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_motion_corrected_movie_filepath(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get the
        "motion_corrected_movie.h5" information for a given
        ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        [type] -- [description]
    """
    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 886523092 AND
     attachable_id = {0}
    '''.format(ophys_experiment_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_extracted_trace_filepath(ophys_experiment_id):
    mixin = lims_engine
    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 486797213 AND
    attachable_id = {0}
    '''.format(ophys_experiment_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_session_pkl_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        session pkl file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    mixin = lims_engine
    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 610487715 AND
     attachable_id = {0}

    '''.format(ophys_session_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_session_h5_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        session h5 file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    mixin = lims_engine
    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 610487713 AND
     attachable_id = {0}

    '''.format(ophys_session_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_behavior_avi_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        video-0 avi (behavior video) file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    mixin = lims_engine
    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 695808672 AND
    attachable_id = {0}

    '''.format(ophys_session_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_behavior_h5_filepath(ophys_session_id):
    avi_filepath = get_behavior_avi_filepath(ophys_session_id)
    h5_filepath = avi_filepath[:-3] + "h5"
    return h5_filepath


def get_eye_tracking_avi_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        video-1 avi (eyetracking video) file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    mixin = lims_engine
    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 695808172 AND
    attachable_id = {0}

    '''.format(ophys_session_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


