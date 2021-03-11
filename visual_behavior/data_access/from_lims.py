import os
import h5py
import warnings
import numpy as np
import pandas as pd


from allensdk.internal.api import PostgresQueryMixin
from visual_behavior.data_access import from_lims_utilities as utils


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
    warn_string = 'failed to set up LIMS/mtrain credentials\n{}\n\ninternal AIBS users should set up \
                   environment variables appropriately\nfunctions requiring database access will fail'.format(e)
    warnings.warn(warn_string)

# building querys
mixin = lims_engine

### ID TYPES ###      # noqa: E266

# mouse related IDS


def get_donor_id_for_specimen_id(specimen_id):
    specimen_id = int(specimen_id)
    query = '''
    SELECT
    donor_id

    FROM
    specimens

    WHERE
    id = {}
    '''.format(specimen_id)
    donor_ids = mixin.select(query)
    return donor_ids


def get_specimen_id_for_donor_id(donor_id):
    donor_id = int(donor_id)
    query = '''
    SELECT
    id

    FROM
    specimens

    WHERE
    donor_id = {}
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
    behavior.id as behavior_session_id

    FROM
    behavior_sessions behavior

    JOIN
    ophys_experiments oe on oe.ophys_session_id = behavior.ophys_session_id

    WHERE
    oe.id = {}
    '''.format(ophys_experiment_id)
    behavior_session_id = mixin.select(query)
    behavior_session_id = behavior_session_id["behavior_session_id"][0]
    return behavior_session_id


def get_ophys_container_id_for_ophys_experiment_id(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    SELECT
    visual_behavior_experiment_container_id as ophys_container_id

    FROM
    ophys_experiments_visual_behavior_experiment_containers

    WHERE
    ophys_experiment_id = {}
    '''.format(ophys_experiment_id)
    ophys_container_id = mixin.select(query)
    ophys_container_id = ophys_container_id["ophys_container_id"][0]
    return ophys_container_id


def get_super_container_id_for_ophys_experiment_id(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    SELECT
    sessions.visual_behavior_supercontainer_id as supercontainer_id

    FROM
    ophys_experiments experiments

    JOIN ophys_sessions sessions
    ON sessions.id = experiments.ophys_session_id

    WHERE
    experiments.id = {}
    '''.format(ophys_experiment_id)
    supercontainer_id = mixin.select(query)
    supercontainer_id = supercontainer_id["supercontainer_id"][0]
    return supercontainer_id


def get_all_ids_for_ophys_experiment_id(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    SELECT
    experiments.id as ophys_experiment_id,
    experiments.ophys_session_id,
    behavior.id as behavior_session_id,
    container.visual_behavior_experiment_container_id as ophys_container_id,
    sessions.visual_behavior_supercontainer_id as super_container_id

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


def get_general_info_for_ophys_experiment_id(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    oe.ophys_session_id,
    bs.id AS behavior_session_id,
    os.foraging_id,
    containers.id AS ophys_container_id,
    os.visual_behavior_supercontainer_id AS supercontainer_id,

    oe.workflow_state AS experiment_workflow_state,
    os.workflow_state AS session_workflow_state,
    containers.workflow_state AS container_workflow_state,

    os.specimen_id,
    specimens.donor_id,
    specimens.name AS specimen_name,

    os.date_of_acquisition,
    os.stimulus_name AS session_type,
    structures.acronym AS targeted_structure,
    imaging_depths.depth,
    equipment.name AS rig

    FROM
    ophys_experiments oe

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id
    JOIN behavior_sessions bs
    ON bs.foraging_id = os.foraging_id
    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oe.id = oevbec.ophys_experiment_id
    JOIN visual_behavior_experiment_containers containers
    ON containers.id = oevbec.visual_behavior_experiment_container_id
    JOIN specimens ON specimens.id = os.specimen_id
    JOIN structures ON structures.id = oe.targeted_structure_id
    JOIN imaging_depths ON imaging_depths.id = oe.imaging_depth_id
    JOIN equipment ON equipment.id = os.equipment_id

    WHERE
    oe.id ={}
    '''.format(ophys_experiment_id)
    general_info = mixin.select(query)
    return general_info

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

    FROM ophys_experiments

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
    id AS behavior_session_id

    FROM behavior_sessions

    WHERE
    ophys_session_id = {}
    '''.format(ophys_session_id)
    behavior_session_id = mixin.select(query)
    behavior_session_id = behavior_session_id["behavior_session_id"][0]
    return behavior_session_id


def get_ophys_container_ids_for_ophys_session_id(ophys_session_id):
    ophys_session_id = int(ophys_session_id)
    query = '''
    SELECT
    container.visual_behavior_experiment_container_id
    AS ophys_container_id

    FROM ophys_experiments_visual_behavior_experiment_containers container
    JOIN ophys_experiments oe ON oe.id = container.ophys_experiment_id

    WHERE
    oe.ophys_session_id = {}
    '''.format(ophys_session_id)
    ophys_container_ids = mixin.select(query)
    return ophys_container_ids


def get_super_container_id_for_ophys_session_id(ophys_session_id):
    ophys_session_id = int(ophys_session_id)
    query = '''
    SELECT
    visual_behavior_supercontainer_id
    AS super_container_id

    FROM
    ophys_sessions

    WHERE id = {}
    '''.format(ophys_session_id)
    super_container_id = mixin.select(query)
    super_container_id = super_container_id["super_container_id"][0]
    return super_container_id


def get_all_ids_for_ophys_session_id(ophys_session_id):
    ophys_session_id = int(ophys_session_id)
    query = '''
    SELECT
    experiments.id as ophys_experiment_id,
    experiments.ophys_session_id,
    behavior.id as behavior_session_id,
    container.visual_behavior_experiment_container_id as ophys_container_id,
    sessions.visual_behavior_supercontainer_id as super_container_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = experiments.ophys_session_id
    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id
    JOIN ophys_sessions sessions on sessions.id = experiments.ophys_session_id

    WHERE
    experiments.ophys_session_id = {}
    '''.format(ophys_session_id)
    all_ids = mixin.select(query)
    return all_ids


def get_general_info_for_ophys_session_id(ophys_session_id):
    ophys_session_id = int(ophys_session_id)
    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    oe.ophys_session_id,
    bs.id AS behavior_session_id,
    os.foraging_id,
    containers.id AS ophys_container_id,
    os.visual_behavior_supercontainer_id AS supercontainer_id,

    oe.workflow_state AS experiment_workflow_state,
    os.workflow_state AS session_workflow_state,
    containers.workflow_state AS container_workflow_state,

    os.specimen_id,
    specimens.donor_id,
    specimens.name AS specimen_name,

    os.date_of_acquisition,
    os.stimulus_name AS session_type,
    structures.acronym AS targeted_structure,
    imaging_depths.depth,
    equipment.name AS rig

    FROM
    ophys_experiments oe

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id

    JOIN behavior_sessions bs
    ON bs.foraging_id = os.foraging_id

    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oe.id = oevbec.ophys_experiment_id

    JOIN visual_behavior_experiment_containers containers
    ON containers.id = oevbec.visual_behavior_experiment_container_id

    JOIN specimens ON specimens.id = os.specimen_id
    JOIN structures ON structures.id = oe.targeted_structure_id
    JOIN imaging_depths ON imaging_depths.id = oe.imaging_depth_id
    JOIN equipment ON equipment.id = os.equipment_id

    WHERE
    os.id = {}
    '''.format(ophys_session_id)
    general_info = mixin.select(query)
    return general_info


# for behavior_session_id


def get_ophys_experiment_ids_for_behavior_session_id(behavior_session_id):
    behavior_session_id = int(behavior_session_id)
    query = '''
    SELECT
    oe.id
    AS ophys_experiment_id

    FROM behavior_sessions behavior

    JOIN ophys_experiments oe
    ON oe.ophys_session_id = behavior.ophys_session_id

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

    FROM behavior_sessions

    WHERE id = {}'''.format(behavior_session_id)
    ophys_session_id = mixin.select(query)
    ophys_session_id = ophys_session_id["ophys_session_id"][0]
    return ophys_session_id


def get_ophys_container_ids_for_behavior_session_id(behavior_session_id):
    behavior_session_id = int(behavior_session_id)
    query = '''
    SELECT
    container.visual_behavior_experiment_container_id
    AS ophys_container_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = experiments.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id

    JOIN ophys_sessions sessions on sessions.id = experiments.ophys_session_id

    WHERE
    behavior.id = {}
    '''.format(behavior_session_id)
    container_id = mixin.select(query)
    return container_id


def get_super_container_id_for_behavior_session_id(behavior_session_id):
    behavior_session_id = int(behavior_session_id)
    query = '''
    SELECT
    sessions.visual_behavior_supercontainer_id AS super_container_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = experiments.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id

    JOIN ophys_sessions sessions on sessions.id = experiments.ophys_session_id

    WHERE
    behavior.id = {}
    '''.format(behavior_session_id)
    super_container_id = mixin.select(query)
    super_container_id = super_container_id["super_container_id"][0]
    return super_container_id


def get_all_ids_for_behavior_session_id(behavior_session_id):
    behavior_session_id = int(behavior_session_id)
    query = '''
    SELECT
    experiments.id AS ophys_experiment_id,
    experiments.ophys_session_id,
    behavior.id AS behavior_session_id,
    container.visual_behavior_experiment_container_id AS ophys_container_id,
    sessions.visual_behavior_supercontainer_id AS super_container_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = experiments.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id

    JOIN ophys_sessions sessions on sessions.id = experiments.ophys_session_id

    WHERE
    behavior.id = {}
    '''.format(behavior_session_id)
    all_ids = mixin.select(query)
    return all_ids


def get_general_info_for_behavior_session_id(behavior_session_id):
    behavior_session_id = int(behavior_session_id)
    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    oe.ophys_session_id,
    bs.id AS behavior_session_id,
    os.foraging_id,
    containers.id AS ophys_container_id,
    os.visual_behavior_supercontainer_id AS supercontainer_id,

    oe.workflow_state AS experiment_workflow_state,
    os.workflow_state AS session_workflow_state,
    containers.workflow_state AS container_workflow_state,

    os.specimen_id,
    specimens.donor_id,
    specimens.name AS specimen_name,

    os.date_of_acquisition,
    os.stimulus_name AS session_type,
    structures.acronym AS targeted_structure,
    imaging_depths.depth,
    equipment.name AS rig

    FROM
    ophys_experiments oe

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id

    JOIN behavior_sessions bs
    ON bs.foraging_id = os.foraging_id

    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oe.id = oevbec.ophys_experiment_id

    JOIN visual_behavior_experiment_containers containers
    ON containers.id = oevbec.visual_behavior_experiment_container_id

    JOIN specimens ON specimens.id = os.specimen_id
    JOIN structures ON structures.id = oe.targeted_structure_id
    JOIN imaging_depths ON imaging_depths.id = oe.imaging_depth_id
    JOIN equipment ON equipment.id = os.equipment_id

    WHERE
    bs.id = {}
    '''.format(behavior_session_id)
    general_info = mixin.select(query)
    return general_info

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

    JOIN ophys_experiments oe on oe.id = container.ophys_experiment_id

    WHERE
    container.visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)
    ophys_session_ids = mixin.select(query)
    return ophys_session_ids


def get_behavior_session_ids_for_ophys_container_id(ophys_container_id):
    ophys_container_id = int(ophys_container_id)
    query = '''
    SELECT
    behavior.id
    AS behavior_session_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = experiments.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id

    JOIN ophys_sessions sessions on sessions.id = experiments.ophys_session_id

    WHERE
    container.visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)
    behavior_session_id = mixin.select(query)
    return behavior_session_id


def get_supercontainer_id_for_ophys_container_id(ophys_container_id):
    ophys_container_id = int(ophys_container_id)
    query = '''
    SELECT
    sessions.visual_behavior_supercontainer_id as super_container_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = experiments.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id

    JOIN ophys_sessions sessions on sessions.id = experiments.ophys_session_id

    WHERE
    container.visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)
    super_container_id = mixin.select(query)
    super_container_id = int(super_container_id['super_container_id'][0])
    return super_container_id


def get_all_ids_for_ophys_container_id(ophys_container_id):
    ophys_container_id = int(ophys_container_id)
    query = '''
    SELECT
    experiments.id AS ophys_experiment_id,
    experiments.ophys_session_id,
    behavior.id AS behavior_session_id,
    container.visual_behavior_experiment_container_id AS ophys_container_id,
    sessions.visual_behavior_supercontainer_id AS super_container_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = experiments.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id

    JOIN ophys_sessions sessions ON sessions.id = experiments.ophys_session_id

    WHERE
    container.visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)
    all_ids = mixin.select(query)
    return all_ids


def get_general_info_for_ophys_container_id(ophys_container_id):
    ophys_container_id = int(ophys_container_id)
    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    oe.ophys_session_id,
    bs.id AS behavior_session_id,
    os.foraging_id,
    containers.id AS ophys_container_id,
    os.visual_behavior_supercontainer_id AS supercontainer_id,

    oe.workflow_state AS experiment_workflow_state,
    os.workflow_state AS session_workflow_state,
    containers.workflow_state AS container_workflow_state,

    os.specimen_id,
    specimens.donor_id,
    specimens.name AS specimen_name,

    os.date_of_acquisition,
    os.stimulus_name AS session_type,
    structures.acronym AS targeted_structure,
    imaging_depths.depth,
    equipment.name AS rig

    FROM
    ophys_experiments oe

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id
    JOIN behavior_sessions bs
    ON bs.foraging_id = os.foraging_id
    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oe.id = oevbec.ophys_experiment_id
    JOIN visual_behavior_experiment_containers containers
    ON containers.id = oevbec.visual_behavior_experiment_container_id
    JOIN specimens ON specimens.id = os.specimen_id
    JOIN structures ON structures.id = oe.targeted_structure_id
    JOIN imaging_depths ON imaging_depths.id = oe.imaging_depth_id
    JOIN equipment ON equipment.id = os.equipment_id

    WHERE
    containers.id = {}
    '''.format(ophys_container_id)
    general_info = mixin.select(query)
    return general_info

# for supercontainer_id


def get_ophys_experiment_ids_for_supercontainer_id(supercontainer_id):
    supercontainer_id = int(supercontainer_id)
    query = '''
    SELECT
    experiments.id AS ophys_experiment_id

    FROM
    ophys_experiments experiments

    JOIN ophys_sessions sessions ON sessions.id = experiments.ophys_session_id

    WHERE
    sessions.visual_behavior_supercontainer_id = {}
    '''.format(supercontainer_id)
    ophys_experiment_ids = mixin.select(query)
    return ophys_experiment_ids


def get_ophys_session_ids_for_supercontainer_id(supercontainer_id):
    supercontainer_id = int(supercontainer_id)
    query = '''
    SELECT
    id AS ophys_session_id

    FROM
    ophys_sessions

    WHERE
    visual_behavior_supercontainer_id = {}
    '''.format(supercontainer_id)
    ophys_session_ids = mixin.select(query)
    return ophys_session_ids


def get_behavior_session_id_for_supercontainer_id(supercontainer_id):
    supercontainer_id = int(supercontainer_id)
    query = '''
    SELECT
    behavior.id AS behavior_session_id

    FROM
    ophys_sessions sessions

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = sessions.id

    WHERE
    sessions.visual_behavior_supercontainer_id = {}
    '''.format(supercontainer_id)
    behavior_session_ids = mixin.select(query)
    return behavior_session_ids


def get_ophys_container_ids_for_supercontainer_id(supercontainer_id):
    supercontainer_id = int(supercontainer_id)
    query = '''
    SELECT
    container.visual_behavior_experiment_container_id
    AS ophys_container_id

    FROM
    ophys_experiments experiments

    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id

    JOIN ophys_sessions sessions ON sessions.id = experiments.ophys_session_id

    WHERE
    sessions.visual_behavior_supercontainer_id = {}
    '''.format(supercontainer_id)
    ophys_container_ids = mixin.select(query)
    return ophys_container_ids


def get_all_ids_for_supercontainer_id(supercontainer_id):
    supercontainer_id = int(supercontainer_id)
    query = '''
    SELECT
    experiments.id as ophys_experiment_id,
    experiments.ophys_session_id,
    behavior.id as behavior_session_id,
    container.visual_behavior_experiment_container_id as ophys_container_id,
    sessions.visual_behavior_supercontainer_id as super_container_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = experiments.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id

    JOIN ophys_sessions sessions on sessions.id = experiments.ophys_session_id

    WHERE
    sessions.visual_behavior_supercontainer_id = {}
    '''.format(supercontainer_id)
    all_ids = mixin.select(query)
    return all_ids


def get_general_info_for_supercontainer_id(supercontainer_id):
    supercontainer_id = int(supercontainer_id)
    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    oe.ophys_session_id,
    bs.id AS behavior_session_id,
    os.foraging_id,
    containers.id AS ophys_container_id,
    os.visual_behavior_supercontainer_id AS supercontainer_id,

    oe.workflow_state AS experiment_workflow_state,
    os.workflow_state AS session_workflow_state,
    containers.workflow_state AS container_workflow_state,

    os.specimen_id,
    specimens.donor_id,
    specimens.name AS specimen_name,

    os.date_of_acquisition,
    os.stimulus_name AS session_type,
    structures.acronym AS targeted_structure,
    imaging_depths.depth,
    equipment.name AS rig

    FROM
    ophys_experiments oe

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id
    JOIN behavior_sessions bs
    ON bs.foraging_id = os.foraging_id
    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oe.id = oevbec.ophys_experiment_id
    JOIN visual_behavior_experiment_containers containers
    ON containers.id = oevbec.visual_behavior_experiment_container_id
    JOIN specimens ON specimens.id = os.specimen_id
    JOIN structures ON structures.id = oe.targeted_structure_id
    JOIN imaging_depths ON imaging_depths.id = oe.imaging_depth_id
    JOIN equipment ON equipment.id = os.equipment_id

    WHERE
    os.visual_behavior_supercontainer_id = {}
    '''.format(supercontainer_id)
    general_info = mixin.select(query)
    return general_info


### TABLES ###    # noqa: E266

def get_cell_segmentation_runs_table(ophys_experiment_id):
    """Queries LIMS via AllenSDK PostgresQuery function to retrieve
        information on all segmentations run in the
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
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    select
    *

    FROM
    ophys_cell_segmentation_runs

    WHERE
    ophys_experiment_id = {}
    '''.format(ophys_experiment_id)
    return mixin.select(query)


def get_ophys_experiments_table(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    oe.workflow_state AS experiment_workflow_state,
    oe.storage_directory,
    oe.ophys_session_id,
    structures.acronym AS targeted_structure,
    imaging_depths.depth AS imaging_depth,
    oe.calculated_depth,
    container.visual_behavior_experiment_container_id AS ophys_container_id,
    oe.raw_movie_number_of_frames,
    oe.raw_movie_width,
    oe.raw_movie_height,
    oe.movie_frame_rate_hz,
    oe.ophys_imaging_plane_group_id

    FROM
    ophys_experiments_visual_behavior_experiment_containers container
    JOIN ophys_experiments oe ON oe.id = container.ophys_experiment_id
    JOIN structures ON structures.id = oe.targeted_structure_id
    JOIN imaging_depths ON imaging_depths.id = oe.imaging_depth_id

    WHERE
    oe.id = {}
    '''.format(ophys_experiment_id)
    ophys_experiments_table = mixin.select(query)
    return ophys_experiments_table


def get_ophys_sessions_table(ophys_session_id):
    ophys_session_id = int(ophys_session_id)
    query = '''
    SELECT
    os.id as ophys_session_id,
    os.storage_directory,
    os.workflow_state as session_workflow_state,
    os.specimen_id,
    os.isi_experiment_id,
    os.parent_session_id,
    os.date_of_acquisition,
    equipment.name as equipment_name,
    projects.code as project,
    os.stimulus_name as session_type,
    os.foraging_id,
    os.stim_delay,
    os.visual_behavior_supercontainer_id as supercontainer_id

    FROM
    ophys_sessions os
    JOIN equipment ON equipment.id = os.equipment_id
    JOIN projects ON projects.id = os.project_id

    WHERE
    os.id = {}
    '''.format(ophys_session_id)
    ophys_sessions_table = mixin.select(query)
    return ophys_sessions_table


def get_behavior_sessions_table(behavior_session_id):
    behavior_session_id = int(behavior_session_id)
    query = '''
    SELECT
    bs.id as behavior_session_id,
    bs.ophys_session_id,
    bs.behavior_training_id,
    bs.foraging_id,
    bs.donor_id,
    equipment.name as equipment_name,
    bs.created_at,
    bs.date_of_acquisition

    FROM
    behavior_sessions bs
    JOIN equipment ON equipment.id = bs.equipment_id

    WHERE
    bs.id = {}
    '''.format(behavior_session_id)
    behavior_sessions_table = mixin.select(query)
    return behavior_sessions_table


def get_visual_behavior_experiment_containers_table(ophys_container_id):
    ophys_container_id = int(ophys_container_id)
    query = '''
    SELECT
    containers.id AS container_id,
    projects.code AS project,
    containers.specimen_id,
    containers.storage_directory,
    containers.workflow_state AS container_workflow_state

    FROM
    visual_behavior_experiment_containers as containers
    JOIN projects on projects.id = containers.project_id
    WHERE
    containers.id = {}
    '''.format(ophys_container_id)
    visual_behavior_experiment_containers_table = mixin.select(query)
    return visual_behavior_experiment_containers_table


### ROI ###       # noqa: E266


def get_cell_exclusion_labels(ophys_experiment_id):
    mixin = lims_engine
    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    cr.id AS cell_roi_id,
    rel.name AS exclusion_label

    FROM ophys_experiments oe

    JOIN ophys_cell_segmentation_runs ocsr
    ON ocsr.ophys_experiment_id=oe.id 
    AND ocsr.current = 't'

    JOIN cell_rois cr ON cr.ophys_cell_segmentation_run_id=ocsr.id
    JOIN cell_rois_roi_exclusion_labels crrel ON crrel.cell_roi_id=cr.id
    JOIN roi_exclusion_labels rel ON rel.id=crrel.roi_exclusion_label_id

    WHERE
    oe.id = {}
    '''.format(ophys_experiment_id)
    return mixin.select(query)


### WELL KNOWN FILE FILEPATHS ###      # noqa: E266

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
    query = '''
    SELECT
    wkf.storage_directory || wkf.filename

    FROM
    well_known_files wkf

    JOIN well_known_file_types wkft ON wkft.id=wkf.well_known_file_type_id
    JOIN specimens sp ON sp.id=wkf.attachable_id
    JOIN ophys_sessions os ON os.specimen_id=sp.id

    WHERE
    wkft.name = 'SciVivoMetadata'
    AND wkf.storage_directory LIKE '%ophys_session_{0}%'
    AND os.id = {0}
    '''.format(ophys_session_id)

    try:
        RealDict_object = mixin.select(query)
        filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
        return filepath
    except KeyError:
        print("Must be collected on a scientifica microscope to have timeseries ini. \n Check that equipment_name = CAM2P.2, CAM2P.3, CAM2P.4, CAM2P.5 or CAM2P.6")


def get_dff_traces_filepath(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    # build query
    query = '''
    SELECT
    wkf.storage_directory || wkf.filename AS dff_file

    FROM
    ophys_experiments oe

    JOIN well_known_files wkf ON wkf.attachable_id = oe.id
    JOIN well_known_file_types wkft ON wkft.id = wkf.well_known_file_type_id

    WHERE
    wkft.name = 'OphysDffTraceFile'
    AND oe.id = {}
    '''.format(ophys_experiment_id)
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

    JOIN well_known_files wkf ON wkf.attachable_id = oe.id
    JOIN well_known_file_types wkft ON wkft.id = wkf.well_known_file_type_id

    WHERE
    w`kf.attachable_type = 'OphysExperiment'
    AND wkft.name = 'OphysMotionXyOffsetData'
    AND oe.id = {}
    '''.format(ophys_experiment_id)
    RealDict_object = mixin.select(query)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_objectlist_filepath(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get the
       location information for the objectlist.txt file for a
       given cell segmentation run

    Arguments:
        segmentation_run_id {int} -- 9 digit segmentation run id

    Returns:
        list -- list with storage directory and filename
    """
    current_segmentation_run_id = \
    int(get_current_segmentation_run_id_for_ophys_experiment_id(ophys_experiment_id))
    
    mixin = lims_engine
    query = '''
    SELECT
    wkf.storage_directory, wkf.filename

    FROM
    well_known_files wkf

    JOIN well_known_file_types wkft ON wkf.well_known_file_type_id = wkft.id
    JOIN ophys_cell_segmentation_runs ocsr ON wkf.attachable_id = ocsr.id

    WHERE
    wkft.name = 'OphysSegmentationObjects'
    AND wkf.attachable_type = 'OphysCellSegmentationRun'
    AND ocsr.id = {0}
    '''.format(current_segmentation_run_id)
    RealDict_object = mixin.select(query)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_demixed_traces_filepath(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    query = """
    SELECT
    wkf.storage_directory || wkf.filename AS demix_file

    FROM
    ophys_experiments oe

    JOIN well_known_files wkf ON wkf.attachable_id = oe.id
    JOIN well_known_file_types wkft ON wkft.id = wkf.well_known_file_type_id

    WHERE wkf.attachable_type = 'OphysExperiment'
    AND wkft.name = 'DemixedTracesFile'
    AND oe.id = {};
    """.format(ophys_experiment_id)
    RealDict_object = mixin.select(query)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_neuropil_traces_filepath(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    query = '''
    SELECT
    storage_directory || filename

    FROM
    well_known_files

    WHERE
    well_known_file_type_id = 514173078
    AND attachable_id = {0}
    '''.format(ophys_experiment_id)
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
     SELECT
     storage_directory || filename

     FROM
     well_known_files

     WHERE well_known_file_type_id = 886523092
     AND attachable_id = {0}
    '''.format(ophys_experiment_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_extracted_trace_filepath(ophys_experiment_id):
    mixin = lims_engine
    QUERY = '''
    SELECT
    storage_directory || filename

    FROM
    well_known_files

    WHERE
    well_known_file_type_id = 486797213
    AND attachable_id = {0}
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
     SELECT
     storage_directory || filename

     FROM
     well_known_files

     WHERE
     well_known_file_type_id = 610487715
     AND attachable_id = {0}
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
     SELECT
     storage_directory || filename

     FROM
     well_known_files

     WHERE
     well_known_file_type_id = 610487713
     AND attachable_id = {0}
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
    SELECT
    storage_directory || filename

    FROM
    well_known_files

    WHERE
    well_known_file_type_id = 695808672
    AND attachable_id = {0}
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
    SELECT
    storage_directory || filename

    FROM
    well_known_files

    WHERE
    well_known_file_type_id = 695808172
    AND attachable_id = {0}
    '''.format(ophys_session_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_ellipse_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        ellipse.h5 file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    mixin = lims_engine
    QUERY = '''
    SELECT
    storage_directory || filename

    FROM well_known_files

    WHERE
    well_known_file_type_id = 914623492
    AND attachable_id = {0}
    '''.format(ophys_session_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_platform_json_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        platform.json file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    mixin = lims_engine
    QUERY = '''
    SELECT
    storage_directory || filename

    FROM well_known_files

    WHERE
    well_known_file_type_id = 746251277
    AND attachable_id = {0}
    '''.format(ophys_session_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_screen_mapping_h5_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        screen mapping .h5 file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    mixin = lims_engine
    QUERY = '''
     SELECT
     storage_directory || filename

     FROM well_known_files

     WHERE
     well_known_file_type_id = 916857994
     AND attachable_id = {0}
    '''.format(ophys_session_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


def get_deepcut_h5_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        screen mapping .h5 file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    QUERY = '''
     SELECT
     storage_directory || filename

     FROM
     well_known_files

     WHERE
     well_known_file_type_id = 990460508
     AND attachable_id = {0}
    '''.format(ophys_session_id)
    RealDict_object = mixin.select(QUERY)
    filepath = utils.get_filepath_from_wkf_realdict_object(RealDict_object)
    return filepath


### WELL KNOWN FILES ###      # noqa: E303, E266

def load_average_intensity_projection(ophys_experiment_id):
    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    # build query
    query = '''
    SELECT
    wkf.storage_directory || wkf.filename AS avgint_file

    FROM ophys_experiments oe
    JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id
    JOIN well_known_files wkf ON wkf.attachable_id=ocsr.id
    JOIN well_known_file_types wkft ON wkft.id = wkf.well_known_file_type_id

    WHERE
    ocsr.current = 't'
    AND wkf.attachable_type = 'OphysCellSegmentationRun'
    AND wkft.name = 'OphysAverageIntensityProjectionImage'
    AND oe.id = {}
    '''.format(ophys_experiment_id)
    average_intensity_projection = mixin.select(query)
    return average_intensity_projection


def load_demixed_traces_array(ophys_experiment_id):
    """use SQL and the LIMS well known file system to find and load
            "demixed_traces.h5" then return the traces as an array

        Arguments:
            ophys_experiment_id {int} -- 9 digit ophys experiment ID

        Returns:
            demixed_traces_array -- mxn array where m = rois and n = time
        """
    filepath = get_demixed_traces_filepath(ophys_experiment_id)
    demix_file = h5py.File(filepath, 'r')
    demixed_traces_array = np.asarray(demix_file['data'])
    demix_file.close()
    return demixed_traces_array


def load_neuropil_traces_array(ophys_experiment_id):
    """use SQL and the LIMS well known file system to find and load
            "neuropil_traces.h5" then return the traces as an array

        Arguments:
            ophys_experiment_id {int} -- 9 digit ophys experiment ID

        Returns:
            neuropil_traces_array -- mxn array where m = rois and n = time
        """
    filepath = get_neuropil_traces_filepath(ophys_experiment_id)
    f = h5py.File(filepath, 'r')
    neuropil_traces_array = np.asarray(f['data'])
    f.close()
    return neuropil_traces_array


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
    filepath = get_motion_corrected_movie_filepath(ophys_experiment_id)
    motion_corrected_movie_file = h5py.File(filepath, 'r')
    motion_corrected_movie = motion_corrected_movie_file['data']
    return motion_corrected_movie


def load_rigid_motion_transform(ophys_experiment_id):
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
    filepath = get_rigid_motion_transform_filepath(ophys_experiment_id)
    rigid_motion_transform_df = pd.read_csv(filepath)
    return rigid_motion_transform_df


def load_objectlist(ophys_experiment_id):
    """loads the objectlist.txt file for the current segmentation run,
       then "cleans" the column names and returns a dataframe

    Arguments:
        experiment_id {[int]} -- 9 digit unique identifier for the experiment

    Returns:
        dataframe -- dataframe with the following columns:
                    (from http://confluence.corp.alleninstitute.org/display/IT/Ophys+Segmentation)
            trace_index:The index to the corresponding trace plot computed
                        (order of the computed traces in file _somaneuropiltraces.h5)
            center_x: The x coordinate of the centroid of the object in image pixels
            center_y:The y coordinate of the centroid of the object in image pixels
            frame_of_max_intensity_masks_file: The frame the object mask is in maxInt_masks2.tif
            frame_of_enhanced_movie: The frame in the movie enhimgseq.tif that best shows the object
            layer_of_max_intensity_file: The layer of the maxInt file where the object can be seen
            bbox_min_x: coordinates delineating a bounding box that
                        contains the object, in image pixels (upper left corner)
            bbox_min_y: coordinates delineating a bounding box that
                        contains the object, in image pixels (upper left corner)
            bbox_max_x: coordinates delineating a bounding box that
                        contains the object, in image pixels (bottom right corner)
            bbox_max_y: coordinates delineating a bounding box that
                        contains the object, in image pixels (bottom right corner)
            area: Total area of the segmented object
            ellipseness: The "ellipticalness" of the object, i.e. length of
                         long axis divided by length of short axis
            compactness: Compactness :  perimeter^2 divided by area
            exclude_code: A non-zero value indicates the object should be excluded from
                         further analysis.  Based on measurements in objectlist.txt
                        0 = not excluded
                        1 = doublet cell
                        2 = boundary cell
                        Others = classified as not complete soma, apical dendrite, ....
            mean_intensity: Correlates with delta F/F.  Mean brightness of the object
            mean_enhanced_intensity: Mean enhanced brightness of the object
            max_intensity: Max brightness of the object
            max_enhanced_intensity: Max enhanced brightness of the object
            intensity_ratio: (max_enhanced_intensity - mean_enhanced_intensity)
                             / mean_enhanced_intensity, for detecting dendrite objects
            soma_minus_np_mean: mean of (soma trace – its neuropil trace)
            soma_minus_np_std: 1-sided stdv of (soma trace – its neuropil trace)
            sig_active_frames_2_5:# frames with significant detected activity (spiking):
                                  Sum ( soma_trace > (np_trace + Snpoffsetmean+ 2.5 * Snpoffsetstdv)
                                  trace_38_soma.png  See example traces attached.
            sig_active_frames_4: # frames with significant detected activity (spiking):
                                 Sum ( soma_trace > (np_trace + Snpoffsetmean+ 4.0 * Snpoffsetstdv)
            overlap_count: 	Number of other objects the object overlaps with
            percent_area_overlap: the percentage of total object area that overlaps with other objects
            overlap_obj0_index: The index of the first object with which this object overlaps
            overlap_obj1_index: The index of the second object with which this object overlaps
            soma_obj0_overlap_trace_corr: trace correlation coefficient
                                          between soma and overlap soma0
                                          (-1.0:  excluded cell,  0.0 : NA)
            soma_obj1_overlap_trace_corr: trace correlation coefficient
                                          between soma and overlap soma1
    """
    filepath = get_objectlist_filepath(ophys_experiment_id)
    objectlist_dataframe = pd.read_csv(filepath)
    objectlist_dataframe = utils.update_objectlist_column_labels(objectlist_dataframe)
    return objectlist_dataframe
