import os
import warnings
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