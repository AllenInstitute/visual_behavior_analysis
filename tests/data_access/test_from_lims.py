import os
import pytest

import visual_behavior.data_access import from_lims

# CONSTANTS

# SPECIMEN_ID = 
# DONOR_ID    = 

# MESOSCOPE IDS
OPHYS_EXPERIMENT_ID_MESO = 960410026
OPHYS_SESSION_ID_MESO    = 959458018
BEHAVIOR_SESSION_ID_MESO = 959681045
CONTAINER_ID_MESO        = 1018028345
SUPERCONTAINER_ID_MESO   = 1027966320

# SCIENTIFICA IDS
OPHYS_EXPERIMENT_ID_SCI = 963394081
OPHYS_SESSION_ID_SCI    = 962736894
BEHAVIOR_SESSION_ID_SCI = 962922731
CONTAINER_ID_SCI        = 969421516

GENERAL_INFO_COLUMNS = ['ophys_experiment_id', 'ophys_session_id', 'behavior_session_id',
                        'foraging_id', 'ophys_container_id', 'supercontainer_id',
                        'experiment_workflow_state', 'session_workflow_state',
                        'container_workflow_state', 'specimen_id', 'donor_id', 'specimen_name',
                        'date_of_acquisition', 'session_type', 'targeted_structure', 'depth',
                        'equipment_name', 'project', 'experiment_storage_directory',
                        'session_storage_directory', 'container_storage_directory']



# @pytest.mark.onprem
# def test_lims_access():


@pytest.mark.onprem
def test_get_current_segmentation_run_id_for_ophys_experiment_id():
    assert from_lims.get_current_segmentation_run_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO) == 1080658052
    assert from_lims.get_current_segmentation_run_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI)  == 1080679144


@pytest.mark.onprem
def test_get_ophys_session_id_for_ophys_experiment_id():
    assert from_lims.get_ophys_session_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO) == OPHYS_SESSION_ID_MESO
    assert from_lims.get_ophys_session_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI)  == OPHYS_SESSION_ID_SCI


@pytest.mark.onprem
def test_get_behavior_session_id_for_ophys_experiment_id():
    assert from_lims.get_behavior_session_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO) == BEHAVIOR_SESSION_ID_MESO
    assert from_lims.get_behavior_session_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI)  == BEHAVIOR_SESSION_ID_SCI


@pytest.mark.onprem
def test_get_ophys_container_id_for_ophys_experiment_id():
    assert from_lims.get_ophys_container_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO) == CONTAINER_ID_MESO
    assert from_lims.get_ophys_container_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI)  == CONTAINER_ID_SCI


@pytest.mark.onprem
def test_get_supercontainer_id_for_ophys_experiment_id():
    assert from_lims.get_super_container_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO) == SUPERCONTAINER_ID_MESO
    assert from_lims.get_super_container_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI)  == None


@pytest.mark.onprem
def test_get_all_ids_for_ophys_experiment_id():
    meso_table = from_lims.get_all_ids_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO)
    assert meso_table["ophys_experiment_id"][0] == OPHYS_EXPERIMENT_ID_MESO
    assert meso_table["ophys_session_id"][0]    == OPHYS_SESSION_ID_MESO
    assert meso_table["behavior_session_id"][0] == BEHAVIOR_SESSION_ID_MESO
    assert meso_table["ophys_container_id"][0]  == CONTAINER_ID_MESO
    assert meso_table["super_container_id"][0]  == SUPERCONTAINER_ID_MESO

    sci_table = from_lims.get_all_ids_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI)
    assert sci_table["ophys_experiment_id"][0] == OPHYS_EXPERIMENT_ID_SCI
    assert sci_table["ophys_session_id"][0]    == OPHYS_SESSION_ID_SCI
    assert sci_table["behavior_session_id"][0] == BEHAVIOR_SESSION_ID_SCI
    assert sci_table["ophys_container_id"][0]  == CONTAINER_ID_SCI
    assert sci_table["super_container_id"][0]  == None

@pytest.mark.onprem
def test_get_general_info_for_ophys_experiment_id_columns():
    meso_gi_columns = from_lims.get_general_info_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO).columns
    assert any(meso_gi_columns == GENERAL_INFO_COLUMNS)

    sci_gi_columns = from_lims.get_general_info_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI).columns
    assert any(sci_gi_columns == GENERAL_INFO_COLUMNS)

@pytest.mark.onprem
def test_get_general_info_for_ophys_experiment_id():
    meso_table = from_lims.get_general_info_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO)
    
