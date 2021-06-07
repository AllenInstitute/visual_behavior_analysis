import os
import pytest

from visual_behavior.data_access import from_lims

# CONSTANTS

SPECIMEN_ID = 850862430
DONOR_ID    = 850862423

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


@pytest.mark.onprem
def test_get_behavior_session_id_for_ophys_experiment_id():
    assert from_lims.get_behavior_session_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI)  == BEHAVIOR_SESSION_ID_SCI


@pytest.mark.onprem
def test_get_ophys_container_id_for_ophys_experiment_id():
    assert from_lims.get_ophys_container_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO) == CONTAINER_ID_MESO


@pytest.mark.onprem
def test_get_supercontainer_id_for_ophys_experiment_id():
    assert from_lims.get_supercontainer_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO) == SUPERCONTAINER_ID_MESO
    assert from_lims.get_supercontainer_id_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI)  == None


@pytest.mark.onprem
def test_get_all_ids_for_ophys_experiment_id():
    sci_table = from_lims.get_all_ids_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI)
    assert sci_table["ophys_experiment_id"][0] == OPHYS_EXPERIMENT_ID_SCI
    assert sci_table["ophys_session_id"][0]    == OPHYS_SESSION_ID_SCI
    assert sci_table["behavior_session_id"][0] == BEHAVIOR_SESSION_ID_SCI
    assert sci_table["ophys_container_id"][0]  == CONTAINER_ID_SCI
    assert sci_table["supercontainer_id"][0]  == None
    
    meso_table = from_lims.get_all_ids_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO)
    assert meso_table["ophys_experiment_id"][0] == OPHYS_EXPERIMENT_ID_MESO
    assert meso_table["ophys_session_id"][0]    == OPHYS_SESSION_ID_MESO
    assert meso_table["behavior_session_id"][0] == BEHAVIOR_SESSION_ID_MESO
    assert meso_table["ophys_container_id"][0]  == CONTAINER_ID_MESO
    assert meso_table["supercontainer_id"][0]  == SUPERCONTAINER_ID_MESO


@pytest.mark.onprem
def test_get_general_info_for_ophys_experiment_id_columns():
    meso_gi_columns = from_lims.get_general_info_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO).columns
    assert any(meso_gi_columns == GENERAL_INFO_COLUMNS)

    sci_gi_columns = from_lims.get_general_info_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI).columns
    assert any(sci_gi_columns == GENERAL_INFO_COLUMNS)


@pytest.mark.onprem
def test_get_general_info_for_ophys_experiment_id():
    meso_table = from_lims.get_general_info_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO)
    assert meso_table["ophys_experiment_id"][0] == OPHYS_EXPERIMENT_ID_MESO
    assert meso_table["ophys_session_id"][0]    == OPHYS_SESSION_ID_MESO
    assert meso_table["behavior_session_id"][0] == BEHAVIOR_SESSION_ID_MESO
    assert meso_table["ophys_container_id"][0]  == CONTAINER_ID_MESO
    assert meso_table["supercontainer_id"][0]  == SUPERCONTAINER_ID_MESO
    
    assert meso_table["experiment_workflow_state"][0] == "passed"
    assert meso_table["session_workflow_state"][0]    == "uploaded"
    assert meso_table["container_workflow_state"][0]  == "published"
    
    assert meso_table["specimen_id"][0]   == 850862430
    assert meso_table["donor_id"][0]      == 850862423
    assert meso_table["specimen_name"][0] == 'Sst-IRES-Cre;Ai148-457841'

    assert meso_table["session_type"][0]       == 'OPHYS_6_images_B'
    assert meso_table["targeted_structure"][0] == 'VISp'
    assert meso_table["depth"][0]              == 225
    assert meso_table["equipment_name"][0]     == 'MESO.1'
    assert meso_table["project"][0]            == 'VisualBehaviorMultiscope'


@pytest.mark.onprem
def test_get_ophys_experiment_ids_for_ophys_session_id():
    assert from_lims.get_ophys_experiment_ids_for_ophys_session_id(OPHYS_SESSION_ID_SCI) == OPHYS_EXPERIMENT_ID_SCI
    
    meso_table = from_lims.get_ophys_experiment_ids_for_ophys_session_id(OPHYS_SESSION_ID_MESO)
    assert meso_table[0] == 960410023
    assert meso_table[1] == 960410026
    assert meso_table[2] == 960410028
    assert meso_table[3] == 960410032
    assert meso_table[4] == 960410035
    assert meso_table[5] == 960410038
    assert meso_table[6] == 960410040
    assert meso_table[7] == 960410042


@pytest.mark.onprem
def test_get_behavior_session_id_for_ophys_session_id():
    assert from_lims.get_behavior_session_id_for_ophys_session_id(OPHYS_SESSION_ID_MESO) == 959681045


@pytest.mark.onprem
def test_get_ophys_container_ids_for_ophys_session_id():
    sci_table = from_lims.get_ophys_container_ids_for_ophys_session_id(OPHYS_SESSION_ID_SCI)
    assert sci_table["ophys_container_id"][0] == CONTAINER_ID_SCI

    meso_table = sci_table = from_lims.get_ophys_container_ids_for_ophys_session_id(OPHYS_SESSION_ID_MESO)
    assert meso_table["ophys_container_id"][0] == 1018028342
    assert meso_table["ophys_container_id"][1] == 1018028345
    assert meso_table["ophys_container_id"][2] == 1018028339
    assert meso_table["ophys_container_id"][3] == 1018028348
    assert meso_table["ophys_container_id"][4] == 1018028354
    assert meso_table["ophys_container_id"][5] == 1018028357
    assert meso_table["ophys_container_id"][6] == 1018028351
    assert meso_table["ophys_container_id"][7] == 1018028360


@pytest.mark.onprem
def test_get_supercontainer_id_for_ophys_session_id():
    assert from_lims.get_supercontainer_id_for_ophys_session_id(OPHYS_SESSION_ID_SCI)  == None
    assert from_lims.get_supercontainer_id_for_ophys_session_id(OPHYS_SESSION_ID_MESO) == 1027966320


@pytest.mark.onprem
def test_get_all_ids_for_ophys_session_id():
    sci_table = from_lims.get_all_ids_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_SCI)
    assert sci_table["ophys_experiment_id"][0] == OPHYS_EXPERIMENT_ID_SCI
    assert sci_table["ophys_session_id"][0]    == OPHYS_SESSION_ID_SCI
    assert sci_table["behavior_session_id"][0] == BEHAVIOR_SESSION_ID_SCI
    assert sci_table["ophys_container_id"][0]  == CONTAINER_ID_SCI
    assert sci_table["supercontainer_id"][0]   == None

    meso_table = from_lims.get_all_ids_for_ophys_experiment_id(OPHYS_EXPERIMENT_ID_MESO)
    assert meso_table["ophys_experiment_id"][0] == 960410023
    assert meso_table["ophys_session_id"][2]    == 959458018
    assert meso_table["behavior_session_id"][4] == 959681045
    assert meso_table["ophys_container_id"][5]  == 1018028357
    assert meso_table["supercontainer_id"][7]   == 1027966320


@pytest.mark.onprem
def test_get_general_info_for_ophys_session_id():
    meso_table = from_lims.get_general_info_for_ophys_session_id(OPHYS_EXPERIMENT_ID_MESO)
    assert meso_table["ophys_experiment_id"][0] == 960410042
    assert meso_table["ophys_session_id"][1]    == 959458018
    assert meso_table["behavior_session_id"][2] == 959681045
    assert meso_table["ophys_container_id"][2]  == 1018028357
    assert meso_table["supercontainer_id"][2]   == 1027966320
    
    assert meso_table["experiment_workflow_state"][0] == "passed"
    assert meso_table["session_workflow_state"][0]    == "uploaded"
    assert meso_table["container_workflow_state"][0]  == "published"
    
    assert meso_table["specimen_id"][0]   == 850862430
    assert meso_table["donor_id"][0]      == 850862423
    assert meso_table["specimen_name"][0] == 'Sst-IRES-Cre;Ai148-457841'

    assert meso_table["session_type"][0]       == 'OPHYS_6_images_B'
    assert meso_table["targeted_structure"][0] == 'VISl'
    assert meso_table["depth"][0]              == 300
    assert meso_table["equipment_name"][0]     == 'MESO.1'
    assert meso_table["project"][0]            == 'VisualBehaviorMultiscope'


@pytest.mark.onprem
def test_get_ophys_experiment_ids_for_behavior_session_id():

    meso_table = from_lims.get_ophys_experiment_ids_for_behavior_session_id(BEHAVIOR_SESSION_ID_MESO)
    assert meso_table["ophys_experiment_id"][0] == 960410023
    assert meso_table["ophys_experiment_id"][1] == 960410026
    assert meso_table["ophys_experiment_id"][4] == 960410035


@pytest.mark.onprem
def test_get_ophys_session_id_for_behavior_session_id():
    assert from_lims.get_ophys_session_id_for_behavior_session_id(BEHAVIOR_SESSION_ID_MESO) == OPHYS_SESSION_ID_MESO


@pytest.mark.onprem
def test_get_ophys_container_ids_for_behavior_session_id():
    meso_table = from_lims.get_ophys_container_ids_for_behavior_session_id(BEHAVIOR_SESSION_ID_MESO)
    meso_table["ophys_container_id"][0] == 1018028360
    meso_table["ophys_container_id"][4] == 1018028348
    meso_table["ophys_container_id"][7] == 1018028342


@pytest.mark.onprem
def test_get_supercontainer_id_for_behavior_session_id():
    assert from_lims.get_supercontainer_id_for_behavior_session_id(BEHAVIOR_SESSION_ID_SCI)  == None
    assert from_lims.get_supercontainer_id_for_behavior_session_id(BEHAVIOR_SESSION_ID_MESO) == 1027966320


@pytest.mark.onprem
def test_get_all_ids_for_behavior_session_id():
    sci_table = from_lims.get_all_ids_for_behavior_session_id(BEHAVIOR_SESSION_ID_SCI)
    assert sci_table["ophys_experiment_id"][0] == OPHYS_EXPERIMENT_ID_SCI
    assert sci_table["ophys_session_id"][0]    == OPHYS_SESSION_ID_SCI
    assert sci_table["behavior_session_id"][0] == BEHAVIOR_SESSION_ID_SCI
    assert sci_table["ophys_container_id"][0]  == CONTAINER_ID_SCI
    assert sci_table["supercontainer_id"][0]   == None

    meso_table = from_lims.get_all_ids_for_behavior_session_id(OPHYS_EXPERIMENT_ID_MESO)
    assert meso_table["ophys_experiment_id"][0] == 960410042
    assert meso_table["ophys_session_id"][2]    == 959458018
    assert meso_table["behavior_session_id"][4] == 959681045
    assert meso_table["ophys_container_id"][5]  == 1018028339
    assert meso_table["supercontainer_id"][7]   == 1027966320


@pytest.mark.onprem
def test_get_general_info_for_behavior_session_id():
    meso_table = from_lims.get_general_info_for_behavior_session_id(BEHAVIOR_SESSION_ID_MESO)
    assert meso_table["ophys_experiment_id"][0] == 960410042
    assert meso_table["ophys_session_id"][1]    == 959458018
    assert meso_table["behavior_session_id"][2] == 959681045
    assert meso_table["ophys_container_id"][2]  == 1018028357
    assert meso_table["supercontainer_id"][2]   == 1027966320
    
    assert meso_table["experiment_workflow_state"][0] == "passed"
    assert meso_table["session_workflow_state"][0]    == "uploaded"
    assert meso_table["container_workflow_state"][0]  == "published"
    
    assert meso_table["specimen_id"][0]   == 850862430
    assert meso_table["donor_id"][0]      == 850862423
    assert meso_table["specimen_name"][0] == 'Sst-IRES-Cre;Ai148-457841'

    assert meso_table["session_type"][0]       == 'OPHYS_6_images_B'
    assert meso_table["targeted_structure"][0] == 'VISl'
    assert meso_table["depth"][0]              == 300
    assert meso_table["equipment_name"][0]     == 'MESO.1'
    assert meso_table["project"][0]            == 'VisualBehaviorMultiscope'


@pytest.mark.onprem
def test_get_filtered_ophys_session_table():
    from visual_behavior.data_access. from_lims import get_id_type
    assert get_id_type(914580664) == 'ophys_experiment_id'
    assert get_id_type(914211263) == 'behavior_session_id'
    assert get_id_type(1086515263) == 'cell_specimen_id'
    assert get_id_type(914161594) == 'ophys_session_id'
    assert get_id_type(1080784881) == 'cell_roi_id'
    assert get_id_type(1234) == 'unknown_id'
