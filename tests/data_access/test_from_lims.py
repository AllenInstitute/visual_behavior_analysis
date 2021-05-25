import pytest

@pytest.mark.onprem
def test_get_filtered_ophys_session_table():
    from visual_behavior.data_access. from_lims import get_id_type
    assert get_id_type(914580664) == 'ophys_experiment_id'
    assert get_id_type(914211263) == 'behavior_session_id'
    assert get_id_type(1086515263) == 'cell_specimen_id'
    assert get_id_type(914161594) == 'ophys_session_id'
    assert get_id_type(1080784881) == 'cell_roi_id'
    assert get_id_type(1234) == 'unknown_id'
