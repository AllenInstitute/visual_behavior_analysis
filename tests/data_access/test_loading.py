import os
import pytest

def test_basic_import():
    from visual_behavior.data_access import loading

@pytest.mark.onprem
def test_get_filtered_ophys_session_table():
    import visual_behavior.data_access.loading as loading
    session_table = loading.get_filtered_ophys_session_table()

@pytest.mark.onprem
def test_get_filtered_ophys_experiment_table():
    import visual_behavior.data_access.loading as loading
    experiment_table = loading.get_filtered_ophys_experiment_table()
