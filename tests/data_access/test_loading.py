import os
import numpy as np
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

@pytest.mark.onprem
def test_get_dff_traces_for_roi():
    import visual_behavior.data_access.loading as loading
    cell_roi_id = 1080867088

    dff = loading.get_dff_traces_for_roi(cell_roi_id)
    np.isclose(dff[100],-0.13958465)
