import os
import pytest

CIRCLECI = os.environ.get('PYTHONPATH', '').startswith('/home/circleci')

@pytest.mark.skipif(CIRCLECI, reason='Function requires LIMS access')
def test_get_filtered_ophys_session_table():
    import visual_behavior.data_access.loading as loading
    session_table = session_table = loading.get_filtered_ophys_session_table()

@pytest.mark.skipif(CIRCLECI, reason='Function requires LIMS access')
def test_get_filtered_ophys_experiment_table():
    import visual_behavior.data_access.loading as loading
    experiment_table = loading.get_filtered_ophys_experiment_table()
