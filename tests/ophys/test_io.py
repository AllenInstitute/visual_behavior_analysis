import os
import pytest
import platform
pytestmark = pytest.mark.skipif(platform.python_version() < '3.0.0',
                                reason='Requires Python 3')

def setup_module(module):
    import matplotlib
    matplotlib.use('Agg')
    from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
    from visual_behavior.ophys.io.create_analysis_files import create_analysis_files
    from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df
    EXPERIMENT_ID = 702134928

@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('PYTHONPATH','').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
def test_convert_level_1_to_level_2(cache_dir):

    convert_level_1_to_level_2(EXPERIMENT_ID, cache_dir=cache_dir)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('PYTHONPATH','').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
def test_create_analysis(cache_dir):

    create_analysis_files(EXPERIMENT_ID, cache_dir, overwrite_analysis_files=True)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('PYTHONPATH','').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
def test_create_multi_session_mean_df(cache_dir):

    get_multi_session_mean_df([EXPERIMENT_ID], cache_dir=cache_dir)
