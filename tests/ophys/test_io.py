import os
import pytest
import platform
import matplotlib
matplotlib.use('Agg')

EXPERIMENT_ID = 901559828 #702134928
PLATFORM = 'scientifica'


@pytest.mark.slow
@pytest.mark.onprem
def test_convert_level_1_to_level_2(cache_dir):
    # import here to avoid errors on circleci
    from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
    print('experiment_id = {}'.format(EXPERIMENT_ID))
    convert_level_1_to_level_2(EXPERIMENT_ID, cache_dir=cache_dir)


# @pytest.mark.slow
# @pytest.mark.skipif(os.environ.get('PYTHONPATH', '').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
# def test_create_analysis(cache_dir):
#     # import here to avoid errors on circleci
#     from visual_behavior.ophys.io.create_analysis_files import create_analysis_files
#     create_analysis_files(EXPERIMENT_ID, cache_dir, overwrite_analysis_files=True)


# this test is hopelessly broken. It seems that the underlying function has changed dramatically since the test was written 
# commenting out for now (dro, 7/23/202)
# @pytest.mark.slow
# @pytest.mark.skipif(os.environ.get('PYTHONPATH', '').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
# def test_create_multi_session_mean_df(cache_dir):
#     # import here to avoid errors on circleci
#     from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df
#     get_multi_session_mean_df([EXPERIMENT_ID], cache_dir=cache_dir, platform=PLATFORM)
