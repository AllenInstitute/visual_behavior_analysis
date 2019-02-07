
from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
from visual_behavior.ophys.io.create_analysis_files import create_analysis_files
from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysSession
from visual_behavior.ophys.io.filesystem_api import VisualBehaviorFileSystemAPI

import pytest
import os
import matplotlib
matplotlib.use('Agg')


EXPERIMENT_ID = 702134928


@pytest.mark.skipif(os.environ.get('PYTHONPATH','').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
def test_convert_level_1_to_level_2(cache_dir):

    data_set = convert_level_1_to_level_2(EXPERIMENT_ID, cache_dir=cache_dir)
    analysis_dir = os.path.join(cache_dir, str(EXPERIMENT_ID))
    data_set2 = VisualBehaviorOphysSession(EXPERIMENT_ID, api=VisualBehaviorFileSystemAPI(analysis_dir))

    assert data_set2 == data_set


@pytest.mark.skipif(os.environ.get('PYTHONPATH','').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
def test_create_analysis(cache_dir):

    create_analysis_files(EXPERIMENT_ID, cache_dir, overwrite_analysis_files=True)


@pytest.mark.skipif(os.environ.get('PYTHONPATH','').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
def test_create_multi_session_mean_df(cache_dir):

    get_multi_session_mean_df([EXPERIMENT_ID], cache_dir=cache_dir)
