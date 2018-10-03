
import pytest
import os

from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
from visual_behavior.ophys.io.create_analysis_files import create_analysis_files


EXPERIMENT_ID = 702134928

@pytest.mark.skipif(os.environ.get('PYTHONPATH','').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
def test_generate_level_1(cache_dir):

    convert_level_1_to_level_2(EXPERIMENT_ID, cache_dir=cache_dir)


@pytest.mark.skipif(os.environ.get('PYTHONPATH','').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
def test_create_analysis(cache_dir):

    create_analysis_files(EXPERIMENT_ID, cache_dir, overwrite_analysis_files=True)
