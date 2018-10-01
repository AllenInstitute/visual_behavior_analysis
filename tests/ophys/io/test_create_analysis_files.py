import os
import pytest
from visual_behavior.ophys.io.create_analysis_files import create_analysis_files


@pytest.mark.skipif(os.environ.get('PYTHONPATH','').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
def test_create_analysis(tmpdir):

    experiment_id = 712860764
    cache_dir = os.path.join(str(tmpdir))
    create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)
