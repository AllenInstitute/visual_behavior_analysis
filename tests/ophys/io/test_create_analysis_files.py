import os
import pytest
from visual_behavior.ophys.io.create_analysis_files import create_analysis_files


@pytest.mark.skipif(not os.path.exists('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis'), reason='Source files not available')
def test_create_analysis(tmpdir):

    experiment_id = 712860764
    cache_dir = os.path.join(str(tmpdir))
    create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)