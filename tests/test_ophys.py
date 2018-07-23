import pytest
import tempfile
import py
import os

from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis


@pytest.fixture(scope='module')
def tmpdir(request):
    """
    Get a temp dir to build in
    """

    dir = py.path.local(tempfile.mkdtemp())
    request.addfinalizer(lambda: dir.remove(rec=True))

    return dir

@pytest.mark.skipif(not os.path.exists('/allen/programs/braintv/production/neuralcoding/'), reason='Source files not available')
def test_generate_level_2(tmpdir):
    
    cache_dir = os.path.join(str(tmpdir))
    convert_level_1_to_level_2(702134928, cache_dir=cache_dir)


@pytest.mark.skipif(not os.path.exists('/allen/programs/braintv/production/neuralcoding/'), reason='Source files not available')
def test_read_level_2_write_level_3(tmpdir):
    
    experiment_id = 702134928
    cache_dir = os.path.join(str(tmpdir))
    assert os.path.exists(cache_dir)
    dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
    assert dataset.cache_dir == cache_dir

    ResponseAnalysis(dataset)


    