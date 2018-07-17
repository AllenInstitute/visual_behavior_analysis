import pytest
import tempfile
import py
import os

from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2

@pytest.fixture(scope='function')
def tmpdir(request):
    """
    Get a temp dir to build in
    """

    dir = py.path.local(tempfile.mkdtemp())
    request.addfinalizer(lambda: dir.remove(rec=True))

    return dir

def test_generate_level_1(tmpdir):
    
    cache_dir = os.path.join(str(tmpdir))
    # print cache_dir
    # core_data = get_core_data(stage, mouse)
