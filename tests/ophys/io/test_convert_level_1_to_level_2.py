import pytest
import tempfile
import py
import os

from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2


@pytest.mark.skipif(os.environ.get('PYTHONPATH','').startswith('/home/circleci'), reason='Cannot test against real files on CircleCI')
def test_generate_level_1(tmpdir):

    print(os.environ)

    cache_dir = os.path.join(str(tmpdir))
    convert_level_1_to_level_2(702134928, cache_dir=cache_dir)
