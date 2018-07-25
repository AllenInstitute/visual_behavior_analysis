import pytest
import tempfile
import py
import os

from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2


@pytest.mark.skipif(not os.path.exists('/allen/programs/braintv/production/neuralcoding/'), reason='Source files not available')
def test_generate_level_1(tmpdir):

    cache_dir = os.path.join(str(tmpdir))
    convert_level_1_to_level_2(702134928, cache_dir=cache_dir)
