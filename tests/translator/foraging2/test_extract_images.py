import numpy as np
from numpy.testing import assert_allclose
from visual_behavior.translator.foraging2 import extract_images

EXPECTED_METADATA = {
    'image_set': '//allen/aibs/mpe/Software/stimulus_files/nimages_0_20170714.zip',
}


def test_get_image_metadata(foraging2_data_stage4_2018_05_10):

    image_metadata = extract_images.get_image_metadata(foraging2_data_stage4_2018_05_10)
    assert image_metadata == EXPECTED_METADATA
