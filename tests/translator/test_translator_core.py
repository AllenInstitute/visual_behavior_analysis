import os
import pytest
import pandas as pd

from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core

test_data_path = os.path.join('/fixtures', '815160769.pkl')
test_data_exists = os.path.isfile(test_data_path)

@pytest.fixture
def test_data():
    return pd.read_pickle(test_data_path)


@pytest.mark.skipif(
    not test_data_exists, 
    reason='test data not present to run test', 
)
def test_fix_482_endframe_bug(test_data):
    core_data = data_to_change_detection_core(test_data)
    trials = create_extended_dataframe(is_ophys=True, **core_data)
    last_two_endframes = trials.tail()['endframe'][-2:]

    assert last_two_endframes.iloc[1] - last_two_endframes.iloc[0] < 10000, \
        'more than 10000 frames between 2nd to last frame and last frame, fix_482 might not be working...'