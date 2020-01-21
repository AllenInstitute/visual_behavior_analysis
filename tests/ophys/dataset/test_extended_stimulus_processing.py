import os
import pandas as pd
import numpy as np
import pytest
from visual_behavior.ophys.dataset import extended_stimulus_processing as esp

CIRCLECI = os.environ.get('PYTHONPATH', '').startswith('/home/circleci')

@pytest.mark.parametrize('flash_times, other_times, expected', [
    (
        np.array([1, 2, 3, 4, 5], dtype=float),
        np.array([2.5], dtype=float),
        np.array([np.nan, np.nan, 0.5, 1.5, 2.5], dtype=float),
    ),
    (
        np.array([1, 2, 3, 4, 5], dtype=float),
        np.array([2], dtype=float),
        np.array([np.nan, 0, 1, 2, 3], dtype=float),
    )
])
def test_time_from_last(flash_times, other_times, expected):
    np.testing.assert_array_equal(
        esp.time_from_last(flash_times, other_times),
        expected)

@pytest.mark.parametrize('values, timestamps, start_time, stop_time, expected', [
    (
        np.array([1, 2, 3, 4, 5]),
        np.array([1, 2, 3, 4, 5]),
        2,
        4,
        2.5
    )
])
def test_trace_average(values, timestamps, start_time, stop_time, expected):
    assert esp.trace_average(values, timestamps, start_time, stop_time) == expected

@pytest.mark.parametrize('image_index, omitted_index, expected', [
    (
        pd.Series([0, 0, 0, 1, 1, 1, 2, 3, 2, 2, 2]),
        3,
        np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).astype(bool),
    ),
    (
        pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3]),
        3,
        np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).astype(bool),
    ),
    (
        pd.Series([3, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]),
        3,
        np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).astype(bool),
    ),
])
def test_find_change(image_index, omitted_index, expected):
    assert np.all(esp.find_change(image_index, omitted_index) == expected)

@pytest.fixture
def stimulus_block_df():
    '''
    A small mock stim df with just the names (and block index?)
    '''
    # The image index is just the identitiy of the image
    image_index =            [0, 0, 0, 1, 1, 1, 2, 8, 2, 3, 3, 3, 1, 1, 1, 0, 0, 0]

    # A block is a period of time when the same image is being presented
    image_block =            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]

    # Is this the first block of this image? second? etc. 
    image_block_repetition = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    # Which unique flash within the block is this presentation?
    repeat_within_block =    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    return pd.DataFrame({'image_index':image_index,
                         'image_block':image_block,
                         'image_block_repetition':image_block_repetition,
                         'repeat_within_block':repeat_within_block})

def test_get_block_index(stimulus_block_df):
    assert np.all(esp.get_block_index(stimulus_block_df['image_index'], 8) == stimulus_block_df['image_block'])

def test_get_image_block_repetition(stimulus_block_df):
    pass
    #  assert np.all(esp.get_image_block_repetitions(stimulus_block_df['image_index'], 8) == stimulus_block_df['image_block'])
