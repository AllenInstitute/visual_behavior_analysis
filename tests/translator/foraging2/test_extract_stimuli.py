import pytest
import numpy as np
import pandas as pd

from visual_behavior.translator.foraging2 import extract
from visual_behavior.translator.foraging2 import extract_stimuli


@pytest.mark.parametrize("set_frame, expected", [
    (237, "im040", ),
    (800, "im037", ),
    (9753, "im046", ),
])
def test__resolve_image_category(foraging2_data_stage_0_2018_05_16, set_frame, expected):
    change_log = foraging2_data_stage_0_2018_05_16["items"]["behavior"]["stimuli"]["images"]["change_log"]
    assert extract_stimuli._resolve_image_category(change_log, set_frame) == expected


@pytest.mark.parametrize("idx, start_frame, last_trial_frame, expected", [
    (0, 0, 20000, (0, 237, ), ),
    (1, 238, 20000, (238, 800, ), ),
    (21, 19128, 20000, (19128, 20000, ), ),
])
def test__get_stimulus_epoch(foraging2_data_stage_0_2018_05_16, idx, start_frame, last_trial_frame, expected):
    set_log = foraging2_data_stage_0_2018_05_16["items"]["behavior"]["stimuli"]["images"]["set_log"]
    assert extract_stimuli._get_stimulus_epoch(set_log, idx, start_frame, last_trial_frame) == expected


@pytest.mark.parametrize("args, expected", [
    (
        (
            ([1] * 3 + [0] * 5 + [1] * 4 + [0] * 11 + [1] * 2),
            5,
            22,
        ),
        [(8, 12, ), ],
    ),
    (
        (
            ([1] * 3 + [0] * 5 + [1] * 4 + [0] * 11 + [1] * 2 + [0] * 10),
            0,
            25,
        ),
        [(0, 3, ), (8, 12, ), (23, 25, ), ],
    ),
    (
        (
            [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, ],
            0,
            12,
        ),
        [(1, 2, ), (3, 5, ), (6, 11, ), ],
    ),
])
def test__get_draw_epochs(args, expected):
    assert extract_stimuli._get_draw_epochs(*args) == expected


def test_get_visual_stimuli(foraging2_data_stage_0_2018_05_16):
    expected = pd.DataFrame(data=[
        {
            "duration": 0.24021686678929832,
            "end_frame": 19,
            "frame": 4,
            "image_category": "im111",
            "image_name": "im111",
            "orientation": np.nan,
            "time": 0.06456185853540397,
        },
        {
            "duration": 0.25631751922853896,
            "end_frame": 67,
            "frame": 51,
            "image_category": "im111",
            "image_name": "im111",
            "orientation": np.nan,
            "time": 0.8165832663787371,
        },
        {
            "duration": 0.2400167541733691,
            "end_frame": 113,
            "frame": 98,
            "image_category": "im111",
            "image_name": "im111",
            "orientation": np.nan,
            "time": 1.568588749344226,
        },
    ])

    stimuli = foraging2_data_stage_0_2018_05_16["items"]["behavior"]["stimuli"]
    time = extract.get_time(foraging2_data_stage_0_2018_05_16)

    pd.testing.assert_frame_equal(
        pd.DataFrame(data=extract_stimuli.get_visual_stimuli(stimuli, time)[:3]),
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


# def test_wut():
#     from visual_behavior.translator.foraging2 import data_to_change_detection_core
#
#     datapath = r'\\allen\aibs\mpe\Software\data\behavior\validation\stage_4\doc_images_efd6ed0_StupidDoCMouse.pkl'
#     data = pd.read_pickle(datapath)
#     core_data = data_to_change_detection_core(data)
#     periodic_flash = core_data['metadata']['params']['periodic_flash']
#
#     print validate_flash_blank_durations(core_data['visual_stimuli'], periodic_flash)
#     assert False
