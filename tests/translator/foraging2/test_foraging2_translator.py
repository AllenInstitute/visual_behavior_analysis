import pytest
import datetime
import pandas as pd
import numpy as np

from visual_behavior.translator.foraging2 import data_to_monolith


EXPECTED_MONOLITH = pd.DataFrame(data={
    "auto_rewarded_trial": {0: False, 1: False, 2: False, },
    "change_contrast": {0: None, 1: None, 2: None, },
    "change_frame": {0: 413, 1: 781, 2: 1241, },
    "change_image_category": {0: "im037", 1: "im111", 2: "im040", },
    "change_image_name": {0: "im037", 1: "im111", 2: "im040", },
    "change_orientation": {0: None, 1: None, 2: None, },
    "change_time": {0: 12.000089, 1: 18.138141, 2: 25.8119, },
    "cumulative_reward_number": {0: 1, 1: 2, 2: 3, },
    "cumulative_volume": {0: 0.008, 1: 0.016, 2: 0.024, },
    "delta_orientation": {0: None, 1: None, 2: None, },
    "initial_contrast": {0: None, 1: None, 2: None, },
    "initial_image_category": {0: "im111", 1: "im037", 2: "im111", },
    "initial_image_name": {0: "im111", 1: "im037", 2: "im111", },
    "initial_orientation": {0: None, 1: None, 2: None, },
    "lick_frames": {0: [426], 1: [793], 2: [1253], },
    "lick_times": {
        0: [12.215056513123665],
        1: [18.33671554138576],
        2: [26.009632692371557],
    },
    "optogenetics": {0: None, 1: None, 2: None, },
    "response_frame": {0: 426, 1: 793, 2: 1253, },
    "response_latency": {0: 0.215514, 1: 0.198992, 2: 0.198241, },
    "response_time": {0: 12.215603, 1: 18.337132, 2: 26.010133, },
    "response_type": {0: None, 1: None, 2: None, },
    "reward_frames": {0: [426], 1: [793], 2: [1253], },
    "reward_times": {
        0: [12.215641293879314],
        1: [18.337147830803804],
        2: [26.01016544341492],
    },
    "reward_volume": {0: None, 1: None, 2: None, },
    "rewarded": {0: True, 1: True, 2: True, },
    "scheduled_change_time": {0: None, 1: None, 2: None, },
    "start_frame": {0: 0, 1: 552, 2: 1058, },
    "start_time": {0: 5.056719, 1: 14.301958, 2: 22.742372, },
    "end_time": {0: 14.017085, 1: 22.591682, 2: 31.015604, },
    "end_frame": {0: 534, 1: 1048, 2: 1553, },
    "startdatetime": datetime.datetime(2018, 3, 9, 16, 6, 59, 199000),
    "year": 2018,
    "month": 3,
    "day": 9,
    "hour": 16,
    "date": datetime.date(2018, 3, 9),
    "dayofweek": 4,
    "mouse_id": "testmouse",
    "user_id": None,
    "blank_duration_range": {0: (0.5, 0.5, ), 1: (0.5, 0.5, ), 2: (0.5, 0.5, ), },
    "session_id": None,
    "filename": "",
    "device_name": None,
    "session_duration": 28406.585909910973,
    "stimulus_duration": 6.0,
    "pre_change_time": 2.25,
    "task_id": "DoC",
    "image_path": "//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_0_2017.07.14.pkl",
    "scheduled_trial_duration": None,
    "trial_duration": {0: 8.960366089911771, 1: 8.289724131823364, 2: 8.273231820043616, },
    "trial_type": {0: "go", 1: "go", 2: "go", },
    "response_window": {0: (0.15, 2.0, ), 1: (0.15, 2.0, ), 2: (0.15, 2.0, ), },
    "stimulus_on_frames": None,  # we don't want to check this because it is obnoxiously massive...we just have a nice little obnoxious unit test instead...
})


def test_data_to_monolith(foraging2_data_fixture):
    monolith = data_to_monolith(foraging2_data_fixture)

    assert monolith.stimulus_on_frames.notnull().all(), "stimulus_on_frame is not what we expect..."

    monolith.stimulus_on_frames = None

    pd.testing.assert_frame_equal(
        monolith,
        EXPECTED_MONOLITH,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )
