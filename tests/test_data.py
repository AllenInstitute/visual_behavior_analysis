import pytest
import datetime
import numpy as np
import pandas as pd

from visual_behavior import data


@pytest.mark.parametrize("df, pickle_data, keydict, expected", [
    (
        pd.DataFrame(data={"test": {17: "test", 18: "value", }, }),
        {"some_thing": "some_value", },
        {"another_value": "some_thing", "doesnt_exist": "doesnt_exist", },
        pd.DataFrame(data={
            "test": {17: "test", 18: "value", },
            "another_value": {17: "some_value", 18: "some_value", },
            "doesnt_exist": {17: None, 18: None, },
        }, columns=["test", "doesnt_exist", "another_value", ]),
    ),
    (
        pd.DataFrame(data={"test": {17: "test", 18: "value", }, }),
        {"some_thing": "some_value", },
        None,
        pd.DataFrame(data={"test": {17: "test", 18: "value", }, }),
    ),
])
def test_annotate_parameters(df, pickle_data, keydict, expected):
    pd.testing.assert_frame_equal(
        data.annotate_parameters(df, pickle_data, keydict),
        expected,
        check_like=True
    )


@pytest.mark.parametrize("df, expected", [
    (
        pd.DataFrame(data={
            "startdatetime": {17: pd.Timestamp("2018-01-22 08:47:09.440000"), },
        }),
        pd.DataFrame(data={
            "startdatetime": {17: pd.Timestamp("2018-01-22 08:47:09.440000"), },
            "date": {17: "2018-01-22", },
            "year": {17: 2018, },
            "month": {17: 1, },
            "day": {17: 22, },
            "hour": {17: 8, },
            "dayofweek": {17: 0, },
        }, columns=[
            "startdatetime",
            "date",
            "year",
            "month",
            "day",
            "hour",
            "dayofweek",
        ]),
    ),
])
def test_explode_startdatetime(df, expected):
    pd.testing.assert_frame_equal(data.explode_startdatetime(df), expected)


@pytest.mark.parametrize("df, expected", [
    (
        pd.DataFrame(data={
            "reward_times": {17: [37.507777565158904, ], },
        }),
        pd.DataFrame(data={
            "reward_times": {17: [37.507777565158904, ], },
            "number_of_rewards": {17: 1, },
        }, columns=["reward_times", "number_of_rewards", ]),
    )
])
def test_annotate_n_rewards(df, expected):
    pd.testing.assert_frame_equal(data.annotate_n_rewards(df), expected)


@pytest.mark.parametrize("df, pickled_data, expected", [
    (
        pd.DataFrame(data={
            "test": {17: None, },
        }),
        {"rig_id": "test_rig", },
        pd.DataFrame(data={
            "test": {17: None, },
            "rig_id": {17: "test_rig", },
        }, columns=["test", "rig_id", ]),
    ),
])
def test_annotate_rig_id(df, pickled_data, expected):
    pd.testing.assert_frame_equal(
        data.annotate_rig_id(df, pickled_data),
        expected
    )


@pytest.mark.parametrize("df, pickled_data, expected", [
    (
        pd.DataFrame(data={
            "test": {17: None, },
        }),
        {"startdatetime": "2018-01-22 08:47:09.440000", },
        pd.DataFrame(data={
            "test": {17: None, },
            "startdatetime": {17: pd.Timestamp("2018-01-22 08:47:09.440000"), },
        }, columns=["test", "startdatetime", ]),
    ),
])
def test_annotate_startdatetime(df, pickled_data, expected):
    pd.testing.assert_frame_equal(
        data.annotate_startdatetime(df, pickled_data),
        expected
    )


@pytest.mark.parametrize("df, pickled_data, expected", [
    (
        pd.DataFrame(data={
            "number_of_rewards": {17: 2, },
        }),
        {"rewardvol": 0.005, },
        pd.DataFrame(data={
            "reward_volume": {17: 0.01, },
            "number_of_rewards": {17: 2, },
            "cumulative_volume": {17: 0.01, },
        }, columns=["number_of_rewards", "reward_volume", "cumulative_volume", ]),
    ),
    (
        pd.DataFrame(data={
            "reward_volume": {17: 0.005, },
            "number_of_rewards": {17: 2, },
        }),
        {"rewardvol": 0.005, },
        pd.DataFrame(data={
            "reward_volume": {17: 0.005, },
            "number_of_rewards": {17: 2, },
            "cumulative_volume": {17: 0.005, },
        }, columns=["number_of_rewards", "reward_volume", "cumulative_volume", ]),
    )
])
def test_annotate_cumulative_reward(df, pickled_data, expected):
    pd.testing.assert_frame_equal(
        data.annotate_cumulative_reward(df, pickled_data),
        expected
    )


@pytest.mark.parametrize("df, filename, expected", [
    (
        pd.DataFrame(data={"test": {17: "test", }, }),
        "test/path/fname.py",
        pd.DataFrame(data={
            "test": {17: "test", },
            "filepath": {17: "test/path", },
            "filename": {17: "fname.py", },
        }, columns=["test", "filepath", "filename", ])
    ),
])
def test_annotate_filename(df, filename, expected):
    pd.testing.assert_frame_equal(
        data.annotate_filename(df, filename),
        expected
    )


@pytest.mark.parametrize("df, expected", [
    (
        pd.DataFrame(data={
            "auto_rearded": {17, True, },
        }),
        pd.DataFrame(data={
            "auto_rewarded": {17, True, },
        }),
    ),
])
def test_fix_autorearded(df, expected):
    pd.testing.assert_frame_equal(data.fix_autorearded(df), expected)


@pytest.mark.parametrize("df, expected", [
    (
        pd.DataFrame(data={
            "lick_times": {17: [0.1, 0.2, 0.3, 0.4, ], },
        }),
        pd.DataFrame(data={
            "lick_times": {17: [0.1, 0.2, 0.3, 0.4, ], },
            "number_of_licks": {17: 4, },
            "lick_rate_Hz": {17: 10.0, },
        }, columns=["lick_times", "number_of_licks", "lick_rate_Hz", ]),
    ),
])
def test_annotate_lick_vigor(df, expected):
    pd.util.testing.assert_almost_equal(data.annotate_lick_vigor(df), expected)
