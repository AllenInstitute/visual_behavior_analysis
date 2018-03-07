import pytest
import datetime
import numpy as np
import pandas as pd

from visual_behavior import legacy_data

TIMESTAMP_ISO = "2018-01-22 08:47:09.440000"
TIMESTAMP = pd.Timestamp(TIMESTAMP_ISO)


def test_annotate_parameters():

    df = pd.DataFrame(
        [
            {"key0": 0},
            {"key0": 3},
            ]
        )

    keydict = {
        # the column "another_value" should be filled with the values of "some_thing"
        "another_value": "some_thing",
        # the column "dne" should be filled with values of "dne", which doesn't exist
        "dne": "dne",
        }

    pickled_data = {
        "some_thing": "some_value",
        }

    expected = pd.DataFrame(
        [
            # since "dne" doesn't exist, it should be filled with None
            {"key0": 0, "another_value": "some_value", "dne": None},
            {"key0": 3, "another_value": "some_value", "dne": None},
            ]
        )

    df_out = legacy_data.annotate_parameters(df, pickled_data, keydict)

    pd.testing.assert_frame_equal(
        df_out,
        expected,
        check_like=True,
    )

    # if no keydict is passed, we shouldn't modify the dataframe
    df_out = legacy_data.annotate_parameters(df, pickled_data, None)

    pd.testing.assert_frame_equal(
        df_out,
        df_out,
        check_like=True,
    )

def test_explode_startdatetime():


    df = pd.DataFrame(
        [
            {
            "startdatetime": TIMESTAMP,
            },
        ]
    )

    df_expected = pd.DataFrame(
        [
            {
            "startdatetime": TIMESTAMP,
            "date": "2018-01-22",
            "year": 2018,
            "month": 1,
            "day": 22,
            "hour": 8,
            "dayofweek": 0,
            },
        ]
    )

    df_out = legacy_data.explode_startdatetime(df)

    pd.testing.assert_frame_equal(
        df_out,
        df_expected,
        check_like=True,
        )



def test_annotate_n_rewards():
    df = pd.DataFrame(
        [
            {"reward_times": [37.50, ]},
            {"reward_times": [1.2, 37.50, ]},
            {"reward_times": []},
        ]
    )


    expected_vals = [
        1,
        2,
        0,
    ]
    expected_col = pd.Series(expected_vals,name='number_of_rewards')

    df_out = legacy_data.annotate_n_rewards(df)

    pd.testing.assert_series_equal(
        df_out['number_of_rewards'],
        expected_col,
        )

def test_annotate_rig_id():

    df = pd.DataFrame(
        [
            {"key0": 0},
            {"key0": 3},
            ]
        )

    pickled_data = {
        "rig_id": "TEST_RIG",
        }

    expected = pd.DataFrame(
        [
            # since "dne" doesn't exist, it should be filled with None
            {"key0": 0, "rig_id": "TEST_RIG",},
            {"key0": 3, "rig_id": "TEST_RIG",},
            ]
        )

    df_out = legacy_data.annotate_rig_id(df, pickled_data)

    pd.testing.assert_frame_equal(
        df_out,
        expected
    )

def test_annotate_startdatetime():

    df = pd.DataFrame(
        [
            {"test": None,},
        ],
    )

    pickled_data = {"startdatetime": TIMESTAMP_ISO, }

    expected = pd.DataFrame(
        [
            {"test": None, "startdatetime": TIMESTAMP,},
        ],
    )

    df_out = legacy_data.annotate_startdatetime(df, pickled_data)

    pd.testing.assert_frame_equal(
        df_out,
        expected,
        check_like=True,
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
        legacy_data.annotate_cumulative_reward(df, pickled_data),
        expected
    )

def test_annotate_filename():


    df = pd.DataFrame([{"test": "test"}])
    filename = "test/path/fname.py"

    expected = pd.DataFrame([{
        "test": "test",
        "filepath": "test/path",
        "filename": "fname.py",
        }])

    df_out = legacy_data.annotate_filename(df, filename)

    pd.testing.assert_frame_equal(
        df_out,
        expected,
        check_like=True,
    )


def test_fix_autorearded():

    df = pd.DataFrame(
        [
            {"auto_rearded": True},
            ]
        )

    expected = pd.DataFrame(
        [
            {"auto_rewarded": True},
            ]
        )

    df_out = legacy_data.fix_autorearded(df)

    pd.testing.assert_frame_equal(df_out, expected)


# @pytest.mark.parametrize("df, expected", [
#     (
#         pd.DataFrame(data={
#             "lick_times": {17: [0.1, 0.2, 0.3, 0.4, ], },
#         }),
#         pd.DataFrame(data={
#             "lick_times": {17: [0.1, 0.2, 0.3, 0.4, ], },
#             "number_of_licks": {17: 4, },
#             "lick_rate_Hz": {17: 10.0, },
#         }, columns=["lick_times", "number_of_licks", "lick_rate_Hz", ]),
#     ),
# ])
# def test_annotate_lick_vigor(df, expected):
#     pd.util.testing.assert_almost_equal(data.annotate_lick_vigor(df), expected)
