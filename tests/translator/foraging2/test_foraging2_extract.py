import pytest
import datetime
import numpy as np
import pandas as pd
from six import b, iteritems
from collections import OrderedDict

from visual_behavior.translator.foraging2 import extract


@pytest.fixture
def foraging2_trial_fixture(foraging2_data_fixture):
    return extract.get_trial_log(foraging2_data_fixture)[0]


@pytest.fixture
def foraging2_stimuli_fixture():
    return OrderedDict([
        ('gratings', {
            'sequence': None,
            'correct_table': None,
            'pos': (0, 0),
            'change_log': [
                (('group0', None), ('group1', 90), 6.511695924235673, 183),
                (('group1', None), ('group0', 0), 17.551550240638086, 873),
                (('group0', None), ('group1', 90), 25.647350767025248, 1379)
            ],
            'incorrect_table': None,
            'obj_type': 'DoCGratingStimulus',
            'log': [],
            'stim_groups': OrderedDict([
                ('group0', ('Ori', [0])),
                ('group1', ('Ori', [90]))
            ]),
            'on_draw': {},
            'fps': 60.0,
            'kwargs': {},
            'units': 'deg',
            'size': None,
            'possibility_table': None,
            'draw_log': [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1
            ],
            'unpickleable': ['changeOccurred', 'flash_ended'],
            'set_log': [
                ('Ori', 0, 0, 0, ),
                ('Ori', 90, 6.509936595531521, 183, ),
                ('Ori', 0, 17.550367668425903, 873, ),
                ('Ori', 90, 25.645875388494208, 1379, )
            ],
            'update_count': 1702,
            'correct_freq': 0.5,
            'obj_text': '',
            'stimulus': {
                'pos': (0, 0)
            },
            'items': OrderedDict(),
            'param_names': None,
            'flash_interval_sec': (0.25, 0.5)
        })
    ])


@pytest.mark.parametrize("data, expected", [
    ({"items": {"behavior": {"wut": "wew", }, }, }, {"wut": "wew", }, ),
    ({"wut": "wew", }, {"wut": "wew", }, ),
])
def test_behavior_items_or_top_level(data, expected):
    assert extract.behavior_items_or_top_level(data) == expected


def test_get_params(foraging2_data_fixture):
    assert extract.get_params(foraging2_data_fixture) == {
        'auto_reward_volume': 0.007,
        'catch_freq': 0.5,
        'failure_repeats': 5,
        'max_task_duration_min': 60.0,
        'nidevice': 'Dev1',
        'periodic_flash': (0.25, 0.5),
        'pre_change_time': 2.25,
        'response_window': (0.15, 2.0),
        'reward_volume': 0.007,
        'stimulus_window': 6.0,
        'task_id': 'DoC',
        'volume_limit': 1.5,
        'warm_up_trials': 1,
        'trial_translator': True,
    }


def test_get_trials(foraging2_data_fixture):
    expected = pd.DataFrame(data=[{
        'index': 0,
        'cumulative_rewards': 1,
        'cumulative_volume': 0.008,
        'stimulus_changes': [
            (('group0', None), ('group1', 90), 6.511695924235673, 183)
        ],
        'success': True,
        'licks': [
            (6.7089195225430425, 196)
        ],
        'trial_params': {
            'catch': False,
            'auto_reward': True,
            'change_time': 0.0960600196801137
        },
        'rewards': [
            (0.007, 6.513225269904879, 184)
        ],
        'events': [
            ["initial_blank", "enter", 3.564911950538426, 0],
            ["initial_blank", "exit", 3.5651510664736765, 0],
            ["pre_change", "enter", 3.5653807727540676, 0],
            ["pre_change", "exit", 5.816857950189715, 140],
            ["stimulus_window", "enter", 5.817755741965123, 140],
            ["stimulus_changed", "", 6.512486888752972, 184],
            ["auto_reward", "", 6.51318873830366, 184],
            ["response_window", "enter", 6.662831309796281, 193],
            ["response_window", "exit", 8.516887009417957, 309],
            ["stimulus_window", "exit", 11.82326398786708, 516],
            ["no_lick", "exit", 11.824898500267068, 516]
        ],
    }, ])

    pd.testing.assert_frame_equal(
        extract.get_trials(foraging2_data_fixture).iloc[:1],
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_get_time(monkeypatch, foraging2_data_fixture):
    monkeypatch.setattr(
        extract,
        "get_vsyncs",
        lambda data: [16] * data["items"]["behavior"]["update_count"]
    )

    np.testing.assert_almost_equal(
        extract.get_time(foraging2_data_fixture)[:10],
        np.array([0.0 , 0.016, 0.032, 0.048, 0.064, 0.08, 0.096, 0.112, 0.128, 0.144])
    )


def test_get_rewards(foraging2_data_fixture):
    """numpy array of [<time>, <frame number>]
    """
    expected = pd.DataFrame(data={
        "frame": np.array([184.0, 886.0, 1392.0, ]),
        "time": np.array([6.513225, 17.755239, 25.846743, ])
    })

    pd.testing.assert_frame_equal(
        extract.get_rewards(foraging2_data_fixture),
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_get_vsyncs(foraging2_data_fixture):
    assert list(extract.get_vsyncs(foraging2_data_fixture)[:10]) == \
        [None, None, None, None, None, None, None, None, None, None]


def test_get_dx(foraging2_data_fixture):
    np.testing.assert_almost_equal(
        extract.get_dx(foraging2_data_fixture)[:10],
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )


def test_get_licks(monkeypatch, foraging2_data_fixture):
    monkeypatch.setattr(
        extract,
        "get_vsyncs",
        lambda data: [16] * data["items"]["behavior"]["update_count"]
    )

    expected = pd.DataFrame(data={
        "frame": np.array([]),
        "time": np.array([]),
    })  # the test data doesnt have lick sensor data so i guess this is more of a smoke test for now...?

    pd.testing.assert_frame_equal(
        extract.get_licks(foraging2_data_fixture),
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


@pytest.mark.parametrize("kwargs, expected", [
    (
        {},
        pd.DataFrame(data={
            "time": np.array([0.0, 0.016, 0.032, 0.048, 0.064, ]),
            "speed (cm/s)": np.array([0, 0, 0, 0, 0, ]),
            "acceleration (cm/s^2)": np.array([0, 0, 0, 0, 0, ]),
            "jerk (cm/s^3)": np.array([0, 0, 0, 0, 0, ]),
        }),
    ),
])
def test_get_running_speed(monkeypatch, foraging2_data_fixture, kwargs, expected):
    monkeypatch.setattr(
        extract,
        "get_vsyncs",
        lambda data: [16] * (data["items"]["behavior"]["update_count"] - 1)  # n vsyncs should be update_count - 1
    )

    pd.testing.assert_frame_equal(
        extract.get_running_speed(foraging2_data_fixture, **kwargs).iloc[:5],
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_get_stimulus_log(foraging2_data_fixture):
    expected = {
        "gratings": np.array([
            True, True, True, True, True,
            True, True, True, True, True,
        ]),
    }

    actual = extract.get_stimulus_log(foraging2_data_fixture)

    for k, v in iteritems(expected):
        np.testing.assert_equal(actual[k][:10], v[:10])  # not amazing but whatever...


def test_get_datetime_info(foraging2_data_fixture):
    assert extract.get_datetime_info(foraging2_data_fixture) == \
        {
            "startdatetime": datetime.datetime(2018, 4, 4, 15, 20, 53, 665000),
            "year": 2018,
            "month": 4,
            "day": 4,
            "hour": 15,
            "date": datetime.date(2018, 4, 4),
            "dayofweek": 2,
        }


def test_get_mouse_id(foraging2_data_fixture):
    assert extract.get_mouse_id(foraging2_data_fixture) == "testmouse"


def test_get_blank_duration_range(foraging2_data_fixture):
    assert extract.get_blank_duration_range(foraging2_data_fixture) == \
        (0.5, 0.5, )


def test_get_user_id(foraging2_data_fixture):
    assert extract.get_user_id(foraging2_data_fixture) == ''


def test_get_session_id(foraging2_data_fixture):
    assert extract.get_session_id(foraging2_data_fixture) is None


def test_get_filename(foraging2_data_fixture):
    assert extract.get_filename(foraging2_data_fixture) == ""


def test_get_device_name(foraging2_data_fixture):
    assert extract.get_device_name(foraging2_data_fixture) == 'localhost'


def test_get_session_duration(monkeypatch, foraging2_data_fixture):
    monkeypatch.setattr(
        extract,
        "get_vsyncs",
        lambda data: [16] * data["items"]["behavior"]["update_count"]
    )

    assert extract.get_session_duration(foraging2_data_fixture) == 27.232


def test_get_stimulus_duration(foraging2_data_fixture):
    assert extract.get_stimulus_duration(foraging2_data_fixture) == 6.0


def test_get_stimuli(foraging2_data_fixture, foraging2_stimuli_fixture):
    assert extract.get_stimuli(foraging2_data_fixture) == \
        foraging2_stimuli_fixture


def test_annotate_licks(foraging2_trial_fixture):
    assert extract.annotate_licks(foraging2_trial_fixture) == {
        "lick_times": [6.7089195225430425],
    }


def test_annotate_optogenetics(foraging2_data_fixture):
    assert extract.annotate_optogenetics(extract.get_trial_log(foraging2_data_fixture)[0]) == \
        {"optogenetics": None, }


def test_annotate_responses(foraging2_data_fixture):
    trial = extract.get_trial_log(foraging2_data_fixture)[1]  # cannot use the first trial because it is a warm up

    assert extract.annotate_responses(trial) == \
        {
            'response_time': [],
            'response_latency': 0.2031074001400377,
            'response_type': [],
            'change_time': 17.552098214656368,
            'change_frame': 874,
        }


def test_annotate_rewards(foraging2_trial_fixture):
    assert extract.annotate_rewards(foraging2_trial_fixture) == \
        {
            'auto_rewarded_trial': True,
            'cumulative_volume': 0.008,
            'cumulative_reward_number': 1,
            'reward_volume': 0.007,
            'reward_times': [6.513225269904879],
            'reward_frames': [184],
            'rewarded': True
        }


def test_annotate_schedule_time(foraging2_data_fixture, foraging2_trial_fixture):
    pre_change_time = extract.get_pre_change_time(foraging2_data_fixture)
    assert extract.annotate_schedule_time(foraging2_trial_fixture, pre_change_time, 0) == \
        {
            'scheduled_change_time': 2.3460600196801136,
            'start_frame': 0,
            'start_time': 3.564911950538426,
            'end_frame': 516,
            'end_time': 11.824898500267068,
            'trial_length': 8.25998654972864,
        }


def test_annotate_stimuli(foraging2_trial_fixture, foraging2_stimuli_fixture):
    annotated_stimuli = extract.annotate_stimuli(
        foraging2_trial_fixture,
        foraging2_stimuli_fixture
    )

    assert annotated_stimuli == {
        'initial_image_category': None,
        'initial_image_name': None,
        'change_image_name': None,
        'change_image_category': None,
        'change_orientation': 90,
        'change_contrast': None,
        'initial_orientation': 0,
        'initial_contrast': None,
        "delta_orientation": 90,
    }


def test__get_trial_frame_bounds(foraging2_trial_fixture):
    assert extract._get_trial_frame_bounds(foraging2_trial_fixture) == \
        (0, 516, )


def test__resolve_stimulus_dict(foraging2_stimuli_fixture):
    assert extract._resolve_stimulus_dict(foraging2_stimuli_fixture, "group0") == \
        ("gratings", foraging2_stimuli_fixture["gratings"], )


def test__get_stimulus_attr_changes(foraging2_stimuli_fixture):
    assert extract._get_stimulus_attr_changes(foraging2_stimuli_fixture["gratings"], 183, 0, 516) == \
        ({"ori": 0, }, {'ori': 90, }, )


def test_get_pre_change_time(foraging2_data_fixture):
    assert extract.get_pre_change_time(foraging2_data_fixture) == 2.25


def test_get_task_id(foraging2_data_fixture):
    assert extract.get_task_id(foraging2_data_fixture) == "DoC"


def test_get_image_path(foraging2_data_fixture):
    assert extract.get_image_path(foraging2_data_fixture) == {'gratings': None}


def test_get_response_window(foraging2_data_fixture):
    assert extract.get_response_window(foraging2_data_fixture) == (0.15, 2.0, )


def test_get_unified_draw_log(foraging2_data_fixture):
    np.testing.assert_almost_equal(
        extract.get_unified_draw_log(foraging2_data_fixture)[:20],
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    )  # testing the first 20 items should be sufficient


def test_get_initial_blank_duration(foraging2_data_fixture):
    assert extract.get_initial_blank_duration(foraging2_data_fixture) == 0.0


@pytest.mark.parametrize("mock_data, expected, exception_type", [
    (
        {"items": {"behavior": {"params": {"task_id": "wut", }, }, }, },
        "wut",
        None,
    ),
    (
        {"items": {"behavior": {"params": {}, }, }, },
        None,
        None,
    ),
    (
        {"items": {"behavior": {}, }, },
        None,
        KeyError,
    ),
])
def test_regression_get_task_id(mock_data, expected, exception_type):
    if exception_type is None:
        assert extract.get_task_id(mock_data) == expected
    else:
        pytest.raises(exception_type, extract.get_task_id, mock_data)


@pytest.mark.parametrize("mock_data, expected, exception_type", [
    (
        {"items": {"behavior": {"params": {"stage": "wut", }, }, }, },
        "wut",
        None,
    ),
    (
        {"items": {"behavior": {"params": {}, }, }, },
        None,
        None,
    ),
    (
        {"items": {"behavior": {}, }, },
        None,
        KeyError,
    ),
])
def test_get_stage(mock_data, expected, exception_type):
    if exception_type is None:
        assert extract.get_stage(mock_data) == expected
    else:
        pytest.raises(exception_type, extract.get_stage, mock_data)


@pytest.mark.parametrize("start_frame, expected", [
    (183, ('natual_scenes', 'im065', 'im065'), ),
    (184, ('natual_scenes', 'im065', 'im065'), ),
    (874, ('natual_scenes', 'im077', 'im077'), ),
    (900, ('natual_scenes', 'im077', 'im077'), ),
    (1380, ('natual_scenes', 'im065', 'im065'), ),
])
def test__resolve_initial_image(foraging2_data_fixture_issue_73, start_frame, expected):
    stimuli = extract.get_stimuli(foraging2_data_fixture_issue_73)
    assert extract._resolve_initial_image(stimuli, start_frame) == expected


@pytest.mark.regression
def test_regression_annotate_stimuli_natural_scenes(foraging2_natural_scenes_data_fixture):
    annotated_stimuli = extract.annotate_stimuli(
        extract.get_trial_log(foraging2_natural_scenes_data_fixture)[1],
        extract.get_stimuli(foraging2_natural_scenes_data_fixture)
    )

    assert annotated_stimuli == {
        'initial_image_category': 'im008',
        'initial_image_name': 'im008',
        'change_image_name': 'im008',
        'change_image_category': 'im008',
        'change_orientation': None,
        'change_contrast': None,
        'initial_orientation': None,
        'initial_contrast': None,
        "delta_orientation": None,
    }


def test_get_reward_volume(foraging2_data_fixture):
    assert extract.get_reward_volume(foraging2_data_fixture) == 0.007


def test_get_auto_reward_volume(foraging2_data_fixture):
    assert extract.get_auto_reward_volume(foraging2_data_fixture) == 0.007


@pytest.mark.regression
def test_regression_annotate_stimuli_73(foraging2_data_fixture_issue_73):
    stimuli = extract.get_stimuli(foraging2_data_fixture_issue_73)
    trials = extract.get_trials(foraging2_data_fixture_issue_73).copy(deep=True)

    non_image_trial_count = 0

    for _, trial in trials.iterrows():
        trial_stim = extract.annotate_stimuli(trial, stimuli)
        non_image_trial_count += trial_stim["initial_image_name"] is None or \
            trial_stim["change_image_name"] is None or \
            trial_stim["initial_image_category"] is None or \
            trial_stim["change_image_category"] is None

    assert non_image_trial_count == 0


@pytest.mark.regression
@pytest.mark.parametrize("args, expected", [
    (
        (np.nan, 0, 881, ),
        ({"ori": 0, }, {}, ),
    ),
    (
        (12327, 12140, 12655, ),
        ({"ori": 90, }, {"ori": 0, }, ),
    ),
])
def test_regression__get_stimulus_attr_changes(
        foraging2_data_fixture_issue_116, args, expected
):
    stimuli = extract.get_stimuli(foraging2_data_fixture_issue_116)
    trials = extract.get_trials(foraging2_data_fixture_issue_116).copy(deep=True)

    assert extract._get_stimulus_attr_changes(stimuli["grating"], *args) == \
        expected



@pytest.mark.regression
def test_regression_annotate_stimuli_116(foraging2_data_fixture_issue_116):
    stimuli = extract.get_stimuli(foraging2_data_fixture_issue_116)
    trials = extract.get_trials(foraging2_data_fixture_issue_116).copy(deep=True)

    non_ori_trial_count = 0

    for _, trial in trials.iterrows():
        trial_stim = extract.annotate_stimuli(trial, stimuli)
        initial_ori = trial_stim["initial_orientation"]
        change_ori = trial_stim["change_orientation"]

        non_ori_trial_count += initial_ori is None or np.isnan(initial_ori) or \
            change_ori is None or np.isnan(change_ori)

    assert non_ori_trial_count == 0


@pytest.mark.parametrize("stim_dict, args, expected", [
    (
        {
            "set_log": [
                ('Ori', 0, 0.6300831158435708, 0),
                ('Contrast', 0.5, 20.123506742232717, 1218),
                ('Ori', 90, 197.87752266590869, 12327),
            ],
        },
        (12327, 12140, 12655, ),
        ({"ori": 0, "contrast": 0.5, }, {"ori": 90, }, ),
    ),
])
def test__get_stimulus_attr_changes_many_changes(stim_dict, args, expected):
    assert extract._get_stimulus_attr_changes(stim_dict, *args) == expected


def test_get_warm_up_trials(foraging2_data_stage4_2018_05_10):
    assert extract.get_warm_up_trials(foraging2_data_stage4_2018_05_10) == 5


def test_get_stimulus_window(foraging2_data_stage4_2018_05_10):
    assert extract.get_stimulus_window(foraging2_data_stage4_2018_05_10) == 6.0


def test_get_volume_limit(foraging2_data_stage4_2018_05_10):
    assert extract.get_volume_limit(foraging2_data_stage4_2018_05_10) == 1.5


def test_get_failure_repeats(foraging2_data_stage4_2018_05_10):
    assert extract.get_failure_repeats(foraging2_data_stage4_2018_05_10) == 5


def test_get_catch_frequency(foraging2_data_stage4_2018_05_10):
    assert extract.get_catch_frequency(foraging2_data_stage4_2018_05_10) == None


def test_get_free_reward_trials(foraging2_data_stage4_2018_05_10):
    assert extract.get_free_reward_trials(foraging2_data_stage4_2018_05_10) == 10


def test_get_min_no_lick_time(foraging2_data_stage4_2018_05_10):
    assert extract.get_min_no_lick_time(foraging2_data_stage4_2018_05_10) == 0.0


def test_get_max_session_duration(foraging2_data_stage4_2018_05_10):
    assert extract.get_max_session_duration(foraging2_data_stage4_2018_05_10) == 60.0


def test_get_abort_on_early_response(foraging2_data_stage4_2018_05_10):
    assert extract.get_abort_on_early_response(foraging2_data_stage4_2018_05_10) == True
