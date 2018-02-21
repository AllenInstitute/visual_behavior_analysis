import pytest
from six import b
import numpy as np
import pandas as pd

from visual_behavior import processing


@pytest.mark.parametrize("data, expected", [
    ({"items": {"behavior": {"wut": "wew", }, }, }, {"wut": "wew", }, ),
    ({"wut": "wew", }, {"wut": "wew", }, ),
])
def test_behavior_items_or_top_level(data, expected):
    assert processing.behavior_items_or_top_level(data) == expected


def test_get_params(pizza_data_fixture):
    assert processing.get_params(pizza_data_fixture) == {
        'auto_reward_volume': 0.007,
        'catch_freq': 0.125,
        'failure_repeats': 5,
        'max_task_duration_min': 60.0,
        'mouse_id': 'M347745',
        'nidevice': b'Dev1',
        'periodic_flash': (0.25, 0.5),
        'pre_change_time': 2.25,
        'response_window': (0.15, 0.75),
        'reward_volume': 0.007,
        'script': 'c:\\users\\svc_nc~1\\appdata\\local\\temp\\tmpulwuhh\\M347745 - spmuoq.py',
        'stimulus_window': 6.0,
        'task_id': b'DoC',
        'trial_translator': True,
        'user_id': 'kylam',
        'volume_limit': 5,
        'warm_up_trials': 5,
    }


def test_get_trials(pizza_data_fixture):
    expected = pd.DataFrame([
        {
            'index': 0,
            'cumulative_rewards': 0,
            'cumulative_volume': 0.0,
            'stimulus_changes': [],
            'success': False,
            'licks': [(7.934872470921013, 25)],
            'trial_params': {
                'catch': False,
                'auto_reward': True,
                'change_time': 0.7802031782835195
            },
            'rewards': [],
            'events': [
                ['initial_blank', 'enter', 7.513004185308317, 0],
                ['initial_blank', 'exit', 7.5131104567819404, 0],
                ['pre_change', 'enter', 7.513202823576774, 0],
                ['early_response', '', 7.935050252171282, 25],
                ['abort', '', 7.935161158537837, 25],
                ['timeout', 'enter', 7.935301198517099, 25],
                ['timeout', 'exit', 7.935412104883654, 25]
            ]
        }
    ])

    pd.testing.assert_frame_equal(
        processing.get_trials(pizza_data_fixture).iloc[:1],
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_get_time(pizza_data_fixture):
    expected = np.array([
         0. , 16.7038231, 33.3285219, 50.0144676, 66.6954472, 83.341334,
         100.0696558, 116.7347444, 133.4120824, 150.1053114,
    ])

    np.testing.assert_almost_equal(
        processing.get_time(pizza_data_fixture)[:10],
        expected
    )


def test_get_rewards(pizza_data_fixture):
    """numpy array of [<time>, <frame number>]
    """
    expected = pd.DataFrame(data={
        "frame": np.array([230.0, 828.0, 1472.0, 1932.0, 2622.0, ]),
        "time": np.array([11.338415383077212, 21.346828706937707, 32.12227642104162, 39.82892841937571, 51.37191175428711])
    })

    pd.testing.assert_frame_equal(
        processing.get_rewards(pizza_data_fixture).iloc[:5],
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_get_vsyncs(pizza_data_fixture):
    expected = np.array([
        16.7038231, 16.6246988, 16.6859457, 16.6809796, 16.6458868, 16.7283218,
        16.6650886, 16.677338 , 16.693229 , 16.688263
    ])

    np.testing.assert_almost_equal(
        processing.get_vsyncs(pizza_data_fixture)[:10],
        expected
    )


def test_get_dx(pizza_data_fixture):
    expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    np.testing.assert_almost_equal(
        processing.get_dx(pizza_data_fixture)[:10],
        expected
    )


def test_get_licks(pizza_data_fixture):
    expected = pd.DataFrame(data={
        "frame": np.array([25, 844, 1074, 1257, 1488, ]),
        "time": np.array([416.96787951514125, 14128.407060168684, 17981.629601912573, 21034.25351413898, 24904.09280906897])
    })

    pd.testing.assert_frame_equal(
        processing.get_licks(pizza_data_fixture).iloc[:5],
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
            "time": np.array([0.0, 16.70382311567664, 33.328521880321205, 50.01446756068617, 66.69544719625264]),
            "speed (cm/s)": np.array([0, 0, 0, 0, 0]),
            "acceleration (cm/s^2)": np.array([0, 0, 0, 0, 0]),
            "jerk (cm/s^3)": np.array([0, 0, 0, 0, 0]),
        }),
    ),
])
def test_get_running_speed(pizza_data_fixture, kwargs, expected):
    pd.testing.assert_frame_equal(
        processing.get_running_speed(pizza_data_fixture, **kwargs).iloc[:5],
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_get_stimulus_log(pizza_data_fixture):
    np.testing.assert_equal(
        processing.get_stimulus_log(pizza_data_fixture)[:10],
        np.array([True, True, True, True, True, True, True, True, True, True, ])
    )
