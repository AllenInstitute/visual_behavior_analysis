import pytest
import datetime
from six import b
import numpy as np
import pandas as pd
from collections import OrderedDict

from visual_behavior.translator.foraging2 import extract


@pytest.fixture
def foraging2_trial_fixture(foraging2_data_fixture):
    return extract.get_trial_log(foraging2_data_fixture)[0]


@pytest.fixture
def foraging2_stimuli_fixture():
    return {
        'size': (1174, 918),
        'sequence': None,
        'correct_table': None,
        'pos': (0, 0),
        'sampling': 'random',
        'log': [],
        'stim_groups': [
            ('im111', ['im111']),
            ('im008', ['im008']),
            ('im029', ['im029']),
            ('im040', ['im040']),
            ('im046', ['im046']),
            ('im083', ['im083']),
            ('im037', ['im037']),
            ('im053', ['im053'])],
            'on_draw': {},
            'fps': 60.0,
            'kwargs': {},
            'units': 'pix',
            'incorrect_table': None,
            'possibility_table': None,
            'image_path': '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_0_2017.07.14.pkl',
            'unpickleable': ['changeOccurred', 'flash_ended'],
            'update_count': 1703,
            'correct_freq': 0.5,
            'image_walk': [],
            'obj_text': '',
            'stimulus': "ImageStimNumpyuByte(autoLog=True, color=array([1., 1., 1.]), colorSpace='rgb', contrast=1.0, depth=0, flipHoriz=False, flipVert=True, image=ndarray(...), interpolate=False, mask=None, maskParams=None, name='unnamed ImageStimNumpyuByte', opacity=1.0, ori=0.0, pos=array([0., 0.]), size=array([1174.,  918.]), texRes=128, units='pix', win=Window(...))",
            'items': OrderedDict(),
            'param_names': None,
            'flash_interval_sec': (0.25, 0.5),
            'draw_log': [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            ],
    }




@pytest.mark.parametrize("data, expected", [
    ({"items": {"behavior": {"wut": "wew", }, }, }, {"wut": "wew", }, ),
    ({"wut": "wew", }, {"wut": "wew", }, ),
])
def test_behavior_items_or_top_level(data, expected):
    assert extract.behavior_items_or_top_level(data) == expected


def test_get_params(foraging2_data_fixture):
    assert extract.get_params(foraging2_data_fixture) == {
        'auto_reward_volume': 0.007,
        'catch_freq': 0.125,
        'failure_repeats': 5,
        'max_task_duration_min': 60.0,
        'nidevice': 'Dev1',
        'periodic_flash': (0.25, 0.5),
        'pre_change_time': 2.25,
        'response_window': (0.15, 2.0),
        'reward_volume': 0.007,
        'stimulus_window': 6.0,
        'sync_sqr': True,
        'task_id': 'DoC',
        'volume_limit': 1.5,
        'warm_up_trials': 0,
    }


def test_get_trials(foraging2_data_fixture):
    expected = pd.DataFrame(data=[{
        'index': 0,
        'cumulative_rewards': 1,
        'cumulative_volume': 0.008,
        'stimulus_changes': [
            (('im111', 'im111'), ('im037', 'im037'), 413, 12.00008911471998)
        ],
        'success': True,
        'licks': [
            (12.215056513123665, 426)
        ],
        'trial_params': {
            'catch': False,
            'auto_reward': False,
            'change_time': 4.32909793498625
        },
        'rewards': [
            (12.215641293879314, 426)
        ],
        'events': [
            ['initial_blank', 'enter', 5.056719305237289, 0],
            ['initial_blank', 'exit', 5.056973641969159, 0],
            ['pre_change', 'enter', 5.057221059856309, 0],
            ['pre_change', 'exit', 7.361711943586507, 135],
            ['stimulus_window', 'enter', 7.362480212104104, 135],
            ['stimulus_changed', '', 12.000634319683837, 414],
            ['response_window', 'enter', 12.165425149170293, 423],
            ['hit', '', 12.215603378610254, 426],
            ['stimulus_window', 'exit', 13.366606334340718, 495],
            ['no_lick', 'exit', 13.368076173712819, 495],
            ['response_window', 'exit', 14.01708539514906, 534]
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


def test_get_time(foraging2_data_fixture):
    np.testing.assert_almost_equal(
        extract.get_time(foraging2_data_fixture)[:10],
        np.array([
            0., 16.5396367, 33.2287204, 49.9249997, 66.5601165, 83.2115617,
            99.9150366, 116.5919431, 133.2821339, 150.0199263,
        ])
    )


def test_get_rewards(foraging2_data_fixture):
    """numpy array of [<time>, <frame number>]
    """
    expected = pd.DataFrame(data={
        "frame": np.array([426.0, 793.0, 1253.0, ]),
        "time": np.array([12.215641, 18.337148, 26.010165, ])
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
    np.testing.assert_almost_equal(
        extract.get_vsyncs(foraging2_data_fixture)[:10],
        np.array([
            16.5396367, 16.6890837, 16.6962793, 16.6351167, 16.6514452,
            16.7034749, 16.6769066, 16.6901907, 16.7377924, 16.6788438,
        ])
    )


def test_get_dx(foraging2_data_fixture):
    np.testing.assert_almost_equal(
        extract.get_dx(foraging2_data_fixture)[:10],
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )


def test_get_licks(foraging2_data_fixture):
    expected = pd.DataFrame(data={
        "frame": np.array([426, 793, 1253, ]),
        "time": np.array([7105.795778, 13227.606467, 20900.418728, ])
    })

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
            "time": np.array([0.0, 16.539636677407543, 33.22872040098446, 49.924999722861685, 66.56011645827675]),
            "speed (cm/s)": np.array([0, 0, 0, 0, 0]),
            "acceleration (cm/s^2)": np.array([0, 0, 0, 0, 0]),
            "jerk (cm/s^3)": np.array([0, 0, 0, 0, 0]),
        }),
    ),
])
def test_get_running_speed(foraging2_data_fixture, kwargs, expected):
    pd.testing.assert_frame_equal(
        extract.get_running_speed(foraging2_data_fixture, **kwargs).iloc[:5],
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_get_stimulus_log(foraging2_data_fixture):
    np.testing.assert_equal(
        extract.get_stimulus_log(foraging2_data_fixture)[:10],
        np.array([True, True, True, True, True, True, True, True, True, True, ])
    )


def test_get_datetime_info(foraging2_data_fixture):
    assert extract.get_datetime_info(foraging2_data_fixture) == \
        {
            "startdatetime": datetime.datetime(2018, 3, 9, 16, 6, 59, 199000),
            "year": 2018,
            "month": 3,
            "day": 9,
            "hour": 16,
            "date": datetime.date(2018, 3, 9),
            "dayofweek": 4,
        }


def test_get_mouse_id(foraging2_data_fixture):
    assert extract.get_mouse_id(foraging2_data_fixture) == "testmouse"


def test_get_blank_duration_range(foraging2_data_fixture):
    assert extract.get_blank_duration_range(foraging2_data_fixture) == \
        (0.5, 0.5, )


def test_get_user_id(foraging2_data_fixture):
    assert extract.get_user_id(foraging2_data_fixture) == None


def test_get_session_id(foraging2_data_fixture):
    assert extract.get_session_id(foraging2_data_fixture) is None


def test_get_filename(foraging2_data_fixture):
    assert extract.get_filename(foraging2_data_fixture) == ""


def test_get_device_name(foraging2_data_fixture):
    assert extract.get_device_name(foraging2_data_fixture) is None


def test_get_session_duration(foraging2_data_fixture):
    assert extract.get_session_duration(foraging2_data_fixture) == 28406.585909910973


def test_get_stimulus_duration(foraging2_data_fixture):
    assert extract.get_stimulus_duration(foraging2_data_fixture) == 6.0


def test_get_stimulus_on_frames(foraging2_data_fixture, foraging2_stimuli_fixture):
    assert extract.get_stimulus_on_frames(foraging2_data_fixture).equals(
        pd.Series([stim_on == 1 for stim_on in foraging2_stimuli_fixture["draw_log"]])
    )


def test_get_stimuli(foraging2_data_fixture, foraging2_stimuli_fixture):
    assert extract.get_stimuli(foraging2_data_fixture) == \
        foraging2_stimuli_fixture


def test_annotate_licks(foraging2_data_fixture):
    assert extract.annotate_licks(extract.get_trial_log(foraging2_data_fixture)[0]) == {
        "lick_times": [12.215056513123665],
        "lick_frames": [426],
    }


def test_annotate_optogenetics(foraging2_data_fixture):
    assert extract.annotate_optogenetics(extract.get_trial_log(foraging2_data_fixture)[0]) == \
        {"optogenetics": None, }


def test_annotate_responses(foraging2_trial_fixture):
    assert extract.annotate_responses(foraging2_trial_fixture) == \
        {
            'response_frame': 426,
            'response_time': 12.215603378610254,
            'response_latency': 0.21551426389027384,
            'response_type': None
        }


def test_annotate_rewards(foraging2_trial_fixture):
    assert extract.annotate_rewards(foraging2_trial_fixture) == \
        {
            'auto_rewarded_trial': False,
            'cumulative_volume': 0.008,
            'cumulative_reward_number': 1,
            'reward_volume': None,
            'reward_times': [12.215641293879314],
            'reward_frames': [426],
            'rewarded': True
        }


def test_annotate_schedule_time(foraging2_data_fixture, foraging2_trial_fixture):
    pre_change_time = extract.get_pre_change_time(foraging2_data_fixture)
    assert extract.annotate_schedule_time(foraging2_trial_fixture, pre_change_time) == \
        {
            'scheduled_change_time': None,
            'start_frame': 0,
            'start_time': 5.056719305237289,
            'end_frame': 534,
            'end_time': 14.01708539514906,
        }


def test_annotate_stimuli(foraging2_trial_fixture, foraging2_stimuli_fixture):
    assert extract.annotate_stimuli(foraging2_trial_fixture, foraging2_stimuli_fixture) == {
        'initial_image_category': 'im111',
        'initial_image_name': 'im111',
        'change_image_name': 'im037',
        'change_image_category': 'im037',
        'change_frame': 413,
        'change_time': 12.00008911471998,
        'change_orientation': None,
        'change_contrast': None,
        'initial_orientation': None,
        'initial_contrast': None,
        "delta_orientation": None,
        "stimulus_on_frames": [
            True, True, True, True, True, True, True, True, True, True, True,
            True, True, True, True, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, True, True, True, True,
            True, True, True, True, True, True, True, True, True, True, True,
            True, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, True, True, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, True, True, True,
            True, True, True, True, True, True, True, True, True, True, True,
            True, True, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, True, True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, True, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, True, True, True, True,
            True, True, True, True, True, True, True, True, True, True, True,
            True, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, True, True, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, True, True, True,
            True, True, True, True, True, True, True, True, True, True, True,
            True, True, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, True, True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, True, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, True, True,
            True, True, True, True, True, True, True, True, True, True, True,
            True, True, True, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, True, True, True, True, True,
            True, True, True, True, True, True, True, True, True, True, True,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, True, True, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False,
        ],
    }


def test_annotate_trials(foraging2_trial_fixture):
    assert extract.annotate_trials(foraging2_trial_fixture) == {
        "trial_type": "go",
        "trial_duration": 8.960366089911771,
    }


def test_get_pre_change_time(foraging2_data_fixture):
    assert extract.get_pre_change_time(foraging2_data_fixture) == 2.25


def test_get_task_id(foraging2_data_fixture):
    assert extract.get_task_id(foraging2_data_fixture) == "DoC"


def test_get_image_path(foraging2_data_fixture):
    assert extract.get_image_path(foraging2_data_fixture) == "//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_0_2017.07.14.pkl"


def test_get_response_window(foraging2_data_fixture):
    assert extract.get_response_window(foraging2_data_fixture) == (0.15, 2.0, )
