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


def test_get_params(pizza_data_fixture):
    assert extract.get_params(pizza_data_fixture) == {
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
        extract.get_trials(pizza_data_fixture).iloc[:1],
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
        extract.get_time(pizza_data_fixture)[:10],
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
        extract.get_rewards(pizza_data_fixture).iloc[:5],
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
        extract.get_vsyncs(pizza_data_fixture)[:10],
        expected
    )


def test_get_dx(pizza_data_fixture):
    expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    np.testing.assert_almost_equal(
        extract.get_dx(pizza_data_fixture)[:10],
        expected
    )


def test_get_licks(pizza_data_fixture):
    expected = pd.DataFrame(data={
        "frame": np.array([25, 844, 1074, 1257, 1488, ]),
        "time": np.array([416.96787951514125, 14128.407060168684, 17981.629601912573, 21034.25351413898, 24904.09280906897])
    })

    pd.testing.assert_frame_equal(
        extract.get_licks(pizza_data_fixture).iloc[:5],
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
        extract.get_running_speed(pizza_data_fixture, **kwargs).iloc[:5],
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_get_stimulus_log(pizza_data_fixture):
    np.testing.assert_equal(
        extract.get_stimulus_log(pizza_data_fixture)[:10],
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
