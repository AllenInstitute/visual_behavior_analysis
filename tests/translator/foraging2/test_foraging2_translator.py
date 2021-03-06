import pytest
import datetime
import pandas as pd
import numpy as np
from uuid import UUID

from visual_behavior.translator import foraging2


EXPECTED_TRIALS = pd.DataFrame(data={
    "auto_rewarded": {0: True, 1: False, 2: False, },
    "change_contrast": {0: None, 1: None, 2: None, },
    "change_frame": {0: 184, 1: 874, 2: 1380, },
    "change_image_category": {0: '', 1: '', 2: '', },
    "change_image_name": {0: '', 1: '', 2: '', },
    "change_ori": {0: 90, 1: 0, 2: 90, },
    "change_time": {
        0: 6.511695924235673,
        1: 17.551550240638086,
        2: 25.647350767025248,
    },
    "cumulative_reward_number": {0: 1, 1: 2, 2: 3, },
    "cumulative_volume": {0: 0.008, 1: 0.016, 2: 0.024, },
    "delta_ori": {0: 90.0, 1: -90.0, 2: 90.0, },
    "initial_contrast": {0: None, 1: None, 2: None, },
    "initial_image_category": {0: '', 1: '', 2: '', },
    "initial_image_name": {0: '', 1: '', 2: '', },
    "initial_ori": {0: 0, 1: 90.0, 2: 0.0, },
    # "lick_frames": {0: [196], 1: [886], 2: [1392], },
    "lick_times": {
        0: [6.7089195225430425],
        1: [17.754299520384357],
        2: [25.84577631036363],
    },
    "optogenetics": {0: False, 1: False, 2: False, },
    # "response_frame": {0: np.nan, 1: 886, 2: 1392, },
    "response_latency": {
        0: np.inf,
        1: 0.2031074001400377,
        2: 0.19859408686218671,
    },
    "response_time": {
        0: np.nan,
        1: 17.755205614796406,
        2: 25.846704268385498,
    },
    "response_type": {0: [], 1: [], 2: [], },
    "reward_frames": {0: [184], 1: [886], 2: [1392], },
    "reward_times": {
        0: [6.513225269904879],
        1: [17.755205614796406],
        2: [25.846704268385498],
    },
    "reward_volume": {0: 0.007, 1: 0.008, 2: 0.008, },
    "rewarded": {0: True, 1: True, 2: True, },
    "scheduled_change_time": {
        0: 2.346,
        1: 4.437,
        2: 4.1706,
    },
    "startframe": {0: 0, 1: 552, 2: 1104, },
    "starttime": {
        0: 3.564911950538426,
        1: 12.401147424383737,
        2: 21.23261967558831,
    },
    "endtime": {
        0: 11.824898500267068,
        1: 20.66769388732215,
        2: 29.50127058016057,
    },
    "endframe": {0: 516, 1: 1068, 2: 1620, },
    "trial_length": {
        0: 8.25998654972864,
        1: 8.266546462938415,
        2: 8.268650904572262,
    },
    "index": {0: 0, 1: 1, 2:2, },
})


def test_data_to_metadata(monkeypatch, foraging2_data_fixture):
    monkeypatch.setattr(
        foraging2.extract,
        "get_vsyncs",
        lambda data: [16] * data["items"]["behavior"]["update_count"]
    )

    foraging2_metadata = foraging2.data_to_metadata(foraging2_data_fixture)

    assert foraging2.data_to_metadata(foraging2_data_fixture) == \
        {
            'startdatetime': '2018-04-04T15:20:53.665000-07:00',
            'rig_id': 'localhost',
            'computer_name': 'localhost',
            'reward_vol': 0.007,
            'rewardvol': 0.007,
            'auto_reward_vol': 0.007,
            'params': {
                'reward_volume': 0.007,
                'auto_reward_volume': 0.007,
                'task_id': 'DoC',
                'failure_repeats': 5,
                'volume_limit': 1.5,
                'catch_freq': 0.5,
                'max_task_duration_min': 60.0,
                'periodic_flash': (0.25, 0.5),
                'nidevice': 'Dev1',
                'stimulus_window': 6.0,
                'pre_change_time': 2.25,
                'response_window': [0.15, 2.0],
                'trial_translator': True,
                'warm_up_trials': 1
            },
            'mouseid': 'testmouse',
            'response_window': [0.15, 2.0],
            'task': 'DoC',
            'stage': None,
            'stoptime': 27.232,
            'userid': '',
            'lick_detect_training_mode': 'single',
            'blankscreen_on_timeout': False,
            'stim_duration': 6000.0,
            'blank_duration_range': [0.5, 0.5, ],
            'delta_minimum': 2.25,
            'stimulus_distribution': 'exponential',
            'delta_mean': 2.0,
            'trial_duration': None,
            'n_stimulus_frames': 592,
            'stimulus': 'gratings',
            "warm_up_trials": 1,
            "stimulus_window": 6.0,
            "volume_limit": 1.5,
            "failure_repeats": 5,
            "catch_frequency": 0.5,
            "free_reward_trials": 10,
            "min_no_lick_time": 0.0,
            "max_session_duration": 60.0,
            "abort_on_early_response": None,
            "initial_blank_duration": 0.0,
            'even_sampling_enabled': False,
            "behavior_session_uuid": UUID('{30ff3fba-f349-40c3-a43f-d8c898b7670d}'),
            "abort_on_early_response": True,
            "auto_reward_delay": 0.0,
            "periodic_flash": (0.25, 0.5),
            "platform_info": {
                'camstim': '0.2.5',
                'camstim_git_hash': '4bca34e',
                'computer_name': 'localhost',
                'hardware': ('Intel64 Family 6 Model 62 Stepping 4, GenuineIntel', 'AMD64'),
                'opengl': '4.4.0',
                'os': ('Windows', '7', '6.1.7601'),
                'psychopy': '1.82.01',
                'pyglet': '1.2.4',
                'python': '2.7.13',
            }
        }


def test_data_to_rewards(foraging2_data_fixture):
    expected = pd.DataFrame(data={
        "frame": {0: 184, 1: 886, 2: 1392, },
        "time": {0: 6.513225269904879, 1: 17.755238825342968, 2: 25.84674329077771, },
        "volume": {0: 0.007, 1: 0.008, 2: 0.008, },
        "lickspout": {0: None, 1: None, 2: None, },
    })

    pd.testing.assert_frame_equal(
        foraging2.data_to_rewards(foraging2_data_fixture),
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_data_to_running(monkeypatch, foraging2_data_fixture):
    monkeypatch.setattr(
        foraging2.extract,
        "get_vsyncs",
        lambda data: [16] * (data["items"]["behavior"]["update_count"] - 1)  # n vsyncs should be update_count - 1
    )

    expected = pd.DataFrame(data={
        "speed": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, },
        "time": {0: 0.000000, 1: 0.016, 2: 0.032, 3: 0.048, 4: 0.064, },
        "frame": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, },
        "dx": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, },
        "v_in": {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, },
        "v_sig": {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, },
    })

    pd.testing.assert_frame_equal(
        foraging2.data_to_running(foraging2_data_fixture).iloc[:5],
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_data_to_time(monkeypatch, foraging2_data_fixture):
    monkeypatch.setattr(
        foraging2.extract,
        "get_vsyncs",
        lambda data: [16] * (data["items"]["behavior"]["update_count"] - 1)  # n vsyncs should be update_count - 1
    )

    np.testing.assert_almost_equal(
        foraging2.data_to_time(foraging2_data_fixture)[:10],
        np.array([0., 0.016, 0.032, 0.048, 0.064, 0.08, 0.096, 0.112, 0.128, 0.144])
    )


def test_data_to_licks(monkeypatch, foraging2_data_fixture):
    monkeypatch.setattr(
        foraging2.extract,
        "get_vsyncs",
        lambda data: [16] * data["items"]["behavior"]["update_count"]
    )

    expected = pd.DataFrame(data={
        "frame": np.array([]),
        "time": np.array([]),
    })  # the test data doesnt have lick sensor data so i guess this is more of a smoke test for now...?

    pd.testing.assert_frame_equal(
        foraging2.data_to_licks(foraging2_data_fixture),
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True,
        check_names=False
    )


def test_data_to_trials(foraging2_data_fixture):
    trials = foraging2.data_to_trials(foraging2_data_fixture)

    pd.testing.assert_frame_equal(
        trials,
        EXPECTED_TRIALS,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True,
        check_exact=False,
        check_less_precise=3,
    )

    # regression test for https://github.com/AllenInstitute/visual_behavior_analysis/issues/289
    data = foraging2_data_fixture.copy()
    data['items']['behavior']['trial_log'][-1].pop('trial_params')
    expected = EXPECTED_TRIALS.iloc[:-1]
    # expected.set_index([0, 1])

    trials = foraging2.data_to_trials(data)

    pd.testing.assert_frame_equal(
        trials,
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True,
        check_exact=False,
        check_less_precise=3,
    )


def test_data_to_visual_stimuli(
        foraging2_data_stage0_2018_05_10,
        foraging2_expected_visual_stimuli_stage0_2018_05_10
):
    visual_stimuli = foraging2.data_to_visual_stimuli(foraging2_data_stage0_2018_05_10)

    pd.testing.assert_frame_equal(
        visual_stimuli,
        foraging2_expected_visual_stimuli_stage0_2018_05_10,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True,
        check_names=False
    )


def test_data_to_visual_stimuli_images(foraging2_data_stage4_2018_05_10):
    visual_stimuli = foraging2.data_to_visual_stimuli(foraging2_data_stage4_2018_05_10)

    pd.testing.assert_series_equal(
        visual_stimuli['image_category'],
        visual_stimuli['image_name'],
        check_names=False,
    )
