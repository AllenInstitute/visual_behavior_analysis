import pytest
import numpy as np
import pandas as pd

from visual_behavior.validation import extended_trials


@pytest.fixture
def exemplar_extended_trials_fixture():
    return pd.DataFrame(data={
        "Unnamed: 0": [0, 1, 2, 3, ],
        "auto_rewarded": [True, True, True, False, ],
        "change_contrast": [1.0, 1.0, 1.0, 1.0, ],
        "change_frame": [90.0, 315.0, 435.0, np.nan, ],
        "change_image_category": [np.nan, np.nan, np.nan, np.nan, ],
        "change_image_name": [np.nan, np.nan, np.nan, np.nan, ],
        "change_ori": [180.0, 270.0, 360.0, np.nan, ],
        "change_time": [4.503416, 15.761926, 21.766479, np.nan, ],
        "cumulative_reward_number": [1, 2, 3, 3, ],
        "cumulative_volume": [0.005, 0.010, 0.015, 0.015, ],
        "delta_ori": [90.0, 90.0, 90.0, 90.0, ],
        "index": [0, 1, 2, 3, ],
        "initial_contrast": [1.0, 1.0, 1.0, 1.0, ],
        "initial_image_category": [np.nan, np.nan, np.nan, np.nan, ],
        "initial_image_name": [np.nan, np.nan, np.nan, np.nan, ],
        "initial_ori": [90.0, 180.0, 270.0, 360.0, ],
        "lick_times": [
            [5.1038541570305824, ],
            [16.012116840109229, 16.612536765635014, 16.762686072383076, 16.862761508207768, 17.012875908520073, 17.162988923955709, 17.463219941128045, ],
            [21.966633949428797, 22.216823508497328, 22.467007804196328, 22.61712248204276, 22.717199579812586, 22.867311764042825, 23.017423948273063, 25.419245491269976, ],
            [25.769513533916324, 26.020016693975776, 26.219824876636267, ],
        ],
        "optogenetics": [False, False, False, False, ],
        "publish_time": [
            "2017-07-19T10:35:17.799000",
            "2017-07-19T10:35:27.559000",
            "2017-07-19T10:35:35.069000",
            "2017-07-19T10:35:35.819000",
        ],
        "response_latency": [0.600438, 0.250190, 0.200154, np.inf, ],
        "response_time": [[], [], [], [], ],
        "response_type": ["HIT", "HIT", "HIT", "MISS", ],
        "reward_frames": [[90, ], [315, ], [435, ], [], ],
        "reward_times": [[4.50341622, ], [15.76192645, ], [21.76647948, ], [], ],
        "reward_volume": [0.005, 0.005, 0.005, 0.005, ],
        "rewarded": [True, True, True, True, ],
        "scheduled_change_time": [3.96059, 15.242503, 21.582325, 28.156679, ],
        "startframe": [0, 165, 360, 510, ],
        "starttime": [0.0, 8.256248, 18.013633, 25.519331, ],
        "stim_on_frames": [
            [],
            [],
            [
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1.,
            ],
            [
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            ],
        ],
        "stimulus_distribution": [
            "exponential",
            "exponential",
            "exponential",
            "exponential",
        ],
        "task": [
            "DetectionOfChange_General",
            "DetectionOfChange_General",
            "DetectionOfChange_General",
            "DetectionOfChange_General",
        ],
        "user_id": ["linzyc", "linzyc", "linzyc", "linzyc", ],
        "stim_duration": [0.75, 0.75, 0.75, 0.75, ],
        "LDT_mode": ["single", "single", "single", "single"],
        "distribution_mean": [2, 2, 2, 2, ],
        "trial_duration": [8.25, 8.25, 8.25, 8.25, ],
        "stimulus": [
            "full_screen_square_wave",
            "full_screen_square_wave",
            "full_screen_square_wave",
            "full_screen_square_wave",
        ],
        "mouse_id": ["M327444", "M327444", "M327444", "M327444", ],
        "prechange_minimum": [2.25, 2.25, 2.25, 2.25, ],
        "computer_name": [
            "W7VS-SYSLOGIC36",
            "W7VS-SYSLOGIC36",
            "W7VS-SYSLOGIC36",
            "W7VS-SYSLOGIC36",
        ],
        "blank_duration_range": [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "response_window": [
            [0.15, 1],
            [0.15, 1],
            [0.15, 1],
            [0.15, 1],
        ],
        "blank_screen_timeout": [False, False, False, False, ],
        "session_duration": [
            3600.0983944,
            3600.0983944,
            3600.0983944,
            3600.0983944,
        ],
        "stage": [
            "static_full_field_gratings",
            "static_full_field_gratings",
            "static_full_field_gratings",
            "static_full_field_gratings",
        ],
        "startdatetime": [
            "2017-07-19 10:35:08.369",
            "2017-07-19 10:35:08.369",
            "2017-07-19 10:35:08.369",
            "2017-07-19 10:35:08.369",
        ],
        "date": ["2017-07-19", "2017-07-19", "2017-07-19", "2017-07-19", ],
        "year": [2017, 2017, 2017, 2017, ],
        "month": [7, 7, 7, 7, ],
        "day": [19, 19, 19, 19, ],
        "hour": [10, 10, 10, 10, ],
        "dayofweek": [2, 2, 2, 2, ],
        "number_of_rewards": [1, 1, 1, 0, ],
        "rig_id": ["E6", "E6", "E6", "E6", ],
        "trial_type": ["go", "go", "go", "aborted", ],
        "endframe": [164, 359, 509, 524, ],
        "lick_frames": [
            [102, ],
            [320, 332, 335, 337, 340, 343, 349, ],
            [439, 444, 449, 452, 454, 457, 460, 508, ],
            [515, 520, 524, ],
        ],
        "endtime": [8.206208, 17.963599, 25.469284, 26.219825, ],
        "reward_licks": [
            [0.60043793, ],
            [0.25019039, 0.85061032, 1.00075962, 1.10083506, 1.25094946, 1.40106247, 1.70129349, ],
            [0.20015447, 0.45034403, 0.70052833, 0.85064301, 0.9507201, 1.10083229, 1.25094447, ],
            np.nan,
        ],
        "reward_lick_count": [1.0, 7.0, 7.0, np.nan, ],
        "reward_lick_latency": [0.600438, 0.250190, 0.200154, np.nan, ],
        "reward_rate": [np.inf, np.inf, np.inf, np.inf, ],
        "response": [1.0, 1.0, 1.0, 1.0, ],
        "trial_length": [8.256248, 9.757385, 7.505698, 0.750566, ],
        "color": ["blue", "blue", "blue", "red", ],
    }, columns=[
        'Unnamed: 0', 'auto_rewarded', 'change_contrast', 'change_frame',
        'change_image_category', 'change_image_name', 'change_ori',
        'change_time', 'cumulative_reward_number', 'cumulative_volume',
        'delta_ori', 'index', 'initial_contrast', 'initial_image_category',
        'initial_image_name', 'initial_ori', 'lick_times', 'optogenetics',
        'publish_time', 'response_latency', 'response_time', 'response_type',
        'reward_frames', 'reward_times', 'reward_volume', 'rewarded',
        'scheduled_change_time', 'startframe', 'starttime', 'stim_on_frames',
        'stimulus_distribution', 'task', 'user_id', 'stim_duration', 'LDT_mode',
        'distribution_mean', 'trial_duration', 'stimulus', 'mouse_id',
        'prechange_minimum', 'computer_name', 'blank_duration_range',
        'response_window', 'blank_screen_timeout', 'session_duration', 'stage',
        'startdatetime', 'date', 'year', 'month', 'day', 'hour', 'dayofweek',
        'number_of_rewards', 'rig_id', 'trial_type', 'endframe', 'lick_frames',
        'endtime', 'reward_licks', 'reward_lick_count', 'reward_lick_latency',
        'reward_rate', 'response', 'trial_length', 'color'
    ])

# TRIAL
def test_validate_min_change_time_success(exemplar_extended_trials_fixture):
    assert exemplar_extended_trials_fixture.apply(extended_trials.validate_min_change_time, axis=1).tolist() == [True, True, True, True, ]


def test_validate_max_change_time_success():
    good_trials = pd.DataFrame(data={
        "change_time": [4.503416, 15.761926, 21.766479, np.nan, ],
        "prechange_minimum": [2.25, 2.25, 2.25, 2.25, ],
        "starttime": [0.0, 8.256248, 18.013633, 25.519331, ],
    })

    validate_max_change_time = lambda c: extended_trials.validate_max_change_time(c, stimulus_window=6.0)

    assert good_trials.apply(validate_max_change_time, axis=1).tolist() == \
        [True, True, True, True, ]


def test_validate_max_change_time_failure():
    bad_trials = pd.DataFrame(data={
        "change_time": [27.766479, np.nan, ],
        "prechange_minimum": [2.25, 2.25, ],
        "starttime": [18.013633, 29.766479, ],
    })

    validate_max_change_time = lambda c: extended_trials.validate_max_change_time(c, stimulus_window=6.0)

    assert bad_trials.apply(validate_max_change_time, axis=1).tolist() == \
        [False, True, ]


def test_validate_min_change_time_failure():
    bad_trials = pd.DataFrame(data={
        "prechange_minimum": [2.25, ],
        "change_time": [1.5, ],
        "starttime": [0, ],
    })

    assert bad_trials.apply(extended_trials.validate_min_change_time, axis=1).tolist() == [False, ]


def test_validate_autoreward_volume_success(exemplar_extended_trials_fixture):
    exemplar_extended_trials_fixture.apply(extended_trials.validate_min_change_time, axis=1).tolist() == [True, True, True, True, ]


# TRIALS
@pytest.mark.parametrize("expected_warmup_trial_count, expected", [
    (None, True, ),
    (3, True, ),
    (2, False, ),
    (4, False, ),
])
def test_validate_number_of_warmup_trials(exemplar_extended_trials_fixture, expected_warmup_trial_count, expected):
    assert extended_trials.validate_number_of_warmup_trials(exemplar_extended_trials_fixture, expected_warmup_trial_count) == expected


def test_validate_reward_delivery_on_warmup_trials(exemplar_extended_trials_fixture):
    assert extended_trials.validate_reward_delivery_on_warmup_trials(exemplar_extended_trials_fixture) is True


def test_validate_autorewards_after_N_consecutive_misses_success():
    good_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "go", "go", ],
        "auto_rewarded": [False, False, False, True, ],
        "response": [0, 1, 0, 0, ],
    })

    assert extended_trials.validate_autorewards_after_N_consecutive_misses(good_trials, 2) is True


def test_validate_autorewards_after_N_consecutive_misses_failure():
    bad_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "go", "go", ],
        "auto_rewarded": [False, False, False, False, ],
        "response": [0, 1, 0, 0, ],
    })

    assert extended_trials.validate_autorewards_after_N_consecutive_misses(bad_trials, 2) is False


def test_validate_change_on_all_go_trials_success(exemplar_extended_trials_fixture):
    assert extended_trials.validate_change_on_all_go_trials(exemplar_extended_trials_fixture) is True


def test_validate_change_on_all_go_trials_failure():
    bad_trials = pd.DataFrame(data={
        "trial_type": ["go", "go", "catch", ],
        "initial_image_name": ["test_1", "test_1", "test_1", ],
        "change_image_name": ["test_1", "test_1", "test_1", ],
        "initial_ori": [90, 90, 90, ],
        "change_ori": [90, 180, 90, ],
    })

    assert extended_trials.validate_change_on_all_go_trials(bad_trials) is False


def test_validate_no_change_on_all_catch_trials_success():
    good_trials = pd.DataFrame(data={
        "trial_type": ["catch", "catch", "go", ],
        "initial_image_name": ["test_1", "test_1", "test_1", ],
        "change_image_name": ["test_1", "test_1", "test_1", ],
        "initial_ori": [90, 90, 90, ],
        "change_ori": [90, 90, 180, ],
    })

    assert extended_trials.validate_no_change_on_all_catch_trials(good_trials) is True


def test_validate_no_change_on_all_catch_trials_failure():
    bad_trials = pd.DataFrame(data={
        "trial_type": ["catch", "catch", "go", ],
        "initial_image_name": ["test_1", "test_1", "test_1", ],
        "change_image_name": ["test_1", "test_1", "test_1", ],
        "initial_ori": [90, 90, 90, ],
        "change_ori": [90, 180, 180, ],
    })

    assert extended_trials.validate_no_change_on_all_catch_trials(bad_trials) is False


def test_validate_intial_and_final_in_non_aborted_success():
    good_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "aborted", ],
        "initial_image_name": [np.nan, "test", np.nan, ],
        "change_image_name": [np.nan, "test_1", np.nan, ],
        "initial_ori": [90, 90, np.nan, ],
        "change_ori": [180, 90, np.nan, ],
    })

    assert extended_trials.validate_intial_and_final_in_non_aborted(good_trials) is True


def test_validate_intial_and_final_in_non_aborted_failure():
    bad_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "aborted", ],
        "initial_image_name": [np.nan, np.nan, np.nan, ],
        "change_image_name": [np.nan, np.nan, np.nan, ],
        "initial_ori": [90, np.nan, np.nan, ],
        "change_ori": [180, np.nan, np.nan, ],
    })

    assert extended_trials.validate_intial_and_final_in_non_aborted(bad_trials) is False


# def test_validate_reward_follows_first_lick_in_window_success(exemplar_extended_trials_fixture):
#     extended_trials.validate_reward_follows_first_lick_in_window(exemplar_extended_trials_fixture) is True
#
#
# def test_validate_reward_follows_first_lick_in_window_failure():
#     bad_trials = {}
#     extended_trials.validate_reward_follows_first_lick_in_window(bad_trials) is False


@pytest.mark.parametrize("trials, expected", [
    (
        pd.DataFrame(data={"number_of_rewards": [0, 0, 1, 1, 0, 1, ], }),
        True,
    ),
    (
        pd.DataFrame(data={"number_of_rewards": [0, 0, 1, 1, 0, 1, 2, ], }),
        False,
    ),
])
def test_validate_never_more_than_one_reward(trials, expected):
    assert extended_trials.validate_never_more_than_one_reward(trials) == expected


def test_validate_lick_before_scheduled_on_aborted_trials_success():
    good_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "aborted", ],
        "lick_times": [[0.55, 0.56, 0.57, ], [1.05, 1.06, 1.07, ], [1.55, 1.56, 1.57, ], ],
        "scheduled_change_time": [0.45, 1.00, 2.55, ],
    })

    assert extended_trials.validate_lick_before_scheduled_on_aborted_trials(good_trials) == True


def test_validate_lick_before_scheduled_on_aborted_trials_failure():
    bad_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "aborted", ],
        "lick_times": [[0.55, 0.56, 0.57, ], [1.05, 1.06, 1.07, ], [1.55, 1.56, 1.57, ], ],
        "scheduled_change_time": [0.45, 1.00, 1.54, ],
    })

    assert extended_trials.validate_lick_before_scheduled_on_aborted_trials(bad_trials) == False


def test_validate_lick_after_scheduled_on_go_catch_trials_success():
    good_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "aborted", ],
        "lick_times": [[0.55, 0.56, 0.57, ], [], [1.55, 1.56, 1.57, ], ],
        "scheduled_change_time": [0.45, 1.00, 2.55, ],
    })

    assert extended_trials.validate_lick_after_scheduled_on_go_catch_trials(good_trials) == True


def test_validate_lick_after_scheduled_on_go_catch_trials_failure():
    bad_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "aborted", ],
        "lick_times": [[0.43, 0.55, 0.56, 0.57, ], [], [1.55, 1.56, 1.57, ], ],
        "scheduled_change_time": [0.45, 1.00, 2.55, ],
    })

    assert extended_trials.validate_lick_after_scheduled_on_go_catch_trials(bad_trials) == False


def test_validate_initial_matches_final_success():
    good_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "go", ],
        "initial_image_name": ["im0", "im1", "im0", ],
        "change_image_name": ["im1", "im0", "im1", ],
        "initial_image_category": ["im0", "im1", "im0", ],
        "change_image_category": ["im1", "im0", "im1", ],
        "initial_ori": [None, None, None, ],
        "change_ori": [None, None, None, ],
    })

    assert extended_trials.validate_initial_matches_final(good_trials) == True


def test_validate_initial_matches_final_failure():
    bad_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "go", ],
        "initial_image_name": ["im0", "im0", "im0", ],
        "change_image_name": ["im1", "im0", "im1", ],
        "initial_image_category": ["im0", "im0", "im0", ],
        "change_image_category": ["im1", "im0", "im1", ],
        "initial_ori": [None, None, None, ],
        "change_ori": [None, None, None, ],
    })

    assert extended_trials.validate_initial_matches_final(bad_trials) == False


def test_validate_first_lick_after_change_on_nonaborted_success():
    good_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "aborted", ],
        "lick_times": [[0.55, 0.56, 0.57, ], [1.05, 1.06, 1.07, ], [1.55, 1.56, 1.57, ], ],
        "change_time": [0.45, 1.00, 1.54, ],
    })

    assert extended_trials.validate_first_lick_after_change_on_nonaborted(good_trials) == True


def test_validate_first_lick_after_change_on_nonaborted_failure():
    bad_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "aborted", ],
        "lick_times": [[0.43, 0.55, 0.56, 0.57, ], [], [1.55, 1.56, 1.57, ], ],
        "change_time": [0.45, 1.00, 2.55, ],
    })

    assert extended_trials.validate_first_lick_after_change_on_nonaborted(bad_trials) == False


def test_validate_trial_ends_without_licks_success():
    MINIMUM_NO_LICK_TIME = 0.3  # 300ms according to dougo

    good_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "aborted", ],
        "lick_times": [[0.43, 0.55, 0.56, 0.57, ], [], [1.55, 1.56, 1.57, ], ],
        "endtime": [1.00, 2.00, 2.55, ],
    })

    assert extended_trials.validate_trial_ends_without_licks(good_trials, MINIMUM_NO_LICK_TIME) == True


def test_validate_trial_ends_without_licks_failure():
    MINIMUM_NO_LICK_TIME = 0.3  # 300ms according to dougo

    bad_trials = pd.DataFrame(data={
        "trial_type": ["go", "catch", "aborted", ],
        "lick_times": [[0.43, 0.55, 0.56, 0.57, ], [], [1.55, 1.56, 1.57, ], ],
        "endtime": [0.80, 2.00, 2.55, ],
    })

    assert extended_trials.validate_trial_ends_without_licks(bad_trials, MINIMUM_NO_LICK_TIME) == False


@pytest.mark.parametrize("trials, expected_duration_seconds, tolerance, expected", [
    (pd.DataFrame(data={"endtime": [5429, ], }), 5400, 30, True, ),
    (pd.DataFrame(data={"endtime": [5430, ], }), 5400, 30, False, ),
    (pd.DataFrame(data={"endtime": [5430, ], }), 5400, 15, False, ),
    (pd.DataFrame(data={"endtime": [5430, ], }), 4400, 30, False, ),
])
def test_validate_session_within_expected_duration(trials, expected_duration_seconds, tolerance, expected):
    assert extended_trials.validate_session_within_expected_duration(trials, expected_duration_seconds, tolerance) == expected


@pytest.mark.parametrize("trials, volume_limit, tolerance, expected", [
    (pd.DataFrame(data={"cumulative_volume": [1.00, ], }), 1.00, 0.01, True, ),
    (pd.DataFrame(data={"cumulative_volume": [1.01, ], }), 1.00, 0.01, True, ),
    (pd.DataFrame(data={"cumulative_volume": [1.00, ], }), 0.90, 0.01, False, ),
    (pd.DataFrame(data={"cumulative_volume": [1.00, ], }), 0.90, 0.01, False, ),
])
def test_validate_session_ends_at_max_cumulative_volume(trials, volume_limit, tolerance, expected):
    assert extended_trials.validate_session_ends_at_max_cumulative_volume(trials, volume_limit, tolerance) == expected


@pytest.mark.parametrize("trials, expected_duration, volume_limit, time_tolerance, expected", [
    (pd.DataFrame(data={"cumulative_volume": [1.00, ], "endtime": [5400, ], }), 5430, 1.00, 30, True, ),
    (pd.DataFrame(data={"cumulative_volume": [1.00, ], "endtime": [5399, ], }), 5430, 1.00, 30, False, ),
    (pd.DataFrame(data={"cumulative_volume": [0.90, ], "endtime": [5000, ], }), 5400, 1.00, 30, False, ),
])
def test_validate_duration_and_volume_limit(trials, expected_duration, volume_limit, time_tolerance, expected):
    assert extended_trials.validate_duration_and_volume_limit(trials, expected_duration, volume_limit, time_tolerance) == expected
