import pytest
import os
import datetime
import numpy as np
import pandas as pd
from copy import deepcopy
from shutil import copyfile
from collections import Mapping
from six import PY3
from six.moves import cPickle as pickle

from visual_behavior.pizza import we_can_unpizza_that  # this is terrible but hopefully will be an external dependency very soon


TESTING_DIR = os.path.realpath(os.path.dirname(__file__))
TESTING_RES_DIR = os.path.join(TESTING_DIR, "res")
TEST_SESSION_FNAME = "180122094711-task=DoC_NaturalImages_CAMMatched_n=8_stage=natural_images_mouse=M363894.pkl"


if PY3:
    load_pickle = lambda pstream: pickle.load(pstream, encoding="latin1")
else:
    load_pickle = lambda pstream: pickle.load(pstream)


@pytest.fixture
def exemplar_extended_trials_fixture():
    return pd.DataFrame(data={
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
            datetime.datetime(2017, 7, 19, 10, 35, 8, 369000),
            datetime.datetime(2017, 7, 19, 10, 35, 8, 369000),
            datetime.datetime(2017, 7, 19, 10, 35, 8, 369000),
            datetime.datetime(2017, 7, 19, 10, 35, 8, 369000),
        ],
        "date": [
            datetime.date(2017, 7, 19),
            datetime.date(2017, 7, 19),
            datetime.date(2017, 7, 19),
            datetime.date(2017, 7, 19),
        ],
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
        'auto_rewarded', 'change_contrast', 'change_frame',
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


@pytest.fixture(scope="module")
def trials_df_fixture():
    trials = pd.read_pickle(os.path.join(TESTING_RES_DIR, "trials.pkl"))
    trials["scheduled_change_time"] = trials.apply(
        lambda row: row["scheduled_change_time"] - row["starttime"],
        axis=1
    )  # change scheduled_change_time from time relative to experiment start to time relative to trial start
    del trials['stim_on_frames']
    return trials


@pytest.fixture(scope="module")
def annotated_trials_df_fixture():
    return pd.read_pickle(os.path.join(TESTING_RES_DIR, "trials_annotated.pkl"))


@pytest.fixture(scope="module")
def behavioral_session_output_fixture():
    return pd.read_pickle(os.path.join(TESTING_RES_DIR, TEST_SESSION_FNAME))


@pytest.fixture(scope="module")
def behavioral_session_output_pickle_fixture():
    return os.path.join(TESTING_RES_DIR, TEST_SESSION_FNAME)


@pytest.fixture(scope="module")
def trials_df_pickle_fixture():
    return os.path.join(TESTING_RES_DIR, "trials.pkl")


@pytest.fixture(scope="module")
def annotated_trials_df_pickle_fixture():
    return os.path.join(TESTING_RES_DIR, "trials_annotated.pkl")


@pytest.fixture(scope="session")
def pizza_data_fixture():
    return we_can_unpizza_that(os.path.join(TESTING_RES_DIR, "180215110844176000.h5"))


@pytest.fixture(scope="session")
def foraging_data_fixture():
    with open(os.path.join(TESTING_RES_DIR, "nat_images__legacy.pkl"), "rb") as \
            pstream:
        return load_pickle(pstream)


@pytest.fixture(scope="session")
def foraging2_data_fixture():
    foraging2_data = pd.read_pickle(
        os.path.join(TESTING_RES_DIR, "foraging2_chris.pkl")
    )

    # this is the hack way we are using to edit the stim_log item ordering due to derricw new changes
    old_set_log = foraging2_data["items"]["behavior"]["stimuli"]["gratings"]["set_log"]
    new_set_log = []

    new_set_log.append(
        ('Ori', 0, 0, 0, ),
    )  # this iteration of the foraging2 output is doesnt have first image diplay in set_log

    for set_attr, set_value, set_frame, set_time in old_set_log:
        new_set_log.append((set_attr, set_value, set_time, set_frame, ))

    foraging2_data["items"]["behavior"]["stimuli"]["gratings"]["set_log"] = \
        new_set_log

    return foraging2_data


@pytest.fixture(scope="session")
def foraging2_natural_scenes_data_fixture():
    foraging2_data = pd.read_pickle(
        os.path.join(TESTING_RES_DIR, "foraging2_natural_images.pkl")
    )

    # this is the hack way we are using to edit the stim_log item ordering due to derricw new changes
    old_set_log = foraging2_data["items"]["behavior"]["stimuli"]["natural_scenes"]["set_log"]
    new_set_log = []

    for set_value, set_frame, set_time in old_set_log:
        new_set_log.append(("image", set_value, set_time, set_frame, ))  # old pickle, need to hardcode set_attr to "image"

    foraging2_data["items"]["behavior"]["stimuli"]["natural_scenes"]["set_log"] = \
        new_set_log

    return foraging2_data


@pytest.fixture(scope="session")
def foraging2_data_fixture_issue_73():
    foraging2_data = pd.read_pickle(
        os.path.join(TESTING_RES_DIR, "issue_73.pkl")
    )

    # this is the hack way we are using to edit the stim_log item ordering due to derricw new changes
    old_set_log = foraging2_data["items"]["behavior"]["stimuli"]["natual_scenes"]["set_log"]
    new_set_log = []

    for set_value, set_frame, set_time in old_set_log:
        new_set_log.append(("image", set_value, set_time, set_frame, ))

    foraging2_data["items"]["behavior"]["stimuli"]["natual_scenes"]["set_log"] = \
        new_set_log

    return foraging2_data


@pytest.fixture(scope="session")
def foraging2_data_fixture_issue_116():
    foraging2_data = pd.read_pickle(
        os.path.join(TESTING_RES_DIR, "doc_gratings_9364d72_StupidDoCMouse.pkl")
    )

    return foraging2_data


@pytest.fixture(scope="session")
def foraging2_data_stage4_2018_05_10():
    return pd.read_pickle(
        os.path.join(TESTING_RES_DIR, "doc_images_21f064f_PerfectDoCMouse.pkl")
    )


@pytest.fixture(scope="session")
def foraging2_data_stage0_2018_05_10():
    return pd.read_pickle(
        os.path.join(TESTING_RES_DIR, "doc_gratings_8910798_EarlyDoCMouse.pkl")
    )
