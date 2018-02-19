import pytest
import os
import numpy as np
from numpy import nan
import pandas as pd
from six.moves import cPickle as pickle

from visual_behavior import io


@pytest.mark.parametrize("value, to_pickle", [
    (pd.DataFrame(data={"foo": {0: 1, 1: 2, }, }), False, ),
    (pd.DataFrame(data={"foo": {0: 1, 1: 2, }, }), True, ),
])
def test_data_or_pkl(tmpdir, value, to_pickle):
    if to_pickle:
        pickle_path = os.path.join(str(tmpdir), "test.pkl")
        value.to_pickle(pickle_path)
        pd.testing.assert_frame_equal(
            io.data_or_pkl(lambda value: value)(pickle_path).sort_index(axis=1),
            value.sort_index(axis=1)
        )
    else:
        pd.testing.assert_frame_equal(
            io.data_or_pkl(lambda value: value)(value).sort_index(axis=1),
            value.sort_index(axis=1)
        )


def test_load_trials(behavioral_session_output_fixture, trials_df_fixture):
    pd.testing.assert_frame_equal(
        io.load_trials(behavioral_session_output_fixture).sort_index(axis=1),
        trials_df_fixture.sort_index(axis=1)
    )


def test_load_time(behavioral_session_output_fixture):
    np.testing.assert_almost_equal(
        io.load_time(behavioral_session_output_fixture)[:11],
        np.array([
            0., 0.04998129, 0.09999849, 0.15004392, 0.20007941, 0.23343854,
            0.26680312, 0.30016385, 0.33352555, 0.36688468, 0.40024766,
        ])
    )


# def test_load_licks(behavioral_session_output_fixture):
#     assert io.load_licks(behavioral_session_output_fixture).iloc[:10].equals(
#         pd.DataFrame(data={
#             "frame": {
#                 0: 18, 1: 22, 2: 27, 3: 28, 4: 147, 5: 148, 6: 152, 7: 153,
#                 8: 294, 9: 295,
#             },
#             "time": {
#                 0: 0.66713256202638149,
#                 1: 0.85061739757657051,
#                 2: 1.0674719456583261,
#                 3: 1.1008249819278717,
#                 4: 5.6712611606344581,
#                 5: 5.7046244600787759,
#                 6: 5.8380689965561032,
#                 7: 5.8714306922629476,
#                 8: 11.259211733005941,
#                 9: 11.292535263113678,
#             },
#         }, columns=["frame", "time", ])
#     )
#
#     assert io.load_licks(behavioral_session_output_fixture, io.load_time(behavioral_session_output_fixture)).iloc[:10].equals(
#         pd.DataFrame(data={
#             "frame": {
#                 0: 18, 1: 22, 2: 27, 3: 28, 4: 147, 5: 148, 6: 152, 7: 153,
#                 8: 294, 9: 295,
#             },
#             "time": {
#                 0: 0.66713256202638149,
#                 1: 0.85061739757657051,
#                 2: 1.0674719456583261,
#                 3: 1.1008249819278717,
#                 4: 5.6712611606344581,
#                 5: 5.7046244600787759,
#                 6: 5.8380689965561032,
#                 7: 5.8714306922629476,
#                 8: 11.259211733005941,
#                 9: 11.292535263113678,
#             },
#         }, columns=["frame", "time", ])
#     )


def test_load_rewards(behavioral_session_output_fixture):
    pd.testing.assert_frame_equal(
        io.load_rewards(behavioral_session_output_fixture).iloc[:10],
        pd.DataFrame(data={
            "frame": {
                0.0: 600, 1.0: 980, 2.0: 2080, 3.0: 2440, 4.0: 3200, 5.0: 4133,
                6.0: 6731, 7.0: 7854, 8.0: 10709, 9.0: 10944,
            },
            "time": {
                0.0: 22.9355107285, 1.0: 37.4975822689, 2.0: 79.5989893544,
                3.0: 93.3269995861, 4.0: 122.417696304, 5.0: 158.180579774,
                6.0: 257.612732749, 7.0: 300.564845749, 8.0: 409.78845397,
                9.0: 418.779212811,
            },
        }, columns=["frame", "time", ]),
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )

    pd.testing.assert_frame_equal(
        io.load_rewards(behavioral_session_output_fixture, io.load_time(behavioral_session_output_fixture)).iloc[:10],
        pd.DataFrame(data={
            "frame": {
                0.0: 600, 1.0: 980, 2.0: 2080, 3.0: 2440, 4.0: 3200, 5.0: 4133,
                6.0: 6731, 7.0: 7854, 8.0: 10709, 9.0: 10944,
            },
            "time": {
                0.0: 22.9355107285, 1.0: 37.4975822689, 2.0: 79.5989893544,
                3.0: 93.3269995861, 4.0: 122.417696304, 5.0: 158.180579774,
                6.0: 257.612732749, 7.0: 300.564845749, 8.0: 409.78845397,
                9.0: 418.779212811,
            },
        }, columns=["frame", "time", ]),
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


# def test_load_flashes(behavioral_session_output_fixture):
#     EXPECTED_FLASHES_DF = pd.DataFrame(
#         data={
#             'delta_minimum': {0.0: 2.25, 1.0: 2.25, 2.0: 2.25, 3.0: 2.25, 4.0: 2.25, 5.0: 2.25},
#             'lick': {0.0: True, 1.0: False, 2.0: False, 3.0: False, 4.0: False, 5.0: False},
#             'frame': {0.0: 20, 1.0: 40, 2.0: 60, 3.0: 80, 4.0: 100, 5.0: 120},
#             'datetime': {0.0: '2018-01-22 08:47:09.440000', 1.0: '2018-01-22 08:47:09.440000', 2.0: '2018-01-22 08:47:09.440000', 3.0: '2018-01-22 08:47:09.440000', 4.0: '2018-01-22 08:47:09.440000', 5.0: '2018-01-22 08:47:09.440000'},
#             'mouse_id': {0.0: u'M363894', 1.0: u'M363894', 2.0: u'M363894', 3.0: u'M363894', 4.0: u'M363894', 5.0: u'M363894'},
#             'time_reward': {0.0: nan, 1.0: nan, 2.0: nan, 3.0: nan, 4.0: nan, 5.0: nan},
#             'initial_blank': {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0},
#             'stimulus_distribution': {0.0: u'exponential', 1.0: u'exponential', 2.0: u'exponential', 3.0: u'exponential', 4.0: u'exponential', 5.0: u'exponential'},
#             'frame_reward': {0.0: nan, 1.0: nan, 2.0: nan, 3.0: nan, 4.0: nan, 5.0: nan},
#             'frame_lick': {0.0: 22.0, 1.0: nan, 2.0: nan, 3.0: nan, 4.0: nan, 5.0: nan},
#             'ori': {0.0: 90, 1.0: 90, 2.0: 90, 3.0: 90, 4.0: 90, 5.0: 90},
#             'delta_mean': {0.0: 2, 1.0: 2, 2.0: 2, 3.0: 2, 4.0: 2, 5.0: 2},
#             'ori_change': {0.0: 1, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0},
#             'change': {0.0: 1, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0},
#             'stage': {0.0: u'natural_images', 1.0: u'natural_images', 2.0: u'natural_images', 3.0: u'natural_images', 4.0: u'natural_images', 5.0: u'natural_images'},
#             'prior_ori': {0.0: nan, 1.0: 90.0, 2.0: 90.0, 3.0: 90.0, 4.0: 90.0, 5.0: 90.0},
#             'task': {0.0: u'DoC_NaturalImages_CAMMatched_n=8', 1.0: u'DoC_NaturalImages_CAMMatched_n=8', 2.0: u'DoC_NaturalImages_CAMMatched_n=8', 3.0: u'DoC_NaturalImages_CAMMatched_n=8', 4.0: u'DoC_NaturalImages_CAMMatched_n=8', 5.0: u'DoC_NaturalImages_CAMMatched_n=8'},
#             'last_lick': {0.0: nan, 1.0: 1.0, 2.0: 2.0, 3.0: 3.0, 4.0: 4.0, 5.0: 5.0},
#             'stimulus': {0.0: u'Natural_Images_Lum_Matched_set_training_2017.07.14.pkl', 1.0: u'Natural_Images_Lum_Matched_set_training_2017.07.14.pkl', 2.0: u'Natural_Images_Lum_Matched_set_training_2017.07.14.pkl', 3.0: u'Natural_Images_Lum_Matched_set_training_2017.07.14.pkl', 4.0: u'Natural_Images_Lum_Matched_set_training_2017.07.14.pkl', 5.0: u'Natural_Images_Lum_Matched_set_training_2017.07.14.pkl'},
#             'time_lick': {0.0: 0.85061739757657051, 1.0: nan, 2.0: nan, 3.0: nan, 4.0: nan, 5.0: nan},
#             'flashed': {0.0: True, 1.0: True, 2.0: True, 3.0: True, 4.0: True, 5.0: True},
#             'trial': {0.0: 1, 1.0: 2, 2.0: 2, 3.0: 2, 4.0: 2, 5.0: 2},
#             'time': {0.0: 0.75053327344357967, 1.0: 1.517834628932178, 2.0: 2.2851340612396598, 3.0: 3.0524164950475097, 4.0: 3.8197345286607742, 5.0: 4.5870599392801523},
#             'reward': {0.0: False, 1.0: False, 2.0: False, 3.0: False, 4.0: False, 5.0: False},
#             'trial_duration': {0.0: 8.25, 1.0: 8.25, 2.0: 8.25, 3.0: 8.25, 4.0: 8.25, 5.0: 8.25},
#         },
#         index=pd.Int64Index(
#             [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, ],
#             dtype="float64",
#             name="flash_index"
#         ),
#         columns=[
#             "ori", "frame", "prior_ori", "ori_change", "change",
#             "time", "frame_lick", "time_lick", "lick", "last_lick", "frame_reward",
#             "time_reward", "reward", "trial", "flashed", "mouse_id",
#             "datetime", "task", "stage", "initial_blank", "delta_minimum",
#             "delta_mean", "stimulus_distribution", "trial_duration", "stimulus",
#         ]
#     )  # bleh
#
#     pd.testing.assert_frame_equal(
#         io.load_flashes(behavioral_session_output_fixture).iloc[:6],  # i hate pandas...why does this replace normal slicing in just p27!?
#         EXPECTED_FLASHES_DF,
#         check_column_type=False,
#         check_index_type=False,
#         check_dtype=False,
#         check_like=True
#     )
#
#     pd.testing.assert_frame_equal(
#         io.load_flashes(behavioral_session_output_fixture, io.load_time(behavioral_session_output_fixture)).iloc[:6],
#         EXPECTED_FLASHES_DF,
#         check_column_type=False,
#         check_index_type=False,
#         check_dtype=False,
#         check_like=True
#     )


def test_load_running_speed(behavioral_session_output_fixture):
    EXPECTED_RUNNING_DF = pd.DataFrame(
        data={
            'acceleration (cm/s^2)': {
                0: 2.9123735258347376,
                1: 1.4561867629173688,
                2: -0.00053134947429403175,
                3: -0.00053134947429403175,
                4: -10.038109468389708
            },
            'jerk (cm/s^3)': {
                0: -29.134639877456738,
                1: -29.129489658054311,
                2: -14.562169719325942,
                3: -100.3045892838318,
                4: -218.06477955362016
            },
            'speed (cm/s)': {
                0: 0.0,
                1: 0.14556417360329282,
                2: 0.14556417360329282,
                3: 0.14551099037872797,
                4: 0.14551099037872797
            },
            'time': {
                0: 0.0,
                1: 0.049981285817921162,
                2: 0.099998492747545242,
                3: 0.1500439215451479,
                4: 0.2000794094055891
            }
        },
        columns=[
            u'acceleration (cm/s^2)', u'jerk (cm/s^3)', u'speed (cm/s)',
            u'time',
        ]
    )

    pd.testing.assert_frame_equal(
        io.load_running_speed(behavioral_session_output_fixture).iloc[:5],
        EXPECTED_RUNNING_DF,
        check_like=True
    )

    pd.testing.assert_frame_equal(
        io.load_running_speed(behavioral_session_output_fixture, time=io.load_time(behavioral_session_output_fixture)).iloc[:5],
        EXPECTED_RUNNING_DF,
        check_like=True
    )
