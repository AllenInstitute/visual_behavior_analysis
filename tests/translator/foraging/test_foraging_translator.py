# TRANSLATOR
# from visual_behavior.translator import foraging
#
#
# def test_data_to_monolith(foraging_data_fixture):
#     foraging.data_to_monolith(foraging_data_fixture)
#
#
# def test_data_to_change_detection_core(foraging_data_fixture):
#     raise ValueErrort(foraging.data_to_change_detection_core(foraging_data_fixture).iloc[:3])


# EXTRACTOR
# import pytest
# import os
# import numpy as np
# from numpy import nan
# import pandas as pd
# from six.moves import cPickle as pickle
#
# from visual_behavior.translator.foraging import extract
#
#
# def test_load_trials(behavioral_session_output_fixture, trials_df_fixture):
#     pd.testing.assert_frame_equal(
#         extract.load_trials(behavioral_session_output_fixture).sort_index(axis=1),
#         trials_df_fixture.sort_index(axis=1)
#     )
#
#
# def test_load_time(behavioral_session_output_fixture):
#     np.testing.assert_almost_equal(
#         extract.load_time(behavioral_session_output_fixture)[:11],
#         np.array([
#             0., 0.04998129, 0.09999849, 0.15004392, 0.20007941, 0.23343854,
#             0.26680312, 0.30016385, 0.33352555, 0.36688468, 0.40024766,
#         ])
#     )
#
#
# def test_load_rewards(behavioral_session_output_fixture):
#     pd.testing.assert_frame_equal(
#         extract.load_rewards(behavioral_session_output_fixture).iloc[:10],
#         pd.DataFrame(data={
#             "frame": {
#                 0.0: 600, 1.0: 980, 2.0: 2080, 3.0: 2440, 4.0: 3200, 5.0: 4133,
#                 6.0: 6731, 7.0: 7854, 8.0: 10709, 9.0: 10944,
#             },
#             "time": {
#                 0.0: 22.9355107285, 1.0: 37.4975822689, 2.0: 79.5989893544,
#                 3.0: 93.3269995861, 4.0: 122.417696304, 5.0: 158.180579774,
#                 6.0: 257.612732749, 7.0: 300.564845749, 8.0: 409.78845397,
#                 9.0: 418.779212811,
#             },
#         }, columns=["frame", "time", ]),
#         check_column_type=False,
#         check_index_type=False,
#         check_dtype=False,
#         check_like=True
#     )
#
#     pd.testing.assert_frame_equal(
#         extract.load_rewards(behavioral_session_output_fixture, extract.load_time(behavioral_session_output_fixture)).iloc[:10],
#         pd.DataFrame(data={
#             "frame": {
#                 0.0: 600, 1.0: 980, 2.0: 2080, 3.0: 2440, 4.0: 3200, 5.0: 4133,
#                 6.0: 6731, 7.0: 7854, 8.0: 10709, 9.0: 10944,
#             },
#             "time": {
#                 0.0: 22.9355107285, 1.0: 37.4975822689, 2.0: 79.5989893544,
#                 3.0: 93.3269995861, 4.0: 122.417696304, 5.0: 158.180579774,
#                 6.0: 257.612732749, 7.0: 300.564845749, 8.0: 409.78845397,
#                 9.0: 418.779212811,
#             },
#         }, columns=["frame", "time", ]),
#         check_column_type=False,
#         check_index_type=False,
#         check_dtype=False,
#         check_like=True
#     )
#
#
# def test_load_running_speed(behavioral_session_output_fixture):
#     EXPECTED_RUNNING_DF = pd.DataFrame(
#         data={
#             'acceleration (cm/s^2)': {
#                 0: 2.9123735258347376,
#                 1: 1.4561867629173688,
#                 2: -0.00053134947429403175,
#                 3: -0.00053134947429403175,
#                 4: -10.038109468389708
#             },
#             'jerk (cm/s^3)': {
#                 0: -29.134639877456738,
#                 1: -29.129489658054311,
#                 2: -14.562169719325942,
#                 3: -100.3045892838318,
#                 4: -218.06477955362016
#             },
#             'speed (cm/s)': {
#                 0: 0.0,
#                 1: 0.14556417360329282,
#                 2: 0.14556417360329282,
#                 3: 0.14551099037872797,
#                 4: 0.14551099037872797
#             },
#             'time': {
#                 0: 0.0,
#                 1: 0.049981285817921162,
#                 2: 0.099998492747545242,
#                 3: 0.1500439215451479,
#                 4: 0.2000794094055891
#             }
#         },
#         columns=[
#             u'acceleration (cm/s^2)', u'jerk (cm/s^3)', u'speed (cm/s)',
#             u'time',
#         ]
#     )
#
#     pd.testing.assert_frame_equal(
#         extract.load_running_speed(behavioral_session_output_fixture).iloc[:5],
#         EXPECTED_RUNNING_DF,
#         check_like=True
#     )
#
#     pd.testing.assert_frame_equal(
#         extract.load_running_speed(behavioral_session_output_fixture, time=extract.load_time(behavioral_session_output_fixture)).iloc[:5],
#         EXPECTED_RUNNING_DF,
#         check_like=True
#     )


# # TRANSFORM
# import pytest
# import datetime
# import numpy as np
# import pandas as pd
#
# from visual_behavior.translator.foraging import transform
#
#
# TIMESTAMP_ISO = "2018-01-22 08:47:09.440000"
# TIMESTAMP = pd.Timestamp(TIMESTAMP_ISO)
#
#
# def test_annotate_parameters():
#     df = pd.DataFrame(
#         [
#             {"key0": 0},
#             {"key0": 3},
#             ]
#         )
#
#     keydict = {
#         # the column "another_value" should be filled with the values of "some_thing"
#         "another_value": "some_thing",
#         # the column "dne" should be filled with values of "dne", which doesn't exist
#         "dne": "dne",
#         }
#
#     pickled_data = {
#         "some_thing": "some_value",
#         }
#
#     expected = pd.DataFrame(
#         [
#             # since "dne" doesn't exist, it should be filled with None
#             {"key0": 0, "another_value": "some_value", "dne": None},
#             {"key0": 3, "another_value": "some_value", "dne": None},
#             ]
#         )
#
#     df_out = transform.annotate_parameters(df, pickled_data, keydict)
#
#     pd.testing.assert_frame_equal(
#         df_out,
#         expected,
#         check_like=True,
#     )
#
#     # if no keydict is passed, we shouldn't modify the dataframe
#     df_out = transform.annotate_parameters(df, pickled_data, None)
#
#     pd.testing.assert_frame_equal(
#         df_out,
#         df_out,
#         check_like=True,
#     )
#
#
# def test_explode_startdatetime():
#     df = pd.DataFrame(
#         [
#             {
#             "startdatetime": TIMESTAMP,
#             },
#         ]
#     )
#
#     df_expected = pd.DataFrame(
#         [
#             {
#             "startdatetime": TIMESTAMP,
#             "date": "2018-01-22",
#             "year": 2018,
#             "month": 1,
#             "day": 22,
#             "hour": 8,
#             "dayofweek": 0,
#             },
#         ]
#     )
#
#     df_out = transform.explode_startdatetime(df)
#
#     pd.testing.assert_frame_equal(
#         df_out,
#         df_expected,
#         check_like=True,
#         )
#
#
# def test_annotate_n_rewards():
#     df = pd.DataFrame(
#         [
#             {"reward_times": [37.50, ]},
#             {"reward_times": [1.2, 37.50, ]},
#             {"reward_times": []},
#         ]
#     )
#
#     expected_vals = [
#         1,
#         2,
#         0,
#     ]
#     expected_col = pd.Series(expected_vals,name='number_of_rewards')
#
#     df_out = transform.annotate_n_rewards(df)
#
#     pd.testing.assert_series_equal(
#         df_out['number_of_rewards'],
#         expected_col,
#         )
#
#
# def test_annotate_rig_id():
#     df = pd.DataFrame(
#         [
#             {"key0": 0},
#             {"key0": 3},
#             ]
#         )
#
#     pickled_data = {
#         "rig_id": "TEST_RIG",
#         }
#
#     expected = pd.DataFrame(
#         [
#             # since "dne" doesn't exist, it should be filled with None
#             {"key0": 0, "rig_id": "TEST_RIG",},
#             {"key0": 3, "rig_id": "TEST_RIG",},
#             ]
#         )
#
#     df_out = transform.annotate_rig_id(df, pickled_data)
#
#     pd.testing.assert_frame_equal(
#         df_out,
#         expected
#     )
#
#
# def test_annotate_startdatetime():
#     df = pd.DataFrame(
#         [
#             {"test": None,},
#         ],
#     )
#
#     pickled_data = {"startdatetime": TIMESTAMP_ISO, }
#
#     expected = pd.DataFrame(
#         [
#             {"test": None, "startdatetime": TIMESTAMP,},
#         ],
#     )
#
#     df_out = transform.annotate_startdatetime(df, pickled_data)
#
#     pd.testing.assert_frame_equal(
#         df_out,
#         expected,
#         check_like=True,
#     )
#
#
# @pytest.mark.parametrize("df, pickled_data, expected", [
#     (
#         pd.DataFrame(data={
#             "number_of_rewards": {17: 2, },
#         }),
#         {"rewardvol": 0.005, },
#         pd.DataFrame(data={
#             "reward_volume": {17: 0.01, },
#             "number_of_rewards": {17: 2, },
#             "cumulative_volume": {17: 0.01, },
#         }, columns=["number_of_rewards", "reward_volume", "cumulative_volume", ]),
#     ),
#     (
#         pd.DataFrame(data={
#             "reward_volume": {17: 0.005, },
#             "number_of_rewards": {17: 2, },
#         }),
#         {"rewardvol": 0.005, },
#         pd.DataFrame(data={
#             "reward_volume": {17: 0.005, },
#             "number_of_rewards": {17: 2, },
#             "cumulative_volume": {17: 0.005, },
#         }, columns=["number_of_rewards", "reward_volume", "cumulative_volume", ]),
#     )
# ])
# def test_annotate_cumulative_reward(df, pickled_data, expected):
#     pd.testing.assert_frame_equal(
#         transform.annotate_cumulative_reward(df, pickled_data),
#         expected
#     )
#
#
# def test_annotate_filename():
#     df = pd.DataFrame([{"test": "test"}])
#     filename = "test/path/fname.py"
#
#     expected = pd.DataFrame([{
#         "test": "test",
#         "filepath": "test/path",
#         "filename": "fname.py",
#         }])
#
#     df_out = transform.annotate_filename(df, filename)
#
#     pd.testing.assert_frame_equal(
#         df_out,
#         expected,
#         check_like=True,
#     )
#
#
# def test_fix_autorearded():
#     df = pd.DataFrame(
#         [
#             {"auto_rearded": True},
#             ]
#         )
#
#     expected = pd.DataFrame(
#         [
#             {"auto_rewarded": True},
#             ]
#         )
#
#     df_out = transform.fix_autorearded(df)
#
#     pd.testing.assert_frame_equal(df_out, expected)
#
#
# # @pytest.mark.parametrize("df, expected", [
# #     (
# #         pd.DataFrame(data={
# #             "lick_times": {17: [0.1, 0.2, 0.3, 0.4, ], },
# #         }),
# #         pd.DataFrame(data={
# #             "lick_times": {17: [0.1, 0.2, 0.3, 0.4, ], },
# #             "number_of_licks": {17: 4, },
# #             "lick_rate_Hz": {17: 10.0, },
# #         }, columns=["lick_times", "number_of_licks", "lick_rate_Hz", ]),
# #     ),
# # ])
# # def test_annotate_lick_vigor(df, expected):
# #     pd.util.testing.assert_almost_equal(data.annotate_lick_vigor(df), expected)
