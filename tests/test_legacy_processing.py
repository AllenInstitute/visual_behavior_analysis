import pytest
import os
import numpy as np
from numpy import nan
import pandas as pd
from six.moves import cPickle as pickle

from visual_behavior import legacy_processing


def test_load_trials(behavioral_session_output_fixture, trials_df_fixture):
    pd.testing.assert_frame_equal(
        legacy_processing.load_trials(behavioral_session_output_fixture).sort_index(axis=1),
        trials_df_fixture.sort_index(axis=1)
    )


def test_load_time(behavioral_session_output_fixture):
    np.testing.assert_almost_equal(
        legacy_processing.load_time(behavioral_session_output_fixture)[:11],
        np.array([
            0., 0.04998129, 0.09999849, 0.15004392, 0.20007941, 0.23343854,
            0.26680312, 0.30016385, 0.33352555, 0.36688468, 0.40024766,
        ])
    )


def test_load_rewards(behavioral_session_output_fixture):
    pd.testing.assert_frame_equal(
        legacy_processing.load_rewards(behavioral_session_output_fixture).iloc[:10],
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
        legacy_processing.load_rewards(behavioral_session_output_fixture, legacy_processing.load_time(behavioral_session_output_fixture)).iloc[:10],
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
        legacy_processing.load_running_speed(behavioral_session_output_fixture).iloc[:5],
        EXPECTED_RUNNING_DF,
        check_like=True
    )

    pd.testing.assert_frame_equal(
        legacy_processing.load_running_speed(behavioral_session_output_fixture, time=legacy_processing.load_time(behavioral_session_output_fixture)).iloc[:5],
        EXPECTED_RUNNING_DF,
        check_like=True
    )
