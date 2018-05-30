import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
from visual_behavior.translator.core.annotate import annotate_startdatetime, \
    make_deterministic_session_uuid, make_trials_contiguous

def test_annotate_startdatetime():

    ISO_TIMESTAMP = '2018-05-22T20:55:42.118000-07:00'

    trials = pd.DataFrame({'foo':[1,2,3],'bar':['a','b','c']})
    metadata = dict(startdatetime=ISO_TIMESTAMP)

    annotated_trials = annotate_startdatetime(trials, metadata)

    assert annotated_trials['startdatetime'][0].isoformat()==ISO_TIMESTAMP

def test_make_deterministic_session_uuid():
    metadata = {
        'mouseid': 'M999999',
        'startdatetime': '2018-05-22T20:55:42.118000-07:00',
    }
    EXPECTED_UUID = '1b48889e-465d-5732-a32e-cc2f67d2f581'
    behavior_session_uuid = make_deterministic_session_uuid(metadata)

    assert str(behavior_session_uuid) == EXPECTED_UUID

    metadata = {
        'mouseid': 'M999999',
        'startdatetime': '2018-05-23T03:55:42.118000+00:00',
    }
    EXPECTED_UUID = '1b48889e-465d-5732-a32e-cc2f67d2f581'
    behavior_session_uuid = make_deterministic_session_uuid(metadata)

    assert str(behavior_session_uuid) == EXPECTED_UUID

def test_make_trials_contiguous():

    EXPECTED_TRIALS = pd.DataFrame(dict(
        starttime=[0, 0.1, 0.2],
        endtime=[0.1, 0.2, 0.99],
        startframe=[0, 10, 20],
        endframe=[10, 20, 99],
        trial_length=[0.1, 0.1, 0.79],
    ))

    input_trials = pd.DataFrame(dict(
        starttime=[0, 0.1, 0.2],
        endtime=[0.05, 0.15, 0.25],
        startframe=[0, 10, 20],
        endframe=[5, 15, 25],
    ))

    time = np.arange(0,1,0.01)

    output_trials = make_trials_contiguous(input_trials,time)

    assert_frame_equal(output_trials, EXPECTED_TRIALS, check_dtype=False)
