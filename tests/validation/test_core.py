import numpy as np
import pandas as pd
from visual_behavior.validation.core import *


def test_parse_log():

    EXPECTED = dict(
        levelname="ERROR",
        name="package.module",
        message="This is the error"
    )

    parsed = parse_log("{levelname}::{name}::{message}".format(**EXPECTED))
    assert parsed == EXPECTED


def test_count_read_errors():

    EMPTY = dict(
        log=[]
    )
    results = count_read_errors(EMPTY)
    print(results)
    assert 'ERROR' not in results

    INFO = dict(
        log=["INFO::package.module::informative message"]
    )

    results = count_read_errors(INFO)
    print(results)
    assert 'ERROR' not in results
    assert results['INFO'] == 1


def test_validate_no_read_errors():

    WITH_ERRORS = dict(
        log=["ERROR::package.module::error message"]
    )

    WITHOUT_ERRORS = dict(
        log=["INFO::package.module::informative message"]
    )

    assert validate_no_read_errors(WITH_ERRORS) == False
    assert validate_no_read_errors(WITHOUT_ERRORS) == True


def test_validate_running_data():
    # good data: length matches time and not all values the same
    GOOD_DATA = {}
    GOOD_DATA['running'] = pd.DataFrame({
        'speed': [3, 4, 5]
    })
    GOOD_DATA['time'] = np.array([1, 2, 3])
    assert validate_running_data(GOOD_DATA) == True

    # bad data: length does not matche time
    BAD_DATA = {}
    BAD_DATA['running'] = pd.DataFrame({
        'speed': [3, 4, 5, 6]
    })
    BAD_DATA['time'] = np.array([1, 2, 3])
    assert validate_running_data(BAD_DATA) == False

    # bad data: all values of speed are zero
    BAD_DATA = {}
    BAD_DATA['running'] = pd.DataFrame({
        'speed': [0, 0, 0]
    })
    BAD_DATA['time'] = np.array([1, 2, 3])
    assert validate_running_data(BAD_DATA) == False


def test_validate_licks():
    GOOD_DATA = {}
    GOOD_DATA['licks'] = pd.DataFrame({
        'time': [10, 20],
        'frame': [1, 2]
    })
    assert validate_licks(GOOD_DATA, lick_spout_present=True) == True

    BAD_DATA = {}
    BAD_DATA['licks'] = pd.DataFrame({
        'time': [],
        'frame': []
    })
    assert validate_licks(BAD_DATA, lick_spout_present=True) == False

    NO_LICKSPOUT_DATA = {}
    NO_LICKSPOUT_DATA['licks'] = pd.DataFrame({
        'time': [],
        'frame': []
    })
    assert validate_licks(NO_LICKSPOUT_DATA, lick_spout_present=False) == True


def test_validate_minimal_dropped_frames():
    GOOD_DATA = pd.DataFrame({
        'time': [0, 1. / 60, 2. / 60, 3. / 60]
    })
    assert validate_minimal_dropped_frames(GOOD_DATA) == True

    BAD_DATA = pd.DataFrame({
        'time': [0, 1. / 60, 3. / 60, 4. / 60]
    })
    assert validate_minimal_dropped_frames(BAD_DATA) == False


def test_validate_omitted_flashes():
    GOOD_DATA = {
        'omitted_stimuli': pd.DataFrame({
            'frame': [
                646,
                871,
                1051,
                1141,
                1456,
            ],
            'time': [
                10.773585,
                14.526905,
                17.529540,
                19.030820,
                24.285160,
            ],
        }),
        'visual_stimuli': pd.DataFrame({
            'duration': [
                0.250409,
                0.250187,
                0.250102,
                0.250097,
                0.250148,
            ],
            'end_frame': [
                75,
                120,
                165,
                210,
                255,
            ],
            'frame': [
                760,
                105,
                150,
                195,
                240,
            ],
            'image_category': [
                'im065',
                'im065',
                'im065',
                'im065',
                'im065',
            ],
            'image_name': [
                'im065',
                'im065',
                'im065',
                'im065',
                'im065',
            ],
            'orientation': [
                np.NAN,
                np.NAN,
                np.NAN,
                np.NAN,
                np.NAN,
            ],
            'time': [
                0.998457,
                1.748978,
                2.499544,
                3.250245,
                4.000947,
            ]
        }),
    }
    BAD_DATA = {
        'omitted_stimuli': pd.DataFrame({
            'frame': [
                915,
                1050,
            ],
            'time': [
                15.258188,
                17.510235,
            ]
        }),
        'visual_stimuli': pd.DataFrame({
            'duration': [
                0.248559,
                0.238798,
                0.248601
            ],
            'end_frame': [
                1020,
                1066,
                1111,
            ],
            'frame': [
                1006,
                1051,
                1096,
            ],
            'image_category': [
                'im065',
                'im065',
                'im065',
            ],
            'image_name': [
                'im065',
                'im065',
                'im065',
            ],
            'orientation': [
                np.NAN,
                np.NAN,
                np.NAN,
            ],
            'time': [
                16.777,  # approximate...i took a bad screenshot...:(
                17.5383,
                18.27894,
            ]
        }),
    }

    assert validate_omitted_flashes(GOOD_DATA) is True and \
        validate_omitted_flashes(BAD_DATA) is False
