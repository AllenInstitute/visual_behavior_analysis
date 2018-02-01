import pytest
import os
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle

from visual_behavior import utilities


TESTING_DIR = os.path.realpath(os.path.dirname(__file__))
TESTING_RES_DIR = os.path.join(TESTING_DIR, "res")
TEST_SESSION_NAME = "180122094711-task=DoC_NaturalImages_CAMMatched_n=8_stage=natural_images_mouse=M363894.pkl"


def assert_frames_equal(actual, expected, use_close=False):
    """
    Compare DataFrame items by index and column and
    raise AssertionError if any item is not equal.

    Ordering is unimportant, items are compared only by label.
    NaN and infinite values are supported.

    Parameters
    ----------
    actual : pandas.DataFrame
    expected : pandas.DataFrame
    use_close : bool, optional
        If True, use numpy.testing.assert_allclose instead of
        numpy.testing.assert_equal.

    Notes
    -----
    - stolen from:
        https://gist.github.com/jiffyclub/ac2e7506428d5e1d587b
    """
    if use_close:
        comp = np.testing.assert_allclose
    else:
        comp = np.testing.assert_equal

    assert (isinstance(actual, pd.DataFrame) and
            isinstance(expected, pd.DataFrame)), \
        'Inputs must both be pandas DataFrames.'

    for i, exp_row in expected.iterrows():
        assert i in actual.index, 'Expected row {!r} not found.'.format(i)

        act_row = actual.loc[i]

        for j, exp_item in exp_row.iteritems():
            assert j in act_row.index, \
                'Expected column {!r} not found.'.format(j)

            act_item = act_row[j]

            try:
                comp(act_item, exp_item)
            except AssertionError as e:
                raise AssertionError(
                    e.message + '\n\nColumn: {!r}\nRow: {!r}'.format(j, i))


def test_load_from_folder(
    tmpdir,
    behavioral_session_output_fixture,
    annotated_trials_df_fixture
):
    output_dir_pypath = tmpdir.mkdir("data")
    pickle_pypath = output_dir_pypath.join("p.pkl")
    with open(str(pickle_pypath), "wb") as pstream:
        pickle.dump(behavioral_session_output_fixture, pstream, protocol=2)

    loaded_df = utilities.load_from_folder(str(output_dir_pypath), False, False)

    assert_frames_equal(
        loaded_df.drop(columns=["filepath", "filename", ], index=1),  # filepath and filename will be disagree because they come from different sources
        annotated_trials_df_fixture.drop(columns=["filepath", "filename", ], index=1)
    )


def test_create_doc_dataframe(
    behavioral_session_output_pickle_fixture,
    annotated_trials_df_fixture
):
    assert_frames_equal(
        utilities.create_doc_dataframe(behavioral_session_output_pickle_fixture).drop(columns=["filepath", "filename", ], index=1),
        annotated_trials_df_fixture.drop(columns=["filepath", "filename", ], index=1)
    )
