import pytest
import os
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle

from visual_behavior import utilities


TESTING_DIR = os.path.realpath(os.path.dirname(__file__))
TESTING_RES_DIR = os.path.join(TESTING_DIR, "res")
TEST_SESSION_NAME = "180122094711-task=DoC_NaturalImages_CAMMatched_n=8_stage=natural_images_mouse=M363894.pkl"


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

    pd.testing.assert_frame_equal(
        loaded_df.drop(columns=["filepath", "filename", "training_day", ], index=1),  # filepath and filename will be disagree because they come from different sources
        annotated_trials_df_fixture.drop(columns=["filepath", "filename", ], index=1),
        check_like=True
    )


def test_create_doc_dataframe(
    behavioral_session_output_pickle_fixture,
    annotated_trials_df_fixture
):
    pd.testing.assert_frame_equal(
        utilities.create_doc_dataframe(behavioral_session_output_pickle_fixture).drop(columns=["filepath", "filename", ], index=1),
        annotated_trials_df_fixture.drop(columns=["filepath", "filename", ], index=1),
        check_like=True
    )
