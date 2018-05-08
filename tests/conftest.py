import pytest
import os
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


@pytest.fixture(scope="module")
def trials_df_fixture():
    trials = pd.read_pickle(os.path.join(TESTING_RES_DIR, "trials.pkl"))
    trials["scheduled_change_time"] = trials.apply(
        lambda row: row["scheduled_change_time"] - row["starttime"],
        axis=1
    )  # change scheduled_change_time from time relative to experiment start to time relative to trial start
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

    for set_attr, set_value, set_frame, set_time in old_set_log:
        new_set_log.append((set_attr, set_value, set_time, set_frame, ))

    foraging2_data["items"]["behavior"]["stimuli"]["gratings"]["set_log"] = \
        new_set_log

    return foraging2_data


@pytest.fixture(scope="session")
def foraging2_natural_images_data_fixture():
    foraging2_data = pd.read_pickle(
        os.path.join(TESTING_RES_DIR, "foraging2_natural_images.pkl")
    )

    # this is the hack way we are using to edit the stim_log item ordering due to derricw new changes
    old_set_log = foraging2_data["items"]["behavior"]["stimuli"]["natural_images"]["set_log"]
    new_set_log = []

    for set_attr, set_value, set_frame, set_time in old_set_log:
        new_set_log.append((set_attr, set_value, set_time, set_frame, ))

    foraging2_data["items"]["behavior"]["stimuli"]["natural_images"]["set_log"] = \
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
