import pytest
import numpy as np
import pandas as pd
from copy import deepcopy

from visual_behavior.translator.core import annotate


@pytest.mark.parametrize("args, expected", [
    (
        (
            pd.DataFrame(data={"a": [0, 1, ], }),
            {"b": "test", "c": "another_test", },
            {"d": "b", },
        ),
        pd.DataFrame(data={"a": [0, 1, ], "d": ["test", "test", ], }),
    ),
    (
        (
            pd.DataFrame(data={"a": [0, 1, ], }),
            {"b": "test", "c": "another_test", },
            None,
        ),
        pd.DataFrame(data={"a": [0, 1, ], }),
    ),
])
def test_annotate_parameters(args, expected):
    pd.testing.assert_frame_equal(
        annotate.annotate_parameters(*args),
        expected,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_annotate_startdatetime(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
        exemplar_metadata_fixture
):
    annotated_trials = annotate.annotate_startdatetime(
        exemplar_trials_fixture,
        exemplar_metadata_fixture
    )

    pd.testing.assert_frame_equal(
        annotated_trials[["startdatetime", "date", "year", "month", "day", "hour", "dayofweek", ]],
        exemplar_extended_trials_fixture[["startdatetime", "date", "year", "month", "day", "hour", "dayofweek", ]],
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_annotate_n_rewards(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture
):
    pd.testing.assert_series_equal(
        annotate.annotate_n_rewards(exemplar_trials_fixture)["number_of_rewards"],
        exemplar_extended_trials_fixture["number_of_rewards"],
        check_index_type=False,
        check_dtype=False
    )


def test_annotate_rig_id(
        monkeypatch,
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
        exemplar_metadata_fixture
):
    monkeypatch.setattr(
        annotate,
        "get_rig_id",
        lambda computer_name: "test_computer_name",
    )

    metadata = deepcopy(exemplar_metadata_fixture)

    # if rig id is present
    pd.testing.assert_series_equal(
        annotate.annotate_rig_id(exemplar_trials_fixture, metadata)["rig_id"],
        exemplar_extended_trials_fixture["rig_id"],
        check_index_type=False,
        check_dtype=False
    )
    # if it isnt
    metadata.pop("rig_id")

    annotate.annotate_rig_id(exemplar_trials_fixture, metadata)["rig_id"].values[0] == "test_computer_name"  # just assume they're all the same


def test_fix_autorearded(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
):
    pd.testing.assert_series_equal(
        annotate.fix_autorearded(exemplar_trials_fixture)["auto_rewarded"],
        exemplar_extended_trials_fixture["auto_rewarded"],
        check_index_type=False,
        check_dtype=False
    )


def test_annotate_cumulative_reward(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
        exemplar_metadata_fixture
):
    annotated_trials = annotate.annotate_cumulative_reward(
        exemplar_trials_fixture,
        exemplar_metadata_fixture
    )

    pd.testing.assert_frame_equal(
        annotated_trials[["cumulative_volume", ]],
        exemplar_extended_trials_fixture[["cumulative_volume", ]],
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_categorize_trials(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture
):
    np.testing.assert_equal(
        annotate.categorize_trials(exemplar_trials_fixture).values,
        exemplar_extended_trials_fixture["trial_type"].values
    )


def test_get_lick_frames(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
        exemplar_licks_fixture
):
    np.testing.assert_equal(
        annotate.get_lick_frames(exemplar_trials_fixture, exemplar_licks_fixture),
        exemplar_extended_trials_fixture["lick_frames"].values
    )


def test_annotate_lick_vigor(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
        exemplar_licks_fixture
):
    annotated_trials = annotate.annotate_lick_vigor(
        exemplar_trials_fixture,
        exemplar_licks_fixture
    )

    pd.testing.assert_frame_equal(
        annotated_trials[["reward_licks", "reward_lick_count", "reward_lick_latency", ]],
        exemplar_extended_trials_fixture[["reward_licks", "reward_lick_count", "reward_lick_latency", ]],
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=False
    )


def test_calculate_latency(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
        exemplar_metadata_fixture
):
    annotated_trials = annotate.annotate_parameters(
        exemplar_trials_fixture,
        exemplar_metadata_fixture,
        keydict={"response_window": "response_window", }
    )  # no non-inplace option

    pd.testing.assert_series_equal(
        annotate.calculate_latency(annotated_trials)["response_latency"],
        exemplar_extended_trials_fixture["response_latency"],
        check_index_type=False,
        check_dtype=False
    )


def test_calculate_reward_rate(
    exemplar_extended_trials_fixture,
    exemplar_trials_fixture,
    exemplar_metadata_fixture,
    exemplar_licks_fixture
):
    """Don't test it with parameters for now...its optional parameters never
    get supplied in `create_extended_dataframe`
    """
    annotated_trials = exemplar_trials_fixture.copy(deep=True)  # this function modifies the original object no matter what...

    annotated_trials = annotate.annotate_startdatetime(
        annotated_trials,
        exemplar_metadata_fixture
    )
    annotated_trials["trial_type"] = annotate.categorize_trials(annotated_trials)

    np.testing.assert_equal(
        annotate.calculate_reward_rate(annotated_trials)["reward_rate"].values,
        exemplar_extended_trials_fixture["reward_rate"].values
    )


def test_check_responses(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
        exemplar_metadata_fixture
):
    annotated_trials = annotate.annotate_parameters(
        exemplar_trials_fixture,
        exemplar_metadata_fixture,
        keydict={"response_window": "response_window", }
    )  # no non-inplace option
    annotated_trials = annotate.calculate_latency(annotated_trials)

    np.testing.assert_equal(
        annotate.check_responses(annotated_trials),
        exemplar_extended_trials_fixture["response"].values
    )


def test_assign_color(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture
):
    annotated_trials = exemplar_trials_fixture.copy(deep=True)
    annotated_trials["trial_type"] = annotate.categorize_trials(annotated_trials)

    np.testing.assert_equal(
        annotate.assign_color(annotated_trials),
        exemplar_extended_trials_fixture["color"].values
    )


def test_get_response_type(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
        exemplar_metadata_fixture
):
    annotated_trials = annotate.annotate_parameters(
        exemplar_trials_fixture,
        exemplar_metadata_fixture,
        keydict={"response_window": "response_window", }
    )  # no non-inplace option
    annotated_trials = annotate.calculate_latency(annotated_trials)
    annotated_trials['response'] = annotate.check_responses(annotated_trials)

    np.testing.assert_equal(
        annotate.get_response_type(annotated_trials),
        exemplar_extended_trials_fixture["response_type"].values
    )


def test_remove_repeated_licks(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
        exemplar_licks_fixture
):
    annotated_trials = exemplar_trials_fixture.copy(deep=True)
    annotated_trials["lick_frames"] = annotate.get_lick_frames(
        annotated_trials,
        exemplar_licks_fixture
    )

    pd.testing.assert_frame_equal(
        annotate.remove_repeated_licks(annotated_trials)[["lick_times", "lick_frames", ]],
        exemplar_extended_trials_fixture[["lick_times", "lick_frames", ]],
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=False
    )
