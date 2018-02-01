import pytest
import numpy as np
import pandas as pd

from visual_behavior import masks


@pytest.mark.parametrize(
    "annotated_trials_df_fixture, trial_types, expected",
    [
        ((), None, np.array([True, True, True, True, True, ]), ),
        ((), ["aborted"], pd.Series(data=[True, True, True, True, True, ], name="trial_type", dtype="bool"), ),
    ],
    indirect=["annotated_trials_df_fixture", ]
)
def test_trial_types(annotated_trials_df_fixture, trial_types, expected):
    if isinstance(expected, pd.Series):
        assert masks.trial_types(annotated_trials_df_fixture, trial_types).iloc[:5].equals(expected)
    else:
        np.testing.assert_equal(
            masks.trial_types(annotated_trials_df_fixture, trial_types)[:5],
            expected
        )


def test_contingent_trials(annotated_trials_df_fixture):
    assert masks.contingent_trials(annotated_trials_df_fixture).iloc[:5] \
        .equals(pd.Series(data=[False, False, False, False, False, ], name="trial_type", dtype="bool"))


@pytest.mark.parametrize("thresh, expected", [
    (
        2.0,
        pd.Series(data=[True, True, True, True, True, ], name="reward_rate", dtype="bool"),
    ),
    (
        0.0,
        pd.Series(data=[True, True, True, True, True, ], name="reward_rate", dtype="bool"),
    ),
])
def test_reward_rate(annotated_trials_df_fixture, thresh, expected):
    assert masks.reward_rate(annotated_trials_df_fixture, thresh).iloc[:5].equals(expected)
