import pytest
import pandas as pd

from visual_behavior.schemas import extended_trials


def test_dump_load_schema(exemplar_extended_trials_fixture):
    extended_trial_schema = extended_trials.ExtendedTrialSchema()

    serialized_extended_trials = extended_trial_schema.dumps(exemplar_extended_trials_fixture, many=True).data
    deserialized_extended_trials = extended_trial_schema.loads(serialized_extended_trials, many=True).data

    # print(deserialized_extended_trials["change_frame"].iloc[3])

    left = deserialized_extended_trials.iloc[0].index
    right = exemplar_extended_trials_fixture.iloc[0].index

    print set(left) - set(right)
    print set(right) - set(left)

    pd.testing.assert_frame_equal(
        deserialized_extended_trials,
        exemplar_extended_trials_fixture,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )
