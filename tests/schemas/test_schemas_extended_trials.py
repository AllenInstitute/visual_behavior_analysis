import pytest
import pandas as pd

from visual_behavior.schemas import extended_trials


def test_dump_load_schema(exemplar_extended_trials_fixture):
    extended_trial_schema = extended_trials.ExtendedTrialSchema()

    exemplar_extended_trials_records = (
        exemplar_extended_trials_fixture
        .to_dict(orient='records')
    )

    serialized_extended_trials = extended_trial_schema.dumps(
        exemplar_extended_trials_records,
        many=True,
    )
    deserialized_extended_trials_records = extended_trial_schema.loads(
        serialized_extended_trials,
        many=True,
    )

    # Useful for debugging
    # import json
    # print exemplar_extended_trials_records[0]['startdatetime']
    # print json.loads(serialized_extended_trials)[0]['startdatetime']
    # print deserialized_extended_trials_records[0]['startdatetime']
    assert exemplar_extended_trials_records[0] == deserialized_extended_trials_records[0]

    deserialized_extended_trials = pd.DataFrame(deserialized_extended_trials_records)

    pd.testing.assert_frame_equal(
        exemplar_extended_trials_fixture,
        deserialized_extended_trials,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )

def test_extended_trial_validation(exemplar_extended_trials_fixture):
    extended_trial_schema = extended_trials.ExtendedTrialSchema()

    exemplar_extended_trials_records = (
        exemplar_extended_trials_fixture
        .to_dict(orient='records')
    )

    # print type(exemplar_extended_trials_fixture.iloc[0]['startdatetime'])
    # print type(exemplar_extended_trials_records[0]['startdatetime'])


    errors = extended_trial_schema.validate(exemplar_extended_trials_records, many=True)
    assert len(errors) == 0
    errors = extended_trial_schema.validate([{'foo': 'bar'}], many=True)
    assert len(errors) == 1

    one_record = exemplar_extended_trials_records[0]
    one_record['behavior_session_uuid'] = ''
    errors = extended_trial_schema.validate(one_record)
    assert errors == {'behavior_session_uuid': ['Not a valid UUID.']}
