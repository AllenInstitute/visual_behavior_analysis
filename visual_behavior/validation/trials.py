import pandas as pd

from ..schemas.core import TrialSchema
from ..validation import assert_is_valid_dataframe

def assert_is_int(x):
    assert int(x)==x

def validate_change_times(tr):
    if not pd.isnull(tr['change_time']):
        assert tr['starttime'] < tr['change_time']

def validate_change_frame_int(tr):
    if not pd.isnull(tr['change_frame']):
        assert_is_int(tr['change_frame'])

def validate_blank_duration_range_reasonable(tr):
    for t in tr['blank_duration_range']:
        assert t < 10


def validate_trials(trials):

    row_validators = (
        #validate_blank_duration_range_reasonable,
        validate_change_times,
        validate_change_frame_int,
    )

    for validator in row_validators:
        trials.apply(validator, axis=1)

def validate_schema(trials):
    assert_is_valid_dataframe(trials,TrialSchema())


def validate(trials):
    validate_schema(trials)
    validate_trials(trials)
