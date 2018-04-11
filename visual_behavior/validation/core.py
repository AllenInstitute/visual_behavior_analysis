import pandas as pd

from ..schemas.core import TrialSchema
from ..validation import is_valid_dataframe


def validate_change_times(tr):
    if ~pd.isnull(tr['change_time']):
        assert tr['starttime'] < tr['change_time']

def validate_blank_duration_range_reasonable(tr):
    for t in tr['blank_duration_range']:
        assert t < 10


def validate_trials(trials):

    row_validators = (
        validate_blank_duration_range_reasonable,
        validate_change_times,
    )

    for validator in row_validators:
        trials.apply(validator, axis=0)


def validate_schema(trials):
    assert is_valid_dataframe(trials, TrialSchema())


def validate(trials):
    validate_schema(trials)
    validate_trials(trials)
