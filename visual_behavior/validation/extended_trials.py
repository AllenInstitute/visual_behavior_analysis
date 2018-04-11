import pandas as pd

from .core import validate_trials
from ..schemas.extended_trials import ExtendedTrialSchema
from ..validation import is_valid_dataframe


def validate_schema(extended_trials):
    assert is_valid_dataframe(extended_trials, ExtendedTrialSchema())


def validate_aborted_change_time(tr):
    if tr['trial_type'] == 'aborted':
        assert pd.isnull(tr['change_time'])


def validate_extended_trials(extended_trials):

    row_validators = (
        validate_aborted_change_time,
    )

    for validator in row_validators:
        extended_trials.apply(validator, axis=0)


def validate(extended_trials):
    validate_schema(extended_trials)
    validate_trials(extended_trials)
    validate_extended_trials(extended_trials)
