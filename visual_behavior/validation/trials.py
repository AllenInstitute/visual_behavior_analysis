import pandas as pd

from ..schemas.core import TrialSchema
from .utils import assert_is_valid_dataframe


def assert_is_int(x):
    assert int(x) == x


def validate_change_times(tr):
    if not pd.isnull(tr['change_time']):
        assert tr['starttime'] < tr['change_time']


def validate_change_frame_int(tr):
    if not pd.isnull(tr['change_frame']):
        assert_is_int(tr['change_frame'])


def validate_blank_duration_range_reasonable(tr):
    for t in tr['blank_duration_range']:
        assert t < 10


def validate_orientation(tr):
    # checks to make sure all orientation data is present
    raise NotImplementedError


def validate_images(tr):
    # checks to make sure all image data is present
    raise NotImplementedError


def validate_stimulus(tr):
    # checks to make sure one of orientation or images are present, not both
    # stimulus data only expected on "change" trials (e.g. when "change_frame" is defined)
    raise NotImplementedError


def validate_cumulative_rewards(trials):
    # cumulative_reward_number should increase monotonically
    # cumulative_volume should increase monotonically
    raise NotImplementedError


def validate_lick_times(tr):
    # all lick times should be greater than start_time
    # if there is a change, all lick times should be greater than change time
    raise NotImplementedError


def validate_response_latency(tr):
    # if change, this should be positive, else empty
    raise NotImplementedError


def validate_reward_times(tr):
    # list of floats
    # all times should be greater than start_time
    # all times should be greater than or equal to change time
    # if not auto-reward, should be greater than first lick
    raise NotImplementedError


def validate_trials(trials):
    row_validators = (
        # validate_blank_duration_range_reasonable,
        validate_change_times,
        validate_change_frame_int,
    )

    for validator in row_validators:
        trials.apply(validator, axis=1)


def validate_schema(trials):
    assert_is_valid_dataframe(trials, TrialSchema())


def validate(trials):
    validate_schema(trials)
    validate_trials(trials)
