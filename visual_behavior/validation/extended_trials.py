import logging
import numpy as np
import pandas as pd


from .trials import validate_trials
# from ..schemas.extended_trials import ExtendedTrialSchema
# from ..validation import is_valid_dataframe

logger = logging.getLogger(__name__)


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


# UTIL
def is_equal(v1, v2):
    """checks equality, but deals with nulls (e.g., will return True for
    np.nan == None)
    """
    return (None if pd.isnull(v1) else v1) == (None if pd.isnull(v2) else v2)


def get_warmup_trials(trials):
    '''finds all warmup trials'''
    first_non_warmup_trial = trials[trials['auto_rewarded'] == False].index[0]
    return trials.iloc[:first_non_warmup_trial]


def get_first_lick_in_response_window(row):
    '''returns first lick that falls in response window, nan if no such lick exists'''
    licks = np.array(row['lick_times']) - row['change_time']
    licks_in_window = licks[np.logical_and(
        licks >= row['response_window'][0],
        licks <= row['response_window'][1]
    )]

    if len(licks_in_window) > 0:
        return licks_in_window[0]
    else:
        return np.nan


def get_first_lick_relative_to_scheduled_change(row):
    """returns first lick relative to scheduled change time, nan if no such lick
    exists
    """
    licks = np.array(row['lick_times']) - row['scheduled_change_time']
    return licks[0] if len(licks) > 0 else np.nan


def get_first_lick_relative_to_change(row):
    '''returns first lick relative to scheduled change time, nan if no such lick exists'''
    licks = np.array(row['lick_times']) - row['change_time']
    return licks[0] if len(licks) > 0 else np.nan


# TRIAL
def validate_min_change_time(trial):
    '''change time in trial should never be less than pre_change_time'''
    change_time = trial["change_time"]
    return change_time - trial["starttime"] > trial["prechange_minimum"] \
        if np.isfinite(change_time) else True  # check is not nan, inf


def validate_max_change_time(trial, stimulus_window):
    '''Changes should never occur at a time greater than `pre_change_time` + `stimulus_window`'''
    change_time = trial["change_time"]
    return change_time - trial["starttime"] < trial["prechange_minimum"] + stimulus_window \
        if np.isfinite(change_time) else True


def validate_auto_reward_volume(trial, auto_reward_volume):
    '''Each dispense of water for autorewards should be `auto_reward_vol`.'''
    if trial["auto_rewarded"] and trial["number_of_rewards"] > 0:
        return trial["reward_volume"] == auto_reward_volume
    else:
        return True


# TRIALS
def validate_number_of_warmup_trials(trials, expected_warmup_trial_count=None):
    warmup_trials = get_warmup_trials(trials)

    if expected_warmup_trial_count:
        return len(warmup_trials[warmup_trials.trial_type == 'go']) == \
            expected_warmup_trial_count
    else:
        return len(warmup_trials[warmup_trials.trial_type == 'go']) == \
            len(trials[trials.trial_type == 'go'])


def validate_reward_delivery_on_warmup_trials(trials, tolerance=0.001):
    '''all warmup trials should have rewards presented simultaneously with the
    change image
    '''
    # find all of the warmup trials
    warmup_trials = get_warmup_trials(trials)

    # find only go warmup trials
    go_warmup_trials = warmup_trials[warmup_trials.trial_type == 'go']

    # get time difference between each reward and change_time
    try:
        # check to see that all time differences fall within threshold
        return all((
            abs(rt[0] - ct) < tolerance
            for rt, ct in zip(go_warmup_trials.reward_times, go_warmup_trials.change_time)
        ))
    except:
        # if the above fails, fail the test
        return False


def validate_autorewards_after_N_consecutive_misses(
        trials,
        autoreward_after_consecutive_misses=10
):
    go_trials = trials[trials.trial_type == 'go']
    auto_reward_when_expected = []
    consecutive_misses = 0

    for idx,row in go_trials.iterrows():
        miss = 1 if row['response'] == 0 else 0
        auto_rewarded = 1 if row['auto_rewarded'] else 0
        consecutive_misses = (consecutive_misses*miss+miss) * abs(1 - auto_rewarded)
        if consecutive_misses > autoreward_after_consecutive_misses:
            # these are trials when an autoreward is expected
            auto_reward_when_expected.append(row['auto_rewarded'])

    # all trials with expected autorewards should be True
    return all(auto_reward_when_expected)


def validate_change_on_all_go_trials(trials):
    '''ensure that either the orientation or the image_name changes on every go
    trial
    '''
    for idx, row in trials[trials.trial_type == "go"].iterrows():
        if is_equal(row["initial_image_name"], row["change_image_name"]) and \
                is_equal(row["initial_ori"], row["change_ori"]):
            return False
    else:
        return True


def validate_no_change_on_all_catch_trials(trials):
    '''ensure that neither the orientation nor the image_name changes on every
    catch trial
    '''
    for idx, row in trials[trials.trial_type == 'catch'].iterrows():
        if not is_equal(row['initial_image_name'], row['change_image_name']) or \
                not is_equal(row['initial_ori'],row['change_ori']):
            return False
    else:
        return True


def validate_intial_and_final_in_non_aborted(trials):
    '''all non-aborted trials should have either an intial and final image OR an
    initial and final orientation
    '''
    non_aborted_trials = trials[trials.trial_type != 'aborted']

    all_initial = all(~pd.isnull(non_aborted_trials['initial_image_name'])) or \
        all(~pd.isnull(non_aborted_trials['initial_ori']))

    all_final = all(~pd.isnull(non_aborted_trials['change_image_name'])) or \
        all(~pd.isnull(non_aborted_trials['change_ori']))

    return all_initial and all_final


def validate_reward_follows_first_lick_in_window(trials, tolerance=0.001):
    '''reward should happen immediately (within tolerance) of first lick in window on
    non-autorewarded go trials
    '''
    # get all go_trials without auto_rewards
    contingent_go_trials = trials[(trials.trial_type == 'go') & (trials.auto_rewarded == False)]

    # get first lick in response window, relative to change time
    first_lick_in_window = contingent_go_trials.apply(get_first_lick_in_response_window, axis=1)
    first_lick_in_window.name = 'first_lick'

    # get reward time, relative to change time
    reward_time = contingent_go_trials['reward_times'] - contingent_go_trials['change_time']
    reward_time.name = 'reward_time'

    # check that, whenever a first lick exists, a reward was given with tolerance
    for idx, row in pd.concat([first_lick_in_window, reward_time], axis=1).iterrows():
        # if there was no lick, there should be no reward
        if pd.isnull(row['first_lick']):
            if len(row['reward_time'] > 0):
                return False
        # if there was a lick, the reward should happen immediately
        elif abs(row['first_lick'] - row['reward_time'][0]) > tolerance:
            return False

    return True


def validate_never_more_than_one_reward(trials):
    '''Only a maximum of one reward should be delivered per trial'''
    return np.max(trials['number_of_rewards']) <= 1


def validate_lick_before_scheduled_on_aborted_trials(trials):
    '''if licks occur before a scheduled change time/flash, the trial ends
    Therefore, every aborted trial should have a lick before the scheduled
    change time
    '''
    aborted_trials = trials[trials.trial_type == 'aborted']
    first_lick = aborted_trials.apply(
        get_first_lick_relative_to_scheduled_change,
        axis=1
    )
    # don't use nanmax. If there's a nan, we expect this to fail
    return np.max(first_lick.values) < 0


def validate_lick_after_scheduled_on_go_catch_trials(trials):
    '''if licks occur before a scheduled change time/flash, the trial ends
    Therefore, no non-aborted trials should have a lick before the scheduled
    change time
    '''
    nonaborted_trials = trials[trials.trial_type != 'aborted']
    first_lick = nonaborted_trials.apply(
        get_first_lick_relative_to_scheduled_change,
        axis=1
    )
    # use nanmin, it is possible
    return np.nanmin(first_lick.values) > 0


def validate_initial_matches_final(trials):
    trials_to_test = trials[trials['trial_type'].isin(['go', 'catch'])]
    last_trial_final_im = trials_to_test.iloc[0]['initial_image_name']
    last_trial_final_ori = trials_to_test.iloc[0]['initial_ori']
    image_match = []
    ori_match = []
    for idx, row in trials_to_test.iterrows():
        image_match.append(
            is_equal(row['initial_image_name'], last_trial_final_im)
        )
        ori_match.append(
            is_equal(row['initial_ori'], last_trial_final_ori)
        )

        last_trial_final_im = row['change_image_name']
        last_trial_final_ori = row['change_ori']

    return all(image_match) and all(ori_match)


def validate_first_lick_after_change_on_nonaborted(trials):
    '''on GO and CATCH trials, licks should never be observed between the trial
    start and the first frame of an image change
    '''
    non_aborted_trials = trials[trials.trial_type != 'aborted']
    first_lick_relative_to_change=non_aborted_trials.apply(
        get_first_lick_relative_to_change,
        axis=1
    )
    return np.nanmin(first_lick_relative_to_change.values) > 0


def validate_trial_ends_without_licks(trials, minimum_no_lick_time):
    '''There should never be a lick within 'minimum_no_lick_time' of trial end
    Task logic should extend trial if mouse is licking near trial end
    '''
    non_aborted_trials = trials[trials.trial_type.isin(['go', 'catch'])]
    for idx, row in non_aborted_trials.iterrows():
        # fail if the last lick is within 'minimum_no_lick_time' of the endtime
        if len(row['lick_times']) > 0 and \
                row['endtime'] - row['lick_times'][-1] < minimum_no_lick_time:
            return False
    return True


def validate_session_within_expected_duration(
        trials, expected_duration_seconds, tolerance=30
):
    '''ensure that last trial end time does not exceed expected duration by more
    than 30 seconds
    '''
    return trials['endtime'].values[-1] < (expected_duration_seconds + tolerance)


def validate_session_ends_at_max_cumulative_volume(
        trials, volume_limit, tolerance=0.01
):
    '''ensure that earned volume does not exceed volume limit
    build in a tolerance in case reward limit is not even multiple of reward
    volumes
    '''
    return trials['cumulative_volume'].max() <= (volume_limit + tolerance)


def validate_duration_and_volume_limit(
        trials, expected_duration, volume_limit, time_tolerance=30
):
    '''The length of a session should not be less than `duration` by more than
    30 seconds UNLESS `volume_limit` has been met
    '''
    if trials.cumulative_volume.max() > volume_limit:
        # return True if volume limit is met
        return True
    else:
        if trials['endtime'].values[-1] < (expected_duration - time_tolerance):
            # return False if end time is less than duration by more than 30 seconds
            return False
        else:
            # otherwise, return True
            return True
