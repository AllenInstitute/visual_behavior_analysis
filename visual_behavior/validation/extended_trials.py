import numpy as np
import pandas as pd

from ..schemas.extended_trials import ExtendedTrialSchema
from .utils import assert_is_valid_dataframe, nanis_equal, all_close


def validate_schema(extended_trials):
    assert_is_valid_dataframe(extended_trials, ExtendedTrialSchema())


def _check_aborted_change_time(tr):
    if tr['trial_type'] == 'aborted':
        return pd.isnull(tr['change_time'])
    else:
        return True


def validate_aborted_change_time(trials):
    return all(trials.apply(_check_aborted_change_time, axis=1))


# helper functions for tests:
def get_autoreward_trials(trials):
    '''finds all auto-reward trials'''
    return trials[trials['auto_rewarded'] == True]  # noqa: E712


def get_warmup_trials(trials):
    '''finds all warmup trials'''
    first_non_warmup_trial_df = trials[trials['auto_rewarded'] != True]
    if len(first_non_warmup_trial_df) > 0:
        first_non_warmup_trial = first_non_warmup_trial_df.index[0]  # noqa: E712
        return trials.iloc[:first_non_warmup_trial]
    else:
        return pd.DataFrame()


def get_first_lick_in_response_window(row):
    '''
    returns first lick that falls in response window, nan if no such lick
    exists
    '''
    licks = np.array(row['lick_times']) - row['change_time']
    licks_in_window = licks[
        np.logical_and(
            licks >= row['response_window'][0],
            licks <= row['response_window'][1],
        )
    ]
    if len(licks_in_window) > 0:
        return licks_in_window[0]
    else:
        return np.nan

def get_first_lick_relative_to_change(row):
    '''
    returns first lick relative to scheduled change time, nan if no such lick
    exists
    '''
    licks = np.array(row['lick_times']) - row['change_time']
    if len(licks) > 0:
        return licks[0]
    else:
        return np.nan


def get_first_lick_relative_to_scheduled_change(row):
    '''
    returns first lick relative to scheduled change time, nan if no such lick
    exists
    '''
    licks = np.array(row['lick_times']) - row['scheduled_change_time'] - row['starttime']
    if len(licks) > 0:
        return licks[0]
    else:
        return np.nan


def identify_consecutive_aborted_blocks(trials, failure_repeats):
    '''
    adds columns to the dataframe that:
        1 - track the number of consecutive aborted trials
        2 - assigns a unique integer to each 'failure_repeats' long block of aborted trials
    '''
    # instantiate some counters and empty lists
    # first counter will keep track of consecutive aborted trials
    consecutive_aborted = 0
    consecutive_aborted_col = []
    # this counter will assign a unique integer to every block aborted trials
    # that are even multiples of 'failure_repeats'
    consecutive_aborted_should_match = 0
    consecutive_aborted_should_match_col = []

    # iterate through trials
    for idx, row in trials.iterrows():
        # check whether or not to iterate consecutive aborted
        if row['trial_type'] == 'aborted':
            # increment an integer that keeps track of which block of aborted
            # trials should have matching change times
            if consecutive_aborted % (failure_repeats + 1) == 0:  # a new block whenever modulus = 0
                consecutive_aborted_should_match += 1
            consecutive_aborted += 1
        else:
            consecutive_aborted = 0

        consecutive_aborted_col.append(consecutive_aborted)
        consecutive_aborted_should_match_col.append(consecutive_aborted_should_match)

    # append columns to dataframe to allow slicing
    trials['consecutive_aborted'] = consecutive_aborted_col
    trials['consecutive_aborted_should_match'] = consecutive_aborted_should_match_col

    return trials


def identify_licks_in_response_window(row):
    '''a method for counting licks in the response window'''
    if len(row['lick_times']) > 0 and pd.isnull(row['change_time']) == False:
        licks_relative_to_change = np.array(row['lick_times']) - row['change_time']
        licks_in_window = licks_relative_to_change[
            np.logical_and(
                licks_relative_to_change >= row['response_window'][0],
                licks_relative_to_change <= row['response_window'][1]
            )
        ]
        return len(licks_in_window)
    else:
        return 0


def count_stimuli_per_trial(trials, visual_stimuli):
    '''
    iterates over trials
    counts stimulus groups in range of trial frames for each trial
    returns vector with all numbers of stimuli per trial
    '''
    # preallocate array
    stimuli_per_trial = np.empty(len(trials))

    # iterate over dataframe
    for idx, trial in trials.reset_index().iterrows():
        # get visual stimuli in range
        stimuli = visual_stimuli[
            (visual_stimuli['frame'] >= trial['startframe']) &
            (visual_stimuli['frame'] <= trial['endframe'])
        ]['image_category'].unique()
        # add to array
        stimuli_per_trial[idx] = len(stimuli)
    return stimuli_per_trial


# test functions
def validate_autoreward_volume(trials, auto_reward_volume):
    '''Each dispense of water for autorewards should be `auto_reward_vol`.'''

    # get all auto reward trials
    auto_reward_trials = get_autoreward_trials(trials)
    # check to see if any trials contain a reward volume that does not match
    # expected volume
    return all(auto_reward_trials[auto_reward_trials.number_of_rewards > 0].reward_volume == auto_reward_volume)


def validate_number_of_warmup_trials(trials, expected_number_of_warmup_trials):
    '''
    The first `warmup_trials` go-trials (excluding aborted trials) of the
    session should be change trials

    The number of warmup trials should not exceed `warmup_trials` UNLESS
    `warmup_trials` is -1, in which case all non-aborted trials should be
    autoreward trials
    '''

    # find all of the warmup trials
    warmup_trials = get_warmup_trials(trials)

    # check to ensure that the number of GO warmup trials matches the expected number
    if len(warmup_trials) == 0 and expected_number_of_warmup_trials > 0:
        # if there are fewer go/catch trials than the expected number of warmup trials, return True
        # This will result if so many trials were aborted that the number of warmup trials were never met
        if len(trials[trials.trial_type.isin(['go', 'catch'])]) < expected_number_of_warmup_trials:
            return True
        else:
            return False
    if len(warmup_trials) == 0 and expected_number_of_warmup_trials == 0:
        return True
    elif expected_number_of_warmup_trials != -1:
        return len(warmup_trials[warmup_trials.trial_type == 'go']) == expected_number_of_warmup_trials
    # if -1, all will be warmup trials. 
    elif expected_number_of_warmup_trials == -1:
        return len(trials[trials.auto_rewarded==True])==len(trials)


def validate_reward_delivery_on_warmup_trials(trials, tolerance=0.001):
    '''
    all warmup trials should have rewards presented simultaneously with the
    change image
    '''

    # find all of the warmup trials
    warmup_trials = get_warmup_trials(trials)

    # can only perform this validation if there were warmup trials
    if len(warmup_trials) > 0:
        # find only go warmup trials
        go_warmup_trials = warmup_trials[warmup_trials.trial_type == 'go']

        # get time difference between each reward and change_time
        try:
            time_delta = [
                abs(rt[0] - ct)
                for rt, ct
                in zip(go_warmup_trials.reward_times, go_warmup_trials.change_time)
            ]
            # check to see that all time differences fall within threshold
            return all([td < tolerance for td in time_delta])
        except Exception:
            # if the above fails, fail the test
            return False
    # if no warmup trials, return True
    elif len(warmup_trials) == 0:
        return True


def validate_autorewards_after_N_consecutive_misses(trials, autoreward_after_consecutive_misses):
    '''
    validate that an autoreward is delivered after N consecutive misses
    '''
    go_trials = trials[trials.trial_type == 'go']
    auto_reward_when_expected = []
    consecutive_misses = 0

    for idx, row in go_trials.iterrows():
        miss = 1 if row['response'] == 0 else 0
        auto_rewarded = 1 if row['auto_rewarded'] else 0
        consecutive_misses = (consecutive_misses * miss + miss) * abs(1 - auto_rewarded)
        if consecutive_misses > autoreward_after_consecutive_misses:
            # these are trials when an autoreward is expected
            auto_reward_when_expected.append(row['auto_rewarded'])

    # all trials with expected autorewards should be True
    return all(auto_reward_when_expected)


def validate_change_on_all_go_trials(trials):
    '''
    ensure that either the orientation or the image_name changes on every go
    trial
    '''
    go_trials = trials[trials.trial_type == 'go']
    for idx, row in go_trials.iterrows():
        same_image = nanis_equal(row['initial_image_name'], row['change_image_name'])
        same_ori = nanis_equal(row['initial_ori'], row['change_ori'])
        if (same_image is True) and (same_ori is True):
            return False
    return True


def validate_no_change_on_all_catch_trials(trials):
    '''ensure that neither the orientation nor the image_name changes on every catch trial'''
    go_trials = trials[trials.trial_type == 'catch']
    for idx, row in go_trials.iterrows():
        same_image = nanis_equal(row['initial_image_name'], row['change_image_name'])
        same_ori = nanis_equal(row['initial_ori'], row['change_ori'])
        if (same_image is False) and (same_ori is False):
            return False
    return True


def validate_intial_and_final_in_non_aborted(trials):
    '''all non-aborted trials should have either an intial and final image OR an initial and final orientation'''
    non_aborted_trials = trials[trials.trial_type != 'aborted']
    all_initial = (
        all(~pd.isnull(non_aborted_trials['initial_image_name']))
        or all(~pd.isnull(non_aborted_trials['initial_ori']))
    )
    all_final = (
        all(~pd.isnull(non_aborted_trials['change_image_name']))
        or all(~pd.isnull(non_aborted_trials['change_ori']))
    )
    return all_initial and all_final


def validate_min_change_time(trials, pre_change_time):
    '''change time in trial should never be less than pre_change_time'''
    change_times_trial_referenced = (trials['change_time'] - trials['starttime']).values
    # can only run validation if there are some non-null values
    if all(pd.isnull(change_times_trial_referenced))==False:
        return np.nanmin(change_times_trial_referenced) > pre_change_time
    # cannot run valiation function if all null, just return True
    elif all(pd.isnull(change_times_trial_referenced))==True:
        return True


def validate_max_change_time(trials, pre_change_time, stimulus_window, tolerance=0.05):
    '''Changes should never occur at a time greater than `pre_change_time` + `stimulus_window`'''
    change_times_trial_referenced = (trials['change_time'] - trials['starttime']).values
    # can only run validation if there are some non-null values
    if all(pd.isnull(change_times_trial_referenced))==False:
        return np.nanmax(change_times_trial_referenced) < (pre_change_time + stimulus_window + tolerance)
    # cannot run valiation function if all null, just return True
    elif all(pd.isnull(change_times_trial_referenced))==True:
        return True


def validate_reward_follows_first_lick_in_window(trials, tolerance=0.001):
    '''reward should happen immediately (within tolerance) of first lick in window on non-autorewarded go trials'''
    # get all go_trials without auto_rewards
    contingent_go_trials = trials[
        (trials.trial_type == 'go')
        & (trials.auto_rewarded == False)  # noqa: E712
    ]

    # get first lick in response window, relative to change time
    first_lick_in_window = contingent_go_trials.apply(
        get_first_lick_in_response_window,
        axis=1,
    )
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


def validate_lick_before_scheduled_on_aborted_trials(trials,expected_flash_duration=0.25,expected_blank_duration=0.5):
    '''
    if licks occur before a scheduled change time/flash, the trial ends
    Therefore, every aborted trial should have a lick before the scheduled change time
    '''
    aborted_trials = trials[trials.trial_type == 'aborted']
    # can only run this test if there are aborted trials
    if len(aborted_trials) > 0:
        first_lick = aborted_trials.apply(
            get_first_lick_relative_to_scheduled_change,
            axis=1,
        )
        # don't use nanmax. If there's a nan, we expect this to fail
        return np.max(first_lick.values) < (0+expected_flash_duration+expected_blank_duration)
    # if no aborted trials, return True
    else:
        return True


def validate_lick_after_scheduled_on_go_catch_trials(trials):
    '''
    if licks occur before a scheduled change time/flash, the trial ends
    Therefore, no non-aborted trials should have a lick before the scheduled change time
    '''
    nonaborted_trials = trials[trials.trial_type != 'aborted']
    # We can only check this if there is at least 1 nonaborted trial.
    if len(nonaborted_trials) > 0:
        first_lick = nonaborted_trials.apply(get_first_lick_relative_to_scheduled_change, axis=1)
        # use nanmin
        return np.nanmin(first_lick.values) > 0
    # if there are no nonaborted trials, just return True
    elif len(nonaborted_trials) == 0:
        return True


def validate_initial_matches_final(trials):
    '''
    On go and catch trials, the initial image (or ori) on a given trial should match the final image (or ori) on the previous trial
    '''
    trials_to_test = trials[trials['trial_type'].isin(['go', 'catch'])]

    # Can only run this validation if there are at least two go or catch trials
    if len(trials_to_test) > 2:
        last_trial_final_im = trials_to_test.iloc[0]['initial_image_name']
        last_trial_final_ori = trials_to_test.iloc[0]['initial_ori']
        image_match = []
        ori_match = []
        for idx, row in trials_to_test.iterrows():
            image_match.append(nanis_equal(row['initial_image_name'], last_trial_final_im))
            ori_match.append(nanis_equal(row['initial_ori'], last_trial_final_ori))

            last_trial_final_im = row['change_image_name']
            last_trial_final_ori = row['change_ori']

        return all(image_match) and all(ori_match)

    # If too few trials to validate, return True
    else:
        return True


def validate_first_lick_after_change_on_nonaborted(trials):
    '''
    on GO and CATCH trials, licks should never be observed between the trial start and the first frame of an image change
    '''
    non_aborted_trials = trials[trials.trial_type != 'aborted']
    # we can only validate this if there is at least one nonaborted trial
    if len(non_aborted_trials) > 0:
        first_lick_relative_to_change = non_aborted_trials.apply(get_first_lick_relative_to_change, axis=1)
        return np.nanmin(first_lick_relative_to_change.values) > 0
    # if no nonaborted trials, just return True
    elif len(non_aborted_trials) == 0:
        return True


def validate_trial_ends_without_licks(trials, minimum_no_lick_time):
    '''
    There should never be a lick within 'minimum_no_lick_time' of trial end
    Task logic should extend trial if mouse is licking near trial end
    '''
    non_aborted_trials = trials[trials.trial_type.isin(['go', 'catch'])]
    for idx, row in non_aborted_trials.iterrows():
        # fail if the last lick is within 'minimum_no_lick_time' of the endtime
        if len(row['lick_times']) > 0 and row['endtime'] - row['lick_times'][-1] < minimum_no_lick_time:
            return False
    return True


def validate_session_within_expected_duration(trials, expected_duration_seconds, tolerance=30):
    '''
    ensure that last trial end time does not exceed expected duration by more than 30 seconds
    '''
    return trials['endtime'].values[-1] < expected_duration_seconds + tolerance


def validate_session_ends_at_max_cumulative_volume(trials, volume_limit, tolerance=0.01):
    '''
    ensure that earned volume does not exceed volume limit
    build in a tolerance in case reward limit is not even multiple of reward volumes
    '''
    return trials['cumulative_volume'].max() <= volume_limit + tolerance


def validate_duration_and_volume_limit(trials, expected_duration, volume_limit, time_tolerance=30):
    '''
    The length of a session should not be less than `duration` by more than 30 seconds UNLESS `volume_limit` has been met
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


def validate_catch_frequency(trials, expected_catch_frequency, rejection_probability=0.05):
    '''
    non-aborted catch trials should comprise `catc_freq` of all non-aborted trials
    uses scipy's binomial test to ensure that the null hypothesis (catch trials come from a binomial distribution drawn with catch_freq) is true with 95% probability
    '''
    from scipy import stats
    go_trials = trials[trials['trial_type'] == 'go']
    catch_trials = trials[trials['trial_type'] == 'catch']

    probability_of_null_hypothesis = stats.binom_test(
        len(catch_trials),
        n=len(catch_trials) + len(go_trials),
        p=expected_catch_frequency,
    )

    if probability_of_null_hypothesis < (1 - rejection_probability):
        return True
    else:
        return False


def validate_stage_present(trials):
    '''
    The parameter `stage` should be recorded in the output file.
    '''
    return all(~pd.isnull(trials['stage']))


def validate_task_id_present(trials):
    '''
    The parameter `task_id` should be recorded in the output file.
    '''
    try:
        trials['task']
        return all(~pd.isnull(trials['task']))
    except KeyError:
        return False


def validate_number_aborted_trial_repeats(trials, failure_repeats, tolerance=0.01):
    '''
    on aborted trials (e.g. early licks), the trial's stimulus parameters should be repeated `failure_repeats` times
    '''

    # assign columns to the dataframe to make test possible
    trials = identify_consecutive_aborted_blocks(trials, failure_repeats)

    # get all aborted trials
    aborted_trials = trials[trials['trial_type'] == 'aborted']

    # identify all blocks which should have matching scheduled change times (blocks with lengths in multiples of 'failure_repeats')
    block_ids = aborted_trials.consecutive_aborted_should_match.unique()

    # check each block to make sure all scheduled change times are equal, within a tolerance
    block_has_matching_scheduled_change_time = []
    for block_id in block_ids:
        scheduled_times_this_block = aborted_trials[aborted_trials.consecutive_aborted_should_match == block_id]['scheduled_change_time'].values
        this_block_matches = all_close(scheduled_times_this_block, tolerance=tolerance)
        block_has_matching_scheduled_change_time.append(this_block_matches)

    # return True if all blocks matched
    return all(block_has_matching_scheduled_change_time)


def validate_params_change_after_aborted_trial_repeats(trials, failure_repeats):
    '''
    failure_repeats: for the `failure_repeats`+2 aborted trial, new parameters should be sampled
    '''

    # assign columns to the dataframe to make test possible
    trials = identify_consecutive_aborted_blocks(trials, failure_repeats)

    # get all aborted trials
    aborted_trials = trials[trials['trial_type'] == 'aborted']

    # identify all blocks which should have matching scheduled change times (blocks with lengths in multiples of 'failure_repeats')
    block_ids = aborted_trials.consecutive_aborted_should_match.unique()

    # compare scheduled change times across blocks. They should be different
    block_has_different_scheduled_change_time_than_last = []
    for ii, block_id in enumerate(block_ids):
        scheduled_times_this_block = aborted_trials[aborted_trials.consecutive_aborted_should_match == block_id]['scheduled_change_time'].values
        first_change_time_in_block = scheduled_times_this_block[0]
        if ii > 0:  # can't compare first block to anything
            # ensure change time is different than last change time on last block
            block_has_different_scheduled_change_time_than_last.append(
                first_change_time_in_block != last_change_time_in_block  # noqa F821
            )
        last_change_time_in_block = scheduled_times_this_block[-1]  # noqa F841

    # return True if every block has different params than the last
    return all(block_has_different_scheduled_change_time_than_last)


def validate_ignore_false_alarms(trials, abort_on_early_response):
    '''
    If 'abort_on_early_response' is False, no aborted trials or timeout periods should occur.
    This is False on the 'day 0' session, where we don't want to penalize exploratory licking
    '''
    if abort_on_early_response == False:
        # if ignore_false_alarms is True, there should be 0 aborted trials
        # Of course, this depends on the aborted trials having been properly identified
        if len(trials[trials.trial_type == 'aborted']) == 0:
            return True
        else:
            return False
    # test is only valid if the 'abort_on_early_response' parameter is False. If not the case, just pass the test
    else:
        return True


def validate_trial_times_never_overlap(trials):
    '''
    a trial cannot start until the prior trial is complete
    '''
    previous_end = 0
    trial_start_greater_than_last_end = []
    for idx, row in trials.iterrows():
        trial_start_greater_than_last_end.append(row['starttime'] >= previous_end)
        previous_end = row['endtime']
    return all(trial_start_greater_than_last_end)


def validate_stimulus_distribution_key(trials, expected_distribution_name):
    '''
    stimulus_distribution:The distribution of change times should match the 'stimulus_distribution' (default='exponential')
    '''
    return all(trials['stimulus_distribution'] == expected_distribution_name)


def validate_change_time_mean(trials, expected_mean, tolerance=0.5):
    '''
    stimulus_distribution: the mean of the stimulus distribution should match the 'distribution_mean' parameter
    '''
    change_time_trial_referenced = trials['change_time'] - trials['starttime'] - trials['prechange_minimum']
    # if all of the change times are Nan, cannot check mean. Return True
    if all(pd.isnull(change_time_trial_referenced)) == True:
        return True
    # otherwise, ensure that the non-null change times are within tolerance of the expected mean
    else:
        return abs(change_time_trial_referenced[~np.isnan(change_time_trial_referenced)].mean() - expected_mean) <= tolerance


def validate_monotonically_decreasing_number_of_change_times(trials, expected_distribution):
    '''
    If stimulus_distribution is 'exponential', then the number of trials with changes on each flash should be monotonically decreasing from the first flash after the `min_change_time`
    '''
    change_times = (trials['change_time'] - trials['starttime']).values
    if expected_distribution == 'exponential' and all(pd.isnull(change_times)) != True:

        change_times = change_times[~np.isnan(change_times)]

        nvals, edges = np.histogram(change_times, bins=np.arange(0, 10, 0.5))

        nonzero_bins = nvals[nvals > 0]
        return all(
            [
                first_val >= second_val
                for first_val, second_val
                in zip(nonzero_bins[:-1], nonzero_bins[1:])
            ]
        )
    else:
        return True


def validate_no_abort_on_lick_before_response_window(trials):
    '''
    If licks occur between the change time and `response_window[0]`, the trial should continue.
    Method ensures that all cases matching this condition are labeled as 'go' or 'catch', not 'aborted'
    '''
    def identify_first_lick_before_response_window(row):
        '''
        a method for identifying trials with licks between the change time and the start of the response window
        '''
        # get first lick from lick list if it isn't empty
        first_lick = row['lick_times'][0] if len(row['lick_times']) > 0 else np.nan
        # get offset between first lick and change time
        first_lick_relative_to_change = first_lick - row['change_time']
        # return True if first lick is between the change time and the start of the response window, False otherwise
        if np.logical_and(first_lick_relative_to_change >= 0, first_lick_relative_to_change < row['response_window'][0]):
            return True
        else:
            return False

    nonaborted_trials = trials[trials['trial_type'] != 'aborted']

    # Can only check this if there is at least one non-aborted trial
    if len(nonaborted_trials) > 0:
        # identify all trials with licks between the change time and the start of the response window
        trials['response_before_response_window'] = trials.apply(identify_first_lick_before_response_window, axis=1)

        # ensure that all trials with first lick between the change time and the start of the response window are labeled as 'go' or 'catch' (not 'aborted')
        return all(trials[trials['response_before_response_window'] == True].trial_type.isin(['go', 'catch']))
    # if no non-aborted trials, just return True
    else:
        return True


def validate_licks_on_go_trials_earn_reward(trials):
    '''
    all go trials with licks in response window should have 1 reward
    '''
    number_of_licks_in_window = trials.apply(identify_licks_in_response_window, axis=1)
    number_of_rewards_on_go_lick_trials = trials[
        (number_of_licks_in_window > 0) &
        (trials['trial_type'] == 'go')
    ]['number_of_rewards']
    return all(number_of_rewards_on_go_lick_trials) == 1


def validate_licks_on_catch_trials_do_not_earn_reward(trials):
    '''
    all catch trials with licks in window should have 0 rewards
    '''
    number_of_licks_in_window = trials.apply(identify_licks_in_response_window, axis=1)
    number_of_rewards_on_catch_lick_trials = trials[
        (number_of_licks_in_window > 0) &
        (trials['trial_type'] == 'catch')
    ]['number_of_rewards']
    # can only evaluate if there have been some false alarm trials
    if len(number_of_rewards_on_catch_lick_trials) > 0:
        return all(number_of_rewards_on_catch_lick_trials) == 0
    # if no false alarm trials, return True
    else:
        return True


def validate_even_sampling(trials, even_sampling_enabled):
    '''
    if even_sampling mode is enabled, no stimulus change combinations (including catch conditions) are sampled more than one more time than any other.
    Note that even sampling only applies to images, so this does not need to be written to deal with grating stimuli
    '''
    if even_sampling_enabled == True:
        nonaborted_trials = trials[trials['trial_type'] != 'aborted']

        # build a table with the counts of all combinations
        transition_matrix = pd.pivot_table(
            nonaborted_trials,
            values='response',
            index=['initial_image_name'],
            columns=['change_image_name'],
            aggfunc=np.size
        )
        # Return true if the range of max to min is less than or equal to 1
        return abs(transition_matrix.values.max() - transition_matrix.values.min()) <= 1
    else:
        return True


def validate_flash_blank_durations(visual_stimuli, periodic_flash, tolerance=0.02):
    '''
    The duty cycle of the stimulus onset/offset is maintained across trials
    (e.g., if conditions for ending a trial are met, the stimulus presentation is not truncated)

    Takes core_data['visual_stimuli'] as input
    '''
    if periodic_flash is not None:
        expected_flash_duration = periodic_flash[0]
        expected_blank_duration = periodic_flash[1]
        # get all blank durations
        blank_durations = visual_stimuli['time'].diff() - visual_stimuli['duration']
        blank_durations = blank_durations[~np.isnan(blank_durations)].values

        # get all flash durations
        flash_durations = visual_stimuli.duration.values

        # make sure all flashes and blanks are within tolerance
        blank_durations_consistent = all(
            np.logical_and(
                blank_durations > expected_blank_duration - tolerance,
                blank_durations < expected_blank_duration + tolerance
            )
        )
        flash_durations_consistent = all(
            np.logical_and(
                flash_durations > expected_flash_duration - tolerance,
                flash_durations < expected_flash_duration + tolerance
            )
        )

        return blank_durations_consistent and flash_durations_consistent
    # cannot evaluate if periodic_flash is None. just return True
    else:
        return True


def validate_two_stimuli_per_go_trial(trials, visual_stimuli):
    '''
    all 'go' trials should have two stimulus groups
    '''
    stimuli_per_trial = count_stimuli_per_trial(trials[trials['trial_type'] == 'go'], visual_stimuli)
    return all(stimuli_per_trial == 2)


def validate_one_stimulus_per_catch_trial(trials, visual_stimuli):
    '''
    all 'catch' trials should have one stimulus group
    '''
    stimuli_per_trial = count_stimuli_per_trial(trials[trials['trial_type'] == 'catch'], visual_stimuli)
    return all(stimuli_per_trial == 1)


def validate_one_stimulus_per_aborted_trial(trials, visual_stimuli):
    '''
    all 'aborted' trials should have one stimulus group
    '''
    stimuli_per_trial = count_stimuli_per_trial(trials[trials['trial_type'] == 'aborted'], visual_stimuli)
    return all(stimuli_per_trial == 1)


def validate_change_frame_at_flash_onset(trials, visual_stimuli):
    '''
    if `periodic_flash` is not null, changes should always coincide with a stimulus onset
    '''
    # get all non-null change frames
    change_frames = trials[~pd.isnull(trials['change_frame'])]['change_frame'].values

    # confirm that all change frames coincide with flash onsets in the visual stimulus log
    return all(np.in1d(change_frames, visual_stimuli['frame']))


def validate_initial_blank(trials, visual_stimuli, initial_blank, tolerance=0.005):
    '''
    iterates over trials
    Verifies that there is a blank screen of duration `initial_blank` at the start of every trial.
    If initial blank is 0, first frame of flash should be coincident with trial start
    '''
    # preallocate array
    initial_blank_in_tolerance = np.empty(len(trials))

    # iterate over dataframe
    for idx, trial in trials.reset_index().iterrows():
        # get visual greater than initial frame
        first_stim_index = visual_stimuli[visual_stimuli['frame'] >= trial['startframe']].index[0]
        # get offset between trial start and first stimulus of frame
        first_stim_time_offset = visual_stimuli.loc[first_stim_index]['time'] - trial['starttime']
        # check to see if offset is within tolerance of expected blank
        initial_blank_in_tolerance[idx] = np.isclose(first_stim_time_offset, initial_blank, atol=tolerance)
    # ensure all initial blanks were within tolerance
    return all(initial_blank_in_tolerance)


def validate_new_params_on_nonaborted_trials(trials):
    '''
    new parameters should be sampled after any non-aborted trial
    '''
    nonaborted_trials = trials[trials['trial_type'] != 'aborted']

    return all(nonaborted_trials['scheduled_change_time'] != nonaborted_trials['scheduled_change_time'].shift())
