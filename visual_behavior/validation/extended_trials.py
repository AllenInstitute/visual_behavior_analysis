import numpy as np
import pandas as pd
import six
from scipy import stats

from visual_behavior.translator.core import annotate

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


def get_first_lick_in_response_window(row, tolerance=0.01, tolerance_direction='inside', response_window=None):
    '''
    returns first lick that falls in response window, nan if no such lick
    exists
    '''
    assert response_window, "must pass a response window"
    # only look for licks in window if there were licks and a change time
    if len(row['lick_times']) > 0 and not pd.isnull(row['change_time']):
        left_edge = np.subtract(response_window[0], tolerance) if tolerance_direction == 'outside' else np.add(response_window[0], tolerance)
        right_edge = np.add(response_window[1], tolerance) if tolerance_direction == 'outside' else np.subtract(response_window[1], tolerance)

        licks = np.array(row['lick_times']) - row['change_time']
        licks_in_window = licks[
            np.logical_and(
                licks >= left_edge,
                licks <= right_edge,
            )
        ]
    else:
        licks_in_window = []

    if len(licks_in_window) > 0:
        return licks_in_window[0]
    else:
        return np.nan


def identify_licks_in_response_window(row, tolerance=0.01, tolerance_direction='inside'):
    '''a method for counting licks in the response window'''
    if len(row['lick_times']) > 0 and pd.isnull(row['change_time']) == False:
        licks_relative_to_change = np.array(row['lick_times']) - row['change_time']
        left_edge = np.subtract(row['response_window'][0], tolerance) if tolerance_direction == 'outside' else np.add(row['response_window'][0], tolerance)
        right_edge = np.add(row['response_window'][1], tolerance) if tolerance_direction == 'outside' else np.subtract(row['response_window'][1], tolerance)

        licks_in_window = licks_relative_to_change[
            np.logical_and(
                licks_relative_to_change >= left_edge,
                licks_relative_to_change <= right_edge
            )
        ]
        return len(licks_in_window)
    else:
        return 0


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


def get_first_lick_after_change(row):
    '''
    returns first lick after change time, nan if no such lick
    exists
    '''
    licks = np.array(row['lick_times']) - row['change_time']
    # keep only positive licks
    licks = licks[licks > 0]
    if len(licks) > 0:
        return licks[0]
    else:
        return np.nan


def get_accumlated_rewards_licks(df):
    """Calculates how many rewards acculated before the first lick after change time.
    
    
    TODO: produces shorter trial arrays because drop na, consider how to handle"""

    # explode reward times
    reward_times = df['reward_times'].explode()

    # explode lick times
    lick_times = df['lick_times'].explode()


    # iterate diff_reward_times, check for any licks between the two rewards, if so True, else false
    # if True, add 1 to the counter
    accumulated_rewards = 1
    consumed = False
    consumed_list = []
    accumulated_rewards_list = []

    # drop nan
    reward_times = reward_times.dropna().reset_index(drop=True)
    for i in range(len(reward_times)-1):
        if np.any(lick_times.between(reward_times[i], reward_times[i+1])):
            consumed = True
            accumulated_rewards = 1
        else:
            accumulated_rewards += 1
            consumed = False

        consumed_list.append(consumed)
        accumulated_rewards_list.append(accumulated_rewards)


    return consumed_list, accumulated_rewards_list





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


def fix_visual_stimuli(core_data):
    '''
    #patch end_frame in core_data['visual_stimuli']
    #lets us move ahead while awaiting a fix to https://github.com/AllenInstitute/visual_behavior_analysis/issues/156
    '''
    trials = core_data['trials']
    core_data['visual_stimuli'].at[core_data['visual_stimuli'].index[-1], 'end_frame'] = trials.iloc[-1].endframe
    return core_data


def count_stimuli_per_trial(trials, visual_stimuli):
    '''
    iterates over trials
    counts stimulus groups in range of trial frames for each trial
    returns vector with all numbers of stimuli per trial
    '''
    # preallocate array
    stimuli_per_trial = np.empty(len(trials))

    # check to see if images or gratings are being used
    if all(pd.isnull(visual_stimuli.image_category)) == False:
        col_to_check = 'image_category'
    elif all(pd.isnull(visual_stimuli.orientation)) == False:
        col_to_check = 'orientation'

    # iterate over dataframe
    for idx, trial in trials.reset_index().iterrows():

        try:
            stimuli = np.unique(visual_stimuli[
                (visual_stimuli['frame'].isin(range(trial.startframe, trial.endframe)))
                | (visual_stimuli['end_frame'].isin(range(trial.startframe, trial.endframe)))
            ][col_to_check])
            # print('checking frame {} to {}'.format(trial.startframe, trial.endframe))
            # print('unique stimuli = {}'.format(stimuli))
            # add to array
            stimuli_per_trial[idx] = len(stimuli)
        except (IndexError, UnboundLocalError):
            stimuli_per_trial[idx] = 0
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
        return len(warmup_trials[warmup_trials.trial_type.isin(['go', 'autorewarded'])]) == expected_number_of_warmup_trials
    # if -1, all will be warmup trials.
    elif expected_number_of_warmup_trials == -1:
        return len(trials[trials.auto_rewarded == True]) == len(trials)


def validate_reward_delivery_on_warmup_trials(trials, autoreward_delay, tolerance=0.017):
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
            return all([abs(td - autoreward_delay) < tolerance for td in time_delta])
        except Exception:
            # if the above fails, fail the test
            return False
    # if no warmup trials, return True
    elif len(warmup_trials) == 0:
        return True


def validate_autorewards_after_N_consecutive_misses(extended_trials, autoreward_after_consecutive_misses, warmup_trials):
    '''
    validate that an autoreward is delivered after N consecutive misses

    returns False if any expected autorewards are missing OR any autorewards come earlier than expected

    Ignores warmup trials
    '''

    # if warmup_trials == -1, skip this validation altogether
    if warmup_trials == -1:
        return True
    else:
        # get all go trials, ignore the first `warmup_trials` trials
        go_trials = extended_trials[extended_trials.trial_type.isin(['go', 'autorewarded'])].iloc[warmup_trials:]
        auto_reward_when_expected = []
        consecutive_misses = 0
        reward_expected_on_next = False

        for idx, row in go_trials.iterrows():
            # use the presence/absence of rewards to determine whether foraging2 labeled this as a hit or miss
            miss = 1 if len(row['reward_times']) == 0 else 0
            auto_rewarded = 1 if row['auto_rewarded'] else 0
            consecutive_misses = (consecutive_misses * miss + miss) * abs(1 - auto_rewarded)

            if reward_expected_on_next == True:
                # these are trials when an autoreward is expected
                auto_reward_when_expected.append(row['auto_rewarded'])

            # check for autorewards that come at unexpected times
            if reward_expected_on_next == False and auto_rewarded == 1:
                return False

            # keep track of whether an autoreward is expected on the next go trial
            if consecutive_misses == autoreward_after_consecutive_misses:
                reward_expected_on_next = True
            else:
                reward_expected_on_next = False

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


def validate_min_change_time(trials, pre_change_time, tolerance=0.015):
    '''change time in trial should never be less than pre_change_time'''
    change_times_trial_referenced = (trials['change_time'] - trials['starttime']).values
    # can only run validation if there are some non-null values
    if all(pd.isnull(change_times_trial_referenced)) == False:
        return np.nanmin(change_times_trial_referenced) > pre_change_time - tolerance
    # cannot run valiation function if all null, just return True
    elif all(pd.isnull(change_times_trial_referenced)) == True:
        return True


def validate_max_change_time(trials, pre_change_time, stimulus_window, distribution_type, tolerance=0.05):
    if distribution_type.lower() == 'geometric':
        return True
    else:
        '''Changes should never occur at a time greater than `pre_change_time` + `stimulus_window`'''
        change_times_trial_referenced = (trials['change_time'] - trials['starttime']).values
        # can only run validation if there are some non-null values
        if all(pd.isnull(change_times_trial_referenced)) == False:
            return np.nanmax(change_times_trial_referenced) < (pre_change_time + stimulus_window + tolerance)
        # cannot run valiation function if all null, just return True
        elif all(pd.isnull(change_times_trial_referenced)) == True:
            return True


def validate_reward_when_lick_in_window(core_data, tolerance=0.01):
    '''for every trial with a lick in the response window, there should be a reward'''
    # get all go_trials without auto_rewards
    trials = core_data['trials']
    trial_type = annotate.categorize_trials(trials)
    contingent_go_trials = trials[
        (trial_type == 'go') &
        (trials['auto_rewarded'] == False)
    ]

    # get first lick in response window, relative to change time
    first_lick_in_window = contingent_go_trials.apply(
        get_first_lick_in_response_window,
        axis=1,
        tolerance=0.01,
        tolerance_direction='inside',
        response_window=core_data['metadata']['response_window'],
    )
    first_lick_in_window.name = 'first_lick'

    # get reward time, relative to change time
    def get_reward_relative_to_change(row):
        if len(row['reward_times']) > 0:
            return row['reward_times'][0] - row['change_time']
        else:
            return np.nan
    reward_time = contingent_go_trials.apply(get_reward_relative_to_change, axis=1)
    reward_time.name = 'reward_time'

    # check that, whenever a first lick exists, a reward was given with tolerance
    for idx, row in pd.concat([first_lick_in_window, reward_time], axis=1).iterrows():
        # if there was a lick in the window and there was no reward, fail
        if not pd.isnull(row['first_lick']) and pd.isnull(row['reward_time']):
            return False

    return True


def validate_licks_near_every_reward(trials, tolerance=0.005):
    '''
    validates that there is a lick near every reward, excluding auto_rewards
    '''
    earned_reward_trials = trials[
        (trials.number_of_rewards > 0) &
        (trials.auto_rewarded == False)
    ]

    for idx, row in earned_reward_trials.iterrows():
        # if there aren't any licks, something is wrong. Return False
        if len(row['lick_times']) == 0:
            return False
        # assuming there are licks, there should be one within tolerance of the reward
        elif not np.isclose(min(abs(row['reward_times'][0] - np.array(row['lick_times']))), 0, atol=tolerance):
            return False

    # if no failed trials found, return True
    return True


def validate_never_more_than_one_reward(trials):
    '''Only a maximum of one reward should be delivered per trial'''
    return np.max(trials['number_of_rewards']) <= 1


def validate_lick_after_scheduled_on_go_catch_trials(core_data, abort_on_early_response, distribution_type, tolerance=0.01):
    '''
    if licks occur before a scheduled change time/flash, the trial ends
    Therefore, no non-aborted trials should have a lick before the scheduled change time,
    except when abort_on_early_response is False
    '''

    trials = core_data['trials']
    trial_type = annotate.categorize_trials(trials)
    nonaborted_trials = trials[trial_type != 'aborted']
    # We can only check this if there is at least 1 nonaborted trial.
    if distribution_type.lower() == 'geometric':
        return True
    elif abort_on_early_response == True and len(nonaborted_trials) > 0:
        first_lick = nonaborted_trials.apply(get_first_lick_relative_to_scheduled_change, axis=1)
        # use nanmin, apply tolerance to account for same frame licks
        if np.nanmin(first_lick.values) < 0 - tolerance:
            return False
        else:
            return True
    # if there are no nonaborted trials, just return True
    elif abort_on_early_response == False or len(nonaborted_trials) == 0:
        return True


def validate_initial_matches_final(trials):
    '''
    On go and catch trials, the initial image (or ori) on a given trial should match the final image (or ori) on the previous trial
    '''
    trials_to_test = trials[trials['trial_type'].isin(['go', 'catch', 'autorewarded'])]

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


def validate_first_lick_after_change_on_nonaborted(trials, abort_on_early_response):
    '''
    on GO and CATCH trials, licks should never be observed between the trial start and the first frame of an image change,
    except when abort_on_early_response == False
    '''
    non_aborted_trials = trials[trials.trial_type != 'aborted']
    # we can only validate this if there is at least one nonaborted trial
    if abort_on_early_response == True and len(non_aborted_trials) > 0:
        first_lick_relative_to_change = non_aborted_trials.apply(get_first_lick_relative_to_change, axis=1)
        if np.nanmin(first_lick_relative_to_change.values) < 0:
            return False
        else:
            return True
    # if no nonaborted trials, just return True
    elif abort_on_early_response == False or len(non_aborted_trials) == 0:
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
    return (trials['starttime'].values[-1] - trials['starttime'].values[0]) < (expected_duration_seconds + tolerance)


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


def validate_catch_frequency(trials, expected_catch_frequency, rejection_probability=0.001):
    '''
    non-aborted catch trials should comprise `catc_freq` of all non-aborted trials
    uses scipy's binomial test to ensure that the null hypothesis (catch trials come from a binomial distribution drawn with catch_freq) is true with 95% probability
    '''
    go_trials = trials[trials['trial_type'] == 'go']
    catch_trials = trials[trials['trial_type'] == 'catch']

    probability_of_null_hypothesis = stats.binom_test(
        len(catch_trials),
        n=len(catch_trials) + len(go_trials),
        p=expected_catch_frequency,
    )

    if probability_of_null_hypothesis > rejection_probability:
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


def validate_params_change_after_aborted_trial_repeats(trials, failure_repeats, distribution_type):
    '''
    failure_repeats: for the `failure_repeats`+2 aborted trial, new parameters should be sampled
    '''

    if distribution_type.lower() == 'geometric':
        return True
    else:
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


def validate_change_time_mean(trials, expected_mean, distribution_type, tolerance=1):
    '''
    stimulus_distribution: the mean of the stimulus distribution should match the 'distribution_mean' parameter
    '''
    change_time_trial_referenced = trials['scheduled_change_time'] - trials['prechange_minimum']
    # if all of the change times are Nan, cannot check mean. Return True
    if all(pd.isnull(change_time_trial_referenced)) == True:
        return True
    # or if the distribution is geometric, this doesn't make sense, so return True
    elif distribution_type.lower() == 'geometric':
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
    Method ensures that all cases matching this condition are labeled as 'go', 'catch', or 'autorewarded', not 'aborted'
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
        return all(trials[trials['response_before_response_window'] == True].trial_type.isin(['go', 'catch', 'autorewarded']))
    # if no non-aborted trials, just return True
    else:
        return True


def validate_licks_on_go_trials_earn_reward(trials):
    '''
    all go trials with licks in response window should have 1 reward
    '''
    # note: make tolerance direction 'inside' to only look at licks that were unambiguously inside the window
    number_of_licks_in_window = trials.apply(identify_licks_in_response_window, axis=1, tolerance=1 / 60., tolerance_direction='inside')
    number_of_rewards_on_go_lick_trials = trials[
        (number_of_licks_in_window > 0) &
        (trials['trial_type'] == 'go')
    ]['number_of_rewards']
    return all(number_of_rewards_on_go_lick_trials) == 1


def validate_licks_on_catch_trials_do_not_earn_reward(trials):
    '''
    all catch trials with licks in window should have 0 rewards
    '''
    # note: make tolerance direction 'outside' to identify licks that may have been just outside the window, but counted at runtime
    number_of_licks_in_window = trials.apply(identify_licks_in_response_window, axis=1, tolerance=0.01, tolerance_direction='outside')
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
    go_trials = trials[trials['trial_type'].isin(['go', 'autorewarded'])]
    if even_sampling_enabled == True and len(go_trials) > 0:
        # build a table with the counts of all combinations
        transition_matrix = pd.pivot_table(
            go_trials,
            values='response',
            index=['initial_image_name'],
            columns=['change_image_name'],
            aggfunc=np.size,
            fill_value=0,
        ).astype(float)

        # if catch trials are drawn during autorewards, foraging2 skips those, violating the even_sampling rule
        # fill the diagonal with NaN to prevent skipped catch trials during autorewards from failing this validation
        np.fill_diagonal(transition_matrix.values, np.nan)

        sample_range = abs(np.nanmax(transition_matrix.values) - np.nanmin(transition_matrix.values))
        # Return true if the range of max to min is less than or equal to 1 AND sum of values matches trial number
        return sample_range <= 1 and transition_matrix.sum().sum() == len(go_trials)
    else:
        return True


def merge_in_omitted_flashes(visual_stimuli, omitted_stimuli):
    visual_stimuli['omitted'] = False

    # get all blank durations
    visual_stimuli['previous_blank_duration'] = visual_stimuli['time'].diff() - visual_stimuli['duration']

    if omitted_stimuli is not None:
        omitted_stimuli['omitted'] = True
        if six.PY2:
            visual_stimuli = pd.concat((visual_stimuli, omitted_stimuli)).sort_values(by='frame').reset_index()
        elif six.PY3:
            visual_stimuli = pd.concat((visual_stimuli, omitted_stimuli), sort=True).sort_values(by='frame').reset_index()
        else:
            raise(RuntimeError)

    # was previous flash omitted?
    visual_stimuli['previous_omitted'] = visual_stimuli['omitted'].shift()

    return visual_stimuli


def get_flash_blank_durations(visual_stimuli, omitted_stimuli, periodic_flash):

    visual_stimuli = merge_in_omitted_flashes(visual_stimuli, omitted_stimuli)

    # get all blank durations, but ignore omitted flashes and flashes immediately following omitted flashes
    blank_durations = visual_stimuli[
        (visual_stimuli['omitted'] == False)
        & (visual_stimuli['previous_omitted'] == False)
    ]['previous_blank_duration']

    # get all flash durations, but ignore omitted flashes and flashes immediately following omitted flashes
    flash_durations = visual_stimuli[
        (visual_stimuli['omitted'] == False)
        & (visual_stimuli['previous_omitted'] == False)
    ]['duration']

    return flash_durations[~np.isnan(flash_durations)], blank_durations[~np.isnan(blank_durations)]


def validate_flash_blank_durations(visual_stimuli, omitted_stimuli, periodic_flash, mean_tolerance=1 / 60., std_tolerance=1 / 60.):
    '''
    The periodicity of the stimulus onset/offset is maintained across trials
    (e.g., if conditions for ending a trial are met, the stimulus presentation is not truncated)

    default tolerances on mean and std are 1 frame (1/60 seconds)

    Takes core_data['visual_stimuli'] and core_data['omitted_stimuli'] as input
    '''

    if periodic_flash is not None:
        # if there are omitted flashes, exclude them from this validation
        flash_durations, blank_durations = get_flash_blank_durations(visual_stimuli, omitted_stimuli, periodic_flash)

        # make sure all flashes and blanks are within tolerance
        # mean should be within tolernace of expected value
        # std should be within tolerance of 0
        # leave the very last flash and blank out of the validation to avoid failures when the session was terminated early
        flash_durations_consistent = np.logical_and(
            np.isclose(flash_durations[:-1].mean(), periodic_flash[0], atol=mean_tolerance),
            np.isclose(flash_durations[:-1].std(), 0, atol=std_tolerance)
        )
        blank_durations_consistent = np.logical_and(
            np.isclose(blank_durations[:-1].mean(), periodic_flash[1], atol=mean_tolerance),
            np.isclose(blank_durations[:-1].std(), 0, atol=2 * std_tolerance)
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
    all 'catch' trials should have no more than one stimulus group
    '''
    stimuli_per_trial = count_stimuli_per_trial(trials[trials['trial_type'] == 'catch'], visual_stimuli)
    return all(stimuli_per_trial <= 1)


def validate_one_stimulus_per_aborted_trial(trials, visual_stimuli):
    '''
    all 'aborted' trials should have no more than one stimulus group
    '''
    stimuli_per_trial = count_stimuli_per_trial(trials[trials['trial_type'] == 'aborted'], visual_stimuli)
    return all(stimuli_per_trial <= 1)


def validate_change_frame_at_flash_onset(trials, visual_stimuli, periodic_flash):
    '''
    if `periodic_flash` is not null, changes should always coincide with a stimulus onset
    '''
    if periodic_flash is None:
        return True
    else:
        # get all non-null change frames
        change_frames = trials[~pd.isnull(trials['change_frame'])]['change_frame'].values + 1

        # confirm that all change frames coincide with flash onsets in the visual stimulus log
        return all(np.in1d(change_frames, visual_stimuli['frame']))


def validate_initial_blank(trials, visual_stimuli, omitted_stimuli, initial_blank, periodic_flash=True, frame_tolerance=2):
    '''
    iterates over trials
    Verifies that there is a blank screen of duration `initial_blank` at the start of every trial.
    If initial blank is 0, first frame of flash should be coincident with trial start (within 1 frame)
    '''

    # this test doesn't make sense for static stimuli with no initial blank. just return True
    if periodic_flash is None and initial_blank == 0:
        return True
    else:

        if omitted_stimuli is not None:
            if six.PY2:
                visual_stimuli = pd.concat((visual_stimuli, omitted_stimuli)).sort_values(by='frame').reset_index()
            elif six.PY3:
                visual_stimuli = pd.concat((visual_stimuli, omitted_stimuli), sort=True).sort_values(by='frame').reset_index()
            else:
                raise(RuntimeError)

        # preallocate array
        initial_blank_in_tolerance = np.empty(len(trials))

        # iterate over dataframe
        for idx, trial in trials.reset_index().iterrows():
            # get visual greater than initial frame
            first_stim_index = visual_stimuli[visual_stimuli['frame'] >= trial['startframe']].index[0]
            # get offset between trial start and first stimulus of frame
            first_stim_frame_offset = visual_stimuli.loc[first_stim_index]['frame'] - trial['startframe']
            # convert expected blank duration to frames
            initial_blank_in_frames = int(initial_blank * 60.)
            # check to see if offset is within tolerance of expected blank
            initial_blank_in_tolerance[idx] = np.isclose(first_stim_frame_offset, initial_blank_in_frames, atol=(frame_tolerance))
        # ensure all initial blanks were within tolerance
        return all(initial_blank_in_tolerance)


def validate_new_params_on_nonaborted_trials(trials, distribution_type):
    '''
    new parameters should be sampled after any non-aborted trial
    '''
    nonaborted_trials = trials[trials['trial_type'] != 'aborted']

    if distribution_type.lower() == 'geometric':
        return True
    else:
        return all(nonaborted_trials['scheduled_change_time'] != nonaborted_trials['scheduled_change_time'].shift())
