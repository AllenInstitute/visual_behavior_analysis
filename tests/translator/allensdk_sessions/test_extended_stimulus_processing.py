import os
import pandas as pd
import numpy as np
import pytest
from visual_behavior.translator.allensdk_sessions import extended_stimulus_processing as esp
from visual_behavior.translator.allensdk_sessions import attribute_formatting as af

CIRCLECI = os.environ.get('PYTHONPATH', '').startswith('/home/circleci')

@pytest.mark.parametrize('flash_times, other_times, expected', [
    (
        np.array([1, 2, 3, 4, 5], dtype=float),
        np.array([2.5], dtype=float),
        np.array([np.nan, np.nan, 0.5, 1.5, 2.5], dtype=float),
    ),
    (
        np.array([1, 2, 3, 4, 5], dtype=float),
        np.array([2], dtype=float),
        np.array([np.nan, 0, 1, 2, 3], dtype=float),
    )
])
def test_time_from_last(flash_times, other_times, expected):
    np.testing.assert_array_equal(
        esp.time_from_last(flash_times, other_times),
        expected)

@pytest.mark.parametrize('values, timestamps, start_time, stop_time, expected', [
    (
        np.array([1, 2, 3, 4, 5]),
        np.array([1, 2, 3, 4, 5]),
        2,
        4,
        2.5
    )
])
def test_trace_average(values, timestamps, start_time, stop_time, expected):
    assert esp.trace_average(values, timestamps, start_time, stop_time) == expected

@pytest.mark.parametrize('image_index, omitted_index, expected', [
    (
        pd.Series([0, 0, 0, 1, 1, 1, 2, 3, 2, 2, 2]),
        3,
        np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).astype(bool),
    ),
    (
        pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3]),
        3,
        np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).astype(bool),
    ),
    (
        pd.Series([3, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]),
        3,
        np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).astype(bool),
    ),
])
def test_find_change(image_index, omitted_index, expected):
    assert np.all(esp.find_change(image_index, omitted_index) == expected)



@pytest.fixture
def stimulus_onsets_df():
    onset_times = np.array([1, 2, 3, 4, 5], dtype=float)
    return pd.DataFrame({'start_time':onset_times})

@pytest.fixture
def mock_licks_df():
    lick_times = np.array([1.1, 2.2, 3.9, 4.75])
    return pd.DataFrame({'timestamps':lick_times})

@pytest.fixture
def mock_rewards_df():
    reward_times = np.array([1.1, 2.2, 3.9, 4.75])
    return pd.DataFrame({'timestamps':reward_times})

@pytest.fixture
def expected_stimulus_df_with_licks_and_rewards(stimulus_onsets_df):
    time_lists = [[1.1], [2.2], [], [], []]
    stimulus_onsets_df['licks'] = time_lists
    stimulus_onsets_df['rewards'] = time_lists
    return stimulus_onsets_df

def _test_add_licks_each_flash(stimulus_presentations,
                              licks_df, 
                              expected):
    stimulus_presentations = esp.add_licks_each_flash(stimulus_presentations, licks_df)

    for ind_row in range(len(stimulus_presentations)):
        left = stimulus_presentations.iloc[ind_row]['licks']
        right = expected.iloc[ind_row]['licks']
        assert len(left) == len(right)
        if len(left)>0:
            np.testing.assert_array_equal(left, right)

def test_add_licks_each_flash_mock(stimulus_onsets_df,
                                   mock_licks_df, 
                                   expected_stimulus_df_with_licks_and_rewards):
    _test_add_licks_each_flash(stimulus_onsets_df, mock_licks_df,
                               expected_stimulus_df_with_licks_and_rewards)

@pytest.mark.skipif(CIRCLECI, reason='Cannot test against real files on CircleCI')
def test_add_licks_each_flash_sfn(sfn_sdk_stimulus_presentations,
                                  sfn_sdk_licks,
                                  sfn_sdk_extended_stimulus_presentations):
    _test_add_licks_each_flash(sfn_sdk_stimulus_presentations,
                               sfn_sdk_licks,
                               sfn_sdk_extended_stimulus_presentations)

def _test_add_rewards_each_flash(stimulus_onsets_df,
                              rewards_df, 
                              expected_stimulus_df_with_rewards_and_rewards):
    stimulus_onsets_df = esp.add_rewards_each_flash(stimulus_onsets_df, rewards_df)

    for ind_row in range(len(stimulus_onsets_df)):
        left = stimulus_onsets_df.iloc[ind_row]['rewards']
        right = expected_stimulus_df_with_rewards_and_rewards.iloc[ind_row]['rewards']
        assert len(left) == len(right)
        if len(left)>0:
            np.testing.assert_array_equal(left, right)



@pytest.fixture
def stimulus_names_df():
    '''
    A small mock stim df with just the names (and block index?)
    '''
    pass


def test_repeat_within_block(


# NOTE: Find_changes will mark the first 5 auto-rewarded trials as changes
@pytest.mark.skipif(CIRCLECI, reason='Cannot test against real files on CircleCI')
def test_find_change_sfn_fixtures(sfn_sdk_stimulus_presentations, sfn_sdk_trials):
    trials_changes = sfn_sdk_trials.query('go or auto_rewarded')['change_time']
    image_index = sfn_sdk_stimulus_presentations['image_index']
    assert np.sum(esp.find_change(image_index, 8)) == len(trials_changes)

def _test_extended_stimulus_presentations(
        stimulus_presentations,
        licks,
        rewards,
        running_speed,
        extended_stimulus_presentations
):
    extended_stim = esp.get_extended_stimulus_presentations(
        stimulus_presentations,
        licks, 
        rewards, 
        running_speed)

    # Make sure all columns are accounted for
    assert len(set(extended_stim.columns.to_list()) ^ \
               set(extended_stimulus_presentations.columns.to_list())) == 0

    # TODO:Time from last change is (was always?) broken, so the fixture is wrong for this one.
    # Other 'time from last' columns have changed some, so we will need to verify they are working.
    columns_to_drop = [
        'time_from_last_change',
        'time_from_last_lick',
        'time_from_last_reward',
        'response_latency',
        'response_binary',
        'licks',
        'rewards'
    ]
    pd.testing.assert_frame_equal(
        extended_stim.drop(columns=columns_to_drop), 
        extended_stimulus_presentations.drop(columns=columns_to_drop),
        check_like=True
    )

    #Check licks
    for ind_row in range(len(extended_stim)):
        left = extended_stim.iloc[ind_row]['licks']
        right = extended_stimulus_presentations.iloc[ind_row]['licks']
        assert len(left) == len(right)
        if len(left)>0:
            np.testing.assert_array_equal(left, right)

    #Check rewards
    for ind_row in range(len(extended_stim)):
        left = extended_stim.iloc[ind_row]['rewards']
        right = extended_stimulus_presentations.iloc[ind_row]['rewards']
        assert len(left) == len(right)
        if len(left)>0:
            np.testing.assert_array_equal(left, right)
            

@pytest.mark.skipif(CIRCLECI, reason='Cannot test against real files on CircleCI')
def test_extended_stimulus_presentations_sfn_session(
        sfn_sdk_stimulus_presentations,
        sfn_sdk_licks,
        sfn_sdk_rewards,
        sfn_sdk_running_speed,
        sfn_sdk_extended_stimulus_presentations,
):
    _test_extended_stimulus_presentations(
        sfn_sdk_stimulus_presentations,
        sfn_sdk_licks,
        sfn_sdk_rewards,
        sfn_sdk_running_speed,
        sfn_sdk_extended_stimulus_presentations
    )

@pytest.fixture
@pytest.mark.skipif(CIRCLECI, reason='Cannot test against real files on CircleCI')
def sdk_session():
    # Get sdk session from cache
    from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache
    oeid = 880961028
    cache = BehaviorProjectCache.from_lims(manifest='manifest.json')
    session = cache.get_session_data(oeid)
    os.remove('manifest.json')
    return session

@pytest.mark.skipif(True, reason='need to debug this one more deeply')
@pytest.mark.skipif(CIRCLECI, reason='Cannot test against real files on CircleCI')
def test_extended_stimulus_presentations_sfn_session(
    sdk_session,
    sfn_sdk_extended_stimulus_presentations,
):

    _test_extended_stimulus_presentations(
        sdk_session.stimulus_presentations,
        af.convert_licks(sdk_session.licks),
        af.convert_rewards(sdk_session.rewards),
        af.convert_running_speed(sdk_session.running_speed),
        sfn_sdk_extended_stimulus_presentations
    )
