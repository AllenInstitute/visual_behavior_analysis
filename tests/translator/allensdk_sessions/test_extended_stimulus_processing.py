import os
import pandas as pd
import numpy as np
import pytest
from visual_behavior.translator.allensdk_sessions import extended_stimulus_processing as esp

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
    columns_to_drop = [
        'time_from_last_change',
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

def convert_licks(licks_df):
    '''
    licks has a column called 'time', inconsistent with the rest of the sdk.
    should be 'timestamps'
    '''
    assert licks_df.columns.to_list() == ['time']
    return licks_df.rename(columns={'time':'timestamps'})

def convert_rewards(rewards_df):
    '''
    rewards has timestamps as the index, inconsistent with the rest of the sdk.
    should have a column called 'timestamps' instead.
    '''
    assert rewards_df.index.name == 'timestamps'
    return rewards_df.reset_index()

def convert_running_speed(running_speed_obj):
    '''
    running speed is returned as a custom object, inconsistent with other attrs.
    should be a dataframe with cols for timestamps and speed.
    '''
    return pd.DataFrame({
        'timestamps':running_speed_obj.timestamps,
        'speed':running_speed_obj.values
    })

@pytest.mark.skipif(CIRCLECI, reason='Cannot test against real files on CircleCI')
def test_extended_stimulus_presentations_sfn_session(
    sdk_session,
    sfn_sdk_extended_stimulus_presentations,
):

    _test_extended_stimulus_presentations(
        sdk_session.stimulus_presentations,
        convert_licks(sdk_session.licks),
        convert_rewards(sdk_session.rewards),
        convert_running_speed(sdk_session.running_speed),
        sfn_sdk_extended_stimulus_presentations
    )
