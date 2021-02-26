import pytest
import pandas as pd
import os
import numpy as np
from visual_behavior.translator.allensdk_sessions import sdk_utils
from visual_behavior.ophys.dataset import extended_stimulus_processing as esp


@pytest.fixture
@pytest.mark.onprem
def sdk_session(tmp_path):
    # Get sdk session from cache
    from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache
    oeid = 880961028
    cache = BehaviorProjectCache.from_lims(manifest=os.path.join(tmp_path, 'manifest.json'))
    session = cache.get_session_data(oeid)
    return session

@pytest.mark.onprem
def test_add_stimulus_presentations_analysis(sdk_session):
    '''
    Tests that we can add extended stim columns to session objects, and that the resulting session has
    one of the new columns.
    '''
    sdk_utils.add_stimulus_presentations_analysis(sdk_session)
    assert 'time_from_last_reward' in sdk_session.stimulus_presentations.columns

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
    licks_each_flash = esp.licks_each_flash(stimulus_presentations, licks_df)

    for ind_row in range(len(stimulus_presentations)):
        left = licks_each_flash.iloc[ind_row]
        right = expected.iloc[ind_row]['licks']
        assert len(left) == len(right)
        if len(left)>0:
            np.testing.assert_array_equal(left, right)

def test_add_licks_each_flash_mock(stimulus_onsets_df,
                                   mock_licks_df, 
                                   expected_stimulus_df_with_licks_and_rewards):
    '''
    Tests that adding licks each flash works with small mock dataframes
    '''
    _test_add_licks_each_flash(stimulus_onsets_df, mock_licks_df,
                               expected_stimulus_df_with_licks_and_rewards)

@pytest.mark.onprem
def test_add_licks_each_flash_sfn(sfn_sdk_stimulus_presentations,
                                  sfn_sdk_licks,
                                  sfn_sdk_extended_stimulus_presentations):
    '''
    Tests that adding licks each flash works with saved out session dataframes. 
    '''
    _test_add_licks_each_flash(sfn_sdk_stimulus_presentations,
                               sfn_sdk_licks,
                               sfn_sdk_extended_stimulus_presentations)

def _test_add_rewards_each_flash(stimulus_onsets_df,
                              rewards_df, 
                              expected_stimulus_df_with_licks_and_rewards):
    rewards_each_flash = esp.rewards_each_flash(stimulus_onsets_df, rewards_df)

    for ind_row in range(len(stimulus_onsets_df)):
        left = rewards_each_flash.iloc[ind_row]
        right = expected_stimulus_df_with_licks_and_rewards.iloc[ind_row]['rewards']
        assert len(left) == len(right)
        if len(left)>0:
            np.testing.assert_array_equal(left, right)


def test_add_rewards_each_flash_mock(stimulus_onsets_df,
                                   mock_rewards_df, 
                                   expected_stimulus_df_with_licks_and_rewards):
    '''
    Tests that adding rewards each flash works with small mock dataframes
    '''
    _test_add_rewards_each_flash(stimulus_onsets_df, mock_rewards_df,
                               expected_stimulus_df_with_licks_and_rewards)

@pytest.mark.onprem
def test_add_rewards_each_flash_sfn(sfn_sdk_stimulus_presentations,
                                  sfn_sdk_rewards,
                                  sfn_sdk_extended_stimulus_presentations):
    '''
    Tests that adding rewards each flash works with saved out session dataframes. 
    '''
    _test_add_rewards_each_flash(sfn_sdk_stimulus_presentations,
                               sfn_sdk_rewards,
                               sfn_sdk_extended_stimulus_presentations)
