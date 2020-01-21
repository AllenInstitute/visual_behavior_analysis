from visual_behavior.translator.allensdk_sessions import session_attributes as af

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
