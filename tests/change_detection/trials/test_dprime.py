import visual_behavior.utilities as vbu


def test_d_prime(mock_trials_fixture):
    hr_no_limits, far_no_limits, dp_no_limits = vbu.get_response_rates(mock_trials_fixture, apply_trial_number_limit=False)
    assert dp_no_limits[2] == 4.6526957480816815

    hr_with_limits, far_with_limits, dp_with_limits = vbu.get_response_rates(mock_trials_fixture, apply_trial_number_limit=True)
    assert dp_with_limits[2] == 1.3489795003921634
