import numpy as np
import pandas as pd
from visual_behavior.validation.extended_trials import *

def test_validate_autoreward_volume():

    TRUE_VOLUME = 1.0
    GOOD_TRIALS = pd.DataFrame(
        dict(auto_rewarded=True,number_of_rewards=1,reward_volume=1.0),
        dict(auto_rewarded=True,number_of_rewards=1,reward_volume=1.0),
    )

    assert validate_autoreward_volume(GOOD_TRIALS,TRUE_VOLUME)==True

    BAD_TRIALS = pd.DataFrame(
        dict(auto_rewarded=True,number_of_rewards=1,reward_volume=0.5),
        dict(auto_rewarded=True,number_of_rewards=1,reward_volume=1.0),
    )
    assert validate_autoreward_volume(BAD_TRIALS,TRUE_VOLUME)==False

def test_validate_trial_times_never_overlap():
    GOOD_TRIALS = pd.DataFrame({
        'starttime':[ 0.        ,  0.76716053,  6.9556934 ],
        'endtime':[  0.75048087,   6.93895647,  12.32681855]
    })

    assert validate_trial_times_never_overlap(GOOD_TRIALS)==True

    #bad trials have the second starttime before the first endtime
    BAD_TRIALS = pd.DataFrame({
        'starttime':[ 0.        ,  0.76716053-0.02,  6.9556934 ],
        'endtime':[  0.75048087,   6.93895647,  12.32681855]
    })

    assert validate_trial_times_never_overlap(BAD_TRIALS)==False

def test_validate_stimulus_distribution_key():
    expected_distribution_name='exponential'
    GOOD_TRIALS = pd.DataFrame({
        'stimulus_distribution':['exponential','exponential'],
    })
    assert validate_stimulus_distribution_key(GOOD_TRIALS,expected_distribution_name)==True

    BAD_TRIALS = pd.DataFrame({
        'stimulus_distribution':['exponential','not_exponential'],
    })
    assert validate_stimulus_distribution_key(BAD_TRIALS,expected_distribution_name)==False

def test_validate_change_time_mean():
    EXPECTED_MEAN = 2

    np.random.seed(seed=100)
    simulated_change_times_good=np.random.exponential(scale=2,size=100,)
    GOOD_TRIALS = pd.DataFrame({
        'change_time':simulated_change_times_good,
        'starttime':np.zeros_like(simulated_change_times_good),
        'prechange_minimum':np.zeros_like(simulated_change_times_good)
    })

    assert validate_change_time_mean(GOOD_TRIALS,EXPECTED_MEAN,tolerance=0.5)==True

    np.random.seed(seed=100)
    simulated_change_times_bad=np.random.exponential(scale=3,size=100)
    BAD_TRIALS = pd.DataFrame({
        'change_time':simulated_change_times_bad,
        'starttime':np.zeros_like(simulated_change_times_bad),
        'prechange_minimum':np.zeros_like(simulated_change_times_bad),
    })

    assert validate_change_time_mean(BAD_TRIALS,EXPECTED_MEAN,tolerance=0.5)==False

def test_validate_monotonically_decreasing_number_of_change_times():
    np.random.seed(10)
    ## chose change times from an exponential distribution
    exponential_change_times = np.random.exponential(3,5000)
    GOOD_TRIALS = pd.DataFrame({
        'change_time':exponential_change_times,
        'starttime':np.zeros_like(exponential_change_times)
    })
    #exponentially distributed values should pass monotonocity test
    assert validate_monotonically_decreasing_number_of_change_times(GOOD_TRIALS,'exponential') == True

    uniform_change_times = np.random.uniform(low=0,high=8,size=5000)
    BAD_TRIALS = pd.DataFrame({
        'change_time':uniform_change_times,
        'starttime':np.zeros_like(uniform_change_times)
    })
    #uniformly distributed values should fail monotonocity test
    assert validate_monotonically_decreasing_number_of_change_times(BAD_TRIALS,'exponential') == False

    #Bad trials should pass if distribution is not exponential
    assert validate_monotonically_decreasing_number_of_change_times(BAD_TRIALS,'uniform') == True


def test_validate_no_abort_on_lick_before_response_window():
    #3 trials with first lick after change, but before response window. Correctly labeled as 'go' or 'catch'
    #1 trial with first lick before change, correctly labeled as 'aborted'
    GOOD_DATA = pd.DataFrame({
        'trial_type':['go','go','catch','aborted'],
        'change_time':[867.943295,2132.735950,2359.626023,2500],
        'lick_times':[[868.07670185, 868.243553692, 868.343638907],[2132.8193812, 2133.00285741, 2134.73767532],[2359.72610696, 2359.95964697],[2499]],
        'response_window':[[0.15, 0.75],[0.15, 0.75],[0.15, 0.75],[0.15, 0.75]]
    })

    assert validate_no_abort_on_lick_before_response_window(GOOD_DATA)==True

    #3 trials with first lick after change, but before response window. Correctly labeled as 'go' or 'catch'
    #1 trial with first lick but before response window. Incorrectly labeled as 'aborted'
    BAD_DATA = pd.DataFrame({
        'trial_type':['go','go','catch','aborted'],
        'change_time':[867.943295,2132.735950,2359.626023,2500],
        'lick_times':[[868.07670185, 868.243553692, 868.343638907],[2132.8193812, 2133.00285741, 2134.73767532],[2359.72610696, 2359.95964697],[2500.1]],
        'response_window':[[0.15, 0.75],[0.15, 0.75],[0.15, 0.75],[0.15, 0.75]]
    })

    assert validate_no_abort_on_lick_before_response_window(BAD_DATA)==False
