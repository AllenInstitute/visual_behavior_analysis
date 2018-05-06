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

def test_validate_licks_on_go_trials_earn_reward():
    #3 trials with first lick after change, but before response window. Correctly labeled as 'go' or 'catch'
    #1 trial with first lick before change, correctly labeled as 'aborted'
    #Rewards only on 'go' trials
    GOOD_DATA = pd.DataFrame({
        'trial_type':['go','go','catch','aborted'],
        'change_time':[867.943295,2132.735950,2359.626023,2500],
        'lick_times':[[868.07670185, 868.243553692, 868.343638907],[2132.8193812, 2133.00285741, 2134.73767532],[2359.72610696, 2359.95964697],[2499]],
        'response_window':[[0.15, 0.75],[0.15, 0.75],[0.15, 0.75],[0.15, 0.75]],
        'number_of_rewards':[1,1,0,0]
    })

    assert validate_licks_on_go_trials_earn_reward(GOOD_DATA)==True

    #3 trials with first lick after change, but before response window. Correctly labeled as 'go' or 'catch'
    #1 'go' trial without reward
    BAD_DATA = pd.DataFrame({
        'trial_type':['go','go','catch'],
        'change_time':[867.943295,2132.735950,2359.626023],
        'lick_times':[[868.07670185, 868.243553692, 868.343638907],[2132.8193812, 2133.00285741, 2134.73767532],[2359.72610696, 2359.95964697]],
        'response_window':[[0.15, 0.75],[0.15, 0.75],[0.15, 0.75]],
        'number_of_rewards':[1,0,0]
    })

    assert validate_licks_on_go_trials_earn_reward(BAD_DATA) == False

def test_validate_licks_on_catch_trials_do_not_earn_reward():
    #3 trials with first lick after change, but before response window. Correctly labeled as 'go' or 'catch'
    #1 trial with first lick before change, correctly labeled as 'aborted'
    #Rewards only on 'go' trials
    GOOD_DATA = pd.DataFrame({
        'trial_type':['go','go','catch','aborted'],
        'change_time':[867.943295,2132.735950,2359.626023,2500],
        'lick_times':[[868.07670185, 868.243553692, 868.343638907],[2132.8193812, 2133.00285741, 2134.73767532],[2359.72610696, 2359.95964697],[2499]],
        'response_window':[[0.15, 0.75],[0.15, 0.75],[0.15, 0.75],[0.15, 0.75]],
        'number_of_rewards':[1,1,0,0]
    })

    assert validate_licks_on_catch_trials_do_not_earn_reward(GOOD_DATA)==True

    #3 trials with first lick after change, but before response window. Correctly labeled as 'go' or 'catch'
    #1 'catch' trial with reward
    BAD_DATA = pd.DataFrame({
        'trial_type':['go','go','catch'],
        'change_time':[867.943295,2132.735950,2359.626023],
        'lick_times':[[868.07670185, 868.243553692, 868.343638907],[2132.8193812, 2133.00285741, 2134.73767532],[2359.72610696, 2359.95964697]],
        'response_window':[[0.15, 0.75],[0.15, 0.75],[0.15, 0.75]],
        'number_of_rewards':[1,1,1]
    })

    assert validate_licks_on_catch_trials_do_not_earn_reward(BAD_DATA) == False

def test_validate_number_aborted_trial_repeats():
    failure_repeats=2 #should repeat params twice after failure, for a total of 3 consecutive
    ## good data with two blocks of 3 aborted trials with matching scheduled change times
    GOOD_DATA = pd.DataFrame({
        'trial_type':(["go", ] * 2) + (["aborted", ] * 7),
        'scheduled_change_time':[3,4,2.5,2.5,2.5,4,4,4,5],
    })
    assert validate_number_aborted_trial_repeats(GOOD_DATA,failure_repeats) == True

    failure_repeats=2 #should repeat params twice after failure, for a total of 3 consecutive
    ## bad data with a block of 4 aborted trials with matching scheduled change times, followed by another block of 2
    BAD_DATA = pd.DataFrame({
        'trial_type':(["go", ] * 2) + (["aborted", ] * 7),
        'scheduled_change_time':[3,4,2.5,2.5,2.5,2.5,4,4,5],
    })
    assert validate_number_aborted_trial_repeats(BAD_DATA,failure_repeats) == False


def test_validate_even_sampling():

    ## good data with 1 sample of 3 combinations, 2 samples of (a to b)
    GOOD_DATA = pd.DataFrame({
        'trial_type':['go','go','catch','go','catch'],
        'initial_image_name':['a','b','a','a','b'],
        'change_image_name':['b','a','a','b','b'],
        'response':[0,0,0,0,0]
    })
    assert validate_even_sampling(GOOD_DATA,even_sampling_enabled=True) == True

    ## good data with 1 sample of 3 combinations, 3 samples of (a to b)
    BAD_DATA = pd.DataFrame({
        'trial_type':['go','go','catch','go','catch','go'],
        'initial_image_name':['a','b','a','a','b','a'],
        'change_image_name':['b','a','a','b','b','b'],
        'response':[0,0,0,0,0,0]
    })
    assert validate_even_sampling(BAD_DATA,even_sampling_enabled=True) == False

def test_validate_params_change_after_aborted_trial_repeats():
    failure_repeats=2 #should repeat params twice after failure, for a total of 3 consecutive
    ## good data with two blocks of 3 aborted trials with matching scheduled change times
    GOOD_DATA = pd.DataFrame({
        'trial_type':(["go", ] * 2) + (["aborted", ] * 7),
        'scheduled_change_time':[3,4,2.5,2.5,2.5,4,4,4,5],
    })
    assert validate_params_change_after_aborted_trial_repeats(GOOD_DATA,failure_repeats) == True

    failure_repeats=2 #should repeat params twice after failure, for a total of 3 consecutive
    ## bad data with a block of 4 aborted trials with matching scheduled change times, followed by another block of 2
    BAD_DATA = pd.DataFrame({
        'trial_type':(["go", ] * 2) + (["aborted", ] * 7),
        'scheduled_change_time':[3,4,2.5,2.5,2.5,2.5,4,4,5],
    })
    assert validate_params_change_after_aborted_trial_repeats(BAD_DATA,failure_repeats) == False

def test_validate_flash_blank_durations():
    ## good data: flashes at ~250ms, blanks at ~500ms
    GOOD_DATA=pd.DataFrame({
        'time':[  1.66710466e-03,   7.68953469e-01,   1.53593739e+00, 2.30325807e+00,   3.07040749e+00],
        'duration':[ 0.25189872,  0.25191379,  0.25223227,  0.25221559,  0.25236601]
    })

    assert validate_flash_blank_durations(GOOD_DATA,expected_flash_duration=0.25,expected_blank_duration=0.5)==True

    ## bad data: flashes at ~250ms with one over 300ms, blanks at ~500ms
    BAD_DATA=pd.DataFrame({
        'time':[  1.66710466e-03,   7.68953469e-01,   1.53593739e+00, 2.30325807e+00,   3.07040749e+00],
        'duration':[ 0.25189872,  0.25191379,  0.25223227,  0.30221559,  0.25236601]
    })
    assert validate_flash_blank_durations(BAD_DATA,expected_flash_duration=0.25,expected_blank_duration=0.5)==False

def test_validate_two_stimuli_per_go_trial():
    #good data: two stimulus categoris per go trial
    GOOD_DATA_TRIALS = pd.DataFrame({
        'startframe':[1,3],
        'endframe':[2,4],
        'trial_type':['go','go']
    })
    GOOD_DATA_VISUAL_STIMULI = pd.DataFrame({
        'frame':[1,2,3,4],
        'image_category':['a','b','a','b']
    })
    assert validate_two_stimuli_per_go_trial(GOOD_DATA_TRIALS,GOOD_DATA_VISUAL_STIMULI) == True

    #bad data: only one stimulus group on second go trial
    BAD_DATA_TRIALS = pd.DataFrame({
        'startframe':[1,3],
        'endframe':[2,4],
        'trial_type':['go','go']
    })
    BAD_DATA_VISUAL_STIMULI = pd.DataFrame({
        'frame':[1,2,3,4],
        'image_category':['a','b','a','a']
    })
    assert validate_two_stimuli_per_go_trial(BAD_DATA_TRIALS,BAD_DATA_VISUAL_STIMULI) == False

def test_validate_one_stimulus_per_catch_trial():
    #good data: only one stimulus group per catch trial
    GOOD_DATA_TRIALS = pd.DataFrame({
        'startframe':[1,3],
        'endframe':[2,4],
        'trial_type':['catch','catch']
    })
    GOOD_DATA_VISUAL_STIMULI = pd.DataFrame({
        'frame':[1,2,3,4],
        'image_category':['a','a','a','a']
    })
    assert validate_one_stimulus_per_catch_trial(GOOD_DATA_TRIALS,GOOD_DATA_VISUAL_STIMULI) == True

    #bad data: two stimulus group on first catch trial
    BAD_DATA_TRIALS = pd.DataFrame({
        'startframe':[1,3],
        'endframe':[2,4],
        'trial_type':['catch','catch']
    })
    BAD_DATA_VISUAL_STIMULI = pd.DataFrame({
        'frame':[1,2,3,4],
        'image_category':['a','b','a','a']
    })
    assert validate_one_stimulus_per_catch_trial(BAD_DATA_TRIALS,BAD_DATA_VISUAL_STIMULI) == False

def test_validate_one_stimulus_per_aborted_trial():
    #good data: only one stimulus group per aborted trial
    GOOD_DATA_TRIALS = pd.DataFrame({
        'startframe':[1,3],
        'endframe':[2,4],
        'trial_type':['aborted','aborted']
    })
    GOOD_DATA_VISUAL_STIMULI = pd.DataFrame({
        'frame':[1,2,3,4],
        'image_category':['a','a','a','a']
    })
    assert validate_one_stimulus_per_aborted_trial(GOOD_DATA_TRIALS,GOOD_DATA_VISUAL_STIMULI) == True

    #bad data: two stimulus group on first aborted trial
    BAD_DATA_TRIALS = pd.DataFrame({
        'startframe':[1,3],
        'endframe':[2,4],
        'trial_type':['aborted','aborted']
    })
    BAD_DATA_VISUAL_STIMULI = pd.DataFrame({
        'frame':[1,2,3,4],
        'image_category':['a','b','a','a']
    })
    assert validate_one_stimulus_per_aborted_trial(BAD_DATA_TRIALS,BAD_DATA_VISUAL_STIMULI) == False

def test_validate_change_frame_at_flash_onset():
    #good data: changes coincide with flashes
    GOOD_DATA_TRIALS = pd.DataFrame({
        'change_frame':[10,20,30],
    })
    GOOD_DATA_VISUAL_STIMULI = pd.DataFrame({
        'frame':[10,20,30],
    })
    assert validate_change_frame_at_flash_onset(GOOD_DATA_TRIALS,GOOD_DATA_VISUAL_STIMULI) == True
                                                
    #bad data: 3rd change does not coincide with flash
    BAD_DATA_TRIALS = pd.DataFrame({
        'change_frame':[10,20,31],
    })
    BAD_DATA_VISUAL_STIMULI = pd.DataFrame({
        'frame':[10,20,30],
    })
    assert validate_change_frame_at_flash_onset(BAD_DATA_TRIALS,BAD_DATA_VISUAL_STIMULI) == False

def test_validate_running_data():
    #good data: length matches time and not all values the same
    GOOD_DATA={}
    GOOD_DATA['running']=pd.DataFrame({
        'speed':[3,4,5]
    })
    GOOD_DATA['time']=np.array([1,2,3])
    assert validate_running_data(GOOD_DATA)==True
    
    #bad data: length does not matche time
    BAD_DATA={}
    BAD_DATA['running']=pd.DataFrame({
        'speed':[3,4,5,6]
    })
    BAD_DATA['time']=np.array([1,2,3])
    assert validate_running_data(BAD_DATA)==False
    
    #bad data: all values of speed are zero
    BAD_DATA={}
    BAD_DATA['running']=pd.DataFrame({
        'speed':[0,0,0]
    })
    BAD_DATA['time']=np.array([1,2,3])
    assert validate_running_data(BAD_DATA)==False

def test_validate_licks():
    GOOD_DATA = {}
    GOOD_DATA['licks']=pd.DataFrame({
        'time':[10,20],
        'frame':[1,2]
    })
    assert validate_licks(GOOD_DATA)==True
    
    BAD_DATA = {}
    BAD_DATA['licks']=pd.DataFrame({
        'time':[],
        'frame':[]
    })
    assert validate_licks(BAD_DATA)==False

def test_validate_frame_intervals_exists():
    test_data={}
    test_data['items']={}
    test_data['items']['behavior']={}
    #populated list should pass
    test_data['items']['behavior']['intervalsms']=[16,16]
    assert validate_frame_intervals_exists(test_data)==True
    
    #empty list should fail
    test_data['items']['behavior']['intervalsms']=[]
    assert validate_frame_intervals_exists(test_data)==False

def test_validate_initial_blank():
    initial_blank=0
    #good data: stimuli coincident with onset frames
    GOOD_DATA_TRIALS = pd.DataFrame({
        'startframe':[0,5,10,15],
        'starttime':[0.1,5.1,10.101,15.1]
    })
    GOOD_DATA_VISUAL_STIMULI = pd.DataFrame({
        'frame':[0,5,10,15],
        'time':[0.1,5.1,10.101,15.1]
    })
    assert validate_initial_blank(GOOD_DATA_TRIALS,GOOD_DATA_VISUAL_STIMULI,initial_blank) == True

    #bad data: 2nd stimulus 200 ms early, 3rd stimulus 200 ms late
    BAD_DATA_TRIALS = pd.DataFrame({
        'startframe':[0,5,10,15],
        'starttime':[0.1,5.1,10.101,15.1]
    })
    BAD_DATA_VISUAL_STIMULI = pd.DataFrame({
        'frame':[0,5,10,15],
        'time':[0.1,4.9,10.201,15.1]
    })
    assert validate_initial_blank(BAD_DATA_TRIALS,BAD_DATA_VISUAL_STIMULI,initial_blank) == False