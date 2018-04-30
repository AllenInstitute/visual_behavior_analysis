import pandas as pd
from visual_behavior.validation.extended_trials import validate_autoreward_volume

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