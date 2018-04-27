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
    
test_validate_autoreward_volume()