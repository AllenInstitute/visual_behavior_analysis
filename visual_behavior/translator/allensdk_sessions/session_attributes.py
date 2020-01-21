import pandas as pd

# This file contains functions to reformat sdk session attributes to conform with our 
# design decisions. If we can get some of these changes backported into the SDK, then they
# can be removed from this module.

def convert_licks(licks_df):
    '''
    licks has a column called 'time', inconsistent with the rest of the sdk.
    should be 'timestamps'
    '''
    assert 'time' in licks_df.columns
    return licks_df.rename(columns={'time':'timestamps'})

def convert_rewards(rewards_df):
    '''
    rewards has timestamps as the index, inconsistent with the rest of the sdk.
    should have a column called 'timestamps' instead.
    '''
    assert rewards_df.index.name == 'timestamps'
    return rewards_df.reset_index()

def convert_licks_inplace(licks_df):
    '''
    licks has a column called 'time', inconsistent with the rest of the sdk.
    should be 'timestamps'
    '''
    assert 'time' in licks_df.columns
    licks_df.rename(columns={'time':'timestamps'}, inplace=True)

def convert_rewards_inplace(rewards_df):
    '''
    rewards has timestamps as the index, inconsistent with the rest of the sdk.
    should have a column called 'timestamps' instead.
    '''
    assert rewards_df.index.name == 'timestamps'
    rewards_df.reset_index(inplace=True)

def convert_running_speed(running_speed_obj):
    '''
    running speed is returned as a custom object, inconsistent with other attrs.
    should be a dataframe with cols for timestamps and speed.
    '''
    return pd.DataFrame({
        'timestamps':running_speed_obj.timestamps,
        'speed':running_speed_obj.values
    })
