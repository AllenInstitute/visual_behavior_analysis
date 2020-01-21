import pandas as pd
import visual_behavior.ophys.dataset.extended_stimulus_processing as esp

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

def add_change_each_flash_inplace(session):
    changes = esp.find_change(session.stimulus_presentations['image_index'], get_omitted_index(session.stimulus_presentations))
    session.stimulus_presentations['change'] = changes

def add_mean_running_speed_inplace(session,range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the mean running speed between

    Args:
        session object with:
            stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations. 
                Must contain: 'start_time'
        running_speed_df (pd.DataFrame): dataframe of running speed. 
            Must contain: 'speed', 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed. 
    Returns:
        nothing, modifies session in place. Same as the input, but with 'mean_running_speed' column added
    '''
    mean_running_speed_df = esp.mean_running_speed(session.stimulus_presentations,
                                            session.running_speed,
                                            range_relative_to_stimulus_start)
    session.stimulus_presentations["mean_running_speed"] = mean_running_speed_df


def add_licks_each_flash_inplace(session,range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of licks that occur
    in a range relative to the onset of the stimulus.

    Args:
        session object with:
            stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations. 
                Must contain: 'start_time'
            licks_df (pd.DataFrame): lick dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed. 
    Returns:
        nothing, modifies session in place. Same as the input, but with 'licks' column added
    '''

    licks_each_flash_df = esp.licks_each_flash(session.stimulus_presentations,
                                            session.licks,
                                            range_relative_to_stimulus_start)
    session.stimulus_presentations["licks"] = licks_each_flash_df


def add_rewards_each_flash_inplace(session,range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        session object, with attributes:
            stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations. 
                Must contain: 'start_time'
            rewards_df (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed. 
    Returns:
        nothing. session.stimulus_presentations is modified in place with 'rewards' column added
    '''

    rewards_each_flash_df = rewards_each_flash(session.stimulus_presentations,
                                                 session.rewards,
                                                 range_relative_to_stimulus_start)
    session.stimulus_presentations["rewards"] = rewards_each_flash_df
