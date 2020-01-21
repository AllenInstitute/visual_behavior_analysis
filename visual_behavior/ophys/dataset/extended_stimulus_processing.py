import numpy as np
import pandas as pd

def time_from_last(flash_times, other_times):

    last_other_index = np.searchsorted(a=other_times, v=flash_times, side='right') - 1
    time_from_last_other = flash_times - other_times[last_other_index]

    # flashes that happened before the other thing happened should return nan
    time_from_last_other[last_other_index == -1] = np.nan

    return time_from_last_other

def trace_average(values, timestamps, start_time, stop_time):

    if len(values) != len(timestamps):
        raise ValueError('values and timestamps must be the same length')
    if np.any(np.diff(timestamps)<=0):
        raise ValueError('timestamps must be monotonically increasing')
    if start_time < timestamps[0]:
        raise ValueError('start time must be within range of timestamps')
    if stop_time > timestamps[-1]:
        raise ValueError('stop time must be within range of timestamps')

    values_this_range = values[((timestamps >= start_time) & (timestamps < stop_time))]
    return values_this_range.mean()

def find_change(image_index, omitted_index):
    '''
    Args: 
        image_index (pd.Series): The index of the presented image for each flash
        omitted_index (int): The index value for omitted stimuli
    Returns:
        change (np.array of bool): Whether each flash was a change flash
    '''

    change = np.diff(image_index) != 0
    change = np.concatenate([np.array([False]), change])  # First flash not a change
    omitted = image_index == omitted_index
    omitted_inds = np.flatnonzero(omitted)
    change[omitted_inds] = False

    if image_index.iloc[-1] == omitted_index:
        # If the last flash is omitted we can't set the +1 for that omitted idx
        change[omitted_inds[:-1] + 1] = False
    else:
        change[omitted_inds + 1] = False
                                               
    return change

def get_omitted_index(stimulus_presentations_df):
    if 'omitted' in stimulus_presentations_df['image_name'].unique():
        omitted_index = stimulus_presentations_df.groupby("image_name").apply(
            lambda group: group["image_index"].unique()[0]
        )["omitted"]
    else:
        omitted_index=None
    return omitted_index

def mean_running_speed(stimulus_presentations_df, running_speed_df,
                           range_relative_to_stimulus_start=[0, 0.25]):
    '''
    Append a column to stimulus_presentations which contains the mean running speed between

    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations. 
            Must contain: 'start_time'
        running_speed_df (pd.DataFrame): dataframe of running speed. 
            Must contain: 'speed', 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed. 
    Returns:
        stimulus_presentations_df (pd.DataFrame): Same as the input, but with 'mean_running_speed'
    '''
    flash_running_speed = stimulus_presentations_df.apply(
        lambda row: trace_average(
            running_speed_df['speed'].values,
            running_speed_df['timestamps'].values,
            row["start_time"]+range_relative_to_stimulus_start[0],
            row["start_time"]+range_relative_to_stimulus_start[1],
        ),
        axis=1,
    )
    return flash_running_speed

def licks_each_flash(stimulus_presentations_df, licks_df,
                         range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of licks that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations. 
            Must contain: 'start_time'
        licks_df (pd.DataFrame): lick dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed. 
    Returns:
        stimulus_presentations_df (pd.DataFrame): Same as the input, but with 'licks' column added
    '''

    lick_times = licks_df['timestamps'].values
    licks_each_flash = stimulus_presentations_df.apply(
        lambda row: lick_times[
            ((
                lick_times > row["start_time"]+range_relative_to_stimulus_start[0]
            ) & (
                lick_times < row["start_time"]+range_relative_to_stimulus_start[1]
            ))
        ],
        axis=1,
    )
    return licks_each_flash

def rewards_each_flash(stimulus_presentations_df, rewards_df,
                           range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations_df (pd.DataFrame): dataframe of stimulus presentations. 
            Must contain: 'start_time'
        rewards_df (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed. 
    Returns:
        stimulus_presentations_df (pd.DataFrame): Same as the input, but with 'rewards' column added
    '''

    reward_times = rewards_df['timestamps'].values
    rewards_each_flash = stimulus_presentations_df.apply(
        lambda row: reward_times[
            ((
                reward_times > row["start_time"]+range_relative_to_stimulus_start[0]
            ) & (
                reward_times < row["start_time"]+range_relative_to_stimulus_start[1]
            ))
        ],
        axis=1,
    )
    stimulus_presentations_df["rewards"] = rewards_each_flash
    return stimulus_presentations_df

def get_block_index(image_index, omitted_index):
    changes = find_change(image_index, omitted_index)

    #Include the first presentation as a 'change'
    changes[0] = 1
    change_indices = np.flatnonzero(changes)

    flash_inds = np.arange(len(image_index))
    block_inds = np.searchsorted(a=change_indices, v=flash_inds, side="right") - 1
    return block_inds

#############################################################################################################


# TODO: Confusing me right now
#  def get_block_repetition_number(image_index, omitted_index):
#  
#      blocks_this_image = {}
#      for ind_enum, ind_image in enumerate(image_index[1:]):
#          last_image = image_index[ind_enum - 1]
#  
#  
#  
#      changes = find_change(image_index, omitted_index)
#      post_change_image = image_index[changes]
#      for image in post_change_image:
#          if image not in blocks_per_image:
#              blocks_per_image[image] = 0
#          else:
#              blocks_per_image[image] += 1
#  




def get_repeat_within_block(stimulus_presentations_df):
    repeat_number = np.full(len(stimulus_presentations_df), np.nan)
    assert (
        stimulus_presentations_df.iloc[0].name == 0
    )  # Assuming that the row index starts at zero
    for ind_group, group in stimulus_presentations_df.groupby("block_index"):
        repeat = 0
        for ind_row, row in group.iterrows():
            if row["image_name"] != "omitted":
                repeat_number[ind_row] = repeat
                repeat += 1

    stimulus_presentations_df["index_within_block"] = repeat_number
    return stimulus_presentations_df




#############################################################################################################

def add_response_latency(stimulus_presentations):
    st = stimulus_presentations.copy()
    st['response_latency'] = st['licks']-st['start_time']
    # st = st[st.response_latency.isnull()==False] #get rid of random NaN values
    st['response_latency'] = [response_latency[0] if len(response_latency)>0 else np.nan for response_latency in st['response_latency'].values ]
    st['response_binary'] = [True if np.isnan(response_latency)==False else False for response_latency in st.response_latency.values]
    st['early_lick'] = [True if response_latency<0.15 else False for response_latency in st['response_latency'].values ]
    return st

def add_inter_flash_lick_diff_to_stimulus_presentations(stimulus_presentations):
    st = stimulus_presentations.copy()
    st['first_lick'] = [licks[0] if len(licks)>0 else np.nan for licks in st['licks'].values]
    st['last_lick'] = [licks[-1] if len(licks)>0 else np.nan for licks in st['licks'].values]
    st['previous_trial_last_lick'] = np.hstack((np.nan, st.last_lick.values[:-1]))
    st['inter_flash_lick_diff'] = st['previous_trial_last_lick']- st['first_lick']
    return st

def add_first_lick_in_bout_to_stimulus_presentations(stimulus_presentations):
    st = stimulus_presentations.copy() # get median inter lick interval to threshold
    lick_times = st[st.licks.isnull()==False].licks.values
    median_inter_lick_interval = np.median(np.diff(np.hstack(list(lick_times))))
    # create first lick in bout boolean
    st['first_lick_in_bout'] = False # set all to False
    indices = st[st.response_binary].index
    st.at[indices, 'first_lick_in_bout'] = True # set stimulus_presentations with a lick to True
    indices = st[(st.response_binary)&(st.inter_flash_lick_diff<median_inter_lick_interval*3)].index
    st.at[indices, 'first_lick_in_bout'] = False # set stimulus_presentations with low inter lick interval back to False
    return st

def get_consumption_licks(stimulus_presentations):
    st = stimulus_presentations.copy()
    lick_times = st[st.licks.isnull()==False].licks.values
    # need to make this a hard threshold
    median_inter_lick_interval = np.median(np.diff(np.hstack(list(lick_times))))
    st['consumption_licks'] = False
    for row in range(len(st)):
        row_data = st.iloc[row]
        if (row_data.change==True) and (row_data.first_lick_in_bout==True):
            st.loc[row,'consumption_licks'] = True
        if (st.iloc[row-1].consumption_licks==True) & (st.iloc[row].inter_flash_lick_diff < median_inter_lick_interval*3):
            st.loc[row, 'consumption_licks'] = True
    return st

def annotate_licks(licks, rewards, bout_threshold=0.7):
    if 'bout_number' in licks:
        raise Exception('You already annotated this session, reload session first')
    licks['pre_ili'] = np.concatenate([[np.nan], np.diff(licks.timestamps.values)])
    licks['post_ili'] = np.concatenate([np.diff(licks.timestamps.values), [np.nan]])
    licks['rewarded'] = False
    for index, row in rewards.iterrows():
        licks_before_reward = np.where(licks.timestamps <= row.timestamps)[0]
        if len(licks_before_reward) > 0:
            mylick = licks_before_reward[-1]
            licks.at[mylick, 'rewarded'] = True
    licks['bout_start'] = licks['pre_ili'] > bout_threshold
    licks['bout_end'] = licks['post_ili'] > bout_threshold
    licks.at[licks['pre_ili'].apply(np.isnan), 'bout_start'] = True
    licks.at[licks['post_ili'].apply(np.isnan), 'bout_end'] = True
    licks['bout_number'] = np.cumsum(licks['bout_start'])
    x = licks.groupby('bout_number').any('rewarded').rename(columns={'rewarded': 'bout_rewarded'})
    licks['bout_rewarded'] = False
    temp = licks.reset_index().set_index('bout_number')
    temp.update(x)
    temp = temp.reset_index().set_index('index')
    licks['bout_rewarded'] = temp['bout_rewarded']


def annotate_bouts(stimulus_presentations, licks):
    '''
        Uses the bout annotations in session.licks to annotate session.stimulus_presentations
    '''
    bout_starts = licks[licks['bout_start']]
    stimulus_presentations['bout_start'] = False
    for index, x in bout_starts.iterrows():
        filter_start = stimulus_presentations[stimulus_presentations['start_time'].gt(x.timestamps)]
        if len(filter_start) > 0:
            stimulus_presentations.at[
                stimulus_presentations[stimulus_presentations['start_time'].gt(x.timestamps)].index[
                    0] - 1, 'bout_start'] = True

    bout_ends = licks[licks['bout_end']]
    stimulus_presentations['bout_end'] = False
    for index, x in bout_ends.iterrows():
        filter_start = stimulus_presentations[stimulus_presentations['start_time'].gt(x.timestamps)]
        if len(filter_start) > 0:
            stimulus_presentations.at[
                stimulus_presentations[stimulus_presentations['start_time'].gt(x.timestamps)].index[
                    0] - 1, 'bout_end'] = True

    stimulus_presentations.drop(-1, inplace=True, errors='ignore')


def annotate_flash_rolling_metrics(stimulus_presentations, win_dur=320, win_type='triang'):
    '''
        Get rolling flash level metrics for lick rate, reward rate, and bout_rate
    '''
    # Get Lick Rate / second
    stimulus_presentations['licked'] = [1 if len(this_lick) > 0 else 0 for this_lick in stimulus_presentations['licks']]
    stimulus_presentations['lick_rate'] = stimulus_presentations['licked'].rolling(win_dur, min_periods=1,
                                                                                   win_type=win_type).mean() / .75
    # Get Reward Rate / second
    stimulus_presentations['rewarded'] = [1 if len(this_reward) > 0 else 0 for this_reward in
                                          stimulus_presentations['rewards']]
    stimulus_presentations['reward_rate'] = stimulus_presentations['rewarded'].rolling(win_dur, min_periods=1,
                                                                                       win_type=win_type).mean() / .75
    # Get Running / Second
    stimulus_presentations['running_rate'] = stimulus_presentations['mean_running_speed'].rolling(win_dur,
                                                                                                  min_periods=1,
                                                                                                  win_type=win_type).mean() / .75
    # Get Bout Rate / second
    stimulus_presentations['bout_rate'] = stimulus_presentations['bout_start'].rolling(win_dur, min_periods=1,
                                                                                       win_type=win_type).mean() / .75
    return stimulus_presentations


def classify_by_flash_metrics(stimulus_presentations, lick_threshold=0.1, reward_threshold=2 / 80, use_bouts=True):
    '''
        Use the flash level rolling metrics to classify into three states based on the thresholds
    '''
    if use_bouts:
        stimulus_presentations['high_lick'] = [True if x > lick_threshold else False for x in
                                               stimulus_presentations['bout_rate']]
    else:
        stimulus_presentations['high_lick'] = [True if x > lick_threshold else False for x in
                                               stimulus_presentations['lick_rate']]
    stimulus_presentations['high_reward'] = [True if x > reward_threshold else False for x in
                                             stimulus_presentations['reward_rate']]
    stimulus_presentations['flash_metrics_epochs'] = [0 if (not x[0]) & (not x[1]) else 1 if x[1] else 2 for x in
                                                      zip(stimulus_presentations['high_lick'],
                                                          stimulus_presentations['high_reward'])]
    stimulus_presentations['flash_metrics_labels'] = [
        'low-lick,low-reward' if x == 0  else 'high-lick,high-reward' if x == 1 else 'high-lick,low-reward' for x in
        stimulus_presentations['flash_metrics_epochs']]

def get_metrics(sp, licks, rewards):
    '''
        Top level function that appeneds a few columns to session.stimulus_presentations
        ARGUMENTS:
            sp, an extended stimulus presentations table
            licks, the session.licks table
        Adds to the stimulus_presentations table
            bout_start, (boolean)
            licked, (boolean)
            lick_rate, licks/flash
            rewarded, (boolean)
            reward_rate, rewards/flash
            running_rate
            bout_rate, bouts/flash
            high_lick, (boolean)
            high_reward, (boolean)
            flash_metrics_epochs, (int)
            flash_metrics_labels, (string)
    '''
    annotate_licks(licks, rewards)
    annotate_bouts(sp, licks)
    annotate_flash_rolling_metrics(sp)
    classify_by_flash_metrics(sp)
    return sp

def add_prior_image_to_stimulus_presentations(stimulus_presentations):
    prior_image_name = [None]
    prior_image_name = prior_image_name + list(stimulus_presentations.image_name.values[:-1])
    stimulus_presentations['prior_image_name'] = prior_image_name
    return stimulus_presentations




#  def get_omitted(stimulus_presentations_df, omitted_index):
#      omitted = stimulus_presentations_df["image_index"] == omitted_index
#      stimulus_presentations_df["omitted"] = omitted
#      return stimulus_presentations_df

#  def get_changes(stimulus_presentations_df, omitted):
#      changes = find_change(stimulus_presentations_df["image_index"], omitted_index)
#      stimulus_presentations_df["change"] = changes
#      return stimulus_presentations_df

def get_extended_stimulus_presentations(stimulus_presentations_df,
                                        licks_df,
                                        rewards_df,
                                        running_speed_df):

    stimulus_presentations_df = stimulus_presentations_df.copy()
    #  stimulus_presentations_df = add_prior_image_to_stimulus_presentations(stimulus_presentations_df)

    flash_times = stimulus_presentations_df["start_time"].values
    # Time from last other for each flash

    # TODO: Move this into the logic of the individual funcs
    lick_times = licks_df['timestamps'].values
    reward_times = rewards_df['timestamps'].values
    if len(lick_times) < 5: #Passive sessions
        time_from_last_lick = np.full(len(flash_times), np.nan)
    else:
        time_from_last_lick = time_from_last(flash_times, lick_times)

    if len(reward_times) < 1: # Sometimes mice are bad
        time_from_last_reward = np.full(len(flash_times), np.nan)
    else:
        time_from_last_reward = time_from_last(flash_times, reward_times)

    stimulus_presentations_df["time_from_last_lick"] = time_from_last_lick
    stimulus_presentations_df["time_from_last_reward"] = time_from_last_reward



    change_times = stimulus_presentations_df.query('change')['start_time'].values
    time_from_last_change = time_from_last(flash_times, change_times)
    stimulus_presentations_df["time_from_last_change"] = time_from_last_change


    # Lists of licks/rewards on each flash
    stimulus_presentations_df = add_licks_each_flash(stimulus_presentations_df, licks_df)
    stimulus_presentations_df = add_rewards_each_flash(stimulus_presentations_df, rewards_df)

    stimulus_presentations_df = add_mean_running_speed(stimulus_presentations_df, running_speed_df)

    # add flash after omitted
    stimulus_presentations_df['flash_after_omitted'] = np.hstack((False, stimulus_presentations_df.omitted.values[:-1]))
    stimulus_presentations_df['flash_after_change'] = np.hstack((False, stimulus_presentations_df.change.values[:-1]))
    # add licking responses
    stimulus_presentations_df = add_response_latency(stimulus_presentations_df)
    stimulus_presentations_df = add_inter_flash_lick_diff_to_stimulus_presentations(stimulus_presentations_df)
    stimulus_presentations_df = add_first_lick_in_bout_to_stimulus_presentations(stimulus_presentations_df)
    stimulus_presentations_df = get_consumption_licks(stimulus_presentations_df)
    stimulus_presentations_df = get_metrics(stimulus_presentations_df, licks_df, rewards_df)
    # reward rate, lick rate, bout rate
    stimulus_presentations_df = annotate_flash_rolling_metrics(stimulus_presentations_df, win_dur=320, win_type='triang')

    return stimulus_presentations_df
