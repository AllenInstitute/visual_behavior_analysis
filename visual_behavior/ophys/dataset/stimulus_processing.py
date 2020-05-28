# @nickponvert

import numpy as np
import pandas as pd
import pickle
from allensdk.brain_observatory.behavior import IMAGE_SETS

IMAGE_SETS_REV = {val: key for key, val in IMAGE_SETS.items()}


def convert_filepath_caseinsensitive(filename_in):
    if filename_in == '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_6_2017.07.14.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_6_2017.07.14.pkl'
    elif filename_in == '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl'
    elif filename_in == '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_TRAINING_2017.07.14.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl'
    elif filename_in == '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl'
    elif filename_in == '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_6_2017.07.14.pkl':
        return '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/image_dictionaries/Natural_Images_Lum_Matched_set_ophys_6_2017.07.14.pkl'
    else:
        raise NotImplementedError(filename_in)


def load_pickle(pstream):
    return pickle.load(pstream, encoding="bytes")


def add_run_speed_to_trials(running_speed_df, trials):
    trial_running_speed = trials.apply(lambda row: trace_average(
        running_speed_df['speed'].values,
        running_speed_df['time'].values,
        row["change_time"],
        row["change_time"] + 0.25, ), axis=1, )
    trials["mean_running_speed"] = trial_running_speed
    return trials


def get_stimulus_presentations(data, stimulus_timestamps):
    stimulus_table = get_visual_stimuli_df(data, stimulus_timestamps)
    # workaround to rename columns to harmonize with visual coding and rebase timestamps to sync time
    stimulus_table.insert(loc=0, column='flash_number', value=np.arange(0, len(stimulus_table)))
    stimulus_table = stimulus_table.rename(
        columns={'frame': 'start_frame', 'time': 'start_time', 'flash_number': 'stimulus_presentations_id'})
    stimulus_table.start_time = [stimulus_timestamps[start_frame] for start_frame in stimulus_table.start_frame.values]
    end_time = []
    for end_frame in stimulus_table.end_frame.values:
        if not np.isnan(end_frame):
            end_time.append(stimulus_timestamps[int(end_frame)])
        else:
            end_time.append(float('nan'))

    stimulus_table.insert(loc=4, column='stop_time', value=end_time)
    stimulus_table.set_index('stimulus_presentations_id', inplace=True)
    stimulus_table = stimulus_table[sorted(stimulus_table.columns)]
    return stimulus_table


def get_images_dict(pkl):
    # Sometimes the source is a zipped pickle:
    metadata = {'image_set': pkl["items"]["behavior"]["stimuli"]["images"]["image_path"]}

    # Get image file name; these are encoded case-insensitive in the pickle file :/
    filename = convert_filepath_caseinsensitive(metadata['image_set'])

    image_set = load_pickle(open(filename, 'rb'))
    images = []
    images_meta = []

    ii = 0
    for cat, cat_images in image_set.items():
        for img_name, img in cat_images.items():
            meta = dict(
                image_category=cat.decode("utf-8"),
                image_name=img_name.decode("utf-8"),
                image_index=ii,
            )

            images.append(img)
            images_meta.append(meta)

            ii += 1

    images_dict = dict(
        metadata=metadata,
        images=images,
        image_attributes=images_meta,
    )

    return images_dict


def get_stimulus_templates(pkl):
    images = get_images_dict(pkl)
    image_set_filename = convert_filepath_caseinsensitive(images['metadata']['image_set'])
    return {IMAGE_SETS_REV[image_set_filename]: np.array(images['images'])}


def get_stimulus_metadata(pkl):
    images = get_images_dict(pkl)
    stimulus_index_df = pd.DataFrame(images['image_attributes'])
    image_set_filename = convert_filepath_caseinsensitive(images['metadata']['image_set'])
    stimulus_index_df['image_set'] = IMAGE_SETS_REV[image_set_filename]

    # Add an entry for omitted stimuli
    omitted_df = pd.DataFrame({'image_category': ['omitted'],
                               'image_name': ['omitted'],
                               'image_set': ['omitted'],
                               'image_index': [stimulus_index_df['image_index'].max() + 1]})
    stimulus_index_df = stimulus_index_df.append(omitted_df, ignore_index=True, sort=False)
    stimulus_index_df.set_index(['image_index'], inplace=True, drop=True)
    return stimulus_index_df


def _resolve_image_category(change_log, frame):
    for change in (unpack_change_log(c) for c in change_log):
        if frame < change['frame']:
            return change['from_category']

    return change['to_category']


def _get_stimulus_epoch(set_log, current_set_index, start_frame, n_frames):
    try:
        next_set_event = set_log[current_set_index + 1]  # attr_name, attr_value, time, frame
    except IndexError:  # assume this is the last set event
        next_set_event = (None, None, None, n_frames,)

    return (start_frame, next_set_event[3])  # end frame isnt inclusive


def _get_draw_epochs(draw_log, start_frame, stop_frame):
    """start_frame inclusive, stop_frame non-inclusive
    """
    draw_epochs = []
    current_frame = start_frame

    while current_frame <= stop_frame:
        epoch_length = 0
        while current_frame < stop_frame and draw_log[current_frame] == 1:
            epoch_length += 1
            current_frame += 1
        else:
            current_frame += 1

        if epoch_length:
            draw_epochs.append(
                (current_frame - epoch_length - 1, current_frame - 1,)
            )

    return draw_epochs


def unpack_change_log(change):
    (from_category, from_name), (to_category, to_name,), time, frame = change

    return dict(
        frame=frame,
        time=time,
        from_category=from_category,
        to_category=to_category,
        from_name=from_name,
        to_name=to_name,
    )


def get_visual_stimuli_df(data, time):
    stimuli = data['items']['behavior']['stimuli']
    n_frames = len(time)
    visual_stimuli_data = []
    for stimuli_group_name, stim_dict in stimuli.items():
        for idx, (attr_name, attr_value, _time, frame,) in \
                enumerate(stim_dict["set_log"]):
            orientation = attr_value if attr_name.lower() == "ori" else np.nan
            image_name = attr_value if attr_name.lower() == "image" else np.nan

            stimulus_epoch = _get_stimulus_epoch(
                stim_dict["set_log"],
                idx,
                frame,
                n_frames,
            )
            draw_epochs = _get_draw_epochs(
                stim_dict["draw_log"],
                *stimulus_epoch
            )

            for idx, (epoch_start, epoch_end,) in enumerate(draw_epochs):
                # visual stimulus doesn't actually change until start of
                # following frame, so we need to bump the epoch_start & epoch_end
                # to get the timing right
                epoch_start += 1
                epoch_end += 1

                visual_stimuli_data.append({
                    "orientation": orientation,
                    "image_name": image_name,
                    "frame": epoch_start,
                    "end_frame": epoch_end,
                    "time": time[epoch_start],
                    "duration": time[epoch_end] - time[epoch_start],
                    # this will always work because an epoch will never occur near the end of time
                    "omitted": False,
                })

    visual_stimuli_df = pd.DataFrame(data=visual_stimuli_data)

    # ensure that every rising edge in the draw_log is accounted for in the visual_stimuli_df
    #  draw_log_rising_edges = len(np.where(np.diff(stimuli['images']['draw_log'])==1)[0])
    #  discrete_flashes = len(visual_stimuli_data)
    #  assert draw_log_rising_edges == discrete_flashes, "the number of rising edges in the draw log is expected to match the number of flashes in the stimulus table"

    # Add omitted flash info:
    if 'omitted_flash_frame_log' in data['items']['behavior']:
        omitted_flash_list = []
        omitted_flash_frame_log = data['items']['behavior']['omitted_flash_frame_log']
        for stimuli_group_name, omitted_flash_frames in omitted_flash_frame_log.items():
            stim_frames = visual_stimuli_df['frame'].values
            omitted_flash_frames = np.array(omitted_flash_frames)

            #  Test offsets of omitted flash frames to see if they are in the stim log
            offsets = np.arange(-3, 4)
            offset_arr = np.add(np.repeat(omitted_flash_frames[:, np.newaxis], offsets.shape[0], axis=1), offsets)
            matched_any_offset = np.any(np.isin(offset_arr, stim_frames), axis=1)

            #  Remove omitted flashes that also exist in the stimulus log
            was_true_omitted = np.logical_not(matched_any_offset)  # bool
            omitted_flash_frames_to_keep = omitted_flash_frames[was_true_omitted]

            # Have to remove frames that are double-counted in omitted log
            omitted_flash_list += list(np.unique(omitted_flash_frames_to_keep))

        omitted = np.ones_like(omitted_flash_list).astype(bool)
        time = [time[fi] for fi in omitted_flash_list]
        omitted_df = pd.DataFrame({'omitted': omitted, 'frame': omitted_flash_list, 'time': time,
                                   'image_name': 'omitted'})

        df = pd.concat((visual_stimuli_df, omitted_df), sort=False).sort_values('frame').reset_index()
    else:
        df = visual_stimuli_df
    return df


def time_from_last(flash_times, other_times):
    last_other_index = np.searchsorted(a=other_times, v=flash_times) - 1
    time_from_last_other = flash_times - other_times[last_other_index]

    # flashes that happened before the other thing happened should return nan
    time_from_last_other[last_other_index == -1] = np.nan

    return time_from_last_other


def trace_average(values, timestamps, start_time, stop_time):
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


def add_response_latency(stimulus_presentations):
    st = stimulus_presentations.copy()
    st['response_latency'] = st['licks'] - st['start_time']
    # st = st[st.response_latency.isnull()==False] #get rid of random NaN values
    st['response_latency'] = [response_latency[0] if len(response_latency) > 0 else np.nan for response_latency in
                              st['response_latency'].values]
    st['response_binary'] = [True if np.isnan(response_latency) == False else False for response_latency in
                             st.response_latency.values]
    st['early_lick'] = [True if response_latency < 0.15 else False for response_latency in
                        st['response_latency'].values]
    return st


def add_inter_flash_lick_diff_to_stimulus_presentations(stimulus_presentations):
    st = stimulus_presentations.copy()
    st['first_lick'] = [licks[0] if len(licks) > 0 else np.nan for licks in st['licks'].values]
    st['last_lick'] = [licks[-1] if len(licks) > 0 else np.nan for licks in st['licks'].values]
    st['previous_trial_last_lick'] = np.hstack((np.nan, st.last_lick.values[:-1]))
    st['inter_flash_lick_diff'] = st['previous_trial_last_lick'] - st['first_lick']
    return st


def add_first_lick_in_bout_to_stimulus_presentations(stimulus_presentations):
    st = stimulus_presentations.copy()
    # get median inter lick interval to threshold
    lick_times = st[st.licks.isnull() == False].licks.values
    median_inter_lick_interval = np.median(np.diff(np.hstack(list(lick_times))))
    # create first lick in bout boolean
    st['first_lick_in_bout'] = False  # set all to False
    indices = st[st.response_binary].index
    st.at[indices, 'first_lick_in_bout'] = True  # set stimulus_presentations with a lick to True
    indices = st[(st.response_binary) & (st.inter_flash_lick_diff < median_inter_lick_interval * 3)].index
    st.at[
        indices, 'first_lick_in_bout'] = False  # set stimulus_presentations with low inter lick interval back to False
    return st


def get_consumption_licks(stimulus_presentations):
    st = stimulus_presentations.copy()
    lick_times = st[st.licks.isnull() == False].licks.values
    # need to make this a hard threshold
    median_inter_lick_interval = np.median(np.diff(np.hstack(list(lick_times))))
    st['consumption_licks'] = False
    for row in range(len(st)):
        row_data = st.iloc[row]
        if (row_data.change == True) and (row_data.first_lick_in_bout == True):
            st.loc[row, 'consumption_licks'] = True
        if (st.iloc[row - 1].consumption_licks == True) & (st.iloc[row].inter_flash_lick_diff < median_inter_lick_interval * 3):
            st.loc[row, 'consumption_licks'] = True
    return st


def annotate_licks(licks, rewards, bout_threshold=0.7):
    if 'bout_number' in licks:
        raise Exception('You already annotated this session, reload session first')
    licks['pre_ili'] = np.concatenate([[np.nan], np.diff(licks.timestamps.values)])
    licks['post_ili'] = np.concatenate([np.diff(licks.timestamps.values), [np.nan]])
    licks['rewarded'] = False
    for index, row in rewards.iterrows():
        mylick = np.where(licks.timestamps <= row.timestamps)[0][-1]
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
    # # Get Bout Rate / second
    # stimulus_presentations['bout_rate'] = stimulus_presentations['bout_start'].rolling(win_dur, min_periods=1,
    #                                                                                    win_type=win_type).mean() / .75
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
        'low-lick,low-reward' if x == 0 else 'high-lick,high-reward' if x == 1 else 'high-lick,low-reward' for x in
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


def get_extended_stimulus_presentations(stimulus_presentations_df,
                                        licks,
                                        rewards,
                                        change_times,
                                        running_speed_df,
                                        pupil_area):
    # stimulus_presentations_df = stimulus_presentations_df.copy()
    stimulus_presentations_df = add_prior_image_to_stimulus_presentations(stimulus_presentations_df)

    flash_times = stimulus_presentations_df["start_time"].values
    lick_times = licks['time'].values
    reward_times = rewards['time'].values
    # Time from last other for each flash

    if len(lick_times) < 5:  # Passive sessions
        time_from_last_lick = np.full(len(flash_times), np.nan)
    else:
        time_from_last_lick = time_from_last(flash_times, lick_times)

    if len(reward_times) < 1:  # Sometimes mice are bad
        time_from_last_reward = np.full(len(flash_times), np.nan)
    else:
        time_from_last_reward = time_from_last(flash_times, reward_times)

    time_from_last_change = time_from_last(flash_times, change_times)

    stimulus_presentations_df["time_from_last_lick"] = time_from_last_lick
    stimulus_presentations_df["time_from_last_reward"] = time_from_last_reward
    stimulus_presentations_df["time_from_last_change"] = time_from_last_change

    # Was the flash a change flash?
    if 'omitted' in stimulus_presentations_df['image_name'].unique():
        omitted_index = stimulus_presentations_df.groupby("image_name").apply(
            lambda group: group["image_index"].unique()[0]
        )["omitted"]
    else:
        omitted_index = None
    changes = find_change(stimulus_presentations_df["image_index"], omitted_index)
    omitted = stimulus_presentations_df["image_index"] == omitted_index

    stimulus_presentations_df["change"] = changes
    stimulus_presentations_df["omitted"] = omitted

    # Index of each image block
    changes_including_first = np.copy(changes)
    changes_including_first[0] = True
    change_indices = np.flatnonzero(changes_including_first)
    flash_inds = np.arange(len(stimulus_presentations_df))
    block_inds = np.searchsorted(a=change_indices, v=flash_inds, side="right") - 1

    stimulus_presentations_df["block_index"] = block_inds

    # Block repetition number
    blocks_per_image = stimulus_presentations_df.groupby("image_name").apply(
        lambda group: np.unique(group["block_index"])
    )
    block_repetition_number = np.copy(block_inds)

    for image_name, image_blocks in blocks_per_image.iteritems():
        if image_name != "omitted":
            for ind_block, block_number in enumerate(image_blocks):
                # block_rep_number starts as a copy of block_inds, so we can go write over the index number with the rep number
                block_repetition_number[block_repetition_number == block_number] = ind_block

    stimulus_presentations_df["image_block_repetition"] = block_repetition_number

    # Repeat number within a block
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

    # Lists of licks/rewards on each flash
    licks_each_flash = stimulus_presentations_df.apply(
        lambda row: lick_times[
            ((lick_times > row["start_time"]) & (lick_times < row["start_time"] + 0.75))
        ],
        axis=1,
    )
    rewards_each_flash = stimulus_presentations_df.apply(
        lambda row: reward_times[
            (
                (reward_times > row["start_time"])
                & (reward_times < row["start_time"] + 0.75)
            )
        ],
        axis=1,
    )

    stimulus_presentations_df["licks"] = licks_each_flash
    stimulus_presentations_df["rewards"] = rewards_each_flash

    # Average running speed on each flash
    flash_running_speed = stimulus_presentations_df.apply(
        lambda row: trace_average(
            running_speed_df['speed'].values,
            running_speed_df['time'].values,
            row["start_time"],
            row["start_time"] + 0.25, ), axis=1, )
    stimulus_presentations_df["mean_running_speed"] = flash_running_speed

    # Average running speed before each flash
    pre_flash_running_speed = stimulus_presentations_df.apply(
        lambda row: trace_average(
            running_speed_df['speed'].values,
            running_speed_df['time'].values,
            row["start_time"] - 0.25,
            row["start_time"], ), axis=1, )
    stimulus_presentations_df["pre_flash_running_speed"] = pre_flash_running_speed

    if pupil_area is not None:
        # Average running speed on each flash
        flash_pupil_area = stimulus_presentations_df.apply(
            lambda row: trace_average(
                pupil_area['pupil_area'].values,
                pupil_area['time'].values,
                row["start_time"],
                row["start_time"] + 0.25, ), axis=1, )
        stimulus_presentations_df["mean_pupil_area"] = flash_pupil_area

        # Average running speed before each flash
        pre_flash_pupil_area = stimulus_presentations_df.apply(
            lambda row: trace_average(
                pupil_area['pupil_area'].values,
                pupil_area['time'].values,
                row["start_time"] - 0.25,
                row["start_time"], ), axis=1, )
        stimulus_presentations_df["pre_flash_pupil_area"] = pre_flash_pupil_area

    # add flass after omitted
    stimulus_presentations_df['flash_after_omitted'] = np.hstack((False, stimulus_presentations_df.omitted.values[:-1]))
    stimulus_presentations_df['flash_after_change'] = np.hstack((False, stimulus_presentations_df.change.values[:-1]))
    # add licking responses
    stimulus_presentations_df = add_response_latency(stimulus_presentations_df)

    # stimulus_presentations_df = add_inter_flash_lick_diff_to_stimulus_presentations(stimulus_presentations_df)
    # stimulus_presentations_df = add_first_lick_in_bout_to_stimulus_presentations(stimulus_presentations_df)
    # stimulus_presentations_df = get_consumption_licks(stimulus_presentations_df)
    # stimulus_presentations_df = get_metrics(stimulus_presentations_df, licks, rewards)
    stimulus_presentations_df = annotate_flash_rolling_metrics(stimulus_presentations_df)

    return stimulus_presentations_df


def add_window_running_speed(running_speed, stimulus_presentations, response_params):
    window_running_speed = stimulus_presentations.apply(lambda row: trace_average(
        running_speed['speed'].values,
        running_speed['time'].values,
        row["start_time"] + response_params['window_around_timepoint_seconds'][0],
        row["start_time"] + response_params['window_around_timepoint_seconds'][1], ), axis=1, )
    stimulus_presentations["window_running_speed"] = window_running_speed
    return stimulus_presentations
