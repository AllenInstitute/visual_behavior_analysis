import os
import numpy as np
import pandas as pd

from visual_behavior.data_access import loading
from visual_behavior.data_access import utilities
import visual_behavior.ophys.dataset.extended_stimulus_processing as esp


# FUNCTIONS TO REFORMAT DATA LOADED FROM ALLENSDK TO ADDRESS ISSUES WITH MISSING, INCORRECT OR IMPROPERLY STRUCTURED DATA ###

# THESE FUNCTIONS ARE TEMPORARY WORKAROUNDS UNTIL THE ISSUES CAN BE RESOLVED IN THE SDK ###


# REFORMATTING MANIFEST DATA #


def add_mouse_seeks_fail_tags_to_experiments_table(experiments):
    mouse_seeks_report_file_base = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    report_file = 'ophys_session_log_100120.xlsx'
    vb_report_path = os.path.join(mouse_seeks_report_file_base, report_file)
    vb_report_df = pd.read_excel(vb_report_path)

    def clean_columns(columns):
        return [c.lower().replace(' ', '_') for c in columns]

    vb_report_df.columns = clean_columns(vb_report_df.columns)
    vb_report_df = vb_report_df.rename(columns={'session_id': 'ophys_session_id'})
    # merge fail tags into all_experiments manifest
    experiments = experiments.merge(vb_report_df[['ophys_session_id', 'session_tags', 'failure_tags']],
                                    right_on='ophys_session_id', left_on='ophys_session_id')
    return experiments


def add_location_to_expts(expts):
    expts['location'] = [expts.loc[x]['cre_line'].split('-')[0] + '_' + expts.loc[x]['targeted_structure'] + '_' + str(
        int(expts.loc[x]['imaging_depth'])) for x in expts.index.values]
    return expts


def get_exposure_number_for_group(group):
    order = np.argsort(group['date_of_acquisition'].values)
    group['exposure_number'] = order
    return group


def add_exposure_number_to_experiments_table(experiments):
    experiments = experiments.groupby(['super_container_id', 'container_id', 'session_type']).apply(
        get_exposure_number_for_group)
    return experiments


def add_model_outputs_availability_to_table(table):
    """
    Evaluates whether model output files are available for each experiment/session in the table
        Requires that the table has a column 'behavior_session_id' with 9-digit behavior session identifiers
    :param table: table of experiment or session level metadata (can be experiments_table or ophys_sessions_table)
    :return: table with added column 'model_outputs_available', values are Boolean
    """
    table['model_outputs_available'] = [utilities.model_outputs_available_for_behavior_session(behavior_session_id)
                                        for behavior_session_id in table.behavior_session_id.values]
    return table


def reformat_experiments_table(experiments):
    experiments = experiments.reset_index()
    experiments['super_container_id'] = experiments['specimen_id'].values
    # clean up cre_line naming
    experiments['cre_line'] = [driver_line[1] if driver_line[0] == 'Camk2a-tTA' else driver_line[0] for driver_line in
                               experiments.driver_line.values]
    experiments = experiments[experiments.cre_line != 'Cux2-CreERT2']  # why is this here?
    # replace session types that are NaN with string None
    experiments.at[experiments[experiments.session_type.isnull()].index.values, 'session_type'] = 'None'
    experiments = add_mouse_seeks_fail_tags_to_experiments_table(experiments)
    experiments = add_exposure_number_to_experiments_table(experiments)
    experiments = add_model_outputs_availability_to_table(experiments)
    if 'level_0' in experiments.columns:
        experiments = experiments.drop(columns='level_0')
    if 'index' in experiments.columns:
        experiments = experiments.drop(columns='index')
    experiments = add_location_to_expts(experiments)
    return experiments


def add_all_qc_states_to_ophys_session_table(session_table):
    """ Add 'experiment_workflow_state', 'container_workflow_state', and 'session_workflow_state' to session_table.
            :param session_table: session_table from SDK cache
            :return: session_table: with additional columns added
            """
    experiment_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
    session_table = add_session_workflow_state_to_ophys_session_table(session_table, experiment_table)
    session_table = add_container_workflow_state_to_ophys_session_table(session_table, experiment_table)
    return session_table


def add_session_workflow_state_to_ophys_session_table(session_table, experiment_table):
    """
    Define session_workflow_state as 'passing' if at least one of the experiments from that session passed QC.
    If all experiments failed, it is likely a session level failure cause, such as abehavior failure.
    :param session_table: session_table from SDK cache
    :return: session_table: with additional column for session_workflow_state added
    """
    passed_experiments = experiment_table[experiment_table.experiment_workflow_state == 'passed'].copy()
    session_ids = session_table.index.values
    session_table['at_least_one_experiment_passed'] = [any(passed_experiments['ophys_session_id'] == x) for x in
                                                       session_ids]
    session_table['session_workflow_state'] = ['passed' if criterion == True else 'failed' for criterion in
                                               session_table.at_least_one_experiment_passed.values]
    return session_table


def add_container_workflow_state_to_ophys_session_table(session_table, experiment_table):
    """
        Add 'container_workflow_state' to session_table by merging with experiment_table
        :param session_table: session_table from SDK cache
        :return: session_table: with additional column for container_workflow_state added
        """
    session_table = session_table.reset_index()
    session_table = session_table[session_table.ophys_session_id.isin(experiment_table.ophys_session_id.unique())]
    experiments = experiment_table[['ophys_session_id', 'container_id', 'container_workflow_state']].drop_duplicates(
        ['ophys_session_id'])
    session_table = session_table.merge(experiments, left_on='ophys_session_id', right_on='ophys_session_id')
    session_table = session_table.set_index(keys='ophys_session_id')
    return session_table


# REFORMATING SDK SESSION OBJECT DATA #


# This section contains functions to reformat sdk session attributes to conform with our
# design decisions. If we can get some of these changes backported into the SDK, then they
# can be removed from this module.

def add_trial_type_to_trials_table(trials):
    trials['trial_type'] = None
    trials.loc[trials[trials.auto_rewarded].index, 'trial_type'] = 'auto_rewarded'
    trials.loc[trials[trials.hit].index, 'trial_type'] = 'hit'
    trials.loc[trials[trials.miss].index, 'trial_type'] = 'miss'
    trials.loc[trials[trials.correct_reject].index, 'trial_type'] = 'correct_reject'
    trials.loc[trials[trials.false_alarm].index, 'trial_type'] = 'false_alarm'
    return trials


def convert_metadata_to_dataframe(original_metadata):
    metadata = original_metadata.copy()
    metadata['reporter_line'] = metadata['reporter_line'][0]
    if len(metadata['driver_line']) > 1:
        index = [i for i, driver_line in enumerate(metadata['driver_line']) if 'Camk2a-tTA' in driver_line][0]
        metadata['driver_line'] = metadata['driver_line'][index]
    else:
        metadata['driver_line'] = metadata['driver_line'][0]
    return pd.DataFrame(metadata, index=[metadata['ophys_experiment_id']])


def convert_licks(licks_df):
    '''
    licks has a column called 'time', inconsistent with the rest of the sdk.
    should be 'timestamps'

    ARGS: licks_df, from the SDK
    RETURNS: licks_df
    MODIFIES: licks_df, renaming column 'time' to 'timestamps'

    WARNING. session.trials will not load after this function is called. Therefore, before this
        function is called, you need to run session.trials and then it will be cached and will work.
        This is done for you in sdk_utils.add_stimulus_presentations_analysis.
    '''
    assert 'time' in licks_df.columns
    return licks_df.rename(columns={'time': 'timestamps'})


def convert_rewards(rewards_df):
    '''
    rewards has timestamps as the index, inconsistent with the rest of the sdk.
    should have a column called 'timestamps' instead.

    ARGS: rewards_df, from the SDK
    RETURNS: rewards_df
    MODIFIES: rewards_df, moving 'timestamps' to a column, not index

    WARNING. session.trials will not load after this function is called. Therefore, before this
        function is called, you need to run session.trials and then it will be cached and will work.
        This is done for you in sdk_utils.add_stimulus_presentations_analysis.
    '''
    assert rewards_df.index.name == 'timestamps'
    return rewards_df.reset_index()


def convert_running_speed(running_speed_obj):
    '''
    running speed is returned as a custom object, inconsistent with other attrs.
    should be a dataframe with cols for timestamps and speed.

    ARGS: running_speed object
    RETURNS: dataframe with columns timestamps and speed
    '''
    return pd.DataFrame({
        'timestamps': running_speed_obj.timestamps,
        'speed': running_speed_obj.values
    })


def add_change_each_flash(stimulus_presentations):
    '''
        Adds a column to stimulus_presentations, ['change'], which is True if the stimulus was a change image, and False otherwise

        ARGS: stimulus_presentations dataframe
        RETURNS: modified stimulus_presentations dataframe
    '''
    changes = esp.find_change(stimulus_presentations['image_index'], esp.get_omitted_index(stimulus_presentations))
    stimulus_presentations['change'] = changes
    return stimulus_presentations


def add_mean_running_speed(stimulus_presentations, running_speed, range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the mean running speed between

    Args:
        stimulus_presentations(pd.DataFrame): dataframe of stimulus presentations.
                Must contain: 'start_time'
        running_speed (pd.DataFrame): dataframe of running speed.
            Must contain: 'speed', 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        nothing, modifies session in place. Same as the input, but with 'mean_running_speed' column added
    '''
    if isinstance(running_speed, pd.DataFrame):
        mean_running_speed_df = esp.mean_running_speed(stimulus_presentations,
                                                       running_speed,
                                                       range_relative_to_stimulus_start)
    else:
        mean_running_speed_df = esp.mean_running_speed(stimulus_presentations,
                                                       convert_running_speed(running_speed),
                                                       range_relative_to_stimulus_start)

    stimulus_presentations["mean_running_speed"] = mean_running_speed_df
    return stimulus_presentations


def add_licks_each_flash(stimulus_presentations, licks, range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of licks that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        licks (pd.DataFrame): lick dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        nothing, modifies session in place. Same as the input, but with 'licks' column added
    '''

    licks_each_flash = esp.licks_each_flash(stimulus_presentations,
                                            licks,
                                            range_relative_to_stimulus_start)
    stimulus_presentations['licks'] = licks_each_flash
    return stimulus_presentations


def add_rewards_each_flash(stimulus_presentations, rewards, range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the timestamps of rewards that occur
    in a range relative to the onset of the stimulus.

    Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        rewards (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
    Returns:
        nothing. session.stimulus_presentations is modified in place with 'rewards' column added
    '''

    rewards_each_flash = esp.rewards_each_flash(stimulus_presentations,
                                                rewards,
                                                range_relative_to_stimulus_start)
    stimulus_presentations['rewards'] = rewards_each_flash
    return stimulus_presentations


def add_time_from_last_lick(stimulus_presentations, licks):
    '''
        Adds a column in place to session.stimulus_presentations['time_from_last_lick'], which is the time, in seconds
        since the last lick

        Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        licks (pd.DataFrame): lick dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.

        Returns:
            modified stimulus_presentations table
    '''
    lick_times = licks['timestamps'].values
    flash_times = stimulus_presentations["start_time"].values
    if len(lick_times) < 5:  # Passive sessions
        time_from_last_lick = np.full(len(flash_times), np.nan)
    else:
        time_from_last_lick = esp.time_from_last(flash_times, lick_times)
    stimulus_presentations["time_from_last_lick"] = time_from_last_lick
    return stimulus_presentations


def add_time_from_last_reward(stimulus_presentations, rewards):
    '''
        Adds a column to stimulus_presentations['time_from_last_reward'], which is the time, in seconds
        since the last reward

        Args:
        stimulus_presentations (pd.DataFrame): dataframe of stimulus presentations.
            Must contain: 'start_time'
        rewards (pd.DataFrame): rewards dataframe. Must contain 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the running speed.
        Returns:
            modified stimulus_presentations table
    '''
    reward_times = rewards['timestamps'].values
    flash_times = stimulus_presentations["start_time"].values

    if len(reward_times) < 1:  # Sometimes mice are bad
        time_from_last_reward = np.full(len(flash_times), np.nan)
    else:
        time_from_last_reward = esp.time_from_last(flash_times, reward_times)
    stimulus_presentations["time_from_last_reward"] = time_from_last_reward
    return stimulus_presentations


def add_time_from_last_change(stimulus_presentations):
    '''
        Adds a column to session.stimulus_presentations, 'time_from_last_change', which is the time, in seconds
        since the last image change

        ARGS: SDK session object
        MODIFIES: session.stimulus_presentations
        RETURNS: stimulus_presentations
    '''
    flash_times = stimulus_presentations["start_time"].values
    change_times = stimulus_presentations.query('change')['start_time'].values
    time_from_last_change = esp.time_from_last(flash_times, change_times)
    stimulus_presentations["time_from_last_change"] = time_from_last_change
    return stimulus_presentations


def add_time_from_last_omission(stimulus_presentations):
    '''
        Adds a column to session.stimulus_presentations, 'time_from_last_omission', which is the time, in seconds
        since the last stimulus omission

        ARGS: SDK session object
        MODIFIES: session.stimulus_presentations
        RETURNS: stimulus_presentations
    '''
    flash_times = stimulus_presentations["start_time"].values
    omission_times = stimulus_presentations.query('omitted')['start_time'].values
    time_from_last_omission = esp.time_from_last(flash_times, omission_times, side='left')
    stimulus_presentations["time_from_last_omission"] = time_from_last_omission
    return stimulus_presentations


def add_epoch_times(df, time_column='start_time', epoch_duration_mins=10):
    """
    Add column called 'epoch' with values as an index for the epoch within a session, for a given epoch duration.

    :param df: dataframe with a column indicating event start times. Can be stimulus_presentations or trials table.
    :param time_column: name of column in dataframe indicating event times
    :param epoch_duration_mins: desired epoch length in minutes
    :return: input dataframe with epoch column added
    """
    start_time = df[time_column].values[0]
    stop_time = df[time_column].values[-1]
    epoch_times = np.arange(start_time, stop_time, epoch_duration_mins * 60)
    df['epoch'] = None
    for i, time in enumerate(epoch_times):
        if i < len(epoch_times) - 1:
            indices = df[(df[time_column] >= epoch_times[i]) & (df[time_column] < epoch_times[i + 1])].index.values
        else:
            indices = df[(df[time_column] >= epoch_times[i])].index.values
        df.at[indices, 'epoch'] = i
    return df


def get_tidy_dff_traces(dataset):
    '''
    returns a tidy formatted version of the dff_traces dataframe
    '''
    dff_dfs = []
    for csid in dataset.cell_specimen_ids:
        dff_dfs.append(
            pd.DataFrame({
                'timestamps': dataset.ophys_timestamps,
                'cell_specimen_id': [csid] * len(dataset.ophys_timestamps),
                'dff': dataset.dff_traces.loc[csid]['dff'],
            })
        )
    return pd.concat(dff_dfs)

# INPLACE VERSIONS ###


def convert_licks_inplace(licks_df):
    '''
    licks has a column called 'time', inconsistent with the rest of the sdk.
    should be 'timestamps'

    ARGS: licks_df, from the SDK
    RETURNS: nothing
    MODIFIES: licks_df, renaming column 'time' to 'timestamps'
    '''
    assert 'time' in licks_df.columns
    licks_df.rename(columns={'time': 'timestamps'}, inplace=True)


def convert_rewards_inplace(rewards_df):
    '''
    rewards has timestamps as the index, inconsistent with the rest of the sdk.
    should have a column called 'timestamps' instead.

    ARGS: rewards_df, from the SDK
    RETURNS: nothing
    MODIFIES: rewards_df, moving 'timestamps' to a column, not index
    '''
    assert rewards_df.index.name == 'timestamps'
    rewards_df.reset_index(inplace=True)


def add_change_each_flash_inplace(session):
    '''
        Adds a column to session.stimulus_presentations, ['change'], which is True if the stimulus was a change image, and False otherwise

        ARGS: SDK session object
        RETURNS: nothing
        MODIFIES: session.stimulus_presentations dataframe
    '''
    changes = esp.find_change(session.stimulus_presentations['image_index'],
                              esp.get_omitted_index(session.stimulus_presentations))
    session.stimulus_presentations['change'] = changes


def add_mean_running_speed_inplace(session, range_relative_to_stimulus_start=[0, 0.75]):
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
    if isinstance(session.running_speed, pd.DataFrame):
        mean_running_speed_df = esp.mean_running_speed(session.stimulus_presentations,
                                                       session.running_speed,
                                                       range_relative_to_stimulus_start)
    else:
        mean_running_speed_df = esp.mean_running_speed(session.stimulus_presentations,
                                                       convert_running_speed(session.running_speed),
                                                       range_relative_to_stimulus_start)

    session.stimulus_presentations["mean_running_speed"] = mean_running_speed_df


def add_licks_each_flash_inplace(session, range_relative_to_stimulus_start=[0, 0.75]):
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

    licks_each_flash = esp.licks_each_flash(session.stimulus_presentations,
                                            session.licks,
                                            range_relative_to_stimulus_start)
    session.stimulus_presentations['licks'] = licks_each_flash


def add_rewards_each_flash_inplace(session, range_relative_to_stimulus_start=[0, 0.75]):
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

    rewards_each_flash = esp.rewards_each_flash(session.stimulus_presentations,
                                                session.rewards,
                                                range_relative_to_stimulus_start)
    session.stimulus_presentations['rewards'] = rewards_each_flash


def add_time_from_last_lick_inplace(session):
    '''
        Adds a column in place to session.stimulus_presentations['time_from_last_lick'], which is the time, in seconds
        since the last lick

        ARGS: SDK session object
        MODIFIES: session.stimulus_presentations
        RETURNS: nothing
    '''
    lick_times = session.licks['timestamps'].values
    flash_times = session.stimulus_presentations["start_time"].values
    if len(lick_times) < 5:  # Passive sessions
        time_from_last_lick = np.full(len(flash_times), np.nan)
    else:
        time_from_last_lick = esp.time_from_last(flash_times, lick_times)
    session.stimulus_presentations["time_from_last_lick"] = time_from_last_lick


def add_time_from_last_reward_inplace(session):
    '''
        Adds a column in place to session.stimulus_presentations['time_from_last_reward'], which is the time, in seconds
        since the last reward

        ARGS: SDK session object
        MODIFIES: session.stimulus_presentations
        RETURNS: nothing
    '''
    reward_times = session.rewards['timestamps'].values
    flash_times = session.stimulus_presentations["start_time"].values

    if len(reward_times) < 1:  # Sometimes mice are bad
        time_from_last_reward = np.full(len(flash_times), np.nan)
    else:
        time_from_last_reward = esp.time_from_last(flash_times, reward_times)
    session.stimulus_presentations["time_from_last_reward"] = time_from_last_reward


def add_time_from_last_change_inplace(session):
    '''
        Adds a column to session.stimulus_presentations, 'time_from_last_change', which is the time, in seconds
        since the last image change

        ARGS: SDK session object
        MODIFIES: session.stimulus_presentations
        RETURNS: nothing
    '''
    flash_times = session.stimulus_presentations["start_time"].values
    change_times = session.stimulus_presentations.query('change')['start_time'].values
    time_from_last_change = esp.time_from_last(flash_times, change_times)
    session.stimulus_presentations["time_from_last_change"] = time_from_last_change


def filter_invalid_rois_inplace(session):
    '''
    Remove invalid ROIs from the cell specimen table, dff_traces, and corrected_fluorescence_traces.

    Args:
        session (allensdk.brain_observatory.behavior_ophys_session.BehaviorOphysSession): The session to filter.

    Returns nothing, modifies the session object inplace.
    '''
    invalid_cell_specimen_ids = session.cell_specimen_table.query('valid_roi == False').index.values

    # Drop stuff
    session.dff_traces.drop(index=invalid_cell_specimen_ids, inplace=True)
    session.corrected_fluorescence_traces.drop(index=invalid_cell_specimen_ids, inplace=True)
    session.cell_specimen_table.drop(index=invalid_cell_specimen_ids, inplace=True)


def annotate_licks(licks, rewards, bout_threshold=0.7):
    '''
        Appends several columns to dataset.licks. Calculates licking bouts based on a
        interlick interval (ILI) of bout_threshold. Default of 700ms based on examining
        histograms of ILI distributions
        Adds to dataset.licks
            pre_ili,        (seconds)
            post_ili,       (seconds)
            rewarded,       (boolean)
            bout_start,     (boolean)
            bout_end,       (boolean)
            bout_number,    (int)
            bout_rewarded,  (boolean)
    '''

    # Computing ILI for each lick
    licks['pre_ili'] = np.concatenate([[np.nan], np.diff(licks.timestamps.values)])
    licks['post_ili'] = np.concatenate([np.diff(licks.timestamps.values), [np.nan]])
    licks['rewarded'] = False
    for index, row in rewards.iterrows():
        if len(np.where(licks.timestamps <= row.timestamps)[0]) == 0:
            if (row.autorewarded) & (row.timestamps <= licks.timestamps.values[0]):
                # mouse hadn't licked before first auto-reward
                mylick = 0
            else:
                print(
                    'First lick was after first reward, but it wasnt an auto-reward. This is very strange, but Im annotating the first lick as rewarded.')
                mylick = 0
        else:
            mylick = np.where(licks.timestamps <= row.timestamps)[0][-1]
        licks.at[mylick, 'rewarded'] = True

    # Segment licking bouts
    licks['bout_start'] = licks['pre_ili'] > bout_threshold
    licks['bout_end'] = licks['post_ili'] > bout_threshold
    licks.at[licks['pre_ili'].apply(np.isnan), 'bout_start'] = True
    licks.at[licks['post_ili'].apply(np.isnan), 'bout_end'] = True

    # Annotate bouts by number, and reward
    licks['bout_number'] = np.cumsum(licks['bout_start'])
    x = licks.groupby('bout_number').any('rewarded').rename(columns={'rewarded': 'bout_rewarded'})
    licks['bout_rewarded'] = False
    temp = licks.reset_index().set_index('bout_number')
    temp.update(x)
    temp = temp.reset_index().set_index('index')
    licks['bout_rewarded'] = temp['bout_rewarded']
    return licks


def annotate_bouts(stimulus_presentations, licks):
    '''
        Uses the bout annotations in licks to annotate stimulus_presentations
        Adds to stimulus_presentations
            bout_start,     (boolean)
            bout_end,       (boolean)
    '''
    # Annotate Bout Starts
    bout_starts = licks[licks['bout_start']]
    stimulus_presentations['bout_start'] = False
    stimulus_presentations['num_bout_start'] = 0
    for index, x in bout_starts.iterrows():
        filter_start = stimulus_presentations[stimulus_presentations['start_time'].gt(x.timestamps)]
        if (x.timestamps > stimulus_presentations.iloc[0].start_time) & (len(filter_start) > 0):
            stimulus_presentations.at[
                stimulus_presentations[stimulus_presentations['start_time'].gt(x.timestamps)].index[
                    0] - 1, 'bout_start'] = True
            stimulus_presentations.at[
                stimulus_presentations[stimulus_presentations['start_time'].gt(x.timestamps)].index[
                    0] - 1, 'num_bout_start'] += 1
    # Annotate Bout Ends
    bout_ends = licks[licks['bout_end']]
    stimulus_presentations['bout_end'] = False
    stimulus_presentations['num_bout_end'] = 0
    for index, x in bout_ends.iterrows():
        filter_start = stimulus_presentations[stimulus_presentations['start_time'].gt(x.timestamps)]
        if (x.timestamps > stimulus_presentations.iloc[0].start_time) & (len(filter_start) > 0):
            stimulus_presentations.at[
                stimulus_presentations[stimulus_presentations['start_time'].gt(x.timestamps)].index[
                    0] - 1, 'bout_end'] = True
            stimulus_presentations.at[
                stimulus_presentations[stimulus_presentations['start_time'].gt(x.timestamps)].index[
                    0] - 1, 'num_bout_end'] += 1
            # Check to see if bout started before stimulus, if so, make first flash as bout_starts
            bout_start_time = \
                licks.query('bout_number == @x.bout_number').query('bout_start').timestamps.values[0]
            bout_end_time = x.timestamps
            if (bout_start_time < stimulus_presentations.iloc[0].start_time) & (
                bout_end_time > stimulus_presentations.iloc[0].start_time):
                stimulus_presentations.at[0, 'bout_start'] = True
                stimulus_presentations.at[0, 'num_bout_start'] += 1
    # Clean Up
    stimulus_presentations.drop(-1, inplace=True, errors='ignore')

    return stimulus_presentations


def annotate_bout_start_time(stimulus_presentations):
    stimulus_presentations['bout_start_time'] = np.nan
    stimulus_presentations.at[stimulus_presentations['bout_start'] == True, 'bout_start_time'] = \
        stimulus_presentations[stimulus_presentations['bout_start'] == True].licks.str[0]
    return stimulus_presentations


def add_rolling_metrics_to_stimulus_presentations(stimulus_presentations, licks, rewards, win_dur=320, win_type='triang'):
    '''
        Get rolling flash level metrics for lick rate, reward rate, and bout_rate
        Computes over a rolling window of win_dur (s) duration, with a window type given by win_type
        Adds to stimulus_presentations
            licked,         (boolean)
            lick_rate,      (licks/flash)
            rewarded,       (boolean)
            reward_rate,    (rewards/flash)
            running_rate,   (cm/s)
            bout_rate,      (bouts/flash)
    '''
    from scipy.stats import norm

    # Annotate licks & add lick bout info to stimulus presentatinos
    # licks = annotate_licks(licks, rewards, bout_threshold=0.7)
    # stimulus_presentations = annotate_bouts(stimulus_presentations, licks)
    # stimulus_presentations = annotate_bout_start_time(stimulus_presentations)

    # # Get Lick Rate / second
    # stimulus_presentations['licked'] = [1 if len(this_lick) > 0 else 0 for this_lick in
    #                                             stimulus_presentations['licks']]
    # stimulus_presentations['lick_rate'] = stimulus_presentations['licked'].rolling(win_dur, min_periods=1, win_type=win_type).mean() / .75
    #
    # # Get Reward Rate / second
    # stimulus_presentations['rewarded'] = [1 if len(this_reward) > 0 else 0 for this_reward in
    #                                               stimulus_presentations['rewards']]
    # stimulus_presentations['reward_rate'] = stimulus_presentations['rewarded'].rolling(win_dur, min_periods=1, win_type=win_type).mean() / .75

#     # Get Running / Second
#     stimulus_presentations['running_rate'] = stimulus_presentations['mean_running_speed'].rolling(
#             win_dur, min_periods=1, win_type=win_type).mean() / .75
#
#     # Get Bout Rate / second
#     stimulus_presentations['bout_rate'] = stimulus_presentations['bout_start'].rolling(win_dur, min_periods=1, win_type=win_type).mean() / .75
#
#     # Get Hit Fraction. % of licks that are rewarded
#     stimulus_presentations['hit_bout'] = [np.nan if (not x[0]) else 1 if (x[1] == 1) else 0 for x in
#                                                   zip(stimulus_presentations['bout_start'],
#                                                       stimulus_presentations['rewarded'])]
#     stimulus_presentations['hit_fraction'] = stimulus_presentations['hit_bout'].rolling(win_dur, min_periods=1, win_type=win_type).mean().fillna( 0)
#
#     # Get Hit Rate, % of change flashes with licks
#     stimulus_presentations['change_with_lick'] = [np.nan if (not x[0]) else 1 if (x[1]) else 0 for x in
#                                                           zip(stimulus_presentations['change'],
#                                                               stimulus_presentations['bout_start'])]
#     stimulus_presentations['hit_rate'] = stimulus_presentations['change_with_lick'].rolling(win_dur, min_periods=1, win_type=win_type).mean().fillna(0)
#
#     # Get Miss Rate, % of change flashes without licks
#     stimulus_presentations['change_without_lick'] = [np.nan if (not x[0]) else 0 if (x[1]) else 1 for x in
#                                                              zip(stimulus_presentations['change'],
#                                                                  stimulus_presentations['bout_start'])]
#     stimulus_presentations['miss_rate'] = stimulus_presentations['change_without_lick'].rolling(win_dur, min_periods=1, win_type=win_type).mean().fillna(0)
#
#     # Get False Alarm Rate, % of non-change flashes with licks
#     stimulus_presentations['non_change_with_lick'] = [np.nan if (x[0]) else 1 if (x[1]) else 0 for x in
#                                                               zip(stimulus_presentations['change'],
#                                                                   stimulus_presentations['bout_start'])]
#     stimulus_presentations['false_alarm_rate'] = stimulus_presentations['non_change_with_lick'].rolling(
#         win_dur, min_periods=1, win_type=win_type).mean().fillna(0)
#
#     # Get Correct Reject Rate, % of non-change flashes without licks
#     stimulus_presentations['non_change_without_lick'] = [np.nan if (x[0]) else 0 if (x[1]) else 1 for x in
#                                                                  zip(stimulus_presentations['change'],
#                                                                      stimulus_presentations['bout_start'])]
#     stimulus_presentations['correct_reject_rate'] = stimulus_presentations[
#         'non_change_without_lick'].rolling(win_dur, min_periods=1, win_type=win_type).mean().fillna(0)
#
#     # Get dPrime and Criterion metrics on a flash level
#     Z = norm.ppf
#     stimulus_presentations['d_prime'] = Z(np.clip(stimulus_presentations['hit_rate'], 0.01, 0.99)) - Z(
#         np.clip(stimulus_presentations['false_alarm_rate'], 0.01, 0.99))
#     stimulus_presentations['criterion'] = 0.5 * (Z(np.clip(stimulus_presentations['hit_rate'], 0.01, 0.99)) + Z(
#         np.clip(stimulus_presentations['false_alarm_rate'], 0.01, 0.99)))
#     # Computing the criterion to be negative
#
#     return stimulus_presentations
#
#
# def classify_rolling_metrics(stimulus_presentations, lick_threshold=0.1, reward_threshold=2 / 80, use_bouts=True):
#     '''
#         Use the flash level rolling metrics to classify into three states based on the thresholds
#         lick_threshold is the licking rate / flash that divides high and low licking states
#         reward_threshold is the rewards/flash that divides high and low reward states (2/80 is equivalent to 2 rewards/minute).
#     '''
#     if use_bouts:
#         stimulus_presentations['high_lick'] = [True if x > lick_threshold else False for x in
#                                                        stimulus_presentations['bout_rate']]
#     else:
#         stimulus_presentations['high_lick'] = [True if x > lick_threshold else False for x in
#                                                        stimulus_presentations['lick_rate']]
#     stimulus_presentations['high_reward'] = [True if x > reward_threshold else False for x in
#                                                      stimulus_presentations['reward_rate']]
#     stimulus_presentations['flash_metrics_epochs'] = [0 if (not x[0]) & (not x[1]) else 1 if x[1] else 2 for x
#                                                               in zip(stimulus_presentations['high_lick'],
#                                                                      stimulus_presentations['high_reward'])]
#     stimulus_presentations['flash_metrics_labels'] = [
#         'low-lick,low-reward' if x == 0  else 'high-lick,high-reward' if x == 1 else 'high-lick,low-reward' for x in
#         stimulus_presentations['flash_metrics_epochs']]

    return stimulus_presentations


def add_model_outputs_to_stimulus_presentations(stimulus_presentations, behavior_session_id):
    '''
       Adds additional columns to stimulus table for model weights and related metrics
    '''

    if loading.check_if_model_output_available(behavior_session_id):
        model_outputs = pd.read_csv(
            os.path.join(loading.get_behavior_model_outputs_dir(), loading.get_model_output_file(behavior_session_id)[0]))
        model_outputs = model_outputs[['stimulus_presentations_id', 'omissions1', 'task0', 'timing1D', 'bias']]
        stimulus_presentations = stimulus_presentations.merge(model_outputs, right_on='stimulus_presentations_id',
            left_on='stimulus_presentations_id').set_index('stimulus_presentations_id')
        return stimulus_presentations
    else:
        print('no model outputs saved for behavior_session_id:', behavior_session_id)
    return stimulus_presentations
