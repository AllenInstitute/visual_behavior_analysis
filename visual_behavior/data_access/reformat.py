import os
import glob
import numpy as np
import pandas as pd

import visual_behavior.ophys.dataset.extended_stimulus_processing as esp


# FUNCTIONS TO REFORMAT DATA LOADED FROM ALLENSDK TO ADDRESS ISSUES WITH MISSING, INCORRECT OR IMPROPERLY STRUCTURED DATA ###

# THESE FUNCTIONS ARE TEMPORARY WORKAROUNDS UNTIL THE ISSUES CAN BE RESOLVED IN THE SDK ###


# REFORMATTING MANIFEST DATA #


def add_mouse_seeks_fail_tags_to_experiments_table(experiments):
    mouse_seeks_report_file_base = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/ophys_session_logs'
    report_file = glob.glob(os.path.join(mouse_seeks_report_file_base, 'ophys_session_log_*.xlsx'))[-1]
    # report_file = 'ophys_session_log_080121.xlsx'
    vb_report_path = os.path.join(mouse_seeks_report_file_base, report_file)
    vb_report_df = pd.read_excel(vb_report_path)

    def clean_columns(columns):
        return [c.lower().replace(' ', '_') for c in columns]

    vb_report_df.columns = clean_columns(vb_report_df.columns)
    vb_report_df = vb_report_df.rename(columns={'session_id': 'ophys_session_id'})
    # merge fail tags into all_experiments manifest
    experiments = experiments.reset_index()
    experiments = experiments.merge(vb_report_df[['ophys_session_id', 'session_tags', 'failure_tags']],
                                    right_on='ophys_session_id', left_on='ophys_session_id')
    experiments = experiments.set_index('ophys_experiment_id')
    return experiments


def add_location_to_expts(expts):
    expts['location'] = [expts.loc[x]['cre_line'].split('-')[0] + '_' + expts.loc[x]['targeted_structure'] + '_' + str(
        int(expts.loc[x]['imaging_depth'])) for x in expts.index.values]
    return expts


def get_exposure_number_for_group(group):
    order = np.argsort(group['date_of_acquisition'].values)
    group['prior_exposures_to_session_type'] = order
    return group


def add_session_type_exposure_number_to_experiments_table(experiments):
    experiments = experiments.groupby(['mouse_id', 'ophys_container_id', 'session_type']).apply(
        get_exposure_number_for_group)
    return experiments


def add_reward_rate_to_trials_table(trials, extended_stimulus_presentations):
    '''
    adds 'reward_rate' field to the trial table from extended_stimulus_presentations
    pulled from the value of the stimulus table that is closest to the time at the start of each trial
    '''
    extended_stimulus_presentations = extended_stimulus_presentations

    # for each trial, find the stimulus index that is closest to the trial start
    # add to a new column called 'first_stim_presentation_index'
    for idx, trial in trials.iterrows():
        start_time = trial['start_time']
        query_string = 'start_time > @start_time - 1 and start_time < @start_time + 1'
        first_stim_presentation_index = (np.abs(start_time - extended_stimulus_presentations.query(query_string)['start_time'])).idxmin()
        trials.at[idx, 'first_stim_presentation_index'] = first_stim_presentation_index

    # define the columns from extended_stimulus_presentations that we want to merge into trials
    cols_to_merge = [
        'reward_rate',
    ]

    # merge the desired columns into trials on the stimulus_presentations_id indices
    trials = trials.merge(
        extended_stimulus_presentations[cols_to_merge].reset_index(),
        left_on='first_stim_presentation_index',
        right_on='stimulus_presentations_id',
    )

    return trials


def add_engagement_state_to_trials_table(trials, extended_stimulus_presentations):
    '''
    adds `engaged` and `engagement_state` fields to the trial table
    both are pulled from the value of the stimulus table that is closest to the time at the start of each trial
    '''
    extended_stimulus_presentations = extended_stimulus_presentations

    # for each trial, find the stimulus index that is closest to the trial start
    # add to a new column called 'first_stim_presentation_index'
    for idx, trial in trials.iterrows():
        start_time = trial['start_time']
        query_string = 'start_time > @start_time - 1 and start_time < @start_time + 1'
        first_stim_presentation_index = (np.abs(start_time - extended_stimulus_presentations.query(query_string)['start_time'])).idxmin()
        trials.at[idx, 'first_stim_presentation_index'] = first_stim_presentation_index

    # define the columns from extended_stimulus_presentations that we want to merge into trials
    cols_to_merge = [
        'engaged',
        'engagement_state'
    ]

    # merge the desired columns into trials on the stimulus_presentations_id indices
    trials = trials.merge(
        extended_stimulus_presentations[cols_to_merge].reset_index(),
        left_on='first_stim_presentation_index',
        right_on='stimulus_presentations_id',
    )

    return trials


def get_image_set_exposures_for_behavior_session_id(behavior_session_id, behavior_session_table):
    """
    Gets the number of sessions an image set has been presented in prior to the date of the given behavior_session_id
    :param behavior_session_id:
    :return:
    """
    behavior_session_table = behavior_session_table[behavior_session_table.session_type.isnull() == False]  # FIX THIS - SHOULD NOT BE ANY NaNs!
    mouse_id = behavior_session_table.loc[behavior_session_id].mouse_id
    session_type = behavior_session_table.loc[behavior_session_id].session_type
    image_set = session_type.split('_')[3]
    date = behavior_session_table.loc[behavior_session_id].date_of_acquisition
    # check how many behavior sessions prior to this date had the same image set
    cdf = behavior_session_table[(behavior_session_table.mouse_id == mouse_id)].copy()
    pre_expts = cdf[(cdf.date_of_acquisition < date)]
    image_set_exposures = int(len([session_type for session_type in pre_expts.session_type if 'images_' + image_set in session_type]))
    return image_set_exposures


def add_image_set_exposure_number_to_experiments_table(experiments, behavior_session_table):
    exposures = []
    for row in range(len(experiments)):
        try:
            behavior_session_id = experiments.iloc[row].behavior_session_id
            image_set_exposures = get_image_set_exposures_for_behavior_session_id(behavior_session_id, behavior_session_table)
            exposures.append(image_set_exposures)
        except Exception:
            exposures.append(np.nan)
    experiments['prior_exposures_to_image_set'] = exposures
    return experiments


def get_omission_exposures_for_behavior_session_id(behavior_session_id, behavior_session_table):
    """
    Gets the number of sessions that had omitted stimuli prior to the date of the given behavior_session_id
    Note: Omitted flashes were accidentally included in OPHYS_0_images_X_habituation prior to Feb 14, 2019
    This commit to mtrain_regiments removed omissions from habituation sessions:
        https://github.com/AllenInstitute/mtrain_regimens/commits/7ee1da717a4445cc6418fc91dda4623b9958e7a0
    A fix was pushed to the ophys rigs to implement this change sometime around Feb 14, 2019 but it is unknown exactly when.
    *** TO DO: determine exact mtrain commit / version date for each session to determine whether omissions
        were incuded in OPHYS_0_images_X_habituation) ***
    :param behavior_session_id:
    :return: The number of behavior sessions where omitted flashes were present, prior to the current session
    """
    behavior_session_table = behavior_session_table[behavior_session_table.session_type.isnull() == False]  # FIX THIS - SHOULD NOT BE ANY NaNs!
    mouse_id = behavior_session_table.loc[behavior_session_id].mouse_id
    date = behavior_session_table.loc[behavior_session_id].date_of_acquisition
    # check how many behavior sessions prior to this date had the same image set
    cdf = behavior_session_table[(behavior_session_table.mouse_id == mouse_id)].copy()
    pre_expts = cdf[(cdf.date_of_acquisition < date)]
    # check how many behavior sessions prior to this date had omissions
    import datetime
    # THIS IS A HACK, NEED TO REPLACE THIS WITH REFERENCE TO MTRAIN REGIMENS COMMIT HASH FOR WHEN THE CHANGE TO
    # REMOVE OMISSIONS FROM HABITUATION SESSIONS OCCURED
    date_of_change = 'Feb 15 2019 12:00AM'
    date_of_change = datetime.datetime.strptime(date_of_change, '%b %d %Y %I:%M%p')
    if date < date_of_change:
        omission_exposures = len([session_type for session_type in pre_expts.session_type if 'OPHYS' in session_type])
    else:
        omission_exposures = len([session_type for session_type in pre_expts.session_type if
                                  ('OPHYS' in session_type) and ('habituation' not in session_type)])
    return omission_exposures


def add_omission_exposure_number_to_experiments_table(experiments, behavior_session_table):

    exposures = []
    for row in range(len(experiments)):
        try:
            behavior_session_id = experiments.iloc[row].behavior_session_id
            omission_exposures = get_omission_exposures_for_behavior_session_id(behavior_session_id, behavior_session_table)
            exposures.append(omission_exposures)
        except Exception:
            exposures.append(np.nan)
    experiments['prior_exposures_to_omissions'] = exposures
    return experiments


def reformat_experiments_table(experiments):
    """
    adds extra columns to experiments table
    :param experiments:
    :return:
    """
    experiments = experiments.reset_index()
    # clean up cre_line naming
    experiments['cre_line'] = [
        full_genotype.split('/')[0] if 'Ai94' not in full_genotype else full_genotype.split('/')[0] + ';Ai94' for
        full_genotype in
        experiments.full_genotype.values]
    experiments = experiments[experiments.cre_line != 'Cux2-CreERT2']  # why is this in the VB dataset?
    # replace session types that are NaN with string None
    experiments.at[experiments[experiments.session_type.isnull()].index.values, 'session_type'] = 'None'
    experiments = add_mouse_seeks_fail_tags_to_experiments_table(experiments)
    if 'level_0' in experiments.columns:
        experiments = experiments.drop(columns='level_0')
    if 'index' in experiments.columns:
        experiments = experiments.drop(columns='index')
    experiments = add_location_to_expts(experiments)
    return experiments


def add_all_qc_states_to_ophys_session_table(session_table, experiment_table):
    """ Add 'experiment_workflow_state', 'container_workflow_state', and 'session_workflow_state' to session_table.
            :param session_table: session_table from SDK cache
            :return: session_table: with additional columns added
            """
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
    experiments = experiment_table[['ophys_session_id', 'ophys_container_id', 'container_workflow_state']].drop_duplicates(
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


def add_mean_pupil_area(stimulus_presentations, eye_tracking, range_relative_to_stimulus_start=[0, 0.75]):
    '''
    Append a column to stimulus_presentations which contains the mean pupil area (in pixels^2) in the window provided.

    Args:
        stimulus_presentations(pd.DataFrame): dataframe of stimulus presentations.
                Must contain: 'start_time'
        eye_tracking (pd.DataFrame): dataframe of eye tracking data.
            Must contain: 'pupil_area', 'timestamps'
        range_relative_to_stimulus_start (list with 2 elements): start and end of the range
            relative to the start of each stimulus to average the pupil area.
    Returns:
        nothing, modifies session in place. Same as the input, but with 'mean_pupil_area' column added
    '''
    mean_pupil_area_df = esp.mean_pupil_area(stimulus_presentations,
                                             eye_tracking,
                                             range_relative_to_stimulus_start)

    stimulus_presentations["mean_pupil_area"] = mean_pupil_area_df
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


def add_image_contrast_to_stimulus_presentations(stimulus_presentations):
    """
    Get image contrast values from saved file and merge with stimulus presentations table using image_name column to merge.
    """
    cache_dir = "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/2020_cache/production_cache"
    df = pd.read_hdf(os.path.join(cache_dir, 'image_metrics_df.h5'), key='df')
    st = stimulus_presentations.copy()
    st = st.reset_index()
    st = st.merge(df[['image_name', 'cropped_image_std', 'warped_image_std']], on='image_name', how='left')
    st = st.set_index('stimulus_presentations_id')
    st.at[st[st.image_name == 'omitted'].index, 'warped_image_std'] = 0.
    st.at[st[st.image_name == 'omitted'].index, 'cropped_image_std'] = 0.
    st['warped_image_std'] = [float(w) for w in st.warped_image_std.values]
    st['cropped_image_std'] = [float(w) for w in st.cropped_image_std.values]
    return st


def add_behavior_performance_metrics_to_experiment_table(experiment_table):
    """
    Get behavior performance metrics from saved file and merge with experiment_table on ophys_experiment_id.
    Performance metrics were obtained from the SDK method .get_performance_metrics() on the dataset object for each experiment.
    """
    save_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/2020_cache/production_cache'
    performance_df = pd.read_csv(os.path.join(save_dir, 'behavior_performance_table.csv'), index_col=0)
    experiment_table = experiment_table.join(performance_df, on='ophys_experiment_id')
    return experiment_table
