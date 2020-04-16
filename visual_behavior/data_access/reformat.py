
import os
import numpy as np
import pandas as pd

from visual_behavior.data_access import loading
import visual_behavior.ophys.dataset.extended_stimulus_processing as esp


### FUNCTIONS TO REFORMAT DATA LOADED FROM ALLENSDK TO ADDRESS ISSUES WITH MISSING, INCORRECT OR IMPROPERLY STRUCTURED DATA ###

### THESE FUNCTIONS ARE TEMPORARY WORKAROUNDS UNTIL THE ISSUES CAN BE RESOLVED IN THE SDK ###


# REFORMATTING MANIFEST DATA #


def add_mouse_seeks_fail_tags_to_experiments_table(experiments):
    mouse_seeks_report_file_base = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    report_file = 'ophys_session_log_031820.xlsx'
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
    expts['location'] = [expts.loc[x]['cre_line'].split('-')[0] + '_' + expts.loc[x]['targeted_structure'] + '_' + str(int(expts.loc[x]['imaging_depth'])) for x in expts.index.values]
    return expts


def get_exposure_number_for_group(group):
    order = np.argsort(group['date_of_acquisition'].values)
    group['exposure_number'] = order
    return group


def add_exposure_number_to_experiments_table(experiments):
    experiments = experiments.groupby(['super_container_id', 'container_id', 'session_type']).apply(get_exposure_number_for_group)
    return experiments


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


def add_change_each_flash_inplace(session):
    '''
        Adds a column to session.stimulus_presentations, ['change'], which is True if the stimulus was a change image, and False otherwise

        ARGS: SDK session object
        RETURNS: nothing
        MODIFIES: session.stimulus_presentations dataframe
    '''
    changes = esp.find_change(session.stimulus_presentations['image_index'], esp.get_omitted_index(session.stimulus_presentations))
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


