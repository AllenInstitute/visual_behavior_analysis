from visual_behavior.translator.allensdk_sessions import session_attributes as sa
import numpy as np

# SDK utilities
# Created by Alex Piet & Nick Ponvert, 01/17/2019
# Currently works on SDK v.1.3.2


def get_bsid_from_ophys_session_id(ophys_session_id, cache):
    '''
        Finds the behavior_session_id associated with an ophys_session_id
        ARGS
            ophys_session_id    ophys_session_id
            cache   cache from BehaviorProjectCache
        Returns
            bsid    behavior_session_id for that ophys_session
    '''
    ophys_sessions = cache.get_session_table()
    if ophys_session_id not in ophys_sessions.index:
        raise Exception('ophys_session_id not in session table')
    return ophys_sessions.loc[ophys_session_id].behavior_session_id


def get_ophys_session_id_from_bsid(bsid, cache):
    '''
        Finds the ophys_session_id associated with an behavior_session_id
        ARGS
            bsid    behavior_session_id
            cache   cache from BehaviorProjectCache
        Returns
            ophys_session_id    ophys_session_id for that behavior_session
    '''
    behavior_sessions = cache.get_behavior_session_table()
    if bsid not in behavior_sessions.index:
        raise Exception('behavior_session_id not in behavior session table')
    return behavior_sessions.loc[bsid].ophys_session_id.astype(int)


def get_ophys_experiment_id_from_bsid(bsid, cache, exp_num=0):
    '''
        Finds the ophys_experiment_id associated with an behavior_session_id
        ARGS
            bsid    behavior_session_id
            cache   cache from BehaviorProjectCache
            exp_num index for which experiment to grab the id for
        Returns
            ophys_experiment_id    ophys_experiment_id for that behavior_session
                    For scientifica sessions, there is only one experiment per behavior_session, so exp_num = 0
                    For mesoscope, there are 8 experiments, so exp_num = (0,7)
    '''
    ophys_session_id = get_ophys_session_id_from_bsid(bsid, cache)
    return get_ophys_experiment_id_from_ophys_session_id(ophys_session_id, cache, exp_num=exp_num)


def get_ophys_experiment_id_from_ophys_session_id(ophys_session_id, cache, exp_num=0):
    '''
        Finds the behavior_session_id associated with an ophys_session_id
        ARGS
            ophys_session_id    ophys_session_id
            cache   cache from BehaviorProjectCache
            exp_num index for which experiment to grab the id for
        Returns
            ophys_experiment_id    ophys_experiment_id for that ophys_session
                    For scientifica sessions, there is only one experiment per ophys_session, so exp_num = 0
                    For mesoscope, there are 8 experiments, so exp_num = (0,7)
    '''
    ophys_sessions = cache.get_session_table()
    if ophys_session_id not in ophys_sessions.index:
        raise Exception('ophys_session_id not in session table')
    experiments = ophys_sessions.loc[ophys_session_id].ophys_experiment_id
    return experiments[0]


def get_bsid_from_ophys_experiment_id(ophys_experiment_id, cache):
    '''
        Finds the behavior_session_id associated with an ophys_experiment_id
        ARGS
            ophys_experiment_id    ophys_experiment_id
            cache   cache from BehaviorProjectCache
        Returns
            bsid    behavior_session_id for that ophys_experiment
    '''
    ophys_experiments = cache.get_experiment_table()
    if ophys_experiment_id not in ophys_experiments.index:
        raise Exception('ophys_experiment_id not in experiment table')
    return ophys_experiments.loc[ophys_experiment_id].behavior_session_id


def get_ophys_session_id_from_ophys_experiment_id(ophys_experiment_id, cache):
    '''
        Finds the ophys_session_id associated with an ophys_experiment_id
        ARGS
            ophys_experiment_id    ophys_experiment_id
            cache   cache from BehaviorProjectCache
        Returns
            ophys_session_id    ophys_session_id for that ophys_experiment
    '''
    ophys_experiments = cache.get_experiment_table()
    if ophys_experiment_id not in ophys_experiments.index:
        raise Exception('ophys_experiment_id not in experiment table')
    return ophys_experiments.loc[ophys_experiment_id].ophys_session_id


def get_specimen_id_from_donor_id(d_id, cache):
    '''
        Gets the specimen_id associated with a donor_id
        ARGS
            d_id    donor_id
            cache   cache from BehaviorProjectCache

        WARNING, this will not work if the donor_id does not have an associated specimen_id.

        WARNING, this function was meant as a temporary holdover while waiting for SDK support
    '''
    ophys_sessions = cache.get_session_table()
    behavior_sessions = cache.get_behavior_session_table()
    x = behavior_sessions.query('donor_id == @d_id')['ophys_session_id']
    ophys_session_id = x[~x.isnull()].values[0].astype(int)  # noqa: F841
    specimen_id = ophys_sessions.query('ophys_session_id ==@ophys_session_id')['specimen_id'].values[0]
    return specimen_id


def get_donor_id_from_specimen_id(s_id, cache):
    '''
        Gets the donor_id associated with a specimen_id
        ARGS
            s_id    specimen_id
            cache   cache from BehaviorProjectCache

        WARNING, this function was meant as a temporary holdover while waiting for SDK support
    '''
    ophys_sessions = cache.get_session_table()
    behavior_sessions = cache.get_behavior_session_table()
    ophys_session_id = ophys_sessions.query('specimen_id == @s_id').iloc[0].name  # noqa: F841
    donor_id = behavior_sessions.query('ophys_session_id ==@ophys_session_id')['donor_id'].values[0]
    return donor_id


def add_stimulus_presentations_analysis(session):
    '''
        Adds a series of columns to the stimulus_presentations table

        WARNING, this function was meant as a temporary holdover while waiting for SDK support.
            If these columns are already implemented in the SDK, then using this function will
            overwrite them. Check before using.
    '''
    trials = session.trials  # noqa: F841
    # NEED TO QUERY THIS FIRST BECAUSE OF CONVERT_LICKS()
    # allensdk/brain_observatory/behavior/trials_processing.get_trials() has an assertion
    # that session.rewards has timestamps as an index. And the code requres that session.licks
    # has 'time' as a column. Therefore, before modifying those attributes, we load the trials
    # table once, and this uses the memoize attribute to calculate this dataframe first. We realize
    # this is a terrible hack, but its the easiest way forward until the naming conventions are fixed
    sa.convert_licks_inplace(session.licks)
    sa.convert_rewards_inplace(session.rewards)
    sa.add_licks_each_flash_inplace(session)
    sa.add_rewards_each_flash_inplace(session)
    sa.add_change_each_flash_inplace(session)
    sa.add_time_from_last_lick_inplace(session)
    sa.add_time_from_last_reward_inplace(session)
    sa.add_time_from_last_change_inplace(session)
    sa.add_mean_running_speed_inplace(session)


def get_filtered_sessions_table(cache, require_cell_matching=False, require_full_container=True, require_exp_pass=True, include_multiscope=False):
    '''
        Applies some filters to the ophys_sessions_table. It will always filter out all sessions that do not have
        project codes of 'VisualBehavior' or 'VisualBehaviorTask1B'. This currently removes all Mesoscope sessions.
        It will filter out all sessions that are not found in the behavior_sessions_table, or in the ophys_experiment_table.
        Optionally, you can require experiments that currently pass QC workflow. Or only sessions from containers that pass
        QC workflow. Or only sessions from containers that have cell matching completed.

        WARNING! This function is meant as a hold over until SDK support

        ARGS:
            cache                       cache from BehaviorProjectCache
            require_cell_matching       If True, forces require_full_container and require_exp_pass = True
                                        Only returns ophys_sessions from full containers that have gone through cell_matching
                                        Asserts that each container has 6 sessions
            require_full_container      If True, returns sessions from containers with container_workflow_state of "container_qc" or "completed"
                                        Unless require_exp_pass is True, it will return failed sessions within that container
            require_exp_pass            if True, returns sessions with experiment_workflow_state = passed
            include_multiscope          if True, returns sessions under project codes VisualBehaviorMultiScope and
                                        VisualBehaviorMultiscope4areasx2d in addition to VisualBehavior and VisualBehaviorTask1b

        RETURNS
            The ophys_sessions_table filtered by the constraints above.
    '''
    # Only get cell matching for full containers, on passed experiments
    if require_cell_matching:
        require_full_container = True
        require_exp_pass = True

    ophys_sessions = cache.get_session_table()
    ophys_experiments = cache.get_experiment_table()
    behavior_sessions = cache.get_behavior_session_table()

    # Ensure sessions are in the other tables
    session_ids = np.array(ophys_sessions.index)
    session_in_experiment_table = [any(ophys_experiments['ophys_session_id'] == x) for x in session_ids]
    session_in_bsession_table = [any(behavior_sessions['ophys_session_id'] == x) for x in session_ids]
    ophys_sessions['in_experiment_table'] = session_in_experiment_table
    ophys_sessions['in_bsession_table'] = session_in_bsession_table

    # Check Project Code
    if include_multiscope:
        good_code = ophys_sessions['project_code'].isin(['VisualBehavior', 'VisualBehaviorTask1B',
                                                         'VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d'])
    else:
        good_code = ophys_sessions['project_code'].isin(['VisualBehavior', 'VisualBehaviorTask1B'])
    ophys_sessions['good_project_code'] = good_code

    # Check Session Type
    good_session = ophys_sessions['session_type'].isin(['OPHYS_1_images_A', 'OPHYS_3_images_A', 'OPHYS_4_images_B',
                                                        'OPHYS_5_images_B_passive', 'OPHYS_6_images_B', 'OPHYS_2_images_A_passive', 'OPHYS_1_images_B',
                                                        'OPHYS_2_images_B_passive', 'OPHYS_3_images_B', 'OPHYS_4_images_A', 'OPHYS_5_images_A_passive', 'OPHYS_6_images_A'])
    ophys_sessions['good_session'] = good_session

    # Check Experiment Workflow state
    ophys_experiments['good_exp_workflow'] = ophys_experiments['experiment_workflow_state'] == "passed"

    # Check Container Workflow state
    if require_cell_matching:
        ophys_experiments['good_container_workflow'] = ophys_experiments['container_workflow_state'] == "container_qc"
    else:
        ophys_experiments['good_container_workflow'] = ophys_experiments['container_workflow_state'].isin(['container_qc', 'completed'])

    # Compile workflow state info into ophys_sessions
    ophys_experiments_good_workflow = ophys_experiments.query('good_exp_workflow')
    ophys_experiments_good_container = ophys_experiments.query('good_container_workflow')
    session_good_workflow = [any(ophys_experiments_good_workflow['ophys_session_id'] == x) for x in session_ids]
    container_good_workflow = [any(ophys_experiments_good_container['ophys_session_id'] == x) for x in session_ids]
    ophys_sessions['good_exp_workflow'] = session_good_workflow
    ophys_sessions['good_container_workflow'] = container_good_workflow

    # do final filtering
    if require_full_container and require_exp_pass:
        filtered = ophys_sessions.query('good_project_code & good_session & in_bsession_table & in_experiment_table & good_container_workflow & good_exp_workflow')
    elif require_full_container:
        filtered = ophys_sessions.query('good_project_code & good_session & in_bsession_table & in_experiment_table & good_container_workflow')
    elif require_exp_pass:
        filtered = ophys_sessions.query('good_project_code & good_session & in_bsession_table & in_experiment_table & good_exp_workflow')
    else:
        filtered = ophys_sessions.query('good_project_code & good_session & in_bsession_table & in_experiment_table')

    if require_cell_matching:
        if not (np.mod(len(filtered), 6) == 0):
            print('WARNING: number of experiments not divisible by 6, likely incomplete containers')
    elif require_full_container and require_exp_pass:
        if not (np.mod(len(filtered), 6) == 0):
            print('WARNING: number of experiments not divisible by 6, likely incomplete containers')

    return filtered
