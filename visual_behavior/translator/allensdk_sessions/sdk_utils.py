import numpy as np
from visual_behavior.translator.allensdk_sessions import session_attributes as sa


# SDK utilities 
# Created by Alex Piet & Nick Ponvert, 01/17/2019
# Currently works on SDK v.1.3.2


def get_bsid_from_osid(osid,cache):
    '''
        Finds the behavior_session_id associated with an ophys_session_id
        ARGS
            osid    ophys_session_id
            cache   cache from BehaviorProjectCache
        Returns
            bsid    behavior_session_id for that ophys_session
    '''
    ophys_sessions = cache.get_session_table()
    if osid not in ophys_sessions.index:
        raise Exception('ophys_session_id not in session table')
    return ophys_sessions.loc[osid].behavior_session_id

def get_osid_from_bsid(bsid,cache):
    '''
        Finds the ophys_session_id associated with an behavior_session_id
        ARGS
            bsid    behavior_session_id
            cache   cache from BehaviorProjectCache
        Returns
            osid    ophys_session_id for that behavior_session
    '''
    behavior_sessions = cache.get_behavior_session_table()
    if bsid not in behavior_sessions.index:
        raise Exception('behavior_session_id not in behavior session table')
    return behavior_sessions.loc[bsid].ophys_session_id.astype(int)

def get_oeid_from_bsid(bsid,cache,exp_num=0):
    '''
        Finds the ophys_experiment_id associated with an behavior_session_id
        ARGS
            bsid    behavior_session_id
            cache   cache from BehaviorProjectCache
            exp_num index for which experiment to grab the id for
        Returns
            oeid    ophys_experiment_id for that behavior_session
                    For scientifica sessions, there is only one experiment per behavior_session, so exp_num = 0
                    For mesoscope, there are 8 experiments, so exp_num = (0,7)
    '''
    osid = get_osid_from_bsid(bsid,cache)
    return get_oeid_from_osid(osid,exp_num=exp_num)

def get_oeid_from_osid(osid,cache,exp_num = 0):
    '''
        Finds the behavior_session_id associated with an ophys_session_id
        ARGS
            osid    ophys_session_id
            cache   cache from BehaviorProjectCache
            exp_num index for which experiment to grab the id for
        Returns
            oeid    ophys_experiment_id for that ophys_session
                    For scientifica sessions, there is only one experiment per ophys_session, so exp_num = 0
                    For mesoscope, there are 8 experiments, so exp_num = (0,7)
    '''
    ophys_sessions = cache.get_session_table()
    if osid not in ophys_sessions.index:
        raise Exception('ophys_session_id not in session table')
    experiments= ophys_sessions.loc[osid].ophys_experiment_id
    return experiments[0]

def get_bsid_from_oeid(oeid,cache):
    '''
        Finds the behavior_session_id associated with an ophys_experiment_id
        ARGS
            oeid    ophys_experiment_id
            cache   cache from BehaviorProjectCache
        Returns
            bsid    behavior_session_id for that ophys_experiment
    '''
    ophys_experiments = cache.get_experiment_table()
    if oeid not in ophys_experiments.index:
        raise Exception('ophys_experiment_id not in experiment table')
    return ophys_experiments.loc[oeid].behavior_session_id


def get_osid_from_oeid(oeid,cache):
    '''
        Finds the ophys_session_id associated with an ophys_experiment_id
        ARGS
            oeid    ophys_experiment_id
            cache   cache from BehaviorProjectCache
        Returns
            osid    ophys_session_id for that ophys_experiment
    '''   
    ophys_experiments = cache.get_experiment_table()
    if oeid not in ophys_experiments.index:
        raise Exception('ophys_experiment_id not in experiment table')
    return ophys_experiments.loc[oeid].ophys_session_id


def get_specimen_id_from_donor_id(d_id):
    '''
        Gets the specimen_id associated with a donor_id
        
        WARNING, this will not work if the donor_id does not have an associated specimen_id. 
        
        WARNING, this function was meant as a temporary holdover while waiting for SDK support 
    '''
    cache = get_cache()
    ophys_sessions = cache.get_session_table()   
    behavior_sessions = cache.get_behavior_session_table()
    x = behavior_sessions.query('donor_id == @d_id')['ophys_session_id']
    osid = x[~x.isnull()].values[0].astype(int)
    specimen_id = ophys_sessions.query('ophys_session_id ==@osid')['specimen_id'].values[0]
    return specimen_id

def get_donor_id_from_specimen_id(s_id):
    '''
        Gets the donor_id associated with a specimen_id
        
        WARNING, this function was meant as a temporary holdover while waiting for SDK support 
    '''
    cache = get_cache()
    ophys_sessions = cache.get_session_table()   
    behavior_sessions = cache.get_behavior_session_table()
    osid = ophys_sessions.query('specimen_id == @s_id').iloc[0].name
    donor_id = behavior_sessions.query('ophys_session_id ==@osid')['donor_id'].values[0]
    return donor_id


def add_stimulus_presentations_analysis(session):
    '''
        Adds a series of columns to the stimulus_presentations table  
        
        WARNING, this function was meant as a temporary holdover while waiting for SDK support. 
            If these columns are already implemented in the SDK, then using this function will 
            overwrite them. Check before using. 
    '''
    trials = session.trials # NEED TO QUERY THIS FIRST BECAUSE OF CONVERT_LICKS() 
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


