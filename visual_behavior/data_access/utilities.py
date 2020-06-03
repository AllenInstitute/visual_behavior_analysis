import os
from visual_behavior.data_access import loading


# CONVENIENCE FUNCTIONS TO GET VARIOUS INFORMATION #

# put functions here such as get_ophys_experiment_id_for_ophys_session_id()

class LazyLoadable(object):
    def __init__(self, name, calculate):
        ''' Wrapper for attributes intended to be computed or loaded once, then held in memory by a containing object.

        Parameters
        ----------
        name : str
            The name of the hidden attribute in which this attribute's data will be stored.
        calculate : fn
            a function (presumably expensive) used to calculate or load this attribute's data

        '''

        self.name = name
        self.calculate = calculate


def check_for_model_outputs(behavior_session_id):
    """
    Checks whether model output file with omission regressors exists (does not say '_training' at end of filename)
    :param behavior_session_id:
    :return:
    """
    model_output_dir = loading.get_behavior_model_outputs_dir()
    model_output_file = [file for file in os.listdir(model_output_dir) if
                         (str(behavior_session_id) in file) and ('training' not in file)]
    if len(model_output_file) > 0:
        return True
    else:
        return False


# retrieve data from cache
def get_behavior_session_id_from_ophys_session_id(ophys_session_id, cache):
    """finds the behavior_session_id assocciated with an ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit, unique identifier for an ophys_session
        cache {object} -- cache from BehaviorProjectCache

    Raises:
        Exception: [description]

    Returns:
        int -- behavior_session_id : 9 digit, unique identifier for a
                behavior_session
    """
    ophys_sessions_table = cache.get_session_table()
    if ophys_session_id not in ophys_sessions_table.index:
        raise Exception('ophys_session_id not in session table')
    return ophys_sessions_table.loc[ophys_session_id].behavior_session_id


def get_ophys_session_id_from_behavior_session_id(behavior_session_id, cache):
    """Finds the behavior_session_id associated with an ophys_session_id

    Arguments:
        behavior_session_id {int} -- 9 digit, unique identifier for a behavior_session
        cache {object} -- cache from BehaviorProjectCache

    Raises:
        Exception: [description]

    Returns:
        int -- ophys_session_id: 9 digit, unique identifier for an ophys_session
    """
    behavior_sessions = cache.get_behavior_session_table()
    if behavior_session_id not in behavior_sessions.index:
        raise Exception('behavior_session_id not in behavior session table')
    return behavior_sessions.loc[behavior_session_id].ophys_session_id.astype(int)


def get_ophys_experiment_id_from_behavior_session_id(behavior_session_id, cache, exp_num=0):
    """Finds the ophys_experiment_id associated with an behavior_session_id. It is possible
    that there are multiple ophys_experiments for a single behavior session- as is the case
    for data collected on the multiscope microscopes

    Arguments:
        behavior_session_id {int} -- [description]
        cache {object} -- cache from BehaviorProjectCache

    Keyword Arguments:
        exp_num {int} -- number of expected ophys_experiments
                        For scientifica sessions, there is only one experiment
                        per behavior_session, so exp_num = 0
                        For mesoscope, there are 8 experiments,
                        so exp_num = (0,7) (default: {0})

    Returns:
        int -- ophys_experiment_id(s), 9 digit unique identifier for an ophys_experiment
                possible that there are multip ophys_experiments for one behavior_session
    """
    ophys_session_id = get_ophys_session_id_from_behavior_session_id(behavior_session_id, cache)
    ophys_experiment_id = get_ophys_experiment_id_from_ophys_session_id(ophys_session_id, cache, exp_num=exp_num)
    return ophys_experiment_id


def get_ophys_experiment_id_from_ophys_session_id(ophys_session_id, cache, exp_num=0):
    """finds the ophys_experiment_id associated with an ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit, unique identifier for an ophys_session
        cache {object} -- cache from BehaviorProjectCache

    Keyword Arguments:
        exp_num {int} -- number of expected ophys_experiments
                        For scientifica sessions, there is only one experiment
                        per ophys_session, so exp_num = 0
                        For mesoscope, there are 8 experiments,
                        so exp_num = (0,7) (default: {0})

    Raises:
        Exception: [description]

    Returns:
        int -- ophys_experiment_id(s), 9 digit unique identifier for an ophys_experiment
        possible that there are multip ophys_experiments for one ophys_session
    """
    ophys_sessions = cache.get_session_table()
    if ophys_session_id not in ophys_sessions.index:
        raise Exception('ophys_session_id not in session table')
    experiments = ophys_sessions.loc[ophys_session_id].ophys_experiment_id
    return experiments[0]


def get_behavior_session_id_from_ophys_experiment_id(ophys_experiment_id, cache):
    """finds the behavior_session_id associated with an ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit, unique identifier for an ophys_experimet
        cache {object} -- cache from BehaviorProjectCache

    Raises:
        Exception: [description]

    Returns:
        int -- behavior_session_id, 9 digit, unique identifier for a behavior_session
    """
    ophys_experiments = cache.get_experiment_table()
    if ophys_experiment_id not in ophys_experiments.index:
        raise Exception('ophys_experiment_id not in experiment table')
    return ophys_experiments.loc[ophys_experiment_id].behavior_session_id


def get_ophys_session_id_from_ophys_experiment_id(ophys_experiment_id, cache):
    """finds the ophys_session_id associated with an ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit, unique identifier for an ophys_experimet
        cache {object} -- cache from BehaviorProjectCache

    Raises:
        Exception: [description]

    Returns:
        int -- ophys_session_id: 9 digit, unique identifier for an ophys_session
    """
    ophys_experiments = cache.get_experiment_table()
    if ophys_experiment_id not in ophys_experiments.index:
        raise Exception('ophys_experiment_id not in experiment table')
    return ophys_experiments.loc[ophys_experiment_id].ophys_session_id


def get_donor_id_from_specimen_id(specimen_id, cache):
    """gets a donor_id associated with a specimen_id. Both donor_id
        and specimen_id are identifiers for a mouse.

    Arguments:
        specimen_id {int} -- 9 digit unique identifier for a mouse
        cache {object} -- cache from BehaviorProjectCache

    Returns:
        int -- donor id
    """
    ophys_sessions = cache.get_session_table()
    behavior_sessions = cache.get_behavior_session_table()
    ophys_session_id = ophys_sessions.query('specimen_id == @specimen_id').iloc[0].name  # noqa: F841
    donor_id = behavior_sessions.query('ophys_session_id ==@ophys_session_id')['donor_id'].values[0]
    return donor_id


def model_outputs_available_for_behavior_session(behavior_session_id):
    """
    Check whether behavior model outputs are available in the default directory

    :param behavior_session_id: 9-digit behavior session ID
    :return: Boolean, True if outputs are available, False if not
    """
    model_output_dir = loading.get_behavior_model_outputs_dir()
    model_output_file = [file for file in os.listdir(model_output_dir) if str(behavior_session_id) in file]
    if len(model_output_file) > 0:
        return True
    else:
        return False
