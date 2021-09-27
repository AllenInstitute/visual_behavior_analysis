import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from visual_behavior.data_access import loading
from visual_behavior.ophys.io.lims_database import LimsDatabase
from visual_behavior.ophys.sync.sync_dataset import Dataset as SyncDataset
from visual_behavior.ophys.sync.process_sync import filter_digital, calculate_delay
from visual_behavior import database as db

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
from allensdk.brain_observatory.behavior.behavior_session import BehaviorSession
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession

import logging
logger = logging.getLogger(__name__)


# warning
gen_depr_str = 'this function is deprecated and will be removed in a future version, ' \
               + 'please use {}.{} instead'

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
    return len(model_output_file) > 0


def get_all_session_ids(ophys_experiment_id=None, ophys_session_id=None, behavior_session_id=None, foraging_id=None):
    '''
    a function to get all ID types for a given experiment ID
    Arguments:
        ophys_experiment_id {int} -- unique identifier for an ophys_experiment
        ophys_session_id {int} -- unique identifier for an ophys_session
        behavior_session_id {int} -- unique identifier for a behavior_session
        foraging_id {int} -- unique identifier for a behavior_session (1:1 with behavior session ID)

    Only one experiment ID type should be passed
    Returns:
        dataframe with one column for each ID type (potentially multiple rows)
    '''
    if ophys_experiment_id:
        table = 'oe'
        search_key = 'id'
        id_to_search = ophys_experiment_id
    elif ophys_session_id:
        table = 'os'
        search_key = 'id'
        id_to_search = ophys_session_id
    elif behavior_session_id:
        table = 'bs'
        search_key = 'id'
        id_to_search = behavior_session_id
    elif foraging_id:
        table = 'bs'
        search_key = 'foraging_id'
        id_to_search = "'{}'".format(foraging_id)
    lims_query = '''
        select
            bs.id as behavior_session_id,
            bs.foraging_id as foraging_id,
            os.id as ophys_session_id,
            oe.id as ophys_experiment_id,
            oevbec.visual_behavior_experiment_container_id as container_id,
            os.visual_behavior_supercontainer_id as supercontainer_id
        from behavior_sessions as bs
        join ophys_sessions as os on os.id = bs.ophys_session_id
        join ophys_experiments as oe on os.id = oe.ophys_session_id
        join ophys_experiments_visual_behavior_experiment_containers AS oevbec on oevbec.ophys_experiment_id=oe.id
        where {}.{} = {}
    '''
    return db.lims_query(lims_query.format(table, search_key, id_to_search))


def get_behavior_session_id_from_ophys_session_id(ophys_session_id, cache=None):
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
    if cache:
        ophys_sessions_table = cache.get_session_table()
        if ophys_session_id not in ophys_sessions_table.index:
            raise Exception('ophys_session_id not in session table')
        return ophys_sessions_table.loc[ophys_session_id].behavior_session_id
    else:
        # if cache not passed, go to lims
        lims_query_string = '''
            select bs.id
            from behavior_sessions as bs
            where bs.ophys_session_id = {}
        '''
        return db.lims_query(lims_query_string.format(ophys_session_id)).astype(int)


def get_ophys_session_id_from_behavior_session_id(behavior_session_id, cache=None):
    """Finds the behavior_session_id associated with an ophys_session_id

    Arguments:
        behavior_session_id {int} -- 9 digit, unique identifier for a behavior_session
        cache {object} -- cache from BehaviorProjectCache (optional)

    Raises:
        Exception: [description]

    Returns:
        int -- ophys_session_id: 9 digit, unique identifier for an ophys_session
    """
    if cache:
        behavior_sessions = cache.get_behavior_session_table()
        if behavior_session_id not in behavior_sessions.index:
            raise Exception('behavior_session_id not in behavior session table')
        return behavior_sessions.loc[behavior_session_id].ophys_session_id.astype(int)
    else:
        # if cache not passed, go to lims
        lims_query_string = '''
            select bs.ophys_session_id
            from behavior_sessions as bs
            where bs.id = {}
        '''
        return db.lims_query(lims_query_string.format(behavior_session_id)).astype(int)


def get_ophys_experiment_id_from_behavior_session_id(behavior_session_id, cache=None, exp_num=0):
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
    if cache:
        ophys_session_id = get_ophys_session_id_from_behavior_session_id(behavior_session_id, cache)
        ophys_experiment_id = get_ophys_experiment_id_from_ophys_session_id(ophys_session_id, cache, exp_num=exp_num)
        return ophys_experiment_id
    else:
        # if cache not passed, go to lims
        lims_query_string = '''
            select oe.id
            from behavior_sessions as bs
            join ophys_sessions as os on os.id = bs.ophys_session_id
            join ophys_experiments as oe on os.id = oe.ophys_session_id
            where bs.id = {}

        '''
        result = db.lims_query(lims_query_string.format(behavior_session_id))
        if isinstance(result, (int, np.int64)):
            return result.astype(int)
        else:
            return result['id'].astype(int).to_list()


def get_ophys_experiment_id_from_ophys_session_id(ophys_session_id, cache=None, exp_num=0):
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
    if cache:
        ophys_sessions = cache.get_session_table()
        if ophys_session_id not in ophys_sessions.index:
            raise Exception('ophys_session_id not in session table')
        experiments = ophys_sessions.loc[ophys_session_id].ophys_experiment_id
        return experiments[0]
    else:
        # if cache not passed, go to lims
        lims_query_string = '''
            select oe.id
            from ophys_sessions as os
            join ophys_experiments as oe on os.id = oe.ophys_session_id
            where os.id = {}

        '''
        result = db.lims_query(lims_query_string.format(ophys_session_id))
        if isinstance(result, (int, np.int64)):
            return result.astype(int)
        else:
            return result['id'].astype(int).to_list()


def get_behavior_session_id_from_ophys_experiment_id(ophys_experiment_id, cache=None):
    """finds the behavior_session_id associated with an ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit, unique identifier for an ophys_experimet
        cache {object} -- cache from BehaviorProjectCache

    Raises:
        Exception: [description]

    Returns:
        int -- behavior_session_id, 9 digit, unique identifier for a behavior_session
    """
    if cache:
        ophys_experiments = cache.get_experiment_table()
        if ophys_experiment_id not in ophys_experiments.index:
            raise Exception('ophys_experiment_id not in experiment table')
        return ophys_experiments.loc[ophys_experiment_id].behavior_session_id
    else:
        # if cache not passed, go to lims
        lims_query_string = '''
            select bs.id
            from behavior_sessions as bs
            join ophys_sessions as os on os.id = bs.ophys_session_id
            join ophys_experiments as oe on os.id = oe.ophys_session_id
            where oe.id = {}

        '''
        return db.lims_query(lims_query_string.format(ophys_experiment_id)).astype(int)


def get_ophys_session_id_from_ophys_experiment_id(ophys_experiment_id, cache=None):
    """finds the ophys_session_id associated with an ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit, unique identifier for an ophys_experimet
        cache {object} -- cache from BehaviorProjectCache

    Raises:
        Exception: [description]

    Returns:
        int -- ophys_session_id: 9 digit, unique identifier for an ophys_session
    """
    if cache:
        ophys_experiments = cache.get_experiment_table()
        if ophys_experiment_id not in ophys_experiments.index:
            raise Exception('ophys_experiment_id not in experiment table')
        return ophys_experiments.loc[ophys_experiment_id].ophys_session_id
    else:
        # if cache not passed, go to lims
        lims_query_string = '''
            select os.id
            from ophys_sessions as os
            join ophys_experiments as oe on os.id = oe.ophys_session_id
            where oe.id = {}

        '''
        return db.lims_query(lims_query_string.format(ophys_experiment_id)).astype(int)


def get_donor_id_from_specimen_id(specimen_id, cache=None):
    """gets a donor_id associated with a specimen_id. Both donor_id
        and specimen_id are identifiers for a mouse.

    Arguments:
        specimen_id {int} -- 9 digit unique identifier for a mouse
        cache {object} -- cache from BehaviorProjectCache

    Returns:
        int -- donor id
    """
    if cache:
        ophys_sessions = cache.get_session_table()
        behavior_sessions = cache.get_behavior_session_table()
        ophys_session_id = ophys_sessions.query('specimen_id == @specimen_id').iloc[0].name  # noqa: F841
        donor_id = behavior_sessions.query('ophys_session_id ==@ophys_session_id')['donor_id'].values[0]
        return donor_id
    else:
        # if cache not passed, go to lims
        lims_query_string = '''
            select donor_id
            from specimens
            where specimens.id = '{}'
        '''
        return db.lims_query(lims_query_string.format(specimen_id)).astype(int)


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


# def get_cell_matching_output_dir_for_container(container_id, experiments_table):
#     container_expts = experiments_table[experiments_table.container_id==container_id]
#     ophys_experiment_id = container_expts.index[0]
#     lims_data = get_lims_data(ophys_experiment_id)
#     session_dir = lims_data.ophys_session_dir.values[0]
#     cell_matching_dir = os.path.join(session_dir[:-23], 'experiment_container_'+str(container_id), 'OphysNwayCellMatchingStrategy')
#     cell_matching_output_dir = os.path.join(cell_matching_dir, np.sort(os.listdir(cell_matching_dir))[-1])
#     return cell_matching_output_dir
#


def get_cell_matching_output_dir_for_container(experiment_id):
    from visual_behavior import database

    query = '''
            SELECT DISTINCT sp.external_specimen_name, sp.name, vbec.id AS vbec_id, vbec.workflow_state AS vbec_state, vbcr.run_number, vbcr.storage_directory AS matching_dir
            FROM ophys_experiments_visual_behavior_experiment_containers oevbec
            JOIN visual_behavior_experiment_containers vbec ON vbec.id=oevbec.visual_behavior_experiment_container_id
            JOIN ophys_experiments oe ON oe.id=oevbec.ophys_experiment_id
            JOIN ophys_sessions os ON os.id=oe.ophys_session_id JOIN specimens sp ON sp.id=os.specimen_id
            JOIN projects p ON p.id=vbec.project_id
            LEFT JOIN visual_behavior_container_runs vbcr ON vbcr.visual_behavior_experiment_container_id=vbec.id AND vbcr.current = 't'
            WHERE
            --sp.external_specimen_name NOT IN ('398691')
            oe.id = {};
            '''.format(experiment_id)

    lims_df = database.lims_query(query)
    return lims_df.matching_dir.values[0]


def get_ssim(img0, img1):
    from skimage.measure import compare_ssim as ssim
    ssim_pair = ssim(img0, img1, gaussian_weights=True)
    return ssim_pair


def get_lims_data(lims_id):
    ld = LimsDatabase(int(lims_id))
    lims_data = ld.get_qc_param()
    lims_data.insert(loc=2, column='experiment_id', value=lims_data.lims_id.values[0])
    lims_data.insert(loc=2, column='session_type',
                     value='behavior_' + lims_data.experiment_name.values[0].split('_')[-1])
    lims_data.insert(loc=2, column='ophys_session_dir', value=lims_data.datafolder.values[0][:-28])
    return lims_data


def get_timestamps(lims_data):
    if '2P6' in lims_data.rig.values[0]:
        use_acq_trigger = True
    else:
        use_acq_trigger = False
    sync_data = get_sync_data(lims_data, use_acq_trigger)
    timestamps = pd.DataFrame(sync_data)
    return timestamps


def get_sync_path(lims_data):
    ophys_session_dir = get_ophys_session_dir(lims_data)

    sync_file = [file for file in os.listdir(ophys_session_dir) if 'sync' in file]
    if len(sync_file) > 0:
        sync_file = sync_file[0]
    else:
        json_path = [file for file in os.listdir(ophys_session_dir) if '_platform.json' in file][0]
        with open(os.path.join(ophys_session_dir, json_path)) as pointer_json:
            json_data = json.load(pointer_json)
            sync_file = json_data['sync_file']
    sync_path = os.path.join(ophys_session_dir, sync_file)
    return sync_path


def get_sync_data(lims_data, use_acq_trigger):
    logger.info('getting sync data')
    sync_path = get_sync_path(lims_data)
    sync_dataset = SyncDataset(sync_path)
    # Handle mesoscope missing labels
    try:
        sync_dataset.get_rising_edges('2p_vsync')
    except ValueError:
        sync_dataset.line_labels = ['2p_vsync', '', 'stim_vsync', '', 'photodiode', 'acq_trigger', '', '',
                                    'behavior_monitoring', 'eye_tracking', '', '', '', '', '', '', '', '', '', '', '',
                                    '', '', '', '', '', '', '', '', '', '', 'lick_sensor']
        sync_dataset.meta_data['line_labels'] = sync_dataset.line_labels

    meta_data = sync_dataset.meta_data
    sample_freq = meta_data['ni_daq']['counter_output_freq']
    # 2P vsyncs
    vs2p_r = sync_dataset.get_rising_edges('2p_vsync')
    vs2p_f = sync_dataset.get_falling_edges(
        '2p_vsync', )  # new sync may be able to do units = 'sec', so conversion can be skipped
    vs2p_rsec = vs2p_r / sample_freq
    vs2p_fsec = vs2p_f / sample_freq
    if use_acq_trigger:  # if 2P6, filter out solenoid artifacts
        vs2p_r_filtered, vs2p_f_filtered = filter_digital(vs2p_rsec, vs2p_fsec, threshold=0.01)
        frames_2p = vs2p_r_filtered
    else:  # dont need to filter out artifacts in pipeline data
        frames_2p = vs2p_rsec
    # use rising edge for Scientifica, falling edge for Nikon http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
    # Convert to seconds - skip if using units in get_falling_edges, otherwise convert before doing filter digital
    # vs2p_rsec = vs2p_r / sample_freq
    # frames_2p = vs2p_rsec
    # stimulus vsyncs
    # vs_r = d.get_rising_edges('stim_vsync')
    vs_f = sync_dataset.get_falling_edges('stim_vsync')
    # convert to seconds
    # vs_r_sec = vs_r / sample_freq
    vs_f_sec = vs_f / sample_freq
    # vsyncs = vs_f_sec
    # add display lag
    monitor_delay = calculate_delay(sync_dataset, vs_f_sec, sample_freq)
    vsyncs = vs_f_sec + monitor_delay  # this should be added, right!?
    # line labels are different on 2P6 and production rigs - need options for both
    if 'lick_times' in meta_data['line_labels']:
        lick_times = sync_dataset.get_rising_edges('lick_1') / sample_freq
    elif 'lick_sensor' in meta_data['line_labels']:
        lick_times = sync_dataset.get_rising_edges('lick_sensor') / sample_freq
    else:
        lick_times = None
    if '2p_trigger' in meta_data['line_labels']:
        trigger = sync_dataset.get_rising_edges('2p_trigger') / sample_freq
    elif 'acq_trigger' in meta_data['line_labels']:
        trigger = sync_dataset.get_rising_edges('acq_trigger') / sample_freq
    if 'stim_photodiode' in meta_data['line_labels']:
        stim_photodiode = sync_dataset.get_rising_edges('stim_photodiode') / sample_freq
    elif 'photodiode' in meta_data['line_labels']:
        stim_photodiode = sync_dataset.get_rising_edges('photodiode') / sample_freq
    if 'cam2_exposure' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('cam2_exposure') / sample_freq
    elif 'cam2' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('cam2') / sample_freq
    elif 'eye_tracking' in meta_data['line_labels']:
        eye_tracking = sync_dataset.get_rising_edges('eye_tracking') / sample_freq
    if 'cam1_exposure' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('cam1_exposure') / sample_freq
    elif 'cam1' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('cam1') / sample_freq
    elif 'behavior_monitoring' in meta_data['line_labels']:
        behavior_monitoring = sync_dataset.get_rising_edges('behavior_monitoring') / sample_freq
    # some experiments have 2P frames prior to stimulus start - restrict to timestamps after trigger for 2P6 only
    if use_acq_trigger:
        frames_2p = frames_2p[frames_2p > trigger[0]]
    # print(len(frames_2p))
    if lims_data.rig.values[0][0] == 'M':  # if Mesoscope
        roi_group = get_roi_group(lims_data)  # get roi_group order
        frames_2p = frames_2p[roi_group::4]  # resample sync times
    # print(len(frames_2p))
    logger.info('stimulus frames detected in sync: {}'.format(len(vsyncs)))
    logger.info('ophys frames detected in sync: {}'.format(len(frames_2p)))
    # put sync data in format to be compatible with downstream analysis
    times_2p = {'timestamps': frames_2p}
    times_vsync = {'timestamps': vsyncs}
    times_lick = {'timestamps': lick_times}
    times_trigger = {'timestamps': trigger}
    times_eye_tracking = {'timestamps': eye_tracking}
    times_behavior_monitoring = {'timestamps': behavior_monitoring}
    times_stim_photodiode = {'timestamps': stim_photodiode}
    sync_data = {'ophys_frames': times_2p,
                 'stimulus_frames': times_vsync,
                 'lick_times': times_lick,
                 'eye_tracking': times_eye_tracking,
                 'behavior_monitoring': times_behavior_monitoring,
                 'stim_photodiode': times_stim_photodiode,
                 'ophys_trigger': times_trigger,
                 }
    return sync_data


def get_ophys_session_dir(lims_data):
    ophys_session_dir = lims_data.ophys_session_dir.values[0]
    return ophys_session_dir


def get_ophys_experiment_dir(lims_data):
    lims_id = get_lims_id(lims_data)
    ophys_session_dir = get_ophys_session_dir(lims_data)
    ophys_experiment_dir = os.path.join(ophys_session_dir, 'ophys_experiment_' + str(lims_id))
    return ophys_experiment_dir


def get_roi_group(lims_data):
    experiment_id = int(lims_data.experiment_id.values[0])
    ophys_session_dir = get_ophys_session_dir(lims_data)
    import json
    json_file = [file for file in os.listdir(ophys_session_dir) if ('SPLITTING' in file) and ('input.json' in file)]
    json_path = os.path.join(ophys_session_dir, json_file[0])
    with open(json_path, 'r') as w:
        jin = json.load(w)
    # figure out which roi_group the current experiment belongs to
    # plane_data = pd.DataFrame()
    for i, roi_group in enumerate(range(len(jin['plane_groups']))):
        group = jin['plane_groups'][roi_group]['ophys_experiments']
        for j, plane in enumerate(range(len(group))):
            expt_id = int(group[plane]['experiment_id'])
            if expt_id == experiment_id:
                expt_roi_group = i
    return expt_roi_group


def get_lims_id(lims_data):
    lims_id = lims_data.lims_id.values[0]
    return lims_id


def bsid_to_oeid(behavior_session_id):
    '''
    convert a behavior_session_id to an ophys_experiment_id
    '''
    warn_str = gen_depr_str.format('from_lims',
                                   'get_ophys_experiment_ids_for_behavior_session_id')
    warnings.warn(warn_str)

    oeid = db.lims_query(
        '''
        select oe.id
        from behavior_sessions
        join ophys_experiments oe on oe.ophys_session_id = behavior_sessions.ophys_session_id
        where behavior_sessions.id = {}
        '''.format(behavior_session_id)
    )
    if isinstance(oeid, pd.DataFrame):
        return oeid.iloc[0][0]
    else:
        return oeid


def get_cache():
    MANIFEST_PATH = "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/2020_cache/production_cache/manifest.json"
    cache = bpc.from_lims(manifest=MANIFEST_PATH)
    return cache


def is_ophys(behavior_session_id):
    cache = get_cache()

    behavior_session_table = cache.get_behavior_session_table()

    return pd.notnull(behavior_session_table.loc[behavior_session_id]['ophys_session_id'])


def get_sdk_session(behavior_session_id, is_ophys):

    if is_ophys:
        ophys_experiment_id = bsid_to_oeid(behavior_session_id)
        return BehaviorOphysSession.from_lims(ophys_experiment_id)
    else:
        return BehaviorSession.from_lims(behavior_session_id)

# getting filepaths for well known files from LIMS


def get_filepath_from_wkf_info(wkf_info):
    """takes a RealDictRow object returned by one of the wkf_info functions
    and parses it to return the filepath to the well known file.

    Args:
        wkf_info ([type]): [description]

    Returns:
        [type]: [description]
    """
    warn_str = gen_depr_str.format('from_lims_utilities',
                                   'get_filepath_from_realdict_object')
    warnings.warn(warn_str)

    filepath = wkf_info[0]['?column?']  # idk why it's ?column? but it is :(
    filepath = filepath.replace('/allen', '//allen')  # works with windows and linux filepaths
    return filepath


def get_wkf_timeseries_ini_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the timeseries_XYT.ini file
        for a given ophys session. only works for sessions *from a Scientifica rig*

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session id

    Returns:

    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_timeseries_ini_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT wkf.storage_directory || wkf.filename
    FROM well_known_files wkf
    JOIN well_known_file_types wkft ON wkft.id=wkf.well_known_file_type_id
    JOIN specimens sp ON sp.id=wkf.attachable_id
    JOIN ophys_sessions os ON os.specimen_id=sp.id
    WHERE wkft.name = 'SciVivoMetadata'
    AND wkf.storage_directory LIKE '%ophys_session_{0}%'
    AND os.id = {0}

    '''.format(ophys_session_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())
    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath

#
# def pmt_gain_from_timeseries_ini(timeseries_ini_path):
#     """parses the timeseries ini file (scientifica experiments only)
#         and extracts the pmt gain setting
#
#     Arguments:
#         timeseries_ini_path {[type]} -- [description]
#
#     Returns:
#         int -- int of the pmt gain
#     """
#     config.read(timeseries_ini_path)
#     pmt_gain = int(float(config['_']['PMT.2']))
#     return pmt_gain


def get_wkf_dff_h5_filepath(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        dff traces h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the dff.h5 file
                    for the given ophys_experiment_id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_dff_traces_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 514173073 AND
    attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())
    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_roi_trace_h5_filepath(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        roi_traces.h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the roi_traces.h5 file
                    for the given ophys_experiment_id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_roi_traces_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 514173076 AND
    attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())
    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_neuropil_trace_h5_filepath(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        neuropil_traces.h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the neuropil_traces.h5 file
                    for the given ophys_experiment_id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_neuropil_traces_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 514173078 AND
    attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())
    filepath = get_filepath_from_wkf_info(wkf_storage_info)

    return filepath


def get_wkf_extracted_trace_h5_filepath(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        neuropil_traces.h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the neuropil_traces.h5 file
                    for the given ophys_experiment_id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_extracted_traces_input_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 486797213 AND
    attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())
    filepath = get_filepath_from_wkf_info(wkf_storage_info)

    return filepath


def get_wkf_demixed_traces_h5_filepath(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        roi_traces.h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the roi_traces.h5 file
                    for the given ophys_experiment_id
    """

    warn_str = gen_depr_str.format('from_lims',
                                   'get_demixed_traces_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 820011707 AND
    attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())
    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_events_h5_filepath(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        _event.h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the event.h5 file
        for the given ophys_experiment_id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_event_trace_filepath')
    warnings.warn(warn_str)

    QUERY = '''
        SELECT storage_directory || filename
        FROM well_known_files
        WHERE well_known_file_type_id = 1074961818 AND
        attachable_id = {0}

        '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())
    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_motion_corrected_movie_h5_filepath(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get the
        "motion_corrected_movie.h5" information for a given
        ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_motion_corrected_movie_filepath')
    warnings.warn(warn_str)
    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 886523092 AND
     attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())
    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_rigid_motion_transform_csv_filepath(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get the
        "rigid_motion_transform.csv" information for a given
        ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'load_rigid_motion_transform')
    warnings.warn(warn_str)

    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 514167000 AND
     attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())
    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_session_pkl_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        session pkl file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_stimulus_pkl_filepath')
    warnings.warn(warn_str)

    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 610487715 AND
     attachable_id = {0}

    '''.format(ophys_session_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())

    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_session_h5_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        session h5 file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_session_h5_filepath')
    warnings.warn(warn_str)

    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 610487713 AND
     attachable_id = {0}

    '''.format(ophys_session_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())

    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_behavior_avi_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        video-0 avi (behavior video) file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_behavior_avi_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 695808672 AND
    attachable_id = {0}

    '''.format(ophys_session_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())

    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_behavior_h5_filepath(ophys_session_id):
    warn_str = gen_depr_str.format('from_lims',
                                   'get_behavior_h5_filepath')
    warnings.warn(warn_str)

    avi_filepath = get_wkf_behavior_avi_filepath(ophys_session_id)
    h5_filepath = avi_filepath[:-3] + "h5"
    return h5_filepath


def get_wkf_eye_tracking_avi_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        video-1 avi (eyetracking video) file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_eye_tracking_avi_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 695808172 AND
    attachable_id = {0}

    '''.format(ophys_session_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())

    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_eye_tracking_h5_filepath(ophys_session_id):
    avi_filepath = get_wkf_eye_tracking_avi_filepath(ophys_session_id)
    h5_filepath = avi_filepath[:-3] + "h5"
    return h5_filepath


def get_wkf_ellipse_h5_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        ellipse.h5 file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_ellipse_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 914623492 AND
    attachable_id = {0}

    '''.format(ophys_session_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())

    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_platform_json_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        platform.json file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_platform_json_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 746251277 AND
    attachable_id = {0}

    '''.format(ophys_session_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())

    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_screen_mapping_h5_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        screen mapping .h5 file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_screen_mapping_h5_filepath')
    warnings.warn(warn_str)

    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 916857994 AND
     attachable_id = {0}

    '''.format(ophys_session_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())

    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_wkf_deepcut_h5_filepath(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
        screen mapping .h5 file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_deepcut_h5_filepath')
    warnings.warn(warn_str)

    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 990460508 AND
     attachable_id = {0}

    '''.format(ophys_session_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    wkf_storage_info = (lims_cursor.fetchall())
    filepath = get_filepath_from_wkf_info(wkf_storage_info)
    return filepath


def get_cell_timeseries_dict(session, cell_specimen_id):
    '''
    for a given cell_specimen ID, this function creates a dictionary with the following keys
    * timestamps: ophys timestamps
    * cell_roi_id
    * cell_specimen_id
    * dff
    * events
    * filtered events
    This is useful for generating a tidy dataframe

    arguments:
        session object
        cell_specimen_id

    returns
        dict

    '''
    cell_dict = {
        'timestamps': session.ophys_timestamps,
        'cell_roi_id': [session.dff_traces.loc[cell_specimen_id]['cell_roi_id']] * len(session.ophys_timestamps),
        'cell_specimen_id': [cell_specimen_id] * len(session.ophys_timestamps),
        'dff': session.dff_traces.loc[cell_specimen_id]['dff'],
        'events': session.events.loc[cell_specimen_id]['events'],
        'filtered_events': session.events.loc[cell_specimen_id]['filtered_events'],

    }

    return cell_dict


def build_tidy_cell_df(session):
    '''
    builds a tidy dataframe describing activity for every cell in session containing the following columns
    * timestamps: the ophys timestamps
    * cell_roi_id: the cell roi id
    * cell_specimen_id: the cell specimen id
    * dff: measured deltaF/F for every timestep
    * events: extracted events for every timestep
    * filtered events: filtered events for every timestep

    Takes a few seconds to build

    arguments:
        session

    returns:
        pandas dataframe
    '''
    return pd.concat([pd.DataFrame(get_cell_timeseries_dict(session, cell_specimen_id)) for cell_specimen_id in session.dff_traces.reset_index()['cell_specimen_id']]).reset_index(drop=True)


def correct_filepath(filepath):
    """using the pathlib python module, takes in a filepath from an
    arbitrary operating system and returns a filepath that should work
    for the users operating system

    Parameters
    ----------
    filepath : string
        given filepath

    Returns
    -------
    string
        filepath adjusted for users operating system
    """
    filepath = filepath.replace('/allen', '//allen')
    corrected_path = Path(filepath)
    return corrected_path


def correct_dataframe_filepath(dataframe, column_string):
    """applies the correct_filepath function to a given dataframe
    column, replacing the filepath in that column in place


    Parameters
    ----------
    dataframe : table
        pandas dataframe with the column
    column_string : string
        the name of the column that contains the filepath to be
        replaced

    Returns
    -------
    dataframe
        returns the input dataframe with the filepath in the given
        column 'corrected' for the users operating system, in place
    """
    dataframe[column_string] = dataframe[column_string].apply(lambda x: correct_filepath(x))
    return dataframe


# functions to annotate experiments table with conditions to use for platform paper analysis ######

def add_session_number_to_experiment_table(experiments):
    # add session number column, extracted frrom session_type
    experiments['session_number'] = [int(session_type[6]) if 'OPHYS' in session_type else None for session_type in
                                     experiments.session_type.values]
    return experiments


def add_experience_level_to_experiment_table(experiments):
    """
    adds a column to ophys_experiment_table that contains a string indicating whether a session had
    exposure level of Familiar, Novel 1, or Novel >1, based on session number and prior_exposure_to_image_set
    """
    # add experience_level column with strings indicating relevant conditions
    experiments['experience_level'] = 'None'

    familiar_indices = experiments[experiments.session_number.isin([1, 2, 3])].index.values
    experiments.loc[familiar_indices, 'experience_level'] = 'Familiar'

    novel_indices = experiments[(experiments.session_number == 4) &
                                (experiments.prior_exposures_to_image_set == 0)].index.values
    experiments.loc[novel_indices, 'experience_level'] = 'Novel 1'

    novel_greater_than_1_indices = experiments[(experiments.session_number.isin([4, 5, 6])) &
                                               (experiments.prior_exposures_to_image_set != 0)].index.values
    experiments.loc[novel_greater_than_1_indices, 'experience_level'] = 'Novel >1'

    return experiments


def add_experience_exposure_column(experiments_table):
    """
    adds a column to ophys_experiment_table that contains a string indicating the experience level and
    image set exposure number for Novel sessions, and experience level and prior omissions exposure for familiar sessions
    """
    experience_exposure_list = []
    for experiment_id in experiments_table.index.values:
        expt = experiments_table.loc[experiment_id]
        if 'Familiar' in expt.experience_level:
            if expt.prior_exposures_to_omissions <= 3:
                exp = 'Familiar ' + str(int(expt.prior_exposures_to_omissions))
            else:
                exp = 'Familiar > 3'
            experience_exposure_list.append(exp)
        elif 'Novel' in expt.experience_level:
            if expt.prior_exposures_to_image_set <= 3:
                exp = 'Novel ' + str(int(expt.prior_exposures_to_image_set))
            else:
                exp = 'Novel > 3'
            experience_exposure_list.append(exp)
    experiments_table.loc[:, 'experience_exposure'] = experience_exposure_list
    return experiments_table


def get_experience_level_colors():
    """
    get color map corresponding to Familiar, Novel 1 and Novel >1
    Familiar = red
    Novel 1 = blue
    Novel >1 = lighter blue
    """
    import seaborn as sns

    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    purples = sns.color_palette('Purples_r', 6)[:5][::2]

    colors = [reds[0], blues[0], purples[0]]

    return colors


def add_passive_flag_to_ophys_experiment_table(experiments):
    """
    adds a column to ophys_experiment_table that contains a Boolean indicating whether a session was
    passive or not based on session number
    """
    experiments['passive'] = 'None'

    passive_indices = experiments[experiments.session_number.isin([2, 5])].index.values
    experiments.loc[passive_indices, 'passive'] = True

    active_indices = experiments[experiments.session_number.isin([2, 5]) == False].index.values
    experiments.loc[active_indices, 'passive'] = False

    return experiments


def add_passive_to_engagement_state_column(df):
    """
    adds a column to any df that contains a string indicating whether
    the row values correspond to engaged, disengaged, or passive conditions.
    input dataframe must contain already a column called 'engagement_state', and column 'session_number'
    this function just makes sure that any passive sessions are labeled 'passive' rather than 'disengaged'
    """

    passive_indices = df[df.session_number.isin([2, 5])].index.values
    df.loc[passive_indices, 'engagement_state'] = 'passive'

    return df


def get_engagement_state_order(df):
    """
    returns preferred order of engagement_state column
    must pass df containing a column 'engagement_state' in order to determine what values are present
    """
    if 'engagement_state' not in df.keys():
        print('there is no engagement_state in this df, must provide a df with engagement_state column to evaluate')
    elif 'passive' in df.engagement_state.unique():
        order = ['engaged', 'disengaged', 'passive']
    else:
        order = ['engaged', 'disengaged']

    return order


def add_cell_type_column(df):
    """
    adds a column with abbreviated version of cre_line, i.e. Vip, Sst, Exc
    """
    cre_indices = df[df.cre_line == 'Vip-IRES-Cre'].index.values
    df.loc[cre_indices, 'cell_type'] = 'Vip Inhibitory'

    cre_indices = df[df.cre_line == 'Sst-IRES-Cre'].index.values
    df.loc[cre_indices, 'cell_type'] = 'Sst Inhibitory'

    cre_indices = df[df.cre_line == 'Slc17a7-IRES2-Cre'].index.values
    df.loc[cre_indices, 'cell_type'] = 'Excitatory'

    return df


def add_binned_depth_column(df):
    """
    for a dataframe with column 'imaging_depth', bin the depth values into 100um bins and assign the mean depth for each bin
    :param df:
    :return:
    """
    df = df.copy()
    df.loc[:, 'depth'] = None

    indices = df[(df.imaging_depth < 100)].index.values
    df.loc[indices, 'depth'] = 75

    indices = df[(df.imaging_depth < 200) &
                 (df.imaging_depth >= 100)].index.values
    df.loc[indices, 'depth'] = 150

    indices = df[
        (df.imaging_depth >= 200) & (df.imaging_depth < 300)].index.values
    df.loc[indices, 'depth'] = 250

    indices = df[
        (df.imaging_depth >= 300) & (df.imaging_depth < 400)].index.values
    df.loc[indices, 'depth'] = 350

    return df


def add_first_novel_column(df):
    """
    Adds a column called 'first_novel' that indicates (with a Boolean) whether a session is the first true novel image session or not
    Equivalent to experience_level == 'Novel 1'
    """
    df.loc[:, 'first_novel'] = False
    indices = df[(df.session_number == 4) & (df.prior_exposures_to_image_set == 0)].index.values
    df.loc[indices, 'first_novel'] = True
    return df


def get_n_relative_to_first_novel(group):
    """
    Function to apply to experiments_table data grouped on 'ophys_container_id'
    For each container, determines the numeric order of sessions relative to the first novel image session
    returns a pandas Series with column 'n_relative_to_first_novel' indicating this value for all session in the container
    If the container does not have a truly novel session, all values are set to NaN
    """
    group = group.sort_values(by='date_of_acquisition')  # must sort for relative ordering to be accurate
    if 'Novel 1' in group.experience_level.values:
        novel_ind = np.where(group.experience_level == 'Novel 1')[0][0]
        n_relative_to_first_novel = np.arange(-novel_ind, len(group) - novel_ind, 1)
    else:
        n_relative_to_first_novel = np.empty(len(group))
        n_relative_to_first_novel[:] = np.nan
    return pd.Series({'n_relative_to_first_novel': n_relative_to_first_novel})


def add_n_relative_to_first_novel_column(df):
    """
    Add a column called 'n_relative_to_first_novel' that indicates the session number relative to the first novel session for each experiment in a container.
    If a container does not have a first novel session, the value of n_relative_to_novel for all experiments in the container is NaN.
    Input df must have column 'experience_level'
    Input df is typically ophys_experiment_table
    """
    df = df.sort_values(by=['ophys_container_id', 'date_of_acquisition'])  # must sort for ordering to be accurate
    numbers = df.groupby('ophys_container_id').apply(get_n_relative_to_first_novel)
    df['n_relative_to_first_novel'] = np.nan
    for container_id in df.ophys_container_id.unique():
        indices = df[df.ophys_container_id == container_id].index.values
        df.loc[indices, 'n_relative_to_first_novel'] = list(numbers.loc[container_id].n_relative_to_first_novel)
    return df


def add_last_familiar_column(df):
    """
    adds column to df called 'last_familiar' which indicates (with a Boolean) whether
    a session is the last familiar image session prior to the first novel session for each container
    If a container has no truly first novel session, all sessions are labeled as NaN
    input df must have 'experience_level' and 'n_relative_to_first_novel'
    """
    df['last_familiar'] = False
    indices = df[(df.n_relative_to_first_novel == -1) & (df.experience_level == 'Familiar')].index.values
    df.loc[indices, 'last_familiar'] = True
    return df


def get_last_familiar_active(group):
    """
    Function to apply to experiments_table data grouped by 'ophys_container_id'
    determines whether each session in the container was the last active familiar image session prior to the first novel session
    input df must have column 'n_relative_to_first_novel'
    """
    group = group.sort_values(by='date_of_acquisition')
    last_familiar_active = np.empty(len(group))
    last_familiar_active[:] = False
    indices = np.where((group.passive == False) & (group.n_relative_to_first_novel < 0))[0]
    if len(indices) > 0:
        index = indices[-1]  # use last (most recent) index
        last_familiar_active[index] = True
    return pd.Series({'last_familiar_active': last_familiar_active})


def add_last_familiar_active_column(df):
    """
    Adds a column 'last_familiar_active' that indicates (with a Boolean) whether
    a session is the last active familiar image session prior to the first novel session in each container
    If a container has no truly first novel session, all sessions are labeled as NaN
    input df must have 'experience_level' and 'n_relative_to_first_novel'
    """
    df = df.sort_values(by=['ophys_container_id', 'date_of_acquisition'])
    values = df.groupby('ophys_container_id').apply(get_last_familiar_active)
    df['last_familiar_active'] = False
    for container_id in df.ophys_container_id.unique():
        indices = df[df.ophys_container_id == container_id].index.values
        df.loc[indices, 'last_familiar_active'] = list(values.loc[container_id].last_familiar_active)
    # change to boolean
    df.loc[df[df.last_familiar_active == 0].index.values, 'last_familiar_active'] = False
    df.loc[df[df.last_familiar_active == 1].index.values, 'last_familiar_active'] = True
    return df


def add_second_novel_column(df):
    """
    Adds a column called 'second_novel' that indicates (with a Boolean) whether a session
    was the second passing novel image session after the first truly novel session, including passive sessions.
    If a container has no truly first novel session, all sessions are labeled as NaN
    input df must have 'experience_level' and 'n_relative_to_first_novel'
    """
    df['second_novel'] = False
    indices = df[(df.n_relative_to_first_novel == 1) & (df.experience_level == 'Novel >1')].index.values
    df.loc[indices, 'second_novel'] = True
    return df


def get_second_novel_active(group):
    """
    Function to apply to experiments_table data grouped by 'ophys_container_id'
    determines whether each session in the container was the second passing novel image session
    after the first novel session, and was an active behavior session
    input df must have column 'n_relative_to_first_novel'
    """
    group = group.sort_values(by='date_of_acquisition')
    second_novel_active = np.empty(len(group))
    second_novel_active[:] = False
    indices = np.where((group.passive == False) & (group.n_relative_to_first_novel > 0))[0]
    if len(indices) > 0:
        index = indices[0]  # use first (most recent) index
        second_novel_active[index] = True
    return pd.Series({'second_novel_active': second_novel_active})


def add_second_novel_active_column(df):
    """
    Adds a column called 'second_novel_active' that indicates (with a Boolean) whether a session
    was the second passing novel image session after the first truly novel session, and was an active behavior session.
    If a container has no truly first novel session, all sessions are labeled as NaN
    input df must have 'experience_level' and 'n_relative_to_first_novel'
    """
    df = df.sort_values(by=['ophys_container_id', 'date_of_acquisition'])
    values = df.groupby('ophys_container_id').apply(get_second_novel_active)
    df['second_novel_active'] = False
    for container_id in df.ophys_container_id.unique():
        indices = df[df.ophys_container_id == container_id].index.values
        df.loc[indices, 'second_novel_active'] = list(values.loc[container_id].second_novel_active)
    # change to boolean
    df.loc[df[df.second_novel_active == 0].index.values, 'second_novel_active'] = False
    df.loc[df[df.second_novel_active == 1].index.values, 'second_novel_active'] = True
    return df


def get_containers_with_all_experience_levels(experiments_table):
    """
    identifies containers with all 3 experience levels in ['Familiar', 'Novel 1', 'Novel >1']
    """
    experience_level_counts = experiments_table.groupby(['ophys_container_id', 'experience_level']).count().reset_index().groupby(['ophys_container_id']).count()[['experience_level']]
    containers_with_all_experience_levels = experience_level_counts[experience_level_counts.experience_level==3].index.unique()
    return containers_with_all_experience_levels


def limit_to_get_containers_with_all_experience_levels(experiments_table):
    """
    returns experiment_table limited to containers with all 3 experience levels in ['Familiar', 'Novel 1', 'Novel >1']
    """
    containers_with_all_experience_levels = get_containers_with_all_experience_levels(experiments_table)
    experiments_table = experiments_table[experiments_table.ophys_container_id.isin(containers_with_all_experience_levels)]
    return experiments_table

#
# def get_matched_cells_for_set_of_conditions(ophys_experiment_table, ophys_cells_table, column_name, column_values):
#     """
#     Adds a column 'image_set' to the experiment_table, determined based on the image set listed in the session_type column string
#     """
#     experiment_table['image_set'] = [session_type[15] for session_type in experiment_table.session_type.values]
#     return experiment_table
