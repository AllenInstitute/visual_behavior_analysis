import os
import numpy as np
import pandas as pd
import json
import shutil
from visual_behavior.data_access import loading
from visual_behavior.ophys.io.lims_database import LimsDatabase
from visual_behavior.ophys.sync.sync_dataset import Dataset as SyncDataset
from visual_behavior.ophys.sync.process_sync import filter_digital, calculate_delay
from visual_behavior import database as db

from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc
from allensdk.brain_observatory.behavior.behavior_data_session import BehaviorDataSession
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession

import logging
logger = logging.getLogger(__name__)


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


def get_ophys_session_id_from_ophys_experiment_id(ophys_experiment_id):
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
    from allensdk.internal.api import PostgresQueryMixin

    lims_dbname = os.environ["LIMS_DBNAME"]
    lims_user = os.environ["LIMS_USER"]
    lims_host = os.environ["LIMS_HOST"]
    lims_password = os.environ["LIMS_PASSWORD"]
    lims_port = os.environ["LIMS_PORT"]
    api = PostgresQueryMixin(dbname=lims_dbname, user=lims_user, host=lims_host, password=lims_password, port=lims_port)

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

    lims_df = pd.read_sql(query, api.get_connection())
    return lims_df.matching_dir.values[0]


def get_ssim(img0, img1):
    from skimage.measure import compare_ssim as ssim
    ssim_pair = ssim(img0, img1, gaussian_weights=True)
    return ssim_pair


def get_lims_data(lims_id):
    ld = LimsDatabase(lims_id)
    lims_data = ld.get_qc_param()
    lims_data.insert(loc=2, column='experiment_id', value=lims_data.lims_id.values[0])
    lims_data.insert(loc=2, column='session_type',
                     value='behavior_' + lims_data.experiment_name.values[0].split('_')[-1])
    lims_data.insert(loc=2, column='ophys_session_dir', value=lims_data.datafolder.values[0][:-28])
    return lims_data


def get_timestamps(lims_data, analysis_dir):
    if '2P6' in analysis_dir:
        use_acq_trigger = True
    else:
        use_acq_trigger = False
    sync_data = get_sync_data(lims_data, analysis_dir, use_acq_trigger)
    timestamps = pd.DataFrame(sync_data)
    return timestamps


def get_sync_path(lims_data, analysis_dir):
    #    import shutil
    ophys_session_dir = get_ophys_session_dir(lims_data)
    # analysis_dir = get_analysis_dir(lims_data)

    # First attempt
    sync_file = [file for file in os.listdir(ophys_session_dir) if 'sync' in file]
    if len(sync_file) > 0:
        sync_file = sync_file[0]
    else:
        json_path = [file for file in os.listdir(ophys_session_dir) if '_platform.json' in file][0]
        with open(os.path.join(ophys_session_dir, json_path)) as pointer_json:
            json_data = json.load(pointer_json)
            sync_file = json_data['sync_file']
    sync_path = os.path.join(ophys_session_dir, sync_file)

    if sync_file not in os.listdir(analysis_dir):
        logger.info('moving %s to analysis dir', sync_file)  # flake8: noqa: E999
        #        print(sync_path, os.path.join(analysis_dir, sync_file))
        try:
            shutil.copy2(sync_path, os.path.join(analysis_dir, sync_file))
        except:  # NOQA E722
            print('shutil.copy2 gave an error perhaps related to copying stat data... passing!')
            pass
    return sync_path


def get_sync_data(lims_data, analysis_dir, use_acq_trigger):
    logger.info('getting sync data')
    sync_path = get_sync_path(lims_data, analysis_dir)
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
        print('resampling mesoscope 2P frame times')
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
        return BehaviorDataSession.from_lims(behavior_session_id)
