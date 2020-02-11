from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

from allensdk.internal.api import PostgresQueryMixin

import configparser as configp  # for parsing scientifica ini files
import os


get_psql_dict_cursor = convert.get_psql_dict_cursor #to load well-known files
config = configp.ConfigParser()

####function inputs
# ophys_experiment_id
# ophys_session_id 
# behavior_session_id 
# ophys_container_id 


################  FROM SDK  ################ # NOQA: E402


def get_sdk_session_obj(ophys_experiment_id):
    """Use LIMS API from SDK to return session object

    Arguments:
        experiment_id {int} -- 9 digit ophys experiment ID 

    Returns:
        session object -- session object from SDK
    """
    api = BehaviorOphysLimsApi(experiment_id)
    session = BehaviorOphysSession(api)
    return session


def get_sdk_max_projection(ophys_experiment_id):
    """ uses SDK to return 2d max projection image of the microscope field of view
   
    
    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID
    
    Returns:
        image -- can be visualized via plt.imshow(max_projection)
    """
    session = get_sdk_session_obj(experiment_id)
    max_projection = session.max_projection
    return max_projection


def get_sdk_ave_projection(ophys_experiment_id):
    """uses SDK to return 2d image of the 2-photon microscope filed of view, averaged
        across the experiment. 
    
    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID 
    
    Returns:
        image -- can be visualized via plt.imshow(ave_projection)
    """
    session = get_sdk_session_obj(experiment_id)
    ave_projection = session.average_projection
    return ave_projection


def get_sdk_segmentation_mask_image(ophys_experiment_id):
    """uses SDK to return an array containing the masks of all cell ROIS
    
    Arguments:
       ophys_experiment_id {int} -- 9 digit ophys experiment ID 
    
    Returns:
       array -- a 2D boolean array
                visualized via plt.imshow(seg_mask_image)
    """
    session = get_sdk_session_obj(experiment_id)
    seg_mask_image = session.segmentation_mask_image
    return seg_mask_image


def get_sdk_roi_masks(ophys_experiment_id):
    """uses sdk to return a dictionary with individual ROI masks for each cell specimen ID. 
    
    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID
    
    Returns:
        dictonary -- keys are cell specimen ids(ints)
                    values are 2d numpy arrays(binary array the size of the motion corrected 2photon FOV where 1's are the ROI/Cell mask).
                    specific cell masks can be visualized via plt.imshow(roi_masks[cell_specimen_id])
    """

    session = get_sdk_session_obj(experiment_id)
    roi_masks = session.get_roi_masks
    return roi_masks



################  FROM LIMS DATABASE  ################ # NOQA: E402

####### EXPERIMENT ####### # NOQA: E402


def get_lims_experiment_info(experiment_id):
    """uses an sqlite query to retrieve data from the lims2 database

    Arguments:
        experiment_id {[type]} -- [description]

    Returns:
        table -- table with the following columns:
                    "experiment_id":
                    "workflow_state":
                    "ophys_session_id":
                    "container_id":
                    "date_of_acquisition":
                    "stage_name":
                    "foraging_id":
                    "mouse_info":
                    "mouse_donor_id":
                    "targeted_structure":
                    "depth":
                    "rig":

    """
    experiment_id = int(experiment_id)
    mixin = PostgresQueryMixin()
    # build query
    query = '''
    select

    oe.id as experiment_id,
    oe.workflow_state,
    oe.ophys_session_id,

    container.visual_behavior_experiment_container_id as container_id,

    os.date_of_acquisition,
    os.stimulus_name as stage_name,
    os.foraging_id,
    oe.workflow_state,

    specimens.name as mouse_info,
    specimens.donor_id as mouse_donor_id,
    structures.acronym as targeted_structure,
    imaging_depths.depth,
    equipment.name as rig

    from
    ophys_experiments_visual_behavior_experiment_containers container
    join ophys_experiments oe on oe.id = container.ophys_experiment_id
    join ophys_sessions os on os.id = oe.ophys_session_id
    join specimens on specimens.id = os.specimen_id
    join structures on structures.id = oe.targeted_structure_id
    join imaging_depths on imaging_depths.id = oe.imaging_depth_id
    join equipment on equipment.id = os.equipment_id

    where oe.id = {}'''.format(experiment_id)

    lims_experiment_info = mixin.select(query)

    return lims_experiment_info


####### CONTAINER ####### # NOQA: E402


def get_lims_container_info(ophys_container_id):
    """"uses an sqlite query to retrieve data from the lims2 database

    Arguments:
        container_id {[type]} -- [description]

    Returns:
       table -- table with the following columns:
                    "container_id":
                    "ophys_experiment_id":
                    "ophys_session_id":
                    "stage_name":
                    "foraging_id":
                    "workflow_state":
                    "mouse_info":
                    "mouse_donor_id":
                    "targeted_structure":
                    "depth":
                    "rig":
                    "date_of_acquisition":
    """
    container_id = int(ophys_container_id)

    mixin = PostgresQueryMixin()
    # build query
    query = '''
    SELECT
    container.visual_behavior_experiment_container_id as container_id,
    oe.id as ophys_experiment_id,
    oe.ophys_session_id,
    os.stimulus_name as stage_name,
    os.foraging_id,
    oe.workflow_state,
    specimens.name as mouse_info,
    specimens.donor_id as mouse_donor_id,
    structures.acronym as targeted_structure,
    imaging_depths.depth,
    equipment.name as rig,
    os.date_of_acquisition

    FROM
    ophys_experiments_visual_behavior_experiment_containers container
    join ophys_experiments oe on oe.id = container.ophys_experiment_id
    join ophys_sessions os on os.id = oe.ophys_session_id
    join structures on structures.id = oe.targeted_structure_id
    join imaging_depths on imaging_depths.id = oe.imaging_depth_id
    join specimens on specimens.id = os.specimen_id
    join equipment on equipment.id = os.equipment_id

    where
    container.visual_behavior_experiment_container_id ={}'''.format(ophys_container_id)

    lims_container_info = mixin.select(query)
    return lims_container_info




################  FROM WELL KNOWN FILE  ################ # NOQA: E402


def get_timeseries_ini_wkf_info(lims_session_id):
    """use SQL and the LIMS well known file system to get the timeseries_XYT.ini file
        for a given ophys session from a Scientifica rig 

    Arguments:
        lims_session_id {int} -- 9 digit lims_session_id

    Returns:
        list -- nested list with
    """

    QUERY = '''
    SELECT wkf.storage_directory || wkf.filename
    FROM well_known_files wkf
    JOIN well_known_file_types wkft ON wkft.id=wkf.well_known_file_type_id
    JOIN specimens sp ON sp.id=wkf.attachable_id
    JOIN ophys_sessions os ON os.specimen_id=sp.id
    WHERE wkft.name = 'SciVivoMetadata'
    AND wkf.storage_directory LIKE '%ophys_session_{0}%'
    AND os.id = {0}

    '''.format(lims_session_id)

    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    timeseries_ini_wkf_info = (lims_cursor.fetchall())
    return timeseries_ini_wkf_info


def get_timeseries_ini_location(lims_session_id):
    """use SQL and the LIMS well known file system to get info for the timeseries_XYT.ini file
        for a given ophys session, and then parses that information to get the filepath

    Arguments:
        llims_session_id {int} -- 9 digit lims_session_id

    Returns:
        filepath -- [description]
    """
    timeseries_ini_wkf_info = get_timeseries_ini_wkf_info(lims_session_id)
    timeseries_ini_path = timeseries_ini_wkf_info[0]['?column?'] #idk why it's ?column? but it is :(
    timeseries_ini_path = timeseries_ini_path.replace('/allen', '//allen')  # works with windows and linux filepaths
    return timeseries_ini_path


def pmt_gain_from_timeseries_ini(timeseries_ini_path):
    config.read(timeseries_ini_path)
    pmt_gain = int(float(config['_']['PMT.2']))
    return pmt_gain


def get_pmt_gain_for_session(lims_session_id):
    timeseries_ini_path = get_timeseries_ini_location(lims_session_id)
    pmt_gain = pmt_gain_from_timeseries_ini(timeseries_ini_path)
    return pmt_gain

# def get_pmt_gain_for_container(container_id):

