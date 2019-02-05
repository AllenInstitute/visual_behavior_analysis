"""
Created on Tue Oct 20 13:49:10 2015

@author: nicholasc
modified by @marinag from code by jeromel for incorporation in visual_behavior_ophys

"""
import os
import json
import psycopg2
import pandas as pd
from pytz import timezone
import numpy as np
import json
import h5py

import logging
import matplotlib.image as mpimg  # NOQA: E402

logger = logging.getLogger(__name__)

from visual_behavior.ophys.timestamps import get_sync_data, match_dff_and_ophys_timestamp_len
from visual_behavior.ophys.roi_processing import get_roi_metrics, get_roi_masks
from visual_behavior.translator import foraging2, foraging  # NOQA: E402

def one(x):
    assert len(x) == 1
    if isinstance(x,set):
        return list(x)[0]
    else:
        return x[0]


def convert_lims_path(data_folder):
    # We need to convert internal storage path to real path on titan
    data_folder = data_folder.replace('/projects', '/allen/programs/braintv/production')
    data_folder = data_folder.replace('/vol1', '')

    if os.name == 'nt':
        data_folder = data_folder.replace('/', '\\')
        if data_folder[0:2] == '\\a':
            data_folder = '\\' + data_folder
    return data_folder


def correct_time_zone(utc_time):
    zoned_time = utc_time.replace(tzinfo=timezone('UTC'))
    correct_time = zoned_time.astimezone(timezone('US/Pacific'))
    return correct_time.replace(tzinfo=None)



class VisualBehaviorLimsAPI:


    def query(self, q):
        conn = psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2", password="limsro", port=5432)
        cur = conn.cursor()
        cur.execute(q)
        return one(one(cur.fetchall()))

    def get_ophys_experiment_dir(self, obj):
        query = '''
                SELECT oe.storage_directory
                FROM ophys_experiments oe
                WHERE oe.id= {};
                '''
        return self.query(query.format(obj.ophys_experiment_id))

    def get_maxint_file(self, obj):
        query = '''
                SELECT obj.storage_directory || 'maxInt_a13a.png' AS maxint_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files obj ON obj.attachable_id=ocsr.id AND obj.attachable_type = 'OphysCellSegmentationRun' AND obj.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysSegmentationObjects')
                WHERE oe.id= {};
                '''
        return self.query(query.format(obj.ophys_experiment_id))

    def get_sync_file(self, obj):
        query = '''
                SELECT sync.storage_directory || sync.filename AS sync_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN well_known_files sync ON sync.attachable_id=os.id AND sync.attachable_type = 'OphysSession' AND sync.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysRigSync')
                WHERE oe.id= {};
                '''
        return self.query(query.format(obj.ophys_experiment_id))

    def get_input_extract_traces_file(self, obj):
        query = '''
                SELECT oe.storage_directory || 'processed/' || oe.id || '_input_extract_traces.json'
                FROM ophys_experiments oe
                WHERE oe.id= {};
                '''
        return self.query(query.format(obj.ophys_experiment_id))

    def get_objectlist_file(self, obj):
        query = '''
                SELECT obj.storage_directory || obj.filename AS obj_file
                FROM ophys_experiments oe
                LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
                LEFT JOIN well_known_files obj ON obj.attachable_id=ocsr.id AND obj.attachable_type = 'OphysCellSegmentationRun' AND obj.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysSegmentationObjects')
                WHERE oe.id= {};
                '''
        return self.query(query.format(obj.ophys_experiment_id))

    def get_dff_file(self, obj):
        query = '''
                SELECT dff.storage_directory || dff.filename AS dff_file
                FROM ophys_experiments oe
                LEFT JOIN well_known_files dff ON dff.attachable_id=oe.id AND dff.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysDffTraceFile')
                WHERE oe.id= {};
                '''
        return self.query(query.format(obj.ophys_experiment_id))

    def get_stim_file(self, obj):
        query = '''
                SELECT stim.storage_directory || stim.filename AS stim_file
                FROM ophys_experiments oe
                JOIN ophys_sessions os ON oe.ophys_session_id = os.id
                LEFT JOIN well_known_files stim ON stim.attachable_id=os.id AND stim.attachable_type = 'OphysSession' AND stim.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'StimulusPickle')
                WHERE oe.id= {};
                '''
        return self.query(query.format(obj.ophys_experiment_id))


    def get_max_projection(self, obj):
        maxInt_a13_file = self.get_maxint_file(obj)
        max_projection = mpimg.imread(maxInt_a13_file)
        return max_projection


    def get_roi_metrics(self, obj):
        input_extract_traces_file = self.get_input_extract_traces_file(obj)
        objectlist_file = self.get_objectlist_file(obj)
        ophys_experiment_id = obj.ophys_experiment_id
        return get_roi_metrics(input_extract_traces_file, ophys_experiment_id, objectlist_file)['filtered']


    def get_roi_masks(self, obj):
        roi_metrics = roi_metrics = self.get_roi_metrics(obj)
        input_extract_traces_file = self.get_input_extract_traces_file(obj)

        return get_roi_masks(roi_metrics, input_extract_traces_file)


    def get_cell_specimen_ids(self, obj):

        input_extract_traces_file = self.get_input_extract_traces_file(obj)
        with open(input_extract_traces_file, 'r') as w:
            jin = json.load(w)
        return [roi['id'] for roi in jin['rois']]


    def get_sync_data(self, obj):
        sync_path = self.get_sync_file(obj)
        return get_sync_data(sync_path, obj.use_acq_trigger)


    def get_stimulus_timestamps(self, obj):
        return self.get_sync_data(obj)['stimulus_frames']


    def get_core_data(self, obj):
        pkl = pd.read_pickle(self.get_stim_file(obj))
        stimulus_timestamps = self.get_stimulus_timestamps(obj)
        try:
            core_data = foraging.data_to_change_detection_core(pkl, time=stimulus_timestamps)
        except KeyError:
            core_data = foraging2.data_to_change_detection_core(pkl, time=stimulus_timestamps)
        return core_data


    def get_running_speed(self, obj):
        return self.get_core_data(obj)['running']




    def get_dff_traces(self, obj):
        dff_path = self.get_dff_file(obj)
        g = h5py.File(dff_path)
        dff_traces_orig = np.asarray(g['data'])
        g.close()

        ophys_timestamps_orig = self.get_sync_data(obj)['ophys_frames']

        dff_traces, timestamps = match_dff_and_ophys_timestamp_len(dff_traces_orig, ophys_timestamps_orig)

        cell_specimen_id_list = self.get_cell_specimen_ids(obj)
        df = pd.DataFrame({'cell_specimen_id':cell_specimen_id_list, 'dff':list(dff_traces)})
        df['timestamps'] = [timestamps]*len(df)

        return df


from allensdk.experimental.lazy_property import LazyProperty

class TestDataSet(object):

    ophys_experiment_dir = LazyProperty(api_method='get_ophys_experiment_dir')
    maxInt_file = LazyProperty(api_method='get_maxint_file')
    sync_file = LazyProperty(api_method='get_sync_file')
    input_extract_traces_file = LazyProperty(api_method='get_input_extract_traces_file')
    objectlist_file = LazyProperty(api_method='get_objectlist_file')
    dff_file = LazyProperty(api_method='get_dff_file')
    stim_file = LazyProperty(api_method='get_stim_file')


    def __init__(self, ophys_experiment_id, api=None):

        self.ophys_experiment_id = ophys_experiment_id
        self.api = VisualBehaviorLimsAPI() if api is None else api

def test_lims_api(ophys_experiment_id):

    lims_data = TestDataSet(ophys_experiment_id)

    # assert lims_data.ophys_experiment_id == ophys_experiment_id
    # assert lims_data.specimen_id == 652073919
    # assert lims_data.session_id == 702013508
    # assert lims_data.experiment_container_id == 700821114
    # assert lims_data.project_id == 'VisualBehaviorDevelopment'
    # assert lims_data.external_specimen_id == 363887
    # assert lims_data.experiment_name == '20180524_363887_sessionC'
    # assert str(lims_data.experiment_date) == '2018-05-24 14:27:25'
    # assert lims_data.targeted_structure == 'VISal'
    # assert lims_data.imaging_depth == 175
    # assert lims_data.stimulus_type is None
    # assert lims_data.operator == 'saharm'
    # assert lims_data.rig == 'CAM2P.6'
    # assert lims_data.workflow_state == 'qc'
    assert lims_data.ophys_experiment_dir == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/'
    assert lims_data.sync_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/702013508_363887_20180524142941_sync.h5'
    # assert lims_data.specimen_id == 652073919
    assert lims_data.stim_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/702013508_363887_20180524142941_stim.pkl'
    assert 'objectlist.txt' in lims_data.objectlist_file
    assert '_input_extract_traces.json' in lims_data.input_extract_traces_file
    assert lims_data.dff_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/702134928_dff.h5'
    # assert lims_data.demix_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/demix/702134928_demixed_traces.h5'
    # assert lims_data.rigid_motion_transform_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/702134928_rigid_motion_transform.csv'
    assert lims_data.maxInt_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_814561221/maxInt_a13a.png'
    # assert lims_data.roi_traces_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/roi_traces.h5'
    # assert lims_data.avgint_a1X_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_814561221/avgInt_a1X.png'

test_lims_api(702134928)





    # query = '''
    #         SELECT oe.id, oe.name, oe.storage_directory, os.specimen_id
    #         , sp.external_specimen_name, os.date_of_acquisition, u.login as operator
    #         , e.name as rig, id.depth, st.acronym, os.parent_session_id, oe.workflow_state
    #         , im1.jp2, im2.jp2, p.code, os.stimulus_name, os.storage_directory, os.id

    #         ,sync.storage_directory || sync.filename AS sync_file
    #         ,stim.storage_directory || stim.filename AS stim_file
    #         ,dff.storage_directory || dff.filename AS dff_file
    #         ,obj.storage_directory || obj.filename AS obj_file
    #         ,oe.storage_directory || 'processed/' || oe.id || '_input_extract_traces.json' AS trace_input_file
    #         ,tra.storage_directory || tra.filename AS transform_file
    #         ,obj.storage_directory || 'maxInt_a13a.png' AS maxint_file
    #         ,sync.storage_directory || sync.filename AS sync_file
    #         ,sp.id
    #         ,stim.storage_directory || stim.filename AS stim_file
    #         ,oe.storage_directory || 'demix/' || oe.id || '_demixed_traces.h5' AS demix_file
    #         ,roi.storage_directory || roi.filename AS roi_traces_file
    #         ,obj.storage_directory || 'avgInt_a1X.png' AS avgint_file

    #         FROM ophys_experiments oe
    #         JOIN ophys_sessions os ON oe.ophys_session_id = os.id
    #         JOIN specimens sp ON sp.id=os.specimen_id
    #         LEFT JOIN imaging_depths id ON id.id=os.imaging_depth_id
    #         LEFT JOIN equipment e ON e.id=os.equipment_id
    #         LEFT JOIN users u ON u.id=os.operator_id
    #         LEFT JOIN structures st ON st.id=oe.targeted_structure_id
    #         LEFT JOIN images im1 ON oe.averaged_surface_image_id = im1.id
    #         LEFT JOIN images im2 ON oe.averaged_depth_image_id = im2.id
    #         LEFT JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id = oe.id AND ocsr.current = 't'
    #         LEFT JOIN well_known_files sync ON sync.attachable_id=os.id AND sync.attachable_type = 'OphysSession' AND sync.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysRigSync')
    #         LEFT JOIN well_known_files stim ON stim.attachable_id=os.id AND stim.attachable_type = 'OphysSession' AND stim.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'StimulusPickle')
    #         LEFT JOIN well_known_files dff ON dff.attachable_id=oe.id AND dff.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysDffTraceFile')
    #         LEFT JOIN well_known_files obj ON obj.attachable_id=ocsr.id AND obj.attachable_type = 'OphysCellSegmentationRun' AND obj.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysSegmentationObjects')
    #         LEFT JOIN well_known_files tra ON tra.attachable_id=oe.id AND tra.attachable_type = 'OphysExperiment' AND tra.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysMotionXyOffsetData')
    #         LEFT JOIN well_known_files roi ON roi.attachable_id=oe.id AND roi.attachable_type = 'OphysExperiment' AND roi.well_known_file_type_id IN (SELECT id FROM well_known_file_types WHERE name = 'OphysRoiTraces')

    #         JOIN projects p ON p.id = os.project_id
    #         WHERE oe.id= {};
    # '''

    # def __init__(self, ophys_experiment_id, filter_solenoid_2P6=False):
    #     self.ophys_experiment_id = ophys_experiment_id
    #     self.filter_solenoid_2P6 = filter_solenoid_2P6

    #     # Extract data from LIMS:
    #     conn = psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2", password="limsro", port=5432)
    #     cur = conn.cursor()
    #     cur.execute(LimsDatabase.query.format(self.ophys_experiment_id))
    #     data = one(cur.fetchall())

    #     self.experiment_name = data[1]
    #     self.data_folder = data[2]
    #     self.specimen_id = data[3]
    #     self.external_specimen_id = int(data[4])
    #     self.experiment_date = correct_time_zone(data[5])
    #     self.operator = data[6]
    #     self.rig = data[7]
    #     self.imaging_depth = data[8]
    #     self.targeted_structure = data[9]
    #     self.experiment_container_id = data[10]
    #     self.workflow_state = data[11]
    #     self.project_id = data[14]
    #     self.session_id = data[17]
    #     self.stimulus_type = data[15]
    #     self.sync_file = data[25]
    #     self.specimen_id = data[26]
    #     self.stim_file = data[27]
    #     self.dff_file = data[20]
    #     self.curr_objectlist_file = data[21]
    #     self.curr_input_extract_traces_file = data[22]
    #     self.demix_file = data[28]
    #     self.rigid_motion_transform_file = data[23]
    #     self.maxInt_a13_file = data[24]
    #     self.roi_traces_file = data[29]
    #     self.avgint_a1X_file = data[30]

    #     # TODO: replace with marshmallow https://github.com/AllenInstitute/visual_behavior_analysis/issues/452
    #     self.validate()

    # def validate(self):
    #     assert self.rig in ['CAM2P.6']
    #     assert isinstance(self.external_specimen_id, int)
    #     assert isinstance(self.specimen_id, int)

    # @property
    # def processed_dir(self):
    #     return os.path.join(self.data_folder, 'processed')





# """
# Created on Tue Oct 20 13:49:10 2015

# @author: jeromel
# modified by @marinag for incorporation in visual_behavior_ophys

# """
# import os
# import json
# import psycopg2
# import pandas as pd
# from pytz import timezone

# import logging

# logger = logging.getLogger(__name__)


# def convert_lims_path(data_folder):
#     # We need to convert internal storage path to real path on titan
#     data_folder = data_folder.replace('/projects', '/allen/programs/braintv/production')
#     data_folder = data_folder.replace('/vol1', '')

#     if os.name == 'nt':
#         data_folder = data_folder.replace('/', '\\')
#         if data_folder[0:2] == '\\a':
#             data_folder = '\\' + data_folder
#     return data_folder


# class LimsDatabase:
#     def __init__(self, lims_id):

#         self.lims_id = lims_id

#         # We first gather all information from LIMS
#         try:
#             conn = psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2", password="limsro", port=5432)
#             cur = conn.cursor()

#             query = ' '.join((
#                 "SELECT oe.id, oe.name, oe.storage_directory, os.specimen_id",
#                 ", sp.external_specimen_name, os.date_of_acquisition, u.login as operator",
#                 ", e.name as rig, id.depth, st.acronym, os.parent_session_id, oe.workflow_state",
#                 ", im1.jp2, im2.jp2, p.code, os.stimulus_name, os.storage_directory, os.id",
#                 "FROM ophys_experiments oe",
#                 "JOIN ophys_sessions os ON oe.ophys_session_id = os.id",
#                 "JOIN specimens sp ON sp.id=os.specimen_id",
#                 "LEFT JOIN imaging_depths id ON id.id=os.imaging_depth_id",
#                 "LEFT JOIN equipment e ON e.id=os.equipment_id",
#                 "LEFT JOIN users u ON u.id=os.operator_id",
#                 "LEFT JOIN structures st ON st.id=oe.targeted_structure_id",
#                 "LEFT JOIN images im1 ON oe.averaged_surface_image_id = im1.id",
#                 "LEFT JOIN images im2 ON oe.averaged_depth_image_id = im2.id",
#                 "JOIN projects p ON p.id = os.project_id",
#                 "WHERE oe.id='{}'",
#             ))

#             cur.execute(query.format(self.lims_id))

#             lims_data = cur.fetchall()
#             if lims_data == []:
#                 self.data_present = False
#             else:
#                 self.data_pointer = lims_data[0]
#                 self.data_present = True

#             conn.close()
#         except Exception as e:
#             logger.error("Unable to query LIMS database: {}".format(e))
#             self.data_present = False

#     def is_valid(self):
#         return self.data_present

#     def get_qc_param(self):
#         if not (hasattr(self, 'qc_data')):
#             qc_data = pd.DataFrame()

#             qc_data['lims_id'] = [self.get_lims_id()]
#             qc_data['session_id'] = [self.get_session_id()]
#             qc_data['parent_session_id'] = [self.get_parent()]
#             qc_data['specimen_id'] = [self.get_specimen_id()]
#             qc_data['external_specimen_id'] = [self.get_external_specimen_id()]
#             qc_data['experiment_date'] = [self.get_experiment_date()]
#             qc_data['experiment_name'] = [self.get_experiment_name()]
#             qc_data['specimen_driver_line'] = [self.get_specimen_driver_line()]
#             qc_data['specimen_reporter_line'] = [self.get_specimen_reporter_line()]
#             qc_data['structure'] = [self.get_structure()]
#             qc_data['depth'] = [self.get_depth()]
#             qc_data['operator'] = [self.get_operator()]
#             qc_data['rig'] = [self.get_rig()]
#             qc_data['project_id'] = [self.get_project_id()]
#             qc_data['datafolder'] = [self.get_datafolder()]
#             qc_data['session_datafolder'] = [self.get_datafolder()]
#             # qc_data['stimulus_type'] = [self.get_stimulus_type()]
#             # qc_data['workflow_state'] = [self.get_workflow_state()]

#             # We save the qc internally
#             self.qc_data = qc_data

#         return self.qc_data

#     def get_lims_id(self):
#         return self.lims_id

#     def get_specimen_id(self):
#         return self.data_pointer[3]

#     def get_session_id(self):
#         return self.data_pointer[17]

#     def get_parent(self):
#         return self.data_pointer[10]

#     def get_project_id(self):
#         return self.data_pointer[14]

#     def get_external_specimen_id(self):
#         return self.data_pointer[4]

#     def get_experiment_name(self):
#         return self.data_pointer[1]

#     def get_experiment_date(self):

#         utc_time = self.data_pointer[5]
#         zoned_time = utc_time.replace(tzinfo=timezone('UTC'))
#         correct_time = zoned_time.astimezone(timezone('US/Pacific'))
#         return correct_time.replace(tzinfo=None)

#     def get_structure(self):
#         return self.data_pointer[9]

#     def get_depth(self):
#         return self.data_pointer[8]

#     def get_stimulus_type(self):
#         return self.data_pointer[15]

#     def get_operator(self):
#         return self.data_pointer[6]

#     def get_rig(self):
#         return self.data_pointer[7]

#     def get_workflow_state(self):
#         return self.data_pointer[11]

#     def get_specimen_driver_line(self):

#         try:
#             conn = psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2", password="limsro", port=5432)
#             cur = conn.cursor()

#             query = ' '.join((
#                 "SELECT g.name as driver_line",
#                 "FROM ophys_experiments oe",
#                 "JOIN ophys_sessions os ON oe.ophys_session_id = os.id",
#                 "JOIN specimens sp ON sp.id=os.specimen_id",
#                 "JOIN donors d ON d.id=sp.donor_id",
#                 "JOIN donors_genotypes dg ON dg.donor_id=d.id",
#                 "JOIN genotypes g ON g.id=dg.genotype_id",
#                 "JOIN genotype_types gt ON gt.id=g.genotype_type_id AND gt.name = 'driver'",
#                 "WHERE oe.id=%s",
#             ))

#             cur.execute(query, [self.lims_id])
#             genotype_data = cur.fetchall()

#             final_genotype = ''
#             link_string = ''
#             for local_text in genotype_data:
#                 local_gene = local_text[0]
#                 final_genotype = local_gene + link_string + final_genotype
#                 link_string = ';'

#             conn.close()

#         except Exception as e:
#             logger.error("cannot query specimen driver line: {}".format(e))
#             final_genotype = ''

#         return final_genotype

#     def get_specimen_reporter_line(self):

#         try:
#             conn = psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2", password="limsro", port=5432)
#             cur = conn.cursor()

#             query = ' '.join((
#                 "SELECT g.name as reporter_line",
#                 "FROM ophys_experiments oe",
#                 "JOIN ophys_sessions os ON oe.ophys_session_id = os.id",
#                 "JOIN specimens sp ON sp.id=os.specimen_id",
#                 "JOIN donors d ON d.id=sp.donor_id",
#                 "JOIN donors_genotypes dg ON dg.donor_id=d.id",
#                 "JOIN genotypes g ON g.id=dg.genotype_id",
#                 "JOIN genotype_types gt ON gt.id=g.genotype_type_id AND gt.name = 'reporter'",
#                 "WHERE oe.id=%s",
#             ))

#             cur.execute(query, [self.lims_id])
#             genotype_data = cur.fetchall()

#             final_genotype = ''
#             link_string = ''
#             for local_text in genotype_data:
#                 local_gene = local_text[0]
#                 final_genotype = final_genotype + link_string + local_gene
#                 link_string = ';'

#             conn.close()

#         except Exception as e:
#             logger.error("cannot query specimen reporter line: {}".format(e))
#             final_genotype = ''

#         return final_genotype

#     def get_datafolder(self):
#         if not (hasattr(self, 'data_folder')):
#             data_folder = self.data_pointer[2]

#             data_folder = convert_lims_path(data_folder)

#             self.data_folder = data_folder

#         return self.data_folder

#     def get_session_datafolder(self):
#         if not (hasattr(self, 'session_data_folder')):
#             data_folder = self.data_pointer[16]
#             data_folder = convert_lims_path(data_folder)
#             self.session_data_folder = data_folder

#         return self.session_data_folder

#     def get_surface_image_file(self):
#         return self.data_pointer[12]

#     def get_depth_image_file(self):
#         return self.data_pointer[13]

#     def get_qc_report_kv_pairs(self, mode):
#         if mode == u'standard':
#             return [(u'Exp ID', self.get_lims_id()),
#                     (u'Exp Name', self.get_experiment_name()),
#                     (u'Stimulus Type', self.get_stimulus_type()),
#                     (u'Mouse ID', self.get_external_specimen_id()),
#                     (u'Driver Line', self.get_specimen_driver_line()),
#                     (u'Exp Date', self.get_experiment_date()),
#                     (u'Rig', self.get_rig()),
#                     (u'Operator', self.get_operator()),
#                     (u'Project Code', self.get_project_id())]
#         elif mode == u'isi_target':
#             return [(u'ISI', self.get_isi()[0]),
#                     (u'Depth', self.get_depth()),
#                     (u'Parent Expt', self.get_parent()),
#                     (u'Structure', self.get_structure())]
#         else:
#             return None

#     def get_json_info(self):
#         session_folder = self.get_session_datafolder()

#         for file in os.listdir(session_folder):
#             if file.endswith("_platform.json"):
#                 full_path = os.path.join(session_folder, file)
#                 json_point = open(full_path).read()
#                 json_data = json.loads(json_point)

#         return json_data

#     def get_all_ophys_lims_columns_names(self, table_name=u'ophys_experiments'):
#         conn = psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2", password="limsro", port=5432)
#         cur = conn.cursor()

#         query = ' '.join((
#             "SELECT column_name ",
#             "FROM information_schema.columns",
#             "WHERE table_name   = '{}'".format(table_name),
#         ))

#         cur.execute(query)

#         lims_data = cur.fetchall()
#         conn.close()
#         return lims_data

#     def save_qc_param(self, saved_folder):
#         self.get_qc_param()
#         file_qc = os.path.join(saved_folder, 'lims_database_qcdata.pkl')
#         self.qc_data.to_pickle(file_qc)
