"""
Created on Tue Oct 20 13:49:10 2015

@author: jeromel
modified by @marinag for incorporation in visual_behavior_ophys

"""
import os
import json
import pandas as pd
from pytz import timezone

from allensdk.internal.api import PostgresQueryMixin
from allensdk.core.authentication import credential_injector
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP

import logging

logger = logging.getLogger(__name__)


def convert_lims_path(data_folder):
    # We need to convert internal storage path to real path on titan
    data_folder = data_folder.replace('/projects', '/allen/programs/braintv/production')
    data_folder = data_folder.replace('/vol1', '')

    if os.name == 'nt':
        data_folder = data_folder.replace('/', '\\')
        if data_folder[0:2] == '\\a':
            data_folder = '\\' + data_folder
    return data_folder


class LimsDatabase:
    def __init__(self, lims_id):

        self.lims_id = lims_id

        # We first gather all information from LIMS
        try:
            api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)(PostgresQueryMixin)())
            conn = api.get_connection()
            cur = conn.cursor()

            query = ' '.join((
                "SELECT oe.id, oe.name, oe.storage_directory, os.specimen_id",
                ", sp.external_specimen_name, os.date_of_acquisition, u.login as operator",
                ", e.name as rig, id.depth, st.acronym, os.parent_session_id, oe.workflow_state",
                ", im1.jp2, im2.jp2, p.code, os.stimulus_name, os.storage_directory, os.id",
                "FROM ophys_experiments oe",
                "JOIN ophys_sessions os ON oe.ophys_session_id = os.id",
                "JOIN specimens sp ON sp.id=os.specimen_id",
                "LEFT JOIN imaging_depths id ON id.id=os.imaging_depth_id",
                "LEFT JOIN equipment e ON e.id=os.equipment_id",
                "LEFT JOIN users u ON u.id=os.operator_id",
                "LEFT JOIN structures st ON st.id=oe.targeted_structure_id",
                "LEFT JOIN images im1 ON oe.averaged_surface_image_id = im1.id",
                "LEFT JOIN images im2 ON oe.averaged_depth_image_id = im2.id",
                "JOIN projects p ON p.id = os.project_id",
                "WHERE oe.id='{}'",
            ))

            cur.execute(query.format(self.lims_id))

            lims_data = cur.fetchall()
            if lims_data == []:
                self.data_present = False
            else:
                self.data_pointer = lims_data[0]
                self.data_present = True

            conn.close()
        except Exception as e:
            logger.error("Unable to query LIMS database: {}".format(e))
            self.data_present = False

    def is_valid(self):
        return self.data_present

    def get_qc_param(self):
        if not (hasattr(self, 'qc_data')):
            qc_data = pd.DataFrame()

            qc_data['lims_id'] = [self.get_lims_id()]
            qc_data['session_id'] = [self.get_session_id()]
            qc_data['parent_session_id'] = [self.get_parent()]
            qc_data['specimen_id'] = [self.get_specimen_id()]
            qc_data['external_specimen_id'] = [self.get_external_specimen_id()]
            qc_data['experiment_date'] = [self.get_experiment_date()]
            qc_data['experiment_name'] = [self.get_experiment_name()]
            qc_data['specimen_driver_line'] = [self.get_specimen_driver_line()]
            qc_data['specimen_reporter_line'] = [self.get_specimen_reporter_line()]
            qc_data['structure'] = [self.get_structure()]
            qc_data['depth'] = [self.get_depth()]
            qc_data['operator'] = [self.get_operator()]
            qc_data['rig'] = [self.get_rig()]
            qc_data['project_id'] = [self.get_project_id()]
            qc_data['datafolder'] = [self.get_datafolder()]
            qc_data['session_datafolder'] = [self.get_datafolder()]
            # qc_data['stimulus_type'] = [self.get_stimulus_type()]
            # qc_data['workflow_state'] = [self.get_workflow_state()]

            # We save the qc internally
            self.qc_data = qc_data

        return self.qc_data

    def get_lims_id(self):
        return self.lims_id

    def get_specimen_id(self):
        return self.data_pointer[3]

    def get_session_id(self):
        return self.data_pointer[17]

    def get_parent(self):
        return self.data_pointer[10]

    def get_project_id(self):
        return self.data_pointer[14]

    def get_external_specimen_id(self):
        return self.data_pointer[4]

    def get_experiment_name(self):
        return self.data_pointer[1]

    def get_experiment_date(self):

        utc_time = self.data_pointer[5]
        zoned_time = utc_time.replace(tzinfo=timezone('UTC'))
        correct_time = zoned_time.astimezone(timezone('US/Pacific'))
        return correct_time.replace(tzinfo=None)

    def get_structure(self):
        return self.data_pointer[9]

    def get_depth(self):
        return self.data_pointer[8]

    def get_stimulus_type(self):
        return self.data_pointer[15]

    def get_operator(self):
        return self.data_pointer[6]

    def get_rig(self):
        return self.data_pointer[7]

    def get_workflow_state(self):
        return self.data_pointer[11]

    def get_specimen_driver_line(self):

        try:
            api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)(PostgresQueryMixin)())
            conn = api.get_connection()
            cur = conn.cursor()

            query = ' '.join((
                "SELECT g.name as driver_line",
                "FROM ophys_experiments oe",
                "JOIN ophys_sessions os ON oe.ophys_session_id = os.id",
                "JOIN specimens sp ON sp.id=os.specimen_id",
                "JOIN donors d ON d.id=sp.donor_id",
                "JOIN donors_genotypes dg ON dg.donor_id=d.id",
                "JOIN genotypes g ON g.id=dg.genotype_id",
                "JOIN genotype_types gt ON gt.id=g.genotype_type_id AND gt.name = 'driver'",
                "WHERE oe.id=%s",
            ))

            cur.execute(query, [self.lims_id])
            genotype_data = cur.fetchall()

            final_genotype = ''
            link_string = ''
            for local_text in genotype_data:
                local_gene = local_text[0]
                final_genotype = local_gene + link_string + final_genotype
                link_string = ';'

            conn.close()

        except Exception as e:
            logger.error("cannot query specimen driver line: {}".format(e))
            final_genotype = ''

        return final_genotype

    def get_specimen_reporter_line(self):

        try:
            api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)(PostgresQueryMixin)())
            conn = api.get_connection()
            cur = conn.cursor()

            query = ' '.join((
                "SELECT g.name as reporter_line",
                "FROM ophys_experiments oe",
                "JOIN ophys_sessions os ON oe.ophys_session_id = os.id",
                "JOIN specimens sp ON sp.id=os.specimen_id",
                "JOIN donors d ON d.id=sp.donor_id",
                "JOIN donors_genotypes dg ON dg.donor_id=d.id",
                "JOIN genotypes g ON g.id=dg.genotype_id",
                "JOIN genotype_types gt ON gt.id=g.genotype_type_id AND gt.name = 'reporter'",
                "WHERE oe.id=%s",
            ))

            cur.execute(query, [self.lims_id])
            genotype_data = cur.fetchall()

            final_genotype = ''
            link_string = ''
            for local_text in genotype_data:
                local_gene = local_text[0]
                final_genotype = final_genotype + link_string + local_gene
                link_string = ';'

            conn.close()

        except Exception as e:
            logger.error("cannot query specimen reporter line: {}".format(e))
            final_genotype = ''

        return final_genotype

    def get_datafolder(self):
        if not (hasattr(self, 'data_folder')):
            data_folder = self.data_pointer[2]

            data_folder = convert_lims_path(data_folder)

            self.data_folder = data_folder

        return self.data_folder

    def get_session_datafolder(self):
        if not (hasattr(self, 'session_data_folder')):
            data_folder = self.data_pointer[16]
            data_folder = convert_lims_path(data_folder)
            self.session_data_folder = data_folder

        return self.session_data_folder

    def get_surface_image_file(self):
        return self.data_pointer[12]

    def get_depth_image_file(self):
        return self.data_pointer[13]

    def get_qc_report_kv_pairs(self, mode):
        if mode == u'standard':
            return [(u'Exp ID', self.get_lims_id()),
                    (u'Exp Name', self.get_experiment_name()),
                    (u'Stimulus Type', self.get_stimulus_type()),
                    (u'Mouse ID', self.get_external_specimen_id()),
                    (u'Driver Line', self.get_specimen_driver_line()),
                    (u'Exp Date', self.get_experiment_date()),
                    (u'Rig', self.get_rig()),
                    (u'Operator', self.get_operator()),
                    (u'Project Code', self.get_project_id())]
        elif mode == u'isi_target':
            return [(u'ISI', self.get_isi()[0]),
                    (u'Depth', self.get_depth()),
                    (u'Parent Expt', self.get_parent()),
                    (u'Structure', self.get_structure())]
        else:
            return None

    def get_json_info(self):
        session_folder = self.get_session_datafolder()

        for file in os.listdir(session_folder):
            if file.endswith("_platform.json"):
                full_path = os.path.join(session_folder, file)
                json_point = open(full_path).read()
                json_data = json.loads(json_point)

        return json_data

    def get_all_ophys_lims_columns_names(self, table_name=u'ophys_experiments'):
        api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)(PostgresQueryMixin)())
        conn = api.get_connection()
        cur = conn.cursor()

        query = ' '.join((
            "SELECT column_name ",
            "FROM information_schema.columns",
            "WHERE table_name   = '{}'".format(table_name),
        ))

        cur.execute(query)

        lims_data = cur.fetchall()
        conn.close()
        return lims_data

    def save_qc_param(self, saved_folder):
        self.get_qc_param()
        file_qc = os.path.join(saved_folder, 'lims_database_qcdata.pkl')
        self.qc_data.to_pickle(file_qc)
