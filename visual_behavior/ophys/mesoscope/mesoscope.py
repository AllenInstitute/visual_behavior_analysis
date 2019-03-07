
import psycopg2
import psycopg2.extras
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MesoscopeDataset(object):

    def __init__(self, session_id, experiment_id):

        self.session_id = session_id
        self.experiment_id = experiment_id
        self.data_present = False

        self._database = 'lims2'
        self._host = 'limsdb2'
        self._port = 5432
        self._username = 'limsreader'
        self._password = 'limsro'

        # initialize other attributes here

    def psycopg2_select(self, query):

        connection = psycopg2.connect(host=self._host, port=self._port,
                                      dbname=self._database, user=self._username,
                                      password=self._password,
                                      cursor_factory=psycopg2.extras.RealDictCursor)
        cursor = connection.cursor()

        try:
            cursor.execute(query)
            response = cursor.fetchall()
        finally:
            cursor.close()
            connection.close()

        return response

    def get_mesoscope_session_data(self):

        try:

            query = ' '.join((
                "SELECT oe.id as experiment_id, os.id as session_id",
                ", os.storage_directory as session_folder, oe.storage_directory as experiment_folder",
                ", sp.name as specimen",
                ", os.date_of_acquisition as date",
                ", imaging_depths.depth as depth",
                ", st.acronym as structure",
                ", os.parent_session_id as parent_id",
                ", oe.workflow_state",
                ", os.stimulus_name as stimulus",
                " FROM ophys_experiments oe",
                "join ophys_sessions os on os.id = oe.ophys_session_id "
                "join specimens sp on sp.id = os.specimen_id "
                "join projects p on p.id = os.project_id "
                "join imaging_depths on imaging_depths.id = oe.imaging_depth_id "
                "join structures st on st.id = oe.targeted_structure_id "
                "where p.code = 'MesoscopeDevelopment' and (oe.workflow_state = 'processing' or oe.workflow_state = 'qc') and os.workflow_state ='uploaded' "
                " and os.id='{}'  ",
            ))

            lims_data = self.psycopg2_select(query.format(self.session_id))

            if lims_data == []:
                self.data_present = False
            else:
                self.data_pointer = lims_data
                self.data_present = True

        except Exception as e:
            logger.error("Unable to query LIMS database: {}".format(e))
            self.data_present = False

        return lims_data

    def get_mesoscope_experiment_data(self):

        try:

            query = ' '.join((
                "SELECT oe.id as experiment_id, os.id as session_id",
                ", os.storage_directory as session_folder, oe.storage_directory as experiment_folder",
                ", sp.name as specimen",
                ", os.date_of_acquisition as date",
                ", imaging_depths.depth as depth",
                ", st.acronym as structure",
                ", os.parent_session_id as parent_id",
                ", oe.workflow_state",
                ", os.stimulus_name as stimulus",
                " FROM ophys_experiments oe",
                "join ophys_sessions os on os.id = oe.ophys_session_id "
                "join specimens sp on sp.id = os.specimen_id "
                "join projects p on p.id = os.project_id "
                "join imaging_depths on imaging_depths.id = oe.imaging_depth_id "
                "join structures st on st.id = oe.targeted_structure_id "
                "where p.code = 'MesoscopeDevelopment' and (oe.workflow_state = 'processing' or oe.workflow_state = 'qc') and os.workflow_state ='uploaded' "
                " and oe.id='{}'  ",
            ))

            lims_data = self.psycopg2_select(query.format(self.experiment_id))

            if lims_data == []:
                self.data_present = False
            else:
                self.data_pointer = lims_data
                self.data_present = True

        except Exception as e:
            logger.error("Unable to query LIMS database: {}".format(e))
            self.data_present = False

        return lims_data

    def get_session_folder(self):

        _session = pd.DataFrame(self.get_mesoscope_experiment_data())

        session_folder = _session['session_folder']

        return session_folder.values[0]



