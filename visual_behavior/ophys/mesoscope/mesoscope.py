
import psycopg2
import psycopg2.extras
import pandas as pd
import logging
import json
import tifffile
import numpy as np
import os


logger = logging.getLogger(__name__)


class MesoscopeDataset(object):

    def __init__(self, session_id='', experiment_id=''):

        self.session_id = session_id
        self.experiment_id = experiment_id
        self.data_pointer = None
        self.data_present = False

        self._database = 'lims2'
        self._host = 'limsdb2'
        self._port = 5432
        self._username = 'limsreader'
        self._password = 'limsro'
        self.session_folder = None
        self.splitting_json = None
        self.full_field_path = None
        self.full_field_present = None
        self.data_present = None
        self.full_field_present = None
        self.splitting_json_present = None

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

    def get_mesoscope_session_data(self, session_id=''):
        if session_id != '':
            lims_data = []
            self.session_id = session_id
            try:
                query = ' '.join((
                    "SELECT oe.id as experiment_id, os.id as session_id",
                    ", os.storage_directory as session_folder, oe.storage_directory as experiment_folder",
                    ", sp.name as specimen",
                    ", os.date_of_acquisition as date",
                    ", imaging_depths.depth as depth",
                    ", st.acronym as structure",
                    ", os.≤parent_session_id as parent_id",
                    ", oe.workflow_state",
                    ", os.stimulus_name as stimulus",
                    " FROM ophys_experiments oe",
                    "join ophys_sessions os on os.id = oe.ophys_session_id "
                    "join specimens sp on sp.id = os.specimen_id "
                    "join projects p on p.id = os.project_id "
                    "join imaging_depths on imaging_depths.id = oe.imaging_depth_id "
                    "join structures st on st.id = oe.targeted_structure_id "
                    "where p.code = 'MesoscopeDevelopment' and (oe.workflow_state = 'processing' or oe.workflow_state "
                    "= 'qc') and os.workflow_state ='uploaded' "
                    " and os.id='{}'  ",
                ))

                lims_data = self.psycopg2_select(query.format(session_id))
                if not lims_data:
                    self.data_present = False
                else:
                    self.data_pointer = lims_data
                    self.data_present = True
            except Exception as e:
                logger.error("Unable to query LIMS database: {}".format(e))
                self.data_present = False
        else:
            lims_data = []
            logger.error("Provide Session ID")

        return lims_data

    def get_mesoscope_experiment_data(self, experiment_id=''):
        lims_data = []
        if experiment_id != '':
            self.experiment_id = experiment_id
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
                    "where p.code = 'MesoscopeDevelopment' and (oe.workflow_state = 'processing' or oe.workflow_state "
                    "= 'qc') and os.workflow_state ='uploaded' "
                    " and oe.id='{}'  ",
                ))

                lims_data = self.psycopg2_select(query.format(experiment_id))

                if not lims_data:
                    self.data_present = False
                else:
                    self.data_pointer = lims_data
                    self.data_present = True

            except Exception as e:
                logger.error("Unable to query LIMS database: {}".format(e))
                self.data_present = False
        else:
            logger.error("Provide experiment ID")

        return lims_data

    def get_session_folder(self):

        _session = pd.DataFrame(self.get_mesoscope_session_data(self.session_id))
        self.session_folder = _session['session_folder'].values[0]

        return self.session_folder

    def get_splitting_json(self):

        splitting_json = os.path.join(self.session_folder,
                                      f"MESOSCOPE_FILE_SPLITTING_QUEUE_{self.session_id}_input.json")

        if os.path.isfile(splitting_json):
            self.splitting_json_present = True
        else:
            logger.error("Unable to find splitting json")
            self.splitting_json_present = False

        return splitting_json

    def get_paired_planes(self):

        splitting_json = self.get_splitting_json()

        with open(splitting_json, "r") as f:
            data = json.load(f)

        pairs = []
        for pg in data.get("plane_groups", []):
            pairs.append([p["experiment_id"] for p in pg.get("ophys_experiments", [])])

        return pairs

    def get_exp_by_structure(self, structure):

        experiment = pd.DataFrame(self.get_mesoscope_session_data(self.session_id))

        return experiment.loc[experiment.structure == structure]

    def get_full_field_tiff(self, full_field_path_offline=''):

        if full_field_path_offline != '':
            # use full field path to the tiff:
            full_field_path = os.path.join(full_field_path_offline, f"{self.session_id}_fullfield.tif")
            self.full_field_path = full_field_path

            if os.path.isfile(full_field_path):
                self.full_field_present = True
            else:
                full_field_path = ''
                logger.error("Can't find full field tiff at offline path, check if file exists")
                self.full_field_present = False

        else:
            # see if file exists in lims:
            session_folder = self.get_session_folder()
            full_field_path = os.path.join(session_folder, f"{self.session_id}_fullfield.tif")

            if os.path.isfile(full_field_path):
                self.full_field_path = full_field_path
                self.full_field_present = True
            else:
                full_field_path = ''
                logger.error("Full field tiff is absent in session folder, provide offline path")
                self.full_field_present = False

        return full_field_path

    def stitch_full_field(self, summ=True):
        full_field_tiff_path = self.full_field_path
        ff_path = full_field_tiff_path
        image_stitched = None
        image_sum = None
        if full_field_tiff_path != '':
            tiff = tifffile.TiffFile(full_field_tiff_path)
            meta = tiff.scanimage_metadata
            image = tiff.asarray()
            tiff.close()
            slices = image.shape[0]
            rois = meta['RoiGroups']['imagingRoiGroup']['rois']
            roi_num = len(rois)
            pixel_res_y = rois[1]['scanfields']['pixelResolutionXY'][1]
            pixel_res_x = rois[1]['scanfields']['pixelResolutionXY'][0]
            image_nlines = image.shape[1]
            image_npixels = image.shape[2]
            y_gap = np.int16((image_nlines - pixel_res_y * roi_num) / (roi_num - 1))
            image_stitched = np.zeros([slices, pixel_res_y, pixel_res_x * roi_num], dtype='int16')
            ff_image_name = ff_path.replace(os.path.dirname(ff_path) + '/', '')
            ff_stitch_name = os.path.dirname(ff_path) + '/' + ff_image_name.split('.')[0] + '_stitched.tif'
            ff_stitch_summ_name = os.path.dirname(ff_path) + '/' + ff_image_name.split('.')[0] + '_stitched_sum.tif'
            for j in range(slices):
                for i in range(roi_num):
                    image_stitched[j, :, i * image_npixels:(i + 1) * image_npixels] = image[j, i * (pixel_res_y + y_gap):(i + 1) * pixel_res_y + i * y_gap,:]
            if summ:
                image_sum = np.int16(image_stitched.mean(axis=0))
                tifffile.imsave(ff_stitch_summ_name, image_sum)
            tifffile.imsave(ff_stitch_name, image_stitched)

        else:
            full_field_path = ''
            logger.error("Full field tiff is absent in session folder, provide offline path")
        return image_stitched, image_sum, meta