import psycopg2
import psycopg2.extras
import pandas as pd
import logging
import os
import tifffile
import json
import numpy as np
from skimage.transform import resize

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

DEFAULT_DATABASE = 'lims2'
DEFAULT_HOST = 'limsdb2'
DEFAULT_PORT = 5432
DEFAULT_USERNAME = 'limsreader'
PW = 'limsro'


def psycopg2_select(query, database=DEFAULT_DATABASE, host=DEFAULT_HOST, port=DEFAULT_PORT, username=DEFAULT_USERNAME,
                    password=PW):
    connection = psycopg2.connect(
        host=host, port=port, dbname=database, user=username, password=password,
        cursor_factory=psycopg2.extras.RealDictCursor
    )
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        response = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()
    return response

def get_all_mesoscope_data():
    query = ("select os.id as session_id, oe.id as experiment_id, "
             "os.storage_directory as session_folder, oe.storage_directory as experiment_folder, "
             "sp.name as specimen, "
             "os.date_of_acquisition as date, "
             "oe.workflow_state as exp_workflow_state, "
             "os.workflow_state as session_workflow_state " 
             "from ophys_experiments oe "
             "join ophys_sessions os on os.id = oe.ophys_session_id "
             "join specimens sp on sp.id = os.specimen_id "
             "join projects p on p.id = os.project_id "
             "where (p.code = 'VisualBehaviorMultiscope' or p.code = 'VisualBehaviorMultiscope4areasx2d' ) and os.workflow_state ='uploaded' " # and 'MesoscopeDevelopment' or p.code =  (oe.workflow_state = 'processing' or oe.workflow_state = 'qc') and os.workflow_state ='uploaded' "
             "order by session_id")
    return pd.DataFrame(psycopg2_select(query))

class MesoscopeDataset(object):
    def __init__(self, session_id, experiment_id=None):
        self.exp_folder = None
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
        self.exp_id = None

    def get_session_id(self):
        return self.session_id

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

    def get_mesoscope_session_data(self, session_id=None):
        lims_data = None
        session_id = session_id or self.session_id
        if not session_id:
            lims_data = None
            logger.error("Provide Session ID")
        else:
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
                    "where (p.code = 'MesoscopeDevelopment' or p.code = 'VisualBehaviorMultiscope' or p.code = 'VisualBehaviorMultiscope4areasx2d' ) and os.workflow_state ='uploaded' "
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
        return lims_data

    def get_mesoscope_experiment_data(self, experiment_id):

        lims_data = None

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
                "where (p.code = 'MesoscopeDevelopment' or p.code = 'VisualBehaviorMultiscope' or p.code = 'VisualBehaviorMultiscope4areasx2d' ) and os.workflow_state ='uploaded' "
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
        return lims_data

    def get_exp_folder(self, exp_id):
        self.exp_id = exp_id
        exp = pd.DataFrame(self.get_mesoscope_experiment_data(exp_id))
        exp_folder = exp['experiment_folder'].values[0]
        self.exp_folder = exp_folder
        return exp_folder

    def get_session_folder(self):
        data = self.data_pointer or self.get_mesoscope_session_data()
        if data:
            data_frame = pd.DataFrame(data)
            self.session_folder = data_frame['session_folder'].values[0]
        else:
            logger.error("Session does not exist in LIMS")
            self.session_folder = None
        return self.session_folder

    def get_splitting_json(self):
        session_folder = self.get_session_folder()
        splitting_json = os.path.join(session_folder, f"MESOSCOPE_FILE_SPLITTING_QUEUE_{self.session_id}_input.json")
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
        experiment = pd.DataFrame(self.get_mesoscope_session_data())
        return experiment.loc[experiment.structure == structure]

    def get_full_field_tiff(self, full_field_path_offline=None):
        if not full_field_path_offline:
            # use full field path to the tiff:
            full_field_path = os.path.join(full_field_path_offline, f"{self.session_id}_fullfield.tif")
            self.full_field_path = full_field_path
            if os.path.isfile(full_field_path):
                self.full_field_present = True
            else:
                full_field_path = None
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
        if full_field_tiff_path:
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
                    image_stitched[j, :, i * image_npixels:(i + 1) * image_npixels] = image[j,i * (pixel_res_y + y_gap):(i + 1) * pixel_res_y + i * y_gap, :]
            if summ:
                image_sum = np.int16(image_stitched.mean(axis=0))
                tifffile.imsave(ff_stitch_summ_name, image_sum)
            tifffile.imsave(ff_stitch_name, image_stitched)
        else:
            logger.error("Full field tiff is absent in session folder, provide offline path")
        return image_stitched, image_sum, meta

    def register_rois_to_FullFOV(self):

        ses = self.get_mesoscope_session_data()

        # get full field image -> stitch full field image
        # !!!not tested!!! - test on a dataset with valid full field
        json_output = os.path.join(self.get_session_folder(), 'MESOSCOPE_FILE_SPLITTING_QUEUE_', self.session_id, '_output.json')

        _ = self.get_full_field_tiff()
        _, ff_image, ff_meta = self.stitch_full_field()

        # from MDF of scanimage, or tiff.scanimagemetadata
        # factor to translate from angular to linear coordinates, microns per degree
        mpg = ff_meta['FrameData']['SI.objectiveResolution']

        input_json = os.path.join(self.get_session_folder(),
                                  f'MESOSCOPE_FILE_SPLITTING_QUEUE_{self.session_id}_input.json')

        with open(input_json) as f:
            all_meta = json.load(f)

        surface_raw_file = all_meta["surface_tif"]

        json_output = os.path.join(self.get_session_folder(),
                                   f'MESOSCOPE_FILE_SPLITTING_QUEUE_{self.session_id}_output.json')

        with open(json_output) as f:
            all_meta = json.load(f)

        for file in all_meta["file_metadata"]:
            if file['input_tif'] == surface_raw_file:
                rois_meta = file['roi_metadata']

        ff_rois = ff_meta['RoiGroups']['imagingRoiGroup']['rois']
        rois_rois = rois_meta['RoiGroups']['imagingRoiGroup']['rois']

        roi_data = pd.DataFrame(
            index=('FF_deg', 'roi1_deg', 'roi2_deg', 'FF_pix', 'roi1_pix', 'roi2_pix', 'FF_um', 'roi1_um', 'roi2_um'),
            columns=('sizeX', 'sizeY', 'centerX', 'centerY', 'resX', 'resY'))

        roi_data.loc['FF_deg']['sizeX'] = ff_rois[0]['scanfields']['sizeXY'][0] * len(ff_rois)
        roi_data.loc['FF_deg']['sizeY'] = ff_rois[1]['scanfields']['sizeXY'][1]

        roi_data.loc['FF_deg']['centerX'] = ff_rois[0]['scanfields']['centerXY'][0] + (
                ff_rois[len(ff_rois) - 1]['scanfields']['centerXY'][0] - ff_rois[0]['scanfields']['centerXY'][0]) / 2
        roi_data.loc['FF_deg']['centerY'] = ff_rois[0]['scanfields']['centerXY'][1]

        roi_data.loc['FF_pix']['sizeX'] = ff_rois[1]['scanfields']['pixelResolutionXY'][0] * len(ff_rois)
        roi_data.loc['FF_pix']['sizeY'] = ff_rois[1]['scanfields']['pixelResolutionXY'][1]
        roi_data.loc['FF_pix']['centerX'] = roi_data.loc['FF_pix']['sizeX'] / 2
        roi_data.loc['FF_pix']['centerY'] = roi_data.loc['FF_pix']['sizeY'] / 2

        roi_data.loc['FF_deg']['resX'] = roi_data.loc['FF_pix']['sizeX'] / roi_data.loc['FF_deg']['sizeX']
        roi_data.loc['FF_deg']['resY'] = roi_data.loc['FF_pix']['sizeY'] / roi_data.loc['FF_deg']['sizeY']

        roi_data.loc['FF_um']['sizeX'] = roi_data.loc['FF_deg']['sizeX'] * mpg
        roi_data.loc['FF_um']['sizeY'] = roi_data.loc['FF_deg']['sizeY'] * mpg
        roi_data.loc['FF_um']['centerX'] = roi_data.loc['FF_deg']['centerX'] * mpg
        roi_data.loc['FF_um']['centerY'] = roi_data.loc['FF_deg']['centerY'] * mpg

        roi_data.loc['FF_um']['resX'] = roi_data.loc['FF_um']['sizeX'] / roi_data.loc['FF_pix']['sizeX']
        roi_data.loc['FF_um']['resY'] = roi_data.loc['FF_um']['sizeY'] / roi_data.loc['FF_pix']['sizeY']

        k = 0

        for roi in rois_rois:
            k += 1

            roi_data.loc['roi{}_deg'.format(k)]['sizeX'] = roi['scanfields']['sizeXY'][0]
            roi_data.loc['roi{}_deg'.format(k)]['sizeY'] = roi['scanfields']['sizeXY'][1]
            roi_data.loc['roi{}_deg'.format(k)]['centerX'] = roi['scanfields']['centerXY'][0]
            roi_data.loc['roi{}_deg'.format(k)]['centerY'] = roi['scanfields']['centerXY'][1]

            roi_data.loc['roi{}_pix'.format(k)]['sizeX'] = roi['scanfields']['pixelResolutionXY'][0]
            roi_data.loc['roi{}_pix'.format(k)]['sizeY'] = roi['scanfields']['pixelResolutionXY'][1]

            roi_data.loc['roi{}_deg'.format(k)]['resX'] = roi_data.loc['roi{}_pix'.format(k)]['sizeX'] / \
                                                          roi_data.loc['roi{}_deg'.format(k)]['sizeX']
            roi_data.loc['roi{}_deg'.format(k)]['resY'] = roi_data.loc['roi{}_pix'.format(k)]['sizeY'] / \
                                                          roi_data.loc['roi{}_deg'.format(k)]['sizeY']

            roi_data.loc['roi{}_um'.format(k)]['sizeX'] = roi_data.loc['roi{}_deg'.format(k)]['sizeX'] * mpg
            roi_data.loc['roi{}_um'.format(k)]['sizeY'] = roi_data.loc['roi{}_deg'.format(k)]['sizeY'] * mpg
            roi_data.loc['roi{}_um'.format(k)]['centerX'] = roi_data.loc['roi{}_deg'.format(k)]['centerX'] * mpg
            roi_data.loc['roi{}_um'.format(k)]['centerY'] = roi_data.loc['roi{}_deg'.format(k)]['centerY'] * mpg

            roi_data.loc['roi{}_um'.format(k)]['resX'] = roi_data.loc['roi{}_um'.format(k)]['sizeX'] / \
                                                         roi_data.loc['roi{}_pix'.format(k)]['sizeX']
            roi_data.loc['roi{}_um'.format(k)]['resY'] = roi_data.loc['roi{}_um'.format(k)]['sizeY'] / \
                                                         roi_data.loc['roi{}_pix'.format(k)]['sizeY']

            roi_data.loc['roi{}_pix'.format(k)]['centerX'] = roi_data.loc['FF_pix']['centerX'] + (
                    roi_data.loc['roi{}_deg'.format(k)]['centerX'] - roi_data.loc['FF_deg']['centerX']) * \
                                                             roi_data.loc['FF_deg']['resX']
            roi_data.loc['roi{}_pix'.format(k)]['centerY'] = roi_data.loc['FF_pix']['centerY'] + (
                    roi_data.loc['roi{}_deg'.format(k)]['centerY'] - roi_data.loc['FF_deg']['centerY']) * \
                                                             roi_data.loc['FF_deg']['resY']
        # finding and reading surface tiffs
        structures = ses.drop_duplicates('structure')
        exp_folder_unique_structure = {}
        surface_roi = np.zeros([structures.shape[0], 512, 512])
        k = 0
        image_ff_roi = ff_image
        # loop that reads surface tiff files and inserts them into full field tiff
        for i, _ in structures.iterrows():
            exp_id = str(structures.loc[i]['experiment_id'])
            exp_folder_unique_structure[k] = os.path.join(structures.loc[i]['experiment_folder'],
                                                          f'{exp_id}_surface.tif')
            if os.path.isfile(exp_folder_unique_structure[k]):
                roi_tiff = tifffile.TiffFile(exp_folder_unique_structure[k])
                surface_roi[k, :, :] = roi_tiff.asarray()
                roi_tiff.close()
                # scaling factors:
                roi_scale_x = roi_data.loc[f'roi{k}_um']['resX'] / roi_data.loc['FF_um']['resX']
                roi_scale_y = roi_data.loc[f'roi{k}_um']['resY'] / roi_data.loc['FF_um']['resY']
                # new size:
                roi_new_size_x = np.int16(np.round(roi_data.loc[f'roi{k}_pix']['sizeX'] * roi_scale_x))
                roi_new_size_y = np.int16(np.round(roi_data.loc[f'roi{k}_pix']['sizeY'] * roi_scale_y))
                # scaling roi:
                roi_image_ds = np.uint16(
                    resize(surface_roi[k, :, :], np.int16([roi_new_size_x, roi_new_size_y])) * 2 ** 16)
                # calc insertion coordinates for roi1:
                a = np.uint16(np.round(roi_data.loc[f'roi{k}_pix']['centerX'] - roi_image_ds.shape[0] / 2.0))
                b = np.uint16(np.round(roi_data.loc[f'roi{k}_pix']['centerX'] + roi_image_ds.shape[0] / 2.0))
                c = np.uint16(np.round(roi_data.loc[f'roi{k}_pix']['centerY'] - roi_image_ds.shape[1] / 2.0))
                d = np.uint16(np.round(roi_data.loc[f'roi{k}_pix']['centerY'] + roi_image_ds.shape[1] / 2.0))
                # insert roi1 into full field image
                image_ff_roi[c:d, a:b] = roi_image_ds
            k += 1
        return image_ff_roi
