"""
Created on Thursday May 31 2018

@author: marinag
"""
import os
import h5py
import json
import platform
import numpy as np
import pandas as pd
from visual_behavior.ophys.io.lims_database import LimsDatabase


class CellMatchingDataset(object):
    def __init__(self, lims_id, cache_dir=None, from_processed_data=True):
        """loads files necessary to do cell matching.
                    loads experiment data from lims network locations, or from processed data files in analysis_dir if from_processed_data is True.
                    importantly, this class loads all segmented cell masks, whether they are valid or not, in contrast to
                    the visual_behavior_scientifica_dataset class which only gets valid ROIs (valid = whether or not it is filtered by segmentation)

                Parameters
                ----------
                lims_id : ophys experiment ID (not session ID)
                cache_dir : directory to save or load analysis files from
                from_processed_data : create dataset object using saved processed data files, not from lims directory

                """
        self.lims_id = lims_id
        self.cache_dir = cache_dir
        self.from_processed_data = from_processed_data
        self.get_cache_dir()
        self.get_directories()
        self.filter_valid_rois = False
        self.process_roi_masks()
        self.get_roi_metrics()
        self.make_roi_dict()
        self.get_max_projection()
        self.get_average_image()

    def get_cache_dir(self):
        if self.cache_dir is None:
            if platform.system() == 'Linux':
                cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'
            else:
                cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'
            print('using default cache_dir:', cache_dir)
        else:
            cache_dir = self.cache_dir
        self.cache_dir = cache_dir
        return self.cache_dir

    def get_lims_data(self):
        # if self.from_processed_data is True:
        #     lims_data = pd.read_hdf(os.path.join(self.analysis_dir, 'metadata.h5'), key='df', )
        #     self.experiment_id = lims_data.ophys_experiment_id.values[0]
        #     self.session_name = lims_data.session_type.values[0].split('_')[-1]
        #     self.structure = lims_data.targeted_structure.values[0]
        #     self.specimen_driver_line = lims_data.cre_line.values[0]
        #     self.depth = lims_data.imaging_depth.values[0]
        #     self.experiment_date = str(lims_data.experiment_date.values[0])[:10]
        #     self.experiment_name = lims_data.session_type.values[0]
        #     mouse_id = lims_data.specimen_id.values[0]
        #     self.mouse_id = np.int(mouse_id)
        # else:
        ld = LimsDatabase(self.lims_id)
        lims_data = ld.get_qc_param()
        lims_data.insert(loc=2, column='experiment_id', value=lims_data.lims_id.values[0])
        self.experiment_id = lims_data.lims_id.values[0]
        self.session_name = lims_data.experiment_name.values[0].split('_')[-1]
        self.structure = lims_data.structure.values[0]
        self.specimen_driver_line = lims_data.specimen_driver_line.values[0]
        self.depth = lims_data.depth.values[0]
        self.experiment_date = str(lims_data.experiment_date.values[0])[:10]
        self.experiment_name = lims_data.experiment_name.values[0]
        mouse_id = lims_data.external_specimen_id.values[0]
        lims_data.mouse_id = mouse_id
        self.mouse_id = np.int(mouse_id)
        self.session_id = lims_data.session_id.values[0]
        self.ophys_session_dir = lims_data.datafolder.values[0][:-28]
        if platform.system() == 'Linux':
            self.ophys_session_dir = self.ophys_session_dir.replace('\\', '/')
        elif (os.name == 'nt') and (self.ophys_session_dir.startswith('/')):
            self.ophys_session_dir = self.ophys_session_dir.replace('/', '\\')
            self.ophys_session_dir = '\\' + self.ophys_session_dir

        self.lims_data = lims_data
        return self.lims_data

    def get_analysis_folder_name(self):
        folder = [file for file in os.listdir(self.cache_dir) if str(self.experiment_id) in file]
        if len(folder) > 0:
            self.analysis_folder_name = folder[0]
        else:
            print('no analysis folder in cache')
            print('creating analysis folder')
            lims_data = self.lims_data
            date = str(lims_data.experiment_date)[:10].split('-')
            analysis_folder_name = str(lims_data.lims_id) + '_' + \
                str(lims_data.mouse_id) + '_' + date[0][2:] + date[1] + date[2] + '_' + \
                lims_data.structure + '_' + str(lims_data.depth) + '_' + \
                lims_data.specimen_driver_line.split('-')[0] + '_' + lims_data.rig[3:5] + \
                lims_data.rig[6] + '_' + lims_data.session_type
            self.analysis_folder_name = analysis_folder_name
        return self.analysis_folder_name

    def get_directories(self):
        print(self.lims_id)
        cache_dir = self.get_cache_dir()
        if self.from_processed_data:
            # find existing analysis folder name
            self.analysis_folder_name = \
                [folder for folder in os.listdir(self.cache_dir) if str(self.lims_id) in folder][0]
            self.analysis_dir = os.path.join(self.cache_dir, self.analysis_folder_name)
            self.lims_data = self.get_lims_data()
        else:
            # create analysis folder name from lims_data
            self.lims_data = self.get_lims_data()
            self.analysis_folder_name = self.get_analysis_folder_name()
            analysis_dir = os.path.join(cache_dir, self.analysis_folder_name)
            if not os.path.exists(analysis_dir):
                os.mkdir(analysis_dir)
            self.analysis_dir = analysis_dir
        self.ophys_experiment_dir = os.path.join(self.ophys_session_dir, 'ophys_experiment_' + str(self.experiment_id))
        self.demix_dir = os.path.join(self.ophys_experiment_dir, 'demix')
        self.processed_dir = os.path.join(self.ophys_experiment_dir, 'processed')
        segmentation_folder = [file for file in os.listdir(self.processed_dir) if 'segmentation' in file]
        self.segmentation_dir = os.path.join(self.processed_dir, segmentation_folder[0])

    def get_objectlist(self):
        # objectlist.txt contains metrics associated with segmentation masks
        if self.from_processed_data is True:
            objectlist = pd.read_csv(os.path.join(self.analysis_dir, 'objectlist.txt'))
        else:
            seg_folder = [file for file in os.listdir(self.processed_dir) if 'segmentation' in file][0]
            objectlist = pd.read_csv(
                os.path.join(self.processed_dir, seg_folder, 'objectlist.txt'))  # segmentation metrics
        self.objectlist = objectlist
        return self.objectlist

    def get_roi_names(self):
        if self.from_processed_data:
            f = h5py.File(os.path.join(self.analysis_dir, 'roi_names.h5'), 'r')
            self.roi_names = np.asarray(f['data'])
            f.close()
        else:
            file_path = os.path.join(self.ophys_experiment_dir, 'neuropil_correction.h5')
            f = h5py.File(file_path)
            self.roi_names = np.asarray(f['roi_names'])
            f.close()
        return self.roi_names

    # convert ruby json array ouput to python 2D array - needed for segmentation output prior to 10/10/17 due to change in how masks were saved
    def parse_mask_string(self, mask_string):
        mask = []
        row_length = -1
        for i in range(1, len(mask_string) - 1):
            c = mask_string[i]
            if c == '{':
                row = []
            elif c == '}':
                mask.append(row)
                if row_length < 1:
                    row_length = len(row)
            elif c == 'f':
                row.append(False)
            elif c == 't':
                row.append(True)
        return np.asarray(mask)

    def process_roi_masks(self):
        print('getting roi masks')
        if self.from_processed_data:
            json_file = [file for file in os.listdir(self.analysis_dir) if 'input_extract_traces.json' in file]
            json_path = os.path.join(self.analysis_dir, json_file[0])
        else:
            json_file = [file for file in os.listdir(self.processed_dir) if 'input_extract_traces.json' in file]
            json_path = os.path.join(self.processed_dir, json_file[0])
        with open(json_path, 'r') as w:
            jin = json.load(w)
        self.image_height = jin["image"]["height"]
        self.image_width = jin["image"]["width"]
        rois = jin["rois"]
        # get data out of json and into dataframe
        roi_df_list = []
        for i in range(len(rois)):
            roi = rois[i]
            if roi['mask'][0] == '{':
                mask = self.parse_mask_string(roi['mask'])
            else:
                mask = roi["mask"]
            roi_df_list.append([roi["id"], roi["x"], roi["y"], roi["width"], roi["height"], roi["valid"], mask])
        roi_df = pd.DataFrame(data=roi_df_list, columns=['id', 'x', 'y', 'width', 'height', 'valid', 'mask'])
        # get indices for ids
        roi_names = self.get_roi_names()
        roi_df['unfiltered_trace_index'] = [np.where(roi_names == str(id))[0][0] for id in roi_df.id.values]
        self.roi_df = roi_df
        # add roi ids to objectlist
        ids = []
        df = self.get_objectlist()
        for row in df.index:
            minx = df.iloc[row][' minx']
            miny = df.iloc[row][' miny']
            id = roi_df[(roi_df.x == minx) & (roi_df.y == miny)].id.values[0]
            ids.append(id)
        df['id'] = ids
        # add indices to objectlist corresponding to id order
        df['unfiltered_trace_index'] = [roi_df[roi_df.id == t_id]['unfiltered_trace_index'].values[0] for t_id in
                                        df['id'].values]
        df['valid'] = [roi_df[roi_df.id == c_id]['valid'].values[0] for c_id in df['id'].values]
        self.objectlist = df

    def get_cell_specimen_ids(self):
        if self.filter_valid_rois:
            self.cell_specimen_ids = np.sort(self.roi_df[self.roi_df['valid'] == True]['id'].values)
        else:
            self.cell_specimen_ids = np.sort(self.roi_df['id'].values)
        return self.cell_specimen_ids

    # def get_filtered_cell_indices(self):
    #     # only want to use traces for rois that meet filtering criteria
    #     # pipeline filter step removes edge cells, dendrites, cells that can't be demixed, etc.
    #     self.cell_indices = np.sort(self.roi_df[self.roi_df['valid'] == True]['unfiltered_trace_index'].values)
    #     return self.cell_indices

    def get_cell_specimen_id_for_index(self, index):
        cell_specimen_id = self.cell_specimen_ids[index]
        return cell_specimen_id

    def get_index_for_cell_specimen_id(self, cell_specimen_id):
        index = np.where(self.cell_specimen_ids == cell_specimen_id)[0][0]
        return index

    def get_roi_metrics(self):
        # get roi metrics such as brightness, shape, size, etc from segmentation output dataframe
        if self.filter_valid_rois:
            self.roi_metrics = self.objectlist[self.objectlist['valid'] == True]
        else:
            self.roi_metrics = self.objectlist
        return self.roi_metrics

    def make_roi_dict(self):
        # make roi_dict with ids as keys and roi_mask_array
        h = self.image_height
        w = self.image_width
        roi_dict = {}
        roi_mask_array = np.zeros((len(self.get_cell_specimen_ids()), h, w))
        for i, id in enumerate(self.get_cell_specimen_ids()):
            m = self.roi_df[self.roi_df.id == id]
            mask = np.asarray(m['mask'].values[0])
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            binary_mask[int(m.y):int(m.y) + int(m.height), int(m.x):int(m.x) + int(m.width)] = mask
            roi_dict[id] = binary_mask
            roi_mask_array[i, :, :] = binary_mask
        self.roi_dict = roi_dict
        self.roi_mask_array = roi_mask_array

    def get_max_projection(self):
        if self.from_processed_data:
            f = h5py.File(os.path.join(self.analysis_dir, 'max_projection.h5'), 'r')
            self.max_projection = np.asarray(f['data'])
        else:
            import matplotlib.image as mpimg
            self.max_projection = mpimg.imread(os.path.join(self.segmentation_dir, 'maxInt_a13a.png'))
        return self.max_projection

    def get_average_image(self):
        import matplotlib.image as mpimg
        self.average_image = mpimg.imread(os.path.join(self.segmentation_dir, 'avgInt_a1X.png'))
        return self.average_image
