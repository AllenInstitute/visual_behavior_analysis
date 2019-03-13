from allensdk.brain_observatory import roi_masks
import visual_behavior.ophys.mesoscope.mesoscope as ms
import glob
import os
import h5py
from scipy.ndimage import label
import numpy as np
import logging
logger = logging.getLogger(__name__)

from sklearn.decomposition import FastICA
import scipy.optimize as opt
from scipy.linalg import sqrtm,inv
import scipy.ndimage as ndim
import matplotlib.pylab as plt
import pandas as pd
import psycopg2
import psycopg2.extras
import json

from PIL import Image

IMAGEH, IMAGEW = 512, 512


def get_traces(movie_h5, masks):
    rois = [roi_masks.create_roi_mask(IMAGEW, IMAGEH, border=[IMAGEW-1,0,IMAGEH-1,0], roi_mask=roi, label='{}'.format(i))
                for i, roi in enumerate(masks)]
    traces = roi_masks.calculate_roi_and_neuropil_traces(movie_h5, rois, motion_border=[IMAGEW-1,0,IMAGEH-1,0])[0]
    return traces


def get_roi_masks(hdf_filename):
    masks = []
    with h5py.File(hdf_filename, "r") as f:
        stack = f["data"].value
        if stack.ndim == 2:
            stack = stack.reshape((1,) + stack.shape)
    for page in stack:
        labeled_page, nlabels = label(page)
        for i in range(nlabels):
            masks.append(labeled_page == i+1)
    return masks


class Mesoscope_ICA(object):

    def __init__(self, session_id, cache):

        self.session_id = session_id
        self.dataset = ms.MesoscopeDataset(session_id)

        self.plane1_exp_id = None
        self.plane2_exp_id = None

        self.ica_traces_dir = None
        self.session_cache_dir = None

        self.found_ica_traces = None  # output of get_traces

        self.plane1_traces_orig = None
        self.plane1_traces_orig_pointer = None
        self.plane1_ica_input = None
        self.plane1_ica_input_pointer = None
        self.plane1_ica_output = None
        self.plane1_ica_output_pointer = None

        self.plane2_traces_orig = None  # output of get_traces
        self.plane2_traces_orig_pointer = None
        self.plane2_ica_input = None  # output of combine_debiase
        self.plane2_ica_input_pointer = None
        self.plane2_ica_output = None  # output of unmix_traces
        self.plane2_ica_output_pointer = None

        self.plane1_offset = None
        self.plane2_offset = None

        self.found_solution = None  # output of unmix_traces

        self.matrix = None
        self.cache = cache

    def get_ica_traces(self, pair):

        self.found_ica_traces = [False, False]

        plane1_exp_id = pair[0]
        plane2_exp_id = pair[1]

        self.plane1_exp_id = plane1_exp_id
        self.plane2_exp_id = plane2_exp_id

        # let's see if traces exist aready:
        session_dir = os.path.join(self.cache, f'session_{self.session_id}')
        self.session_cache_dir = session_dir
        ica_traces_dir = os.path.join(session_dir, f'ica_traces_{plane1_exp_id}_{plane2_exp_id}/')
        self.ica_traces_dir = ica_traces_dir
        path_traces_plane1 = f'{ica_traces_dir}traces_original_{plane1_exp_id}.h5'
        path_traces_plane2 = f'{ica_traces_dir}traces_original_{plane2_exp_id}.h5'

        if not (os.path.isfile(path_traces_plane1) and os.path.isfile(path_traces_plane2)):
            # -------------------------------------------------------------------------------------------
            # retrieve both planes experiment path
            logger.warning('Traces dont exist in cache dir, extracting')

            plane1_folder = self.dataset.get_exp_folder(plane1_exp_id)
            plane2_folder = self.dataset.get_exp_folder(plane2_exp_id)
            plane1_path = glob.glob(plane1_folder + f"{plane1_exp_id}.h5")
            plane2_path = glob.glob(plane2_folder + f"{plane2_exp_id}.h5")
            plane1_h5_mask_file = glob.glob(plane1_folder + f"processed/ophys_cell_segmentation_run*/maxInt_masks.h5")
            plane2_h5_mask_file = glob.glob(plane2_folder + f"processed/ophys_cell_segmentation_run*/maxInt_masks.h5")

            # find most recent segmentation run for pane1 experiment
            if len(plane1_h5_mask_file) > 1:
                plane1_h5_mask_name = max(plane1_h5_mask_file, key=os.path.getctime)
                plane1_masks = get_roi_masks(plane1_h5_mask_name)
            else:
                plane1_masks = get_roi_masks(plane1_h5_mask_file[0])

            # extract signal and crosstalk traces for plane 1
            plane1_sig_traces = get_traces(plane1_path[0], plane1_masks)
            plane1_ct_traces = get_traces(plane2_path[0], plane1_masks)
            self.plane1_sig = plane1_sig_traces
            self.plane1_ct = plane1_sig_traces

            # find most recent segmentation run for plane 2 experiment
            if len(plane2_h5_mask_file) > 1:
                plane2_h5_mask_name = max(plane2_h5_mask_file, key=os.path.getctime)
                plane2_masks = get_roi_masks(plane2_h5_mask_name)
            else:
                plane2_masks = get_roi_masks(plane2_h5_mask_file[0])

            # extract signal and crosstalk traces for plane 2
            plane2_sig_traces = get_traces(plane2_path[0], plane2_masks)
            plane2_ct_traces = get_traces(plane1_path[0], plane2_masks)
            self.plane2_sig = plane2_sig_traces
            self.plane2_ct = plane2_ct_traces

            if (not plane2_sig_traces.any() == None) and (not plane2_ct_traces.any() == None):
                self.found_ica_traces[0] = True
            if (not plane2_sig_traces.any() == None) and (not plane2_ct_traces.any() == None):
                self.found_ica_traces[1] = True

            plane1_traces_original = np.array([plane1_sig_traces, plane1_ct_traces])
            plane2_traces_original = np.array([plane2_sig_traces, plane2_ct_traces])

            self.plane1_traces_orig = plane1_traces_original
            self.plane2_traces_orig = plane2_traces_original

            # saving extracted traces:
            if not os.path.isdir(session_dir):
                os.mkdir(session_dir)
            if not os.path.isdir(ica_traces_dir):
                os.mkdir(ica_traces_dir)

            if self.found_ica_traces[0]:
                if not os.path.isfile(path_traces_plane1):
                    with h5py.File(path_traces_plane1, "w") as f:
                        f.create_dataset(f"data", data=self.plane1_traces_orig)

            if self.found_ica_traces[1]:
                if not os.path.isfile(path_traces_plane2):
                    with h5py.File(path_traces_plane2, "w") as f:
                        f.create_dataset(f"data", data=self.plane2_traces_orig)
        else:
            logger.warning('Found traces, reading form file')
            # read traces form h5 file:
            with h5py.File(path_traces_plane1, "r") as f:
                plane1_traces_original = f["data"].value
            with h5py.File(path_traces_plane2, "r") as f:
                plane2_traces_original = f["data"].value

            self.plane1_traces_orig_pointer = path_traces_plane1
            self.plane2_traces_orig_pointer = path_traces_plane2

            self.plane1_traces_orig = plane1_traces_original
            self.plane2_traces_orig = plane2_traces_original

            # set foudn traces flag True
            self.found_ica_traces = [True, True]

        return self.found_ica_traces

    def combine_debias_traces(self):

        plane1_ica_input_pointer = os.path.join(self.ica_traces_dir,
                                                f'traces_ica_input_{self.plane1_exp_id}.h5')

        if os.path.isfile(plane1_ica_input_pointer):
            self.plane1_ica_input_pointer = plane1_ica_input_pointer
            # file already exists, skip debiasing

        plane2_ica_input_pointer = os.path.join(self.ica_traces_dir,
                                                f'traces_ica_input_{self.plane2_exp_id}.h5')

        if os.path.isfile(plane2_ica_input_pointer):
            self.plane2_ica_input_pointer = plane2_ica_input_pointer
            # file already exists, skip debiasing

        if not (self.plane1_ica_input_pointer and self.plane2_ica_input_pointer):

            self.plane1_ica_input_pointer = plane1_ica_input_pointer
            self.plane2_ica_input_pointer = plane2_ica_input_pointer

            logger.warning("debiased traces do not exist, running offset subtraction")

            if self.found_ica_traces:

                plane1_sig = self.plane1_traces_orig[0]
                plane1_ct = self.plane1_traces_orig[1]

                plane2_sig = self.plane2_traces_orig[0]
                plane2_ct = self.plane2_traces_orig[1]

                # subtract offset plane 1:
                nc = plane1_sig.shape[0]
                plane1_sig_offset = np.mean(plane1_sig, axis=1).reshape(nc, 1)
                plane1_sig_m0 = plane1_sig - plane1_sig_offset

                nc = plane1_ct.shape[0]
                plane1_ct_offset = np.mean(plane1_ct, axis=1).reshape(nc, 1)
                plane1_ct_m0 = plane1_ct - plane1_ct_offset

                # subtract offset for plane 2:
                nc = plane2_sig.shape[0]
                plane2_sig_offset = np.mean(plane2_sig, axis=1).reshape(nc, 1)
                plane2_sig_m0 = plane2_sig - plane2_sig_offset

                nc = plane2_ct.shape[0]
                plane2_ct_offset = np.mean(plane2_ct, axis=1).reshape(nc, 1)
                plane2_ct_m0 = plane2_ct - plane2_ct_offset

                self.plane1_offset = {'plane1_sig_offset': plane1_sig_offset, 'plane1_ct_offset': plane1_ct_offset}
                self.plane2_offset = {'plane2_sig_offset': plane2_sig_offset, 'plane2_ct_offset': plane2_ct_offset, }

                trace_sig_p1 = plane1_sig_m0.flatten()
                trace_ct_p1 = plane1_ct_m0.flatten()
                trace_sig_p2 = plane2_sig_m0.flatten()
                trace_ct_p2 = plane2_ct_m0.flatten()

                plane1_ica_input = np.append(trace_sig_p1, trace_ct_p2, axis=0)
                plane2_ica_input = np.append(trace_ct_p1, trace_sig_p2, axis=0)

                self.plane1_ica_input = plane1_ica_input
                self.plane2_ica_input = plane2_ica_input

                # write ica input traces to disk

                if not os.path.isfile(self.plane1_ica_input_pointer):
                    with h5py.File(self.plane1_ica_input_pointer, "w") as f:
                        f.create_dataset("debiased_traces", data=self.plane1_ica_input)
                        f.create_dataset('sig_offset', data=plane1_sig_offset)
                        f.create_dataset('ct_offset', data=plane1_ct_offset)

                if not os.path.isfile(self.plane2_ica_input_pointer):
                    with h5py.File(self.plane2_ica_input_pointer, "w") as f:
                        f.create_dataset("debiased_traces", data=self.plane2_ica_input)
                        f.create_dataset('sig_offset', data=plane2_sig_offset)
                        f.create_dataset('ct_offset', data=plane2_ct_offset)
            else:
                logger.error('Extract traces first')
        else:
            logger.warning("Debiased traces exist, reading from h5 file")
            with h5py.File(self.plane1_ica_input_pointer, "r") as f:
                plane1_ica_input = f["debiased_traces"].value
                plane1_sig_offset = f['sig_offset'].value
                plane1_ct_offset = f['ct_offset'].value
            with h5py.File(self.plane2_ica_input_pointer, "r") as f:
                plane2_ica_input = f["debiased_traces"].value
                plane2_sig_offset = f['sig_offset'].value
                plane2_ct_offset = f['ct_offset'].value
            self.plane1_ica_input = plane1_ica_input
            self.plane2_ica_input = plane2_ica_input
            self.plane1_offset = {'plane1_sig_offset': plane1_sig_offset, 'plane1_ct_offset': plane1_ct_offset}
            self.plane2_offset = {'plane2_sig_offset': plane2_sig_offset, 'plane2_ct_offset': plane2_ct_offset, }
        return 