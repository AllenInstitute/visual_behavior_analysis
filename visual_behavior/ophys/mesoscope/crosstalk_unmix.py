from allensdk.brain_observatory import roi_masks
import visual_behavior.ophys.mesoscope.mesoscope as ms
import glob
import os
import h5py
from scipy.ndimage import label
import numpy as np

from sklearn.decomposition import FastICA
from scipy.linalg import sqrtm,inv
import scipy.ndimage as ndim
import matplotlib.pylab as plt
import pandas as pd
import psycopg2
import psycopg2.extras
import json
import scipy.optimize as opt
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

    def __init__(self, session_id):

        self.session_id = session_id
        self.dataset = ms.MesoscopeDataset(session_id)
        self.plane1_exp_id = None
        self.plane2_exp_id = None
        self.found_ica_traces = None
        self.plane2_sig_m0 = None
        self.plane2_ct_m0 = None
        self.plane1_sig_m0 = None
        self.plane1_ct_m0 = None
        self.plane2_sig = None
        self.plane2_ct = None
        self.plane1_sig = None
        self.plane1_ct = None
        self.found_solution = None
        self.traces_unmix = None
        self.matrix = None

    def get_ica_traces(self, pair):

        self.found_ica_traces = [False, False]

        plane1_exp_id = pair[0]
        plane2_exp_id = pair[1]
#-------------------------------------------------------------------------------------------
#retrieve both planes experiment path
        plane1_folder = self.dataset.get_exp_folder(plane1_exp_id)
        plane2_folder = self.dataset.get_exp_folder(plane2_exp_id)
        plane1_path = glob.glob(plane1_folder + f"{plane1_exp_id}.h5")
        plane2_path = glob.glob(plane2_folder + f"{plane2_exp_id}.h5")
        plane1_h5_mask_file = glob.glob(plane1_folder + f"processed/ophys_cell_segmentation_run*/maxInt_masks.h5")
        plane2_h5_mask_file = glob.glob(plane2_folder + f"processed/ophys_cell_segmentation_run*/maxInt_masks.h5")

        #find most recent segmentation run for pane1 experiment
        if len(plane1_h5_mask_file) > 1:
            plane1_h5_mask_name = max(plane1_h5_mask_file, key=os.path.getctime)
            plane1_masks = get_roi_masks(plane1_h5_mask_name)
        else:
            plane1_masks = get_roi_masks(plane1_h5_mask_file[0])
        # extract signal and crosstalk traces for plane 1
        plane1_sig_traces = get_traces(plane1_path [0], plane1_masks)
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
        # subtract offset, flatten traces for plane 1:
        nc = plane2_sig_traces.shape[0]
        plane1_sig_m0 = plane1_sig_traces - np.mean(plane1_sig_traces, axis=1).reshape(nc, 1)
        nc = plane1_ct_traces.shape[0]
        plane1_ct_m0 = plane1_ct_traces - np.mean(plane1_ct_traces, axis=1).reshape(nc, 1)
        self.plane1_sig_m0 = plane1_sig_m0
        self.plane1_ct_m0 = plane1_ct_m0
        # subtract offset, flatten traces for plane 2:
        nc = plane2_sig_traces.shape[0]
        plane2_sig_m0 = plane2_sig_traces - np.mean(plane2_sig_traces, axis=1).reshape(nc, 1)
        nc = plane2_ct_traces.shape[0]
        plane2_ct_m0 = plane2_ct_traces - np.mean(plane2_ct_traces, axis=1).reshape(nc, 1)
        self.plane2_sig_m0 = plane2_sig_m0
        self.plane2_ct_m0 = plane2_ct_m0
        if (not np.max(plane2_sig_traces) == 0) and (not np.max(plane2_ct_traces) == 0):
            self.found_ica_traces[0] = True
        if (not np.max(plane2_sig_traces) == 0) and (not np.max(plane2_ct_traces) == 0):
            self.found_ica_traces[1] = True
        return self.found_ica_traces

    def unmix_traces(self, traces, max_iter=10):
        self.found_solution = False
        for i in range(max_iter):
            ica = FastICA(n_components=2)
            self.traces_unmix = ica.fit_transform(traces)  # Reconstruct signals
            self.matrix = ica.mixing_  # Get estimated mixing matrix
            if (np.all(a > 0)) & (a[0][0] > a[1][0]):
                self.found_solution = True
                break
            if not self.found_solution:
                raise ValueError("Failed to find solution, try increasing `max_iter`")
        return self.found_solution