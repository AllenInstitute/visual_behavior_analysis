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

    def get_ica_traces(self, pair):

        found_Traces = [False, False]

        plane1_exp_id = pair[0]
        plane2_exp_id = pair[1]

        #read movie and masks for plane 1
        plane1_folder = self.dataset.get_exp_folder(plane1_exp_id)
        plane1_path = glob.glob(plane1_folder + f"{plane1_exp_id}.h5")
        plane2_path = glob.glob(plane1_folder + f"{plane2_exp_id}.h5")
        plane1_h5_mask_file = glob.glob(plane1_folder + f"processed/ophys_cell_segmentation_run*/maxInt_masks.h5")

        #find most recent segmentation run for this experiment
        if len(plane1_h5_mask_file) > 1:
            plane1_h5_mask_name = max(plane1_h5_mask_file, key=os.path.getctime)
            plane1_masks = get_roi_masks(plane1_h5_mask_name)
        else:
            plane1_masks = get_roi_masks(plane1_h5_mask_file[0])

        # extract signal and crosstalk traces for plane 1
        plane1_sig_traces = get_traces(plane1_path [0], plane1_masks)
        plane1_ct_traces = get_traces(plane2_path[0], plane1_masks)

        # read movie and masks for plane 2
        plane2_folder = self.dataset.get_exp_folder(plane2_exp_id)
        plane2_h5_mask_file = glob.glob(plane2_folder + f"processed/ophys_cell_segmentation_run*/maxInt_masks.h5")

        # find most recent segmentation run for this experiment
        if len(plane2_h5_mask_file) > 1:
            plane2_h5_mask_name = max(plane2_h5_mask_file, key=os.path.getctime)
            plane2_masks = get_roi_masks(plane2_h5_mask_name)
        else:
            plane2_masks = get_roi_masks(plane1_h5_mask_file[0])

        # extract signal and crosstalk traces for plane 2
        plane2_sig_traces = get_traces(plane2_path[0], plane2_masks)
        plane2_ct_traces = get_traces(plane2_path[0], plane2_masks)

        # subtract offset, flatten:
        nc = plane1_sig_traces.shape[0]
        plane1_sig_m0 = plane1_sig_traces - np.mean(plane1_sig_traces, axis=1).reshape(nc, 1)

        nc = plane1_ct_traces.shape[0]
        plane1_ct_m0 = plane1_ct_traces - np.mean(plane1_ct_traces, axis=1).reshape(nc, 1)

        nc = plane2_sig_traces.shape[0]
        plane2_sig_m0 = plane2_sig_traces - np.mean(plane2_sig_traces, axis=1).reshape(nc, 1)
        nc = plane2_ct_traces.shape[0]
        plane2_ct_m0 = plane2_ct_traces - np.mean(plane2_ct_traces, axis=1).reshape(nc, 1)

        if (not np.max(plane1_sig_traces) == 0) and (not np.max(plane1_ct_traces) == 0):
            found_Traces[0] = True

        if (not np.max(plane2_sig_traces) == 0) and (not np.max(plane2_ct_traces) == 0):
            found_Traces[1] = True

        return plane1_sig_m0, plane1_ct_m0, plane2_sig_m0, plane2_ct_m0, found_Traces