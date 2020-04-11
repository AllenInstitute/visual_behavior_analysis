import matplotlib
from allensdk.brain_observatory import roi_masks
import visual_behavior.ophys.mesoscope.dataset as ms
import allensdk.internal.core.lims_utilities as lu
import os
import h5py
import numpy as np
import logging
import json
from sklearn.decomposition import FastICA
import scipy.optimize as opt
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import allensdk.core.json_utilities as ju
from scipy import linalg
from scipy.stats import linregress
from matplotlib.colors import LogNorm
import visual_behavior.ophys.mesoscope.active_traces as at
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
import pandas as pd

logger = logging.getLogger(__name__)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

IMAGEH, IMAGEW = 512, 512
CELL_EXTRACT_JSON_FORMAT = ['OPHYS_EXTRACT_TRACES_QUEUE_%s_input.json', 'processed/%s_input_extract_traces.json']
ROI_NAME = "roi_ica"
NP_NAME = "neuropil_ica"
VBA_CACHE = "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis"

def get_traces(movie_exp_dir, movie_exp_id, mask_exp_dir, mask_exp_id):
    """
    Functions to read existing traces from LIMS
    :param movie_exp_dir: str, LIMS directory for the movie to apply ROIS masks to
    :param movie_exp_id: int,  LIMS experiment ID for the movie
    :param mask_exp_dir: LIMS directory to read masks file from
    :param mask_exp_id: LIMS experiment ID for the rois mask file
    :return: traces, neuropil traces, roi_names
    """
    for filename in CELL_EXTRACT_JSON_FORMAT:
        jin_mask_path = os.path.join(mask_exp_dir, filename % mask_exp_id)
        if os.path.isfile(jin_mask_path):
            break
    else:
        raise ValueError('Cell extract json does not exist')

    for filename in CELL_EXTRACT_JSON_FORMAT:
        jin_movie_path = os.path.join(movie_exp_dir, filename % movie_exp_id)
        if os.path.isfile(jin_movie_path):
            break
    else:
        raise ValueError('Cell extract json does not exist')

    with open(jin_movie_path, "r") as f:
        jin_movie = json.load(f)

    motion_border = jin_movie.get("motion_border", [])
    motion_border = [motion_border["x0"], motion_border["x1"], motion_border["y0"], motion_border["y1"], ]
    movie_h5 = jin_movie["motion_corrected_stack"]
    with h5py.File(movie_h5, "r") as f:
        d = f["data"]
        h = d.shape[1]
        w = d.shape[2]

    # reading traces extracting json for masks
    with open(jin_mask_path, "r") as f:
        jin_mask = json.load(f)

    rois = jin_mask["rois"]

    roi_mask_list = create_roi_masks(rois, w, h, motion_border)
    roi_names = [roi.label for roi in roi_mask_list]

    traces, neuropil_traces, excl = roi_masks.calculate_roi_and_neuropil_traces(movie_h5, roi_mask_list, motion_border)
    return traces, neuropil_traces, roi_names


def create_roi_masks(rois, w, h, motion_border):
    """
    create roi masks from roi list, width, height and motion border
    :param rois:
    :param w:
    :param h:
    :param motion_border:
    :return: roi list
    """
    roi_list = []
    for roi in rois:
        mask = np.array(roi["mask"], dtype=bool)
        px = np.argwhere(mask)
        px[:, 0] += roi["y"]
        px[:, 1] += roi["x"]

        mask = roi_masks.create_roi_mask(w, h, motion_border,
                                         pix_list=px[:, [1, 0]],
                                         label=str(roi["id"]),
                                         mask_group=roi.get("mask_page", -1))
        roi_list.append(mask)
    # sort by roi id
    roi_list.sort(key=lambda x: x.label)
    return roi_list


class MesoscopeICA(object):
    """
    Class to perform ica-based demixing on a pair of mesoscope pls
    """

    def __init__(self, session_id, cache, debug_mode=False, roi_name="roi_ica", np_name="neuropil_ica"):
        """
        :param session_id: LIMS session ID
        :param cache: directory to store/find ins/outs
        :param debug_mode: flag that controls whether debug logger messages are out
        :param roi_name: string, default name for roi-related in/outs
        :param np_name: string, default name for neuropil-related ins/outs
        """
        # self.tkeys = ['roi', 'np']
        # self.pkeys = ['pl1', 'pl2']

        self.session_id = session_id
        self.dataset = ms.MesoscopeDataset(session_id)
        self.session_cache_dir = None
        self.debug_mode = debug_mode

        # self.names = {'roi':roi_name, 'np':np_name}
        self.roi_name = roi_name  # prefix for files related to roi traces
        self.np_name = np_name  # prefix for files related to nuropil traces
        self.cache = cache  # analysis directory

        # self.exp_ids = {key:None for key in self.pkeys}
        self.pl1_exp_id = None  # plane 1 experiment id
        self.pl2_exp_id = None  # plane 2 experiment id

        self.dirs = {key:None for key in self.tkeys}
        self.roi_dir = None  # path to subdir containing roi-related files
        self.np_dir = None  # path to subdir containing neuropil-related files

        # pointers and attributes related to raw traces

        # self.raws = {}
        # self.paths = {}
        # for pkey in self.pkeys:
        #     self.raws[pkey] = {}
        #     for tkey in self.tkeys:
        #         self.raws[pkey][tkey] = None
        #
        # self.raws['pl1']['roi']

        self.pl1_roi_raw = None  # raw extracted traces for rois, plane 1
        self.pl1_roi_raw_path = None  # path to raw extracted traces for rois, plane 1

        self.pl2_roi_raw = None  # raw extracted traces for rois, plane 2
        self.pl2_roi_raw_path = None  # path to raw extracted traces for rois, plane 2

        self.pl1_np_raw = None  # raw extracted traces for neuropil, plane 1
        self.pl1_np_raw_path = None  # path to raw extracted traces for neuropil, plane 1

        self.pl2_np_raw = None  # raw extracted traces for rois, plane 2
        self.pl2_np_raw_path = None  # path to raw extracted traces for neuropil, plane 2

        self.found_raw_roi_traces = None  # flag if raw roi traces exist in self.roi_dir output of get_traces
        self.found_raw_np_traces = None  # flag if raw neuropil tarces exist in self.np_dir output of get_traces

        # pointers and attributes for validation jsons
        # only roi because they are the same for neuropil
        self.pl1_roi_names = None
        self.pl2_roi_names = None
        self.pl1_rois_valid = None
        self.pl2_rois_valid = None
        self.pl1_rois_valid_path = None
        self.pl2_rois_valid_path = None

        # pointers and attributes related to ica input traces
        # for roi
        self.pl1_roi_in = None  # debiased roi traces, plane 1
        self.pl2_roi_in = None  # debiased roi traces, plane 2
        self.pl1_roi_in_path = None  # path to debiased roi traces, plane 1
        self.pl2_roi_in_path = None  # path to debiased roi traces, plane 2
        self.pl1_roi_offset = None  # offsets for roi traces, plane 1
        self.pl2_roi_offset = None  # offsets for roi traces, plane 2
        self.found_roi_in = [None, None]  # flag for traces exists, roi
        self.found_roi_offset = [None, None]  # flag for offset data exists, roi
        # for neuropil
        self.pl1_np_in = None  # debiased neuropil traces, plane 1
        self.pl2_np_in = None  # debiased neuropil traces, plane 2
        self.pl1_np_in_path = None  # path to debiased neuropil traces, plane 1
        self.pl2_np_in_path = None  # path to debiased neuropil traces, plane 2
        self.pl1_np_offset = None  # offsets for neuropil traces, plane 1
        self.pl2_np_offset = None  # offsets for neuropil traces, plane 2
        self.found_np_in = [None, None]  # flag for traces exists, neuropil
        self.found_np_offset = [None, None]  # flag for offset data exists, neuropil

        # pointers and attirbutes for ica output files
        # for roi
        self.pl1_roi_out = None  # demixed roi traces, plane 1
        self.pl2_roi_out = None  # demixed roi traces, plane 2
        self.pl1_roi_out_path = None  # path to demixed roi traces, plane 1
        self.pl2_roi_out_path = None  # path to demixed roi traces, plane 2
        self.pl1_roi_crosstalk = None  # crosstalk data for roi traces, plane 1
        self.pl2_roi_crosstalk = None  # crosstalk data for roi traces, plane 2
        # for neuropil
        self.pl1_np_out = None  # demixed neuropil traces, plane 1
        self.pl2_np_out = None  # demixed neuropil traces, plane 2
        self.pl1_np_out_path = None  # path to demixed neuropil traces, plane 1
        self.pl2_np_out_path = None  # path to demixed neuropil traces, plane 2
        self.pl1_np_crosstalk = None  # crosstalk data for neuropil traces, plane 1
        self.pl2_np_crosstalk = None  # crosstalk data for neuropil traces, plane 2


        self.found_solution = None  # out of unmix_pls
        self.found_solution_neuropil = None
        self.roi_matrix = None
        self.neuropil_matrix = None
        self.roi_unmix = None
        self.neuropil_unmix = None
        self.pl1_roi_err = None
        self.pl2_roi_err = None
        self.pl1_np_err = None
        self.pl2_np_err = None
        self.neuropil_ica_out = None
        self.ica_mixing_matrix_traces_path = None
        self.roi_ica_out = None
        self.pl1_ica_out = None
        self.pl2_ica_out = None

    def set_analysis_session_dir(self):
        """
        crete path to the session-level dir
        :return: string - path to session level dir
        """
        self.session_cache_dir = os.path.join(self.cache, f'session_{self.session_id}')
        return self.session_cache_dir

    def set_ica_roi_dir(self, pair, roi_name=None):
        """
        create path to ica-related inputs/outs for the pair
        :param pair: list[int, int] - pair of LIMS exp IDs
        :param roi_name: roi_nam if different form self.roi_name to use to locate old inputs/outs
        :return: None
        """
        if not roi_name:
            roi_name = self.roi_name

        session_dir = self.set_analysis_session_dir()
        self.roi_dir = os.path.join(session_dir, f'{roi_name}_{pair[0]}_{pair[1]}/')
        self.pl1_roi_out_path = os.path.join(self.roi_dir,
                                             f'{self.roi_name}_out_{pair[0]}.h5')
        self.pl2_roi_out_path = os.path.join(self.roi_dir,
                                             f'{self.roi_name}_out_{pair[1]}.h5')

        return

    def set_ica_neuropil_dir(self, pair, np_name=None):
        """
        create path to neuropil-related inputs/outs for the pair
        :param pair: list[int, int] - pair of LIMS exp IDs
        :param np_name: roi_nam if different form self.roi_name to use to locate old inputs/outs
        :return: None
        """
        if not np_name:
            np_name = self.np_name

        session_dir = self.set_analysis_session_dir()
        self.np_dir = os.path.join(session_dir, f'{np_name}_{pair[0]}_{pair[1]}/')
        self.pl1_np_out_path = os.path.join(self.np_dir,
                                            f'{self.np_name}_out_{pair[0]}.h5')
        self.pl2_np_out_path = os.path.join(self.np_dir,
                                            f'{self.np_name}_out_{pair[1]}.h5')
        self.ica_mixing_matrix_neuropil_path = os.path.join(self.np_dir, f'{self.np_name}_mixing.h5')

        return

    def get_ica_traces(self, pair, roi_dir_name=None, np_dir_name=None):
        """
        function to apply roi set to two image pls, first check if the traces have been extracted before,
        can use a different roi_name, if traces don't exist in cache, read roi set name form LIMS< apply to both signal and crosstalk pls
        :param pair: list[int, int] : LIMS exp IDs for the pair
        :param roi_dir_name: string, new name for roi-related files to use, different form self.roi_name
        :param np_dir_name: string, new name for neuropil-related files to use, if need to be different form self.np_name
        :return: list[bool bool]: flags to see if traces where extracted successfully
        """
        if not roi_dir_name:
            roi_dir_name = self.roi_name

        if not np_dir_name:
            np_dir_name = self.np_name

        if self.debug_mode:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        # we will first check if traces exist, if yes - read them, if not - extract them
        self.pl1_roi_names = None
        self.pl2_roi_names = None

        self.found_raw_roi_traces = [False, False]
        self.found_raw_np_traces = [False, False]

        pl1_exp_id = pair[0]
        pl2_exp_id = pair[1]

        self.pl1_exp_id = pl1_exp_id
        self.pl2_exp_id = pl2_exp_id

        session_dir = os.path.join(self.cache, f'session_{self.session_id}')
        self.session_cache_dir = session_dir

        # path to ica traces:
        # for roi
        ica_traces_dir = os.path.join(session_dir, f'{roi_dir_name}_{pl1_exp_id}_{pl2_exp_id}/')
        self.roi_dir = ica_traces_dir
        path_traces_pl1 = f'{ica_traces_dir}traces_original_{pl1_exp_id}.h5'
        path_traces_pl2 = f'{ica_traces_dir}traces_original_{pl2_exp_id}.h5'
        # for neuropil
        ica_neuropil_dir = os.path.join(session_dir, f'{np_dir_name}_{pl1_exp_id}_{pl2_exp_id}/')
        self.np_dir = ica_neuropil_dir
        path_neuropil_pl1 = f'{ica_neuropil_dir}neuropil_original_{pl1_exp_id}.h5'
        path_neuropil_pl2 = f'{ica_neuropil_dir}neuropil_original_{pl2_exp_id}.h5'

        # let's see if all traces exist already:
        if os.path.isfile(path_traces_pl1) and os.path.isfile(path_traces_pl2) and os.path.isfile(
                path_neuropil_pl1) and os.path.isfile(path_neuropil_pl2):
            # if both traces exist, skip extracting:

            logger.info('Found traces in cache, reading from h5 file')
            # read traces form h5 file:
            with h5py.File(path_traces_pl1, "r") as f:
                pl1_traces_original = f["data"][()]
                pl1_roi_names = f["roi_names"][()]
            with h5py.File(path_traces_pl2, "r") as f:
                pl2_traces_original = f["data"][()]
                pl2_roi_names = f["roi_names"][()]

            # read neuropil traces form h5 file:
            with h5py.File(path_neuropil_pl1, "r") as f:
                pl1_neuropil_original = f["data"][()]

            with h5py.File(path_neuropil_pl2, "r") as f:
                pl2_neuropil_original = f["data"][()]

            self.pl1_roi_raw_path = path_traces_pl1
            self.pl2_roi_raw_path = path_traces_pl2
            self.pl1_roi_raw = pl1_traces_original
            self.pl2_roi_raw = pl2_traces_original
            self.pl1_roi_names = pl1_roi_names
            self.pl2_roi_names = pl2_roi_names

            #  same for neuropil:
            self.pl1_np_raw_path = path_neuropil_pl1
            self.pl2_np_raw_path = path_neuropil_pl2
            self.pl1_np_raw = pl1_neuropil_original
            self.pl2_np_raw = pl2_neuropil_original
            # set found traces flag True
            self.found_raw_roi_traces = [True, True]
            # set found neuropil traces flag True
            self.found_raw_np_traces = [True, True]

        else:
            # some traces are missing, run extraction:
            logger.info('Traces dont exist in cache, extracting')

            # if traces don't exist, do we need to reset unmixed and debiased traces flaggs to none?
            # yes, as we want unmixed traces be out on ICA using original traces
            self.pl1_roi_in_path = None
            self.pl2_roi_in_path = None
            self.pl1_roi_out_path = None
            self.pl2_roi_out_path = None

            self.pl1_np_in_path = None
            self.pl2_np_in_path = None
            self.pl1_np_out_path = None
            self.pl2_np_out_path = None

            self.pl1_roi_names = None
            self.pl2_roi_names = None

            pl1_folder = self.dataset.get_exp_folder(pl1_exp_id)
            pl2_folder = self.dataset.get_exp_folder(pl2_exp_id)

            # extract signal and crosstalk traces for pl 1
            pl1_sig_traces, pl1_sig_neuropil, pl1_roi_names = get_traces(pl1_folder, pl1_exp_id,
                                                                         pl1_folder, pl1_exp_id)
            pl1_ct_traces, pl1_ct_neuropil, _ = get_traces(pl2_folder, pl2_exp_id, pl1_folder,
                                                           pl1_exp_id)
            # extract signal and crosstalk traces for pl 2
            pl2_sig_traces, pl2_sig_neuropil, pl2_roi_names = get_traces(pl2_folder, pl2_exp_id,
                                                                         pl2_folder, pl2_exp_id)
            pl2_ct_traces, pl2_ct_neuropil, _ = get_traces(pl1_folder, pl1_exp_id, pl2_folder,
                                                           pl2_exp_id)

            # setting traces valid flag: if none is None
            if (not pl1_sig_traces.any() is None) and (not pl1_ct_traces.any() is None):
                self.found_raw_roi_traces[0] = True
            if (not pl2_sig_traces.any() is None) and (not pl2_ct_traces.any() is None):
                self.found_raw_roi_traces[1] = True
            if (not pl1_sig_neuropil.any() is None) and (not pl1_ct_neuropil.any() is None):
                self.found_raw_np_traces[0] = True
            if (not pl2_sig_neuropil.any() is None) and (not pl2_ct_neuropil.any() is None):
                self.found_raw_np_traces[1] = True

            # DOES ROI traces DIR EXIST?
            if not os.path.isdir(session_dir):
                os.mkdir(session_dir)
            if not os.path.isdir(ica_traces_dir):
                os.mkdir(ica_traces_dir)
            # if extracted traces valid, save to disk:
            if self.found_raw_roi_traces[0] and self.found_raw_roi_traces[1]:
                # combining traces, saving to self, writing to disk:
                pl1_traces_original = np.array([pl1_sig_traces, pl1_ct_traces])

                self.pl1_roi_raw = pl1_traces_original
                self.pl1_roi_raw_path = path_traces_pl1
                self.pl1_roi_names = pl1_roi_names
                with h5py.File(path_traces_pl1, "w") as f:
                    f.create_dataset(f"data", data=pl1_traces_original)
                    f.create_dataset(f"roi_names", data=np.int_(pl1_roi_names))

                # same for pl 2:
                pl2_traces_original = np.array([pl2_sig_traces, pl2_ct_traces])
                self.pl2_roi_raw = pl2_traces_original
                self.pl2_roi_raw_path = path_traces_pl2
                self.pl2_roi_names = pl2_roi_names
                with h5py.File(path_traces_pl2, "w") as f:
                    f.create_dataset(f"data", data=pl2_traces_original)
                    f.create_dataset(f"roi_names", data=np.int_(pl2_roi_names))

            if not os.path.isdir(session_dir):
                os.mkdir(session_dir)
            if not os.path.isdir(ica_neuropil_dir):
                os.mkdir(ica_neuropil_dir)

            # if extracted traces not None, save to disk:
            if self.found_raw_np_traces[0] and self.found_raw_np_traces[1]:
                # combining traces, saving to self, writing to disk:
                pl1_neuropil_original = np.array([pl1_sig_neuropil, pl1_ct_neuropil])
                self.pl1_np_raw = pl1_neuropil_original
                self.pl1_np_raw_path = path_neuropil_pl1
                with h5py.File(path_neuropil_pl1, "w") as f:
                    f.create_dataset(f"data", data=pl1_neuropil_original)
                    f.create_dataset(f"roi_names", data=np.int_(pl1_roi_names))
                # same for pl 2:
                pl2_neuropil_original = np.array([pl2_sig_neuropil, pl2_ct_neuropil])
                self.pl2_np_raw = pl2_neuropil_original
                self.pl2_np_raw_path = path_neuropil_pl2
                with h5py.File(path_neuropil_pl2, "w") as f:
                    f.create_dataset(f"data", data=pl2_neuropil_original)
                    f.create_dataset(f"roi_names", data=np.int_(pl2_roi_names))

        return self.found_raw_roi_traces, self.found_raw_np_traces

    def validate_traces(self, return_vba=True):
        """
        fn to check if the traces don't have Nans, writes {exp_id}_valid.json to cache for each pl in pair
        return_vba: bool, flag to control whether to validate against vba roi set or return ica roi set
        :return: None
        """
        if self.debug_mode:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        self.pl1_rois_valid_path = None
        self.pl2_rois_valid_path = None

        self.pl1_neuropil_traces_valid_path = None
        self.pl2_neuropil_traces_valid_path = None

        pl1_roi_traces_valid_path = os.path.join(self.roi_dir, f'{self.pl1_exp_id}_valid.json')
        pl2_roi_traces_valid_path = os.path.join(self.roi_dir, f'{self.pl2_exp_id}_valid.json')

        pl1_neuropil_traces_valid_path = os.path.join(self.np_dir, f'{self.pl1_exp_id}_valid.json')
        pl2_neuropil_traces_valid_path = os.path.join(self.np_dir, f'{self.pl2_exp_id}_valid.json')

        # validation json already exists, skip validating
        if os.path.isfile(pl1_roi_traces_valid_path) and os.path.isfile(
                pl2_roi_traces_valid_path) and os.path.isfile(
                pl1_neuropil_traces_valid_path) and os.path.isfile(
                pl2_neuropil_traces_valid_path):
            self.pl1_rois_valid_path = pl1_roi_traces_valid_path
            self.pl2_rois_valid_path = pl2_roi_traces_valid_path
            self.pl1_neuropil_traces_valid_path = pl1_neuropil_traces_valid_path
            self.pl2_neuropil_traces_valid_path = pl2_neuropil_traces_valid_path
        else:
            for pkey in self.pkeys:
                for tkey in self.tkeys:
                    self.valid_path.pkey.tkey = None

            self.pl1_rois_valid_path = None
            self.pl2_rois_valid_path = None
            self.pl1_neuropil_traces_valid_path = None
            self.pl2_neuropil_traces_valid_path = None

        if (not self.pl1_rois_valid_path) and (not self.pl2_rois_valid_path) and (
                not self.pl1_neuropil_traces_valid_path) and (not self.pl2_neuropil_traces_valid_path):

            # traces need validation:
            # traces exist?
            if self.found_raw_roi_traces[0] and self.found_raw_roi_traces[1] and self.found_raw_np_traces[0] and \
                    self.found_raw_np_traces[1]:
                # traces, pl 1:
                pl1_sig_trace_valid = {}
                pl1_sig_neuropil_valid = {}
                pl1_ct_trace_valid = {}
                pl1_ct_neuropil_valid = {}

                num_traces_roi_sig, _ = self.pl1_roi_raw[0].shape
                num_traces_neuropil_sig, _ = self.pl1_np_raw[0].shape
                num_traces_roi_ct, _ = self.pl1_roi_raw[1].shape
                num_traces_neuropil_ct, _ = self.pl1_np_raw[1].shape

                if not (num_traces_roi_sig == num_traces_neuropil_sig) or not (
                        num_traces_roi_ct == num_traces_neuropil_ct):
                    logger.info('Neuropil and ROI traces are not aligned')
                else:
                    for n in range(num_traces_roi_sig):
                        trace_roi_sig = self.pl1_roi_raw[0][n]
                        trace_neuropil_sig = self.pl1_np_raw[0][n]
                        trace_roi_ct = self.pl1_roi_raw[1][n]
                        trace_neuropil_ct = self.pl1_np_raw[1][n]

                        if np.any(np.isnan(trace_roi_sig)) or np.any(np.isnan(trace_neuropil_sig)) or np.any(
                                np.isnan(trace_roi_ct)) or np.any(np.isnan(trace_neuropil_ct)):
                            pl1_sig_trace_valid[str(self.pl1_roi_names[n])] = False
                            pl1_sig_neuropil_valid[str(self.pl1_roi_names[n])] = False
                            pl1_ct_trace_valid[str(self.pl1_roi_names[n])] = False
                            pl1_ct_neuropil_valid[str(self.pl1_roi_names[n])] = False
                        else:
                            pl1_sig_trace_valid[str(self.pl1_roi_names[n])] = True
                            pl1_sig_neuropil_valid[str(self.pl1_roi_names[n])] = True
                            pl1_ct_trace_valid[str(self.pl1_roi_names[n])] = True
                            pl1_ct_neuropil_valid[str(self.pl1_roi_names[n])] = True

                # traces, pl 2:
                # signal:
                pl2_sig_trace_valid = {}
                pl2_sig_neuropil_valid = {}
                pl2_ct_trace_valid = {}
                pl2_ct_neuropil_valid = {}

                num_traces_roi_sig, _ = self.pl2_roi_raw[0].shape
                num_traces_neuropil_sig, _ = self.pl2_np_raw[0].shape
                num_traces_roi_ct, _ = self.pl2_roi_raw[1].shape
                num_traces_neuropil_ct, _ = self.pl2_np_raw[1].shape

                if not (num_traces_roi_sig == num_traces_neuropil_sig) or not (
                        num_traces_roi_ct == num_traces_neuropil_ct):
                    logger.info('Neuropil and ROI traces are not aligned')
                else:
                    for n in range(num_traces_roi_sig):
                        trace_roi_sig = self.pl2_roi_raw[0][n]
                        trace_neuropil_sig = self.pl2_np_raw[0][n]
                        trace_roi_ct = self.pl2_roi_raw[1][n]
                        trace_neuropil_ct = self.pl2_np_raw[1][n]

                        if np.any(np.isnan(trace_roi_sig)) or np.any(np.isnan(trace_neuropil_sig)) or np.any(
                                np.isnan(trace_roi_ct)) or np.any(np.isnan(trace_neuropil_ct)):
                            pl2_sig_trace_valid[str(self.pl2_roi_names[n])] = False
                            pl2_sig_neuropil_valid[str(self.pl2_roi_names[n])] = False
                            pl2_ct_trace_valid[str(self.pl2_roi_names[n])] = False
                            pl2_ct_neuropil_valid[str(self.pl2_roi_names[n])] = False
                        else:
                            pl2_sig_trace_valid[str(self.pl2_roi_names[n])] = True
                            pl2_sig_neuropil_valid[str(self.pl2_roi_names[n])] = True
                            pl2_ct_trace_valid[str(self.pl2_roi_names[n])] = True
                            pl2_ct_neuropil_valid[str(self.pl2_roi_names[n])] = True

                # combining dictionaries for signal and crosstalk
                pl1_roi_traces_valid = {"signal": pl1_sig_trace_valid,
                                        "crosstalk": pl1_ct_trace_valid}
                pl2_roi_traces_valid = {"signal": pl2_sig_trace_valid,
                                        "crosstalk": pl2_ct_trace_valid}

                pl1_neuropil_traces_valid = {"signal": pl1_sig_neuropil_valid,
                                             "crosstalk": pl1_ct_neuropil_valid}
                pl2_neuropil_traces_valid = {"signal": pl2_sig_neuropil_valid,
                                             "crosstalk": pl2_ct_neuropil_valid}

                # validating agains VBA rois set:
                if return_vba:
                    pl1_roi_traces_valid = self.validate_against_vba(pl1_roi_traces_valid,
                                                                     self.pl1_exp_id, VBA_CACHE)
                    pl2_roi_traces_valid = self.validate_against_vba(pl2_roi_traces_valid,
                                                                     self.pl2_exp_id, VBA_CACHE)

                    pl1_neuropil_traces_valid = self.validate_against_vba(pl1_neuropil_traces_valid,
                                                                          self.pl1_exp_id, VBA_CACHE)
                    pl2_neuropil_traces_valid = self.validate_against_vba(pl2_neuropil_traces_valid,
                                                                          self.pl2_exp_id, VBA_CACHE)
                # saving to json:

                self.pl1_rois_valid_path = pl1_roi_traces_valid_path
                ju.write(pl1_roi_traces_valid_path, pl1_roi_traces_valid)
                self.pl1_rois_valid = pl1_roi_traces_valid

                self.pl2_rois_valid_path = pl2_roi_traces_valid_path
                ju.write(pl2_roi_traces_valid_path, pl2_roi_traces_valid)
                self.pl2_rois_valid = pl2_roi_traces_valid

                self.pl1_neuropil_traces_valid_path = pl1_neuropil_traces_valid_path
                ju.write(pl1_neuropil_traces_valid_path, pl1_neuropil_traces_valid)
                self.pl1_neuropil_traces_valid = pl1_neuropil_traces_valid

                self.pl2_neuropil_traces_valid_path = pl2_neuropil_traces_valid_path
                ju.write(pl2_neuropil_traces_valid_path, pl2_neuropil_traces_valid)
                self.pl2_neuropil_traces_valid = pl2_neuropil_traces_valid

            else:
                logger.info('ROI traces dont exist in cache, run get_ica_traces first')
        else:
            # traces have been validated

            # read the jsons for ROIs
            pl1_roi_traces_valid = ju.read(pl1_roi_traces_valid_path)
            pl2_roi_traces_valid = ju.read(pl2_roi_traces_valid_path)
            self.pl1_rois_valid = pl1_roi_traces_valid
            self.pl2_rois_valid = pl2_roi_traces_valid

            # read the jsons for neuropil
            pl1_neuropil_traces_valid = ju.read(pl1_neuropil_traces_valid_path)
            pl2_neuropil_traces_valid = ju.read(pl2_neuropil_traces_valid_path)
            self.pl1_neuropil_traces_valid = pl1_neuropil_traces_valid
            self.pl2_neuropil_traces_valid = pl2_neuropil_traces_valid

        return

    def debias_rois(self):
        """
        fn to combine all roi traces for the pair to two num_cells x num_frames_in_timeseries vectors,
        write them to cache as ica_roi_input
        :return: None
        """

        if self.debug_mode:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        self.pl1_roi_in_path = None
        self.pl2_roi_in_path = None
        pl1_ica_in_path = os.path.join(self.roi_dir, f'{self.pl1_exp_id}_in.h5')
        pl2_ica_in_path = os.path.join(self.roi_dir, f'{self.pl2_exp_id}_in.h5')
        if os.path.isfile(pl1_ica_in_path) and os.path.isfile(pl2_ica_in_path):
            # file already exists, skip debiasing
            self.pl1_roi_in_path = pl1_ica_in_path
            self.pl2_roi_in_path = pl2_ica_in_path
        else:
            self.pl1_roi_in_path = None
            self.pl2_roi_in_path = None
        # original traces exist, run debiasing:
        if self.found_raw_roi_traces[0] and self.found_raw_roi_traces[1]:
            # if debiased traces don't exist, run debiasing - paths are both None
            if (not self.pl1_roi_in_path) and (not self.pl2_roi_in_path):
                self.found_roi_in = [False, False]
                self.found_roi_offset = [False, False]
                logger.info("Debiased ROI traces do not exist in cache, running offset subtraction")

                pl1_sig = self.pl1_roi_raw[0]
                pl1_ct = self.pl1_roi_raw[1]
                pl1_valid = self.pl1_rois_valid

                pl1_valid_sig = pl1_valid['signal']
                pl1_valid_ct = pl1_valid['crosstalk']

                pl2_sig = self.pl2_roi_raw[0]
                pl2_ct = self.pl2_roi_raw[1]

                pl2_valid = self.pl2_rois_valid
                pl2_valid_sig = pl2_valid['signal']
                pl2_valid_ct = pl2_valid['crosstalk']

                # only include cells that don't have nans  (valid = True)
                # check if traces aligned:
                if len(self.pl1_roi_names) == len(pl1_sig):
                    pl1_sig_valid_idx = np.array([pl1_valid_sig[str(tid)] for tid in self.pl1_roi_names])
                    pl1_sig_valid = pl1_sig[pl1_sig_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.pl1_roi_names) == len(pl1_ct):
                    pl1_ct_valid_idx = np.array([pl1_valid_ct[str(tid)] for tid in self.pl1_roi_names])
                    pl1_ct_valid = pl1_ct[pl1_ct_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.pl2_roi_names) == len(pl2_sig):
                    pl2_sig_valid_idx = np.array([pl2_valid_sig[str(tid)] for tid in self.pl2_roi_names])
                    pl2_sig_valid = pl2_sig[pl2_sig_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.pl2_roi_names) == len(pl2_ct):
                    pl2_ct_valid_idx = np.array([pl2_valid_ct[str(tid)] for tid in self.pl2_roi_names])
                    pl2_ct_valid = pl2_ct[pl2_ct_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                pl1_sig = pl1_sig_valid
                pl1_ct = pl1_ct_valid
                pl2_sig = pl2_sig_valid
                pl2_ct = pl2_ct_valid

                # subtract offset pl 1:
                nc = pl1_sig.shape[0]
                pl1_sig_offset = np.mean(pl1_sig, axis=1).reshape(nc, 1)
                pl1_sig_m0 = pl1_sig - pl1_sig_offset
                nc = pl1_ct.shape[0]
                pl1_ct_offset = np.mean(pl1_ct, axis=1).reshape(nc, 1)
                pl1_ct_m0 = pl1_ct - pl1_ct_offset
                # subtract offset for pl 2:
                nc = pl2_sig.shape[0]
                pl2_sig_offset = np.mean(pl2_sig, axis=1).reshape(nc, 1)
                pl2_sig_m0 = pl2_sig - pl2_sig_offset
                nc = pl2_ct.shape[0]
                pl2_ct_offset = np.mean(pl2_ct, axis=1).reshape(nc, 1)
                pl2_ct_m0 = pl2_ct - pl2_ct_offset
                # check if traces aren't none
                if (not pl1_sig_m0.any() is None) and (not pl1_ct_m0.any() is None):
                    self.found_roi_in[0] = True
                if (not pl2_sig_m0.any() is None) and (not pl2_ct_m0.any() is None):
                    self.found_roi_in[1] = True
                if (not pl1_sig_offset.any() is None) and (not pl1_ct_offset.any() is None):
                    self.found_roi_offset[1] = True
                if (not pl2_sig_offset.any() is None) and (not pl2_ct_offset.any() is None):
                    self.found_roi_offset[1] = True
                # if all flags true, combine, flatten, write to disk:

                if self.found_roi_in and self.found_roi_offset:
                    self.pl1_roi_offset = {'pl1_sig_offset': pl1_sig_offset, 'pl1_ct_offset': pl1_ct_offset}
                    self.pl2_roi_offset = {'pl2_sig_offset': pl2_sig_offset, 'pl2_ct_offset': pl2_ct_offset}
                    self.pl1_roi_in = np.array([pl1_sig_m0, pl1_ct_m0])
                    self.pl2_roi_in = np.array([pl2_sig_m0, pl2_ct_m0])
                    self.pl1_roi_in_path = pl1_ica_in_path
                    self.pl2_roi_in_path = pl2_ica_in_path
                    # write ica input traces to disk
                    with h5py.File(self.pl1_roi_in_path, "w") as f:
                        f.create_dataset("debiased_traces", data=self.pl1_roi_in)
                        f.create_dataset('sig_offset', data=pl1_sig_offset)
                        f.create_dataset('ct_offset', data=pl1_ct_offset)
                    with h5py.File(self.pl2_roi_in_path, "w") as f:
                        f.create_dataset("debiased_traces", data=self.pl2_roi_in)
                        f.create_dataset('sig_offset', data=pl2_sig_offset)
                        f.create_dataset('ct_offset', data=pl2_ct_offset)
                else:
                    logger.info("ROI traces Debiasing failed")
            else:
                logger.info("Debiased ROI traces exist in cache, reading from h5 file")
                with h5py.File(self.pl1_roi_in_path, "r") as f:
                    pl1_ica_in = f["debiased_traces"][()]
                    pl1_sig_offset = f['sig_offset'][()]
                    pl1_ct_offset = f['ct_offset'][()]
                with h5py.File(self.pl2_roi_in_path, "r") as f:
                    pl2_ica_in = f["debiased_traces"][()]
                    pl2_sig_offset = f['sig_offset'][()]
                    pl2_ct_offset = f['ct_offset'][()]

                self.found_roi_in = [True, True]
                self.pl1_roi_in = pl1_ica_in
                self.pl2_roi_in = pl2_ica_in
                self.pl1_roi_in_path = pl1_ica_in_path
                self.pl2_roi_in_path = pl2_ica_in_path
                self.pl1_roi_offset = {'pl1_sig_offset': pl1_sig_offset, 'pl1_ct_offset': pl1_ct_offset}
                self.pl2_roi_offset = {'pl2_sig_offset': pl2_sig_offset, 'pl2_ct_offset': pl2_ct_offset}
                self.found_roi_offset = [True, True]
        else:
            raise ValueError('Extract ROI traces first')
        return

    def debias_np(self):
        """
        fn to combine all roi traces for the pair to two num_cells x num_frames_in_timeseries vectors,
        write them to cache as ica_roi_input
        :return: None
        """

        if self.debug_mode:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        self.pl1_np_in_path = None
        self.pl2_np_in_path = None
        pl1_ica_in_path = os.path.join(self.np_dir, f'{self.pl1_exp_id}_in.h5')
        pl2_ica_in_path = os.path.join(self.np_dir, f'{self.pl2_exp_id}_in.h5')
        if os.path.isfile(pl1_ica_in_path) and os.path.isfile(pl2_ica_in_path):
            # file already exists, skip debiasing
            self.pl1_np_in_path = pl1_ica_in_path
            self.pl2_np_in_path = pl2_ica_in_path
        else:
            self.pl1_np_in_path = None
            self.pl2_np_in_path = None
        # original traces exist, run debiasing:
        if self.found_raw_np_traces[0] and self.found_raw_np_traces[1]:
            # if debiased traces don't exist, run debiasing - paths are both None
            if (not self.pl1_np_in_path) and (not self.pl2_np_in_path):
                self.found_np_in = [False, False]
                self.found_np_offset = [False, False]
                logger.info("Debiased neuropil traces do not exist in cache, running offset subtraction")

                pl1_sig = self.pl1_np_raw[0]
                pl1_ct = self.pl1_np_raw[1]
                pl1_valid = self.pl1_rois_valid
                pl1_valid_sig = pl1_valid['signal']
                pl1_valid_ct = pl1_valid['crosstalk']

                pl2_sig = self.pl2_np_raw[0]
                pl2_ct = self.pl2_np_raw[1]
                pl2_valid = self.pl2_rois_valid
                pl2_valid_sig = pl2_valid['signal']
                pl2_valid_ct = pl2_valid['crosstalk']

                # only include cells that don't have nans  (valid = True)
                # check if traces aligned:
                if len(self.pl1_roi_names) == len(pl1_sig):
                    pl1_sig_valid_idx = np.array([pl1_valid_sig[str(tid)] for tid in self.pl1_roi_names])
                    pl1_sig_valid = pl1_sig[pl1_sig_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.pl1_roi_names) == len(pl1_ct):
                    pl1_ct_valid_idx = np.array([pl1_valid_ct[str(tid)] for tid in self.pl1_roi_names])
                    pl1_ct_valid = pl1_ct[pl1_ct_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.pl2_roi_names) == len(pl2_sig):
                    pl2_sig_valid_idx = np.array([pl2_valid_sig[str(tid)] for tid in self.pl2_roi_names])
                    pl2_sig_valid = pl2_sig[pl2_sig_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.pl2_roi_names) == len(pl2_ct):
                    pl2_ct_valid_idx = np.array([pl2_valid_ct[str(tid)] for tid in self.pl2_roi_names])
                    pl2_ct_valid = pl2_ct[pl2_ct_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                pl1_sig = pl1_sig_valid
                pl1_ct = pl1_ct_valid
                pl2_sig = pl2_sig_valid
                pl2_ct = pl2_ct_valid

                # subtract offset pl 1:
                nc = pl1_sig.shape[0]
                pl1_sig_offset = np.mean(pl1_sig, axis=1).reshape(nc, 1)
                pl1_sig_m0 = pl1_sig - pl1_sig_offset
                nc = pl1_ct.shape[0]
                pl1_ct_offset = np.mean(pl1_ct, axis=1).reshape(nc, 1)
                pl1_ct_m0 = pl1_ct - pl1_ct_offset
                # subtract offset for pl 2:
                nc = pl2_sig.shape[0]
                pl2_sig_offset = np.mean(pl2_sig, axis=1).reshape(nc, 1)
                pl2_sig_m0 = pl2_sig - pl2_sig_offset
                nc = pl2_ct.shape[0]
                pl2_ct_offset = np.mean(pl2_ct, axis=1).reshape(nc, 1)
                pl2_ct_m0 = pl2_ct - pl2_ct_offset
                # check if traces aren't none
                if (not pl1_sig_m0.any() is None) and (not pl1_ct_m0.any() is None):
                    self.found_np_in[0] = True
                if (not pl2_sig_m0.any() is None) and (not pl2_ct_m0.any() is None):
                    self.found_np_in[1] = True
                if (not pl1_sig_offset.any() is None) and (not pl1_ct_offset.any() is None):
                    self.found_np_offset[1] = True
                if (not pl2_sig_offset.any() is None) and (not pl2_ct_offset.any() is None):
                    self.found_np_offset[1] = True
                # if all flags true, combine, flatten, write to disk:

                if self.found_np_in and self.found_np_offset:
                    self.pl1_np_offset = {'pl1_sig_offset': pl1_sig_offset, 'pl1_ct_offset': pl1_ct_offset}
                    self.pl2_np_offset = {'pl2_sig_offset': pl2_sig_offset, 'pl2_ct_offset': pl2_ct_offset}
                    self.pl1_np_in = np.array([pl1_sig_m0, pl1_ct_m0])
                    self.pl2_np_in = np.array([pl2_sig_m0, pl2_ct_m0])
                    self.pl1_np_in_path = pl1_ica_in_path
                    self.pl2_np_in_path = pl2_ica_in_path
                    # write ica input traces to disk
                    with h5py.File(self.pl1_np_in_path, "w") as f:
                        f.create_dataset("debiased_traces", data=self.pl1_np_in)
                        f.create_dataset('sig_offset', data=pl1_sig_offset)
                        f.create_dataset('ct_offset', data=pl1_ct_offset)
                    with h5py.File(self.pl2_np_in_path, "w") as f:
                        f.create_dataset("debiased_traces", data=self.pl2_np_in)
                        f.create_dataset('sig_offset', data=pl2_sig_offset)
                        f.create_dataset('ct_offset', data=pl2_ct_offset)
                else:
                    logger.info("Neuropil traces Debiasing failed")
            else:
                logger.info("Debiased neuropil traces exist in cache, reading from h5 file")
                with h5py.File(self.pl1_np_in_path, "r") as f:
                    pl1_ica_in = f["debiased_traces"][()]
                    pl1_sig_offset = f['sig_offset'][()]
                    pl1_ct_offset = f['ct_offset'][()]
                with h5py.File(self.pl2_np_in_path, "r") as f:
                    pl2_ica_in = f["debiased_traces"][()]
                    pl2_sig_offset = f['sig_offset'][()]
                    pl2_ct_offset = f['ct_offset'][()]

                self.found_np_in = [True, True]
                self.pl1_np_in = pl1_ica_in
                self.pl2_np_in = pl2_ica_in
                self.pl1_np_in_path = pl1_ica_in_path
                self.pl2_np_in_path = pl2_ica_in_path
                self.pl1_np_offset = {'pl1_sig_offset': pl1_sig_offset, 'pl1_ct_offset': pl1_ct_offset}
                self.pl2_np_offset = {'pl2_sig_offset': pl2_sig_offset, 'pl2_ct_offset': pl2_ct_offset}
                self.found_np_offset = [True, True]
        else:
            raise ValueError('Neuropil traces do not exist - extract them first')
        return


    @staticmethod
    def unmix_plane(ica_in, ica_roi_valid):
        roi_names = [roi for roi, valid in ica_roi_valid.items() if valid]
        traces_sig = ica_in[0, :, :]
        traces_ct = ica_in[1, :, :]
        # extract events
        traces_sig_evs, traces_ct_evs = extract_active(ica_in, len_ne=20, th_ag=10, do_plots=0)
        pl_crosstalk = np.empty((2, ica_in.shape[1]))
        pl_mixing = []
        figs_ct_in = []
        figs_ct_out = []
        # run ica on active traces, apply unmixing matrix to entire trace
        ica_pl_out = np.empty(ica_in.shape)
        for i in range(len(roi_names)):
            # get roi names
            roi_name = roi_names[i]
            # get events traces
            trace_sig_evs = traces_sig_evs[i]
            trace_ct_evs = traces_ct_evs[i]
            mix, a_mix, a_unmix, r_sources_evs = run_ica(trace_sig_evs, trace_ct_evs)
            adjusted_mixing_matrix = a_mix

            trace_sig_evs_out = r_sources_evs[:, 0]
            trace_ct_evs_out = r_sources_evs[:, 1]
            rescaled_trace_sig_evs_out = rescale(trace_sig_evs, trace_sig_evs_out)
            rescaled_trace_ct_evs_out = rescale(trace_ct_evs, trace_ct_evs_out)
            # calculating crosstalk : on events
            fig_ct_in, slope_in, _, r_value_in = plot_pixel_hist2d(trace_sig_evs, trace_ct_evs,
                                                                   title=f'raw, cell {roi_name}',
                                                                   save_fig=False,
                                                                   save_path=None,
                                                                   fig_show=False,
                                                                   colorbar=True)
            fig_ct_out, slope_out, _, r_value_out = plot_pixel_hist2d(rescaled_trace_sig_evs_out,
                                                                      rescaled_trace_ct_evs_out,
                                                                      title=f'clean, cell {roi_name}',
                                                                      save_fig=False,
                                                                      save_path=None,
                                                                      fig_show=False,
                                                                      colorbar=True)
            crosstalk_before_demixing_evs = slope_in * 100
            crosstalk_after_demixing_evs = slope_out * 100
            pl_crosstalk[:, i] = [crosstalk_before_demixing_evs, crosstalk_after_demixing_evs]
            pl_mixing.append(adjusted_mixing_matrix)

            # applying unmixing matrix to the entire trace
            trace_sig = traces_sig[i]
            trace_ct = traces_ct[i]
            # recontructing sources
            traces = np.array([trace_sig, trace_ct]).T
            r_sources = np.dot(a_unmix, traces.T).T
            trace_sig_out = r_sources[:, 0]
            trace_ct_out = r_sources[:, 1]
            rescaled_trace_sig_out = rescale(trace_sig, trace_sig_out)
            rescaled_trace_ct_out = rescale(trace_ct, trace_ct_out)
            ica_pl_out[0, i, :] = rescaled_trace_sig_out
            ica_pl_out[1, i, :] = rescaled_trace_ct_out
            figs_ct_in.append(fig_ct_in)
            figs_ct_out.append(fig_ct_out)

        return ica_pl_out, pl_crosstalk, pl_mixing, figs_ct_in, figs_ct_out

    def unmix_pair(self, do_plots=True):
        """
        function to unmix a pair of panels:
        """
        pl1_out_path = os.path.join(self.roi_dir,
                                    f'{self.pl1_exp_id}_out.h5')
        pl2_out_path = os.path.join(self.roi_dir,
                                    f'{self.pl2_exp_id}_out.h5')

        # file already exists, skip unmixing
        if os.path.isfile(pl1_out_path) and os.path.isfile(pl2_out_path):
            self.pl1_roi_out_path = pl1_out_path
            self.pl2_roi_out_path = pl2_out_path
        else:
            self.pl1_roi_out_path = None
            self.pl2_roi_out_path = None

        if (self.pl1_roi_out_path is None) or (self.pl2_roi_out_path is None):
            # if unmixed traces don't exist, run unmixing
            if np.any(np.isnan(self.pl1_roi_in)) or np.any(np.isinf(self.pl2_roi_in)) or np.any(
                    np.isnan(self.pl1_roi_in)) or np.any(np.isinf(self.pl2_roi_in)):
                raise ValueError(
                    "ValueError: ICA input contains NaN, infinity or a value too large for dtype('float64')")
            else:
                logger.info("Unmixed traces do not exist in cache, running ICA")

                # unmixing pl 1
                pl1_roi_valid = self.pl1_rois_valid['signal']
                pl1_traces_in = self.pl1_roi_in
                pl1_out, pl1_crosstalk, pl1_mixing, _, _ = self.unmix_plane(pl1_traces_in, pl1_roi_valid)

                # unmixing pl 2
                pl2_roi_valid = self.pl2_rois_valid['signal']
                pl2_traces_in = self.pl2_roi_in
                pl2_out, pl2_crosstalk, pl2_mixing, _, _ = self.unmix_plane(pl2_traces_in, pl2_roi_valid)

                # saving to self
                self.pl1_roi_out = pl1_out
                self.pl2_roi_out = pl2_out
                self.pl1_roi_crosstalk = pl1_crosstalk
                self.pl2_roi_crosstalk = pl2_crosstalk
                self.pl1_roi_out_path = pl1_out_path
                self.pl2_roi_out_path = pl2_out_path
                self.pl1_roi_out_path = pl1_out_path
                self.pl2_roi_out_path = pl2_out_path

                # writing ica out traces to disk
                with h5py.File(self.pl1_roi_out_path, "w") as f:
                    f.create_dataset(f"data", data=pl1_out)
                    f.create_dataset(f"crosstalk", data=pl1_crosstalk)
                    f.create_dataset(f"mixing_matrix", data=pl1_mixing)

                with h5py.File(self.pl2_roi_out_path, "w") as f:
                    f.create_dataset(f"data", data=pl2_out)
                    f.create_dataset(f"crosstalk", data=pl2_crosstalk)
                    f.create_dataset(f"mixing_matrix", data=pl2_mixing)


            # plotting goes here

        else:
            logger.info("Unmixed traces exist in cache, reading from h5 file")
            self.pl1_roi_out_path = pl1_out_path
            self.pl2_roi_out_path = pl2_out_path

            self.found_solution = True
            with h5py.File(self.pl1_roi_out_path, "r") as f:
                pl1_out = f["data"][()]
            with h5py.File(self.pl2_roi_out_path, "r") as f:
                pl2_out = f["data"][()]
            self.pl1_ica_out = pl1_out
            self.pl2_ica_out = pl2_out
        return

    def plot_ica_traces(self, pair, samples_per_plot=10000, dir_name=None, cell_num=None, fig_show=True, fig_save=True):
        """
        function to plot demixed traces
        :param dir_name: directory name prefix for where to save plotted traces
        :param pair: [int, int]: LIMS IDs for the two paired pls
        :param samples_per_plot: int, samples ot visualize on one plot, decreasing will make plotting very slow
        :param cell_num: int, number of rois to plot
        :param fig_show: bool, controlling whether to show a figure in jupyter/iphython as it's being generated or not
        :param fig_save: bool, controlling whether to save the figure in cache
        :return: None
        """

        if not dir_name:
            dir_name = self.roi_name

        if self.debug_mode:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        if not fig_show:
            print(f'Switching backend to Agg')
            plt.switch_backend('Agg')

        if self.pl1_roi_out_path and self.pl2_roi_out_path:

            raw_trace_pl1_sig = self.pl1_roi_raw[0, :, :]
            raw_trace_pl1_ct = self.pl1_roi_raw[1, :, :]
            pl1_roi_names = self.pl1_roi_names
            pl1_roi_valid = self.pl1_rois_valid['signal']
            ica_trace_pl1_sig = self.pl1_ica_out[0, :, :]
            ica_trace_pl1_ct = self.pl1_ica_out[1, :, :]

            logging.info(f'creating figures for experiment {pair[0]}')

            plot_dir = os.path.join(self.session_cache_dir, f'{dir_name}_{pair[0]}_{pair[1]}/ica_plots_{pair[0]}')
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)

            cell_valid = 0

            if not cell_num:
                cells_to_plot = range(raw_trace_pl1_sig.shape[0])
            else:
                cells_to_plot = range(cell_num)

            for cell_orig in cells_to_plot:
                # check in this roi is valid:
                if pl1_roi_valid[str(pl1_roi_names[cell_orig])]:
                    # Plot cell
                    pdf_name = os.path.join(plot_dir, f"ica_plots_{pair[0]}_cell_{pl1_roi_names[cell_orig]}.pdf")
                    if os.path.isfile(pdf_name):
                        logging.info(f"cell trace figure exist for {pair[0]} cell {pl1_roi_names[cell_orig]}")
                        continue
                    else:
                        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
                        logging.info(f"creating figures for cell {pl1_roi_names[cell_orig]}")
                        raw_y_min = min(min(raw_trace_pl1_sig[cell_orig, :]), min(raw_trace_pl1_ct[cell_orig, :]))
                        raw_y_max = max(max(raw_trace_pl1_sig[cell_orig, :]), max(raw_trace_pl1_ct[cell_orig, :]))
                        for i in range(int(raw_trace_pl1_sig.shape[1] / samples_per_plot) + 1):
                            orig_pl1_sig = raw_trace_pl1_sig[cell_orig, i * samples_per_plot:(i + 1) * samples_per_plot]
                            orig_pl1_ct = raw_trace_pl1_ct[cell_orig, i * samples_per_plot:(i + 1) * samples_per_plot]
                            ica_pl1_sig = ica_trace_pl1_sig[cell_valid, i * samples_per_plot:(i + 1) * samples_per_plot]
                            ica_pl1_ct = ica_trace_pl1_ct[cell_valid, i * samples_per_plot:(i + 1) * samples_per_plot]
                            f = plt.figure(figsize=(20, 10))
                            plt.subplot(211)
                            plt.ylim(raw_y_min, raw_y_max)
                            plt.plot(orig_pl1_sig, 'r-', label='signal pl')
                            plt.plot(orig_pl1_ct, 'g-', label='cross-talk pl')
                            plt.title(f'original traces for cell {pl1_roi_names[cell_orig]}')
                            plt.legend(loc='upper left')
                            plt.subplot(212)
                            plt.ylim(raw_y_min, raw_y_max)
                            plt.plot(ica_pl1_sig, 'r-', label='signal pl')
                            plt.plot(ica_pl1_ct, 'g-', label='cross-talk pl')
                            plt.title(f'post-ica traces, cell # {pl1_roi_names[cell_orig]}')
                            plt.legend(loc='upper left')
                            if fig_save:
                                pdf.savefig(f)
                            if not fig_show:
                                plt.close()
                        pdf.close()
                        cell_valid = cell_valid + 1
                else:
                    logging.info(f'Cell {pl1_roi_names[cell_orig]} is invalid, skipping plotting')
                    cell_valid = cell_valid

            raw_trace_pl2_sig = self.pl2_roi_raw[0, :, :]
            raw_trace_pl2_ct = self.pl2_roi_raw[1, :, :]
            pl2_roi_names = self.pl2_roi_names
            pl2_roi_valid = self.pl2_rois_valid['signal']
            ica_trace_pl2_sig = self.pl2_ica_out[0, :, :]
            ica_trace_pl2_ct = self.pl2_ica_out[1, :, :]
            logging.info(f'creating figures for experiment {pair[1]}')
            plot_dir = os.path.join(self.session_cache_dir, f'{dir_name}_{pair[0]}_{pair[1]}/ica_plots_{pair[1]}')
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)
            cell_valid = 0

            if not cell_num:
                cells_to_plot = range(raw_trace_pl2_sig.shape[0])
            else:
                cells_to_plot = range(cell_num)

            for cell_orig in cells_to_plot:
                # check in this roi is valid:
                if pl2_roi_valid[str(pl2_roi_names[cell_orig])]:
                    # Plot cell
                    pdf_name = os.path.join(plot_dir, f"ica_plots_{pair[1]}_cell_{pl2_roi_names[cell_orig]}.pdf")
                    if os.path.isfile(pdf_name):
                        logging.info(f"cell trace figure exist for {pair[1]} cell {pl2_roi_names[cell_orig]}")
                        continue
                    else:
                        logging.info(f'creating figures for cell {pl2_roi_names[cell_orig]}')
                        raw_y_min = min(min(raw_trace_pl2_sig[cell_orig, :]), min(raw_trace_pl2_ct[cell_orig, :]))
                        raw_y_max = max(max(raw_trace_pl2_sig[cell_orig, :]), max(raw_trace_pl2_ct[cell_orig, :]))
                        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
                        for i in range(int(raw_trace_pl2_sig.shape[1] / samples_per_plot) + 1):
                            orig_pl2_sig = raw_trace_pl2_sig[cell_orig, i * samples_per_plot:(i + 1) * samples_per_plot]
                            orig_pl2_ct = raw_trace_pl2_ct[cell_orig, i * samples_per_plot:(i + 1) * samples_per_plot]
                            ica_pl2_sig = ica_trace_pl2_sig[cell_valid, i * samples_per_plot:(i + 1) * samples_per_plot]
                            ica_pl2_ct = ica_trace_pl2_ct[cell_valid, i * samples_per_plot:(i + 1) * samples_per_plot]
                            f = plt.figure(figsize=(20, 10))
                            plt.subplot(211)
                            plt.ylim(raw_y_min, raw_y_max)
                            plt.plot(orig_pl2_sig, 'r-', label='signal pl')
                            plt.plot(orig_pl2_ct, 'g-', label='cross-talk pl')
                            plt.title(f'original traces for cell # {pl2_roi_names[cell_orig]}')
                            plt.legend(loc='upper left')
                            plt.subplot(212)
                            plt.ylim(raw_y_min, raw_y_max)
                            plt.plot(ica_pl2_sig, 'r-', label='signal pl')
                            plt.plot(ica_pl2_ct, 'g-', label='cross-talk pl')
                            plt.title(f'post-ica traces, cell # {pl2_roi_names[cell_orig]}')
                            plt.legend(loc='upper left')
                            if fig_save:
                                pdf.savefig(f)
                            if not fig_show:
                                plt.close()
                        pdf.close()
                    cell_valid = cell_valid + 1

                else:
                    logging.info(f'Cell {pl2_roi_names[cell_orig]} is invalid, skipping plotting')
                    cell_valid = cell_valid
        else:
            logging.info(f'ICA traces for pair {pair[0]}/{pair[1]} don''t exist, nothing to plot.')

        return

    def plot_raw_traces(self, pair, samples_per_plot=10000, dir_name=None, cell_num=None, figshow=True, figsave=True):
        """
        fn to plot raw traces
        :param dir_name: directory name prefix for where to save plotted traces
        :param pair: [int, int]: LIMS IDs for the two paired pls
        :param samples_per_plot: int, samples ot visualize on one plot, decreasing will make plotting very slow
        :param cell_num: int, number of rois to plot
        :param figshow: bool, controlling whether to show a figure in jupyter/iphython as it's being generated or not
        :param figsave: bool, controlling whether to save the figure in cache
        :return: None
        """
        if not dir_name:
            dir_name = self.roi_name

        if self.debug_mode:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        if not figshow:
            print(f'Switching backend to Agg')
            plt.switch_backend('Agg')

        if self.pl1_roi_raw_path and self.pl2_roi_raw_path:

            raw_trace_pl1_sig = self.pl1_roi_raw[0, :, :]
            raw_trace_pl1_ct = self.pl1_roi_raw[1, :, :]
            pl1_roi_names = self.pl1_roi_names
            pl1_roi_valid = self.pl1_rois_valid['signal']

            logging.info(f'plotting raw traces for experiment {pair[0]}')

            plot_dir = os.path.join(self.session_cache_dir, f'{dir_name}_{pair[0]}_{pair[1]}/raw_traces_plots_{pair[0]}')
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)

            if not cell_num:
                cells_to_plot = range(raw_trace_pl1_sig.shape[0])
            else:
                cells_to_plot = range(cell_num)

            for cell_orig in cells_to_plot:
                # check in this roi is valid:
                if pl1_roi_valid[str(pl1_roi_names[cell_orig])]:
                    # Plot cell
                    pdf_name = os.path.join(plot_dir, f"raw_traces_plots_{pair[0]}_cell_{pl1_roi_names[cell_orig]}.pdf")
                    if os.path.isfile(pdf_name):
                        logging.info(f"raw trace plots exist for {pair[0]} cell {pl1_roi_names[cell_orig]}")
                        continue
                    else:
                        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
                        logging.info(f"plotting original traces for cell {pl1_roi_names[cell_orig]}")
                        for i in range(int(raw_trace_pl1_sig.shape[1] / samples_per_plot) + 1):
                            orig_pl1_sig = raw_trace_pl1_sig[cell_orig, i * samples_per_plot:(i + 1) * samples_per_plot]
                            orig_pl1_ct = raw_trace_pl1_ct[cell_orig, i * samples_per_plot:(i + 1) * samples_per_plot]
                            f = plt.figure(figsize=(20, 10))
                            plt.plot(orig_pl1_sig, 'r-', label='signal pl')
                            plt.plot(orig_pl1_ct, 'g-', label='cross-talk pl')
                            plt.title(f'original traces for cell {pl1_roi_names[cell_orig]}')
                            plt.legend(loc='upper left')
                            if figsave:
                                pdf.savefig(f)
                            if not figshow:
                                plt.close()
                        pdf.close()
                else:
                    logging.info(f'Cell {pl1_roi_names[cell_orig]} is invalid, skipping plotting')

            raw_trace_pl2_sig = self.pl2_roi_raw[0, :, :]
            raw_trace_pl2_ct = self.pl2_roi_raw[1, :, :]
            pl2_roi_names = self.pl2_roi_names
            pl2_roi_valid = self.pl2_rois_valid['signal']
            logging.info(f'creating plots for experiment {pair[1]}')
            plot_dir = os.path.join(self.session_cache_dir, f'{dir_name}_{pair[0]}_{pair[1]}/raw_traces_plots_{pair[1]}')
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)

            if not cell_num:
                cells_to_plot = range(raw_trace_pl2_sig.shape[0])
            else:
                cells_to_plot = range(cell_num)

            for cell_orig in cells_to_plot:
                # check in this roi is valid:
                if pl2_roi_valid[str(pl2_roi_names[cell_orig])]:
                    # Plot cell
                    pdf_name = os.path.join(plot_dir, f"raw_traces_plots_{pair[1]}_cell_{pl2_roi_names[cell_orig]}.pdf")
                    if os.path.isfile(pdf_name):
                        logging.info(f"raw trace plots exist for {pair[1]} cell {pl2_roi_names[cell_orig]}")
                        continue
                    else:
                        logging.info(f'creating plots for cell {pl2_roi_names[cell_orig]}')
                        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
                        for i in range(int(raw_trace_pl2_sig.shape[1] / samples_per_plot) + 1):
                            orig_pl2_sig = raw_trace_pl2_sig[cell_orig, i * samples_per_plot:(i + 1) * samples_per_plot]
                            orig_pl2_ct = raw_trace_pl2_ct[cell_orig, i * samples_per_plot:(i + 1) * samples_per_plot]
                            f = plt.figure(figsize=(20, 10))
                            plt.plot(orig_pl2_sig, 'r-', label='signal pl')
                            plt.plot(orig_pl2_ct, 'g-', label='cross-talk pl')
                            plt.title(f'original traces for cell # {pl2_roi_names[cell_orig]}')
                            plt.legend(loc='upper left')
                            if figsave:
                                pdf.savefig(f)
                            if not figshow:
                                plt.close()
                        pdf.close()

                else:
                    logging.info(f'Cell {pl2_roi_names[cell_orig]} is invalid, skipping plotting')
        else:
            logging.info(f'raw traces for pair {pair[0]}/{pair[1]} don''t exist, nothing to plot.')

        return

    @staticmethod
    def validate_against_vba(rois_valid_ica, exp_id, vba_cache=VBA_CACHE):
        """
        :param rois_valid_ica: dict, returned by MesoscopeICA.pl1_rois_valid or MesoscopeICA.pl2_rois_valid
        :param exp_id: int, LIMS experiment ID, can be retrieved by MesoscopeICA.pl1_exp_id or MesoscopeICA.pl2_exp_id
        :param vba_cache: str, path to visual behavior analysis package cache directory
        :return: rois_valid_vba: dict, same structure as rois_valid_ica
        """
        dataset = VisualBehaviorOphysDataset(exp_id, cache_dir=vba_cache)
        roi_names_vba = dataset.cell_specimen_ids
        # invalid rois in ica validation json
        rois_valid_vba = rois_valid_ica
        for roi_name in rois_valid_ica["signal"]:
            if int(roi_name) not in roi_names_vba:
                rois_valid_vba['signal'][str(roi_name)] = False
                rois_valid_vba['crosstalk'][str(roi_name)] = False
        return rois_valid_vba

    @staticmethod
    def ica_err(scale, ica_traces, trace_orig):
        """
        calculate difference of standard deviation between post- and pre-ICA traces:
        :param scale: scaling factor - used to optimize
        :param ica_traces: post-ica trace
        :param trace_orig: raw trace
        :return:
        """
        return np.sqrt((ica_traces * scale[0] - trace_orig) ** 2).mean()

    @staticmethod
    def get_valid_seg_run(exp_id):
        """
        queries  LIMS to retrieve an ID of the valid segmentation run for given expeirment
        :param exp_id: LIMS experiment ID
        :return: int
        """
        query = f"""
        select *
        from ophys_cell_segmentation_runs
        where current = True and ophys_experiment_id = {exp_id}
        """
        seg_run = lu.query(query)[0]['id']
        return seg_run

    def find_scale_ica_roi(self):
        """
        find scaling factor that will minimize difference of standard deviations between ICA input and ICA out
        for ROI traces
        :return: [int, int]
        """
        # for traces:
        scale_top = opt.minimize(self.ica_err, [1], (self.roi_unmix[:, 0], self.pl1_roi_in))
        scale_bot = opt.minimize(self.ica_err, [1], (self.roi_unmix[:, 1], self.pl2_roi_in))

        return scale_top.x, scale_bot.x

    def find_scale_ica_neuropil(self):
        """
        returns scaling factor that will minimize difference of standard deviations between ICA input and ICA out
        for neuropil traces
        :return: [int, int]
        """
        scale_top_neuropil = opt.minimize(self.ica_err, [1],
                                          (self.neuropil_unmix[:, 0], self.pl1_np_in))
        scale_bot_neuropil = opt.minimize(self.ica_err, [1],
                                          (self.neuropil_unmix[:, 1], self.pl2_np_in))

        return scale_top_neuropil.x, scale_bot_neuropil.x

    @staticmethod
    def get_crosstalk_before_and_after(valid, traces_in, traces_out, path_out, fig_save=False, fig_overwrite=False):
        """
        estimate crosstalk before and after ica demixing
        :param fig_overwrite: flag to control whether cross talk plots should be re-plotted
        :param valid: valid roi json
        :param traces_in: numpy array containing ICA input traces in [NxMxT] where N = 2, M = number of cells, T = number of timestamps
        :param traces_out: numpy array containing ICA out traces in [NxMxT] where N = 2, M = number of cells, T = number of timestamps
        :param path_out: str, name of the json file to save the data
        :param fig_save: bool, flag to save the figures or not in the same folder as path_out
        :return:
        """

        i = 0
        roi_names = list(valid['signal'].keys())
        num_traces = len(roi_names)
        valid_mask = np.array([valid['signal'][str(tid)] for tid in roi_names])
        traces_in_valid = traces_in[:, valid_mask, :]
        crosstalk_in = dict.fromkeys(roi_names)
        crosstalk_out = dict.fromkeys(roi_names)
        r_values_in = dict.fromkeys(roi_names)
        r_values_out = dict.fromkeys(roi_names)
        if fig_save:
            ct_plot_dir = os.path.join(os.path.split(path_out)[0],
                                       f'{os.path.splitext(os.path.split(path_out)[1])[0]}_plots')
            if not os.path.isdir(ct_plot_dir):
                os.mkdir(ct_plot_dir)
            else:
                if not fig_overwrite:
                    logging.info(f"Crosstalk plots exist at {ct_plot_dir}, set fig_overwrite to True to overwrite")
                    fig_save = False
                else:
                    logging.info(f"Crosstalk plots exist at {ct_plot_dir}, overwriting")

        # get active traces:
        len_ne = 20
        th_ag = 10
        do_plots = 0

        # extract events for input, signal
        traces_evs, evs_ind = at.get_traces_evs(traces_in_valid[0], th_ag, len_ne, do_plots)

        for n in range(num_traces):
            roi_name = roi_names[n]
            if fig_save:
                pdf_name = os.path.join(ct_plot_dir, f"{roi_name}_crosstalk.pdf")
                pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
            if valid['signal'][roi_name]:
                if not np.any(np.isnan(evs_ind[i])):
                    sig_trace_in = traces_evs[i]
                    ct_trace_in = traces_in_valid[1][i][evs_ind[i]]
                    sig_trace_out = traces_out[0][i][evs_ind[i]]
                    ct_trace_out = traces_out[1][i][evs_ind[i]]
                    # estimate crosstalk and plot pixel histograms
                    fig_in, slope_in, _, r_value_in = plot_pixel_hist2d(sig_trace_in, ct_trace_in,
                                                                        title=f'raw, cell {roi_name}', save_fig=False,
                                                                        save_path=None, fig_show=False, colorbar=True)
                    fig_out, slope_out, _, r_value_out = plot_pixel_hist2d(sig_trace_out, ct_trace_out,
                                                                           title=f'clean, cell {roi_name}', save_fig=False,
                                                                           save_path=None, fig_show=False, colorbar=True)
                    if fig_save:
                        pdf.savefig(fig_in)
                        pdf.savefig(fig_out)
                    else:
                        del fig_in
                        del fig_out
                    crosstalk_in[roi_name] = slope_in
                    crosstalk_out[roi_name] = slope_out
                    r_values_in[roi_name] = r_value_in
                    r_values_out[roi_name] = r_value_out
                else:
                    logging.info(f"Neuron {roi_name} has no events, skipping calculating crosstalk")
                    crosstalk_in[roi_name] = np.nan
                    crosstalk_out[roi_name] = np.nan
                    r_values_in[roi_name] = np.nan
                    r_values_out[roi_name] = np.nan
                i += 1
            else:
                crosstalk_in[roi_name] = np.nan
                crosstalk_out[roi_name] = np.nan
                r_values_in[roi_name] = np.nan
                r_values_out[roi_name] = np.nan
            if fig_save:
                pdf.close()

        roi_crosstalk = {"crosstalk_raw": crosstalk_in, "crosstalk_demixed": crosstalk_out, "r_values_raw": r_values_in,
                         "r_values_out": r_values_out}
        ju.write(path_out, roi_crosstalk)
        return roi_crosstalk


def plot_pixel_hist2d(x, y, xlabel='signal', ylabel='crosstalk', title=None, save_fig=False, save_path=None,
                      fig_show=True, colorbar=False):
    fig = plt.figure(figsize=(3, 3))
    h, xedges, yedges = np.histogram2d(x, y, bins=(30, 30))
    h = h.T
    plt.rcParams.update({'font.size': 14})
    plt.imshow(h, interpolation='nearest', origin='low',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', norm=LogNorm())

    if colorbar:
        cbar = plt.colorbar()
        cbar.set_label('# of counts', rotation=270, fontsize=14, labelpad=20)
        cbar.ax.tick_params(labelsize=14)

    slope, offset, r_value, p_value, std_err = linregress(x, y)
    fit_fn = np.poly1d([slope, offset])

    plt.plot(x, fit_fn(x), '--k')

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if not title:
        title = '%s    R2=%.2f' % (fit_fn, r_value ** 2)

    plt.title(title, fontsize=12)

    if save_fig:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)

    if not fig_show:
        plt.close()

    return fig, slope, offset, r_value


def ica_err(scale, ica_traces, trace_orig):
    """
    calculate difference of standard deviation between post- and pre-ICA traces:
    :param scale: scaling factor - used to optimize
    :param ica_traces: post-ica trace
    :param trace_orig: raw trace
    :return:
    """
    return np.sqrt((ica_traces * scale[0] - trace_orig) ** 2).mean()


def rescale(trace_in, trace_out):
    """
    fn to rescale post-ica traces back to it's originla variances.
    :param trace_in: numpy.array, trace before crosstalk correction
    :param trace_out: numpy.array, trace after crosstalk correction
    :return: rescaled trace: numpy.array, rescaled trace post crosstalk correction
    """
    scale = find_scale_ica_roi(trace_in, trace_out)
    if scale < 0:
        scale *= -1
    rescaled_trace = trace_out * scale
    return rescaled_trace


def find_scale_ica_roi(ica_in, ica_out):
    """
    find scaling factor that will minimize difference of standard deviations between ICA input and ICA out
    for ROI traces
    :return: [int, int]
    """
    # for traces:
    scale = opt.minimize(ica_err, [1], (ica_out, ica_in))
    return scale.x

def run_ica(trace1, trace2):
    traces = np.array([trace1, trace2]).T
    f_ica = FastICA(n_components=2, max_iter=50)
    _ = f_ica.fit_transform(traces)  # Reconstruct signals
    mix = f_ica.mixing_  # Get estimated mixing matrix
    # make sure no negative coeffs (inversion of traces)
    mix[mix < 0] *= -1
    # switch columns if needed (source assignment inverted) - check something is off here!
    if mix[0, 0] < mix[1, 0]:
        a_mix = np.array([mix[1, :], mix[0, :]])
    else:
        a_mix = mix
    if a_mix[0, 1] > a_mix[1, 1]:
        b_mix = np.array([[a_mix[0, 0], a_mix[1, 1]], [a_mix[1, 0], a_mix[0, 1]]])
    else:
        b_mix = a_mix
    a_unmix = linalg.pinv(b_mix)
    # recontructing signals: dot product of unmixing matrix and input traces
    r_source = np.dot(a_unmix, traces.T).T
    return mix, a_mix, a_unmix, r_source


def extract_active(traces, len_ne=20, th_ag=10, do_plots=0):
    traces_sig = traces[0, :, :]
    traces_ct = traces[1, :, :]
    # extract events for input, signal
    traces_sig_evs, evs_ind = at.get_traces_evs(traces_sig, th_ag, len_ne, do_plots)
    # apply active indeces to ct trace as well:
    traces_ct_evs = []
    for i in range(evs_ind.shape[0]):
        trace_ct = traces_ct[i, evs_ind[i]]
        traces_ct_evs.append(trace_ct)
    return traces_sig_evs, traces_ct_evs


# for pkey in self.pkeys:
#     for tkey in self.tkeys:
#         out_paths[pkey][tkey] = os.path.join(self.dirs[tkey],
#                                              f'{self.exp_ids[pkey]}_out.h5')
