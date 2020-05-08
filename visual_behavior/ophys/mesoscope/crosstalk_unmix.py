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
import matplotlib.backends.backend_pdf as plt_pdf
import matplotlib.pyplot as plt
import allensdk.core.json_utilities as ju
from scipy import linalg
from scipy.stats import linregress
from matplotlib.colors import LogNorm
import visual_behavior.ophys.mesoscope.active_traces as at
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
import copy

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
    Class to perform ica-based demixing on a pair of Multiscope planes
    """

    def __init__(self, session_id, cache, debug_mode=False, roi_name="ica_traces", np_name="ica_neuropil"):
        """
        :param session_id: LIMS session ID
        :param cache: directory to store/find ins/outs of ICA and all complimentary files
        :param debug_mode: flag that controls whether debug logger messages are out
        :param roi_name: string, default name for roi-related in/outs
        :param np_name: string, default name for neuropil-related ins/outs
        """
        self.tkeys = ['roi', 'np']
        self.pkeys = ['pl1', 'pl2']
        self.session_id = session_id
        self.dataset = ms.MesoscopeDataset(session_id)
        self.session_dir = None
        self.debug_mode = debug_mode
        self.cache = cache  # analysis directory
        self.session_dir = os.path.join(self.cache, f'session_{self.session_id}')

        # prefix for files related to roi and neuropil traces
        self.names = {'roi': roi_name, 'np': np_name}

        self.exp_ids = {key: None for key in self.pkeys}
        self.dirs = {key: None for key in self.tkeys}
        self.set_ica_dirs()

        # pointers and attributes related to raw traces
        self.raws = {}
        self.raw_paths = {}
        self.rois_names = {}
        self.rois_names_valid = {}
        # pointers and attributes related to ica input traces
        self.ins = {}
        self.ins_paths = {}
        self.offsets = {}
        # pointers and attributes for validation jsons
        self.rois_valid = {}
        self.rois_valid_paths = {}
        self.rois_valid_ct = {}
        self.rois_names_valid_ct = {}
        self.rois_valid_ct_paths = {}
        # pointers and attirbutes for ica output files
        self.outs = {}  # output unmixing traces
        self.outs_paths = {}  # paths to save unmixing output
        self.crosstalk = {}  # slope of linear fit to the 2D plot of signal Vs Crosstalk
        self.mixing = {}  # raw mixing matrix on the output of FastICA
        self.a_mixing = {}  # adjusted mixing matrix on the output of FastICA
        self.plot_dirs = {}
        # dff files filtering attributes
        self.dff_files = {}
        self.dff = {}
        self.dff_ct_files = {}
        self.dff_ct = {}
        # demixing filtering attributes:
        self.dem_files = {}
        self.dem = {}
        self.dem_ct_files = {}
        self.dem_ct = {}
        # neuriopil corected traces filtering attributes:
        self.np_cor_files = {}
        self.np_cor = {}
        self.np_cor_ct_files = {}
        self.np_cor_ct = {}

        for pkey in self.pkeys:
            self.rois_names[pkey] = {}
            self.rois_names_valid[pkey] = {}
            self.rois_names_valid_ct[pkey] = {}
            self.rois_valid[pkey] = {}
            self.rois_valid_paths[pkey] = {}
            self.rois_valid_ct[pkey] = {}
            self.rois_valid_ct_paths[pkey] = {}
            self.raw_paths[pkey] = {}
            self.raws[pkey] = {}
            self.ins[pkey] = {}
            self.ins_paths[pkey] = {}
            self.offsets[pkey] = {}
            self.outs[pkey] = {}
            self.outs_paths[pkey] = {}
            self.crosstalk[pkey] = {}
            self.mixing[pkey] = {}
            self.a_mixing[pkey] = {}
            self.plot_dirs[pkey] = {}
            for tkey in self.tkeys:
                self.rois_names[pkey][tkey] = None
                self.rois_names_valid[pkey][tkey] = None
                self.rois_names_valid_ct[pkey][tkey] = None
                self.raws[pkey][tkey] = None
                self.raw_paths[pkey][tkey] = None
                self.rois_valid_paths[pkey][tkey] = None
                self.rois_valid_ct_paths[pkey][tkey] = None
                self.ins[pkey][tkey] = None
                self.ins_paths[pkey][tkey] = None
                self.offsets[pkey][tkey] = None
                self.outs[pkey][tkey] = None
                self.outs_paths[pkey][tkey] = None
                self.crosstalk[pkey][tkey] = None
                self.mixing[pkey][tkey] = None
                self.a_mixing[pkey][tkey] = None
                self.plot_dirs[pkey][tkey] = None

        self.found_raws = {}  # flag if raw traces exist in self.dirs; output of get_traces
        self.found_ins = {}  # flag if ica input traces exists
        self.found_offsets = {}  # flag if offset data exists
        self.found_solution = {}
        for pkey in self.pkeys:
            self.found_raws[pkey] = {}
            self.found_ins[pkey] = {}
            self.found_offsets[pkey] = {}
            self.found_solution[pkey] = {}
            for tkey in self.tkeys:
                self.found_raws[pkey][tkey] = None
                self.found_ins[pkey][tkey] = [None, None]
                self.found_offsets[pkey][tkey] = [None, None]
                self.found_solution[pkey][tkey] = False

    def set_exp_ids(self, pair):
        """
        fn to set self.exp_ids
        :return:
        """
        self.exp_ids["pl1"] = pair[0]
        self.exp_ids["pl2"] = pair[1]
        return

    def set_ica_dirs(self, names=None):
        """
        create path to ica-related inputs/outs for the pair
        :param names: roi_nam if different form self.names["roi"] to use to locate old inputs/outs
        :return: None
        """
        if not names:
            names = self.names

        session_dir = self.session_dir

        for tkey in self.tkeys:
            self.dirs[tkey] = os.path.join(session_dir, f'{names[tkey]}_{self.exp_ids["pl1"]}_{self.exp_ids["pl2"]}/')

        return

    def set_raws_paths(self):
        for pkey in self.pkeys:
            for tkey in self.tkeys:
                self.raw_paths[pkey][tkey] = os.path.join(self.dirs[tkey],
                                                          f'{self.exp_ids[pkey]}_raw.h5')
        return

    def set_out_paths(self):
        for pkey in self.pkeys:
            for tkey in self.tkeys:
                self.outs_paths[pkey][tkey] = os.path.join(self.dirs[tkey],
                                                           f'{self.exp_ids[pkey]}_out.h5')
        return

    def set_valid_paths(self):
        for pkey in self.pkeys:
            for tkey in self.tkeys:
                self.rois_valid_paths[pkey][tkey] = os.path.join(self.dirs[tkey],
                                                                 f'{self.exp_ids[pkey]}_valid.json')
                self.rois_valid_ct_paths[pkey][tkey] = os.path.join(self.dirs[tkey],
                                                                    f'{self.exp_ids[pkey]}_valid_ct.json')
        return

    def set_ica_input_paths(self):
        for pkey in self.pkeys:
            for tkey in self.tkeys:
                self.ins_paths[pkey][tkey] = os.path.join(self.dirs[tkey],
                                                          f'{self.exp_ids[pkey]}_in.h5')
        return

    def set_plot_dirs(self, dir_name=None):
        if dir_name is None:
            dir_name = self.names

        for pkey in self.pkeys:
            for tkey in self.tkeys:
                self.plot_dirs[pkey][tkey] = os.path.join(self.dirs[tkey], f"{dir_name[tkey]}_ica_plots_{self.exp_ids[pkey]}")
        return

    def get_ica_traces(self):
        """
        function to apply roi set to two image pls, first check if the traces have been extracted before,
        can use a different roi_name, if traces don't exist in cache, read roi set name form LIMS< apply to both signal and crosstalk pls
        :return: list[bool bool]: flags to see if traces where extracted successfully
        """

        names_prefix = [{"roi": 'traces_original', "np": 'neuropil_original'},
                        {"roi": 'raw', "np": 'raw'}]

        if self.debug_mode:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        self.set_ica_dirs()

        # we will first check if traces exist, if yes - read them, if not - extract them

        for pkey in self.pkeys:
            for tkey in self.tkeys:
                self.found_raws[pkey][tkey] = True

        traces_exist = None
        for i in range(len(names_prefix)):
            traces_exist = True
            # check ith set of names:
            name_prefix = names_prefix[i]
            name = {}
            for tkey in self.tkeys:
                name[tkey] = name_prefix[tkey]
            # define paths to traces
            path = {}
            for pkey in self.pkeys:
                path[pkey] = {}
                for tkey in self.tkeys:
                    path[pkey][tkey] = os.path.join(self.dirs[tkey], f"{self.exp_ids[pkey]}_{name[tkey]}.h5")
            # check if traces exist already:
            for pkey in self.pkeys:
                for tkey in self.tkeys:
                    if not os.path.isfile(path[pkey][tkey]):
                        traces_exist = False
                        self.found_raws[pkey][tkey] = False
            if traces_exist:
                break

        if traces_exist:
            # if both traces exist, skip extracting:
            logger.info('Found traces in cache, reading from h5 file')

            # read traces and roi namesfrom file:
            for pkey in self.pkeys:
                for tkey in self.tkeys:
                    with h5py.File(path[pkey]["roi"], "r") as f:
                        self.rois_names[pkey][tkey] = f["roi_names"][()]
                    self.found_raws[pkey][tkey] = True  # set found traces flag True
                    self.raw_paths[pkey][tkey] = path[pkey][tkey]
                    with h5py.File(path[pkey][tkey], "r") as f:
                        self.raws[pkey][tkey] = f["data"][()]

        else:
            # some traces are missing, run extraction:
            logger.info('Traces dont exist in cache, extracting')

            # if traces don't exist, do we need to reset unmixed and debiased traces flaggs to none?
            # yes, as we want unmixed traces be out on ICA using original traces
            folders = {}
            for pkey in self.pkeys:
                folders[pkey] = self.dataset.get_exp_folder(self.exp_ids[pkey])
                for tkey in self.tkeys:
                    self.rois_names[pkey][tkey] = None
                    self.ins_paths[pkey][tkey] = None
                    self.outs_paths[pkey][tkey] = None

            sig = {}
            ct = {}
            roi_names = {}
            for pkey in self.pkeys:
                sig[pkey] = {}
                ct[pkey] = {}
                roi_names[pkey] = {}
                for tkey in self.tkeys:
                    sig[pkey][tkey] = {}
                    ct[pkey][tkey] = {}
                    roi_names[pkey][tkey] = {}

            # extract signal and crosstalk traces for pl 1
            sig["pl1"]["roi"], sig["pl1"]["np"], roi_names["pl1"]['roi'] = get_traces(folders["pl1"], self.exp_ids["pl1"],
                                                                                      folders["pl1"], self.exp_ids["pl1"])
            roi_names["pl1"]['np'] = roi_names["pl1"]['roi']
            ct["pl1"]["roi"], ct["pl1"]["np"], _ = get_traces(folders["pl2"], self.exp_ids["pl2"], folders["pl1"],
                                                              self.exp_ids["pl1"])
            # extract signal and crosstalk traces for pl 2
            sig["pl2"]["roi"], sig["pl2"]["np"], roi_names["pl2"]['roi'] = get_traces(folders["pl2"], self.exp_ids["pl2"],
                                                                                      folders["pl2"], self.exp_ids["pl2"])
            roi_names["pl2"]['np'] = roi_names["pl2"]['roi']
            ct["pl2"]["roi"], ct["pl2"]["np"], _ = get_traces(folders["pl1"], self.exp_ids["pl1"], folders["pl2"],
                                                              self.exp_ids["pl2"])

            for pkey in self.pkeys:
                for tkey in self.tkeys:
                    if (not sig[pkey][tkey].any() is None) and (not ct[pkey][tkey].any() is None):
                        self.found_raws[pkey][tkey] = True

            if not os.path.isdir(self.session_dir):
                os.mkdir(self.session_dir)
            if not os.path.isdir(self.dirs["roi"]):
                os.mkdir(self.dirs["roi"])
            if not os.path.isdir(self.dirs["np"]):
                os.mkdir(self.dirs["np"])

            # if extracted traces valid, save to disk:
            for pkey in self.pkeys:
                for tkey in self.tkeys:
                    if self.found_raws[pkey][tkey]:
                        self.rois_names[pkey][tkey] = roi_names[pkey][tkey]
                    if self.found_raws[pkey][tkey] and self.found_raws[pkey][tkey]:
                        # combining traces, saving to self, writing to disk:
                        traces_raw = np.array([sig[pkey][tkey], ct[pkey][tkey]])
                        self.raws[pkey][tkey] = traces_raw
                        self.raw_paths[pkey][tkey] = path[pkey][tkey]
                        with h5py.File(path[pkey][tkey], "w") as f:
                            f.create_dataset("data", data=traces_raw, compression="gzip")
                            f.create_dataset("roi_names", data=np.int_(roi_names[pkey][tkey]))
        return

    def validate_traces(self, return_vba=False):
        """
        fn to check if the traces don't have Nans, writes {exp_id}_valid.json to cache for each pl in pair
        return_vba: bool, flag to control whether to validate against vba roi set or return ica roi set
        :return: None
        """

        rois_valid_paths = {}
        rois_valid = {}
        for pkey in self.pkeys:
            rois_valid_paths[pkey] = {}
            rois_valid[pkey] = {}
            for tkey in self.tkeys:
                rois_valid[pkey][tkey] = {}
                rois_valid_paths[pkey][tkey] = os.path.join(self.dirs[tkey],
                                                            f'{self.exp_ids[pkey]}_valid.json')  # first define locally
        # validation json already exists, skip validating
        paths_valid = True  # first set to exist
        for pkey in self.pkeys:
            for tkey in self.tkeys:
                if not os.path.isfile(rois_valid_paths[pkey][tkey]):
                    paths_valid = False  # change to doesn't exist if it's not there of any of four files
        if paths_valid:  # if all exsist, set corresponding attributes : paths and read jsons
            for pkey in self.pkeys:
                for tkey in self.tkeys:
                    logger.info("Validation jsons exist, skipping validation")
                    self.rois_valid_paths[pkey][tkey] = rois_valid_paths[pkey][tkey]
                    rois_valid[pkey][tkey] = ju.read(rois_valid_paths[pkey][tkey])
                    self.rois_valid[pkey] = rois_valid[pkey][tkey]
                    self.rois_names_valid[pkey][tkey] = [int(roi_name) for roi_name, valid in
                                                         self.rois_valid[pkey].items() if valid]
        else:  # if not - run validation, set attributes, save to disk.
            # check if  extracted traces exist?
            traces_exist = True
            for pkey in self.pkeys:
                for tkey in self.tkeys:
                    if not self.found_raws[pkey][tkey]:
                        traces_exist = False
            if not traces_exist:
                logger.info('Raw traces dont exist, run extract traces first')
            else:
                num_traces_sig = {}
                num_traces_ct = {}
                for pkey in self.pkeys:
                    num_traces_sig[pkey] = {}
                    num_traces_ct[pkey] = {}
                    for tkey in self.tkeys:
                        # check signal and crosstalk for given trace:
                        num_traces_sig[pkey][tkey] = self.raws[pkey][tkey][0].shape[0]
                        num_traces_ct[pkey][tkey] = self.raws[pkey][tkey][1].shape[0]

                # check if traces are aligned:
                traces_aligned = True
                for pkey in self.pkeys:
                    if not num_traces_sig[pkey]["roi"] == num_traces_sig[pkey]["np"]:
                        traces_aligned = False
                        logger.info('Neuropil and ROI traces are not aligned')

                if traces_aligned:
                    trace_sig = {}
                    trace_ct = {}
                    traces_valid = {}
                    # set flags to true:
                    for pkey in self.pkeys:
                        trace_sig[pkey] = {}
                        trace_ct[pkey] = {}
                        traces_valid[pkey] = {}
                        for tkey in self.tkeys:
                            traces_valid[pkey][tkey] = {}
                            for n in range(num_traces_sig[pkey]['roi']):
                                roi_name = str(self.rois_names[pkey]['roi'][n])
                                traces_valid[pkey][tkey][roi_name] = True
                                # check if both roi and np trace have no NaNs:
                                trace_sig[pkey][tkey] = self.raws[pkey][tkey][0][n]
                                trace_ct[pkey][tkey] = self.raws[pkey][tkey][1][n]
                                if np.any(np.isnan(trace_sig[pkey][tkey])) or np.any(np.isnan(trace_ct[pkey][tkey])):
                                    traces_valid[pkey][tkey][roi_name] = False
                                if not traces_valid[pkey][tkey][roi_name]:
                                    rois_valid[pkey][tkey][roi_name] = False
                                else:
                                    rois_valid[pkey][tkey][roi_name] = True

                    self.rois_valid = {}
                    for pkey in self.pkeys:
                        self.rois_valid[pkey] = {}
                        for roi_name in rois_valid[pkey]['roi'].keys():
                            tf = rois_valid[pkey]['roi'][roi_name] and rois_valid[pkey]['np'][roi_name]
                            self.rois_valid[pkey][roi_name] = tf

                    # validating agains VBA rois set:
                    if return_vba:
                        for pkey in self.pkeys:
                            rois_valid[pkey] = self.validate_against_vba(self.rois_valid[pkey],
                                                                         self.exp_ids[pkey], VBA_CACHE)
                # saving to json:
                for pkey in self.pkeys:
                    for tkey in self.tkeys:
                        self.rois_valid_paths[pkey][tkey] = rois_valid_paths[pkey][tkey]
                        ju.write(rois_valid_paths[pkey][tkey], self.rois_valid[pkey])
                        self.rois_names_valid[pkey][tkey] = [int(roi_name) for roi_name, valid in
                                                             self.rois_valid[pkey].items() if valid]
        return

    def debias_traces(self):
        """
        fn to combine all roi traces for the pair to two num_cells x num_frames_in_timeseries vectors,
        write them to cache as ica_roi_input
        :return: None
        """

        if self.debug_mode:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        # local vars for out paths:

        ins_paths = {}
        for pkey in self.pkeys:
            ins_paths[pkey] = {}
            for tkey in self.tkeys:
                self.ins_paths[pkey][tkey] = None
                ins_paths[pkey][tkey] = os.path.join(self.dirs[tkey], f'{self.exp_ids[pkey]}_in.h5')

        # check if debiased traces exist:
        in_traces_exist = True
        for pkey in self.pkeys:
            for tkey in self.tkeys:
                if not os.path.isfile(ins_paths[pkey][tkey]):
                    in_traces_exist = False

        for pkey in self.pkeys:
            for tkey in self.tkeys:
                if in_traces_exist:  # if input traces exist, set self.ins_paths
                    self.ins_paths[pkey][tkey] = ins_paths[pkey][tkey]
                else:
                    self.ins_paths[pkey][tkey] = None

        if not in_traces_exist:  # if debiased traces don't exist - run debiasing
            # check if raw traces exist
            raw_traces_exist = True
            for pkey in self.pkeys:
                for tkey in self.tkeys:
                    if not self.found_raws[pkey][tkey]:
                        raw_traces_exist = False

            if raw_traces_exist:  # Raw traces exist, run debiasing
                logger.info("Debiased ROI traces do not exist in cache, running offset subtraction")
                sig = {}
                ct = {}
                roi_valid = {}
                sig_valid = {}
                ct_valid = {}
                sig_offset = {}
                sig_m0 = {}
                ct_offset = {}
                ct_m0 = {}
                for pkey in self.pkeys:
                    sig[pkey] = {}
                    ct[pkey] = {}
                    roi_valid[pkey] = {}
                    sig_valid[pkey] = {}
                    ct_valid[pkey] = {}
                    sig_offset[pkey] = {}
                    sig_m0[pkey] = {}
                    ct_offset[pkey] = {}
                    ct_m0[pkey] = {}
                    for tkey in self.tkeys:
                        self.found_ins[pkey][tkey] = False
                        sig[pkey][tkey] = self.raws[pkey][tkey][0]
                        ct[pkey][tkey] = self.raws[pkey][tkey][1]
                        roi_valid[pkey][tkey] = self.rois_valid[pkey]
                        # check if traces aligned and separate signal/crosstalk
                        if len(self.rois_names[pkey][tkey]) == len(sig[pkey][tkey]):
                            valid_idx_mask = np.array(
                                [roi_valid[pkey][tkey][str(tid)] for tid in self.rois_names[pkey][tkey]])
                            sig_valid[pkey][tkey] = sig[pkey][tkey][valid_idx_mask, :]
                            ct_valid[pkey][tkey] = ct[pkey][tkey][valid_idx_mask, :]
                        else:
                            logging.info('Traces are not aligned')
                            raise ValueError("Traces are not aligned")

                        sig[pkey][tkey] = sig_valid[pkey][tkey]
                        ct[pkey][tkey] = ct_valid[pkey][tkey]

                        # subtract offset
                        nc = sig[pkey][tkey].shape[0]
                        sig_offset[pkey][tkey] = np.mean(sig[pkey][tkey], axis=1).reshape(nc, 1)
                        sig_m0[pkey][tkey] = sig[pkey][tkey] - sig_offset[pkey][tkey]
                        ct_offset[pkey][tkey] = np.mean(ct[pkey][tkey], axis=1).reshape(nc, 1)
                        ct_m0[pkey][tkey] = ct[pkey][tkey] - ct_offset[pkey][tkey]

                        # check if traces don't have None values
                        if (not (sig_m0[pkey][tkey].any() is None)) and (not (ct_m0[pkey][tkey].any() is None)):
                            self.found_ins[pkey][tkey] = [True, True]
                            self.found_offsets[pkey][tkey] = [True, True]

                        # if all flags true - combine sig and ct, write to disk:
                        if self.found_ins[pkey][tkey] and self.found_offsets[pkey][tkey]:
                            self.offsets[pkey][tkey] = {'sig_offset': sig_offset[pkey][tkey],
                                                        'ct_offset': ct_offset[pkey][tkey]}
                            self.ins[pkey][tkey] = np.array([sig_m0[pkey][tkey], ct_m0[pkey][tkey]])
                            self.ins_paths[pkey][tkey] = ins_paths[pkey][tkey]

                            # write ica input traces to disk
                            with h5py.File(self.ins_paths[pkey][tkey], "w") as f:
                                f.create_dataset("debiased_traces", data=self.ins[pkey][tkey], compression="gzip")
                                f.create_dataset("roi_names", data=self.rois_names_valid[pkey][tkey])
                                f.create_dataset('sig_offset', data=sig_offset[pkey][tkey])
                                f.create_dataset('ct_offset', data=ct_offset[pkey][tkey])
                        else:
                            logger.info("Debiasing failed - output contains Nones, check raw traces")
            else:
                raise ValueError('Extract traces first')
        else:  # if debiased traces exist - read them form h5 files
            logger.info("Debiased ROI traces exist in cache, reading from h5 file")

            ins = {}
            sig_offset = {}
            ct_offset = {}
            for pkey in self.pkeys:
                ins[pkey] = {}
                sig_offset[pkey] = {}
                ct_offset[pkey] = {}
                for tkey in self.tkeys:
                    self.ins_paths[pkey][tkey] = ins_paths[pkey][tkey]
                    with h5py.File(self.ins_paths[pkey][tkey], "r") as f:
                        ins[pkey][tkey] = f["debiased_traces"][()]
                        sig_offset[pkey][tkey] = f['sig_offset'][()]
                        ct_offset[pkey][tkey] = f['ct_offset'][()]
                    self.found_ins[pkey][tkey] = [True, True]
                    self.ins[pkey][tkey] = ins[pkey][tkey]
                    self.offsets[pkey][tkey] = {'sig_offset': sig_offset[pkey][tkey],
                                                'ct_offset': ct_offset[pkey][tkey]}
                    self.found_offsets[pkey][tkey] = [True, True]
        return

    def unmix_pair(self):
        """
        function to unmix a pair of panels:
        """
        # lcoal vars for out paths:
        outs_paths = {}
        for pkey in self.pkeys:
            outs_paths[pkey] = {}
            for tkey in self.tkeys:
                self.outs_paths[pkey][tkey] = None
                outs_paths[pkey][tkey] = os.path.join(self.dirs[tkey], f'{self.exp_ids[pkey]}_out.h5')

        # check if unmixed traces exist:
        out_traces_exist = True
        for pkey in self.pkeys:
            for tkey in self.tkeys:
                if not os.path.isfile(outs_paths[pkey][tkey]):
                    out_traces_exist = False

        for pkey in self.pkeys:
            for tkey in self.tkeys:
                if out_traces_exist:  # if output traces exist, set self.outs_paths
                    self.outs_paths[pkey][tkey] = outs_paths[pkey][tkey]
                else:
                    self.outs_paths[pkey][tkey] = None

        if not out_traces_exist:  # if ica output traces don't exist - run demixing
            # check if ica input traces exist
            in_traces_exist = True
            for pkey in self.pkeys:
                for tkey in self.tkeys:
                    if not self.found_ins[pkey][tkey]:
                        in_traces_exist = False

            if in_traces_exist:  # if ica input traces exist:

                # check if traces dont have nans or infs:
                in_traces_valid = True
                for pkey in self.pkeys:
                    for tkey in self.tkeys:
                        if np.any(np.isnan(self.ins[pkey][tkey])) or np.any(np.isinf(self.ins[pkey][tkey])):
                            in_traces_valid = False
                if not in_traces_valid:
                    raise ValueError(
                        "ValueError: ICA input contains NaN, infinity or a value too large for dtype float64")
                else:
                    logger.info("Unmixed traces do not exist in cache, running ICA")

                    rois_valid = {}
                    traces_in = {}
                    traces_out = {}
                    crosstalk = {}
                    mixing = {}
                    a_mixing = {}
                    figs_ct_in = {}
                    figs_ct_out = {}
                    for pkey in self.pkeys:
                        rois_valid[pkey] = {}
                        traces_in[pkey] = {}
                        traces_out[pkey] = {}
                        crosstalk[pkey] = {}
                        mixing[pkey] = {}
                        a_mixing[pkey] = {}
                        figs_ct_in[pkey] = {}
                        figs_ct_out[pkey] = {}
                        for tkey in self.tkeys:
                            rois_valid[pkey][tkey] = self.rois_valid[pkey]
                            traces_in[pkey][tkey] = self.ins[pkey][tkey]
                            # don't run unmixing if neuropil, instead read roi unmixing matrix
                            if tkey == 'np':
                                mixing[pkey][tkey] = self.a_mixing[pkey]['roi']

                                traces_out[pkey][tkey], crosstalk[pkey][tkey], mixing[pkey][tkey], a_mixing[pkey][tkey] \
                                    = self.unmix_plane(traces_in[pkey][tkey], rois_valid[pkey][tkey],
                                                       mixing[pkey][tkey])
                                crosstalk[pkey][tkey] = crosstalk[pkey]['roi']  # use same crosstalk value as per Roi traces (since the mixing matrix is assumed to be the same)
                            else:
                                traces_out[pkey][tkey], crosstalk[pkey][tkey], mixing[pkey][tkey], a_mixing[pkey][tkey] \
                                    = self.unmix_plane(traces_in[pkey][tkey], rois_valid[pkey][tkey])
                            # saving to self
                            self.outs[pkey][tkey] = np.array(
                                [traces_out[pkey][tkey][0] + self.offsets[pkey][tkey]['sig_offset'],
                                 traces_out[pkey][tkey][1] + self.offsets[pkey][tkey]['ct_offset']])
                            self.crosstalk[pkey][tkey] = crosstalk[pkey][tkey]
                            self.outs_paths[pkey][tkey] = outs_paths[pkey][tkey]
                            self.mixing[pkey][tkey] = mixing[pkey][tkey]
                            self.a_mixing[pkey][tkey] = a_mixing[pkey][tkey]
                            self.found_solution[pkey][tkey] = True
                            # writing ica ouput traces to disk
                            with h5py.File(self.outs_paths[pkey][tkey], "w") as f:
                                f.create_dataset("data", data=self.outs[pkey][tkey], compression="gzip")
                                f.create_dataset("roi_names", data=self.rois_names_valid[pkey][tkey])
                                f.create_dataset("crosstalk", data=self.crosstalk[pkey][tkey])
                                f.create_dataset("mixing_matrix_adjusted", data=self.a_mixing[pkey][tkey])
                                f.create_dataset("mixing_matrix", data=self.mixing[pkey][tkey])
        else:
            logger.info("Unmixed traces exist in cache, reading from h5 file")
            for pkey in self.pkeys:
                for tkey in self.tkeys:
                    self.outs_paths[pkey][tkey] = outs_paths[pkey][tkey]
                    self.found_solution[pkey][tkey] = True

                    with h5py.File(self.outs_paths[pkey][tkey], "r") as f:
                        self.outs[pkey][tkey] = f["data"][()]
                        self.crosstalk[pkey][tkey] = f["crosstalk"][()]
                        self.mixing[pkey][tkey] = f["mixing_matrix"][()]
                        self.a_mixing[pkey][tkey] = f["mixing_matrix_adjusted"][()]
        return

    def plot_ica_pair(self, pair, dir_name=None, samples=5000):
        if not dir_name:
            dir_name = self.names
        for pkey in self.pkeys:
            if pkey == 'pl1':
                plane = pair[0]
            else:
                plane = pair[1]
            for tkey in self.tkeys:
                plot_dir = os.path.join(self.session_dir,
                                        f'{dir_name[tkey]}_{pair[0]}_{pair[1]}/ica_plots_{plane}')
                if not os.path.isdir(plot_dir):
                    os.mkdir(plot_dir)

                for i in range(len(self.rois_names_valid[pkey][tkey])):
                    roi_name = self.rois_names_valid[pkey][tkey][i]
                    before_sig = self.ins[pkey][tkey][0][i] + self.offsets[pkey][tkey]['sig_offset'][i]
                    after_sig = self.outs[pkey][tkey][0][i]
                    before_ct = self.ins[pkey][tkey][1][i] + self.offsets[pkey][tkey]['ct_offset'][i]
                    after_ct = self.outs[pkey][tkey][1][i]
                    traces_before = [before_sig, before_ct]
                    traces_after = [after_sig, after_ct]
                    mixing = self.mixing[pkey][tkey][i]
                    a_mixing = self.a_mixing[pkey][tkey][i]
                    crosstalk_before = self.crosstalk[pkey][tkey][0][i]
                    crosstalk_after = self.crosstalk[pkey][tkey][1][i]
                    crosstalk = [crosstalk_before, crosstalk_after]
                    self.plot_roi(traces_before, traces_after, mixing, a_mixing, crosstalk, roi_name, plot_dir, samples)
                self.plot_dirs[pkey][tkey] = plot_dir
        return

    def validate_cells_crosstalk(self):
        """
        validate cells based ont the amount of crosstalk: if ct amount > 130 : cell is detected based on activity form it's paired plane
        :return:

        """
        for pkey in self.pkeys:
            ct_fn_roi = add_suffix_to_path(self.rois_valid_paths[pkey]['roi'], '_ct')
            ct_fn_np = add_suffix_to_path(self.rois_valid_paths[pkey]['np'], '_ct')
            if not os.path.isfile(ct_fn_roi) or not os.path.isfile(ct_fn_np):
                #             logging.info(f"Validating traces against crosstalk")
                tkey = 'roi'
                self.rois_valid_ct[pkey] = copy.deepcopy(self.rois_valid[pkey])
                crosstalk = self.crosstalk[pkey][tkey]
                crosstalk_before = crosstalk[0]

                roi_names = self.rois_names_valid[pkey][tkey]

                assert len(roi_names) == len(
                    crosstalk_before), "number of crosstalk values doesn't align with number of valid rois in ica.rois_names_valid"

                for i in range(len(roi_names)):
                    roi_name = str(roi_names[i])
                    ct_before = crosstalk_before[i]
                    if ct_before > 130:
                        self.rois_valid_ct[pkey][roi_name] = False
                ju.write(ct_fn_roi, self.rois_valid_ct[pkey])
                ju.write(ct_fn_np, self.rois_valid_ct[pkey])
            else:
                logging.info(f"Crosstalk validation json exists, skipping")
                self.rois_valid_ct[pkey] = ju.read(ct_fn_roi)
            self.rois_names_valid_ct[pkey]['roi'] = [roi for roi, valid in self.rois_valid_ct[pkey].items() if valid]
            self.rois_names_valid_ct[pkey]['np'] = [roi for roi, valid in self.rois_valid_ct[pkey].items() if valid]
            self.rois_valid_ct_paths[pkey]['roi'] = ct_fn_roi
            self.rois_valid_ct_paths[pkey]['np'] = ct_fn_np
        return

    def filter_dff_traces_crosstalk(self):
        for pkey in self.pkeys:
            tkey = 'roi'
            exp_id = self.exp_ids[pkey]
            self.dff_files[pkey] = os.path.join(self.session_dir, f"{exp_id}_dff.h5")
            self.dff_ct_files[pkey] = add_suffix_to_path(self.dff_files[pkey], '_ct')

            if not os.path.isfile(self.dff_ct_files[pkey]):
                logging.info(f"Filtering dff traces for exp: {self.exp_ids[pkey]}")
                if os.path.isfile(self.dff_files[pkey]):
                    with h5py.File(self.dff_files[pkey], 'r') as f:
                        self.dff[pkey] = f['data'][()]
                else:
                    print(f"Dff traces don't exist at {self.dff_files[pkey]}")
                    continue

                traces_dict = {}
                assert self.dff[pkey].shape[0] == len(self.rois_names_valid[pkey][tkey]), f"dff traces are not aligned " \
                                                       f"to validation json for {self.exp_ids[pkey]}"
                for i in range(len(self.rois_names_valid[pkey][tkey])):
                    roi_name = self.rois_names_valid[pkey][tkey][i]
                    traces_dict[roi_name] = self.dff[pkey][i]

                self.dff_ct[pkey] = [traces for roi_name, traces in traces_dict.items() if
                                     self.rois_valid_ct[pkey][str(roi_name)]]

                with h5py.File(self.dff_ct_files[pkey], "w") as f:
                    f.create_dataset("data", data=self.dff_ct[pkey], compression="gzip")
                    f.create_dataset("roi_names", data=[int(roi) for roi in self.rois_names_valid_ct[pkey][tkey]])
            else:
                logging.info(f"Filtered dff traces for exp: {self.exp_ids[pkey]} exist, reading from h5 file")
                with h5py.File(self.dff_ct_files[pkey], "r") as f:
                    self.dff_ct[pkey] = f["data"][()]
                with h5py.File(self.dff_files[pkey], "r") as f:
                    self.dff[pkey] = f["data"][()]
        return

    def filter_demixed_traces(self):
        for pkey in self.pkeys:
            tkey = 'roi'
            exp_id = self.exp_ids[pkey]
            dem_dir = os.path.join(self.session_dir, f"demixing_{exp_id}")
            self.dem_files[pkey] = os.path.join(dem_dir, f"traces_demixing_output_{self.exp_ids[pkey]}.h5")
            self.dem_ct_files[pkey] = add_suffix_to_path(self.dem_files[pkey], '_ct')

            if not os.path.isfile(self.dem_ct_files[pkey]):
                logging.info(f"Filtering demixed traces for exp: {self.exp_ids[pkey]}")
                if os.path.isfile(self.dem_files[pkey]):
                    with h5py.File(self.dem_files[pkey], 'r') as f:
                        self.dem[pkey] = f['data'][()]
                else:
                    print(f"Demixed traces don't exist at {self.dem_files[pkey]}")
                    continue

                assert self.dem[pkey].shape[0] == len(
                    self.rois_names_valid[pkey][tkey]), f"Demixed traces are not aligned to validation json for {self.exp_ids[pkey]}"

                traces_dict = {}
                for i in range(len(self.rois_names_valid[pkey][tkey])):
                    roi_name = self.rois_names_valid[pkey][tkey][i]
                    traces_dict[roi_name] = self.dem[pkey][i]

                self.dem_ct[pkey] = [traces for roi_name, traces in traces_dict.items() if
                                     self.rois_valid_ct[pkey][str(roi_name)]]

                with h5py.File(self.dem_ct_files[pkey], "w") as f:
                    f.create_dataset("data", data=self.dem_ct[pkey], compression="gzip")
                    f.create_dataset("roi_names",
                                     data=[int(roi) for roi in self.rois_names_valid_ct[pkey][tkey]])
            else:
                logging.info(f"Filtered demixed traces for exp: {self.exp_ids[pkey]} exist, reading from h5 file")
                with h5py.File(self.dem_ct_files[pkey], "r") as f:
                    self.dem_ct[pkey] = f["data"][()]
                with h5py.File(self.dem_files[pkey], "r") as f:
                    self.dem[pkey] = f["data"][()]
        return

    def filter_np_corrected_traces(self):
        for pkey in self.pkeys:
            tkey = 'roi'
            exp_id = self.exp_ids[pkey]

            np_cor_dir = os.path.join(self.session_dir, f"neuropil_corrected_{exp_id}")
            self.np_cor_files[pkey] = os.path.join(np_cor_dir, f"neuropil_correction.h5")

            self.np_cor_ct_files[pkey] = add_suffix_to_path(self.np_cor_files[pkey], '_ct')

            if not os.path.isfile(self.np_cor_ct_files[pkey]):
                logging.info(f"Filtering neuropil corrected traces for exp: {self.exp_ids[pkey]}")
                self.np_cor_ct[pkey] = {}
                self.np_cor[pkey] = {}
                if os.path.isfile(self.np_cor_files[pkey]):
                    with h5py.File(self.np_cor_files[pkey], 'r') as f:
                        self.np_cor[pkey]['FC'] = f['FC'][()]
                        self.np_cor[pkey]["RMSE"] = f["RMSE"][()]
                        self.np_cor[pkey]["r"] = f["r"][()]
                        self.np_cor_ct[pkey]['RMSE'] = f['RMSE'][()]
                        self.np_cor_ct[pkey]['r'] = f['r'][()]

                else:
                    print(f"Neuropil corrected traces don't exist at {self.np_cor_files[pkey]}")
                assert self.np_cor[pkey]['FC'].shape[0] == len(
                    self.rois_names_valid[pkey][tkey]), f"Neuropil corrected traces are not aligned to validation json for {self.exp_ids[pkey]}"

                traces_dict = {}
                for i in range(len(self.rois_names_valid[pkey][tkey])):
                    roi_name = self.rois_names_valid[pkey][tkey][i]
                    traces_dict[roi_name] = self.np_cor[pkey]['FC'][i]

                self.np_cor_ct[pkey]['FC'] = [traces for roi_name, traces in traces_dict.items() if
                                              self.rois_valid_ct[pkey][str(roi_name)]]

                with h5py.File(self.np_cor_ct_files[pkey], "w") as f:
                    f.create_dataset("data", data=self.np_cor_ct[pkey]['FC'], compression="gzip")
                    f.create_dataset("roi_names",
                                     data=[int(roi) for roi in self.rois_names_valid_ct[pkey][tkey]])
                    f.create_dataset("RMSE", self.np_cor_ct[pkey]["RMSE"])
                    f.create_dataset("r", self.np_cor[pkey]["r"])

            else:
                logging.info(f"Filtered neuropil corrected traces for exp: {self.exp_ids[pkey]} exist, reading from h5 file")
                with h5py.File(self.np_cor_ct_files[pkey], "r") as f:
                    self.np_cor_ct[pkey] = f["data"][()]
                    self.np_cor_ct[pkey]["RMSE"] = f["RMSE"][()]
                    self.np_cor_ct[pkey]["r"] = f["r"][()]
                with h5py.File(self.np_cor_files[pkey], "r") as f:
                    self.np_cor[pkey] = f["FC"][()]
                    self.np_cor[pkey]["RMSE"] = f["RMSE"][()]
                    self.np_cor[pkey]["r"] = f["r"][()]
        return


    @staticmethod
    def plot_roi(traces_before, traces_after, mixing, a_mixing, crosstalk, roi_name, plot_dir, samples):

        pdf_name = os.path.join(plot_dir, f"cell_{roi_name}.pdf")
        if os.path.isfile(pdf_name):
            logging.info(f"cell trace figure exist for  cell {roi_name}")
        else:
            logging.info(f"creating figures for cell {roi_name}")

            # define pdf filename:
            pdf = plt_pdf.PdfPages(pdf_name)

            # get crosstalk data and plot on first page of the pdf:
            # before demixing
            slope_before, offset_before, r_value_b, [hist_before, xedges_b, yedges_b, fitfn_b] = get_crosstalk_data(traces_before[0],
                                                                                                                    traces_before[1],
                                                                                                                    generate_plot_data=True)
            f = plt.figure(figsize=(30, 10))
            plt.rcParams.update({'font.size': 28})
            plt.suptitle(f"Crosstalk plost for cell {roi_name}\n", linespacing=0.5)
            xlabel = "signal"
            ylabel = "crosstalk"
            # plot crosstalk before demxing
            plt.subplot(131)
            plt.imshow(hist_before, interpolation='nearest', origin='low',
                       extent=[xedges_b[0], xedges_b[-1], yedges_b[0], yedges_b[-1]], aspect='auto', norm=LogNorm())
            cbar = plt.colorbar()
            cbar.set_label('# of counts', rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=18)
            plt.plot(hist_before, fitfn_b(hist_before), '--k')
            plt.xlim((xedges_b[0], xedges_b[-1]))
            plt.ylim((yedges_b[0], yedges_b[-1]))
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            title = f"Crosstalk before: {round(crosstalk[0], 3)}\n{fitfn_b} R2={round(r_value_b, 2)}"
            plt.title(title, linespacing=0.5, fontsize=18)

            # after demxing
            slope_after, offset_after, r_value_a, [hist_after, xedges_a, yedges_a, fitfn_a] = get_crosstalk_data(traces_after[0],
                                                                                                                 traces_after[1],
                                                                                                                 generate_plot_data=True)
            plt.subplot(132)
            plt.rcParams.update({'font.size': 28})
            plt.imshow(hist_after, interpolation='nearest', origin='low',
                       extent=[xedges_a[0], xedges_a[-1], yedges_a[0], yedges_a[-1]], aspect='auto', norm=LogNorm())
            cbar = plt.colorbar()
            cbar.set_label('# of counts', rotation=270, labelpad=20)
            cbar.ax.tick_params(labelsize=18)
            plt.plot(hist_after, fitfn_a(hist_after), '--k')
            plt.xlim((xedges_a[0], xedges_a[-1]))
            plt.ylim((yedges_a[0], yedges_a[-1]))
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            title = f"Crosstalk after: {round(crosstalk[1], 2)}\n{fitfn_a} R2={round(r_value_a, 2)}"
            plt.title(title, linespacing=0.5, fontsize=18)

            # add mixing matrix info to the plot
            plt.subplot(133)
            ax = plt.gca()
            plt.text(0.5, 0.65, f"Raw mixing matrix:\n{np.round(mixing, 2)}", fontsize=35, linespacing=2.0,
                     horizontalalignment='center',
                     verticalalignment='center', )
            plt.text(0.5, 0.25, f"Adjusted mixing matrix:\n{np.round(a_mixing, 2)}", fontsize=35, linespacing=2.0,
                     horizontalalignment='center',
                     verticalalignment='center', )
            ax.set_axis_off()
            plt.tick_params(left=False, labelleft=False)
            pdf.savefig(f)
            plt.close()

            # plot traces of {roi_name} roi : two plots per page: before ica, after ica
            y_min = min(min(traces_before[0]), min(traces_before[1]), min(traces_after[0]), min(traces_after[1]))
            y_max = max(max(traces_before[0]), max(traces_before[1]), max(traces_after[0]), max(traces_after[1]))

            for i in range(int(traces_before[0].shape[0] / samples) + 1):
                sig_before_i = traces_before[0][i * samples:(i + 1) * samples]
                ct_before_i = traces_before[1][i * samples:(i + 1) * samples]
                sig_after_i = traces_after[0][i * samples:(i + 1) * samples]
                ct_after_i = traces_after[1][i * samples:(i + 1) * samples]

                f1 = plt.figure(figsize=(20, 10))
                plt.rcParams.update({'font.size': 22})
                plt.subplot(211)
                plt.ylim(y_min, y_max)
                plt.plot(sig_before_i, 'r-', label='signal pl')
                plt.plot(ct_before_i, 'g-', label='cross-talk pl')
                plt.title(f'raw traces for cell {roi_name}', fontsize=18)
                plt.legend(loc='best')

                plt.subplot(212)
                plt.ylim(y_min, y_max)
                plt.plot(sig_after_i, 'r-', label='signal pl')
                plt.plot(ct_after_i, 'g-', label='cross-talk pl')
                plt.title(f'post-ica traces, cell # {roi_name}', fontsize=18)
                plt.legend(loc='best')
                pdf.savefig(f1)
                plt.close()
            pdf.close()
        return

    @staticmethod
    def unmix_plane(ica_in, ica_roi_valid, mixing=None):
        roi_names = [roi for roi, valid in ica_roi_valid.items() if valid]
        traces_sig = ica_in[0, :, :]
        traces_ct = ica_in[1, :, :]
        pl_mixing = []
        pl_a_mixing = []
        pl_crosstalk = np.empty((2, ica_in.shape[1]))
        ica_pl_out = np.empty(ica_in.shape)

        if mixing is not None:  # this is indicative that traces are form neuropil, use provided mixing to unmix them
            for i in range(len(roi_names)):
                mixing_roi = mixing[i]
                trace_sig = traces_sig[i]
                trace_ct = traces_ct[i]
                traces = np.array([trace_sig, trace_ct]).T
                # get unmixing matrix by inverting roi mixing matrix
                a_unmix = linalg.pinv(mixing_roi)
                # recontructing sources
                r_sources = np.dot(a_unmix, traces.T).T
                pl_mixing.append(mixing_roi)
                pl_a_mixing.append(mixing_roi)
                trace_sig_out = r_sources[:, 0]
                trace_ct_out = r_sources[:, 1]
                rescaled_trace_sig_out = rescale(trace_sig, trace_sig_out)
                rescaled_trace_ct_out = rescale(trace_ct, trace_ct_out)
                ica_pl_out[0, i, :] = rescaled_trace_sig_out
                ica_pl_out[1, i, :] = rescaled_trace_ct_out

        else:  # traces are rois, do full unmixing.
            # extract events
            traces_sig_evs, traces_ct_evs, valid = extract_active(ica_in, len_ne=20, th_ag=10, do_plots=0)
            # run ica on active traces, apply unmixing matrix to the entire trace
            ica_pl_out = np.empty(ica_in.shape)
            # perform unmixing separately on eah ROI:
            for i in range(len(roi_names)):
                # get events traces
                trace_sig_evs = traces_sig_evs[i]
                trace_ct_evs = traces_ct_evs[i]
                mix, a_mix, a_unmix, r_sources_evs = run_ica(trace_sig_evs, trace_ct_evs)
                adjusted_mixing_matrix = a_mix
                mixing_matrix = mix
                trace_sig_evs_out = r_sources_evs[:, 0]
                trace_ct_evs_out = r_sources_evs[:, 1]
                rescaled_trace_sig_evs_out = rescale(trace_sig_evs, trace_sig_evs_out)
                rescaled_trace_ct_evs_out = rescale(trace_ct_evs, trace_ct_evs_out)

                # calculating crosstalk : on events
                slope_in, _, _, _ = get_crosstalk_data(trace_sig_evs, trace_ct_evs, generate_plot_data=False)
                slope_out, _, _, _ = get_crosstalk_data(rescaled_trace_sig_evs_out, rescaled_trace_ct_evs_out,
                                                        generate_plot_data=False)
                crosstalk_before_demixing_evs = slope_in * 100
                crosstalk_after_demixing_evs = slope_out * 100
                pl_crosstalk[:, i] = [crosstalk_before_demixing_evs, crosstalk_after_demixing_evs]

                # applying unmixing matrix to the entire trace
                trace_sig = traces_sig[i]
                trace_ct = traces_ct[i]
                # recontructing sources
                traces = np.array([trace_sig, trace_ct]).T
                r_sources = np.dot(a_unmix, traces.T).T
                pl_a_mixing.append(adjusted_mixing_matrix)
                pl_mixing.append(mixing_matrix)
                trace_sig_out = r_sources[:, 0]
                trace_ct_out = r_sources[:, 1]
                rescaled_trace_sig_out = rescale(trace_sig, trace_sig_out)
                rescaled_trace_ct_out = rescale(trace_ct, trace_ct_out)
                ica_pl_out[0, i, :] = rescaled_trace_sig_out
                ica_pl_out[1, i, :] = rescaled_trace_ct_out

        return ica_pl_out, pl_crosstalk, pl_mixing, pl_a_mixing

    @staticmethod
    def validate_against_vba(rois_valid_ica, exp_id, vba_cache=VBA_CACHE):
        """
        :param rois_valid_ica: dict, returned by MesoscopeICA.rois_valid or MesoscopeICA.rois_valid
        :param exp_id: int, LIMS experiment ID, can be retrieved by MesoscopeICA.pl1_exp_id or MesoscopeICA.pl2_exp_id
        :param vba_cache: str, path to visual behavior analysis package cache directory
        :return: rois_valid_vba: dict, same structure as rois_valid_ica
        """
        dataset = VisualBehaviorOphysDataset(exp_id, cache_dir=vba_cache)
        roi_names_vba = dataset.cell_specimen_ids
        # invalid rois in ica validation json
        rois_valid_vba = rois_valid_ica
        for roi_name in rois_valid_ica:
            if int(roi_name) not in roi_names_vba:
                rois_valid_vba[str(roi_name)] = False
                rois_valid_vba[str(roi_name)] = False
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


def get_crosstalk_data(x, y, generate_plot_data=True):

    slope, offset, r_value, p_value, std_err = linregress(x, y)

    if generate_plot_data:
        h, xedges, yedges = np.histogram2d(x, y, bins=(30, 30))
        h = h.T
        fit_fn = np.poly1d([slope, offset])
        plot_output = [h, xedges, yedges, fit_fn]
    else:
        plot_output = None

    return slope, offset, r_value, plot_output


def plot_pixel_hist2d(x, y, xlabel='signal', ylabel='crosstalk', title=None, save_fig=False, save_path=None,
                      fig_show=False, colorbar=False):
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


def run_ica(sig, ct):
    traces = np.array([sig, ct]).T
    f_ica = FastICA(n_components=2, max_iter=50)
    _ = f_ica.fit_transform(traces)  # Reconstruct signals
    mix = f_ica.mixing_  # Get estimated mixing matrix
    # make sure no negative coeffs (inversion of traces)
    a_mix = mix
    a_mix[a_mix < 0] *= -1
    var_sig = np.var(sig)
    var_ct = np.var(ct)
    if a_mix[0, 0] < a_mix[1, 0]:
        a_mix[[0, 1], 0] = a_mix[[1, 0], 0]
    if a_mix[1, 1] < a_mix[0, 1]:
        a_mix[[0, 1], 1] = a_mix[[1, 0], 1]
    if var_ct > var_sig:
        if (a_mix[0, 0] + a_mix[1, 0]) > (a_mix[0, 1] + a_mix[1, 1]):
            #        swap rows:
            a_mix[[0, 1], :] = a_mix[[1, 0], :]
            # swap columns:
            a_mix[:, [0, 1]] = a_mix[:, [1, 0]]
    else:
        if (a_mix[0, 0] + a_mix[1, 0]) < (a_mix[0, 1] + a_mix[1, 1]):
            # swap rows:
            a_mix[[0, 1], :] = a_mix[[1, 0], :]
            # swap columns:
            a_mix[:, [0, 1]] = a_mix[:, [1, 0]]
    a_unmix = linalg.pinv(a_mix)
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
    valid = []
    for i in range(evs_ind.shape[0]):
        if not np.any(np.isnan(evs_ind[i])):
            trace_ct = traces_ct[i, evs_ind[i]]
            traces_ct_evs.append(trace_ct)
            valid.append(True)
        else:
            logger.info(f"No events detected")
            traces_sig_evs[i] = traces_sig[i]
            traces_ct_evs.append(traces_ct[i])
            valid.append(False)
    return traces_sig_evs, traces_ct_evs, valid


def add_suffix_to_path(abs_path, suffix):
    file_ext = os.path.splitext(abs_path)[1]
    file_name = os.path.splitext(abs_path)[0]
    new_file_name = f"{file_name}{suffix}{file_ext}"
    return new_file_name
