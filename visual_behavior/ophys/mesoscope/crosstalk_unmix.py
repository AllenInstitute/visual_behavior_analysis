import matplotlib

matplotlib.use('Agg')
from allensdk.brain_observatory import roi_masks
import visual_behavior.ophys.mesoscope.mesoscope as ms
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

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

IMAGEH, IMAGEW = 512, 512

def get_traces(movie_exp_dir, movie_exp_id, mask_exp_dir, mask_exp_id):
    jin_movie_path = os.path.join(movie_exp_dir, f"processed/{movie_exp_id}_input_extract_traces.json")

    jin_mask_path = os.path.join(mask_exp_dir, f"processed/{mask_exp_id}_input_extract_traces.json")

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
    roi_names = [ roi.label for roi in roi_mask_list ]

    traces, neuropil_traces = roi_masks.calculate_roi_and_neuropil_traces(movie_h5, roi_mask_list, motion_border)
    return traces, neuropil_traces, roi_names


def create_roi_masks(rois, w, h, motion_border):
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

    def __init__(self, session_id, cache):

        self.session_id = session_id
        self.dataset = ms.MesoscopeDataset(session_id)
        self.session_cache_dir = None

        self.plane1_exp_id = None
        self.plane2_exp_id = None

        self.ica_traces_dir = None
        self.ica_neuropil_dir = None
        self.found_original_traces = None  # output of get_traces
        self.found_original_neuropil = None  # output of get_traces

        self.plane1_traces_orig = None
        self.plane1_neuropil_orig = None
        self.plane1_traces_orig_pointer = None
        self.plane1_neuropil_orig_pointer = None
        self.plane1_ica_input = None
        self.plane1_ica_neuropil_input = None
        self.plane1_ica_input_pointer = None
        self.plane1_ica_neuropil_input_pointer = None
        self.plane1_ica_output = None
        self.plane1_ica_neuropil_output = None
        self.plane1_ica_output_pointer = None
        self.plane1_ica_neuropil_output_pointer = None

        self.plane2_traces_orig = None  # output of get_traces
        self.plane2_neuropil_orig = None
        self.plane2_traces_orig_pointer = None
        self.plane2_neuropil_orig_pointer = None
        self.plane2_ica_input = None  # output of combine_debiase
        self.plane2_ica_neuropil_input = None
        self.plane2_ica_input_pointer = None
        self.plane2_ica_neuropil_input_pointer = None
        self.plane2_ica_output = None  # output of unmix_traces
        self.plane2_ica_neuropil_output = None
        self.plane2_ica_output_pointer = None
        self.plane2_ica_neuropil_output_pointer = None

        self.ica_mixing_matrix_traces_pointer = None
        self.ica_mixing_matrix_neuropil_pointer = None

        self.plane1_offset = None
        self.plane2_offset = None

        self.plane1_neuropil_offset = None
        self.plane2_neuropil_offset = None

        self.ica_traces_scale_top = None
        self.ica_traces_scale_bot = None
        self.ica_neuropil_scale_top = None
        self.ica_neuropil_scale_bot = None

        self.found_solution = None  # output of unmix_traces
        self.found_solution_neuropil = None

        self.found_ica_input = [None, None]
        self.found_ica_offset = [None, None]
        self.found_ica_neuropil_input = [None, None]
        self.found_ica_neuropil_offset = [None, None]

        self.traces_matrix = None
        self.neuropil_matrix = None

        self.traces_unmix = None
        self.neuropil_unmix = None

        self.cache = cache

        self.plane1_roi_names = None
        self.plane2_roi_names = None

        self.plane1_roi_traces_valid = None
        self.plane2_roi_traces_valid = None

        self.plane1_roi_traces_valid_pointer = None
        self.plane2_roi_traces_valid_pointer = None

        self.plane1_neuropil_traces_valid_pointer = None
        self.plane2_neuropil_traces_valid_pointer = None

        self.plane1_neuropil_traces_valid = None
        self.plane2_neuropil_traces_valid = None

    def set_analysis_session_dir(self):
        self.session_cache_dir = os.path.join(self.cache, f'session_{self.session_id}')
        return self.session_cache_dir

    def set_ica_traces_dir(self, pair):
        session_dir = self.set_analysis_session_dir()
        self.ica_traces_dir = os.path.join(session_dir, f'ica_traces_{pair[0]}_{pair[1]}/')
        self.plane1_ica_output_pointer = os.path.join(self.ica_traces_dir,
                                                      f'traces_ica_output_{pair[0]}.h5')
        self.plane2_ica_output_pointer = os.path.join(self.ica_traces_dir,

                                                      f'traces_ica_output_{pair[1]}.h5')
        self.ica_mixing_matrix_pointer = os.path.join(self.ica_traces_dir,
                                                      f'traces_ica_mixing.h5')
        return

    def set_neuropil_ica_dir(self, pair):
            session_dir = self.set_analysis_session_dir()
            self.ica_neuropil_dir = os.path.join(session_dir, f'ica_neuropil_{pair[0]}_{pair[1]}/')

            self.plane1_ica_neuropil_output_pointer = os.path.join(self.ica_neuropil_dir, f'neuropil_ica_output_{pair[0]}.h5')
            self.plane2_ica_neuropil_output_pointer = os.path.join(self.ica_neuropil_dir, f'neuropil_ica_output_{pair[1]}.h5')
            self.ica_mixing_matrix_neuropil_pointer = os.path.join(self.ica_neuropil_dir, f'neuropil_ica_mixing.h5')

            return

    def get_ica_traces(self, pair):

        # we will first check if traces exist, if yes - read them, if not - extract them
        self.plane1_roi_names = None
        self.plane2_roi_names = None

        self.found_original_traces = [False, False]
        self.found_original_neuropil = [False, False]

        plane1_exp_id = pair[0]
        plane2_exp_id = pair[1]

        self.plane1_exp_id = plane1_exp_id
        self.plane2_exp_id = plane2_exp_id

        session_dir = os.path.join(self.cache, f'session_{self.session_id}')
        self.session_cache_dir = session_dir

        # path to ica traces:
        # for roi
        ica_traces_dir = os.path.join(session_dir, f'ica_traces_{plane1_exp_id}_{plane2_exp_id}/')
        self.ica_traces_dir = ica_traces_dir
        path_traces_plane1 = f'{ica_traces_dir}traces_original_{plane1_exp_id}.h5'
        path_traces_plane2 = f'{ica_traces_dir}traces_original_{plane2_exp_id}.h5'
        # for neuropil
        ica_neuropil_dir = os.path.join(session_dir, f'ica_neuropil_{plane1_exp_id}_{plane2_exp_id}/')
        self.ica_neuropil_dir = ica_neuropil_dir
        path_neuropil_plane1 = f'{ica_neuropil_dir}neuropil_original_{plane1_exp_id}.h5'
        path_neuropil_plane2 = f'{ica_neuropil_dir}neuropil_original_{plane2_exp_id}.h5'

        # let's see if all traces exist already:
        if os.path.isfile(path_traces_plane1) and os.path.isfile(path_traces_plane2) and os.path.isfile(
                path_neuropil_plane1) and os.path.isfile(path_neuropil_plane2):
            # if both traces exist, skip extracting:

            logger.info('Found traces in cache, reading from h5 file')
            # read traces form h5 file:
            with h5py.File(path_traces_plane1, "r") as f:
                plane1_traces_original = f["data"].value
                plane1_roi_names = f["roi_names"].value
            with h5py.File(path_traces_plane2, "r") as f:
                plane2_traces_original = f["data"].value
                plane2_roi_names = f["roi_names"].value

            # read neuropil traces form h5 file:
            with h5py.File(path_neuropil_plane1, "r") as f:
                plane1_neuropil_original = f["data"].value

            with h5py.File(path_neuropil_plane2, "r") as f:
                plane2_neuropil_original = f["data"].value

            self.plane1_traces_orig_pointer = path_traces_plane1
            self.plane2_traces_orig_pointer = path_traces_plane2
            self.plane1_traces_orig = plane1_traces_original
            self.plane2_traces_orig = plane2_traces_original
            self.plane1_roi_names = plane1_roi_names
            self.plane2_roi_names = plane2_roi_names

            #  same for neuropil:
            self.plane1_neuropil_orig_pointer = path_neuropil_plane1
            self.plane2_neuropil_orig_pointer = path_neuropil_plane2
            self.plane1_neuropil_orig = plane1_neuropil_original
            self.plane2_neuropil_orig = plane2_neuropil_original
            # set found traces flag True
            self.found_original_traces = [True, True]
            # set found neuropil traces flag True
            self.found_original_neuropil = [True, True]

        else:
            # some traces are missing, run extraction:
            logger.info('Traces dont exist in cache, extracting')

            # if traces don't exist, do we need to re-set unmixed and debiased traces flaggs to none?
            # yes, as we want unmixed traces be output on ICA using original traces
            self.plane1_ica_input_pointer = None
            self.plane2_ica_input_pointer = None
            self.plane1_ica_output_pointer = None
            self.plane2_ica_output_pointer = None

            self.plane1_ica_neuropil_input_pointer = None
            self.plane2_ica_neuropil_input_pointer = None
            self.plane1_ica_neuropil_output_pointer = None
            self.plane2_ica_neuropil_output_pointer = None

            self.plane1_roi_names = None
            self.plane2_roi_names = None

            plane1_folder = self.dataset.get_exp_folder(plane1_exp_id)
            plane2_folder = self.dataset.get_exp_folder(plane2_exp_id)

            # extract signal and crosstalk traces for plane 1
            plane1_sig_traces, plane1_sig_neuropil, plane1_roi_names = get_traces(plane1_folder, plane1_exp_id,
                                                                                      plane1_folder, plane1_exp_id)
            plane1_ct_traces, plane1_ct_neuropil, _ = get_traces(plane2_folder, plane2_exp_id, plane1_folder,
                                                                     plane1_exp_id)
            # extract signal and crosstalk traces for plane 2
            plane2_sig_traces, plane2_sig_neuropil, plane2_roi_names = get_traces(plane2_folder, plane2_exp_id,
                                                                                      plane2_folder, plane2_exp_id)
            plane2_ct_traces, plane2_ct_neuropil, _ = get_traces(plane1_folder, plane1_exp_id, plane2_folder,
                                                                     plane2_exp_id)

            # setting traces valid flag: if none is None
            if (not plane1_sig_traces.any() is None) and (not plane1_ct_traces.any() is None):
                self.found_original_traces[0] = True
            if (not plane2_sig_traces.any() is None) and (not plane2_ct_traces.any() is None):
                self.found_original_traces[1] = True
            if (not plane1_sig_neuropil.any() is None) and (not plane1_ct_neuropil.any() is None):
                self.found_original_neuropil[0] = True
            if (not plane2_sig_neuropil.any() is None) and (not plane2_ct_neuropil.any() is None):
                self.found_original_neuropil[1] = True

            # DOES ROI traces DIR EXIST?
            if not os.path.isdir(session_dir):
                os.mkdir(session_dir)
            if not os.path.isdir(ica_traces_dir):
                os.mkdir(ica_traces_dir)
            # if extracted traces valid, save to disk:
            if self.found_original_traces[0] and self.found_original_traces[1]:
                # combining traces, saving to self, writing to disk:
                plane1_traces_original = np.array([plane1_sig_traces, plane1_ct_traces])

                self.plane1_traces_orig = plane1_traces_original
                self.plane1_traces_orig_pointer = path_traces_plane1
                self.plane1_roi_names = plane1_roi_names
                with h5py.File(path_traces_plane1, "w") as f:
                    f.create_dataset(f"data", data=plane1_traces_original)
                    f.create_dataset(f"roi_names", data=np.int_(plane1_roi_names))

                # same for plane 2:
                plane2_traces_original = np.array([plane2_sig_traces, plane2_ct_traces])
                self.plane2_traces_orig = plane2_traces_original
                self.plane2_traces_orig_pointer = path_traces_plane2
                self.plane2_roi_names = plane2_roi_names
                with h5py.File(path_traces_plane2, "w") as f:
                    f.create_dataset(f"data", data=plane2_traces_original)
                    f.create_dataset(f"roi_names", data=np.int_(plane2_roi_names))

            if not os.path.isdir(session_dir):
                os.mkdir(session_dir)
            if not os.path.isdir(ica_neuropil_dir):
                os.mkdir(ica_neuropil_dir)

            # if extracted traces not None, save to disk:
            if self.found_original_neuropil[0] and self.found_original_neuropil[1]:
                # combining traces, saving to self, writing to disk:
                plane1_neuropil_original = np.array([plane1_sig_neuropil, plane1_ct_neuropil])
                self.plane1_neuropil_orig = plane1_neuropil_original
                self.plane1_neuropil_orig_pointer = path_neuropil_plane1
                with h5py.File(path_neuropil_plane1, "w") as f:
                    f.create_dataset(f"data", data=plane1_neuropil_original)
                    f.create_dataset(f"roi_names", data=np.int_(plane1_roi_names))
                # same for plane 2:
                plane2_neuropil_original = np.array([plane2_sig_neuropil, plane2_ct_neuropil])
                self.plane2_neuropil_orig = plane2_neuropil_original
                self.plane2_neuropil_orig_pointer = path_neuropil_plane2
                with h5py.File(path_neuropil_plane2, "w") as f:
                    f.create_dataset(f"data", data=plane2_neuropil_original)
                    f.create_dataset(f"roi_names", data=np.int_(plane2_roi_names))

        return self.found_original_traces, self.found_original_neuropil

    def validate_traces(self):

        self.plane1_roi_traces_valid_pointer = None
        self.plane2_roi_traces_valid_pointer = None

        self.plane1_neuropil_traces_valid_pointer = None
        self.plane2_neuropil_traces_valid_pointer = None

        plane1_roi_traces_valid_pointer = os.path.join(self.ica_traces_dir, f'valid_{self.plane1_exp_id}.json')
        plane2_roi_traces_valid_pointer = os.path.join(self.ica_traces_dir, f'valid_{self.plane2_exp_id}.json')

        plane1_neuropil_traces_valid_pointer = os.path.join(self.ica_neuropil_dir, f'valid_{self.plane1_exp_id}.json')
        plane2_neuropil_traces_valid_pointer = os.path.join(self.ica_neuropil_dir, f'valid_{self.plane2_exp_id}.json')

        # validation json already exists, skip validating
        if os.path.isfile(plane1_roi_traces_valid_pointer) and os.path.isfile(
                plane2_roi_traces_valid_pointer) and os.path.isfile(
            plane1_neuropil_traces_valid_pointer) and os.path.isfile(
            plane2_neuropil_traces_valid_pointer):
            self.plane1_roi_traces_valid_pointer = plane1_roi_traces_valid_pointer
            self.plane2_roi_traces_valid_pointer = plane2_roi_traces_valid_pointer
            self.plane1_neuropil_traces_valid_pointer = plane1_neuropil_traces_valid_pointer
            self.plane2_neuropil_traces_valid_pointer = plane2_neuropil_traces_valid_pointer
        else:
            self.plane1_roi_traces_valid_pointer = None
            self.plane2_roi_traces_valid_pointer = None
            self.plane1_neuropil_traces_valid_pointer = None
            self.plane2_neuropil_traces_valid_pointer = None

        if (not self.plane1_roi_traces_valid_pointer) and (not self.plane2_roi_traces_valid_pointer) and (
        not self.plane1_neuropil_traces_valid_pointer) and (not self.plane2_neuropil_traces_valid_pointer):

            # traces need validation:
            if self.found_original_traces[0] and self.found_original_traces[1] and self.found_original_neuropil[0] and \
                    self.found_original_neuropil[1]:
                # traces, plane 1:

                plane1_sig_trace_valid = {}
                plane1_sig_neuropil_valid = {}
                plane1_ct_trace_valid = {}
                plane1_ct_neuropil_valid = {}

                num_traces_roi_sig, _ = self.plane1_traces_orig[0].shape
                num_traces_neuropil_sig, _ = self.plane1_neuropil_orig[0].shape
                num_traces_roi_ct, _ = self.plane1_traces_orig[1].shape
                num_traces_neuropil_ct, _ = self.plane1_neuropil_orig[1].shape

                if not (num_traces_roi_sig == num_traces_neuropil_sig) or not (
                        num_traces_roi_ct == num_traces_neuropil_ct):
                    logger.info('Neuropil and ROI traces are not aligned')
                else:
                    for n in range(num_traces_roi_sig):
                        trace_roi_sig = self.plane1_traces_orig[0][n]
                        trace_neuropil_sig = self.plane1_neuropil_orig[0][n]
                        trace_roi_ct = self.plane1_traces_orig[1][n]
                        trace_neuropil_ct = self.plane1_neuropil_orig[1][n]

                        if np.any(np.isnan(trace_roi_sig)) or np.any(np.isnan(trace_neuropil_sig)) or np.any(
                                np.isnan(trace_roi_ct)) or np.any(np.isnan(trace_neuropil_ct)):
                            plane1_sig_trace_valid[str(self.plane1_roi_names[n])] = False
                            plane1_sig_neuropil_valid[str(self.plane1_roi_names[n])] = False
                            plane1_ct_trace_valid[str(self.plane1_roi_names[n])] = False
                            plane1_ct_neuropil_valid[str(self.plane1_roi_names[n])] = False
                        else:
                            plane1_sig_trace_valid[str(self.plane1_roi_names[n])] = True
                            plane1_sig_neuropil_valid[str(self.plane1_roi_names[n])] = True
                            plane1_ct_trace_valid[str(self.plane1_roi_names[n])] = True
                            plane1_ct_neuropil_valid[str(self.plane1_roi_names[n])] = True

                # traces, plane 2:
                # signal:
                plane2_sig_trace_valid = {}
                plane2_sig_neuropil_valid = {}
                plane2_ct_trace_valid = {}
                plane2_ct_neuropil_valid = {}

                num_traces_roi_sig, _ = self.plane2_traces_orig[0].shape
                num_traces_neuropil_sig, _ = self.plane2_neuropil_orig[0].shape
                num_traces_roi_ct, _ = self.plane2_traces_orig[1].shape
                num_traces_neuropil_ct, _ = self.plane2_neuropil_orig[1].shape

                if not (num_traces_roi_sig == num_traces_neuropil_sig) or not (
                        num_traces_roi_ct == num_traces_neuropil_ct):
                    logger.info('Neuropil and ROI traces are not aligned')
                else:
                    for n in range(num_traces_roi_sig):
                        trace_roi_sig = self.plane2_traces_orig[0][n]
                        trace_neuropil_sig = self.plane2_neuropil_orig[0][n]
                        trace_roi_ct = self.plane2_traces_orig[1][n]
                        trace_neuropil_ct = self.plane2_neuropil_orig[1][n]

                        if np.any(np.isnan(trace_roi_sig)) or np.any(np.isnan(trace_neuropil_sig)) or np.any(
                                np.isnan(trace_roi_ct)) or np.any(np.isnan(trace_neuropil_ct)):
                            plane2_sig_trace_valid[str(self.plane2_roi_names[n])] = False
                            plane2_sig_neuropil_valid[str(self.plane2_roi_names[n])] = False
                            plane2_ct_trace_valid[str(self.plane2_roi_names[n])] = False
                            plane2_ct_neuropil_valid[str(self.plane2_roi_names[n])] = False
                        else:
                            plane2_sig_trace_valid[str(self.plane2_roi_names[n])] = True
                            plane2_sig_neuropil_valid[str(self.plane2_roi_names[n])] = True
                            plane2_ct_trace_valid[str(self.plane2_roi_names[n])] = True
                            plane2_ct_neuropil_valid[str(self.plane2_roi_names[n])] = True

                # combining dictionaries for signal and crosstalk
                plane1_roi_traces_valid = {"signal": plane1_sig_trace_valid,
                                           "crosstalk": plane1_ct_trace_valid}
                plane2_roi_traces_valid = {"signal": plane2_sig_trace_valid,
                                           "crosstalk": plane2_ct_trace_valid}

                plane1_neuropil_traces_valid = {"signal": plane1_sig_neuropil_valid,
                                                "crosstalk": plane1_ct_neuropil_valid}
                plane2_neuropil_traces_valid = {"signal": plane2_sig_neuropil_valid,
                                                "crosstalk": plane2_ct_neuropil_valid}
                # saving to json:

                self.plane1_roi_traces_valid_pointer = plane1_roi_traces_valid_pointer
                ju.write(plane1_roi_traces_valid_pointer, plane1_roi_traces_valid)
                self.plane1_roi_traces_valid = plane1_roi_traces_valid

                self.plane2_roi_traces_valid_pointer = plane2_roi_traces_valid_pointer
                ju.write(plane2_roi_traces_valid_pointer, plane2_roi_traces_valid)
                self.plane2_roi_traces_valid = plane2_roi_traces_valid

                self.plane1_neuropil_traces_valid_pointer = plane1_neuropil_traces_valid_pointer
                ju.write(plane1_neuropil_traces_valid_pointer, plane1_neuropil_traces_valid)
                self.plane1_neuropil_traces_valid = plane1_neuropil_traces_valid

                self.plane2_neuropil_traces_valid_pointer = plane2_neuropil_traces_valid_pointer
                ju.write(plane2_neuropil_traces_valid_pointer, plane2_neuropil_traces_valid)
                self.plane2_neuropil_traces_valid = plane2_neuropil_traces_valid

            else:
                logger.info('ROI traces dont exist in cache, run get_ica_traces first')
        else:
            # traces has been validated

            # read the jsons for ROIs
            plane1_roi_traces_valid = ju.read(plane1_roi_traces_valid_pointer)
            plane2_roi_traces_valid = ju.read(plane2_roi_traces_valid_pointer)
            self.plane1_roi_traces_valid = plane1_roi_traces_valid
            self.plane2_roi_traces_valid = plane2_roi_traces_valid

            # read the jsons for neuropil
            plane1_neuropil_traces_valid = ju.read(plane1_neuropil_traces_valid_pointer)
            plane2_neuropil_traces_valid = ju.read(plane2_neuropil_traces_valid_pointer)
            self.plane1_neuropil_traces_valid = plane1_neuropil_traces_valid
            self.plane2_neuropil_traces_valid = plane2_neuropil_traces_valid

        return

    def combine_debias_traces(self):

        self.plane1_ica_input_pointer = None
        self.plane2_ica_input_pointer = None

        plane1_ica_input_pointer = os.path.join(self.ica_traces_dir, f'traces_ica_input_{self.plane1_exp_id}.h5')
        plane2_ica_input_pointer = os.path.join(self.ica_traces_dir, f'traces_ica_input_{self.plane2_exp_id}.h5')

        if os.path.isfile(plane1_ica_input_pointer) and os.path.isfile(plane2_ica_input_pointer):
            # file already exists, skip debiasing
            self.plane1_ica_input_pointer = plane1_ica_input_pointer
            self.plane2_ica_input_pointer = plane2_ica_input_pointer
        else:
            self.plane1_ica_input_pointer = None
            self.plane2_ica_input_pointer = None

        # original traces exist, run debiasing:
        if self.found_original_traces[0] and self.found_original_traces[1]:
            # if debiased traces don't exist, run debiasing - pointers are both None
            if (not self.plane1_ica_input_pointer) and (not self.plane2_ica_input_pointer):
                self.found_ica_input = [False, False]
                self.found_ica_offset = [False, False]
                logger.info("Debiased ROI traces do not exist in cache, running offset subtraction")

                plane1_sig = self.plane1_traces_orig[0]
                plane1_ct = self.plane1_traces_orig[1]
                plane1_valid = self.plane1_roi_traces_valid

                plane1_valid_sig = plane1_valid['signal']
                plane1_valid_ct = plane1_valid['crosstalk']

                plane2_sig = self.plane2_traces_orig[0]
                plane2_ct = self.plane2_traces_orig[1]

                plane2_valid = self.plane2_roi_traces_valid
                plane2_valid_sig = plane2_valid['signal']
                plane2_valid_ct = plane2_valid['crosstalk']

                # only include cells that don't have nans  (valid = True)
                # check if traces aligned:
                if len(self.plane1_roi_names) == len(plane1_sig):
                    plane1_sig_valid_idx = np.array([plane1_valid_sig[str(tid)] for tid in self.plane1_roi_names])
                    plane1_sig_valid = plane1_sig[plane1_sig_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.plane1_roi_names) == len(plane1_ct):
                    plane1_ct_valid_idx = np.array([plane1_valid_ct[str(tid)] for tid in self.plane1_roi_names])
                    plane1_ct_valid = plane1_ct[plane1_ct_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.plane2_roi_names) == len(plane2_sig):
                    plane2_sig_valid_idx = np.array([plane2_valid_sig[str(tid)] for tid in self.plane2_roi_names])
                    plane2_sig_valid = plane2_sig[plane2_sig_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.plane2_roi_names) == len(plane2_ct):
                    plane2_ct_valid_idx = np.array([plane2_valid_ct[str(tid)] for tid in self.plane2_roi_names])
                    plane2_ct_valid = plane2_ct[plane2_ct_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                plane1_sig = plane1_sig_valid
                plane1_ct = plane1_ct_valid
                plane2_sig = plane2_sig_valid
                plane2_ct = plane2_ct_valid

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
                # check if traces aren't none
                if (not plane1_sig_m0.any() is None) and (not plane1_ct_m0.any() is None):
                    self.found_ica_input[0] = True
                if (not plane2_sig_m0.any() is None) and (not plane2_ct_m0.any() is None):
                    self.found_ica_input[1] = True
                if (not plane1_sig_offset.any() is None) and (not plane1_ct_offset.any() is None):
                    self.found_ica_offset[1] = True
                if (not plane2_sig_offset.any() is None) and (not plane2_ct_offset.any() is None):
                    self.found_ica_offset[1] = True
                # if all flags true, combine, flatten, write to disk:
                if self.found_ica_input and self.found_ica_offset:
                    self.plane1_offset = {'plane1_sig_offset': plane1_sig_offset, 'plane1_ct_offset': plane1_ct_offset}
                    self.plane2_offset = {'plane2_sig_offset': plane2_sig_offset, 'plane2_ct_offset': plane2_ct_offset}
                    trace_sig_p1 = plane1_sig_m0.flatten()
                    trace_ct_p1 = plane1_ct_m0.flatten()
                    trace_sig_p2 = plane2_sig_m0.flatten()
                    trace_ct_p2 = plane2_ct_m0.flatten()
                    plane1_ica_input = np.append(trace_sig_p1, trace_ct_p2, axis=0)
                    plane2_ica_input = np.append(trace_ct_p1, trace_sig_p2, axis=0)
                    self.plane1_ica_input = plane1_ica_input
                    self.plane2_ica_input = plane2_ica_input
                    self.plane1_ica_input_pointer = plane1_ica_input_pointer
                    self.plane2_ica_input_pointer = plane2_ica_input_pointer
                    # write ica input traces to disk
                    with h5py.File(self.plane1_ica_input_pointer, "w") as f:
                        f.create_dataset("debiased_traces", data=self.plane1_ica_input)
                        f.create_dataset('sig_offset', data=plane1_sig_offset)
                        f.create_dataset('ct_offset', data=plane1_ct_offset)
                    with h5py.File(self.plane2_ica_input_pointer, "w") as f:
                        f.create_dataset("debiased_traces", data=self.plane2_ica_input)
                        f.create_dataset('sig_offset', data=plane2_sig_offset)
                        f.create_dataset('ct_offset', data=plane2_ct_offset)
                else:
                    logger.info("ROI traces Debiasing failed")
            else:
                logger.info("Debiased ROI traces exist in cache, reading from h5 file")
                with h5py.File(self.plane1_ica_input_pointer, "r") as f:
                    plane1_ica_input = f["debiased_traces"].value
                    plane1_sig_offset = f['sig_offset'].value
                    plane1_ct_offset = f['ct_offset'].value
                with h5py.File(self.plane2_ica_input_pointer, "r") as f:
                    plane2_ica_input = f["debiased_traces"].value
                    plane2_sig_offset = f['sig_offset'].value
                    plane2_ct_offset = f['ct_offset'].value

                self.found_ica_input = [True, True]                
                self.plane1_ica_input = plane1_ica_input
                self.plane2_ica_input = plane2_ica_input
                self.plane1_ica_input_pointer = plane1_ica_input_pointer
                self.plane2_ica_input_pointer = plane2_ica_input_pointer
                self.plane1_offset = {'plane1_sig_offset': plane1_sig_offset, 'plane1_ct_offset': plane1_ct_offset}
                self.plane2_offset = {'plane2_sig_offset': plane2_sig_offset, 'plane2_ct_offset': plane2_ct_offset}
        else:
            logger.error('Extract ROI traces first')
        return

    def combine_debias_neuropil(self):

        self.plane1_ica_neuropil_input_pointer = None
        self.plane2_ica_neuropil_input_pointer = None

        plane1_ica_neuropil_input_pointer = os.path.join(self.ica_neuropil_dir, f'neuropil_ica_input_{self.plane1_exp_id}.h5')
        plane2_ica_neuropil_input_pointer = os.path.join(self.ica_neuropil_dir, f'neuropil_ica_input_{self.plane2_exp_id}.h5')

        if os.path.isfile(plane1_ica_neuropil_input_pointer) and os.path.isfile(plane2_ica_neuropil_input_pointer):
            # file already exists, skip debiasing
            self.plane1_ica_neuropil_input_pointer = plane1_ica_neuropil_input_pointer
            self.plane2_ica_neuropil_input_pointer = plane2_ica_neuropil_input_pointer
        else:
            self.plane1_ica_neuropil_input_pointer = None
            self.plane2_ica_neuropil_input_pointer = None

        # original traces exist, run debiasing:
        if self.found_original_neuropil[0] and self.found_original_neuropil[1]:
            # if debiased traces don't exist, run debiasing - pointers are both None
            if (self.plane1_ica_neuropil_input_pointer is None) and (self.plane2_ica_neuropil_input_pointer is None):
                self.found_ica_neuropil_input = [False, False]
                self.found_ica_neuropil_offset = [False, False]
                logger.info("Debiased neuropil traces do not exist in cache, running offset subtraction")
                plane1_sig = self.plane1_neuropil_orig[0]
                plane1_ct = self.plane1_neuropil_orig[1]

                plane1_valid = self.plane1_neuropil_traces_valid
                plane1_valid_sig = plane1_valid['signal']
                plane1_valid_ct = plane1_valid['crosstalk']

                plane2_sig = self.plane2_neuropil_orig[0]
                plane2_ct = self.plane2_neuropil_orig[1]

                plane2_valid = self.plane2_neuropil_traces_valid
                plane2_valid_sig = plane2_valid['signal']
                plane2_valid_ct = plane2_valid['crosstalk']

                # only include cells that don't have nans in traces (valid = True)
                # check if traces aligned:
                if len(self.plane1_roi_names) == len(plane1_sig):
                    plane1_sig_valid_idx = np.array([plane1_valid_sig[str(tid)] for tid in self.plane1_roi_names])
                    plane1_sig_valid = plane1_sig[plane1_sig_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.plane1_roi_names) == len(plane1_ct):
                    plane1_ct_valid_idx = np.array([plane1_valid_ct[str(tid)] for tid in self.plane1_roi_names])
                    plane1_ct_valid = plane1_ct[plane1_ct_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.plane2_roi_names) == len(plane2_sig):
                    plane2_sig_valid_idx = np.array([plane2_valid_sig[str(tid)] for tid in self.plane2_roi_names])
                    plane2_sig_valid = plane2_sig[plane2_sig_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                if len(self.plane2_roi_names) == len(plane2_ct):
                    plane2_ct_valid_idx = np.array([plane2_valid_ct[str(tid)] for tid in self.plane2_roi_names])
                    plane2_ct_valid = plane2_ct[plane2_ct_valid_idx, :]
                else:
                    logging.info('Traces are not aligned')

                plane1_sig = plane1_sig_valid
                plane1_ct = plane1_ct_valid
                plane2_sig = plane2_sig_valid
                plane2_ct = plane2_ct_valid

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
                # check in traces aren't none
                if (not plane1_sig_m0.any() is None) and (not plane1_ct_m0.any() is None):
                    self.found_ica_neuropil_input[0] = True
                if (not plane2_sig_m0.any() is None) and (not plane2_ct_m0.any() is None):
                    self.found_ica_neuropil_input[1] = True
                if (not plane1_sig_offset.any() is None) and (not plane1_ct_offset.any() is None):
                    self.found_ica_neuropil_offset[1] = True
                if (not plane2_sig_offset.any() is None) and (not plane2_ct_offset.any() is None):
                    self.found_ica_neuropil_offset[1] = True

                if self.found_ica_neuropil_input and self.found_ica_neuropil_offset:
                    self.plane1_neuropil_offset = {'plane1_sig_neuropil_offset': plane1_sig_offset, 'plane1_ct_neuropil_offset': plane1_ct_offset}
                    self.plane2_neuropil_offset = {'plane2_sig_neuropil_offset': plane2_sig_offset, 'plane2_ct_neuropil_offset': plane2_ct_offset}
                    neuropil_sig_p1 = plane1_sig_m0.flatten()
                    neuropil_ct_p1 = plane1_ct_m0.flatten()
                    neuropil_sig_p2 = plane2_sig_m0.flatten()
                    neuropil_ct_p2 = plane2_ct_m0.flatten()
                    plane1_ica_neuropil_input = np.append(neuropil_sig_p1, neuropil_ct_p2, axis=0)
                    plane2_ica_neuropil_input = np.append(neuropil_ct_p1, neuropil_sig_p2, axis=0)
                    self.plane1_ica_neuropil_input = plane1_ica_neuropil_input
                    self.plane2_ica_neuropil_input = plane2_ica_neuropil_input
                    self.plane1_ica_neuropil_input_pointer = plane1_ica_neuropil_input_pointer
                    self.plane2_ica_neuropil_input_pointer = plane2_ica_neuropil_input_pointer
                    # write ica neuropil input traces to disk
                    if not os.path.isfile(self.plane1_ica_neuropil_input_pointer):
                        with h5py.File(self.plane1_ica_neuropil_input_pointer, "w") as f:
                            f.create_dataset("debiased_traces", data=self.plane1_ica_neuropil_input)
                            f.create_dataset('sig_offset', data=plane1_sig_offset)
                            f.create_dataset('ct_offset', data=plane1_ct_offset)

                    if not os.path.isfile(self.plane2_ica_neuropil_input_pointer):
                        with h5py.File(self.plane2_ica_neuropil_input_pointer, "w") as f:
                            f.create_dataset("debiased_traces", data=self.plane2_ica_neuropil_input)
                            f.create_dataset('sig_offset', data=plane2_sig_offset)
                            f.create_dataset('ct_offset', data=plane2_ct_offset)
                else:
                    logger.info("Neuropil debiasing failed")
            else:
                logger.info("Debiased neuropil traces exist in cache, reading from h5 file")
                self.found_ica_neuropil_input = [True, True]
                self.plane1_ica_neuropil_input_pointer = plane1_ica_neuropil_input_pointer
                self.plane2_ica_neuropil_input_pointer = plane2_ica_neuropil_input_pointer
                with h5py.File(self.plane1_ica_neuropil_input_pointer, "r") as f:
                    plane1_ica_neuropil_input = f["debiased_traces"].value
                    plane1_sig_neuropil_offset = f['sig_offset'].value
                    plane1_ct_neuropil_offset = f['ct_offset'].value
                with h5py.File(self.plane2_ica_neuropil_input_pointer, "r") as f:
                    plane2_ica_neuropil_input = f["debiased_traces"].value
                    plane2_sig_neuropil_offset = f['sig_offset'].value
                    plane2_ct_neuropil_offset = f['ct_offset'].value
                self.plane1_ica_neuropil_input = plane1_ica_neuropil_input
                self.plane2_ica_neuropil_input = plane2_ica_neuropil_input
                self.plane1_neuropil_offset = {'plane1_sig_neuropil_offset': plane1_sig_neuropil_offset, 'plane1_ct_neuropil_offset': plane1_ct_neuropil_offset}
                self.plane2_neuropil_offset = {'plane2_sig_neuropil_offset': plane2_sig_neuropil_offset, 'plane2_ct_neuropil_offset': plane2_ct_neuropil_offset, }
        else:
            logger.error('Extract neuropil traces first')
        return

    def unmix_traces(self, max_iter=50):

        plane1_ica_output_pointer = os.path.join(self.ica_traces_dir,
                                                 f'traces_ica_output_{self.plane1_exp_id}.h5')
        plane2_ica_output_pointer = os.path.join(self.ica_traces_dir,
                                                 f'traces_ica_output_{self.plane2_exp_id}.h5')
        ica_mixing_matrix_traces_pointer = os.path.join(self.ica_traces_dir,
                                                        f'traces_ica_mixing.h5')
        # file already exists, skip unmixing
        if os.path.isfile(plane1_ica_output_pointer) and os.path.isfile(plane2_ica_output_pointer) and os.path.isfile(
                ica_mixing_matrix_traces_pointer):
            self.plane1_ica_output_pointer = plane1_ica_output_pointer
            self.plane2_ica_output_pointer = plane2_ica_output_pointer
            self.ica_mixing_matrix_traces_pointer = ica_mixing_matrix_traces_pointer
        else:
            self.plane1_ica_output_pointer = None
            self.plane2_ica_output_pointer = None
            self.ica_mixing_matrix_traces_pointer = None

        if (self.plane1_ica_output_pointer is None) or (self.plane2_ica_output_pointer is None):
            # if unmixed traces don't exist, run unmixing
            if np.any(np.isnan(self.plane1_ica_input)) or np.any(np.isinf(self.plane1_ica_input)) or np.any(
                    np.isnan(self.plane2_ica_input)) or np.any(np.isinf(self.plane2_ica_input)):
                logger.info("ValueError: ICA input contains NaN, infinity or a value too large for dtype('float64')")
            else:
                logger.info("unmixed traces do not exist in cache, running ICA")
                traces = np.array([self.plane1_ica_input, self.plane2_ica_input]).T
                self.found_solution = False
                for i in range(max_iter):
                    ica = FastICA(n_components=2)
                    s = ica.fit_transform(traces)  # Reconstruct signals
                    a = ica.mixing_  # Get estimated mixing matrix
                    if (np.all(a > 0)) & (a[0][0] > a[1][0]):
                        self.found_solution = True
                        logger.info("ICA successful")
                        self.traces_matrix = a
                        self.traces_unmix = s
                        break
                if not self.found_solution:
                    logger.error("Failed to find solution, try increasing `max_iter`")

            if self.found_solution:
                # rescaling traces back:
                self.ica_traces_scale_top, self.ica_traces_scale_bot = self.find_scale_ica_traces()

                plane1_ica_output = self.traces_unmix[:, 0] * self.ica_traces_scale_top
                plane2_ica_output = self.traces_unmix[:, 1] * self.ica_traces_scale_bot

                # reshaping traces
                plane1_new_shape = [int(plane1_ica_output.shape[0] / self.plane1_traces_orig.shape[2]),
                                    self.plane1_traces_orig.shape[2]]
                plane1_ica_output = plane1_ica_output.reshape(plane1_new_shape)

                plane2_new_shape = [int(plane2_ica_output.shape[0] / self.plane2_traces_orig.shape[2]),
                                    self.plane2_traces_orig.shape[2]]
                plane2_ica_output = plane2_ica_output.reshape(plane2_new_shape)

                #
                plane1_valid_shape = np.array(
                    [self.plane1_roi_traces_valid['signal'][str(tid)] for tid in self.plane1_roi_names])
                plane1_out_sig = plane1_ica_output[0:plane1_valid_shape.sum(), :]
                plane1_out_ct = plane2_ica_output[0:plane1_valid_shape.sum(), :]

                plane2_valid_shape = np.array(
                    [self.plane2_roi_traces_valid['signal'][str(tid)] for tid in self.plane2_roi_names])
                plane2_out_ct = plane1_ica_output[
                                plane1_valid_shape.sum():plane1_valid_shape.sum() + plane2_valid_shape.sum(), :]
                plane2_out_sig = plane2_ica_output[
                                 plane1_valid_shape.sum():plane1_valid_shape.sum() + plane2_valid_shape.sum(), :]

                # adding offset
                plane1_out_sig = plane1_out_sig + self.plane1_offset['plane1_sig_offset']
                plane1_out_ct = plane1_out_ct + self.plane1_offset['plane1_ct_offset']

                plane2_out_sig = plane2_out_sig + self.plane2_offset['plane2_sig_offset']
                plane2_out_ct = plane2_out_ct + self.plane2_offset['plane2_ct_offset']

                plane1_ica_output = np.array([plane1_out_sig, plane1_out_ct])
                plane2_ica_output = np.array([plane2_out_sig, plane2_out_ct])

                self.plane1_ica_output = plane1_ica_output
                self.plane2_ica_output = plane2_ica_output

                self.plane1_ica_output_pointer = plane1_ica_output_pointer
                self.plane2_ica_output_pointer = plane2_ica_output_pointer
                self.ica_mixing_matrix_traces_pointer = ica_mixing_matrix_traces_pointer

                # writing ica output traces to disk
                with h5py.File(self.plane1_ica_output_pointer, "w") as f:
                    f.create_dataset(f"data", data=plane1_ica_output)

                with h5py.File(self.plane2_ica_output_pointer, "w") as f:
                    f.create_dataset(f"data", data=plane2_ica_output)

                with h5py.File(self.ica_mixing_matrix_traces_pointer, "w") as f:
                    f.create_dataset(f"data", data=self.traces_matrix)
        else:
            logger.info("Unmixed traces exist in cache, reading from h5 file")
            self.plane1_ica_output_pointer = plane1_ica_output_pointer
            self.plane2_ica_output_pointer = plane2_ica_output_pointer
            self.ica_mixing_matrix_traces_pointer = ica_mixing_matrix_traces_pointer
            self.found_solution = True
            with h5py.File(self.plane1_ica_output_pointer, "r") as f:
                plane1_ica_output = f["data"].value
            with h5py.File(self.plane2_ica_output_pointer, "r") as f:
                plane2_ica_output = f["data"].value
            with h5py.File(self.ica_mixing_matrix_traces_pointer, "r") as f:
                traces_matrix = f["data"].value

            self.plane1_ica_output = plane1_ica_output
            self.plane2_ica_output = plane2_ica_output
            self.traces_matrix = traces_matrix

        return

    def unmix_neuropil(self, max_iter=10):

        plane1_ica_neuropil_output_pointer = os.path.join(self.ica_neuropil_dir,
                                                          f'neuropil_ica_output_{self.plane1_exp_id}.h5')
        plane2_ica_neuropil_output_pointer = os.path.join(self.ica_neuropil_dir,
                                                          f'neuropil_ica_output_{self.plane2_exp_id}.h5')
        ica_mixing_matrix_neuropil_pointer = os.path.join(self.ica_neuropil_dir,
                                                          f'neuropil_ica_mixing.h5')

        # file already exists, skip unmixing
        if os.path.isfile(plane1_ica_neuropil_output_pointer) and os.path.isfile(
                plane2_ica_neuropil_output_pointer) and os.path.isfile(ica_mixing_matrix_neuropil_pointer):
            self.plane1_ica_neuropil_output_pointer = plane1_ica_neuropil_output_pointer
            self.plane2_ica_neuropil_output_pointer = plane2_ica_neuropil_output_pointer
            self.ica_mixing_matrix_neuropil_pointer = ica_mixing_matrix_neuropil_pointer
        else:
            self.plane1_ica_neuropil_output_pointer = None
            self.plane2_ica_neuropil_output_pointer = None
            self.ica_mixing_matrix_neuropil_pointer = None

        if (self.plane1_ica_neuropil_output_pointer is None) or (self.plane2_ica_neuropil_output_pointer is None):
            # if unmixed traces don't exist, run unmixing
            logger.info("Unmixed neuropil traces do not exist in cache, running ICA")

            if np.any(np.isnan(self.plane1_ica_neuropil_input)) or np.any(
                    np.isinf(self.plane1_ica_neuropil_input)) or np.any(
                np.isnan(self.plane2_ica_neuropil_input)) or np.any(np.isinf(self.plane2_ica_neuropil_input)):
                logger.info("ValueError: ICA input contains NaN, infinity or a value too large for dtype('float64')")

            else:
                traces = np.array([self.plane1_ica_neuropil_input, self.plane2_ica_neuropil_input]).T
                self.found_solution_neuropil = False
                for i in range(max_iter):
                    ica = FastICA(n_components=2)
                    s = ica.fit_transform(traces)  # Reconstruct signals
                    a = ica.mixing_  # Get estimated mixing matrix
                    if (np.all(a > 0)) & (a[0][0] > a[1][0]):
                        self.found_solution_neuropil = True
                        logger.info("ICA successful")
                        self.neuropil_matrix = a
                        self.neuropil_unmix = s
                        break
                if not self.found_solution_neuropil:
                    logger.error("Failed to find solution, try increasing `max_iter`")

                if self.found_solution_neuropil:
                    # rescaling traces back:
                    self.ica_neuropil_scale_top, self.ica_neuropil_scale_bot = self.find_scale_ica_neuropil()

                    plane1_ica_neuropil_output = self.neuropil_unmix[:, 0] * self.ica_neuropil_scale_top
                    plane2_ica_neuropil_output = self.neuropil_unmix[:, 1] * self.ica_neuropil_scale_bot

                    # reshaping traces
                    plane1_new_shape = [int(plane1_ica_neuropil_output.shape[0] / self.plane1_neuropil_orig.shape[2]),
                                        self.plane1_neuropil_orig.shape[2]]
                    plane1_ica_output = plane1_ica_neuropil_output.reshape(plane1_new_shape)

                    plane2_new_shape = [int(plane2_ica_neuropil_output.shape[0] / self.plane2_neuropil_orig.shape[2]),
                                        self.plane2_neuropil_orig.shape[2]]
                    plane2_ica_output = plane2_ica_neuropil_output.reshape(plane2_new_shape)

                    #
                    plane1_valid_shape = np.array(
                        [self.plane1_neuropil_traces_valid['signal'][str(tid)] for tid in self.plane1_roi_names])
                    plane1_out_sig = plane1_ica_output[0:plane1_valid_shape.sum(), :]
                    plane1_out_ct = plane2_ica_output[0:plane1_valid_shape.sum(), :]

                    plane2_valid_shape = np.array(
                        [self.plane2_neuropil_traces_valid['signal'][str(tid)] for tid in self.plane2_roi_names])
                    plane2_out_ct = plane1_ica_output[
                                    plane1_valid_shape.sum():plane1_valid_shape.sum() + plane2_valid_shape.sum(), :]
                    plane2_out_sig = plane2_ica_output[
                                     plane1_valid_shape.sum():plane1_valid_shape.sum() + plane2_valid_shape.sum(), :]

                    # adding offset
                    plane1_out_sig = plane1_out_sig + self.plane1_neuropil_offset['plane1_sig_neuropil_offset']
                    plane1_out_ct = plane1_out_ct + self.plane1_neuropil_offset['plane1_ct_neuropil_offset']

                    plane2_out_sig = plane2_out_sig + self.plane2_neuropil_offset['plane2_sig_neuropil_offset']
                    plane2_out_ct = plane2_out_ct + self.plane2_neuropil_offset['plane2_ct_neuropil_offset']

                    plane1_ica_neuropil_output = np.array([plane1_out_sig, plane1_out_ct])
                    plane2_ica_neuropil_output = np.array([plane2_out_sig, plane2_out_ct])

                    self.plane1_ica_neuropil_output = plane1_ica_neuropil_output
                    self.plane2_ica_neuropil_output = plane2_ica_neuropil_output
                    self.plane1_ica_neuropil_output_pointer = plane1_ica_neuropil_output_pointer
                    self.plane2_ica_neuropil_output_pointer = plane2_ica_neuropil_output_pointer
                    self.ica_mixing_matrix_neuropil_pointer = ica_mixing_matrix_neuropil_pointer
                    # writing ica output traces to disk
                    with h5py.File(self.plane1_ica_neuropil_output_pointer, "w") as f:
                        f.create_dataset(f"data", data=plane1_ica_neuropil_output)
                    with h5py.File(self.plane2_ica_neuropil_output_pointer, "w") as f:
                        f.create_dataset(f"data", data=plane2_ica_neuropil_output)
                    with h5py.File(self.ica_mixing_matrix_neuropil_pointer, "w") as f:
                        f.create_dataset(f"data", data=self.neuropil_matrix)
        else:
            logger.info("Unmixed neuropil traces exist in cache, reading from h5 file")

            self.found_solution_neuropil = True

            self.plane1_ica_neuropil_output_pointer = plane1_ica_neuropil_output_pointer
            self.plane2_ica_neuropil_output_pointer = plane2_ica_neuropil_output_pointer
            self.ica_mixing_matrix_neuropil_pointer = ica_mixing_matrix_neuropil_pointer

            with h5py.File(self.plane1_ica_neuropil_output_pointer, "r") as f:
                plane1_ica_neuropil_output = f["data"].value

            with h5py.File(self.plane2_ica_neuropil_output_pointer, "r") as f:
                plane2_ica_neuropil_output = f["data"].value

            with h5py.File(self.ica_mixing_matrix_neuropil_pointer, "r") as f:
                neuropil_matrix = f["data"].value

            self.plane1_ica_neuropil_output = plane1_ica_neuropil_output
            self.plane2_ica_neuropil_output = plane2_ica_neuropil_output
            self.neuropil_matrix = neuropil_matrix

        return

    def plot_ica_traces(self, pair):
        #    if figures don't exist!

        if self.plane1_ica_output_pointer and self.plane2_ica_output_pointer:

            orig_trace_plane1_sig = self.plane1_traces_orig[0, :, :]
            orig_trace_plane1_ct = self.plane1_traces_orig[1, :, :]
            plane1_roi_names = self.plane1_roi_names
            plane1_roi_valid = self.plane1_roi_traces_valid['signal']
            ica_trace_plane1_sig = self.plane1_ica_output[0, :, :]
            ica_trace_plane1_ct = self.plane1_ica_output[1, :, :]

            logging.info(f'creating figures for experiment {pair[0]}')

            plot_dir = os.path.join(self.session_cache_dir, f'ica_traces_{pair[0]}_{pair[1]}/ica_plots_{pair[0]}')
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)

            cell_valid = 0
            for cell_orig in range(orig_trace_plane1_sig.shape[0]):
                # check in this roi is valid:
                if plane1_roi_valid[str(plane1_roi_names[cell_orig])]:
                    # Plot cell
                    pdf_name = os.path.join(plot_dir, f"ica_plots_{pair[0]}_cell_{plane1_roi_names[cell_orig]}.pdf")
                    if os.path.isfile(pdf_name):
                        logging.info(f"cell trace figure exist for {pair[0]} cell {plane1_roi_names[cell_orig]}")
                        continue
                    else:
                        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
                        logging.info(f"creating figures for cell {plane1_roi_names[cell_orig]}")
                        for i in range(int(orig_trace_plane1_sig.shape[1] / 10000) + 1):
                            orig_plane1_sig = orig_trace_plane1_sig[cell_orig, i * 10000:(i + 1) * 10000]
                            orig_plane1_ct = orig_trace_plane1_ct[cell_orig, i * 10000:(i + 1) * 10000]
                            ica_plane1_sig = ica_trace_plane1_sig[cell_valid, i * 10000:(i + 1) * 10000]
                            ica_plane1_ct = ica_trace_plane1_ct[cell_valid, i * 10000:(i + 1) * 10000]
                            f = plt.figure(figsize=(20, 10))
                            plt.subplot(211)
                            plt.plot(orig_plane1_sig, 'r-', label='signal plane')
                            plt.plot(orig_plane1_ct, 'g-', label='cross-talk plane')
                            plt.title(f'original traces for cell {plane1_roi_names[cell_orig]}')
                            plt.legend(loc='upper left')
                            plt.subplot(212)
                            plt.plot(ica_plane1_sig, 'r-', label='signal plane')
                            plt.plot(ica_plane1_ct, 'g-', label='cross-talk plane')
                            plt.title(f'post-ica traces, cell # {plane1_roi_names[cell_valid]}')
                            plt.legend(loc='upper left')
                            pdf.savefig(f)
                        pdf.close()
                        cell_valid = cell_valid + 1
                else:
                    logging.info(f'Cell {plane1_roi_names[cell_orig]} is invalid, skipping plotting')
                    cell_valid = cell_valid

            orig_trace_plane2_sig = self.plane2_traces_orig[0, :, :]
            orig_trace_plane2_ct = self.plane2_traces_orig[1, :, :]
            plane2_roi_names = self.plane2_roi_names
            plane2_roi_valid = self.plane2_roi_traces_valid['signal']
            ica_trace_plane2_sig = self.plane2_ica_output[0, :, :]
            ica_trace_plane2_ct = self.plane2_ica_output[1, :, :]
            logging.info(f'creating figures for experiment {pair[1]}')
            plot_dir = os.path.join(self.session_cache_dir, f'ica_traces_{pair[0]}_{pair[1]}/ica_plots_{pair[1]}')
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)
            cell_valid = 0
            for cell_orig in range(orig_trace_plane2_sig.shape[0]):
                # check in this roi is valid:
                if plane2_roi_valid[str(plane2_roi_names[cell_orig])]:
                    # Plot cell
                    pdf_name = os.path.join(plot_dir, f"ica_plots_{pair[1]}_cell_{plane2_roi_names[cell_orig]}.pdf")
                    if os.path.isfile(pdf_name):
                        logging.info(f"cell trace figure exist for {pair[1]} cell {plane2_roi_names[cell_orig]}")
                        continue
                    else:
                        logging.info(f'creating figures for cell {plane2_roi_names[cell_orig]}')
                        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
                        for i in range(int(orig_trace_plane2_sig.shape[1] / 10000) + 1):
                            orig_plane2_sig = orig_trace_plane2_sig[cell_orig, i * 10000:(i + 1) * 10000]
                            orig_plane2_ct = orig_trace_plane2_ct[cell_orig, i * 10000:(i + 1) * 10000]
                            ica_plane2_sig = ica_trace_plane2_sig[cell_valid, i * 10000:(i + 1) * 10000]
                            ica_plane2_ct = ica_trace_plane2_ct[cell_valid, i * 10000:(i + 1) * 10000]
                            f = plt.figure(figsize=(20, 10))
                            plt.subplot(211)
                            plt.plot(orig_plane2_sig, 'r-', label='signal plane')
                            plt.plot(orig_plane2_ct, 'g-', label='cross-talk plane')
                            plt.title(f'original traces for cell # {plane2_roi_names[cell_orig]}')
                            plt.legend(loc='upper left')
                            plt.subplot(212)
                            plt.plot(ica_plane2_sig, 'r-', label='signal plane')
                            plt.plot(ica_plane2_ct, 'g-', label='cross-talk plane')
                            plt.title(f'post-ica traces, cell # {plane2_roi_names[cell_valid]}')
                            plt.legend(loc='upper left')
                            pdf.savefig(f)
                            plt.close()
                        pdf.close()
                    cell_valid = cell_valid + 1

                else:
                    logging.info(f'Cell {plane2_roi_names[cell_orig]} is invalid, skipping plotting')
                    cell_valid = cell_valid
        else:
            logging.info(f'ICA traces for pair {pair[0]}/{pair[1]} don''t exist, nothing to plot.')

        return

    @staticmethod
    def ica_err(scale, traces_ica, trace_orig):
        return np.sqrt((traces_ica * scale[0] - trace_orig) ** 2).mean()

    @staticmethod
    def get_valid_seg_run(exp_id):
        query = f"""
        select *
        from ophys_cell_segmentation_runs
        where current = True and ophys_experiment_id = {exp_id}
        """
        seg_run = lu.query(query)[0]['id']
        return seg_run

    def find_scale_ica_traces(self):
        # for traces:
        scale_top = opt.minimize(self.ica_err, [1], (self.traces_unmix[:, 0], self.plane1_ica_input))
        scale_bot = opt.minimize(self.ica_err, [1], (self.traces_unmix[:, 1], self.plane2_ica_input))

        return scale_top.x, scale_bot.x

    def find_scale_ica_neuropil(self):
        scale_top_neuropil = opt.minimize(self.ica_err, [1],
                                          (self.neuropil_unmix[:, 0], self.plane1_ica_neuropil_input))
        scale_bot_neuropil = opt.minimize(self.ica_err, [1],
                                          (self.neuropil_unmix[:, 1], self.plane2_ica_neuropil_input))

        return scale_top_neuropil.x, scale_bot_neuropil.x

