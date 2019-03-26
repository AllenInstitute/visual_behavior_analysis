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
import matplotlib


matplotlib.use('Agg')

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

    traces, neuropil_traces = roi_masks.calculate_roi_and_neuropil_traces(movie_h5, roi_mask_list, motion_border)

    return traces, neuropil_traces


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
        self.plane1_offset_neuropil = None
        self.plane2_offset_neuropil = None

        self.session_id = session_id
        self.dataset = ms.MesoscopeDataset(session_id)
        self.session_cache_dir = None

        self.plane1_exp_id = None
        self.plane2_exp_id = None

        self.ica_traces_dir = None
        self.ica_neuropil_dir = None
        self.found_ica_traces = None  # output of get_traces
        self.found_ica_neuropil = None  # output of get_traces

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

        self.traces_matrix = None
        self.neuropil_matrix = None

        self.traces_unmix = None
        self.neuropil_unmix = None

        self.cache = cache

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
        return

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
            logger.info('Traces dont exist in cache, extracting')

            self.plane1_ica_input_pointer = None
            self.plane2_ica_input_pointer = None
            self.plane1_ica_output_pointer = None
            self.plane2_ica_output_pointer = None

            plane1_folder = self.dataset.get_exp_folder(plane1_exp_id)
            plane2_folder = self.dataset.get_exp_folder(plane2_exp_id)

            # extract signal and crosstalk traces for plane 1
            plane1_sig_traces, _ = get_traces(plane1_folder, plane1_exp_id, plane1_folder, plane1_exp_id)
            plane1_ct_traces, _ = get_traces(plane2_folder, plane2_exp_id, plane1_folder, plane1_exp_id)

            # extract signal and crosstalk traces for plane 2
            plane2_sig_traces, _ = get_traces(plane2_folder, plane2_exp_id, plane2_folder, plane2_exp_id)
            plane2_ct_traces, _ = get_traces(plane1_folder, plane1_exp_id, plane2_folder, plane2_exp_id)

            if (not plane2_sig_traces.any() is None) and (not plane2_ct_traces.any() is None):
                self.found_ica_traces[0] = True
            if (not plane2_sig_traces.any() is None) and (not plane2_ct_traces.any() is None):
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
                        f.create_dataset(f"data", data=plane1_traces_original)

            if self.found_ica_traces[1]:
                if not os.path.isfile(path_traces_plane2):
                    with h5py.File(path_traces_plane2, "w") as f:
                        f.create_dataset(f"data", data=plane2_traces_original)
        else:
            logger.info('Found traces in cache, reading from h5 file')
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

            logger.info("debiased traces do not exist in cache, running offset subtraction")

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
            logger.info("Debiased traces exist in cache, reading from h5 file")

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

    def combine_debias_neuropil(self):

        plane1_ica_neuropil_input_pointer = os.path.join(self.ica_neuropil_dir,
                                                         f'neuropil_ica_input_{self.plane1_exp_id}.h5')

        if os.path.isfile(plane1_ica_neuropil_input_pointer):
            self.plane1_ica_neuropil_input_pointer = plane1_ica_neuropil_input_pointer
            # file already exists, skip debiasing

        plane2_ica_neuropil_input_pointer = os.path.join(self.ica_neuropil_dir,
                                                         f'neuropil_ica_input_{self.plane2_exp_id}.h5')

        if os.path.isfile(plane2_ica_neuropil_input_pointer):
            self.plane2_ica_neuropil_input_pointer = plane2_ica_neuropil_input_pointer
            # file already exists, skip debiasing

        if not (self.plane1_ica_neuropil_input_pointer and self.plane2_ica_neuropil_input_pointer):

            self.plane1_ica_neuropil_input_pointer = plane1_ica_neuropil_input_pointer
            self.plane2_ica_neuropil_input_pointer = plane2_ica_neuropil_input_pointer

            logger.info("debiased neuropil traces do not exist in cache, running offset subtraction")

            if self.found_ica_neuropil:

                plane1_sig = self.plane1_neuropil_orig[0]
                plane1_ct = self.plane1_neuropil_orig[1]

                plane2_sig = self.plane2_neuropil_orig[0]
                plane2_ct = self.plane2_neuropil_orig[1]

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

                self.plane1_offset_neuropil = {'plane1_sig_offset': plane1_sig_offset,
                                               'plane1_ct_offset': plane1_ct_offset}
                self.plane2_offset_neuropil = {'plane2_sig_offset': plane2_sig_offset,
                                               'plane2_ct_offset': plane2_ct_offset, }

                neuropil_sig_p1 = plane1_sig_m0.flatten()
                neuropil_ct_p1 = plane1_ct_m0.flatten()
                neuropil_sig_p2 = plane2_sig_m0.flatten()
                neuropil_ct_p2 = plane2_ct_m0.flatten()

                plane1_ica_neuropil_input = np.append(neuropil_sig_p1, neuropil_ct_p2, axis=0)
                plane2_ica_neuropil_input = np.append(neuropil_ct_p1, neuropil_sig_p2, axis=0)

                self.plane1_ica_neuropil_input = plane1_ica_neuropil_input
                self.plane2_ica_neuropil_input = plane2_ica_neuropil_input

                # write ica neuropil input traces to disk

                #                same for neuropil:

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
                logger.error('Extract neuropil traces first')
        else:
            logger.info("Debiased neuropil traces exist in cache, reading from h5 file")

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
            self.plane1_neuropil_offset = {'plane1_sig_neuropil_offset': plane1_sig_neuropil_offset,
                                           'plane1_ct_neuropil_offset': plane1_ct_neuropil_offset}
            self.plane2_neuropil_offset = {'plane2_sig_neuropil_offset': plane2_sig_neuropil_offset,
                                           'plane2_ct_neuropil_offset': plane2_ct_neuropil_offset, }

        return

    def unmix_traces(self, max_iter=10):

        plane1_ica_output_pointer = os.path.join(self.ica_traces_dir,
                                                 f'traces_ica_output_{self.plane1_exp_id}.h5')

        if os.path.isfile(plane1_ica_output_pointer):
            self.plane1_ica_output_pointer = plane1_ica_output_pointer
            # file already exists, skip unmixing

        plane2_ica_output_pointer = os.path.join(self.ica_traces_dir,
                                                 f'traces_ica_output_{self.plane2_exp_id}.h5')

        if os.path.isfile(plane2_ica_output_pointer):
            self.plane2_ica_output_pointer = plane2_ica_output_pointer
            # file already exists, skip unmixing

        if not (self.plane1_ica_output_pointer and self.plane2_ica_output_pointer):
            self.plane1_ica_output_pointer = plane1_ica_output_pointer
            self.plane2_ica_output_pointer = plane2_ica_output_pointer
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
                self.ica_traces_scale_top, self.ica_traces_scale_bot, _, _ = self.find_scale_ica

                plane1_ica_output = self.traces_unmix[:, 0] * self.ica_traces_scale_top
                plane2_ica_output = self.traces_unmix[:, 1] * self.ica_traces_scale_bot

                # reshaping traces
                plane1_ica_output = plane1_ica_output.reshape(
                    [self.plane1_traces_orig.shape[1] + self.plane2_traces_orig.shape[1],
                     self.plane1_traces_orig.shape[2]])
                plane2_ica_output = plane2_ica_output.reshape(
                    [self.plane1_traces_orig.shape[1] + self.plane2_traces_orig.shape[1],
                     self.plane1_traces_orig.shape[2]])

                plane1_out_sig = plane1_ica_output[0:self.plane1_traces_orig.shape[1], :]
                plane1_out_ct = plane2_ica_output[0:self.plane1_traces_orig.shape[1], :]

                plane2_out_ct = plane1_ica_output[self.plane1_traces_orig.shape[1]:plane1_ica_output.shape[0], :]
                plane2_out_sig = plane2_ica_output[self.plane1_traces_orig.shape[1]:plane1_ica_output.shape[0], :]

                # adding offset
                plane1_out_sig = plane1_out_sig + self.plane1_offset['plane1_sig_offset']
                plane1_out_ct = plane1_out_ct + self.plane1_offset['plane1_ct_offset']

                plane2_out_sig = plane2_out_sig + self.plane2_offset['plane2_sig_offset']
                plane2_out_ct = plane2_out_ct + self.plane2_offset['plane2_ct_offset']

                plane1_ica_output = np.array([plane1_out_sig, plane1_out_ct])
                plane2_ica_output = np.array([plane2_out_sig, plane2_out_ct])

                self.plane1_ica_output = plane1_ica_output
                self.plane2_ica_output = plane2_ica_output

                # writing ica output traces to disk
                with h5py.File(self.plane1_ica_output_pointer, "w") as f:
                    f.create_dataset(f"data", data=plane1_ica_output)

                with h5py.File(self.plane2_ica_output_pointer, "w") as f:
                    f.create_dataset(f"data", data=plane2_ica_output)
        else:
            logger.info("Unmixed traces exist in cache, reading from h5 file")
            self.found_solution = True
            with h5py.File(self.plane1_ica_output_pointer, "r") as f:
                plane1_ica_output = f["data"].value

            with h5py.File(self.plane2_ica_output_pointer, "r") as f:
                plane2_ica_output = f["data"].value

            self.plane1_ica_output = plane1_ica_output
            self.plane2_ica_output = plane2_ica_output
        return

    def unmix_neuropil(self, max_iter=10):

        plane1_ica_neuropil_output_pointer = os.path.join(self.ica_neuropil_dir,
                                                          f'traces_ica_output_{self.plane1_exp_id}.h5')

        if os.path.isfile(plane1_ica_neuropil_output_pointer):
            self.plane1_ica_neuropil_output_pointer = plane1_ica_neuropil_output_pointer
            # file already exists, skip unmixing

        plane2_ica_neuropil_output_pointer = os.path.join(self.ica_neuropil_dir,
                                                          f'traces_ica_output_{self.plane2_exp_id}.h5')

        if os.path.isfile(plane2_ica_neuropil_output_pointer):
            self.plane2_ica_output_pointer = plane2_ica_neuropil_output_pointer
            # file already exists, skip unmixing

        if not (self.plane1_ica_neuropil_output_pointer and self.plane2_ica_neuropil_output_pointer):
            self.plane1_ica_neuropil_output_pointer = plane1_ica_neuropil_output_pointer
            self.plane2_ica_neuropil_output_pointer = plane2_ica_neuropil_output_pointer
            logger.info("unmixed neuropil traces do not exist in cache, running ICA")
            traces = np.array([self.plane1_ica_neuropil_input, self.plane2_ica_neuropil_input]).T
            self.found_solution = False
            for i in range(max_iter):
                ica = FastICA(n_components=2)
                s = ica.fit_transform(traces)  # Reconstruct signals
                a = ica.mixing_  # Get estimated mixing matrix
                if (np.all(a > 0)) & (a[0][0] > a[1][0]):
                    self.found_solution = True
                    logger.info("ICA successful")
                    self.neuropil_matrix = a
                    self.neuropil_unmix = s
                    break
            if not self.found_solution_neuropil:
                logger.error("Failed to find solution, try increasing `max_iter`")

            if self.found_solution_neuropil:
                # rescaling traces back:
                _, _, self.ica_neuropil_scale_top, self.ica_neuropil_scale_bot = self.find_scale_ica

                plane1_ica_neuropil_output = self.neuropil_unmix[:, 0] * self.ica_neuropil_scale_top
                plane2_ica_neuropil_output = self.neuropil_unmix[:, 1] * self.ica_neuropil_scale_bot

                # reshaping traces
                plane1_ica_neuropil_output = plane1_ica_neuropil_output.reshape(
                    [self.plane1_neuropil_orig.shape[1] + self.plane2_neuropil_orig.shape[1],
                     self.plane1_neuropil_orig.shape[2]])
                plane2_ica_neuropil_output = plane2_ica_neuropil_output.reshape(
                    [self.plane1_neuropil_orig.shape[1] + self.plane2_neuropil_orig.shape[1],
                     self.plane1_neuropil_orig.shape[2]])

                plane1_out_sig = plane1_ica_neuropil_output[0:self.plane1_neuropil_orig.shape[1], :]
                plane1_out_ct = plane2_ica_neuropil_output[0:self.plane1_neuropil_orig.shape[1], :]

                plane2_out_ct = plane1_ica_neuropil_output[
                                self.plane1_neuropil_orig.shape[1]:plane1_ica_neuropil_output.shape[0], :]
                plane2_out_sig = plane2_ica_neuropil_output[
                                 self.plane1_neuropil_orig.shape[1]:plane1_ica_neuropil_output.shape[0], :]

                # adding offset
                plane1_out_sig = plane1_out_sig + self.plane1_neuropil_offset['plane1_sig_offset']
                plane1_out_ct = plane1_out_ct + self.plane1_neuropil_offset['plane1_ct_offset']

                plane2_out_sig = plane2_out_sig + self.plane2_neuropil_offset['plane2_sig_offset']
                plane2_out_ct = plane2_out_ct + self.plane2_neuropil_offset['plane2_ct_offset']

                plane1_ica_neuropil_output = np.array([plane1_out_sig, plane1_out_ct])
                plane2_ica_neuropil_output = np.array([plane2_out_sig, plane2_out_ct])

                self.plane1_ica_neuropil_output = plane1_ica_neuropil_output
                self.plane2_ica_neuropil_output = plane2_ica_neuropil_output

                # writing ica output traces to disk
                with h5py.File(self.plane1_ica_neuropil_output_pointer, "w") as f:
                    f.create_dataset(f"data", data=plane1_ica_neuropil_output)

                with h5py.File(self.plane2_ica_neuropil_output_pointer, "w") as f:
                    f.create_dataset(f"data", data=plane2_ica_neuropil_output)
        else:
            logger.info("Unmixed traces exist in cache, reading from h5 file")
            self.found_solution_neuropil = True
            with h5py.File(self.plane1_ica_neuropil_output_pointer, "r") as f:
                plane1_ica_neuropil_output = f["data"].value

            with h5py.File(self.plane2_ica_neuropil_output_pointer, "r") as f:
                plane2_ica_neuropil_output = f["data"].value

            self.plane1_ica_neuropil_output = plane1_ica_neuropil_output
            self.plane2_ica_neuropil_output = plane2_ica_neuropil_output
        return

    def plot_ica_traces(self, pair):

        if self.plane1_ica_output_pointer and self.plane2_ica_output_pointer:

            orig_trace_plane1_sig = self.plane1_traces_orig[0, :, :]
            orig_trace_plane1_ct = self.plane1_traces_orig[1, :, :]
            ica_trace_plane1_sig = self.plane1_ica_output[0, :, :]
            ica_trace_plane1_ct = self.plane1_ica_output[1, :, :]
            logging.info(f'creating figures for experiment {pair[0]}')
            for cell in range(orig_trace_plane1_sig.shape[0]):
                plot_dir = os.path.join(self.session_cache_dir, f'ica_traces_{pair[0]}_{pair[1]}/ica_plots_{pair[0]}')
                if not os.path.isdir(plot_dir):
                    os.mkdir(plot_dir)
                logging.info(f"creating figures for cell {cell}")
                pdf = matplotlib.backends.backend_pdf.PdfPages(
                    os.path.join(plot_dir, f"ica_plots_{pair[0]}_cell_{cell}.pdf"))
                for i in range(int(orig_trace_plane1_sig.shape[1] / 10000) + 1):
                    orig_plane1_sig = orig_trace_plane1_sig[cell, i * 10000:(i + 1) * 10000]
                    orig_plane1_ct = orig_trace_plane1_ct[cell, i * 10000:(i + 1) * 10000]
                    ica_plane1_sig = ica_trace_plane1_sig[cell, i * 10000:(i + 1) * 10000]
                    ica_plane1_ct = ica_trace_plane1_ct[cell, i * 10000:(i + 1) * 10000]

                    f = plt.figure(figsize=(20, 10))
                    plt.subplot(211)
                    plt.plot(orig_plane1_sig, 'r-', label='signal plane')
                    plt.plot(orig_plane1_ct, 'g-', label='cross-talk plane')
                    plt.title(f'original traces for cell # {cell}')
                    plt.legend(loc='upper left')
                    plt.subplot(212)
                    plt.plot(ica_plane1_sig, 'r-', label='signal plane')
                    plt.plot(ica_plane1_ct, 'g-', label='cross-talk plane')
                    plt.title(f'post-ica traces, cell # {cell}')
                    plt.legend(loc='upper left')
                    pdf.savefig(f)
                pdf.close()
            orig_trace_plane2_sig = self.plane2_traces_orig[0, :, :]
            orig_trace_plane2_ct = self.plane2_traces_orig[1, :, :]
            ica_trace_plane2_sig = self.plane2_ica_output[0, :, :]
            ica_trace_plane2_ct = self.plane2_ica_output[1, :, :]
            logging.info(f'creating figures for experiment {pair[1]}')
            for cell in range(orig_trace_plane2_sig.shape[0]):
                plot_dir = os.path.join(self.session_cache_dir, f'ica_traces_{pair[0]}_{pair[1]}/ica_plots_{pair[1]}')
                if not os.path.isdir(plot_dir):
                    os.mkdir(plot_dir)
                logging.info(f'creating figures for cell {cell}')
                pdf = matplotlib.backends.backend_pdf.PdfPages(
                    os.path.join(plot_dir, f"ica_plots_{pair[1]}_cell_{cell}.pdf"))
                for i in range(int(orig_trace_plane2_sig.shape[1] / 10000) + 1):
                    orig_plane2_sig = orig_trace_plane2_sig[cell, i * 10000:(i + 1) * 10000]
                    orig_plane2_ct = orig_trace_plane2_ct[cell, i * 10000:(i + 1) * 10000]
                    ica_plane2_sig = ica_trace_plane2_sig[cell, i * 10000:(i + 1) * 10000]
                    ica_plane2_ct = ica_trace_plane2_ct[cell, i * 10000:(i + 1) * 10000]
                    f = plt.figure(figsize=(20, 10))
                    plt.subplot(211)
                    plt.plot(orig_plane2_sig, 'r-', label='signal plane')
                    plt.plot(orig_plane2_ct, 'g-', label='cross-talk plane')
                    plt.title(f'original traces for cell # {cell}')
                    plt.legend(loc='upper left')
                    plt.subplot(212)
                    plt.plot(ica_plane2_sig, 'r-', label='signal plane')
                    plt.plot(ica_plane2_ct, 'g-', label='cross-talk plane')
                    plt.title(f'post-ica traces, cell # {cell}')
                    plt.legend(loc='upper left')
                    pdf.savefig(f)
                    plt.close()
                pdf.close()
        else:
            logging.info(f'ICA tracs for pair {pair[0]}/{pair[1]} don''t exist, nothing to plot.')

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

    @property
    def find_scale_ica(self):
        # for traces:
        scale_top = opt.minimize(self.ica_err, [1], (self.traces_unmix[:, 0], self.plane1_ica_input))
        scale_bot = opt.minimize(self.ica_err, [1], (self.traces_unmix[:, 1], self.plane2_ica_input))
        # for neuropil:
        scale_top_neuropil = opt.minimize(self.ica_err, [1],
                                          (self.neuropil_unmix[:, 0], self.plane1_ica_neuropil_input))
        scale_bot_neuropil = opt.minimize(self.ica_err, [1],
                                          (self.neuropil_unmix[:, 1], self.plane2_ica_neuropil_input))

        return scale_top.x, scale_bot.x, scale_top_neuropil.x, scale_bot_neuropil.x
