import matplotlib
import logging
import os
import allensdk.internal.brain_observatory.demixer as demixer
import allensdk.internal.core.lims_utilities as lu
import allensdk.core.json_utilities as ju
import h5py
import numpy as np
import visual_behavior.ophys.mesoscope.crosstalk_unmix as ica
import visual_behavior.ophys.mesoscope.mesoscope as ms
import shutil

matplotlib.use('agg')
import matplotlib.pyplot as plt
from allensdk.brain_observatory.r_neuropil import estimate_contamination_ratios


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

meso_data = ms.get_all_mesoscope_data()
meso_data['ICA_demix_exp'] = 0
meso_data['ICA_demix_session'] = 0


def assert_exists(file_name):
    if not os.path.exists(file_name):
        raise IOError("file does not exist: %s" % file_name)


def get_path(obj, key, check_exists):
    try:
        path = obj[key]
    except KeyError:
        raise KeyError("required input field '%s' does not exist" % key)
    if check_exists:
        assert_exists(path)
    return path


def run_ica_on_session(session):
    ica_obj = ica.MesoscopeICA(session_id=session, cache='/media/NCRAID/MesoscopeAnalysis/')
    pairs = ica_obj.dataset.get_paired_planes()
    for pair in pairs:
        ica_obj.get_ica_traces(pair)
        ica_obj.validate_traces()
        ica_obj.combine_debias_traces()
        ica_obj.combine_debias_neuropil()
        ica_obj.unmix_traces(max_iter=50)
        ica_obj.unmix_neuropil(max_iter=100)
        ica_obj.plot_ica_traces(pair)

        # if np.all(ica_obj.found_solution == True):
        #     meso_data['ICA_demix_exp'].loc[meso_data['experiment_id'] == pair[0]] = 1
    return

def get_ica_sessions():
    meso_data = ms.get_all_mesoscope_data()
    meso_data['ICA_demix_exp'] = 0
    meso_data['ICA_demix_session'] = 0
    sessions = meso_data['session_id']
    sessions = sessions.drop_duplicates()

    for session in sessions:
        dataset = ms.MesoscopeDataset(session)
        pairs = dataset.get_paired_planes()
        for pair in pairs:
            ica_obj = ica.MesoscopeICA(session, cache='/media/NCRAID/MesoscopeAnalysis')
            ica_obj.set_ica_traces_dir(pair)
            if os.path.isfile(ica_obj.plane1_ica_output_pointer):
                meso_data['ICA_demix_exp'].loc[meso_data['experiment_id'] == pair[0]] = 1
            if os.path.isfile(ica_obj.plane2_ica_output_pointer):
                meso_data['ICA_demix_exp'].loc[meso_data['experiment_id'] == pair[1]] = 1
            session_data = meso_data.loc[meso_data['session_id'] == session]
            if all(session_data.ICA_demix_exp == 1):
                for exp in session_data.experiment_id:
                    meso_data['ICA_demix_session'].loc[meso_data.experiment_id == exp] = 1

    ica_success = meso_data.loc[meso_data['ICA_demix_session'] == 1]
    ica_fail = meso_data.loc[meso_data['ICA_demix_session'] == 0]
    return ica_success, ica_fail


def parse_input(data, exclude_labels=["union", "duplicate", "motion_border"]):
    exclude_labels = ["union", "duplicate", "motion_border"]
    movie_h5 = get_path(data, "movie_h5", True)
    traces_h5 = get_path(data, "traces_h5", True)
    traces_h5_ica = get_path(data, "traces_h5_ica", True)
    output_h5 = get_path(data, "output_file", False)
    traces_valid = get_path(data, "traces_valid", True)

    with h5py.File(movie_h5, "r") as f:
        movie_shape = f["data"].shape[1:]

    with h5py.File(traces_h5_ica, "r") as f:
        traces = f["data"].value
    traces = traces[0, :, :]

    with h5py.File(traces_h5, "r") as f:
        trace_ids = [int(rid) for rid in f["roi_names"].value]

    exp_traces_valid = ju.read(traces_valid)["signal"]

    rois = get_path(data, "roi_masks", False)
    masks = None
    valid = None

    exclude_labels = set(exclude_labels)

    for roi in rois:
        if exp_traces_valid[str(roi["id"])]:
            mask = np.zeros(movie_shape, dtype=bool)
            mask_matrix = np.array(roi["mask"], dtype=bool)
            mask[roi["y"]:roi["y"] + roi["height"], roi["x"]:roi["x"] + roi["width"]] = mask_matrix

            if masks is None:
                masks = np.zeros((len(rois), mask.shape[0], mask.shape[1]), dtype=bool)
                valid = np.zeros(len(rois), dtype=bool)

            rid = int(roi["id"])
            try:
                ridx = trace_ids.index(rid)
            except ValueError as e:
                raise ValueError("Could not find cell roi id %d in roi traces file" % rid)

            masks[ridx, :, :] = mask

            current_exclusion_labels = set(roi.get("exclusion_labels", []))
            valid[ridx] = len(exclude_labels & current_exclusion_labels) == 0

    return traces, masks, valid, np.array(trace_ids), movie_h5, output_h5

def run_demixing_on_ica(session, an_dir='/media/NCRAID/MesoscopeAnalysis/'):
    mds, _ = get_ica_sessions()
    dataset = ms.MesoscopeDataset(session)
    pairs = dataset.get_paired_planes()

    for pair in pairs:
        for exp_id in pair:
            demix_path = os.path.join(an_dir, f'session_{session}/demixing_{exp_id}')

            if os.path.isfile(demix_path):
                logging.info("Demixed traces exist for experiment {exp_id}, skipping demixing")
                continue
            else:
                logging.info("Demixing {exp_id}")
                exp_dir = mds['experiment_folder'].loc[mds['experiment_id'] == exp_id].values[0]
                query = f"""
                select *
                from ophys_cell_segmentation_runs
                where current = True and ophys_experiment_id = {exp_id}
                """
                seg_run = lu.query(query)[0]['id']

                query = f"""
                select *
                from cell_rois
                where ophys_cell_segmentation_run_id = {seg_run}
                """
                rois = lu.query(query)

                nrois = {roi['id']: dict(width=roi['width'],
                                         height=roi['height'],
                                         x=roi['x'],
                                         y=roi['y'],
                                         id=roi['id'],
                                         valid=roi['valid_roi'],
                                         mask=roi['mask_matrix'],
                                         exclusion_labels=[])
                         for roi in rois}

                demix_path = os.path.join(an_dir, f'session_{session}/demixing_{exp_id}')
                ica_dir = f'session_{session}/ica_traces_{pair[0]}_{pair[1]}/'

                traces_valid = os.path.join(an_dir, ica_dir, f'valid_{exp_id}.json')

                data = {
                    "movie_h5": os.path.join(exp_dir, "processed", "concat_31Hz_0.h5"),
                    "traces_h5_ica": os.path.join(an_dir, ica_dir, f'traces_ica_output_{exp_id}.h5'),
                    "traces_h5": os.path.join(exp_dir, "processed", "roi_traces.h5"),
                    "roi_masks": nrois.values(),
                    "traces_valid": traces_valid,
                    "output_file": os.path.join(demix_path, f"traces_demixing_output_{exp_id}.h5")
                }

                traces, masks, valid, trace_ids, movie_h5, output_h5 = parse_input(data)

                # only demix non-union, non-duplicate ROIs
                valid_idxs = np.where(valid)
                demix_traces = traces
                demix_masks = masks[valid_idxs]

                with h5py.File(movie_h5, 'r') as f:
                    movie = f['data'].value

                demixed_traces, drop_frames = demixer.demix_time_dep_masks(demix_traces, movie, demix_masks)

                if not os.path.isdir(demix_path):
                    os.mkdir(demix_path)
                plot_dir = os.path.join(demix_path, 'plots')
                if not os.path.isdir(plot_dir):
                    os.mkdir(plot_dir)

                nt_inds = demixer.plot_negative_transients(demix_traces,
                                                           demixed_traces,
                                                           valid[valid_idxs],
                                                           demix_masks,
                                                           trace_ids[valid_idxs],
                                                           plot_dir)

                logging.debug("rois with negative transients: %s", str(trace_ids[valid_idxs][nt_inds]))

                nb_inds = demixer.plot_negative_baselines(demix_traces,
                                                          demixed_traces,
                                                          demix_masks,
                                                          trace_ids[valid_idxs],
                                                          plot_dir)

                # negative baseline rois (and those that overlap with them) become nans
                logging.debug("rois with negative baselines (or overlap with them): %s",
                              str(trace_ids[valid_idxs][nb_inds]))
                demixed_traces[nb_inds, :] = np.nan
                logging.info("Saving output")
                out_traces = np.zeros(traces.shape, dtype=demix_traces.dtype)

                out_traces = demixed_traces

                with h5py.File(output_h5, 'w') as f:
                    f.create_dataset("data", data=out_traces, compression="gzip")
                    f.create_dataset("roi_names", data=[np.string_(rn) for rn in trace_ids])

    return

def debug_plot(file_name, roi_trace, neuropil_trace, corrected_trace, r, r_vals=None, err_vals=None):
    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(211)
    ax.plot(roi_trace, 'r', label="raw")
    ax.plot(corrected_trace, 'b', label="fc")
    ax.plot(neuropil_trace, 'g', label="neuropil")
    ax.set_xlim(0, roi_trace.size)
    ax.set_title('raw(%.02f, %.02f) fc(%.02f, %.02f) r(%f)' % (
    roi_trace.min(), roi_trace.max(), corrected_trace.min(), corrected_trace.max(), r))
    ax.legend()

    if r_vals is not None:
        ax = fig.add_subplot(212)
        ax.plot(r_vals, err_vals, "o")

    plt.savefig(file_name)
    plt.close()


def run_neuropil_correction_on_ica(session, an_dir='/media/NCRAID/MesoscopeAnalysis/'):
    dataset = ms.MesoscopeDataset(session)
    pairs = dataset.get_paired_planes()
    for pair in pairs:
        for exp_id in pair:
            #######################################################################
            # prelude -- get processing metadata
            ses_dir = os.path.join(an_dir, f'session_{session}')

            neuropil_file = os.path.join(ses_dir, f'ica_neuropil_{pair[0]}_{pair[1]}',
                                         f'neuropil_ica_output_{exp_id}.h5')

            if os.path.isfile(neuropil_file):
                logging.info("Neuropil corrected traces exist for experiment {exp_id}, skipping neuropil correction")
                continue
            else:
                logging.info("Running neuropil correction on {exp_id}")
                demix_dir = os.path.join(ses_dir, f'demixing_{exp_id}')
                trace_file = os.path.join(demix_dir, f'traces_demixing_output_{exp_id}.h5')
                storage_dir = os.path.join(ses_dir, f'neuropil_corrected_{exp_id}')
                if not os.path.isdir(storage_dir):
                    os.mkdir(storage_dir)

                plot_dir = os.path.join(storage_dir, "neuropil_subtraction_plots")

                if os.path.exists(plot_dir):
                    shutil.rmtree(plot_dir)

                try:
                    os.makedirs(plot_dir)
                except:
                    pass

                logging.info("Neuropil correcting '%s'", trace_file)

                ########################################################################
                # process data

                try:
                    roi_traces = h5py.File(trace_file, "r")
                except:
                    logging.error("Error: unable to open ROI trace file '%s'", trace_file)
                    raise

                try:
                    neuropil_traces = h5py.File(neuropil_file, "r")
                except:
                    logging.error("Error: unable to open neuropil trace file '%s'", neuropil_file)
                    raise

                '''
                get number of traces, length, etc.
                '''
                num_traces, T = roi_traces['data'].shape
                T_orig = T
                T_cross_val = int(T / 2)
                if (T - T_cross_val > T_cross_val):
                    T = T - 1

                # make sure that ROI and neuropil trace files are organized the same
                n_id = roi_traces["roi_names"]
                r_id = roi_traces["roi_names"]

                logging.info("Processing %d traces", len(n_id))
                assert len(n_id) == len(r_id), "Input trace files are not aligned (ROI count)"
                for i in range(len(n_id)):
                    assert n_id[i] == r_id[i], "Input trace files are not aligned (ROI IDs)"
                '''
                initialize storage variables and analysis routine
                '''
                r_list = [None] * num_traces
                RMSE_list = [-1] * num_traces
                roi_names = n_id
                corrected = np.zeros((num_traces, T_orig))
                r_vals = [None] * num_traces

                for n in range(num_traces):
                    roi = roi_traces['data'][n]
                    neuropil = neuropil_traces['data'][0][n]

                    if np.any(np.isnan(neuropil)):
                        logging.warning("neuropil trace for roi %d contains NaNs, skipping", n)
                        continue

                    if np.any(np.isnan(roi)):
                        logging.warning("roi trace for roi %d contains NaNs, skipping", n)
                        continue

                    r = None

                    logging.info("Correcting trace %d (roi %s)", n, str(n_id[n]))
                    results = estimate_contamination_ratios(roi, neuropil)
                    logging.info("r=%f err=%f it=%d", results["r"], results["err"], results["it"])

                    r = results["r"]
                    fc = roi - r * neuropil
                    RMSE_list[n] = results["err"]
                    r_vals[n] = results["r_vals"]

                    debug_plot(os.path.join(plot_dir, "initial_%04d.png" % n),
                               roi, neuropil, fc, r, results["r_vals"], results["err_vals"])

                    # mean of the corrected trace must be positive
                    if fc.mean() > 0:
                        r_list[n] = r
                        corrected[n, :] = fc
                    else:
                        logging.warning("fc has negative baseline, skipping this r value")

                # compute mean valid r value
                r_mean = np.array([r for r in r_list if r is not None]).mean()

                # fill in empty r values
                for n in range(num_traces):
                    roi = roi_traces['data'][n]
                    neuropil = neuropil_traces['data'][0][n]

                    if r_list[n] is None:
                        logging.warning("Error estimated r for trace %d. Setting to zero.", n)
                        r_list[n] = 0
                        corrected[n, :] = roi

                    # save a debug plot
                    debug_plot(os.path.join(plot_dir, "final_%04d.png" % n),
                               roi, neuropil, corrected[n, :], r_list[n])

                    # one last sanity check
                    eps = -0.0001
                    if np.mean(corrected[n, :]) < eps:
                        raise Exception("Trace %d baseline is still negative value after correction" % n)

                    if r_list[n] < 0.0:
                        raise Exception("Trace %d ended with negative r" % n)

                ########################################################################
                # write out processed data

                try:
                    savefile = os.path.join(storage_dir, "neuropil_correction.h5")
                    hf = h5py.File(savefile, 'w')
                    hf.create_dataset("r", data=r_list)
                    hf.create_dataset("RMSE", data=RMSE_list)
                    hf.create_dataset("FC", data=corrected, compression="gzip")
                    hf.create_dataset("roi_names", data=roi_names)

                    for n in range(num_traces):
                        r = r_vals[n]
                        if r is not None:
                            hf.create_dataset("r_vals/%d" % n, data=r)
                    hf.close()
                except:
                    logging.error("Error creating output h5 file")
                    raise

                roi_traces.close()
                neuropil_traces.close()

                logging.info("finished")
    return