import logging
import os
import allensdk.internal.brain_observatory.demixer as demixer
import allensdk.internal.core.lims_utilities as lu
import allensdk.core.json_utilities as ju
from allensdk.brain_observatory.dff import calculate_dff
import h5py
import numpy as np
import visual_behavior.ophys.mesoscope.crosstalk_unmix as ica
import visual_behavior.ophys.mesoscope.dataset as ms
import gc
import shutil
import time

import matplotlib.pyplot as plt
from allensdk.brain_observatory.r_neuropil import estimate_contamination_ratios


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
CACHE = '/media/rd-storage/Z/MesoscopeAnalysis/'


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


def run_ica_on_session(session, roi_name=None, np_name=None):
    """
    helper function to run all crosstalk-demixing functions on a given session
    :param session: int, LIMS session ID
    :param roi_name: str, filename prefix to use for roi-related data, if different form default "roi_ica"
    :param np_name: str, filename prefix to use for neuropil-related data, if different from default "neuropil_ica"
    :return: None
    """
    ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name=roi_name, np_name=np_name)
    pairs = ica_obj.dataset.get_paired_planes()
    for pair in pairs:
        ica_obj.set_exp_ids(pair)
        ica_obj.get_ica_traces()
        ica_obj.validate_traces(return_vba=False)
        ica_obj.debias_traces()
        ica_obj.unmix_pair()
        ica_obj.plot_ica_pair(pair)
        gc.collect()
    return ica_obj


def run_ica_on_pair(session, pair, roi_name=None, np_name=None):
    """
    helper function to run all crosstalk-demixing functions on a given pair of planes
    :param pair: [int, int]: list size 2 of two LIMS experiment IDs for two planes
    :param session: int, LIMS session ID
    :param roi_name: str, filename prefix to use for roi-related data, if different form default "roi_ica"
    :param np_name: str, filename prefix to use for neuropil-related data, if different from default "neuropil_ica"
    :return: None
    """
    ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name=roi_name, np_name=np_name)
    ica_obj.set_exp_ids(pair)
    ica_obj.get_ica_traces()
    ica_obj.validate_traces(return_vba=False)
    ica_obj.debias_traces()
    ica_obj.unmix_pair()
    ica_obj.plot_ica_pair(pair)
    gc.collect()
    return


def get_ica_done_sessions(session_list=None):
    """
    function to scan all LIMS sessions nad return lists of ones that have been run through crosstalk demixing successfully, and that have not.
    as well as all mesoscope data found in lime
    :return: [pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
    """
    if not session_list:
        meso_data = ms.get_all_mesoscope_data()
        meso_data['ICA_demix_roi_session'] = 0
        sessions = meso_data['session_id']
        sessions = sessions.drop_duplicates()
    else:
        sessions = session_list
    ica_done_sessions = []
    for session in sessions:
        dataset = ms.MesoscopeDataset(session)
        pairs = dataset.get_paired_planes()
        session_ica_complete = True
        for pair in pairs:
            ica_obj = ica.MesoscopeICA(session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
            ica_obj.set_exp_ids(pair)
            ica_obj.set_ica_dirs()
            ica_obj.set_out_paths()
            # check if both roi and europil outputs exist:
            for pkey in ica_obj.pkeys:
                for tkey in ica_obj.tkeys:
                    if not os.path.isfile(ica_obj.outs_paths[pkey][tkey]):
                        session_ica_complete = False
        if session_ica_complete:
            ica_done_sessions.append(session)
    return ica_done_sessions


def get_demixing_done_sessions(session_list=None):
    """
    function to find all post-ica sessions that also ran through LIMS demixing module
    :return: [pandas.DataFrame, pandas.DataFrame, pandas.DataFrame] : demixing_done, demixing_not_done, ica_success
    """
    if not session_list:
        meso_data = ms.get_all_mesoscope_data()
        sessions = meso_data['session_id'].drop_duplicates()
    else:
        sessions = session_list

    demixing_done_sessions = []
    for session in sessions:
        demixing_session_done = True
        dataset = ms.MesoscopeDataset(session)
        pairs = dataset.get_paired_planes()
        for pair in pairs:
            ica_obj = ica.MesoscopeICA(session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
            if not os.path.isdir(os.path.join(ica_obj.session_dir, f"demixing_{pair[0]}")):
                demixing_session_done = False
            if not os.path.isdir(os.path.join(ica_obj.session_dir, f"demixing_{pair[1]}")):
                demixing_session_done = False
        if demixing_session_done:
            demixing_done_sessions.append(session)
    return demixing_done_sessions


def get_lims_done_sessions(session_list=None):
    """
    function to find all post-ica sessions that also ran through LIMS modules
    :return: [pandas.DataFrame, pandas.DataFrame, pandas.DataFrame] : lims_roi_success, lims_roi_fail, ica_success
    """
    if not session_list:
        meso_data = ms.get_all_mesoscope_data()
        sessions = meso_data['session_id'].drop_duplicates()
    else:
        sessions = session_list
    lims_sessions = []
    for session in sessions:
        dataset = ms.MesoscopeDataset(session)
        pairs = dataset.get_paired_planes()
        lims_session_done = True
        for pair in pairs:
            ica_obj = ica.MesoscopeICA(session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
            ica_obj.set_exp_ids(pair)
            ica_obj.set_ica_dirs()
            if not os.path.isfile(os.path.join(ica_obj.session_dir, f"{pair[0]}_dff.h5")):
                lims_session_done = False
            if not os.path.isfile(os.path.join(ica_obj.session_dir, f"{pair[1]}_dff.h5")):
                lims_session_done = False

            if lims_session_done:
                lims_sessions.append(session)

    return lims_sessions


def get_ica_exp_by_cre_line(cre_line, md):
    """
    helper function to get ica experiments by cre line
    :param cre_line: str, cre line name form LIMS
    :param md: pandas.Dataframe - all mesoscope data - returned by mesoscope.get_all_mesoscope_data()
    :return: [pandas.DataFrame, pandas.DataFrame] : data frames with information on cre-line specific
    experiments that failed or succeed crosstalk correction
    """
    md_success = md.loc[md['ICA_demix_session'] == 1]
    md_fail = md.loc[md['ICA_demix_session'] == 0]
    cre_md_success = md_success.loc[md_success['specimen'].str.contains(cre_line)]
    cre_md_fail = md_fail.loc[md_fail['specimen'].str.contains(cre_line)]
    cre_exp_success = cre_md_success['experiment_id']
    cre_exp_fail = cre_md_fail['experiment_id']
    return cre_exp_success, cre_exp_fail


def get_ica_ses_by_cre_line(cre_line, md):
    """
    helper function to get ica sessions by cre line
    :param cre_line: str, cre line name form LIMS
    :param md: pandas.Dataframe - all mesoscope data - returned by mesoscope.get_all_mesoscope_data()
    :return: [pandas.DataFrame, pandas.DataFrame] : data frames with information on cre-line specific
    sessions that failed or succeed crosstalk correction
    """
    md_success = md.loc[md['ICA_demix_session'] == 1]
    md_fail = md.loc[md['ICA_demix_session'] == 0]
    cre_md_success = md_success.loc[md_success['specimen'].str.contains(cre_line)]
    cre_md_fail = md_fail.loc[md_fail['specimen'].str.contains(cre_line)]
    cre_ses_success = cre_md_success.drop_duplicates('session_id')['session_id']
    cre_ses_fail = cre_md_fail.drop_duplicates('session_id')['session_id']
    return cre_ses_success, cre_ses_fail


def parse_input(data):

    exclude_labels = ["union", "duplicate", "motion_border"]
    movie_h5 = get_path(data, "movie_h5", True)
    traces_h5 = get_path(data, "traces_h5", True)
    traces_h5_ica = get_path(data, "traces_h5_ica", True)
    output_h5 = get_path(data, "output_file", False)
    traces_valid = get_path(data, "traces_valid", True)

    with h5py.File(movie_h5, "r") as f:
        movie_shape = f["data"].shape[1:]

    with h5py.File(traces_h5_ica, "r") as f:
        traces = f["data"][()]
    traces = traces[0, :, :]

    with h5py.File(traces_h5, "r") as f:
        trace_ids = [int(rid) for rid in f["roi_names"][()]]

    exp_traces_valid = ju.read(traces_valid)

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
                raise ValueError(f"Could not find cell roi id %d in roi traces file: {e}" % rid)
            masks[ridx, :, :] = mask
            current_exclusion_labels = set(roi.get("exclusion_labels", []))
            valid[ridx] = len(exclude_labels & current_exclusion_labels) == 0

    return traces, masks, valid, np.array(trace_ids), movie_h5, output_h5


def run_demixing_on_session(session, cache=CACHE):
    """
    run LIMS demixing on crosstalk corrected traces
    :param session: LIMS session id
    :param cache: directory containing crosstalk corrected traces
    :return:
    """

    dataset = ms.MesoscopeDataset(session)
    pairs = dataset.get_paired_planes()

    # if self.debug_mode:
    #     logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # else:
    #     logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    for pair in pairs:
        for exp_id in pair:
            exp_files_data = ms.get_mesoscope_exp_files(exp_id)
            if len(exp_files_data) == 0:
                logger.info(f"Movie file does not exist for experiment {exp_id}, skipping")
                continue
            else:
                movie_dir = exp_files_data.movie_dir.values[0]
                movie_file = exp_files_data.movie_name.values[0]
                movie_path = os.path.join(movie_dir, movie_file)
            # check if file exist:
                if not os.path.exists(movie_path):
                    logger.info(f"Movie file does not exist for experiment {exp_id}, skipping")
                    continue
                else:
                    demix_path = os.path.join(cache, f'session_{session}/demixing_{exp_id}')
                    demix_file = os.path.join(demix_path, f'traces_demixing_output_{exp_id}.h5')
                if os.path.isfile(demix_file):
                    logging.info(f"Demixed traces exist for experiment {exp_id}, skipping demixing")
                    continue
                else:
                    logging.info(f"Demixing {exp_id}")
                    exp_dir = exp_files_data.exp_dir.values[0]
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
                    demix_path = os.path.join(cache, f'session_{session}/demixing_{exp_id}')
                    ica_dir = f'session_{session}/ica_traces_{pair[0]}_{pair[1]}/'
                    traces_valid = os.path.join(cache, ica_dir, f'{exp_id}_valid.json')
                    data = {
                        "movie_h5": os.path.join(exp_dir, "processed", "concat_31Hz_0.h5"),
                        "traces_h5_ica": os.path.join(cache, ica_dir, f'{exp_id}_out.h5'),
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
                        movie = f['data'][()]

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
                    out_traces = demixed_traces
                    with h5py.File(output_h5, 'w') as f:
                        f.create_dataset("data", data=out_traces, compression="gzip")
                        f.create_dataset("roi_names", data=[np.string_(rn) for rn in trace_ids])
    return


def debug_plot(file_name, roi_trace, neuropil_trace, corrected_trace, r, r_vals=None, err_vals=None, figshow=False):

    if not figshow:
        print(f'Switching backend to Agg')
        plt.switch_backend('Agg')

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


def run_neuropil_correction_on_ica(session, ica_cache_dir=CACHE):
    """
    run neuropil correction on CIA output files
    :param session: LIMS session id
    :param ica_cache_dir: directory to find processed ica-demixed outputs
    :return: None
    """
    ica_obj = ica.MesoscopeICA(session_id=session, cache=ica_cache_dir, roi_name="ica_traces", np_name="ica_neuropil")
    pairs = ica_obj.dataset.get_paired_planes()
    for pair in pairs:
        ica_obj.set_exp_ids(pair)
        ica_obj.set_ica_dirs()
        ica_obj.set_ica_input_paths()
        ica_obj.set_out_paths()
        # prelude -- get processing metadata
        ses_dir = ica_obj.session_dir
        for pkey in ica_obj.pkeys:
            exp_id = ica_obj.exp_ids[pkey]
            logging.info(f"Processing experiment {exp_id}")
            neuropil_file = os.path.join(ses_dir, f'neuropil_corrected_{exp_id}',
                                         f'neuropil_correction.h5')
            if os.path.isfile(neuropil_file):
                logging.info(f"Neuropil corrected traces exist for experiment {exp_id}, skipping neuropil correction")
                continue
            else:
                demix_dir = os.path.join(ses_dir, f'demixing_{exp_id}')
                trace_file = os.path.join(demix_dir, f'traces_demixing_output_{exp_id}.h5')
                neuropil_trace_file = ica_obj.outs_paths[pkey]['np']
                storage_dir = os.path.join(ses_dir, f'neuropil_corrected_{exp_id}')
                if not os.path.isdir(storage_dir):
                    os.mkdir(storage_dir)

                plot_dir = os.path.join(storage_dir, "neuropil_subtraction_plots")

                if os.path.exists(plot_dir):
                    shutil.rmtree(plot_dir)

                try:
                    os.makedirs(plot_dir)
                except Exception as e:
                    logging.error(f"Error: {e}")
                    pass

                logging.info(f"Running neuropil correction on {exp_id}")
                logging.info(f"Neuropil correcting {trace_file}")

                # process data

                try:
                    roi_traces = h5py.File(trace_file, "r")
                except Exception as e:
                    logging.error(f"Error: {e}, most likely unable to open ROI trace file {trace_file}")
                    raise

                try:
                    neuropil_traces = h5py.File(neuropil_trace_file, "r")
                except Exception as e:
                    logging.error(f"Error: {e} most likely unable to open neuropil trace file {neuropil_trace_file}")

                # get number of traces, length, etc.
                num_traces, t = roi_traces['data'].shape
                t_orig = t

                # make sure that ROI and neuropil trace files are organized the same
                n_id = roi_traces["roi_names"]
                r_id = roi_traces["roi_names"]

                logging.info("Processing %d traces", len(n_id))
                assert len(n_id) == len(r_id), "Input trace files are not aligned (ROI count)"
                for i in range(len(n_id)):
                    assert n_id[i] == r_id[i], "Input trace files are not aligned (ROI IDs)"

                # initialize storage variables and analysis routine
                r_list = [None] * num_traces
                rmse_list = [-1] * num_traces
                roi_names = n_id
                corrected = np.zeros((num_traces, t_orig))
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

                    # r = None

                    logging.info("Correcting trace %d (roi %s)", n, str(n_id[n]))
                    results = estimate_contamination_ratios(roi, neuropil)
                    logging.info("r=%f err=%f it=%d", results["r"], results["err"], results["it"])

                    r = results["r"]
                    fc = roi - r * neuropil
                    rmse_list[n] = results["err"]
                    r_vals[n] = results["r_vals"]

                    debug_plot(os.path.join(plot_dir, "initial_%04d.png" % n),
                               roi, neuropil, fc, r, results["r_vals"], results["err_vals"])

                    # mean of the corrected trace must be positive
                    if fc.mean() > 0:
                        r_list[n] = r
                        corrected[n, :] = fc
                    else:
                        logging.warning("fc has negative baseline, skipping this r value")

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

                # write out processed data

                try:
                    savefile = os.path.join(storage_dir, "neuropil_correction.h5")
                    hf = h5py.File(savefile, 'w')
                    hf.create_dataset("r", data=r_list)
                    hf.create_dataset("RMSE", data=rmse_list)
                    hf.create_dataset("FC", data=corrected, compression="gzip")
                    hf.create_dataset("roi_names", data=roi_names)

                    for n in range(num_traces):
                        r = r_vals[n]
                        if r is not None:
                            hf.create_dataset("r_vals/%d" % n, data=r)
                    hf.close()
                except Exception as e:
                    logging.error(f"Error creating output h5 file: {e}")
                    raise

                roi_traces.close()
                neuropil_traces.close()
                logging.info(f"finished experiment {exp_id}")
    return


def run_dff_on_ica(session, an_dir=CACHE):
    """
    run LIMS dff extraction on ICA output files, after demixing and neuropil correction
    :param session: LIMS session id
    :param an_dir: directory where the traces are stored
    :return:
    """
    dataset = ms.MesoscopeDataset(session)
    pairs = dataset.get_paired_planes()
    ses_dir = os.path.join(an_dir, f'session_{session}')

    for pair in pairs:
        for exp_id in pair:

            input_file = os.path.join(ses_dir, f'neuropil_corrected_{exp_id}/neuropil_correction.h5')
            output_file = os.path.join(ses_dir, f'{exp_id}_dff.h5')

            if os.path.exists(output_file):
                logging.info(f"df/f traces exist for experiment {exp_id} in {output_file}")
                continue
            else:
                if not os.path.exists(input_file):
                    raise IOError("input file does not exists: %s" % input_file)
                logging.info(f"Calculating df/f traces for {input_file}")

                # read from "data"
                input_h5 = h5py.File(input_file, "r")
                traces = input_h5['FC'][()]
                input_h5.close()

                dff = calculate_dff(traces)

                logging.info(f"Writing to disk {output_file}")
                # write to "data"
                output_h5 = h5py.File(output_file, "w")
                output_h5['data'] = dff
                output_h5.close()

    return


def before_date(file_date, thr_date="01/01/2020"):
    # convert th_date to time
    thr = time.strptime(thr_date, "%m/%d/%Y")
    # get file's time of modification:
    file_time = time.ctime(os.path.getmtime(file_date))
    ft = time.strptime(file_time, "%a %b %d %H:%M:%S %Y")
    # compare
    if time.mktime(ft) < time.mktime(thr):
        flag = True
    else:
        flag = False
    return flag


def clean_up_cache(sessions, cache, remove_inputs=False, remove_by_date="01/01/2020"):
    """
    deletes ica outputs from cache:
        neuropil_ica_output_pair{i}.h5
        neuropil_valid_pair{i}.json
        neuropil_ica_mixing_pair{i}.h5
        ica_traces_output_pair{i}.h5
        ica_valid_pair{i}.json
        ica_mixing_pair{i}.h5
        neuropil correction files
        demixing files
        dff traces files
    :param remove_by_date: file creating date threshold - don't remove if created after this date
    :param sessions: list of LIMS session ids
    :param remove_inputs: falg that controls whether we are deleting extracted traces or not
    :param cache: cache directory
    :return: None
    """

    for session in sessions:
        ica_obj = ica.MesoscopeICA(session_id=session, cache=cache, roi_name="ica_traces", np_name="ica_neuropil")
        ses_dir = ica_obj.session_dir
        if os.path.isdir(ses_dir):
            pairs = ica_obj.dataset.get_paired_planes()
            for pair in pairs:
                ica_obj.set_exp_ids(pair)
                ica_obj.set_ica_dirs()
                ica_obj.set_raws_paths()
                ica_obj.set_out_paths()
                ica_obj.set_valid_paths()
                ica_obj.set_plot_dirs(dir_name=None)
                ica_obj.set_ica_input_paths()
                for tkey in ica_obj.tkeys:
                    exp_dir = ica_obj.dirs[tkey]
                    if os.path.isdir(exp_dir):
                        for pkey in ica_obj.pkeys:
                                out = ica_obj.outs_paths[pkey][tkey]
                                valid = ica_obj.rois_valid_paths[pkey][tkey]
                                ica_input = ica_obj.ins_paths[pkey][tkey]
                                plot_dir = ica_obj.plot_dirs[pkey][tkey]
                                if os.path.isfile(out):
                                    if before_date(out, remove_by_date):
                                        print(f'deteling {out}')
                                        os.remove(out)
                                if os.path.isfile(valid):
                                    if before_date(valid, remove_by_date):
                                        print(f'deteling {valid}')
                                        os.remove(valid)
                                if os.path.isfile(ica_input):
                                    if before_date(ica_input, remove_by_date):
                                        print(f'deteling {ica_input}')
                                        os.remove(ica_input)
                                if os.path.isdir(plot_dir):
                                    if before_date(plot_dir, remove_by_date):
                                        print(f'deteling {plot_dir}')
                                        shutil.rmtree(plot_dir, ignore_errors=True)
                                if remove_inputs:
                                    raw = ica_obj.raw_paths[pkey][tkey]
                                    if os.path.isfile(raw):
                                        if before_date(raw, remove_by_date):
                                            print(f'deteling {raw}')
                                            os.remove(raw)
                    else:
                        print(f"ICA ROI dir does not exist: {exp_dir}")
                # removing LIMS processing outputs:
                dem_out_p1 = os.path.join(ses_dir, f'demixing_{pair[0]}')
                dem_out_p2 = os.path.join(ses_dir, f'demixing_{pair[1]}')
                np_out_p1 = os.path.join(ses_dir, f'neuropil_corrected_{pair[0]}')
                np_out_p2 = os.path.join(ses_dir, f'neuropil_corrected_{pair[1]}')
                dff_p1 = os.path.join(ses_dir, f'{pair[0]}_dff.h5')
                dff_p2 = os.path.join(ses_dir, f'{pair[1]}_dff.h5')
                # removing dff files
                if os.path.isfile(dff_p1):
                    if before_date(dff_p1, remove_by_date):
                        os.remove(dff_p1)
                        print(f'deteling {dff_p2}')
                if os.path.isfile(dff_p2):
                    if before_date(dff_p2, remove_by_date):
                        os.remove(dff_p2)
                        print(f'deteling {dff_p2}')
                # removing directories for demixing plane 1
                if os.path.isdir(dem_out_p1):
                    if before_date(dem_out_p1, remove_by_date):
                        shutil.rmtree(dem_out_p1, ignore_errors=True)
                        print(f'deteling {dem_out_p1}')
                # removing directories for demixing plane 2
                if os.path.isdir(dem_out_p2):
                    if before_date(dem_out_p2, remove_by_date):
                        shutil.rmtree(dem_out_p2, ignore_errors=True)
                        print(f'deteling {dem_out_p2}')
                # removing directories for neuropil correction plane 1
                if os.path.isdir(np_out_p1):
                    if before_date(np_out_p1, remove_by_date):
                        shutil.rmtree(np_out_p1, ignore_errors=True)
                        print(f'deteling {np_out_p1}')
                # removing directories for neuropil correction plane 2
                if os.path.isdir(np_out_p2):
                    if before_date(np_out_p2, remove_by_date):
                        shutil.rmtree(np_out_p2, ignore_errors=True)
                        print(f'deteling {np_out_p2}')

    return


def delete_old_files(sessions, CACHE, names_files= None, remove_by_date='04/01/2020'):
    if not names_files:
        names_files = {'roi': ['ica_traces', 'traces_ica', 'roi_traces', 'traces_roi', 'crosstalk', 'mixing'],
                       'np': ['ica_neuropil', 'neuropil_ica', 'crosstalk', 'mixing']}
    for session in sessions:
        ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
        ses_dir = ica_obj.session_dir
        print(f'scanning sesion : {session}')
        if os.path.isdir(ses_dir):
            pairs = ica_obj.dataset.get_paired_planes()
            for pair in pairs:
                ica_obj.set_exp_ids(pair)
                ica_obj.set_ica_dirs()
                for tkey in ica_obj.tkeys:
                    exp_dir = ica_obj.dirs[tkey]
                    if os.path.isdir(exp_dir):
                        print(f'scanning experiment :{exp_dir}')
                        items = os.listdir(exp_dir)
                        for item in items:
                            item_path = os.path.join(exp_dir, item)
                            if os.path.isfile(item_path):
                                delete_file_flag = False
                                for name in names_files[tkey]:
                                    if name in item:
                                        delete_file_flag = True
                                if delete_file_flag:
                                    os.remove(item_path)
                                    print(f'deleting {item_path}')
                            elif os.path.isdir(item_path):
                                if before_date(item_path, remove_by_date):
                                    print(f'deteling {item_path}')
                                    shutil.rmtree(item_path, ignore_errors=True)
                            else:
                                print(f'{item} is not a file or a dir')
                    else:
                        print(f"experiment {exp_dir} doesn't exist")
                        continue
        else:
            print(f"sessions {session} doens't exist")
            continue
    return


def refactor_valid(sessions):
    list_exp = {}
    for session in sessions:
        ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
        pairs = ica_obj.dataset.get_paired_planes()
        for pair in pairs:
            ica_obj.set_exp_ids(pair)
            ica_obj.set_ica_dirs()
            ica_obj.set_valid_paths()
            for pkey in ica_obj.pkeys:
                for tkey in ica_obj.tkeys:
                    old_path = ica_obj.rois_valid_paths[pkey][tkey]
                    # make a paht to the new filename
                    new_valid_path = os.path.join(ica_obj.dirs[tkey], f"{ica_obj.exp_ids[pkey]}_valid_1.json")
                    # copy valid json to the new filename
                    shutil.copy(old_path, new_valid_path)
                    # read old valid json, reformat and save to disk to the same name.
                    valid_old = ju.read(old_path)
                    if 'signal' in valid_old:
                        print(f"Refactoring {ica_obj.exp_ids[pkey]}")
                        valid_new_r = valid_old['signal']
                        if 'roi' in valid_new_r:
                            del valid_new_r['roi']
                        if 'np' in valid_new_r:
                            del valid_new_r['np']
                        ju.write(old_path, valid_new_r)
                    else:
                        list_exp[tkey] = ica_obj.exp_ids[pkey]
    return list_exp


def refactor_outputs(sessions):
    for session in sessions:
        print(f'processing session: {session}')
        ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
        pairs = ica_obj.dataset.get_paired_planes()
        for pair in pairs:
            ica_obj.set_exp_ids(pair)
            ica_obj.set_ica_dirs()
            ica_obj.set_ica_input_paths()
            ica_obj.set_out_paths()
            self = ica_obj
            ct_offset = {}
            traces_out = {}
            sig_offset = {}
            for pkey in self.pkeys:
                sig_offset[pkey] = {}
                ct_offset[pkey] = {}
                traces_out[pkey] = {}
                for tkey in self.tkeys:
                    sig_offset[pkey][tkey] = {}
                    ct_offset[pkey][tkey] = {}
                    traces_out[pkey][tkey] = {}
                    # copy old output ot out_1.h5
                    old_path = self.outs_paths[pkey][tkey]
                    # make a paht to the new filename
                    new_path = sc.makefilepath(filename = f"{self.exp_ids[pkey]}_out_1.h5", folder = self.dirs[tkey])
                    # copy valid json to the new filename
                    if os.path.isfile(new_path):
                        print(f"skipping experint {self.exp_ids[pkey]}")
                        continue # skipping to next exp
                    else:
                        if os.path.isfile(self.ins_paths[pkey][tkey]) and os.path.isfile(self.outs_paths[pkey][tkey]):
                            print(f'processing experiment {self.exp_ids[pkey]}, {tkey}')
                            with h5py.File(self.ins_paths[pkey][tkey], "r") as f:
                                sig_offset[pkey][tkey] = f['sig_offset'][()]
                                ct_offset[pkey][tkey] = f['ct_offset'][()]
                            with h5py.File(self.outs_paths[pkey][tkey], "r") as f:
                                traces_out[pkey][tkey] = f["data"][()]
                                self.crosstalk[pkey][tkey] = f["crosstalk"][()]
                                self.mixing[pkey][tkey] = f["mixing_matrix"][()]
                                self.a_mixing[pkey][tkey] = f["mixing_matrix_adjusted"][()]
                            self.offsets[pkey][tkey] = {'sig_offset': sig_offset[pkey][tkey],
                                                        'ct_offset': ct_offset[pkey][tkey]}
                            self.outs[pkey][tkey] = np.array(
                                                    [traces_out[pkey][tkey][0] + self.offsets[pkey][tkey]['sig_offset'],
                                                     traces_out[pkey][tkey][1] + self.offsets[pkey][tkey]['ct_offset']])
                            shutil.copy(old_path, new_path)
                            with h5py.File(self.outs_paths[pkey][tkey], "w") as f:
                                f.create_dataset(f"data", data=self.outs[pkey][tkey])
                                f.create_dataset(f"crosstalk", data=self.crosstalk[pkey][tkey])
                                f.create_dataset(f"mixing_matrix_adjusted", data=self.a_mixing[pkey][tkey])
                                f.create_dataset(f"mixing_matrix", data=self.mixing[pkey][tkey])
                        else:
                            print(f"input or putput for exp {self.exp_ids[pkey]}, {tkey} doesn't exist")
    return
