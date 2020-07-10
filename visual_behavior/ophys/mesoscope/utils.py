import allensdk.core.json_utilities as ju
from allensdk.brain_observatory.dff import calculate_dff
from allensdk.brain_observatory.r_neuropil import estimate_contamination_ratios
import allensdk.internal.brain_observatory.demixer as demixer
import allensdk.internal.core.lims_utilities as lu
import visual_behavior.ophys.mesoscope.crosstalk_unmix as ica
import visual_behavior.ophys.mesoscope.dataset as ms
import logging
import os
import shutil
import time
import sciris as sc
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
CACHE = '/media/rd-storage/Z/MesoscopeAnalysis/'
DEFAULT_DATABASE = 'lims2'
DEFAULT_HOST = 'limsdb2'
DEFAULT_PORT = 5432
DEFAULT_USERNAME = 'limsreader'
PW = 'limsro'


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


def psycopg2_select(query, database=DEFAULT_DATABASE, host=DEFAULT_HOST, port=DEFAULT_PORT, username=DEFAULT_USERNAME,
                    password=PW):
    connection = psycopg2.connect(
        host=host, port=port, dbname=database, user=username, password=password,
        cursor_factory=psycopg2.extras.RealDictCursor
    )
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        response = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()
    return response


def get_all_mesoscope_files():
    query = (""" SELECT
    e.name as rig_name, 
    p.code as project_code, 
    os.id AS ses_id, 
    os.date_of_acquisition as date, 
    oe.id AS exp_id, 
    oe.storage_directory as exp_dir,
    wkft.name AS wkf_type, 
    wkf.storage_directory as movie_dir, 
    wkf.filename as movie_name
    FROM ophys_sessions os
    JOIN ophys_experiments oe ON os.id = oe.ophys_session_id
    JOIN projects p ON p.id = os.project_id
    JOIN equipment e ON e.id = os.equipment_id
    JOIN well_known_files wkf ON wkf.attachable_id = oe.id
    JOIN welL_known_file_types wkft ON wkft.id = wkf.welL_known_file_type_id AND wkft.name = 'MotionCorrectedImageStack'
    WHERE e.name = 'MESO.1'""")
    return pd.DataFrame(psycopg2_select(query))


def get_mesoscope_exp_files(exp_id):
    query = (f""" SELECT
    e.name as rig_name, 
    p.code as project_code, 
    os.id AS ses_id, 
    os.date_of_acquisition as date, 
    oe.id AS exp_id, 
    oe.storage_directory as exp_dir,
    wkft.name AS wkf_type, 
    wkf.storage_directory as movie_dir, 
    wkf.filename as movie_name
    FROM ophys_sessions os
    JOIN ophys_experiments oe ON os.id = oe.ophys_session_id
    JOIN projects p ON p.id = os.project_id
    JOIN equipment e ON e.id = os.equipment_id
    JOIN well_known_files wkf ON wkf.attachable_id = oe.id
    JOIN welL_known_file_types wkft ON wkft.id = wkf.welL_known_file_type_id AND wkft.name = 'MotionCorrectedImageStack'
    WHERE e.name = 'MESO.1' AND oe.id = {exp_id}""")
    return pd.DataFrame(psycopg2_select(query))


def get_all_mesoscope_data():
    query = ("select os.id as session_id, oe.id as experiment_id, "
             "os.storage_directory as session_folder, "
             "oe.storage_directory as experiment_folder, "
             "sp.name as specimen, "
             "os.date_of_acquisition as date, "
             "oe.workflow_state as exp_workflow_state, "
             "os.workflow_state as session_workflow_state "
             "from ophys_experiments oe "
             "join ophys_sessions os on os.id = oe.ophys_session_id "
             "join specimens sp on sp.id = os.specimen_id "
             "join projects p on p.id = os.project_id "
             "where (p.code = 'VisualBehaviorMultiscope' "
             "or p.code = 'VisualBehaviorMultiscope4areasx2d' "
             "or p.code = 'MesoscopeDevelopment') and os.workflow_state ='uploaded' "
             "order by session_id")
    return pd.DataFrame(psycopg2_select(query))


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
    return


def get_ica_done_sessions(session_list=None):
    """
    function to scan all LIMS sessions nad return lists of ones that have been run through crosstalk demixing successfully, and that have not.
    as well as all mesoscope data found in lime
    :return: [pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
    """
    if not session_list:
        meso_data = get_all_mesoscope_data()
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
        meso_data = get_all_mesoscope_data()
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
        meso_data = get_all_mesoscope_data()
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


def get_exp_by_cre_line(cre_line):
    """
    helper function to get ica experiments by cre line: i.e Vip-IRES-Cre
    :param cre_line: str, cre line name from LIMS
    :return: list of mesoscope experiments for cre line
    """
    md = get_all_mesoscope_data()
    exp_cre = list(md.loc[md['specimen'].str.contains(cre_line)]['experiment_id'].drop_duplicates())
    return exp_cre


def get_ses_by_cre_line(cre_line):
    """
    helper function to get ica sessions by cre line
    :param cre_line: str, cre line name from LIMS, i.e. Vip-IRES-Cre
    :return: list of mesoscope sessoins for cre line
    """
    md = get_all_mesoscope_data()
    ses_cre = list(md.loc[md['specimen'].str.contains(cre_line)]['session_id'].drop_duplicates())
    return ses_cre


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


def run_demixing_on_session(session, cache=CACHE, roi_name=None):
    """
    run LIMS demixing on crosstalk corrected traces
    :param roi_name: prefix to use for roi ica dir
    :param session: LIMS session id
    :param cache: directory containing crosstalk corrected traces
    :return:
    """
    if roi_name is None:
        roi_name = "ica_traces"

    dataset = ms.MesoscopeDataset(session)
    pairs = dataset.get_paired_planes()

    # if self.debug_mode:
    #     logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # else:
    #     logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    for pair in pairs:
        for exp_id in pair:
            exp_files_data = get_mesoscope_exp_files(exp_id)
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
                    ica_dir = f'session_{session}/{roi_name}_{pair[0]}_{pair[1]}/'
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


def run_neuropil_correction_on_ica(session, ica_cache_dir=CACHE, roi_name=None, np_name=None):
    """
    run neuropil correction on CIA output files
    :param session: LIMS session id
    :param ica_cache_dir: directory to find processed ica-demixed outputs
    :param roi_name: prefix to use for roi ica dir
    :param np_name: prefix to use for neuropil ica dir
    :return: None
    """
    if roi_name is None:
        roi_name = "ica_traces"

    if np_name is None:
        np_name = 'ica_neuropil'

    ica_obj = ica.MesoscopeICA(session_id=session, cache=ica_cache_dir, roi_name=roi_name, np_name=np_name)
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
    """
    fn to check if file was last modified before specific date
    :param file_date:
    :param thr_date:
    :return:
    """
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


def clean_up_session(session,
                     remove_raw_traces=False,
                     remove_valid=False, remove_ica_input=False, remove_ica_output=False, remove_crosstalk=False):
	ica_o = ica.MesoscopeICA(session_id=session, cache=CACHE)
	os.chdir(ica_o.session_dir)
	if remove_raw_traces:
	    sc.runcommand('rm -rf */*_raw.h5')
	if remove_ica_input:
		sc.runcommand('rm -rf */*_in.h5')
	if remove_valid:
		sc.runcommand('rm -rf */*.json')
	if remove_ica_output:
		sc.runcommand('rm -rf */*_out.h5')
	if remove_crosstalk:
	    sc.runcommand('rm -rf */*_ct.h5')
	    sc.runcommand('rm -rf *_ct.h5')


def delete_old_files(sessions, ch, names_files=None, remove_by_date='04/01/2020'):
    if not names_files:
        names_files = {'roi': ['ica_traces', 'traces_ica', 'roi_traces', 'traces_roi', 'crosstalk', 'mixing'],
                       'np': ['ica_neuropil', 'neuropil_ica', 'crosstalk', 'mixing']}
    for session in sessions:
        ica_obj = ica.MesoscopeICA(session_id=session, cache=ch, roi_name="ica_traces", np_name="ica_neuropil")
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
    """
    this fn adds offset to the outputs is they where written to disk debiased
    :param sessions:
    :return:
    """
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
                    new_path = os.path.join(self.dirs[tkey], f"{self.exp_ids[pkey]}_out_1.h5")
                    # copy valid json to the new filename
                    if os.path.isfile(new_path):
                        print(f"skipping experint {self.exp_ids[pkey]}")
                        continue  # skipping to next exp
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
                                f.create_dataset(f"data", data=self.outs[pkey][tkey], compression="gzip")
                                f.create_dataset(f"crosstalk", data=self.crosstalk[pkey][tkey])
                                f.create_dataset(f"mixing_matrix_adjusted", data=self.a_mixing[pkey][tkey])
                                f.create_dataset(f"mixing_matrix", data=self.mixing[pkey][tkey])
                        else:
                            print(f"input or putput for exp {self.exp_ids[pkey]}, {tkey} doesn't exist")
    return


def plot_ica_traces(sig, ct, name, title):
    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 22})
    plt.subplot(211)
    plt.ylim(min(min(sig), min(ct)), max(max(sig), max(ct)))
    plt.plot(sig, 'r-', label='signal pl')
    plt.plot(ct, 'g-', label='cross-talk pl')
    plt.title(f'{title} for cell {name}', fontsize=18)
    plt.legend(loc='best')
    return


def get_all_rois_crosstalk(session_list=None):
    """
    fn to collect crosstalk information from all ica-processed sessions
    :param session_list: one can supply a list of sessions, if not - we will find all ica-done sessions and extract crosstalk data
    :return: rois_crosstalk_list (dict): {'session_id' : {'roi_id' : {'crosstalk_before' : float}, {'crosstalk_after': float}, ...}, ...}
             rois_crosstalk_before_list (list): [[array], [array], ...] each array is crosstalk before demixing values for experiment
             rois_crosstalk_after_list (list): [[array], [array], ...] each array is crosstalk after demixing values for experiment
    """
    if not session_list:
        sessions = get_ica_done_sessions()
    else:
        sessions = session_list
    rois_crosstalk_dict = {}
    rois_crosstalk_before_list = []
    rois_crosstalk_after_list = []
    for session in sessions:
        rois_crosstalk_dict[session] = {}
        ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
        pairs = ica_obj.dataset.get_paired_planes()
        for pair in pairs:
            ica_obj.set_exp_ids(pair)
            ica_obj.set_ica_dirs()
            ica_obj.set_ica_input_paths()
            ica_obj.set_out_paths()
            ica_obj.set_valid_paths()
            for pkey in ica_obj.pkeys:
                tkey = 'roi'
                # read ica input, output and valid json data
                with h5py.File(ica_obj.outs_paths[pkey][tkey], "r") as f:
                    ica_obj.crosstalk[pkey][tkey] = f["crosstalk"][()]
                rois_valid = ju.read(ica_obj.rois_valid_paths[pkey][tkey])
                crosstalk = ica_obj.crosstalk[pkey][tkey]
                crosstalk_before = crosstalk[0]
                crosstalk_after = crosstalk[1]
                rois_crosstalk_before_list.append(crosstalk_before)
                rois_crosstalk_after_list.append(crosstalk_after)
                roi_names = [roi for roi, valid in rois_valid.items() if valid]
                for i in range(len(roi_names)):
                    roi_name = roi_names[i]
                    rois_crosstalk_dict[session][roi_name] = {'crosstalk_before': crosstalk_before[i], 'crosstalk_after': crosstalk_after[i]}

    return rois_crosstalk_dict, rois_crosstalk_before_list, rois_crosstalk_after_list


def get_errors_lims(sessions):
    lims_fail_log = {}
    for session in sessions:
        try:
            print(f"Running neuropil correction on {session}")
            run_neuropil_correction_on_ica(str(session))
        except Exception as e:
            print(f"Error in np correction, session: {session}")
            lims_fail_log[session] = {}
            lims_fail_log[session] = {'proccess': 'neuropil_correcton'}, {'error': e}
        else:
            print(f"Finihsed np correction  for session {session}")

            try:
                print(f"Calculating dff for {session}")
                run_dff_on_ica(str(session))
            except Exception as e:
                print(f"Error in dff calculation session: {session}")
                lims_fail_log[session] = {}
                lims_fail_log[session] = {'proccess': 'calculating dff'}, {'error': e}
            else:
                print(f"Finihsed dff calculation for session {session}")
    return lims_fail_log


def del_files_with_str(sessions, ext):
    """
    :param sessions: list of sessons to go through and clean up
    :param ext: list of strings to look int he filenames for deletion
    :return:
    """
    for session in sessions:
        ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE)
        pairs = ica_obj.dataset.get_paired_planes()
        for pair in pairs:
            ica_obj.set_exp_ids(pair)
            ica_obj.set_ica_dirs()
            for tkey in ica_obj.tkeys:
                os.chdir(ica_obj.dirs[tkey])
                for ext_i in ext:
                    found_out = sc.runcommand(f'find *{ext_i}*')
                    if "No such file" not in found_out:
                        print(f"Found files:{found_out}Deleting")
                        sc.runcommand(f"rm -rf *{ext_i}*")


def refactor_outputs_to_lims(sessions):
    for session in sessions:
        logging.info(f"Processing session {session}")
        ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
        pairs = ica_obj.dataset.get_paired_planes()
        for pair in pairs:
            ica_obj.set_exp_ids(pair)
            ica_obj.get_ica_traces()
            ica_obj.validate_traces(return_vba=False)
            ica_obj.debias_traces()
            ica_obj.unmix_pair()
            for pkey in ica_obj.pkeys:
                rois_valid = [i for i, valid in ica_obj.rois_valid[pkey].items() if valid]
                num_rois_valid = len(rois_valid)
                for tkey in ica_obj.tkeys:
                    num_outputs = ica_obj.outs[pkey][tkey][0].shape[0]
                    assert num_outputs == num_rois_valid, "Number of outputs not equal to number of valid roi names"
                    # make new output: add _lims at the end
                    lims_path = add_suffix_to_path(ica_obj.outs_paths[pkey][tkey], '_lims')
                    if not os.path.isfile(lims_path):
                        with h5py.File(lims_path, "w") as f:
                            f.create_dataset("data", data=ica_obj.outs[pkey][tkey], compression="gzip")
                            f.create_dataset("roi_names", data=[np.string_(rn) for rn in ica_obj.rois_names_valid[pkey][tkey]])
                            f.create_dataset("crosstalk", data=ica_obj.crosstalk[pkey][tkey])
                            f.create_dataset("mixing_matrix_adjusted", data=ica_obj.a_mixing[pkey][tkey])
                            f.create_dataset("mixing_matrix", data=ica_obj.mixing[pkey][tkey])
                    else:
                        logging.info(f"Lims output exists for exp {ica_obj.exp_ids[pkey]}")
        del ica_obj
    return


def add_suffix_to_path(abs_path, suffix):
    file_ext = os.path.splitext(abs_path)[1]
    file_name = os.path.splitext(abs_path)[0]
    new_file_name = f"{file_name}{suffix}{file_ext}"
    return new_file_name


def filter_outputs_crosstalk(session):
    # read traces for these rois:
    ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
    pairs = ica_obj.dataset.get_paired_planes()
    for pair in pairs:
        ica_obj.set_exp_ids(pair)
        ica_obj.get_ica_traces()
        ica_obj.validate_traces(return_vba=False)
        ica_obj.debias_traces()
        ica_obj.unmix_pair()
        ica_obj.validate_cells_crosstalk()
        ica_obj.outs_ct = {}
        ica_obj.roi_names_valid_ct = {}
        ica_obj.outs_ct_path = {}
        for pkey in ica_obj.pkeys:
            ica_obj.outs_ct[pkey] = {}
            ica_obj.roi_names_valid_ct[pkey] = {}
            ica_obj.outs_ct_path[pkey] = {}
            rois_valid_ct = ica_obj.rois_valid_ct[pkey]
            for tkey in ica_obj.tkeys:
                sig_outs = ica_obj.outs[pkey][tkey][0]
                ct_outs = ica_obj.outs[pkey][tkey][1]
                traces_dict = {}
                if ct_outs.shape[0] == len(ica_obj.rois_names_valid[pkey][tkey]):
                    for i in range(len(ica_obj.rois_names_valid[pkey][tkey])):
                        roi_name = ica_obj.rois_names_valid[pkey][tkey][i]
                        traces_dict[roi_name] = {}
                        traces_dict[roi_name]['sig'] = sig_outs[i]
                        traces_dict[roi_name]['ct'] = ct_outs[i]
                traces_out_sig = [traces['sig'] for roi_name, traces in traces_dict.items() if
                                  rois_valid_ct[str(roi_name)]]
                traces_out_ct = [traces['ct'] for roi_name, traces in traces_dict.items() if
                                 rois_valid_ct[str(roi_name)]]
                traces_out = np.array([traces_out_sig, traces_out_ct])
                ct_outs_path = add_suffix_to_path(ica_obj.outs_paths[pkey][tkey], '_ct')
                roi_names_valid_ct = [roi_name for roi_name, valid in ica_obj.rois_valid_ct[pkey].items() if valid]
                if not os.path.isfile(ct_outs_path):
                    with h5py.File(ct_outs_path, "w") as f:
                        f.create_dataset("data", data=traces_out, compression="gzip")
                        f.create_dataset("roi_names", data=[str(rn) for rn in roi_names_valid_ct])
                        f.create_dataset("crosstalk", data=ica_obj.crosstalk[pkey][tkey])
                        f.create_dataset("mixing_matrix_adjusted", data=ica_obj.a_mixing[pkey][tkey])
                        f.create_dataset("mixing_matrix", data=ica_obj.mixing[pkey][tkey])
                else:
                    with h5py.File(ct_outs_path, "r") as f:
                        traces_out = f["data"][()]
                        roi_names_valid_ct = f["roi_names"][()]
                        ica_obj.crosstalk[pkey][tkey] = f["crosstalk"][()]
                        ica_obj.a_mixing[pkey][tkey] = f["mixing_matrix_adjusted"]
                        ica_obj.mixing[pkey][tkey] = f["mixing_matrix"]

                ica_obj.outs_ct[pkey][tkey] = traces_out
                ica_obj.roi_names_valid_ct[pkey][tkey] = roi_names_valid_ct
                ica_obj.outs_ct_path[pkey][tkey] = ct_outs_path
    return ica_obj


def rename_old_traces(sessions):
    raw_names = {"roi" : "traces_original", "np": "neuropil_original"}
    list_ses_renamed = []
    for session in sessions:
        found_renaming = False
        ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE)
        pairs = ica_obj.dataset.get_paired_planes()
        for pair in pairs:
            print(f"Processing pair: {pair}")
            ica_obj.set_exp_ids(pair)
            ica_obj.set_ica_dirs()

            for tkey in ica_obj.tkeys:
                if os.path.isdir(ica_obj.dirs[tkey]):
                    for pkey in ica_obj.pkeys:
                            os.chdir(ica_obj.dirs[tkey])
                            old_raw_name = f"{raw_names[tkey]}_{ica_obj.exp_ids[pkey]}.h5"
                            if os.path.isfile(old_raw_name):
                                print(f"Found old file names: {old_raw_name}")
                                sc.runcommand(f"cp {old_raw_name} {ica_obj.exp_ids[pkey]}_raw.h5")
                                sc.runcommand(f"rm -rf {old_raw_name}")
                                found_renaming = True
                else:
                    ica_obj.set_ica_dirs(names={'roi': "roi", 'np': "neuropil"})
                    for pkey in ica_obj.pkeys:
                        for tkey in ica_obj.tkeys:
                            os.chdir(ica_obj.dirs[tkey])
                            old_raw_name = f"{raw_names[tkey]}_{ica_obj.exp_ids[pkey]}.h5"
                            if os.path.isfile(old_raw_name):
                                print(f"Found old file names: {old_raw_name}")
                                sc.runcommand(f"cp {old_raw_name} {ica_obj.exp_ids[pkey]}_raw.h5")
                                sc.runcommand(f"rm -rf {old_raw_name}")
                                found_renaming = True
        if found_renaming:
            list_ses_renamed.append(session)

    return list_ses_renamed


def plot_traces(x1, x1_name, x2, x2_name, roi_id, title, c_x1='red', c_x2='green', r_ica=None, r_raw=None,
                rmse_ica=None, rmse_raw=None):
    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 22})
    plt.subplot(211)
    plt.ylim(min(min(x1), min(x2)), max(max(x1), max(x2)))
    plt.plot(x1, c_x1, label=x1_name, alpha=0.7)
    plt.plot(x2, c_x2, label=x2_name, alpha=0.7)
    plt.title(f'{title} for cell {roi_id}', fontsize=18)
    if r_ica is not None and r_raw is not None:
        ax = plt.gca()
        plt.text(0, 0.8, f" r raw: {np.round(r_raw, 2)}\n r ica: {np.round(r_ica, 2)}", transform=ax.transAxes)
        plt.text(0, 0.6, f"mean d: {np.round(x2.mean() * r_raw - x1.mean() * r_ica, 2)}", transform=ax.transAxes)

    if r_ica is not None and r_raw is not None:
        ax = plt.gca()
        plt.text(0.2, 0.8, f" rmse raw: {np.round(rmse_raw, 2)}\n rmse ica: {np.round(rmse_ica, 2)}",
                 transform=ax.transAxes)
    plt.legend(loc='best')
    return


def plot_trace(x1, x1_name, roi_id, title, c_x1 = 'blue'):
    plt.figure(figsize=(20, 5))
    plt.rcParams.update({'font.size': 22})
    plt.plot(x1, c_x1, label=x1_name)
    plt.title(f'{title} for cell {roi_id}', fontsize=18)
    plt.legend(loc='best')
    return


def plot_rois_from_plane(ica_obj, pkey, plot_interval='all', roi_id=None):
    """
    fn to plot either all rois from plane that are valid (Accroding to ica_obj.rois_names_valid_ct)
    or a single roi
    """
    tkey = 'roi'
    session = ica_obj.session_id
    exp_id = ica_obj.exp_ids[pkey]
    traces_raw_sig_roi = {}
    traces_out_sig_roi = {}

    roi_names = ica_obj.rois_names[pkey][tkey]
    roi_names_valid = ica_obj.rois_names_valid[pkey][tkey]
    roi_names_valid_ct = ica_obj.rois_names_valid_ct[pkey][tkey]
    raws_sig_roi = ica_obj.raws[pkey][tkey][0]
    ica_outs_sig_roi = ica_obj.outs[pkey][tkey][0]

    if plot_interval == 'all':
        start = 0
        end = raws_sig_roi.shape[1]
    else:
        start = plot_interval[0]
        end = plot_interval[1]

    assert len(roi_names) == raws_sig_roi.shape[0], f"Raw traces sig not aligned for exp {exp_id}"
    assert len(roi_names_valid) == ica_outs_sig_roi.shape[0], f"ICA outs sig is not aligned for exp {exp_id}"

    # create dictionaries of raw and ica output traces
    i = 0
    for roi in roi_names:
        traces_raw_sig_roi[str(roi)] = raws_sig_roi[i][start:end]
        i += 1
    i = 0
    for roi in roi_names_valid:
        traces_out_sig_roi[str(roi)] = ica_outs_sig_roi[i][start:end]
        i += 1

    for roi in roi_names_valid_ct:
        if roi_id is None:  # plot all roi traces
            plot_traces(traces_raw_sig_roi[str(roi)], "raw signal roi", traces_out_sig_roi[str(roi)], "ica signal roi",
                        roi, f"{session}/{exp_id}: ROI traces, signal: RAW and after ICA")
        elif roi == str(roi_id):  # plot only roi of interest:
            plot_traces(traces_raw_sig_roi[str(roi)], "raw signal roi", traces_out_sig_roi[str(roi)], "ica signal roi",
                        roi, f"{session}/{exp_id}: ROI traces, signal: RAW and after ICA")
        else:
            continue
    return roi_names_valid_ct
