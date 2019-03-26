import psutil
import resource
import time
from multiprocessing import Process
import visual_behavior.ophys.mesoscope.crosstalk_unmix as ica
import visual_behavior.ophys.mesoscope.mesoscope as ms
import allensdk.internal.core.lims_utilities as lu
import os
import h5py
import numpy as np
import allensdk.internal.brain_observatory.demixer as demixer
import logging

meso_data = ms.get_all_mesoscope_data()
meso_data['ICA_demix_exp'] = 0
meso_data['ICA_demix_session'] = 0


def run_ica_on_session(session):
    ica_obj = ica.Mesoscope_ICA(session_id=session, cache='/media/NCRAID/MesoscopeAnalysis/')
    pairs = ica_obj.dataset.get_paired_planes()
    for pair in pairs:
        ica_obj.get_ica_traces(pair)
        ica_obj.combine_debias_traces()
        ica_obj.unmix_traces()
        ica_obj.plot_ica_traces(pair)
        # if np.all(ica_obj.found_solution == True) :
        #     meso_data['ICA_demix_exp'].loc[meso_data['experiment_id'] == pair[0]] = 1
    return


def parallelize(sessions, thread_count=20):
    process_name = []
    process_status = []
    nproc = resource.getrlimit(resource.RLIMIT_NPROC)
    if nproc[0] < nproc[1]:
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc[1] - 1000, nproc[1]))

    while len(sessions) > 0:
        while thread_count > 0:
            for session in sessions:
                p = Process(target=run_ica_on_session, args=(session, ))
                p.daemon = True
                p.start()
                process_name.append([p.pid])
                process_status.append([p.is_alive])
                thread_count = -1
                sessions = sessions.drop(sessions.index[sessions == session])
        if process_status.count(True) == thread_count:
            time.sleep(0.25)
            # update current process statuses here:
            process_status = []
            for pid in process_name:
                process_status.append(psutil.pid_exists(pid))
        else:
            thread_count = thread_count + process_status.count(False)


def get_ica_sessions(sessions):
    meso_data = ms.get_all_mesoscope_data()
    meso_data['ICA_demix_exp'] = 0
    meso_data['ICA_demix_session'] = 0

    for session in sessions:
        dataset = ms.MesoscopeDataset(session)
        pairs = dataset.get_paired_planes()
        for pair in pairs:
            ica_obj = ica.Mesoscope_ICA(session, cache='/media/NCRAID/MesoscopeAnalysis')
            ica_obj.set_ica_traces_dir(pair)
            ica_obj.plane1_ica_output_pointer
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


def run_demixing_on_ica(exp_id, an_dir='/media/NCRAID/MesoscopeAnalysis/'):
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
    md = ms.get_all_mesoscope_data()
    s = md.session_id
    us = s.drop_duplicates()
    mds, _ = get_ica_sessions(us)
    exp_dir = mds['experiment_folder'].loc[mds['experiment_id'] == exp_id].values[0]

    rois = f'traces_ica_output_{exp_id}.h5'

    input_data = {
        "movie_h5": os.path.join(exp_dir, "processed", "concat_31Hz_0.h5"),
        "traces_h5_ica": os.path.join(an_dir, rois),
        "traces_h5": os.path.join(exp_dir, "processed", "roi_traces.h5"),
        "roi_masks": nrois.values(),
        "output_file": os.path.join(an_dir, f"traces_ica_output_{exp_id}.h5")
    }
    traces, masks, valid, trace_ids, movie_h5, output_h5 = parse_input(input_data)

    # only demix non-union, non-duplicate ROIs
    valid_idxs = np.where(valid)
    demix_traces = traces[valid_idxs]
    demix_masks = masks[valid_idxs]

    with h5py.File(movie_h5, 'r') as f:
        movie = f['data'].value

    demixed_traces, drop_frames = demixer.demix_time_dep_masks(demix_traces, movie, demix_masks)

    demix_path = os.path.join(exp_dir, f'demixing_{exp_id}')
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
    logging.debug("rois with negative baselines (or overlap with them): %s", str(trace_ids[valid_idxs][nb_inds]))
    demixed_traces[nb_inds, :] = np.nan
    logging.info("Saving output")
    out_traces = np.zeros(traces.shape, dtype=demix_traces.dtype)
    out_traces[:] = np.nan
    out_traces[valid_idxs] = demixed_traces

    with h5py.File(output_h5, 'w') as f:
        f.create_dataset("data", data=out_traces, compression="gzip")
        f.create_dataset("roi_names", data=[str(rn) for rn in trace_ids])
    return


def parse_input(data, exclude_labels=["union", "duplicate", "motion_border"]):
    movie_h5 = get_path(data, "movie_h5", True)
    traces_h5 = get_path(data, "traces_h5", True)
    traces_h5_ica = get_path(data, "traces_h5_ica", True)
    output_h5 = get_path(data, "output_file", False)

    with h5py.File(movie_h5, "r") as f:
        movie_shape = f["data"].shape[1:]

    with h5py.File(traces_h5_ica, "r") as f:
        traces = f["data"].value
    traces = traces[0, :, :]

    with h5py.File(traces_h5, "r") as f:
        trace_ids = [int(rid) for rid in f["roi_names"].value]

    rois = get_path(data, "roi_masks", False)
    masks = None
    valid = None

    for roi in rois:
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

        valid[ridx] = len(set(exclude_labels) & set(roi.get("exclusion_labels", []))) == 0

    return traces, masks, valid, np.array(trace_ids), movie_h5, output_h5


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


def run_neuropil_s_on_exp(exp_id):
    return
