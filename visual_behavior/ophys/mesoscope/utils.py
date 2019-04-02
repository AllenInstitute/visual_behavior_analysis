import logging
import os
import allensdk.internal.brain_observatory.demixer as demixer
import allensdk.internal.core.lims_utilities as lu
import h5py
import numpy as np

import visual_behavior.ophys.mesoscope.crosstalk_unmix as ica
import visual_behavior.ophys.mesoscope.mesoscope as ms

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


def run_demixing_on_ica(session, an_dir='/media/NCRAID/MesoscopeAnalysis/'):
    mds, _ = get_ica_sessions()
    dataset = ms.MesoscopeDataset(session)
    pairs = dataset.get_paired_planes()

    for pair in pairs:
        for exp_id in pair:
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

            ica_dir = f'session_{session}/ica_traces_{pair[0]}_{pair[1]}/traces_ica_output_{exp_id}.h5'
            demix_path = os.path.join(an_dir, f'session_{session}/demixing_{exp_id}')
            input_data = {
                "movie_h5": os.path.join(exp_dir, "processed", "concat_31Hz_0.h5"),
                "traces_h5_ica": os.path.join(an_dir, ica_dir),
                "traces_h5": os.path.join(exp_dir, "processed", "roi_traces.h5"),
                "roi_masks": nrois.values(),
                "output_file": os.path.join(demix_path, f"traces_demixing_output_{exp_id}.h5")
            }

            traces, masks, valid, trace_ids, movie_h5, output_h5 = parse_input(input_data)

            # only demix non-union, non-duplicate ROIs
            valid_idxs = np.where(valid)
            demix_traces = traces[valid_idxs]
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
            out_traces[:] = np.nan
            out_traces[valid_idxs] = demixed_traces

            with h5py.File(output_h5, 'w') as f:
                f.create_dataset("data", data=out_traces, compression="gzip")
                f.create_dataset("roi_names", data=[np.string_(rn) for rn in trace_ids])

    return

