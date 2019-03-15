import psutil
import resource
import time
from multiprocessing import Process
import visual_behavior.ophys.mesoscope.crosstalk_unmix as ica
import visual_behavior.ophys.mesoscope.mesoscope as ms
import os

meso_data = ms.get_all_mesoscope_data()


def run_ica_on_session(session):
    ica_obj = ica.Mesoscope_ICA(session_id=session, cache='/media/NCRAID/MesoscopeAnalysis/')
    pairs = ica_obj.dataset.get_paired_planes()
    for pair in pairs:
        ica_obj.get_ica_traces(pair)
        ica_obj.combine_debias_traces()
        ica_obj.unmix_traces()
        if ica_obj.found_solution:
            meso_data['ICA_demix'].loc[meso_data['experiment_id'] == pair[0]] = 0
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
                p = Process(target=run_ica_on_session, args=(session,))
                p.daemon = True
                p.start()
                process_name.append([p.pid])
                process_status.append([p.is_alive])
                thread_count = -1
        if process_status.count(True) == thread_count:
            time.sleep(0.25)
            # update current process statuses here:
            process_status = []
            for pid in process_name:
                process_status.append(psutil.pid_exists(pid))
        else:
            thread_count = thread_count + process_status.count(False)


def get_ica_sessions(sessions):
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