import psutil
import resource
import time
from multiprocessing import Pool
import visual_behavior.ophys.mesoscope.utils as mu
import visual_behavior.ophys.mesoscope.mesoscope as ms



if __name__ == '__main__':

    meso_data = ms.get_all_mesoscope_data()
    meso_data['ICA_demix_exp'] = 0
    meso_data['ICA_demix_session'] = 0

    sessions = meso_data['session_id']
    sessions = sessions.drop_duplicates()

    thread_count = 20
    process_name = []
    process_status = []
    # get current limit for number of processes:
    nproc = resource.getrlimit(resource.RLIMIT_NPROC)
    if nproc[0] < nproc[1]:
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc[1] - 1000, nproc[1]))
    with Pool(thread_count) as p:
        p.map(mu.run_ica_on_session, sessions)

