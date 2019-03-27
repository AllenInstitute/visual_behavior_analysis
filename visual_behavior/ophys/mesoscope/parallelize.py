import psutil
import resource
import time
from multiprocessing import Process
import visual_behavior.ophys.mesoscope.utils as mu
import visual_behavior.ophys.mesoscope.mesoscope as ms

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

while len(sessions) > 0:
    for session in sessions:
        if thread_count > 0:
            print(f'number of active threads : {thread_count}')
            p = Process(target=mu.run_ica_on_session, args=(session, meso_data,))
            print(f'processing session :{session}')
            p.daemon = True
            p.start()
            process_name.append([p.pid])
            process_status.append([p.is_alive()])
            thread_count -= 1
            sessions = sessions.drop(sessions.index[sessions == session])
            print(f'remaining session: {len(sessions)}')
        else:
            if process_status.count(True) == thread_count:
                time.sleep(0.25)
                # update current process statuses here:
                process_status = []
                for pid in process_name:
                    process_status.append(psutil.pid_exists(pid[0]))
            else:
                thread_count = thread_count + process_status.count(False)
