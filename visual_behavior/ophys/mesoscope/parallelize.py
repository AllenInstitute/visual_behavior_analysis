import multiprocessing as mp
import visual_behavior.ophys.mesoscope.mesoscope as ms
import visual_behavior.ophys.mesoscope.utils as mu


def main ():
    meso_data = ms.get_all_mesoscope_data()
    sessions = meso_data['session_id']
    sessions = sessions.drop_duplicates()
    thread_count = 20
    pool = mp.Pool(thread_count)
    for _ in pool.imap_unordered(mu.run_ica_on_session, sessions):
        pass
if __name__ == '__main__':
    #main()
    pass
