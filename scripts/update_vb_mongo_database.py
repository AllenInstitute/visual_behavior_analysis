#!/usr/bin/env python

import visual_behavior.database as db
from multiprocessing import Pool
import pandas as pd


def get_sessions_to_load():
    vb = db.Database('visual_behavior_data')
    ms = db.Database('mouseseeks')

    vb_summary = vb.query('behavior_data', 'summary')
    ms_sessions = ms.query('db', 'behavior_session_log')
    merged = ms_sessions[['foraging_id']].merge(
        vb_summary[['behavior_session_uuid']],
        left_on='foraging_id',
        right_on='behavior_session_uuid',
        how='outer'
    )
    merged['to_load'] = (pd.isnull(merged['behavior_session_uuid'])) & (~pd.isnull(merged['foraging_id']))

    vb.close()
    ms.close()

    return merged.query('to_load == True')['foraging_id'].drop_duplicates()


def add_session_to_db(session_id):
    print('adding session {}'.format(session_id))
    try:
        db.add_behavior_record(session_id)
    except Exception as e:
        print('\terror on {}'.format(session_id))
        print('\t{}'.format(e))


if __name__ == '__main__':
    print("getting list of sessions in mouseseeks that aren't also in the VB database")
    # get all sessions that are listed in mouseseeks but not in the VB database
    to_load = get_sessions_to_load()
    print('about to process {} session data files'.format(len(to_load)))

    pool = Pool(16)
    ans = pool.map(add_session_to_db, to_load.values[::-1])
