#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:56:29 2019

@author: farzaneh
"""

import numpy as np
import sys
from ophysextractor.utils.util import mongo, get_psql_dict_cursor
from get_mouse_pkls import *
import pandas as pd
import os
import datetime

# %%


def set_mouse_trainHist_all2(all_mice_id, saveResults=1):

    mouse_trainHist_all2 = pd.DataFrame(
        [], columns=['mouse_id', 'date', 'stage'])

    for mouse_id in all_mice_id:
        print(mouse_id)
        # mouse_id = 449653
        pkl_all_sess = get_mouse_pkls(mouse_id)  # make a sql query from LIMS.

        # For each mouse go through all pkl files to get the stages
        mouse_trainHist2 = pd.DataFrame(
            [], columns=['mouse_id', 'date', 'stage'])  # list()
        for i in range(len(pkl_all_sess)):
            print('pickle file %d / %d' % (i, len(pkl_all_sess)))
            pkl = pd.read_pickle(pkl_all_sess[i]['pkl_path'])
            try:
                stage = pkl['items']['behavior']['params']['stage']
            except:
                stage = ''

            try:
                date = str(pkl['start_time'].date())
            except:
                date = str(pkl['startdatetime'].date())
    #            import datetime
    #            datetime.datetime.fromtimestamp(1548449865.568)

            mouse_trainHist2.at[i, ['mouse_id', 'date',
                                    'stage']] = mouse_id, date, stage

        mouse_trainHist_all2 = mouse_trainHist_all2.append(mouse_trainHist2)

    # save the variable after all mice are done!
    if saveResults:

        now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

        print('Saving .h5 file')
        dir_Farz = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'
        name = 'mouse_trainHist_all2_%s' % now
        # os.path.join(d, svmn+os.path.basename(pnevFileName))
        Name = os.path.join(dir_Farz, name + '.h5')
        print(Name)

        # Save to a h5 file
        mouse_trainHist_all2.to_hdf(Name, key='mouse_trainHist_all2', mode='w')

    return mouse_trainHist_all2


# %%
print(len(sys.argv))
print(sys.argv)

#all_mice_id = sys.argv[1]


# all_mice_id = np.array([392241, 409296, 411922, 412035, 414146, 429956, 431151, 435431,
#        438912, 440631, 447934, 448366, 449653, 451787, 453909, 453911,
#        453988, 453989, 453990, 453991, 456915, 457841, 477204, 479839,
#        482853, 484627, 485152, 488458])

# data release march 2021
all_mice_id = np.array([435431, 438912, 440631, 448366, 449653, 451787, 453911, 453988,
       453989, 453990, 453991, 456915, 457841, 479839, 482853, 484627,
       485152, 523922, 528097])


set_mouse_trainHist_all2(all_mice_id)


# %% Get stage, date info for all behavioral sessions from mongo.db.behavior_session_log


def get_stage_mongo_behavior_log(mouse_id):  # mouse_id = '456915'

    DB = mongo.db.behavior_session_log
#    db_cursor_sess = db_cursor_sess = DB.find({"id":766753713})
    db_cursor_sess = DB.find({"external_specimen_name": str(mouse_id)})

    date_now = str(db_cursor_sess[0]['created_at'])
    stage_now = db_cursor_sess[0]['stage']

    return date_now, stage_now
