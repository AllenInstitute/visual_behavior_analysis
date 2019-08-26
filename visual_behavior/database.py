from pymongo import MongoClient
import yaml
import pandas as pd
import numpy as np
import json
import os
import glob
import traceback
import datetime
from .translator.foraging2 import data_to_change_detection_core
from .translator.core import create_extended_dataframe
from .change_detection.trials import summarize

class Database(object):
    '''
    utilities for connecting to MongoDB databases (mouseseeks or visual_behavior_data)

    parameter:
      database: defines database to connect to. Can be 'mouseseeks' or 'visual_behavior_data'
    '''

    def __init__(self, database, ):
        # get database ip/port info from a text file on the network (maybe not a good idea to commit it)
        db_info_filepath = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/mongo_db_info.yml'
        with open(db_info_filepath, 'r') as stream:
            db_info = yaml.safe_load(stream)

        # connect to the client
        ip = db_info[database]['ip']
        port = db_info[database]['port']
        self.client = MongoClient('mongodb://{}:{}'.format(ip,port))

        # set each table as an attribute of the class (but not admin) and as an entry in a dictionary
        # this will provide flexibility in how the tables are called
        self.database = {}
        self.database_names = []
        for db in [db for db in self.client.database_names() if db != 'admin']:
            self.database_names.append(db)
            self.database[db] = self.client[db]
            setattr(self,db, self.client[db])

    def query(self,database,collection,query={},return_as='dataframe'):
        '''
        Run a query on a collection in the database.
        The query should be formated as set of key/value pairs
        Sending an empty query will return the entire collection 
        '''

        return pd.DataFrame(list(self.database[database][collection].find(query)))

    def close(self):
        '''
        close connection to client
        '''
        self.client.close()


def get_behavior_data(table_name, session_id=None, id_type='behavior_session_uuid'):
    '''
    look up behavior data for a given behavior session
    look up by behavior_session_uuid or lims_id
    available data types:
        trials = all trials for the session
        time = time vector
        running = encoder data
        licks = all licks
        rewards = all rewards
        visual_stimuli = flash-wise stimulus table
        omitted_stimuli = table of omitted stimuli
        metadata = session metadata
        log = error log from VBA
        
        all = a dictionary with each of the above table names as a key and the associated table as the value
    '''
    db = Database('visual_behavior_data')
    session_id, id_type = _check_name_schema('visual_behavior_data', session_id, id_type)
    nested_tables = ['time','running','licks','rewards','visual_stimuli','omitted_stimuli'] #,'metadata','log']

    if table_name.lower() in nested_tables:
        # these collections have the data of interest in a document called 'data'
        return _get_table(db, table_name, session_id, id_type)
    elif table_name.lower() == 'trials':
        # trials is a flat collection: each entry is a row in the desired trials table
        return _get_trials(db, table_name, session_id, id_type)
    elif table_name.lower() == 'all':
        res = {'trials':_get_trials(db, table_name, session_id, id_type)}
        for table_name in nested_tables:
            res[table_name] = _get_table(db, table_name, session_id, id_type)
        return res

    db.close()
    
def _check_name_schema(database,session_id,id_type):
    '''
    lims_id should be int
    behavior_session_uuid should be string
    
    in mouseseeks, behavior_session_uuid is foraging_id
    '''
    if id_type == 'lims_id':
        session_id = int(session_id)
    elif id_type == 'behavior_session_uuid' and database == 'mouseseeks':
        id_type = 'foraging_id' # the name is different in mouseseeks
        
    return session_id, id_type

def _get_table(db, table_name, session_id=None, id_type='behavior_session_uuid'):
    session_id, id_type = _check_name_schema('visual_behavior_data', session_id, id_type)
    return pd.DataFrame(list(db.behavior_data[table_name].find({id_type:session_id}))[0]['data'])

def _get_trials(db, table_name, session_id=None, id_type='behavior_session_uuid'):
    '''
    get trials table for a given session
    '''
    session_id, id_type = _check_name_schema('visual_behavior_data', session_id, id_type)
    return pd.DataFrame(list(db.behavior_data[table_name].find({id_type:session_id})))

def get_pkl_path(session_id=None, id_type='behavior_session_uuid'):
    '''
    get the path to a pkl file for a given session
    '''
    session_id, id_type = _check_name_schema('mouseseeks', session_id, id_type)
        
    db = Database('mouseseeks')
    res = db.query('db','behavior_session_log',query={id_type:session_id})
    db.close()
    
    if res is not None and len(res) > 0:
        if len(res) == 1:
            storage_directory = res['storage_directory'].item()
        elif len(res) > 1:
            # there are occassionally duplicate entries in mouseseeks. return only the first:
            storage_directory = res['storage_directory'].iloc[0]
        pkls = glob.glob(os.path.join(storage_directory,'*.pkl'))
        if len(pkls) == 1:
            return pkls[0]
        
    
def uuid_to_lims_id(uuid):
    '''translates uuid to lims_id'''
    db = Database('mouseseeks')
    res = db.query('db','behavior_session_log',query={'foraging_id':uuid})
    db.close
    
    if len(res) == 0:
        return None
    elif len(res) == 1:
        return int(res['id'].item())
    elif len(res) > 1:
        return int(res['id'].iloc[0])
    
def is_int(n):
    return isinstance(n, (int, np.integer))

def is_float(n):
    return isinstance(n, (float, np.float))
    
def add_behavior_record(behavior_session_uuid, overwrite=False,db_connection=None):
    '''
    for a given behavior_session_uuid:
      - opens the data with VBA
      - adds a row to the visual_behavior_data database summary
      - adds an entry to each of:
        - running
        - licks
        - time
        - rewards
        - visual_stimuli
        - omitted_stimuli
        - metadata
        - log
    '''
    
    def insert(db, table, lims_id, behavior_session_uuid, dtype, data_to_insert):
        db[table].insert_one({
            'lims_id':lims_id,
            'behavior_session_uuid':behavior_session_uuid,
            'dtype':dtype,
            'data':data_to_insert
        })
        
    def insert_summary_row(summary_row):
        if type(summary_row) == pd.DataFrame:
            summary_row = summary_row.iloc[0].to_dict()
        # cast to simple int or float
        summary_row = {k:(int(v) if is_int(v) else v) for k,v in summary_row.items()}
        summary_row = {k:(float(v) if is_float(v) else v) for k,v in summary_row.items()}
        db.summary.insert_one(summary_row)
        
    def add_metadata_to_summary(summary, lims_id, pkl_path, behavior_session_uuid):
        summary['lims_id'] = lims_id
        summary['pkl_path'] = pkl_path
        summary['behavior_session_uuid'] = behavior_session_uuid
        return summary
    
    def log_error_data(err):
        tb = traceback.format_tb(err.__traceback__)
        db['error_log'].insert_one({
            'lims_id':lims_id,
            'behavior_session_uuid':behavior_session_uuid,
            'error_log':tb,
            'local_time':str(datetime.datetime.now()),
            'utc_time':str(datetime.datetime.utcnow())
        })
        print('error on session {}, writing to error table'.format(behavior_session_uuid))
        summary = {}
        summary = add_metadata_to_summary(summary, lims_id, pkl_path, behavior_session_uuid)
        summary['error_on_load'] = True
        insert_summary_row(summary)
        
    if db_connection is None:
        db_conn = Database('visual_behavior_data')
        db = db_conn.behavior_data
    else:
        db = db_connection
    
    # get lims ID
    lims_id = uuid_to_lims_id(behavior_session_uuid)
    
    # load behavior data
    try:
        pkl_path = get_pkl_path(behavior_session_uuid)
        data = pd.read_pickle(pkl_path)
        core_data = data_to_change_detection_core(data)
        trials = create_extended_dataframe(**core_data).drop(columns=['date','LDT_mode'])
        summary = summarize.session_level_summary(trials).sort_values(by='startdatetime').reset_index(drop=True)
    except Exception as err:
        log_error_data(err)
        return
        
    # add some columns to trials and summary
    trials['lims_id'] = lims_id
    summary = add_metadata_to_summary(summary, lims_id, pkl_path, behavior_session_uuid)
    trials['behavior_session_uuid'] = behavior_session_uuid

    if behavior_session_uuid in db.summary.distinct('behavior_session_uuid'):
        print('session with uuid {} already in summary'.format(behavior_session_uuid))
    else:
        # insert summary
        insert_summary_row(summary)

    if behavior_session_uuid in db.trials.distinct('behavior_session_uuid'):
        print('session with uuid {} already in trials'.format(behavior_session_uuid,'trials'))
    else:
        # insert trials
        for idx,row in trials.iterrows():
            trials_row = {k:v for k,v in zip(list(row.keys()),list(row.values))}
            trials_row['reward_times'] = trials_row['reward_times'].tolist()
            trials_row['startdatetime'] = pd.to_datetime(str(trials_row['startdatetime']))
            # cast entries in arrays and lists as lists with basic python int/float types
            for key in [key for key in trials_row.keys() if type(trials_row[key]) in [list,np.ndarray]]:
                trials_row[key] = [int(v) if is_int(v) else float(v) for v in trials_row[key]]

            db.trials.insert_one(trials_row)

    # insert data for each of the following core_data tables
    # note that running data is excluded - it exceeds the 16 Mb document maximum for Mongo
    # it's dealth with seperately below
    for key in ['licks','rewards','visual_stimuli','omitted_stimuli','metadata','log']:
        if behavior_session_uuid in db[key].distinct('behavior_session_uuid'):
            print('session with uuid {} already in {}'.format(behavior_session_uuid,key))
        else:
            if key in core_data:
                if type(core_data[key]) == pd.DataFrame:
                    data_to_insert = json.loads(core_data[key].to_json(orient='records'))
                    dtype='dataframe'
                elif type(core_data[key]) == list:
                    data_to_insert = core_data[key]
                    dtype = 'list'
                elif type(core_data[key]) == dict:
                    data_to_insert = core_data[key]
                    dtype = 'dict'
                insert(db, key, lims_id, behavior_session_uuid, dtype, data_to_insert)

    # insert running
    if behavior_session_uuid in db.running.distinct('behavior_session_uuid'):
        print('session with uuid {} already in {}'.format(behavior_session_uuid,'running'))
    else:
        # drop the time column from running and set index to frame to make it smaller
        running = core_data['running'].drop(columns=['time','dx']).set_index('frame')
        dtype = 'dataframe'
        data_to_insert = json.loads(running.to_json(orient='records'))
        insert(db, 'running', lims_id, behavior_session_uuid, dtype, data_to_insert)
    
    # insert time
    if behavior_session_uuid in db.time.distinct('behavior_session_uuid'):
        print('session with uuid {} already in time'.format(behavior_session_uuid))
    else:
        insert(db, 'time', lims_id, behavior_session_uuid, 'list', core_data['time'].tolist())
        
    if db_connection is None:
        db_conn.close()
