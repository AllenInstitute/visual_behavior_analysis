from pymongo import MongoClient
import yaml
import pandas as pd
import numpy as np
import json
import os
import glob
import traceback
import datetime

from allensdk.internal.api import PostgresQueryMixin


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
        self.client = MongoClient('mongodb://{}:{}'.format(ip, port))

        # set each table as an attribute of the class (but not admin) and as an entry in a dictionary
        # this will provide flexibility in how the tables are called
        self.database = {}
        self.database_names = []
        databases = [db for db in self.client.database_names() if db != 'admin']
        for db in databases:
            self.database_names.append(db)
            self.database[db] = self.client[db]
            setattr(self, db, self.client[db])
        # make subscriptable
        self._db_names = {db: self.client[db] for db in databases}

    def __getitem__(self, item):
        # this allows databases to be accessed by name
        return self._db_names[item]

    def query(self, database, collection, query={}, return_as='dataframe'):
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
    look up by behavior_session_uuid or behavior_session_id (the latter is the LIMS ID)
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
    nested_tables = ['time', 'running', 'licks', 'rewards', 'visual_stimuli', 'omitted_stimuli']
    nested_dicts = ['metadata', 'log']

    if table_name.lower() in nested_tables:
        # these collections have the data of interest in a document called 'data'
        return _get_table(db, table_name, session_id, id_type, return_as='dataframe')
    elif table_name.lower() in nested_dicts:
        return _get_table(db, table_name, session_id, id_type, return_as='dict')
    elif table_name.lower() == 'trials':
        # trials is a flat collection: each entry is a row in the desired trials table
        return _get_trials(db, table_name, session_id, id_type)
    elif table_name.lower() == 'all':
        res = {'trials': _get_trials(db, table_name, session_id, id_type)}
        for table_name in nested_tables:
            res[table_name] = _get_table(db, table_name, session_id, id_type)
        return res

    db.close()


def _check_name_schema(database, session_id, id_type):
    '''
    lims_id (aka behavior_session_id) should be int
    behavior_session_uuid should be string

    in mouseseeks, behavior_session_uuid is foraging_id
    '''
    if id_type == 'behavior_session_id':
        session_id = int(session_id)
        if database == 'mouseseeks':
            id_type = 'lims_id'  # this is what it is called in mouseseeks
    elif id_type == 'behavior_session_uuid' and database == 'mouseseeks':
        id_type = 'foraging_id'  # the name is different in mouseseeks

    return session_id, id_type


def _get_table(db, table_name, session_id=None, id_type='behavior_session_uuid', db_name='behavior_data', return_as='dataframe'):
    '''
    a general function for getting behavior data tables
    special cases:
        `time` has an unlabeled column - label it
        `running` is missing time data, which was done to reduce storage space. Merge time back in
    '''
    session_id, id_type = _check_name_schema('visual_behavior_data', session_id, id_type)
    if return_as == 'dataframe':
        res = pd.DataFrame(list(db[db_name][table_name].find({id_type: session_id}))[0]['data'])
    else:
        res = list(db[db_name][table_name].find({id_type: session_id}))[0]['data']
    if table_name == 'time':
        res = res.rename(columns={0: 'time'})
    if table_name == 'running':
        # time was stripped from running to save space. add it back in:
        time_df = pd.DataFrame(list(db[db_name]['time'].find({id_type: session_id}))[0]['data']).rename(columns={0: 'time'})
        res = res.merge(time_df, left_index=True, right_index=True)
    return res


def _get_trials(db, table_name, session_id=None, id_type='behavior_session_uuid', db_name='behavior_data'):
    '''
    get trials table for a given session
    '''
    session_id, id_type = _check_name_schema('visual_behavior_data', session_id, id_type)
    return pd.DataFrame(list(db[db_name][table_name].find({id_type: session_id})))


def get_behavior_session_summary(exclude_error_sessions=True):
    '''
    a convenience function to get the summary dataframe from the visual behavior database
    by default: sessions that VBA could not load are excluded
    '''
    vb = Database('visual_behavior_data')
    summary = vb.query('behavior_data', 'summary')
    if exclude_error_sessions:
        # missing values imply false, but are returned as NaN. Cast to float, then filter out ones:
        summary = summary[summary['error_on_load'].astype(float) != 1].reset_index(drop=True)
    vb.close()
    return summary


def get_pkl_path(session_id=None, id_type='behavior_session_uuid'):
    '''
    get the path to a pkl file for a given session
    '''
    session_id, id_type = _check_name_schema('mouseseeks', session_id, id_type)

    db = Database('mouseseeks')
    res = db.query('db', 'behavior_session_log', query={id_type: session_id})
    db.close()

    if res is not None and len(res) > 0:
        if len(res) == 1:
            storage_directory = res['storage_directory'].item()
        elif len(res) > 1:
            # there are occassionally duplicate entries in mouseseeks. return only the first:
            storage_directory = res['storage_directory'].iloc[0]
        pkls = glob.glob(os.path.join(storage_directory, '*.pkl'))
        if len(pkls) == 1:
            return pkls[0]


def get_value_from_table(search_key, search_value, target_table, target_key):
    '''
    a general function for getting a value from a LIMS table
    '''
    api = PostgresQueryMixin()
    query = f'''
    select {target_key}
    from {target_table}
    where {search_key} = '{search_value}'
    '''
    result = pd.read_sql(query, api.get_connection())
    if len(result) == 1:
        return result[target_key].item()
    else:
        return None


def populate_id_dict(input_id_dict):
    '''
    an ophys session has 5 different keys by which it can be identified
    this function will take an input key/value pair that represents one possible ID and returns a dictionary containing all IDs

    * `behavior_session_uuid` (aka `foraging_id` in LIMS): a 36 digit string generated with the UUID python module by camstim at runtime
      * `foraging_id` in the LIMS `ophys_sessions` table
      * `foraging_id` in the LIMS `behavior_sessions` table
    * `behavior_session_id`: a 9 digit integer identifying the behavior session
      * `id` in the LIMS `behavior_sessions` table
    * `ophys_session_id`: a 9 digit integer identifying the ophys session. Every ophys session has an associated behavior session.
      * `id` in the LIMS `ophys_sessions` table
      * `ophys_session_id` in LIMS `behavior_sessions` table
      * `ophys_session_id` in LIMS `ophys_experiments` table
    * `ophys_experiment_id`: a 9 digit integer. Each ophys session has one experiment.
      * `id` in the LIMS `ophys_experiments` table

    For example, for the session run by mouse 450471 on 2P3 on June 4, 2019:
    * behavior_session_uuid = 4d4dfd3e-e1bf-4ad8-9775-1273ce7e5189
    * LIMS behavior session ID = 880784794
    * LIMS ophys session ID = 880753403
    * LIMS ophys_experiment ID = 880961028

    example:
        >> populate_id_dict({'ophys_experiment_id': 880961028,})

        returns:
        {'behavior_session_id': 880784794,
         'behavior_session_uuid': '4d4dfd3e-e1bf-4ad8-9775-1273ce7e5189',
         'foraging_id': '4d4dfd3e-e1bf-4ad8-9775-1273ce7e5189',
         'ophys_experiment_id': 880961028,
         'ophys_session_id': 880753403}

    '''
    ids = {
        'behavior_session_uuid': None,
        'foraging_id': None,
        'behavior_session_id': None,
        'ophys_session_id': None,
        'ophys_experiment_id': None,
    }

    assert(len(input_id_dict) == 1), "use only one ID type to identify others"
    for key in input_id_dict:
        assert key in ids.keys(), "input key must be one of {}".format(list(ids.keys()))
        ids[key] = input_id_dict[key]

    for i in range(3):  # maximum of three passes are required to fill all key/value pairs
        # foraging_id and behavior_session_id are equivalent:
        if ids['behavior_session_uuid']:
            ids['foraging_id'] = ids['behavior_session_uuid']

        # if we have foraging_id, get behavior_session_uuid, behavior_session_id, ophys_session_id
        if ids['foraging_id']:
            ids['behavior_session_uuid'] = ids['foraging_id']
            ids['behavior_session_id'] = get_value_from_table('foraging_id', ids['foraging_id'], 'behavior_sessions', 'id')
            ids['ophys_session_id'] = get_value_from_table('foraging_id', ids['foraging_id'], 'ophys_sessions', 'id')

        # if we have behavior_session_id, get foraging_id and ophys_session_id:
        if ids['behavior_session_id']:
            ids['foraging_id'] = get_value_from_table('id', ids['behavior_session_id'], 'behavior_sessions', 'foraging_id')

        # if we have ophys_session_id, get ophys_experiment_id:
        if ids['ophys_session_id']:
            ids['ophys_experiment_id'] = get_value_from_table('ophys_session_id', ids['ophys_session_id'], 'ophys_experiments', 'id')
            ids['foraging_id'] = get_value_from_table('id', ids['ophys_session_id'], 'ophys_sessions', 'foraging_id')

        # if we have ophys_experiment_id, get ophys_session_id:
        if ids['ophys_experiment_id']:
            ids['ophys_session_id'] = get_value_from_table('id', ids['ophys_experiment_id'], 'ophys_experiments', 'ophys_session_id')

        existing_keys = [(k, v) for k, v in ids.items() if v]
        if len(existing_keys) == 5:
            break

    return ids


def get_alternate_ids(uuid):
    '''
    get all LIMS ids for a given session UUID
    '''
    return populate_id_dict({'behavior_session_uuid': uuid})


def convert_id(input_dict, desired_key):
    '''
    an ophys session has 5 different keys by which it can be identified
    this function will convert between keys.

    It takes the following inputs:
        a key/value pair that represents one possible ID
        the desired key

    It returns
        the value matching the desired key

    The following keys may exist for a given Visual Behavior session:

    * `behavior_session_uuid`: a 36 digit string generated with the UUID python module by camstim at runtime
    * `foraging_id`: Alternate nomenclature for `behavior_session_uuid` used by LIMS
    * `behavior_session_id`: a 9 digit integer identifying the behavior session
    * `ophys_session_id`: a 9 digit integer identifying the ophys session. Every ophys session has an associated behavior session.
    * `ophys_experiment_id`: a 9 digit integer. Each ophys session has one experiment.

    For example, for the session run by mouse 450471 on 2P3 on June 4, 2019:
    * behavior_session_uuid = 4d4dfd3e-e1bf-4ad8-9775-1273ce7e5189
    * LIMS behavior session ID = 880784794
    * LIMS ophys session ID = 880753403
    * LIMS ophys_experiment ID = 880961028

    example:
        >> convert_id({'ophys_session_id': 880753403}, 'behavior_session_uuid')

            '4d4dfd3e-e1bf-4ad8-9775-1273ce7e5189'

    '''
    all_ids = populate_id_dict(input_dict)
    return all_ids[desired_key]


def get_mouseseeks_qc_results(session_id=None, id_type='behavior_session_uuid'):
    '''get qc results from mouseseeks'''
    session_id, id_type = _check_name_schema('mouseseeks', session_id, id_type)
    if id_type == 'foraging_id':
        session_id = convert_id({'foraging_id': session_id}, 'behavior_session_id')

    mouseseeks = Database('mouseseeks')
    res = list(mouseseeks.qc.metrics.find({'behavior_session_id': session_id}))
    mouseseeks.close()
    if len(res) > 0:
        return res[0]


def is_int(n):
    return isinstance(n, (int, np.integer))


def is_float(n):
    return isinstance(n, (float, np.float))


def add_behavior_record(behavior_session_uuid=None, pkl_path=None, overwrite=False, db_connection=None, db_name='behavior_data', data_type='foraging2'):
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
    if the session fails to open with VBA:
        - adds a row to summary with 'error_on_load' = True
        - saves traceback and time to 'error_log' table
    '''

    from .translator.foraging2 import data_to_change_detection_core as foraging2_translator
    from .translator.foraging import data_to_change_detection_core as foraging1_translator
    from .translator.core import create_extended_dataframe
    from .change_detection.trials import summarize

    if data_type.lower() == 'foraging2':
        data_to_change_detection_core = foraging2_translator
    elif data_type.lower() == 'foraging1':
        data_to_change_detection_core = foraging1_translator
    else:
        raise NameError('data_type must be either `foraging1` or `foraging2`')

    def insert(db, table, behavior_session_uuid, dtype, data_to_insert):
        entry = {
            'behavior_session_uuid': behavior_session_uuid,
            'dtype': dtype,
            'data': data_to_insert
        }
        entry.update(get_alternate_ids(behavior_session_uuid))
        db[table].insert_one(entry)

    def insert_summary_row(summary_row):
        if type(summary_row) == pd.DataFrame:
            summary_row = summary_row.iloc[0].to_dict()
        # cast to simple int or float
        summary_row = {k: (int(v) if is_int(v) else v) for k, v in summary_row.items()}
        summary_row = {k: (float(v) if is_float(v) else v) for k, v in summary_row.items()}
        db.summary.insert_one(summary_row)

    def add_metadata_to_summary(summary, pkl_path, behavior_session_uuid):
        summary['pkl_path'] = pkl_path
        summary['behavior_session_uuid'] = behavior_session_uuid
        summary.update(get_alternate_ids(behavior_session_uuid))
        return summary

    def log_error_data(err):
        tb = traceback.format_tb(err.__traceback__)
        entry = {
            'behavior_session_uuid': behavior_session_uuid,
            'error_log': tb,
            'local_time': str(datetime.datetime.now()),
            'utc_time': str(datetime.datetime.utcnow())
        }
        entry.update(get_alternate_ids(behavior_session_uuid))
        db['error_log'].insert_one(entry)
        print('error on session {}, writing to error table'.format(behavior_session_uuid))
        summary = {}
        summary = add_metadata_to_summary(summary, pkl_path, behavior_session_uuid)
        summary['error_on_load'] = True
        insert_summary_row(summary)

    if db_connection is None:
        db_conn = Database('visual_behavior_data')
        db = db_conn[db_name]
    else:
        db = db_connection

    assert not all((behavior_session_uuid is None, pkl_path is None)), "either a behavior_session_uuid or a pkl_path must be specified"
    assert behavior_session_uuid is None or pkl_path is None, "both a behavior_session_uuid and a pkl_path cannot be specified, choose one"

    # load behavior data
    try:
        if pkl_path is None:
            pkl_path = get_pkl_path(behavior_session_uuid)
        data = pd.read_pickle(pkl_path)
        core_data = data_to_change_detection_core(data)
        if behavior_session_uuid is None:
            behavior_session_uuid = str(core_data['metadata']['behavior_session_uuid'])
        trials = create_extended_dataframe(**core_data).drop(columns=['date', 'LDT_mode'])
        summary = summarize.session_level_summary(trials).iloc[0].to_dict()
    except Exception as err:
        log_error_data(err)
        return

    # add some columns to trials and summary
    trials['behavior_session_id'] = convert_id(
        {'behavior_session_uuid': behavior_session_uuid},
        'behavior_session_id'
    )
    summary = add_metadata_to_summary(summary, pkl_path, behavior_session_uuid)
    trials['behavior_session_uuid'] = behavior_session_uuid

    # insert summary if not already in table
    if behavior_session_uuid in db.summary.distinct('behavior_session_uuid'):
        print('session with uuid {} already in summary'.format(behavior_session_uuid))
    else:
        insert_summary_row(summary)

    # insert trials if not already in table
    if behavior_session_uuid in db.trials.distinct('behavior_session_uuid'):
        print('session with uuid {} already in trials'.format(behavior_session_uuid))
    else:
        for _, row in trials.iterrows():
            trials_row = {k: v for k, v in zip(list(row.keys()), list(row.values))}
            trials_row['reward_times'] = trials_row['reward_times'].tolist()
            trials_row['startdatetime'] = pd.to_datetime(str(trials_row['startdatetime']))
            # cast entries in arrays and lists as lists with basic python int/float types
            for key in [key for key in trials_row.keys() if type(trials_row[key]) in [list, np.ndarray]]:
                trials_row[key] = [int(v) if is_int(v) else float(v) for v in trials_row[key]]

            db.trials.insert_one(trials_row)

    # insert data for each of the following core_data tables
    # note that running data is excluded - it exceeds the 16 Mb document maximum for Mongo
    # it's dealth with seperately below
    for key in ['licks', 'rewards', 'visual_stimuli', 'omitted_stimuli', 'metadata', 'log']:
        if behavior_session_uuid in db[key].distinct('behavior_session_uuid'):
            print('session with uuid {} already in {}'.format(behavior_session_uuid, key))
        else:
            if key in core_data:
                if type(core_data[key]) == pd.DataFrame:
                    data_to_insert = json.loads(core_data[key].to_json(orient='records'))
                    dtype = 'dataframe'
                elif type(core_data[key]) == list:
                    data_to_insert = core_data[key]
                    dtype = 'list'
                elif type(core_data[key]) == dict:
                    data_to_insert = core_data[key]
                    dtype = 'dict'
                insert(db, key, behavior_session_uuid, dtype, data_to_insert)

    # insert running if not already in table
    if behavior_session_uuid in db.running.distinct('behavior_session_uuid'):
        print('session with uuid {} already in {}'.format(behavior_session_uuid, 'running'))
    else:
        # drop the time column from running and set index to frame to make it smaller
        running = core_data['running'].drop(columns=['time', 'dx']).set_index('frame')
        dtype = 'dataframe'
        data_to_insert = json.loads(running.to_json(orient='records'))
        insert(db, 'running', behavior_session_uuid, dtype, data_to_insert)

    # insert time if not already in table
    if behavior_session_uuid in db.time.distinct('behavior_session_uuid'):
        print('session with uuid {} already in time'.format(behavior_session_uuid))
    else:
        insert(db, 'time', behavior_session_uuid, 'list', core_data['time'].tolist())

    if db_connection is None:
        db_conn.close()


def get_manifest(server='visual_behavior_data'):
    '''
    convenience function to get full manifest
    '''
    vb = Database(server)
    man = vb['ophys_data']['manifest'].find({})
    vb.close()
    return pd.DataFrame(list(man))


def get_well_known_files(ophys_session_id):
    lims_api = PostgresQueryMixin()
    query = '''
    select * from well_known_files wkf
    join well_known_file_types wkft
    on wkft.id = wkf.well_known_file_type_id
    where wkf.attachable_type = 'OphysSession'
    and wkf.attachable_id in ({});
    '''.format(ophys_session_id)

    result = pd.read_sql(query, lims_api.get_connection())
    return result
