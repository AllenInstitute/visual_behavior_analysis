from pymongo import MongoClient
import yaml
import pandas as pd
import numpy as np
import json
import traceback
import datetime
import uuid
import warnings
from psycopg2 import extras

from allensdk.internal.api import PostgresQueryMixin
from allensdk.core.authentication import credential_injector
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP


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


def get_psql_dict_cursor():
    """Set up a connection to a psql db server with a dict cursor"""
    api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)(PostgresQueryMixin)())
    con = api.get_connection()
    con.set_session(readonly=True, autocommit=True)
    return con.cursor(cursor_factory=extras.RealDictCursor)


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


def get_pkl_path(session_id=None, id_type='behavior_session_id'):
    '''
    get the path to a pkl file for a given session
    '''
    if id_type == 'behavior_session_id':
        rec = get_well_known_files(session_id, 'BehaviorSession').loc['StimulusPickle']
    elif id_type == 'ophys_session_id':
        rec = get_well_known_files(session_id, 'OphysSession').loc['StimulusPickle']
    pkl_path = ''.join([rec['storage_directory'], rec['filename']])
    return pkl_path


def get_value_from_table(search_key, search_value, target_table, target_key):
    '''
    a general function for getting a value from a LIMS table
    '''
    api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)
           (PostgresQueryMixin)())
    query = '''
        select {}
        from {}
        where {} = '{}'
    '''
    result = pd.read_sql(query.format(target_key, target_table, search_key, search_value), api.get_connection())
    if len(result) == 1:
        return result[target_key].iloc[0]
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

    assert (len(input_id_dict) == 1), "use only one ID type to identify others"
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


def is_uuid(n):
    return isinstance(n, uuid.UUID)


def is_bool(n):
    return isinstance(n, (bool, np.bool_))


def is_array(n):
    return isinstance(n, np.ndarray)


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
        # db[table].insert_one(simplify_entry(entry))
        keys_to_check = 'behavior_session_uuid'
        update_or_create(table, entry, keys_to_check, force_write=False)

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


def get_well_known_files(session_id, attachable_id_type='OphysSession'):
    '''
    return well_known_files table with names as index
    inputs:
        session_id (int): session id from LIMS
        attachable_id_type (str): session id type. Choose from 'OphysSession' (default) or 'EcephysSession'
    returns:
        pandas dataframe with all LIMS well known files for the given session
    '''

    query = '''
    select * from well_known_files wkf
    join well_known_file_types wkft
    on wkft.id = wkf.well_known_file_type_id
    where wkf.attachable_type = '{}'
    and wkf.attachable_id in ({});
    '''.format(attachable_id_type, session_id)

    return lims_query(query).set_index('name')


def simplify_type(x):
    if is_int(x):
        return int(x)
    elif is_bool(x):
        return int(x)
    elif is_float(x):
        return float(x)
    elif is_uuid(x):
        return str(x)
    elif is_array(x):
        return [simplify_type(e) for e in x]
    else:
        return x


def simplify_entry(entry):
    '''
    entry is one document
    '''
    entry = {k: simplify_type(v) for k, v in entry.items()}
    return entry


def clean_and_timestamp(entry):
    '''make sure float and int types are basic python types (e.g., not np.float)'''
    entry = simplify_entry(entry)
    entry.update({'entry_time_utc': str(datetime.datetime.utcnow())})
    return entry


def update_or_create(collection, document, keys_to_check, force_write=False):
    '''
    updates a collection of the document exists
    inserts if it does not exist
    uses keys in `keys_to_check` to determine if document exists. Other keys will be written, but not used for checking uniqueness
    '''
    if force_write:
        collection.insert_one(simplify_entry(document))
    else:
        query = {key: simplify_type(document[key]) for key in keys_to_check}
        if collection.find_one(query) is None:
            # insert a document if this experiment/cell doesn't already exist
            collection.insert_one(simplify_entry(document))
        else:
            # update a document if it does exist
            collection.update_one(query, {"$set": simplify_entry(document)})


def get_labtracks_id_from_specimen_id(specimen_id, show_warnings=True):
    '''
    for a given mouse:
        convert
            9 or 10 digit specimen_id (from LIMS)
        to
            6 digit labtracks ID
    '''
    res = lims_query("select external_specimen_name from specimens where specimens.id = {}".format(specimen_id))

    if isinstance(res, (str, int, np.int64)):
        return int(res)
    elif isinstance(res, pd.Series):
        if res.empty:
            # an empty query returns an empty Series. Just return None in this case
            return None
        elif len(res) > 1:
            # if we get multiple results, return only the first, show a warning
            if show_warnings:
                warnings.warn('specimen_id {} is associated with {} specimen_ids:\n{}\nreturning only the first'.format(
                    specimen_id,
                    len(res),
                    res,
                ))
            return int(res.iloc[0])


def get_specimen_id_from_labtracks_id(labtracks_id, show_warnings=True):
    '''
    for a given mouse:
        convert
            6 digit labtracks ID
        to
            9 or 10 digit specimen_id (from LIMS)
    '''
    res = lims_query("select id from specimens where specimens.external_specimen_name = '{}'".format(labtracks_id))

    if isinstance(res, (str, int, np.int64)):
        return int(res)
    elif isinstance(res, pd.Series):
        if res.empty:
            # an empty query returns an empty Series. Just return None in this case
            return None
        elif len(res) > 1:
            # if we get multiple results, return only the first, show a warning
            if show_warnings:
                warnings.warn('labtracks_id {} is associated with {} specimen_ids:\n{}\nreturning only the first'.format(
                    labtracks_id,
                    len(res),
                    res.values,
                ))
            return int(res.iloc[0])


def get_mouse_ids(id_type, id_number):
    '''
    returns a dataframe of all variations of mouse ID for a given input ID

    inputs:
        id_type: (string) the type of ID to search on
        id_number: (int,string, list of ints or list of strings) the associated ID number(s)

    allowable id_types:
        donor_id: LIMS donor_id
        specimen_id: LIMS specimen ID
        labtracks_id: Labtracks ID (6 digit ID on mouse cage)
        external_specimen_name: alternate name for labtracks_id (used in specimens table)
        external_donor_name: alternate name for labtracks_id (used in donors table)

    returns:
        a dataframe with columns for `donor_id`, `labtracks_id`, `specimen_id`

    Note: in rare cases, a single donor_id/labtracks_id was associated with multiple specimen_ids
          this occured for IDs used as test_mice (e.g. labtracks_id 900002)
          and should not have occured for real experimental mice
    '''

    if id_type.lower() == 'donor_id':
        id_type_string = 'donors.id'
    elif id_type.lower() == 'specimen_id':
        id_type_string = 'specimens.id'
    elif id_type.lower() in ['labtracks_id', 'external_specimen_name', 'external_donor_name']:
        id_type_string = 'donors.external_donor_name'
    else:
        raise TypeError('invalid `id_type` {}'.format(id_type))

    if isinstance(id_number, (str, int, np.int64)):
        id_number = [id_number]
    id_number = [str(i) for i in id_number]

    query = """
    select donors.id donor_id, donors.external_donor_name as labtracks_id, specimens.id as specimen_id
    from donors
    join specimens on donors.external_donor_name = specimens.external_specimen_name
    where {} in {}
    """.format(id_type_string, tuple(id_number)).replace(',)', ')')

    return lims_query(query)


def lims_query(query):
    '''
    execute a SQL query in LIMS
    returns:
        * the result if the result is a single element
        * results in a pandas dataframe otherwise

    Examples:

        >> lims_query('select ophys_session_id from ophys_experiments where id = 878358326')

        returns 877907546

        >> lims_query('select * from ophys_experiments where id = 878358326')

        returns a single line dataframe with all columns from the ophys_experiments table for ophys_experiment_id =  878358326

        >> lims_query('select * from ophys_sessions where id in (877907546, 876522267, 869118259)')

        returns a three line dataframe with all columns from the ophys_sessions table for ophys_session_id in the list [877907546, 876522267, 869118259]

        >> lims_query('select * from ophys_sessions where specimen_id = 830901424')

        returns all rows and columns in the ophys_sessions table for specimen_id = 830901424
    '''
    api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)(PostgresQueryMixin)())
    conn = api.get_connection()

    df = pd.read_sql(query, conn)

    conn.close()

    if df.shape == (1, 1):
        # if the result is a single element, return only that element
        return df.iloc[0][0]
    else:
        # otherwise return in dataframe format
        return df


def log_cell_dff_data(record):
    '''
    writes a cell record to 'dff_summary' collection in 'ophys_data' mongo database
    record should contain stats about the cell's deltaF/F trace
    references cell by cell_roi_id
    if record exists for roi_id and cell_specimen_id has changed, will add old cell_specimen_id to list 'previous_cell_specimen_ids'
    returns None
    '''
    db_conn = Database('visual_behavior_data')
    collection = db_conn['ophys_data']['dff_summary']
    existing_record = collection.find_one({'cell_roi_id': record['cell_roi_id']})

    # if the cell specimen_id doesn't match what was in the record, log old ID to a list called 'previous_cell_specimen_ids'
    if existing_record and record['cell_specimen_id'] != existing_record['cell_specimen_id']:
        if 'previous_cell_specimen_ids' in existing_record.keys():
            previous_cell_specimen_ids = existing_record['previous_cell_specimen_ids']
            previous_cell_specimen_ids.append(existing_record['cell_specimen_id'])
        else:
            previous_cell_specimen_ids = [existing_record['cell_specimen_id']]
    else:
        previous_cell_specimen_ids = []

    # write record to database
    record['previous_cell_specimen_ids'] = previous_cell_specimen_ids
    update_or_create(collection, record, keys_to_check=['cell_roi_id'])
    db_conn.close()


def get_cell_dff_data(search_dict={}, return_id=False):
    '''
    retrieve information from the 'dff_summary' collection in 'ophys_data' mongo database
    pass in a `search_dict` to constrain the search
    passing an empty dict (default) returns the full collection
    Note that the query itself is fast (tens of ms), but returning a large table can be slow (~30 seconds for the full collection)
    inputs:
        search_dict: a dictionary of key/value pairs to constrain the search (columns must be one of those available as an output - see below)
        return_id: If True, returns a column containing the internal mongo index has of each entry (default = False)
    returns: pandas dataframe
        columns:
            The following come directly from the `cell_specimen_table` in the AllenSDK
                cell_specimen_id: ID of each cell to facilitate cell matching. ID will be shared across ROIs from unique sessions that are identified as matches
                cell_roi_id: Unique ID for each cell
                height: height, in pixels, of ROI mask
                width: width, in pixels, of ROI mask
                x: x-position of mask, in pixels
                y: y-position of mask, in pixels
                mask_image_plane: image plane of corresponding mask. Integer ranging from 0 to 7
                max_correction_{up, down, left, right}: the maximum translation, in pixels, of this ROI during motion correction
                valid_roi: boolean denoting whether the ROI was deemed valid after ROI filtering
            The following are appended and come from the pandas.describe() method on the deltaF/F (`dff`) trace for the cell
                ophys_experiment_id: the associated ophys_experiment_id
                previous_cell_specimen_ids: a list of cell_specimen_ids associated with this ROI in previous cell_matchting runs (not exhaustive)
                count: length of dff vector
                mean: mean of dff vector
                std: standard devitation of dff vector
                min: min of dff vector
                max: max of dff vector
                25%, 50%, 75%: lower, middle (median) and upper quartile values
    '''
    db_conn = Database('visual_behavior_data')
    collection = db_conn['ophys_data']['dff_summary']
    res = pd.DataFrame(list(collection.find(search_dict)))
    db_conn.close()

    if not return_id:
        res = res.drop(columns=['_id'])

    # drop `index` column if it exists
    if 'index' in res.columns:
        res = res.drop(columns=['index'])

    return res
