import traceback
import allensdk
import datetime
import sys
import platform
import pandas as pd
import argparse
import plotly.graph_objs as go
import warnings
from visual_behavior import database as db
from visual_behavior.data_access import utilities as data_access_utilities
from visual_behavior.data_access import loading
from multiprocessing import Pool


def log_error_to_mongo(behavior_session_id, failed_attribute, error_class, traceback):
    conn = db.Database('visual_behavior_data')
    entry = {
        'timestamp': str(datetime.datetime.now()),
        'sdk_version': allensdk.__version__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'behavior_session_id': behavior_session_id,
        'failed_attribute': failed_attribute,
        'error_class': str(error_class),
        'traceback': traceback
    }
    conn['sdk_validation']['error_logs'].insert_one(entry)
    conn.close()


def log_validation_results_to_mongo(behavior_session_id, validation_results):
    validation_results.update({'behavior_session_id': behavior_session_id})
    validation_results.update({'is_ophys': data_access_utilities.is_ophys(behavior_session_id)})
    validation_results.update({'timestamp': str(datetime.datetime.now())})
    conn = db.Database('visual_behavior_data')
    collection = conn['sdk_validation']['validation_results']
    document = validation_results
    keys_to_check = ['behavior_session_id']
    db.update_or_create(collection, document, keys_to_check, force_write=False)
    conn.close()


def get_error_logs(behavior_session_id):
    conn = db.Database('visual_behavior_data')
    res = conn['sdk_validation']['error_logs'].find({'behavior_session_id': behavior_session_id})
    error_logs = pd.DataFrame(list(res))
    conn.close()
    return error_logs


def build_error_df(behavior_session_table):
    '''
    build a dataframe of all error logs for attribute failures in the behavior session table
    '''
    attributes_to_check = [
        'average_projection',
        'cell_specimen_table',
        'corrected_fluorescence_traces',
        'dff_traces',
        'eye_tracking',
        'licks',
        'max_projection',
        'metadata',
        'motion_correction',
        'ophys_timestamps',
        'rewards',
        'running_data_df',
        'running_speed',
        'segmentation_mask_image',
        'stimulus_presentations',
        'stimulus_templates',
        'stimulus_timestamps',
        'task_parameters',
        'trials'
    ]

    error_list = []
    # check ophys sessions
    sessions_with_failures = behavior_session_table[~behavior_session_table[attributes_to_check].fillna(1).apply(all, axis=1)]
    sessions_to_check = sessions_with_failures.reset_index()
    for idx, session in sessions_to_check.iterrows():
        for attribute in attributes_to_check:
            if session[attribute] == 0:
                error_list.append((session['behavior_session_id'], attribute))

    with Pool(32) as pool:
        ans = pool.starmap(error_query, error_list)

    return pd.concat(ans).reset_index()


def get_validation_results(behavior_session_id=None):
    conn = db.Database('visual_behavior_data')
    if behavior_session_id is None:
        res = conn['sdk_validation']['validation_results'].find({})
    else:
        res = conn['sdk_validation']['validation_results'].find({'behavior_session_id': behavior_session_id})

    if res.count() > 0:
        ans = pd.DataFrame(list(res)).drop(columns=['_id']).set_index('behavior_session_id')
        conn.close()
        return ans
    else:
        conn.close()
        return pd.DataFrame()


def error_query(behavior_session_id, attribute):
    '''
    query mongo for the most recent error log for a given session/attribute
    '''
    query = {
        "behavior_session_id": int(behavior_session_id),
        "failed_attribute": attribute
    }
    conn = db.Database('visual_behavior_data')
    matching_errors = pd.DataFrame(list(conn['sdk_validation']['error_logs'].find(query))).drop(columns='_id')
    conn.close()

    return matching_errors.sort_values(by='timestamp').drop_duplicates(subset=['behavior_session_id'], keep='last')


def get_donor_from_specimen_id(specimen_id):
    res = db.lims_query('select * from specimens where id = {}'.format(specimen_id))
    if len(res['donor_id']) == 1:
        return res['donor_id'].iloc[0]
    elif len(res['donor_id']) == 0:
        return None
    elif len(res['donor_id']) > 1:
        print('found more than one donor ID for specimen ID {}'.format(specimen_id))
        return res['donor_id'].iloc[0]


def get_behavior_session_table():
    '''
    returns a table of every behavior session that meets the following criteria:
        * is an ophys session and has at least one passing experiment associated with it
        * is a behavior only session performed by a mouse with at least one passing ophys experiment
    '''
    cache = loading.get_visual_behavior_cache()
    behavior_session_table = cache.get_behavior_session_table().reset_index()
    filtered_ophys_experiment_table = loading.get_filtered_ophys_experiment_table()

    # get donor id for experiment_table
    filtered_ophys_experiment_table['donor_id'] = filtered_ophys_experiment_table['specimen_id'].map(
        lambda sid: get_donor_from_specimen_id(sid)
    )

    # get behavior donor dataframe - all mice in behavior table
    behavior_donors = pd.DataFrame({'donor_id': behavior_session_table['donor_id'].unique()})
    # add a flag identifying which donors have associated ophys sessions
    behavior_donors['donor_in_ophys'] = behavior_donors['donor_id'].map(
        lambda did: did in list(filtered_ophys_experiment_table['donor_id'].unique())
    )

    # merge back in behavior donors to determine which behavior sessions have associated ophys
    behavior_session_table = behavior_session_table.merge(
        behavior_donors,
        left_on='donor_id',
        right_on='donor_id',
        how='left',
    )

    # get project table
    project_table = db.lims_query("select id,code from projects")
    query = '''SELECT behavior_sessions.id, specimens.project_id FROM specimens
    JOIN donors ON specimens.donor_id=donors.id
    JOIN behavior_sessions ON donors.id=behavior_sessions.donor_id'''
    behavior_id_project_id_map = db.lims_query(query).rename(columns={'id': 'behavior_session_id'}).merge(
        project_table,
        left_on='project_id',
        right_on='id',
        how='left',
    ).drop(columns=['id']).rename(columns={'code': 'project_code'}).drop_duplicates('behavior_session_id').set_index('behavior_session_id')

    # merge project table with behavior sessions
    behavior_session_table = behavior_session_table.merge(
        behavior_id_project_id_map.reset_index(),
        left_on='behavior_session_id',
        right_on='behavior_session_id',
        how='left'
    )

    # add a boolean for whether or not a session is in the filtered experiment table
    def osid_in_filtered_experiments(osid):
        if pd.notnull(osid):
            return osid in filtered_ophys_experiment_table['ophys_session_id'].unique()
        else:
            return True
    behavior_session_table['in_filtered_experiments'] = behavior_session_table['ophys_session_id'].apply(osid_in_filtered_experiments)

    # add missing session types (I have no idea why some are missing!)
    def get_session_type(osid):
        if osid in filtered_ophys_experiment_table['ophys_session_id'].unique().tolist():
            return filtered_ophys_experiment_table.query('ophys_session_id == {}'.format(osid)).iloc[0]['session_type']
        else:
            return None

    # drop sessions for mice with no ophys AND sessions that aren't in the filtered list
    behavior_session_table = behavior_session_table.set_index('behavior_session_id').query('donor_in_ophys and in_filtered_experiments').copy()
    for idx, row in behavior_session_table.iterrows():
        if pd.isnull(row['session_type']):
            behavior_session_table.at[idx, 'session_type'] = get_session_type(row['ophys_session_id'])

    validation_results = get_validation_results().sort_index()
    behavior_session_table = behavior_session_table.merge(
        validation_results,
        left_index=True,
        right_index=True,
        how='left'
    )
    return behavior_session_table


def validate_attribute(behavior_session_id, attribute):
    session = data_access_utilities.get_sdk_session(
        behavior_session_id,
        data_access_utilities.is_ophys(behavior_session_id)
    )
    if attribute in dir(session):
        # if the attribute exists, try to load it
        try:
            getattr(session, attribute)
            return True
        except Exception:
            log_error_to_mongo(
                behavior_session_id,
                attribute,
                sys.exc_info()[0],
                traceback.format_exc(),
            )
            return False
    else:
        # return None if attribute doesn't exist
        return None


def make_sdk_heatmap(validation_results, title_addendum=''):
    '''input is validation matrix, output is plotly figure'''

    behavior_only_cols = [
        'licks',
        'metadata',
        'rewards',
        'running_data_df',
        'running_speed',
        'stimulus_presentations',
        'stimulus_templates',
        'stimulus_timestamps',
        'task_parameters',
        'trials',
    ]
    ophys_cols = [
        'average_projection',
        'cell_specimen_table',
        'corrected_fluorescence_traces',
        'dff_traces',
        'eye_tracking',
        'max_projection',
        'motion_correction',
        'ophys_timestamps',
        'segmentation_mask_image',
    ]

    results_to_plot = validation_results[behavior_only_cols + ophys_cols]

    x = results_to_plot.columns
    y = ['behavior_session_id:\n  {}'.format(bsid) for bsid in results_to_plot.index]
    session_type = ['session_type: {}'.format(st) for st in validation_results['session_type']]
    equipment_name = ['equipment_name: {}'.format(en) for en in validation_results['equipment_name']]
    project_code = ['project_code: {}'.format(pj) for pj in validation_results['project_code']]
    z = results_to_plot.values

    hovertext = list()
    for yi, yy in enumerate(y):
        hovertext.append(list())
        for xi, xx in enumerate(x):
            hovertext[-1].append('attribute: {}<br />{}<br />{}<br />{}<br />{}<br />Successfully Loaded: {}'.format(
                xx,
                yy,
                session_type[yi],
                equipment_name[yi],
                project_code[yi],
                z[yi][xi]
            ))

    fig = go.Figure(
        data=go.Heatmap(
            x=results_to_plot.columns,
            y=results_to_plot.index,
            z=results_to_plot.values,
            hoverongaps=True,
            showscale=False,
            colorscale='inferno',
            xgap=2,
            ygap=0,
            hoverinfo='text',
            text=hovertext
        )
    )

    timestamp = datetime.datetime.now()
    timestamp_string = 'last updated on {} @ {}'.format(timestamp.strftime('%D'), timestamp.strftime('%H:%M:%S'))

    fig.update_layout(
        autosize=False,
        width=1200,
        height=900,
        margin=dict(
            l=0,  # NOQA E741
            r=0,
            b=0,
            t=50,
            pad=0
        ),
        xaxis_title='SDK attribute',
        yaxis_title='Behavior Session ID',
        title='SDK Attribute Validation {} (black = failed) {}'.format(title_addendum, timestamp_string)
    )
    fig.update_yaxes(autorange="reversed", type='category', showticklabels=False, showgrid=False)
    fig.update_xaxes(dtick=1, showgrid=False)

    return fig


class ValidateSDK(object):
    '''
    SDK validation class
    inputs:
        behavior_session_id (int): behavior session to validate
        validate_only_failed (boolean, default = False):
            if True, check for previous validation and check only attributes that previously failed
            if False, check all attributes
    '''

    def __init__(self, behavior_session_id, validate_only_failed=False):
        self.behavior_session_id = behavior_session_id
        self.is_ophys = data_access_utilities.is_ophys(behavior_session_id)
        self.session = data_access_utilities.get_sdk_session(
            behavior_session_id,
            self.is_ophys
        )

        if validate_only_failed == True:
            previous_results_df = get_validation_results(self.behavior_session_id)
            if len(previous_results_df) == 0:
                warnings.warn('no existing attributes for session {}, validating all'.format(self.behavior_session_id))
                attributes_to_validate = self.get_attributes()
            else:
                previous_results_dict = dict(previous_results_df.iloc[0])
                attributes_to_validate = [key for key, value in previous_results_dict.items() if (key != 'is_ophys' and value == 0)]
        else:
            attributes_to_validate = self.get_attributes()
        self.validate_attributes(attributes_to_validate)
        log_validation_results_to_mongo(
            behavior_session_id,
            self.validation_results
        )

    def get_attributes(self):
        expected_attributes = [
            'average_projection',
            'cell_specimen_table',
            'corrected_fluorescence_traces',
            'dff_traces',
            'eye_tracking',
            'licks',
            'max_projection',
            'metadata',
            'motion_correction',
            'ophys_timestamps',
            'rewards',
            'running_data_df',
            'running_speed',
            'segmentation_mask_image',
            'stimulus_presentations',
            'stimulus_templates',
            'stimulus_timestamps',
            'task_parameters',
            'trials'
        ]
        return expected_attributes

    def validate_attributes(self, attributes_to_validate):
        self.validation_results = {}
        for attribute in attributes_to_validate:
            print('checking {}'.format(attribute))
            self.validation_results[attribute] = validate_attribute(self.behavior_session_id, attribute)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run sdk validation')
    parser.add_argument("--behavior-session-id", type=str, default=None, metavar='behavior session id')
    parser.add_argument("--validate-only-failed", type=str2bool, default=False, metavar='validate only previously failed attributes')
    args = parser.parse_args()
    print('args.validate_only_failed: {}'.format(args.validate_only_failed))
    validation = ValidateSDK(int(args.behavior_session_id), args.validate_only_failed)
