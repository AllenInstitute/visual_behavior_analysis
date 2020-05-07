from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc
from allensdk.brain_observatory.behavior.behavior_data_session import BehaviorDataSession
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
import traceback
import allensdk
import datetime
import sys
import platform
import pandas as pd
import argparse
import plotly.graph_objs as go

from visual_behavior import database as db


def get_cache():
    MANIFEST_PATH = "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/2020_cache/production_cache/manifest.json"
    cache = bpc.from_lims(manifest=MANIFEST_PATH)
    return cache


def is_ophys(behavior_session_id):
    cache = get_cache()

    behavior_session_table = cache.get_behavior_session_table()

    return pd.notnull(behavior_session_table.loc[behavior_session_id]['ophys_session_id'])


def bsid_to_oeid(behavior_session_id):
    oeid = db.lims_query(
        '''
        select oe.id 
        from behavior_sessions
        join ophys_experiments oe on oe.ophys_session_id = behavior_sessions.ophys_session_id
        where behavior_sessions.id = {}
        '''.format(behavior_session_id)
    )
    if isinstance(oeid, pd.DataFrame):
        return oeid.iloc[0][0]
    else:
        return oeid


def get_sdk_session(behavior_session_id, is_ophys):
    cache = get_cache()

    if is_ophys:
        ophys_experiment_id = bsid_to_oeid(behavior_session_id)
        return BehaviorOphysSession.from_lims(ophys_experiment_id)
    else:
        return BehaviorDataSession.from_lims(behavior_session_id)


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
    validation_results.update({'is_ophys': is_ophys(behavior_session_id)})
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


def validate_attribute(behavior_session_id, attribute):
    session = get_sdk_session(behavior_session_id, is_ophys(behavior_session_id))
    if attribute in dir(session):
        # if the attribute exists, try to load it
        try:
            res = getattr(session, attribute)
            return True
        except Exception as e:
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
        self.is_ophys = is_ophys(behavior_session_id)
        self.session = get_sdk_session(behavior_session_id, self.is_ophys)

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
