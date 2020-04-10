from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc
import traceback
import allensdk
import datetime
import sys
import platform
import pandas as pd

from visual_behavior import database as db


def get_cache():
    MANIFEST_PATH = "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/2020_cache/production_cache/manifest.json"
    cache = bpc.from_lims(manifest=MANIFEST_PATH)
    return cache


def is_ophys(behavior_session_id):
    cache = get_cache()

    behavior_session_table = cache.get_behavior_session_table()

    return pd.notnull(behavior_session_table.loc[behavior_session_id]['ophys_session_id'])


def get_sdk_session(behavior_session_id, is_ophys):
    cache = get_cache()

    if is_ophys:
        behavior_session = cache.get_behavior_session_data(behavior_session_id)
        ophys_experiment_id = behavior_session.ophys_experiment_ids[0]
        return cache.get_session_data(ophys_experiment_id)
    else:
        return cache.get_behavior_session_data(behavior_session_id)


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


def get_error_logs(behavior_session_id):
    conn = db.Database('visual_behavior_data')
    res = conn['sdk_validation']['error_logs'].find({'behavior_session_id': behavior_session_id})
    conn.close()
    return pd.DataFrame(list(res))


def validate_attribute(session, attribute):
    if attribute in dir(session):
        # if the attribute exists, try to load it
        try:
            res = getattr(session, attribute)
            return True
        except:
            log_error_to_mongo(
                session.behavior_session_id,
                attribute,
                sys.exc_info()[0],
                traceback.format_exc(),
            )
            return False
    else:
        return None


class ValidateSDK(object):
    def __init__(self, behavior_session_id):
        self.behavior_session_id = behavior_session_id
        self.attributes = self.get_attributes()
        self.is_ophys = is_ophys(behavior_session_id)
        self.session = get_sdk_session(behavior_session_id, self.is_ophys)
        self.session.behavior_session_id = behavior_session_id

        self.results = self.validate_all_attributes()

    def get_attributes(self):
        expected_attributes = [
            'running_data_df',
            'licks',
            'trials',
            'average_projection',
            'corrected_fluorescence_traces',
        ]
        return expected_attributes

    def validate_all_attributes(self):
        validation_results = {}
        for attribute in self.attributes:
            validation_results[attribute] = validate_attribute(self.session, attribute)

        return validation_results
