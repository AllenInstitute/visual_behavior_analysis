from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc
import traceback
import allensdk
import datetime
import sys
import platform
import pandas as pd
import argparse

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
    conn.close()
    return pd.DataFrame(list(res))


def get_validation_results(behavior_session_id=None):
    conn = db.Database('visual_behavior_data')
    if behavior_session_id is None:
        res = conn['sdk_validation']['validation_results'].find({})
    else:
        res = conn['sdk_validation']['validation_results'].find({'behavior_session_id': behavior_session_id})
    return pd.DataFrame(list(res)).drop(columns=['_id']).set_index('behavior_session_id')


def validate_attribute(behavior_session_id, attribute):
    session = get_sdk_session(behavior_session_id, is_ophys(behavior_session_id))
    if attribute in dir(session):
        # if the attribute exists, try to load it
        try:
            res = getattr(session, attribute)
            return True
        except:
            log_error_to_mongo(
                behavior_session_id,
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

        self.results = self.validate_all_attributes()
        log_validation_results_to_mongo(
            behavior_session_id,
            self.results
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

    def validate_all_attributes(self):
        validation_results = {}
        for attribute in self.attributes:
            validation_results[attribute] = validate_attribute(self.behavior_session_id, attribute)

        return validation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run sdk validation')
    parser.add_argument("--behavior-session-id", type=str, default=None, metavar='behavior session id')
    args = parser.parse_args()

    validation = ValidateSDK(int(args.behavior_session_id))
