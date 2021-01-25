import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np
import visual_behavior.validation.sdk as sdk_validation
from visual_behavior.data_access import loading
import visual_behavior.database as db
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='run sdk validation')
parser.add_argument('--env', type=str, default='visual_behavior', metavar='name of conda environment to use')
parser.add_argument("--validate-only-failed", type=str2bool, default=False, metavar='validate only previously failed attributes')

job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/sdk_validation"

job_settings = {'queue': 'braintv',
                'mem': '32g',
                'walltime': '0:30:00',
                'ppn': 1,
                }


if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/code/visual_behavior_analysis/visual_behavior/validation/sdk.py".format(os.path.expanduser('~'))

    def get_donor_from_specimen_id(specimen_id):
        res = db.lims_query('select * from specimens where id = {}'.format(specimen_id))
        if len(res['donor_id']) == 1:
            return res['donor_id'].iloc[0]
        elif len(res['donor_id']) == 0:
            return None
        elif len(res['donor_id']) > 1:
            print('found more than one donor ID for specimen ID {}'.format(specimen_id))
            return res['donor_id'].iloc[0]

    cache = loading.get_visual_behavior_cache()
    behavior_session_table = cache.get_behavior_session_table().reset_index()
    filtered_ophys_experiment_table = loading.get_filtered_ophys_experiment_table()

    # get donor id for experiment_table
    filtered_ophys_experiment_table['donor_id'] = filtered_ophys_experiment_table['specimen_id'].map(
        lambda sid: get_donor_from_specimen_id(sid)
    )

    # get behavior donor dataframe - all mice in behavior table
    behavior_donors = pd.DataFrame({'donor_id':behavior_session_table['donor_id'].unique()})
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
    behavior_id_project_id_map = db.lims_query(query).rename(columns={'id':'behavior_session_id'}).merge(
        project_table,
        left_on='project_id',
        right_on='id',
        how='left',
    ).drop(columns=['id']).rename(columns={'code':'project_code'}).drop_duplicates('behavior_session_id').set_index('behavior_session_id')

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
    for idx,row in behavior_session_table.iterrows():
        if pd.isnull(row['session_type']):
            behavior_session_table.at[idx, 'session_type'] = get_session_type(row['ophys_session_id'])

    # merge in existing validation results
    validation_results = sdk_validation.get_validation_results().sort_index()
    behavior_session_table = behavior_session_table.merge(
        validation_results,
        left_index=True,
        right_index=True,
        how='left'
    )
    # find sessions with failures (by applying all function after setting null values to True)
    sessions_with_failures = validation_results[~validation_results.drop(columns=['timestamp','is_ophys']).fillna(0).apply(all,axis=1)]

    validate_only_failed = args.validate_only_failed

    if validate_only_failed == True:
        sessions_to_validate = sessions_with_failures.index.values
    else:
        # validate everything in the behavior session table
        sessions_to_validate = behavior_session_table.query('donor_in_ophys').index.values

    # THIS IS TEMPORARY: FIXING SOME MISSING SESSIONS
    missing_sessions = behavior_session_table[pd.isnull(behavior_session_table['metadata'])].index.values
    sessions_to_validate = missing_sessions

    print('missing sessions:')
    print(missing_sessions)

    for ii, behavior_session_id in enumerate(sessions_to_validate):

        print('starting cluster job for {}'.format(behavior_session_id))
        job_title = 'sdk_validation_bsid_{}'.format(behavior_session_id)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args="--behavior-session-id {} --validate-only-failed {}".format(behavior_session_id, validate_only_failed),
            jobname=job_title,
            jobdir=job_dir,
            **job_settings
        ).run(dryrun=False)
        time.sleep(0.001)


