import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np
import visual_behavior.validation.sdk as sdk_validation
import visual_behavior.data_access.loading as loading
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402
import visual_behavior.database as db

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
                'mem': '10g',
                'walltime': '0:10:00',
                'ppn': 1,
                }

def get_donor_from_specimen_id(specimen_id):
    res = db.lims_query('select * from specimens where id = {}'.format(specimen_id))
    if len(res['donor_id']) == 1:
        return res['donor_id'].iloc[0]
    elif len(res['donor_id']) == 0:
        return None
    elif len(res['donor_id']) > 1:
        print('found more than one donor ID for specimen ID {}'.format(specimen_id))
        return res['donor_id'].iloc[0]

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/code/visual_behavior_analysis/visual_behavior/validation/sdk.py".format(os.path.expanduser('~'))

    cache = loading.get_visual_behavior_cache()
    behavior_session_table = cache.get_behavior_session_table()
    filtered_ophys_experiment_table = loading.get_filtered_ophys_experiment_table()

    # get donor id for experiment_table
    filtered_ophys_experiment_table['donor_id'] = filtered_ophys_experiment_table['specimen_id'].map(
        lambda sid: get_donor_from_specimen_id(sid)
    )

    # get behavior donor dataframe - all mice in behavior table
    behavior_donors = pd.DataFrame({'donor_id':behavior_session_table['donor_id'].unique()})
    # add a flag identifying which donors have associated ophys sessions
    behavior_donors['has_ophys'] = behavior_donors['donor_id'].map(
        lambda did: did in list(filtered_ophys_experiment_table['donor_id'].unique())
    )

    # merge back in behavior donors to determine which behavior sessions have associated ophys
    behavior_session_table = behavior_session_table.merge(
        behavior_donors,
        left_on='donor_id',
        right_on='donor_id',
        how='left',
    )

    # get behavior session IDs from foraging IDs
    fid_list = behavior_session_table['foraging_id'].to_list()
    fid_tuple = tuple(f for f in fid_list if f is not None)

    # merge foraging IDS into behavior session_table
    foraging_id_df = db.lims_query("select id from behavior_sessions where foraging_id in {}".format(fid_tuple)).rename(columns={'id':'behavior_session_id'})
    foraging_id_df['foraging_id'] = list(fid_tuple)
    behavior_session_table = behavior_session_table.merge(
        foraging_id_df,
        left_on='foraging_id',
        right_on='foraging_id'
    )

    validation_results = sdk_validation.get_validation_results().sort_index()
    # find sessions with failures (by applying all function after setting null values to True)
    sessions_with_failures = validation_results[~validation_results.drop(columns=['timestamp','is_ophys']).fillna(1).apply(all,axis=1)]

    validate_only_failed = args.validate_only_failed

    if validate_only_failed == True:
        sessions_to_validate = sessions_with_failures.index.values
    else:
        # validate everything in the behavior session table
        sessions_to_validate = behavior_session_table.query('has_ophys')['behavior_session_id'].values

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


