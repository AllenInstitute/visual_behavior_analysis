import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np
import visual_behavior.validation.sdk as sdk_validation
import visual_behavior.visualization.qc.data_loading as dl
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402


parser = argparse.ArgumentParser(description='run sdk validation')
parser.add_argument('--env', type=str, default='visual_behavior', metavar='name of conda environment to use')

job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/sdk_validation"

job_settings = {'queue': 'braintv',
                'mem': '10g',
                'walltime': '0:10:00',
                'ppn': 1,
                }


if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = "{}/code/visual_behavior_analysis/visual_behavior/validation/sdk.py".format(os.path.expanduser('~'))

    cache = sdk_validation.get_cache()
    behavior_session_table = cache.get_behavior_session_table()
    filtered_ophys_session_table = dl.get_filtered_ophys_sessions_table()

    validation_results = sdk_validation.get_validation_results().sort_index()

    for ii, behavior_session_id in enumerate(behavior_session_table.index.values):
        condition_1 = behavior_session_id not in validation_results.index
        condition_2 = pd.notnull(behavior_session_table.loc[behavior_session_id]['ophys_session_id'])
        condition_3 = behavior_session_table.loc[behavior_session_id]['ophys_session_id'] in filtered_ophys_session_table['ophys_session_id'].tolist()

        if condition_1:# and condition_2 and condition_3:
            print('starting cluster job for {}'.format(behavior_session_id))
            job_title = 'sdk_validation_bsid_{}'.format(behavior_session_id)
            pbstools.PythonJob(
                python_file,
                python_executable,
                python_args="--behavior-session-id {}".format(behavior_session_id),
                jobname=job_title,
                jobdir=job_dir,
                **job_settings
            ).run(dryrun=False)
            time.sleep(0.001)

