import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np
import visual_behavior.validation.sdk as sdk_validation
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
    behavior_session_table = sdk_validation.get_behavior_session_table()
    sessions_with_failures = behavior_session_table[~behavior_session_table[expected_attributes].fillna(1).apply(all,axis=1)]

    validate_only_failed = args.validate_only_failed

    if validate_only_failed == True:
        sessions_to_validate = sessions_with_failures.index.values
    else:
        # validate everything in the behavior session table
        sessions_to_validate = behavior_session_table.index.values

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


