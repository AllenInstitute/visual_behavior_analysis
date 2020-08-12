import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np
from visual_behavior import database as db
import visual_behavior.data_access.loading as loading
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402

parser = argparse.ArgumentParser(description='run encoder smoothing check')
parser.add_argument('--env', type=str, default='visual_behavior', metavar='name of conda environment to use')

job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/encoder_check"

job_settings = {'queue': 'braintv',
                'mem': '128g',
                'walltime': '6:00:00',
                'ppn': 1,
                }

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/code/visual_behavior_analysis/scripts/spline_regression_check.py".format(os.path.expanduser('~'))

    cache = loading.get_visual_behavior_cache()
    behavior_session_table = cache.get_behavior_session_table()

    bsids = behavior_session_table.sample(250,random_state=0).index

    for bsid in bsids:
        print('starting cluster job for {}'.format(bsid))
        job_title = 'bsid_{}_encoder_smooth_check'.format(bsid)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args="--bsid {}".format(bsid),
            jobname=job_title,
            jobdir=job_dir,
            **job_settings
        ).run(dryrun=False)
        time.sleep(0.001)