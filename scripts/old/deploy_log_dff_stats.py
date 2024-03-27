import os
import argparse
import sys
import time
import pandas as pd
import visual_behavior.data_access.loading as loading
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402

parser = argparse.ArgumentParser(description='run sdk validation')
parser.add_argument('--env', type=str, default='vba', metavar='name of conda environment to use')

job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/job_records"

job_settings = {'queue': 'braintv',
                'mem': '2g',
                'walltime': '0:05:00',
                'ppn': 1,
                }

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/code/visual_behavior_analysis/scripts/log_dff_stats.py".format(os.path.expanduser('~'))
    print('python file = {}'.format(python_file))
    ophys_experiment_table = loading.get_filtered_ophys_experiment_table()

    for ii, ophys_experiment_id in enumerate(ophys_experiment_table.index):

        print('starting cluster job for {}'.format(ophys_experiment_id))
        job_title = '{}_log_dff_stats'.format(ophys_experiment_id)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args="--oeid {}".format(ophys_experiment_id),
            jobname=job_title,
            jobdir=job_dir,
            **job_settings
        ).run(dryrun=False)
        time.sleep(0.001)
