import os
import sys
import platform
import pandas as pd
if platform.system() == 'Linux':
    # sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob # flake8: noqa: E999


import visual_behavior.data_access.loading as loading
filepath = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/decrosstalk_validation/20210125_decrosstalk_runthrough.csv"
dev_sessions = pd.read_csv(filepath)
experiment_ids = dev_sessions.oe_id.values

python_file = r"/home/marinag/visual_behavior_analysis/scripts/create_files_for_decrosstalk_testing.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords'

job_settings = {'queue': 'braintv',
                'mem': '100g',
                'walltime': '10:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/marinag/anaconda2/envs/visual_behavior_sdk/bin/python',
        python_args=experiment_id,
        conda_env=None,
        jobname='process_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)
