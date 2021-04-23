import os
import sys
import platform
if platform.system() == 'Linux':
    # sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob # flake8: noqa: E999

import visual_behavior.data_access.loading as loading

ophys_experiment_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
ophys_experiment_ids = ophys_experiment_table.index.values

python_file = r"/home/marinag/visual_behavior_analysis/scripts/create_cell_metrics_table.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords'

job_settings = {'queue': 'braintv',
                'mem': '60g',
                'walltime': '8:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for ophys_experiment_id in ophys_experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/marinag/anaconda2/envs/visual_behavior_sdk/bin/python',
        python_args=ophys_experiment_id,
        conda_env=None,
        jobname='process_{}'.format(ophys_experiment_id),
        **job_settings
    ).run(dryrun=False)
