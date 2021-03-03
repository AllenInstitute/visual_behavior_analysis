import os
import sys
import platform
if platform.system() == 'Linux':
    # sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob # flake8: noqa: E999


import visual_behavior.data_access.loading as loading

python_file = r"/home/marinag/visual_behavior_analysis/scripts/create_multi_session_df.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords'

job_settings = {'queue': 'braintv',
                'mem': '100g',
                'walltime': '20:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)

for project_code in experiments_table.project_code.unique():
    for session_number in experiments_table.session_number.unique():

        PythonJob(
            python_file,
            python_executable='/home/marinag/anaconda2/envs/visual_behavior_sdk/bin/python',
            python_args=[project_code, session_number],
            conda_env=None,
            jobname='multi_session_df_'+project_code+'_'+str(session_number),
            **job_settings
        ).run(dryrun=False)
