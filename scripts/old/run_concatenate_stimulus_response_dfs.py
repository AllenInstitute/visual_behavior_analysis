import os
import sys
import platform
if platform.system() == 'Linux':
    # sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob # flake8: noqa: E999


import visual_behavior.data_access.loading as loading

python_file = r"/home/marinag/visual_behavior_analysis/scripts/concatenate_stimulus_response_dfs.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords'

job_settings = {'queue': 'braintv',
                'mem': '75g',
                'walltime': '10:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

experiments_table = loading.get_filtered_ophys_experiment_table()

for project_code in experiments_table.project_code.unique():
    for session_number in experiments_table.session_number.unique():
        for cre_line in experiments_table.cre_line.unique():
            PythonJob(
                python_file,
                python_executable='/home/marinag/anaconda2/envs/visual_behavior_sdk/bin/python',
                python_args=[project_code, cre_line, session_number],
                conda_env=None,
                jobname='stimulus_presentations_' + cre_line + '_' + project_code + '_' + str(session_number),
                **job_settings
            ).run(dryrun=False)
